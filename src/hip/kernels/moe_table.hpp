// moe_table.hpp (HIP) — a whole MoE layer from one launch, with the expert
// matrices chosen on the DEVICE from a pointer table.
//
// The path this replaces, per layer: the router probabilities come back to the
// host, the host scans them for the top-k, and for each selected expert it calls
// acquire() to learn three device pointers before issuing three GEMMs. That is
// k host round trips of dependency per layer, and the measurement said the
// dependency rather than the copy is the cost: on Mellum2.1 `moe.route.fetch`
// is 0.523 ms/layer against 0.077 ms for a readback whose queue is already
// retired.
//
// Here the ids never leave the device. Each block owns ONE token and walks its
// k slots; each slot's gate/up/down come from `table[layer * n_expert + id]`.
//
// Arithmetic, chosen to match the per-expert path so the only difference left
// is the device softmax:
//   * one warp per output row, lane l summing chunks l, l+32, ... and one
//     wave reduction at the end -- the same `dot_chunks` call in the same order
//     as gemv_kernel, so a projection is bit-identical to the launch it
//     replaces;
//   * the gate and up results are rounded to the activation type before the
//     activation, because that is what the two-GEMM path stores before
//     silu_mul reads them;
//   * the routed sum rounds each expert's weighted contribution into the f16
//     accumulator (the scatter does), rather than accumulating f32 and rounding
//     once.
// The slot order is token-then-slot, not expert-ascending, so a token that
// selects two experts sums in a different order than the host loop does; that
// is the one deliberate difference and it is in the report.
#pragma once

#include "krk/moe_table.hpp"
#include "dequant.hpp"
#include "gemm.hpp"

namespace krk {

// One row of a quantized matrix against an f16 activation row, with gemv_kernel's
// lane split and reduction -- the primitive both paths share.
__device__ __forceinline__ f32 table_row_dot(const void *w, int wt, i64 row,
                                             i64 row_bytes, const _Float16 *x,
                                             i64 n_in) {
    if (w == nullptr || row_bytes <= 0 || n_in <= 0) return 0.0f;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const u8 *wrow = static_cast<const u8 *>(w) +
                     static_cast<size_t>(row) * static_cast<size_t>(row_bytes);
    const int nch = static_cast<int>(n_in / 32);
    f32 part = 0.0f;
    for (int c = lane; c < nch; c += 32)
        part += dot_chunks(wt, wrow, x, c, c + 1);
    return d_wave_reduce_sum(part);
}

// The shared-memory layout: the expert's activation row (f16), this token's
// accumulator (f32), then the row-max reduction and the row scale it produced.
// The last two are why a row that leaves f16 is SCALED rather than clipped --
// the host's silu_mul does the same, and it is the one quantity on this path
// that can leave the range its operands live in (silu(x) ~ x for large x, so
// silu(gate)*up is ~x*x; measured 785411 against f16's 65504 on
// Laguna-XS-2.1-IQ3_XXS L38 e161).
struct TableSmem {
    _Float16 *mid;
    f32 *acc;
    f32 *red;  // [kWarps] partial maxima, then the row scale at [kWarps]
};

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    moe_table_kernel(const MoeDispatchEntry *__restrict__ table, i32 layer,
                     i32 n_expert, const i32 *__restrict__ ids,
                     const f32 *__restrict__ wts, const _Float16 *__restrict__ xn,
                     _Float16 *__restrict__ out, i32 n_embd, i32 n_ff, i32 k) {
    // Dynamic shared memory: the expert's activation row, then this token's
    // accumulator. Both are bounded by the model's geometry, which the caller
    // sizes the launch with.
    // The dynamic shared block is declared `u8` because every `extern
    // __shared__` in one translation unit must agree on the element type;
    // the two views below are the partition.
    extern __shared__ u8 moe_smem[];
    _Float16 *mid = reinterpret_cast<_Float16 *>(moe_smem);
    f32 *acc = reinterpret_cast<f32 *>(mid + n_ff);
    f32 *red = acc + n_embd;

    constexpr int kWarps = THREADS / 32;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const i64 t = static_cast<i64>(blockIdx.x);
    const _Float16 *x = xn + t * n_embd;

    for (i32 j = tid; j < n_embd; j += THREADS) acc[j] = 0.0f;
    __syncthreads();

    for (i32 s = 0; s < k; s++) {
        const i32 id = ids[t * k + s];
        if (id < 0 || id >= n_expert) continue;
        const MoeDispatchEntry &en =
            table[static_cast<i64>(layer) * n_expert + id];
        if (!en.present[kMoeGate] || !en.present[kMoeUp] || !en.present[kMoeDown])
            continue;
        const f32 alpha = wts[t * k + s];

        // gate/up: the pair one activation produces, in the pair's own order
        // (gate first, then up) and rounded to f16 before the activation.
        f32 mx = 0.0f;
        for (i32 j = warp; j < n_ff; j += kWarps) {
            const f32 g = table_row_dot(en.p[kMoeGate], en.type[kMoeGate], j,
                                        en.row_bytes[kMoeGate], x, n_embd);
            const f32 u = table_row_dot(en.p[kMoeUp], en.type[kMoeUp], j,
                                        en.row_bytes[kMoeUp], x, n_embd);
            const f32 gf = static_cast<f32>(static_cast<_Float16>(g));
            const f32 uf = static_cast<f32>(static_cast<_Float16>(u));
            const f32 p = d_silu(gf) * uf;
            mid[j] = static_cast<_Float16>(p);
            const f32 a = fabsf(p);
            if (a > mx) mx = a;
        }
        // The row scale, exactly as silu_mul_scaled_kernel derives it: smallest
        // power of two that puts the row's maximum under f16's ceiling. 1.0 for
        // every row already in range, and then the store above is the final one.
        mx = d_wave_reduce_max(mx);
        if (lane == 0) red[warp] = mx;
        __syncthreads();
        if (tid == 0) {
            f32 m = red[0];
            for (int w = 1; w < kWarps; w++) m = fmaxf(m, red[w]);
            f32 mul = 1.0f;
            while (m > 65504.0f * mul && mul < 16777216.0f) mul *= 2.0f;
            red[kWarps] = mul;
        }
        __syncthreads();
        const f32 row_scale = red[kWarps];
        if (row_scale > 1.0f) {
            // Out of range: the f16 stores above are clipped, so the row is
            // recomputed at the scale rather than un-clipped. Rare by
            // construction -- the host pays the same second pass implicitly,
            // because its scaled kernel never stores the unscaled product.
            for (i32 j = warp; j < n_ff; j += kWarps) {
                const f32 g = table_row_dot(en.p[kMoeGate], en.type[kMoeGate], j,
                                            en.row_bytes[kMoeGate], x, n_embd);
                const f32 u = table_row_dot(en.p[kMoeUp], en.type[kMoeUp], j,
                                            en.row_bytes[kMoeUp], x, n_embd);
                const f32 gf = static_cast<f32>(static_cast<_Float16>(g));
                const f32 uf = static_cast<f32>(static_cast<_Float16>(u));
                mid[j] = static_cast<_Float16>(d_silu(gf) * uf / row_scale);
            }
        }
        __syncthreads();

        // down: rounded contribution into the routed sum, as the scatter does
        // (the scatter is also what folds the row scale back in).
        //
        // The three multiplies are EXPLICIT round-to-nearest ops, not an
        // expression. Written as `acc + alpha * v * scale` the compiler is free
        // to contract one of them into an FMA, which has a single rounding where
        // the host path has two -- measured as exactly 1 f16 ulp on 15% of
        // elements (1443 against 1444 at an f16 ulp of 1), i.e. a difference
        // that reads as correct-but-not-identical and hides any real one.
        //
        // `v` is rounded to the activation type FIRST, because that is what the
        // path it replaces stores: the down GEMM writes f16 to ws_x2_ and the
        // scatter reads it back. Keeping the f32 dot left the sum a fraction of
        // an ulp away from the per-expert path on multi-slot tokens -- measured
        // as exactly 1 f16 ulp (1443 against 1444) on 14% of elements, which is
        // the difference between "same arithmetic" and "nearly".
        for (i32 j = warp; j < n_embd; j += kWarps) {
            const f32 v = static_cast<f32>(static_cast<_Float16>(table_row_dot(
                en.p[kMoeDown], en.type[kMoeDown], j, en.row_bytes[kMoeDown],
                mid, n_ff)));
            const f32 vs = __fmul_rn(v, row_scale);
            const f32 w = __fmul_rn(alpha, vs);
            acc[j] = static_cast<f32>(
                static_cast<_Float16>(__fadd_rn(acc[j], w)));
        }
        __syncthreads();
    }

    for (i32 j = tid; j < n_embd; j += THREADS)
        out[t * n_embd + j] = static_cast<_Float16>(acc[j]);
}

// ---- tiled, many blocks per expert ----------------------------------------
//
// moe_table_kernel launches one block per TOKEN and walks that token's k slots
// inside it. That is the right shape for a prefill row set and the wrong one for
// a single row: at n = 1 a 64-CU card runs one block, one CU, for the whole
// layer -- measured on Qwen3.5-35B-A3B as 2.51 ms/layer of device time and 154.9
// ms/step against 127.5 for the per-expert path the arm replaces.
//
// Giving each (token, slot) its own block got the count to k -- eight blocks on
// a card with sixty-four CUs -- and measured 58.2 ms/step for the layer, still
// behind the per-expert path, because a block that walks an expert's rows in
// order issues one row's loads at a time.
//
// These three kernels tile the OUTPUT ROWS as well, so one expert's work spans
// ceil(n_ff / kGuRows) blocks for gate/up and ceil(n_embd / kDownRows) blocks for
// the down projection: on the same model that is 256 + 1024 blocks for a decode
// step instead of 8. Two quantities have to survive the split, and they are the
// whole design:
//
//   * the activation row, which every down block needs in full. It goes to
//     GLOBAL scratch as f32 -- the exact pre-rounding product, so the f16 the
//     down dot reads is f16(silu(g)*u) and not a value rounded twice. Keeping
//     it f16 instead would have needed the row scale before the row exists.
//   * the row scale, which is a max over the whole row. Each gate/up tile
//     writes the max it saw; one tiny pass per (token, slot) folds them, derives
//     the same power-of-two scale the serial kernel derives, and stores
//     f16(p / scale) -- which for scale 1 is exactly the byte the serial form
//     stores, and for scale > 1 is exactly the second pass the serial form runs
//     (the f32 scratch is never clipped, so no recomputation from weights is
//     needed).
//
// The arithmetic is therefore element-for-element the single-block form's: same
// `table_row_dot`, same round-to-f16 before the activation, same three explicit
// round-to-nearest multiplies into the partial, and the reduce still replays the
// serial accumulation order. Only the block count changes.
//
// A skipped slot (bad id, missing matrix) writes zeros: the serial loop skips
// the step without rounding, and `f16(a + 0) == a` because `a` is already an
// f16 value, so zeroing is the same sum.
template <int THREADS, int ROWS>
__global__ void __launch_bounds__(THREADS)
    moe_table_gu_tiled(const MoeDispatchEntry *__restrict__ table, i32 layer,
                       i32 n_expert, const i32 *__restrict__ ids,
                       const _Float16 *__restrict__ xn, f32 *__restrict__ mid,
                       i32 n_embd, i32 n_ff, i32 k, i32 tiles) {
    constexpr int kWarps = THREADS / 32;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const i64 linear = static_cast<i64>(blockIdx.x);
    const i64 ts = linear / tiles;
    const i32 tile = static_cast<i32>(linear - ts * tiles);
    const i64 t = ts / k;
    const i32 s = static_cast<i32>(ts - t * k);

    const i32 row0 = tile * ROWS;
    if (row0 >= n_ff) return;
    const i32 row_end = row0 + ROWS < n_ff ? row0 + ROWS : static_cast<i32>(n_ff);
    // A skipped slot writes nothing: the down pass zeroes its contribution.
    const i32 id = ids[t * k + s];
    if (id < 0 || id >= n_expert) return;
    const MoeDispatchEntry &en = table[static_cast<i64>(layer) * n_expert + id];
    if (!en.present[kMoeGate] || !en.present[kMoeUp]) return;

    const _Float16 *x = xn + t * n_embd;
    f32 *mrow = mid + ts * static_cast<i64>(n_ff);
    for (i32 j = row0 + warp; j < row_end; j += kWarps) {
        const f32 g = table_row_dot(en.p[kMoeGate], en.type[kMoeGate], j,
                                    en.row_bytes[kMoeGate], x, n_embd);
        const f32 u = table_row_dot(en.p[kMoeUp], en.type[kMoeUp], j,
                                    en.row_bytes[kMoeUp], x, n_embd);
        const f32 gf = static_cast<f32>(static_cast<_Float16>(g));
        const f32 uf = static_cast<f32>(static_cast<_Float16>(u));
        mrow[j] = d_silu(gf) * uf;
    }
}

// The row scale and the f16 activation row the down dot consumes: one block per
// (token, slot). The max is taken from the f32 row itself rather than from a
// per-tile summary, which is the same value (a max does not care about order)
// and one buffer smaller -- and it is what lets the gate/up pass be split by
// EXPERT as well, where a per-tile max has no single (token, slot) to report to.
// `1 / row_scale` is exact because the scale is a power of two, so the store is
// `f16(p / row_scale)`: the serial kernel's first pass when the scale is 1, its
// second pass when it is not.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    moe_table_scale_kernel(const MoeDispatchEntry *__restrict__ table, i32 layer,
                           i32 n_expert, const i32 *__restrict__ ids,
                           const f32 *__restrict__ mid,
                           _Float16 *__restrict__ midf, f32 *__restrict__ rscale,
                           i32 n_ff, i32 k) {
    constexpr int kWarps = THREADS / 32;
    const i64 ts = static_cast<i64>(blockIdx.x);
    const i64 t = ts / k;
    const i32 s = static_cast<i32>(ts - t * k);
    const i32 id = ids[t * k + s];
    const bool live = id >= 0 && id < n_expert &&
                      table[static_cast<i64>(layer) * n_expert + id].present[kMoeGate] &&
                      table[static_cast<i64>(layer) * n_expert + id].present[kMoeUp];
    const f32 *mrow = mid + ts * static_cast<i64>(n_ff);
    _Float16 *frow = midf + ts * static_cast<i64>(n_ff);

    f32 mx = 0.0f;
    if (live)
        for (i32 j = static_cast<int>(threadIdx.x); j < n_ff; j += THREADS) {
            const f32 a = fabsf(mrow[j]);
            if (a > mx) mx = a;
        }
    mx = d_wave_reduce_max(mx);
    __shared__ f32 red[kWarps];
    __shared__ f32 sscale;
    if ((static_cast<int>(threadIdx.x) & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        f32 m = red[0];
        for (int w = 1; w < kWarps; w++) m = fmaxf(m, red[w]);
        f32 mul = 1.0f;
        while (m > 65504.0f * mul && mul < 16777216.0f) mul *= 2.0f;
        sscale = mul;
        rscale[ts] = mul;
    }
    __syncthreads();
    const f32 inv = 1.0f / sscale;
    for (i32 j = static_cast<int>(threadIdx.x); j < n_ff; j += THREADS)
        frow[j] = static_cast<_Float16>(__fmul_rn(mrow[j], inv));
}

// The down projection for a row range: the contribution the serial loop adds for
// this slot, written to the per-slot partial the reduce replays.
template <int THREADS, int ROWS>
__global__ void __launch_bounds__(THREADS)
    moe_table_down_tiled(const MoeDispatchEntry *__restrict__ table, i32 layer,
                         i32 n_expert, const i32 *__restrict__ ids,
                         const f32 *__restrict__ wts,
                         const _Float16 *__restrict__ midf,
                         const f32 *__restrict__ rscale, f32 *__restrict__ part,
                         i32 n_embd, i32 n_ff, i32 k, i32 tiles) {
    constexpr int kWarps = THREADS / 32;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const i64 linear = static_cast<i64>(blockIdx.x);
    const i64 ts = linear / tiles;
    const i32 tile = static_cast<i32>(linear - ts * tiles);
    const i32 row0 = tile * ROWS;
    f32 *acc = part + ts * static_cast<i64>(n_embd);
    if (row0 >= n_embd) return;
    const i32 row_end = row0 + ROWS < n_embd ? row0 + ROWS : static_cast<i32>(n_embd);

    auto zero = [&]() {
        for (i32 j = row0 + warp; j < row_end; j += kWarps) acc[j] = 0.0f;
    };
    const i64 t = ts / k;
    const i32 s = static_cast<i32>(ts - t * k);
    const i32 id = ids[t * k + s];
    if (id < 0 || id >= n_expert) {
        zero();
        return;
    }
    const MoeDispatchEntry &en = table[static_cast<i64>(layer) * n_expert + id];
    if (!en.present[kMoeGate] || !en.present[kMoeUp] || !en.present[kMoeDown]) {
        zero();
        return;
    }
    const f32 row_scale = rscale[ts];
    const f32 alpha = wts[ts];
    const _Float16 *mrow = midf + ts * static_cast<i64>(n_ff);
    for (i32 j = row0 + warp; j < row_end; j += kWarps) {
        const f32 v = static_cast<f32>(static_cast<_Float16>(table_row_dot(
            en.p[kMoeDown], en.type[kMoeDown], j, en.row_bytes[kMoeDown], mrow,
            n_ff)));
        const f32 vs = __fmul_rn(v, row_scale);
        acc[j] = __fmul_rn(alpha, vs);
    }
}

// Four activation rows against one weight row, from one pass over its bytes.
//
// moe_table_gu_grouped called table_row_dot once per token in the group, which
// is a serial chain of n_in/32 dependent chunk loads per token with nothing to
// overlap them: the group loop has runtime bounds, so no unrolling happens and
// the second token's loads wait for the first token's FMAs. This interleaves four
// tokens over the SAME chunk -- the bytes are shared, the accumulators are
// independent, and the loop body is wide enough for the scheduler to hide the
// latency. The chunk ORDER per token is unchanged (c = lane, lane + 32, ...), so
// every per-token sum is the same value it was.
template <int G>
__device__ __forceinline__ void table_row_dot_g(const void *w, int wt, i64 row,
                                               i64 row_bytes,
                                               const _Float16 *const *xs, i64 n_in,
                                               f32 *out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const u8 *wrow = static_cast<const u8 *>(w) +
                     static_cast<size_t>(row) * static_cast<size_t>(row_bytes);
    const int nch = static_cast<int>(n_in / 32);
    f32 part[G];
#pragma unroll
    for (int q = 0; q < G; q++) part[q] = 0.0f;
    for (int c = lane; c < nch; c += 32) {
#pragma unroll
        for (int q = 0; q < G; q++)
            part[q] += dot_chunks(wt, wrow, xs[q], c, c + 1);
    }
#pragma unroll
    for (int q = 0; q < G; q++) out[q] = d_wave_reduce_sum(part[q]);
}

// One (row, group) step of the gate/up phase, for a batch of G <= 4 tokens.
// G = 1 is the scalar path, so the three batch sizes share one body and the
// group's tail costs a scalar pass rather than a second kernel.
template <int G>
__device__ __forceinline__ void table_gu_batch(const MoeDispatchEntry &en, i32 j,
                                               const i32 *__restrict__ idx, i32 at,
                                               const _Float16 *__restrict__ xn,
                                               f32 *__restrict__ mid, i32 n_embd,
                                               i32 n_ff, i32 k) {
    const _Float16 *xs[G];
    i32 ts[G];
#pragma unroll
    for (int q = 0; q < G; q++) {
        ts[q] = idx[at + q];
        xs[q] = xn + static_cast<i64>(ts[q] / k) * n_embd;
    }
    f32 gv[G], uv[G];
    table_row_dot_g<G>(en.p[kMoeGate], en.type[kMoeGate], j,
                       en.row_bytes[kMoeGate], xs, n_embd, gv);
    table_row_dot_g<G>(en.p[kMoeUp], en.type[kMoeUp], j, en.row_bytes[kMoeUp],
                       xs, n_embd, uv);
#pragma unroll
    for (int q = 0; q < G; q++) {
        const f32 gf = static_cast<f32>(static_cast<_Float16>(gv[q]));
        const f32 uf = static_cast<f32>(static_cast<_Float16>(uv[q]));
        mid[static_cast<i64>(ts[q]) * n_ff + j] = d_silu(gf) * uf;
    }
}

// The same for the down phase, including the two round-to-nearest multiplies
// that fold the row scale and the routing weight back in.
template <int G>
__device__ __forceinline__ void table_down_batch(const MoeDispatchEntry &en, i32 j,
                                                 const i32 *__restrict__ idx, i32 at,
                                                 const _Float16 *__restrict__ midf,
                                                 const f32 *__restrict__ rscale,
                                                 const f32 *__restrict__ wts,
                                                 f32 *__restrict__ part, i32 n_embd,
                                                 i32 n_ff) {
    const _Float16 *xs[G];
    i32 ts[G];
#pragma unroll
    for (int q = 0; q < G; q++) {
        ts[q] = idx[at + q];
        xs[q] = midf + static_cast<i64>(ts[q]) * n_ff;
    }
    f32 v[G];
    table_row_dot_g<G>(en.p[kMoeDown], en.type[kMoeDown], j,
                       en.row_bytes[kMoeDown], xs, n_ff, v);
#pragma unroll
    for (int q = 0; q < G; q++) {
        const f32 vf = static_cast<f32>(static_cast<_Float16>(v[q]));
        const f32 vs = __fmul_rn(vf, rscale[ts[q]]);
        part[static_cast<i64>(ts[q]) * n_embd + j] = __fmul_rn(wts[ts[q]], vs);
    }
}

// ---- expert-major, for a BATCH -------------------------------------------
//
// The two kernels above key their grid by (token, slot), which is right for a
// decode step -- one token, k experts, nothing to share -- and wrong for a chunk.
// A chunk routes 191 tokens x 8 slots through 256 experts, so those pairs name
// at most 256 distinct matrices while the (token, slot) form reads one per PAIR:
// measured on Qwen3.5-35B-A3B, 3.5 s of prefill experts where the per-expert
// path spends 150 ms, and row tiling over the pairs changed nothing (54.3 vs
// 54.3 tok/s) because the duplicated READ, not the block count, is the cost.
//
// These key the grid by EXPERT instead: a block owns an expert and a row range,
// and walks the tokens that routed to it, so one weight row is read once and
// dotted against every token that wants it. The row is about a kilobyte, so the
// second token onward is served from L1 rather than from the tier.
//
// The group comes from `moe_group_plan_kernel` (`off`, `idx`), which emits the
// same token-then-slot order the host plan does. A block is thus (expert, row
// range) and the arithmetic per (row, token) pair is the per-pair form's exactly
// -- same dot, same f16 round before the activation -- so the two orderings
// produce the same bytes, not merely close ones.
template <int THREADS, int ROWS>
__global__ void __launch_bounds__(THREADS)
    moe_table_gu_grouped(const MoeDispatchEntry *__restrict__ table, i32 layer,
                         i32 n_expert, const i32 *__restrict__ off,
                         const i32 *__restrict__ idx,
                         const _Float16 *__restrict__ xn, f32 *__restrict__ mid,
                         i32 n_embd, i32 n_ff, i32 k) {
    constexpr int kWarps = THREADS / 32;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const i32 e = static_cast<i32>(blockIdx.y);
    const i32 beg = off[e];
    const i32 end = off[e + 1];
    if (beg >= end) return;
    const MoeDispatchEntry &en = table[static_cast<i64>(layer) * n_expert + e];
    if (!en.present[kMoeGate] || !en.present[kMoeUp]) return;
    const i32 row0 = static_cast<i32>(blockIdx.x) * ROWS;
    if (row0 >= n_ff) return;
    const i32 row_end = row0 + ROWS < n_ff ? row0 + ROWS : static_cast<i32>(n_ff);

    // Up to four tokens per pass over a weight row (see table_row_dot_g); 2 and
    // then 1 cover the group's tail, so no token drops to the serial path just
    // because the group is not a multiple of four.
    for (i32 j = row0 + warp; j < row_end; j += kWarps) {
        i32 g = beg;
        for (; g + 4 <= end; g += 4)
            table_gu_batch<4>(en, j, idx, g, xn, mid, n_embd, n_ff, k);
        for (; g + 2 <= end; g += 2)
            table_gu_batch<2>(en, j, idx, g, xn, mid, n_embd, n_ff, k);
        for (; g < end; g++)
            table_gu_batch<1>(en, j, idx, g, xn, mid, n_embd, n_ff, k);
    }
}

// The down projection, expert-major for the same reason: one weight row per row
// range, dotted against every token in the group.
template <int THREADS, int ROWS>
__global__ void __launch_bounds__(THREADS)
    moe_table_down_grouped(const MoeDispatchEntry *__restrict__ table, i32 layer,
                           i32 n_expert, const i32 *__restrict__ off,
                           const i32 *__restrict__ idx, const f32 *__restrict__ wts,
                           const _Float16 *__restrict__ midf,
                           const f32 *__restrict__ rscale, f32 *__restrict__ part,
                           i32 n_embd, i32 n_ff, i32 k) {
    constexpr int kWarps = THREADS / 32;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const i32 e = static_cast<i32>(blockIdx.y);
    const i32 beg = off[e];
    const i32 end = off[e + 1];
    if (beg >= end) return;
    const i32 row0 = static_cast<i32>(blockIdx.x) * ROWS;
    if (row0 >= n_embd) return;
    const i32 row_end = row0 + ROWS < n_embd ? row0 + ROWS : static_cast<i32>(n_embd);
    const MoeDispatchEntry &en = table[static_cast<i64>(layer) * n_expert + e];
    // A slot whose expert is incomplete contributes zeros, exactly as the serial
    // loop's skip does (`f16(a + 0) == a`).
    if (!en.present[kMoeGate] || !en.present[kMoeUp] || !en.present[kMoeDown]) {
        for (i32 g = beg; g < end; g++) {
            f32 *acc = part + static_cast<i64>(idx[g]) * n_embd;
            for (i32 j = row0 + warp; j < row_end; j += kWarps) acc[j] = 0.0f;
        }
        return;
    }
    for (i32 j = row0 + warp; j < row_end; j += kWarps) {
        i32 g = beg;
        for (; g + 4 <= end; g += 4)
            table_down_batch<4>(en, j, idx, g, midf, rscale, wts, part, n_embd, n_ff);
        for (; g + 2 <= end; g += 2)
            table_down_batch<2>(en, j, idx, g, midf, rscale, wts, part, n_embd, n_ff);
        for (; g < end; g++)
            table_down_batch<1>(en, j, idx, g, midf, rscale, wts, part, n_embd, n_ff);
    }
}

// The serial loop's accumulation, in slot order, with the same f16 rounding
// after every add. Sequential on purpose: reordering this sum is how a "faster"
// reduction stops matching the path the arm is measured against.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    moe_table_reduce_kernel(const f32 *__restrict__ part, _Float16 *__restrict__ out,
                            i32 n_embd, i32 k) {
    const i64 t = static_cast<i64>(blockIdx.x);
    const f32 *p = part + t * static_cast<i64>(k) * n_embd;
    for (i32 j = static_cast<int>(threadIdx.x); j < n_embd; j += THREADS) {
        f32 a = 0.0f;
        for (i32 s = 0; s < k; s++)
            a = static_cast<f32>(static_cast<_Float16>(
                __fadd_rn(a, p[static_cast<i64>(s) * n_embd + j])));
        out[t * n_embd + j] = static_cast<_Float16>(a);
    }
}

}  // namespace krk
