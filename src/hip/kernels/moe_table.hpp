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

}  // namespace krk
