// ============================================================================
//  fused.hpp — fused decode-layer GEMM groups (prototype).
//
//  A decode layer's Q4_K projections that share one activation source
//  and have no dependency between them — q/k/v (all read the
//  attention-norm output) and gate/up (both read the FFN-norm output)
//  — can stream from one kernel launch instead of separate launches.
//  o and down cannot join a group: their inputs (the attention output,
//  the silu result) are produced mid-layer, so the engine fuses one
//  group per launch. The norms, rope, kv_append, attention and
//  silu_mul between the projections stay as their own launches — they
//  have real data dependencies a single non-cooperative kernel cannot
//  express.
//
//  Parallelism per matrix, in priority order:
//    1. row-parallel blocks — one warp per output row, eight rows per
//       256-thread block, exactly the gemv_kernel scheme; and
//    2. split-K top-up — a matrix whose row-block count cannot fill the
//       device (k/v: 192 rows = 24 blocks on 72 CUs) is cut along K so
//       its weight loads overlap across more blocks. Slices follow the
//       §2d rule (round32 length, >= 64 channels, last slice never
//       empty) and their fp32 partials land in the same persistent
//       split-K scratch gemm_launch uses, reduced by gemm_reduce_kernel.
//
//  Within a slice the accumulation order matches gemv_kernel exactly
//  (lane-strided chunks, wave shuffle reduce), so a split==1 matrix is
//  bit-identical to its separate gemv launch; a split matrix differs only
//  in the fp32 regrouping order, which the all-ones probe data makes
//  bit-exact too (every partial is an exact integer).
// ============================================================================
#ifndef KRK_KERNEL_FUSED_HPP
#define KRK_KERNEL_FUSED_HPP

#include "dequant.hpp"
#include "gemm.hpp"

namespace krk {

// The projections of one decode-layer GEMM group, in layer order.
// `count` matrices at most (the arrays are sized 8). `in` is the
// per-matrix activation source: xn for q/k/v/gate/up, the attention
// output for o, the silu(gate*up) result for down. `part`/`part_off`
// address the persistent split-K partials scratch (§2d); matrices that
// run split==1 never touch it. A group is the set of projections that
// share one activation source and have no dependency between them —
// q/k/v in the attention phase, gate/up in the FFN phase — so the
// engine fuses a group per launch, not the whole layer at once.
struct FusedLayer {
    int count;
    // One weight type per matrix. A checkpoint mixes formats across the
    // projections of a group routinely (Qwen3.5-0.8B: attn_qkv Q5_K, ssm_alpha
    // and ssm_beta Q4_K), and the weight addressing was always per matrix --
    // `w_row_bytes[m]` -- so the dequant dispatch tag was the only thing that
    // had to be shared to keep a mixed group from being expressible.
    int wt[8];
    const u8 *w[8];
    const _Float16 *in[8];
    _Float16 *out[8];
    i64 n_out[8];
    i64 n_in[8];
    size_t w_row_bytes[8];
    int splits[8];
    f32 *part;
    i64 part_off[8];
};

// All seven projections from one launch. blockIdx.x names a
// (matrix, K-slice, row-block) triple via a prefix scan over the
// per-matrix block budgets, so the grid size is the sum of the seven
// budgets — one dispatch for the whole GEMM phase.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    fused_layer_gemv_kernel(const FusedLayer fl) {
    const int warp_rows = THREADS / 32;
    int base[9];
    base[0] = 0;
#pragma unroll
    for (int m = 0; m < fl.count; m++) {
        const int rb = static_cast<int>(
            (fl.n_out[m] + warp_rows - 1) / warp_rows);
        base[m + 1] = base[m] + rb * fl.splits[m];
    }
    int m = 0;
    for (int mm = 0; mm < fl.count; mm++)
        if (blockIdx.x < base[mm + 1]) {
            m = mm;
            break;
        }
    const int row_blocks =
        static_cast<int>((fl.n_out[m] + warp_rows - 1) / warp_rows);
    const int rel = static_cast<int>(blockIdx.x) - base[m];
    const int kslice = rel / row_blocks;
    const int rblk = rel % row_blocks;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const i64 row =
        static_cast<i64>(rblk) * warp_rows + (threadIdx.x >> 5);
    if (row >= fl.n_out[m]) return;

    // K slice: every slice but the last is exactly `slice` long (a
    // multiple of 32 so chunk addressing stays integral); the last
    // runs to n_in. Empty slices are valid and contribute zero.
    const int sp = fl.splits[m];
    i64 slice = (fl.n_in[m] + sp - 1) / sp;
    slice = (slice + 31) / 32 * 32;
    const i64 c0 = static_cast<i64>(kslice) * slice;
    const i64 c1 = (kslice == sp - 1) ? fl.n_in[m] : c0 + slice;

    const u8 *wrow =
        fl.w[m] + static_cast<size_t>(row) * fl.w_row_bytes[m];
    const int nch0 = static_cast<int>(c0 / 32);
    const int nch1 = static_cast<int>(c1 / 32);
    f32 part = 0.0f;
    for (int c = lane + nch0; c < nch1; c += 32)
        part += dot_chunks(fl.wt[m], wrow, fl.in[m], c, c + 1);

    part = d_wave_reduce_sum(part);
    if (lane == 0) {
        if (sp == 1)
            fl.out[m][row] = static_cast<_Float16>(part);
        else
            fl.part[fl.part_off[m] +
                    static_cast<i64>(kslice) * fl.n_out[m] + row] =
                part;
    }
}

// ----------------------------------------------------------------- gate/up --
//
// The FFN's gate and up projections share one activation and are followed by
// silu_mul, which reads the SAME row index of both. One warp per output row
// computes both dots, so the activation is applied to registers and the two
// launches that would produce the operands disappear into the launch that
// already reads them.
//
// Bit-identical to gemm_group() (two gemv rows) + silu_mul_kernel(), and by the
// same argument the residual epilogue uses: each dot keeps gemv_kernel's
// lane-strided chunk order and wave reduce, so the f32 partial is the same
// number; both are rounded to f16 before the activation, which is exactly what
// the separate silu_mul_kernel reads back from memory; and the activation and
// the product are computed in f32 and rounded once, as that kernel does.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    fused_gate_silu_kernel(const u8 *__restrict__ wg, const u8 *__restrict__ wu,
                           int wtg, int wtu, i64 n_out, i64 n_in,
                           size_t w_row_bytes, const _Float16 *__restrict__ x,
                           _Float16 *__restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const i64 row = static_cast<i64>(blockIdx.x) * (THREADS / 32) +
                    static_cast<i64>(threadIdx.x >> 5);
    if (row >= n_out) return;

    const int nch = static_cast<int>(n_in / 32);
    const u8 *grow = wg + static_cast<size_t>(row) * w_row_bytes;
    const u8 *urow = wu + static_cast<size_t>(row) * w_row_bytes;

    f32 g = 0.0f, u = 0.0f;
    for (int c = lane; c < nch; c += 32) {
        g += dot_chunks(wtg, grow, x, c, c + 1);
        u += dot_chunks(wtu, urow, x, c, c + 1);
    }
    g = d_wave_reduce_sum(g);
    u = d_wave_reduce_sum(u);
    if (lane == 0) {
        const f32 hg = static_cast<f32>(static_cast<_Float16>(g));
        const f32 hu = static_cast<f32>(static_cast<_Float16>(u));
        out[row] = d_sat_f16_count(d_silu(hg) * hu, &g_krk_sat_silu);
    }
}

// Launch the fused GEMM phase. `n_mat` matrices share the launch
// (a dependency group of the layer, see FusedLayer). `cu_count` is
// the driver's multiprocessor count the backend already queries — on
// RDNA3/4 that is the WGP count (2 CUs per WGP), which §2h's
// blk_target sweep measured as the right fill-the-device target; `blk_target` is
// the block budget each matrix is topped up to (default: one block
// per multiprocessor — the point §2g shows the bandwidth knee at).
// `splits_out` receives the per-matrix split decisions for reporting.
// Partials beyond `part_cap` fall back to row-parallel for that
// matrix, so the scratch can never be overrun.
// The return value is the SILU answer and nothing else: true when `silu` was
// asked for and the pair kernel ran (out[0] then holds silu(gate)*up and the
// caller must NOT issue silu_mul()), false for every other outcome, including
// the ordinary group launch.
inline bool fused_layer_launch(const u8 *const w[],
                               const _Float16 *const in[],
                               _Float16 *const out[],
                               const i64 n_out[], const i64 n_in[],
                               const size_t w_row_bytes[],
                               int n_mat,
                               f32 *part, i64 part_cap,
                               int cu_count, int wt, int blk_target = 0,
                               int splits_out[8] = nullptr,
                               hipStream_t st = nullptr, bool silu = false,
                               const int wts[] = nullptr) {
    if (n_mat <= 0) return false;
    if (n_mat > 8) n_mat = 8;
    if (blk_target <= 0) blk_target = cu_count;
    FusedLayer fl{};
    fl.count = n_mat;
    int total = 0;
    i64 off = 0;
    for (int m = 0; m < n_mat; m++) {
        fl.wt[m] = wts ? wts[m] : wt;
        fl.w[m] = w[m];
        fl.in[m] = in[m];
        fl.out[m] = out[m];
        fl.n_out[m] = n_out[m];
        fl.n_in[m] = n_in[m];
        fl.w_row_bytes[m] = w_row_bytes[m];
        const int row_blocks =
            static_cast<int>((n_out[m] + 7) / 8);
        // Decode split policy — unlike pick_split, whose cost model
        // is the prefill staging chain, the decode GEMM is
        // latency-bound: a matrix that cannot fill the device on
        // row-parallelism alone is topped up with K-slices so more
        // weight loads overlap. Ties and invalid slices stay at 1.
        int sp = 1;
#if defined(KRK_GFX11) || defined(KRK_GFX12)
        if (row_blocks < blk_target && n_in[m] >= 2 * 64) {
            sp = (blk_target + row_blocks - 1) / row_blocks;
            if (sp > 8) sp = 8;
            i64 slice = (n_in[m] + sp - 1) / sp;
            slice = (slice + 31) / 32 * 32;
            if (slice < 64 || slice * (sp - 1) >= n_in[m])
                sp = 1;
            // A matrix whose partials would not fit the scratch
            // stays row-parallel; its output is then written
            // directly and needs no scratch at all.
            if (sp > 1 && off + sp * n_out[m] > part_cap)
                sp = 1;
        }
#endif
        fl.splits[m] = sp;
        fl.part_off[m] = off;
        if (sp > 1) off += static_cast<i64>(sp) * n_out[m];
        total += row_blocks * sp;
    }
    fl.part = part;
    if (splits_out)
        for (int m = 0; m < n_mat; m++) splits_out[m] = fl.splits[m];
    // gate/up followed by the activation: both matrices read the same
    // activation and neither needs split-K, so one warp can compute the same
    // row of both and apply silu in the epilogue -- one launch where the
    // separate path runs three (gate, up, silu_mul).
    if (silu && n_mat == 2 && n_out[0] == n_out[1] &&
        w_row_bytes[0] == w_row_bytes[1] && in[0] == in[1] &&
        fl.splits[0] == 1 && fl.splits[1] == 1) {
        fused_gate_silu_kernel<256><<<static_cast<unsigned>((n_out[0] + 7) / 8),
                                      256, 0, st>>>(
            w[0], w[1], fl.wt[0], fl.wt[1], n_out[0], n_in[0],
            w_row_bytes[0], in[0], out[0]);
        return true;
    }
    fused_layer_gemv_kernel<256>
        <<<static_cast<unsigned>(total), 256, 0, st>>>(fl);
#if defined(KRK_GFX11) || defined(KRK_GFX12)
    // Split-K tail, identical to gemm_launch's: one small reduce per
    // split matrix, serially summing the fp32 partials to fp16.
    for (int m = 0; m < n_mat; m++)
        if (fl.splits[m] > 1)
            gemm_reduce_kernel<256><<<static_cast<unsigned>(
                                           (fl.n_out[m] + 255) / 256),
                                       256, 0, st>>>(
                part + fl.part_off[m], fl.splits[m], fl.n_out[m],
                fl.n_out[m], out[m]);
#endif
    return false;
}

} // namespace krk

#endif // KRK_KERNEL_FUSED_HPP
