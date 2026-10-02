// ============================================================================
//  gemm_dp4a.hpp — Q8_0 prefill GEMM on the int8 dot pipe (v_dot4c_i32_i8).
//
//  WHAT THIS IS
//    out[rows, n_out] = X[rows, n_in] * W[n_out, n_in]^T   for W in Q8_0.
//
//    Q8_0 is the one GGUF weight format whose payload *is* int8: 32 signed
//    codes behind a single fp16 scale. v_dot4c_i32_i8 consumes those bytes
//    directly — four int8 MACs per lane per instruction, lane-local on this
//    part — so the weight tile needs no dequantize-to-fp16 staging at all. The
//    activations are quantized to int8 per 32-value block on the way in, and
//    each k-tile's int32 partial is folded into the fp32 accumulator with the
//    product of the two blocks' scales.
//
//  WHY IT IS BEHIND A FLAG (KRK_GEMM_INT8=1) AND OFF BY DEFAULT
//    * It is a LOSSY path. Every other kernel in this tree dequantizes exactly
//      and produces bit-identical results run to run and path to path; int8
//      activations carry about seven significant bits, so this path's logits
//      differ from the fp16 path's in the last hundredths. That is the deal
//      every DP4A engine makes, but it is a different deal from the one the
//      rest of kraken makes, so it earns its place with a measurement rather
//      than by becoming the default.
//    * On gfx1031 (RX 6700 XT) the measured register-only ceilings are
//      52.08 TOP/s for int8 against 25.3 TFLOP/s for packed fp16 — 2x, not the
//      4x the raw issue rates suggest — and the per-block scale folding costs
//      part of that back in VALU ops. It is most likely to win where the
//      weights are the traffic (Q8_0 is 1.06 B/value against fp16's 2 B) and
//      least likely where the multiply is the wall.
//    * With the flag absent, nothing here runs and no default path changes.
//
//  MEASURED ON THE CARD (gfx1031, Llama-3.2-1B-Q8_0, one binary, the flag the
//  only difference): prefill 1109 -> 1371 tok/s (+24%); decode unchanged, since
//  rows == 1 is the gemv path and never reaches here. The greedy token stream
//  is identical, the top-3 log-probs differ by at most 0.03 on ~19.0 (0.2%),
//  and two runs are byte-identical. rocprofv3 on this kernel: 1.97 bank
//  conflicts per LDS instruction (the 2-way floor for a 32-byte row), 112
//  VGPRs, 0 scratch.
//
//  LDS TILING. Same tile and the same lane-rotated fetch as the fp16 SIMT
//  kernel, for the same reason: a tile row is BK = 32 bytes = 8 words, so the
//  row index cancels out of the bank number and the plain k-quad index puts
//  all 16 tx lanes on ONE bank (16 different addresses, 16 replays). Rotating
//  the quad by the thread's tx (q -> q ^ tx) is a bijection over the 8 quads,
//  so every thread still consumes every quad exactly once and the int32 sum is
//  unchanged. tools/gemm_i8.cpp measured the rotation on this exact tile:
//  SQC_LDS_BANK_CONFLICT/SQ_INSTS_LDS 19.4 -> 7.4 and 6.20 -> 9.27 TOP/s at
//  4096^3 (11.26 with a BK=16 tile), maxerr 0 against its CPU reference.
//
//  NO SPLIT-K. The expert blocks this exists for are short-K and few-row, and
//  split-K's fp32 partial traffic costs more than the parallelism it buys at
//  those shapes. The grid is (n_out/BN, rows/BM).
// ============================================================================
#ifndef KRK_KERNEL_GEMM_DP4A_HPP
#define KRK_KERNEL_GEMM_DP4A_HPP

#include "dequant.hpp"

namespace krk {

namespace dp4a_cfg {
// 64 rows rather than 128: the routed-expert GEMMs that motivate this path run
// a few dozen rows per block, and a taller tile mostly multiplies the wasted
// staging. TX=16 keeps the rotated fetch on 16 banks.
constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32; // one Q8_0 block; see the fold's static_assert
constexpr int TM = 4;
constexpr int TN = 8;
constexpr int STAGES = 2;
constexpr int TX = BN / TN; // 16
constexpr int TY = BM / TM; // 16
constexpr int THREADS = TX * TY;
// Bytes of one Q8_0 block: 2 B fp16 scale + 32 int8 codes.
constexpr size_t Q8_BLOCK_BYTES = 34;

// The fold multiplies one k-tile's int32 partial by a single scale pair, so a
// k-tile that straddled two blocks of different scale would be wrong. Q8_0
// blocks are 32 values, hence BK == 32.
static_assert(BK == 32, "the per-block fold assumes one Q8_0 block per k-tile");
static_assert(TX == 16, "the rotated fetch wants 16 tx lanes");
} // namespace dp4a_cfg

// ---------------------------------------------------------------------------
// activation quantization: fp16 rows -> int8 codes + one scale per 32-block
// ---------------------------------------------------------------------------
//
// One warp per (row, block) pair: the 32 lanes hold the block, the absmax is a
// five-step shuffle reduction, and lane 0 stores the scale. The scale is the
// block's absmax over 127 so every code lands inside [-127, 127]; a block of
// zeros keeps scale 0 and codes 0, which contributes nothing instead of
// dividing by zero. Same convention the int8 GEMM harness used, which is what
// its maxerr-0 verification was against.
__global__ void quantize_act_kernel(const _Float16 *__restrict__ x,
                                    i8 *__restrict__ codes,
                                    f32 *__restrict__ scales, i64 rows, i64 n_in) {
    const i64 nblk = n_in / 32;
    const i64 pair = (static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (pair >= rows * nblk) return;

    const i64 row = pair / nblk;
    const i64 blk = pair - row * nblk;
    const i64 off = row * n_in + blk * 32;

    const f32 v = static_cast<f32>(x[off + lane]);
    const f32 amax = d_wave_reduce_max(fabsf(v));
    const f32 d = amax > 0.0f ? amax / 127.0f : 0.0f;

    i32 q = 0;
    if (d > 0.0f) {
        q = static_cast<i32>(__float2int_rn(v / d));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
    }
    codes[off + lane] = static_cast<i8>(q);
    if (lane == 0) scales[pair] = d;
}

// ---------------------------------------------------------------------------
// the tile kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(dp4a_cfg::THREADS)
    gemm_dp4a_kernel(const u8 *__restrict__ w, i64 n_out, i64 n_in,
                     size_t w_row_bytes, const i8 *__restrict__ codes,
                     const f32 *__restrict__ scales, i64 rows,
                     _Float16 *__restrict__ out) {
    using namespace dp4a_cfg;
    __shared__ __align__(16) i8 aw[STAGES][BM * BK];
    __shared__ __align__(16) i8 bw[STAGES][BN * BK];
    // One f32 per row per stage: with BK == 32 a k-tile is exactly one block,
    // so a stage needs one scale per row, not one per row per k.
    __shared__ f32 asc[STAGES][BM];
    __shared__ f32 bsc[STAGES][BN];

    const i64 row_base = static_cast<i64>(blockIdx.y) * BM;
    const i64 col_base = static_cast<i64>(blockIdx.x) * BN;
    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid % TX;
    const int ty = tid / TX;
    const i64 nblk = n_in / BK;
    const i64 k_tiles = n_in / BK;

    f32 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    // Stage `stage` holds k-tile `kt`. The activation side is already int8,
    // row-major, and 4-byte aligned (rows are n_in wide and n_in is a multiple
    // of 32), so it copies as u32. The weight side is a Q8_0 block: 2 bytes of
    // fp16 scale plus 32 codes at an arbitrary byte offset inside the row, so
    // the codes are assembled from bytes rather than a misaligned u32 read.
    auto load_stage = [&](int stage, i64 kt) {
        const i64 k0 = kt * BK;

        for (int i = tid; i < BM * (BK / 4); i += THREADS) {
            const int r = i / (BK / 4);
            const int z = i - r * (BK / 4);
            const i64 gr = row_base + r;
            u32 v = 0u;
            if (gr < rows)
                v = *reinterpret_cast<const u32 *>(codes + gr * n_in + k0 + z * 4);
            *reinterpret_cast<u32 *>(&aw[stage][r * BK + z * 4]) = v;
        }
        for (int r = tid; r < BM; r += THREADS) {
            const i64 gr = row_base + r;
            asc[stage][r] = gr < rows ? scales[gr * nblk + kt] : 0.0f;
        }

        for (int r = tid; r < BN; r += THREADS) {
            const i64 gc = col_base + r;
            u32 *dst = reinterpret_cast<u32 *>(&bw[stage][r * BK]);
            if (gc >= n_out) {
#pragma unroll
                for (int z = 0; z < BK / 4; z++) dst[z] = 0u;
                bsc[stage][r] = 0.0f;
                continue;
            }
            const u8 *blk = w + static_cast<size_t>(gc) * w_row_bytes +
                            static_cast<size_t>(kt) * Q8_BLOCK_BYTES;
            bsc[stage][r] = d_h2f(static_cast<u16>(blk[0] | (blk[1] << 8)));
            const u8 *c = blk + 2;
#pragma unroll
            for (int z = 0; z < BK / 4; z++) {
                const u8 *c4 = c + z * 4;
                dst[z] = static_cast<u32>(c4[0]) |
                         (static_cast<u32>(c4[1]) << 8) |
                         (static_cast<u32>(c4[2]) << 16) |
                         (static_cast<u32>(c4[3]) << 24);
            }
        }
    };

    if (k_tiles > 0) load_stage(0, 0);
    __syncthreads();

    int stage = 0;
    for (i64 kt = 0; kt < k_tiles; kt++) {
        const int nxt = (stage + 1) % STAGES;
        if (kt + 1 < k_tiles) load_stage(nxt, kt + 1);

        i32 iacc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) iacc[i][j] = 0;

        // Packed quad fetch with the lane-rotated index; see the header note.
#pragma unroll
        for (int q = 0; q < BK / 4; q++) {
            const int qq = (q ^ tx) & (BK / 4 - 1);
            i32 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
                a[i] = *reinterpret_cast<const i32 *>(&aw[stage][(ty * TM + i) * BK + qq * 4]);
#pragma unroll
            for (int j = 0; j < TN; j++)
                b[j] = *reinterpret_cast<const i32 *>(&bw[stage][(tx * TN + j) * BK + qq * 4]);
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    iacc[i][j] = d_dot4(a[i], b[j], iacc[i][j]);
        }

        // Fold this k-tile's int32 partial with its two block scales. One
        // k-tile is one 32-code block, so the fp32 accumulator never sums a
        // partial whose scale it does not know: the multiply is exact for the
        // block it came from.
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const f32 sa = asc[stage][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const f32 sb = bsc[stage][tx * TN + j];
                acc[i][j] = fmaf(static_cast<f32>(iacc[i][j]), sa * sb, acc[i][j]);
            }
        }

        __syncthreads();
        stage = nxt;
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        const i64 gr = row_base + ty * TM + i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const i64 gc = col_base + tx * TN + j;
            if (gc >= n_out) continue;
            out[gr * n_out + gc] = static_cast<_Float16>(acc[i][j]);
        }
    }
}

// ---------------------------------------------------------------------------
// host-side entry point
// ---------------------------------------------------------------------------

// Legality, not policy: the caller still decides whether the path is enabled
// at all. Shapes below a full tile row are excluded because a block would then
// spend more on staging its own tile than on the rows it actually has.
inline bool dp4a_ok(int wt, i64 rows, i64 n_out, i64 n_in) {
    return wt == static_cast<int>(DType::Q8_0) && rows >= 16 && n_out > 0 &&
           n_in >= dp4a_cfg::BK && (n_in & (dp4a_cfg::BK - 1)) == 0;
}

// Quantize then tile. `codes` is [rows, n_in] int8 and `scales` is
// [rows, n_in/32] f32 — both owned by the caller (the backend keeps them
// across calls) so nothing allocates on this path.
inline void gemm_dp4a_launch(const u8 *w, i64 n_out, i64 n_in, size_t w_row_bytes,
                             const _Float16 *x, i64 rows, _Float16 *out,
                             i8 *codes, f32 *scales) {
    const i64 pairs = rows * (n_in / dp4a_cfg::BK);
    const unsigned qblocks = static_cast<unsigned>((pairs + 7) / 8);
    quantize_act_kernel<<<qblocks, 256>>>(x, codes, scales, rows, n_in);

    const dim3 grid(static_cast<unsigned>(
                        (n_out + dp4a_cfg::BN - 1) / dp4a_cfg::BN),
                    static_cast<unsigned>(
                        (rows + dp4a_cfg::BM - 1) / dp4a_cfg::BM));
    gemm_dp4a_kernel<<<grid, dp4a_cfg::THREADS>>>(w, n_out, n_in, w_row_bytes,
                                                 codes, scales, rows, out);
}

} // namespace krk

#endif // KRK_KERNEL_GEMM_DP4A_HPP
