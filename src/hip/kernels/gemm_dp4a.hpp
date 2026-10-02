// ============================================================================
//  gemm_dp4a.hpp — int8 dot-pipe GEMM (DP4A) for the quantized weight formats.
//
//  WHAT THIS IS
//    out[rows, n_out] = X[rows, n_in] * W[n_out, n_in]^T
//
//    for W in Q8_0, Q4_K, Q5_K or Q6_K. v_dot4c_i32_i8 computes four int8 MACs
//    per lane per instruction and is lane-local on RDNA2 (probe_gemm_tile.hip
//    §1 settles that by running it), so the kernel is a plain int8 tile GEMM:
//    the weights are staged into LDS as the *codes* the format already stores,
//    the activations are quantized to int8 per 32-value block, and each k-tile
//    folds its int32 partial into the fp32 accumulator with the block scales.
//    No dequantize-to-fp16 staging on the weight side at all.
//
//  THE AFFINE FORM. Q8_0's codes are values on their own (value = d*q), but the
//  K-quants are affine: value = scale*q - min, with `min` from a second
//  quantized field (Q4_K/Q5_K: dmin*m per 32-value sub-block; Q6_K: a fixed
//  32*scale offset). A dot product against an affine operand splits exactly:
//
//      sum_k a_k * (S*q_k - M)  =  S * sum_k (a_k q_k)  -  M * sum_k a_k
//
//  so the kernel needs the sum of the ACTIVATION codes per block as well as
//  their dot with the weight codes. The activation sum is computed once per
//  row per k-tile during staging and folded with the same FMA that applies the
//  scale, which is why the K-quant path costs one extra FMA per output element
//  per tile rather than a second accumulator.
//
//  Q6_K's SCALE GRANULARITY IS 16, NOT 32. Its int8 scale vector has two
//  entries per 32-value chunk, so the tile is split into two halves: the int32
//  partials are accumulated per half and folded twice, each with its own
//  (scale, min) and its own activation sum. Q4_K/Q5_K/Q8_0 fold once.
//
//  THIS FILE MIRRORS dequant.hpp's FORMAT KNOWLEDGE. Which byte holds which
//  code and where a sub-block scale lives is the format's business, and
//  dequant.hpp already decodes all four formats for the fp16 kernels. The
//  staging below reads the same fields; tools/gemm_i8k_verify.cpp checks every
//  produced code against the HOST reference decoder in quant.hpp, so a drift
//  between the two implementations fails a test rather than silently
//  mis-reading weights.
//
//  WHY IT IS BEHIND A FLAG (KRK_GEMM_INT8=1) AND OFF BY DEFAULT
//    * It is a LOSSY path: int8 activations carry about seven significant bits,
//      so logits differ from the fp16 path's in the last hundredths. Every
//      other kernel here dequantizes exactly; that is a different promise, so
//      this one earns its place with a measurement instead of a default.
//    * Measured ceilings on gfx1031: 52.08 TOP/s int8 against 25.3 TFLOP/s
//      packed fp16, and the per-block folding costs part of that back. On
//      Llama-3.2-1B-Q8_0 prefill (one binary, the flag the only difference):
//      1109 -> 1371 tok/s, identical greedy token stream, top-3 log-probs
//      within 0.2%, identical runs, and the kernel profiles at 1.97 bank
//      conflicts per LDS instruction with 112 VGPRs and no scratch.
//    * With the flag absent nothing here runs and no default path changes.
//
//  LDS TILING. Same tile and the same lane-rotated fetch as the fp16 SIMT
//  kernel: a tile row is BK = 32 bytes = 8 words, so the row index cancels out
//  of the bank number and the plain k-quad index puts all 16 tx lanes on ONE
//  bank. Rotating the quad by the thread's tx (q -> q ^ tx) is a bijection over
//  the 8 quads, so every thread still consumes every quad exactly once and the
//  int32 sum is unchanged. tools/gemm_i8.cpp measured it: SQC_LDS_BANK_CONFLICT
//  /SQ_INSTS_LDS 19.4 -> 7.4 and 6.20 -> 9.27 TOP/s at 4096^3 (11.26 with a
//  BK=16 tile), maxerr 0 against its CPU reference. The same harness also
//  settled the fetch width: a 16-byte-group rotation emits the identical ISA
//  and the identical time as the per-quad rotation, so the fetch is no longer
//  the limiter and the wide/narrow question is closed (measurement, not
//  preference — see §3c of that file).
//
//  NO SPLIT-K. The expert blocks this exists for are short-K and few-row, and
//  split-K's fp32 partial traffic costs more than the parallelism it buys at
//  those shapes. The grid is (n_out/BN, rows/BM).
// ============================================================================
#ifndef KRK_KERNEL_GEMM_DP4A_HPP
#define KRK_KERNEL_GEMM_DP4A_HPP

#include "dequant.hpp"

namespace krk {

// ---------------------------------------------------------------------------
// the weight formats this kernel stages, and their per-32-code layout
// ---------------------------------------------------------------------------

enum Dp4aWeight {
    kDp4aWq8 = 0, // Q8_0: 32 signed codes, one scale per 32
    kDp4aWq4 = 1, // Q4_K: 32 nibbles, scale + min per 32-value sub-block
    kDp4aWq5 = 2, // Q5_K: as Q4_K plus a 5th bit from a second plane
    kDp4aWq6 = 3, // Q6_K: 32 six-bit codes, scale + 32*scale per 16 values
};

// -1 when the format cannot be staged as int8 codes.
inline int dp4a_kind(int wt) {
    switch (static_cast<DType>(wt)) {
        case DType::Q8_0: return kDp4aWq8;
        case DType::Q4_K: return kDp4aWq4;
        case DType::Q5_K: return kDp4aWq5;
        case DType::Q6_K: return kDp4aWq6;
        default: return -1;
    }
}

namespace dp4a_cfg {
// 64 rows rather than 128: the routed-expert GEMMs that motivate this path run
// a few dozen rows per block, and a taller tile mostly multiplies the wasted
// staging. TX=16 keeps the rotated fetch on 16 banks.
constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32; // one 32-value sub-block; see the fold's static_assert
constexpr int TM = 4;
constexpr int TN = 8;
constexpr int STAGES = 2;
constexpr int TX = BN / TN; // 16
constexpr int TY = BM / TM; // 16
constexpr int THREADS = TX * TY;
// A K-quant superblock is 256 values = 8 sub-blocks of 32, so eight k-tiles
// share one block header.
constexpr int KBLOCK_CHUNKS = 8;
constexpr size_t Q8_BLOCK_BYTES = 34;

// The fold applies one (scale, min) pair to one k-tile's int32 partial, so a
// k-tile that straddled two sub-blocks would be wrong. Every format above has
// 32-value sub-blocks, except Q6_K whose 16-value halves are handled by
// splitting the tile.
static_assert(BK == 32, "the per-block fold assumes 32-value sub-blocks");
static_assert(TX == 16, "the rotated fetch wants 16 tx lanes");
static_assert(KBLOCK_CHUNKS == 8, "K-quant superblocks hold eight 32-value chunks");
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
// weight staging: one output row's 32 codes for one k-tile
// ---------------------------------------------------------------------------
//
// Every branch mirrors exactly one decoder in dequant.hpp (dequant_chunk<T>);
// the fields it reads are named for the format they come from. `kt` is the
// global k-tile index and `cpb` the chunks per superblock, so the block header
// is only re-read when the tile crosses into a new superblock — the point of
// staging 32 values at a time is that a K-quant superblock's scale bytes are
// read 8x less often than the codes.
template <int WQ>
__device__ __forceinline__ void stage_weight_tile(const u8 *__restrict__ w,
                                                  size_t w_row_bytes, i64 gc,
                                                  i64 kt, i8 *dst, f32 *sc,
                                                  f32 *mn) {
    const u8 *b = w + static_cast<size_t>(gc) * w_row_bytes;

    if (WQ == kDp4aWq8) {
        // 34-byte block: f16 scale then 32 signed codes, 4-byte aligned only
        // when the tile index is even, so the codes are assembled from bytes.
        const u8 *blk = b + static_cast<size_t>(kt) * dp4a_cfg::Q8_BLOCK_BYTES;
        sc[0] = d_h2f(static_cast<u16>(blk[0] | (blk[1] << 8)));
        mn[0] = 0.0f; // Q8_0 is not affine
        const u8 *c = blk + 2;
#pragma unroll
        for (int z = 0; z < dp4a_cfg::BK / 4; z++) {
            const u8 *c4 = c + z * 4;
            *reinterpret_cast<u32 *>(dst + z * 4) = static_cast<u32>(c4[0]) |
                                                    (static_cast<u32>(c4[1]) << 8) |
                                                    (static_cast<u32>(c4[2]) << 16) |
                                                    (static_cast<u32>(c4[3]) << 24);
        }
        return;
    }

    const int chunk = static_cast<int>(kt % dp4a_cfg::KBLOCK_CHUNKS);
    const u8 *blk = b + (static_cast<size_t>(kt) / dp4a_cfg::KBLOCK_CHUNKS) *
                            dtype_block_bytes_dev(static_cast<int>(
                                WQ == kDp4aWq4   ? DType::Q4_K
                                : WQ == kDp4aWq5 ? DType::Q5_K
                                                 : DType::Q6_K));

    if (WQ == kDp4aWq4 || WQ == kDp4aWq5) {
        const f32 d = d_h2f(static_cast<u16>(blk[0] | (blk[1] << 8)));
        const f32 dmin = d_h2f(static_cast<u16>(blk[2] | (blk[3] << 8)));
        const u8 *scales = blk + 4;
        const int g = chunk / 2;
        const int half = chunk % 2;
        u8 si = 0, mi = 0;
        dev_scale_min_k4(g * 2 + half, scales, &si, &mi);
        sc[0] = d * static_cast<f32>(si);
        mn[0] = dmin * static_cast<f32>(mi);
        if (WQ == kDp4aWq4) {
            const u8 *q = blk + 16;
            const int shift = half == 0 ? 0 : 4;
#pragma unroll
            for (int z = 0; z < dp4a_cfg::BK / 4; z++) {
                const u32 pack = d_u32(q + g * 32 + z * 4);
                u32 out = 0u;
#pragma unroll
                for (int e = 0; e < 4; e++)
                    out |= static_cast<u32>((pack >> (8 * e + shift)) & 0xF) << (8 * e);
                *reinterpret_cast<u32 *>(dst + z * 4) = out;
            }
        } else {
            // Q5_K: the 4-bit plane plus a high bit per code, held in a plane
            // whose bit position depends on the sub-block (2*g or 2*g+1).
            const u8 *qh = blk + 16;
            const u8 *q = blk + 48;
            const u8 bit = static_cast<u8>(half == 0 ? (1u << (2 * g)) : (2u << (2 * g)));
#pragma unroll
            for (int z = 0; z < dp4a_cfg::BK / 4; z++) {
                u32 out = 0u;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int l = z * 4 + e;
                    const int base = half == 0 ? (q[g * 32 + l] & 0xF)
                                               : (q[g * 32 + l] >> 4);
                    const int v = base + ((qh[l] & bit) ? 16 : 0);
                    out |= static_cast<u32>(v) << (8 * e);
                }
                *reinterpret_cast<u32 *>(dst + z * 4) = out;
            }
        }
        return;
    }

    // Q6_K: 6-bit codes from a 4-bit plane and a 2-bit plane, one int8 scale
    // per 16 values. The tile is two halves of 16; each half gets its own
    // (scale, min) pair, which is why the fold runs twice for this format.
    const f32 d = d_h2f(static_cast<u16>(blk[208] | (blk[209] << 8)));
    const int g = chunk / 4;
    const int p = chunk % 4;
    const u8 *ql = blk + g * 64;
    const u8 *qh = blk + 128 + g * 32;
    const i8 *sb = reinterpret_cast<const i8 *>(blk + 192 + g * 8);
    // The codes are built and stored one dword at a time, with the quadrant (p)
    // selected ONCE rather than per code. Measured on gfx1031, production
    // flags: the byte-store form with a per-code quadrant branch needed 206
    // VGPRs and 260 B of scratch against 116/117 and 0 for Q4_K/Q5_K, and cost
    // 9.6 ms per dispatch against 0.46. Building all eight dwords into a local
    // array first (the previous shape of this code) still needed 206/260 -- the
    // array stayed live across the quadrant branch -- so the store sits inside
    // the loop, where only one dword is live at a time.
    u32 *dst32 = reinterpret_cast<u32 *>(dst);
    if (p == 0) {
#pragma unroll
        for (int z = 0; z < 8; z++) {
            u32 v = 0u;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int l = z * 4 + e;
                v |= static_cast<u32>((ql[l] & 0xF) | ((qh[l] & 3) << 4)) << (8 * e);
            }
            dst32[z] = v;
        }
    } else if (p == 1) {
#pragma unroll
        for (int z = 0; z < 8; z++) {
            u32 v = 0u;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int l = z * 4 + e;
                v |= static_cast<u32>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))
                     << (8 * e);
            }
            dst32[z] = v;
        }
    } else if (p == 2) {
#pragma unroll
        for (int z = 0; z < 8; z++) {
            u32 v = 0u;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int l = z * 4 + e;
                v |= static_cast<u32>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4))
                     << (8 * e);
            }
            dst32[z] = v;
        }
    } else {
#pragma unroll
        for (int z = 0; z < 8; z++) {
            u32 v = 0u;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int l = z * 4 + e;
                v |= static_cast<u32>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))
                     << (8 * e);
            }
            dst32[z] = v;
        }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const f32 s = d * static_cast<f32>(sb[h + 2 * p]);
        sc[h] = s;
        mn[h] = 32.0f * s; // the format stores (q - 32) * scale
    }
}

// ---------------------------------------------------------------------------
// the tile kernel
// ---------------------------------------------------------------------------

// Which quad group the rotated fetch reads for a given (half, round). Host-
// callable on purpose: the property the per-half dot loop depends on -- that
// the NH*RH (half, round) pairs visit all BK/4 quads exactly once, and that each
// pair's rotated index lands inside its OWN half -- follows from this mapping
// alone, so tools/gemm_i8k_verify.cpp checks it on the CPU.
//
// tx enters the rotation as `q ^ tx`, so the index whose image falls in half h
// is the one whose own bit 2 equals h ^ tx's bit 2; that is what the ^ below
// selects. Deriving the half from the loop's `q` instead of the rotated `qq`
// pairs one half's activations with the other half's scale for every lane whose
// tx flips that bit -- columns 32 and up at TN=8, which is exactly where the
// code probe put the mismatch when this was first written.
__host__ __device__ inline int dp4a_quad_index(int h, int r, int tx, int nh) {
    constexpr int rounds = dp4a_cfg::BK / 4;
    const int rh = rounds / nh;
    const int q = (nh == 2) ? (((h ^ ((tx >> 2) & 1)) * rh) | r) : r;
    return (q ^ tx) & (rounds - 1);
}

// waves_per_eu is the register budget that actually binds on AMD: 8 waves per
// EU asks for two 256-thread blocks per CU, which on gfx10 means
// 65536 / (2 * 256) = 128 VGPRs per thread. The second argument of
// __launch_bounds__ does NOT do this -- adding it changed neither arch's
// counts by a single register -- so the budget is spelled this way instead.
// Q6_K is the format that needs it: with its scratch gone it still sat at 161
// VGPRs against 116/117 for Q4_K/Q5_K, and anything above 128 drops a block to
// one per CU.
template <int WQ>
__global__ void __launch_bounds__(dp4a_cfg::THREADS)
                       __attribute__((amdgpu_waves_per_eu(8)))
    gemm_dp4a_kernel(const u8 *__restrict__ w, i64 n_out, i64 n_in,
                     size_t w_row_bytes, const i8 *__restrict__ codes,
                     const f32 *__restrict__ scales, i64 rows,
                     _Float16 *__restrict__ out) {
    using namespace dp4a_cfg;
    constexpr int NH = (WQ == kDp4aWq6) ? 2 : 1; // (scale, min) pairs per tile

    __shared__ __align__(16) i8 aw[STAGES][BM * BK];
    __shared__ __align__(16) i8 bw[STAGES][BN * BK];
    // One scale per row per tile, one (scale, min) per row per half for the
    // weight side, and the activation code sums the affine form needs.
    __shared__ f32 asc[STAGES][BM];
    __shared__ f32 asum[STAGES][2][BM];
    // Per row, then per half, so the staging writes both of a Q6_K row's
    // (scale, min) pairs through one pointer.
    __shared__ f32 bsc[STAGES][BN][NH];
    __shared__ f32 bmin[STAGES][BN][NH];

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
    // of 32), so it copies as u32.
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
        // The affine formats need sum_k a_k per half. Taken from GLOBAL rather
        // than from the LDS tile written just above: that copy is spread across
        // threads, so summing the tile here reads rows other threads have not
        // finished filling. Measured: that race put ~1e+3 of absolute error into
        // Q6_K's outputs (whose -32 offset multiplies any error in this sum by
        // 32*scale), which is what tools/gemm_i8k_verify.cpp exists to catch.
        // Global reads need no barrier and hit L1/L2, and 8 dword loads + 32
        // signed byte adds per row per tile is nothing against TM*TN*8 dots.
        // Q8_0 is not affine, so it neither computes nor reads this at all.
        if (WQ != kDp4aWq8) {
            for (int r = tid; r < BM; r += THREADS) {
                const i64 gr = row_base + r;
                i32 lo = 0, hi = 0;
                if (gr < rows) {
#pragma unroll
                    for (int z = 0; z < BK / 4; z++) {
                        const u32 v = *reinterpret_cast<const u32 *>(
                            codes + gr * n_in + k0 + z * 4);
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            const i32 c = static_cast<i32>(
                                static_cast<i8>((v >> (8 * e)) & 0xFF));
                            if (z < BK / 8) lo += c;
                            else hi += c;
                        }
                    }
                }
                asum[stage][0][r] = static_cast<f32>(lo);
                asum[stage][1][r] = static_cast<f32>(hi);
            }
        }

        for (int r = tid; r < BN; r += THREADS) {
            const i64 gc = col_base + r;
            i8 *dst = &bw[stage][r * BK];
#pragma unroll
            for (int h = 0; h < NH; h++) {
                bsc[stage][r][h] = 0.0f;
                bmin[stage][r][h] = 0.0f;
            }
            if (gc >= n_out) {
#pragma unroll
                for (int z = 0; z < BK / 4; z++)
                    *reinterpret_cast<u32 *>(dst + z * 4) = 0u;
                continue;
            }
            stage_weight_tile<WQ>(w, w_row_bytes, gc, kt, dst, &bsc[stage][r][0],
                                  &bmin[stage][r][0]);
        }
    };

    if (k_tiles > 0) load_stage(0, 0);
    __syncthreads();

    int stage = 0;
    for (i64 kt = 0; kt < k_tiles; kt++) {
        const int nxt = (stage + 1) % STAGES;
        if (kt + 1 < k_tiles) load_stage(nxt, kt + 1);

        // Packed quad fetch with the lane-rotated index; see the header note.
        //
        // The k-tile is dotted one HALF at a time, with TM*TN int accumulators
        // live rather than TM*TN per half. Q6_K is the format that needs two
        // halves, and holding both at once measured 206 VGPRs and 260 B of
        // scratch on gfx1031 against 116/117 and 0 for Q4_K/Q5_K -- the live
        // accumulators were the whole difference, which is why reshaping the
        // staging never moved those numbers. For NH == 1 this is the single
        // pass it always was: RH == BK/4 and h is always 0.
        constexpr int ROUNDS = BK / 4; // quad groups per k-tile
        constexpr int RH = ROUNDS / NH; // quad groups per half
        i32 iacc[TM][TN];
#pragma unroll
        for (int h = 0; h < NH; h++) {
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) iacc[i][j] = 0;

#pragma unroll
            for (int r = 0; r < RH; r++) {
                // Which quad group this pass reads; the mapping and the reason
                // it is written that way are in dp4a_quad_index above.
                const int qq = dp4a_quad_index(h, r, tx, NH);
                i32 a[TM], b[TN];
#pragma unroll
                for (int i = 0; i < TM; i++)
                    a[i] = *reinterpret_cast<const i32 *>(
                        &aw[stage][(ty * TM + i) * BK + qq * 4]);
#pragma unroll
                for (int j = 0; j < TN; j++)
                    b[j] = *reinterpret_cast<const i32 *>(
                        &bw[stage][(tx * TN + j) * BK + qq * 4]);
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        iacc[i][j] = d_dot4(a[i], b[j], iacc[i][j]);
            }

            // Fold: acc += (sa*S) * dot - (sa*sum) * M, the two terms of the
            // affine product. Q8_0 has M == 0 and one half, Q4_K/Q5_K one half
            // with M from dmin*m, Q6_K two halves with M = 32*S.
#pragma unroll
            for (int i = 0; i < TM; i++) {
                const int ri = ty * TM + i;
                const f32 sa = asc[stage][ri];
                const f32 ssum = (NH == 2) ? sa * asum[stage][h][ri]
                                           : sa * (asum[stage][0][ri] +
                                                   asum[stage][1][ri]);
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    const int cj = tx * TN + j;
                    acc[i][j] = fmaf(sa * bsc[stage][cj][h],
                                     static_cast<f32>(iacc[i][j]), acc[i][j]);
                    if (WQ != kDp4aWq8)
                        acc[i][j] = fmaf(-ssum, bmin[stage][cj][h], acc[i][j]);
                }
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

// Legality, not policy: the caller still decides whether the path is enabled at
// all. Shapes below a full tile row are excluded because a block would then
// spend more on staging its own tile than on the rows it actually has. The
// K-quants additionally need whole 256-value superblocks in the row, because
// the block header is addressed from the k-tile index.
//
// Q6_K is excluded on measured cost, not on correctness: the format's mixed
// 6-bit planes plus a per-16-value scale *and* offset make the staging body
// wide enough that the compiler spills (208 VGPR, 272 B scratch) whatever
// packing it is written with. On the 9B Q4_K_M MoE model that one kernel cost
// 245 ms over 32 dispatches, which on its own turned the whole flagged path
// into a net loss (327 ms of GEMM against the fp16 tile's 256 ms). It stays
// compiled and is covered by tools/gemm_i8k_verify.cpp so that a register
// budget can re-enable it here once the staging is narrow enough.
constexpr bool kDp4aEnableQ6K = false;

// The staging only amortises over a long K. Every call pays the activation
// quantize (rows*n_in bytes written plus one scale per 32 values) and every
// k-tile pays its own unpack, so the win is a function of how much arithmetic
// those fixed costs are spread over. Measured model-level A/B, one binary, the
// flag the only difference: SmolLM2-135M (K = 576/1536) 2827 -> 1641 tok/s,
// Llama-3.2-1B (K = 2048/8192) 1120 -> 2060, Qwen3.5-9B (K = 4096/12288)
// 175.6 -> 198.7. The floor sits between the two groups, and per-shape
// attribution of the 135M loss (K against output width against tile occupancy)
// is still open — this is the measured boundary, not a derived one.
constexpr i64 kDp4aMinK = 2048;

inline bool dp4a_ok(int wt, i64 rows, i64 n_out, i64 n_in) {
    const int kind = dp4a_kind(wt);
    if (kind < 0 || rows < 16 || n_out <= 0 || n_in < kDp4aMinK) return false;
    if (kind == kDp4aWq6 && !kDp4aEnableQ6K) return false;
    if ((n_in & (dp4a_cfg::BK - 1)) != 0) return false;
    if (kind != kDp4aWq8 && (n_in % 256) != 0) return false;
    return true;
}

// Quantize then tile. `codes` is [rows, n_in] int8 and `scales` is
// [rows, n_in/32] f32 — both owned by the caller (the backend keeps them
// across calls) so nothing allocates on this path.
template <int WQ>
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
    gemm_dp4a_kernel<WQ><<<grid, dp4a_cfg::THREADS>>>(w, n_out, n_in, w_row_bytes,
                                                     codes, scales, rows, out);
}

// Runtime dtype -> kernel. Returns false when the format has no int8 staging,
// which the caller turns into "keep the fp16 tile".
inline bool gemm_dp4a_dispatch(int wt, const u8 *w, i64 n_out, i64 n_in,
                               size_t w_row_bytes, const _Float16 *x, i64 rows,
                               _Float16 *out, i8 *codes, f32 *scales) {
    switch (dp4a_kind(wt)) {
        case kDp4aWq8:
            gemm_dp4a_launch<kDp4aWq8>(w, n_out, n_in, w_row_bytes, x, rows, out,
                                       codes, scales);
            return true;
        case kDp4aWq4:
            gemm_dp4a_launch<kDp4aWq4>(w, n_out, n_in, w_row_bytes, x, rows, out,
                                       codes, scales);
            return true;
        case kDp4aWq5:
            gemm_dp4a_launch<kDp4aWq5>(w, n_out, n_in, w_row_bytes, x, rows, out,
                                       codes, scales);
            return true;
        case kDp4aWq6:
            gemm_dp4a_launch<kDp4aWq6>(w, n_out, n_in, w_row_bytes, x, rows, out,
                                       codes, scales);
            return true;
        default:
            return false;
    }
}

} // namespace krk

#endif // KRK_KERNEL_GEMM_DP4A_HPP
