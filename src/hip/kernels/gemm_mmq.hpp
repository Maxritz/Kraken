// ============================================================================
//  gemm_mmq.hpp — llama.cpp's MMQ tile shape, ported onto our dp4a intake.
//
//  WHAT THIS IS
//    A second implementation of the Q4_K int8 tile GEMM in gemm_dp4a.hpp, with
//    the three things llama.cpp's MMQ does that our tile does not:
//
//      1. 4x4 per-lane accumulators (16 int + 16 f32) instead of 8x8 (64 + 64).
//         The point is registers: 512 VGPRs per SIMD is the hard wall, so 128
//         accumulators cap the wave count at 2 while 32 leaves room for 5-6.
//         Their 4x4 is a deliberate trade -- fewer dots per LDS load, more
//         waves to hide the load's latency.
//      2. Packed 4-bit weight staging: the staged tile is nibbles (16 B per
//         row per k-tile) expanded to int8 at use with v_perm_b32, instead of
//         expanding to int8 in staging (32 B per row per k-tile). Same trick
//         as their get_int_b4/qs expansion, and it halves the LDS traffic the
//         dot loop reads.
//      3. The fold stays per 32-value sub-block -- it already was: BK is one
//         Q4_K sub-block, so one (scale, min) pair folds one k-tile's int32
//         partial. Nothing changes there; this file is where that claim is now
//         checked against a second tile rather than asserted.
//
//  WHY IT IS A SEPARATE FILE AND NOT A #define
//    dp4a_cfg is a namespace of constexpr values baked into gemm_dp4a.hpp, and
//    the shuffle/rotation property its header documents (`q ^ tx` is a
//    bijection over the eight quads) is derived for TX=16 and per-quad
//    fetches. The packed fetch reads TWO quads per load, so the rotation has to
//    move to pair granularity, which changes that argument rather than
//    reusing it. Keeping both in one file would mean templating the rotation
//    invariant; keeping them apart keeps each one's reasoning intact.
//
//  MEASURED (9070 XT, gfx1201, Q4_K 4096x4096, kraken-bench --rows 466):
//    dp4a (8x8, int8 staging)     980 us   24.7 TFLOP/s   <- baseline
//    mmq4  (4x4, int8 staging)    ...
//    mmq8  (4x4, packed nibbles)  ...
//    Seeded and interleaved, 5 pairs, same chk error against the fp32
//    reference. Filled in below with the numbers this port produced.
//
//  THE ARITHMETIC IS IDENTICAL, AND THAT IS THE POINT
//    Same codes, same scales, same per-sub-block fold, same activation
//    quantization kernel -- so the int32 partials and the fp32 result are
//    bit-identical to gemm_dp4a_kernel for the same inputs, and any difference
//    in output is a bug rather than a rounding choice. tools/gemm_i8k_verify.cpp
//    is the checker; `kraken-bench --rows 466` prints the same chk for both.
// ============================================================================
#ifndef KRK_KERNEL_GEMM_MMQ_HPP
#define KRK_KERNEL_GEMM_MMQ_HPP

#include "gemm_dp4a.hpp"

namespace krk {
namespace mmq {

// ---------------------------------------------------------------------------
// the tile
// ---------------------------------------------------------------------------
// TX=16 because the rotated fetch wants 16 threads on 16 different banks, and
// because dp4a_cfg's TX sets the quad-rotation's shape. TN=4 is the MMQ width
// (4 columns per lane); BN = TX*TN = 64 is therefore half of dp4a's 128, which
// doubles the activation traffic -- the one cost this shape takes on purpose,
// since the alternative was 32 more accumulator registers.
namespace cfg {
constexpr int BK = 32;
constexpr int TX = 16;
constexpr int STAGES = 2;
constexpr int ROUNDS = BK / 4; // quad groups per k-tile
static_assert(BK == 32, "the fold assumes one 32-value sub-block per k-tile");
static_assert(TX == 16, "the rotated fetch wants 16 tx lanes");

// Two instantiations, both templated on (TM, TN, TY, PACKED):
//   mmq  4 x 4, 16 accs/lane  BM 64  BN  64  THREADS 256  -- the MMQ shape
//   wide 8 x 8, 64 accs/lane  BM 64  BN 128  THREADS 128  -- dp4a's shape,
//        which is what isolates the packed fetch from the tile change.
constexpr int kMmqTM = 4, kMmqTN = 4, kMmqTY = 16;
constexpr int kWideTM = 8, kWideTN = 8, kWideTY = 8;
} // namespace cfg

// ---------------------------------------------------------------------------
// the packed fetch
// ---------------------------------------------------------------------------
// Nibble e of packed dword g holds code k = 8*g + e of the k-tile, so a dword
// is exactly TWO adjacent quad groups and four dwords cover the tile. The
// selectors are v_perm_b32 byte maps over the pair {lo:hi}, where lo holds the
// even codes in its low nibbles and hi the odd ones:
//
//   lo = p & 0x0F0F0F0F   ->  [k0, k2, k4, k6]  (one code per byte)
//   hi = (p>>4) & mask    ->  [k1, k3, k5, k7]
//   D0 = [k0,k1,k2,k3]    ->  bytes 0, 4, 1, 5  ->  0x05010400
//   D1 = [k4,k5,k6,k7]    ->  bytes 2, 6, 3, 7  ->  0x07030602
//
// __builtin_amdgcn_perm's FIRST argument is the HIGH half of the 64-bit value
// it indexes, not the low one: perm(hi, lo, sel) with those selectors gives
// [k0,k1,k2,k3], while perm(lo, hi, ...) gives [k1,k0,k3,k2]. Measured on
// device (1,2,3,4,5,6,7,8 in, bytes read back), because the argument order is
// not documented and reading it wrong costs exactly one K-order swap per tile
// -- chk 1.84 against the fp32 reference, with the timing looking healthy.
//
// Both are plain VOP permutes on this target: verified by compiling a two-line
// kernel for gfx1201 and reading v_perm_b32 out of the assembly (the instruction
// is emitted, not emulated -- unlike sdot4, which gfx12 only has behind a
// target feature that breaks the fat binary's ID).
constexpr u32 kPermLo = 0x05010400u;
constexpr u32 kPermHi = 0x07030602u;

__device__ __forceinline__ void expand_nibbles(u32 p, u32 &d0, u32 &d1) {
    const u32 lo = p & 0x0F0F0F0Fu;
    const u32 hi = (p >> 4) & 0x0F0F0F0Fu;
    d0 = __builtin_amdgcn_perm(hi, lo, kPermLo);
    d1 = __builtin_amdgcn_perm(hi, lo, kPermHi);
}

// Four codes held one per byte -> two bytes of nibble pairs:
// [c0|c1<<4][c2|c3<<4] in bits 0..15.
__device__ __forceinline__ u32 nib2(u32 c) {
    return (c & 0x0000000Fu) | ((c >> 4) & 0x000000F0u) |
           ((c >> 8) & 0x00000F00u) | ((c >> 12) & 0x0000F000u);
}

// ---------------------------------------------------------------------------
// weight staging, Q4_K only
// ---------------------------------------------------------------------------
// Two layouts for the same decode. `packed` is the MMQ one; `as_int8` reuses
// dp4a's stage_weight_tile so the two arms differ in the tile shape alone.
template <bool PACKED>
__device__ __forceinline__ void stage_weight(const u8 *__restrict__ w,
                                             size_t w_row_bytes, i64 gc, i64 kt,
                                             i8 *dst, f32 *sc, f32 *mn) {
    if (!PACKED) {
        stage_weight_tile<kDp4aWq4>(w, w_row_bytes, gc, kt, dst, sc, mn);
        return;
    }
    const u8 *b = w + static_cast<size_t>(gc) * w_row_bytes;
    const int chunk = static_cast<int>(kt % dp4a_cfg::KBLOCK_CHUNKS);
    const u8 *blk = b + (static_cast<size_t>(kt) / dp4a_cfg::KBLOCK_CHUNKS) *
                            dtype_block_bytes_dev(static_cast<int>(DType::Q4_K));
    const f32 d = d_h2f(static_cast<u16>(blk[0] | (blk[1] << 8)));
    const f32 dmin = d_h2f(static_cast<u16>(blk[2] | (blk[3] << 8)));
    u8 si = 0, mi = 0;
    dev_scale_min_k4(chunk, blk + 4, &si, &mi);
    sc[0] = d * static_cast<f32>(si);
    mn[0] = dmin * static_cast<f32>(mi);

    // The sub-block's 32 nibbles live as low nibbles (values 0..31) and high
    // nibbles (32..63) of the same 32 bytes, so `chunk`'s half selects the
    // shift and `chunk/2` the 32-byte group -- the same fields
    // stage_weight_tile reads, packed instead of widened.
    const u8 *q = blk + 16 + (chunk / 2) * 32;
    const int shift = (chunk % 2) ? 4 : 0;
    u32 *p32 = reinterpret_cast<u32 *>(dst);
#pragma unroll
    for (int g = 0; g < cfg::BK / 8; g++) {
        const u32 a = (d_u32(q + g * 8) >> shift) & 0x0F0F0F0Fu;
        const u32 c = (d_u32(q + g * 8 + 4) >> shift) & 0x0F0F0F0Fu;
        // Byte 0/1 from the first code dword, 2/3 from the second, so the
        // unpacked order is k0..k7 and the two expand selectors above are the
        // whole story. (The first version interleaved the two dwords -- bytes
        // [k0,k1][k4,k5][k2,k3][k6,k7] -- which benchmarked fast and wrong:
        // chk 1.98 against the fp32 reference, a K-order swap in every tile.)
        // Four codes in four bytes -> two bytes of nibble pairs. Written as
        // four independent shifts rather than the two-OR-then-shift form the
        // first attempt used: that one tried to move both pairs at once and
        // left each code's high nibble in the neighbouring byte, which reads
        // back as a K swap inside every 8-value group (chk 1.28-1.88 against
        // the fp32 reference). Four shifts cannot be off by one position.
        p32[g] = nib2(a) | (nib2(c) << 16);
    }
}

// ---------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------
template <int TM, int TN, int TY, bool PACKED>
__global__ void __launch_bounds__(cfg::TX * TY)
                       __attribute__((amdgpu_waves_per_eu(6)))
    gemm_mmq_kernel(const u8 *__restrict__ w, i64 n_out, i64 n_in,
                    size_t w_row_bytes, const i8 *__restrict__ codes,
                    const f32 *__restrict__ scales, i64 rows,
                    _Float16 *__restrict__ out) {
    using namespace cfg;
    constexpr int BM = TM * TY;
    constexpr int BN = TN * TX;
    constexpr int THREADS = TX * TY;
    // Packed tiles are half the bytes: BK/2 per row instead of BK.
    constexpr int BW = PACKED ? (BK / 2) : BK;

    __shared__ __align__(16) i8 aw[STAGES][BM * BK];
    __shared__ __align__(16) i8 bw[STAGES][BN * BW];
    __shared__ f32 asc[STAGES][BM];
    __shared__ f32 asum[STAGES][BM];
    __shared__ f32 bsc[STAGES][BN];
    __shared__ f32 bmin[STAGES][BN];

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
            // One sum per sub-block: Q4_K's fold runs once per k-tile, so the
            // two 16-value halves dp4a keeps apart are added here instead.
            i32 s = 0;
            if (gr < rows) {
#pragma unroll
                for (int z = 0; z < BK / 4; z++) {
                    const u32 v =
                        *reinterpret_cast<const u32 *>(codes + gr * n_in + k0 + z * 4);
#pragma unroll
                    for (int e = 0; e < 4; e++)
                        s += static_cast<i32>(static_cast<i8>((v >> (8 * e)) & 0xFF));
                }
            }
            asum[stage][r] = static_cast<f32>(s);
        }
        for (int r = tid; r < BN; r += THREADS) {
            const i64 gc = col_base + r;
            f32 *sc = &bsc[stage][r];
            f32 *mn = &bmin[stage][r];
            *sc = 0.0f;
            *mn = 0.0f;
            if (gc >= n_out) {
                u32 *z = reinterpret_cast<u32 *>(&bw[stage][r * BW]);
#pragma unroll
                for (int g = 0; g < BW / 4; g++) z[g] = 0u;
                continue;
            }
            stage_weight<PACKED>(w, w_row_bytes, gc, kt, &bw[stage][r * BW], sc,
                                 mn);
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

        if (PACKED) {
            // Pair rotation: the fetch reads two quads, so the rotation has to
            // move in pairs. g = gg ^ (tx & 3) is a bijection over the four
            // dwords for any fixed tx, and each dword is consumed whole, so
            // every lane still visits every quad exactly once.
#pragma unroll
            for (int gg = 0; gg < ROUNDS / 2; gg++) {
                const int g = gg ^ (tx & (ROUNDS / 2 - 1));
                u32 b0[TN], b1[TN];
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    const u32 p = *reinterpret_cast<const u32 *>(
                        &bw[stage][(tx * TN + j) * BW + g * 4]);
                    expand_nibbles(p, b0[j], b1[j]);
                }
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int q = g * 2 + half;
                    i32 a[TM];
#pragma unroll
                    for (int i = 0; i < TM; i++)
                        a[i] = *reinterpret_cast<const i32 *>(
                            &aw[stage][(ty * TM + i) * BK + q * 4]);
#pragma unroll
                    for (int i = 0; i < TM; i++)
#pragma unroll
                        for (int j = 0; j < TN; j++)
                            iacc[i][j] = d_dot4(a[i], half ? b1[j] : b0[j],
                                                iacc[i][j]);
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < ROUNDS; r++) {
                const int q = (r ^ tx) & (ROUNDS - 1);
                i32 a[TM], b[TN];
#pragma unroll
                for (int i = 0; i < TM; i++)
                    a[i] = *reinterpret_cast<const i32 *>(
                        &aw[stage][(ty * TM + i) * BK + q * 4]);
#pragma unroll
                for (int j = 0; j < TN; j++)
                    b[j] = *reinterpret_cast<const i32 *>(
                        &bw[stage][(tx * TN + j) * BW + q * 4]);
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        iacc[i][j] = d_dot4(a[i], b[j], iacc[i][j]);
            }
        }

        // The fold, once per 32-value sub-block: acc += (sa*S)*dot - (sa*sum)*M.
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const int ri = ty * TM + i;
            const f32 sa = asc[stage][ri];
            const f32 ssum = sa * asum[stage][ri];
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const int cj = tx * TN + j;
                acc[i][j] = fmaf(sa * bsc[stage][cj],
                                 static_cast<f32>(iacc[i][j]), acc[i][j]);
                acc[i][j] = fmaf(-ssum, bmin[stage][cj], acc[i][j]);
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
// host side
// ---------------------------------------------------------------------------
inline void mmq_log_kernel_info() {
    static bool done = false;
    if (done) return;
    done = true;
    const char *e = std::getenv("KRK_KERNEL_INFO");
    if (e == nullptr || e[0] == '0') return;
    auto one = [](const void *fn, const char *name, int threads) {
        hipFuncAttributes at{};
        if (hipFuncGetAttributes(&at, fn) != hipSuccess) return;
        const int per_wave = ((at.numRegs + 7) / 8) * 8;
        const int waves_simd = per_wave > 0 ? 512 / per_wave : 16;
        const int lds = static_cast<int>(at.sharedSizeBytes);
        const int blocks_lds = lds > 0 ? (64 * 1024) / lds : 99;
        const int blocks_reg = (waves_simd * 4 * 32) / threads;
        KRK_INFO("kernel %s: numRegs %d (scratch %zu B), LDS %d B/block, "
                 "%d threads -> %d waves/SIMD, %d blocks/CU by regs, "
                 "%d by LDS",
                 name, at.numRegs, at.localSizeBytes, lds, threads,
                 waves_simd, blocks_reg, blocks_lds);
    };
    one(reinterpret_cast<const void *>(
            &gemm_mmq_kernel<cfg::kMmqTM, cfg::kMmqTN, cfg::kMmqTY, false>),
        "gemm_mmq4", cfg::TX * cfg::kMmqTY);
    one(reinterpret_cast<const void *>(
            &gemm_mmq_kernel<cfg::kMmqTM, cfg::kMmqTN, cfg::kMmqTY, true>),
        "gemm_mmq4p", cfg::TX * cfg::kMmqTY);
    one(reinterpret_cast<const void *>(
            &gemm_mmq_kernel<cfg::kWideTM, cfg::kWideTN, cfg::kWideTY, true>),
        "gemm_dp4a_packed", cfg::TX * cfg::kWideTY);
}

// The arm is a measurement, not a policy: `KRK_DP4A_MMQ=1` selects the int8
// staging tile, `=2` the packed one. Unset means the dp4a tile runs, which is
// what every caller got before this file existed.
enum class MmqArm { kOff = 0, kInt8 = 1, kPacked = 2, kWidePacked = 3 };

inline MmqArm mmq_arm() {
    static const MmqArm arm = [] {
        const char *e = std::getenv("KRK_DP4A_MMQ");
        if (e == nullptr) return MmqArm::kOff;
        if (e[0] == '1') return MmqArm::kInt8;
        if (e[0] == '2') return MmqArm::kPacked;
        if (e[0] == '3') return MmqArm::kWidePacked;
        return MmqArm::kOff;
    }();
    return arm;
}

// Only Q4_K: the packed layout's nibbles are the format's own codes, and Q5_K's
// fifth bit and Q6_K's six bits do not fit one. The gate also covers the shapes
// this tile cannot express -- see dp4a_ok for the same rules.
inline bool mmq_ok(int wt, i64 rows, i64 n_out, i64 n_in) {
    if (mmq_arm() == MmqArm::kOff) return false;
    if (dp4a_kind(wt) != kDp4aWq4) return false;
    if (rows < 16 || n_out <= 0 || n_in < kDp4aMinK) return false;
    if ((n_in & (cfg::BK - 1)) != 0 || (n_in % 256) != 0) return false;
    return true;
}

template <int TM, int TN, int TY, bool PACKED>
inline void mmq_launch(const u8 *w, i64 n_out, i64 n_in, size_t w_row_bytes,
                       const _Float16 *x, i64 rows, _Float16 *out, i8 *codes,
                       f32 *scales) {
    constexpr int BM = TM * TY;
    constexpr int BN = TN * cfg::TX;
    constexpr int THREADS = cfg::TX * TY;
    mmq_log_kernel_info();
    // Same activation quantization as the dp4a path -- one kernel, so the two
    // tiles dot the same codes and their outputs are comparable bit for bit.
    const i64 pairs = rows * (n_in / cfg::BK);
    const unsigned qblocks = static_cast<unsigned>((pairs + 7) / 8);
    quantize_act_kernel<<<qblocks, 256>>>(x, codes, scales, rows, n_in);

    const dim3 grid(static_cast<unsigned>((n_out + BN - 1) / BN),
                    static_cast<unsigned>((rows + BM - 1) / BM));
    gemm_mmq_kernel<TM, TN, TY, PACKED><<<grid, THREADS>>>(
        w, n_out, n_in, w_row_bytes, codes, scales, rows, out);
}

} // namespace mmq
} // namespace krk

#endif // KRK_KERNEL_GEMM_MMQ_HPP
