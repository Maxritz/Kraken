// ============================================================================
//  gemm.hpp — the three matrix paths.
//
//    * gemv_kernel    — decode. M == 1. One warp per output row, eight
//                       rows per thread block, weights streamed once and
//                       dequantized into registers. This is the
//                       memory-latency-bound path that decides
//                       single-token tok/s, so it never round-trips through
//                       shared memory.
//    * gemm_wmma      — prefill. gfx11/gfx12 dense WMMA, 16x16x16 f16 with f32
//                       accumulation. The A fragment comes from the activation
//                       tile, the B fragment from the dequantized weight tile.
//    * gemm_simt      — gfx10 (no WMMA) and the portable fallback. Same tiling,
//                       packed-math compute stage (v_dot2_f32_f16 on RDNA2).
//
//  Layout choice: out[M, N] = X[M, K] * W[N, K]^T. Both X and the dequantized
//  W tile live in LDS as row-major [idx][k], which makes the WMMA A and B
//  fragment loads symmetric — the same `lane % 16` row selector for both.
// ============================================================================
#ifndef KRK_KERNEL_GEMM_HPP
#define KRK_KERNEL_GEMM_HPP

#include "dequant.hpp"

namespace krk {

// ---------------------------------------------------------------------------
// packed-math primitive
// ---------------------------------------------------------------------------

// The packed primitive itself (d_dot2 / d_dot2_pk, v_dot2_f32_f16) lives in
// krk_hip.hpp next to the other device helpers, because the decode attention
// kernel needs the same instruction in its register form.
//
// The SIMT tile loop picks its pair reduction with KRK_SIMT_USE_DOT2 rather
// than KRK_HAS_DOT2: the instruction is available on gfx12, but in this loop
// (16 packed dots per k-pair per thread) gfx1201 retires the scalar pair
// faster — gemm_simt rows=51 measured 68 us scalar vs 195 us packed, so only
// the families where the packed form wins select it. The latency-bound decode
// attention kernel is the opposite case and uses d_dot2_pk unconditionally.
#if defined(KRK_SIMT_USE_DOT2)
__device__ __forceinline__ f32 simt_dot2(const _Float16 *a, const _Float16 *b, f32 c) {
    return d_dot2(a, b, c);
}
#else
__device__ __forceinline__ f32 simt_dot2(const _Float16 *a, const _Float16 *b, f32 c) {
    return c + static_cast<f32>(a[0]) * static_cast<f32>(b[0]) +
           static_cast<f32>(a[1]) * static_cast<f32>(b[1]);
}
#endif

// ---------------------------------------------------------------------------
// GEMV — M == 1 (decode)
//
// One warp per output channel: lane l covers chunks l, l+32, ... of the row
// and the warp reduces with shuffles. Consecutive lanes read consecutive
// 32-value chunks, so a warp's weight loads coalesce into full cache lines
// and no shared-memory reduction or block barrier is needed. A 256-thread
// block therefore covers 8 rows per launch instead of one, which is what
// makes the short-K decode rows (q/k/v/o, gate/up, down) efficient: the old
// block-per-row form left 238 of 256 threads idle and paid a block-wide
// reduction for every single row.
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    gemv_kernel(const u8 *__restrict__ w, int wt, i64 n_out, i64 n_in,
                size_t w_row_bytes, const _Float16 *__restrict__ x,
                _Float16 *__restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const i64 row = static_cast<i64>(blockIdx.x) * (THREADS / 32) +
                    static_cast<i64>(threadIdx.x >> 5);
    if (row >= n_out) return;

    const int nch = static_cast<int>(n_in / 32);
    const u8 *wrow = w + static_cast<size_t>(row) * w_row_bytes;

    f32 part = 0.0f;
    for (int c = lane; c < nch; c += 32)
        part += dot_chunks(wt, wrow, x, c, c + 1);

    part = d_wave_reduce_sum(part);
    if (lane == 0) out[row] = static_cast<_Float16>(part);
}

// ---------------------------------------------------------------------------
// WMMA GEMM — prefill on gfx11 / gfx12
//
// Block tile  BM=32 (tokens) x BN=64 (channels), K tile BK=128.
// 256 threads = 8 wave32 wavefronts: 2 along M, 4 along N. Each wavefront owns
// one 16x16 accumulator tile.
// ---------------------------------------------------------------------------

#if defined(KRK_GFX11) || defined(KRK_GFX12)

struct wmma_cfg {
    static constexpr int BM = 32;
    static constexpr int BN = 64;
// BK sets how much weight staging one serial round covers. §2c measured the
// staged dequant as the dominant prefill cost and showed the cost is per
// *round*, not per byte: a block stages BN rows of BK values, and every round
// re-exposes the full weight-load latency, so the tile wants to be as deep as
// LDS allows. BK=128 makes all 256 threads stage exactly one 32-value chunk
// each (BK=64 fed only 128 of them) and halves the number of rounds: 5 tiles
// for K=576, 12 for 1536. LDS cost is (BM+BN)*(BK+LD_PAD)*2 B = 26 kB.
// (A BK=256 tile was measured at sustained clocks and reverted: neutral.
// The staging cost is proportional to total work — bytes + decode ALU —
// not to the round count; §2c showed it is latency-bound with only
// ~27 blocks for 32 CUs, and deeper K tiles do not raise that MLP.)
    static constexpr int BK = 128;
    static constexpr int THREADS = 256;
    static constexpr int WAVES_M = 2;
    static constexpr int WAVES_N = 4;
static constexpr int LD_PAD = 8; // half-element padding to break LDS bank conflicts
}; // struct wmma_cfg

// A half-width N tile (BN=32, 128 threads, grid doubled along N)
// was measured as the §2e occupancy probe and reverted: it was
// 32-145% slower on every prefill shape (down 71.1 vs 53.8 us
// staging, gate 47.5 vs 19.2, kv 46.5 vs 17.7). The staging
// chain is latency-bound, and halving the threads per block
// halves the loads in flight per round — the extra blocks buy
// back nothing. §2b had already shown the wall time is flat from
// 5 to 27 blocks: block count was never the constraint.

// RDNA3: 16 fp16 per lane, lanes 0-15 and 16-31 replicated.
// RDNA4: 8 fp16 per lane, the two lane groups split K in 4-value interleaves.
#if defined(KRK_GFX12)
constexpr int kFrag = 8;
#else
constexpr int kFrag = 16;
#endif

template <typename CFG>
__device__ __forceinline__ void wmma_load_a(const _Float16 *xs, int m_off, int k,
                                            int lane, _Float16 (&frag)[kFrag]) {
    const int row = m_off + (lane & 15);
#if defined(KRK_GFX12)
    const int koff = k + 8 * (lane >> 4);
#else
    const int koff = k;
#endif
#pragma unroll
    for (int e = 0; e < kFrag; e++) frag[e] = xs[row * (CFG::BK + CFG::LD_PAD) + koff + e];
}

template <typename CFG>
__device__ __forceinline__ void wmma_load_b(const _Float16 *ws, int n_off, int k,
                                            int lane, _Float16 (&frag)[kFrag]) {
    const int col = n_off + (lane & 15);
#if defined(KRK_GFX12)
    const int koff = k + 8 * (lane >> 4);
#else
    const int koff = k;
#endif
#pragma unroll
    for (int e = 0; e < kFrag; e++) frag[e] = ws[col * (CFG::BK + CFG::LD_PAD) + koff + e];
}

__device__ __forceinline__ void wmma_mma(v8float &acc, const _Float16 (&a)[kFrag],
                                         const _Float16 (&b)[kFrag]) {
#if defined(KRK_GFX12)
    const v8half av = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
    const v8half bv = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
#else
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
        reinterpret_cast<const v16half &>(a), reinterpret_cast<const v16half &>(b), acc);
#endif
}

// Maps one accumulator slot back to its (row, col) inside the 16x16 tile.
__device__ __forceinline__ void wmma_slot(int g, int lane, int *i, int *j) {
#if defined(KRK_GFX12)
    *i = (lane >> 4) * 8 + g; // contiguous 8-row blocks per lane group
    *j = lane & 15;
#else
    *i = g * 2 + (lane >> 4); // interleaved rows across the lane groups
    *j = lane & 15;
#endif
}

// ---------------------------------------------------------------------------
// Q4_K raw chunk — the bytes one 32-value chunk needs, held in registers.
//
// §2c measured the staged dequant as 55-97% of prefill GEMM time and the
// chain as latency-bound: each thread had exactly one chunk's worth of VMEM
// loads in flight per K-tile round, far below what the memory system needs
// to hide RDNA4's load latency. The fix is software pipelining — round kt+1's
// bytes are issued while round kt decodes — which needs the raw bytes in
// registers, so this is the Q4_K decoder split into a load half and a
// pure-ALU decode half. The decode is the device twin of the Q4_K branch of
// dequant_chunk<DType::Q4_K>, bit for bit.
// ---------------------------------------------------------------------------
struct Q4KRaw {
    u32 hdr;   // d (low 16 bits) | dmin (high 16 bits)
    u32 sc[3]; // the 12 scale bytes
    u32 p[8];  // this chunk's 32 packed nibbles
};

// One chunk of one 144-byte Q4_K block. `chunk` (0..7) picks the scale pair
// and the packed run, exactly as in dequant_chunk.
//
// The four-dword header (d | dmin | 12 scale bytes) and the chunk's 32 packed
// bytes all sit at 16-byte-aligned offsets inside the block, so when the row
// base and stride are 16-byte aligned the whole chunk is three 128-bit loads.
// `vec` is that alignment test, hoisted to the kernel prologue: it is uniform
// across the grid, so the branch is free, and the byte-wise path it falls
// back to produces the identical twenty dwords.
//
// Why it matters here: the staging stage is 90% of prefill GEMM time (§2c) and
// this is its only global traffic. d_u32() is four byte loads that LLVM cannot
// always prove mergeable, so the scalar form left up to 48 load instructions
// per thread per K round in flight against 3.
__device__ __forceinline__ void q4k_load_raw(const u8 *blk, int chunk, bool vec,
                                                   Q4KRaw &r) {
    const int g = chunk / 2;
    if (vec) {
        const uint4 h = *reinterpret_cast<const uint4 *>(blk);
        r.hdr = h.x;
        r.sc[0] = h.y;
        r.sc[1] = h.z;
        r.sc[2] = h.w;
        const u8 *packed = blk + 16 + g * 32;
        const uint4 p0 = *reinterpret_cast<const uint4 *>(packed);
        const uint4 p1 = *reinterpret_cast<const uint4 *>(packed + 16);
        r.p[0] = p0.x;
        r.p[1] = p0.y;
        r.p[2] = p0.z;
        r.p[3] = p0.w;
        r.p[4] = p1.x;
        r.p[5] = p1.y;
        r.p[6] = p1.z;
        r.p[7] = p1.w;
    } else {
        r.hdr = d_u32(blk);
        r.sc[0] = d_u32(blk + 4);
        r.sc[1] = d_u32(blk + 8);
        r.sc[2] = d_u32(blk + 12);
#pragma unroll
        for (int w = 0; w < 8; w++)
            r.p[w] = d_u32(blk + 16 + g * 32 + w * 4);
    }
}

// Scale byte i (0..11) of the 12-byte scale array, from registers.
__device__ __forceinline__ u8 q4k_scale_at(const u32 *sc, int i) {
    if (i < 4) return static_cast<u8>((sc[0] >> (8 * i)) & 0xFF);
    if (i < 8) return static_cast<u8>((sc[1] >> (8 * (i - 4))) & 0xFF);
    return static_cast<u8>((sc[2] >> (8 * (i - 8))) & 0xFF);
}

// Pure-ALU decode of a raw chunk into 32 fp16 values.
__device__ __forceinline__ void q4k_decode_raw(const Q4KRaw &r, int chunk,
                                                     _Float16 *y) {
    const f32 d = d_h2f(static_cast<u16>(r.hdr & 0xFFFF));
    const f32 dmin = d_h2f(static_cast<u16>(r.hdr >> 16));
    u8 si, mi;
    if (chunk < 4) {
        si = static_cast<u8>(q4k_scale_at(r.sc, chunk) & 63);
        mi = static_cast<u8>(q4k_scale_at(r.sc, chunk + 4) & 63);
    } else {
        si = static_cast<u8>((q4k_scale_at(r.sc, chunk + 4) & 0xF) |
                             ((q4k_scale_at(r.sc, chunk - 4) >> 6) << 4));
        mi = static_cast<u8>((q4k_scale_at(r.sc, chunk + 4) >> 4) |
                             ((q4k_scale_at(r.sc, chunk) >> 6) << 4));
    }
    const f32 dl = d * static_cast<f32>(si);
    const f32 mn = dmin * static_cast<f32>(mi);
    const int shift = (chunk & 1) == 0 ? 0 : 4;
#pragma unroll
    for (int w = 0; w < 8; w++) {
        const u32 pack = r.p[w];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int v = static_cast<int>((pack >> (8 * e + shift)) & 0xF);
            y[w * 4 + e] = static_cast<_Float16>(dl * static_cast<f32>(v) - mn);
        }
    }
}

// MODE exists for the probe's staging-vs-MMA split (probe_decode §2c): the
// engine always launches MODE 0. MODE 1 keeps the staging loops and the
// barriers but skips the WMMA, folding a cheap LDS read of the staged tiles
// into the accumulator so the staging cannot be dead-code eliminated. MODE 2
// skips the staging loops and keeps the WMMA, i.e. times the fragment-load +
// packed-math path on its own.
//
// DT selects a compile-time dtype for the staging pipeline: the Q4_K value
// enables the register-pipelined staging above (the engine's and the probe's
// only quantized format); anything else keeps the runtime-switch staging.
// Split-K: the K slice is picked by blockIdx.z, so a split GEMM
// is a single launch — grid = (N tiles, M tiles, splits) — and the
// slices need no per-launch rebinding. Every block covers the K
// range [blockIdx.z*slice_len, ...+slice_len) of every (row, col)
// and, when `partials` is set, accumulates into fp32
// partials[blockIdx.z*part_stride + row*n_out + col] instead of
// writing the fp16 output; the caller reduces the partials
// (gemm_reduce_kernel). n_split == 1 with null partials is the
// plain single-pass GEMM, bit-identical to the pre-split kernel:
// k_off == 0, k_n == n_in, and the K loop and the per-tile
// accumulation order are unchanged.
//
// The slices keep the total weight bytes constant while shortening
// each block's staging chain, which is the prefill bottleneck: §2b
// showed the wall time flat from 5 to 27 blocks (the per-block
// latency chain is the cost, not CU occupancy), so cutting the
// chain length per block is what buys the win.
//
// `pick_split` below is the shared decision the engine backend and
// the probe both take, so §2d measures exactly what the engine runs.
// The split decision, shared by the engine backend and the probe
// so §2d measures exactly what the engine runs.
//
// The cost model comes from §2b: while the block grid fits the
// device, wall time is the per-block K chain (staging rounds),
// not CU occupancy — and past the resident capacity it is
// waves x rounds. Two blocks fit per CU of the 64 kB LDS budget
// (26 kB each), so `resident` = 2 * cu_count blocks run at once.
// Splitting K multiplies the block count and divides the chain:
//
//   waves  = ceil(base_blocks * split / resident)
//   rounds = ceil(slice / bk), slice = round32(ceil(n_in / split))
//
// The split minimizing waves * rounds wins; ties keep the smaller
// split (fewer partials, no extra reduce launch), and a split
// must cut the modeled cost by a real margin — a quarter — before
// the reduce launch is worth it. Every slice but the last is
// exactly `slice` long, a multiple of 32 so the chunk addressing
// stays integral, keeps at least half a K tile of useful K
// (deeper padding is pure waste), and never empties the final
// split.
inline int pick_split(int base_blocks, i64 n_in, int bk,
                      int cu_count, i64 *slice_out = nullptr) {
    i64 best_slice = n_in;
    int best = 1;
    i64 best_cost = (n_in + bk - 1) / bk; // split 1: one wave
    if (base_blocks > 0 && base_blocks < 2 * cu_count &&
        n_in >= 2 * bk) {
        const i64 resident = 2 * static_cast<i64>(cu_count);
        for (int sp = 2; sp <= 8; sp++) {
            i64 slice = (n_in + sp - 1) / sp;
            slice = (slice + 31) / 32 * 32;
            if (slice < bk / 2) break;
            // A rounded slice that already covers n_in (sp-1)
            // times would empty the final split. slice shrinks
            // as sp grows, so a later sp can become valid
            // again — skip rather than stop.
            if (slice * (sp - 1) >= n_in) continue;
            const i64 waves =
                (static_cast<i64>(base_blocks) * sp + resident - 1) /
                resident;
            const i64 rounds = (slice + bk - 1) / bk;
            const i64 cost = waves * rounds;
            if (cost < best_cost && cost * 4 < best_cost * 3) {
                best_cost = cost;
                best = sp;
                best_slice = slice;
            }
        }
    }
    if (slice_out) *slice_out = best_slice;
    return best;
}

template <typename CFG, int MODE = 0, int DT = -1>
__global__ void __launch_bounds__(CFG::THREADS)
    gemm_wmma_kernel(const u8 *__restrict__ w, int wt, i64 n_out, i64 n_in,
                     size_t w_row_bytes, const _Float16 *__restrict__ x, i64 rows,
                     _Float16 *__restrict__ out, i64 n_split, i64 slice_len,
                     f32 *__restrict__ partials, i64 part_stride) {
    constexpr int BM = CFG::BM;
    constexpr int BN = CFG::BN;
    constexpr int BK = CFG::BK;
    constexpr int THREADS = CFG::THREADS;
    constexpr int WAVES_M = CFG::WAVES_M;
    constexpr int WAVES_N = CFG::WAVES_N;
    constexpr int LD_PAD = CFG::LD_PAD;
    __shared__ _Float16 xs[BM][BK + LD_PAD];
    __shared__ _Float16 ws[BN][BK + LD_PAD];

    // This block's K slice, from the z grid dimension (see the
    // Split-K note above). The last slice may be shorter than
    // slice_len; the staging bounds checks below handle that.
    const i64 k_off = static_cast<i64>(blockIdx.z) * slice_len;
    const i64 k_n = k_off + slice_len <= n_in
                        ? slice_len
                        : n_in - k_off;
    const i64 part_off = static_cast<i64>(blockIdx.z) * part_stride;

    const i64 row_base = static_cast<i64>(blockIdx.y) * BM;
    const i64 col_base = static_cast<i64>(blockIdx.x) * BN;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int wave_m = (wid % WAVES_M) * 16;
    const int wave_n = (wid / WAVES_M) * 16;

    const int block_bytes = dtype_block_bytes_dev(wt);
    const int cpb = dtype_chunks_per_block_dev(wt);

    // 128-bit weight loads need 16-byte-aligned rows. Q4_K rows are
    // 144-byte blocks, so the test is just the tensor base and the row
    // stride; both are true for the engine's own arena, and the fallback
    // keeps any other layout working (and bit-identical).
    const bool vec_w = (reinterpret_cast<uintptr_t>(w) & 15u) == 0 &&
                       (w_row_bytes & 15u) == 0;

    v8float acc = v8float{};
    _Float16 af[kFrag];
    _Float16 bf[kFrag];
    _Float16 tmp[32];

    // Q4_K pipeline state: the raw bytes of the chunk this thread stages
    // in the current K-tile round (cur) and in the next (nxt). With
    // BK=128 the weight tile holds exactly THREADS 32-value chunks, so
    // thread tid always stages chunk tid of the tile — one chunk per
    // thread per round.
    Q4KRaw q4k_cur{}, q4k_nxt{};
    bool q4k_cur_ok = false;
    if constexpr (DT == static_cast<int>(DType::Q4_K) && MODE != 2) {
        static_assert(BN * (BK / 32) == THREADS,
                      "pipelined staging: one 32-value chunk per thread");
        // Prologue: round 0's bytes. Chunk cchunk of round 0 lives in
        // block 0 of the row, which is always inside the allocation
        // once the row exists (any nonzero n_in allocates >= 1 block).
        const int st_r = tid / (BK / 32);
        const int st_cchunk = tid % (BK / 32);
        const i64 st_gc = col_base + st_r;
        // Round 0's chunk of this slice is global chunk
        // (k_off/32)+st_cchunk, which may sit in a block past the
        // row's last when k_off > 0: only prefetch when the whole
        // 144-byte block is inside the row's allocation.
        const int gchunk0 = static_cast<int>(k_off / 32) + st_cchunk;
        const int bidx0 = gchunk0 / cpb;
        if (st_gc < n_out &&
            static_cast<size_t>(bidx0) * block_bytes + block_bytes <=
                w_row_bytes) {
            const u8 *wrow = w + static_cast<size_t>(st_gc) * w_row_bytes;
            q4k_load_raw(wrow + static_cast<size_t>(bidx0) * block_bytes,
                         gchunk0 % cpb, vec_w, q4k_cur);
            q4k_cur_ok = true;
        }
    }

    const i64 k_tiles = (k_n + BK - 1) / BK;
    for (i64 kt = 0; kt < k_tiles; kt++) {
        const i64 k0 = kt * BK;

        if constexpr (MODE != 2) {
        // ---- stage the activation tile (BM x BK), zero past the row end ----
        for (int i = tid; i < BM * BK; i += THREADS) {
            const int r = i / BK;
            const int c = i % BK;
            const i64 gr = row_base + r;
            xs[r][c] = (gr < rows && k0 + c < k_n)
                           ? x[gr * n_in + k_off + k0 + c]
                           : static_cast<_Float16>(0.0f);
        }

        // ---- stage the dequantized weight tile (BN x BK) ----
        if constexpr (DT == static_cast<int>(DType::Q4_K)) {
            // Software-pipelined Q4_K staging: cur's decode is pure ALU
            // while nxt's twelve dword loads are in flight.
            const int st_r = tid / (BK / 32);
            const int st_cchunk = tid % (BK / 32);
            const i64 st_gc = col_base + st_r;
            const int gchunk =
                static_cast<int>((k_off + k0) / 32) + st_cchunk;

            // Prefetch round kt+1. A chunk is only readable when the
            // whole 144-byte block holding it sits inside the row;
            // chunks past n_in are zero-filled below, never decoded.
            bool nxt_ok = false;
            if (kt + 1 < k_tiles && st_gc < n_out) {
                const int gchunk_n =
                    static_cast<int>((k_off + k0 + BK) / 32) + st_cchunk;
                const int bidx_n = gchunk_n / cpb;
                if (static_cast<size_t>(bidx_n) * block_bytes + block_bytes <=
                    w_row_bytes) {
                    const u8 *wrow =
                        w + static_cast<size_t>(st_gc) * w_row_bytes;
                    q4k_load_raw(wrow + static_cast<size_t>(bidx_n) * block_bytes,
                                 gchunk_n % cpb, vec_w, q4k_nxt);
                    nxt_ok = true;
                }
            }

            const bool cur_ok = q4k_cur_ok && (k0 + st_cchunk * 32 < k_n);
            if (cur_ok) {
                q4k_decode_raw(q4k_cur, gchunk % cpb, tmp);
                // Paired dword stores: the staged values come out of the
                // decoder as register-resident pairs (both the row stride
                // BK+8 and the pair offset are even, so the stores stay
                // aligned).
#pragma unroll
                for (int z = 0; z < 32; z += 2)
                    *reinterpret_cast<u32 *>(&ws[st_r][st_cchunk * 32 + z]) =
                        *reinterpret_cast<const u32 *>(&tmp[z]);
            } else {
#pragma unroll
                for (int z = 0; z < 32; z++)
                    ws[st_r][st_cchunk * 32 + z] = static_cast<_Float16>(0.0f);
            }
            q4k_cur = q4k_nxt;
            q4k_cur_ok = nxt_ok;
        } else {
        for (int i = tid; i < BN * (BK / 32); i += THREADS) {
            const int r = i / (BK / 32);
            const int cchunk = i % (BK / 32);
            const i64 gc = col_base + r;
            if (gc >= n_out) {
#pragma unroll
                for (int z = 0; z < 32; z++) ws[r][cchunk * 32 + z] = static_cast<_Float16>(0.0f);
                continue;
            }
            const u8 *wrow = w + static_cast<size_t>(gc) * w_row_bytes;
            const int gchunk =
                static_cast<int>((k_off + k0) / 32) + cchunk;
            if (k0 + cchunk * 32 >= k_n) {
#pragma unroll
                for (int z = 0; z < 32; z++) ws[r][cchunk * 32 + z] = static_cast<_Float16>(0.0f);
                continue;
            }
            const u8 *blk = wrow + static_cast<size_t>(gchunk / cpb) * block_bytes;
            dequant_chunk_dev(wt, blk, gchunk % cpb, tmp);
            // Two halves per store: the staged values come out of the decoder
            // as register-resident pairs, so narrowing the LDS traffic halves
            // the store instructions the staging stage costs (§2c measured it
            // as the dominant prefill cost). Both the row stride (BK+8 halves)
            // and the pair offset are even, so the dword stores stay aligned.
#pragma unroll
            for (int z = 0; z < 32; z += 2)
                *reinterpret_cast<u32 *>(&ws[r][cchunk * 32 + z]) =
                    *reinterpret_cast<const u32 *>(&tmp[z]);
        }
        } // DT generic
        } // MODE != 2
        __syncthreads();

        if constexpr (MODE == 1) {
            // Staging-only timing: touch the tiles so nothing above can be
            // eliminated, but skip the fragment loads and the WMMA itself.
            acc[0] += static_cast<f32>(xs[tid & (BM - 1)][0]) +
                      static_cast<f32>(ws[tid & (BN - 1)][0]);
        } else {
            // ---- WMMA over the K tile ----
#pragma unroll 1
            for (int k = 0; k < BK; k += 16) {
                wmma_load_a<CFG>(&xs[0][0], wave_m, k, lane, af);
                wmma_load_b<CFG>(&ws[0][0], wave_n, k, lane, bf);
                wmma_mma(acc, af, bf);
            }
        }
        __syncthreads();
    }

    // ---- scatter the accumulator ----
#pragma unroll
    for (int g = 0; g < 8; g++) {
        int i, j;
        wmma_slot(g, lane, &i, &j);
        const i64 gr = row_base + wave_m + i;
        const i64 gc = col_base + wave_n + j;
        if (gr < rows && gc < n_out) {
            if constexpr (MODE == 0) {
                if (partials)
                    partials[part_off + gr * n_out + gc] = acc[g];
                else
                    out[gr * n_out + gc] = static_cast<_Float16>(acc[g]);
            }
        }
    }
}

// Split-K tail: one fp32 partial per (split, row, col), reduced to
// the fp16 output. The reduction is a plain serial sum over the
// splits — the partials are small (rows*n_out*S*4 B) and the kernel
// is memory-bound on them, so no tiling.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    gemm_reduce_kernel(const f32 *__restrict__ partials, i64 splits,
                       i64 part_stride, i64 total,
                       _Float16 *__restrict__ out) {
    const i64 idx = static_cast<i64>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    f32 acc = 0.0f;
    for (i64 s = 0; s < splits; s++)
        acc += partials[s * part_stride + idx];
    out[idx] = static_cast<_Float16>(acc);
}

#endif // KRK_GFX11 || KRK_GFX12

// ---------------------------------------------------------------------------
// SIMT GEMM — gfx10 and the portable fallback
//
// 64x64 block tile, 32-deep K tile, 16x16 threads each owning a 4x4 register
// tile. The compute stage uses v_dot2_f32_f16 where the target has it.
// ---------------------------------------------------------------------------

namespace simt_cfg {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int TX = BN / TN; // 16
constexpr int TY = BM / TM; // 16
constexpr int THREADS = TX * TY;
} // namespace simt_cfg

__global__ void __launch_bounds__(simt_cfg::THREADS)
    gemm_simt_kernel(const u8 *__restrict__ w, int wt, i64 n_out, i64 n_in,
                     size_t w_row_bytes, const _Float16 *__restrict__ x, i64 rows,
                     _Float16 *__restrict__ out) {
    using namespace simt_cfg;
    __shared__ _Float16 xs[BM][BK + 8];
    __shared__ _Float16 ws[BN][BK + 8];

    const i64 row_base = static_cast<i64>(blockIdx.y) * BM;
    const i64 col_base = static_cast<i64>(blockIdx.x) * BN;
    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid % TX;
    const int ty = tid / TX;

    f32 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    const int block_bytes = dtype_block_bytes_dev(wt);
    const int cpb = dtype_chunks_per_block_dev(wt);
    _Float16 tmp[32];

    const i64 k_tiles = (n_in + BK - 1) / BK;
    for (i64 kt = 0; kt < k_tiles; kt++) {
        const i64 k0 = kt * BK;

        for (int i = tid; i < BM * BK; i += THREADS) {
            const int r = i / BK;
            const int c = i % BK;
            const i64 gr = row_base + r;
            xs[r][c] = (gr < rows && k0 + c < n_in)
                           ? x[gr * n_in + k0 + c]
                           : static_cast<_Float16>(0.0f);
        }
        for (int i = tid; i < BN; i += THREADS) {
            const i64 gc = col_base + i;
            if (gc >= n_out || k0 >= n_in) {
#pragma unroll
                for (int z = 0; z < BK; z++) ws[i][z] = static_cast<_Float16>(0.0f);
                continue;
            }
            const u8 *wrow = w + static_cast<size_t>(gc) * w_row_bytes;
            const int gchunk = static_cast<int>(k0 / 32);
            const u8 *blk = wrow + static_cast<size_t>(gchunk / cpb) * block_bytes;
            dequant_chunk_dev(wt, blk, gchunk % cpb, tmp);
#pragma unroll
            for (int z = 0; z < BK; z += 2)
                *reinterpret_cast<u32 *>(&ws[i][z]) =
                    *reinterpret_cast<const u32 *>(&tmp[z]);
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k += 2) {
            _Float16 a[TM];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = xs[ty * TM + i][k];
            _Float16 b[TN];
#pragma unroll
            for (int j = 0; j < TN; j++) b[j] = ws[tx * TN + j][k];
#pragma unroll
            for (int i = 0; i < TM; i++) {
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    // 2-wide dot: consume the k and k+1 halves in one op
                    const _Float16 ap[2] = {a[i], xs[ty * TM + i][k + 1]};
                    const _Float16 bp[2] = {b[j], ws[tx * TN + j][k + 1]};
                    acc[i][j] = simt_dot2(ap, bp, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const i64 gr = row_base + ty * TM + i;
            const i64 gc = col_base + tx * TN + j;
            if (gr < rows && gc < n_out) out[gr * n_out + gc] = static_cast<_Float16>(acc[i][j]);
        }
    }
}

} // namespace krk

#endif // KRK_KERNEL_GEMM_HPP
