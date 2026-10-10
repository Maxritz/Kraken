// ============================================================================
//  attention.hpp — embedding gather, KV append, tiled flash attention, and
//  the fused decode chain (RoPE + KV append + attention in one launch).
//
//  The attention kernel is a genuine online-softmax (flash) formulation: it
//  walks the KV cache in tiles of TILE_K keys, keeps a running max and running
//  normalizer, and rescales the output accumulator instead of ever
//  materializing the score vector. That keeps shared memory bounded and, more
//  importantly, keeps the KV cache streaming at bandwidth instead of bouncing
//  through a scores buffer.
//
//  One block per (query head, query token). Each thread owns one output
//  channel; the per-key dot product is computed by the thread that owns that
//  key, reading the head vector from shared memory.
//
//  The decode side adds the fused non-GEMM chain (§2i of probe_decode):
//  RoPE + KV append + attention from ONE launch. The separate chain costs
//  four launches and a global-memory round trip of the q/k rows between
//  them (rope rewrites the q/k buffers in place, append copies them into
//  the cache, attention reads the cache back); the fused kernel applies
//  RoPE to q in registers, ropes k straight into the cache slot, copies v
//  next to it, and attends over the cache including the fresh row.
// ============================================================================
#ifndef KRK_KERNEL_ATTENTION_HPP
#define KRK_KERNEL_ATTENTION_HPP

#include "normalize.hpp"
#include "dequant.hpp" // embed_kernel's device dequant helpers

namespace krk {

constexpr int kAttnBlock = 128; // threads per attention block
constexpr int kMaxOv = 2;       // output channels owned per thread (hd <= 256)

// ---------------------------------------------------------------------------
// embedding gather
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    embed_kernel(_Float16 *__restrict__ out, const u8 *__restrict__ tok_embd, int tt,
                 i64 n_vocab, i64 n_embd, size_t row_bytes,
                 const i32 *__restrict__ tokens, i64 n_tok) {
    const i64 t = blockIdx.x;
    if (t >= n_tok) return;
    _Float16 *dst = out + t * n_embd;

    const i32 id = tokens[t];
    if (id < 0 || id >= n_vocab) {
        for (i64 i = threadIdx.x; i < n_embd; i += THREADS)
            dst[i] = static_cast<_Float16>(0.0f);
        return;
    }

    const u8 *src = tok_embd + static_cast<size_t>(id) * row_bytes;
    const int block_bytes = dtype_block_bytes_dev(tt);
    const int cpb = dtype_chunks_per_block_dev(tt);
    const int nch = static_cast<int>(n_embd / 32);
    _Float16 tmp[32];
    for (int c = threadIdx.x; c < nch; c += THREADS) {
        dequant_chunk_dev(tt, src + static_cast<size_t>(c / cpb) * block_bytes,
                          c % cpb, tmp);
#pragma unroll
        for (int z = 0; z < 32; z++) dst[c * 32 + z] = tmp[z];
    }
}

// ---------------------------------------------------------------------------
// KV append: copies this step's K/V rows into the cache at pos0..
// ---------------------------------------------------------------------------

__global__ void kv_append_kernel(_Float16 *__restrict__ kc, _Float16 *__restrict__ vc,
                                 const _Float16 *__restrict__ k,
                                 const _Float16 *__restrict__ v,
                                 const i64 *__restrict__ pos0,
                                 i64 n_tok, i64 pos_stride) {
    const i64 total = n_tok * pos_stride;
    // Device-side position: see rope_kernel. A graph replay of a
    // decode step rewrites the same cache slot, which is idempotent
    // for timing.
    const i64 p0 = *pos0;
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const i64 t = i / pos_stride;
    const i64 d = i % pos_stride;
    kc[(p0 + t) * pos_stride + d] = k[i];
    vc[(p0 + t) * pos_stride + d] = v[i];
}

// ---------------------------------------------------------------------------
// flash attention
//
// Shared memory layout (bytes):
//   [0, TILE_K*4)                scores / probabilities (f32)
//   then k_sh[TILE_K * hd], v_sh[TILE_K * hd], q_sh[hd]  (all fp16)
// The host sizes the dynamic allocation; see attn_smem_bytes().
// ---------------------------------------------------------------------------

// Pure arithmetic, so it is callable from both the host launch helper
// (which sizes the dynamic shared-memory allocation) and device code.
__host__ __device__ __forceinline__ size_t attn_smem_bytes(int tile_k, i64 hd) {
    const size_t scores = static_cast<size_t>(tile_k) * sizeof(f32);
    const size_t aligned = (scores + 15u) & ~static_cast<size_t>(15u);
    return aligned + sizeof(_Float16) * static_cast<size_t>(tile_k) *
                         static_cast<size_t>(hd) * 2u +
           sizeof(_Float16) * static_cast<size_t>(hd);
}

__global__ void __launch_bounds__(kAttnBlock)
    attention_kernel(_Float16 *__restrict__ out, const _Float16 *__restrict__ q,
                     const _Float16 *__restrict__ kcache,
                     const _Float16 *__restrict__ vcache, i64 n_head, i64 n_kv,
                     i64 hd, i64 n_tok, i64 pos0, i64 pos_stride, i64 causal,
                     i64 window, f32 scale, int tile_k) {
    extern __shared__ u8 smem[];
    f32 *s_sh = reinterpret_cast<f32 *>(smem);
    const size_t sbytes = (static_cast<size_t>(tile_k) * sizeof(f32) + 15u) &
                          ~static_cast<size_t>(15u);
    _Float16 *k_sh = reinterpret_cast<_Float16 *>(smem + sbytes);
    _Float16 *v_sh = k_sh + static_cast<i64>(tile_k) * hd;
    _Float16 *q_sh = v_sh + static_cast<i64>(tile_k) * hd;

    const i64 t = blockIdx.y;
    const i64 h = blockIdx.x;
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    const i64 pos = pos0 + t;
    const i64 n_keys = causal ? pos + 1 : pos0 + n_tok;
    if (n_keys <= 0) return;
    // Sliding window (laguna's hybrid layers): the oldest key this query can
    // see. The tile loop starts at the tile *containing* the bound, not at the
    // bound itself, so the first tile always holds at least one unmasked key
    // and the online softmax's running max stays finite -- a tile in which
    // every key is masked would contribute exp(0) = 1 to the denominator.
    const i64 k_lo = window > 0 && window < n_keys ? n_keys - window : 0;

    const int tid = static_cast<int>(threadIdx.x);
    const _Float16 *qrow = q + (t * n_head + h) * hd;
    const _Float16 *krow = kcache + hkv * hd;
    const _Float16 *vrow = vcache + hkv * hd;

    for (i64 i = tid; i < hd; i += kAttnBlock) q_sh[i] = qrow[i];
    __syncthreads();

    f32 m = -1.0e30f;
    f32 l = 0.0f;
    f32 acc[kMaxOv];
#pragma unroll
    for (int i = 0; i < kMaxOv; i++) acc[i] = 0.0f;

    __shared__ f32 bcast[2]; // [0] = new running max, [1] = rescale factor

    for (i64 k0 = (k_lo / tile_k) * tile_k; k0 < n_keys; k0 += tile_k) {
        const int tile = static_cast<int>(
            (n_keys - k0) < tile_k ? (n_keys - k0) : tile_k);

        // stage a tile of K and V
        const i64 tile_elems = static_cast<i64>(tile) * hd;
        for (i64 i = tid; i < tile_elems; i += kAttnBlock) {
            const i64 j = i / hd;
            const i64 d = i % hd;
            k_sh[i] = krow[(k0 + j) * pos_stride + d];
        }
        for (i64 i = tid; i < tile_elems; i += kAttnBlock) {
            const i64 j = i / hd;
            const i64 d = i % hd;
            v_sh[i] = vrow[(k0 + j) * pos_stride + d];
        }
        __syncthreads();

        // one key per thread: full head-width dot product. Keys below the
        // window bound stay in the tile -- the staging is uniform across the
        // block -- but never reach the max or the denominator.
        if (tid < tile) {
            if (k0 + static_cast<i64>(tid) < k_lo) {
                s_sh[tid] = -1.0e30f;
            } else {
                f32 dot = 0.0f;
                const _Float16 *kj = k_sh + static_cast<i64>(tid) * hd;
#pragma unroll 8
                for (i64 d = 0; d < hd; d++)
                    dot += static_cast<f32>(q_sh[d]) * static_cast<f32>(kj[d]);
                s_sh[tid] = dot * scale;
            }
        }
        __syncthreads();

        if (tid == 0) {
            f32 tmax = -1.0e30f;
            for (int j = 0; j < tile; j++) tmax = fmaxf(tmax, s_sh[j]);
            const f32 mnew = fmaxf(m, tmax);
            bcast[0] = mnew;
            bcast[1] = __expf(m - mnew);
            m = mnew;
        }
        __syncthreads();

        const f32 mnew = bcast[0];
        const f32 alpha = bcast[1];
        for (int j = tid; j < tile; j += kAttnBlock) s_sh[j] = __expf(s_sh[j] - mnew);
        __syncthreads();

        f32 part = 0.0f;
        for (int j = tid; j < tile; j += kAttnBlock) part += s_sh[j];
        const f32 tilesum = block_reduce_sum<kAttnBlock>(part);
        l = l * alpha + tilesum;

        const int nov = static_cast<int>((hd + kAttnBlock - 1) / kAttnBlock);
        for (int o = 0; o < nov && o < kMaxOv; o++) {
            const i64 d = tid + static_cast<i64>(o) * kAttnBlock;
            f32 a = 0.0f;
            if (d < hd) {
                for (int j = 0; j < tile; j++)
                    a += s_sh[j] * static_cast<f32>(v_sh[static_cast<i64>(j) * hd + d]);
            }
            acc[o] = acc[o] * alpha + a;
        }
        __syncthreads();
    }

    const f32 inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    _Float16 *orow = out + (t * n_head + h) * hd;
    const int nov = static_cast<int>((hd + kAttnBlock - 1) / kAttnBlock);
    for (int o = 0; o < nov && o < kMaxOv; o++) {
        const i64 d = tid + static_cast<i64>(o) * kAttnBlock;
        if (d < hd) orow[d] = static_cast<_Float16>(acc[o] * inv_l);
    }
}

// LDS for QT queries: scores [QT][tile], q [QT][hd], then the K and V tiles.
//
// ONE owner for this arithmetic. The kernel used to place its pointers with a
// copy of this expression and the host sized the launch with another copy, and
// the two disagreed by a factor of two -- so the tier that could fit a 64-key
// tile was launched with a 32-key one on every card, for no reason anyone could
// see in either site. Both now call these two functions.
//
// This must stay ABOVE attention_qtile_kernel: the kernel calls it with an
// explicit template argument, and a dependent call to a function that is only
// declared later is not found by argument-dependent lookup. clang rejects that
// on Linux ("neither visible in the template definition nor found by
// argument-dependent lookup"); the Windows toolchain's two-phase handling
// happened to accept it, which is how it passed a green gate here.
template <int QT>
__host__ __device__ __forceinline__ size_t attn_qtile_scores_bytes(int tile_k) {
    const size_t scores = static_cast<size_t>(QT) * static_cast<size_t>(tile_k) *
                          sizeof(f32);
    return (scores + 15u) & ~static_cast<size_t>(15u);
}

template <int QT>
__host__ __device__ __forceinline__ size_t attn_qtile_smem_bytes(int tile_k, i64 hd) {
    return attn_qtile_scores_bytes<QT>(tile_k) +
           sizeof(_Float16) *
               (static_cast<size_t>(QT) + 2u * static_cast<size_t>(tile_k)) *
               static_cast<size_t>(hd);
}

// ---------------------------------------------------------------------------
// query-tiled flash prefill
//
// The tiled kernel above launches one block per (head, token) and re-stages the
// entire K and V history in every one of them. Measured on Qwen3.5-0.8B at a
// 2048-token chunk: 2048 blocks x 2048 keys x 2 x 256 x 2 B = 4.29 GB of KV
// staging per layer, against 4.2 MB of unique KV -- 1024x redundancy -- and
// 96.2 ms per call for score math that costs 353 us at the card's 48.7
// TFLOP/s. It was also only using 32 of its 128 threads for the scores, one
// key each doing a 256-long serial FMA.
//
// This kernel gives each block QT consecutive query tokens, so the K/V tile is
// staged once and amortized across all of them, and every thread carries a
// query's scores. Same online-softmax formulation, so it is a reformulation:
// only the summation order changes.
//
// Thread map (kAttnBlock = 128 = 4 waves, QT queries, tile_k keys):
//   * one wave owns QPW = QT/4 queries; lane L owns key L of the tile and
//     computes that query's dot product over the head, then the wave reduces
//     the max and the sum;
//   * every thread owns output channels {tid, tid+kAttnBlock} for *every*
//     query, so the V accumulation is also fully parallel over threads.
// Accumulators are kMaxOv * QT floats per thread (2*8 = 16 for QT=8), which is
// what bounds QT here rather than LDS.
// ---------------------------------------------------------------------------
template <int QT>
__global__ void __launch_bounds__(kAttnBlock)
    attention_qtile_kernel(_Float16 *__restrict__ out, const _Float16 *__restrict__ q,
                           const _Float16 *__restrict__ kcache,
                           const _Float16 *__restrict__ vcache, i64 n_head, i64 n_kv,
                           i64 hd, i64 n_tok, i64 pos0, i64 pos_stride, i64 causal,
                           f32 scale, int tile_k) {
    constexpr int kWaves = kAttnBlock / kWaveSize;
    constexpr int kQPW = QT / kWaves; // queries per wave
    extern __shared__ u8 smem[];
    f32 *s_sh = reinterpret_cast<f32 *>(smem); // [QT][tile_k], read-only after staging
    const size_t sbytes = attn_qtile_scores_bytes<QT>(tile_k);
    _Float16 *q_sh = reinterpret_cast<_Float16 *>(smem + sbytes); // [QT][hd]
    _Float16 *k_sh = q_sh + static_cast<i64>(QT) * hd;           // [tile_k][hd]
    _Float16 *v_sh = k_sh + static_cast<i64>(tile_k) * hd;

    const i64 t0 = static_cast<i64>(blockIdx.y) * QT;
    const i64 h = blockIdx.x;
    if (t0 >= n_tok) return;
    const int nq = static_cast<int>(min(static_cast<i64>(QT), n_tok - t0));
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    // Causal: the newest query in the block sees the most keys, so the shared
    // bound is that one. Queries inside the tile mask themselves off via
    // `key_ok`, which keeps the K/V staging identical for every query in it.
    const i64 tmax = pos0 + t0 + nq - 1;
    const i64 n_keys = (causal ? tmax + 1 : pos0 + n_tok);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wave = tid >> 5;

    // Q rows are strided by n_head*hd, NOT contiguous: query qi of this block
    // lives at ((t0+qi)*n_head + h)*hd. Staging nq*hd elements as one run walks
    // straight into the next head's row, so every query past the first scored
    // against the wrong head's vector. That is what made the prefill output
    // read as fluent garbage rather than fail outright.
    for (i64 i = tid; i < static_cast<i64>(nq) * hd; i += kAttnBlock) {
        const i64 qi = i / hd, d = i % hd;
        q_sh[i] = q[((t0 + qi) * n_head + h) * hd + d];
    }

    const _Float16 *kb = kcache + hkv * hd;
    const _Float16 *vb = vcache + hkv * hd;

    f32 acc[QT][kMaxOv];
    f32 mq[QT], lq[QT];
#pragma unroll
    for (int r = 0; r < QT; r++) {
        mq[r] = -1.0e30f;
        lq[r] = 0.0f;
#pragma unroll
        for (int o = 0; o < kMaxOv; o++) acc[r][o] = 0.0f;
    }

    for (i64 k0 = 0; k0 < n_keys; k0 += tile_k) {
        const int tile = static_cast<int>(
            (n_keys - k0) < tile_k ? (n_keys - k0) : tile_k);
        const i64 tile_elems = static_cast<i64>(tile) * hd;
        for (i64 i = tid; i < tile_elems; i += kAttnBlock) {
            const i64 j = i / hd, d = i % hd;
            k_sh[i] = kb[(k0 + j) * pos_stride + d];
        }
        for (i64 i = tid; i < tile_elems; i += kAttnBlock) {
            const i64 j = i / hd, d = i % hd;
            v_sh[i] = vb[(k0 + j) * pos_stride + d];
        }
        __syncthreads();

        // Scores: wave owns kQPW queries, lane owns key `lane` of the tile.
        // Only key `lane` is computed, so the wave needs tile_k == kWaveSize
        // keys per pass; larger tiles loop.
        for (int jb = 0; jb < tile; jb += kWaveSize) {
            const int j = jb + lane;
            const bool live = j < tile;
#pragma unroll
            for (int r = 0; r < kQPW; r++) {
                const int qi = wave * kQPW + r;
                if (!live || qi >= nq) continue;
                const _Float16 *qj = q_sh + static_cast<i64>(qi) * hd;
                const _Float16 *kj = k_sh + static_cast<i64>(j) * hd;
                f32 dot = 0.0f;
#pragma unroll 8
                for (i64 d = 0; d < hd; d++)
                    dot += static_cast<f32>(qj[d]) * static_cast<f32>(kj[d]);
                // Each lane writes its OWN key's score. Reducing across the
                // wave here would collapse 32 different keys into one value;
                // the per-query max and sum are taken from LDS below.
                s_sh[static_cast<i64>(qi) * tile_k + j] = dot * scale;
            }
        }
        __syncthreads();

        // Online softmax per query, then the V accumulation. Both are fully
        // parallel over the block: the reductions are over LDS.
#pragma unroll
        for (int qi = 0; qi < QT; qi++) {
            if (qi >= nq) continue;
            // Absolute position of this query. blockIdx.y indexes the chunk,
            // but the KV cache rows it attends to are at absolute positions, so
            // the causal bound needs pos0 added -- the tiled kernel computes
            // `pos = pos0 + t` and the causal bound from that. Dropping pos0
            // silently lets every query after the first chunk attend to keys it
            // must not see.
            const i64 my = pos0 + t0 + qi;
            // A tile that starts past this query's bound holds no live keys.
            // Skipping the query entirely is what keeps alpha = exp(m - m) = 1;
            // letting it fall through with an empty tile gives tmax = -1e30 and
            // alpha = exp(0) = 1 only by luck, while acc = acc*alpha + 0 wipes
            // everything the earlier tiles contributed.
            if (k0 > my) continue;
            f32 tmax = -1.0e30f;
            for (int j = 0; j < tile; j++) {
                const i64 kk = k0 + j;
                if (causal && kk > my) break; // keys are contiguous: stop at the bound
                tmax = fmaxf(tmax, s_sh[static_cast<i64>(qi) * tile_k + j]);
            }
            const f32 mnew = fmaxf(mq[qi], tmax);
            const f32 alpha = __expf(mq[qi] - mnew);
            // The weights are NOT written back. The score row is shared: every
            // thread in the block reads it, so a thread that ran ahead of its
            // neighbours replaced the raw scores before they had read them, and
            // those neighbours then took max and exp over values that had
            // already been exponentiated once. The row stays read-only from the
            // staging pass on, and the accumulation loop below recomputes exp
            // from it -- the same value, from a source nobody can invalidate.
            f32 rowsum = 0.0f;
            for (int j = 0; j < tile; j++) {
                const i64 kk = k0 + j;
                if (causal && kk > my) break;
                rowsum += __expf(s_sh[static_cast<i64>(qi) * tile_k + j] - mnew);
            }
            lq[qi] = lq[qi] * alpha + rowsum;
            mq[qi] = mnew;
#pragma unroll
            for (int o = 0; o < kMaxOv; o++) {
                const i64 d = tid + static_cast<i64>(o) * kAttnBlock;
                if (d >= hd) continue;
                f32 a = 0.0f;
                for (int j = 0; j < tile; j++) {
                    const i64 kk = k0 + j;
                    if (causal && kk > my) break;
                    a += __expf(s_sh[static_cast<i64>(qi) * tile_k + j] - mnew) *
                         static_cast<f32>(v_sh[static_cast<i64>(j) * hd + d]);
                }
                acc[qi][o] = acc[qi][o] * alpha + a;
            }
        }
        __syncthreads();
    }

    const int nov = static_cast<int>((hd + kAttnBlock - 1) / kAttnBlock);
#pragma unroll
    for (int qi = 0; qi < QT; qi++) {
        if (qi >= nq) continue;
        const f32 inv = lq[qi] > 0.0f ? 1.0f / lq[qi] : 0.0f;
        for (int o = 0; o < nov && o < kMaxOv; o++) {
            const i64 d = tid + static_cast<i64>(o) * kAttnBlock;
            if (d < hd)
                out[((t0 + qi) * n_head + h) * hd + d] =
                    static_cast<_Float16>(acc[qi][o] * inv);
        }
    }
}

// ---------------------------------------------------------------------------
// flash attention — decode (n_tok == 1)
//
// The tiled kernel above parallelizes over output channels and hands each whole
// per-key dot product to a single lane. At decode time that leaves one block of
// ~9 blocks of work on a 32-CU device, so the step is latency-bound instead of
// bandwidth-bound (measured 42-50 us per layer at 100 keys, ~190 us per head at
// 1024 keys).
//
// This kernel turns the decomposition inside out. One block covers one query
// head's whole key range with kAttnDecWarps warps; inside a warp every lane owns
// VPT = hd/32 *consecutive* head elements, so an hd = 64 head is exactly one
// v_dot2_f32_f16 per lane per key (a single 128-byte coalesced K load per wave)
// and the score is a 5-step shuffle reduction. K and V stream straight from the
// cache — no LDS staging tile, no barrier per tile, and the per-lane online
// softmax runs entirely in registers. The per-warp (m, l, O) triples are merged
// once at the end, which is the standard flash-decoding combine.
//
// The math is the same online-softmax formulation as the tiled kernel; only the
// summation order changes (packed pair per lane + warp tree instead of one
// lane's serial FMA chain), so results differ by fp32 rounding only.
// ---------------------------------------------------------------------------

constexpr int kAttnDecWarps = 8;
constexpr int kAttnDecThreads = kAttnDecWarps * kWaveSize;

// One lane's slice of a head-wide dot product. VPT is even so every partial
// pairs up: lane L covers elements [L*VPT, (L+1)*VPT) of the head.
template <int VPT>
__device__ __forceinline__ f32 attn_lane_dot(const u32 *qp,
                                             const _Float16 *__restrict__ krow, int lane) {
    static_assert(VPT % 2 == 0, "decode attention lane slice must pair up");
    f32 s = 0.0f;
#pragma unroll
    for (int e = 0; e < VPT / 2; e++)
        s = d_dot2_pk(qp[e],
                      *reinterpret_cast<const u32 *>(krow + lane * VPT + 2 * e), s);
    return s;
}

template <int VPT>
__global__ void __launch_bounds__(kAttnDecThreads)
    attention_decode_kernel(_Float16 *__restrict__ out, const _Float16 *__restrict__ q,
                           const _Float16 *__restrict__ kcache,
                           const _Float16 *__restrict__ vcache, i64 n_head, i64 n_kv,
                           i64 hd, const i64 *__restrict__ pos0, i64 pos_stride,
                           i64 causal, i64 window, f32 scale) {
    const i64 h = blockIdx.x;
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    // n_tok == 1: the causal and non-causal key ranges both end at this
    // query's own position, so no per-key masking is needed. pos0 is
    // device memory so a captured graph replays without re-binding.
    const i64 p0 = *pos0;
    const i64 n_keys = causal ? p0 + 1 : p0 + 1;
    // Sliding window: keys below the bound are skipped rather than masked,
    // so they never enter the online softmax at all.
    const i64 j_lo = window > 0 && window < n_keys ? n_keys - window : 0;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    // q stays in registers for the whole kernel: VPT consecutive fp16 per lane
    // means one 4-byte load per pair.
    u32 qp[VPT / 2];
#pragma unroll
    for (int e = 0; e < VPT / 2; e++)
        qp[e] = *reinterpret_cast<const u32 *>(q + h * hd + lane * VPT + 2 * e);

    const _Float16 *kb = kcache + hkv * hd;
    const _Float16 *vb = vcache + hkv * hd;

    f32 m = -1.0e30f;
    f32 l = 0.0f;
    f32 acc[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) acc[e] = 0.0f;

    for (i64 j = j_lo + warp; j < n_keys; j += kAttnDecWarps) {
        const _Float16 *krow = kb + j * pos_stride;
        const _Float16 *vrow = vb + j * pos_stride;
        const f32 s = d_wave_reduce_sum(attn_lane_dot<VPT>(qp, krow, lane)) * scale;
        const f32 mnew = fmaxf(m, s);
        const f32 a = __expf(m - mnew);
        const f32 p = __expf(s - mnew);
        l = l * a + p;
#pragma unroll
        for (int e = 0; e < VPT; e++)
            acc[e] = acc[e] * a + p * static_cast<f32>(vrow[lane * VPT + e]);
        m = mnew;
    }

    // Merge the per-warp (m, l, O) triples. A warp that drew no keys keeps
    // m = -1e30 and l = 0, so exp(m - mstar) underflows to zero and it drops
    // out; if every warp is empty the denominator is zero and the output is
    // zero, matching the tiled kernel's l > 0 guard.
    __shared__ f32 sh_m[kAttnDecWarps];
    __shared__ f32 sh_l[kAttnDecWarps];
    __shared__ f32 sh_o[kAttnDecWarps * kWaveSize * VPT];

    if (lane == 0) {
        sh_m[warp] = m;
        sh_l[warp] = l;
    }
    __syncthreads();

    f32 mstar = -1.0e30f;
#pragma unroll
    for (int w = 0; w < kAttnDecWarps; w++) mstar = fmaxf(mstar, sh_m[w]);

    const f32 aw = __expf(m - mstar);
    f32 *slot = sh_o + (warp * kWaveSize + lane) * VPT;
#pragma unroll
    for (int e = 0; e < VPT; e++) slot[e] = acc[e] * aw;
    if (lane == 0) sh_l[warp] = l * aw;
    __syncthreads();

    f32 lt = 0.0f;
    f32 ot[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) ot[e] = 0.0f;
#pragma unroll
    for (int w = 0; w < kAttnDecWarps; w++) {
        lt += sh_l[w];
        const f32 *s2 = sh_o + (w * kWaveSize + lane) * VPT;
#pragma unroll
        for (int e = 0; e < VPT; e++) ot[e] += s2[e];
    }

    const f32 inv = lt > 0.0f ? 1.0f / lt : 0.0f;
    _Float16 *orow = out + h * hd + lane * VPT;
#pragma unroll
    for (int e = 0; e < VPT; e++) orow[e] = static_cast<_Float16>(ot[e] * inv);
}

// Static shared memory is enough (at most kAttnDecWarps * (kWaveSize*VPT + 1)
// floats), so unlike the tiled kernel this path needs no dynamic allocation
// and no hipFuncSetAttribute ceiling raise — and, per the ISA, LDS space is
// allocated in 1024-byte blocks, so keep the worst case (hd = 128) under the
// 64 kB per-work-group limit: 8 * (32*4 + 1) * 4 B = 4.2 kB.
template <int VPT>
inline void attention_decode_launch(void *out, const void *q, const void *kcache,
                                    const void *vcache, i64 n_head, i64 n_kv, i64 hd,
                                    const i64 *pos0, i64 pos_stride, int causal,
                                    i64 window, f32 scale,
                                    hipStream_t stream = nullptr) {
    attention_decode_kernel<VPT><<<static_cast<unsigned>(n_head), kAttnDecThreads, 0,
                                   stream>>>(
        static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
        static_cast<const _Float16 *>(kcache), static_cast<const _Float16 *>(vcache),
        n_head, n_kv, hd, pos0, pos_stride, causal, window, scale);
}

// ---------------------------------------------------------------------------
// key-split decode
//
// One block per head is not enough of a grid. Qwen3.5-0.8B has 8 query heads
// per layer on a 64-CU card: the decode kernel launched 8 blocks and left 56
// CUs idle, and because the tiled fallback has grid (n_head, n_tok) it did the
// same while additionally staging K and V tiles through LDS. Measured at ~4900
// keys that is 4.80 ms for one call over 10.0 MB of KV — 2.1 GB/s, 270x below
// what the card sustains on a large read.
//
// The fix is the split-K pattern the GEMMs already use here: cut the key range
// into `n_split` chunks, give each chunk a block, and merge the per-chunk
// (m, l, O) triples. Online softmax composes exactly, so the merge is the same
// arithmetic the single-block epilogue already does — it is not an
// approximation, and n_split == 1 reproduces the old kernel bit for bit.
//
// `n_split` is chosen from the key count, not fixed: chunks must be long
// enough to amortise a block (64 keys), so at a short context this collapses
// to 1 and changes nothing, while at long context it fills the device.
// ---------------------------------------------------------------------------

// Per split: running max, denominator, then VPT output channels.
template <int VPT>
__global__ void __launch_bounds__(kAttnDecThreads)
    attention_decode_split_kernel(_Float16 *__restrict__ out,
                                  const _Float16 *__restrict__ q,
                                  const _Float16 *__restrict__ kcache,
                                  const _Float16 *__restrict__ vcache, i64 n_head,
                                  i64 n_kv, i64 hd, const i64 *__restrict__ pos0,
                                  i64 pos_stride, i64 causal, i64 window, f32 scale,
                                  f32 *__restrict__ part, int n_split) {
    const i64 h = blockIdx.x;
    const int sp = static_cast<int>(blockIdx.y);
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    const i64 p0 = *pos0;
    const i64 n_keys = p0 + 1;
    // Sliding window: the splits tile the *visible* range. Chunks entirely
    // below the bound would launch a block to do nothing but their epilogue.
    const i64 j_lo = window > 0 && window < n_keys ? n_keys - window : 0;
    const i64 n_vis = n_keys - j_lo;

    // Split into equal chunks, rounded to a multiple of the warp count so each
    // block's warps still get whole keys and the ranges tile exactly.
    const i64 per = ((n_vis + n_split - 1) / n_split + kAttnDecWarps - 1) /
                    kAttnDecWarps * kAttnDecWarps;
    const i64 j0 = j_lo + static_cast<i64>(sp) * per;
    const i64 j1 = min(n_keys, j0 + per);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    u32 qp[VPT / 2];
#pragma unroll
    for (int e = 0; e < VPT / 2; e++)
        qp[e] = *reinterpret_cast<const u32 *>(q + h * hd + lane * VPT + 2 * e);

    const _Float16 *kb = kcache + hkv * hd;
    const _Float16 *vb = vcache + hkv * hd;

    f32 m = -1.0e30f;
    f32 l = 0.0f;
    f32 acc[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) acc[e] = 0.0f;

    for (i64 j = j0 + warp; j < j1; j += kAttnDecWarps) {
        const _Float16 *krow = kb + j * pos_stride;
        const _Float16 *vrow = vb + j * pos_stride;
        const f32 s = d_wave_reduce_sum(attn_lane_dot<VPT>(qp, krow, lane)) * scale;
        const f32 mnew = fmaxf(m, s);
        const f32 a = __expf(m - mnew);
        const f32 p = __expf(s - mnew);
        l = l * a + p;
#pragma unroll
        for (int e = 0; e < VPT; e++)
            acc[e] = acc[e] * a + p * static_cast<f32>(vrow[lane * VPT + e]);
        m = mnew;
    }

    // Reduce the warps of this block, exactly as the single-block epilogue
    // does: scale each warp's (m, l, O) by exp(m - mstar) into shared memory
    // and then sum. Doing it the other way (keeping raw l and re-deriving
    // exp(m_w - mstar) per warp) is where the block's denominator used to be
    // written by EIGHT threads at once with eight different values: `lt`
    // started at `s_l[warp] * aw`, so every lane-0 thread seeded the sum with
    // its own warp's term, and all eight store to p[1]. The store that landed
    // last won, so a head's output depended on warp scheduling -- a decode at
    // >= 128 keys (attention_decode_splits() starts splitting there) was
    // nondeterministic run to run, while the same model at 64 keys, where
    // n_split stays 1, was bit-stable. Measured: 4 GPU runs of the same
    // 128-token prompt gave one md5 once and another md5 three times.
    //
    // Scaling s_l in place (as the single-block kernel does) makes every
    // thread compute the same `lt` and `ot`, and restricting the write to
    // warp 0 removes the redundant global stores as well.
    __shared__ f32 s_m[kAttnDecWarps];
    __shared__ f32 s_l[kAttnDecWarps];
    __shared__ f32 s_o[kAttnDecWarps * kWaveSize * VPT];
    if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();
    f32 mstar = -1.0e30f;
#pragma unroll
    for (int w = 0; w < kAttnDecWarps; w++) mstar = fmaxf(mstar, s_m[w]);
    const f32 aw = __expf(m - mstar);
    f32 *slot = s_o + (warp * kWaveSize + lane) * VPT;
#pragma unroll
    for (int e = 0; e < VPT; e++) slot[e] = acc[e] * aw;
    if (lane == 0) s_l[warp] = l * aw;
    __syncthreads();
    f32 lt = 0.0f, ot[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) ot[e] = 0.0f;
#pragma unroll
    for (int w = 0; w < kAttnDecWarps; w++) {
        lt += s_l[w];
        const f32 *s2 = s_o + (w * kWaveSize + lane) * VPT;
#pragma unroll
        for (int e = 0; e < VPT; e++) ot[e] += s2[e];
    }
    // Write this block's partial in the same lane-major shape the single-block
    // epilogue uses: lane L owns output channels [L*VPT, (L+1)*VPT), so the
    // merge kernel can be the same reduction with one extra loop over splits.
    // One warp writes: `lt` and `ot` are functions of (lane, w) and not of
    // `warp`, so the other seven warps would store identical values.
    if (warp == 0) {
        f32 *p = part + (h * n_split + sp) * (2 + kWaveSize * VPT);
        if (lane == 0) {
            p[0] = mstar;
            p[1] = lt;
        }
        f32 *pa = p + 2 + lane * VPT;
#pragma unroll
        for (int e = 0; e < VPT; e++) pa[e] = ot[e];
    }
}

template <int VPT>
__global__ void __launch_bounds__(kAttnDecThreads)
    attention_decode_merge_kernel(_Float16 *__restrict__ out, i64 n_head, i64 hd,
                                  const f32 *__restrict__ part, int n_split) {
    const i64 h = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int stride = 2 + kWaveSize * VPT;
    const f32 *p = part + h * n_split * stride;
    f32 mstar = -1.0e30f;
    for (int s = 0; s < n_split; s++) mstar = fmaxf(mstar, p[s * stride]);
    f32 ot[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) ot[e] = 0.0f;
    f32 den = 0.0f;
    for (int s = 0; s < n_split; s++) {
        const f32 *e = p + s * stride;
        const f32 aw = __expf(e[0] - mstar);
        den += e[1] * aw;
        const f32 *a = e + 2 + lane * VPT;
#pragma unroll
        for (int v = 0; v < VPT; v++) ot[v] += a[v] * aw;
    }
    const f32 inv = den > 0.0f ? 1.0f / den : 0.0f;
    _Float16 *orow = out + h * hd + lane * VPT;
#pragma unroll
    for (int e = 0; e < VPT; e++) orow[e] = static_cast<_Float16>(ot[e] * inv);
}

// Host dispatch: VPT = hd/32 must be an even lane slice that the packed pair
// form covers (4 bytes per load, 8 bytes per wave lane group). hd = 256 is the
// Qwen3.5-0.8B head width and was missing here, which is why that model ran
// the tiled kernel: 8 blocks, no split, 2.1 GB/s.
inline bool attention_decode_supported(i64 hd) {
    return hd == 64 || hd == 128 || hd == 256;
}

// How many key chunks to cut into, given the key count and the device. One
// block per head is the floor; the target is a few blocks per CU, and a chunk
// must be long enough to pay for its own block.
inline int attention_decode_splits(i64 n_keys, i64 n_head, int cu_count) {
    const i64 target_blocks = static_cast<i64>(cu_count) * 4;
    if (n_head >= target_blocks) return 1;
    i64 want = (target_blocks + n_head - 1) / n_head;
    // A chunk shorter than kAttnDecWarps*8 keys cannot keep 8 warps fed.
    const i64 min_keys = kAttnDecWarps * 8;
    const i64 by_len = n_keys / min_keys;
    if (want > by_len) want = by_len;
    if (want < 1) want = 1;
    if (want > 256) want = 256;
    return static_cast<int>(want);
}

inline void attention_decode(void *out, const void *q, const void *kcache,
                             const void *vcache, i64 n_head, i64 n_kv, i64 hd,
                             const i64 *pos0, i64 pos_stride, int causal,
                             i64 window, f32 scale, hipStream_t stream = nullptr,
                             f32 *part = nullptr, int n_split = 1,
                             int cu_count = 0) {
    (void)causal;
    if (n_split < 1 || part == nullptr) n_split = 1;
    if (n_split == 1) {
        if (hd == 64)
            attention_decode_launch<2>(out, q, kcache, vcache, n_head, n_kv, hd,
                                       pos0, pos_stride, causal, window, scale,
                                       stream);
        else if (hd == 128)
            attention_decode_launch<4>(out, q, kcache, vcache, n_head, n_kv, hd,
                                       pos0, pos_stride, causal, window, scale,
                                       stream);
        else
            attention_decode_launch<8>(out, q, kcache, vcache, n_head, n_kv, hd,
                                       pos0, pos_stride, causal, window, scale,
                                       stream);
        return;
    }
#define KRK_ATTN_SPLIT(VPT)                                                   \
    do {                                                                      \
        const dim3 g(static_cast<unsigned>(n_head),                          \
                     static_cast<unsigned>(n_split));                         \
        attention_decode_split_kernel<VPT><<<g, kAttnDecThreads, 0, stream>>>( \
            static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),   \
            static_cast<const _Float16 *>(kcache),                            \
            static_cast<const _Float16 *>(vcache), n_head, n_kv, hd, pos0,     \
            pos_stride, causal, window, scale, part, n_split);                \
        attention_decode_merge_kernel<VPT>                                    \
            <<<static_cast<unsigned>(n_head), kAttnDecThreads, 0, stream>>>(  \
                static_cast<_Float16 *>(out), n_head, hd, part, n_split);      \
    } while (0)
    if (hd == 64) KRK_ATTN_SPLIT(2);
    else if (hd == 128) KRK_ATTN_SPLIT(4);
    else KRK_ATTN_SPLIT(8);
#undef KRK_ATTN_SPLIT
}

// ---------------------------------------------------------------------------
// fused decode chain: RoPE + KV append + attention, one launch
//
// Replaces the engine's four decode launches (rope q, rope k, kv_append,
// attention) with one. Per head block:
//   1. every lane rotates its own VPT/2 q pairs in registers — the roped
//      query never touches global memory again;
//   2. the block ropes the KV head's k row straight into the cache slot
//      (one pair per thread) and copies the v row beside it, so the
//      separate k buffer and its two round trips disappear;
//   3. the attention loop runs exactly as attention_decode_kernel does,
//      over the cache including the row this block just wrote.
//
// The append is deliberately redundant: every query block that maps to a
// KV head re-ropes and rewrites that head's cache row (n_rep query heads
// share one KV head). Restricting the append to one block per KV head
// would race — the sibling blocks must observe the fresh row, and there is
// no cross-block ordering inside one launch. The redundancy is idempotent
// (identical values, same address) and costs a fraction of the cache
// traffic the attention loop itself streams. A __syncthreads() between the
// append and the key loop makes the row visible to every thread of the
// block that wrote it.
//
// The math matches the separate chain pair for pair (same theta — the
// kernel divides the position by rope_scale exactly as rope_kernel does,
// so the match holds for every model, not just rope_scale 1 — same
// sincos, same fp32 narrow-at-store), so the appended cache rows and the
// attention output are bit-identical to rope + kv_append + attention run
// separately — §2i checks exactly that.
// ---------------------------------------------------------------------------

template <int VPT>
__global__ void __launch_bounds__(kAttnDecThreads)
    attn_fused_decode_kernel(_Float16 *__restrict__ out,
                             const _Float16 *__restrict__ q,
                             _Float16 *__restrict__ kcache,
                             _Float16 *__restrict__ vcache,
                             const _Float16 *__restrict__ k,
                             const _Float16 *__restrict__ v,
                             const f32 *__restrict__ wq,
                             const f32 *__restrict__ wk, f32 eps, bool neox,
                             i64 n_head, i64 n_kv, i64 hd,
                             const i64 *__restrict__ pos0,
                             i64 pos_stride, const f32 *__restrict__ inv_freq,
                             f32 rope_scale, f32 scale, f32 frac) {
    const i64 h = blockIdx.x;
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    const i64 p0 = *pos0;
    const i64 n_keys = p0 + 1; // decode: causal and non-causal agree

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const i64 half = hd / 2;
    const i64 rot = static_cast<i64>(static_cast<f32>(half) * frac);
    // rope_scale divides the position (LLaMA-style RoPE scaling):
    // theta = (pos / rope_scale) * inv_freq, exactly as rope_kernel
    // computes it, so the chain is bit-exact for every model.
    const f32 pf =
        static_cast<f32>(p0) / (rope_scale > 0.0f ? rope_scale : 1.0f);

    // (0) QK-norm scales, bit-identical to head_norm_kernel<128>: threads
    // 0..hd-1 contribute one element's square in channel order, the reduce
    // is the same block_reduce_sum<128> the separate kernel calls (from
    // this 256-thread block wave 0's window is red[0..3], which is exactly
    // that tree), and thread 0 computes the same rsqrtf(ss/hd + eps). The
    // normed operands are formed in the loads below with the same f32
    // expression the separate kernel stores f16 with, so the rotation sees
    // the same f16 values the separate chain feeds it.
    __shared__ f32 qscale, kscale;
    __shared__ f32 qred[4], kred[4];
    const auto nv = [](_Float16 x, f32 s, f32 wi) -> _Float16 {
        return static_cast<_Float16>(static_cast<f32>(x) * s * wi);
    };
    // block_reduce_sum<128> cannot be called from this 256-thread block:
    // its red[wave] write would run to red[4..7] and smash `total` (an
    // out-of-bounds shared write that showed up as a wrong md5, not a
    // crash). The tree is reproduced by hand instead, in the same order the
    // separate kernel's reduce walks: per-wave lane reduce, red[] in wave
    // order, wave 0's window over red[0..3] with exact zeros after -- so
    // the total, and therefore the f16-rounded normed values, are
    // bit-identical to head_norm_kernel<128>.
    if (wq) {
        f32 v = 0.0f;
        if (tid < hd) {
            const f32 x = static_cast<f32>(q[h * hd + tid]);
            v = x * x;
        }
        v = d_wave_reduce_sum(v);
        if (lane == 0 && warp < 4) qred[warp] = v;
        __syncthreads();
        if (warp == 0) {
            const f32 t = lane < 4 ? qred[lane] : 0.0f;
            const f32 tot = d_wave_reduce_sum(t);
            if (lane == 0)
                qscale = rsqrtf(tot / static_cast<f32>(hd) + eps);
        }
        __syncthreads();
    }
    if (wk) {
        f32 v = 0.0f;
        if (tid < hd) {
            const f32 x = static_cast<f32>(k[hkv * hd + tid]);
            v = x * x;
        }
        v = d_wave_reduce_sum(v);
        if (lane == 0 && warp < 4) kred[warp] = v;
        __syncthreads();
        if (warp == 0) {
            const f32 t = lane < 4 ? kred[lane] : 0.0f;
            const f32 tot = d_wave_reduce_sum(t);
            if (lane == 0)
                kscale = rsqrtf(tot / static_cast<f32>(hd) + eps);
        }
        __syncthreads();
    }

    // (1) RoPE on q, in registers. Lane L owns head elements
    // [L*VPT, (L+1)*VPT), i.e. VPT/2 consecutive rotation pairs
    // (L*(VPT/2) .. L*(VPT/2) + VPT/2 - 1).
    union Pair {
        u32 u;
        _Float16 h[2];
    };
    u32 qp[VPT / 2];
    if (!neox) {
#pragma unroll
        for (int e = 0; e < VPT / 2; e++) {
            f32 a, b;
            if (wq) {
                const i64 ja = static_cast<i64>(lane) * VPT + 2 * e;
                a = static_cast<f32>(nv(q[h * hd + ja], qscale, wq[ja]));
                b = static_cast<f32>(
                    nv(q[h * hd + ja + 1], qscale, wq[ja + 1]));
            } else {
                Pair p;
                p.u = *reinterpret_cast<const u32 *>(
                    q + h * hd + lane * VPT + 2 * e);
                a = static_cast<f32>(p.h[0]);
                b = static_cast<f32>(p.h[1]);
            }
            const i64 pr = static_cast<i64>(lane) * (VPT / 2) + e;
            if (pr < rot) {
                const f32 theta = pf * inv_freq[pr];
                f32 sn, cs;
                __sincosf(theta, &sn, &cs);
                const f32 ra = a * cs - b * sn;
                const f32 rb = a * sn + b * cs;
                a = ra;
                b = rb;
            }
            Pair o;
            o.h[0] = static_cast<_Float16>(a);
            o.h[1] = static_cast<_Float16>(b);
            qp[e] = o.u;
        }
    } else {
        // NeoX pairing: channel j rotates with j + rot (rope_kernel's neox
        // arm, ia = i / ib = i + rot). The engine takes this arm only when
        // rot == half, where the partner of j is j ^ half and sits in the
        // same element slot of the lane half/VPT = 16 away, so one wave
        // shuffle fetches it; pair index pj names the theta for both
        // halves, exactly as the separate kernel indexes inv_freq.
        const int poff = static_cast<int>(half / VPT);
        _Float16 xv[VPT];
#pragma unroll
        for (int e = 0; e < VPT; e++) {
            const i64 j = static_cast<i64>(lane) * VPT + e;
            f32 x;
            if (wq)
                x = static_cast<f32>(nv(q[h * hd + j], qscale, wq[j]));
            else
                x = static_cast<f32>(q[h * hd + j]);
            const f32 y = __shfl_xor(x, poff);
            const i64 pj = j < half ? j : j - half;
            f32 o = x;
            if (pj < rot) {
                const f32 theta = pf * inv_freq[pj];
                f32 sn, cs;
                __sincosf(theta, &sn, &cs);
                o = j < half ? x * cs - y * sn : y * sn + x * cs;
            }
            xv[e] = static_cast<_Float16>(o);
        }
#pragma unroll
        for (int e = 0; e < VPT / 2; e++) {
            Pair o;
            o.h[0] = xv[2 * e];
            o.h[1] = xv[2 * e + 1];
            qp[e] = o.u;
        }
    }

    // (2) QK-norm + rope k into the cache and copy v beside it. One thread
    // per rotation pair (a pair is the unit rope touches); v is a plain
    // elementwise copy. Only the first hd/2 / hd threads of the block
    // have work; the rest idle through this phase. The cache row keeps
    // channel order under both pairings, so the append writes exactly
    // where kv_append_kernel would have.
    const i64 koff = p0 * pos_stride + hkv * hd;
    if (!neox) {
        for (i64 i = tid; i < half; i += blockDim.x) {
            f32 a, b;
            if (wk) {
                a = static_cast<f32>(
                    nv(k[hkv * hd + 2 * i], kscale, wk[2 * i]));
                b = static_cast<f32>(
                    nv(k[hkv * hd + 2 * i + 1], kscale, wk[2 * i + 1]));
            } else {
                a = static_cast<f32>(k[hkv * hd + 2 * i]);
                b = static_cast<f32>(k[hkv * hd + 2 * i + 1]);
            }
            if (i < rot) {
                const f32 theta = pf * inv_freq[i];
                f32 sn, cs;
                __sincosf(theta, &sn, &cs);
                const f32 ra = a * cs - b * sn;
                const f32 rb = a * sn + b * cs;
                a = ra;
                b = rb;
            }
            kcache[koff + 2 * i] = static_cast<_Float16>(a);
            kcache[koff + 2 * i + 1] = static_cast<_Float16>(b);
        }
    } else {
        for (i64 i = tid; i < half; i += blockDim.x) {
            const i64 ia = i, ib = i + half;
            f32 a, b;
            if (wk) {
                a = static_cast<f32>(nv(k[hkv * hd + ia], kscale, wk[ia]));
                b = static_cast<f32>(nv(k[hkv * hd + ib], kscale, wk[ib]));
            } else {
                a = static_cast<f32>(k[hkv * hd + ia]);
                b = static_cast<f32>(k[hkv * hd + ib]);
            }
            if (i < rot) {
                const f32 theta = pf * inv_freq[i];
                f32 sn, cs;
                __sincosf(theta, &sn, &cs);
                const f32 ra = a * cs - b * sn;
                const f32 rb = a * sn + b * cs;
                a = ra;
                b = rb;
            }
            kcache[koff + ia] = static_cast<_Float16>(a);
            kcache[koff + ib] = static_cast<_Float16>(b);
        }
    }
    for (i64 i = tid; i < hd; i += blockDim.x)
        vcache[koff + i] = v[hkv * hd + i];
    __syncthreads();

    // (3) attention over the cache, exactly as attention_decode_kernel:
    // q (roped, in registers), per-warp online softmax, one merge.
    const _Float16 *kb = kcache + hkv * hd;
    const _Float16 *vb = vcache + hkv * hd;

    f32 m = -1.0e30f;
    f32 l = 0.0f;
    f32 acc[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) acc[e] = 0.0f;

    for (i64 j = warp; j < n_keys; j += kAttnDecWarps) {
        const _Float16 *krow = kb + j * pos_stride;
        const _Float16 *vrow = vb + j * pos_stride;
        const f32 s = d_wave_reduce_sum(attn_lane_dot<VPT>(qp, krow, lane)) * scale;
        const f32 mnew = fmaxf(m, s);
        const f32 a = __expf(m - mnew);
        const f32 p = __expf(s - mnew);
        l = l * a + p;
#pragma unroll
        for (int e = 0; e < VPT; e++)
            acc[e] = acc[e] * a + p * static_cast<f32>(vrow[lane * VPT + e]);
        m = mnew;
    }

    __shared__ f32 sh_m[kAttnDecWarps];
    __shared__ f32 sh_l[kAttnDecWarps];
    __shared__ f32 sh_o[kAttnDecWarps * kWaveSize * VPT];

    if (lane == 0) {
        sh_m[warp] = m;
        sh_l[warp] = l;
    }
    __syncthreads();

    f32 mstar = -1.0e30f;
#pragma unroll
    for (int w = 0; w < kAttnDecWarps; w++) mstar = fmaxf(mstar, sh_m[w]);

    const f32 aw = __expf(m - mstar);
    f32 *slot = sh_o + (warp * kWaveSize + lane) * VPT;
#pragma unroll
    for (int e = 0; e < VPT; e++) slot[e] = acc[e] * aw;
    if (lane == 0) sh_l[warp] = l * aw;
    __syncthreads();

    f32 lt = 0.0f;
    f32 ot[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) ot[e] = 0.0f;
#pragma unroll
    for (int w = 0; w < kAttnDecWarps; w++) {
        lt += sh_l[w];
        const f32 *s2 = sh_o + (w * kWaveSize + lane) * VPT;
#pragma unroll
        for (int e = 0; e < VPT; e++) ot[e] += s2[e];
    }

    const f32 inv = lt > 0.0f ? 1.0f / lt : 0.0f;
    _Float16 *orow = out + h * hd + lane * VPT;
#pragma unroll
    for (int e = 0; e < VPT; e++) orow[e] = static_cast<_Float16>(ot[e] * inv);
}

template <int VPT>
inline void attn_fused_decode_launch(void *out, const void *q,
                                     void *kcache, void *vcache,
                                     const void *k, const void *v,
                                     const f32 *wq, const f32 *wk, f32 eps,
                                     bool neox, i64 n_head, i64 n_kv, i64 hd,
                                     const i64 *pos0, i64 pos_stride,
                                     const f32 *inv_freq, f32 rope_scale,
                                     f32 scale, f32 frac,
                                     hipStream_t stream = nullptr) {
    attn_fused_decode_kernel<VPT><<<static_cast<unsigned>(n_head),
                                     kAttnDecThreads, 0, stream>>>(
        static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
        static_cast<_Float16 *>(kcache), static_cast<_Float16 *>(vcache),
        static_cast<const _Float16 *>(k), static_cast<const _Float16 *>(v),
        wq, wk, eps, neox, n_head, n_kv, hd, pos0, pos_stride, inv_freq,
        rope_scale, scale, frac);
}

// Head widths the packed lane slice covers (same rule as the decode
// attention path).
inline bool attn_fused_decode_supported(i64 hd) {
    return hd == 64 || hd == 128;
}inline void attn_fused_decode(void *out, const void *q, void *kcache,
                              void *vcache, const void *k, const void *v,
                              const f32 *wq, const f32 *wk, f32 eps, bool neox,
                              i64 n_head, i64 n_kv, i64 hd,
                              const i64 *pos0, i64 pos_stride,
                              const f32 *inv_freq, f32 rope_scale, f32 scale,
                              f32 frac, hipStream_t stream = nullptr) {
    if (hd == 64)
        attn_fused_decode_launch<2>(out, q, kcache, vcache, k, v, wq, wk, eps,
                                    neox, n_head, n_kv, hd, pos0, pos_stride,
                                    inv_freq, rope_scale, scale, frac, stream);
    else if (hd == 128)
        attn_fused_decode_launch<4>(out, q, kcache, vcache, k, v, wq, wk, eps,
                                    neox, n_head, n_kv, hd, pos0, pos_stride,
                                    inv_freq, rope_scale, scale, frac, stream);
}

} // namespace krk

#endif // KRK_KERNEL_ATTENTION_HPP
