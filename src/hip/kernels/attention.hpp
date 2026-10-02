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
                     f32 scale, int tile_k) {
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

    for (i64 k0 = 0; k0 < n_keys; k0 += tile_k) {
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

        // one key per thread: full head-width dot product
        if (tid < tile) {
            f32 dot = 0.0f;
            const _Float16 *kj = k_sh + static_cast<i64>(tid) * hd;
#pragma unroll 8
            for (i64 d = 0; d < hd; d++)
                dot += static_cast<f32>(q_sh[d]) * static_cast<f32>(kj[d]);
            s_sh[tid] = dot * scale;
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
                           i64 causal, f32 scale) {
    const i64 h = blockIdx.x;
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    // n_tok == 1: the causal and non-causal key ranges both end at this
    // query's own position, so no per-key masking is needed. pos0 is
    // device memory so a captured graph replays without re-binding.
    const i64 p0 = *pos0;
    const i64 n_keys = causal ? p0 + 1 : p0 + 1;

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
                                    f32 scale, hipStream_t stream = nullptr) {
    attention_decode_kernel<VPT><<<static_cast<unsigned>(n_head), kAttnDecThreads, 0,
                                   stream>>>(
        static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
        static_cast<const _Float16 *>(kcache), static_cast<const _Float16 *>(vcache),
        n_head, n_kv, hd, pos0, pos_stride, causal, scale);
}

// Host dispatch: VPT = hd/32 must be an even lane slice that the packed pair
// form covers (4 bytes per load, 8 bytes per wave lane group). Heads of 64 and
// 128 are the ones this engine actually sees; anything else falls back to the
// tiled kernel.
inline bool attention_decode_supported(i64 hd) { return hd == 64 || hd == 128; }

inline void attention_decode(void *out, const void *q, const void *kcache,
                             const void *vcache, i64 n_head, i64 n_kv, i64 hd,
                             const i64 *pos0, i64 pos_stride, int causal,
                             f32 scale, hipStream_t stream = nullptr) {
    if (hd == 64)
        attention_decode_launch<2>(out, q, kcache, vcache, n_head, n_kv, hd, pos0,
                                   pos_stride, causal, scale, stream);
    else if (hd == 128)
        attention_decode_launch<4>(out, q, kcache, vcache, n_head, n_kv, hd, pos0,
                                   pos_stride, causal, scale, stream);
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
                             i64 n_head, i64 n_kv, i64 hd,
                             const i64 *__restrict__ pos0,
                             i64 pos_stride, const f32 *__restrict__ inv_freq,
                             f32 rope_scale, f32 scale, f32 frac) {
    const i64 h = blockIdx.x;
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    const i64 p0 = *pos0;
    const i64 n_keys = p0 + 1; // decode: causal and non-causal agree

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const i64 half = hd / 2;
    const i64 rot = static_cast<i64>(static_cast<f32>(half) * frac);
    // rope_scale divides the position (LLaMA-style RoPE scaling):
    // theta = (pos / rope_scale) * inv_freq, exactly as rope_kernel
    // computes it, so the chain is bit-exact for every model.
    const f32 pf =
        static_cast<f32>(p0) / (rope_scale > 0.0f ? rope_scale : 1.0f);

    // (1) RoPE on q, in registers. Lane L owns head elements
    // [L*VPT, (L+1)*VPT), i.e. VPT/2 consecutive rotation pairs
    // (L*(VPT/2) .. L*(VPT/2) + VPT/2 - 1).
    union Pair {
        u32 u;
        _Float16 h[2];
    };
    u32 qp[VPT / 2];
#pragma unroll
    for (int e = 0; e < VPT / 2; e++) {
        Pair p;
        p.u = *reinterpret_cast<const u32 *>(
            q + h * hd + lane * VPT + 2 * e);
        const i64 pr = static_cast<i64>(lane) * (VPT / 2) + e;
        if (pr < rot) {
            const f32 theta = pf * inv_freq[pr];
            f32 sn, cs;
            __sincosf(theta, &sn, &cs);
            const f32 a = static_cast<f32>(p.h[0]);
            const f32 b = static_cast<f32>(p.h[1]);
            p.h[0] = static_cast<_Float16>(a * cs - b * sn);
            p.h[1] = static_cast<_Float16>(a * sn + b * cs);
        }
        qp[e] = p.u;
    }

    // (2) rope k into the cache and copy v beside it. One thread per
    // rotation pair (a pair is the unit rope touches); v is a plain
    // elementwise copy. Only the first hd/2 / hd threads of the block
    // have work; the rest idle through this phase.
    const i64 koff = p0 * pos_stride + hkv * hd;
    for (i64 i = threadIdx.x; i < half; i += blockDim.x) {
        Pair p;
        p.h[0] = k[hkv * hd + 2 * i];
        p.h[1] = k[hkv * hd + 2 * i + 1];
        if (i < rot) {
            const f32 theta = pf * inv_freq[i];
            f32 sn, cs;
            __sincosf(theta, &sn, &cs);
            const f32 a = static_cast<f32>(p.h[0]);
            const f32 b = static_cast<f32>(p.h[1]);
            p.h[0] = static_cast<_Float16>(a * cs - b * sn);
            p.h[1] = static_cast<_Float16>(a * sn + b * cs);
        }
        kcache[koff + 2 * i] = p.h[0];
        kcache[koff + 2 * i + 1] = p.h[1];
    }
    for (i64 i = threadIdx.x; i < hd; i += blockDim.x)
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
                                     i64 n_head, i64 n_kv, i64 hd,
                                     const i64 *pos0, i64 pos_stride,
                                     const f32 *inv_freq, f32 rope_scale,
                                     f32 scale, f32 frac,
                                     hipStream_t stream = nullptr) {
    attn_fused_decode_kernel<VPT><<<static_cast<unsigned>(n_head),
                                     kAttnDecThreads, 0, stream>>>(
        static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
        static_cast<_Float16 *>(kcache), static_cast<_Float16 *>(vcache),
        static_cast<const _Float16 *>(k), static_cast<const _Float16 *>(v),
        n_head, n_kv, hd, pos0, pos_stride, inv_freq, rope_scale, scale,
        frac);
}

// Head widths the packed lane slice covers (same rule as the decode
// attention path).
inline bool attn_fused_decode_supported(i64 hd) {
    return hd == 64 || hd == 128;
}

inline void attn_fused_decode(void *out, const void *q, void *kcache,
                              void *vcache, const void *k, const void *v,
                              i64 n_head, i64 n_kv, i64 hd,
                              const i64 *pos0, i64 pos_stride,
                              const f32 *inv_freq, f32 rope_scale,
                              f32 scale, f32 frac,
                              hipStream_t stream = nullptr) {
    if (hd == 64)
        attn_fused_decode_launch<2>(out, q, kcache, vcache, k, v, n_head,
                                     n_kv, hd, pos0, pos_stride, inv_freq,
                                     rope_scale, scale, frac, stream);
    else if (hd == 128)
        attn_fused_decode_launch<4>(out, q, kcache, vcache, k, v, n_head,
                                     n_kv, hd, pos0, pos_stride, inv_freq,
                                     rope_scale, scale, frac, stream);
}

} // namespace krk

#endif // KRK_KERNEL_ATTENTION_HPP
