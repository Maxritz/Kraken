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
// The 16x16x16 fragment loaders and the gfx12 slot map, shared with the GEMM.
// One definition of the fragment layout is what keeps this kernel and the
// prefill GEMM reading the same lanes for the same (row, column).
#include "gemm.hpp"

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
// int8 K/V: a row is `kv_dim` signed bytes followed by one f32 scale per 32
// values, so the row stride is a BYTE count. A NEGATIVE pos_stride is the flag
// that says so: its magnitude is the stride and no kernel needs an extra
// argument to learn the format. `kv_dim` is n_kv * hd, always available at the
// read site, and (idx >> 5) is the scale block because the scales are a
// contiguous plane at the end of the row.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool attn_kv_q8(i64 pos_stride) { return pos_stride < 0; }
__device__ __forceinline__ i64 attn_kv_stride(i64 pos_stride) {
    return pos_stride < 0 ? -pos_stride : pos_stride;
}
__device__ __forceinline__ f32 attn_kv_at_q8(const u8 *row, i64 kv_dim, i64 idx) {
    const f32 *sc = reinterpret_cast<const f32 *>(row + kv_dim);
    // Rounded through f16 on the way out, because the tiled and bit-exact
    // kernels STAGE a dequantised value into f16 shared memory and dot from
    // there. Without the round trip a chunk-1 decode row (f32 dequant) and a
    // chunk-8 prefill (f16 staged) differ by up to one f16 ulp per channel,
    // which moves the prefill trace hash and breaks chunk invariance for a
    // reason that has nothing to do with the cache bytes.
    return static_cast<f32>(static_cast<_Float16>(
        static_cast<f32>(static_cast<i8>(row[idx])) * sc[idx >> 5]));
}

// Quantise this step's K/V rows into the int8 layout. One thread per 32-value
// block of one (token, plane): it reads the block, writes the scale and the 32
// bytes. Per-row, per-32 scaling is what makes a chunked prefill and a decode
// step produce the same cache bytes.
// The int8 lane slice: the same channel partition and the same ARITHMETIC as
// attn_lane_dot. Each pair is dequantised into f16 (which attn_kv_at_q8
// already rounds to) and consumed by one v_dot2_f32_f16, exactly as the tiled
// and bit-exact kernels consume a dequantised pair staged in LDS. A per-element
// f32 FMA chain here instead -- `s += q[e] * dequant(k[e])` -- is a different
// reduction: two separate FMAs against one packed dot, and it put a chunk-1
// decode row one f16 ulp per channel away from a chunk-8 prefill over the same
// cache bytes. That is not a cache defect, so it cannot be excused as one.
template <int VPT>
__device__ __forceinline__ f32 attn_lane_dot_q8(const u32 *qp, const u8 *row,
                                                i64 kv_dim, i64 head_off,
                                                int lane) {
    static_assert(VPT % 2 == 0, "decode attention lane slice must pair up");
    union Pair {
        u32 u;
        _Float16 h[2];
    };
    const i64 base = head_off + static_cast<i64>(lane) * VPT;
    f32 s = 0.0f;
#pragma unroll
    for (int e = 0; e < VPT / 2; e++) {
        Pair kp;
        kp.h[0] = static_cast<_Float16>(
            attn_kv_at_q8(row, kv_dim, base + 2 * e));
        kp.h[1] = static_cast<_Float16>(
            attn_kv_at_q8(row, kv_dim, base + 2 * e + 1));
        s = d_dot2_pk(qp[e], kp.u, s);
    }
    return s;
}

__global__ void kv_append_i8_kernel(u8 *__restrict__ kc, u8 *__restrict__ vc,
                                    const _Float16 *__restrict__ k,
                                    const _Float16 *__restrict__ v,
                                    const i64 *__restrict__ pos0, i64 n_tok,
                                    i64 kv_dim, i64 pos_stride) {
    const i64 blocks = (kv_dim + 31) / 32;
    const i64 total = n_tok * 2 * blocks;
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const i64 b = i % blocks;
    const i64 uv = i / blocks;
    const bool is_v = uv >= n_tok;
    const i64 t = is_v ? uv - n_tok : uv;
    const _Float16 *src = (is_v ? v : k) + t * kv_dim + b * 32;
    u8 *row = (is_v ? vc : kc) + (*pos0 + t) * pos_stride;
    const i64 nn = min<i64>(32, kv_dim - b * 32);
    f32 mx = 0.0f;
    for (i64 z = 0; z < nn; z++) mx = fmaxf(mx, fabsf(static_cast<f32>(src[z])));
    const f32 s = mx > 0.0f ? mx / 127.0f : 0.0f;
    reinterpret_cast<f32 *>(row + kv_dim)[b] = s;
    const f32 inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (i64 z = 0; z < nn; z++) {
        const f32 q = static_cast<f32>(src[z]) * inv;
        i64 r = static_cast<i64>(q + (q >= 0.0f ? 0.5f : -0.5f));
        r = r > 127 ? 127 : (r < -127 ? -127 : r);
        row[b * 32 + z] = static_cast<u8>(static_cast<i8>(r));
    }
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
                     i64 window, f32 scale, int tile_k,
                     const f32 *__restrict__ sinks, i64 skip_from) {
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
    // A negative pos_stride marks an int8 cache; its magnitude is the row
    // stride in bytes. The tiled kernel stages into f16 LDS in both formats, so
    // everything downstream of the staging loop is unchanged.
    const bool kv8 = attn_kv_q8(pos_stride);
    const i64 kstride = attn_kv_stride(pos_stride);
    const i64 kv_dim = n_kv * hd;
    const _Float16 *krow = kcache + hkv * hd;
    const _Float16 *vrow = vcache + hkv * hd;
    // Row base, not the head slice: the scales plane is at row + kv_dim, so a
    // pointer already offset by hkv*hd would read the scales of a different
    // head. The head's channels are an INDEX offset below instead.
    const u8 *krow8 = reinterpret_cast<const u8 *>(kcache);
    const u8 *vrow8 = reinterpret_cast<const u8 *>(vcache);

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
            k_sh[i] = kv8
                          ? attn_kv_at_q8(krow8 + (k0 + j) * kstride, kv_dim,
                                          hkv * hd + d)
                          : krow[(k0 + j) * kstride + d];
        }
        for (i64 i = tid; i < tile_elems; i += kAttnBlock) {
            const i64 j = i / hd;
            const i64 d = i % hd;
            v_sh[i] = kv8
                          ? attn_kv_at_q8(vrow8 + (k0 + j) * kstride, kv_dim,
                                          hkv * hd + d)
                          : vrow[(k0 + j) * kstride + d];
        }
        __syncthreads();

        // one key per thread: full head-width dot product. Keys below the
        // window bound stay in the tile -- the staging is uniform across the
        // block -- but never reach the max or the denominator.
        if (tid < tile) {
            const i64 jk = k0 + static_cast<i64>(tid);
            // Windowed-out keys keep their slot in the score plane and never
            // reach the max or the denominator; a masked-block sibling row does
            // the same, and because `pos` is this query's own position the
            // test is a range, not an equality: [skip_from, pos).
            if (jk < k_lo || (skip_from >= 0 && jk >= skip_from && jk < pos)) {
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

    // Sink: a zero-value softmax entry at this row's final running max, so it
    // only thickens the denominator -- acc is left alone. Added once after
    // the tile loop, so no tile's arithmetic changes.
    if (sinks && sinks[h] != 0.0f) l += __expf(sinks[h] - m);
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
                           i64 causal, i64 window, f32 scale,
                           const f32 *__restrict__ sinks) {
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

    const bool kv8 = attn_kv_q8(pos_stride);
    const i64 kstride = attn_kv_stride(pos_stride);
    const i64 kv_dim = n_kv * hd;
    const u8 *kb8 = reinterpret_cast<const u8 *>(kcache);
    const u8 *vb8 = reinterpret_cast<const u8 *>(vcache);
    const _Float16 *kb = kcache + hkv * hd;
    const _Float16 *vb = vcache + hkv * hd;

    f32 m = -1.0e30f;
    f32 l = 0.0f;
    f32 acc[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) acc[e] = 0.0f;

    for (i64 j = j_lo + warp; j < n_keys; j += kAttnDecWarps) {
        const _Float16 *krow = kb + j * kstride;
        const _Float16 *vrow = vb + j * kstride;
        const u8 *krow8 = kb8 + j * kstride;
        const u8 *vrow8 = vb8 + j * kstride;
        const f32 s = d_wave_reduce_sum(
            kv8 ? attn_lane_dot_q8<VPT>(qp, krow8, kv_dim, hkv * hd, lane)
                : attn_lane_dot<VPT>(qp, krow, lane)) * scale;
        const f32 mnew = fmaxf(m, s);
        const f32 a = __expf(m - mnew);
        const f32 p = __expf(s - mnew);
        l = l * a + p;
#pragma unroll
        for (int e = 0; e < VPT; e++)
            acc[e] = acc[e] * a +
                     p * (kv8 ? attn_kv_at_q8(vrow8, kv_dim, hkv * hd + lane * VPT + e)
                              : static_cast<f32>(vrow[lane * VPT + e]));
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

    // Sink at the merged max, once per head -- never inside the warp loop or
    // the eight-warp merge: each of those seeding its own term would count
    // it several times and the denominator would scale with the warp count.
    if (sinks && sinks[h] != 0.0f) lt += __expf(sinks[h] - mstar);
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
                                    hipStream_t stream = nullptr,
                                    const f32 *sinks = nullptr) {
    attention_decode_kernel<VPT><<<static_cast<unsigned>(n_head), kAttnDecThreads, 0,
                                   stream>>>(
        static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
        static_cast<const _Float16 *>(kcache), static_cast<const _Float16 *>(vcache),
        n_head, n_kv, hd, pos0, pos_stride, causal, window, scale, sinks);
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

    const bool kv8 = attn_kv_q8(pos_stride);
    const i64 kstride = attn_kv_stride(pos_stride);
    const i64 kv_dim = n_kv * hd;
    const u8 *kb8 = reinterpret_cast<const u8 *>(kcache);
    const u8 *vb8 = reinterpret_cast<const u8 *>(vcache);
    const _Float16 *kb = kcache + hkv * hd;
    const _Float16 *vb = vcache + hkv * hd;

    f32 m = -1.0e30f;
    f32 l = 0.0f;
    f32 acc[VPT];
#pragma unroll
    for (int e = 0; e < VPT; e++) acc[e] = 0.0f;

    for (i64 j = j0 + warp; j < j1; j += kAttnDecWarps) {
        const _Float16 *krow = kb + j * kstride;
        const _Float16 *vrow = vb + j * kstride;
        const u8 *krow8 = kb8 + j * kstride;
        const u8 *vrow8 = vb8 + j * kstride;
        const f32 s = d_wave_reduce_sum(
            kv8 ? attn_lane_dot_q8<VPT>(qp, krow8, kv_dim, hkv * hd, lane)
                : attn_lane_dot<VPT>(qp, krow, lane)) * scale;
        const f32 mnew = fmaxf(m, s);
        const f32 a = __expf(m - mnew);
        const f32 p = __expf(s - mnew);
        l = l * a + p;
#pragma unroll
        for (int e = 0; e < VPT; e++)
            acc[e] = acc[e] * a +
                     p * (kv8 ? attn_kv_at_q8(vrow8, kv_dim, hkv * hd + lane * VPT + e)
                              : static_cast<f32>(vrow[lane * VPT + e]));
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
                                  const f32 *__restrict__ part, int n_split,
                                  const f32 *__restrict__ sinks) {
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
    // The sink is folded in here and NEVER in the split kernel: a split that
    // seeded its own term would count the sink once per key range and the
    // denominator would scale with n_split.
    if (sinks && sinks[h] != 0.0f) den += __expf(sinks[h] - mstar);
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
                             int cu_count = 0, const f32 *sinks = nullptr) {
    (void)causal;
    if (n_split < 1 || part == nullptr) n_split = 1;
    if (n_split == 1) {
        if (hd == 64)
            attention_decode_launch<2>(out, q, kcache, vcache, n_head, n_kv, hd,
                                       pos0, pos_stride, causal, window, scale,
                                       stream, sinks);
        else if (hd == 128)
            attention_decode_launch<4>(out, q, kcache, vcache, n_head, n_kv, hd,
                                       pos0, pos_stride, causal, window, scale,
                                       stream, sinks);
        else
            attention_decode_launch<8>(out, q, kcache, vcache, n_head, n_kv, hd,
                                       pos0, pos_stride, causal, window, scale,
                                       stream, sinks);
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
                static_cast<_Float16 *>(out), n_head, hd, part, n_split,      \
                sinks);                                                        \
    } while (0)
    if (hd == 64) KRK_ATTN_SPLIT(2);
    else if (hd == 128) KRK_ATTN_SPLIT(4);
    else KRK_ATTN_SPLIT(8);
#undef KRK_ATTN_SPLIT
}

// ---------------------------------------------------------------------------
// bit-exact prefill: the decode row's arithmetic over a tile of query rows
//
// `attention_decode_kernel` and the tiled prefill kernels disagree in the last
// bit on a handful of channels, and on a hybrid stack that difference flips an
// argmax after a few dozen tokens -- so a chunked prefill and a token-by-token
// decode produce different text from the same prompt (measured on Tiel:
// `--chunk 1` 613eb9b3 against `--chunk 2..8` e1a040e0, first divergence
// `attn.out` on the first full-attention layer at position 1).
//
// This kernel keeps what the tiled form is for -- K/V staged in LDS once and
// shared by QT query rows -- and reproduces decode's arithmetic exactly:
//   * warp w owns the keys j with j % 8 == w, in increasing order, which is
//     decode's key partition (`for (j = j_lo + warp; j < n_keys; j += 8)`);
//   * one (row, key) score is decode's lane dot: lane L covers channels
//     [L*VPT, (L+1)*VPT), VPT/2 packed v_dot2 pairs accumulated in order, then
//     one 32-lane shuffle reduce (attn_lane_dot + d_wave_reduce_sum);
//   * the online softmax advances PER KEY in decode's own order (mnew, a, p),
//     not once per key tile;
//   * the eight warps' (m, l, acc) triples are merged in warp order with the
//     same expression decode's tail uses.
// Only the K/V staging is shared, so this is a prefill kernel that lands on
// decode's numbers rather than a batch of decode launches -- which would re-read
// the whole K/V history once per query row, the traffic the tiled form exists
// to avoid.
//
// VPT = hd/32, so the head width must be one decode supports (64/128/256).
// A windowed layer stays on the tiled kernel: decode's partition starts at
// `j_lo + warp`, so which warp owns a key depends on that row's window bound,
// and one block cannot serve rows whose offsets differ.
// ---------------------------------------------------------------------------

template <int VPT, int QT>
__host__ __device__ __forceinline__ size_t attn_bexact_smem_bytes(int tile_k, i64 hd) {
    const size_t halves =
        (static_cast<size_t>(QT) + 2u * static_cast<size_t>(tile_k)) *
        static_cast<size_t>(hd);
    const size_t floats = 2u * kAttnDecWarps +
                          static_cast<size_t>(kAttnDecWarps) *
                              static_cast<size_t>(kWaveSize) *
                              static_cast<size_t>(VPT);
    return halves * sizeof(_Float16) + floats * sizeof(f32);
}

template <int VPT, int QT>
__global__ void __launch_bounds__(kAttnDecThreads)
    attention_bexact_kernel(_Float16 *__restrict__ out,
                            const _Float16 *__restrict__ q,
                            const _Float16 *__restrict__ kcache,
                            const _Float16 *__restrict__ vcache, i64 n_head, i64 n_kv,
                            i64 hd,                            i64 n_tok, i64 pos0, i64 pos_stride, f32 scale,
                            int tile_k, const f32 *__restrict__ sinks,
                            i64 skip_from) {
    extern __shared__ u8 smem[];
    _Float16 *q_sh = reinterpret_cast<_Float16 *>(smem);
    _Float16 *k_sh = q_sh + static_cast<i64>(QT) * hd;
    _Float16 *v_sh = k_sh + static_cast<i64>(tile_k) * hd;
    f32 *m_sh = reinterpret_cast<f32 *>(v_sh + static_cast<i64>(tile_k) * hd);
    f32 *l_sh = m_sh + kAttnDecWarps;
    f32 *o_sh = l_sh + kAttnDecWarps;

    const i64 h = blockIdx.x;
    const i64 t0 = static_cast<i64>(blockIdx.y) * QT;
    if (t0 >= n_tok) return;
    const int nq = static_cast<int>(min(static_cast<i64>(QT), n_tok - t0));
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWaveSize - 1);
    const int warp = tid / kWaveSize;

    // Q rows are strided by n_head*hd, not contiguous: staging them as one
    // run walks straight into the next head's row.
    // staging them as one run walks into the next head's row.
    for (i64 i = tid; i < static_cast<i64>(nq) * hd; i += kAttnDecThreads) {
        const i64 qi = i / hd, dd = i % hd;
        q_sh[i] = q[((t0 + qi) * n_head + h) * hd + dd];
    }
    __syncthreads();

    const bool kv8 = attn_kv_q8(pos_stride);
    const i64 kstride = attn_kv_stride(pos_stride);
    const i64 kv_dim = n_kv * hd;
    const _Float16 *kb = kcache + hkv * hd;
    const _Float16 *vb = vcache + hkv * hd;
    const u8 *kb8 = reinterpret_cast<const u8 *>(kcache);
    const u8 *vb8 = reinterpret_cast<const u8 *>(vcache);

    // The newest row's causal bound is the furthest key any row here needs.
    const i64 n_keys = pos0 + t0 + nq;

    f32 acc[QT][VPT];
    f32 mq[QT], lq[QT];
#pragma unroll
    for (int r = 0; r < QT; r++) {
        mq[r] = -1.0e30f;
        lq[r] = 0.0f;
#pragma unroll
        for (int e = 0; e < VPT; e++) acc[r][e] = 0.0f;
    }

    for (i64 k0 = 0; k0 < n_keys; k0 += tile_k) {
        const int tile =
            static_cast<int>((n_keys - k0) < tile_k ? (n_keys - k0) : tile_k);
        const i64 tile_elems = static_cast<i64>(tile) * hd;
        for (i64 i = tid; i < tile_elems; i += kAttnDecThreads) {
            const i64 j = i / hd, dd = i % hd;
            k_sh[i] = kv8
                          ? attn_kv_at_q8(kb8 + (k0 + j) * kstride, kv_dim,
                                          hkv * hd + dd)
                          : kb[(k0 + j) * kstride + dd];
        }
        for (i64 i = tid; i < tile_elems; i += kAttnDecThreads) {
            const i64 j = i / hd, dd = i % hd;
            v_sh[i] = kv8
                          ? attn_kv_at_q8(vb8 + (k0 + j) * kstride, kv_dim,
                                          hkv * hd + dd)
                          : vb[(k0 + j) * kstride + dd];
        }
        __syncthreads();

        // This warp's keys, in decode's order: j = warp, warp + 8, ...
        for (int i = warp; i < tile; i += kAttnDecWarps) {
            const i64 j = k0 + i;
#pragma unroll
            for (int r = 0; r < QT; r++) {
                if (r >= nq) continue;
                const i64 rpos = pos0 + t0 + r;
                if (j > rpos) continue; // this row's causal bound
                // Masked-block sibling: visible to the causal bound, masked
                // out for this row because it is another candidate of the
                // same block. Its own position rpos stays visible.
                if (skip_from >= 0 && j >= skip_from && j < rpos) continue;
                f32 s = 0.0f;
#pragma unroll
                for (int e = 0; e < VPT / 2; e++)
                    s = d_dot2_pk(
                        *reinterpret_cast<const u32 *>(
                            q_sh + static_cast<i64>(r) * hd + lane * VPT + 2 * e),
                        *reinterpret_cast<const u32 *>(
                            k_sh + static_cast<i64>(i) * hd + lane * VPT + 2 * e),
                        s);
                if (VPT % 2) {
                    // Odd lane slice (hd 96, 160, 224): the last channel of
                    // this lane's run has no partner, so it is one scalar FMA
                    // after the pairs. Decode has no kernel at these widths, so
                    // there is no bit pattern to match here -- the point of the
                    // branch is that the only prefill kernel covers them at all
                    // rather than dropping to the tiled fallback.
                    const i64 d = static_cast<i64>(lane) * VPT + (VPT - 1);
                    s += static_cast<f32>(q_sh[static_cast<i64>(r) * hd + d]) *
                         static_cast<f32>(k_sh[static_cast<i64>(i) * hd + d]);
                }
                s = d_wave_reduce_sum(s) * scale;
                const f32 mnew = fmaxf(mq[r], s);
                const f32 a = __expf(mq[r] - mnew);
                const f32 p = __expf(s - mnew);
                lq[r] = lq[r] * a + p;
#pragma unroll
                for (int e = 0; e < VPT; e++)
                    acc[r][e] =
                        acc[r][e] * a +
                        p * static_cast<f32>(
                                v_sh[static_cast<i64>(i) * hd + lane * VPT + e]);
                mq[r] = mnew;
            }
        }
        __syncthreads();
    }

    // Decode's tail, once per row and in warp order. `nq` is block-uniform, so
    // the guarded iterations keep every __syncthreads() on one path.
#pragma unroll
    for (int r = 0; r < QT; r++) {
        if (r >= nq) continue;
        if (lane == 0) {
            m_sh[warp] = mq[r];
            l_sh[warp] = lq[r];
        }
        __syncthreads();
        f32 mstar = -1.0e30f;
#pragma unroll
        for (int w = 0; w < kAttnDecWarps; w++) mstar = fmaxf(mstar, m_sh[w]);
        const f32 aw = __expf(mq[r] - mstar);
        f32 *slot = o_sh + (warp * kWaveSize + lane) * VPT;
#pragma unroll
        for (int e = 0; e < VPT; e++) slot[e] = acc[r][e] * aw;
        if (lane == 0) l_sh[warp] = lq[r] * aw;
        __syncthreads();
        f32 lt = 0.0f;
        f32 ot[VPT];
#pragma unroll
        for (int e = 0; e < VPT; e++) ot[e] = 0.0f;
#pragma unroll
        for (int w = 0; w < kAttnDecWarps; w++) {
            lt += l_sh[w];
            const f32 *s2 = o_sh + (w * kWaveSize + lane) * VPT;
#pragma unroll
            for (int e = 0; e < VPT; e++) ot[e] += s2[e];
        }
        // Sink for this row's head at the row's merged max -- the same
        // expression the decode row uses, which is what keeps a chunked
        // prefill and a decode step bit-identical on a sink model too.
        if (sinks && sinks[h] != 0.0f) lt += __expf(sinks[h] - mstar);
        const f32 inv = lt > 0.0f ? 1.0f / lt : 0.0f;
        _Float16 *orow = out + ((t0 + r) * n_head + h) * hd + lane * VPT;
#pragma unroll
        for (int e = 0; e < VPT; e++) orow[e] = static_cast<_Float16>(ot[e] * inv);
        __syncthreads();
    }
}

// Host dispatch. QT is how many query rows share one staged K/V tile; the
// accumulators are VPT*QT registers per thread, so it is chosen to hold that
// product at 32 (QT=4 at hd 256, QT=8 at hd 128 and 64). KRK_ATTN_BEXACT_QT=8
// doubles it for hd 256 as an A/B arm.
// Head widths this kernel expresses: a whole number of 32-channel lane runs,
// up to decode's widest (hd 256). Anything else -- and any window, whose key
// range starts at `j_lo + warp` per row -- takes the tiled fallback.
inline bool attn_bexact_supported(i64 hd) {
    return hd >= 64 && hd <= 256 && (hd % 32) == 0;
}

// Attention sinks (V4 report Eq. 27 / gpt-oss): one logit per query head,
// entered into the softmax DENOMINATOR at the row's running max -- it
// contributes no value. The kernels above do exactly this inline behind
// `if (sinks)`; this device-level entry exists so the standalone chain
// (rope -> append -> sink -> attention) gets the same term, and it is the
// place a test drives without a whole model.
__global__ void attention_sink_kernel(const f32 *__restrict__ sinks,
                                      f32 *__restrict__ l, i64 n_head,
                                      i64 pos_stride, i64 n_keys) {
    const i64 h = static_cast<i64>(blockIdx.x) * kAttnDecThreads + threadIdx.x;
    if (h >= n_head) return;
    const i64 row = n_keys - 1;
    if (row < 0) return;  // SINK_ROWS: an empty prefix is a no-op
    const f32 sink_logit = sinks[h];
    if (sink_logit == 0.0f) return;  // zeroed arm = no sink
    l[row * pos_stride + h] +=
        __expf(sink_logit - l[row * pos_stride + n_head]);
}

inline void attention_sink_launch(const f32 *sinks, f32 *l, i64 n_head,
                                  i64 pos_stride, i64 n_keys,
                                  hipStream_t stream = nullptr) {
    if (!sinks || !l || n_keys <= 0) return;
    const unsigned threads = 256;
    const unsigned grid = static_cast<unsigned>((n_head + threads - 1) / threads);
    attention_sink_kernel<<<grid, threads, 0, stream>>>(sinks, l, n_head,
                                                        pos_stride, n_keys);
}

// The whole device API, one entry: sink row into the scratch, then the
// kernel whose `sinks` parameter does the rest.
inline void attention_sink(const f32 *sinks, f32 *l, i64 n_head,
                           i64 pos_stride, i64 n_keys,
                           hipStream_t stream = nullptr) {
    attention_sink_launch(sinks, l, n_head, pos_stride, n_keys, stream);
}

inline bool attn_bexact_qt8() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_ATTN_BEXACT_QT");
        return e && e[0] == '8';
    }();
    return v;
}

inline void attention_bexact(void *out, const void *q, const void *kcache,
                             const void *vcache, i64 n_head, i64 n_kv, i64 hd,
                             i64 n_tok, i64 pos0, i64 pos_stride, f32 scale,
                             int lds_bytes, hipStream_t stream = nullptr,
                             const f32 *sinks = nullptr, i64 skip_from = -1) {
    // tile_k is a multiple of the warp count so the per-tile warp stride lands
    // on exactly decode's key partition (j % 8), and small enough for the
    // staged pair to fit the device's LDS with the accumulators' merge buffer.
    int tile_k = 64;
    if (hd >= 256) tile_k = 32;
#define KRK_ATTN_BEXACT(VPTV, QTV)                                            \
    do {                                                                     \
        int kk = tile_k;                                                     \
        while (kk > kAttnDecWarps &&                                         \
               attn_bexact_smem_bytes<VPTV, QTV>(kk, hd) >                    \
                   static_cast<size_t>(lds_bytes))                           \
            kk /= 2;                                                         \
        const size_t bs = attn_bexact_smem_bytes<VPTV, QTV>(kk, hd);         \
        const dim3 g(static_cast<unsigned>(n_head),                          \
                     static_cast<unsigned>((n_tok + (QTV) - 1) / (QTV)));    \
        hipFuncSetAttribute(                                                 \
            reinterpret_cast<const void *>(&attention_bexact_kernel<VPTV, QTV>), \
            hipFuncAttributeMaxDynamicSharedMemorySize,                      \
            static_cast<int>(bs + 512));                                     \
        attention_bexact_kernel<VPTV, QTV>                                   \
            <<<g, kAttnDecThreads, bs, stream>>>(                            \
                static_cast<_Float16 *>(out),                                \
                static_cast<const _Float16 *>(q),                            \
                static_cast<const _Float16 *>(kcache),                       \
                static_cast<const _Float16 *>(vcache), n_head, n_kv, hd,     \
                n_tok, pos0, pos_stride, scale, kk, sinks, skip_from);        \
    } while (0)
    // VPT = hd/32, so every width whose lane slice is a whole number of
    // 32-channel runs lands here: 64..256 in steps of 32. The widths decode has
    // a kernel for (64/128/256) are additionally bit-identical to it; the odd
    // ones (96, 160, 224) have no decode kernel at all, so this is simply their
    // prefill path. QT holds VPT*QT at ~32 registers.
    switch (static_cast<int>(hd)) {
        case 64: KRK_ATTN_BEXACT(2, 8); break;
        case 96: KRK_ATTN_BEXACT(3, 8); break;
        case 128: KRK_ATTN_BEXACT(4, 8); break;
        case 160: KRK_ATTN_BEXACT(5, 4); break;
        case 192: KRK_ATTN_BEXACT(6, 4); break;
        case 224: KRK_ATTN_BEXACT(7, 4); break;
        case 256:
            if (attn_bexact_qt8()) KRK_ATTN_BEXACT(8, 8);
            else KRK_ATTN_BEXACT(8, 4);
            break;
        default: break;
    }
#undef KRK_ATTN_BEXACT
}

// ---------------------------------------------------------------------------
// flash attention — prefill on the matrix cores (hd 256)
//
// The bit-exact kernel above is a decode-shaped formulation on purpose: it
// reproduces attention_decode_kernel's arithmetic so a chunked prefill and a
// decode step agree bit for bit, which is what the chunk-invariance gate pins.
// That identity costs the matrix cores: every (row, key) score is one lane dot,
// so a 512-row chunk runs at a fraction of the card. The figure that motivated
// this file -- "3.16 s of a 10.31 s prefill, 19.8 ms a call" on the 35B-A3B at
// 4096 tokens -- does NOT reproduce on the current tree and must not be quoted:
// re-measured at 4140 tokens with `KRK_TIME=attention`, all 90 calls together
// cost 54.9 ms bit-exact and 75.9 ms on the tiled kernel (0.6-0.8 ms a call),
// i.e. attention is ~0.5% of that prefill, not 30%. Same trap as the `--prefill`
// rung elsewhere in this repo: a device span is only comparable within one rung
// (under `--prefill exact` the same three kernels read 335/417/338 ms), so a
// number lifted without its rung is not a measurement. What survives is the
// ordering -- matrix cores 44.9 ms < bit-exact 54.9 < tiled 75.9 -- and that is
// the whole claim this file makes.
//
// This is the throughput arm for the same shapes, and it is selected first:
// 16 query rows per block, K and V staged once per 32-key group and shared by
// all 16 rows, and BOTH matmuls on v_wmma_f32_16x16x16_f16. It is NOT
// bit-identical to the decode row -- that is the trade -- so it is never the
// arm a hash is recorded from: `--prefill-hash` pins the bit-exact kernel.
// Its own chunk invariance IS exact: every row walks absolute 32-key groups
// from 0, so a row's arithmetic does not depend on how many rows share its
// block or on where the chunk boundary fell.
//
// Why 16 rows and four waves: the 16x16 accumulator gives 8 f32 slots a lane
// for rows 8*(lane>>4)+g at column lane&15, so a 16-row tile is one MMA and a
// row's 16 keys live in the 16 lanes of one half-wave. The PV accumulator is
// 16 rows x 256 channels = 128 f32 a lane if one wave owns it, which is why
// the output is split four ways (64 channels each, 4 n-fragments, 32 f32 a
// lane). V is staged transposed because a B fragment wants its reduction
// index contiguous, and for PV the reduction index is the KEY.
// ---------------------------------------------------------------------------

#if defined(KRK_WMMA_COMPILED)

constexpr int kAwRows = 16;                  // query rows per block
constexpr int kAwKeys = 32;                  // keys per staged group
constexpr int kAwWaves = 4;                  // one 64-channel slice each
constexpr int kAwThreads = kAwWaves * kWaveSize;
constexpr int kAwDim = 256;                  // the head width this is written for
constexpr int kAwSlice = kAwDim / kAwWaves;  // 64 channels a wave
constexpr int kAwPad = 8;                    // stride padding, both layouts
constexpr int kAwStrideHd = kAwDim + kAwPad;
constexpr int kAwStrideKv = kAwKeys + kAwPad;

// The two LDS strides the shared fragment loaders take from a config.
struct aw_cfg_hd {
    static constexpr int BK = kAwDim;
    static constexpr int LD_PAD = kAwPad;
};
struct aw_cfg_kv {
    static constexpr int BK = kAwKeys;
    static constexpr int LD_PAD = kAwPad;
};

__host__ __device__ __forceinline__ size_t attn_wmma_smem_bytes() {
    const size_t f16 = sizeof(_Float16) *
                       (static_cast<size_t>(kAwRows) * kAwStrideHd +
                        static_cast<size_t>(kAwKeys) * kAwStrideHd +
                        static_cast<size_t>(kAwWaves) * kAwSlice * kAwStrideKv +
                        static_cast<size_t>(kAwRows) * kAwStrideKv);
    const size_t f32 = sizeof(f32) * static_cast<size_t>(2 * kAwRows * 16 +
                                                         3 * kAwRows);
    return f16 + f32;
}

__global__ void __launch_bounds__(kAwThreads)
    attention_wmma_kernel(_Float16 *__restrict__ out,
                          const _Float16 *__restrict__ q,
                          const _Float16 *__restrict__ kcache,
                          const _Float16 *__restrict__ vcache, i64 n_head,
                          i64 n_kv, i64 hd, i64 n_tok, i64 pos0,
                          i64 pos_stride, f32 scale,
                          const f32 *__restrict__ sinks) {
    extern __shared__ u8 smem[];
    _Float16 *q_sh = reinterpret_cast<_Float16 *>(smem);
    _Float16 *k_sh = q_sh + kAwRows * kAwStrideHd;
    _Float16 *v_sh = k_sh + kAwKeys * kAwStrideHd;
    _Float16 *p_sh = v_sh + kAwWaves * kAwSlice * kAwStrideKv;
    f32 *s_sh = reinterpret_cast<f32 *>(p_sh + kAwRows * kAwStrideKv);
    f32 *rm_sh = s_sh + 2 * kAwRows * 16; // running max, per query row
    f32 *rl_sh = rm_sh + kAwRows;         // running denominator
    f32 *ra_sh = rl_sh + kAwRows;         // this group's rescale factor

    const i64 h = blockIdx.x;
    const i64 t0 = static_cast<i64>(blockIdx.y) * kAwRows;
    const int n_rows = static_cast<int>(n_tok - t0 < kAwRows ? n_tok - t0
                                                             : kAwRows);
    if (n_rows <= 0) return;
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const i64 n_rep = n_kv > 0 ? n_head / n_kv : 1;
    const i64 hkv = n_rep > 0 ? h / n_rep : 0;
    const i64 kstride = attn_kv_stride(pos_stride);
    const _Float16 *khead = kcache + hkv * hd;
    const _Float16 *vhead = vcache + hkv * hd;
    const _Float16 *qbase = q + (t0 * n_head + h) * hd;
    // The last row of this block sees the most keys; every row masks the rest.
    const i64 n_keys = pos0 + t0 + n_rows;
    const f32 sink_add = sinks ? sinks[h] : 0.0f;
    const bool sunk = sink_add != 0.0f;

    // ---- Q staged once (16 rows x 256) ----------------------------------
    for (int i = tid; i < kAwRows * kAwStrideHd; i += kAwThreads) {
        const int r = i / kAwStrideHd;
        const int d = i % kAwStrideHd;
        q_sh[i] = (r < n_rows && d < kAwDim)
                      ? qbase[static_cast<i64>(r) * n_head * hd + d]
                      : static_cast<_Float16>(0);
    }
    if (tid < kAwRows) {
        rm_sh[tid] = -1.0e30f;
        rl_sh[tid] = 0.0f;
        ra_sh[tid] = 1.0f;
    }
    __syncthreads();

    constexpr int NF = kAwSlice / 16;
    v8float acc[NF]; // rows x this wave's 64 channels
#pragma unroll
    for (int nf = 0; nf < NF; nf++)
#pragma unroll
        for (int g = 0; g < 8; g++) acc[nf][g] = 0.0f;

    for (i64 k0 = 0; k0 < n_keys; k0 += kAwKeys) {
        const int kg = static_cast<int>(
            n_keys - k0 < kAwKeys ? n_keys - k0 : kAwKeys);
        // ---- stage K[kg][256] and this wave's V slice transposed ----------
        for (int i = tid; i < kg * kAwStrideHd; i += kAwThreads) {
            const int j = i / kAwStrideHd;
            const int d = i % kAwStrideHd;
            k_sh[i] = d < kAwDim
                          ? khead[(k0 + j) * kstride + d]
                          : static_cast<_Float16>(0);
        }
        _Float16 *vt = v_sh + static_cast<i64>(wave) * kAwSlice * kAwStrideKv;
        for (int i = lane; i < kAwSlice * kAwStrideKv; i += kWaveSize) {
            const int dloc = i / kAwStrideKv;
            const int kk = i % kAwStrideKv;
            vt[i] = kk < kg
                        ? vhead[(k0 + kk) * kstride + wave * kAwSlice + dloc]
                        : static_cast<_Float16>(0);
        }
        __syncthreads();

        // ---- scores: waves 0 and 1 own one 16-key tile each ---------------
        if (wave < 2) {
            v8float sc;
#pragma unroll
            for (int g = 0; g < 8; g++) sc[g] = 0.0f;
            _Float16 af[kFrag];
            _Float16 bf[kFrag];
            const _Float16 *ks = k_sh + wave * 16 * kAwStrideHd;
            for (int k = 0; k < kAwDim; k += 16) {
                wmma_load_a<aw_cfg_hd>(q_sh, 0, k, lane, af);
                wmma_load_b<aw_cfg_hd>(ks, 0, k, lane, bf);
                wmma_mma(sc, af, bf);
            }
#pragma unroll
            for (int g = 0; g < 8; g++) {
                int i, j;
                wmma_slot(g, lane, &i, &j);
                s_sh[(wave * kAwRows + i) * 16 + j] = sc[g] * scale;
            }
        }
        __syncthreads();

        // ---- online softmax, one thread per query row --------------------
        // Sixteen threads scanning 32 staged scores is a few hundred LDS
        // reads; it is noise beside the 32 MMAs of one group, and it keeps
        // the row state (max, denominator, rescale) in ONE place instead of
        // replicated per wave.
        if (tid < kAwRows) {
            const int r = tid;
            const i64 p = pos0 + t0 + r;
            f32 tmax = -1.0e30f;
            for (int T = 0; T < 2; T++)
                for (int j = 0; j < 16; j++) {
                    const int jj = T * 16 + j;
                    if (jj >= kg || k0 + jj > p) continue;
                    const f32 sc = s_sh[(T * kAwRows + r) * 16 + j];
                    tmax = fmaxf(tmax, sc);
                }
            const f32 mold = rm_sh[r];
            const f32 mnew = fmaxf(mold, tmax);
            const f32 alpha = __expf(mold - mnew);
            f32 sum = 0.0f;
            for (int T = 0; T < 2; T++)
                for (int j = 0; j < 16; j++) {
                    const int jj = T * 16 + j;
                    const bool vis = jj < kg && k0 + jj <= p;
                    const f32 e = vis
                                      ? __expf(s_sh[(T * kAwRows + r) * 16 + j] - mnew)
                                      : 0.0f;
                    p_sh[r * kAwStrideKv + jj] = static_cast<_Float16>(e);
                    sum += e;
                }
            rm_sh[r] = mnew;
            ra_sh[r] = alpha;
            rl_sh[r] = rl_sh[r] * alpha + sum;
        }
        __syncthreads();

        // ---- P x V on the matrix cores, one 64-channel slice per wave -----
        _Float16 af[kFrag];
        _Float16 bf[kFrag];
        f32 al[kFrag];
#pragma unroll
        for (int g = 0; g < 8; g++) {
            int i, j;
            wmma_slot(g, lane, &i, &j);
            al[g] = ra_sh[i];
        }
#pragma unroll
        for (int nf = 0; nf < NF; nf++)
#pragma unroll
            for (int g = 0; g < 8; g++) acc[nf][g] *= al[g];
        for (int kf = 0; kf < 2; kf++) {
            wmma_load_a<aw_cfg_kv>(p_sh, 0, kf * 16, lane, af);
#pragma unroll
            for (int nf = 0; nf < NF; nf++) {
                wmma_load_b<aw_cfg_kv>(vt, nf * 16, kf * 16, lane, bf);
                wmma_mma(acc[nf], af, bf);
            }
        }
        __syncthreads();
    }

    // ---- normalise and store -------------------------------------------
#pragma unroll
    for (int nf = 0; nf < NF; nf++)
#pragma unroll
        for (int g = 0; g < 8; g++) {
            int i, j;
            wmma_slot(g, lane, &i, &j);
            if (i >= n_rows) continue;
            f32 l = rl_sh[i];
            if (sunk) l += __expf(sink_add - rm_sh[i]);
            const f32 inv = l > 0.0f ? 1.0f / l : 0.0f;
            _Float16 *orow = out + ((t0 + i) * n_head + h) * hd;
            orow[wave * kAwSlice + nf * 16 + j] =
                static_cast<_Float16>(acc[nf][g] * inv);
        }
}

// True when the kernel was launched, false when the device cannot hold its
// LDS (then the caller falls through to the bit-exact arm).
inline bool attention_wmma(void *out, const void *q, const void *kcache,
                           const void *vcache, i64 n_head, i64 n_kv, i64 hd,
                           i64 n_tok, i64 pos0, i64 pos_stride, f32 scale,
                           int lds_bytes, hipStream_t stream = nullptr,
                           const f32 *sinks = nullptr) {
    const size_t smem = attn_wmma_smem_bytes();
    if (static_cast<int>(smem) > lds_bytes) return false;
    hipFuncSetAttribute(
        reinterpret_cast<const void *>(&attention_wmma_kernel),
        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    const dim3 g(static_cast<unsigned>(n_head),
                 static_cast<unsigned>((n_tok + kAwRows - 1) / kAwRows));
    attention_wmma_kernel<<<g, kAwThreads, smem, stream>>>(
        static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
        static_cast<const _Float16 *>(kcache), static_cast<const _Float16 *>(vcache),
        n_head, n_kv, hd, n_tok, pos0, pos_stride, scale, sinks);
    return true;
}

// Whether this build has the kernel at all: the widths and formats it is
// written for, and a device that compiles the gfx12 matrix core it uses.
inline bool attention_wmma_ok(i64 hd, bool kv_int8, i64 window) {
    return hd == kAwDim && !kv_int8 && window == 0;
}

#else // !KRK_WMMA_COMPILED

inline bool attention_wmma_ok(i64, bool, i64) { return false; }

inline bool attention_wmma(void *, const void *, const void *, const void *, i64,
                           i64, i64, i64, i64, i64, f32, int,
                           hipStream_t = nullptr, const f32 * = nullptr) {
    return false;
}

#endif

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

template <int VPT, bool DumpQ = false>
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
                             f32 rope_scale, f32 scale, f32 frac,
                             _Float16 *qdebug = nullptr) {
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
        // Full NeoX pairing: i with i + half. The backend refuses partial
        // rotation because its partner index is i + rot, not i + half.
        // One thread owns both outputs of a rotation pair, as in rope_kernel.
        // Materialize the f16 row in LDS before repacking the attention lanes.
        __shared__ _Float16 qr[128];
        for (i64 i = tid; i < half; i += blockDim.x) {
            f32 a = static_cast<f32>(q[h * hd + i]);
            f32 b = static_cast<f32>(q[h * hd + i + half]);
            if (wq) {
                a = static_cast<f32>(nv(q[h * hd + i], qscale, wq[i]));
                b = static_cast<f32>(nv(q[h * hd + i + half], qscale, wq[i + half]));
            }
            const f32 theta = pf * inv_freq[i];
            f32 sn, cs;
            __sincosf(theta, &sn, &cs);
            qr[i] = static_cast<_Float16>(a * cs - b * sn);
            qr[i + half] = static_cast<_Float16>(a * sn + b * cs);
        }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < VPT / 2; e++) {
            Pair o;
            o.h[0] = qr[lane * VPT + 2 * e];
            o.h[1] = qr[lane * VPT + 2 * e + 1];
            qp[e] = o.u;
        }
    }

    if (DumpQ && warp == 0) {
#pragma unroll
        for (int e = 0; e < VPT / 2; ++e)
            *reinterpret_cast<u32 *>(qdebug + h * hd + lane * VPT + 2 * e) = qp[e];
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
