// ============================================================================
//  mla.hpp — multi-head latent attention (DeepSeek-V2/V3/V4), device side.
//
//  The cache row is NOT n_kv x head_dim. One row per token holds the compressed
//  latent (kv_lora, normed at write time) followed by the decoupled RoPE key
//  (qk_rope, stored UNROTATED so each row can be rotated at its own position).
//
//  Attention runs on the latent, and the two per-head projections are folded
//  where they cost the least:
//
//    qk_h = Wk_h^T q_nope_h                       absorb, per call
//    s_t  = qk_h . c_t + rope(q_pe_h) . rope(k_pe_t)
//    p    = softmax(s * scale)
//    cv   = sum_t p_t c_t                         still latent width
//    out_h = Wv_h^T cv
//
//  `attn_kv_b` is the legacy unsplit table, and its file shape is
//  [kv_lora, n_head*(nope+v_dim)] with GGUF's fastest axis the latent: feature
//  j = h*(nope+v) + f owns the row at offset j*kv_lora, the latent k contiguous
//  inside it. Head h therefore owns a COLUMN slice of a row-major matrix, which
//  is why neither the query absorb nor the value fold is a `gemm()` here -- and
//  why the table is staged once at load as f16 in exactly this order
//  (LayerWeights::wkv_b16, Model::load), so the kernel's indices ARE the file's
//  and a mismatch is arithmetic rather than a different weight. The host oracle
//  in src/mla.cpp takes the same table HEAD-major ([n_head][kv_lora][hv]); the
//  bench arm transposes when it builds the oracle's arrays, which is the one
//  place the two conventions meet.
//
//  One block per (query row, head). That is 27 blocks on a decode step (a
//  V2-Lite stack) and 27 x chunk on a prefill, and each block owns its whole
//  softmax, so no cross-block merge exists to get wrong.
// ============================================================================
#ifndef KRK_KERNEL_MLA_HPP
#define KRK_KERNEL_MLA_HPP

#include "../krk_hip.hpp"
#include "normalize.hpp" // block_reduce_sum

namespace krk {

// Block-wide max, the attention kernels' own reduction (normalize.hpp has the
// wave/sum pair; this one is the max every softmax here needs).
template <int THREADS>
__device__ __forceinline__ f32 mla_block_max(f32 v) {
    __shared__ f32 red[THREADS / 32];
    __shared__ f32 total;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wave = static_cast<int>(threadIdx.x) >> 5;
    v = d_wave_reduce_max(v);
    if (lane == 0) red[wave] = v;
    __syncthreads();
    if (wave == 0) {
        f32 t = lane < (THREADS / 32) ? red[lane] : -1e30f;
        t = d_wave_reduce_max(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    return total;
}

// cache[pos0 + t] = [ rmsnorm(proj[t][0..kv_lora)) * gamma | proj[t][kv_lora..) ]
// One block per row: the norm is over the latent half alone (the decoupled key
// is not normed), and the key half is copied UNROTATED.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    mla_kv_write_kernel(_Float16 *__restrict__ cache,
                        const _Float16 *__restrict__ proj,
                        const f32 *__restrict__ gamma, i64 kv_lora, i64 rope,
                        i64 pos0, i64 pos_stride, f32 eps) {
    const i64 t = blockIdx.x;
    const i64 width = kv_lora + rope;
    const _Float16 *src = proj + t * width;
    _Float16 *dst = cache + (pos0 + t) * pos_stride;

    f32 ss = 0.0f;
    for (i64 i = threadIdx.x; i < kv_lora; i += THREADS) {
        const f32 v = static_cast<f32>(src[i]);
        ss += v * v;
    }
    ss = block_reduce_sum<THREADS>(ss);

    __shared__ f32 scale_sh;
    if (threadIdx.x == 0)
        scale_sh = rsqrtf(ss / static_cast<f32>(kv_lora) + eps);
    __syncthreads();
    const f32 scale = scale_sh;

    for (i64 i = threadIdx.x; i < kv_lora; i += THREADS)
        dst[i] = static_cast<_Float16>(static_cast<f32>(src[i]) * scale * gamma[i]);
    for (i64 i = threadIdx.x; i < rope; i += THREADS)
        dst[kv_lora + i] = src[kv_lora + i];
}

// out[r][h*v_dim + d] = Wv_h^T ( sum_t p_t c_t ), p from the absorbed score.
// grid = (n_tok, n_head); dynamic shared = (2*kv_lora + rope + n_keys_max) f32.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    mla_attention_kernel(_Float16 *__restrict__ out,
                         const _Float16 *__restrict__ q,
                         const _Float16 *__restrict__ cache,
                         const _Float16 *__restrict__ wkv, i64 n_head,
                         i64 qk, i64 nope, i64 rope, i64 kv_lora, i64 v_dim,
                         i64 pos0, i64 pos_stride, f32 scale,
                         const f32 *__restrict__ inv_freq,
                         i64 n_keys_max) {
    // Named for this family: every kernel lands in one HIP translation unit,
    // and `extern __shared__` is one symbol per TYPE -- a second `smem` here
    // redeclares the one in attention.hpp with a different type.
    extern __shared__ f32 mla_smem[];
    f32 *s_qk = mla_smem;         // [kv_lora]  absorbed query (K side)
    f32 *s_cv = s_qk + kv_lora;   // [kv_lora]  accumulated latent
    f32 *s_qp = s_cv + kv_lora;   // [rope]     rotated query key
    f32 *s_p = s_qp + rope;       // [n_keys]   scores, then probabilities

    const i64 r = blockIdx.x;
    const i64 h = blockIdx.y;
    const i64 pos = pos0 + r;
    const i64 n_keys = pos + 1;   // causal: this row sees its own prefix
    const i64 hv = nope + v_dim;
    (void)n_keys_max;

    const _Float16 *qrow = q + r * (n_head * qk) + h * qk;
    const _Float16 *qn = qrow;
    const _Float16 *qr = qrow + nope;

    // 1. the query's rope half, rotated at this row's position
    for (i64 i = threadIdx.x; i < rope / 2; i += THREADS) {
        const f32 th = static_cast<f32>(pos) * inv_freq[i];
        f32 sn, cs;
        __sincosf(th, &sn, &cs);
        const f32 a = static_cast<f32>(qr[2 * i]);
        const f32 b = static_cast<f32>(qr[2 * i + 1]);
        s_qp[2 * i] = a * cs - b * sn;
        s_qp[2 * i + 1] = a * sn + b * cs;
    }
    // 2. the absorbed query: one [kv_lora] vector per head
    //
    // attn_kv_b is stored [kv_lora, n_head*(nope+v)] and GGUF's fastest axis is
    // the latent, so feature (h*(nope+v) + d) owns the ROW at offset
    // (h*hv+d)*kv_lora and the latent is contiguous inside it. Thread k reads
    // that row at column k, so a wave reads 32 consecutive latents of one
    // feature: the access is coalesced, and the index arithmetic is the file's.
    for (i64 k = threadIdx.x; k < kv_lora; k += THREADS) {
        f32 s = 0.0f;
        for (i64 d = 0; d < nope; d++)
            s += static_cast<f32>(wkv[(h * hv + d) * kv_lora + k]) *
                 static_cast<f32>(qn[d]);
        s_qk[k] = s;
    }
    __syncthreads();

    // 3. scores over the prefix: latent dot plus the decoupled key
    f32 mx = -1e30f;
    for (i64 t = threadIdx.x; t < n_keys; t += THREADS) {
        const _Float16 *c = cache + t * pos_stride;
        f32 s = 0.0f;
        for (i64 k = 0; k < kv_lora; k++)
            s += s_qk[k] * static_cast<f32>(c[k]);
        const _Float16 *kpe = c + kv_lora;
        for (i64 i = 0; i < rope / 2; i++) {
            const f32 th = static_cast<f32>(t) * inv_freq[i];
            f32 sn, cs;
            __sincosf(th, &sn, &cs);
            const f32 a = static_cast<f32>(kpe[2 * i]);
            const f32 b = static_cast<f32>(kpe[2 * i + 1]);
            s += s_qp[2 * i] * (a * cs - b * sn) +
                 s_qp[2 * i + 1] * (a * sn + b * cs);
        }
        s *= scale;
        s_p[t] = s;
        mx = fmaxf(mx, s);
    }
    mx = mla_block_max<THREADS>(mx);

    // 4. exponentiate against the block max, and total
    f32 sum = 0.0f;
    for (i64 t = threadIdx.x; t < n_keys; t += THREADS) {
        const f32 e = __expf(s_p[t] - mx);
        s_p[t] = e;
        sum += e;
    }
    sum = block_reduce_sum<THREADS>(sum);
    __syncthreads();   // s_p is read by every thread in step 5

    // 5. accumulate the latent itself, one latent channel per thread
    for (i64 k = threadIdx.x; k < kv_lora; k += THREADS) {
        f32 a = 0.0f;
        for (i64 t = 0; t < n_keys; t++)
            a += s_p[t] * static_cast<f32>(cache[t * pos_stride + k]);
        s_cv[k] = a / sum;
    }
    __syncthreads();

    // 6. the value half of the same table, folded into the accumulated latent
    //
    // Feature (h*hv + nope + d) owns one contiguous [kv_lora] row, so a thread
    // streams it in order -- the same bytes the quantized table holds, now f16.
    for (i64 d = threadIdx.x; d < v_dim; d += THREADS) {
        const _Float16 *wcol = wkv + (h * hv + nope + d) * kv_lora;
        f32 a = 0.0f;
        for (i64 k = 0; k < kv_lora; k++)
            a += s_cv[k] * static_cast<f32>(wcol[k]);
        out[r * (n_head * v_dim) + h * v_dim + d] = static_cast<_Float16>(a);
    }
}

} // namespace krk

#endif // KRK_KERNEL_MLA_HPP
