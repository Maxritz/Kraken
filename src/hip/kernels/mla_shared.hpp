// ============================================================================
//  mla_shared.hpp — DeepSeek-V4 shared-latent MLA, device side.
//
//  A different graph from kernels/mla.hpp, not a variant of it. One normed
//  latent row per token is the cache:
//
//      c   = rmsnorm(x @ Wkv) * gamma                [kv_lora]  <- the row
//      K_h = rope_tail(copy of c, pos)               (the tail rotates)
//      V   = c                                        (unrotated)
//      s_h = q_h . K_h * scale                        (+ sink, denominator only)
//      a_h = sum_t softmax(s_h)_t c_t                 [kv_lora]
//      y_g = Woa_g (concat of group g's a_h)          [o_lora]
//      out = Wob y                                    [n_embd]
//
//  Three launches, because the middle one is per (row, head) and the last two
//  are per row (and per row, group): attention writes the head stack, then the
//  grouped output folds it. `q` arrives already projected ([n_head*kv_lora]
//  per row) because the query trunk is two ordinary GEMMs; only the rotary tail
//  of the query is work this file owns, and rotating it here is what keeps the
//  cache unrotated so the value side can use the row as it is.
//
//  Positions are the cache slots (dense cache), which is the engine's layout:
//  row t of the cache is position t. The oracle takes an explicit kpos array
//  because the suite compacts a cache to exercise window and causality; that
//  flexibility is not expressible here and does not need to be -- the engine
//  never compacts for this layout (QSA refuses with it).
//
//  Same `mla_smem` symbol as kernels/mla.hpp on purpose: `extern __shared__` is
//  one symbol per TYPE in a translation unit, so a distinct name for the same
//  f32 type is a redefinition, not an alias.
// ============================================================================
#ifndef KRK_KERNEL_MLA_SHARED_HPP
#define KRK_KERNEL_MLA_SHARED_HPP

#include "../krk_hip.hpp"
#include "normalize.hpp" // block_reduce_sum

namespace krk {

template <int THREADS>
__device__ __forceinline__ f32 mla_shared_block_max(f32 v) {
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

// cache[pos0 + t][i] = rmsnorm(proj[t])[i] * gamma[i], the WHOLE row. One block
// per row: the norm covers the rotary tail too, which is the layout difference
// that a kernel copied from mla.hpp would get wrong (there the tail is a
// decoupled key and is explicitly NOT normed).
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    mla_shared_latent_kernel(_Float16 *__restrict__ cache,
                             const _Float16 *__restrict__ proj,
                             const f32 *__restrict__ gamma, i64 kv_lora,
                             i64 pos0, i64 pos_stride, f32 eps) {
    const i64 t = blockIdx.x;
    const _Float16 *src = proj + t * kv_lora;
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
        dst[i] = static_cast<_Float16>(static_cast<f32>(src[i]) * scale *
                                       gamma[i]);
}

// a[r][h*kv_lora + i] = sum_t p_t c_t[i], p from the rotated latent's dot with
// head h's query. grid = (n_tok, n_head); dynamic shared =
// (kv_lora + kv_lora + n_keys_max) f32: the rotated query, one rotated key, and
// that head's scores.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    mla_shared_attention_kernel(f32 *__restrict__ a,
                                const _Float16 *__restrict__ q,
                                const _Float16 *__restrict__ cache, i64 n_head,
                                i64 kv_lora, i64 rope, i64 pos0, i64 pos_stride,
                                f32 scale, i32 window,
                                const f32 *__restrict__ inv_freq,
                                const f32 *__restrict__ sinks,
                                i64 n_keys_max) {
    extern __shared__ f32 mla_smem[];
    f32 *s_q = mla_smem;                 // [kv_lora] query, rotated at `pos`
    f32 *s_p = s_q + kv_lora;            // [n_keys]  scores, then weights

    const i64 r = blockIdx.x;
    const i64 h = blockIdx.y;
    const i64 pos = pos0 + r;
    (void)n_keys_max;

    // 1. the query, its rotary TAIL rotated at this row's position
    const _Float16 *qrow = q + r * (n_head * kv_lora) + h * kv_lora;
    for (i64 i = threadIdx.x; i < kv_lora; i += THREADS)
        s_q[i] = static_cast<f32>(qrow[i]);
    __syncthreads();
    const i64 base = kv_lora - rope;
    for (i64 i = threadIdx.x; i < rope / 2; i += THREADS) {
        const f32 th = static_cast<f32>(pos) * inv_freq[i];
        f32 sn, cs;
        __sincosf(th, &sn, &cs);
        const f32 u = s_q[base + 2 * i];
        const f32 v = s_q[base + 2 * i + 1];
        s_q[base + 2 * i] = u * cs - v * sn;
        s_q[base + 2 * i + 1] = u * sn + v * cs;
    }
    __syncthreads();

    // 2. scores over the keys this row may see: never the future, and inside
    // the window when the layer has one.
    //
    // The key is rope-rotated at ITS position and the query at the row's, so
    // the score is R_pos(q) . R_t(c). Rotations commute, so that is
    // R_{pos-t}(q) . c -- and un-rotating the (already rotated) query per key
    // keeps the whole dot in registers. The obvious alternative, staging each
    // key in one shared buffer, is a data race: two keys mean two threads
    // writing the same buffer, which is exactly how this arm failed at every
    // row past the first (one key is one writer, and that row was exact).
    const i64 t_lo = window > 0 ? (pos - window + 1 > 0 ? pos - window + 1 : 0) : 0;
    f32 mx = -1e30f;
    for (i64 t = t_lo + threadIdx.x; t <= pos; t += THREADS) {
        const _Float16 *c = cache + t * pos_stride;
        f32 s = 0.0f;
        for (i64 i = 0; i < base; i++) s += s_q[i] * static_cast<f32>(c[i]);
        for (i64 i = 0; i < rope / 2; i++) {
            const f32 th = static_cast<f32>(t) * inv_freq[i];
            const f32 cs = __cosf(th), sn = __sinf(th);
            const f32 u = s_q[base + 2 * i];
            const f32 v = s_q[base + 2 * i + 1];
            s += (u * cs + v * sn) * static_cast<f32>(c[base + 2 * i]) +
                 (-u * sn + v * cs) * static_cast<f32>(c[base + 2 * i + 1]);
        }
        s *= scale;
        s_p[t - t_lo] = s;
        mx = fmaxf(mx, s);
    }
    // The sink enters the DENOMINATOR only, at the running max: it can take
    // probability mass away from every key and is never attended.
    if (sinks) mx = fmaxf(mx, sinks[h]);
    mx = mla_shared_block_max<THREADS>(mx);

    f32 sum = 0.0f;
    for (i64 t = t_lo + threadIdx.x; t <= pos; t += THREADS) {
        const f32 e = __expf(s_p[t - t_lo] - mx);
        s_p[t - t_lo] = e;
        sum += e;
    }
    // The sink is ONE term in the denominator, so exactly one thread adds it --
    // and only after being subtracted from this row's max, or a large sink
    // would make every probability overflow under it.
    if (sinks && threadIdx.x == 0) sum += __expf(sinks[h] - mx);
    sum = block_reduce_sum<THREADS>(sum);
    __syncthreads();

    // 3. accumulate the latent itself, one channel per thread
    f32 *out = a + r * (n_head * kv_lora) + h * kv_lora;
    const i64 n_keys = pos - t_lo + 1;
    for (i64 i = threadIdx.x; i < kv_lora; i += THREADS) {
        f32 acc = 0.0f;
        for (i64 t = 0; t < n_keys; t++)
            acc += s_p[t] * static_cast<f32>(cache[(t_lo + t) * pos_stride + i]);
        out[i] = acc / sum;
    }
}

// y[row][g*o_lora + j] = sum_i Woa[(g*o_lora + j)][i] * a[row][g*hpg*kv + i].
// One block per (row, group): the group's column block is contiguous in the
// file's [out][in] table, so its rows start at (g*o_lora)*gin.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    mla_shared_wo_a_kernel(f32 *__restrict__ y, const f32 *__restrict__ a,
                           const _Float16 *__restrict__ wo_a, i64 n_head,
                           i64 kv_lora, i64 o_groups, i64 o_lora) {
    const i64 r = blockIdx.x;
    const i64 g = blockIdx.y;
    const i64 gin = (n_head / o_groups) * kv_lora;
    const f32 *src = a + r * (n_head * kv_lora) + g * gin;
    const _Float16 *w = wo_a + g * o_lora * gin;
    f32 *dst = y + r * (o_groups * o_lora) + g * o_lora;
    for (i64 j = threadIdx.x; j < o_lora; j += THREADS) {
        f32 s = 0.0f;
        for (i64 i = 0; i < gin; i++)
            s += static_cast<f32>(w[j * gin + i]) * src[i];
        dst[j] = s;
    }
}

// out[row][o] = sum_j Wob[o][j] * y[row][j]. One block per row.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    mla_shared_wo_b_kernel(_Float16 *__restrict__ out,
                           const f32 *__restrict__ y,
                           const _Float16 *__restrict__ wo_b, i64 n_embd,
                           i64 gout) {
    const i64 r = blockIdx.x;
    const f32 *src = y + r * gout;
    _Float16 *dst = out + r * n_embd;
    for (i64 o = threadIdx.x; o < n_embd; o += THREADS) {
        f32 s = 0.0f;
        for (i64 j = 0; j < gout; j++)
            s += static_cast<f32>(wo_b[o * gout + j]) * src[j];
        dst[o] = static_cast<_Float16>(s);
    }
}

} // namespace krk

#endif // KRK_KERNEL_MLA_SHARED_HPP
