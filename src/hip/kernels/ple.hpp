// ============================================================================
//  ple.hpp — qwen4exp's per-layer embedding block: the two ops in it that are
//  neither a matmul nor an elementwise pass.
//
//  A PLE layer reads N n-gram table rows per token (the hash runs host-side,
//  see include/krk/ple.hpp), projects them to a key and a value, and gates the
//  value with a dot product against the LANE STACK's own normed rows:
//
//      s[t,c] = sum_e key[t,c,e] * qry[t,c,e] / sqrt(ne)
//      gate   = sigmoid(sign(s) * sqrt(|s|))
//
//  so the reduction spans one lane's n_embd channels and there are hc of them
//  per token -- a row reduction would sum the lanes together, which is exactly
//  what this must not do. The gate is written LANE-major so the caller can read
//  one lane's column contiguously.
//
//  The convolution is depthwise and dilated along the TOKEN axis: tap k scales
//  channel c by w[k, c] and reaches `(kern-1-k) * dil` tokens back, with dil the
//  n-gram size. The caller supplies the window (history rows in front of this
//  pass's rows), because a chunked prefill and a token-at-a-time decode have to
//  agree -- that is the whole property the history exists for.
// ============================================================================
#ifndef KRK_KERNEL_PLE_HPP
#define KRK_KERNEL_PLE_HPP

#include "../krk_hip.hpp"

namespace krk {

// out[c*n_tok + t] = sigmoid(sign(s) * sqrt(max(|s|, 1e-6))) with
// s = (sum_e key*qry) / sqrt(ne). One block per (t, c): the reduction is over a
// lane's whole n_embd (10240 on the REAP-288 file), too long for one thread.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    ple_gate_kernel(_Float16 *__restrict__ out, const _Float16 *__restrict__ key,
                    const _Float16 *__restrict__ qry, i64 n_tok, i64 hc,
                    i64 ne) {
    __shared__ f32 sdata[THREADS];
    const i64 tc = static_cast<i64>(blockIdx.x);
    const i64 t = tc / hc;
    const i64 c = tc - t * hc;
    const i64 base = (t * hc + c) * ne;
    f32 acc = 0.0f;
    for (i64 e = threadIdx.x; e < ne; e += THREADS) {
        acc += static_cast<f32>(key[base + e]) * static_cast<f32>(qry[base + e]);
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int step = THREADS / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step)
            sdata[threadIdx.x] += sdata[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    const f32 s = sdata[0] * rsqrtf(static_cast<f32>(ne));
    const f32 m = fabsf(s) > 1e-6f ? fabsf(s) : 1e-6f;
    const f32 signed_mag = (s < 0.0f ? -1.0f : 1.0f) * sqrtf(m);
    out[c * n_tok + t] =
        static_cast<_Float16>(1.0f / (1.0f + __expf(-signed_mag)));
}

// out[t, c] = sum_k w[k*ch + c] * x[t + (kern-1-k)*dil, c], one thread per
// (t, c). `x` carries (kern-1)*dil history rows in front of this pass's rows, so
// the taps never read outside the window the caller built.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    conv1d_dw_kernel(_Float16 *__restrict__ out, const _Float16 *__restrict__ x,
                     const f32 *__restrict__ w, i64 n_tok, i64 ch, i64 kern,
                     i64 dil) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tok * ch) return;
    const i64 t = i / ch;
    const i64 c = i - t * ch;
    f32 acc = 0.0f;
    for (i64 k = 0; k < kern; k++) {
        const i64 row = t + (kern - 1 - k) * dil;
        acc += static_cast<f32>(x[row * ch + c]) * w[k * ch + c];
    }
    out[i] = static_cast<_Float16>(acc);
}

// out[t, c, e] = value[t, e] * g[c*n_tok + t], one thread per output element.
// `value` is one row per token (the block's value projection, shared by every
// lane) and `g` is lane-major, which is the layout ple_gate_kernel writes.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    lane_bcast_mul_kernel(_Float16 *__restrict__ out,
                          const _Float16 *__restrict__ value,
                          const _Float16 *__restrict__ g, i64 n_tok, i64 hc,
                          i64 ne) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tok * hc * ne) return;
    const i64 t = i / (hc * ne);
    const i64 c = (i / ne) % hc;
    const i64 e = i % ne;
    out[i] = static_cast<_Float16>(static_cast<f32>(value[t * ne + e]) *
                                   static_cast<f32>(g[c * n_tok + t]));
}

} // namespace krk

#endif // KRK_KERNEL_PLE_HPP
