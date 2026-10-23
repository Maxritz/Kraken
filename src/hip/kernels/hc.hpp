// ============================================================================
//  hc.hpp — hyper-connection (qwen4exp) lane stacking, mixing and scattering.
//
//  A qwen4exp stack keeps `hc` residual streams of n_embd each, token-major as
//  [n_tok, hc, n_embd]. Three of the four operations a block needs are plain
//  elementwise or a reduction ACROSS the lane axis, which no other kernel here
//  expresses: the lane axis is inside a token's row, so it is neither a row
//  reduction (rmsnorm) nor a column broadcast (add_bias_cols).
//
//  The arithmetic order is the host oracle's (krk::hc_mix / krk::hc_combine in
//  src/hc.cpp), not a convenient one: lanes are accumulated c = 0..hc-1 in f32
//  and the mean divides ONCE at the end, because with all lanes equal that is
//  what keeps the mixed stream exactly `xn * sigmoid(gate)` and the lane stack
//  bit-equal after a combine whose injection is zero. `kraken-bench --hc`
//  compares these kernels against that oracle, and the suite pins the invariant
//  on the CPU backend.
// ============================================================================
#ifndef KRK_KERNEL_HC_HPP
#define KRK_KERNEL_HC_HPP

#include "../krk_hip.hpp"

namespace krk {

// lanes[t, c, e] <- x[t, e]: every lane starts as a copy of the embedding row
// (the reference's repeat_4d), which is what makes the first mix a single-lane
// computation.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    hc_init_kernel(_Float16 *__restrict__ lanes, const _Float16 *__restrict__ x,
                   i64 n_tok, i64 hc, i64 ne) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tok * hc * ne) return;
    const i64 e = i % ne;
    const i64 tc = i / ne;
    const i64 t = tc / hc;
    lanes[i] = x[t * ne + e];
}

// mixed[t, e] <- (1/hc) * sum_c xn[t, c, e] * sigmoid(gate[t, c, e]) with the
// gate pre-sigmoid. One thread per (t, e); the lane loop is the whole reduction
// and it is short (hc is 4 on every file that has one).
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    hc_mix_kernel(_Float16 *__restrict__ mixed, const _Float16 *__restrict__ xn,
                  const _Float16 *__restrict__ gate, i64 n_tok, i64 hc,
                  i64 ne) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tok * ne) return;
    const i64 t = i / ne;
    const i64 e = i - t * ne;
    f32 acc = 0.0f;
    for (i64 c = 0; c < hc; c++) {
        const i64 k = (t * hc + c) * ne + e;
        const f32 g = 1.0f / (1.0f + __expf(-static_cast<f32>(gate[k])));
        acc += static_cast<f32>(xn[k]) * g;
    }
    acc /= static_cast<f32>(hc);
    mixed[i] = d_sat_f16_count(acc, &g_krk_sat_sum);
}

// res[t, c, e] += block_out[t, e] * (2 * sigmoid(inject[t, c] / hc)).
// The 2*sigmoid centring makes a zero injection a plain residual add on every
// lane, so a block that outputs nothing leaves the lanes untouched and equal.
// Writes are disjoint per (t, c, e); no atomics, no ordering between threads.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    hc_combine_kernel(_Float16 *__restrict__ res,
                      const _Float16 *__restrict__ block_out,
                      const _Float16 *__restrict__ inject, i64 n_tok, i64 hc,
                      i64 ne) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tok * ne) return;
    const i64 t = i / ne;
    const i64 e = i - t * ne;
    const f32 b = static_cast<f32>(block_out[i]);
    for (i64 c = 0; c < hc; c++) {
        const i64 k = (t * hc + c) * ne + e;
        const f32 inj = static_cast<f32>(inject[t * hc + c]);
        const f32 w = 2.0f / (1.0f + __expf(-inj / static_cast<f32>(hc)));
        res[k] = d_sat_f16_count(static_cast<f32>(res[k]) + b * w,
                                 &g_krk_sat_resid);
    }
}

// x <- x * sigmoid(x) in place: the mixer's low-rank nonlinearity. silu_mul()
// cannot stand in for it (it needs a second operand) and neither can the
// elementwise sigmoid, which drops the x factor.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    silu_kernel(_Float16 *__restrict__ x, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = d_sat_f16_count(d_silu(static_cast<f32>(x[i])), &g_krk_sat_silu);
}

// x <- gelu(x) in place (gemma's GeGLU gate). Same shape as silu_kernel above,
// and it lives here for the same reason: the elementwise activation kernels are
// one launch per FFN and there is no other header that owns them.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    gelu_kernel(_Float16 *__restrict__ x, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = d_sat_f16_count(d_gelu(static_cast<f32>(x[i])), &g_krk_sat_silu);
}

} // namespace krk

#endif // KRK_KERNEL_HC_HPP
