// ============================================================================
//  normalize.hpp — normalization, positional encoding and elementwise kernels.
//
//  Everything is fused wherever fusion is free: the RMSNorm keeps the sum of
//  squares in a wave reduction and folds the weight multiply into the store,
//  RoPE computes its sin/cos at full fp32 and only narrows at the store.
// ============================================================================
#ifndef KRK_KERNEL_NORMALIZE_HPP
#define KRK_KERNEL_NORMALIZE_HPP

#include "../krk_hip.hpp"

namespace krk {

// ---------------------------------------------------------------------------
// block reduction helpers
// ---------------------------------------------------------------------------

// Every thread gets the total back, not just wave 0 — the attention kernel
// feeds the block sum straight into its running normalizer on all lanes, so a
// wave-local partial would corrupt the softmax denominator.
template <int THREADS>
__device__ __forceinline__ f32 block_reduce_sum(f32 v) {
    __shared__ f32 red[THREADS / 32];
    __shared__ f32 total;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wave = static_cast<int>(threadIdx.x) >> 5;

    v = d_wave_reduce_sum(v);
    if (lane == 0) red[wave] = v;
    __syncthreads();
    if (wave == 0) {
        f32 t = lane < (THREADS / 32) ? red[lane] : 0.0f;
        t = d_wave_reduce_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------
// RMSNorm: out[r][i] = x[r][i] * rsqrt(mean(x^2) + eps) * w[i]
// One thread block per row. The weight vector is re-read per row, which is
// fine: it is one row of n_embd and stays resident in L1.
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    rmsnorm_kernel(_Float16 *__restrict__ out, const _Float16 *__restrict__ x,
                   const f32 *__restrict__ w, i64 n, f32 eps) {
    const i64 r = blockIdx.x;
    const _Float16 *xr = x + r * n;
    _Float16 *orow = out + r * n;

    f32 ss = 0.0f;
    for (i64 i = threadIdx.x; i < n; i += THREADS)
        ss += static_cast<f32>(xr[i]) * static_cast<f32>(xr[i]);
    ss = block_reduce_sum<THREADS>(ss);

    __shared__ f32 scale_sh;
    if (threadIdx.x == 0)
        scale_sh = rsqrtf(ss / static_cast<f32>(n) + eps);
    __syncthreads();
    const f32 scale = scale_sh;

    for (i64 i = threadIdx.x; i < n; i += THREADS)
        orow[i] = static_cast<_Float16>(static_cast<f32>(xr[i]) * scale * w[i]);
}

// ---------------------------------------------------------------------------
// Per-head RMSNorm (Qwen3 / Gemma3 QK-norm). grid.x = heads, grid.y = tokens.
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    head_norm_kernel(_Float16 *__restrict__ x, const f32 *__restrict__ w,
                     i64 n_heads, i64 hd, f32 eps) {
    const i64 h = blockIdx.x;
    const i64 t = blockIdx.y;
    _Float16 *xh = x + (t * n_heads + h) * hd;

    f32 ss = 0.0f;
    for (i64 i = threadIdx.x; i < hd; i += THREADS)
        ss += static_cast<f32>(xh[i]) * static_cast<f32>(xh[i]);
    ss = block_reduce_sum<THREADS>(ss);

    __shared__ f32 scale_sh;
    if (threadIdx.x == 0)
        scale_sh = rsqrtf(ss / static_cast<f32>(hd) + eps);
    __syncthreads();
    const f32 scale = scale_sh;

    for (i64 i = threadIdx.x; i < hd; i += THREADS)
        xh[i] = static_cast<_Float16>(static_cast<f32>(xh[i]) * scale * w[i]);
}

// ---------------------------------------------------------------------------
// RoPE (interleaved convention, original LLaMA / llama.cpp).
// grid.x = heads, grid.y = tokens. Pairs adjacent channels (2i, 2i+1),
// matching the layout the HF exporter writes into GGUF q/k tensors.
//
// `frac` limits rotation to a prefix of the (hd/2) pairs, which is what Phi-2
// and Gemma's partial RoPE need. `scale` divides the position, i.e. the linear
// RoPE scaling factor.
// ---------------------------------------------------------------------------

__global__ void rope_kernel(_Float16 *__restrict__ x, i64 n_heads, i64 hd,
                            const i64 *__restrict__ pos0,
                            const f32 *__restrict__ inv_freq, f32 scale,
                            f32 frac) {
    const i64 h = blockIdx.x;
    const i64 t = blockIdx.y;
    const i64 half = hd / 2;
    const i64 rot = static_cast<i64>(static_cast<f32>(half) * frac);
    // pos0 lives on the device: a captured HIP graph replays this
    // kernel without re-binding its arguments, so the position has
    // to be readable from device memory at replay time.
    const f32 p = static_cast<f32>(*pos0 + t) / (scale > 0.0f ? scale : 1.0f);
    _Float16 *xh = x + (t * n_heads + h) * hd;

    for (i64 i = threadIdx.x; i < rot && i < half; i += blockDim.x) {
        const f32 theta = p * inv_freq[i];
        f32 sn, cs;
        __sincosf(theta, &sn, &cs);
        const f32 a = static_cast<f32>(xh[2 * i]);
        const f32 b = static_cast<f32>(xh[2 * i + 1]);
        xh[2 * i] = static_cast<_Float16>(a * cs - b * sn);
        xh[2 * i + 1] = static_cast<_Float16>(a * sn + b * cs);
    }
}

// ---------------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------------

__global__ void silu_mul_kernel(_Float16 *__restrict__ out,
                                const _Float16 *__restrict__ gate,
                                const _Float16 *__restrict__ up, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<_Float16>(d_silu(static_cast<f32>(gate[i])) *
                                   static_cast<f32>(up[i]));
}

__global__ void add_inplace_kernel(_Float16 *__restrict__ a,
                                   const _Float16 *__restrict__ b, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = static_cast<_Float16>(static_cast<f32>(a[i]) + static_cast<f32>(b[i]));
}

__global__ void add_bias_rows_kernel(_Float16 *__restrict__ out,
                                     const f32 *__restrict__ bias, i64 n,
                                     i64 total) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = static_cast<_Float16>(static_cast<f32>(out[i]) +
                                   bias[i % n]);
}

__global__ void copy_h2h_kernel(_Float16 *__restrict__ dst,
                                const _Float16 *__restrict__ src, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

// dst += alpha * src, used to weight routed-expert outputs in the MoE path.
__global__ void axpy_kernel(_Float16 *__restrict__ dst,
                            const _Float16 *__restrict__ src, f32 alpha, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = static_cast<_Float16>(static_cast<f32>(dst[i]) +
                                   alpha * static_cast<f32>(src[i]));
}

// ---------------------------------------------------------------------------
// MoE batching: permute the tokens that picked an expert into a contiguous
// block (gather), run one GEMM over the block, then fold the results back to
// their original rows with their gate weight (scatter). The row plan is built
// on the host and staged to device memory once per layer.
// ---------------------------------------------------------------------------

// dst[i][j] = src[rows[i]][j], one block per source row.
__global__ void gather_rows_kernel(_Float16 *__restrict__ dst,
                                   const _Float16 *__restrict__ src,
                                   const i32 *__restrict__ rows, i64 n) {
    const i64 r = static_cast<i64>(rows[blockIdx.x]);
    const i64 base = static_cast<i64>(blockIdx.x) * n;
    for (i64 j = static_cast<i64>(threadIdx.x); j < n;
         j += static_cast<i64>(blockDim.x)) {
        dst[base + j] = src[r * n + j];
    }
}

// dst[rows[i]][j] += alpha[i] * src[i][j]. One block per grouped row; alpha is
// a small device array staged from the host. Experts write disjoint row sets,
// so plain adds are race-free.
__global__ void scatter_axpy_rows_kernel(_Float16 *__restrict__ dst,
                                         const _Float16 *__restrict__ src,
                                         const i32 *__restrict__ rows,
                                         const f32 *__restrict__ alpha, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x);
    const f32 a = alpha[i];
    const i64 r = static_cast<i64>(rows[i]);
    const _Float16 *sr = src + i * n;
    _Float16 *dr = dst + r * n;
    for (i64 j = static_cast<i64>(threadIdx.x); j < n;
         j += static_cast<i64>(blockDim.x)) {
        dr[j] = static_cast<_Float16>(static_cast<f32>(dr[j]) +
                                      a * static_cast<f32>(sr[j]));
    }
}

} // namespace krk

#endif // KRK_KERNEL_NORMALIZE_HPP
