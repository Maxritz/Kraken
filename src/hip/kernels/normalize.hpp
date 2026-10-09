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
                   const f32 *__restrict__ w, i64 n, f32 eps,
                   const i32 *__restrict__ row_exp = nullptr) {
    const i64 r = blockIdx.x;
    const _Float16 *xr = x + r * n;
    _Float16 *orow = out + r * n;
    // The residual stream may be stored divided by one power of two per token
    // (see add_residual_kernel): a per-token exponent cannot ride the
    // arithmetic -- silu(2^-e g) is not 2^-e silu(g), and q and k scaled
    // together put 2^-2e in the scores, which softmax reads as a temperature --
    // so it lives on the storage, and the norm is where it is put back. The
    // multiply is a power of two, so it is exact, and it is 1.0 (leaving this
    // kernel bit-identical to the unscaled one) for every row that never left
    // f16.
    const f32 sc = row_exp ? exp2f(static_cast<f32>(row_exp[r])) : 1.0f;

    f32 ss = 0.0f;
    for (i64 i = threadIdx.x; i < n; i += THREADS) {
        const f32 v = static_cast<f32>(xr[i]) * sc;
        ss += v * v;
    }
    ss = block_reduce_sum<THREADS>(ss);

    __shared__ f32 scale_sh;
    if (threadIdx.x == 0)
        scale_sh = rsqrtf(ss / static_cast<f32>(n) + eps);
    __syncthreads();
    const f32 scale = scale_sh;

    for (i64 i = threadIdx.x; i < n; i += THREADS)
        orow[i] = static_cast<_Float16>(static_cast<f32>(xr[i]) * sc * scale * w[i]);
}

// ---------------------------------------------------------------------------
// Grouped RMSNorm (Mamba-2's ssm_norm): one block per (token, group) row, whose
// weight vector is that group's own -- out[r][i] = x[r][i] * rsqrt(mean + eps)
// * w[(r % n_group) * width + i]. Structurally rmsnorm_kernel above; the only
// additions are the row's group decomposition and the per-group weight select.
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    rmsnorm_grouped_kernel(_Float16 *__restrict__ out,
                           const _Float16 *__restrict__ x,
                           const f32 *__restrict__ w, i64 n_group, i64 width,
                           i64 stride, f32 eps) {
    const i64 r = blockIdx.x;
    const i64 g = r % n_group;
    const i64 base = (r / n_group) * stride + g * width;
    const _Float16 *xr = x + base;
    _Float16 *orow = out + base;
    const f32 *wr = w + g * width;

    f32 ss = 0.0f;
    for (i64 i = threadIdx.x; i < width; i += THREADS)
        ss += static_cast<f32>(xr[i]) * static_cast<f32>(xr[i]);
    ss = block_reduce_sum<THREADS>(ss);

    __shared__ f32 scale_sh;
    if (threadIdx.x == 0)
        scale_sh = rsqrtf(ss / static_cast<f32>(width) + eps);
    __syncthreads();
    const f32 scale = scale_sh;

    for (i64 i = threadIdx.x; i < width; i += THREADS)
        orow[i] = static_cast<_Float16>(static_cast<f32>(xr[i]) * scale * wr[i]);
}

// ---------------------------------------------------------------------------
// Residual add folded into RMSNorm: out[r][i] = rmsnorm(x[r] + res[r]) * w[i],
// with x[r] left holding the sum.
//
// Bit-identical to add_inplace_kernel() followed by rmsnorm_kernel(): the sum
// is rounded to the activation type before it is squared, exactly as the two
// kernels do, so the norm views the same values it would have read back from
// memory. That matters more than the launch it saves -- a fused kernel that is
// merely close turns a greedy decoder into a different decoder, and no unit
// test in this repo would say so.
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    rmsnorm_add_kernel(_Float16 *__restrict__ out, _Float16 *__restrict__ x,
                       const _Float16 *__restrict__ res,
                       const f32 *__restrict__ w, i64 n, f32 eps,
                       i32 *__restrict__ rexp = nullptr) {
    constexpr f32 kCeil = 65504.0f;
    const i64 r = blockIdx.x;
    _Float16 *xr = x + r * n;
    const _Float16 *rr = res + r * n;
    _Float16 *orow = out + r * n;
    // Same contract as add_residual_kernel, in the shape this kernel already
    // has: the residual row is stored divided by 2^e, the contribution is at
    // true scale, so the sum is formed in f32, the row's own maximum picks the
    // smallest exponent that keeps it inside f16, and the norm reads the
    // un-scaled values. e == e' == 0 (and every expression below identical to
    // the unscaled kernel) whenever the row fits, which is every model except
    // the two laguna files.
    const f32 sc = rexp ? exp2f(static_cast<f32>(rexp[r])) : 1.0f;

    f32 mx = 0.0f;
    for (i64 i = threadIdx.x; i < n; i += THREADS) {
        const f32 v = static_cast<f32>(xr[i]) * sc + static_cast<f32>(rr[i]);
        const f32 av = fabsf(v);
        if (av > mx) mx = av;
    }
    mx = d_wave_reduce_max(mx);
    __shared__ f32 msh[THREADS / 32];
    if ((threadIdx.x & 31) == 0) msh[threadIdx.x >> 5] = mx;
    __syncthreads();
    __shared__ f32 mul_sh;
    if (threadIdx.x == 0) {
        f32 m = msh[0];
        for (int k = 1; k < static_cast<int>(THREADS / 32); k++)
            m = fmaxf(m, msh[k]);
        // From ZERO, not from the exponent the row arrived under: `sc` above
        // has already applied that one to read the row at true scale, so the
        // smallest exponent that fits this pass's sum is a fresh decision. It
        // used to start at rexp[r], which double-counted: a row stored at 2^-3
        // whose true maximum was still 8x over f16 left at 2^-6, the next pass
        // took it to -9, and a row that crossed f16 once climbed every layer
        // until it hit the clamp. Measured on the build before this line: XS-2.1
        // `max exponent 30` on rows whose values needed ~5, and S-2.1 `max
        // exponent 60` -- the clamp -- with `residual 0` beside it, i.e. nothing
        // saturated and the scale still ran away, and at 2^-60 the row's small
        // elements underflow f16 entirely (min subnormal 6e-8).
        int e2 = 0;
        while (m > kCeil && e2 < 60) {
            m *= 0.5f;
            e2++;
        }
        if (rexp) {
            rexp[r] = e2;
            if (e2 > 0) atomicMax(&g_krk_rexp_max, e2);
        }
        mul_sh = exp2f(static_cast<f32>(e2));
    }
    __syncthreads();
    const f32 inv = 1.0f / mul_sh;

    f32 ss = 0.0f;
    for (i64 i = threadIdx.x; i < n; i += THREADS) {
        const f32 v = static_cast<f32>(xr[i]) * sc + static_cast<f32>(rr[i]);
        const _Float16 h = d_sat_f16_count(v * inv, &g_krk_sat_resid);
        xr[i] = h;
        const f32 hv = static_cast<f32>(h) * mul_sh;
        ss += hv * hv;
    }
    ss = block_reduce_sum<THREADS>(ss);

    __shared__ f32 scale_sh;
    if (threadIdx.x == 0)
        scale_sh = rsqrtf(ss / static_cast<f32>(n) + eps);
    __syncthreads();
    const f32 scale = scale_sh;

    for (i64 i = threadIdx.x; i < n; i += THREADS)
        orow[i] = static_cast<_Float16>(static_cast<f32>(xr[i]) * mul_sh * scale * w[i]);
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
// Whole-row RMSNorm (OLMoE QK-norm). grid.x = tokens; one block per row.
//
// The per-head kernel above reduces over head_dim within one head and leaves
// every head with its own scale. OLMoE instead normalizes the [n_embd]
// projection *before* the head split, so there is one scale for the whole row
// and the weight vector spans all n = n_head * head_dim channels. Reducing per
// head there is the difference between the reference and a wrong answer.
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    row_norm_kernel(_Float16 *__restrict__ x, const f32 *__restrict__ w,
                    i64 n, f32 eps) {
    const i64 t = blockIdx.x;
    _Float16 *xr = x + t * n;

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
        xr[i] = static_cast<_Float16>(static_cast<f32>(xr[i]) * scale * w[i]);
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

// Rotary embedding over one (head, token) row per block. `neox` selects the
// pair convention: false rotates the adjacent channels (2i, 2i+1) — llama.cpp's
// "norm" layout — and true rotates channel i with channel i + rot, the
// half-split layout the NeoX family is exported in. Which one a file needs is
// a property of its architecture (ArchSpec::rope_neox), because the two are
// indistinguishable from the tensor alone: both stay finite and in range.
__global__ void rope_kernel(_Float16 *__restrict__ x, i64 n_heads, i64 hd,
                            const i64 *__restrict__ pos0,
                            const f32 *__restrict__ inv_freq, f32 scale,
                            f32 frac, bool neox) {
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
        const i64 ia = neox ? i : 2 * i;
        const i64 ib = neox ? i + rot : 2 * i + 1;
        const f32 a = static_cast<f32>(xh[ia]);
        const f32 b = static_cast<f32>(xh[ib]);
        xh[ia] = static_cast<_Float16>(a * cs - b * sn);
        xh[ib] = static_cast<_Float16>(a * sn + b * cs);
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
    out[i] = d_sat_f16_count(d_silu(static_cast<f32>(gate[i])) *
                                         static_cast<f32>(up[i]),
                               &g_krk_sat_silu);
}

// One block per row of a silu product: store silu(gate) * up with the row
// magnitude kept inside f16 by one power-of-two scale per row.
//
// The product is the one quantity on this path that can leave the range its
// operands live in -- silu(x) -> x for large positive x, so silu(gate) * up is
// ~x*x, and a model whose expert activations are large enough (measured: 785411
// at Laguna-XS-2.1-IQ3_XXS L38 e161, against f16 65504) overflows on the
// store. The reference does not fail there because a quantized matmul output
// is f32 (docs/DIAG-iq3xxs-nan.md); on this side the activation buffer is f16,
// so the row has to be scaled to fit it.
//
// `row_len` is one token expert activation width, which is exactly the
// dimension the down projection reduces over -- so one scale per row is what
// its consumer needs to undo. Each thread computes its share of the row into
// f32 registers and keeps the largest magnitude, the block reduces to the row
// maximum, and every product is stored divided by the smallest power of two
// that puts that maximum under the ceiling.
//
// A power-of-two divide is exact in f16, so the stored row carries the same
// mantissa it would have carried unscaled: this buys exponent range rather
// than spending precision. `scale_out[row]` is the multiplier (2^k) the caller
// folds back in, and k is 0 -- with the store bit-for-bit the plain kernel --
// whenever the row is already in range, which is every row of every model
// measured here except the one this exists for.
//
// `out` may alias `gate` (the MoE path passes the same buffer): every read
// happens before the block-wide barrier and every store after it.
__global__ void silu_mul_scaled_kernel(_Float16 *__restrict__ out,
                                       const _Float16 *__restrict__ gate,
                                       const _Float16 *__restrict__ up,
                                       i64 row_len, f32 *__restrict__ scale_out) {
    constexpr int kMaxVec = 8;       // 8 * 256 threads = 2048-wide rows
    constexpr f32 kCeil = 65504.0f;  // f16 largest finite value
    const i64 base = static_cast<i64>(blockIdx.x) * row_len;
    const int t = threadIdx.x;
    f32 v[kMaxVec];
    f32 mx = 0.0f;
    for (int j = 0; j < kMaxVec; j++) {
        const i64 i = static_cast<i64>(t) + static_cast<i64>(j) * blockDim.x;
        if (i < row_len) {
            const f32 p = d_silu(static_cast<f32>(gate[base + i])) *
                          static_cast<f32>(up[base + i]);
            v[j] = p;
            const f32 a = fabsf(p);
            if (a > mx) mx = a;
        } else {
            v[j] = 0.0f;
        }
    }
    __shared__ f32 smx[8];
    mx = d_wave_reduce_max(mx);
    if ((t & 31) == 0) smx[t >> 5] = mx;
    __syncthreads();
    if (t == 0) {
        f32 m = smx[0];
        for (int w = 1; w < static_cast<int>(blockDim.x >> 5); w++)
            m = fmaxf(m, smx[w]);
        f32 mul = 1.0f;
        while (m > kCeil * mul && mul < 16777216.0f) mul *= 2.0f;
        smx[0] = mul;
    }
    __syncthreads();
    const f32 mul = smx[0];
    if (t == 0) scale_out[blockIdx.x] = mul;
    for (int j = 0; j < kMaxVec; j++) {
        const i64 i = static_cast<i64>(t) + static_cast<i64>(j) * blockDim.x;
        // Through the bound as well: the row scale is what brings the row into
        // range, and a row still over after 2^k is a fact the counters should
        // carry rather than an inf the model eats.
        if (i < row_len)
            out[base + i] = d_sat_f16_count(v[j] / mul, &g_krk_sat_silu);
    }
}

// ------------------------------------------------- residual stream storage --
//
// Storage-only per-token residual scaling.
//
// The residual stream is the one buffer the model itself drives out of f16: a
// "massive activation" row reaches ~3e5 while every store on the path has a
// 65504 ceiling, and unlike a product (bounded row by row at the store) the
// hidden state cannot be bounded at all -- a saturated hidden state is wrong by
// a *factor*, and it goes on to poison every later layer. What it can be is
// stored at a different exponent: the residual is the one buffer every block
// reads through a norm, the norm is scale-invariant, and the exponent is a
// power of two, so it rides the storage for free.
//
// The contract, in one place: `a` holds X * 2^-e for its row, a layer's
// contribution `b` arrives at TRUE scale (every block computes in true scale,
// because the norm that starts it un-scales), the sum is formed in f32, the
// smallest e' >= e that puts the whole row back inside f16 is chosen, the row is
// stored as sum * 2^-e', and rexp[row] = e' for every later reader (the norms,
// and whichever store comes next).
//
// e only ever rises, only a row that left f16 has e > 0, and every scale is a
// power of two -- exponent-only on an f16 mantissa -- so a model that stays in
// range takes e == e' == 0 and this store is f16(f32(a) + f32(b)), which is
// exactly add_inplace_kernel's arithmetic. In range the whole change is
// bit-identical by construction.
//
// Two passes over the row: the first finds the magnitude the exponent needs,
// the second stores (the row is at most n_embd elements of L1-resident data).
// `bexp` (nullable) is b's own storage exponent when the contribution is
// already carried scaled -- the MoE staging fold, where the routed experts'
// sum has its own row exponent by the time the shared expert is added. Both
// sides are then un-scaled to true scale and the sum is renormalized once, so
// the result is the same number either way; null (or zero) is the ordinary
// case, a contribution that arrives at true scale.
__global__ void add_residual_kernel(_Float16 *__restrict__ a,
                                    const _Float16 *__restrict__ b,
                                    i32 *__restrict__ rexp, i64 n,
                                    const i32 *__restrict__ bexp = nullptr) {
    constexpr f32 kCeil = 65504.0f;
    const i64 r = blockIdx.x;
    _Float16 *ar = a + r * n;
    const _Float16 *br = b + r * n;
    const f32 sc = rexp ? exp2f(static_cast<f32>(rexp[r])) : 1.0f;
    const f32 scb = bexp ? exp2f(static_cast<f32>(bexp[r])) : 1.0f;

    f32 mx = 0.0f;
    for (i64 i = threadIdx.x; i < n; i += blockDim.x) {
        const f32 v = static_cast<f32>(ar[i]) * sc +
                      static_cast<f32>(br[i]) * scb;
        const f32 av = fabsf(v);
        if (av > mx) mx = av;
    }
    mx = d_wave_reduce_max(mx);
    __shared__ f32 msh[8];
    if ((threadIdx.x & 31) == 0) msh[threadIdx.x >> 5] = mx;
    __syncthreads();
    __shared__ f32 mul_sh;
    if (threadIdx.x == 0) {
        f32 m = msh[0];
        for (int k = 1; k < static_cast<int>(blockDim.x >> 5); k++)
            m = fmaxf(m, msh[k]);
        // From ZERO, not from the exponent the row arrived under: `sc` above
        // has already applied that one to read the row at true scale, so the
        // smallest exponent that fits this pass's sum is a fresh decision. It
        // used to start at rexp[r], which double-counted: a row stored at 2^-3
        // whose true maximum was still 8x over f16 left at 2^-6, the next pass
        // took it to -9, and a row that crossed f16 once climbed every layer
        // until it hit the clamp. Measured on the build before this line: XS-2.1
        // `max exponent 30` on rows whose values needed ~5, and S-2.1 `max
        // exponent 60` -- the clamp -- with `residual 0` beside it, i.e. nothing
        // saturated and the scale still ran away, and at 2^-60 the row's small
        // elements underflow f16 entirely (min subnormal 6e-8).
        int e2 = 0;
        while (m > kCeil && e2 < 60) {
            m *= 0.5f;
            e2++;
        }
        if (rexp) {
            rexp[r] = e2;
            if (e2 > 0) atomicMax(&g_krk_rexp_max, e2);
        }
        mul_sh = exp2f(static_cast<f32>(e2));
    }
    __syncthreads();
    const f32 inv = 1.0f / mul_sh;

    for (i64 i = threadIdx.x; i < n; i += blockDim.x) {
        const f32 v = static_cast<f32>(ar[i]) * sc +
                      static_cast<f32>(br[i]) * scb;
        ar[i] = d_sat_f16_count(v * inv, &g_krk_sat_resid);
    }
}

// dst[i] = v for i < n. The silu_mul fallback (a row too wide for the scaled
// kernel register budget) still has to answer the row-scale contract, and 1.0
// per row is the multiplier that leaves the row exactly as the plain store
// wrote it.
// Zeroes the activation-range counters (krk_hip.hpp). One thread, one launch,
// once per engine: the numbers a run prints are meant to describe that run.
__global__ void krk_zero_sat_counters_kernel() {
    g_krk_sat_silu = 0u;
    g_krk_sat_sum = 0u;
    g_krk_sat_resid = 0u;
    g_krk_rexp_max = 0;
}

__global__ void fill_f32_kernel(f32 *__restrict__ dst, f32 v, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = v;
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
//
// `scale` (nullable) is the row scale a prior silu_mul wrote for these same
// rows: it multiplies alpha[i], which is how the power of two that kept the
// product inside f16 is undone in the pass that already walks the row.
// `row_exp` (nullable) is the destination row's own storage exponent, in and
// out: the row was written under it and is written under the (possibly
// larger) one this pass picks. Nullable so a caller with no storage contract
// gets the saturating store it had before.
__global__ void scatter_axpy_rows_kernel(_Float16 *__restrict__ dst,
                                         const _Float16 *__restrict__ src,
                                         const i32 *__restrict__ rows,
                                         const f32 *__restrict__ alpha,
                                         const f32 *__restrict__ scale, i64 n,
                                         i32 *__restrict__ row_exp) {
    constexpr int kMaxVec = 8;       // 8 x 256 threads = 2048-wide rows
    constexpr f32 kCeil = 65504.0f;  // f16's largest finite value
    const i64 i = static_cast<i64>(blockIdx.x);
    const f32 a = alpha[i] * (scale ? scale[i] : 1.0f);
    const i64 r = static_cast<i64>(rows[i]);
    const _Float16 *sr = src + i * n;
    _Float16 *dr = dst + r * n;
    // The destination is the layer's contribution to the residual stream, and
    // it is the last store in the MoE chain. A row of it is the SUM of ten
    // experts that can each be brought into range on their own and still add up
    // past f16 (measured: Laguna-XS-2.1 `experts-sum 1020`, all of them in the
    // prefill chunk), so the row carries a storage exponent exactly as the
    // residual does: sum in f32, store the row divided by the smallest power of
    // two that keeps the row's own maximum inside f16, and hand the exponent to
    // the fold (`ws_fexp_`, read by add_residual) that puts the scale back. AT
    // EXPONENT 0 THIS IS `f16(f32(dst) + alpha*src)`, expression for expression,
    // which is what keeps every in-range model's numbers where they were.
    const int e0 = row_exp ? row_exp[r] : 0;
    const f32 sc = exp2f(static_cast<f32>(e0));
    // The row is held in registers when it fits that budget (the common case), so
    // the maximum and the store need one pass; otherwise the maximum is taken
    // first and the sum recomputed in the store pass, which is the same f32
    // value because the recomputation repeats the same expression.
    const bool hold =
        n <= static_cast<i64>(kMaxVec) * static_cast<i64>(blockDim.x);
    f32 v[kMaxVec];
    f32 mx = 0.0f;
    if (hold) {
        for (int j = 0; j < kMaxVec; j++) {
            const i64 k = static_cast<i64>(threadIdx.x) +
                          static_cast<i64>(j) * blockDim.x;
            if (k < n) {
                const f32 t = static_cast<f32>(dr[k]) * sc +
                              a * static_cast<f32>(sr[k]);
                v[j] = t;
                const f32 av = fabsf(t);
                if (av > mx) mx = av;
            } else {
                v[j] = 0.0f;
            }
        }
    } else {
        for (i64 k = static_cast<i64>(threadIdx.x); k < n;
             k += static_cast<i64>(blockDim.x)) {
            const f32 t = static_cast<f32>(dr[k]) * sc +
                          a * static_cast<f32>(sr[k]);
            const f32 av = fabsf(t);
            if (av > mx) mx = av;
        }
    }
    mx = d_wave_reduce_max(mx);
    __shared__ f32 msh[8];
    if ((threadIdx.x & 31) == 0) msh[threadIdx.x >> 5] = mx;
    __syncthreads();
    __shared__ f32 mul_sh;
    if (threadIdx.x == 0) {
        f32 m = msh[0];
        for (int w = 1; w < static_cast<int>(blockDim.x >> 5); w++)
            m = fmaxf(m, msh[w]);
        int e2 = e0;
        while (m > kCeil && e2 < 60) {
            m *= 0.5f;
            e2++;
        }
        if (row_exp) row_exp[r] = e2;
        mul_sh = exp2f(static_cast<f32>(e2));
    }
    __syncthreads();
    const f32 inv = 1.0f / mul_sh;
    if (hold) {
        for (int j = 0; j < kMaxVec; j++) {
            const i64 k = static_cast<i64>(threadIdx.x) +
                          static_cast<i64>(j) * blockDim.x;
            if (k < n)
                dr[k] = d_sat_f16_count(v[j] * inv, &g_krk_sat_sum);
        }
    } else {
        for (i64 k = static_cast<i64>(threadIdx.x); k < n;
             k += static_cast<i64>(blockDim.x)) {
            const f32 t = static_cast<f32>(dr[k]) * sc +
                          a * static_cast<f32>(sr[k]);
            dr[k] = d_sat_f16_count(t * inv, &g_krk_sat_sum);
        }
    }
}

} // namespace krk

#endif // KRK_KERNEL_NORMALIZE_HPP
