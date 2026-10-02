// ============================================================================
//  gdn.hpp — gated delta net kernels (Qwen3.5 / Qwen3-Next linear attention).
//
//  Eleven ops, one launch each, f16 activations over an f32 recurrent state.
//  The shapes the engine produces are the Qwen3.5 ones: 16 key heads over 32
//  value heads, head/state width 128, conv width 2*2048 + 4096 = 8192.
//
//  What the shapes buy, per kernel:
//
//   l2norm      one block per (token, head). The head's sum of squares lives
//               in a wave shuffle and the scale is folded into the store, so
//               the head is read once and written once. hd is a compile-time
//               thread count, which is what makes the reduction exact.
//
//   conv1d_silu one thread per channel for the WHOLE token loop, with the tap
//               window held in registers and shifted instead of re-read from
//               the conv state. Per token that is one f16 load, KS fused
//               multiply-adds against registers, one f16 store — the depthwise
//               convolution never touches memory twice, and the sliding window
//               costs no traffic at all. The window is also what makes the
//               IN-PLACE call correct: output row t reaches back ksize-1 input
//               rows that earlier outputs have already overwritten, so those
//               rows have to be held rather than re-read.
//
//   delta_rule  one block per value head. A thread owns one state column and
//               RT rows of it in registers, and the rows are split over
//               d_state/RT groups, so a block covers a head's whole state and
//               the token loop is the only sequential part. The decayed tile
//               stays in registers across the update, which is the whole
//               point: the naive three-pass form (decay, then read for kv,
//               then read-modify-write) moves the state three times, this one
//               reads it once and writes it once. The two cross-group
//               reductions (kv, then out) are the only LDS traffic.
//
//  The conv taps are a constant weight, so they are decoded once per layer by
//  the backend into a device-side f16 cache (see stage_gdn_taps) rather than
//  per token: the dequant is 32K values that every channel reads, and paying
//  for it on the critical path of every decode step would be pure waste.
//
//  Dtype contract (the whole of it):
//    * activation buffers are the backend's act_type() — f16 here, and every
//      kernel widens to f32 in registers and narrows once at the store;
//    * the conv state and the delta-rule state are always f32, because they
//      outlive the forward and accumulate across thousands of tokens;
//    * the per-head vectors (ssm_dt, ssm_a, the norm weights) are f32 and
//      already resident on the device — the model loader uploads them;
//    * `kern` is a device weight in whatever DType the checkpoint used, and is
//      the one argument that is not activation-typed.
//  The CPU backend keeps all four in f32 and is the reference oracle; the only
//  deliberate precision difference is the conv taps, which are decoded to f16
//  here because the activations they feed are f16 anyway.
// ============================================================================
#ifndef KRK_KERNEL_GDN_HPP
#define KRK_KERNEL_GDN_HPP

#include <algorithm>

#include "../krk_hip.hpp"
#include "dequant.hpp"
#include "normalize.hpp" // block_reduce_sum

namespace krk {

// Softplus, matching the CPU reference's tail behaviour: above 20 softplus(v)
// is v to f32, and below -20 it is exp(v) to f32, while the middle band uses
// the real logarithm. The decay this feeds is exp(-exp(A) * softplus), so a
// saturated tail is harmless but a wrong middle band would not be.
__device__ __forceinline__ f32 d_softplus(f32 v) {
    if (v > 20.0f) return v;
    if (v < -20.0f) return __expf(v);
    return __logf(1.0f + __expf(v));
}

// ---------------------------------------------------------------------------
// per-head L2 normalization: o[t, h, i] = x[t, h, i] / sqrt(sum_j x^2 + eps)
// grid.x = heads, grid.y = tokens. `stride` is the element distance between
// tokens, because q/k/v are three regions of ONE fused post-conv row rather
// than three dense blocks.
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    gdn_l2norm_kernel(_Float16 *__restrict__ o, const _Float16 *__restrict__ x,
                      i64 hd, i64 stride, f32 eps) {
    const i64 base = static_cast<i64>(blockIdx.y) * stride +
                     static_cast<i64>(blockIdx.x) * hd;
    _Float16 *oh = o + base;
    const _Float16 *xh = x + base;

    f32 ss = 0.0f;
    for (i64 i = threadIdx.x; i < hd; i += THREADS) {
        const f32 v = static_cast<f32>(xh[i]);
        ss += v * v;
    }
    ss = block_reduce_sum<THREADS>(ss);

    __shared__ f32 inv_sh;
    if (threadIdx.x == 0) inv_sh = rsqrtf(ss + eps);
    __syncthreads();
    const f32 inv = inv_sh;

    for (i64 i = threadIdx.x; i < hd; i += THREADS)
        oh[i] = static_cast<_Float16>(static_cast<f32>(xh[i]) * inv);
}

// ---------------------------------------------------------------------------
// activations and scaling
// ---------------------------------------------------------------------------

__global__ void gdn_sigmoid_kernel(_Float16 *__restrict__ x, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const f32 v = static_cast<f32>(x[i]);
    x[i] = static_cast<_Float16>(1.0f / (1.0f + __expf(-v)));
}

__global__ void gdn_softplus_kernel(_Float16 *__restrict__ x, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = static_cast<_Float16>(d_softplus(static_cast<f32>(x[i])));
}

// x[t, i] *= alpha over `width` elements spaced `stride` apart. The token
// index is the grid's y dimension instead of a division by width, and the
// grid is capped at 65535 blocks in y with a stride loop, so a long prefill
// chunk cannot overflow a launch dimension.
__global__ void gdn_scale_act_kernel(_Float16 *__restrict__ x, f32 alpha,
                                     i64 width, i64 stride, i64 n_tok) {
    for (i64 t = blockIdx.y; t < n_tok; t += gridDim.y) {
        _Float16 *row = x + t * stride;
        for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < width; i += static_cast<i64>(gridDim.x) * blockDim.x)
            row[i] = static_cast<_Float16>(static_cast<f32>(row[i]) * alpha);
    }
}

__global__ void gdn_mul_act_kernel(_Float16 *__restrict__ a,
                                   const _Float16 *__restrict__ b, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = static_cast<_Float16>(static_cast<f32>(a[i]) * static_cast<f32>(b[i]));
}

// x[i, j] += bias[j]  /  x[i, j] *= col[j]   over n_row rows of n_col.
// One block per row, so the per-head vector is read once per row and stays
// resident in L1 for the whole row.
__global__ void gdn_add_bias_cols_kernel(_Float16 *__restrict__ x,
                                         const f32 *__restrict__ bias,
                                         i64 n_col, i64 n_row) {
    for (i64 i = blockIdx.y; i < n_row; i += gridDim.y) {
        _Float16 *row = x + i * n_col;
        for (i64 j = threadIdx.x; j < n_col; j += blockDim.x)
            row[j] = static_cast<_Float16>(static_cast<f32>(row[j]) + bias[j]);
    }
}

__global__ void gdn_scale_cols_kernel(_Float16 *__restrict__ x,
                                      const f32 *__restrict__ col, i64 n_col,
                                      i64 n_row) {
    for (i64 i = blockIdx.y; i < n_row; i += gridDim.y) {
        _Float16 *row = x + i * n_col;
        for (i64 j = threadIdx.x; j < n_col; j += blockDim.x)
            row[j] = static_cast<_Float16>(static_cast<f32>(row[j]) * col[j]);
    }
}

// x[i, :] *= alpha[i], alpha activation-typed and one scalar per row.
__global__ void gdn_scale_rows_kernel(_Float16 *__restrict__ x,
                                      const _Float16 *__restrict__ alpha, i64 n,
                                      i64 n_row) {
    for (i64 i = blockIdx.y; i < n_row; i += gridDim.y) {
        const f32 a = static_cast<f32>(alpha[i]);
        _Float16 *row = x + i * n;
        for (i64 j = threadIdx.x; j < n; j += blockDim.x)
            row[j] = static_cast<_Float16>(static_cast<f32>(row[j]) * a);
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5's packed attention query. Each head's 2*hd block interleaves the
// query with its output gate, so neither is contiguous:
//     q[t, h, i]    <- packed[t, h*2*hd + i]
//     gate[t, h, i] <- packed[t, h*2*hd + hd + i]
// `gate` may be null. In place is safe by index order: head h writes at
// (h*2*hd + hd) - hd, and no other head reads from below that.
// grid.x = heads, grid.y = tokens.
// ---------------------------------------------------------------------------

__global__ void gdn_qwen3_next_split_kernel(_Float16 *__restrict__ q,
                                            _Float16 *__restrict__ gate,
                                            const _Float16 *__restrict__ packed,
                                            i64 hd, i64 n_tok) {
    const i64 n_head = gridDim.x;
    const i64 h = blockIdx.x;
    for (i64 t = blockIdx.y; t < n_tok; t += gridDim.y) {
        const _Float16 *src = packed + ((t * n_head) + h) * 2 * hd;
        _Float16 *qd = q + ((t * n_head) + h) * hd;
        for (i64 i = threadIdx.x; i < hd; i += blockDim.x) {
            const f32 v = static_cast<f32>(src[i]);
            qd[i] = static_cast<_Float16>(v);
            if (gate) gate[((t * n_head) + h) * hd + i] =
                static_cast<_Float16>(static_cast<f32>(src[hd + i]));
        }
    }
}

// ---------------------------------------------------------------------------
// conv taps: one thread per 32-value chunk, so a K-quant's shared scales are
// decoded once by the thread that needs them and the row is read once.
// Only whole chunks are decoded, which is every real shape (the conv width is
// a multiple of 32) and the same limitation the host dequant has.
// ---------------------------------------------------------------------------

__global__ void gdn_dequant_taps_kernel(_Float16 *__restrict__ dst,
                                        const u8 *__restrict__ src, int tt,
                                        i64 n) {
    const i64 c = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c * 32 >= n) return;
    const int cpb = dtype_chunks_per_block_dev(tt);
    const int bb = dtype_block_bytes_dev(tt);
    dequant_chunk_dev(tt, src + (c / cpb) * bb, static_cast<int>(c % cpb),
                      dst + c * 32);
}

// ---------------------------------------------------------------------------
// the causal short convolution, depthwise over `chan` channels:
//     out[t, c] = silu( sum_{j<ksize} kern[j, c] * X(t + j, c) )
// where X is `state` (the ksize-1 previous steps) followed by this call's rows
// in `in`, and the newest tap is last. On return `state` holds the last
// ksize-1 rows of that window, which is what makes the next call continue the
// same sequence.
//
// One thread per channel, whole token loop, window in registers. `in` and
// `out` may be the same pointer — the engine convolves in place — and the
// window is what makes that safe: output row t reads input row t-keep, which
// an earlier output has already overwritten, so it is held in a register
// rather than fetched again.
// ---------------------------------------------------------------------------

template <int KS, int THREADS>
__global__ void __launch_bounds__(THREADS)
    gdn_conv1d_silu_kernel(_Float16 *__restrict__ out,
                           const _Float16 *__restrict__ in,
                           f32 *__restrict__ state,
                           const _Float16 *__restrict__ kern, i64 n_tok,
                           i64 chan) {
    constexpr int kKeep = KS - 1;
    const i64 c = static_cast<i64>(blockIdx.x) * THREADS + threadIdx.x;
    if (c >= chan) return;

    // The taps are constant for the whole call, so they are read once and kept
    // in registers: every token of the loop reuses them out of a register file
    // instead of re-fetching 4*chan values from L1.
    f32 tap[KS];
#pragma unroll
    for (int j = 0; j < KS; j++) tap[j] = static_cast<f32>(kern[j * chan + c]);

    // win[i] holds X(m - i) with m the newest step, so win[0] is this call's
    // first input row and win[i > 0] walks back through the stored window:
    // win[i] = state[keep - i], the tail stored by the previous call.
    f32 win[KS];
#pragma unroll
    for (int i = 1; i < KS; i++) win[i] = state[(KS - 1 - i) * chan + c];
    win[0] = static_cast<f32>(in[c]);

    for (i64 t = 0; t < n_tok; t++) {
        // X(t + j) == win[kKeep - j], newest tap last.
        f32 acc = 0.0f;
#pragma unroll
        for (int j = 0; j < KS; j++) acc += tap[j] * win[kKeep - j];
        out[t * chan + c] = static_cast<_Float16>(d_silu(acc));

        // Slide toward the newest end: the row just consumed moves up and the
        // next input row becomes the new win[0].
#pragma unroll
        for (int i = KS - 1; i > 0; i--) win[i] = win[i - 1];
        if (t + 1 < n_tok) win[0] = static_cast<f32>(in[(t + 1) * chan + c]);
    }
    // The window now spans X(n_tok + kKeep - i), so the ksize-1 rows the next
    // call needs are win[kKeep - j] for j < kKeep. win[0] would be the row
    // that call has yet to produce, which is why it is not kept.
#pragma unroll
    for (int j = 0; j < kKeep; j++) state[j * chan + c] = win[kKeep - j];
}

// Kernel widths no real checkpoint uses, kept correct rather than supported:
// the window lives in local memory here, which is slower but only ever runs
// when ssm.conv_kernel is something the register path cannot express.
__global__ void gdn_conv1d_silu_generic_kernel(_Float16 *__restrict__ out,
                                               const _Float16 *__restrict__ in,
                                               f32 *__restrict__ state,
                                               const _Float16 *__restrict__ kern,
                                               i64 n_tok, i64 chan,
                                               i64 ksize) {
    const i64 c = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= chan) return;
    const i64 keep = ksize - 1;
    // Same sliding window as the register kernel, in local memory: the call is
    // in place, so a row cannot be re-read once an earlier output has landed
    // on top of it.
    f32 win[16];
    for (i64 i = 1; i < ksize; i++) win[i] = state[(ksize - 1 - i) * chan + c];
    win[0] = static_cast<f32>(in[c]);
    for (i64 t = 0; t < n_tok; t++) {
        f32 acc = 0.0f;
        for (i64 j = 0; j < ksize; j++)
            acc += static_cast<f32>(kern[j * chan + c]) * win[keep - j];
        out[t * chan + c] = static_cast<_Float16>(d_silu(acc));
        for (i64 i = ksize - 1; i > 0; i--) win[i] = win[i - 1];
        if (t + 1 < n_tok) win[0] = static_cast<f32>(in[(t + 1) * chan + c]);
    }
    for (i64 j = 0; j < keep; j++) state[j * chan + c] = win[keep - j];
}

// ---------------------------------------------------------------------------
// the delta rule, over n_tok consecutive tokens of one sequence:
//     S *= exp(g[t]);  kv = S^T k;  d = (v - kv) * beta;  S += k (x) d
//     out[t] = S^T q
// `state` is f32 [n_v_head, d_state, hd] and persists across calls. Value head
// h runs the token loop; its state tile lives in registers, RT rows per thread,
// so the whole update is one read and one write of the state per token.
// ---------------------------------------------------------------------------

template <int RT>
__global__ void __launch_bounds__(1024)
    gdn_delta_rule_kernel(_Float16 *__restrict__ out, f32 *__restrict__ state,
                          const _Float16 *__restrict__ q,
                          const _Float16 *__restrict__ k,
                          const _Float16 *__restrict__ v,
                          const _Float16 *__restrict__ g,
                          const _Float16 *__restrict__ beta, i64 n_tok,
                          i64 n_k_head, i64 n_v_head, i64 d_state, i64 hd,
                          i64 row_stride) {
    const i64 h = blockIdx.x;
    const int hd_i = static_cast<int>(hd);
    const int col = static_cast<int>(threadIdx.x) % hd_i;
    const int grp = static_cast<int>(threadIdx.x) / hd_i;
    const int ngrp = static_cast<int>(blockDim.x) / hd_i;
    f32 *__restrict__ Sh = state + h * d_state * hd;
    // Qwen3-Next runs 16 key heads over 32 value heads by tiling.
    const i64 kbase = (h % n_k_head) * d_state;
    const i64 vbase = h * hd;

    // Two reduction buffers, one for kv and one for out, so the second phase
    // never has to wait for the first to be re-read.
    extern __shared__ f32 red[];
    f32 *red_kv = red;
    f32 *red_o = red + static_cast<size_t>(ngrp) * hd;

    for (i64 t = 0; t < n_tok; t++) {
        const f32 decay = __expf(static_cast<f32>(g[t * n_v_head + h]));
        const f32 bta = static_cast<f32>(beta[t * n_v_head + h]);
        const f32 vv = static_cast<f32>(v[t * row_stride + vbase + col]);
        const _Float16 *kt = k + t * row_stride + kbase;
        const _Float16 *qt = q + t * row_stride + kbase;

        // Pass one: decay the tile in registers and project it onto k. The
        // store is deferred to pass two, so the state is written once.
        f32 tile[RT];
        f32 kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const i64 i = static_cast<i64>(grp) * RT + r;
            const f32 s = Sh[i * hd + col] * decay;
            tile[r] = s;
            kv += s * static_cast<f32>(kt[i]);
        }
        red_kv[static_cast<i64>(grp) * hd + col] = kv;
        __syncthreads();
        f32 kvt = 0.0f;
        for (int u = 0; u < ngrp; u++) kvt += red_kv[u * hd + col];
        const f32 d = (vv - kvt) * bta;

        // Pass two: the rank-one update, folded into the output projection —
        // out uses the UPDATED state, so the two cannot be split without
        // re-reading what pass one just computed.
        f32 o = 0.0f;
#pragma unroll
        for (int r = 0; r < RT; r++) {
            const i64 i = static_cast<i64>(grp) * RT + r;
            const f32 s = tile[r] + static_cast<f32>(kt[i]) * d;
            Sh[i * hd + col] = s;
            o += s * static_cast<f32>(qt[i]);
        }
        red_o[static_cast<i64>(grp) * hd + col] = o;
        __syncthreads();
        f32 ot = 0.0f;
        for (int u = 0; u < ngrp; u++) ot += red_o[u * hd + col];
        out[(t * n_v_head + h) * hd + col] = static_cast<_Float16>(ot);
        // Both buffers are read again on the next token.
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------

inline void gdn_conv1d_silu_launch(_Float16 *out, const _Float16 *in, f32 *state,
                                   const _Float16 *kern, i64 n_tok, i64 chan,
                                   i64 ksize) {
    if (n_tok <= 0 || chan <= 0 || ksize <= 0 || ksize > 16) return;
    constexpr int kThreads = 64;
    const unsigned grid =
        static_cast<unsigned>((chan + kThreads - 1) / kThreads);
#define KRK_GDN_CONV(KS)                                                         \
    gdn_conv1d_silu_kernel<KS, kThreads><<<grid, kThreads>>>(out, in, state,    \
                                                              kern, n_tok, chan)
    switch (ksize) {
        case 1: KRK_GDN_CONV(1); break;
        case 2: KRK_GDN_CONV(2); break;
        case 3: KRK_GDN_CONV(3); break;
        case 4: KRK_GDN_CONV(4); break;
        default:
            gdn_conv1d_silu_generic_kernel<<<grid, kThreads>>>(
                out, in, state, kern, n_tok, chan, ksize);
            break;
    }
#undef KRK_GDN_CONV
}

inline void gdn_delta_rule_launch(_Float16 *out, f32 *state, const _Float16 *q,
                                  const _Float16 *k, const _Float16 *v,
                                  const _Float16 *g, const _Float16 *beta,
                                  i64 n_tok, i64 n_k_head, i64 n_v_head,
                                  i64 d_state, i64 hd, i64 row_stride) {
    if (n_tok <= 0 || n_v_head <= 0 || d_state <= 0 || hd <= 0) return;
    if (hd > 1024) {
        KRK_ERROR("delta_rule: head width %lld exceeds a block",
                  static_cast<long long>(hd));
        std::abort();
    }
    // A thread owns RT state rows in registers, so RT is a compile-time tile and
    // the block is hd columns x (d_state/RT) row groups. The largest tile that
    // keeps the group count inside the block limit wins: 128-wide state lands
    // on RT=32, i.e. 4 groups of 32 rows and a 512-thread block.
    static const i64 kRT[] = {32, 16, 64, 8, 4, 1};
    i64 rt = 0;
    for (i64 c : kRT) {
        if (d_state % c != 0) continue;
        const i64 grp = d_state / c;
        if (grp > 8 || hd * grp > 1024) continue;
        rt = c;
        break;
    }
    if (rt == 0) {
        KRK_ERROR("delta_rule: state %lld x %lld has no register tiling",
                  static_cast<long long>(d_state), static_cast<long long>(hd));
        std::abort();
    }
    const unsigned grid = static_cast<unsigned>(n_v_head);
    const unsigned threads = static_cast<unsigned>(hd * (d_state / rt));
    const size_t lds = static_cast<size_t>(2 * (d_state / rt) * hd) * sizeof(f32);
#define KRK_GDN_DELTA(RT)                                                        \
    gdn_delta_rule_kernel<RT><<<grid, threads, lds>>>(                           \
        out, state, q, k, v, g, beta, n_tok, n_k_head, n_v_head, d_state, hd,   \
        row_stride)
    switch (rt) {
        case 32: KRK_GDN_DELTA(32); break;
        case 16: KRK_GDN_DELTA(16); break;
        case 64: KRK_GDN_DELTA(64); break;
        case 8: KRK_GDN_DELTA(8); break;
        case 4: KRK_GDN_DELTA(4); break;
        default: KRK_GDN_DELTA(1); break;
    }
#undef KRK_GDN_DELTA
}

} // namespace krk

#endif // KRK_KERNEL_GDN_HPP
