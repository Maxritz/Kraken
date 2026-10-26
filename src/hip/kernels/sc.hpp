// ============================================================================
//  sc.hpp — LFM2.5's short convolution (`lfm2moe`), the mixer on 30 of its 40
//  layers.
//
//  The block is not a scan and not attention. Its projection emits three
//  n_embd-wide slices -- B, C and X -- and the whole mixer is
//
//      bx[t] = B[t] * X[t]
//      y[t]  = sum_{k<kern} w[k, c] * bx[t - (kern-1) + k, c]  (causal, depthwise)
//      out[t]= C[t] * y[t]
//
//  The convolution runs on B*X, NOT on X, and C gates the CONVOLVED row -- this
//  is the reference's `hidden_states = B * x` immediately before the conv and
//  its `C * conv(...)` immediately after (transformers' Lfm2ShortConv, and the
//  same three lines in llama.cpp's graph). Reading it as `B*conv(X) + C*X` is
//  the plausible-looking mistake: it has the same shape, the same op count and
//  one fewer multiply, and it produces a finite, fluent, wrong model.
//  There is NO activation between the convolution and the multiply, which is
//  the one place this differs from the gated delta net's `conv1d_silu`.
//
//  The history is a RING keyed by ABSOLUTE position, not a shift register. A
//  shift has to move (kern-1) rows per call and, on a one-row decode, moves them
//  into slots the same launch is reading -- a race that surfaces exactly as a
//  chunk-size dependence, which is the property the whole block exists to keep.
//  With a ring a call writes one row per history slot at `pos % (kern-1)` and
//  nothing moves; the two launches are ordered by the stream, so a slot written
//  after the convolution cannot be read by it. The rows it holds are B*X rows,
//  because that is what the convolution consumes.
//
//  The state is f16, the width the activations have, so a chunked prefill sees
//  the bytes a token-at-a-time decode would have written.
// ============================================================================
#pragma once

#include "../krk_hip.hpp"

namespace krk {

// out[t, c] = C * conv(B*X), one thread per (t, c).
//   proj  [n_tok, 3*ch]  the projection, slices B | C | X in that order
//   state [hist, ch]     ring of B*X rows, slot = absolute pos % hist
//   w     [kern * ch]    taps, `w[k*ch + c]`
//
// `hist` is the RING's depth, not the window's: it is at least kern-1 and may
// be deeper, so a speculation round that rewinds to an accepted prefix still
// finds the two rows it needs. Nothing in the arithmetic depends on it beyond
// the slot function -- the window's taps are always the last kern-1 rows -- so
// a deeper ring is bit-identical to a shallow one.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    shortconv_kernel(_Float16 *__restrict__ out, const _Float16 *__restrict__ proj,
                     const _Float16 *__restrict__ state,
                     const f32 *__restrict__ w, i64 n_tok, i64 ch, i64 kern,
                     i64 hist, i64 pos0) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tok * ch) return;
    const i64 t = i / ch;
    const i64 c = i - t * ch;
    const i64 stride = 3 * ch;
    f32 acc = 0.0f;
    for (i64 k = 0; k < kern; k++) {
        const i64 lag = kern - 1 - k;
        const i64 p = pos0 + t - lag;
        if (p < 0) continue;
        // The product is ROUNDED TO f16 before the tap, in EVERY case, because
        // the ring stores f16. A row this call produced still in f32 would
        // otherwise carry one bit more than the same row read back from the
        // ring, and that single ulp is a chunk dependence: measured, `--chunk 1`
        // and `--chunk 2` gave different hashes and different text (0.00055837
        // against 0.00055800 at pos 1 of layer 0). The ring is what forces the
        // precision, so the arithmetic is pinned to it here.
        const f32 bx = p >= pos0
                           ? static_cast<f32>(static_cast<_Float16>(
                                 static_cast<f32>(proj[(t - lag) * stride + c]) *
                                 static_cast<f32>(proj[(t - lag) * stride + 2 * ch + c])))
                           : static_cast<f32>(state[(p % hist) * ch + c]);
        acc += bx * w[k * ch + c];
    }
    const f32 cg = static_cast<f32>(proj[t * stride + ch + c]);
    out[i] = static_cast<_Float16>(cg * acc);
}

// state[(p % hist, c)] = B[p]*X[p] for the last `hist` rows this call made.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    shortconv_save_kernel(_Float16 *__restrict__ state,
                          const _Float16 *__restrict__ proj, i64 n_tok, i64 ch,
                          i64 hist, i64 pos0) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_tok * ch) return;
    const i64 r = i / ch;
    const i64 c = i - r * ch;
    // ONE ROW PER POSITION: every row of this call is stored, not just the
    // newest `kern-1`. The ring is deeper than the window on purpose -- a
    // speculation round rewinds to an accepted prefix and the rows it must read
    // afterwards are rows this pass already produced.
    const i64 p = pos0 + r;
    if (p < pos0) return;
    // The RING slot of that position, not r: a short pass writes only some of
    // the slots and the one it writes is decided by the absolute position.
    const i64 src = (p - pos0) * (3 * ch);
    state[(p % hist) * ch + c] =
        static_cast<_Float16>(static_cast<f32>(proj[src + c]) *
                              static_cast<f32>(proj[src + 2 * ch + c]));
}

inline void shortconv_launch(void *out, const void *proj, void *state,
                             const void *w, i64 n_tok, i64 ch, i64 kern,
                             i64 hist, i64 pos0, hipStream_t stream = nullptr) {
    if (n_tok <= 0 || ch <= 0 || kern <= 1 || hist < kern - 1) return;
    const i64 total = n_tok * ch;
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
    shortconv_kernel<256><<<blocks, 256, 0, stream>>>(
        static_cast<_Float16 *>(out), static_cast<const _Float16 *>(proj),
        static_cast<const _Float16 *>(state), static_cast<const f32 *>(w), n_tok,
        ch, kern, hist, pos0);
    const i64 rows = n_tok * ch;
    const unsigned hb = static_cast<unsigned>((rows + 255) / 256);
    shortconv_save_kernel<256><<<hb, 256, 0, stream>>>(
        static_cast<_Float16 *>(state), static_cast<const _Float16 *>(proj),
        n_tok, ch, hist, pos0);
}

} // namespace krk
