// ============================================================================
//  hc.hpp — qwen4exp hyper-connections: the residual is a lane stack, not one
//  vector.
//
//  Every other schema this engine runs carries one residual stream per token,
//  `x` of width n_embd. qwen4exp carries `hc` of them (4), so the block input is
//  [hc, n_embd] wide (10240 for Qwen3.8-Flash-Next) and each sub-block reads a
//  MIX of the lanes while writing back into all of them:
//
//      xn    = rms_norm(x_lane) * gamma_lane                  (per lane)
//      lo    = silu(hc_attn_down * xn / hc)                   (10240 -> 320)
//      gate  = hc_attn_up * lo                                (320 -> 10240)
//      mixed = mean_over_lanes( xn_lane * sigmoid(gate_lane) ) (read by the block)
//      inj   = hc_attn_inject * xn                           (10240 -> hc)
//      x_c  += block_out * 2 * sigmoid(inj_c / hc)            (combine)
//
//  The two functions here are the parts that are not a matmul or an rmsnorm:
//  the lane mean with its sigmoid gate, and the lane scatter whose weight is
//  centred on exactly 1. They are host code on purpose — the CPU backend needs
//  them, the suite can call them directly, and the HIP kernels are checked
//  against these before they are trusted.
//
//  The `2 * sigmoid(x / hc)` centring is load-bearing, not cosmetic: at
//  injection 0 every lane weight is exactly 1, so a block whose output is zero
//  is exactly the identity and the lanes are still bit-equal copies after the
//  combine. A plain `sigmoid(x)` would halve the residual at initialisation and
//  the model would run with a shrunk stream that no shape check catches.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// The geometry a qwen4exp stack declares: `hc` lanes of `n_embd`, mixed through
// a `low_rank` bottleneck.
struct HcGeometry {
    i32 n_embd = 0;   // 2560
    i32 hc = 0;       // 4
    i32 low_rank = 0; // 320

    i64 lane_width() const { return static_cast<i64>(n_embd) * hc; }
    bool valid() const { return n_embd > 0 && hc > 1 && low_rank > 0; }
};

// Weight one lane applies to a block output: 2 * sigmoid(inject / hc). Exactly
// 1.0f when `inject` is 0 (the centring rule above), strictly inside (0, 2).
f32 hc_lane_weight(f32 inject, i32 hc);

// Per-lane RMSNorm, gamma per (element, lane): `gamma` is [hc, n_embd] laid out
// lane-major, matching the file's [n_embd, hc] tensor read as rows of n_embd.
// `out` may alias `x`. `x`/`out` are [n, hc, n_embd] token-major.
void hc_norm_lanes(f32 *out, const f32 *x, const f32 *gamma, i32 n,
                   const HcGeometry &g, f32 eps);

// The mixed stream a sub-block reads: mean over lanes of
// `xn_lane * sigmoid(gate_lane)`. `xn` and `gate` are both [n, hc, n_embd],
// `gate` pre-sigmoid (the up-projection's output), `mixed` is [n, n_embd].
//
// With all `hc` lanes of `xn` equal and all lanes of `gate` equal, this is
// exactly `xn * sigmoid(gate)` within one ulp — the identical-lane invariant the
// suite pins, because the reference starts the stack as `hc` copies of the
// embedding and a wrong lane stride shows up here first.
void hc_mix(f32 *mixed, const f32 *xn, const f32 *gate, i32 n,
            const HcGeometry &g);

// The combine: `res_lane += block_out * hc_lane_weight(inject_lane)`, over every
// token and lane. `res` is [n, hc, n_embd], `block_out` is [n, n_embd],
// `inject` is [n, hc]. A zero `block_out` leaves `res` untouched whatever the
// injection is; that is what makes the "still identical before the first combine"
// case checkable instead of assumed.
void hc_combine(f32 *res, const f32 *block_out, const f32 *inject, i32 n,
                const HcGeometry &g);



// ============================================================================
//  DeepSeek-V4.1's hyper-connections: the same lane stack, a DIFFERENT mixer.
//
//  V4.1 carries `hc` lanes too (4) and mixes them per sub-block, but nothing
//  about the mixer is qwen4exp's. There is no low-rank bottleneck and no
//  per-lane gamma: one full-rank projection reads the whole flattened stack and
//  emits `hc_mix_dim = (2 + hc) * hc` numbers per token, which split into three
//  pieces that play three different roles.
//
//      flat  = reshape(lanes, hc * n_embd)              (token-major)
//      fmix  = rms_norm(flat)                           (one norm, no weight)
//      mix   = hc_fn^T @ fmix                           (hc*n_embd -> (2+hc)*hc)
//
//      pre[c]        = sigmoid(mix[c] * scale[0] + base[c]) + eps
//      post[c]       = sigmoid(mix[hc+c] * scale[1] + base[hc+c]) * 2
//      comb[d][s]    = sinkhorn(mix[2hc + d*hc + s] * scale[2] + base[2hc + d*hc + s])
//
//  `pre` collapses the lanes to the row a sub-block reads. `post` and `comb`
//  scatter the block's output back: lane d becomes
//
//      lanes'[d] = block_out * post[d] + sum_s lanes[s] * comb[d][s]
//
//  so a sub-block does not ADD to a residual -- it REPLACES every lane with a
//  mix of all of them plus a gated copy of its own output. That is why the
//  block's input can be collapsed with a pre-mix computed one sub-layer earlier
//  (V4.1's "single-pass mHC") without the lanes themselves lagging: post and
//  comb are always from the current state.
//
//  The sinkhorn is the part worth pinning: `comb` must stay a doubly stochastic
//  matrix, because that is what keeps the lane stack's scale from drifting over
//  60 sub-layers. Its definition, in this order and no other:
//
//      comb = softmax_over_d(comb) + eps
//      comb /= (sum_s comb + eps)                       ("column" normalisation)
//      repeat iters-1 times: comb /= (sum_d comb + eps); comb /= (sum_s + eps)
//
//  The eps is added at every step, including inside each normaliser, and the
//  first pass is columns only -- not rows. Getting that order wrong still
//  produces a matrix whose rows sum to ~1, so it decodes, and it is not this
//  model. The reference for all of it is llama.cpp's deepseek4.cpp
//  (build_hc_mixes / build_hc_sinkhorn / build_hc_post) and the DeepSeek-V4.1
//  inference code it was written against.
// ============================================================================

// Everything the mixer needs that is not a tensor.
struct HcSinkhornGeometry {
    i32 n_embd = 0;
    i32 hc = 0;                 // lanes (V4.1 asserts 4)
    i32 sinkhorn_iters = 0;     // > 0; 1 means "columns once"
    f32 eps = 0.0f;

    i64 lane_width() const { return static_cast<i64>(n_embd) * hc; }
    i64 mix_dim() const { return static_cast<i64>(2 + hc) * hc; }
    bool valid() const {
        return n_embd > 0 && hc > 1 && sinkhorn_iters > 0 && eps > 0.0f;
    }
};

// The three pieces one `mixes` row splits into. `pre`/`post` are [n, hc];
// `comb` is [n, hc, hc] with dst-major rows (comb[t*hc*hc + d*hc + s] is the
// weight lane d puts on source lane s). `mixes` is [n, mix_dim], `scale` is
// [3] and `base` is [mix_dim].
void hc_sinkhorn_split(f32 *pre, f32 *post, f32 *comb, const f32 *mixes,
                       const f32 *scale, const f32 *base, i32 n,
                       const HcSinkhornGeometry &g);

// lanes'[d] = block_out * post[d] + sum_s lanes[s] * comb[d][s], for every token.
// `lanes_in` and `lanes_out` are [n, hc, n_embd] and MAY NOT alias: every lane
// of the output reads every lane of the input.
void hc_sinkhorn_post(f32 *lanes_out, const f32 *lanes_in, const f32 *block_out,
                      const f32 *post, const f32 *comb, i32 n,
                      const HcSinkhornGeometry &g);

// out[t, e] = sum_c lanes[t, c, e] * pre[t, c] -- the collapse a sub-block reads,
// and the same call the head makes on the last pre-mix (V4.1 has no head mixer).
void hc_sinkhorn_pre(f32 *out, const f32 *lanes, const f32 *pre, i32 n,
                     const HcSinkhornGeometry &g);

// The sinkhorn normalisation itself, exposed because it is the piece with an
// order to get wrong: normalises one [hc, hc] matrix in place.
void hc_sinkhorn_normalise(f32 *m, i32 hc, i32 iters, f32 eps);

} // namespace krk
