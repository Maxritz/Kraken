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

} // namespace krk
