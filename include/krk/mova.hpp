// ============================================================================
//  mova.hpp — MoVA: the attention VALUES are a routed expert set (k2-horizon).
//
//  k2-horizon has no `attn_v.weight` on its MoE layers. V is built the same way
//  its FFN is built -- route, then mix experts -- but with a bank of its own and
//  an activation of its own:
//
//      gate  = attn_v_gate @ x                        [n_value_expert]
//      probs = softmax(gate) | sigmoid(gate)          (expert_gating_func)
//      score = probs + attn_v_gate_b                  SELECTION ONLY
//      sel   = top-k(score)                           k = value_expert_used
//      w_j   = probs[sel_j] / sum_i probs[sel_i]      (only if expert_weights_norm)
//      V     = sum_j w_j * silu(attn_v_exps[sel_j] @ x)
//
//  Three things here are easy to get wrong in ways that still decode, which is
//  why the pure part of them lives in this file where a test can call it:
//
//  * the bias moves SELECTION and never the mixing weight. It is laguna's /
//    gpt-oss's selection-bias rule, and adding it to `probs` too changes the
//    mixture rather than the choice.
//  * silu is applied to each EXPERT'S OUTPUT, before the weight and before the
//    sum. `sum w_j silu(z_j)` is not `silu(sum w_j z_j)`, and only the first is
//    MoVA.
//  * the renormalisation divides by the sum of the SELECTED probabilities and
//    is skipped when the file does not ask for it: a sigmoid router without it
//    is a sum of independent gates, not a convex mixture.
//
//  With `k == n_value_expert` and `expert_weights_norm` the route is the whole
//  bank, and the result must equal the same sum over every expert in id order --
//  the invariant test_mova_router pins, because the selection loop is where an
//  off-by-one in `top-k` shows up as a missing expert rather than a crash.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// The routing decision for ONE token. Pure: no weights, no backend.
struct MovaRoute {
    // `logits` is [n_value_expert] pre-activation; `bias` is nullable;
    // `sel`/`w` are written with `k` entries, in descending score order with
    // ties broken by the LOWER expert id (deterministic, and the rule the FFN
    // router in engine.cpp uses too).
    static void route(i32 *sel, f32 *w, const f32 *logits, const f32 *bias,
                      i32 n_expert, i32 k, bool sigmoid_gate,
                      bool renorm);

    // Sum of the selected experts' outputs. `exps` is the bank laid out
    // [n_expert][n_in][n_out] as rows -- expert e occupies rows [e*n_out,
    // (e+1)*n_out) -- `x` is [n_in], `out` is [n_out]. `scratch` is [n_out]
    // working space for one expert. Returns false on a degenerate route.
    //
    // Kept next to route() so the mixture order (expert by expert, in `sel`
    // order, weight applied AFTER silu) is stated once.
    static bool mix(f32 *out, f32 *scratch, const i32 *sel, const f32 *w, i32 k,
                    const f32 *x, i32 n_in, const f32 *exps, i32 n_out);

    // silu(z) = z * sigmoid(z), the activation MoVA applies to each routed
    // expert's output. Exposed for the test's own oracle.
    static f32 silu(f32 z);
};

} // namespace krk
