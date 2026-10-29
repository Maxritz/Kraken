// ============================================================================
//  mova.cpp — the pure half of MoVA (see include/krk/mova.hpp).
// ============================================================================
#include "krk/mova.hpp"

#include <cmath>

namespace krk {

f32 MovaRoute::silu(f32 z) {
    // z * sigmoid(z), with the branch that keeps exp() from overflowing on a
    // large negative input -- the same shape as the hc.cpp logistics.
    if (z >= 0.0f) {
        const f32 e = std::exp(-z);
        return z / (1.0f + e);
    }
    const f32 e = std::exp(z);
    return z * e / (1.0f + e);
}

void MovaRoute::route(i32 *sel, f32 *w, const f32 *logits, const f32 *bias,
                      i32 n_expert, i32 k, bool sigmoid_gate, bool renorm) {
    if (!sel || !w || !logits || n_expert <= 0) return;
    if (k <= 0) k = 1;
    if (k > n_expert) k = n_expert;

    // The mixing distribution. Softmax is max-shifted so a large logit cannot
    // overflow; sigmoid is left unnormalised, which is what the reference does.
    f32 probs[256];
    const i32 n = n_expert < 256 ? n_expert : 256;
    if (sigmoid_gate) {
        for (i32 e = 0; e < n; e++) probs[e] = 1.0f / (1.0f + std::exp(-logits[e]));
    } else {
        f32 mx = logits[0];
        for (i32 e = 1; e < n; e++) mx = std::max(mx, logits[e]);
        f32 sum = 0.0f;
        for (i32 e = 0; e < n; e++) { probs[e] = std::exp(logits[e] - mx); sum += probs[e]; }
        const f32 inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (i32 e = 0; e < n; e++) probs[e] *= inv;
    }

    // Selection by the BIASED score, weight from the UNBIASED probability.
    for (i32 j = 0; j < k; j++) {
        i32 best = -1;
        f32 best_score = 0.0f;
        for (i32 e = 0; e < n; e++) {
            bool taken = false;
            for (i32 q = 0; q < j; q++)
                if (sel[q] == e) { taken = true; break; }
            if (taken) continue;
            const f32 s = probs[e] + (bias ? bias[e] : 0.0f);
            // `>` and not `>=` keeps the lower id on an exact tie.
            if (best < 0 || s > best_score) { best = e; best_score = s; }
        }
        sel[j] = best;
        w[j] = best >= 0 ? probs[best] : 0.0f;
    }

    if (renorm) {
        f32 sum = 0.0f;
        for (i32 j = 0; j < k; j++) sum += w[j];
        // The reference clamps at 2^-14 before dividing, so a route whose
        // selected probabilities are all ~0 mixes to zeros rather than to NaN.
        const f32 inv = sum > 6.103515625e-5f ? 1.0f / sum : 0.0f;
        for (i32 j = 0; j < k; j++) w[j] *= inv;
    }
}

bool MovaRoute::mix(f32 *out, f32 *scratch, const i32 *sel, const f32 *w, i32 k,
                    const f32 *x, i32 n_in, const f32 *exps, i32 n_out) {
    if (!out || !scratch || !sel || !w || !x || !exps || n_out <= 0) return false;
    for (i32 o = 0; o < n_out; o++) out[o] = 0.0f;
    for (i32 j = 0; j < k; j++) {
        const i32 e = sel[j];
        if (e < 0) continue;
        const f32 *row = exps + static_cast<i64>(e) * n_out * n_in;
        for (i32 o = 0; o < n_out; o++) {
            f32 acc = 0.0f;
            for (i32 i = 0; i < n_in; i++) acc += row[static_cast<i64>(o) * n_in + i] * x[i];
            scratch[o] = silu(acc);      // silu on the OUTPUT, before the weight
        }
        for (i32 o = 0; o < n_out; o++) out[o] += scratch[o] * w[j];
    }
    return true;
}

} // namespace krk
