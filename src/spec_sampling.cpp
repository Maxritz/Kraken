#include "krk/spec_sampling.hpp"

#include <cmath>

namespace krk {
namespace {

// The mass a proposal survives with: min(p, q). Everything else in this file
// follows from it.
f64 accept_mass(f64 p, f64 q) { return p < q ? p : q; }

} // namespace

SpecStep spec_accept_step(const std::vector<f32> &p, const std::vector<f32> &q, i32 x,
                          f32 u_accept, f32 u_residual) {
    SpecStep r;
    if (x < 0 || static_cast<size_t>(x) >= p.size() || p.size() != q.size()) return r;
    const f64 px = p[static_cast<size_t>(x)];
    const f64 qx = q[static_cast<size_t>(x)];
    // q(x) == 0: accepted, see the header. p(x) == 0 with q(x) > 0: rejected by
    // construction (min(1, 0) = 0), which the comparison below already gives.
    const bool accepts = qx <= 0.0 || static_cast<f64>(u_accept) * qx < px;
    if (accepts) {
        r.token = x;
        r.accepted = true;
        return r;
    }
    std::vector<f32> res;
    if (!spec_residual(p, q, &res)) {
        // p == q: there is no correction, so the replacement is just a draw
        // from the target. Returning x would be the always-accept bug in
        // disguise (it would bias toward the proposal), which is why this arm
        // samples instead.
        r.token = spec_draw(p, u_residual);
        return r;
    }
    r.token = spec_draw(res, u_residual);
    return r;
}

bool spec_residual(const std::vector<f32> &p, const std::vector<f32> &q,
                   std::vector<f32> *out) {
    if (p.size() != q.size()) return false;
    out->assign(p.size(), 0.0f);
    f64 total = 0;
    for (size_t i = 0; i < p.size(); i++) {
        const f64 d = static_cast<f64>(p[i]) - static_cast<f64>(q[i]);
        if (d > 0) {
            (*out)[i] = static_cast<f32>(d);
            total += d;
        }
    }
    if (total <= 1e-12) {
        out->clear();
        return false;
    }
    const f32 inv = static_cast<f32>(1.0 / total);
    for (f32 &v : *out) v *= inv;
    return true;
}

bool spec_verify_identity(const std::vector<f32> &p, const std::vector<f32> &q,
                          f64 tol, std::string *why) {
    if (p.size() != q.size() || p.empty()) {
        if (why) *why = "p and q must be the same non-zero width";
        return false;
    }
    f64 sum_p = 0, sum_q = 0;
    for (size_t i = 0; i < p.size(); i++) {
        sum_p += p[i];
        sum_q += q[i];
    }
    if (std::fabs(sum_p - 1.0) > tol || std::fabs(sum_q - 1.0) > tol) {
        if (why)
            *why = "the rows are not distributions: sum(p) = " + std::to_string(sum_p) +
                   ", sum(q) = " + std::to_string(sum_q);
        return false;
    }
    f64 reject_mass = 0;
    for (size_t i = 0; i < p.size(); i++)
        reject_mass += static_cast<f64>(q[i]) - accept_mass(p[i], q[i]);
    if (reject_mass < 0) reject_mass = 0;
    f64 residual_total = 0;
    for (size_t i = 0; i < p.size(); i++) {
        const f64 d = static_cast<f64>(p[i]) - static_cast<f64>(q[i]);
        if (d > 0) residual_total += d;
    }
    f64 worst = 0;
    size_t worst_i = 0;
    for (size_t i = 0; i < p.size(); i++) {
        const f64 d = static_cast<f64>(p[i]) - static_cast<f64>(q[i]);
        const f64 resid = d > 0 && residual_total > 0 ? d / residual_total : 0.0;
        const f64 got = accept_mass(p[i], q[i]) + reject_mass * resid;
        const f64 err = std::fabs(got - static_cast<f64>(p[i]));
        if (err > worst) {
            worst = err;
            worst_i = i;
        }
    }
    if (worst > tol) {
        if (why)
            *why = "token " + std::to_string(worst_i) + " gets " +
                   std::to_string(accept_mass(p[worst_i], q[worst_i]) +
                                  reject_mass *
                                      ((p[worst_i] > q[worst_i] && residual_total > 0)
                                           ? (p[worst_i] - q[worst_i]) / residual_total
                                           : 0.0)) +
                   " where the target has " + std::to_string(p[worst_i]) +
                   " (error " + std::to_string(worst) + ")";
        return false;
    }
    return true;
}

void spec_softmax(const f32 *logits, i32 n, f32 temp, std::vector<f32> *prob) {
    prob->assign(static_cast<size_t>(n > 0 ? n : 0), 0.0f);
    if (!logits || n <= 0) return;
    if (temp <= 0.0f) {
        i32 best = 0;
        for (i32 i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        (*prob)[static_cast<size_t>(best)] = 1.0f;
        return;
    }
    f32 mx = logits[0];
    for (i32 i = 1; i < n; i++)
        if (logits[i] > mx) mx = logits[i];
    const f32 inv_temp = 1.0f / temp;
    f64 sum = 0;
    for (i32 i = 0; i < n; i++) {
        const f32 e = std::exp((logits[i] - mx) * inv_temp);
        (*prob)[static_cast<size_t>(i)] = e;
        sum += e;
    }
    if (sum <= 0) {
        // Every exponent underflowed: the row is degenerate, and one-hot on the
        // maximum is the only distribution it describes.
        i32 best = 0;
        for (i32 i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        (*prob)[static_cast<size_t>(best)] = 1.0f;
        return;
    }
    const f32 inv = static_cast<f32>(1.0 / sum);
    for (f32 &v : *prob) v *= inv;
}

void spec_map_dist(const f32 *draft_prob, i32 n_draft, const i32 *d2t, i32 n_target,
                   std::vector<f32> *out) {
    out->assign(static_cast<size_t>(n_target > 0 ? n_target : 0), 0.0f);
    if (!draft_prob || n_draft <= 0 || n_target <= 0) return;
    f64 total = 0;
    for (i32 i = 0; i < n_draft; i++) {
        const i32 t = d2t ? d2t[i] : i;
        if (t < 0 || t >= n_target) continue;
        const f32 v = draft_prob[i];
        if (!(v > 0.0f)) continue;
        (*out)[static_cast<size_t>(t)] += v;
        total += v;
    }
    if (total <= 0) return;
    const f32 inv = static_cast<f32>(1.0 / total);
    for (f32 &v : *out) v *= inv;
}

i32 spec_draw(const std::vector<f32> &dist, f32 u) {
    if (dist.empty()) return -1;
    f64 acc = 0;
    i32 last = -1;
    const f64 target = static_cast<f64>(u < 0.0f ? 0.0f : (u >= 1.0f ? 0.99999994f : u));
    for (size_t i = 0; i < dist.size(); i++) {
        const f32 v = dist[i];
        if (!(v > 0.0f)) continue;
        acc += v;
        last = static_cast<i32>(i);
        if (target < acc) return last;
    }
    // Rounding left the sum just short of u: the last token with mass is the
    // only answer that does not invent probability the row does not have.
    return last;
}

} // namespace krk
