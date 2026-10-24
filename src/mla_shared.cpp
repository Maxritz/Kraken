// ============================================================================
//  mla_shared.cpp — DeepSeek-V4 shared-latent MLA, host oracle.
//
//  Plain f32 loops, one addition at a time, so the three consumers that will
//  be checked against this file (the CPU backend, the suite, and the HIP
//  kernels' A/B) all have the same reference. See include/krk/mla_shared.hpp
//  for the graph and for the one assumption (V is the unrotated latent).
// ============================================================================
#include "krk/mla_shared.hpp"

#include <cmath>
#include <vector>

namespace krk {
namespace {

// y = rmsnorm(v) * gamma, the engine's row_norm formula.
void rms_norm(f32 *y, const f32 *v, const f32 *gamma, i64 n, f32 eps) {
    f64 ss = 0.0;
    for (i64 i = 0; i < n; i++) {
        const f64 x = v[i];
        ss += x * x;
    }
    const f32 inv = 1.0f / std::sqrt(static_cast<f32>(ss / n) + eps);
    for (i64 i = 0; i < n; i++) y[i] = v[i] * inv * (gamma ? gamma[i] : 1.0f);
}

// out[j] = sum_i a[i*cols + j] * v[i], a row-major [rows, cols].
void proj(f32 *out, const f32 *a, const f32 *v, i64 rows, i64 cols) {
    for (i64 j = 0; j < cols; j++) out[j] = 0.0f;
    for (i64 i = 0; i < rows; i++)
        for (i64 j = 0; j < cols; j++) out[j] += a[i * cols + j] * v[i];
}

// Same, with a row stride wider than `cols`: the grouped output projection is
// one [in, o_groups*o_lora] matrix read one o_lora-wide column block at a
// time, so the stride is the whole output width and the column offset is the
// group's. Getting this wrong reads the neighbouring group and still produces
// a full-size, finite vector -- the reason the suite pins the stride.
void proj_strided(f32 *out, const f32 *a, const f32 *v, i64 rows, i64 cols,
                  i64 stride) {
    for (i64 j = 0; j < cols; j++) out[j] = 0.0f;
    for (i64 i = 0; i < rows; i++)
        for (i64 j = 0; j < cols; j++) out[j] += a[i * stride + j] * v[i];
}

// True when key t takes part in this query: never the future, and inside the
// window when the layer has one (V4's SWA layers do).
bool key_in_range(i64 kpos, i64 q_pos, i32 window) {
    if (kpos > q_pos) return false;
    if (window > 0 && kpos <= q_pos - window) return false;
    return true;
}

} // namespace

void mla_shared_rope_tail(f32 *v, i64 d, i64 n_rot, i64 pos,
                          const f32 *inv_freq) {
    if (!v || !inv_freq || n_rot <= 0 || d < n_rot) return;
    const i64 base = d - n_rot; // the rotary dims are the TAIL
    for (i64 i = 0; i + 1 < n_rot; i += 2) {
        const f32 ang = static_cast<f32>(pos) * inv_freq[i / 2];
        const f32 c = std::cos(ang), s = std::sin(ang);
        const f32 a = v[base + i], b = v[base + i + 1];
        v[base + i] = a * c - b * s;
        v[base + i + 1] = a * s + b * c;
    }
}

void mla_shared_latent(f32 *c, const f32 *x, const MlaSharedWeights &w,
                       const MlaSharedGeometry &g, f32 eps) {
    if (!c || !x || !w.wkv || !g.valid()) return;
    std::vector<f32> raw(static_cast<size_t>(g.kv_lora));
    proj(raw.data(), w.wkv, x, g.n_embd, g.kv_lora);
    // The norm covers the WHOLE latent, tail included: on this layout the rope
    // is inside the row that gets normed, not beside it.
    rms_norm(c, raw.data(), w.wkv_norm, g.kv_lora, eps);
}

namespace {

// Shared front half: q_a -> norm -> q_b (per-head q), with the rotary tail
// already applied at q_pos. Both orders read the same queries; only the
// value accumulation differs.
void build_queries(std::vector<f32> &q, const f32 *x, i64 q_pos,
                   const MlaSharedWeights &w, const MlaSharedGeometry &g,
                   const f32 *inv_freq, f32 eps) {
    const i64 ql = g.q_lora;
    const i64 qd = static_cast<i64>(g.n_head) * g.kv_lora;
    std::vector<f32> qa(static_cast<size_t>(ql)), qn(static_cast<size_t>(ql));
    q.assign(static_cast<size_t>(qd), 0.0f);
    proj(qa.data(), w.wq_a, x, g.n_embd, ql);
    rms_norm(qn.data(), qa.data(), w.wq_a_norm, ql, eps);
    proj(q.data(), w.wq_b, qn.data(), ql, qd);
    for (i32 h = 0; h < g.n_head; h++)
        mla_shared_rope_tail(q.data() + static_cast<size_t>(h) * g.kv_lora,
                             g.kv_lora, g.qk_rope, q_pos, inv_freq);
}

// scores[h][t] for every cached key, with the sink folded in later (it is a
// denominator term, so it must not enter the dot product).
void build_scores(std::vector<f32> &scores, const std::vector<f32> &q,
                  const f32 *c_cache, const i64 *kpos, i64 n_keys, i64 q_pos,
                  const MlaSharedGeometry &g, const f32 *inv_freq, f32 scale,
                  i32 window) {
    const i64 d = g.kv_lora;
    scores.assign(static_cast<size_t>(g.n_head) * n_keys, 0.0f);
    std::vector<f32> k(static_cast<size_t>(d));
    for (i64 t = 0; t < n_keys; t++) {
        if (!key_in_range(kpos[t], q_pos, window)) continue;
        const f32 *c = c_cache + t * d;
        // K is the rope-rotated COPY of the cached latent; the cache itself
        // stays unrotated so the value side can use it (see the header).
        for (i64 i = 0; i < d; i++) k[static_cast<size_t>(i)] = c[i];
        mla_shared_rope_tail(k.data(), d, g.qk_rope, kpos[t], inv_freq);
        for (i32 h = 0; h < g.n_head; h++) {
            const f32 *qh = q.data() + static_cast<size_t>(h) * d;
            f32 s = 0.0f;
            for (i64 i = 0; i < d; i++) s += qh[i] * k[static_cast<size_t>(i)];
            scores[static_cast<size_t>(h) * n_keys + t] = s * scale;
        }
    }
}

// The grouped low-rank output: per-group column block of Woa over that
// group's concatenated heads, then Wob over the concatenation.
void grouped_output(f32 *out, const std::vector<f32> &a,
                    const MlaSharedWeights &w, const MlaSharedGeometry &g) {
    const i64 hpg = g.heads_per_group();
    const i64 d = g.kv_lora;
    const i64 gin = hpg * d;
    const i64 gout = static_cast<i64>(g.o_groups) * g.o_lora;
    std::vector<f32> in(static_cast<size_t>(gin)), y(static_cast<size_t>(gout));
    for (i32 grp = 0; grp < g.o_groups; grp++) {
        for (i64 h = 0; h < hpg; h++)
            for (i64 i = 0; i < d; i++)
                in[static_cast<size_t>(h * d + i)] =
                    a[static_cast<size_t>(grp) * hpg * d + h * d + i];
        proj_strided(y.data() + static_cast<size_t>(grp) * g.o_lora,
                     w.wo_a + static_cast<size_t>(grp) * g.o_lora, in.data(),
                     gin, g.o_lora, gout);
    }
    proj(out, w.wo_b, y.data(), gout, g.n_embd);
}

} // namespace

void mla_shared_forward_accum(f32 *out, const f32 *x, const f32 *c_cache,
                              const i64 *kpos, i64 n_keys, i64 q_pos,
                              const MlaSharedWeights &w,
                              const MlaSharedGeometry &g, const f32 *inv_freq,
                              f32 scale, f32 eps, i32 window) {
    if (!out || !x || !c_cache || !kpos || !g.valid() || !inv_freq ||
        n_keys <= 0)
        return;
    const i64 d = g.kv_lora;
    std::vector<f32> q;
    build_queries(q, x, q_pos, w, g, inv_freq, eps);
    std::vector<f32> scores;
    build_scores(scores, q, c_cache, kpos, n_keys, q_pos, g, inv_freq, scale,
                 window);
    // Per head: softmax over the keys in range (with the sink in the
    // denominator), then accumulate that key's latent into the head output.
    std::vector<f32> a(static_cast<size_t>(g.n_head) * d, 0.0f);
    for (i32 h = 0; h < g.n_head; h++) {
        f32 *row = scores.data() + static_cast<size_t>(h) * n_keys;
        f32 mx = w.sinks ? w.sinks[h] : -INFINITY;
        for (i64 t = 0; t < n_keys; t++)
            if (key_in_range(kpos[t], q_pos, window) && row[t] > mx) mx = row[t];
        f32 sum = 0.0f;
        if (w.sinks) sum = std::exp(w.sinks[h] - mx);
        for (i64 t = 0; t < n_keys; t++) {
            if (!key_in_range(kpos[t], q_pos, window)) {
                row[t] = 0.0f;
                continue;
            }
            row[t] = std::exp(row[t] - mx);
            sum += row[t];
        }
        if (sum <= 0.0f) continue;
        f32 *oh = a.data() + static_cast<size_t>(h) * d;
        for (i64 t = 0; t < n_keys; t++) {
            const f32 p = row[t] / sum;
            if (p == 0.0f) continue;
            const f32 *c = c_cache + t * d;
            for (i64 i = 0; i < d; i++) oh[i] += p * c[i];
        }
    }
    grouped_output(out, a, w, g);
}

void mla_shared_forward_gather(f32 *out, const f32 *x, const f32 *c_cache,
                               const i64 *kpos, i64 n_keys, i64 q_pos,
                               const MlaSharedWeights &w,
                               const MlaSharedGeometry &g, const f32 *inv_freq,
                               f32 scale, f32 eps, i32 window) {
    if (!out || !x || !c_cache || !kpos || !g.valid() || !inv_freq ||
        n_keys <= 0)
        return;
    const i64 d = g.kv_lora;
    std::vector<f32> q;
    build_queries(q, x, q_pos, w, g, inv_freq, eps);
    std::vector<f32> scores;
    build_scores(scores, q, c_cache, kpos, n_keys, q_pos, g, inv_freq, scale,
                 window);
    std::vector<f32> a(static_cast<size_t>(g.n_head) * d, 0.0f);
    for (i32 h = 0; h < g.n_head; h++) {
        f32 *row = scores.data() + static_cast<size_t>(h) * n_keys;
        f32 mx = w.sinks ? w.sinks[h] : -INFINITY;
        for (i64 t = 0; t < n_keys; t++)
            if (key_in_range(kpos[t], q_pos, window) && row[t] > mx) mx = row[t];
        f32 sum = 0.0f;
        if (w.sinks) sum = std::exp(w.sinks[h] - mx);
        for (i64 t = 0; t < n_keys; t++) {
            if (!key_in_range(kpos[t], q_pos, window)) {
                row[t] = 0.0f;
                continue;
            }
            row[t] = std::exp(row[t] - mx);
            sum += row[t];
        }
        if (sum <= 0.0f) continue;
        f32 *oh = a.data() + static_cast<size_t>(h) * d;
        // Weights first, then one pass over the cache per output dim.
        for (i64 t = 0; t < n_keys; t++) row[t] /= sum;
        for (i64 i = 0; i < d; i++) {
            f32 acc = 0.0f;
            for (i64 t = 0; t < n_keys; t++) {
                if (row[t] == 0.0f) continue;
                acc += row[t] * c_cache[t * d + i];
            }
            oh[i] = acc;
        }
    }
    grouped_output(out, a, w, g);
}

} // namespace krk
