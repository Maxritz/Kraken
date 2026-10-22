// ============================================================================
//  mla.cpp — host implementation of multi-head latent attention.
//
//  Plain f32 loops so the CPU backend, the suite and the HIP kernels' oracle
//  all agree on the arithmetic: head by head, element by element, one
//  addition at a time. See include/krk/mla.hpp for the two orders.
// ============================================================================
#include "krk/mla.hpp"

#include <cmath>
#include <vector>

namespace krk {
namespace {

// y = rmsnorm(v) * gamma: the engine's row_norm formula (mean square + eps,
// per-element gamma). Matches what the loader uploads as *_norm weights.
void rms_norm(f32 *y, const f32 *v, const f32 *gamma, i64 n, f32 eps) {
    f64 ss = 0.0;
    for (i64 i = 0; i < n; i++) {
        const f64 x = v[i];
        ss += x * x;
    }
    const f32 inv = 1.0f / std::sqrt(static_cast<f32>(ss / n) + eps);
    for (i64 i = 0; i < n; i++) y[i] = v[i] * inv * (gamma ? gamma[i] : 1.0f);
}

// r = A^T v with A [rows, cols] row-major: out[j] = sum_i A[i*cols+j]*v[i].
// Every projection here reads its matrix in the same orientation, so one
// helper keeps the orders comparable line by line.
void proj(f32 *out, const f32 *a, const f32 *v, i64 rows, i64 cols) {
    for (i64 j = 0; j < cols; j++) out[j] = 0.0f;
    for (i64 i = 0; i < rows; i++)
        for (i64 j = 0; j < cols; j++) out[j] += a[i * cols + j] * v[i];
}

// Wqb packs all heads in slices wider than one head's columns, so the row
// stride is not the column count. Same additions as proj, explicit stride.
void proj_strided(f32 *out, const f32 *a, const f32 *v, i64 rows, i64 cols,
                  i64 stride) {
    for (i64 j = 0; j < cols; j++) out[j] = 0.0f;
    for (i64 i = 0; i < rows; i++)
        for (i64 j = 0; j < cols; j++) out[j] += a[i * stride + j] * v[i];
}

} // namespace

void mla_absorb_q(f32 *wq_abs, const f32 *wq_b, const f32 *wuk,
                  const MlaGeometry &g) {
    if (!wq_abs || !wq_b || !wuk || !g.valid()) return;
    const i64 head_slice = static_cast<i64>(g.qk_nope) + g.qk_rope;
    for (i32 h = 0; h < g.n_head; h++) {
        // Wqb_nope[h] is [q_lora, nope] at slice offset h*(nope+rope);
        // Wuk[h] is [kv_lora, nope]; Wq_abs[h] = Wqb_nope @ Wuk^T.
        const f32 *qb = wq_b + static_cast<i64>(h) * head_slice;
        const f32 *uk = wuk + static_cast<i64>(h) * g.kv_lora * g.qk_nope;
        f32 *abs = wq_abs + static_cast<i64>(h) * g.q_lora * g.kv_lora;
        for (i64 i = 0; i < g.q_lora; i++)
            for (i64 k = 0; k < g.kv_lora; k++) {
                f32 s = 0.0f;
                for (i64 d = 0; d < g.qk_nope; d++)
                    s += qb[i * head_slice + d] * uk[k * g.qk_nope + d];
                abs[i * g.kv_lora + k] = s;
            }
    }
}

void mla_forward_unfolded(f32 *out, const f32 *x, const f32 *c_cache,
                          const f32 *kr_cache, i64 n_keys, const MlaWeights &w,
                          const MlaGeometry &g, f32 scale, f32 eps) {
    if (!out || !x || !c_cache || !kr_cache || !g.valid() || n_keys <= 0) return;
    const i64 head_slice = static_cast<i64>(g.qk_nope) + g.qk_rope;
    std::vector<f32> qc(g.q_lora), qn(g.qk_nope), qr(g.qk_rope);
    std::vector<f32> tmp(std::max<i64>(g.kv_lora, g.q_lora));
    // Shared query trunk: compress, norm, expand, split.
    proj(tmp.data(), w.wq_a, x, g.n_embd, g.q_lora);
    rms_norm(qc.data(), tmp.data(), w.wqa_norm, g.q_lora, eps);
    for (i64 i = 0; i < g.n_embd; i++) out[i] = 0.0f;
    std::vector<f32> scores(static_cast<size_t>(n_keys));
    std::vector<f32> oh(g.v_dim);
    for (i32 h = 0; h < g.n_head; h++) {
        const f32 *qb = w.wq_b + static_cast<i64>(h) * head_slice;
        proj_strided(qn.data(), qb, qc.data(), g.q_lora, g.qk_nope, head_slice);
        proj_strided(qr.data(), qb + g.qk_nope, qc.data(), g.q_lora, g.qk_rope,
                     head_slice);
        const f32 *uk = w.wuk + static_cast<i64>(h) * g.kv_lora * g.qk_nope;
        const f32 *uv = w.wuv + static_cast<i64>(h) * g.kv_lora * g.v_dim;
        f32 mx = 0.0f;
        for (i64 t = 0; t < n_keys; t++) {
            const f32 *c = c_cache + t * g.kv_lora;
            const f32 *kr = kr_cache + t * g.qk_rope;
            f32 s = 0.0f;
            // K_nope[t] = c @ Wuk, dotted as it is built: same additions as
            // the absorbed path's qk . c, in the same per-element order.
            for (i64 d = 0; d < g.qk_nope; d++) {
                f32 k = 0.0f;
                for (i64 k2 = 0; k2 < g.kv_lora; k2++)
                    k += c[k2] * uk[k2 * g.qk_nope + d];
                s += qn[d] * k;
            }
            for (i64 d = 0; d < g.qk_rope; d++) s += qr[d] * kr[d];
            scores[static_cast<size_t>(t)] = s * scale;
            if (t == 0 || scores[static_cast<size_t>(t)] > mx)
                mx = scores[static_cast<size_t>(t)];
        }
        f32 sum = 0.0f;
        for (i64 t = 0; t < n_keys; t++) {
            const f32 e = std::exp(scores[static_cast<size_t>(t)] - mx);
            scores[static_cast<size_t>(t)] = e;
            sum += e;
        }
        for (i64 d = 0; d < g.v_dim; d++) oh[static_cast<size_t>(d)] = 0.0f;
        for (i64 t = 0; t < n_keys; t++) {
            const f32 p = scores[static_cast<size_t>(t)] / sum;
            const f32 *c = c_cache + t * g.kv_lora;
            for (i64 d = 0; d < g.v_dim; d++) {
                f32 v = 0.0f;
                for (i64 k2 = 0; k2 < g.kv_lora; k2++)
                    v += c[k2] * uv[k2 * g.v_dim + d];
                oh[static_cast<size_t>(d)] += p * v;
            }
        }
        // Concatenate heads, then the output projection.
        for (i64 d = 0; d < g.v_dim; d++) {
            const f32 *worow =
                w.wo + (static_cast<i64>(h) * g.v_dim + d) * g.n_embd;
            for (i64 i = 0; i < g.n_embd; i++)
                out[i] += oh[static_cast<size_t>(d)] * worow[i];
        }
    }
}

void mla_forward_absorbed(f32 *out, const f32 *x, const f32 *c_cache,
                          const f32 *kr_cache, i64 n_keys, const MlaWeights &w,
                          const f32 *wq_abs, const MlaGeometry &g, f32 scale,
                          f32 eps) {
    if (!out || !x || !c_cache || !kr_cache || !wq_abs || !g.valid() ||
        n_keys <= 0)
        return;
    const i64 head_slice = static_cast<i64>(g.qk_nope) + g.qk_rope;
    std::vector<f32> qc(g.q_lora), qn(g.qk_nope), qr(g.qk_rope);
    std::vector<f32> tmp(std::max<i64>(g.kv_lora, g.q_lora));
    std::vector<f32> qk(g.kv_lora), cv(g.kv_lora);
    proj(tmp.data(), w.wq_a, x, g.n_embd, g.q_lora);
    rms_norm(qc.data(), tmp.data(), w.wqa_norm, g.q_lora, eps);
    // The second norm's input: kv compression is normed at write time (the
    // cache holds normed rows), so the read path here must NOT norm again.
    // The test builds c_cache through the same rms_norm; double norming is
    // the silent-shrink defect this comment exists to prevent.
    for (i64 i = 0; i < g.n_embd; i++) out[i] = 0.0f;
    std::vector<f32> scores(static_cast<size_t>(n_keys));
    std::vector<f32> oh(g.v_dim);
    for (i32 h = 0; h < g.n_head; h++) {
        const f32 *qb = w.wq_b + static_cast<i64>(h) * head_slice;
        proj_strided(qn.data(), qb, qc.data(), g.q_lora, g.qk_nope, head_slice);
        proj_strided(qr.data(), qb + g.qk_nope, qc.data(), g.q_lora, g.qk_rope,
                     head_slice);
        // The absorbed query: qc @ Wq_abs[h], scored straight against c.
        const f32 *abs = wq_abs + static_cast<i64>(h) * g.q_lora * g.kv_lora;
        proj(qk.data(), abs, qc.data(), g.q_lora, g.kv_lora);
        const f32 *uv = w.wuv + static_cast<i64>(h) * g.kv_lora * g.v_dim;
        f32 mx = 0.0f;
        for (i64 t = 0; t < n_keys; t++) {
            const f32 *c = c_cache + t * g.kv_lora;
            const f32 *kr = kr_cache + t * g.qk_rope;
            f32 s = 0.0f;
            for (i64 k = 0; k < g.kv_lora; k++) s += qk[k] * c[k];
            for (i64 d = 0; d < g.qk_rope; d++) s += qr[d] * kr[d];
            scores[static_cast<size_t>(t)] = s * scale;
            if (t == 0 || scores[static_cast<size_t>(t)] > mx)
                mx = scores[static_cast<size_t>(t)];
        }
        f32 sum = 0.0f;
        for (i64 t = 0; t < n_keys; t++) {
            const f32 e = std::exp(scores[static_cast<size_t>(t)] - mx);
            scores[static_cast<size_t>(t)] = e;
            sum += e;
        }
        for (i64 k = 0; k < g.kv_lora; k++) cv[static_cast<size_t>(k)] = 0.0f;
        for (i64 t = 0; t < n_keys; t++) {
            const f32 p = scores[static_cast<size_t>(t)] / sum;
            const f32 *c = c_cache + t * g.kv_lora;
            for (i64 k = 0; k < g.kv_lora; k++)
                cv[static_cast<size_t>(k)] += p * c[k];
        }
        for (i64 d = 0; d < g.v_dim; d++) {
            f32 v = 0.0f;
            for (i64 k = 0; k < g.kv_lora; k++)
                v += cv[static_cast<size_t>(k)] * uv[k * g.v_dim + d];
            oh[static_cast<size_t>(d)] = v;
        }
        for (i64 d = 0; d < g.v_dim; d++) {
            const f32 *worow =
                w.wo + (static_cast<i64>(h) * g.v_dim + d) * g.n_embd;
            for (i64 i = 0; i < g.n_embd; i++)
                out[i] += oh[static_cast<size_t>(d)] * worow[i];
        }
    }
}

} // namespace krk
