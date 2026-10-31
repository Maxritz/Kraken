// ============================================================================
//  msa.cpp -- see include/krk/msa.hpp.
// ============================================================================
#include "krk/msa.hpp"

#include <algorithm>
#include <cmath>

namespace krk {

i32 msa_pool(f32 *out, const f32 *keys, i32 head_dim, i32 n_kv, i32 block) {
    if (!out || !keys || head_dim <= 0 || n_kv <= 0 || block <= 0) return 0;
    const i32 n_blocks = (n_kv + block - 1) / block;
    for (i32 b = 0; b < n_blocks; b++) {
        const i32 lo = b * block;
        const i32 hi = std::min(n_kv, lo + block);
        f32 *dst = out + static_cast<i64>(b) * head_dim;
        for (i32 d = 0; d < head_dim; d++) dst[d] = 0.0f;
        for (i32 k = lo; k < hi; k++) {
            const f32 *src = keys + static_cast<i64>(k) * head_dim;
            for (i32 d = 0; d < head_dim; d++) dst[d] += src[d];
        }
        // Mean over the keys the block HAS. Padding to `block` would make the
        // trailing block look weaker than the others purely because the context
        // ended, and the budget would then systematically drop the newest keys.
        const f32 inv = 1.0f / static_cast<f32>(hi - lo);
        for (i32 d = 0; d < head_dim; d++) dst[d] *= inv;
    }
    return n_blocks;
}

i32 msa_select(i64 *out, const f32 *pooled, const f32 *q, i32 head_dim,
               i32 n_kv, i32 n_qpos, i32 block, i32 budget) {
    if (!out || !pooled || !q || head_dim <= 0 || n_kv <= 0 || block <= 0)
        return 0;
    const i32 n_blocks = (n_kv + block - 1) / block;
    // Causality: a block is visible when it starts at or before the query's
    // last position. The local block is always visible by that rule.
    std::vector<std::pair<f32, i32>> scored;
    scored.reserve(static_cast<size_t>(n_blocks));
    for (i32 b = 0; b < n_blocks; b++) {
        if (static_cast<i64>(b) * block > n_qpos) break;
        const f32 *p = pooled + static_cast<i64>(b) * head_dim;
        f32 s = 0.0f;
        for (i32 d = 0; d < head_dim; d++) s += q[d] * p[d];
        scored.emplace_back(s, b);
    }
    if (scored.empty()) return 0;
    const i32 local = static_cast<i32>(std::min<i64>(
        n_qpos / block, static_cast<i64>(n_blocks) - 1));
    std::vector<i64> keep;
    if (budget <= 0 || budget >= static_cast<i32>(scored.size())) {
        for (const auto &s : scored) keep.push_back(s.second);
    } else {
        // Highest score first; ties broken by block id so two runs of the same
        // selection return the same set. nth_element would be faster and would
        // also make the tie-break depend on the container's internal order,
        // which is the one thing a selection must not do.
        std::sort(scored.begin(), scored.end(),
                  [](const std::pair<f32, i32> &a, const std::pair<f32, i32> &b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        const i32 take = std::min<i32>(budget, static_cast<i32>(scored.size()));
        for (i32 i = 0; i < take; i++) keep.push_back(scored[static_cast<size_t>(i)].second);
        // The local block is force-included: a query must always see the tokens
        // it is sitting on, whatever the pooled scores say. If the budget is
        // already full, the lowest-scoring non-local block gives up its place.
        if (std::find(keep.begin(), keep.end(), static_cast<i64>(local)) == keep.end()) {
            i32 worst = -1;
            f32 worst_s = INFINITY;
            for (size_t i = 0; i < keep.size(); i++) {
                if (static_cast<i32>(keep[i]) == local) continue;
                const f32 s = [&] {
                    for (const auto &c : scored)
                        if (c.second == static_cast<i32>(keep[i])) return c.first;
                    return 0.0f;
                }();
                if (s <= worst_s) {
                    worst_s = s;
                    worst = static_cast<i32>(i);
                }
            }
            if (worst >= 0) keep[static_cast<size_t>(worst)] = local;
            else keep.push_back(local);
        }
        std::sort(keep.begin(), keep.end());
    }
    for (size_t i = 0; i < keep.size(); i++) out[i] = keep[i];
    return static_cast<i32>(keep.size());
}

i32 msa_runs(i64 *out, const i64 *blocks, i32 n_blocks, i32 block, i32 n_kv) {
    if (!out || !blocks || n_blocks <= 0 || block <= 0) return 0;
    i32 n_runs = 0;
    i64 lo = blocks[0] * block;
    i64 hi = std::min<i64>(n_kv, lo + block);
    for (i32 i = 1; i < n_blocks; i++) {
        const i64 b_lo = blocks[i] * block;
        const i64 b_hi = std::min<i64>(n_kv, b_lo + block);
        if (b_lo == hi) {
            hi = b_hi;  // adjacent: extend the run
        } else {
            out[2 * n_runs + 0] = lo;
            out[2 * n_runs + 1] = hi;
            n_runs++;
            lo = b_lo;
            hi = b_hi;
        }
    }
    out[2 * n_runs + 0] = lo;
    out[2 * n_runs + 1] = hi;
    return n_runs + 1;
}

} // namespace krk
