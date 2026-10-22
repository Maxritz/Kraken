// ============================================================================
//  qsa.cpp — the QSA indexer's host maths (see include/krk/qsa.hpp).
// ============================================================================
#include "krk/qsa.hpp"

#include "par_pool.hpp"

#include <algorithm>
#include <cmath>

namespace krk {

namespace {

// The same shape as backend_cpu.cpp's helper: below the flop floor the pool
// round trip costs more than the arithmetic, so small rows stay serial.
constexpr f64 kParFlopFloor = 1.0e5;

template <class F>
inline void par_for(i64 n, i64 grain, f64 est_flops, F &&body) {
    if (n <= 0) return;
    if (est_flops < kParFlopFloor || grain < 1) {
        body(static_cast<i64>(0), n);
        return;
    }
    Pool::get().run(n, grain, std::forward<F>(body));
}

} // namespace

i32 qsa_pool(f32 *out, const f32 *keys, i32 head_dim, i32 n_kv, i32 ratio) {
    if (!out || !keys || head_dim <= 0 || ratio <= 0 || n_kv <= 0) return 0;
    const i32 n_blocks = n_kv / ratio; // the trailing partial block is not pooled
    par_for(n_blocks, 1, 2.0 * static_cast<f64>(n_blocks) * ratio * head_dim,
            [&](i64 b, i64 e) {
                for (i64 blk = b; blk < e; blk++) {
                    const f32 *base = keys + blk * ratio * head_dim;
                    f32 *dst = out + blk * head_dim;
                    for (i32 d = 0; d < head_dim; d++) dst[d] = 0.0f;
                    for (i32 k = 0; k < ratio; k++) {
                        const f32 *src = base + static_cast<i64>(k) * head_dim;
                        for (i32 d = 0; d < head_dim; d++) dst[d] += src[d];
                    }
                    const f32 inv = 1.0f / static_cast<f32>(ratio);
                    for (i32 d = 0; d < head_dim; d++) dst[d] *= inv;
                }
            });
    return n_blocks;
}

void qsa_norm_rope(f32 *x, const f32 *w, i32 n_rows, i32 head_dim,
                   const i64 *pos, const QsaConfig &cfg) {
    if (!x || !w || !pos || n_rows <= 0 || head_dim <= 0) return;
    const i32 half = head_dim / 2;
    const i32 rot = std::max<i32>(1, std::min<i32>(cfg.n_rot > 0 ? cfg.n_rot / 2 : 0,
                                                   half));
    const f32 s = cfg.rope_scale > 0 ? cfg.rope_scale : 1.0f;
    par_for(n_rows, 1, 6.0 * static_cast<f64>(n_rows) * head_dim,
            [&](i64 b, i64 e) {
                for (i64 r = b; r < e; r++) {
                    f32 *row = x + r * head_dim;
                    f32 ss = 0;
                    for (i32 i = 0; i < head_dim; i++) ss += row[i] * row[i];
                    const f32 inv =
                        1.0f / std::sqrt(ss / static_cast<f32>(head_dim) + cfg.eps);
                    for (i32 i = 0; i < head_dim; i++) row[i] = row[i] * inv * w[i];
                    if (!cfg.inv_freq || rot <= 0) continue;
                    const f32 p = static_cast<f32>(pos[r]) / s;
                    for (i32 i = 0; i < rot; i++) {
                        const f32 theta = p * cfg.inv_freq[i];
                        const f32 c = std::cos(theta), sn = std::sin(theta);
                        const i32 ia = cfg.rope_neox ? i : 2 * i;
                        const i32 ib = cfg.rope_neox ? i + rot : 2 * i + 1;
                        const f32 a = row[ia], bb = row[ib];
                        row[ia] = a * c - bb * sn;
                        row[ib] = a * sn + bb * c;
                    }
                }
            });
}

void qsa_scores(f32 *score, const f32 *q, const f32 *pooled, i32 n_tok,
                i32 n_head, i32 n_blocks, i32 head_dim) {
    if (!score || !q || !pooled || n_tok <= 0 || n_head <= 0 || n_blocks <= 0 ||
        head_dim <= 0)
        return;
    // (token, BLOCK) pairs own one output cell each: a cell's whole head sum is
    // computed here and stored once, so no two units ever touch the same f32.
    //
    // It has to be that way. The natural split -- one unit per (token, head),
    // `dst[bb] +=` under a per-head rectification -- is what this was, and it
    // was a race: the row's zeroing ran inside head 0's unit, so a thread
    // holding head 1 could accumulate onto the PREVIOUS chunk's row before
    // head 0 cleared it. The result was a score row that depended on the thread
    // count and on what the buffer last held, which read as a flaky selection:
    // at 36 tokens and blocks of two, the same prompt produced the first
    // seventeen cells in index order (every score equal) on one run and a real
    // ranking on the next. Summing in head order into a local keeps the
    // reference's own addition order (`sum_h relu(...)`, heads ascending) and
    // makes a chunk-1 prefill and a chunk-8 prefill agree.
    const i64 units = static_cast<i64>(n_tok) * n_blocks;
    par_for(units, 1, 2.0 * static_cast<f64>(units) * n_head * head_dim,
            [&](i64 b0, i64 e) {
                for (i64 u = b0; u < e; u++) {
                    const i64 t = u / n_blocks;
                    const i32 bb = static_cast<i32>(u % n_blocks);
                    const f32 *kh = pooled + static_cast<i64>(bb) * head_dim;
                    f32 s = 0;
                    for (i32 h = 0; h < n_head; h++) {
                        const f32 *qh = q + (static_cast<i64>(t) * n_head + h) * head_dim;
                        f32 dot = 0;
                        for (i32 d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                        // rectified per head, THEN summed -- the lightning
                        // indexer's rule, and the reason a head cannot cancel
                        // another's evidence.
                        if (dot > 0) s += dot;
                    }
                    score[t * n_blocks + bb] = s;
                }
            });
}

i32 qsa_select(std::vector<i64> *cells, const f32 *score, i32 n_blocks,
               i32 n_kv, i64 qpos, const QsaConfig &cfg) {
    if (!cells) return 0;
    cells->clear();
    if (!cfg.valid() || n_kv <= 0 || qpos < 0) return 0;
    const i32 r = cfg.ratio;
    const i64 cap = std::min<i64>(n_kv, cfg.budget());
    if (cap <= 0) return 0;

    // The block the query sits in, and whether it is still filling: its last
    // cell position is past the query, so no score for it exists yet and the
    // reference pins it visible instead.
    const i32 tail_blk = static_cast<i32>(qpos / r);
    const bool force_tail = (static_cast<i64>(tail_blk) * r + r - 1) > qpos;

    std::vector<i32> order;
    order.reserve(static_cast<size_t>(n_blocks));
    for (i32 b = 0; b < n_blocks; b++) {
        // A block that starts past the query is not addressable yet.
        if (static_cast<i64>(b) * r > qpos) break;
        order.push_back(b);
    }
    // Highest score first; ties by position, so a run replays exactly.
    std::sort(order.begin(), order.end(), [&](i32 a, i32 b) {
        if (score[a] != score[b]) return score[a] > score[b];
        return a < b;
    });
    if (force_tail) {
        auto it = std::find(order.begin(), order.end(), tail_blk);
        if (it != order.end()) order.erase(it);
        // The pinned block need not have a score at all: `n_blocks` counts the
        // COMPLETE blocks, so when n_kv is not a multiple of `ratio` the block
        // the query sits in is the one past the pooled range. It still has
        // cells, and cell qpos is the query's own -- dropping it because the
        // block has no mean yet would make the layer blind to the token it is
        // predicting from. The reference gives that block a bias instead of a
        // score (its `dead_bid`), which is the same pin: first, then its
        // visible cells.
        order.insert(order.begin(), tail_blk);
    }

    // Cells of block b that exist and are visible to this query.
    auto cells_of = [&](i32 b) -> i64 {
        const i64 lo = static_cast<i64>(b) * r;
        const i64 hi = std::min<i64>(lo + r, qpos + 1);
        return hi > lo ? hi - lo : 0;
    };

    i64 count = 0;
    for (i32 b : order) {
        const i64 n = cells_of(b);
        if (n <= 0) continue;
        const i64 take = std::min<i64>(n, cap - count);
        for (i64 i = 0; i < take; i++) cells->push_back(static_cast<i64>(b) * r + i);
        count += take;
        if (count >= cap) break;
    }
    std::sort(cells->begin(), cells->end());
    return static_cast<i32>(cells->size());
}

} // namespace krk
