// ============================================================================
//  dsv4_compress.cpp — DeepSeek-V4 compressed attention, host reference.
//
//  Plain f32 loops, one addition at a time, for the same reason mla_shared.cpp
//  is written that way: the CPU path, the suite and any kernel A/B all compare
//  against this file, so it must be the slowest and most literal version of the
//  arithmetic rather than the fastest.
//
//  See include/krk/dsv4_compress.hpp for the graph and for the three details
//  (per-channel softmax, rope tail at the compress base, CSA overlap halves)
//  that are finite-but-wrong when read the other way round.
// ============================================================================
#include "krk/dsv4_compress.hpp"

#include <cmath>
#include <vector>

namespace krk {
namespace {

// out[j] = sum_i a[i*cols + j] * v[i], a row-major [rows, cols].
void proj(f32 *out, const f32 *a, const f32 *v, i64 rows, i64 cols) {
    for (i64 j = 0; j < cols; j++) out[j] = 0.0f;
    for (i64 i = 0; i < rows; i++) {
        const f32 vi = v[i];
        if (vi == 0.0f) continue;
        const f32 *row = a + i * cols;
        for (i64 j = 0; j < cols; j++) out[j] += vi * row[j];
    }
}

// y = rmsnorm(v) * gamma, the engine's row_norm formula (a f64 accumulator, so
// the result does not depend on the row length's summation order).
void rms_norm(f32 *y, const f32 *v, const f32 *gamma, i64 n, f32 eps) {
    f64 ss = 0.0;
    for (i64 i = 0; i < n; i++) ss += static_cast<f64>(v[i]) * v[i];
    const f32 inv = 1.0f / std::sqrt(static_cast<f32>(ss / n) + eps);
    for (i64 i = 0; i < n; i++) y[i] = v[i] * inv * (gamma ? gamma[i] : 1.0f);
}

} // namespace

void dsv4_comp_inv_freq(f32 *out, i32 rope, f32 base) {
    if (!out || rope <= 0 || base <= 0.0f) return;
    for (i32 j = 0; j < rope / 2; j++)
        out[j] = std::pow(base, -static_cast<f32>(2 * j) / static_cast<f32>(rope));
}

void dsv4_comp_rope(f32 *v, i32 hd, i32 rope, i64 pos, f32 base) {
    if (!v || hd <= 0 || rope <= 0 || rope > hd || base <= 0.0f) return;
    const i32 off = hd - rope; // the rotary dims are the TAIL
    for (i32 i = 0; i + 1 < rope; i += 2) {
        // theta = pos / base^(i / rope): the same exponent convention as the
        // attention rope, so a file that sets both bases apart gets two
        // different angles for the same position -- which is the point.
        const f32 inv_freq =
            std::pow(base, -static_cast<f32>(i) / static_cast<f32>(rope));
        const f32 ang = static_cast<f32>(pos) * inv_freq;
        const f32 c = std::cos(ang), s = std::sin(ang);
        const f32 a = v[off + i], b = v[off + i + 1];
        v[off + i] = a * c - b * s;
        v[off + i + 1] = a * s + b * c;
    }
}

bool dsv4_comp_state(const Dsv4Compressor &c, const f32 *x, i32 pos_in_block,
                     f32 *kv_out, f32 *score_out) {
    if (!c.valid() || !x || !kv_out || !score_out) return false;
    if (pos_in_block < 0 || pos_in_block >= c.ratio) return false;
    const i32 w = c.state_width();
    proj(kv_out, c.wkv, x, c.n_embd, w);
    proj(score_out, c.wgate, x, c.n_embd, w);
    // APE is [coff*hd, ratio]: column `pos_in_block`, one value per channel.
    for (i32 i = 0; i < w; i++)
        score_out[i] += c.ape[static_cast<i64>(i) * c.ratio + pos_in_block];
    return true;
}

bool dsv4_comp_pool(const Dsv4Compressor &c, const f32 *kv, const f32 *score,
                    f32 *out) {
    if (!c.valid() || !kv || !score || !out) return false;
    const i32 hd = c.hd;
    const i32 coeff = c.coff();
    const i32 n_rows = c.pool_rows();

    // Values in pool order. For the dense layout row t is just its state row,
    // narrowed to hd (coff == 1 makes that the whole row). For the overlapping
    // layout rows [0, ratio) are the PREVIOUS block's tokens and contribute
    // their first half; rows [ratio, 2*ratio) are this block's tokens and
    // contribute their second half.
    std::vector<f32> val(static_cast<size_t>(n_rows) * hd);
    std::vector<f32> sc(static_cast<size_t>(n_rows) * hd);
    const i32 wdt = c.state_width();
    for (i32 t = 0; t < n_rows; t++) {
        const bool cur = (coeff == 1) || (t >= c.ratio);
        const i64 src_off = static_cast<i64>(t) * wdt +
                            (coeff == 1 ? 0 : (cur ? hd : 0));
        for (i32 i = 0; i < hd; i++) {
            val[static_cast<size_t>(t) * hd + i] = kv[src_off + i];
            sc[static_cast<size_t>(t) * hd + i] = score[src_off + i];
        }
    }

    // Per-channel softmax over the rows, then the weighted sum of that
    // channel's values. Channel-major so the inner loop is over rows.
    std::vector<f32> acc(static_cast<size_t>(hd), 0.0f);
    for (i32 i = 0; i < hd; i++) {
        f32 mx = -INFINITY;
        for (i32 t = 0; t < n_rows; t++) {
            const f32 s = sc[static_cast<size_t>(t) * hd + i];
            if (s > mx) mx = s;
        }
        f32 sum = 0.0f;
        for (i32 t = 0; t < n_rows; t++) {
            const f32 e = std::exp(sc[static_cast<size_t>(t) * hd + i] - mx);
            sc[static_cast<size_t>(t) * hd + i] = e;
            sum += e;
        }
        if (sum <= 0.0f) continue; // every row was -inf: no mass, key stays 0
        f32 a = 0.0f;
        for (i32 t = 0; t < n_rows; t++)
            a += sc[static_cast<size_t>(t) * hd + i] * val[static_cast<size_t>(t) * hd + i];
        acc[static_cast<size_t>(i)] = a / sum;
    }

    rms_norm(out, acc.data(), c.norm, hd, c.eps);
    return true;
}

bool dsv4_indexer_scores(i32 n_head, i32 head, const f32 *q, const f32 *k,
                         const f32 *w, const f32 *mask, i32 n_blocks,
                         f32 *out) {
    if (n_head <= 0 || head <= 0 || !q || !k || !out) return false;
    if (n_blocks < 0) return false;
    for (i32 b = 0; b < n_blocks; b++) {
        f32 s = 0.0f;
        const f32 *kb = k + static_cast<i64>(b) * head;
        for (i32 h = 0; h < n_head; h++) {
            const f32 *qh = q + static_cast<i64>(h) * head;
            f32 dot = 0.0f;
            for (i32 i = 0; i < head; i++) dot += qh[i] * kb[i];
            // relu THEN weight: the weight scales a non-negative similarity,
            // so the sum is non-negative and a head cannot vote against a key.
            if (dot > 0.0f && w) s += dot * w[h];
        }
        if (mask) s += mask[b];
        out[b] = s;
    }
    return true;
}

i32 dsv4_top_k(i32 *idx, i32 k, const f32 *scores, i32 n) {
    if (!idx || !scores || n <= 0 || k <= 0) return 0;
    if (k > n) k = n;
    for (i32 j = 0; j < k; j++) {
        i32 best = -1;
        for (i32 i = 0; i < n; i++) {
            // Already chosen? The chosen ones are tracked by marking, below.
            bool taken = false;
            for (i32 t = 0; t < j; t++)
                if (idx[t] == i) {
                    taken = true;
                    break;
                }
            if (taken) continue;
            // Strictly greater, so an earlier index wins a tie: the selection
            // is a function of the scores alone.
            if (best < 0 || scores[i] > scores[best]) best = i;
        }
        if (best < 0) break;
        idx[j] = best;
    }
    return k;
}

} // namespace krk
