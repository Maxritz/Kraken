// ============================================================================
//  hc.cpp — the host implementation of qwen4exp's hyper-connection mixing.
//
//  See include/krk/hc.hpp for the algebra. Everything here is plain f32 loops so
//  the CPU backend, the suite and the HIP kernels' oracle all agree on the
//  arithmetic order: lane by lane, element by element, one addition at a time.
// ============================================================================
#include "krk/hc.hpp"

#include <cmath>

namespace krk {
namespace {

// Numerically stable logistic. The reference's sigmoid saturates at the same
// place (ggml_sigmoid is 1/(1+exp(-x)) in f32), and the branch keeps a large
// negative injection from overflowing exp() to inf before the divide.
inline f32 sigmoid(f32 x) {
    if (x >= 0.0f) {
        const f32 e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
    const f32 e = std::exp(x);
    return e / (1.0f + e);
}

} // namespace

f32 hc_lane_weight(f32 inject, i32 hc) {
    if (hc <= 0) return 1.0f;
    // 2*sigmoid(0) is exactly 1.0f -- 0.5f is representable and 2*0.5f is exact
    // -- which is why an un-injected block is the identity on every lane.
    return 2.0f * sigmoid(inject / static_cast<f32>(hc));
}

void hc_norm_lanes(f32 *out, const f32 *x, const f32 *gamma, i32 n,
                   const HcGeometry &g, f32 eps) {
    if (!out || !x || !g.valid() || n <= 0) return;
    const i64 ne = static_cast<i64>(g.n_embd);
    for (i32 t = 0; t < n; t++) {
        for (i32 c = 0; c < g.hc; c++) {
            const i64 base = (static_cast<i64>(t) * g.hc + c) * ne;
            f64 ss = 0.0;
            for (i64 e = 0; e < ne; e++) {
                const f64 v = x[base + e];
                ss += v * v;
            }
            const f32 inv = 1.0f / std::sqrt(static_cast<f32>(ss / ne) + eps);
            const f32 *gc = gamma ? gamma + static_cast<i64>(c) * ne : nullptr;
            for (i64 e = 0; e < ne; e++) {
                const f32 v = x[base + e] * inv;
                out[base + e] = gc ? v * gc[e] : v;
            }
        }
    }
}

void hc_mix(f32 *mixed, const f32 *xn, const f32 *gate, i32 n,
            const HcGeometry &g) {
    if (!mixed || !xn || !gate || !g.valid() || n <= 0) return;
    const i64 ne = static_cast<i64>(g.n_embd);
    const f32 inv_hc = 1.0f / static_cast<f32>(g.hc);
    for (i32 t = 0; t < n; t++) {
        const i64 row = static_cast<i64>(t) * ne;
        for (i64 e = 0; e < ne; e++) mixed[row + e] = 0.0f;
        for (i32 c = 0; c < g.hc; c++) {
            const i64 base = (static_cast<i64>(t) * g.hc + c) * ne;
            for (i64 e = 0; e < ne; e++)
                mixed[row + e] += xn[base + e] * sigmoid(gate[base + e]);
        }
        // Divide once at the end rather than scaling every term: the reference
        // sums the lanes and scales, and matching its order is what keeps the
        // identical-lane case exact.
        for (i64 e = 0; e < ne; e++) mixed[row + e] *= inv_hc;
    }
}

void hc_combine(f32 *res, const f32 *block_out, const f32 *inject, i32 n,
                const HcGeometry &g) {
    if (!res || !g.valid() || n <= 0) return;
    const i64 ne = static_cast<i64>(g.n_embd);
    for (i32 t = 0; t < n; t++) {
        const i64 row = static_cast<i64>(t) * ne;
        const f32 *b = block_out ? block_out + row : nullptr;
        const f32 *inj = inject ? inject + static_cast<i64>(t) * g.hc : nullptr;
        for (i32 c = 0; c < g.hc; c++) {
            const i64 base = (static_cast<i64>(t) * g.hc + c) * ne;
            const f32 w = hc_lane_weight(inj ? inj[c] : 0.0f, g.hc);
            if (!b) continue; // nothing to add: the lane keeps its value
            for (i64 e = 0; e < ne; e++) res[base + e] += b[e] * w;
        }
    }
}

// ---- DeepSeek-V4.1: the Sinkhorn mixer ------------------------------------
//
// LAYOUT, because two conventions meet here. My `m` is dst-major:
// `m[d * hc + s]` is the weight destination lane d puts on source lane s. The
// FILE's `base` vector is the other way round -- ggml stores a [hc, hc] block
// with dim 0 fastest and dim 0 is the destination -- so the flat index k of
// `base[2*hc + k]` is `d + hc*s`, and hc_split() below reads it that way. Get
// this backwards and every matrix is transposed, which still gives a doubly
// stochastic matrix, so nothing complains and the lanes are mixed wrongly.
//
// The normalisation order is the reference's, including its names being the
// opposite of what they look like: `norm_cols` in llama.cpp permutes the matrix
// and sums over dim 0, which after the permutation is the SOURCE index -- so it
// divides each destination row by its sum over sources. `norm_rows` divides each
// source column by its sum over destinations. The first pass is sources only,
// then (iters-1) rounds of destinations-then-sources, and eps is added inside
// every normaliser as well as after the softmax.
void hc_sinkhorn_normalise(f32 *m, i32 hc, i32 iters, f32 eps) {
    if (!m || hc <= 0 || iters <= 0) return;
    const i64 hh = static_cast<i64>(hc) * hc;
    // softmax over the DESTINATION index, for each source: the reference's
    // ggml_soft_max over dim 0, which is dst.
    for (i32 s = 0; s < hc; s++) {
        f32 mx = m[s];
        for (i32 d = 1; d < hc; d++)
            mx = std::max(mx, m[static_cast<i64>(d) * hc + s]);
        f32 sum = 0.0f;
        for (i32 d = 0; d < hc; d++) {
            f32 &v = m[static_cast<i64>(d) * hc + s];
            v = std::exp(v - mx);
            sum += v;
        }
        const f32 inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (i32 d = 0; d < hc; d++) m[static_cast<i64>(d) * hc + s] *= inv;
    }
    for (i64 i = 0; i < hh; i++) m[i] += eps;
    // divide each destination row by its sum over sources (+ eps)
    auto norm_over_src = [&]() {
        for (i32 d = 0; d < hc; d++) {
            f32 *row = m + static_cast<i64>(d) * hc;
            f32 sum = 0.0f;
            for (i32 s = 0; s < hc; s++) sum += row[s];
            const f32 inv = 1.0f / (sum + eps);
            for (i32 s = 0; s < hc; s++) row[s] *= inv;
        }
    };
    // divide each source column by its sum over destinations (+ eps)
    auto norm_over_dst = [&]() {
        for (i32 s = 0; s < hc; s++) {
            f32 sum = 0.0f;
            for (i32 d = 0; d < hc; d++) sum += m[static_cast<i64>(d) * hc + s];
            const f32 inv = 1.0f / (sum + eps);
            for (i32 d = 0; d < hc; d++) m[static_cast<i64>(d) * hc + s] *= inv;
        }
    };
    norm_over_src();
    for (i32 it = 1; it < iters; it++) {
        norm_over_dst();
        norm_over_src();
    }
}

void hc_sinkhorn_split(f32 *pre, f32 *post, f32 *comb, const f32 *mixes,
                       const f32 *scale, const f32 *base, i32 n,
                       const HcSinkhornGeometry &g) {
    if (!mixes || !scale || !base || !g.valid() || n <= 0) return;
    const i32 hc = g.hc;
    const i64 hh = static_cast<i64>(hc) * hc;
    for (i32 t = 0; t < n; t++) {
        const f32 *mix = mixes + static_cast<i64>(t) * g.mix_dim();
        for (i32 c = 0; c < hc; c++) {
            if (pre)
                pre[static_cast<i64>(t) * hc + c] =
                    sigmoid(mix[c] * scale[0] + base[c]) + g.eps;
            if (post)
                post[static_cast<i64>(t) * hc + c] =
                    2.0f * sigmoid(mix[hc + c] * scale[1] + base[hc + c]);
        }
        if (!comb) continue;
        f32 *m = comb + static_cast<i64>(t) * hh;
        for (i32 d = 0; d < hc; d++) {
            for (i32 s = 0; s < hc; s++) {
                // file index d + hc*s, my index d*hc + s
                const i64 k = static_cast<i64>(d) + static_cast<i64>(hc) * s;
                m[static_cast<i64>(d) * hc + s] =
                    mix[2 * hc + k] * scale[2] + base[2 * hc + k];
            }
        }
        hc_sinkhorn_normalise(m, hc, g.sinkhorn_iters, g.eps);
    }
}

void hc_head_pre(f32 *out, const f32 *mixes, const f32 *scale, const f32 *base,
                 i32 hc, f32 eps) {
    if (!out || !mixes || !scale || !base || hc <= 0) return;
    for (i32 c = 0; c < hc; c++)
        out[c] = sigmoid(mixes[c] * scale[0] + base[c]) + eps;
}

void hc_sinkhorn_post(f32 *lanes_out, const f32 *lanes_in, const f32 *block_out,
                      const f32 *post, const f32 *comb, i32 n,
                      const HcSinkhornGeometry &g) {
    if (!lanes_out || !lanes_in || !g.valid() || n <= 0) return;
    const i64 ne = static_cast<i64>(g.n_embd);
    const i32 hc = g.hc;
    for (i32 t = 0; t < n; t++) {
        const i64 lrow = static_cast<i64>(t) * hc * ne;
        const f32 *bo = block_out ? block_out + static_cast<i64>(t) * ne : nullptr;
        const f32 *pt = post ? post + static_cast<i64>(t) * hc : nullptr;
        const f32 *cb = comb ? comb + static_cast<i64>(t) * hc * hc : nullptr;
        for (i32 d = 0; d < hc; d++) {
            f32 *dst = lanes_out + lrow + static_cast<i64>(d) * ne;
            const f32 w = pt ? pt[d] : 0.0f;
            for (i64 e = 0; e < ne; e++) dst[e] = bo ? bo[e] * w : 0.0f;
            if (!cb) continue;
            for (i32 s = 0; s < hc; s++) {
                const f32 c = cb[static_cast<i64>(d) * hc + s];
                const f32 *src = lanes_in + lrow + static_cast<i64>(s) * ne;
                for (i64 e = 0; e < ne; e++) dst[e] += src[e] * c;
            }
        }
    }
}

void hc_sinkhorn_pre(f32 *out, const f32 *lanes, const f32 *pre, i32 n,
                     const HcSinkhornGeometry &g) {
    if (!out || !lanes || !pre || !g.valid() || n <= 0) return;
    const i64 ne = static_cast<i64>(g.n_embd);
    const i32 hc = g.hc;
    for (i32 t = 0; t < n; t++) {
        f32 *o = out + static_cast<i64>(t) * ne;
        for (i64 e = 0; e < ne; e++) o[e] = 0.0f;
        for (i32 c = 0; c < hc; c++) {
            const f32 w = pre[static_cast<i64>(t) * hc + c];
            const f32 *src = lanes + (static_cast<i64>(t) * hc + c) * ne;
            for (i64 e = 0; e < ne; e++) o[e] += src[e] * w;
        }
    }
}

} // namespace krk
