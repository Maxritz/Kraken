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

} // namespace krk
