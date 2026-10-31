// backend_cpu.cpp — the portable reference backend.
//
// This is the oracle: it is simple and obviously correct. Ops run
// across INDEPENDENT units (output columns, attention heads, token
// rows) on the worker pool, but no reduction is ever split, so
// results stay bit-identical to the serial path — the GPU backends
// are tested against exactly these numbers. Activations are f32
// here, which is why the engine asks the backend for its activation
// type instead of assuming f16.
#include "krk/backend.hpp"
#include "krk/kernel_time.hpp"
#include "krk/hc.hpp"
#include "krk/vram_guard.hpp"
#include "npu/npu_offload.hpp"
#include "par_pool.hpp"

#include <algorithm>
#include <cfloat>
#include <mutex>
#include <unordered_map> // FLT_MAX bounding the SIMT/CPU rsqrt and the clamp paths
#include <cmath>
#include <cstdlib>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <vector>

namespace krk {

namespace {
// Static names so the recorder compares pointers, not strings. The dtype is in
// the row name because "gemm_other" hid the fact that one format was 41% of a
// decode run.
inline const char *dtype_tag_r1(DType t, i64 n_in) {
    const bool i8ok = (n_in % 256) == 0 && vec_dot_int8();
    switch (t) {
        case DType::Q4_K: return i8ok ? "gemm_q4k_i8_r1" : "gemm_q4k_x_r1";
        case DType::Q6_K: return "gemm_q6k_r1";
        case DType::Q3_K: return "gemm_q3k_r1";
        case DType::Q5_K: return "gemm_q5k_r1";
        case DType::Q8_0: return i8ok ? "gemm_q8_0_i8_r1" : "gemm_q8_0_r1";
        case DType::F32:  return "gemm_f32_r1";
        case DType::F16:  return "gemm_f16_r1";
        default:          return "gemm_oth_r1";
    }
}
inline const char *dtype_tag_rn(DType t, i64 n_in) {
    const bool i8ok = (n_in % 256) == 0 && vec_dot_int8();
    switch (t) {
        case DType::Q4_K: return i8ok ? "gemm_q4k_i8_rN" : "gemm_q4k_x_rN";
        case DType::Q6_K: return "gemm_q6k_rN";
        case DType::Q3_K: return "gemm_q3k_rN";
        case DType::Q5_K: return "gemm_q5k_rN";
        case DType::Q8_0: return i8ok ? "gemm_q8_0_i8_rN" : "gemm_q8_0_rN";
        case DType::F32:  return "gemm_f32_rN";
        case DType::F16:  return "gemm_f16_rN";
        default:          return "gemm_oth_rN";
    }
}


// Below this many estimated flops, the pool round trip costs more
// than the computation, so ops stay serial. Decode's elementwise ops
// (RMSNorm over one 576-wide row, RoPE over 9 heads, silu_mul over
// 1536 lanes) fall under the floor; every GEMM, the head GEMV
// (~57 MFLOP) and attention at context length do not. The pool's
// spin-then-park wake policy makes the round trip nearly free when
// ops arrive back to back, so the floor can stay low.
constexpr f64 kParFlopFloor = 1.0e5;

// The gated delta net's short convolution is a fixed 4-tap FIR in the shipped
// checkpoints; 16 is a generous ceiling that still keeps the rolling window on
// the stack.
constexpr i64 kMaxConvKernel = 16;

inline f32 silu_scalar(f32 x) { return x / (1.0f + std::exp(-x)); }

// Parallel-for over disjoint index ranges. `est_flops` gates pool use;
// the serial fallback skips std::function construction entirely.
template <class F>
inline void par_for(i64 n, i64 grain, f64 est_flops, F &&body) {
    if (n <= 0) return;
    // The floor decides THROUGHPUT only: the same compiled body runs either
    // way (see Pool::run's note -- two copies of the arithmetic round
    // differently, and this backend's whole contract is that its numbers do
    // not depend on how the work was scheduled).
    const bool split = est_flops >= kParFlopFloor && grain >= 1;
    Pool::get().run(n, grain, split, std::forward<F>(body));
}

#if defined(KRK_DOT_AVX2)
// ---- vector transcendentals for the elementwise stages ---------------------
//
// `silu_mul` was 7.6% of a run and 13 ms of a 134 ms prefill: a scalar loop
// calling std::exp once per element. exp is ~20 ns of latency and nothing in
// that loop could overlap it, so the FFN activation cost more than the GEMM
// that feeds it.
//
// exp here is the Cephes polynomial evaluated eight at a time, with 2^n folded
// in by adding to the exponent field. Its relative error is ~1e-7 -- larger
// than a correctly-rounded libm exp, which is why this lives behind the WIDE
// tier and not in the reference path: the default build still calls std::exp
// and still produces the bits the golden hashes pin.
#define KRK_EXP_POLY(dst, x)                                                  \
    do {                                                                      \
        const __m256 _hi = _mm256_set1_ps(88.3762626647949f);                 \
        const __m256 _lo = _mm256_set1_ps(-87.3365478515625f);                \
        const __m256 _l2e = _mm256_set1_ps(1.44269504088896341f);             \
        __m256 y = _mm256_min_ps(_mm256_max_ps((x), _lo), _hi);               \
        __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(y, _l2e,                 \
                                                    _mm256_set1_ps(0.5f)));   \
        y = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), y);            \
        y = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), y);         \
        const __m256 z = _mm256_mul_ps(y, y);                                 \
        __m256 p = _mm256_set1_ps(1.9875691500e-4f);                          \
        p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.3981999507e-3f));          \
        p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(8.3334519073e-3f));          \
        p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(4.1665795894e-2f));          \
        p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.6666665459e-1f));          \
        p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(5.0000001201e-1f));          \
        p = _mm256_fmadd_ps(p, z, _mm256_add_ps(y, _mm256_set1_ps(1.0f)));    \
        const __m256i n = _mm256_cvtps_epi32(fx);                             \
        const __m256i e2 = _mm256_slli_epi32(_mm256_add_epi32(                \
                                                n, _mm256_set1_epi32(127)),   \
                                            23);                              \
        (dst) = _mm256_mul_ps(p, _mm256_castsi256_ps(e2));                    \
    } while (0)

__attribute__((target("avx2,fma"))) void silu_mul_avx2(f32 *o, const f32 *g,
                                                      const f32 *u, i64 n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(g + i);
        __m256 e;
        // exp(-x), evaluated on -x directly: one negation of the sign bit beats
        // a reciprocal and keeps the accuracy of the polynomial.
        KRK_EXP_POLY(e, _mm256_xor_ps(x, sign));
        const __m256 silu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_storeu_ps(o + i, _mm256_mul_ps(silu, _mm256_loadu_ps(u + i)));
    }
    for (; i < n; i++) {
        const f32 x = g[i];
        o[i] = (x / (1.0f + std::exp(-x))) * u[i];
    }
}
#endif // KRK_DOT_AVX2

void rmsnorm_rows(f32 *out, const f32 *x, const f32 *w, i64 rows, i64 n, f32 eps,
                  const i32 *row_exp = nullptr) {
    // Rows are independent and each row keeps its own reduction.
    par_for(rows, 1, 4.0 * static_cast<f64>(rows) * n,
            [&](i64 b, i64 e) {
                for (i64 r = b; r < e; r++) {
                    const f32 *xr = x + r * n;
                    f32 *orow = out + r * n;
                    // f32 activations already have the exponent range the
                    // residual storage contract exists to restore, so this
                    // backend never raises an exponent and every row it reads is
                    // true scale. The factor is applied anyway (2^0 == 1) so both
                    // backends read the same array.
                    const f32 sc = row_exp ? std::ldexp(1.0f, row_exp[r]) : 1.0f;
                    f32 ss = 0;
                    for (i64 i = 0; i < n; i++) {
                        const f32 v = xr[i] * sc;
                        ss += v * v;
                    }
                    const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(n) + eps);
                    // A null weight is the weightless norm (gemma4's value
                    // row): one multiply per element fewer, which is the only
                    // difference from a gamma of ones and must not be a sparse
                    // branch inside the loop.
                    if (w) {
                        for (i64 i = 0; i < n; i++) orow[i] = xr[i] * sc * inv * w[i];
                    } else {
                        for (i64 i = 0; i < n; i++) orow[i] = xr[i] * sc * inv;
                    }
                }
            });
}

// ---------------------------------------------------------------------------
// Attention, one (token, head) row at a time.
//
// The reference body here is a scalar loop, and at any context length worth
// calling real, attention is the TOP row of the profile: measured on Tiny-Moe
// with a 1400-token prompt, 63.4% of the run at 5.2% of the f32 peak, while the
// GEMMs that had absorbed three rounds of kernel work were down to 27%.
//
// Two loops matter and they want different treatment:
//
//   score   s_j = <q, k_j>          a REDUCTION over hd
//   output  o_i = sum_j p_j * v_j   an accumulation over j, indexed by i
//
// The output loop is vectorised ACROSS i, so every output dimension still sums
// over j in the same order it always did -- that half is bit-identical to the
// scalar body, not merely close. The score loop has to reduce, so a horizontal
// sum is unavoidable and it changes the order of the last few bits. That is the
// same trade the int8 GEMV kernels make, and the tests hold it to their
// tolerances rather than to memcmp.
//
// Neither loop can be made 8x: each is roughly one load and one FMA per eight
// lanes, and the score loop pays a horizontal reduce per key. Measured below.
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define KRK_ATTN_AVX2 1
#endif

#if defined(KRK_ATTN_AVX2)
__attribute__((target("avx2,fma"))) inline f32 attn_dot_avx2(const f32 *a,
                                                             const f32 *b,
                                                             i64 n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    i64 i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                             _mm256_loadu_ps(b + i + 8), a1);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    a0 = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(a0);
    __m128 hi = _mm256_extractf128_ps(a0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    f32 s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

// 64 dimensions of q, held in REGISTERS across every key of the row.
//
// The obvious inner loop reloads q for each key, and that is the whole cost:
// with 256-bit operands every FMA needs one load for k and one for q, so the
// loop is load-port bound at twice the FMA count and the FMA units idle. q is
// the same 64 floats for all n_vis keys, so it belongs in registers -- that
// halves the loads and takes the loop to the FMA limit instead.
//
// Four accumulators, two FMAs deep each. A single accumulator is a 64-deep
// dependent chain (measured: 59 ns for one hd=64 scalar dot, i.e. ~153 cycles
// for 64 multiply-adds), which is why the reference body lost to latency rather
// than to arithmetic. Four chains is 8 cycles of latency for the same work.
__attribute__((target("avx2,fma"))) inline void attn_score64(
    const f32 *qh, const f32 *kc, i64 pos_stride, i64 k_off, i64 j_lo,
    i64 n_vis, f32 *scores) {
    const __m256 q0 = _mm256_loadu_ps(qh + 0);
    const __m256 q1 = _mm256_loadu_ps(qh + 8);
    const __m256 q2 = _mm256_loadu_ps(qh + 16);
    const __m256 q3 = _mm256_loadu_ps(qh + 24);
    const __m256 q4 = _mm256_loadu_ps(qh + 32);
    const __m256 q5 = _mm256_loadu_ps(qh + 40);
    const __m256 q6 = _mm256_loadu_ps(qh + 48);
    const __m256 q7 = _mm256_loadu_ps(qh + 56);
    const f32 *kj = kc + j_lo * pos_stride + k_off;
    for (i64 j = 0; j < n_vis; j++, kj += pos_stride) {
        __m256 a0 = _mm256_mul_ps(q0, _mm256_loadu_ps(kj + 0));
        __m256 a1 = _mm256_mul_ps(q1, _mm256_loadu_ps(kj + 8));
        __m256 a2 = _mm256_mul_ps(q2, _mm256_loadu_ps(kj + 16));
        __m256 a3 = _mm256_mul_ps(q3, _mm256_loadu_ps(kj + 24));
        a0 = _mm256_fmadd_ps(q4, _mm256_loadu_ps(kj + 32), a0);
        a1 = _mm256_fmadd_ps(q5, _mm256_loadu_ps(kj + 40), a1);
        a2 = _mm256_fmadd_ps(q6, _mm256_loadu_ps(kj + 48), a2);
        a3 = _mm256_fmadd_ps(q7, _mm256_loadu_ps(kj + 56), a3);
        const __m256 s01 = _mm256_add_ps(a0, a1);
        const __m256 s23 = _mm256_add_ps(a2, a3);
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(_mm256_add_ps(s01, s23)),
                              _mm256_extractf128_ps(_mm256_add_ps(s01, s23), 1));
        t = _mm_hadd_ps(t, t);
        t = _mm_hadd_ps(t, t);
        scores[j] += _mm_cvtss_f32(t);
    }
}

// The visible-key window for one query row, computed once and shared.
//
// It used to be written out three times -- the scalar reference, the AVX2 row
// and the NPU hand-off -- because each was added at a different time. Three
// copies of `window`, `skip_from`, the causal clamp and the two masked-run
// bounds is three chances for the paths to disagree about which keys a row can
// see, and a disagreement there is silent: the output stays plausible and only
// a fixture comparison finds it. The arithmetic lives here; the paths consume
// the result.
struct AttnRange {
    i64 n_keys = 0;  // keys this row may attend to, from position 0
    i64 j_lo = 0;    // first visible key
    i64 n_vis = 0;   // j_lo + n_vis == n_keys
    i64 m0 = 0;      // masked sub-run, j-relative
    i64 m1 = 0;
};

// A row's own position. With no per-row table this is the old consecutive
// layout; with one, each row carries its own -- which is what lets a single
// call mix several sequences at different points in their history.
// ALiBi slope for one head, exactly as llama.cpp derives it.
//
// ggml-cpu's soft_max_ext builds the same table from the same
// `attention.max_alibi_bias` key and the same head count:
//
//     n_head_log2 = 1 << (uint32_t) floor(log2(n_head));
//     m0 = powf(2, -max_bias / n_head_log2);
//     m1 = powf(2, -(max_bias/2) / n_head_log2);
//     slope(h) = h < n_head_log2 ? powf(m0, h + 1)
//                                : powf(m1, 2*(h - n_head_log2) + 1);
//
// The floor(log2(n_head)) is the part worth naming: for a head count that is
// not a power of two the geometric ramp is anchored at the largest power of two
// BELOW it, so the tail heads take the halved base instead. Getting this wrong
// is not a rounding difference -- it re-scales every head's distance penalty,
// which decodes as plausible text with the wrong context weighting.
inline f32 alibi_slope_impl(i64 h, i64 n_head, f32 max_bias) {
    if (max_bias <= 0.0f) return 0.0f;
    u32 nh = static_cast<u32>(n_head > 0 ? n_head : 1);
    u32 nh_log2 = 1u;
    while ((nh_log2 << 1) <= nh) nh_log2 <<= 1;
    const f32 m0 = std::pow(2.0f, -max_bias / static_cast<f32>(nh_log2));
    const f32 m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<f32>(nh_log2));
    const u32 hu = static_cast<u32>(h);
    return hu < nh_log2 ? std::pow(m0, static_cast<f32>(hu + 1))
                        : std::pow(m1, static_cast<f32>(2 * (hu - nh_log2) + 1));
}

inline i64 attn_pos(const AttnDesc &d, i64 t) {
    return d.row_pos ? d.row_pos[t] : d.pos0 + t;
}

// A row's element offset into the KV plane, before layer_stride/pos_stride.
// Zero for the single-sequence layout, where the caller already offset the
// plane to the start of its context.
inline i64 attn_kv_base(const AttnDesc &d, i64 t) {
    return d.row_kv_base ? d.row_kv_base[t] : 0;
}

// True when this call addresses more than one KV region. The gather path
// copies one head's slices out of ONE contiguous run of keys, so it has no
// meaning here and the per-row paths below are used instead.
inline bool attn_multiseq(const AttnDesc &d) {
    return d.row_pos != nullptr || d.row_kv_base != nullptr;
}

inline AttnRange attn_range(const AttnDesc &d, i64 t) {
    AttnRange r;
    const i64 pos = attn_pos(d, t);
    // Non-causal batches are contiguous by construction today, so the chunk's
    // last position is the bound. A per-row table changes that, and there is
    // no caller that both wants non-causal attention and mixes sequences, so
    // the multiseq case uses each row's own position instead of guessing.
    r.n_keys = d.causal ? pos + 1
                        : (d.row_pos ? pos + 1 : d.pos0 + d.n_tok);
    r.j_lo = d.window > 0 ? std::max<i64>(0, r.n_keys - d.window) : 0;
    r.n_vis = r.n_keys - r.j_lo;
    // skip_from: a query at pos >= skip_from keeps the context before
    // skip_from and its own row, and sees its block's siblings as masked. That
    // is one contiguous run in j, so the hot loop carries no branch.
    if (d.skip_from >= 0 && pos >= d.skip_from) {
        const i64 s_lo = std::max(r.j_lo, d.skip_from);
        const i64 s_hi = std::min(pos, r.n_keys);
        r.m0 = std::max<i64>(0, s_lo - r.j_lo);
        r.m1 = std::min(r.n_vis, s_hi - r.j_lo);
    }
    return r;
}

// One (token, head) row on the f32 cache. Same arithmetic as the scalar body,
// same masks, same softmax, same sink rule -- only the two inner loops change.
// `kbase`/`vbase` address the first VISIBLE key's head slice and `row_stride`
// is the distance between keys. That is the whole addressing contract, and it
// is what lets one body read either the cache in place (row_stride =
// pos_stride, 3072 B on this model) or a gathered tile (row_stride = hd)
// without a second copy of the arithmetic.
__attribute__((target("avx2,fma"))) void attn_row_f32_avx2(
    const f32 *qf, const f32 *kbase, const f32 *vbase, i64 row_stride, f32 *of,
    const AttnDesc &d, i64 t, i64 h, i64 kpos0, const AttnRange &r,
    f32 *scores) {
    const i64 hd = d.hd;
    const i64 n_vis = r.n_vis;
    if (n_vis <= 0) return;
    const i64 m0 = r.m0;
    const i64 m1 = r.m1;
    const f32 *qh = qf + t * (d.n_head * hd) + h * hd;

    // 64-dimension chunks with q resident in registers; anything left over takes
    // the generic dot. The mask is applied in the second pass so the hot loop
    // carries no branch at all.
    // 64-dimension chunks ACCUMULATE, so hd = 128 and 256 are covered by the
    // same code as hd = 64. The array is zeroed only when a chunk will run;
    // a row narrower than one chunk is a single dot and assigns.
    //
    // Both mistakes this shape of code invites were caught by the fixture in
    // tools/ta_attn_acc.cpp and not by reading it: an assigning chunk loses all
    // but the last 64 dimensions at hd = 128/256, and an assigning remainder
    // loses the first 64 at hd = 80/96. The fixture is the reason those two
    // arms exist.
    i64 off = 0;
    if (hd >= 64) {
        for (i64 j = 0; j < n_vis; j++) scores[j] = 0.0f;
        for (; off + 64 <= hd; off += 64)
            attn_score64(qh + off, kbase, row_stride, off, 0, n_vis, scores);
        if (off < hd) {
            for (i64 j = 0; j < n_vis; j++)
                scores[j] += attn_dot_avx2(qh + off, kbase + j * row_stride + off,
                                           hd - off);
        }
    } else {
        for (i64 j = 0; j < n_vis; j++)
            scores[j] = attn_dot_avx2(qh, kbase + j * row_stride, hd);
    }
    // ALiBi's slope is a property of the HEAD, so it is one multiply-add in
    // this per-key pass rather than a table lookup. `kpos0` is the absolute
    // position of j == 0, which on a causal row is the row's own position
    // minus (n_vis - 1).
    const f32 slope = d.alibi_max_bias > 0.0f
                          ? alibi_slope_impl(h, d.n_head, d.alibi_max_bias)
                          : 0.0f;
    for (i64 j = 0; j < n_vis; j++) {
        if (j >= m0 && j < m1) {
            scores[j] = -FLT_MAX;
            continue;
        }
        f32 sv = scores[j] * d.scale;
        if (d.softcap > 0.0f) sv = d.softcap * std::tanh(sv / d.softcap);
        if (slope != 0.0f) sv -= slope * static_cast<f32>(kpos0 - j);
        scores[j] = sv;
    }
    f32 mx = scores[0];
    for (i64 j = 1; j < n_vis; j++) mx = std::max(mx, scores[j]);
    f32 sum = 0.0f;
    for (i64 j = 0; j < n_vis; j++) {
        const f32 ex = std::exp(scores[j] - mx);
        scores[j] = ex;
        sum += ex;
    }
    const f32 sink_logit = d.sinks ? d.sinks[h] : 0.0f;
    if (sink_logit != 0.0f) sum += std::exp(sink_logit - mx);
    const f32 inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    f32 *oh = of + t * (d.n_head * hd) + h * hd;

    // Chunks of 64 output dimensions in eight live accumulators; the j order is
    // untouched, so this is the scalar sum, eight dimensions at a time.
    i64 i = 0;
    for (; i + 64 <= hd; i += 64) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        const f32 *vj = vbase + i;
        for (i64 j = 0; j < n_vis; j++, vj += row_stride) {
            const __m256 p = _mm256_set1_ps(scores[j] * inv);
            a0 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 0), a0);
            a1 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 8), a1);
            a2 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 16), a2);
            a3 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 24), a3);
            a4 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 32), a4);
            a5 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 40), a5);
            a6 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 48), a6);
            a7 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 56), a7);
        }
        _mm256_storeu_ps(oh + i + 0, a0);
        _mm256_storeu_ps(oh + i + 8, a1);
        _mm256_storeu_ps(oh + i + 16, a2);
        _mm256_storeu_ps(oh + i + 24, a3);
        _mm256_storeu_ps(oh + i + 32, a4);
        _mm256_storeu_ps(oh + i + 40, a5);
        _mm256_storeu_ps(oh + i + 48, a6);
        _mm256_storeu_ps(oh + i + 56, a7);
    }
    // 16 dims in two accumulators, 8 in one. The second accumulator is over
    // DIMENSIONS, never over j: two j-chains added together at the end is a
    // reordered sum, and this half of the kernel is supposed to be exact.
    for (; i + 16 <= hd; i += 16) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        const f32 *vj = vbase + i;
        for (i64 j = 0; j < n_vis; j++, vj += row_stride) {
            const __m256 p = _mm256_set1_ps(scores[j] * inv);
            a0 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj), a0);
            a1 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vj + 8), a1);
        }
        _mm256_storeu_ps(oh + i, a0);
        _mm256_storeu_ps(oh + i + 8, a1);
    }
    for (; i + 8 <= hd; i += 8) {
        __m256 a0 = _mm256_setzero_ps();
        const f32 *vj = vbase + i;
        for (i64 j = 0; j < n_vis; j++, vj += row_stride)
            a0 = _mm256_fmadd_ps(_mm256_set1_ps(scores[j] * inv),
                                 _mm256_loadu_ps(vj), a0);
        _mm256_storeu_ps(oh + i, a0);
    }
    for (; i < hd; i++) {
        const f32 *vj = vbase + i;
        f32 acc = 0.0f;
        for (i64 j = 0; j < n_vis; j++, vj += row_stride)
            acc += scores[j] * inv * vj[0];
        oh[i] = acc;
    }
}
#endif // KRK_ATTN_AVX2

} // namespace

class CpuBackend final : public Backend {
public:
    // `npu` is the virtual-NPU offload handle when this instance is the NPU
    // backend, and null for the plain reference backend. Only the GEMM path
    // consults it -- see npu_offload.hpp for why that is the whole of the
    // hand-over and not the first slice of a larger one.
    explicit CpuBackend(const DeviceCaps &c, npu::Offload *npu = nullptr)
        : caps_(c), npu_(npu) {}
    ~CpuBackend() override {
        if (npu_) npu::destroy(npu_);
    }

    // Submits everything queued for the device and waits for it, so a host
    // kernel that is about to read an activation buffer sees what the device
    // wrote. Called at the top of every op this file implements itself, and
    // nowhere else: the six ops that CAN go to the device (gemm, rmsnorm,
    // qk_norm, silu_mul, add_inplace, axpy, mul_act) reach it only in the
    // fallback branches -- queueing there and flushing here is what turns ten
    // small ops per layer into zero extra round trips. Cheap when the queue is
    // empty, which is every call on a plain CPU build.
    void npu_flush_() {
        if (npu_) npu::flush(npu_);
    }

    BackendKind kind() const override {
        return npu_ ? BackendKind::NPU : BackendKind::CPU;
    }
    const DeviceCaps &caps() const override { return caps_; }
    // The device computes in place on the memory it is handed, so there is no
    // separate tier for the engine's VRAM ledger, expert cache or weight
    // streaming to plan against.
    bool has_device_memory() const override { return false; }
    DType act_type() const override { return DType::F32; }
    size_t act_size() const override { return 4; }

    static size_t pool_cap() {
        static const size_t cap = [] {
            const char *e = std::getenv("KRK_CPU_POOL_MB");
            if (e && e[0] != '\0') {
                const long mb = std::atol(e);
                return mb <= 0 ? size_t(0) : static_cast<size_t>(mb) << 20;
            }
            return static_cast<size_t>(512) << 20;
        }();
        return cap;
    }
    mutable std::mutex pool_mu_;
    std::unordered_map<size_t, std::vector<void *>> pool_free_;
    size_t pool_free_bytes_ = 0, pool_reuse_ = 0, pool_alloc_ = 0, pool_free_n_ = 0,
           pool_drop_ = 0;

    void *alloc(size_t bytes) override { return host_alloc(bytes); }
    void release(void *p) override { npu_flush_(); host_free(p); }

    // ---- expert slot pool ------------------------------------------------
    //
    // The base class recycles nothing on this backend: alloc_pooled() is
    // alloc() and release_pooled() is free(), so every cold expert miss asks
    // the kernel for ~1.6 MB that has never been touched, the copy below
    // faults those pages in one at a time, and the slot then stays live until
    // it is evicted. Nothing is ever reused, so the process keeps first-touching
    // new memory for the whole run.
    //
    // That is the entire `expert-path read` line on the 32-expert q8_0 fixture.
    // Measured on the same bytes, same file, warm page cache:
    //
    //   destination reused         5.25 GB/s
    //   malloc+free per chunk      9.78 GB/s
    //   every slot live at once    2.05 GB/s   <- what this backend did
    //
    // against the 1.73 GB/s the engine actually reported, with `read 97.6 ms`
    // on the prefill phase and `0.0 ms` on decode. A recycled slot keeps its
    // pages mapped, so the copy writes into already-present memory.
    //
    // Bounded: slots beyond the budget are genuinely freed, so a model whose
    // churn exceeds the cap degrades to the old behaviour instead of growing
    // without limit. KRK_CPU_POOL_MB overrides, 0 disables the pool.
    void *alloc_pooled(size_t bytes) override {
        {
            std::lock_guard<std::mutex> g(pool_mu_);
            const size_t cap = pool_cap();
            auto it = pool_free_.find(bytes);
            if (it != pool_free_.end() && !it->second.empty()) {
                void *p = it->second.back();
                it->second.pop_back();
                pool_free_bytes_ -= bytes;
                ++pool_reuse_;
                return p;
            }
            (void)cap;
        }
        void *p = host_alloc(bytes);
        std::lock_guard<std::mutex> g(pool_mu_);
        ++pool_alloc_;
        return p;
    }

    void release_pooled(void *p, size_t bytes) override {
        if (!p || bytes == 0) return;
        {
            std::lock_guard<std::mutex> g(pool_mu_);
            if (pool_cap() > 0 && pool_free_bytes_ + bytes <= pool_cap()) {
                pool_free_[bytes].push_back(p);
                pool_free_bytes_ += bytes;
                ++pool_free_n_;
                return;
            }
            ++pool_drop_;
        }
        npu_flush_();
        host_free(p);
    }

    ExpertPoolStats expert_pool_stats() const override {
        std::lock_guard<std::mutex> g(pool_mu_);
        ExpertPoolStats st;
        st.budget = pool_cap();
        st.committed = pool_free_bytes_;
        st.live = pool_free_bytes_;
        st.reusable = pool_free_bytes_;
        st.allocations = pool_alloc_;
        st.reuses = pool_reuse_;
        return st;
    }
    void upload(void *dst, const void *src, size_t bytes, size_t off) override {
        npu_flush_();
        std::memcpy(static_cast<char *>(dst) + off, src, bytes);
    }
    void download(void *dst, const void *src, size_t bytes, size_t off) override {
        npu_flush_();
        std::memcpy(dst, static_cast<const char *>(src) + off, bytes);
    }
    void fill0(void *dst, size_t bytes) override { npu_flush_(); std::memset(dst, 0, bytes); }
    void sync() override { npu_flush_(); }
    void download_f32(f32 *dst, const void *src, i64 n) override {
        npu_flush_();
        std::memcpy(dst, src, static_cast<size_t>(n) * 4);
    }
    std::string backend_stats() const override {
        if (!npu_) return std::string();
        // The ops the device has no opcode for are named once, here, so the run
        // summary says what "whole graph" actually covered instead of implying
        // it covered everything.
        // attention is named here only when it is NOT offered to the device.
        // With KRK_NPU_ATTN off (the default) it runs entirely on the reference,
        // so the summary must say so; with it on, npu::stats reports how many
        // calls the device took and names whatever blocked the rest, and
        // claiming it was a host op would be wrong.
        std::string s = npu::stats(npu_) + "; host: rope, ";
        if (!npu::attention_offered(npu_)) s += "attention, ";
        s += "kv cache, embedding, MoE routing, top-k, ssm/MLA/HC blocks "
             "(no VNPU opcode)";
        return s;
    }

    // The device is the one with a one-time weight conversion, so the seam is
    // forwarded rather than implemented here: with W8A8 off there is nothing to
    // move, and the prepare pass is skipped entirely.
    bool wants_prepare() const override {
        return npu_ && npu::wants_prepare(npu_);
    }
    void set_prepare(bool on) override {
        if (npu_) npu::set_preparing(npu_, on);
    }
    bool preparing() const override { return npu_ && npu::preparing(npu_); }

    std::string describe() const override {
        // Report placement with the part: on a hybrid host this line is the
        // difference between "why is it slow" and "it is on the E cores".
        const std::string topo = Pool::get().topology().summary();
        if (npu_)
            return format("CPU (scalar reference) + %s, %s, %s",
                          npu::describe(npu_).c_str(), caps_.gfx.c_str(), topo.c_str());
        return format("CPU (scalar reference), %s, %s", caps_.gfx.c_str(), topo.c_str());
    }
    // No card here, so the memory policy has nothing to measure and would take
    // its "cannot say" path for every run. `KRK_VRAM_TOTAL_MB` says "behave as if
    // the card were N MiB", which is what lets the expert-budget refusal be
    // driven end to end from the test suite without opening a device: the suite
    // builds an MoE fixture, asks for a budget larger than the simulated card,
    // and requires Engine::init to RETURN false. It is consulted only while this
    // backend has no capacity of its own to report, and caps() deliberately
    // stays 0, so the branches that key off "cannot report capacity" are
    // untouched -- a real backend's number always wins over this, and a CPU run
    // that does not set the variable reports 0 exactly as it did before.
    size_t device_total_bytes() const override {
        return krk::simulated_total_bytes();
    }
    size_t device_free_bytes() const override {
        return krk::simulated_total_bytes();
    }
    size_t host_available_bytes() const override { return krk::host_available_bytes(); }
    size_t host_total_bytes() const override { return krk::host_total_bytes(); }

    void embed(void *out, const void *tok_embd, DType tt, i64 n_vocab, i64 n_embd,
               const i32 *tokens, i64 n_tok) override {
        KRK_KTIME("embed");
        npu_flush_();
        const size_t row_bytes = dtype_row_bytes(tt, n_embd);
        f32 *o = static_cast<f32 *>(out);
        // Token rows are independent; each dequantizes into its
        // own output slice.
        //
        // `n_vocab` was `(void)`-ed here before this line existed, and the id
        // went straight into the address arithmetic: an id at or past n_vocab
        // read the row AFTER the table (on a real file that is the next tensor's
        // bytes, or in ASan terms a heap-buffer-overflow 0 bytes past the
        // region). Engine::init now refuses a file whose tokenizer can produce
        // such an id, so this is the belt to that brace -- and it is the only
        // check a caller that builds ids itself ever hits. An out-of-range id
        // gets a zero row: a defined, visible value, not another tensor's data.
        par_for(n_tok, 1, 2.0 * static_cast<f64>(n_tok) * n_embd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        f32 *dst = o + t * n_embd;
                        const i32 id = tokens[t];
                        if (id < 0 || static_cast<i64>(id) >= n_vocab) {
                            std::memset(dst, 0,
                                        static_cast<size_t>(n_embd) * sizeof(f32));
                            continue;
                        }
                        dequant_row(tt, static_cast<const u8 *>(tok_embd) +
                                           static_cast<size_t>(id) * row_bytes,
                                    dst, n_embd);
                    }
                });
    }

    void rmsnorm(void *out, const void *x, const f32 *w, i64 rows, i64 n,
                 f32 eps, const i32 *row_exp = nullptr) override {
        KRK_KTIME("rmsnorm");
        // `row_exp` is the residual stream's per-row storage exponent, which
        // only a backend whose activations cannot hold the range ever sets; the
        // device has no way to apply it, so a row that carries one takes the
        // reference body. The engine hands this array over unconditionally,
        // though, and on this backend every entry in it is zero (see
        // rmsnorm_rows: f32 activations already hold the range), so an array of
        // zeros is the same request as no array and goes to the device.
        if (npu_) {
            // The scan is O(rows) against O(rows*n) of work, and it only runs
            // when a device is attached at all.
            bool scaling = row_exp != nullptr;
            for (i64 r = 0; scaling && r < rows; r++) scaling = row_exp[r] != 0;
            if (!scaling && w && npu::rmsnorm(npu_, out, x, w, rows, n, eps)) return;
        }
        npu_flush_();
        rmsnorm_rows(static_cast<f32 *>(out), static_cast<const f32 *>(x), w, rows,
                     n, eps, row_exp);
    }

    bool group_norm_supported() const override { return true; }
    // k2-horizon's grouped reduction with ONE weight over the whole row. See
    // Backend::rmsnorm_bands: only the mean square is per group.
    void rmsnorm_bands(void *o, const void *x, const f32 *w, i64 n_tok,
                       i64 n_group, i64 ne, f32 eps) override {
        npu_flush_();
        const f32 *s = static_cast<const f32 *>(x);
        f32 *d = static_cast<f32 *>(o);
        if (n_group <= 1) {           // the ordinary norm, same arithmetic
            rmsnorm(o, x, w, n_tok, ne, eps);
            return;
        }
        const i64 width = ne / n_group;
        for (i64 t = 0; t < n_tok; t++) {
            const f32 *sr = s + t * ne;
            f32 *dr = d + t * ne;
            for (i64 g = 0; g < n_group; g++) {
                const i64 off = g * width;
                f32 ss = 0.0f;
                for (i64 i = 0; i < width; i++) ss += sr[off + i] * sr[off + i];
                const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(width) + eps);
                for (i64 i = 0; i < width; i++)
                    dr[off + i] = w ? sr[off + i] * inv * w[off + i]
                                    : sr[off + i] * inv;
            }
        }
    }
    void rmsnorm_grouped(void *o, const void *x, const f32 *w, i64 n_tok,
                         i64 n_group, i64 width, i64 stride,
                         f32 eps) override {
        npu_flush_();
        const f32 *s = static_cast<const f32 *>(x);
        f32 *d = static_cast<f32 *>(o);
        // Salient difference from rmsnorm_rows above: the reduction and the
        // weight both belong to the (token, group) pair, so the weight index is
        // the row's group, not a row-invariant w[i]. Safe in place: the
        // reduction completes before the first write to that group's channels,
        // and no other row touches them.
        const i64 rows = n_tok * n_group;
        par_for(rows, 1, 4.0 * static_cast<f64>(rows) * width,
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++) {
                        const i64 t = r / n_group;
                        const i64 g = r % n_group;
                        const f32 *sr = s + t * stride + g * width;
                        f32 *dr = d + t * stride + g * width;
                        const f32 *wr = w + g * width;
                        f32 ss = 0;
                        for (i64 i = 0; i < width; i++) ss += sr[i] * sr[i];
                        const f32 inv = 1.0f /
                            std::sqrt(ss / static_cast<f32>(width) + eps);
                        for (i64 i = 0; i < width; i++)
                            dr[i] = sr[i] * inv * wr[i];
                    }
                });
    }
    // The scalar twin of the device radix select (kernels/topk.hpp). Same
    // contract and same order -- transform first, then rank by (value, id)
    // descending -- so the two backends can be compared token for token. This
    // one has no download to save, so it exists for comparability, not speed.
    void logits_topk(const void *logits, i64 n, i32 k, const void *pen_mask,
                     f32 rep_pen, f32 softcap, f32 temp, i32 *ids, f32 *vals,
                     f32 *mass_out, void *scratch, i32 cand_cap,
                     i64 *count_out) override {
        npu_flush_();
        (void)scratch;
        (void)cand_cap;
        const f32 *l = static_cast<const f32 *>(logits);
        const u32 *pm = static_cast<const u32 *>(pen_mask);
        auto key = [&](i32 i) {
            f32 v = l[i];
            if (softcap > 0.0f)
                v = v > softcap ? softcap : (v < -softcap ? -softcap : v);
            if (pm && rep_pen > 0.0f && rep_pen != 1.0f &&
                ((pm[static_cast<u32>(i) >> 5] >> (static_cast<u32>(i) & 31)) & 1u))
                v = v > 0.0f ? v / rep_pen : v * rep_pen;
            return v;
        };
        const i64 keep = std::min<i64>(n, k);
        std::vector<i32> idx(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++) idx[static_cast<size_t>(i)] = static_cast<i32>(i);
        // This O(n log k) sort is exactly the host work the device kernels
        // exist to avoid; here it is what makes the scalar backend a usable
        // oracle for the fast path.
        std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                          [&](i32 a, i32 b) {
                              const f32 ka = key(a), kb = key(b);
                              return ka != kb ? ka > kb : a < b;
                          });
        const f32 maxv = key(idx[0]);
        for (i32 r = 0; r < k; r++) {
            if (r < keep) {
                ids[r] = idx[static_cast<size_t>(r)];
                vals[r] = key(ids[r]);
            } else {
                // Fewer than k elements exist. The device kernel writes the
                // same marker, so the two backends stay interchangeable.
                ids[r] = -1;
                vals[r] = -FLT_MAX;
            }
        }
        if (mass_out) {
            const f32 inv = temp > 0.0f ? 1.0f / temp : 1.0f;
            f32 m = 0.0f;
            for (i64 i = 0; i < n; i++)
                m += std::exp((key(static_cast<i32>(i)) - maxv) * inv);
            *mass_out = m;
        }
        if (count_out) *count_out = keep;
    }


    void gemm(void *out, const void *x, const void *w, DType wt, i64 n_out,
              i64 n_in, i64 rows) override {
        // The virtual NPU computes this dot product block on its own SIMD
        // kernels. It returns false -- having written nothing -- for a shape or
        // weight type it cannot take, and the reference loop below then runs
        // only after the queue has landed, because `x` is very often a buffer
        // the device itself just wrote (the RMSNorm in front of this GEMM).
        if (npu_ &&
            npu::gemm(npu_, static_cast<f32 *>(out), static_cast<const f32 *>(x),
                      w, wt, n_out, n_in, rows))
            return;
        npu_flush_();
        // One row in the table per distinct kernel, because "gemm" as a single
        // row hides the thing that matters: a decode step and a batched
        // prefill run different kernels and belong on different lines.
        // MACs, not FLOPs: one per multiply-add, which is the unit %peak is
        // measured in. Every GEMM row is compared against the ceiling for the
        // instruction class it actually executes -- int8 dot where the int8
        // fast path below takes it, f32 FMA otherwise.
        KRK_KTIME_MAC(rows == 1 ? dtype_tag_r1(wt, n_in) : dtype_tag_rn(wt, n_in),
                      static_cast<double>(rows) * static_cast<double>(n_out) *
                          static_cast<double>(n_in),
                      isa_peak_gmacs((wt == DType::Q4_K || wt == DType::Q6_K ||
                                      wt == DType::Q8_0) &&
                                     n_in > 0 && (n_in % 256) == 0 &&
                                     vec_dot_int8()));
        const size_t row_bytes = dtype_row_bytes(wt, n_in);
        const f32 *xf = static_cast<const f32 *>(x);
        f32 *of = static_cast<f32 *>(out);
        // Q4_K fast path. The activation is quantised to int8 ONCE for the
        // whole call and every output element is then an integer dot against a
        // weight row that never leaves its stored blocks. Doing it here rather
        // than inside vec_dot() is the whole point: vec_dot() is called once
        // per output element, so a per-call quantisation there would undo the
        // win it is trying to buy.
        // Ternary joins here on the same terms: TQ1_0 (BitNet) and PTQ1_0
        // (Ternary-Bonsai) hold {-1,0,+1} behind one f16 scale, so their dot is
        // an int8 dot with no bias term and no sidecar -- the cheapest of any
        // format in this list per stored byte. The gate is the same 256-value
        // alignment because that is what the activation quantiser produces.
        // Q2_K and Q3_K join here too: both are Q6_K's 256-value block with a
        // different unpack, so they take the same dispatch and the same
        // eight-row sharing.
        if ((wt == DType::Q4_K || wt == DType::Q6_K || wt == DType::Q8_0 ||
             wt == DType::TQ1_0 || wt == DType::PTQ1_0 ||
             wt == DType::TQ2_0 || wt == DType::PQ2_0 || wt == DType::Q2_0 ||
             wt == DType::Q2_0_64 || wt == DType::I2_S ||
             wt == DType::Q2_K || wt == DType::Q3_K) &&
            n_in > 0 && (n_in % 256) == 0 && rows > 0 && vec_dot_int8()) {
            const bool q6 = wt == DType::Q6_K;
            const bool q80 = wt == DType::Q8_0;
            const bool tq1 = wt == DType::TQ1_0;
            const bool ptq1 = wt == DType::PTQ1_0;
            // The 2-bit CODE families. They are NOT trits: the ladder is
            // {-1,0,+1,+2}, so they get their own kernel (which stays exact for
            // q = 3) rather than the trit one. Their d is one f16 per block and
            // their codes carry no dmin, so like the trits they need no sum(x)
            // sidecar.
            const bool tq2 = wt == DType::TQ2_0;
            // I2_S rides the same code kernel: 128-value planar blocks, no
            // per-block delta (its one f32 is per TENSOR, folded in as gs), so
            // like the trits it needs no sum(x) sidecar.
            const bool i2s = wt == DType::I2_S;
            const bool pq2 = wt == DType::PQ2_0;
            const bool q20 = wt == DType::Q2_0;
            const bool q20_64 = wt == DType::Q2_0_64;
            const bool q3k = wt == DType::Q3_K;
            const bool q2k = wt == DType::Q2_K;
            // Only Q4_K reads the per-32 `xs` sidecar. Q8_0 and the ternary
            // formats need no sum of the activation at all, and Q2_K/Q3_K take
            // their sums per 16-value sub-block inside the kernel -- where the
            // bytes are already in registers -- so none of them allocates it.
            const bool none_aux = q80 || tq1 || ptq1 || tq2 || pq2 || q20 ||
                                  q20_64 || i2s || q2k || q3k;
            const i64 nb = n_in / 32;
            // Owned by the CALLING thread and captured by value below. These
            // were thread_local, which looks like a free optimisation and is a
            // null dereference: par_for runs the body on pool workers, and a
            // worker's own thread_local vector has never been resized, so
            // data() is nullptr there. ASan caught the read at address 0.
            std::vector<i8> xq(static_cast<size_t>(rows * n_in));
            std::vector<f32> xd(static_cast<size_t>(rows * nb));
            // The two auxiliary arrays are allocated ONLY for the format that
            // reads them. They used to be value-initialised unconditionally,
            // so every Q8_0 call -- the hot format, and the one that needs
            // neither -- paid to zero rows*nb*4 plus rows*(n_in/4)*4 bytes:
            // 442 KiB per call at 512x768, times one call per GEMM per layer.
            // An empty vector's data() is a valid pointer for a size-0
            // span, and the kernels are only reached from the branch that
            // filled the matching array.
            //
            // sum(xi) per sub-block, for formats whose dmin term needs it. The
            // Q4_K kernel reads it; Q6_K does not. Computing it here means it is
            // done once per GEMM call instead of once per weight row.
            const i64 nq4 = n_in / 4;
            std::vector<i32> xs(q6 || none_aux ? 0 : static_cast<size_t>(rows * nb), 0);
            // Q6_K's scale changes every 16 values and its bias is 32*sum(16),
            // so it needs its own array; Q4_K's is per 32. Only one is filled.
            std::vector<i32> xs4(q6 ? static_cast<size_t>(rows * nq4) : 0, 0);
            for (i64 r = 0; r < rows; r++) {
                if (none_aux)
                    // No sx[]: Q8_0's magnitude split is exact and the ternary
                    // formats have no bias term at all, so neither needs a
                    // per-block sum of the activation.
                    quantize_q8_row(xf + r * n_in, n_in,
                                    xq.data() + r * n_in, xd.data() + r * nb);
                else if (q6)
                    quantize_q8_row_sq4(xf + r * n_in, n_in,
                                        xq.data() + r * n_in,
                                        xd.data() + r * nb,
                                        xs4.data() + r * nq4);
                else
                    quantize_q8_row_sx(xf + r * n_in, n_in,
                                       xq.data() + r * n_in, xd.data() + r * nb,
                                       xs.data() + r * nb);
            }
            const i8 *xqp = xq.data();
            const f32 *xdp = xd.data();
            const i32 *xsp = xs.data();
            const i32 *xs4p = xs4.data();
            const u8 *wb = static_cast<const u8 *>(w);
            // Which loop is outermost decides the memory traffic, and the
            // traffic was the whole cost.
            //
            // The original order was `for each output column { for each row
            // tile }`. That is the obvious shape -- one weight row, eight
            // activations, eight results -- and it streams the ENTIRE
            // activation matrix once per output column. On the 512x2048x768 FF
            // shape that is 512*768 bytes re-read 2048 times: 805 MB per GEMM
            // call, against 1.67 MB of unique weights. Measured against the
            // 301 GMAC/s int8 ceiling the kernel sat at 34.5, and no amount of
            // widening moved it, because widening changes instructions and the
            // instructions were never the limit.
            //
            // Blocking the columns fixes it the standard way: an activation
            // tile is 8*768 bytes, small enough to stay in L1 across a whole
            // block of output columns, and each weight row is then read once
            // per block instead of once per row tile. Activation traffic drops
            // by the block factor (805 MB -> 6.3 MB at NC=128) and weight
            // traffic drops to one pass.
            //
            // KRK_CPU_GEMM_NC=0 restores the original order, which is how the
            // two are compared.
            const i64 nc = [&]() -> i64 {
                const char *e = std::getenv("KRK_CPU_GEMM_NC");
                if (e && e[0] != '\0') {
                    const i64 v = std::atoll(e);
                    if (v <= 0) return 0;                    // old order
                    return std::min<i64>(v, n_out);
                }
                if (n_out < 2) return 0;
                // Aim a block's weights at a little under 128 KiB, which keeps
                // the weight block in L2 while the activation tile stays in L1.
                const i64 per = std::max<i64>(1, static_cast<i64>(row_bytes));
                return std::max<i64>(16, std::min<i64>(512, 131072 / per));
            }();
            const f64 flops = 2.0 * static_cast<f64>(rows) *
                              static_cast<f64>(n_out) * static_cast<f64>(n_in);
            const i64 rows8 = rows - rows % 8;
            auto dot8 = [&](const u8 *wr, i64 r, i64 o) {
                if (q3k)
                    dot_q3_K_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                   n_in, nb, of + r * n_out + o, n_out);
                else if (q2k)
                    dot_q2_K_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                   n_in, nb, of + r * n_out + o, n_out);
                else if (tq1)
                    dot_tq1_0_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                    n_in, nb, of + r * n_out + o, n_out);
                else if (ptq1)
                    dot_ptq1_0_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                     n_in, nb, of + r * n_out + o, n_out);
                else if (tq2)
                    dot_tq2_0_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                    n_in, nb, of + r * n_out + o, n_out);
                else if (i2s)
                    dot_i2_s_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in, n_in,
                                   nb, of + r * n_out + o, n_out);
                else if (pq2)
                    dot_pq2_0_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                    n_in, nb, of + r * n_out + o, n_out);
                else if (q20)
                    dot_q2_0_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                   n_in, nb, of + r * n_out + o, n_out);
                else if (q20_64)
                    dot_q2_0_64_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in,
                                      n_in, nb, of + r * n_out + o, n_out);
                else if (q80)
                    dot_q8_0_q8_x8(wr, xqp + r * n_in, xdp + r * nb, n_in, n_in,
                                   nb, of + r * n_out + o, n_out);
                else if (q6)
                    dot_q6_K_q8_sq4_x8(wr, xqp + r * n_in, xdp + r * nb,
                                       xs4p + r * nq4, n_in, n_in, nb, nq4,
                                       of + r * n_out + o, n_out);
                else
                    dot_q4_K_q8_sx_x8(wr, xqp + r * n_in, xdp + r * nb,
                                      xsp + r * nb, n_in, n_in, nb, nb,
                                      of + r * n_out + o, n_out);
            };
            auto dot1 = [&](const u8 *wr, i64 r, i64 o) {
                if (q3k)
                    return dot_q3_K_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (q2k)
                    return dot_q2_K_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (tq1)
                    return dot_tq1_0_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (ptq1)
                    return dot_ptq1_0_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (tq2)
                    return dot_tq2_0_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (i2s)
                    return dot_i2_s_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (pq2)
                    return dot_pq2_0_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (q20)
                    return dot_q2_0_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (q20_64)
                    return dot_q2_0_64_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (q80)
                    return dot_q8_0_q8(wr, xqp + r * n_in, xdp + r * nb, n_in);
                if (q6)
                    return dot_q6_K_q8_sq4(wr, xqp + r * n_in, xdp + r * nb,
                                           xs4p + r * nq4, n_in);
                return dot_q4_K_q8_sx(wr, xqp + r * n_in, xdp + r * nb,
                                      xsp + r * nb, n_in);
            };
            // q8_0 with VNNI keeps every output row's f32 block scales in a side
            // array, built once per COLUMN BLOCK and reused by every activation
            // row tile. Keyed by the output row -- see build_q8_0_wscales: an
            // earlier version keyed it by the row TILE and was wrong by up to
            // 83% on matrices whose rows have different scales.
            // KRK_CPU_Q80_WSC=0 turns the scale array off, which is both the
            // control for benchmarks and the escape hatch if a future host
            // disagrees with it. Same arithmetic either way -- dot_q8_0_q8_wsc8
            // falls back to dot_q8_0_q8_x8 internally.
            static const bool wsc_enabled = [] {
                const char *e = std::getenv("KRK_CPU_Q80_WSC");
                return !(e != nullptr && e[0] == '0');
            }();
            const bool use_wsc = wsc_enabled && q80 && n_in > 0 &&
                                 (n_in / 32) >= 2 && ((n_in / 32) & 1) == 0 &&
                                 q8_0_wsc_supported();
            const i64 npair = (n_in / 32) / 2;
            // KRK_CPU_XSC_HOIST=0 rebuilds the activation-scale array inside the
            // column loop, which is the shape this had before the hoist. Only
            // there to keep the comparison measurable.
            static const bool xsc_hoist = [] {
                const char *e = std::getenv("KRK_CPU_XSC_HOIST");
                return !(e != nullptr && e[0] == '0');
            }();
            std::vector<f32> xsc_all;
        // Decode, q8_0, VNNI: four output rows per call. Each is an
        // independent weight stream, so a thread keeps four in flight instead
        // of one, which is what this kernel is short of (see the comment on
        // dot_q8_0_q8_x4_gemv). Falls through to the per-element path for every
        // other tier, so nothing else changes.
        if (rows == 1 && q80 && n_in > 0 && (n_in % 64) == 0 && n_out >= 4 &&
            ((n_in / 32) & 1) == 0) {
            const i32 kw = q8_0_gemv_width();
            const i64 n4 = n_out - n_out % kw;
            const i64 g4 = std::max<i64>(1, std::min<i64>(1024, (n4 / kw) / 16));
            par_for(n4 / kw, g4, 2.0 * static_cast<f64>(n4) * n_in,
                    [&](i64 b, i64 e) {
                        for (i64 g = b; g < e; g++) {
                            const i64 o = g * kw;
                            dot_q8_0_q8_r1xN(
                                wb + static_cast<size_t>(o) * row_bytes,
                                row_bytes, xqp, xdp, n_in, nb, of + o, 1, kw);
                        }
                    });
            for (i64 o = n4; o < n_out; o++)
                of[o] = dot_q8_0_q8(wb + static_cast<size_t>(o) * row_bytes,
                                    xqp, xdp, n_in);
            return;
        }
            if (nc > 0) {
                // One row tile at a time, every column in the block against it.
                // The activation pointers for the tile are hoisted out of the
                // column loop so the compiler keeps them in registers.
                const i64 nblk = (n_out + nc - 1) / nc;
                par_for(nblk, 1, flops, [&](i64 cb, i64 ce) {
                    // One scale array for the whole column block: every output
                    // row's f32 block scales, converted and expanded to the
                    // lanes that consume them, once, then reused by every
                    // activation row tile below. This is the hoist, and it is
                    // only correct because the array is indexed by the OUTPUT
                    // row -- see build_q8_0_wscales. Built this way the inner
                    // loop has no scale build at all: it is load, vpdpbusd,
                    // vcvtdq2ps, vmulps, vfmadd.
                    std::vector<f32> wsc;
                    // Activation scales for EVERY row of the call, built once.
                    // They do not depend on the column block, so building them
                    // inside the column loop below would repeat the work once
                    // per block -- n_out/nc times -- for a result that is
                    // already in hand. Sized rows8 * npair * 16 floats, which is
                    // one byte per activation element.
                    if (use_wsc && xsc_hoist) {
                        xsc_all.resize(static_cast<size_t>(rows8 * npair * 16));
                        build_q8_0_xscales(xdp, nb, rows8, n_in, xsc_all.data());
                    }
                    for (i64 c = cb; c < ce; c++) {
                        const i64 o0 = c * nc;
                        const i64 o1 = std::min<i64>(n_out, o0 + nc);
                        if (use_wsc) {
                            wsc.resize(static_cast<size_t>((o1 - o0) * npair * 16));
                            build_q8_0_wscales(
                                wb + static_cast<size_t>(o0) * row_bytes,
                                row_bytes, o1 - o0, n_in, wsc.data());
                        }
                        for (i64 r = 0; r < rows8; r += 8) {
                            if (use_wsc && !xsc_hoist) {
                                xsc_all.resize(static_cast<size_t>(8 * npair * 16));
                                build_q8_0_xscales(xdp + r * nb, nb, 8, n_in,
                                                   xsc_all.data());
                            }
                            for (i64 o = o0; o < o1; o++) {
                                const u8 *wr =
                                    wb + static_cast<size_t>(o) * row_bytes;
                                if (use_wsc)
                                    dot_q8_0_q8_wsc8(
                                        wr, xqp + r * n_in, xdp + r * nb, n_in,
                                        n_in, nb, of + r * n_out + o, n_out,
                                        wsc.data() + (o - o0) * npair * 16,
                                        xsc_all.data() +
                                            (xsc_hoist ? r * npair * 16 : 0));
                                else
                                    dot8(wr, r, o);
                            }
                        }
                        for (i64 r = rows8; r < rows; r++)
                            for (i64 o = o0; o < o1; o++)
                                of[r * n_out + o] =
                                    dot1(wb + static_cast<size_t>(o) * row_bytes,
                                         r, o);
                    }
                });
                return;
            }
            const i64 grain = std::max<i64>(8, std::min<i64>(256, n_out / 8));
            par_for(n_out, grain, flops, [&](i64 b, i64 e) {
                for (i64 o = b; o < e; o++) {
                    const u8 *wr = wb + static_cast<size_t>(o) * row_bytes;
                    i64 r = 0;
                    // Eight rows share one decode of the weight block.
                    for (; r + 8 <= rows; r += 8) dot8(wr, r, o);
                    for (; r < rows; r++) of[r * n_out + o] = dot1(wr, r, o);
                }
            });
            return;
        }

        // Every output element is an independent dot product, so the
        // flat (row, column) index space splits without sharing any
        // reduction — results are bit-identical to the serial loop.
        // This is the decode hot path: the head GEMV (n_vocab x n_embd)
        // is ~91% of decode FLOPs and parallelizes across all cores.
        const i64 total = rows * n_out;
        const i64 grain = std::max<i64>(32, std::min<i64>(2048, n_out / 32));

        // Batched prefill. The loop below splits on (row, column) and calls
        // vec_dot() per element, and vec_dot() decodes the weight as it goes --
        // so a batch of `rows` decoded every weight row `rows` times and
        // prefill ran at GEMV speed. Measured on the 2 GiB q8_0 MoE at --cpu
        // with KRK_CPU_DOT_AVX2=1: prefill cost per token was FLAT at 34.1,
        // 29.1, 27.8 and 28.3 ms across prompts of 58, 149, 331 and 695 tokens,
        // where a batched GEMM must fall steeply. vec_dot_tiled() decodes each
        // weight block once and reuses it across a tile of rows, and returns
        // bit-identical values, so this changes speed and nothing else.
        //
        // Kept to rows > 1: at rows == 1 (every decode step) there is nothing
        // to amortize, and the fused kernel below is the better shape.
        if (rows > 1 && n_in > 0) {
            const i64 tile = std::max<i64>(1, std::min<i64>(64, n_out));
            par_for(n_out, tile,
                    2.0 * static_cast<f64>(total) * static_cast<f64>(n_in),
                    [&](i64 b, i64 e) {
                        vec_dot_tiled(wt, w, xf, n_in, b, e, rows, of, n_out);
                    });
            return;
        }
        par_for(total, grain, 2.0 * static_cast<f64>(total) * n_in,
                [&](i64 b, i64 e) {
                    i64 r = b / n_out;
                    i64 o = b - r * n_out;
                    for (i64 f = b; f < e; f++) {
                        const f32 *xr = xf + r * n_in;
                        const u8 *wrow = static_cast<const u8 *>(w) +
                                         static_cast<size_t>(o) * row_bytes;
                        // The decode shape. vec_dot_wide1 is vec_dot unless
                        // the wide tier is on, so the default path is byte for
                        // byte what it was.
                        of[f] = vec_dot_wide1(wt, wrow, xr, n_in);
                        if (++o >= n_out) {
                            o = 0;
                            ++r;
                        }
                    }
                });
    }

    void rope(void *q, void *k, i64 n_head, i64 n_kv, i64 hd, i64 n_tok, i64 pos0,
              const f32 *inv_freq, f32 scale, f32 frac, bool neox) override {
        KRK_KTIME("rope");
        npu_flush_();
        rope_apply(static_cast<f32 *>(q), n_head, hd, n_tok, pos0, inv_freq, scale, frac, neox);
        rope_apply(static_cast<f32 *>(k), n_kv, hd, n_tok, pos0, inv_freq, scale, frac, neox);
    }

    // The reference implementation of the batched form: the same rope_apply,
    // once per row, with that row's own position. The arithmetic is literally
    // the single-sequence code, which is what makes it an oracle for a device
    // kernel that batches the rotations.
    bool rope_rows(void *q, void *k, i64 n_head, i64 n_kv, i64 hd, i64 n_tok,
                   const i64 *row_pos, const f32 *inv_freq, f32 scale, f32 frac,
                   bool neox) override {
        KRK_KTIME("rope");
        npu_flush_();
        for (i64 t = 0; t < n_tok; t++) {
            rope_apply(static_cast<f32 *>(q) + t * n_head * hd, n_head, hd, 1, 0,
                       inv_freq, scale, frac, neox, row_pos + t);
            rope_apply(static_cast<f32 *>(k) + t * n_kv * hd, n_kv, hd, 1, 0,
                       inv_freq, scale, frac, neox, row_pos + t);
        }
        return true;
    }

    void qk_norm(void *q, void *k, const f32 *wq, const f32 *wk, i64 n_head,
                 i64 n_kv, i64 hd, i64 n_tok, f32 eps, bool wide) override {
        // Both arms of this are one RMSNorm per row with a row-wide weight, so
        // the device takes them as one call each: `wide` normalizes over the
        // whole q row, and the per-head form is the same arithmetic on hd-wide
        // rows, of which there are n_tok * n_head.
        bool queued = true;
        if (wq) {
            queued = wide
                         ? npu::rmsnorm(npu_, q, q, wq, n_tok, n_head * hd, eps)
                         : npu::rmsnorm(npu_, q, q, wq, n_tok * n_head, hd, eps);
            if (!queued) npu_flush_();
        }
        if (wq && queued) {
            if (wk) {
                if (wide)
                    npu::rmsnorm(npu_, k, k, wk, n_tok, n_kv * hd, eps);
                else
                    npu::rmsnorm(npu_, k, k, wk, n_tok * n_kv, hd, eps);
            }
            return;
        }
        if (wq) {
            if (wide)
                row_norm(static_cast<f32 *>(q), wq, n_tok, n_head * hd, eps);
            else
                head_norm(static_cast<f32 *>(q), wq, n_head, hd, n_tok, eps);
        }
        if (wk) {
            if (wide)
                row_norm(static_cast<f32 *>(k), wk, n_tok, n_kv * hd, eps);
            else
                head_norm(static_cast<f32 *>(k), wk, n_kv, hd, n_tok, eps);
        }
    }

    // Int8 cache: the row is `kv_dim` signed bytes followed by one f32 scale
    // per 32 of them. The workspace rows are still f32 and still `kv_dim`
    // wide, so the SOURCE stride is the head width while the DESTINATION
    // stride is d.pos_stride (the quantized row size). Quantising per row per
    // 32 means the same activation always produces the same bytes, whatever
    // chunk it arrived in: that is what keeps a chunked prefill and a decode
    // step agreeing on a quantized cache.
    bool kv_quant_supported(KvQuant q) const override {
        return q == KvQuant::F16 || q == KvQuant::Int8;
    }

    size_t kv_row_bytes(i64 kv_dim, KvQuant q) const override {
        if (q == KvQuant::F16) return static_cast<size_t>(kv_dim) * sizeof(f32);
        const size_t blocks = static_cast<size_t>((kv_dim + 31) / 32);
        return static_cast<size_t>(kv_dim) + blocks * sizeof(f32);
    }

    void kv_append(void *kcache, void *vcache, const void *k, const void *v,
                   const AttnDesc &d) override {
        KRK_KTIME("kv_append");
        npu_flush_();
        if (d.kv_int8) {
            const i64 kv_dim = d.n_kv * d.hd;
            u8 *kc = static_cast<u8 *>(kcache) + d.layer * d.layer_stride;
            u8 *vc = static_cast<u8 *>(vcache) + d.layer * d.layer_stride;
            const f32 *ks = static_cast<const f32 *>(k);
            const f32 *vs = static_cast<const f32 *>(v);
            // K rows and V rows are disjoint destinations.
            par_for(2 * d.n_tok, 1,
                    4.0 * static_cast<f64>(d.n_tok) * static_cast<f64>(kv_dim),
                    [&](i64 b, i64 e) {
                        for (i64 u = b; u < e; u++) {
                            const bool is_v = u >= d.n_tok;
                            const i64 t = is_v ? u - d.n_tok : u;
                            const f32 *src = (is_v ? vs : ks) + t * kv_dim;
                            // Per-row base and position: zero and pos0 + t for
                            // every caller that does not set the tables, so this
                            // is the old address for the old shape.
                            u8 *row = (is_v ? vc : kc) + attn_kv_base(d, t) +
                                      attn_pos(d, t) * d.pos_stride;
                            f32 *sc = reinterpret_cast<f32 *>(row + kv_dim);
                            for (i64 o = 0; o < kv_dim; o += 32) {
                                const i64 nn = std::min<i64>(32, kv_dim - o);
                                f32 mx = 0.0f;
                                for (i64 i = 0; i < nn; i++)
                                    mx = std::max(mx, std::fabs(src[o + i]));
                                const f32 s = mx > 0.0f ? mx / 127.0f : 0.0f;
                                sc[o / 32] = s;
                                const f32 inv = s > 0.0f ? 1.0f / s : 0.0f;
                                for (i64 i = 0; i < nn; i++) {
                                    const f32 qv = src[o + i] * inv;
                                    i64 r = static_cast<i64>(std::lround(qv));
                                    r = std::max<i64>(-127, std::min<i64>(127, r));
                                    row[o + i] = static_cast<u8>(static_cast<i8>(r));
                                }
                            }
                        }
                    });
            return;
        }
        f32 *kc = static_cast<f32 *>(kcache) + d.layer * d.layer_stride;
        f32 *vc = static_cast<f32 *>(vcache) + d.layer * d.layer_stride;
        const f32 *ks = static_cast<const f32 *>(k);
        const f32 *vs = static_cast<const f32 *>(v);
        // Tokens land at disjoint cache slots.
        par_for(d.n_tok, 4, 2.0 * static_cast<f64>(d.n_tok) * d.pos_stride,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        const i64 dst = attn_kv_base(d, t) +
                                        attn_pos(d, t) * d.pos_stride;
                        std::memcpy(kc + dst, ks + t * d.pos_stride,
                                    static_cast<size_t>(d.pos_stride) * 4);
                        std::memcpy(vc + dst, vs + t * d.pos_stride,
                                    static_cast<size_t>(d.pos_stride) * 4);
                    }
                });
    }

    bool gather_rows_supported() const override { return true; }

    void gather_rows(void *dst, const void *src, const i64 *idx, i64 n_rows,
                     size_t row_bytes) override {
        npu_flush_();
        if (!dst || !src || !idx || n_rows <= 0 || row_bytes == 0) return;
        u8 *o = static_cast<u8 *>(dst);
        const u8 *s = static_cast<const u8 *>(src);
        // Rows are disjoint destination ranges, so the pool may split them.
        par_for(n_rows, 4,
                2.0 * static_cast<f64>(n_rows) * static_cast<f64>(row_bytes),
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++) {
                        const i64 cell = idx[r];
                        if (cell < 0) continue;
                        std::memcpy(o + static_cast<size_t>(r) * row_bytes,
                                    s + static_cast<size_t>(cell) * row_bytes,
                                    row_bytes);
                    }
                });
    }

    // ---- block-sparse rows (MSA) -----------------------------------------
    // A separate body on purpose. Threading a run list through attn_range would
    // touch the one function every model's attention reads, to serve one family
    // that no other path uses; this way a run that sets no key_runs executes
    // exactly the instructions it did before.
    //
    // The arithmetic is the scalar reference shape: score over the visible keys
    // in run order, one softmax over all of them, then the weighted sum. Run
    // order is the caller's order, so the same runs always produce the same
    // bits -- a selected-block set has no canonical order and inventing one
    // would make two equal selections disagree in the last place.
    void attn_sparse_rows(void *out, const void *q, const void *kcache,
                          const void *vcache, const AttnDesc &d) {
        const i64 hd = d.hd, nh = d.n_head, nkv = d.n_kv, nt = d.n_tok;
        const f32 *qp = static_cast<const f32 *>(q);
        f32 *op = static_cast<f32 *>(out);
        std::vector<f32> sc;
        for (i64 t = 0; t < nt; t++) {
            const i64 pos = attn_pos(d, t);
            const i64 n_keys = d.causal ? pos + 1 : d.pos0 + d.n_tok;
            const i64 j_lo = d.window > 0 ? std::max<i64>(0, n_keys - d.window) : 0;
            // Resolve the runs once per row: clamp, intersect with the window,
            // drop empties. What survives is the exact key set.
            std::vector<std::pair<i64, i64>> runs;
            runs.reserve(static_cast<size_t>(d.n_key_runs));
            for (i64 r = 0; r < d.n_key_runs; r++) {
                const i64 lo = std::max(d.key_runs[2 * r + 0], j_lo);
                const i64 hi = std::min(d.key_runs[2 * r + 1], n_keys);
                if (hi > lo) runs.emplace_back(lo, hi);
            }
            i64 n_vis = 0;
            for (const auto &r : runs) n_vis += r.second - r.first;
            for (i64 h = 0; h < nh; h++) {
                const f32 *qh = qp + (t * nh + h) * hd;
                const i64 kh = h / (nh / nkv);
                f32 *oh = op + (t * nh + h) * hd;
                if (n_vis <= 0) {
                    for (i64 i = 0; i < hd; i++) oh[i] = 0.0f;
                    continue;
                }
                sc.assign(static_cast<size_t>(n_vis), 0.0f);
                i64 k = 0;
                f32 mx = -INFINITY;
                for (const auto &rn : runs) {
                    for (i64 j = rn.first; j < rn.second; j++, k++) {
                        const char *kb = static_cast<const char *>(kcache);
                        // Base in ELEMENTS plus j's stride, the same address
                        // every other attention path forms. This one used to
                        // multiply the region base by pos_stride as well, which
                        // was invisible while no caller set row_kv_base and
                        // would have read past the cache as soon as one did.
                        const i64 off = (d.layer * d.layer_stride +
                                         attn_kv_base(d, t) +
                                         j * d.pos_stride + kh * hd);
                        const f32 *kv = reinterpret_cast<const f32 *>(kb + off * 4);
                        f32 s = 0.0f;
                        for (i64 i = 0; i < hd; i++) s += qh[i] * kv[i];
                        s *= d.scale;
                        if (d.softcap > 0.0f)
                            s = d.softcap * std::tanh(s / d.softcap);
                        sc[static_cast<size_t>(k)] = s;
                        if (s > mx) mx = s;
                    }
                }
                f32 l = 0.0f;
                if (d.sinks) {
                    const f32 sk = d.sinks[h];
                    if (sk != 0.0f) l = std::exp(sk - mx);
                }
                for (i64 i = 0; i < n_vis; i++) {
                    const f32 e = std::exp(sc[static_cast<size_t>(i)] - mx);
                    sc[static_cast<size_t>(i)] = e;
                    l += e;
                }
                for (i64 i = 0; i < hd; i++) oh[i] = 0.0f;
                if (l <= 0.0f) continue;
                const f32 inv = 1.0f / l;
                k = 0;
                for (const auto &rn : runs) {
                    for (i64 j = rn.first; j < rn.second; j++, k++) {
                        const f32 p = sc[static_cast<size_t>(k)] * inv;
                        const char *vb = static_cast<const char *>(vcache);
                        // Base in ELEMENTS plus j's stride, the same address
                        // every other attention path forms. This one used to
                        // multiply the region base by pos_stride as well, which
                        // was invisible while no caller set row_kv_base and
                        // would have read past the cache as soon as one did.
                        const i64 off = (d.layer * d.layer_stride +
                                         attn_kv_base(d, t) +
                                         j * d.pos_stride + kh * hd);
                        const f32 *vv = reinterpret_cast<const f32 *>(vb + off * 4);
                        for (i64 i = 0; i < hd; i++) oh[i] += p * vv[i];
                    }
                }
            }
        }
    }

    void attention(void *out, const void *q, const void *kcache, const void *vcache,
                   const AttnDesc &d) override {
        // Attention is a GEMM pair -- QK^T then AV -- over the cached keys, so
        // its MACs are countable and its ceiling is the f32 FMA rate. It is the
        // one non-GEMM component with a roofline worth printing.
        KRK_KTIME_MAC("attention",
                      static_cast<double>(d.n_tok) * static_cast<double>(d.n_head) *
                          static_cast<double>(d.pos0 + d.n_tok) *
                          static_cast<double>(d.hd) * 2.0,
                      isa_peak_gmacs(false));
        // npu_flush_ first: q is usually the output of a GEMM still queued on
        // the device, and the gather below reads it on the host.
        npu_flush_();
        // Block-sparse (MSA) before anything else: the NPU hand-off and the
        // gather path both assume a contiguous key range, so an MSA call is
        // taken here rather than being handed to a kernel that would read it
        // densely. An int8 cache is a different layout altogether and is
        // refused rather than reinterpreted.
        if (d.key_runs != nullptr && d.n_key_runs > 0) {
            if (d.kv_int8) {
                KRK_WARN("block-sparse attention with an int8 cache is not "
                         "implemented; refusing rather than attending densely");
                std::memset(out, 0, static_cast<size_t>(d.n_tok) * d.n_head * d.hd * 4);
                return;
            }
            attn_sparse_rows(out, q, kcache, vcache, d);
            return;
        }
        if (npu_) {
            // Everything that can make this call unrepresentable is handed over
            // rather than tested here, so the decision, the counting and the
            // run-summary wording all come from one place (npu::attention).
            npu::AttnCall c;
            c.out = out;
            c.q = q;
            c.kcache = kcache;
            c.vcache = vcache;
            c.n_head = d.n_head;
            c.n_kv = d.n_kv;
            c.hd = d.hd;
            c.n_tok = d.n_tok;
            c.pos0 = d.pos0;
            c.layer = d.layer;
            c.layer_stride = d.layer_stride;
            c.pos_stride = d.pos_stride;
            c.scale = d.scale;
            c.causal = d.causal;
            c.kv_int8 = d.kv_int8;
            c.has_sinks = d.sinks != nullptr;
            c.window = d.window;
            c.skip_from = d.skip_from;
            c.softcap = d.softcap;
            c.alibi_max_bias = d.alibi_max_bias;
            if (npu::attention(npu_, c)) return;
        }
        const i64 n_rep = d.n_kv > 0 ? d.n_head / d.n_kv : 1;
        if (d.kv_int8) {
            // Same formulation as the f32 path, reading (signed byte x scale)
            // instead of a float. The dequantised value is what the dot and the
            // value accumulation consume, so the only difference from f16 is
            // the 8-bit quantisation of the cached activation itself.
            const i64 kv_dim = d.n_kv * d.hd;
            const f32 *qf = static_cast<const f32 *>(q);
            const u8 *kc = static_cast<const u8 *>(kcache) + d.layer * d.layer_stride;
            const u8 *vc = static_cast<const u8 *>(vcache) + d.layer * d.layer_stride;
            f32 *of = static_cast<f32 *>(out);
            const i64 units = d.n_tok * d.n_head;
            par_for(units, 1,
                    4.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * d.hd,
                    [&](i64 b, i64 e) {
                        std::vector<f32> scores;
                        for (i64 u = b; u < e; u++) {
                            const i64 t = u / d.n_head;
                            const i64 h = u - t * d.n_head;
                            const i64 pos = attn_pos(d, t);
                            const i64 kvb = attn_kv_base(d, t);
                            const i64 n_keys = pos + 1;
                            const i64 j_lo = d.window > 0
                                                 ? std::max<i64>(0, n_keys - d.window)
                                                 : 0;
                            const i64 n_vis = n_keys - j_lo;
                            // skip_from: a masked-block row keeps the context
                            // before the block and its own row. The range is
                            // halved here rather than tested per key, so the
                            // visible set stays two contiguous runs.
                            const bool skip = d.skip_from >= 0 && pos >= d.skip_from;
                            const i64 s_lo = skip ? std::max<i64>(j_lo, d.skip_from) : 0;
                            const i64 s_hi = skip ? std::min<i64>(pos, n_keys) : 0;
                            scores.resize(static_cast<size_t>(n_vis));
                            const i64 hkv = n_rep > 0 ? h / n_rep : 0;
                            const i64 base = hkv * d.hd;
                            // One slope per (head), computed once outside the
                            // key loop rather than per key.
                            const bool alibi = d.alibi_max_bias > 0.0f;
                            const f32 slope =
                                alibi ? alibi_slope_impl(h, d.n_head,
                                                         d.alibi_max_bias)
                                      : 0.0f;
                            const f32 *qh = qf + t * (d.n_head * d.hd) + h * d.hd;
                            for (i64 j = 0; j < n_vis; j++) {
                                if (s_hi > s_lo && j_lo + j >= s_lo &&
                                    j_lo + j < s_hi) {
                                    scores[static_cast<size_t>(j)] = -FLT_MAX;
                                    continue;
                                }
                                const u8 *row = kc + kvb + (j_lo + j) * d.pos_stride;
                                const f32 *sc =
                                    reinterpret_cast<const f32 *>(row + kv_dim);
                                f32 dot = 0.0f;
                                for (i64 i = 0; i < d.hd; i++) {
                                    const i64 c = base + i;
                                    const f32 kv = static_cast<f32>(
                                                       reinterpret_cast<const i8 *>(row)[c]) *
                                                   sc[c / 32];
                                    dot += qh[i] * kv;
                                }
                                // Gemma 2's attention-logit bound, applied
                                // where the score is formed (see
                                // AttnDesc::softcap); the masked branch above
                                // never reaches here, so the mask stays -inf.
                                f32 sv = dot * d.scale;
                                if (d.softcap > 0.0f)
                                    sv = d.softcap * std::tanh(sv / d.softcap);
                                // ALiBi: the same additive distance bias
                                // llama.cpp folds into its softmax, computed
                                // from the same scalar with the same formula
                                // (see AttnDesc::alibi_max_bias). The distance
                                // is signed the way llama.cpp's mask is --
                                // -(pos_q - pos_k) -- and the row is causal, so
                                // every key seen here is at or before the
                                // query.
                                if (alibi) sv -= slope * static_cast<f32>(
                                                       pos - (j_lo + j));
                                scores[static_cast<size_t>(j)] = sv;
                            }
                            f32 mx = scores[0];
                            for (i64 j = 1; j < n_vis; j++)
                                mx = std::max(mx, scores[static_cast<size_t>(j)]);
                            f32 sum = 0.0f;
                            for (i64 j = 0; j < n_vis; j++) {
                                const f32 ex =
                                    std::exp(scores[static_cast<size_t>(j)] - mx);
                                scores[static_cast<size_t>(j)] = ex;
                                sum += ex;
                            }
                            const f32 sink_logit = d.sinks ? d.sinks[h] : 0.0f;
                            if (sink_logit != 0.0f) sum += std::exp(sink_logit - mx);
                            const f32 inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                            f32 *oh = of + t * (d.n_head * d.hd) + h * d.hd;
                            for (i64 i = 0; i < d.hd; i++) oh[i] = 0.0f;
                            for (i64 j = 0; j < n_vis; j++) {
                                const f32 p = scores[static_cast<size_t>(j)] * inv;
                                const u8 *row = vc + (j_lo + j) * d.pos_stride;
                                const f32 *sc =
                                    reinterpret_cast<const f32 *>(row + kv_dim);
                                for (i64 i = 0; i < d.hd; i++) {
                                    const i64 c = base + i;
                                    const f32 vv = static_cast<f32>(
                                                       reinterpret_cast<const i8 *>(row)[c]) *
                                                   sc[c / 32];
                                    oh[i] += p * vv;
                                }
                            }
                        }
                    });
            return;
        }
        const f32 *qf = static_cast<const f32 *>(q);
        const f32 *kc = static_cast<const f32 *>(kcache) + d.layer * d.layer_stride;
        const f32 *vc = static_cast<const f32 *>(vcache) + d.layer * d.layer_stride;
        f32 *of = static_cast<f32 *>(out);
        // (token, head) pairs are independent: each owns its
        // output rows and keeps a PRIVATE score scratch, so
        // nothing is shared across blocks and the softmax
        // max/sum stay inside one head — bit-identical to the
        // serial loop (which used one shared scratch).
        const i64 units = d.n_tok * d.n_head;
        // One binary, two ISAs: the vector path where the host has AVX2+FMA,
        // the scalar body below everywhere else. KRK_CPU_ATTN_SCALAR=1 forces
        // the reference -- that is the A/B this was measured with, and the arm
        // a bit-exactness run wants.
#if defined(KRK_ATTN_AVX2)
        // Latched once: the ISA probe is a cpuid-derived bit test, but it is
        // still read on every layer of every token, and the answer cannot
        // change within a process. KRK_CPU_ATTN_SCALAR=1 is the reference arm
        // the accuracy fixture and the A/B runs compare against.
        static const bool attn_vec = [] {
            const char *e = std::getenv("KRK_CPU_ATTN_SCALAR");
            if (e && e[0] == '1') return false;
            return __builtin_cpu_supports("avx2") &&
                   __builtin_cpu_supports("fma");
        }();
        // Gather arm, latched with the ISA probe. KRK_CPU_ATTN_GATHER=0 keeps
        // the in-place path, which is the A/B and the arm a cache-layout
        // experiment wants.
        static const i64 n_gather = [] {
            const char *e = std::getenv("KRK_CPU_ATTN_GATHER");
            return (e && e[0] == '0') ? 0 : 1;
        }();
        // Gather the visible key/value slices into a contiguous tile, per head,
        // for the whole token chunk.
        //
        // The cache is [position][head][hd], so a head's slice of one key is 256
        // B out of a 3072 B stride on this model. The score loop then reads
        // four cache lines and uses one -- measured on the same arithmetic,
        // 16.14 ns/key strided against 2.99 ns/key contiguous, a 5.4x gap that
        // no amount of loop work recovers (four keys in flight: 14.47; software
        // prefetch: 13.77).
        //
        // The gather does not remove that traffic, it AMORTISES it: one pass
        // over the strided cache serves every query row in the chunk, instead
        // of once per (token, head). That is why it is gated on n_tok > 1 --
        // for a decode step there is nothing to amortise and the copy is pure
        // cost. It is also why the rows are walked head-major here: the tile
        // belongs to a head, and every token in the chunk reuses it.
        //
        // The tile is exactly the bytes each row would have read, in the same
        // order, so the row body runs the identical arithmetic -- accuracy is
        // checked against the in-place path on ten shapes, not argued.
        // The gather copies one head's slices out of a single contiguous run of
        // keys, so it cannot express rows that address different sequences.
        // Multi-sequence batches therefore take the per-row paths below; the
        // copy is an amortisation, not a requirement, and it is the next thing
        // to generalise once the batched path is measured.
        if (attn_vec && n_gather > 0 && d.n_tok > 1 && !attn_multiseq(d)) {
            par_for(d.n_head, 1,
                    4.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * d.hd,
                    [&](i64 hb, i64 he) {
                        std::vector<f32> kg, vg, scores;
                        for (i64 h = hb; h < he; h++) {
                            const i64 hkv = n_rep > 0 ? h / n_rep : 0;
                            const i64 base_j = attn_range(d, 0).j_lo;
                            const i64 span = attn_range(d, d.n_tok - 1).n_keys -
                                             base_j;
                            if (span <= 0) continue;
                            const size_t need = static_cast<size_t>(span * d.hd);
                            if (kg.size() < need) {
                                kg.resize(need);
                                vg.resize(need);
                            }
                            const f32 *kb = kc + base_j * d.pos_stride + hkv * d.hd;
                            const f32 *vb = vc + base_j * d.pos_stride + hkv * d.hd;
                            for (i64 j = 0; j < span; j++) {
                                std::memcpy(kg.data() + j * d.hd,
                                            kb + j * d.pos_stride,
                                            static_cast<size_t>(d.hd) * sizeof(f32));
                                std::memcpy(vg.data() + j * d.hd,
                                            vb + j * d.pos_stride,
                                            static_cast<size_t>(d.hd) * sizeof(f32));
                            }
                            for (i64 t = 0; t < d.n_tok; t++) {
                                const AttnRange r = attn_range(d, t);
                                if (r.n_vis <= 0) continue;
                                // The row's own position, for ALiBi's distance.
                                const i64 pos = attn_pos(d, t);
                                scores.resize(static_cast<size_t>(r.n_vis));
                                const i64 off = (r.j_lo - base_j) * d.hd;
                                attn_row_f32_avx2(
                                    qf, kg.data() + off, vg.data() + off, d.hd,
                                    of, d, t, h, pos - (r.n_vis - 1), r,
                                    scores.data());
                            }
                        }
                    });
            return;
        }
        if (attn_vec) {
            par_for(units, 1,
                    4.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * d.hd,
                    [&](i64 b, i64 e) {
                        std::vector<f32> scores;
                        for (i64 u = b; u < e; u++) {
                            const i64 t = u / d.n_head;
                            const i64 h = u - t * d.n_head;
                            const AttnRange r = attn_range(d, t);
                            const i64 pos = attn_pos(d, t);
                            scores.resize(static_cast<size_t>(r.n_vis));
                            const i64 hkv = n_rep > 0 ? h / n_rep : 0;
                            // The row's own KV region, so a batch can carry rows
                            // from different sequences. Zero for every call that
                            // does not set row_kv_base.
                            const i64 kvb = attn_kv_base(d, t);
                            attn_row_f32_avx2(
                                qf,
                                kc + kvb + r.j_lo * d.pos_stride + hkv * d.hd,
                                vc + kvb + r.j_lo * d.pos_stride + hkv * d.hd,
                                d.pos_stride, of, d, t, h,
                                pos - (r.n_vis - 1), r, scores.data());
                        }
                    });
            return;
        }
#endif
        par_for(units, 1,
                4.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * d.hd,
                [&](i64 b, i64 e) {
                    std::vector<f32> scores;
                    for (i64 u = b; u < e; u++) {
                        const i64 t = u / d.n_head;
                        const i64 h = u - t * d.n_head;
                        const i64 pos = attn_pos(d, t);
                        const i64 kvb = attn_kv_base(d, t);
                        const i64 n_keys = pos + 1;
                        // Sliding window (laguna's hybrid layers): the query
                        // sees only the `window` most recent keys, so the ones
                        // below the bound are skipped outright -- they never
                        // enter the max or the denominator.
                        const i64 j_lo = d.window > 0
                                             ? std::max<i64>(0, n_keys - d.window)
                                             : 0;
                        const i64 n_vis = n_keys - j_lo;
                        // See the int8 path above: masked-block rows keep the
                        // context and their own row, and drop the block's other
                        // candidates. A masked key contributes exp(-FLT_MAX - m)
                        // = 0 to the denominator and nothing to the output.
                        const bool skip = d.skip_from >= 0 && pos >= d.skip_from;
                        const i64 s_lo = skip ? std::max<i64>(j_lo, d.skip_from) : 0;
                        const i64 s_hi = skip ? std::min<i64>(pos, n_keys) : 0;
                        scores.resize(static_cast<size_t>(n_vis));
                        const i64 hkv = n_rep > 0 ? h / n_rep : 0;
                        const f32 *qh = qf + t * (d.n_head * d.hd) + h * d.hd;
                        // ALiBi, hoisted out of the key loop: the slope is a
                        // property of the head. Same formula and same signed
                        // distance as the int8 body above -- this f32 path
                        // ignored the bias entirely until test_alibi_slope_ladder
                        // recovered the ladder from the softmax output.
                        const f32 slope =
                            d.alibi_max_bias > 0.0f
                                ? alibi_slope_impl(h, d.n_head, d.alibi_max_bias)
                                : 0.0f;
                        for (i64 j = 0; j < n_vis; j++) {
                            if (s_hi > s_lo && j_lo + j >= s_lo &&
                                j_lo + j < s_hi) {
                                scores[static_cast<size_t>(j)] = -FLT_MAX;
                                continue;
                            }
                            const f32 *kj =
                                kc + kvb + (j_lo + j) * d.pos_stride + hkv * d.hd;
                            f32 dot = 0;
                            for (i64 i = 0; i < d.hd; i++) dot += qh[i] * kj[i];
                            f32 sv = dot * d.scale;
                            if (d.softcap > 0.0f) sv = d.softcap * std::tanh(sv / d.softcap);
                            if (slope != 0.0f)
                                sv -= slope * static_cast<f32>(pos - (j_lo + j));
                            scores[static_cast<size_t>(j)] = sv;
                        }
                        f32 mx = scores[0];
                        for (i64 j = 1; j < n_vis; j++)
                            mx = std::max(mx, scores[static_cast<size_t>(j)]);
                        f32 sum = 0;
                        for (i64 j = 0; j < n_vis; j++) {
                            const f32 ex = std::exp(scores[static_cast<size_t>(j)] - mx);
                            scores[static_cast<size_t>(j)] = ex;
                            sum += ex;
                        }
                        // The sink enters at this row's running max and
                        // contributes no value, so `scores` and the output
                        // sum are untouched. Guarded twice: an absent sink
                        // keeps the exact arithmetic above, and a ZEROED
                        // sink logit is treated as no sink, so a zeroed
                        // sink row is bit-identical to the absent path --
                        // the A/B arm the test group pins with `==`.
                        const f32 sink_logit = d.sinks ? d.sinks[h] : 0.0f;
                        if (sink_logit != 0.0f)
                            sum += std::exp(sink_logit - mx);
                        const f32 inv = sum > 0 ? 1.0f / sum : 0.0f;
                        f32 *oh = of + t * (d.n_head * d.hd) + h * d.hd;
                        for (i64 i = 0; i < d.hd; i++) oh[i] = 0;
                        for (i64 j = 0; j < n_vis; j++) {
                            const f32 p = scores[static_cast<size_t>(j)] * inv;
                            const f32 *vj =
                                vc + kvb + (j_lo + j) * d.pos_stride + hkv * d.hd;
                            for (i64 i = 0; i < d.hd; i++) oh[i] += p * vj[i];
                        }
                    }
                });
    }

    // ---- multi-head latent attention (see MlaDesc) -----------------------
    //
    // The reference the HIP kernel is judged against, in the same order the
    // kernel uses: absorb, score the latent plus the decoupled key, softmax,
    // accumulate the latent, fold the value half. `wkv_b` is the f16 staging
    // the loader built from the file's quantized table, so both paths read
    // the same numbers -- a mismatch between them is arithmetic, never a
    // different weight.
    bool mla_supported() const override { return true; }

    void mla_kv_write(void *cache, const void *proj, const void *gamma,
                      const MlaDesc &d) override {
        npu_flush_();
        f32 *c = static_cast<f32 *>(cache);
        const f32 *p = static_cast<const f32 *>(proj);
        const f32 *g = static_cast<const f32 *>(gamma);
        const i64 kv = d.kv_lora;
        const i64 rope = d.rope;
        const i64 width = kv + rope;
        const f32 eps = d.eps;
        par_for(d.n_tok, 1, 4.0 * static_cast<f64>(d.n_tok) * width,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        const f32 *src = p + t * width;
                        f32 *dst = c + (d.pos0 + t) * d.pos_stride;
                        // The norm covers the LATENT alone: the decoupled key
                        // is not normed, and it is cached UNROTATED.
                        f64 ss = 0.0;
                        for (i64 i = 0; i < kv; i++) {
                            const f64 v = src[i];
                            ss += v * v;
                        }
                        const f32 inv = 1.0f / std::sqrt(
                            static_cast<f32>(ss / static_cast<f64>(kv)) + eps);
                        for (i64 i = 0; i < kv; i++)
                            dst[i] = src[i] * inv * g[i];
                        for (i64 i = 0; i < rope; i++) dst[kv + i] = src[kv + i];
                    }
                });
    }

    void mla_attention(void *out, const void *q, const void *cache,
                       const void *wkv_b, const MlaDesc &d) override {
        npu_flush_();
        const f32 *qf = static_cast<const f32 *>(q);
        const f32 *cf = static_cast<const f32 *>(cache);
        const u16 *w = static_cast<const u16 *>(wkv_b);
        f32 *of = static_cast<f32 *>(out);
        const i64 nope = d.nope;
        const i64 rope = d.rope;
        const i64 kv = d.kv_lora;
        const i64 vd = d.v_dim;
        const i64 hv = nope + vd;
        // (row, head) pairs are independent and own their whole softmax.
        const i64 units = d.n_tok * d.n_head;
        par_for(units, 1,
                8.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * kv,
                [&](i64 b, i64 e) {
                    std::vector<f32> qk(static_cast<size_t>(kv));
                    std::vector<f32> cv(static_cast<size_t>(kv));
                    std::vector<f32> qpe(static_cast<size_t>(rope));
                    std::vector<f32> scores;
                    for (i64 u = b; u < e; u++) {
                        const i64 r = u / d.n_head;
                        const i64 h = u - r * d.n_head;
                        const i64 pos = d.pos0 + r;
                        const i64 n_keys = pos + 1; // causal
                        const f32 *qrow =
                            qf + r * (d.n_head * d.qk) + h * d.qk;
                        const f32 *qn = qrow;
                        const f32 *qr = qrow + nope;
                        for (i64 i = 0; i < rope; i += 2) {
                            const f32 th = static_cast<f32>(pos) *
                                           d.inv_freq[i / 2];
                            const f32 cs = std::cos(th), sn = std::sin(th);
                            qpe[static_cast<size_t>(i)] =
                                qr[i] * cs - qr[i + 1] * sn;
                            qpe[static_cast<size_t>(i + 1)] =
                                qr[i] * sn + qr[i + 1] * cs;
                        }
                        // The absorb: the K half of head h folded into q_nope.
                        // Feature j = h*hv + dd owns the contiguous row at
                        // j*kv (the file's own order, staged as f16).
                        for (i64 k = 0; k < kv; k++) {
                            f32 s = 0.0f;
                            for (i64 dd = 0; dd < nope; dd++)
                                s += fp16_to_fp32(
                                         w[(h * hv + dd) * kv + k]) * qn[dd];
                            qk[static_cast<size_t>(k)] = s;
                        }
                        scores.assign(static_cast<size_t>(n_keys), 0.0f);
                        f32 mx = -1e30f;
                        for (i64 t = 0; t < n_keys; t++) {
                            const f32 *c = cf + t * d.pos_stride;
                            f32 s = 0.0f;
                            for (i64 k = 0; k < kv; k++)
                                s += qk[static_cast<size_t>(k)] * c[k];
                            const f32 *kpe = c + kv;
                            for (i64 i = 0; i < rope; i += 2) {
                                const f32 th = static_cast<f32>(t) *
                                               d.inv_freq[i / 2];
                                const f32 cs = std::cos(th), sn = std::sin(th);
                                const f32 a = kpe[i], bb = kpe[i + 1];
                                s += qpe[static_cast<size_t>(i)] *
                                         (a * cs - bb * sn) +
                                     qpe[static_cast<size_t>(i + 1)] *
                                         (a * sn + bb * cs);
                            }
                            s *= d.scale;
                            scores[static_cast<size_t>(t)] = s;
                            if (s > mx) mx = s;
                        }
                        f32 sum = 0.0f;
                        for (i64 t = 0; t < n_keys; t++) {
                            const f32 ex =
                                std::exp(scores[static_cast<size_t>(t)] - mx);
                            scores[static_cast<size_t>(t)] = ex;
                            sum += ex;
                        }
                        for (i64 k = 0; k < kv; k++) {
                            f32 a = 0.0f;
                            for (i64 t = 0; t < n_keys; t++)
                                a += scores[static_cast<size_t>(t)] *
                                     cf[t * d.pos_stride + k];
                            cv[static_cast<size_t>(k)] = a / sum;
                        }
                        f32 *oh = of + r * (d.n_head * vd) + h * vd;
                        for (i64 dd = 0; dd < vd; dd++) {
                            const u16 *wc =
                                w + (h * hv + nope + dd) * kv;
                            f32 a = 0.0f;
                            for (i64 k = 0; k < kv; k++)
                                a += cv[static_cast<size_t>(k)] *
                                     fp16_to_fp32(wc[k]);
                            oh[dd] = a;
                        }
                    }
                });
    }

    // ---- V4 shared-latent MLA (see MlaSharedDesc) ------------------------
    // The reference the HIP kernels are judged against, in the kernels' own
    // order and on the kernels' own f16 weights, so a difference between the
    // two arms is arithmetic and never a different table.
    bool mla_shared_supported() const override { return true; }

    void mla_shared_latent(void *cache, const void *proj, const void *gamma,
                           const MlaSharedDesc &d) override {
        npu_flush_();
        f32 *c = static_cast<f32 *>(cache);
        const f32 *p = static_cast<const f32 *>(proj);
        const f32 *g = static_cast<const f32 *>(gamma);
        const i64 kv = d.kv_lora;
        par_for(d.n_tok, 1, 4.0 * static_cast<f64>(d.n_tok) * kv,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        const f32 *src = p + t * kv;
                        f32 *dst = c + (d.pos0 + t) * d.pos_stride;
                        f64 ss = 0.0;
                        for (i64 i = 0; i < kv; i++) {
                            const f64 v = src[i];
                            ss += v * v;
                        }
                        const f32 inv = 1.0f / std::sqrt(
                            static_cast<f32>(ss / static_cast<f64>(kv)) + d.eps);
                        for (i64 i = 0; i < kv; i++)
                            dst[i] = src[i] * inv * g[i];
                    }
                });
    }

    void mla_shared_attention(void *a, const void *q, const void *cache,
                              const MlaSharedDesc &d) override {
        npu_flush_();
        f32 *af = static_cast<f32 *>(a);
        const f32 *qf = static_cast<const f32 *>(q);
        const f32 *cf = static_cast<const f32 *>(cache);
        const i64 nh = d.n_head, kv = d.kv_lora, rope = d.qk_rope;
        const i64 base = kv - rope;
        const i64 units = d.n_tok * nh;
        par_for(units, 1,
                8.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * kv,
                [&](i64 b, i64 e) {
                    std::vector<f32> qr(static_cast<size_t>(kv));
                    std::vector<f32> kr(static_cast<size_t>(kv));
                    std::vector<f32> p;
                    for (i64 u = b; u < e; u++) {
                        const i64 r = u / nh;
                        const i64 h = u - r * nh;
                        const i64 pos = d.pos0 + r;
                        const f32 *qrow = qf + r * (nh * kv) + h * kv;
                        for (i64 i = 0; i < kv; i++) qr[static_cast<size_t>(i)] = qrow[i];
                        for (i64 i = 0; i + 1 < rope; i += 2) {
                            const f32 th =
                                static_cast<f32>(pos) * d.inv_freq[i / 2];
                            const f32 cs = std::cos(th), sn = std::sin(th);
                            const f32 x0 = qr[static_cast<size_t>(base + i)];
                            const f32 x1 = qr[static_cast<size_t>(base + i + 1)];
                            qr[static_cast<size_t>(base + i)] = x0 * cs - x1 * sn;
                            qr[static_cast<size_t>(base + i + 1)] = x0 * sn + x1 * cs;
                        }
                        const i64 t_lo = d.window > 0 ? std::max<i64>(0, pos - d.window + 1) : 0;
                        const i64 nk = pos - t_lo + 1;
                        p.assign(static_cast<size_t>(nk), 0.0f);
                        f32 mx = d.sinks ? d.sinks[h] : -1e30f;
                        for (i64 t = t_lo; t <= pos; t++) {
                            const f32 *c = cf + t * d.pos_stride;
                            for (i64 i = 0; i < kv; i++) kr[static_cast<size_t>(i)] = c[i];
                            for (i64 i = 0; i + 1 < rope; i += 2) {
                                const f32 th =
                                    static_cast<f32>(t) * d.inv_freq[i / 2];
                                const f32 cs = std::cos(th), sn = std::sin(th);
                                const f32 x0 = kr[static_cast<size_t>(base + i)];
                                const f32 x1 = kr[static_cast<size_t>(base + i + 1)];
                                kr[static_cast<size_t>(base + i)] = x0 * cs - x1 * sn;
                                kr[static_cast<size_t>(base + i + 1)] = x0 * sn + x1 * cs;
                            }
                            f32 s = 0.0f;
                            for (i64 i = 0; i < kv; i++) s += qr[static_cast<size_t>(i)] * kr[static_cast<size_t>(i)];
                            s *= d.scale;
                            p[static_cast<size_t>(t - t_lo)] = s;
                            if (s > mx) mx = s;
                        }
                        f32 sum = 0.0f;
                        if (d.sinks) sum = std::exp(d.sinks[h] - mx);
                        for (i64 t = 0; t < nk; t++) {
                            const f32 ex = std::exp(p[static_cast<size_t>(t)] - mx);
                            p[static_cast<size_t>(t)] = ex;
                            sum += ex;
                        }
                        f32 *out = af + r * (nh * kv) + h * kv;
                        for (i64 i = 0; i < kv; i++) {
                            f32 acc = 0.0f;
                            for (i64 t = 0; t < nk; t++)
                                acc += p[static_cast<size_t>(t)] /
                                       sum * cf[(t_lo + t) * d.pos_stride + i];
                            out[i] = acc;
                        }
                    }
                });
    }

    // The compressed variant: the same score/softmax/accumulate, over TWO row
    // families. Raw rows are windowed and roped at their own position; pooled
    // rows are always visible and roped at the block's first token with the
    // compression table. Both contribute their UNROTATED row as the value,
    // exactly as the dense path does -- the pooled row is already the normed
    // pool, so the only difference on the value side is where it came from.
    //
    // The two tables are passed rather than looked up: the compression base is
    // a per-FILE property (attention.compress_rope_freq_base) and the attention
    // base is per-layer, so a kernel that used one table for both would be
    // right on neither.
    bool mla_shared_compressed_supported() const override { return true; }
    void mla_shared_attention_compressed(void *a, const void *q,
                                         const void *cache, const void *pooled,
                                         const i64 *pooled_pos, i64 n_pooled,
                                         const i32 *sel, i64 n_sel,
                                         const MlaSharedDesc &d,
                                         const f32 *compress_inv_freq) override {
        npu_flush_();
        f32 *af = static_cast<f32 *>(a);
        const f32 *qf = static_cast<const f32 *>(q);
        const f32 *cf = static_cast<const f32 *>(cache);
        const f32 *pf = static_cast<const f32 *>(pooled);
        const i64 nh = d.n_head, kv = d.kv_lora, rope = d.qk_rope;
        const i64 base = kv - rope;
        const i64 nsel = sel ? n_sel : n_pooled;
        const i64 units = d.n_tok * nh;
        par_for(units, 1,
                8.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * kv,
                [&](i64 b, i64 e) {
                    std::vector<f32> qr(static_cast<size_t>(kv));
                    std::vector<f32> kr(static_cast<size_t>(kv));
                    std::vector<f32> p;
                    for (i64 u = b; u < e; u++) {
                        const i64 r = u / nh;
                        const i64 h = u - r * nh;
                        const i64 pos = d.pos0 + r;
                        const f32 *qrow = qf + r * (nh * kv) + h * kv;
                        for (i64 i = 0; i < kv; i++) qr[static_cast<size_t>(i)] = qrow[i];
                        for (i64 i = 0; i + 1 < rope; i += 2) {
                            const f32 th =
                                static_cast<f32>(pos) * d.inv_freq[i / 2];
                            const f32 cs = std::cos(th), sn = std::sin(th);
                            const f32 x0 = qr[static_cast<size_t>(base + i)];
                            const f32 x1 = qr[static_cast<size_t>(base + i + 1)];
                            qr[static_cast<size_t>(base + i)] = x0 * cs - x1 * sn;
                            qr[static_cast<size_t>(base + i + 1)] = x0 * sn + x1 * cs;
                        }
                        const i64 t_lo =
                            d.window > 0 ? std::max<i64>(0, pos - d.window + 1) : 0;
                        const i64 nk = pos - t_lo + 1;
                        const i64 nkeys = nk + nsel;
                        p.assign(static_cast<size_t>(nkeys), 0.0f);
                        f32 mx = d.sinks ? d.sinks[h] : -1e30f;
                        auto score_row = [&](const f32 *c, i64 keypos,
                                             const f32 *freq) {
                            for (i64 i = 0; i < kv; i++)
                                kr[static_cast<size_t>(i)] = c[i];
                            for (i64 i = 0; i + 1 < rope; i += 2) {
                                const f32 th = static_cast<f32>(keypos) * freq[i / 2];
                                const f32 cs = std::cos(th), sn = std::sin(th);
                                const f32 x0 = kr[static_cast<size_t>(base + i)];
                                const f32 x1 = kr[static_cast<size_t>(base + i + 1)];
                                kr[static_cast<size_t>(base + i)] = x0 * cs - x1 * sn;
                                kr[static_cast<size_t>(base + i + 1)] = x0 * sn + x1 * cs;
                            }
                            f32 s = 0.0f;
                            for (i64 i = 0; i < kv; i++)
                                s += qr[static_cast<size_t>(i)] * kr[static_cast<size_t>(i)];
                            return s * d.scale;
                        };
                        for (i64 t = t_lo; t <= pos; t++) {
                            const f32 s = score_row(cf + t * d.pos_stride, t,
                                                    d.inv_freq);
                            p[static_cast<size_t>(t - t_lo)] = s;
                            if (s > mx) mx = s;
                        }
                        for (i64 j = 0; j < nsel; j++) {
                            const i64 blk = sel ? sel[j] : j;
                            const f32 s = score_row(pf + blk * kv,
                                                    pooled_pos[blk],
                                                    compress_inv_freq);
                            p[static_cast<size_t>(nk + j)] = s;
                            if (s > mx) mx = s;
                        }
                        f32 sum = 0.0f;
                        if (d.sinks) sum = std::exp(d.sinks[h] - mx);
                        for (i64 t = 0; t < nkeys; t++) {
                            const f32 ex = std::exp(p[static_cast<size_t>(t)] - mx);
                            p[static_cast<size_t>(t)] = ex;
                            sum += ex;
                        }
                        f32 *out = af + r * (nh * kv) + h * kv;
                        for (i64 i = 0; i < kv; i++) {
                            f32 acc = 0.0f;
                            for (i64 t = 0; t < nk; t++)
                                acc += p[static_cast<size_t>(t)] / sum *
                                       cf[(t_lo + t) * d.pos_stride + i];
                            for (i64 j = 0; j < nsel; j++) {
                                const i64 blk = sel ? sel[j] : j;
                                acc += p[static_cast<size_t>(nk + j)] / sum *
                                       pf[blk * kv + i];
                            }
                            out[i] = acc;
                        }
                    }
                });
    }

    void mla_shared_output(void *out, const void *a, void *scratch,
                           const void *wo_a, const void *wo_b,
                           const MlaSharedDesc &d) override {
        npu_flush_();
        const f32 *af = static_cast<const f32 *>(a);
        f32 *y = static_cast<f32 *>(scratch);
        const u16 *wa = static_cast<const u16 *>(wo_a);
        const u16 *wb = static_cast<const u16 *>(wo_b);
        const i64 hpg = d.heads_per_group(), kv = d.kv_lora;
        const i64 gin = hpg * kv, gout = d.gout();
        par_for(d.n_tok * d.o_groups, 1,
                2.0 * static_cast<f64>(d.n_tok) * gout * gin,
                [&](i64 b, i64 e) {
                    for (i64 u = b; u < e; u++) {
                        const i64 r = u / d.o_groups;
                        const i64 g = u - r * d.o_groups;
                        const f32 *src = af + r * (d.n_head * kv) + g * gin;
                        const u16 *w = wa + g * d.o_lora * gin;
                        f32 *dst = y + r * gout + g * d.o_lora;
                        for (i64 j = 0; j < d.o_lora; j++) {
                            f32 s = 0.0f;
                            for (i64 i = 0; i < gin; i++)
                                s += fp16_to_fp32(w[j * gin + i]) * src[i];
                            dst[j] = s;
                        }
                    }
                });
        par_for(d.n_tok, 1, 2.0 * static_cast<f64>(d.n_tok) * d.n_embd * gout,
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++) {
                        const f32 *src = y + r * gout;
                        f32 *dst = static_cast<f32 *>(out) + r * d.n_embd;
                        for (i64 o = 0; o < d.n_embd; o++) {
                            f32 s = 0.0f;
                            for (i64 j = 0; j < gout; j++)
                                s += fp16_to_fp32(wb[o * gout + j]) * src[j];
                            dst[o] = s;
                        }
                    }
                });
    }

    void silu_mul(void *out, const void *gate, const void *up, i64 n,
                  i64 row_len = 0, f32 *row_scale = nullptr) override {
        KRK_KTIME("silu_mul");
        // The device's SwiGLU, which is this formula with its own exp: measured
        // 8.4e-05 relative against the exact silu. That is the whole accuracy
        // story of the graph arm in one place -- see npu_offload.hpp.
        if (npu_ && npu::swiglu(npu_, out, gate, up, n)) {
            // f32 activations need no exponent restore; the contract still asks
            // for one scale per row and 1.0 is that scale.
            if (row_scale && row_len > 0 && n % row_len == 0)
                for (i64 r = 0; r < n / row_len; r++) row_scale[r] = 1.0f;
            return;
        }
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        const f32 *g = static_cast<const f32 *>(gate);
        const f32 *u = static_cast<const f32 *>(up);
#if defined(KRK_DOT_AVX2)
        if (vec_dot_wide() && cpu_has_avx2()) {
            // Grain 4096 rather than 8192: these blocks are now cheap enough
            // that 8192 leaves one thread holding most of a 4096-wide row.
            par_for(n, 2048, 2.0 * static_cast<f64>(n),
                    [&](i64 b, i64 e) { silu_mul_avx2(o + b, g + b, u + b, e - b); });
        } else
#endif
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const f32 x = g[i];
                        o[i] = (x / (1.0f + std::exp(-x))) * u[i];
                    }
                });
        // f32 activations already have the exponent range the row scale exists
        // to restore, so nothing is rescaled here; the caller still gets the
        // 1.0 per row its contract expects.
        if (row_scale && row_len > 0 && n % row_len == 0)
            for (i64 r = 0; r < n / row_len; r++) row_scale[r] = 1.0f;
    }

    // gpt-oss's FFN activation, the same formula as the device kernel and as
    // ggml_compute_forward_swiglu_oai_f32: one-sided clamp on the gate,
    // two-sided on the up operand, 1.702 in the sigmoid, (y + 1) as the
    // multiplier. Written out rather than folded so the two arms are
    // comparable line by line.
    void swiglu_oai(void *out, const void *gate, const void *up, i64 n) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        const f32 *g = static_cast<const f32 *>(gate);
        const f32 *u = static_cast<const f32 *>(up);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const f32 x = std::fmin(g[i], 7.0f);
                        const f32 y = std::fmin(std::fmax(u[i], -7.0f), 7.0f);
                        o[i] = (x / (1.0f + std::exp(-1.702f * x))) * (y + 1.0f);
                    }
                });
    }

    void add_bias_rows(void *out, const f32 *bias, i64 n, i64 rows) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        par_for(rows, 1, 2.0 * static_cast<f64>(rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++)
                        for (i64 i = 0; i < n; i++) o[r * n + i] += bias[i];
                });
    }

    void add_inplace(void *a, const void *b, i64 n) override {
        if (npu_ && npu::add(npu_, a, b, n)) return;
        npu_flush_();
        f32 *af = static_cast<f32 *>(a);
        const f32 *bf = static_cast<const f32 *>(b);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 lo, i64 hi) {
                    for (i64 i = lo; i < hi; i++) af[i] += bf[i];
                });
    }

    void copy_act(void *dst, const void *src, i64 n) override {
        npu_flush_();
        std::memcpy(dst, src, static_cast<size_t>(n) * 4);
    }

    void axpy(void *dst, const void *src, f32 alpha, i64 n) override {
        if (npu_ && npu::add_scaled(npu_, dst, src, alpha, n)) return;
        npu_flush_();
        f32 *d = static_cast<f32 *>(dst);
        const f32 *s = static_cast<const f32 *>(src);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    if (alpha == 1.0f) {
                        for (i64 i = b; i < e; i++) d[i] += s[i];
                    } else {
                        for (i64 i = b; i < e; i++) d[i] += alpha * s[i];
                    }
                });
    }

    // The reference backend shares one address space, so "staging" is a copy.
    void upload_i32(i32 *dst, const i32 *src, i64 n) override {
        npu_flush_();
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(i32));
    }

    // ---- prism.hadamard activation transform (Backend::hadamard_act) ----
    //
    // The reference definition, and deliberately the plain one: per row, an
    // optional grouped-V perm, an optional +-1 sign multiply, then a
    // normalized Sylvester FWHT over consecutive blocks of `block`. The FWHT
    // is the textbook butterfly (H[ i ][ j ] = (-1)^popcount(i & j)), scaled
    // by 1/sqrt(block) at the end so H . H = I -- which is what makes the
    // same routine both the fold before a rotated matmul and the inverse
    // after a rotated embedding lookup; only the sign order differs.
    // Everything stays in f32, exactly like the fork's graph tensors.
    void hadamard_act(void *out, const void *in, i64 rows,
                      const Backend::HadDesc &d) override {
        npu_flush_();
        if (rows <= 0 || d.n <= 0) return;
        const i64 n = d.n;
        const i64 block = d.block;
        if (block <= 0 || (block & (block - 1)) != 0 || n % block != 0) {
            KRK_ERROR("hadamard_act: width %lld is not a whole number of "
                      "%lld-element blocks", (long long)n, (long long)block);
            std::abort();
        }
        const bool perm = d.perm_hd > 0;
        const f32 *signs = d.signs;
        const f32 scale = 1.0f / std::sqrt(static_cast<f32>(block));
        f32 *o = static_cast<f32 *>(out);
        const f32 *x = static_cast<const f32 *>(in);
        // A row is transformed into a scratch row before anything is written
        // back, so out == in is safe: the whole input row is read first.
        par_for(rows, 1, 8.0 * static_cast<f64>(rows) * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    std::vector<f32> cur(static_cast<size_t>(n));
                    std::vector<f32> tmp(static_cast<size_t>(n));
                    for (i64 r = b; r < e; r++) {
                        f32 *a = cur.data();
                        const f32 *src = x + r * n;
                        for (i64 i = 0; i < n; i++) a[i] = src[i];
                        if (perm) {
                            const i64 H = d.perm_hd;
                            const i64 Nk = d.perm_nk;
                            const i64 R = d.perm_rep;
                            f32 *b2 = tmp.data();
                            for (i64 j = 0; j < R; j++)
                                for (i64 k = 0; k < Nk; k++)
                                    for (i64 i = 0; i < H; i++)
                                        b2[i + j * H + k * H * R] =
                                            a[i + k * H + j * H * Nk];
                            a = b2;
                        }
                        if (d.signs_first && signs)
                            for (i64 i = 0; i < n; i++) a[i] *= signs[i];
                        for (i64 off = 0; off < n; off += block) {
                            f32 *y = a + off;
                            for (i64 len = 1; len < block; len <<= 1)
                                for (i64 g = 0; g < block; g += 2 * len)
                                    for (i64 j = 0; j < len; j++) {
                                        const f32 u = y[g + j];
                                        const f32 v = y[g + len + j];
                                        y[g + j] = u + v;
                                        y[g + len + j] = u - v;
                                    }
                            for (i64 i = 0; i < block; i++) y[i] *= scale;
                        }
                        if (!d.signs_first && signs)
                            for (i64 i = 0; i < n; i++) a[i] *= signs[i];
                        f32 *dst = o + r * n;
                        for (i64 i = 0; i < n; i++) dst[i] = a[i];
                    }
                });
    }

    // ---- gated delta net (the reference oracle) -------------------------
    bool gdn_supported() const override { return true; }

    void l2norm(void *o, const void *x, i64 n_tok, i64 n_head, i64 hd, i64 stride,
                f32 eps) override {
        npu_flush_();
        const f32 *s = static_cast<const f32 *>(x);
        f32 *d = static_cast<f32 *>(o);
        // Each head carries its own reduction, so (token, head) pairs are the
        // units. Safe in place: the reduction finishes before the write.
        par_for(n_tok, 1, 3.0 * static_cast<f64>(n_tok) * n_head * hd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        for (i64 h = 0; h < n_head; h++) {
                            const f32 *sh = s + t * stride + h * hd;
                            f32 *dh = d + t * stride + h * hd;
                            f32 ss = 0;
                            for (i64 i = 0; i < hd; i++) ss += sh[i] * sh[i];
                            const f32 inv = 1.0f / std::sqrt(ss + eps);
                            for (i64 i = 0; i < hd; i++) dh[i] = sh[i] * inv;
                        }
                    }
                });
    }

    // ---- hyper-connections (qwen4exp) ------------------------------------
    //
    // The CPU backend is the reference arm for the qwen4exp stack, so these
    // delegate to the host maths in hc.cpp rather than repeating it: the suite
    // pins hc.cpp's arithmetic, and a second copy here would be a second thing
    // to keep in step.
    bool hc_supported() const override { return true; }
    // V4.1's Sinkhorn mixer: the host functions in src/hc.cpp ARE the
    // reference, so the CPU backend is a thin passthrough. f32 activations
    // only -- a backend whose lanes are f16 has to convert, and the pointer
    // checks here say so rather than reinterpreting f16 bytes as f32 (the
    // exact bug hc_combine's comment records).
    bool hc_sinkhorn_supported() const override { return true; }
    void hc_sk_split(void *pre, void *post, void *comb, const void *mixes,
                     const f32 *scale, const f32 *base, i64 n_tok, i64 hc,
                     i64 iters, f32 eps) override {
        HcSinkhornGeometry g;
        g.n_embd = 1;   // the split does not need the embedding width
        g.hc = static_cast<i32>(hc);
        g.sinkhorn_iters = static_cast<i32>(iters);
        g.eps = eps;
        hc_sinkhorn_split(static_cast<f32 *>(pre), static_cast<f32 *>(post),
                          static_cast<f32 *>(comb),
                          static_cast<const f32 *>(mixes), scale, base,
                          static_cast<i32>(n_tok), g);
    }
    void hc_sk_pre(void *out, const void *lanes, const void *pre, i64 n_tok,
                   i64 hc, i64 ne) override {
        HcSinkhornGeometry g;
        g.n_embd = static_cast<i32>(ne);
        g.hc = static_cast<i32>(hc);
        g.sinkhorn_iters = 1;
        g.eps = 1e-6f;
        hc_sinkhorn_pre(static_cast<f32 *>(out), static_cast<const f32 *>(lanes),
                        static_cast<const f32 *>(pre), static_cast<i32>(n_tok), g);
    }
    void hc_sk_post(void *lanes_out, const void *lanes_in, const void *block_out,
                    const void *post, const void *comb, i64 n_tok, i64 hc,
                    i64 ne) override {
        HcSinkhornGeometry g;
        g.n_embd = static_cast<i32>(ne);
        g.hc = static_cast<i32>(hc);
        g.sinkhorn_iters = 1;
        g.eps = 1e-6f;
        hc_sinkhorn_post(static_cast<f32 *>(lanes_out),
                         static_cast<const f32 *>(lanes_in),
                         static_cast<const f32 *>(block_out),
                         static_cast<const f32 *>(post), static_cast<const f32 *>(comb),
                         static_cast<i32>(n_tok), g);
    }

    void hc_init(void *lanes, const void *x, i64 n_tok, i64 hc, i64 ne) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(lanes);
        const f32 *s = static_cast<const f32 *>(x);
        if (!o || !s || n_tok <= 0 || hc <= 0 || ne <= 0) return;
        // One thread per embedding element, writing its hc copies: the lanes of
        // one token are a contiguous block, so this is hc stores per element
        // and no second pass.
        par_for(n_tok * ne, 4096, static_cast<f64>(n_tok * ne) * hc,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const i64 t = i / ne;
                        const i64 r = i % ne;
                        const f32 v = s[i];
                        for (i64 c = 0; c < hc; c++) o[(t * hc + c) * ne + r] = v;
                    }
                });
    }

    void hc_mix(void *mixed, const void *xn, const void *gate, i64 n_tok,
                i64 hc, i64 ne) override {
        npu_flush_();
        HcGeometry g;
        g.n_embd = static_cast<i32>(ne);
        g.hc = static_cast<i32>(hc);
        g.low_rank = 1; // not a mixer geometry; only the lane count is read
        // Qualified: the member name is the op, the free function is the maths.
        krk::hc_mix(static_cast<f32 *>(mixed), static_cast<const f32 *>(xn),
                    static_cast<const f32 *>(gate), static_cast<i32>(n_tok), g);
    }

    void hc_combine(void *res, const void *block_out, const void *inject,
                    i64 n_tok, i64 hc, i64 ne) override {
        npu_flush_();
        HcGeometry g;
        g.n_embd = static_cast<i32>(ne);
        g.hc = static_cast<i32>(hc);
        g.low_rank = 1;
        krk::hc_combine(static_cast<f32 *>(res),
                        static_cast<const f32 *>(block_out),
                        static_cast<const f32 *>(inject), static_cast<i32>(n_tok), g);
    }

    // ---- per-layer embedding (PLE), qwen4exp ------------------------------
    void ple_gate(void *out, const void *key, const void *qry, i64 n_tok,
                  i64 hc, i64 ne) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        const f32 *kf = static_cast<const f32 *>(key);
        const f32 *qf = static_cast<const f32 *>(qry);
        if (!o || !kf || !qf || n_tok <= 0 || hc <= 0 || ne <= 0) return;
        const f32 inv = 1.0f / std::sqrt(static_cast<f32>(ne));
        par_for(n_tok * hc, 1, 4.0 * static_cast<f64>(n_tok * hc) * ne,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const i64 t = i / hc;
                        const i64 c = i % hc;
                        const i64 base = (t * hc + c) * ne;
                        f32 s = 0.0f;
                        for (i64 j = 0; j < ne; j++) s += kf[base + j] * qf[base + j];
                        s *= inv;
                        const f32 mag =
                            std::sqrt(std::fmax(std::fabs(s), 1e-6f));
                        o[c * n_tok + t] =
                            1.0f / (1.0f + std::exp(s < 0.0f ? mag : -mag));
                    }
                });
    }

    // out[t,c,e] = value[t,e] * g[c*n_tok + t]: one scalar per LANE broadcast
    // across that lane's whole row. The CPU body is the reference the device
    // kernel is checked against, and it has to exist: the base class body is a
    // no-op, so a missing implementation leaves the destination holding
    // whatever the allocator last put there -- measured as a test that passed
    // single-threaded and failed 1 run in 3 with the pool live, because only
    // then is the heap reused.
    void lane_bcast_mul(void *out, const void *value, const void *g, i64 n_tok,
                        i64 hc, i64 ne) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        const f32 *vf = static_cast<const f32 *>(value);
        const f32 *gf = static_cast<const f32 *>(g);
        if (!o || !vf || !gf || n_tok <= 0 || hc <= 0 || ne <= 0) return;
        par_for(n_tok * hc, 1, 2.0 * static_cast<f64>(n_tok * hc) * ne,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const i64 t = i / hc;
                        const i64 c = i % hc;
                        const f32 *src = vf + t * ne;
                        f32 *dst = o + i * ne;
                        const f32 w = gf[c * n_tok + t];
                        for (i64 j = 0; j < ne; j++) dst[j] = src[j] * w;
                    }
                });
    }

    void conv1d_dw(void *out, const void *x, const void *w, i64 n_tok, i64 ch,
                   i64 kern, i64 dil) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        const f32 *xf = static_cast<const f32 *>(x);
        const f32 *wf = static_cast<const f32 *>(w);
        if (!o || !xf || !wf || n_tok <= 0 || ch <= 0 || kern <= 0) return;
        par_for(n_tok * ch, 8192, 2.0 * static_cast<f64>(n_tok * ch) * kern,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const i64 t = i / ch;
                        const i64 c = i % ch;
                        f32 acc = 0.0f;
                        for (i64 k = 0; k < kern; k++)
                            acc += xf[(t + (kern - 1 - k) * dil) * ch + c] *
                                   wf[k * ch + c];
                        o[i] = acc;
                    }
                });
    }

    // LFM2.5 short convolution. Same arithmetic as the device op
    // (kernels/sc.hpp), including the ring: the reference backend is what the
    // device arm is diffed against on a shape the suite can build, so a
    // different history convention here would be a silent disagreement.
    void shortconv(void *out, const void *proj, void *state, const void *w,
                   i64 n_tok, i64 ch, i64 kern, i64 hist, i64 pos0) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        const f32 *p = static_cast<const f32 *>(proj);
        f32 *st = static_cast<f32 *>(state);
        const f32 *wf = static_cast<const f32 *>(w);
        if (!o || !p || !wf || n_tok <= 0 || ch <= 0 || kern <= 1 ||
            hist < kern - 1)
            return;
        par_for(n_tok * ch, 8192, 2.0 * static_cast<f64>(n_tok * ch) * kern,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const i64 t = i / ch;
                        const i64 c = i % ch;
                        f32 acc = 0.0f;
                        for (i64 k = 0; k < kern; k++) {
                            const i64 lag = kern - 1 - k;
                            const i64 pos = pos0 + t - lag;
                            if (pos < 0) continue;
                            // The conv's input is B*X, not X: the reference
                            // multiplies inside the gate and convolves the
                            // product.
                            const f32 bx = pos >= pos0
                                               ? p[(t - lag) * 3 * ch + c] *
                                                     p[(t - lag) * 3 * ch + 2 * ch + c]
                                               : st[(pos % hist) * ch + c];
                            acc += bx * wf[k * ch + c];
                        }
                        o[i] = p[t * 3 * ch + ch + c] * acc;
                    }
                });
        if (st) {
            // One row per position, into its own ring slot -- the ring is at
            // least `kern-1` deep and may be deeper, so every row this call
            // produced is kept (a speculation rewind reads them back).
            for (i64 r = 0; r < n_tok; r++) {
                const i64 pos = pos0 + r;
                if (pos < 0) continue;
                const f32 *src = p + (pos - pos0) * 3 * ch;
                f32 *dst = st + (pos % hist) * ch;
                for (i64 c = 0; c < ch; c++)
                    dst[c] = src[c] * src[2 * ch + c];
            }
        }
    }

    void silu_act(void *x, i64 n) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 4.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const f32 v = p[i];
                        p[i] = v / (1.0f + std::exp(-v));
                    }
                });
    }

    void gelu_act(void *x, i64 n) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 6.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const f32 v = p[i];
                        p[i] = krk_gelu_tanh(v);
                    }
                });
    }

    void sigmoid_act(void *x, i64 n) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 4.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) p[i] = 1.0f / (1.0f + std::exp(-p[i]));
                });
    }

    void relu_sqr_act(void *x, i64 n) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 4.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const f32 v = p[i];
                        const f32 r = v > 0.0f ? v : 0.0f;
                        p[i] = r * r;
                    }
                });
    }

    void softplus_act(void *x, i64 n) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 4.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        // log1p(exp(x)) overflows well before the value matters;
                        // above the threshold softplus(x) == x to f32 accuracy.
                        const f32 v = p[i];
                        p[i] = v > 20.0f ? v : std::log1p(std::exp(v));
                    }
                });
    }

    void scale_act(void *x, f32 alpha, i64 n_tok, i64 width, i64 stride) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n_tok, 1, static_cast<f64>(n_tok) * width,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        f32 *row = p + t * stride;
                        for (i64 i = 0; i < width; i++) row[i] *= alpha;
                    }
                });
    }

    // OLMo-1's attention clamp: q, k and v are bounded after the projection and
    // before the rotary, so the bound is on the pre-rope rows. Elementwise and
    // branch-free; the CPU reference is the oracle for the device kernels.
    bool clamp_supported() const override { return true; }
    void clamp_act(void *x, f32 lo, f32 hi, i64 n) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 1, static_cast<f64>(n), [&](i64 b, i64 e) {
            for (i64 i = b; i < e; i++) {
                const f32 v = p[i];
                p[i] = v < lo ? lo : (v > hi ? hi : v);
            }
        });
    }

    void mul_slice(void *a, const void *b, i64 rows, i64 width, i64 stride,
                   i64 off) override {
        npu_flush_();
        f32 *pa = static_cast<f32 *>(a);
        const f32 *pb = static_cast<const f32 *>(b);
        // Rows are independent, so the split is over rows and the inner loop
        // stays a contiguous multiply -- the same shape mul_act has.
        par_for(rows, 1, 4.0 * static_cast<f64>(rows) * width, [&](i64 r0, i64 r1) {
            for (i64 r = r0; r < r1; r++) {
                f32 *ar = pa + r * width;
                const f32 *br = pb + r * stride + off;
                for (i64 i = 0; i < width; i++) ar[i] *= br[i];
            }
        });
    }

    void mul_act(void *a, const void *b, i64 n) override {
        if (npu_ && npu::mul(npu_, a, b, n)) return;
        npu_flush_();
        f32 *p = static_cast<f32 *>(a);
        const f32 *s = static_cast<const f32 *>(b);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 lo, i64 e) {
                    for (i64 i = lo; i < e; i++) p[i] *= s[i];
                });
    }

    void mul_head_broadcast(void *x, const void *g, i64 n_tok, i64 n_head,
                            i64 hd) override {
        npu_flush_();
        f32 *xf = static_cast<f32 *>(x);
        const f32 *gf = static_cast<const f32 *>(g);
        par_for(n_tok, 1, 2.0 * static_cast<f64>(n_tok) * n_head * hd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        for (i64 h = 0; h < n_head; h++) {
                            const f32 w = gf[t * n_head + h];
                            f32 *row = xf + (t * n_head + h) * hd;
                            for (i64 i = 0; i < hd; i++) row[i] *= w;
                        }
                    }
                });
    }

    void add_bias_cols(void *x, const f32 *bias, i64 n_row, i64 n_col) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n_row, 1, static_cast<f64>(n_row) * n_col,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *row = p + i * n_col;
                        for (i64 j = 0; j < n_col; j++) row[j] += bias[j];
                    }
                });
    }

    void scale_cols(void *x, const f32 *col, i64 n_row, i64 n_col) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        par_for(n_row, 1, static_cast<f64>(n_row) * n_col,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *row = p + i * n_col;
                        for (i64 j = 0; j < n_col; j++) row[j] *= col[j];
                    }
                });
    }

    void scale_rows(void *x, const void *alpha, i64 n_row, i64 n) override {
        npu_flush_();
        f32 *p = static_cast<f32 *>(x);
        const f32 *a = static_cast<const f32 *>(alpha);
        par_for(n_row, 1, 2.0 * static_cast<f64>(n_row) * n,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *row = p + i * n;
                        const f32 s = a[i];
                        for (i64 j = 0; j < n; j++) row[j] *= s;
                    }
                });
    }

    void qwen3_next_split(void *q, void *gate, const void *packed, i64 n_tok,
                          i64 n_head, i64 hd) override {
        npu_flush_();
        const f32 *p = static_cast<const f32 *>(packed);
        f32 *qd = static_cast<f32 *>(q);
        f32 *gd = static_cast<f32 *>(gate);
        auto one = [&](i64 t) {
            for (i64 h = 0; h < n_head; h++) {
                const f32 *src = p + (t * n_head + h) * 2 * hd;
                f32 *dst = qd + (t * n_head + h) * hd;
                std::memcpy(dst, src, static_cast<size_t>(hd) * sizeof(f32));
                if (gd)
                    std::memcpy(gd + (t * n_head + h) * hd, src + hd,
                                static_cast<size_t>(hd) * sizeof(f32));
            }
        };
        // Callers must not alias `packed` (see the contract in backend.hpp):
        // the row being written is half the width of the row being read, so a
        // token's output lands inside another token's input. The scalar
        // backend can still be exact where the device cannot — a token's
        // writes only ever hit the sources of the tokens before it, so
        // walking tokens in order is safe — and it is the reference, so it
        // should not be the one that quietly races.
        if (qd == p) {
            for (i64 t = 0; t < n_tok; t++) one(t);
            return;
        }
        par_for(n_tok, 1, 2.0 * static_cast<f64>(n_tok) * n_head * hd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) one(t);
                });
    }

    void conv1d_silu(void *out, const void *in, void *state, const void *kern,
                     DType wt, i64 n_tok, i64 chan, i64 ksize,
                     const void *bias = nullptr) override {
        npu_flush_();
        if (ksize <= 0 || ksize > kMaxConvKernel) return;
        const f32 *xp = static_cast<const f32 *>(in);
        f32 *st = static_cast<f32 *>(state);
        f32 *o = static_cast<f32 *>(out);
        const i64 keep = ksize - 1;
        // One tap value per (tap, channel), dequantized once per call — the
        // kernel is tiny and every channel reads all of it.
        const i64 nk = ksize * chan;
        std::vector<f32> w(static_cast<size_t>(nk));
        if (wt == DType::F32) {
            std::memcpy(w.data(), kern, static_cast<size_t>(nk) * sizeof(f32));
        } else {
            dequant_row(wt, kern, w.data(), nk);
        }
        // Channels are independent down the whole time axis, and each worker
        // touches only its own column of `state`, so the pool can split here.
        par_for(chan, 32, 4.0 * static_cast<f64>(n_tok) * chan * ksize,
                [&](i64 cb, i64 ce) {
                    for (i64 c = cb; c < ce; c++) {
                        // The ksize-1 stored steps, then this call's rows, held
                        // in registers rather than re-read: the engine convolves
                        // IN PLACE (out == in), and the receptive field of row t
                        // reaches back ksize-1 rows that earlier outputs have
                        // already overwritten. A window that slides forward is
                        // the only way to read those rows before they die.
                        f32 win[kMaxConvKernel];
                        for (i64 i = 1; i < ksize; i++)
                            win[i] = st[(ksize - 1 - i) * chan + c];
                        win[0] = xp[c];
                        const f32 b = bias ? static_cast<const f32 *>(bias)[c] : 0.0f;
                        for (i64 t = 0; t < n_tok; t++) {
                            f32 acc = b;
                            for (i64 j = 0; j < ksize; j++)
                                acc += w[static_cast<size_t>(j * chan + c)] * win[keep - j];
                            o[t * chan + c] = silu_scalar(acc);
                            for (i64 i = ksize - 1; i > 0; i--) win[i] = win[i - 1];
                            if (t + 1 < n_tok) win[0] = xp[(t + 1) * chan + c];
                        }
                        // The window now spans X(n_tok + keep - i), so these are
                        // the rows the next call continues from.
                        for (i64 j = 0; j < keep; j++)
                            st[j * chan + c] = win[keep - j];
                    }
                });
    }

    // The scalar Mamba-2 SSD scan — the reference oracle the device kernel is
    // checked against. One thread-per-slot recurrence per block-methodology
    // claim in the header above: slot (h, p, s) owns an f32 cell of `state`
    // (layout [head][p][s]), walks the token loop in order, and writes nothing
    // but out[t, pair] — so the token sequence is the only sequential part.
    //
    // Inputs are regions of one fused post-conv post-SiLU row of conv_dim
    // elements per token; a/b/dp are per-head vectors; `out` is dense
    // [n_tok, inner]. The GDN path keeps f32 state at higher precision than
    // its f16 activations; this path keeps it too, for the same reason.
    void ssd_scan(void *out, void *state, const void *xbc_v, const void *dt_v,
                   const void *a_v, const void *db_v, const void *dp_v,
                   i64 n_tok, i64 n_head, i64 head_dim, i64 d_state,
                   i64 n_group, i64 inner, i64 conv_dim, bool has_d) override {
        npu_flush_();
        f32 *o = static_cast<f32 *>(out);
        f32 *S = static_cast<f32 *>(state);
        const f32 *xbc = static_cast<const f32 *>(xbc_v);
        const f32 *dtv = static_cast<const f32 *>(dt_v);
        const f32 *a = static_cast<const f32 *>(a_v);
        const f32 *db = static_cast<const f32 *>(db_v);
        const f32 *dp = has_d ? static_cast<const f32 *>(dp_v) : nullptr;

        // dt = softplus(dt_raw + db[h]), the step size. The recurrence is
        //   h' = exp(a[h]*dt) * h + dt * u * B[g,s]
        // -- the DECAY takes A, the input term does NOT. ggml's reference:
        //   dA = expf(dt_soft_plus * A[h]);
        //   x_dt = x[ii] * dt_soft_plus;
        //   s    = s*dA + B*x_dt;
        // Both backends used to scale the input term by a[h]*dt as well, which
        // is invisible to a device-vs-CPU test because they shared it.
        auto dt_step = [&](i64 t, i64 h) {
            const f32 x = dtv[t * n_head + h] + db[h];
            // Same tail behaviour as the device d_softplus: above 20 it is v
            // to f32, below -20 it is exp(v), else the real logarithm.
            return x > 20.0f ? x : (x < -20.0f ? std::exp(x)
                                               : std::log1p(std::exp(x)));
        };

        // Per-slot recurrences. Slots partition over (h, p): one state cell +
        // its own output accumulator, so the pool can split over heads with
        // whole-head ownership. The token loop inside a head is sequential.
        par_for(n_head, 1, 8.0 * static_cast<f64>(n_tok) * n_head * head_dim *
                               d_state,
                [&](i64 hb, i64 he) {
                    std::vector<f32> y(n_tok);
                    for (i64 h = hb; h < he; h++) {
                        f32 *Sh = S + static_cast<size_t>(h) * head_dim * d_state;
                        const i64 g = h * n_group / n_head; // grouped state
                        for (i64 t = 0; t < n_tok; t++) {
                            const f32 dtv_ = dt_step(t, h);
                            const f32 decay = std::exp(a[h] * dtv_);
                            const f32 *Bt = xbc + t * conv_dim + inner + g * d_state;
                            const f32 *Ct = Bt + n_group * d_state;
                            f32 *ot = o + (t * n_head + h) * head_dim;
                            for (i64 p = 0; p < head_dim; p++) {
                                const f32 up = xbc[t * conv_dim + h * head_dim + p];
                                f32 *Sp = Sh + p * d_state;
                                f32 acc = 0;
                                for (i64 s = 0; s < d_state; s++) {
                                    f32 cell = Sp[s] * decay + dtv_ * up * Bt[s];
                                    Sp[s] = cell;
                                    acc += cell * Ct[s];
                                }
                                // D is per HEAD and broadcast over the
                                // head's channels: the reference adds
                                // ggml_mul(x, ssm_d) with x [head_dim, n_head]
                                // and ssm_d [1, n_head], so the head index
                                // selects the scalar. n_head == head_dim on the
                                // 30B file, which is exactly why a per-position
                                // read here survives every test that uses that
                                // one shape.
                                ot[p] = acc + (dp ? dp[h] * up : 0.0f);
                            }
                        }
                    }
                });
    }

    void delta_rule(void *out, void *state, const void *q, const void *k,
                    const void *v, const void *g, const void *beta, i64 n_tok,
                    i64 n_k_head, i64 n_v_head, i64 d_state, i64 hd,
                    i64 row_stride) override {
        npu_flush_();
        f32 *S = static_cast<f32 *>(state);
        const f32 *qp = static_cast<const f32 *>(q);
        const f32 *kp = static_cast<const f32 *>(k);
        const f32 *vp = static_cast<const f32 *>(v);
        const f32 *gp = static_cast<const f32 *>(g);
        const f32 *bp = static_cast<const f32 *>(beta);
        f32 *op = static_cast<f32 *>(out);
        // Value heads own disjoint state, so they are the parallel units; the
        // token loop inside one head is sequential by definition.
        par_for(n_v_head, 1, 8.0 * static_cast<f64>(n_tok) * n_v_head * d_state * hd,
                [&](i64 hb, i64 he) {
                    std::vector<f32> kv(static_cast<size_t>(hd));
                    for (i64 h = hb; h < he; h++) {
                        f32 *Sh = S + h * d_state * hd;
                        const i64 kh = h % n_k_head; // tiled key head
                        for (i64 t = 0; t < n_tok; t++) {
                            const f32 decay = std::exp(gp[t * n_v_head + h]);
                            const f32 bta = bp[t * n_v_head + h];
                            const f32 *qt = qp + t * row_stride + kh * d_state;
                            const f32 *kt = kp + t * row_stride + kh * d_state;
                            const f32 *vt = vp + t * row_stride + h * hd;

                            // S <- S * exp(g)
                            for (i64 i = 0; i < d_state; i++) {
                                f32 *row = Sh + i * hd;
                                for (i64 j = 0; j < hd; j++) row[j] *= decay;
                            }
                            // kv = S^T k  (kv[j] = sum_i S[i, j] k[i])
                            for (i64 j = 0; j < hd; j++) {
                                f32 acc = 0;
                                for (i64 i = 0; i < d_state; i++)
                                    acc += Sh[i * hd + j] * kt[i];
                                kv[static_cast<size_t>(j)] = acc;
                            }
                            // S += k (x) d, d = (v - kv) * beta
                            for (i64 i = 0; i < d_state; i++) {
                                f32 *row = Sh + i * hd;
                                const f32 ki = kt[i];
                                for (i64 j = 0; j < hd; j++)
                                    row[j] += ki * (vt[j] - kv[static_cast<size_t>(j)]) * bta;
                            }
                            // out = S^T q
                            f32 *ot = op + (t * n_v_head + h) * hd;
                            for (i64 j = 0; j < hd; j++) {
                                f32 acc = 0;
                                for (i64 i = 0; i < d_state; i++)
                                    acc += Sh[i * hd + j] * qt[i];
                                ot[j] = acc;
                            }
                        }
                    }
                });
    }

    void gather_rows(void *dst, const void *src, const i32 *rows, i64 n_rows,
                     i64 n) override {
        const f32 *s = static_cast<const f32 *>(src);
        f32 *d = static_cast<f32 *>(dst);
        par_for(n_rows, 1, 2.0 * static_cast<f64>(n_rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const i64 r = rows[i];
                        std::memcpy(d + i * n, s + r * n,
                                    static_cast<size_t>(n) * sizeof(f32));
                    }
                });
    }

    void scatter_axpy_rows(void *dst, const void *src, const i32 *rows,
                           const f32 *alpha, i64 n_rows, i64 n,
                           const f32 *row_scale = nullptr,
                           i32 *row_exp = nullptr) override {
        npu_flush_();
        // The scalar reference works in f32, so no row here ever leaves a
        // representable range and the storage exponent stays 0: the
        // contract is answered by leaving the caller's slot alone.
        (void)row_exp;
        f32 *d = static_cast<f32 *>(dst);
        const f32 *s = static_cast<const f32 *>(src);
        // Each row writes a distinct destination slice.
        par_for(n_rows, 1, 2.0 * static_cast<f64>(n_rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *dr = d + static_cast<i64>(rows[i]) * n;
                        const f32 *sr = s + i * n;
                        const f32 a = alpha[i] * (row_scale ? row_scale[i] : 1.0f);
                        if (a == 1.0f) {
                            for (i64 j = 0; j < n; j++) dr[j] += sr[j];
                        } else {
                            for (i64 j = 0; j < n; j++) dr[j] += a * sr[j];
                        }
                    }
                });
    }

private:
    // RoPE in the interleaved (original LLaMA / llama.cpp) convention: pair
    // channels (2i, 2i+1). GGUF q/k weights for Llama-family models are
    // stored in this layout (the HF exporter permutes the half-split rows),
    // so pairing adjacent channels reproduces HF's rotate_half semantics.
    // `frac` limits rotation to a prefix of the hd/2 pairs (Phi-2/Gemma
    // partial RoPE); `scale` divides the position (linear scaling). `neox`
    // switches to the half-split convention the NeoX family is exported with:
    // pair i is channel i with channel i + rot, where rot is the number of
    // rotated pairs. Both conventions leave every other channel untouched, so
    // the only difference is *which* channels a pair is made of — and since
    // only the relative rotation between q and k reaches the score, that is
    // the difference between the reference answer and a plausible wrong one.
    // `row_pos` (optional) is the per-row position table: row t rotates at
    // row_pos[t] instead of pos0 + t, which is the only difference between a
    // single sequence's prefill and a batch of rows from different sequences.
    static void rope_apply(f32 *x, i64 n_heads, i64 hd, i64 n_tok, i64 pos0,
                           const f32 *inv_freq, f32 scale, f32 frac, bool neox,
                           const i64 *row_pos = nullptr) {
        const i64 half = hd / 2;
        const i64 rot = std::max<i64>(1, static_cast<i64>(static_cast<f32>(half) * frac));
        const f32 s = scale > 0 ? scale : 1.0f;
        // (token, head) pairs write disjoint lanes.
        const i64 units = n_tok * n_heads;
        par_for(units, 1, 6.0 * static_cast<f64>(units) * rot,
                [&](i64 b, i64 e) {
                    for (i64 u = b; u < e; u++) {
                        const i64 t = u / n_heads;
                        const i64 h = u - t * n_heads;
                        const f32 p = static_cast<f32>(row_pos ? row_pos[t]
                                                              : pos0 + t) / s;
                        f32 *hx = x + (t * n_heads + h) * hd;
                        for (i64 i = 0; i < rot && i < half; i++) {
                            const f32 theta = p * inv_freq[i];
                            const f32 c = std::cos(theta), sn = std::sin(theta);
                            const i64 ia = neox ? i : 2 * i;
                            const i64 ib = neox ? i + rot : 2 * i + 1;
                            const f32 a = hx[ia], bb = hx[ib];
                            hx[ia] = a * c - bb * sn;
                            hx[ib] = a * sn + bb * c;
                        }
                    }
                });
    }

    // OLMoE's convention: one RMS over the whole projected row (n = n_head *
    // hd), the weight spanning that row. One row per token, so the unit of
    // parallelism is the token, not the (token, head) pair.
    static void row_norm(f32 *x, const f32 *w, i64 n_rows, i64 n, f32 eps) {
        par_for(n_rows, 1, 4.0 * static_cast<f64>(n_rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++) {
                        f32 *xr = x + r * n;
                        f32 ss = 0;
                        for (i64 i = 0; i < n; i++) ss += xr[i] * xr[i];
                        const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(n) + eps);
                        for (i64 i = 0; i < n; i++) xr[i] = xr[i] * inv * w[i];
                    }
                });
    }

    static void head_norm(f32 *x, const f32 *w, i64 n_heads, i64 hd, i64 n_tok,
                          f32 eps) {
        // (token, head) pairs are independent; each
        // keeps its own reduction.
        const i64 units = n_tok * n_heads;
        par_for(units, 1, 4.0 * static_cast<f64>(units) * hd,
                [&](i64 b, i64 e) {
                    for (i64 u = b; u < e; u++) {
                        const i64 t = u / n_heads;
                        const i64 h = u - t * n_heads;
                        f32 *hx = x + (t * n_heads + h) * hd;
                        f32 ss = 0;
                        for (i64 i = 0; i < hd; i++) ss += hx[i] * hx[i];
                        const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(hd) + eps);
                        for (i64 i = 0; i < hd; i++) hx[i] = hx[i] * inv * w[i];
                    }
                });
    }

    DeviceCaps caps_;
    npu::Offload *npu_ = nullptr;
};

Backend *make_cpu_backend() {
    DeviceCaps c;
    c.name = "CPU";
    c.gfx = "cpu";
    c.family = GfxFamily::Unknown;
    c.wave_size = 64;
    c.cu_count = 1;
    return new CpuBackend(c);
}

Backend *make_npu_backend(int device_index, bool whole_graph, std::string *err) {
    std::string why;
    npu::Offload *o =
        npu::create(device_index, whole_graph ? npu::Mode::Graph : npu::Mode::Gemm,
                    &why);
    if (!o) {
        if (err) *err = why;
        return nullptr;
    }
    DeviceCaps c;
    c.name = npu::describe(o);
    c.gfx = "vnpu";
    c.family = GfxFamily::Unknown;
    c.wave_size = 1; // one command at a time per stream, not lockstep lanes
    c.cu_count = 1;
    return new CpuBackend(c, o);
}

bool npu_available() { return npu::available(); }

} // namespace krk
