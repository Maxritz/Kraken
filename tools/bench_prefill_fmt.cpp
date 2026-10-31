// Which prefill layout can carry a longer scale group? (plan 4.1)
//
// The shipped q8_0 prefill GEMM sits at the V1 instruction-mix ceiling, so the
// only lever left is a longer scale run: G2 measured ~1.1x and G4 ~1.19x over
// the shipped kernel at 768x2048x512. Imposing that run on q8_0's bytes at
// runtime is lossy -- one f16 scale over 128 values cannot represent a row whose
// magnitude moves inside it -- and the runtime requantisation that tried to fix
// it was a measured dead end. The format has to CARRY the longer run, so the
// bytes have to change.
//
// This tool settles the layout on paper and on the machine before any engine
// code exists. It builds the same weight rows, quantises them every way, and
// reports the two things a decision needs:
//
//   error  -- relative RMS, max, and the dot-product error against f32 with a
//             random activation row (the logit-scale number), per layout
//   speed  -- a prototype x8 dot per layout over the real sweep shape, so the
//             op-count argument is measured rather than asserted
//
// Layouts, all with int8 payloads and int8 activations:
//
//   q8_0    32-value blocks, one f16 scale each          34 B / 32   (shipped)
//   g4_max  128-value blocks, one f16 scale, max-based  130 B / 128
//   g4_lsq  128-value blocks, one f16 scale, least squares (same bytes)
//   s2/s4/s8  128-value group: one f16 ANCHOR plus 2/4/8-bit per-32 exponents.
//           The exponents are powers of two, so the kernel merges a group's
//           four blocks with integer shifts and applies ONE float scale per
//           128 -- the G4 op count with q8_0's per-32 grid.
//
// The scale error is the whole question. A max-based scale is minimax but not
// least-squares: an outlier at 1.0 in a row of 0.2s costs every other value
// precision it did not have to lose. Least squares is the same bytes and the
// same kernel, so if it wins on error it wins for free.
//
// Synthetic rows on purpose: no checkpoint is used in this repository (the
// rule is build -> test -> delete), and the row shapes that matter here are
// characterised by their block-maximum spread, which is printed.

#include <krk/common.hpp>
#include <krk/quant.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// The project builds for a baseline ISA and dispatches at runtime, so the
// prototypes below are target-attributed and reached through a feature check --
// never through a compile-time macro, which would silently delete them from a
// baseline build (the first version of this tool printed q8_0 and nothing else).
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KRK_BENCH_AVX512 1
#endif

using namespace krk;

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---- the row generator ------------------------------------------------------
// Three shapes, because the layout question is about the SPREAD of the values
// inside a group, and one distribution cannot show that:
//   gauss   iid N(0,1): block maxima within a group are within ~1.5x
//   heavy   Student-t(3): outliers 5-10x the median magnitude
//   hetero  N(0,1) with a magnitude ramp along the row: the group's four blocks
//           differ by up to an order of magnitude by construction
enum class RowKind { Gauss, Heavy, Hetero };

const char *kind_name(RowKind k) {
    switch (k) {
    case RowKind::Gauss: return "gauss";
    case RowKind::Heavy: return "heavy";
    case RowKind::Hetero: return "hetero";
    }
    return "?";
}

void make_row(RowKind kind, i64 n, std::mt19937 &rng, f32 *out) {
    std::normal_distribution<f32> g(0.0f, 1.0f);
    std::uniform_real_distribution<f32> u(0.0f, 1.0f);
    for (i64 i = 0; i < n; i++) {
        f32 v = g(rng);
        if (kind == RowKind::Heavy) {
            // t(3) = g / sqrt(chi2_3 / 3), built from three normals.
            f32 c = (g(rng) * g(rng) + g(rng) * g(rng) + g(rng) * g(rng)) / 3.0f;
            v = g(rng) / std::sqrt(std::max(c, 1e-6f));
        } else if (kind == RowKind::Hetero) {
            const f32 ramp = 0.05f + 2.0f * static_cast<f32>(i) /
                                        static_cast<f32>(std::max<i64>(1, n));
            v = g(rng) * ramp;
        } else {
            v = v * (0.5f + u(rng)); // per-value spread, keeps the tail short
        }
        out[i] = v;
    }
}

// ---- layouts ----------------------------------------------------------------
// One row's quantised bytes, plus the error this layout costs on that row.
struct Quant {
    std::string name;
    f64 bytes_per_value = 0.0;
    std::vector<u8> row; // one row's storage, already in the layout's shape
};

// q8_0: [f16 d][32 i8] per 32 values. Uses the engine's own quantiser so the
// baseline is the baseline, not a re-implementation of it.
void quant_q80(const f32 *w, i64 n, std::vector<u8> *out) {
    const i64 nb = n / 32;
    out->assign(static_cast<size_t>(nb * 34), 0);
    std::vector<i8> q(static_cast<size_t>(n));
    std::vector<f32> d(static_cast<size_t>(nb));
    quantize_q8_row(w, n, q.data(), d.data());
    for (i64 b = 0; b < nb; b++) {
        const u16 h = fp32_to_fp16(d[static_cast<size_t>(b)]);
        const size_t o = static_cast<size_t>(b * 34);
        (*out)[o] = static_cast<u8>(h & 0xFF);
        (*out)[o + 1] = static_cast<u8>(h >> 8);
        for (i64 i = 0; i < 32; i++)
            (*out)[o + 2 + static_cast<size_t>(i)] =
                static_cast<u8>(q[static_cast<size_t>(b * 32 + i)]);
    }
}

void dequant_q80(const u8 *row, i64 n, f32 *out) {
    const i64 nb = n / 32;
    for (i64 b = 0; b < nb; b++) {
        const size_t o = static_cast<size_t>(b * 34);
        const f32 d = fp16_to_fp32(static_cast<u16>(row[o] | (row[o + 1] << 8)));
        for (i64 i = 0; i < 32; i++)
            out[b * 32 + i] =
                d * static_cast<f32>(static_cast<i8>(row[o + 2 + i]));
    }
}

// A 128-value group with one f16 scale. `lsq` picks the scale by least squares
// (two refinement passes) instead of max|w|/127; the bytes are identical.
void quant_g4(const f32 *w, i64 n, bool lsq, std::vector<u8> *out) {
    const i64 ng = n / 128;
    out->assign(static_cast<size_t>(ng * 130), 0);
    for (i64 g = 0; g < ng; g++) {
        const f32 *src = w + g * 128;
        f32 amax = 0.0f;
        for (i64 i = 0; i < 128; i++) amax = std::max(amax, std::fabs(src[i]));
        f32 s = amax > 0.0f ? amax / 127.0f : 0.0f;
        i8 q[128] = {};
        for (int pass = 0; pass < (lsq ? 3 : 1); pass++) {
            // The stored scale is f16: refine against what will be WRITTEN, so
            // the number the decoder multiplies by is the number fitted here.
            const f32 sf = fp16_to_fp32(fp32_to_fp16(s));
            const f32 inv = sf > 0.0f ? 1.0f / sf : 0.0f;
            f64 num = 0.0, den = 0.0;
            for (i64 i = 0; i < 128; i++) {
                int v = static_cast<int>(std::lround(src[i] * inv));
                v = std::max(-127, std::min(127, v));
                q[i] = static_cast<i8>(v);
                num += static_cast<f64>(src[i]) * v;
                den += static_cast<f64>(v) * v;
            }
            if (!lsq || den <= 0.0) break;
            s = static_cast<f32>(num / den);
        }
        const u16 h = fp32_to_fp16(s);
        const size_t o = static_cast<size_t>(g * 130);
        (*out)[o] = static_cast<u8>(h & 0xFF);
        (*out)[o + 1] = static_cast<u8>(h >> 8);
        // Re-quantise against the written scale, so err is the decoder's error.
        const f32 sf = fp16_to_fp32(h);
        const f32 inv = sf > 0.0f ? 1.0f / sf : 0.0f;
        for (i64 i = 0; i < 128; i++) {
            int v = static_cast<int>(std::lround(src[i] * inv));
            v = std::max(-127, std::min(127, v));
            (*out)[o + 2 + static_cast<size_t>(i)] = static_cast<u8>(v);
        }
    }
}

void dequant_g4(const u8 *row, i64 n, f32 *out) {
    const i64 ng = n / 128;
    for (i64 g = 0; g < ng; g++) {
        const size_t o = static_cast<size_t>(g * 130);
        const f32 d = fp16_to_fp32(static_cast<u16>(row[o] | (row[o + 1] << 8)));
        for (i64 i = 0; i < 128; i++)
            out[g * 128 + i] =
                d * static_cast<f32>(static_cast<i8>(row[o + 2 + i]));
    }
}

// The s-layout: one f16 ANCHOR per 128 (the largest of the four blocks' ideal
// scales) and, per 32-value block, a power-of-two exponent so that block keeps
// its own grid: scale_b = anchor / 2^e_b. `bits` is the exponent width, so the
// exponent byte budget is 4*bits/8 bytes per group and the range is
// anchor..anchor/2^(2^bits-1).
f64 s_bytes_per_value(int bits) { return (128.0 + 2.0 + 4.0 * bits / 8.0) / 128.0; }

void quant_s(const f32 *w, i64 n, int bits, std::vector<u8> *out) {
    const i64 ng = n / 128;
    const i64 exp_bytes = 4 * bits / 8;
    const i64 row_bytes = 130 + exp_bytes - 2; // f16 anchor + exponents + 128
    const int emax = (1 << bits) - 1;
    out->assign(static_cast<size_t>(ng * (2 + exp_bytes + 128)), 0);
    (void)row_bytes;
    for (i64 g = 0; g < ng; g++) {
        const f32 *src = w + g * 128;
        f32 bmax[4];
        for (int b = 0; b < 4; b++) {
            f32 m = 0.0f;
            for (int i = 0; i < 32; i++) m = std::max(m, std::fabs(src[b * 32 + i]));
            bmax[b] = m;
        }
        f32 gmax = 0.0f;
        for (int b = 0; b < 4; b++) gmax = std::max(gmax, bmax[b]);
        // The anchor is the LARGEST block's scale, and every other block gets
        // the power of two that brings its own grid back to that block's size.
        f32 anchor = gmax > 0.0f ? gmax / 127.0f : 0.0f;
        const u16 h = fp32_to_fp16(anchor);
        anchor = fp16_to_fp32(h);
        u32 exps = 0;
        for (int b = 0; b < 4; b++) {
            int e = 0;
            if (bmax[b] > 0.0f && anchor > 0.0f) {
                // The finest grid that still covers this block: the largest e
                // with bmax_b <= 127 * anchor / 2^e. Computing it from
                // floor(log2(.)) instead is off by one whenever the ratio sits
                // within an f16-ulp of 1.0 -- which is exactly the common case
                // -- and that halved the scale and saturated the block. The
                // tolerance is 0.4 of a quantisation step, so a value that
                // rounds to 127.4 still fits.
                for (int cand = 0; cand <= emax; cand++) {
                    const f32 sc = anchor / static_cast<f32>(1u << cand);
                    if (bmax[b] <= sc * 127.4f) e = cand;
                    else break;
                }
            }
            exps |= static_cast<u32>(e) << (b * bits);
        }
        const size_t o = static_cast<size_t>(g * (2 + exp_bytes + 128));
        (*out)[o] = static_cast<u8>(h & 0xFF);
        (*out)[o + 1] = static_cast<u8>(h >> 8);
        for (i64 i = 0; i < exp_bytes; i++)
            (*out)[o + 2 + static_cast<size_t>(i)] =
                static_cast<u8>((exps >> (8 * i)) & 0xFF);
        for (int b = 0; b < 4; b++) {
            const int e = static_cast<int>((exps >> (b * bits)) & static_cast<u32>(emax));
            const f32 scale = anchor / static_cast<f32>(1u << e);
            const f32 inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int i = 0; i < 32; i++) {
                int v = static_cast<int>(std::lround(src[b * 32 + i] * inv));
                v = std::max(-127, std::min(127, v));
                (*out)[o + 2 + static_cast<size_t>(exp_bytes) +
                               static_cast<size_t>(b * 32 + i)] = static_cast<u8>(v);
            }
        }
    }
}

void dequant_s(const u8 *row, i64 n, int bits, f32 *out) {
    const i64 ng = n / 128;
    const i64 exp_bytes = 4 * bits / 8;
    const size_t stride = static_cast<size_t>(2 + exp_bytes + 128);
    const int emax = (1 << bits) - 1;
    for (i64 g = 0; g < ng; g++) {
        const size_t o = static_cast<size_t>(g) * stride;
        const f32 anchor =
            fp16_to_fp32(static_cast<u16>(row[o] | (row[o + 1] << 8)));
        u32 exps = 0;
        for (i64 i = 0; i < exp_bytes; i++)
            exps |= static_cast<u32>(row[o + 2 + static_cast<size_t>(i)])
                    << (8 * i);
        for (int b = 0; b < 4; b++) {
            const int e = static_cast<int>((exps >> (b * bits)) & static_cast<u32>(emax));
            const f32 scale = anchor / static_cast<f32>(1u << e);
            for (int i = 0; i < 32; i++)
                out[g * 128 + b * 32 + i] =
                    scale * static_cast<f32>(
                                static_cast<i8>(row[o + 2 + static_cast<size_t>(exp_bytes) +
                                                    static_cast<size_t>(b * 32 + i)]));
        }
    }
}

// ---- error ----------------------------------------------------------------
struct Err {
    f64 rms = 0.0;      // ||w - wq|| / ||w||
    f64 maxrel = 0.0;   // max|w - wq| / max|w|
    f64 dot = 0.0;      // |(w - wq).x| / (||w|| ||x||)
};

// ---- speed -----------------------------------------------------------------
// The cost the longer group removes is not the multiply, it is the SCALE VECTOR
// BUILD: the shipped kernel assembles a 512-bit vector of per-block scale
// products (two broadcasts and an insert) for every 64 values, and one f16
// load per 32. A kernel can only broadcast ONE scalar per group if both sides
// share the group -- the product s_w * s_x is a single number only when the
// weight and the activation are quantised over the same run. That is the claim
// these four kernels test, and it is the reason the layout question is really
// an ACTIVATION-group question:
//
//   q8_0        weight 32, activation 32   (shipped)
//   g4w32       weight 128, activation 32  (weight group lengthened alone)
//   g4w128      weight 128, activation 128 (both sides)
//   s4w128      weight 128 + 4-bit per-32 exponents, activation 128
//
// The last one keeps a per-32 grid on the weight side and merges the four
// blocks with integer shifts, so it should cost a little more than g4w128 and
// a little less than q8_0.
#ifdef KRK_BENCH_AVX512
#define KRK_ATTR __attribute__((target("avx512f,avx512bw")))

// Loads a 128-value weight group: two 64-byte payload halves, each sign-split
// into magnitude bytes (maddubs needs the sign on the OTHER operand).
struct WGroup {
    __m512i mag0, mag1;
    __mmask64 neg0, neg1;
    f32 d;
};

KRK_ATTR static WGroup load_wgroup(const u8 *w) {
    const __m512i zero = _mm512_setzero_si512();
    WGroup g;
    g.d = fp16_to_fp32(static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
    const __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void *>(w + 2));
    const __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void *>(w + 66));
    g.neg0 = _mm512_cmp_epi8_mask(v0, zero, _MM_CMPINT_LT);
    g.neg1 = _mm512_cmp_epi8_mask(v1, zero, _MM_CMPINT_LT);
    g.mag0 = _mm512_mask_sub_epi8(v0, g.neg0, zero, v0);
    g.mag1 = _mm512_mask_sub_epi8(v1, g.neg1, zero, v1);
    return g;
}

// 8 rows x one 128-value group against per-128 activation scales: one scalar
// product per group, broadcast once.
KRK_ATTR static void dot_g4w128_x8(const u8 *w, const i8 *xq, const f32 *dx,
                                   i64 n, i64 rstride, i64 dstride, f32 *out,
                                   i64 out_stride) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    for (i64 g = 0; g < n / 128; g++, w += 130) {
        const WGroup wg = load_wgroup(w);
        for (int r = 0; r < 8; r++) {
            const i8 *xp = xq + static_cast<i64>(r) * rstride + g * 128;
            const __m512i x0 = _mm512_loadu_si512(reinterpret_cast<const void *>(xp));
            const __m512i x1 = _mm512_loadu_si512(reinterpret_cast<const void *>(xp + 64));
            const __m512i b0 = _mm512_mask_sub_epi8(x0, wg.neg0, zero, x0);
            const __m512i b1 = _mm512_mask_sub_epi8(x1, wg.neg1, zero, x1);
            __m512i iv = _mm512_madd_epi16(_mm512_maddubs_epi16(wg.mag0, b0), ones);
            iv = _mm512_add_epi32(
                iv, _mm512_madd_epi16(_mm512_maddubs_epi16(wg.mag1, b1), ones));
            const f32 s = wg.d * dx[static_cast<i64>(r) * dstride + g];
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), _mm512_set1_ps(s), acc[r]);
        }
    }
    for (int r = 0; r < 8; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}

// The same weight bytes against per-32 activation scales: the scale vector has
// to be rebuilt per 64 values exactly as the shipped kernel does, which is the
// whole point of the comparison.
KRK_ATTR static void dot_g4w32_x8(const u8 *w, const i8 *xq, const f32 *dx,
                                  i64 n, i64 rstride, i64 dstride, f32 *out,
                                  i64 out_stride) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    for (i64 g = 0; g < n / 128; g++, w += 130) {
        const WGroup wg = load_wgroup(w);
        for (int half = 0; half < 2; half++) {
            const __m512i mag = half == 0 ? wg.mag0 : wg.mag1;
            const __mmask64 neg = half == 0 ? wg.neg0 : wg.neg1;
            for (int r = 0; r < 8; r++) {
                const i8 *xp = xq + static_cast<i64>(r) * rstride + g * 128 +
                               half * 64;
                const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(xp));
                const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
                const __m512i iv = _mm512_madd_epi16(_mm512_maddubs_epi16(mag, xb), ones);
                const f32 *dp = dx + static_cast<i64>(r) * dstride + g * 4 + half * 2;
                const f32 s0 = wg.d * dp[0];
                const f32 s1 = wg.d * dp[1];
                const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
                    _mm512_castps_si512(_mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
                    _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
            }
        }
    }
    for (int r = 0; r < 8; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}

// s4 against per-128 activation scales. The weight keeps a per-32 grid; the four
// blocks of the group are brought onto the anchor's grid with a variable
// right shift on the int32 lanes that hold each block, then ONE product is
// applied per 128. The shift counts are a lane vector, built once per group.
KRK_ATTR static void dot_s4w128_x8(const u8 *w, const i8 *xq, const f32 *dx,
                                   i64 n, i64 rstride, i64 dstride, f32 *out,
                                   i64 out_stride) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    // 128 int8 values + f16 anchor + 4 exponent nibbles.
    for (i64 g = 0; g < n / 128; g++, w += 2 + 2 + 128) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
        const u32 exps = static_cast<u32>(w[2]) |
                         (static_cast<u32>(w[3]) << 8);
        const u8 e[4] = {static_cast<u8>(exps & 0xF), static_cast<u8>((exps >> 4) & 0xF),
                         static_cast<u8>((exps >> 8) & 0xF),
                         static_cast<u8>((exps >> 12) & 0xF)};
        const __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void *>(w + 4));
        const __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void *>(w + 68));
        const __mmask64 n0 = _mm512_cmp_epi8_mask(v0, zero, _MM_CMPINT_LT);
        const __mmask64 n1 = _mm512_cmp_epi8_mask(v1, zero, _MM_CMPINT_LT);
        const __m512i m0 = _mm512_mask_sub_epi8(v0, n0, zero, v0);
        const __m512i m1 = _mm512_mask_sub_epi8(v1, n1, zero, v1);
        // Lanes 0-7 hold block 0 (values 0-31), lanes 8-15 block 1; the second
        // 64-byte half holds blocks 2 and 3.
        const __m512i sh0 = _mm512_set_epi32(
            e[1], e[1], e[1], e[1], e[1], e[1], e[1], e[1], e[0], e[0], e[0], e[0],
            e[0], e[0], e[0], e[0]);
        const __m512i sh1 = _mm512_set_epi32(
            e[3], e[3], e[3], e[3], e[3], e[3], e[3], e[3], e[2], e[2], e[2], e[2],
            e[2], e[2], e[2], e[2]);
        // Round to nearest before the shift. A plain arithmetic shift truncates
        // toward -inf on every lane, and the bias does not cancel across lanes:
        // measured 3.2e-2 relative against dequant-then-f32 with the truncating
        // version, 3 decimals worse than the format's own error.
        const __m512i rnd0 = _mm512_set_epi32(
            e[1] ? (1 << (e[1] - 1)) : 0, e[1] ? (1 << (e[1] - 1)) : 0,
            e[1] ? (1 << (e[1] - 1)) : 0, e[1] ? (1 << (e[1] - 1)) : 0,
            e[1] ? (1 << (e[1] - 1)) : 0, e[1] ? (1 << (e[1] - 1)) : 0,
            e[1] ? (1 << (e[1] - 1)) : 0, e[1] ? (1 << (e[1] - 1)) : 0,
            e[0] ? (1 << (e[0] - 1)) : 0, e[0] ? (1 << (e[0] - 1)) : 0,
            e[0] ? (1 << (e[0] - 1)) : 0, e[0] ? (1 << (e[0] - 1)) : 0,
            e[0] ? (1 << (e[0] - 1)) : 0, e[0] ? (1 << (e[0] - 1)) : 0,
            e[0] ? (1 << (e[0] - 1)) : 0, e[0] ? (1 << (e[0] - 1)) : 0);
        const __m512i rnd1 = _mm512_set_epi32(
            e[3] ? (1 << (e[3] - 1)) : 0, e[3] ? (1 << (e[3] - 1)) : 0,
            e[3] ? (1 << (e[3] - 1)) : 0, e[3] ? (1 << (e[3] - 1)) : 0,
            e[3] ? (1 << (e[3] - 1)) : 0, e[3] ? (1 << (e[3] - 1)) : 0,
            e[3] ? (1 << (e[3] - 1)) : 0, e[3] ? (1 << (e[3] - 1)) : 0,
            e[2] ? (1 << (e[2] - 1)) : 0, e[2] ? (1 << (e[2] - 1)) : 0,
            e[2] ? (1 << (e[2] - 1)) : 0, e[2] ? (1 << (e[2] - 1)) : 0,
            e[2] ? (1 << (e[2] - 1)) : 0, e[2] ? (1 << (e[2] - 1)) : 0,
            e[2] ? (1 << (e[2] - 1)) : 0, e[2] ? (1 << (e[2] - 1)) : 0);
        for (int r = 0; r < 8; r++) {
            const i8 *xp = xq + static_cast<i64>(r) * rstride + g * 128;
            const __m512i x0 = _mm512_loadu_si512(reinterpret_cast<const void *>(xp));
            const __m512i x1 = _mm512_loadu_si512(reinterpret_cast<const void *>(xp + 64));
            const __m512i b0 = _mm512_mask_sub_epi8(x0, n0, zero, x0);
            const __m512i b1 = _mm512_mask_sub_epi8(x1, n1, zero, x1);
            __m512i i0 = _mm512_madd_epi16(_mm512_maddubs_epi16(m0, b0), ones);
            __m512i i1 = _mm512_madd_epi16(_mm512_maddubs_epi16(m1, b1), ones);
            i0 = _mm512_srav_epi32(_mm512_add_epi32(i0, rnd0), sh0);
            i1 = _mm512_srav_epi32(_mm512_add_epi32(i1, rnd1), sh1);
            const __m512i iv = _mm512_add_epi32(i0, i1);
            const f32 s = d * dx[static_cast<i64>(r) * dstride + g];
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), _mm512_set1_ps(s), acc[r]);
        }
    }
    for (int r = 0; r < 8; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}
#endif

// Per-128 activation scales: what the both-sides-long group needs.
void quantize_q8_128(const f32 *x, i64 n, i8 *xq, f32 *d) {
    for (i64 g = 0; g < n / 128; g++) {
        f32 m = 0.0f;
        for (i64 i = 0; i < 128; i++) m = std::max(m, std::fabs(x[g * 128 + i]));
        const f32 s = m > 0.0f ? m / 127.0f : 0.0f;
        const f32 inv = s > 0.0f ? 1.0f / s : 0.0f;
        d[g] = s;
        for (i64 i = 0; i < 128; i++) {
            int v = static_cast<int>(std::lround(x[g * 128 + i] * inv));
            v = std::max(-127, std::min(127, v));
            xq[g * 128 + i] = static_cast<i8>(v);
        }
    }
}

// Best-of-`reps` inside one process: the box is shared, so a single pass per
// kernel makes a kernel's own run-to-run noise look like a format difference
// (measured: g4w+act128 swung 31.1..43.3 across five processes).
template <class K>
void bench_sweep(const char *label, K kern, const u8 *w, i64 n_out, i64 row_bytes,
                 i64 n_in, i64 nb, const i8 *xq, const f32 *xd, f32 *out,
                 i64 rows, int reps) {
    double best = 0.0, worst = 0.0;
    const double m = static_cast<double>(rows / 8 * 8) * n_out * n_in;
    for (int rep = 0; rep < reps; rep++) {
        const double t0 = now_s();
        for (i64 r = 0; r + 8 <= rows; r += 8)
            for (i64 o = 0; o < n_out; o++)
                kern(w + static_cast<size_t>(o) * row_bytes, xq + r * n_in,
                     xd + r * nb, n_in, n_in, nb, out + r * n_out + o, n_out);
        const double dt = now_s() - t0;
        const double g = m / dt / 1e9;
        best = std::max(best, g);
        if (rep == 0) worst = g;
        worst = std::min(worst, g);
    }
    std::printf("  %-24s %8.2f GMAC/s  (best of %d; worst %8.2f)\n", label, best,
                reps, worst);
}

} // namespace

int main(int argc, char **argv) {
    const i64 n_in = argc > 1 ? std::atoll(argv[1]) : 768;
    const i64 n_out = argc > 2 ? std::atoll(argv[2]) : 2048;
    const i64 rows = argc > 3 ? std::atoll(argv[3]) : 512;
    const i64 nrow_err = argc > 4 ? std::atoll(argv[4]) : 256;
    const int reps = argc > 5 ? static_cast<int>(std::atoll(argv[5])) : 1;
    std::mt19937 rng(20261031u);

    std::printf("prefill layout comparison  n_in=%lld n_out=%lld rows=%lld\n",
                static_cast<long long>(n_in), static_cast<long long>(n_out),
                static_cast<long long>(rows));

    // ---- error, per row kind --------------------------------------------
    struct Case {
        const char *name;
        int bits; // 0 = q8_0, 1 = g4 max, 2 = g4 lsq, else s-layout bits
        f64 bpv;
    };
    const Case cases[] = {
        {"q8_0 (shipped)", 0, 34.0 / 32.0},
        {"g4 max scale", 1, 130.0 / 128.0},
        {"g4 lsq scale", 2, 130.0 / 128.0},
        {"s2 (2-bit exp)", 2 | 0x100, s_bytes_per_value(2)},
        {"s4 (4-bit exp)", 4 | 0x100, s_bytes_per_value(4)},
        {"s8 (8-bit exp)", 8 | 0x100, s_bytes_per_value(8)},
    };
    for (RowKind kind : {RowKind::Gauss, RowKind::Heavy, RowKind::Hetero}) {
        std::printf("\n%s rows (%lld rows x %lld values, f32 reference)\n",
                    kind_name(kind), static_cast<long long>(nrow_err),
                    static_cast<long long>(n_in));
        std::vector<f32> w(static_cast<size_t>(n_in));
        std::vector<f32> wq(static_cast<size_t>(n_in));
        std::vector<f32> x(static_cast<size_t>(n_in));
        std::vector<u8> bytes;
        std::printf("  %-18s %8s %10s %10s %10s\n", "layout", "B/value", "rms",
                    "max", "logit");
        for (const Case &c : cases) {
            Err e{}, worst{};
            for (i64 ri = 0; ri < nrow_err; ri++) {
                make_row(kind, n_in, rng, w.data());
                for (i64 i = 0; i < n_in; i++) x[i] = 0.0f;
                std::normal_distribution<f32> gn(0.0f, 1.0f);
                for (i64 i = 0; i < n_in; i++) x[i] = gn(rng);
                if (c.bits == 0) {
                    quant_q80(w.data(), n_in, &bytes);
                    dequant_q80(bytes.data(), n_in, wq.data());
                } else if (c.bits == 1 || c.bits == 2) {
                    quant_g4(w.data(), n_in, c.bits == 2, &bytes);
                    dequant_g4(bytes.data(), n_in, wq.data());
                } else {
                    const int bits = c.bits & 0xFF;
                    quant_s(w.data(), n_in, bits, &bytes);
                    dequant_s(bytes.data(), n_in, bits, wq.data());
                }
                f64 nn = 0, ne = 0, mx = 0, dn = 0, de = 0, xn = 0;
                for (i64 i = 0; i < n_in; i++) {
                    const f64 d = static_cast<f64>(w[i]) - wq[i];
                    nn += static_cast<f64>(w[i]) * w[i];
                    ne += d * d;
                    mx = std::max(mx, std::fabs(d));
                    dn += static_cast<f64>(w[i]) * x[i];
                    de += d * x[i];
                    xn += static_cast<f64>(x[i]) * x[i];
                }
                f64 amax = 0;
                for (i64 i = 0; i < n_in; i++)
                    amax = std::max(amax, static_cast<f64>(std::fabs(w[i])));
                e.rms += std::sqrt(ne / std::max(nn, 1e-30)) / nrow_err;
                e.maxrel += mx / std::max(amax, 1e-30) / nrow_err;
                e.dot += std::fabs(de) / std::max(1.0, std::sqrt(nn * xn)) / nrow_err;
                worst.rms = std::max(worst.rms, std::sqrt(ne / std::max(nn, 1e-30)));
                worst.dot = std::max(worst.dot, std::fabs(de) / std::max(1.0, std::sqrt(nn * xn)));
            }
            std::printf("  %-18s %8.4f %10.3e %10.3e %10.3e   worst rms %.3e\n",
                        c.name, c.bpv, e.rms, e.maxrel, e.dot, worst.rms);
        }
    }

    // ---- the activation side --------------------------------------------
    // The both-sides-long group changes the ACTIVATION quantiser too, and that
    // is where its error comes from: weights are fixed and can be fitted
    // offline, an activation row is quantised as it is produced. Same rows, f32
    // weights, only the activation group differs.
    std::printf("\nactivation group: relative dot error against f32 activations\n");
    std::printf("  %-12s %12s %12s %10s\n", "rows", "q8 per-32", "q8 per-128",
                "ratio");
    for (RowKind kind : {RowKind::Gauss, RowKind::Heavy, RowKind::Hetero}) {
        f64 e32 = 0.0, e128 = 0.0;
        std::vector<f32> w(static_cast<size_t>(n_in));
        std::vector<f32> x(static_cast<size_t>(n_in));
        std::vector<i8> q(static_cast<size_t>(n_in));
        std::vector<f32> d32(static_cast<size_t>(n_in / 32));
        std::vector<f32> d128(static_cast<size_t>(n_in / 128));
        std::normal_distribution<f32> gn(0.0f, 1.0f);
        for (i64 ri = 0; ri < nrow_err; ri++) {
            make_row(kind, n_in, rng, w.data());
            make_row(RowKind::Gauss, n_in, rng, x.data());
            f64 nn = 0, xn = 0, exact = 0;
            for (i64 i = 0; i < n_in; i++) {
                nn += static_cast<f64>(w[i]) * w[i];
                xn += static_cast<f64>(x[i]) * x[i];
                exact += static_cast<f64>(w[i]) * x[i];
            }
            const f64 scale = std::max(1e-30, std::sqrt(nn * xn));
            f64 dq = 0, dq128 = 0;
            quantize_q8_row(x.data(), n_in, q.data(), d32.data());
            for (i64 i = 0; i < n_in; i++)
                dq += static_cast<f64>(w[i]) * q[i] * d32[static_cast<size_t>(i / 32)];
            quantize_q8_128(x.data(), n_in, q.data(), d128.data());
            for (i64 i = 0; i < n_in; i++)
                dq128 += static_cast<f64>(w[i]) * q[i] * d128[static_cast<size_t>(i / 128)];
            e32 += std::fabs(dq - exact) / scale / nrow_err;
            e128 += std::fabs(dq128 - exact) / scale / nrow_err;
        }
        std::printf("  %-12s %12.3e %12.3e %10.2fx\n", kind_name(kind), e32, e128,
                    e128 / std::max(1e-30, e32));
    }

    // ---- correctness of the prototypes ----------------------------------
    // A fast kernel that computes the wrong dot is worth nothing, so every
    // prototype is compared against dequant-then-f32 on the same bytes before
    // its speed is quoted. Tolerance 1e-5 relative: the integer paths are exact
    // and the only float work is the scale product.
    {
        const i64 nc = n_in;
        std::vector<u8> wg(static_cast<size_t>((nc / 128) * 130), 0);
        std::vector<u8> ws(static_cast<size_t>((nc / 128) * 132), 0);
        std::vector<f32> wf(static_cast<size_t>(nc));
        make_row(RowKind::Gauss, nc, rng, wf.data());
        quant_g4(wf.data(), nc, false, &wg);
        quant_s(wf.data(), nc, 4, &ws);
        std::vector<f32> wd(static_cast<size_t>(nc)), wsd(static_cast<size_t>(nc));
        dequant_g4(wg.data(), nc, wd.data());
        dequant_s(ws.data(), nc, 4, wsd.data());
        std::vector<f32> xf8(static_cast<size_t>(8 * nc));
        std::vector<i8> xq8(static_cast<size_t>(8 * nc));
        std::vector<f32> d32(static_cast<size_t>(8 * (nc / 32)));
        std::vector<f32> d128(static_cast<size_t>(8 * (nc / 128)));
        for (int r = 0; r < 8; r++) {
            make_row(RowKind::Gauss, nc, rng, xf8.data() + r * nc);
            quantize_q8_row(xf8.data() + r * nc, nc, xq8.data() + r * nc,
                            d32.data() + r * (nc / 32));
            quantize_q8_128(xf8.data() + r * nc, nc, xq8.data() + r * nc,
                            d128.data() + r * (nc / 128));
        }
        // Reference: dequantised weights times dequantised activations, in f64.
        auto ref = [&](const f32 *w, const f32 *x) {
            f64 acc = 0;
            for (i64 i = 0; i < nc; i++) acc += static_cast<f64>(w[i]) * x[i];
            return acc;
        };
        std::vector<f32> xd8(static_cast<size_t>(8 * nc));
        for (int r = 0; r < 8; r++) {
            // dequantise the activation row the same way its quantiser did
            for (i64 b = 0; b < nc / 32; b++)
                for (i64 i = 0; i < 32; i++)
                    xd8[r * nc + b * 32 + i] =
                        d32[static_cast<size_t>(r * (nc / 32) + b)] *
                        static_cast<f32>(xq8[static_cast<size_t>(r * nc + b * 32 + i)]);
        }
        std::vector<f32> xd128(static_cast<size_t>(8 * nc));
        for (int r = 0; r < 8; r++)
            for (i64 g = 0; g < nc / 128; g++)
                for (i64 i = 0; i < 128; i++)
                    xd128[r * nc + g * 128 + i] =
                        d128[static_cast<size_t>(r * (nc / 128) + g)] *
                        static_cast<f32>(xq8[static_cast<size_t>(r * nc + g * 128 + i)]);
#ifdef KRK_BENCH_AVX512
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
            std::printf("\nprototype correctness against dequant-then-f32 (n=%lld)\n",
                        static_cast<long long>(nc));
            std::vector<f32> got(8);
            // Each kernel against ITS OWN dequantised weights: comparing the
            // s-layout's bytes with the g4 reference read as a 3e-2 failure
            // that was the harness's, not the kernel's.
            auto report = [&](const char *name, const std::vector<f32> &g,
                              const std::vector<f32> &x, const std::vector<f32> &wref) {
                f64 worst = 0;
                for (int r = 0; r < 8; r++) {
                    const f64 e = ref(wref.data(), x.data() + r * nc);
                    const f64 d = std::fabs(g[static_cast<size_t>(r)] - e) /
                                  std::max(1e-9, std::fabs(e));
                    worst = std::max(worst, d);
                }
                std::printf("  %-14s worst relative %.3e  %s\n", name, worst,
                            worst < 1e-5 ? "ok" : "FAIL");
            };
            std::vector<f32> gotg4(8), gots4(8), got32(8);
            dot_g4w128_x8(wg.data(), xq8.data(), d128.data(), nc, nc, nc / 128,
                          gotg4.data(), 1);
            dot_s4w128_x8(ws.data(), xq8.data(), d128.data(), nc, nc, nc / 128,
                          gots4.data(), 1);
            dot_g4w32_x8(wg.data(), xq8.data(), d32.data(), nc, nc, nc / 32,
                         got32.data(), 1);
            (void)got;
            report("g4w+act128", gotg4, xd128, wd);
            report("s4w+act128", gots4, xd128, wsd);
            report("g4w+act32", got32, xd8, wd);
        }
#endif
    }

    // ---- speed ----------------------------------------------------------
    // Same data shape as the shipped bench: random int8 payloads, legal f16
    // scales, q8_0 activations. Only the layouts' kernels differ, so a ratio
    // here is the kernel's op mix and nothing else.
    std::printf("\nspeed, sweep shape (8 rows per call), random int8 payloads\n");
    const i64 nb = n_in / 32;
    const i64 q80_row = n_in + 2 * nb;
    const i64 g4_row = (n_in / 128) * 130;
    const i64 s4_row = (n_in / 128) * 132;
    std::vector<u8> w80(static_cast<size_t>(n_out * q80_row));
    std::vector<u8> wg4(static_cast<size_t>(n_out * g4_row));
    std::vector<u8> ws4(static_cast<size_t>(n_out * s4_row));
    auto fill_payload = [&rng](std::vector<u8> *w, i64 stride) {
        for (size_t i = 0; i < w->size(); i++) {
            if (i % static_cast<size_t>(stride) == 0) (*w)[i] = 0x00;
            else if (i % static_cast<size_t>(stride) == 1) (*w)[i] = 0x1C; // 2^-9
            else (*w)[i] = static_cast<u8>(static_cast<i8>(static_cast<int>(rng() % 256) - 128));
        }
    };
    fill_payload(&w80, 34);
    fill_payload(&wg4, 130);
    fill_payload(&ws4, 132);
    // Legal exponents: 0..3 in each nibble, so the s4 kernel's shift is < 4.
    for (size_t i = 0; i < ws4.size(); i += 132) {
        ws4[i + 2] = static_cast<u8>((rng() & 0xFF));
        ws4[i + 3] = static_cast<u8>((rng() & 0xFF));
    }
    std::vector<f32> xf(static_cast<size_t>(rows * n_in));
    std::uniform_real_distribution<f32> d(-3.0f, 3.0f);
    for (auto &v : xf) v = d(rng);
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    std::vector<f32> xd(static_cast<size_t>(rows * nb)); // per 32
    std::vector<f32> xd128(static_cast<size_t>(rows * n_in / 128));
    std::vector<f32> xd64(static_cast<size_t>(rows * n_in / 64));
    for (i64 r = 0; r < rows; r++) {
        quantize_q8_row(xf.data() + r * n_in, n_in, xq.data() + r * n_in,
                        xd.data() + r * nb);
        quantize_q8_128(xf.data() + r * n_in, n_in, xq.data() + r * n_in,
                        xd128.data() + r * (n_in / 128));
    }
    (void)xd64;
    std::vector<f32> out(static_cast<size_t>(rows * n_out));

    bench_sweep("q8_0 (shipped)", dot_q8_0_q8_x8, w80.data(), n_out, q80_row, n_in, nb,
                xq.data(), xd.data(), out.data(), rows, reps);
#ifdef KRK_BENCH_AVX512
    const bool have512 = __builtin_cpu_supports("avx512f") &&
                         __builtin_cpu_supports("avx512bw");
    if (!have512)
        std::printf("  (no avx512f+bw on this host: the 128-group kernels are "
                    "not runnable here)\n");
    // The g4w32 kernel indexes dx per 32 (dstride = nb); the per-128 kernels
    // index dx per 128 (dstride = n_in/128).
    if (have512) {
        bench_sweep("g4w+act32", dot_g4w32_x8, wg4.data(), n_out, g4_row, n_in, nb,
                    xq.data(), xd.data(), out.data(), rows, reps);
        const i64 nb128 = n_in / 128;
        bench_sweep("g4w+act128", dot_g4w128_x8, wg4.data(), n_out, g4_row, n_in,
                    nb128, xq.data(), xd128.data(), out.data(), rows, reps);
        bench_sweep("s4w+act128", dot_s4w128_x8, ws4.data(), n_out, s4_row, n_in,
                    nb128, xq.data(), xd128.data(), out.data(), rows, reps);
    }
#endif
    return 0;
}
