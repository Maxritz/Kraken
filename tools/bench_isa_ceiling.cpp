// What is the fastest an 8-bit x 8-bit dot can possibly run on this machine?
//
// The question keeps coming up as "why is the GEMM only doing 24 GMAC/s", and
// the honest answer needs a ceiling measured on the same box with the same
// instruction mix -- not a figure derived from an assumed clock and an assumed
// port count. This tool measures that ceiling four ways:
//
//   A  registers only, integer accumulate            (maddubs + maddwd + paddd)
//   B  registers only, the real float tail           (... + cvtepi32_ps + fmadd)
//   C  B plus the real operand loads                 (weight bytes + activation)
//   D  C at four rows in flight instead of eight
//
// Compare C against the production kernel's own rate. That ratio is how much
// is left on the table inside the inner loop; if it is within ~10% there is
// nothing left to win there and any further speedup has to come from doing
// fewer operations per MAC (VNNI) or moving fewer bytes (decode, not prefill).
//
// Two traps this tool is built to avoid, both of which produced false ceilings
// in earlier rounds:
//   * A probe whose operands are loop-invariant gets hoisted, so it measures
//     the accumulator chain and nothing else. Every operand here is bumped by
//     the iteration counter, and the sink is volatile.
//   * A compile-time-constant scale lets the compiler drop the whole
//     int-to-float tail. The scale vector is built from a runtime value.
#include <immintrin.h>

#include "krk/common.hpp"
#include "krk/quant.hpp"

using krk::f32;
using krk::i64;
using krk::u8;
using krk::u16;
using krk::u32;
using krk::i8;
using krk::quantize_q8_row;
using krk::dot_q8_0_q8_x8;
using krk::dot_q8_0_q8_wsc8;
using krk::build_q8_0_wscales;
using krk::build_q8_0_xscales;

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

static double now_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}

static __m512i g_sink = {};
__attribute__((target("avx512f,avx512bw"))) static int sink_value() {
    return _mm512_reduce_add_epi32(g_sink);
}
static unsigned char *g_w = nullptr;
static signed char *g_a = nullptr;

// One loop iteration is one block pair: `rows` rows, 64 weights each, so
// rows * 64 MACs, which is exactly what the Q8_0 kernel does per iteration.
__attribute__((target("avx512f,avx512bw,avx512vl"))) static double run(int mode,
                                                                     int rows,
                                                                     f32 scale) {
    __m512i acc[8];
    __m512 facc[8];
    const __m512 dv = _mm512_set1_ps(scale);  // runtime: cannot be folded away
    const __m512i one16 = _mm512_set1_epi16(1);
    for (int r = 0; r < 8; r++) {
        acc[r] = _mm512_setzero_si512();
        facc[r] = _mm512_setzero_ps();
    }
    constexpr long kIters = 60000;
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        for (long i = 0; i < kIters; i++) {
            // Walk two L1-resident buffers so the loads keep their real shape.
            const unsigned char *w = g_w + ((i * 544) & 8191);
            const signed char *a = g_a + ((i * 256) & 4095);
            for (int r = 0; r < rows; r++) {
                __m512i wv, av;
                if (mode >= 2) {
                    wv = _mm512_loadu_si512(w + r * 68);
                    av = _mm512_loadu_si512(a + r * 32);
                } else {
                    // Bumped by i: a loop-invariant operand would be hoisted
                    // out of the loop and the ceiling would come out double.
                    wv = _mm512_set1_epi8(static_cast<char>(i + r));
                    av = _mm512_set1_epi8(static_cast<char>(2 + (i & 1)));
                }
                const __m512i p16 = _mm512_maddubs_epi16(av, wv);
                const __m512i p32 = _mm512_madd_epi16(p16, one16);
                if (mode == 0)
                    acc[r] = _mm512_add_epi32(acc[r], p32);
                else
                    facc[r] = _mm512_fmadd_ps(dv, _mm512_cvtepi32_ps(p32), facc[r]);
            }
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        volatile double sink = 0.0;
        for (int r = 0; r < 8; r++)
            sink += (mode == 0 ? static_cast<double>(_mm512_reduce_add_epi32(acc[r]))
                               : static_cast<double>(_mm512_reduce_add_ps(facc[r])));
        (void)sink;
    }
    return static_cast<double>(kIters) * static_cast<double>(rows) * 64.0 / best / 1e9;
}

// The ceiling for the instruction mix this engine actually ships. A-D all
// measure the maddubs + madd pair, which needs TWO instructions per 64 MACs;
// the production q8_0 kernel uses vpdpbusd, which needs ONE. Comparing the
// production kernel against a maddubs ceiling therefore flatters it, and the
// first run of this tool did exactly that (E read 105% of C, which cannot be a
// fraction of anything meaningful).
//
// Same body, same guard against hoisting: operands walk two L1-resident
// buffers, each bumped by the iteration counter, the scale vector is built from
// a runtime value, and the sink is volatile.
__attribute__((target("avx512f,avx512bw,avx512vnni"))) static double run_vnni(
    int rows, f32 scale, int tail) {
    __m512i iacc[8];
    __m512 facc[8];
    const __m512 dv = _mm512_set1_ps(scale);
    for (int r = 0; r < 8; r++) {
        iacc[r] = _mm512_setzero_si512();
        facc[r] = _mm512_setzero_ps();
    }
    constexpr long kIters = 60000;
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        for (long i = 0; i < kIters; i++) {
            const unsigned char *w = g_w + ((i * 544) & 8191);
            const signed char *a = g_a + ((i * 256) & 4095);
            for (int r = 0; r < rows; r++) {
                const __m512i wv = _mm512_loadu_si512(w + r * 68);
                const __m512i av = _mm512_loadu_si512(a + r * 32);
                const __m512i p32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                                        wv, av);
                if (tail)
                    facc[r] = _mm512_fmadd_ps(dv, _mm512_cvtepi32_ps(p32),
                                              facc[r]);
                else
                    iacc[r] = _mm512_add_epi32(iacc[r], p32);
            }
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        volatile double sink = 0.0;
        for (int r = 0; r < 8; r++)
            sink += tail ? static_cast<double>(_mm512_reduce_add_ps(facc[r]))
                         : static_cast<double>(_mm512_reduce_add_epi32(iacc[r]));
        (void)sink;
    }
    return static_cast<double>(kIters) * static_cast<double>(rows) * 64.0 /
           best / 1e9;
}

// The shipped kernel on a realistically shaped problem, measured in the same
// process as the ceiling above: a 1.67 MB weight matrix (n_out x 816 B for 768
// inputs) and 512 activation rows. Reporting the two together is the only way
// the ratio means anything -- this host's absolute throughput drifts by up to
// 1.5x between processes.
static double shipped(i64 n_in, i64 n_out, i64 rows) {
    const i64 nb = n_in / 32;
    const i64 row_bytes = nb * 34;
    std::vector<u8> w(static_cast<size_t>(n_out * row_bytes));
    u32 st = 12345;
    for (auto &b : w) {
        st = st * 1664525u + 1013904223u;
        b = static_cast<u8>(st >> 24);
    }
    // Half-precision scales that are legal Q8_0 values, not zeros.
    for (i64 o = 0; o < n_out; o++)
        for (i64 b = 0; b < nb; b++) {
            const u16 d = static_cast<u16>(0x2000 + ((o * 31 + b * 7) & 0x1fff));
            w[static_cast<size_t>(o * row_bytes + b * 34)] = static_cast<u8>(d & 0xff);
            w[static_cast<size_t>(o * row_bytes + b * 34 + 1)] = static_cast<u8>(d >> 8);
        }
    std::vector<f32> xf(static_cast<size_t>(rows * n_in));
    for (auto &v : xf) {
        st = st * 1664525u + 1013904223u;
        v = static_cast<f32>(st >> 8) / 16777216.0f * 6.0f - 3.0f;
    }
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    std::vector<f32> xd(static_cast<size_t>(rows * nb));
    for (i64 r = 0; r < rows; r++)
        quantize_q8_row(xf.data() + r * n_in, n_in, xq.data() + r * n_in,
                        xd.data() + r * nb);
    std::vector<f32> out(static_cast<size_t>(rows * n_out));

    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        for (i64 r = 0; r + 8 <= rows; r += 8)
            for (i64 o = 0; o < n_out; o++)
                dot_q8_0_q8_x8(w.data() + static_cast<size_t>(o) * row_bytes,
                               xq.data() + r * n_in, xd.data() + r * nb, n_in,
                               n_in, nb, out.data() + r * n_out + o, n_out);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double macs = static_cast<double>(rows / 8 * 8) *
                        static_cast<double>(n_out) * static_cast<double>(n_in);
    volatile double sink = out[0];
    (void)sink;
    return macs / best / 1e9;
}

// The repack the uop count asks for, applied on the axis that is legal: both
// scale factors are EXPANDED to the 16 lanes that use them at load time, so the
// inner loop has no broadcast, no permute and no per-row scale build at all.
//   ws[o][p]  = [d0 x8 | d1 x8] for output row o, block pair p  (64 B)
//   sx[r][p]  = [e0 x8 | e1 x8] for activation row r, pair p     (64 B)
// The inner loop is then load, vpdpbusd, vcvtdq2ps, vmulps, vfmadd -- five uops
// per 64 MACs, the same shape as the V1 ceiling.
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c"))) static double
tiled_expanded(i64 n_in, i64 n_out, i64 rows) {
    const i64 nb = n_in / 32;
    const i64 row_bytes = nb * 34;
    std::vector<u8> w(static_cast<size_t>(n_out * row_bytes));
    u32 st = 12345;
    for (auto &b : w) {
        st = st * 1664525u + 1013904223u;
        b = static_cast<u8>(st >> 24);
    }
    for (i64 o = 0; o < n_out; o++)
        for (i64 b = 0; b < nb; b++) {
            const u16 d = static_cast<u16>(0x2000 + ((o * 31 + b * 7) & 0x1fff));
            w[static_cast<size_t>(o * row_bytes + b * 34)] = static_cast<u8>(d & 0xff);
            w[static_cast<size_t>(o * row_bytes + b * 34 + 1)] = static_cast<u8>(d >> 8);
        }
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    for (auto &v : xq) {
        st = st * 1664525u + 1013904223u;
        v = static_cast<i8>(st >> 24);
    }
    const i64 npair = nb / 2;
    const __m512i k_pair =
        _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    // Load-time repack: fold the pair of scales straight into the shape the
    // FMA wants. Two multiplies per pair instead of two per (row, pair).
    std::vector<f32> ws(static_cast<size_t>(n_out * npair * 16));
    std::vector<f32> sw(static_cast<size_t>(n_out * npair * 16));
    build_q8_0_wscales(w.data(), row_bytes, n_out, n_in, sw.data());
    for (i64 o = 0; o < n_out; o++)
        for (i64 p = 0; p < npair; p++) {
            f32 *dst = ws.data() + (o * npair + p) * 16;
            const f32 *src = sw.data() + (o * npair + p) * 16;  // [d0 x8|d1 x8]
            for (int i = 0; i < 8; i++) dst[i] = src[0];
            for (int i = 8; i < 16; i++) dst[i] = src[8];
        }
    std::vector<f32> sx(static_cast<size_t>(rows * npair * 16));
    for (i64 r = 0; r < rows; r++)
        for (i64 p = 0; p < npair; p++) {
            f32 *dst = sx.data() + (r * npair + p) * 16;
            const f32 e0 = 0.011f + 0.001f * static_cast<f32>(p & 7);
            const f32 e1 = 0.013f + 0.001f * static_cast<f32>(p & 3);
            for (int i = 0; i < 8; i++) dst[i] = e0;
            for (int i = 8; i < 16; i++) dst[i] = e1;
        }
    std::vector<f32> out(static_cast<size_t>(rows * n_out));
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        for (i64 r = 0; r + 8 <= rows; r += 8)
            for (i64 o = 0; o < n_out; o++) {
                const u8 *wr = w.data() + static_cast<size_t>(o) * row_bytes;
                const f32 *wsr = ws.data() + o * npair * 16;
                const f32 *sxr = sx.data() + r * npair * 16;
                __m512 acc[8];
                for (int i = 0; i < 8; i++) acc[i] = _mm512_setzero_ps();
                const u8 *wp = wr;
                for (i64 p = 0; p < npair; p++, wp += 68) {
                    const __m256i lo = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(wp + 2));
                    const __m256i hi = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(wp + 36));
                    const __m512i wv =
                        _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                    const __mmask64 neg = _mm512_cmp_epi8_mask(wv, _mm512_setzero_si512(),
                                                               _MM_CMPINT_LT);
                    const __m512i mag = _mm512_mask_sub_epi8(
                        wv, neg, _mm512_setzero_si512(), wv);
                    const __m512 wsv =
                        _mm512_loadu_ps(wsr + p * 16);
                    const f32 *sxp = sxr + p * 16;
                    for (int i = 0; i < 8; i++) {
                        const __m512i xv = _mm512_loadu_si512(
                            reinterpret_cast<const void *>(
                                xq.data() + (r + i) * n_in + p * 64));
                        const __m512i xb = _mm512_mask_sub_epi8(
                            xv, neg, _mm512_setzero_si512(), xv);
                        const __m512i iv = _mm512_dpbusd_epi32(
                            _mm512_setzero_si512(), mag, xb);
                        acc[i] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(iv),
                            _mm512_mul_ps(wsv, _mm512_loadu_ps(sxp)), acc[i]);
                    }
                }
                for (int i = 0; i < 8; i++)
                    out[static_cast<size_t>((r + i) * n_out + o)] =
                        _mm512_reduce_add_ps(acc[i]);
            }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double macs = static_cast<double>(rows / 8 * 8) *
                        static_cast<double>(n_out) * static_cast<double>(n_in);
    volatile double sink = out[0];
    (void)sink;
    return macs / best / 1e9;
}

__attribute__((target("avx512f,avx512bw,avx512vnni,f16c"))) static double
tiled_expanded16(i64 n_in, i64 n_out, i64 rows) {
    const i64 nb = n_in / 32;
    const i64 row_bytes = nb * 34;
    std::vector<u8> w(static_cast<size_t>(n_out * row_bytes));
    u32 st = 12345;
    for (auto &b : w) {
        st = st * 1664525u + 1013904223u;
        b = static_cast<u8>(st >> 24);
    }
    for (i64 o = 0; o < n_out; o++)
        for (i64 b = 0; b < nb; b++) {
            const u16 d = static_cast<u16>(0x2000 + ((o * 31 + b * 7) & 0x1fff));
            w[static_cast<size_t>(o * row_bytes + b * 34)] = static_cast<u8>(d & 0xff);
            w[static_cast<size_t>(o * row_bytes + b * 34 + 1)] = static_cast<u8>(d >> 8);
        }
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    for (auto &v : xq) {
        st = st * 1664525u + 1013904223u;
        v = static_cast<i8>(st >> 24);
    }
    const i64 npair = nb / 2;
    const __m512i k_pair =
        _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    // Load-time repack: fold the pair of scales straight into the shape the
    // FMA wants. Two multiplies per pair instead of two per (row, pair).
    std::vector<u16> ws(static_cast<size_t>(n_out * npair * 16));
    std::vector<f32> sw(static_cast<size_t>(n_out * npair * 16));
    build_q8_0_wscales(w.data(), row_bytes, n_out, n_in, sw.data());
    for (i64 o = 0; o < n_out; o++)
        for (i64 p = 0; p < npair; p++) {
            u16 *dst = ws.data() + (o * npair + p) * 16;
            const f32 *src = sw.data() + (o * npair + p) * 16;  // [d0 x8|d1 x8]
            for (int i = 0; i < 8; i++) dst[i] = krk::fp32_to_fp16(src[0]);
            for (int i = 8; i < 16; i++) dst[i] = krk::fp32_to_fp16(src[8]);
        }
    std::vector<f32> sx(static_cast<size_t>(rows * npair * 16));
    for (i64 r = 0; r < rows; r++)
        for (i64 p = 0; p < npair; p++) {
            f32 *dst = sx.data() + (r * npair + p) * 16;
            const f32 e0 = 0.011f + 0.001f * static_cast<f32>(p & 7);
            const f32 e1 = 0.013f + 0.001f * static_cast<f32>(p & 3);
            for (int i = 0; i < 8; i++) dst[i] = e0;
            for (int i = 8; i < 16; i++) dst[i] = e1;
        }
    std::vector<f32> out(static_cast<size_t>(rows * n_out));
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        for (i64 r = 0; r + 8 <= rows; r += 8)
            for (i64 o = 0; o < n_out; o++) {
                const u8 *wr = w.data() + static_cast<size_t>(o) * row_bytes;
                const u16 *wsr = ws.data() + o * npair * 16;
                const f32 *sxr = sx.data() + r * npair * 16;
                __m512 acc[8];
                for (int i = 0; i < 8; i++) acc[i] = _mm512_setzero_ps();
                const u8 *wp = wr;
                for (i64 p = 0; p < npair; p++, wp += 68) {
                    const __m256i lo = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(wp + 2));
                    const __m256i hi = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(wp + 36));
                    const __m512i wv =
                        _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                    const __mmask64 neg = _mm512_cmp_epi8_mask(wv, _mm512_setzero_si512(),
                                                               _MM_CMPINT_LT);
                    const __m512i mag = _mm512_mask_sub_epi8(
                        wv, neg, _mm512_setzero_si512(), wv);
                    const __m512 wsv = _mm512_cvtph_ps(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(wsr + p * 16)));
                    const f32 *sxp = sxr + p * 16;
                    for (int i = 0; i < 8; i++) {
                        const __m512i xv = _mm512_loadu_si512(
                            reinterpret_cast<const void *>(
                                xq.data() + (r + i) * n_in + p * 64));
                        const __m512i xb = _mm512_mask_sub_epi8(
                            xv, neg, _mm512_setzero_si512(), xv);
                        const __m512i iv = _mm512_dpbusd_epi32(
                            _mm512_setzero_si512(), mag, xb);
                        acc[i] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(iv),
                            _mm512_mul_ps(wsv, _mm512_loadu_ps(sxp)), acc[i]);
                    }
                }
                for (int i = 0; i < 8; i++)
                    out[static_cast<size_t>((r + i) * n_out + o)] =
                        _mm512_reduce_add_ps(acc[i]);
            }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double macs = static_cast<double>(rows / 8 * 8) *
                        static_cast<double>(n_out) * static_cast<double>(n_in);
    volatile double sink = out[0];
    (void)sink;
    return macs / best / 1e9;
}

// The kernel the GEMM path ACTUALLY ships: the eight-row tile fed from a
// per-OUTPUT-ROW f32 scale array (build_q8_0_wscales + dot_q8_0_q8_wsc8), which
// is what backend_cpu.cpp calls. `shipped()` below still drives
// dot_q8_0_q8_x8, the kernel that predates the scale array.
//
// An earlier arm of this tool built ONE table per row tile from row 0 and reused
// it for every output row, which is wrong for any matrix whose rows have
// different q8_0 block scales -- measured up to 83% relative error. It reported
// a flattering number for a kernel that could not be shipped.
static double shipped_tab(i64 n_in, i64 n_out, i64 rows) {
    const i64 nb = n_in / 32;
    const i64 row_bytes = nb * 34;
    std::vector<u8> w(static_cast<size_t>(n_out * row_bytes));
    u32 st = 12345;
    for (auto &b : w) {
        st = st * 1664525u + 1013904223u;
        b = static_cast<u8>(st >> 24);
    }
    for (i64 o = 0; o < n_out; o++)
        for (i64 b = 0; b < nb; b++) {
            const u16 d = static_cast<u16>(0x2000 + ((o * 31 + b * 7) & 0x1fff));
            w[static_cast<size_t>(o * row_bytes + b * 34)] = static_cast<u8>(d & 0xff);
            w[static_cast<size_t>(o * row_bytes + b * 34 + 1)] = static_cast<u8>(d >> 8);
        }
    std::vector<f32> xf(static_cast<size_t>(rows * n_in));
    for (auto &v : xf) {
        st = st * 1664525u + 1013904223u;
        v = static_cast<f32>(st >> 8) / 16777216.0f * 6.0f - 3.0f;
    }
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    std::vector<f32> xd(static_cast<size_t>(rows * nb));
    for (i64 r = 0; r < rows; r++)
        quantize_q8_row(xf.data() + r * n_in, n_in, xq.data() + r * n_in,
                        xd.data() + r * nb);
    std::vector<f32> out(static_cast<size_t>(rows * n_out));
    const i64 npair = (n_in / 32) / 2;
    std::vector<f32> wsc(static_cast<size_t>(n_out * npair * 16));
    build_q8_0_wscales(w.data(), row_bytes, n_out, n_in, wsc.data());
    std::vector<f32> xsc(static_cast<size_t>(8 * npair * 16));
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        for (i64 r = 0; r + 8 <= rows; r += 8) {
            build_q8_0_xscales(xd.data() + r * nb, nb, 8, n_in, xsc.data());
            for (i64 o = 0; o < n_out; o++)
                dot_q8_0_q8_wsc8(w.data() + static_cast<size_t>(o) * row_bytes,
                                 xq.data() + r * n_in, xd.data() + r * nb, n_in,
                                 n_in, nb, out.data() + r * n_out + o, n_out,
                                 wsc.data() + o * npair * 16, xsc.data());
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double macs = static_cast<double>(rows / 8 * 8) *
                        static_cast<double>(n_out) * static_cast<double>(n_in);
    volatile double sink = out[0];
    (void)sink;
    return macs / best / 1e9;
}

// The shipped kernel body over the same 1.67 MB matrix, with the per-block
// scale load and its dependency chain removed but the scale itself still a
// runtime value, so nothing can be hoisted or folded away. F vs E is the cost
// of that chain; F vs C is what is left over once it is gone.
__attribute__((target("avx512f,avx512bw"))) static double no_scale_chain(
    i64 n_in, i64 n_out, i64 rows, f32 dv) {
    const i64 nb = n_in / 32;
    const i64 row_bytes = nb * 34;
    std::vector<u8> w(static_cast<size_t>(n_out * row_bytes));
    u32 st = 999;
    for (auto &b : w) {
        st = st * 1664525u + 1013904223u;
        b = static_cast<u8>(st >> 24);
    }
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    for (auto &v : xq) {
        st = st * 1664525u + 1013904223u;
        v = static_cast<i8>(st >> 24);
    }
    std::vector<f32> out(static_cast<size_t>(rows * n_out));
    const __m512 dvec = _mm512_set1_ps(dv);
    const __m512i one16 = _mm512_set1_epi16(1);
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        for (i64 r = 0; r + 8 <= rows; r += 8) {
            const i8 *x[8];
            for (int k = 0; k < 8; k++) x[k] = xq.data() + (r + k) * n_in;
            for (i64 o = 0; o < n_out; o++) {
                const u8 *wr = w.data() + static_cast<size_t>(o) * row_bytes;
                __m512 acc[8];
                for (int k = 0; k < 8; k++) acc[k] = _mm512_setzero_ps();
                for (i64 b = 0; b < nb; b++) {
                    for (int k = 0; k < 8; k++) {
                        const __m512i wv = _mm512_loadu_si512(wr + b * 34 + 2);
                        const __m512i av =
                            _mm512_loadu_si512(x[k] + b * 32);
                        const __m512i p16 = _mm512_maddubs_epi16(av, wv);
                        const __m512i p32 = _mm512_madd_epi16(p16, one16);
                        acc[k] = _mm512_fmadd_ps(dvec, _mm512_cvtepi32_ps(p32),
                                                 acc[k]);
                    }
                }
                f32 sum[8];
                for (int k = 0; k < 8; k++) sum[k] = _mm512_reduce_add_ps(acc[k]);
                out[r * n_out + o] = sum[0] + sum[1] + sum[2] + sum[3] +
                                     sum[4] + sum[5] + sum[6] + sum[7];
            }
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    volatile double sink = out[0];
    (void)sink;
    return static_cast<double>(rows / 8 * 8) * static_cast<double>(n_out) *
           static_cast<double>(n_in) / best / 1e9;
}

// Exactly the loop the in-engine roofline probe runs, reproduced so it can be
// compared against A in the same process. The probe reports several times what
// A does for the same eight maddubs per iteration, and one of the two numbers
// is wrong. This variant is the arbitration.
__attribute__((target("avx512f,avx512bw"))) static double probe_shape() {
    constexpr int kIters = 20000;
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        __m512i a = _mm512_set1_epi8(3);
        const __m512i bump = _mm512_set1_epi8(1);
        const __m512i ones = _mm512_set1_epi16(1);
        __m512i s0 = _mm512_setzero_si512(), s1 = _mm512_setzero_si512();
        __m512i s2 = _mm512_setzero_si512(), s3 = _mm512_setzero_si512();
        __m512i s4 = _mm512_setzero_si512(), s5 = _mm512_setzero_si512();
        __m512i s6 = _mm512_setzero_si512(), s7 = _mm512_setzero_si512();
        const double t0 = now_s();
        for (int i = 0; i < kIters; i++) {
            a = _mm512_add_epi8(a, bump);
            s0 = _mm512_add_epi32(s0, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(2)), ones));
            s1 = _mm512_add_epi32(s1, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(3)), ones));
            s2 = _mm512_add_epi32(s2, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(4)), ones));
            s3 = _mm512_add_epi32(s3, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(5)), ones));
            s4 = _mm512_add_epi32(s4, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(6)), ones));
            s5 = _mm512_add_epi32(s5, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(7)), ones));
            s6 = _mm512_add_epi32(s6, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(8)), ones));
            s7 = _mm512_add_epi32(s7, _mm512_madd_epi16(_mm512_maddubs_epi16(a, _mm512_set1_epi8(9)), ones));
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        g_sink += s0;
        g_sink = _mm512_add_epi32(g_sink, _mm512_add_epi32(
                  _mm512_add_epi32(s1, s2), _mm512_add_epi32(
                  _mm512_add_epi32(s3, s4), _mm512_add_epi32(s5, _mm512_add_epi32(s6, s7)))));
    }
    return static_cast<double>(kIters) * 8.0 * 64.0 / best / 1e9;
}

int main(int argc, char **argv) {
    const f32 scale = argc > 1 ? static_cast<f32>(std::atof(argv[1])) : 0.0123f;
    g_w = static_cast<unsigned char *>(std::aligned_alloc(64, 16384));
    g_a = static_cast<signed char *>(std::aligned_alloc(64, 8192));
    for (int i = 0; i < 16384; i++) g_w[i] = static_cast<unsigned char>(i * 7 + 3);
    for (int i = 0; i < 8192; i++) g_a[i] = static_cast<signed char>((i * 5) % 255 - 127);

    std::printf("%-42s %7s %7s\n", "variant", "GMAC/s", "of C");
    const double c = run(2, 8, scale);
    const double a = run(0, 8, scale);
    const double b = run(1, 8, scale);
    const double d = run(2, 4, scale);
    std::printf("%-42s %7.2f %6.0f%%\n", "A  regs, integer accumulate", a, 100.0 * a / c);
    std::printf("%-42s %7.2f %6.0f%%\n", "B  regs, + int->float tail", b, 100.0 * b / c);
    std::printf("%-42s %7.2f %6.0f%%\n", "C  loads, + int->float tail", c, 100.0);
    std::printf("%-42s %7.2f %6.0f%%\n", "D  as C, four rows", d, 100.0 * d / c);
    std::printf("%-42s %7.2f %6.0f%%  <- the in-engine probe's loop shape\n",
                "P  probe shape (one operand, eight multipliers)", probe_shape(),
                100.0 * probe_shape() / c);
    if (__builtin_cpu_supports("avx512vnni")) {
        const double v0 = run_vnni(8, scale, 0);
        const double v1 = run_vnni(8, scale, 1);
        std::printf("%-42s %7.2f %6.0f%%  <- vpdpbusd, integer only\n",
                    "V0 VNNI registers only", v0, 100.0 * v0 / c);
        std::printf("%-42s %7.2f %6.0f%%  <- the mix the engine ships\n",
                    "V1 VNNI + loads + int->float tail", v1, 100.0 * v1 / c);
        const double e0 = shipped_tab(768, 2048, 512);
        std::printf("%-42s %7.2f %6.0f%%  <- SHIPPED GEMM as a fraction of V1\n",
                    "T  production kernel, of V1", e0, 100.0 * e0 / v1);
    } else {
        std::printf("%-42s    n/a  (no avx512vnni on this host)\n",
                    "V  VNNI ceiling");
    }
    const double f = no_scale_chain(768, 2048, 512, scale);
    std::printf("%-42s %7.2f %6.0f%%  <- same body, no scale chain\n",
                "F  E without the per-block scale chain", f, 100.0 * f / c);
    const double e = shipped(768, 2048, 512);
    std::printf("%-42s %7.2f %6.0f%%  <- dot_q8_0_q8_x8, pre-table\n",
                "E  kernel before the scale table", e, 100.0 * e / c);
    const double et = shipped_tab(768, 2048, 512);
    std::printf("%-42s %7.2f %6.0f%%  <- what the GEMM path ships now\n",
                "T  production kernel WITH the scale array", et,
                100.0 * et / c);
    const double wb = tiled_expanded(768, 2048, 512);
    const double wb2 = tiled_expanded16(768, 2048, 512);
    std::printf("%-42s %7.2f %6.0f%%\n", "W2 repacked, f16 scales (half traffic)",
                wb2, 100.0 * wb2 / c);
    std::printf("%-42s %7.2f %6.0f%%  <- expanded scales, no per-row scale build\n",
                "W  repacked scale array", wb, 100.0 * wb / c);
    std::printf("%-42s %7.2f %6.0f%%\n", "   W over T (same process)", wb / et,
                100.0 * wb / et);
    std::printf("  (probe sink %d)\n", sink_value());
    std::free(g_w);
    std::free(g_a);
    return 0;
}
