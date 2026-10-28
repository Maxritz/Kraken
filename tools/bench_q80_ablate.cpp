// Which part of the Q8_0 kernel costs what.
//
// The shipped kernel measured 29-53% of `bench_isa_ceiling`'s variant C (the
// same instruction mix with loads and the real float tail). Two structural
// explanations were tested and both came back null: block/column traffic
// (already blocked, worth 1.12x) and the per-block scale chain (three rewrites,
// all neutral or worse). So the next question is not "what structure should
// change" but "which instruction group is actually burning the cycles".
//
// This tool answers that by ablation. Every rung is the production kernel with
// exactly one component changed and everything else byte-for-byte the same, and
// all rungs run in one process against the same data, because this host's
// absolute throughput moves by up to 1.8x between processes.
//
// Rungs marked WRONG produce a different answer on purpose. They exist to price
// an instruction, not to be shippable, and they are never used to compute a
// result anything else consumes.
#include <immintrin.h>

#include "krk/common.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

using krk::f32;
using krk::f64;
using krk::i64;
using krk::u8;
using krk::i8;
using krk::u16;
using krk::u32;

static double now_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}

// One rung. MODE selects the single component under test:
//   0  production (reference)
//   1  activation scale is a constant            (drop dx load + d0*d1)
//   2  weight scale is a constant                (drop fp16 decode)
//   3  weights load as one 512-bit access        (drop 2x256 + insert)
//   4  sign mask is a constant                   (drop cmp_epi8_mask)
//   5  activations are not flipped per block     (drop mask_sub on x)   WRONG
//   6  weight magnitudes are not formed          (drop mask_sub on w)   WRONG
//   7  the float tail is an integer accumulate   (drop cvt + fmadd)     WRONG
template <int MODE>
__attribute__((target("avx512f,avx512bw"))) static void rung(const u8 *w,
                                                             const i8 *xq,
                                                             const f32 *dx,
                                                             i64 nb, i64 n_out,
                                                             i64 rows,
                                                             i64 row_bytes,
                                                             f32 scale,
                                                             __mmask64 kmask,
                                                             f32 *out) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    for (i64 r0 = 0; r0 + 8 <= rows; r0 += 8) {
        for (i64 o = 0; o < n_out; o++) {
            const u8 *wr = w + static_cast<size_t>(o) * static_cast<size_t>(row_bytes);
            __m512 acc[8];
            __m512i acci[8];
            for (int r = 0; r < 8; r++) {
                acc[r] = _mm512_setzero_ps();
                acci[r] = _mm512_setzero_si512();
            }
            for (i64 b = 0; b + 2 <= nb; b += 2, wr += 68) {
                f32 d0, d1;
                if (MODE == 2) {
                    d0 = d1 = scale;
                } else {
                    d0 = krk::fp16_to_fp32(
                        static_cast<u16>(wr[0] | (static_cast<u16>(wr[1]) << 8)));
                    d1 = krk::fp16_to_fp32(
                        static_cast<u16>(wr[34] | (static_cast<u16>(wr[35]) << 8)));
                }
                __m512i wv;
                if (MODE == 3) {
                    // Contiguous 64 bytes from offset 2. Not the real layout --
                    // it reads into the next block's header -- but it isolates
                    // the cost of two 256-bit loads plus the lane insert.
                    wv = _mm512_loadu_si512(reinterpret_cast<const void *>(wr + 2));
                } else {
                    const __m256i lo =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wr + 2));
                    const __m256i hi =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wr + 36));
                    wv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                }
                const __mmask64 neg =
                    (MODE == 4) ? kmask : _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
                const __m512i mag =
                    (MODE == 6) ? wv : _mm512_mask_sub_epi8(wv, neg, zero, wv);
                for (int r = 0; r < 8; r++) {
                    const __m512i xv = _mm512_loadu_si512(
                        reinterpret_cast<const void *>(
                            xq + static_cast<i64>(r0 + r) * static_cast<i64>(nb * 32) +
                            b * 32));
                    const __m512i xb =
                        (MODE == 5) ? xv : _mm512_mask_sub_epi8(xv, neg, zero, xv);
                    const __m512i iv =
                        _mm512_madd_epi16(_mm512_maddubs_epi16(mag, xb), ones);
                    if (MODE == 7) {
                        acci[r] = _mm512_add_epi32(acci[r], iv);
                    } else {
                        f32 s0, s1;
                        if (MODE == 1) {
                            s0 = s1 = scale;
                        } else {
                            s0 = d0 * dx[(r0 + r) * nb + b];
                            s1 = d1 * dx[(r0 + r) * nb + b + 1];
                        }
                        const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
                            _mm512_castps_si512(
                                _mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
                            _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
                        acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
                    }
                }
            }
            f32 sum = 0.0f;
            if (MODE == 7) {
                for (int r = 0; r < 8; r++)
                    sum += static_cast<f32>(_mm512_reduce_add_epi32(acci[r]));
            } else {
                for (int r = 0; r < 8; r++) sum += _mm512_reduce_add_ps(acc[r]);
            }
            out[(r0 + 0) * n_out + o] = sum;
        }
    }
}


// ---------------------------------------------------------------------------
// Row depth as a parameter.
//
// At 38.7 GMAC/s across two cores the production kernel retires roughly 10
// MACs per cycle against ~34 for the instruction mix on its own, i.e. under
// 1 IPC. That is the signature of dependency stalls rather than instruction
// count, and the ablation below confirms it from the other side: deleting the
// single largest instruction group buys 24%, not the 2-3x a dispatch-bound loop
// would give. Two tiles of eight rows share one weight decode, which both
// halves the weight traffic per MAC and doubles the number of independent
// chains the out-of-order engine can find.
template <int NROWS>
__attribute__((target("avx512f,avx512bw,f16c"))) static void rung_rows(
    const u8 *w, const i8 *xq, const f32 *dx, i64 nb, i64 n_out, i64 rows,
    i64 row_bytes, f32 *out) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    for (i64 r0 = 0; r0 + NROWS <= rows; r0 += NROWS) {
        for (i64 o = 0; o < n_out; o++) {
            const u8 *wr = w + static_cast<size_t>(o) * static_cast<size_t>(row_bytes);
            __m512 acc[NROWS];
            for (int r = 0; r < NROWS; r++) acc[r] = _mm512_setzero_ps();
            for (i64 b = 0; b + 2 <= nb; b += 2, wr += 68) {
                const __m128i two = _mm_setr_epi16(
                    static_cast<short>(wr[0] | (static_cast<u16>(wr[1]) << 8)),
                    static_cast<short>(wr[34] | (static_cast<u16>(wr[35]) << 8)),
                    0, 0, 0, 0, 0, 0);
                const __m128 d2 = _mm_cvtph_ps(two);
                const f32 d0 = _mm_cvtss_f32(d2);
                const f32 d1 = _mm_cvtss_f32(_mm_shuffle_ps(d2, d2, 0x55));
                const __m256i lo =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wr + 2));
                const __m256i hi =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wr + 36));
                const __m512i wv =
                    _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
                const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
                for (int r = 0; r < NROWS; r++) {
                    const __m512i xv = _mm512_loadu_si512(
                        reinterpret_cast<const void *>(
                            xq + static_cast<i64>(r0 + r) * static_cast<i64>(nb * 32) +
                            b * 32));
                    const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
                    const __m512i iv =
                        _mm512_madd_epi16(_mm512_maddubs_epi16(mag, xb), ones);
                    const f32 s0 = d0 * dx[(r0 + r) * nb + b];
                    const f32 s1 = d1 * dx[(r0 + r) * nb + b + 1];
                    const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
                        _mm512_castps_si512(
                            _mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
                        _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
                    acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
                }
            }
            f32 sum = 0.0f;
            for (int r = 0; r < NROWS; r++) sum += _mm512_reduce_add_ps(acc[r]);
            out[r0 * n_out + o] = sum;
        }
    }
}

template <int NROWS>
static double time_rows(const u8 *w, const i8 *xq, const f32 *dx, i64 nb,
                        i64 n_out, i64 rows, i64 row_bytes, f32 *out) {
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        rung_rows<NROWS>(w, xq, dx, nb, n_out, rows, row_bytes, out);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double macs = static_cast<double>(rows / NROWS * NROWS) *
                        static_cast<double>(n_out) * static_cast<double>(nb) * 32.0;
    return macs / best / 1e9;
}

struct Case {
    const char *name;
    int mode;
    bool wrong;
};

static double time_rung(const Case &c, const u8 *w, const i8 *xq, const f32 *dx,
                        i64 nb, i64 n_out, i64 rows, i64 row_bytes, f32 scale,
                        __mmask64 kmask, f32 *out) {
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = now_s();
        switch (c.mode) {
            case 0: rung<0>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
            case 1: rung<1>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
            case 2: rung<2>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
            case 3: rung<3>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
            case 4: rung<4>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
            case 5: rung<5>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
            case 6: rung<6>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
            default: rung<7>(w, xq, dx, nb, n_out, rows, row_bytes, scale, kmask, out); break;
        }
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double macs = static_cast<double>(rows / 8 * 8) * static_cast<double>(n_out) *
                        static_cast<double>(nb) * 32.0;
    return macs / best / 1e9;
}

int main(int argc, char **argv) {
    const i64 n_in = argc > 1 ? std::atoll(argv[1]) : 768;
    const i64 n_out = argc > 2 ? std::atoll(argv[2]) : 2048;
    const i64 rows = argc > 3 ? std::atoll(argv[3]) : 512;
    const i64 nb = n_in / 32;
    const i64 row_bytes = nb * 34;

    // Deterministic data: legal fp16 scales, full-range signed weight bytes,
    // and full-range activations. Nothing is left zero, because a zero operand
    // makes several of these instructions free and flatters the rung.
    u32 st = 20250501u;
    auto rnd = [&] {
        st = st * 1664525u + 1013904223u;
        return static_cast<u32>(st >> 8);
    };
    std::vector<u8> w(static_cast<size_t>(n_out * row_bytes));
    for (i64 o = 0; o < n_out; o++) {
        for (i64 b = 0; b < nb; b++) {
            const u16 h = static_cast<u16>(0x2000 + (rnd() & 0x1fff));
            u8 *blk = w.data() + o * row_bytes + b * 34;
            blk[0] = static_cast<u8>(h & 0xff);
            blk[1] = static_cast<u8>(h >> 8);
            for (i64 i = 0; i < 32; i++) blk[2 + i] = static_cast<u8>(rnd() >> 24);
        }
    }
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    for (auto &v : xq) v = static_cast<i8>(static_cast<int>(rnd() >> 24) - 128);
    std::vector<f32> dx(static_cast<size_t>(rows * nb));
    for (auto &v : dx) v = 0.001f + static_cast<f32>(rnd() & 0xffff) / 65536.0f * 0.02f;
    std::vector<f32> out(static_cast<size_t>(rows * n_out));

    const f32 scale = 0.0123f + static_cast<f32>(rnd() & 0xff) / 65536.0f;
    const __mmask64 kmask = static_cast<__mmask64>(0xa5a5a5a5a5a5a5a5ULL) ^
                            (static_cast<__mmask64>(rnd() & 0xff) << 8);

    static const Case cases[] = {
        {"L0 production", 0, false},
        {"L1 constant act scale", 1, true},
        {"L2 constant w scale", 2, true},
        {"L3 contiguous w load", 3, true},
        {"L4 constant sign mask", 4, true},
        {"L5 no per-block x flip", 5, true},
        {"L6 no weight magnitude", 6, true},
        {"L7 integer accumulate", 7, true},
    };
    // Warm the pages in first: the first rung otherwise pays for the faults of
    // a 1.7 MB weight matrix and a 400 KB activation tile.
    rung<0>(w.data(), xq.data(), dx.data(), nb, n_out, rows, row_bytes, scale, kmask,
            out.data());

    std::printf("Q8_0 kernel ablation, %lldx%lldx%lld, one thread\n", n_in, n_out,
                rows);
    std::printf("%-26s %8s %7s %9s %9s\n", "rung", "GMAC/s", "of L0", "L0-Lx s",
                "cost %");
    std::vector<double> g;
    for (const Case &c : cases)
        g.push_back(time_rung(c, w.data(), xq.data(), dx.data(), nb, n_out, rows,
                              row_bytes, scale, kmask, out.data()));
    const double ref = g[0];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        // Time per MAC difference, expressed as a share of L0's time per MAC.
        const double share = 100.0 * (ref - g[i]) / ref;
        std::printf("%-26s %8.2f %6.0f%% %9s %8.1f%%%s\n", cases[i].name, g[i],
                    100.0 * g[i] / ref, "", share, cases[i].wrong ? "  (WRONG)" : "");
    }
    std::printf("\nrow depth, production kernel, same data\n");
    std::printf("%-26s %8s %7s %10s\n", "rows in flight", "GMAC/s", "of 8",
                "us per row");
    const double r8 = time_rows<8>(w.data(), xq.data(), dx.data(), nb, n_out,
                                   rows, row_bytes, out.data());
    for (int n : {4, 8, 16}) {
        const double g2 = n == 8 ? r8
                         : (n == 4 ? time_rows<4>(w.data(), xq.data(), dx.data(), nb,
                                                  n_out, rows, row_bytes, out.data())
                                   : time_rows<16>(w.data(), xq.data(), dx.data(), nb,
                                                   n_out, rows, row_bytes, out.data()));
        // Microseconds of wall time per (row, output column) pair, so depth can
        // be compared directly even though the tiles differ in size.
        const double us_row = 1e6 / g2 / static_cast<double>(n_out * nb * 32) * 1e0;
        std::printf("%-26d %8.2f %6.0f%% %10.2f\n", n, g2, 100.0 * g2 / r8, us_row);
    }
    (void)time_rung;
}
