// Isolated Q8_0 GEMM kernel throughput.
//
// The engine's --kprof reports the GEMM at ~35 GMAC/s against a measured int8
// ceiling of 301, and two different explanations (instruction count, then
// memory traffic) each predicted a much larger win than they delivered. So this
// times the KERNEL alone, with the data already laid out and nothing else in
// the loop, and reports what the kernel itself can sustain:
//
//   kiters   -- back-to-back x8 calls over the same activation tile, i.e. the
//               pure instruction throughput with no cache pressure
//   sweep    -- the real access pattern: every output column against every
//               activation tile, one pass (what one GEMM call does)
//
// If those two numbers differ, the gap is memory. If they agree and both sit
// far below the probe ceiling, the kernel's instruction mix is the cost.

#include <krk/quant.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace krk;

namespace {
double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
} // namespace

// --- variant experiments -----------------------------------------------------
// The shipped kernel spills its eight accumulators to the stack inside the hot
// loop (objdump shows vmovaps to/from %rsp around every vfmadd). Eight zmm
// accumulators plus eight rows' live temporaries exceed the 32 architectural
// registers, so the compiler reloads them each iteration. These variants test
// the register-pressure explanation by reducing the number of rows in flight.

#if defined(__AVX512BW__)
#include <immintrin.h>

#define KRK_AVX512 __attribute__((target("avx512f,avx512bw")))

// Same arithmetic as dot_q8_0_q8_x8_bw, but four rows in flight.
KRK_AVX512 static void q80_x8bw_4row(const u8 *w, const i8 *xq, const f32 *dx,
                                     i64 n, i64 rstride, i64 dstride, f32 *out,
                                     i64 out_stride, int nrow) {
    const i64 nb = n / 32;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc[4];
    for (int r = 0; r < 4; r++) acc[r] = _mm512_setzero_ps();
    for (i64 b = 0; b + 2 <= nb; b += 2, w += 68) {
        const f32 d0 = fp16_to_fp32(
            static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
        const f32 d1 = fp16_to_fp32(
            static_cast<u16>(w[34] | (static_cast<u16>(w[35]) << 8)));
        const __m512i wv = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 2))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36)), 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        const __m512i wp = _mm512_maskz_mov_epi8(~neg, wv);
        const __m512i wn = _mm512_maskz_sub_epi8(neg, zero, wv);
        for (int r = 0; r < nrow; r++) {
            const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(
                xq + static_cast<i64>(r) * rstride + b * 32));
            const __m512i pi =
                _mm512_madd_epi16(_mm512_maddubs_epi16(wp, xv), ones);
            const __m512i ni =
                _mm512_madd_epi16(_mm512_maddubs_epi16(wn, xv), ones);
            const __m512i iv = _mm512_sub_epi32(pi, ni);
            const f32 s0 = d0 * dx[static_cast<i64>(r) * dstride + b];
            const f32 s1 = d1 * dx[static_cast<i64>(r) * dstride + b + 1];
            const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
                _mm512_castps_si512(
                    _mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
                _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
        }
    }
    for (int r = 0; r < nrow; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}

// Sign-flip instead of magnitude-split: |w| is still exact (a masked subtract,
// which is 0-w in u8 and so gives 128 for -128, where abs_epi8 would saturate
// to 127), but the sign is applied to the ACTIVATION instead of being carried
// as a second magnitude. One maddubs then covers both signs:
//
//   sum(w*x) = sum(|w| * (x * sign(w)))
//
// and |w| <= 128, |x*sign(w)| <= 127 keeps every pair inside i16. Per row per
// 64 values that is 5 instructions (masked-sub, maddubs, madd, cvt, fma) where
// the magnitude split needs 8, and it holds one fewer live vector per row,
// which is what the shipped kernel is spilling over.
KRK_AVX512 static void q80_x8bw_signflip(const u8 *w, const i8 *xq,
                                         const f32 *dx, i64 n, i64 rstride,
                                         i64 dstride, f32 *out, i64 out_stride,
                                         int nrow) {
    const i64 nb = n / 32;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    for (i64 b = 0; b + 2 <= nb; b += 2, w += 68) {
        const f32 d0 = fp16_to_fp32(
            static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
        const f32 d1 = fp16_to_fp32(
            static_cast<u16>(w[34] | (static_cast<u16>(w[35]) << 8)));
        const __m512i wv = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 2))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36)), 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
        for (int r = 0; r < nrow; r++) {
            const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(
                xq + static_cast<i64>(r) * rstride + b * 32));
            // Where the weight is negative the product needs -x, so flip x
            // there. Same mask, one instruction.
            const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
            const __m512i iv =
                _mm512_madd_epi16(_mm512_maddubs_epi16(mag, xb), ones);
            const f32 s0 = d0 * dx[static_cast<i64>(r) * dstride + b];
            const f32 s1 = d1 * dx[static_cast<i64>(r) * dstride + b + 1];
            const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
                _mm512_castps_si512(
                    _mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
                _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
        }
    }
    for (int r = 0; r < nrow; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}

template <class K>
static void bench_variant(const char *label, K kern, int nrow, const u8 *w, i64 n_in,
                          i64 n_out, i64 row_bytes, i64 nb, const i8 *xq,
                          const f32 *xd, f32 *out, i64 rows) {
    const double t0 = now_s();
    for (i64 r = 0; r + nrow <= rows; r += nrow)
        for (i64 o = 0; o < n_out; o++)
            kern(w + static_cast<size_t>(o) * row_bytes, xq + r * n_in,
                 xd + r * nb, n_in, n_in, nb, out + r * n_out + o, n_out,
                 nrow);
    const double dt = now_s() - t0;
    const double m = static_cast<double>(rows / nrow * nrow) * n_out * n_in;
    std::printf("%-24s %8.2f GMAC/s\n", label, m / dt / 1e9);
}
#endif

int main(int argc, char **argv) {
    const i64 n_in = argc > 1 ? std::atoll(argv[1]) : 768;
    const i64 n_out = argc > 2 ? std::atoll(argv[2]) : 2048;
    const i64 rows = argc > 3 ? std::atoll(argv[3]) : 512;
    const i64 nb = n_in / 32;
    const i64 row_bytes = n_in + 2 * nb;

    std::mt19937 rng(12345);
    std::vector<u8> w(static_cast<size_t>(n_out * row_bytes));
    for (size_t i = 0; i < w.size(); i++) {
        // Legal f16 scale every 34 bytes, random int8 payload otherwise.
        if (i % 34 == 0) w[i] = 0x00;
        else if (i % 34 == 1) w[i] = 0x1C;   // 2^-9
        else w[i] = static_cast<u8>(static_cast<i8>(static_cast<int>(rng() % 256) - 128));
    }
    std::vector<f32> xf(static_cast<size_t>(rows * n_in));
    std::uniform_real_distribution<f32> d(-3.0f, 3.0f);
    for (auto &v : xf) v = d(rng);
    std::vector<i8> xq(static_cast<size_t>(rows * n_in));
    std::vector<f32> xd(static_cast<size_t>(rows * nb));
    for (i64 r = 0; r < rows; r++)
        quantize_q8_row(xf.data() + r * n_in, n_in, xq.data() + r * n_in,
                        xd.data() + r * nb);
    std::vector<f32> out(static_cast<size_t>(rows * n_out));

    const double macs_call =
        static_cast<double>(rows) * static_cast<double>(n_out) *
        static_cast<double>(n_in);

    // 1. Pure instruction throughput: the same activation tile, 4096 different
    //    weight rows, no cache-missing pattern. Everything the kernel needs for
    //    one tile is 8*n_in bytes of activation and one weight row.
    {
        const i64 reps = 8;
        const double t0 = now_s();
        for (i64 rep = 0; rep < reps; rep++)
            for (i64 o = 0; o < n_out; o++)
                dot_q8_0_q8_x8(w.data() + static_cast<size_t>(o) * row_bytes,
                               xq.data(), xd.data(), n_in, n_in, nb,
                               out.data() + o, static_cast<i64>(1));
        const double dt = now_s() - t0;
        const double m = static_cast<double>(reps) * n_out * 8.0 * n_in;
        std::printf("kiters   n_in=%lld n_out=%lld rows=8     %8.2f GMAC/s\n",
                    static_cast<long long>(n_in),
                    static_cast<long long>(n_out), m / dt / 1e9);
    }

    // 2. The real sweep: one full GEMM call, same order the backend uses.
    {
        const double t0 = now_s();
        for (i64 r = 0; r + 8 <= rows; r += 8)
            for (i64 o = 0; o < n_out; o++)
                dot_q8_0_q8_x8(w.data() + static_cast<size_t>(o) * row_bytes,
                               xq.data() + r * n_in, xd.data() + r * nb, n_in,
                               n_in, nb, out.data() + r * n_out + o, n_out);
        const double dt = now_s() - t0;
        std::printf("sweep    n_in=%lld n_out=%lld rows=%lld   %8.2f GMAC/s\n",
                    static_cast<long long>(n_in),
                    static_cast<long long>(n_out),
                    static_cast<long long>(rows), macs_call / dt / 1e9);
    }

    // 3. The same sweep with the columns blocked, activation tile held in L1.
    {
        const i64 nc = 128;
        const double t0 = now_s();
        for (i64 c = 0; c < n_out; c += nc) {
            const i64 o1 = std::min<i64>(n_out, c + nc);
            for (i64 r = 0; r + 8 <= rows; r += 8)
                for (i64 o = c; o < o1; o++)
                    dot_q8_0_q8_x8(
                        w.data() + static_cast<size_t>(o) * row_bytes,
                        xq.data() + r * n_in, xd.data() + r * nb, n_in, n_in,
                        nb, out.data() + r * n_out + o, n_out);
        }
        const double dt = now_s() - t0;
        std::printf("blocked  n_in=%lld n_out=%lld nc=%lld      %8.2f GMAC/s\n",
                    static_cast<long long>(n_in),
                    static_cast<long long>(n_out),
                    static_cast<long long>(nc), macs_call / dt / 1e9);
    }
#if defined(__AVX512BW__)
    bench_variant("4 rows in flight", q80_x8bw_4row, 4, w.data(), n_in, n_out, row_bytes, nb,
                  xq.data(), xd.data(), out.data(), rows);
    bench_variant("2 rows in flight", q80_x8bw_4row, 2, w.data(), n_in, n_out, row_bytes, nb,
                  xq.data(), xd.data(), out.data(), rows);
    bench_variant("signflip 8 rows", q80_x8bw_signflip, 8, w.data(), n_in, n_out, row_bytes, nb,
                  xq.data(), xd.data(), out.data(), rows);
    bench_variant("signflip 4 rows", q80_x8bw_signflip, 4, w.data(), n_in, n_out, row_bytes, nb,
                  xq.data(), xd.data(), out.data(), rows);
#endif
    return 0;
}

