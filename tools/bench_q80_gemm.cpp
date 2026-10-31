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

#include <immintrin.h>

#include <krk/quant.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// wsc8 with the eight row sums reduced TOGETHER instead of one at a time.
// `_mm512_reduce_add_ps` costs ~9 uops and the kernel calls it eight times per
// (row tile, output row), so the reduction is ~11% of the kernel's uops. This
// variant halves the lane count of each accumulator, then reduces eight vectors
// with three rounds of hadd. It changes the summation ORDER, so it is not
// bit-identical -- it exists to price the reduction, and nothing ships on it
// without re-recording every pinned number.
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void q80_wsc8_tred(const u8 *w, const i8 *xq, i64 n, i64 rstride,
                          f32 *out, i64 out_stride, const f32 *wsc,
                          const f32 *xsc) {
    const i64 nb = n / 32;
    const i64 npair = nb / 2;
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    for (i64 p = 0; p < npair; p++, w += 68) {
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36));
        const __m512i wv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
        const __m512 wsv = _mm512_loadu_ps(wsc + p * 16);
        for (int r = 0; r < 8; r++) {
            const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(
                xq + static_cast<i64>(r) * rstride + p * 64));
            const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
            const __m512i iv = _mm512_dpbusd_epi32(zero, mag, xb);
            const __m512 xsv =
                _mm512_loadu_ps(xsc + (static_cast<i64>(r) * npair + p) * 16);
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv),
                                     _mm512_mul_ps(wsv, xsv), acc[r]);
        }
    }
    __m256 t[8];
    for (int r = 0; r < 8; r++)
        t[r] = _mm256_add_ps(_mm512_castps512_ps256(acc[r]),
                             _mm256_castpd_ps(_mm512_extractf64x4_pd(
                                 _mm512_castps_pd(acc[r]), 1)));
    __m256 a = _mm256_hadd_ps(t[0], t[1]);
    __m256 b = _mm256_hadd_ps(t[2], t[3]);
    __m256 c = _mm256_hadd_ps(t[4], t[5]);
    __m256 d = _mm256_hadd_ps(t[6], t[7]);
    a = _mm256_hadd_ps(a, b);
    c = _mm256_hadd_ps(c, d);
    a = _mm256_hadd_ps(a, c);
    // a = [s0,s1,s2,s3, s4,s5,s6,s7]
    f32 v[8];
    _mm256_storeu_ps(v, a);
    for (int r = 0; r < 8; r++) out[static_cast<i64>(r) * out_stride] = v[r];
}

// ---- pv8, the compact pair table: MEASURED AND REJECTED --------------------
//
// This kernel lives in the bench, not in src/quant.cpp, because it is a
// measurement and not a candidate: on this host (768x2048x512, nc=128, best of
// five, four runs) it sustains 44.5-45.9 GMAC/s against the shipping wsc8's
// 48.1-49.1, and 33.99-35.69 for the no-table x8 fallback. It is bit-identical
// to x8 -- the second `identical:` line below proves it every run -- it simply
// is not faster. Do not wire it in.
//
// What this removes, and what the ablation says it is worth. `kraken-q80-ablate`
// on this host, 768x2048x512, one thread, two runs: making the ACTIVATION scale
// constant buys 27-60%, an integer accumulate buys 70-76%, and the per-row
// scale-vector build is the chain that both of them are deleting. In
// `dot_q8_0_q8_x8_vnni` that chain is seven uops per (row, block pair) -- two
// strided f32 loads, two multiplies, two broadcasts and an insert.
//
// Two of those seven are the f16 header of the WEIGHT row, which is the same for
// every activation row, so it belongs outside the row loop. The other five
// describe the activation row and cannot be hoisted -- but they can be done
// shorter: one 64-bit load of the (sx0, sx1) pair, one 128-bit multiply against
// the weight pair, one broadcast and one permute that fans (a, b) out to
// (a x8, b x8). Three uops instead of five, and no table growth on the
// activation side (the `xsc` table in dot_q8_0_q8_wsc8 costs 64 B per row per
// pair, i.e. as much traffic as the activation row itself).
//
// The weight header is what the pair table holds: two f32 per (output row, block
// pair), 8 bytes, 0.125 B per weight value. It is built once per COLUMN BLOCK
// and reused by every eight-row tile, so the weight stream grows by 12% and the
// byte pluck, the convert and two shuffles leave the inner loop.
//
// Bit-identity is by construction, not by luck: the value multiplied into every
// lane is `d0 * sx0` (or `d1 * sx1`), the same two f32 operands in the same
// order as the x8 kernel's `d0 * dx[...]`, and the accumulator is summed over
// block pairs in the same ascending order. This file checks it byte for byte
// against both shipping kernels on random data, every run.
static void build_q8_0_wpairs(const u8 *w, i64 row_bytes, i64 nrows, i64 n, f32 *out) {
    const i64 npair = (n / 32) / 2;
    for (i64 o = 0; o < nrows; o++) {
        const u8 *wr = w + o * row_bytes;
        f32 *dst = out + o * npair * 2;
        for (i64 p = 0; p < npair; p++) {
            const u8 *wp = wr + p * 68;
            dst[2 * p] = fp16_to_fp32(
                static_cast<u16>(wp[0] | (static_cast<u16>(wp[1]) << 8)));
            dst[2 * p + 1] = fp16_to_fp32(
                static_cast<u16>(wp[34] | (static_cast<u16>(wp[35]) << 8)));
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void dot_q8_0_q8_pv8_vnni(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                          i64 rstride, i64 dstride, f32 *out, i64 out_stride,
                          const f32 *wp) {
    const i64 nb = n / 32;
    const i64 npair = nb / 2;
    const __m512i zero = _mm512_setzero_si512();
    // [0 x8, 1 x8]: the two lanes of a block pair fanned out to the eight lanes
    // each one scales. Same index vector the decode kernel uses.
    const __m512i k_pair = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                             1, 1, 1);
    __m512 acc[8];
    for (i32 r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    for (i64 p = 0; p < npair; p++, w += 68) {
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36));
        const __m512i wv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
        const __m128 dpair = _mm_loadl_pi(
            _mm_setzero_ps(), reinterpret_cast<const __m64 *>(wp + p * 2));
        for (i32 r = 0; r < 8; r++) {
            const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(
                xq + static_cast<i64>(r) * rstride + p * 64));
            const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
            const __m512i iv = _mm512_dpbusd_epi32(zero, mag, xb);
            // (sx0, sx1) as one 64-bit load; the upper half is zero, so the
            // unused lanes of the product are zero rather than garbage.
            const __m128 xpair = _mm_loadl_pi(
                _mm_setzero_ps(),
                reinterpret_cast<const __m64 *>(dx + static_cast<i64>(r) * dstride +
                                                p * 2));
            const __m128 sp = _mm_mul_ps(dpair, xpair);
            const __m512 sc =
                _mm512_permutexvar_ps(k_pair, _mm512_broadcast_f32x4(sp));
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
        }
    }
    for (i32 r = 0; r < 8; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}

// Same fallback rule as the other table kernels: no VNNI or an odd block count
// means the answer comes from the no-table kernel, unchanged. The vnni test is
// the bench's own because quant.cpp's is file-local.
static void dot_q8_0_q8_pv8(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                            i64 rstride, i64 dstride, f32 *out, i64 out_stride,
                            const f32 *wp) {
    static const bool vnni = __builtin_cpu_supports("avx512vnni");
    if (wp != nullptr && vnni && ((n / 32) & 1) == 0) {
        dot_q8_0_q8_pv8_vnni(w, xq, dx, n, rstride, dstride, out, out_stride, wp);
        return;
    }
    (void)wp;
    dot_q8_0_q8_x8(w, xq, dx, n, rstride, dstride, out, out_stride);
}

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
    // 4. The three q8_0 x8 kernels the q8_0 prefill path can choose between,
    //    on the same data, best of five, and the same access order the backend
    //    uses (column block, then row tile). The scale tables are built the way
    //    the backend builds them -- per column block for the weights, per row
    //    tile for the activations -- and the three outputs must be identical
    //    byte for byte, because the kernels differ in uops, not in arithmetic.
    {
        const i64 npair = nb / 2;
        std::vector<f32> wsc(static_cast<size_t>(n_out * npair) * 16);
        std::vector<f32> xsc(static_cast<size_t>(8 * npair) * 16);
        std::vector<f32> wp(static_cast<size_t>(n_out * npair) * 2);
        std::vector<f32> o0(out.size()), o1(out.size()), o2(out.size()),
            o3(out.size());
        auto sweep = [&](int which, f32 *dst, i64 nc) {
            for (i64 c = 0; c < n_out; c += nc) {
                const i64 o1e = std::min<i64>(n_out, c + nc);
                build_q8_0_wscales(w.data() + static_cast<size_t>(c) * row_bytes,
                                   row_bytes, o1e - c, n_in,
                                   wsc.data() + static_cast<size_t>(c * npair) * 16);
                build_q8_0_wpairs(w.data() + static_cast<size_t>(c) * row_bytes,
                                  row_bytes, o1e - c, n_in,
                                  wp.data() + static_cast<size_t>(c * npair) * 2);
                for (i64 r = 0; r + 8 <= rows; r += 8) {
                    build_q8_0_xscales(xd.data() + r * nb, nb, 8, n_in, xsc.data());
                    for (i64 o = c; o < o1e; o++) {
                        const u8 *wr = w.data() + static_cast<size_t>(o) * row_bytes;
                        f32 *d = dst + r * n_out + o;
                        if (which == 0)
                            dot_q8_0_q8_x8(wr, xq.data() + r * n_in,
                                           xd.data() + r * nb, n_in, n_in, nb, d,
                                           n_out);
                        else if (which == 1)
                            dot_q8_0_q8_wsc8(wr, xq.data() + r * n_in,
                                             xd.data() + r * nb, n_in, n_in, nb, d,
                                             n_out,
                                             wsc.data() + static_cast<size_t>(o * npair) * 16,
                                             xsc.data());
                        else if (which == 2)
                            dot_q8_0_q8_pv8(wr, xq.data() + r * n_in,
                                            xd.data() + r * nb, n_in, n_in, nb, d,
                                            n_out,
                                            wp.data() + static_cast<size_t>(o * npair) * 2);
                        else
                            q80_wsc8_tred(wr, xq.data() + r * n_in, n_in, n_in, d,
                                          n_out,
                                          wsc.data() + static_cast<size_t>(o * npair) * 16,
                                          xsc.data());
                    }
                }
            }
        };
        const i64 nc = 128;
        double best[4] = {1e9, 1e9, 1e9, 1e9};
        for (int rep = 0; rep < 5; rep++) {
            for (int which = 0; which < 4; which++) {
                f32 *dst = which == 0   ? o0.data()
                           : which == 1 ? o1.data()
                           : which == 2 ? o2.data()
                                        : o3.data();
                const double t0 = now_s();
                sweep(which, dst, nc);
                const double dt = now_s() - t0;
                if (dt < best[which]) best[which] = dt;
            }
        }
        static const char *names[4] = {"x8 no table", "wsc8 16-lane tables",
                                      "pv8 pair table", "wsc8 + tree reduce"};
        std::printf("\nproduction prefill candidates, nc=%lld, best of 5\n", nc);
        for (int which = 0; which < 4; which++)
            std::printf("  %-22s %8.2f GMAC/s  %6.4f ms%s\n", names[which],
                        macs_call / best[which] / 1e9, best[which] * 1e3,
                        which == 3 ? "  (reduction order changed)" : "");
        // The whole point: same answer, byte for byte. A faster kernel that
        // disagrees is a different model, and on a real file that reads as a
        // slightly worse token rather than as a bug.
        const bool same01 = std::memcmp(o0.data(), o1.data(), o0.size() * 4) == 0;
        const bool same02 = std::memcmp(o0.data(), o2.data(), o0.size() * 4) == 0;
        std::printf("  identical: x8 vs wsc8 %s | x8 vs pv8 %s\n",
                    same01 ? "yes" : "NO", same02 ? "yes" : "NO");
        if (!same01 || !same02) {
            // Report the first differing cell, so a mismatch names itself.
            for (size_t i = 0; i < o0.size(); i++) {
                if (o0[i] != o1[i] || o0[i] != o2[i]) {
                    std::printf("  first difference at %zu: %.9g %.9g %.9g\n", i,
                                static_cast<double>(o0[i]),
                                static_cast<double>(o1[i]),
                                static_cast<double>(o2[i]));
                    break;
                }
            }
        }
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

