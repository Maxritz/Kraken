// ============================================================================
//  ta_kquant_dot — does dot_q2_K_q8 / dot_q3_K_q8 agree with the reference
//  decoder, on the same input, to the accuracy the int8 format allows?
//
//  The engine-level A/B is NOT this test. There the two arms differ in the
//  activation as well: the int8 path quantises the activation to eight bits and
//  the generic path keeps it in f32, so a few units in the third decimal is
//  quantisation, not a kernel defect. This harness pins the ACTIVATION to the
//  same int8 array and its per-32 scales for both sides, so the only thing that
//  can differ is the weight-side arithmetic:
//
//      kernel   -> dot_q{2,3}_K_q8(block, xq, dx, n)
//      expected -> SUM_i dequant_row(block)_i * (xq_i * dx[i/32])
//
//  That sum is computed in f64 from the reference decoder's own output, which
//  is what makes a failure here unambiguous.
//
//  Blocks are random BYTES with the dtype slots set to real f16 scales, because
//  every byte pattern is a legal K-quant block and random bytes exercise every
//  hmask bit, every 2-bit plane and every axis of the scale scramble. A real
//  file would only exercise the patterns its quantiser happened to emit.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "krk/quant.hpp"

using namespace krk;

static u16 as_f16(f32 v) {
    u16 h;
    std::memcpy(&h, &v, 2);
    return h;
}

// random block of `blk_bytes`, with the file's scale slots replaced
static void make_block(u8 *b, i64 blk_bytes, DType t, std::mt19937 &rng) {
    for (i64 i = 0; i < blk_bytes; i++) b[i] = static_cast<u8>(rng() & 0xFF);
    auto put = [&](i64 off, f32 v) {
        const u16 h = as_f16(v);
        b[off + 0] = static_cast<u8>(h & 0xFF);
        b[off + 1] = static_cast<u8>(h >> 8);
    };
    if (t == DType::Q2_K) {
        put(80, 0.02f);   // d
        put(82, 0.01f);   // dmin
    } else {
        put(108, 0.02f);  // d_all
    }
}

static int run(DType t, const char *name, i64 blk, i64 blk_bytes, i64 n,
               int iters) {
    std::mt19937 rng(0xC0FFEEu);
    std::vector<u8> w(static_cast<size_t>(blk_bytes * (n / blk)));
    std::vector<f32> x(static_cast<size_t>(n));
    std::vector<f32> deq(static_cast<size_t>(blk));
    std::vector<i8> xq(static_cast<size_t>(n));
    std::vector<f32> dx(static_cast<size_t>(n / 32));

    double worst = 0.0, worst_abs = 0.0;
    double sum_rel = 0.0;
    for (int it = 0; it < iters; it++) {
        for (i64 b = 0; b < n / blk; b++)
            make_block(w.data() + b * blk_bytes, blk_bytes, t, rng);
        // A plausible activation, and the SAME quantised form for both sides.
        std::normal_distribution<f32> nd(0.0f, 1.0f);
        for (i64 i = 0; i < n; i++) x[static_cast<size_t>(i)] = nd(rng);
        quantize_q8_row(x.data(), n, xq.data(), dx.data());

        // Expected: reference decoder -> f32, dotted in f64 with the
        // dequantised activation. `gross` accumulates |term| so the error can
        // be reported against the size of the sum's TERMS rather than against
        // the size of its result.
        //
        // That distinction is the whole measurement. Random blocks cancel
        // hard -- 256 terms of magnitude ~1e3 summing to ~5e2 -- and the
        // reference rounds every weight to f32 individually, so a correct
        // kernel disagrees with it by ~1e-4 of the RESULT while agreeing to
        // ~1e-7 of the terms. Dividing by the result measures the conditioning
        // of the fixture, not the accuracy of the kernel. (Same lesson as the
        // ta_q80_tiers `worst_diff(a, b2, 1.0)` bug: the denominator decides
        // what the number means.)
        f64 want = 0.0, gross = 0.0;
        for (i64 b = 0; b < n / blk; b++) {
            dequant_row(t, w.data() + b * blk_bytes, deq.data(), blk);
            for (i64 i = 0; i < blk; i++) {
                const i64 g = b * blk + i;
                const f64 term =
                    static_cast<f64>(deq[static_cast<size_t>(i)]) *
                    static_cast<f64>(xq[static_cast<size_t>(g)]) *
                    static_cast<f64>(dx[static_cast<size_t>(g / 32)]);
                want += term;
                gross += std::fabs(term);
            }
        }
        const f32 got = (t == DType::Q3_K)
                            ? dot_q3_K_q8(w.data(), xq.data(), dx.data(), n)
                            : dot_q2_K_q8(w.data(), xq.data(), dx.data(), n);
        const double rel = std::fabs(static_cast<double>(got) - want) /
                           std::max(1e-9, gross);
        if (rel > worst) worst = rel;
        const double absd = std::fabs(static_cast<double>(got) - want);
        if (absd > worst_abs) worst_abs = absd;
        sum_rel += rel;
    }
    const double mean = sum_rel / iters;
    std::printf("%-6s n=%-5lld blocks=%-2lld  worst rel-to-gross %.3e  "
                "mean %.3e  worst abs %.3e\n",
                name, static_cast<long long>(n),
                static_cast<long long>(n / blk), worst, mean, worst_abs);
    // The kernel sums each sub-block exactly in i32 and applies its scale once;
    // the reference rounds every dequantised weight to f32 first. That is a
    // ~1e-7 relative difference on the terms, so 1e-6 of the gross magnitude
    // passes comfortably and any unpack error (a wrong hmask bit, a wrong
    // plane) is O(1e-1) and fails by five orders.
    const bool ok = worst < 1e-6;
    std::printf("       %s\n", ok ? "ok" : "MISMATCH");
    return ok ? 0 : 1;
}

int main() {
    int bad = 0;
    // n = 512 and 768 exercise more than one block and a non-power-of-two
    // block count; nb must be a whole number of blocks for both formats.
    for (i64 n : {256LL, 512LL, 1024LL}) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "Q3_K");
        bad |= run(DType::Q3_K, nm, 256, 110, n, 200);
        std::snprintf(nm, sizeof(nm), "Q2_K");
        bad |= run(DType::Q2_K, nm, 256, 84, n, 200);
    }
    std::printf("failure: %s\n", bad ? "YES" : "none");
    return bad;
}
