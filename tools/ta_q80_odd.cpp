// Q8_0 x8 across every block count, including the odd ones.
//
// The 512-bit kernel consumes two 32-value blocks per iteration (`b += 2`,
// `b + 2 <= nb`), so an odd block count has no home for its last block. The
// dispatcher currently refuses odd counts and falls back to the 256-bit
// kernel, which is correct -- but that refusal is a one-line condition on a
// path where the failure mode would be a silently short dot product, i.e.
// plausible-looking wrong logits rather than a crash. This test pins the
// behaviour down from the outside, so a future edit to either the guard or the
// kernel's loop bound fails here instead of in a model.
//
// The reference is computed in f64 straight from the quantised weights and
// activations, so the check is against the arithmetic the kernel claims to do,
// not against another SIMD path that could share the same bug.
#include "krk/quant.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace krk;

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<f32> d(-3.0f, 3.0f);
    bool bad = false;
    // 2..24 covers the paired path, both parities, the 8-row boundary at 24
    // (the width the kernels are written for) and a single leftover block.
    for (i64 nb = 2; nb <= 24 && !bad; nb++) {
        const i64 n = nb * 32;
        const i64 rows = 8;

        std::vector<u8> w(static_cast<size_t>(nb) * 34);
        for (i64 b = 0; b < nb; b++) {
            f32 amax = 0.0f;
            for (i64 i = 0; i < 32; i++)
                amax = std::max(amax, std::fabs(d(rng)));
            // Legal Q8_0 blocks: fp16 scale, then the 32 signed codes.
            const f32 scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            const u16 h = fp32_to_fp16(scale);
            w[b * 34] = static_cast<u8>(h & 0xff);
            w[b * 34 + 1] = static_cast<u8>(h >> 8);
            for (i64 i = 0; i < 32; i++) {
                const f32 q = std::round(d(rng) / scale);
                w[b * 34 + 2 + i] =
                    static_cast<u8>(static_cast<i8>(std::max(-127.0f, std::min(127.0f, q))));
            }
        }

        std::vector<f32> xf(static_cast<size_t>(rows * n));
        for (auto &v : xf) v = d(rng);
        std::vector<i8> xq(static_cast<size_t>(rows * n));
        std::vector<f32> dx(static_cast<size_t>(rows * nb));
        for (i64 r = 0; r < rows; r++)
            quantize_q8_row(xf.data() + r * n, n, xq.data() + r * n,
                            dx.data() + r * nb);

        std::vector<f32> out(static_cast<size_t>(rows));
        dot_q8_0_q8_x8(w.data(), xq.data(), dx.data(), n, n, nb, out.data(), 1);

        for (i64 r = 0; r < rows && !bad; r++) {
            f64 ref = 0.0;
            // Scale the tolerance by the sum of the absolute block
            // contributions. A raw relative test fails a correct kernel when
            // the blocks cancel: at nb=2 this dot lands near 0.7 while its
            // terms are ~30, so f32 accumulation differs from an f64
            // reference by 1.2e-6 of the *result* and 3e-8 of the magnitude.
            f64 mag = 0.0;
            for (i64 b = 0; b < nb; b++) {
                const f32 dw = fp16_to_fp32(
                    static_cast<u16>(w[b * 34] | (static_cast<u16>(w[b * 34 + 1]) << 8)));
                i64 acc = 0;
                for (i64 i = 0; i < 32; i++)
                    acc += static_cast<i64>(static_cast<i8>(w[b * 34 + 2 + i])) *
                           static_cast<i64>(xq[r * n + b * 32 + i]);
                const f64 term = static_cast<f64>(dw) *
                                 static_cast<f64>(dx[r * nb + b]) *
                                 static_cast<f64>(acc);
                ref += term;
                mag += std::fabs(term);
            }
            const f64 got = out[r];
            const f64 den = std::max<f64>(1.0, mag);
            if (std::fabs(got - ref) / den > 1e-6) {
                std::printf("nb=%2lld row=%lld  got=%.9g ref=%.9g rel=%.3e  FAIL\n",
                            static_cast<long long>(nb), static_cast<long long>(r), got,
                            ref, std::fabs(got - ref) / den);
                bad = true;
            }
        }
        if (!bad) {
            std::printf("nb=%2lld (%s)  ok  %lld rows\n", static_cast<long long>(nb),
                        (nb & 1) ? "odd, 256-bit fallback" : "paired, 512-bit",
                        static_cast<long long>(rows));
        }
    }
    std::printf(bad ? "Q8_0 ODD-BLOCK COVERAGE FAILED\n"
                    : "Q8_0 DOT COVERS ALL BLOCK COUNTS\n");
    return bad ? 1 : 0;
}
