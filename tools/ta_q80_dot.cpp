// Q8_0 integer-dot accuracy fixture.
//
// The activation-quantising path is a different algorithm from the f32-dequant
// path it replaced, so "does it match the old one" is the wrong question: it
// differs by design, exactly as the Q4_K and Q6_K paths already do. The
// question this answers is whether the INT8 KERNEL computes the dot it claims
// to, i.e. sum(d_w[q] * dx[b] * q_k * xq_k), which is checkable exactly.
//
// Three implementations are compared over random weights and activations that
// include the endpoints the format allows:
//   1. a scalar reference doing the same integer arithmetic in f64,
//   2. dot_q8_0_q8        (per-row entry point; the x8 kernel's scalar twin),
//   3. dot_q8_0_q8_x8     (8-row batched, dispatched 512-bit or 256-bit).
//
// The x8 result must be bit-identical between the 512-bit and 256-bit kernels
// (KRK_CPU_Q80_NOBW=1 selects the latter) because they sum the same integers in
// the same block order and differ only in how many lanes one instruction
// covers. That is the assertion worth having: a chunk bug like the ones the
// attention fixture caught would show up as a difference here, not as a
// plausible-looking tolerance.

#include <krk/quant.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace krk;

namespace {

constexpr int kRows = 8;

// Random Q8_0 weight row: nb blocks of (f16 scale, 32 int8). Values are drawn
// over the FULL legal range including -128, which is where an abs_epi8-based
// magnitude would silently read 127 instead of 128.
std::vector<u8> make_weights(i64 nb, std::mt19937 &rng) {
    std::vector<u8> w(static_cast<size_t>(nb) * 34);
    std::uniform_int_distribution<int> qd(-32768, 32767);
    for (i64 b = 0; b < nb; b++) {
        // A normal, finite f16 scale: just take a small value and encode it the
        // way the block does. 0.0078125 is exactly representable.
        const u16 bits = static_cast<u16>(0x1C00u); // 2^-9 in f16
        (void)qd;
        w[static_cast<size_t>(b) * 34 + 0] = static_cast<u8>(bits & 0xFF);
        w[static_cast<size_t>(b) * 34 + 1] = static_cast<u8>(bits >> 8);
        for (int k = 0; k < 32; k++) {
            // Include the extremes on purpose: -128 is legal in the format.
            int v = static_cast<int>(rng() % 256u) - 128;
            w[static_cast<size_t>(b) * 34 + 2 + k] =
                static_cast<u8>(static_cast<i8>(v));
        }
    }
    return w;
}

// Activation rows quantised the way the backend does it.
void quant_acts(i64 n, std::vector<i8> &xq, std::vector<f32> &dx,
                std::mt19937 &rng) {
    std::uniform_real_distribution<f32> d(-3.0f, 3.0f);
    std::vector<f32> xf(static_cast<size_t>(n));
    for (i64 r = 0; r < kRows; r++) {
        for (i64 k = 0; k < n; k++) xf[static_cast<size_t>(k)] = d(rng);
        quantize_q8_row(xf.data(), n, xq.data() + r * n,
                        dx.data() + r * (n / 32));
    }
}

// The arithmetic the kernels claim to implement, in f64.
//
// A dot product's answer can sit near zero while its terms are large, so the
// error is measured against the sum of |terms| rather than against the result.
// Dividing by a near-zero result reports 1e-6 of "error" for a value that is
// correct to the last f32 bit, which is how a tolerance test stops testing
// anything. `mag` is that denominator.
f64 reference(const std::vector<u8> &w, i64 n, i64 block_stride_unused,
              const i8 *xq, const f32 *dx, f64 *mag) {
    (void)block_stride_unused;
    const i64 nb = n / 32;
    f64 acc = 0.0;
    *mag = 0.0;
    for (i64 b = 0; b < nb; b++) {
        // fp16 -> f32 for the weight scale, matching the kernel's own decode.
        const u16 bits = static_cast<u16>(w[static_cast<size_t>(b) * 34 + 0]) |
                         (static_cast<u16>(w[static_cast<size_t>(b) * 34 + 1])
                          << 8);
        const f32 d = fp16_to_fp32(bits);
        i64 isum = 0;
        for (int k = 0; k < 32; k++) {
            const i8 q = static_cast<i8>(w[static_cast<size_t>(b) * 34 + 2 + k]);
            isum += static_cast<i64>(q) * static_cast<i64>(xq[b * 32 + k]);
        }
        acc += static_cast<f64>(d) * static_cast<f64>(dx[b]) *
               static_cast<f64>(isum);
        f64 amag = 0.0;
        for (int k = 0; k < 32; k++) {
            const i8 q = static_cast<i8>(w[static_cast<size_t>(b) * 34 + 2 + k]);
            amag += std::fabs(static_cast<f64>(q) *
                              static_cast<f64>(xq[b * 32 + k]));
        }
        *mag += std::fabs(static_cast<f64>(d) * static_cast<f64>(dx[b])) * amag;
    }
    return acc;
}

int failures = 0;
void check(bool ok, const char *what, f64 got, f64 want, f64 rel) {
    if (!ok) {
        std::printf("  FAIL %-34s got %.9g want %.9g rel %.3e\n", what, got,
                    want, rel);
        failures++;
    }
}

} // namespace

int main(int argc, char **argv) {
    // Optional dump: run twice, with and without KRK_CPU_Q80_NOBW, and compare
    // the two files. That is how the 512-bit and 256-bit kernels get checked
    // against each other -- the dispatcher latches its choice once per process.
    std::FILE *dump = nullptr;
    if (argc > 1) dump = std::fopen(argv[1], "wb");
    std::mt19937 rng(0xC0FFEEu);
    // 768 and 2048 are the real shapes; 96 is odd in blocks (3) so it exercises
    // the dispatcher's wholesale fallback; 32 is the single-block minimum.
    const i64 shapes[] = {768, 2048, 96, 32};
    f64 worst = 0.0;
    f64 worst_x = 0.0;

    for (i64 n : shapes) {
        const i64 nb = n / 32;
        std::vector<u8> w = make_weights(nb, rng);
        std::vector<i8> xq(static_cast<size_t>(kRows * n));
        std::vector<f32> dx(static_cast<size_t>(kRows * nb));
        quant_acts(n, xq, dx, rng);

        std::vector<f32> out8(static_cast<size_t>(kRows));
        dot_q8_0_q8_x8(w.data(), xq.data(), dx.data(), n, n, nb, out8.data(),
                       1);

        std::printf("n=%lld blocks=%lld\n", static_cast<long long>(n),
                    static_cast<long long>(nb));
        for (int r = 0; r < kRows; r++) {
            f64 mag = 0.0;
            const f64 want =
                reference(w, n, 0, xq.data() + r * n, dx.data() + r * nb, &mag);
            const f32 got1 = dot_q8_0_q8(w.data(), xq.data() + r * n,
                                         dx.data() + r * nb, n);
            const f64 den = std::max(1e-12, mag);
            const f64 rel1 = std::fabs(static_cast<f64>(got1) - want) / den;
            const f64 rel8 = std::fabs(static_cast<f64>(out8[r]) - want) / den;
            // The two kernels do NOT agree bit-for-bit and must not be required
            // to: the per-row kernel accumulates 8 i32 lanes per block while the
            // 512-bit x8 kernel accumulates 16 per two blocks, so the reduction
            // tree differs even though the integers summed are identical. The
            // disagreement has to stay at rounding level, measured against the
            // same magnitude denominator.
            const f64 relx = std::fabs(static_cast<f64>(got1) - out8[r]) / den;
            worst_x = std::max(worst_x, relx);
            {
                char bx[64];
                std::snprintf(bx, sizeof bx, "n=%lld row %d single vs x8",
                              static_cast<long long>(n), r);
                check(relx < 1e-6, bx, got1, out8[r], relx);
            }
            if (dump) {
                std::fwrite(&out8[r], sizeof(f32), 1, dump);
            }
            worst = std::max(worst, std::max(rel1, rel8));
            char buf[64];
            std::snprintf(buf, sizeof buf, "n=%lld row %d single", 
                          static_cast<long long>(n), r);
            check(rel1 < 1e-6, buf, got1, want, rel1);
            std::snprintf(buf, sizeof buf, "n=%lld row %d x8", 
                          static_cast<long long>(n), r);
            check(rel8 < 1e-6, buf, out8[r], want, rel8);
        }
    }

    if (dump) std::fclose(dump);
    std::printf("\nworst error vs f64 reference (magnitude-relative): %.3e\n",
                worst);
    std::printf("worst single-vs-x8 disagreement:                  %.3e\n",
                worst_x);
    if (failures == 0) std::printf("Q8_0 INT8 DOT ACCURACY OK\n");
    return failures == 0 ? 0 : 1;
}
