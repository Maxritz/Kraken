// ============================================================================
//  gemm_i8k_verify.cpp — check the DP4A weight staging against the HOST
//  reference decoder, code by code and format by format.
//
//  WHY THIS EXISTS. src/hip/kernels/gemm_dp4a.hpp re-reads the Q8_0, Q4_K,
//  Q5_K and Q6_K block layouts that dequant.hpp decodes for the fp16 kernels,
//  because the int8 path wants the *codes* and the sub-block scales rather than
//  fp16 values. Two implementations of one format is exactly where a silent
//  mis-read hides: a wrong nibble, a swapped sub-block scale or an off-by-one
//  in a bit plane costs a few percent of accuracy and never crashes. So this
//  tool checks positions, not averages:
//
//    A. CODE PROBE. The activation rows carry a single nonzero code (127) at
//       k = k0 each. The int8 path then produces
//           out[m][j] = sa[m] * 127 * w[j][k0]
//       where w is what the HOST decoder (quant.hpp's dequant_row, the CPU
//       backend's own reference) produces for that position. Sweeping k0 over
//       the row checks every code position of every format. A layout bug shows
//       up as a factor of 2..31 at one position, not as noise.
//    B. RANDOM GEMM. Random block bytes (all fields at once, so no field can be
//       mapped wrong without moving the answer) and random activations, against
//       a double-precision reference built from the host decoder. The tolerance
//       is scaled by sum_k |a_k * w_k|, so a near-zero output cannot pass or
//       fail by cancellation.
//
//  The activation codes and scales the reference uses are the ones the KERNEL
//  quantized (copied back from device memory), so this test isolates the weight
//  staging and the fold; the activation quantizer itself is exercised by the
//  engine A/B, where the whole model runs through it.
//
//  Build: see CMakeLists.txt (kraken-i8k, linked against krk_core + krk_hip).
// ============================================================================
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "krk/common.hpp"
#include "krk/quant.hpp"
#include "kernels/gemm_dp4a.hpp"

using namespace krk;

#define HIP_CHECK(x)                                                              \
    do {                                                                          \
        const hipError_t e_ = (x);                                                \
        if (e_ != hipSuccess) {                                                   \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_),         \
                        __FILE__, __LINE__);                                      \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace {

// The output is fp16, so the tightest honest bound on a comparison against an
// exact reference is the format's own rounding step: 2^-11 = 4.9e-4 relative.
// A staging bug is 100x larger (one wrong code moves the weight by 1/sub-block-
// scale worth of value), so the threshold sits just above the rounding bound.
constexpr double kFp16Round = 6e-4;

struct Format {
    DType wt;
    const char *name;
    i64 n_in;      // a whole number of blocks for this format
    int kind;      // dp4a_ok's kind, to prove the format is staged at all
};

// Random block bytes, not random values: the point is to leave no field
// hand-written, so the host decoder is the only thing that says what the bytes
// mean. Block sizes come from the library, so this file does not restate them.
std::vector<u8> random_weights(DType wt, i64 n_out, i64 n_in, std::mt19937 &rng) {
    const i64 row_bytes = static_cast<i64>(dtype_row_bytes(wt, n_in));
    std::vector<u8> w(static_cast<size_t>(row_bytes) * static_cast<size_t>(n_out));
    // Uniform bytes everywhere except the fp16 scale fields, which are left to
    // the same distribution: a random bit pattern in a half can be NaN, and a
    // NaN scale would make every comparison meaningless rather than failing.
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t i = 0; i < w.size(); i++) w[i] = static_cast<u8>(byte(rng));
    return w;
}

// f16 scales that are small and positive, written where each format keeps them.
void set_scale_fields(DType wt, std::vector<u8> &w, i64 n_out, i64 n_in,
                      std::mt19937 &rng) {
    const i64 row_bytes = static_cast<i64>(dtype_row_bytes(wt, n_in));
    std::uniform_real_distribution<float> d01(0.004f, 0.06f);
    auto put_h = [&](u8 *p, float v) {
        const __half h = __float2half(v);
        u16 bits = 0;
        std::memcpy(&bits, &h, 2);
        p[0] = static_cast<u8>(bits & 0xFF);
        p[1] = static_cast<u8>(bits >> 8);
    };
    for (i64 j = 0; j < n_out; j++) {
        u8 *row = w.data() + static_cast<size_t>(j) * static_cast<size_t>(row_bytes);
        if (wt == DType::Q8_0) {
            for (i64 b = 0; b < n_in / 32; b++) put_h(row + b * 34, d01(rng));
        } else if (wt == DType::Q4_K || wt == DType::Q5_K) {
            const i64 blocks = n_in / 256;
            const i64 bb = (wt == DType::Q4_K) ? 144 : 176;
            for (i64 b = 0; b < blocks; b++) {
                u8 *p = row + b * bb;
                put_h(p, d01(rng));      // d
                put_h(p + 2, d01(rng));  // dmin
            }
        } else { // Q6_K: one f16 scale per 256, plus int8 sub-block scales
            const i64 blocks = n_in / 256;
            for (i64 b = 0; b < blocks; b++) {
                u8 *p = row + b * 210;
                put_h(p + 208, d01(rng));
                for (int s = 0; s < 16; s++)
                    p[192 + s] = static_cast<u8>(
                        static_cast<i8>(static_cast<int>(std::uniform_int_distribution<int>(-20, 20)(rng))));
            }
        }
    }
}

// What the kernel's own activation quantizer produced for this launch, copied
// back: the reference must use the same codes and scales or it is testing the
// quantizer instead of the staging.
struct Act {
    std::vector<i8> codes;   // [rows, n_in]
    std::vector<f32> scales; // [rows, n_in/32]
};

} // namespace

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    std::printf("device: %s  %s\n\n", p.name, p.gcnArchName);

    const Format formats[] = {
        {DType::Q8_0, "Q8_0", 96, kDp4aWq8},
        {DType::Q4_K, "Q4_K", 256, kDp4aWq4},
        {DType::Q5_K, "Q5_K", 256, kDp4aWq5},
        {DType::Q6_K, "Q6_K", 256, kDp4aWq6},
    };

    std::mt19937 rng(20261003u);
    bool all_ok = true;

    for (const Format &f : formats) {
        std::printf("=== %s ===\n", f.name);
        if (dp4a_kind(static_cast<int>(f.wt)) != f.kind) {
            std::printf("  FAIL: dp4a_kind does not stage this format\n\n");
            all_ok = false;
            continue;
        }

        // ---------------- A. code probe ------------------------------------
        // 16 rows, each with a single nonzero code at a different k, one launch
        // per group of 16 positions, until the whole row has been covered.
        {
            const i64 rows = 16, n_out = 128, n_in = f.n_in;
            std::vector<u8> w = random_weights(f.wt, n_out, n_in, rng);
            set_scale_fields(f.wt, w, n_out, n_in, rng);

            i8 *dcodes = nullptr;
            f32 *dscales = nullptr;
            _Float16 *dx = nullptr, *dout = nullptr;
            u8 *dw = nullptr;
            HIP_CHECK(hipMalloc(&dcodes, static_cast<size_t>(rows * n_in)));
            HIP_CHECK(hipMalloc(&dscales, static_cast<size_t>(rows) * (n_in / 32) * 4));
            HIP_CHECK(hipMalloc(&dx, static_cast<size_t>(rows * n_in) * 2));
            HIP_CHECK(hipMalloc(&dout, static_cast<size_t>(rows * n_out) * 2));
            HIP_CHECK(hipMalloc(&dw, w.size()));
            HIP_CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));

            std::vector<f32> wrow(static_cast<size_t>(n_in));
            std::vector<_Float16> out(static_cast<size_t>(rows * n_out));
            std::vector<i8> acodes(static_cast<size_t>(rows * n_in));
            std::vector<f32> ascales(static_cast<size_t>(rows) * (n_in / 32));

            double worst = 0.0;
            i64 worst_at = -1, worst_j = -1, skipped = 0;
            int shown = 0;
            for (i64 c0 = 0; c0 < n_in; c0 += rows) {
                std::vector<_Float16> x(static_cast<size_t>(rows * n_in), 0);
                for (i64 m = 0; m < rows; m++)
                    x[static_cast<size_t>(m) * n_in + ((c0 + m) % n_in)] = 1.0f;
                HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
                gemm_dp4a_dispatch(static_cast<int>(f.wt), dw, n_out, n_in,
                                   static_cast<size_t>(dtype_row_bytes(f.wt, n_in)),
                                   dx, rows, dout, dcodes, dscales);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(out.data(), dout, out.size() * 2,
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(acodes.data(), dcodes, acodes.size(),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ascales.data(), dscales, ascales.size() * 4,
                                    hipMemcpyDeviceToHost));

                for (i64 j = 0; j < n_out; j++) {
                    dequant_row(f.wt,
                                w.data() + static_cast<size_t>(j) *
                                               static_cast<size_t>(dtype_row_bytes(f.wt, n_in)),
                                wrow.data(), n_in);
                    for (i64 m = 0; m < rows; m++) {
                        const i64 k = (c0 + m) % n_in;
                        const double sa = ascales[static_cast<size_t>(m) * (n_in / 32) + k / 32];
                        const double code =
                            static_cast<double>(acodes[static_cast<size_t>(m) * n_in + k]);
                        const double want =
                            sa * code * static_cast<double>(wrow[static_cast<size_t>(k)]);
                        const double got = static_cast<double>(
                            static_cast<f32>(out[static_cast<size_t>(m) * n_out + j]));
                        // The magnitude of the single term, so a position whose
                        // weight is accidentally ~0 is judged by its own scale.
                        const double mag = std::fabs(sa * code * static_cast<double>(wrow[static_cast<size_t>(k)]));
                        // A weight the host decoder puts at exactly zero has no
                        // relative error to measure: Q6_K's center code does
                        // this on purpose. Counted, not hidden.
                        if (mag < 1e-30) {
                            skipped++;
                            continue;
                        }
                        const double rel = std::fabs(got - want) / mag;
                        if (rel > worst) {
                            worst = rel;
                            worst_at = k;
                            worst_j = j;
                        }
                        // The first few disagreements, with coordinates: a
                        // wrong nibble, a swapped sub-block scale and a shifted
                        // bit plane all look different here.
                        if (rel > kFp16Round && shown < 4) {
                            std::printf("     mismatch k=%lld m=%lld out=%lld: got "
                                        "%.6g want %.6g (code %lld, scale %.4g, "
                                        "weight %.6g)\n",
                                        static_cast<long long>(k),
                                        static_cast<long long>(m),
                                        static_cast<long long>(j), got, want,
                                        static_cast<long long>(static_cast<int>(code)),
                                        sa, static_cast<double>(wrow[static_cast<size_t>(k)]));
                            shown++;
                        }
                    }
                }
            }
            const bool ok = worst < kFp16Round;
            all_ok = all_ok && ok;
            std::printf("  A. code probe over %lld positions x %lld outputs: "
                        "worst relative error %.3g at k=%lld out=%lld (%lld "
                        "zero-weight positions skipped)  %s\n",
                        static_cast<long long>(n_in), static_cast<long long>(n_out),
                        worst, static_cast<long long>(worst_at),
                        static_cast<long long>(worst_j),
                        static_cast<long long>(skipped), ok ? "OK" : "FAIL");
            HIP_CHECK(hipFree(dcodes));
            HIP_CHECK(hipFree(dscales));
            HIP_CHECK(hipFree(dx));
            HIP_CHECK(hipFree(dout));
            HIP_CHECK(hipFree(dw));
        }

        // ---------------- B. random GEMM -----------------------------------
        // Partial tiles on both axes: 17 rows is not a multiple of BM and 200
        // outputs is not a multiple of BN, so the bound checks are exercised.
        {
            const i64 rows = 17, n_out = 200, n_in = f.n_in;
            std::vector<u8> w = random_weights(f.wt, n_out, n_in, rng);
            set_scale_fields(f.wt, w, n_out, n_in, rng);

            std::uniform_real_distribution<float> xr(-4.0f, 4.0f);
            std::vector<_Float16> x(static_cast<size_t>(rows * n_in));
            for (auto &v : x) v = static_cast<_Float16>(xr(rng));

            i8 *dcodes = nullptr;
            f32 *dscales = nullptr;
            _Float16 *dx = nullptr, *dout = nullptr;
            u8 *dw = nullptr;
            HIP_CHECK(hipMalloc(&dcodes, static_cast<size_t>(rows * n_in)));
            HIP_CHECK(hipMalloc(&dscales, static_cast<size_t>(rows) * (n_in / 32) * 4));
            HIP_CHECK(hipMalloc(&dx, x.size() * 2));
            HIP_CHECK(hipMalloc(&dout, static_cast<size_t>(rows * n_out) * 2));
            HIP_CHECK(hipMalloc(&dw, w.size()));
            HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));

            gemm_dp4a_dispatch(static_cast<int>(f.wt), dw, n_out, n_in,
                               static_cast<size_t>(dtype_row_bytes(f.wt, n_in)), dx,
                               rows, dout, dcodes, dscales);
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<_Float16> out(static_cast<size_t>(rows * n_out));
            std::vector<i8> acodes(static_cast<size_t>(rows * n_in));
            std::vector<f32> ascales(static_cast<size_t>(rows) * (n_in / 32));
            HIP_CHECK(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(acodes.data(), dcodes, acodes.size(),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(ascales.data(), dscales, ascales.size() * 4,
                                hipMemcpyDeviceToHost));

            // Reference in double, from the host decoder and the kernel's own
            // activation codes. The tolerance is relative to the sum of the
            // absolute terms, which is what cancellation cannot shrink.
            std::vector<f32> wrow(static_cast<size_t>(n_in));
            double worst = 0.0, worst_abs = 0.0;
            i64 worst_m = -1, worst_j = -1;
            for (i64 j = 0; j < n_out; j++) {
                dequant_row(f.wt,
                            w.data() + static_cast<size_t>(j) *
                                           static_cast<size_t>(dtype_row_bytes(f.wt, n_in)),
                            wrow.data(), n_in);
                for (i64 m = 0; m < rows; m++) {
                    double ref = 0.0, mag = 0.0;
                    for (i64 k = 0; k < n_in; k++) {
                        const double a =
                            static_cast<double>(ascales[static_cast<size_t>(m) * (n_in / 32) + k / 32]) *
                            static_cast<double>(acodes[static_cast<size_t>(m) * n_in + k]);
                        const double term = a * static_cast<double>(wrow[static_cast<size_t>(k)]);
                        ref += term;
                        mag += std::fabs(term);
                    }
                    const double got = static_cast<double>(
                        static_cast<f32>(out[static_cast<size_t>(m) * n_out + j]));
                    const double rel = std::fabs(got - ref) / (mag > 1e-30 ? mag : 1e-30);
                    if (rel > worst) {
                        worst = rel;
                        worst_abs = std::fabs(got - ref);
                        worst_m = m;
                        worst_j = j;
                    }
                }
            }
            // fp16's own rounding step (4.9e-4) dominates here: the sum of
            // magnitudes is an upper bound on |ref|, so this stays below
            // 4.9e-4 * |ref|/sum|terms| plus f32 folding noise. A staging bug
            // lands at 1e-1 or worse.
            const bool ok = worst < kFp16Round;
            all_ok = all_ok && ok;
            std::printf("  B. random GEMM %lldx%lld K=%lld: worst relative-to-"
                        "magnitude error %.3g (abs %.3g) at (%lld,%lld)  %s\n\n",
                        static_cast<long long>(rows), static_cast<long long>(n_out),
                        static_cast<long long>(n_in), worst, worst_abs,
                        static_cast<long long>(worst_m), static_cast<long long>(worst_j),
                        ok ? "OK" : "FAIL");
            HIP_CHECK(hipFree(dcodes));
            HIP_CHECK(hipFree(dscales));
            HIP_CHECK(hipFree(dx));
            HIP_CHECK(hipFree(dout));
            HIP_CHECK(hipFree(dw));
        }
    }

    std::printf("%s\n", all_ok ? "ALL FORMATS STAGE EVERY CODE THE HOST DECODER "
                                 "SEES"
                               : "*** A FORMAT MIS-STAGES ITS WEIGHTS — the int8 "
                                 "path must stay behind its flag ***");
    return all_ok ? 0 : 1;
}
