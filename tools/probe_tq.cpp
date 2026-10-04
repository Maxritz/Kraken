// probe_tq.cpp — validate the TQ1_0/TQ2_0 host decoders against ggml's own
// reference *encoder* (copied from ggml-quants.c, quantize_row_tq{1,2}_0_ref).
//
// The decoders in src/quant.cpp were written from the ggml dequantizers, so
// decoding the encoder's output and comparing to the intended reconstruction is
// an independent check: encoder and decoder were derived from different source
// functions. If a packing detail is wrong, the round trip diverges.
//
// Build (standalone, no HIP):
//   c++ -O2 -Iinclude probe_tq.cpp src/quant.cpp src/common.cpp -o probe_tq
#include "krk/quant.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace krk;

// ---------------------------------------------------------------------------
// ggml reference encoders, verbatim in structure from ggml-quants.c
// ---------------------------------------------------------------------------

static void quantize_row_tq2_0_ref(const float *x, u8 *y, int k) {
    const int QK = 256;
    const int nb = k / QK;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK; j++) amax = std::fabs(x[j]) > amax ? std::fabs(x[j]) : amax;
        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;
        const u16 dh = fp32_to_fp16(d);
        u8 *b = y + i * 66;
        b[64] = static_cast<u8>(dh & 0xFF);
        b[65] = static_cast<u8>(dh >> 8);
        std::memset(b, 0, 64);
        for (size_t j = 0; j < 64; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                u8 q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    int xi = static_cast<int>(std::lroundf(x[m + n * 32] * id)) + 1;
                    q = static_cast<u8>(q + ((xi & 3) << (2 * n)));
                }
                b[j + m] = q;
            }
            x += 4 * 32;
        }
    }
}

static void quantize_row_tq1_0_ref(const float *x, u8 *y, int k) {
    const int QK = 256;
    const int nb = k / QK;
    const u8 qs_size = 48, qh_size = 4;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK; j++) amax = std::fabs(x[j]) > amax ? std::fabs(x[j]) : amax;
        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;
        u8 *b = y + i * 54;
        const u16 dh = fp32_to_fp16(d);
        b[52] = static_cast<u8>(dh & 0xFF);
        b[53] = static_cast<u8>(dh >> 8);
        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < qs_size - qs_size % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                u8 q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = static_cast<int>(std::lroundf(x[m + n * 32] * id)) + 1;
                    q = static_cast<u8>(q * 3);
                    q = static_cast<u8>(q + xi);
                }
                q = static_cast<u8>((static_cast<u16>(q) * 256 + (243 - 1)) / 243);
                b[j + m] = q;
            }
            x += 5 * 32;
        }
        // along 16 bytes
        for (size_t j = qs_size - qs_size % 32; j < qs_size; j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                u8 q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = static_cast<int>(std::lroundf(x[m + n * 16] * id)) + 1;
                    q = static_cast<u8>(q * 3);
                    q = static_cast<u8>(q + xi);
                }
                q = static_cast<u8>((static_cast<u16>(q) * 256 + (243 - 1)) / 243);
                b[j + m] = q;
            }
            x += 5 * 16;
        }
        // 4 elements per byte
        for (size_t j = 0; j < qh_size; ++j) {
            u8 q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = static_cast<int>(std::lroundf(x[j + m * qh_size] * id)) + 1;
                q = static_cast<u8>(q * 3);
                q = static_cast<u8>(q + xi);
            }
            q = static_cast<u8>(q * 3);
            q = static_cast<u8>((static_cast<u16>(q) * 256 + (243 - 1)) / 243);
            b[qs_size + j] = q;
        }
        x += 4 * qh_size;
    }
}

// ---------------------------------------------------------------------------

static int g_fail = 0;
#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL " __VA_ARGS__);                              \
            std::printf("\n");                                               \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

// Round trip: values that quantize exactly to {-1, 0, +1} * d, so the
// reconstruction is exact and any layout error shows up as a mismatch.
static void roundtrip(DType t, const char *name, void (*enc)(const float *, u8 *, int),
                      u64 block_bytes) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> trit(-1, 1);
    const int k = 256 * 8; // eight super-blocks
    std::vector<float> src(k);
    for (int i = 0; i < k; i++) src[i] = static_cast<float>(trit(rng));

    const size_t bytes = static_cast<size_t>(block_bytes) * (k / 256);
    std::vector<u8> blk(bytes);
    enc(src.data(), blk.data(), k);

    std::vector<f32> out(k);
    dequant_row(t, blk.data(), out.data(), k);

    int mism = 0, first = -1;
    for (int i = 0; i < k; i++) {
        // d == 1.0 here (amax of {-1,0,1} is 1), so out must equal src exactly.
        if (std::fabs(out[i] - src[i]) > 1e-6f) {
            if (first < 0) first = i;
            mism++;
        }
    }
    CHECK(mism == 0, "%s round trip: %d/%d mismatch, first at %d (src %.0f out %.4f)",
          name, mism, k, first, first >= 0 ? src[first] : 0.0f,
          first >= 0 ? out[first] : 0.0f);
    if (mism == 0) std::printf("  ok   %s round trip: %d/%d exact\n", name, k, k);

    // Scale handling: d must scale the reconstruction.
    for (int i = 0; i < k; i++) src[i] = static_cast<float>(trit(rng)) * 0.5f;
    enc(src.data(), blk.data(), k);
    dequant_row(t, blk.data(), out.data(), k);
    mism = 0;
    for (int i = 0; i < k; i++)
        if (std::fabs(out[i] - src[i]) > 1e-6f) mism++;
    CHECK(mism == 0, "%s scaled round trip: %d/%d mismatch", name, mism, k);
    if (mism == 0) std::printf("  ok   %s scaled round trip (d=0.5): %d/%d exact\n", name, k, k);

    // vec_dot against the scalar sum of the same decoded values.
    std::vector<f32> act(k);
    for (int i = 0; i < k; i++) act[i] = 0.25f * static_cast<float>((i % 7) - 3);
    enc(src.data(), blk.data(), k);
    dequant_row(t, blk.data(), out.data(), k);
    f64 ref = 0;
    for (int i = 0; i < k; i++) ref += static_cast<f64>(out[i]) * act[i];
    const f32 got = vec_dot(t, blk.data(), act.data(), k);
    CHECK(std::fabs(got - static_cast<f32>(ref)) < 1e-2f,
          "%s vec_dot: got %.4f want %.4f", name, got, static_cast<f32>(ref));
    if (std::fabs(got - static_cast<f32>(ref)) < 1e-2f)
        std::printf("  ok   %s vec_dot matches scalar: %.4f\n", name, got);
}

// The zero-block pattern: an all-zero weight block must decode to all zeros.
// This is the property the MoE paging policy exploits, so it is checked
// explicitly rather than inferred from the round trip.
static void zero_block(DType t, const char *name, u64 block_bytes, bool (*is_zero)(const u8 *)) {
    std::vector<u8> blk(block_bytes);
    std::vector<float> src(256, 0.0f);
    // TQ2_0: code 1 everywhere. TQ1_0: the base-3 packing of 1,1,1,1,1.
    if (t == DType::TQ2_0) {
        std::memset(blk.data(), 0x55, 64);
        const u16 dh = fp32_to_fp16(1.0f);
        blk[64] = static_cast<u8>(dh & 0xFF);
        blk[65] = static_cast<u8>(dh >> 8);
    } else {
        // 1,1,1,1,1 in base 3 = 121 -> ceil(121*256/243) = 128
        for (int i = 0; i < 48; i++) blk[i] = 128;
        // qh packs 4 trits then shifts one place up (q *= 3), so all-ones is
        // 1*3^4 + 1*3^3 + 1*3^2 + 1*3 = 120 -> ceil(120*256/243) = 127
        for (int i = 0; i < 4; i++) blk[48 + i] = 127;
        const u16 dh = fp32_to_fp16(1.0f);
        blk[52] = static_cast<u8>(dh & 0xFF);
        blk[53] = static_cast<u8>(dh >> 8);
    }
    std::vector<f32> out(256);
    dequant_row(t, blk.data(), out.data(), 256);
    int nz = 0;
    for (int i = 0; i < 256; i++)
        if (out[i] != 0.0f) nz++;
    CHECK(nz == 0, "%s all-zero block decodes to %d nonzero", name, nz);
    if (nz == 0) std::printf("  ok   %s all-zero block -> all zeros\n", name);
    if (is_zero) {
        const bool z = is_zero(blk.data());
        CHECK(z, "%s zero-block detector disagrees with the decoder", name);
        if (z) std::printf("  ok   %s zero-block detector agrees\n", name);
    }
    (void)src;
}

int main() {
    std::printf("TQ1_0/TQ2_0 decoder validation (round trip against ggml's encoder)\n");

    std::printf("\ngeometry:\n");
    CHECK(dtype_block_size(DType::TQ1_0) == 256, "TQ1_0 block size");
    CHECK(dtype_block_bytes(DType::TQ1_0) == 54, "TQ1_0 block bytes");
    CHECK(dtype_block_size(DType::TQ2_0) == 256, "TQ2_0 block size");
    CHECK(dtype_block_bytes(DType::TQ2_0) == 66, "TQ2_0 block bytes");
    CHECK(dtype_supported(DType::TQ1_0) && dtype_supported(DType::TQ2_0), "supported");
    std::printf("  ok   TQ1_0 54 B/256 (1.6875 bpw), TQ2_0 66 B/256 (2.0625 bpw)\n");

    std::printf("\nround trips:\n");
    roundtrip(DType::TQ1_0, "TQ1_0", quantize_row_tq1_0_ref, 54);
    roundtrip(DType::TQ2_0, "TQ2_0", quantize_row_tq2_0_ref, 66);

    std::printf("\nzero blocks (the MoE paging property):\n");
    zero_block(DType::TQ1_0, "TQ1_0", 54, nullptr);
    zero_block(DType::TQ2_0, "TQ2_0", 66, nullptr);

    std::printf("\n%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}