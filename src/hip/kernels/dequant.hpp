// ============================================================================
//  dequant.hpp — GGML block decoders that run on the GPU.
//
//  One primitive drives everything: dequant_chunk<T>(block, chunk, out)
//  materializes 32 consecutive values of a block into registers. Every GGUF
//  format is a whole number of 32-value chunks (one for the legacy 32-value
//  blocks, eight for the 256-value K-quants, one for raw fp16/bf16/fp32), so
//  the GEMM, the GEMV and the embedding gather all share the same inner loop
//  and the same tiling rule: "K tiles are multiples of 32".
//
//  These are the device twins of src/quant.cpp. Reorder a block here and you
//  must reorder it there too, or tests/test_quant.cpp will catch the drift.
// ============================================================================
#ifndef KRK_KERNEL_DEQUANT_HPP
#define KRK_KERNEL_DEQUANT_HPP

#include "../krk_hip.hpp"

namespace krk {

__device__ __forceinline__ u32 d_u32(const u8 *p) {
    return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
           (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}

__device__ __forceinline__ f32 bf16_to_fp32_dev(u16 h) {
    return __uint_as_float(static_cast<u32>(h) << 16);
}

// UE4M3 scale byte to fp32 (ggml_ue4m3_to_fp32), device twin of the
// host helper in src/quant.cpp. ldexp(v, e) is v * 2^e and every 2^e
// here is exact in fp32, so we build it with __uint_as_float.
__device__ __forceinline__ f32 ue4m3_to_fp32_dev(u8 x) {
    if (x == 0 || x == 0x7F) return 0.f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 7;
    const f32 raw =
        (exp == 0)
            ? static_cast<f32>(man) * __uint_as_float((127 - 9) << 23)
            : (1.f + static_cast<f32>(man) * 0.125f) *
                  __uint_as_float(static_cast<u32>(exp - 7 + 127) << 23);
    return raw * 0.5f;
}

// E8M0 exponent byte to MXFP4's half scale (its value table is doubled).
// Dev twin of e8m0_to_fp32_half in src/quant.cpp; the two denormal bytes
// (2^-128, 2^-127) are the x<2 branch.
__device__ __forceinline__ f32 e8m0_half_dev(u8 x) {
    return __uint_as_float((x < 2) ? (0x00200000u << x)
                                   : (static_cast<u32>(x - 1) << 23));
}

__device__ static const i8 kDevValuesFp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
// ROCmFPX fork value set (kvalues_rocmfp4): top level 10, not 12.
__device__ static const i8 kDevValuesRocmFp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10,
};
__device__ static const i8 kDevIq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
__device__ static const u8 kDevMaskIq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};
__device__ static const u8 kDevSignsIq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
__device__ static const u64 kDevIq2xxsGrid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// ---------------------------------------------------------------------------
// block geometry, expressed in 32-value chunks
// ---------------------------------------------------------------------------

// How many 32-value chunks one block holds: 1 for the legacy 32-value blocks
// and for raw f16/bf16/f32, 8 for the 256-value K-quants.
template <DType T>
struct QTraits {
    static constexpr int chunks = 1;
};

#define KRK_QTRAIT(T, CHUNKS)                                                      \
    template <>                                                                    \
    struct QTraits<T> {                                                            \
        static constexpr int chunks = CHUNKS;                                      \
    }

KRK_QTRAIT(DType::F16, 1);
KRK_QTRAIT(DType::BF16, 1);
KRK_QTRAIT(DType::F32, 1);
KRK_QTRAIT(DType::Q4_0, 1);
KRK_QTRAIT(DType::Q4_1, 1);
KRK_QTRAIT(DType::Q5_0, 1);
KRK_QTRAIT(DType::Q5_1, 1);
KRK_QTRAIT(DType::Q8_0, 1);
KRK_QTRAIT(DType::Q8_1, 1);
KRK_QTRAIT(DType::IQ4_NL, 1);
KRK_QTRAIT(DType::IQ2_XXS, 8);
KRK_QTRAIT(DType::IQ2_XS, 8);
KRK_QTRAIT(DType::IQ3_XXS, 8);
KRK_QTRAIT(DType::IQ1_S, 8);
KRK_QTRAIT(DType::IQ3_S, 8);
KRK_QTRAIT(DType::IQ2_S, 8);
KRK_QTRAIT(DType::IQ1_M, 8);
KRK_QTRAIT(DType::IQ4_XS, 8);
KRK_QTRAIT(DType::Q1_0, 4);
KRK_QTRAIT(DType::Q2_0, 4);
KRK_QTRAIT(DType::Q2_0_64, 2);
KRK_QTRAIT(DType::NVFP4, 2);
KRK_QTRAIT(DType::MXFP4, 1);
KRK_QTRAIT(DType::ROCMFP4, 1);
KRK_QTRAIT(DType::ROCMFP4_FAST, 1);
KRK_QTRAIT(DType::Q2_K, 8);
KRK_QTRAIT(DType::Q3_K, 8);
KRK_QTRAIT(DType::Q4_K, 8);
KRK_QTRAIT(DType::Q5_K, 8);
KRK_QTRAIT(DType::Q6_K, 8);
KRK_QTRAIT(DType::Q8_K, 8);
KRK_QTRAIT(DType::TQ1_0, 8);
KRK_QTRAIT(DType::TQ2_0, 8);
#undef KRK_QTRAIT

// True byte stride of one block. This is what maps a global chunk index onto
// the block that contains it.
__device__ __forceinline__ int dtype_block_bytes_dev(int t) {
    switch (t) {
        case static_cast<int>(DType::F16): return 64;
        case static_cast<int>(DType::BF16): return 64;
        case static_cast<int>(DType::F32): return 128;
        case static_cast<int>(DType::Q4_0): return 18;
        case static_cast<int>(DType::Q4_1): return 20;
        case static_cast<int>(DType::Q5_0): return 22;
        case static_cast<int>(DType::Q5_1): return 24;
        case static_cast<int>(DType::Q8_0): return 34;
        case static_cast<int>(DType::Q8_1): return 36;
        case static_cast<int>(DType::IQ4_NL): return 18;
        case static_cast<int>(DType::IQ2_XXS): return 66;
        case static_cast<int>(DType::IQ2_XS): return 74;
        // sizeof(block_*) in ggml-common.h, which pins each with its own
        // static_assert. Q1_0, Q2_0 and Q2_0_64 are not QK_K blocks: 128, 128
        // and 64 values respectively, so their byte strides are 18, 34 and 18.
        case static_cast<int>(DType::IQ3_XXS): return 98;
        case static_cast<int>(DType::IQ1_S): return 50;
        case static_cast<int>(DType::IQ3_S): return 110;
        case static_cast<int>(DType::IQ2_S): return 82;
        case static_cast<int>(DType::IQ1_M): return 56;
        case static_cast<int>(DType::Q1_0): return 18;
        case static_cast<int>(DType::Q2_0): return 34;
        case static_cast<int>(DType::Q2_0_64): return 18;
        case static_cast<int>(DType::IQ4_XS): return 136;
        case static_cast<int>(DType::NVFP4): return 36;
        case static_cast<int>(DType::MXFP4): return 17;
        case static_cast<int>(DType::ROCMFP4): return 18;
        case static_cast<int>(DType::ROCMFP4_FAST): return 17;
        case static_cast<int>(DType::Q2_K): return 84;
        case static_cast<int>(DType::Q3_K): return 110;
        case static_cast<int>(DType::Q4_K): return 144;
        case static_cast<int>(DType::Q5_K): return 176;
        case static_cast<int>(DType::Q6_K): return 210;
        case static_cast<int>(DType::Q8_K): return 292;
        case static_cast<int>(DType::TQ1_0): return 54;
        case static_cast<int>(DType::TQ2_0): return 66;
        default: return 0;
    }
}

// Chunks per block, at runtime: 8 for the 256-value K-quants, 1 otherwise.
__device__ __forceinline__ int dtype_chunks_per_block_dev(int t) {
    switch (t) {
        case static_cast<int>(DType::Q2_K):
        case static_cast<int>(DType::Q3_K):
        case static_cast<int>(DType::Q4_K):
        case static_cast<int>(DType::Q5_K):
        case static_cast<int>(DType::Q6_K):
        case static_cast<int>(DType::IQ2_XS):
        case static_cast<int>(DType::IQ2_XXS):
        case static_cast<int>(DType::IQ4_XS):
        case static_cast<int>(DType::Q8_K): return 8;
        case static_cast<int>(DType::IQ3_XXS):
        case static_cast<int>(DType::IQ1_S):
        case static_cast<int>(DType::IQ3_S):
        case static_cast<int>(DType::IQ2_S):
        case static_cast<int>(DType::IQ1_M):
        case static_cast<int>(DType::TQ1_0): return 8;
        case static_cast<int>(DType::TQ2_0): return 8;
        case static_cast<int>(DType::Q1_0): return 4;
        case static_cast<int>(DType::NVFP4): return 2;
        case static_cast<int>(DType::Q2_0): return 4;
        case static_cast<int>(DType::Q2_0_64): return 2;
        default: return 1;
    }
}

// Unpacks the 6-bit scale/min pair used by Q4_K/Q5_K (ggml get_scale_min_k4).
__device__ __forceinline__ void dev_scale_min_k4(int j, const u8 *q, u8 *d, u8 *m) {
    if (j < 4) {
        *d = static_cast<u8>(q[j] & 63);
        *m = static_cast<u8>(q[j + 4] & 63);
    } else {
        *d = static_cast<u8>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<u8>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---------------------------------------------------------------------------
// ternary (TQ1_0 / TQ2_0) device helpers
//
// Both formats are planar, so a 32-element "chunk" is never a contiguous byte
// run. See quant.cpp for the host reference of the same layouts.
// ---------------------------------------------------------------------------

// ggml's multiply-high trit decode: ((uint16_t)(q * 3^n) * 3) >> 8, minus 1.
// The & 0xFF is load-bearing: ggml stores the product into a uint8_t first, so
// it wraps to a byte before the *3 >> 8 (see the host decoder in quant.cpp).
__device__ __forceinline__ int tq1_trit_of_dev(int q, int n) {
    const int p = (n == 0) ? 1 : (n == 1) ? 3 : (n == 2) ? 9 : (n == 3) ? 27
                : (n == 4) ? 81 : 243;
    return static_cast<int>(((static_cast<u32>(q * p) & 0xFFu) * 3u) >> 8) - 1;
}

// Element `idx` (0..255) of a TQ1_0 block. 240 elements come from qs in two
// runs of 5 trit planes (32-byte groups then 16-byte groups); the last 16
// come from qh, 4 trits per byte.
__device__ __forceinline__ int tq1_trit_at_dev(const u8 *b, int idx) {
    if (idx < 160) return tq1_trit_of_dev(b[idx % 32], idx / 32);
    if (idx < 240) {
        const int t = idx - 160;
        return tq1_trit_of_dev(b[32 + (t % 16)], t / 16);
    }
    // qh is emitted plane-major: byte index t % 4, trit plane t / 4.
    const int t = idx - 240;
    return tq1_trit_of_dev(b[48 + (t % 4)], t / 4);
}

// True when every code in a TQ2_0 block decodes to exactly 0. TQ2_0 stores
// 0 as the code 1, so an all-zero weight block is the byte pattern 0x55 --
// one 8-byte compare per 32 bytes, and no multiply. This is the whole point
// of the format for MoE paging: a skipped block is a block never read.
__device__ __forceinline__ bool tq2_0_block_is_zero(const u8 *b) {
    const u64 ones = 0x0101010101010101ull;
#pragma unroll
    for (int i = 0; i < 64; i += 8) {
        u64 v;
        __builtin_memcpy(&v, b + i, 8);
        if (v != ones) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Dequantizers for the IQ family added for the GSQ-RCO and Bonsai exports:
// IQ2_XS (17), IQ2_S (22), IQ3_XXS (18), IQ1_S (19), IQ3_S (21), IQ1_M (29)
// and BitNet's Q1_0 (41) / Q2_0 (42). Each writes exactly the 32 values of
// one chunk, matching its host twin in src/quant_fmt_*.inc value for value
// (the host side is the oracle the tests compare against). Included here --
// after the shared tables and helpers and before the ladder that calls them.
// ---------------------------------------------------------------------------
#include "dequant_fmt_a.inc"
#include "dequant_fmt_b.inc"
#include "dequant_fmt_c.inc"

// dequant_chunk — 32 values of one block into registers
// ---------------------------------------------------------------------------

template <DType T>
__device__ __forceinline__ void dequant_chunk(const u8 *b, int chunk, _Float16 *y) {
    if constexpr (T == DType::F16) {
        const u16 *h = reinterpret_cast<const u16 *>(b);
#pragma unroll
        for (int i = 0; i < 32; i++)
            y[i] = *reinterpret_cast<const _Float16 *>(&h[i]);
    } else if constexpr (T == DType::BF16) {
        const u16 *h = reinterpret_cast<const u16 *>(b);
#pragma unroll
        for (int i = 0; i < 32; i++)
            y[i] = static_cast<_Float16>(bf16_to_fp32_dev(h[i]));
    } else if constexpr (T == DType::F32) {
        const f32 *f = reinterpret_cast<const f32 *>(b);
#pragma unroll
        for (int i = 0; i < 32; i++) y[i] = static_cast<_Float16>(f[i]);
    } else if constexpr (T == DType::Q4_0) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
#pragma unroll
        for (int i = 0; i < 16; i++) {
            y[i] = static_cast<_Float16>(d * static_cast<f32>((b[2 + i] & 0xF) - 8));
            y[i + 16] = static_cast<_Float16>(d * static_cast<f32>((b[2 + i] >> 4) - 8));
        }
    } else if constexpr (T == DType::Q4_1) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 m = d_h2f(static_cast<u16>(b[2] | (b[3] << 8)));
#pragma unroll
        for (int i = 0; i < 16; i++) {
            y[i] = static_cast<_Float16>(d * static_cast<f32>(b[4 + i] & 0xF) + m);
            y[i + 16] = static_cast<_Float16>(d * static_cast<f32>(b[4 + i] >> 4) + m);
        }
    } else if constexpr (T == DType::Q5_0) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        const u32 qh = d_u32(b + 2);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int lo = (b[6 + i] & 0xF) | (static_cast<int>((qh >> i) & 1u) << 4);
            const int hi = (b[6 + i] >> 4) |
                           (static_cast<int>((qh >> (i + 16)) & 1u) << 4);
            y[i] = static_cast<_Float16>(d * static_cast<f32>(lo - 16));
            y[i + 16] = static_cast<_Float16>(d * static_cast<f32>(hi - 16));
        }
    } else if constexpr (T == DType::Q5_1) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 m = d_h2f(static_cast<u16>(b[2] | (b[3] << 8)));
        const u32 qh = d_u32(b + 4);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int lo = (b[8 + i] & 0xF) | (static_cast<int>((qh >> i) & 1u) << 4);
            const int hi = (b[8 + i] >> 4) |
                           (static_cast<int>((qh >> (i + 16)) & 1u) << 4);
            y[i] = static_cast<_Float16>(d * static_cast<f32>(lo) + m);
            y[i + 16] = static_cast<_Float16>(d * static_cast<f32>(hi) + m);
        }
    } else if constexpr (T == DType::Q8_0) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
#pragma unroll
        for (int i = 0; i < 32; i++)
            y[i] = static_cast<_Float16>(d * static_cast<f32>(static_cast<i8>(b[2 + i])));
    } else if constexpr (T == DType::Q8_1) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 s = d_h2f(static_cast<u16>(b[2] | (b[3] << 8)));
#pragma unroll
        for (int i = 0; i < 32; i++)
            y[i] = static_cast<_Float16>(
                d * static_cast<f32>(static_cast<i8>(b[4 + i])) + s);
    } else if constexpr (T == DType::IQ4_NL) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
#pragma unroll
        for (int i = 0; i < 16; i++) {
            y[i] = static_cast<_Float16>(d * static_cast<f32>(kDevIq4nl[b[2 + i] & 0xF]));
            y[i + 16] = static_cast<_Float16>(d * static_cast<f32>(kDevIq4nl[b[2 + i] >> 4]));
        }
    } else if constexpr (T == DType::IQ2_XXS) {
        // 2.3125 bpw. Chunk c = 32 values from the 8-byte group at
        // b+2+8*c: aux32[0] holds four codebook bytes (8 values each),
        // aux32[1] holds the scale nibble (top) and 4x7 sign bits.
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        u32 aux32[2];
        aux32[0] = d_u32(b + 2 + 8 * chunk);
        aux32[1] = d_u32(b + 6 + 8 * chunk);
        const u8 *aux8 = reinterpret_cast<const u8 *>(aux32);
        const f32 db = d * (0.5f + static_cast<f32>(aux32[1] >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const u64 grid = kDevIq2xxsGrid[aux8[l]];
            const u8 signs = kDevSignsIq2xs[(aux32[1] >> (7 * l)) & 127u];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const f32 g = static_cast<f32>(
                    static_cast<int>((grid >> (8 * j)) & 0xFFu));
                y[8 * l + j] = static_cast<_Float16>(
                    db * g * ((signs & kDevMaskIq2xs[j]) ? -1.f : 1.f));
            }
        }
    } else if constexpr (T == DType::IQ4_XS) {
        // 4.25 bpw. Chunk c = 32 values from qs[c*16..], split into
        // two 16-value nibbles sharing scale ls.
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        const u16 scales_h = static_cast<u16>(b[2] | (b[3] << 8));
        const u8 *scales_l = b + 4;
        const u8 *qs = b + 8 + 16 * chunk;
        const int ls = ((scales_l[chunk / 2] >> (4 * (chunk % 2))) & 0xF) |
                       (((scales_h >> (2 * chunk)) & 3) << 4);
        const f32 dl = d * static_cast<f32>(ls - 32);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            y[j] = static_cast<_Float16>(
                dl * static_cast<f32>(kDevIq4nl[qs[j] & 0xF]));
            y[j + 16] = static_cast<_Float16>(
                dl * static_cast<f32>(kDevIq4nl[qs[j] >> 4]));
        }
    } else if constexpr (T == DType::NVFP4) {
        // E2M1 weights, one UE4M3 scale per 16-element sub-block.
        // Chunk c covers sub-blocks s = 2*c and s = 2*c + 1.
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int s = 2 * chunk + k;
            const f32 d = ue4m3_to_fp32_dev(b[s]);
            const u8 *qs = b + 4 + 8 * s;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                y[k * 16 + j] = static_cast<_Float16>(
                    static_cast<f32>(kDevValuesFp4[qs[j] & 0xF]) * d);
                y[k * 16 + j + 8] = static_cast<_Float16>(
                    static_cast<f32>(kDevValuesFp4[qs[j] >> 4]) * d);
            }
        }
    } else if constexpr (T == DType::MXFP4) {
        // One E8M0 exponent then 16 packed nibbles: low -> j, high -> j+16.
        const f32 d = e8m0_half_dev(b[0]);
        const u8 *qs = b + 1;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            y[j] = static_cast<_Float16>(
                static_cast<f32>(kDevValuesFp4[qs[j] & 0xF]) * d);
            y[j + 16] = static_cast<_Float16>(
                static_cast<f32>(kDevValuesFp4[qs[j] >> 4]) * d);
        }
    } else if constexpr (T == DType::ROCMFP4) {
        // 16 packed nibbles then two UE4M3 half-block scales.
        const f32 d0 = ue4m3_to_fp32_dev(b[16]);
        const f32 d1 = ue4m3_to_fp32_dev(b[17]);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            y[j] = static_cast<_Float16>(
                static_cast<f32>(kDevValuesRocmFp4[b[j] & 0xF]) * d0);
            y[j + 16] = static_cast<_Float16>(
                static_cast<f32>(kDevValuesRocmFp4[b[j] >> 4]) * d1);
        }
    } else if constexpr (T == DType::ROCMFP4_FAST) {
        const f32 d = ue4m3_to_fp32_dev(b[16]);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            y[j] = static_cast<_Float16>(
                static_cast<f32>(kDevValuesRocmFp4[b[j] & 0xF]) * d);
            y[j + 16] = static_cast<_Float16>(
                static_cast<f32>(kDevValuesRocmFp4[b[j] >> 4]) * d);
        }
    } else if constexpr (T == DType::Q2_K) {
        // Four-plane layout: chunk c is group g = c/4, plane j = c%4.
        const f32 d = d_h2f(static_cast<u16>(b[80] | (b[81] << 8)));
        const f32 dmin = d_h2f(static_cast<u16>(b[82] | (b[83] << 8)));
        const int g = chunk / 4;
        const int j = chunk % 4;
        const int shift = j * 2;
        const u8 *q = b + 16 + g * 32;
        const u8 sc0 = b[g * 8 + j * 2];
        const u8 sc1 = b[g * 8 + j * 2 + 1];
        const f32 dl0 = d * static_cast<f32>(sc0 & 0xF);
        const f32 ml0 = dmin * static_cast<f32>(sc0 >> 4);
        const f32 dl1 = d * static_cast<f32>(sc1 & 0xF);
        const f32 ml1 = dmin * static_cast<f32>(sc1 >> 4);
#pragma unroll
        for (int l = 0; l < 16; l++) {
            y[l] = static_cast<_Float16>(
                dl0 * static_cast<f32>((q[l] >> shift) & 3) - ml0);
            y[l + 16] = static_cast<_Float16>(
                dl1 * static_cast<f32>((q[l + 16] >> shift) & 3) - ml1);
        }
    } else if constexpr (T == DType::Q3_K) {
        const f32 d_all = d_h2f(static_cast<u16>(b[108] | (b[109] << 8)));
        const u8 *hm = b;
        const u8 *q = b + 32;
        const u8 *scales_raw = b + 96;
        // 16 six-bit scales shuffled across 12 bytes (ggml unpack, verbatim).
        u32 aux[4];
        const u32 a0 = d_u32(scales_raw);
        const u32 a1 = d_u32(scales_raw + 4);
        const u32 a2 = d_u32(scales_raw + 8);
        aux[0] = (a0 & 0x0f0f0f0fu) | (((a2 >> 0) & 0x03030303u) << 4);
        aux[1] = (a1 & 0x0f0f0f0fu) | (((a2 >> 2) & 0x03030303u) << 4);
        aux[2] = ((a0 >> 4) & 0x0f0f0f0fu) | (((a2 >> 4) & 0x03030303u) << 4);
        aux[3] = ((a1 >> 4) & 0x0f0f0f0fu) | (((a2 >> 6) & 0x03030303u) << 4);
        const i8 *scales = reinterpret_cast<const i8 *>(aux);
        const int g = chunk / 4;
        const int j = chunk % 4;
        const int shift = j * 2;
        const u8 m = static_cast<u8>(1u << (g * 4 + j));
        const u8 *qg = q + g * 32;
        const f32 dl0 = d_all * static_cast<f32>(scales[g * 8 + j * 2] - 32);
        const f32 dl1 = d_all * static_cast<f32>(scales[g * 8 + j * 2 + 1] - 32);
#pragma unroll
        for (int l = 0; l < 16; l++) {
            y[l] = static_cast<_Float16>(
                dl0 * static_cast<f32>(static_cast<i8>((qg[l] >> shift) & 3) -
                                       ((hm[l] & m) ? 0 : 4)));
            y[16 + l] = static_cast<_Float16>(
                dl1 * static_cast<f32>(static_cast<i8>((qg[l + 16] >> shift) & 3) -
                                       ((hm[l + 16] & m) ? 0 : 4)));
        }
    } else if constexpr (T == DType::Q4_K) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 dmin = d_h2f(static_cast<u16>(b[2] | (b[3] << 8)));
        const u8 *scales = b + 4;
        const u8 *q = b + 16;
        const int g = chunk / 2;
        const int half = chunk % 2;
        u8 si, mi;
        dev_scale_min_k4(g * 2 + half, scales, &si, &mi);
        const f32 dl = d * static_cast<f32>(si);
        const f32 mn = dmin * static_cast<f32>(mi);
        // The 32 packed bytes this chunk consumes are one contiguous run, so
        // load them as 8 dwords and shift the nibbles out. The byte-at-a-time
        // form compiled to 32 separate 8-bit loads per chunk on both producers
        // (GEMV rows and the prefill weight tile) — the single densest source
        // of memory instructions in the whole pipeline. Both bytes of a dword
        // are used (nibbles 8e and 8e+4), so nothing is fetched twice.
        const int shift = half == 0 ? 0 : 4;
#pragma unroll
        for (int w = 0; w < 8; w++) {
            const u32 pack = d_u32(q + g * 32 + w * 4);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int v = static_cast<int>((pack >> (8 * e + shift)) & 0xF);
                y[w * 4 + e] = static_cast<_Float16>(dl * static_cast<f32>(v) - mn);
            }
        }
    } else if constexpr (T == DType::Q5_K) {
        const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 dmin = d_h2f(static_cast<u16>(b[2] | (b[3] << 8)));
        const u8 *scales = b + 4;
        const u8 *qh = b + 16;
        const u8 *q = b + 48;
        const int g = chunk / 2;
        const int half = chunk % 2;
        u8 si, mi;
        dev_scale_min_k4(g * 2 + half, scales, &si, &mi);
        const f32 dl = d * static_cast<f32>(si);
        const f32 mn = dmin * static_cast<f32>(mi);
        const u8 bit = static_cast<u8>(half == 0 ? (1u << (2 * g)) : (2u << (2 * g)));
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const int base = half == 0 ? (q[g * 32 + l] & 0xF) : (q[g * 32 + l] >> 4);
            const int v = base + ((qh[l] & bit) ? 16 : 0);
            y[l] = static_cast<_Float16>(dl * static_cast<f32>(v) - mn);
        }
    } else if constexpr (T == DType::Q6_K) {
        const f32 d = d_h2f(static_cast<u16>(b[208] | (b[209] << 8)));
        const int g = chunk / 4;
        const int p = chunk % 4;
        const u8 *ql = b + g * 64;
        const u8 *qh = b + 128 + g * 32;
        const i8 *sc = reinterpret_cast<const i8 *>(b + 192 + g * 8);
#pragma unroll
        for (int l = 0; l < 32; l++) {
            int qv;
            if (p == 0) qv = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
            else if (p == 1) qv = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (p == 2) qv = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else qv = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            const int is = l / 16;
            y[l] = static_cast<_Float16>(d * static_cast<f32>(sc[is + 2 * p]) *
                                         static_cast<f32>(qv - 32));
        }
    } else if constexpr (T == DType::Q8_K) {
        const f32 d = __uint_as_float(d_u32(b));
        const i8 *qs = reinterpret_cast<const i8 *>(b + 4);
#pragma unroll
        for (int l = 0; l < 32; l++)
            y[l] = static_cast<_Float16>(d * static_cast<f32>(qs[chunk * 32 + l]));
    } else if constexpr (T == DType::TQ2_0) {
        // 2 bits per element, q in {0,1,2} -> q - 1. Planar: chunks 0-3 use
        // bytes 0..31 with shifts 0/2/4/6, chunks 4-7 use bytes 32..63.
        const f32 d = d_h2f(static_cast<u16>(b[64] | (b[65] << 8)));
        const int off = (chunk >= 4) ? 32 : 0;
        const int sh = (chunk & 3) * 2;
#pragma unroll
        for (int m = 0; m < 32; m++)
            y[m] = static_cast<_Float16>(
                d * static_cast<f32>(static_cast<int>((b[off + m] >> sh) & 3) - 1));
    } else if constexpr (T == DType::TQ1_0) {
        // 5 trits per byte, base-3 packed; 54-byte block.
        const f32 d = d_h2f(static_cast<u16>(b[52] | (b[53] << 8)));
#pragma unroll
        for (int m = 0; m < 32; m++)
            y[m] = static_cast<_Float16>(
                d * static_cast<f32>(tq1_trit_at_dev(b, chunk * 32 + m)));
    } else if constexpr (T == DType::IQ2_S) {
        dequant_chunk_iq2_s(b, chunk, y);
    } else if constexpr (T == DType::IQ2_XS) {
        dequant_chunk_iq2_xs(b, chunk, y);
    } else if constexpr (T == DType::IQ3_XXS) {
        dequant_chunk_iq3_xxs(b, chunk, y);
    } else if constexpr (T == DType::IQ1_S) {
        dequant_chunk_iq1_s(b, chunk, y);
    } else if constexpr (T == DType::IQ3_S) {
        dequant_chunk_iq3_s(b, chunk, y);
    } else if constexpr (T == DType::IQ1_M) {
        dequant_chunk_iq1_m(b, chunk, y);
    } else if constexpr (T == DType::Q1_0) {
        dequant_chunk_q1_0(b, chunk, y);
    } else if constexpr (T == DType::Q2_0) {
        dequant_chunk_q2_0(b, chunk, y);
    } else if constexpr (T == DType::Q2_0_64) {
        dequant_chunk_q2_0_64(b, chunk, y);
    } else {
        (void)b;
        (void)chunk;
        (void)y;
    }
}

// Runtime-dtype chunk decode: the branch is uniform across a launch, so it
// resolves to a jump table of fully inlined decoders.
__device__ __forceinline__ void dequant_chunk_dev(int t, const u8 *b, int chunk,
                                                  _Float16 *y) {
    switch (t) {
        case static_cast<int>(DType::F16): dequant_chunk<DType::F16>(b, chunk, y); break;
        case static_cast<int>(DType::BF16): dequant_chunk<DType::BF16>(b, chunk, y); break;
        case static_cast<int>(DType::F32): dequant_chunk<DType::F32>(b, chunk, y); break;
        case static_cast<int>(DType::Q4_0): dequant_chunk<DType::Q4_0>(b, chunk, y); break;
        case static_cast<int>(DType::Q4_1): dequant_chunk<DType::Q4_1>(b, chunk, y); break;
        case static_cast<int>(DType::Q5_0): dequant_chunk<DType::Q5_0>(b, chunk, y); break;
        case static_cast<int>(DType::Q5_1): dequant_chunk<DType::Q5_1>(b, chunk, y); break;
        case static_cast<int>(DType::Q8_0): dequant_chunk<DType::Q8_0>(b, chunk, y); break;
        case static_cast<int>(DType::Q8_1): dequant_chunk<DType::Q8_1>(b, chunk, y); break;
        case static_cast<int>(DType::IQ4_NL): dequant_chunk<DType::IQ4_NL>(b, chunk, y); break;
        case static_cast<int>(DType::IQ2_XXS): dequant_chunk<DType::IQ2_XXS>(b, chunk, y); break;
        case static_cast<int>(DType::IQ4_XS): dequant_chunk<DType::IQ4_XS>(b, chunk, y); break;
        case static_cast<int>(DType::NVFP4): dequant_chunk<DType::NVFP4>(b, chunk, y); break;
        case static_cast<int>(DType::MXFP4): dequant_chunk<DType::MXFP4>(b, chunk, y); break;
        case static_cast<int>(DType::ROCMFP4): dequant_chunk<DType::ROCMFP4>(b, chunk, y); break;
        case static_cast<int>(DType::ROCMFP4_FAST): dequant_chunk<DType::ROCMFP4_FAST>(b, chunk, y); break;
        case static_cast<int>(DType::Q2_K): dequant_chunk<DType::Q2_K>(b, chunk, y); break;
        case static_cast<int>(DType::Q3_K): dequant_chunk<DType::Q3_K>(b, chunk, y); break;
        case static_cast<int>(DType::Q4_K): dequant_chunk<DType::Q4_K>(b, chunk, y); break;
        case static_cast<int>(DType::Q5_K): dequant_chunk<DType::Q5_K>(b, chunk, y); break;
        case static_cast<int>(DType::Q6_K): dequant_chunk<DType::Q6_K>(b, chunk, y); break;
        case static_cast<int>(DType::Q8_K): dequant_chunk<DType::Q8_K>(b, chunk, y); break;
        case static_cast<int>(DType::TQ1_0): dequant_chunk<DType::TQ1_0>(b, chunk, y); break;
        case static_cast<int>(DType::TQ2_0): dequant_chunk<DType::TQ2_0>(b, chunk, y); break;
        case static_cast<int>(DType::IQ2_XS): dequant_chunk<DType::IQ2_XS>(b, chunk, y); break;
        case static_cast<int>(DType::IQ2_S): dequant_chunk<DType::IQ2_S>(b, chunk, y); break;
        case static_cast<int>(DType::IQ3_XXS): dequant_chunk<DType::IQ3_XXS>(b, chunk, y); break;
        case static_cast<int>(DType::IQ1_S): dequant_chunk<DType::IQ1_S>(b, chunk, y); break;
        case static_cast<int>(DType::IQ3_S): dequant_chunk<DType::IQ3_S>(b, chunk, y); break;
        case static_cast<int>(DType::IQ1_M): dequant_chunk<DType::IQ1_M>(b, chunk, y); break;
        case static_cast<int>(DType::Q1_0): dequant_chunk<DType::Q1_0>(b, chunk, y); break;
        case static_cast<int>(DType::Q2_0): dequant_chunk<DType::Q2_0>(b, chunk, y); break;
        case static_cast<int>(DType::Q2_0_64): dequant_chunk<DType::Q2_0_64>(b, chunk, y); break;
        default:
#pragma unroll
            for (int i = 0; i < 32; i++) y[i] = static_cast<_Float16>(0.0f);
            break;
    }
}

// ---------------------------------------------------------------------------
// chunk-range dot products — the GEMV inner loop
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// packed decode for the two 5/6-bit K-quants (Q5_K, Q6_K)
//
// Measured, not assumed: kraken-bench rows=1 puts Q4_K at 563-674 GB/s and
// Q6_K at 143 GB/s on a 4096x4096 matrix that fits in the 64 MB Infinity
// Cache, and at 195 GB/s where Q4_K does 572 on a 208 MB one. A matrix that
// is *in* cache and still four times slower than its neighbour is not
// bandwidth-bound, it is instruction-bound — and the count agrees: the
// generic path re-reads `p`, which varies per lane because lane L owns chunks
// L, L+32, ... of the same row, through a four-way branch inside the value
// loop, and then converts every value through f32.
//
// The packed form rests on one identity: f16(1024 + n) is *exactly*
// 0x6400 | n for n in [0, 1023] (that exponent field is 2^10 with a zero
// mantissa), so a 5- or 6-bit code becomes a half with no convert and no
// branch. Weights enter the dot as f16(1024 + code); the 1024 comes back out
// as one f32 correction against the chunk's own activation sum, which the
// same loop produces for free as the dot against 1.0. Four weights cost two
// dword loads, six shift/mask ops, two byte moves and four dot2 — and the
// accumulator is f32 throughout, so this is *more* accurate than the path it
// replaces, which rounded every weight to f16 first.
//
//   Q5_K: w = dl*v - mn     ->  dl*dot - (1024*dl + mn) * sum(x)
//   Q6_K: w = ds*(q - 32)   ->  ds * (dot - 1056 * sum(x))   per 16-value half
//
// kraken-bench 4096x4096 rows=1, gfx1201: Q6_K 143 -> 541 GB/s, head
// (248320x1024) 195 -> 564, Q5_K 352 -> 535, Q4_K unchanged at 584.
// ---------------------------------------------------------------------------

#if defined(KRK_HAS_DOT2)

// f16 1.0 in both halves: the second operand of the activation-sum dot.
constexpr u32 kDotOnes = 0x3C003C00u;

// Four 5/6-bit codes (one per byte) -> two packed f16 pairs, each half
// exactly f16(1024 + code). The f16 exponent byte of both halves is the
// constant 0x64, so a pair is "two code bytes moved into the low byte of each
// half, or in 0x6400".
//
// This was written first as v_perm_b32 (one instruction, byte 0/4/1/4 of an
// 8-byte source pair). The inline asm compiled and *silently returned
// garbage* — every Q6_K/Q5_K output was NaN — while this form is correct and
// the same speed, so the compiler is already fusing it. Kept as C on purpose.
__device__ __forceinline__ void dot_pair_codes(u32 codes, u32 *w01, u32 *w23) {
    *w01 = (codes & 0x0000003Fu) | ((codes & 0x00003F00u) << 8) | 0x64006400u;
    *w23 = ((codes >> 16) & 0x0000003Fu) | ((codes >> 8) & 0x003F0000u) |
           0x64006400u;
}

// One packed f16 pair of activations, as the dot2 operand. Callers pass
// x + c*32 + l with l a multiple of 4, so this is an aligned 32-bit load.
__device__ __forceinline__ u32 dot_pair_act(const _Float16 *p) {
    return *reinterpret_cast<const u32 *>(p);
}

// Q6_K. A chunk is 32 codes: 32 bytes of `ql` (offset and nibble selected by
// p) plus 32 bytes of `qh`, split into two 16-value halves that carry
// different int8 scales.
__device__ __forceinline__ f32 dot_chunk_q6k(const u8 *b, int cc,
                                              const _Float16 *x) {
    const int g = cc >> 2;
    const int p = cc & 3;
    const u8 *ql = b + g * 64 + ((p & 1) ? 32 : 0);
    const u8 *qh = b + 128 + g * 32;
    const i8 *sc = reinterpret_cast<const i8 *>(b + 192 + g * 8);
    const u32 nsh = (p & 2) ? 4u : 0u; // low nibble for p<2, high for p>=2
    const u32 hsh = static_cast<u32>(p) * 2u;

    f32 ad = 0.0f, as = 0.0f, bd = 0.0f, bs = 0.0f;
#pragma unroll
    for (int l = 0; l < 32; l += 4) {
        const u32 lo = (d_u32(ql + l) >> nsh) & 0x0F0F0F0Fu;
        const u32 hi = (d_u32(qh + l) >> hsh) & 0x03030303u;
        u32 w01 = 0, w23 = 0;
        dot_pair_codes(lo | (hi << 4), &w01, &w23);
        const u32 a01 = dot_pair_act(x + l);
        const u32 a23 = dot_pair_act(x + l + 2);
        if (l < 16) {
            ad = d_dot2_pk(w01, a01, ad);
            ad = d_dot2_pk(w23, a23, ad);
            as = d_dot2_pk(kDotOnes, a01, as);
            as = d_dot2_pk(kDotOnes, a23, as);
        } else {
            bd = d_dot2_pk(w01, a01, bd);
            bd = d_dot2_pk(w23, a23, bd);
            bs = d_dot2_pk(kDotOnes, a01, bs);
            bs = d_dot2_pk(kDotOnes, a23, bs);
        }
    }
    const f32 dd = d_h2f(static_cast<u16>(b[208] | (b[209] << 8)));
    return dd * static_cast<f32>(sc[2 * p]) * (ad - 1056.0f * as) +
           dd * static_cast<f32>(sc[2 * p + 1]) * (bd - 1056.0f * bs);
}

// Q5_K. Same shape; the nibble selector is `half = cc&1` rather than p>>1,
// and both dl and mn are constant across the whole chunk, so one correction
// covers all 32 values.
__device__ __forceinline__ f32 dot_chunk_q5k(const u8 *b, int cc,
                                              const _Float16 *x) {
    const int g = cc >> 1;
    const int half = cc & 1;
    u8 si, mi;
    dev_scale_min_k4(g * 2 + half, b + 4, &si, &mi);
    const u32 nsh = static_cast<u32>(half) * 4u;
    const u32 hsh = static_cast<u32>(g * 2 + half);

    f32 acc = 0.0f, sum = 0.0f;
#pragma unroll
    for (int l = 0; l < 32; l += 4) {
        const u32 lo = (d_u32(b + 48 + g * 32 + l) >> nsh) & 0x0F0F0F0Fu;
        const u32 hi = (d_u32(b + 16 + l) >> hsh) & 0x01010101u;
        u32 w01 = 0, w23 = 0;
        dot_pair_codes(lo | (hi << 4), &w01, &w23);
        const u32 a01 = dot_pair_act(x + l);
        const u32 a23 = dot_pair_act(x + l + 2);
        acc = d_dot2_pk(w01, a01, acc);
        acc = d_dot2_pk(w23, a23, acc);
        sum = d_dot2_pk(kDotOnes, a01, sum);
        sum = d_dot2_pk(kDotOnes, a23, sum);
    }
    const f32 dl =
        d_h2f(static_cast<u16>(b[0] | (b[1] << 8))) * static_cast<f32>(si);
    const f32 mn =
        d_h2f(static_cast<u16>(b[2] | (b[3] << 8))) * static_cast<f32>(mi);
    return dl * acc - (1024.0f * dl + mn) * sum;
}

#endif // KRK_HAS_DOT2

// dot over chunks [c0, c1) of a weight row against fp16 activations.
template <DType T>
__device__ __forceinline__ f32 dot_chunks_t(const u8 *wrow, const _Float16 *x, int c0,
                                            int c1) {
    constexpr int kChunks = QTraits<T>::chunks;
    const int block_bytes = dtype_block_bytes_dev(static_cast<int>(T));
    f32 acc0 = 0, acc1 = 0;
    _Float16 tmp[32];
    for (int c = c0; c < c1; c++) {
        const u8 *blk = wrow + (c / kChunks) * block_bytes;
        dequant_chunk<T>(blk, c % kChunks, tmp);
        const _Float16 *xc = x + static_cast<i64>(c) * 32;
#pragma unroll
        for (int i = 0; i < 32; i += 2) {
            acc0 += static_cast<f32>(tmp[i]) * static_cast<f32>(xc[i]);
            acc1 += static_cast<f32>(tmp[i + 1]) * static_cast<f32>(xc[i + 1]);
        }
    }
    return acc0 + acc1;
}

// The packed specializations. They accumulate in f32 with no intermediate f16
// rounding, so for these two formats they replace the generic loop outright.
#if defined(KRK_HAS_DOT2)
template <>
__device__ __forceinline__ f32 dot_chunks_t<DType::Q6_K>(const u8 *wrow,
                                                         const _Float16 *x,
                                                         int c0, int c1) {
    f32 acc = 0.0f;
    for (int c = c0; c < c1; c++)
        acc += dot_chunk_q6k(wrow + (c >> 3) * 210, c & 7,
                             x + static_cast<i64>(c) * 32);
    return acc;
}

template <>
__device__ __forceinline__ f32 dot_chunks_t<DType::Q5_K>(const u8 *wrow,
                                                         const _Float16 *x,
                                                         int c0, int c1) {
    f32 acc = 0.0f;
    for (int c = c0; c < c1; c++)
        acc += dot_chunk_q5k(wrow + (c >> 3) * 176, c & 7,
                             x + static_cast<i64>(c) * 32);
    return acc;
}
#endif

// Runtime dispatch; the dtype is uniform across a launch.
__device__ __forceinline__ f32 dot_chunks(int t, const u8 *wrow, const _Float16 *x,
                                          int c0, int c1) {
    switch (t) {
        case static_cast<int>(DType::F16): return dot_chunks_t<DType::F16>(wrow, x, c0, c1);
        case static_cast<int>(DType::BF16): return dot_chunks_t<DType::BF16>(wrow, x, c0, c1);
        case static_cast<int>(DType::F32): return dot_chunks_t<DType::F32>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q4_0): return dot_chunks_t<DType::Q4_0>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q4_1): return dot_chunks_t<DType::Q4_1>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q5_0): return dot_chunks_t<DType::Q5_0>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q5_1): return dot_chunks_t<DType::Q5_1>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q8_0): return dot_chunks_t<DType::Q8_0>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q8_1): return dot_chunks_t<DType::Q8_1>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ4_NL): return dot_chunks_t<DType::IQ4_NL>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ2_XXS): return dot_chunks_t<DType::IQ2_XXS>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ4_XS): return dot_chunks_t<DType::IQ4_XS>(wrow, x, c0, c1);
        case static_cast<int>(DType::NVFP4): return dot_chunks_t<DType::NVFP4>(wrow, x, c0, c1);
        case static_cast<int>(DType::MXFP4): return dot_chunks_t<DType::MXFP4>(wrow, x, c0, c1);
        case static_cast<int>(DType::ROCMFP4): return dot_chunks_t<DType::ROCMFP4>(wrow, x, c0, c1);
        case static_cast<int>(DType::ROCMFP4_FAST): return dot_chunks_t<DType::ROCMFP4_FAST>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q2_K): return dot_chunks_t<DType::Q2_K>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q3_K): return dot_chunks_t<DType::Q3_K>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q4_K): return dot_chunks_t<DType::Q4_K>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q5_K): return dot_chunks_t<DType::Q5_K>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q6_K): return dot_chunks_t<DType::Q6_K>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q8_K): return dot_chunks_t<DType::Q8_K>(wrow, x, c0, c1);
        case static_cast<int>(DType::TQ1_0): return dot_chunks_t<DType::TQ1_0>(wrow, x, c0, c1);
        case static_cast<int>(DType::TQ2_0): return dot_chunks_t<DType::TQ2_0>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ2_XS): return dot_chunks_t<DType::IQ2_XS>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ2_S): return dot_chunks_t<DType::IQ2_S>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ3_XXS): return dot_chunks_t<DType::IQ3_XXS>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ1_S): return dot_chunks_t<DType::IQ1_S>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ3_S): return dot_chunks_t<DType::IQ3_S>(wrow, x, c0, c1);
        case static_cast<int>(DType::IQ1_M): return dot_chunks_t<DType::IQ1_M>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q1_0): return dot_chunks_t<DType::Q1_0>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q2_0): return dot_chunks_t<DType::Q2_0>(wrow, x, c0, c1);
        case static_cast<int>(DType::Q2_0_64): return dot_chunks_t<DType::Q2_0_64>(wrow, x, c0, c1);
        default: return 0.0f;
    }
}

// ---------------------------------------------------------------------------
// row dequantization (embedding gather)
// ---------------------------------------------------------------------------

template <DType T>
__device__ __forceinline__ void dequant_row_t(const u8 *src, _Float16 *dst, i64 n) {
    constexpr int kChunks = QTraits<T>::chunks;
    const int block_bytes = dtype_block_bytes_dev(static_cast<int>(T));
    const i64 nch = n / 32;
    for (i64 c = 0; c < nch; c++)
        dequant_chunk<T>(src + (c / kChunks) * block_bytes, static_cast<int>(c % kChunks),
                         dst + c * 32);
}

__device__ __forceinline__ void dequant_row_dev(int t, const u8 *src, _Float16 *dst,
                                                i64 n) {
    switch (t) {
        case static_cast<int>(DType::F16): dequant_row_t<DType::F16>(src, dst, n); break;
        case static_cast<int>(DType::BF16): dequant_row_t<DType::BF16>(src, dst, n); break;
        case static_cast<int>(DType::F32): dequant_row_t<DType::F32>(src, dst, n); break;
        case static_cast<int>(DType::Q4_0): dequant_row_t<DType::Q4_0>(src, dst, n); break;
        case static_cast<int>(DType::Q4_1): dequant_row_t<DType::Q4_1>(src, dst, n); break;
        case static_cast<int>(DType::Q5_0): dequant_row_t<DType::Q5_0>(src, dst, n); break;
        case static_cast<int>(DType::Q5_1): dequant_row_t<DType::Q5_1>(src, dst, n); break;
        case static_cast<int>(DType::Q8_0): dequant_row_t<DType::Q8_0>(src, dst, n); break;
        case static_cast<int>(DType::Q8_1): dequant_row_t<DType::Q8_1>(src, dst, n); break;
        case static_cast<int>(DType::IQ4_NL): dequant_row_t<DType::IQ4_NL>(src, dst, n); break;
        case static_cast<int>(DType::IQ2_XXS): dequant_row_t<DType::IQ2_XXS>(src, dst, n); break;
        case static_cast<int>(DType::IQ4_XS): dequant_row_t<DType::IQ4_XS>(src, dst, n); break;
        case static_cast<int>(DType::NVFP4): dequant_row_t<DType::NVFP4>(src, dst, n); break;
        case static_cast<int>(DType::MXFP4): dequant_row_t<DType::MXFP4>(src, dst, n); break;
        case static_cast<int>(DType::ROCMFP4): dequant_row_t<DType::ROCMFP4>(src, dst, n); break;
        case static_cast<int>(DType::ROCMFP4_FAST): dequant_row_t<DType::ROCMFP4_FAST>(src, dst, n); break;
        case static_cast<int>(DType::Q2_K): dequant_row_t<DType::Q2_K>(src, dst, n); break;
        case static_cast<int>(DType::Q3_K): dequant_row_t<DType::Q3_K>(src, dst, n); break;
        case static_cast<int>(DType::Q4_K): dequant_row_t<DType::Q4_K>(src, dst, n); break;
        case static_cast<int>(DType::Q5_K): dequant_row_t<DType::Q5_K>(src, dst, n); break;
        case static_cast<int>(DType::Q6_K): dequant_row_t<DType::Q6_K>(src, dst, n); break;
        case static_cast<int>(DType::Q8_K): dequant_row_t<DType::Q8_K>(src, dst, n); break;
        case static_cast<int>(DType::TQ1_0): dequant_row_t<DType::TQ1_0>(src, dst, n); break;
        case static_cast<int>(DType::TQ2_0): dequant_row_t<DType::TQ2_0>(src, dst, n); break;
        case static_cast<int>(DType::IQ2_XS): dequant_row_t<DType::IQ2_XS>(src, dst, n); break;
        case static_cast<int>(DType::IQ2_S): dequant_row_t<DType::IQ2_S>(src, dst, n); break;
        case static_cast<int>(DType::IQ3_XXS): dequant_row_t<DType::IQ3_XXS>(src, dst, n); break;
        case static_cast<int>(DType::IQ1_S): dequant_row_t<DType::IQ1_S>(src, dst, n); break;
        case static_cast<int>(DType::IQ3_S): dequant_row_t<DType::IQ3_S>(src, dst, n); break;
        case static_cast<int>(DType::IQ1_M): dequant_row_t<DType::IQ1_M>(src, dst, n); break;
        case static_cast<int>(DType::Q1_0): dequant_row_t<DType::Q1_0>(src, dst, n); break;
        case static_cast<int>(DType::Q2_0): dequant_row_t<DType::Q2_0>(src, dst, n); break;
        case static_cast<int>(DType::Q2_0_64): dequant_row_t<DType::Q2_0_64>(src, dst, n); break;
        default: break;
    }
}

} // namespace krk

#endif // KRK_KERNEL_DEQUANT_HPP
