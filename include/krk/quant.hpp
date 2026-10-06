// ============================================================================
//  quant.hpp — GGML block quantization formats.
//
//  Type ids are the on-disk GGUF ids and must not be renumbered. Every block
//  layout here mirrors ggml's reference dequantizers byte for byte; the host
//  implementations below double as the correctness oracle for the device
//  kernels (tests/test_quant.cpp compares device output against these).
// ============================================================================
#ifndef KRK_QUANT_HPP
#define KRK_QUANT_HPP

#include "krk/common.hpp"

namespace krk {

enum class DType : i32 {
    Unknown = -1,
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    // IQ2_XS (id 17): the same 256-value block as IQ2_XXS (66 bytes) but
    // with a 16-bit scale per 32 values and 9-bit grid indices, so 74
    // bytes a block. Used by the GSQ-RCO exports.
    IQ2_XS = 17,
    // The remaining IQ family. Same 256-value block shape as IQ2_XXS and the
    // same grid-plus-signs construction, so each is 8 chunks of 32; what
    // differs is the grid table, the scale layout and how many index bits each
    // value gets. Byte sizes below are the reference's own static_assert
    // (sizeof(block_*) in ggml-common.h), not a recomputation.
    IQ3_XXS = 18, // 98 bytes: 3*QK_K/8 of packed grid indices + d
    IQ1_S = 19,   // 50 bytes: QK_K/8 indices + QK_K/16 of high bits + d
    IQ4_NL = 20,
    IQ3_S = 21,   // 110 bytes: qs + qh + signs + 4 scale bytes + d
    IQ2_S = 22,   //  82 bytes: qs + qh + 16 scale bytes + d
    IQ4_XS = 23,
    // IQ1_M (id 29): 56 bytes and NO f16 delta field at all -- the four block
    // scales and the 3-bit group scales share one packed word, which is why
    // this one is not a variant of IQ1_S but its own layout.
    IQ1_M = 29,
    BF16 = 30,
    // BitNet-style ternary: 1.6875 bpw and 2.0625 bpw. Both pack
    // {-1, 0, +1} codes, so a third of the weights are exactly zero --
    // which is a *filter*, not just compression (see ExpertCache's
    // zero-block skipping).
    TQ1_0 = 34,
    TQ2_0 = 35,
    // MXFP4 (ggml id 39): 32-value blocks of E2M1 codes behind one E8M0
    // exponent byte. The value table is doubled, so the scale is used at
    // half strength (ggml_e8m0_to_fp32_half).
    MXFP4 = 39,
    NVFP4 = 40,
    // BitNet's own 1-bit and 2-bit ternary formats (ids 41/42), distinct from
    // TQ1_0/TQ2_0 above: 128-value blocks with a plain f16 delta and no trit
    // packing trick, so the codes are {0, 1} for Q1_0 and {-1, 0, +1, +2} for
    // Q2_0 rather than the 5-value TQ ladder. Q1_0 is 1.125 bpw; used by the
    // Bonsai ternary exports.
    //
    // Q2_0 is the one id with two live geometries in the wild, and the file's
    // own tensor offsets -- not the id -- are what say which one a tensor is:
    //   id 42: 128 values / 34 bytes = 2.125 bpw. ggml-common.h in the
    //          llama-dx fork (Maxritz/LLAMA-DX) has QK2_0 = 128 there, so its
    //          block_q2_0 is {f16 d; u8 qs[32]}. The Ternary-Bonsai Q2_0
    //          exports are this variant and measure exactly 2.125 bpw.
    //   id 48: 64 values / 18 bytes = 2.25 bpw, that fork's own Q2_0_64, and
    //          byte-identical to upstream llama.cpp's Q2_0. A file made by
    //          upstream declares 42 but measures 2.25 bpw (the Swift
    //          Qwen3.8-Flash shards do).
    // Both are decoded here; Model::load picks between them per file.
    Q1_0 = 41,
    Q2_0 = 42,
    Q2_0_64 = 48,
    // ROCmFPX fork formats (github.com/charlie12345/ROCmFPX). These ids
    // are not upstream: 100 is Q4_0_ROCMFP4 (dual UE4M3 half-block
    // scales) and 101 is Q4_0_ROCMFP4_FAST (one scale per 32 values).
    // Both pack an E2M1-derived codebook whose top level is 10, not 12.
    ROCMFP4 = 100,
    ROCMFP4_FAST = 101,
    // Prism-ML llama.cpp fork formats (PrismML-Eng/llama.cpp, branch prism),
    // the Ternary-Bonsai-2 group-128 ternary family. The ids and block layouts
    // come from that fork's ggml.h / ggml-common.h:
    //   GGML_TYPE_PQ2_0  = 142: {f16 d; u8 qs[32]} = 34 B per 128 values,
    //     same 2-bit codec as Q2_0 group-64 (byte-major, shifts 0/2/4/6,
    //     value = (q - 1) * d). Confirmed against the fork's
    //     ggml_vec_dot_pq2_0_q8_0_generic and Prism's published
    //     pq2_dequant reference script.
    //   GGML_TYPE_PTQ1_0 = 143: {u8 qs[24]; u8 qh[2]; f16 d} = 28 B per 128
    //     values, TQ1_0's base-3 trit packing at group 128 (256 -> 128).
    // Both files also carry prism.hadamard.* metadata; that transform is the
    // loader/engine's concern, not this enum's.
    PQ2_0 = 142,
    PTQ1_0 = 143,
    // ROCmFPX fork ternary formats (github.com/charlie12345/ROCmFPX).
    // All three are 32-value blocks of packed fp codes behind TWO UE4M3
    // half-block scales (e[2]); the difference is only the code width and
    // the packing. Authority: block_rocmfp{2,3,6} in that fork's
    // ggml/rocmfpx/rocmfpx.h, dequantize_row_fp{2,3,6} in rocmfpx.c, and
    // the fork's ggml_ue4m3_to_fp32 / rocmfpx_scale_is_valid.
    //
    //   Q6_0_ROCMFPX (id 102): {u8 qs[24]; u8 e[2]} = 26 B / 32 values,
    //     6-bit signed codes, mag = code & 31, sign in bit 5, mag 0 negates
    //     to -32. Codes packed 4 to a 3-byte group (pack4).
    //   Q3_0_ROCMFPX (id 104): {u8 qs[12]; u8 e[2]} = 14 B / 32 values,
    //     3-bit codes, mag in {0,1,2,4}, sign in bit 4. Codes packed
    //     8 to a 3-byte group (pack8).
    //   Q2_0_ROCMFPX (id 107): {u8 qs[8]; u8 e[2]} = 10 B / 32 values,
    //     2-bit codes from the S40 ladder {-4,-1,+1,+4}, packed 4 to a byte
    //     (LSB first, shifts 0/2/4/6).
    // Scale bytes: UE4M3, e <= 0x7E valid, e == 0 or 0x7F -> 0.
    Q6_0_ROCMFPX = 102,
    Q3_0_ROCMFPX = 104,
    Q2_0_ROCMFPX = 107,
};

const char *dtype_name(DType t);

// True when this engine can dequantize and multiply against the format.
bool dtype_supported(DType t);

// Elements per quantization block (1 for the float formats).
int dtype_block_size(DType t);

// Bytes per quantization block.
int dtype_block_bytes(DType t);

// Bytes needed for one row of n elements (blocks are not padded at row ends
// by any known GGUF producer, so n must be a multiple of the block size).
size_t dtype_row_bytes(DType t, i64 n);

// True when the row length divides evenly into whole blocks.
bool dtype_row_aligned(DType t, i64 n);

// ---------------------------------------------------------------------------
// host reference kernels
// ---------------------------------------------------------------------------

// out[i] = dequant(src)[i] for i in [0, n). n must be a whole number of blocks.
void dequant_row(DType t, const void *src, f32 *out, i64 n);

// out[i] as fp16, for comparison against the device kernels.
void dequant_row_f16(DType t, const void *src, u16 *out, i64 n);

// y = dot(dequant(src), x) over n elements. This is the reference GEMV inner
// product; the device `dot_*` kernels are validated against it.
f32 vec_dot(DType t, const void *src, const f32 *x, i64 n);

// Number of activations quantized at once by the Q8_1 activation format.
constexpr i64 kAttentionQuantBlock = 32;

} // namespace krk

#endif // KRK_QUANT_HPP
