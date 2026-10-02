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
    IQ4_NL = 20,
    IQ4_XS = 23,
    BF16 = 30,
    NVFP4 = 40,
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
