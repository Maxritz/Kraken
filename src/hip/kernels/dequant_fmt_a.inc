// ============================================================================
//  dequant_fmt_a.inc — device twins of deq_q1_0 / deq_q2_0 / deq_iq3_xxs.
//
//  #included INSIDE namespace krk in src/hip/kernels/dequant.hpp, immediately
//  BEFORE dequant_chunk is defined, so the sign table and the sign mask it
//  shares with IQ2_XXS (kDevSignsIq2xs, kDevMaskIq2xs) are already in scope.
//  No #include, no namespace, plain definitions only.
//
//  Chunk c is the c-th group of 32 values of the block:
//    Q1_0    4 chunks (128 values), 18-byte block
//    Q2_0    4 chunks (128 values), 34-byte block
//    Q2_0_64 2 chunks ( 64 values), 18-byte block
//    IQ3_XXS 8 chunks (256 values), 98-byte block
//  Every value must equal what the host function of the same name produces;
//  the only difference is the final rounding through _Float16.
// ============================================================================

// IQ3_XXS value grid, device copy of kIq3xxsGrid (ggml iq3xxs_grid, 256
// entries). Byte j of a word is one 3-bit magnitude.
__device__ static const u32 kDevIq3xxsGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// Q1_0, chunk c = values 32c..32c+31. Value j reads bit (j % 8) of byte
// (j / 8) of the 16-byte bit plane at b+2; the bit selects d over -d.
__device__ __forceinline__ void dequant_chunk_q1_0(const u8 *b, int chunk, _Float16 *y) {
    const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
    const f32 neg_d = -d;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const int j = 32 * chunk + i;
        const u8 bit = static_cast<u8>((b[2 + (j >> 3)] >> (j & 7)) & 1u);
        y[i] = static_cast<_Float16>(bit ? d : neg_d);
    }
}

// Q2_0, chunk c = values 32c..32c+31. Two bits per value, code - 1. Four
// chunks make one 128-value block, so chunk 3 reaches byte 33 of 34.
__device__ __forceinline__ void dequant_chunk_q2_0(const u8 *b, int chunk, _Float16 *y) {
    const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const int j = 32 * chunk + i;
        const int q = (b[2 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
        y[i] = static_cast<_Float16>(static_cast<f32>(q - 1) * d);
    }
}

// Q2_0_64, chunk c = values 32c..32c+31 of a 64-value block. The codes and the
// byte order are identical to Q2_0; only the block width differs, so chunk 1
// reaches byte 17 of 18.
__device__ __forceinline__ void dequant_chunk_q2_0_64(const u8 *b, int chunk, _Float16 *y) {
    const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const int j = 32 * chunk + i;
        const int q = (b[2 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
        y[i] = static_cast<_Float16>(static_cast<f32>(q - 1) * d);
    }
}

// IQ3_XXS, chunk c = the c-th 32-value group. Re-derives its own group scale
// and sign word from the block rather than sharing state across calls: the
// four bytes at b+2+64+4c hold one 4-bit scale (top nibble) and four 7-bit
// sign selectors, and re-reading them costs a single 32-bit load.
__device__ __forceinline__ void dequant_chunk_iq3_xxs(const u8 *b, int chunk,
                                                      _Float16 *y) {
    const f32 d = d_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
    const u8 *qs = b + 2 + 8 * chunk;
    const u32 aux32 = d_u32(b + 2 + 64 + 4 * chunk);
    const f32 db = d * (0.5f + static_cast<f32>(aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const u8 signs = kDevSignsIq2xs[(aux32 >> (7 * l)) & 127u];
        const u32 g1 = kDevIq3xxsGrid[qs[2 * l + 0]];
        const u32 g2 = kDevIq3xxsGrid[qs[2 * l + 1]];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const f32 v1 = static_cast<f32>(
                static_cast<int>((g1 >> (8 * j)) & 0xFFu));
            const f32 v2 = static_cast<f32>(
                static_cast<int>((g2 >> (8 * j)) & 0xFFu));
            y[8 * l + j] = static_cast<_Float16>(
                db * v1 * ((signs & kDevMaskIq2xs[j]) ? -1.f : 1.f));
            y[8 * l + 4 + j] = static_cast<_Float16>(
                db * v2 * ((signs & kDevMaskIq2xs[4 + j]) ? -1.f : 1.f));
        }
    }
}
