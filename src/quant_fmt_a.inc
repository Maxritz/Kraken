// ============================================================================
//  quant_fmt_a.inc — Q1_0 (id 41), Q2_0 (id 42) and IQ3_XXS (id 18).
//
//  #included INSIDE the anonymous namespace of src/quant.cpp, next to
//  deq_tq2_0. Plain definitions only: no #include, no namespace, nothing that
//  is only valid at file scope. The grid below is named `kIq3xxsGrid` because
//  every name here is global to that namespace and a bare `grid` would collide
//  with the sibling format files.
//
//  Each mirrors ggml's reference decoder exactly, value for value and in the
//  order the values are written into y:
//    Q1_0     dequantize_row_q1_0
//    Q2_0     dequantize_row_q2_0      (128-value block)
//    Q2_0_64  dequantize_row_q2_0_64   ( 64-value block)
//    IQ3_XXS  dequantize_row_iq3_xxs
//
//  Block geometry is ggml-common.h's own static_assert, not a recomputation:
//    Q1_0    = sizeof(block_q1_0)    = 2 + QK1_0/8 = 2 + 16 = 18 bytes, 128
//    Q2_0    = sizeof(block_q2_0)    = 2 + QK2_0/4 = 2 + 32 = 34 bytes, 128
//    Q2_0_64 = sizeof(block_q2_0_64) = 2 + 64/4    = 2 + 16 = 18 bytes,  64
//    IQ3_XXS = sizeof(block_iq3_xxs) = 2 + 3*QK_K/8 = 2 + 96 = 98 bytes, 256
//  None of these are QK_K blocks, which is why dtype_block_size() has its own
//  cases for them. Q2_0's 128 vs 64 is the llama-dx-fork vs upstream split; see
//  the comment on DType::Q2_0 in include/krk/quant.hpp for why 42 is the 128.
//
//  The sign table (kSignsIq2xs) and the sign mask (kMaskIq2xs) already exist
//  above, shared with IQ2_XXS/IQ2_XS; IQ3_XXS uses the same 7-bit sign code.
// ============================================================================

// IQ3_XXS value grid: 256 packed uint32, byte j (LSB first) is one 3-bit
// magnitude (ggml iq3xxs_grid, 256 entries). Four values per word, which is
// why the reference reads it through a `const uint8_t *` view.
const u32 kIq3xxsGrid[256] = {
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

// Q1_0: 1.125 bpw ternary. One bit per value; the bit selects d or -d. Note
// there is no zero code: the reference writes neg_d for a 0 bit, so the format
// is {-d, +d} with no exact zero, unlike the TQ1_0/TQ2_0 ladder.
void deq_q1_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 18, y += 128) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 neg_d = -d;
        for (int j = 0; j < 128; j++) {
            const int byte_index = j / 8;
            const int bit_offset = j % 8;
            const u8 bit = static_cast<u8>((b[2 + byte_index] >> bit_offset) & 1u);
            y[j] = bit ? d : neg_d;
        }
    }
}

// Q2_0: 2.125 bpw, 128 values per block, one f16 delta for the whole block.
// Two bits per value and the code is offset by one: 00 -> -1, 01 -> 0,
// 10 -> +1, 11 -> +2. The last step is a real +2 and not a clamp, because that
// is what the fork's own dequantize_row_q2_0 computes; in the files on this
// machine it is dead code. Over 254976 blocks of the Ternary-Bonsai export the
// code histogram is 35.1 / 29.8 / 35.1% with code 3 at 0.00% -- so what the
// container actually holds is a symmetric ternary {-d, 0, +d}. The 32-byte
// plane is 4 values per byte, low pair first.
void deq_q2_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 34, y += 128) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        for (int j = 0; j < 128; j++) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const int q = (b[2 + byte_index] >> bit_offset) & 0x03;
            y[j] = static_cast<f32>(q - 1) * d;
        }
    }
}

// Q2_0_64: the same 2-bit codes over a 64-value block, so 18 bytes and
// 2.25 bpw. Upstream llama.cpp spells this layout Q2_0 (id 42); the fork that
// split the two calls it Q2_0_64 and gives it id 48, which is the id here.
void deq_q2_0_64(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 18, y += 64) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        for (int j = 0; j < 64; j++) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const int q = (b[2 + byte_index] >> bit_offset) & 0x03;
            y[j] = static_cast<f32>(q - 1) * d;
        }
    }
}

// IQ3_XXS: 3.0625 bpw. 98 bytes = fp16 d, 64 bytes of grid indices (8 per
// 32-value group) and 32 bytes of "scale and signs" (4 per group). For each
// group the top nibble of the little-endian uint32 is a 4-bit group scale and
// the low 28 bits are four 7-bit sign selectors, 7 bits each. Each selector
// pairs with TWO grid bytes: the first supplies values 0..3 of an 8-value run
// and the second supplies values 4..7, which is why the reference reads the
// grid table through a byte view rather than unpacking 3-bit fields.
void deq_iq3_xxs(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 98, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const u8 *qs = b + 2;
        const u8 *scales_and_signs = qs + 64;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            u32 aux32 = 0;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, 4);
            const f32 db = d * (0.5f + static_cast<f32>(aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                const u8 signs = kSignsIq2xs[(aux32 >> (7 * l)) & 127u];
                const u32 g1 = kIq3xxsGrid[qs[8 * ib32 + 2 * l + 0]];
                const u32 g2 = kIq3xxsGrid[qs[8 * ib32 + 2 * l + 1]];
                for (int j = 0; j < 4; j++) {
                    const f32 v1 = static_cast<f32>(
                        static_cast<int>((g1 >> (8 * j)) & 0xFFu));
                    const f32 v2 = static_cast<f32>(
                        static_cast<int>((g2 >> (8 * j)) & 0xFFu));
                    y[32 * ib32 + 8 * l + j] =
                        db * v1 * ((signs & kMaskIq2xs[j]) ? -1.f : 1.f);
                    y[32 * ib32 + 8 * l + 4 + j] =
                        db * v2 * ((signs & kMaskIq2xs[4 + j]) ? -1.f : 1.f);
                }
            }
        }
    }
}
