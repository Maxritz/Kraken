// ============================================================================
//  quant_fmt_c.inc — host dequantizers for IQ3_S, IQ1_S and IQ1_M.
//
//  Included INSIDE the anonymous namespace of src/quant.cpp, next to
//  deq_iq2_xxs / deq_iq2_xs, so it must contain plain definitions only: no
//  #include of anything but the shared codebook file, no namespace lines.
//
//  These are byte-for-byte twins of the reference in ggml-quants.c:
//  dequantize_row_iq3_s, dequantize_row_iq1_s, dequantize_row_iq1_m. Block
//  layouts are the reference's own structs from ggml-common.h, whose
//  static_asserts pin the sizes (110 / 50 / 56 bytes for 256 values):
//
//    IQ3_S  d:2  qs:64  qh:8  signs:32  scales:4      = 110 bytes
//    IQ1_S  d:2  qs:32  qh:16 (8 x u16)               =  50 bytes
//    IQ1_M  qs:32  qh:16  scales:8   -- no d field    =  56 bytes
//
//  All three are 8 chunks of 32 values, so the device twins written against
//  this file decode one 32-value chunk at a time.
// ============================================================================
#include "quant_fmt_c_tables.inc"

// ggml IQ1S_DELTA; IQ1M_DELTA is the same 0.125f, so one constant serves both.
const f32 kIq1sDelta = 0.125f;

// ---------------------------------------------------------------------------
// IQ3_S (type 21) — 3.4375 bpw
// ---------------------------------------------------------------------------
// qs holds 8-bit grid indices, qh supplies the 9th index bit (one per 4-wide
// group, so one byte per 32 values), signs holds one sign byte per group of 8
// and kmask_iq2xs turns a byte into eight sign bits. The scale byte per ib32
// pair packs two 4-bit scales, and they are magnitudes rather than exponents:
// db = d * (1 + 2*s), which is why the low nibble is not simply an exponent.
//
// The grid entries are read as UNSIGNED (the magnitudes live in 1..15); the
// sign is applied separately from `signs`.
void deq_iq3_s(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 110, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const u8 *qs = b + 2;
        const u8 *qh = b + 66;
        const u8 *signs = b + 74;
        const u8 *scales = b + 106;
        // ib32 takes the values 0, 2, 4, 6; p is that index halved. Each p
        // iteration writes 64 values: 32 at the db1 scale, then 32 at db2.
        for (int p = 0; p < 4; p++) {
            const f32 db1 = d * (1.0f + 2.0f * static_cast<f32>(scales[p] & 0x0f));
            const f32 db2 = d * (1.0f + 2.0f * static_cast<f32>(scales[p] >> 4));
            const u8 *qs_p = qs + 16 * p;
            const u8 *sg_p = signs + 8 * p;
            const u8 qhb[2] = {qh[2 * p], qh[2 * p + 1]};
            f32 *y_p = y + 64 * p;
            for (int half = 0; half < 2; half++) {
                const f32 db = half ? db2 : db1;
                const u8 *qs_h = qs_p + (half ? 8 : 0);
                const u8 *sg_h = sg_p + (half ? 4 : 0);
                for (int l = 0; l < 4; l++) {
                    // The 9th bit of each index comes from qh, one bit per
                    // group, walking down from bit 8 for the odd group and
                    // bit 7 for the even one.
                    const u32 g1 = kIq3sGrid[qs_h[2 * l] |
                                             ((qhb[half] << (8 - 2 * l)) & 256)];
                    const u32 g2 = kIq3sGrid[qs_h[2 * l + 1] |
                                             ((qhb[half] << (7 - 2 * l)) & 256)];
                    for (int j = 0; j < 4; j++) {
                        const f32 v1 = static_cast<f32>(
                            static_cast<int>((g1 >> (8 * j)) & 0xFFu));
                        const f32 v2 = static_cast<f32>(
                            static_cast<int>((g2 >> (8 * j)) & 0xFFu));
                        const u8 sb = sg_h[l];
                        y_p[8 * l + j] =
                            db * v1 * ((sb & kMaskIq2xs[j]) ? -1.f : 1.f);
                        y_p[8 * l + j + 4] =
                            db * v2 * ((sb & kMaskIq2xs[j + 4]) ? -1.f : 1.f);
                    }
                }
                y_p += 32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IQ1_S (type 19) — 1.5625 bpw
// ---------------------------------------------------------------------------
// qh is EIGHT 16-bit words, one per 32-value chunk: bits 0..11 hold four
// 3-bit high indices (3 bits each, for the four groups of 8), bits 12..14 a
// 3-bit scale, and bit 15 the SIGN OF THE DELTA. The values are the codebook
// entry plus or minus 0.125, so the sign extension below is load-bearing: an
// unsigned read turns -1 into 255.
void deq_iq1_s(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 50, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const u8 *qs = b + 2;
        const u8 *qh = b + 34;
        for (int ib = 0; ib < 8; ib++) {
            const u16 h = static_cast<u16>(qh[2 * ib] | (qh[2 * ib + 1] << 8));
            const f32 dl = d * (2.0f * static_cast<f32>((h >> 12) & 7) + 1.0f);
            const f32 delta = (h & 0x8000) ? -kIq1sDelta : kIq1sDelta;
            for (int l = 0; l < 4; l++) {
                const int idx = qs[4 * ib + l] | ((static_cast<int>((h >> (3 * l)) & 7)) << 8);
                const u64 grid = kIq1sGrid[idx];
                for (int j = 0; j < 8; j++) {
                    const f32 g = static_cast<f32>(
                        static_cast<i8>((grid >> (8 * j)) & 0xFFu));
                    y[32 * ib + 8 * l + j] = dl * (g + delta);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IQ1_M (type 29) — 1.75 bpw
// ---------------------------------------------------------------------------
// The odd one out: there is no f16 delta field. The f16 scale is reassembled
// from the top nibble of each of the four 16-bit words in `scales`, which is
// the reference's `iq1m_scale_t` trick -- one nibble per word, low nibble
// first. Those same four words each carry two 3-bit group scales, one per
// half, 6 bits apart. The delta sign bits live in qh, two per byte.
void deq_iq1_m(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 56, y += 256) {
        const u8 *qs = b;
        const u8 *qh = b + 32;
        const u8 *scales = b + 48;
        const u16 sc[4] = {
            static_cast<u16>(scales[0] | (scales[1] << 8)),
            static_cast<u16>(scales[2] | (scales[3] << 8)),
            static_cast<u16>(scales[4] | (scales[5] << 8)),
            static_cast<u16>(scales[6] | (scales[7] << 8)),
        };
        // The four scale words spell one f16 between them, one nibble each:
        // sc0's top nibble is the low nibble of the result, sc3's the high.
        const u16 d_bits = static_cast<u16>(
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
            (sc[3] & 0xf000));
        const f32 d = fp16_to_fp32(d_bits);

        for (int ib = 0; ib < 8; ib++) {
            const u16 s = sc[ib >> 1];
            const int sh = 6 * (ib & 1);
            const f32 dl1 = d * (2.0f * static_cast<f32>((s >> sh) & 7) + 1.0f);
            const f32 dl2 = d * (2.0f * static_cast<f32>((s >> (sh + 3)) & 7) + 1.0f);
            const u8 *qs4 = qs + 4 * ib;
            const u8 *qh2 = qh + 2 * ib;
            const u64 idx[4] = {
                static_cast<u64>(qs4[0] | ((qh2[0] << 8) & 0x700)),
                static_cast<u64>(qs4[1] | ((qh2[0] << 4) & 0x700)),
                static_cast<u64>(qs4[2] | ((qh2[1] << 8) & 0x700)),
                static_cast<u64>(qs4[3] | ((qh2[1] << 4) & 0x700)),
            };
            const f32 delta[4] = {
                (qh2[0] & 0x08) ? -kIq1sDelta : kIq1sDelta,
                (qh2[0] & 0x80) ? -kIq1sDelta : kIq1sDelta,
                (qh2[1] & 0x08) ? -kIq1sDelta : kIq1sDelta,
                (qh2[1] & 0x80) ? -kIq1sDelta : kIq1sDelta,
            };
            for (int l = 0; l < 4; l++) {
                const u64 grid = kIq1sGrid[idx[l]];
                const f32 sc_l = (l < 2) ? dl1 : dl2;
                for (int j = 0; j < 8; j++) {
                    const f32 g = static_cast<f32>(
                        static_cast<i8>((grid >> (8 * j)) & 0xFFu));
                    y[32 * ib + 8 * l + j] = sc_l * (g + delta[l]);
                }
            }
        }
    }
}
