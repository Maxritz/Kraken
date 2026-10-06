// ============================================================================
//  dequant_fmt_c.inc — device twins of the IQ3_S / IQ1_S / IQ1_M host
//  dequantizers in src/quant_fmt_c.inc.
//
//  Included INSIDE namespace krk in src/hip/kernels/dequant.hpp, immediately
//  before the `dequant_chunk` chain is defined, so these helpers are visible to
//  it. The coordinator adds the `else if constexpr` branches and the runtime
//  switch entries; nothing here is called by name from the rest of the kernel
//  set.
//
//  Each helper writes EXACTLY the 32 values of one 0-based chunk of a
//  256-value block, matching the value order of the host function (and so of
//  the reference). Value for value, host and device must agree; the host side
//  is the oracle, so a divergence here shows up as "device and CPU reference
//  disagree" with nothing pointing at the table. Both sides take their
//  codebook from the SAME shared transcription
//  (src/quant_fmt_c_tables.inc) precisely so they cannot drift apart.
//
//  The chunk index is the reference's 32-value group:
//    IQ3_S  chunk>>1 is the ib32/2 pair, chunk&1 picks the db1 or db2 half.
//    IQ1_S  chunk IS ib (qh has one 16-bit word per chunk).
//    IQ1_M  chunk IS ib.
//  Where the reference precomputes something once for two chunks, it is
//  re-derived here from the block bytes: a chunk helper has no state between
//  calls, and the re-derivation is a couple of shifts.
// ============================================================================

#define KRK_FMT_TABLE_DEVICE 1
#include "../../quant_fmt_c_tables.inc"
#undef KRK_FMT_TABLE_DEVICE

// Local f16 read rather than a dependency on the file's helper names, so this
// include stays self-contained wherever it is placed.
__device__ __forceinline__ f32 dequant_fmtc_h2f(u16 h) {
    return static_cast<f32>(*reinterpret_cast<const _Float16 *>(&h));
}

// IQ3_S (type 21): d:2 qs:64 qh:8 signs:32 scales:4 = 110 bytes.
// Grid entries are magnitudes in 1..15 and are read UNSIGNED; the sign comes
// from the `signs` byte through kDevMaskIq2xs.
__device__ __forceinline__ void dequant_chunk_iq3_s(const u8 *b, int chunk,
                                                    _Float16 *y) {
    const f32 d = dequant_fmtc_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
    const u8 *qs = b + 2;
    const u8 *qh = b + 66;
    const u8 *signs = b + 74;
    const u8 *scales = b + 106;
    const int p = chunk >> 1;
    const int half = chunk & 1;
    // db = d * (1 + 2*s), s being the low nibble for the first half of the
    // pair and the high nibble for the second.
    const f32 db = d * (1.0f + 2.0f * static_cast<f32>(
                                     (scales[p] >> (half ? 4 : 0)) & 0x0f));
    const u8 *qs_h = qs + 16 * p + (half ? 8 : 0);
    const u8 *sg_h = signs + 8 * p + (half ? 4 : 0);
    const u8 qhb = qh[2 * p + half];
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const u32 g1 = kIq3sGrid[qs_h[2 * l] | ((qhb << (8 - 2 * l)) & 256)];
        const u32 g2 = kIq3sGrid[qs_h[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256)];
        const u8 sb = sg_h[l];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const f32 v1 =
                static_cast<f32>(static_cast<int>((g1 >> (8 * j)) & 0xFFu));
            const f32 v2 =
                static_cast<f32>(static_cast<int>((g2 >> (8 * j)) & 0xFFu));
            y[8 * l + j] = static_cast<_Float16>(
                db * v1 * ((sb & kDevMaskIq2xs[j]) ? -1.f : 1.f));
            y[8 * l + j + 4] = static_cast<_Float16>(
                db * v2 * ((sb & kDevMaskIq2xs[j + 4]) ? -1.f : 1.f));
        }
    }
}

// IQ1_S (type 19): d:2 qs:32 qh:16 = 50 bytes; qh is 8 x u16, one per chunk.
// bits 0..11 = four 3-bit high indices, bits 12..14 = scale, bit 15 = delta
// sign. The codebook entries are SIGNED int8, so they are sign-extended.
__device__ __forceinline__ void dequant_chunk_iq1_s(const u8 *b, int chunk,
                                                    _Float16 *y) {
    const f32 d = dequant_fmtc_h2f(static_cast<u16>(b[0] | (b[1] << 8)));
    const u8 *qs = b + 2;
    const u8 *qh = b + 34;
    const u16 h = static_cast<u16>(qh[2 * chunk] | (qh[2 * chunk + 1] << 8));
    const f32 dl = d * (2.0f * static_cast<f32>((h >> 12) & 7) + 1.0f);
    const f32 delta = (h & 0x8000) ? -0.125f : 0.125f;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const int idx =
            qs[4 * chunk + l] | ((static_cast<int>((h >> (3 * l)) & 7)) << 8);
        const u64 grid = kIq1sGrid[idx];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const f32 g = static_cast<f32>(
                static_cast<i8>((grid >> (8 * j)) & 0xFFu));
            y[8 * l + j] = static_cast<_Float16>(dl * (g + delta));
        }
    }
}

// IQ1_M (type 29): qs:32 qh:16 scales:8 = 56 bytes and NO d field. The f16
// scale is spelled one nibble per 16-bit scale word (low nibble from sc0's top
// nibble, up to sc3's top nibble as the high nibble); those same words carry
// two 3-bit group scales each, 6 bits apart.
__device__ __forceinline__ void dequant_chunk_iq1_m(const u8 *b, int chunk,
                                                    _Float16 *y) {
    const u8 *qs = b;
    const u8 *qh = b + 32;
    const u8 *scales = b + 48;
    const u16 sc0 = static_cast<u16>(scales[0] | (scales[1] << 8));
    const u16 sc1 = static_cast<u16>(scales[2] | (scales[3] << 8));
    const u16 sc2 = static_cast<u16>(scales[4] | (scales[5] << 8));
    const u16 sc3 = static_cast<u16>(scales[6] | (scales[7] << 8));
    const u16 d_bits = static_cast<u16>((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
                                        ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
    const f32 d = dequant_fmtc_h2f(d_bits);

    const int ib = chunk;
    // One scale word covers an ib32 pair, so this chunk reads word ib >> 1;
    // its own 3-bit group scale is then the low field or the one 3 bits up.
    const u16 scw[4] = {sc0, sc1, sc2, sc3};
    const u16 sc_ib = scw[ib >> 1];
    const int sh = 6 * (ib & 1);
    const f32 dl1 = d * (2.0f * static_cast<f32>((sc_ib >> sh) & 7) + 1.0f);
    const f32 dl2 = d * (2.0f * static_cast<f32>((sc_ib >> (sh + 3)) & 7) + 1.0f);

    const u8 *qs4 = qs + 4 * ib;
    const u8 *qh2 = qh + 2 * ib;
    const u64 idx[4] = {
        static_cast<u64>(qs4[0] | ((qh2[0] << 8) & 0x700)),
        static_cast<u64>(qs4[1] | ((qh2[0] << 4) & 0x700)),
        static_cast<u64>(qs4[2] | ((qh2[1] << 8) & 0x700)),
        static_cast<u64>(qs4[3] | ((qh2[1] << 4) & 0x700)),
    };
    const f32 delta[4] = {
        (qh2[0] & 0x08) ? -0.125f : 0.125f,
        (qh2[0] & 0x80) ? -0.125f : 0.125f,
        (qh2[1] & 0x08) ? -0.125f : 0.125f,
        (qh2[1] & 0x80) ? -0.125f : 0.125f,
    };
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const u64 grid = kIq1sGrid[idx[l]];
        const f32 sc_l = (l < 2) ? dl1 : dl2;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const f32 g = static_cast<f32>(
                static_cast<i8>((grid >> (8 * j)) & 0xFFu));
            y[8 * l + j] = static_cast<_Float16>(sc_l * (g + delta[l]));
        }
    }
}
