// quant.cpp — GGML block-format dequantization (host reference oracle).
//
// Every routine here mirrors ggml's dequantize_row_* byte for byte. The device
// kernels in src/hip/kernels/dequant.hpp are validated against these by
// tests/test_quant.cpp, so keep the two in lockstep when editing.
#include "krk/quant.hpp"

#include <cmath>

namespace krk {

namespace {

// NVFP4 / MXFP4 E2M1 grid, doubled (ggml kvalues_fp4).
const i8 kValuesFp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

// IQ4_NL non-linear 4-bit grid (ggml kvalues_iq4nl).
const i8 kIq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// IQ2/XS sign masks and sign table (ggml kmask_iq2xs / ksigns_iq2xs).
const u8 kMaskIq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};
const u8 kSignsIq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// IQ2_XXS value grid: 256 packed uint64, byte j (LSB first) is
// one of {8, 25, 43} (ggml iq2xxs_grid).
const u64 kIq2xxsGrid[256] = {
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

// IQ2_XS grid (ggml iq2xs_grid, 512 entries): byte j (LSB first) of the
// selected uint64 is one of {8, 25, 43}. Same value set as IQ2_XXS, but a
// 9-bit index, so the codebook is twice as large.
const u64 kIq2xsGrid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

// The ROCmFP4 code ladder: E2M1 magnitudes 0..8 with the top level 10
// instead of 12 (fork rocmfp4_decode). Bit 3 is the sign.
inline int rocmfp4_code(int q) {
    const int m = q & 7;
    const int mag = m <= 4 ? m : 2 * m - 4;
    return (q & 8) ? -mag : mag;
}

// E8M0 exponent byte to the half scale MXFP4 uses (its table is doubled).
// Mirrors ggml_e8m0_to_fp32_half exactly, including its two denormal
// exponent bytes (2^-128 and 2^-127) and its choice not to special-case
// the 0xFF NaN pattern.
f32 e8m0_to_fp32_half(u8 x) {
    const u32 bits = (x < 2) ? (0x00200000u << x)
                              : (static_cast<u32>(x - 1) << 23);
    f32 f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void deq_q4_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 18, y += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        for (int k = 0; k < 16; k++) {
            y[k] = d * static_cast<f32>(static_cast<int>(b[2 + k] & 0xF) - 8);
            y[k + 16] = d * static_cast<f32>(static_cast<int>(b[2 + k] >> 4) - 8);
        }
    }
}

void deq_q4_1(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 20, y += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 m = fp16_to_fp32(static_cast<u16>(b[2] | (b[3] << 8)));
        for (int k = 0; k < 16; k++) {
            y[k] = d * static_cast<f32>(b[4 + k] & 0xF) + m;
            y[k + 16] = d * static_cast<f32>(b[4 + k] >> 4) + m;
        }
    }
}

void deq_q5_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 22, y += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        u32 qh;
        std::memcpy(&qh, b + 2, 4);
        for (int k = 0; k < 16; k++) {
            const int lo = (b[6 + k] & 0xF) | (((qh >> k) & 1u) << 4);
            const int hi = (b[6 + k] >> 4) | (((qh >> (k + 16)) & 1u) << 4);
            y[k] = d * static_cast<f32>(lo - 16);
            y[k + 16] = d * static_cast<f32>(hi - 16);
        }
    }
}

void deq_q5_1(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 24, y += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 m = fp16_to_fp32(static_cast<u16>(b[2] | (b[3] << 8)));
        u32 qh;
        std::memcpy(&qh, b + 4, 4);
        for (int k = 0; k < 16; k++) {
            const int lo = (b[8 + k] & 0xF) | (((qh >> k) & 1u) << 4);
            const int hi = (b[8 + k] >> 4) | (((qh >> (k + 16)) & 1u) << 4);
            y[k] = d * static_cast<f32>(lo) + m;
            y[k + 16] = d * static_cast<f32>(hi) + m;
        }
    }
}

void deq_q8_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 34, y += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        for (int k = 0; k < 32; k++)
            y[k] = d * static_cast<f32>(static_cast<i8>(b[2 + k]));
    }
}

void deq_q8_1(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 36, y += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 s = fp16_to_fp32(static_cast<u16>(b[2] | (b[3] << 8)));
        for (int k = 0; k < 32; k++)
            y[k] = d * static_cast<f32>(static_cast<i8>(b[4 + k])) + s;
    }
}

void deq_q2_k(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 84, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[80] | (b[81] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(b[82] | (b[83] << 8)));
        // ggml "four planes": two 128-value groups; within a group each of the
        // four 2-bit planes (shift 0/2/4/6) scales two 16-value halves of qs.
        int is = 0;
        f32 *yy = y;
        for (int g = 0; g < 2; g++) {
            const u8 *q = b + 16 + g * 32;
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                u8 sc = b[is++];
                f32 dl = d * static_cast<f32>(sc & 0xF);
                f32 ml = dmin * static_cast<f32>(sc >> 4);
                for (int l = 0; l < 16; l++)
                    yy[l] = dl * static_cast<f32>((q[l] >> shift) & 3) - ml;
                sc = b[is++];
                dl = d * static_cast<f32>(sc & 0xF);
                ml = dmin * static_cast<f32>(sc >> 4);
                for (int l = 0; l < 16; l++)
                    yy[l + 16] = dl * static_cast<f32>((q[l + 16] >> shift) & 3) - ml;
                yy += 32;
                shift += 2;
            }
        }
    }
}

void get_scale_min_k4(int j, const u8 *q, u8 *d, u8 *m) {
    if (j < 4) {
        *d = static_cast<u8>(q[j] & 63);
        *m = static_cast<u8>(q[j + 4] & 63);
    } else {
        *d = static_cast<u8>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<u8>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void deq_q4_k(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 144, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(b[2] | (b[3] << 8)));
        const u8 *scales = b + 4;
        const u8 *q = b + 16;
        int is = 0;
        f32 *yy = y;
        for (int j = 0; j < 256; j += 64) {
            u8 si, mi;
            get_scale_min_k4(is + 0, scales, &si, &mi);
            const f32 d1 = d * static_cast<f32>(si), m1 = dmin * static_cast<f32>(mi);
            get_scale_min_k4(is + 1, scales, &si, &mi);
            const f32 d2 = d * static_cast<f32>(si), m2 = dmin * static_cast<f32>(mi);
            for (int l = 0; l < 32; l++) {
                yy[l] = d1 * static_cast<f32>(q[l] & 0xF) - m1;
                yy[l + 32] = d2 * static_cast<f32>(q[l] >> 4) - m2;
            }
            yy += 64;
            q += 32;
            is += 2;
        }
    }
}

void deq_q5_k(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 176, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(b[2] | (b[3] << 8)));
        const u8 *scales = b + 4;
        const u8 *qh = b + 16;
        const u8 *q = b + 48;
        int is = 0;
        u8 u1 = 1, u2 = 2;
        f32 *yy = y;
        for (int j = 0; j < 256; j += 64) {
            u8 si, mi;
            get_scale_min_k4(is + 0, scales, &si, &mi);
            const f32 d1 = d * static_cast<f32>(si), m1 = dmin * static_cast<f32>(mi);
            get_scale_min_k4(is + 1, scales, &si, &mi);
            const f32 d2 = d * static_cast<f32>(si), m2 = dmin * static_cast<f32>(mi);
            for (int l = 0; l < 32; l++) {
                yy[l] = d1 * static_cast<f32>((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                yy[l + 32] = d2 * static_cast<f32>((q[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            }
            yy += 64;
            q += 32;
            is += 2;
            u1 = static_cast<u8>(u1 << 2);
            u2 = static_cast<u8>(u2 << 2);
        }
    }
}

void deq_q6_k(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 210, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[208] | (b[209] << 8)));
        for (int n = 0; n < 256; n += 128) {
            const u8 *ql = b + (n / 2);
            const u8 *qh = b + 128 + (n / 4); // group stride is 32 bytes
            const i8 *sc = reinterpret_cast<const i8 *>(b + 192) + (n / 16);
            f32 *yy = y + n;
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                const int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                const int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                const int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                yy[l] = d * static_cast<f32>(sc[is + 0]) * static_cast<f32>(q1 - 32);
                yy[l + 32] = d * static_cast<f32>(sc[is + 2]) * static_cast<f32>(q2 - 32);
                yy[l + 64] = d * static_cast<f32>(sc[is + 4]) * static_cast<f32>(q3 - 32);
                yy[l + 96] = d * static_cast<f32>(sc[is + 6]) * static_cast<f32>(q4 - 32);
            }
        }
    }
}

void deq_q3_k(const u8 *b, f32 *y, i64 nblk) {
    constexpr u32 kMask1 = 0x03030303u;
    constexpr u32 kMask2 = 0x0f0f0f0fu;
    for (i64 i = 0; i < nblk; i++, b += 110, y += 256) {
        const f32 d_all = fp16_to_fp32(static_cast<u16>(b[108] | (b[109] << 8)));
        const u8 *hm = b;         // hmask[32]
        const u8 *q = b + 32;     // qs[64]
        const u8 *scales_raw = b + 96;

        u32 aux[4];
        std::memset(aux, 0, sizeof(aux));
        std::memcpy(aux, scales_raw, 12);
        const u32 tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kMask2) | (((tmp >> 4) & kMask1) << 4);
        aux[3] = ((aux[1] >> 4) & kMask2) | (((tmp >> 6) & kMask1) << 4);
        aux[0] = (aux[0] & kMask2) | (((tmp >> 0) & kMask1) << 4);
        aux[1] = (aux[1] & kMask2) | (((tmp >> 2) & kMask1) << 4);
        const i8 *scales = reinterpret_cast<const i8 *>(aux);

        int is = 0;
        u8 m = 1;
        f32 *yy = y;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                f32 dl = d_all * static_cast<f32>(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *yy++ = dl * static_cast<f32>(
                                      static_cast<i8>((q[l] >> shift) & 3) -
                                      ((hm[l] & m) ? 0 : 4));
                dl = d_all * static_cast<f32>(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *yy++ = dl * static_cast<f32>(
                                      static_cast<i8>((q[l + 16] >> shift) & 3) -
                                      ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = static_cast<u8>(m << 1);
            }
            q += 32;
        }
    }
}

void deq_q8_k(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 292, y += 256) {
        f32 d;
        std::memcpy(&d, b, 4);
        for (int k = 0; k < 256; k++)
            y[k] = d * static_cast<f32>(static_cast<i8>(b[4 + k]));
    }
}

void deq_iq4_nl(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 18, y += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        for (int k = 0; k < 16; k++) {
            y[k] = d * static_cast<f32>(kIq4nl[b[2 + k] & 0xF]);
            y[k + 16] = d * static_cast<f32>(kIq4nl[b[2 + k] >> 4]);
        }
    }
}

// UE4M3 scale byte to fp32 (ggml_ue4m3_to_fp32): 0x00 and 0x7F map to 0.
f32 ue4m3_to_fp32(u8 x) {
    if (x == 0 || x == 0x7F) return 0.f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 7;
    const f32 raw =
        (exp == 0) ? std::ldexp(static_cast<f32>(man), -9)
                   : std::ldexp(1.f + static_cast<f32>(man) / 8.f, exp - 7);
    return raw * 0.5f;
}

void deq_iq2_xxs(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 66, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        for (int ib32 = 0; ib32 < 8; ib32++) {
            u32 aux32[2];
            std::memcpy(aux32, b + 2 + 8 * ib32, 8);
            const u8 *aux8 = reinterpret_cast<const u8 *>(aux32);
            const f32 db = d * (0.5f + static_cast<f32>(aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const u64 grid = kIq2xxsGrid[aux8[l]];
                const u8 signs = kSignsIq2xs[(aux32[1] >> (7 * l)) & 127u];
                for (int j = 0; j < 8; j++) {
                    const f32 g = static_cast<f32>(
                        static_cast<int>((grid >> (8 * j)) & 0xFFu));
                    y[32 * ib32 + 8 * l + j] =
                        db * g * ((signs & kMaskIq2xs[j]) ? -1.f : 1.f);
                }
            }
        }
    }
}

// IQ2_XS: 74-byte blocks of 256 values (ggml block_iq2_xs).
//   bytes  0..1   fp16 d
//   bytes  2..65  32 little-endian uint16: bits 0..8 grid index, bits 9..15 sign code
//   bytes 66..73  8 scale bytes, one nibble per group of 16 (low = first half)
// Mirrors ggml dequantize_row_iq2_xs exactly.
void deq_iq2_xs(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 74, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const u8 *scales = b + 2 + 64;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            const f32 db0 =
                d * (0.5f + static_cast<f32>(scales[ib32] & 0x0F)) * 0.25f;
            const f32 db1 =
                d * (0.5f + static_cast<f32>(scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const int idx = 4 * ib32 + l;
                const u16 q = static_cast<u16>(b[2 + 2 * idx] |
                                               (b[3 + 2 * idx] << 8));
                const u64 grid = kIq2xsGrid[q & 511u];
                const u8 signs = kSignsIq2xs[(q >> 9) & 127u];
                const f32 db = (l >> 1) ? db1 : db0;
                for (int j = 0; j < 8; j++) {
                    const f32 g = static_cast<f32>(
                        static_cast<int>((grid >> (8 * j)) & 0xFFu));
                    y[32 * ib32 + 8 * l + j] =
                        db * g * ((signs & kMaskIq2xs[j]) ? -1.f : 1.f);
                }
            }
        }
    }
}

void deq_iq4_xs(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 136, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[0] | (b[1] << 8)));
        const u16 scales_h = static_cast<u16>(b[2] | (b[3] << 8));
        const u8 *scales_l = b + 4;
        const u8 *qs = b + 8;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                           (((scales_h >> (2 * ib)) & 3) << 4);
            const f32 dl = d * static_cast<f32>(ls - 32);
            for (int j = 0; j < 16; j++) {
                y[j] = dl * static_cast<f32>(kIq4nl[qs[j] & 0xF]);
                y[j + 16] = dl * static_cast<f32>(kIq4nl[qs[j] >> 4]);
            }
            y += 32;
            qs += 16;
        }
    }
}

void deq_nvfp4(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 36, y += 64) {
        for (int s = 0; s < 4; s++) {
            const f32 d = ue4m3_to_fp32(b[s]);
            const u8 *qs = b + 4 + 8 * s;
            for (int j = 0; j < 8; j++) {
                y[s * 16 + j] = static_cast<f32>(kValuesFp4[qs[j] & 0xF]) * d;
                y[s * 16 + j + 8] = static_cast<f32>(kValuesFp4[qs[j] >> 4]) * d;
            }
        }
    }
}

// MXFP4: one E8M0 exponent byte then 16 bytes of packed E2M1 nibbles.
// Element j is the LOW nibble of qs[j] and j+16 the HIGH nibble, and the
// value table is doubled, so the exponent decodes at half scale.
void deq_mxfp4(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 17, y += 32) {
        const f32 d = e8m0_to_fp32_half(b[0]);
        const u8 *qs = b + 1;
        for (int j = 0; j < 16; j++) {
            y[j] = static_cast<f32>(kValuesFp4[qs[j] & 0xF]) * d;
            y[j + 16] = static_cast<f32>(kValuesFp4[qs[j] >> 4]) * d;
        }
    }
}

// ROCmFP4 (type 100): 16 packed nibbles then two UE4M3 half-block scales.
// Same nibble pairing as MXFP4: low -> j, high -> j+16, with e[0] scaling
// the first 16 values and e[1] the second 16. The code ladder is the
// fork's kvalues_rocmfp4 (0..4, 6, 8, 10 with sign in bit 3).
void deq_rocmfp4(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 18, y += 32) {
        const f32 d0 = ue4m3_to_fp32(b[16]);
        const f32 d1 = ue4m3_to_fp32(b[17]);
        for (int j = 0; j < 16; j++) {
            y[j] = static_cast<f32>(rocmfp4_code(b[j] & 0xF)) * d0;
            y[j + 16] = static_cast<f32>(rocmfp4_code(b[j] >> 4)) * d1;
        }
    }
}

// ROCmFP4_FAST (type 101): the same 16 packed nibbles with a single
// UE4M3 scale for all 32 values. 17 bytes, still 4.25 bpw.
void deq_rocmfp4_fast(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 17, y += 32) {
        const f32 d = ue4m3_to_fp32(b[16]);
        for (int j = 0; j < 16; j++) {
            y[j] = static_cast<f32>(rocmfp4_code(b[j] & 0xF)) * d;
            y[j + 16] = static_cast<f32>(rocmfp4_code(b[j] >> 4)) * d;
        }
    }
}

// ---------------------------------------------------------------------------
// ternary (BitNet) formats
//
// Both store {-1, 0, +1} codes with one fp16 scale per 256-value super-block.
// The difference is only how the codes are packed.
//
//   TQ2_0: 2 bits per element, q in {0,1,2} -> q - 1. 64 bytes + d = 66.
//           Four 2-bit planes: byte j carries elements j, j+32, j+64, j+96.
//
//   TQ1_0: 5 trits per byte, base-3 packed. 3^5 = 243 < 256, so a byte holds
//          five codes (the quantizer scales by 256/243 to use the range).
//          Decoding is q * 3^n >> 8 with a multiply high, NOT a division --
//          ggml's own decoder. 48 bytes of qs cover 240 elements; the last 16
//          come from qh, 4 trits per byte. 48 + 4 + 2 = 54 bytes.
//
// The layouts are planar: the low 32 elements of a group live in one trit
// plane, the next 32 in the next plane, and so on. A "chunk" of 32
// consecutive elements is therefore one (group, plane) pair, never a
// contiguous byte run.
// ---------------------------------------------------------------------------

// ((uint16_t)(q * 3^n) * 3) >> 8 -- ggml's multiply-high trit decode.
//
// The 8-bit truncation is load-bearing, not an accident of typing: ggml writes
// `uint8_t q = x.qs[j+m] * pow3[n];`, so the product wraps to a byte BEFORE the
// *3 >> 8. Keeping the full product (the obvious-looking reading) decodes every
// plane above n=0 to the wrong trit. Confirmed against ggml's own encoder by
// tools/probe_tq.cpp: with the wrap the round trip is exact, without it 1467 of
// 2048 values differ.
inline int tq1_trit_of(int q, int n) {
    static const int pow3[6] = {1, 3, 9, 27, 81, 243};
    const u8 w = static_cast<u8>(q * pow3[n]);
    return static_cast<int>((static_cast<u16>(w) * 3u) >> 8) - 1;
}

// Element `idx` (0..255) of a TQ1_0 block, as a trit in {-1, 0, 1}.
inline int tq1_trit_at(const u8 *b, int idx) {
    if (idx < 160) {
        const int n = idx / 32, m = idx % 32;
        return tq1_trit_of(b[m], n);
    }
    if (idx < 240) {
        const int t = idx - 160;
        return tq1_trit_of(b[32 + (t % 16)], t / 16);
    }
    // ggml emits qh as: for n in 0..3 { for j in 0..3 { qh[j] } }, so the
    // byte index is t % 4 and the trit plane is t / 4 -- not the other way
    // round. (Swapping them costs 66 of 2048 values in the round trip.)
    const int t = idx - 240;
    return tq1_trit_of(b[48 + (t % 4)], t / 4);
}

void deq_tq1_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 54, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[52] | (b[53] << 8)));
        for (int v = 0; v < 256; v++)
            y[v] = d * static_cast<f32>(tq1_trit_at(b, v));
    }
}

void deq_tq2_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 66, y += 256) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[64] | (b[65] << 8)));
        for (int j = 0; j < 64; j += 32)
            for (int l = 0; l < 4; l++)
                for (int m = 0; m < 32; m++)
                    y[j * 4 + l * 32 + m] =
                        d * static_cast<f32>(
                                static_cast<int>((b[j + m] >> (l * 2)) & 3) - 1);
    }
}

// ---------------------------------------------------------------------------
// The IQ family added for the GSQ-RCO and Bonsai exports: IQ2_S (22),
// IQ3_XXS (18), IQ1_S (19), IQ3_S (21), IQ1_M (29) and BitNet's Q1_0 (41) /
// Q2_0 (42). Each .inc mirrors one ggml reference decoder and carries its own
// codebook, so there is exactly one copy of every table to diff against
// ggml-common.h. Included inside this anonymous namespace, after the helpers
// they use and before the dispatcher that calls them.
// ---------------------------------------------------------------------------
#include "quant_fmt_a.inc"
#include "quant_fmt_b.inc"
#include "quant_fmt_c.inc"

// One whole block of `t` decoded into buf (>= block_size floats).
void dequant_block(DType t, const u8 *p, f32 *buf) {
    switch (t) {
        case DType::F16:
            buf[0] = fp16_to_fp32(static_cast<u16>(p[0] | (p[1] << 8)));
            break;
        case DType::BF16:
            buf[0] = bf16_to_fp32(static_cast<u16>(p[0] | (p[1] << 8)));
            break;
        case DType::Q4_0: deq_q4_0(p, buf, 1); break;
        case DType::Q4_1: deq_q4_1(p, buf, 1); break;
        case DType::Q5_0: deq_q5_0(p, buf, 1); break;
        case DType::Q5_1: deq_q5_1(p, buf, 1); break;
        case DType::Q8_0: deq_q8_0(p, buf, 1); break;
        case DType::Q8_1: deq_q8_1(p, buf, 1); break;
        case DType::Q2_K: deq_q2_k(p, buf, 1); break;
        case DType::Q3_K: deq_q3_k(p, buf, 1); break;
        case DType::Q4_K: deq_q4_k(p, buf, 1); break;
        case DType::Q5_K: deq_q5_k(p, buf, 1); break;
        case DType::Q6_K: deq_q6_k(p, buf, 1); break;
        case DType::Q8_K: deq_q8_k(p, buf, 1); break;
        case DType::IQ4_NL: deq_iq4_nl(p, buf, 1); break;
        case DType::IQ2_XXS: deq_iq2_xxs(p, buf, 1); break;
        case DType::IQ2_XS: deq_iq2_xs(p, buf, 1); break;
        case DType::IQ4_XS: deq_iq4_xs(p, buf, 1); break;
        case DType::NVFP4: deq_nvfp4(p, buf, 1); break;
        case DType::MXFP4: deq_mxfp4(p, buf, 1); break;
        case DType::ROCMFP4: deq_rocmfp4(p, buf, 1); break;
        case DType::ROCMFP4_FAST: deq_rocmfp4_fast(p, buf, 1); break;
        case DType::TQ1_0: deq_tq1_0(p, buf, 1); break;
        case DType::TQ2_0: deq_tq2_0(p, buf, 1); break;
        case DType::IQ2_S: deq_iq2_s(p, buf, 1); break;
        case DType::IQ3_XXS: deq_iq3_xxs(p, buf, 1); break;
        case DType::IQ1_S: deq_iq1_s(p, buf, 1); break;
        case DType::IQ3_S: deq_iq3_s(p, buf, 1); break;
        case DType::IQ1_M: deq_iq1_m(p, buf, 1); break;
        case DType::Q1_0: deq_q1_0(p, buf, 1); break;
        case DType::Q2_0: deq_q2_0(p, buf, 1); break;
        case DType::Q2_0_64: deq_q2_0_64(p, buf, 1); break;
        default: break;
    }
}

} // namespace

// ---------------------------------------------------------------------------

const char *dtype_name(DType t) {
    switch (t) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        case DType::Q5_0: return "Q5_0";
        case DType::Q5_1: return "Q5_1";
        case DType::Q8_0: return "Q8_0";
        case DType::Q8_1: return "Q8_1";
        case DType::Q2_K: return "Q2_K";
        case DType::Q3_K: return "Q3_K";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::Q8_K: return "Q8_K";
        case DType::IQ4_NL: return "IQ4_NL";
        case DType::IQ2_XXS: return "IQ2_XXS";
        case DType::IQ2_XS: return "IQ2_XS";
        case DType::IQ3_XXS: return "IQ3_XXS";
        case DType::IQ1_S: return "IQ1_S";
        case DType::IQ3_S: return "IQ3_S";
        case DType::IQ2_S: return "IQ2_S";
        case DType::IQ1_M: return "IQ1_M";
        case DType::IQ4_XS: return "IQ4_XS";
        case DType::Q1_0: return "Q1_0";
        case DType::Q2_0: return "Q2_0";
        case DType::Q2_0_64: return "Q2_0_64";
        case DType::MXFP4: return "MXFP4";
        case DType::NVFP4: return "NVFP4";
        case DType::ROCMFP4: return "Q4_0_ROCMFP4";
        case DType::ROCMFP4_FAST: return "Q4_0_ROCMFP4_FAST";
        case DType::TQ1_0: return "TQ1_0";
        case DType::TQ2_0: return "TQ2_0";
        default: return "UNKNOWN";
    }
}

bool dtype_supported(DType t) {
    switch (t) {
        case DType::F32:
        case DType::F16:
        case DType::BF16:
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q5_0:
        case DType::Q5_1:
        case DType::Q8_0:
        case DType::Q8_1:
        case DType::Q2_K:
        case DType::Q3_K:
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K:
        case DType::Q8_K:
        case DType::IQ4_NL:
        case DType::IQ2_XXS:
        case DType::IQ2_XS:
        case DType::IQ3_XXS:
        case DType::IQ1_S:
        case DType::IQ3_S:
        case DType::IQ2_S:
        case DType::IQ1_M:
        case DType::IQ4_XS:
        case DType::NVFP4:
        case DType::MXFP4:
        case DType::ROCMFP4:
        case DType::ROCMFP4_FAST:
        case DType::TQ1_0:
        case DType::TQ2_0:
        case DType::Q1_0:
        case DType::Q2_0:
        case DType::Q2_0_64:
            return true;
        default:
            return false;
    }
}

int dtype_block_size(DType t) {
    switch (t) {
        case DType::F32:
        case DType::F16:
        case DType::BF16: return 1;
        case DType::Q2_K:
        case DType::Q3_K:
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K:
        case DType::Q8_K:
        case DType::IQ2_XXS:
        case DType::IQ2_XS:
        case DType::IQ3_XXS:
        case DType::IQ1_S:
        case DType::IQ3_S:
        case DType::IQ2_S:
        case DType::IQ1_M:
        case DType::IQ4_XS:
        case DType::TQ1_0:
        case DType::TQ2_0: return 256;
        case DType::Q1_0: return 128;
        case DType::Q2_0: return 128;
        case DType::Q2_0_64: return 64;
        case DType::NVFP4: return 64;
        default: return 32;
    }
}

int dtype_block_bytes(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::Q4_0: return 18;
        case DType::Q4_1: return 20;
        case DType::Q5_0: return 22;
        case DType::Q5_1: return 24;
        case DType::Q8_0: return 34;
        case DType::Q8_1: return 36;
        case DType::Q2_K: return 84;
        case DType::Q3_K: return 110;
        case DType::Q4_K: return 144;
        case DType::Q5_K: return 176;
        case DType::Q6_K: return 210;
        case DType::Q8_K: return 292;
        case DType::IQ4_NL: return 18;
        case DType::IQ2_XXS: return 66;
        // Each of these is the reference's sizeof(block_*), which ggml-common.h
        // pins with its own static_assert; the number here is what maps a chunk
        // index onto the block that contains it, so a wrong one reads the wrong
        // block and produces fluent garbage.
        case DType::IQ2_XS: return 74;   // 2 + QK_K/4 + QK_K/32
        case DType::IQ3_XXS: return 98;  // 2 + 3*QK_K/8
        case DType::IQ1_S: return 50;    // 2 + QK_K/8 + QK_K/16
        case DType::IQ3_S: return 110;   // 2 + 13*(QK_K/32) + QK_K/64
        case DType::IQ2_S: return 82;    // 2 + QK_K/4 + QK_K/16
        case DType::IQ1_M: return 56;    // QK_K/8 + QK_K/16 + QK_K/32
        case DType::Q1_0: return 18;     // 2 + QK1_0/8, QK1_0 = 128
        case DType::Q2_0: return 34;     // 2 + QK2_0/4, QK2_0 = 128 (llama-dx)
        case DType::Q2_0_64: return 18;  // 2 + 64/4, upstream's own Q2_0
        case DType::IQ4_XS: return 136;
        case DType::NVFP4: return 36;
        case DType::MXFP4: return 17;
        case DType::ROCMFP4: return 18;
        case DType::ROCMFP4_FAST: return 17;
        case DType::TQ1_0: return 54;
        case DType::TQ2_0: return 66;
        default: return 0;
    }
}

size_t dtype_row_bytes(DType t, i64 n) {
    const int bs = dtype_block_size(t);
    const i64 nblk = (n + bs - 1) / bs;
    return static_cast<size_t>(nblk) * static_cast<size_t>(dtype_block_bytes(t));
}

bool dtype_row_aligned(DType t, i64 n) {
    const int bs = dtype_block_size(t);
    return bs <= 1 || (n % bs) == 0;
}

void dequant_row(DType t, const void *src, f32 *out, i64 n) {
    const u8 *p = static_cast<const u8 *>(src);
    if (t == DType::F32) {
        std::memcpy(out, p, static_cast<size_t>(n) * 4);
        return;
    }
    if (t == DType::F16) {
        const u16 *h = static_cast<const u16 *>(src);
        for (i64 i = 0; i < n; i++) out[i] = fp16_to_fp32(h[i]);
        return;
    }
    if (t == DType::BF16) {
        const u16 *h = static_cast<const u16 *>(src);
        for (i64 i = 0; i < n; i++) out[i] = bf16_to_fp32(h[i]);
        return;
    }
    const int bs = dtype_block_size(t);
    const int bb = dtype_block_bytes(t);
    f32 buf[256];
    i64 i = 0;
    while (i < n) {
        const int cnt = static_cast<int>((n - i) < bs ? (n - i) : bs);
        dequant_block(t, p, buf);
        std::memcpy(out + i, buf, static_cast<size_t>(cnt) * sizeof(f32));
        i += cnt;
        p += bb;
    }
}

void dequant_row_f16(DType t, const void *src, u16 *out, i64 n) {
    // Chunked so the scratch stays small even for long rows.
    constexpr i64 kChunk = 1024;
    f32 buf[kChunk];
    const u8 *p = static_cast<const u8 *>(src);
    i64 done = 0;
    while (done < n) {
        const i64 take = (n - done) < kChunk ? (n - done) : kChunk;
        dequant_row(t, p, buf, take);
        for (i64 i = 0; i < take; i++) out[done + i] = fp32_to_fp16(buf[i]);
        const int bs = dtype_block_size(t);
        const int bb = dtype_block_bytes(t);
        p += static_cast<size_t>(take / bs) * static_cast<size_t>(bb);
        done += take;
    }
}

f32 vec_dot(DType t, const void *src, const f32 *x, i64 n) {
    // The unquantized formats have no block structure, so widen them directly
    // rather than routing through the block loop (which would look for a
    // 1-element "block" decoder that does not exist).
    if (t == DType::F32) {
        const f32 *w = static_cast<const f32 *>(src);
        f32 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        i64 i = 0;
        for (; i + 4 <= n; i += 4) {
            acc0 += w[i] * x[i];
            acc1 += w[i + 1] * x[i + 1];
            acc2 += w[i + 2] * x[i + 2];
            acc3 += w[i + 3] * x[i + 3];
        }
        f32 s = (acc0 + acc1) + (acc2 + acc3);
        for (; i < n; i++) s += w[i] * x[i];
        return s;
    }
    if (t == DType::F16 || t == DType::BF16) {
        const u16 *h = static_cast<const u16 *>(src);
        const bool bf = (t == DType::BF16);
        f32 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        i64 i = 0;
        for (; i + 4 <= n; i += 4) {
            acc0 += (bf ? bf16_to_fp32(h[i]) : fp16_to_fp32(h[i])) * x[i];
            acc1 += (bf ? bf16_to_fp32(h[i + 1]) : fp16_to_fp32(h[i + 1])) * x[i + 1];
            acc2 += (bf ? bf16_to_fp32(h[i + 2]) : fp16_to_fp32(h[i + 2])) * x[i + 2];
            acc3 += (bf ? bf16_to_fp32(h[i + 3]) : fp16_to_fp32(h[i + 3])) * x[i + 3];
        }
        f32 s = (acc0 + acc1) + (acc2 + acc3);
        for (; i < n; i++)
            s += (bf ? bf16_to_fp32(h[i]) : fp16_to_fp32(h[i])) * x[i];
        return s;
    }
    const int bs = dtype_block_size(t);
    f32 buf[256];
    f32 sum = 0;
    const u8 *p = static_cast<const u8 *>(src);
    i64 i = 0;
    while (i < n) {
        const int cnt = static_cast<int>((n - i) < bs ? (n - i) : bs);
        dequant_block(t, p, buf);
        for (int k = 0; k < cnt; k++) sum += buf[k] * x[i + k];
        i += cnt;
        p += dtype_block_bytes(t);
    }
    return sum;
}

} // namespace krk
