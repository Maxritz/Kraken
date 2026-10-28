// quant.cpp — GGML block-format dequantization (host reference oracle).
//
// Every routine here mirrors ggml's dequantize_row_* byte for byte. The device
// kernels in src/hip/kernels/dequant.hpp are validated against these by
// tests/test_quant.cpp, so keep the two in lockstep when editing.
#include "krk/quant.hpp"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <thread>

#if defined(__F16C__) && defined(__FMA__) && \
    (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define KRK_DOT_F16C 1
#endif

// The fused block kernels need AVX2 for the 8-wide widen, but must also build
// on a target that lacks it: -march=native is an option (KRK_NATIVE_CPU), so
// __AVX2__ is not something this file may assume. Target attributes put the
// code in the binary; __builtin_cpu_supports decides whether it is entered.
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define KRK_DOT_AVX2 1
// VNNI is a SEPARATE guard: these are GCC/clang target attributes, so the code
// is in every binary, but a compiler that predates avx512vnni would reject the
// attribute outright. Guarding it keeps old toolchains building the AVX2 path.
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 11)
#define KRK_DOT_AVX512 1
#elif defined(__clang__)
#define KRK_DOT_AVX512 1
#endif
#endif

namespace krk {

namespace {

// Keep-alive sinks for the roofline probes below.
std::atomic<int> g_probe_sink{0};
std::atomic<float> g_probe_sink_f{0.0f};
std::atomic<int> g_probe_sink_i{0};


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
// ROCmFPX fork ternary formats (github.com/charlie12345/ROCmFPX).
//
// All three are 32-value blocks with TWO UE4M3 half-block scales (e[2]),
// one per 16-element half. The difference is only the code width and packing.
// Authority: rocmfpx_dequantize_row_fp{2,3,6} in that fork's rocmfpx.c.
//
// UE4M3: exp = e >> 3, mant = e & 7; if e <= 0x7E:
//   exp == 0 -> mant * 2^-10 ; else (8 + mant) * 2^(exp - 11).
// Already shared via ue4m3_to_fp32().
//
// FP2 (S40): mag in {-4,-1,+1,+4} (frozen MORD order).
//   code & 3 selects the magnitude, e[half] scales it.
//   qs[half*4 + j/4] >> (2*(j%4)) & 3  ->  code
//
// FP3: mag in {0,1,2,4}, sign in bit 4.
//   8 codes packed into 3 bytes (pack8, LSB-first).
//   Half 0 -> qs[0..5] (2 groups), Half 1 -> qs[6..11] (2 groups).
//
// FP6: mag = code & 31, sign in bit 5; mag==0 -> -32 (not 0).
//   4 codes packed into 3 bytes (pack6, LSB-first).
//   Half 0 -> qs[0..11] (4 groups), Half 1 -> qs[12..23] (4 groups).
// ---------------------------------------------------------------------------

// FP2 code -> signed magnitude (S40, frozen MORD order {-4,-1,+1,+4}).
inline int rocmfpx_code_fp2(int c) {
    static const int m[] = {-4, -1, 1, 4};
    return m[c & 3];
}

// FP3 code -> signed magnitude (mag in {0,1,2,4}, sign in bit 4).
inline int rocmfpx_code_fp3(int c) {
    static const int m[] = {0, 1, 2, 4};
    const int v = m[c & 3];
    return (c & 4) ? -v : v;
}

// FP6 code -> signed magnitude (mag = code & 31, sign in bit 5,
// mag 0 -> -32).
inline int rocmfpx_code_fp6(int c) {
    const int mag = c & 31;
    return (c & 32) ? -(mag == 0 ? 32 : mag) : mag;
}

void deq_q2_0_rocmfpx(const u8 *b, f32 *y, i64 nblk) {
    // block_rocmfp2: {u8 qs[8]; u8 e[2]} = 10 bytes / 32 values.
    for (i64 i = 0; i < nblk; i++, b += 10, y += 32) {
        const f32 d0 = ue4m3_to_fp32(b[8]);
        const f32 d1 = ue4m3_to_fp32(b[9]);
        for (int j = 0; j < 16; j++) {
            const int c = (b[j / 4] >> (2 * (j % 4))) & 3;
            y[j] = static_cast<f32>(rocmfpx_code_fp2(c)) * d0;
        }
        for (int j = 0; j < 16; j++) {
            const int c = (b[4 + j / 4] >> (2 * (j % 4))) & 3;
            y[16 + j] = static_cast<f32>(rocmfpx_code_fp2(c)) * d1;
        }
    }
}

void deq_q3_0_rocmfpx(const u8 *b, f32 *y, i64 nblk) {
    // block_rocmfp3: {u8 qs[12]; u8 e[2]} = 14 bytes / 32 values.
    // 8 codes packed into 3 bytes (pack8).
    for (i64 i = 0; i < nblk; i++, b += 14, y += 32) {
        const f32 d0 = ue4m3_to_fp32(b[12]);
        const f32 d1 = ue4m3_to_fp32(b[13]);
        // Unpack all 32 codes (4 groups of 8).
        // Group 0: qs[0..2], Group 1: qs[3..5], Group 2: qs[6..8], Group 3: qs[9..11].
        for (int g = 0; g < 4; g++) {
            const u8 *src = b + g * 3;
            const u8 c0 = src[0] & 7;
            const u8 c1 = (src[0] >> 3) & 7;
            const u8 c2 = ((src[0] >> 6) & 3) | ((src[1] & 1) << 2);
            const u8 c3 = (src[1] >> 1) & 7;
            const u8 c4 = (src[1] >> 4) & 7;
            const u8 c5 = ((src[1] >> 7) & 1) | ((src[2] & 3) << 1);
            const u8 c6 = (src[2] >> 2) & 7;
            const u8 c7 = (src[2] >> 5) & 7;
            const f32 d = (g < 2) ? d0 : d1;
            const int base = g * 8;
            y[base + 0] = static_cast<f32>(rocmfpx_code_fp3(c0)) * d;
            y[base + 1] = static_cast<f32>(rocmfpx_code_fp3(c1)) * d;
            y[base + 2] = static_cast<f32>(rocmfpx_code_fp3(c2)) * d;
            y[base + 3] = static_cast<f32>(rocmfpx_code_fp3(c3)) * d;
            y[base + 4] = static_cast<f32>(rocmfpx_code_fp3(c4)) * d;
            y[base + 5] = static_cast<f32>(rocmfpx_code_fp3(c5)) * d;
            y[base + 6] = static_cast<f32>(rocmfpx_code_fp3(c6)) * d;
            y[base + 7] = static_cast<f32>(rocmfpx_code_fp3(c7)) * d;
        }
    }
}

void deq_q6_0_rocmfpx(const u8 *b, f32 *y, i64 nblk) {
    // block_rocmfp6: {u8 qs[24]; u8 e[2]} = 26 bytes / 32 values.
    // 4 codes packed into 3 bytes (pack4).
    for (i64 i = 0; i < nblk; i++, b += 26, y += 32) {
        const f32 d0 = ue4m3_to_fp32(b[24]);
        const f32 d1 = ue4m3_to_fp32(b[25]);
        // Unpack all 32 codes (8 groups of 4).
        for (int g = 0; g < 8; g++) {
            const u8 *src = b + g * 3;
            const u8 c0 = src[0] & 0x3F;
            const u8 c1 = ((src[0] >> 6) & 3) | ((src[1] & 0xF) << 2);
            const u8 c2 = ((src[1] >> 4) & 0xF) | ((src[2] & 3) << 4);
            const u8 c3 = (src[2] >> 2) & 0x3F;
            const f32 d = (g < 4) ? d0 : d1;
            const int base = g * 4;
            y[base + 0] = static_cast<f32>(rocmfpx_code_fp6(c0)) * d;
            y[base + 1] = static_cast<f32>(rocmfpx_code_fp6(c1)) * d;
            y[base + 2] = static_cast<f32>(rocmfpx_code_fp6(c2)) * d;
            y[base + 3] = static_cast<f32>(rocmfpx_code_fp6(c3)) * d;
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

// PTQ1_0 (Prism fork type 143): TQ1_0's base-3 trit packing at group 128, so
// the block is {u8 qs[24]; u8 qh[2]; f16 d} = 28 bytes / 128 values (1.75 bpw).
// Authority: block_ptq1_0 in PrismML-Eng/llama.cpp ggml-common.h, and that
// fork's ggml_vec_dot_ptq1_0_q8_0_generic, which decodes "using the same
// traversal as dequantize_row_ptq1_0": stages {32, 16, 8} over qs with five
// trit planes per stage-window, then four planes over qh. At 24 bytes of qs
// the stage-32 window never fits, stage 16 takes qs[0..15] once (16+16 > 24)
// and stage 8 takes qs[16..23] once, giving:
//   idx [0, 80):    qs[idx % 16], plane idx / 16
//   idx [80, 120):  qs[16 + t % 8], plane t / 8   (t = idx - 80)
//   idx [120, 128): qh[t % 2], plane t / 2        (t = idx - 120)
// 80 + 40 + 8 = 128. The trit decode itself is tq1_trit_of, unchanged.
// Element `idx` (0..127) of a PTQ1_0 block, as a trit in {-1, 0, 1}. The
// traversal is the fork's dequantize_row_ptq1_0 order; see deq_ptq1_0 below.
inline int ptq1_trit_at(const u8 *b, int idx) {
    if (idx < 80)
        return tq1_trit_of(b[idx % 16], idx / 16);
    if (idx < 120) {
        const int t = idx - 80;
        return tq1_trit_of(b[16 + (t % 8)], t / 8);
    }
    const int t = idx - 120;
    return tq1_trit_of(b[24 + (t % 2)], t / 2);
}

void deq_ptq1_0(const u8 *b, f32 *y, i64 nblk) {
    for (i64 i = 0; i < nblk; i++, b += 28, y += 128) {
        const f32 d = fp16_to_fp32(static_cast<u16>(b[26] | (b[27] << 8)));
        for (int v = 0; v < 128; v++)
            y[v] = d * static_cast<f32>(ptq1_trit_at(b, v));
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

// ---------------------------------------------------------------------------
// TERNARY int8 dots: TQ1_0 (BitNet, 256 values / 54 B) and PTQ1_0
// (Ternary-Bonsai, 128 values / 28 B).
//
// Both formats store {-1, 0, +1} behind one f16 scale per block, so the dot
// against an int8 activation collapses to
//
//     dot = d_w * SUM_c dx_c * SUM_{i in chunk c} t_i * xq_i
//
// -- no bias term, no per-sub-block weight table, and therefore no `sum(x)`
// sidecar of the kind Q4_K's dmin term needs. The whole weight-side cost per
// value is one nibble of the packed trit stream expanded to {-1,0,1}; the
// activation is never touched beyond the int8 it is already stored as.
//
// That is why these matter: at 1.69 and 1.75 bits per weight the stored form is
// 9x to 18x smaller than the f32 expansion the generic path walks, and the
// generic path walks it once per row tile.
//
// `_mm256_sign_epi8(x, t)` IS x*t for t in {-1,0,1} -- zero stays zero, a
// negative negates -- so no multiply is needed at all: maddubs then madd reduce
// 32 trit-weighted products to eight i32 lanes.
//
// The trit expansion calls the SAME tq1_trit_at / ptq1_trit_at the dequantiser
// uses, deliberately. A second "optimised" bit-twiddling copy of the base-3
// unpack would be a second thing to get wrong, and this one is already pinned
// against the reference decoder.
static inline f32 unpack_tq1_trits(const u8 *w, i8 *tr) {
    for (int v = 0; v < 256; v++) tr[v] = static_cast<i8>(tq1_trit_at(w, v));
    return fp16_to_fp32(static_cast<u16>(w[52] | (w[53] << 8)));
}

static inline f32 unpack_ptq1_trits(const u8 *w, i8 *tr) {
    for (int v = 0; v < 128; v++) tr[v] = static_cast<i8>(ptq1_trit_at(w, v));
    return fp16_to_fp32(static_cast<u16>(w[26] | (w[27] << 8)));
}

// SUM over 32 values of t*x, as an i32.
static inline i32 ternary_chunk_sum(const i8 *xv, const i8 *tv,
                                    const __m256i ones8, const __m256i ones16) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xv));
    const __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tv));
    const __m256i sx = _mm256_sign_epi8(x, t);
    __m256i pr = _mm256_maddubs_epi16(ones8, sx);
    pr = _mm256_madd_epi16(pr, ones16);
    alignas(32) i32 l[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(l), pr);
    return l[0] + l[1] + l[2] + l[3] + l[4] + l[5] + l[6] + l[7];
}

// B values per block, BLK bytes per block. The trits are expanded once per
// block and shared by all eight rows -- the same amortisation the K-quant
// kernels get from unpacking q/hmask/scales once.
static void ternary_dot_x8_impl(const u8 *w, const i8 *xq, const f32 *dx,
                                i64 n, i64 rstride, i64 dstride, f32 *out,
                                i64 out_stride, i64 B, i64 BLK,
                                f32 (*unpack)(const u8 *, i8 *)) {
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const i64 chunks = B / 32;
    f32 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    alignas(32) i8 tr[256];
    for (i64 nb = 0; nb < n / B; nb++, w += BLK) {
        const f32 d = unpack(w, tr);
        for (int r = 0; r < 8; r++) {
            const i8 *xr = xq + static_cast<i64>(r) * rstride + nb * B;
            const f32 *dr = dx + static_cast<i64>(r) * dstride + nb * chunks;
            f32 a = 0.0f;
            for (i64 c = 0; c < chunks; c++)
                a = std::fma(dr[c],
                             static_cast<f32>(ternary_chunk_sum(
                                 xr + c * 32, tr + c * 32, ones8, ones16)),
                             a);
            acc[r] = std::fma(d, a, acc[r]);
        }
    }
    for (int r = 0; r < 8; r++) out[static_cast<i64>(r) * out_stride] = acc[r];
}

void tq1_0_q8_x8_internal(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                     i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    ternary_dot_x8_impl(w, xq, dx, n, rstride, dstride, out, out_stride, 256, 54,
                        unpack_tq1_trits);
}

void ptq1_0_q8_x8_internal(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                      i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    ternary_dot_x8_impl(w, xq, dx, n, rstride, dstride, out, out_stride, 128, 28,
                        unpack_ptq1_trits);
}

// Single-row form: the trailing (rows % 8) rows, and decode.
static f32 ternary_dot_1(const u8 *w, const i8 *xq, const f32 *dx, i64 n, i64 B,
                         i64 BLK, f32 (*unpack)(const u8 *, i8 *)) {
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const i64 chunks = B / 32;
    f32 acc = 0.0f;
    alignas(32) i8 tr[256];
    for (i64 nb = 0; nb < n / B; nb++, w += BLK) {
        const f32 d = unpack(w, tr);
        f32 a = 0.0f;
        for (i64 c = 0; c < chunks; c++)
            a = std::fma(dx[nb * chunks + c],
                         static_cast<f32>(ternary_chunk_sum(
                             xq + nb * B + c * 32, tr + c * 32, ones8, ones16)),
                         a);
        acc = std::fma(d, a, acc);
    }
    return acc;
}

f32 tq1_0_q8_internal(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    return ternary_dot_1(w, xq, dx, n, 256, 54, unpack_tq1_trits);
}

f32 ptq1_0_q8_internal(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    return ternary_dot_1(w, xq, dx, n, 128, 28, unpack_ptq1_trits);
}

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
        case DType::PQ2_0: deq_pq2_0(p, buf, 1); break;
        case DType::PTQ1_0: deq_ptq1_0(p, buf, 1); break;
        case DType::Q6_0_ROCMFPX: deq_q6_0_rocmfpx(p, buf, 1); break;
        case DType::Q3_0_ROCMFPX: deq_q3_0_rocmfpx(p, buf, 1); break;
        case DType::Q2_0_ROCMFPX: deq_q2_0_rocmfpx(p, buf, 1); break;
        default: break;
    }
}

} // namespace

// ---- ternary dot kernels, public -------------------------------------------
// The implementations live inside the anonymous namespace above because they
// call the private base-3 trit unpackers -- deliberately the same ones the
// dequantiser runs, so a kernel and the reference cannot drift apart. These
// wrappers are the only exported surface.
void dot_tq1_0_q8_x8(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                     i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    tq1_0_q8_x8_internal(w, xq, dx, n, rstride, dstride, out, out_stride);
}

void dot_ptq1_0_q8_x8(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                      i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    ptq1_0_q8_x8_internal(w, xq, dx, n, rstride, dstride, out, out_stride);
}

f32 dot_tq1_0_q8(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    return tq1_0_q8_internal(w, xq, dx, n);
}

f32 dot_ptq1_0_q8(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    return ptq1_0_q8_internal(w, xq, dx, n);
}

// ---------------------------------------------------------------------------

const char *dtype_name(DType t) {
    switch (t) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I32: return "I32";
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
        case DType::PQ2_0: return "PQ2_0";
        case DType::PTQ1_0: return "PTQ1_0";
        case DType::Q6_0_ROCMFPX: return "Q6_0_ROCMFPX";
        case DType::Q3_0_ROCMFPX: return "Q3_0_ROCMFPX";
        case DType::Q2_0_ROCMFPX: return "Q2_0_ROCMFPX";
        default: return "UNKNOWN";
    }
}

bool dtype_supported(DType t) {
    switch (t) {
        case DType::F32:
        case DType::F16:
        case DType::BF16:
        case DType::I32:
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
        case DType::PQ2_0:
        case DType::PTQ1_0:
        case DType::Q6_0_ROCMFPX:
        case DType::Q3_0_ROCMFPX:
        case DType::Q2_0_ROCMFPX:
            return true;
        default:
            return false;
    }
}

int dtype_block_size(DType t) {
    switch (t) {
        case DType::F32:
        case DType::F16:
        case DType::BF16:
        case DType::I32: return 1;
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
        case DType::PQ2_0: return 128;
        case DType::PTQ1_0: return 128;
        case DType::Q2_0_64: return 64;
        case DType::NVFP4: return 64;
        case DType::Q6_0_ROCMFPX: return 32;
        case DType::Q3_0_ROCMFPX: return 32;
        case DType::Q2_0_ROCMFPX: return 32;
        default: return 32;
    }
}

int dtype_block_bytes(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::I32: return 4;
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
        case DType::PQ2_0: return 34;   // 2 + 128/4, block_pq2_0 (Prism)
        case DType::PTQ1_0: return 28;  // 24 + 2 + 2, block_ptq1_0 (Prism)
        case DType::Q6_0_ROCMFPX: return 26;  // 24 + 2, block_rocmfp6 (ROCmFPX)
        case DType::Q3_0_ROCMFPX: return 14;  // 12 + 2, block_rocmfp3 (ROCmFPX)
        case DType::Q2_0_ROCMFPX: return 10;  //  8 + 2, block_rocmfp2 (ROCmFPX)
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
    // I32 (ggml type 26): plain int32, widened. Not a quantization, so there
    // is no block to walk -- the values ARE the data, and the only question
    // is the sign and the width.
    if (t == DType::I32) {
        const i32 *w = static_cast<const i32 *>(src);
        for (i64 i = 0; i < n; i++) out[i] = static_cast<f32>(w[i]);
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

namespace {
#if defined(KRK_DOT_AVX2)
// ---------------------------------------------------------------------------
// Fused block kernels (KRK_CPU_DOT_AVX2).
//
// What this replaces: dequant_block() widening a 32-value block into a stack
// buffer one element at a time, then a second pass multiplying that buffer by
// the activation. Two round trips through L1 per weight, and on a q8_0 expert
// matrix -- 768x768, the shape a routed expert has in every MoE -- it is what
// the host-side prefill time is made of.
//
// What this does instead: widen eight int8 to eight f32 with two instructions,
// scale by the block's d, and FMA straight into the accumulator. No stack
// buffer, no second pass.
//
// Why this is allowed near an "exact" backend: it performs the SAME arithmetic
// in the SAME order as the four-lane path. Element k is d * (f32)w[k] -- one
// rounding, exactly as the scalar dequant rounds it -- and term k is added to
// accumulator k mod 4 in increasing k, exactly as the four-lane loop adds it.
// Both fold as (a0+a1)+(a2+a3). The result is BIT-IDENTICAL to
// KRK_CPU_DOT_FAST, not merely close, and test_vec_dot_avx2_matches_fast
// asserts that rather than a tolerance. This is not a third rounding tier.
//
// It is still opt-in, because the four-lane path it accelerates is opt-in: the
// flag alone turns that order on.
__attribute__((target("avx2,fma"))) inline void fma8(__m128 &acc, __m256 w,
                                                    const f32 *x) {
    acc = _mm_fmadd_ps(_mm256_castps256_ps128(w), _mm_loadu_ps(x), acc);
    acc = _mm_fmadd_ps(_mm256_extractf128_ps(w, 1), _mm_loadu_ps(x + 4), acc);
}

// q8_0: d (f16) + 32 int8, 34 bytes per block.
__attribute__((target("avx2,fma"))) f32 dot_q8_0_avx2(const u8 *p, const f32 *x,
                                                      i64 nb) {
    __m128 acc = _mm_setzero_ps();
    for (i64 b = 0; b < nb; b++, p += 34, x += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(
            p[0] | (static_cast<u16>(p[1]) << 8)));
        const __m256 dv = _mm256_set1_ps(d);
        for (int k = 0; k < 32; k += 8) {
            const __m128i q8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(p + 2 + k));
            const __m256 w = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)), dv);
            fma8(acc, w, x + k);
        }
    }
    alignas(16) f32 a[4];
    _mm_store_ps(a, acc);
    return (a[0] + a[1]) + (a[2] + a[3]);
}

// q4_0: d (f16) + 16 bytes of nibbles, 18 bytes per block. Low nibbles are
// values 0..15 and high nibbles 16..31, so the four widen-and-FMA passes below
// walk the block in that order and land on the same accumulators the four-lane
// path would have used.
__attribute__((target("avx2,fma"))) f32 dot_q4_0_avx2(const u8 *p, const f32 *x,
                                                      i64 nb) {
    __m128 acc = _mm_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i off = _mm256_set1_epi32(8);
    for (i64 b = 0; b < nb; b++, p += 18, x += 32) {
        const f32 d = fp16_to_fp32(static_cast<u16>(
            p[0] | (static_cast<u16>(p[1]) << 8)));
        const __m256 dv = _mm256_set1_ps(d);
        const __m128i raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(p + 2));
        const __m128i lo = _mm_and_si128(raw, m4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        const __m256 w0 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), off)),
            dv);
        const __m256 w1 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)), off)),
            dv);
        const __m256 w2 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), off)),
            dv);
        const __m256 w3 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)), off)),
            dv);
        fma8(acc, w0, x);
        fma8(acc, w1, x + 8);
        fma8(acc, w2, x + 16);
        fma8(acc, w3, x + 24);
    }
    alignas(16) f32 a[4];
    _mm_store_ps(a, acc);
    return (a[0] + a[1]) + (a[2] + a[3]);
}

// Asked once. A build made without AVX2 in -march still contains the code
// above, so the answer has to come from the CPU, not the preprocessor.
bool cpu_has_avx2() {
    static const bool v = __builtin_cpu_supports("avx2") != 0;
    return v;
}

// AVX-512F + AVX-512BW, WITHOUT assuming VNNI: the 512-bit int8 pair kernels
// need only maddubs/madd, which are BW. Skylake-SP is exactly this machine --
// it has the wide byte ops and none of the VNNI accumulator forms -- so
// checking avx512vnni here would throw away half the register width on the very
// parts most likely to run this code.
bool cpu_has_avx512bw() {
    static const bool v = __builtin_cpu_supports("avx512f") != 0 &&
                          __builtin_cpu_supports("avx512bw") != 0;
    return v;
}

// VNNI is what makes a 4-bit dot cheap: vpdpbusd accumulates 32 int8 products
// into int32 in ONE instruction, where the AVX2 path needs maddubs + madd.
// Measured on the Tiny-Moe GEMM shapes: 3.5 -> 7.4 MAC/instruction.
//
// Zen 3 does NOT have it (it shipped with Zen 4), so this is a runtime branch
// and the AVX2 kernel below stays the fallback -- one binary, as required.
bool cpu_has_vnni() {
    // Detection AND the fallback switch: KRK_CPU_NO_VNNI=1 forces the AVX2
    // kernels on a host that has VNNI, which is what makes them measurable on
    // this box and what a user with a misreporting hypervisor needs.
    //
    // Only avx512vnni counts. The VNNI kernels in this file are compiled with
    // the avx512vnni target attribute, so their vpdpbusd is the EVEX form: a
    // host that has AVX-VNNI (Alder Lake) and NOT avx512vnni would raise SIGILL
    // if it were dispatched here. Checking the feature the encoding actually
    // requires is the difference between a fallback and a crash.
    static const bool v = !(std::getenv("KRK_CPU_NO_VNNI") &&
                            std::getenv("KRK_CPU_NO_VNNI")[0] == '1') &&
                          __builtin_cpu_supports("avx512vnni") != 0;
    return v;
}
#endif // KRK_DOT_AVX2

// ---------------------------------------------------------------------------
// Q4_K against an int8 activation (KRK_CPU_DOT_INT8).
//
// This is the K-quant kernel that was missing. The generic tile widens every
// block to f32 and FMA's four lanes, which is ~12 GFLOPS on this box; the
// format itself supports something far cheaper.
//
// Q4_K stores, per 32-value sub-block,
//     w_k = d * s * q_k - dmin * m          q_k in 0..15, s and m 6-bit
// so for any activation x,
//     sum(w*x) = d*s * sum(q_k*x_k) - dmin*m * sum(x_k)
// and if x is quantised to int8 as x ~ dx*xi the two sums are EXACT integers:
//     sum(w*x) = dx * ( d*s*I - dmin*m*S ),  I = sum(q_k*xi_k), S = sum(xi_k)
//
// So the weights are read in the format they are stored in -- no f32 widening
// at any point -- and the multiply is vpmaddubsw + vpmaddwd instead of four
// f32 FMA lanes. I and S come out of the same two-instruction pair.
//
// Accuracy: this is an int8 activation against a 4-bit weight. The activation
// quantisation is the only loss and it is the same one llama.cpp ships. The
// gate is not a tolerance, it is the text: the greedy trajectory must be
// identical to the exact path, and docs/DECODE-REVIEW.md records the measured
// relative error per format next to it.
__attribute__((target("avx2,fma"))) inline i32 hsum8_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// One 32-value block of the activation row -> int8 plus its scale. Four
// eight-wide steps: the max has to span all 32 before any of them is rounded,
// or the four halves would be scaled by different amounts and the block would
// no longer have a single dx to apply at the end.
__attribute__((target("avx2,fma"))) inline void quantize_q8_32_d(const f32 *x,
                                                                i8 *xq, f32 *dx) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 v[4];
    for (int i = 0; i < 4; i++) v[i] = _mm256_loadu_ps(x + 8 * i);
    __m256 a = _mm256_max_ps(_mm256_andnot_ps(sign, v[0]),
                             _mm256_andnot_ps(sign, v[1]));
    a = _mm256_max_ps(a, _mm256_andnot_ps(sign, v[2]));
    a = _mm256_max_ps(a, _mm256_andnot_ps(sign, v[3]));
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    const f32 amax = _mm_cvtss_f32(m);
    const f32 d = amax > 0.0f ? amax / 127.0f : 0.0f;
    *dx = d;
    if (d == 0.0f) {
        std::memset(xq, 0, 32);
        return;
    }
    const __m256 inv = _mm256_set1_ps(1.0f / d);
    const __m256i lo32 = _mm256_set1_epi32(-127), hi32 = _mm256_set1_epi32(127);
    for (int i = 0; i < 4; i++) {
        __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(v[i], inv));
        q = _mm256_max_epi32(q, lo32);
        q = _mm256_min_epi32(q, hi32);
        const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(q),
                                           _mm256_extracti128_si256(q, 1));
        const __m128i p8 = _mm_packs_epi16(p16, p16);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(xq + 8 * i), p8);
    }
}

// n must be a multiple of 32; xq gets n int8, dx gets n/32 scales.
__attribute__((target("avx2,fma"))) void quantize_q8_row_d(const f32 *x, i64 n,
                                                          i8 *xq, f32 *dx) {
    for (i64 b = 0; b < n; b += 32)
        quantize_q8_32_d(x + b, xq + b, dx + b / 32);
}

// Sum of 32 int8 as int32. This is the `dmin` term's other half and it depends
// ONLY on the activation, so it is computed once per quantise, not once per
// weight row. maddubs pairs (1*xi) twice into int16, then madd widens to int32:
// |xi| <= 127 so a pair is <= 254 and nothing overflows.
__attribute__((target("avx2"))) inline i32 sum_i8_32(const i8 *p) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    const __m256i m = _mm256_maddubs_epi16(_mm256_set1_epi8(1), v);
    const __m256i s = _mm256_madd_epi16(m, _mm256_set1_epi16(1));
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(s),
                               _mm256_extracti128_si256(s, 1));
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

// One weight row of Q4_K against one quantized activation row.
//
// The reduction is the whole design. A first version closed every 32-value
// sub-block with two horizontal sums, and a horizontal sum is a dependent
// chain -- per 32 values it cost about 20 cycles against the eight f32 FMA
// instructions it replaced, which is why it measured 1.27x and not 4x.
//
// Here the eight lanes of a __m256i carry the eight partial sums of a
// sub-block, which are converted to f32, scaled by that sub-block's
// (dx, d*s, dmin*m) and ACCUMULATED into one vector across the whole row. The
// horizontal sum happens once, at the end, instead of twice per sub-block.
__attribute__((target("avx2,fma"))) f32 dot_q4_K_q8_d(const u8 *w, const i8 *xq,
                                                      const f32 *dx, i64 n) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8 = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 144) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (w[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(w[2] | (w[3] << 8)));
        const u8 *sc = w + 4;
        const u8 *qs = w + 16;
        for (int j = 0; j < 256; j += 64) {
            const __m256i qb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qs));
            const __m256i q_lo = _mm256_and_si256(qb, kLo);
            const __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), kLo);
            const i64 base = (nb * 256) + j;
            const __m256i x_lo = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base));
            const __m256i x_hi = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base + 32));
            u8 s1, m1, s2, m2;
            get_scale_min_k4(static_cast<int>(j / 32) + 0, sc, &s1, &m1);
            get_scale_min_k4(static_cast<int>(j / 32) + 1, sc, &s2, &m2);
            const auto accumulate = [&](__m256i qn, __m256i xv, f32 sxk,
                                        f32 smk) -> void {
                const __m256 fI = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(qn, xv), ones16));
                const __m256 fS = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(ones8, xv), ones16));
                acc = _mm256_fmadd_ps(fI, _mm256_set1_ps(sxk), acc);
                acc = _mm256_fnmadd_ps(fS, _mm256_set1_ps(smk), acc);
            };
            const f32 dx1 = dx[base / 32];
            const f32 dx2 = dx[(base + 32) / 32];
            accumulate(q_lo, x_lo, dx1 * d * static_cast<f32>(s1),
                       dx1 * dmin * static_cast<f32>(m1));
            accumulate(q_hi, x_hi, dx2 * d * static_cast<f32>(s2),
                       dx2 * dmin * static_cast<f32>(m2));
            qs += 32;
        }
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// Eight activation rows against one weight row, with the weight block decoded
// ONCE and reused eight times. This is the prefill shape: without it the row is
// re-read per input row and the kernel runs at GEMV speed, which is what the
// 1.53x measured -- the same defect vec_dot_tiled() was written to fix, one
// level down.
__attribute__((target("avx2,fma"))) void dot_q4_K_q8_x8_d(
    const u8 *w, const i8 *xq, const f32 *dx, i64 n, i64 rstride, i64 dstride,
    f32 *out, i64 out_stride) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8 = _mm256_set1_epi8(1);
    __m256 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 144) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (w[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(w[2] | (w[3] << 8)));
        const u8 *sc = w + 4;
        const u8 *qs = w + 16;
        for (int j = 0; j < 256; j += 64) {
            const __m256i qb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qs));
            const __m256i q_lo = _mm256_and_si256(qb, kLo);
            const __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), kLo);
            const i64 base = (nb * 256) + j;
            u8 s1, m1, s2, m2;
            get_scale_min_k4(static_cast<int>(j / 32) + 0, sc, &s1, &m1);
            get_scale_min_k4(static_cast<int>(j / 32) + 1, sc, &s2, &m2);
            for (i64 r = 0; r < 8; r++) {
                const i8 *xr = xq + r * rstride;
                const f32 *dr = dx + r * dstride;
                const __m256i x_lo = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xr + base));
                const __m256i x_hi = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xr + base + 32));
                const auto accumulate = [&](__m256i qn, __m256i xv, f32 sxk,
                                            f32 smk) -> void {
                    const __m256 fI = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                        _mm256_maddubs_epi16(qn, xv), ones16));
                    const __m256 fS = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                        _mm256_maddubs_epi16(ones8, xv), ones16));
                    acc[r] = _mm256_fmadd_ps(fI, _mm256_set1_ps(sxk), acc[r]);
                    acc[r] = _mm256_fnmadd_ps(fS, _mm256_set1_ps(smk), acc[r]);
                };
                const f32 dx1 = dr[base / 32];
                const f32 dx2 = dr[(base + 32) / 32];
                accumulate(q_lo, x_lo, dx1 * d * static_cast<f32>(s1),
                           dx1 * dmin * static_cast<f32>(m1));
                accumulate(q_hi, x_hi, dx2 * d * static_cast<f32>(s2),
                           dx2 * dmin * static_cast<f32>(m2));
            }
            qs += 32;
        }
    }
    for (int r = 0; r < 8; r++) {
        __m128 lo = _mm256_castps256_ps128(acc[r]);
        __m128 hi = _mm256_extractf128_ps(acc[r], 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[static_cast<i64>(r) * out_stride] = _mm_cvtss_f32(lo);
    }
}

// The same eight rows, with the min term taken out of the inner loop.
//
// dot_q4_K_q8_x8_d() above still rebuilds  S_s = sum(x_k)  for every weight
// row: two maddubs + madd + cvt + fnmadd per 32 values per row, which is as
// many instructions as the dot product itself. That is the whole reason the
// prefill kernel measured 3.5 MAC/instruction while the kernel's own ceiling
// with per-32 scales is about 6.4.
//
// With sx[] (computed once per activation row by quantize_q8_row_sx) the term
// is  sum_s (dmin * m_s) * dx_s * S_s  -- pipelined as ONE eight-lane f32
// vector per 256-value block: lane s is sub-block s, the eight m_s come from
// one get_scale_min_k4 sweep of the block, and the eight S_s are one 32-byte
// load. One cvt + one fnmadd per row per block replaces thirty-two
// instructions per row per block. Nothing is dequantised: the block bytes are
// read where they are stored, the scales are decoded into a register, not a
// tensor.
//
// VNNI's vpdpbusd (Q4_K's nibbles are 0..15, so the weight side is already the
// unsigned operand and the int8 activation the signed one) then does a 32-value
// sub-block in a single instruction, against two for maddubs + madd.
#if defined(KRK_DOT_AVX512)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
void dot_q4_K_q8_sx_x8_vnni_d(const u8 *w, const i8 *xq, const f32 *dx,
                              const i32 *sx, i64 n, i64 rstride, i64 dstride,
                              i64 sstride, f32 *out, i64 out_stride) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 144) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (w[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(w[2] | (w[3] << 8)));
        const u8 *sc = w + 4;
        const u8 *qs = w + 16;
        // The block's eight (scale, min) pairs, widened once and shared by all
        // eight rows. get_scale_min_k4() is packed and branchy; doing it 8x8
        // times per block is most of what the scalar tail was spending.
        f32 sf[8], mf[8];
        for (int s = 0; s < 8; s++) {
            u8 a = 0, b = 0;
            get_scale_min_k4(s, sc, &a, &b);
            sf[s] = static_cast<f32>(a) * d;
            mf[s] = static_cast<f32>(b) * dmin;
        }
        const __m256 sfv = _mm256_loadu_ps(sf);
        const __m256 mfv = _mm256_loadu_ps(mf);
        // The block decoded ONCE: eight 32-value unsigned vectors, in stored
        // order. sub-block 2g is the low nibble of qs[32g..], 2g+1 the high.
        __m256i qv[8];
        for (int g = 0; g < 4; g++) {
            const __m256i qb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qs + 32 * g));
            qv[2 * g] = _mm256_and_si256(qb, kLo);
            qv[2 * g + 1] = _mm256_and_si256(_mm256_srli_epi16(qb, 4), kLo);
        }
        for (int r = 0; r < 8; r++) {
            const i8 *xr = xq + static_cast<i64>(r) * rstride + nb * 256;
            const __m256 dxv =
                _mm256_loadu_ps(dx + static_cast<i64>(r) * dstride + nb * 8);
            // dx_s * d * sc_s for the eight sub-blocks, then broadcast one lane
            // at a time. The store/broadcast pair keeps these on the load
            // ports; permuting would spend eight ALU slots instead.
            f32 af[8];
            _mm256_storeu_ps(af, _mm256_mul_ps(sfv, dxv));
            for (int s = 0; s < 8; s++) {
                const __m256i xv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xr + 32 * s));
                const __m256i iv = _mm256_dpbusd_epi32(zero, qv[s], xv);
                acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iv),
                                         _mm256_broadcast_ss(af + s), acc[r]);
            }
            const __m256i s8 = _mm256_loadu_si256(reinterpret_cast<
                const __m256i *>(sx + static_cast<i64>(r) * sstride + nb * 8));
            acc[r] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(s8),
                                      _mm256_mul_ps(mfv, dxv), acc[r]);
        }
    }
    for (int r = 0; r < 8; r++) {
        __m128 lo = _mm256_castps256_ps128(acc[r]);
        __m128 hi = _mm256_extractf128_ps(acc[r], 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[static_cast<i64>(r) * out_stride] = _mm_cvtss_f32(lo);
    }
}

#endif // KRK_DOT_AVX512

#if defined(KRK_DOT_AVX2)
// The AVX2 twin, for hosts without VNNI: maddubs + madd instead of dpbusd.
// Same batched min term, which is where most of the gain is.
__attribute__((target("avx2,fma")))
void dot_q4_K_q8_sx_x8_avx2_d(const u8 *w, const i8 *xq, const f32 *dx,
                              const i32 *sx, i64 n, i64 rstride, i64 dstride,
                              i64 sstride, f32 *out, i64 out_stride) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 144) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (w[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(w[2] | (w[3] << 8)));
        const u8 *sc = w + 4;
        const u8 *qs = w + 16;
        f32 sf[8], mf[8];
        for (int s = 0; s < 8; s++) {
            u8 a = 0, b = 0;
            get_scale_min_k4(s, sc, &a, &b);
            sf[s] = static_cast<f32>(a) * d;
            mf[s] = static_cast<f32>(b) * dmin;
        }
        const __m256 sfv = _mm256_loadu_ps(sf);
        const __m256 mfv = _mm256_loadu_ps(mf);
        __m256i qv[8];
        for (int g = 0; g < 4; g++) {
            const __m256i qb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qs + 32 * g));
            qv[2 * g] = _mm256_and_si256(qb, kLo);
            qv[2 * g + 1] = _mm256_and_si256(_mm256_srli_epi16(qb, 4), kLo);
        }
        for (int r = 0; r < 8; r++) {
            const i8 *xr = xq + static_cast<i64>(r) * rstride + nb * 256;
            const __m256 dxv =
                _mm256_loadu_ps(dx + static_cast<i64>(r) * dstride + nb * 8);
            f32 af[8];
            _mm256_storeu_ps(af, _mm256_mul_ps(sfv, dxv));
            for (int s = 0; s < 8; s++) {
                const __m256i xv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xr + 32 * s));
                const __m256i iv = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(qv[s], xv), ones16);
                acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iv),
                                         _mm256_broadcast_ss(af + s), acc[r]);
            }
            const __m256i s8 = _mm256_loadu_si256(reinterpret_cast<
                const __m256i *>(sx + static_cast<i64>(r) * sstride + nb * 8));
            acc[r] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(s8),
                                      _mm256_mul_ps(mfv, dxv), acc[r]);
        }
    }
    for (int r = 0; r < 8; r++) {
        __m128 lo = _mm256_castps256_ps128(acc[r]);
        __m128 hi = _mm256_extractf128_ps(acc[r], 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[static_cast<i64>(r) * out_stride] = _mm_cvtss_f32(lo);
    }
}

#endif // KRK_DOT_AVX2

// ---------------------------------------------------------------------------
// Q6_K with the per-16 bias folded into the dot product.
//
// The kernel below (dot_q6_K_q8_d) rebuilds  S = sum(x) over 16  for every
// weight row and then computes  t = I - 32*S  in int32, before converting. Two
// things are wrong with that here. The sum is per ROW, so it is recomputed for
// every one of the n_out weight rows; and the subtract-then-convert costs a
// multiply and a subtract per 16 values on top of the dot.
//
// x86 can do better: vpdpbusd takes an ACCUMULATOR, so the -32*S correction can
// be sitting in the lanes before the dot lands. Q6_K's nibbles are 0..63 (u8,
// the unsigned operand) and the activation is int8 (the signed one), so one
// vpdpbusd per 32 values replaces maddubs+madd+mullo+sub, and the bias that
// used to be a correction is free. sq4[] gives the biased sums once per activation
// row instead of once per weight row.
//
// Lane structure: both vpdpbusd and maddubs+madd land on 8 int32 lanes of four
// values each, so lanes 0-3 are the first 16 values and 4-7 the second -- which
// is exactly where `sc` changes. One f32 vector carries both scales.
// One 32-value group's Q6_K nibbles: 4 bits from ql, 2 from qh, plus the 4.
__attribute__((target("avx2")))
inline __m256i q6_group(const u8 *ql, __m256i qhv, int g) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i k3 = _mm256_set1_epi8(0x03);
    // Groups 0 and 1 take the low nibble of ql[0..31] and ql[32..63]; groups 2
    // and 3 take the HIGH nibble of the same 64 bytes.
    const __m256i qlv = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(ql + 32 * (g & 1)));
    __m256i lo4 = qlv;
    if (g >= 2) lo4 = _mm256_srli_epi16(qlv, 4);
    lo4 = _mm256_and_si256(lo4, kLo);
    __m256i hi = qhv;
    if (g) hi = _mm256_srli_epi16(qhv, 2 * g);
    hi = _mm256_and_si256(hi, k3);
    hi = _mm256_slli_epi16(hi, 4);
    return _mm256_or_si256(lo4, hi);
}

#if defined(KRK_DOT_AVX512)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
f32 dot_q6_K_q8_sq4_vnni_d(const u8 *w, const i8 *xq, const f32 *dx,
                           const i32 *sq4, i64 n) {
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 210) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[208] | (w[209] << 8)));
        for (int half = 0; half < 2; half++) {
            const u8 *ql = w + half * 64;
            const u8 *qh = w + 128 + half * 32;
            const i8 *sc = reinterpret_cast<const i8 *>(w + 192 + half * 8);
            const __m256i qhv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qh));
            const i64 vbase = nb * 256 + half * 128;
            for (int g = 0; g < 4; g++) {
                const __m256i q = q6_group(ql, qhv, g);
                const __m256i xv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xq + vbase + 32 * g));
                // -32 * sum(4) per lane, precomputed: the dot lands on it.
                const __m256i pm = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(sq4 + (vbase + 32 * g) / 4));
                const __m256i iv = _mm256_dpbusd_epi32(pm, q, xv);
                const f32 dxg = dx[(vbase + 32 * g) / 32];
                const f32 s0 = d * dxg * static_cast<f32>(sc[2 * g + 0]);
                const f32 s1 = d * dxg * static_cast<f32>(sc[2 * g + 1]);
                acc = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iv),
                    _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1), acc);
            }
        }
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif // KRK_DOT_AVX512

// The AVX2 twin. maddubs + madd give the same four-values-per-lane structure,
// so the pre-biased sums are one vpaddd instead of being the accumulator.
__attribute__((target("avx2,fma")))
f32 dot_q6_K_q8_sq4_avx2_d(const u8 *w, const i8 *xq, const f32 *dx,
                           const i32 *sq4, i64 n) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 210) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[208] | (w[209] << 8)));
        for (int half = 0; half < 2; half++) {
            const u8 *ql = w + half * 64;
            const u8 *qh = w + 128 + half * 32;
            const i8 *sc = reinterpret_cast<const i8 *>(w + 192 + half * 8);
            const __m256i qhv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qh));
            const i64 vbase = nb * 256 + half * 128;
            for (int g = 0; g < 4; g++) {
                const __m256i q = q6_group(ql, qhv, g);
                const __m256i xv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xq + vbase + 32 * g));
                const __m256i pm = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(sq4 + (vbase + 32 * g) / 4));
                const __m256i iv = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones16), pm);
                const f32 dxg = dx[(vbase + 32 * g) / 32];
                const f32 s0 = d * dxg * static_cast<f32>(sc[2 * g + 0]);
                const f32 s1 = d * dxg * static_cast<f32>(sc[2 * g + 1]);
                acc = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iv),
                    _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1), acc);
            }
        }
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// Eight activation rows against one weight row. The weight block -- 210 bytes,
// and the nibble unpack with it -- is decoded ONCE and used eight times. That
// is the prefill shape, and without it Q6_K prefill runs at GEMV speed.
#if defined(KRK_DOT_AVX512)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
void dot_q6_K_q8_sq4_x8_vnni_d(const u8 *w, const i8 *xq, const f32 *dx,
                               const i32 *sq4, i64 n, i64 rstride, i64 dstride,
                               i64 sstride, f32 *out, i64 out_stride) {
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 210) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[208] | (w[209] << 8)));
        // q and the weight scale, once per block, shared by all eight rows.
        __m256i qv[8];
        __m256 bwv[8];
        for (int half = 0; half < 2; half++) {
            const u8 *ql = w + half * 64;
            const u8 *qh = w + 128 + half * 32;
            const i8 *sc = reinterpret_cast<const i8 *>(w + 192 + half * 8);
            const __m256i qhv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qh));
            for (int g = 0; g < 4; g++) {
                qv[half * 4 + g] = q6_group(ql, qhv, g);
                const f32 b0 = d * static_cast<f32>(sc[2 * g + 0]);
                const f32 b1 = d * static_cast<f32>(sc[2 * g + 1]);
                bwv[half * 4 + g] =
                    _mm256_setr_ps(b0, b0, b0, b0, b1, b1, b1, b1);
            }
        }
        for (int r = 0; r < 8; r++) {
            const i8 *xr = xq + static_cast<i64>(r) * rstride + nb * 256;
            const i32 *sr = sq4 + static_cast<i64>(r) * sstride + nb * 64;
            const f32 *dr = dx + static_cast<i64>(r) * dstride + nb * 8;
            for (int half = 0; half < 2; half++) {
                for (int g = 0; g < 4; g++) {
                    const i64 off = half * 128 + 32 * g;
                    const __m256i xv = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(xr + off));
                    const __m256i pm = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(sr + off / 4));
                    const __m256i iv = _mm256_dpbusd_epi32(pm, qv[half * 4 + g],
                                                           xv);
                    const __m256 scale = _mm256_mul_ps(
                        bwv[half * 4 + g],
                        _mm256_broadcast_ss(dr + half * 4 + g));
                    acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iv), scale,
                                             acc[r]);
                }
            }
        }
    }
    for (int r = 0; r < 8; r++) {
        __m128 lo = _mm256_castps256_ps128(acc[r]);
        __m128 hi = _mm256_extractf128_ps(acc[r], 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[static_cast<i64>(r) * out_stride] = _mm_cvtss_f32(lo);
    }
}
#endif // KRK_DOT_AVX512

__attribute__((target("avx2,fma")))
void dot_q6_K_q8_sq4_x8_avx2_d(const u8 *w, const i8 *xq, const f32 *dx,
                               const i32 *sq4, i64 n, i64 rstride, i64 dstride,
                               i64 sstride, f32 *out, i64 out_stride) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 210) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[208] | (w[209] << 8)));
        __m256i qv[8];
        __m256 bwv[8];
        for (int half = 0; half < 2; half++) {
            const u8 *ql = w + half * 64;
            const u8 *qh = w + 128 + half * 32;
            const i8 *sc = reinterpret_cast<const i8 *>(w + 192 + half * 8);
            const __m256i qhv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qh));
            for (int g = 0; g < 4; g++) {
                qv[half * 4 + g] = q6_group(ql, qhv, g);
                const f32 b0 = d * static_cast<f32>(sc[2 * g + 0]);
                const f32 b1 = d * static_cast<f32>(sc[2 * g + 1]);
                bwv[half * 4 + g] =
                    _mm256_setr_ps(b0, b0, b0, b0, b1, b1, b1, b1);
            }
        }
        for (int r = 0; r < 8; r++) {
            const i8 *xr = xq + static_cast<i64>(r) * rstride + nb * 256;
            const i32 *sr = sq4 + static_cast<i64>(r) * sstride + nb * 64;
            const f32 *dr = dx + static_cast<i64>(r) * dstride + nb * 8;
            for (int half = 0; half < 2; half++) {
                for (int g = 0; g < 4; g++) {
                    const i64 off = half * 128 + 32 * g;
                    const __m256i xv = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(xr + off));
                    const __m256i pm = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(sr + off / 4));
                    const __m256i iv = _mm256_add_epi32(
                        _mm256_madd_epi16(
                            _mm256_maddubs_epi16(qv[half * 4 + g], xv),
                            ones16),
                        pm);
                    const __m256 scale = _mm256_mul_ps(
                        bwv[half * 4 + g],
                        _mm256_broadcast_ss(dr + half * 4 + g));
                    acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iv), scale,
                                             acc[r]);
                }
            }
        }
    }
    for (int r = 0; r < 8; r++) {
        __m128 lo = _mm256_castps256_ps128(acc[r]);
        __m128 hi = _mm256_extractf128_ps(acc[r], 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[static_cast<i64>(r) * out_stride] = _mm_cvtss_f32(lo);
    }
}

// ---------------------------------------------------------------------------
// Q6_K against an int8 activation.
//
// Q6_K stores   w_k = d * sc_s * (q_k - 32)   q_k in 0..63, sc one int8 per 16
// so
//     sum(w*x) = d * sum_s sc_s * (I_s - 32*S_s)
// with I_s = sum(q_k*x_k) and S_s = sum(x_k), both EXACT integers once x is
// int8. Weights are read from the blocks they are stored in.
//
// Lane structure: maddubs puts values 0..15 in lanes 0-3 and 16..31 in lanes
// 4-7, which is exactly where the scale changes (one sc per 16 values), so a
// single 8-lane float vector carries both halves' scales.
__attribute__((target("avx2,fma"))) f32 dot_q6_K_q8_d(const u8 *w, const i8 *xq,
                                                      const f32 *dx, i64 n) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i c32 = _mm256_set1_epi32(32);
    __m256 acc = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 210) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[208] | (w[209] << 8)));
        for (int half = 0; half < 2; half++) {
            const u8 *ql = w + half * 64;
            const u8 *qh = w + 128 + half * 32;
            const i8 *sc = reinterpret_cast<const i8 *>(w + 192 + half * 8);
            const __m256i qhv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qh));
            const i64 base = nb * 256 + half * 128;
            const f32 *dxh = dx + base / 32;
            for (int g = 0; g < 4; g++) {
                // Groups 0 and 1 take the LOW nibble of ql[0..31] and
                // ql[32..63]; groups 2 and 3 take the HIGH nibble of the same
                // 64 bytes. Reading ql+64 and ql+96 instead -- which is what
                // the obvious bit layout suggests -- is a different tensor:
                // measured 32% relative error against the exact dequant.
                const __m256i qlv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(ql + 32 * (g & 1)));
                __m256i lo4 = qlv;
                if (g >= 2) lo4 = _mm256_srli_epi16(qlv, 4);
                lo4 = _mm256_and_si256(lo4, kLo);
                __m256i hi = qhv;
                if (g) hi = _mm256_srli_epi16(qhv, 2 * g);
                hi = _mm256_and_si256(hi, _mm256_set1_epi8(0x03));
                hi = _mm256_slli_epi16(hi, 4);
                const __m256i q = _mm256_or_si256(lo4, hi);
                const __m256i xv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(xq + base + 32 * g));
                const __m256i iv = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(q, xv), ones16);
                const __m256i sv = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(ones8, xv), ones16);
                const __m256i t = _mm256_sub_epi32(iv, _mm256_mullo_epi32(sv, c32));
                // lanes 0-3 -> first 16 values, lanes 4-7 -> second 16, and the
                // scale changes exactly there.
                // Two STRIDES, and getting them confused is silent: the
                // activation scale `dx` is one per 32 values, so the whole
                // group shares dxh[g]; the weight scale `sc` is one per 16, so
                // the two halves of the group take sc[2g] and sc[2g+1].
                // Indexing dx with 2g read a neighbouring group's scale and
                // measured 32% relative error on synthetic data.
                const f32 s0 = d * dxh[g] * static_cast<f32>(sc[2 * g + 0]);
                const f32 s1 = d * dxh[g] * static_cast<f32>(sc[2 * g + 1]);
                const __m256 scale = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(t), scale, acc);
            }
        }
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ---------------------------------------------------------------------------
// Q4_K against int8, with vpdpbusd (AVX-512 VNNI, 256-bit form).
//
// Same algebra as dot_q4_K_q8_d -- the integer dot is exact, d/s and dmin/m are
// applied once per sub-block -- but the dot itself is one instruction per 32
// values instead of two. Measured at the GEMM shapes Tiny-Moe actually calls:
// 3.5 MAC/instr (AVX2) -> 7.4 (VNNI).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
f32 dot_q4_K_q8_vnni_d(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i ones8 = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    for (i64 nb = 0; nb < n / 256; nb++, w += 144) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (w[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(w[2] | (w[3] << 8)));
        const u8 *sc = w + 4;
        const u8 *qs = w + 16;
        for (int j = 0; j < 256; j += 64) {
            const __m256i qb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qs));
            const __m256i q_lo = _mm256_and_si256(qb, kLo);
            const __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), kLo);
            const i64 base = (nb * 256) + j;
            const __m256i x_lo = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base));
            const __m256i x_hi = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base + 32));
            u8 s1, m1, s2, m2;
            get_scale_min_k4(static_cast<int>(j / 32) + 0, sc, &s1, &m1);
            get_scale_min_k4(static_cast<int>(j / 32) + 1, sc, &s2, &m2);
            // Both sums come out of vpdpbusd: q*x for the integer dot, and
            // 1*x for the activation sum the dmin term needs.
            const __m256i i1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_lo, x_lo);
            const __m256i i2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_hi, x_hi);
            const __m256i t1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones8, x_lo);
            const __m256i t2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones8, x_hi);
            const f32 dx1 = dx[base / 32];
            const f32 dx2 = dx[(base + 32) / 32];
            const f32 sA = dx1 * d * static_cast<f32>(s1);
            const f32 sB = dx2 * d * static_cast<f32>(s2);
            const f32 mA = dx1 * dmin * static_cast<f32>(m1);
            const f32 mB = dx2 * dmin * static_cast<f32>(m2);
            __m256 t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), _mm256_set1_ps(sA),
                                       _mm256_mul_ps(_mm256_cvtepi32_ps(i2),
                                                     _mm256_set1_ps(sB)));
            t = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(t1), _mm256_set1_ps(mA), t);
            t = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(t2), _mm256_set1_ps(mB), t);
            acc = _mm256_add_ps(acc, t);
            qs += 32;
        }
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// Q4_K with the dmin term HOISTED. sx[] carries sum(xi) per 32-value sub-block,
// which the quantise pass already knows. What that removes from the inner loop
// is not small: the previous form spent two vpdpbusd (against an all-ones
// operand), two cvt and two fnmadd per 64 values recomputing a number that is
// identical for every one of the n_out weight rows. Here it is one scalar fma
// per sub-block against a value that was computed once.
//
// Measured on Tiny-Moe: see the commit.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
f32 dot_q4_K_q8_sx_vnni_d(const u8 *w, const i8 *xq, const f32 *dx,
                          const i32 *sx, i64 n) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    f32 macc = 0.0f;
    for (i64 nb = 0; nb < n / 256; nb++, w += 144) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (w[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(w[2] | (w[3] << 8)));
        const u8 *sc = w + 4;
        const u8 *qs = w + 16;
        for (int j = 0; j < 256; j += 64) {
            const __m256i qb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qs));
            const __m256i q_lo = _mm256_and_si256(qb, kLo);
            const __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), kLo);
            const i64 base = (nb * 256) + j;
            const i64 sb = base / 32;
            const __m256i x_lo = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base));
            const __m256i x_hi = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base + 32));
            u8 s1, m1, s2, m2;
            get_scale_min_k4(static_cast<int>(j / 32) + 0, sc, &s1, &m1);
            get_scale_min_k4(static_cast<int>(j / 32) + 1, sc, &s2, &m2);
            const __m256i i1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                                   q_lo, x_lo);
            const __m256i i2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                                   q_hi, x_hi);
            const f32 dx1 = dx[sb];
            const f32 dx2 = dx[sb + 1];
            __m256 t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1),
                                       _mm256_set1_ps(dx1 * d * static_cast<f32>(s1)),
                                       _mm256_mul_ps(_mm256_cvtepi32_ps(i2),
                                                     _mm256_set1_ps(dx2 * d * static_cast<f32>(s2))));
            acc = _mm256_add_ps(acc, t);
            // Scalar, once per sub-block: the term the vector loop used to
            // rebuild for every weight row.
            macc += dx1 * dmin * static_cast<f32>(m1) * static_cast<f32>(sx[sb]);
            macc += dx2 * dmin * static_cast<f32>(m2) * static_cast<f32>(sx[sb + 1]);
            qs += 32;
        }
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo) - macc;
}

// The same, on the AVX2 path (no VNNI: maddubs + madd instead of dpbusd).
__attribute__((target("avx2,fma")))
f32 dot_q4_K_q8_sx_avx2_d(const u8 *w, const i8 *xq, const f32 *dx,
                          const i32 *sx, i64 n) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    f32 macc = 0.0f;
    for (i64 nb = 0; nb < n / 256; nb++, w += 144) {
        const f32 d = fp16_to_fp32(static_cast<u16>(w[0] | (w[1] << 8)));
        const f32 dmin = fp16_to_fp32(static_cast<u16>(w[2] | (w[3] << 8)));
        const u8 *sc = w + 4;
        const u8 *qs = w + 16;
        for (int j = 0; j < 256; j += 64) {
            const __m256i qb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(qs));
            const __m256i q_lo = _mm256_and_si256(qb, kLo);
            const __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), kLo);
            const i64 base = (nb * 256) + j;
            const i64 sb = base / 32;
            const __m256i x_lo = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base));
            const __m256i x_hi = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + base + 32));
            u8 s1, m1, s2, m2;
            get_scale_min_k4(static_cast<int>(j / 32) + 0, sc, &s1, &m1);
            get_scale_min_k4(static_cast<int>(j / 32) + 1, sc, &s2, &m2);
            const __m256i i1 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(q_lo, x_lo), ones16);
            const __m256i i2 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(q_hi, x_hi), ones16);
            const f32 dx1 = dx[sb];
            const f32 dx2 = dx[sb + 1];
            const __m256 t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1),
                                             _mm256_set1_ps(dx1 * d * static_cast<f32>(s1)),
                                             _mm256_mul_ps(_mm256_cvtepi32_ps(i2),
                                                           _mm256_set1_ps(dx2 * d * static_cast<f32>(s2))));
            acc = _mm256_add_ps(acc, t);
            macc += dx1 * dmin * static_cast<f32>(m1) * static_cast<f32>(sx[sb]);
            macc += dx2 * dmin * static_cast<f32>(m2) * static_cast<f32>(sx[sb + 1]);
            qs += 32;
        }
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo) - macc;
}

std::atomic<bool> g_dot_int8{false};
std::atomic<bool> g_dot_int8_set{false};

bool env_dot_int8() {
    const char *e = std::getenv("KRK_CPU_DOT_INT8");
    if (!e || e[0] == '\0') {
#if defined(KRK_DOT_AVX2)
        return cpu_has_avx2();
#else
        return false;
#endif
    }
    return e[0] != '0';
}

std::atomic<bool> g_dot_avx2{false};
std::atomic<bool> g_dot_avx2_set{false};

// Unset means "ask the CPU". The two flags exist to choose a ROUNDING TIER,
// not to unlock a feature: these kernels are bit-identical to the four-lane
// path (test_vec_dot_avx2_matches_fast asserts that, not a tolerance), so a
// host that can run them should run them without being told. Measured on
// Tiny-Moe Q4_K_M, --cpu, 46-token prefill: 22.2 -> 47.8 tok/s prefill and
// 13.9 -> 21.3 tok/s decode, with the same 24 greedy tokens to the byte.
// `KRK_CPU_DOT_AVX2=0` still forces the scalar path for a bit-exactness run.
bool dot_avx2_default() {
#if defined(KRK_DOT_AVX2)
    return cpu_has_avx2();
#else
    return false;
#endif
}
bool env_dot_avx2() {
    const char *e = std::getenv("KRK_CPU_DOT_AVX2");
    if (!e || e[0] == '\0') return dot_avx2_default();
    return e[0] != '0';
}
} // namespace

// The kernels live in the anonymous namespace above; these are the names the
// backend links against. Without AVX2 there is no int8 kernel and the flag is
// false, so the stubs are never reached -- they exist so the translation unit
// links on every target.
// Per-FOUR biased sums, for Q6_K.
//
// Q6_K stores  w = d * sc_s * (q_k - 32)  and its dot needs  sum((q-32)*x).
// The kernel forms that with one vpdpbusd, whose 8 int32 lanes each own FOUR
// consecutive values -- so the -32 correction has to be present once per lane,
// not once per 16 (putting a 16-sum in all four lanes subtracts it four times;
// measured 3.8e+02 relative, which is how this was found).
//
// sx[i] = -32 * sum(x[4i .. 4i+3]), computed once per activation row. It is the
// ACCUMULATOR the dot lands on, so the correction costs the kernel nothing at
// all -- not a subtract, not a multiply, not a permute. One int32 per four
// values is n bytes per row.
__attribute__((target("avx2"))) inline __m256i bias4_of_32(const i8 *p) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    const __m256i pairs = _mm256_maddubs_epi16(_mm256_set1_epi8(1), v);
    const __m256i quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    return _mm256_sub_epi32(_mm256_setzero_si256(),
                            _mm256_slli_epi32(quads, 5));
}

void quantize_q8_row_sq4(const f32 *x, i64 n, i8 *xq, f32 *dx, i32 *sq4) {
#if defined(KRK_DOT_AVX2)
    quantize_q8_row_d(x, n, xq, dx);
    for (i64 b = 0; b < n; b += 32) {
        const __m256i c = bias4_of_32(xq + b);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sq4 + b / 4), c);
    }
#else
    (void)x; (void)n; (void)xq; (void)dx; (void)sq4;
#endif
}

void quantize_q8_row_sx(const f32 *x, i64 n, i8 *xq, f32 *dx, i32 *sx) {
#if defined(KRK_DOT_AVX2)
    quantize_q8_row_d(x, n, xq, dx);
    for (i64 b = 0; b < n; b += 32) sx[b / 32] = sum_i8_32(xq + b);
#else
    (void)x; (void)n; (void)xq; (void)dx; (void)sx;
#endif
}

void quantize_q8_row(const f32 *x, i64 n, i8 *xq, f32 *dx) {
#if defined(KRK_DOT_AVX2)
    quantize_q8_row_d(x, n, xq, dx);
#else
    (void)x; (void)n; (void)xq; (void)dx;
#endif
}

f32 dot_q4_K_q8(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
#if defined(KRK_DOT_AVX512)
    // One binary, three ISAs: VNNI where the host has it, the AVX2 kernel
    // otherwise. Zen 3 falls through to AVX2 by construction.
    if (cpu_has_vnni()) return dot_q4_K_q8_vnni_d(w, xq, dx, n);
#endif
#if defined(KRK_DOT_AVX2)
    return dot_q4_K_q8_d(w, xq, dx, n);
#else
    (void)w; (void)xq; (void)dx; (void)n;
    return 0.0f;
#endif
}

f32 dot_q4_K_q8_sx(const u8 *w, const i8 *xq, const f32 *dx, const i32 *sx,
                  i64 n) {
#if defined(KRK_DOT_AVX512)
    if (cpu_has_vnni()) return dot_q4_K_q8_sx_vnni_d(w, xq, dx, sx, n);
#endif
#if defined(KRK_DOT_AVX2)
    return dot_q4_K_q8_sx_avx2_d(w, xq, dx, sx, n);
#else
    (void)w; (void)xq; (void)dx; (void)sx; (void)n;
    return 0.0f;
#endif
}

f32 dot_q6_K_q8_sq4(const u8 *w, const i8 *xq, const f32 *dx, const i32 *sq4,
                    i64 n) {
#if defined(KRK_DOT_AVX512)
    if (cpu_has_vnni()) return dot_q6_K_q8_sq4_vnni_d(w, xq, dx, sq4, n);
#endif
#if defined(KRK_DOT_AVX2)
    return dot_q6_K_q8_sq4_avx2_d(w, xq, dx, sq4, n);
#else
    (void)w; (void)xq; (void)dx; (void)sq4; (void)n;
    return 0.0f;
#endif
}

void dot_q6_K_q8_sq4_x8(const u8 *w, const i8 *xq, const f32 *dx,
                        const i32 *sq4, i64 n, i64 rstride, i64 dstride,
                        i64 sstride, f32 *out, i64 out_stride) {
    static const char *e = std::getenv("KRK_CPU_DOT_Q6SQ4");
    const bool use = !e || e[0] != '0';
    if (!use) {
        for (int r = 0; r < 8; r++)
            out[static_cast<i64>(r) * out_stride] =
                dot_q6_K_q8(w, xq + static_cast<i64>(r) * rstride,
                            dx + static_cast<i64>(r) * dstride, n);
        return;
    }
#if defined(KRK_DOT_AVX512)
    if (cpu_has_vnni()) {
        dot_q6_K_q8_sq4_x8_vnni_d(w, xq, dx, sq4, n, rstride, dstride, sstride,
                                  out, out_stride);
        return;
    }
#endif
#if defined(KRK_DOT_AVX2)
    dot_q6_K_q8_sq4_x8_avx2_d(w, xq, dx, sq4, n, rstride, dstride, sstride, out,
                              out_stride);
#else
    (void)w; (void)xq; (void)dx; (void)sq4; (void)n; (void)rstride;
    (void)dstride; (void)sstride; (void)out; (void)out_stride;
#endif
}

f32 dot_q6_K_q8(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
#if defined(KRK_DOT_AVX2)
    return dot_q6_K_q8_d(w, xq, dx, n);
#else
    (void)w; (void)xq; (void)dx; (void)n;
    return 0.0f;
#endif
}

// Default on: the batched-min kernel is the fast path. KRK_CPU_DOT_X8SX=0
// routes the eight-row prefill shape back through the per-row min term, which
// is the A/B used to measure it and the fallback if a host ever disagrees.
bool env_dot_x8sx() {
    const char *e = std::getenv("KRK_CPU_DOT_X8SX");
    return !e || e[0] != '0';
}

// Q8_0 against int8 activations, WITHOUT expanding either side to f32.
//
// Q8_0 was the last format whose batched prefill still ran through a dequant
// loop: dot_q8_0_tiled() widened every weight byte to f32 with cvtepi8_epi32
// and multiplied by the block scale, eight rows at a time. Measured on the
// 768x12 q8_0 fixture that path took 81.3% of a 2040-token prefill run at
// 19.3 GMAC/s, i.e. the largest single cost in the engine and a direct
// violation of "never dequant".
//
// Q8_0's weight is a SIGNED byte, and maddubs wants its first operand unsigned.
// The obvious fix -- bias by 128 and correct with -128*sum(x) -- is the one
// Intel's VNNI sample uses, and it is unsafe here: 255*127*2 overflows the i16
// pair accumulator and maddubs would silently saturate. So the weight is split
// into its magnitude and its sign instead, and both magnitudes are unsigned:
//
//     w = wp - wn,   wp = max(w,0) in 0..127,  wn = |min(w,0)| in 1..128
//     sum(w*x) = madd(maddubs(wp, x), 1) - madd(maddubs(wn, x), 1)
//
// Both products are bounded by 128*127*2 = 32512 < 32767, so nothing saturates
// for ANY legal byte pair, including the -128 that the format permits and that
// an abs_epi8 (which saturates to 127) would get wrong. |min(w,0)| is computed
// as a wrapping negate of the bit pattern, which is exactly 256-pattern, so
// -128 becomes 128 without ever being negated as a signed value.
//
// No bias term and no sx[] array are needed: the split is exact, so the only
// per-block cost above the dot is the scale product. Q8_0 blocks 32 values, so
// the scale changes every 32 and one cvt + one fma per block per row is the
// floor for this layout.
__attribute__((target("avx2,fma")))
// ---------------------------------------------------------------------------
// Single-row q8_0 dot: the DECODE kernel.
//
// Everything in this file that got faster over the last few rounds -- the sign
// flip, the f16c scale decode, VNNI -- was applied to the eight-row prefill
// kernel. Decode calls this one, once per output element with one activation
// row, and it was still the original 256-bit form: two maddubs per block for
// the positive and negative halves, a scalar fp16 decode, no VNNI, no 512-bit
// width. On a top-4 MoE decode step this is the kernel that runs the most.
//
// Same three ideas, applied here:
//   * sign flip: sum(w*x) = sum(|w| * (x * sign(w))), one maddubs instead of
//     two, and the negative half never exists as a register;
//   * one F16C convert for two block scales;
//   * VNNI where the host has it, 512-bit BW where it does not, the existing
//     256-bit kernel below that, and scalar as the floor.
//
// The rungs are bit-compatible with each other to the f32 reassociation the
// other fixtures bound at 1e-6, and tools/ta_q80_tiers.cpp checks them.

// Scalar floor. Same rounding points as the vector rungs so a rung that matches
// it is showing identical arithmetic rather than a lucky approximation.
f32 dot_q8_0_q8_scalar(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    const i64 nb = n / 32;
    f32 acc = 0.0f;
    for (i64 b = 0; b < nb; b++, w += 34) {
        const f32 d = fp16_to_fp32(
            static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
        i64 sum = 0;
        for (i64 i = 0; i < 32; i++) {
            const i32 wv =
                static_cast<i32>(static_cast<i8>(static_cast<u8>(w[2 + i])));
            const i32 xv =
                static_cast<i32>(static_cast<i8>(static_cast<u8>(xq[b * 32 + i])));
            sum += static_cast<i64>(wv) * static_cast<i64>(xv);
        }
        acc = std::fma(static_cast<f32>(sum), d * dx[b], acc);
    }
    return acc;
}

// 512-bit, two blocks per iteration, sign flipped. BW only -- no VNNI needed.
__attribute__((target("avx512f,avx512bw,f16c")))
f32 dot_q8_0_q8_bw(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    const i64 nb = n / 32;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc = _mm512_setzero_ps();
    for (i64 b = 0; b + 2 <= nb; b += 2, w += 68) {
        const __m128i two = _mm_setr_epi16(
            static_cast<short>(w[0] | (static_cast<u16>(w[1]) << 8)),
            static_cast<short>(w[34] | (static_cast<u16>(w[35]) << 8)), 0, 0, 0, 0,
            0, 0);
        const __m128 d2 = _mm_cvtph_ps(two);
        const f32 d0 = _mm_cvtss_f32(d2);
        const f32 d1 = _mm_cvtss_f32(_mm_shuffle_ps(d2, d2, 0x55));
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36));
        const __m512i wv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
        const __m512i xv = _mm512_loadu_si512(
            reinterpret_cast<const void *>(xq + b * 32));
        const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
        const __m512i iv = _mm512_madd_epi16(_mm512_maddubs_epi16(mag, xb), ones);
        const f32 s0 = d0 * dx[b];
        const f32 s1 = d1 * dx[b + 1];
        const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
            _mm512_castps_si512(
                _mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
            _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc);
    }
    // An odd block count leaves the last block to the tail below.
    f32 tail = 0.0f;
    for (i64 b = (nb / 2) * 2; b < nb; b++) {
        const u8 *blk = w + (b - (nb / 2) * 2) * 34;
        const f32 d = fp16_to_fp32(
            static_cast<u16>(blk[0] | (static_cast<u16>(blk[1]) << 8)));
        i64 sum = 0;
        for (i64 i = 0; i < 32; i++) {
            const i32 wv2 =
                static_cast<i32>(static_cast<i8>(static_cast<u8>(blk[2 + i])));
            const i32 xv2 = static_cast<i32>(
                static_cast<i8>(static_cast<u8>(xq[b * 32 + i])));
            sum += static_cast<i64>(wv2) * static_cast<i64>(xv2);
        }
        tail = std::fma(static_cast<f32>(sum), d * dx[b], tail);
    }
    return _mm512_reduce_add_ps(acc) + tail;
}

// VNNI: one dpbusd where the BW rung needs maddubs + madd.
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
f32 dot_q8_0_q8_vnni(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    const i64 nb = n / 32;
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc = _mm512_setzero_ps();
    for (i64 b = 0; b + 2 <= nb; b += 2, w += 68) {
        const __m128i two = _mm_setr_epi16(
            static_cast<short>(w[0] | (static_cast<u16>(w[1]) << 8)),
            static_cast<short>(w[34] | (static_cast<u16>(w[35]) << 8)), 0, 0, 0, 0,
            0, 0);
        const __m128 d2 = _mm_cvtph_ps(two);
        const f32 d0 = _mm_cvtss_f32(d2);
        const f32 d1 = _mm_cvtss_f32(_mm_shuffle_ps(d2, d2, 0x55));
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36));
        const __m512i wv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
        const __m512i xv = _mm512_loadu_si512(
            reinterpret_cast<const void *>(xq + b * 32));
        const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
        // Unsigned operand second: see the note on the eight-row VNNI kernel.
        const __m512i iv = _mm512_dpbusd_epi32(zero, mag, xb);
        const f32 s0 = d0 * dx[b];
        const f32 s1 = d1 * dx[b + 1];
        const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
            _mm512_castps_si512(
                _mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
            _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc);
    }
    const i64 done = (nb / 2) * 2;
    f32 tail = 0.0f;
    for (i64 b = done; b < nb; b++) {
        const u8 *blk = w + (b - done) * 34;
        const f32 d = fp16_to_fp32(
            static_cast<u16>(blk[0] | (static_cast<u16>(blk[1]) << 8)));
        i64 sum = 0;
        for (i64 i = 0; i < 32; i++) {
            const i32 wv2 =
                static_cast<i32>(static_cast<i8>(static_cast<u8>(blk[2 + i])));
            const i32 xv2 = static_cast<i32>(
                static_cast<i8>(static_cast<u8>(xq[b * 32 + i])));
            sum += static_cast<i64>(wv2) * static_cast<i64>(xv2);
        }
        tail = std::fma(static_cast<f32>(sum), d * dx[b], tail);
    }
    return _mm512_reduce_add_ps(acc) + tail;
}

f32 dot_q8_0_q8(const u8 *w, const i8 *xq, const f32 *dx, i64 n) {
    // Decode's ladder, same rungs and same guards as the eight-row kernel.
    // KRK_CPU_Q80_KERNEL forces a rung for testing (r1|bw|vnni|scalar).
    static const i32 forced = [] {
        const char *e = std::getenv("KRK_CPU_Q80_KERNEL");
        if (!e || !e[0]) return 0;
        if (std::strcmp(e, "vnni") == 0) return 3;
        if (std::strcmp(e, "bw") == 0) return 2;
        if (std::strcmp(e, "avx2") == 0) return 1;
        if (std::strcmp(e, "scalar") == 0) return -1;
        return 0;
    }();
#if defined(KRK_DOT_AVX512)
    if ((forced == 0 || forced == 3) && cpu_has_avx512bw() && cpu_has_vnni())
        return dot_q8_0_q8_vnni(w, xq, dx, n);
    if ((forced == 0 || forced == 2) && cpu_has_avx512bw())
        return dot_q8_0_q8_bw(w, xq, dx, n);
#endif
    if (forced == -1) return dot_q8_0_q8_scalar(w, xq, dx, n);
    if (forced == 3 || forced == 2) {
        // Asked for a rung this build or this CPU cannot run: fall through to
        // the best available rather than returning a wrong number.
        if (forced == 3 && !cpu_has_vnni() && cpu_has_avx512bw())
            return dot_q8_0_q8_bw(w, xq, dx, n);
    }
    if (!cpu_has_avx2()) return dot_q8_0_q8_scalar(w, xq, dx, n);
    const i64 nb = n / 32;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (i64 b = 0; b < nb; b++, w += 34) {
        const f32 d = fp16_to_fp32(
            static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
        const __m256i wv =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i sm = _mm256_cmpgt_epi8(zero, wv);
        const __m256i wp = _mm256_andnot_si256(sm, wv);
        const __m256i wn =
            _mm256_and_si256(sm, _mm256_sub_epi8(zero, wv));
        const __m256i xv = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(xq + b * 32));
        const __m256i pv = _mm256_madd_epi16(_mm256_maddubs_epi16(wp, xv), ones);
        const __m256i nv = _mm256_madd_epi16(_mm256_maddubs_epi16(wn, xv), ones);
        const __m256i iv = _mm256_sub_epi32(pv, nv);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iv),
                              _mm256_set1_ps(d * dx[b]), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// Eight activation rows against one weight row. The block decode -- the f16
// scale, the sign mask and the two magnitudes -- is done once and every one of
// the eight rows reuses it, which is the same 8x share the Q4_K/Q6_K kernels
// get. This is the shape batched prefill actually calls.
//
// The magnitude split costs a second maddubs, and at 256 bits that is the whole
// cost: 8 instructions per 32 MACs. Q8_0's block is 32 bytes, so two blocks are
// exactly one 512-bit register and one maddubs pair does 64 values. The
// activations for two consecutive blocks are contiguous (unlike the weights,
// which carry a 2-byte scale between them), so the x side is one aligned-width
// load while the weight side costs two loads plus an insert -- amortised over
// all eight rows. That halves the per-row instruction count to 8 per 64 MACs.
#if defined(KRK_DOT_AVX512)
template <bool kF16C>
__attribute__((target("avx512f,avx512bw,f16c")))
void dot_q8_0_q8_x8_bw_impl(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                            i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    const i64 nb = n / 32;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    for (i64 b = 0; b + 2 <= nb; b += 2, w += 68) {
        // Both block scales in one F16C convert. The scalar fp16_to_fp32 is
        // branchy (it has a denormal-normalisation loop) and the ablation in
        // tools/bench_q80_ablate.cpp prices the pair at 34.5% of this kernel --
        // the largest single cost in the inner loop. _mm_cvtph_ps is bit-identical
        // to it across all 65536 half values, NaN payloads excluded.
        f32 d0, d1;
        if constexpr (kF16C) {
            const __m128i two = _mm_setr_epi16(
                static_cast<short>(w[0] | (static_cast<u16>(w[1]) << 8)),
                static_cast<short>(w[34] | (static_cast<u16>(w[35]) << 8)), 0, 0,
                0, 0, 0, 0);
            const __m128 d2 = _mm_cvtph_ps(two);
            d0 = _mm_cvtss_f32(d2);
            d1 = _mm_cvtss_f32(_mm_shuffle_ps(d2, d2, 0x55));
        } else {
            d0 = fp16_to_fp32(
                static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
            d1 = fp16_to_fp32(
                static_cast<u16>(w[34] | (static_cast<u16>(w[35]) << 8)));
        }
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36));
        const __m512i wv =
            _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        // |w| as an unsigned magnitude. A masked subtract, not abs_epi8: for
        // -128 the negate is 0x80 as u8 = 128, which is the correct magnitude,
        // while abs_epi8 saturates it to 127 and would be wrong for one value
        // in 256.
        const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
        for (int r = 0; r < 8; r++) {
            const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(
                xq + static_cast<i64>(r) * rstride + b * 32));
            // Where the weight is negative the product needs -x, so flip the
            // activation there. One instruction, and it removes the second
            // maddubs and the second magnitude register entirely.
            const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
            const __m512i iv =
                _mm512_madd_epi16(_mm512_maddubs_epi16(mag, xb), ones);
            const f32 s0 = d0 * dx[static_cast<i64>(r) * dstride + b];
            const f32 s1 = d1 * dx[static_cast<i64>(r) * dstride + b + 1];
            // Lanes 0-7 are block b, 8-15 block b+1, because maddubs/madd keep
            // the 512-bit input order. One scale vector, one fma, both blocks.
            const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
                _mm512_castps_si512(_mm512_castps256_ps512(
                    _mm256_broadcast_ss(&s0))),
                _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
        }
    }
    for (int r = 0; r < 8; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}
#endif // KRK_DOT_AVX512

__attribute__((target("avx2,fma")))
void dot_q8_0_q8_x8_avx2(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                         i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    const i64 nb = n / 32;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_ps();
    for (i64 b = 0; b < nb; b++, w += 34) {
        const f32 d = fp16_to_fp32(
            static_cast<u16>(w[0] | (static_cast<u16>(w[1]) << 8)));
        const __m256i wv =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i sm = _mm256_cmpgt_epi8(zero, wv);
        // Same sign-flip as the 512-bit kernel: |w| once, activation flipped
        // per row, one maddubs instead of two.
        const __m256i mag =
            _mm256_blendv_epi8(wv, _mm256_sub_epi8(zero, wv), sm);
        for (int r = 0; r < 8; r++) {
            const __m256i xv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(xq + static_cast<i64>(r) *
                                                          rstride + b * 32));
            const __m256i xb =
                _mm256_blendv_epi8(xv, _mm256_sub_epi8(zero, xv), sm);
            const __m256i iv =
                _mm256_madd_epi16(_mm256_maddubs_epi16(mag, xb), ones);
            const f32 sc = d * dx[static_cast<i64>(r) * dstride + b];
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iv),
                                     _mm256_set1_ps(sc), acc[r]);
        }
    }
    for (int r = 0; r < 8; r++) {
        __m128 lo = _mm256_castps256_ps128(acc[r]);
        __m128 hi = _mm256_extractf128_ps(acc[r], 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[static_cast<i64>(r) * out_stride] = _mm_cvtss_f32(lo);
    }
}

// The entry point the backend calls: widest kernel the host can run. An odd
// block count falls back wholesale rather than growing a second tail path --
// one code path that is always exercised beats two where one is never tested.
// ---------------------------------------------------------------------------
// The Q8_0 x8 kernel ladder: VNNI -> AVX-512BW -> AVX2 -> scalar.
//
// Every rung computes the same eight dot products and every rung is selected at
// runtime from the CPU's own feature bits, so one binary runs on a Zen 1 VM and
// on a Sapphire Rapids part. The ladder exists because the previous dispatcher
// called the AVX2 kernel UNCONDITIONALLY: on any host without AVX2 that is
// SIGILL, not a slow path. A feature that is not available here is still
// implemented -- it just has to be reachable, testable and skippable.
//
// The scalar rung is the floor and the reference every other rung is checked
// against by tools/ta_q80_tiers.cpp.

// Reference. Same rounding points as the vector rungs: the block scale product
// is rounded once as f32, then one fma per block, so a rung that matches this
// bit for bit is showing that it does the identical arithmetic and not merely a
// close one.
void dot_q8_0_q8_x8_scalar(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                           i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    const i64 nb = n / 32;
    for (i32 r = 0; r < 8; r++) {
        const i8 *xr = xq + static_cast<i64>(r) * rstride;
        const f32 *dr = dx + static_cast<i64>(r) * dstride;
        f32 acc = 0.0f;
        for (i64 b = 0; b < nb; b++) {
            const u8 *blk = w + b * 34;
            const f32 dw = fp16_to_fp32(
                static_cast<u16>(blk[0] | (static_cast<u16>(blk[1]) << 8)));
            // Widened through an explicit unsigned-then-signed step and summed
            // in i64. The earlier i32 form measured WRONG against every vector
            // rung on this compiler+CPU: with products bounded by 128*127 and
            // only 32 of them, overflow is impossible, so the accumulation was
            // being miscompiled rather than saturating. i64 costs nothing here
            // -- this is the fallback floor, not a hot path -- and a floor that
            // is wrong is worse than a floor that is slow.
            i64 sum = 0;
            for (i64 i = 0; i < 32; i++) {
                const i32 wv = static_cast<i32>(
                    static_cast<i8>(static_cast<u8>(blk[2 + i])));
                const i32 xv =
                    static_cast<i32>(static_cast<i8>(static_cast<u8>(xr[b * 32 + i])));
                sum += static_cast<i64>(wv) * static_cast<i64>(xv);
            }
            const f32 sc = dw * dr[b];  // rounded here, as the vector rungs do
            acc = std::fma(static_cast<f32>(sum), sc, acc);
        }
        out[static_cast<i64>(r) * out_stride] = acc;
    }
}

// VNNI. vpdpbusd accumulates u8 x s8 products into int32 in one instruction,
// replacing maddubs + madd. The sign-flipped layout this kernel already uses is
// exactly what it wants: the weight magnitude is the unsigned operand and the
// sign-flipped activation is the signed one.
//
// The reason it matters beyond one saved instruction: the float tail
// (vcvtdq2ps + fmadd) measured 44% of the BW kernel in the ablation, and it
// contends for the same issue port as vpmaddwd. Removing vpmaddwd from the row
// is the point.
//
// NOT MEASURED HERE. This host has no VNNI (checked: avx512vnni absent), so the
// kernel is compiled, dispatched and accuracy-tested only on a host that has
// it. tools/ta_q80_tiers.cpp reports it as absent rather than skipping silently.
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
void dot_q8_0_q8_x8_vnni(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                         i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    const i64 nb = n / 32;
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[8];
    for (i32 r = 0; r < 8; r++) acc[r] = _mm512_setzero_ps();
    for (i64 b = 0; b + 2 <= nb; b += 2, w += 68) {
        const __m128i two = _mm_setr_epi16(
            static_cast<short>(w[0] | (static_cast<u16>(w[1]) << 8)),
            static_cast<short>(w[34] | (static_cast<u16>(w[35]) << 8)), 0, 0, 0, 0,
            0, 0);
        const __m128 d2 = _mm_cvtph_ps(two);
        const f32 d0 = _mm_cvtss_f32(d2);
        const f32 d1 = _mm_cvtss_f32(_mm_shuffle_ps(d2, d2, 0x55));
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 2));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + 36));
        const __m512i wv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __mmask64 neg = _mm512_cmp_epi8_mask(wv, zero, _MM_CMPINT_LT);
        // Unsigned magnitude, -128 included (a masked subtract, not abs_epi8).
        const __m512i mag = _mm512_mask_sub_epi8(wv, neg, zero, wv);
        for (i32 r = 0; r < 8; r++) {
            const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(
                xq + static_cast<i64>(r) * rstride + b * 32));
            const __m512i xb = _mm512_mask_sub_epi8(xv, neg, zero, xv);
            // One instruction where the BW rung needs two. The intrinsic takes
            // the UNSIGNED operand second and the signed one third -- the
            // reverse of the natural reading, and the reverse of what the
            // maddubs pair next door does. Checked against a scalar reference
            // before it was believed.
            const __m512i iv = _mm512_dpbusd_epi32(zero, mag, xb);
            const f32 s0 = d0 * dx[static_cast<i64>(r) * dstride + b];
            const f32 s1 = d1 * dx[static_cast<i64>(r) * dstride + b + 1];
            const __m512 sc = _mm512_castsi512_ps(_mm512_inserti64x4(
                _mm512_castps_si512(
                    _mm512_castps256_ps512(_mm256_broadcast_ss(&s0))),
                _mm256_castps_si256(_mm256_broadcast_ss(&s1)), 1));
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iv), sc, acc[r]);
        }
    }
    for (i32 r = 0; r < 8; r++)
        out[static_cast<i64>(r) * out_stride] = _mm512_reduce_add_ps(acc[r]);
}

// One rung, named explicitly. -1 = scalar, 1 = avx2, 2 = avx512bw, 3 = vnni.
// Falls through to whatever the host can actually run, so asking for an absent
// rung is never a crash. Exposed for tools/ta_q80_tiers.cpp; production code
// goes through dot_q8_0_q8_x8() below.
void dot_q8_0_q8_x8_tier(i32 tier, const u8 *w, const i8 *xq, const f32 *dx,
                         i64 n, i64 rstride, i64 dstride, f32 *out,
                         i64 out_stride) {
    const i64 nb = n / 32;
    const bool even = (nb & 1) == 0;
#if defined(KRK_DOT_AVX512)
    if (tier == 3 && even && cpu_has_avx512bw() && cpu_has_vnni()) {
        dot_q8_0_q8_x8_vnni(w, xq, dx, n, rstride, dstride, out, out_stride);
        return;
    }
    if (tier == 2 && even && cpu_has_avx512bw()) {
        dot_q8_0_q8_x8_bw_impl<true>(w, xq, dx, n, rstride, dstride, out,
                                     out_stride);
        return;
    }
#else
    (void)even;
#endif
#if defined(KRK_DOT_AVX2)
    if (tier == 1 && cpu_has_avx2()) {
        dot_q8_0_q8_x8_avx2(w, xq, dx, n, rstride, dstride, out, out_stride);
        return;
    }
#endif
    dot_q8_0_q8_x8_scalar(w, xq, dx, n, rstride, dstride, out, out_stride);
}

void dot_q8_0_q8_x8(const u8 *w, const i8 *xq, const f32 *dx, i64 n,
                    i64 rstride, i64 dstride, f32 *out, i64 out_stride) {
    const i64 nb = n / 32;
#if defined(KRK_DOT_AVX512)
    static const bool bw = [] {
        const char *e = std::getenv("KRK_CPU_Q80_NOBW");
        return cpu_has_avx512bw() && !(e && e[0] == '1');
    }();
    // The two decode paths are separate instantiations so the fp16 conversion
    // is chosen at dispatch, not branched on inside the block loop. Both are
    // bit-identical (checked across all 65536 half values); KRK_CPU_Q80_NOF16C=1
    // restores the scalar one, which is how the A/B is run.
    static const bool f16c = [] {
        const char *e = std::getenv("KRK_CPU_Q80_NOF16C");
        return !(e && e[0] == '1');
    }();
    // The ladder. Every rung is guarded by the CPU's own feature bits, and the
    // last rung has no guard because scalar needs none -- that is what makes
    // this a fallback chain rather than a fast path with a crash at the end.
    //
    // KRK_CPU_Q80_KERNEL=vnni|bw|avx2|scalar forces a rung, which is how each
    // one is tested on a host that has it and how the rungs that are absent
    // here are exercised elsewhere. A forced rung that the CPU cannot run falls
    // through to the next rather than raising SIGILL.
    static const i32 forced = [] {
        const char *e = std::getenv("KRK_CPU_Q80_KERNEL");
        if (!e || !e[0]) return 0;
        if (std::strcmp(e, "vnni") == 0) return 3;
        if (std::strcmp(e, "bw") == 0) return 2;
        if (std::strcmp(e, "avx2") == 0) return 1;
        if (std::strcmp(e, "scalar") == 0) return -1;
        return 0;  // unknown value: auto
    }();
    auto want = [&](i32 rung) { return forced == 0 || forced == rung; };
    if (bw && (nb & 1) == 0 && want(3) && cpu_has_vnni()) {
        dot_q8_0_q8_x8_vnni(w, xq, dx, n, rstride, dstride, out, out_stride);
        return;
    }
    if (bw && (nb & 1) == 0 && want(2)) {
        if (f16c)
            dot_q8_0_q8_x8_bw_impl<true>(w, xq, dx, n, rstride, dstride, out,
                                         out_stride);
        else
            dot_q8_0_q8_x8_bw_impl<false>(w, xq, dx, n, rstride, dstride, out,
                                          out_stride);
        return;
    }
#else
    (void)nb;
#endif
    // Previously unconditional: on a host without AVX2 this was SIGILL.
#if defined(KRK_DOT_AVX2)
    if (want(1) && cpu_has_avx2()) {
        dot_q8_0_q8_x8_avx2(w, xq, dx, n, rstride, dstride, out, out_stride);
        return;
    }
#endif
    // The floor. No feature test, no target attribute, no way to be unavailable.
    dot_q8_0_q8_x8_scalar(w, xq, dx, n, rstride, dstride, out, out_stride);
}

void dot_q4_K_q8_sx_x8(const u8 *w, const i8 *xq, const f32 *dx, const i32 *sx,
                       i64 n, i64 rstride, i64 dstride, i64 sstride, f32 *out,
                       i64 out_stride) {
    static const bool use_sx = env_dot_x8sx();
    if (!use_sx) {
        dot_q4_K_q8_x8(w, xq, dx, n, rstride, dstride, out, out_stride);
        return;
    }
#if defined(KRK_DOT_AVX512)
    if (cpu_has_vnni()) {
        dot_q4_K_q8_sx_x8_vnni_d(w, xq, dx, sx, n, rstride, dstride, sstride,
                                 out, out_stride);
        return;
    }
#endif
#if defined(KRK_DOT_AVX2)
    dot_q4_K_q8_sx_x8_avx2_d(w, xq, dx, sx, n, rstride, dstride, sstride, out,
                             out_stride);
#else
    (void)w; (void)xq; (void)dx; (void)sx; (void)n; (void)rstride;
    (void)dstride; (void)sstride; (void)out; (void)out_stride;
#endif
}

void dot_q4_K_q8_x8(const u8 *w, const i8 *xq, const f32 *dx, i64 n, i64 rstride,
                    i64 dstride, f32 *out, i64 out_stride) {
#if defined(KRK_DOT_AVX2)
    dot_q4_K_q8_x8_d(w, xq, dx, n, rstride, dstride, out, out_stride);
#else
    (void)w; (void)xq; (void)dx; (void)n; (void)rstride; (void)dstride;
    (void)out; (void)out_stride;
#endif
}

// ---------------------------------------------------------------------------
// The roofline probe.
//
// A "%peak" column is worthless if the peak is a datasheet number, and it is
// actively misleading when the kernel under test is 256-bit and the datasheet
// quotes the 512-bit rate. So the ceiling is MEASURED here, in this process,
// with the same instruction width the kernels use: sixteen independent chains
// (enough to saturate the ports, no loop-carried latency), register-only
// operands, and a MAC count that is exactly what those instructions retire.
#if defined(KRK_DOT_AVX512)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
double probe_int8_vnni_gmacs() {
    const int iters = 20000;
    __m256i a0 = _mm256_set1_epi8(3), a1 = _mm256_set1_epi8(5);
    __m256i x = _mm256_set1_epi8(7);
    __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
    __m256i s2 = _mm256_setzero_si256(), s3 = _mm256_setzero_si256();
    __m256i s4 = _mm256_setzero_si256(), s5 = _mm256_setzero_si256();
    __m256i s6 = _mm256_setzero_si256(), s7 = _mm256_setzero_si256();
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        s0 = _mm256_dpbusd_epi32(s0, a0, x);
        s1 = _mm256_dpbusd_epi32(s1, a1, x);
        s2 = _mm256_dpbusd_epi32(s2, a0, x);
        s3 = _mm256_dpbusd_epi32(s3, a1, x);
        s4 = _mm256_dpbusd_epi32(s4, a0, x);
        s5 = _mm256_dpbusd_epi32(s5, a1, x);
        s6 = _mm256_dpbusd_epi32(s6, a0, x);
        s7 = _mm256_dpbusd_epi32(s7, a1, x);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    s0 = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
    s4 = _mm256_add_epi32(_mm256_add_epi32(s4, s5), _mm256_add_epi32(s6, s7));
    s0 = _mm256_add_epi32(s0, s4);
    // Keep the loop alive; the value is irrelevant.
    g_probe_sink.store(_mm256_extract_epi32(s0, 0), std::memory_order_relaxed);
    const double us = static_cast<double>(t1.tv_sec - t0.tv_sec) * 1e6 +
                      static_cast<double>(t1.tv_nsec - t0.tv_nsec) / 1e3;
    // 8 dpbusd per iteration, 32 MACs each, 256-bit.
    const double macs = static_cast<double>(iters) * 8.0 * 32.0;
    return us > 0 ? macs / us / 1e3 : 0.0;
}

#else
double probe_int8_vnni_gmacs() { return 0.0; }
#endif

#if defined(KRK_DOT_AVX2)
__attribute__((target("avx2,fma"))) double probe_f32_avx2_gmacs() {
    const int iters = 20000;
    __m256 p = _mm256_set1_ps(1.0001f), q = _mm256_set1_ps(0.9999f);
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    __m256 s4 = _mm256_setzero_ps(), s5 = _mm256_setzero_ps();
    __m256 s6 = _mm256_setzero_ps(), s7 = _mm256_setzero_ps();
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        s0 = _mm256_fmadd_ps(p, q, s0);
        s1 = _mm256_fmadd_ps(p, q, s1);
        s2 = _mm256_fmadd_ps(p, q, s2);
        s3 = _mm256_fmadd_ps(p, q, s3);
        s4 = _mm256_fmadd_ps(p, q, s4);
        s5 = _mm256_fmadd_ps(p, q, s5);
        s6 = _mm256_fmadd_ps(p, q, s6);
        s7 = _mm256_fmadd_ps(p, q, s7);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    s4 = _mm256_add_ps(_mm256_add_ps(s4, s5), _mm256_add_ps(s6, s7));
    s0 = _mm256_add_ps(s0, s4);
    g_probe_sink_f.store(_mm256_cvtss_f32(s0), std::memory_order_relaxed);
    const double us = static_cast<double>(t1.tv_sec - t0.tv_sec) * 1e6 +
                      static_cast<double>(t1.tv_nsec - t0.tv_nsec) / 1e3;
    const double macs = static_cast<double>(iters) * 8.0 * 8.0;
    return us > 0 ? macs / us / 1e3 : 0.0;
}

#else
double probe_f32_avx2_gmacs() { return 0.0; }
#endif

// The int8 ceiling, MEASURED rather than assumed.
//
// isa_peak_gmacs() used to answer int8 with `f32 peak x 4`, on the argument that
// maddubs retires 16 MACs per instruction where an fma retires 8. That argument
// assumes the wide form issues at the same rate as the narrow one, and the Q8_0
// GEMM disproved it: widening that kernel from 256 to 512 bits made it 1.07x
// faster, not 2x, which is what a part that issues one 512-bit op per cycle and
// two 256-bit ops per cycle looks like. So the multiplier is not 4, and every
// %peak column computed from it was dividing by a number nobody had measured.
//
// This probe runs the instruction mix the int8 GEMM actually issues per 64
// values -- maddubs, madd, accumulate -- with eight independent chains so it
// measures throughput and not latency. Both widths are reported: if they agree,
// the core is not width-limited and the 256-bit path is the one to build.
#if defined(KRK_DOT_AVX2)
__attribute__((target("avx2,fma"))) double probe_int8_avx2_gmacs() {
    const int iters = 20000;
    // `a` is bumped every iteration on purpose. With both operands loop
    // invariant the whole maddubs is a constant, the compiler hoists it out,
    // and the loop measures integer ADD throughput labelled as a dot-product
    // ceiling -- which is how the first version of this probe reported 301
    // GMAC/s that no kernel could reach.
    __m256i a = _mm256_set1_epi8(3);
    // Distinct multiplier per chain -- see the AVX-512 probe below for why.
    const __m256i b0 = _mm256_set1_epi8(2), b1 = _mm256_set1_epi8(3);
    const __m256i b2 = _mm256_set1_epi8(4), b3 = _mm256_set1_epi8(5);
    const __m256i b4 = _mm256_set1_epi8(6), b5 = _mm256_set1_epi8(7);
    const __m256i b6 = _mm256_set1_epi8(8), b7 = _mm256_set1_epi8(9);
    const __m256i bump = _mm256_set1_epi8(1);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
    __m256i s2 = _mm256_setzero_si256(), s3 = _mm256_setzero_si256();
    __m256i s4 = _mm256_setzero_si256(), s5 = _mm256_setzero_si256();
    __m256i s6 = _mm256_setzero_si256(), s7 = _mm256_setzero_si256();
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        a = _mm256_add_epi8(a, bump);
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b0), ones));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b1), ones));
        s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b2), ones));
        s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b3), ones));
        s4 = _mm256_add_epi32(s4, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b4), ones));
        s5 = _mm256_add_epi32(s5, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b5), ones));
        s6 = _mm256_add_epi32(s6, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b6), ones));
        s7 = _mm256_add_epi32(s7, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b7), ones));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    s0 = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
    s4 = _mm256_add_epi32(_mm256_add_epi32(s4, s5), _mm256_add_epi32(s6, s7));
    s0 = _mm256_add_epi32(s0, s4);
    g_probe_sink_i.store(_mm256_extract_epi32(s0, 0), std::memory_order_relaxed);
    const double us = static_cast<double>(t1.tv_sec - t0.tv_sec) * 1e6 +
                      static_cast<double>(t1.tv_nsec - t0.tv_nsec) / 1e3;
    // maddubs retires 32 byte-products per instruction.
    const double macs = static_cast<double>(iters) * 8.0 * 32.0;
    return us > 0 ? macs / us / 1e3 : 0.0;
}
#else
double probe_int8_avx2_gmacs() { return 0.0; }
#endif

#if defined(KRK_DOT_AVX512)
__attribute__((target("avx512f,avx512bw"))) double probe_int8_bw_gmacs() {
    const int iters = 20000;
    __m512i a = _mm512_set1_epi8(3);
    // One multiplier per chain. With a shared `b` all eight maddubs are the
    // same expression, and with both operands loop-invariant the compiler is
    // entitled to compute one of them outside the loop -- which it does: the
    // disassembly of the previous version had a single vpmaddubsw and a single
    // vpaddd in the body while crediting eight dots, overstating the ceiling by
    // roughly 8x. Every %peak column divided by that number.
    const __m512i b0 = _mm512_set1_epi8(2), b1 = _mm512_set1_epi8(3);
    const __m512i b2 = _mm512_set1_epi8(4), b3 = _mm512_set1_epi8(5);
    const __m512i b4 = _mm512_set1_epi8(6), b5 = _mm512_set1_epi8(7);
    const __m512i b6 = _mm512_set1_epi8(8), b7 = _mm512_set1_epi8(9);
    const __m512i bump = _mm512_set1_epi8(1);
    const __m512i ones = _mm512_set1_epi16(1);
    __m512i s0 = _mm512_setzero_si512(), s1 = _mm512_setzero_si512();
    __m512i s2 = _mm512_setzero_si512(), s3 = _mm512_setzero_si512();
    __m512i s4 = _mm512_setzero_si512(), s5 = _mm512_setzero_si512();
    __m512i s6 = _mm512_setzero_si512(), s7 = _mm512_setzero_si512();
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        a = _mm512_add_epi8(a, bump);
        s0 = _mm512_add_epi32(s0, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b0), ones));
        s1 = _mm512_add_epi32(s1, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b1), ones));
        s2 = _mm512_add_epi32(s2, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b2), ones));
        s3 = _mm512_add_epi32(s3, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b3), ones));
        s4 = _mm512_add_epi32(s4, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b4), ones));
        s5 = _mm512_add_epi32(s5, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b5), ones));
        s6 = _mm512_add_epi32(s6, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b6), ones));
        s7 = _mm512_add_epi32(s7, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b7), ones));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    s0 = _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3));
    s4 = _mm512_add_epi32(_mm512_add_epi32(s4, s5), _mm512_add_epi32(s6, s7));
    s0 = _mm512_add_epi32(s0, s4);
    g_probe_sink_i.store(_mm512_reduce_add_epi32(s0), std::memory_order_relaxed);
    const double us = static_cast<double>(t1.tv_sec - t0.tv_sec) * 1e6 +
                      static_cast<double>(t1.tv_nsec - t0.tv_nsec) / 1e3;
    const double macs = static_cast<double>(iters) * 8.0 * 64.0;
    return us > 0 ? macs / us / 1e3 : 0.0;
}
#else
double probe_int8_bw_gmacs() { return 0.0; }
#endif

// Reported by the CPU backend so the trace's %peak column is against this
// host, at this kernel width, with every core it has.
double isa_peak_gmacs(bool int8) {
    // Off unless something is going to print it: a probe is a few milliseconds
    // of the user's run and a normal run must not pay for it.
    {
        const char *e = std::getenv("KRK_KPROF");
        if (!e || e[0] == '\0' || e[0] == '0') return 0.0;
    }
    static const double i8 = [] {
        double per = 0.0;
#if defined(KRK_DOT_AVX512)
        if (cpu_has_vnni()) per = probe_int8_vnni_gmacs();
        // The widest byte-dot form the host has, measured. Falling back to
        // f32*4 below was the old guess and it is not what this core does.
        if (per <= 0.0) per = probe_int8_bw_gmacs();
#endif
        if (per <= 0.0) per = probe_int8_avx2_gmacs();
        return per;
    }();
    static const double f32 = [] { return probe_f32_avx2_gmacs(); }();
    const double cores = static_cast<double>(
        std::max(1u, std::thread::hardware_concurrency()));
    return (int8 ? i8 : f32) * cores;
}

void set_vec_dot_int8(bool on) {
    g_dot_int8.store(on, std::memory_order_relaxed);
    g_dot_int8_set.store(true, std::memory_order_relaxed);
}

bool vec_dot_int8() {
    if (!g_dot_int8_set.load(std::memory_order_relaxed)) {
        g_dot_int8.store(env_dot_int8(), std::memory_order_relaxed);
        g_dot_int8_set.store(true, std::memory_order_relaxed);
    }
#if defined(KRK_DOT_AVX2)
    return g_dot_int8.load(std::memory_order_relaxed) && cpu_has_avx2();
#else
    return false;
#endif
}

void set_vec_dot_avx2(bool on) {
    g_dot_avx2.store(on, std::memory_order_relaxed);
    g_dot_avx2_set.store(true, std::memory_order_relaxed);
}

bool vec_dot_avx2() {
    if (!g_dot_avx2_set.load(std::memory_order_relaxed)) {
        g_dot_avx2.store(env_dot_avx2(), std::memory_order_relaxed);
        g_dot_avx2_set.store(true, std::memory_order_relaxed);
    }
    // The flag AND the CPU: a caller that logs this gets the truth on a build
    // where the kernels exist but the host cannot run them.
#if defined(KRK_DOT_AVX2)
    return g_dot_avx2.load(std::memory_order_relaxed) && cpu_has_avx2();
#else
    return false;
#endif
}

namespace {
std::atomic<bool> g_dot_fast{false};
std::atomic<bool> g_dot_fast_set{false};

// The same policy: a host whose vectors are wide enough runs the four-lane
// order unless it is told not to. The order is the numerics tier, so
// `KRK_CPU_DOT_FAST=0` is the way back to the strictly serial reference.
bool env_dot_fast() {
    const char *e = std::getenv("KRK_CPU_DOT_FAST");
    if (!e || e[0] == '\0') return dot_avx2_default() || cpu_has_avx2();
    return e[0] != '0';
}
} // namespace

void set_vec_dot_fast(bool on) {
    g_dot_fast.store(on, std::memory_order_relaxed);
    g_dot_fast_set.store(true, std::memory_order_relaxed);
}

bool vec_dot_fast() {
    if (!g_dot_fast_set.load(std::memory_order_relaxed)) {
        // One racy init: every thread that gets here writes the same value.
        g_dot_fast.store(env_dot_fast(), std::memory_order_relaxed);
        g_dot_fast_set.store(true, std::memory_order_relaxed);
    }
    return g_dot_fast.load(std::memory_order_relaxed);
}

// ---- the batched prefill kernel -------------------------------------------
//
// vec_dot() is a GEMV inner product: it decodes the weight row as it goes, so a
// GEMM that calls it once per output element decodes every weight row once per
// ROW of the batch. That is the whole of the prefill problem below -- measured
// before this existed, on the 2 GiB q8_0 MoE at --cpu with KRK_CPU_DOT_AVX2=1,
// cost per prefilled token was FLAT at 34.1, 29.1, 27.8, 28.3 ms across prompts
// of 58, 149, 331 and 695 tokens. A batched GEMM is supposed to pay the weight
// read once and reuse it for the whole batch, so prefill was running at GEMV
// speed and its throughput could not improve with batch size.
//
// This decodes each weight block ONCE into registers and FMAs it against
// `kTileRows` activation rows before moving on. Weight decode is amortized
// kTileRows times instead of once, and there is no f32 scratch buffer: a
// widening pass was tried first and made prefill 1.9x SLOWER, because it wrote
// 4x the weight bytes to memory and replaced a fused vector decode with a
// scalar one.
//
// Bit-identity is exact, not approximate. For one (weight row, activation row)
// pair this performs the same operations in the same order as the fused kernel
// it generalises -- per 8-element step, fma8() adds weight element k to
// accumulator lane k mod 4, walking k upward -- so its result for that pair is
// bit-identical to dot_q8_0_avx2/dot_q4_0_avx2, which are in turn bit-identical
// to the four-lane arm. test_gemm_batched_bit_identical pins it against
// vec_dot() on the same rows.

#if defined(KRK_DOT_AVX2)
namespace {

constexpr i64 kTileRows = 8;

// q8_0: d (f16) + 32 int8 per 34-byte block.
__attribute__((target("avx2,fma"))) void dot_q8_0_tiled(
    const u8 *w, const f32 *x, i64 n, i64 rows, f32 *out, i64 stride) {
    const i64 nb = n / 32;
    for (i64 r0 = 0; r0 + kTileRows <= rows; r0 += kTileRows) {
        __m128 acc[kTileRows];
        for (i64 j = 0; j < kTileRows; j++) acc[j] = _mm_setzero_ps();
        const f32 *xr[kTileRows];
        for (i64 j = 0; j < kTileRows; j++) xr[j] = x + (r0 + j) * n;
        const u8 *p = w;
        for (i64 b = 0; b < nb; b++, p += 34) {
            const f32 d = fp16_to_fp32(
                static_cast<u16>(p[0] | (static_cast<u16>(p[1]) << 8)));
            const __m256 dv = _mm256_set1_ps(d);
            for (int k = 0; k < 32; k += 8) {
                const __m128i q8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(p + 2 + k));
                const __m256 wv = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)), dv);
                for (i64 j = 0; j < kTileRows; j++)
                    fma8(acc[j], wv, xr[j] + b * 32 + k);
            }
        }
        for (i64 j = 0; j < kTileRows; j++) {
            alignas(16) f32 a[4];
            _mm_store_ps(a, acc[j]);
            out[(r0 + j) * stride] = (a[0] + a[1]) + (a[2] + a[3]);
        }
    }
}

// q4_0: d (f16) + 16 bytes of nibbles, 18-byte block. Low nibble is value k,
// high nibble is value k+16, both biased by 8.
__attribute__((target("avx2,fma"))) void dot_q4_0_tiled(
    const u8 *w, const f32 *x, i64 n, i64 rows, f32 *out, i64 stride) {
    const i64 nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i off = _mm256_set1_epi32(8);
    for (i64 r0 = 0; r0 + kTileRows <= rows; r0 += kTileRows) {
        __m128 acc[kTileRows];
        for (i64 j = 0; j < kTileRows; j++) acc[j] = _mm_setzero_ps();
        const f32 *xr[kTileRows];
        for (i64 j = 0; j < kTileRows; j++) xr[j] = x + (r0 + j) * n;
        const u8 *p = w;
        for (i64 b = 0; b < nb; b++, p += 18) {
            const f32 d = fp16_to_fp32(
                static_cast<u16>(p[0] | (static_cast<u16>(p[1]) << 8)));
            const __m256 dv = _mm256_set1_ps(d);
            const __m128i raw = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(p + 2));
            const __m128i lo = _mm_and_si128(raw, m4);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            const __m256 w0 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), off)),
                dv);
            const __m256 w1 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)), off)),
                dv);
            const __m256 w2 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), off)),
                dv);
            const __m256 w3 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)), off)),
                dv);
            for (i64 j = 0; j < kTileRows; j++) {
                const f32 *xv = xr[j] + b * 32;
                fma8(acc[j], w0, xv);
                fma8(acc[j], w1, xv + 8);
                fma8(acc[j], w2, xv + 16);
                fma8(acc[j], w3, xv + 24);
            }
        }
        for (i64 j = 0; j < kTileRows; j++) {
            alignas(16) f32 a[4];
            _mm_store_ps(a, acc[j]);
            out[(r0 + j) * stride] = (a[0] + a[1]) + (a[2] + a[3]);
        }
    }
}

} // namespace
#endif

// The unquantized formats, tiled. They have no block decode to hoist, but they
// have the same problem for a different reason: vec_dot() widens the weight
// element-by-element INSIDE the row loop, so every row re-converts the whole
// matrix. On a 1 GiB f16 dense model that showed up as prefill 10.5 tok/s
// against a decode of ~7.5 -- a 1.4x ratio, i.e. no batch amortization at all.
// Converting each 4-element group once and reusing it across the tile fixes it
// with the same arithmetic F16C already does, in the same order.
#if defined(KRK_DOT_F16C)
namespace {

template <bool IsF16>
__attribute__((target("f16c,fma"))) void dot_wide_tiled(
    const void *w, const f32 *x, i64 n, i64 rows, f32 *out, i64 stride) {
    constexpr i64 kT = 8;
    for (i64 r0 = 0; r0 + kT <= rows; r0 += kT) {
        __m128 acc[kT];
        for (i64 j = 0; j < kT; j++) acc[j] = _mm_setzero_ps();
        i64 i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 wv;
            if constexpr (IsF16)
                wv = _mm_cvtph_ps(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(
                        static_cast<const u16 *>(w) + i)));
            else
                wv = _mm_loadu_ps(static_cast<const f32 *>(w) + i);
            for (i64 j = 0; j < kT; j++)
                acc[j] = _mm_fmadd_ps(wv, _mm_loadu_ps(x + (r0 + j) * n + i),
                                      acc[j]);
        }
        for (i64 j = 0; j < kT; j++) {
            alignas(16) f32 a[4];
            _mm_store_ps(a, acc[j]);
            f32 sum = (a[0] + a[1]) + (a[2] + a[3]);
            const f32 *xr = x + (r0 + j) * n;
            for (i64 k = i; k < n; k++)
                sum += (IsF16 ? fp16_to_fp32(static_cast<const u16 *>(w)[k])
                              : static_cast<const f32 *>(w)[k]) *
                       xr[k];
            out[(r0 + j) * stride] = sum;
        }
    }
}

} // namespace
#endif

// Every block format without a hand-written kernel, tiled the same way: decode
// a block ONCE and multiply it against kTileRows activation rows. This is the
// path that carries K-quants (Q4_K/Q5_K/Q6_K), iq-quants and the rest of a real
// collection, and it is the one that matters for anything not converted to
// q8_0/q4_0 -- the fixture used for the headline measurement is q8_0, but the
// sweep in docs/MODEL-STATUS.md is mostly Q4_K_M.
//
// Bit-identity comes from doing what vec_dot() does, per row, in its order.
// vec_dot() walks the row block by block and has two shapes: a single in-order
// chain (the oracle) and four accumulators folded (a0+a1)+(a2+a3) with a `tail`
// for any run shorter than four. Both are reproduced here, and the `tail` is
// accumulated ACROSS blocks exactly as there -- moving it inside the block
// would change the summation order on any format whose block size is not a
// multiple of four.
namespace {

// Decode each block once, then run it against every row of the tile. The inner
// loop keeps kTileRows independent accumulators, which is what turns the weight
// read from "once per row" into "once per tile".
template <bool FourLane>
void dot_generic_tiled(DType t, const u8 *w, const f32 *x, i64 n, i64 o_begin,
                       i64 o_end, i64 rows, f32 *out, i64 out_stride) {
    const int bs = dtype_block_size(t);
    const int bb = dtype_block_bytes(t);
    const size_t rb = dtype_row_bytes(t, n);
    constexpr i64 kT = 8;
    f32 buf[256];
    for (i64 o = o_begin; o < o_end; o++) {
        const u8 *p0 = w + static_cast<size_t>(o) * rb;
        i64 r0 = 0;
        for (; r0 + kT <= rows; r0 += kT) {
            f32 a0[kT] = {}, a1[kT] = {}, a2[kT] = {}, a3[kT] = {}, tl[kT] = {};
            if constexpr (!FourLane) {
                f32 one[kT] = {};
                i64 i = 0;
                const u8 *p = p0;
                while (i < n) {
                    const int cnt = static_cast<int>((n - i) < bs ? (n - i) : bs);
                    dequant_block(t, p, buf);
                    for (i64 j = 0; j < kT; j++) {
                        const f32 *xr = x + (r0 + j) * n + i;
                        f32 s = one[j];
                        for (int k = 0; k < cnt; k++) s += buf[k] * xr[k];
                        one[j] = s;
                    }
                    i += cnt;
                    p += bb;
                }
                for (i64 j = 0; j < kT; j++) out[(r0 + j) * out_stride + o] = one[j];
            } else {
                // The four-lane arm, in the shape vec_dot() itself uses. On an
                // F16C host that shape is a FUSED multiply-add, and a separate
                // multiply and add rounds differently -- using the scalar form
                // here made Q4_K disagree with vec_dot() in the last few ulp,
                // which is what this test caught.
#if defined(KRK_DOT_F16C)
                __m128 acc[kT];
                for (i64 j = 0; j < kT; j++) acc[j] = _mm_setzero_ps();
#endif
                i64 i = 0;
                const u8 *p = p0;
                while (i < n) {
                    const int cnt = static_cast<int>((n - i) < bs ? (n - i) : bs);
                    dequant_block(t, p, buf);
                    for (i64 j = 0; j < kT; j++) {
                        const f32 *xr = x + (r0 + j) * n + i;
                        int k = 0;
#if defined(KRK_DOT_F16C)
                        for (; k + 4 <= cnt; k += 4)
                            acc[j] = _mm_fmadd_ps(_mm_loadu_ps(buf + k),
                                                  _mm_loadu_ps(xr + k), acc[j]);
#else
                        for (; k + 4 <= cnt; k += 4) {
                            a0[j] += buf[k] * xr[k];
                            a1[j] += buf[k + 1] * xr[k + 1];
                            a2[j] += buf[k + 2] * xr[k + 2];
                            a3[j] += buf[k + 3] * xr[k + 3];
                        }
#endif
                        for (; k < cnt; k++) tl[j] += buf[k] * xr[k];
                    }
                    i += cnt;
                    p += bb;
                }
                for (i64 j = 0; j < kT; j++) {
#if defined(KRK_DOT_F16C)
                    alignas(16) f32 a[4];
                    _mm_store_ps(a, acc[j]);
                    out[(r0 + j) * out_stride + o] =
                        ((a[0] + a[1]) + (a[2] + a[3])) + tl[j];
#else
                    out[(r0 + j) * out_stride + o] =
                        ((a0[j] + a1[j]) + (a2[j] + a3[j])) + tl[j];
#endif
                }
            }
        }
        // Leftover rows: the per-element oracle, same bits.
        for (i64 r = rows - (rows % kT); r < rows; r++)
            out[r * out_stride + o] = vec_dot(t, p0, x + r * n, n);
    }
}

} // namespace

void vec_dot_tiled(DType t, const void *w, const f32 *x, i64 n_in, i64 o_begin,
                   i64 o_end, i64 rows, f32 *out, i64 out_stride) {
    const size_t rb = dtype_row_bytes(t, n_in);
    const u8 *wb = static_cast<const u8 *>(w);
#if defined(KRK_DOT_AVX2)
    if ((t == DType::Q8_0 || t == DType::Q4_0) && n_in > 0 && (n_in % 32) == 0 &&
        rows > 0 && vec_dot_avx2()) {
        for (i64 o = o_begin; o < o_end; o++) {
            const u8 *wr = wb + static_cast<size_t>(o) * rb;
            if (t == DType::Q8_0)
                dot_q8_0_tiled(wr, x, n_in, rows, out + o, out_stride);
            else
                dot_q4_0_tiled(wr, x, n_in, rows, out + o, out_stride);
            // Rows past the last whole tile: the per-element path, same bits.
            for (i64 r = rows - (rows % kTileRows); r < rows; r++)
                out[r * out_stride + o] = vec_dot(t, wr, x + r * n_in, n_in);
        }
        return;
    }
#endif
#if defined(KRK_DOT_F16C)
    // f16 and f32: the widening arm, tiled. bf16 and i32 keep the per-element
    // loop -- they are rare enough in a collection that they are not worth a
    // fourth kernel, and they are correct either way.
    if ((t == DType::F16 || t == DType::F32) && rows > 0 && n_in > 0) {
        for (i64 o = o_begin; o < o_end; o++) {
            const u8 *wr = wb + static_cast<size_t>(o) * rb;
            if (t == DType::F16)
                dot_wide_tiled<true>(wr, x, n_in, rows, out + o, out_stride);
            else
                dot_wide_tiled<false>(wr, x, n_in, rows, out + o, out_stride);
            for (i64 r = rows - (rows % 8); r < rows; r++)
                out[r * out_stride + o] = vec_dot(t, wr, x + r * n_in, n_in);
        }
        return;
    }
#endif
    // Every other block format: the generic tile, which decodes each block once
    // and reuses it across rows.
    if (t != DType::F16 && t != DType::F32 && t != DType::BF16 &&
        t != DType::I32 && rows > 0 && n_in > 0) {
        if (vec_dot_fast() || vec_dot_avx2())
            dot_generic_tiled<true>(t, wb, x, n_in, o_begin, o_end, rows, out,
                                    out_stride);
        else
            dot_generic_tiled<false>(t, wb, x, n_in, o_begin, o_end, rows, out,
                                     out_stride);
        return;
    }
    // Every other format, and every host without AVX2: the per-element path,
    // which is what the fused kernels are also validated against.
    for (i64 o = o_begin; o < o_end; o++) {
        const u8 *wr = wb + static_cast<size_t>(o) * rb;
        for (i64 r = 0; r < rows; r++)
            out[r * out_stride + o] = vec_dot(t, wr, x + r * n_in, n_in);
    }
}

f32 vec_dot(DType t, const void *src, const f32 *x, i64 n) {
    // The unquantized formats have no block structure, so widen them directly
    // rather than routing through the block loop (which would look for a
    // 1-element "block" decoder that does not exist).
    //
    // F32 and F16 are evaluated four lanes at a time where the ISA has F16C:
    // lane k of the vector IS accumulator k of the scalar loop below, fed the
    // same elements in the same order, and the final combination is still
    // (a0 + a1) + (a2 + a3). That is what makes the fast path safe to have in
    // an oracle -- it is the same arithmetic, not a rearranged version of it,
    // so every golden hash in the suite stays valid.
    //
    // Why it matters here: the scalar form below called fp16_to_fp32() once per
    // weight element, and that is an out-of-line function with a subnormal
    // normalisation loop in it. One core / two SMT threads at 2.6 GHz measured
    // 0.8 GMAC/s on a 1 GiB f16 model -- 583 ms per prefill token, identical at
    // rows=1..64 because the cost is per weight element and not per row. The
    // same machine does ~4 GMAC/s through this path (see BENCH-CPU-DOT.md).
#if defined(KRK_DOT_F16C)
    if (t == DType::F32) {
        const f32 *w = static_cast<const f32 *>(src);
        __m128 acc = _mm_setzero_ps();
        i64 i = 0;
        for (; i + 4 <= n; i += 4)
            acc = _mm_fmadd_ps(_mm_loadu_ps(w + i), _mm_loadu_ps(x + i), acc);
        alignas(16) f32 a[4];
        _mm_store_ps(a, acc);
        f32 s = (a[0] + a[1]) + (a[2] + a[3]);
        for (; i < n; i++) s += w[i] * x[i];
        return s;
    }
    if (t == DType::F16) {
        const u16 *h = static_cast<const u16 *>(src);
        __m128 acc = _mm_setzero_ps();
        i64 i = 0;
        for (; i + 4 <= n; i += 4) {
            const __m128i h4 =
                _mm_loadl_epi64(reinterpret_cast<const __m128i *>(h + i));
            acc = _mm_fmadd_ps(_mm_cvtph_ps(h4), _mm_loadu_ps(x + i), acc);
        }
        alignas(16) f32 a[4];
        _mm_store_ps(a, acc);
        f32 s = (a[0] + a[1]) + (a[2] + a[3]);
        for (; i < n; i++) s += fp16_to_fp32(h[i]) * x[i];
        return s;
    }
#endif
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
    if (t == DType::I32) {
        const i32 *w = static_cast<const i32 *>(src);
        f32 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        i64 i = 0;
        for (; i + 4 <= n; i += 4) {
            acc0 += static_cast<f32>(w[i]) * x[i];
            acc1 += static_cast<f32>(w[i + 1]) * x[i + 1];
            acc2 += static_cast<f32>(w[i + 2]) * x[i + 2];
            acc3 += static_cast<f32>(w[i + 3]) * x[i + 3];
        }
        f32 s = (acc0 + acc1) + (acc2 + acc3);
        for (; i < n; i++) s += static_cast<f32>(w[i]) * x[i];
        return s;
    }
    const int bs = dtype_block_size(t);
    const int bb = dtype_block_bytes(t);
    f32 buf[256];
#if defined(KRK_DOT_AVX2)
    // The fused arm, for the two formats it has kernels for. Everything else --
    // and any run where the host lacks AVX2 -- falls through to the four-lane
    // loop below, which is the same arithmetic, so the answer does not depend
    // on which of the two ran.
    if (n > 0 && (n % 32) == 0 && vec_dot_avx2()) {
        if (t == DType::Q8_0)
            return dot_q8_0_avx2(static_cast<const u8 *>(src), x, n / 32);
        if (t == DType::Q4_0)
            return dot_q4_0_avx2(static_cast<const u8 *>(src), x, n / 32);
    }
#endif
    if (vec_dot_fast() || vec_dot_avx2()) {
        // Four lanes over the widened block. dequant_block is unchanged, so the
        // VALUES are identical to the loop below; only the summation order
        // differs. See the contract on set_vec_dot_fast in krk/quant.hpp.
        f32 tail = 0.0f;
        const u8 *p = static_cast<const u8 *>(src);
        i64 i = 0;
#if defined(KRK_DOT_F16C)
        __m128 acc = _mm_setzero_ps();
        while (i < n) {
            const int cnt = static_cast<int>((n - i) < bs ? (n - i) : bs);
            dequant_block(t, p, buf);
            int k = 0;
            for (; k + 4 <= cnt; k += 4)
                acc = _mm_fmadd_ps(_mm_loadu_ps(buf + k),
                                   _mm_loadu_ps(x + i + k), acc);
            for (; k < cnt; k++) tail += buf[k] * x[i + k];
            i += cnt;
            p += bb;
        }
        alignas(16) f32 a[4];
        _mm_store_ps(a, acc);
        return ((a[0] + a[1]) + (a[2] + a[3])) + tail;
#else
        f32 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        while (i < n) {
            const int cnt = static_cast<int>((n - i) < bs ? (n - i) : bs);
            dequant_block(t, p, buf);
            int k = 0;
            for (; k + 4 <= cnt; k += 4) {
                acc0 += buf[k] * x[i + k];
                acc1 += buf[k + 1] * x[i + k + 1];
                acc2 += buf[k + 2] * x[i + k + 2];
                acc3 += buf[k + 3] * x[i + k + 3];
            }
            for (; k < cnt; k++) tail += buf[k] * x[i + k];
            i += cnt;
            p += bb;
        }
        return ((acc0 + acc1) + (acc2 + acc3)) + tail;
#endif
    }
    f32 sum = 0;
    const u8 *p = static_cast<const u8 *>(src);
    i64 i = 0;
    while (i < n) {
        const int cnt = static_cast<int>((n - i) < bs ? (n - i) : bs);
        dequant_block(t, p, buf);
        for (int k = 0; k < cnt; k++) sum += buf[k] * x[i + k];
        i += cnt;
        p += bb;
    }
    return sum;
}

} // namespace krk
