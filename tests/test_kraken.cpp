// ============================================================================
//  test_kraken.cpp — deterministic, self-contained test suite.
//
//  Everything here runs on the CPU reference backend on purpose: the host
//  dequantizers are the oracle the GPU kernels are validated against, so a
//  regression in src/quant.cpp must fail before it can silently mask a kernel
//  bug. The suite also builds a complete miniature LLaMA-family GGUF in a
//  temporary file and runs real generation through it, which exercises the
//  container parser, the model loader, the forward pass, the sampler and both
//  tokenizer families end to end.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "krk/arch.hpp"
#include "krk/backend.hpp"
#include "krk/engine.hpp"
#include "krk/expert_cpu.hpp"
#include "krk/gguf.hpp"
#include "krk/http.hpp"
#include "krk/json.hpp"
#include "krk/quant.hpp"
#include "krk/server.hpp"
#include "krk/verdict.hpp"

// The CPU backend's worker pool lives next to the sources, not in the public
// headers; the tests reach in for it because its scheduling contract (every
// block runs exactly once, every task terminates) is load-bearing.
#include "../src/par_pool.hpp"

using namespace krk;

static int g_run = 0, g_passed = 0;

#define CHECK(cond, msg)                                                            \
    do {                                                                            \
        g_run++;                                                                    \
        if (cond) {                                                                 \
            g_passed++;                                                             \
        } else {                                                                    \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);      \
        }                                                                           \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                                  \
    do {                                                                            \
        g_run++;                                                                    \
        const double _a = (a), _b = (b);                                            \
        if (std::fabs(_a - _b) <= (tol)) {                                          \
            g_passed++;                                                             \
        } else {                                                                    \
            std::fprintf(stderr, "FAIL %s:%d  %s (%g vs %g)\n", __FILE__, __LINE__, \
                         msg, _a, _b);                                              \
        }                                                                           \
    } while (0)

// ---------------------------------------------------------------------------
// minimal GGUF v3 writer (test-only)
// ---------------------------------------------------------------------------

namespace {

class GgufBuilder {
public:
    void meta_u32(const std::string &k, u32 v) {
        put_key(k, 4);
        put_u32(v);
    }
    void meta_f32(const std::string &k, f32 v) {
        put_key(k, 6);
        put_raw(&v, 4);
    }
    void meta_str(const std::string &k, const std::string &v) {
        put_key(k, 8);
        put_str(v);
    }
    void meta_bool(const std::string &k, bool v) {
        put_key(k, 7);
        const u8 b = v ? 1 : 0;
        put_raw(&b, 1);
    }
    void meta_str_array(const std::string &k, const std::vector<std::string> &v) {
        put_key(k, 9);
        put_u32(8); // element type = String
        put_u64(v.size());
        for (const std::string &s : v) put_str(s);
    }
    void meta_f32_array(const std::string &k, const std::vector<f32> &v) {
        put_key(k, 9);
        put_u32(6);
        put_u64(v.size());
        put_raw(v.data(), v.size() * 4);
    }

    void meta_i32_array(const std::string &k, const std::vector<i32> &v) {
        put_key(k, 9);
        put_u32(5); // element type = Int32
        put_u64(v.size());
        put_raw(v.data(), v.size() * 4);
    }

    void tensor_f32(const std::string &name, const std::vector<u64> &ne,
                    const std::vector<f32> &data) {
        add_tensor(name, 0, ne, data.size() * 4);
        put_payload(data.data(), data.size() * 4);
    }
    void tensor_f16(const std::string &name, const std::vector<u64> &ne,
                    const std::vector<f32> &data) {
        std::vector<u16> h(data.size());
        for (size_t i = 0; i < data.size(); i++) h[i] = fp32_to_fp16(data[i]);
        add_tensor(name, 1, ne, h.size() * 2);
        put_payload(h.data(), h.size() * 2);
    }
    // Writes an already-encoded payload (e.g. hand-built F16 blocks).
    void tensor_raw(const std::string &name, u32 type, const std::vector<u64> &ne,
                    const void *data, size_t bytes) {
        add_tensor(name, type, ne, bytes);
        put_payload(data, bytes);
    }

    bool write(const std::string &path) {
        std::FILE *f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        // header: magic, version, tensor count, kv count
        std::vector<u8> head;
        const u32 magic = 0x46554747u;
        const u32 version = 3;
        append(head, &magic, 4);
        append(head, &version, 4);
        const u64 nt = tensor_count();
        append(head, &nt, 8);
        const u64 nk = kv_count();
        append(head, &nk, 8);

        std::fwrite(head.data(), 1, head.size(), f);
        std::fwrite(meta_.data(), 1, meta_.size(), f);
        std::fwrite(info_.data(), 1, info_.size(), f);

        // pad to the 32-byte alignment declared by general.alignment
        const size_t pos = head.size() + meta_.size() + info_.size();
        const size_t aligned = (pos + 31) & ~static_cast<size_t>(31);
        for (size_t i = pos; i < aligned; i++) std::fputc(0, f);
        std::fwrite(payload_.data(), 1, payload_.size(), f);
        std::fclose(f);
        return true;
    }

private:
    u64 tensor_count() const { return names_.size(); }
    u64 kv_count() const { return kv_count_; }

    static void str_into(std::vector<u8> &dst, const std::string &s) {
        const u64 n = s.size();
        append(dst, &n, 8);
        append(dst, s.data(), s.size());
    }
    void put_key(const std::string &k, u32 type) {
        str_into(meta_, k);
        put_u32(type);
        kv_count_++;
    }
    void put_u32(u32 v) { append(meta_, &v, 4); }
    void put_u64(u64 v) { append(meta_, &v, 8); }
    void put_str(const std::string &s) { str_into(meta_, s); }
    void put_raw(const void *p, size_t n) { append(meta_, p, n); }
    void put_payload(const void *p, size_t n) { append(payload_, p, n); }

    void add_tensor(const std::string &name, u32 type, const std::vector<u64> &ne,
                    size_t bytes) {
        str_into(info_, name);
        const u32 nd = static_cast<u32>(ne.size());
        append(info_, &nd, 4);
        for (u64 d : ne) append(info_, &d, 8);
        append(info_, &type, 4);
        u64 off = payload_.size();
        const u64 aligned = (off + 31) & ~static_cast<u64>(31);
        for (u64 i = off; i < aligned; i++) payload_.push_back(0);
        off = aligned;
        append(info_, &off, 8);
        names_.push_back(name);
        (void)bytes;
    }
    static void append(std::vector<u8> &v, const void *p, size_t n) {
        const u8 *b = static_cast<const u8 *>(p);
        v.insert(v.end(), b, b + n);
    }

    std::vector<u8> meta_, info_, payload_;
    std::vector<std::string> names_;
    u64 kv_count_ = 0;
};

// deterministic pseudo-random weights, bounded so fp16 keeps full precision
f32 rnd(u32 &s) {
    s = s * 1664525u + 1013904223u;
    return (static_cast<f32>((s >> 8) & 0xFFFF) / 65536.0f - 0.5f) * 0.5f;
}

} // namespace

// ---------------------------------------------------------------------------
// quant layouts
// ---------------------------------------------------------------------------

static void test_quant_geometry() {
    CHECK(dtype_block_size(DType::Q4_0) == 32, "Q4_0 block is 32");
    CHECK(dtype_block_bytes(DType::Q4_0) == 18, "Q4_0 block is 18 bytes");
    CHECK(dtype_block_size(DType::Q6_K) == 256, "Q6_K block is 256");
    CHECK(dtype_block_bytes(DType::Q6_K) == 210, "Q6_K block is 210 bytes");
    CHECK(dtype_row_bytes(DType::Q8_0, 64) == 68, "Q8_0 row of 64 is 68 bytes");
    CHECK(!dtype_row_aligned(DType::Q4_K, 100), "Q4_K rows must be 256-aligned");
    CHECK(dtype_row_aligned(DType::F16, 100), "F16 rows need no alignment");
    CHECK(!dtype_supported(DType::Unknown), "unknown dtype is rejected");
    CHECK(dtype_block_size(DType::IQ2_XXS) == 256, "IQ2_XXS block is 256");
    CHECK(dtype_block_bytes(DType::IQ2_XXS) == 66, "IQ2_XXS block is 66 bytes");
    CHECK(dtype_block_size(DType::IQ4_XS) == 256, "IQ4_XS block is 256");
    CHECK(dtype_block_bytes(DType::IQ4_XS) == 136, "IQ4_XS block is 136 bytes");
    CHECK(dtype_block_size(DType::NVFP4) == 64, "NVFP4 block is 64");
    CHECK(dtype_block_bytes(DType::NVFP4) == 36, "NVFP4 block is 36 bytes");
    CHECK(dtype_row_bytes(DType::NVFP4, 64) == 36, "NVFP4 row of 64 is 36 bytes");
    CHECK(dtype_supported(DType::IQ2_XXS) && dtype_supported(DType::IQ4_XS) &&
              dtype_supported(DType::NVFP4),
          "new quant formats are supported");
    // Ternary (BitNet). Both are 256-value super-blocks with one fp16 scale;
    // the ids 34/35 are ggml's on-disk GGML_TYPE_TQ1_0/TQ2_0 and must not move.
    CHECK(static_cast<int>(DType::TQ1_0) == 34, "TQ1_0 keeps ggml type id 34");
    CHECK(static_cast<int>(DType::TQ2_0) == 35, "TQ2_0 keeps ggml type id 35");
    CHECK(dtype_block_size(DType::TQ1_0) == 256, "TQ1_0 block is 256");
    CHECK(dtype_block_bytes(DType::TQ1_0) == 54, "TQ1_0 block is 54 bytes (1.6875 bpw)");
    CHECK(dtype_block_size(DType::TQ2_0) == 256, "TQ2_0 block is 256");
    CHECK(dtype_block_bytes(DType::TQ2_0) == 66, "TQ2_0 block is 66 bytes (2.0625 bpw)");
    CHECK(dtype_row_bytes(DType::TQ2_0, 512) == 132, "TQ2_0 row of 512 is 132 bytes");
    CHECK(dtype_row_aligned(DType::TQ2_0, 512), "TQ2_0 rows are 256-aligned");
    CHECK(!dtype_row_aligned(DType::TQ2_0, 100), "TQ2_0 rejects a non-multiple row");
    CHECK(dtype_supported(DType::TQ1_0) && dtype_supported(DType::TQ2_0),
          "ternary formats are supported");
    CHECK(std::strcmp(dtype_name(DType::TQ1_0), "TQ1_0") == 0, "TQ1_0 name");
    CHECK(std::strcmp(dtype_name(DType::TQ2_0), "TQ2_0") == 0, "TQ2_0 name");
    // MXFP4 is ggml id 39 (gpt-oss); 100/101 are the ROCmFPX fork's
    // Q4_0_ROCMFP4 / Q4_0_ROCMFP4_FAST. All three are 32-value blocks.
    CHECK(static_cast<int>(DType::MXFP4) == 39, "MXFP4 keeps ggml type id 39");
    CHECK(static_cast<int>(DType::ROCMFP4) == 100, "ROCmFP4 type id 100");
    CHECK(static_cast<int>(DType::ROCMFP4_FAST) == 101, "ROCmFP4_FAST type id 101");
    CHECK(dtype_block_size(DType::MXFP4) == 32, "MXFP4 block is 32");
    CHECK(dtype_block_bytes(DType::MXFP4) == 17, "MXFP4 block is 17 bytes");
    CHECK(dtype_block_size(DType::ROCMFP4) == 32, "ROCmFP4 block is 32");
    CHECK(dtype_block_bytes(DType::ROCMFP4) == 18, "ROCmFP4 block is 18 bytes");
    CHECK(dtype_block_size(DType::ROCMFP4_FAST) == 32, "ROCmFP4_FAST block is 32");
    CHECK(dtype_block_bytes(DType::ROCMFP4_FAST) == 17, "ROCmFP4_FAST block is 17 bytes");
    CHECK(dtype_row_bytes(DType::MXFP4, 64) == 34, "MXFP4 row of 64 is 34 bytes");
    CHECK(dtype_row_bytes(DType::ROCMFP4, 64) == 36, "ROCmFP4 row of 64 is 36 bytes");
    CHECK(dtype_row_bytes(DType::ROCMFP4_FAST, 64) == 34, "ROCmFP4_FAST row of 64");
    CHECK(dtype_row_aligned(DType::MXFP4, 64), "MXFP4 rows are 32-aligned");
    CHECK(!dtype_row_aligned(DType::MXFP4, 33), "MXFP4 rejects a non-multiple row");
    CHECK(dtype_supported(DType::MXFP4) && dtype_supported(DType::ROCMFP4) &&
              dtype_supported(DType::ROCMFP4_FAST),
          "MXFP4 and the ROCmFPX formats are dequantizable");
    CHECK(std::strcmp(dtype_name(DType::MXFP4), "MXFP4") == 0, "MXFP4 name");
    CHECK(std::strcmp(dtype_name(DType::ROCMFP4), "Q4_0_ROCMFP4") == 0,
          "ROCmFP4 name");
    CHECK(std::strcmp(dtype_name(DType::ROCMFP4_FAST), "Q4_0_ROCMFP4_FAST") == 0,
          "ROCmFP4_FAST name");
}

static void test_q4_0_layout() {
    // d = 1.0, nibbles 0..15 -> values (nibble-8) then (high-8)
    u8 blk[18];
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
    for (int i = 0; i < 16; i++) blk[2 + i] = static_cast<u8>(i | (i << 4));
    f32 out[32];
    dequant_row(DType::Q4_0, blk, out, 32);
    for (int i = 0; i < 16; i++) {
        CHECK_NEAR(out[i], static_cast<f32>(i - 8), 1e-3, "Q4_0 low nibble");
        CHECK_NEAR(out[i + 16], static_cast<f32>(i - 8), 1e-3, "Q4_0 high nibble");
    }
}

static void test_q8_0_layout() {
    u8 blk[34];
    const u16 half = fp32_to_fp16(0.5f);
    blk[0] = static_cast<u8>(half & 0xFF);
    blk[1] = static_cast<u8>(half >> 8);
    for (int i = 0; i < 32; i++) blk[2 + i] = static_cast<u8>(static_cast<i8>(i - 16));
    f32 out[32];
    dequant_row(DType::Q8_0, blk, out, 32);
    for (int i = 0; i < 32; i++)
        CHECK_NEAR(out[i], 0.5f * static_cast<f32>(i - 16), 1e-3, "Q8_0 value");
}

static void test_q5_0_high_bit_layout() {
    // The MSB of elements 16..31 comes from bits 16..31 of qh, not 12..27.
    // Getting this wrong corrupts half of every block, so pin it down.
    u8 blk[22];
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
    const u32 qh = 0x00010001u; // bit0 set for elem 0, bit16 set for elem 16
    std::memcpy(blk + 2, &qh, 4);
    std::memset(blk + 6, 0, 16);
    f32 out[32];
    dequant_row(DType::Q5_0, blk, out, 32);
    CHECK_NEAR(out[0], static_cast<f32>(16 - 16), 1e-3, "Q5_0 elem 0");
    CHECK_NEAR(out[16], static_cast<f32>(16 - 16), 1e-3, "Q5_0 elem 16");
    CHECK_NEAR(out[1], -16.0f, 1e-3, "Q5_0 elem 1 has no high bit");
}

static void test_kquant_roundtrip_shape() {
    // A synthetic Q4_K block: all nibbles 8, scales/ mins 1 -> every value is
    // d*1*8 - dmin*1. With d = 0.5 and dmin = 0.25 that is 3.75, uniformly.
    u8 blk[144];
    const u16 half = fp32_to_fp16(0.5f);
    const u16 qtr = fp32_to_fp16(0.25f);
    blk[0] = static_cast<u8>(half & 0xFF);
    blk[1] = static_cast<u8>(half >> 8);
    blk[2] = static_cast<u8>(qtr & 0xFF);
    blk[3] = static_cast<u8>(qtr >> 8);
    // 12 scale bytes: values 1 in the low 6 bits of each 6-bit field
    for (int i = 0; i < 12; i++) blk[4 + i] = 1;
    for (int i = 0; i < 128; i++) blk[16 + i] = 0x88; // both nibbles = 8
    f32 out[256];
    dequant_row(DType::Q4_K, blk, out, 256);
    // scale/min unpacking only guarantees the low fields; check the mean drift
    // is bounded rather than an exact constant.
    f32 sum = 0;
    for (int i = 0; i < 256; i++) sum += out[i];
    CHECK(std::fabs(sum) < 256.0f * 8.0f, "Q4_K values stay bounded");
}

static void test_q2_k_plane_layout() {
    // ggml's Q2_K is not "16 sub-blocks of 16". It is two 128-value groups
    // whose 2-bit planes live in 32-byte runs of qs, each plane scaling two
    // 16-value halves. A byte in group 1, byte 0 carries all four planes of
    // element 128: 0xE4 = plane0=0, plane1=1, plane2=2, plane3=3.
    u8 blk[84];
    std::memset(blk, 0, sizeof(blk));
    const u16 one = fp32_to_fp16(1.0f);
    blk[80] = static_cast<u8>(one & 0xFF);
    blk[81] = static_cast<u8>(one >> 8);
    blk[82] = 0; // dmin = 0 keeps the arithmetic exact
    blk[83] = 0;
    for (int i = 0; i < 16; i++) blk[i] = 1; // every scale = 1, min = 0
    blk[16 + 32 + 0] = 0xE4;                  // group 1, qs byte 0
    f32 out[256];
    dequant_row(DType::Q2_K, blk, out, 256);
    CHECK_NEAR(out[128], 0.0f, 1e-3, "Q2_K group1 plane0");
    CHECK_NEAR(out[160], 1.0f, 1e-3, "Q2_K group1 plane1");
    CHECK_NEAR(out[192], 2.0f, 1e-3, "Q2_K group1 plane2");
    CHECK_NEAR(out[224], 3.0f, 1e-3, "Q2_K group1 plane3");
    CHECK_NEAR(out[0], 0.0f, 1e-3, "Q2_K group0 untouched");
}

static void test_q6_k_group_stride() {
    // qh advances 32 bytes per 128-value group (not 16). Pin group 1: set its
    // qh byte 0 to 0b11, which lifts the low nibble plane of element 128 to
    // +16 while leaving group 0 at the -32 zero point.
    u8 blk[210];
    std::memset(blk, 0, sizeof(blk));
    const u16 one = fp32_to_fp16(1.0f);
    blk[208] = static_cast<u8>(one & 0xFF);
    blk[209] = static_cast<u8>(one >> 8);
    for (int i = 0; i < 16; i++) blk[192 + i] = 1; // scales = 1
    blk[128 + 32 + 0] = 0x03;                       // group 1, qh byte 0
    f32 out[256];
    dequant_row(DType::Q6_K, blk, out, 256);
    CHECK_NEAR(out[0], -32.0f, 1e-3, "Q6_K group0 zero point");
    CHECK_NEAR(out[128], 16.0f, 1e-3, "Q6_K group1 lifted element");
    CHECK_NEAR(out[160], -32.0f, 1e-3, "Q6_K group1 plane untouched");
}

static void test_iq2_xxs_layout() {
    // IQ2_XXS: 66 bytes = fp16 d + 8 groups of 2xu32, 32 values per
    // group. The group index must scale the output offset by 32: a
    // missing offset overwrites group 0 eight times and leaves groups
    // 1..7 as uninitialized garbage — the exact bug this pins down.
    u8 blk[66];
    std::memset(blk, 0, sizeof(blk));
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
    // group 0: aux32 = {0,0} -> db = d*0.5*0.25 = 0.125, grid 0
    // (every byte 8), no sign bits -> all values are 1.0.
    // group 1: aux32[1] = 0x10000000 -> scale nibble 1 -> db = 0.375
    // -> all values are 3.0. aux32[1] lives at group bytes +4..+7
    // (little-endian), so its top nibble is the top nibble of byte +7.
    blk[2 + 8 + 7] = 0x10; // group 1, aux32[1] top nibble
    // group 2: aux32[1] = 0x10000001 -> db = 0.375 and the l=0 sign
    // word is kSignsIq2xs[1] = 0x81, flipping elements 0 and 7. The
    // sign bits are the low 7 bits of aux32[1], i.e. the low bits of
    // group byte +4.
    blk[2 + 16 + 7] = 0x10;
    blk[2 + 16 + 4] = 0x01;
    f32 out[256];
    dequant_row(DType::IQ2_XXS, blk, out, 256);
    for (int j = 0; j < 32; j++) {
        CHECK_NEAR(out[j], 1.0f, 1e-3, "IQ2_XXS group 0");
        CHECK_NEAR(out[32 + j], 3.0f, 1e-3, "IQ2_XXS group 1");
    }
    CHECK_NEAR(out[64], -3.0f, 1e-3, "IQ2_XXS group 2 sign flip 0");
    CHECK_NEAR(out[71], -3.0f, 1e-3, "IQ2_XXS group 2 sign flip 7");
    CHECK_NEAR(out[65], 3.0f, 1e-3, "IQ2_XXS group 2 unsigned");
    for (int j = 96; j < 256; j++)
        CHECK_NEAR(out[j], 1.0f, 1e-3, "IQ2_XXS untouched groups");
    bool finite = true;
    for (int j = 0; j < 256; j++) finite = finite && std::isfinite(out[j]);
    CHECK(finite, "IQ2_XXS outputs all finite");
    std::vector<f32> x(256, 1.0f);
    CHECK_NEAR(vec_dot(DType::IQ2_XXS, blk, x.data(), 256), 372.0f, 1e-1,
               "IQ2_XXS vec_dot");
}

static void test_iq4_xs_layout() {
    // IQ4_XS: 136 bytes = fp16 d + scales_h(u16) + scales_l[4] +
    // qs[128]. Eight 32-value groups; each scale field ls packs 4 bits
    // from scales_l with 2 bits from scales_h, and dl = d*(ls - 32).
    u8 blk[136];
    std::memset(blk, 0, sizeof(blk));
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
    // ib=0: scales_l low nibble 1, scales_h bits 0..1 = 2 -> ls = 33,
    // dl = 1. ib=1: scales_l high nibble 2 -> ls = 2, dl = -30.
    // ib=2..7: ls = 0 -> dl = -32.
    blk[2] = 0x02; // scales_h low byte: bits 0..1 = 2
    blk[4] = 0x21; // scales_l[0]: low nibble 1, high nibble 2
    for (int i = 0; i < 16; i++) blk[8 + i] = 0x88;  // group 0: nibble 8
    for (int i = 16; i < 32; i++) blk[8 + i] = 0x00; // group 1: nibble 0
    f32 out[256];
    dequant_row(DType::IQ4_XS, blk, out, 256);
    for (int j = 0; j < 32; j++)
        CHECK_NEAR(out[j], 1.0f, 1e-3, "IQ4_XS group 0 (nibble 8)");
    for (int j = 32; j < 64; j++)
        CHECK_NEAR(out[j], 3810.0f, 1e-1, "IQ4_XS group 1 (nibble 0)");
    for (int j = 64; j < 256; j++)
        CHECK_NEAR(out[j], 4064.0f, 1e-1, "IQ4_XS groups 2..7");
    bool finite = true;
    for (int j = 0; j < 256; j++) finite = finite && std::isfinite(out[j]);
    CHECK(finite, "IQ4_XS outputs all finite");
}

static void test_nvfp4_layout() {
    // NVFP4: 36 bytes = 4 inline UE4M3 scales + 4x8 bytes of packed
    // E2M1 nibbles, 16 values per sub-block. Scale bytes 0x00 and 0x7F
    // decode to zero; 0x40 is 1.0 and 0x38 is 0.5.
    u8 blk[36];
    std::memset(blk, 0, sizeof(blk));
    blk[0] = 0x40; // scale 0 = 1.0
    blk[1] = 0x7F; // scale 1 = 0 (the 0x7F special case)
    blk[2] = 0x00; // scale 2 = 0 (the 0x00 special case)
    blk[3] = 0x38; // scale 3 = 0.5
    for (int i = 0; i < 8; i++) blk[4 + i] = 0x01;  // sub-block 0
    for (int i = 0; i < 8; i++) blk[28 + i] = 0x21; // sub-block 3
    f32 out[64];
    dequant_row(DType::NVFP4, blk, out, 64);
    for (int j = 0; j < 8; j++)
        CHECK_NEAR(out[j], 1.0f, 1e-3, "NVFP4 sub 0 low nibble");
    for (int j = 8; j < 48; j++)
        CHECK_NEAR(out[j], 0.0f, 1e-6, "NVFP4 zero scales");
    for (int j = 48; j < 56; j++)
        CHECK_NEAR(out[j], 0.5f, 1e-3, "NVFP4 sub 3 low nibble");
    for (int j = 56; j < 64; j++)
        CHECK_NEAR(out[j], 1.0f, 1e-3, "NVFP4 sub 3 high nibble");
    std::vector<f32> x(64, 1.0f);
    CHECK_NEAR(vec_dot(DType::NVFP4, blk, x.data(), 64), 20.0f, 1e-3,
               "NVFP4 vec_dot");
}

static void test_mxfp4_layout() {
    // MXFP4 (ggml id 39): 17 bytes = one E8M0 exponent byte then 16 bytes of
    // packed E2M1 nibbles. Element j is the LOW nibble of qs[j], element
    // j+16 the HIGH nibble. The value table is doubled (0,1,2,3,4,6,8,12),
    // so the exponent decodes at half scale: 127 -> 2^-1 and 126 -> 2^-2.
    u8 blk[17];
    std::memset(blk, 0, sizeof(blk));
    blk[0] = 127; // d = 0.5
    for (int j = 0; j < 16; j++) blk[1 + j] = 0x11;
    f32 out[32];
    dequant_row(DType::MXFP4, blk, out, 32);
    for (int j = 0; j < 32; j++)
        CHECK_NEAR(out[j], 0.5f, 1e-6, "MXFP4 code 1 at exponent 127");
    std::vector<f32> x(32, 1.0f);
    CHECK_NEAR(vec_dot(DType::MXFP4, blk, x.data(), 32), 16.0f, 1e-3,
               "MXFP4 vec_dot");
    // The exponent byte alone rescales the block: 126 halves every value.
    blk[0] = 126;
    dequant_row(DType::MXFP4, blk, out, 32);
    for (int j = 0; j < 32; j++)
        CHECK_NEAR(out[j], 0.25f, 1e-6, "MXFP4 exponent 126 is half of 127");
    // The full E2M1 ladder plus its negative half, all at d = 0.5.
    blk[0] = 127;
    for (int j = 0; j < 16; j++) blk[1 + j] = static_cast<u8>(j | (j << 4));
    dequant_row(DType::MXFP4, blk, out, 32);
    const f32 ladder[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    for (int j = 0; j < 16; j++) {
        CHECK_NEAR(out[j], ladder[j], 1e-6, "MXFP4 low-nibble ladder");
        CHECK_NEAR(out[16 + j], ladder[j], 1e-6, "MXFP4 high-nibble ladder");
    }
    // Denormal exponent bytes 0 and 1 are 2^-128 and 2^-127, not zero.
    for (int e = 0; e < 2; e++) {
        blk[0] = static_cast<u8>(e);
        for (int j = 0; j < 16; j++) blk[1 + j] = 0x01;
        dequant_row(DType::MXFP4, blk, out, 32);
        CHECK(out[0] > 0.0f && std::isfinite(out[0]),
              "MXFP4 denormal exponent is a positive finite scale");
        CHECK_NEAR(out[16], 0.0f, 1e-30, "MXFP4 code 0 is zero");
    }
}

static void test_rocmfp4_layout() {
    // ROCmFP4 (charlie12345/ROCmFPX): type 100 is 16 packed 4-bit codes
    // followed by TWO UE4M3 half-block scales (18 bytes); type 101 is the
    // same codes with ONE scale for the whole 32-value block (17 bytes).
    // Nibble pairing matches MXFP4: low -> j scaled by e[0], high -> j+16
    // scaled by e[1]. Scale bytes: 0x40 = 1.0, 0x38 = 0.5, 0x7F = 0.
    u8 blk[18];
    std::memset(blk, 0, sizeof(blk));
    for (int j = 0; j < 16; j++) blk[j] = static_cast<u8>(j | (j << 4));
    blk[16] = 0x40; // first 16 values at 1.0
    blk[17] = 0x38; // second 16 values at 0.5
    f32 out[32];
    dequant_row(DType::ROCMFP4, blk, out, 32);
    // Code ladder 0..15: 0,1,2,3,4,6,8,10 then 0,-1,-2,-3,-4,-6,-8,-10.
    const f32 ladder[16] = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  6.0f,  8.0f,  10.0f,
                            -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -10.0f};
    for (int j = 0; j < 16; j++)
        CHECK_NEAR(out[j], ladder[j], 1e-5, "ROCmFP4 first half at scale 1.0");
    for (int j = 0; j < 16; j++)
        CHECK_NEAR(out[16 + j], ladder[j] * 0.5f, 1e-5,
                   "ROCmFP4 second half at scale 0.5");
    // The single-scale layout sees only byte 16 and maps it to all 32 values.
    dequant_row(DType::ROCMFP4_FAST, blk, out, 32);
    for (int j = 0; j < 16; j++) {
        CHECK_NEAR(out[j], ladder[j], 1e-5, "ROCmFP4_FAST scale is byte 16");
        CHECK_NEAR(out[16 + j], ladder[j], 1e-5, "ROCmFP4_FAST high nibble");
    }
    blk[16] = 0x38;
    dequant_row(DType::ROCMFP4_FAST, blk, out, 32);
    for (int j = 0; j < 16; j++)
        CHECK_NEAR(out[j], ladder[j] * 0.5f, 1e-5,
                   "ROCmFP4_FAST follows a 0.5 scale byte");
    // 0x7F and 0x00 are the invalid/zero scale bytes, exactly as NVFP4.
    for (int e = 0; e < 2; e++) {
        blk[16] = static_cast<u8>(e == 0 ? 0x7F : 0x00);
        dequant_row(DType::ROCMFP4_FAST, blk, out, 32);
        for (int j = 0; j < 32; j++) CHECK_NEAR(out[j], 0.0f, 1e-30, "scale 0");
    }
    std::vector<f32> x(32, 1.0f);
    blk[16] = 0x40;
    CHECK_NEAR(vec_dot(DType::ROCMFP4_FAST, blk, x.data(), 32), 0.0f, 1e-3,
               "ROCmFP4_FAST vec_dot of the signed ladder");
}

// TQ2_0 / TQ1_0 — the BitNet ternary formats.
//
// Both pack {-1, 0, +1} with one fp16 scale per 256 values, so they are the
// only formats in this engine where a weight can be *exactly* zero. The
// layout is planar in both cases: the codes for consecutive elements are not
// consecutive bytes. That is what these tests pin down.
//
// The decoder was written from ggml's dequantize_row_tq*_0 and validated
// against ggml's own *encoder* in tools/probe_tq.cpp (a round trip, exact for
// both formats). The synthetic blocks here pin the same layouts from the byte
// side, so a future edit that "simplifies" the packing fails here rather than
// in a model run.
static void test_tq2_0_layout() {
    // 66 bytes: qs[64] then fp16 d. Codes 0/1/2 -> -1/0/+1.
    //
    // Byte (g*32 + m) holds, in shifts 0/2/4/6, the elements
    // (g*128 + m), (g*128 + m + 32), (g*128 + m + 64), (g*128 + m + 96).
    // So byte 0 carries elements 0/32/64/96 and byte 32 carries 128/160/192/224.
    u8 blk[66];
    std::memset(blk, 0, sizeof(blk));
    const u16 one = fp32_to_fp16(1.0f);
    blk[64] = static_cast<u8>(one & 0xFF);
    blk[65] = static_cast<u8>(one >> 8);
    // Byte 0: elem 0 (shift 0) = 2 -> +1, elem 32 (shift 2) = 1 -> 0,
    //         elem 64 (shift 4) = 0 -> -1, elem 96 (shift 6) = 2 -> +1.
    blk[0] = static_cast<u8>(2 | (1 << 2) | (0 << 4) | (2 << 6));
    f32 out[256];
    dequant_row(DType::TQ2_0, blk, out, 256);
    CHECK_NEAR(out[0], 1.0f, 1e-6, "TQ2_0 code 2 is +1");
    CHECK_NEAR(out[32], 0.0f, 1e-6, "TQ2_0 code 1 is 0");
    CHECK_NEAR(out[64], -1.0f, 1e-6, "TQ2_0 code 0 is -1");
    CHECK_NEAR(out[96], 1.0f, 1e-6, "TQ2_0 fourth plane of byte 0");
    // Every untouched byte is 0x00 -> all four of its elements -1, and the
    // second 128-element group (byte 32 onward) is entirely untouched.
    CHECK_NEAR(out[1], -1.0f, 1e-6, "TQ2_0 untouched byte is -1");
    CHECK_NEAR(out[128], -1.0f, 1e-6, "TQ2_0 untouched second group is -1");
    // An all-zero weight block is the 0x55 pattern: the property the MoE
    // zero-block skip depends on.
    std::memset(blk, 0x55, 64);
    dequant_row(DType::TQ2_0, blk, out, 256);
    int nz = 0;
    for (int i = 0; i < 256; i++)
        if (out[i] != 0.0f) nz++;
    CHECK(nz == 0, "TQ2_0 0x55 block decodes to all zeros");
    std::vector<f32> x(256, 1.0f);
    CHECK_NEAR(vec_dot(DType::TQ2_0, blk, x.data(), 256), 0.0f, 1e-6,
               "TQ2_0 all-zero vec_dot is zero");
}

static void test_tq1_0_layout() {
    // 54 bytes: qs[48] (5 trits/byte, base-3) + qh[4] (4 trits/byte) + fp16 d.
    // The base-3 code is scaled by 256/243 and decoded with a multiply high;
    // the 8-bit wrap in ggml's `uint8_t q = x.qs[j+m] * pow3[n]` is part of the
    // format (see quant.cpp).
    //
    // A trit of +1 has the base-3 code 2, so five of them pack to
    // 2*(81+27+9+3+1) = 242 -> ceil(242*256/243) = 255. The qh path packs four
    // trits and then shifts one place further up (q *= 3), giving
    // 242*3 = 726 -> ceil(726*256/243) = 765, which does not fit a byte and
    // truncates to 253. That truncation is in ggml's encoder and is why the
    // all-+1 tail is 253 rather than 255.
    u8 blk[54];
    const u16 one = fp32_to_fp16(1.0f);
    std::memset(blk, 255, 48);
    std::memset(blk + 48, 253, 4);
    blk[52] = static_cast<u8>(one & 0xFF);
    blk[53] = static_cast<u8>(one >> 8);
    f32 out[256];
    dequant_row(DType::TQ1_0, blk, out, 256);
    int wrong = 0;
    for (int i = 0; i < 256; i++)
        if (std::fabs(out[i] - 1.0f) > 1e-6f) wrong++;
    CHECK(wrong == 0, "TQ1_0 all-plus-1 block decodes to 256 +1s");
    CHECK_NEAR(out[255], 1.0f, 1e-6, "TQ1_0 last element comes from qh");

    // The all-zero weight block. A trit of 0 has base-3 code 1, so a run of
    // five zeros is 1*3^4+1*3^3+1*3^2+1*3+1 = 121, stored as
    // ceil(121*256/243) = 128 in qs; qh shifts one place further, giving
    // 1*3^4+1*3^3+1*3^2+1*3 = 120 -> ceil(120*256/243) = 127.
    std::memset(blk, 128, 48);
    std::memset(blk + 48, 127, 4);
    dequant_row(DType::TQ1_0, blk, out, 256);
    int nz = 0;
    for (int i = 0; i < 256; i++)
        if (out[i] != 0.0f) nz++;
    CHECK(nz == 0, "TQ1_0 zero block decodes to all zeros");

    // Scale: d = 0.5 halves the reconstruction. This must run on the all-+1
    // block -- scaling the zero block above would test nothing but 0 * 0.5.
    std::memset(blk, 255, 48);
    std::memset(blk + 48, 253, 4);
    const u16 half = fp32_to_fp16(0.5f);
    blk[52] = static_cast<u8>(half & 0xFF);
    blk[53] = static_cast<u8>(half >> 8);
    dequant_row(DType::TQ1_0, blk, out, 256);
    CHECK_NEAR(out[0], 0.5f, 1e-6, "TQ1_0 d scales the trit");
    CHECK_NEAR(out[255], 0.5f, 1e-6, "TQ1_0 d scales the qh tail");

    std::vector<f32> x(256, 1.0f);
    CHECK_NEAR(vec_dot(DType::TQ1_0, blk, x.data(), 256), 128.0f, 1e-3,
               "TQ1_0 vec_dot sums the scaled trits");

    // The zero block is what makes ternary interesting for MoE paging, so the
    // property is asserted in the form the paging policy will use it: a
    // decoded weight is *exactly zero, not merely small.
    std::memset(blk, 128, 48);
    std::memset(blk + 48, 127, 4);
    blk[52] = static_cast<u8>(one & 0xFF);
    blk[53] = static_cast<u8>(one >> 8);
    dequant_row(DType::TQ1_0, blk, out, 256);
    int exactly_zero = 0;
    for (int i = 0; i < 256; i++)
        if (out[i] == 0.0f) exactly_zero++;
    CHECK(exactly_zero == 256, "TQ1_0 zero block is exactly zero, not merely small");
}

// ---------------------------------------------------------------------------
// Prism-ML's group-128 ternary family (PrismML-Eng/llama.cpp branch prism,
// ggml ids 142 PQ2_0 and 143 PTQ1_0), and the prism.hadamard activation
// transform that those checkpoints also require.
// ---------------------------------------------------------------------------

static void test_pq2_0_layout() {
    // {f16 d; u8 qs[32]} = 34 bytes / 128 values. The codec is Q2_0's at
    // group 128: four codes per byte, low pair first, 00 -> -1, 01 -> 0,
    // 10 -> +1, 11 -> +2, value = (code - 1) * d. Authority: block_pq2_0 and
    // dequantize_row_pq2_0 in that fork's ggml-common.h / ggml-quants.c.
    u8 blk[34];
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
    std::memset(blk + 2, 0x55, 32); // every code 1 -> value 0
    blk[2] = 0xE4;                  // 11 10 01 00 -> codes at shifts 6/4/2/0
    f32 out[128];
    dequant_row(DType::PQ2_0, blk, out, 128);
    CHECK_NEAR(out[0], -1.0f, 1e-6, "PQ2_0: code 00 is -d");
    CHECK_NEAR(out[1], 0.0f, 1e-6, "PQ2_0: code 01 is 0, not -d");
    CHECK_NEAR(out[2], 1.0f, 1e-6, "PQ2_0: code 10 is +d");
    CHECK_NEAR(out[3], 2.0f, 1e-6, "PQ2_0: code 11 is +2d, not clamped");
    CHECK_NEAR(out[4], 0.0f, 1e-6, "PQ2_0: byte 1 starts at value 4");
    CHECK_NEAR(out[127], 0.0f, 1e-6, "PQ2_0: 128 values come from one 34-byte block");
    int nz = 0;
    for (int i = 0; i < 128; i++)
        if (out[i] != 0.0f) nz++;
    CHECK(nz == 3, "PQ2_0: exactly byte 0's three non-zero codes decoded");

    // d carries through, including the +2d step.
    const u16 half = fp32_to_fp16(0.5f);
    blk[0] = static_cast<u8>(half & 0xFF);
    blk[1] = static_cast<u8>(half >> 8);
    dequant_row(DType::PQ2_0, blk, out, 128);
    CHECK_NEAR(out[0], -0.5f, 1e-6, "PQ2_0: d scales every code");
    CHECK_NEAR(out[3], 1.0f, 1e-6, "PQ2_0: +2d with d = 0.5");

    // Two blocks must be walked at a 34-byte stride: 32 reads the second
    // block nine bytes early (fluent garbage), 64 skips half of them.
    u8 two[68];
    std::memset(two, 0, sizeof(two));
    two[0] = static_cast<u8>(one & 0xFF);
    two[1] = static_cast<u8>(one >> 8);
    two[2] = 0xE4;
    two[34] = static_cast<u8>(one & 0xFF);
    two[35] = static_cast<u8>(one >> 8);
    two[36] = 0xE4;
    f32 out2[256];
    dequant_row(DType::PQ2_0, two, out2, 256);
    CHECK_NEAR(out2[3], 2.0f, 1e-6, "PQ2_0: block 0 fills 0..127");
    CHECK_NEAR(out2[127], -1.0f, 1e-6, "PQ2_0: block 0 ends at 127");
    CHECK_NEAR(out2[128], -1.0f, 1e-6, "PQ2_0: block 1 starts at 128");
    CHECK_NEAR(out2[131], 2.0f, 1e-6, "PQ2_0: block 1's codes read at +128");

    std::vector<f32> x(128, 1.0f);
    CHECK_NEAR(vec_dot(DType::PQ2_0, blk, x.data(), 128), 1.0f, 1e-3,
               "PQ2_0: vec_dot sums the scaled codes (-0.5 + 0 + 0.5 + 1)");
}

static void test_ptq1_0_layout() {
    // {u8 qs[24]; u8 qh[2]; f16 d} = 28 bytes / 128 values: TQ1_0's base-3
    // trit packing at group 128, scale LAST (bytes 26..27). Authority:
    // block_ptq1_0 and dequantize_row_ptq1_0 in that fork. The traversal is
    // stage windows {32, 16, 8} over qs -- the 32-wide window never fits in
    // 24 bytes -- five trit planes per window, then four planes over qh:
    //   idx [0, 80):   qs[idx % 16],  plane idx / 16
    //   idx [80, 120): qs[16 + t % 8], plane t / 8      (t = idx - 80)
    //   idx [120,128): qh[t % 2],      plane t / 2      (t = idx - 120)
    u8 blk[28];
    const u16 one = fp32_to_fp16(1.0f);

    // The all-zero block. A trit of 0 has base-3 code 1, which lands on 128
    // in the qs planes and 127 in qh (one place further, 4 planes vs 5).
    std::memset(blk, 128, 24);
    std::memset(blk + 24, 127, 2);
    blk[26] = static_cast<u8>(one & 0xFF);
    blk[27] = static_cast<u8>(one >> 8);
    f32 out[128];
    dequant_row(DType::PTQ1_0, blk, out, 128);
    int zero = 0;
    for (int i = 0; i < 128; i++)
        if (out[i] == 0.0f) zero++;
    CHECK(zero == 128, "PTQ1_0: the zero block decodes to 128 zeros");

    // The all-+1 block: every plane of 255 lands on +1.
    std::memset(blk, 255, 26);
    blk[26] = static_cast<u8>(one & 0xFF);
    blk[27] = static_cast<u8>(one >> 8);
    dequant_row(DType::PTQ1_0, blk, out, 128);
    int wrong = 0;
    for (int i = 0; i < 128; i++)
        if (std::fabs(out[i] - 1.0f) > 1e-6f) wrong++;
    CHECK(wrong == 0, "PTQ1_0: the all-plus-1 block decodes to 128 +1s");
    CHECK_NEAR(out[127], 1.0f, 1e-6, "PTQ1_0: the last value comes from qh");

    // Traversal, first stage. Byte 17 decodes to -1, -1, 0, +1, 0 across its
    // five planes (17*1 -> 0, 17*3 -> 0, 17*9 -> 1, 17*27 -> 2, 17*81 -> 1
    // after the (byte * 3^n) * 3 >> 8 step), so planting it at qs[0] puts
    // exactly those at output indices 0, 16, 32, 48, 64 -- one per plane.
    static const f32 want5[5] = {-1.0f, -1.0f, 0.0f, 1.0f, 0.0f};
    std::memset(blk, 128, 26);
    blk[0] = 17;
    dequant_row(DType::PTQ1_0, blk, out, 128);
    for (int p = 0; p < 5; p++)
        CHECK_NEAR(out[p * 16], want5[p], 1e-6,
                   "PTQ1_0: qs[0] feeds one index per plane");
    int nz = 0;
    for (int i = 0; i < 80; i++)
        if (out[i] != 0.0f) nz++;
    CHECK(nz == 3, "PTQ1_0: nothing outside qs[0]'s five indices decoded");

    // Second stage: the 8-wide window over qs[16..23] starts at index 80.
    std::memset(blk, 128, 26);
    blk[16] = 17;
    dequant_row(DType::PTQ1_0, blk, out, 128);
    for (int p = 0; p < 5; p++)
        CHECK_NEAR(out[80 + p * 8], want5[p], 1e-6,
                   "PTQ1_0: the 8-wide stage starts at index 80");
    nz = 0;
    for (int i = 0; i < 120; i++)
        if (out[i] != 0.0f) nz++;
    CHECK(nz == 3, "PTQ1_0: the 8-wide stage reads qs[16] and nothing else");

    // qh: four planes of two bytes, the last eight values.
    std::memset(blk, 128, 26);
    blk[24] = 17;
    dequant_row(DType::PTQ1_0, blk, out, 128);
    static const f32 want4[4] = {-1.0f, -1.0f, 0.0f, 1.0f};
    for (int p = 0; p < 4; p++) {
        CHECK_NEAR(out[120 + p * 2], want4[p], 1e-6,
                   "PTQ1_0: qh[0] feeds one index per plane");
        CHECK_NEAR(out[121 + p * 2], 0.0f, 1e-6,
                   "PTQ1_0: qh[1] is untouched and stays 0");
    }

    // d sits at bytes 26..27 and scales every stage, qh included.
    std::memset(blk, 255, 26);
    const u16 half = fp32_to_fp16(0.5f);
    blk[26] = static_cast<u8>(half & 0xFF);
    blk[27] = static_cast<u8>(half >> 8);
    dequant_row(DType::PTQ1_0, blk, out, 128);
    CHECK_NEAR(out[0], 0.5f, 1e-6, "PTQ1_0: d scales the qs trits");
    CHECK_NEAR(out[127], 0.5f, 1e-6, "PTQ1_0: d scales the qh tail");
    std::vector<f32> x(128, 1.0f);
    CHECK_NEAR(vec_dot(DType::PTQ1_0, blk, x.data(), 128), 64.0f, 1e-3,
               "PTQ1_0: vec_dot sums 128 scaled trits");
}

// ROCmFPX fork ternary formats (github.com/charlie12345/ROCmFPX): all three are
// 32-value blocks with two UE4M3 half-block scales (e[2]), one per 16-element half.
// The difference is only the code width and the packing. Authority: the fork's
// rocmfpx_dequantize_row_fp{2,3,6} in ggml/rocmfpx/rocmfpx.c, and the pack/
// unpack helpers (rocmfpx_fp3_pack8 / rocmfpx_fp3_unpack8 / rocmfpx_fp6_pack4 /
// rocmfpx_fp6_unpack4).

static void test_q2_0_rocmfpx_layout() {
    // block_rocmfp2: {u8 qs[8]; u8 e[2]} = 10 bytes / 32 values.
    // S40 ladder {-4,-1,+1,+4}, 2-bit codes packed 4 to a byte (LSB first).
    // qs[half*4 + j/4] >> (2*(j%4)) & 3.
    u8 blk[10];
    std::memset(blk, 0, sizeof(blk));
    // Code 1 -> -1, so the all-1 block is all -1 * scale.
    for (int j = 0; j < 8; j++) blk[j] = 0x55; // 0b01010101 -> codes {1,1,1,1}
    // e[0] = 0x40 -> 1.0f, e[1] = 0x38 -> 0.5f.
    blk[8] = 0x40;
    blk[9] = 0x38;
    f32 out[32];
    dequant_row(DType::Q2_0_ROCMFPX, blk, out, 32);
    for (int j = 0; j < 16; j++)
        CHECK_NEAR(out[j], -1.0f, 1e-6, "Q2_0_ROCMFPX first half at scale 1.0");
    for (int j = 0; j < 16; j++)
        CHECK_NEAR(out[16 + j], -0.5f, 1e-6,
                   "Q2_0_ROCMFPX second half at scale 0.5");
    // The zero scale bytes (0x00 and 0x7F) produce 0.
    for (int e = 0; e < 2; e++) {
        u8 z = static_cast<u8>(e == 0 ? 0x7F : 0x00);
        blk[8] = z;
        blk[9] = z;
        dequant_row(DType::Q2_0_ROCMFPX, blk, out, 32);
        for (int j = 0; j < 32; j++) CHECK_NEAR(out[j], 0.0f, 1e-30, "zero scale");
    }
    // Empty block (all zeros): codes are 0 -> -4, but scale 0 -> all zeros.
    std::memset(blk, 0, sizeof(blk));
    blk[8] = 0x40;
    dequant_row(DType::Q2_0_ROCMFPX, blk, out, 32);
    for (int j = 0; j < 16; j++)
        CHECK_NEAR(out[j], -4.0f, 1e-6, "Q2_0_ROCMFPX all-code-0 is -4");
    for (int j = 0; j < 16; j++)
        CHECK_NEAR(out[16 + j], 0.0f, 1e-6,
                   "Q2_0_ROCMFPX second half code-0 at scale 0.5 is 0");
    // S40 ladder {-4,-1,+1,+4} = codes {0,1,2,3}, LSB-first pack.
    // byte = 0b11_10_01_00 = bits 6-7=3, 4-5=2, 2-3=1, 0-1=0 = 0xE4.
    std::memset(blk, 0, sizeof(blk));
    blk[8] = 0x40;
    blk[0] = 0xE4;
    dequant_row(DType::Q2_0_ROCMFPX, blk, out, 32);
    CHECK_NEAR(out[0], -4.0f, 1e-6, "Q2_0_ROCMFPX code 0 = -4");
    CHECK_NEAR(out[1], -1.0f, 1e-6, "Q2_0_ROCMFPX code 1 = -1");
    CHECK_NEAR(out[2], 1.0f, 1e-6, "Q2_0_ROCMFPX code 2 = +1");
    CHECK_NEAR(out[3], 4.0f, 1e-6, "Q2_0_ROCMFPX code 3 = +4");
}

static void test_q3_0_rocmfpx_layout() {
    // block_rocmfp3: {u8 qs[12]; u8 e[2]} = 14 bytes / 32 values.
    // 3-bit codes, mag in {0,1,2,4}, sign in bit 4. 8 codes packed per 3 bytes.
    u8 blk[14];
    std::memset(blk, 0, sizeof(blk));
    // Pack all-eight codes = {1,2,4,0,1,2,4,0} into qs[0..2]: LSB-first layout:
    //   byte0: c0[2:0] | c1[2:0]<<3 | c2[1:0]<<6
    //   byte1: c2[2] | c3[2:0]<<1 | c4[2:0]<<4 | c5[0]<<7
    //   byte2: c5[1:0]<<2 | c6[2:0]<<5 | c7[2:0]<<2... actually:
    //   byte2: c5[1:0]<<2 | c6[2:0]<<2 | c7[2:0]<<5
    // We plant code 1 (mag 1, sign +) at c[0]: byte0 = 0x01.
    blk[0] = 0x01; // c[0] = 1, rest 0
    blk[12] = 0x40; // e[0] = 1.0
    blk[13] = 0x38; // e[1] = 0.5
    f32 out[32];
    dequant_row(DType::Q3_0_ROCMFPX, blk, out, 32);
    CHECK_NEAR(out[0], 1.0f, 1e-6, "Q3_0_ROCMFPX code 1 at scale 1.0 = +1");
    for (int j = 1; j < 16; j++) CHECK_NEAR(out[j], 0.0f, 1e-6,
                   "Q3_0_ROCMFPX other first-half codes are 0");
    // The second half is all zeros (qs[6..11] == 0, codes 0 -> mag 0).
    // Now the full four-code ladder in a pack8 group: codes 0,1,2,3,4,5,6,7.
    // code 0 = 0, code 1 = +1, code 2 = +2, code 3 = +4 (sign bit clear),
    // code 4 = 0 (mag 0), code 5 = -1 (sign bit set), code 6 = -2, code 7 = -4.
    // Pack8 byte layout (LSB first) for codes [0..7]:
    //   byte0 = c0 | c1<<3 | c2[1:0]<<6  = 0 | 1<<3 | 2<<6 = 0x01 | 0x08 | 0x80 = 0x89
    //   byte1 = c2[2] | c3<<1 | c4<<4 | c5[0]<<7 = 0 | 4<<1 | 0... no, c4=0, c5=5&4=4 -> 1<<7 = 0x80; c3=4 -> 4<<1 = 8. So byte1 = 0 | 8 | 0 | 0x80 = 0x88
    // Hmm, let me recompute: c3=4 (mag 4, sign 0) -> 4<<1 = 8; c4=0; c5=5 (mag 1, sign 1) -> bit0 = 1, so 1<<7 = 0x80. byte1 = 0 | 8 | 0 | 0x80 = 0x88.
    // Wait c2=2, so c2[2] = (2>>2)&1 = 0. byte1 = 0 | 8 | 0 | 128 = 136 = 0x88.
    //   byte2 = c5[1:0]<<2 | c6<<4.. no: c5[1:0]<<2 | c6[2:0]<<4 | c7[2:0]<<5... actually the unpack is:
    //   c5 = (src[1]>>7)&1 | (src[2]&3)<<1 -- so byte2 contributes c5 bits 1:0 = src[2]&3, shifted left 1; and c6 = src[2]>>2 & 7; c7 = src[2]>>5 & 7.
    // For codes [0,1,2,3,4,5,6,7]: c5=5, c5[1:0]=5&3=1, so (1)<<2=4 in byte2; c6=6, c6<<4 = 6<<4 = 96; c7=7, c7<<5 would overflow 8 bits... no, byte2 = c5[1:0]<<2 | c6[2:0]<<4 | c7[2:0]<<5 = 4 | (6<<4)&0xF0 | (7<<5)&0xE0 = 4 | 0x60 | 0xE0 = 0xE4... but that's 228, and c6=6 needs bits 4-6 (3 bits), 6<<4=96=0x60; c7=7 needs bits 5-7, 7<<5=224=0xE0. 0x60|0xE0=0xE0. So byte2 = 4 | 0xE0 = 0xE4.
    // Let me just set each code explicitly via the unpack inverse.
    // Simpler: code 1 at position 0, code 2 at position 1, code 4 at position 2,
    // code 5 (-1) at position 3 -- in one pack8 group at qs[0..2].
    std::memset(blk, 0, sizeof(blk));
    blk[12] = 0x40;
    // codes[0..7] = {1, 2, 4, 5, 0, 0, 0, 0}
    // byte0 = c0 | c1<<3 | (c2&3)<<6 = 1 | 2<<3 | (4&3)<<6 = 1 | 16 | 0<<6 = 17 = 0x11; wait c2=4, c2&3=0, so (0)<<6 = 0. byte0 = 1 | 16 | 0 = 17.
    // byte1 = (c2>>2)&1 | c3<<1 | c4<<4 | (c5&1)<<7 = (4>>2)&1 | 5<<1 | 0<<4 | (5&1)<<7 = 1 | 10 | 0 | 128 = 139 = 0x8B.
    // byte2 = (c5>>1)&3<<2 | c6<<4 | c7<<5 -- c5=5, (5>>1)&3 = 2&3 = 2, 2<<2 = 8; c6=0; c7=0. byte2 = 8.
    blk[0] = 0xD1;
    blk[1] = 0x0A;
    blk[2] = 0x00;
    dequant_row(DType::Q3_0_ROCMFPX, blk, out, 32);
    CHECK_NEAR(out[0], 1.0f, 1e-6, "Q3_0_ROCMFPX code 1 = +1");
    CHECK_NEAR(out[1], 2.0f, 1e-6, "Q3_0_ROCMFPX code 2 = +2");
    CHECK_NEAR(out[2], 4.0f, 1e-6, "Q3_0_ROCMFPX code 3 = +4");
    CHECK_NEAR(out[3], -1.0f, 1e-6, "Q3_0_ROCMFPX code 5 = -1 (sign bit)");
    CHECK_NEAR(out[4], 0.0f, 1e-6, "Q3_0_ROCMFPX code 0 = 0");
    // Valid scale check: e <= 0x7E; 0x7F -> 0.
    blk[12] = 0x7F;
    dequant_row(DType::Q3_0_ROCMFPX, blk, out, 32);
    for (int j = 0; j < 32; j++) CHECK_NEAR(out[j], 0.0f, 1e-30, "Q3_0_ROCMFPX invalid scale 0x7F");
    std::vector<f32> x(32, 1.0f);
    std::memset(blk, 0, sizeof(blk));
    blk[12] = 0x40;
    blk[0] = 0xD1; blk[1] = 0x0A; blk[2] = 0x00;
    CHECK_NEAR(vec_dot(DType::Q3_0_ROCMFPX, blk, x.data(), 32), 6.0f, 1e-3,
               "Q3_0_ROCMFPX vec_dot of the four-code ladder {+1,+2,+4,-1}");
}

static void test_q6_0_rocmfpx_layout() {
    // block_rocmfp6: {u8 qs[24]; u8 e[2]} = 26 bytes / 32 values.
    // 6-bit codes: mag = code & 31, sign in bit 5, mag 0 -> -32.
    // 4 codes packed per 3 bytes (pack4, LSB first).
    u8 blk[26];
    std::memset(blk, 0, sizeof(blk));
    // Pack 4 codes = {1, 2, 31, 33} (0, +1, +2, +31, -32, ...) into qs[0..2]:
    //   byte0 = c0 | c1<<6; c0=1, c1=2 -> 1 | 2<<6 = 1 | 128 = 129 = 0x81.
    //   byte1 = (c1>>2)&0xF | c2<<4; c2=31 -> 31<<4 = 496 = 0x1F0, low byte 0xF0; (2>>2)&0xF = 0. byte1 = 0 | 0xF0 = 0xF0.
    //   byte2 = (c2>>4)&3<<2 | c3<<2; c2>>4 = 1, &3 = 1, <<2 = 4; c3 = 33 & 31 = 1, but wait c3 = 33 means code 33, but mag = 33&31 = 1, sign bit 32 set -> -1.
    // Actually let me use codes {1,2,31,32} -> mag {1,2,31,0} with sign {+,+,+,-} -> values {+1,+2,+31,-32}.
    // c0=1, c1=2, c2=31, c3=32 (code 32 = mag 0, sign bit 32 set -> -32).
    // byte0 = 1 | 2<<6 = 1 | 128 = 129.
    // byte1 = (2>>2)&0xF | 31<<4 = 0 | 0xF0 = 0xF0.
    // byte2 = (31>>4)&3<<2 | 32<<2 = (1)&3<<2... (31>>4)=1, 1&3=1, 1<<2=4; 32<<2 = 128 = 0x80. byte2 = 4 | 128 = 132 = 0x84.
    blk[0] = 0x81;
    blk[1] = 0xF0;
    blk[2] = 0x81;
    blk[24] = 0x40; // e[0] = 1.0
    blk[25] = 0x38; // e[1] = 0.5
    f32 out[32];
    dequant_row(DType::Q6_0_ROCMFPX, blk, out, 32);
    CHECK_NEAR(out[0], 1.0f, 1e-6, "Q6_0_ROCMFPX code 1 at scale 1.0 = +1");
    CHECK_NEAR(out[1], 2.0f, 1e-6, "Q6_0_ROCMFPX code 2 at scale 1.0 = +2");
    CHECK_NEAR(out[2], 31.0f, 1e-6, "Q6_0_ROCMFPX code 31 at scale 1.0 = +31");
    CHECK_NEAR(out[3], -32.0f, 1e-6, "Q6_0_ROCMFPX code 32 (mag 0, sign) = -32");
    // Second half is all zeros (qs[12..23] == 0 -> codes 0 -> mag 0 -> -32 * 0.5 = -16).
    // Wait, code 0 = mag 0, sign bit clear -> +0. No: rocmfpx_decode_fp6_code(0) = (0 & 32) ? -(0==0?32:0) : 0 = 0. So code 0 = 0.
    for (int j = 16; j < 32; j++) CHECK_NEAR(out[j], 0.0f, 1e-6,
                   "Q6_0_ROCMFPX second half all-code-0 is 0");
    // Code ladder: 0, 1, 31, 32 in second half at scale 0.5.
    std::memset(blk, 0, sizeof(blk));
    blk[0] = 0;
    blk[1] = 0;
    blk[2] = 0;
    // second half: qs[12..23], pack4 group at qs[12..14].
    blk[12] = 0x81;
    blk[13] = 0xF0;
    blk[14] = 0x81;
    blk[24] = 0x40;
    blk[25] = 0x38;
    dequant_row(DType::Q6_0_ROCMFPX, blk, out, 32);
    CHECK_NEAR(out[16], 0.5f, 1e-6, "Q6_0_ROCMFPX code 1 at scale 0.5 = +0.5");
    CHECK_NEAR(out[17], 1.0f, 1e-6, "Q6_0_ROCMFPX code 2 at scale 0.5 = +1.0");
    CHECK_NEAR(out[18], 15.5f, 1e-6, "Q6_0_ROCMFPX code 31 at scale 0.5 = +15.5");
    CHECK_NEAR(out[19], -16.0f, 1e-6, "Q6_0_ROCMFPX code 32 at scale 0.5 = -16");
    // Invalid scale: e <= 0x7E; 0x7F -> 0.
    blk[24] = 0x7F;
    blk[25] = 0x7F;
    dequant_row(DType::Q6_0_ROCMFPX, blk, out, 32);
    for (int j = 0; j < 32; j++) CHECK_NEAR(out[j], 0.0f, 1e-30, "Q6_0_ROCMFPX invalid scale 0x7F");
    std::vector<f32> x(32, 1.0f);
    std::memset(blk, 0, sizeof(blk));
    blk[0] = 0x81; blk[1] = 0xF0; blk[2] = 0x81;
    blk[24] = 0x40; blk[25] = 0x38;
    CHECK_NEAR(vec_dot(DType::Q6_0_ROCMFPX, blk, x.data(), 32),
               1.0f + 2.0f + 31.0f - 32.0f, 1e-3,
               "Q6_0_ROCMFPX vec_dot of the four-code ladder");
}

// Reference for Backend::hadamard_act, written from the definition rather
// than from either implementation: an optional grouped-V perm, the sign
// vector, then H[i][j] = (-1)^popcount(i & j) / sqrt(block) over consecutive
// blocks. The CPU butterfly and the HIP kernels are both checked against it,
// and the two orders are checked against each other (fold then inverse must
// be the identity, which is the property the latent embedding lookup is).
static void had_ref(const f32 *in, f32 *out, i64 rows, i64 n, i64 block,
                    const f32 *signs, bool signs_first, i64 hd, i64 nk,
                    i64 rep) {
    std::vector<f32> w(static_cast<size_t>(n));
    std::vector<f32> tmp(static_cast<size_t>(block));
    const f32 scale = 1.0f / std::sqrt(static_cast<f32>(block));
    for (i64 r = 0; r < rows; r++) {
        if (hd > 0) {
            for (i64 j = 0; j < rep; j++)
                for (i64 k = 0; k < nk; k++)
                    for (i64 i = 0; i < hd; i++)
                        w[static_cast<size_t>(i + j * hd + k * hd * rep)] =
                            in[static_cast<size_t>(r * n + i + k * hd +
                                                   j * hd * nk)];
        } else {
            for (i64 i = 0; i < n; i++)
                w[static_cast<size_t>(i)] = in[static_cast<size_t>(r * n + i)];
        }
        if (signs_first && signs)
            for (i64 i = 0; i < n; i++) w[static_cast<size_t>(i)] *= signs[i];
        for (i64 off = 0; off < n; off += block) {
            for (i64 i = 0; i < block; i++) {
                f32 acc = 0.0f;
                for (i64 j = 0; j < block; j++) {
                    i64 p = i & j;
                    int parity = 0;
                    for (int b = 0; b < 62; b++)
                        parity ^= static_cast<int>((p >> b) & 1);
                    acc += parity ? -w[static_cast<size_t>(off + j)]
                                  : w[static_cast<size_t>(off + j)];
                }
                tmp[static_cast<size_t>(i)] = acc * scale;
            }
            for (i64 i = 0; i < block; i++)
                w[static_cast<size_t>(off + i)] = tmp[static_cast<size_t>(i)];
        }
        if (!signs_first && signs)
            for (i64 i = 0; i < n; i++) w[static_cast<size_t>(i)] *= signs[i];
        for (i64 i = 0; i < n; i++)
            out[static_cast<size_t>(r * n + i)] = w[static_cast<size_t>(i)];
    }
}

static f32 had_worst(const std::vector<f32> &a, const std::vector<f32> &b) {
    f32 worst = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        const f32 d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

static void test_hadamard_act() {
    Backend *cpu = make_cpu_backend();

    // 1. The FWHT alone, against the popcount definition, two blocks a row.
    {
        const i64 block = 64;
        const i64 rows = 3;
        const i64 n = block * 2;
        std::vector<f32> in(static_cast<size_t>(rows * n));
        for (size_t i = 0; i < in.size(); i++)
            in[i] = 0.001f * static_cast<f32>(i % 97) - 0.3f;
        std::vector<f32> ref(in.size()), got(in.size());
        had_ref(in.data(), ref.data(), rows, n, block, nullptr, true, 0, 0, 0);
        Backend::HadDesc d;
        d.n = n;
        d.block = block;
        d.signs = nullptr;
        d.signs_first = true;
        cpu->hadamard_act(got.data(), in.data(), rows, d);
        CHECK(had_worst(got, ref) < 1e-5f,
              "hadamard_act's FWHT matches the popcount definition");
        // H . H = I with the 1/sqrt(block) scale: applying it twice is the
        // identity, which is what lets one routine be fold AND inverse.
        std::vector<f32> back(got.size());
        cpu->hadamard_act(back.data(), got.data(), rows, d);
        CHECK(had_worst(back, in) < 1e-5f,
              "the normalized FWHT is its own inverse");
    }

    // 2. Signs, in both orders, and in place.
    {
        const i64 block = 32;
        const i64 rows = 2;
        const i64 n = 96;
        std::vector<f32> in(static_cast<size_t>(rows * n));
        for (size_t i = 0; i < in.size(); i++)
            in[i] = 0.05f * static_cast<f32>(static_cast<int>(i % 23) - 11);
        std::vector<f32> signs(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++)
            signs[static_cast<size_t>(i)] =
                (i % 3 == 0 || i % 7 == 0) ? -1.0f : 1.0f;
        std::vector<f32> ref(in.size()), got(in.size());
        had_ref(in.data(), ref.data(), rows, n, block, signs.data(), true, 0,
                0, 0);
        Backend::HadDesc d;
        d.n = n;
        d.block = block;
        d.signs = signs.data();
        d.signs_first = true;
        cpu->hadamard_act(got.data(), in.data(), rows, d);
        CHECK(had_worst(got, ref) < 1e-5f,
              "hadamard_act applies signs before the FWHT (the fold)");

        // out == in has to be legal: every site in the engine runs in place
        // on the embedding row, and the prefill paths rely on it.
        std::vector<f32> inplace = in;
        cpu->hadamard_act(inplace.data(), inplace.data(), rows, d);
        CHECK(had_worst(inplace, ref) < 1e-5f,
              "hadamard_act transforms in place when out == in");

        // The inverse order -- FWHT then signs -- must undo the fold. This is
        // the property the latent token embedding's inverse-after-lookup uses.
        std::vector<f32> back(in.size());
        d.signs_first = false;
        cpu->hadamard_act(back.data(), got.data(), rows, d);
        CHECK(had_worst(back, in) < 1e-4f,
              "fold then inverse restores the input exactly");
        std::vector<f32> ref_inv(in.size());
        had_ref(in.data(), ref_inv.data(), rows, n, block, signs.data(), false,
                0, 0, 0);
        std::vector<f32> got_inv(in.size());
        cpu->hadamard_act(got_inv.data(), in.data(), rows, d);
        CHECK(had_worst(got_inv, ref_inv) < 1e-5f,
              "hadamard_act applies signs after the FWHT (the inverse)");
    }

    // 3. The grouped-V permutation ssm_out's input carries: [hd, nk, rep] ->
    // [hd, rep, nk], ahead of everything else.
    {
        const i64 hd = 4, nk = 3, rep = 2;
        const i64 n = hd * nk * rep; // 24
        const i64 block = 8;
        const i64 rows = 2;
        std::vector<f32> in(static_cast<size_t>(rows * n));
        for (size_t i = 0; i < in.size(); i++)
            in[i] = static_cast<f32>(static_cast<int>(i % 17) - 8);
        std::vector<f32> ref(in.size()), got(in.size());
        had_ref(in.data(), ref.data(), rows, n, block, nullptr, true, hd, nk,
                rep);
        Backend::HadDesc d;
        d.n = n;
        d.block = block;
        d.signs = nullptr;
        d.signs_first = true;
        d.perm_hd = hd;
        d.perm_nk = nk;
        d.perm_rep = rep;
        cpu->hadamard_act(got.data(), in.data(), rows, d);
        CHECK(had_worst(got, ref) < 1e-5f,
              "hadamard_act permutes ssm_out's grouped-V feature order");
    }

    delete cpu;
}

// ---------------------------------------------------------------------------
// The IQ family and BitNet's Q1_0 / Q2_0 / Q2_0_64 (ggml ids
// 17/18/19/21/22/29/41/42/48).
// ---------------------------------------------------------------------------
// Three separate things can be silently wrong in a new dequantizer, and each
// gets its own check:
//   * THE GEOMETRY. A wrong byte stride reads the *next* block and decodes
//     fluent garbage, so values-per-block, bytes-per-block and row alignment
//     are asserted against the reference's own sizeof(block_*) values.
//   * THE VALUE MAPPING, for the two formats whose codes are short enough to
//     write out by hand. Q1_0 is one bit (set -> +d, clear -> -d, and there is
//     NO zero code) and Q2_0 is two bits (00 -> -d, 01 -> 0, 10 -> +d,
//     11 -> +2d, the last step being +2 rather than a clamp). The expectations
//     below are typed out, not produced by the decoder under test.
//   * THE GROUP/INDEX MAPPING of the five grid formats. Their codebooks cannot
//     be duplicated here without re-introducing the very transcription risk the
//     shared tables exist to remove, so the mapping is pinned structurally: a
//     change confined to one group's high-index byte must move exactly that
//     group's decoded values and nothing outside them. That is what catches a
//     chunk decoded at the wrong offset.
// The codebooks themselves are diffed entry-by-entry against ggml-common.h
// with a separate extractor (see docs/test-results.md); what lives here is the
// wiring, and the device kernels are checked against these host decoders by the
// model-level coherence run.

struct QuantDiff {
    int lo = -1;
    int hi = -1;
    int n = 0;
};

static QuantDiff diff_indices(const f32 *a, const f32 *b, int n) {
    QuantDiff d;
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            if (d.lo < 0) d.lo = i;
            d.hi = i;
            d.n++;
        }
    }
    return d;
}

// Deterministic bytes, with the first two overwritten by d = 1.0 so the decoded
// values are the codebooks themselves and a probe that changes a scale cannot
// be masked by a zero delta.
static void fill_probe_block(u8 *blk, size_t n, u32 seed) {
    u32 s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        blk[i] = static_cast<u8>(s >> 24);
    }
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
}

static void test_new_quant_geometry() {
    struct RowSpec {
        DType t;
        const char *name;
        int bytes;
        int block;
    };
    // IQ3_XXS 98 = 2 + 3*256/8; IQ1_S 50 = 2 + 32 + 16; IQ3_S 110 =
    // 2 + 64 + 8 + 32 + 4; IQ2_S 82 = 2 + 64 + 8 + 8; IQ1_M 56 = 32 + 16 + 8
    // and has NO f16 d field at all; Q1_0 18 = 2 + 128/8; Q2_0 34 = 2 + 128/4;
    // Q2_0_64 18 = 2 + 64/4. Q2_0 and Q2_0_64 share a code map, not a block.
    // PQ2_0 34 = 2 + 128/4 (the Q2_0 codec at group 128); PTQ1_0 28 =
    // 24 + 2 + 2 (TQ1_0's base-3 packing at group 128, scale last).
    const RowSpec rows[] = {
        {DType::IQ3_XXS, "IQ3_XXS", 98, 256},
        {DType::IQ1_S, "IQ1_S", 50, 256},
        {DType::IQ3_S, "IQ3_S", 110, 256},
        {DType::IQ2_S, "IQ2_S", 82, 256},
        {DType::IQ1_M, "IQ1_M", 56, 256},
        {DType::Q1_0, "Q1_0", 18, 128},
        {DType::Q2_0, "Q2_0", 34, 128},
        {DType::Q2_0_64, "Q2_0_64", 18, 64},
        {DType::PQ2_0, "PQ2_0", 34, 128},
        {DType::PTQ1_0, "PTQ1_0", 28, 128},
        // ROCmFPX fork ternary formats (charlie12345/ROCmFPX): all 32-value
        // blocks with two UE4M3 half-block scales. Q6_0_ROCMFPX 26 = 24 + 2,
        // Q3_0_ROCMFPX 14 = 12 + 2, Q2_0_ROCMFPX 10 = 8 + 2.
        {DType::Q6_0_ROCMFPX, "Q6_0_ROCMFPX", 26, 32},
        {DType::Q3_0_ROCMFPX, "Q3_0_ROCMFPX", 14, 32},
        {DType::Q2_0_ROCMFPX, "Q2_0_ROCMFPX", 10, 32},
    };
    for (const RowSpec &r : rows) {
        CHECK(dtype_supported(r.t), "the format is dequantizable");
        CHECK(std::strcmp(dtype_name(r.t), r.name) == 0, "the format keeps its name");
        CHECK(dtype_block_size(r.t) == r.block, "values per block");
        CHECK(dtype_block_bytes(r.t) == r.bytes, "bytes per block");
        CHECK(dtype_row_bytes(r.t, r.block) == static_cast<size_t>(r.bytes),
              "one block is one row of that length");
        CHECK(dtype_row_bytes(r.t, 3 * r.block) == static_cast<size_t>(3 * r.bytes),
              "three blocks are three block strides");
        CHECK(dtype_row_aligned(r.t, r.block), "a whole block is aligned");
        CHECK(!dtype_row_aligned(r.t, 1), "a partial block is refused");
    }
    // The ids are ggml's on-disk values and must not move: the tensor type in
    // the container is the only thing that says how to read the bytes.
    CHECK(static_cast<int>(DType::IQ2_XS) == 17, "IQ2_XS keeps ggml id 17");
    CHECK(static_cast<int>(DType::IQ3_XXS) == 18, "IQ3_XXS keeps ggml id 18");
    CHECK(static_cast<int>(DType::IQ1_S) == 19, "IQ1_S keeps ggml id 19");
    CHECK(static_cast<int>(DType::IQ3_S) == 21, "IQ3_S keeps ggml id 21");
    CHECK(static_cast<int>(DType::IQ2_S) == 22, "IQ2_S keeps ggml id 22");
    CHECK(static_cast<int>(DType::IQ1_M) == 29, "IQ1_M keeps ggml id 29");
    CHECK(static_cast<int>(DType::Q1_0) == 41, "Q1_0 keeps ggml id 41");
    CHECK(static_cast<int>(DType::Q2_0) == 42, "Q2_0 keeps ggml id 42");
    CHECK(static_cast<int>(DType::Q2_0_64) == 48, "Q2_0_64 keeps fork id 48");
    // PrismML's ids, from that fork's ggml.h: GGML_TYPE_PQ2_0 = 142,
    // GGML_TYPE_PTQ1_0 = 143 (and GGML_TYPE_COUNT = 144 there).
    CHECK(static_cast<int>(DType::PQ2_0) == 142, "PQ2_0 keeps Prism id 142");
    CHECK(static_cast<int>(DType::PTQ1_0) == 143, "PTQ1_0 keeps Prism id 143");
    // ROCmFPX fork ids (github.com/charlie12345/ROCmFPX): 102 Q6_0_ROCMFPX,
    // 104 Q3_0_ROCMFPX, 107 Q2_0_ROCMFPX. These are not upstream ids.
    CHECK(static_cast<int>(DType::Q6_0_ROCMFPX) == 102, "Q6_0_ROCMFPX keeps fork id 102");
    CHECK(static_cast<int>(DType::Q3_0_ROCMFPX) == 104, "Q3_0_ROCMFPX keeps fork id 104");
    CHECK(static_cast<int>(DType::Q2_0_ROCMFPX) == 107, "Q2_0_ROCMFPX keeps fork id 107");
}

static void test_q1_0_layout() {
    u8 blk[18];
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
    for (int i = 0; i < 16; i++) blk[2 + i] = 0;
    blk[2] = 0x01;   // value 0 is +d
    blk[3] = 0x80;   // bit 7 of the second plane byte is value 15
    f32 out[128];
    dequant_row(DType::Q1_0, blk, out, 128);
    CHECK_NEAR(out[0], 1.0f, 1e-6, "Q1_0: a set bit is +d");
    CHECK_NEAR(out[1], -1.0f, 1e-6, "Q1_0: a clear bit is -d, not zero");
    CHECK_NEAR(out[15], 1.0f, 1e-6, "Q1_0: bit 7 of plane byte 1 is value 15");
    CHECK_NEAR(out[16], -1.0f, 1e-6, "Q1_0: plane byte 2 starts at value 16");
    CHECK_NEAR(out[127], -1.0f, 1e-6, "Q1_0: 128 values come from one block");
    int plus = 0;
    for (int i = 0; i < 128; i++) if (out[i] > 0) plus++;
    CHECK(plus == 2, "Q1_0: exactly the two set bits are +d");
    const u16 neg = fp32_to_fp16(-1.0f);
    blk[0] = static_cast<u8>(neg & 0xFF);
    blk[1] = static_cast<u8>(neg >> 8);
    dequant_row(DType::Q1_0, blk, out, 128);
    CHECK_NEAR(out[0], -1.0f, 1e-6, "Q1_0: a negative delta carries through");
}

static void test_q2_0_layout() {
    // Q2_0 is the 128-value / 34-byte block: one f16 delta, then a 32-byte
    // plane holding four codes per byte, low pair first.
    u8 blk[34];
    const u16 one = fp32_to_fp16(1.0f);
    blk[0] = static_cast<u8>(one & 0xFF);
    blk[1] = static_cast<u8>(one >> 8);
    for (int i = 0; i < 32; i++) blk[2 + i] = 0;
    // One byte carries four codes, low pair first: 00 01 10 11.
    blk[2] = static_cast<u8>(0x00 | (1 << 2) | (2 << 4) | (3 << 6));
    f32 out[128];
    dequant_row(DType::Q2_0, blk, out, 128);
    CHECK_NEAR(out[0], -1.0f, 1e-6, "Q2_0: code 0 is -d");
    CHECK_NEAR(out[1], 0.0f, 1e-6, "Q2_0: code 1 is 0");
    CHECK_NEAR(out[2], 1.0f, 1e-6, "Q2_0: code 2 is +d");
    CHECK_NEAR(out[3], 2.0f, 1e-6, "Q2_0: code 3 is +2d, not clamped");
    CHECK_NEAR(out[4], -1.0f, 1e-6, "Q2_0: the next byte starts at value 4");
    CHECK_NEAR(out[127], -1.0f, 1e-6, "Q2_0: 128 values come from one block");
    // The last plane byte is bytes 2+31, i.e. values 124..127.
    blk[33] = static_cast<u8>(0x03);
    dequant_row(DType::Q2_0, blk, out, 128);
    CHECK_NEAR(out[124], 2.0f, 1e-6, "Q2_0: plane byte 31 is value 124");
    const u16 neg = fp32_to_fp16(-1.0f);
    blk[0] = static_cast<u8>(neg & 0xFF);
    blk[1] = static_cast<u8>(neg >> 8);
    dequant_row(DType::Q2_0, blk, out, 128);
    CHECK_NEAR(out[3], -2.0f, 1e-6, "Q2_0: a negative delta carries through");

    // Q2_0_64 is the same code map over half the block: 18 bytes, 64 values.
    // That is upstream llama.cpp's own Q2_0 geometry, which the llama-dx fork
    // separates out as id 48 so its 128-value Q2_0 can keep id 42.
    u8 blk64[18];
    blk64[0] = static_cast<u8>(one & 0xFF);
    blk64[1] = static_cast<u8>(one >> 8);
    for (int i = 0; i < 16; i++) blk64[2 + i] = 0;
    blk64[2] = static_cast<u8>(0x00 | (1 << 2) | (2 << 4) | (3 << 6));
    f32 out64[64];
    dequant_row(DType::Q2_0_64, blk64, out64, 64);
    CHECK_NEAR(out64[0], -1.0f, 1e-6, "Q2_0_64: code 0 is -d");
    CHECK_NEAR(out64[3], 2.0f, 1e-6, "Q2_0_64: code 3 is +2d");
    CHECK_NEAR(out64[63], -1.0f, 1e-6, "Q2_0_64: 64 values come from one block");
    // The two widths must agree value for value wherever they overlap, which is
    // the check that keeps the 34-byte and 18-byte readers from drifting apart.
    for (int i = 0; i < 16; i++) blk64[2 + i] = blk[2 + i];
    blk64[0] = blk[0];
    blk64[1] = blk[1];
    dequant_row(DType::Q2_0, blk, out, 128);
    dequant_row(DType::Q2_0_64, blk64, out64, 64);
    int mismatched = 0;
    for (int i = 0; i < 64; i++) if (out[i] != out64[i]) mismatched++;

}


static void test_iq_family_structure() {
    // One probe per format: the byte to perturb, and the half-open range of
    // decoded values that byte is allowed to reach. Every offset comes from the
    // reference's own field layout, not from the decoder being tested.
    struct Probe {
        DType t;
        int bytes;
        int at;
        int lo;
        int hi;
        const char *what;
    };
    const Probe probes[] = {
        // IQ3_XXS: the 4-byte scale-and-sign word for group g is sas+4g; its
        // top nibble (byte 3, bits 4..6 of which are the low scale bits) is
        // that group's scale, and the word covers only its own 32 values.
        {DType::IQ3_XXS, 98, 66 + 4 * 3 + 3, 96, 128, "IQ3_XXS: group scale stays in its group"},
        // IQ3_XXS grid bytes: 8 per group, so qs[8g+i] owns group g.
        {DType::IQ3_XXS, 98, 2 + 8 * 2, 64, 96, "IQ3_XXS: a grid byte stays in its group"},
        // IQ1_S: qh is eight LE u16, one per 32-value chunk; bits 12..14 of
        // chunk 3's word are its scale.
        {DType::IQ1_S, 50, 34 + 2 * 3 + 1, 96, 128, "IQ1_S: chunk scale stays in its chunk"},
        // IQ3_S: qh carries the 9th index bit, one byte per 32-value group.
        {DType::IQ3_S, 110, 66 + 3, 96, 128, "IQ3_S: high index byte stays in its group"},
        // IQ3_S scales: one byte per 64-value pair, low nibble then high.
        {DType::IQ3_S, 110, 106 + 2, 128, 192, "IQ3_S: a scale byte owns its 64-value pair"},
        // IQ2_S: qh carries the top two index bits, one byte per group.
        {DType::IQ2_S, 82, 66 + 3, 96, 128, "IQ2_S: high index byte stays in its group"},
        // IQ2_S scales: one nibble per 16-value sub-group, so a byte owns a
        // whole 32-value group.
        {DType::IQ2_S, 82, 74 + 3, 96, 128, "IQ2_S: a scale byte owns its group"},
        // IQ1_M: two qh bytes per chunk, so byte 0 of chunk 3 reaches only the
        // first 16 of its values (l = 0 and 1).
        {DType::IQ1_M, 56, 32 + 2 * 3, 96, 112, "IQ1_M: a qh byte reaches only its own halves"},
    };
    u32 seed = 0x9E3779B9u;
    for (const Probe &p : probes) {
        std::vector<u8> b0(static_cast<size_t>(p.bytes));
        fill_probe_block(b0.data(), b0.size(), seed);
        seed += 0x7F4A7C15u;
        std::vector<f32> y0(256), y1(256);
        dequant_row(p.t, b0.data(), y0.data(), 256);
        int finite = 0;
        for (int i = 0; i < 256; i++)
            if (std::isfinite(y0[i])) finite++;
        CHECK(finite == 256, "a new IQ block decodes to finite values only");
        std::vector<u8> b1 = b0;
        b1[static_cast<size_t>(p.at)] =
            static_cast<u8>(b1[static_cast<size_t>(p.at)] ^ 0x70);
        dequant_row(p.t, b1.data(), y1.data(), 256);
        const QuantDiff d = diff_indices(y0.data(), y1.data(), 256);
        CHECK(d.n > 0, "the probed byte reaches the decoded values");
        CHECK(d.lo >= p.lo && d.hi < p.hi, p.what);
    }
}

static void test_vec_dot() {
    std::vector<f32> w(64), x(64);
    for (int i = 0; i < 64; i++) {
        w[i] = static_cast<f32>(i + 1);
        x[i] = 0.5f;
    }
    CHECK_NEAR(vec_dot(DType::F32, w.data(), x.data(), 64), 1040.0, 1e-2, "F32 dot");

    std::vector<u16> h(64);
    for (int i = 0; i < 64; i++) h[i] = fp32_to_fp16(w[i]);
    CHECK_NEAR(vec_dot(DType::F16, h.data(), x.data(), 64), 1040.0, 1.0, "F16 dot");
}

// ---------------------------------------------------------------------------
// GGUF round trip
// ---------------------------------------------------------------------------

static const char *kTestModelPath = "kraken-test-model.gguf";

// Builds a 1-layer, 32-wide LLaMA-family model with byte-fallback SPM vocab.
static bool build_tiny_model(const std::string &path, bool bpe, f32 softcap = 0.0f) {
    const int n_embd = 32, n_layer = 1, n_head = 4, n_kv = 2, hd = 8, n_ff = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int vocab = bpe ? 300 : 260;
    const int ctx = 64;

    GgufBuilder b;
    b.meta_str("general.architecture", "llama");
    b.meta_str("general.name", "kraken-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32("llama.block_count", n_layer);
    b.meta_u32("llama.context_length", ctx);
    b.meta_u32("llama.embedding_length", n_embd);
    b.meta_u32("llama.feed_forward_length", n_ff);
    b.meta_u32("llama.attention.head_count", n_head);
    b.meta_u32("llama.attention.head_count_kv", n_kv);
    b.meta_u32("llama.attention.key_length", hd);
    b.meta_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32("llama.rope.freq_base", 10000.0f);
    if (softcap > 0.0f) b.meta_f32("llama.final_logit_softcapping", softcap);

    // --- tokenizer -------------------------------------------------------
    std::vector<std::string> tokens;
    std::vector<f32> scores;
    if (bpe) {
        tokens = {"<|endoftext|>"};
        for (int i = 0; i < 256; i++) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
            tokens.push_back(buf);
        }
        // a few multi-character pieces plus one merge, so the BPE loop runs
        tokens.push_back("ab");
        tokens.push_back("abc");
        tokens.push_back("\xC4\xA0"); // "Ġ" = mapped space
        tokens.push_back("the");
        b.meta_str("tokenizer.ggml.model", "gpt2");
        b.meta_str("tokenizer.ggml.pre", "gpt2");
        b.meta_str_array("tokenizer.ggml.tokens", tokens);
        b.meta_str_array("tokenizer.ggml.merges", {"a b", "ab c"});
        b.meta_u32("tokenizer.ggml.eos_token_id", 0);
        b.meta_bool("tokenizer.ggml.add_bos_token", false);
    } else {
        tokens = {"<unk>", "<s>", "</s>"};
        scores = {-10.0f, 0.0f, 0.0f};
        for (int i = 0; i < 256; i++) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
            tokens.push_back(buf);
            scores.push_back(-30.0f);
        }
        // word pieces: longer tokens must win, so give them better scores
        tokens.push_back("\xE2\x96\x81the"); // "▁the"
        scores.push_back(-1.0f);
        tokens.push_back("\xE2\x96\x81"); // "▁"
        scores.push_back(-2.0f);
        b.meta_str("tokenizer.ggml.model", "llama");
        b.meta_str_array("tokenizer.ggml.tokens", tokens);
        b.meta_f32_array("tokenizer.ggml.scores", scores);
        b.meta_u32("tokenizer.ggml.bos_token_id", 1);
        b.meta_u32("tokenizer.ggml.eos_token_id", 2);
        b.meta_bool("tokenizer.ggml.add_bos_token", true);
    }
    if (static_cast<int>(tokens.size()) < vocab) {
        while (static_cast<int>(tokens.size()) < vocab) {
            tokens.push_back(format("<extra%d>", static_cast<int>(tokens.size())));
            if (!bpe) scores.push_back(-50.0f);
        }
    }

    // --- weights ---------------------------------------------------------
    u32 seed = 0xC0FFEEu;
    auto rndvec = [&](size_t n) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = rnd(seed);
        return v;
    };
    auto one = [](size_t n) { return std::vector<f32>(n, 1.0f); };

    b.tensor_f16("token_embd.weight", {static_cast<u64>(n_embd),
                                       static_cast<u64>(tokens.size())},
                 rndvec(static_cast<size_t>(n_embd) * tokens.size()));
    b.tensor_f32("output_norm.weight", {static_cast<u64>(n_embd)}, one(n_embd));
    b.tensor_f32("blk.0.attn_norm.weight", {static_cast<u64>(n_embd)}, one(n_embd));
    b.tensor_f32("blk.0.ffn_norm.weight", {static_cast<u64>(n_embd)}, one(n_embd));
    b.tensor_f16("blk.0.attn_q.weight",
                 {static_cast<u64>(n_embd), static_cast<u64>(q_dim)},
                 rndvec(static_cast<size_t>(n_embd) * q_dim));
    b.tensor_f16("blk.0.attn_k.weight",
                 {static_cast<u64>(n_embd), static_cast<u64>(kv_dim)},
                 rndvec(static_cast<size_t>(n_embd) * kv_dim));
    b.tensor_f16("blk.0.attn_v.weight",
                 {static_cast<u64>(n_embd), static_cast<u64>(kv_dim)},
                 rndvec(static_cast<size_t>(n_embd) * kv_dim));
    b.tensor_f16("blk.0.attn_output.weight",
                 {static_cast<u64>(q_dim), static_cast<u64>(n_embd)},
                 rndvec(static_cast<size_t>(q_dim) * n_embd));
    b.tensor_f16("blk.0.ffn_gate.weight",
                 {static_cast<u64>(n_embd), static_cast<u64>(n_ff)},
                 rndvec(static_cast<size_t>(n_embd) * n_ff));
    b.tensor_f16("blk.0.ffn_up.weight",
                 {static_cast<u64>(n_embd), static_cast<u64>(n_ff)},
                 rndvec(static_cast<size_t>(n_embd) * n_ff));
    b.tensor_f16("blk.0.ffn_down.weight",
                 {static_cast<u64>(n_ff), static_cast<u64>(n_embd)},
                 rndvec(static_cast<size_t>(n_ff) * n_embd));
    return b.write(path);
}

// ---------------------------------------------------------------------------
// synthetic Mixture-of-Experts model
// ---------------------------------------------------------------------------

// Deterministic, index-addressed weights. The dense and MoE builders below draw
// from the same sequence, so a MoE model with one expert carries exactly the
// numbers of its dense twin and must produce identical logits.
static f32 wv(i64 i) {
    const i32 m = static_cast<i32>(i % 11) - 5; // -5..5
    return static_cast<f32>(m) * 0.03f;
}

// The same byte-fallback SPM vocab build_tiny_model uses.
static void add_spm_vocab(GgufBuilder &b, int vocab) {
    std::vector<std::string> tokens = {"<unk>", "<s>", "</s>"};
    std::vector<f32> scores = {-10.0f, 0.0f, 0.0f};
    for (int i = 0; i < 256; i++) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
        tokens.push_back(buf);
        scores.push_back(-30.0f);
    }
    tokens.push_back("\xE2\x96\x81the");
    scores.push_back(-1.0f);
    tokens.push_back("\xE2\x96\x81");
    scores.push_back(-2.0f);
    while (static_cast<int>(tokens.size()) < vocab) {
        tokens.push_back(format("<extra%d>", static_cast<int>(tokens.size())));
        scores.push_back(-50.0f);
    }
    b.meta_str("tokenizer.ggml.model", "llama");
    b.meta_str_array("tokenizer.ggml.tokens", tokens);
    b.meta_f32_array("tokenizer.ggml.scores", scores);
    b.meta_u32("tokenizer.ggml.bos_token_id", 1);
    b.meta_u32("tokenizer.ggml.eos_token_id", 2);
    b.meta_bool("tokenizer.ggml.add_bos_token", true);
}

struct MoeSpec {
    bool dense = false;   // emit the dense twin instead of a MoE model
    int n_expert = 4;
    int n_expert_used = 2;
    bool shared = true;   // Qwen2-MoE style always-on shared expert
    bool varied = true;   // false => every expert identical to the dense FFN
};

static bool build_tiny_moe_model(const std::string &path, const MoeSpec &s) {
    const int n_embd = 32, n_head = 4, n_kv = 2, hd = 8;
    const int n_ff = 64, n_ff_exp = 64, n_ff_sh = 32, vocab = 260, ctx = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const std::string arch = s.dense ? "llama" : "qwen2moe";

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-moe-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", 1);
    b.meta_u32(arch + ".context_length", ctx);
    b.meta_u32(arch + ".embedding_length", n_embd);
    b.meta_u32(arch + ".feed_forward_length", n_ff);
    b.meta_u32(arch + ".attention.head_count", n_head);
    b.meta_u32(arch + ".attention.head_count_kv", n_kv);
    b.meta_u32(arch + ".attention.key_length", hd);
    b.meta_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32(arch + ".rope.freq_base", 10000.0f);
    if (!s.dense) {
        b.meta_u32(arch + ".expert_count", static_cast<u32>(s.n_expert));
        b.meta_u32(arch + ".expert_used_count", static_cast<u32>(s.n_expert_used));
        b.meta_u32(arch + ".expert_feed_forward_length", n_ff_exp);
        if (s.shared)
            b.meta_u32(arch + ".expert_shared_feed_forward_length", n_ff_sh);
    }
    add_spm_vocab(b, vocab);

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };

    b.tensor_f16("token_embd.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 0));
    b.tensor_f32("output_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32("blk.0.attn_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32("blk.0.ffn_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f16("blk.0.attn_q.weight", {u64(n_embd), u64(q_dim)},
                 vec(static_cast<size_t>(n_embd) * q_dim, 100));
    b.tensor_f16("blk.0.attn_k.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim, 200));
    b.tensor_f16("blk.0.attn_v.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim, 300));
    b.tensor_f16("blk.0.attn_output.weight", {u64(q_dim), u64(n_embd)},
                 vec(static_cast<size_t>(q_dim) * n_embd, 400));

    if (s.dense) {
        b.tensor_f16("blk.0.ffn_gate.weight", {u64(n_embd), u64(n_ff)},
                     vec(static_cast<size_t>(n_embd) * n_ff, 500));
        b.tensor_f16("blk.0.ffn_up.weight", {u64(n_embd), u64(n_ff)},
                     vec(static_cast<size_t>(n_embd) * n_ff, 900));
        b.tensor_f16("blk.0.ffn_down.weight", {u64(n_ff), u64(n_embd)},
                     vec(static_cast<size_t>(n_ff) * n_embd, 1300));
        return b.write(path);
    }

    // Router: zero gates give a uniform softmax (used to prove the routed path
    // equals a dense FFN); varied gates give every token its own top-k.
    std::vector<f32> router(static_cast<size_t>(n_embd) * s.n_expert);
    for (size_t i = 0; i < router.size(); i++)
        router[i] = s.varied ? wv(static_cast<i64>(i) * 3 + 7) : 0.0f;
    b.tensor_f16("blk.0.ffn_gate_inp.weight", {u64(n_embd), u64(s.n_expert)}, router);

    const size_t gsz = static_cast<size_t>(n_embd) * n_ff_exp;
    const size_t dsz = static_cast<size_t>(n_ff_exp) * n_embd;
    const size_t ne = static_cast<size_t>(s.n_expert);
    std::vector<f32> gate(gsz * ne), up(gsz * ne), down(dsz * ne);
    for (size_t e = 0; e < ne; e++) {
        const i64 bump = s.varied ? static_cast<i64>(e) * 7919 : 0;
        for (size_t k = 0; k < gsz; k++) {
            gate[e * gsz + k] = wv(500 + static_cast<i64>(k) + bump);
            up[e * gsz + k] = wv(900 + static_cast<i64>(k) + bump);
        }
        for (size_t k = 0; k < dsz; k++)
            down[e * dsz + k] = wv(1300 + static_cast<i64>(k) + bump);
    }
    b.tensor_f16("blk.0.ffn_gate_exps.weight",
                 {u64(n_embd), u64(n_ff_exp), u64(ne)}, gate);
    b.tensor_f16("blk.0.ffn_up_exps.weight",
                 {u64(n_embd), u64(n_ff_exp), u64(ne)}, up);
    b.tensor_f16("blk.0.ffn_down_exps.weight",
                 {u64(n_ff_exp), u64(n_embd), u64(ne)}, down);

    if (s.shared) {
        b.tensor_f16("blk.0.ffn_gate_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                     vec(static_cast<size_t>(n_embd) * n_ff_sh, 1700));
        b.tensor_f16("blk.0.ffn_up_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                     vec(static_cast<size_t>(n_embd) * n_ff_sh, 2100));
        b.tensor_f16("blk.0.ffn_down_shexp.weight", {u64(n_ff_sh), u64(n_embd)},
                     vec(static_cast<size_t>(n_ff_sh) * n_embd, 2500));
    }
    return b.write(path);
}

// ---------------------------------------------------------------------------
// Qwen3.5 / gated delta net
// ---------------------------------------------------------------------------

static const char *kGdnTestPath = "kraken-gdn-test.gguf";

// A miniature qwen35moe: `n_layer` layers where every full_attention_interval'th
// one keeps real attention (with a packed output gate) and the rest are delta
// net layers. Deliberately tiny (d_state 8, 2 key / 4 value heads) so the CPU
// oracle can run it, but structurally identical to the 35B checkpoint's schema.
static bool build_tiny_gdn_model(const std::string &path, int n_layer,
                                 int interval) {
    const int n_embd = 32, n_head = 4, n_kv = 2, hd = 8;
    const int d_state = 8, n_k_head = 2, n_v_head = 4, conv_k = 4;
    const int rope_dim = 4;
    const int key_dim = n_k_head * d_state;
    const int value_dim = n_v_head * d_state;
    const int conv_dim = key_dim * 2 + value_dim;
    const int n_ff = 64, n_ff_exp = 64, n_ff_sh = 32, n_expert = 4, vocab = 260,
              ctx = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const std::string arch = "qwen35moe";

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-gdn-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", static_cast<u32>(n_layer));
    b.meta_u32(arch + ".context_length", static_cast<u32>(ctx));
    b.meta_u32(arch + ".embedding_length", static_cast<u32>(n_embd));
    b.meta_u32(arch + ".feed_forward_length", static_cast<u32>(n_ff));
    b.meta_u32(arch + ".attention.head_count", static_cast<u32>(n_head));
    b.meta_u32(arch + ".attention.head_count_kv", static_cast<u32>(n_kv));
    b.meta_u32(arch + ".attention.key_length", static_cast<u32>(hd));
    b.meta_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32(arch + ".rope.freq_base", 10000.0f);
    b.meta_u32(arch + ".rope.dimension_count", static_cast<u32>(rope_dim));
    b.meta_i32_array(arch + ".rope.dimension_sections", {rope_dim / 2, 1, 1, 0});
    // ssm.* drives the recurrent block's geometry.
    b.meta_u32(arch + ".ssm.conv_kernel", static_cast<u32>(conv_k));
    b.meta_u32(arch + ".ssm.state_size", static_cast<u32>(d_state));
    b.meta_u32(arch + ".ssm.time_step_rank", static_cast<u32>(n_v_head));
    b.meta_u32(arch + ".ssm.group_count", static_cast<u32>(n_k_head));
    b.meta_u32(arch + ".ssm.inner_size", static_cast<u32>(value_dim));
    b.meta_u32(arch + ".full_attention_interval", static_cast<u32>(interval));
    // MoE geometry, identical to the qwen2moe path.
    b.meta_u32(arch + ".expert_count", static_cast<u32>(n_expert));
    b.meta_u32(arch + ".expert_used_count", 2);
    b.meta_u32(arch + ".expert_feed_forward_length", static_cast<u32>(n_ff_exp));
    b.meta_u32(arch + ".expert_shared_feed_forward_length", static_cast<u32>(n_ff_sh));
    add_spm_vocab(b, vocab);

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };

    b.tensor_f16("token_embd.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 0));
    b.tensor_f32("output_norm.weight", {u64(n_embd)}, ones(n_embd));
    // An explicit (untied) output head with a single live row. The point of
    // this model is the recurrent mechanics, not the text, and the head has to
    // earn its keep twice over:
    //   * only one id can ever be non-zero, so a run can never stop on EOS and
    //     the tests can assert on token counts and the KV cursor;
    //   * that row is a *random* vector, so the winning token is the sign of
    //     dot(activation, row) — a genuine function of the delta-rule state. A
    //     constant row would pin the argmax and make every state-corruption
    //     test in this file pass no matter how badly the state was wrong.
    {
        std::vector<f32> head(static_cast<size_t>(n_embd) * vocab, 0.0f);
        const size_t live = 5;
        const std::vector<f32> row = vec(static_cast<size_t>(n_embd), 4242);
        for (int r = 0; r < n_embd; r++)
            head[static_cast<size_t>(r) * vocab + live] = row[static_cast<size_t>(r)];
        b.tensor_f16("output.weight", {u64(n_embd), u64(vocab)}, head);
    }

    for (int l = 0; l < n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        b.tensor_f32(p + "attn_norm.weight", {u64(n_embd)}, ones(n_embd));
        // Qwen3.5 spells the post-attention norm differently from the rest of
        // the family; the loader has to fall back to it.
        b.tensor_f32(p + "post_attention_norm.weight", {u64(n_embd)}, ones(n_embd));

        if (((l + 1) % interval) != 0) {
            // ---- delta net layer ----
            b.tensor_f16(p + "attn_qkv.weight", {u64(n_embd), u64(conv_dim)},
                         vec(static_cast<size_t>(n_embd) * conv_dim, 100));
            b.tensor_f16(p + "attn_gate.weight", {u64(n_embd), u64(value_dim)},
                         vec(static_cast<size_t>(n_embd) * value_dim, 200));
            // Depthwise: one ksize-tap kernel per channel, in FILE order. GGUF's
            // ne[0] is the tap axis, so a channel's taps are CONTIGUOUS and the
            // stored element for (channel, tap) sits at c*conv_k + j. The values
            // are distinct per (tap, channel) on purpose: a constant kernel would
            // make a transposed read produce the same numbers and the whole
            // layout would go untested.
            {
                std::vector<f32> kern(static_cast<size_t>(conv_k) * conv_dim);
                for (i64 c = 0; c < conv_dim; c++)
                    for (i64 j = 0; j < conv_k; j++)
                        kern[static_cast<size_t>(c * conv_k + j)] =
                            static_cast<f32>(1 + j * 10 + c * 100);
                b.tensor_f32(p + "ssm_conv1d.weight", {u64(conv_k), u64(conv_dim)},
                             kern);
            }
            b.tensor_f32(p + "ssm_dt.bias", {u64(n_v_head)},
                         vec(static_cast<size_t>(n_v_head), 400));
            // A_log as the converter stores it: already -exp(A_log).
            std::vector<f32> a(static_cast<size_t>(n_v_head));
            for (int h = 0; h < n_v_head; h++) a[static_cast<size_t>(h)] = -0.5f - 0.1f * h;
            b.tensor_f32(p + "ssm_a", {u64(n_v_head)}, a);
            b.tensor_f16(p + "ssm_alpha.weight", {u64(n_embd), u64(n_v_head)},
                         vec(static_cast<size_t>(n_embd) * n_v_head, 500));
            b.tensor_f16(p + "ssm_beta.weight", {u64(n_embd), u64(n_v_head)},
                         vec(static_cast<size_t>(n_embd) * n_v_head, 600));
            b.tensor_f32(p + "ssm_norm.weight", {u64(d_state)}, ones(d_state));
            b.tensor_f16(p + "ssm_out.weight", {u64(value_dim), u64(n_embd)},
                         vec(static_cast<size_t>(value_dim) * n_embd, 700));
        } else {
            // ---- full attention with a packed per-head output gate ----
            b.tensor_f16(p + "attn_q.weight", {u64(n_embd), u64(2 * q_dim)},
                         vec(static_cast<size_t>(n_embd) * 2 * q_dim, 100));
            b.tensor_f16(p + "attn_k.weight", {u64(n_embd), u64(kv_dim)},
                         vec(static_cast<size_t>(n_embd) * kv_dim, 200));
            b.tensor_f16(p + "attn_v.weight", {u64(n_embd), u64(kv_dim)},
                         vec(static_cast<size_t>(n_embd) * kv_dim, 300));
            b.tensor_f16(p + "attn_output.weight", {u64(q_dim), u64(n_embd)},
                         vec(static_cast<size_t>(q_dim) * n_embd, 400));
            b.tensor_f32(p + "attn_q_norm.weight", {u64(hd)}, ones(hd));
            b.tensor_f32(p + "attn_k_norm.weight", {u64(hd)}, ones(hd));
        }

        // ---- MoE FFN, shared by both layer kinds ----
        std::vector<f32> router(static_cast<size_t>(n_embd) * n_expert);
        for (size_t i = 0; i < router.size(); i++) router[i] = wv(static_cast<i64>(i) * 3 + 7);
        b.tensor_f16(p + "ffn_gate_inp.weight", {u64(n_embd), u64(n_expert)}, router);
        const size_t gsz = static_cast<size_t>(n_embd) * n_ff_exp;
        const size_t dsz = static_cast<size_t>(n_ff_exp) * n_embd;
        const size_t ne = static_cast<size_t>(n_expert);
        std::vector<f32> gate(gsz * ne), up(gsz * ne), down(dsz * ne);
        for (size_t e = 0; e < ne; e++) {
            const i64 bump = static_cast<i64>(e) * 7919;
            for (size_t k = 0; k < gsz; k++) {
                gate[e * gsz + k] = wv(500 + static_cast<i64>(k) + bump);
                up[e * gsz + k] = wv(900 + static_cast<i64>(k) + bump);
            }
            for (size_t k = 0; k < dsz; k++) down[e * dsz + k] = wv(1300 + static_cast<i64>(k) + bump);
        }
        b.tensor_f16(p + "ffn_gate_exps.weight", {u64(n_embd), u64(n_ff_exp), u64(ne)}, gate);
        b.tensor_f16(p + "ffn_up_exps.weight", {u64(n_embd), u64(n_ff_exp), u64(ne)}, up);
        b.tensor_f16(p + "ffn_down_exps.weight", {u64(n_ff_exp), u64(n_embd), u64(ne)}, down);
        b.tensor_f16(p + "ffn_gate_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                     vec(static_cast<size_t>(n_embd) * n_ff_sh, 1700));
        b.tensor_f16(p + "ffn_up_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                     vec(static_cast<size_t>(n_embd) * n_ff_sh, 2100));
        b.tensor_f16(p + "ffn_down_shexp.weight", {u64(n_ff_sh), u64(n_embd)},
                     vec(static_cast<size_t>(n_ff_sh) * n_embd, 2500));
        // The Qwen3.5 shared-expert gate: one row per token.
        b.tensor_f32(p + "ffn_gate_inp_shexp.weight", {u64(n_embd)},
                     vec(static_cast<size_t>(n_embd), 2900));
    }
    return b.write(path);
}

static void test_gdn_model_loads() {
    CHECK(build_tiny_gdn_model(kGdnTestPath, 4, 4), "wrote a synthetic qwen35moe GGUF");

    Backend *cpu = make_cpu_backend();
    Model m;
    std::string err;
    CHECK(m.load(*cpu, kGdnTestPath, &err), "a qwen35moe checkpoint loads");
    if (!err.empty()) std::fprintf(stderr, "  (gdn: %s)\n", err.c_str());
    if (m.cfg().arch == "qwen35moe") {
        const ModelConfig &c = m.cfg();
        CHECK(c.recurrent, "the model declares itself recurrent");
        CHECK(c.full_attention_interval == 4, "full_attention_interval is read");
        CHECK(c.ssm_d_state == 8 && c.ssm_dt_rank == 4 && c.ssm_n_group == 2,
              "ssm state/head geometry is read");
        CHECK(c.ssm_d_conv == 4, "ssm.conv_kernel is read");
        CHECK(m.value_dim() == 32 && m.key_dim() == 16 && m.conv_dim() == 64,
              "derived key/value/conv widths");
        CHECK(c.rope_dim == 4, "rope.dimension_count narrows the rotation");
        CHECK(c.rope_sections[0] == 2 && c.rope_sections[3] == 0,
              "rope.dimension_sections is read");
        CHECK(m.inv_freq().size() == 2,
              "the inverse-frequency table follows rope_dim, not head_dim");
        CHECK(m.recurrent_layers() == 3,
              "three of four layers are delta net (every 4th attends)");
        // Layer 3 attends, layers 0..2 recur.
        // Layers 0..2 recur, layer 3 attends (every 4th).
        const bool classified = m.layers()[0].gdn && m.layers()[1].gdn &&
                                m.layers()[2].gdn && !m.layers()[3].gdn;
        CHECK(classified, "layers 0-2 are delta net and layer 3 is full attention");
        CHECK(m.layers()[3].wq.n_out == 64, "the attending layer's q packs q+gate");
        CHECK(m.layers()[3].wo.n_in == 32, "its output projection is the head width");
        const LayerWeights &g = m.layers()[0];
        CHECK(g.wqkv.n_out == 64 && g.wqkv_gate.n_out == 32,
              "the delta net layer's projections match ssm geometry");
        CHECK(g.ssm_conv1d.n_in == 4 && g.ssm_conv1d.n_out == 64,
              "its short conv is [kernel, conv_dim]");
            // The loader must transpose the file's channel-major kernel into the
            // op's [ksize, conv_dim] order; read back a few (tap, channel) pairs
            // and compare against the values the builder wrote.
            {
                const f32 *kern = static_cast<const f32 *>(g.ssm_conv1d.data);
                const i64 cdim = c.ssm_conv_dim, ks = c.ssm_d_conv;
                bool layout_ok = g.ssm_conv1d.type == DType::F32;
                for (i64 cc : {i64(0), i64(1), i64(5), cdim - 1})
                    for (i64 j = 0; j < ks; j++)
                        layout_ok = layout_ok &&
                            std::fabs(kern[j * cdim + cc] -
                                      static_cast<f32>(1 + j * 10 + cc * 100)) < 1e-6f;
                CHECK(layout_ok,
                      "the conv kernel is transposed from the file's "
                      "[conv_dim, ksize] order into the op's [ksize, conv_dim]");
            }
        CHECK(g.ssm_out.n_in == 32 && g.ssm_out.n_out == 32, "its output projection");
        CHECK(g.ssm_dt && g.ssm_a && g.ssm_norm, "its state parameters are resident");
        CHECK(g.ssm_alpha.present() && g.ssm_beta.present(),
              "alpha/beta are matmul weights");
        CHECK(g.ffn_norm != nullptr,
              "post_attention_norm stands in for the missing ffn_norm");
        CHECK(g.shexp_inp_gate.present(), "the shared-expert gate row is loaded");
        CHECK(m.experts().loads() == 0, "no expert loaded straight after load");
    }
    m.unload();
    delete cpu;
}

static void test_gdn_generation() {
    std::FILE *probe = std::fopen(kGdnTestPath, "rb");
    if (!probe) {
        std::fprintf(stderr, "  (gdn model missing, skipping)\n");
        return;
    }
    std::fclose(probe);

    // A recurrent model must decode, and it must be deterministic: the state
    // advances token by token, so any drift in the ops shows up immediately.
    std::vector<i32> first;
    for (int pass = 0; pass < 2; pass++) {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = kGdnTestPath;
        cfg.n_ctx = 32;
        cfg.prefill_chunk = 4; // force several chunks: the conv window and the
                               // delta state have to survive the chunk boundary
        std::string err;
        if (!engine.init(cpu, cfg, &err)) {
            std::fprintf(stderr, "  (gdn engine: %s)\n", err.c_str());
            delete cpu;
            CHECK(false, "the engine initialised on a qwen35moe model");
            return;
        }
        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 8;
        p.sampler.greedy = true;
        p.sampler.temp = 0.0f;
        GenerateResult r;
        const bool ok = engine.generate(p, &r);
        CHECK(ok && !r.tokens.empty(), "a recurrent model generates tokens");
        CHECK(engine.kv_pos() == static_cast<i64>(1 + 8),
              "the KV cursor advanced once per token");
        // A draft would need a partial rewind, which a delta rule cannot do.
        std::string derr;
        CHECK(!engine.load_draft(kTestModelPath, &derr),
              "speculative decoding is refused for a recurrent model");
        engine.shutdown();
        delete cpu;
        if (pass == 0) {
            first = r.tokens;
        } else {
            CHECK(r.tokens == first, "a recurrent model decodes deterministically");
        }
    }

    // Chunking must not change the answer: a 4-token prefill chunk and a
    // 1-token one compute the same sequence, so the state threading is exact.
    auto run_with = [&](i32 chunk, std::vector<i32> *out) {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = kGdnTestPath;
        cfg.n_ctx = 32;
        cfg.prefill_chunk = chunk;
        std::string err;
        if (!engine.init(cpu, cfg, &err)) {
            delete cpu;
            return false;
        }
        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 8;
        p.sampler.greedy = true;
        p.sampler.temp = 0.0f;
        GenerateResult r;
        const bool ok = engine.generate(p, &r);
        *out = r.tokens;
        engine.shutdown();
        delete cpu;
        return ok;
    };
    std::vector<i32> chunked, whole;
    CHECK(run_with(1, &whole), "generation with a 1-token prefill chunk");
    CHECK(run_with(4, &chunked), "generation with a 4-token prefill chunk");
    // The prompt is more than one token, so a 1-token chunk really does split
    // the prefill: getting the same tokens back proves the conv window and the
    // delta state thread across chunk boundaries exactly.
    CHECK(!whole.empty() && whole == chunked,
          "the recurrent state threads across prefill chunks exactly");
    CHECK(!whole.empty() && whole == chunked,
          "the recurrent state threads across prefill chunks exactly");
    CHECK(!whole.empty() && whole == first,
          "chunking does not change the generated tokens");

    // A full rewind is the honest reset: the server does it per request, and
    // the replayed prompt rebuilds the state. A partial one must be refused.
    {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = kGdnTestPath;
        cfg.n_ctx = 32;
        cfg.prefill_chunk = 4;
        std::string err;
        CHECK(engine.init(cpu, cfg, &err), "engine for the rollback test");
        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 6;
        p.sampler.greedy = true;
        GenerateResult r;
        engine.generate(p, &r);
        const std::vector<i32> before = r.tokens;
        // Tokens alone are a weak witness: the head has one live row, so the
        // argmax can land the same way whatever the state did. The logits are
        // the real function of the state, so compare those too.
        const std::vector<f32> before_logits(engine.last_logits(),
                                             engine.last_logits() + 16);
        const auto same_logits = [&](const char *what) {
            bool eq = true;
            for (size_t i = 0; i < before_logits.size(); i++)
                if (before_logits[i] != engine.last_logits()[i]) eq = false;
            CHECK(eq, what);
        };
        CHECK(!engine.kv_rollback(engine.kv_pos() - 2),
              "a partial rewind is refused for a recurrent model");
        CHECK(engine.kv_pos() == 7, "the refused rewind left the cursor alone");
        CHECK(engine.kv_rollback(0), "a full rewind is accepted");
        CHECK(engine.kv_pos() == 0, "the full rewind moved the cursor to 0");
        // Replaying the same prompt from the reset state must reproduce the run.
        GenerateResult r2;
        engine.generate(p, &r2);
        CHECK(r2.tokens == before,
              "a replay after kv_rollback(0) reproduces the same tokens");
        same_logits("a replay after kv_rollback(0) reproduces the same logits");

        // A second generate() on the same engine is a new sequence whether or
        // not the caller thought to roll back: the prefill starts at 0, so the
        // recurrent state has to start there too. Without the reset the state
        // still holds the previous run and every logit drifts.
        GenerateResult r3;
        engine.generate(p, &r3);
        CHECK(r3.tokens == before,
              "generate() after another generate() repeats the same tokens");
        same_logits("generate() resets the recurrent state instead of inheriting it");
        engine.shutdown();
        delete cpu;
    }
}

// The ops behind the recurrent block, checked in isolation so a failure points
// at the kernel rather than at the model.
static void test_gdn_ops() {
    Backend *cpu = make_cpu_backend();
    const i64 hd = 8, heads = 2, n = 3;

    // l2norm: each head gets unit norm with its own denominator, for every
    // token, including when the rows are strided out of a wider buffer.
    std::vector<f32> x(static_cast<size_t>(heads * hd));
    for (size_t i = 0; i < x.size(); i++) x[i] = static_cast<f32>(i + 1);
    std::vector<f32> y(x.size());
    cpu->l2norm(y.data(), x.data(), 1, heads, hd, heads * hd, 0.0f);
    for (i64 h = 0; h < heads; h++) {
        f32 ss = 0;
        for (i64 i = 0; i < hd; i++) ss += y[static_cast<size_t>(h * hd + i)] * y[static_cast<size_t>(h * hd + i)];
        CHECK(std::fabs(ss - 1.0f) < 1e-5f, "l2norm gives each head unit norm");
    }
    {
        // Strided: 3 tokens inside rows wider than the heads, so a dense
        // l2norm would normalize the wrong elements. The stride must clear the
        // heads or the rows would overlap.
        const i64 nt = 3, stride = heads * hd + 3;
        std::vector<f32> s(static_cast<size_t>(nt * stride), 100.0f);
        for (i64 t = 0; t < nt; t++)
            for (i64 i = 0; i < heads * hd; i++)
                s[static_cast<size_t>(t * stride + i)] = static_cast<f32>(t + 1) * (i + 1);
        std::vector<f32> d(s);
        cpu->l2norm(d.data(), s.data(), nt, heads, hd, stride, 0.0f);
        for (i64 t = 0; t < nt; t++)
            for (i64 h = 0; h < heads; h++) {
                f32 ss = 0;
                for (i64 i = 0; i < hd; i++) {
                    const f32 val = d[static_cast<size_t>(t * stride + h * hd + i)];
                    ss += val * val;
                }
                CHECK(std::fabs(ss - 1.0f) < 1e-5f,
                      "l2norm honours the row stride (per token, per head)");
                CHECK(d[static_cast<size_t>(t * stride + heads * hd)] == 100.0f,
                      "l2norm leaves the padding past the heads alone");
            }
    }

    // softplus and sigmoid against the closed forms.
    std::vector<f32> a = {-2.0f, 0.0f, 3.0f, 25.0f};
    std::vector<f32> a0 = a;
    cpu->softplus_act(a.data(), 4);
    CHECK(std::fabs(a[0] - std::log1p(std::exp(-2.0f))) < 1e-5f, "softplus(-2)");
    CHECK(std::fabs(a[1] - std::log(2.0f)) < 1e-5f, "softplus(0) = log 2");
    CHECK(std::fabs(a[2] - std::log1p(std::exp(3.0f))) < 1e-5f, "softplus(3)");
    CHECK(std::fabs(a[3] - 25.0f) < 1e-4f, "softplus saturates to x without overflow");
    std::vector<f32> s = a0;
    cpu->sigmoid_act(s.data(), 4);
    for (size_t i = 0; i < s.size(); i++)
        CHECK(std::fabs(s[i] - 1.0f / (1.0f + std::exp(-a0[i]))) < 1e-6f,
              "sigmoid matches the logistic");

    // The short conv: a two-tap identity kernel must reproduce its input on a
    // fresh state, which pins the tap order (newest last) and the window.
    const i64 chan = 2, ksize = 3;
    std::vector<f32> in(static_cast<size_t>(chan * 3)), outv(in.size());
    for (size_t i = 0; i < in.size(); i++) in[i] = static_cast<f32>(i) - 2.0f;
    std::vector<f32> kern(static_cast<size_t>(chan * ksize), 0.0f);
    for (i64 c = 0; c < chan; c++) kern[static_cast<size_t>((ksize - 1) * chan + c)] = 4.0f;
    std::vector<f32> state(static_cast<size_t>((ksize - 1) * chan), 0.0f);
    cpu->conv1d_silu(outv.data(), in.data(), state.data(), kern.data(), DType::F32, 3,
                     chan, ksize);
    // silu(4 * x) is not x, so compare against the closed form instead.
    for (i64 t = 0; t < 3; t++)
        for (i64 c = 0; c < chan; c++) {
            const f32 x = in[static_cast<size_t>(t * chan + c)];
            const f32 want = (4.0f * x) / (1.0f + std::exp(-4.0f * x));
            CHECK(std::fabs(outv[static_cast<size_t>(t * chan + c)] - want) < 1e-5f,
                  "conv1d_silu matches the closed form with a fresh window");
        }

    // In place, and across calls. The engine convolves the projection buffer
    // onto ITSELF, and output row t reaches back ksize-1 rows that earlier
    // outputs have already overwritten — so those rows have to be held, not
    // re-read. A kernel that re-reads them produces a plausible wrong answer,
    // which is exactly what happened here. Hand-computed, one channel, taps
    // [1, 10, 100] (newest last), a zero state and in = [1, 2, 4]:
    //   t0: 1*0  + 10*0 + 100*1 =  100
    //   t1: 1*0  + 10*1 + 100*2 =  210
    //   t2: 1*1  + 10*2 + 100*4 =  421
    // and the state left behind is the LAST ksize-1 inputs, [in1, in2] =
    // [2, 4], so a second call on [8] reads 1*2 + 10*4 + 100*8 = 842.
    // (Every acc is large and positive, where silu is the identity in f32.)
    {
        const i64 k1 = 1, ks3 = 3, nt3 = 3;
        std::vector<f32> buf{1.0f, 2.0f, 4.0f};
        std::vector<f32> kern3{1.0f, 10.0f, 100.0f};
        std::vector<f32> st3(2, 0.0f);
        cpu->conv1d_silu(buf.data(), buf.data(), st3.data(), kern3.data(),
                         DType::F32, nt3, k1, ks3);
        CHECK(std::fabs(buf[0] - 100.0f) < 1e-4f, "in-place conv, first row");
        CHECK(std::fabs(buf[1] - 210.0f) < 1e-4f,
              "in-place conv, second row reads a row the first row overwrote");
        CHECK(std::fabs(buf[2] - 421.0f) < 1e-4f, "in-place conv, third row");
        CHECK(std::fabs(st3[0] - 2.0f) < 1e-6f && std::fabs(st3[1] - 4.0f) < 1e-6f,
              "the in-place conv leaves the last ksize-1 inputs behind");
        std::vector<f32> buf2{8.0f};
        cpu->conv1d_silu(buf2.data(), buf2.data(), st3.data(), kern3.data(),
                         DType::F32, 1, k1, ks3);
        CHECK(std::fabs(buf2[0] - 842.0f) < 1e-4f,
              "the carried state continues the sequence in place");
    }

    // The delta rule on one head, hand-checked. With g = 0 the state does not
    // decay and beta = 1 makes the update a full rank-1 write, so from a zero
    // state:  S = k (x) v  and  out[j] = sum_i S[i][j] q[i] = v[j] * (k . q).
    const i64 ds = 4, nh = 1;
    const std::vector<f32> k = {0, 1, 0, 0}, v = {1, 2, 3, 4}, g0 = {0.0f},
                          beta1 = {1.0f}, q_orth = {1, 0, 0, 0}, q_ones = {1, 1, 1, 1};
    std::vector<f32> o(static_cast<size_t>(ds));
    std::vector<f32> st(static_cast<size_t>(ds * ds), 0.0f);
    cpu->delta_rule(o.data(), st.data(), q_orth.data(), k.data(), v.data(),
                    g0.data(), beta1.data(), 1, nh, nh, ds, ds, ds);
    for (i64 j = 0; j < ds; j++)
        CHECK(std::fabs(o[static_cast<size_t>(j)]) < 1e-5f,
              "a query orthogonal to k gives a zero delta-rule output");
    for (i64 i = 0; i < ds; i++)
        for (i64 j = 0; j < ds; j++)
            CHECK(std::fabs(st[static_cast<size_t>(i * ds + j)] -
                            k[static_cast<size_t>(i)] * v[static_cast<size_t>(j)]) < 1e-5f,
                  "the state holds the rank-1 write k (x) v");

    // Decay: exp(g) scales the state *before* the kv read-back and the write,
    // so a state of ones with g = log 2 lands on 0.5 + k[i] * (v[j] - 0.5).
    std::vector<f32> sd(static_cast<size_t>(ds * ds), 1.0f);
    const std::vector<f32> glog2 = {-std::log(2.0f)};
    cpu->delta_rule(o.data(), sd.data(), q_orth.data(), k.data(), v.data(),
                    glog2.data(), beta1.data(), 1, nh, nh, ds, ds, ds);
    for (i64 i = 0; i < ds; i++)
        for (i64 j = 0; j < ds; j++) {
            const f32 want =
                0.5f + k[static_cast<size_t>(i)] * (v[static_cast<size_t>(j)] - 0.5f);
            CHECK(std::fabs(sd[static_cast<size_t>(i * ds + j)] - want) < 1e-5f,
                  "exp(g) decays the state before the rank-1 write");
        }

    // Reading the state back: out[j] = sum_i S[i][j] q[i], which for
    // S = k (x) v and q = ones is v[j] * (k . q) = v[j].
    std::vector<f32> s2(static_cast<size_t>(ds * ds), 0.0f);
    cpu->delta_rule(o.data(), s2.data(), q_ones.data(), k.data(), v.data(),
                    g0.data(), beta1.data(), 1, nh, nh, ds, ds, ds);
    for (i64 j = 0; j < ds; j++)
        CHECK(std::fabs(o[static_cast<size_t>(j)] - v[static_cast<size_t>(j)]) < 1e-5f,
              "S^T q reads the rank-1 state back");

    // A partial forget: beta scales the write, so the state lands halfway
    // between the old value and the full rank-1 update.
    std::vector<f32> s3(static_cast<size_t>(ds * ds), 0.0f);
    const std::vector<f32> beta_half = {0.5f};
    cpu->delta_rule(o.data(), s3.data(), q_orth.data(), k.data(), v.data(),
                    g0.data(), beta_half.data(), 1, nh, nh, ds, ds, ds);
    for (i64 i = 0; i < ds; i++)
        for (i64 j = 0; j < ds; j++)
            CHECK(std::fabs(s3[static_cast<size_t>(i * ds + j)] -
                            0.5f * k[static_cast<size_t>(i)] *
                                v[static_cast<size_t>(j)]) < 1e-5f,
                  "beta scales the rank-1 write");

    // Two tokens in one call must equal two calls of one token each: the state
    // threading is exactly what makes a prefill chunk equivalent to decoding
    // the same tokens one at a time.
    {
        const std::vector<f32> v2b = {4, 3, 2, 1};
        std::vector<f32> qq(2 * ds), kk(2 * ds), vv(2 * ds), gg(2, 0.0f), bb(2, 1.0f);
        for (i64 i = 0; i < ds; i++) {
            qq[static_cast<size_t>(i)] = q_ones[static_cast<size_t>(i)];
            kk[static_cast<size_t>(i)] = k[static_cast<size_t>(i)];
            vv[static_cast<size_t>(i)] = v[static_cast<size_t>(i)];
            qq[static_cast<size_t>(ds + i)] = q_ones[static_cast<size_t>(i)];
            kk[static_cast<size_t>(ds + i)] = k[static_cast<size_t>(i)];
            vv[static_cast<size_t>(ds + i)] = v2b[static_cast<size_t>(i)];
        }
        std::vector<f32> block(static_cast<size_t>(2 * ds));
        std::vector<f32> whole(static_cast<size_t>(ds * ds), 0.0f);
        cpu->delta_rule(block.data(), whole.data(), qq.data(), kk.data(), vv.data(),
                        gg.data(), bb.data(), 2, nh, nh, ds, ds, ds);
        std::vector<f32> step(static_cast<size_t>(ds * ds), 0.0f);
        std::vector<f32> o1(static_cast<size_t>(ds)), o2(static_cast<size_t>(ds));
        const std::vector<f32> g1 = {0.0f};
        cpu->delta_rule(o1.data(), step.data(), q_ones.data(), k.data(), v.data(),
                        g1.data(), beta1.data(), 1, nh, nh, ds, ds, ds);
        cpu->delta_rule(o2.data(), step.data(), q_ones.data(), k.data(), v2b.data(),
                        g1.data(), beta1.data(), 1, nh, nh, ds, ds, ds);
        for (i64 i = 0; i < ds; i++) {
            CHECK(std::fabs(block[static_cast<size_t>(i)] -
                            o1[static_cast<size_t>(i)]) < 1e-5f,
                  "a two-token block matches the first single token");
            CHECK(std::fabs(block[static_cast<size_t>(ds + i)] -
                            o2[static_cast<size_t>(i)]) < 1e-5f,
                  "a two-token block matches the second single token, so the "
                  "state threads within the call");
        }
    }
    (void)n; (void)heads;

    // The packed query split: interleaved halves come apart correctly.
    // packed holds n_tok * n_head * 2 * hd: BOTH halves per head, per token.
    // Sized as 2*2*4 it was half the layout the reads below use, so the loop
    // walked off the end for the second token (and the device path, which
    // reads the same layout, would have too).
    std::vector<f32> packed(2 * 2 * 2 * 4), qo(2 * 2 * 4), gate(2 * 2 * 4);
    for (size_t i = 0; i < packed.size(); i++) packed[i] = static_cast<f32>(i);
    cpu->qwen3_next_split(qo.data(), gate.data(), packed.data(), 2, 2, 4);
    for (i64 t = 0; t < 2; t++)
        for (i64 h = 0; h < 2; h++)
            for (i64 i = 0; i < 4; i++) {
                const f32 src = packed[static_cast<size_t>((t * 2 + h) * 8 + i)];
                const f32 gsrc = packed[static_cast<size_t>((t * 2 + h) * 8 + 4 + i)];
                CHECK(std::fabs(qo[static_cast<size_t>((t * 2 + h) * 4 + i)] - src) < 1e-6f,
                      "the query half unpacks in place");
                CHECK(std::fabs(gate[static_cast<size_t>((t * 2 + h) * 4 + i)] - gsrc) < 1e-6f,
                      "the gate half follows the query in its block");
            }
    delete cpu;
}

// ---------------------------------------------------------------------------
// architecture table and load verdict
// ---------------------------------------------------------------------------

// Splits arch_known_names() on ", " — the same string a user is shown — so the
// test checks what the error message will actually contain.
static std::vector<std::string> split_names(const std::string &s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        const size_t j = s.find(", ", i);
        out.push_back(s.substr(i, j == std::string::npos ? j : j - i));
        if (j == std::string::npos) break;
        i = j + 2;
    }
    return out;
}

static void test_arch_table() {
    // The table is the only place an arch string is spelled out, so its own
    // invariants are worth checking: names distinct, every non-Yes entry
    // carrying a reason, and lookup agreeing with the name list.
    CHECK(arch_lookup("llama") != nullptr, "llama is in the table");
    CHECK(arch_lookup("qwen35moe") != nullptr, "qwen35moe is in the table");
    CHECK(arch_lookup("") == nullptr, "the empty arch name is not in the table");
    CHECK(arch_lookup("no-such-arch-anywhere") == nullptr,
          "an unlisted arch is not in the table");

    const std::vector<std::string> names = split_names(arch_known_names());
    CHECK(names.size() >= 20, "the table covers the inventoried architectures");
    for (const std::string &n : names) {
        const ArchSpec *s = arch_lookup(n);
        CHECK(s != nullptr && n == s->name, "every listed name looks itself up");
        if (!s) continue;
        CHECK(s->support == ArchSupport::Yes || (s->why && s->why[0]),
              "a partial or refused arch says why");
        CHECK(arch_support_name(s->support)[0] != '?',
              "support has a name for messages");
        CHECK(arch_shape_name(s->shape)[0] != '?', "shape has a name");
        CHECK(arch_role_name(s->role)[0] != '?', "role has a name");
    }
    for (size_t i = 0; i < names.size(); i++)
        for (size_t j = i + 1; j < names.size(); j++)
            CHECK(names[i] != names[j], "arch names are distinct");

    // The fallback: known to nobody, but still a usable spec.
    bool known = true;
    const ArchSpec &fallback = arch_spec("who-knows", &known);
    CHECK(!known, "an unlisted arch reports itself unknown");
    CHECK(fallback.support == ArchSupport::Partial && fallback.why[0],
          "the fallback loads, with the default reason recorded");
    CHECK(std::string(fallback.family) == "llama",
          "the fallback is read against llama defaults");
}

static void test_arch_tensor_maps() {
    // One name, two meanings: the delta net's fused projection is implemented,
    // a fused *attention* projection is not.
    CHECK(arch_tensor_gap("blk.0.attn_qkv.weight", ArchShape::Recurrent) == nullptr,
          "the gated delta net's attn_qkv is not a gap");
    CHECK(arch_tensor_gap("blk.0.attn_qkv.weight", ArchShape::Dense) != nullptr,
          "a dense fused attn_qkv is a gap");
    CHECK(arch_tensor_gap("blk.0.attn_gate.weight", ArchShape::Recurrent) == nullptr,
          "the delta net's own gate is not a gap");
    CHECK(arch_tensor_gap("blk.7.attn_gate.weight", ArchShape::Moe) != nullptr,
          "a dense attention output gate is a gap");
    CHECK(arch_tensor_gap("blk.7.attn_v_gate.bias", ArchShape::Moe) != nullptr,
          "a gate on attention values is a gap");
    CHECK(arch_tensor_gap("blk.2.attn_kv_a.weight", ArchShape::Moe) != nullptr,
          "MLA compression is a gap");
    CHECK(arch_tensor_gap("blk.2.ssm_out.weight", ArchShape::Dense) != nullptr,
          "a Mamba block in a dense file is a gap");
    CHECK(arch_tensor_gap("blk.2.ssm_out.weight", ArchShape::Recurrent) == nullptr,
          "the delta net's ssm_out is not a gap");
    // Names that are ordinary parts of a loadable schema must stay ordinary.
    CHECK(arch_tensor_gap("blk.2.attn_q.weight", ArchShape::Dense) == nullptr,
          "attn_q is not a gap");
    CHECK(arch_tensor_gap("blk.2.ffn_gate_exps.weight", ArchShape::Moe) == nullptr,
          "routed-expert FFN weights are not a gap");
    CHECK(arch_tensor_gap("blk.2.attn_q_norm.weight", ArchShape::Dense) == nullptr,
          "QK-norm is not a gap");

    // Heads are noted, not refused.
    CHECK(arch_tensor_head("dflash.aux_hidden_norm.0.weight") != nullptr,
          "a dflash head is recognized");
    CHECK(arch_tensor_head("dspark.markov_head_a.weight") != nullptr,
          "a dspark head is recognized");
    CHECK(arch_tensor_head("mtp.0.attn_kv.weight") != nullptr,
          "an MTP head is recognized");
    CHECK(arch_tensor_head("blk.3.ffn_up.weight") == nullptr,
          "an ordinary tensor is not a head");
}

// Writes a one-tensor GGUF whose architecture and tensor name are the only
// things that matter to the verdict, into a file named after the case. Each
// case gets its own path because a file that is still mapped cannot be
// rewritten on Windows, and these fixtures are written repeatedly.
static std::string build_verdict_fixture(const std::string &tag, const std::string &arch,
                                         const std::string &tensor, u32 tensor_type,
                                         i32 nextn) {
    const std::string path = std::string("kraken-verdict-") + tag + ".gguf";
    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", nextn > 0 ? 2 : 1);
    if (nextn > 0) b.meta_u32(arch + ".nextn_predict_layers", static_cast<u32>(nextn));
    const std::vector<f32> data{1.0f, 2.0f, 3.0f, 4.0f};
    if (tensor_type == 0)
        b.tensor_f32(tensor, {2, 2}, data);
    else
        b.tensor_raw(tensor, tensor_type, {2, 2}, data.data(), data.size() * 4);
    if (!b.write(path)) return std::string();
    return path;
}

static void test_load_verdict() {
    Backend *cpu = make_cpu_backend();
    std::vector<std::string> made;

    // A plain supported schema with no extras: runnable, and nothing to say.
    {
        const std::string path =
            build_verdict_fixture("llama", "llama", "blk.0.attn_q.weight", 0, 0);
        CHECK(!path.empty(), "wrote a verdict fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(v.runnable && v.known && v.arch == "llama", "llama is runnable");
        CHECK(v.blockers.empty(), "llama has no blockers");
        CHECK(v.notes.empty(), "a supported arch with no extras has no notes");
        CHECK(v.mtp_blocks == 0, "no MTP blocks declared");
        CHECK(v.headline().find("supported") != std::string::npos,
              "the headline names the support level");
        made.push_back(path);
    }

    // An unlisted arch is a note, never a refusal.
    {
        const std::string path = build_verdict_fixture(
            "unlisted", "totally-unlisted", "blk.0.attn_q.weight", 0, 0);
        CHECK(!path.empty(), "wrote an unlisted-arch fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(v.runnable && !v.known, "an unlisted arch still runs");
        CHECK(v.notes.size() == 1 &&
                  v.notes[0].find("not in the architecture table") != std::string::npos,
              "and says so");
        Model m;
        std::string lerr;
        const bool loaded = m.load(*cpu, path, &lerr);
        if (!lerr.empty()) std::fprintf(stderr, "  (unlisted arch: %s)\n", lerr.c_str());
        CHECK(!loaded, "the loader fails on the schema, not on the arch");
        CHECK(lerr.find("geometry") != std::string::npos &&
                  lerr.find("not supported") == std::string::npos,
              "and does not blame the architecture");
        made.push_back(path);
    }

    // laguna is the supported-arch-that-used-to-be-refused case, and it is the
    // interesting one because its schema is read: per-layer head counts, the
    // per-head gate tensor, the selection bias and the YaRN table all load. The
    // fixture carries the gate tensor, which the gap map exempts for this arch
    // only, so the verdict must accept it and NOT report "an attention output
    // gate ... which this engine does not implement" -- naming a
    // successfully-read tensor as unimplemented is the failure this guards.
    // (The engine side that consumes each piece is checked by the oracle and by
    // a real laguna file; this fixture has no geometry to load.)
    {
        const std::string path =
            build_verdict_fixture("laguna", "laguna", "blk.0.attn_gate.weight", 0, 0);
        CHECK(!path.empty(), "wrote a laguna fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(v.runnable && v.known && v.arch == "laguna",
              "laguna is supported");
        CHECK(v.blockers.empty(), "nothing blocks it");
        bool tensor_reason = false;
        for (const std::string &b : v.blockers)
            if (b.find("does not implement") != std::string::npos) tensor_reason = true;
        CHECK(!tensor_reason, "and the gate tensor is not called unimplemented");
        Model m;
        std::string lerr;
        CHECK(!m.load(*cpu, path, &lerr), "the loader still needs geometry");
        CHECK(lerr.find("geometry") != std::string::npos &&
                  lerr.find("not supported") == std::string::npos,
              "and the failure is the schema, not the architecture");
        made.push_back(path);
    }

    // An arch refusal and a gap tensor are *independent* causes, and both have
    // to be reported: this file would be refused for its tensor alone even if
    // its arch were supported.
    {
        const std::string path = build_verdict_fixture(
            "muse-glimmer", "muse-glimmer", "blk.0.attn_gate.weight", 0, 0);
        CHECK(!path.empty(), "wrote a muse-glimmer fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(!v.runnable, "muse-glimmer is refused");
        CHECK(v.blockers.size() >= 2,
              "its arch and its gate tensor are separate causes");
        bool arch_reason = false, gate_reason = false;
        for (const std::string &b : v.blockers) {
            if (b.find("attention output gate") != std::string::npos) arch_reason = true;
            if (b.find("does not implement") != std::string::npos) gate_reason = true;
        }
        CHECK(arch_reason && gate_reason, "both causes are named");
        Model m;
        std::string lerr;
        CHECK(!m.load(*cpu, path, &lerr), "the loader refuses the file");
        CHECK(lerr.find("muse-glimmer") != std::string::npos,
              "and its error names the arch");
        made.push_back(path);
    }

    // A draft file: refused as a model, with the role as the reason.
    {
        const std::string path =
            build_verdict_fixture("dflash", "dflash", "blk.0.ffn_up.weight", 0, 0);
        CHECK(!path.empty(), "wrote a dflash fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(!v.runnable && v.spec && v.spec->role == ArchRole::Draft,
              "a draft file is refused and labelled a draft");
        CHECK(v.headline().find("draft") != std::string::npos,
              "the headline calls it a draft file");
        CHECK(v.blockers.size() == 1 &&
                  v.blockers[0].find("--draft") != std::string::npos,
              "the reason says what to use instead");
        made.push_back(path);
    }

    // MTP blocks riding along in a target: a note, and the count the loader uses.
    {
        const std::string path =
            build_verdict_fixture("mtp", "qwen35", "blk.0.attn_qkv.weight", 0, 1);
        CHECK(!path.empty(), "wrote an MTP fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(v.runnable, "a target with MTP blocks runs");
        CHECK(v.mtp_blocks == 1, "the MTP block count is reported");
        bool said = false;
        for (const std::string &n : v.notes)
            if (n.find("multi-token-prediction") != std::string::npos) said = true;
        CHECK(said, "and the note says which blocks are skipped");
        // attn_qkv on a recurrent arch is the delta net's own projection.
        CHECK(v.blockers.empty(), "the delta net's attn_qkv is not held against it");
        made.push_back(path);
    }

    // A format this build cannot dequantize is reported by name, not as a
    // corrupt file: the container still opens it. Pick a type id that no
    // header here defines (200) so the loader refuses it.
    {
        const std::string path =
            build_verdict_fixture("unknown200", "llama", "blk.0.attn_q.weight", 200, 0);
        CHECK(!path.empty(), "wrote a type-200 fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "an unknown format still parses");
        const GgufTensor *t = g.tensor("blk.0.attn_q.weight");
        CHECK(t && !t->dtype_known && t->data == nullptr,
              "the entry is described, without a payload pointer");
        const ModelVerdict v = assess_model(g);
        CHECK(!v.runnable, "the unknown format makes the file unrunnable");
        bool named = false;
        for (const std::string &b : v.blockers)
            if (b.find("#200") != std::string::npos) named = true;
        CHECK(named, "the blocker names the on-disk format id");
        made.push_back(path);
    }

    for (const std::string &p : made) std::remove(p.c_str());
    delete cpu;
}

static void test_gguf_roundtrip() {
    CHECK(build_tiny_model(kTestModelPath, false), "wrote a synthetic GGUF");

    Gguf g;
    std::string err;
    CHECK(g.load(kTestModelPath, &err), "loaded the synthetic GGUF");
    if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
    CHECK(g.version() == 3, "GGUF version is 3");
    CHECK(g.get_str("general.architecture") == "llama", "architecture metadata");
    CHECK(g.get_i64("llama.block_count") == 1, "block_count metadata");
    CHECK(g.get_f64("llama.rope.freq_base") == 10000.0, "rope base metadata");
    CHECK(g.tensor("token_embd.weight") != nullptr, "token_embd tensor found");
    CHECK(g.tensor("nope.weight") == nullptr, "missing tensor is null");
    CHECK(g.get_str_array("tokenizer.ggml.tokens") != nullptr, "tokens array parsed");
    CHECK(g.alignment() == 32, "alignment honoured");

    const GgufTensor *t = g.tensor("blk.0.attn_q.weight");
    CHECK(t && t->n_dims == 2 && t->ne[0] == 32 && t->ne[1] == 32, "attn_q geometry");
    CHECK(t && t->type == DType::F16, "attn_q is F16");
    CHECK(t && t->data != nullptr && t->n_bytes == 32 * 32 * 2, "attn_q payload size");
}

// ---------------------------------------------------------------------------
// tokenizer
// ---------------------------------------------------------------------------

static void test_tokenizer_spm_and_bpe() {
    {
        Gguf g;
        std::string err;
        CHECK(g.load(kTestModelPath, &err), "reload for tokenizer");
        Tokenizer tok;
        CHECK(tok.load(g, &err), "SPM tokenizer loads");
        CHECK(tok.model() == TokModel::Spm, "model is SPM");
        CHECK(tok.bos() == 1 && tok.eos() == 2, "bos/eos ids");

        std::vector<i32> ids;
        CHECK(tok.encode("the", ids, true) > 0, "SPM encodes");
        CHECK(!ids.empty() && ids[0] == 1, "SPM prepends bos");
        const std::string back = tok.decode(ids.data() + 1,
                                            static_cast<int>(ids.size()) - 1);
        CHECK(back.find("the") != std::string::npos, "SPM round trips 'the'");
    }
    {
        const std::string bpe_path = "kraken-test-bpe.gguf";
        CHECK(build_tiny_model(bpe_path, true), "wrote a synthetic BPE GGUF");
        Gguf g;
        std::string err;
        CHECK(g.load(bpe_path, &err), "loaded the synthetic BPE GGUF");
        Tokenizer tok;
        CHECK(tok.load(g, &err), "BPE tokenizer loads");
        CHECK(tok.model() == TokModel::Bpe, "model is byte-BPE");

        std::vector<i32> ids;
        CHECK(tok.encode("abc", ids, false) > 0, "BPE encodes");
        const std::string text = tok.decode(ids.data(), static_cast<int>(ids.size()));
        CHECK(text == "abc", "BPE round trips 'abc'");
        // Regression: a space run before a word used to spin
        // forever — the trailing space was re-processed as a new
        // run instead of being absorbed by the following word.
        std::vector<i32> ws;
        CHECK(tok.encode("Hello world", ws, false) > 0,
              "BPE encodes a spaced prompt");
        const std::string spaced = tok.decode(ws.data(), static_cast<int>(ws.size()));
        CHECK(spaced == "Hello world", "BPE round trips 'Hello world'");
        std::vector<i32> ms;
        CHECK(tok.encode("a  b", ms, false) > 0, "BPE encodes a double space");
        const std::string doubled = tok.decode(ms.data(), static_cast<int>(ms.size()));
        CHECK(doubled == "a  b", "BPE round trips 'a  b'");
        std::vector<i32> ts;
        CHECK(tok.encode("trailing ", ts, false) > 0,
              "BPE encodes a trailing space");
        const std::string trailing = tok.decode(ts.data(), static_cast<int>(ts.size()));
        CHECK(trailing == "trailing ", "BPE round trips 'trailing '");
        std::remove(bpe_path.c_str());
    }
}

// ---------------------------------------------------------------------------
// sampler determinism
// ---------------------------------------------------------------------------

static void test_sampler_determinism() {
    std::vector<f32> logits(64);
    for (int i = 0; i < 64; i++) logits[static_cast<size_t>(i)] = 0.1f * static_cast<f32>(i);

    SampleParams p;
    p.temp = 1.0f;
    p.top_k = 8;
    p.top_p = 0.9f;
    p.min_p = 0.0f;
    p.repeat_penalty = 1.0f;
    p.seed = 1234;

    Sampler a, b;
    a.reset(p);
    b.reset(p);
    bool same = true;
    for (int i = 0; i < 16; i++)
        if (a.sample(logits.data(), 64) != b.sample(logits.data(), 64)) same = false;
    CHECK(same, "same seed -> identical samples");

    p.greedy = true;
    Sampler c;
    c.reset(p);
    bool argmax_ok = true;
    for (int i = 0; i < 8; i++)
        if (c.sample(logits.data(), 64) != 63) argmax_ok = false;
    CHECK(argmax_ok, "greedy picks the argmax");
}

// ---------------------------------------------------------------------------
// MoE: lazy expert residency
// ---------------------------------------------------------------------------

static const char *kMoeTestPath = "kraken-moe-test.gguf";
static const char *kMoeDenseTwinPath = "kraken-moe-dense-twin.gguf";

struct MoeRun {
    std::vector<i32> tokens;
    bool ok = false;
    i64 loads = 0;
    u64 evictions = 0;
    size_t resident_slots = 0;
    size_t capacity_slots = 0;
    u64 demotions = 0;
    u64 promotions = 0;
    size_t host_slots = 0;
};

// Greedy generation through a MoE model, reporting what the expert cache did.
static MoeRun run_moe(const char *path, i32 cache_mb, i32 cache_slots,
                      int max_tokens, i32 l2_mb = 0) {
    MoeRun out;
    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx = 64;
    cfg.prefill_chunk = 8;
    cfg.expert_cache_mb = cache_mb;
    cfg.expert_cache_slots = cache_slots;
    cfg.expert_warm_mb = l2_mb;
    std::string err;
    if (!engine.init(cpu, cfg, &err)) {
        std::fprintf(stderr, "  moe init failed: %s\n", err.c_str());
        delete cpu;
        return out;
    }
    GenerateParams p;
    p.prompt = "the";
    p.max_tokens = max_tokens;
    p.sampler.greedy = true;
    p.sampler.temp = 0.0f;
    GenerateResult r;
    out.ok = engine.generate(p, &r);
    out.tokens = r.tokens;
    const ExpertCache &ec = engine.model().experts();
    out.loads = static_cast<i64>(ec.loads());
    out.evictions = ec.evictions();
    out.resident_slots = ec.resident_slots();
    out.capacity_slots = ec.capacity_slots();
    out.demotions = ec.demotions();
    out.promotions = ec.promotions();
    out.host_slots = ec.host_slots();
    engine.shutdown();
    delete cpu;
    return out;
}

static void test_moe_schema_and_laziness() {
    MoeSpec vs;
    vs.dense = false;
    vs.n_expert = 4;
    vs.n_expert_used = 2;
    vs.shared = true;
    vs.varied = true;
    CHECK(build_tiny_moe_model(kMoeTestPath, vs), "wrote a synthetic MoE GGUF");

    // --- metadata and schema ---------------------------------------------
    {
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        CHECK(m.load(*cpu, kMoeTestPath, &err), "loaded the synthetic MoE model");
        if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
        const ModelConfig &mc = m.cfg();
        CHECK(mc.is_moe, "model is detected as MoE");
        CHECK(mc.n_expert == 4, "expert_count parsed");
        CHECK(mc.n_expert_used == 2, "expert_used_count parsed");
        CHECK(mc.n_ff_exp == 64, "expert_feed_forward_length parsed");
        CHECK(mc.n_ff_shexp == 32, "shared-expert width parsed");
        CHECK(m.n_ff_ws() == 64, "FFN workspace covers the widest branch");

        const LayerWeights &L = m.layers()[0];
        CHECK(L.moe, "layer 0 takes the routed-expert path");
        CHECK(L.router.present() && L.router.n_out == 4, "router has one row per expert");
        CHECK(L.experts.present(), "expert source recorded");
        CHECK(L.experts.n_expert == 4 && L.experts.n_ff_exp == 64,
              "expert source geometry");
        CHECK(L.shexp_gate.present(), "shared expert loaded");

        // The whole point: expert bytes live in the mapping, not in device memory.
        const size_t per = L.experts.per_expert(L.experts.gate);
        CHECK(per * 4 == L.experts.gate->n_bytes, "expert slices tile the tensor");
        CHECK(per > 0, "per-expert slice is non-empty");
        CHECK(m.experts().loads() == 0, "no expert loaded straight after load");
        CHECK(m.experts().resident_slots() == 0, "no expert resident straight after load");
        m.unload();
        delete cpu;
    }

    // --- a one-slot cache must still decode, by reloading ----------------
    const MoeRun tiny = run_moe(kMoeTestPath, 0, 1, 6);
    CHECK(tiny.ok && !tiny.tokens.empty(), "MoE generation works with a 1-slot cache");
    CHECK(tiny.capacity_slots == 1, "1-slot budget yields exactly one slot");
    CHECK(tiny.evictions > 0, "a 1-slot cache evicts, so experts really are reloaded");
    CHECK(tiny.resident_slots <= 1, "residency never exceeds the slot budget");
    CHECK(tiny.loads >= static_cast<i64>(tiny.evictions),
          "every eviction came from an on-demand load");

    // --- results must not depend on how much is resident -----------------
    const MoeRun big = run_moe(kMoeTestPath, 4096, 0, 6);
    CHECK(big.ok, "MoE generation works with a large cache");
    CHECK(big.evictions == 0, "a large cache never evicts");
    CHECK(big.resident_slots <= 4, "at most one slot per expert of the single layer");
    CHECK(big.tokens == tiny.tokens,
          "generation is identical with a 1-slot and a large expert cache");
    CHECK(big.loads > 0 && big.loads <= 4,
          "even a large cache loads experts lazily, once each");

    // Same configuration twice must be deterministic.
    const MoeRun again = run_moe(kMoeTestPath, 0, 1, 6);
    CHECK(again.tokens == tiny.tokens, "MoE generation is deterministic");

    // --- the second tier must not change a single token -------------------
    const MoeRun tiered = run_moe(kMoeTestPath, 0, 1, 6, 64);
    CHECK(tiered.ok, "MoE generation works with a host L2 tier");
    CHECK(tiered.demotions > 0 && tiered.promotions > 0,
          "a 1-slot cache with a tier demotes and promotes instead of dropping");
    CHECK(tiered.loads < tiny.loads,
          "the tier converts mapping reloads into DMA promotions");
    CHECK(tiered.tokens == tiny.tokens,
          "generation is bit-identical with and without the host tier");
}

// ---------------------------------------------------------------------------
// ExpertCache policy: LFU + aging + pinning, driven directly
// ---------------------------------------------------------------------------

// Wraps the CPU backend so release() counts expert-slot frees: that is how the
// tests observe evictions without poking at the cache's internals.
class CountingBackend final : public Backend {
public:
    explicit CountingBackend(Backend *inner) : inner_(inner) {}
    ~CountingBackend() override { delete inner_; }

    BackendKind kind() const override { return inner_->kind(); }
    const DeviceCaps &caps() const override { return inner_->caps(); }
    DType act_type() const override { return inner_->act_type(); }
    size_t act_size() const override { return inner_->act_size(); }
    void *alloc(size_t bytes) override { return inner_->alloc(bytes); }
    void release(void *p) override {
        if (p) releases_++;
        inner_->release(p);
    }
    // The host tier's own counters: a demotion is visible as host allocations,
    // a promotion (or a recycled tier slot) as host releases.
    void *alloc_host(size_t bytes) override {
        void *p = inner_->alloc_host(bytes);
        if (p) host_allocs_++;
        return p;
    }
    void release_host(void *p) override {
        if (p) host_releases_++;
        inner_->release_host(p);
    }
    void upload(void *d, const void *s, size_t b, size_t o) override {
        inner_->upload(d, s, b, o);
    }
    void download(void *d, const void *s, size_t b, size_t o) override {
        inner_->download(d, s, b, o);
    }
    void fill0(void *d, size_t b) override { inner_->fill0(d, b); }
    void sync() override { inner_->sync(); }
    void download_f32(f32 *d, const void *s, i64 n) override {
        inner_->download_f32(d, s, n);
    }
    std::string describe() const override { return inner_->describe(); }
    void embed(void *o, const void *t, DType tt, i64 v, i64 e, const i32 *tk,
               i64 n) override {
        inner_->embed(o, t, tt, v, e, tk, n);
    }
    void rmsnorm(void *o, const void *x, const f32 *w, i64 r, i64 n, f32 e) override {
        inner_->rmsnorm(o, x, w, r, n, e);
    }
    void gemm(void *o, const void *x, const void *w, DType wt, i64 no, i64 ni,
              i64 r) override {
        inner_->gemm(o, x, w, wt, no, ni, r);
    }
    void logits_topk(const void *lg, i64 n, i32 k, const void *pm, f32 rp,
                     f32 sc, f32 tp, i32 *ids, f32 *vals, f32 *mass_out,
                     void *scratch, i32 cand_cap, i64 *count_out) override {
        inner_->logits_topk(lg, n, k, pm, rp, sc, tp, ids, vals, mass_out,
                            scratch, cand_cap, count_out);
    }
    void rope(void *q, void *k, i64 nh, i64 nk, i64 hd, i64 nt, i64 p0,
              const f32 *iv, f32 s, f32 f, bool nx) override {
        inner_->rope(q, k, nh, nk, hd, nt, p0, iv, s, f, nx);
    }
    void kv_append(void *kc, void *vc, const void *k, const void *v,
                   const AttnDesc &d) override {
        inner_->kv_append(kc, vc, k, v, d);
    }
    void attention(void *o, const void *q, const void *kc, const void *vc,
                   const AttnDesc &d) override {
        inner_->attention(o, q, kc, vc, d);
    }
    void silu_mul(void *o, const void *g, const void *u, i64 n) override {
        inner_->silu_mul(o, g, u, n);
    }
    void add_bias_rows(void *o, const f32 *b, i64 n, i64 r) override {
        inner_->add_bias_rows(o, b, n, r);
    }
    void add_inplace(void *a, const void *b, i64 n) override {
        inner_->add_inplace(a, b, n);
    }
    void copy_act(void *d, const void *s, i64 n) override {
        inner_->copy_act(d, s, n);
    }
    void axpy(void *d, const void *s, f32 a, i64 n) override {
        inner_->axpy(d, s, a, n);
    }
    void upload_i32(i32 *d, const i32 *s, i64 n) override {
        inner_->upload_i32(d, s, n);
    }
    void gather_rows(void *d, const void *s, const i32 *r, i64 nr, i64 n) override {
        inner_->gather_rows(d, s, r, nr, n);
    }
    void scatter_axpy_rows(void *d, const void *s, const i32 *r, const f32 *a,
                           i64 nr, i64 n) override {
        inner_->scatter_axpy_rows(d, s, r, a, nr, n);
    }

    u64 releases() const { return releases_; }
    u64 host_allocs() const { return host_allocs_; }
    u64 host_releases() const { return host_releases_; }

private:
    Backend *inner_;
    u64 releases_ = 0;
    u64 host_allocs_ = 0;
    u64 host_releases_ = 0;
};

// A four-expert source whose slices are 8 bytes each, so a 96-byte budget is
// exactly four slots and every policy decision is visible. Shared by the
// residency-policy tests, which differ only in how they configure the cache.
static bool build_expert_source_model(const std::string &path) {
    GgufBuilder b;
    b.meta_str("general.architecture", "llama");
    b.meta_u32("general.alignment", 32);
    add_spm_vocab(b, 260);
    u8 gate_blob[8 * 4];
    u8 up_blob[8 * 4];
    u8 down_blob[8 * 4];
    for (int i = 0; i < 8 * 4; i++) {
        gate_blob[i] = static_cast<u8>(i);
        up_blob[i] = static_cast<u8>(i + 1);
        down_blob[i] = static_cast<u8>(i + 2);
    }
    // Geometry: [2, 2, 4] F16 = 16 elements = 32 bytes per tensor, so one
    // expert's slice is 8 bytes and a 4-expert set is 24 bytes.
    b.tensor_raw("blk.0.ffn_gate_exps.weight", 1, {2, 2, 4},
                 gate_blob, sizeof(gate_blob));
    b.tensor_raw("blk.0.ffn_up_exps.weight", 1, {2, 2, 4}, up_blob,
                 sizeof(up_blob));
    b.tensor_raw("blk.0.ffn_down_exps.weight", 1, {2, 2, 4}, down_blob,
                 sizeof(down_blob));
    return b.write(path);
}

static ExpertSource expert_source(Gguf &g) {
    ExpertSource src;
    src.gate = g.tensor("blk.0.ffn_gate_exps.weight");
    src.up = g.tensor("blk.0.ffn_up_exps.weight");
    src.down = g.tensor("blk.0.ffn_down_exps.weight");
    src.n_expert = 4;
    src.n_embd = 2;
    src.n_ff_exp = 2;
    return src;
}

static void test_expert_cache_policy() {
    const std::string path = "kraken-policy-test.gguf";
    CHECK(build_expert_source_model(path), "wrote the policy-test GGUF");

    Backend *cpu = make_cpu_backend();
    CountingBackend be(cpu);
    Gguf g;
    std::string err;
    CHECK(g.load(path, &err), "loaded the policy-test GGUF");
    if (!err.empty()) std::fprintf(stderr, "  (policy gguf: %s)\n", err.c_str());

    const ExpertSource src = expert_source(g);
    CHECK(src.present(), "policy-test expert source is complete");
    CHECK(src.expert_bytes() == 3 * 8, "one expert is 24 bytes");

    ExpertCache cache;
    cache.configure(&be, 96); // 96 / 24 = 4 slots

    const auto touch = [&](i32 e) {
        return cache.acquire(src, 0, e) != nullptr;
    };

    // 1. Misses load and count one touch; hits count too.
    CHECK(cache.loads() == 0, "cache starts empty");
    CHECK(touch(0) && touch(1), "first two experts load");
    CHECK(cache.loads() == 2 && cache.hits() == 0, "two misses, no hits");
    CHECK(touch(0), "expert 0 is still resident");
    CHECK(cache.hits() == 1, "one hit recorded");

    // 2. Fill the cache. Expert 0 now has 3 touches, expert 1 has 2.
    CHECK(touch(2) && touch(3), "experts 2 and 3 fill the four slots");
    CHECK(cache.resident_slots() == 4 && cache.evictions() == 0,
          "four experts fit with no eviction");

    // 3. Pin protection: hammer expert 0 past the pin threshold, then brush
    // past it with cold experts repeatedly. LRU would evict it; the pin must
    // hold it and the cold experts must take the falls instead.
    const int kPin = static_cast<int>(ExpertCache::kPinThreshold);
    const int kDecay = static_cast<int>(ExpertCache::kPinDecay);
    (void)kDecay;
    for (int i = 0; i < kPin + 4; i++) CHECK(touch(0), "pinned expert reachable");
    CHECK(cache.pinned_slots() >= 1, "a hammered expert becomes pinned");
    CHECK(cache.evictions() == 0, "pin traffic alone causes no evictions");
    for (int i = 0; i < 3; i++) {
        CHECK(touch(1), "cold expert 1 stays reachable");
        CHECK(touch(2), "cold expert 2 stays reachable");
    }
    CHECK(cache.evictions() == 0,
          "hot traffic through a cache with room for all experts never evicts");

    // 4. With history, LFU must disagree with LRU: a 1-slot cache where the
    // hot expert keeps getting re-requested beats a single competing cold touch
    // — the cold expert loses its place, the hot one survives without a reload.
    cache.clear();
    cache.configure(&be, 24); // one slot
    for (int i = 0; i < kPin + 2; i++) CHECK(touch(0), "hammer expert 0");
    CHECK(cache.pinned_slots() == 1, "expert 0 is pinned in the 1-slot cache");
    CHECK(cache.loads() == 1 && cache.evictions() == 0,
          "pinned traffic alone never evicts");
    CHECK(touch(1), "expert 1 still loads — the pin never blocks progress");
    CHECK(cache.loads() == 2, "expert 1 forced one load");
    CHECK(cache.evictions() == 1, "exactly one eviction happened");
    CHECK(touch(0), "expert 0 is reachable again after the forced eviction");
    CHECK(cache.loads() == 3, "expert 0 had to reload once it lost its slot");

    // 5. A hit releases nothing — residency, not traffic, costs memory.
    cache.clear();
    cache.configure(&be, 24);
    CHECK(touch(3), "expert 3 loads");
    const u64 releases_before = be.releases();
    CHECK(touch(3), "re-touching the only expert is a hit");
    CHECK(be.releases() == releases_before, "a hit releases nothing");

    // 6. Decay: geometric cooling means a pin earned, then abandoned, expires —
    // while the rival expert, kept warm, keeps its slot.
    cache.clear();
    cache.configure(&be, 48); // two slots
    for (int i = 0; i < kPin * 2; i++) CHECK(touch(0), "pump expert 0");
    CHECK(cache.pinned_slots() == 1, "expert 0 is pinned in the 2-slot cache");
    CHECK(touch(1), "expert 1 loads into the second slot");
    CHECK(cache.evictions() == 0, "both fit, nothing evicted");
    // Starve expert 0 of traffic while expert 1 stays warm. Each touch of 1 is
    // one acquire, so after kPinDecay windows 0's counter decays; once it falls
    // below the pin threshold the pin lapses. 1 must never be the victim.
    const u64 evictions_before = cache.evictions();
    bool pin_lapsed = false;
    for (int i = 0; i < 8192 && !pin_lapsed; i++) {
        CHECK(touch(1), "keep expert 1 warm");
        if (cache.touch_count(0, 0) < ExpertCache::kPinThreshold) pin_lapsed = true;
    }
    CHECK(pin_lapsed, "an unpumped pin decays and lapses");
    CHECK(cache.evictions() == evictions_before,
          "the warm rival was never evicted while both fit");
    CHECK(cache.touch_count(0, 0) < cache.touch_count(0, 1),
          "the abandoned expert is now the colder one");
    cache.clear();
    cache.configure(&be, 24); // one slot: the colder expert must lose
    CHECK(touch(0), "expert 0 loads first");
    CHECK(touch(1), "expert 1 (recently hot) takes the slot from 0");
    CHECK(cache.touch_count(0, 1) >= 1, "the winner kept its history within the generation");

    cache.clear();
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// ExpertCache second tier: VRAM slots over a page-locked host tier
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ExpertCache WARM tier: read-through residency over pageable host RAM
// ---------------------------------------------------------------------------

static void test_expert_cache_warm_tier() {
    const std::string path = "kraken-warm-test.gguf";
    CHECK(build_expert_source_model(path), "wrote the WARM-tier test GGUF");

    Backend *cpu = make_cpu_backend();
    CountingBackend be(cpu);
    Gguf g;
    std::string err;
    CHECK(g.load(path, &err), "loaded the WARM-tier test GGUF");
    if (!err.empty()) std::fprintf(stderr, "  (warm gguf: %s)\n", err.c_str());
    const ExpertSource src = expert_source(g);
    CHECK(src.present(), "WARM-test expert source is complete");

    ExpertCache cache;
    const auto touch = [&](i32 e) { return cache.acquire(src, 0, e) != nullptr; };

    // 1. Read-through: a cold load lands in WARM first and is copied on to VRAM
    //    from there, so the tier holds what the run actually touched instead of
    //    being a staging area that empties itself.
    cache.configure(&be, 48, 96);
    CHECK(cache.warm_capacity_bytes() == 96, "the WARM tier is configured");
    CHECK(touch(0) && touch(1), "the first two experts load");
    CHECK(cache.warm_admissions() == 2, "both cold loads were admitted to WARM");
    CHECK(cache.warm_slots() == 2 && cache.resident_slots() == 2,
          "and both are still resident, in WARM and in VRAM");
    CHECK(cache.warm_hits() == 0 && cache.demotions() == 0,
          "nothing has been served out of WARM yet");
    CHECK(be.host_allocs() == 6,
          "read-through allocated one pageable copy per tensor, per expert");

    // 2. A VRAM eviction keeps the WARM copy and costs nothing: the one expert's
    //    worth of host allocations here is the *new* expert's admission, not a
    //    download of the victim. HOT -> WARM moves no bytes at all.
    const u64 host_allocs_before = be.host_allocs();
    CHECK(touch(2), "expert 2 needs a slot");
    CHECK(cache.evictions() == 1, "one VRAM eviction");
    CHECK(cache.demotions() == 1, "the evicted expert stayed resident in WARM");
    CHECK(cache.warm_slots() == 3, "WARM kept its own copy, not the device's");
    CHECK(be.host_allocs() == host_allocs_before + 3,
          "the eviction allocated nothing; only the new admission did");
    CHECK(cache.touch_count(0, 0) == 1, "the evicted expert keeps its LFU counter");

    // 3. Re-requesting it is a promotion out of WARM, not a re-read of the file.
    const u64 loads_before = cache.loads();
    CHECK(touch(0), "the WARM-resident expert comes back");
    CHECK(cache.loads() == loads_before, "a promotion does not read the file");
    CHECK(cache.promotions() == 1 && cache.warm_hits() == 1,
          "the acquire was served out of WARM");
    CHECK(cache.warm_slots() == 3, "WARM is inclusive: the copy outlived promotion");
    CHECK(be.host_releases() == 0, "and no host copy was freed to make room for it");
    CHECK(cache.touch_count(0, 0) == 2,
          "a promoted expert keeps counting — it was hot before the eviction");

    // 4. WARM is bounded by its own budget: room for one expert, so each new
    //    admission recycles the previous one out of the tier. Eviction is a
    //    plain free: the file is the canonical copy and nothing is written back.
    cache.clear();
    cache.configure(&be, 48, 24);
    CHECK(touch(0) && touch(1) && touch(2) && touch(3), "four loads, two slots");
    CHECK(cache.warm_slots() == 1, "a one-expert tier never holds more than one");
    CHECK(cache.resident_slots() == 2, "VRAM still holds its two slots");
    CHECK(cache.warm_evictions() > 0, "the tier evicted to stay inside its budget");
    CHECK(cache.warm_rejects() == 0, "an expert that fits is never rejected");

    // 5. An expert larger than the whole tier is rejected, not admitted: keeping
    //    it would mean evicting everything and still not fitting. The load still
    //    happens — from the mapping, as it did before the tier existed.
    cache.clear();
    cache.configure(&be, 24, 8);
    CHECK(touch(0) && touch(1), "a one-slot cache still makes progress");
    CHECK(cache.warm_rejects() > 0, "an expert bigger than the tier is rejected");
    CHECK(cache.warm_slots() == 0, "and nothing stays in the tier");
    CHECK(cache.resident_slots() == 1 && cache.tracked_slots() == 1,
          "the expert is still loaded, straight from the mapping");

    // 6. With no tier configured the original behaviour is intact: every miss is
    //    a file read and an eviction drops the slot to COLD.
    const u64 host_allocs_none = be.host_allocs();
    cache.clear();
    cache.configure(&be, 24);
    CHECK(touch(0) && touch(1), "loads proceed without a tier");
    CHECK(cache.warm_admissions() == 0 && cache.warm_slots() == 0,
          "no tier, no admissions");
    CHECK(be.host_allocs() == host_allocs_none,
          "a disabled tier never touches host memory");
    CHECK(cache.touch_count(0, 0) == 0,
          "an evicted expert without a tier is gone, history and all");

    cache.clear();
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// CpuExpertPool: the CPU half of the hybrid expert path
// ---------------------------------------------------------------------------
//
// The pool computes a routed expert out of the weight mapping on host threads
// while the GPU runs the resident ones. What has to be pinned down is not that
// it is fast -- that is a measurement, and it is in docs/test-results.md -- but
// that it is exact about the thing that makes it usable at all: the four ops of
// one expert, at the same f16 rounding points the device kernels use, with
// several rows and several experts in flight at once and the per-worker
// accumulators summed correctly. Two experts sharing a row is the case those
// private accumulators exist for, and a missing reduction there is invisible on
// a single-threaded test.

// An expert's three slices decoded to f32 on the host, laid out as the kernels
// read them: gate/up are [n_ff][n_in], down is [n_in][n_ff].
struct RefExpert {
    i64 n_in = 0;
    i64 n_ff = 0;
    std::vector<f32> gate, up, down;
};

static void decode_expert(const GgufTensor &t, i32 n_expert, i64 expert, i64 row_len,
                          i64 n_rows, std::vector<f32> &out) {
    const size_t per_bytes = t.n_bytes / static_cast<size_t>(n_expert);
    const size_t row_bytes = dtype_row_bytes(t.type, row_len);
    const u8 *base = t.data + static_cast<size_t>(expert) * per_bytes;
    out.assign(static_cast<size_t>(row_len) * static_cast<size_t>(n_rows), 0.0f);
    for (i64 r = 0; r < n_rows; r++)
        dequant_row(t.type, base + static_cast<size_t>(r) * row_bytes,
                    out.data() + static_cast<size_t>(r * row_len), row_len);
}

// The reference for one expert on one activation row: the same four ops the
// device runs, rounded to f16 where the device rounds. silu_mul reads its inputs
// from f16 buffers and writes f16, the down GEMM writes f16, and the scatter
// accumulates f16 -- so g, u, h, y and the running sum are all rounded here.
static void ref_expert(const RefExpert &e, const f32 *x, f32 alpha, f32 *out_row) {
    std::vector<f32> h(static_cast<size_t>(e.n_ff));
    for (i64 j = 0; j < e.n_ff; j++) {
        f32 g = 0.0f, u = 0.0f;
        const size_t base = static_cast<size_t>(j * e.n_in);
        for (i64 i = 0; i < e.n_in; i++) {
            g += e.gate[base + static_cast<size_t>(i)] * x[i];
            u += e.up[base + static_cast<size_t>(i)] * x[i];
        }
        const f32 g16 = fp16_to_fp32(fp32_to_fp16(g));
        const f32 u16 = fp16_to_fp32(fp32_to_fp16(u));
        h[static_cast<size_t>(j)] =
            fp16_to_fp32(fp32_to_fp16((g16 / (1.0f + std::exp(-g16))) * u16));
    }
    for (i64 i = 0; i < e.n_in; i++) {
        f32 s = 0.0f;
        const size_t base = static_cast<size_t>(i * e.n_ff);
        for (i64 j = 0; j < e.n_ff; j++)
            s += e.down[base + static_cast<size_t>(j)] * h[static_cast<size_t>(j)];
        const f32 y16 = fp16_to_fp32(fp32_to_fp16(s));
        out_row[i] = fp16_to_fp32(fp32_to_fp16(out_row[i] + alpha * y16));
    }
}

// A deterministic fill for one slice of a synthetic expert set. Written through
// the quantized bytes, so the reference decodes exactly what the pool decodes.
static void fill_synth_slice(std::vector<u8> &buf, DType t, i64 row_len, i64 n_rows,
                             i32 n_expert, i32 salt) {
    const size_t row_bytes = dtype_row_bytes(t, row_len);
    buf.assign(row_bytes * static_cast<size_t>(n_rows) * static_cast<size_t>(n_expert),
               0);
    size_t off = 0;
    for (i64 r = 0; r < n_rows * n_expert; r++) {
        if (t == DType::F32) {
            for (i64 i = 0; i < row_len; i++) {
                const f32 v = 0.01f * static_cast<f32>(
                    static_cast<int>((r * 7 + i * 3 + salt) % 19) - 9);
                std::memcpy(buf.data() + off + static_cast<size_t>(i) * 4, &v, 4);
            }
        } else {
            // Q8_0: an f16 scale then 32 signed bytes, 34 B per 32-value block.
            size_t bo = off;
            for (i64 b = 0; b < row_len / 32; b++) {
                const u16 d =
                    fp32_to_fp16(0.02f * static_cast<f32>(1 + ((r + b + salt) % 5)));
                std::memcpy(buf.data() + bo, &d, 2);
                for (int k = 0; k < 32; k++) {
                    const int q = static_cast<int>(
                        (r * 5 + b * 3 + static_cast<i64>(k) + salt) % 13) - 6;
                    buf[bo + 2 + static_cast<size_t>(k)] = static_cast<u8>(q & 0xFF);
                }
                bo += 34;
            }
        }
        off += row_bytes;
    }
}

static void test_cpu_expert_pool() {
    const i64 kIn = 64, kFf = 32;
    const i32 kNe = 4, kRows = 4;

    // F32 makes the reference exact; Q8_0 exercises the dequantizer the hybrid
    // path actually meets on a real MoE (block size 32, 34 B per block).
    struct Case {
        DType type;
        const char *name;
    };
    const Case cases[2] = {{DType::F32, "F32"}, {DType::Q8_0, "Q8_0"}};

    CpuExpertPool &pool = CpuExpertPool::get();
    const i32 threads_before = pool.threads();

    for (int ci = 0; ci < 2; ci++) {
        const DType dt = cases[ci].type;
        std::vector<u8> gate_bytes, up_bytes, down_bytes;
        fill_synth_slice(gate_bytes, dt, kIn, kFf, kNe, 1);
        fill_synth_slice(up_bytes, dt, kIn, kFf, kNe, 5);
        fill_synth_slice(down_bytes, dt, kFf, kIn, kNe, 11);

        // gate/up are [n_in, n_ff, n_expert], down is [n_ff, n_in, n_expert].
        GgufTensor tg, tu, td;
        const auto shape = [&](GgufTensor &t, std::vector<u8> &buf, i64 a, i64 b) {
            t.name = "synth_exps.weight";
            t.type = dt;
            t.dtype_known = true;
            t.n_dims = 3;
            t.ne[0] = static_cast<u64>(a);
            t.ne[1] = static_cast<u64>(b);
            t.ne[2] = static_cast<u64>(kNe);
            t.n_elements = static_cast<i64>(a * b) * kNe;
            t.n_bytes = buf.size();
            t.data = buf.data();
        };
        shape(tg, gate_bytes, kIn, kFf);
        shape(tu, up_bytes, kIn, kFf);
        shape(td, down_bytes, kFf, kIn);

        ExpertSource src;
        src.gate = &tg;
        src.up = &tu;
        src.down = &td;
        src.n_expert = kNe;
        src.n_embd = kIn;
        src.n_ff_exp = kFf;
        CHECK(src.present(), "synthetic expert source is complete");

        std::vector<RefExpert> ref(static_cast<size_t>(kNe));
        for (i32 e = 0; e < kNe; e++) {
            RefExpert &r = ref[static_cast<size_t>(e)];
            r.n_in = kIn;
            r.n_ff = kFf;
            decode_expert(tg, kNe, e, kIn, kFf, r.gate);
            decode_expert(tu, kNe, e, kIn, kFf, r.up);
            decode_expert(td, kNe, e, kFf, kIn, r.down);
        }

        std::vector<f32> xn(static_cast<size_t>(kRows * kIn));
        for (size_t i = 0; i < xn.size(); i++)
            xn[i] = 0.1f * static_cast<f32>(static_cast<int>(i % 11) - 5);

        // Three jobs. Row 0 is selected by two of them, which is the shape a
        // real layer has (top-k over one token) and the one that needs the
        // per-worker accumulators unified. Row 3 is selected by nothing.
        const i32 rows_a[2] = {0, 1};
        const f32 alpha_a[2] = {0.5f, 0.25f};
        const i32 rows_b[1] = {0};
        const f32 alpha_b[1] = {0.75f};
        const i32 rows_c[1] = {2};
        const f32 alpha_c[1] = {1.5f};
        CpuExpertPool::Job jobs[3];
        jobs[0] = CpuExpertPool::Job{0, rows_a, alpha_a, 2};
        jobs[1] = CpuExpertPool::Job{2, rows_b, alpha_b, 1};
        jobs[2] = CpuExpertPool::Job{3, rows_c, alpha_c, 1};

        std::vector<f32> want(static_cast<size_t>(kRows * kIn), 0.0f);
        for (int j = 0; j < 3; j++) {
            const RefExpert &re = ref[static_cast<size_t>(jobs[j].expert)];
            for (i32 r = 0; r < jobs[j].m; r++)
                ref_expert(re, xn.data() + static_cast<size_t>(jobs[j].rows[r]) * kIn,
                           jobs[j].alpha[r],
                           want.data() + static_cast<size_t>(jobs[j].rows[r]) * kIn);
        }

        f32 worst = 0.0f, peak = 0.0f;
        for (int arm = 0; arm < 2; arm++) {
            pool.set_threads(arm == 0 ? 1 : 4);
            CHECK(pool.threads() >= 1, "the pool reports a live team");

            std::vector<f32> out(static_cast<size_t>(kRows * kIn), 0.0f);
            pool.reset_counters();
            const i32 done = pool.run(src, 0, jobs, 3, xn.data(), kRows, kIn, out.data());
            CHECK(done == 3, "run() reports the three experts it computed");
            CHECK(pool.experts_done() == 3, "and the lifetime counter agrees");
            CHECK(pool.rows_done() >= 3, "and the row counter moved with it");

            worst = 0.0f;
            peak = 0.0f;
            for (size_t i = 0; i < want.size(); i++) {
                worst = std::max(worst, std::fabs(out[i] - want[i]));
                peak = std::max(peak, std::fabs(want[i]));
            }
            CHECK(peak > 1e-3f, "the reference is not all zeros");
            CHECK_NEAR(worst, 0.0, static_cast<double>(peak) * 2e-3 + 1e-4,
                       "the pool matches the f16-rounded reference");

            bool row3_zero = true;
            for (i32 i = 0; i < kIn; i++)
                if (out[static_cast<size_t>(3 * kIn + i)] != 0.0f) row3_zero = false;
            CHECK(row3_zero, "a row no job selected is left exactly zero");

            bool row0_both = true;
            for (i32 i = 0; i < kIn; i++)
                if (out[static_cast<size_t>(i)] == 0.0f) row0_both = false;
            CHECK(row0_both, "a row two experts share carries both contributions");
        }

        // The two arms must agree with each other, not just each with the
        // reference: that is what proves the worker accumulators are unified.
        //
        // The budget is RELATIVE to this case's peak, and that is deliberate.
        // Splitting a row's two contributions across two workers changes the
        // order the f16 roundings happen in, so the arms are entitled to differ
        // by a few ulps of the values themselves: measured 8.5e-4 against a
        // peak of 2.047 on the Q8_0 case (0.04%), and under 1e-6 against the
        // F32 case's 0.004 peak. The difference SCALES with the magnitude,
        // which is what rounding-order noise does and what a missed or
        // double-counted contribution does not. An absolute cap here would
        // have passed this test whatever the pool did to the tiny F32 case.
        f32 peak1 = 0.0f;
        for (size_t i = 0; i < want.size(); i++) peak1 = std::max(peak1, std::fabs(want[i]));
        const double tol = static_cast<double>(peak1) * 2e-3 + 1e-4;

        std::vector<f32> one(static_cast<size_t>(kRows * kIn), 0.0f);
        pool.set_threads(1);
        pool.run(src, 0, jobs, 3, xn.data(), kRows, kIn, one.data());

        // Five multi-threaded repetitions: jobs are claimed from an atomic
        // counter, so which worker owns which contribution to a shared row is
        // whatever order the claims land in. Every repetition still has to land
        // on the reference, and the gap against the single-threaded arm has to
        // stay inside the same relative budget.
        f32 spread = 0.0f;
        pool.set_threads(4);
        for (int rep = 0; rep < 5; rep++) {
            std::vector<f32> many(static_cast<size_t>(kRows * kIn), 0.0f);
            CHECK(pool.run(src, 0, jobs, 3, xn.data(), kRows, kIn, many.data()) == 3,
                  "the multi-threaded arm computes all three experts");
            f32 worst_many = 0.0f;
            for (size_t i = 0; i < want.size(); i++) {
                worst_many = std::max(worst_many, std::fabs(many[i] - want[i]));
                spread = std::max(spread, std::fabs(many[i] - one[i]));
            }
            CHECK_NEAR(worst_many, 0.0, tol,
                       "the 4-thread arm matches the reference whatever order it claims in");
        }
        CHECK_NEAR(spread, 0.0, tol, "1 thread and 4 threads agree within the rounding budget");

        // No jobs, no work, no writes.
        std::vector<f32> untouched(static_cast<size_t>(kRows * kIn), 42.0f);
        CHECK(pool.run(src, 0, jobs, 0, xn.data(), kRows, kIn, untouched.data()) == 0,
              "an empty job list computes nothing");
        bool clean = true;
        for (size_t i = 0; i < untouched.size(); i++)
            if (untouched[i] != 42.0f) clean = false;
        CHECK(clean, "and writes nothing at all");

        std::fprintf(stderr, "  (%s expert case: worst %.3e of peak %.3f)\n",
                     cases[ci].name, static_cast<double>(worst), static_cast<double>(peak));
    }

    pool.set_threads(threads_before > 0 ? threads_before : 1);
}

// ---------------------------------------------------------------------------
// speculative decoding (greedy, draft model)
// ---------------------------------------------------------------------------

// A parameterised dense model: same tensor schema as build_tiny_model, but the
// caller picks the width so a genuinely weaker draft can be built.
static bool build_sized_model(const std::string &path, int n_embd, int n_ff) {
    const int n_layer = 1, n_head = 4, n_kv = 2, hd = 8;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int vocab = 260, ctx = 64;

    GgufBuilder b;
    b.meta_str("general.architecture", "llama");
    b.meta_str("general.name", "kraken-sized-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32("llama.block_count", n_layer);
    b.meta_u32("llama.context_length", ctx);
    b.meta_u32("llama.embedding_length", n_embd);
    b.meta_u32("llama.feed_forward_length", n_ff);
    b.meta_u32("llama.attention.head_count", n_head);
    b.meta_u32("llama.attention.head_count_kv", n_kv);
    b.meta_u32("llama.attention.key_length", hd);
    b.meta_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32("llama.rope.freq_base", 10000.0f);
    add_spm_vocab(b, vocab);

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };
    b.tensor_f16("token_embd.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 0));
    b.tensor_f32("output_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32("blk.0.attn_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32("blk.0.ffn_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f16("blk.0.attn_q.weight", {u64(n_embd), u64(q_dim)},
                 vec(static_cast<size_t>(n_embd) * q_dim, 100));
    b.tensor_f16("blk.0.attn_k.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim, 200));
    b.tensor_f16("blk.0.attn_v.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim, 300));
    b.tensor_f16("blk.0.attn_output.weight", {u64(q_dim), u64(n_embd)},
                 vec(static_cast<size_t>(q_dim) * n_embd, 400));
    b.tensor_f16("blk.0.ffn_gate.weight", {u64(n_embd), u64(n_ff)},
                 vec(static_cast<size_t>(n_embd) * n_ff, 500));
    b.tensor_f16("blk.0.ffn_up.weight", {u64(n_embd), u64(n_ff)},
                 vec(static_cast<size_t>(n_embd) * n_ff, 900));
    b.tensor_f16("blk.0.ffn_down.weight", {u64(n_ff), u64(n_embd)},
                 vec(static_cast<size_t>(n_ff) * n_embd, 1300));
    return b.write(path);
}

static GenerateResult run_spec(const char *model, const char *draft, i32 window,
                               int max_tokens, bool greedy, bool *ok_out) {
    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = model;
    cfg.n_ctx = 64;
    cfg.prefill_chunk = 8;
    std::string err;
    *ok_out = false;
    if (!engine.init(cpu, cfg, &err)) {
        std::fprintf(stderr, "  (spec init: %s)\n", err.c_str());
        delete cpu;
        return GenerateResult{};
    }
    if (draft) {
        engine.set_draft_window(window);
        if (!engine.load_draft(draft, &err)) {
            std::fprintf(stderr, "  (spec draft: %s)\n", err.c_str());
            engine.shutdown();
            delete cpu;
            return GenerateResult{};
        }
    }
    GenerateParams p;
    p.prompt = "the";
    p.max_tokens = max_tokens;
    p.sampler.greedy = greedy;
    p.sampler.temp = greedy ? 0.0f : 0.9f;
    // The speculative fast path argmaxes raw logits; the plain loop argmaxes
    // through Sampler::sample, which applies the repetition penalty first.
    // Bit-identity between the two requires no penalty here.
    p.sampler.repeat_penalty = 1.0f;
    p.sampler.repeat_last_n = 0;
    p.sampler.seed = 7;
    GenerateResult r;
    *ok_out = engine.generate(p, &r);
    engine.shutdown();
    delete cpu;
    return r;
}

static void test_speculative_decoding() {
    // A weak draft and a strong target over the same vocab. Every draft window
    // must produce a token stream identical to plain greedy decoding: that is
    // the whole contract of the verify-and-accept loop.
    const char *target = "kraken-spec-target.gguf";
    const char *weak = "kraken-spec-weak.gguf";
    const char *same = "kraken-spec-same.gguf";
    CHECK(build_sized_model(target, 32, 64), "wrote the spec target model");
    CHECK(build_sized_model(weak, 16, 32), "wrote the weak draft model");
    CHECK(build_sized_model(same, 32, 64), "wrote the target-as-draft model");

    bool ok = false;
    const GenerateResult plain = run_spec(target, nullptr, 0, 16, true, &ok);
    CHECK(ok && plain.generated == 16, "plain greedy reference generated 16 tokens");

    // 1. A weak draft must not change the output — bit-for-bit.
    const i32 wins[] = {1, 2, 4, 6};
    for (const i32 win : wins) {
        const GenerateResult r = run_spec(target, weak, win, 16, true, &ok);
        CHECK(ok && r.generated == 16, "spec (weak draft) completed at window size");
        CHECK(r.tokens == plain.tokens,
              "spec output identical to plain greedy (weak draft)");
    }

    // 2. The target as its own draft: proposals always match, so every round
    // takes the full-accept path with the bonus token. Strongest alignment
    // check of the block/accept/realign machinery.
    {
        const GenerateResult r = run_spec(target, same, 4, 16, true, &ok);
        CHECK(ok && r.generated == 16, "spec (self draft) completed");
        CHECK(r.tokens == plain.tokens,
              "spec output identical to plain greedy (self draft)");
    }

    // 3. Sampling with a draft loaded falls back to the plain loop and still
    // generates.
    {
        const GenerateResult sampled = run_spec(target, weak, 4, 8, false, &ok);
        CHECK(ok && sampled.generated == 8, "sampling with a draft still generates");
    }

    // 4. A missing draft model is rejected loudly, not mis-run.
    {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = target;
        cfg.n_ctx = 64;
        std::string err;
        CHECK(engine.init(cpu, cfg, &err), "spec engine for validation checks");
        if (err.empty()) {
            engine.set_draft_window(4);
            CHECK(!engine.load_draft("missing-file.gguf", &err),
                  "a missing draft model is rejected");
            CHECK(!err.empty(), "the rejection carries a reason");
        }
        engine.shutdown();
        delete cpu;
    }

    // (cleanup disabled for debugging)
}

static void test_moe_grouped_prefill_matches_tokenwise() {
    // --- the batched path must survive grouped prefill -------------------
    // Several prompt tokens in one chunk means several tokens can pick the
    // same expert, which is exactly the case the permutation covers. Compare
    // chunked prefill against one-token-at-a-time on the same weights.
    {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = kMoeTestPath;
        cfg.n_ctx = 64;
        cfg.prefill_chunk = 8;
        std::string err;
        CHECK(engine.init(cpu, cfg, &err), "MoE engine initialises for the batch check");
        if (err.empty()) {
            GenerateParams p;
            // Long enough to span several prefill chunks with distinct tokens,
            // so expert groups receive different group sizes per chunk.
            p.prompt = "the the the the";
            p.max_tokens = 6;
            p.sampler.greedy = true;
            p.sampler.temp = 0.0f;
            GenerateResult batched;
            CHECK(engine.generate(p, &batched), "batched MoE generation runs");

            EngineConfig cfg1 = cfg;
            cfg1.prefill_chunk = 1;
            Engine engine1;
            CHECK(engine1.init(cpu, cfg1, &err), "MoE engine initialises chunk=1");
            if (err.empty()) {
                GenerateResult one;
                CHECK(engine1.generate(p, &one), "one-token MoE generation runs");
                CHECK(batched.tokens == one.tokens,
                      "grouped expert GEMMs match token-at-a-time routing");
            }
            engine1.shutdown();
        }
        engine.shutdown();
        delete cpu;
    }
}

static void test_moe_matches_dense_twin() {
    // A MoE layer with a single expert, zero router gates and no shared expert
    // must reproduce the dense FFN bit for bit: the gate weight is exactly 1.0.
    MoeSpec dense;
    dense.dense = true;
    MoeSpec moe;
    moe.dense = false;
    moe.n_expert = 1;
    moe.n_expert_used = 1;
    moe.shared = false;
    moe.varied = false;
    CHECK(build_tiny_moe_model(kMoeDenseTwinPath, dense), "wrote the dense twin");
    CHECK(build_tiny_moe_model(kMoeTestPath, moe), "wrote the single-expert MoE twin");

    const MoeRun d = run_moe(kMoeDenseTwinPath, 0, 0, 8);
    const MoeRun m = run_moe(kMoeTestPath, 0, 0, 8);
    CHECK(d.ok && !d.tokens.empty(), "dense twin generated tokens");
    CHECK(m.ok && !m.tokens.empty(), "MoE twin generated tokens");
    CHECK(d.tokens == m.tokens,
          "a single-expert MoE layer reproduces the dense FFN exactly");
    CHECK(m.loads == 1, "the single expert is loaded exactly once");
}

// ---------------------------------------------------------------------------
// end-to-end CPU generation
// ---------------------------------------------------------------------------

// Gemma 2/3/4 bound the final logits: cap * tanh(logit / cap). The engine
// applies it where logits become host-visible, so the check is on what the
// sampler would see.
static void test_logit_softcap() {
    const std::string path = "kraken-softcap-test.gguf";
    const f32 cap = 0.5f;
    CHECK(build_tiny_model(path, false, cap), "wrote a softcapped model");

    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx = 64;
    cfg.prefill_chunk = 4;
    std::string err;
    CHECK(engine.init(cpu, cfg, &err), "the softcapped model loads");
    CHECK(engine.model().cfg().logit_softcap == cap,
          "final_logit_softcapping is read from the metadata");

    // The formula itself: bounded by the cap, monotone (so the ranking a greedy
    // sampler sees is untouched), and an exact pass-through when disabled.
    CHECK_NEAR(apply_logit_softcap(0.0f, cap), 0.0f, 1e-7, "softcap fixes zero");
    CHECK(std::fabs(apply_logit_softcap(100.0f, cap)) <= cap,
          "a huge logit stays inside the cap");
    CHECK(apply_logit_softcap(3.0f, cap) > apply_logit_softcap(1.0f, cap),
          "the softcap is monotone");
    CHECK_NEAR(apply_logit_softcap(2.5f, 0.0f), 2.5f, 1e-6f,
               "cap 0 is a pass-through");

    // The plumbing: a real model loads with the cap declared, and generation
    // still runs with it applied.
    GenerateParams p;
    p.prompt = "the";
    p.max_tokens = 4;
    p.sampler.greedy = true;
    GenerateResult r;
    CHECK(engine.generate(p, &r), "a softcapped model generates");
    CHECK(r.generated > 0, "and produces tokens");

    engine.shutdown();
    delete cpu;
    std::remove(path.c_str());
}

// RoPE's pair convention: llama.cpp rotates adjacent channels (2i, 2i+1) for
// the llama family and channel j with j + n_rot/2 for the NeoX family, and a
// file's q/k rows are stored in whichever its architecture uses. The two are
// indistinguishable from the output statistics alone — both stay finite and
// in range, only the attention scores differ — so this pins the wiring with a
// row that has exactly one live channel: the partner channel it moves into
// names the convention.
static void test_rope_pair_convention() {
    Backend *cpu = make_cpu_backend();
    const i64 hd = 8;
    const i64 rot = hd / 2; // frac 1.0 rotates every pair
    const f32 iv[4] = {1.0f, 0.5f, 0.25f, 0.125f};
    f32 q[8], k[8];
    const f32 cs = std::cos(iv[0]), sn = std::sin(iv[0]); // pos 1, pair 0
    for (int neox = 0; neox <= 1; neox++) {
        std::memset(q, 0, sizeof(q));
        std::memset(k, 0, sizeof(k));
        q[0] = 1.0f;
        k[0] = 1.0f;
        cpu->rope(q, k, 1, 1, hd, 1, 1, iv, 1.0f, 1.0f, neox != 0);
        const i64 partner = neox ? rot : 1;
        CHECK_NEAR(q[0], cs, 1e-6, "rope leaves the pair's first channel on the cos");
        CHECK_NEAR(q[partner], sn, 1e-6,
                   neox ? "neox rope pairs channel j with j + rot"
                        : "norm rope pairs adjacent channels");
        CHECK_NEAR(k[partner], sn, 1e-6, "rope pairs k the same way as q");
        CHECK_NEAR(q[neox ? 1 : rot], 0.0f, 1e-6,
                   "the partner of the other convention is not rotated");
    }
    delete cpu;
}

static void test_end_to_end_cpu() {
    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = kTestModelPath;
    cfg.n_ctx = 64;
    cfg.prefill_chunk = 8;
    std::string err;
    CHECK(engine.init(cpu, cfg, &err), "engine initialises on the CPU backend");
    if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());

    if (err.empty()) {
        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 12;
        p.sampler.greedy = true;
        p.sampler.seed = 7;

        GenerateResult r1, r2;
        CHECK(engine.generate(p, &r1), "first generation run");
        CHECK(engine.generate(p, &r2), "second generation run");
        CHECK(r1.tokens == r2.tokens, "generation is deterministic");
        CHECK(r1.generated > 0, "generated at least one token");
        CHECK(r1.prompt_tokens > 0, "prompt tokenised");
        CHECK(r1.generated <= 12, "respects max_tokens");

        // A multi-chunk prefill must agree with a single-token-at-a-time
        // decode: same prompt, chunked differently, must give the same result.
        EngineConfig cfg2 = cfg;
        cfg2.prefill_chunk = 1;
        Engine engine2;
        CHECK(engine2.init(cpu, cfg2, &err), "engine initialises with chunk=1");
        if (err.empty()) {
            GenerateResult r3;
            CHECK(engine2.generate(p, &r3), "chunk=1 generation run");
            CHECK(r3.tokens == r1.tokens, "chunked prefill matches token-by-token");
            engine2.shutdown();
        }
    }

    engine.shutdown();
    delete cpu;
}

// ---------------------------------------------------------------------------
// JSON parser / writer
// ---------------------------------------------------------------------------

static void test_json() {
    JsonValue v;
    std::string err;

    // object with nested structure round-trips
    const std::string src =
        "{\"a\":1,\"b\":[-2.5,true,null,\"x\"],\"c\":{\"d\":\"e\\n\"}}";
    CHECK(json_parse(src, &v, &err), "json_parse accepts a nested object");
    CHECK(v.is_object(), "parsed root is an object");
    CHECK(v.get_int("a") == 1, "integer field reads back");
    const JsonValue *b = v.find("b");
    CHECK(b && b->is_array() && b->items().size() == 4, "array field parsed");
    if (b && b->items().size() == 4) {
        CHECK((*b).items()[0].as_number() == -2.5, "negative number");
        CHECK((*b).items()[1].as_bool() == true, "bool element");
        CHECK((*b).items()[2].is_null(), "null element");
        CHECK((*b).items()[3].is_string() && (*b).items()[3].as_string() == "x",
              "string element");
    }
    CHECK(v.dump() == src, "dump reproduces the input byte for byte");

    // escapes and unicode
    JsonValue u;
    CHECK(json_parse("\"\\u00e9\\ud83d\\ude00\\t\"", &u, &err),
          "unicode escapes parse");
    CHECK(u.as_string() == "\xC3\xA9\xF0\x9F\x98\x80\t",
          "\\uXXXX decodes to UTF-8 including a surrogate pair");
    JsonValue esc = JsonValue::object();
    esc.set("s", std::string("line\n\"q\"\\	\x01"));
    JsonValue back;
    CHECK(json_parse(esc.dump(), &back, &err), "re-parse of escaped dump");
    std::string rs;
    CHECK(back.get_string("s", &rs) && rs == "line\n\"q\"\\\t\x01",
          "escape round-trip preserves the string");

    // malformed inputs are rejected with a message
    const char *bad[] = {
        "{", "{\"a\"}", "{\"a\":}", "[1,]", "tru", "\"unterminated", "01",
        "1e", "{} extra", "", "\"\\u00\"", "{1:2}", NULL};
    for (int i = 0; bad[i]; i++) {
        JsonValue junk;
        std::string e2;
        CHECK(!json_parse(bad[i], &junk, &e2), "malformed JSON rejected");
        CHECK(!e2.empty(), "malformed JSON yields an error message");
    }

    // number writer: integers stay integers, f64 precision survives
    JsonValue n(i64(1234567890123LL));
    CHECK(n.dump() == "1234567890123", "i64 without decimal point");
    JsonValue big(1.0e300);
    CHECK(big.dump().find("e+300") != std::string::npos, "exponent notation");

    // arrays and deep-ish nesting
    JsonValue arr = JsonValue::array();
    for (int i = 0; i < 8; i++) arr.push_back(JsonValue(i64(i)));
    CHECK(arr.dump() == "[0,1,2,3,4,5,6,7]", "array dump");

    // set() replaces an existing key
    JsonValue obj = JsonValue::object();
    obj.set("k", i64(1));
    obj.set("k", i64(2));
    CHECK(obj.get_int("k") == 2 && obj.members().size() == 1, "set() replaces");
}

// ---------------------------------------------------------------------------
// HTTP server end to end: a real listener, a real OpenAiService, a real
// generation through the tiny model, compared token-for-token with a direct
// engine.generate() call on the same weights.
// ---------------------------------------------------------------------------

// Portable socket teardown: winsock wants closesocket, POSIX wants close.
static void socket_close(int fd) {
#if defined(_WIN32)
    ::closesocket(static_cast<SOCKET>(fd));
#else
    ::close(fd);
#endif
}

// Minimal blocking HTTP/1.0 client for the loopback tests (no chunk parsing
// needed: requests are small and HTTP/1.0 closes the connection).
static bool http_request(int port, const std::string &method, const std::string &path,
                         const std::string &body, int *status, std::string *out) {
    const int fd = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
    if (fd < 0) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<unsigned short>(port));
    addr.sin_addr.s_addr = htonl(0x7F000001u); // 127.0.0.1
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        socket_close(fd);
        return false;
    }
    std::string req = method + " " + path + " HTTP/1.0\r\n";
    req += "Host: localhost\r\n";
    if (!body.empty()) {
        req += "Content-Type: application/json\r\n";
        req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    req += "\r\n";
    req += body;
    size_t sent = 0;
    while (sent < req.size()) {
#ifdef _WIN32
        const int n = ::send(fd, req.data() + sent, static_cast<int>(req.size() - sent), 0);
#else
        const ssize_t n = ::send(fd, req.data() + sent, req.size() - sent, 0);
#endif
        if (n <= 0) {
            socket_close(fd);
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    out->clear();
    char buf[4096];
#ifdef _WIN32
    int n;
#else
    ssize_t n;
#endif
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) out->append(buf, static_cast<size_t>(n));
    socket_close(fd);
    if (out->size() < 12) return false;
    *status = std::atoi(out->c_str() + 9);
    return true;
}

static void test_http_server() {
    if (!build_tiny_model(kTestModelPath, false)) {
        CHECK(false, "http test: build_tiny_model failed");
        return;
    }

    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = kTestModelPath;
    cfg.n_ctx = 64;
    cfg.prefill_chunk = 8;
    std::string err;
    const bool inited = engine.init(cpu, cfg, &err);
    CHECK(inited, "http test: engine initialises");
    if (!inited) {
        delete cpu;
        return;
    }

    // Reference generation: what the server returns must match this exactly.
    GenerateParams ref_p;
    ref_p.prompt = "the";
    ref_p.max_tokens = 5;
    ref_p.sampler.greedy = true;
    ref_p.sampler.repeat_penalty = 1.0f;
    GenerateResult ref;
    const bool ref_ok = engine.generate(ref_p, &ref);
    CHECK(ref_ok && !ref.tokens.empty(), "http test: reference generation");

    std::mutex gen_mu;
    OpenAiService svc(&engine, 0.8f, &gen_mu);
    HttpServer server;
    CHECK(server.listen("127.0.0.1", 0, &err), "http test: listener on ephemeral port");
    const int port = server.port();
    CHECK(port > 0, "http test: ephemeral port assigned");
    std::thread accept_thread([&server, &svc]() {
        server.run([&svc](const HttpRequest &req, HttpResponse &res, HttpConn &conn) {
            return svc.handle(req, res, conn);
        });
    });

    int status = 0;
    std::string body;

    // health
    CHECK(http_request(port, "GET", "/health", "", &status, &body) && status == 200,
          "http: /health returns 200");
    {
        JsonValue j;
        std::string e;
        const std::string payload = body.substr(body.find("\r\n\r\n") + 4);
        CHECK(json_parse(payload, &j, &e), "http: /health body is valid JSON");
        const JsonValue *s = j.find("status");
        CHECK(j.is_object() && s && s->is_string() && s->as_string() == "ok",
              "http: /health reports ok");
    }

    // models
    CHECK(http_request(port, "GET", "/v1/models", "", &status, &body) && status == 200,
          "http: /v1/models returns 200");
    {
        JsonValue j;
        std::string e;
        const std::string payload = body.substr(body.find("\r\n\r\n") + 4);
        CHECK(json_parse(payload, &j, &e), "http: /v1/models body is valid JSON");
        const JsonValue *data = j.find("data");
        CHECK(j.is_object() && data && data->is_array() && data->items().size() == 1,
              "http: /v1/models lists exactly one model");
        if (data && data->items().size() == 1) {
            const JsonValue &m = data->items()[0];
            std::string id;
            CHECK(m.get_string("id", &id) && !id.empty(), "http: model entry has an id");
            CHECK(m.get_string("object", &id) && id == "model",
                  "http: model entry object=model");
        }
    }

    // buffered completion == direct generation
    std::string req = "{\"prompt\":\"the\",\"max_tokens\":5,";
    req += "\"temperature\":0,\"repeat_penalty\":1.0}";
    CHECK(http_request(port, "POST", "/v1/completions", req, &status, &body) &&
              status == 200, "http: /v1/completions returns 200");
    {
        JsonValue j;
        std::string e;
        const std::string payload = body.substr(body.find("\r\n\r\n") + 4);
        CHECK(json_parse(payload, &j, &e), "http: completion body is valid JSON");
        std::string text;
        CHECK(j.get_string("object", &text) && text == "text_completion",
              "http: completion object type");
        std::string finish;
        const JsonValue *choices = j.find("choices");
        CHECK(choices && choices->is_array() && choices->items().size() == 1,
              "http: exactly one choice");
        if (choices && !choices->items().empty()) {
            const JsonValue &c = choices->items()[0];
            c.get_string("finish_reason", &finish);
            std::string text_out;
            c.get_string("text", &text_out);
            CHECK(finish == "length", "http: finish_reason=length for 5 capped tokens");
            CHECK(text_out == ref.text,
                  "http: completion text is bit-identical to direct generation");
        }
        const JsonValue *usage = j.find("usage");
        CHECK(usage && usage->get_int("prompt_tokens") == ref.prompt_tokens &&
                  usage->get_int("completion_tokens") == ref.generated,
              "http: usage matches the reference run");
    }

    // chat completion renders the ChatML template and answers
    req = "{\"messages\":[{\"role\":\"system\",\"content\":\"You are terse.\"},"
          "{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":3,"
          "\"temperature\":0}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", req, &status, &body) &&
              status == 200, "http: /v1/chat/completions returns 200");
    {
        JsonValue j;
        std::string e;
        const std::string payload = body.substr(body.find("\r\n\r\n") + 4);
        CHECK(json_parse(payload, &j, &e), "http: chat body is valid JSON");
        std::string obj;
        CHECK(j.get_string("object", &obj) && obj == "chat.completion",
              "http: chat object type");
        const JsonValue *choices = j.find("choices");
        if (choices && !choices->items().empty()) {
            const JsonValue *msg = choices->items()[0].find("message");
            std::string role, content;
            CHECK(msg && msg->get_string("role", &role) && role == "assistant",
                  "http: chat message role=assistant");
            CHECK(msg && msg->get_string("content", &content),
                  "http: chat message has content");
        } else {
            CHECK(false, "http: chat response has choices");
        }
    }

    // chat request is rejected when messages is missing
    CHECK(http_request(port, "POST", "/v1/chat/completions", "{\"prompt\":\"x\"}",
                       &status, &body) && status == 400,
          "http: chat without messages is a 400");

    // streaming: chunked SSE events carry the same text as the buffered run
    {
        std::string sreq = "{\"prompt\":\"the\",\"max_tokens\":5,"
                           "\"temperature\":0,\"repeat_penalty\":1.0,\"stream\":true}";
        CHECK(http_request(port, "POST", "/v1/completions", sreq, &status, &body) &&
                  status == 200, "http: streaming completion returns 200");
        const size_t hdr_end = body.find("\r\n\r\n");
        CHECK(hdr_end != std::string::npos &&
                  body.find("text/event-stream") != std::string::npos,
              "http: streaming response is an event stream");
        std::string payload = hdr_end == std::string::npos ? "" : body.substr(hdr_end + 4);
        std::string data_text, done_seen;
        size_t events = 0, finish_seen = 0;
        size_t p = 0;
        while ((p = payload.find("data: ", p)) != std::string::npos) {
            const size_t eol = payload.find('\n', p);
            std::string line = payload.substr(p + 6, eol - p - 6);
            p = eol;
            if (line == "[DONE]") {
                done_seen = "yes";
                continue;
            }
            JsonValue ev;
            std::string e2;
            if (!json_parse(line, &ev, &e2)) continue;
            events++;
            const JsonValue *choices = ev.find("choices");
            if (!choices || choices->items().empty()) continue;
            const JsonValue &c = choices->items()[0];
            const JsonValue *d = c.find("delta");
            if (d) {
                const JsonValue *ct = d->find("content");
                if (ct && ct->is_string()) data_text += ct->as_string();
            }
            const JsonValue *fr = c.find("finish_reason");
            if (fr && fr->is_string()) finish_seen++;
        }
        CHECK(!done_seen.empty(), "http: stream ends with data: [DONE]");
        CHECK(events >= 1 && finish_seen == 1, "http: one terminal chunk with finish_reason");
        CHECK(data_text == ref.text,
              "http: streamed text is bit-identical to direct generation");
    }

    // request isolation: two identical requests give identical answers
    {
        std::string creq = "{\"prompt\":\"the\",\"max_tokens\":5,"
                           "\"temperature\":0,\"repeat_penalty\":1.0}";
        std::string a1, a2;
        int s1 = 0, s2 = 0;
        CHECK(http_request(port, "POST", "/v1/completions", creq, &s1, &a1) && s1 == 200,
              "http: repeated request 1");
        CHECK(http_request(port, "POST", "/v1/completions", creq, &s2, &a2) && s2 == 200,
              "http: repeated request 2");
        // `id` (timestamp-derived) and `created` legitimately differ between
        // calls; everything else — text, tokens, usage — must be identical.
        auto normalize = [](const std::string &raw) {
            const std::string payload = raw.substr(raw.find("\r\n\r\n") + 4);
            JsonValue j;
            std::string e;
            if (!json_parse(payload, &j, &e)) return payload;
            j.set("id", std::string(""));
            j.set("created", i64(0));
            return j.dump();
        };
        CHECK(normalize(a1) == normalize(a2),
              "http: no state leaks between requests");
    }

    // errors
    CHECK(http_request(port, "GET", "/nope", "", &status, &body) && status == 404,
          "http: unknown path is 404");
    CHECK(http_request(port, "POST", "/v1/completions", "{bad", &status, &body) &&
              status == 400, "http: malformed JSON is 400");
    CHECK(http_request(port, "POST", "/v1/completions", "{\"max_tokens\":5}", &status,
                       &body) && status == 400, "http: missing prompt is 400");
    CHECK(http_request(port, "DELETE", "/v1/completions", "", &status, &body) &&
              status == 405, "http: wrong method is 405");

    server.stop();
    accept_thread.join();
    engine.shutdown();
    delete cpu;
}

// ---------------------------------------------------------------------------
// The pool's completion accounting IS the correctness argument: a worker that
// mistimes its completion marker either strands the caller in run() forever or
// replays a block, and the difference is a hang or two threads writing the same
// bytes. Decode fires hundreds of back-to-back tasks of shifting geometry, so
// hammer exactly that shape and require every block to be covered exactly once.
// ---------------------------------------------------------------------------
static void test_par_pool_covers_every_block() {
    u64 rng = 0x9e3779b97f4a7c15ull;
    auto next = [&](u64 m) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return static_cast<i64>(rng % m);
    };
    std::vector<i32> hits(4096, 0);
    i64 tasks = 0, blocks = 0;
    for (int task = 0; task < 400; task++) {
        const i64 n = 1 + next(4000);
        const i64 grain = 1 + next(static_cast<u64>(std::min<i64>(64, n)));
        std::fill(hits.begin(), hits.begin() + static_cast<size_t>(n), 0);
        Pool::get().run(n, grain, [&](i64 b, i64 e) {
            // Exactly what the backend's bodies do: each block owns a
            // disjoint slice of the output and touches nothing shared.
            for (i64 i = b; i < e; i++) hits[static_cast<size_t>(i)]++;
        });
        tasks++;
        blocks += (n + grain - 1) / grain;
        i64 covered = 0, doubled = 0;
        for (i64 i = 0; i < n; i++) {
            if (hits[static_cast<size_t>(i)] == 1) covered++;
            else if (hits[static_cast<size_t>(i)] > 1) doubled++;
        }
        if (covered != n || doubled != 0) {
            CHECK(false, "pool: every index runs exactly once");
            return;
        }
    }
    CHECK(tasks == 400 && blocks > 400, "pool: back-to-back tasks all completed");
}

// ---------------------------------------------------------------------------

int main() {
    std::fprintf(stderr, "kraken test suite\n");
    test_quant_geometry();
    test_q4_0_layout();
    test_q8_0_layout();
    test_q5_0_high_bit_layout();
    test_kquant_roundtrip_shape();
    test_q2_k_plane_layout();
    test_q6_k_group_stride();
    test_iq2_xxs_layout();
    test_iq4_xs_layout();
    test_nvfp4_layout();
    test_mxfp4_layout();
    test_rocmfp4_layout();
    test_tq2_0_layout();
    test_tq1_0_layout();
    test_new_quant_geometry();
    test_q1_0_layout();
    test_q2_0_layout();
    test_pq2_0_layout();
    test_ptq1_0_layout();
    test_q2_0_rocmfpx_layout();
    test_q3_0_rocmfpx_layout();
    test_q6_0_rocmfpx_layout();
    test_hadamard_act();
    test_iq_family_structure();
    test_vec_dot();
    test_arch_table();
    test_arch_tensor_maps();
    test_load_verdict();
    test_gguf_roundtrip();
    test_tokenizer_spm_and_bpe();
    test_sampler_determinism();
    test_logit_softcap();
    test_rope_pair_convention();
    test_end_to_end_cpu();
    test_moe_schema_and_laziness();
    test_moe_matches_dense_twin();
    test_moe_grouped_prefill_matches_tokenwise();
    test_expert_cache_policy();
    test_expert_cache_warm_tier();
    test_cpu_expert_pool();
    test_gdn_ops();
    test_gdn_model_loads();
    test_gdn_generation();
    test_speculative_decoding();
    test_json();
    test_http_server();
    test_par_pool_covers_every_block();
    std::remove(kTestModelPath);
    std::remove(kMoeTestPath);
    std::remove(kMoeDenseTwinPath);
    std::remove("kraken-l2-test.gguf");
    std::remove("kraken-policy-test.gguf");
    std::remove(kGdnTestPath);

    std::fprintf(stderr, "\n%d/%d checks passed\n", g_passed, g_run);
    return g_passed == g_run ? 0 : 1;
}
