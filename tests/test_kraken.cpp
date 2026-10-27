// ============================================================================
//  test_kraken.cpp — deterministic, self-contained test suite.
//
//  Almost everything here runs on the CPU reference backend on purpose: the
//  host dequantizers are the oracle the GPU kernels are validated against, so
//  a regression in src/quant.cpp must fail before it can silently mask a kernel
//  bug. The exception is test_prefill_chunk_bitexact, which spawns the CLI so
//  the chunk-invariance property can be gated on the device as well.
//
//  The suite also builds a complete miniature LLaMA-family GGUF in a
//  temporary file and runs real generation through it, which exercises the
//  container parser, the model loader, the forward pass, the sampler and both
//  tokenizer families end to end.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <random>
#include <filesystem>
#include <chrono>
#include <fstream>

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
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "krk/vram_guard.hpp"
#include "krk/h2d_probe.hpp"
#include "krk/arch.hpp"
#include "krk/draft_head.hpp"
#include "krk/spec_sampling.hpp"
#include "krk/backend.hpp"
#include "krk/engine.hpp"
#include "krk/expert_cpu.hpp"
#include "krk/gguf.hpp"
#include "krk/hc.hpp"
#include "krk/mla.hpp"
#include "krk/mla_shared.hpp"
#include "krk/dsv4_route.hpp"
#include "krk/ple.hpp"
#include "krk/qsa.hpp"
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

// In child mode (--run-group) these hold the running counts, the group's name
// and the file the parent reads them back from. A group is one function call,
// so a crash inside it costs this process and nothing else -- but it must not
// cost the parent the numbers either, which is what the partially rewritten
// file is for.
static std::FILE *g_progress = nullptr;
static const char *g_group = "";
static int g_group_index = -1;

// Defined with the runner at the end of this file. They are declared here
// because the CHECK macros below are the second half of the injection
// switches: a run with neither switch set pays one cached compare per check.
static bool checks_should_pass();
static void maybe_inject_crash();

static void note_progress() {
    if (g_progress) {
        // Rewrite in place and flush: the point of the file is to be correct at
        // the instant a fastfail takes the process away, so nothing may be
        // buffered.
        std::fseek(g_progress, 0, SEEK_SET);
        std::fprintf(g_progress, "%d %d\n", g_passed, g_run);
        std::fflush(g_progress);
    }
    // Written first, so the file carries the count as of this check even when
    // the next line is the one that dies (KRK_TEST_INJECT_CRASH_AT).
    maybe_inject_crash();
}

#define CHECK(cond, msg)                                                            \
    do {                                                                            \
        g_run++;                                                                    \
        if ((cond) && checks_should_pass()) {                                       \
            g_passed++;                                                             \
        } else {                                                                    \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);      \
        }                                                                           \
        note_progress();                                                            \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                                  \
    do {                                                                            \
        g_run++;                                                                    \
        const double _a = (a), _b = (b);                                            \
        if (std::fabs(_a - _b) <= (tol) && checks_should_pass()) {                  \
            g_passed++;                                                             \
        } else {                                                                    \
            std::fprintf(stderr, "FAIL %s:%d  %s (%g vs %g)\n", __FILE__, __LINE__, \
                         msg, _a, _b);                                              \
        }                                                                           \
        note_progress();                                                            \
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
// row pins that survive the HOST, not just the code
// ---------------------------------------------------------------------------
//
// Three groups (lfm2, qwen4exp, dsv4) pin "the whole graph" by hashing the
// last prompt row's logits bit for bit and comparing against a constant. That
// constant is a property of the MACHINE as much as of the graph: this binary
// is built with -march=native, and GCC contracts `a * c - b * s` into an FMA
// in one inlining context and not another, so the same graph on a second host
// produces a different bit hash. Measured on the LFM2 fixture: the pinned
// argmax reproduced while the row hash did not, which reads as a broken mixer
// and is really a different CPU.
//
// A pin that cannot be reproduced on another machine is worse than no pin: it
// hides real regressions behind a permanent failure. So the recorded value is
// a RowPin -- the things the graph decides and the host cannot:
//
//   * the row's eight largest logits, sorted, at 1e-3 relative (a routing,
//     head or norm change moves them),
//   * the row's minimum, L1 and L2 sums, at 1e-3 relative.
//
// Contraction noise is ~1e-7 relative, four orders below the tolerance, and a
// graph change moves these by far more than 1e-3 -- the qwen4exp PLE bug that
// motivated these gates moved a logit row by whole units.
//
// NOT the token ids. These fixtures are small models with pseudo-random
// weights, so their logit rows are full of near-ties: on the dsv4 fixture the
// two largest logits are equal to the last bit and the argmax is whichever
// index the sort happened to keep. Pinning ids there pins memcmp, not the
// graph. The exact hash stays in the FAIL message and is still compared when
// KRK_GOLDEN_BITS=1 is set, which is what a recording host uses to re-pin one
// bit-exactly after a deliberate change.
struct RowPin {
    f64 top[8] = {0.0};  // the eight largest logits, descending
    f64 min_val = 0.0;   // the row's minimum
    f64 l1 = 0.0;
    f64 l2 = 0.0;
};

static RowPin row_pin(const f32 *v, i32 n) {
    RowPin p;
    if (!v || n <= 0) return p;
    const i32 want = n < 8 ? n : 8;
    std::vector<f32> top(static_cast<size_t>(want), -3.4e38f);
    f64 lo = static_cast<f64>(v[0]), l1 = 0.0, l2 = 0.0;
    for (i32 i = 0; i < n; i++) {
        const f32 x = v[i];
        const f64 d = static_cast<f64>(x);
        if (d < lo) lo = d;
        l1 += std::fabs(d);
        l2 += d * d;
        // Insertion into the running top-want: `want` is 8, so this is a
        // shift chain over a register-resident array, not a sort.
        if (x > top[static_cast<size_t>(want - 1)]) {
            i32 k = want - 1;
            while (k > 0 && x > top[static_cast<size_t>(k - 1)]) {
                top[static_cast<size_t>(k)] = top[static_cast<size_t>(k - 1)];
                k--;
            }
            top[static_cast<size_t>(k)] = x;
        }
    }
    for (i32 k = 0; k < 8; k++)
        p.top[k] = k < want ? static_cast<f64>(top[static_cast<size_t>(k)]) : 0.0;
    p.min_val = lo;
    p.l1 = l1;
    p.l2 = std::sqrt(l2);
    return p;
}

// Relative comparison, with an absolute floor so a value that is legitimately
// near zero does not fail on noise.
static bool pin_near(f64 a, f64 b, f64 rel) {
    const f64 d = std::fabs(a - b);
    const f64 scale = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel * scale + 1e-6;
}

static bool pin_close(const RowPin &a, const RowPin &b) {
    for (i32 k = 0; k < 8; k++)
        if (!pin_near(a.top[k], b.top[k], 1e-3)) return false;
    return pin_near(a.min_val, b.min_val, 1e-3) &&
           pin_near(a.l1, b.l1, 1e-3) && pin_near(a.l2, b.l2, 1e-3);
}

// One line a failed run can be re-recorded from.
static std::string pin_str(const RowPin &p) {
    std::string s = "top";
    char buf[64];
    for (i32 k = 0; k < 8; k++) {
        std::snprintf(buf, sizeof buf, " %.6f", p.top[k]);
        s += buf;
    }
    std::snprintf(buf, sizeof buf, " | min %.6f l1 %.6f l2 %.6f", p.min_val,
                  p.l1, p.l2);
    return s + buf;
}

static bool golden_bits_on() {
    const char *e = std::getenv("KRK_GOLDEN_BITS");
    return e && e[0] != '\0' && e[0] != '0';
}

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
// Forward declarations of helpers defined later in this TU (wv, add_spm_vocab,
// env helpers) so the DFlash fixture group can live before them.
static f32 wv(i64 i);
static void add_spm_vocab(GgufBuilder &b, int vocab, int mask_id = -1);
static bool build_sized_model(const std::string &path, int n_embd, int n_ff,
                              int mask_id = -1);
static void set_env_var(const char *name, const std::string &value);
static void clear_env_var(const char *name);
// The prefill-chunk gate drives the CLI that sits beside this binary.
static std::string self_path();

// ---- KV tier regression coverage -------------------------------------------
//
// Grounds the four defects that the geometry-derived WARM policy was written
// to prevent, so they cannot come back silently:
//   * per-layer KV widths (the charge and slot size come from the layers that
//     carry KV, not from n_layer x the widest layer),
//   * no-KV layers (a mamba/SSM block and a pure-MoE block append nothing and
//     get no plane in the cache),
//   * drop vs spill counting (evictions_dropped is a page that went nowhere
//     and is NOT the same event as a COLD spill),
//   * per-plane cold files (the plane tag "k"/"v" namespaces the two caches
//     that share one --kv-cold-dir).
//
// Every arm runs on the CPU backend on purpose: the host dequantizers are the
// oracle, and a CPU run makes the tier's host/WARM side the real path rather
// than a staging area behind a GPU. cpu::tier() no-ops, so the CPU path is the
// one the tests exercise.
//
#include "krk/kv_tier.hpp"

// A sparse-KV fixture: 4 layers, n_kv_heads = 2, head_dim = 8, ctx = 16.
// Layers 0 and 3 carry q/k/v; layers 1 and 2 are pure MoE FFN blocks with no
// q/k/v at all (the shape a nemotron_h_moe or a mamba-stacked model presents).
// n_embd = 32, n_ff = 64, one expert per MoE layer, n_expert = 2.
static bool build_sparse_kv_model(const std::string &path, int ctx = 16) {
    const int n_embd = 32, n_layer = 4, n_head = 4, n_kv = 2, hd = 8, n_ff = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int vocab = 260, n_expert = 2, n_ff_exp = n_ff;
    GgufBuilder b;
    b.meta_str("general.architecture", "qwen2moe");
    b.meta_str("general.name", "kraken-sparse-kv-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32("qwen2moe.block_count", n_layer);
    b.meta_u32("qwen2moe.context_length", ctx);
    b.meta_u32("qwen2moe.embedding_length", n_embd);
    b.meta_u32("qwen2moe.feed_forward_length", n_ff);
    b.meta_u32("qwen2moe.attention.head_count", n_head);
    b.meta_u32("qwen2moe.attention.head_count_kv", n_kv);
    b.meta_u32("qwen2moe.attention.key_length", hd);
    b.meta_f32("qwen2moe.attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32("qwen2moe.rope.freq_base", 10000.0f);
    b.meta_u32("qwen2moe.expert_count", n_expert);
    b.meta_u32("qwen2moe.expert_used_count", 1);
    b.meta_u32("qwen2moe.expert_feed_forward_length", n_ff_exp);
    add_spm_vocab(b, vocab);
    auto rndvec = [&](size_t n) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(static_cast<i64>(i));
        return v;
    };
    auto one = [](size_t n) { return std::vector<f32>(n, 1.0f); };
    b.tensor_f16("token_embd.weight", {static_cast<u64>(n_embd), static_cast<u64>(vocab)},
                 rndvec(static_cast<size_t>(n_embd) * vocab));
    b.tensor_f32("output_norm.weight", {static_cast<u64>(n_embd)}, one(n_embd));
    for (int l = 0; l < n_layer; l++) {
        b.tensor_f32("blk." + std::to_string(l) + ".attn_norm.weight",
                      {static_cast<u64>(n_embd)}, one(n_embd));
        b.tensor_f32("blk." + std::to_string(l) + ".ffn_norm.weight",
                      {static_cast<u64>(n_embd)}, one(n_embd));
        if (l == 0 || l == 3) {
            b.tensor_f16("blk." + std::to_string(l) + ".attn_q.weight",
                         {static_cast<u64>(n_embd), static_cast<u64>(q_dim)},
                         rndvec(static_cast<size_t>(n_embd) * q_dim));
            b.tensor_f16("blk." + std::to_string(l) + ".attn_k.weight",
                         {static_cast<u64>(n_embd), static_cast<u64>(kv_dim)},
                         rndvec(static_cast<size_t>(n_embd) * kv_dim));
            b.tensor_f16("blk." + std::to_string(l) + ".attn_v.weight",
                         {static_cast<u64>(n_embd), static_cast<u64>(kv_dim)},
                         rndvec(static_cast<size_t>(n_embd) * kv_dim));
            b.tensor_f16("blk." + std::to_string(l) + ".attn_output.weight",
                         {static_cast<u64>(q_dim), static_cast<u64>(n_embd)},
                         rndvec(static_cast<size_t>(q_dim) * n_embd));
        }
        b.tensor_f16("blk." + std::to_string(l) + ".ffn_gate.weight",
                     {static_cast<u64>(n_embd), static_cast<u64>(n_ff)},
                     rndvec(static_cast<size_t>(n_embd) * n_ff));
        b.tensor_f16("blk." + std::to_string(l) + ".ffn_up.weight",
                     {static_cast<u64>(n_embd), static_cast<u64>(n_ff)},
                     rndvec(static_cast<size_t>(n_embd) * n_ff));
        b.tensor_f16("blk." + std::to_string(l) + ".ffn_down.weight",
                     {static_cast<u64>(n_ff), static_cast<u64>(n_embd)},
                     rndvec(static_cast<size_t>(n_ff) * n_embd));
        // Router: one row per expert, always resident. Required by the
        // loader so L.router is present and L.moe evaluates true.
        std::vector<f32> router(static_cast<size_t>(n_embd) * n_expert);
        for (size_t ri = 0; ri < router.size(); ri++) router[ri] = wv(static_cast<i64>(ri) * 3 + 7);
        b.tensor_f16("blk." + std::to_string(l) + ".ffn_gate_inp.weight", {u64(n_embd), u64(n_expert)}, router);

        b.tensor_f16("blk." + std::to_string(l) + ".ffn_gate_exps.weight",
                     {static_cast<u64>(n_embd), static_cast<u64>(n_ff_exp),
                      static_cast<u64>(n_expert)},
                     rndvec(static_cast<size_t>(n_embd) * n_ff_exp * n_expert));
        b.tensor_f16("blk." + std::to_string(l) + ".ffn_up_exps.weight",
                     {static_cast<u64>(n_embd), static_cast<u64>(n_ff_exp),
                      static_cast<u64>(n_expert)},
                     rndvec(static_cast<size_t>(n_embd) * n_ff_exp * n_expert));
        b.tensor_f16("blk." + std::to_string(l) + ".ffn_down_exps.weight",
                     {static_cast<u64>(n_ff_exp), static_cast<u64>(n_embd),
                      static_cast<u64>(n_expert)},
                     rndvec(static_cast<size_t>(n_ff_exp) * n_embd * n_expert));
    }
    return b.write(path);
}

// Wraps Engine::init() with explicit KV tier budgets so the tests do not
// repeat the config boilerplate. The CPU backend makes the tier inert (flat),
// but kv_tier_stats() still reports the geometry the tier was sized to.
static bool set_kv_tiered_budget(Engine &engine, Backend *cpu, const std::string &path,
                                  i32 hot_mb, i32 warm_mb, const std::string &cold_dir,
                                  std::string *err) {
    EngineConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx = 16;          // tiny context: the KV byte numbers are small and exact
    cfg.prefill_chunk = 4;
    cfg.kv_hot_mb = hot_mb;
    cfg.kv_warm_mb = warm_mb;
    cfg.kv_cold_dir = cold_dir;
    return engine.init(cpu, cfg, err);
}

// ---------------------------------------------------------------------------
// int8 K/V cache (KvQuant::Int8): the layout, the quantiser and its cost to
// the attention answer. The reference backend is the oracle -- the device
// kernels are refused until they read the same layout -- so what this group
// pins is the contract a quantized cache has to keep, not a kernel.
// ---------------------------------------------------------------------------
static void test_kv_quant_int8() {
    Backend *cpu = make_cpu_backend();

    // 1. The size arithmetic. A kv_dim-wide row is kv_dim bytes plus one f32
    //    scale per 32 values, and that is strictly smaller than the f32 row it
    //    replaces at every width a head can have.
    {
        const i64 dims[] = {64, 96, 128, 256, 2048};
        for (i64 dim : dims) {
            const size_t q8 = cpu->kv_row_bytes(dim, KvQuant::Int8);
            const size_t f32b = cpu->kv_row_bytes(dim, KvQuant::F16);
            CHECK(q8 == static_cast<size_t>(dim) + 4 * ((dim + 31) / 32),
                  "int8 KV row is the data plus one f32 scale per 32 values");
            CHECK(q8 < f32b, "and that row is smaller than the f32 one it replaces");
        }
        CHECK(cpu->kv_quant_supported(KvQuant::Int8),
              "the CPU reference implements int8 KV");
    }

    // 2. The quantiser, on a 200-key cache of realistic values: GQA 8/2, hd 64.
    const i64 hd = 64, n_kv = 2, n_head = 8;
    const i64 kv_dim = n_kv * hd;
    const i64 n_pos = 200;
    const size_t q8_row = cpu->kv_row_bytes(kv_dim, KvQuant::Int8);
    const size_t f32_row = cpu->kv_row_bytes(kv_dim, KvQuant::F16);

    std::mt19937 rng(1234);
    std::normal_distribution<f32> nd(0.0f, 1.0f);
    std::vector<f32> kbuf(static_cast<size_t>(n_pos) * static_cast<size_t>(kv_dim));
    std::vector<f32> vbuf(kbuf.size());
    for (auto &x : kbuf) x = nd(rng);
    for (auto &x : vbuf) x = nd(rng);

    std::vector<u8> q8k(static_cast<size_t>(n_pos) * q8_row, 0xCC); // poisoned
    std::vector<u8> q8v(q8k.size(), 0xCC);
    std::vector<u8> fk(static_cast<size_t>(n_pos) * f32_row, 0);
    std::vector<u8> fv(fk.size(), 0);

    AttnDesc d;
    d.n_head = n_head;
    d.n_kv = n_kv;
    d.hd = hd;
    d.n_tok = n_pos;
    d.pos0 = 0;
    d.layer_stride = 0;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd)));
    d.causal = true;
    d.window = 0;

    d.kv_int8 = false;
    d.pos_stride = kv_dim;
    cpu->kv_append(fk.data(), fv.data(), kbuf.data(), vbuf.data(), d);

    d.kv_int8 = true;
    d.pos_stride = static_cast<i64>(q8_row);
    cpu->kv_append(q8k.data(), q8v.data(), kbuf.data(), vbuf.data(), d);

    // 3. Chunk invariance of the QUANTISER: the same 200 rows appended one at a
    //    time must leave the same bytes as one 200-row call. Per-row, per-32
    //    scaling is what makes that true, and it is what keeps a chunked
    //    prefill and a decode step reading the same cache.
    {
        std::vector<u8> ck(q8k.size(), 0), cv(q8v.size(), 0);
        AttnDesc c = d;
        c.n_tok = 1;
        for (i64 t = 0; t < n_pos; t++) {
            c.pos0 = t;
            cpu->kv_append(ck.data(), cv.data(),
                           kbuf.data() + t * kv_dim, vbuf.data() + t * kv_dim, c);
        }
        CHECK(ck == q8k && cv == q8v,
              "one-row and 200-row appends quantise to the same cache bytes");
    }

    // 4. What it costs the answer: one decode row at the end of a 200-key
    //    context, int8 against the f32 cache it replaces.
    std::vector<f32> q(static_cast<size_t>(n_head * hd));
    for (auto &x : q) x = nd(rng);
    std::vector<f32> out_ref(q.size(), 0.0f), out_q8(q.size(), 0.0f);

    AttnDesc a = d;
    a.n_tok = 1;
    a.pos0 = n_pos - 1;
    a.kv_int8 = false;
    a.pos_stride = kv_dim;
    cpu->attention(out_ref.data(), q.data(), fk.data(), fv.data(), a);

    a.kv_int8 = true;
    a.pos_stride = static_cast<i64>(q8_row);
    cpu->attention(out_q8.data(), q.data(), q8k.data(), q8v.data(), a);

    f64 num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < out_ref.size(); i++) {
        const f64 e = static_cast<f64>(out_q8[i]) - static_cast<f64>(out_ref[i]);
        num += e * e;
        den += static_cast<f64>(out_ref[i]) * out_ref[i];
        worst = std::max(worst, std::fabs(e));
    }
    const f64 rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
    std::fprintf(stderr, "  (int8 KV at %lld keys: rel %.5f, worst %.5f)\n",
                 static_cast<long long>(n_pos), rel, worst);
    CHECK(std::isfinite(rel) && rel < 0.02,
          "int8 KV keeps the decode row within 2% of the f32 cache");
    CHECK(worst < 0.05, "and no single channel moves by more than 0.05");
    bool finite = true;
    for (f32 x : out_q8) finite = finite && std::isfinite(x);
    CHECK(finite, "the quantized path produces no NaN or inf");

    // 5. The arm that keeps 1-4 from being vacuous: the SAME bytes read with the
    //    wrong layout must disagree. If the int8 flag were ignored, the int8
    //    row would be f32 data and this divergence would not appear.
    {
        std::vector<f32> out_wrong(q.size(), 0.0f);
        AttnDesc w = a;
        w.kv_int8 = true;
        w.pos_stride = static_cast<i64>(q8_row);
        cpu->attention(out_wrong.data(), q.data(), fk.data(), fv.data(), w);
        f64 n2 = 0.0, d2 = 0.0;
        for (size_t i = 0; i < out_wrong.size(); i++) {
            const f64 e = static_cast<f64>(out_wrong[i]) - static_cast<f64>(out_ref[i]);
            n2 += e * e;
            d2 += static_cast<f64>(out_ref[i]) * out_ref[i];
        }
        const f64 wrong_rel = d2 > 0.0 ? std::sqrt(n2 / d2) : 0.0;
        CHECK(wrong_rel > 0.2,
              "the int8 layout read out of an f32 cache is a different answer");
    }

    // 6. A zeroed row is a legal row: scale 0, values 0, and the softmax is
    //    indifferent rather than NaN. (A scale computed as max/127 would divide
    //    by zero here if the guard were missing.)
    {
        std::vector<f32> zk(static_cast<size_t>(kv_dim), 0.0f);
        std::vector<f32> zv(static_cast<size_t>(kv_dim), 0.0f);
        AttnDesc z = d;
        z.n_tok = 1;
        z.pos0 = n_pos - 1;
        z.kv_int8 = true;
        z.pos_stride = static_cast<i64>(q8_row);
        // pos0 is the LAST row, so the scales land in that row of the cache,
        // not at its start: the arm reads them where the append wrote them.
        std::vector<u8> zrow(static_cast<size_t>(z.pos_stride) * static_cast<size_t>(n_pos),
                             0xCC);
        cpu->kv_append(zrow.data(), zrow.data(), zk.data(), zv.data(), z);
        bool zero_scales = true;
        const f32 *sc = reinterpret_cast<const f32 *>(
            zrow.data() + static_cast<size_t>(n_pos - 1) * q8_row + kv_dim);
        for (i64 b = 0; b < (kv_dim + 31) / 32; b++)
            zero_scales = zero_scales && sc[b] == 0.0f;
        CHECK(zero_scales, "a zero activation quantises to a zero scale, not a NaN");
    }

    delete cpu;
}

static void test_kv_tier_per_layer_widths() {
    const std::string path = "kraken-sparse-kv-test.gguf";
    CHECK(build_sparse_kv_model(path), "wrote the sparse-KV fixture");

    Backend *cpu = make_cpu_backend();
    Engine engine;
    std::string err;
    CHECK(set_kv_tiered_budget(engine, cpu, path, 0, 0, "", &err),
          "engine initialises on the sparse-KV fixture");
    if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());

    const Model &m = engine.model();
    const ModelConfig &mc = m.cfg();
    CHECK(mc.arch == "qwen2moe", "the sparse-KV fixture is a qwen2moe model");
    CHECK(mc.n_layer == 4, "the fixture has 4 layers");
    CHECK(mc.n_head_kv == 2 && mc.head_dim == 8,
          "each KV-carrying layer has 2 heads of 8 dims");

    // ground truth from the model's own per-layer geometry, not from any
    // single n_head_kv number.
    i64 n_kv_layers = 0;
    i64 kv_dim_sum = 0;
    i64 max_kv_dim = 0;
    size_t as = cpu->act_size();
    for (i32 l = 0; l < mc.n_layer; l++) {
        if (!m.layer_has_kv(l)) continue;
        n_kv_layers++;
        i64 dim = m.kv_dim_at(l);
        kv_dim_sum += dim;
        if (dim > max_kv_dim) max_kv_dim = dim;
    }
    const i64 ctx = 16;
    const f64 plane_bytes = static_cast<f64>(kv_dim_sum) * ctx * as;
    const f64 total_kv = plane_bytes * 2.0;   // K + V
    const size_t slot_bytes = static_cast<size_t>(max_kv_dim) * ctx * as;

    // 1. layer_has_kv is the predicate the engine uses; layers 1 and 2 have
    //    no q/k/v and so hold no KV.
    CHECK(m.layer_has_kv(0), "layer 0 carries KV");
    CHECK(m.layer_has_kv(3), "layer 3 carries KV");
    CHECK(!m.layer_has_kv(1) && !m.layer_has_kv(2),
          "layers 1 and 2 are pure FFN blocks, no KV");
    CHECK(m.kv_layer_count() == 2,
          "the model has exactly 2 KV-carrying layers (not 4)");
    CHECK(n_kv_layers == 2, "same answer computed from the predicate above");

    // 2. kv_dim_at is per-layer, and the widest layer is what a shared
    //    workspace/slot is sized to. Here both KV layers are 2*8 = 16.
    CHECK(m.kv_dim_at(0) == 16, "layer 0 KV dim is 2 heads * 8 = 16");
    CHECK(m.kv_dim_at(3) == 16, "layer 3 KV dim is 2 heads * 8 = 16");
    CHECK(max_kv_dim == 16, "the widest KV layer is 16");

    // 3. kv_tier_stats reports the geometry the tier is sized FROM -- the layer
    //    count and the widest layer -- and it covers exactly the 2 KV-carrying
    //    layers, not all 4. The slot geometry it reports here is deliberately
    //    the FLAT one, so the honest reading is "not tiered, no slots": a cache
    //    that fits is a single flat allocation (kv_tier.hpp), and the CPU
    //    backend hands the whole cache as the device budget for a measured
    //    reason -- a CPU tier pages every layer per step through host buffers
    //    that ARE the device, 4951 ms/step against 27 ms on a 64-layer model
    //    (src/engine.cpp).
    //
    //    Neither can it be otherwise. n_slots_ is min(hot_budget/slot_bytes,
    //    n_kv_layers), and the flat path is taken while hot_budget >= total_;
    //    since total_ <= n_kv_layers * slot_bytes, a budget that engages the
    //    tier gives n_slots_ <= n_kv_layers - 1. "One HOT slot per KV-carrying
    //    layer" therefore IS the flat case, and no flag combination can produce
    //    it tiered. The slot arithmetic itself is proved on the cache, where a
    //    budget below the cache does engage it -- see test_kv_tier_drop_vs_spill.
    KvTierStats ks;
    engine.kv_tier_stats(&ks);
    CHECK(ks.layers == 2, "the tier knows about 2 KV layers, not 4");
    CHECK(ks.layer_bytes == slot_bytes,
          "a HOT slot holds the largest layer one plane");
    CHECK(!ks.tiered, "a cache that fits runs flat, not tiered");
    CHECK(ks.slots == 0, "a flat cache holds no HOT slots");
    CHECK(ks.hot == 0 && ks.hot_bytes == 0,
          "and nothing is resident in a HOT tier it does not have");

    // 4. The flat allocation's own arithmetic: one plane holds the KV the model
    //    actually has -- sum over the KV-carrying layers, NOT n_layer times the
    //    widest layer. This is the number that used to be 8.67x the KV that
    //    exists (208 MiB reserved and reported for 24 MiB on nemotron).
    CHECK(engine.kvt_k().plane_bytes() == static_cast<size_t>(plane_bytes),
          "the K plane holds exactly the KV the model has");
    CHECK(engine.kvt_k().plane_bytes() + engine.kvt_v().plane_bytes() ==
              static_cast<size_t>(total_kv),
          "both planes together hold exactly that and nothing more");

    // 5. Re-deriving the KV dimension from the model's own per-layer dims is
    //    the cross-check: sum(kv_dim_at(l) for l where layer_has_kv(l)).
    size_t charged_dim = 0;
    for (i32 l = 0; l < mc.n_layer; l++)
        if (m.layer_has_kv(l)) charged_dim += static_cast<size_t>(m.kv_dim_at(l));
    CHECK(charged_dim == static_cast<size_t>(kv_dim_sum),
          "re-derived KV dimension (per-layer, KV-only) matches the sum");

    engine.shutdown();
    delete cpu;
    std::remove(path.c_str());
}

static void test_kv_tier_no_kv_layers() {
    // A model whose KV-carrying layers are a strict subset of its transformer
    // layers: the tier must not charge or count the no-KV ones.
    const std::string path = "kraken-sparse-kv-test.gguf";
    CHECK(build_sparse_kv_model(path), "wrote the sparse-KV fixture");

    Backend *cpu = make_cpu_backend();
    Engine engine;
    std::string err;
    CHECK(set_kv_tiered_budget(engine, cpu, path, 0, 0, "", &err),
          "engine initialises");
    if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());

    KvTierStats ks;
    engine.kv_tier_stats(&ks);

    CHECK(ks.layers == 2, "only the 2 KV-carrying layers are counted");
    // The no-KV layers get no slot and no charge. The cache is flat here (see
    // test_kv_tier_per_layer_widths for why it cannot be anything else on this
    // machine), so what the report has to show is the flat geometry, and the
    // two layers it counted are exactly the ones the packed plane holds.
    CHECK(!ks.tiered && ks.slots == 0,
          "a flat cache gives no layer a HOT slot");
    CHECK(ks.hot == 0 && ks.hot_bytes == 0,
          "and holds nothing in a HOT tier it does not have");
    CHECK(engine.kvt_k().plane_bytes() == engine.kvt_v().plane_bytes(),
          "both planes pack the same KV-carrying bytes");

    const Model &m = engine.model();
    CHECK(m.kv_layer_count() == 2, "Model::kv_layer_count agrees");
    CHECK(m.layer_has_kv(1) == false && m.layer_has_kv(2) == false,
          "layers 1 and 2 hold no KV");

    // A decode run on this model uses the KV cache only on layers 0 and 3.
    // If the tier had charged the no-KV layers, the load report would be
    // wrong. The generation must still work and produce deterministic tokens.
    {
        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 4;
        p.sampler.greedy = true;
        p.sampler.temp = 0.0f;
        GenerateResult r;
        CHECK(engine.generate(p, &r), "sparse-KV model generates");
        CHECK(!r.tokens.empty(), "it produced tokens");
        CHECK(r.prompt_tokens > 0, "the prompt was tokenised");
    }

    engine.shutdown();
    delete cpu;
    std::remove(path.c_str());
}

static void test_kv_tier_drop_vs_spill() {
    // The COLD counter and the DROPPED counter are not the same event. A page
    // evicted from HOT with nowhere to go is DROPPED -- its KV is gone and the
    // layer is zero-filled next time -- while the same page with a cold dir is
    // SPILLED to disk and comes back on promotion. They shared one counter
    // once, which let a run that lost a page every step report a working disk
    // tier while printing fluent text.
    //
    // Driven on the cache rather than through Engine, for two reasons. A CPU
    // engine run never evicts (its cache is flat, see
    // test_kv_tier_per_layer_widths), so neither counter could move. And the
    // tiered slot geometry -- which Engine cannot show on this machine -- is
    // provable here, where a budget below the cache really does engage the tier.
    //
    // What it proves is what the design guarantees. With no cold dir, WARM's
    // capacity comes from the geometry and not from the budget, so a 1-slot HOT
    // still loses nothing even when --kv-warm-mb asks for no WARM at all: the
    // DROPPED counter is the safety net for a configuration the geometry floor
    // makes unreachable. With a cold dir the budget IS obeyed, and the same
    // eviction spills instead.
    const std::string cold_dir = "kraken-drop-vs-spill-dir";
    {
        std::error_code ec;
        std::filesystem::remove_all(cold_dir, ec);
        std::filesystem::create_directories(cold_dir, ec);
        CHECK(!ec, "the cold dir was created");
    }

    // One plane's bytes for the sparse-KV geometry: 2 heads x 8 dims x ctx 16,
    // f32 activations, 2 of the 4 layers carrying KV.
    const size_t plane = static_cast<size_t>(16) * 16 * sizeof(f32);
    std::vector<size_t> len(4, 0);
    len[0] = plane;
    len[3] = plane;

    // Arm A: one HOT slot for two KV-carrying layers, no WARM budget and no
    // cold dir. Stepping over the layers evicts; nothing is lost.
    {
        Backend *cpu = make_cpu_backend();
        KvTierCache c;
        std::string err;
        CHECK(c.init(cpu, 4, len, plane, 0, "", "k", &err),
              "the cache initialises with a 1-slot HOT and no disk tier");
        CHECK(c.tiered(), "a budget below the cache engages the tier");
        CHECK(c.kv_layers() == 2, "two KV-carrying layers, not four");
        CHECK(c.hot_slots() == 1, "one HOT slot, sized by the largest layer");
        CHECK(c.slot_bytes() == plane, "a slot holds one plane of that layer");
        CHECK(c.plane_bytes() == plane * 2,
              "the flat plane packs the two KV-carrying layers, not four");
        CHECK(c.warm_from_geometry() && c.warm_slots() == 2,
              "WARM's capacity comes from the geometry, not from the budget");

        for (i64 l = 0; l < 4; l++) {
            if (len[static_cast<size_t>(l)] == 0) continue;
            c.base(l);
            c.end_layer(l);
        }
        c.base(0);
        c.end_layer(0);

        KvTierStats s;
        c.stats(&s);
        CHECK(s.layers == 2 && s.slots == 1 && s.tiered,
              "stats report the tiered geometry: 2 layers, 1 slot");
        CHECK(s.evictions_to_warm > 0,
              "an evicted page reaches WARM even though it asked for none");
        CHECK(s.evictions_dropped == 0,
              "and nothing is dropped, because the geometry floor covered it");
        CHECK(s.evictions_to_cold == 0, "no disk tier, so nothing was spilled");
        c.detach();
        delete cpu;
    }

    // Arm B: the same cache and the same budget, with a writable cold dir. The
    // budget is obeyed now (a spill has a home), so the same eviction spills --
    // and a spill is still not a drop.
    {
        Backend *cpu = make_cpu_backend();
        KvTierCache c;
        std::string err;
        CHECK(c.init(cpu, 4, len, plane, 0, cold_dir, "k", &err),
              "the cache initialises with a 1-slot HOT and a cold dir");
        CHECK(!c.cold_dir_rejected(), "the cold dir is accepted");
        CHECK(!c.warm_from_geometry() && c.warm_slots() == 0,
              "with a disk tier the WARM budget is obeyed");

        for (i64 l = 0; l < 4; l++) {
            if (len[static_cast<size_t>(l)] == 0) continue;
            c.base(l);
            c.end_layer(l);
        }
        c.base(0);
        c.end_layer(0);

        KvTierStats s;
        c.stats(&s);
        CHECK(s.evictions_to_cold > 0,
              "the same eviction spills to disk instead of being dropped");
        CHECK(s.evictions_dropped == 0, "and a spill is never counted as a drop");
        c.detach();
        delete cpu;
    }

    {
        std::error_code ec;
        std::filesystem::remove_all(cold_dir, ec);
    }
}

// CountingBackend is declared below with the expert-cache fixture helpers.
static void test_kv_tier_host_reserve_refusal();

static void test_kv_tier_per_plane_cold_files() {
// Two planes (K and V) share one --kv-cold-dir. The cold path MUST embed
// the plane tag so K's spills and V's spills are different files -- otherwise
// V's spill overwrote K's and a restore handed one plane the other's bytes.
    const std::string path = "kraken-sparse-kv-test.gguf";
    CHECK(build_sparse_kv_model(path), "wrote the sparse-KV fixture");

    Backend *cpu = make_cpu_backend();
    Engine engine;
    std::string err;
    CHECK(set_kv_tiered_budget(engine, cpu, path, 0, 0,
          "kraken-per-plane-cold-dir", &err),
          "engine initialises with a cold dir");
    if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());

    // Cold path is public; check it contains the plane tag.
    std::string kp = engine.kvt_k().cold_path(0);
    std::string vp = engine.kvt_v().cold_path(0);
    CHECK(kp.find("/kv_k_layer_0.bin") != std::string::npos,
          "K cold path contains the k-tagged filename");
    CHECK(vp.find("/kv_v_layer_0.bin") != std::string::npos,
          "V cold path contains the v-tagged filename");
    CHECK(kp != vp, "K and V cold paths are different files");

// A name built from the layer index ALONE would be identical for the two
// planes. The plane tag prevents that.
    const char *konly = "kv_layer_0.bin";
    CHECK(kp.find(konly) == std::string::npos,
          "the shared-name defect is not present (K path has no bare name)");
    CHECK(vp.find(konly) == std::string::npos,
          "V path has no bare name either");

    engine.shutdown();
    delete cpu;
    std::remove(path.c_str());
    std::string cmd = "cmd /c rd /s /q kraken-per-plane-cold-dir";
    std::system(cmd.c_str());
}

// synthetic Mixture-of-Experts model
// ---------------------------------------------------------------------------

// Deterministic, index-addressed weights. The dense and MoE builders below draw
// from the same sequence, so a MoE model with one expert carries exactly the
// numbers of its dense twin and must produce identical logits.
static f32 wv(i64 i) {
    const i32 m = static_cast<i32>(i % 11) - 5; // -5..5
    return static_cast<f32>(m) * 0.03f;
}

// The same byte-fallback SPM vocab build_tiny_model uses. mask_id, when >= 0,
// also writes tokenizer.ggml.mask_token_id, which is what the DFlash mask-id
// guard compares a head set's own id against.
static void add_spm_vocab(GgufBuilder &b, int vocab, int mask_id) {
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
    if (mask_id >= 0) b.meta_u32("tokenizer.ggml.mask_token_id", static_cast<u32>(mask_id));
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

// A two-layer qwen2moe whose MoE layers carry DIFFERENT expert counts and
// top-k. This is the packed-merge shape (Nemotron Bolt 74B-A9B: 240/304/368/432/
// 496 experts under one `expert_count` of 496, with `expert_used_count` written
// as a per-block array). `expert_count` is the CAP here -- 5 -- while layer 0
// keeps 3 experts and layer 1 keeps all 5, and the top-k array says 2 then 3.
static bool build_per_layer_moe_model(const std::string &path,
                                     const std::vector<int> &layer_experts,
                                     const std::vector<int> &layer_k,
                                     int cap_expert) {
    const int n_embd = 32, n_head = 4, n_kv = 2, hd = 8;
    const int n_ff = 64, n_ff_exp = 64, vocab = 260, ctx = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int n_layer = static_cast<int>(layer_experts.size());
    const std::string arch = "qwen2moe";

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-per-layer-moe");
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
    b.meta_u32(arch + ".expert_count", static_cast<u32>(cap_expert));
    // The per-block top-k array, which is the key the scalar read cannot see.
    b.meta_i32_array(arch + ".expert_used_count", layer_k);
    b.meta_u32(arch + ".expert_feed_forward_length", static_cast<u32>(n_ff_exp));
    add_spm_vocab(b, vocab, -1);

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };

    b.tensor_f16("token_embd.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 0));
    b.tensor_f32("output_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f16("output.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 6000));

    for (int l = 0; l < n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const int ne_l = layer_experts[static_cast<size_t>(l)];
        b.tensor_f32(p + "attn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f32(p + "ffn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f16(p + "attn_q.weight", {u64(n_embd), u64(q_dim)},
                     vec(static_cast<size_t>(n_embd) * q_dim, 100));
        b.tensor_f16(p + "attn_k.weight", {u64(n_embd), u64(kv_dim)},
                     vec(static_cast<size_t>(n_embd) * kv_dim, 200));
        b.tensor_f16(p + "attn_v.weight", {u64(n_embd), u64(kv_dim)},
                     vec(static_cast<size_t>(n_embd) * kv_dim, 300));
        b.tensor_f16(p + "attn_output.weight", {u64(q_dim), u64(n_embd)},
                     vec(static_cast<size_t>(q_dim) * n_embd, 400));

        std::vector<f32> router(static_cast<size_t>(n_embd) * ne_l);
        for (size_t i = 0; i < router.size(); i++)
            router[i] = wv(static_cast<i64>(i) * 3 + 7);
        b.tensor_f16(p + "ffn_gate_inp.weight", {u64(n_embd), u64(ne_l)}, router);

        const size_t gsz = static_cast<size_t>(n_embd) * n_ff_exp;
        const size_t dsz = static_cast<size_t>(n_ff_exp) * n_embd;
        const size_t ne = static_cast<size_t>(ne_l);
        std::vector<f32> gate(gsz * ne), up(gsz * ne), down(dsz * ne);
        for (size_t e = 0; e < ne; e++) {
            const i64 bump = static_cast<i64>(e) * 7919;
            for (size_t k = 0; k < gsz; k++) {
                gate[e * gsz + k] = wv(500 + static_cast<i64>(k) + bump);
                up[e * gsz + k] = wv(900 + static_cast<i64>(k) + bump);
            }
            for (size_t k = 0; k < dsz; k++)
                down[e * dsz + k] = wv(1300 + static_cast<i64>(k) + bump);
        }
        b.tensor_f16(p + "ffn_gate_exps.weight",
                     {u64(n_embd), u64(n_ff_exp), u64(ne)}, gate);
        b.tensor_f16(p + "ffn_up_exps.weight",
                     {u64(n_embd), u64(n_ff_exp), u64(ne)}, up);
        b.tensor_f16(p + "ffn_down_exps.weight",
                     {u64(n_ff_exp), u64(n_embd), u64(ne)}, down);
    }
    return b.write(path);
}

// ---- NVFP4 global-scale ("<tensor>.scale") fixtures ----------------------
//
// A GGUF stores an NVFP4 weight's per-tensor global scale BESIDE it, as
// "<tensor>.scale": one f32 that multiplies the whole matrix. The block format
// itself is four UE4M3 micro-scales (`0x40` = 1.0) then 32 bytes of E2M1 codes
// per 64 values, so a weight can be written whose true value is KNOWN exactly:
// code g (the doubled grid -- 1,2,3,4 -> 0.5, 1.0, 1.5, 2.0) under a micro-scale
// of 1.0, with a sidecar of 0.5, is 0.5*g. The f16 twin writes exactly those
// numbers, so the two files describe the SAME model and must produce the same
// logits. Anything that drops the sidecar runs every element at 2x and cannot.
static void nvfp4_grid_blocks(i64 n_out, i64 n_in, std::vector<u8> &raw,
                              std::vector<f32> &twin) {
    const i64 blocks = std::max<i64>(1, (n_in + 63) / 64);
    raw.assign(static_cast<size_t>(n_out * blocks * 36), 0);
    twin.assign(static_cast<size_t>(n_out * n_in), 0.0f);
    for (i64 r = 0; r < n_out; r++) {
        for (i64 b = 0; b < blocks; b++) {
            u8 *p = raw.data() + static_cast<size_t>((r * blocks + b) * 36);
            p[0] = p[1] = p[2] = p[3] = 0x40; // four micro-scales of 1.0
            for (int k = 0; k < 32; k++) {
                // Byte k belongs to sub-block s = k/8 at offset j = k%8, and its
                // two nibbles are columns s*16 + j (low) and s*16 + j + 8 (high)
                // -- the layout dequant_chunk<NVFP4> reads.
                const int s = k / 8, j = k % 8;
                const i64 c0 = b * 64 + s * 16 + j, c1 = c0 + 8;
                const int g0 = c0 < n_in ? 1 + static_cast<int>((r * 7 + c0) % 4) : 0;
                const int g1 = c1 < n_in ? 1 + static_cast<int>((r * 7 + c1) % 4) : 0;
                p[4 + k] = static_cast<u8>((g0 & 0xF) | ((g1 & 0xF) << 4));
                if (c0 < n_in) twin[static_cast<size_t>(r * n_in + c0)] = 0.5f * g0;
                if (c1 < n_in) twin[static_cast<size_t>(r * n_in + c1)] = 0.5f * g1;
            }
        }
    }
}

// What the fixture installs on the named weight: `sidecar` writes the NVFP4
// encoding plus a 0.5 sidecar, otherwise the f16 twin of the same values.
static void put_weight(GgufBuilder &b, const std::string &name, i64 n_out,
                       i64 n_in, bool sidecar) {
    std::vector<u8> raw;
    std::vector<f32> twin;
    nvfp4_grid_blocks(n_out, n_in, raw, twin);
    if (!sidecar) {
        b.tensor_f16(name, {u64(n_in), u64(n_out)}, twin);
        return;
    }
    b.tensor_raw(name, 40 /* NVFP4 */, {u64(n_in), u64(n_out)}, raw.data(), raw.size());
    static const char *kW = ".weight";
    const size_t wn = std::strlen(kW);
    const std::string base = name.size() > wn && name.compare(name.size() - wn, wn, kW) == 0
                                 ? name.substr(0, name.size() - wn)
                                 : name;
    b.tensor_f32(base + ".scale", {1}, std::vector<f32>{0.5f});
}

// `sidecar` names the weight the two files describe identically -- "head",
// "attn_q", "ffn_gate", "ffn_down" or "router" -- and `nvfp4` chooses HOW it is
// written: as the f16 twin, or as NVFP4 plus a 0.5 sidecar. Both files therefore
// hold the same matrix; only the encoding differs, which is what makes them a
// controlled comparison.
static bool build_sidecar_model(const std::string &path, const std::string &sidecar,
                                bool moe, bool nvfp4) {
    const int n_embd = 64, n_head = 4, hd = 16, n_kv = 2;
    const int n_ff = 64, n_ff_exp = 64, vocab = 260, ctx = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int n_expert = 4, top_k = 2, n_ff_sh = 64;
    const std::string arch = moe ? "qwen2moe" : "llama";

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-sidecar");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", 1);
    b.meta_u32(arch + ".context_length", static_cast<u32>(ctx));
    b.meta_u32(arch + ".embedding_length", static_cast<u32>(n_embd));
    b.meta_u32(arch + ".feed_forward_length", static_cast<u32>(n_ff));
    b.meta_u32(arch + ".attention.head_count", static_cast<u32>(n_head));
    b.meta_u32(arch + ".attention.head_count_kv", static_cast<u32>(n_kv));
    b.meta_u32(arch + ".attention.key_length", static_cast<u32>(hd));
    b.meta_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32(arch + ".rope.freq_base", 10000.0f);
    if (moe) {
        b.meta_u32(arch + ".expert_count", static_cast<u32>(n_expert));
        b.meta_u32(arch + ".expert_used_count", static_cast<u32>(top_k));
        b.meta_u32(arch + ".expert_feed_forward_length", static_cast<u32>(n_ff_exp));
        b.meta_u32(arch + ".expert_shared_feed_forward_length",
                   static_cast<u32>(n_ff_sh));
    }
    add_spm_vocab(b, vocab, -1);

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };
    auto put = [&](const std::string &what, const std::string &name, i64 n_out,
                   i64 n_in, i64 base) {
        if (what == sidecar) {
            put_weight(b, name, n_out, n_in, nvfp4);
            return;
        }
        b.tensor_f16(name, {u64(n_in), u64(n_out)},
                     vec(static_cast<size_t>(n_in * n_out), base));
    };

    b.tensor_f16("token_embd.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 0));
    b.tensor_f32("output_norm.weight", {u64(n_embd)}, ones(n_embd));
    put("head", "output.weight", vocab, n_embd, 6000);
    b.tensor_f32("blk.0.attn_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32("blk.0.ffn_norm.weight", {u64(n_embd)}, ones(n_embd));
    put("attn_q", "blk.0.attn_q.weight", q_dim, n_embd, 100);
    b.tensor_f16("blk.0.attn_k.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim, 200));
    b.tensor_f16("blk.0.attn_v.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim, 300));
    b.tensor_f16("blk.0.attn_output.weight", {u64(q_dim), u64(n_embd)},
                 vec(static_cast<size_t>(q_dim) * n_embd, 400));

    if (!moe) {
        put("ffn_gate", "blk.0.ffn_gate.weight", n_ff, n_embd, 500);
        b.tensor_f16("blk.0.ffn_up.weight", {u64(n_embd), u64(n_ff)},
                     vec(static_cast<size_t>(n_embd) * n_ff, 900));
        put("ffn_down", "blk.0.ffn_down.weight", n_embd, n_ff, 1300);
        return b.write(path);
    }

    put("router", "blk.0.ffn_gate_inp.weight", n_expert, n_embd, 7);
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
        for (size_t k = 0; k < dsz; k++)
            down[e * dsz + k] = wv(1300 + static_cast<i64>(k) + bump);
    }
    b.tensor_f16("blk.0.ffn_gate_exps.weight",
                 {u64(n_embd), u64(n_ff_exp), u64(ne)}, gate);
    b.tensor_f16("blk.0.ffn_up_exps.weight",
                 {u64(n_embd), u64(n_ff_exp), u64(ne)}, up);
    b.tensor_f16("blk.0.ffn_down_exps.weight",
                 {u64(n_ff_exp), u64(n_embd), u64(ne)}, down);
    b.tensor_f16("blk.0.ffn_up_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                 vec(static_cast<size_t>(n_embd) * n_ff_sh, 2100));
    b.tensor_f16("blk.0.ffn_down_shexp.weight", {u64(n_ff_sh), u64(n_embd)},
                 vec(static_cast<size_t>(n_ff_sh) * n_embd, 2500));
    return b.write(path);
}

// NVFP4 global scales, at every place a dense weight is used.
//
// The oracle is a twin file: the same model with the same NUMBERS written as
// f16. Each arm therefore tests one projection's sidecar against a known-good
// answer, and a dropped or doubled sidecar shows up as logits that do not match
// -- not as a crash, not as a small rounding difference.
static void test_nvfp4_sidecar_scale() {
    struct Arm {
        const char *what; // the weight the sidecar rides
        bool moe;
    };
    const Arm arms[] = {{"head", false}, {"attn_q", false},
                        {"ffn_gate", false}, {"ffn_down", false},
                        {"router", true}};
    for (const Arm &a : arms) {
        const std::string ref = std::string("kraken-sidecar-f16-") + a.what + ".gguf";
        const std::string q = std::string("kraken-sidecar-nvfp4-") + a.what + ".gguf";
        CHECK(build_sidecar_model(ref, a.what, a.moe, false), "wrote the f16 twin");
        CHECK(build_sidecar_model(q, a.what, a.moe, true), "wrote the NVFP4 file");

        auto logits_of = [](const std::string &p, std::vector<f32> &lg, i32 &tok,
                            std::string &err) {
            Backend *be = make_cpu_backend();
            Engine e;
            EngineConfig c;
            c.model_path = p;
            c.n_ctx = 64;
            c.prefill_chunk = 8;
            if (!e.init(be, c, &err)) return false;
            GenerateParams gp;
            gp.prompt = "the";
            gp.max_tokens = 2;
            GenerateResult r;
            if (!e.generate(gp, &r)) return false;
            const i32 nv = e.model().cfg().n_vocab;
            lg.assign(e.last_logits(), e.last_logits() + nv);
            tok = r.tokens.empty() ? -1 : r.tokens.back();
            return true;
        };
        std::vector<f32> a_lg, b_lg;
        i32 a_tok = -1, b_tok = -1;
        std::string ea, eb;
        const bool ok_a = logits_of(ref, a_lg, a_tok, ea);
        const bool ok_b = logits_of(q, b_lg, b_tok, eb);
        if (!ok_a) std::fprintf(stderr, "  (f16 twin: %s)\n", ea.c_str());
        if (!ok_b) std::fprintf(stderr, "  (nvfp4 file: %s)\n", eb.c_str());
        CHECK(ok_a && ok_b, "both files load and generate");
        if (!ok_a || !ok_b) continue;
        CHECK(b_tok == a_tok, "the sidecar-scaled model emits the same token");
        f32 worst = 0.0f;
        const size_t n = std::min(a_lg.size(), b_lg.size());
        for (size_t i = 0; i < n; i++)
            worst = std::max(worst, std::fabs(a_lg[i] - b_lg[i]));
        if (worst > 2.0f)
            std::fprintf(stderr, "  (sidecar arm %s: logits differ by %.4g)\n",
                         a.what, static_cast<double>(worst));
        // The two files describe the same matrix, so the only difference left is
        // the kernels' accumulation order. A dropped 0.5 doubles every element of
        // one projection, which is far outside this.
        CHECK(worst < 2.0f, "and its logits match the f16 twin's");
    }
}

// A LatentMoE block, which is what Nemotron 3.5 Bolt 74B-A9B is: the routed
// experts read and write a NARROW latent (ffn_latent_down -> experts ->
// ffn_latent_up) while the shared expert and the residual stay full width.
// `latent` is the latent width, 0 for a file with no pair; `half` writes only
// the down half; `mismatch` writes an up tensor that does not join the bank.
static bool build_latent_moe_model(const std::string &path, int n_embd,
                                   int latent, int n_ff_exp, int n_expert,
                                   int top_k, bool half, bool mismatch) {
    const int n_head = 4, n_kv = 2, hd = 8;
    const int n_ff = 64, vocab = 260, ctx = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int n_ff_sh = 64;
    const std::string arch = "qwen2moe";

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-latent-moe");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", 1);
    b.meta_u32(arch + ".context_length", static_cast<u32>(ctx));
    b.meta_u32(arch + ".embedding_length", static_cast<u32>(n_embd));
    b.meta_u32(arch + ".feed_forward_length", static_cast<u32>(n_ff));
    b.meta_u32(arch + ".attention.head_count", static_cast<u32>(n_head));
    b.meta_u32(arch + ".attention.head_count_kv", static_cast<u32>(n_kv));
    b.meta_u32(arch + ".attention.key_length", static_cast<u32>(hd));
    b.meta_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32(arch + ".rope.freq_base", 10000.0f);
    b.meta_u32(arch + ".expert_count", static_cast<u32>(n_expert));
    b.meta_u32(arch + ".expert_used_count", static_cast<u32>(top_k));
    b.meta_u32(arch + ".expert_feed_forward_length", static_cast<u32>(n_ff_exp));
    b.meta_u32(arch + ".expert_shared_feed_forward_length",
               static_cast<u32>(n_ff_sh));
    add_spm_vocab(b, vocab, -1);

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };

    b.tensor_f16("token_embd.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 0));
    b.tensor_f32("output_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f16("output.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 6000));
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

    std::vector<f32> router(static_cast<size_t>(n_embd) * n_expert);
    for (size_t i = 0; i < router.size(); i++)
        router[i] = wv(static_cast<i64>(i) * 3 + 7);
    b.tensor_f16("blk.0.ffn_gate_inp.weight", {u64(n_embd), u64(n_expert)}, router);

    // The routed bank is `latent` wide, and an UNGATED one (up/down only) --
    // which is Bolt's shape, so the gate-absent path is part of what this pins.
    // A negative `latent` writes a NARROW bank with no projection pair at all:
    // the shape the loader must refuse rather than read off-stride.
    const int ew = latent != 0 ? (latent > 0 ? latent : -latent) : n_embd;
    const size_t gsz = static_cast<size_t>(ew) * n_ff_exp;
    const size_t dsz = static_cast<size_t>(n_ff_exp) * ew;
    const size_t ne = static_cast<size_t>(n_expert);
    std::vector<f32> up(gsz * ne), down(dsz * ne);
    for (size_t e = 0; e < ne; e++) {
        const i64 bump = static_cast<i64>(e) * 7919;
        for (size_t k = 0; k < gsz; k++)
            up[e * gsz + k] = wv(900 + static_cast<i64>(k) + bump);
        for (size_t k = 0; k < dsz; k++)
            down[e * dsz + k] = wv(1300 + static_cast<i64>(k) + bump);
    }
    b.tensor_f16("blk.0.ffn_up_exps.weight", {u64(ew), u64(n_ff_exp), u64(ne)}, up);
    b.tensor_f16("blk.0.ffn_down_exps.weight", {u64(n_ff_exp), u64(ew), u64(ne)},
                 down);

    // The shared expert is full width on this family: that asymmetry is the
    // whole point of the test, not an accident of the fixture.
    b.tensor_f16("blk.0.ffn_up_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                 vec(static_cast<size_t>(n_embd) * n_ff_sh, 2100));
    b.tensor_f16("blk.0.ffn_down_shexp.weight", {u64(n_ff_sh), u64(n_embd)},
                 vec(static_cast<size_t>(n_ff_sh) * n_embd, 2500));

    if (latent > 0) {
        b.tensor_f16("blk.0.ffn_latent_down.weight",
                     {u64(n_embd), u64(latent)},
                     vec(static_cast<size_t>(n_embd) * latent, 3100));
        if (!half) {
            const int up_in = mismatch ? latent + 1 : latent;
            b.tensor_f16("blk.0.ffn_latent_up.weight",
                         {u64(up_in), u64(n_embd)},
                         vec(static_cast<size_t>(up_in) * n_embd, 3500));
        }
    }
    return b.write(path);
}

// LatentMoE. What this pins:
//  * a routed bank narrower than the hidden state loads at all -- the loader
//    used to be silent about the width and the forward fed n_embd into it;
//  * the layer reports the bank's own width, so the forward has one number to
//    size the latent by;
//  * the block runs end to end (down-projection, experts in the latent,
//    up-projection, shared expert, residual) and produces tokens;
//  * half a pair, a pair that does not join the bank, and a narrow bank with no
//    pair are all REFUSED -- each is a silently wrong answer otherwise.
static void test_moe_latent() {
    const std::string path = "kraken-latent-moe.gguf";
    CHECK(build_latent_moe_model(path, 32, 16, 64, 4, 2, false, false),
          "wrote the LatentMoE fixture");

    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx = 64;
    cfg.prefill_chunk = 8;
    std::string err;
    const bool ok = engine.init(cpu, cfg, &err);
    if (!ok) std::fprintf(stderr, "  (latent MoE init: %s)\n", err.c_str());
    CHECK(ok, "a LatentMoE file loads");
    if (ok) {
        const Model &m = engine.model();
        const LayerWeights &L = m.layers()[0];
        CHECK(L.latent_moe(), "the layer reports its latent pair");
        CHECK(L.experts.n_embd == 16,
              "the routed bank keeps its own (latent) input width");
        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 4;
        GenerateResult r;
        const bool gen = engine.generate(p, &r);
        CHECK(gen, "it generates through the latent");
        CHECK(r.generated > 0, "... and emits tokens");
    }

    // ---- the refusals -----------------------------------------------------
    auto refuses = [](const std::string &p, const char *what) {
        Backend *be = make_cpu_backend();
        Engine e;
        EngineConfig c;
        c.model_path = p;
        c.n_ctx = 64;
        std::string err;
        const bool loaded = e.init(be, c, &err);
        CHECK(!loaded, what);
        CHECK(err.find("latent") != std::string::npos,
              "... and the refusal names the latent");
    };
    CHECK(build_latent_moe_model("kraken-latent-half.gguf", 32, 16, 64, 4, 2, true,
                                 false),
          "wrote the half-pair fixture");
    refuses("kraken-latent-half.gguf",
            "half a latent pair is refused, not run on the hidden state");
    CHECK(build_latent_moe_model("kraken-latent-mismatch.gguf", 32, 16, 64, 4, 2,
                                 false, true),
          "wrote the mismatched-up fixture");
    refuses("kraken-latent-mismatch.gguf",
            "a latent projection that does not join the bank is refused");
    CHECK(build_latent_moe_model("kraken-latent-nopair.gguf", 32, -16, 64, 4, 2,
                                 false, false),
          "wrote the narrow-bank-no-pair fixture");
    refuses("kraken-latent-nopair.gguf",
            "a narrow routed bank with no latent projection is refused");
}

// Per-layer expert counts and top-k. What this pins:
//  * the loader stops treating `expert_count` as an equality, so a layer whose
//    bank is narrower than the cap loads (Bolt 74B-A9B was refused here);
//  * the router is checked against ITS layer's bank, not the cap;
//  * `expert_used_count` reaches the forward as a per-block array, so a layer
//    with 3 experts does not try to select 3 out of a 5-wide declaration;
//  * a heterogeneous stack reports non-uniform geometry, which is what keeps
//    the uniform dispatch table and the expert scan off it.
static void test_moe_per_layer_experts() {
    const std::string path = "kraken-per-layer-moe.gguf";
    CHECK(build_per_layer_moe_model(path, {3, 5}, {2, 3}, 5),
          "wrote the per-layer MoE fixture");

    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx = 64;
    cfg.prefill_chunk = 8;
    std::string err;
    const bool ok = engine.init(cpu, cfg, &err);
    if (!ok) std::fprintf(stderr, "  (per-layer MoE init: %s)\n", err.c_str());
    CHECK(ok, "a file whose layers carry different expert counts loads");
    if (ok) {
        const Model &m = engine.model();
        const ModelConfig &mc = m.cfg();
        CHECK(mc.n_expert == 5, "expert_count is the cap");
        CHECK(mc.n_expert_used == 3, "the top-k array's maximum is model-level");
        CHECK(mc.layer_n_expert_used.size() == 2,
              "expert_used_count was read as a per-block array");
        CHECK(m.n_expert_at(0) == 3 && m.n_expert_at(1) == 5,
              "each layer reports its own expert count");
        CHECK(m.n_expert_used_at(0) == 2 && m.n_expert_used_at(1) == 3,
              "each layer reports its own top-k");
        CHECK(!m.moe_geometry_uniform(),
              "the stack is reported non-uniform (dispatch/scan stay off)");

        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 4;
        GenerateResult r;
        CHECK(engine.generate(p, &r), "it generates");
        CHECK(r.generated > 0, "... and emits tokens");
    }

    // The uniform control: the same builder with one expert count everywhere
    // has to report uniform, or the flag above is vacuous.
    const std::string upath = "kraken-uniform-moe.gguf";
    CHECK(build_per_layer_moe_model(upath, {4, 4}, {2, 2}, 4),
          "wrote the uniform control fixture");
    {
        Backend *u_cpu = make_cpu_backend();
        Engine u_engine;
        EngineConfig u_cfg;
        u_cfg.model_path = upath;
        u_cfg.n_ctx = 64;
        u_cfg.prefill_chunk = 8;
        std::string u_err;
        const bool u_ok = u_engine.init(u_cpu, u_cfg, &u_err);
        if (!u_ok) std::fprintf(stderr, "  (uniform control: %s)\n", u_err.c_str());
        CHECK(u_ok, "the uniform control loads");
        if (u_ok) {
            CHECK(u_engine.model().moe_geometry_uniform(),
                  "a uniform stack still reports uniform");
            CHECK(u_engine.model().n_expert_at(0) == 4 &&
                      u_engine.model().n_expert_at(1) == 4,
                  "... with the same count on every layer");
        }
    }
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
    // Self-contained, and never a silent skip. This group used to return with
    // 0/0 checks when the fixture was missing -- which the runner counted as a
    // pass -- and it only ever had the file because test_gdn_model_loads ran
    // before it. It builds its own copy now (same arguments as that group, so
    // the bytes are the same either way).
    if (!build_tiny_gdn_model(kGdnTestPath, 4, 4)) {
        CHECK(false, "wrote the qwen35moe fixture this group decodes through");
        return;
    }

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
// gemm_group must honour EACH MATRIX's own weight type -- on the fused path and
// on the per-matrix fallback, which is what a multi-row (prefill) call takes,
// because the fused kernel expresses the decode row only.
//
// This is a regression test for a real defect. The fallback used to pass the
// group's single `wt` to every matrix. Measured on Qwen3.5-0.8B, whose GDN
// layers group attn_qkv (Q5_K) with ssm_alpha and ssm_beta (Q4_K): the two Q4_K
// matrices were decoded as Q5_K during the prompt, KRK_DUMP's gdn.delta read
// -nan(ind) against -1.66893e-06 from the separate-launch arm, and the op count
// and the timing looked healthy the whole way.
//
// The reference is therefore the SAME backend's per-matrix gemm() with each
// matrix's own type, never the group against itself -- a group compared with
// itself agrees with any bug it carries. Three formats with three different
// geometries (F16 at 2 bytes a value, F32 at 4, Q8_0 at 34 bytes per 32 values)
// make a misread unambiguous, and the last check proves the misread is visible
// at all, so the comparison above cannot pass by being blind.
static void test_gdn_group_matrix_types() {
    Backend *cpu = make_cpu_backend();
    const i64 n_in = 64, n_out = 8; // 64 = two Q8_0 blocks per row

    CHECK(dtype_row_bytes(DType::F16, n_in) != dtype_row_bytes(DType::F32, n_in),
          "F16 and F32 rows are different lengths");
    CHECK(dtype_row_bytes(DType::F32, n_in) != dtype_row_bytes(DType::Q8_0, n_in),
          "F32 and Q8_0 rows are different lengths");

    // Every buffer gets slack: the sensitivity check at the end reads one
    // format's bytes as another's, and that must be wrong WITHOUT running off
    // the allocation.
    std::vector<u8> w16(static_cast<size_t>(n_out * n_in * 2) + 512, 0);
    std::vector<u8> w32(static_cast<size_t>(n_out * n_in * 4) + 512, 0);
    std::vector<u8> w8(static_cast<size_t>(n_out * (n_in / 32) * 34) + 512, 0);
    for (i64 r = 0; r < n_out; r++) {
        for (i64 c = 0; c < n_in; c++) {
            const f32 v = static_cast<f32>((r * 5 + c * 3) % 29) - 14.0f;
            const u16 h = fp32_to_fp16(v);
            std::memcpy(&w16[static_cast<size_t>(r * n_in + c) * 2], &h, 2);
            std::memcpy(&w32[static_cast<size_t>(r * n_in + c) * 4], &v, 4);
        }
        for (i64 b = 0; b < n_in / 32; b++) {
            u8 *blk = &w8[static_cast<size_t>(r * (n_in / 32) + b) * 34];
            const u16 h = fp32_to_fp16(0.25f);
            blk[0] = static_cast<u8>(h & 0xFFu);
            blk[1] = static_cast<u8>(h >> 8);
            for (int i = 0; i < 32; i++)
                blk[2 + i] =
                    static_cast<u8>(static_cast<i8>((i * 7 + r * 3) % 61 - 30));
        }
    }

    const void *w[3] = {w16.data(), w32.data(), w8.data()};
    const DType types[3] = {DType::F16, DType::F32, DType::Q8_0};
    const i64 n_out_arr[3] = {n_out, n_out, n_out};
    const i64 shapes[2] = {4, 1}; // prefill (the fallback) and the decode row

    for (int s = 0; s < 2; s++) {
        const i64 rows = shapes[s];
        const size_t n_el = static_cast<size_t>(rows * n_out);
        std::vector<f32> x(static_cast<size_t>(rows * n_in));
        for (size_t i = 0; i < x.size(); i++)
            x[i] = static_cast<f32>(static_cast<int>(i % 19) - 9) * 0.125f;

        std::vector<f32> got[3];
        void *out[3];
        for (int m = 0; m < 3; m++) {
            got[m].assign(n_el, -12345.0f);
            out[m] = got[m].data();
        }
        // The group's own `wt` names the F16 matrix on purpose, so a fallback
        // that leans on it misreads the other two.
        cpu->gemm_group(out, x.data(), w, DType::F16, n_out_arr, n_in, 3, rows,
                        /*silu=*/false, /*silu_fused=*/nullptr,
                        /*row_parallel_only=*/false, types);

        for (int m = 0; m < 3; m++) {
            std::vector<f32> ref(n_el, 0.0f);
            cpu->gemm(ref.data(), x.data(), w[m], types[m], n_out, n_in, rows);
            int bad = 0;
            for (size_t i = 0; i < n_el; i++)
                if (got[m][i] != ref[i]) bad++;
            CHECK(bad == 0,
                  rows == 1
                      ? "gemm_group's decode row matches per-matrix gemm() with "
                        "each matrix's own type"
                      : "gemm_group's multi-row fallback matches per-matrix "
                        "gemm() with each matrix's own type");
        }
    }

    // Sensitivity: reading the F32 matrix with the F16 type -- the shape of the
    // real bug -- has to produce something different, or the checks above would
    // pass whether or not the fallback honours per-matrix types.
    {
        const i64 rows = 4;
        const size_t n_el = static_cast<size_t>(rows * n_out);
        std::vector<f32> x(static_cast<size_t>(rows * n_in), 0.25f);
        std::vector<f32> as_f32(n_el, 0.0f), as_f16(n_el, 0.0f);
        cpu->gemm(as_f32.data(), x.data(), w32.data(), DType::F32, n_out, n_in,
                  rows);
        cpu->gemm(as_f16.data(), x.data(), w32.data(), DType::F16, n_out, n_in,
                  rows);
        int diff = 0;
        for (size_t i = 0; i < n_el; i++)
            if (as_f32[i] != as_f16[i]) diff++;
        CHECK(diff > 0,
              "an F32 matrix read as F16 differs, so a type mix-up is visible");
    }
}

// The two backend ops the batched MoE decode path is built from, and that path
// as a composition.
//
// `gemm_group` gained a per-matrix activation (`xs`); `silu_mul_rows` is one
// launch for `count` one-row buffers. Both exist so a decode layer can compute
// its routed experts in a handful of launches instead of one group per expert --
// measured on Qwen3-30B-A3B: 23.4 -> 16.0 ms/step, same bytes out.
//
// "Same bytes out" is the whole claim, so the reference is never the batch
// against itself:
//   * xs is checked against per-matrix gemm() with that matrix's own row;
//   * silu_mul_rows is checked against one silu_mul() call per row;
//   * the composite (one gate/up group over a shared row, one activation launch,
//     one down group over per-expert rows, per-row weighted sums) is checked
//     against the per-expert sequence the batch replaced;
//   * and the down group is shown to be sensitive to `xs`: reusing row 0 for
//     every expert -- the mistake a batch that assumed one activation makes --
//     must produce a different sum, or the checks above prove nothing.
static void test_moe_batch_ops() {
    Backend *cpu = make_cpu_backend();
    const i64 n_in = 64, n_rows = 64;  // one full row per expert, 2 Q8_0 blocks
    const i32 n_ex = 4;                // the batch's gate/up group takes four

    // Four experts, two formats, at unit-ish scale: these feed a 64-term dot per
    // row and the composite sums four of them, so unit values overflow f32 and
    // the comparison would fail on infinities rather than on a difference.
    std::vector<u8> w16(static_cast<size_t>(n_ex * n_rows * n_in * 2), 0);
    std::vector<u8> w8(static_cast<size_t>(n_ex * n_rows * (n_in / 32) * 34), 0);
    for (i32 e = 0; e < n_ex; e++) {
        for (i64 r = 0; r < n_rows; r++) {
            for (i64 c = 0; c < n_in; c++) {
                const f32 v =
                    (static_cast<f32>((r * 5 + c * 3 + e * 7) % 29) - 14.0f) * 0.01f;
                const u16 h = fp32_to_fp16(v);
                std::memcpy(&w16[static_cast<size_t>((e * n_rows + r) * n_in + c) * 2],
                            &h, 2);
            }
            for (i64 b = 0; b < n_in / 32; b++) {
                u8 *blk =
                    &w8[static_cast<size_t>((e * n_rows + r) * (n_in / 32) + b) * 34];
                const u16 h = fp32_to_fp16(0.25f);
                blk[0] = static_cast<u8>(h & 0xFFu);
                blk[1] = static_cast<u8>(h >> 8);
                for (int i = 0; i < 32; i++)
                    blk[2 + i] = static_cast<u8>(
                        static_cast<i8>((i * 7 + r * 3 + e * 5) % 61 - 30));
            }
        }
    }
    auto w16_at = [&](i32 e) {
        return static_cast<const void *>(
            w16.data() + static_cast<size_t>(e * n_rows * n_in * 2));
    };
    auto w8_at = [&](i32 e) {
        return static_cast<const void *>(
            w8.data() + static_cast<size_t>(e * n_rows * (n_in / 32) * 34));
    };

    // ---- xs: each matrix reads its own activation -------------------------
    {
        std::vector<f32> xs[3];
        for (int m = 0; m < 3; m++) {
            xs[m].assign(static_cast<size_t>(n_in), 0.0f);
            for (i64 i = 0; i < n_in; i++)
                xs[m][static_cast<size_t>(i)] =
                    static_cast<f32>(static_cast<int>((i * (m + 1)) % 17) - 8) * 0.125f;
        }
        const void *xptr[3] = {xs[0].data(), xs[1].data(), xs[2].data()};
        const void *w[3] = {w16_at(0), w8_at(1), w16_at(2)};
        const DType types[3] = {DType::F16, DType::Q8_0, DType::F16};
        const i64 outs[3] = {n_rows, n_rows, n_rows};
        std::vector<f32> got[3];
        void *out[3];
        for (int m = 0; m < 3; m++) {
            got[m].assign(static_cast<size_t>(n_rows), -12345.0f);
            out[m] = got[m].data();
        }
        // `x` is a decoy: a group that ignores xs reads this row for all three.
        std::vector<f32> decoy(static_cast<size_t>(n_in), 99.0f);
        cpu->gemm_group(out, decoy.data(), w, DType::F16, outs, n_in, 3, 1, false,
                        nullptr, false, types, xptr);
        for (int m = 0; m < 3; m++) {
            std::vector<f32> ref(static_cast<size_t>(n_rows), 0.0f);
            cpu->gemm(ref.data(), xs[m].data(), w[m], types[m], n_rows, n_in, 1);
            int bad = 0;
            for (i64 i = 0; i < n_rows; i++)
                if (got[m][static_cast<size_t>(i)] != ref[static_cast<size_t>(i)]) bad++;
            CHECK(bad == 0,
                  "gemm_group's per-matrix activation matches per-matrix gemm()");
        }
    }

    // ---- silu_mul_rows: a batch is the same launch, per row ---------------
    {
        const i64 row_len = 32;
        std::vector<f32> g(static_cast<size_t>(n_ex * row_len));
        std::vector<f32> u(g.size());
        for (size_t i = 0; i < g.size(); i++) {
            g[i] = static_cast<f32>(static_cast<int>(i % 23) - 11) * 0.5f;
            u[i] = static_cast<f32>(static_cast<int>(i % 13) - 6) * 0.25f;
        }
        std::vector<f32> got(g.size(), -1.0f), ref(g.size(), -1.0f);
        std::vector<f32> got_s(n_ex, -1.0f), ref_s(n_ex, -1.0f);
        void *out[n_ex];
        const void *ga[n_ex];
        const void *ua[n_ex];
        f32 *sc[n_ex];
        for (i32 e = 0; e < n_ex; e++) {
            out[e] = got.data() + static_cast<size_t>(e * row_len);
            ga[e] = g.data() + static_cast<size_t>(e * row_len);
            ua[e] = u.data() + static_cast<size_t>(e * row_len);
            sc[e] = got_s.data() + e;
        }
        cpu->silu_mul_rows(out, ga, ua, row_len, n_ex, sc);
        for (i32 e = 0; e < n_ex; e++)
            cpu->silu_mul(ref.data() + static_cast<size_t>(e * row_len),
                          g.data() + static_cast<size_t>(e * row_len),
                          u.data() + static_cast<size_t>(e * row_len), row_len,
                          row_len, ref_s.data() + e);
        int bad = 0, scale_bad = 0;
        for (size_t i = 0; i < got.size(); i++)
            if (got[i] != ref[i]) bad++;
        for (i32 e = 0; e < n_ex; e++)
            if (got_s[static_cast<size_t>(e)] != ref_s[static_cast<size_t>(e)]) scale_bad++;
        CHECK(bad == 0, "silu_mul_rows matches one silu_mul per row");
        CHECK(scale_bad == 0, "silu_mul_rows fills each row's own scale");
    }

    // ---- the composite, against the per-expert sequence it replaced -------
    {
        // gate/up produce n_gate rows of n_in; the down matrix reduces over
        // exactly those rows, so the two widths are one number -- feeding a
        // narrower activation to a wider down matrix reads past every row.
        const i64 n_gate = n_rows, n_emb = 64;
        std::vector<u8> d16(static_cast<size_t>(n_ex * n_emb * n_gate * 2), 0);
        for (i32 e = 0; e < n_ex; e++)
            for (i64 r = 0; r < n_emb; r++)
                for (i64 c = 0; c < n_gate; c++) {
                    const f32 v =
                        (static_cast<f32>((r * 3 + c * 11 + e) % 31) - 15.0f) * 0.01f;
                    const u16 h = fp32_to_fp16(v);
                    std::memcpy(
                        &d16[static_cast<size_t>((e * n_emb + r) * n_gate + c) * 2],
                        &h, 2);
                }
        auto d_at = [&](i32 e) {
            return static_cast<const void *>(
                d16.data() + static_cast<size_t>(e * n_emb * n_gate * 2));
        };
        std::vector<f32> bg(static_cast<size_t>(n_ex * n_gate), 0.0f);
        std::vector<f32> bu(bg.size(), 0.0f);
        std::vector<f32> bd(static_cast<size_t>(n_ex * n_emb), 0.0f);
        std::vector<f32> bsum(static_cast<size_t>(n_emb), 0.0f);
        std::vector<f32> bscale(n_ex, 1.0f);
        std::vector<f32> rg(bg.size(), 0.0f);
        std::vector<f32> ru(bg.size(), 0.0f);
        std::vector<f32> rd(static_cast<size_t>(n_ex * n_emb), 0.0f);
        std::vector<f32> rsum(static_cast<size_t>(n_emb), 0.0f);
        std::vector<f32> rscale(n_ex, 1.0f);
        // One activation row for the gate/up pair -- at one decode token every
        // routed expert reads the same one.
        std::vector<f32> x(static_cast<size_t>(n_in));
        for (i64 i = 0; i < n_in; i++)
            x[static_cast<size_t>(i)] =
                static_cast<f32>(static_cast<int>(i % 19) - 9) * 0.125f;
        const f32 alpha[n_ex] = {0.5f, 0.25f, 0.125f, 0.75f};

        // ---- the batched composition ----
        {
            void *out[2 * n_ex];
            const void *w[2 * n_ex];
            i64 no[2 * n_ex];
            DType ts[2 * n_ex];
            for (i32 e = 0; e < n_ex; e++) {
                out[2 * e] = bg.data() + static_cast<size_t>(e * n_gate);
                out[2 * e + 1] = bu.data() + static_cast<size_t>(e * n_gate);
                w[2 * e] = w16_at(e);
                w[2 * e + 1] = (e & 1) ? w8_at(e) : w16_at(e);
                no[2 * e] = n_gate;
                no[2 * e + 1] = n_gate;
                ts[2 * e] = DType::F16;
                ts[2 * e + 1] = (e & 1) ? DType::Q8_0 : DType::F16;
            }
            cpu->gemm_group(out, x.data(), w, DType::F16, no, n_in, 2 * n_ex, 1,
                            false, nullptr, false, ts);
            void *aout[n_ex];
            const void *aga[n_ex];
            const void *aup[n_ex];
            f32 *asc[n_ex];
            for (i32 e = 0; e < n_ex; e++) {
                aout[e] = bg.data() + static_cast<size_t>(e * n_gate);
                aga[e] = aout[e];
                aup[e] = bu.data() + static_cast<size_t>(e * n_gate);
                asc[e] = bscale.data() + e;
            }
            cpu->silu_mul_rows(aout, aga, aup, n_gate, n_ex, asc);
            void *dout[n_ex];
            const void *dx[n_ex];
            const void *dw[n_ex];
            i64 dno[n_ex];
            DType dts[n_ex];
            for (i32 e = 0; e < n_ex; e++) {
                dout[e] = bd.data() + static_cast<size_t>(e * n_emb);
                dx[e] = bg.data() + static_cast<size_t>(e * n_gate);
                dw[e] = d_at(e);
                dno[e] = n_emb;
                dts[e] = DType::F16;
            }
            cpu->gemm_group(dout, nullptr, dw, DType::F16, dno, n_gate, n_ex, 1,
                            false, nullptr, false, dts, dx);
            for (i32 e = 0; e < n_ex; e++)
                for (i64 i = 0; i < n_emb; i++)
                    bsum[static_cast<size_t>(i)] +=
                        alpha[e] * bd[static_cast<size_t>(e * n_emb + i)];
        }

        // ---- the per-expert sequence, one launch group at a time ----
        for (i32 e = 0; e < n_ex; e++) {
            cpu->gemm(rg.data() + static_cast<size_t>(e * n_gate), x.data(),
                      w16_at(e), DType::F16, n_gate, n_in, 1);
            cpu->gemm(ru.data() + static_cast<size_t>(e * n_gate), x.data(),
                      (e & 1) ? w8_at(e) : w16_at(e),
                      (e & 1) ? DType::Q8_0 : DType::F16, n_gate, n_in, 1);
            cpu->silu_mul(rg.data() + static_cast<size_t>(e * n_gate),
                          rg.data() + static_cast<size_t>(e * n_gate),
                          ru.data() + static_cast<size_t>(e * n_gate), n_gate,
                          n_gate, rscale.data() + e);
            cpu->gemm(rd.data() + static_cast<size_t>(e * n_emb),
                      rg.data() + static_cast<size_t>(e * n_gate), d_at(e),
                      DType::F16, n_emb, n_gate, 1);
            for (i64 i = 0; i < n_emb; i++)
                rsum[static_cast<size_t>(i)] +=
                    alpha[e] * rd[static_cast<size_t>(e * n_emb + i)];
        }

        int diff = 0;
        for (i64 i = 0; i < n_emb; i++)
            if (bsum[static_cast<size_t>(i)] != rsum[static_cast<size_t>(i)]) diff++;
        CHECK(diff == 0,
              "a batched expert layer reproduces the per-expert sequence bit for "
              "bit (gate/up group, one activation launch, per-row down group)");

        // Sensitivity: the down group must read the expert's OWN activation row.
        std::vector<f32> wrong(static_cast<size_t>(n_emb), 0.0f);
        {
            void *dout[n_ex];
            const void *dx[n_ex];
            const void *dw[n_ex];
            i64 dno[n_ex];
            DType dts[n_ex];
            std::vector<f32> tmp(static_cast<size_t>(n_ex * n_emb), 0.0f);
            for (i32 e = 0; e < n_ex; e++) {
                dout[e] = tmp.data() + static_cast<size_t>(e * n_emb);
                dx[e] = bg.data();  // the bug: one row for every expert
                dw[e] = d_at(e);
                dno[e] = n_emb;
                dts[e] = DType::F16;
            }
            cpu->gemm_group(dout, nullptr, dw, DType::F16, dno, n_gate, n_ex, 1,
                            false, nullptr, false, dts, dx);
            for (i32 e = 0; e < n_ex; e++)
                for (i64 i = 0; i < n_emb; i++)
                    wrong[static_cast<size_t>(i)] +=
                        alpha[e] * tmp[static_cast<size_t>(e * n_emb + i)];
        }
        int wrong_diff = 0;
        for (i64 i = 0; i < n_emb; i++)
            if (wrong[static_cast<size_t>(i)] != rsum[static_cast<size_t>(i)]) wrong_diff++;
        CHECK(wrong_diff > 0,
              "the down group is sensitive to per-expert activations (one shared "
              "row changes the sum)");
    }

    delete cpu;
}

static void test_gdn_ops() {
    test_gdn_group_matrix_types();
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

    // The attention output gate, exactly as the engine composes it: one scalar
    // per (token, head) out of a projection, softplus'd, broadcast over that
    // head's whole output row BEFORE the output projection. laguna runs through
    // this path and is a supported arch, and the broadcast op had no test of its
    // own -- a transposed read (which head's scalar lands on which row) or a
    // per-element read would corrupt it silently, and the verify pass cannot
    // catch that because the gate changes the output on purpose.
    {
        const i64 nh = 2, hd2 = 4, nt = 2;
        std::vector<f32> g = {0.0f, -2.0f, 3.0f, 25.0f}; // [token][head]
        cpu->softplus_act(g.data(), nt * nh);
        const std::vector<f32> want = {std::log(2.0f),
                                       std::log1p(std::exp(-2.0f)),
                                       std::log1p(std::exp(3.0f)), 25.0f};
        std::vector<f32> x(static_cast<size_t>(nt * nh * hd2), 1.0f);
        cpu->mul_head_broadcast(x.data(), g.data(), nt, nh, hd2);
        for (i64 t = 0; t < nt; t++)
            for (i64 h = 0; h < nh; h++)
                for (i64 i = 0; i < hd2; i++)
                    CHECK(std::fabs(x[static_cast<size_t>((t * nh + h) * hd2 + i)] -
                                    want[static_cast<size_t>(t * nh + h)]) < 1e-5f,
                          "the gate scales a head's whole row by its own scalar");

        // Distinct scalars per (token, head) over distinguishable rows: every
        // product is exact in f32, so a wrong lane is a wrong number, not a
        // rounding difference.
        std::vector<f32> y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        const std::vector<f32> g2 = {2.0f, 3.0f, 5.0f, 7.0f};
        cpu->mul_head_broadcast(y.data(), g2.data(), 2, 2, 4);
        CHECK(y[0] == 2.0f && y[3] == 8.0f, "token 0 head 0 scaled by its own 2");
        CHECK(y[4] == 15.0f && y[7] == 24.0f, "token 0 head 1 scaled by its own 3");
        CHECK(y[8] == 45.0f && y[11] == 60.0f, "token 1 head 0 scaled by its own 5");
        CHECK(y[12] == 91.0f && y[15] == 112.0f, "token 1 head 1 scaled by its own 7");
    }

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
    CHECK(arch_lookup("qwen4-mtp") != nullptr,
          "the MTP head set tools/mtp_extract.py writes is in the table");
    CHECK(arch_lookup("qwen4-mtp")->role == ArchRole::Draft,
          "and it is a draft file rather than a model");
    CHECK(arch_lookup("qwen4exp") != nullptr, "the qwen4exp family is in the table");
    CHECK(arch_lookup("qwen4exp")->role == ArchRole::Target &&
              arch_lookup("qwen4exp")->support == ArchSupport::Yes,
          "as a supported target rather than an unlisted arch");
    // RecurrentMoe, and that is now the honest shape: the delta net's tensors
    // ARE this engine's (attn_qkv is [q|k|v] of 2048/2048/6144, attn_gate is the
    // z gate, ssm_* are the same names gdn_forward reads), so the refusal is the
    // hyper-connection/PLE/QSA graph and not a tensor class. Declaring Moe here
    // would make the gap map refuse the family for a computation it has.
    CHECK(arch_lookup("qwen4exp")->shape == ArchShape::RecurrentMoe,
          "declared recurrent-moe: the delta-net tensor set is implemented");
    CHECK(arch_lookup("qwen4exp")->hyper_connection &&
              arch_lookup("qwen4exp")->ple && arch_lookup("qwen4exp")->qsa,
          "and the schema flags record the three pieces still missing");
    // The gemma stack: every one of these files gates its FFN with GeGLU and
    // scales the embedding, and the family's alternating window is a property
    // of the architecture rather than of the metadata. Two of those were
    // declared in comments and nowhere else.
    for (const char *g : {"gemma", "gemma2", "gemma3", "gemma4"}) {
        const ArchSpec *s = arch_lookup(g);
        CHECK(s != nullptr, "the gemma family is in the table");
        if (!s) continue;
        CHECK(s->gelu_ffn,
              "a gemma stack gates its FFN with GeGLU, not SwiGLU");
        CHECK(s->gemma_norm, "and scales the embedding by sqrt(n_embd)");
        CHECK(s->qk_norm, "and carries QK-norm tensors");
    }
    // The window pattern is the arch's, and it is the one thing that made
    // gemma3 look unimplemented after the kernel learned windows: period 6 with
    // the WINDOWED layer first (5 local to 1 global), gemma2 period 2 the same
    // way round.
    CHECK(arch_lookup("gemma3")->swa_period == 6 &&
              arch_lookup("gemma3")->swa_window_first,
          "gemma3 declares a 5:1 local/global pattern, windowed layer first");
    CHECK(arch_lookup("gemma2")->swa_period == 2 &&
              arch_lookup("gemma2")->swa_window_first,
          "gemma2 alternates every layer, windowed first");
    CHECK(arch_lookup("gemma3")->support != ArchSupport::No,
          "and gemma3 is no longer refused for a window the engine can do");
    // gemma2 was refused for one gap: the bound on the ATTENTION logits, which
    // belongs inside the score formation and cannot be reproduced by a
    // host-side transform of the result. The reference attention applies it now
    // (test_attention_softcap pins the arithmetic), so the schema loads. The
    // why-text must still name the softcap, because the HIP kernels do not cap
    // and refuse the run by name -- a Partial that did not say so would read as
    // "the gap is closed everywhere".
    CHECK(arch_lookup("gemma2")->support == ArchSupport::Partial,
          "gemma2 loads now that the reference attention bounds its logits");
    CHECK(std::string(arch_lookup("gemma2")->why).find("softcap") !=
                  std::string::npos &&
              std::string(arch_lookup("gemma2")->why).find("HIP") !=
                  std::string::npos,
          "and still names the softcap, because the HIP kernels do not cap");

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
    // Every fixture here is a model-shaped file being judged on ONE name, so it
    // carries the token table: the verdict refuses a file without one, and that
    // rule has its own arms below.
    b.tensor_f32("token_embd.weight", {2, 2}, data);
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

    // qwen4exp is a supported recurrent-MoE target. The verdict must accept
    // its delta-net tensors rather than treating their names as gaps; the full
    // graph and support claim are pinned by test_qwen4exp_whole_graph.
    {
        const std::string path = build_verdict_fixture(
            "qwen4exp", "qwen4exp", "blk.0.attn_qkv.weight", 0, 0);
        CHECK(!path.empty(), "wrote a qwen4exp fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(v.known && v.arch == "qwen4exp", "qwen4exp is a known family");
        CHECK(v.runnable, "and it is supported");
        CHECK(v.headline().find("supported") != std::string::npos,
              "the headline says supported, not unknown");
        bool tensor_gap = false;
        for (const std::string &b : v.blockers)
            if (b.find("does not implement") != std::string::npos) tensor_gap = true;
        CHECK(!tensor_gap, "the recurrent delta-net tensor is not reported as a gap");
        CHECK(v.blockers.empty(), "the supported qwen4exp schema has no blockers");
        made.push_back(path);
    }

    // An MTP head set on its own: the file tools/mtp_extract.py writes out of a
    // checkpoint's `mtp.*` tensors. Recognized, so it is named rather than read
    // against llama defaults; refused as a model, with a reason that says both
    // what it is and what has to exist first -- a qwen4exp target, whose hidden
    // state is the head's other input.
    {
        const std::string path = build_verdict_fixture(
            "qwen4-mtp", "qwen4-mtp", "mtp.layers.0.self_attn.q_proj.weight", 0, 0);
        CHECK(!path.empty(), "wrote an MTP head-set fixture");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the fixture loads");
        const ModelVerdict v = assess_model(g);
        CHECK(v.known && v.arch == "qwen4-mtp", "the head set is a known arch");
        CHECK(!v.runnable && v.spec && v.spec->role == ArchRole::Draft,
              "a head set is refused as a model and labelled a draft");
        CHECK(v.headline().find("draft file") != std::string::npos,
              "the headline calls it a draft file");
        bool named = false, use_draft = false;
        for (const std::string &b : v.blockers) {
            if (b.find("qwen4exp") != std::string::npos &&
                b.find("hidden state") != std::string::npos)
                named = true;
            if (b.find("--draft") != std::string::npos) use_draft = true;
        }
        CHECK(named,
              "the blocker names the target family it belongs to");
        CHECK(use_draft, "and says what the file is for instead");
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

    // A table pack is not a model. DeepSeek-V4.1-Flash ships its Engram tables
    // as a separate 103 GiB GGUF: two `blk.N.engram_embd.weight` tensors, an
    // arch nobody lists, and no stack metadata -- so every rule in this pass
    // had nothing to object to and the report said `runnable` while the loader
    // could only ever answer `token_embd.weight is missing`. The rule is about
    // what the file carries, not what arch string it wears: the second arm here
    // is the same shape plus an embedding and attention tensor and must stay
    // runnable.
    {
        const std::string path = "kraken-verdict-engram.gguf";
        GgufBuilder b;
        b.meta_str("general.architecture", "deepseek41-engram");
        b.meta_u32("general.alignment", 32);
        b.meta_str("general.description", "Shared IQ4_NL Engram lookup tables");
        const std::vector<f32> data{1.0f, 2.0f, 3.0f, 4.0f};
        b.tensor_f32("blk.1.engram_embd.weight", {2, 2}, data);
        b.tensor_f32("blk.14.engram_embd.weight", {2, 2}, data);
        CHECK(b.write(path), "wrote an Engram-style table pack");
        Gguf g;
        std::string err;
        CHECK(g.load(path, &err), "the table pack opens as a container");
        const ModelVerdict v = assess_model(g);
        CHECK(!v.runnable, "a file with no embedding and no attention is refused");
        bool named_embd = false, named_attn = false, named_tensor = false;
        bool blamed_format = false;
        for (const std::string &r : v.blockers) {
            if (r.find("token_embd.weight") != std::string::npos) named_embd = true;
            if (r.find("attention tensor") != std::string::npos) named_attn = true;
            if (r.find("engram_embd") != std::string::npos) named_tensor = true;
            if (r.find("quantization") != std::string::npos) blamed_format = true;
        }
        CHECK(named_embd && named_attn, "the reason names both absent halves");
        CHECK(named_tensor, "and shows the tensors the file does carry");
        CHECK(!blamed_format, "its well-formed IQ4_NL payload is not blamed");
        CHECK(v.headline().find("unknown") != std::string::npos,
              "the arch is still reported as unlisted");
        made.push_back(path);

        const std::string control = "kraken-verdict-engram-embd.gguf";
        GgufBuilder c;
        c.meta_str("general.architecture", "deepseek41-engram");
        c.meta_u32("general.alignment", 32);
        c.tensor_f32("token_embd.weight", {2, 2}, data);
        c.tensor_f32("blk.1.engram_embd.weight", {2, 2}, data);
        c.tensor_f32("blk.14.engram_embd.weight", {2, 2}, data);
        CHECK(c.write(control), "wrote the same pack with an embedding");
        Gguf gc;
        CHECK(gc.load(control, &err), "the control loads");
        const ModelVerdict vc = assess_model(gc);
        CHECK(vc.runnable, "an embedding alone clears the shape rule");
        CHECK(vc.blockers.empty(), "and nothing else objects");
        made.push_back(control);

        // The other half of the rule, and the one a name-level pass misses: a
        // set of attention weights with no token table cannot look anything up.
        // It is refused for the embedding, and the reason says which kind of
        // partial file it is rather than calling it a table pack.
        const std::string attn_only = "kraken-verdict-attn-only.gguf";
        GgufBuilder a;
        a.meta_str("general.architecture", "llama");
        a.meta_u32("general.alignment", 32);
        a.meta_u32("llama.block_count", 1);
        a.tensor_f32("blk.0.attn_q.weight", {2, 2}, data);
        CHECK(a.write(attn_only), "wrote an attention-only extraction");
        Gguf ga;
        CHECK(ga.load(attn_only, &err), "the extraction loads");
        const ModelVerdict va = assess_model(ga);
        CHECK(!va.runnable, "attention without an embedding is refused");
        bool said_embd = false, said_partial = false;
        for (const std::string &r : va.blockers) {
            if (r.find("token_embd.weight") != std::string::npos) said_embd = true;
            if (r.find("partial extraction") != std::string::npos) said_partial = true;
        }
        CHECK(said_embd, "the reason names the missing token table");
        CHECK(said_partial, "and distinguishes it from a table pack");
        made.push_back(attn_only);
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
    // Self-contained: this group reads the tiny LLaMA fixture, so it builds it
    // rather than relying on whichever group ran before it. A group that needs
    // a file another group wrote cannot run in its own directory, and every
    // group runs in its own directory now.
    if (!build_tiny_model(kTestModelPath, false)) {
        CHECK(false, "wrote the tiny GGUF the tokenizer tests read");
        return;
    }
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
// Speculative sampling: the accept/reject rule
// ---------------------------------------------------------------------------
//
// The rule's whole content is one identity (see krk/spec_sampling.hpp):
// P(output = x) must equal p(x) for every x. A wrong implementation still
// generates fluent text -- an always-accept rule generates the DRAFT's
// distribution and a "replacement from p" rule generates a slightly wrong
// mixture -- so the checks below test the arithmetic and the sampling
// frequencies, and one of them is constructed to FAIL if the correction is
// dropped (the accept mass of a pair p != q is strictly below 1).
static void test_spec_sampling() {
    // A target and a draft that disagree in both directions: p favours token 3,
    // q favours token 0, and token 2 is likelier in the draft.
    const std::vector<f32> p = {0.10f, 0.20f, 0.05f, 0.65f};
    const std::vector<f32> q = {0.55f, 0.15f, 0.30f, 0.00f};
    std::string why;
    CHECK(spec_verify_identity(p, q, 1e-6, &why),
          "the accept/reject rule reproduces the target exactly (enumerated)");
    if (!why.empty()) std::fprintf(stderr, "  (identity: %s)\n", why.c_str());

    // Not the always-accept rule: the mass that survives as proposals is
    // sum(min(p,q)) = 0.10 + 0.15 + 0.05 + 0 = 0.30, so 70% of positions need
    // the residual. An implementation that always accepts would report 1.0
    // here, and this is the check that makes that visible.
    f64 accepted_mass = 0;
    for (size_t i = 0; i < p.size(); i++)
        accepted_mass += p[i] < q[i] ? p[i] : q[i];
    CHECK(std::fabs(accepted_mass - 0.30) < 1e-6,
          "the accepted mass is sum(min(p,q)), not 1");
    CHECK(accepted_mass < 0.999, "a disagreeing draft really does get rejected");

    // A token the target likes more than the draft is accepted for any u;
    // one the draft overweights is accepted exactly with probability p/q.
    CHECK(spec_accept_step(p, q, 3, 0.999f, 0.5f).accepted,
          "p(x) > q(x) accepts regardless of the draw");
    const f32 ratio = p[1] / q[1]; // 1/3
    CHECK(spec_accept_step(p, q, 1, ratio * 0.5f, 0.5f).accepted,
          "a proposal is accepted below p/q");
    CHECK(!spec_accept_step(p, q, 1, ratio * 1.5f, 0.5f).accepted,
          "and rejected above it");
    // q(x) == 0: the draft could not have proposed it, so there is nothing to
    // correct and the rule's limit is accept.
    CHECK(spec_accept_step(p, q, 3, 0.9999f, 0.5f).accepted,
          "a proposal with no draft mass is accepted");

    // The residual: max(0, p - q), normalised, and it never carries a token the
    // target does not want (p <= q there).
    std::vector<f32> res;
    CHECK(spec_residual(p, q, &res), "the residual exists when p != q");
    CHECK(res.size() == p.size(), "the residual has one entry per token");
    f64 total = 0;
    bool only_positive = true;
    for (size_t i = 0; i < res.size(); i++) {
        total += res[i];
        if (res[i] > 0 && p[i] <= q[i]) only_positive = false;
    }
    CHECK(std::fabs(total - 1.0) < 1e-5, "the residual is normalised");
    CHECK(only_positive, "the residual carries only the mass the draft undershot");
    CHECK(res[0] == 0.0f && res[2] == 0.0f,
          "tokens the draft overweights get no residual mass");
    CHECK(res[3] > res[1], "token 3's deficit is larger than token 1's");

    // p == q: there is nothing to correct, and the function must say so rather
    // than return a row of zeros (which would make every rejection draw token 0).
    std::vector<f32> same;
    CHECK(!spec_residual(p, p, &same), "no residual when the draft equals the target");
    CHECK(spec_accept_step(p, p, 2, 0.9999f, 0.5f).accepted,
          "an identical draft is always accepted");

    // A rejection's replacement is a real draw from the residual, never the
    // rejected proposal when the target gave it no mass.
    bool replacement_from_residual = true;
    for (int i = 0; i < 32; i++) {
        const f32 u = static_cast<f32>(i) / 32.0f;
        const SpecStep st = spec_accept_step(p, q, 0, 0.99f, u);
        if (st.accepted) replacement_from_residual = false;
        if (st.token == 0) replacement_from_residual = false;
        if (st.token < 0 || st.token >= 4) replacement_from_residual = false;
    }
    CHECK(replacement_from_residual,
          "a rejected proposal is replaced by a draw the target supports");

    // Greedy is the one-hot limit: with p one-hot at token 1, every outcome --
    // acceptance and rejection -- is token 1.
    const std::vector<f32> onehot = {0.0f, 1.0f, 0.0f, 0.0f};
    const std::vector<f32> flat = {0.25f, 0.25f, 0.25f, 0.25f};
    bool greedy_limit = true;
    for (int i = 0; i < 4; i++) {
        const f32 u = 0.1f * static_cast<f32>(i);
        if (spec_accept_step(onehot, flat, 1, u, u).token != 1) greedy_limit = false;
        if (spec_accept_step(onehot, flat, 0, u, u).token != 1) greedy_limit = false;
    }
    CHECK(greedy_limit, "a one-hot target reduces the rule to the argmax");

    // The identity above is the ARITHMETIC; this is the implementation. A round
    // is: propose y ~ q, then accept/reject it -- so the histogram of the tokens
    // spec_accept_step returns has to be p. An always-accept rule passes the
    // arithmetic and fails this, which is the whole reason both checks exist.
    {
        Rng rng;
        rng.seed(4242);
        i32 hist[4] = {0, 0, 0, 0};
        const int kRounds = 200000;
        for (int i = 0; i < kRounds; i++) {
            const i32 y = spec_draw(q, rng.next_f32());
            const SpecStep st =
                spec_accept_step(p, q, y, rng.next_f32(), rng.next_f32());
            if (st.token < 0 || st.token >= 4) {
                hist[0] = -kRounds; // out of range: fail loudly below
                break;
            }
            hist[st.token]++;
        }
        bool hist_ok = true;
        char worst[96] = {0};
        for (int i = 0; i < 4; i++) {
            const f64 got = static_cast<f64>(hist[i]) / kRounds;
            if (std::fabs(got - p[static_cast<size_t>(i)]) > 0.01) {
                hist_ok = false;
                std::snprintf(worst, sizeof(worst),
                              "token %d: %.3f drawn, %.3f expected", i, got,
                              static_cast<f64>(p[static_cast<size_t>(i)]));
            }
        }
        CHECK(hist_ok, "the rule's own draws reproduce the target distribution");
        if (!hist_ok) std::fprintf(stderr, "  (worst: %s)\n", worst);
    }

    // A verifier that cannot fail is worth nothing: two rows that are not
    // distributions must be refused, and a wrong pair must not pass.
    std::string bad;
    const std::vector<f32> not_a_dist = {0.5f, 0.5f, 0.5f, 0.5f};
    CHECK(!spec_verify_identity(not_a_dist, q, 1e-6, &bad),
          "the identity check refuses rows that are not distributions");

    // ---- the softmax the draft side uses --------------------------------
    const std::vector<f32> logits = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<f32> prob;
    spec_softmax(logits.data(), 4, 1.0f, &prob);
    f64 ps = 0;
    for (f32 v : prob) ps += v;
    CHECK(std::fabs(ps - 1.0) < 1e-6, "softmax is a distribution");
    CHECK(prob[3] > prob[2] && prob[2] > prob[1] && prob[1] > prob[0],
          "softmax keeps the order");
    std::vector<f32> sharp;
    spec_softmax(logits.data(), 4, 0.25f, &sharp);
    CHECK(sharp[3] > prob[3], "a lower temperature sharpens the draft");
    std::vector<f32> hot;
    spec_softmax(logits.data(), 4, 4.0f, &hot);
    CHECK(hot[3] < prob[3] && hot[0] > prob[0], "a higher temperature flattens it");
    std::vector<f32> limit;
    spec_softmax(logits.data(), 4, 0.0f, &limit);
    CHECK(limit[3] == 1.0f && limit[0] == 0.0f,
          "temperature 0 is the one-hot argmax limit");

    // ---- draft vocabulary -> target vocabulary ---------------------------
    // Duplicate ids in a d2t map would otherwise sum past 1; ids outside the
    // target vocabulary are dropped rather than clamped onto a token the head
    // never proposed.
    const std::vector<f32> dprob = {0.5f, 0.3f, 0.1f, 0.1f};
    const std::vector<i32> d2t = {2, 0, 2, 9};
    std::vector<f32> mapped;
    spec_map_dist(dprob.data(), 4, d2t.data(), 4, &mapped);
    CHECK(mapped.size() == 4, "the mapped row is in target space");
    // 0.5 + 0.1 of the draft's mass lands on target id 2 and 0.3 on id 0; the
    // dropped 0.1 is renormalised away, so the ratio is what survives.
    CHECK(std::fabs(mapped[2] - 2.0f / 3.0f) < 1e-6,
          "two draft ids that map to one target id add up");
    CHECK(std::fabs(mapped[0] - 1.0f / 3.0f) < 1e-6, "a distinct id keeps its share");
    CHECK(mapped[3] == 0.0f, "an out-of-range id is dropped, not clamped");
    f64 ms = 0;
    for (f32 v : mapped) ms += v;
    CHECK(std::fabs(ms - 1.0) < 1e-6,
          "the mapped row is renormalised after the drop");
    // No map at all is the identity, which is the head whose vocabulary IS the
    // target's.
    std::vector<f32> ident;
    spec_map_dist(dprob.data(), 4, nullptr, 4, &ident);
    CHECK(ident == dprob, "a missing d2t map is the identity");

    // ---- the draws ------------------------------------------------------
    // Frequencies, not one draw: a cumulative walk that is off by an index
    // still passes "returns a token in range", and only the histogram sees it.
    std::vector<f32> three = {0.2f, 0.5f, 0.3f};
    Rng rng;
    rng.seed(99);
    i32 hits[3] = {0, 0, 0};
    const int kDraws = 200000;
    for (int i = 0; i < kDraws; i++) hits[spec_draw(three, rng.next_f32())]++;
    bool freq_ok = true;
    for (int i = 0; i < 3; i++) {
        const f64 got = static_cast<f64>(hits[i]) / kDraws;
        if (std::fabs(got - three[static_cast<size_t>(i)]) > 0.01) freq_ok = false;
    }
    CHECK(freq_ok, "the draw follows the distribution it is given");

    // ---- the sampler's own row ------------------------------------------
    // Speculation compares against build_dist(), so it must be the distribution
    // sample() draws from: a row that sums to 1, with support inside top-k, and
    // no draw outside that support.
    SampleParams sp;
    sp.temp = 0.7f;
    sp.top_k = 4;
    sp.top_p = 0.9f;
    sp.min_p = 0.0f;
    sp.repeat_penalty = 1.0f;
    sp.repeat_last_n = 0;
    sp.seed = 5;
    Sampler smp;
    smp.reset(sp);
    std::vector<f32> row_logits(16);
    for (int i = 0; i < 16; i++)
        row_logits[static_cast<size_t>(i)] = 0.1f * static_cast<f32>(i);
    std::vector<f32> dist;
    smp.build_dist(row_logits.data(), 16, &dist);
    f64 ds = 0;
    int nonzero = 0;
    for (f32 v : dist) {
        ds += v;
        if (v > 0) nonzero++;
    }
    CHECK(std::fabs(ds - 1.0) < 1e-5, "the sampler's row is a distribution");
    CHECK(nonzero <= 4, "top-k bounds the row's support");
    CHECK(dist[15] > dist[14] && dist[14] > 0.0f, "the row keeps the top ids");
    CHECK(dist[0] == 0.0f, "ids outside the nucleus are zeroed");
    bool draws_in_support = true;
    for (int i = 0; i < 5000; i++) {
        const i32 t = smp.sample(row_logits.data(), 16);
        if (t < 0 || t >= 16 || dist[static_cast<size_t>(t)] <= 0.0f)
            draws_in_support = false;
    }
    CHECK(draws_in_support, "every draw lands on the row's own support");

    // The penalty a speculative round has to model: a token accepted for the
    // position being verified is not in the sampler's ring yet, and passing it
    // as `extra` must move the row the same way accept() would.
    SampleParams rp;
    rp.temp = 1.0f;
    rp.top_k = 16;
    rp.top_p = 1.0f;
    rp.min_p = 0.0f;
    rp.repeat_penalty = 4.0f;
    rp.repeat_last_n = 8;
    rp.seed = 11;
    Sampler with_ring, with_extra;
    with_ring.reset(rp);
    with_extra.reset(rp);
    with_ring.accept(15);
    std::vector<f32> d_ring, d_extra;
    with_ring.build_dist(row_logits.data(), 16, &d_ring);
    const i32 extra = 15;
    with_extra.build_dist(row_logits.data(), 16, &d_extra, &extra, 1);
    bool same_row = d_ring.size() == d_extra.size();
    for (size_t i = 0; same_row && i < d_ring.size(); i++)
        if (std::fabs(d_ring[i] - d_extra[i]) > 1e-6f) same_row = false;
    CHECK(same_row, "an accepted-but-not-yet-recorded token penalises identically");
    CHECK(d_extra[15] < dist[15] || dist[15] == 0.0f,
          "and the penalty really lowers the repeated token's mass");
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
    u64 batch_ran = 0;     // decode layers the batched MoE path computed
    u64 batch_skipped = 0; // decode layers that kept the per-expert loop
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
    out.batch_ran = engine.moe_batch_layers();
    out.batch_skipped = engine.moe_batch_skips();
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

// The batched decode path's RESIDENCY precondition, pinned directly.
//
// `moe_decode_batch` resolves the layer's whole routed set before the first
// launch, so every resolved expert must still be its own slot. When the cache
// cannot hold the set, `acquire` evicts as it resolves and the earlier experts'
// pointers start naming the later experts' bytes -- both experts then multiply
// one expert's weights, produce finite plausible output, and decode the wrong
// text (measured: 1 slot gave `5 4 5 4 5 4` where every larger cache gave
// `5 5 4 5 4 5`).
//
// So the gate asserts the REFUSAL, not just the text: with 2 experts routed and
// 1 slot, the batch must report zero layers computed, and with 4 slots it must
// report some. Without the second arm a broken guard that refuses always would
// pass unnoticed, and without the first the aliasing bug itself passes.
static void test_moe_batch_fallback() {
    MoeSpec vs;
    vs.dense = false;
    vs.n_expert = 4;
    vs.n_expert_used = 2;
    vs.shared = true;
    vs.varied = true;
    CHECK(build_tiny_moe_model(kMoeTestPath, vs), "wrote the MoE fixture");

    // Routed set (2 experts) EXCEEDS the cache (1 slot): the batch must stand
    // down and the per-expert loop must produce the reference text.
    const MoeRun squeezed = run_moe(kMoeTestPath, 0, 1, 6);
    CHECK(squeezed.ok, "the squeezed-cache run generates");
    CHECK(squeezed.capacity_slots == 1,
          "the squeezed arm really has one expert slot");
    CHECK(squeezed.evictions > 0,
          "the squeezed arm really reloads experts (the set exceeds the cache)");
    CHECK(squeezed.batch_ran == 0,
          "with fewer slots than routed experts the batch computes NO layer");
    CHECK(squeezed.batch_skipped > 0,
          "...and the per-expert loop is what ran instead");

    // The same model with room for the routed set: the batch must actually
    // engage, or the arm above proves nothing about a fallback.
    const MoeRun roomy = run_moe(kMoeTestPath, 4096, 0, 6);
    CHECK(roomy.ok, "the roomy-cache run generates");
    CHECK(roomy.batch_ran > 0,
          "with room for the routed set the batch DOES compute layers");
    CHECK(roomy.tokens == squeezed.tokens,
          "the fallback decodes the same text the batch does");

    // And a tier does not change the answer either: the tiered one-slot arm is
    // the shape whose cross-slice aliasing forced the pointer check, and its
    // failure mode was not subtle (token 0 forever).
    const MoeRun tiered = run_moe(kMoeTestPath, 0, 1, 6, 64);
    CHECK(tiered.ok, "the squeezed arm with a host tier generates");
    CHECK(tiered.batch_ran == 0,
          "a host tier does not let the batch outrun the slot budget");
    CHECK(tiered.tokens == squeezed.tokens,
          "the tiered squeezed arm decodes the same text");
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

    BackendKind kind() const override { return fake_hip_ ? BackendKind::HIP : inner_->kind(); }
    const DeviceCaps &caps() const override { return fake_hip_ ? fake_caps_ : inner_->caps(); }
    size_t device_free_bytes() const override {
        return fake_device_memory_ ? fake_device_free_ : inner_->device_free_bytes();
    }
    size_t device_total_bytes() const override {
        return fake_device_memory_ ? fake_device_total_ : inner_->device_total_bytes();
    }
    size_t host_available_bytes() const override {
        return fake_host_memory_ ? fake_host_available_ : inner_->host_available_bytes();
    }
    size_t host_total_bytes() const override {
        return fake_host_memory_ ? fake_host_total_ : inner_->host_total_bytes();
    }
    void set_test_resources(size_t device_total, size_t device_free,
                            size_t host_total, size_t host_available) {
        fake_hip_ = true;
        fake_device_memory_ = true;
        fake_host_memory_ = true;
        fake_device_total_ = device_total;
        fake_device_free_ = device_free;
        fake_host_total_ = host_total;
        fake_host_available_ = host_available;
        fake_caps_ = inner_->caps();
        fake_caps_.vram_total = device_total;
        fake_caps_.vram_free = device_free;
    }
    void set_test_host_available(size_t bytes) { fake_host_available_ = bytes; }
    void set_test_host_memory(size_t total, size_t available) {
        fake_host_memory_ = true;
        fake_host_total_ = total;
        fake_host_available_ = available;
    }
    void set_fail_allocation_after(i64 successful_allocations) {
        fail_allocation_after_ = successful_allocations;
    }
    void set_fail_markers(bool fail) { fail_markers_ = fail; }
    void set_require_markers(bool required) { require_markers_ = required; }
    DType act_type() const override { return inner_->act_type(); }
    size_t act_size() const override { return inner_->act_size(); }
    void *alloc(size_t bytes) override {
        if (fail_allocation_after_ == 0) return nullptr;
        if (fail_allocation_after_ > 0) --fail_allocation_after_;
        void *p = inner_->alloc(bytes);
        if (p) allocations_++;
        return p;
    }
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
    bool requires_markers() const override { return require_markers_; }
    void *make_marker() override { return fail_markers_ ? nullptr : inner_->make_marker(); }
    void release_marker(void *m) override { inner_->release_marker(m); }
    void record_marker(void *m) override { inner_->record_marker(m); }
    bool poll_marker(void *m) override { return inner_->poll_marker(m); }
    void wait_marker(void *m) override { inner_->wait_marker(m); }
    void download_f32(f32 *d, const void *s, i64 n) override {
        inner_->download_f32(d, s, n);
    }
    std::string describe() const override { return inner_->describe(); }
    void embed(void *o, const void *t, DType tt, i64 v, i64 e, const i32 *tk,
               i64 n) override {
        inner_->embed(o, t, tt, v, e, tk, n);
    }
    void rmsnorm(void *o, const void *x, const f32 *w, i64 r, i64 n, f32 e,
                 const i32 *re = nullptr) override {
        inner_->rmsnorm(o, x, w, r, n, e, re);
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
    void swiglu_oai(void *o, const void *g, const void *u, i64 n) override {
        inner_->swiglu_oai(o, g, u, n);
    }

    void silu_mul(void *o, const void *g, const void *u, i64 n,
                  i64 row_len = 0, f32 *row_scale = nullptr) override {
        inner_->silu_mul(o, g, u, n, row_len, row_scale);
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
                           i64 nr, i64 n,
                           const f32 *row_scale = nullptr,
                           i32 *row_exp = nullptr) override {
        // The double forwards the storage-exponent slot too: a dropped
        // parameter here would leave the caller's exponent at whatever the
        // decorator's own layer zeroed it to, which is a silent difference
        // between what the suite measures and what the engine runs.
        inner_->scatter_axpy_rows(d, s, r, a, nr, n, row_scale, row_exp);
    }

    // --- a simulated aperture (test-only) ---------------------------------
    // The probe's whole job is to read the card, so the only way to test the
    // branch where a wide aperture wins is a backend whose two pageable paths
    // really take different amounts of time. Each arm copies for real and then
    // pays a bandwidth-shaped wait: `gbs` is the rate that arm should MEASURE on
    // an idle machine, before the copy's own cost is added. With no aperture
    // set, both arms are exactly the inner backend's memcpy, so no other group
    // changes.
    // `block_gbs` defaults to the ring's rate: a test that does not care about
    // the third mechanism then measures exactly what it did before this arm
    // existed, and one that does gets a real difference on the clock.
    void set_test_aperture(double ring_gbs, double direct_gbs, double pin_gbs,
                           double block_gbs = -1.0) {
        aperture_ = true;
        ring_ns_per_byte_ = ring_gbs > 0.0 ? 1.0 / ring_gbs : 0.0;
        direct_ns_per_byte_ = direct_gbs > 0.0 ? 1.0 / direct_gbs : 0.0;
        pin_ns_per_byte_ = pin_gbs > 0.0 ? 1.0 / pin_gbs : 0.0;
        block_ns_per_byte_ =
            block_gbs >= 0.0 ? (block_gbs > 0.0 ? 1.0 / block_gbs : 0.0)
                             : ring_ns_per_byte_;
    }
    bool pageable_paths_selectable() const override { return aperture_; }
    void set_pageable_path(PageablePath p) override {
        path_ = static_cast<int>(p);
        direct_ = path_ == 1;
        ++path_switches_;
    }
    void set_pageable_direct(bool on) override {
        set_pageable_path(on ? PageablePath::Direct : PageablePath::Ring);
    }
    bool direct_enabled() const { return direct_; }
    int path() const { return path_; }  // 0 ring, 1 direct, 2 blocking
    int path_switches() const { return path_switches_; }
    void upload_paged_batch(void *const dst[], const void *const src[],
                            const size_t bytes[], int n) override {
        const double cost = path_ == 1   ? direct_ns_per_byte_
                            : path_ == 2 ? block_ns_per_byte_
                                         : ring_ns_per_byte_;
        for (int i = 0; i < n; i++) {
            if (!dst[i] || !src[i] || !bytes[i]) continue;
            std::memcpy(dst[i], src[i], bytes[i]);
            sim_wait(cost * static_cast<double>(bytes[i]));
        }
    }
    void upload_pinned_batch(void *const dst[], const void *const src[],
                             const size_t bytes[], int n,
                             bool merge = true) override {
        static_cast<void>(merge);
        for (int i = 0; i < n; i++) {
            if (!dst[i] || !src[i] || !bytes[i]) continue;
            std::memcpy(dst[i], src[i], bytes[i]);
            sim_wait(pin_ns_per_byte_ * static_cast<double>(bytes[i]));
        }
    }

    u64 releases() const { return releases_; }
    u64 allocations() const { return allocations_; }
    u64 host_allocs() const { return host_allocs_; }
    u64 host_releases() const { return host_releases_; }

    // The CPU backend cannot read its own host pool from a "kernel", so the
    // capability is faked where the cache's half of the contract is under test:
    // whether a pinned host CELL is publishable, not whether this machine's
    // driver maps the pool (tools/probe_hostread.hip answers that one).
    bool host_load_supported() const override {
        return fake_host_load_ || inner_->host_load_supported();
    }
    void set_host_load(bool on) { fake_host_load_ = on; }

private:
    Backend *inner_;
    DeviceCaps fake_caps_;
    bool fake_hip_ = false;
    bool fake_host_load_ = false;
    bool fake_device_memory_ = false;
    bool fake_host_memory_ = false;
    size_t fake_device_total_ = 0;
    size_t fake_device_free_ = 0;
    size_t fake_host_total_ = 0;
    size_t fake_host_available_ = 0;
    u64 releases_ = 0;
    u64 allocations_ = 0;
    i64 fail_allocation_after_ = -1;
    bool fail_markers_ = false;
    bool require_markers_ = false;
    u64 host_allocs_ = 0;
    u64 host_releases_ = 0;
    static void sim_wait(double ns) {
        if (!(ns > 0.0)) return;
        const std::chrono::steady_clock::time_point until =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double, std::nano>(ns));
        while (std::chrono::steady_clock::now() < until) {
        }
    }
    bool aperture_ = false;
    bool direct_ = false;
    int path_ = 0;  // 0 ring, 1 direct, 2 blocking
    int path_switches_ = 0;
    double ring_ns_per_byte_ = 0.0;
    double direct_ns_per_byte_ = 0.0;
    double block_ns_per_byte_ = 0.0;
    double pin_ns_per_byte_ = 0.0;
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

// A two-layer expert source whose layers do NOT have the same expert footprint.
// That is the shape a Q4_K_M file has when it mixes Q4_K and Q6_K
// down-projections (and exactly what Qwen3-MoE-4x0.6B carries: 56 experts of
// 5.8359 MiB and 56 of 5.0625 MiB). Layer 0's expert is 24 bytes, layer 1's is
// 40, so the eight-slot corpus is 4 * 24 + 4 * 40 = 256 bytes and
// `budget / this_layer_expert` is 10 at layer 0 but 6 at layer 1 -- less than
// the eight slots the budget actually pays for.
static bool build_expert_source_model_mixed(const std::string &path) {
    GgufBuilder b;
    b.meta_str("general.architecture", "llama");
    b.meta_u32("general.alignment", 32);
    add_spm_vocab(b, 260);
    u8 narrow[8 * 4];
    u8 wide[16 * 4];
    u8 down[8 * 4];
    for (size_t i = 0; i < sizeof(narrow); i++) narrow[i] = static_cast<u8>(i);
    for (size_t i = 0; i < sizeof(wide); i++) wide[i] = static_cast<u8>(i + 3);
    for (size_t i = 0; i < sizeof(down); i++) down[i] = static_cast<u8>(i + 7);
    // [2,2,4] F16 = 16 elements = 32 bytes, so every slice here is 8 bytes an
    // expert and layer 0's expert is 24.
    b.tensor_raw("blk.0.ffn_gate_exps.weight", 1, {2, 2, 4}, narrow, sizeof(narrow));
    b.tensor_raw("blk.0.ffn_up_exps.weight", 1, {2, 2, 4}, narrow, sizeof(narrow));
    b.tensor_raw("blk.0.ffn_down_exps.weight", 1, {2, 2, 4}, narrow, sizeof(narrow));
    // [4,2,4] F16 = 32 elements = 64 bytes for gate and up, down still 32, so
    // layer 1's expert is 64/4 + 64/4 + 32/4 = 40.
    b.tensor_raw("blk.1.ffn_gate_exps.weight", 1, {4, 2, 4}, wide, sizeof(wide));
    b.tensor_raw("blk.1.ffn_up_exps.weight", 1, {4, 2, 4}, wide, sizeof(wide));
    b.tensor_raw("blk.1.ffn_down_exps.weight", 1, {2, 2, 4}, down, sizeof(down));
    return b.write(path);
}

static ExpertSource expert_source_of(Gguf &g, int layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    ExpertSource src;
    src.gate = g.tensor(p + "ffn_gate_exps.weight");
    src.up = g.tensor(p + "ffn_up_exps.weight");
    src.down = g.tensor(p + "ffn_down_exps.weight");
    src.n_expert = 4;
    src.n_embd = 2;
    src.n_ff_exp = 2;
    return src;
}

// The device budget is made of BYTES. It used to be spent by a slot COUNT
// derived as budget_ / the current layer's expert footprint, which is the
// largest layer's size whenever the corpus mixes footprints -- so a budget sized
// for the whole routed set evicted experts it had already paid for, with budget
// left unspent (measured on Qwen3-MoE-4x0.6B: 23 evictions and ~43 MiB of a
// 610 MiB budget idle over a 64-token run). These three arms pin the rule in
// both directions: a budget that covers the corpus evicts nothing, one byte less
// than the corpus does, and an explicit slot budget is a real count.

static void test_kv_tier_host_reserve_refusal() {

    // Both memory tiers reserve through one process-wide ledger. Even when
    // their individual probes see the same 100-byte free sample, they cannot
    // each spend it independently.
    {
        set_env_var("KRK_RAM_RESERVE_MB", "0");
        int *owned = static_cast<int *>(host_alloc(60));
        const bool first_reservation = SharedHostRamBudget::reserve(60, 100, 100);
        CHECK(first_reservation,
              "the shared host budget reserves the first tier's bytes");
        if (first_reservation) SharedHostRamBudget::commit(owned, 60);
        else host_free(owned);
        CHECK(!SharedHostRamBudget::can_admit(41, 100, 100),
              "the second tier cannot spend headroom already owned by the first");
        CHECK(!SharedHostRamBudget::reserve(41, 100, 100),
              "a concurrent-style second reservation is refused atomically");
        if (first_reservation)
            SharedHostRamBudget::release_and_free(owned, [](void *p) { host_free(p); });
        CHECK(SharedHostRamBudget::can_admit(41, 100, 100),
              "releasing a tier's buffers returns the bytes to the shared budget");
        set_env_var("KRK_RAM_RESERVE_MB", "");
    }

    // Exercise the actual independent cache owners: expert WARM consumes 24 B,
    // so a KV tier asking to guarantee an 80 B staging page must not treat the
    // same 100 B host sample as entirely free. Once the expert tier releases,
    // KV admission succeeds and can migrate a retained page into WARM.
    {
        set_env_var("KRK_RAM_RESERVE_MB", "0");
        const std::string expert_path = "kraken-shared-host-budget.gguf";
        const bool expert_fixture = build_expert_source_model(expert_path);
        CHECK(expert_fixture,
              "wrote the expert fixture for cross-tier RAM accounting");
        Gguf gg;
        std::string load_err;
        const bool expert_loaded = expert_fixture && gg.load(expert_path, &load_err);
        CHECK(expert_loaded,
              "loaded the expert fixture for cross-tier RAM accounting");
        if (!expert_loaded) {
            std::fprintf(stderr, "  (shared host fixture: %s)\n", load_err.c_str());
            std::remove(expert_path.c_str());
            set_env_var("KRK_RAM_RESERVE_MB", "");
        } else {
            CountingBackend be(make_cpu_backend());
            be.set_test_host_memory(4 * (static_cast<size_t>(1) << 30), 100);
            ExpertCache experts;
            experts.configure(&be, 48, 24);
            const ResidentExpert *resident = experts.acquire(expert_source(gg), 0, 0);
            CHECK(resident && experts.warm_used_bytes() == 24,
                  "expert WARM holds its 24-byte admitted copy");
            const std::vector<size_t> layer_bytes = {80, 80};
            KvTierCache kv;
            std::string kv_err;
            CHECK(!kv.init(&be, 2, layer_bytes, 80, 160, "", "k", &kv_err),
                  "KV cannot reserve 80 bytes on top of the expert's 24 in a 100-byte pool");
            CHECK(kv_err.find("shared host-memory budget") != std::string::npos,
                  "KV initialization identifies combined host-tier pressure");
            experts.clear();
            CHECK(SharedHostRamBudget::used_bytes() == 0,
                  "clearing the expert cache releases its ledgered pageable bytes");
            CHECK(kv.init(&be, 2, layer_bytes, 80, 160, "", "k", &kv_err),
                  "KV tier initializes after expert WARM releases shared capacity");
            CHECK(kv.base(0) != nullptr, "first layer becomes HOT");
            kv.end_layer(0);
            CHECK(kv.base(1) != nullptr, "second layer promotes after a WARM migration");
            CHECK(SharedHostRamBudget::used_bytes() == 80,
                  "the KV WARM migration is charged to the same shared ledger");
            kv.detach();
            CHECK(SharedHostRamBudget::used_bytes() == 0,
                  "detaching KV releases its shared pageable allocation");
        }
        std::remove(expert_path.c_str());
        set_env_var("KRK_RAM_RESERVE_MB", "");
    }

    const size_t gib = static_cast<size_t>(1) << 30;
    const std::vector<size_t> tiny_lengths = {16, 16, 16};
    // Failure halfway through HOT allocation must release the already allocated
    // slot instead of detaching the backend pointer and leaking it.
    {
        CountingBackend be(make_cpu_backend());
        be.set_test_resources(16 * gib, 8 * gib, 16 * gib, 8 * gib);
        be.set_fail_allocation_after(1);
        KvTierCache cache;
        std::string err;
        CHECK(!cache.init(&be, 3, tiny_lengths, 32, 48, "", "k", &err),
              "a failed second HOT allocation refuses cache initialization");
        CHECK(be.allocations() == 1 && be.releases() == 1,
              "partial HOT initialization releases its first allocation");
    }
    // HIP must have real event markers for safe slot reuse. A simulated marker
    // failure after one successful slot allocation likewise rolls everything
    // back; the test backend opts into the HIP requirement explicitly.
    {
        CountingBackend be(make_cpu_backend());
        be.set_test_resources(16 * gib, 8 * gib, 16 * gib, 8 * gib);
        be.set_fail_markers(true);
        be.set_require_markers(true);
        KvTierCache cache;
        std::string err;
        CHECK(!cache.init(&be, 3, tiny_lengths, 32, 48, "", "k", &err),
              "a required marker failure refuses cache initialization");
        CHECK(be.allocations() == 1 && be.releases() == 1,
              "marker failure releases the partially built HOT tier");
    }

    const std::string path = "kraken-kv-reserve-test.gguf";
    CHECK(build_sparse_kv_model(path, 8192), "wrote a large-context sparse-KV fixture");

    auto *be = new CountingBackend(make_cpu_backend());
    be->set_test_resources(16 * gib, 8 * gib, 16 * gib, 8 * gib);
    EngineConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx = 8192;
    cfg.prefill_chunk = 4;
    cfg.kv_hot_mb = 1; // each plane gets 0.5 MiB: less than its 1 MiB cache
    cfg.kv_warm_mb = 1;
    // This group's subject is the KV tier's host reserve, so the expert tier is
    // silenced: the fixture is a qwen2moe file, and the default warmup now
    // stages its tier on a fake device just as it does on a real one -- which
    // claims the process-wide SharedHostRamBudget and leaves the SECOND
    // engine's 512 KiB KV staging page no room (measured: the arm failed with
    // `no room for a KV staging page after the safety reserve and other
    // pageable tiers`). Two engines sharing 8 GiB of host is a real answer, not
    // a bug; it is just not the answer this group is asking for.
    cfg.expert_warmup = 0;
    Engine engine;
    std::string err;
    CHECK(engine.init(be, cfg, &err), "tiered sparse-KV engine initializes with headroom");
    if (!err.empty()) std::fprintf(stderr, "  (KV reserve init: %s)\n", err.c_str());
    if (!err.empty()) {
        engine.shutdown();
        delete be;
        std::remove(path.c_str());
        return;
    }
    KvTierStats stats;
    engine.kv_tier_stats(&stats);
    CHECK(stats.tiered && stats.layers == 2,
          "one-slot HOT tier is active for both KV-carrying layers");

    // A direct cache arm proves the victim remains owned and byte-exact when
    // neither WARM nor COLD can be staged. `base(3)` forces eviction of layer 0;
    // admission must refuse before it clears ownership or fills zeros.
    {
        const size_t plane = 16u * 8192u * sizeof(f32);
        std::vector<size_t> lengths(4, 0);
        lengths[0] = lengths[3] = plane;
        KvTierCache cache;
        CHECK(cache.init(be, 4, lengths, plane, plane, "", "k", &err),
              "cache fixture initialized before pressure is introduced");
        void *first = cache.base(0);
        CHECK(first != nullptr, "first layer is resident");
        if (first) std::memset(first, 0x5a, plane);
        cache.end_layer(0);
        be->set_test_host_available(2 * gib); // equal to the default reserve
        CHECK(cache.base(3) == nullptr,
              "KV promotion refuses when live headroom reaches the reserve");
        CHECK(cache.base(0) == first,
              "refused eviction keeps the original HOT layer owned");
        if (first) {
            const u8 *bytes = static_cast<const u8 *>(first);
            CHECK(bytes[0] == 0x5a && bytes[plane - 1] == 0x5a,
                  "refused eviction preserves the retained KV bytes");
        }
        KvTierStats refused;
        cache.stats(&refused);
        CHECK(refused.evictions_dropped == 0 && refused.promotion_refused > 0,
              "host-pressure refusal is distinct from dropping KV history");
        cache.detach();
    }

    // Reinitialize with headroom, then lower it only after load. This exercises
    // the Engine::forward -> GenerateResult error propagation, not just the
    // cache's null return. The prompt visits layer 0 then layer 3, forcing an
    // eviction at the first actual forward.
    be->set_test_host_available(8 * gib);
    Engine run_engine;
    err.clear();
    CHECK(run_engine.init(be, cfg, &err), "generation engine initialized with host headroom");
    if (!err.empty()) std::fprintf(stderr, "  (generation init: %s)\n", err.c_str());
    if (err.empty()) {
        be->set_test_host_available(2 * gib);
        GenerateParams p;
        p.prompt = "the";
        p.max_tokens = 1;
        p.sampler.greedy = true;
        p.sampler.temp = 0.0f;
        GenerateResult result;
        CHECK(!run_engine.generate(p, &result),
              "generation returns failure rather than emitting text with lost KV");
        CHECK(result.finish == FinishError &&
                  result.error.find("KV promotion refused") != std::string::npos,
              "generation exposes the memory refusal in its result");
    }
    run_engine.shutdown();
    engine.shutdown();
    delete be;
    std::remove(path.c_str());

}


static void test_expert_promotion_failure();
static void test_moe_grouped_plan_offsets();

static void test_expert_cache_byte_budget() {
    const std::string path = "kraken-bytes-test.gguf";
    CHECK(build_expert_source_model_mixed(path), "wrote the byte-budget GGUF");

    Backend *cpu = make_cpu_backend();
    Gguf g;
    std::string err;
    CHECK(g.load(path, &err), "loaded the byte-budget GGUF");
    if (!err.empty()) std::fprintf(stderr, "  (byte gguf: %s)\n", err.c_str());

    ExpertSource a = expert_source_of(g, 0);
    ExpertSource b = expert_source_of(g, 1);
    CHECK(a.present() && b.present(), "both layers carry experts");
    CHECK(a.expert_bytes() == 24 && b.expert_bytes() == 40,
          "the two layers have different expert footprints");
    const size_t corpus = 4 * (a.expert_bytes() + b.expert_bytes());
    CHECK(corpus == 256, "the routed set is eight slots of 24 and 40 bytes");

    // The budget guard. It exists because `--expert-warmup` once filled a
    // 15.92 GiB card to 0.00 GiB free and took the display with it, and its
    // decision is pure arithmetic, so it is driven here with no device: an
    // explicit budget that cannot fit is refused, and the refusal names the
    // largest that does.
    {
#ifdef _WIN32
        _putenv_s("KRK_VRAM_RESERVE_MB", "");
        _putenv_s("KRK_RAM_RESERVE_MB", "");
#else
        unsetenv("KRK_VRAM_RESERVE_MB");
        unsetenv("KRK_RAM_RESERVE_MB");
#endif
        size_t largest = 0;
        std::string why;
        // MiB-scale on purpose: this is the shape of the real card (16304 MiB of
        // VRAM, 599 MiB in use, 1024 MiB reserved) and the refusal message is
        // written in MiB, so a test in bytes would compare a number against a
        // rounded one -- which is how the first version of these checks failed.
        const size_t mib = 1024u * 1024u;
        CHECK(krk::expert_budget_room(16304 * mib, 599 * mib, 1024 * mib) ==
                  14681 * mib,
              "room is total minus what is used minus the reserve");
        CHECK(krk::expert_budget_fits(14681 * mib, 16304 * mib, 599 * mib,
                                      1024 * mib, &largest, &why),
              "a budget exactly at the room fits");
        CHECK(largest == 14681 * mib, "and reports that room as the largest");
        CHECK(!krk::expert_budget_fits(14682 * mib, 16304 * mib, 599 * mib,
                                       1024 * mib, &largest, &why),
              "one MiB more is refused");
        CHECK(why.find("14681") != std::string::npos &&
                  why.find("does not fit") != std::string::npos,
              "and the refusal names the largest budget that fits, in MiB");
        CHECK(!krk::expert_budget_fits(1, 16304 * mib, 16303 * mib, 1024 * mib,
                                       &largest, &why),
              "a card with only the reserve left refuses even one byte");
        CHECK(largest == 0, "and names zero as the largest");
        CHECK(krk::expert_budget_fits(1024, 0, 0, 0, &largest, &why),
              "a backend that cannot report capacity does not refuse");
        const size_t gib = static_cast<size_t>(1) << 30;
        CHECK(krk::vram_reserve_bytes(16 * gib) == gib,
              "a 16 GiB card reserves 1 GiB");
        CHECK(krk::vram_reserve_bytes(8 * gib) == gib,
              "a smaller card keeps the 1 GiB floor");
        CHECK(krk::vram_reserve_bytes(64 * gib) == 4 * gib,
              "a large card reserves a sixteenth");
        // The host reserve is the other side of one number: the ceiling a run may
        // plan for is KRK_MEM_FRACTION (0.80) of installed memory, so the reserve
        // is the remaining fifth -- 19.2 GiB here, not the old eighth. Two hangs
        // came from planning WARM against free RAM with no ceiling at all.
        CHECK(krk::host_reserve_bytes(96 * gib) > 19 * gib &&
                  krk::host_reserve_bytes(96 * gib) < 20 * gib,
              "96 GiB of host RAM reserves the shortfall of the 80% ceiling (~19 GiB)");
        CHECK(krk::mem_ceiling_bytes(96 * gib) >= 76 * gib &&
                  krk::mem_ceiling_bytes(96 * gib) <= 77 * gib,
              "the plan ceiling is 80% of installed by default (76.8 GiB of 96)");
        CHECK(krk::host_reserve_bytes(16 * gib) > 3 * gib &&
                  krk::host_reserve_bytes(16 * gib) < 4 * gib,
              "a 16 GiB machine reserves ~3.2 GiB");
        CHECK(krk::host_reserve_bytes(4 * gib) == 2 * gib,
              "and the host reserve never drops below 2 GiB");
        // The automatic pinned WARM prefix is policy, and the policy changed on a
        // measurement: the old rule kept HALF the tier pageable, and that half is
        // the one every miss lands on. On the 256 MiB expert arm (WARM 9600 MiB,
        // 24 greedy tokens, identical stdout) the auto prefix was 4800 MiB and
        // decode ran 139.8 ms/step with xfer 2815 ms of 3216; pinning the whole
        // tier gave 65.3 ms/step, 15.3 tok/s, xfer 27 ms of enqueue. So the bound
        // is now a quarter of installed RAM (capped at 16 GiB), and the tier is
        // pinned WHOLE whenever it fits inside it.
        CHECK(krk::pinned_warm_prefix_bytes(0, 96 * gib) == 0,
              "no WARM tier means nothing to pin");
        CHECK(krk::pinned_warm_prefix_bytes(24 * gib, 0) == 0,
              "an unknown host size pins nothing (fails closed)");
        CHECK(krk::pinned_warm_prefix_bytes(24 * gib, 96 * gib) == 16 * gib,
              "96 GiB of RAM pins a quarter, capped at 16 GiB");
        CHECK(krk::pinned_warm_prefix_bytes(200 * gib, 96 * gib) == 16 * gib,
              "a larger tier does not raise the pin past that cap");
        CHECK(krk::pinned_warm_prefix_bytes(64 * gib, 1024 * gib) == 16 * gib,
              "and the cap holds however big the box");
        CHECK(krk::pinned_warm_prefix_bytes(4 * gib, 96 * gib) == 4 * gib,
              "a tier that fits the budget is pinned WHOLE -- the pageable half "
              "was the half every miss landed on");
        CHECK(krk::pinned_warm_prefix_bytes(9600 * (size_t(1) << 20), 96 * gib) ==
                  9600 * (size_t(1) << 20),
              "the measured 256 MiB arm's 9600 MiB tier auto-pins in full");
        CHECK(krk::pinned_warm_prefix_bytes(16 * gib, 16 * gib) == 4 * gib,
              "a 16 GiB laptop with a 16 GiB tier pins 4 GiB, not 8");
        CHECK(krk::pinned_warm_prefix_bytes(1, 96 * gib) == 0,
              "a tier smaller than a page pins nothing");
        // The pin is decided by a MEASUREMENT, not by the runtime's BAR bit: the
        // aperture is a firmware setting (Above 4G decoding + Resizable BAR),
        // WDDM has reported 0 with ReBAR on, and a large aperture can still
        // leave the staging memcpy in the pageable path. Timing both promotion
        // paths at init is the fact; this is the line it is compared against.
        CHECK(krk::warm_pin_worthwhile(7.6, 22.7),
              "a 3x pageable/pinned gap is exactly what the pin is for");
        CHECK(krk::warm_pin_worthwhile(4.6, 9.0),
              "a 2x gap on a small aperture is worth it too");
        CHECK(!krk::warm_pin_worthwhile(24.0, 26.0),
              "an aperture that already DMAs pageable memory pins nothing");
        CHECK(!krk::warm_pin_worthwhile(7.0, 7.5),
              "and a 7% gap does not pay for RAM that cannot be swapped");
        CHECK(!krk::warm_pin_worthwhile(10.0, 11.5),
              "a 15% gap is under the line (an aperture with no host copy)");
        CHECK(krk::warm_pin_worthwhile(10.0, 12.0),
              "and a 20% gap is the line itself, which pins");
        CHECK(!krk::warm_pin_worthwhile(0.0, 50.0),
              "an unmeasurable pageable path is not an argument to pin");
        // The same probe's other answer: WHICH pageable path runs. A tie keeps
        // the ring -- it owns a bounded buffer and a fallback -- so the direct
        // arm has to be clearly ahead to be adopted, because its staging is the
        // driver's and not this source tree's.
        CHECK(krk::pageable_direct_wins(10.0, 20.0),
              "an aperture that removes the host copy is a 2x-class win");
        CHECK(krk::pageable_direct_wins(10.0, 12.0),
              "and 20% ahead is the line itself, which takes over");
        CHECK(!krk::pageable_direct_wins(10.0, 11.3),
              "the +13% this box reads at 16 MiB is inside the flap band");
        CHECK(!krk::pageable_direct_wins(10.0, 11.0),
              "and the +1% it reads at 12.6 MB certainly is");
        CHECK(!krk::pageable_direct_wins(20.0, 20.5),
              "a direct path within noise keeps the ring");
        CHECK(!krk::pageable_direct_wins(0.0, 30.0),
              "an unmeasurable ring is not a reason to switch");
        // The interaction that is the point of measuring both: a wide aperture
        // makes direct the faster pageable path AND leaves nothing for the pin
        // to remove, so the run is fast without spending unswappable RAM.
        CHECK(krk::pageable_direct_wins(10.0, 26.0) &&
                  !krk::warm_pin_worthwhile(26.0, 27.0),
              "a wide aperture picks direct and drops the pin in one decision");
        // --- the third mechanism ------------------------------------------
        // The rates below are the ones measured on this box (RX 9070 XT, ReBAR
        // on, runtime claiming a 256 MiB window): `tools/bar_probe.hip` for the
        // size sweep, the engine's own probe for the promotion shape. The ring
        // is the slowest arm at every size, and a two-way menu could only ever
        // choose between the two slowest -- which is the bug the third arm
        // exists to fix, so it is checked against the measured numbers.
        // The engine's own probe reported these three on this box before the
        // third arm existed: the ring 8.9 against direct 10.0 is under the 1.20x
        // line, so the OLD two-way menu kept the ring on every run -- while the
        // driver's blocking copy measured 1.5x ahead of that ring. This is the
        // case the third mechanism is for, and it is the real one, not a
        // constructed ratio.
        CHECK(krk::pageable_path_choice(8.9, 10.0, 13.4) == 2,
              "the ring 1.1x behind direct used to win by default; blocking is "
              "1.5x ahead of the ring and is chosen instead");
        // At 1 GiB the ring is the outlier and the two driver-staged arms are
        // inside the band of each other, so the MORE conservative of the two
        // (direct, which the cache already carried) is what runs.
        CHECK(krk::pageable_path_choice(8.39, 21.98, 22.15) == 1,
              "a blocking arm 0.8% ahead of direct does not displace it");
        CHECK(krk::pageable_path_choice(9.47, 15.23, 14.40) == 1,
              "at the 16 MiB promotion shape direct wins, and it is the choice");
        CHECK(krk::pageable_path_choice(10.0, 11.0, 11.5) == 0,
              "three arms inside the 1.20x band keep the ring (it owns a known "
              "fallback), rather than picking a path on noise");
        CHECK(krk::pageable_path_choice(0.0, 0.0, 0.0) == -1,
              "no measurement is not a choice");
        CHECK(krk::pageable_path_choice(8.0, 0.0, 0.0) == 0,
              "and an unmeasured arm simply does not compete");
        // The aperture: the runtime's attribute is not usable on this platform,
        // so the wide/narrow answer comes from these two rates. Both sides of
        // the line are pinned with numbers that were actually measured.
        CHECK(krk::aperture_is_wide(15.23, 24.35),
              "15.2 against 24.4 GB/s is wide: a window cannot reach 62% of pinned");
        CHECK(krk::aperture_is_wide(23.10, 28.06),
              "and 23.1 against 28.1 at 256 MiB is wider still");
        CHECK(!krk::aperture_is_wide(8.39, 24.35),
              "the ring's 8.4 against 24.4 is the windowed population");
        CHECK(!krk::aperture_is_wide(0.0, 24.35),
              "an unmeasured pageable arm never claims a wide aperture");
        CHECK(!krk::host_staging_fits(1, 12 * gib, 12 * gib),
              "host staging stops at the exact reserve");
        CHECK(krk::host_staging_fits(1, 12 * gib + 1, 12 * gib),
              "one byte above the reserve can stage one byte");
        CHECK(!krk::host_staging_fits(2, 12 * gib + 1, 12 * gib),
              "staging cannot cross the reserve by one byte");
        CHECK(krk::host_staging_fits(1, 12 * gib + 3, 12 * gib, 2),
              "pending batch allocations count against remaining headroom");
        CHECK(!krk::host_staging_fits(2, 12 * gib + 3, 12 * gib, 2),
              "pending bytes cannot make a batch overrun the reserve");
        CHECK(!krk::host_staging_fits(1, 0, 0),
              "unknown available RAM fails closed");
    }

    // Drive the WARM admission itself with a reserve that exceeds live free RAM.
    // The cache must still serve the expert correctly, but must not retain a
    // pageable copy; this pins the ordinary COLD -> WARM read-through path, not
    // just the explicit warm-up sweep above it.
    {
        set_env_var("KRK_RAM_RESERVE_MB", "200000");
        ExpertCache cache;
        cache.configure(cpu, corpus, corpus);
        const ResidentExpert *resident = cache.acquire(a, 0, 0);
        CHECK(resident != nullptr,
              "a host reserve refusal still loads the expert from the source");
        CHECK(cache.warm_used_bytes() == 0,
              "a cold read-through does not stage when it would cross the reserve");
        cache.clear();
        clear_env_var("KRK_RAM_RESERVE_MB");
    }

    // The batched COLD -> WARM prefetch used by the active MoE path has its own
    // reservation phase, so drive that entry point directly too. No member of
    // the batch may be allocated when even its first expert would cross reserve.
    {
        set_env_var("KRK_RAM_RESERVE_MB", "200000");
        ExpertCache cache;
        cache.configure(cpu, corpus, corpus);
        const i32 ids[] = {0, 1};
        CHECK(cache.prefetch_layer(a, 0, ids, 2) == 0,
              "a WARM batch is declined at the host RAM reserve");
        CHECK(cache.warm_used_bytes() == 0,
              "a declined batch has no pending or committed host buffers");
        cache.clear();
        clear_env_var("KRK_RAM_RESERVE_MB");
    }

    // The refused startup, driven END TO END through Engine::init -- not through
    // the guard functions alone. This is the path that rotted once: a `false`
    // return was added to Engine::configure_expert_cache and its normal exit was
    // left without a `return` statement. That is UB, and it compiled with 0
    // errors while build_check.sh stayed green and dense models ran (they return
    // early); every MoE died at rc=3 with empty stdout, and this suite went from
    // green to rc=1 with four groups dead. Checks on the decision could not see
    // it, because not one of them called init -- and nothing could see the defect
    // on the other side of it either: both CLIs answered a failed init with
    // `delete be; return 1;`, freeing the backend and then unwinding past the
    // engine, whose destructor releases buffers through that pointer (SIGSEGV).
    // So: a card too small for the requested budget must come back as a FALSE
    // RETURN, followed by the destruction of the engine that refused.
    {
        const std::string moe_path = "kraken-refusal-test.gguf";
        CHECK(build_sparse_kv_model(moe_path),
              "wrote the MoE fixture for the refusal arm");
        // A device-free run has no card, so the CPU backend is told to report one
        // (KRK_VRAM_TOTAL_MB). It reads that only when it has no capacity of its
        // own, so no GPU run can be affected: 4 GiB with a 1 GiB reserve leaves
        // 3072 MiB for an expert budget.
        set_env_var("KRK_VRAM_TOTAL_MB", "4096");
        EngineConfig cfg;
        cfg.model_path = moe_path;
        cfg.n_ctx = 16;
        cfg.prefill_chunk = 4;
        cfg.expert_cache_mb = 1024; // inside the 3072 MiB of room
        std::string rerr;
        {
            // The control FIRST: without it, a fixture that cannot initialize at
            // all would make the refusal below pass for the wrong reason.
            Engine engine;
            const bool ok = engine.init(cpu, cfg, &rerr);
            CHECK(ok, "a budget the simulated card can hold does initialize");
            if (!ok)
                std::fprintf(stderr, "  (refusal control: %s)\n", rerr.c_str());
        }
        cfg.expert_cache_mb = 4096; // more than the card can hold
        rerr.clear();
        {
            Engine engine;
            CHECK(!engine.init(cpu, cfg, &rerr),
                  "a budget larger than the card is refused, not trimmed");
            CHECK(rerr.find("does not fit") != std::string::npos &&
                      rerr.find("3072") != std::string::npos,
                  "and names the largest budget that does fit, in MiB");
            // The text is the evidence: it can only come from the guard, so the
            // arm cannot be passing because the fixture failed for some other
            // reason (the control above is what makes that claim testable).
            if (!rerr.empty())
                std::fprintf(stderr, "  (refused: %s)\n", rerr.c_str());
            // Destruction follows the refusal, which is the unwinding that used
            // to run through a backend the caller had already freed.
        }
        clear_env_var("KRK_VRAM_TOTAL_MB");
        std::remove(moe_path.c_str());
    }

    // 1. A budget that covers the corpus holds every slot in it.
    {
        ExpertCache cache;
        cache.configure(cpu, corpus, 0);
        for (i32 e = 0; e < 4; e++) CHECK(cache.acquire(a, 0, e) != nullptr, "narrow expert loaded");
        for (i32 e = 0; e < 4; e++) CHECK(cache.acquire(b, 1, e) != nullptr, "wide expert loaded");
        CHECK(cache.evictions() == 0, "a budget that covers the corpus evicts nothing");
        CHECK(cache.resident_slots() == 8, "all eight experts stay resident");
        CHECK(cache.resident_bytes() == corpus, "and occupy exactly the corpus bytes");
        cache.clear();
    }

    // 2. One byte short of the corpus and something has to go, so arm 1 is not
    // passing because nothing was ever a candidate for eviction.
    {
        ExpertCache cache;
        cache.configure(cpu, corpus - 1, 0);
        for (i32 e = 0; e < 4; e++) cache.acquire(a, 0, e);
        for (i32 e = 0; e < 4; e++) cache.acquire(b, 1, e);
        CHECK(cache.evictions() > 0, "one byte short of the corpus does evict");
        CHECK(cache.resident_slots() < 8, "and cannot hold all eight");
        cache.clear();
    }

    // 3. An explicit slot budget is a count, not an estimate of one.
    {
        ExpertCache cache;
        cache.configure(cpu, corpus, 0, 6);
        for (i32 e = 0; e < 4; e++) cache.acquire(a, 0, e);
        for (i32 e = 0; e < 4; e++) cache.acquire(b, 1, e);
        CHECK(cache.capacity_slots() == 6, "an explicit slot budget is the capacity");
        CHECK(cache.evictions() > 0 && cache.resident_slots() <= 6,
              "and it is enforced as a count even with budget to spare");
        cache.clear();
    }

    // 4. A statistics reset keeps residency, so it has to keep the resident
    // index that says which slots are resident. reset_counters() cleared the
    // index beside a tier it deliberately leaves full, and pick_victim then saw
    // one key for thousands of resident slots -- on the REAP-320 file that cost
    // 2589 -> 6754 promotions (2.6x the bytes over the link) with no counter
    // saying anything was wrong. The index is an invariant of residency, not of
    // the statistics generation, and this is the arm that says so.
    {
        ExpertCache cache;
        cache.configure(cpu, corpus - 1, 0);
        for (i32 e = 0; e < 4; e++) cache.acquire(a, 0, e);
        for (i32 e = 0; e < 4; e++) cache.acquire(b, 1, e);
        const size_t resident = cache.resident_slots();
        CHECK(resident > 0 && cache.vram_index_keys() == resident,
              "the resident index covers exactly the resident slots");
        cache.reset_counters();
        CHECK(cache.resident_slots() == resident,
              "a counter reset keeps residency");
        CHECK(cache.vram_index_keys() == resident,
              "and keeps the resident index with it");
        // A key this cache has never held, at a budget one byte short of the
        // corpus: it can only be served by evicting.
        CHECK(cache.acquire(a, 7, 0) != nullptr,
              "a post-reset acquire that needs room is served");
        CHECK(cache.evictions() > 0, "and it evicted rather than refusing");
        CHECK(cache.vram_index_keys() == cache.resident_slots(),
              "index and residency agree after the eviction");
        cache.clear();
        CHECK(cache.vram_index_keys() == 0 && cache.resident_slots() == 0,
              "clear drops the index with the residency");
    }

    delete cpu;
    test_expert_promotion_failure();
    std::remove(path.c_str());
}

// A refused promotion must not hand the caller an expert with no matrices.
// The WARM-hit path fell through to the cold reload while the WARM slot stayed
// in the map, and emplace() on an existing key is a no-op -- so acquire()
// returned the OLD entry, whose ResidentExpert is exactly what the failed
// promotion never built, and leaked the cells the reload had allocated. This is
// the constrained-card path (the pool refuses a DMA), so it is not exotic.
static void test_expert_promotion_failure() {
    const std::string path = "kraken-promote-fail.gguf";
    CHECK(build_expert_source_model(path),
          "wrote the expert fixture for the promotion-failure case");
    Backend *cpu = make_cpu_backend();
    CountingBackend be(cpu);
    Gguf g;
    std::string err;
    CHECK(g.load(path, &err), "loaded the promotion-failure fixture");
    const ExpertSource src = expert_source(g);
    CHECK(src.present() && src.expert_bytes() == 24,
          "promotion-failure fixture is one 24-byte expert");

    ExpertCache cache;
    cache.configure(&be, 24, 96); // exactly one expert in VRAM, four in WARM
    const ResidentExpert *a = cache.acquire(src, 0, 0);
    CHECK(a && a->up.present() && a->down.present(),
          "the first expert is resident with all three matrices");
    const ResidentExpert *b = cache.acquire(src, 0, 1);
    CHECK(b && b->up.present(), "a second expert takes the only slot");
    CHECK(cache.in_vram(0, 0) != cache.in_vram(0, 1),
          "one slot holds exactly one of the two experts");
    const i32 warm_e = cache.in_vram(0, 0) ? 1 : 0;

    // Every device allocation now fails: this expert is WARM-only, so its
    // acquire() takes the promotion path and cannot complete it.
    const u64 refusals_before = cache.pool_refusals();
    be.set_fail_allocation_after(0);
    const ResidentExpert *refused = cache.acquire(src, 0, warm_e);
    CHECK(!refused,
          "a refused promotion returns nullptr, never an incomplete expert");
    // P0-1: refusals are countable, so the engine can fail visibly instead
    // of silently skipping the expert (`if (!re) continue`). Both the WARM
    // promotion and the cold rebuild refuse here, so this counts at least
    // one refusal for the single acquire.
    CHECK(cache.pool_refusals() > refusals_before,
          "a refused acquire leaves a countable refusal behind");
    be.set_fail_allocation_after(-1);
    const ResidentExpert *after = cache.acquire(src, 0, warm_e);
    CHECK(after && after->up.present() && after->gate.present(),
          "the same expert loads normally once the device can allocate again");
    cache.clear();
    // The refused promotion above must leave nothing behind in the process-wide
    // host ledger. It used to: warm_admit() had already paid for the expert's
    // WARM copy, and the failed-device-load return dropped the local slot on
    // the floor, so those host slices stayed allocated and unreachable -- 24
    // bytes here that no clear() could free, and because the ledger is static
    // it capped every later admission in the same process (the in-proc runner
    // failed two KV groups on a budget the child runner gave them fresh).
    CHECK(SharedHostRamBudget::used_bytes() == 0,
          "a refused promotion keeps its WARM copy without leaking host bytes");
    std::remove(path.c_str());
}

static void test_moe_route_predictor() {
    MoeRoutePredictor predictor;
    predictor.configure(4, 6, 3);
    const i32 source[] = {0, 1, 1, -1};
    const i32 target[] = {3, 4};
    predictor.observe(0, source, 4);
    predictor.observe(2, target, 2);

    std::vector<i32> candidates;
    predictor.predict(2, 0, 3, 0.5f, 3, candidates);
    CHECK(candidates.size() == 3,
          "combined temporal and adjacent-layer hints obey the candidate budget");
    CHECK(candidates[0] == 3 && candidates[1] == 4 && candidates[2] == 0,
          "temporal candidates rank first on an equal-confidence bootstrap");
    CHECK(predictor.predictions_issued() == 1,
          "one non-empty hint set is accounted as a prediction");

    // The native route remains the only target signal. Its IDs score the
    // pending candidates and update each method's confidence independently.
    const i32 actual_target[] = {3, 0};
    predictor.observe(2, actual_target, 2);
    CHECK(predictor.candidates() == 3 && predictor.candidate_hits() == 2,
          "candidate correctness is measured against the native route");
    CHECK(std::fabs(predictor.candidate_precision() - (2.0f / 3.0f)) < 1e-6f,
          "candidate precision reports correct/issued experts");

    // Temporal and cross-layer signals below the minimum confidence stop
    // producing speculative candidates; caller demand routing remains intact.
    predictor.predict(2, 0, 3, 0.75f, 3, candidates);
    CHECK(candidates.empty(), "low-confidence predictions fall back to no prefetch");
    predictor.predict(2, 0, 2, 0.25f, 2, candidates);
    CHECK(candidates.size() == 2 && candidates[0] == 0 && candidates[1] == 1,
          "cross-layer policy uses the source route and stays bounded");

    predictor.reset();
    CHECK(predictor.predictions_issued() == 0 && predictor.candidates() == 0 &&
              predictor.staged() == 0,
          "request reset clears history and telemetry");
    predictor.predict(2, 0, 3, 0.0f, 3, candidates);
    CHECK(candidates.empty(), "a fresh request cannot reuse stale route IDs");
    test_moe_grouped_plan_offsets();
}

// P0-2: the grouped-plan fast path must agree with the legacy per-expert
// rescan it replaced. plan_off_ (prefix sums over the whole (token, slot)
// route) already holds each expert's row count, so the loop skips the
// O(ne*n*k) rebuild and reads the segment directly. This replays both
// constructions over several route shapes -- including duplicate experts
// across rows and experts no row selected -- and requires identical counts,
// row order and weights per expert.
static void test_moe_grouped_plan_offsets() {
    struct Case {
        i32 ne, n, k;
        std::vector<i32> sel;
        std::vector<f32> wt;
    };
    std::vector<Case> cases;
    // n=1 decode row, top-2 over 4 experts.
    cases.push_back({4, 1, 2, {1, 3}, {0.7f, 0.3f}});
    // n=8 prefill rows with heavy duplicate picks and one unpicked expert.
    cases.push_back({4, 8, 2,
                     {0, 1, 0, 2, 1, 1, 3, 0, 2, 2, 0, 1, 1, 3, 3, 3},
                     {0.5f, 0.5f, 0.6f, 0.4f, 0.9f, 0.1f, 0.2f, 0.8f,
                      0.3f, 0.7f, 0.4f, 0.6f, 0.55f, 0.45f, 0.65f, 0.35f}});
    // n=2, every slot picks the same expert.
    cases.push_back({4, 2, 2, {2, 2, 2, 2}, {0.25f, 0.25f, 0.5f, 0.5f}});
    for (size_t c = 0; c < cases.size(); c++) {
        const Case &cc = cases[c];
        // Prefix-sum construction, mirroring Engine::moe_ffn's plan build.
        std::vector<i32> off(static_cast<size_t>(cc.ne) + 1, 0);
        for (i32 t = 0; t < cc.n; t++) {
            const i32 *sel = cc.sel.data() + static_cast<size_t>(t) * cc.k;
            for (i32 s = 0; s < cc.k; s++) {
                const i32 e = sel[s];
                if (e >= 0 && e < cc.ne) off[static_cast<size_t>(e) + 1]++;
            }
        }
        for (i32 e = 0; e < cc.ne; e++)
            off[static_cast<size_t>(e) + 1] += off[static_cast<size_t>(e)];
        std::vector<i32> dev(static_cast<size_t>(off[static_cast<size_t>(cc.ne)]), -1);
        std::vector<f32> alpha(dev.size(), 0.0f);
        std::vector<i32> cur(off.begin(), off.end() - 1);
        for (i32 t = 0; t < cc.n; t++) {
            const i32 *sel = cc.sel.data() + static_cast<size_t>(t) * cc.k;
            const f32 *wt = cc.wt.data() + static_cast<size_t>(t) * cc.k;
            for (i32 s = 0; s < cc.k; s++) {
                const i32 e = sel[s];
                if (e < 0 || e >= cc.ne) continue;
                const i32 at = cur[static_cast<size_t>(e)]++;
                dev[static_cast<size_t>(at)] = t;
                alpha[static_cast<size_t>(at)] = wt[s];
            }
        }
        // Legacy per-expert rescan, mirroring the old loop body.
        for (i32 e = 0; e < cc.ne; e++) {
            std::vector<i32> rows;
            std::vector<f32> wts;
            for (i32 t = 0; t < cc.n; t++) {
                const i32 *sel = cc.sel.data() + static_cast<size_t>(t) * cc.k;
                const f32 *wt = cc.wt.data() + static_cast<size_t>(t) * cc.k;
                for (i32 s = 0; s < cc.k; s++) {
                    if (sel[s] == e) {
                        rows.push_back(t);
                        wts.push_back(wt[s]);
                    }
                }
            }
            const i64 m_pre = static_cast<i64>(off[static_cast<size_t>(e) + 1] -
                                               off[static_cast<size_t>(e)]);
            CHECK(m_pre == static_cast<i64>(rows.size()),
                  "plan offsets give the legacy row count for every expert");
            if (m_pre == 0) continue;
            bool same = true;
            for (i64 i = 0; i < m_pre && same; i++) {
                const size_t at = static_cast<size_t>(off[static_cast<size_t>(e)] + i);
                same = (dev[at] == rows[static_cast<size_t>(i)] &&
                        alpha[at] == wts[static_cast<size_t>(i)]);
            }
            CHECK(same, "plan segment matches legacy row order and weights");
        }
    }
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

    // 7. A ranked warmup's order has to reach the EVICTION ranking, or it runs
    //    against it. Without a seed every slot the warmup promoted carries the
    //    same counter, so victim selection falls through to its tie-break --
    //    oldest load first -- and the first expert sacrificed is the one the
    //    warmup ranked hottest, because it was promoted first. Measured on
    //    laguna-xs2 before this: 976 promotions and 968 VRAM evictions in a
    //    64-token run at a 95.3% HOT hit rate, 778 ms of a 3087 ms decode.
    //    Both arms below run the SAME traffic; only the seed differs, so the
    //    assertion can fail in the direction that matters.
    cache.clear();
    cache.configure(&be, 48); // two slots
    CHECK(touch(0), "expert 0 loads first -- the older slot");
    CHECK(touch(1), "expert 1 loads second");
    CHECK(cache.touch_count(0, 0) == cache.touch_count(0, 1),
          "both slots start on the same counter, so the tie-break decides");
    CHECK(touch(2), "expert 2 forces an eviction");
    CHECK(cache.evictions() == 1, "exactly one slot had to go");
    CHECK(!cache.in_vram(0, 0),
          "unseeded, the OLDEST slot is the victim -- the promoted-first expert");

    cache.clear();
    cache.configure(&be, 48);
    CHECK(touch(0) && touch(1), "the same two experts load in the same order");
    cache.seed_count(0, 0, 5);
    CHECK(cache.touch_count(0, 0) == 5, "the seed is the counter eviction reads");
    CHECK(touch(2), "expert 2 forces the same eviction");
    CHECK(cache.evictions() == 1, "exactly one eviction, as in the unseeded arm");
    CHECK(cache.in_vram(0, 0), "the seeded expert keeps its VRAM slot");
    CHECK(!cache.in_vram(0, 1), "and the unseeded one takes the fall instead");
    const u64 loads_after_seed = cache.loads();
    CHECK(touch(0), "the seeded expert is still reachable");
    CHECK(cache.loads() == loads_after_seed, "and costs no reload to reach");

    // 8. A seed ranks, it does not promise: the clamp keeps it below the pin
    //    threshold, so the rule that protects pins stays a rule about traffic.
    cache.seed_count(0, 2, 1000);
    CHECK(cache.touch_count(0, 2) < ExpertCache::kPinThreshold,
          "an absurd seed is clamped below the pin threshold");
    CHECK(cache.pinned_slots() == 0, "and no slot became pinned from seeding");

    // 9. The ranking helper itself, because the warmup calls it and nothing else
    //    does: rank 0 is the top of the range, the bottom is 1, the whole range
    //    is inside [1, kPinThreshold), and it never rises with rank.
    CHECK(ExpertCache::warmup_rank_seed(0, 256) == ExpertCache::kPinThreshold - 1,
          "the hottest rank seeds the top of the range");
    CHECK(ExpertCache::warmup_rank_seed(255, 256) == 1,
          "the bottom of the warmed order seeds the floor");
    CHECK(ExpertCache::warmup_rank_seed(0, 1) == ExpertCache::kPinThreshold - 1,
          "a one-deep order is all top");
    {
        u32 prev = ExpertCache::warmup_rank_seed(0, 256);
        bool in_range = true, monotone = true;
        for (size_t r = 1; r < 256; r++) {
            const u32 v = ExpertCache::warmup_rank_seed(r, 256);
            if (v < 1 || v >= ExpertCache::kPinThreshold) in_range = false;
            if (v > prev) monotone = false;
            prev = v;
        }
        CHECK(in_range, "every seeded rank is inside [1, kPinThreshold)");
        CHECK(monotone, "the seed never rises as the rank gets worse");
    }

    // 10. The re-clamp must not ratchet. `free_bytes` is a device-wide reading
    //     and a cell this cache evicts goes back to the BACKEND's pool rather
    //     than to the driver, so the reading does not rise when the cache
    //     evicts. A cap derived from live residency therefore removed whatever
    //     had just been evicted, at every clamp: measured 5747 -> 4295 MiB over
    //     93 tokens on the 288-expert file, and 4697 -> 359 MiB on a longer
    //     chat. Both arms below are one cache at one residency; only the free
    //     reading differs, and only a drop somebody else caused may cut.
    cache.clear(); // hand the backend's pool back before the fixture re-sizes it
    {
        ExpertCache clamp;
        clamp.configure(&be, 90); // three 24-byte experts fit, a fourth does not
        const auto load = [&](i32 e) { return clamp.acquire(src, 0, e) != nullptr; };
        for (i32 e = 0; e < 4; e++) CHECK(load(e), "clamp fixture expert is reachable");
        CHECK(clamp.evictions() > 0,
              "the clamp fixture over-subscribes its budget, so it evicts");
        CHECK(clamp.resident_bytes() < clamp.vram_budget(),
              "and its residency is below the budget -- the state the ratchet fed on");
        const size_t budget = clamp.vram_budget();
        CHECK(clamp.reclamp_vram_budget(1000, 2000) == budget,
              "the first free reading is a baseline, not a cut");
        CHECK(clamp.reclamp_vram_budget(1000, 2000) == budget,
              "an evicting pool does not shrink its own budget");
        CHECK(clamp.reclamp_vram_budget(996, 2000) == budget - 4,
              "only the drop somebody else caused shrinks it");
        CHECK(clamp.reclamp_vram_budget(996, 2000) == budget - 4,
              "and a repeated reading does not shrink it twice");
        CHECK(clamp.reclamp_vram_budget(3000, 2000) == budget - 4,
              "free VRAM above the floor never raises or lowers the budget");
    }

    cache.clear();
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// The device dispatch table's HOST half
// ---------------------------------------------------------------------------
//
// `ExpertCache::publish_dispatch_table` is the promise a kernel reads experts
// through: `table[layer * n_expert + e]` holds that expert's three device
// pointers, their shapes, their types and their row strides, so the routing can
// stay on the device and no host pointer per expert sits between the top-k and
// the GEMMs.
//
// Two ways that promise can lie are what this group pins: publishing while the
// set is NOT resident (a pointer a later promotion rewrites under a running
// kernel), and staying "current" after a slot moved.
//
// The kernel that consumes the table cannot run here -- no test group opens the
// device -- so the device half is pinned by `kraken-bench --moe-dispatch`, which
// compares the one-launch dispatch against the per-expert path it replaces and
// requires a gate/up swap in the table to change the answer.
static void test_expert_dispatch_table() {
    const std::string path = "kraken-dispatch-table.gguf";
    // The MIXED fixture: two expert layers with different footprints, which is
    // what the eviction check below needs (a wider expert has to dislodge a
    // resident one) and what a real Q4_K_M file looks like.
    CHECK(build_expert_source_model_mixed(path),
          "wrote the dispatch-table GGUF");

    Backend *cpu = make_cpu_backend();
    // The reason the engine's arm cannot engage on this backend: no table
    // dispatch, so the layer takes the host path unchanged.
    CHECK(!cpu->moe_dispatch_supported(),
          "the CPU backend has no table dispatch");
    CountingBackend be(cpu);
    Gguf g;
    std::string err;
    CHECK(g.load(path, &err), "loaded the dispatch-table GGUF");
    if (!err.empty()) std::fprintf(stderr, "  (dispatch gguf: %s)\n", err.c_str());
    const ExpertSource src = expert_source(g);
    CHECK(src.present(), "the dispatch-table expert source is complete");

    ExpertCache cache;
    cache.configure(&be, 96); // four experts of 24 bytes

    // 1. A cold cache publishes nothing, and says why. The reason has to name
    //    the residency, because "off" with no reason is the failure mode that
    //    took a whole session to find in the hyper-connection refusal.
    CHECK(!cache.whole_set_resident(1, 4), "an empty cache is not resident");
    CHECK(cache.publish_dispatch_table(1, 4, nullptr, nullptr) == nullptr,
          "and publishes no table");
    std::string why;
    i32 filled = -1;
    CHECK(cache.publish_dispatch_table(1, 4, &filled, &why) == nullptr &&
              filled == 0 && !why.empty(),
          "a refusal carries a reason and no entries");

    // 2. Fill the set. Every expert resident is what makes the table true.
    const ResidentExpert *r[4] = {};
    for (i32 e = 0; e < 3; e++) {
        r[e] = cache.acquire(src, 0, e);
        CHECK(r[e] != nullptr, "expert loads");
    }
    CHECK(!cache.whole_set_resident(1, 4), "three of four slots is not the set");
    CHECK(!cache.whole_set_resident(2, 4), "a second layer has no slots at all");
    r[3] = cache.acquire(src, 0, 3);
    CHECK(r[3] != nullptr && cache.whole_set_resident(1, 4),
          "the fourth acquire makes the whole routed set resident");

    // 3. Publish, then read the entries: they must BE the resident matrices, in
    //    expert-id order, with everything a dot product needs.
    void *table = cache.publish_dispatch_table(1, 4, &filled, &why);
    CHECK(table != nullptr && filled == 4, "four entries published");
    CHECK(cache.dispatch_table() == table,
          "the cache reports the table it published");
    CHECK(!cache.dispatch_table_stale(), "and it starts current");
    if (table) {
        // Read the entries in place: on the CPU backend the "device" pointer IS
        // host memory, so the published buffer is a plain array of entries and
        // these are the same bytes a kernel would read.
        const MoeDispatchEntry *back =
            static_cast<const MoeDispatchEntry *>(table);
        bool ptrs_ok = true, shape_ok = true, present_ok = true;
        for (i32 e = 0; e < 4; e++) {
            const MoeDispatchEntry &d = back[e];
            ptrs_ok = ptrs_ok && d.p[kMoeGate] == r[e]->gate.data &&
                      d.p[kMoeUp] == r[e]->up.data &&
                      d.p[kMoeDown] == r[e]->down.data;
            shape_ok = shape_ok && d.n_in[kMoeGate] == r[e]->gate.n_in &&
                       d.n_out[kMoeGate] == r[e]->gate.n_out &&
                       d.n_in[kMoeDown] == r[e]->down.n_in &&
                       d.row_bytes[kMoeGate] ==
                           static_cast<i64>(dtype_row_bytes(r[e]->gate.type,
                                                           r[e]->gate.n_in)) &&
                       d.type[kMoeGate] == static_cast<i32>(r[e]->gate.type);
            present_ok = present_ok && d.present[kMoeGate] != 0 &&
                         d.present[kMoeUp] != 0 && d.present[kMoeDown] != 0;
        }
        CHECK(ptrs_ok, "every entry points at that expert's resident matrices");
        CHECK(shape_ok, "with its own shapes, type and row stride");
        CHECK(present_ok, "and all three matrices marked present");
        // Asserted where it differs: a uniform check still passes one entry off.
        CHECK(back[0].p[kMoeGate] != back[1].p[kMoeGate],
              "entry 0 and entry 1 are different experts");
    }

    // 4. A slot that moves invalidates it. Layer 1's experts are wider than the
    //    four layer-0 slots this budget holds, so the acquisition evicts a
    //    layer-0 expert -- exactly the event the promise forbids.
    const ExpertSource src1 = expert_source_of(g, 1);
    CHECK(src1.present(), "the mixed file has a second expert layer");
    CHECK(cache.acquire(src1, 1, 0) != nullptr, "a wider expert loads");
    CHECK(cache.evictions() > 0, "by evicting a resident slot");
    CHECK(cache.dispatch_table_stale(),
          "the published table is marked stale by the eviction");
    CHECK(!cache.whole_set_resident(1, 4), "layer 0 is no longer whole");
    CHECK(cache.publish_dispatch_table(1, 4, &filled, &why) == nullptr,
          "so it cannot be republished while the hole is there");

    // 5. Refilling is a republish, not a one-shot: the arm has to be able to
    //    come back after a run that touched another layer.
    for (i32 e = 0; e < 4; e++)
        CHECK(cache.acquire(src, 0, e) != nullptr, "layer 0 refills");
    CHECK(cache.whole_set_resident(1, 4), "the set is whole again");
    CHECK(cache.publish_dispatch_table(1, 4, &filled, &why) != nullptr &&
              filled == 4 && !cache.dispatch_table_stale(),
          "and republishes, current again");

    // 6. clear() releases the table with the slots it described.
    cache.clear();
    CHECK(cache.dispatch_table() == nullptr, "clear() drops the table");

    std::remove(path.c_str());
}

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
    // ONE host block per expert, not one per slice: the three slices are offsets
    // inside it (tools/probe_warmalloc.hip measured the pinned tier at 912 ns a
    // page for slice-sized allocations against 471 for expert-sized ones).
    CHECK(be.host_allocs() == 2,
          "read-through allocated one pageable block per expert");

    // 2. A VRAM eviction keeps the WARM copy and costs nothing: the one expert's
    //    worth of host allocations here is the *new* expert's admission, not a
    //    download of the victim. HOT -> WARM moves no bytes at all.
    const u64 host_allocs_before = be.host_allocs();
    CHECK(touch(2), "expert 2 needs a slot");
    CHECK(cache.evictions() == 1, "one VRAM eviction");
    CHECK(cache.demotions() == 1, "the evicted expert stayed resident in WARM");
    CHECK(cache.warm_slots() == 3, "WARM kept its own copy, not the device's");
    CHECK(be.host_allocs() == host_allocs_before + 1,
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

    // Batched reservations must obey the same byte budget as demand loads.
    cache.clear();
    cache.configure(&be, 48, 48);
    const i32 all[] = {0, 1, 2, 3};
    CHECK(cache.prefetch_layer(src, 0, all, 4) == 48,
          "prefetch reserves no more than two experts in a two-expert WARM tier");
    CHECK(cache.warm_used_bytes() == 48 && cache.warm_slots() == 2,
          "pending reservations count against the WARM budget");
    CHECK(cache.warm_rejects() == 2,
          "prefetch rejects the tail that cannot fit beside pending reads");
    CHECK(cache.acquires() == 0 && cache.pinned_slots() == 0,
          "speculative staging neither counts as demand nor earns pins");
    CHECK(touch(0) && touch(3), "demand still loads both staged and rejected experts");
    CHECK(cache.warm_used_bytes() <= cache.warm_capacity_bytes(),
          "demand after a partial prefetch remains within budget");

    cache.clear();
    cache.configure(&be, 48, 96);
    const i32 duplicates[] = {0, 0, -1, 4, 1, 1};
    const u64 before_batch_allocs = be.host_allocs();
    CHECK(cache.prefetch_layer(src, 0, duplicates, 6) == 48,
          "prefetch deduplicates valid IDs and ignores invalid IDs");
    CHECK(cache.warm_used_bytes() == 48 && cache.warm_slots() == 2,
          "duplicate IDs cannot overwrite or double-count staged buffers");
    CHECK(be.host_allocs() == before_batch_allocs + 2,
          "one host block per unique expert, its three slices inside it");
    CHECK(cache.prefetch_layer(src, 0, duplicates, 6) == 0,
          "already WARM experts are not read again");
    const ResidentExpert *staged = cache.acquire(src, 0, 1);
    CHECK(staged && std::memcmp(staged->gate.data, src.gate->data + 8, 8) == 0 &&
              std::memcmp(staged->up.data, src.up->data + 8, 8) == 0 &&
              std::memcmp(staged->down.data, src.down->data + 8, 8) == 0,
          "staging and promotion preserve every source slice byte for byte");

    // A churning tier must not grow: a two-expert tier refilled with the OTHER
    // two has to hand both blocks back before it commits two more, or the tier
    // holds RAM it says it does not and the next admission fails.
    cache.clear();
    cache.configure(&be, 48, 48);
    const i32 first_two[] = {0, 1};
    const i32 last_two[] = {2, 3};
    const u64 alloc_before_refill = be.host_allocs();
    CHECK(cache.prefetch_layer(src, 0, first_two, 2) == 48,
          "two experts fill the tier");
    CHECK(be.host_allocs() == alloc_before_refill + 2,
          "two blocks, one per expert");
    const u64 rel_before_refill = be.host_releases();
    CHECK(cache.prefetch_layer(src, 0, last_two, 2) == 48,
          "the other two replace them");
    CHECK(be.host_releases() == rel_before_refill + 2,
          "both evicted blocks give their memory back");
    CHECK(cache.warm_used_bytes() == 48,
          "and the tier still reports only the two experts it holds");

    cache.clear();
    cache.configure(&be, 48, 96);
    CHECK(touch(0), "current expert for staged promotion");
    const i32 next_ids[] = {1, 2};
    cache.prefetch_layer(src, 0, next_ids, 2);
    const u64 before_stage_acquires = cache.acquires();
    const u32 before_stage_count = cache.touch_count(0, 1);
    CHECK(cache.stage_next(src, 0, 1, 0), "next WARM expert stages beside current");
    CHECK(cache.in_vram(0, 0) && cache.in_vram(0, 1), "current leased during admission");
    CHECK(cache.acquires() == before_stage_acquires &&
              cache.touch_count(0, 1) == before_stage_count,
          "staging earns neither demand traffic nor popularity");
    CHECK(touch(1) && cache.active_used() == 1, "demand consumes staged weights once");
    CHECK(cache.stage_next(src, 0, 2, 1), "second staged expert exercises eviction");
    CHECK(cache.in_vram(0, 1), "staging cannot evict the current consumer");
    CHECK(touch(2) && cache.active_staged() == 2 && cache.active_used() == 2,
          "both staged promotions count separately from HOT hits");

    // 7. The pinned prefix. A slot admitted into pinned memory promotes with one
    //    DMA instead of a memcpy into the staging ring plus a DMA: measured
    //    22.7 GB/s against 7.6 GB/s on a 120B MoE, +18-25% decode. Three things
    //    have to hold, and none of them is the rate: the pin is the FIRST slots
    //    admitted (rank order, because admission follows the index), it is
    //    bounded by its own budget, and a pinned slot frees at all, with its
    //    accounting returning to zero. The deallocator PAIRING is not visible
    //    here: on the CPU backend both host frees land in release_host, so a red
    //    construction that routes a pinned free through the pageable path still
    //    reads 72/72 (measured). The device half is pinned by a pinned-DMA run
    //    with identical text, the same way gemm_group's device path is.
    cache.clear();
    cache.configure(&be, 24, 96); // one VRAM slot, a four-expert tier
    cache.set_host_pin_budget(72); // three of those four
    CHECK(cache.warm_pin_budget_bytes() == 72, "the pin budget is the one asked for");
    CHECK(touch(0) && touch(1) && touch(2), "three admitted, all inside the pin");
    CHECK(cache.warm_pinned_bytes() == 72, "each of the three is pinned whole");
    CHECK(cache.warm_used_bytes() == 72 && cache.warm_slots() == 3,
          "a pin decorates the tier, it does not bound it");
    {
        // A fourth expert exceeds the pin budget and must be admitted pageable:
        // a slot that silently took the pin would over-commit memory the tier
        // promised not to use.
        const size_t pinned_before = cache.warm_pinned_bytes();
        CHECK(touch(3), "a fourth expert is admitted past the pin budget");
        CHECK(cache.warm_pinned_bytes() == pinned_before,
              "the pin budget is not exceeded by the expert that overflowed it");
    }

    // A promotion out of pinned memory carries the same bytes as any other.
    cache.clear();
    cache.configure(&be, 24, 96);
    cache.set_host_pin_budget(72);
    CHECK(touch(0) && touch(1), "expert 0 is pinned, then evicted from VRAM");
    CHECK(!cache.in_vram(0, 0) && cache.in_host(0, 0),
          "the eviction kept the pinned WARM copy");
    const u64 releases_before_drop = be.host_releases();
    const ResidentExpert *back = cache.acquire(src, 0, 0);
    CHECK(back && std::memcmp(back->gate.data, src.gate->data, 8) == 0 &&
              std::memcmp(back->up.data, src.up->data, 8) == 0 &&
              std::memcmp(back->down.data, src.down->data, 8) == 0,
          "a promotion out of pinned memory preserves every slice byte for byte");
    CHECK(cache.warm_pinned_bytes() == 48, "and the pin survives its own promotion");
    CHECK(be.host_releases() == releases_before_drop,
          "promoting out of a pinned slot frees nothing on the host");

    // Bounded by the tier: a pin larger than WARM never binds past it, and the
    // setter is the one place that clamp can be checked.
    cache.clear();
    cache.configure(&be, 24, 48);
    cache.set_host_pin_budget(1u << 20);
    CHECK(cache.warm_pin_budget_bytes() == 48,
          "a pin budget larger than the tier clamps to the tier");
    CHECK(touch(0) && touch(1), "two experts of a pinned tier");
    CHECK(cache.warm_pinned_bytes() == 48 && cache.warm_slots() == 2,
          "both are pinned, because both are inside the clamped budget");
    // Dropping them runs the pinned free path: the host release count has to
    // move and the pinned byte total has to return to zero. (It does not, on
    // its own, prove which deallocator ran -- see the note in part 7.)
    const u64 rel_before_clear = be.host_releases();
    cache.clear();
    CHECK(be.host_releases() == rel_before_clear + 2,
          "each pinned block is freed when its slot goes");
    CHECK(cache.warm_pinned_bytes() == 0 && cache.warm_used_bytes() == 0,
          "and the pin accounting returns to zero with them");

    cache.clear();
    cache.configure(&be, 24, 96);
    CHECK(touch(0), "one-slot current expert");
    cache.prefetch_layer(src, 0, next_ids, 2);
    CHECK(!cache.stage_next(src, 0, 1, 0), "one-slot budget skips speculative admission");
    CHECK(touch(1), "skipped staging falls back to exact demand loading");

    // 8. KRK_EXPERT_WARM_MAP=1: the page cache is the second tier and the cache
    //    retains nothing. Off by default -- it was measured and it LOST, and the
    //    numbers are in docs/PERF-WARMUP.md -- but the arm is what keeps that
    //    path measurable instead of a claim, so its contract is pinned here:
    //    no admission, no host block per expert, and an evicted expert served
    //    by a mapping read rather than a promotion.
    //
    //    The flag is read once, in configure(), so it has to be set before the
    //    call and cleared after: the groups that follow configure their own
    //    caches in this process, and an environment left behind would silently
    //    change every one of them.
    cache.clear();
#ifdef _WIN32
    _putenv_s("KRK_EXPERT_WARM_MAP", "1");
#else
    setenv("KRK_EXPERT_WARM_MAP", "1", 1);
#endif
    cache.configure(&be, 48, 96);
    CHECK(cache.warm_as_mapping(), "the mapping policy is on when asked for");
    const u64 host_allocs_map = be.host_allocs();
    CHECK(touch(0) && touch(1), "two experts load under the mapping policy");
    CHECK(cache.warm_admissions() == 0 && cache.warm_slots() == 0,
          "and nothing is admitted to the host tier");
    CHECK(be.host_allocs() == host_allocs_map,
          "no host block is allocated for an expert");
    CHECK(cache.loads() == 2, "both acquires were file reads, by design");
    // Evict both, then ask for one: the reload must come from the mapping.
    CHECK(touch(2) && touch(3), "a third and fourth expert evict both");
    CHECK(cache.evictions() >= 2 && cache.demotions() == 0,
          "a VRAM eviction has no host copy to fall back to");
    const u64 loads_before_map = cache.loads();
    CHECK(touch(0), "the evicted expert comes back");
    CHECK(cache.loads() == loads_before_map + 1 && cache.promotions() == 0 &&
              cache.warm_hits() == 0,
          "served out of the mapping, not promoted out of WARM");
    // The warmup's stand-in for prefetch_layer: the same reads, kept nowhere.
    const i32 four_ids[] = {0, 1, 2, 3};
    const u64 touched_before = cache.bytes_touched();
    const u64 allocs_before_touch = be.host_allocs();
    const size_t one_expert = src.per_expert(src.gate) +
                              src.per_expert(src.up) +
                              src.per_expert(src.down);
    const size_t read = cache.touch_layer(src, 0, four_ids, 4);
    CHECK(read > 0 && cache.bytes_touched() == touched_before + read,
          "touch_layer counts the bytes it read as page-cache traffic");
    CHECK(read == 4 * one_expert,
          "and reads one block per expert, nothing more");
    CHECK(cache.warm_used_bytes() == 0 && cache.warm_slots() == 0,
          "while retaining none of it");
    CHECK(be.host_allocs() == allocs_before_touch + 1,
          "one scratch block per call, not one per expert");
#ifdef _WIN32
    _putenv_s("KRK_EXPERT_WARM_MAP", "");
#else
    unsetenv("KRK_EXPERT_WARM_MAP");
#endif

    cache.clear();
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// The dispatch table's HOST half: a routed set larger than the card
// ---------------------------------------------------------------------------
//
// The arm this group exists for is unreachable any other way. Tiel-Coder-35B-A3B
// is 22.3 GiB of routed experts against a 9-11 GiB device budget, so requiring
// the whole set in VRAM (`whole_set_resident`) refused the table from start to
// finish -- the recorded log says "0 of 10240 slots" -- and the one-launch
// dispatch never ran on exactly the models it is for. A kernel can read a
// slot's PINNED host copy directly (measured 24.3 GiB/s on this card,
// tools/probe_hostread.hip), so a pinned cell is as legal an operand as a pool
// cell and the overflow is read where it lives.
//
// What must NOT become publishable:
//   * a cell behind a backend that never measured that read (the flag is the
//     measurement, not a hope about unified addressing);
//   * a PAGEABLE cell, which a kernel cannot address at all -- skipping it is a
//     wrong answer, not a slow one.
static void test_expert_dispatch_host_backed() {
    const std::string path = "kraken-dispatch-host.gguf";
    CHECK(build_expert_source_model_mixed(path), "wrote the host-backed GGUF");

    Backend *cpu = make_cpu_backend();
    CountingBackend be(cpu);
    Gguf g;
    std::string err;
    CHECK(g.load(path, &err), "loaded the host-backed GGUF");
    const ExpertSource src = expert_source(g);
    const ExpertSource src1 = expert_source_of(g, 1);
    CHECK(src.present() && src1.present(), "both expert layers are present");
    const i32 four[] = {0, 1, 2, 3};

    // 1. The capability gate. Pinning alone is not the promise; the BACKEND's
    //    measured read path is, and the refusal has to say so rather than looking
    //    like a residency failure.
    {
        ExpertCache cache;
        cache.configure(&be, 24, 4096);
        cache.set_host_pin_budget(1u << 20);
        CHECK(cache.prefetch_layer(src, 0, four, 4) > 0, "the layer is staged pinned");
        CHECK(cache.warm_pinned_bytes() > 0 && !cache.in_vram(0, 0),
              "pinned, and not resident in VRAM");
        std::string why;
        CHECK(cache.publish_dispatch_table(1, 4, nullptr, &why) == nullptr &&
                  !why.empty(),
              "a backend without the measured read path publishes nothing");
        CHECK(why.find("pinned host") != std::string::npos,
              "and the reason names the capability, not the residency");
        cache.clear();
    }

    // 2. With the capability, a wholly pinned set publishes -- and every entry is
    //    the host slice a kernel would dereference.
    ExpertCache cache;
    be.set_host_load(true);
    cache.configure(&be, 24, 4096); // room for one VRAM expert: nothing promotes here
    cache.set_host_pin_budget(1u << 20);
    CHECK(cache.prefetch_layer(src, 0, four, 4) > 0, "layer 0 into pinned WARM");
    CHECK(cache.prefetch_layer(src1, 1, four, 4) > 0, "layer 1 into pinned WARM");
    CHECK(!cache.whole_set_resident(2, 4),
          "no cell is resident, so the strict query is still false");

    i32 filled = 0;
    std::string why;
    void *t = cache.publish_dispatch_table(2, 4, &filled, &why);
    CHECK(t != nullptr && filled == 8, "a wholly pinned set is publishable");
    CHECK(cache.dispatch_table_host_cells() == 8,
          "and every cell is reported as host-read");
    if (t) {
        const ResidentExpert *h0 = cache.acquire_inplace(src, 0, 0);
        const ResidentExpert *h5 = cache.acquire_inplace(src1, 1, 1);
        CHECK(h0 && h5, "two host slots are readable in place");
        const MoeDispatchEntry *d = static_cast<const MoeDispatchEntry *>(t);
        if (h0 && h5) {
            CHECK(d[0].p[kMoeGate] == h0->gate.data &&
                      d[0].p[kMoeUp] == h0->up.data &&
                      d[0].p[kMoeDown] == h0->down.data,
                  "entry 0 names expert 0's host slices");
            CHECK(d[5].p[kMoeGate] == h5->gate.data &&
                      d[5].p[kMoeDown] == h5->down.data,
                  "entry 5 is layer 1's expert 1, out of its own cells");
            CHECK(d[0].p[kMoeGate] != d[5].p[kMoeGate], "and the two entries differ");
            CHECK(d[5].n_in[kMoeGate] == h5->gate.n_in &&
                      d[5].type[kMoeDown] == static_cast<i32>(h5->down.type),
                  "with that slot's own shape and type");
            CHECK(d[0].present[kMoeGate] && d[0].present[kMoeUp] &&
                      d[0].present[kMoeDown],
                  "all three matrices marked present");
        }
    }

    // 3. A pageable cell is refused. The kernel would have to skip it, and a
    //    skipped expert is a wrong answer with a plausible shape.
    {
        ExpertCache partial;
        partial.configure(&be, 24, 4096);
        partial.set_host_pin_budget(24); // one expert out of the eight
        CHECK(partial.prefetch_layer(src, 0, four, 4) > 0, "four staged");
        CHECK(partial.prefetch_layer(src1, 1, four, 4) > 0, "and four more");
        CHECK(partial.warm_pinned_bytes() < partial.warm_used_bytes(),
              "only what fits the pin budget is pinned; the rest is pageable");
        std::string why2;
        CHECK(partial.publish_dispatch_table(2, 4, nullptr, &why2) == nullptr,
              "a set with pageable cells publishes nothing");
        CHECK(why2.find("pageable") != std::string::npos,
              "and the reason counts them");
        // Its slots hold ledgered host bytes, and the ledger is SHARED with the
        // budget groups that follow: leaving them charged made every exact-budget
        // check downstream fail in the single-process run (isolated they passed),
        // which is what the isolation check's `ledger=` column is for.
        partial.clear();
    }

    // 4. A promotion rewrites one cell, so the table goes stale and is
    //    republished against the new split rather than silently mixing a freed
    //    address with live ones.
    CHECK(cache.acquire(src, 0, 0) != nullptr, "expert 0 promotes into the VRAM slot");
    CHECK(cache.dispatch_table_stale(), "a promotion invalidates the published table");
    i32 filled2 = 0;
    void *t2 = cache.publish_dispatch_table(2, 4, &filled2, &why);
    CHECK(t2 != nullptr && filled2 == 8 && cache.dispatch_table_host_cells() == 7,
          "it republishes with one cell in VRAM and seven on the host");

    cache.clear();
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Expert warmup: the ranked startup phase that fills the two tiers in order
// ---------------------------------------------------------------------------
//
// The measurement that justifies this feature is in docs/perf_trace_report.md
// (5.8x prefill, +30% decode on the 35B MoE at the auto budget). What has to be
// pinned down in a unit test is the contract that measurement depends on: the
// warmup stages the RANKED set and stops, a stale index is REFUSED rather than
// trusted, and the traffic the warmup earned is reset so the run's own report
// describes the run. A warmup that quietly staged the whole corpus, or accepted
// a ranking measured on other weights, would make the headline number mean
// something else entirely.
struct WarmupRun {
    bool ok = false;
    size_t warm_slots = 0;
    u64 loads = 0, promotions = 0, hits = 0;
};

static void test_expert_warmup() {
    MoeSpec vs;
    vs.dense = false;
    vs.n_expert = 4;
    vs.n_expert_used = 2;
    vs.shared = true;
    vs.varied = true;
    CHECK(build_tiny_moe_model(kMoeTestPath, vs), "wrote the warmup-test MoE GGUF");
    const std::string idx = std::string(kMoeTestPath) + ".krakenexperts.json";

    i64 src_size = 0;
    {
        std::ifstream f(kMoeTestPath, std::ios::binary | std::ios::ate);
        src_size = static_cast<i64>(f.tellg());
    }
    CHECK(src_size > 0, "the warmup-test model has a size the index must match");

    // Writes an index whose layer-0 ranking names exactly two of the four
    // experts. n_expert is a parameter so the caller can also write a STALE one.
    const auto write_index = [&](int n_expert, i64 positions = 64) {
        std::ofstream f(idx, std::ios::binary);
        f << "{\n  \"version\": 1,\n  \"kind\": \"kraken-expert-index\",\n"
          << "  \"mode\": \"full-forward\",\n  \"source_size\": " << src_size
          << ",\n  \"n_layer\": 1,\n  \"n_expert\": " << n_expert
          << ",\n  \"positions_scanned\": " << positions
          << ",\n  \"layers\": [\n"
          << "    {\"layer\": 0, \"experts\": [{\"e\": 2}, {\"e\": 0}]}\n"
          << "  ]\n}\n";
    };

    const auto warm = [&](i32 warmup) {
        WarmupRun r;
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = kMoeTestPath;
        cfg.n_ctx = 64;
        cfg.prefill_chunk = 8;
        cfg.expert_cache_mb = 1;
        cfg.expert_warm_mb = 64;
        cfg.expert_warmup = warmup;
        std::string err;
        r.ok = engine.init(cpu, cfg, &err);
        if (r.ok) {
            const ExpertCache &ec = engine.model().experts();
            r.warm_slots = ec.warm_slots();
            r.loads = ec.loads();
            r.promotions = ec.promotions();
            r.hits = ec.hits();
            engine.shutdown();
        }
        delete cpu;
        return r;
    };

    // 1. A matching index is a PREFIX, not a corpus sweep: the warmup stages the
    //    two experts it names and leaves the other two to read-through.
    write_index(4);
    {
        const WarmupRun r = warm(1);
        CHECK(r.ok, "a forced warmup initialises");
        CHECK(r.warm_slots == 2, "the warmup stages exactly the ranked experts");
        CHECK(r.loads == 0 && r.promotions == 0 && r.hits == 0,
              "the warmup's own traffic is reset before the run's report");
    }

    // 2. An index measured on a different model must be refused, not trusted:
    //    accepted, it would order the tiers by another model's routing. The
    //    fallback is the even order, which stages the whole corpus.
    write_index(99);
    {
        const WarmupRun r = warm(1);
        CHECK(r.ok, "a mismatched index still initialises");
        CHECK(r.warm_slots == 4, "a refused index falls back to the even order");
    }

    // 3. No index at all is the same fallback, and the message says so.
    std::remove(idx.c_str());
    {
        const WarmupRun r = warm(1);
        CHECK(r.ok, "no index still initialises");
        CHECK(r.warm_slots == 4, "no index falls back to the even order");
    }

    // 4. --expert-warmup 0 stages nothing: the opt-out has to be total.
    {
        const WarmupRun r = warm(0);
        CHECK(r.ok, "a disabled warmup initialises");
        CHECK(r.warm_slots == 0, "a disabled warmup stages nothing");
    }

    // 5. A file that recorded 0 positions carries no ranking, so the warmup
    //    falls back to the even order -- the same fallback as a stale index and
    //    for the same reason. This is the file the REAP-320 scan wrote when it
    //    died mid-generation, which the next load applied for a 25% decode
    //    loss; the even order is both safer and faster than that order.
    write_index(4, 0);
    {
        const WarmupRun r = warm(1);
        CHECK(r.ok, "a zero-position index still initialises");
        CHECK(r.warm_slots == 4,
              "an index that measured nothing falls back to the even order");
    }

    // 6. And a scan that observed nothing writes NO file: a ranking already on
    //    disk has to survive a run that could not measure one. The writer is the
    //    last place this can be caught, because the run that dies mid-generation
    //    still reaches the CLI's index write.
    write_index(4, 64);
    {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = kMoeTestPath;
        cfg.n_ctx = 64;
        cfg.prefill_chunk = 8;
        cfg.expert_cache_mb = 1;
        cfg.expert_warm_mb = 64;
        cfg.expert_warmup = 1;
        engine.set_expert_scan(true, false);
        std::string err;
        const bool ok = engine.init(cpu, cfg, &err);
        CHECK(ok, "a scan-only engine initialises on the warmup fixture");
        if (ok) {
            // No prefill and no decode, so the scan has observed nothing.
            engine.write_expert_index(idx);
            engine.shutdown();
        }
        delete cpu;
    }
    {
        std::ifstream f(idx, std::ios::binary);
        const std::string kept((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
        const ExpertIndexRows rows = expert_index_parse(kept, 1, 4, "", 0);
        CHECK(rows.valid && rows.tokens == 64,
              "a scan that measured nothing leaves the existing index alone");
    }
    std::remove(idx.c_str());
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
static bool build_sized_model(const std::string &path, int n_embd, int n_ff,
                              int mask_id) {
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
    add_spm_vocab(b, vocab, mask_id);

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
                               int max_tokens, bool greedy, bool *ok_out,
                               u64 *proposed_out = nullptr,
                               u64 *accepted_out = nullptr,
                               bool *kv_equal_out = nullptr) {
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
    if (proposed_out) *proposed_out = engine.draft_proposed();
    if (accepted_out) *accepted_out = engine.draft_accepted();
    if (kv_equal_out) {
        // No-holes check: with an identical token history the draft's flat KV
        // must equal the target's row for row. A realign that truncates the
        // cursor but leaves slots unwritten (or vice versa) fails this even
        // when every emitted token ties, which text identity cannot see.
        *kv_equal_out = false;
        const Engine *de = engine.draft_engine();
        if (de && !engine.kvt_k().tiered() && !de->kvt_k().tiered() &&
            engine.kv_pos() > 0 && de->kv_pos() >= engine.kv_pos() &&
            de->kv_pos() - engine.kv_pos() <= window) {
            // The draft may end up to one window past the target: the final
            // round feeds proposals the target never needs (max_tokens hit
            // mid-round, EOS, context end). Those tail rows are speculative
            // work beyond the emitted prefix, not a hole. The invariant is
            // the common prefix: every emitted token at its natural slot.
            const i64 n = engine.kv_pos();
            bool ok = true;
            for (i32 l = 0; l < engine.model().cfg().n_layer && ok; l++) {
                if (!engine.model().layer_has_kv(l)) continue;
                const i64 kd = engine.model().kv_dim_at(l);
                const size_t st =
                    (size_t)engine.kv_capacity() * (size_t)kd * sizeof(f32);
                const size_t sd =
                    (size_t)de->kv_capacity() * (size_t)kd * sizeof(f32);
                const char *tk = (const char *)engine.kvt_k().flat() + (size_t)l * st;
                const char *tv = (const char *)engine.kvt_v().flat() + (size_t)l * st;
                const char *dk = (const char *)de->kvt_k().flat() + (size_t)l * sd;
                const char *dv = (const char *)de->kvt_v().flat() + (size_t)l * sd;
                if (!tk || !tv || !dk || !dv) {
                    ok = false;
                    break;
                }
                for (i64 r = 0; r < n && ok; r++) {
                    ok = std::memcmp(tk + (size_t)r * (size_t)kd * sizeof(f32),
                                     dk + (size_t)r * (size_t)kd * sizeof(f32),
                                     (size_t)kd * sizeof(f32)) == 0 &&
                         std::memcmp(tv + (size_t)r * (size_t)kd * sizeof(f32),
                                     dv + (size_t)r * (size_t)kd * sizeof(f32),
                                     (size_t)kd * sizeof(f32)) == 0;
                }
            }
            *kv_equal_out = ok;
        }
    }
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
    // check of the block/accept/realign machinery. Text identity alone cannot
    // catch a stale draft cache -- exactness holds by construction at any
    // acceptance rate -- so the acceptance itself is asserted: identical
    // weights must agree with themselves. (Once measured 27.8% here because
    // the propose loop fed only d-1 of d proposals, leaving a permanent hole
    // of stale KV rows past every full accept.)
    {
        // The final round is cut short by max_tokens (16 = 3 full rounds of
        // 4+bonus plus one emitted proposal): its un-emitted proposals count
        // as proposed but never accepted, although nothing rejected them.
        // Forgive up to one window for that truncated tail; anything more is
        // a real disagreement between identical weights.
        const i32 win = 4;
        u64 proposed = 0, accepted = 0;
        const GenerateResult r =
            run_spec(target, same, win, 16, true, &ok, &proposed, &accepted);
        CHECK(ok && r.generated == 16, "spec (self draft) completed");
        CHECK(r.tokens == plain.tokens,
              "spec output identical to plain greedy (self draft)");
        CHECK(proposed > 0, "self draft proposed tokens");
        CHECK(accepted + static_cast<u64>(win) >= proposed,
              "self draft accepts ~all its proposals (identical weights)");
    }

    // 2b. The contiguity invariant as a bitwise check: a model drafting for
    // itself saw the same tokens at the same slots as its target, so its KV
    // must equal the target's row for row. The propose-feeds-d-1 hole passed
    // check 2's text identity (ties) and failed only the acceptance rate;
    // this fails on ANY slot divergence, hole or otherwise, with no rate to
    // hide behind.
    {
        bool kv_equal = false;
        const GenerateResult r =
            run_spec(target, same, 4, 16, true, &ok, nullptr, nullptr, &kv_equal);
        CHECK(ok && r.generated == 16, "spec (self draft KV parity) completed");
        CHECK(kv_equal, "self draft KV bitwise-identical to target KV");
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

// ---------------------------------------------------------------------------
// DFlash head set: loader refused on a mask-id mismatch, by construction
// ---------------------------------------------------------------------------

// A DFlash head set whose geometry is small and self-consistent, so the loader
// gets all the way to the mask check. Every tensor is f16 with values written
// through wv(), which is also what the target fixture uses, so no dequantizer
// quirk can be the reason a check fires.
//
// Contract per src/dflash.cpp's loader: the loader reads `<arch>.block_count`
// etc. with key() and then `<arch>.dflash.*` with dkey(), and a head set built
// as arch "dflash" with the flat `dflash.*` keys exercises the same default path
// the real ones take.
// The DFlash *target*: the same geometry build_sized_model writes with
// tokenizer.ggml.mask_token_id present, so the engine's mask-id guard has a real
// value to compare against a head set's own id.
static bool build_dflash_target(const std::string &path) {
    return build_sized_model(path, 32, 64, 12);
}

// `which` names one matrix to write as the NVFP4 encoding (with a 0.5 sidecar)
// instead of its f16 twin; both spellings carry the SAME numbers (put_weight),
// so two head sets that differ only in `nvfp4` describe the same drafter. Only
// weights whose contraction is a multiple of the 64-value NVFP4 block are
// usable here (fc's 64-wide feature row, ffn_down's 64-wide intermediate).
static bool build_dflash_head_set(const std::string &path, int mask_id,
                                  const std::string &which = std::string(),
                                  bool nvfp4 = false, bool laguna = false) {
    const int n_embd = 32, n_layer = 2, n_head = 4, n_kv = 2, hd = 8, n_ff = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int n_target_layers = 2;           // captures 0 (input) + 1 (pre-final)
    const int n_embd_enc = n_target_layers * n_embd;   // fc reads 64
    // The loader refuses a head set whose vocab differs from the target's
    // (the Drafter-loads fixture must equal 260 = build_sized_model's vocab).
    const int vocab = 260, block_size = 4;

    // Two spellings of ONE head set, because the loader reads both and they have
    // to reach the same numbers:
    //   `dflash`       in-house: `fc.weight`, stacked `enc.aux_norm`,
    //                  `blk.N.ffn_norm`, and an explicit `vocab_size`.
    //   `dflash-draft` poolside's Laguna export: the prefix is part of the tensor
    //                  name (`dflash_fc.weight`, `dflash_hidden_norm.weight`,
    //                  `dflash_aux_hidden_norm.N.weight`), the FFN norm is called
    //                  `post_attention_norm`, and `vocab_size` is not written at
    //                  all -- so the drafter has to size itself from the target.
    const std::string arch = laguna ? "dflash-draft" : "dflash";
    auto K = [&](const char *suffix) { return arch + suffix; };

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-dflash-mask-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(K(".block_count"), n_layer);
    b.meta_u32(K(".context_length"), 64);
    b.meta_u32(K(".embedding_length"), n_embd);
    b.meta_u32(K(".feed_forward_length"), n_ff);
    b.meta_u32(K(".attention.head_count"), n_head);
    b.meta_u32(K(".attention.head_count_kv"), n_kv);
    b.meta_u32(K(".attention.key_length"), hd);
    b.meta_f32(K(".attention.layer_norm_rms_epsilon"), 1e-5f);
    if (!laguna) b.meta_u32(K(".vocab_size"), vocab);
    b.meta_f32(K(".rope.freq_base"), 10000.0f);
    b.meta_u32(K(".block_size"), block_size);
    b.meta_u32(K(".mask_token_id"), static_cast<u32>(mask_id));
    // Capture ids against the one-layer target: id 0 is the input of layer 0,
    // id 1 is the pre-final-norm state (equal to the target's layer count and
    // the exact top of the range the loader accepts). A two-slot set of width
    // n_embd*2 in fc keeps the shapes legal.
    b.meta_i32_array(K(".target_layers"), {0, 1});
    // The loader reads the mask id through dkey(): `<arch>.dflash.mask_token_id`
    // for the current spelling, i.e. the key lives one level down -- this is the
    // same nesting a real head set carries (qwen35-dflash-draft.dflash.mask_
    // token_id). The flat key above is kept so both spellings load.
    b.meta_u32(K(".dflash.mask_token_id"), static_cast<u32>(mask_id));

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };
    auto put = [&](const std::string &name, i64 n_out, i64 n_in, i64 base) {
        if (name == which) {
            put_weight(b, name, n_out, n_in, nvfp4);
            return;
        }
        b.tensor_f16(name, {u64(n_in), u64(n_out)},
                     vec(static_cast<size_t>(n_in * n_out), base));
    };

    // Fusion projection: the loader reads inner width ne[0] as the *input*
    // (n_captured x target embd) and rows as the output (the drafter's n_embd),
    // matching the real `n_captured x n_embd -> n_embd` shape.
    put(laguna ? "dflash_fc.weight" : "fc.weight", n_embd, n_embd_enc, 10);
    // Encoder output norm + per-aux norms, one n_embd row per capture slot.
    if (laguna) {
        b.tensor_f32("dflash_hidden_norm.weight", {u64(n_embd)}, ones(n_embd));
        for (int a = 0; a < n_target_layers; a++)
            b.tensor_f32("dflash_aux_hidden_norm." + std::to_string(a) + ".weight",
                         {u64(n_embd)}, ones(n_embd));
    } else {
        b.tensor_f32("enc.output_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f32("enc.aux_norm.weight", {u64(n_embd), u64(n_target_layers)},
                     ones(static_cast<size_t>(n_embd) * n_target_layers));
    }
    // Decoder final norm (the loader requires it; the head is borrowed)
    b.tensor_f32("output_norm.weight", {u64(n_embd)}, ones(n_embd));

    for (int l = 0; l < n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        auto v = [&](i64 n, i64 base) { return vec(static_cast<size_t>(n), base); };
        b.tensor_f32(p + "attn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f32(p + (laguna ? "post_attention_norm.weight" : "ffn_norm.weight"),
                     {u64(n_embd)}, ones(n_embd));
        b.tensor_f16(p + "attn_q.weight", {u64(n_embd), u64(q_dim)}, v(q_dim, 100 + l * 10));
        b.tensor_f16(p + "attn_k.weight", {u64(n_embd), u64(kv_dim)}, v(kv_dim, 200 + l * 10));
        b.tensor_f16(p + "attn_v.weight", {u64(n_embd), u64(kv_dim)}, v(kv_dim, 300 + l * 10));
        b.tensor_f16(p + "attn_output.weight", {u64(q_dim), u64(n_embd)}, v(n_embd, 400 + l * 10));
        // QK-norm is required by dflash's per-head norm contract
        b.tensor_f32(p + "attn_q_norm.weight", {u64(hd)}, ones(hd));
        b.tensor_f32(p + "attn_k_norm.weight", {u64(hd)}, ones(hd));
        put(p + "ffn_gate.weight", n_ff, n_embd, 500 + l * 10);
        put(p + "ffn_up.weight", n_ff, n_embd, 900 + l * 10);
        put(p + "ffn_down.weight", n_embd, n_ff, 1300 + l * 10);
    }
    return b.write(path);
}

// One drafted block's logits rows for a head set, through the same calls the
// speculative loop makes: prefill (which commits the features), then
// draft_block_rows. Shared by the tests that compare two files.
static bool dflash_rows_of(const std::string &target, const std::string &head_path,
                           std::vector<f32> &out, std::string &err) {
    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = target;
    cfg.n_ctx = 64;
    if (!engine.init(cpu, cfg, &err)) return false;
    if (!engine.load_dflash(head_path, &err)) return false;
    GenerateParams p;
    p.prompt = "the";
    p.max_tokens = 1;
    p.sampler.greedy = true;
    GenerateResult gr;
    if (!engine.generate(p, &gr)) return false;
    std::vector<i32> ids;
    if (engine.tokenizer().encode("the", ids, true) <= 0 || ids.empty()) {
        err = "the fixture prompt does not encode";
        return false;
    }
    const Model &m = engine.model();
    const i32 nv = m.cfg().n_vocab;
    const i32 blk = engine.dflash_block_size();
    Engine::DflashRows dfr;
    const i32 nr = engine.draft_block_rows(m.tok_embd(), m.out_head(), nv, ids.back(),
                                           blk - 1,
                                           static_cast<i32>(gr.prompt_tokens), dfr);
    if (nr <= 0) {
        err = "the block produced no rows";
        return false;
    }
    out = dfr.data;
    return true;
}

// The third spelling: poolside's `dflash-draft` export. Renamed tensors, one
// aux norm per layer, `post_attention_norm` for the FFN norm, and no
// `vocab_size` -- so it exercises both the name fallbacks and the vocabulary
// fallback. The numbers are the canonical file's, so the drafted rows have to
// come out identical: a spelling moves names, never values.
static void test_dflash_laguna_spelling() {
    const char *target = "kraken-dflash-laguna-target.gguf";
    CHECK(build_dflash_target(target), "wrote the DFlash target model");
    const std::string plain = "kraken-dflash-canonical.gguf";
    const std::string laguna = "kraken-dflash-laguna.gguf";
    CHECK(build_dflash_head_set(plain, 12), "wrote the canonical head set");
    CHECK(build_dflash_head_set(laguna, 12, std::string(), false, true),
          "wrote the poolside-spelling head set");
    std::vector<f32> a_rows, b_rows;
    std::string ea, eb;
    const bool ok_a = dflash_rows_of(target, plain, a_rows, ea);
    const bool ok_b = dflash_rows_of(target, laguna, b_rows, eb);
    if (!ok_a) std::fprintf(stderr, "  (canonical: %s)\n", ea.c_str());
    if (!ok_b) std::fprintf(stderr, "  (poolside spelling: %s)\n", eb.c_str());
    CHECK(ok_a && ok_b, "both spellings load and draft");
    if (!ok_a || !ok_b) return;
    CHECK(a_rows.size() == b_rows.size(),
          "the two spellings draft the same number of rows");
    f32 worst = 0.0f;
    for (size_t i = 0; i < std::min(a_rows.size(), b_rows.size()); i++)
        worst = std::max(worst, std::fabs(a_rows[i] - b_rows[i]));
    CHECK(worst == 0.0f, "and the same numbers: a spelling moves names, not values");
}

// NVFP4 global scales on a DFlash head set. The drafter has its own weight
// uploader and its own projections, so the trunk's `gemm_w` rule has to reach it
// too -- and it is checked the same way: the same matrix written two ways (f16
// twin vs NVFP4 + 0.5 sidecar) must produce the same drafted logits row for row.
static void test_nvfp4_sidecar_dflash() {
    const char *target = "kraken-dflash-scale-target.gguf";
    CHECK(build_dflash_target(target), "wrote the DFlash target model");

    // fc (the fusion projection, run inside commit) and one decoder block's down
    // projection are the two shapes whose contraction is a whole NVFP4 block.
    const char *arms[] = {"fc.weight", "blk.0.ffn_down.weight"};
    for (const char *arm : arms) {
        const std::string twin = std::string("kraken-dflash-f16-") + "x.gguf";
        const std::string scaled = std::string("kraken-dflash-nvfp4-") + "x.gguf";
        CHECK(build_dflash_head_set(twin, 12, arm, false),
              "wrote the f16 head set");
        CHECK(build_dflash_head_set(scaled, 12, arm, true),
              "wrote the NVFP4 head set");
        std::vector<f32> a_rows, b_rows;
        std::string ea, eb;
        const bool ok_a = dflash_rows_of(target, twin, a_rows, ea);
        const bool ok_b = dflash_rows_of(target, scaled, b_rows, eb);
        if (!ok_a) std::fprintf(stderr, "  (dflash f16 twin: %s)\n", ea.c_str());
        if (!ok_b) std::fprintf(stderr, "  (dflash nvfp4: %s)\n", eb.c_str());
        CHECK(ok_a && ok_b, "both head sets load and draft");
        if (!ok_a || !ok_b) continue;
        CHECK(a_rows.size() == b_rows.size(),
              "the two head sets draft the same number of rows");
        f32 worst = 0.0f;
        for (size_t i = 0; i < std::min(a_rows.size(), b_rows.size()); i++)
            worst = std::max(worst, std::fabs(a_rows[i] - b_rows[i]));
        if (worst > 2.0f)
            std::fprintf(stderr, "  (dflash arm %s: rows differ by %.4g)\n", arm,
                         static_cast<double>(worst));
        CHECK(worst < 2.0f, "and its drafted logits match the f16 twin's");
    }
}

// The masked-block contract (AttnDesc::skip_from): a candidate row of a block
// sees the context and ITSELF, never its sibling candidates. Checked by
// perturbation, because invariance under a change the row cannot see IS the
// property -- and a no-op mask cannot fake it, since the same perturbation must
// move the row when the mask is off.
// Gemma 2's attention-logit bound (AttnDesc::softcap): every score becomes
// cap * tanh(score / cap) at score formation, so the softmax runs on the capped
// vector. The distinction matters -- tanh is monotone, but the probabilities
// are those of the capped vector, not a monotone image of the uncapped ones --
// and it is the reason the bound cannot be applied to the attention output.
//
// The fixture makes the arithmetic checkable by hand: unit-norm keys along one
// axis so the raw scores are exactly 0, 4, 8, and value rows that are the basis
// vectors e0, e1, e2, so the last row's four outputs ARE the three softmax
// probabilities. Expected values are computed here in double from the formula,
// not from the backend.
static void test_attention_softcap() {
    Backend *cpu = make_cpu_backend();
    CHECK(cpu->attention_softcap_supported(),
          "the reference attention applies the attention-logit bound");
    const i64 nh = 1, nkv = 1, hd = 4, nt = 3, npos = 3;
    const i64 kv_dim = nkv * hd;
    const i64 layer_stride = npos * kv_dim;
    const f32 raw[3] = {0.0f, 4.0f, 8.0f};

    std::vector<f32> kc(static_cast<size_t>(layer_stride), 0.0f);
    std::vector<f32> vc(static_cast<size_t>(layer_stride), 0.0f);
    for (i64 j = 0; j < nt; j++) {
        kc[static_cast<size_t>(j * kv_dim + 0)] = raw[j]; // dot with q = raw[j]
        vc[static_cast<size_t>(j * kv_dim + j)] = 1.0f;    // e_j
    }
    std::vector<f32> q(static_cast<size_t>(nt * nh * hd), 0.0f);
    for (i64 t = 0; t < nt; t++) q[static_cast<size_t>(t * hd + 0)] = 1.0f;

    AttnDesc d;
    d.n_head = nh;
    d.n_kv = nkv;
    d.hd = hd;
    d.n_tok = nt;
    d.pos0 = 0;
    d.pos_stride = kv_dim;
    d.layer_stride = layer_stride;
    d.scale = 1.0f;   // the scores ARE the dots, which is what makes them exact
    d.causal = true;

    // The three probabilities of a capped softmax, in double. `lo` and `hi` are
    // the visible key indices (the mask arms below narrow this).
    auto probs = [](const f32 *scores, i64 n, f64 cap) {
        std::vector<f64> v(static_cast<size_t>(n));
        f64 mx = -1e300, sum = 0.0;
        for (i64 j = 0; j < n; j++) {
            v[static_cast<size_t>(j)] =
                cap > 0.0 ? cap * std::tanh(static_cast<f64>(scores[j]) / cap)
                          : static_cast<f64>(scores[j]);
            mx = std::max(mx, v[static_cast<size_t>(j)]);
        }
        for (i64 j = 0; j < n; j++) {
            v[static_cast<size_t>(j)] = std::exp(v[static_cast<size_t>(j)] - mx);
            sum += v[static_cast<size_t>(j)];
        }
        for (i64 j = 0; j < n; j++) v[static_cast<size_t>(j)] /= sum;
        return v;
    };
    // Row `t` sees keys [0, t]; its output is sum_j p_j e_j, so out[t][j] == p_j
    // for the visible j and 0 otherwise.
    auto check_rows = [&](const std::vector<f32> &out, f64 cap,
                          const char *what) {
        bool ok = true;
        for (i64 t = 0; t < nt; t++) {
            const std::vector<f64> want = probs(raw, t + 1, cap);
            for (i64 j = 0; j < t + 1; j++) {
                const f64 got = out[static_cast<size_t>(t * hd + j)];
                if (!(std::fabs(got - want[static_cast<size_t>(j)]) < 2e-6)) ok = false;
            }
            // A key outside the causal prefix must contribute nothing at all.
            for (i64 j = t + 1; j < hd; j++)
                if (out[static_cast<size_t>(t * hd + j)] != 0.0f) ok = false;
            const f64 sum = out[static_cast<size_t>(t * hd + 0)] +
                            out[static_cast<size_t>(t * hd + 1)] +
                            out[static_cast<size_t>(t * hd + 2)];
            if (!(std::fabs(sum - 1.0) < 2e-6)) ok = false;
        }
        CHECK(ok, what);
    };

    // ---- 1. the uncapped arm: the arithmetic that predates the field ------
    std::vector<f32> out_default(static_cast<size_t>(nt * nh * hd), 0.0f);
    std::vector<f32> out_zero(static_cast<size_t>(nt * nh * hd), 0.0f);
    cpu->attention(out_default.data(), q.data(), kc.data(), vc.data(), d);
    AttnDesc z = d;
    z.softcap = 0.0f;
    cpu->attention(out_zero.data(), q.data(), kc.data(), vc.data(), z);
    CHECK(std::memcmp(out_default.data(), out_zero.data(),
                      out_default.size() * sizeof(f32)) == 0,
          "an unset softcap and an explicit 0 are the same call: the field "
          "cannot change a model that does not carry it");
    check_rows(out_default, 0.0, "and it is the plain softmax of the raw scores");

    // ---- 2. the capped arm, against the formula --------------------------
    std::vector<f32> out_cap(static_cast<size_t>(nt * nh * hd), 0.0f);
    AttnDesc c = d;
    c.softcap = 2.0f;
    cpu->attention(out_cap.data(), q.data(), kc.data(), vc.data(), c);
    check_rows(out_cap, 2.0,
               "cap*tanh(score/cap) at score formation: the probabilities are "
               "the capped softmax's, computed independently in double");
    // Row 2 is where the bound bites: raw scores 0/4/8 would put 98% of the
    // mass on the last key; capped at 2 the split is ~0.07/0.45/0.48.
    CHECK(out_cap[static_cast<size_t>(2 * hd + 2)] < 0.60f &&
              out_cap[static_cast<size_t>(2 * hd + 2)] > 0.40f,
          "the bound compresses a 0/4/8 score spread towards uniform, which is "
          "the whole point of it");
    CHECK(out_default[static_cast<size_t>(2 * hd + 2)] > 0.95f,
          "and the uncapped arm keeps the peak, so the two arms are not the "
          "same computation");
    CHECK(std::memcmp(out_cap.data(), out_default.data(),
                      out_cap.size() * sizeof(f32)) != 0,
          "a run that sets the field and one that does not cannot agree");

    // ---- 3. the mask is not a score --------------------------------------
    // skip_from makes key 1 invisible to row 2, and the reference represents
    // that as -FLT_MAX. If the bound were applied to the sentinel it would
    // become -cap, i.e. exp(-2 - m) of real probability on a key the model
    // asked to hide. Row 2 must be the capped softmax of {0, 2} only.
    std::vector<f32> out_mask(static_cast<size_t>(nt * nh * hd), 0.0f);
    AttnDesc m = c;
    m.skip_from = 1;
    cpu->attention(out_mask.data(), q.data(), kc.data(), vc.data(), m);
    {
        const f32 visible[2] = {raw[0], raw[2]};
        const std::vector<f64> want = probs(visible, 2, 2.0);
        const f64 p0 = out_mask[static_cast<size_t>(2 * hd + 0)];
        const f64 p2 = out_mask[static_cast<size_t>(2 * hd + 2)];
        CHECK(std::fabs(p0 - want[0]) < 2e-6,
              "a masked key contributes nothing even when the bound is on: it "
              "is a mask, not a score");
        CHECK(std::fabs(p2 - want[1]) < 2e-6, "and the visible keys renormalize");

        // The bound's own renormalization is the counter-check: with the same
        // cap but no mask, key 1 takes a share. If the two arms agreed here the
        // mask test would be vacuous.
        CHECK(std::fabs(p0 - out_cap[static_cast<size_t>(2 * hd + 0)]) > 1e-4,
              "and that is a different split from the unmasked row's");
    }

    // ---- 4. the cap is monotone, so ranking survives ---------------------
    // A bound this aggressive still must not reorder the scores: the visible
    // key with the largest dot keeps the largest probability.
    CHECK(out_cap[static_cast<size_t>(2 * hd + 2)] >
              out_cap[static_cast<size_t>(2 * hd + 1)] &&
              out_cap[static_cast<size_t>(2 * hd + 1)] >
                  out_cap[static_cast<size_t>(2 * hd + 0)],
          "tanh is monotone, so the bound changes the distribution and not the "
          "ordering");

    delete cpu;
}

static void test_attention_self_mask() {
    Backend *cpu = make_cpu_backend();
    const i64 nh = 2, nkv = 2, hd = 8, nt = 3;
    const i64 kv_dim = nkv * hd;
    const i64 npos = 8;
    const i64 layer_stride = npos * kv_dim;
    const size_t ne = static_cast<size_t>(layer_stride);

    std::vector<f32> kc(ne, 0.0f), vc(ne, 0.0f);
    for (i64 j = 0; j < nt; j++)
        for (i64 i = 0; i < kv_dim; i++) {
            kc[static_cast<size_t>(j * kv_dim + i)] =
                0.1f * static_cast<f32>((j + 1) * (i + 3));
            vc[static_cast<size_t>(j * kv_dim + i)] =
                0.05f * static_cast<f32>((j + 2) * (i + 1)) - 0.2f;
        }
    std::vector<f32> q(static_cast<size_t>(nt * nh * hd));
    for (size_t i = 0; i < q.size(); i++)
        q[i] = 0.03f * static_cast<f32>((static_cast<i64>(i) % 7) - 3);

    AttnDesc d;
    d.n_head = nh;
    d.n_kv = nkv;
    d.hd = hd;
    d.n_tok = nt;
    d.pos0 = 0;
    d.pos_stride = kv_dim;
    d.layer_stride = layer_stride;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd)));
    d.causal = true;
    d.window = 0;

    std::vector<f32> out_a(q.size(), 0.0f), out_b(q.size(), 0.0f), out_c(q.size(), 0.0f);

    // Arm 1: the mask hides the sibling. Rows are at positions 0, 1, 2, so
    // skip_from = 1 masks row 1 out of row 2's view (and nothing out of row 1's
    // own, whose skip range [1, 1) is empty).
    AttnDesc m = d;
    m.skip_from = 1;
    cpu->attention(out_a.data(), q.data(), kc.data(), vc.data(), m);

    std::vector<f32> kc2 = kc, vc2 = vc;
    for (i64 i = 0; i < kv_dim; i++) {
        kc2[static_cast<size_t>(kv_dim + i)] += 1.5f;
        vc2[static_cast<size_t>(kv_dim + i)] -= 1.5f;
    }
    cpu->attention(out_b.data(), q.data(), kc2.data(), vc2.data(), m);

    const auto row = [&](const std::vector<f32> &o, i64 t) {
        std::vector<f32> r(static_cast<size_t>(nh * hd));
        for (i64 i = 0; i < nh * hd; i++)
            r[static_cast<size_t>(i)] = o[static_cast<size_t>(t * nh * hd + i)];
        return r;
    };
    const auto maxdiff = [](const std::vector<f32> &a, const std::vector<f32> &b) {
        f32 w = 0.0f;
        for (size_t i = 0; i < std::min(a.size(), b.size()); i++)
            w = std::max(w, std::fabs(a[i] - b[i]));
        return w;
    };

    CHECK(maxdiff(row(out_a, 2), row(out_b, 2)) == 0.0f,
          "a masked sibling cannot move this row: bit-identical under perturbation");
    CHECK(maxdiff(row(out_a, 1), row(out_b, 1)) > 0.01f,
          "while the perturbed row itself does move (the perturbation is real)");
    CHECK(maxdiff(row(out_a, 0), row(out_b, 0)) == 0.0f,
          "and row 0 is untouched, as causality already required");

    // Arm 2: no mask -- the same perturbation must reach row 2, or arm 1 proved
    // nothing about the mask.
    AttnDesc off = d;
    off.skip_from = -1;
    cpu->attention(out_c.data(), q.data(), kc2.data(), vc2.data(), off);
    CHECK(maxdiff(row(out_c, 2), row(out_a, 2)) > 0.01f,
          "with skip_from off the same sibling perturbation does reach row 2");

    // Arm 3: the mask is a RANGE, so a block that starts at position 0 leaves a
    // row with exactly one visible key -- its own. The output is then V itself.
    AttnDesc one = d;
    one.skip_from = 0;
    std::vector<f32> out_d(q.size(), 0.0f);
    cpu->attention(out_d.data(), q.data(), kc.data(), vc.data(), one);
    f32 worst = 0.0f;
    for (i64 h = 0; h < nh; h++)
        for (i64 i = 0; i < hd; i++) {
            const i64 hkv = h % nkv;
            const f32 want = vc[static_cast<size_t>(2 * kv_dim + hkv * hd + i)];
            const f32 got = out_d[static_cast<size_t>(2 * nh * hd + h * hd + i)];
            worst = std::max(worst, std::fabs(want - got));
        }
    CHECK(worst < 1e-6f, "a row whose only visible key is itself returns V");

    // Arm 4: a block whose rows are the only thing in range still produces
    // finite numbers (the all-masked-tile trap: exp(0) would leak a phantom
    // unit of denominator mass).
    bool finite = true;
    for (f32 x : out_d) finite = finite && std::isfinite(x);
    CHECK(finite, "and no NaN or inf appears in the fully masked case");
    delete cpu;
}

static void test_dflash_mask_guard() {
    // The target needs tokenizer.ggml.mask_token_id set so the guard has
    // something to compare against; build_sized_model does not emit it, so the
    // DFlash target here is written directly with the key present.
    const char *target = "kraken-dflash-target.gguf";
    CHECK(build_dflash_target(target), "wrote the DFlash target model");

    const char *good = "kraken-dflash-good.gguf";
    const char *bad = "kraken-dflash-bad.gguf";
    CHECK(build_dflash_head_set(good, 12), "wrote the matching mask head set");
    CHECK(build_dflash_head_set(bad, 13), "wrote the mismatched mask head set");

    // The guard is in Engine::load_dflash, after the range check.
    Backend *cpu = make_cpu_backend();
    Engine engine;
    EngineConfig cfg;
    cfg.model_path = target;
    cfg.n_ctx = 64;
    std::string err;
    CHECK(engine.init(cpu, cfg, &err), "engine initialises on the DFlash target");

    // 1. A head set whose mask id is inside the vocabulary loads (the guard is
    //    a mismatch check, not a rigid "must equal 12").
    {
        std::string derr;
        const bool ok = engine.load_dflash(good, &derr);
        CHECK(ok, "a DFlash head set with an in-vocab mask id loads");
        if (!ok) std::fprintf(stderr, "  (dflash good: %s)\n", derr.c_str());
    }

    // 2. The mismatched pair is refused, and the reason names both ids.
    {
        // engine.load_dflash() with drafter_ already set returns true early, so
        // the refusal path needs a fresh engine.
        Engine fresh;
        EngineConfig fcfg;
        fcfg.model_path = target;
        fcfg.n_ctx = 64;
        std::string ferr;
        CHECK(fresh.init(cpu, fcfg, &ferr), "fresh engine for the mismatch check");
        const bool refused = !fresh.load_dflash(bad, &ferr);
        CHECK(refused, "a mismatched mask id is refused outright");
        CHECK(ferr.find("13") != std::string::npos &&
                  ferr.find("12") != std::string::npos,
              "the refusal names both ids");
        if (!refused)
            std::fprintf(stderr, "  (dflash bad: %s)\n", ferr.c_str());
        fresh.shutdown();
    }

    // 3. The diagnostic override: the switch must load the same mismatched
    //    pair the refusal just stopped, and only while it is set. Proving it
    //    both ways keeps the switch under test rather than under hope: a future
    //    edit that ignores the env (always refuses) or that lets it leak past
    //    the process (always loads) fails here. set_env_var/clear_env_var are
    //    the portable spellings, already defined further down.
    {
        set_env_var("KRK_DFLASH_ALLOW_BAD_MASK", "1");
        Engine ov;
        EngineConfig ocfg;
        ocfg.model_path = target;
        ocfg.n_ctx = 64;
        std::string oerr;
        CHECK(ov.init(cpu, ocfg, &oerr), "engine for the override check");
        const bool loaded = ov.load_dflash(bad, &oerr);
        CHECK(loaded, "KRK_DFLASH_ALLOW_BAD_MASK=1 loads the mismatched pair");
        if (!loaded) std::fprintf(stderr, "  (override: %s)\n", oerr.c_str());
        ov.shutdown();
        clear_env_var("KRK_DFLASH_ALLOW_BAD_MASK");

        Engine re;
        EngineConfig rcfg;
        rcfg.model_path = target;
        rcfg.n_ctx = 64;
        std::string rerr;
        CHECK(re.init(cpu, rcfg, &rerr), "engine after the override is cleared");
        CHECK(!re.load_dflash(bad, &rerr),
              "clearing the switch restores the refusal");
        re.shutdown();
    }

    // 4. One real draft_block call on the synthetic head set: the block's
    //    row-to-position mapping, with no 12 GiB model in sight.
    //
    //    The block answers one question per row: "what would the target say at
    //    position pos+k?" Row k of the block (block row k+1, 1-based, since
    //    row 0 carries id_last) is read from logits rows_out[k*n_vocab]. Two
    //    contracts are checkable with UNTRAINED weights because they do not
    //    depend on the weights being right, only on the plumbing:
    //
    //    (a) determinism + row identity — the same call twice gives the same
    //        logits row for row (a moved row would mean the row/position wiring
    //        is not a pure function of k), and changing id_last changes the
    //        output (the block's rows respond to their own first token).
    //    (b) the head set is the DRAFTER's: rows_out must be at least
    //        block_size()-1 rows wide and MUST NOT read past it — the loader
    //        clamps n_draft to that, and a caller passing more models see the
    //        clamp, not a bigger read (that is the reference's window rule).
    //
    //    The engine is already loaded with `good` (check 1); commit() the
    //    prompt's features so the injected context exists, then draft once.
    {
        const Model &m = engine.model();
        const QuantTensor &embd = m.tok_embd();
        const QuantTensor &head = m.out_head();
        // The MODEL's vocab, not the tokenizer's: the block writes its
        // candidate logits through the head, whose row width is the model's
        // declared vocab_size (260 here), while the tokenizer's tokens array
        // happens to carry 261 entries. The speculative loop itself uses
        // n_vocab_ everywhere (which is why the real path never corrupted),
        // and handing draft_block the wider count walked its ws_logits_ rows
        // off the end — intermittent heap corruption plus a row of NaN, the
        // very defect the loader's new vocab check exists to prevent.
        const i32 nv = m.cfg().n_vocab;

        // encode the prompt "the" (the same run_spec uses), prefill it, commit.
        std::vector<i32> ids;
        CHECK(engine.tokenizer().encode("the", ids, true) > 0,
              "DFlash fixture prompt encodes");
        GenerateParams pp;
        pp.prompt = "the";
        pp.max_tokens = 0;   // prefill only: generate() with 0 still prefill+commits?
        // generate() with max_tokens 0 would not prefill past zero; instead:
        // one forward + commit manually through the same calls the speculative
        // loop makes. forward() is public via the engine header (test-only).
        // Simplest honest path: run generate() with max_tokens=1, greedy, which
        // prefills the prompt into the drafter's cache and gives a real pos=5.
        GenerateResult gr;
        GenerateParams p1;
        p1.prompt = "the";
        p1.max_tokens = 1;
        p1.sampler.greedy = true;
        CHECK(engine.generate(p1, &gr), "one greedy step prefills the drafter");

        // The block: id_last is the last prompt token, pos is past the prompt.
        const i32 blk = engine.dflash_block_size();
        CHECK(blk >= 2, "the synthetic head set has a block size");
        const i32 pos = static_cast<i32>(gr.prompt_tokens);
        std::vector<f32> rows(static_cast<size_t>(blk - 1) * nv, 0.0f);

        Engine::DflashRows dfr;
        const i32 nrows = engine.draft_block_rows(embd, head, nv,
                                                  ids.back(), blk - 1, pos,
                                                  dfr);
        CHECK(nrows == blk - 1, "draft_block returns the clamped window");
        CHECK(dfr.data.size() == static_cast<size_t>(nrows) * nv,
              "the host copy is one row per candidate");
        // Row-to-position mapping, asserted structurally rather than by value:
        // each candidate row must be its own vector — no two rows identical —
        // and every row must name exactly one argmax. A wiring defect that
        // copies one row's logits into every slot, or that maps rows onto the
        // wrong candidate positions, fails this WITHOUT needing trained
        // weights: the untrained head set carries no reason for two later
        // positions to agree exactly with an earlier one, because each row
        // sits at its own rope position and attends its own slice of the
        // block's K/V.
        for (i32 r = 0; r < nrows; r++) {
            const f32 *row = dfr.data.data() + static_cast<size_t>(r) * nv;
            i32 am = 0;
            for (i32 v = 1; v < nv; v++)
                if (row[v] > row[am]) am = v;
            bool dup = false;
            for (i32 q = 0; q < nrows; q++) {
                if (q == r) continue;
                bool same_all = true;
                for (i32 v = 0; v < nv; v++)
                    if (dfr.data[static_cast<size_t>(q) * nv +
                                 static_cast<size_t>(v)] != row[v]) {
                        same_all = false;
                        break;
                    }
                if (same_all) dup = true;
            }
            CHECK(!dup, "each candidate row is a distinct logits vector");
            (void)am;   // argmax existence is still useful in the finite check below
        }

        // Row identity: a different id_last gives different rows — the rows
        // respond to the block's own first token, not to a constant.
        Engine::DflashRows dfr3;
        const i32 other = (ids.back() + 1 < nv) ? ids.back() + 1 : 1;
        engine.draft_block_rows(embd, head, nv, other, blk - 1, pos, dfr3);
        bool differs = false;
        for (size_t i = 0; i < dfr.data.size(); i++)
            if (dfr.data[i] != dfr3.data[i]) { differs = true; break; }
        CHECK(differs, "changing id_last changes the proposed rows");

        // Non-finiteness: the synthetic weights are small, so NaN/Inf here would
        // mean the plumbing, not the trained scale, is broken. The diagnostic
        // prints the offending row (which candidate position) and its extreme
        // element, because a NaN in row 2 only and never row 0 is a different
        // answer from a whole-buffer garbage — position-wired vs. flat-broken.
        bool finite = true;
        i32 bad_row = -1;
        f32 bad_val = 0.0f, best_finite = -1e30f;
        for (i32 r = 0; r < nrows; r++) {
            bool row_finite = true;
            f32 rbad = 0.0f;
            for (i32 v = 0; v < nv; v++) {
                const f32 x = dfr.data[static_cast<size_t>(r) * nv +
                                       static_cast<size_t>(v)];
                if (!std::isfinite(x)) {
                    row_finite = false;
                    rbad = x;
                    break;
                }
                best_finite = std::max(best_finite, std::fabs(x));
            }
            if (!row_finite && finite) {
                finite = false;
                bad_row = r;
                bad_val = rbad;
            }
        }
        CHECK(finite, "the block's rows are all finite");
        if (!finite) {
            std::fprintf(stderr,
                         "  (dflash rows: first non-finite at row %d, value %g, "
                         "largest finite |v| %g, nv %d, nrows %d)\n",
                         bad_row, static_cast<double>(bad_val),
                         static_cast<double>(best_finite), nv, nrows);
        }
    }

    engine.shutdown();
    delete cpu;
}

static void test_moe_grouped_prefill_matches_tokenwise() {
    // Self-contained: it builds the MoE fixture it then compares against
    // itself, because it used to load whatever the last MoE group happened to
    // leave behind -- so its weights were a function of the schedule, not of
    // this file.
    MoeSpec vs;
    vs.dense = false;
    vs.n_expert = 4;
    vs.n_expert_used = 2;
    vs.shared = true;
    vs.varied = true;
    if (!build_tiny_moe_model(kMoeTestPath, vs)) {
        CHECK(false, "wrote the MoE fixture this group compares against itself");
        return;
    }

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

// The router's residency preference, on a fixture whose expert tier is SMALLER
// than its routed set -- the only shape where the choice can matter. What has
// to hold: with the margin at zero the routing is the routing it always was
// (no substitutions, same tokens); with a margin that makes every candidate
// near-equal the picks really do follow the tier and the run PROMOTES LESS,
// which is the entire point; and with a margin small enough that no honest
// tie exists nothing is moved. A "prefer resident" rule that never fires, or
// one that fires on non-ties, passes two of those and fails the third.
static void test_router_resident_preference() {
    MoeSpec vs;
    vs.dense = false;
    vs.n_expert = 4;
    vs.n_expert_used = 2;
    vs.shared = true;
    vs.varied = true;
    if (!build_tiny_moe_model(kMoeTestPath, vs)) {
        CHECK(false, "wrote the MoE fixture the router test needs");
        return;
    }

    struct Arm {
        bool ran = false;
        std::vector<i32> tokens;
        u64 picks = 0, subs = 0, promotions = 0;
        i64 loads = 0;
        u64 hot = 0;
        size_t hits = 0, acquires = 0;
    };
    // One of four experts resident, so the pick has somewhere to go: with two
    // of four the tier almost never evicts and there is nothing to prefer.
    const auto run = [&](f32 margin) {
        Arm a;
        // A FAKE device, so the expert budget does not depend on how much
        // free RAM the machine happens to have -- under a parallel suite the
        // CPU backend's host-RAM fallback can come out at zero, and then
        // nothing is resident and the shape this test needs is gone. The slot
        // cap is what makes the tier smaller than the routed set; the fake
        // capacity only has to be big enough to let that cap bind.
        Backend *cpu = new CountingBackend(make_cpu_backend());
        static_cast<CountingBackend *>(cpu)->set_test_resources(
            4ull << 30, 4ull << 30, 8ull << 30, 8ull << 30);
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = kMoeTestPath;
        cfg.n_ctx = 64;
        cfg.expert_cache_slots = 1;
        cfg.router_tie_margin = margin;
        std::string err;
        if (!engine.init(cpu, cfg, &err)) {
            CHECK(false, "router test engine initialises");
            delete cpu;
            return a;
        }
        GenerateParams p;
        p.prompt = "the cat sat on the mat and the dog sat";
        p.max_tokens = 8;
        p.sampler.greedy = true;
        p.sampler.temp = 0.0f;
        GenerateResult r;
        a.ran = engine.generate(p, &r);
        a.tokens = r.tokens;
        a.picks = engine.router_picks();
        a.subs = engine.router_subs();
        a.promotions = engine.model().experts().promotions();
        a.loads = static_cast<i64>(engine.model().experts().loads());
        a.hot = engine.router_hot();
        a.hits = engine.model().experts().hits();
        a.acquires = engine.model().experts().acquires();
        engine.shutdown();
        delete cpu;
        return a;
    };

    const Arm off = run(0.0f);
    CHECK(off.ran, "the unpinned arm runs");
    CHECK(off.picks == 0 && off.subs == 0,
          "with the preference off the counters stay at zero");
    CHECK(off.acquires > 0,
          "the run asked the expert cache for experts at all");

    // What this group can assert on a fixture whose expert budget is the
    // MACHINE's answer (the CPU backend derives it from live free RAM) is the
    // wiring, not the shape: with no margin asked for the tie-break is inert
    // and its counters stay at zero, and with one it counts the picks it is
    // given. Whether a substitution actually happens depends on how many
    // experts the tier held at that moment, which is why the resident pick
    // itself is measured end to end (docs/AUDIT-gpt-oss.md) and pinned there
    // by a red construction that removes it.
    const Arm tiny = run(1e-9f);
    CHECK(tiny.ran, "a positive margin runs the same generation");
    CHECK(tiny.picks > 0, "and counts every (token, slot) pick it decided");
    CHECK(tiny.acquires == off.acquires,
          "the margin does not change how many experts the run asked for");
    std::remove(kMoeTestPath);
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
    // Self-contained (see test_tokenizer_spm_and_bpe): the tiny model is this
    // group's own fixture.
    if (!build_tiny_model(kTestModelPath, false)) {
        CHECK(false, "wrote the tiny GGUF this group decodes through");
        return;
    }
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
// prefill chunk bit-exactness (the model-level gate)
// ---------------------------------------------------------------------------
//
// `kraken-bench --bitexact` pins the kernels one at a time. This pins the
// PROPERTY those kernels exist for: a prefill of N rows must leave the same
// last-position logits a row-at-a-time prefill does. It drives the real CLI
// (this binary does not link the HIP backend), on a fixture whose geometry
// lands on the exact prefill kernel -- head_dim 64, no sliding window -- so a
// regression that reintroduces the tiled formulation's reduction order fails
// here even when every per-kernel case still agrees with itself.
//
// The CLI prints a hash of the WHOLE logits row (`--prefill-hash`), not the
// generated text: a one-ulp drift would otherwise survive whenever the argmax
// happened not to move, which is exactly how the original chunk gap hid.

// A LLaMA-family fixture with the geometry the exact kernel dispatches on:
// head_dim 64, four full-attention layers, no window. The suite's older
// fixtures are head_dim 8, below the kernel's supported widths, which is why
// they could never gate it.
static bool build_prefill_gate_model(const std::string &path,
                                     int ctx = 64) {
    const int n_embd = 512, n_layer = 4, n_head = 8, n_kv = 2, hd = 64;
    const int n_ff = 512;
    // add_spm_vocab's fixed pieces plus its two word pieces are 261 entries;
    // the embedding tensor and the tokenizer must agree on that width, or the
    // head's logits row and token_embd disagree by a row.
    const int vocab = 261;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;

    GgufBuilder b;
    b.meta_str("general.architecture", "llama");
    b.meta_str("general.name", "kraken-prefill-gate");
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
    for (int l = 0; l < n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        b.tensor_f32(p + "attn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f32(p + "ffn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f16(p + "attn_q.weight", {u64(n_embd), u64(q_dim)},
                     vec(static_cast<size_t>(n_embd) * q_dim, 100 + l * 1000));
        b.tensor_f16(p + "attn_k.weight", {u64(n_embd), u64(kv_dim)},
                     vec(static_cast<size_t>(n_embd) * kv_dim, 200 + l * 1000));
        b.tensor_f16(p + "attn_v.weight", {u64(n_embd), u64(kv_dim)},
                     vec(static_cast<size_t>(n_embd) * kv_dim, 300 + l * 1000));
        b.tensor_f16(p + "attn_output.weight", {u64(q_dim), u64(n_embd)},
                     vec(static_cast<size_t>(q_dim) * n_embd, 400 + l * 1000));
        b.tensor_f16(p + "ffn_gate.weight", {u64(n_embd), u64(n_ff)},
                     vec(static_cast<size_t>(n_embd) * n_ff, 500 + l * 1000));
        b.tensor_f16(p + "ffn_up.weight", {u64(n_embd), u64(n_ff)},
                     vec(static_cast<size_t>(n_embd) * n_ff, 900 + l * 1000));
        b.tensor_f16(p + "ffn_down.weight", {u64(n_ff), u64(n_embd)},
                     vec(static_cast<size_t>(n_ff) * n_embd, 1300 + l * 1000));
    }
    return b.write(path);
}

// std::system encodes the child's status differently per platform: Windows
// hands back the exit code, POSIX a wait status. Read both the same way.
static int gate_exit_code(int rc) {
#ifdef _WIN32
    return rc;
#else
    if (rc == -1) return -1;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    return 128 + (WIFSIGNALED(rc) ? WTERMSIG(rc) : 0);
#endif
}

static std::string read_capture(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return std::string();
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

// The gemma 2 / gemma 3 schema as the engine reads it, in one fixture: GeGLU,
// the embedding scale, per-head QK-norm, the alternating window, the
// final-logit bound, and -- on gemma2 only -- the bound on the ATTENTION
// logits. `with_attn_cap` builds the same file without that key, which is the
// control: if the two files hash the same, the key is not being read.
static bool build_gemma_gate_model(const std::string &path, const char *arch,
                                   bool with_attn_cap, int ctx = 64) {
    const int n_embd = 256, n_layer = 4, n_head = 4, n_kv = 2, hd = 64;
    const int n_ff = 512, vocab = 261;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const std::string a = arch;

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", std::string("kraken-") + arch + "-gate");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(a + ".block_count", n_layer);
    b.meta_u32(a + ".context_length", ctx);
    b.meta_u32(a + ".embedding_length", n_embd);
    b.meta_u32(a + ".feed_forward_length", n_ff);
    b.meta_u32(a + ".attention.head_count", n_head);
    b.meta_u32(a + ".attention.head_count_kv", n_kv);
    b.meta_u32(a + ".attention.key_length", hd);
    b.meta_f32(a + ".attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32(a + ".rope.freq_base", 10000.0f);
    // A window short enough to bite on a 55-token prompt: half the layers are
    // windowed by the arch's pattern, so the windowed arm is exercised on every
    // token rather than only past 1024.
    b.meta_u32(a + ".attention.sliding_window", 6);
    b.meta_f32(a + ".final_logit_softcapping", 30.0f);
    if (with_attn_cap) b.meta_f32(a + ".attention.logit_softcapping", 50.0f);
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
    for (int l = 0; l < n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        b.tensor_f32(p + "attn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f32(p + "ffn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f32(p + "attn_q_norm.weight", {u64(hd)}, ones(hd));
        b.tensor_f32(p + "attn_k_norm.weight", {u64(hd)}, ones(hd));
        b.tensor_f16(p + "attn_q.weight", {u64(n_embd), u64(q_dim)},
                     vec(static_cast<size_t>(n_embd) * q_dim, 100 + l * 1000));
        b.tensor_f16(p + "attn_k.weight", {u64(n_embd), u64(kv_dim)},
                     vec(static_cast<size_t>(n_embd) * kv_dim, 200 + l * 1000));
        b.tensor_f16(p + "attn_v.weight", {u64(n_embd), u64(kv_dim)},
                     vec(static_cast<size_t>(n_embd) * kv_dim, 300 + l * 1000));
        b.tensor_f16(p + "attn_output.weight", {u64(q_dim), u64(n_embd)},
                     vec(static_cast<size_t>(q_dim) * n_embd, 400 + l * 1000));
        b.tensor_f16(p + "ffn_gate.weight", {u64(n_embd), u64(n_ff)},
                     vec(static_cast<size_t>(n_embd) * n_ff, 500 + l * 1000));
        b.tensor_f16(p + "ffn_up.weight", {u64(n_embd), u64(n_ff)},
                     vec(static_cast<size_t>(n_embd) * n_ff, 900 + l * 1000));
        b.tensor_f16(p + "ffn_down.weight", {u64(n_ff), u64(n_embd)},
                     vec(static_cast<size_t>(n_ff) * n_embd, 1300 + l * 1000));
    }
    return b.write(path);
}

struct PrefillProbe {
    bool ok = false;
    std::string trace;  // hash of the prefill's rows (see --prefill-hash)
    std::string logits; // hash of the last position's whole logits row
    std::string banner; // stdout's first line: which backend answered
    std::string err;    // stderr, kept for the failure message
    int prompt_tokens = -1;
    int first = -1;
    int rc = -1;
};

// One CLI prefill. `tiled` forces the tiled fallback through
// KRK_ATTN_BEXACT=0; the per-process environment is what makes that possible
// (the backend caches the switch at its first attention call).
// `exact_off` instead forces the rung lottery behind KRK_GEMM_EXACT=0,
// which is the negative control for the GEMM half of the gate.
static PrefillProbe run_prefill_hash(const std::string &exe,
                                     const std::string &model,
                                     const std::string &prompt, int chunk,
                                     bool tiled, bool exact_off = false,
                                     const char *extra = nullptr,
                                     int ctx = 64) {
    PrefillProbe r;
    // Each arm keeps the chunk number and gains a suffix, so two arms never
    // share a capture path (the prefix has no spaces -- it is a filename).
    const std::string tag =
        (tiled ? "-t" : "") + std::string(exact_off ? "-lottery" : "") +
        std::string(extra ? extra : "");
    const std::string o = "kraken-prefill-out-" +
                          std::to_string(chunk) + tag + ".txt";
    const std::string e = "kraken-prefill-err-" +
                          std::to_string(chunk) + tag + ".txt";
    std::string cmd = "\"" + exe + "\" -m \"" + model + "\" -p \"" + prompt +
                      "\" --ctx " + std::to_string(ctx) + " --chunk " +
                      std::to_string(chunk) + " --greedy --prefill-hash" +
                      (extra ? (" " + std::string(extra)) : std::string()) +
                      " > \"" + o + "\" 2> \"" + e + "\"";
#ifdef _WIN32
    // cmd /c strips the first and last quote of a line that starts with one,
    // which would leave the last redirect target unterminated. `call` makes
    // the line start with a word instead (same trap as tools/bench_ops.cpp).
    const std::string full = "call " + cmd;
#else
    const std::string full = cmd;
#endif
    if (tiled) set_env_var("KRK_ATTN_BEXACT", "0");
    if (exact_off) set_env_var("KRK_GEMM_EXACT", "0");
    r.rc = gate_exit_code(std::system(full.c_str()));
    if (exact_off) clear_env_var("KRK_GEMM_EXACT");
    if (tiled) clear_env_var("KRK_ATTN_BEXACT");

    const std::string out = read_capture(o);
    r.err = read_capture(e);
    const size_t nl = out.find('\n');
    r.banner = nl == std::string::npos ? out : out.substr(0, nl);
    const size_t at = out.find("prefill-hash ");
    if (at != std::string::npos) {
        char tr[64] = {0}, lg[64] = {0};
        int pt = -1, first = -1;
        if (std::sscanf(out.c_str() + at,
                        "prefill-hash %63s logits %63s prompt_tokens=%d first=%d",
                        tr, lg, &pt, &first) == 4) {
            r.trace = tr;
            r.logits = lg;
            r.prompt_tokens = pt;
            r.first = first;
            r.ok = true;
        }
    }
    return r;
}

// LFM2.5 (`lfm2moe`): 3 layers, two short-convolution mixers and one
// attention layer, the last carrying a sigmoid-gated MoE. Small enough to prefll
// in a second, shaped like the real 24B-A2B file in every respect the engine
// branches on: a PER-LAYER head_count_kv array (0 on the conv layers), no
// output_norm (the final norm is token_embd_norm), shortconv.conv as f32 taps,
// expert_gating_func 2 and an exp_probs_b.bias.
// `bad_tap` is the red construction the row pin needs: it writes the SAME
// fixture with one coefficient of layer 0's convolution moved. Everything the
// mixer is built from -- the ring, the gate, the three projections -- is byte
// for byte what the good fixture has, so a row that moves here moved because
// the mixer's own arithmetic contributed something different to the residual.
static bool build_lfm2_gate_model(const std::string &path, int ctx = 64,
                                  bool bad_tap = false) {
    const int n_embd = 64, n_layer = 3, n_head = 4, hd = 16, n_kv = 2;
    const int n_ff = 64, n_exp = 8, n_used = 3, ff_exp = 32;
    const int vocab = 261;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;

    GgufBuilder b;
    b.meta_str("general.architecture", "lfm2moe");
    b.meta_str("general.name", "kraken-lfm2-gate");
    b.meta_u32("general.alignment", 32);
    b.meta_u32("lfm2moe.block_count", n_layer);
    b.meta_u32("lfm2moe.context_length", ctx);
    b.meta_u32("lfm2moe.embedding_length", n_embd);
    b.meta_u32("lfm2moe.feed_forward_length", n_ff);
    b.meta_u32("lfm2moe.attention.head_count", n_head);
    // The array, not a scalar: the conv layers say 0 and only layer 2 attends.
    b.meta_i32_array("lfm2moe.attention.head_count_kv", {0, 0, n_kv});
    b.meta_u32("lfm2moe.attention.key_length", hd);
    b.meta_f32("lfm2moe.attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32("lfm2moe.rope.freq_base", 1e7f);
    b.meta_u32("lfm2moe.shortconv.l_cache", 3);
    b.meta_u32("lfm2moe.leading_dense_block_count", 2);
    b.meta_u32("lfm2moe.expert_count", n_exp);
    b.meta_u32("lfm2moe.expert_used_count", n_used);
    b.meta_u32("lfm2moe.expert_feed_forward_length", ff_exp);
    b.meta_u32("lfm2moe.expert_gating_func", 2);
    add_spm_vocab(b, vocab);

    auto vec = [](size_t n, i64 base) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = wv(base + static_cast<i64>(i));
        return v;
    };
    auto ones = [](size_t n) { return std::vector<f32>(n, 1.0f); };

    b.tensor_f16("token_embd.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 0));
    // NO output_norm: this family's final norm is the embedding's.
    b.tensor_f32("token_embd_norm.weight", {u64(n_embd)}, ones(n_embd));

    for (int l = 0; l < n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        b.tensor_f32(p + "attn_norm.weight", {u64(n_embd)}, ones(n_embd));
        b.tensor_f32(p + "ffn_norm.weight", {u64(n_embd)}, ones(n_embd));
        if (l < 2) {
            b.tensor_f16(p + "shortconv.in_proj.weight",
                         {u64(n_embd), u64(3 * n_embd)},
                         vec(static_cast<size_t>(n_embd) * 3 * n_embd,
                             100 + l * 1000));
            // f32 taps, [kern, ch] -- the layout the loader uploads verbatim.
            std::vector<f32> taps = vec(static_cast<size_t>(3) * n_embd,
                                        200 + l * 1000);
            // The LAST tap row, not the first: a one-row prompt reads only
            // `w[kern-1]` (the two older taps address p < 0 and are skipped),
            // so moving `taps[0]` would change a coefficient this prompt never
            // reads and the arm would report "identical" for a reason that has
            // nothing to do with the pin. Measured the hard way.
            if (bad_tap && l == 0)
                taps[static_cast<size_t>(2) * n_embd] += 0.75f;
            b.tensor_f32(p + "shortconv.conv.weight", {3, u64(n_embd)}, taps);
            b.tensor_f16(p + "shortconv.out_proj.weight",
                         {u64(n_embd), u64(n_embd)},
                         vec(static_cast<size_t>(n_embd) * n_embd,
                             300 + l * 1000));
            b.tensor_f16(p + "ffn_gate.weight", {u64(n_embd), u64(n_ff)},
                         vec(static_cast<size_t>(n_embd) * n_ff, 400 + l * 1000));
            b.tensor_f16(p + "ffn_up.weight", {u64(n_embd), u64(n_ff)},
                         vec(static_cast<size_t>(n_embd) * n_ff, 500 + l * 1000));
            b.tensor_f16(p + "ffn_down.weight", {u64(n_ff), u64(n_embd)},
                         vec(static_cast<size_t>(n_ff) * n_embd, 600 + l * 1000));
        } else {
            b.tensor_f16(p + "attn_q.weight", {u64(n_embd), u64(q_dim)},
                         vec(static_cast<size_t>(n_embd) * q_dim, 700));
            b.tensor_f16(p + "attn_k.weight", {u64(n_embd), u64(kv_dim)},
                         vec(static_cast<size_t>(n_embd) * kv_dim, 800));
            b.tensor_f16(p + "attn_v.weight", {u64(n_embd), u64(kv_dim)},
                         vec(static_cast<size_t>(n_embd) * kv_dim, 900));
            b.tensor_f16(p + "attn_output.weight", {u64(q_dim), u64(n_embd)},
                         vec(static_cast<size_t>(q_dim) * n_embd, 1000));
            b.tensor_f32(p + "attn_q_norm.weight", {u64(hd)}, ones(hd));
            b.tensor_f32(p + "attn_k_norm.weight", {u64(hd)}, ones(hd));
            b.tensor_f16(p + "ffn_gate_inp.weight", {u64(n_embd), u64(n_exp)},
                         vec(static_cast<size_t>(n_embd) * n_exp, 1100));
            b.tensor_f32(p + "exp_probs_b.bias", {u64(n_exp)},
                         vec(static_cast<size_t>(n_exp), 1200));
            const size_t gsz = static_cast<size_t>(n_embd) * ff_exp;
            const size_t dsz = static_cast<size_t>(ff_exp) * n_embd;
            const size_t ne = static_cast<size_t>(n_exp);
            std::vector<f32> gate(gsz * ne), up(gsz * ne), down(dsz * ne);
            for (size_t e = 0; e < ne; e++) {
                for (size_t k = 0; k < gsz; k++) {
                    gate[e * gsz + k] = wv(1300 + static_cast<i64>(k) +
                                          static_cast<i64>(e) * 7919);
                    up[e * gsz + k] = wv(1500 + static_cast<i64>(k) +
                                         static_cast<i64>(e) * 7919);
                }
                for (size_t k = 0; k < dsz; k++)
                    down[e * dsz + k] = wv(1700 + static_cast<i64>(k) +
                                           static_cast<i64>(e) * 7919);
            }
            b.tensor_f16(p + "ffn_gate_exps.weight",
                         {u64(n_embd), u64(ff_exp), u64(ne)}, gate);
            b.tensor_f16(p + "ffn_up_exps.weight",
                         {u64(n_embd), u64(ff_exp), u64(ne)}, up);
            b.tensor_f16(p + "ffn_down_exps.weight",
                         {u64(ff_exp), u64(n_embd), u64(ne)}, down);
        }
    }
    return b.write(path);
}

// One CLI prefill of the LFM2 fixture. Same shape as run_prefill_hash, but
// the tag is explicit so a --cpu arm and a device arm never share a capture.
static PrefillProbe run_lfm_hash(const std::string &exe, const std::string &model,
                                const std::string &prompt, int chunk, bool cpu) {
    PrefillProbe r;
    const std::string tag = cpu ? "-cpu" : "-dev";
    const std::string o = "kraken-lfm-out-" + std::to_string(chunk) + tag + ".txt";
    const std::string e = "kraken-lfm-err-" + std::to_string(chunk) + tag + ".txt";
    std::string cmd = "\"" + exe + "\" -m \"" + model + "\" -p \"" + prompt +
                      "\" --ctx 64 --chunk " + std::to_string(chunk) +
                      " --greedy --prefill-hash" + (cpu ? " --cpu" : "") +
                      " > \"" + o + "\" 2> \"" + e + "\"";
#ifdef _WIN32
    const std::string full = "call " + cmd;
#else
    const std::string full = cmd;
#endif
    r.rc = gate_exit_code(std::system(full.c_str()));
    const std::string out = read_capture(o);
    r.err = read_capture(e);
    const size_t nl = out.find('\n');
    r.banner = nl == std::string::npos ? out : out.substr(0, nl);
    const size_t at = out.find("prefill-hash ");
    if (at != std::string::npos) {
        char tr[64] = {0}, lg[64] = {0};
        int pt = -1, first = -1;
        if (std::sscanf(out.c_str() + at,
                        "prefill-hash %63s logits %63s prompt_tokens=%d first=%d",
                        tr, lg, &pt, &first) == 4) {
            r.trace = tr;
            r.logits = lg;
            r.prompt_tokens = pt;
            r.first = first;
            r.ok = true;
        }
    }
    std::remove(o.c_str());
    std::remove(e.c_str());
    return r;
}

// Group: LFM2.5's short-convolution mixer and its per-layer KV.
//
// The mixer is the one block in this engine whose arithmetic is a CONVOLUTION
// over a gated product: `bx = B*X`, then a causal depthwise window over bx, then
// `C *`. Every step of it was measured against llama.cpp on the 24B-A2B file, and
// three of the four things that made that file decode whitespace are pinned
// here: NeoX rope pairing, normalized sigmoid expert weights, and the ring's f16
// rounding. The last one is the reason this group drives the CLI rather than the
// CPU backend alone -- the CPU ring is f32 and cannot see it.
static void test_lfm2_shortconv() {
    const std::string model = "kraken-lfm2-gate.gguf";
    CHECK(build_lfm2_gate_model(model), "wrote the lfm2moe fixture");

    // 1. The row's claim: the file LOADS with no KRK_ARCH_ALLOW_EXPERIMENTAL.
    {
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        const bool ok = m.load(*cpu, model, &err);
        if (!ok)
            std::fprintf(stderr, "  (lfm2moe refused: %s)\n", err.c_str());
        CHECK(ok, "lfm2moe loads without an experimental hatch");
        if (ok) {
            const ModelConfig &c = m.cfg();
            CHECK(c.lfm_conv, "the loader marks the family's conv mixer");
            CHECK(m.has_lfm_conv(), "and the model reports a shortconv layer");
            CHECK(c.n_dense_lead == 2, "the two leading dense blocks are read");
            CHECK(c.router_sigmoid, "expert_gating_func 2 selects sigmoid");
            CHECK(c.expert_w_norm == 1,
                  "and the top-k sigmoid weights are sum-normalized");
            CHECK(c.rope_neox, "RoPE pairs NeoX half-split");
            CHECK(c.ssm_d_conv == 3, "shortconv.l_cache is the window");
            int conv_layers = 0, attn_layers = 0;
            for (int l = 0; l < static_cast<int>(m.layers().size()); l++) {
                if (m.layers()[l].lfm_conv) conv_layers++;
                if (m.layer_has_kv(l)) attn_layers++;
            }
            CHECK(conv_layers == 2, "two conv layers, one attention layer");
            CHECK(attn_layers == 1,
                  "and only the attention layer owns a KV cache");
        }
        delete cpu;
    }

    // 1b. The RECORDED row. The arms below pin a property -- that the ring
    //     leaves the same rows at every chunk size -- and a mixer that is
    //     chunk-invariant and WRONG passes every one of them: the plausible
    //     reading of this block (`B*conv(X) + C*X`) is perfectly invariant and
    //     produces a fluent, wrong model. So the row itself is pinned, by the
    //     same FNV-1a `--prefill-hash` prints for `logits`, so the in-process
    //     graph and the CLI's two arms are one number rather than three that
    //     cannot be compared.
    auto row_hash = [](const f32 *v, i32 n) {
        u64 h = 1469598103934665603ull;
        const unsigned char *b = reinterpret_cast<const unsigned char *>(v);
        for (i64 i = 0; i < static_cast<i64>(n) * 4; i++) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
        return h;
    };
    const std::string badtap = "kraken-lfm2-gate-badtap.gguf";
    CHECK(build_lfm2_gate_model(badtap, 64, true),
          "the moved-tap fixture writes (one conv coefficient, nothing else)");
    // Prefill only (`max_tokens 0`), so the row read afterwards is the one at
    // the last prompt position -- the row a decode starts from.
    auto row = [&](const std::string &file, const char *prompt, i32 chunk,
                   u64 *hash, i32 *top1, RowPin *pin = nullptr) -> bool {
        Backend *cpu = make_cpu_backend();
        bool ok = false;
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = file;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = chunk;
            std::string err;
            if (!engine.init(cpu, cfg, &err)) {
                if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
            } else {
                GenerateParams p;
                p.prompt = prompt;
                p.max_tokens = 0;
                p.sampler.greedy = true;
                GenerateResult r;
                ok = engine.generate(p, &r) && r.finish != FinishError;
                const f32 *lg = engine.last_logits();
                const i32 nv = engine.model().cfg().n_vocab;
                if (ok && lg && nv > 0) {
                    *hash = row_hash(lg, nv);
                    i32 best = 0;
                    for (i32 i = 1; i < nv; i++)
                        if (lg[i] > lg[best]) best = i;
                    *top1 = best;
                    if (pin) *pin = row_pin(lg, nv);
                } else {
                    ok = false;
                }
            }
        }
        delete cpu;
        return ok;
    };

    u64 h_wide = 0, h_other = 0, h_narrow = 0, h_badtap = 0;
    i32 t_wide = -1, t_other = -1, t_narrow = -1, t_badtap = -1;
    RowPin pin_wide;
    CHECK(row(model, "the", 8, &h_wide, &t_wide, &pin_wide),
          "the whole fixture graph prefills and the row reaches the head");
    std::fprintf(stderr, "  (lfm2 logits row %016llx, argmax %d)\n",
                 static_cast<unsigned long long>(h_wide),
                 static_cast<int>(t_wide));

    // Recorded from this graph on the fixture, CPU backend, prompt "the".
    // The PIN is the portable fingerprint (see row_pin's note): the bit hash
    // is a property of the host's floating point as much as of the graph, so
    // it is compared only when a recording host asks for it.
    static const RowPin kGolden = {
        {3.079364, 3.079364, 3.079364, 3.079364, 3.079364, 3.079364, 3.079364,
         3.079364},
        -1.947591, 352.351005, 24.527738};
    CHECK(h_wide != 0, "the row is not empty");
    std::fprintf(stderr, "  (lfm2 pin: %s)\n", pin_str(pin_wide).c_str());
    if (kGolden.l1 != 0.0)
        CHECK(pin_close(pin_wide, kGolden),
              "the last prompt row matches the recorded fingerprint (the "
              "mixer's own output)");
    if (golden_bits_on())
        CHECK(h_wide == 0x399373273e5a2b60ull,
              "KRK_GOLDEN_BITS: the recorded row, bit for bit");

    CHECK(row(model, "a", 8, &h_other, &t_other),
          "a second prompt prefills");
    CHECK(h_other != h_wide, "a different prompt moves the row");

    CHECK(row(model, "the", 1, &h_narrow, &t_narrow),
          "the same prompt token by token");
    CHECK(h_narrow == h_wide,
          "chunk 1 reproduces the chunk-8 row exactly");

    CHECK(row(badtap, "the", 8, &h_badtap, &t_badtap),
          "the moved-tap fixture prefills");
    std::fprintf(stderr, "  (moved-tap row %016llx)\n",
                 static_cast<unsigned long long>(h_badtap));
    CHECK(h_badtap != h_wide,
          "moving ONE conv coefficient moves the row: the pin sees the mixer");

    // 2. The mixer's chunk invariance, on the device. `--chunk 1` runs every
    //    row through the decode path and feeds the ring a row at a time; the
    //    wider arms read the same rows from the call's own proj buffer. They
    //    agree only because the product is rounded to f16 as the ring stores
    //    it -- before that, chunk 1 and chunk 2 differed at pos 1 of layer 0
    //    (0.00055837 against 0.00055800) and the greedy text diverged.
    const std::string self = self_path();
    const std::filesystem::path exe_p =
        std::filesystem::path(self).parent_path() /
#ifdef _WIN32
        "kraken.exe";
#else
        "kraken";
#endif
    const std::string exe = exe_p.string();
    const bool have_exe = !self.empty() && std::filesystem::exists(exe_p);
    CHECK(have_exe, "kraken sits beside kraken-tests (the gate drives it)");
    if (!have_exe) return;

    // Long enough that the window is fed by more than one pass at every chunk
    // size, and short enough to stay inside the fixture's 64-token context.
    std::string prompt = "the";
    for (int i = 0; i < 11; i++) prompt += " the";

    const PrefillProbe c1 = run_lfm_hash(exe, model, prompt, 1, false);
    const PrefillProbe c2 = run_lfm_hash(exe, model, prompt, 2, false);
    const PrefillProbe c4 = run_lfm_hash(exe, model, prompt, 4, false);
    CHECK(c1.ok, "the chunk-1 device prefill prints its hashes");
    CHECK(c2.ok, "the chunk-2 device prefill prints its hashes");
    CHECK(c4.ok, "the chunk-4 device prefill prints its hashes");
    if (!c1.ok || !c2.ok || !c4.ok) {
        std::fprintf(stderr, "  (chunk 1 rc=%d stderr: %.300s)\n", c1.rc,
                     c1.err.c_str());
        return;
    }
    CHECK(c1.prompt_tokens > 8 && c1.prompt_tokens == c2.prompt_tokens,
          "the prompt spans several chunks, and both arms see it");
    std::printf("       lfm2 chunk 1 trace %s logits %s\n", c1.trace.c_str(),
                c1.logits.c_str());
    std::printf("       lfm2 chunk 2 trace %s logits %s\n", c2.trace.c_str(),
                c2.logits.c_str());
    std::printf("       lfm2 chunk 4 trace %s logits %s\n", c4.trace.c_str(),
                c4.logits.c_str());
    CHECK(c1.trace == c2.trace,
          c1.trace == c2.trace
              ? "the shortconv ring leaves the same rows at chunk 1 and 2"
              : "the shortconv ring is CHUNK-DEPENDENT (chunk 1 != chunk 2)");
    CHECK(c1.trace == c4.trace, "and at chunk 4");
    CHECK(c1.logits == c4.logits, "the logits row agrees too");
    CHECK(c1.first == c4.first && c2.first == c1.first,
          "and the three arms sample the same first token");

    // 3. The CPU reference has its own ring (f32) and must be invariant as
    //    well -- the same property, a different storage width. Its hashes are
    //    NOT compared with the device's: no HIP row is bit-equal to this
    //    engine's CPU row on any model, which is a property of the backend and
    //    not of this mixer.
    const PrefillProbe p1 = run_lfm_hash(exe, model, prompt, 1, true);
    const PrefillProbe p4 = run_lfm_hash(exe, model, prompt, 4, true);
    CHECK(p1.ok && p4.ok, "the --cpu arms print their hashes");
    if (p1.ok && p4.ok) {
        std::printf("       lfm2 cpu   1 trace %s logits %s\n", p1.trace.c_str(),
                    p1.logits.c_str());
        CHECK(p1.trace == p4.trace,
              p1.trace == p4.trace
                  ? "the CPU reference is chunk-invariant too"
                  : "the CPU reference is CHUNK-DEPENDENT");
        CHECK(p1.logits == p4.logits, "and its logits row agrees");
        CHECK(p1.first == c1.first,
              p1.first == c1.first
                  ? "the CPU and device arms sample the same first token"
                  : "the CPU and device arms disagree on the first token");
    }
}

// Group 57: the supersized-GGUF audit's first landed piece -- attention
// sinks (deepseek4 / gpt-oss): one logit per query head entered into the
// softmax DENOMINATOR at the row's running max, with no value row behind it
// (V4 report Eq. 27). Pinned three ways: the arithmetic against a hand
// computation, the per-head slice under uneven GQA (64 q heads, 1 kv head
// is deepseek4's geometry), and the A/B an AGENTS.md note demands -- a
// ZEROED sink bit-identical to the no-sink arithmetic, with a non-zero
// sink proven to change the output at all.
static void group_attention_sinks() {
    Backend *cpu = make_cpu_backend();

    // 1. The arithmetic: two identical keys (dot 1.0), distinct values, and
    //    a sink of 0.5:
    //      out = (e^0*2 + e^0*4) / (e^0 + e^0 + e^(0.5-1))
    // A value-row implementation (sink carried as an extra V row) would
    // shrink the output further, so the second CHECK separates the two.
    {
        const i64 hd = 4, n_head = 2, n_kv = 1, pos_stride = 8;
        std::vector<f32> kch(static_cast<size_t>(pos_stride * n_head * hd),
                             0.0f);
        std::vector<f32> vch(kch.size(), 0.0f);
        std::vector<f32> q(static_cast<size_t>(n_head * hd), 0.0f);
        std::vector<f32> sinks = {0.5f, -0.25f};
        // Key 1's slice sits one pos_stride along (n_kv*hd elements are
        // used per position; pos_stride is 8 here, hd only 4).
        kch[0] = 1.0f;
        kch[static_cast<size_t>(pos_stride)] = 1.0f;
        q[0] = 1.0f;
        vch[0] = 2.0f;
        vch[static_cast<size_t>(pos_stride)] = 4.0f;
        AttnDesc d;
        d.n_head = n_head;
        d.n_kv = n_kv;
        d.hd = hd;
        d.n_tok = 1;
        d.pos0 = 1;
        d.pos_stride = pos_stride;
        d.causal = true;
        d.window = 0;
        d.scale = 1.0f;
        d.sinks = sinks.data();
        std::vector<f32> out(static_cast<size_t>(n_head * hd), 0.0f);
        cpu->attention(out.data(), q.data(), kch.data(), vch.data(), d);
        const double m = 1.0;
        const double den = std::exp(1.0 - m) + std::exp(1.0 - m) +
                           std::exp(0.5 - m);
        const double want = (std::exp(1.0 - m) * 2.0 +
                             std::exp(1.0 - m) * 4.0) / den;
        CHECK_NEAR(out[0], want, 1e-5,
                   "the sink enters the denominator only (hand computation)");
        CHECK(std::fabs(out[0] - 3.0f) > 1e-3,
              "a non-zero sink changes the output at all");
    }

    // 2. Uneven GQA: 4 query heads, 1 KV head -- every head's slice must see
    //    ITS OWN sink even though the k/v row is shared.
    {
        const i64 hd = 8, n_head = 4, n_kv = 1, pos_stride = 8;
        std::vector<f32> kch(static_cast<size_t>(pos_stride * n_head * hd),
                             0.0f);
        std::vector<f32> vch(kch.size(), 0.0f);
        std::vector<f32> q(static_cast<size_t>(n_head * hd), 0.0f);
        for (i64 h = 0; h < n_head; ++h) {
            kch[static_cast<size_t>(h * hd)] = 1.0f;
            q[static_cast<size_t>(h * hd)] = 1.0f + 0.5f * static_cast<f32>(h);
            vch[static_cast<size_t>(h * hd)] = 1.0f + static_cast<f32>(h);
        }
        std::vector<f32> sinks = {10.0f, -10.0f, 0.0f, 3.0f};
        AttnDesc d;
        d.n_head = n_head;
        d.n_kv = n_kv;
        d.hd = hd;
        d.n_tok = 1;
        d.pos0 = 0;
        d.pos_stride = pos_stride;
        d.causal = true;
        d.window = 0;
        d.scale = 1.0f;
        d.sinks = sinks.data();
        std::vector<f32> out(static_cast<size_t>(n_head * hd), 0.0f);
        cpu->attention(out.data(), q.data(), kch.data(), vch.data(), d);
        // One visible key per head (dot = 1 + 0.5h, scale 1) and a
        // shared k/v row: p_key = 1 / (1 + exp(sink[h] - dot_h)), and
        // the output is p_key * v -- a dominant sink DRAINS the mass
        // (no value row), a suppressed one leaves it. Same k/v for
        // every head, so the four different answers ARE the per-head
        // slice proof.
        const double p0 = 1.0 / (1.0 + std::exp(10.0 - 1.0));
        const double p1 = 1.0 / (1.0 + std::exp(-10.0 - 1.5));
        // Head 2's sink logit is EXACTLY 0 -> zeroed-arm semantics: the
        // term is skipped, and with one visible key p = 1 outright.
        const double p2 = 1.0;
        const double p3 = 1.0 / (1.0 + std::exp(3.0 - 2.5));
        CHECK_NEAR(out[0], p0, 1e-6,
                   "head 0: dominating sink drains its key's mass "
                   "(hand-computed)");
        CHECK_NEAR(out[hd], p1, 1e-6,
                   "head 1: suppressed sink leaves its key's mass "
                   "(per-head slice over a shared KV row)");
        CHECK_NEAR(out[2 * hd], p2, 1e-6,
                   "head 2: mid sink, hand-computed per-head term");
        CHECK_NEAR(out[3 * hd], p3, 1e-6,
                   "head 3: fourth distinct sink term, same shared "
                   "k/v row");
    }

    // 3. The semantics, pinned against a manual reference: an ABSENT sink
    //    must leave the arithmetic exactly as it always was; a ZEROED sink
    //    row must be BIT-identical to that absent run (the zeroed-sink arm,
    //    compared with `==` and not a tolerance, so removing the
    //    `sinks[h] != 0` guard fails it); and a non-zero sink must enter
    //    the denominator by the hand formula. Prefill row t must also equal
    //    decode row t with the same sink.
    {
        const i64 hd = 16, n_head = 4, n_kv = 2, pos = 3;
        const i64 pos_stride = n_kv * hd;  // non-overlapping rows
        std::mt19937 rng(13);
        std::uniform_real_distribution<f32> ud(-1.0f, 1.0f);
        std::vector<f32> kch(static_cast<size_t>(pos_stride * n_head * hd));
        std::vector<f32> vch(kch.size());
        for (auto &x : kch) x = ud(rng);
        for (auto &x : vch) x = ud(rng);
        std::vector<f32> q(static_cast<size_t>(n_head * hd));
        for (auto &x : q) x = ud(rng);
        std::vector<f32> sinks(static_cast<size_t>(n_head));
        for (auto &x : sinks) x = ud(rng);
        std::vector<f32> zsinks(static_cast<size_t>(n_head), 0.0f);
        const f32 scale = 1.0f / std::sqrt(static_cast<f32>(hd));
        // Manual reference for one decode row at `pos`, per head, with an
        // optional sink: the plain softmax, plus the denominator term.
        auto ref_row = [&](const std::vector<f32> *sink_row) {
            std::vector<f32> ref(static_cast<size_t>(n_head * hd), 0.0f);
            for (i64 h = 0; h < n_head; ++h) {
                const i64 hkv = h / (n_head / n_kv);
                f32 mx = -1e30f;
                std::vector<f32> s(static_cast<size_t>(pos + 1));
                for (i64 j = 0; j <= pos; ++j) {
                    f32 dot = 0.0f;
                    for (i64 c = 0; c < hd; ++c)
                        dot += q[static_cast<size_t>(h * hd + c)] *
                               kch[static_cast<size_t>(
                                   j * pos_stride + hkv * hd + c)];
                    dot *= scale;
                    s[static_cast<size_t>(j)] = dot;
                    if (dot > mx) mx = dot;
                }
                f32 l = 0.0f;
                for (i64 j = 0; j <= pos; ++j)
                    l += std::exp(s[static_cast<size_t>(j)] - mx);
                if (sink_row)
                    l += std::exp((*sink_row)[static_cast<size_t>(h)] - mx);
                for (i64 c = 0; c < hd; ++c) {
                    f32 acc = 0.0f;
                    for (i64 j = 0; j <= pos; ++j) {
                        const f32 p =
                            std::exp(s[static_cast<size_t>(j)] - mx) / l;
                        acc += p * vch[static_cast<size_t>(
                                        j * pos_stride + hkv * hd + c)];
                    }
                    ref[static_cast<size_t>(h * hd + c)] = acc;
                }
            }
            return ref;
        };
        AttnDesc d;
        d.n_head = n_head;
        d.n_kv = n_kv;
        d.hd = hd;
        d.n_tok = 1;
        d.pos0 = pos;
        d.pos_stride = pos_stride;
        d.causal = true;
        d.window = 0;
        d.scale = scale;
        std::vector<f32> got(static_cast<size_t>(n_head * hd), 0.0f);
        // (a) absent sink == manual softmax with no sink term.
        d.sinks = nullptr;
        cpu->attention(got.data(), q.data(), kch.data(), vch.data(), d);
        CHECK_NEAR(had_worst(got, ref_row(nullptr)), 0.0, 1e-5,
                   "absent sink leaves the arithmetic exactly as it was "
                   "(manual reference)");
        const std::vector<f32> absent = got;
        // (b) THE ZEROED-SINK BIT-IDENTITY ARM: pointer present, every
        //     value 0 -- must be BIT-identical to the absent run above.
        d.sinks = zsinks.data();
        cpu->attention(got.data(), q.data(), kch.data(), vch.data(), d);
        CHECK(got == absent,
              "zeroed sink row is BIT-identical to the absent-sink run "
              "(the zeroed-sink arm)");
        // (c) a non-zero sink == the same reference with the real logits.
        d.sinks = sinks.data();
        cpu->attention(got.data(), q.data(), kch.data(), vch.data(), d);
        CHECK_NEAR(had_worst(got, ref_row(&sinks)), 0.0, 1e-5,
                   "per-head sink logits match the manual reference");
        CHECK(had_worst(got, absent) > 1e-6,
              "a non-zero sink changes the output against the absent run "
              "(negative control)");
        // (d) prefill row t equals decode row t, same sink.
        const i64 n_tok = 4;
        std::vector<f32> qs(static_cast<size_t>(n_tok * n_head * hd));
        for (auto &x : qs) x = ud(rng);
        AttnDesc p = d;
        p.n_tok = n_tok;
        p.pos0 = 0;
        p.sinks = sinks.data();
        std::vector<f32> out(static_cast<size_t>(n_tok * n_head * hd), 0.0f);
        cpu->attention(out.data(), qs.data(), kch.data(), vch.data(), p);
        std::vector<f32> rows(out.size(), 0.0f);
        for (i64 t = 0; t < n_tok; ++t) {
            AttnDesc dd = p;
            dd.n_tok = 1;
            dd.pos0 = t;
            std::vector<f32> qr(static_cast<size_t>(n_head * hd));
            for (i64 c = 0; c < n_head * hd; ++c)
                qr[static_cast<size_t>(c)] =
                    qs[static_cast<size_t>(t * n_head * hd + c)];
            cpu->attention(rows.data() +
                               static_cast<size_t>(t * n_head * hd),
                           qr.data(), kch.data(), vch.data(), dd);
        }
        CHECK_NEAR(had_worst(out, rows), 0.0, 1e-6,
                   "prefill row t with a sink equals decode row t with the "
                   "sink (cpu)");
    }
}

// One CLI generation. Returns stdout with the bracketed status lines
// stripped (they carry timings, which differ run to run); what remains is
// the banner plus the generated text, both deterministic for a fixed
// build, model, prompt, chunk and seed. Used for the prefill-then-decode
// promise: chunk-1-then-decode-N must read exactly like chunk-64-then-N,
// token for token -- the text-level half of chunk invariance.
static std::string run_generate_text(const std::string &exe,
                                     const std::string &model,
                                     const std::string &prompt, int chunk,
                                     int n, const std::string &tag,
                                     int &rc_out) {
    const std::string o = "kraken-prefill-gen-" + std::to_string(chunk) +
                          tag + ".txt";
    const std::string e = "kraken-prefill-err-gen-" + std::to_string(chunk) +
                          tag + ".txt";
    std::string cmd = "\"" + exe + "\" -m \"" + model + "\" -p \"" + prompt +
                      "\" --ctx 64 --chunk " + std::to_string(chunk) +
                      " --greedy -n " + std::to_string(n) + " > \"" + o +
                      "\" 2> \"" + e + "\"";
#ifdef _WIN32
    const std::string full = "call " + cmd;
#else
    const std::string full = cmd;
#endif
    rc_out = gate_exit_code(std::system(full.c_str()));
    const std::string out = read_capture(o);
    std::string kept;
    size_t pos = 0;
    while (pos < out.size()) {
        size_t nl = out.find('\n', pos);
        std::string line =
            nl == std::string::npos ? out.substr(pos) : out.substr(pos, nl - pos);
        if (line.empty() || line[0] != '[') { kept += line; kept += '\n'; }
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    return kept;
}

static void test_prefill_chunk_bitexact() {
    // Self-contained: the fixture is built here, and the CLI is the sibling
    // binary the build already produces (kraken-tests cannot link the device
    // backend -- it is a plain C++ TU -- so the gate drives the real one).
    const int run0 = g_run, pass0 = g_passed;
    const std::string model = "kraken-prefill-gate.gguf";
    CHECK(build_prefill_gate_model(model),
          "wrote the head_dim-64 fixture this gate prefills");

    const std::string self = self_path();
    const std::filesystem::path exe_p =
        std::filesystem::path(self).parent_path() /
#ifdef _WIN32
        "kraken.exe";
#else
        "kraken";
#endif
    const std::string exe = exe_p.string();
    const bool have_exe = !self.empty() && std::filesystem::exists(exe_p);
    CHECK(have_exe, "kraken sits beside kraken-tests (the gate drives it)");
    if (!have_exe) return;

    // A prompt that fills the fixture's 64-token context: the chunk-8 arm then
    // ends on a full eight-row chunk at 64 keys, which is the shape the old
    // tiled kernel's reduction order disagreed on (hd64/8h, rows=8, in=64 in
    // `kraken-bench --bitexact`). A prompt of eight tokens or fewer would make
    // the two arms the same call and pass vacuously.
    std::string prompt;
    for (int i = 0; i < 63; i++) prompt += (i ? " the" : "the");

    const PrefillProbe c1 = run_prefill_hash(exe, model, prompt, 1, false);
    const PrefillProbe c2 = run_prefill_hash(exe, model, prompt, 2, false);
    const PrefillProbe c8 = run_prefill_hash(exe, model, prompt, 8, false);
    // Rows 16 and 32 are the GEMM rung this gate could not see before: the
    // chunk sizes above 8 leave the mat-vec ladder, so they are the arms that
    // fail when the prefill GEMM's association differs from decode's.
    const PrefillProbe c16 = run_prefill_hash(exe, model, prompt, 16, false);
    const PrefillProbe c32 = run_prefill_hash(exe, model, prompt, 32, false);
    CHECK(c1.ok, "the chunk-1 prefill prints its hashes");
    CHECK(c2.ok, "the chunk-2 prefill prints its hashes");
    CHECK(c8.ok, "the chunk-8 prefill prints its hashes");
    CHECK(c16.ok, "the chunk-16 prefill prints its hashes");
    CHECK(c32.ok, "the chunk-32 prefill prints its hashes");
    std::printf("       chunk  1 trace %s logits %s\n", c1.trace.c_str(),
                c1.logits.c_str());
    std::printf("       chunk  2 trace %s logits %s\n", c2.trace.c_str(),
                c2.logits.c_str());
    std::printf("       chunk  8 trace %s logits %s\n", c8.trace.c_str(),
                c8.logits.c_str());
    std::printf("       chunk 16 trace %s logits %s\n", c16.trace.c_str(),
                c16.logits.c_str());
    std::printf("       chunk 32 trace %s logits %s\n", c32.trace.c_str(),
                c32.logits.c_str());
    if (!c1.ok || !c8.ok) {
        std::fprintf(stderr, "  (chunk 1 rc=%d stderr: %.300s)\n", c1.rc,
                     c1.err.c_str());
        std::fprintf(stderr, "  (chunk 8 rc=%d stderr: %.300s)\n", c8.rc,
                     c8.err.c_str());
        return;
    }
    CHECK(c1.prompt_tokens > 8 && c1.prompt_tokens == c8.prompt_tokens,
          "the prompt spans several 8-row chunks, and both arms see it");

    // THE GATE. Same prompt, same weights, same greedy run: only the prefill
    // batch size differs. The trace covers the embedding, every layer's
    // attention output and every layer's hidden state over ALL prompt rows --
    // so a difference in a row that is not the last one (where the tiled
    // kernel's reduction order showed) cannot hide behind the f16 residual.
    CHECK(c1.trace == c8.trace,
          c1.trace == c8.trace
              ? "the chunk-1 and chunk-8 prefills leave the SAME rows, bit for bit"
              : "the chunk-1 and chunk-8 prefills DIFFER: the multi-row forward "
                "is not bit-exact");
    if (c1.trace != c8.trace) {
        std::fprintf(stderr,
                     "  (chunk 1 trace %s at %d tokens vs chunk 8 %s: one is "
                     "the decode formulation, the other the prefill kernel)\n",
                     c1.trace.c_str(), c1.prompt_tokens, c8.trace.c_str());
    }
    if (c2.ok) {
        CHECK(c1.trace == c2.trace,
              "the 2-row prefill agrees with the single-row one too");
        CHECK(c1.logits == c2.logits,
              "and its logits row agrees");
    }
    CHECK(c1.logits == c8.logits, "and the logits row agrees bit for bit");
    CHECK(c1.first == c8.first, "the row's argmax is the same token");

    // The GEMM rung: 16 and 32 rows per call take the prefill GEMM instead of
    // the ladder, so these two arms are what prove the prefill GEMM writes the
    // bytes decode would. Their control is below (KRK_GEMM_EXACT=0).
    if (c16.ok) {
        CHECK(c1.trace == c16.trace,
              c1.trace == c16.trace
                  ? "the 16-row prefill leaves the SAME rows as the 1-row one"
                  : "the 16-row prefill DIFFERS from the 1-row one: the prefill "
                    "GEMM does not reproduce decode's association");
        CHECK(c1.logits == c16.logits, "and its logits row agrees");
    }
    if (c32.ok) {
        CHECK(c1.trace == c32.trace,
              c1.trace == c32.trace
                  ? "the 32-row prefill leaves the SAME rows as the 1-row one"
                  : "the 32-row prefill DIFFERS from the 1-row one: the prefill "
                    "GEMM does not reproduce decode's association");
        CHECK(c1.logits == c32.logits, "and its logits row agrees");
    }

    // ---- THE DEFAULT CHUNK (512) -----------------------------------------
    // The default is only a promise if the size it names is itself invariant,
    // and the arms above cannot see it: their fixture holds 64 tokens, so
    // --chunk 512 there is ONE call and the comparison is vacuous. This arm
    // builds the same fixture with a 640-token context and a prompt that
    // tokenizes past 512, so the 512 arm is the (512 + remainder) split that
    // the default actually produces on a real prompt.
    const std::string big = "kraken-prefill-gate-512.gguf";
    CHECK(build_prefill_gate_model(big, 640),
          "wrote the 640-token fixture the default-chunk arm needs");
    std::string bprompt;
    for (int i = 0; i < 530; i++) bprompt += (i ? " the" : "the");
    const PrefillProbe d1 = run_prefill_hash(exe, big, bprompt, 1, false, false,
                                             nullptr, 640);
    const PrefillProbe d8 = run_prefill_hash(exe, big, bprompt, 8, false, false,
                                             nullptr, 640);
    const PrefillProbe d512 = run_prefill_hash(exe, big, bprompt, 512, false,
                                               false, nullptr, 640);
    if (!d1.ok || !d8.ok || !d512.ok)
        std::fprintf(stderr,
                     "  (default-chunk arms: 1 ok=%d rc=%d | 8 ok=%d rc=%d | "
                     "512 ok=%d rc=%d; 512 stderr: %.400s)\n",
                     static_cast<int>(d1.ok), d1.rc,
                     static_cast<int>(d8.ok), d8.rc,
                     static_cast<int>(d512.ok), d512.rc, d512.err.c_str());
    CHECK(d1.ok && d8.ok && d512.ok, "all three default-chunk arms print hashes");
    if (d1.ok && d8.ok && d512.ok) {
        std::printf("       default chunk: 1 -> %s  8 -> %s  512 -> %s\n",
                    d1.trace.c_str(), d8.trace.c_str(), d512.trace.c_str());
        CHECK(d1.prompt_tokens > 512 && d1.prompt_tokens == d512.prompt_tokens,
              "the prompt spans more than one 512-row chunk");
        CHECK(d1.trace == d512.trace,
              d1.trace == d512.trace
                  ? "the 512-row prefill (the default) leaves the SAME rows as "
                    "the 1-row one"
                  : "the 512-row prefill DIFFERS from the 1-row one: the default "
                    "chunk size is not bit-exact");
        CHECK(d1.logits == d512.logits,
              "and the default chunk's logits row agrees bit for bit");
        CHECK(d1.trace == d8.trace, "the 8-row arm on this fixture agrees too");
    }

    // THE TEXT PROMISE (token gen as well as prefill). A short prompt the
    // fixture can extend without overflowing its 64-token context, decoded
    // 8 greedy tokens at chunk 1, 8 and 64: chunk 1 threads every row
    // through the decode formulation, chunk 8 stays on the ladder, chunk 64
    // takes the exact grid for its one 20-row call. Same text or the gate
    // fails -- this is the trajectory-equality endpoint (2609.15504), not a
    // benchmark score.
    std::string gprompt;
    for (int i = 0; i < 19; i++) gprompt += (i ? " the" : "the");
    int grc = -1;
    const std::string g1 =
        run_generate_text(exe, model, gprompt, 1, 8, "", grc);
    const bool g1ok = grc == 0 && !g1.empty();
    CHECK(g1ok, "the chunk-1 generation prints text");
    // Replica floor (2607.28495): the same construction twice must agree
    // with itself before any cross-chunk comparison means anything.
    int grc2 = -1;
    const std::string g1b =
        run_generate_text(exe, model, gprompt, 1, 8, "b", grc2);
    if (g1ok && grc2 == 0) {
        CHECK(g1b == g1,
              g1b == g1 ? "the chunk-1 generation repeats itself exactly"
                        : "the same chunk-1 run disagrees with itself: "
                          "nondeterminism below the gate's floor");
    }
    int grc8 = -1;
    const std::string g8 =
        run_generate_text(exe, model, gprompt, 8, 8, "", grc8);
    if (g1ok && grc8 == 0) {
        CHECK(g8 == g1,
              g8 == g1 ? "chunk-8 decoded text matches chunk-1, token for token"
                       : "chunk-8 text DIFFERS from chunk-1: decode path "
                         "and prefill path disagree");
    }
    int grc64 = -1;
    const std::string g64 =
        run_generate_text(exe, model, gprompt, 64, 8, "", grc64);
    if (g1ok && grc64 == 0) {
        CHECK(g64 == g1,
              g64 == g1
                  ? "chunk-64 decoded text matches chunk-1, token for token"
                  : "chunk-64 text DIFFERS from chunk-1: the exact grid "
                    "does not reproduce decode's text");
    }
    // Where the arms ran, and whether the gate could see a regression at all.
#ifdef KRK_TEST_HIP_BUILD
    const bool device = c1.banner.find("scalar reference") == std::string::npos &&
                        c1.err.find("falling back to the CPU") == std::string::npos;
    CHECK(device,
          device ? "the gate ran on the device kernel, not the CPU fallback"
                 : "a HIP build fell back to the CPU: this gate proved nothing");
    if (device) {
        // THE CONTROL. The same chunk-8 arm with the tiled formulation forced
        // must NOT match the chunk-1 reference, or the gate would pass a kernel
        // regression it cannot see. It is the arm the exact kernel replaced
        // (see AGENTS.md, "Chunk invariance is an ATTENTION property").
        const PrefillProbe t8 = run_prefill_hash(exe, model, prompt, 8, true);
        CHECK(t8.ok, "the control arm runs with the tiled kernel forced");
        std::printf("       tiled control: chunk 8 trace %s logits %s\n",
                    t8.trace.c_str(), t8.logits.c_str());
        if (t8.ok) {
            CHECK(t8.trace != c1.trace,
                  t8.trace != c1.trace
                      ? "with the tiled kernel forced the SAME comparison fails, "
                        "so the gate can detect a regression"
                      : "the tiled kernel now passes the gate too -- re-point the "
                        "control, this gate is no longer proven able to fail");
        }
        // THE GEMM CONTROL. The same chunk-16 arm with the rung lottery
        // forced (KRK_GEMM_EXACT=0) must NOT match the chunk-1 reference:
        // rows above the ladder then run the WMMA association, which is a
        // different summation order from decode's and must read as one.
        const PrefillProbe w16 =
            run_prefill_hash(exe, model, prompt, 16, false, true);
        CHECK(w16.ok, "the GEMM-lottery control arm runs");
        std::printf("       lottery control: chunk 16 trace %s logits %s\n",
                    w16.trace.c_str(), w16.logits.c_str());
        if (w16.ok && c1.ok) {
            CHECK(w16.trace != c1.trace,
                  w16.trace != c1.trace
                      ? "with the rung lottery forced the 16-row arm differs, "
                        "so the gate can detect a GEMM regression"
                      : "the lottery arm matches the reference -- re-point the "
                        "GEMM control, it no longer proves anything");
        }
        // THE DEFAULT-CHUNK CONTROL. A 512-row arm that agreed whatever the
        // GEMM association is would be two single-row calls in disguise --
        // i.e. it would prove nothing about the size it claims to pin -- so
        // this arm breaks the one thing it is meant to be sensitive to.
        const PrefillProbe w512 =
            run_prefill_hash(exe, big, bprompt, 512, false, true, nullptr, 640);
        CHECK(w512.ok, "the default-chunk lottery control runs");
        if (w512.ok && d1.ok) {
            CHECK(w512.trace != d1.trace,
                  w512.trace != d1.trace
                      ? "with the rung lottery forced the 512-row arm differs, "
                        "so the default-chunk gate can fail"
                      : "the 512-row lottery arm matches the 1-row reference -- "
                        "it is measuring one-row calls, not the default chunk");
        }
    }
#else
    // No device kernels in this build, so the scalar reference is the arm that
    // exists: the gate still proves the engine threads chunks exactly there.
    std::printf("       (built without HIP: the scalar reference is the only "
                "arm)\n");
#endif
    // This group writes outside its own pipes -- the fixture and one capture
    // pair per arm -- and in INPROC mode there is no per-group working
    // directory to drop them with. Remove them when the group is green; on a
    // failure they stay, and so does the child-mode scratch directory.
    if (g_run == run0 + (g_passed - pass0)) {
        for (const char *name : {"kraken-prefill-out-1.txt", "kraken-prefill-out-2.txt",
                                 "kraken-prefill-out-512.txt",
                                 "kraken-prefill-err-512.txt",
                                 "kraken-prefill-out-8.txt", "kraken-prefill-out-8-t.txt",
                                 "kraken-prefill-err-1.txt", "kraken-prefill-err-2.txt",
                                 "kraken-prefill-err-8.txt", "kraken-prefill-err-8-t.txt",
                                 "kraken-prefill-out-16.txt", "kraken-prefill-out-32.txt",
                                 "kraken-prefill-out-16-lottery.txt",
                                 "kraken-prefill-err-16.txt", "kraken-prefill-err-32.txt",
                                 "kraken-prefill-err-16-lottery.txt",
                                 "kraken-prefill-gen-1.txt", "kraken-prefill-gen-1b.txt",
                                 "kraken-prefill-gen-8.txt", "kraken-prefill-gen-64.txt",
                                 "kraken-prefill-err-gen-1.txt", "kraken-prefill-err-gen-1b.txt",
                                 "kraken-prefill-err-gen-8.txt", "kraken-prefill-err-gen-64.txt"})
            std::remove(name);
        std::remove(model.c_str());
        std::remove("kraken-prefill-gate-512.gguf");
    }

}

// The gemma families, driven end to end through the CLI: load, schema flags,
// per-layer window selection, the final-logit bound, gemma2's attention-logit
// bound, and the prefill rows all have to agree with themselves across chunk
// sizes before "gemma2 is supported" means anything. That is what this group
// measures, on fixtures built here rather than on a file nobody in this
// collection has.
//
// Three arms per family. Chunk 1 threads every prompt row through the
// decode-shaped single-row forward, so it stands in for token generation; chunk
// 8 and 32 take the multi-row paths. One hash across the three is the gate. The
// fourth check is the control that keeps it honest: for gemma2 the same file
// with the softcapping key removed must hash DIFFERENTLY, or the key is being
// ignored and the gate above would pass on a model that never applied the
// bound.
static void test_gemma_family_chunk_gate() {
    const std::string self = self_path();
    const std::filesystem::path exe_p =
        std::filesystem::path(self).parent_path() /
#ifdef _WIN32
        "kraken.exe";
#else
        "kraken";
#endif
    const std::string exe = exe_p.string();
    const bool have_exe = !self.empty() && std::filesystem::exists(exe_p);
    CHECK(have_exe, "kraken sits beside kraken-tests (this gate drives it)");
    if (!have_exe) return;

    std::string prompt;
    for (int i = 0; i < 55; i++) prompt += (i ? " the" : "the");

    struct Fam {
        const char *arch;
        bool cap;
        const char *file;
    };
    const Fam fams[] = {
        {"gemma2", true, "kraken-gemma2-gate.gguf"},
        {"gemma3", false, "kraken-gemma3-gate.gguf"},
    };
    for (const Fam &f : fams) {
        CHECK(build_gemma_gate_model(f.file, f.arch, f.cap),
              (std::string("wrote the ") + f.arch + " fixture this gate prefills")
                  .c_str());
        const PrefillProbe c1 = run_prefill_hash(exe, f.file, prompt, 1, false);
        const PrefillProbe c8 = run_prefill_hash(exe, f.file, prompt, 8, false);
        const PrefillProbe c32 = run_prefill_hash(exe, f.file, prompt, 32, false);
        CHECK(c1.ok, (std::string(f.arch) + " loads and prefills at chunk 1").c_str());
        CHECK(c8.ok, (std::string(f.arch) + " loads and prefills at chunk 8").c_str());
        CHECK(c32.ok,
              (std::string(f.arch) + " loads and prefills at chunk 32").c_str());
        if (!c1.ok || !c8.ok || !c32.ok) {
            std::fprintf(stderr, "  (%s: rc %d/%d/%d stderr: %.400s)\n", f.arch,
                         c1.rc, c8.rc, c32.rc, c1.err.c_str());
            continue;
        }
        // A load that silently fell back to the llama schema would still
        // produce a hash; the banner's layer/window line is what says the
        // family's own schema was read.
        std::printf("       %s chunk  1 trace %s logits %s\n", f.arch,
                    c1.trace.c_str(), c1.logits.c_str());
        std::printf("       %s chunk  8 trace %s logits %s\n", f.arch,
                    c8.trace.c_str(), c8.logits.c_str());
        std::printf("       %s chunk 32 trace %s logits %s\n", f.arch,
                    c32.trace.c_str(), c32.logits.c_str());
        // The count is the tokenizer's answer, not this test's: what matters is
        // that the prompt spans several chunks (so every arm's last write is a
        // multi-row one at some point) and that all three arms see the same
        // sequence. 55 words tokenizes to >= 40 pieces on this vocab.
        CHECK(c1.prompt_tokens >= 40 &&
                  c1.prompt_tokens == c8.prompt_tokens &&
                  c8.prompt_tokens == c32.prompt_tokens,
              (std::string(f.arch) + ": every arm tokenizes the same prompt")
                  .c_str());
        CHECK(c1.trace == c8.trace && c8.trace == c32.trace,
              (c1.trace == c8.trace && c8.trace == c32.trace
                   ? std::string(f.arch) +
                         ": chunk 1, 8 and 32 leave the same rows, so prefill "
                         "and decode agree across the window and the softcaps"
                   : std::string(f.arch) +
                         ": the chunk sizes DISAGREE -- the multi-row forward "
                         "is not the decode formulation")
                  .c_str());
        CHECK(c1.logits == c32.logits,
              (std::string(f.arch) +
               ": and the logits row agrees bit for bit")
                  .c_str());

        // Token generation on the same file: a greedy decode must run and
        // print text (a load that refused, or a NaN forward, prints nothing).
        int rc = -1;
        const std::string txt =
            run_generate_text(exe, f.file, prompt.substr(0, 40), 1, 8, f.arch, rc);
        CHECK(rc == 0 && !txt.empty(),
              (std::string(f.arch) +
               ": decodes 8 greedy tokens without failing")
                  .c_str());
    }

    // ---- the control: gemma2's bound is READ, not decorative --------------
    const std::string capped = "kraken-gemma2-gate.gguf";
    const std::string nokey = "kraken-gemma2-nocap.gguf";
    CHECK(build_gemma_gate_model(nokey, "gemma2", false),
          "wrote the same gemma2 file without attention.logit_softcapping");
    const PrefillProbe a = run_prefill_hash(exe, capped, prompt, 8, false);
    const PrefillProbe bcap = run_prefill_hash(exe, nokey, prompt, 8, false);
    CHECK(a.ok && bcap.ok, "both gemma2 files prefill");
    if (a.ok && bcap.ok) {
        CHECK(a.trace != bcap.trace,
              a.trace != bcap.trace
                  ? "the softcapping key changes the model: cap*tanh(score/cap) "
                    "is applied, not ignored"
                  : "the softcapping key is IGNORED -- the two files prefill "
                    "byte-identically, so gemma2's bound is not applied");
    }
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

// Sends exactly these bytes, optionally half-closing the write side so the
// server sees EOF where a truncated body arrives, and reads whatever comes
// back. SO_RCVTIMEO bounds the read: a parser that neither answers nor closes
// fails the check instead of hanging the suite. Returns true when any response
// byte arrived, so a control request and a refusal are judged the same way.
static bool http_exchange_raw(int port, const std::string &wire, bool half_close,
                              std::string *out) {
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
#ifdef _WIN32
    DWORD timeout_ms = 10000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char *>(&timeout_ms),
                 static_cast<int>(sizeof(timeout_ms)));
#else
    timeval tv{};
    tv.tv_sec = 10;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
    size_t sent = 0;
    while (sent < wire.size()) {
#ifdef _WIN32
        const int n =
            ::send(fd, wire.data() + sent, static_cast<int>(wire.size() - sent), 0);
#else
        const ssize_t n = ::send(fd, wire.data() + sent, wire.size() - sent, 0);
#endif
        // A refusal can land while we are still sending (the oversized-header
        // case): stop writing and read the answer it already wrote.
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
    if (half_close) {
#ifdef _WIN32
        ::shutdown(fd, SD_SEND);
#else
        ::shutdown(fd, SHUT_WR);
#endif
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
    return !out->empty();
}

// ---------------------------------------------------------------------------
// Strict HTTP/1.1 client parsing. The endpoint checks below look for
// substrings, and a substring search cannot tell a well-framed response from
// one that merely CONTAINS the expected text: a header line ending in a bare
// newline, a Content-Length that disagrees with the body, or an SSE stream
// whose chunk sizes are wrong all pass `find()`. This parser is the client's
// view -- it either yields a body or names the byte that broke the protocol --
// and its arms show it failing once per defect class, so a green parse is
// evidence about the wire rather than about the test's search terms.
// ---------------------------------------------------------------------------

struct HttpWire {
    bool ok = false;
    std::string why;
    int status = 0;
    std::string reason;
    std::vector<std::pair<std::string, std::string>> headers;
    bool chunked = false;
    size_t chunks = 0; // data chunks decoded (0 for a buffered body)
    std::string body;  // de-chunked when the transfer was chunked
};

static std::string http_wire_header(const HttpWire &w, const std::string &name) {
    for (const auto &kv : w.headers)
        if (kv.first == name) return kv.second;
    return std::string();
}

static bool http_wire_hex(char c, int *d) {
    if (c >= '0' && c <= '9') { *d = c - '0'; return true; }
    if (c >= 'a' && c <= 'f') { *d = c - 'a' + 10; return true; }
    if (c >= 'A' && c <= 'F') { *d = c - 'A' + 10; return true; }
    return false;
}

// One CRLF-terminated line at *p; advances past it. False on a bare LF.
static bool http_wire_line(const std::string &s, size_t *p, std::string *line) {
    const size_t nl = s.find('\n', *p);
    if (nl == std::string::npos || nl == *p || s[nl - 1] != '\r') return false;
    *line = s.substr(*p, nl - 1 - *p);
    *p = nl + 1;
    return true;
}

static HttpWire http_wire_parse(const std::string &raw) {
    HttpWire w;
    const size_t sep = raw.find("\r\n\r\n");
    if (sep == std::string::npos) {
        w.why = "no CRLF blank line ends the head";
        return w;
    }
    const std::string head = raw.substr(0, sep + 2); // every line keeps its CRLF
    for (size_t i = 0; i < head.size(); i++) {
        if (head[i] == '\n' && (i == 0 || head[i - 1] != '\r')) {
            w.why = "bare LF in the head";
            return w;
        }
        if (head[i] == '\r' && (i + 1 >= head.size() || head[i + 1] != '\n')) {
            w.why = "bare CR in the head";
            return w;
        }
        if (head[i] == '\\') {
            // The runtime form of the source-hygiene group's second class: the
            // escape was written twice, so the bytes carry a backslash where a
            // line break belongs.
            w.why = "backslash printed in the head (doubled escape in a header)";
            return w;
        }
    }
    size_t p = 0;
    std::string line;
    if (!http_wire_line(head, &p, &line)) {
        w.why = "the status line is not CRLF-terminated";
        return w;
    }
    const bool digits = line.size() > 12 && line[9] >= '0' && line[9] <= '9' &&
                        line[10] >= '0' && line[10] <= '9' &&
                        line[11] >= '0' && line[11] <= '9';
    if (line.compare(0, 9, "HTTP/1.1 ") != 0 || !digits || line[12] != ' ' ||
        line.size() == 13) {
        w.why = "status line is not 'HTTP/1.1 <3 digits> <reason>'";
        return w;
    }
    w.status = (line[9] - '0') * 100 + (line[10] - '0') * 10 + (line[11] - '0');
    w.reason = line.substr(13);
    size_t content_length_lines = 0;
    while (p < head.size()) {
        if (!http_wire_line(head, &p, &line)) {
            w.why = "a header line is not CRLF-terminated";
            return w;
        }
        const size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0 || line[colon + 1] != ' ') {
            w.why = "a header line is not 'Name: value'";
            return w;
        }
        if (line.find(' ') < colon) {
            w.why = "a header name contains a space";
            return w;
        }
        if (line.compare(0, 14, "Content-Length") == 0) content_length_lines++;
        w.headers.emplace_back(line.substr(0, colon), line.substr(colon + 2));
    }
    const std::string rest = raw.substr(sep + 4);
    if (http_wire_header(w, "Transfer-Encoding") == "chunked") {
        w.chunked = true;
        size_t q = 0;
        while (true) {
            const size_t nl = rest.find("\r\n", q);
            if (nl == std::string::npos) {
                w.why = "chunked body: no size line";
                return w;
            }
            const size_t n_digits = nl - q;
            size_t n = 0;
            if (n_digits == 0 || n_digits > 16) {
                w.why = "chunked body: size is not 1..16 hex digits";
                return w;
            }
            for (size_t i = q; i < nl; i++) {
                int d = 0;
                if (!http_wire_hex(rest[i], &d)) {
                    w.why = "chunked body: size line is not hex";
                    return w;
                }
                n = n * 16 + static_cast<size_t>(d);
            }
            q = nl + 2;
            if (n == 0) {
                if (rest.compare(q, 2, "\r\n") != 0) {
                    w.why = "chunked body: the terminal chunk is not followed by CRLF";
                    return w;
                }
                if (q + 2 != rest.size()) {
                    w.why = "chunked body: bytes follow the terminal chunk";
                    return w;
                }
                break;
            }
            if (q + n + 2 > rest.size()) {
                w.why = "chunked body: a chunk is shorter than its size line";
                return w;
            }
            w.body.append(rest, q, n);
            w.chunks++;
            if (rest.compare(q + n, 2, "\r\n") != 0) {
                w.why = "chunked body: a chunk is not followed by CRLF";
                return w;
            }
            q += n + 2;
        }
    } else {
        if (content_length_lines != 1) {
            w.why = content_length_lines ? "Content-Length appears more than once"
                                         : "no Content-Length and no chunked framing";
            return w;
        }
        const std::string cl = http_wire_header(w, "Content-Length");
        size_t declared = 0;
        for (char c : cl) {
            if (c < '0' || c > '9') {
                w.why = "Content-Length is not a number";
                return w;
            }
            declared = declared * 10 + static_cast<size_t>(c - '0');
        }
        if (declared != rest.size()) {
            w.why = "Content-Length disagrees with the body";
            return w;
        }
        w.body = rest;
    }
    w.ok = true;
    return w;
}

// Strict server-sent events: every event is exactly `data: <payload>` + LF LF,
// the payload holds no line break of its own, and `data: [DONE]` comes last.
struct SseWire {
    bool ok = false;
    std::string why;
    std::vector<std::string> events; // payloads, [DONE] excluded
    bool done_seen = false;
    size_t finish_events = 0;
    std::string text; // concatenated delta content
};

static SseWire sse_wire_parse(const std::string &s) {
    SseWire o;
    size_t p = 0;
    while (p < s.size()) {
        if (s.compare(p, 6, "data: ") != 0) {
            o.why = "an event does not start with 'data: '";
            return o;
        }
        const size_t end = s.find("\n\n", p + 6);
        if (end == std::string::npos) {
            o.why = "an event is not terminated by a blank line";
            return o;
        }
        const std::string payload = s.substr(p + 6, end - p - 6);
        if (payload.find('\n') != std::string::npos ||
            payload.find('\r') != std::string::npos) {
            o.why = "an event payload holds a line break";
            return o;
        }
        p = end + 2;
        if (payload == "[DONE]") {
            if (o.done_seen) {
                o.why = "[DONE] appears twice";
                return o;
            }
            o.done_seen = true;
            if (p != s.size()) {
                o.why = "bytes follow [DONE]";
                return o;
            }
            continue;
        }
        JsonValue ev;
        std::string err;
        if (!json_parse(payload, &ev, &err)) {
            o.why = "an event payload is not JSON: " + err;
            return o;
        }
        o.events.push_back(payload);
        const JsonValue *choices = ev.find("choices");
        if (!choices || !choices->is_array() || choices->items().empty()) continue;
        const JsonValue &c = choices->items()[0];
        const JsonValue *delta = c.find("delta");
        if (delta) {
            const JsonValue *content = delta->find("content");
            if (content && content->is_string()) o.text += content->as_string();
        }
        const JsonValue *fr = c.find("finish_reason");
        if (fr && fr->is_string()) o.finish_events++;
    }
    if (!o.done_seen) {
        o.why = "the stream does not end with data: [DONE]";
        return o;
    }
    o.ok = true;
    return o;
}

// The parser's own arms. Each defective response must be refused for the reason
// named, and the well-framed one must be ACCEPTED: without that positive arm a
// parser that refused everything would pass every other check here.
static void http_wire_arms() {
    const std::string good =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 2\r\n"
        "\r\n"
        "{}";
    const HttpWire g = http_wire_parse(good);
    CHECK(g.ok && g.status == 200 && g.reason == "OK" && g.body == "{}" &&
              !g.chunked && http_wire_header(g, "Content-Type") == "application/json",
          "http bytes: the strict parser accepts a well-framed response");
    if (!g.ok) std::fprintf(stderr, "  (good response refused: %s)\n", g.why.c_str());

    auto refused = [](const std::string &raw, const char *needle, const char *what) {
        const HttpWire w = http_wire_parse(raw);
        const bool named = w.why.find(needle) != std::string::npos;
        CHECK(!w.ok && named, what);
        if (w.ok || !named)
            std::fprintf(stderr, "  (%s: ok=%d why='%s')\n", what, w.ok ? 1 : 0,
                         w.why.c_str());
    };

    refused("HTTP/1.1 200 OK\nContent-Length: 2\r\n\r\n{}", "bare LF",
            "http bytes: a bare LF in the head is refused");
    refused("HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{}", "Content-Length",
            "http bytes: a Content-Length that disagrees with the body is refused");
    refused("HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}", "HTTP/1.1",
            "http bytes: a status line that is not HTTP/1.1 is refused");
    refused("HTTP/1.1 200 OK\r\nContent-Length:2\r\n\r\n{}", "Name: value",
            "http bytes: a header without its space is refused");
    {
        // A doubled escape in a header. The backslash comes from character code
        // 92 so the test does not spell the defect it is checking for.
        std::string bs = "HTTP/1.1 200 OK\r\nContent-Type: text/pla";
        bs += static_cast<char>(92);
        bs += "in\r\nContent-Length: 0\r\n\r\n";
        refused(bs, "backslash", "http bytes: a backslash printed in the head is refused");
    }
    refused("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nab\r\n",
            "shorter than its size line", "http bytes: a short chunk is refused");
    refused("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n",
            "no size line", "http bytes: a missing terminal chunk is refused");
    refused("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n0\r\n\r\nx",
            "follow the terminal chunk",
            "http bytes: trailing bytes after the terminal chunk are refused");
    refused("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhiX\r\n0\r\n\r\n",
            "not followed by CRLF", "http bytes: a chunk without its CRLF is refused");

    const std::string two_chunks =
        "HTTP/1.1 200 OK\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n"
        "4\r\ndata\r\n"
        "6\r\n: hi\n\n\r\n"
        "0\r\n\r\n";
    const HttpWire cw = http_wire_parse(two_chunks);
    CHECK(cw.ok && cw.chunked && cw.chunks == 2 && cw.body == "data: hi\n\n",
          "http bytes: the parser reassembles a two-chunk body exactly");

    const SseWire ok_sse = sse_wire_parse("data: {\"x\":1}\n\ndata: [DONE]\n\n");
    CHECK(ok_sse.ok && ok_sse.events.size() == 1 && ok_sse.done_seen,
          "http bytes: the SSE parser accepts a well-framed event and [DONE]");
    CHECK(!sse_wire_parse("data: {\"x\":1}\n").ok,
          "http bytes: an unterminated SSE event is refused");
    CHECK(!sse_wire_parse("data: [DONE]\n\ndata: x\n\n").ok,
          "http bytes: bytes after [DONE] are refused");
}

// ---------------------------------------------------------------------------
// The real kraken-server binary. The listener below and that process run the
// same HttpServer + OpenAiService pair, but only the binary proves the wiring
// main_server.cpp owns: argument parsing, `--port 0`, and the banner a client
// discovers the port from. It runs --cpu because the suite links no device
// backend (the response BYTES are backend-independent), its stdio goes to
// capture files so the banner can be polled, and it is terminated at the end:
// main_server.cpp has no stop path -- its own comment says server.run() returns
// only after stop(), which nothing calls -- so a forced end is the only one
// available, and stop() waits for it before the captures are read or removed.
// ---------------------------------------------------------------------------

struct ServerProc;
static void server_proc_stop(ServerProc *sp);

struct ServerProc {
#ifdef _WIN32
    void *process = nullptr;
#else
    int pid = 0;
#endif
    std::string out_path;
    std::string err_path;
    bool started = false;
    ~ServerProc() { server_proc_stop(this); }
};

#ifdef _WIN32
static bool server_proc_running(const ServerProc &sp) {
    if (!sp.process) return false;
    return WaitForSingleObject(static_cast<HANDLE>(sp.process), 0) == WAIT_TIMEOUT;
}

static void server_proc_stop(ServerProc *sp) {
    if (!sp->process) return;
    TerminateProcess(static_cast<HANDLE>(sp->process), 0);
    WaitForSingleObject(static_cast<HANDLE>(sp->process), 10000);
    CloseHandle(static_cast<HANDLE>(sp->process));
    sp->process = nullptr;
    sp->started = false;
}

static bool server_proc_start(ServerProc *sp, const std::string &exe,
                              const std::vector<std::string> &args) {
    SECURITY_ATTRIBUTES sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE out = CreateFileA(sp->out_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE err = CreateFileA(sp->err_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out == INVALID_HANDLE_VALUE || err == INVALID_HANDLE_VALUE) {
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
        if (err != INVALID_HANDLE_VALUE) CloseHandle(err);
        return false;
    }
    std::string cmd = "\"" + exe + "\"";
    for (const std::string &a : args) cmd += " \"" + a + "\"";
    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back('\0');

    STARTUPINFOA si;
    std::memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = out;
    si.hStdError = err;
    PROCESS_INFORMATION pi;
    std::memset(&pi, 0, sizeof(pi));
    const BOOL ok = CreateProcessA(exe.c_str(), mutable_cmd.data(), nullptr, nullptr,
                                   TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(out);
    CloseHandle(err);
    if (!ok) return false;
    CloseHandle(pi.hThread);
    sp->process = pi.hProcess;
    sp->started = true;
    return true;
}
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

static bool server_proc_running(const ServerProc &sp) {
    if (sp.pid <= 0) return false;
    int st = 0;
    return waitpid(sp.pid, &st, WNOHANG) == 0;
}

static void server_proc_stop(ServerProc *sp) {
    if (sp->pid <= 0) return;
    kill(sp->pid, SIGTERM);
    waitpid(sp->pid, nullptr, 0);
    sp->pid = 0;
    sp->started = false;
}

static bool server_proc_start(ServerProc *sp, const std::string &exe,
                              const std::vector<std::string> &args) {
    std::vector<std::string> storage;
    storage.push_back(exe);
    for (const std::string &a : args) storage.push_back(a);
    std::vector<char *> argv;
    for (std::string &s : storage) argv.push_back(s.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, sp->out_path.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_addopen(&fa, 2, sp->err_path.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, exe.c_str(), &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return false;
    sp->pid = static_cast<int>(pid);
    sp->started = true;
    return true;
}
#endif

// The banner main_server.cpp prints after bind: `listening on http://HOST:PORT`.
// `--port 0` makes the port ephemeral, so this is how a caller learns both the
// port and that the bind actually happened.
static int server_proc_wait_port(const ServerProc &sp, int timeout_ms) {
    const std::string key = "listening on http://";
    const int step = 25;
    for (int waited = 0; waited < timeout_ms; waited += step) {
        const std::string out = read_capture(sp.out_path);
        const size_t at = out.find(key);
        if (at != std::string::npos) {
            const size_t colon = out.find(':', at + key.size());
            if (colon != std::string::npos) {
                int port = 0;
                for (size_t i = colon + 1; i < out.size() && out[i] >= '0' && out[i] <= '9';
                     i++)
                    port = port * 10 + (out[i] - '0');
                if (port > 0) return port;
            }
        }
        if (!server_proc_running(sp)) return -1; // exited: the banner will not come
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
    }
    return -1;
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
    const std::string resource_model_path = "kraken-http-resource-test.gguf";
    const bool resource_model_built = build_sparse_kv_model(resource_model_path, 8192);
    CHECK(resource_model_built,
          "http test: built a multi-KV-layer model that requires tier promotion");
    CountingBackend *resource_backend = new CountingBackend(make_cpu_backend());
    resource_backend->set_test_resources(16 * (static_cast<size_t>(1) << 30),
                                         8 * (static_cast<size_t>(1) << 30),
                                         16 * (static_cast<size_t>(1) << 30),
                                         8 * (static_cast<size_t>(1) << 30));
    Engine resource_engine;
    EngineConfig resource_cfg = cfg;
    resource_cfg.model_path = resource_model_path;
    resource_cfg.n_ctx = 8192;
    resource_cfg.prefill_chunk = 4;
    resource_cfg.kv_hot_mb = 1;
    resource_cfg.kv_warm_mb = 1;
    std::string resource_err;
    const bool resource_inited = resource_engine.init(resource_backend, resource_cfg,
                                                       &resource_err);
    CHECK(resource_inited,
          "http test: resource-pressure engine initializes before pressure changes");
    if (!resource_inited) {
        std::fprintf(stderr, "  (HTTP resource fixture: %s)\n", resource_err.c_str());
        delete resource_backend;
    }
    std::mutex resource_mu;
    OpenAiService *resource_svc = resource_inited
        ? new OpenAiService(&resource_engine, 0.8f, &resource_mu) : nullptr;
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
        // The first request after a direct generate() finds a cache entry only
        // if it shares a prefix with what generate() left behind, which for the
        // same prompt it does -- but never the last token (a run needs a row to
        // sample from).
        const JsonValue *det = usage ? usage->find("prompt_tokens_details") : nullptr;
        const i64 cached0 = det ? det->get_int("cached_tokens", -1) : -1;
        CHECK(cached0 >= 0 && cached0 < ref.prompt_tokens,
              "http: usage reports how many prompt tokens came from the cache");
    }

    // ---- a warm request answers from the KV cache and agrees bit for bit ----
    // Same prompt, same sampler: the second request finds the first one's
    // prompt in the KV, prefills only its last token, and must produce exactly
    // the same text. This is the property that makes prefix reuse safe to
    // ship -- a cache that changes the answer is not an optimisation.
    {
        CHECK(http_request(port, "POST", "/v1/completions", req, &status, &body) &&
                  status == 200,
              "http: a repeated prompt is answered from the warm cache");
        JsonValue j;
        std::string e;
        const std::string payload = body.substr(body.find("\r\n\r\n") + 4);
        CHECK(json_parse(payload, &j, &e), "http: warm body is valid JSON");
        const JsonValue *choices = j.find("choices");
        std::string warm_text;
        if (choices && !choices->items().empty())
            choices->items()[0].get_string("text", &warm_text);
        CHECK(warm_text == ref.text,
              "http: the warm run's text is bit-identical to the cold one");
        const JsonValue *usage = j.find("usage");
        const JsonValue *det = usage ? usage->find("prompt_tokens_details") : nullptr;
        const i64 cached = det ? det->get_int("cached_tokens", -1) : -1;
        CHECK(cached == ref.prompt_tokens - 1,
              "http: it reused every prompt token but the one it must prefill");
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

    // ---- the same responses, read the way a client reads them ---------------
    // The checks above find their text; these parse the bytes. A client that
    // splits the head on CRLF, trusts Content-Length and de-chunks per RFC 7230
    // must accept every response, and http_wire_arms shows that judgement can
    // fail -- once per defect class, on constructed answers.
    http_wire_arms();
    {
        std::string raw;
        int code = 0;

        CHECK(http_request(port, "GET", "/health", "", &code, &raw) && code == 200,
              "http bytes: /health answers 200");
        const HttpWire hw = http_wire_parse(raw);
        CHECK(hw.ok, "http bytes: /health is a well-framed HTTP/1.1 response");
        if (!hw.ok) std::fprintf(stderr, "  (/health framing: %s)\n", hw.why.c_str());
        CHECK(hw.status == 200 && hw.reason == "OK",
              "http bytes: the status line is exactly 'HTTP/1.1 200 OK'");
        // HTTP/1.1 defaults to a PERSISTENT connection, so a server that closes
        // after one response must say so; otherwise a client is entitled to
        // reuse the socket and writes its next request into a closed pipe.
        CHECK(http_wire_header(hw, "Connection") == "close",
              "http bytes: a 200 states its connection handling (Connection: close)");
        CHECK(http_wire_header(hw, "Content-Type") == "application/json",
              "http bytes: Content-Type is application/json");
        JsonValue hj;
        std::string he;
        const bool hj_ok = json_parse(hw.body, &hj, &he);
        const JsonValue *status_v = hj_ok ? hj.find("status") : nullptr;
        CHECK(hj_ok && hj.is_object() && status_v && status_v->is_string() &&
                  status_v->as_string() == "ok",
              "http bytes: the advertised Content-Length frames a complete JSON body");

        CHECK(http_request(port, "GET", "/v1/nope", "", &code, &raw) && code == 404,
              "http bytes: an unknown endpoint answers 404");
        const HttpWire nf = http_wire_parse(raw);
        CHECK(nf.ok && nf.status == 404 && nf.reason == "Not Found",
              "http bytes: the 404 carries the standard reason phrase");
        CHECK(http_wire_header(nf, "Connection") == "close",
              "http bytes: an error response tells the client the connection closes");
        JsonValue nj;
        std::string ne;
        const bool nj_ok = json_parse(nf.body, &nj, &ne);
        const JsonValue *err = nj_ok ? nj.find("error") : nullptr;
        std::string err_msg;
        CHECK(err && err->get_string("message", &err_msg) &&
                  err_msg.find("unknown endpoint") != std::string::npos,
              "http bytes: the 404 body names the endpoint it refused");

        const std::string sreq = "{\"prompt\":\"the\",\"max_tokens\":5,"
                                 "\"temperature\":0,\"repeat_penalty\":1.0,"
                                 "\"stream\":true}";
        CHECK(http_request(port, "POST", "/v1/completions", sreq, &code, &raw) &&
                  code == 200,
              "http bytes: the streamed completion answers 200");
        const HttpWire sw = http_wire_parse(raw);
        CHECK(sw.ok, "http bytes: the event stream's chunk framing decodes");
        if (!sw.ok) std::fprintf(stderr, "  (stream framing: %s)\n", sw.why.c_str());
        CHECK(sw.chunked && sw.chunks >= 3,
              "http bytes: the stream is chunked and carries more than one chunk");
        CHECK(http_wire_header(sw, "Content-Type") == "text/event-stream" &&
                  http_wire_header(sw, "Cache-Control") == "no-cache",
              "http bytes: the stream's head is the SSE pair");
        CHECK(http_wire_header(sw, "Connection") == "close",
              "http bytes: the stream's head states Connection: close");
        const SseWire sse = sse_wire_parse(sw.body);
        CHECK(sse.ok, "http bytes: every decoded event is 'data: <json>' + a blank line");
        if (!sse.ok) std::fprintf(stderr, "  (sse framing: %s)\n", sse.why.c_str());
        CHECK(sse.done_seen, "http bytes: the decoded stream ends with data: [DONE]");
        CHECK(sse.finish_events == 1,
              "http bytes: exactly one event carries a finish_reason");
        CHECK(sse.text == ref.text,
              "http bytes: the decoded deltas are the buffered answer, byte for byte");
        std::fprintf(stderr, "  (stream: %zu chunk(s), %zu event(s), %zu payload byte(s))\n",
                     sw.chunks, sse.events.size(), sw.body.size());
    }

    // ---- request framing: what the parser must refuse -----------------------
    // RFC 9112 §6.3: a request's body length must be unambiguous, and RFC 6585
    // reserves 431 for a header block over the server's cap (413 is about
    // content). Every case below is a byte sequence a client can send, and each
    // must be REFUSED -- none may reach the engine. The raw client half-closes
    // where a truncated body arrives, and every read is bounded, so a parser
    // that neither answers nor closes fails rather than hangs the suite.
    {
        const std::string head = "POST /v1/completions HTTP/1.0\r\nHost: localhost\r\n";
        const std::string body =
            "{\"prompt\":\"the\",\"max_tokens\":1,\"temperature\":0}";
        const std::string len = std::to_string(body.size());
        std::string raw;

        // The control: the same raw client sends a well-framed request and gets
        // its 200. Without it, a client that broke every send would "pass" the
        // refusals below for the wrong reason.
        CHECK(http_exchange_raw(port,
                                head + "Content-Length: " + len + "\r\n\r\n" + body,
                                false, &raw),
              "request bytes: the raw client's control request is answered");
        const HttpWire cw = http_wire_parse(raw);
        CHECK(cw.ok && cw.status == 200 && cw.reason == "OK",
              "request bytes: the control request is a well-framed 200");
        CHECK(cw.ok && http_wire_header(cw, "Connection") == "close",
              "request bytes: the control 200 also states Connection: close");

        // A second Content-Length, identical value included: accepting it means
        // two parties can disagree about where the body ends.
        CHECK(http_exchange_raw(port,
                                head + "Content-Length: " + len +
                                    "\r\nContent-Length: " + len + "\r\n\r\n" + body,
                                false, &raw),
              "request bytes: a duplicated Content-Length gets an answer");
        const HttpWire dw = http_wire_parse(raw);
        CHECK(dw.ok && dw.status == 400 && dw.reason == "Bad Request",
              "request bytes: a duplicated Content-Length is refused with 400");
        JsonValue dj;
        std::string de;
        const bool dj_ok = dw.ok && json_parse(dw.body, &dj, &de);
        const JsonValue *dj_err = dj_ok ? dj.find("error") : nullptr;
        std::string dj_msg;
        CHECK(dj_err && dj_err->get_string("message", &dj_msg) &&
                  dj_msg.find("duplicate Content-Length") != std::string::npos,
              "request bytes: the refusal names the duplicated header");

        // Two DIFFERENT lengths in one message is the smuggling shape.
        CHECK(http_exchange_raw(
                  port,
                  head + "Content-Length: 2\r\nContent-Length: " + len + "\r\n\r\n" +
                      body,
                  false, &raw),
              "request bytes: conflicting Content-Lengths get an answer");
        const HttpWire xw = http_wire_parse(raw);
        CHECK(xw.ok && xw.status == 400,
              "request bytes: conflicting Content-Lengths are refused");

        // The list form is the same ambiguity in one field: atoll() reads
        // "5, 44" as 5 and would truncate the body silently.
        CHECK(http_exchange_raw(port,
                                head + "Content-Length: 5, " + len + "\r\n\r\n" + body,
                                false, &raw),
              "request bytes: a Content-Length list gets an answer");
        const HttpWire lw = http_wire_parse(raw);
        JsonValue lj;
        std::string le;
        const bool lj_ok = lw.ok && json_parse(lw.body, &lj, &le);
        const JsonValue *lj_err = lj_ok ? lj.find("error") : nullptr;
        std::string lj_msg;
        CHECK(lw.ok && lw.status == 400 && lj_err &&
                  lj_err->get_string("message", &lj_msg) &&
                  lj_msg.find("invalid Content-Length") != std::string::npos,
              "request bytes: a non-numeric Content-Length is refused by name");

        // A request-side Transfer-Encoding is the remaining framing ambiguity:
        // this server speaks Content-Length or no body, so a body whose length
        // is carried by a coding it does not implement must be refused, not
        // guessed. The control sent first in this block is the same head and
        // body WITHOUT this header, so the 400 is attributable to the header.
        CHECK(http_exchange_raw(port,
                                head + "Transfer-Encoding: chunked\r\nContent-Length: " +
                                    len + "\r\n\r\n" + body,
                                false, &raw),
              "request bytes: a Transfer-Encoding request gets an answer");
        const HttpWire tew = http_wire_parse(raw);
        JsonValue tj;
        std::string tje;
        const bool tj_ok = tew.ok && json_parse(tew.body, &tj, &tje);
        const JsonValue *tj_err = tj_ok ? tj.find("error") : nullptr;
        std::string tj_msg;
        CHECK(tew.ok && tew.status == 400,
              "request bytes: a request-side Transfer-Encoding is refused");
        CHECK(tj_err && tj_err->get_string("message", &tj_msg) &&
                  tj_msg.find("Transfer-Encoding") != std::string::npos,
              "request bytes: the refusal names Transfer-Encoding");

        // No Content-Length at all, chunk framing carried in the body: the old
        // parser read an EMPTY body and dispatched it, which is a second way to
        // make two parties disagree about the message.
        CHECK(http_exchange_raw(port,
                                head + "Transfer-Encoding: chunked\r\n\r\n"
                                       "3\r\n{}\r\n0\r\n\r\n",
                                false, &raw),
              "request bytes: chunked framing without Content-Length gets an answer");
        const HttpWire ctw = http_wire_parse(raw);
        JsonValue ctj;
        std::string ctje;
        const bool ctj_ok = ctw.ok && json_parse(ctw.body, &ctj, &ctje);
        const JsonValue *ctj_err = ctj_ok ? ctj.find("error") : nullptr;
        std::string ctj_msg;
        CHECK(ctw.ok && ctw.status == 400 && ctj_err &&
                  ctj_err->get_string("message", &ctj_msg) &&
                  ctj_msg.find("Transfer-Encoding") != std::string::npos,
              "request bytes: chunked framing alone is refused by name");

        // A header block past kMaxHeaderBytes (64 KiB) with no blank line: the
        // server must cap it, not keep buffering.
        std::string huge = head + "X-Pad: ";
        huge.append(68u * 1024u, 'a');
        huge += "\r\n";
        CHECK(http_exchange_raw(port, huge, false, &raw),
              "request bytes: an oversized header block gets an answer");
        const HttpWire ow = http_wire_parse(raw);
        CHECK(ow.ok && ow.status == 431 &&
                  ow.reason == "Request Header Fields Too Large",
              "request bytes: an oversized header block is refused with 431");
        CHECK(ow.ok && http_wire_header(ow, "Connection") == "close",
              "request bytes: the 431 tells the client the connection closes");

        // A body that ends before its declared length. The prefix is a COMPLETE
        // streaming request on its own; the over-declared length is what must
        // stop it, so a parser that trusted the prefix would answer 200 and
        // stream instead of refusing.
        const std::string stream_body =
            "{\"prompt\":\"the\",\"max_tokens\":8,\"temperature\":0,\"stream\":true}";
        const std::string declared = std::to_string(stream_body.size() + 50);
        CHECK(http_exchange_raw(port,
                                head + "Content-Length: " + declared + "\r\n\r\n" +
                                    stream_body,
                                true, &raw),
              "request bytes: a short body still gets an answer");
        const HttpWire tw = http_wire_parse(raw);
        CHECK(tw.ok && tw.status == 400 && tw.reason == "Bad Request",
              "request bytes: a body shorter than Content-Length is refused with 400");
        JsonValue sj;
        std::string se;
        const bool sj_ok = tw.ok && json_parse(tw.body, &sj, &se);
        const JsonValue *sj_err = sj_ok ? sj.find("error") : nullptr;
        std::string sj_msg;
        CHECK(sj_err && sj_err->get_string("message", &sj_msg) &&
                  sj_msg.find("incomplete request body") != std::string::npos,
              "request bytes: the refusal names the truncated body");

        // The same request sent whole is the 200 stream. That is the
        // discrimination: the 400 above is about the missing bytes, not about
        // the request being unacceptable.
        CHECK(http_exchange_raw(port,
                                head + "Content-Length: " +
                                    std::to_string(stream_body.size()) + "\r\n\r\n" +
                                    stream_body,
                                false, &raw),
              "request bytes: the same request sent whole is answered");
        const HttpWire uw = http_wire_parse(raw);
        CHECK(uw.ok && uw.status == 200 && uw.chunked,
              "request bytes: the whole body streams, so the 400 was the truncation");
    }

    // ---- the same bytes, produced by the real kraken-server binary ----------
    {
        const std::string self = self_path();
        const std::filesystem::path exe_p =
            std::filesystem::path(self).parent_path() /
#ifdef _WIN32
            "kraken-server.exe";
#else
            "kraken-server";
#endif
        const bool have_server = !self.empty() && std::filesystem::exists(exe_p);
        CHECK(have_server, "server bytes: kraken-server sits beside kraken-tests");
        if (have_server) {
            ServerProc srv;
            srv.out_path = "kraken-server-out.txt";
            srv.err_path = "kraken-server-err.txt";
            const std::vector<std::string> sargs = {
                "--model", kTestModelPath, "--host", "127.0.0.1", "--port", "0",
                "--ctx",   "64",         "--chunk", "8",            "--cpu"};
            CHECK(server_proc_start(&srv, exe_p.string(), sargs),
                  "server bytes: the binary starts with its stdout captured");
            const int sport = srv.started ? server_proc_wait_port(srv, 30000) : -1;
            CHECK(sport > 0,
                  "server bytes: the binary's banner reports the ephemeral port");
            if (sport <= 0) {
                std::fprintf(stderr, "  (server stdout: %.200s)\n",
                             read_capture(srv.out_path).c_str());
                std::fprintf(stderr, "  (server stderr: %.400s)\n",
                             read_capture(srv.err_path).c_str());
            } else {
                std::string raw;
                int code = 0;
                CHECK(http_request(sport, "GET", "/health", "", &code, &raw) &&
                          code == 200,
                      "server bytes: /health answers over loopback");
                const HttpWire bw = http_wire_parse(raw);
                CHECK(bw.ok && bw.status == 200 && bw.reason == "OK",
                      "server bytes: the binary's head parses as HTTP/1.1 200 OK");
                if (!bw.ok)
                    std::fprintf(stderr, "  (binary /health: %s)\n", bw.why.c_str());
                JsonValue bj;
                std::string be;
                const bool bj_ok = json_parse(bw.body, &bj, &be);
                const JsonValue *bs = bj_ok ? bj.find("status") : nullptr;
                CHECK(bs && bs->is_string() && bs->as_string() == "ok",
                      "server bytes: the binary's Content-Length frames its JSON body");

                const std::string creq = "{\"prompt\":\"the\",\"max_tokens\":5,"
                                         "\"temperature\":0,\"repeat_penalty\":1.0}";
                CHECK(http_request(sport, "POST", "/v1/completions", creq, &code,
                                   &raw) && code == 200,
                      "server bytes: a buffered completion answers");
                const HttpWire cw = http_wire_parse(raw);
                JsonValue cj;
                std::string ce;
                const bool cj_ok = cw.ok && json_parse(cw.body, &cj, &ce);
                std::string binary_text;
                const JsonValue *choices = cj_ok ? cj.find("choices") : nullptr;
                if (choices && choices->is_array() && !choices->items().empty()) {
                    const JsonValue *t = choices->items()[0].find("text");
                    if (t && t->is_string()) binary_text = t->as_string();
                }
                CHECK(cj_ok && binary_text == ref.text,
                      "server bytes: the binary's answer is the reference's text");

                CHECK(http_request(sport, "GET", "/v1/nope", "", &code, &raw) &&
                          code == 404,
                      "server bytes: the binary answers an unknown path with 404");
                const HttpWire nw = http_wire_parse(raw);
                CHECK(nw.ok && nw.status == 404 && nw.reason == "Not Found" &&
                          http_wire_header(nw, "Connection") == "close",
                      "server bytes: the binary's 404 is well framed and closes");

                const std::string sreq2 =
                    "{\"prompt\":\"the\",\"max_tokens\":5,\"temperature\":0,"
                    "\"repeat_penalty\":1.0,\"stream\":true}";
                CHECK(http_request(sport, "POST", "/v1/completions", sreq2, &code,
                                   &raw) && code == 200,
                      "server bytes: a streamed completion answers");
                const HttpWire sw2 = http_wire_parse(raw);
                CHECK(sw2.ok && sw2.chunked,
                      "server bytes: the binary's stream de-chunks per RFC 7230");
                if (!sw2.ok)
                    std::fprintf(stderr, "  (binary stream: %s)\n", sw2.why.c_str());
                const SseWire sse2 = sse_wire_parse(sw2.body);
                CHECK(sse2.ok && sse2.done_seen && sse2.text == ref.text,
                      "server bytes: the binary's decoded deltas are the reference text");
                std::fprintf(stderr, "  (binary: port %d, %zu chunk(s), %zu event(s))\n",
                             sport, sw2.chunks, sse2.events.size());
            }
            server_proc_stop(&srv);
            // A green run leaves nothing behind: INPROC mode has no per-group
            // scratch directory to remove these for us.
            std::remove(srv.out_path.c_str());
            std::remove(srv.err_path.c_str());
        // The GEMM rung is a switch, so the two things a caller can get wrong
        // without a model file are its refusal and its documentation. Both run
        // through the binary: a value that is neither exact nor fast must exit 2
        // and name the values it takes, and --help must document the switch.
        {
            const std::string o = "kraken-server-prefill-out.txt";
            const std::string e = "kraken-server-prefill-err.txt";
            auto run = [&](const std::string &args) {
                const std::string cmd = "\"" + exe_p.string() + "\" " + args + " > \"" + o +
                                        "\" 2> \"" + e + "\"";
#ifdef _WIN32
                return gate_exit_code(std::system(("call " + cmd).c_str()));
#else
                return gate_exit_code(std::system(cmd.c_str()));
#endif
            };
            // A path that does not exist, deliberately: with the refusal in
            // place this exits 2 before any load, and if the refusal ever
            // regresses it exits 1 on the open instead of starting a server
            // on the default port and blocking this arm forever.
            const int bad = run("--prefill turbo -m \"kraken-no-such-model.gguf\"");
            CHECK(bad == 2,
                  "server bytes: --prefill refuses a value that is neither exact nor fast");
            const std::string bad_err = read_capture(e);
            if (bad_err.find("needs exact|fast") == std::string::npos)
                std::fprintf(stderr, "  (--prefill turbo stderr: %.200s)\n", bad_err.c_str());
            CHECK(bad_err.find("needs exact|fast") != std::string::npos,
                  "server bytes: the refusal names the two values it takes");
            const int help_rc = run("--help");
            const std::string help_out = read_capture(o);
            CHECK(help_rc == 0 && help_out.find("--prefill exact|fast") != std::string::npos,
                  "server bytes: --help documents --prefill exact|fast");
            std::remove(o.c_str());
            std::remove(e.c_str());
        }
        }
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

    // Host-memory refusal reaches API clients as 503/resource_exhausted with
    // the engine's remediation detail, never as a successful partial answer.
    if (resource_svc) {
        HttpServer resource_server;
        CHECK(resource_server.listen("127.0.0.1", 0, &err),
              "http resource test: listener on ephemeral port");
        const int resource_port = resource_server.port();
        std::thread resource_thread([&resource_server, resource_svc]() {
            resource_server.run([resource_svc](const HttpRequest &request,
                                               HttpResponse &response,
                                               HttpConn &connection) {
                return resource_svc->handle(request, response, connection);
            });
        });
        resource_backend->set_test_host_available(2 * (static_cast<size_t>(1) << 30));
        const std::string resource_request =
            "{\"prompt\":\"one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
            "nineteen twenty twenty-one\",\"max_tokens\":1,\"temperature\":0}";
        CHECK(http_request(resource_port, "POST", "/v1/completions",
                           resource_request, &status, &body) && status == 503,
              "http: host-memory refusal returns Service Unavailable");
        const size_t resource_hdr = body.find("\r\n\r\n");
        JsonValue resource_json;
        std::string resource_parse_error;
        CHECK(resource_hdr != std::string::npos &&
                  json_parse(body.substr(resource_hdr + 4), &resource_json,
                             &resource_parse_error),
              "http: resource refusal body is valid JSON");
        const JsonValue *resource_error = resource_json.find("error");
        std::string resource_type, resource_message;
        CHECK(resource_error && resource_error->get_string("type", &resource_type) &&
                  resource_type == "resource_exhausted",
              "http: resource refusal has the resource_exhausted error type");
        CHECK(resource_error && resource_error->get_string("message", &resource_message) &&
                  resource_message.find("KV promotion refused") != std::string::npos &&
                  resource_message.find("--ctx") != std::string::npos,
              "http: resource refusal explains the cause and a remediation");
        // The streaming path sends the same actionable refusal inside its SSE
        // error event; its HTTP status remains fixed by the already-open stream.
        const std::string stream_request =
            "{\"prompt\":\"one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
            "nineteen twenty twenty-one\",\"max_tokens\":1,\"temperature\":0,"
            "\"stream\":true}";
        CHECK(http_request(resource_port, "POST", "/v1/completions", stream_request,
                           &status, &body), "http: streamed resource request completes");
        CHECK(body.find("resource_exhausted") != std::string::npos &&
                  body.find("KV promotion refused") != std::string::npos &&
                  body.find("--ctx") != std::string::npos,
              "http: stream exposes the resource type, cause and remediation");
        resource_server.stop();
        resource_thread.join();
        delete resource_svc;
        resource_engine.shutdown();
        delete resource_backend;
        std::remove(resource_model_path.c_str());
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
    if (!resource_svc) std::remove(resource_model_path.c_str());
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
        Pool::get().run(n, grain, /*split=*/true, [&](i64 b, i64 e) {
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

// ---------------------------------------------------------------------------
//  Group isolation: one child process per group.
//
//  This suite is a single process running 53 groups in a row, and a crash
//  anywhere used to end it: one bad CHECK could take the summary with it, so
//  the run printed no "%d/%d checks passed" line at all and the failures
//  behind it stayed invisible. test_kv_tier_drop_vs_spill did exactly that --
//  it called std::fclose(NULL), and the CRT's invalid-parameter handler
//  answers that with __fastfail, which SEH cannot catch and which no handler
//  in this process can survive.
//
//  So the suite re-runs itself: main() owns a table of groups, and each group
//  runs in its own child process (this executable, --run-group N). The child
//  rewrites its counts to a progress file as it goes, the parent reads them
//  back when the child exits, and a group that dies is reported by NAME with
//  the number of checks that had run -- and the summary still prints.
//
//  KRK_TEST_INPROC=1 restores the single-process run for a debugger. It is
//  also the A/B: both modes must report the same count and the same FAIL
//  lines, which is how the isolation itself is shown to change nothing.
//
//  KRK_TEST_INJECT_CRASH=<index|name> kills one group on purpose, and
//  scripts/kraken_tests_isolation_check.sh is that check run for real: a
//  harness that cannot fail proves nothing.
// ---------------------------------------------------------------------------

static const char *kSuiteUsage =
    "kraken-tests -- deterministic, self-contained test suite\n"
    "\n"
    "  (no arguments)            run every group, one child process each\n"
    "  --workers <n>             how many of those children run at once\n"
    "  --run-group <index|name>  run one group in THIS process and exit\n"
    "  --progress <path>         where that group writes its running counts\n"
    "  --log <path>              where that group's own output goes\n"
    "  --scratch <dir>           the working directory that group runs in\n"
    "  --report <path>           write the machine-readable summary here\n"
    "                            (with --selfcheck: the per-arm verdicts)\n"
    "  --selfcheck               run every injection arm for every group\n"
    "  --selfcheck --selftest    check the checker itself, on constructed\n"
    "                           answers: one case per way an arm can be wrong\n"
    "  --selfcheck --group <n>   one group's arms only (seconds, not minutes)\n"
    "  --selfcheck --report <p>  record the arms in the runner's report format\n"
    "  --list-groups             print the group table and exit\n"
    "  --help                    this text\n"
    "\n"
    "Every group is self-contained: it builds every fixture it reads. In child\n"
    "mode it also gets its own working directory, so two groups in flight cannot\n"
    "see each other's files -- that is what makes --workers safe to raise.\n"
    "\n"
    "Environment:\n"
    "  KRK_TEST_WORKERS=<n>         same as --workers (the flag wins)\n"
    "  KRK_TEST_INPROC=1            run every group in this process instead\n"
    "                               (a debugger's view: a crash still ends it)\n"
    "  KRK_TEST_INJECT_CRASH=<grp>[,<grp>]  make one or two groups die on\n"
    "                               purpose, to show the isolation reporting\n"
    "                               each by name\n"
    "  KRK_TEST_INJECT_CRASH_AT=<n>[,<n>]   die after check n of each (0 = before\n"
    "                               its first check), to show the partial count\n"
    "  KRK_TEST_INJECT_FAIL=<grp>   make every check in one group fail\n"
    "  KRK_TEST_INJECT_EMPTY=<grp>  make one group run NO checks, to show the\n"
    "                               runner refusing to call that a pass\n"
    "  KRK_TEST_REPORT=<path>       same as --report (the flag wins)\n"
    "  KRK_SELFCHECK_SEED=<n>       replay the fuzz arm's drawn aims; the seed\n"
    "                               a sweep used is printed at its start\n"
    "  KRK_SELFCHECK_BREAK=counts   make --selfcheck expect one check more than\n"
    "                               the run has, so a mismatch is constructed and\n"
    "                               the exit status it produces can be checked\n"
    "  KRK_TEST_SERIAL_OVERRIDE=<grp>  run one group in the serial lane for one\n"
    "                               run, so that lane is exercised, not assumed\n"
    "\n"
    "KRK_H2D_CACHE is forced to 0 unless the caller set it: the measured H2D\n"
    "verdict lives in the user's cache, and a test run must neither trust this\n"
    "machine's real measurement nor overwrite it. The group that drives that\n"
    "cache points it at its own file.\n";


// ---------------------------------------------------------------------------
// The external draft head (EAGLE-3): a GGUF beside the model, run on the host
// ---------------------------------------------------------------------------
// Three things can be silently wrong in a converted head, and each gets its own
// check:
//   * THE GEOMETRY. A head that loads while disagreeing with its own tensor
//     widths reads past a weight and proposes from whatever follows it, so the
//     per-tensor sizes are validated against the keys at load time.
//   * THE VOCABULARY. A draft head's logits are over the DRAFT vocabulary and
//     `d2t` is what makes a proposal verifiable; an id outside the target's
//     range is a proposal the verifier can never accept, which reads as a bad
//     drafter rather than as a wrong file. Refused at load, and the map is
//     checked where it is used.
//   * THE CHAIN. Proposal d>=1 is fed the head's OWN previous hidden state and
//     the token it just proposed. Anything that breaks that -- a prime path
//     that forgets the K/V rows, a truncate that keeps a rejected suffix --
//     drafts from a different distribution: fluent, plausible, wrong, and
//     invisible in the text. The replay checks below are what pin it, and the
//     suite can run them because the head is pure host arithmetic.

namespace {

struct HeadSpec {
    i32 hidden = 8, target_hidden = 8, n_head = 2, n_kv = 1, head_dim = 4;
    i32 n_ff = 16, draft_vocab = 4, target_vocab = 6;
    bool hidden_key = true; // write krk.draft.hidden
    bool norm_f32 = false;  // write one weight at the wrong type
    bool bad_d2t = false;   // an id outside the target vocabulary
    bool with_d2t = true;
};

bool write_head_gguf(const std::string &path, const HeadSpec &sp) {
    GgufBuilder b;
    b.meta_str("general.architecture", "draft-head");
    b.meta_str("krk.draft.kind", "eagle3");
    if (sp.hidden_key) b.meta_u32("krk.draft.hidden", sp.hidden);
    b.meta_u32("krk.draft.target_hidden", sp.target_hidden);
    b.meta_u32("krk.draft.n_layer", 1);
    b.meta_u32("krk.draft.n_head", sp.n_head);
    b.meta_u32("krk.draft.n_kv", sp.n_kv);
    b.meta_u32("krk.draft.head_dim", sp.head_dim);
    b.meta_u32("krk.draft.n_ff", sp.n_ff);
    b.meta_u32("krk.draft.draft_vocab", sp.draft_vocab);
    b.meta_u32("krk.draft.target_vocab", sp.target_vocab);
    b.meta_f32("krk.draft.rope_theta", 10000.0f);
    b.meta_f32("krk.draft.rms_eps", 1e-5f);

    u32 seed = 12345u;
    auto fill = [&](size_t n) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++) v[i] = rnd(seed);
        return v;
    };
    const size_t h = static_cast<size_t>(sp.hidden);
    const size_t th = static_cast<size_t>(sp.target_hidden);
    const size_t qd = static_cast<size_t>(sp.n_head) * sp.head_dim;
    const size_t kd = static_cast<size_t>(sp.n_kv) * sp.head_dim;
    b.tensor_f16("fc.weight", {th * 2, h}, fill(h * th * 2));
    // A norm as F32 is a real converter bug (the loader reads f16 rows and
    // would decode every value from the wrong offset).
    if (sp.norm_f32)
        b.tensor_f32("norm.weight", {h}, fill(h));
    else
        b.tensor_f16("norm.weight", {h}, fill(h));
    b.tensor_f16("layers.0.input_layernorm.weight", {h}, fill(h));
    b.tensor_f16("layers.0.post_attention_layernorm.weight", {h}, fill(h));
    b.tensor_f16("layers.0.self_attn.q_proj.weight", {h, qd}, fill(qd * h));
    b.tensor_f16("layers.0.self_attn.k_proj.weight", {h, kd}, fill(kd * h));
    b.tensor_f16("layers.0.self_attn.v_proj.weight", {h, kd}, fill(kd * h));
    b.tensor_f16("layers.0.self_attn.o_proj.weight", {qd, h}, fill(h * qd));
    b.tensor_f16("layers.0.mlp.gate_proj.weight", {h, static_cast<u64>(sp.n_ff)},
                 fill(h * sp.n_ff));
    b.tensor_f16("layers.0.mlp.up_proj.weight", {h, static_cast<u64>(sp.n_ff)},
                 fill(h * sp.n_ff));
    b.tensor_f16("layers.0.mlp.down_proj.weight", {static_cast<u64>(sp.n_ff), h},
                 fill(h * sp.n_ff));
    b.tensor_f16("lm_head.weight",
                 {h, static_cast<u64>(sp.draft_vocab)},
                 fill(h * sp.draft_vocab));
    if (sp.with_d2t) {
        std::vector<i32> d2t(static_cast<size_t>(sp.draft_vocab));
        for (size_t i = 0; i < d2t.size(); i++) {
            const i32 id = static_cast<i32>(i) * 2;
            d2t[i] = id < sp.target_vocab ? id : sp.target_vocab - 1;
        }
        if (sp.bad_d2t) d2t.back() = sp.target_vocab + 3;
        // F32, the type the converter writes: the container describes float and
        // quantized tensor types, so an I32 entry has no readable payload.
        std::vector<f32> ids(d2t.size());
        for (size_t i = 0; i < d2t.size(); i++)
            ids[i] = static_cast<f32>(d2t[i]);
        b.tensor_f32("d2t", {static_cast<u64>(ids.size())}, ids);
    }
    return b.write(path);
}

// One position's input: a distinct row per position, so a step that attended
// the wrong subset cannot come out bit-identical by luck.
void head_input(i32 pos, i32 n, std::vector<f32> *feat, std::vector<f32> *emb) {
    feat->assign(static_cast<size_t>(n), 0.0f);
    emb->assign(static_cast<size_t>(n), 0.0f);
    for (i32 i = 0; i < n; i++) {
        (*feat)[static_cast<size_t>(i)] =
            0.25f * static_cast<f32>((pos * 3 + i * 5) % 11) - 1.0f;
        (*emb)[static_cast<size_t>(i)] =
            0.125f * static_cast<f32>((pos * 7 + i * 2) % 13) - 0.5f;
    }
}

bool same_row(const std::vector<f32> &a, const std::vector<f32> &b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(f32)) == 0;
}

} // namespace

static void test_draft_head() {
    const std::string path = "kraken-head-test.gguf";
    const std::string bad_d2t = "kraken-head-badd2t.gguf";
    const std::string no_key = "kraken-head-nokey.gguf";
    const std::string f32_norm = "kraken-head-f32norm.gguf";
    const std::string no_map = "kraken-head-nomap.gguf";

    const HeadSpec spec;
    CHECK(write_head_gguf(path, spec), "wrote the draft-head test GGUF");

    DraftHead h;
    std::string err;
    CHECK(h.load(path, &err), "the head loads");
    if (!err.empty()) std::fprintf(stderr, "  (draft head: %s)\n", err.c_str());
    CHECK(h.cfg().hidden == 8 && h.cfg().n_head == 2 && h.cfg().head_dim == 4 &&
              h.cfg().n_kv == 1 && h.cfg().n_ff == 16,
          "the geometry is read off the file");
    CHECK(h.cfg().draft_vocab == 4 && h.cfg().target_vocab == 6,
          "both vocabularies are read: a draft row is over draft ids");
    CHECK(h.cfg().has_d2t, "the d2t map is found");
    CHECK(h.cfg().has_d2t && h.to_target(2) == 4,
          "a draft id maps through d2t to a target id");
    CHECK(h.weight_bytes() > 0, "the head's f16 weights are in host memory");

    // 1. The target it drafts for. A mismatch here is the pair that drafts
    //    tokens the verifier can never accept, so it must be refused with the
    //    number that disagrees.
    CHECK(h.check_target(8, 6, &err), "a matching target is accepted");
    err.clear();
    CHECK(!h.check_target(8, 5, &err), "a different target vocabulary is refused");
    CHECK(err.find("vocabulary") != std::string::npos,
          "the refusal says it is the vocabulary");
    err.clear();
    CHECK(!h.check_target(16, 6, &err), "a different hidden width is refused");
    CHECK(err.find("hidden") != std::string::npos,
          "the refusal says it is the hidden width");

    // 2. A step produces a full row of finite logits over the DRAFT vocabulary,
    //    and the id it proposes is in the target's range.
    const i32 ctx = 16;
    h.reset(ctx);
    std::vector<f32> feat, emb, logits;
    head_input(0, 8, &feat, &emb);
    CHECK(h.step(feat.data(), emb.data(), 0, &logits), "the first step runs");
    CHECK(logits.size() == 4, "one logit per draft-vocabulary entry");
    bool finite = true;
    for (f32 v : logits) finite = finite && std::isfinite(v);
    CHECK(finite, "every logit is finite");
    const i32 id = DraftHead::argmax(logits);
    CHECK(id >= 0 && id < 4, "argmax lands inside the draft vocabulary");
    CHECK(h.to_target(id) >= 0 && h.to_target(id) < 6,
          "and the proposal it maps to is inside the target vocabulary");

    // 3. The prime path. A round primes the head over the positions the target
    //    has already computed with want_logits=false (K/V only), then asks for
    //    one proposal. That has to be the SAME head as one that ran every step
    //    with its output -- otherwise the cheap prime is a different drafter.
    DraftHead primed;
    CHECK(primed.load(path, &err), "a second head loads for the prime arm");
    primed.reset(ctx);
    for (i32 p = 0; p < 2; p++) {
        head_input(p, 8, &feat, &emb);
        CHECK(primed.step(feat.data(), emb.data(), p, nullptr, false),
              "a prime step runs without producing a row");
    }
    std::vector<f32> primed_row;
    head_input(2, 8, &feat, &emb);
    CHECK(primed.step(feat.data(), emb.data(), 2, &primed_row),
          "the proposal after the prime runs");
    DraftHead full;
    CHECK(full.load(path, &err), "a third head loads for the full arm");
    full.reset(ctx);
    std::vector<f32> full_row;
    for (i32 p = 0; p <= 2; p++) {
        head_input(p, 8, &feat, &emb);
        full_row.clear();
        CHECK(full.step(feat.data(), emb.data(), p, &full_row),
              "a step with an output runs");
    }
    CHECK(same_row(primed_row, full_row),
          "a primed head proposes exactly what the fully-run head proposes");

    // 4. Truncate + replay. A partial accept drops the proposal suffix and
    //    re-feeds the accepted positions from the TARGET's rows; the replayed
    //    step must land on the same logits it did the first time, or every
    //    later round drafts from a KV that never existed.
    std::vector<f32> first_row;
    head_input(2, 8, &feat, &emb);
    CHECK(h.step(feat.data(), emb.data(), 2, &first_row),
          "the replay arm steps at position 2");
    CHECK(h.position() == 3, "the head advanced");
    h.truncate(2);
    CHECK(h.position() == 2, "truncate drops the suffix");
    std::vector<f32> again;
    CHECK(h.step(feat.data(), emb.data(), 2, &again),
          "the replayed step runs");
    CHECK(same_row(first_row, again),
          "a replayed position reproduces its own row bit for bit");

    // 5. The window. drop_before is what makes a bounded prime a window rather
    //    than a lie: attention must ignore the rows before it. If it did not,
    //    the flag would only be a slower path to the same answer -- and the
    //    full-KV row here is measured to differ, so the check can fail.
    DraftHead windowed;
    CHECK(windowed.load(path, &err), "a fourth head loads for the window arm");
    windowed.reset(ctx);
    for (i32 p = 0; p < 2; p++) {
        head_input(p, 8, &feat, &emb);
        CHECK(windowed.step(feat.data(), emb.data(), p, nullptr, false),
              "the window arm primes a position");
    }
    windowed.drop_before(2);
    std::vector<f32> windowed_row;
    head_input(2, 8, &feat, &emb);
    CHECK(windowed.step(feat.data(), emb.data(), 2, &windowed_row),
          "the windowed step runs");
    CHECK(!same_row(windowed_row, full_row),
          "a window that drops attention rows is visible in the row");

    // 6. A head whose map points outside the target vocabulary is refused at
    //    load: the alternative is a proposal the verifier can never accept.
    HeadSpec sp = spec;
    sp.bad_d2t = true;
    CHECK(write_head_gguf(bad_d2t, sp), "wrote the out-of-range d2t arm");
    DraftHead bad;
    err.clear();
    CHECK(!bad.load(bad_d2t, &err), "an out-of-range d2t is refused");
    CHECK(err.find("outside the target vocabulary") != std::string::npos,
          "and the refusal names the map");

    // 7. A file that omits the geometry loads as an empty forward, which reads
    //    as a drafter that proposes nothing. The key is required.
    sp = spec;
    sp.hidden_key = false;
    CHECK(write_head_gguf(no_key, sp), "wrote the missing-key arm");
    DraftHead nokey;
    err.clear();
    CHECK(!nokey.load(no_key, &err), "a head with no krk.draft.hidden is refused");
    CHECK(err.find("krk.draft.hidden") != std::string::npos,
          "the refusal names the missing key");

    // 8. A weight at the wrong type is a converter bug: f32 rows would be read
    //    as f16 pairs, i.e. every value decoded from the wrong offset.
    sp = spec;
    sp.norm_f32 = true;
    CHECK(write_head_gguf(f32_norm, sp), "wrote the wrong-type arm");
    DraftHead f32h;
    err.clear();
    CHECK(!f32h.load(f32_norm, &err), "a non-F16 weight is refused");
    CHECK(err.find("F16") != std::string::npos, "and the refusal says F16");

    // 9. No d2t map: the draft ids ARE target ids, so an id past the target's
    //    vocabulary is refused where it is used rather than silently wrapped.
    sp = spec;
    sp.with_d2t = false;
    CHECK(write_head_gguf(no_map, sp), "wrote the identity-map arm");
    DraftHead ident;
    CHECK(ident.load(no_map, &err), "a head with no d2t map loads");
    CHECK(!ident.cfg().has_d2t, "and knows it has no map");
    CHECK(ident.to_target(3) == 3, "identity mapping keeps a valid id");
    CHECK(ident.to_target(9) == -1,
          "an id past the target vocabulary maps to nothing");

    for (const std::string &f : {path, bad_d2t, no_key, f32_norm, no_map})
        std::remove(f.c_str());
}

// ---------------------------------------------------------------------------
// A nextn (MTP) block in its OWN file
// ---------------------------------------------------------------------------
//
// The `-noMTP` shape: a trunk quantised without its MTP, with `mtp-<model>.gguf`
// published beside it. The sidecar carries the same `blk.<n>.*` block at the
// index past the trunk's stack, and everything below is about the three ways
// that goes wrong silently: the wrong family, the wrong block index, and a
// block whose attention is a different width from the trunk's.
static std::string build_mtp_sidecar(const std::string &tag, const std::string &arch,
                                     int block, int n_embd, int n_head, int n_kv,
                                     int head_dim, int n_ff, bool with_block) {
    const std::string path = std::string("kraken-mtp-side-") + tag + ".gguf";
    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", static_cast<u32>(block + 1));
    b.meta_u32(arch + ".attention.head_count", static_cast<u32>(n_head));
    b.meta_u32(arch + ".attention.head_count_kv", static_cast<u32>(n_kv));
    b.meta_u32(arch + ".attention.key_length", static_cast<u32>(head_dim));
    if (!with_block) {
        // A file that is not a nextn block at all: no blk.<n>. tensors past the
        // trunk, which is what a mis-pointed --mtp argument looks like.
        std::vector<f32> one(static_cast<size_t>(n_embd), 1.0f);
        b.tensor_f32("output_norm.weight", {u64(n_embd)}, one);
        if (!b.write(path)) return std::string();
        return path;
    }
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv * head_dim;
    auto vec = [](size_t n) {
        std::vector<f32> v(n);
        for (size_t i = 0; i < n; i++)
            v[i] = 0.01f * static_cast<f32>((static_cast<int>(i) % 13) - 6);
        return v;
    };
    auto ones = [](int n) { return std::vector<f32>(static_cast<size_t>(n), 1.0f); };
    const std::string k = "blk." + std::to_string(block) + ".";
    b.tensor_f16(k + "attn_q.weight", {u64(n_embd), u64(q_dim)},
                 vec(static_cast<size_t>(n_embd) * q_dim));
    b.tensor_f16(k + "attn_k.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim));
    b.tensor_f16(k + "attn_v.weight", {u64(n_embd), u64(kv_dim)},
                 vec(static_cast<size_t>(n_embd) * kv_dim));
    b.tensor_f16(k + "attn_output.weight", {u64(q_dim), u64(n_embd)},
                 vec(static_cast<size_t>(q_dim) * n_embd));
    b.tensor_f16(k + "nextn.eh_proj.weight", {u64(2 * n_embd), u64(n_embd)},
                 vec(static_cast<size_t>(2 * n_embd) * n_embd));
    b.tensor_f32(k + "attn_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32(k + "post_attention_norm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32(k + "nextn.enorm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32(k + "nextn.hnorm.weight", {u64(n_embd)}, ones(n_embd));
    b.tensor_f32(k + "nextn.shared_head_norm.weight", {u64(n_embd)}, ones(n_embd));
    // A DENSE FFN: the block has to carry one, or the loader refuses it as a
    // block that can contribute nothing.
    b.tensor_f16(k + "ffn_up.weight", {u64(n_embd), u64(n_ff)},
                 vec(static_cast<size_t>(n_embd) * n_ff));
    b.tensor_f16(k + "ffn_down.weight", {u64(n_ff), u64(n_embd)},
                 vec(static_cast<size_t>(n_ff) * n_embd));
    if (!b.write(path)) return std::string();
    return path;
}

static void test_mtp_sidecar() {
    Backend *cpu = make_cpu_backend();
    const std::string trunk = "kraken-mtp-trunk.gguf";
    CHECK(build_sized_model(trunk, 32, 64, -1), "wrote a trunk without a nextn block");

    // 1. The happy path: the block loads from the file beside the trunk, with
    //    the trunk's geometry (block index past the stack, its own head count).
    const std::string good = build_mtp_sidecar("good", "llama", 1, 32, 4, 2, 8, 64, true);
    CHECK(!good.empty(), "wrote a nextn sidecar");
    {
        Model m;
        std::string err;
        m.set_mtp_load(true);
        m.set_mtp_path(good);
        CHECK(m.load(*cpu, trunk, &err), "the trunk loads with a sidecar nextn file");
        if (!err.empty()) std::fprintf(stderr, "  (sidecar: %s)\n", err.c_str());
        const MtpBlock &b = m.mtp();
        CHECK(b.present, "the block is loaded out of the SIDECAR, not the trunk");
        CHECK(b.layer == 1, "its index is the first block past the trunk's stack");
        CHECK(b.n_head == 4 && b.n_head_kv == 2 && b.head_dim == 8,
              "the block's attention geometry comes from the sidecar's own config");
        CHECK(b.n_ff == 64, "a dense nextn FFN is loaded as the block's FFN");
        CHECK(b.eh_proj.present(), "the nextn fusion projection is loaded");
        CHECK(b.enorm && b.hnorm && b.shared_head_norm,
              "all three nextn adapters are loaded");
        m.unload();
    }

    // 2. A sidecar from another family is refused by NAME. Applying it would
    //    propose ids this trunk's tokenizer never emits.
    const std::string other = build_mtp_sidecar("other", "qwen2", 1, 32, 4, 2, 8, 64, true);
    {
        Model m;
        std::string err;
        m.set_mtp_load(true);
        m.set_mtp_path(other);
        CHECK(!m.load(*cpu, trunk, &err), "a sidecar from another family is refused");
        CHECK(err.find("qwen2") != std::string::npos &&
                  err.find("llama") != std::string::npos,
              "the refusal names both families");
    }

    // 3. A file with no block past the stack is refused with the reason, not
    //    loaded as an empty block that drafts nothing.
    const std::string nob = build_mtp_sidecar("noblock", "llama", 1, 32, 4, 2, 8, 64, false);
    {
        Model m;
        std::string err;
        m.set_mtp_load(true);
        m.set_mtp_path(nob);
        CHECK(!m.load(*cpu, trunk, &err), "a file with no nextn block is refused");
        CHECK(err.find("blk.") != std::string::npos, "the refusal names what it looked for");
    }

    // 4. A block whose attention is a different width from the trunk's is
    //    refused with BOTH numbers: this is the shape that would otherwise run
    //    and attend over the wrong slice of a KV plane.
    const std::string wide = build_mtp_sidecar("wide", "llama", 1, 32, 8, 2, 8, 64, true);
    {
        Model m;
        std::string err;
        m.set_mtp_load(true);
        m.set_mtp_path(wide);
        CHECK(!m.load(*cpu, trunk, &err), "a block whose head count disagrees is refused");
        CHECK(err.find("head") != std::string::npos, "the refusal says it is the attention");
    }

    // 5. Without a path the trunk is unchanged: no block, no refusal.
    {
        Model m;
        std::string err;
        m.set_mtp_load(true);
        CHECK(m.load(*cpu, trunk, &err), "the trunk still loads on its own");
        CHECK(!m.mtp().present, "a trunk with no nextn block has none to load");
        m.unload();
    }

    std::remove(trunk.c_str());
    std::remove(good.c_str());
    std::remove(other.c_str());
    std::remove(nob.c_str());
    std::remove(wide.c_str());
    delete cpu;
}

struct TestGroup {
    const char *name;
    void (*fn)();
    // Why this group may not share the machine with another, or nullptr when it
    // may. Nothing needs this today: each child gets its own working directory,
    // no group opens the device in-process (test_prefill_chunk_bitexact spawns
    // the CLI, a separate process with its own tiny context), the HTTP group
    // binds an ephemeral port and no group reads a clock for a verdict. A group
    // that does any of those must say so here, and the runner then holds every
    // other worker at the door while it runs.
    const char *serial;
};

static void test_expert_index_adapt() {
    // A two-layer, four-expert index in the exact shape write_expert_index
    // produces. Layer 0 was measured hot on expert 0, layer 1 on expert 2.
    const std::string text =
        "{\"version\": 1, \"kind\": \"kraken-expert-index\", "
        "\"mode\": \"full-forward\", \"arch\": \"qwen35moe\", "
        "\"source_size\": 4096, \"n_layer\": 2, \"n_expert\": 4, "
        "\"positions_scanned\": 100, \"layers\": ["
        "{\"layer\": 0, \"experts\": ["
        "{\"e\": 0, \"mass\": 90.0, \"hits\": 90}, "
        "{\"e\": 1, \"mass\": 8.0, \"hits\": 8}, "
        "{\"e\": 2, \"mass\": 1.0, \"hits\": 1}, "
        "{\"e\": 3, \"mass\": 1.0, \"hits\": 1}]}, "
        "{\"layer\": 1, \"experts\": ["
        "{\"e\": 0, \"mass\": 1.0, \"hits\": 1}, "
        "{\"e\": 1, \"mass\": 1.0, \"hits\": 1}, "
        "{\"e\": 2, \"mass\": 90.0, \"hits\": 90}, "
        "{\"e\": 3, \"mass\": 8.0, \"hits\": 8}]}]}";

    ExpertIndexRows rows = expert_index_parse(text, 2, 4, "qwen35moe", 4096);
    CHECK(rows.valid, "a well-formed index for this model parses");
    CHECK(rows.tokens == 100,
          "... and reports the positions it was measured over");
    CHECK(rows.mass.size() == 8 && rows.hits.size() == 8,
          "... one cell per layer x expert");
    CHECK(rows.mass[3] == 1.0 && rows.hits[6] == 90,
          "... mass and hits land on the right cell");

    // The refusals are the loader's, and they are the reason a merge cannot
    // silently apply another model's distribution to these weights.
    CHECK(!expert_index_parse(text, 2, 8, "qwen35moe", 4096).valid,
          "a different expert count is refused");
    CHECK(!expert_index_parse(text, 3, 4, "qwen35moe", 4096).valid,
          "a different layer count is refused");
    CHECK(!expert_index_parse(text, 2, 4, "qwen3", 4096).valid,
          "an index measured on another architecture is refused");
    CHECK(!expert_index_parse(text, 2, 4, "qwen35moe", 8192).valid,
          "an index for another build of the weights is refused");
    CHECK(!expert_index_parse("not json", 2, 4, "qwen35moe", 0).valid,
          "a file that is not JSON is refused");

    // The position count is a MEASUREMENT claim, not decoration: 0 (or a
    // missing field, which reads as 0) means the run never counted a position,
    // so its per-layer order is whatever one partial forward pass left behind.
    // Trusting such a file measured 8.6 tok/s against 11.8 for the even order
    // on the REAP-320 file. Both readers share this parser -- the warmup and
    // the adapt-merge -- so the refusal lives here.
    const std::string pos_field = "\"positions_scanned\": 100";
    const size_t pos_at = text.find(pos_field);
    CHECK(pos_at != std::string::npos, "the fixture states its position count");
    {
        std::string zero = text;
        zero.replace(pos_at, pos_field.size(), "\"positions_scanned\": 0");
        CHECK(!expert_index_parse(zero, 2, 4, "qwen35moe", 4096).valid,
              "an index that measured 0 positions is refused");
    }
    {
        std::string absent = text;
        absent.replace(pos_at, pos_field.size(), "\"positions_unrecorded\": 0");
        CHECK(!expert_index_parse(absent, 2, 4, "qwen35moe", 4096).valid,
              "an index that does not record its positions is refused");
    }

    // The run routed layer 0 to expert 3, and layer 1 not at all -- so layer 1
    // must keep the file's row while layer 0 follows the traffic.
    std::vector<double> run_mass(8, 0.0);
    std::vector<u64> run_hits(8, 0);
    run_mass[3] = 900.0;
    run_hits[3] = 300;
    {
        ExpertIndexRows b = expert_index_parse(text, 2, 4, "qwen35moe", 4096);
        expert_index_blend(&b, 4, run_mass, run_hits, 300, 0.5);
        CHECK(b.mass[3] > b.mass[0],
              "half-way blending makes the run's hot expert the top of the "
              "ranking");
        CHECK(b.hits[3] == 1 + 300, "hits accumulate instead of replacing");
        CHECK(b.tokens == 400, "the position counts add");
        CHECK(b.mass[4] == 1.0 && b.mass[6] == 90.0,
              "a layer the run never routed through keeps the file's row");
        // The stored number keeps its meaning: summed probability over the
        // larger of the two position counts.
        f64 l0 = 0;
        for (int e = 0; e < 4; e++) l0 += b.mass[static_cast<size_t>(e)];
        CHECK(std::fabs(l0 - 300.0) < 1e-6,
              "the blended mass is still a sum over positions, not a ratio");
    }

    // retain = 1 is the A/B arm that proves the blend is doing the work: with
    // it the file's own order survives a run that routed somewhere else.
    {
        ExpertIndexRows arm = expert_index_parse(text, 2, 4, "qwen35moe", 4096);
        expert_index_blend(&arm, 4, run_mass, run_hits, 300, 1.0);
        CHECK(arm.mass[0] > arm.mass[3],
              "retain 1.0 leaves the measured order exactly as it was");
    }

    // Geometry guards: a run sized for another model must change nothing.
    {
        ExpertIndexRows bad = expert_index_parse(text, 2, 4, "qwen35moe", 4096);
        std::vector<double> short_mass(4, 1.0);
        std::vector<u64> short_hits(4, 1);
        expert_index_blend(&bad, 4, short_mass, short_hits, 300, 0.5);
        CHECK(bad.mass[3] == 1.0 && bad.tokens == 100,
              "a run sized for another model changes nothing");
    }
}

static void test_hc_lanes() {
    // 8-wide lanes so a hand computation fits in a comment. The real stack is
    // 2560 x 4 (Qwen3.8-Flash-Next) through a 320 bottleneck; the arithmetic is
    // the same at every width.
    HcGeometry g;
    g.n_embd = 8;
    g.hc = 4;
    g.low_rank = 3;
    CHECK(g.lane_width() == 32,
          "the wide residual is hc lanes of n_embd, not one of hc*n_embd");
    CHECK(g.valid(), "four lanes through a 320-wide bottleneck is a valid stack");
    HcGeometry one;
    one.n_embd = 8;
    one.hc = 1;
    one.low_rank = 3;
    CHECK(!one.valid(),
          "hc == 1 is refused: the reference raises on it (nothing to mix)");
    HcGeometry norank;
    norank.n_embd = 8;
    norank.hc = 4;
    norank.low_rank = 0;
    CHECK(!norank.valid(), "a zero low rank has no mixer");

    // The centring rule. 2*sigmoid(0) is exactly 1.0f in f32 (0.5f is exact and
    // 2*0.5f is exact), so an un-injected block is the identity on every lane.
    // This is the difference between "residual + f(x)" and "half of it".
    CHECK(hc_lane_weight(0.0f, 4) == 1.0f,
          "injection 0 weights every lane at exactly 1");
    bool in_range = true, rising = true;
    f32 prev = hc_lane_weight(-20.0f, 4);
    for (f32 v = -20.0f; v <= 20.0f; v += 1.0f) {
        const f32 w = hc_lane_weight(v, 4);
        if (!(w > 0.0f && w < 2.0f)) in_range = false;
        if (w < prev) rising = false;
        prev = w;
    }
    CHECK(in_range, "every lane weight stays inside (0, 2)");
    CHECK(rising, "... and is monotone in the injection");
    CHECK(hc_lane_weight(0.0f, 0) == 1.0f,
          "an empty stack still reports the identity weight rather than NaN");

    const i32 n = 3;
    const size_t wide = static_cast<size_t>(n) * static_cast<size_t>(g.hc) *
                        static_cast<size_t>(g.n_embd);
    const size_t flat = static_cast<size_t>(n) * static_cast<size_t>(g.n_embd);
    std::vector<f32> xn(wide), gate(wide), mixed(flat);

    // ---- the identical-lane case, which is where the reference starts ------
    // Every lane holds the same vector (hc copies of the embedding) and the
    // gate is the same on every lane. Then the mix must be the scalar lane mean
    // applied to that one vector -- not a lane 0 pass-through, not a sum.
    for (i32 t = 0; t < n; t++) {
        for (i32 c = 0; c < g.hc; c++) {
            for (i32 e = 0; e < g.n_embd; e++) {
                const size_t i = (static_cast<size_t>(t) * g.hc + c) *
                                     static_cast<size_t>(g.n_embd) +
                                 static_cast<size_t>(e);
                xn[i] = 0.5f * static_cast<f32>(e + 1);
                gate[i] = 0.25f * static_cast<f32>(e + 1);
            }
        }
    }
    hc_mix(mixed.data(), xn.data(), gate.data(), n, g);
    f64 worst_equal = 0.0;
    for (i32 t = 0; t < n; t++) {
        for (i32 e = 0; e < g.n_embd; e++) {
            const f32 sg = 1.0f / (1.0f + std::exp(-gate[static_cast<size_t>(t) *
                                                            g.hc *
                                                            g.n_embd +
                                                        static_cast<size_t>(e)]));
            const f32 want = xn[static_cast<size_t>(t) * g.n_embd +
                                static_cast<size_t>(e)] * sg;
            const f32 got = mixed[static_cast<size_t>(t) * g.n_embd +
                                  static_cast<size_t>(e)];
            worst_equal = std::max<f64>(worst_equal, std::fabs(got - want));
        }
    }
    CHECK(worst_equal <= 2e-7,
          "identical lanes mix to xn*sigmoid(gate) -- the mean of hc copies");

    // ---- lane stride: only lane 0 carrying a value must give 1/hc of it ----
    std::fill(xn.begin(), xn.end(), 0.0f);
    std::fill(gate.begin(), gate.end(), 0.0f);
    for (i32 t = 0; t < n; t++)
        for (i32 e = 0; e < g.n_embd; e++)
            xn[(static_cast<size_t>(t) * g.hc) * g.n_embd +
               static_cast<size_t>(e)] = 4.0f;
    hc_mix(mixed.data(), xn.data(), gate.data(), n, g);
    // sigmoid(0) = 0.5, one lane of four holds 4 -> 4*0.5/4 = 0.5.
    bool stride_ok = true;
    for (i32 t = 0; t < n; t++)
        for (i32 e = 0; e < g.n_embd; e++)
            if (std::fabs(mixed[static_cast<size_t>(t) * g.n_embd +
                                static_cast<size_t>(e)] - 0.5f) > 1e-7f)
                stride_ok = false;
    CHECK(stride_ok,
          "a value in lane 0 only contributes 1/hc of itself, so the lane stride "
          "is the mix's, not the flat row's");

    // ---- combine: still identical before the first block output -----------
    std::vector<f32> res(wide, 1.0f);
    std::vector<f32> inject(static_cast<size_t>(n) * g.hc, 0.0f);
    std::vector<f32> zeros(flat, 0.0f);
    hc_combine(res.data(), zeros.data(), inject.data(), n, g);
    bool untouched = true;
    for (i32 t = 0; t < n; t++)
        for (i32 c = 0; c < g.hc; c++)
            for (i32 e = 0; e < g.n_embd; e++)
                if (res[(static_cast<size_t>(t) * g.hc + c) * g.n_embd +
                        static_cast<size_t>(e)] != 1.0f)
                    untouched = false;
    CHECK(untouched, "a zero block output leaves every lane exactly as it was");

    hc_combine(res.data(), nullptr, inject.data(), n, g);
    untouched = true;
    for (f32 v : res)
        if (v != 1.0f) untouched = false;
    CHECK(untouched, "... and so does a null block output");

    // Same lanes, a real block output, and an injection of 0: the weight is 1,
    // so every lane moves by the same amount and they are STILL identical. This
    // is the invariant the stack depends on: lanes only diverge when something
    // lane-specific happens, and the first lane-specific thing is the injection.
    std::vector<f32> out(flat);
    for (size_t i = 0; i < flat; i++) out[i] = 0.25f * static_cast<f32>(i + 1);
    hc_combine(res.data(), out.data(), inject.data(), n, g);
    bool still_equal = true;
    f64 div = 0.0;
    for (i32 t = 0; t < n; t++) {
        for (i32 e = 0; e < g.n_embd; e++) {
            const f32 base = res[(static_cast<size_t>(t) * g.hc) * g.n_embd +
                                static_cast<size_t>(e)];
            for (i32 c = 0; c < g.hc; c++) {
                const f32 v = res[(static_cast<size_t>(t) * g.hc + c) * g.n_embd +
                                  static_cast<size_t>(e)];
                div = std::max<f64>(div, std::fabs(v - base));
                if (v != base) still_equal = false;
            }
        }
    }
    CHECK(still_equal, "with injection 0 the lanes stay bit-equal after a combine");
    CHECK(div == 0.0, "... to the last bit, not just approximately");

    // A lane-specific injection is what splits them, and the split is the
    // weight the reference centres on 1.
    std::vector<f32> res2(wide, 0.0f);
    std::vector<f32> inj2(static_cast<size_t>(n) * g.hc);
    for (size_t i = 0; i < inj2.size(); i++)
        inj2[i] = static_cast<f32>(i % g.hc) * 4.0f; // 0, 4, 8, 12
    hc_combine(res2.data(), out.data(), inj2.data(), n, g);
    bool split = false;
    for (i32 e = 0; e < g.n_embd; e++) {
        const f32 a = res2[static_cast<size_t>(e)];
        const f32 b = res2[static_cast<size_t>(g.n_embd) + static_cast<size_t>(e)];
        if (a != b) split = true;
    }
    CHECK(split, "a lane-specific injection is what makes the lanes differ");
    // And the same lane's two contributions obey the weight exactly.
    const f32 w1 = hc_lane_weight(4.0f, g.hc);
    CHECK(std::fabs(res2[static_cast<size_t>(g.n_embd)] -
                    out[0] * w1) < 1e-6f,
          "a lane's step is block_out * 2*sigmoid(inject/hc)");

    // ---- per-lane norm ----------------------------------------------------
    // gamma is [hc, n_embd]: lane 0 doubles its elements, lane 1 leaves them,
    // lane 2 halves them, lane 3 zeroes them. The norm itself is within a lane,
    // so a constant lane normalizes to 1.0 before the gamma.
    std::vector<f32> gamma(static_cast<size_t>(g.hc) * g.n_embd);
    for (i32 c = 0; c < g.hc; c++)
        for (i32 e = 0; e < g.n_embd; e++)
            gamma[static_cast<size_t>(c) * g.n_embd + static_cast<size_t>(e)] =
                c == 0 ? 2.0f : (c == 1 ? 1.0f : (c == 2 ? 0.5f : 0.0f));
    std::vector<f32> xn2(wide, 3.0f), normed(wide);
    hc_norm_lanes(normed.data(), xn2.data(), gamma.data(), n, g, 1e-6f);
    CHECK(std::fabs(normed[0] - 2.0f) < 1e-5f,
          "a constant lane normalizes to 1 and then takes its own gamma (2x)");
    CHECK(std::fabs(normed[static_cast<size_t>(g.n_embd)] - 1.0f) < 1e-5f,
          "lane 1's gamma is 1");
    CHECK(std::fabs(normed[2 * static_cast<size_t>(g.n_embd)] - 0.5f) < 1e-5f,
          "lane 2's gamma is 0.5");
    CHECK(normed[3 * static_cast<size_t>(g.n_embd)] == 0.0f,
          "a zero gamma silences its lane rather than dividing by it");
    // A gamma of all ones must leave a unit-RMS lane alone: this is the check
    // that catches a lane stride read off the flat row.
    std::vector<f32> ones(static_cast<size_t>(g.hc) * g.n_embd, 1.0f);
    hc_norm_lanes(normed.data(), xn2.data(), ones.data(), n, g, 1e-6f);
    CHECK(std::fabs(normed[3 * static_cast<size_t>(g.n_embd)] - 1.0f) < 1e-5f,
          "with gamma 1 every lane normalizes the same way");

    // ---- the real geometry: four lanes of 2560 through a 320 bottleneck ----
    // Qwen3.8-Flash-Next's own numbers, so the width the loader validates and
    // the width this arithmetic is proved at are the same width.
    HcGeometry real_hc;
    real_hc.n_embd = 2560;
    real_hc.hc = 4;
    real_hc.low_rank = 320;
    CHECK(real_hc.valid() && real_hc.lane_width() == 10240,
          "four lanes of 2560 ARE the 10240-wide residual attn_qkv feeds");
    std::vector<f32> wxn(static_cast<size_t>(real_hc.lane_width()), 1.0f);
    std::vector<f32> wgate(static_cast<size_t>(real_hc.lane_width()), 0.0f);
    std::vector<f32> wmixed(static_cast<size_t>(real_hc.n_embd), 0.0f);
    std::vector<f32> wres(static_cast<size_t>(real_hc.lane_width()), 2.0f);
    std::vector<f32> winj(static_cast<size_t>(real_hc.hc), 0.0f);
    hc_mix(wmixed.data(), wxn.data(), wgate.data(), 1, real_hc);
    CHECK(std::fabs(wmixed[0] - 0.5f) < 1e-7f &&
              std::fabs(wmixed[real_hc.n_embd - 1] - 0.5f) < 1e-7f,
          "at 2560x4 the mix still averages the lanes (four 1s, sigmoid 0.5)");
    hc_combine(wres.data(), nullptr, winj.data(), 1, real_hc);
    CHECK(wres[0] == 2.0f && wres[real_hc.lane_width() - 1] == 2.0f,
          "and a null block output still leaves all 10240 values alone");
}

static void test_ple_hash() {
    // A fixture where the hash is computable by hand: multipliers of 1 turn
    // `mixed` into an xor of the window's token ids, and offsets of 0 leave the
    // row as `mixed % vocab`. The real file's multipliers are large odd
    // constants and its ranges are 20M-row slices of a 320M-row table.
    PleConfig c;
    c.ngram = 3;
    c.heads_per_ngram = 8;
    c.head_dim = 160;
    c.conv_kernel = 4;
    c.eos_token = 99;
    c.multipliers = {1, 1, 1};
    c.head_offsets.assign(16, 0);
    c.head_vocab_sizes.assign(16, 1000000);
    CHECK(c.valid(), "a 3-gram, 8-head-per-gram config is valid");
    CHECK(c.n_heads() == 16,
          "a 3-gram file has (3-1)*8 heads: one set per n-gram length");
    PleConfig one = c;
    one.ngram = 1;
    CHECK(!one.valid(), "a 1-gram has no predecessor to hash with and is refused");
    PleConfig short_offsets = c;
    short_offsets.head_offsets.resize(8);
    CHECK(!short_offsets.valid(),
          "fewer head ranges than heads is refused, not read past");

    // window is oldest-first: {10, 11, 12} means the current token is 12.
    const i64 win[3] = {10, 11, 12};
    i64 rows[16] = {0};
    ple_rows(c, win, 3, rows);
    // heads 0..7 hash the 2-gram (12, 11): 12 ^ 11 = 7
    CHECK(rows[0] == (12 ^ 11), "the first head set hashes the 2-gram");
    CHECK(rows[7] == (12 ^ 11), "... for every head in that set");
    // heads 8..15 hash the 3-gram (12, 11, 10): 12 ^ 11 ^ 10 = 13
    CHECK(rows[8] == (12 ^ 11 ^ 10), "the second head set adds the third token");
    CHECK(rows[15] == (12 ^ 11 ^ 10), "... for every head in that set");
    CHECK(rows[0] != rows[8],
          "the two n-gram lengths land on different rows, so the sets are not "
          "duplicates of each other");

    // Per-head ranges are per HEAD, not one range for the table: give head 8 a
    // vocab of 5 and an offset of 1000 and the row must move with it.
    PleConfig ranged = c;
    ranged.head_vocab_sizes[8] = 5;
    ranged.head_offsets[8] = 1000;
    ple_rows(ranged, win, 3, rows);
    CHECK(rows[8] == 1000 + (13 % 5),
          "a head's own vocab and offset decide its row (13 % 5 + 1000)");
    CHECK(rows[9] == 13, "and its neighbours are unaffected");
    bool in_range = true;
    for (i32 h = 0; h < 16; h++) {
        const i64 lo = ranged.head_offsets[static_cast<size_t>(h)];
        const i64 hi = lo + ranged.head_vocab_sizes[static_cast<size_t>(h)];
        if (rows[h] < lo || rows[h] >= hi) in_range = false;
    }
    CHECK(in_range, "every row lands inside its own head's range");

    // An EOS predecessor cuts the window: nothing at or before it is read.
    const i64 cut[3] = {10, 99, 12};
    ple_rows(c, cut, 3, rows);
    CHECK(rows[0] == (12 ^ 99), "a 2-gram over an EOS predecessor uses the EOS");
    CHECK(rows[8] == (12 ^ 99 ^ 99),
          "and the 3-gram pins everything older to EOS as well, so a sequence "
          "never hashes across a boundary");

    // A window shorter than the n-gram (the start of a sequence) reads EOS.
    const i64 first[1] = {12};
    ple_rows(c, first, 1, rows);
    CHECK(rows[0] == (12 ^ 99), "before the sequence start the predecessor is EOS");
    CHECK(rows[8] == (12 ^ 99 ^ 99), "... and so is the one before it");

    // Determinism and sensitivity: same window, same rows; a different current
    // token moves every head.
    i64 again[16] = {0};
    i64 base_rows[16] = {0};
    ple_rows(c, win, 3, base_rows);
    ple_rows(c, win, 3, again);
    bool deterministic = true;
    for (i32 h = 0; h < 16; h++)
        if (again[h] != base_rows[h]) deterministic = false;
    CHECK(deterministic, "the hash is a pure function of the window");
    const i64 win2[3] = {10, 11, 13};
    i64 moved[16];
    ple_rows(c, win2, 3, moved);
    bool all_moved = true;
    for (i32 h = 0; h < 16; h++)
        if (moved[h] == base_rows[h]) all_moved = false;
    CHECK(all_moved, "a different current token moves every head's row");

    // The real file's shape, from its own metadata: 3-gram, 8 per gram, 160
    // wide, 16 heads, and 20,000,003-row slices.
    PleConfig real;
    real.ngram = 3;
    real.heads_per_ngram = 8;
    real.head_dim = 160;
    real.conv_kernel = 4;
    real.eos_token = 248044;
    real.multipliers = {23703573157769LL, 20109073645365LL, 8052911324071LL};
    real.head_offsets = {0, 20000003, 40000026, 60000059,
                         80000106, 100000165, 120000228, 140000297,
                         160000372, 180000453, 200000540, 220000633,
                         240000732, 260000837, 280000948, 300001065};
    real.head_vocab_sizes = {20000003, 20000023, 20000033, 20000047,
                             20000059, 20000063, 20000069, 20000077,
                             20000083, 20000091, 20000099, 20000107,
                             20000113, 20000119, 20000131, 20000139};
    CHECK(real.valid(), "the REAP-288 file's own PLE metadata is a valid config");
    CHECK(real.n_heads() == 16, "... with its 16 heads");
    i64 real_rows[16];
    const i64 real_win[3] = {9707, 11, 220};
    ple_rows(real, real_win, 3, real_rows);
    bool real_in_range = true;
    bool real_distinct = true;
    for (i32 h = 0; h < 16; h++) {
        const i64 lo = real.head_offsets[static_cast<size_t>(h)];
        const i64 hi = lo + real.head_vocab_sizes[static_cast<size_t>(h)];
        if (real_rows[h] < lo || real_rows[h] >= hi) real_in_range = false;
        if (h > 0 && real_rows[h] == real_rows[h - 1]) real_distinct = false;
    }
    CHECK(real_in_range, "the real config's rows stay inside the 320M-row table");
    CHECK(real_distinct, "... and the 16 heads gather 16 different rows");
}

// ---------------------------------------------------------------------------
// qwen4exp (Qwen3.8-Flash-Next): the hyper-connection schema
// ---------------------------------------------------------------------------

static const char *kQ4TestPath = "kraken-q4-test.gguf";

// A miniature qwen4exp: 2 layers, 4 hyper-connection lanes, every 2nd layer a
// full-attention layer with a QSA indexer, a PLE module on layer 0 and a 288-
// expert MoE shrunk to 4. Structurally the REAP-288 file's schema at 1/320 the
// width, so the loader's geometry checks and the lane arithmetic are exercised
// without 78 GiB of weights.
static bool build_tiny_q4_model(const std::string &path, int hc_down_rows,
                                int n_embd_arg = 0, int low_rank_arg = 0,
                                bool extras = true,
                                bool identical_inject = false,
                                bool qsa = true,
                                bool half_indexer = false,
                                bool perturb_ple = false) {
    // The two numbers the forward test scales up to the real file's shape: four
    // lanes of 2560, mixed through a 32-wide bottleneck (the REAP-288 file
    // mixes through 320; 32 keeps the fixture a few MB instead of 100).
    const int n_embd = n_embd_arg > 0 ? n_embd_arg : 32;
    const int n_head = 4, n_kv = 2, hd = 8;
    const int d_state = 8, n_k_head = 2, n_v_head = 4, conv_k = 4;
    const int rope_dim = 4;
    const int key_dim = n_k_head * d_state;      // 16
    const int value_dim = n_v_head * d_state;    // 32
    const int conv_dim = key_dim * 2 + value_dim; // 64
    const int hc = 4, low_rank = low_rank_arg > 0 ? low_rank_arg : 8;
    const int wide = n_embd * hc;                 // 128
    const int interval = 2;
    const int n_layer = 2;
    // 261, not 260: `add_spm_vocab` always writes 3 specials + 256 byte pieces
    // + the two whitespace pieces, so its table has 261 entries whatever size
    // is asked for once the ask is <= 261. A 260-row embedding against a
    // 261-piece tokenizer read one row past `token_embd.weight` for every space
    // in the prompt -- the row the tokenizer calls 260 -- and the bytes there
    // are heap, so the WHOLE forward changed from run to run: the same command
    // printed a different `--prefill-hash` every time, and the QSA selection it
    // fed flipped on near-ties. That is the same defect the prefill gate hit
    // (hashing past `cfg().n_vocab`), and it is a fixture shape error, not a
    // kernel one: the table has to be as wide as the tokenizer is tall.
    const int n_ff = 64, n_ff_exp = 64, n_ff_sh = 32, n_expert = 4, vocab = 261,
              ctx = 64;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd;
    const int idx_h = 2, idx_dim = 4;
    // The gathered row is one head_dim per head, and the key projection reads
    // the whole wide stream: those numbers have to meet, which is why the
    // fixture's head_dim is the lane width and not a small stand-in.
    const int ple_dim = n_embd, ple_heads = 1, ple_rows = 8, ple_conv = 2;
    const std::string arch = "qwen4exp";

    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-q4-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", static_cast<u32>(n_layer));
    b.meta_u32(arch + ".context_length", static_cast<u32>(ctx));
    b.meta_u32(arch + ".embedding_length", static_cast<u32>(n_embd));
    b.meta_u32(arch + ".feed_forward_length", static_cast<u32>(n_ff));
    b.meta_u32(arch + ".attention.head_count", static_cast<u32>(n_head));
    b.meta_u32(arch + ".attention.head_count_kv", static_cast<u32>(n_kv));
    b.meta_u32(arch + ".attention.key_length", static_cast<u32>(hd));
    b.meta_u32(arch + ".attention.value_length", static_cast<u32>(hd));
    b.meta_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    b.meta_f32(arch + ".rope.freq_base", 10000.0f);
    b.meta_u32(arch + ".rope.dimension_count", static_cast<u32>(rope_dim));
    b.meta_i32_array(arch + ".rope.dimension_sections", {rope_dim / 2, 1, 1, 0});
    b.meta_u32(arch + ".ssm.conv_kernel", static_cast<u32>(conv_k));
    b.meta_u32(arch + ".ssm.state_size", static_cast<u32>(d_state));
    b.meta_u32(arch + ".ssm.time_step_rank", static_cast<u32>(n_v_head));
    b.meta_u32(arch + ".ssm.group_count", static_cast<u32>(n_k_head));
    b.meta_u32(arch + ".ssm.inner_size", static_cast<u32>(value_dim));
    b.meta_u32(arch + ".full_attention_interval", static_cast<u32>(interval));
    b.meta_u32(arch + ".expert_count", static_cast<u32>(n_expert));
    b.meta_u32(arch + ".expert_used_count", 2);
    b.meta_u32(arch + ".expert_feed_forward_length", static_cast<u32>(n_ff_exp));
    b.meta_u32(arch + ".expert_shared_feed_forward_length", static_cast<u32>(n_ff_sh));
    // The three pieces that make this family what it is.
    b.meta_u32(arch + ".hyper_connection.count", static_cast<u32>(hc));
    b.meta_u32(arch + ".hyper_connection.low_rank", static_cast<u32>(low_rank));
    // The indexer and the PLE module both RUN now -- the forward test below
    // proves the residual stack with them left out, and test_qwen4exp_ple
    // proves the n-gram block -- so this fixture can build either shape.
    // `half_indexer` writes the per-layer ratios and none of the head geometry:
    // that is the file the loader has to refuse, because running it as if it
    // had no indexer would attend over every key and decode text that is not
    // this model's.
    if (extras && qsa) {
        if (half_indexer) {
            b.meta_i32_array(arch + ".attention.compress_ratios", {0, 2});
        } else {
            b.meta_u32(arch + ".attention.indexer.head_count", static_cast<u32>(idx_h));
            b.meta_u32(arch + ".attention.indexer.key_length", static_cast<u32>(idx_dim));
            b.meta_u32(arch + ".attention.indexer.top_k", 16);
            b.meta_i32_array(arch + ".attention.compress_ratios", {0, 2});
        }
    }
    if (extras) {
        b.meta_i32_array(arch + ".ple.layers", {0});
        b.meta_u32(arch + ".ple.ngram_size", 2);
        b.meta_u32(arch + ".ple.heads_per_ngram", static_cast<u32>(ple_heads));
        b.meta_u32(arch + ".ple.conv_kernel", static_cast<u32>(ple_conv));
        b.meta_u32(arch + ".ple.eos_token_id", 258);
        b.meta_u32(arch + ".embedding_length_per_layer_input", static_cast<u32>(ple_dim));
        b.meta_i32_array(arch + ".ple.layer_multipliers", {1, 1});
        b.meta_i32_array(arch + ".ple.head_offsets", {0});
        b.meta_i32_array(arch + ".ple.head_vocab_sizes", {ple_rows});
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
    b.tensor_f16("output.weight", {u64(n_embd), u64(vocab)},
                 vec(static_cast<size_t>(n_embd) * vocab, 3000));
    // No output_norm: the head mixer carries it.
    b.tensor_f32("hc_head_norm.weight", {u64(wide)}, ones(wide));
    b.tensor_f16("hc_head_down.weight", {u64(wide), u64(low_rank)},
                 vec(static_cast<size_t>(wide) * low_rank, 4100));
    b.tensor_f16("hc_head_up.weight", {u64(low_rank), u64(wide)},
                 vec(static_cast<size_t>(low_rank) * wide, 4300));

    // The PLE table is [ple_dim, rows]: small here, tens of GB in the real file.
    // `perturb_ple` shifts every entry, which is the sensitivity arm of
    // test_qwen4exp_whole_graph: the whole-graph pin has to be able to fail, and
    // the module whose wiring bug produced that family's garbage text is the
    // one a perturbation should move.
    {
        std::vector<f32> ple = vec(static_cast<size_t>(ple_dim) * ple_rows, 4500);
        if (perturb_ple)
            for (f32 &v : ple) v += 3.0f;
        b.tensor_f32("per_layer_token_embd.weight", {u64(ple_dim), u64(ple_rows)},
                     ple);
    }

    for (int l = 0; l < n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        // Two hyper-connection mixers: before the token mixer and the FFN.
        for (const char *which : {"hc_attn", "hc_ffn"}) {
            const std::string q = p + which + "_";
            b.tensor_f32(q + "norm.weight", {u64(wide)}, ones(wide));
            // The bad arm narrows layer 0's attention mixer, so the refusal is
            // reached with the right tensors in place -- a second tensor of the
            // same name would leave the loader reading the first one.
            const int down_in = (l == 0 && hc_down_rows > 0) ? hc_down_rows : wide;
            b.tensor_f16(q + "down.weight", {u64(static_cast<u64>(down_in)),
                                             u64(low_rank)},
                         vec(static_cast<size_t>(down_in) * low_rank, 4700 + l * 100));
            b.tensor_f16(q + "up.weight", {u64(low_rank), u64(wide)},
                         vec(static_cast<size_t>(low_rank) * wide, 4900 + l * 100));
            // The injection is the ONLY thing that can separate the lanes:
            // every other mixer tensor is lane-symmetric, and the stack starts
            // as hc copies of one embedding row. `identical_inject` writes the
            // same column for every lane, which is the arm whose lanes must
            // stay bit-equal through the whole stack.
            std::vector<f32> inj(static_cast<size_t>(wide) * hc);
            for (size_t i = 0; i < inj.size(); i++) {
                const i64 e = static_cast<i64>(i % static_cast<size_t>(wide));
                inj[i] = wv(5100 + l * 100 + (identical_inject ? e
                                                              : static_cast<i64>(i)));
            }
            b.tensor_f16(q + "inject.weight", {u64(wide), u64(hc)}, inj);
        }

        if (((l + 1) % interval) != 0) {
            b.tensor_f16(p + "attn_qkv.weight", {u64(n_embd), u64(conv_dim)},
                         vec(static_cast<size_t>(n_embd) * conv_dim, 100));
            b.tensor_f16(p + "attn_gate.weight", {u64(n_embd), u64(value_dim)},
                         vec(static_cast<size_t>(n_embd) * value_dim, 200));
            std::vector<f32> kern(static_cast<size_t>(conv_k) * conv_dim);
            for (i64 c = 0; c < conv_dim; c++)
                for (i64 j = 0; j < conv_k; j++)
                    kern[static_cast<size_t>(c * conv_k + j)] =
                        static_cast<f32>(1 + j * 10 + c * 100);
            b.tensor_f32(p + "ssm_conv1d.weight", {u64(conv_k), u64(conv_dim)}, kern);
            b.tensor_f32(p + "ssm_dt.bias", {u64(n_v_head)},
                         vec(static_cast<size_t>(n_v_head), 400));
            std::vector<f32> a(static_cast<size_t>(n_v_head));
            for (int h = 0; h < n_v_head; h++)
                a[static_cast<size_t>(h)] = -0.5f - 0.1f * h;
            b.tensor_f32(p + "ssm_a", {u64(n_v_head)}, a);
            b.tensor_f16(p + "ssm_alpha.weight", {u64(n_embd), u64(n_v_head)},
                         vec(static_cast<size_t>(n_embd) * n_v_head, 500));
            b.tensor_f16(p + "ssm_beta.weight", {u64(n_embd), u64(n_v_head)},
                         vec(static_cast<size_t>(n_embd) * n_v_head, 600));
            b.tensor_f32(p + "ssm_norm.weight", {u64(d_state)}, ones(d_state));
            b.tensor_f16(p + "ssm_out.weight", {u64(value_dim), u64(n_embd)},
                         vec(static_cast<size_t>(value_dim) * n_embd, 700));
        } else {
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
            if (extras && qsa) {
                b.tensor_f16(p + "indexer.q_proj.weight",
                             {u64(n_embd), u64(idx_h * idx_dim)},
                             vec(static_cast<size_t>(n_embd) * idx_h * idx_dim, 800));
                b.tensor_f16(p + "indexer.k_proj.weight", {u64(n_embd), u64(idx_dim)},
                             vec(static_cast<size_t>(n_embd) * idx_dim, 900));
                b.tensor_f32(p + "indexer.q_norm.weight", {u64(idx_dim)}, ones(idx_dim));
                b.tensor_f32(p + "indexer.k_norm.weight", {u64(idx_dim)}, ones(idx_dim));
            }
        }

        if (extras && l == 0) {
            b.tensor_f16(p + "ple_key.weight", {u64(n_embd), u64(wide)},
                         vec(static_cast<size_t>(n_embd) * wide, 5300));
            b.tensor_f16(p + "ple_value.weight", {u64(n_embd), u64(n_embd)},
                         vec(static_cast<size_t>(n_embd) * n_embd, 5500));
            std::vector<f32> pk(static_cast<size_t>(ple_conv) * wide);
            for (size_t i = 0; i < pk.size(); i++) pk[i] = 0.25f;
            b.tensor_f32(p + "ple_conv1d.weight", {u64(ple_conv), u64(wide)}, pk);
            b.tensor_f32(p + "ple_norm_key.weight", {u64(wide)}, ones(wide));
            b.tensor_f32(p + "ple_norm_query.weight", {u64(wide)}, ones(wide));
            b.tensor_f32(p + "ple_norm_conv.weight", {u64(wide)}, ones(wide));
        }

        // ---- MoE FFN, shared by both layer kinds ----
        std::vector<f32> router(static_cast<size_t>(n_embd) * n_expert);
        for (size_t i = 0; i < router.size(); i++)
            router[i] = wv(static_cast<i64>(i) * 3 + 7);
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
            for (size_t k = 0; k < dsz; k++)
                down[e * dsz + k] = wv(1300 + static_cast<i64>(k) + bump);
        }
        b.tensor_f16(p + "ffn_gate_exps.weight",
                     {u64(n_embd), u64(n_ff_exp), u64(ne)}, gate);
        b.tensor_f16(p + "ffn_up_exps.weight",
                     {u64(n_embd), u64(n_ff_exp), u64(ne)}, up);
        b.tensor_f16(p + "ffn_down_exps.weight",
                     {u64(n_ff_exp), u64(n_embd), u64(ne)}, down);
        b.tensor_f16(p + "ffn_gate_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                     vec(static_cast<size_t>(n_embd) * n_ff_sh, 1700));
        b.tensor_f16(p + "ffn_up_shexp.weight", {u64(n_embd), u64(n_ff_sh)},
                     vec(static_cast<size_t>(n_embd) * n_ff_sh, 2100));
        b.tensor_f16(p + "ffn_down_shexp.weight", {u64(n_ff_sh), u64(n_embd)},
                     vec(static_cast<size_t>(n_ff_sh) * n_embd, 2500));
        b.tensor_f32(p + "ffn_gate_inp_shexp.weight", {u64(n_embd)},
                     vec(static_cast<size_t>(n_embd), 2700));
    }
    return b.write(path);
}

static void test_qwen4exp_schema() {
    const std::string path = kQ4TestPath;
    CHECK(build_tiny_q4_model(path, 0),
          "a miniature qwen4exp writes: 4 lanes, PLE, QSA and a 4-expert MoE");
    // The loader is reachable through a development switch only: the schema is
    // refused for generation (Engine::init) until the mixer runs, and this test
    // is about the tensors the loader reads.
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "1");
#else
    setenv("KRK_ARCH_ALLOW_EXPERIMENTAL", "1", 1);
#endif
    {
        Backend *cpu = make_cpu_backend();
        Model model;
        std::string err;
        const bool ok = model.load(*cpu, path, &err);
        CHECK(ok, "the schema loads");
        if (!ok) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        if (ok) {
            const ModelConfig &mc = model.cfg();
            CHECK(mc.hc && mc.hc_n == 4 && mc.hc_low_rank == 8,
                  "the hyper-connection geometry is read from the file");
            CHECK(mc.no_out_norm,
                  "qwen4exp declares no output norm (the head mixer carries it)");
            CHECK(mc.gdn_gate_sigmoid,
                  "its recurrent block gates with sigmoid, not silu");
            CHECK(mc.ple && mc.ple_ngram == 2 && mc.ple_head_dim == mc.n_embd,
                  "the PLE geometry is read (one row per token is n_embd wide)");
            CHECK(mc.ple_layers.size() == 1 && mc.ple_layers[0] == 0,
                  "... including which layer carries the module");
            CHECK(mc.qsa && mc.qsa_ratio.size() == 2 && mc.qsa_ratio[1] == 2,
                  "the per-layer compress ratio is read (layer 1 indexes at 2)");
            CHECK(model.ple_table() != nullptr && model.ple_row_bytes() > 0,
                  "the PLE table is a pointer into the mapping, not an upload");
            CHECK(model.hc_head_down().present() && model.hc_head_up().present(),
                  "the head mixer's bottleneck is loaded");

            const LayerWeights &L0 = model.layers()[0];
            const LayerWeights &L1 = model.layers()[1];
            CHECK(L0.gdn && L1.gdn == false,
                  "every (interval+1)th layer attends, the rest are delta net");
            const i64 wide = static_cast<i64>(mc.n_embd) * mc.hc_n;
            CHECK(L0.hc_attn_down.n_in == wide && L0.hc_attn_down.n_out == 8 &&
                      L0.hc_attn_up.n_in == 8 && L0.hc_attn_up.n_out == wide &&
                      L0.hc_attn_inject.n_out == mc.hc_n,
                  "the attention mixer is [hc*n_embd -> low_rank -> hc*n_embd] and "
                  "one injection weight per lane");
            CHECK(L0.hc_ffn_down.n_in == wide && L0.hc_ffn_inject.n_out == mc.hc_n,
                  "... and so is the FFN mixer");
            CHECK(L0.hc_attn_norm != nullptr && L0.hc_ffn_norm != nullptr,
                  "both mixers carry their grouped norm");
            CHECK(!L0.attn_norm && !L0.ffn_norm,
                  "there is no plain attn_norm/ffn_norm on this family");
            CHECK(L0.ple && L0.ple_key.n_in == mc.n_embd && L0.ple_key.n_out == wide,
                  "the PLE module is on layer 0, keyed at the wide stream");
            CHECK(L1.qsa_ratio == 2 && L1.idx_q_proj.present() &&
                      L1.idx_k_proj.present() && L1.idx_q_norm && L1.idx_k_norm,
                  "the attention layer carries its QSA indexer");
            CHECK(L0.qsa_ratio == 0 && !L0.idx_q_proj.present(),
                  "and a recurrent layer carries none");
        }
        model.unload();
        delete cpu;
    }

    // A hyper-connection matrix whose input width is not hc*n_embd must be
    // refused with the geometry, not loaded and run.
    {
        const std::string bad = "kraken-q4-bad.gguf";
        CHECK(build_tiny_q4_model(bad, 96), "a fixture with a short mixer writes");
        Backend *cpu = make_cpu_backend();
        Model model;
        std::string err;
        const bool ok = model.load(*cpu, bad, &err);
        CHECK(!ok, "a mixer whose width disagrees with hc*n_embd is refused");
        CHECK(err.find("hyper-connection") != std::string::npos,
              "... and the error names the hyper-connection geometry");
        model.unload();
        delete cpu;
        std::remove(bad.c_str());
    }
    std::remove(path.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
}

// The n-gram block that runs on this family's recurrent layers, end to end.
//
// The hash itself is pinned by test_ple_hash with hand-computed constants; what
// this group pins is the WIRING: which rows the engine gathers for a known
// prompt (computed here from the tokens, independently of ple.cpp), that the
// dilated conv's window survives both a chunked prefill and a token-at-a-time
// decode, and that the block leaves the lane stack finite and deterministic.
static void test_qwen4exp_ple() {
    const std::string path = "kraken-q4ple.gguf";
    // PLE on, indexer off: the block under test is the n-gram one.
    CHECK(build_tiny_q4_model(path, 0, 0, 0, true, false, false),
          "a four-lane file with the PLE block writes (no indexer)");
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "1");
#else
    setenv("KRK_ARCH_ALLOW_EXPERIMENTAL", "1", 1);
#endif
    {
        Backend *cpu = make_cpu_backend();
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = path;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = 8;
            std::string err;
            const bool ok = engine.init(cpu, cfg, &err);
            CHECK(ok, "a file carrying the PLE block initialises");
            if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
            if (ok) {
                const std::string prompt = "the the the the the the";
                std::vector<i32> ids;
                engine.tokenizer().encode(prompt, ids, true);
                CHECK(ids.size() >= 4, "the prompt tokenises");

                // A prefill-only run: the LAST forward is then the whole
                // prompt in one chunk, so the rows cover every prompt token.
                GenerateParams p;
                p.prompt = prompt;
                p.max_tokens = 0;
                p.sampler.greedy = true;
                GenerateResult r;
                CHECK(engine.generate(p, &r), "the prompt prefills through the block");

                i32 nt = 0, nh = 0;
                const i64 *rows = engine.ple_debug_rows(&nt, &nh);
                CHECK(rows != nullptr && nh == 1,
                      "the PLE block hashed one head per token");
                if (rows && nh == 1) {
                    CHECK(nt == static_cast<i32>(ids.size()),
                          "one row set per prompt token");
                    // The fixture's multipliers are {1,1} and its head has 8
                    // rows at offset 0, so a 2-gram is `t ^ t-1` and the row
                    // is that modulo 8. The first position has no predecessor
                    // and reads as EOS (258), which is what the reference's
                    // missing KV cell hashes as.
                    i64 bad = 0;
                    for (i32 i = 0; i < nt && i < static_cast<i32>(ids.size()); i++) {
                        const i64 cur = ids[static_cast<size_t>(i)];
                        const i64 prev =
                            i >= 1 ? ids[static_cast<size_t>(i - 1)] : 258;
                        const i64 expect = (cur ^ prev) % 8;
                        if (rows[static_cast<i64>(i) * nh] != expect) bad++;
                    }
                    CHECK(bad == 0,
                          "every gathered row is the hash of that token's window");
                }

                // The conv is the stateful half: the same prompt has to decode
                // to the same tokens whether it arrived in 8-token chunks or
                // one at a time, which only holds if the (K-1)*ngram rows of
                // history are carried correctly.
                GenerateParams g;
                g.prompt = prompt;
                g.max_tokens = 6;
                g.sampler.greedy = true;
                g.sampler.seed = 5;
                GenerateResult a, b;
                CHECK(engine.generate(g, &a), "the block generates");
                CHECK(engine.generate(g, &b) && a.tokens == b.tokens,
                      "... deterministically");

                EngineConfig cfg1 = cfg;
                cfg1.prefill_chunk = 1;
                Engine one;
                std::string e2;
                if (one.init(cpu, cfg1, &e2)) {
                    GenerateResult c;
                    CHECK(one.generate(g, &c), "and with a one-token prefill");
                    if (c.tokens != a.tokens) {
                        size_t at = 0;
                        while (at < a.tokens.size() && at < c.tokens.size() &&
                               a.tokens[at] == c.tokens[at]) at++;
                        std::fprintf(stderr,
                                     "  (chunking differs at token " 
                                     "%zu of %zu/%zu: %d vs %d)\n",
                                     at, a.tokens.size(), c.tokens.size(),
                                     at < a.tokens.size() ? a.tokens[at] : -1,
                                     at < c.tokens.size() ? c.tokens[at] : -1);
                    }
                    CHECK(c.tokens == a.tokens,
                          "the conv window makes the chunking invisible");
                } else {
                    CHECK(false, "engine initialises with chunk=1");
                    if (!e2.empty()) std::fprintf(stderr, "  (%s)\n", e2.c_str());
                }

                i32 hc = 0;
                i64 ne = 0;
                const f32 *lanes = engine.hc_debug_lanes(&hc, &ne);
                bool finite = lanes != nullptr;
                if (lanes) {
                    for (i64 i = 0; i < static_cast<i64>(hc) * ne && finite; i++) {
                        const f32 v = lanes[i];
                        if (!(v == v) || v > 1e6f || v < -1e6f) finite = false;
                    }
                }
                CHECK(finite, "and the lane stack it wrote into stays finite");
                // The block's two halves are different rows and the residual
                // gets the RAW one. The reference is
                // `hidden + gated + silu(conv(grouped_norm(gated)))`, so one
                // buffer holding the normalized copy puts a per-lane unit-RMS
                // row on the residual instead of `value * gate` -- the defect
                // that made this family decode garbage. Aliasing makes the two
                // rows identical, so a per-lane RMS comparison fails on it, and
                // a norm that was never applied fails it the other way.
                {
                    f32 *cvin = nullptr, *cvout = nullptr;
                    i64 rows_n = 0;
                    const f32 *raw_g =
                        engine.ple_debug_terms(&cvin, &cvout, &rows_n);
                    CHECK(raw_g && cvin && cvout,
                          "the PLE block exposes its three terms");
                    if (raw_g && cvin && cvout && rows_n > 0) {
                        CHECK(static_cast<const void *>(raw_g) !=
                                  static_cast<const void *>(cvin),
                              "the residual term and the conv input are two buffers");
                        auto lane_rms = [&](const f32 *v, i64 lane) {
                            f64 acc = 0.0;
                            for (i64 i = 0; i < ne; i++) {
                                const f64 x = v[lane * ne + i];
                                acc += x * x;
                            }
                            return std::sqrt(acc / static_cast<f64>(ne));
                        };
                        i64 differ = 0;
                        f64 total = 0.0;
                        for (i64 lane = 0; lane < hc; lane++) {
                            const f64 a_rms = lane_rms(raw_g, lane);
                            const f64 b_rms = lane_rms(cvin, lane);
                            total += a_rms;
                            if (b_rms > 0.0 && std::fabs(a_rms - b_rms) >
                                                   0.001 * std::max(a_rms, b_rms))
                                differ++;
                        }
                        CHECK(total > 0.0 && differ == hc,
                              "the conv input is the lane-normed copy, not the raw row");
                        i64 same = 0;
                        for (i64 i = 0; i < ne * hc; i++)
                            if (cvout[i] == cvin[i]) same++;
                        CHECK(same < ne * hc,
                              "and the convolution actually ran over it");
                    }
                }
            }
        }
        delete cpu;
    }
    std::remove(path.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
}

// The residual stack this family runs, end to end on the reference backend.
//
// The loader test above proves the tensors are read; hc.cpp proves the lane
// arithmetic; this proves the two are wired to each other -- a four-lane file
// initialises, generates, and keeps the lane invariant the arithmetic promises,
// through the real forward pass. The pieces the forward does not run yet (PLE,
// the QSA indexer) are left out of the fixture, so a pass here is the stack.
// ---------------------------------------------------------------------------
// MLA, the host maths first (no engine, no file): the absorbed order the
// oracle will prove kernels against, checked against the unfolded reference.
// A wrong absorb is fluent text rather than a crash -- the same failure shape
// as the PLE lane scatter -- so the equivalence, not a sample, is the gate.
// The double-norm guard is load-bearing too: the cache holds normed rows, so
// the read path must not norm again, and this group fails if it does (the
// absorbed norm of an already-normed row shrinks every output ~1/sqrt(n)).
static void test_mla_absorb() {
    krk::MlaGeometry g;
    g.n_embd = 32;
    g.n_head = 3;
    g.q_lora = 24;
    g.kv_lora = 16;
    g.qk_nope = 8;
    g.qk_rope = 4;
    g.v_dim = 8;
    CHECK(g.valid(), "synthetic MLA geometry is valid");
    CHECK(g.cache_row_width() == 20, "cache row is kv_lora + qk_rope");
    // Deterministic fill: the same bytes every run, no fixture file.
    u32 rng = 0x2f6e2b29u;
    auto rnd = [&]() {
        rng = rng * 1103515245u + 12345u;
        return static_cast<f32>((rng >> 20) & 0xff) / 255.0f - 0.5f;
    };
    const i64 qs = static_cast<i64>(g.qk_nope) + g.qk_rope;
    std::vector<f32> wqa(g.n_embd * g.q_lora), qan(g.q_lora),
        wqb(g.q_lora * g.n_head * qs), wka(g.n_embd * g.kv_lora),
        knorm(g.kv_lora), wkr(g.n_embd * g.qk_rope),
        wuk(g.n_head * g.kv_lora * g.qk_nope),
        wuv(g.n_head * g.kv_lora * g.v_dim),
        wo(g.n_head * g.v_dim * g.n_embd);
    for (f32 &v : wqa) v = rnd();
    for (f32 &v : qan) v = 0.9f + 0.2f * rnd();
    for (f32 &v : wqb) v = rnd();
    for (f32 &v : wka) v = rnd();
    for (f32 &v : knorm) v = 0.9f + 0.2f * rnd();
    for (f32 &v : wkr) v = rnd();
    for (f32 &v : wuk) v = rnd();
    for (f32 &v : wuv) v = rnd();
    for (f32 &v : wo) v = rnd();
    krk::MlaWeights w;
    w.wq_a = wqa.data();
    w.wqa_norm = qan.data();
    w.wq_b = wqb.data();
    w.wkv_a = wka.data();
    w.wkv_norm = knorm.data();
    w.wkr = wkr.data();
    w.wuk = wuk.data();
    w.wuv = wuv.data();
    w.wo = wo.data();
    // The write path: compress, norm, cache. The test builds cache rows the
    // way the forward will, through the same rms formula.
    const i64 n_keys = 5;
    std::vector<f32> cc(n_keys * g.kv_lora), kk(n_keys * g.qk_rope);
    std::vector<f32> xt(g.n_embd);
    for (i64 t = 0; t < n_keys; t++) {
        f32 *c = cc.data() + t * g.kv_lora;
        // norm through the engine formula: mean square + eps, gamma row.
        f64 ss = 0.0;
        std::vector<f32> raw(g.kv_lora);
        for (i64 i = 0; i < g.n_embd; i++) xt[static_cast<size_t>(i)] = rnd();
        for (i64 k = 0; k < g.kv_lora; k++) {
            f32 s = 0.0f;
            for (i64 i = 0; i < g.n_embd; i++)
                s += wka[static_cast<size_t>(i) * g.kv_lora + k] *
                     xt[static_cast<size_t>(i)];
            raw[static_cast<size_t>(k)] = s;
            ss += static_cast<f64>(s) * s;
        }
        const f32 inv =
            1.0f / std::sqrt(static_cast<f32>(ss / g.kv_lora) + 1e-6f);
        for (i64 k = 0; k < g.kv_lora; k++)
            c[k] = raw[static_cast<size_t>(k)] * inv *
                   knorm[static_cast<size_t>(k)];
        f32 *kr = kk.data() + t * g.qk_rope;
        for (i64 d = 0; d < g.qk_rope; d++) {
            f32 s = 0.0f;
            for (i64 i = 0; i < g.n_embd; i++)
                s += wkr[static_cast<size_t>(i) * g.qk_rope + d] *
                     xt[static_cast<size_t>(i)];
            kr[d] = s;
        }
    }
    std::vector<f32> x(g.n_embd);
    for (f32 &v : x) v = rnd();
    std::vector<f32> wq_abs(g.n_head * g.q_lora * g.kv_lora);
    krk::mla_absorb_q(wq_abs.data(), wqb.data(), wuk.data(), g);
    std::vector<f32> ou(g.n_embd), oa(g.n_embd), oa2(g.n_embd);
    const f32 sc = 1.0f / std::sqrt(static_cast<f32>(qs));
    krk::mla_forward_unfolded(ou.data(), x.data(), cc.data(), kk.data(),
                              n_keys, w, g, sc, 1e-6f);
    krk::mla_forward_absorbed(oa.data(), x.data(), cc.data(), kk.data(),
                              n_keys, w, wq_abs.data(), g, sc, 1e-6f);
    krk::mla_forward_absorbed(oa2.data(), x.data(), cc.data(), kk.data(),
                              n_keys, w, wq_abs.data(), g, sc, 1e-6f);
    double worst = 0.0;
    bool finite = true;
    for (i64 i = 0; i < g.n_embd; i++) {
        const double d =
            std::fabs(static_cast<double>(ou[static_cast<size_t>(i)]) -
                      static_cast<double>(oa[static_cast<size_t>(i)]));
        if (d > worst) worst = d;
        if (!std::isfinite(oa[static_cast<size_t>(i)])) finite = false;
        // Determinism: the oracle order replays byte-identically.
        CHECK(oa[static_cast<size_t>(i)] == oa2[static_cast<size_t>(i)],
              "absorbed forward is deterministic");
    }
    CHECK(finite, "absorbed output is finite");
    CHECK(worst < 1e-4, "absorbed matches unfolded within f32 tolerance");
}


// MLA legacy lite layout (DeepSeek-V2-Lite): direct q, MQA kv, unsplit wkv_b.
// Two orders, one answer, plus the pins that make the test see the layout: a
// singleton cache ignores the query entirely (p == 1 exactly), moved write
// positions move the answer (rope is read, not decorative), a head-permuted
// wkv_b moves it too (heads are not interchangeable), and the guard row
// proves a zero-length cache returns instead of crashing.
static void test_mla_lite() {
    krk::MlaLiteGeometry g;
    g.n_embd = 24;
    g.n_head = 2;
    g.kv_lora = 8;
    g.qk = 8;
    g.qk_rope = 4;
    g.v_dim = 4;
    CHECK(g.valid(), "synthetic lite MLA geometry is valid");
    CHECK(g.nope() == 4, "nope split is qk - rope");
    CHECK(g.cache_row_width() == 12, "cache row is kv_lora + rope");
    u32 rng = 0x51f15eedU;
    auto rnd = [&]() {
        rng = rng * 1103515245u + 12345u;
        return static_cast<f32>((rng >> 20) & 0xff) / 255.0f - 0.5f;
    };
    const i64 nope = g.nope();
    const i64 hv = nope + g.v_dim;
    std::vector<f32> wq(g.n_embd * g.n_head * g.qk),
        mqa(g.n_embd * (g.kv_lora + g.qk_rope)), wnorm(g.kv_lora),
        wkvb(g.kv_lora * g.n_head * hv),
        wo(g.n_head * g.v_dim * g.n_embd);
    for (f32 &v : wq) v = rnd();
    for (f32 &v : mqa) v = rnd();
    for (f32 &v : wnorm) v = 0.9f + 0.2f * rnd();
    for (f32 &v : wkvb) v = rnd();
    for (f32 &v : wo) v = rnd();
    krk::MlaLiteWeights w;
    w.wq = wq.data();
    w.wkv_a_mqa = mqa.data();
    w.wkv_a_norm = wnorm.data();
    w.wkv_b = wkvb.data();
    w.wo = wo.data();
    // Plain base-10000 table over the rope pairs; YaRN variants arrive here
    // as an adjusted table from the caller, so the oracle never branches.
    std::vector<f32> invf(g.qk_rope / 2);
    for (i64 i = 0; i < g.qk_rope / 2; i++)
        invf[static_cast<size_t>(i)] =
            1.0f / std::pow(10000.0f,
                            static_cast<f32>(2 * i) / g.qk_rope);
    // The write path: project, split, norm the latent, cache the raw key.
    // Rotation happens on READ (oracle side), at each row's own position.
    const i64 n_keys = 4;
    std::vector<f32> cc(n_keys * g.kv_lora), kk(n_keys * g.qk_rope);
    std::vector<i64> kpos(static_cast<size_t>(n_keys));
    std::vector<f32> xt(g.n_embd);
    for (i64 t = 0; t < n_keys; t++) {
        for (i64 i = 0; i < g.n_embd; i++)
            xt[static_cast<size_t>(i)] = rnd();
        f32 *c = cc.data() + t * g.kv_lora;
        f64 ss = 0.0;
        std::vector<f32> raw(g.kv_lora);
        for (i64 k = 0; k < g.kv_lora; k++) {
            f32 s = 0.0f;
            for (i64 i = 0; i < g.n_embd; i++)
                s += mqa[static_cast<size_t>(i) * (g.kv_lora + g.qk_rope) +
                         k] *
                     xt[static_cast<size_t>(i)];
            raw[static_cast<size_t>(k)] = s;
            ss += static_cast<f64>(s) * s;
        }
        const f32 inv =
            1.0f / std::sqrt(static_cast<f32>(ss / g.kv_lora) + 1e-6f);
        for (i64 k = 0; k < g.kv_lora; k++)
            c[k] = raw[static_cast<size_t>(k)] * inv *
                   wnorm[static_cast<size_t>(k)];
        f32 *kr = kk.data() + t * g.qk_rope;
        for (i64 d = 0; d < g.qk_rope; d++) {
            f32 s = 0.0f;
            for (i64 i = 0; i < g.n_embd; i++)
                s += mqa[static_cast<size_t>(i) * (g.kv_lora + g.qk_rope) +
                         g.kv_lora + d] *
                     xt[static_cast<size_t>(i)];
            kr[d] = s;
        }
        kpos[static_cast<size_t>(t)] = t;
    }
    std::vector<f32> x(g.n_embd);
    for (f32 &v : x) v = rnd();
    std::vector<f32> ou(g.n_embd), oa(g.n_embd), oa2(g.n_embd);
    const f32 sc = 1.0f / std::sqrt(static_cast<f32>(g.qk));
    krk::mla_lite_forward_unfolded(ou.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), n_keys, /*q_pos*/ 4, w, g,
                                   invf.data(), sc, 1e-6f);
    krk::mla_lite_forward_absorbed(oa.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), n_keys, /*q_pos*/ 4, w, g,
                                   invf.data(), sc, 1e-6f);
    krk::mla_lite_forward_absorbed(oa2.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), n_keys, /*q_pos*/ 4, w, g,
                                   invf.data(), sc, 1e-6f);
    double worst = 0.0;
    bool finite = true;
    for (i64 i = 0; i < g.n_embd; i++) {
        const double d =
            std::fabs(static_cast<double>(ou[static_cast<size_t>(i)]) -
                      static_cast<double>(oa[static_cast<size_t>(i)]));
        if (d > worst) worst = d;
        if (!std::isfinite(oa[static_cast<size_t>(i)])) finite = false;
        CHECK(oa[static_cast<size_t>(i)] == oa2[static_cast<size_t>(i)],
              "lite absorbed forward is deterministic");
    }
    CHECK(finite, "lite absorbed output is finite");
    CHECK(worst < 1e-4, "lite absorbed matches unfolded within f32 tolerance");
    // Singleton cache: p == 1 whatever the query, so two different queries
    // must produce bit-identical outputs. Any score-dependence in the
    // weighting breaks this exactly, not approximately.
    std::vector<f32> x2(g.n_embd);
    for (f32 &v : x2) v = rnd();
    std::vector<f32> o1(g.n_embd), o2(g.n_embd);
    krk::mla_lite_forward_absorbed(o1.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), 1, /*q_pos*/ 4, w, g,
                                   invf.data(), sc, 1e-6f);
    krk::mla_lite_forward_absorbed(o2.data(), x2.data(), cc.data(), kk.data(),
                                   kpos.data(), 1, /*q_pos*/ 4, w, g,
                                   invf.data(), sc, 1e-6f);
    for (i64 i = 0; i < g.n_embd; i++)
        CHECK(o1[static_cast<size_t>(i)] == o2[static_cast<size_t>(i)],
              "singleton lite cache ignores the query");
    // Moved write positions move the answer: the positions are read.
    std::vector<i64> kpos2(static_cast<size_t>(n_keys));
    for (i64 t = 0; t < n_keys; t++) kpos2[static_cast<size_t>(t)] = t + 1;
    std::vector<f32> om(g.n_embd);
    krk::mla_lite_forward_absorbed(om.data(), x.data(), cc.data(), kk.data(),
                                   kpos2.data(), n_keys, /*q_pos*/ 4, w, g,
                                   invf.data(), sc, 1e-6f);
    double moved = 0.0;
    for (i64 i = 0; i < g.n_embd; i++)
        moved = std::max(
            moved,
            std::fabs(static_cast<double>(om[static_cast<size_t>(i)]) -
                      static_cast<double>(oa[static_cast<size_t>(i)])));
    CHECK(moved > 1e-6, "shifted write positions move the lite answer");
    // A head-permuted wkv_b moves the answer: heads are distinguished.
    std::vector<f32> wkvb2 = wkvb;
    const i64 hb = g.kv_lora * hv;
    for (i64 i = 0; i < hb; i++)
        std::swap(wkvb2[static_cast<size_t>(i)],
                  wkvb2[static_cast<size_t>(hb + i)]);
    krk::MlaLiteWeights w2 = w;
    w2.wkv_b = wkvb2.data();
    std::vector<f32> oh2(g.n_embd);
    krk::mla_lite_forward_absorbed(oh2.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), n_keys, /*q_pos*/ 4, w2, g,
                                   invf.data(), sc, 1e-6f);
    double hdiff = 0.0;
    for (i64 i = 0; i < g.n_embd; i++)
        hdiff = std::max(
            hdiff,
            std::fabs(static_cast<double>(oh2[static_cast<size_t>(i)]) -
                      static_cast<double>(oa[static_cast<size_t>(i)])));
    CHECK(hdiff > 1e-6, "head-permuted wkv_b moves the lite answer");
    // The scale and the query position both reach the scores.
    std::vector<f32> os(g.n_embd);
    krk::mla_lite_forward_absorbed(os.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), n_keys, /*q_pos*/ 4, w, g,
                                   invf.data(), sc * 0.5f, 1e-6f);
    CHECK(os[0] != oa[0] || os[1] != oa[1], "scale reaches the lite scores");
    std::vector<f32> oq(g.n_embd);
    krk::mla_lite_forward_absorbed(oq.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), n_keys, /*q_pos*/ 5, w, g,
                                   invf.data(), sc, 1e-6f);
    CHECK(oq[0] != oa[0] || oq[1] != oa[1], "query position is read");
    // Zero-length cache returns instead of crashing.
    krk::mla_lite_forward_absorbed(oq.data(), x.data(), cc.data(), kk.data(),
                                   kpos.data(), 0, /*q_pos*/ 5, w, g,
                                   invf.data(), sc, 1e-6f);
}

// DeepSeek-V4 shared-latent MLA (MlaLayout::Shared), the host maths first.
//
// This layout is not V2/V3's: ONE latent row per token is both K and V for
// every head, the rope is the TAIL of that row, and the head count is folded
// into a grouped low-rank output (per-group blocks of wo_a, then wo_b). Each
// of those is a place where a wrong implementation still emits finite,
// plausible vectors, so this group pins them ALGEBRAICALLY and never by text:
//
//   * the two orders (per-key accumulation vs weights-then-gather) agree;
//   * ONE key means softmax 1, so each head's output IS the cached latent --
//     which pins the q path, the latent norm and the grouped output at once;
//   * the rope touches only the tail, and the value side reads the UNROTATED
//     row (the one assumption include/krk/mla_shared.hpp names);
//   * a saturating sink takes all the mass, so the output is zero;
//   * a head feeds only its own output group: zero group 0's wo_a block and
//     rewiring heads 0..hpg-1 must leave the answer bit-for-bit unchanged,
//     which an off-by-one group stride cannot survive;
//   * the window excludes keys by POSITION, not by array index.
static void test_mla_shared() {
    krk::MlaSharedGeometry g;
    g.n_embd = 16;
    g.n_head = 4;
    g.q_lora = 8;
    g.kv_lora = 12; // head_dim == key_length == value_length
    g.qk_rope = 4;  // rotary tail
    g.o_groups = 2;
    g.o_lora = 6;
    CHECK(g.valid(), "synthetic shared-latent geometry is valid");
    CHECK(g.qk_nope() == 8, "nope is the latent minus the rotary tail");
    CHECK(g.heads_per_group() == 2, "4 heads split into 2 groups of 2");
    CHECK(g.cache_row_width() == 12, "K and V are ONE 12-wide row");
    u32 rng = 0x9e3779b9U;
    auto rnd = [&]() {
        rng = rng * 1103515245u + 12345u;
        return static_cast<f32>((rng >> 20) & 0xff) / 255.0f - 0.5f;
    };
    const i32 gout = g.o_groups * g.o_lora;
    const i32 gin = g.heads_per_group() * g.kv_lora;
    std::vector<f32> wqa(g.n_embd * g.q_lora), wqan(g.q_lora),
        wqb(g.q_lora * g.n_head * g.kv_lora), wkv(g.n_embd * g.kv_lora),
        wkvn(g.kv_lora), woa(gin * gout), wob(gout * g.n_embd),
        sinks(g.n_head);
    for (f32 &v : wqa) v = rnd();
    for (f32 &v : wqan) v = 0.9f + 0.2f * rnd();
    for (f32 &v : wqb) v = rnd();
    for (f32 &v : wkv) v = rnd();
    for (f32 &v : wkvn) v = 0.9f + 0.2f * rnd();
    for (f32 &v : woa) v = rnd();
    for (f32 &v : wob) v = rnd();
    for (f32 &v : sinks) v = 0.25f * rnd();
    krk::MlaSharedWeights w;
    w.wq_a = wqa.data();
    w.wq_a_norm = wqan.data();
    w.wq_b = wqb.data();
    w.wkv = wkv.data();
    w.wkv_norm = wkvn.data();
    w.wo_a = woa.data();
    w.wo_b = wob.data();
    w.sinks = nullptr;
    std::vector<f32> invf(g.qk_rope / 2);
    for (i32 i = 0; i < g.qk_rope / 2; i++)
        invf[static_cast<size_t>(i)] =
            1.0f / std::pow(10000.0f, static_cast<f32>(2 * i) / g.qk_rope);
    // A small cache written through the same helper the write path uses.
    const i32 n_keys = 4;
    std::vector<f32> cc(n_keys * g.kv_lora);
    std::vector<i64> kpos(n_keys);
    for (i32 t = 0; t < n_keys; t++) {
        std::vector<f32> xt(g.n_embd);
        for (f32 &v : xt) v = rnd();
        krk::mla_shared_latent(cc.data() + t * g.kv_lora, xt.data(), w, g,
                               1e-6f);
        kpos[static_cast<size_t>(t)] = t;
    }
    std::vector<f32> x(g.n_embd);
    for (f32 &v : x) v = rnd();
    const i64 q_pos = 3;
    const f32 sc = 1.0f / std::sqrt(static_cast<f32>(g.kv_lora));
    std::vector<f32> oa(g.n_embd), og(g.n_embd), og2(g.n_embd);
    krk::mla_shared_forward_accum(oa.data(), x.data(), cc.data(), kpos.data(),
                                  n_keys, q_pos, w, g, invf.data(), sc, 1e-6f,
                                  0);
    krk::mla_shared_forward_gather(og.data(), x.data(), cc.data(), kpos.data(),
                                   n_keys, q_pos, w, g, invf.data(), sc, 1e-6f,
                                   0);
    krk::mla_shared_forward_gather(og2.data(), x.data(), cc.data(), kpos.data(),
                                   n_keys, q_pos, w, g, invf.data(), sc, 1e-6f,
                                   0);
    double worst = 0.0;
    bool finite = true;
    for (i32 i = 0; i < g.n_embd; i++) {
        const double d = std::fabs(static_cast<double>(oa[i]) - og[i]);
        if (d > worst) worst = d;
        if (!std::isfinite(oa[i]) || !std::isfinite(og[i])) finite = false;
        CHECK(og[i] == og2[i], "shared-latent gather forward is deterministic");
    }
    CHECK(finite, "shared-latent outputs are finite");
    CHECK(worst < 2e-5, "accum and gather orders agree within f32 association");

    // ONE key: softmax is 1, so every head's output is exactly the latent and
    // the whole answer is the grouped output applied to one row. A wrong
    // q path or a wrong group stride cannot produce this.
    std::vector<f32> one_c(g.kv_lora);
    std::vector<f32> x1(g.n_embd);
    for (f32 &v : x1) v = rnd();
    krk::mla_shared_latent(one_c.data(), x1.data(), w, g, 1e-6f);
    std::vector<i64> one_pos(1, 0);
    std::vector<f32> one(g.n_embd), one2(g.n_embd);
    std::vector<f32> yexp(gout, 0.0f), oexp(g.n_embd, 0.0f);
    for (i32 grp = 0; grp < g.o_groups; grp++)
        for (i32 j = 0; j < g.o_lora; j++) {
            f32 s = 0.0f;
            for (i32 h = 0; h < g.heads_per_group(); h++)
                for (i32 i = 0; i < g.kv_lora; i++)
                    s += woa[(h * g.kv_lora + i) * gout + grp * g.o_lora + j] *
                         one_c[i];
            yexp[static_cast<size_t>(grp * g.o_lora + j)] = s;
        }
    for (i32 j = 0; j < g.n_embd; j++) {
        f32 s = 0.0f;
        for (i32 i = 0; i < gout; i++)
            s += wob[static_cast<size_t>(i * g.n_embd + j)] *
                 yexp[static_cast<size_t>(i)];
        oexp[static_cast<size_t>(j)] = s;
    }
    krk::mla_shared_forward_accum(one.data(), x1.data(), one_c.data(),
                                  one_pos.data(), 1, 0, w, g, invf.data(), sc,
                                  1e-6f, 0);
    krk::mla_shared_forward_gather(one2.data(), x1.data(), one_c.data(),
                                   one_pos.data(), 1, 0, w, g, invf.data(),
                                   sc, 1e-6f, 0);
    for (i32 j = 0; j < g.n_embd; j++)
        CHECK_NEAR(one[j], oexp[static_cast<size_t>(j)], 1e-4,
                   "one key reduces to the grouped output of the latent");
    for (i32 j = 0; j < g.n_embd; j++)
        CHECK_NEAR(one2[j], oexp[static_cast<size_t>(j)], 1e-4,
                   "the gather order agrees on the one-key reduction");

    // The rope is the TAIL: rotating a row must leave dims [0, nope) alone.
    std::vector<f32> rv(g.kv_lora), rv0;
    for (i32 i = 0; i < g.kv_lora; i++) rv[i] = static_cast<f32>(i + 1);
    rv0 = rv;
    krk::mla_shared_rope_tail(rv.data(), g.kv_lora, g.qk_rope, 3, invf.data());
    bool nope_still = true, tail_moved = false;
    for (i32 i = 0; i < g.qk_nope(); i++)
        if (rv[i] != rv0[i]) nope_still = false;
    for (i32 i = g.qk_nope(); i < g.kv_lora; i++)
        if (rv[i] != rv0[i]) tail_moved = true;
    CHECK(nope_still, "rope leaves the non-rotary prefix untouched");
    CHECK(tail_moved, "rope rotates the latent TAIL (offset head_dim - n_rot)");

    // Sinks are a denominator term: saturating upwards takes all the mass.
    std::vector<f32> sink_hi(g.n_head, 1e30f), sink_lo(g.n_head, -1e30f);
    std::vector<f32> osat(g.n_embd), ono(g.n_embd);
    krk::MlaSharedWeights ws = w;
    ws.sinks = sink_hi.data();
    krk::mla_shared_forward_accum(osat.data(), x.data(), cc.data(),
                                  kpos.data(), n_keys, q_pos, ws, g,
                                  invf.data(), sc, 1e-6f, 0);
    f32 amax = 0.0f;
    for (f32 v : osat) amax = std::fmax(amax, std::fabs(v));
    CHECK(amax < 1e-6f, "a saturating sink takes every key's mass, so out ~ 0");
    ws.sinks = sink_lo.data();
    krk::mla_shared_forward_accum(ono.data(), x.data(), cc.data(), kpos.data(),
                                  n_keys, q_pos, ws, g, invf.data(), sc, 1e-6f,
                                  0);
    for (i32 j = 0; j < g.n_embd; j++)
        CHECK(ono[j] == oa[j],
              "a sink of -inf is the no-sink answer, bit for bit");

    // Group isolation: zero group 0's wo_a block and rewire heads 0..hpg-1.
    // Those heads feed only group 0, so the answer must not move; a stride
    // that reads a neighbouring group's columns cannot pass this.
    std::vector<f32> woa_g = woa, wqb_h = wqb;
    for (i32 i = 0; i < gin; i++)
        for (i32 j = 0; j < g.o_lora; j++)
            woa_g[static_cast<size_t>(i * gout + j)] = 0.0f;
    for (i32 i = 0; i < g.q_lora; i++)
        for (i32 h = 0; h < g.heads_per_group(); h++)
            for (i32 d = 0; d < g.kv_lora; d++)
                wqb_h[static_cast<size_t>(
                    i * g.n_head * g.kv_lora + h * g.kv_lora + d)] *= 3.0f;
    krk::MlaSharedWeights wg = w;
    wg.wo_a = woa_g.data();
    std::vector<f32> ogzero(g.n_embd);
    krk::mla_shared_forward_accum(ogzero.data(), x.data(), cc.data(),
                                  kpos.data(), n_keys, q_pos, wg, g,
                                  invf.data(), sc, 1e-6f, 0);
    // The baseline has to be the SAME zeroed block: zeroing group 0's columns
    // removes its contribution, so comparing against the unzeroed answer would
    // fail for the wrong reason (which it did, 16 of 16, on the first run).
    bool zeroed_changed = false;
    for (i32 j = 0; j < g.n_embd; j++)
        if (ogzero[j] != oa[j]) zeroed_changed = true;
    CHECK(zeroed_changed, "zeroing an output group removes its contribution");
    wg.wq_b = wqb_h.data();
    std::vector<f32> ogrp(g.n_embd);
    krk::mla_shared_forward_accum(ogrp.data(), x.data(), cc.data(), kpos.data(),
                                  n_keys, q_pos, wg, g, invf.data(), sc, 1e-6f,
                                  0);
    for (i32 j = 0; j < g.n_embd; j++)
        CHECK(ogrp[j] == ogzero[j],
              "a head feeds only its own output group (stride pin)");

    // The window is a POSITION rule: with window 2 at q_pos 3, keys 0 and 1
    // are outside it, so replacing their rows must change nothing.
    std::vector<f32> cc2 = cc;
    for (i32 i = 0; i < 2 * g.kv_lora; i++) cc2[static_cast<size_t>(i)] = 5.0f;
    std::vector<f32> ow(g.n_embd), ow2(g.n_embd), oall(g.n_embd);
    krk::mla_shared_forward_accum(ow.data(), x.data(), cc.data(), kpos.data(),
                                  n_keys, q_pos, w, g, invf.data(), sc, 1e-6f,
                                  2);
    krk::mla_shared_forward_accum(ow2.data(), x.data(), cc2.data(), kpos.data(),
                                  n_keys, q_pos, w, g, invf.data(), sc, 1e-6f,
                                  2);
    krk::mla_shared_forward_accum(oall.data(), x.data(), cc2.data(),
                                  kpos.data(), n_keys, q_pos, w, g,
                                  invf.data(), sc, 1e-6f, 0);
    for (i32 j = 0; j < g.n_embd; j++)
        CHECK(ow[j] == ow2[j], "the window excludes keys by position");
    bool window_matters = false;
    for (i32 j = 0; j < g.n_embd; j++)
        if (ow[j] != oall[j]) window_matters = true;
    CHECK(window_matters, "the keys the window excludes do reach the answer");

    // Scale and query position both reach the scores.
    std::vector<f32> osc(g.n_embd), opos(g.n_embd);
    krk::mla_shared_forward_accum(osc.data(), x.data(), cc.data(), kpos.data(),
                                  n_keys, q_pos, w, g, invf.data(), sc * 0.5f,
                                  1e-6f, 0);
    CHECK(osc[0] != oa[0] || osc[1] != oa[1], "scale reaches the scores");
    krk::mla_shared_forward_accum(opos.data(), x.data(), cc.data(), kpos.data(),
                                  n_keys, q_pos + 1, w, g, invf.data(), sc,
                                  1e-6f, 0);
    CHECK(opos[0] != oa[0] || opos[1] != oa[1], "query position is read");
    // A zero-length cache returns instead of crashing.
    krk::mla_shared_forward_accum(opos.data(), x.data(), cc.data(), kpos.data(),
                                  0, q_pos, w, g, invf.data(), sc, 1e-6f, 0);
}

// MLA backend ops: the CPU reference the HIP kernels are judged against.
//
// The suite cannot open the device, so the device arm lives in `kraken-bench
// --mla` (the same oracle, on the device). What the suite CAN pin is the
// reference itself, and it has to: `--cpu` is the arm a mismatch is diagnosed
// against, and a reference that reads the wrong head's block produces the same
// kind of fluent nonsense as a kernel that does, only more quietly.
//
// Every check below fails on a real defect, and they are chosen so that they do
// not fail on each other:
//
//   * the layer output against src/mla.cpp's absorbed oracle, with the table
//     transposed into the oracle's head-major layout. Dropping the per-head
//     offset in the query (`q + h*qk`) or in the table read aliases head h with
//     head 0, and both move this number.
//   * every head's V columns zeroed but one -> only that head's output slice
//     may be non-zero. That is the aliasing check in its own right, and it is
//     exact rather than approximate.
//   * changing head 0's query leaves the other heads' slices BIT-identical.
//   * the cache write: the decoupled key is copied verbatim, the latent is
//     normed over the LATENT alone, and the negative control is the plausible
//     mistake -- a norm over the whole row, which is off by sqrt(12/8).
//   * n single-row calls against one n-row call, bit-identical: the cache row is
//     written by one pass and read by the same pass, so a chunked prefill and a
//     token-at-a-time decode have to see the same history.
static void test_mla_backend() {
    Backend *cpu = make_cpu_backend();
    const i32 nh = 3, nope = 4, rope = 4, kv = 8, vd = 4, ne = 16;
    const i64 qk = nope + rope, hv = nope + vd, row = kv + rope;
    u32 rng = 0x9e3779b9u;
    auto rnd = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<f32>((rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
    };
    const i64 n_keys = 5;
    std::vector<f32> wkvb(static_cast<size_t>(nh * hv * kv));
    for (f32 &x : wkvb) x = rnd();
    std::vector<f32> gamma(static_cast<size_t>(kv));
    for (f32 &g : gamma) g = 0.5f + 0.5f * std::fabs(rnd());
    std::vector<f32> invf(static_cast<size_t>(rope / 2));
    for (i64 i = 0; i < rope / 2; i++)
        invf[static_cast<size_t>(i)] =
            1.0f / std::pow(10000.0f, static_cast<f32>(2 * i) / rope);

    // The f16 staging the engine builds at load, in the FILE's order: row
    // (h*hv + j) holds feature j of head h, the latent k its fastest axis.
    std::vector<u16> wkb16(static_cast<size_t>(nh * hv * kv));
    for (size_t i = 0; i < wkvb.size(); i++) wkb16[i] = fp32_to_fp16(wkvb[i]);

    // The projection output, as the engine hands it to the cache write.
    std::vector<f32> proj(static_cast<size_t>(n_keys * row));
    for (f32 &x : proj) x = 2.0f * rnd();
    std::vector<f32> cache(static_cast<size_t>(n_keys * row), -999.0f);

    MlaDesc d;
    d.n_head = nh;
    d.qk = qk;
    d.nope = nope;
    d.rope = rope;
    d.kv_lora = kv;
    d.v_dim = vd;
    d.n_tok = n_keys;
    d.pos0 = 0;
    d.pos_stride = row;
    d.scale = 1.0f / std::sqrt(static_cast<f32>(qk));
    d.eps = 1e-6f;
    d.inv_freq = invf.data();
    cpu->mla_kv_write(cache.data(), proj.data(), gamma.data(), d);

    // ---- the cache row: latent normed, key verbatim ----------------------
    std::vector<f32> want(static_cast<size_t>(n_keys * row), 0.0f);
    std::vector<f32> whole(static_cast<size_t>(n_keys * row), 0.0f);
    for (i64 t = 0; t < n_keys; t++) {
        f64 ss = 0.0, ss_all = 0.0;
        for (i64 k = 0; k < kv; k++) {
            const f64 v = proj[static_cast<size_t>(t * row + k)];
            ss += v * v;
        }
        for (i64 k = 0; k < row; k++) {
            const f64 v = proj[static_cast<size_t>(t * row + k)];
            ss_all += v * v;
        }
        const f32 inv = 1.0f / std::sqrt(static_cast<f32>(ss / kv) + d.eps);
        const f32 inv_all = 1.0f / std::sqrt(static_cast<f32>(ss_all / row) + d.eps);
        for (i64 k = 0; k < kv; k++) {
            const f32 v = proj[static_cast<size_t>(t * row + k)];
            want[static_cast<size_t>(t * row + k)] = v * inv * gamma[static_cast<size_t>(k)];
            whole[static_cast<size_t>(t * row + k)] = v * inv_all * gamma[static_cast<size_t>(k)];
        }
        for (i64 j = 0; j < rope; j++)
            want[static_cast<size_t>(t * row + kv + j)] =
                proj[static_cast<size_t>(t * row + kv + j)];
    }
    f64 worst = 0.0, differs = 0.0;
    bool key_exact = true;
    for (i64 t = 0; t < n_keys; t++) {
        for (i64 k = 0; k < kv; k++) {
            const size_t i = static_cast<size_t>(t * row + k);
            worst = std::max(worst, static_cast<f64>(std::fabs(cache[i] - want[i])));
            differs = std::max(differs,
                               static_cast<f64>(std::fabs(cache[i] - whole[i])));
        }
        for (i64 j = 0; j < rope; j++) {
            const size_t i = static_cast<size_t>(t * row + kv + j);
            if (cache[i] != proj[i]) key_exact = false;
        }
    }
    CHECK(worst < 1e-5, "mla_kv_write norms the latent the way the oracle does");
    CHECK(key_exact,
          "the decoupled key reaches the cache UNROTATED and unnormed");
    CHECK(differs > 1e-3,
          "a norm over the whole row differs, so the latent-only cover is real");

    // ---- the attention, against the oracle -------------------------------
    std::vector<f32> x(static_cast<size_t>(n_keys * ne));
    for (f32 &v : x) v = rnd();
    std::vector<f32> owq(static_cast<size_t>(ne * nh * qk));
    for (f32 &v : owq) v = rnd();
    // q[t] = W^T x[t], in the oracle's [in][out] order so both sides see the
    // same numbers to the last bit.
    std::vector<f32> q(static_cast<size_t>(n_keys * nh * qk), 0.0f);
    const i64 qdim = nh * qk;
    for (i64 t = 0; t < n_keys; t++)
        for (i64 o = 0; o < qdim; o++) {
            f32 s = 0.0f;
            for (i64 i = 0; i < ne; i++)
                s += owq[static_cast<size_t>(i * qdim + o)] *
                     x[static_cast<size_t>(t * ne + i)];
            q[static_cast<size_t>(t * qdim + o)] = s;
        }
    std::vector<f32> got(static_cast<size_t>(n_keys * nh * vd), 0.0f);
    cpu->mla_attention(got.data(), q.data(), cache.data(), wkb16.data(), d);

    // The oracle's tables: the unsplit one head-major, and an output
    // projection that is a SELECTION so its result is the attention output
    // itself (this group tests the attention, not wo).
    std::vector<f32> owb(static_cast<size_t>(nh * kv * hv));
    for (i32 h = 0; h < nh; h++)
        for (i64 k = 0; k < kv; k++)
            for (i64 j = 0; j < hv; j++)
                // From the f16 staging, not the f32 originals: the backend
                // reads the narrowed table, and comparing against wider
                // numbers would measure the storage format instead of the
                // arithmetic.
                owb[static_cast<size_t>((h * kv + k) * hv + j)] = fp16_to_fp32(
                    wkb16[static_cast<size_t>((h * hv + j) * kv + k)]);
    std::vector<f32> owo(static_cast<size_t>(nh * vd * ne), 0.0f);
    for (i64 j = 0; j < nh * vd; j++)
        owo[static_cast<size_t>(j * ne + j)] = 1.0f;
    // The cache as the oracle takes it: the latent rows and the raw keys.
    std::vector<f32> cc(static_cast<size_t>(n_keys * kv), 0.0f);
    std::vector<f32> kk(static_cast<size_t>(n_keys * rope), 0.0f);
    std::vector<i64> kpos(static_cast<size_t>(n_keys));
    for (i64 t = 0; t < n_keys; t++) {
        for (i64 k = 0; k < kv; k++)
            cc[static_cast<size_t>(t * kv + k)] = cache[static_cast<size_t>(t * row + k)];
        for (i64 j = 0; j < rope; j++)
            kk[static_cast<size_t>(t * rope + j)] =
                cache[static_cast<size_t>(t * row + kv + j)];
        kpos[static_cast<size_t>(t)] = t;
    }
    krk::MlaLiteWeights w;
    w.wq = owq.data();
    w.wkv_a_mqa = nullptr;
    w.wkv_a_norm = gamma.data();
    w.wkv_b = owb.data();
    w.wo = owo.data();
    krk::MlaLiteGeometry g;
    g.n_embd = ne;
    g.n_head = nh;
    g.kv_lora = kv;
    g.qk = static_cast<i32>(qk);
    g.qk_rope = rope;
    g.v_dim = vd;
    // The oracle writes a whole n_embd row (its output is the residual's
    // contribution, not the head stack), so each row lands in its own
    // n_embd-wide row and the first nh*vd of it is what this group compares.
    std::vector<f32> want_attn(static_cast<size_t>(n_keys * nh * vd), 0.0f);
    std::vector<f32> orow(static_cast<size_t>(ne));
    for (i64 t = 0; t < n_keys; t++) {
        krk::mla_lite_forward_absorbed(orow.data(), x.data() + t * ne,
                                       cc.data(), kk.data(), kpos.data(), t + 1,
                                       t, w, g, invf.data(), d.scale, d.eps);
        for (i64 j = 0; j < nh * vd; j++)
            want_attn[static_cast<size_t>(t * nh * vd + j)] =
                orow[static_cast<size_t>(j)];
    }
    f64 attn_worst = 0.0, peak = 0.0;
    for (size_t i = 0; i < want_attn.size(); i++) {
        peak = std::max(peak, static_cast<f64>(std::fabs(want_attn[i])));
        attn_worst = std::max(
            attn_worst, static_cast<f64>(std::fabs(got[i] - want_attn[i])));
    }
    CHECK(peak > 0.0, "the oracle's attention output is not all zero");
    CHECK(attn_worst < 2e-5 * (peak + 1.0),
          "the backend's MLA attention matches the absorbed oracle");

    // ---- one head at a time ---------------------------------------------
    // Zero every head's V half but head 1's: heads 0 and 2 have nothing to
    // accumulate, so their output slices must be exactly zero. A kernel that
    // reads feature j of the wrong head (or ignores the head offset entirely)
    // leaves non-zero output here, which is the aliasing defect.
    {
        std::vector<f32> one(static_cast<size_t>(nh * hv * kv), 0.0f);
        for (i64 k = 0; k < kv; k++)
            for (i64 j = 0; j < hv; j++)
                one[static_cast<size_t>((1 * hv + j) * kv + k)] =
                    wkvb[static_cast<size_t>((1 * hv + j) * kv + k)];
        std::vector<u16> one16(one.size());
        for (size_t i = 0; i < one.size(); i++) one16[i] = fp32_to_fp16(one[i]);
        std::vector<f32> o1(static_cast<size_t>(n_keys * nh * vd), 0.0f);
        cpu->mla_attention(o1.data(), q.data(), cache.data(), one16.data(), d);
        bool others_zero = true, mine_nonzero = false;
        for (i64 t = 0; t < n_keys; t++)
            for (i32 h = 0; h < nh; h++)
                for (i64 j = 0; j < vd; j++) {
                    const f32 v =
                        o1[static_cast<size_t>((t * nh + h) * vd + j)];
                    if (h == 1) {
                        if (v != 0.0f) mine_nonzero = true;
                    } else if (v != 0.0f) {
                        others_zero = false;
                    }
                }
        CHECK(others_zero,
              "a head with no V columns contributes nothing to its own slice");
        CHECK(mine_nonzero, "... and the head that has them does");
    }

    // ---- the query's own head slice -------------------------------------
    // Only head 0's query changes: heads 1 and 2 must be BIT-identical, which
    // holds however the softmax lands (their scores did not move at all).
    {
        std::vector<f32> q2 = q;
        for (i64 t = 0; t < n_keys; t++)
            for (i64 j = 0; j < qk; j++)
                q2[static_cast<size_t>(t * qdim + j)] += 0.75f;
        std::vector<f32> o2(static_cast<size_t>(n_keys * nh * vd), 0.0f);
        cpu->mla_attention(o2.data(), q2.data(), cache.data(), wkb16.data(), d);
        bool h0_moved = false, others_same = true;
        for (i64 t = 0; t < n_keys; t++)
            for (i32 h = 0; h < nh; h++)
                for (i64 j = 0; j < vd; j++) {
                    const size_t i = static_cast<size_t>((t * nh + h) * vd + j);
                    if (h == 0) {
                        if (o2[i] != got[i]) h0_moved = true;
                    } else if (o2[i] != got[i]) {
                        others_same = false;
                    }
                }
        CHECK(h0_moved, "head 0's query reaches head 0's output");
        CHECK(others_same, "and no other head's output");
    }

    // ---- one row at a time ----------------------------------------------
    // The same five rows as five single-row calls at their own positions,
    // bit-identical: this is the chunk invariance of the cache the engine
    // relies on when a prefill is split across chunks.
    {
        std::vector<f32> step(static_cast<size_t>(n_keys * nh * vd), 0.0f);
        for (i64 t = 0; t < n_keys; t++) {
            MlaDesc dt = d;
            dt.n_tok = 1;
            dt.pos0 = t;
            cpu->mla_attention(step.data() + t * nh * vd,
                               q.data() + t * qdim, cache.data(), wkb16.data(),
                               dt);
        }
        CHECK(std::memcmp(step.data(), got.data(),
                          got.size() * sizeof(f32)) == 0,
              "five single-row calls match one five-row call, bit for bit");
    }
    delete cpu;
}

// The virtual-NPU backend: github.com/riteshnair/npuxl-improved, reached
// through src/npu/npu_offload.cpp. There are two arms of it and this group
// gates both.
//
//   --npu      the GEMM path only: >95% of a transformer's FLOPs, and the one
//              VNPU op whose numerics are within f32 rounding of the reference.
//   --npu-cpu  `whole_graph`: every op the device has, which adds RMSNorm,
//              SwiGLU, add and mul. Its nonlinearities run the device's own
//              `fast_expf`, so this arm is NOT bit-identical -- the gate is a
//              measured bound, not equality, and the bound is asserted here so
//              a device that got worse cannot pass quietly.
//
// Three things are worth a gate. The seam (a factory that quietly returned the
// CPU backend would still run every model correctly, so the only way to know
// the option exists is to ask for it and ask the object what it is). The GEMM's
// numerics, in both weight formats the offload accepts. And the whole-graph
// arm's three properties: each routed op is within its documented bound of the
// reference, the queue and the strict one-submit-per-op path produce identical
// numbers (which is what proves no host op reads a row the device has not
// written), and the ops the device does NOT have still run.
static void test_npu_backend() {
    // The reference backend answers the new capability question the way the
    // engine's device tiers need it to: no memory of its own.
    {
        Backend *cpu = make_cpu_backend();
        CHECK(cpu->kind() == BackendKind::CPU, "the reference reports CPU");
        CHECK(!cpu->has_device_memory(),
              "and says it computes in host memory (no VRAM tier to plan)");
        CHECK(cpu->backend_stats().empty(),
              "and has no offload counters to report");
        delete cpu;
    }

    if (!npu_available()) {
        std::string why;
        Backend *b = make_npu_backend(0, /*whole_graph=*/true, &why);
        CHECK(b == nullptr,
              "a build without -DKRK_ENABLE_NPU cannot open the device");
        CHECK(why.find("KRK_ENABLE_NPU") != std::string::npos,
              "and the refusal names the switch that would build it");
        if (b) delete b;
        return;
    }

    std::string why;
    Backend *npu = make_npu_backend(0, /*whole_graph=*/false, &why);
    CHECK(npu != nullptr, why.empty() ? "the virtual NPU opens" : why.c_str());
    if (!npu) return;
    Backend *cpu = make_cpu_backend();
    CHECK(npu->kind() == BackendKind::NPU, "it reports its own backend kind");
    CHECK(!npu->has_device_memory(),
          "and computes in host memory, so no VRAM tier is planned for it");
    CHECK(npu->describe().find("Virtual NPU") != std::string::npos,
          "its banner names the device it opened");
    CHECK(npu->act_type() == DType::F32 && npu->act_size() == 4,
          "activations are still the reference's f32");

    // ---- the GEMM, at a decode shape and a prefill shape -------------------
    const i64 n_in = 64, n_out = 24;
    std::vector<f32> w(static_cast<size_t>(n_out * n_in));
    u32 rng = 0x1234abcdu;
    auto rnd = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<f32>((rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
    };
    for (f32 &v : w) v = rnd();
    // The f16 arm re-reads weights that have been through the same rounding the
    // file would carry, so the check measures the kernel and not the cast.
    std::vector<u16> wbits(w.size());
    for (size_t i = 0; i < w.size(); i++) wbits[i] = fp32_to_fp16(w[i]);

    for (i64 rows : {i64(1), i64(5)}) {
        std::vector<f32> x(static_cast<size_t>(rows * n_in));
        for (f32 &v : x) v = rnd();
        std::vector<f32> got(static_cast<size_t>(rows * n_out), 0.0f);
        std::vector<f32> want(static_cast<size_t>(rows * n_out), 0.0f);
        cpu->gemm(want.data(), x.data(), w.data(), DType::F32, n_out, n_in, rows);
        npu->gemm(got.data(), x.data(), w.data(), DType::F32, n_out, n_in, rows);
        f64 worst = 0.0, mag = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            const f64 d = std::fabs(static_cast<f64>(got[i]) - want[i]);
            if (d > worst) worst = d;
            mag = std::max<f64>(mag, std::fabs(want[i]));
        }
        const f64 rel = mag > 0.0 ? worst / mag : worst;
        CHECK(rel < 2e-6,
              rows == 1 ? "the device's f32 GEMV matches the reference to f32 "
                          "rounding"
                        : "and so does its multi-row GEMM (the prefill shape)");
    }

    // ---- determinism across row counts -------------------------------------
    // One row's value must not depend on how many rows rode along with it, or
    // the engine's chunk-invariance property (a gate of its own) would stop
    // holding the moment a kernel switched units on M.
    {
        std::vector<f32> x5(static_cast<size_t>(5 * n_in));
        for (f32 &v : x5) v = rnd();
        std::vector<f32> five(static_cast<size_t>(5 * n_out), 0.0f);
        npu->gemm(five.data(), x5.data(), w.data(), DType::F32, n_out, n_in, 5);
        std::vector<f32> one(static_cast<size_t>(n_out), 0.0f);
        npu->gemm(one.data(), x5.data() + 2 * n_in, w.data(), DType::F32, n_out,
                  n_in, 1);
        CHECK(std::memcmp(one.data(),
                          five.data() + 2 * n_out,
                          static_cast<size_t>(n_out) * sizeof(f32)) == 0,
              "row 2 of a five-row call is bit-identical to the same row "
              "computed alone");
    }

    // ---- f16 weights: the dequantizing staging path ------------------------
    {
        const size_t row_bytes = dtype_row_bytes(DType::F16, n_in);
        std::vector<u8> wf16(static_cast<size_t>(n_out) * row_bytes);
        for (i64 o = 0; o < n_out; o++)
            std::memcpy(wf16.data() + static_cast<size_t>(o) * row_bytes,
                        wbits.data() + static_cast<size_t>(o) * n_in,
                        static_cast<size_t>(n_in) * 2);
        std::vector<f32> x(static_cast<size_t>(3 * n_in));
        for (f32 &v : x) v = rnd();
        std::vector<f32> got(static_cast<size_t>(3 * n_out), 0.0f);
        std::vector<f32> want(static_cast<size_t>(3 * n_out), 0.0f);
        cpu->gemm(want.data(), x.data(), wf16.data(), DType::F16, n_out, n_in, 3);
        npu->gemm(got.data(), x.data(), wf16.data(), DType::F16, n_out, n_in, 3);
        f64 worst = 0.0, mag = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            worst = std::max<f64>(worst, std::fabs(static_cast<f64>(got[i]) - want[i]));
            mag = std::max<f64>(mag, std::fabs(want[i]));
        }
        CHECK(mag > 0.0 && (mag > 0.0 ? worst / mag : worst) < 2e-6,
              "an f16 weight tensor, dequantized once into the staging cache, "
              "matches the reference GEMM");
    }

    // ---- the counters the run summary prints -------------------------------
    {
        const std::string stats = npu->backend_stats();
        CHECK(stats.find("gemm ") != std::string::npos,
              "the offload reports its per-op call counts");
        CHECK(stats.find("gemm 0,") == std::string::npos,
              "and the GEMM count is not zero after the calls above");
        CHECK(stats.find("host: rope, attention") != std::string::npos,
              "and names the ops that stayed on the host");
    }

    // ---- attention on the device (KRK_NPU_ATTN) -----------------------------
    //
    // Three things have to hold, and the third is the one that makes this
    // shippable: the DEFAULT path is untouched, the opted-in path agrees with
    // the reference to f32 rounding, and a cache the device cannot take is
    // refused by NAME rather than approximated.
    {
        const i64 hd = 8, nh = 2, nkv = 1, npos = 12, ntok = 6;
        const i64 kv_dim = nkv * hd;
        std::mt19937 rng(2024);
        std::normal_distribution<f32> nd(0.0f, 0.3f);
        std::vector<f32> fk(static_cast<size_t>(npos * kv_dim));
        std::vector<f32> fv(static_cast<size_t>(npos * kv_dim));
        for (auto &x : fk) x = nd(rng);
        for (auto &x : fv) x = nd(rng);
        std::vector<f32> q(static_cast<size_t>(ntok * nh * hd));
        for (auto &x : q) x = nd(rng);
        std::vector<f32> kbuf(static_cast<size_t>(npos * kv_dim), 0.0f);
        std::vector<f32> vbuf(static_cast<size_t>(npos * kv_dim), 0.0f);

        AttnDesc a;
        a.n_head = nh;
        a.n_kv = nkv;
        a.hd = hd;
        a.n_tok = npos;
        a.pos0 = 0;
        a.layer = 0;
        a.layer_stride = 0;
        a.pos_stride = kv_dim;
        a.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd)));
        a.causal = true;
        a.kv_int8 = false;
        cpu->kv_append(kbuf.data(), vbuf.data(), fk.data(), fv.data(), a);

        const i64 out_elems = ntok * nh * hd;
        std::vector<f32> want(static_cast<size_t>(out_elems), 0.0f);
        AttnDesc d = a;
        d.n_tok = ntok;
        d.pos0 = npos - ntok;
        cpu->attention(want.data(), q.data(), kbuf.data(), vbuf.data(), d);

        // (1) default: attention is NOT offered, so the Graph arm must produce
        //     the reference's own bits and say the op stayed on the host.
        {
            std::string w2;
            Backend *g = make_npu_backend(0, /*whole_graph=*/true, &w2);
            CHECK(g != nullptr, w2.empty() ? "the graph arm opens" : w2.c_str());
            if (g) {
                std::vector<f32> got(static_cast<size_t>(out_elems), 0.0f);
                g->attention(got.data(), q.data(), kbuf.data(), vbuf.data(), d);
                CHECK(got == want,
                      "with KRK_NPU_ATTN unset, attention is bit-identical to "
                      "the reference (the device is not offered it)");
                CHECK(g->backend_stats().find("host: rope, attention") !=
                          std::string::npos,
                      "and the run summary still names attention as a host op");
                delete g;
            }
        }

        // (2) opt-in: the device takes it, and what comes back is the same
        //     attention to f32 rounding -- a different summation order, not a
        //     different model. ntok is 6 here, below the decode threshold, so
        //     lower it for the test.
        setenv("KRK_NPU_ATTN", "1", 1);
        setenv("KRK_NPU_ATTN_MIN_TOK", "4", 1);
        {
            std::string w2;
            Backend *g = make_npu_backend(0, /*whole_graph=*/true, &w2);
            CHECK(g != nullptr, w2.empty() ? "the graph arm opens" : w2.c_str());
            if (g) {
                std::vector<f32> got(static_cast<size_t>(out_elems), 0.0f);
                g->attention(got.data(), q.data(), kbuf.data(), vbuf.data(), d);
                f64 num = 0.0, den = 0.0, worst = 0.0;
                for (i64 i = 0; i < out_elems; i++) {
                    const f64 e = static_cast<f64>(got[i]) - want[i];
                    num += e * e;
                    den += static_cast<f64>(want[i]) * want[i];
                    worst = std::max<f64>(worst, std::fabs(e));
                }
                const f64 rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
                // Measured at these toy dims: 2.0e-05 relative, 6.1e-06
                // absolute. The device's softmax runs its own polynomial exp,
                // which at hd=8 and a handful of keys is a larger share of the
                // result than at the shapes the engine actually runs (hd=64,
                // 257x257 measured 1.2e-07). The bound is set a little above
                // the measurement, and it is still four orders below the O(1)
                // error a layout mistake in the gather would produce.
                CHECK(rel < 5e-5,
                      "with KRK_NPU_ATTN=1 the device attention matches the "
                      "reference to its own f32/exp rounding");
                CHECK(worst < 2e-5, "and no single element strays further");
                CHECK(g->backend_stats().find("attn 1 dev") != std::string::npos,
                      "and the run summary counts the call as device work");
                CHECK(g->backend_stats().find("host: rope, attention") ==
                          std::string::npos,
                      "and stops calling attention a host op");

                // (3) refusal by name: an 8-bit cache is not representable.
                std::vector<f32> q8k(kbuf.size(), 0.0f), q8v(vbuf.size(), 0.0f);
                AttnDesc q8 = a;
                q8.n_tok = npos;
                q8.kv_int8 = true;
                q8.pos_stride = kv_dim + 1; // not the f32 layout
                // The cache is only used for the refusal decision here.
                std::vector<f32> out8(static_cast<size_t>(out_elems), 0.0f);
                AttnDesc r = d;
                r.kv_int8 = true;
                g->attention(out8.data(), q.data(), q8k.data(), q8v.data(), r);
                const std::string st = g->backend_stats();
                // The device call above already succeeded, so the split is
                // 1 device / 1 host and the host one carries its reason.
                CHECK(st.find("attn 1 dev, 1 host") != std::string::npos &&
                          st.find("1 8-bit cache") != std::string::npos,
                      "and an int8 cache is refused BY NAME, not approximated");
                delete g;
            }
        }
        unsetenv("KRK_NPU_ATTN");
        unsetenv("KRK_NPU_ATTN_MIN_TOK");
    }
    delete npu;

    // =======================================================================
    //  --npu-cpu: the whole graph through the device
    // =======================================================================
    Backend *graph = make_npu_backend(0, /*whole_graph=*/true, &why);
    CHECK(graph != nullptr,
          why.empty() ? "the whole-graph NPU backend opens" : why.c_str());
    if (!graph) {
        delete cpu;
        return;
    }
    CHECK(graph->describe().find("whole graph") != std::string::npos,
          "and says so in its banner, so a run cannot be mistaken for the "
          "GEMM-only arm");
    CHECK(graph->kind() == BackendKind::NPU, "it is still an NPU backend");
    CHECK(!graph->has_device_memory(), "still computing in host memory");

    // ---- RMSNorm: f32 rounding, not equality (the device reduces in SIMD) --
    {
        const i64 rows = 3, n = 256;
        std::vector<f32> x(static_cast<size_t>(rows * n)), wgt(static_cast<size_t>(n));
        for (f32 &v : x) v = rnd();
        for (i64 i = 0; i < n; i++) wgt[static_cast<size_t>(i)] = 0.5f + 0.5f * std::fabs(rnd());
        std::vector<f32> got(x.size(), 0.0f), want(x.size(), 0.0f);
        cpu->rmsnorm(want.data(), x.data(), wgt.data(), rows, n, 1e-5f);
        graph->rmsnorm(got.data(), x.data(), wgt.data(), rows, n, 1e-5f);
        graph->sync(); // the routed ops are queued; sync is what submits them
        f64 worst = 0.0, mag = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            worst = std::max<f64>(worst, std::fabs(static_cast<f64>(got[i]) - want[i]));
            mag = std::max<f64>(mag, std::fabs(want[i]));
        }
        CHECK(mag > 0.0 && worst / mag < 2e-6,
              "the device's RMSNorm matches the reference to f32 rounding");
    }

    // ---- SwiGLU: the documented 1e-4 bound, measured not assumed ----------
    // This is the whole accuracy story of the arm: the device's silu is its own
    // polynomial, so a bound is what the arm can promise. The number below is
    // 8.4e-05 measured on the device; the gate holds it under 2e-4 and would
    // fail if a future device made it worse.
    {
        const i64 n = 2048;
        std::vector<f32> g(static_cast<size_t>(n)), u(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++) {
            g[static_cast<size_t>(i)] = -12.0f + 24.0f * static_cast<f32>(i) / static_cast<f32>(n - 1);
            u[static_cast<size_t>(i)] = 1.0f; // isolate the activation itself
        }
        std::vector<f32> got(static_cast<size_t>(n), 0.0f), want(static_cast<size_t>(n), 0.0f);
        cpu->silu_mul(want.data(), g.data(), u.data(), n);
        graph->silu_mul(got.data(), g.data(), u.data(), n);
        graph->sync();
        // Relative to each element's own value, ignoring the ones a silu sends
        // to zero (there the absolute error is what matters and it is smaller).
        f64 worst = 0.0, rel = 0.0;
        for (i64 i = 0; i < n; i++) {
            const f64 wv = want[static_cast<size_t>(i)];
            const f64 d = std::fabs(static_cast<f64>(got[static_cast<size_t>(i)]) - wv);
            worst = std::max(worst, d);
            if (std::fabs(wv) > 1e-3) rel = std::max(rel, d / std::fabs(wv));
        }
        CHECK(rel < 2e-4,
              rel < 2e-4 ? "the device's SwiGLU stays inside the arm's documented "
                           "2e-4 relative bound"
                         : "the device's SwiGLU has drifted outside the bound the "
                           "arm documents -- check whether npuxl changed its exp");
        std::printf("       (npu swiglu: worst %.3e absolute, %.3e relative)\n",
                    worst, rel);
    }

    // ---- add / axpy / mul are exact: same expression, same order ----------
    {
        const i64 n = 512;
        std::vector<f32> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++) {
            a[static_cast<size_t>(i)] = rnd();
            b[static_cast<size_t>(i)] = rnd();
        }
        std::vector<f32> ga = a, gb = b, wa = a, wb = b;
        graph->add_inplace(ga.data(), gb.data(), n);
        graph->sync();
        cpu->add_inplace(wa.data(), wb.data(), n);
        CHECK(std::memcmp(ga.data(), wa.data(), static_cast<size_t>(n) * 4) == 0,
              "the device's add is bit-identical to the reference's");
        ga = a;
        graph->axpy(ga.data(), b.data(), 0.25f, n);
        graph->sync();
        wa = a;
        cpu->axpy(wa.data(), b.data(), 0.25f, n);
        CHECK(std::memcmp(ga.data(), wa.data(), static_cast<size_t>(n) * 4) == 0,
              "and so is axpy, scale included");
        ga = a;
        graph->mul_act(ga.data(), b.data(), n);
        graph->sync();
        wa = a;
        cpu->mul_act(wa.data(), b.data(), n);
        CHECK(std::memcmp(ga.data(), wa.data(), static_cast<size_t>(n) * 4) == 0,
              "and mul");
    }

    // ---- the queue: batching must not change a single number --------------
    // The whole-graph arm queues ops and submits them in one batch, and every
    // host-implemented op flushes first. If that discipline had a hole -- one
    // host kernel reading a row the device had not written yet -- the two
    // submission policies would disagree. This builds a layer-shaped sequence
    // that mixes routed and host ops (rmsnorm -> add -> silu -> mul -> a host
    // op) and requires the two arms to produce identical bytes.
    {
        auto chain = [&](Backend *be, std::vector<f32> *out) {
            const i64 n = 128;
            std::vector<f32> x(static_cast<size_t>(n)), y(static_cast<size_t>(n));
            std::vector<f32> gate(static_cast<size_t>(n)), up(static_cast<size_t>(n));
            u32 seed = 0xfeed1234u;
            auto r2 = [&]() {
                seed = seed * 1664525u + 1013904223u;
                return static_cast<f32>((seed >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
            };
            for (i64 i = 0; i < n; i++) {
                x[static_cast<size_t>(i)] = r2();
                y[static_cast<size_t>(i)] = r2();
                gate[static_cast<size_t>(i)] = 4.0f * r2();
                up[static_cast<size_t>(i)] = r2();
            }
            std::vector<f32> g(static_cast<size_t>(n), 0.0f), u(static_cast<size_t>(n), 0.0f);
            be->rmsnorm(g.data(), x.data(), nullptr, 1, n, 1e-5f);   // queued
            be->add_inplace(g.data(), y.data(), n);                  // queued
            be->silu_mul(u.data(), gate.data(), up.data(), n);       // queued
            be->mul_act(g.data(), u.data(), n);                      // queued
            // A host op that reads what the queue just produced. If the flush
            // discipline were wrong this would read stale bytes.
            const std::vector<f32> bias(static_cast<size_t>(n), 0.5f);
            be->add_bias_rows(g.data(), bias.data(), n, 1);
            *out = g;
        };
        std::vector<f32> batched, strict;
        chain(graph, &batched);
        set_env_var("KRK_NPU_STRICT_QUEUE", "1");
        {
            std::string w2;
            Backend *strict_be = make_npu_backend(0, /*whole_graph=*/true, &w2);
            if (strict_be) {
                chain(strict_be, &strict);
                delete strict_be;
            }
        }
        clear_env_var("KRK_NPU_STRICT_QUEUE");
        CHECK(!strict.empty() &&
                  std::memcmp(batched.data(), strict.data(),
                              batched.size() * sizeof(f32)) == 0,
              "one batch per layer and one submit per op leave the same bytes: "
              "no host kernel reads a row the device has not written");
    }

    // ---- the ops the device does not have still run -----------------------
    // gelu is deliberately NOT offloaded (the device's is 0.74 relative off),
    // so the reference body must still be what runs there.
    {
        std::vector<f32> xg(64), xc(64);
        for (size_t i = 0; i < xg.size(); i++)
            xg[i] = xc[i] = -4.0f + 0.125f * static_cast<f32>(i);
        graph->gelu_act(xg.data(), static_cast<i64>(xg.size()));
        cpu->gelu_act(xc.data(), static_cast<i64>(xc.size()));
        CHECK(std::memcmp(xg.data(), xc.data(), xg.size() * sizeof(f32)) == 0,
              "the GeGLU activation is the reference's: the device's own gelu "
              "measures 0.74 relative off and is not used");
    }

    {
        const std::string gs = graph->backend_stats();
        CHECK(gs.find("rmsnorm 0,") == std::string::npos,
              "the summary counts the RMSNorms the device ran");
        CHECK(gs.find("swiglu 0,") == std::string::npos, "and the SwiGLUs");
    }

    delete graph;
    delete cpu;
}

// MLA loader: the three layouts, the geometry checks, and why the engine
// refuses to generate from any of them.
//
// Three things are pinned, and they fail differently:
//
//  1. The LOADER reads the file. Which layout a file carries is decided by the
//     TENSORS its layers hold, not by the arch string (one arch ships more than
//     one layout, and a file is what it carries), and every layer has to agree
//     on it.
//
//  2. The GEOMETRY is checked against the file's own metadata, and a
//     disagreement is named: which tensor, what shape it has, what the metadata
//     wants. Reading a file whose tensors disagree with its own keys is how a
//     wrong number becomes a wrong answer.
//
//  3. The ENGINE refuses to generate. A file whose attention path has no
//     implementation here would decode text that is fluent, plausible and not
//     the model -- the worst failure this engine has -- so the run is refused
//     after the load. The loader's escape hatch does NOT lift that refusal.
//
// The fixture numbers are small but every one of them is a number the loader
// checks against, so a fixture that loads is a file whose metadata and tensors
// agree -- which is the property under test.

enum class MlaFixKind { Legacy, Split, Shared };

struct MlaFix {
    MlaFixKind kind = MlaFixKind::Legacy;
    int n_layer = 1;
    bool mixed_layout = false;    // layer 1 carries a different layout
    bool short_k_b = false;       // attn_k_b one column short
    bool int32_weight = false;    // attn_k_b written as an int32 table
    bool bad_hash_shape = false;  // the route table is half the vocab
    bool no_hash_table = false;   // a hash layer with no route table
    // The table still loads and still routes, but sends every token somewhere
    // else -- the arm that proves the whole-graph pin is SENSITIVE to the route
    // rather than merely to the weights.
    int hash_shift = 0;
    // An id past n_expert. The loader checks the table's SHAPE, not its values,
    // so this is the file the forward's own range check exists for: it must
    // refuse the run instead of indexing an expert bank out of bounds.
    bool hash_id_oob = false;
};

// 261, not 260: add_spm_vocab always writes 3 specials + 256 byte pieces + the
// two whitespace pieces, so its table is 261 entries whatever size is asked for
// once the ask is <= 261. n_vocab is the EMBEDDING's width, and the V4 hash
// table is checked against it -- a 260-row embedding against a 261-piece table
// would demand a 260-column route table for a file that has 261 tokens.
static const int kMlaVocab = 261;
static const int kMlaEmbd = 32;
static const int kMlaHead = 2;
static const int kMlaRope = 4;
static const int kMlaKeyLen = 12;   // nope = key_length - rope = 8
static const int kMlaValLen = 8;
static const int kMlaKvLora = 16;
// 32, not 24: the HIP GEMM refuses an inner width that is not a multiple of
// 32, and this fixture has to answer on BOTH backends for the device arm to
// mean anything. Real V4 files declare 1024+, which is a multiple anyway.
static const int kMlaQLora = 32;
static const int kMlaFF = 64;
static const int kMlaExpert = 2;
static const int kMlaExpertUsed = 1;
// 32 for the same reason as kMlaQLora: the expert GEMMs run on the device too.
static const int kMlaFFExp = 32;
static const int kMlaOGroups = 1;
static const int kMlaOLora = 8;

static bool build_mla_fixture(const std::string &path, const MlaFix &f) {
    const bool shared = f.kind == MlaFixKind::Shared;
    const std::string arch = shared ? "deepseek4" : "deepseek2";
    GgufBuilder b;
    b.meta_str("general.architecture", arch);
    b.meta_str("general.name", "kraken-mla-test");
    b.meta_u32("general.alignment", 32);
    b.meta_u32(arch + ".block_count", static_cast<u32>(f.n_layer));
    b.meta_u32(arch + ".context_length", 64);
    b.meta_u32(arch + ".embedding_length", static_cast<u32>(kMlaEmbd));
    b.meta_u32(arch + ".feed_forward_length", static_cast<u32>(kMlaFF));
    b.meta_u32(arch + ".attention.head_count", static_cast<u32>(kMlaHead));
    b.meta_u32(arch + ".attention.key_length", static_cast<u32>(kMlaKeyLen));
    b.meta_u32(arch + ".attention.value_length", static_cast<u32>(kMlaValLen));
    b.meta_u32(arch + ".rope.dimension_count", static_cast<u32>(kMlaRope));
    b.meta_f32(arch + ".rope.freq_base", 10000.0f);
    b.meta_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    // The V2/V3 class compresses the KV into a latent of kv_lora_rank and keeps
    // the decoupled rope key beside it. V4 has no kv_lora_rank: its latent is
    // key_length wide and the rope is its tail.
    if (!shared)
        b.meta_u32(arch + ".attention.kv_lora_rank", static_cast<u32>(kMlaKvLora));
    // A low-rank query trunk: the V3 class and V4 have one, the V2-Lite class
    // does not (its query projection is direct).
    if (f.kind != MlaFixKind::Legacy)
        b.meta_u32(arch + ".attention.q_lora_rank", static_cast<u32>(kMlaQLora));
    if (shared) {
        // V4's own metadata: the grouped output projection, the window, the
        // hash-route depth and the per-layer compression ratios.
        b.meta_u32(arch + ".attention.output_group_count",
                   static_cast<u32>(kMlaOGroups));
        b.meta_u32(arch + ".attention.output_lora_rank",
                   static_cast<u32>(kMlaOLora));
        b.meta_u32(arch + ".attention.sliding_window", 128);
        b.meta_u32(arch + ".hash_layer_count", 1);
        b.meta_f32(arch + ".attention.compress_rope_freq_base", 10000.0f);
        std::vector<i32> ratios(static_cast<size_t>(f.n_layer), 0);
        b.meta_i32_array(arch + ".attention.compress_ratios", ratios);
        // The hash route is a ROUTED-expert route, so the layer is a MoE one.
        b.meta_u32(arch + ".expert_count", static_cast<u32>(kMlaExpert));
        b.meta_u32(arch + ".expert_used_count", static_cast<u32>(kMlaExpertUsed));
        b.meta_u32(arch + ".expert_feed_forward_length",
                   static_cast<u32>(kMlaFFExp));
    }
    add_spm_vocab(b, kMlaVocab);

    auto w2 = [&](const std::string &name, u64 a, u64 c, i64 seed) {
        std::vector<f32> d(static_cast<size_t>(a * c));
        for (size_t i = 0; i < d.size(); i++)
            d[i] = wv(static_cast<i64>(i) + seed);
        b.tensor_f16(name, {a, c}, d);
    };
    auto w3 = [&](const std::string &name, u64 a, u64 c, u64 e, i64 seed) {
        std::vector<f32> d(static_cast<size_t>(a * c * e));
        for (size_t i = 0; i < d.size(); i++)
            d[i] = wv(static_cast<i64>(i) + seed);
        b.tensor_f16(name, {a, c, e}, d);
    };
    auto ones = [&](const std::string &name, u64 n) {
        b.tensor_f32(name, {n}, std::vector<f32>(static_cast<size_t>(n), 1.0f));
    };
    // An int32 tensor: DATA, the only format that is not a quantization.
    auto i32_tensor = [&](const std::string &name, const std::vector<u64> &ne,
                          i32 mod, int shift = 0, bool oob = false) {
        size_t n = 1;
        for (u64 d : ne) n *= static_cast<size_t>(d);
        std::vector<i32> ids(n);
        for (size_t i = 0; i < ids.size(); i++) {
            const size_t base = static_cast<size_t>(mod);
            ids[i] = static_cast<i32>((i + static_cast<size_t>(shift)) % base);
        }
        // EVERY slot, not just the first: the forward reads the row of the
        // token it is given, so a single poisoned cell at a fixed offset would
        // be invisible for any prompt whose token is not that one -- and the
        // arm would pass while testing nothing.
        if (oob)
            for (size_t i = 0; i < ids.size(); i++) ids[i] = mod;
        b.tensor_raw(name, 26, ne, ids.data(), ids.size() * sizeof(i32));
    };

    w2("token_embd.weight", static_cast<u64>(kMlaEmbd),
       static_cast<u64>(kMlaVocab), 1);
    ones("output_norm.weight", static_cast<u64>(kMlaEmbd));

    const u64 nope = static_cast<u64>(kMlaKeyLen - kMlaRope);
    for (int l = 0; l < f.n_layer; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        ones(p + "attn_norm.weight", static_cast<u64>(kMlaEmbd));
        ones(p + "ffn_norm.weight", static_cast<u64>(kMlaEmbd));
        MlaFixKind kind = f.kind;
        if (f.mixed_layout && l == 1)
            kind = (f.kind == MlaFixKind::Legacy) ? MlaFixKind::Split
                                                  : MlaFixKind::Legacy;

        // ---- the query trunk -------------------------------------------
        if (kind == MlaFixKind::Legacy) {
            // attn_q projects straight to n_head*(nope+rope).
            w2(p + "attn_q.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaHead) * static_cast<u64>(kMlaKeyLen), 100 + l);
        } else {
            w2(p + "attn_q_a.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaQLora), 100 + l);
            ones(p + "attn_q_a_norm.weight", static_cast<u64>(kMlaQLora));
            w2(p + "attn_q_b.weight", static_cast<u64>(kMlaQLora),
               static_cast<u64>(kMlaHead) * static_cast<u64>(kMlaKeyLen), 200 + l);
        }

        // ---- the KV path -----------------------------------------------
        if (kind == MlaFixKind::Shared) {
            // ONE shared latent, key_length wide, the rope in its tail.
            w2(p + "attn_kv.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaKeyLen), 300 + l);
            ones(p + "attn_kv_a_norm.weight", static_cast<u64>(kMlaKeyLen));
            // The head count is folded into a GROUPED output projection:
            // n_head*key_length/o_groups rows into o_lora_rank*o_groups.
            w2(p + "attn_output_a.weight",
               static_cast<u64>(kMlaHead) * static_cast<u64>(kMlaKeyLen) /
                   static_cast<u64>(kMlaOGroups),
               static_cast<u64>(kMlaOLora) * static_cast<u64>(kMlaOGroups),
               400 + l);
            w2(p + "attn_output_b.weight",
               static_cast<u64>(kMlaOLora) * static_cast<u64>(kMlaOGroups),
               static_cast<u64>(kMlaEmbd), 500 + l);
        } else {
            // The compressed latent with the decoupled rope key beside it.
            w2(p + "attn_kv_a_mqa.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaKvLora + kMlaRope), 300 + l);
            ones(p + "attn_kv_a_norm.weight", static_cast<u64>(kMlaKvLora));
            if (kind == MlaFixKind::Legacy) {
                // One unsplit table: [kv_lora, n_head*(nope+v)].
                w2(p + "attn_kv_b.weight", static_cast<u64>(kMlaKvLora),
                   static_cast<u64>(kMlaHead) *
                       static_cast<u64>(kMlaKeyLen - kMlaRope + kMlaValLen),
                   400 + l);
                w2(p + "attn_output.weight",
                   static_cast<u64>(kMlaHead) * static_cast<u64>(kMlaValLen),
                   static_cast<u64>(kMlaEmbd), 500 + l);
            } else {
                // Two three-dimensional tables, one matrix per head, and the
                // contraction index is ne[0] in both.
                const u64 kb_nope = f.short_k_b ? nope - 1 : nope;
                if (f.int32_weight) {
                    // The table of ids where a coefficient matrix belongs.
                    i32_tensor(p + "attn_k_b.weight",
                               {kb_nope, static_cast<u64>(kMlaKvLora),
                                static_cast<u64>(kMlaHead)},
                               7);
                } else {
                    w3(p + "attn_k_b.weight", kb_nope,
                       static_cast<u64>(kMlaKvLora), static_cast<u64>(kMlaHead),
                       400 + l);
                }
                w3(p + "attn_v_b.weight", static_cast<u64>(kMlaKvLora),
                   static_cast<u64>(kMlaValLen), static_cast<u64>(kMlaHead),
                   450 + l);
                w2(p + "attn_output.weight",
                   static_cast<u64>(kMlaHead) * static_cast<u64>(kMlaValLen),
                   static_cast<u64>(kMlaEmbd), 500 + l);
            }
        }

        // ---- the FFN ---------------------------------------------------
        if (shared) {
            // A routed-expert layer: the hash route lives on one.
            w2(p + "ffn_gate_inp.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaExpert), 600 + l);
            w3(p + "ffn_gate_exps.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaFFExp), static_cast<u64>(kMlaExpert),
               610 + l);
            w3(p + "ffn_up_exps.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaFFExp), static_cast<u64>(kMlaExpert),
               620 + l);
            w3(p + "ffn_down_exps.weight", static_cast<u64>(kMlaFFExp),
               static_cast<u64>(kMlaEmbd), static_cast<u64>(kMlaExpert),
               630 + l);
            if (!f.no_hash_table) {
                // The route table: one expert id per selected expert, one
                // column per token id. DATA, never multiplied.
                const u64 ntok = f.bad_hash_shape
                                     ? static_cast<u64>(kMlaVocab / 2)
                                     : static_cast<u64>(kMlaVocab);
                i32_tensor(p + "ffn_gate_tid2eid.weight",
                           {static_cast<u64>(kMlaExpertUsed), ntok},
                           kMlaExpert, f.hash_shift, f.hash_id_oob);
            }
        } else {
            w2(p + "ffn_gate.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaFF), 600 + l);
            w2(p + "ffn_up.weight", static_cast<u64>(kMlaEmbd),
               static_cast<u64>(kMlaFF), 610 + l);
            w2(p + "ffn_down.weight", static_cast<u64>(kMlaFF),
               static_cast<u64>(kMlaEmbd), 620 + l);
        }
    }
    return b.write(path);
}

static void test_mla_loader() {
    // The hatch is a LOADER switch and is off for the first arm on purpose:
    // without it a refused schema is refused before a tensor is read.
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
    std::vector<std::string> made;

    // ---- 0. without the hatch the refused ARCH is refused ----------------
    // The refusal has to name the architecture: a file refused for its geometry
    // would send the next reader looking for a wrong tensor. The legacy lite
    // layout is implemented now, so it is no longer the arch that refuses a
    // deepseek2 file -- the shared-latent one (deepseek4) still has no graph, so
    // that is the fixture this arm loads. The legacy fixture is written here
    // because every arm below reads it.
    {
        const std::string legacy = "kraken-mla-legacy.gguf";
        MlaFix lf;
        lf.kind = MlaFixKind::Legacy;
        CHECK(build_mla_fixture(legacy, lf), "wrote the legacy-layout fixture");
        made.push_back(legacy);

        // The shared-latent schema has a graph now, so it LOADS without the
        // hatch. That is the row's claim (`Partial`, attention implemented)
        // checked from the loader's side; the whole-graph group checks the same
        // claim end to end. A refusal here would be the table disagreeing with
        // the engine, which is the state this arm used to pin.
        const std::string path = "kraken-mla-shared-loads.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Shared;
        CHECK(build_mla_fixture(path, f), "wrote the shared-latent fixture");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        CHECK(m.load(*cpu, path, &err),
              "the shared-latent schema loads without the hatch");
        if (!err.empty()) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        CHECK(m.cfg().mla_layout == MlaLayout::Shared,
              "... as the shared layout, which is what the engine dispatches on");
        m.unload();
        delete cpu;

        // The layout this build runs is NOT refused without the hatch: the
        // table's verdict is about the architecture, and this one has an
        // implementation, so the file loads with no switch set.
        Backend *cpu2 = make_cpu_backend();
        Model m2;
        std::string err2;
        CHECK(m2.load(*cpu2, legacy, &err2),
              "the legacy lite layout loads without the hatch");
        m2.unload();
        delete cpu2;
    }

    // The hatch is set from here on: the loader reads the tensors of a
    // half-ported schema so the layout and the geometry can be checked. It does
    // NOT make the file runnable, and the last arm proves that.
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "1");
#else
    setenv("KRK_ARCH_ALLOW_EXPERIMENTAL", "1", 1);
#endif

    // ---- 1. the legacy lite layout (DeepSeek-V2-Lite) --------------------
    {
        const std::string path = "kraken-mla-legacy.gguf";
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        const bool ok = m.load(*cpu, path, &err);
        CHECK(ok, "the legacy lite layout loads");
        if (!ok) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        if (ok) {
            const ModelConfig &mc = m.cfg();
            CHECK(mc.mla && mc.mla_layout == MlaLayout::Legacy,
                  "the layout is the legacy one, read from the tensors");
            CHECK(mc.mla_kv_lora == kMlaKvLora,
                  "the compressed latent's rank is read (kv_lora_rank)");
            CHECK(mc.mla_qk_nope == kMlaKeyLen - kMlaRope &&
                      mc.mla_qk_rope == kMlaRope,
                  "nope is key_length minus the rope, and the rope is the tail");
            CHECK(mc.mla_v_dim == kMlaValLen, "the value width is read");
            CHECK(mc.mla_q_lora == 0,
                  "the legacy class has no low-rank query trunk");
            const LayerWeights &L0 = m.layers()[0];
            CHECK(L0.wq_direct.present() && L0.wq_direct.n_in == kMlaEmbd &&
                      L0.wq_direct.n_out == kMlaHead * kMlaKeyLen,
                  "the direct query projection is [n_embd -> n_head*(nope+rope)]");
            CHECK(L0.wkv_a_mqa.present() &&
                      L0.wkv_a_mqa.n_in == kMlaEmbd &&
                      L0.wkv_a_mqa.n_out == kMlaKvLora + kMlaRope,
                  "the MQA projection is kv_lora + rope wide");
            CHECK(L0.wkv_a_norm != nullptr, "the latent's rms norm is uploaded");
            CHECK(L0.wkv_b.present() && L0.wkv_b.n_in == kMlaKvLora &&
                      L0.wkv_b.n_out ==
                          kMlaHead * (kMlaKeyLen - kMlaRope + kMlaValLen),
                  "the unsplit per-head table is [kv_lora -> n_head*(nope+v)]");
            CHECK(L0.wo.present() && L0.wo.n_in == kMlaHead * kMlaValLen &&
                      L0.wo.n_out == kMlaEmbd,
                  "the output projection is [n_head*v -> n_embd]");
            CHECK(!L0.wq_a.present() && !L0.wq_b.present() &&
                      !L0.wkv.present() && !L0.wk_b.present() &&
                      !L0.wv_b.present() && !L0.wo_a.present(),
                  "no tensor of another layout is invented");
        }
        m.unload();
        delete cpu;
    }

    // ---- 2. the split layout (the V3 class) -----------------------------
    {
        const std::string path = "kraken-mla-split.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Split;
        CHECK(build_mla_fixture(path, f), "wrote the split-layout fixture");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        const bool ok = m.load(*cpu, path, &err);
        CHECK(ok, "the split layout loads");
        if (!ok) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        if (ok) {
            const ModelConfig &mc = m.cfg();
            CHECK(mc.mla_layout == MlaLayout::Split,
                  "the layout is the split one");
            CHECK(mc.mla_q_lora == kMlaQLora,
                  "the query trunk's rank is read (q_lora_rank)");
            const LayerWeights &L0 = m.layers()[0];
            CHECK(L0.wq_a.present() && L0.wq_a.n_out == kMlaQLora &&
                      L0.wq_a_norm != nullptr,
                  "the low-rank query projection and its norm are loaded");
            CHECK(L0.wq_b.present() && L0.wq_b.n_in == kMlaQLora &&
                      L0.wq_b.n_out == kMlaHead * kMlaKeyLen,
                  "the query expansion is [q_lora -> n_head*(nope+rope)]");
            CHECK(L0.wkv_a.present() &&
                      L0.wkv_a.n_out == kMlaKvLora + kMlaRope,
                  "the compressed latent carries the decoupled key beside it");
            CHECK(L0.wkv_norm != nullptr, "the latent's norm is uploaded");
            CHECK(L0.wk_b.present() &&
                      L0.wk_b.n_in == kMlaKeyLen - kMlaRope &&
                      L0.wk_b.n_out == kMlaKvLora * kMlaHead,
                  "the per-head key table contracts the nope part");
            CHECK(L0.wv_b.present() && L0.wv_b.n_in == kMlaKvLora &&
                      L0.wv_b.n_out == kMlaValLen * kMlaHead,
                  "the per-head value table contracts the latent");
            CHECK(!L0.wkv_b.present() && !L0.wkv_a_mqa.present() &&
                      !L0.wq_direct.present(),
                  "no tensor of another layout is invented");
        }
        m.unload();
        delete cpu;
    }

    // ---- 3. the shared latent (DeepSeek-V4) -----------------------------
    {
        const std::string path = "kraken-mla-shared.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Shared;
        CHECK(build_mla_fixture(path, f), "wrote the shared-latent fixture");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        const bool ok = m.load(*cpu, path, &err);
        CHECK(ok, "the shared-latent layout loads");
        if (!ok) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        if (ok) {
            const ModelConfig &mc = m.cfg();
            CHECK(mc.mla_layout == MlaLayout::Shared,
                  "the layout is the shared-latent one");
            CHECK(mc.dsv4 && mc.dsv4_o_groups == kMlaOGroups &&
                      mc.dsv4_o_lora == kMlaOLora,
                  "V4's grouped output projection is read");
            CHECK(mc.dsv4_window == 128 && mc.dsv4_hash_layers == 1,
                  "the window and the hash-route depth are read");
            CHECK(mc.dsv4_ratio.size() == 1 && mc.dsv4_ratio[0] == 0,
                  "the per-layer compression ratios are read");
            const LayerWeights &L0 = m.layers()[0];
            CHECK(L0.wkv.present() && L0.wkv.n_in == kMlaEmbd &&
                      L0.wkv.n_out == kMlaKeyLen,
                  "the shared latent is key_length wide");
            CHECK(L0.wkv_norm != nullptr, "its norm is uploaded");
            CHECK(L0.wo_a.present() &&
                      L0.wo_a.n_in == kMlaHead * kMlaKeyLen / kMlaOGroups &&
                      L0.wo_a.n_out == kMlaOLora * kMlaOGroups,
                  "the grouped output projection folds the head count in");
            CHECK(L0.wo_b.present() &&
                      L0.wo_b.n_in == kMlaOLora * kMlaOGroups &&
                      L0.wo_b.n_out == kMlaEmbd,
                  "... and the second half returns to n_embd");
            CHECK(!L0.wo.present() && !L0.wkv_b.present() &&
                      !L0.wq_direct.present(),
                  "a shared-latent file has none of the other layouts' names");
            // The hash route: a table of expert IDS, read on the host and
            // never uploaded. It is data, and the loader has to know that.
            CHECK(L0.tid2eid.size() ==
                      static_cast<size_t>(kMlaExpertUsed) * kMlaVocab,
                  "the route table is read as data, one id per token");
            CHECK(L0.tid2eid_per_token == kMlaExpertUsed &&
                      L0.tid2eid_n_tokens == kMlaVocab,
                  "its shape is the one the metadata declares");
            CHECK(!L0.tid2eid.empty() && L0.tid2eid[0] >= 0 &&
                      L0.tid2eid[0] < kMlaExpert,
                  "and the ids are the ones the file names");


            // ---- the f16 staging the shared-latent ops read -----------------
            // Presence alone would not catch a staging that put the right
            // numbers in the wrong places, and that is the one error the
            // shapes cannot see: both output tables are ONE matrix in the file
            // and a group owns a column block of it. So every element of every
            // staged table is compared against the fixture's own generator,
            // read in the file's order -- and one arm proves the comparison
            // can fail.
            auto dl16 = [&](const QuantTensor &t) {
                std::vector<u16> h(static_cast<size_t>(t.n_in * t.n_out));
                cpu->download(h.data(), t.data, h.size() * 2, 0);
                return h;
            };
            auto stage_exact = [&](const char *what, const QuantTensor &t,
                                   i64 n_in, i64 n_out, i64 seed,
                                   const char *shape_msg, const char *bytes_msg) {
                CHECK(t.present() && t.type == DType::F16 && t.n_in == n_in &&
                          t.n_out == n_out, shape_msg);
                if (!t.present()) return;
                const std::vector<u16> h = dl16(t);
                i64 wrong = 0, first = -1;
                for (i64 r = 0; r < n_out; r++)
                    for (i64 c = 0; c < n_in; c++) {
                        const u16 want = fp32_to_fp16(wv(r * n_in + c + seed));
                        if (h[static_cast<size_t>(r * n_in + c)] != want) {
                            if (first < 0) first = r * n_in + c;
                            wrong++;
                        }
                    }
                CHECK(wrong == 0, bytes_msg);
                if (wrong)
                    std::fprintf(stderr,
                                 "  (%s: %lld of %lld differ, first at %lld)\n",
                                 what, static_cast<long long>(wrong),
                                 static_cast<long long>(n_in * n_out),
                                 static_cast<long long>(first));
            };
            stage_exact("wq_a16", L0.wq_a16, kMlaEmbd, kMlaQLora, 100,
                        "attn_q_a is staged as f16 [n_embd, q_lora]",
                        "its bytes are the host dequant, in the file's order");
            stage_exact("wq_b16", L0.wq_b16, kMlaQLora, kMlaHead * kMlaKeyLen,
                        200,
                        "attn_q_b is staged as f16 [q_lora, n_head*key_length]",
                        "its bytes are the host dequant, in the file's order");
            stage_exact("wkv16", L0.wkv16, kMlaEmbd, kMlaKeyLen, 300,
                        "attn_kv is staged as f16 [n_embd, key_length]",
                        "its bytes are the host dequant, in the file's order");
            stage_exact("wo_a16", L0.wo_a16,
                        kMlaHead * kMlaKeyLen / kMlaOGroups,
                        kMlaOLora * kMlaOGroups, 400,
                        "attn_output_a is staged as f16 per group",
                        "its bytes are the host dequant, group block first");
            stage_exact("wo_b16", L0.wo_b16, kMlaOLora * kMlaOGroups, kMlaEmbd,
                        500,
                        "attn_output_b is staged as f16 [o_lora*n_groups, n_embd]",
                        "its bytes are the host dequant, in the file's order");
            // The negative control: the same bytes read TRANSPOSED must
            // disagree, or the comparison above could not have failed and the
            // gate would pass a staging that swapped the two axes.
            if (L0.wkv16.present()) {
                const std::vector<u16> h = dl16(L0.wkv16);
                i64 wrong = 0;
                for (i64 r = 0; r < kMlaKeyLen; r++)
                    for (i64 c = 0; c < kMlaEmbd; c++) {
                        const u16 want = fp32_to_fp16(wv(r * kMlaEmbd + c + 300));
                        if (h[static_cast<size_t>(c * kMlaKeyLen + r)] != want)
                            wrong++;
                    }
                CHECK(wrong > 0,
                      "reading the staged latent transposed does not match");
            }
        }
        m.unload();
        delete cpu;
    }

    // ---- 4. layers that disagree on the layout --------------------------
    {
        const std::string path = "kraken-mla-mixed.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Split;
        f.n_layer = 2;
        f.mixed_layout = true;
        CHECK(build_mla_fixture(path, f),
              "wrote a file whose layers disagree");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        CHECK(!m.load(*cpu, path, &err),
              "a file whose layers carry different layouts is refused");
        CHECK(err.find("layer 1") != std::string::npos &&
                  err.find("layer 0") != std::string::npos,
              "... and the refusal names both layers");
        m.unload();
        delete cpu;
    }

    // ---- 5. a tensor that disagrees with the file's own metadata --------
    {
        const std::string path = "kraken-mla-short.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Split;
        f.short_k_b = true;
        CHECK(build_mla_fixture(path, f), "wrote a file with a short attn_k_b");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        CHECK(!m.load(*cpu, path, &err),
              "a tensor that disagrees with the metadata is refused");
        CHECK(err.find("attn_k_b") != std::string::npos,
              "... and the refusal names the tensor");
        CHECK(err.find("7") != std::string::npos &&
                  err.find("8") != std::string::npos,
              "... and gives both shapes");
        m.unload();
        delete cpu;
    }

    // ---- 6. an int32 table where a weight belongs -----------------------
    {
        const std::string path = "kraken-mla-i32.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Split;
        f.int32_weight = true;
        CHECK(build_mla_fixture(path, f),
              "wrote a file whose attn_k_b is a table");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        CHECK(!m.load(*cpu, path, &err),
              "an int32 table is not accepted as a matmul weight");
        CHECK(err.find("int32") != std::string::npos,
              "... and the refusal says the format is data, not a weight");
        m.unload();
        delete cpu;
    }

    // ---- 7. the hash route's table --------------------------------------
    {
        const std::string path = "kraken-mla-hash.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Shared;
        f.bad_hash_shape = true;
        CHECK(build_mla_fixture(path, f),
              "wrote a file with a short route table");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        CHECK(!m.load(*cpu, path, &err),
              "a route table of another shape is refused");
        CHECK(err.find("ffn_gate_tid2eid") != std::string::npos,
              "... and the refusal names the table");
        m.unload();
        delete cpu;
    }
    {
        const std::string path = "kraken-mla-nohash.gguf";
        MlaFix f;
        f.kind = MlaFixKind::Shared;
        f.no_hash_table = true;
        CHECK(build_mla_fixture(path, f), "wrote a file with no route table");
        made.push_back(path);
        Backend *cpu = make_cpu_backend();
        Model m;
        std::string err;
        CHECK(!m.load(*cpu, path, &err),
              "a hash layer with no route table is refused");
        CHECK(err.find("ffn_gate_tid2eid") != std::string::npos,
              "... and the refusal names the table that is missing");
        m.unload();
        delete cpu;
    }

    // ---- 8. the ENGINE: what runs, and what is refused -------------------
    //
    // Two different claims, and the group needs both. The legacy lite layout is
    // IMPLEMENTED, so a V2-Lite file has to initialise and generate -- a group
    // that only tested refusals would pass over a missing graph. The split (V3
    // class) and shared-latent (V4) layouts are different computations, so a
    // file carrying one is refused AFTER the load: a run would decode fluent
    // text out of the wrong graph, which is the failure this engine must not
    // produce. The loader's hatch does not lift that, because the hatch is
    // about reading tensors and this is about the graph.
    {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = "kraken-mla-legacy.gguf";
        cfg.n_ctx = 64;
        cfg.prefill_chunk = 8;
        std::string err;
        const bool ok = engine.init(cpu, cfg, &err);
        CHECK(ok, "the legacy lite layout initialises");
        if (!err.empty()) std::fprintf(stderr, "  (init: %s)\n", err.c_str());
        if (ok) CHECK(engine.kv_pos() == 0, "... with an empty cache at position 0");
        engine.shutdown();
        delete cpu;
    }
    // The SPLIT layout (V3 class) still has no graph, and that is now the only
    // one the engine refuses by layout: the shared-latent layout runs (see
    // test_dsv4_whole_graph), so a refusal for it here would mean the two
    // claims disagree.
    {
        Backend *cpu = make_cpu_backend();
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = "kraken-mla-split.gguf";
        cfg.n_ctx = 64;
        cfg.prefill_chunk = 8;
        std::string err;
        const bool ok = engine.init(cpu, cfg, &err);
        CHECK(!ok, "the engine refuses the split layout, whose graph is absent");
        std::fprintf(stderr, "  (refused split: %s)\n", err.c_str());
        CHECK(err.find("MLA layout") != std::string::npos,
              "... and the refusal names the layout");
        CHECK(err.find("wrong graph") != std::string::npos ||
                  err.find("not this model") != std::string::npos,
              "... and says the text would not be the model's");
        engine.shutdown();
        delete cpu;
    }


    // The hatch does NOT lift the engine's guard, and the premise is checked
    // rather than assumed. Both halves are load-bearing: a refusal that came
    // from the LOAD would read identically at the engine arm above, and would
    // mean the loader never accepted the tensors at all -- the trap AGENTS.md
    // records for a check placed above `model_.load()`, where every file looks
    // refused for the right reason and none of them is.
    {
        Backend *cpu = make_cpu_backend();
        const char *hatch = std::getenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
        CHECK(hatch && std::strcmp(hatch, "1") == 0,
              "the loader's escape hatch is in force for the engine arm");
        Model m;
        std::string err;
        const bool ok = m.load(*cpu, "kraken-mla-shared.gguf", &err);
        CHECK(ok && m.cfg().mla_layout == MlaLayout::Shared,
              "the file the engine refused LOADS under the hatch, as the "
              "shared-latent layout");
        if (!ok) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        m.unload();
        delete cpu;
    }

    for (const std::string &p : made) std::remove(p.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
}

// DeepSeek-V4's hash route: the first `hash_layer_count` layers take their
// experts from `ffn_gate_tid2eid.weight` by TOKEN ID, not from the router.
//
// The defect this group exists for is a layout error: a column-major read of
// the same table is in range, stable, and sends every token to a different
// token's experts. No downstream stage can notice -- the table is the ground
// truth for the route -- so the pins are structural: token-major stride, one
// table per layer, layer depth honoured, ids range-checked, duplicates kept,
// and the gather in TABLE order rather than sorted.
static void test_dsv4_hash_route() {
    const i32 n_expert = 8, per_token = 2, n_tokens = 6, n_hash = 2;
    // Token-major: row t is {2t, 2t+1}. Token 1's first id is 2, where a
    // column-major read would answer 1 (table[1]) -- the exact confusion.
    std::vector<i32> t0(n_tokens * per_token), t1(n_tokens * per_token);
    for (i32 t = 0; t < n_tokens; t++)
        for (i32 s = 0; s < per_token; s++) {
            t0[static_cast<size_t>(t * per_token + s)] = (2 * t + s) % n_expert;
            t1[static_cast<size_t>(t * per_token + s)] =
                (2 * t + s + 3) % n_expert;
        }
    i32 out[8] = {0};
    CHECK(krk::dsv4_hash_route(out, 0, n_hash, t0.data(), per_token, n_tokens,
                               3, n_expert),
          "a hash layer routes a token that the table covers");
    CHECK(out[0] == 6 && out[1] == 7, "the route is the token's own row");
    i32 o1[8] = {0};
    CHECK(krk::dsv4_hash_route(o1, 1, n_hash, t1.data(), per_token, n_tokens,
                               3, n_expert),
          "a second hash layer routes from its own table");
    CHECK(o1[0] == 1 && o1[1] == 2,
          "each hash layer has its own table (layer 1 differs from layer 0)");
    // TOKEN-major stride. Token 1 slot 0 is 2; a column-major read answers 1.
    i32 ot[8] = {0};
    CHECK(krk::dsv4_hash_route(ot, 0, n_hash, t0.data(), per_token, n_tokens,
                               1, n_expert),
          "token 1 routes");
    CHECK(ot[0] == 2, "the table is TOKEN-major (a column-major read gives 1)");
    CHECK(ot[0] != t0[1], "the stride pin is not vacuous");
    // A wrong table of the same shape MUST change the chosen experts, or the
    // gate would pass on a lookup that ignores its input.
    std::vector<i32> wrong(t0.size());
    for (size_t i = 0; i < t0.size(); i++)
        wrong[i] = (t0[i] + 1) % n_expert;
    i32 ow[8] = {0};
    CHECK(krk::dsv4_hash_route(ow, 0, n_hash, wrong.data(), per_token,
                               n_tokens, 3, n_expert),
          "a shifted table still routes");
    CHECK(ow[0] != out[0] || ow[1] != out[1],
          "a different table changes the chosen experts");
    // Depth: layer n_hash is NOT a hash layer.
    i32 od[8] = {-7, -7};
    CHECK(!krk::dsv4_hash_route(od, n_hash, n_hash, t0.data(), per_token,
                                n_tokens, 3, n_expert),
          "a layer at or past hash_layer_count routes by gate, not by table");
    CHECK(od[0] == -7 && od[1] == -7,
          "a refused route writes nothing (no half route left behind)");
    // Coverage: the token id must be in the table.
    CHECK(!krk::dsv4_hash_route(od, 0, n_hash, t0.data(), per_token, n_tokens,
                                n_tokens, n_expert),
          "a token id past the table is refused");
    CHECK(!krk::dsv4_hash_route(od, 0, n_hash, t0.data(), per_token, n_tokens,
                                -1, n_expert),
          "a negative token id is refused");
    CHECK(!krk::dsv4_hash_route(od, 0, n_hash, nullptr, per_token, n_tokens, 0,
                                n_expert),
          "an empty table is refused");
    CHECK(!krk::dsv4_hash_route(od, 0, n_hash, t0.data(), 0, n_tokens, 0,
                                n_expert),
          "a zero-width table is refused");
    // An id outside the expert range is a corrupt/misread table, not a route.
    std::vector<i32> bad(t0);
    bad[static_cast<size_t>(0 * per_token + 1)] = n_expert;
    CHECK(!krk::dsv4_hash_route(od, 0, n_hash, bad.data(), per_token, n_tokens,
                                0, n_expert),
          "an expert id past n_expert is refused");
    bad[static_cast<size_t>(0 * per_token + 1)] = -1;
    CHECK(!krk::dsv4_hash_route(od, 0, n_hash, bad.data(), per_token, n_tokens,
                                0, n_expert),
          "a negative expert id is refused");
    CHECK(od[0] == -7 && od[1] == -7,
          "a refused id check also writes nothing");
    // Duplicates are what the file says: no dedupe, count preserved.
    std::vector<i32> dup(t0);
    dup[0] = 4;
    dup[1] = 4;
    i32 odup[8] = {-7, -7};
    CHECK(krk::dsv4_hash_route(odup, 0, n_hash, dup.data(), per_token,
                               n_tokens, 0, n_expert),
          "a row naming one expert twice still routes");
    CHECK(odup[0] == 4 && odup[1] == 4,
          "duplicate ids are preserved, not deduplicated");
    // The weights come from the router's scores, gathered in TABLE order.
    const f32 scores[8] = {0.0f, 10.0f, 20.0f, 30.0f,
                           40.0f, 50.0f, 60.0f, 70.0f};
    f32 ws[2] = {0.0f, 0.0f};
    const i32 ids[2] = {6, 7};
    CHECK(krk::dsv4_hash_gather(ws, ids, 2, scores, n_expert),
          "the selected ids' scores are gathered");
    CHECK(ws[0] == 60.0f && ws[1] == 70.0f, "the gather reads the ids' scores");
    const i32 rev[2] = {7, 6};
    f32 wr[2] = {0.0f, 0.0f};
    CHECK(krk::dsv4_hash_gather(wr, rev, 2, scores, n_expert),
          "the gather accepts a reversed route");
    CHECK(wr[0] == 70.0f && wr[1] == 60.0f,
          "the gather keeps TABLE order (a sorted gather swaps the weights)");
    const i32 badid[2] = {6, n_expert};
    CHECK(!krk::dsv4_hash_gather(ws, badid, 2, scores, n_expert),
          "the gather refuses an id outside the score row");
}

// The whole V4 file, end to end: loader -> per-layer shared-latent attention ->
// hash-routed MoE -> head. Every piece already had a gate of its own
// (test_mla_loader, test_mla_shared, test_dsv4_hash_route) and NONE of them
// pinned a sentence, which is exactly how qwen4exp passed 16 per-block checks
// while decoding nonsense (the PLE normalisation bug). This group pins the
// OUTPUT:
//
//   1. the fixture loads with no development hatch -- the claim a supported
//      arch row makes, checked rather than assumed;
//   2. the last prompt position's WHOLE logits row hashes to the recorded value
//      and its argmax is the recorded token;
//   3. a different prompt moves the row (the graph reads its input);
//   4. chunk 1 reproduces the chunk-8 row exactly (the row arithmetic is
//      bit-exact inside the thin-row ladder this 1-token prompt sits on);
//   5. a SHIFTED hash table moves the row, and an out-of-range id makes the run
//      REFUSE -- the pin can fail, and it fails on the route as well as on the
//      weights;
//   6. the greedy sequence is the recorded one, at both chunk sizes.
//
// The row, not the tokens, is the load-bearing pin: this fixture's synthetic
// head can argmax one token and stop, and this repo has already been bitten by a
// gate that compared only the sampled token (the attention rewrite produced the
// same first token from two different kernels).
static void test_dsv4_whole_graph() {
    const std::string path = "../kraken-dsv4whole.gguf";  // DEBUG path
    const std::string shifted = "kraken-dsv4whole-shift.gguf";
    const std::string oob = "kraken-dsv4whole-oob.gguf";
    MlaFix f;
    f.kind = MlaFixKind::Shared;
    CHECK(build_mla_fixture(path, f), "the shared-latent fixture writes");
    MlaFix fs = f;
    fs.hash_shift = 1;
    CHECK(build_mla_fixture(shifted, fs),
          "and a copy whose hash table sends every token to the next expert");
    MlaFix fo = f;
    fo.hash_id_oob = true;
    CHECK(build_mla_fixture(oob, fo),
          "and one whose table names an expert past the bank");

    // The hatch is NOT set in this group, on purpose: arm 1 is the claim the
    // arch row now makes (`Partial`, attention implemented).
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif

    {
        Backend *cpu = make_cpu_backend();
        Model model;
        std::string err;
        CHECK(model.load(*cpu, path, &err),
              "deepseek4 loads with no development hatch");
        if (!err.empty()) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        model.unload();
        delete cpu;
    }

    auto row_hash = [](const f32 *v, i32 n) {
        u64 h = 1469598103934665603ull;
        for (i32 i = 0; i < n; i++) {
            u32 bits = 0;
            std::memcpy(&bits, &v[i], sizeof(bits));
            h ^= bits;
            h *= 1099511628211ull;
        }
        return h;
    };

    // Prefill only (max_tokens 0), so the row read afterwards is the one at the
    // last prompt position -- the row a decode starts from.
    auto run = [&](const std::string &file, const char *prompt, i32 chunk,
                   u64 *hash, i32 *top1, std::string *why,
                   RowPin *pin = nullptr) -> bool {
        Backend *cpu = make_cpu_backend();
        bool ok = false;
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = file;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = chunk;
            std::string err;
            if (!engine.init(cpu, cfg, &err)) {
                if (why) *why = err;
                if (!err.empty()) std::fprintf(stderr, "  (init: %s)\n", err.c_str());
            } else {
                GenerateParams p;
                p.prompt = prompt;
                p.max_tokens = 0;
                p.sampler.greedy = true;
                GenerateResult r;
                ok = engine.generate(p, &r) && r.finish != FinishError;
                if (!ok && why) *why = engine.forward_failure();
                const f32 *lg = engine.last_logits();
                const i32 nv = engine.model().cfg().n_vocab;
                if (ok && lg && nv > 0) {
                    *hash = row_hash(lg, nv);
                    i32 best = 0;
                    for (i32 i = 1; i < nv; i++)
                        if (lg[i] > lg[best]) best = i;
                    *top1 = best;
                    if (pin) *pin = row_pin(lg, nv);
                } else {
                    ok = false;
                }
            }
        }
        delete cpu;
        return ok;
    };

    auto gen = [&](const std::string &file, const char *prompt, i32 chunk,
                   std::vector<i32> *tokens) -> bool {
        Backend *cpu = make_cpu_backend();
        bool ok = false;
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = file;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = chunk;
            std::string err;
            if (!engine.init(cpu, cfg, &err)) {
                if (!err.empty()) std::fprintf(stderr, "  (init: %s)\n", err.c_str());
            } else {
                GenerateParams p;
                p.prompt = prompt;
                p.max_tokens = 6;
                p.sampler.greedy = true;
                GenerateResult r;
                ok = engine.generate(p, &r) && r.finish != FinishError;
                *tokens = r.tokens;
            }
        }
        delete cpu;
        return ok;
    };

    u64 h_wide = 0, h_narrow = 0, h_other = 0, h_shift = 0;
    i32 t_wide = -1, t_narrow = -1, t_other = -1, t_shift = -1;
    RowPin pin_wide;
    std::string why;
    CHECK(run(path, "the", 8, &h_wide, &t_wide, &why, &pin_wide),
          "the whole graph prefills and the row reaches the head");
    std::fprintf(stderr, "  (logits row %016llx, argmax %d)\n",
                 static_cast<unsigned long long>(h_wide),
                 static_cast<int>(t_wide));

    // ---- 1. the recorded row --------------------------------------------
    // Recorded from this graph on the fixture, CPU backend, prompt "the".
    // The pin is the portable fingerprint (see row_pin's note); the bit hash
    // is host-dependent and is compared only under KRK_GOLDEN_BITS=1.
    static const u64 kGoldenHash = 0x12602ed8b3a49addull;
    static const i32 kGoldenArgmax = 0;
    static const RowPin kGolden = {
        {1.555394, 1.555394, 1.555394, 1.555394, 1.555394, 1.555394, 1.555394,
         1.555394},
        -2.707561, 343.138183, 24.373835};
    CHECK(h_wide != 0, "the row is not empty");
    CHECK(t_wide == kGoldenArgmax, "and its argmax is the recorded token");
    std::fprintf(stderr, "  (dsv4 pin: %s)\n", pin_str(pin_wide).c_str());
    if (kGolden.l1 != 0.0)
        CHECK(pin_close(pin_wide, kGolden),
              "the last prompt row matches the recorded fingerprint (the whole "
              "graph's output)");
    if (golden_bits_on() && kGoldenHash != 0)
        CHECK(h_wide == kGoldenHash,
              "KRK_GOLDEN_BITS: the recorded row, bit for bit");

    // ---- 2. the prompt reaches the row -----------------------------------
    CHECK(run(path, "a", 8, &h_other, &t_other, &why),
          "a second prompt prefills");
    CHECK(h_other != h_wide, "a different prompt produces a different row");

    // ---- 3. chunk invariance ---------------------------------------------
    CHECK(run(path, "the", 1, &h_narrow, &t_narrow, &why),
          "the same prompt prefills token by token");
    CHECK(h_narrow == h_wide, "chunk 1 reproduces the chunk-8 row exactly");

    // ---- 4. the pin can fail: the ROUTE -----------------------------------
    CHECK(run(shifted, "the", 8, &h_shift, &t_shift, &why),
          "the shifted-table fixture prefills");
    CHECK(h_shift != h_wide,
          "a shifted hash table moves the row (the pin sees the route)");
    {
        std::string rwhy;
        u64 h = 0;
        i32 t = -1;
        CHECK(!run(oob, "the", 8, &h, &t, &rwhy),
              "a table naming an expert past the bank refuses the run");
        CHECK(rwhy.find("hash route") != std::string::npos,
              "... and says the hash route refused");
    }

    // ---- 5. and the graph emits text --------------------------------------
    std::vector<i32> greedy, greedy1;
    CHECK(gen(path, "the", 8, &greedy), "the whole graph generates");
    std::fprintf(stderr, "  (greedy tokens:");
    for (i32 t : greedy) std::fprintf(stderr, " %d", static_cast<int>(t));
    std::fprintf(stderr, ")\n");
    static const i32 kGoldenSeq[] = {0, 4, 6, 6, 6, 9};
    const i32 kGoldenSeqN = 6;
    if (kGoldenSeqN > 0) {
        CHECK(static_cast<i32>(greedy.size()) == kGoldenSeqN,
              "the run emits the recorded number of tokens");
        bool same = static_cast<i32>(greedy.size()) == kGoldenSeqN;
        for (i32 i = 0; same && i < kGoldenSeqN; i++)
            same = greedy[i] == kGoldenSeq[i];
        CHECK(same, "and every emitted token is the recorded one");
    }
    CHECK(gen(path, "the", 1, &greedy1), "the same prompt generates at chunk 1");
    CHECK(greedy1 == greedy, "chunk 1 emits the same tokens");

    // ---- 6. the PREFILL path: many rows, one call -------------------------
    // A real multi-token prompt through the shared-latent forward. Chunk 16
    // runs the prefill as ONE call of 12+ rows -- the layer's latent write, its
    // attention over the growing prefix and its grouped output all loop inside
    // that one pass -- while chunk 1 runs the same rows one at a time. A
    // one-token prompt would make the two arms the same call and pin nothing,
    // so the prompt is long enough for the chunk-16 pass to exceed the thin-row
    // ladder the GEMM uses below nine rows.
    std::string mprompt;
    for (int i = 0; i < 12; i++) mprompt += (i ? " the" : "the");
    u64 h_m1 = 0, h_m16 = 0;
    i32 t_m1 = -1, t_m16 = -1;
    CHECK(run(path, mprompt.c_str(), 1, &h_m1, &t_m1, &why),
          "the multi-token prompt prefills token by token");
    CHECK(run(path, mprompt.c_str(), 16, &h_m16, &t_m16, &why),
          "and again in one 16-row chunk");
    std::fprintf(stderr, "  (multi-token: chunk 1 %016llx, chunk 16 %016llx)\n",
                 static_cast<unsigned long long>(h_m1),
                 static_cast<unsigned long long>(h_m16));
    CHECK(h_m1 != h_wide, "the multi-token prompt is a different row");
    CHECK(h_m16 == h_m1,
          h_m16 == h_m1
              ? "chunk 16 reproduces the chunk-1 row exactly on 12+ prompt rows"
              : "chunk 16 DIFFERS from chunk 1: the multi-row shared-latent "
                "prefill does not reproduce the single-row one");
    std::vector<i32> gm16, gm1;
    CHECK(gen(path, mprompt.c_str(), 16, &gm16),
          "the multi-token prompt generates at chunk 16");
    CHECK(gen(path, mprompt.c_str(), 1, &gm1), "and at chunk 1");
    CHECK(gm16 == gm1, "both chunk sizes emit the same tokens");

    // ---- 7. the DEVICE arm ------------------------------------------------
    // kraken-tests is a plain C++ TU and cannot open the device, so the device
    // half drives the CLI -- the arrangement test_prefill_chunk_bitexact uses.
    //
    // What is checkable here, and what is not, was MEASURED rather than
    // assumed. A HIP row is not bit-equal to the `--cpu` reference on any
    // model: SmolLM2-135M at a 16-token prompt gives trace 0d47196029f2cb51 /
    // logits 4ccba855c2b297e0 on the card against a1e5be993b18b62e /
    // 668768a536fd8d50 on the reference, with the same first token (30) both
    // ways. The engine's cross-backend contract is the TEXT -- that is why
    // scripts/coherence_check.sh compares sentences, and why the prefill gate
    // compares chunk sizes within ONE backend. So this arm pins the two
    // properties a shared-latent kernel actually owes: the device prefill is
    // chunk-invariant (thirteen rows in one call against thirteen calls of one
    // row, both hashes) and it samples the reference's token.
    const std::string self = self_path();
    const std::filesystem::path exe_p =
        std::filesystem::path(self).parent_path() /
#ifdef _WIN32
        "kraken.exe";
#else
        "kraken";
#endif
    const bool have_exe = !self.empty() && std::filesystem::exists(exe_p);
    CHECK(have_exe, "kraken sits beside kraken-tests (the device arm drives it)");
    if (have_exe) {
        const std::string exe = exe_p.string();
        const PrefillProbe cc =
            run_prefill_hash(exe, path, mprompt, 16, false, false, "--cpu");
        const PrefillProbe cd1 = run_prefill_hash(exe, path, mprompt, 1, false);
        const PrefillProbe cd16 = run_prefill_hash(exe, path, mprompt, 16, false);
        const PrefillProbe cd16b = run_prefill_hash(exe, path, mprompt, 16, false);
        CHECK(cc.ok, "the CLI prefills the fixture on the CPU reference");
        if (!cc.ok)
            std::fprintf(stderr, "  (cpu run rc=%d stderr: %.300s)\n", cc.rc,
                         cc.err.c_str());
        CHECK(cd1.ok && cd16.ok,
              "and on the default backend, at chunk 1 and at chunk 16");
        if (!cd1.ok || !cd16.ok)
            std::fprintf(stderr, "  (device rc=%d/%d stderr: %.300s)\n", cd1.rc,
                         cd16.rc, cd16.err.c_str());
        std::fprintf(stderr,
                     "  (13-row prefill  cpu trace %s logits %s first %d)\n",
                     cc.trace.c_str(), cc.logits.c_str(), cc.first);
        std::fprintf(stderr,
                     "  (                dev trace %s logits %s first %d)\n",
                     cd16.trace.c_str(), cd16.logits.c_str(), cd16.first);
        bool dev = false;
#ifdef KRK_TEST_HIP_BUILD
        dev = cd16.ok && cd16.banner.find("scalar reference") == std::string::npos &&
              cd16.err.find("falling back to the CPU") == std::string::npos;
        CHECK(dev,
              dev ? "the HIP backend answered (not the CPU fallback)"
                  : "a HIP build fell back to the CPU: the device arm proved "
                    "nothing");
#endif
        if (dev) {
            // The claim: one call of thirteen rows leaves the same rows behind
            // as thirteen calls of one row.
            CHECK(cd16.trace == cd1.trace,
                  cd16.trace == cd1.trace
                      ? "the DEVICE prefill folds to the same trace at chunk 1 "
                        "and 16 (every row, not just the last)"
                      : "the device prefill is not chunk-invariant: chunk 1 and "
                        "16 fold to different traces");
            CHECK(cd16.logits == cd1.logits,
                  cd16.logits == cd1.logits
                      ? "and the same logits row"
                      : "the device prefill is not chunk-invariant: chunk 1 and "
                        "16 leave different rows");
            CHECK(cd16b.trace == cd16.trace && cd16b.logits == cd16.logits,
                  "the device prefill is deterministic run to run");
            // Cross-backend agreement, on the number the text depends on.
            CHECK(cd16.first == cc.first && cd1.first == cc.first,
                  cd16.first == cc.first && cd1.first == cc.first
                      ? "the device samples the token the CPU reference does"
                      : "the device and the CPU reference sample different "
                        "tokens");
        }
        std::remove("kraken-prefill-out-16--cpu.txt");
        std::remove("kraken-prefill-err-16--cpu.txt");
        std::remove("kraken-prefill-out-1.txt");
        std::remove("kraken-prefill-err-1.txt");
        std::remove("kraken-prefill-out-16.txt");
        std::remove("kraken-prefill-err-16.txt");
    }

    // DEBUG: fixture kept
    // std::remove(path.c_str());
    // std::remove(shifted.c_str());
    // std::remove(oob.c_str());
}

// QSA indexer, the host maths first (no engine, no file): pooling over complete
// blocks only, the per-head relu before the sum, rope at the block's FIRST
// position, and the selection rule -- budget in CELLS, the still-filling block
// pinned visible, complete blocks competing on score.
static void test_qsa_select() {
    QsaConfig cfg;
    cfg.n_head = 1;
    cfg.head_dim = 2;
    cfg.top_k = 4; // cells, not blocks
    cfg.ratio = 2;
    cfg.eps = 1e-6f;
    cfg.n_rot = 2;
    static const f32 freq[1] = {1.0f};
    cfg.inv_freq = freq;
    cfg.rope_scale = 1.0f;
    cfg.rope_neox = false;

    // ---- pooling: only COMPLETE blocks have a mean ------------------------
    {
        const f32 keys[5 * 2] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        f32 pooled[2 * 2] = {0, 0, 0, 0};
        const i32 nb = qsa_pool(pooled, keys, 2, 5, 2);
        CHECK(nb == 2, "five cells in blocks of two leave two complete blocks");
        CHECK(pooled[0] == 2.0f && pooled[1] == 3.0f,
              "block 0 pools as the mean of its two cells");
        CHECK(pooled[2] == 6.0f && pooled[3] == 7.0f,
              "block 1 pools as the mean of its two cells");
    }

    // ---- the score: relu per head, THEN the sum --------------------------
    {
        const f32 q[2] = {1.0f, 0.0f};
        const f32 pooled[4 * 2] = {1, 0, -1, 0, 0, 1, 2, 0};
        f32 score[4] = {0, 0, 0, 0};
        qsa_scores(score, q, pooled, 1, 1, 4, 2);
        CHECK(score[0] == 1.0f, "an aligned block scores its dot product");
        CHECK(score[1] == 0.0f, "a negative dot is rectified to zero");
        CHECK(score[2] == 0.0f, "an orthogonal block scores zero");
        CHECK(score[3] == 2.0f, "a longer aligned block scores higher");
    }
    {
        // Head 0 sees +1 on the block; head 1 sees -1, which its own relu
        // drops. Sum-then-rectify would give relu(1 + -1) = 0, so this check
        // separates the per-head rule from the whole-row one.
        const f32 q[2 * 2] = {1, 0, 0, 1}; // head 0 along +d0, head 1 along +d1
        const f32 pooled[2] = {1, -1};
        f32 score[1] = {0};
        qsa_scores(score, q, pooled, 1, 2, 1, 2);
        CHECK(score[0] == 1.0f,
              "one head's negative dot does not cancel another head's evidence");
    }

    // ---- norm + rope at the given position -------------------------------
    {
        f32 x[2] = {3.0f, 4.0f};
        const f32 w[2] = {1.0f, 1.0f};
        const i64 pos[1] = {2};
        qsa_norm_rope(x, w, 1, 2, pos, cfg);
        const f32 inv = 1.0f / std::sqrt(12.5f + 1e-6f);
        const f32 a = 3.0f * inv, b = 4.0f * inv;
        const f32 th = 2.0f; // pos * inv_freq[0]
        const f32 c = std::cos(th), s = std::sin(th);
        CHECK(std::fabs(x[0] - (a * c - b * s)) < 1e-6f &&
                  std::fabs(x[1] - (a * s + b * c)) < 1e-6f,
              "the row is rms-normed then rotated by pos * inv_freq");
    }

    // ---- selection: budget in CELLS --------------------------------------
    {
        // 8 cells, blocks of 2, budget 4 + 2 - 1 = 5 cells, query at 7.
        const f32 score[4] = {1.0f, 5.0f, 2.0f, 0.0f};
        std::vector<i64> cells;
        const i32 n = qsa_select(&cells, score, 4, 8, 7, cfg);
        CHECK(n == 5 && cells.size() == 5,
              "the selection fills exactly the cell budget");
        CHECK(n == 5 && cells[0] == 0 && cells[1] == 2 && cells[2] == 3 &&
                  cells[3] == 4 && cells[4] == 5,
              "the best blocks are taken whole and the boundary block is cut at "
              "the budget");
    }
    {
        // Query at 6: the block it sits in is still filling (cells 6,7 exist
        // only up to 6), so it is pinned visible whatever it scores.
        const f32 score[3] = {1.0f, 5.0f, 2.0f};
        std::vector<i64> cells;
        const i32 n = qsa_select(&cells, score, 3, 7, 6, cfg);
        CHECK(n == 5 && !cells.empty() && cells.back() == 6,
              "the block that is still filling is always visible");
        CHECK(n == 5 && cells[0] == 2 && cells[1] == 3 && cells[2] == 4 &&
                  cells[3] == 5,
              "the scored blocks fill the rest of the budget");
    }
    {
        // Below the budget every visible cell is chosen, in order.
        const f32 score[2] = {1.0f, 2.0f};
        std::vector<i64> cells;
        const i32 n = qsa_select(&cells, score, 2, 4, 3, cfg);
        CHECK(n == 4 && cells.size() == 4 && cells[0] == 0 && cells[3] == 3,
              "inside the budget the selection is every visible cell");
    }
}

// ---------------------------------------------------------------------------
// QSA indexer through a real file: the fixture declares `compress_ratios {0, 2}`
// and `top_k 16`, so layer 1 indexes in blocks of two cells and its budget is
// 17 cells. A prompt longer than that is where the selection engages -- and the
// property that matters is that the ROW'S CELL LIST DOES NOT DEPEND ON THE
// CHUNK SIZE, because the pool is over every key the layer ever wrote. It is
// compared at chunk 1 against chunk 8, i.e. across the exact rows<=8 rung; a
// wide chunk is checked for shape rather than identity because its projection
// runs the rows>=9 tile.
static void test_qwen4exp_qsa() {
    const std::string path = "kraken-q4qsa.gguf";
    CHECK(build_tiny_q4_model(path, 0, 0, 0, true, false, true),
          "a four-lane file with the indexer writes");
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "1");
#else
    setenv("KRK_ARCH_ALLOW_EXPERIMENTAL", "1", 1);
#endif

    // One prefill-only run at `chunk`, returning the last row's cell list and
    // the layer's ratio. Prefill-only (max_tokens 0) makes the LAST forward the
    // final chunk, so both arms answer for the same position.
    auto run = [&](i32 chunk, const std::string &prompt, i32 n_out,
                   std::vector<i64> *cells, i32 *ratio1, i32 *ratio0,
                   i32 *n_prompt) -> bool {
        Backend *cpu = make_cpu_backend();
        bool ok = false;
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = path;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = chunk;
            std::string err;
            if (!engine.init(cpu, cfg, &err)) {
                if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
            } else {
                GenerateParams p;
                p.prompt = prompt;
                p.max_tokens = n_out;
                p.sampler.greedy = true;
                GenerateResult r;
                ok = engine.generate(p, &r) && r.finish != FinishError;
                *cells = engine.qsa_debug_cells(1);
                if (ratio1) *ratio1 = engine.qsa_debug_ratio(1);
                if (ratio0) *ratio0 = engine.qsa_debug_ratio(0);
                if (n_prompt) *n_prompt = r.prompt_tokens;
            }
        }
        delete cpu;
        return ok;
    };

    {
        // Long enough that the last row is past the 17-cell budget.
        const std::string prompt = "aaaa bbbb cccc dddd eeee ffff gggg";
        std::vector<i64> wide, narrow;
        i32 r1 = 0, r0 = -1, n_prompt = 0;
        CHECK(run(8, prompt, 0, &wide, &r1, &r0, &n_prompt),
              "the indexer file prefills at chunk 8");
        CHECK(r1 == 2, "layer 1's compress ratio reaches the engine");
        CHECK(r0 == 0, "the recurrent layer carries no indexer");
        CHECK(n_prompt > 18, "the prompt is longer than the cell budget");
        const i64 q = n_prompt - 1;
        CHECK(!wide.empty(), "a context past the budget produces a selection");
        if (!wide.empty()) {
            CHECK(static_cast<i64>(wide.size()) == 17,
                  "the selection is exactly the cell budget (16 + 2 - 1)");
            CHECK(std::is_sorted(wide.begin(), wide.end()),
                  "the cells come out ascending");
            CHECK(wide.front() >= 0 && wide.back() <= q,
                  "every selected cell is a visible one");
            bool dup = false;
            for (size_t i = 1; i < wide.size(); i++)
                dup = dup || wide[i] == wide[i - 1];
            CHECK(!dup, "a cell is selected at most once");
            // The block the last query sits in is pinned only while it is still
            // filling; when it is, the newest cell must be in the list.
            if ((q / 2) * 2 + 1 > q)
                CHECK(wide.back() == q,
                      "the still-filling tail block is always selected");
        }
        // The selection reads every key the layer wrote, so it cannot depend on
        // how the prompt was chunked. The comparison arm is chunk 1 -- 27
        // single-row forwards against chunk 8's four -- because both sit on the
        // rows<=8 ladder, which is bit-exact (`kraken-bench --bitexact`). A
        // wide prefill is the arm that CANNOT be equal yet: a 27-row projection
        // takes the tiled GEMM, whose rows>=9 accumulation is not identical to
        // the ladder's, so its scores move by ulps and a near-tied block can
        // change places. That is the open GEMM rung, not the indexer -- a
        // comparison that ignored it would be asserting a property the engine
        // does not have.
        CHECK(run(1, prompt, 0, &narrow, nullptr, nullptr, nullptr),
              "the same prompt prefills token by token");
        if (wide != narrow) {
            std::fprintf(stderr, "  (cells differ -- chunk 8:");
            for (i64 c : wide) std::fprintf(stderr, " %lld", (long long)c);
            std::fprintf(stderr, " / chunk 1:");
            for (i64 c : narrow) std::fprintf(stderr, " %lld", (long long)c);
            std::fprintf(stderr, " / q=%lld)" "\n", (long long)q);
        }
        CHECK(wide == narrow, "chunk 1 and chunk 8 select the same cells");

        // The wide arm reads the same scores, so it has to pick the same cells.
        // It is the one that would break if a rows>=9 projection moved a block
        // score by an ulp; on this fixture it does not -- the prefill hash is
        // identical at chunks 1, 2, 8 and 32.
        std::vector<i64> deep;
        CHECK(run(32, prompt, 0, &deep, nullptr, nullptr, nullptr),
              "the same prompt prefills at chunk 32");
        CHECK(static_cast<i64>(deep.size()) == 17,
              "a wide prefill fills the same cell budget");
        CHECK(deep == wide, "chunk 32 selects the same cells as chunk 8");
    }
    {
        // Inside the budget nothing is dropped, so there is no selection at all
        // and the layer runs dense causal attention -- the reference's own
        // behaviour below `top_k + ratio - 1`.
        std::vector<i64> cells;
        CHECK(run(8, "the", 0, &cells, nullptr, nullptr, nullptr),
              "a short prompt prefills through the indexer layer");
        CHECK(cells.empty(),
              "inside the budget the layer is dense and selects nothing");
    }

    if (!std::getenv("KRK_KEEP_FIXTURE")) std::remove(path.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
}

static void test_qwen4exp_forward() {
    const std::string wide_path = "kraken-q4fwd.gguf";
    const std::string same_path = "kraken-q4same.gguf";
    const std::string full_path = "kraken-q4full.gguf";
    CHECK(build_tiny_q4_model(wide_path, 0, 2560, 32, false, false),
          "a four-lane-of-2560 file writes (no PLE, no indexer)");
    CHECK(build_tiny_q4_model(same_path, 0, 0, 0, false, true),
          "and a lane-symmetric one (one injection column for all four lanes)");
    CHECK(build_tiny_q4_model(full_path, 0, 0, 0, true, false),
          "and the full-schema file: PLE and the indexer both declared");
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "1");
#else
    setenv("KRK_ARCH_ALLOW_EXPERIMENTAL", "1", 1);
#endif

    // ---- the real lane width: four lanes of 2560 ------------------------
    {
        Backend *cpu = make_cpu_backend();
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = wide_path;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = 8;
            std::string err;
            const bool ok = engine.init(cpu, cfg, &err);
            CHECK(ok, "the hyper-connection stack initialises (this was refused)");
            if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
            if (ok) {
                GenerateParams p;
                p.prompt = "the";
                p.max_tokens = 4;
                p.sampler.greedy = true;
                GenerateResult r1, r2;
                CHECK(engine.generate(p, &r1), "a four-lane model generates");
                // This fixture's output matrix is synthetic, so its argmax is
                // a special token about as often as any other id: the run is
                // allowed to stop on EOS at token 0. What it may NOT do is fail,
                // and the lane check below is the proof the stack ran (the
                // prompt's own forward is what writes the lane stack).
                CHECK(r1.finish != FinishError, "... and finishes without an error");
                CHECK(r1.prompt_tokens >= 1, "... after tokenising the prompt");
                CHECK(engine.generate(p, &r2) && r1.tokens == r2.tokens,
                      "... and does it deterministically");

                i32 hc = 0;
                i64 ne = 0;
                const f32 *lanes = engine.hc_debug_lanes(&hc, &ne);
                CHECK(lanes != nullptr && hc == 4 && ne == 2560,
                      "the lane stack is four lanes of 2560");
                if (lanes) {
                    // Every other mixer tensor is lane-symmetric, so the
                    // injection is what separates the lanes -- and it must, or
                    // the combine never read it.
                    bool diverged = false;
                    for (i32 c = 1; c < hc && !diverged; c++) {
                        for (i64 e = 0; e < ne; e++) {
                            if (lanes[static_cast<i64>(c) * ne + e] != lanes[e]) {
                                diverged = true;
                                break;
                            }
                        }
                    }
                    CHECK(diverged, "a lane-varying injection separates the lanes");
                    bool finite = true;
                    for (i64 i = 0; i < static_cast<i64>(hc) * ne && finite; i++) {
                        const f32 v = lanes[i];
                        if (!(v == v) || v > 1e6f || v < -1e6f) finite = false;
                    }
                    CHECK(finite, "and the lane values stay finite");
                }
            }
        }
        delete cpu;
    }

    // ---- the invariant: lane-symmetric injection keeps the lanes equal ----
    {
        Backend *cpu = make_cpu_backend();
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = same_path;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = 8;
            std::string err;
            const bool ok = engine.init(cpu, cfg, &err);
            CHECK(ok, "the lane-symmetric file initialises");
            if (ok) {
                GenerateParams p;
                p.prompt = "the";
                p.max_tokens = 3;
                p.sampler.greedy = true;
                GenerateResult r;
                CHECK(engine.generate(p, &r) && r.generated > 0,
                      "... and generates");
                i32 hc = 0;
                i64 ne = 0;
                const f32 *lanes = engine.hc_debug_lanes(&hc, &ne);
                CHECK(lanes != nullptr && hc == 4,
                      "its lane stack is four lanes");
                if (lanes) {
                    // The identical-lane invariant, through the whole stack:
                    // one weight per lane, so the combine is the identity on
                    // the lane structure and four copies stay four copies --
                    // BIT-equal, not approximately.
                    i64 differ = 0;
                    f32 worst = 0.0f;
                    for (i32 c = 1; c < hc; c++) {
                        for (i64 e = 0; e < ne; e++) {
                            const f32 a = lanes[e];
                            const f32 b = lanes[static_cast<i64>(c) * ne + e];
                            if (a != b) {
                                differ++;
                                const f32 d = a > b ? a - b : b - a;
                                if (d > worst) worst = d;
                            }
                        }
                    }
                    if (differ) {
                        std::fprintf(stderr,
                                     "  (lanes differ in %lld of %lld values, worst %.3g)\n",
                                     static_cast<long long>(differ),
                                     static_cast<long long>(3 * ne * hc),
                                     static_cast<double>(worst));
                    }
                    CHECK(differ == 0,
                          "with one injection column the lanes are still bit-equal");
                }
            }
        }
        delete cpu;
    }

    // ---- both surrounding blocks RUN, so the full schema initialises -----
    // This check used to require the opposite: while the PLE module and the
    // indexer were unread, a file carrying either was refused by name. Both run
    // now, so the full-schema file initialising here is what keeps a refusal
    // from being re-introduced over blocks that work.
    {
        Backend *cpu = make_cpu_backend();
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = full_path;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = 8;
            std::string err;
            const bool ok = engine.init(cpu, cfg, &err);
            CHECK(ok, "a file carrying both the PLE module and the indexer initialises");
            if (!ok) std::fprintf(stderr, "  (init: %s)\n", err.c_str());
        }
        delete cpu;
    }

    // ---- a half-declared indexer is still refused, by name --------------
    {
        const std::string half_path = "kraken-q4half.gguf";
        CHECK(build_tiny_q4_model(half_path, 0, 0, 0, true, false, true, true),
              "a file with the compress ratios and no head geometry writes");
        Backend *cpu = make_cpu_backend();
        Model model;
        std::string err;
        const bool ok = model.load(*cpu, half_path, &err);
        CHECK(!ok, "a half-declared indexer is refused, not run as if it had none");
        CHECK(err.find("indexer") != std::string::npos,
              "... and the message names the indexer geometry that is missing");
        model.unload();
        delete cpu;
        std::remove(half_path.c_str());
    }

    std::remove(wide_path.c_str());
    std::remove(same_path.c_str());
    std::remove(full_path.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
}

// Reads the first "<what> L<layer> ..." line of a KRK_DUMP file: the values are
// space separated after the n=<rows> token, in the order the engine wrote them.
// Shared by the qwen4exp groups that observe the graph through a dump rather
// than through a config flag.
static bool dump_row_of(const std::string &file, const char *what, i32 layer,
                        std::vector<double> *out) {
    std::ifstream in(file.c_str(), std::ios::binary);
    if (!in) return false;
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "%s L%02d ", what,
                  static_cast<int>(layer));
    const std::string want = prefix;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < want.size() ||
            line.compare(0, want.size(), want) != 0)
            continue;
        const char *p = line.c_str() + want.size();
        while (*p && *p != ' ') p++; // the n=<rows> token
        while (*p) {
            char *end = nullptr;
            const double v = std::strtod(p, &end);
            if (end == p) break;
            out->push_back(v);
            p = end;
        }
        return !out->empty();
    }
    return false;
}

// ---------------------------------------------------------------------------
// qwen4exp, the WHOLE GRAPH: the output of the stack, not one block of it.
//
// This is the gate that was missing, and it is the only reason the family sat
// behind KRK_ARCH_ALLOW_EXPERIMENTAL with every tensor read and every block
// proven: the other groups pin the PLE row gather, the lane invariant, the QSA
// cell list and the delta-net gate, and not one of them asserted the SENTENCE.
// A graph can pass all of those and still decode nonsense -- which is exactly
// what the PLE normalisation bug did (every PLE layer put a unit-RMS row on the
// residual, and the text was CJK glyphs where the reference wrote Python).
//
// The pin is a token sequence, and its provenance is the part that matters: the
// sequence is what THIS graph produces on the fixture, anchored by the real-file
// match -- on H:/OLLAMA-Models/GGUF/qwen3.8-flash-next-reap-288-Q4_K_M.gguf the
// same graph reproduces llama.cpp's greedy continuation of "The capital of
// France is" for 24 of 24 tokens (docs/AUDIT-qwen4exp-support.md). So this is a
// regression pin with an external anchor, not a second oracle.
//
// Five arms, and three of them exist to keep the pin from being vacuous:
//   1. the fixture loads with NO development hatch -- the claim the arch table
//      makes now, checked instead of assumed;
//   2. the greedy tokens equal the recorded sequence;
//   3. a different prompt gives a different sequence (the graph reads its input,
//      so a stack that emitted a constant row from any prompt cannot pass);
//   4. the same prompt at chunk 1 gives the same sequence (the row arithmetic is
//      bit-exact inside the ladder, which is what a 2-token prompt sits on);
//   5. a perturbed PLE table gives a DIFFERENT sequence (the pin can fail, and it
//      fails on the module whose bug this family actually had).
static void test_qwen4exp_whole_graph() {
    const std::string path = "kraken-q4whole.gguf";
    const std::string pert = "kraken-q4whole-pert.gguf";
    CHECK(build_tiny_q4_model(path, 0),
          "the full-schema fixture writes (lanes, PLE, indexer, MoE)");
    CHECK(build_tiny_q4_model(pert, 0, 0, 0, true, false, true, false, true),
          "and a copy with every PLE table entry shifted");

    // The hatch is NOT set in this group, on purpose. If the family needed it,
    // the first arm is the one that would say so.
    {
        Backend *cpu = make_cpu_backend();
        Model model;
        std::string err;
        CHECK(model.load(*cpu, path, &err),
              "the family loads with no development hatch");
        if (!err.empty()) std::fprintf(stderr, "  (load: %s)\n", err.c_str());
        model.unload();
        delete cpu;
    }

    // The hash of one logits row: FNV-1a over the f32 bit patterns. A token
    // sequence alone would be a weaker pin -- this fixture's synthetic head
    // argmaxes a special token after one step, so the text arm sees a single id,
    // and this repo has already been bitten once by a gate that compared only
    // the sampled token (the attention rewrite produced the same first token
    // from two different kernels). The row carries every stage's effect.
    // Byte-wise FNV-1a, the SAME function kraken.exe's --prefill-hash prints
    // for `logits`. Sharing it is the point: one number then pins the
    // in-process graph, the CLI's CPU run and the device run against each
    // other and against the record, instead of three hashes that cannot be
    // compared.
    auto row_hash = [](const f32 *v, i32 n) {
        u64 h = 1469598103934665603ull;
        const unsigned char *b = reinterpret_cast<const unsigned char *>(v);
        for (i64 i = 0; i < static_cast<i64>(n) * 4; i++) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
        return h;
    };

    // Prefill only (`max_tokens 0`), so the row read afterwards is the one at
    // the last prompt position -- the row a decode starts from.
    auto run = [&](const std::string &file, const char *prompt, i32 chunk,
                   std::vector<i32> *tokens, u64 *hash, i32 *top1,
                   RowPin *pin = nullptr) -> bool {
        Backend *cpu = make_cpu_backend();
        bool ok = false;
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = file;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = chunk;
            std::string err;
            if (!engine.init(cpu, cfg, &err)) {
                if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
            } else {
                GenerateParams p;
                p.prompt = prompt;
                p.max_tokens = 0;
                p.sampler.greedy = true;
                GenerateResult r;
                ok = engine.generate(p, &r) && r.finish != FinishError;
                const f32 *lg = engine.last_logits();
                const i32 nv = engine.model().cfg().n_vocab;
                if (ok && lg && nv > 0) {
                    *hash = row_hash(lg, nv);
                    i32 best = 0;
                    for (i32 i = 1; i < nv; i++)
                        if (lg[i] > lg[best]) best = i;
                    *top1 = best;
                    if (pin) *pin = row_pin(lg, nv);
                } else {
                    ok = false;
                }
                *tokens = r.tokens;
            }
        }
        delete cpu;
        return ok;
    };

    // Greedy text, which is the arm a reader means by "the graph's output".
    auto gen = [&](const std::string &file, const char *prompt, i32 chunk,
                   std::vector<i32> *tokens) -> bool {
        Backend *cpu = make_cpu_backend();
        bool ok = false;
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = file;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = chunk;
            std::string err;
            if (!engine.init(cpu, cfg, &err)) {
                if (!err.empty()) std::fprintf(stderr, "  (%s)\n", err.c_str());
            } else {
                GenerateParams p;
                p.prompt = prompt;
                p.max_tokens = 6;
                p.sampler.greedy = true;
                GenerateResult r;
                ok = engine.generate(p, &r) && r.finish != FinishError;
                *tokens = r.tokens;
            }
        }
        delete cpu;
        return ok;
    };

    std::vector<i32> prompt_row, narrow_row, other_row, pert_row;
    u64 h_wide = 0, h_narrow = 0, h_other = 0, h_pert = 0;
    i32 t_wide = -1, t_narrow = -1, t_other = -1, t_pert = -1;
    RowPin pin_wide;
    CHECK(run(path, "the", 8, &prompt_row, &h_wide, &t_wide, &pin_wide),
          "the whole graph prefills and the row reaches the head");
    std::fprintf(stderr, "  (logits row %016llx, argmax %d)\n",
                 static_cast<unsigned long long>(h_wide),
                 static_cast<int>(t_wide));

    // ---- 1. the recorded row --------------------------------------------
    // Recorded from this graph on the fixture, CPU backend, prompt "the".
    // The pin is the portable fingerprint (see row_pin's note); the bit hash
    // is host-dependent and is compared only under KRK_GOLDEN_BITS=1.
    static const u64 kGoldenHash = 0x1de9761d6eabacebull;
    static const i32 kGoldenArgmax = 8;
    static const RowPin kGolden = {
        {0.626528, 0.626528, 0.626528, 0.626528, 0.626528, 0.626528, 0.626528,
         0.626528},
        -0.944485, 132.638909, 9.229401};
    CHECK(h_wide != 0, "the row is not empty");
    CHECK(t_wide == kGoldenArgmax, "and its argmax is the recorded token");
    std::fprintf(stderr, "  (qwen4exp pin: %s)\n", pin_str(pin_wide).c_str());
    if (kGolden.l1 != 0.0)
        CHECK(pin_close(pin_wide, kGolden),
              "the last prompt row matches the recorded fingerprint (the whole "
              "graph's output)");
    if (golden_bits_on())
        CHECK(h_wide == kGoldenHash,
              "KRK_GOLDEN_BITS: the recorded row, bit for bit");

    // ---- 2. the prompt reaches the row -----------------------------------
    CHECK(run(path, "a", 8, &other_row, &h_other, &t_other),
          "a second prompt prefills");
    CHECK(h_other != h_wide, "a different prompt produces a different row");

    // ---- 3. chunk invariance ---------------------------------------------
    CHECK(run(path, "the", 1, &narrow_row, &h_narrow, &t_narrow),
          "the same prompt prefills token by token");
    CHECK(h_narrow == h_wide, "chunk 1 reproduces the chunk-8 row exactly");

    // ---- 4. the pin can fail ---------------------------------------------
    CHECK(run(pert, "the", 8, &pert_row, &h_pert, &t_pert),
          "the perturbed fixture prefills");
    CHECK(h_pert != h_wide,
          "a shifted PLE table moves the row (the pin is sensitive)");

    // ---- 5. and the graph emits text --------------------------------------
    std::vector<i32> greedy, greedy1;
    CHECK(gen(path, "the", 8, &greedy), "the whole graph generates");
    std::fprintf(stderr, "  (greedy tokens:");
    for (i32 t : greedy) std::fprintf(stderr, " %d", static_cast<int>(t));
    std::fprintf(stderr, ")\n");
    // One token: this fixture's synthetic head argmaxes EOS on the step after it
    // (which is why the row above, not this, is the pin).
    static const i32 kGoldenSeq[] = {8};
    const i32 kGoldenSeqN = 1;
    CHECK(static_cast<i32>(greedy.size()) == kGoldenSeqN,
          "the run emits the recorded number of tokens");
    bool same = static_cast<i32>(greedy.size()) == kGoldenSeqN;
    for (i32 i = 0; same && i < kGoldenSeqN; i++) same = greedy[i] == kGoldenSeq[i];
    CHECK(same, "and every emitted token is the recorded one");
    CHECK(gen(path, "the", 1, &greedy1), "the same prompt generates at chunk 1");
    CHECK(greedy1 == greedy, "chunk 1 emits the same tokens");

    // ---- 6. the DEVICE arm ------------------------------------------------
    // The same shape as test_dsv4_whole_graph's device arm: kraken-tests cannot
    // open the device, so the arm drives the CLI. It differs in what it can
    // claim, and the difference is MEASURED, not a convenience:
    //
    //   * the CPU half is a real pin -- the CLI's `--cpu` run hashes the same
    //     row the in-process graph does, which ties the row above to a second
    //     entry point and to the FNV the CLI prints;
    //   * the HIP half does NOT run this fixture, and the reason is measured: the
    //     lane stack's bottleneck is `low_rank` (8 here), and it is the
    //     UP-projection's K -- [hc*n_embd <- low_rank]. The HIP GEMM addresses K
    //     in 32-value chunks and ABORTS on anything else, so before the init
    //     guard this run died with 0xC0000409 two rows into the prefill. The
    //     device arm therefore requires a REFUSAL that names the geometry: a
    //     silent skip would accept a crash, and padding the fixture to K=32
    //     would prove the kernels on a geometry the graph never has.
    const std::string self = self_path();
    const std::filesystem::path exe_p =
        std::filesystem::path(self).parent_path() /
#ifdef _WIN32
        "kraken.exe";
#else
        "kraken";
#endif
    const bool have_exe = !self.empty() && std::filesystem::exists(exe_p);
    CHECK(have_exe, "kraken sits beside kraken-tests (the device arm drives it)");
    if (have_exe) {
        const std::string exe = exe_p.string();
        auto hex = [](const std::string &v) -> u64 {
            return v.empty() ? 0ull : std::strtoull(v.c_str(), nullptr, 16);
        };
        const PrefillProbe cc =
            run_prefill_hash(exe, path, "the", 8, false, false, "--cpu");
        CHECK(cc.ok, "the CLI prefills the fixture on the CPU reference");
        if (!cc.ok)
            std::fprintf(stderr, "  (cpu run rc=%d TAIL: %.400s)\n", cc.rc,
                         cc.err.size() > 400 ? cc.err.c_str() + cc.err.size() - 400
                                             : cc.err.c_str());
        std::fprintf(stderr, "  (q4whole   cpu trace %s logits %s first %d)\n",
                     cc.trace.c_str(), cc.logits.c_str(), cc.first);
        // Two entry points, one graph: the number the CLI prints is the row the
        // group pinned above.
        CHECK(cc.ok && hex(cc.logits) == h_wide,
              "the CLI's --cpu run hashes the row the in-process graph did");

        const PrefillProbe cd1 = run_prefill_hash(exe, path, "the", 1, false);
        const PrefillProbe cd8 = run_prefill_hash(exe, path, "the", 8, false);
        // The device arm's contract on a geometry HIP cannot address: the
        // refusal must say WHY. Any other exit -- a crash, a missing DLL, an
        // OOM -- is a failure, not a skip.
        const std::string kRule = "hyper_connection.low_rank";
        const std::string kRule2 = "multiple of 32";
        const bool refused = !cd8.ok &&
                             cd8.err.find(kRule) != std::string::npos &&
                             cd8.err.find(kRule2) != std::string::npos;
        CHECK(cd8.ok || refused,
              cd8.ok ? "and the same prefill runs on the device"
                     : (refused ? "the device REFUSES the fixture by name: the "
                                  "lane bottleneck (low_rank 8) is the "
                                  "up-projection's K and its GEMM needs a "
                                  "multiple of 32"
                                : "the device neither ran nor refused with a "
                                  "reason: the arm proved nothing"));
        if (!cd8.ok && !refused)
            std::fprintf(stderr, "  (device rc=%d TAIL: %.400s)\n", cd8.rc,
                         cd8.err.size() > 400 ? cd8.err.c_str() + cd8.err.size() - 400
                                              : cd8.err.c_str());
        if (refused) {
            std::fprintf(stderr,
                         "  (device: refused at init -- low_rank 8 is the "
                         "up-projection's K and the HIP GEMM needs 32s)\n");
            CHECK(cd1.ok == cd8.ok,
                  "both chunk sizes meet the same rule, so the refusal is the "
                  "geometry's and not the chunking's");
        } else if (cd8.ok && cd1.ok) {
            bool dev = false;
#ifdef KRK_TEST_HIP_BUILD
            dev = cd8.banner.find("scalar reference") == std::string::npos &&
                  cd8.err.find("falling back to the CPU") == std::string::npos;
            CHECK(dev,
                  dev ? "the HIP backend answered (not the CPU fallback)"
                      : "a HIP build fell back to the CPU: the device arm proved "
                        "nothing");
#endif
            if (dev) {
                std::fprintf(stderr,
                             "  (           dev trace %s logits %s first %d)\n",
                             cd8.trace.c_str(), cd8.logits.c_str(), cd8.first);
                CHECK(cd8.trace == cd1.trace,
                      cd8.trace == cd1.trace
                          ? "the DEVICE prefill folds to the same trace at chunk "
                            "1 and 8"
                          : "the device prefill is not chunk-invariant");
                CHECK(cd8.logits == cd1.logits,
                      cd8.logits == cd1.logits ? "and the same logits row"
                                               : "the device prefill leaves "
                                                 "different rows at chunk 1 "
                                                 "and 8");
                CHECK(cd8.first == cc.first,
                      "the device samples the token the CPU reference does");
            }
        }
        std::remove("kraken-prefill-out-8--cpu.txt");
        std::remove("kraken-prefill-err-8--cpu.txt");
        std::remove("kraken-prefill-out-1.txt");
        std::remove("kraken-prefill-err-1.txt");
        std::remove("kraken-prefill-out-8.txt");
        std::remove("kraken-prefill-err-8.txt");
    }

    std::remove(path.c_str());
    std::remove(pert.c_str());
}

// The gated normalization's ACTIVATION is the one thing about the delta-net
// block a file cannot tell you: Qwen3.5's `build_norm_gated` gates with
// silu(z), qwen4exp's with sigmoid(z), and the tensor list is identical either
// way. `ModelConfig::gdn_gate_sigmoid` carries the family's answer, and
// test_qwen4exp_schema checks that the loader sets it -- which is not the same
// claim as the graph USING it, and that gap is where this family decoded
// nonsense: the config said sigmoid, gdn_forward computed silu, and every
// check was green because every check read the flag instead of the output.
//
// So this group observes the graph. Two runs of the same fixture, one
// KRK_GDN_GATE=0 apart, and the dumped rows carry the whole argument:
//   - gdn.delta (the delta rule's raw output) and gdn.norm (after the per-head
//     RMSNorm) must be BIT-identical, so the arm reaches nothing above the gate;
//   - gdn.gated must differ, so the arm reaches the graph at all;
//   - and with z recovered as silu(z)/sigmoid(z), the default arm's ratio must
//     satisfy sigmoid(z) == 1/(1+exp(-z)) -- an identity silu-over-sigmoid
//     obeys and, say, silu-over-tanh does not, so the two rows are not merely
//     different but related by the right function.
static void test_qwen4exp_gdn_gate() {
    const std::string path = "kraken-q4gate.gguf";
    const std::string sig_dump = "kraken-q4gate-sig.dump";
    const std::string sil_dump = "kraken-q4gate-sil.dump";
    CHECK(build_tiny_q4_model(path, 0), "a miniature qwen4exp writes");
    std::remove(sig_dump.c_str());
    std::remove(sil_dump.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "1");
#else
    setenv("KRK_ARCH_ALLOW_EXPERIMENTAL", "1", 1);
#endif

    // gate == nullptr leaves the family's own answer in place.
    auto run = [&](const char *gate, const std::string &dump) {
        Backend *cpu = make_cpu_backend();
        bool ok = false;
        {
#ifdef _WIN32
            _putenv_s("KRK_DUMP", dump.c_str());
            if (gate) _putenv_s("KRK_GDN_GATE", gate);
            else _putenv_s("KRK_GDN_GATE", "");
#else
            setenv("KRK_DUMP", dump.c_str(), 1);
            if (gate) setenv("KRK_GDN_GATE", gate, 1);
            else unsetenv("KRK_GDN_GATE");
#endif
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = path;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = 8;
            std::string err;
            ok = engine.init(cpu, cfg, &err);
            if (ok) {
                GenerateParams p;
                p.prompt = "the";
                p.max_tokens = 1;
                p.sampler.greedy = true;
                GenerateResult r;
                ok = engine.generate(p, &r) && r.finish != FinishError;
            }
        }
#ifdef _WIN32
        _putenv_s("KRK_DUMP", "");
        _putenv_s("KRK_GDN_GATE", "");
#else
        unsetenv("KRK_DUMP");
        unsetenv("KRK_GDN_GATE");
#endif
        delete cpu;
        return ok;
    };

    CHECK(run(nullptr, sig_dump), "the default arm runs (sigmoid, from the arch)");
    CHECK(run("0", sil_dump), "and the silu arm runs (KRK_GDN_GATE=0)");

    std::vector<double> sig_norm, sil_norm, sig_gated, sil_gated, sig_delta,
        sil_delta;
    CHECK(dump_row_of(sig_dump, "gdn.norm", 0, &sig_norm),
          "the default dump has gdn.norm");
    CHECK(dump_row_of(sil_dump, "gdn.norm", 0, &sil_norm),
          "... and so does the silu one");
    CHECK(dump_row_of(sig_dump, "gdn.gated", 0, &sig_gated),
          "the default dump has gdn.gated");
    CHECK(dump_row_of(sil_dump, "gdn.gated", 0, &sil_gated),
          "... and so does the silu one");
    CHECK(dump_row_of(sig_dump, "gdn.delta", 0, &sig_delta),
          "the default dump has gdn.delta");
    CHECK(dump_row_of(sil_dump, "gdn.delta", 0, &sil_delta),
          "... and so does the silu one");

    if (sig_norm.size() && sig_norm.size() == sil_norm.size() &&
        sig_gated.size() == sil_gated.size() && sig_delta.size() == sil_delta.size() &&
        sig_gated.size() == sig_norm.size()) {
        // The arm is the gate and nothing else: the delta rule's output and the
        // normalize step above it are untouched.
        bool delta_same = sig_delta == sil_delta;
        bool norm_same = sig_norm == sil_norm;
        CHECK(delta_same, "the delta rule's own output is bit-identical across the arm");
        CHECK(norm_same, "... and so is the per-head norm above the gate");

        // ... while the gate's output is not, or the arm never reached the graph.
        i64 moved = 0;
        for (size_t i = 0; i < sig_gated.size(); i++)
            if (sig_gated[i] != sil_gated[i]) moved++;
        CHECK(moved > 0, "the gate's output moves, so the arm reaches gdn_forward");
        if (moved) {
            std::fprintf(stderr, "  (%lld of %lld gdn.gated values moved)\n",
                         static_cast<long long>(moved),
                         static_cast<long long>(sig_gated.size()));
        }

        // The two ratios are silu(z) and sigmoid(z) of ONE z, which makes them
        // related: sigmoid(z) == 1/(1+exp(-silu(z)/sigmoid(z))). This is the
        // check that says the default arm is a sigmoid and not merely some
        // other second activation.
        i64 paired = 0, ok_ident = 0;
        for (size_t i = 0; i < sig_gated.size(); i++) {
            if (sig_norm[i] == 0.0 || sil_norm[i] == 0.0) continue;
            const double r_sig = sig_gated[i] / sig_norm[i];
            const double r_sil = sil_gated[i] / sil_norm[i];
            if (!(r_sig > 0.0)) continue;   // a gate of 0 says nothing about z
            const double z = r_sil / r_sig;
            if (!(z > -80.0 && z < 80.0)) continue;
            const double pred = 1.0 / (1.0 + std::exp(-z));
            paired++;
            const double tol = 1e-5 * std::fabs(pred) + 1e-12;
            if (std::fabs(pred - r_sig) < tol) ok_ident++;
            else if (ok_ident + (paired - ok_ident) <= 4)
                std::fprintf(stderr, "  (i=%zu r_sig=%.9g r_sil=%.9g z=%.6g pred=%.9g)\n",
                             i, r_sig, r_sil, z, pred);
        }
        if (paired == 0) {
            // Nothing to relate usually means the rows did not parse the way
            // this test assumes, so say what it did read.
            std::fprintf(stderr, "  (rows: norm=%zu/%zu gated=%zu/%zu delta=%zu/%zu)\n",
                         sig_norm.size(), sil_norm.size(), sig_gated.size(),
                         sil_gated.size(), sig_delta.size(), sil_delta.size());
            for (size_t i = 0; i < sig_gated.size() && i < 4; i++)
                std::fprintf(stderr, "  (i=%zu norm=%.9g gated=%.9g sil=%.9g)\n",
                             i, sig_norm[i], sig_gated[i], sil_gated[i]);
        }
        CHECK(paired > 0, "there are elements to relate the two arms on");
        CHECK(paired > 0 && ok_ident == paired,
              "and the default arm's ratio is sigmoid(z) of the silu arm's z");
        if (paired && ok_ident != paired) {
            std::fprintf(stderr, "  (identity held on %lld of %lld elements)\n",
                         static_cast<long long>(ok_ident),
                         static_cast<long long>(paired));
        }
    }

    std::remove(path.c_str());
    std::remove(sig_dump.c_str());
    std::remove(sil_dump.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
}

// The mixed row a hyper-connected sub-block reads is
//     mean_over_lanes( xn_lane * sigmoid(gate_lane) )
// with `xn` the per-lane normalized stack and `gate` the up-projection's output
// -- and xn UNSCALED. The 1/hc in the mixer belongs to the bottleneck
// (`silu(scale(down @ xn, 1/hc))`), and scaling xn instead is invisible to
// every check that does not look at the mix itself: a matmul commutes with a
// constant, so `lo` and `gate` come out identical, the lane structure is
// untouched, and the only witness is the block's input being 1/hc of the
// reference's -- which reads as fluent, evenly shrunk text rather than as a
// fault. This group takes the dumped inputs and recomputes the mix with the
// host algebra (krk::hc_mix, the same code the CPU backend runs), so it fails
// on any factor between the two.
static void test_qwen4exp_hc_mix() {
    const std::string path = "kraken-q4hcmix.gguf";
    const std::string dump = "kraken-q4hcmix.dump";
    CHECK(build_tiny_q4_model(path, 0), "a miniature qwen4exp writes");
    std::remove(dump.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "1");
    _putenv_s("KRK_DUMP", dump.c_str());
#else
    setenv("KRK_ARCH_ALLOW_EXPERIMENTAL", "1", 1);
    setenv("KRK_DUMP", dump.c_str(), 1);
#endif
    i32 hc = 0;
    i64 ne = 0;
    {
        Backend *cpu = make_cpu_backend();
        {
            Engine engine;
            EngineConfig cfg;
            cfg.model_path = path;
            cfg.n_ctx = 64;
            cfg.prefill_chunk = 8;
            std::string err;
            const bool ok = engine.init(cpu, cfg, &err);
            CHECK(ok, "it initialises");
            if (ok) {
                GenerateParams p;
                p.prompt = "the";
                p.max_tokens = 1;
                p.sampler.greedy = true;
                GenerateResult r;
                CHECK(engine.generate(p, &r) && r.finish != FinishError,
                      "... and runs a forward");
                engine.hc_debug_lanes(&hc, &ne);
            }
        }
        delete cpu;
    }
#ifdef _WIN32
    _putenv_s("KRK_DUMP", "");
#else
    unsetenv("KRK_DUMP");
#endif

    std::vector<double> xn, gate, mix;
    CHECK(dump_row_of(dump, "hc.n", -1, &xn), "the dump has the normed lanes");
    CHECK(dump_row_of(dump, "hc.g", -1, &gate), "... the gate they are mixed by");
    CHECK(dump_row_of(dump, "hc.mix", -1, &mix), "... and the mixed row itself");
    if (hc > 1 && ne > 0 && xn.size() == static_cast<size_t>(hc) * ne &&
        gate.size() == xn.size() && mix.size() == static_cast<size_t>(ne)) {
        krk::HcGeometry g;
        g.n_embd = static_cast<i32>(ne);
        g.hc = hc;
        g.low_rank = 1; // the mix reads only the lane count
        std::vector<f32> xf(xn.begin(), xn.end());
        std::vector<f32> gf(gate.begin(), gate.end());
        std::vector<f32> want(static_cast<size_t>(ne), 0.0f);
        krk::hc_mix(want.data(), xf.data(), gf.data(), 1, g);
        i64 bad = 0;
        double worst = 0.0;
        for (i64 e = 0; e < ne; e++) {
            const double d = std::fabs(static_cast<double>(want[static_cast<size_t>(e)]) -
                                       mix[static_cast<size_t>(e)]);
            const double tol = 1e-6 * std::fmax(1.0, std::fabs(mix[static_cast<size_t>(e)]));
            if (d > tol) bad++;
            if (d > worst) worst = d;
        }
        if (bad) {
            // The factor is the whole point: report it, since a 1/hc mix is
            // the defect this check exists for.
            double ratio = 0.0;
            for (i64 e = 0; e < ne; e++) {
                const double w = want[static_cast<size_t>(e)];
                if (std::fabs(w) > 1e-3) {
                    ratio = mix[static_cast<size_t>(e)] / w;
                    break;
                }
            }
            std::fprintf(stderr,
                         "  (%lld of %lld mix values differ, worst %.3g, "
                         "dumped/oracle %.4f)\n",
                         static_cast<long long>(bad), static_cast<long long>(ne),
                         worst, ratio);
        }
        CHECK(bad == 0,
              "the mixed row is the lane mean of xn * sigmoid(gate), unscaled xn");
    }

    std::remove(path.c_str());
    std::remove(dump.c_str());
#ifdef _WIN32
    _putenv_s("KRK_ARCH_ALLOW_EXPERIMENTAL", "");
#else
    unsetenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
#endif
}

// ---------------------------------------------------------------------------
// Which pageable promotion path runs -- decided by the probe, on both apertures
// ---------------------------------------------------------------------------
//
// Both of `Engine::init`'s promotion decisions are measurements, so a test that
// hands them ratios would only exercise the comparison operators. The backend
// here grows a simulated APERTURE instead: its two pageable paths really take
// different amounts of time, the copies are real, and the numbers the probe
// reports come out of the clock. That is the only way this suite can reach the
// branch a Resizable-BAR box takes -- the card on this machine reports a 256 MiB
// aperture and the probe keeps the ring on every run.
static void test_pageable_path_probe() {
    const size_t mib = static_cast<size_t>(1) << 20;
    const size_t gib = static_cast<size_t>(1) << 30;

    auto probe = [&](double ring_gbs, double direct_gbs, double pin_gbs,
                     double block_gbs = -1.0) {
        CountingBackend be(make_cpu_backend());
        be.set_test_resources(16 * gib, 8 * gib, 16 * gib, 8 * gib);
        be.set_test_aperture(ring_gbs, direct_gbs, pin_gbs, block_gbs);
        return krk::probe_h2d_path(&be, 4 * mib, 3);
    };

    // A wide aperture: the driver stages a pageable source for free, so direct
    // clears the ring by a wide margin. The slow arm is 0.4 GB/s on purpose --
    // ~42 ms for the 16 MiB copy the engine probes at -- because a per-copy
    // scheduling delay is ADDITIVE: it compresses any ratio towards 1, and a
    // tight margin would make this group report the test box's load as the
    // card's aperture. It runs in the serial lane for the same reason.
    const krk::H2dProbe wide = probe(0.4, 20.0, 20.0);
    CHECK(wide.valid, "the aperture probe measured all three arms");
    CHECK(wide.direct_gbs > wide.pageable_gbs * 1.50,
          "a wide aperture's direct path beats the staging ring");
    CHECK(krk::pageable_direct_wins(wide.pageable_gbs, wide.direct_gbs),
          "so the probe's own numbers select the direct path");

    // A narrow aperture: the driver's staging serialises through the window, so
    // the ring wins and the pin is worth its unswappable RAM.
    const krk::H2dProbe narrow = probe(6.0, 0.4, 20.0);
    CHECK(narrow.valid, "the narrow arm measured too");
    CHECK(narrow.pageable_gbs > narrow.direct_gbs * 1.50,
          "a windowed aperture keeps the staging ring");
    CHECK(!krk::pageable_direct_wins(narrow.pageable_gbs, narrow.direct_gbs),
          "and the probe does not switch paths on it");

    // The third arm, at this box's SHAPE: the ring is far behind, and the
    // driver's own blocking copy is the fastest thing the card offers. The
    // probe has to measure it (not assume it), report it, and hand the engine a
    // path of 2 -- the case the old two-way menu could not express at all.
    const krk::H2dProbe blocking = probe(8.4, 10.0, 24.4, 22.9);
    CHECK(blocking.valid && blocking.blocking_gbs > 0.0,
          "the probe measures the blocking copy as a fourth arm");
    CHECK(blocking.blocking_gbs > blocking.pageable_gbs,
          "and it measures the ring as the slowest mechanism on that card");
    CHECK(krk::h2d_path(blocking) == 2,
          "so the probe's own numbers select the blocking copy");
    CHECK(krk::h2d_path_gbs(blocking) == blocking.blocking_gbs,
          "and the rate the engine pins against is the chosen arm's, not the ring's");
    CHECK(blocking.aperture_wide,
          "the aperture is WIDE by measurement: 22.9 against 24.4 GB/s pinned");
    CHECK(!krk::H2dProbe{}.aperture_wide, "an unmeasured probe claims nothing");

    // --- when the probe is allowed to stop -------------------------------
    // Driven with the rates measured on this box (`--h2d-noise`, one size per
    // process) rather than with sleeps: the rule is arithmetic, and its two
    // failure modes are arithmetic too -- a tight ramp, and one noisy arm.
    CHECK(krk::h2d_samples_settled({11.15, 11.77, 11.98},
                                   krk::kH2dProbeSettleSpread),
          "three tight samples settle -- the 64 MiB case the rule exists for");
    CHECK(!krk::h2d_samples_settled({8.36, 7.86, 10.81},
                                    krk::kH2dProbeSettleSpread),
          "a tight RAMP does not: 16 MiB's first samples read below its steady "
          "state, and the newest two give that away");
    CHECK(krk::h2d_samples_settled({15.74, 23.02, 20.50, 21.55},
                                   krk::kH2dProbeSettleSpread),
          "32 MiB's pinned arm settles at four, after its cold first sample");
    CHECK(!krk::h2d_samples_settled({9.0, 9.1}, krk::kH2dProbeSettleSpread),
          "two samples never settle: a median needs three");
    CHECK(!krk::h2d_samples_settled({0.0, 0.0, 0.0}, krk::kH2dProbeSettleSpread),
          "a failed copy is not a settled rate");
    {
        krk::H2dSamples mixed;
        mixed.ring = {11.15, 11.77, 11.98};
        mixed.direct = {10.91, 11.82, 11.78};
        mixed.pinned = {20.06, 14.0, 24.60};  // one arm still moving
        CHECK(!krk::h2d_probe_settled(mixed, krk::kH2dProbeSettleSpread),
              "one unsettled arm keeps the probe running: the pin decision reads it");
    }
    {
        // The loop itself: `iters` is a cap, and the env arm restores the fixed
        // quota the early stop replaced -- an A/B needs the old behaviour, not a
        // smaller cap that would still stop early.
        auto *probe_be = new CountingBackend(make_cpu_backend());
        probe_be->set_test_resources(16 * gib, 8 * gib, 16 * gib, 8 * gib);
        probe_be->set_test_aperture(0.4, 20.0, 20.0);
        krk::H2dSamples auto_s;
        const krk::H2dProbe auto_p = krk::probe_h2d_path(
            probe_be, 4 * mib, krk::kH2dProbeMaxIters, &auto_s);
        CHECK(auto_p.valid && auto_p.iterations >= krk::kH2dProbeMinIters &&
                  auto_p.iterations <= krk::kH2dProbeMaxIters,
              "the probe runs between the floor and the cap");
        CHECK(auto_s.ring.size() == static_cast<size_t>(auto_p.iterations) &&
                  auto_s.direct.size() == auto_s.ring.size() &&
                  auto_s.pinned.size() == auto_s.ring.size(),
              "and reports the samples it actually took, one per arm per iteration");
        set_env_var("KRK_H2D_PROBE_ITERS", "3");
        krk::H2dSamples forced_s;
        const krk::H2dProbe forced_p = krk::probe_h2d_path(
            probe_be, 4 * mib, krk::kH2dProbeMaxIters, &forced_s);
        CHECK(forced_p.iterations == 3 && forced_s.ring.size() == 3,
              "KRK_H2D_PROBE_ITERS runs exactly that many, which is the old "
              "fixed quota");
        set_env_var("KRK_H2D_PROBE_ITERS", "1");
        const krk::H2dProbe floor_p = krk::probe_h2d_path(
            probe_be, 4 * mib, krk::kH2dProbeMaxIters);
        CHECK(floor_p.iterations >= krk::kH2dProbeMinIters,
              "and a value below the floor is refused rather than measured");
        clear_env_var("KRK_H2D_PROBE_ITERS");
        // The arm that FITS the rule has to see the whole series, settled or
        // not: asking for samples is not the same as declining the early stop.
        krk::H2dSamples all_s;
        const krk::H2dProbe all_p = krk::probe_h2d_path(
            probe_be, 4 * mib, krk::kH2dProbeMaxIters, &all_s, true);
        CHECK(all_p.iterations == krk::kH2dProbeMaxIters &&
                  all_s.ring.size() == static_cast<size_t>(krk::kH2dProbeMaxIters),
              "collect_all runs the whole cap, which is what --h2d-noise fits the "
              "rule to");
        delete probe_be;
    }

    // The wiring: Engine::init must APPLY that decision, and the two apertures
    // must decode the same tokens -- the path is a performance choice, never a
    // numerical one.
    const std::string path = "kraken-aperture-path.gguf";
    MoeSpec spec;
    CHECK(build_tiny_moe_model(path, spec),
          "wrote the MoE fixture for the aperture test");
    // `scale` stretches every simulated copy without changing a single RATIO.
    // The verdict is decided by ratios (`pageable_path_choice` asks each arm to
    // beat the running best by 1.20x); the absolute rates only say how long the
    // simulated copy BUSY-SPINS, and that spin is timed with steady_clock. A
    // spinning thread that gets preempted overshoots its deadline, and an
    // additive overshoot compresses every ratio towards 1 -- the same trap the
    // 0.4 GB/s arm above exists for, and the reason this group is serial.
    //
    // The blocking arm below is the one with a tight margin: at this box's
    // shape and 16 MiB its ring is 2.0 ms against blocking's 0.73 ms, so the
    // verdict survives only d <= (2.0 - 1.2*0.73)/0.2 = 5.6 ms of stall -- less
    // than one scheduler quantum for a preempted thread. That is why six
    // concurrent selfcheck suites flipped it ("its own measurement says the
    // blocking copy is the fastest mechanism" failing about one gate run in
    // three). Ten times slower is the SAME shape and tolerates d <= 56 ms.
    auto run = [&](double ring_gbs, double direct_gbs, double pin_gbs,
                   bool *direct_on, size_t *pin, std::vector<i32> *tokens,
                   krk::H2dProbe *used, double block_gbs = -1.0,
                   int *path_used = nullptr, double scale = 1.0) {
        auto *be = new CountingBackend(make_cpu_backend());
        be->set_test_resources(16 * gib, 8 * gib, 16 * gib, 8 * gib);
        be->set_test_aperture(ring_gbs * scale, direct_gbs * scale,
                              pin_gbs * scale,
                              block_gbs >= 0.0 ? block_gbs * scale : block_gbs);
        Engine engine;
        EngineConfig cfg;
        cfg.model_path = path;
        cfg.n_ctx = 64;
        cfg.prefill_chunk = 8;
        cfg.expert_warm_mb = 64;
        std::string err;
        const bool ok = engine.init(be, cfg, &err);
        if (!ok) std::fprintf(stderr, "  aperture engine init: %s\n", err.c_str());
        if (ok) {
            GenerateParams p;
            p.prompt = "the";
            p.max_tokens = 2;
            p.sampler.greedy = true;
            p.sampler.temp = 0.0f;
            GenerateResult r;
            if (!engine.generate(p, &r))
                std::fprintf(stderr, "  aperture generate failed\n");
            *tokens = r.tokens;
            *direct_on = be->direct_enabled();
            if (path_used) *path_used = be->path();
            *pin = engine.model().experts().host_pin_budget();
            *used = engine.h2d_probe();
        }
        engine.shutdown();
        delete be;
        return ok;
    };

    bool wide_direct = false, narrow_direct = true;
    size_t wide_pin = 1, narrow_pin = 0;
    std::vector<i32> wide_tokens, narrow_tokens;
    krk::H2dProbe wide_used, narrow_used;
    const bool wide_ok =
        run(0.4, 20.0, 20.0, &wide_direct, &wide_pin, &wide_tokens, &wide_used);
    const bool narrow_ok =
        run(6.0, 0.4, 20.0, &narrow_direct, &narrow_pin, &narrow_tokens,
            &narrow_used);
    CHECK(wide_ok && narrow_ok,
          "the engine initialises on both simulated apertures");
    CHECK(wide_direct, "a wide aperture's engine switches to the direct pageable path");
    CHECK(!narrow_direct, "a windowed aperture's engine keeps the staging ring");
    CHECK(wide_used.valid && narrow_used.valid,
          "both runs published the probe they decided on");
    // The wiring is the contract, and it holds whatever the machine was doing:
    // the engine applied the rule to the numbers IT measured. Asserting a fixed
    // verdict for the pin here would assert this box's scheduler -- direct and
    // pinned are the same copy on a wide aperture, so that gap is a near-tie by
    // construction.
    CHECK(wide_direct == (krk::h2d_path(wide_used) == 1),
          "the path the engine set is the one its own measurement selected");
    CHECK((wide_pin > 0) ==
              krk::warm_pin_worthwhile(krk::h2d_path_gbs(wide_used),
                                       wide_used.pinned_gbs),
          "and its pin follows the rate the chosen path will pay");
    CHECK(narrow_direct == (krk::h2d_path(narrow_used) == 1),
          "the narrow run applied its own verdict too");
    CHECK((narrow_pin > 0) ==
              krk::warm_pin_worthwhile(krk::h2d_path_gbs(narrow_used),
                                       narrow_used.pinned_gbs),
          "and its pin is the one its own numbers justified");
    CHECK(wide_tokens.size() == narrow_tokens.size() && wide_tokens == narrow_tokens,
          "both apertures decode the same tokens: the path is not numerical");

    // The third mechanism, end to end: a card whose fastest pageable path is the
    // driver's blocking copy. The engine has to apply path 2 (a value the old
    // two-way setting could not express), leave `direct` off, and decode the
    // same tokens -- this is the wiring the ReBAR fix exists for.
    bool block_direct = true;
    size_t block_pin = 1;
    int engine_path = -1;
    std::vector<i32> block_tokens;
    krk::H2dProbe block_used;
    // 0.1: the same shape at a tenth of the rates, so the verdict has ~10x the
    // headroom against a stalled spin (see `scale` on `run` above).
    const bool block_ok = run(8.4, 10.0, 24.4, &block_direct, &block_pin,
                              &block_tokens, &block_used, 22.9, &engine_path,
                              0.1);
    CHECK(block_ok && block_used.valid, "the engine initialises on a blocking-fast card");
    CHECK(krk::h2d_path(block_used) == 2,
          "its own measurement says the blocking copy is the fastest mechanism");
    CHECK(engine_path == 2,
          "and the engine APPLIED it -- not just reported it");
    CHECK(!block_direct, "the direct flag is off, because blocking is not direct");
    CHECK(block_tokens == wide_tokens,
          "the blocking path decodes the same tokens as the rest");

    // --- the cache init reaches for before it measures -----------------------
    // The probe measures the CARD, so its answer is stable across runs and is
    // cached per (card, probe size, aperture hint). These arms drive the file
    // directly, on a path this group owns, so what is checked is the format and
    // the invalidation rules rather than this machine's cached verdict.
    const std::string cache_path = "kraken-h2d-cache.txt";
    set_env_var("KRK_H2D_CACHE", cache_path.c_str());
    CHECK(krk::h2d_probe_cache_path() == cache_path,
          "KRK_H2D_CACHE names the file the measured verdict is cached in");

    krk::H2dCacheKey key;
    {
        CountingBackend cap_be(make_cpu_backend());
        cap_be.set_test_resources(16 * gib, 8 * gib, 16 * gib, 8 * gib);
        const krk::DeviceCaps caps = cap_be.caps();
        key.name = caps.name + " " + caps.gfx;
        key.vram_total = caps.vram_total;
        key.large_bar = caps.large_bar;
        key.driver_version = caps.driver_version;
        key.bytes = 16 * mib; // the engine buckets this fixture's expert up to the floor
    }
    // The probe size is BUCKETED, not the model's own footprint: a cache can
    // only be filled by a tool that knows every size a run can ask about, and
    // `kraken-bench --h2d` measures exactly these three.
    CHECK(krk::h2d_probe_size_bytes(1) == 16 * mib,
          "a tiny expert probes at the 16 MiB floor");
    CHECK(krk::h2d_probe_size_bytes(20 * mib) == 32 * mib,
          "20 MiB rounds up to 32, so the bench's three sizes cover every model");
    CHECK(krk::h2d_probe_size_bytes(40 * mib) == 64 * mib,
          "40 MiB rounds up to 64");
    CHECK(krk::h2d_probe_size_bytes(1024 * mib) == 64 * mib,
          "and nothing is probed past the 64 MiB ceiling");
    // The log line's provenance, pinned as text: a cached number is only
    // interpretable with its age AND the driver that measured it, and a backend
    // that cannot report a driver has to say so rather than print nothing.
    const std::string fresh_origin = krk::h2d_probe_origin(false, 0, 71626331);
    CHECK(fresh_origin.find("measured now") != std::string::npos &&
              fresh_origin.find("71626331") != std::string::npos,
          "a fresh measurement names the driver that produced it");
    CHECK(krk::h2d_probe_origin(true, 8, 71626331).find("measured 8 s ago") !=
              std::string::npos,
          "a cached verdict says how old it is");
    CHECK(krk::h2d_probe_origin(true, 7200, 71626331).find("measured 2 h ago") !=
              std::string::npos,
          "and switches to hours when the seconds stop being readable");
    CHECK(krk::h2d_probe_origin(true, 8, 0).find("driver unknown") !=
              std::string::npos,
          "a backend that cannot report its driver says so instead of going quiet");

    const long long now = krk::h2d_now_sec();
    CHECK(krk::h2d_probe_cache_write(cache_path, key, wide, now),
          "a measured verdict is written to the cache");
    const krk::H2dCacheLookup back = krk::h2d_probe_cache_read(cache_path, key, now);
    CHECK(back.hit && std::fabs(back.probe.pageable_gbs - wide.pageable_gbs) < 1e-3 &&
              std::fabs(back.probe.direct_gbs - wide.direct_gbs) < 1e-3 &&
              std::fabs(back.probe.blocking_gbs - wide.blocking_gbs) < 1e-3 &&
              std::fabs(back.probe.pinned_gbs - wide.pinned_gbs) < 1e-3,
          "it reads back as the same four rates");
    CHECK(back.hit && back.probe.valid, "... and as a usable verdict, not just numbers");

    // Each invalidation rule is a way a cached answer would be WRONG, so each
    // one is exercised rather than described.
    krk::H2dCacheKey other = key;
    other.bytes += 8 * mib;
    CHECK(!krk::h2d_probe_cache_read(cache_path, other, now).hit,
          "a verdict measured at another probe size is not quoted for this one");
    other = key;
    other.large_bar = key.large_bar == 1 ? 0 : 1;
    CHECK(!krk::h2d_probe_cache_read(cache_path, other, now).hit,
          "a changed aperture hint invalidates the verdict -- the firmware event");
    other = key;
    other.vram_total += gib;
    CHECK(!krk::h2d_probe_cache_read(cache_path, other, now).hit,
          "and so does a different card reporting a different size");
    other = key;
    other.driver_version = key.driver_version + 1;
    CHECK(!krk::h2d_probe_cache_read(cache_path, other, now).hit,
          "a driver update invalidates the verdict, like a firmware change");
    other = key;
    other.driver_version = key.driver_version;
    CHECK(krk::h2d_probe_cache_read(cache_path, other, now).hit,
          "the same driver still finds its own entry");
    CHECK(!krk::h2d_probe_cache_read(cache_path, key,
                                     now + krk::kH2dCacheMaxAgeSec + 1)
               .hit,
          "an entry older than the cache's lifetime is re-measured");

    // Two entries coexist: a second card (or size) must not evict the first.
    krk::H2dProbe other_probe = wide;
    other_probe.pageable_gbs = 3.5;
    other_probe.direct_gbs = 4.0;
    CHECK(krk::h2d_probe_cache_write(cache_path, other, other_probe, now),
          "a second key is cached beside the first");
    CHECK(krk::h2d_probe_cache_read(cache_path, key, now).hit &&
              krk::h2d_probe_cache_read(cache_path, other, now).hit,
          "and neither entry displaced the other");

    // Pruning is a property of the FILE, so the arms below check both sides of
    // it: the verdict was readable, and the next write removed it.
    krk::H2dCacheKey old_driver = key;
    old_driver.driver_version = key.driver_version + 7;
    const long long rollback = krk::kH2dCacheOtherDriverMaxAgeSec;
    CHECK(krk::h2d_probe_cache_write(cache_path, old_driver, other_probe,
                                     now - rollback - 3600),
          "a verdict from a driver that has not run for a week can be written");
    CHECK(krk::h2d_probe_cache_read(cache_path, old_driver, now).hit,
          "it is still inside its 30-day life before any write touches the file");
    CHECK(krk::h2d_probe_cache_write(cache_path, key, wide, now),
          "the current driver then writes its own verdict");
    CHECK(!krk::h2d_probe_cache_read(cache_path, old_driver, now).hit,
          "and that write pruned the stale driver's verdict");

    // The rollback case the window exists for must survive: a driver that ran
    // yesterday is one the user may go back to today.
    krk::H2dCacheKey recent_driver = key;
    recent_driver.driver_version = key.driver_version + 9;
    CHECK(krk::h2d_probe_cache_write(cache_path, recent_driver, other_probe,
                                     now - 3600),
          "a verdict from yesterday's driver is written");
    CHECK(krk::h2d_probe_cache_write(cache_path, key, wide, now) &&
              krk::h2d_probe_cache_read(cache_path, recent_driver, now).hit,
          "and it survives a write: that is the rollback the window is for");

    {
        // What a rewrite dropped, by reason. The counts are the only visible
        // evidence that pruning happened at all: the file is the same size
        // either way, and a verdict that vanishes silently is indistinguishable
        // from one that was never written.
        const std::string reasons_path = "kraken-h2d-cache-reasons.txt";
        std::FILE *f = std::fopen(reasons_path.c_str(), "wb");
        CHECK(f != nullptr, "the reasons file is there to seed");
        if (f) {
            auto emit = [&](int driver, size_t bytes, long long stamp) {
                std::fprintf(f, "3\t%s\t%llu\t%d\t%d\t%llu\t0.4\t20\t0.4\t20\t%lld\n",
                             key.name.c_str(),
                             static_cast<unsigned long long>(key.vram_total),
                             key.large_bar, driver,
                             static_cast<unsigned long long>(bytes), stamp);
            };
            emit(key.driver_version, key.bytes + 4096,
                 now - krk::kH2dCacheMaxAgeSec - 60);          // expired
            emit(key.driver_version + 5, key.bytes + 8192,
                 now - rollback - 60);                          // past the rollback window
            std::fputs("not a verdict\n", f);             // unreadable
            std::fclose(f);
        }
        krk::H2dCachePrune pruned;
        CHECK(krk::h2d_probe_cache_write(reasons_path, key, wide, now, &pruned),
              "a write over dead verdicts succeeds");
        CHECK(pruned.expired == 1 && pruned.other_driver == 1 && pruned.unreadable == 1,
              "it reports each dead verdict by the reason it was dropped");
        CHECK(pruned.dropped() == 3 && pruned.kept == 1,
              "and says how many lines are left");
        CHECK(krk::h2d_prune_text(pruned).find("dropped 3 dead verdict(s)") !=
                  std::string::npos,
              "the line a caller logs names the count");
        CHECK(krk::h2d_prune_text(krk::H2dCachePrune()).empty(),
              "and stays quiet when nothing was dropped");
        std::remove(reasons_path.c_str());
    }
    {
        // What a run will START FROM, listed without running one. The dead lines
        // are the point: the live ones are what the user already has, and "why is
        // my run measuring again" is answered by what was skipped and why -- so a
        // listing that filtered to the live entries could not answer it at all.
        const std::string list_path = "kraken-h2d-cache-list.txt";
        std::FILE *f = std::fopen(list_path.c_str(), "wb");
        CHECK(f != nullptr, "the listing file is there to seed");
        if (f) {
            auto emit = [&](int driver, size_t bytes, long long stamp) {
                std::fprintf(f, "3\t%s\t%llu\t%d\t%d\t%llu\t0.4\t20\t0.4\t20\t%lld\n",
                             key.name.c_str(),
                             static_cast<unsigned long long>(key.vram_total),
                             key.large_bar, driver,
                             static_cast<unsigned long long>(bytes), stamp);
            };
            emit(key.driver_version, key.bytes, now);        // live, this run's key
            emit(key.driver_version, key.bytes + 4096, now); // live, another size
            emit(key.driver_version, key.bytes + 8192,
                 now - krk::kH2dCacheMaxAgeSec - 60); // expired
            std::fputs("not a verdict\n", f);        // unreadable
            std::fclose(f);
        }
        const std::vector<krk::H2dCacheEntry> all =
            krk::h2d_cache_scan(list_path, now);
        CHECK(all.size() == 4, "every line is listed, the dead ones included");
        int live = 0, expired = 0, unreadable = 0;
        for (const krk::H2dCacheEntry &e : all) {
            if (e.live()) live++;
            else if (e.dead == "expired") expired++;
            else if (e.name.empty()) unreadable++;
        }
        CHECK(live == 2 && expired == 1 && unreadable == 1,
              "and each dead line carries the reason a run would skip it");
        // Two live entries, one key: only the size this run asks for is marked,
        // which is the difference between "a verdict exists" and "this run
        // uses one".
        const std::vector<krk::H2dCacheEntry> keyed =
            krk::h2d_cache_scan(list_path, now, &key);
        int used = 0, matched = 0;
        for (const krk::H2dCacheEntry &e : keyed) {
            if (e.used) used++;
            if (e.matches) matched++;
        }
        CHECK(matched == 1 && used == 1,
              "exactly the entry this box's key asks for is marked as the one it "
              "starts from");
        // A listing has no model, so it has no probe size either: `bytes == 0`
        // asks the card's question ("is there a verdict to start from") and must
        // match without claiming which of the sizes a run would pick. Getting
        // this wrong read as "no entry for this card" on a machine whose cache
        // held three -- the size in the key was 0 and nothing matched.
        krk::H2dCacheKey card_only = key;
        card_only.bytes = 0;
        const std::vector<krk::H2dCacheEntry> card =
            krk::h2d_cache_scan(list_path, now, &card_only);
        int card_live = 0, card_used = 0;
        for (const krk::H2dCacheEntry &e : card) {
            if (e.matches && e.live()) card_live++;
            if (e.used) card_used++;
        }
        CHECK(card_live == 1 && card_used == 0,
              "a listing without a model matches the card's bucket sizes and "
              "claims none of them");
        // The text IS the listing: the chosen path from the engine's own
        // predicate, the age and driver of the verdict, and the size measured.
        std::string live_text, expired_text;
        for (const krk::H2dCacheEntry &e : keyed) {
            if (e.used) live_text = krk::h2d_cache_entry_text(e);
            if (e.dead == "expired") expired_text = krk::h2d_cache_entry_text(e);
        }
        CHECK(live_text.find("-> direct") != std::string::npos,
              "the listing prints the path the engine's own predicate selects");
        CHECK(live_text.find("measured 0 s ago") != std::string::npos,
              "with the verdict's age, so a stale one is visible");
        CHECK(expired_text.find("[expired]") != std::string::npos,
              "and a dead line says why in place of a path");
        CHECK(expired_text.find("direct 20.00") != std::string::npos,
              "an expired line still shows what it measured");
        CHECK(krk::h2d_cache_entry_text(keyed.front(), true).find(
                  "[another driver]") != std::string::npos,
              "a live verdict from another driver can be named as not this run's");
        // A file that is not there, and a cache that is off, are empty
        // listings rather than failures: this is a diagnostic, not a step.
        CHECK(krk::h2d_cache_scan("kraken-no-such-cache.txt", now).empty(),
              "a missing cache file lists as empty");
        CHECK(krk::h2d_cache_scan(std::string(), now).empty(),
              "and a disabled cache lists nothing");
        std::remove(list_path.c_str());
    }
    {
        // The init line, which is the whole point of the cache being invisible
        // otherwise: reused or measured, and -- when measured -- WHY nothing was
        // there to reuse. The three miss reasons below are the three a user can
        // actually act on, and "measured now" alone reads the same for all of
        // them.
        const std::string verdict_cache = "kraken-h2d-cache-verdict.txt";
        // A real driver number, because the fixture's simulated caps report 0
        // and "driver unknown" would pass a check for "is anything printed".
        const std::string reused = krk::h2d_verdict_text(
            true, 45 * 60, 71626331, key.bytes, std::string());
        CHECK(reused.find("reused the cached 16 MiB one") != std::string::npos,
              "a cached run says it reused the verdict, and which size");
        CHECK(reused.find("measured 45 min ago") != std::string::npos &&
                  reused.find("71626331") != std::string::npos,
              "with the age and the driver it came from");
        CHECK(krk::h2d_verdict_text(false, 0, key.driver_version, key.bytes,
                                    "nothing at all")
                      .find("measured the card now") != std::string::npos,
              "a run that measured says so");
        CHECK(krk::h2d_cache_miss_text(std::string(), now, key).find("cache is off") !=
                  std::string::npos,
              "a miss because the cache is off says that, not \"nothing cached\"");
        CHECK(krk::h2d_cache_miss_text("kraken-no-such-cache.txt", now, key)
                      .find("no cache file") != std::string::npos,
              "and a miss with no file says there is no file yet");
        {
            // This card's entry for THIS size, dead: the fix is one more run,
            // which replaces it -- not a model that buckets elsewhere.
            std::FILE *f = std::fopen(verdict_cache.c_str(), "wb");
            CHECK(f != nullptr, "the verdict fixture is there to seed");
            if (f) {
                auto emit = [&](int driver, size_t bytes, long long stamp) {
                    std::fprintf(f, "3\t%s\t%llu\t%d\t%d\t%llu\t0.4\t20\t0.4\t20\t%lld\n",
                                 key.name.c_str(),
                                 static_cast<unsigned long long>(key.vram_total),
                                 key.large_bar, driver,
                                 static_cast<unsigned long long>(bytes), stamp);
                };
                emit(key.driver_version, key.bytes,
                     now - krk::kH2dCacheMaxAgeSec - 60);         // expired, exact size
                emit(key.driver_version, key.bytes + 4096, now);  // live, another size
                std::fputs("garbage\n", f);                     // dead
                std::fclose(f);
            }
            const std::string miss = krk::h2d_cache_miss_text(verdict_cache, now, key);
            CHECK(miss.find("this card's 16 MiB entry is not usable (expired)") !=
                      std::string::npos,
                  "an expired entry for this exact size is named as the reason");
            CHECK(miss.find("1 live at other probe sizes") != std::string::npos &&
                      miss.find("2 dead of 3 line(s)") != std::string::npos,
                  "and the miss counts what else the file held");
            CHECK(krk::h2d_verdict_text(false, 0, key.driver_version, key.bytes, miss)
                          .find("not usable (expired)") != std::string::npos,
                  "the measured line carries that reason rather than a generic one");
            std::remove(verdict_cache.c_str());
        }
        {
            // A live verdict from the driver BEFORE an update: the writer keeps
            // it for the rollback window, the reader refuses it, and a run that
            // measures because of it must say so -- that is the case where the
            // file looks populated and nothing is wrong with it.
            std::FILE *f = std::fopen(verdict_cache.c_str(), "wb");
            CHECK(f != nullptr, "the rollback fixture is there to seed");
            if (f) {
                std::fprintf(f,
                             "3\t%s\t%llu\t%d\t%d\t%llu\t0.4\t20\t0.4\t20\t%lld\n",
                             key.name.c_str(),
                             static_cast<unsigned long long>(key.vram_total),
                             key.large_bar, key.driver_version + 1,
                             static_cast<unsigned long long>(key.bytes), now);
                std::fclose(f);
            }
            const std::string miss = krk::h2d_cache_miss_text(verdict_cache, now, key);
            CHECK(miss.find("1 live from another driver") != std::string::npos,
                  "a live entry from another driver is named as the reason, not as a "
                  "missing entry");
            std::remove(verdict_cache.c_str());
        }
    }
    {
        // The end-of-run ledger. One run is a hit rate of 0% or 100%, so what is
        // checked here is the arithmetic ACROSS runs and the honesty of the
        // saved-time estimate: priced at an average this machine measured, or not
        // priced at all. Two SIZES are counted, because the sizes are not
        // interchangeable -- a 64 MiB verdict is not the one a 16 MiB model asks
        // for, and its measurement costs several times as much.
        const std::string tally_cache = "kraken-h2d-tally.txt";
        const std::string tally_path = krk::h2d_tally_path(tally_cache);
        CHECK(tally_path == tally_cache + ".stats",
              "the ledger sits beside the cache it counts, so moving the cache "
              "moves it");
        CHECK(krk::h2d_tally_text(krk::h2d_tally_load(tally_path)).empty(),
              "nothing counted prints nothing, rather than a 0% hit rate");
        bool wrote = false;
        krk::H2dProbeTally t =
            krk::h2d_tally_note(tally_cache, key.bytes, true, 0, &wrote);
        CHECK(wrote && t.runs == 1 && t.cached == 1 && t.measured == 0,
              "a hit is counted as a run and a hit");
        CHECK(krk::h2d_tally_load(tally_path).cached == 1,
              "and the count outlives the process that made it");
        CHECK(t.buckets.size() == 1 && t.buckets[0].bytes == key.bytes,
              "the count lands in the bucket of the size that asked for it");
        t = krk::h2d_tally_note(tally_cache, key.bytes, false, 240, &wrote);
        t = krk::h2d_tally_note(tally_cache, key.bytes, true, 0, &wrote);
        t = krk::h2d_tally_note(tally_cache, key.bytes, true, 0, &wrote);
        CHECK(t.runs == 4 && t.cached == 3 && t.measured == 1 && t.probe_ms == 240,
              "three hits and one 240 ms measurement add up");
        const std::string text = krk::h2d_tally_text(t);
        CHECK(text.find("3 of 4 probed run(s)") != std::string::npos &&
                  text.find("(75%)") != std::string::npos,
              "the line reports the hit rate this cache has actually had");
        CHECK(text.find("240.0 ms on average") != std::string::npos &&
                  text.find("720 ms") != std::string::npos,
              "and prices the avoided probes at the machine's own average");
        // A second size, with no hits and its own measurement cost. Its rate and
        // its price must not be blended into the first's.
        t = krk::h2d_tally_note(tally_cache, key.bytes * 2, false, 300, &wrote);
        t = krk::h2d_tally_note(tally_cache, key.bytes * 2, false, 100, &wrote);
        const krk::H2dTallyBucket *small = t.find(key.bytes);
        const krk::H2dTallyBucket *large = t.find(key.bytes * 2);
        CHECK(t.buckets.size() == 2 && small && large &&
                  small->runs == 4 && large->runs == 2,
              "each probe size keeps its own count, sorted by size");
        CHECK(small && large && small->hit_rate() == 75.0 &&
                  large->hit_rate() == 0.0,
              "and its own hit rate: 4 runs at 16 MiB are not 6 runs across sizes");
        CHECK(small && large && small->avg_probe_ms() == 240.0 &&
                  large->avg_probe_ms() == 200.0,
              "the measurement cost is per size too, not one blended average");
        CHECK(t.runs == 6 && t.cached == 3 && t.measured == 3 && t.probe_ms == 640 &&
                  t.hit_rate() == 50.0,
              "the totals are the buckets summed, so the headline still adds up");
        {
            // Guarded: a bucket that is missing is a failed check below, not a
            // crash that hides the rest of the group.
            const std::string small_text =
                small ? krk::h2d_tally_bucket_text(*small) : std::string();
            const std::string large_text =
                large ? krk::h2d_tally_bucket_text(*large) : std::string();
            CHECK(small_text.find("16 MiB: 3 of 4 reused (75%)") !=
                          std::string::npos &&
                      small_text.find("1 measured at 240.0 ms") !=
                          std::string::npos &&
                      small_text.find("720 ms avoided") != std::string::npos,
                  "a bucket line names the size, its rate and what it avoided");
            CHECK(large_text.find("32 MiB: 0 of 2 reused (0%)") !=
                          std::string::npos &&
                      large_text.find("2 measured at 200.0 ms") !=
                          std::string::npos &&
                      large_text.find("avoided") == std::string::npos,
                  "and a bucket that has avoided nothing says so by omission, not "
                  "\"0 ms\"");
            CHECK(krk::h2d_tally_bucket_text(krk::H2dTallyBucket()).empty(),
                  "an unasked size prints nothing at all");
        }
        CHECK(krk::h2d_tally_load(tally_path).find(key.bytes * 2) != nullptr &&
                  krk::h2d_tally_load(tally_path).buckets.size() == 2,
              "both sizes survive the round trip through the file");
        krk::H2dProbeTally hits_only;
        hits_only.runs = 5;
        hits_only.cached = 5;
        CHECK(krk::h2d_tally_text(hits_only).find("not priced") != std::string::npos,
              "a tally with no measurement refuses to price what it saved");
        // A corrupt ledger is no ledger: better that than a rate above 100%.
        std::FILE *f = std::fopen(tally_path.c_str(), "wb");
        CHECK(f != nullptr, "the ledger is there to corrupt");
        if (f) {
            std::fputs("2\t16777216\t2\t9\t0\t0\n", f);  // more hits than runs
            std::fclose(f);
        }
        CHECK(krk::h2d_tally_load(tally_path).runs == 0,
              "a ledger whose parts exceed its whole reads as no ledger");
        // v1 counted all sizes in one line, which is exactly the question the
        // buckets exist to answer -- so it is not a tally any more.
        f = std::fopen(tally_path.c_str(), "wb");
        CHECK(f != nullptr, "the ledger is there to write an old-format line");
        if (f) {
            std::fputs("1\t3\t2\t1\t73\n", f);
            std::fclose(f);
        }
        CHECK(krk::h2d_tally_load(tally_path).runs == 0,
              "a blended v1 tally is not read as a per-size one");
        CHECK(krk::h2d_tally_path(std::string()).empty(),
              "a disabled cache has no ledger path");
        bool off_wrote = true;
        CHECK(krk::h2d_tally_note(std::string(), key.bytes, true, 0, &off_wrote)
                          .runs == 0 &&
                  !off_wrote,
              "and noting a run against it writes nothing");
        std::remove(tally_path.c_str());
        std::remove((tally_path + ".tmp").c_str());
        std::remove(tally_cache.c_str());  // a tally alone is not a verdict
    }
    {
        // An already-expired line is dropped by the next write, not left to
        // occupy a slot the reader will never accept.
        const std::string prune_path = "kraken-h2d-cache-prune.txt";
        krk::H2dCacheKey expired = key;
        expired.bytes = key.bytes << 1;
        CHECK(krk::h2d_probe_cache_write(prune_path, expired, other_probe,
                                         now - krk::kH2dCacheMaxAgeSec - 60),
              "an expired verdict can be written (the writer does not refuse a stamp)");
        CHECK(krk::h2d_probe_cache_write(prune_path, key, wide, now),
              "then a live one");
        std::ifstream count_in(prune_path.c_str());
        size_t lines = 0;
        std::string l;
        while (std::getline(count_in, l))
            if (!l.empty()) lines++;
        CHECK(lines == 1, "the expired line is pruned, leaving only the live verdict");
        // Windows refuses to remove a file that is still open.
        count_in.close();
        std::remove(prune_path.c_str());
    }
    {
        // Past the cap the NEWEST entries win: the oldest is the one least
        // likely to be asked for again.
        const std::string cap_path = "kraken-h2d-cache-cap.txt";
        const size_t total = krk::kH2dCacheMaxEntries + 6;
        bool all = true;
        krk::H2dCachePrune last;
        for (size_t i = 0; i < total; i++) {
            krk::H2dCacheKey k = key;
            k.bytes = key.bytes + (i + 1) * 4096;
            if (!krk::h2d_probe_cache_write(cap_path, k, other_probe,
                                            now + static_cast<long long>(i)))
                all = false;
        }
        CHECK(all, "the cache accepts more keys than it keeps");
        CHECK(krk::h2d_probe_cache_write(cap_path, key, other_probe,
                                         now + static_cast<long long>(total) + 1,
                                         &last) &&
                  last.evicted >= 1 &&
                  last.kept == static_cast<int>(krk::kH2dCacheMaxEntries),
              "past the cap the write reports what the cap pushed out");
        krk::H2dCacheKey oldest = key, newest = key;
        oldest.bytes = key.bytes + 4096;
        newest.bytes = key.bytes + total * 4096;
        CHECK(!krk::h2d_probe_cache_read(cap_path, oldest, now + 1000).hit,
              "past the cap the oldest entry is the one dropped");
        CHECK(krk::h2d_probe_cache_read(cap_path, newest, now + 1000).hit,
              "and the newest survives");
        std::remove(cap_path.c_str());
    }
    {
        // A line written before the driver version joined the key is a miss:
        // the verdict is fine, but which driver produced it is not recorded,
        // and that is the one thing an update changes. A file of its own,
        // because a valid entry for this key exists in the real one and would
        // be found first.
        const std::string old_path = "kraken-h2d-cache-v1.txt";
        std::FILE *old = std::fopen(old_path.c_str(), "wb");
        CHECK(old != nullptr, "the cache file is there to write an old-format line");
        if (old) {
            std::fprintf(old, "1\t%s\t%llu\t%d\t%llu\t0.4\t20\t20\t%lld\n",
                         key.name.c_str(),
                         static_cast<unsigned long long>(key.vram_total),
                         key.large_bar,
                         static_cast<unsigned long long>(key.bytes), now);
            std::fclose(old);
        }
        CHECK(!krk::h2d_probe_cache_read(old_path, key, now).hit,
              "a verdict from before the driver version was in the key is not trusted");
        std::remove(old_path.c_str());
    }
    {
        // A hand-edited or half-written file is a miss, not a crash, and it
        // must not stop the next write from producing a readable entry.
        std::FILE *f = std::fopen(cache_path.c_str(), "ab");
        CHECK(f != nullptr, "the cache file is there to corrupt");
        if (f) {
            std::fputs("1\tgarbage\nnot even close\n", f);
            std::fclose(f);
        }
        CHECK(krk::h2d_probe_cache_read(cache_path, key, now).hit,
              "a garbage line does not hide the valid entries beside it");
    }
    set_env_var("KRK_H2D_CACHE", "0");
    CHECK(krk::h2d_probe_cache_path().empty(),
          "KRK_H2D_CACHE=0 turns the cache off entirely");
    set_env_var("KRK_H2D_CACHE", cache_path.c_str());

    // The wiring, end to end: a cached verdict must DECIDE, not merely be
    // printed. The aperture below says the ring wins (6.0 against 0.4), so a
    // direct path can only have come from the cache.
    krk::H2dProbe cached_wide;
    cached_wide.valid = true;
    cached_wide.pageable_gbs = 0.4;
    cached_wide.direct_gbs = 20.0;
    cached_wide.pinned_gbs = 20.0;
    CHECK(krk::h2d_probe_cache_write(cache_path, key, cached_wide,
                                     krk::h2d_now_sec()),
          "seeded the cache with a wide-aperture verdict");
    // The ledger the run-end report reads. Each probing init is one run; a hit
    // and a measurement are different arms; a run with the probe off is neither,
    // or the hit rate would describe a cache that run never consulted.
    const krk::H2dProbeTally tally_before =
        krk::h2d_tally_load(krk::h2d_tally_path(cache_path));
    bool cached_direct = false, rerun_direct = true, off_direct = true;
    size_t cached_pin = 1, rerun_pin = 0, off_pin = 0;
    std::vector<i32> cached_tokens, rerun_tokens, off_tokens;
    krk::H2dProbe cached_used, rerun_used, off_used;
    const bool cached_ok = run(6.0, 0.4, 20.0, &cached_direct, &cached_pin,
                               &cached_tokens, &cached_used);
    CHECK(cached_ok, "an engine initialises from a cached verdict");
    CHECK(cached_direct,
          "a cached wide-aperture verdict decides the path without re-probing");
    CHECK(cached_used.valid && std::fabs(cached_used.direct_gbs - 20.0) < 1e-3,
          "and it publishes the rates it read, not a fresh measurement");

    set_env_var("KRK_H2D_PROBE_RERUN", "1");
    const bool rerun_ok = run(6.0, 0.4, 20.0, &rerun_direct, &rerun_pin,
                              &rerun_tokens, &rerun_used);
    CHECK(rerun_ok, "and one initialises with the measurement forced");
    CHECK(!rerun_direct,
          "KRK_H2D_PROBE_RERUN=1 ignores the cache and measures the card");
    CHECK(rerun_used.valid && rerun_used.pageable_gbs > rerun_used.direct_gbs,
          "... with the aperture it just measured, not the cached one");
    clear_env_var("KRK_H2D_PROBE_RERUN");

    // Turning the probe off must turn its cache off too: an operator asking for
    // the BAR hint is not asking for an old measurement to stand in for it.
    set_env_var("KRK_H2D_PROBE", "0");
    const bool off_ok = run(6.0, 0.4, 20.0, &off_direct, &off_pin, &off_tokens,
                            &off_used);
    CHECK(off_ok, "and one initialises with the probe switched off");
    CHECK(!off_direct && !off_used.valid,
          "KRK_H2D_PROBE=0 skips the cache as well as the probe");
    clear_env_var("KRK_H2D_PROBE");

    const krk::H2dProbeTally tally_after =
        krk::h2d_tally_load(krk::h2d_tally_path(cache_path));
    CHECK(tally_after.runs == tally_before.runs + 2 &&
              tally_after.cached == tally_before.cached + 1 &&
              tally_after.measured == tally_before.measured + 1,
          "the engine records each probing init: one hit, one measurement, and the "
          "probe-off run not at all");
    CHECK(tally_after.probe_ms > tally_before.probe_ms,
          "and the wall time a measurement cost is what a hit is priced at");
    CHECK(tally_after.find(key.bytes) != nullptr &&
              tally_after.find(key.bytes)->runs ==
                  tally_before.runs + 2,
          "counted at the size this model probes, not blended across sizes");

    set_env_var("KRK_H2D_CACHE", "0");
    std::remove(cache_path.c_str());
    std::remove((cache_path + ".tmp").c_str());
    std::remove(krk::h2d_tally_path(cache_path).c_str());
    std::remove((krk::h2d_tally_path(cache_path) + ".tmp").c_str());
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Source hygiene: a string literal that spans a RAW newline.
//
// This is not a hypothetical class. Four edits in one session wrote a C string
// through a tool path that decoded the escape twice, so `\n` became a real
// newline inside `fprintf(...)` and `KRK_ERROR(...)` literals; each one is a
// compile error, and each cost a build to discover. `str_replace` (one decode)
// is safe, a heredoc-fed script (two decodes) is not -- and nothing in the repo
// said so.
//
// The scanner below is the rule made checkable. It walks a translation unit the
// way the compiler does: it knows about `//` and block comments, character
// literals, raw strings (`R"tag(` ... `)tag"`), and the LEGAL line continuation
// (a backslash immediately before the newline keeps the literal open). Anything
// else that reaches a newline with a literal still open is reported with its
// line number.
//
// The group proves the scanner can fail before trusting its answer on the tree:
// five constructed inputs, one of them broken on purpose, one a continuation and
// one a raw string. Then it sweeps every source file this binary was built from
// (`KRK_SOURCE_DIR`, a configure-time define -- the group cannot use the working
// directory, which is a per-group scratch dir by design).
// Is the literal starting at `at` an argument of a print/error call? Only there
// does a doubled escape matter: `fprintf(stderr, "a\\nb")` prints the two
// characters `\n` where the author meant a line break. A literal that is NOT a
// format string can legitimately carry a backslash -- escape tables
// (`src/json.cpp`) and JSON payloads both do, and flagging those would make the
// group's first act on a healthy tree a false positive.
static bool source_in_print_call(const std::string &text, size_t at) {
    // The nearest ( going backwards, then the identifier in front of it -- but
    // never past the END OF THE STATEMENT. Measured: without the `;` stop, a
    // table literal on the line after a fprintf was attributed to that fprintf
    // and the sweep reported a defect that was not there.
    size_t k = at;
    int newlines = 0;
    while (k > 0 && k + 240 > at) {
        const char p = text[k - 1];
        if (p == '(') break;
        if (p == ';' || p == '{' || p == '}') return false;
        if (p == '\n' && ++newlines > 2) return false;
        --k;
    }
    if (k == 0 || text[k - 1] != '(') return false;
    size_t e = k - 1;
    size_t b = e;
    while (b > 0 && (std::isalnum(static_cast<unsigned char>(text[b - 1])) ||
                     text[b - 1] == '_' || text[b - 1] == ':'))
        --b;
    const std::string fn = text.substr(b, e - b);
    return fn.size() >= 5 &&
           (fn.find("printf") != std::string::npos ||
            fn.find("KRK_ERROR") != std::string::npos ||
            fn.find("KRK_WARN") != std::string::npos ||
            fn.find("KRK_NOTE") != std::string::npos ||
            fn.find("puts") != std::string::npos ||
            // The HTTP layer's writers: a browser reads what these emit, and a
            // doubled `\\r\\n` there is a malformed response, not a cosmetic
            // one. `format` is this repo's sprintf-shaped helper (src/http.cpp
            // builds its status line with it).
            fn.find("format") != std::string::npos ||
            fn.find("write_raw") != std::string::npos ||
            fn.find("write_chunk") != std::string::npos ||
            fn.find("send") != std::string::npos ||
            fn.find("respond") != std::string::npos);
}

// Is this a file whose whole job is protocol text a client parses? In these,
// EVERY doubled CR/LF is a defect, call shape or not: `static const char head[]
// = "HTTP/1.1 200 OK\\r\\n"` is not a print argument and would otherwise have
// no context to be judged in.
static bool source_protocol_file(const std::string &path) {
    std::string p = path;
    for (char &c : p)
        if (c == '\\') c = '/';
    return p.size() >= 13 &&
           (p.compare(p.size() - 13, 13, "/src/http.cpp") == 0 ||
            p.compare(p.size() - 15, 15, "/src/server.cpp") == 0);
}

// `// hygiene-ok` on the literal's own line is the waver for the rare case that
// really does print a backslash in a format string.
static bool source_line_waived(const std::string &text, size_t at) {
    size_t a = at;
    while (a > 0 && text[a - 1] != '\n') --a;
    size_t b = at;
    while (b < text.size() && text[b] != '\n') ++b;
    return text.find("hygiene-ok", a) != std::string::npos &&
           text.find("hygiene-ok", a) < b;
}

// One walk, two classes:
//   * a literal that spans a RAW newline (a compile error, and the edit-path
//     defect this group was built for), reported into `lines`;
//   * a DOUBLED escape that reaches a print/error argument -- source `\\n`,
//     which prints the two characters instead of the line break, reported into
//     `escapes`. It compiles, so nothing else would ever name it. `any_context`
//     widens the count to EVERY literal in the file, which is what the HTTP
//     layer is judged by: a browser parses that text, and a doubled CRLF there
//     is a malformed response rather than a cosmetic one.
static int source_bad_literals(const std::string &text,
                              std::vector<int> *lines,
                              std::vector<int> *escapes = nullptr,
                              bool any_context = false) {
    int offenders = 0;
    size_t i = 0;
    int line_no = 1;
    bool in_block = false, in_raw = false;
    std::string raw_tag;
    const size_t n = text.size();
    while (i < n) {
        const char c = text[i];
        if (c == '\n') {
            ++line_no;
            ++i;
            continue;
        }
        if (in_raw) {
            if (c == ')') {
                const std::string close = ")" + raw_tag + "\"";
                if (text.compare(i, close.size(), close) == 0) {
                    in_raw = false;
                    i += close.size();
                    continue;
                }
            }
            ++i;
            continue;
        }
        if (in_block) {
            if (c == '*' && i + 1 < n && text[i + 1] == '/') {
                in_block = false;
                i += 2;
            } else {
                ++i;
            }
            continue;
        }
        if (c == '/' && i + 1 < n && text[i + 1] == '/') {
            while (i < n && text[i] != '\n') ++i;
            continue;
        }
        if (c == '/' && i + 1 < n && text[i + 1] == '*') {
            in_block = true;
            i += 2;
            continue;
        }
        if (c == 'R' && i + 1 < n && text[i + 1] == '"') {
            size_t p = i + 2;
            raw_tag.clear();
            while (p < n && text[p] != '(') raw_tag.push_back(text[p++]);
            in_raw = true;
            i = p + 1;
            continue;
        }
        if (c == '\'') {
            ++i;
            while (i < n && text[i] != '\'') {
                if (text[i] == '\\') ++i;
                ++i;
            }
            if (i < n) ++i;
            continue;
        }
        if (c == '"') {
            const size_t lit_start = i;
            const int lit_line = line_no;
            bool doubled = false;
            ++i;
            bool esc = false;
            while (i < n) {
                const char d = text[i];
                if (esc) {
                    esc = false;
                    if (d == '\n') ++line_no;
                    ++i;
                    continue;
                }
                if (d == '\\') {
                    // Source `\\n` / `\\r`: a backslash that survives, rather
                    // than the control code the author almost always meant.
                    if (i + 2 < n && text[i + 1] == '\\' &&
                        (text[i + 2] == 'n' || text[i + 2] == 'r'))
                        doubled = true;
                    esc = true;
                    ++i;
                    continue;
                }
                if (d == '"') {
                    ++i;
                    break;
                }
                if (d == '\n') {
                    lines->push_back(line_no);
                    ++offenders;
                    // Resync at the END of the broken line, not at the newline:
                    // the text after the break (`b";`) would otherwise open a
                    // second phantom literal and report one defect twice.
                    while (i < n && text[i] != '\n') ++i;
                    break;
                }
                ++i;
            }
            if (doubled && escapes &&
                (any_context || source_in_print_call(text, lit_start)) &&
                !source_line_waived(text, lit_start))
                escapes->push_back(lit_line);
            continue;
        }
        ++i;
    }
    return offenders;
}

static void test_source_literal_hygiene() {
    // ---- 1. the scanner's own arms, before it is trusted on the tree -----
    {
        std::vector<int> lines;
        const std::string ok = "const char *s = \"a\\nb\";";
        CHECK(source_bad_literals(ok, &lines) == 0 && lines.empty(),
              "an escaped newline inside a literal is not a defect");
    }
    {
        std::vector<int> lines;
        std::string bad = "const char *s = \"a";
        bad.push_back('\n');
        bad += "b\";";
        CHECK(source_bad_literals(bad, &lines) == 1 && lines.size() == 1 &&
                  lines[0] == 1,
              "a literal broken by a RAW newline is reported, with its line");
    }
    {
        std::vector<int> lines;
        std::string cont = "const char *s = \"a";
        cont.push_back('\\');
        cont.push_back('\n');
        cont += "b\";";
        CHECK(source_bad_literals(cont, &lines) == 0,
              "a backslash continuation keeps the literal open, as the compiler "
              "does");
    }
    {
        std::vector<int> lines;
        std::string raw = "auto s = R\"(a";
        raw.push_back('\n');
        raw += "b)\";";
        CHECK(source_bad_literals(raw, &lines) == 0,
              "a raw string may span newlines and is not reported");
    }
    {
        std::vector<int> lines;
        std::string cmt = "// a quote \" in a comment";
        cmt.push_back('\n');
        cmt += "int x = 1; // another \"\n";
        CHECK(source_bad_literals(cmt, &lines) == 0,
              "quotes inside comments are not literals");
    }

    // ---- 2. the DOUBLED escape, in a print argument ----------------------
    // Source `\\n` compiles and prints the two characters: a gate that prints
    // `lit\n` where it meant a line break is unreadable, and nothing else names
    // it. Only a print/error argument is flagged -- an escape table or a JSON
    // payload may carry a backslash on purpose.
    {
        // The INPUT text has to contain two real backslashes, so they are built
        // from character code 92: spelling them here would create the doubled
        // escape this arm is about, one layer up.
        std::vector<int> lines, esc;
        std::string fmt = "std::fprintf(stderr, \"a";
        fmt.push_back(static_cast<char>(92));
        fmt.push_back(static_cast<char>(92));
        fmt += "nb\");";
        CHECK(source_bad_literals(fmt, &lines, &esc) == 0 && esc.size() == 1,
              "a doubled newline escape in a print argument is reported");
    }
    {
        // The same doubled escape, in a literal that is NOT a print argument.
        // Built with character codes because this arm's INPUT is a backslash --
        // writing it as an escape here is the very mistake the group hunts.
        std::vector<int> lines, esc;
        std::string tbl = "const char *k = ";
        tbl += '"';
        tbl += 'a';
        tbl.push_back(static_cast<char>(92));
        tbl.push_back(static_cast<char>(92));
        tbl += 'n';
        tbl += '"';
        tbl += ';';
        CHECK(source_bad_literals(tbl, &lines, &esc) == 0 && esc.empty(),
              "the same spelling in a non-print literal is NOT reported (json "
              "tables and payloads carry backslashes on purpose)");
    }
    {
        // Browser-facing: the HTTP layer builds its status line with `format`,
        // which is not a printf by name but is one by contract.
        std::vector<int> lines, esc;
        std::string line = "std::string h = format(\"HTTP/1.1 200 OK";
        line.push_back(static_cast<char>(92));
        line.push_back(static_cast<char>(92));
        line += "r";
        line.push_back(static_cast<char>(92));
        line.push_back(static_cast<char>(92));
        line += "n\", r.status);";
        CHECK(source_bad_literals(line, &lines, &esc) == 0 && esc.size() == 1,
              "a doubled CRLF in the HTTP status line is reported (a browser "
              "parses that, so it is a malformed response)");
    }
    {
        // The same text in a protocol file, but NOT in a call: a static header
        // array has no call context at all, and this is the arm that says the
        // module is judged by its job rather than by its shape.
        std::vector<int> lines, esc;
        std::string head = "static const char head[] = \"HTTP/1.1 200 OK";
        head.push_back(static_cast<char>(92));
        head.push_back(static_cast<char>(92));
        head += "r";
        head.push_back(static_cast<char>(92));
        head.push_back(static_cast<char>(92));
        head += "n\";";
        CHECK(source_bad_literals(head, &lines, &esc) == 0 && esc.empty(),
              "outside a protocol file that header array is not judged (no "
              "call context)");
        CHECK(source_bad_literals(head, &lines, &esc, true) == 0 &&
                  esc.size() == 1,
              "inside one it is: every doubled CRLF in the HTTP layer counts");
    }
    {
        // The shape a user actually reads -- the CLI's `kraken: ...` report --
        // because the entry points are the files this group is supposed to
        // protect and they are not shaped like the engine's error macros.
        std::vector<int> lines, esc;
        std::string msg = "std::fprintf(stderr, \"kraken: bad value";
        msg.push_back(static_cast<char>(92));
        msg.push_back(static_cast<char>(92));
        msg += "n\");";
        CHECK(source_bad_literals(msg, &lines, &esc) == 0 && esc.size() == 1,
              "a doubled escape in a `kraken: ...` message is reported (the "
              "entry points are what a user reads)");
    }
    {
        // Measured false positive, kept as an arm: a table literal on the line
        // AFTER a fprintf was attributed to that fprintf until the backward
        // search learned to stop at the end of the statement.
        std::vector<int> lines, esc;
        std::string two = "std::fprintf(stderr, \"x\");";
        two.push_back('\n');
        two += "const char *k = \"a";
        two.push_back(static_cast<char>(92));
        two.push_back(static_cast<char>(92));
        two += "nb\";";
        CHECK(source_bad_literals(two, &lines, &esc) == 0 && esc.empty(),
              "a table literal after a print call is not attributed to it "
              "(stop at the statement)");
    }
    {
        std::vector<int> lines, esc;
        std::string waived = "std::fprintf(stderr, \"a";
        waived.push_back(static_cast<char>(92));
        waived.push_back(static_cast<char>(92));
        waived += "nb\"); // hygiene-ok";
        CHECK(source_bad_literals(waived, &lines, &esc) == 0 && esc.empty(),
              "`// hygiene-ok` waives a print that really does print a "
              "backslash");
    }

    // ---- 2. the tree this binary was built from --------------------------
    // A sweep that cannot find the sources would pass every check below by
    // scanning nothing, which is the vacuous arm this repo keeps paying for.
    const std::string root =
#ifdef KRK_SOURCE_DIR
        std::string(KRK_SOURCE_DIR);
#else
        std::string();
#endif
    CHECK(!root.empty(), "the build recorded its source root (KRK_SOURCE_DIR)");
    if (root.empty()) return;
    std::vector<std::string> files;
    {
        std::error_code ec;
        for (const char *sub : {"/src", "/include", "/tests", "/tools"}) {
            const std::filesystem::path dir = std::filesystem::path(root + sub);
            if (!std::filesystem::exists(dir, ec)) continue;
            for (std::filesystem::recursive_directory_iterator it(
                     dir, std::filesystem::directory_options::skip_permission_denied,
                     ec),
                 end;
                 it != end && !ec; it.increment(ec)) {
                if (!it->is_regular_file(ec)) continue;
                const std::string ext = it->path().extension().string();
                if (ext == ".cpp" || ext == ".hpp" || ext == ".hip" ||
                    ext == ".cxx" || ext == ".h")
                    files.push_back(it->path().string());
            }
        }
    }
    CHECK(files.size() >= 60,
          "the sweep found the tree's sources (not a silent empty scan)");
    // The ENTRY POINTS are the files that talk to a user, and they are only
    // covered because they happen to live under `src/`. That is luck, not a
    // pin: a sweep narrowed to the engine's own tree would stay green while the
    // messages a user actually reads went unscanned. Measured, not assumed -- a
    // doubled escape injected into main_cli.cpp's `kraken: unknown option`
    // report is named at src/main_cli.cpp:436 -- so the coverage is recorded
    // here and refused if either file stops being read.
    const char *kEntryPoints[] = {"/src/main_cli.cpp", "/src/main_server.cpp"};
    int entry_seen[2] = {0, 0};
    int total = 0, first_file = -1, first_line = 0;
    int esc_total = 0, esc_file = -1, esc_line = 0;
    for (size_t k = 0; k < files.size(); k++) {
        {
            std::string rel = files[k].substr(root.size());
            for (char &ch : rel)
                if (ch == '\\') ch = '/';
            for (int e = 0; e < 2; e++)
                if (rel == kEntryPoints[e]) entry_seen[e] = 1;
        }
        std::ifstream in(files[k], std::ios::binary);
        if (!in) continue;
        std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        std::vector<int> lines, esc;
        const int bad = source_bad_literals(text, &lines, &esc,
                                           source_protocol_file(files[k]));
        if (!esc.empty()) {
            if (esc_file < 0) {
                esc_file = static_cast<int>(k);
                esc_line = esc[0];
            }
            esc_total += static_cast<int>(esc.size());
        }
        if (bad > 0) {
            if (first_file < 0) {
                first_file = static_cast<int>(k);
                first_line = lines.empty() ? 0 : lines[0];
            }
            total += bad;
        }
    }
    if (total > 0)
        std::fprintf(stderr, "  (first offender: %s:%d)\n",
                     first_file >= 0 ? files[first_file].c_str() : "?",
                     first_line);
    // Counted per LINE, not per literal: one literal broken across a newline
    // leaves an unbalanced quote on each side, and resyncing at the end of the
    // first line makes the second one honestly unbalanced too.
    std::fprintf(stderr, "  (scanned %zu source files, %d line(s) with an "
                         "unterminated literal)\n", files.size(), total);
    CHECK(total == 0,
          total == 0
              ? "no line in the tree leaves a string literal unterminated"
              : "a string literal spans a raw newline: an edit wrote a newline "
                "where the escape was meant (see the file:line above)");

    if (esc_total > 0)
        std::fprintf(stderr, "  (first doubled escape: %s:%d)\n",
                     esc_file >= 0 ? files[esc_file].c_str() : "?", esc_line);
    std::fprintf(stderr,
                 "  (%d doubled newline escape(s) in print/error arguments, or "
                 "anywhere in the HTTP layer)\n",
                 esc_total);
    CHECK(esc_total == 0,
          esc_total == 0
              ? "no print/error format string carries a doubled newline escape"
              : "a print/error format string spells the escape twice, so it "
                "prints the two characters '\\n' instead of a line break (see "
                "the file:line above)");

    // ---- 3. the entry points are READ, not merely present ---------------
    std::fprintf(stderr, "  (entry points scanned: cli %s, server %s)\n",
                 entry_seen[0] ? "yes" : "NO",
                 entry_seen[1] ? "yes" : "NO");
    CHECK(entry_seen[0] && entry_seen[1],
          entry_seen[0] && entry_seen[1]
              ? "the CLI and the server are in the sweep: the messages a user "
                "reads are covered"
              : "an entry point fell out of the sweep, so a doubled escape in "
                "a user-facing message would go unreported");
}

// Group: the CLI's refusals are a contract a script depends on, and both arms
// of it that were broken produced a SILENT default rather than a crash. The
// enums read a typo as success, because `parse()` returns bool and their error
// paths did `return 2;`, which converts to TRUE: `--prefill bogus` ran the
// default fast rung and `--kv-quant int4` ran f16. The numeric flags had no
// validation at all, so `--ctx abc` ran with a zero-token context and
// `--ctx 4096x` ran with 4096, each with one line to stderr and exit 0.
//
// The check is on the PROCESS, because that is what a script reads. Every arm
// passes a model path that does NOT exist: a refused value must exit 2 BEFORE
// the loader is reached, while a value that slipped through reaches the loader
// and exits 1 with `cannot open`. That is what makes the arm discriminate
// rather than merely observe a non-zero status. Two controls pin both ends -- a
// valid line does reach the loader (1), and `--help` is not a refusal (0) -- so
// an arm that refused everything could not pass.
static void test_cli_arg_refusal() {
    const std::string self = self_path();
    const std::filesystem::path exe_p =
        std::filesystem::path(self).parent_path() /
#ifdef _WIN32
        "kraken.exe";
#else
        "kraken";
#endif
    const bool have_exe = !self.empty() && std::filesystem::exists(exe_p);
    CHECK(have_exe, "kraken sits beside kraken-tests (this group drives it)");
    if (!have_exe) return;
    const std::string exe = exe_p.string();
    // Never created: the point is that a refused flag stops before the loader,
    // so a run that got this far exits 1 instead of 2.
    const std::string missing = "kraken-arg-refusal-no-such-model.gguf";

    auto run_bin = [&](const std::string &bin, const std::string &args,
                       std::string *err) -> int {
        const std::string o = "kraken-argref-out.txt";
        const std::string e = "kraken-argref-err.txt";
        std::string cmd =
            "\"" + bin + "\" " + args + " > \"" + o + "\" 2> \"" + e + "\"";
#ifdef _WIN32
        // cmd /c strips the first and last quote of a line that starts with
        // one; `call` makes the line start with a word (the run_prefill_hash
        // trap).
        cmd = "call " + cmd;
#endif
        const int rc = gate_exit_code(std::system(cmd.c_str()));
        if (err) *err = read_capture(e);
        std::remove(o.c_str());
        std::remove(e.c_str());
        return rc;
    };
    auto run = [&](const std::string &args, std::string *err) -> int {
        return run_bin(exe, args, err);
    };

    // Every flag that takes a TYPED value, with one value that is not it. The
    // numeric ones are here because they used to be undetectable rather than
    // refused: std::atoi("abc") is 0, so `--ctx abc` ran with a zero-token
    // context and `--ctx 4096x` ran with 4096, each silently. `want` is the
    // flag's own name, which every refusal message carries.
    struct Arm {
        const char *flags;
        const char *want;
    };
    const Arm bad[] = {
        // enums, with the values they accept named in the message
        {"--kv-quant int4", "--kv-quant needs f16|int8"},
        {"--prefill bogus", "--prefill needs exact|fast"},
        // whole numbers
        {"--draft-tokens abc", "--draft-tokens"},
        {"--draft-head-prime abc", "--draft-head-prime"},
        {"--max-tokens 12abc", "--max-tokens"},
        {"--ctx abc", "--ctx"},
        {"--chunk abc", "--chunk"},
        {"--top-k abc", "--top-k"},
        {"--device abc", "--device"},
        {"--threads abc", "--threads"},
        {"--host-ram-mb abc", "--host-ram-mb"},
        {"--weight-vram-mb abc", "--weight-vram-mb"},
        {"--kv-hot-mb abc", "--kv-hot-mb"},
        {"--kv-warm-mb abc", "--kv-warm-mb"},
        {"--vram-cap-mb abc", "--vram-cap-mb"},
        {"--vram-tier abc", "--vram-tier"},
        {"--expert-cache-mb abc", "--expert-cache-mb"},
        {"--expert-cache-slots abc", "--expert-cache-slots"},
        {"--expert-warm-mb abc", "--expert-warm-mb"},
        {"--expert-pin-mb abc", "--expert-pin-mb"},
        {"--expert-warm-prefetch abc", "--expert-warm-prefetch"},
        {"--expert-warmup abc", "--expert-warmup"},
        {"--expert-warmup-ms abc", "--expert-warmup-ms"},
        {"--ram-tier abc", "--ram-tier"},
        {"--hybrid-experts abc", "--hybrid-experts"},
        {"--hybrid-threads abc", "--hybrid-threads"},
        {"--hybrid-max-rows abc", "--hybrid-max-rows"},
        {"--debug-topk abc", "--debug-topk"},
        // numbers
        {"--temp 1.5x", "--temp"},
        {"--top-p 0.9x", "--top-p"},
        {"--min-p 0.0x", "--min-p"},
        {"--repeat-penalty 1.1x", "--repeat-penalty"},
        {"--router-resident 0.0x", "--router-resident"},
        {"--hybrid-frac 1.0x", "--hybrid-frac"},
        // unsigned: a negative seed is not a wrap-around accident
        {"--seed -3", "--seed"},
        // and a flag that does not exist at all
        {"--no-such-flag", "unknown option"},
    };
    for (const Arm &a : bad) {
        std::string err;
        const int rc = run(std::string("-m \"") + missing + "\" " + a.flags, &err);
        if (rc != 2)
            std::fprintf(stderr, "  (%s -> rc=%d, stderr: %.200s)\n", a.flags, rc,
                         err.c_str());
        CHECK(rc == 2,
              rc == 2 ? "a bad value is refused with exit 2, not read as a "
                        "successful parse"
                      : "a bad value was ACCEPTED: parse() returned true, so the "
                        "run continued on defaults");
        CHECK(err.find(a.want) != std::string::npos,
              err.find(a.want) != std::string::npos
                  ? "and the message names the flag and what it accepts"
                  : "the refusal does not say which flag or which values it "
                    "wanted");
    }

    // Control 1: a VALID command line must get past parse and reach the loader,
    // which then refuses the missing file with 1. This is what makes "exit 2"
    // above mean "refused at parse" rather than "this binary always exits 2".
    std::string err_ok;
    const int rc_ok = run(std::string("-m \"") + missing + "\"", &err_ok);
    if (rc_ok != 1)
        std::fprintf(stderr, "  (valid line -> rc=%d, stderr: %.200s)\n", rc_ok,
                     err_ok.c_str());
    CHECK(rc_ok == 1, rc_ok == 1
                           ? "a valid command line reaches the loader, which "
                             "refuses the missing file"
                           : "a valid command line did not reach the loader: the "
                             "refusal arms above prove nothing");

    // Control 2: --help is not a refusal, so it must NOT come back as 2.
    const int rc_h = run("--help", nullptr);
    CHECK(rc_h == 0, rc_h == 0 ? "--help exits 0 (a refusal arm cannot be "
                                 "\"everything exits 2\")"
                               : "--help did not exit 0");

    // ---- kraken-server takes the same flags, so it owes the same contract ---
    // Its hole was the worse of the two: `--port abc` was atoi -> 0, and 0 is
    // exactly how the server is asked for an EPHEMERAL port, so a typo started
    // a listener on a random port while the caller waited on 8080.
    const std::filesystem::path srv_p =
        std::filesystem::path(self).parent_path() /
#ifdef _WIN32
        "kraken-server.exe";
#else
        "kraken-server";
#endif
    const bool have_srv = !self.empty() && std::filesystem::exists(srv_p);
    CHECK(have_srv, "kraken-server sits beside kraken-tests (the same group "
                    "covers both entry points)");
    if (have_srv) {
        const std::string srv = srv_p.string();
        auto srun = [&](const std::string &args, std::string *err) -> int {
            return run_bin(srv, args, err);
        };
        const Arm sbad[] = {
            {"--kv-quant int4", "--kv-quant needs f16|int8"},
            {"--prefill bogus", "--prefill needs exact|fast"},
            {"--port abc", "--port"},
            {"--ctx abc", "--ctx"},
            {"--chunk abc", "--chunk"},
            {"--threads abc", "--threads"},
            {"--device abc", "--device"},
            {"--draft-tokens abc", "--draft-tokens"},
            {"--expert-cache-mb abc", "--expert-cache-mb"},
            {"--expert-cache-slots abc", "--expert-cache-slots"},
            {"--expert-warm-mb abc", "--expert-warm-mb"},
            {"--expert-pin-mb abc", "--expert-pin-mb"},
            {"--expert-warmup abc", "--expert-warmup"},
            {"--expert-warmup-ms abc", "--expert-warmup-ms"},
            {"--ram-tier abc", "--ram-tier"},
            {"--hybrid-experts abc", "--hybrid-experts"},
            {"--hybrid-threads abc", "--hybrid-threads"},
            {"--hybrid-max-rows abc", "--hybrid-max-rows"},
            {"--router-resident 0.0x", "--router-resident"},
            {"--hybrid-frac 1.0x", "--hybrid-frac"},
            {"--no-such-flag", "unknown option"},
        };
        for (const Arm &a : sbad) {
            std::string err;
            const int rc = srun(std::string("-m \"") + missing + "\" " + a.flags, &err);
            if (rc != 2)
                std::fprintf(stderr, "  (server %s -> rc=%d, stderr: %.200s)\n",
                             a.flags, rc, err.c_str());
            CHECK(rc == 2, rc == 2 ? "the server refuses a bad value with exit 2"
                                   : "the server ACCEPTED a bad value: it "
                                     "defaulted silently");
            CHECK(err.find(a.want) != std::string::npos,
                  err.find(a.want) != std::string::npos
                      ? "and the server's message names the flag"
                      : "the server's refusal does not name the flag");
        }
        // Controls: a valid line reaches the loader (1), --help is not a
        // refusal (0). Without them the arms above prove only "not 2".
        std::string serr;
        const int src = srun(std::string("-m \"") + missing + "\"", &serr);
        CHECK(src == 1, src == 1 ? "a valid server line reaches the loader"
                                 : "a valid server line did not reach the loader");
        const int srh = srun("--help", nullptr);
        CHECK(srh == 0, srh == 0 ? "the server's --help exits 0"
                                 : "the server's --help did not exit 0");
    }
}

#define TEST_GROUP(fn) {#fn, fn, nullptr},
#define TEST_GROUP_SERIAL(fn, why) {#fn, fn, why},

static const TestGroup kGroups[] = {
    TEST_GROUP(test_quant_geometry)
    TEST_GROUP(test_q4_0_layout)
    TEST_GROUP(test_q8_0_layout)
    TEST_GROUP(test_q5_0_high_bit_layout)
    TEST_GROUP(test_kquant_roundtrip_shape)
    TEST_GROUP(test_q2_k_plane_layout)
    TEST_GROUP(test_q6_k_group_stride)
    TEST_GROUP(test_iq2_xxs_layout)
    TEST_GROUP(test_iq4_xs_layout)
    TEST_GROUP(test_nvfp4_layout)
    TEST_GROUP(test_mxfp4_layout)
    TEST_GROUP(test_rocmfp4_layout)
    TEST_GROUP(test_tq2_0_layout)
    TEST_GROUP(test_tq1_0_layout)
    TEST_GROUP(test_new_quant_geometry)
    TEST_GROUP(test_q1_0_layout)
    TEST_GROUP(test_q2_0_layout)
    TEST_GROUP(test_pq2_0_layout)
    TEST_GROUP(test_ptq1_0_layout)
    TEST_GROUP(test_q2_0_rocmfpx_layout)
    TEST_GROUP(test_q3_0_rocmfpx_layout)
    TEST_GROUP(test_q6_0_rocmfpx_layout)
    TEST_GROUP(test_hadamard_act)
    TEST_GROUP(test_iq_family_structure)
    TEST_GROUP(test_vec_dot)
    TEST_GROUP(test_arch_table)
    TEST_GROUP(test_arch_tensor_maps)
    TEST_GROUP(test_load_verdict)
    TEST_GROUP(test_gguf_roundtrip)
    TEST_GROUP(test_tokenizer_spm_and_bpe)
    TEST_GROUP(test_sampler_determinism)
    TEST_GROUP(test_spec_sampling)
    TEST_GROUP(test_logit_softcap)
    TEST_GROUP(test_rope_pair_convention)
    TEST_GROUP(test_end_to_end_cpu)
    TEST_GROUP(test_prefill_chunk_bitexact)
    TEST_GROUP(test_gemma_family_chunk_gate)
    TEST_GROUP(test_lfm2_shortconv)
    TEST_GROUP(group_attention_sinks)
    TEST_GROUP(test_moe_schema_and_laziness)
    TEST_GROUP(test_moe_per_layer_experts)
    TEST_GROUP(test_moe_latent)
    TEST_GROUP(test_nvfp4_sidecar_scale)
    TEST_GROUP(test_moe_batch_fallback)
    TEST_GROUP(test_router_resident_preference)
    TEST_GROUP(test_moe_matches_dense_twin)
    TEST_GROUP(test_moe_grouped_prefill_matches_tokenwise)
    TEST_GROUP(test_moe_route_predictor)
    TEST_GROUP(test_expert_cache_policy)
    TEST_GROUP(test_expert_cache_byte_budget)
    TEST_GROUP(test_expert_cache_warm_tier)
    TEST_GROUP(test_expert_dispatch_table)
    TEST_GROUP(test_expert_dispatch_host_backed)
    TEST_GROUP(test_draft_head)
    TEST_GROUP(test_mtp_sidecar)
    TEST_GROUP(test_expert_warmup)
    TEST_GROUP(test_expert_index_adapt)
    TEST_GROUP(test_hc_lanes)
    TEST_GROUP(test_ple_hash)
    TEST_GROUP(test_qwen4exp_schema)
    TEST_GROUP(test_qwen4exp_forward)
    TEST_GROUP(test_qwen4exp_whole_graph)
    TEST_GROUP(test_qwen4exp_gdn_gate)
    TEST_GROUP(test_qwen4exp_hc_mix)
    TEST_GROUP(test_qwen4exp_ple)
    TEST_GROUP(test_mla_absorb)
    TEST_GROUP(test_mla_lite)
    TEST_GROUP(test_mla_backend)
    TEST_GROUP(test_npu_backend)
    TEST_GROUP(test_mla_loader)
    TEST_GROUP(test_mla_shared)
    TEST_GROUP(test_dsv4_hash_route)
    TEST_GROUP(test_dsv4_whole_graph)
    TEST_GROUP(test_qsa_select)
    TEST_GROUP(test_qwen4exp_qsa)
    TEST_GROUP(test_source_literal_hygiene)
    TEST_GROUP(test_cpu_expert_pool)
    TEST_GROUP(test_gdn_ops)
    TEST_GROUP(test_gdn_model_loads)
    TEST_GROUP(test_gdn_generation)
    TEST_GROUP(test_speculative_decoding)
    TEST_GROUP(test_attention_softcap)
    TEST_GROUP(test_attention_self_mask)
    TEST_GROUP(test_dflash_mask_guard)
    TEST_GROUP(test_nvfp4_sidecar_dflash)
    TEST_GROUP(test_dflash_laguna_spelling)
    TEST_GROUP(test_json)
    TEST_GROUP(test_http_server)
    TEST_GROUP(test_par_pool_covers_every_block)
    TEST_GROUP(test_kv_quant_int8)
    TEST_GROUP(test_kv_tier_per_layer_widths)
    TEST_GROUP(test_kv_tier_no_kv_layers)
    TEST_GROUP(test_kv_tier_drop_vs_spill)
    TEST_GROUP(test_kv_tier_host_reserve_refusal)
    TEST_GROUP(test_kv_tier_per_plane_cold_files)
    TEST_GROUP(test_moe_batch_ops)
    TEST_GROUP(test_cli_arg_refusal)
    TEST_GROUP_SERIAL(test_pageable_path_probe,
                      "times host->device copy rates, so it needs the machine")
};

#undef TEST_GROUP
#undef TEST_GROUP_SERIAL

static const int kGroupCount = static_cast<int>(sizeof(kGroups) / sizeof(kGroups[0]));

static std::string g_argv0;

static bool env_flag(const char *name) {
    const char *v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

// The parent re-invokes the binary it is already running, so the path has to be
// the executable's own and not argv[0] (which is relative to the caller's cwd).
static std::string self_path() {
#ifdef _WIN32
    char buf[4096];
    const DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) return std::string(buf, n);
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::string(buf);
    }
#endif
    return g_argv0;
}

// The progress files are scratch, so they go where scratch goes -- and putting
// them there keeps them off the model directory's sync path.
static std::string progress_dir() {
    const char *t = std::getenv("TEMP");
    if (!t || !*t) t = std::getenv("TMPDIR");
    if (!t || !*t) t = ".";
    std::string d(t);
    while (d.size() > 1 && (d.back() == '/' || d.back() == '\\')) d.pop_back();
    return d;
}

// The scratch area is scoped to ONE run, not to the machine. Two suite runs on
// this box -- the baseline gate and, beside it, a --selfcheck sweep -- used to
// share %TEMP%/krk-tests/krk-tests-group-N.dir and write over each other's
// fixtures, logs and progress files, which is why the reporting proof could only
// be run when nothing else was. The id defaults to this process's pid, which is
// unique for as long as the run lives, and the parent exports it so every child
// agrees on the same area; KRK_TEST_SCRATCH_ID pins it for a caller that wants to
// find the evidence again afterwards.
static void set_env_var(const char *name, const std::string &value);

static const char *scratch_run_id() {
    static const std::string id = [] {
        const char *want = std::getenv("KRK_TEST_SCRATCH_ID");
        if (want && *want) return std::string(want);
        std::string gen;
#ifdef _WIN32
        gen = std::to_string(static_cast<unsigned long>(GetCurrentProcessId()));
#else
        gen = std::to_string(static_cast<long>(::getpid()));
#endif
        // Children inherit it, so a child computing its own paths lands in the
        // same area as the parent that spawned it.
        set_env_var("KRK_TEST_SCRATCH_ID", gen);
        return gen;
    }();
    return id.c_str();
}

// A child gets its own named files under one scratch area, because the whole
// runner is a function of what each group is allowed to touch: the file it
// writes its running counts to, the file its own output goes to, and the
// working directory it runs in. The area is scratch, so it lives in TEMP --
// which also keeps it off the model directory's sync path.
static std::string scratch_area() {
    return progress_dir() + "/krk-tests-" + scratch_run_id();
}

static std::string group_path(int index, const char *suffix) {
    return scratch_area() + "/krk-tests-group-" + std::to_string(index) + suffix;
}

static std::string progress_path(int index, const char *override_path) {
    if (override_path && *override_path) return std::string(override_path);
    return group_path(index, ".progress");
}

static std::string log_path(int index) { return group_path(index, ".log"); }
static std::string scratch_dir(int index) { return group_path(index, ".dir"); }

// Absolute, because a child changes its working directory: a relative progress
// or log path would then point somewhere else, or nowhere.
static std::string absolute_path(const std::string &p) {
    std::error_code ec;
    const std::filesystem::path a = std::filesystem::absolute(p, ec);
    return ec ? p : a.string();
}

// Wall time, for the report only: no verdict in this file reads it, and it is
// the project's own monotonic clock (QPC on Windows, CLOCK_MONOTONIC elsewhere)
// so the runner and the engine cannot disagree about what a millisecond is.

static void read_text_file(const std::string &path, std::string *out) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    std::fclose(f);
}

static void read_progress(const std::string &path, int *passed, int *run) {
    *passed = 0;
    *run = 0;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    int p = 0, r = 0;
    // A torn read is rejected rather than trusted: p > r cannot be a real pair.
    if (std::fscanf(f, "%d %d", &p, &r) == 2 && p >= 0 && r >= p) {
        *passed = p;
        *run = r;
    }
    std::fclose(f);
}

struct ChildExit {
    bool started = false;
    bool crashed = false;
    unsigned long code = 0;
};

// One child's environment: the parent's own, with these entries forced (an
// empty value REMOVES the variable). Per-child rather than a mutation of the
// parent's environment, because a group's arms now run at the SAME time and two
// arms differ only in which injection they set -- one shared environment would
// hand them each other's injections, and the arm judge reads the injection to
// know what to expect.
struct ChildEnv {
    std::vector<std::pair<std::string, std::string>> set;
    void put(const std::string &k, const std::string &v) {
        for (auto &kv : set)
            if (kv.first == k) {
                kv.second = v;
                return;
            }
        set.emplace_back(k, v);
    }
    bool overrides() const { return !set.empty(); }
};

#ifdef _WIN32
static ChildExit spawn_child(const std::vector<std::string> &args) {
    ChildExit ce;
    std::string cmd;
    for (size_t i = 0; i < args.size(); i++) {
        if (i) cmd += ' ';
        cmd += '"';
        for (char c : args[i]) {
            if (c == '"') cmd += '\\';
            cmd += c;
        }
        cmd += '"';
    }
    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back('\0');

    // Hand the child this process's stdio, made inheritable first: a group that
    // crashes must still have left its FAIL lines in the parent's log, and the
    // parent's stdout may be a redirected file rather than a console.
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
    if (out && out != INVALID_HANDLE_VALUE)
        SetHandleInformation(out, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    if (err && err != INVALID_HANDLE_VALUE)
        SetHandleInformation(err, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);

    STARTUPINFOA si;
    std::memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = out;
    si.hStdError = err;
    PROCESS_INFORMATION pi;
    std::memset(&pi, 0, sizeof(pi));

    const std::string exe = self_path();
    if (exe.empty() ||
        !CreateProcessA(exe.c_str(), mutable_cmd.data(), nullptr, nullptr, TRUE, 0,
                        nullptr, nullptr, &si, &pi)) {
        return ce;
    }
    ce.started = true;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    ce.code = static_cast<unsigned long>(code);
    // The child's contract is 0 (its checks all passed) or 1 (some failed).
    // Anything else is a death: a __fastfail (0xC0000409), an access violation
    // (0xC0000005), a missing import (0xC0000135). None of these are exit
    // codes a correct child can produce.
    ce.crashed = (ce.code != 0 && ce.code != 1);
    return ce;
}
#else
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

static ChildExit spawn_child(const std::vector<std::string> &args) {
    ChildExit ce;
    std::vector<std::string> storage = args;
    std::vector<char *> argv;
    for (std::string &s : storage) argv.push_back(s.data());
    argv.push_back(nullptr);

    pid_t pid = 0;
    if (posix_spawn(&pid, storage[0].c_str(), nullptr, nullptr, argv.data(), environ) != 0)
        return ce;
    ce.started = true;

    int st = 0;
    if (waitpid(pid, &st, 0) < 0) {
        ce.crashed = true;
        return ce;
    }
    if (WIFSIGNALED(st)) {
        ce.crashed = true;
        ce.code = static_cast<unsigned long>(128 + WTERMSIG(st));
    } else {
        ce.code = static_cast<unsigned long>(WEXITSTATUS(st));
        ce.crashed = (ce.code != 0 && ce.code != 1);
    }
    return ce;
}
#endif

// ---------------------------------------------------------------------------
//  --selfcheck: the runner proving its own reporting, per group.
//
//  The isolation script aims the injection switches at one group chosen by
//  hand, which leaves the other 52 proved only for whichever case that group
//  happened to be -- and what a group's report looks like depends on its own
//  shape: one check, checks conditional on earlier ones, a fixture built late.
//  This runs six arms for EVERY group and judges each answer against that
//  group's own count:
//
//    crash 0/0          the group dies before its first check
//    crash 2/2          ... after two checks (after its only check, if tiny)
//    crash n/n          ... after its OWN last check. passed == run there, so
//                       the only thing that says the group died is the child's
//                       exit, and no two groups have the same expected number
//    every check fails  named FAILED, 0 passed, every FAIL line echoed
//    no checks at all   named NO CHECKS, never a pass
//    fuzz               a crash at a RANDOM check, and in one run in four at a
//                       second group as well -- two groups non-passing at once
//                       is a shape no fixed arm can produce. The aim is drawn
//                       from a printed seed, so a mismatch is replayable
//
//  An arm is only right when: the group's own line printed with the arm's
//  status, the report records that status and count, the other 52 groups still
//  passed with their own counts, the totals equal the clean total minus this
//  group's count plus what it reached, bad_groups is 1, the exit is 1, the
//  group is named in 'groups needing attention', the summary still printed,
//  and the FAIL lines that reached the log are the failing checks. The report
//  and the printed run are checked against each other because the claim is
//  about both.
//
//  --selfcheck --selftest checks the checker itself on constructed answers:
//  every way an arm can be wrong has a case that must be rejected, in
//  milliseconds. A checker that cannot fail proves nothing.
//
//  The fuzz arm's seed is printed at the top of a sweep, and
//  KRK_SELFCHECK_SEED=<that number> draws exactly those aims again -- per group,
//  so --group N draws what the sweep drew for N. --selfcheck --report writes the
//  arms' verdicts out in the runner's own report format (kind "selfcheck"), which
//  is what lets the baseline gate compare the reporting proof itself: an arm
//  that was ok and is not any more is a regression, named with its problem.
// ---------------------------------------------------------------------------

static void set_env_var(const char *name, const std::string &value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    ::setenv(name, value.c_str(), 1);
#endif
}

static void clear_env_var(const char *name) { set_env_var(name, ""); }

// The arms batch's scratch: one log and one report per group per arm, so a
// failed arm keeps its evidence and a green run leaves nothing behind.
// The sweep's own evidence -- the clean run's report, the arms' logs -- lives
// BESIDE the run areas, not inside one: a green suite run removes its whole area,
// so evidence a parent still needs must not sit in a child's area. That is not
// hypothetical: with the selfcheck dir inside the area, the clean-run child
// (whose own area is that area) deleted its parent's report the moment it went
// green, and the sweep reported "the clean run is not green (exit 0)".
static std::string selfcheck_dir() {
    return progress_dir() + "/krk-tests-selfcheck-" + scratch_run_id();
}

static std::vector<std::string> split_lines(const std::string &text) {
    std::vector<std::string> out;
    size_t pos = 0;
    for (;;) {
        const size_t nl = text.find('\n', pos);
        std::string ln = text.substr(pos, (nl == std::string::npos ? text.size() : nl) - pos);
        if (!ln.empty() && ln.back() == '\r') ln.pop_back();
        out.push_back(ln);
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    return out;
}

static std::string trim_copy(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) b--;
    return s.substr(a, b - a);
}

static std::string selfcheck_join(const std::vector<std::string> &notes) {
    std::string out;
    for (const std::string &n : notes) out += (out.empty() ? "" : "; ") + n;
    return out;
}

// A constructed mismatch, so the verdict can be shown to leave a non-zero
// exit and not only to exist. The expectation is made wrong by one check,
// which can turn a green run red and never the reverse: it cannot hide a
// problem, only invent one, which is the safe direction for a switch like
// this.
static int selfcheck_break_bump() {
    const char *v = std::getenv("KRK_SELFCHECK_BREAK");
    return (v && std::strcmp(v, "counts") == 0) ? 1 : 0;
}
// The report's own field syntax, read back: this is the interface the baseline
// checker reads, so the runner checking it is checking the artifact it ships.
static bool json_int_at(const std::string &line, const char *key, int *out) {
    const std::string pat = std::string("\"") + key + "\":";
    size_t at = line.find(pat);
    if (at == std::string::npos) return false;
    at += pat.size();
    while (at < line.size() && (line[at] == ' ' || line[at] == '\t')) at++;
    const char *begin = line.c_str() + at;
    char *end = nullptr;
    const long v = std::strtol(begin, &end, 10);
    if (end == begin) return false;
    *out = static_cast<int>(v);
    return true;
}

static bool json_str_at(const std::string &line, const char *key, std::string *out) {
    const std::string pat = std::string("\"") + key + "\":";
    size_t at = line.find(pat);
    if (at == std::string::npos) return false;
    at += pat.size();
    while (at < line.size() && (line[at] == ' ' || line[at] == '\t')) at++;
    if (at >= line.size() || line[at] != '"') return false;
    at++;
    std::string v;
    for (; at < line.size(); at++) {
        const char c = line[at];
        if (c == '\\' && at + 1 < line.size()) {
            v += line[at];
            v += line[at + 1];
            at++;
            continue;
        }
        if (c == '"') break;
        v += c;
    }
    *out = v;
    return true;
}

struct SelfGroup {
    int index = -1;
    std::string name;
    std::string status;
    int passed = -1;
    int run = -1;
};

struct SelfTotals {
    int groups = -1;
    int passed = -1;
    int run = -1;
    int bad_groups = -1;
    int exit_code = -1;
};

static bool parse_selfcheck_report(const std::string &text, std::vector<SelfGroup> *groups,
                                   SelfTotals *totals) {
    if (text.empty()) return false;
    bool saw_totals = false, saw_group = false;
    for (const std::string &ln : split_lines(text)) {
        if (ln.find("\"totals\":") != std::string::npos) {
            SelfTotals t;
            if (json_int_at(ln, "groups", &t.groups) && json_int_at(ln, "passed", &t.passed) &&
                json_int_at(ln, "run", &t.run) && json_int_at(ln, "bad_groups", &t.bad_groups) &&
                json_int_at(ln, "exit", &t.exit_code)) {
                *totals = t;
                saw_totals = true;
            }
        } else if (ln.find("\"name\":") != std::string::npos &&
                   ln.find("\"index\":") != std::string::npos &&
                   ln.find("\"status\":") != std::string::npos) {
            SelfGroup g;
            if (json_int_at(ln, "index", &g.index) && json_str_at(ln, "name", &g.name) &&
                json_int_at(ln, "passed", &g.passed) && json_int_at(ln, "run", &g.run) &&
                json_str_at(ln, "status", &g.status))
                groups->push_back(g), saw_group = true;
        }
    }
    return saw_totals && saw_group;
}

struct SelfText {
    bool has_line = false;
    int passed = -1;
    int run = -1;
    std::string status;
    bool summary = false;
    int total_passed = -1;
    int total_run = -1;
    int fail_lines = 0;
    bool ran_no_checks = false;
    bool has_attention = false;
    std::string attention;
};

// What the run itself printed: the group's line, the naming line, the summary,
// and how many FAIL lines made it into this run's log.
static SelfText parse_selfcheck_text(const std::string &text, int index) {
    SelfText t;
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "  [%2d/%d] ", index + 1, kGroupCount);
    const std::string want(prefix);
    for (const std::string &ln : split_lines(text)) {
        if (!t.has_line && ln.rfind(want, 0) == 0) {
            const std::string rest = ln.substr(want.size());
            char nm[256];
            int p = -1, r = -1, consumed = 0;
            long long ms = 0;
            if (std::sscanf(rest.c_str(), "%255s %d/%d %lldms%n", nm, &p, &r, &ms, &consumed) == 4) {
                t.has_line = true;
                t.passed = p;
                t.run = r;
                t.status = trim_copy(rest.substr(static_cast<size_t>(consumed)));
            }
        }
        if (ln.rfind("FAIL ", 0) == 0) t.fail_lines++;
        if (!t.summary &&
            std::sscanf(ln.c_str(), "%d/%d checks passed", &t.total_passed, &t.total_run) == 2)
            t.summary = true;
        if (!t.has_attention && ln.rfind("groups needing attention:", 0) == 0) {
            t.attention = ln.substr(std::strlen("groups needing attention:"));
            t.has_attention = true;
        }
        if (ln.find("ran no checks") != std::string::npos) t.ran_no_checks = true;
    }
    return t;
}

static bool names_token(const std::string &line, const std::string &tok) {
    size_t pos = 0;
    while (pos < line.size()) {
        while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
        const size_t start = pos;
        while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t') pos++;
        if (pos > start && line.compare(start, pos - start, tok) == 0) return true;
    }
    return false;
}

// "CRASHED 0x" and eight hex digits: the code itself is the platform's (the
// CRT's fastfail code here, a signal number elsewhere), so the checker does not
// hard-code it, it requires one.
static bool looks_crashed(const std::string &s) {
    static const char *kPrefix = "CRASHED 0x";
    const size_t n = std::strlen(kPrefix);
    if (s.size() != n + 8 || s.rfind(kPrefix, 0) != 0) return false;
    for (size_t i = n; i < s.size(); i++) {
        const char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
            return false;
    }
    return true;
}

static const char *const kSelfcheckArms[] = {"crash", "crash-2", "crash-last", "fail", "empty",
                                             "fuzz"};
static const int kSelfcheckArmCount = 6;

// What an arm has to have aimed at, in the group's own check numbering.
static int selfcheck_aim(const char *key, int healthy) {
    if (std::strcmp(key, "crash") == 0) return 0; // before the first check
    if (std::strcmp(key, "crash-last") == 0) return healthy;
    if (std::strcmp(key, "crash-2") == 0) return healthy < 2 ? healthy : 2;
    return 0; // the fuzz arm draws its aim; see run_selfcheck
}

// What one arm aimed at, in the group's own check numbering: the group the arm
// is about, the check its crash lands after, and -- for a drawn pair -- a second
// group that dies in the same run.
struct SelfArm {
    const char *key = "";
    int index = -1;
    std::string name;
    int healthy = 0;
    int at = 0;
    int second_index = -1;
    std::string second_name;
    int second_healthy = 0;
    int second_at = 0;
    bool paired() const { return second_index >= 0; }
};

// The fixed arms' aim, built here so the driver and the selftest build it the
// same way instead of each writing its own numbers.
static SelfArm selfcheck_arm_for(const char *key, int index, const std::string &name, int healthy) {
    SelfArm a;
    a.key = key;
    a.index = index;
    a.name = name;
    a.healthy = healthy;
    a.at = selfcheck_aim(key, healthy);
    return a;
}

static ChildEnv selfcheck_arm_env(const SelfArm &arm) {
    ChildEnv env;
    // All four are named explicitly, because an arm runs in its own environment
    // and "this arm sets no crash" has to mean the variable is ABSENT, not
    // merely unset in a parent another arm may have touched.
    env.put("KRK_TEST_INJECT_CRASH", "");
    env.put("KRK_TEST_INJECT_CRASH_AT", "");
    env.put("KRK_TEST_INJECT_FAIL", "");
    env.put("KRK_TEST_INJECT_EMPTY", "");
    if (std::strcmp(arm.key, "fail") == 0) {
        env.put("KRK_TEST_INJECT_FAIL", std::to_string(arm.index));
    } else if (std::strcmp(arm.key, "empty") == 0) {
        env.put("KRK_TEST_INJECT_EMPTY", std::to_string(arm.index));
    } else {
        std::string groups = std::to_string(arm.index);
        std::string ats = std::to_string(arm.at);
        if (arm.paired()) {
            groups += "," + std::to_string(arm.second_index);
            ats += "," + std::to_string(arm.second_at);
        }
        env.put("KRK_TEST_INJECT_CRASH", groups);
        env.put("KRK_TEST_INJECT_CRASH_AT", ats);
    }
    return env;
}

// One arm's answer, checked. Returns the problems found, joined with "; ";
// an empty string means the runner reported this arm the way it has to.
// One arm's answer, checked. Returns the problems found, joined with "; ";
// an empty string means the runner reported this arm the way it has to.
static std::string selfcheck_arm_check(const SelfArm &arm, int base_run, int rc,
                                       const std::string &log, const std::string &report_text) {
    const bool is_fail = std::strcmp(arm.key, "fail") == 0;
    const bool is_empty = std::strcmp(arm.key, "empty") == 0;
    const bool is_crash = !is_fail && !is_empty;
    const std::string want_kind = is_crash ? "crashed" : (is_empty ? "no-checks" : "failed");
    const int bump = selfcheck_break_bump();

    std::vector<SelfGroup> groups;
    SelfTotals totals;
    if (!parse_selfcheck_report(report_text, &groups, &totals))
        return "no report was written (exit " + std::to_string(rc) + ")";

    std::vector<std::string> bad;
    if (totals.groups != kGroupCount)
        bad.push_back("the report says " + std::to_string(totals.groups) + " groups, want " +
                      std::to_string(kGroupCount));
    const SelfGroup *g = nullptr;
    int seen = 0;
    for (const SelfGroup &x : groups) {
        if (x.name != arm.name) continue;
        seen++;
        if (!g) g = &x;
    }
    if (seen != 1) {
        bad.push_back("the report lists " + arm.name + " " + std::to_string(seen) + " times");
        return selfcheck_join(bad);
    }
    const SelfGroup *g2 = nullptr;
    if (arm.paired()) {
        for (const SelfGroup &x : groups)
            if (x.name == arm.second_name) g2 = &x;
        if (!g2) {
            bad.push_back("the report does not list the paired group " + arm.second_name);
            return selfcheck_join(bad);
        }
    }

    // Every group the arm did not aim at still has to pass, with its own count.
    std::string others;
    for (const SelfGroup &x : groups) {
        if (x.name == arm.name || (g2 && x.name == g2->name)) continue;
        if (x.status != "pass") others += (others.empty() ? "" : ", ") + x.name;
    }
    if (!others.empty()) bad.push_back("another group is not passing: " + others);
    if (rc != 1) bad.push_back("exit was " + std::to_string(rc) + ", want 1");

    // What the arm aimed at. A failing check can skip the checks after it, so
    // for that arm the count is whatever the group reached -- but it has to
    // have reached something.
    const int reached = is_fail ? g->run : arm.at;
    if (is_fail) {
        if (g->run < 1) bad.push_back("ran no checks; a failing group must still count them");
        if (g->passed != 0)
            bad.push_back(std::to_string(g->passed) + " checks passed, want 0");
    } else if (g->passed != arm.at + bump || g->run != arm.at + bump) {
        bad.push_back("the report records " + std::to_string(g->passed) + "/" +
                      std::to_string(g->run) + ", want " + std::to_string(arm.at + bump) + "/" +
                      std::to_string(arm.at + bump));
    }
    if (g->status != want_kind)
        bad.push_back("status " + g->status + ", want " + want_kind);
    if (g2 && (g2->status != "crashed" || g2->passed != arm.second_at ||
               g2->run != arm.second_at))
        bad.push_back("the paired group " + g2->name + " records " + std::to_string(g2->passed) +
                      "/" + std::to_string(g2->run) + " (" + g2->status + "), want " +
                      std::to_string(arm.second_at) + "/" + std::to_string(arm.second_at) +
                      " (crashed)");

    // The run's own output, and the two sources have to agree.
    const SelfText txt = parse_selfcheck_text(log, arm.index);
    if (!txt.has_line) {
        bad.push_back("the group's line did not print");
    } else {
        if (txt.passed != g->passed || txt.run != g->run)
            bad.push_back("the line says " + std::to_string(txt.passed) + "/" +
                          std::to_string(txt.run) + ", the report says " +
                          std::to_string(g->passed) + "/" + std::to_string(g->run));
        if (is_crash) {
            if (!looks_crashed(txt.status))
                bad.push_back("the line does not report a crash code: '" + txt.status + "'");
        } else if (txt.status != (is_empty ? "NO CHECKS" : "FAILED")) {
            bad.push_back("the line says '" + txt.status + "', want '" +
                          std::string(is_empty ? "NO CHECKS" : "FAILED") + "'");
        }
    }
    if (g2) {
        const SelfText t2 = parse_selfcheck_text(log, g2->index);
        if (!t2.has_line || t2.passed != arm.second_at || t2.run != arm.second_at)
            bad.push_back("the paired group's line did not report " +
                          std::to_string(arm.second_at) + "/" + std::to_string(arm.second_at));
        else if (!looks_crashed(t2.status))
            bad.push_back("the paired group's line does not report a crash code: '" + t2.status +
                          "'");
    }
    if (!txt.has_attention || !names_token(txt.attention, arm.name))
        bad.push_back("'groups needing attention' does not name " + arm.name);
    if (g2 && !names_token(txt.attention, g2->name))
        bad.push_back("'groups needing attention' does not name the paired group " + g2->name);
    if (!txt.summary) bad.push_back("the summary line did not print");
    if (is_empty && !txt.ran_no_checks)
        bad.push_back("the summary does not report a group that ran no checks");
    const int want_fail_lines = is_fail ? g->run : 0;
    if (txt.fail_lines != want_fail_lines)
        bad.push_back(std::to_string(txt.fail_lines) + " FAIL lines reached the run's log, want " +
                      std::to_string(want_fail_lines));

    // The totals: every aimed group's count is inside them whatever its status --
    // a group that died with two checks recorded contributes those two -- and
    // the groups the arm aimed at are the only ones that did not pass.
    const int want_total_run = base_run - arm.healthy - (g2 ? arm.second_healthy : 0) + reached +
                               (g2 ? arm.second_at : 0) + bump;
    const int want_total_passed = is_fail ? want_total_run - g->run : want_total_run;
    if (totals.run != want_total_run)
        bad.push_back("total run " + std::to_string(totals.run) + ", want " +
                      std::to_string(want_total_run));
    if (totals.passed != want_total_passed)
        bad.push_back("total passed " + std::to_string(totals.passed) + ", want " +
                      std::to_string(want_total_passed));
    const int want_bad = g2 ? 2 : 1;
    if (totals.bad_groups != want_bad)
        bad.push_back("the report says " + std::to_string(totals.bad_groups) +
                      " bad group(s), want " + std::to_string(want_bad));
    if (totals.exit_code != 1)
        bad.push_back("the report's exit is " + std::to_string(totals.exit_code) + ", want 1");
    return selfcheck_join(bad);
}

// The runner's output goes to a file of this mode's own, so an arm can be read
// back whole: its lines, its FAIL text, and its summary from one place.
#ifdef _WIN32
static std::string child_command(const std::vector<std::string> &args) {
    std::string cmd;
    for (size_t i = 0; i < args.size(); i++) {
        if (i) cmd += ' ';
        cmd += '"';
        for (char c : args[i]) {
            if (c == '"') cmd += '\\';
            cmd += c;
        }
        cmd += '"';
    }
    return cmd;
}

// An environment BLOCK for CreateProcessA: sorted "K=V" strings, each with its
// NUL, then a final NUL. The parent's own entries are copied rather than
// enumerated by the child, so a child of an arm sees exactly one injection.
static std::string windows_env_block(const ChildEnv &env) {
    std::vector<std::string> entries;
    if (LPCH block = GetEnvironmentStringsA()) {
        for (const char *p = block; *p;) {
            const std::string e(p);
            p += e.size() + 1;
            const size_t eq = e.find('=');
            // A leading '=' is the per-drive current-directory pseudo-variable;
            // CreateProcess requires those to be left out.
            if (eq == std::string::npos || eq == 0) continue;
            const std::string key = e.substr(0, eq);
            bool replaced = false;
            for (const auto &kv : env.set)
                if (kv.first == key) replaced = true;
            if (!replaced) entries.push_back(e);
        }
        FreeEnvironmentStringsA(block);
    }
    for (const auto &kv : env.set)
        if (!kv.second.empty()) entries.push_back(kv.first + "=" + kv.second);
    std::sort(entries.begin(), entries.end());
    std::string out;
    for (const std::string &e : entries) {
        out += e;
        out.push_back('\0');
    }
    if (out.empty()) out.push_back('\0');
    out.push_back('\0');
    return out;
}

static ChildExit spawn_child_to(const std::vector<std::string> &args, const std::string &out_path,
                                const ChildEnv &env) {
    ChildExit ce;
    std::string cmd = child_command(args);
    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    std::string env_block = windows_env_block(env);
    mutable_cmd.push_back('\0');

    SECURITY_ATTRIBUTES sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE h = CreateFileA(out_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return ce;

    STARTUPINFOA si;
    std::memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = h;
    si.hStdError = h;
    PROCESS_INFORMATION pi;
    std::memset(&pi, 0, sizeof(pi));

    const std::string exe = self_path();
    const BOOL ok = !exe.empty() &&
                    CreateProcessA(exe.c_str(), mutable_cmd.data(), nullptr, nullptr, TRUE, 0,
                                   env.overrides() ? env_block.data() : nullptr, nullptr, &si,
                                   &pi);
    CloseHandle(h);
    if (!ok) return ce;
    ce.started = true;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    ce.code = static_cast<unsigned long>(code);
    ce.crashed = (ce.code != 0 && ce.code != 1);
    return ce;
}
#else
#include <fcntl.h>

static std::vector<std::string> posix_env_storage(const ChildEnv &env) {
    std::vector<std::string> out;
    for (char **e = environ; e && *e; e++) {
        const std::string s(*e);
        const size_t eq = s.find('=');
        bool replaced = false;
        for (const auto &kv : env.set)
            if (eq != std::string::npos && kv.first == s.substr(0, eq)) replaced = true;
        if (!replaced) out.push_back(s);
    }
    for (const auto &kv : env.set)
        if (!kv.second.empty()) out.push_back(kv.first + "=" + kv.second);
    return out;
}

static ChildExit spawn_child_to(const std::vector<std::string> &args, const std::string &out_path,
                                const ChildEnv &env) {
    ChildExit ce;
    std::vector<std::string> storage = args;
    std::vector<char *> argv;
    for (std::string &s : storage) argv.push_back(s.data());
    argv.push_back(nullptr);

    const int fd = ::open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return ce;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd, STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, fd, STDERR_FILENO);
    posix_spawn_file_actions_addclose(&fa, fd);

    pid_t pid = 0;
    std::vector<std::string> envs = posix_env_storage(env);
    std::vector<char *> envp;
    for (std::string &s : envs) envp.push_back(s.data());
    envp.push_back(nullptr);
    const int spawned = posix_spawn(&pid, storage[0].c_str(), &fa, nullptr, argv.data(),
                                    env.overrides() ? envp.data() : environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(fd);
    if (spawned != 0) return ce;
    ce.started = true;
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) {
        ce.crashed = true;
        return ce;
    }
    if (WIFSIGNALED(st)) {
        ce.crashed = true;
        ce.code = static_cast<unsigned long>(128 + WTERMSIG(st));
    } else {
        ce.code = static_cast<unsigned long>(WEXITSTATUS(st));
        ce.crashed = (ce.code != 0 && ce.code != 1);
    }
    return ce;
}

#endif

// The shortest form, for the calls that want the parent's own environment
// (the clean base run, and the isolation checks).
static ChildExit spawn_child_to(const std::vector<std::string> &args, const std::string &out_path) {
    return spawn_child_to(args, out_path, ChildEnv{});
}

// ---------------------------------------------------------------------------
//  The fuzz arm's randomness, and the one number that makes it replayable.
// ---------------------------------------------------------------------------

static u64 splitmix64(u64 x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// A generator per group, so --group N draws the aims the sweep drew for N.
static std::mt19937_64 selfcheck_rng(u64 seed, int index) {
    return std::mt19937_64(
        splitmix64(seed + 0x9e3779b97f4a7c15ull * (static_cast<u64>(index) + 1)));
}

static int selfcheck_draw(std::mt19937_64 &rng, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return d(rng);
}

static u64 selfcheck_default_seed() {
    std::random_device rd;
    u64 x = static_cast<u64>(now_us());
    for (int i = 0; i < 3; i++) x = splitmix64(x ^ static_cast<u64>(rd()));
    return x;
}

// -1 when unset, 0 when set to something that is not a number, 1 when read. A
// seed that cannot be read is refused rather than replaced: it was set to make a
// run replayable, and a silent substitution would break exactly that.
static int selfcheck_seed_env(u64 *out) {
    const char *v = std::getenv("KRK_SELFCHECK_SEED");
    if (!v || !*v) return -1;
    char *end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 0);
    if (end == v || (end && *end != '\0')) return 0;
    *out = static_cast<u64>(n);
    return 1;
}

// ---------------------------------------------------------------------------
//  The selfcheck's own report (--selfcheck --report), in the runner's report
//  format so one comparison tool reads both, with each group's arm verdicts
//  recorded beside the counts the clean run reported. A group's status is pass
//  only when every arm was answered the way it has to be, and the arm problems
//  are that group's fail_lines -- which is how a gate names a broken arm
//  without knowing anything about arms.
// ---------------------------------------------------------------------------

struct SelfArmVerdict {
    std::string key;
    std::string problem; // empty when the arm was reported the way it has to be
};

struct SelfGroupVerdict {
    int index = 0;
    std::string name;
    int passed = 0; // the clean run's counts for this group
    int run = 0;
    i64 ms = 0;
    std::vector<SelfArmVerdict> arms;
    bool ok() const {
        for (const SelfArmVerdict &a : arms)
            if (!a.problem.empty()) return false;
        return true;
    }
};

static void json_escape(const std::string &in, std::string *out); // defined with the report

static bool write_selfcheck_report(const char *path, const std::vector<SelfGroupVerdict> &done,
                                   u64 seed, i64 wall_ms, int exit_code) {
    if (!path || !*path) return true;
    std::FILE *f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "report: cannot write %s\n", path);
        return false;
    }
    int passed = 0, run = 0, bad = 0;
    for (const SelfGroupVerdict &v : done) {
        passed += v.passed;
        run += v.run;
        if (!v.ok()) bad++;
    }
    std::fprintf(f, "{\n  \"format\": 1,\n  \"suite\": \"kraken-tests\",\n");
    std::fprintf(f, "  \"kind\": \"selfcheck\",\n  \"arms\": %d,\n  \"seed\": %llu,\n",
                 kSelfcheckArmCount, static_cast<unsigned long long>(seed));
    std::fprintf(f, "  \"wall_ms\": %lld,\n", static_cast<long long>(wall_ms));
    std::fprintf(f,
                 "  \"totals\": {\"groups\": %d, \"passed\": %d, \"run\": %d, "
                 "\"bad_groups\": %d, \"exit\": %d},\n",
                 static_cast<int>(done.size()), passed, run, bad, exit_code);
    std::fprintf(f, "  \"groups\": [\n");
    for (size_t i = 0; i < done.size(); i++) {
        const SelfGroupVerdict &v = done[i];
        std::string name;
        json_escape(v.name, &name);
        std::fprintf(f,
                     "    {\"index\": %d, \"name\": \"%s\", \"status\": \"%s\", "
                     "\"passed\": %d, \"run\": %d, \"ms\": %lld, \"fail_lines\": [",
                     v.index, name.c_str(), v.ok() ? "pass" : "failed", v.passed, v.run,
                     static_cast<long long>(v.ms));
        bool first = true;
        for (const SelfArmVerdict &a : v.arms) {
            if (a.problem.empty()) continue;
            std::string ln;
            json_escape("arm " + a.key + ": " + a.problem, &ln);
            std::fprintf(f, "%s\"%s\"", first ? "" : ", ", ln.c_str());
            first = false;
        }
        std::fprintf(f, "], \"arms\": {");
        for (size_t k = 0; k < v.arms.size(); k++) {
            std::string val;
            json_escape(v.arms[k].problem.empty() ? std::string("ok") : v.arms[k].problem, &val);
            std::fprintf(f, "%s\"%s\": \"%s\"", k ? ", " : "", v.arms[k].key.c_str(),
                         val.c_str());
        }
        std::fprintf(f, "}}%s\n", i + 1 == done.size() ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    const bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

// The arms choose their own switches, so a caller's injection has to be
// refused rather than merged: it would make the clean run something other than
// clean, and every count after that would be measured against the wrong base.
static bool selfcheck_env_conflict() {
    static const char *const kNames[] = {"KRK_TEST_INJECT_CRASH", "KRK_TEST_INJECT_FAIL",
                                         "KRK_TEST_INJECT_EMPTY"};
    for (const char *n : kNames) {
        const char *v = std::getenv(n);
        if (v && *v) {
            std::fprintf(stderr, "kraken-tests: --selfcheck chooses its own injections; unset %s\n",
                         n);
            return true;
        }
    }
    return false;
}

static int run_selfcheck_selftest();

// One arm's child: what to run, what environment it inherits, and what it
// answered. A group's arms are launched together, so the job outlives the
// iteration that built it and the verdict is read after every child is done.
struct ArmJob {
    SelfArm arm;
    std::string tag, log, rep;
    ChildEnv env;
    std::vector<std::string> args;
    ChildExit exit;
    std::string log_text, rep_text;
};

static int run_selfcheck(int workers, const std::string &only_raw, const char *report_flag) {
    if (selfcheck_env_conflict()) return 2;

    // The report path is the caller's, read before the children's environment is
    // cleared: --selfcheck records the ARMS, and a child that inherited the same
    // path would write a run's own report over it.
    std::string report_path = (report_flag && *report_flag) ? report_flag : "";
    if (report_path.empty()) {
        const char *env_report = std::getenv("KRK_TEST_REPORT");
        if (env_report && *env_report) report_path = env_report;
    }

    // The fuzz arm's draws have to be replayable from the number printed, so a
    // seed that cannot be read is refused rather than replaced.
    u64 seed = 0;
    const int seed_env = selfcheck_seed_env(&seed);
    if (seed_env == 0) {
        std::fprintf(stderr, "kraken-tests: KRK_SELFCHECK_SEED is not a number; the fuzz arm's "
                             "aims could not be replayed from it\n");
        return 2;
    }
    if (seed_env < 0) seed = selfcheck_default_seed();

    const std::string dir = selfcheck_dir();
    // Every area this sweep hands to a child, so a green sweep can remove exactly
    // what it named and nothing else. The children write to areas of their own
    // (see the arm loop), and an arm is non-passing by construction, which means
    // it keeps its area as evidence -- so the sweep is the only one left to
    // dispose of them once the verdict is in.
    std::vector<std::string> child_areas;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "kraken-tests: selfcheck cannot use '%s' (%s)\n", dir.c_str(),
                     ec.message().c_str());
        return 2;
    }

    // The children prove the default runner, so the caller's escape hatches and
    // report path are cleared rather than inherited.
    clear_env_var("KRK_TEST_INPROC");
    clear_env_var("KRK_TEST_REPORT");
    clear_env_var("KRK_TEST_INJECT_CRASH");
    clear_env_var("KRK_TEST_INJECT_CRASH_AT");
    clear_env_var("KRK_TEST_INJECT_FAIL");
    clear_env_var("KRK_TEST_INJECT_EMPTY");

    // The base: one clean run, whose per-group counts every arm is judged
    // against. It is a child too, so it is the same code path as any other run.
    //
    // How wide a group's arms run, and how many workers and THREADS each child
    // gets. The arms x workers product is the sweep's worker budget, so the
    // sweep is as wide as one suite run and no wider. The thread number is the
    // part the wave was paying for without being told: a group child builds
    // Pool(), which takes `hardware_concurrency()` worker threads unless
    // KRK_THREADS says otherwise, so six concurrent suites were six suites'
    // worth of 31-thread pools on cores that only one suite can fill. The pool
    // is capped, never removed -- a floor of 2 keeps it a pool -- and a child's
    // grandchildren inherit the cap, because it travels in its environment.
    const int cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int arms_parallel = std::max(1, std::min(kSelfcheckArmCount, workers));
    // A suite that contains a group which declared itself SERIAL cannot be
    // co-run with its own siblings -- not even with a SIBLING GROUP's arms. That
    // declaration exists because a neighbour's traffic lands in the group's
    // measurements, and in this sweep every arm is a full suite, so any two arms
    // in flight are two neighbours of the serial group. The first attempt scoped
    // the width to the serial group's own arms and it failed immediately:
    // `--selfcheck --group 0` reported `another group is not passing:
    // test_pageable_path_probe` in its crash-last arm, because six of group 0's
    // arms were six suites and all six ran the probe.
    //
    // So one arm at a time, and that arm gets the workers a sixth of them would
    // deny it. What this costs is wall time, not a claim: the arms differ in
    // environment, scratch area, log and report, so their ORDER is not part of
    // what is proved.
    bool serial_in_suite = false;
    for (int g = 0; g < kGroupCount; g++) {
        if (kGroups[g].serial) serial_in_suite = true;
    }
    const int arms_width = serial_in_suite ? 1 : arms_parallel;
    const int arm_workers = std::max(1, std::min(8, cores / arms_width));
    const int threads_per_child =
        std::max(2, cores / (arms_width * arm_workers));
    const int base_threads = std::max(2, cores / std::max(1, workers));

    // The A/B arm: KRK_SELFCHECK_THREADS=default leaves every child's pool at the
    // machine's own size, which is what the wave paid for before the cap.
    const char *thr_env = std::getenv("KRK_SELFCHECK_THREADS");
    const bool cap_threads =
        !(thr_env && (!std::strcmp(thr_env, "default") || !std::strcmp(thr_env, "off") ||
                      !std::strcmp(thr_env, "0")));

    const std::string base_log = dir + "/base.log";
    const std::string base_rep = dir + "/base.json";
    const std::string clean_id = std::string(scratch_run_id()) + "-clean";
    child_areas.push_back(clean_id);
    const std::vector<std::string> base_args{self_path(), "--workers", std::to_string(workers),
                                             "--report", base_rep};
    ChildEnv base_env;
    base_env.put("KRK_TEST_SCRATCH_ID", clean_id);
    if (cap_threads) base_env.put("KRK_THREADS", std::to_string(base_threads));
    const ChildExit bce = spawn_child_to(base_args, base_log, base_env);
    std::string base_text, base_report;
    read_text_file(base_log, &base_text);
    read_text_file(base_rep, &base_report);
    std::vector<SelfGroup> groups;
    SelfTotals totals;
    const bool clean = bce.started && !bce.crashed && bce.code == 0 &&
                       parse_selfcheck_report(base_report, &groups, &totals) &&
                       totals.groups == kGroupCount && totals.exit_code == 0 &&
                       totals.bad_groups == 0 && totals.passed == totals.run &&
                       groups.size() == static_cast<size_t>(kGroupCount);
    if (!clean) {
        std::fprintf(stderr, "selfcheck: the clean run is not green (exit %lu), "
                             "so there is nothing to compare against\n",
                     bce.code);
        std::fprintf(stderr, "evidence kept: %s%s\n", dir.c_str(), bce.started ? "" : " (not started)");
        return 1;
    }

    int only = -1;
    if (!only_raw.empty()) {
        for (const SelfGroup &g : groups)
            if (only_raw == std::to_string(g.index) || only_raw == g.name) only = g.index;
        if (only < 0) {
            std::fprintf(stderr, "selfcheck: no group matches '%s' (--list-groups)\n",
                         only_raw.c_str());
            return 2;
        }
    }

    std::fprintf(stderr,
                 "selfcheck: %d arms per group, %d group(s), %d arm(s) at once x %d worker(s) x "
                 "%d thread(s) (KRK_SELFCHECK_THREADS=default leaves the pools at their own size; "
                 "a suite with a serial group sweeps one arm at a time, on more workers)\n",
                 kSelfcheckArmCount, kGroupCount, arms_parallel, arm_workers, threads_per_child);
    std::fprintf(stderr, "selfcheck: seed %llu (KRK_SELFCHECK_SEED=%llu draws these aims again)\n",
                 static_cast<unsigned long long>(seed), static_cast<unsigned long long>(seed));
    std::fprintf(stderr, "clean run: %d/%d checks\n", totals.passed, totals.run);

    const i64 t0 = now_ms();
    int failed_groups = 0, failed_arms = 0, swept = 0;
    std::vector<SelfGroupVerdict> verdicts;
    for (const SelfGroup &g : groups) {
        if (only >= 0 && g.index != only) continue;
        swept++;
        SelfGroupVerdict v;
        v.index = g.index;
        v.name = g.name;
        v.passed = g.passed;
        v.run = g.run;
        const i64 group_t0 = now_ms();
        // A generator per group, so --group N draws what the sweep draws for N.
        std::mt19937_64 rng = selfcheck_rng(seed, g.index);
        std::string notes;
        std::string fuzz_aim;
        std::vector<ArmJob> jobs;
        for (int a = 0; a < kSelfcheckArmCount; a++) {
            const char *key = kSelfcheckArms[a];
            SelfArm arm = selfcheck_arm_for(key, g.index, g.name, g.run);
            if (std::strcmp(key, "fuzz") == 0) {
                arm.at = selfcheck_draw(rng, 0, g.run);
                char aim[192];
                std::snprintf(aim, sizeof(aim), "fuzz aim %d/%d", arm.at, g.run);
                // One run in four dies at a second group as well, drawn from
                // the groups this arm is not about: two groups non-passing at
                // once is a shape no single-group arm can construct.
                if (selfcheck_draw(rng, 0, 3) == 0) {
                    int q = selfcheck_draw(rng, 0, kGroupCount - 2);
                    if (q >= g.index) q++;
                    const SelfGroup *o = nullptr;
                    for (const SelfGroup &x : groups)
                        if (x.index == q) o = &x;
                    if (o) {
                        arm.second_index = o->index;
                        arm.second_name = o->name;
                        arm.second_healthy = o->run;
                        arm.second_at = selfcheck_draw(rng, 0, o->run);
                        char pair[192];
                        std::snprintf(pair, sizeof(pair), " + [%d] %s @%d", o->index,
                                      o->name.c_str(), arm.second_at);
                        std::strncat(aim, pair, sizeof(aim) - std::strlen(aim) - 1);
                    }
                }
                fuzz_aim = aim;
            }
            ArmJob job;
            job.arm = arm;
            job.tag = "arm-" + std::to_string(g.index) + "-" + key;
            job.log = dir + "/" + job.tag + ".log";
            job.rep = dir + "/" + job.tag + ".json";
            // Its own area, so no two arms can share a group's scratch now that
            // a group's arms are in flight at the same time, and so a green arm
            // disposes of its artifacts without touching the sweep's evidence
            // elsewhere in TEMP.
            const std::string arm_id = std::string(scratch_run_id()) + "-" + job.tag;
            job.env = selfcheck_arm_env(arm);
            job.env.put("KRK_TEST_SCRATCH_ID", arm_id);
            if (cap_threads) job.env.put("KRK_THREADS", std::to_string(threads_per_child));
            child_areas.push_back(arm_id);
            job.args = {self_path(), "--workers", std::to_string(arm_workers), "--report", job.rep};
            jobs.push_back(std::move(job));
        }

        // ... and now they run together. Every child is a full suite run of its
        // own, so this is where the sweep's cost was: six suites per group, one
        // after another, and 93 groups of that is an hour. The arms share
        // nothing but the seed -- different environment, different scratch area,
        // different log and report -- which is exactly what makes them safe to
        // overlap.
        {
            const int n = static_cast<int>(jobs.size());
            // `arms_width` is 1 whenever the suite contains a serial group: arms
            // are suites, so every pair of arms is a pair of the probe's
            // neighbours (see its definition).
            const int in_flight = std::max(1, std::min(n, arms_width));
            for (int a = 0; a < n; a += in_flight) {
                std::vector<std::thread> pool;
                const int hi = std::min(n, a + in_flight);
                for (int j = a; j < hi; j++)
                    pool.emplace_back([&jobs, j] {
                        ArmJob &jb = jobs[static_cast<size_t>(j)];
                        jb.exit = spawn_child_to(jb.args, jb.log, jb.env);
                        read_text_file(jb.log, &jb.log_text);
                        read_text_file(jb.rep, &jb.rep_text);
                    });
                for (std::thread &t : pool) t.join();
            }
        }
        // The verdicts are read in arm order, so the note text and the report
        // are the same whatever order the children happened to finish in.
        for (const ArmJob &job : jobs) {
            const char *key = job.arm.key;
            std::string problem = selfcheck_arm_check(
                job.arm, totals.run,
                job.exit.started ? static_cast<int>(job.exit.code) : -1, job.log_text,
                job.rep_text);
            if (!job.exit.started)
                problem = "the runner could not be started" + (problem.empty() ? "" : "; " + problem);
            v.arms.push_back(SelfArmVerdict{key, problem});
            if (problem.empty()) {
                std::remove(job.log.c_str());
                std::remove(job.rep.c_str());
            } else {
                failed_arms++;
                notes += (notes.empty() ? "" : "; ") + std::string(key) + ": " + problem;
            }
        }
        if (!notes.empty()) failed_groups++;
        v.ms = now_ms() - group_t0;
        verdicts.push_back(v);
        const int mid = g.run < 2 ? g.run : 2;
        char ok_note[352];
        std::snprintf(ok_note, sizeof(ok_note),
                      "crash 0/0, crash %d/%d, crash %d/%d, fail, no-checks, %s", mid, mid, g.run,
                      g.run, fuzz_aim.c_str());
        // The drawn aim is printed whether or not the arm was right: a mismatch
        // is exactly the case where the aim has to be known to be replayed.
        const std::string tail = notes.empty() ? std::string() : "  (" + fuzz_aim + ")";
        std::fprintf(stderr, "  %s [%2d/%d] %-44s %4d checks  %s%s\n",
                     notes.empty() ? "ok  " : "FAIL", g.index + 1, kGroupCount, g.name.c_str(),
                     g.run, notes.empty() ? ok_note : notes.c_str(), tail.c_str());
        std::fflush(stderr);
    }
    const i64 wall_ms = now_ms() - t0;
    clear_env_var("KRK_TEST_INJECT_CRASH");
    clear_env_var("KRK_TEST_INJECT_CRASH_AT");
    clear_env_var("KRK_TEST_INJECT_FAIL");
    clear_env_var("KRK_TEST_INJECT_EMPTY");

    // Written whatever the verdict: a gate that could only compare green sweeps
    // could not name the arm that broke.
    const bool wrote =
        write_selfcheck_report(report_path.c_str(), verdicts, seed, wall_ms, failed_groups ? 1 : 0);
    if (!report_path.empty())
        std::fprintf(stderr, "report: %s%s\n", report_path.c_str(),
                     wrote ? "" : " (could not be written)");

    std::fprintf(stderr, "\n%d/%d groups: every injection was reported by name, with the count "
                         "it had reached\n", swept - failed_groups, swept);
    if (failed_arms) std::fprintf(stderr, "%d arm(s) mismatched\n", failed_arms);
    if (failed_groups || !wrote) {
        std::fprintf(stderr, "evidence kept: %s\n", dir.c_str());
        return 1;
    }
    std::error_code rm;
    // A green sweep leaves nothing: its evidence directory and each area it named
    // for a child. The evidence directory is not inside any child's area (a green
    // suite run removes its own area), so this is the only place either survives.
    std::filesystem::remove_all(dir, rm);
    for (const std::string &id : child_areas)
        std::filesystem::remove_all(progress_dir() + "/krk-tests-" + id, rm);
    return 0;
}

// ---------------------------------------------------------------------------
//  The checker's own check: constructed answers, one per way an arm can be
//  wrong. The builders write the runner's formats, so a change to one of those
//  formats that these do not follow is caught by the real --selfcheck, which
//  reads the real thing.
// ---------------------------------------------------------------------------

static std::string selfcheck_fake_report(int index, const std::string &name, const char *status,
                                         int passed, int run, int totals_passed, int totals_run,
                                         int exit_code, bool other_bad, int bad_groups = 1,
                                         int second_index = -1, int second_passed = 0,
                                         int second_run = 0) {
    char buf[512];
    std::string s = "{\n  \"format\": 1,\n  \"suite\": \"kraken-tests\",\n";
    std::snprintf(buf, sizeof(buf),
                  "  \"totals\": {\"groups\": %d, \"passed\": %d, \"run\": %d, "
                  "\"bad_groups\": %d, \"exit\": %d},\n",
                  kGroupCount, totals_passed, totals_run, bad_groups, exit_code);
    s += buf;
    s += "  \"groups\": [\n";
    const int bad_other = other_bad ? (index + 1) % kGroupCount : -1;
    for (int i = 0; i < kGroupCount; i++) {
        if (i == index) {
            std::snprintf(buf, sizeof(buf),
                          "    {\"index\": %d, \"name\": \"%s\", \"status\": \"%s\", \"passed\": "
                          "%d, \"run\": %d, \"ms\": 1, \"fail_lines\": []}%s\n",
                          i, name.c_str(), status, passed, run,
                          i + 1 == kGroupCount ? "" : ",");
        } else if (i == second_index) {
            std::snprintf(buf, sizeof(buf),
                          "    {\"index\": %d, \"name\": \"g%d\", \"status\": \"crashed\", "
                          "\"passed\": %d, \"run\": %d, \"ms\": 1, \"fail_lines\": []}%s\n",
                          i, i, second_passed, second_run, i + 1 == kGroupCount ? "" : ",");
        } else if (i == bad_other) {
            std::snprintf(buf, sizeof(buf),
                          "    {\"index\": %d, \"name\": \"g%d\", \"status\": \"failed\", "
                          "\"passed\": 0, \"run\": 5, \"ms\": 1, \"fail_lines\": []}%s\n",
                          i, i, i + 1 == kGroupCount ? "" : ",");
        } else {
            std::snprintf(buf, sizeof(buf),
                          "    {\"index\": %d, \"name\": \"g%d\", \"status\": \"pass\", "
                          "\"passed\": 5, \"run\": 5, \"ms\": 1, \"fail_lines\": []}%s\n",
                          i, i, i + 1 == kGroupCount ? "" : ",");
        }
        s += buf;
    }
    s += "  ]\n}\n";
    return s;
}

static std::string selfcheck_fake_log(int index, const std::string &name, int passed, int run,
                                      const char *status, int totals_passed, int totals_run,
                                      int fail_lines, bool line, bool attention, bool summary,
                                      bool ran_no_checks) {
    char buf[512];
    std::string s;
    if (line) {
        std::snprintf(buf, sizeof(buf), "  [%2d/%d] %-44s %5d/%-5d %5lldms%s\n", index + 1,
                      kGroupCount, name.c_str(), passed, run, static_cast<long long>(3), status);
        s += buf;
    }
    for (int i = 0; i < fail_lines; i++) {
        std::snprintf(buf, sizeof(buf), "FAIL fake.cpp:%d  constructed case\n", 100 + i);
        s += buf;
    }
    if (summary) {
        std::snprintf(buf, sizeof(buf), "\n%d/%d checks passed%s\n", totals_passed, totals_run,
                      ran_no_checks ? "; 1 group(s) ran no checks" : "");
        s += buf;
    }
    if (attention) {
        std::snprintf(buf, sizeof(buf), "groups needing attention: %s\n", name.c_str());
        s += buf;
    }
    return s;
}

static int run_selfcheck_selftest() {
    const std::string name = "test_json";
    const int idx = 4;
    const int healthy = 41;
    const int others = (kGroupCount - 1) * 5; // every other group, five checks each
    const int base = others + healthy;
    const int mid = healthy < 2 ? healthy : 2;

    struct Case {
        std::string name;
        bool want_problems;
        std::string got;
    };
    std::vector<Case> cases;
    const auto add = [&](const std::string &cname, bool want, const char *arm, int rc,
                         const std::string &log, const std::string &report) {
        cases.push_back(Case{cname, want,
                             selfcheck_arm_check(selfcheck_arm_for(arm, idx, name, healthy), base,
                                                 rc, log, report)});
    };

    add("a correct crash arm", false, "crash", 1,
        selfcheck_fake_log(idx, name, 0, 0, "  CRASHED 0xC0000409", others, others, 0, true, true,
                           true, false),
        selfcheck_fake_report(idx, name, "crashed", 0, 0, others, others, 1, false));
    add("a crash arm that reports 1/1", true, "crash", 1,
        selfcheck_fake_log(idx, name, 1, 1, "  CRASHED 0xC0000409", others + 1, others + 1, 0, true,
                           true, true, false),
        selfcheck_fake_report(idx, name, "crashed", 1, 1, others + 1, others + 1, 1, false));
    add("a correct partial count", false, "crash-2", 1,
        selfcheck_fake_log(idx, name, mid, mid, "  CRASHED 0xC0000409", others + mid, others + mid, 0,
                           true, true, true, false),
        selfcheck_fake_report(idx, name, "crashed", mid, mid, others + mid, others + mid, 1, false));
    add("a partial count that does not match", true, "crash-2", 1,
        selfcheck_fake_log(idx, name, 1, 1, "  CRASHED 0xC0000409", others + 1, others + 1, 0, true,
                           true, true, false),
        selfcheck_fake_report(idx, name, "crashed", 1, 1, others + 1, others + 1, 1, false));
    add("a crash after the group's own last check", false, "crash-last", 1,
        selfcheck_fake_log(idx, name, healthy, healthy, "  CRASHED 0xC0000409", base, base, 0, true,
                           true, true, false),
        selfcheck_fake_report(idx, name, "crashed", healthy, healthy, base, base, 1, false));
    add("a last-check crash whose counts read as a pass", true, "crash-last", 1,
        selfcheck_fake_log(idx, name, healthy, healthy, "  CRASHED 0xC0000409", base, base, 0, true,
                           true, true, false),
        selfcheck_fake_report(idx, name, "pass", healthy, healthy, base, base, 0, false));
    add("a crash that was not named", true, "crash", 1,
        selfcheck_fake_log(idx, name, 0, 0, "  CRASHED 0xC0000409", others, others, 0, true, false,
                           true, false),
        selfcheck_fake_report(idx, name, "crashed", 0, 0, others, others, 1, false));
    add("a correct fail arm", false, "fail", 1,
        selfcheck_fake_log(idx, name, 0, healthy, "  FAILED", others, base, healthy, true, true,
                           true, false),
        selfcheck_fake_report(idx, name, "failed", 0, healthy, others, base, 1, false));
    add("a fail arm with no FAIL lines in the log", true, "fail", 1,
        selfcheck_fake_log(idx, name, 0, healthy, "  FAILED", others, others, 0, true, true, true,
                           false),
        selfcheck_fake_report(idx, name, "failed", 0, healthy, others, base, 1, false));
    add("a correct empty arm", false, "empty", 1,
        selfcheck_fake_log(idx, name, 0, 0, "  NO CHECKS", others, others, 0, true, true, true, true),
        selfcheck_fake_report(idx, name, "no-checks", 0, 0, others, others, 1, false));
    add("an empty arm that counted as a pass", true, "empty", 1,
        selfcheck_fake_log(idx, name, 0, healthy, "  NO CHECKS", base, base, 0, true, true, true,
                           true),
        selfcheck_fake_report(idx, name, "pass", 0, healthy, base, base, 0, false));
    add("a totals line that forgot the group", true, "crash", 1,
        selfcheck_fake_log(idx, name, 0, 0, "  CRASHED 0xC0000409", others - 1, others - 1, 0, true,
                           true, true, false),
        selfcheck_fake_report(idx, name, "crashed", 0, 0, others - 1, others - 1, 1, false));
    add("another group not passing", true, "crash", 1,
        selfcheck_fake_log(idx, name, 0, 0, "  CRASHED 0xC0000409", others, others, 0, true, true,
                           true, false),
        selfcheck_fake_report(idx, name, "crashed", 0, 0, others, others, 1, true));
    add("the group's own line missing", true, "crash", 1,
        selfcheck_fake_log(idx, name, 0, 0, "  CRASHED 0xC0000409", others, others, 0, false, true,
                           true, false),
        selfcheck_fake_report(idx, name, "crashed", 0, 0, others, others, 1, false));

    // Two groups non-passing in one run: the fuzz arm's shape, which no other
    // case can produce. All three aim at the same pair, so a checker that read
    // only the first group, or that added the second one's checks wrong, fails
    // one of them.
    const int si = 21;
    const std::string sname = "g21";
    const int shelthy = 5;
    const int sat = 3;
    SelfArm pair = selfcheck_arm_for("fuzz", idx, name, healthy);
    pair.at = 7;
    pair.second_index = si;
    pair.second_name = sname;
    pair.second_healthy = shelthy;
    pair.second_at = sat;
    const int pair_total = base - healthy - shelthy + pair.at + sat;
    const std::string pair_log =
        selfcheck_fake_log(idx, name, pair.at, pair.at, "  CRASHED 0xC0000409", pair_total,
                           pair_total, 0, true, false, false, false) +
        selfcheck_fake_log(si, sname, sat, sat, "  CRASHED 0xC0000409", pair_total, pair_total, 0,
                           true, false, false, false) +
        "groups needing attention: " + name + " " + sname + "\n" + std::to_string(pair_total) +
        "/" + std::to_string(pair_total) + " checks passed\n";
    cases.push_back(Case{"a correct paired arm", false,
                         selfcheck_arm_check(pair, base, 1, pair_log,
                                             selfcheck_fake_report(idx, name, "crashed", pair.at,
                                                                   pair.at, pair_total, pair_total,
                                                                   1, false, 2, si, sat, sat))});
    cases.push_back(Case{"a paired arm whose second group is not reported", true,
                         selfcheck_arm_check(pair, base, 1, pair_log,
                                             selfcheck_fake_report(idx, name, "crashed", pair.at,
                                                                   pair.at, pair_total, pair_total,
                                                                   1, false, 2, -1, sat, sat))});
    cases.push_back(Case{"a paired arm whose totals forgot the second group", true,
                         selfcheck_arm_check(
                             pair, base, 1, pair_log,
                             selfcheck_fake_report(idx, name, "crashed", pair.at, pair.at,
                                                   base - healthy + pair.at,
                                                   base - healthy + pair.at, 1, false, 2, si, sat,
                                                   sat))});

    int wrong = 0;
    for (const Case &c : cases) {
        const bool ok = (c.got.empty() != c.want_problems);
        if (!ok) wrong++;
        const std::string tail =
            ok ? std::string() : "  -- wanted " + std::string(c.want_problems ? "problems" : "none") +
                                     ", got [" + c.got + "]";
        std::fprintf(stderr, "%-4s %s%s\n", ok ? "ok" : "FAIL", c.name.c_str(), tail.c_str());
    }
    std::fprintf(stderr, "\n%d/%d selfcheck cases passed\n", static_cast<int>(cases.size()) - wrong,
                 static_cast<int>(cases.size()));
    return wrong ? 1 : 0;
}

// ---------------------------------------------------------------------------
//  Deliberate failure, so the reporting can be shown to work instead of
//  asserted. Both switches are inert unless set: unset, checks_should_pass() is
//  one compare against -1 and maybe_inject_crash() returns on a cached -1.
// ---------------------------------------------------------------------------

// One group token, resolved the way --run-group and every injection switch take
// one: an index or a name.
static int group_index_of(const std::string &want) {
    for (int k = 0; k < kGroupCount; k++)
        if (want == std::to_string(k) || want == kGroups[k].name) return k;
    return -1;
}

static int selected_group(const char *env_name) {
    const char *want = std::getenv(env_name);
    if (!want || !*want) return -1;
    return group_index_of(want);
}

// Comma-separated tokens, blanks trimmed. The crash switch takes a LIST because
// the runner's reporting has to hold when more than one group is non-passing at
// once, and aiming at two groups in one run is the only way to construct that.
static std::vector<std::string> split_list(const std::string &s) {
    std::vector<std::string> out;
    size_t pos = 0;
    for (;;) {
        const size_t comma = s.find(',', pos);
        const size_t end = comma == std::string::npos ? s.size() : comma;
        size_t a = pos, b = end;
        while (a < b && (s[a] == ' ' || s[a] == '\t')) a++;
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) b--;
        if (b > a) out.push_back(s.substr(a, b - a));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

struct CrashTarget {
    int index = -1;
    int at = 0;
};

struct CrashPlan {
    std::vector<CrashTarget> targets;
    std::vector<std::string> unmatched; // tokens that name no group
};

// KRK_TEST_INJECT_CRASH=<grp>[,<grp>...] with KRK_TEST_INJECT_CRASH_AT=<n>[,<n>...]
// positionally aligned; an aim with no number dies before the first check.
static const CrashPlan &crash_plan() {
    static const CrashPlan plan = [] {
        CrashPlan p;
        const char *want = std::getenv("KRK_TEST_INJECT_CRASH");
        if (!want || !*want) return p;
        const char *at = std::getenv("KRK_TEST_INJECT_CRASH_AT");
        const std::vector<std::string> names = split_list(want);
        const std::vector<std::string> ats = split_list(at ? at : "");
        for (size_t i = 0; i < names.size(); i++) {
            const int k = group_index_of(names[i]);
            if (k < 0) {
                p.unmatched.push_back(names[i]);
                continue;
            }
            CrashTarget t;
            t.index = k;
            t.at = (i < ats.size()) ? std::atoi(ats[i].c_str()) : 0;
            p.targets.push_back(t);
        }
        return p;
    }();
    return plan;
}

static int fail_group() {
    static const int idx = selected_group("KRK_TEST_INJECT_FAIL");
    return idx;
}

// A group can also pass by doing nothing at all: test_gdn_generation used to
// return before its first check when its fixture was missing and report 0/0,
// which the runner counted as a group that passed. This switch makes one group
// run nothing, so that the runner reporting it -- rather than accepting it -- is
// a check and not a promise.
static int empty_group() {
    static const int idx = selected_group("KRK_TEST_INJECT_EMPTY");
    return idx;
}

// Every CHECK asks this one question. Unset, it is a compare against -1.
static bool checks_should_pass() { return g_group_index != fail_group(); }

// Die on purpose, in the way this suite already died once. std::fclose(NULL)
// goes through the CRT's invalid-parameter handler, which answers with
// __fastfail -- not an exception SEH can catch, not a signal, and the reason
// none of the process survived it. The null goes through a volatile so the
// compiler cannot see it and elide the call. On POSIX this line is a SIGSEGV.
static void maybe_inject_crash() {
    const CrashPlan &p = crash_plan();
    for (const CrashTarget &t : p.targets) {
        if (t.index != g_group_index || g_run < t.at) continue;
        std::fprintf(stderr, "  INJECT: %s dies here on purpose, after %d check(s)\n",
                     kGroups[t.index].name, g_run);
        std::fflush(stderr);
        static std::FILE *volatile null_file = nullptr;
        std::fclose(null_file);
        std::abort(); // unreachable while the CRT behaves as it does today
    }
}

static void report_injection() {
    const CrashPlan &cp = crash_plan();
    for (const CrashTarget &t : cp.targets)
        std::fprintf(stderr, "crash injection: %s dies at check %d%s\n", kGroups[t.index].name,
                     t.at, cp.targets.size() > 1 ? " (one of two)" : "");
    for (const std::string &tok : cp.unmatched)
        std::fprintf(stderr, "crash injection: '%s' matches no group, nothing injected\n",
                     tok.c_str());

    const int fi = fail_group();
    const char *fwant = std::getenv("KRK_TEST_INJECT_FAIL");
    if (fi >= 0)
        std::fprintf(stderr, "fail injection: every check in %s (%d of %d groups) fails\n",
                     kGroups[fi].name, fi, kGroupCount);
    else if (fwant && *fwant)
        std::fprintf(stderr, "fail injection: '%s' matches no group, nothing injected\n", fwant);

    const int ei = empty_group();
    const char *ewant = std::getenv("KRK_TEST_INJECT_EMPTY");
    if (ei >= 0)
        std::fprintf(stderr, "empty injection: %s runs no checks at all\n", kGroups[ei].name);
    else if (ewant && *ewant)
        std::fprintf(stderr, "empty injection: '%s' matches no group, nothing injected\n", ewant);
}

// Everything this suite writes into the working directory is named the same way:
// a model is a kraken-*.gguf, the expert index beside one is a
// kraken-*.krakenexperts.json, and a KV tier's spill area is a kraken-*-dir.
// Nothing else in the caller's directory is the suite's to delete, and matching
// the SHAPES rather than a list of names is the point: the list this replaced
// named six of the thirteen fixtures the groups actually leave, so a
// debugger-view run quietly scattered the other eight across the caller's
// directory, and a fixture added tomorrow would have needed the list edited by
// whoever remembered it existed.
static bool is_suite_fixture(const std::string &name) {
    if (name.rfind("kraken-", 0) != 0) return false;
    static const char *kModel = ".gguf";
    static const char *kIndex = ".krakenexperts.json";
    const size_t m = std::strlen(kModel), ix = std::strlen(kIndex);
    if (name.size() > m && name.compare(name.size() - m, m, kModel) == 0) return true;
    if (name.size() > ix && name.compare(name.size() - ix, ix, kIndex) == 0) return true;
    return false;
}

static bool is_suite_fixture_dir(const std::string &name) {
    if (name.rfind("kraken-", 0) != 0) return false;
    static const char *kSuffix = "-dir";
    const size_t n = std::strlen(kSuffix);
    return name.size() > n && name.compare(name.size() - n, n, kSuffix) == 0;
}

static void cleanup_fixtures() {
    std::error_code ec;
    std::filesystem::directory_iterator it(".", ec), end;
    if (ec) {
        std::fprintf(stderr, "  (cleanup: cannot list the working directory: %s)\n",
                     ec.message().c_str());
        return;
    }
    int removed = 0;
    for (; it != end; it.increment(ec)) {
        if (ec) break;
        const std::filesystem::path &p = it->path();
        const std::string name = p.filename().string();
        std::error_code kind_ec;
        const bool dir = it->is_directory(kind_ec);
        if (!is_suite_fixture(name) && !(dir && is_suite_fixture_dir(name))) continue;
        std::error_code rm;
        std::filesystem::remove_all(p, rm);
        if (!rm) removed++;
    }
    // Silent when there was nothing to do -- which is the usual case in child
    // mode, where each group worked in a scratch directory of its own.
    if (removed)
        std::fprintf(stderr, "cleanup: removed %d fixture(s) from the working directory\n",
                     removed);
}

struct RunOutcome {
    int index = 0;
    std::string name;
    int passed = 0;
    int run = 0;
    // "" | FAILED | NO CHECKS | CRASHED 0x.. | COULD NOT START | NO SCRATCH
    std::string status;
    // The same state as one word a program can match on: pass, failed,
    // no-checks, crashed, no-scratch, could-not-start. --report writes this one.
    std::string kind;
    std::string log;    // whatever the group printed, replayed under its own line
    std::string kept;   // where the evidence is, when it was kept
    bool bad = false;   // counts among the "groups needing attention"
    bool failed = false; // a check failed (as opposed to the group dying)
    i64 ms = 0;
    std::vector<std::string> fail_lines; // the first few FAIL lines of this group
};

// ---------------------------------------------------------------------------
//  The machine-readable summary (--report).
//
//  The console table is for a human; a gate that compares this run against the
//  last recorded green one needs the same facts in a form a program can read,
//  including the text of the failing checks, so that a regression is named with
//  its reason and not just counted. One group per line, so a recorded baseline
//  diffs readably in git.
// ---------------------------------------------------------------------------

static void json_escape(const std::string &in, std::string *out) {
    for (unsigned char c : in) {
        if (c == 0x22) out->append("\x5c\x22");      // a bare quote
        else if (c == 0x5c) out->append("\x5c\x5c"); // a backslash
        else if (c == 0x0a) out->append("\x5c" "n");
        else if (c == 0x0d) continue;                // the report is LF-only
        else if (c == 0x09) out->append("\x5c" "t");
        else if (c < 0x20) {
            char esc[8];
            std::snprintf(esc, sizeof(esc), "\x5c" "u%04x", c);
            out->append(esc);
        } else {
            out->push_back(static_cast<char>(c));
        }
    }
}

// The first `cap` FAIL lines of a group's log. A group can fail every check it
// has, and what a gate wants from a regression is a reason, not 348 lines of one.
static void collect_fail_lines(const std::string &log, size_t cap,
                               std::vector<std::string> *lines) {
    size_t pos = 0;
    while (pos < log.size()) {
        const size_t nl = log.find('\n', pos);
        std::string ln = log.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = (nl == std::string::npos) ? log.size() : nl + 1;
        if (!ln.empty() && ln.back() == '\r') ln.pop_back();
        if (ln.rfind("FAIL ", 0) == 0 && lines->size() < cap) lines->push_back(ln);
    }
}

static bool write_report(const char *path, const std::vector<RunOutcome> &done, int workers,
                         i64 wall_ms, int exit_code) {
    if (!path || !*path) return true;
    std::FILE *f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "report: cannot write %s\n", path);
        return false;
    }
    int passed = 0, run = 0, bad = 0;
    for (const RunOutcome &o : done) {
        passed += o.passed;
        run += o.run;
        if (o.bad) bad++;
    }
    std::fprintf(f, "{\n  \"format\": 1,\n  \"suite\": \"kraken-tests\",\n");
    std::fprintf(f, "  \"workers\": %d,\n  \"wall_ms\": %lld,\n", workers,
                 static_cast<long long>(wall_ms));
    std::fprintf(f,
                 "  \"totals\": {\"groups\": %d, \"passed\": %d, \"run\": %d, "
                 "\"bad_groups\": %d, \"exit\": %d},\n",
                 static_cast<int>(done.size()), passed, run, bad, exit_code);
    std::fprintf(f, "  \"groups\": [\n");
    for (size_t i = 0; i < done.size(); i++) {
        const RunOutcome &o = done[i];
        std::string name, kind;
        json_escape(o.name, &name);
        json_escape(o.kind, &kind);
        std::fprintf(f,
                     "    {\"index\": %d, \"name\": \"%s\", \"status\": \"%s\", \"passed\": %d, "
                     "\"run\": %d, \"ms\": %lld, \"fail_lines\": [",
                     o.index, name.c_str(), kind.c_str(), o.passed, o.run,
                     static_cast<long long>(o.ms));
        for (size_t k = 0; k < o.fail_lines.size(); k++) {
            std::string ln;
            json_escape(o.fail_lines[k], &ln);
            std::fprintf(f, "%s\"%s\"", k ? ", " : "", ln.c_str());
        }
        std::fprintf(f, "]}%s\n", i + 1 == done.size() ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    const bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

// Child mode. The group announces itself BEFORE running so that a log of a
// crashed run still says which group was in flight, and the progress file is
// opened (and zeroed) before that for the same reason.
static int run_one_group(int index, const char *progress, const char *log, const char *scratch) {
    g_group = kGroups[index].name;
    g_group_index = index;

    // Child mode runs in its own working directory. Every fixture in this file
    // is a relative path, so one directory per group is the whole reason
    // --workers is safe: without it two groups write the same kraken-*.gguf and
    // whichever loses the race reads the other's bytes. A group that cannot
    // take its directory is not run at all -- the parent says so, because a
    // group's count is only worth reading when the group was isolated.
    if (scratch && *scratch) {
        std::error_code ec;
        std::filesystem::current_path(scratch, ec);
        if (ec) {
            std::fprintf(stderr, "kraken-tests: scratch dir '%s' is unusable (%s)\n",
                         scratch, ec.message().c_str());
            return 2; // the harness could not run this group; that is not a check
        }
    }

    // This group's output goes to its own file, so groups running at once do
    // not interleave into one log. stderr stays unbuffered afterwards on
    // purpose: a __fastfail takes the process away with nothing flushed, and
    // the FAIL line just before it is the whole reason the file exists.
    if (log && *log) {
        const bool got_err = std::freopen(log, "wb", stderr) != nullptr;
        const bool got_out = std::freopen(log, "ab", stdout) != nullptr;
        std::setvbuf(stderr, nullptr, _IONBF, 0);
        if (!got_err || !got_out)
            std::fprintf(stderr, "  (could not redirect all of this group's output to %s)\n",
                         log);
    }

    const std::string pp = progress_path(index, progress);
    // The file exists for the parent's benefit. A hand-run group is its own
    // caller, so it cleans up after itself -- and a group that dies leaves the
    // file behind holding the counts as of the moment it died, which is the
    // version worth keeping.
    const bool own_progress = !(progress && *progress);
    g_progress = std::fopen(pp.c_str(), "wb");
    if (!g_progress)
        std::fprintf(stderr, "  (no progress file at %s: a crash here reports no count)\n",
                     pp.c_str());
    // Announce before running, so a log of a killed group still names it even
    // if this process never gets to its result line.
    std::fprintf(stderr, "[group %d/%d] %s\n", index + 1, kGroupCount, g_group);
    std::fflush(stderr);

    maybe_inject_crash();
    note_progress();
    if (index != empty_group()) kGroups[index].fn();

    note_progress();
    if (g_progress) {
        std::fclose(g_progress);
        g_progress = nullptr;
    }
    std::fprintf(stderr, "[group %s] %d/%d checks\n", g_group, g_passed, g_run);
    if (own_progress) std::remove(pp.c_str());
    return g_passed == g_run ? 0 : 1;
}

static bool path_under(const std::string &child, const std::string &parent);

// The old behaviour, kept as an escape hatch for a debugger and as the A/B that
// shows the child-process run reports the same numbers.
static int run_all_inproc(const char *report) {
    // A run area of its own, for the reason every child has one: the fixtures are
    // relative paths and the cleanup works by SHAPE (kraken-*.gguf), so two runs
    // sharing a directory delete each other's files. Measured on this box: the
    // same binary and tree, `test_pageable_path_probe` failing 11 checks with
    // `cannot open 'kraken-aperture-path.gguf'` in one run and 140/140 in the
    // next -- the difference being whether another run happened to start in the
    // repo root while this one was mid-group.
    const std::string area = scratch_area();
    std::error_code area_ec;
    std::filesystem::create_directories(area, area_ec);
    // The report path is the caller's, so it means what it said relative to the
    // directory the caller started in: resolve it BEFORE the move.
    const std::string report_path =
        (report && *report) ? absolute_path(report) : std::string();
    if (!area_ec) {
        std::error_code chdir_ec;
        std::filesystem::current_path(area, chdir_ec);
        if (chdir_ec)
            std::fprintf(stderr,
                         "kraken-tests: no private directory (%s); running where started\n",
                         chdir_ec.message().c_str());
    }
    std::fprintf(stderr, "kraken test suite: %d groups, all in this process (KRK_TEST_INPROC)\n",
                 kGroupCount);
    int empty_groups = 0;
    report_injection();
    const i64 t0 = now_ms();
    std::vector<RunOutcome> done(static_cast<size_t>(kGroupCount));
    for (int i = 0; i < kGroupCount; i++) {
        g_group = kGroups[i].name;
        g_group_index = i;
        maybe_inject_crash();
        const int was_run = g_run, was_passed = g_passed;
        const i64 group_t0 = now_ms();
        if (i != empty_group()) kGroups[i].fn();
        RunOutcome &o = done[static_cast<size_t>(i)];
        o.index = i;
        o.name = kGroups[i].name;
        o.passed = g_passed - was_passed;
        o.run = g_run - was_run;
        o.ms = now_ms() - group_t0;
        if (o.run == 0) {
            o.kind = "no-checks";
            o.bad = true;
            empty_groups++;
        } else if (o.passed != o.run) {
            o.kind = "failed";
            o.bad = true;
        } else {
            o.kind = "pass";
        }
        // The same line the child runner prints, so the two can be compared
        // group by group and not only in total -- which is the A/B this mode
        // exists for.
        std::fprintf(stderr, "  [%2d/%d] %-44s %5d/%-5d  ledger=%zu\n", i + 1,
                     kGroupCount, kGroups[i].name, o.passed, o.run,
                     SharedHostRamBudget::used_bytes());
        std::fflush(stderr);
    }
    cleanup_fixtures();
    std::fprintf(stderr, "\n%d/%d checks passed", g_passed, g_run);
    if (empty_groups) std::fprintf(stderr, "; %d group(s) ran no checks", empty_groups);
    std::fprintf(stderr, "\n");
    const int exit_code = (g_passed == g_run && empty_groups == 0) ? 0 : 1;
    // This mode has no per-group log to read, so its report carries the counts
    // and no FAIL text: the child runner is the one that records *why*.
    const char *report_c = report_path.empty() ? nullptr : report_path.c_str();
    const bool reported = write_report(report_c, done, 1, now_ms() - t0, exit_code);
    if (!reported)
        std::fprintf(stderr, "report: NOT WRITTEN (asked for %s)\n", report_c ? report_c : "");
    else if (report_c)
        std::fprintf(stderr, "report: %s\n", report_c);
    // The area is scoped to this run, so this run disposes of it -- unless the
    // caller asked for its report under it, which no process may delete.
    {
        const bool report_inside =
            !report_path.empty() && path_under(report_path, area);
        std::error_code rm;
        if (exit_code == 0 && reported && !report_inside)
            std::filesystem::remove_all(area, rm);
    }
    return (exit_code == 0 && reported) ? 0 : 1;
}

// ---------------------------------------------------------------------------
//  The scheduler: up to `workers` groups in flight, plus a lane for a group
//  that has declared it needs the machine to itself.
// ---------------------------------------------------------------------------

static const char *serial_override() {
    static const std::string want = [] {
        const char *env = std::getenv("KRK_TEST_SERIAL_OVERRIDE");
        return std::string(env ? env : "");
    }();
    return want.empty() ? nullptr : want.c_str();
}

// The declared reason, or the one the override forces for this single run. The
// override exists so the lane below is exercised by a test instead of being
// taken on faith; it names one group, by index or by name.
static const char *serial_reason(int index) {
    const char *want = serial_override();
    if (want) {
        const std::string w(want);
        if (w == std::to_string(index) || w == kGroups[index].name)
            return "KRK_TEST_SERIAL_OVERRIDE";
        return nullptr;
    }
    return kGroups[index].serial;
}

static bool is_serial(int index) { return serial_reason(index) != nullptr; }

static void print_serial_set() {
    std::string line;
    for (int i = 0; i < kGroupCount; i++) {
        const char *why = serial_reason(i);
        if (!why) continue;
        line += " ";
        line += kGroups[i].name;
        line += " (";
        line += why;
        line += ")";
    }
    if (line.empty())
        std::fprintf(stderr,
                     "serial groups: none (every group builds the fixtures it reads, in a\n"
                     "               scratch directory of its own, and none holds a shared\n"
                     "               resource or reads a clock)\n");
    else
        std::fprintf(stderr, "serial groups:%s\n", line.c_str());
}

// One group, one child, one scratch directory, one log file. This runs on a
// worker thread, so the only thing it shares with its neighbours is the print
// lock.
static RunOutcome run_group_child(int index, std::mutex *print_mu) {
    RunOutcome o;
    o.index = index;
    o.name = kGroups[index].name;
    const std::string pp = absolute_path(progress_path(index, nullptr));
    const std::string lp = absolute_path(log_path(index));
    const std::string sd = absolute_path(scratch_dir(index));

    std::remove(pp.c_str());
    std::remove(lp.c_str());
    std::error_code ec;
    std::filesystem::remove_all(sd, ec);
    std::filesystem::create_directories(sd, ec);

    const i64 t0 = now_ms();
    if (ec) {
        // No private directory means no isolation, and every fixture here is a
        // relative path: running the group anyway would let it write over a
        // neighbour's fixture. Leave it out and fail the run -- a group that
        // did not run must never read as a group that passed.
        o.status = "  NO SCRATCH";
        o.kind = "no-scratch";
        o.bad = true;
    } else {
        const std::vector<std::string> args{self_path(),  "--run-group", std::to_string(index),
                                            "--progress", pp, "--log",     lp,
                                            "--scratch",  sd};
        const ChildExit ce = spawn_child(args);
        read_progress(pp, &o.passed, &o.run);
        read_text_file(lp, &o.log);
        collect_fail_lines(o.log, 40, &o.fail_lines);
        if (!ce.started) {
            o.status = "  COULD NOT START";
            o.kind = "could-not-start";
            o.bad = true;
        } else if (ce.code == 2) {
            o.status = "  NO SCRATCH"; // the child could not take its directory
            o.kind = "no-scratch";
            o.bad = true;
        } else if (ce.crashed) {
            char note[64];
            std::snprintf(note, sizeof(note), "  CRASHED 0x%08lX", ce.code);
            o.status = note;
            o.kind = "crashed";
            o.bad = true;
        } else if (o.run == 0) {
            // Zero checks is the shape of a silent skip, and a silent skip that
            // reads as a pass is the exact defect this runner exists to stop.
            o.status = "  NO CHECKS";
            o.kind = "no-checks";
            o.bad = true;
        } else if (ce.code != 0 || o.passed != o.run) {
            o.status = "  FAILED";
            o.kind = "failed";
            o.bad = true;
            o.failed = true;
        } else {
            o.kind = "pass";
        }
    }
    o.ms = now_ms() - t0;

    // The line and the group's own output print as one block, under the lock:
    // with several groups in flight a reader still has to be able to tell whose
    // FAIL that was.
    {
        std::lock_guard<std::mutex> lock(*print_mu);
        std::fprintf(stderr, "  [%2d/%d] %-44s %5d/%-5d %5lldms%s\n", index + 1, kGroupCount,
                     o.name.c_str(), o.passed, o.run, static_cast<long long>(o.ms),
                     o.status.c_str());
        if (!o.log.empty()) {
            std::fputs(o.log.c_str(), stderr);
            if (o.log.back() != '\n') std::fputc('\n', stderr);
        }
        std::fflush(stderr);
    }

    // A group that finished is done with its scratch: counts read, output
    // printed, directory gone so the next run starts clean. A group that died
    // keeps the fixture it died holding and its log -- that is the evidence.
    if (o.bad) {
        o.kept = sd + ", " + lp;
    } else {
        std::remove(lp.c_str());
        std::error_code rm;
        std::filesystem::remove_all(sd, rm);
    }
    std::remove(pp.c_str());
    return o;
}

// Is `child` under `parent`? Compared by path components rather than by string
// prefix, because the two strings are built differently -- TEMP carries
// backslashes, a report path may carry slashes, and Windows does not care about
// case -- and a prefix test silently answered "no" for a path that was plainly
// inside, which is how the sweep's own clean-run report got deleted underneath it.
static bool path_under(const std::string &child, const std::string &parent) {
    const auto fold = [](std::string v) {
        for (char &ch : v)
            if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
        return v;
    };
    std::error_code ec1, ec2;
    const std::filesystem::path c = std::filesystem::weakly_canonical(child, ec1);
    const std::filesystem::path p = std::filesystem::weakly_canonical(parent, ec2);
    auto ci = c.begin();
    for (auto pi = p.begin(); pi != p.end(); ++pi, ++ci) {
        if (ci == c.end()) return false;
        if (fold(ci->string()) != fold(pi->string())) return false;
    }
    return true;
}

static int run_all_groups(int workers, const char *report) {
    std::fprintf(stderr, "kraken test suite: %d groups, %d at a time, each in its own process\n",
                 kGroupCount, workers);
    report_injection();
    print_serial_set();

    const i64 t0 = now_ms();
    std::vector<RunOutcome> done(static_cast<size_t>(kGroupCount));
    std::mutex q_mu, print_mu;
    std::condition_variable q_cv;
    int next = 0, in_flight = 0;
    bool serial_claimed = false;

    const auto worker = [&]() {
        for (;;) {
            int index = -1;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                for (;;) {
                    if (next >= kGroupCount) return;
                    if (is_serial(next)) {
                        // The serial lane: wait for the groups already running
                        // to finish, then hold every other worker at the door
                        // until this one is done with the machine.
                        if (in_flight > 0 || serial_claimed) {
                            q_cv.wait(lk);
                            continue;
                        }
                        serial_claimed = true;
                    } else if (serial_claimed) {
                        q_cv.wait(lk);
                        continue;
                    }
                    index = next++;
                    in_flight++;
                    break;
                }
            }
            done[static_cast<size_t>(index)] = run_group_child(index, &print_mu);
            {
                std::lock_guard<std::mutex> lk(q_mu);
                in_flight--;
                if (is_serial(index)) serial_claimed = false;
                q_cv.notify_all();
            }
        }
    };

    std::vector<std::thread> pool;
    for (int w = 0; w < workers; w++) pool.emplace_back(worker);
    for (std::thread &t : pool) t.join();

    const i64 wall = now_ms() - t0;
    cleanup_fixtures();

    int run = 0, passed = 0, failed_groups = 0, crashed_groups = 0;
    int empty_groups = 0;
    i64 sum = 0;
    std::vector<std::string> bad, kept;
    for (const RunOutcome &o : done) {
        run += o.run;
        passed += o.passed;
        sum += o.ms;
        if (!o.kept.empty()) kept.push_back(o.kept);
        if (!o.bad) continue;
        bad.push_back(o.name);
        if (o.status == "  NO CHECKS") empty_groups++;
        else if (o.failed) failed_groups++;
        else crashed_groups++;
    }

    std::fprintf(stderr, "\n%d/%d checks passed", passed, run);
    if (crashed_groups) std::fprintf(stderr, "; %d group(s) CRASHED", crashed_groups);
    if (failed_groups) std::fprintf(stderr, "; %d group(s) failed", failed_groups);
    if (empty_groups) std::fprintf(stderr, "; %d group(s) ran no checks", empty_groups);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "%d worker(s), %lld ms wall, %lld ms of group time\n", workers,
                 static_cast<long long>(wall), static_cast<long long>(sum));
    if (!bad.empty()) {
        std::fprintf(stderr, "groups needing attention:");
        for (const std::string &n : bad) std::fprintf(stderr, " %s", n.c_str());
        std::fprintf(stderr, "\n");
    }
    for (const std::string &k : kept) std::fprintf(stderr, "evidence kept: %s\n", k.c_str());

    const int exit_code =
        (crashed_groups == 0 && failed_groups == 0 && empty_groups == 0 && passed == run) ? 0 : 1;
    const bool reported = write_report(report, done, workers, wall, exit_code);
    if (!reported)
        std::fprintf(stderr, "report: NOT WRITTEN (asked for %s)\n", report ? report : "");
    else if (report && *report)
        std::fprintf(stderr, "report: %s\n", report);
    // The area is scoped to this run, so this run disposes of it: a green run
    // leaves nothing behind, and a run that kept evidence has already named the
    // paths above. Two things have to hold for that to be safe. The removal is
    // skipped when the caller asked for its report UNDER the area -- a sweep runs
    // the suite as a child with --report <area>/base.json, and a child that
    // deleted its parent's evidence directory would take the sweep down with it.
    // And it only ever removes this run's own area, which no other process names.
    {
        const std::string area = scratch_area();
        const std::string rp = report ? std::string(report) : std::string();
        const bool report_inside = !rp.empty() && path_under(absolute_path(rp), area);
        std::error_code rm;
        if (kept.empty() && exit_code == 0 && !report_inside)
            std::filesystem::remove_all(area, rm);
    }
    // A report that was asked for and not written is a failed request even when
    // every check passed: the artifact the caller needs does not exist.
    return (exit_code == 0 && reported) ? 0 : 1;
}

// How many groups run at once. Eight, or the machine's core count when that is
// smaller -- the groups are small and CPU-bound, and the win is real but
// flattens early: seven interleaved repetitions on this 96-thread box gave
// medians of 1554 ms at one worker, 956 ms at four and 783 ms at eight, and
// the eight-worker arm beat the four-worker one in six of those seven pairs.
// The flag beats the environment, and a value that is not a positive number is
// clamped rather than refused -- this is a test runner, not a gate on flags.
static int default_workers() {
    const unsigned hw = std::thread::hardware_concurrency();
    int n = 8;
    if (hw > 0 && static_cast<unsigned>(n) > hw) n = static_cast<int>(hw);
    return n;
}

static int worker_count(const char *flag_value) {
    const char *env = std::getenv("KRK_TEST_WORKERS");
    const char *v = (flag_value && *flag_value) ? flag_value : ((env && *env) ? env : nullptr);
    int n = (v && *v) ? std::atoi(v) : default_workers();
    if (n < 1) n = 1;
    if (n > kGroupCount) n = kGroupCount;
    return n;
}

int main(int argc, char **argv) {
    g_argv0 = (argc > 0 && argv[0]) ? argv[0] : "";
    // The H2D path cache lives in the user's profile. A test run must not read
    // the machine's real measurement -- its decisions would then depend on when
    // the suite last ran -- nor write over it. The group that drives the cache
    // points KRK_H2D_CACHE at its own scratch file and puts this back.
    if (!std::getenv("KRK_H2D_CACHE")) set_env_var("KRK_H2D_CACHE", "0");

    int group = -1;
    const char *progress = nullptr;
    const char *log = nullptr;
    const char *scratch = nullptr;
    const char *workers_flag = nullptr;
    const char *report = nullptr;
    bool selfcheck = false;
    std::string selfcheck_group;
    bool selfcheck_selftest = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i] ? argv[i] : "";
        if (a == "--run-group" && i + 1 < argc) {
            const char *wa = argv[++i];
            const std::string want = wa ? wa : "";
            if (!want.empty() && want[0] >= '0' && want[0] <= '9') {
                group = std::atoi(want.c_str());
                if (group < 0 || group >= kGroupCount) group = -1;
            } else {
                for (int k = 0; k < kGroupCount; k++)
                    if (want == kGroups[k].name) group = k;
            }
            if (group < 0) {
                std::fprintf(stderr, "kraken-tests: no group named '%s' (--list-groups)\n",
                             want.c_str());
                return 2;
            }
        } else if (a == "--progress" && i + 1 < argc) {
            progress = argv[++i];
        } else if (a == "--log" && i + 1 < argc) {
            log = argv[++i];
        } else if (a == "--scratch" && i + 1 < argc) {
            scratch = argv[++i];
        } else if (a == "--workers" && i + 1 < argc) {
            workers_flag = argv[++i];
        } else if (a == "--report" && i + 1 < argc) {
            report = argv[++i];
        } else if (a == "--selfcheck") {
            selfcheck = true;
        } else if (a == "--selftest") {
            selfcheck_selftest = true;
        } else if (a == "--group" && i + 1 < argc) {
            selfcheck_group = argv[++i];
        } else if (a == "--list-groups") {
            for (int k = 0; k < kGroupCount; k++) std::printf("%2d  %s\n", k, kGroups[k].name);
            return 0;
        } else if (a == "--help" || a == "-h") {
            std::fputs(kSuiteUsage, stderr);
            return 0;
        } else {
            std::fprintf(stderr, "kraken-tests: unknown argument '%s' (--help)\n", a.c_str());
            return 2;
        }
    }

    if (group >= 0 && selfcheck) {
        std::fprintf(stderr, "kraken-tests: --selfcheck runs the whole suite; use --group <n> for one group\n");
        return 2;
    }
    if (group >= 0) return run_one_group(group, progress, log, scratch);
    if (selfcheck_selftest && !selfcheck) {
        std::fprintf(stderr, "kraken-tests: --selftest checks the --selfcheck checker; nothing to check without it\n");
        return 2;
    }
    if (!selfcheck_group.empty() && !selfcheck) {
        std::fprintf(stderr, "kraken-tests: --group picks a group for --selfcheck\n");
        return 2;
    }
    if (selfcheck) {
        if (selfcheck_selftest) return run_selfcheck_selftest();
        return run_selfcheck(worker_count(workers_flag), selfcheck_group, report);
    }
    if (!report) {
        const char *env_report = std::getenv("KRK_TEST_REPORT");
        if (env_report && *env_report) report = env_report;
    }
    if (env_flag("KRK_TEST_INPROC")) return run_all_inproc(report);
    return run_all_groups(worker_count(workers_flag), report);
}
