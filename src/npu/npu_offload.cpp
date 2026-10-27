// ============================================================================
//  npu_offload.cpp — VNPU device lifecycle and the GEMM hand-over.
//
//  WHAT IS OFFLOADED, AND WHY ONLY THAT
//
//  The GEMM. w[n_out, n_in] stays in its GGUF block format and is dequantized
//  once into a staging buffer the first time the tensor is multiplied, after
//  which `vnpu_matmul_f32` sees exactly the layout it wants for a decode step:
//  A [rows, n_in] row-major, B [n_out, n_in] with trans_b = 1, C [rows, n_out].
//
//  The activation-type ops stay on the reference implementation on purpose.
//  Measured against a double-precision reference on this host: VNPU's rmsnorm
//  agrees to 2.2e-07 and the GEMM to 2.7e-07 relative, but swiglu (a polynomial
//  sigmoid) misses by 1.8e-05 -- two orders of magnitude worse than an f32
//  rounding difference, and an FFN applies that error once per layer per token.
//  Handing it over would make the NPU arm a slightly different model rather
//  than a different execution schedule, so it does not happen. A caller who
//  wants VNPU's activations can add them here and measure; the seam exists.
//
//  DETERMINISM
//
//  `vnpu_matmul_f32` tiles M and N and runs the tiles on its thread pool, but
//  every output element's K reduction is sequential inside its own tile, so a
//  row's value does not depend on how many rows were in the call. Measured: the
//  same row computed with M=1 and with M=3 is bit-identical, which is what keeps
//  a chunked prefill and a one-token decode in agreement on this backend the
//  way the reference backend's single-trampoline pool keeps them.
// ============================================================================
#include "npu_offload.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace krk {
namespace npu {

#if defined(KRK_ENABLE_NPU)

#include <vnpu.h>

namespace {

const char *env_str(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

bool env_on(const char *name) {
    const char *v = env_str(name);
    return v && std::strcmp(v, "0") != 0;
}

i64 env_i64(const char *name, i64 dflt) {
    const char *v = env_str(name);
    if (!v) return dflt;
    char *end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || n < 0) return dflt;
    return static_cast<i64>(n);
}

struct Entry {
    std::vector<f32> data;
    DType wt = DType::Unknown;
    i64 n_out = 0;
    i64 n_in = 0;
};

} // namespace

class Offload {
public:
    vnpu_device_t *dev = nullptr;
    vnpu_stream_t *stream = nullptr;
    vnpu_device_info_t info{};
    std::string trace_path;

    // Dequantized weight staging. Keyed by the weight pointer the engine holds,
    // which is stable for the life of the model: the loader allocates once.
    std::unordered_map<const void *, Entry> stage;
    size_t stage_bytes = 0;
    size_t stage_budget = 1024ull * 1024ull * 1024ull;
    // Which compute unit the scheduler is asked for. AUTO lets VNPU pick per
    // call, which means a one-row call runs the CELL unit and a prefill-sized
    // call the VOODOO one -- two kernels with different reduction orders, so
    // the same row would come out differently depending on how many rows rode
    // along with it and the chunk-invariance the engine is built around would
    // stop holding the moment a prompt had more than 4 tokens. The default is
    // therefore one unit for every call; KRK_NPU_UNIT=auto restores VNPU's own
    // choice for a run that would rather have the decode latency.
    uint32_t unit = 1; // VNPU_UNIT_VOODOO
    bool stage_refused = false; // budget hit at least once -- say so, once
    mutable std::mutex mu;

    // Counters. Plain u64: the engine is single-threaded around a backend call
    // and the mutex above covers the staging map for anything that is not.
    unsigned long long calls = 0;
    unsigned long long on_device = 0;
    unsigned long long fell_back = 0;
    unsigned long long flops = 0;

    // The fallback for a weight that is not f32 and has no staging room.
    const f32 *weight(const void *w, DType wt, i64 n_out, i64 n_in);
    std::string describe() const;
};

const f32 *Offload::weight(const void *w, DType wt, i64 n_out, i64 n_in) {
    if (wt == DType::F32) return static_cast<const f32 *>(w);
    std::lock_guard<std::mutex> lk(mu);
    auto it = stage.find(w);
    if (it != stage.end()) return it->second.data.data();
    const size_t need = static_cast<size_t>(n_out) * static_cast<size_t>(n_in) * 4;
    if (stage_bytes + need > stage_budget) {
        if (!stage_refused) {
            stage_refused = true;
            KRK_WARN("npu: weight staging budget is %zu MiB (KRK_NPU_WEIGHT_MB); "
                     "further non-f32 weights stay on the reference path",
                     stage_budget / (1024ull * 1024ull));
        }
        return nullptr;
    }
    Entry e;
    e.wt = wt;
    e.n_out = n_out;
    e.n_in = n_in;
    try {
        e.data.resize(static_cast<size_t>(n_out) * static_cast<size_t>(n_in));
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
    const size_t row_bytes = dtype_row_bytes(wt, n_in);
    const u8 *src = static_cast<const u8 *>(w);
    f32 *dst = e.data.data();
    for (i64 o = 0; o < n_out; o++)
        dequant_row(wt, src + static_cast<size_t>(o) * row_bytes,
                    dst + static_cast<size_t>(o) * static_cast<size_t>(n_in), n_in);
    const f32 *out = e.data.data();
    stage_bytes += need;
    stage.emplace(w, std::move(e));
    return out;
}

std::string Offload::describe() const {
    const uint32_t simd = info.simd_bits;
    return format("NPU (Virtual NPU %s: %s, %u-bit SIMD, %u thread(s), "
                  "%ux%ux%u tiles)",
                  info.name, info.backend, simd, info.num_threads, info.tile_m,
                  info.tile_k, info.tile_n);
}

bool available() { return true; }

Offload *create(int device_index, std::string *why) {
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    uint32_t unit = VNPU_UNIT_VOODOO;
    if (const char *u = env_str("KRK_NPU_UNIT")) {
        if (!std::strcmp(u, "auto")) unit = VNPU_UNIT_AUTO;
        else if (!std::strcmp(u, "cell")) unit = VNPU_UNIT_CELL;
        else if (!std::strcmp(u, "wildcat")) unit = VNPU_UNIT_WILDCAT;
        else if (!std::strcmp(u, "voodoo")) unit = VNPU_UNIT_VOODOO;
        else
            KRK_WARN("npu: unknown KRK_NPU_UNIT '%s' (auto|voodoo|cell|wildcat)",
                     u);
    }
    // 0 = the device's own default (hardware concurrency). Setting it from the
    // environment is how a run is pinned to fewer workers, mirroring
    // KRK_THREADS on the reference side.
    cfg.num_threads = static_cast<uint32_t>(env_i64("KRK_NPU_THREADS", 0));
    // HIGH_THROUGHPUT lets the scheduler pick the wide tiling for multi-row
    // (prefill) calls; a one-row call is routed to the CELL unit anyway, so
    // decode keeps the low-latency path.
    cfg.flags = VNPU_CONFIG_HIGH_THROUGHPUT;
    std::string trace;
    if (const char *tp = env_str("KRK_NPU_TRACE")) {
        trace = tp;
        cfg.flags |= VNPU_CONFIG_TRACE | VNPU_CONFIG_TRACE_CHROME;
        cfg.trace_file = tp;
    }
    if (env_on("KRK_NPU_PIN_THREADS")) cfg.flags |= VNPU_CONFIG_PIN_THREADS;

    Offload *o = new Offload();
    o->trace_path = trace;
    o->unit = unit;
    vnpu_status_t rc = vnpu_device_create(&cfg, &o->dev);
    if (rc != VNPU_OK) {
        if (why)
            *why = format("virtual NPU %d could not be opened: %s", device_index,
                          vnpu_status_string(rc));
        delete o;
        return nullptr;
    }
    rc = vnpu_stream_create(o->dev, 0, &o->stream);
    if (rc != VNPU_OK) {
        if (why)
            *why = format("virtual NPU %d has no usable stream: %s",
                          device_index, vnpu_status_string(rc));
        vnpu_device_destroy(o->dev);
        delete o;
        return nullptr;
    }
    vnpu_device_get_info(o->dev, &o->info);
    o->stage_budget =
        static_cast<size_t>(env_i64("KRK_NPU_WEIGHT_MB", 1024)) * 1024ull * 1024ull;
    return o;
}

void destroy(Offload *o) {
    if (!o) return;
    if (o->stream) vnpu_stream_destroy(o->stream);
    if (o->dev) vnpu_device_destroy(o->dev);
    delete o;
}

std::string describe(const Offload *o) {
    return o ? o->describe() : std::string("NPU (not opened)");
}

std::string stats(const Offload *o) {
    if (!o) return std::string();
    return format("gemm %llu call(s), %llu on the device, %llu fell back, "
                  "%.1f GFLOP handed over, %.1f MiB of weights staged",
                  o->calls, o->on_device, o->fell_back,
                  static_cast<double>(o->flops) / 1.0e9,
                  static_cast<double>(o->stage_bytes) / (1024.0 * 1024.0));
}

bool gemm(Offload *o, f32 *out, const f32 *x, const void *w, DType wt,
          i64 n_out, i64 n_in, i64 rows) {
    if (!o || !o->stream || !out || !x || !w) return false;
    if (rows <= 0 || n_out <= 0 || n_in <= 0) return false;
    // vnpu_matmul_f32 takes its geometry as uint32; a tensor that does not fit
    // is not something this device can express.
    constexpr i64 kMax = 0x7fffffffLL;
    if (rows > kMax || n_out > kMax || n_in > kMax) return false;
    if (rows > kMax / n_out || rows * n_out > kMax / n_in) return false;

    o->calls++;
    const f32 *b = o->weight(w, wt, n_out, n_in);
    if (!b) {
        o->fell_back++;
        return false;
    }
    vnpu_matmul_f32_args_t a{};
    a.A = x;
    a.B = b;
    a.bias = nullptr;
    a.C = out;
    a.M = static_cast<uint32_t>(rows);
    a.K = static_cast<uint32_t>(n_in);
    a.N = static_cast<uint32_t>(n_out);
    // Kraken's weights are [n_out][n_in] -- one contiguous row per output.
    // That is VNPU's trans_b = 1 with ldb = K, so no transpose and no copy.
    a.trans_a = 0;
    a.trans_b = 1;
    a.lda = 0;
    a.ldb = 0;
    a.ldc = 0;
    a.alpha = 1.0f;
    a.beta = 0.0f; // do not read C: the caller's GEMM overwrites it
    a.epilogue = VNPU_EPILOGUE_NONE;
    a.unit = o->unit;
    if (vnpu_matmul_f32(o->stream, &a) != VNPU_OK) {
        // Argument validation happens before any tile runs, so a refusal here
        // has not written out[] and the caller's own body is still correct.
        o->fell_back++;
        return false;
    }
    o->on_device++;
    o->flops += 2ull * static_cast<unsigned long long>(rows) *
                static_cast<unsigned long long>(n_out) *
                static_cast<unsigned long long>(n_in);
    return true;
}

#else // !KRK_ENABLE_NPU ------------------------------------------------------

bool available() { return false; }

Offload *create(int, std::string *why) {
    if (why)
        *why = "this build has no NPU support (configure with "
               "-DKRK_ENABLE_NPU=ON and -DKRK_VNPU_ROOT=<npuxl-improved>)";
    return nullptr;
}

void destroy(Offload *) {}

std::string describe(const Offload *) {
    return std::string("NPU (this build has no NPU support)");
}

std::string stats(const Offload *) { return std::string(); }

bool gemm(Offload *, f32 *, const f32 *, const void *, DType, i64, i64, i64) {
    return false;
}

#endif

} // namespace npu
} // namespace krk
