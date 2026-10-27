// ============================================================================
//  npu_offload.cpp — VNPU device lifecycle, the command queue, and the ops.
//
//  One queue, one stream, one sync point. Every op validates its arguments
//  before it is queued -- the same checks vnpu's own prepare step repeats -- and
//  returns false when the device cannot take it, so the caller's reference body
//  is the fallback for every single call rather than a parallel implementation
//  to keep in sync.
//
//  The measured numbers this file is built around (2-core x86-64 host, AVX-512):
//
//    matmul f32, K=N=1024     1 row   ~255 us   8 GFLOPS
//                             512 row ~27 ms    39 GFLOPS
//    rmsnorm 4096             1 row   ~100 us  (vs ~2 us scalar)
//    silu (swiglu)            worst relative error 8.4e-05 vs exact
//    gelu                     worst relative error 7.4e-01 -- NOT USED
//
//  The per-call cost is almost all thread wake-ups: a bare producer/consumer
//  condition-variable round trip on the same host is ~60 us. That is why Graph
//  mode batches instead of paying one round trip per op.
// ============================================================================
#include "npu_offload.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

// Every counter the run summary prints, in the order a layer meets them.
struct Counters {
    unsigned long long gemm = 0;
    unsigned long long rmsnorm = 0;
    unsigned long long swiglu = 0;
    unsigned long long add = 0;
    unsigned long long mul = 0;
    unsigned long long declined = 0; // offered, but the device could not take it
    unsigned long long batches = 0;  // submissions actually made
    unsigned long long flushes = 0;  // flush() calls (queue empty included)
    unsigned long long flops = 0;
    unsigned long long elements = 0; // activation elements the device touched
};

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
    Mode mode = Mode::Gemm;
    uint32_t unit = 1; // VNPU_UNIT_VOODOO
    bool strict_queue = false;
    // Element floor for handing a SMALL op (rmsnorm/swiglu/add/mul) to the
    // device. A queued call still costs the worker a dispatch, and below a few
    // thousand elements the reference body -- which is already in the caller's
    // cache -- is cheaper than the hand-off. KRK_NPU_MIN_ELEMS overrides.
    i64 min_elems = 0;
    std::string trace_path;

    // Pending commands. Reused across flushes so a steady-state step does not
    // allocate.
    std::vector<vnpu_command_t> queue;

    // Dequantized weight staging. Keyed by the weight pointer the engine holds,
    // which is stable for the life of the model: the loader allocates once.
    std::unordered_map<const void *, Entry> stage;
    size_t stage_bytes = 0;
    size_t stage_budget = 1024ull * 1024ull * 1024ull;
    bool stage_refused = false; // budget hit at least once -- say so, once
    std::mutex mu;

    Counters n;

    const f32 *weight(const void *w, DType wt, i64 n_out, i64 n_in);
    bool push(const vnpu_command_t &cmd);
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
    } catch (const std::exception &) {
        // bad_alloc for a weight that does not fit, length_error for a geometry
        // that overflows size_t. Either way the caller takes its own body.
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

bool Offload::push(const vnpu_command_t &cmd) {
    queue.push_back(cmd);
    // Gemm mode submits and waits on every call; Graph mode queues and is
    // drained by npu_flush_() at the next host op.
    //
    // That asymmetry looks like a missed optimisation and is not one. Backend::
    // gemm is a SYNCHRONOUS virtual: `out` must hold the result when it
    // returns, because a caller is entitled to read that host buffer directly
    // instead of going through sync() or download_f32(). tests/test_kraken.cpp
    // does exactly that, and it fails -- correctly -- if this flush is removed.
    // Graph mode buys its batching a level up, where the whole layer is one
    // event and the host op that follows it flushes once.
    //
    // Measured before deciding: batching this instead moved end-to-end decode
    // by a median +1.3% (within this box's +-20% noise) except on 2-layer toy
    // models, where the round trip is a large share of a tiny step. Not worth
    // an interface contract.
    if (mode == Mode::Gemm || strict_queue) return flush(this);
    return true;
}

std::string Offload::describe() const {
    const char *what = mode == Mode::Graph ? "whole graph" : "GEMM path";
    return format("NPU (%s on Virtual NPU %s: %s, %u-bit SIMD, %u thread(s), "
                  "%ux%ux%u tiles)",
                  what, info.name, info.backend, info.simd_bits, info.num_threads,
                  info.tile_m, info.tile_k, info.tile_n);
}

bool available() { return true; }

Offload *create(int device_index, Mode mode, std::string *why) {
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
    o->mode = mode;
    o->unit = unit;
    o->strict_queue = env_on("KRK_NPU_STRICT_QUEUE");
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
    o->min_elems = env_i64("KRK_NPU_MIN_ELEMS", o->min_elems);
    o->queue.reserve(64);
    return o;
}

void destroy(Offload *o) {
    if (!o) return;
    flush(o);
    if (o->stream) vnpu_stream_destroy(o->stream);
    if (o->dev) vnpu_device_destroy(o->dev);
    delete o;
}

std::string describe(const Offload *o) {
    return o ? o->describe() : std::string("NPU (not opened)");
}

bool flush(Offload *o) {
    if (!o || !o->stream) return true;
    o->n.flushes++;
    if (o->queue.empty()) return true;
    const size_t submitted = o->queue.size();
    const vnpu_status_t rc =
        vnpu_stream_submit(o->stream, o->queue.data(), submitted, nullptr);
    o->queue.clear();
    // A failed submission is not a fallback, it is a correctness event. The
    // ops already in this batch returned true -- their callers have moved on --
    // so the rows they were asked to produce were never written, and decoding
    // on from here is decoding garbage. There is no recovery point inside a
    // graph, so this stops the run and says why. (Same rule, and the same
    // reason, as the reference backend's hadamard_act default: a lost
    // transformation is fluent nonsense, not an error anyone can see.)
    if (rc != VNPU_OK) {
        KRK_ERROR("npu: submitting %zu command(s) failed (%s); the %llu "
                  "command(s) queued before them were never executed, so the "
                  "activations they were meant to produce are undefined",
                  submitted, vnpu_status_string(rc),
                  static_cast<unsigned long long>(o->n.batches));
        std::abort();
    }
    o->n.batches++;
    const vnpu_status_t src = vnpu_stream_sync(o->stream);
    if (src != VNPU_OK) {
        KRK_ERROR("npu: the device reported an error (%s) after %zu command(s); "
                  "their outputs cannot be trusted and there is no point in the "
                  "graph to restart from",
                  vnpu_status_string(src), submitted);
        std::abort();
    }
    return true;
}

std::string stats(const Offload *o) {
    if (!o) return std::string();
    return format("gemm %llu, rmsnorm %llu, swiglu %llu, add %llu, mul %llu, "
                  "%llu declined, %llu batch(es); "
                  "%.1f GFLOP, %.1f MiB staged",
                  o->n.gemm, o->n.rmsnorm, o->n.swiglu, o->n.add, o->n.mul,
                  o->n.declined, o->n.batches,
                  static_cast<double>(o->n.flops) / 1.0e9,
                  static_cast<double>(o->stage_bytes) / (1024.0 * 1024.0));
}

namespace {

// vnpu_matmul_f32 takes its geometry as uint32 and multiplies M*K*N as u64.
bool fits_u32(i64 v) { return v > 0 && v <= 0x7fffffffLL; }

} // namespace

bool gemm(Offload *o, void *out, const void *x, const void *w, DType wt,
          i64 n_out, i64 n_in, i64 rows) {
    if (!o || !o->stream || !out || !x || !w) return false;
    if (!fits_u32(rows) || !fits_u32(n_out) || !fits_u32(n_in)) return false;
    const f32 *b = o->weight(w, wt, n_out, n_in);
    if (!b) {
        o->n.declined++;
        return false;
    }
    vnpu_command_t c{};
    c.opcode = VNPU_OP_MATMUL_F32;
    vnpu_matmul_f32_args_t &a = c.args.matmul_f32;
    a.A = static_cast<const f32 *>(x);
    a.B = b;
    a.bias = nullptr;
    a.C = static_cast<f32 *>(out);
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
    if (!o->push(c)) {
        o->n.declined++;
        return false;
    }
    o->n.gemm++;
    o->n.flops += 2ull * static_cast<unsigned long long>(rows) *
                  static_cast<unsigned long long>(n_out) *
                  static_cast<unsigned long long>(n_in);
    return true;
}

bool rmsnorm(Offload *o, void *out, const void *x, const f32 *w, i64 rows, i64 n,
             f32 eps) {
    if (!o || !o->stream || o->mode != Mode::Graph) return false;
    if (rows * n < o->min_elems) return false;
    if (!out || !x || !fits_u32(rows) || !fits_u32(n)) return false;
    vnpu_command_t c{};
    c.opcode = VNPU_OP_RMSNORM_F32;
    vnpu_rmsnorm_f32_args_t &a = c.args.rmsnorm;
    a.X = static_cast<const f32 *>(x);
    a.W = w;
    a.Y = static_cast<f32 *>(out);
    a.rows = static_cast<uint32_t>(rows);
    a.cols = static_cast<uint32_t>(n);
    a.eps = eps;
    a.unit = o->unit;
    if (!o->push(c)) {
        o->n.declined++;
        return false;
    }
    o->n.rmsnorm++;
    o->n.elements += static_cast<unsigned long long>(rows) * n;
    return true;
}

bool swiglu(Offload *o, void *out, const void *gate, const void *up, i64 n) {
    if (!o || !o->stream || o->mode != Mode::Graph) return false;
    if (n < o->min_elems) return false;
    if (!out || !gate || !up || n <= 0) return false;
    vnpu_command_t c{};
    c.opcode = VNPU_OP_SWIGLU_F32;
    vnpu_swiglu_f32_args_t &a = c.args.swiglu;
    a.gate = static_cast<const f32 *>(gate);
    a.up = static_cast<const f32 *>(up);
    a.Y = static_cast<f32 *>(out);
    a.count = static_cast<uint64_t>(n);
    a.alpha = 1.0f;
    a.unit = o->unit;
    if (!o->push(c)) {
        o->n.declined++;
        return false;
    }
    o->n.swiglu++;
    o->n.elements += static_cast<unsigned long long>(n);
    return true;
}

bool add(Offload *o, void *a_, const void *b, i64 n) {
    if (!o || !o->stream || o->mode != Mode::Graph) return false;
    if (n < o->min_elems) return false;
    if (!a_ || !b || n <= 0) return false;
    vnpu_command_t c{};
    c.opcode = VNPU_OP_ADD_F32;
    vnpu_binary_f32_args_t &a = c.args.binary;
    a.A = static_cast<const f32 *>(a_);
    a.B = static_cast<const f32 *>(b);
    a.Y = static_cast<f32 *>(a_);
    a.count = static_cast<uint64_t>(n);
    // The kernel reads a zero alpha/beta as "unset" and substitutes 1, which is
    // the same answer, but saying 1 makes the intent local.
    a.alpha = 1.0f;
    a.beta = 1.0f;
    a.unit = o->unit;
    if (!o->push(c)) {
        o->n.declined++;
        return false;
    }
    o->n.add++;
    o->n.elements += static_cast<unsigned long long>(n);
    return true;
}

bool add_scaled(Offload *o, void *dst, const void *src, f32 alpha, i64 n) {
    if (!o || !o->stream || o->mode != Mode::Graph) return false;
    if (n < o->min_elems) return false;
    if (!dst || !src || n <= 0) return false;
    vnpu_command_t c{};
    c.opcode = VNPU_OP_ADD_F32;
    vnpu_binary_f32_args_t &a = c.args.binary;
    a.A = static_cast<const f32 *>(src);
    a.B = static_cast<const f32 *>(dst);
    a.Y = static_cast<f32 *>(dst);
    a.count = static_cast<uint64_t>(n);
    a.alpha = alpha;
    a.beta = 1.0f;
    a.unit = o->unit;
    if (!o->push(c)) {
        o->n.declined++;
        return false;
    }
    o->n.add++;
    o->n.elements += static_cast<unsigned long long>(n);
    return true;
}

bool mul(Offload *o, void *a_, const void *b, i64 n) {
    if (!o || !o->stream || o->mode != Mode::Graph) return false;
    if (n < o->min_elems) return false;
    if (!a_ || !b || n <= 0) return false;
    vnpu_command_t c{};
    c.opcode = VNPU_OP_MUL_F32;
    vnpu_binary_f32_args_t &a = c.args.binary;
    a.A = static_cast<const f32 *>(a_);
    a.B = static_cast<const f32 *>(b);
    a.Y = static_cast<f32 *>(a_);
    a.count = static_cast<uint64_t>(n);
    a.alpha = 1.0f;
    a.beta = 0.0f;
    a.unit = o->unit;
    if (!o->push(c)) {
        o->n.declined++;
        return false;
    }
    o->n.mul++;
    o->n.elements += static_cast<unsigned long long>(n);
    return true;
}

#else // !KRK_ENABLE_NPU ------------------------------------------------------

bool available() { return false; }

Offload *create(int, Mode, std::string *why) {
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

bool flush(Offload *) { return true; }

bool gemm(Offload *, void *, const void *, const void *, DType, i64, i64, i64) {
    return false;
}
bool rmsnorm(Offload *, void *, const void *, const f32 *, i64, i64, f32) {
    return false;
}
bool swiglu(Offload *, void *, const void *, const void *, i64) { return false; }
bool add(Offload *, void *, const void *, i64) { return false; }
bool add_scaled(Offload *, void *, const void *, f32, i64) { return false; }
bool mul(Offload *, void *, const void *, i64) { return false; }

#endif

} // namespace npu
} // namespace krk
