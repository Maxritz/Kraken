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

#include <algorithm>
#include <cmath>
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

// vnpu_matmul_f32 takes its geometry as uint32 and multiplies M*K*N as u64.
bool fits_u32(i64 v) { return v > 0 && v <= 0x7fffffffLL; }

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
    // W8A8 (KRK_NPU_W8A8): GEMMs the int8 path took, and GEMMs it handed back
    // to the device's f32 kernel because a weight or a scratch buffer did not
    // fit. A fallback is not a silent substitution -- it is the more accurate
    // path on the same device -- but it is counted so a run can say so.
    unsigned long long w8a8_gemm = 0;
    unsigned long long w8a8_fallback = 0;
};

struct Entry {
    std::vector<f32> data;
    DType wt = DType::Unknown;
    i64 n_out = 0;
    i64 n_in = 0;
};

// A weight the device has packed into its own int8 layout. The handle belongs
// to this layer and is released with vnpu_packed_weights_destroy at teardown;
// `bytes` is what the pack actually costs (n_out*n_in, a quarter of the f32
// staging entry for the same tensor), so the two budgets are comparable.
struct EntryI8 {
    vnpu_packed_weights_t *packed = nullptr;
    size_t bytes = 0;
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

    // ---- int8 weights (KRK_NPU_W8A8) ------------------------------------
    //
    // Decode is bandwidth-bound on the f32 staging cache (measured: 50.2 MiB
    // staged feeds 290-315 tok/s on a 4-layer fixture, i.e. ~12-15 GiB/s), so
    // the only lever left that is not a better kernel is FEWER BYTES. The
    // device's int8 GEMM multiplies a packed int8 weight by int8 activations
    // and multiplies both scales back in, which quarters the weight traffic and
    // measured 1.54x on the decode GEMM overhead-free.
    //
    // It is opt-in and it is lossy: W8A8 quantizes the activations too, because
    // the device has no "int8 weights x f32 activations" op. Measured worst
    // case on the probe: 3.36e-03 absolute, 3.32e-04 relative to the row peak.
    bool w8a8 = false;
    std::unordered_map<const void *, EntryI8> stage_i8;
    size_t stage_i8_bytes = 0;
    // Activation scratch for the int8 path: one int8 row block plus the
    // per-row scale the quantize op writes. Reused across calls; the stream is
    // ordered, so a later call may reuse the buffer while earlier commands are
    // still queued -- but a RESIZE must flush first, or a queued command would
    // be left pointing at freed storage.
    std::vector<int8_t> a8;
    std::vector<f32> a_scale;

    std::mutex mu;

    Counters n;

    const f32 *weight(const void *w, DType wt, i64 n_out, i64 n_in);
    // The same weight, packed for the device's int8 GEMM. nullptr means the
    // caller must fall back -- the device could not pack it, or the staging
    // budget is spent.
    vnpu_packed_weights_t *weight_i8(const void *w, DType wt, i64 n_out,
                                     i64 n_in);
    bool push(const vnpu_command_t &cmd);
    // Several commands submitted as one batch. The int8 GEMM is a quantize
    // followed by a matmul; submitting them separately would pay two round
    // trips per GEMM on the arm that flushes every call, which is the arm the
    // int8 path exists to speed up.
    bool push_batch(const vnpu_command_t *cmds, size_t count);
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

vnpu_packed_weights_t *Offload::weight_i8(const void *w, DType wt, i64 n_out,
                                          i64 n_in) {
    if (!w || n_out <= 0 || n_in <= 0) return nullptr;
    if (!fits_u32(n_out) || !fits_u32(n_in)) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    auto it = stage_i8.find(w);
    if (it != stage_i8.end()) return it->second.packed;

    const size_t elems = static_cast<size_t>(n_out) * static_cast<size_t>(n_in);
    if (stage_i8_bytes + elems > stage_budget) {
        if (!stage_refused) {
            stage_refused = true;
            KRK_WARN("npu: weight staging budget is %zu MiB (KRK_NPU_WEIGHT_MB); "
                     "further weights stay on the reference path",
                     stage_budget / (1024ull * 1024ull));
        }
        return nullptr;
    }

    std::vector<int8_t> q;
    std::vector<f32> scales;
    std::vector<f32> row;
    try {
        q.resize(elems);
        scales.resize(static_cast<size_t>(n_out));
        row.resize(static_cast<size_t>(n_in));
    } catch (const std::exception &) {
        return nullptr; // bad_alloc / length_error: caller takes its own body
    }

    // One row at a time: dequantize into the scratch row, take its peak, store
    // the per-row scale and the int8 row. Per-ROW, not per-tensor, because a
    // GGUF tensor's rows routinely differ by an order of magnitude and a single
    // scale for the whole matrix would spend most of the int8 range on one
    // outlier row.
    const size_t src_row = (wt == DType::F32) ? static_cast<size_t>(n_in) * 4u
                                              : dtype_row_bytes(wt, n_in);
    const u8 *src = static_cast<const u8 *>(w);
    for (i64 o = 0; o < n_out; o++) {
        const u8 *sp = src + static_cast<size_t>(o) * src_row;
        if (wt == DType::F32) {
            std::memcpy(row.data(), sp, static_cast<size_t>(n_in) * sizeof(f32));
        } else {
            dequant_row(wt, sp, row.data(), n_in);
        }
        f32 amax = 0.0f;
        for (i64 k = 0; k < n_in; k++) amax = std::max(amax, std::fabs(row[k]));
        const f32 s = amax > 0.0f ? amax / 127.0f : 0.0f;
        scales[static_cast<size_t>(o)] = s;
        const f32 inv = s > 0.0f ? 1.0f / s : 0.0f;
        int8_t *dst = q.data() + static_cast<size_t>(o) * static_cast<size_t>(n_in);
        // Round half away from zero in FLOAT, not std::lround on a double.
        // Same rounding rule (lround is also half-away-from-zero), same result
        // for every value in range, and measured 8.5x faster on a real tensor
        // (20.3 ms to 2.4 ms for 1376x512). Done in doubles this pass was the
        // single largest part of the int8 path's one-time warm-up: the first
        // prefill chunk paid ~95 ms of it and read as a 40% prefill
        // regression, which is not a property of int8 at all.
        for (i64 k = 0; k < n_in; k++) {
            const f32 f = row[k] * inv;
            int v = static_cast<int>(f >= 0.0f ? f + 0.5f : f - 0.5f);
            if (v > 127) v = 127;
            if (v < -127) v = -127; // symmetric: -127..127, so no -128 corner
            dst[k] = static_cast<int8_t>(v);
        }
    }

    // Packing is a synchronous device call, and the queue may hold commands
    // referencing buffers it is about to be handed. Drain first so the pack
    // cannot interleave with work already issued. This is once per weight, on
    // the same warm-up pass that stages the f32 copies.
    flush(this);

    vnpu_packed_weights_t *packed = nullptr;
    const vnpu_status_t rc =
        vnpu_pack_weights_i8(dev, q.data(), scales.data(),
                             static_cast<uint32_t>(n_in),
                             static_cast<uint32_t>(n_out),
                             /*ldb=*/0, /*trans_b=*/1, &packed);
    if (rc != VNPU_OK || !packed) return nullptr;

    stage_i8_bytes += elems;
    EntryI8 e;
    e.packed = packed;
    e.bytes = elems;
    stage_i8.emplace(w, e);
    return packed;
}

bool Offload::push(const vnpu_command_t &cmd) { return push_batch(&cmd, 1); }

bool Offload::push_batch(const vnpu_command_t *cmds, size_t count) {
    queue.insert(queue.end(), cmds, cmds + count);
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
    o->w8a8 = env_on("KRK_NPU_W8A8");
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

// Does the trace we just asked the device to write carry a usable time base?
//
// The device's own clock is fine; its JSON writer was not. Absolute monotonic
// microseconds near 1.2e9 printed through an ostream at the default six
// significant digits land on a 10 ms grid, so every command inside one cell
// shares a start instant while its duration -- a small number, same six digits
// -- stays exact. The trace then looks plausible and cannot answer a single
// gap or duty-cycle question.
//
// The fix lives in the device's writer (patches/npuxl-0001-trace-timestamp-
// precision.patch, which rebases timestamps to the first event), and this build
// links whichever library the user built. So check the artifact instead of
// trusting it: a rebased trace starts at ts 0, and anything else arrives with
// an absolute stamp. Say so at capture time, because the alternative is a user
// drawing conclusions from a timeline quantised by 10 ms and finding out weeks
// later -- which is what happened here.
static void warn_if_trace_timebase_is_absolute(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    char buf[4096];
    const size_t got = std::fread(buf, 1, sizeof(buf), f);
    std::fclose(f);
    if (got == 0) return;
    const char *end = buf + got;
    const char *hit = nullptr;
    for (const char *p = buf; p + 5 < end; p++)
        if (p[0] == '"' && p[1] == 't' && p[2] == 's' && p[3] == '"' &&
            p[4] == ':') {
            hit = p + 5;
            break;
        }
    if (!hit) return; // trace is on but nothing was recorded
    const double ts = std::strtod(hit, nullptr);
    // A rebased trace's first event is at 0 (or a few ns of rounding). An
    // absolute monotonic stamp is >= 1e6 us -- a second of uptime -- and this
    // box had been up for ~1000 s, so the two cases are separated by orders
    // of magnitude rather than by a threshold anyone has to tune.
    if (ts < 1e6) return;
    KRK_WARN("npu: trace '%s' starts at ts=%.0f us, i.e. an ABSOLUTE stamp, so "
             "its start times are quantised to ~10 ms (only durations are "
             "exact). Apply patches/npuxl-0001-trace-timestamp-precision.patch "
             "to the NPU engine and relink; tools/trace_stats.py detects this "
             "and withholds the derived numbers.", path.c_str(), ts);
}

void destroy(Offload *o) {
    if (!o) return;
    flush(o);
    // Packed int8 weights belong to the device, so they go before it does.
    for (auto &kv : o->stage_i8)
        if (kv.second.packed) vnpu_packed_weights_destroy(kv.second.packed);
    o->stage_i8.clear();
    if (o->stream) vnpu_stream_destroy(o->stream);
    if (o->dev) vnpu_device_destroy(o->dev);
    if (!o->trace_path.empty()) warn_if_trace_timebase_is_absolute(o->trace_path);
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
    std::string out = format(
        "gemm %llu, rmsnorm %llu, swiglu %llu, add %llu, mul %llu, "
        "%llu declined, %llu batch(es); %.1f GFLOP, %.1f MiB staged",
        o->n.gemm, o->n.rmsnorm, o->n.swiglu, o->n.add, o->n.mul,
        o->n.declined, o->n.batches,
        static_cast<double>(o->n.flops) / 1.0e9,
        static_cast<double>(o->stage_bytes) / (1024.0 * 1024.0));
    // Only when the switch is on: a default run should not have to read a
    // zero-row about a path it never took.
    if (o->w8a8) {
        out += format("; w8a8 %llu int8 (%.1f MiB packed), %llu on f32",
                      o->n.w8a8_gemm,
                      static_cast<double>(o->stage_i8_bytes) / (1024.0 * 1024.0),
                      o->n.w8a8_fallback);
    }
    return out;
}

namespace {

// The int8 GEMM, as two device commands: quantize the activation rows, then
// multiply by the packed weight.
//
//   C[m,n] = (sum_k A8[m,k]*B8[n,k]) * scale_a[m] * scale_b[n]
//
// which is the device kernel's own composition (store_i8_c multiplies both
// scales back in), so no scale arithmetic is left to the host. Both commands
// go out as ONE batch: submitting the quantize on its own would add a round
// trip to the arm that flushes every call, which is precisely the arm this
// path is meant to speed up.
bool gemm_i8(Offload *o, void *out, const void *x, const void *w, DType wt,
             i64 n_out, i64 n_in, i64 rows) {
    vnpu_packed_weights_t *pk = o->weight_i8(w, wt, n_out, n_in);
    if (!pk) return false;

    const size_t need = static_cast<size_t>(rows) * static_cast<size_t>(n_in);
    if (need > o->a8.size() || static_cast<size_t>(rows) > o->a_scale.size()) {
        // A queued command may still hold these pointers, so drain before the
        // reallocation. Growth is monotonic and stops after the first few
        // tokens, so this costs a handful of flushes over a run, not one per
        // call.
        if (!flush(o)) return false;
        try {
            o->a8.resize(std::max(need, static_cast<size_t>(1)));
            o->a_scale.resize(std::max(static_cast<size_t>(rows),
                                       static_cast<size_t>(1)));
        } catch (const std::exception &) {
            return false;
        }
    }

    vnpu_command_t c[2]{};
    c[0].opcode = VNPU_OP_QUANTIZE_I8;
    vnpu_quantize_i8_args_t &q = c[0].args.quantize_i8;
    q.X = static_cast<const f32 *>(x);
    q.Y = o->a8.data();
    q.scales = o->a_scale.data();
    q.rows = static_cast<uint64_t>(rows);
    q.cols = static_cast<uint64_t>(n_in);
    // Per ROW, because the matmul takes one scale per activation row (scale_a
    // is length M). Per-tensor would be simpler and strictly worse: every
    // token's activation row has its own peak, and a shared scale would spend
    // the int8 range on whichever row happens to be largest.
    q.axis = VNPU_QUANT_PER_ROW;
    q.unit = o->unit;

    c[1].opcode = VNPU_OP_MATMUL_I8;
    vnpu_matmul_i8_args_t &m = c[1].args.matmul_i8;
    m.A = o->a8.data();
    m.B = pk;
    m.scale_a = o->a_scale.data();
    m.bias = nullptr;
    m.C = static_cast<f32 *>(out);
    m.M = static_cast<uint32_t>(rows);
    m.K = static_cast<uint32_t>(n_in);
    m.N = static_cast<uint32_t>(n_out);
    m.lda = 0; // 0 = K, the device's default for a dense activation block
    m.ldc = 0; // 0 = N
    m.epilogue = VNPU_EPILOGUE_NONE;
    m.unit = o->unit;

    if (!o->push_batch(c, 2)) return false;
    o->n.w8a8_gemm++;
    o->n.gemm++;
    o->n.flops += 2ull * static_cast<unsigned long long>(rows) *
                  static_cast<unsigned long long>(n_out) *
                  static_cast<unsigned long long>(n_in);
    return true;
}

} // namespace

bool gemm(Offload *o, void *out, const void *x, const void *w, DType wt,
          i64 n_out, i64 n_in, i64 rows) {
    if (!o || !o->stream || !out || !x || !w) return false;
    if (!fits_u32(rows) || !fits_u32(n_out) || !fits_u32(n_in)) return false;
    // Opt-in W8A8. A decline here is not a substitution: it falls through to
    // the device's own f32 GEMM, which is the MORE accurate path on the same
    // device, and the run summary counts it.
    if (o->w8a8 && gemm_i8(o, out, x, w, wt, n_out, n_in, rows)) return true;
    if (o->w8a8) o->n.w8a8_fallback++;
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
