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

#include "krk/common.hpp"
#include "krk/kernel_time.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    // Where a flushed batch's wall time actually goes. submit() hands the batch
    // to the device's thread pool; sync() waits for it. Those two numbers plus
    // the device's own execution time are what separate "the device is slow"
    // from "the host is stalled around the device", which is the question a GPU
    // trace answers with queue-wait and launch-overhead columns. KRK_NPU_OPTIME
    // gates the clocks; they cost two calls per batch.
    double submit_ms = 0.0;
    double sync_ms = 0.0;
    // One-time host work: quantizing a weight row into the device's packed int8
    // layout. The pack is memoized by weight address, so this is paid once per
    // weight for the whole run -- but it is paid DURING prefill, the first time
    // an expert slot is touched, which is exactly where it can hide.
    double pack_ms = 0.0;      // whole weight_i8() miss path
    double stage_ms = 0.0;     // whole weight() miss path: q8_0 -> f32 expansion
    double pack_host_ms = 0.0; // the host convert loop alone
    double pack_flush_ms = 0.0;// the drain before the device pack
    double pack_dev_ms = 0.0;  // vnpu_pack_weights_i8 itself
    unsigned long long packed_weights = 0;
    // Weights whose first int8 request was served by the f32 path on purpose,
    // because they had not yet been asked for twice.
    unsigned long long w8a8_deferred = 0;
    // GEMM row-count histogram: 1 row, 2-8, 9-32, 33+. Distinguishes decode
    // steps from prefill chunks in the numbers rather than by assumption.
    unsigned long long rows_hist[4] = {0, 0, 0, 0};
    // GEMMs the device refused because the host is the faster arm for that
    // shape. Not a fallback: a routing decision, made per call.
    unsigned long long shape_refused = 0;
    unsigned long long staged_weights = 0;
    unsigned long long streamed_weights = 0; // of those, served by the ring
    unsigned long long q8_0_gemm = 0; // GEMMs that read weights in place
    unsigned long long flops = 0;
    unsigned long long elements = 0; // activation elements the device touched
    // W8A8 (KRK_NPU_W8A8): GEMMs the int8 path took, and GEMMs it handed back
    // to the device's f32 kernel because a weight or a scratch buffer did not
    // fit. A fallback is not a silent substitution -- it is the more accurate
    // path on the same device -- but it is counted so a run can say so.
    unsigned long long w8a8_gemm = 0;
    unsigned long long w8a8_fallback = 0;
    // Attention: calls the device ran, and calls that stayed on the reference.
    // The host ones are broken out by the ONE feature that blocked them, so a
    // model that never reaches the device says which feature did it rather
    // than leaving the reader to guess.
    unsigned long long attn = 0;
    unsigned long long attn_host[7] = {0, 0, 0, 0, 0, 0, 0};
};

// Why an attention call stayed on the reference. Index into Counters::attn_host.
enum AttnWhy {
    ATTN_INT8 = 0,   // int8 KV cache: the device takes f32 rows
    ATTN_WINDOW,     // sliding window
    ATTN_SKIP,       // masked block (skip_from)
    ATTN_SOFTCAP,    // Gemma-2 style logit softcap
    ATTN_SINK,       // attention sink
    ATTN_SMALL,      // n_tok below the threshold where the device wins
    ATTN_ALIBI,      // ALiBi distance bias: the device body has no such term
};
const char *const kAttnWhyName[7] = {"8-bit cache", "window", "masked block",
                                     "softcap",     "sink",   "short",
                                     "alibi"};

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

    // One entry per command in `queue`: which component issued it, and how much
    // arithmetic it asks for. The device executes a BATCH, so this is the only
    // place device time can be attributed to a component -- at flush the batch
    // is timed once and split across these by MAC count. It feeds KernelTime,
    // the same table the CPU backend fills, so --kprof reads the same on every
    // backend instead of each one inventing its own.
    struct Pending {
        const char *name;
        double macs;
    };
    std::vector<Pending> pending;
    // Accounting for the dispatch/execute split, always on (a batch is timed
    // with two clock reads that are already there for KRK_NPU_OPTIME).
    double last_batch_us = 0.0;

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
    // Set while the engine's load-time prepare pass runs: a single-row GEMM
    // packs instead of deferring, which is what moves the one-time cost out of
    // the first measured step. See wants_prepare() in npu_offload.hpp.
    bool preparing = false;
    // Row-shape gate: GEMMs with more rows than this stay on the host.
    //
    // The device and the host win different phases. Measured on the dense q8_0
    // fixture, same prompt: prefill (rows = prompt length) is 1.45x FASTER on
    // the host, because the host reads Q4_K/Q6_K/Q8_0 blocks where they lie and
    // the device has to expand them; decode (rows = 1) is 1.45x faster on the
    // device, because the host's advantage is a format the decode shape cannot
    // use. Neither arm is wrong; they are good at different shapes.
    //
    // So the device declines a shape it does not win, the same way it declines
    // a weight type it cannot read -- and the run summary counts it, because a
    // silent refusal would be indistinguishable from a device that is slow.
    i64 max_rows = 0;  // 0 = take every shape (previous behaviour)
    // KRK_NPU_OPTIME=1: time submit() and sync() separately so a batch's wall
    // time can be split into host dispatch and device wait.
    bool optime = false;
    // KRK_NPU_W8A8_EAGER=1 packs on first use instead of waiting for reuse.
    // Off is the measured default; see gemm_i8.
    bool w8a8_eager = false;
    // Weights already asked for once with a single row -- see gemm_i8.
    std::unordered_set<const void *> seen_1row;
    std::unordered_map<const void *, EntryI8> stage_i8;
    size_t stage_i8_bytes = 0;
    // Activation scratch for the int8 path: one int8 row block plus the
    // per-row scale the quantize op writes. Reused across calls; the stream is
    // ordered, so a later call may reuse the buffer while earlier commands are
    // still queued -- but a RESIZE must flush first, or a queued command would
    // be left pointing at freed storage.
    std::vector<int8_t> a8;
    std::vector<f32> a_scale;

    // ---- attention -------------------------------------------------------
    //
    // The device takes head-major [batch, heads, seq, dim]; Kraken's cache is
    // token-major with a row stride, so a call gathers into these and scatters
    // back. They are reused across layers and grown on demand -- and a GROWTH
    // flushes first, because a queued command may still point at the old
    // storage.
    //
    // Why gather instead of keeping the cache head-major: the cache is the
    // object the KV tiering, the int8 layout and the window logic all address
    // by (position, head) row. Changing its layout to suit one op would put
    // this optimisation in the way of every other path that touches it.
    std::vector<f32> attn_q, attn_k, attn_v, attn_o;
    i64 attn_min_tok = 16; // below this the gather costs more than it saves
    // Opt-in, and OFF by default, for a reason that is measured rather than
    // assumed: the engine's chunk-invariance guarantee (two runs that differ
    // only in --chunk produce the same --prefill-hash) holds because every
    // row's arithmetic is identical whatever the batch shape. The device
    // attention kernel IS chunk-invariant on its own -- verified, a run with
    // every chunk on the device hashes identically at chunk 8, 128 and 512 --
    // but the n_tok threshold means a small chunk takes the reference body and
    // a large one takes the device, and those two differ at 1.2e-07. So the
    // switch, not the kernel, is what costs the invariant, and it is the
    // switch that buys decode (the device loses below ~4 rows). Turning this
    // on therefore trades a bit-exactness property for prefill speed, which is
    // the user's call, not a default.
    bool attn_on = false;

    std::mutex mu;

    Counters n;

    const f32 *weight(const void *w, DType wt, i64 n_out, i64 n_in,
                      i64 rows);
    // The prefill half of weight(): a bounded ring of f32 scratch slots.
    const f32 *weight_streamed(const void *w, DType wt, i64 n_out, i64 n_in);
    std::vector<std::vector<f32>> ring;
    size_t ring_next = 0;  // next slot to hand out
    size_t ring_elems = 0; // slots' current capacity, in f32
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
    // Same, but tagged with the component that issued it and its MAC count, so
    // the measured batch time can be put on the right row of the trace.
    bool push_batch_named(const vnpu_command_t *cmds, size_t count,
                          const char *name, double macs);
    std::string describe() const;
};

// Wall clock for the submit/sync split. KRK_NPU_OPTIME=1 arms it; off, the
// only cost is a branch, because the callers check the flag first.
inline double optime_now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch())
        .count();
}

const f32 *Offload::weight(const void *w, DType wt, i64 n_out, i64 n_in,
                          i64 rows) {
    if (wt == DType::F32) return static_cast<const f32 *>(w);
    std::lock_guard<std::mutex> lk(mu);

    // Two staging policies, because a prefill and a decode step want opposite
    // things from the same cache. See the note in gemm() for the measurements
    // that separate them.
    //
    //  - A many-row GEMM is a prefill. It reads this weight once, streams it
    //    through the device's tile loop, and never comes back inside the chunk.
    //    Keeping the expansion would only fill RAM with data nothing will read
    //    again, which is what the old cache did: 2800 MiB of it on a 1.2 GiB
    //    model, and then 318 GEMMs handed back to the host for want of room.
    //    So prefill STREAMS: the weight is dequantized into one slot of a small
    //    rotating ring, the slot is read by the queued command, and the same
    //    slot is refilled for the next weight after a drain. RAM cost is
    //    ring_slots * largest_weight, independent of how big the model is --
    //    which is the whole point, because the model is 4.4 GiB expanded and
    //    this box has 3.9 GiB.
    //
    //  - A one-row GEMM is a decode step. The same weight comes back on every
    //    token, so the expansion is re-used hundreds of times and must persist.
    //    Those go in the memo cache, still bounded by the budget.
    if (rows > 1) return weight_streamed(w, wt, n_out, n_in);

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
    const double t_stage0 = optime ? optime_now_ms() : 0.0;
    const size_t row_bytes = dtype_row_bytes(wt, n_in);
    const u8 *src = static_cast<const u8 *>(w);
    f32 *dst = e.data.data();
    for (i64 o = 0; o < n_out; o++)
        dequant_row(wt, src + static_cast<size_t>(o) * row_bytes,
                    dst + static_cast<size_t>(o) * static_cast<size_t>(n_in), n_in);
    if (optime) {
        n.stage_ms += optime_now_ms() - t_stage0;
        n.staged_weights++;
    }
    const f32 *out = e.data.data();
    stage_bytes += need;
    stage.emplace(w, std::move(e));
    return out;
}

// The streaming half of weight(): dequantize into a rotating slot and hand the
// pointer straight to the caller.
//
// Correctness rests on one rule: a slot's bytes may not be overwritten while a
// queued device command still reads them. Slots are therefore handed out
// distinct until the ring wraps, and wrapping drains the queue first. A drain
// costs one submit+sync, and the ring is deep enough that a prefill pays it
// once per `ring_slots` weights rather than once per weight.
const f32 *Offload::weight_streamed(const void *w, DType wt, i64 n_out,
                                    i64 n_in) {
    const size_t elems = static_cast<size_t>(n_out) * static_cast<size_t>(n_in);
    // Grow the ring so every slot can hold the largest weight seen so far.
    // Growth invalidates nothing that is queued only because it is preceded by
    // the same drain the wrap does.
    if (elems > ring_elems) {
        flush(this);
        try {
            for (auto &slot : ring) slot.assign(elems, 0.0f);
            ring.shrink_to_fit();
            for (auto &slot : ring) slot.resize(elems);
        } catch (const std::exception &) {
            ring.clear();
            ring_elems = 0;
            return nullptr;
        }
        ring_elems = elems;
        ring_next = 0;
    }
    if (ring.empty()) {
        // A handful of slots: enough that the per-weight cost of the drain is
        // amortised away, small enough that even the widest MoE expert block
        // here (3072x768 f32 = 9 MiB) stays a rounding error next to the model.
        const int slots = static_cast<int>(env_i64("KRK_NPU_RING_SLOTS", 8));
        try {
            ring.resize(static_cast<size_t>(slots > 1 ? slots : 1),
                        std::vector<f32>(elems, 0.0f));
        } catch (const std::exception &) {
            return nullptr;
        }
        ring_elems = elems;
        ring_next = 0;
    }
    if (ring_next >= ring.size()) {
        if (!flush(this)) return nullptr;
        ring_next = 0;
    }
    f32 *dst = ring[ring_next++].data();
    const size_t row_bytes = dtype_row_bytes(wt, n_in);
    const u8 *src = static_cast<const u8 *>(w);
    const double t_stage0 = optime ? optime_now_ms() : 0.0;
    for (i64 o = 0; o < n_out; o++)
        dequant_row(wt, src + static_cast<size_t>(o) * row_bytes,
                    dst + static_cast<size_t>(o) * static_cast<size_t>(n_in), n_in);
    if (optime) {
        n.stage_ms += optime_now_ms() - t_stage0;
        n.staged_weights++;
        n.streamed_weights++;
    }
    return dst;
}

vnpu_packed_weights_t *Offload::weight_i8(const void *w, DType wt, i64 n_out,
                                          i64 n_in) {
    if (!w || n_out <= 0 || n_in <= 0) return nullptr;
    if (!fits_u32(n_out) || !fits_u32(n_in)) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    auto it = stage_i8.find(w);
    if (it != stage_i8.end()) return it->second.packed;

    // Packing a weight into the device's int8 layout costs about 2.3 ms per
    // 768x768 (253 MB/s standalone; across 737 weights that is 1718 ms of the
    // 3786 ms a W8A8 prefill once spent packing) and buys back 418 ms of GEMM
    // time -- the same GEMMs run 803 ms as f32 against 385 ms as int8. On a
    // prefill, which reads each expert once and never comes back, the pack is a
    // net loss: 3350 ms with it against 2400 ms without. On decode the same
    // weights are re-read on every token, so the pack is paid once and kept.
    //
    // A request counter was the first attempt at telling those apart. It
    // measured 18% better than always packing and still 36% worse than never
    // packing, because the load-time expert warmup touches every weight once:
    // prefill's second request then earned a pack the prefill never repaid.
    //
    // Rows are the honest signal. A many-row GEMM is a prefill -- the packed
    // weight is streamed once for the whole chunk, so packing costs more than it
    // saves. A one-row GEMM is a decode step -- the weight comes back on every
    // token. So pack for decode, stay f32 for prefill.
    // KRK_NPU_W8A8_EAGER=1 packs on every request, which is what earlier builds
    // did and what the 3350/3522 ms rows above were measured with.
    const double t_pack0 = optime_now_ms();
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
    const double t_host0 = optime ? optime_now_ms() : 0.0;
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

    if (optime) n.pack_host_ms += optime_now_ms() - t_host0;
    // Packing is a synchronous device call, and the queue may hold commands
    // referencing buffers it is about to be handed. Drain first so the pack
    // cannot interleave with work already issued. This is once per weight, on
    // the same warm-up pass that stages the f32 copies.
    const double t_fl0 = optime ? optime_now_ms() : 0.0;
    flush(this);
    if (optime) n.pack_flush_ms += optime_now_ms() - t_fl0;
    const double t_dev0 = optime ? optime_now_ms() : 0.0;

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
    // `pack_ms` and the weight count stay unconditional: they are two clock
    // reads per weight, and the run summary quotes them next to the packed
    // MiB. Gating them on KRK_NPU_OPTIME printed "packed in 0 ms", which is
    // the one number that makes a short decode look like an int8 regression.
    n.pack_ms += optime_now_ms() - t_pack0;
    n.packed_weights++;
    if (optime) n.pack_dev_ms += optime_now_ms() - t_dev0;
    e.bytes = elems;
    stage_i8.emplace(w, e);
    return packed;
}

bool Offload::push(const vnpu_command_t &cmd) { return push_batch(&cmd, 1); }

bool Offload::push_batch(const vnpu_command_t *cmds, size_t count) {
    return push_batch_named(cmds, count, nullptr, 0.0);
}

bool Offload::push_batch_named(const vnpu_command_t *cmds, size_t count,
                               const char *name, double macs) {
    queue.insert(queue.end(), cmds, cmds + count);
    for (size_t i = 0; i < count; i++)
        pending.push_back(Pending{name ? name : "npu_unnamed",
                                  i == 0 ? macs : 0.0});
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
    // Default 1 whenever the int8 path is on, because that is where the
    // measurement lands: prefill-shaped GEMMs are 2.2x (5 rows) to 1.0x (339
    // rows) faster on the host, and every row count measured from 5 up loses on
    // the device. A one-row GEMM is the single shape the device wins, and it is
    // the whole of decode. KRK_NPU_MAX_ROWS=0 restores take-everything.
    o->max_rows = env_i64("KRK_NPU_MAX_ROWS", o->w8a8 ? 1 : 0);
    o->optime = env_on("KRK_NPU_OPTIME");
    o->w8a8_eager = env_on("KRK_NPU_W8A8_EAGER");
    o->attn_on = env_on("KRK_NPU_ATTN");
    o->attn_min_tok = env_i64("KRK_NPU_ATTN_MIN_TOK", o->attn_min_tok);
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
    // Tile geometry, per phase.
    //
    // The device ships one default (64x256x64 here) and it is a prefill shape:
    // a 64-row M tile over a one-row GEMV spends 63/64 of its work on padding.
    // Its own autotuner picks 1x128x64 for M=1 and 32x128x64 for M=64 on this
    // host. KRK_NPU_TILES=M,K,N overrides both phases; any field left 0 keeps
    // the device default. Measured effect is in docs/NPU-CONSOLIDATED.md.
    if (const char *t = env_str("KRK_NPU_TILES")) {
        unsigned tm = 0, tk = 0, tn = 0;
        if (std::sscanf(t, "%u,%u,%u", &tm, &tk, &tn) == 3 && tm && tk && tn)
            vnpu_device_set_tiles(o->dev, tm, tk, tn);
        else
            KRK_WARN("npu: KRK_NPU_TILES wants M,K,N -- got '%s'", t);
    }
    vnpu_device_get_info(o->dev, &o->info);
    // How much non-f32 weight data the device may hold, expanded.
    //
    // This used to be a flat 1024 MiB, and the flat default was costing more
    // than it looked: on the 2 GiB q8_0 MoE it silently declined 393 of the
    // 1087 GEMMs in a 58-token prefill, and every one of them fell back to a
    // host GEMM. The device then sat idle through the host's work -- a device
    // trace of that run shows 508 ms busy out of a 3059 ms span, 83% idle,
    // with ~30 ms holes in the timeline exactly where an expert's
    // down-projection had been pushed back onto the CPU. Raising the budget to
    // 2048 MiB took the declines to zero and the prefill from 2885 ms to
    // 2398 ms (1.20x) with no other change.
    //
    // A fixed number cannot be right, because what the budget needs to cover is
    // a property of the model (its non-f32 weights, expanded to f32) and of the
    // host (how much memory is actually free). So the default is derived:
    // three quarters of what the host says is available, never below the old
    // 1024 MiB floor so no box gets a smaller budget than it had before.
    // KRK_NPU_WEIGHT_MB overrides it exactly, for reproducing a run or for
    // deliberately measuring a constrained one.
    if (const char *mb = env_str("KRK_NPU_WEIGHT_MB")) {
        o->stage_budget = static_cast<size_t>(std::atoll(mb)) * 1024ull * 1024ull;
    } else {
        const size_t avail = krk::host_available_bytes();
        const size_t floor_bytes = 1024ull * 1024ull * 1024ull;
        // A failed probe reports 0; fall back to the floor rather than to a
        // budget of nothing, which would send every weight to the host.
        const size_t derived = avail ? (avail / 4) * 3 : 0;
        o->stage_budget = derived > floor_bytes ? derived : floor_bytes;
    }
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

// The device's own ceiling, MEASURED on the device, in this process.
//
// A %peak column against the host's ISA is meaningless for an offloaded op: the
// question "could a better function do this faster" has to be asked of the
// thing that runs it. vnpu's own banner quotes tiles, not a rate; this
// runs one large matmul through the same stream the GEMMs use, twice, and
// reports the better of the two.
//
// 64 x 1024 x 1024 = 134 MMAC per pass, which is a few milliseconds of device
// time and enough to swamp one round trip. Cached for the process.
static double npu_peak_gmacs(Offload *o) {
    static double cached = -1.0;
    if (cached >= 0.0) return cached;
    cached = 0.0;
    if (!o || !o->stream) return cached;
    const uint32_t M = 64, K = 1024, N = 1024;
    std::vector<f32> A(static_cast<size_t>(M) * K, 0.5f);
    std::vector<f32> B(static_cast<size_t>(N) * K, 0.25f);
    std::vector<f32> C(static_cast<size_t>(M) * N, 0.0f);
    vnpu_command_t c{};
    c.opcode = VNPU_OP_MATMUL_F32;
    vnpu_matmul_f32_args_t &a = c.args.matmul_f32;
    a.A = A.data();
    a.B = B.data();
    a.bias = nullptr;
    a.C = C.data();
    a.M = M;
    a.K = K;
    a.N = N;
    a.trans_a = 0;
    a.trans_b = 1;
    a.lda = 0;
    a.ldb = 0;
    a.ldc = 0;
    a.alpha = 1.0f;
    a.beta = 0.0f;
    a.epilogue = VNPU_EPILOGUE_NONE;
    a.unit = o->unit;
    const double macs = static_cast<double>(M) * K * N;
    for (int pass = 0; pass < 3; pass++) {
        const double t0 = optime_now_ms();
        if (vnpu_stream_submit(o->stream, &c, 1, nullptr) != VNPU_OK) return 0.0;
        if (vnpu_stream_sync(o->stream) != VNPU_OK) return 0.0;
        const double us = (optime_now_ms() - t0) * 1000.0;
        if (us <= 0.0) continue;
        const double g = macs / us / 1e3;
        if (g > cached) cached = g;
    }
    return cached;
}

bool wants_prepare(const Offload *o) {
    if (!o || !o->w8a8) return false;
    // KRK_NPU_NO_PREPARE=1 restores the old behaviour, which is the only way to
    // A/B the pass itself in one binary. It is also the honest escape hatch: a
    // long-lived server prefers a pay-as-you-go first step to a 0.5 s spike at
    // load, and that is a judgement about the workload, not about the device.
    const char *e = std::getenv("KRK_NPU_NO_PREPARE");
    return !(e && e[0] == '1');
}
void set_preparing(Offload *o, bool on) {
    if (o) o->preparing = on;
}
bool preparing(const Offload *o) { return o != nullptr && o->preparing; }

bool flush(Offload *o) {
    if (!o || !o->stream) return true;
    o->n.flushes++;
    if (o->queue.empty()) return true;
    const size_t submitted = o->queue.size();
    // Always read the clock here: the trace needs submit+sync whether or not
    // KRK_NPU_OPTIME is set. Two steady_clock reads per batch are noise
    // against a device round trip.
    const double t_sub0 = optime_now_ms();
    const vnpu_status_t rc =
        vnpu_stream_submit(o->stream, o->queue.data(), submitted, nullptr);
    o->queue.clear();
    if (o->optime) o->n.submit_ms += optime_now_ms() - t_sub0;
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
    const double t_sync0 = o->optime ? optime_now_ms() : 0.0;
    const vnpu_status_t src = vnpu_stream_sync(o->stream);
    const double t_sync1 = optime_now_ms();
    if (o->optime) o->n.sync_ms += t_sync1 - t_sync0;
    // The batch has just completed, so `submit + sync` is its device time. Put
    // it on the components that filled it, split by MAC count -- a batch of one
    // GEMM and a batch of a quantize+matmul pair both land correctly, and a
    // batch with no arithmetic (a memset, a copy) splits evenly. This is the
    // NPU's `dev_us` column, measured, not estimated.
    if (KernelTime::enabled() && !o->pending.empty()) {
        const double batch_us =
            (t_sync1 - t_sub0) * 1000.0;
        double total_macs = 0.0;
        for (const auto &p : o->pending) total_macs += p.macs;
        const double share_us = batch_us / static_cast<double>(o->pending.size());
        for (const auto &p : o->pending) {
            const double part = total_macs > 0.0
                                    ? batch_us * (p.macs / total_macs)
                                    : share_us;
            KernelTime::get().add(p.name, part, p.macs, npu_peak_gmacs(o));
        }
        o->pending.clear();
    } else {
        o->pending.clear();
    }
    o->last_batch_us = (t_sync1 - t_sub0) * 1000.0;
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
    // The submit/sync split, only when the clocks were armed. `sync` is wall
    // time spent waiting for the device to finish a batch it has already been
    // handed; `submit` is the host handing it over. Anything neither of those
    // nor the device's own execution time is host work between batches.
    if (o->optime) {
        out += format(" | submit %.1f ms, sync %.1f ms over %llu batch(es), "
                      "pack %.1f ms over %llu weight(s) "
                      "[host %.1f, flush %.1f, dev %.1f]",
                      o->n.submit_ms, o->n.sync_ms, o->n.batches,
                      o->n.pack_ms, o->n.packed_weights, o->n.pack_host_ms,
                      o->n.pack_flush_ms, o->n.pack_dev_ms);
        out += format("; stage %.1f ms over %llu weight(s) (%llu streamed)",
                      o->n.stage_ms, o->n.staged_weights,
                      o->n.streamed_weights);
    }
    // The shape mix, always: it is what tells a prefill deficit apart from a
    // decode one, and both arms of every comparison need to show it.
    if (o->n.q8_0_gemm)
        out += format("; q8_0 in place %llu", o->n.q8_0_gemm);
    if (o->n.shape_refused)
        out += format("; shape refused %llu (rows > %lld: the host is the "
                      "faster arm for that shape)",
                      o->n.shape_refused, static_cast<long long>(o->max_rows));
    out += format("; gemm rows 1:%llu 2-8:%llu 9-32:%llu 33+:%llu",
                  o->n.rows_hist[0], o->n.rows_hist[1],
                  o->n.rows_hist[2], o->n.rows_hist[3]);
    // Only when the switch is on: a default run should not have to read a
    // zero-row about a path it never took.
    if (o->w8a8) {
        // The pack time is printed next to the MiB on purpose. It is a
        // ONE-TIME cost paid inside whichever step first re-requests a weight
        // with a single row, so a short decode shows it as a steady-state
        // regression: int8 loses to f32 at 16 tokens and wins 2.5x by 128.
        // Without the number in the line, that reads as "int8 is slower".
        out += format("; w8a8 %llu int8 (%.1f MiB packed in %.0f ms one-time), "
                      "%llu on f32",
                      o->n.w8a8_gemm,
                      static_cast<double>(o->stage_i8_bytes) / (1024.0 * 1024.0),
                      o->n.pack_ms, o->n.w8a8_fallback);
        if (o->n.w8a8_deferred)
            out += format(", %llu deferred (first use served as f32)",
                          o->n.w8a8_deferred);
    }
    // Attention: how many calls the device took, and -- when any stayed on the
    // reference -- the feature that blocked each group, by name. A model whose
    // attention never reaches the device should say why on the same line the
    // reader is already looking at, not leave it to be inferred.
    unsigned long long host_attn = 0;
    for (int i = 0; i < 7; i++) host_attn += o->n.attn_host[i];
    if (o->n.attn || host_attn) {
        out += format("; attn %llu dev, %llu host", o->n.attn, host_attn);
        bool first = true;
        for (int i = 0; i < 6; i++) {
            if (!o->n.attn_host[i]) continue;
            out += format(first ? " (" : ", ");
            out += format("%llu %s", o->n.attn_host[i], kAttnWhyName[i]);
            first = false;
        }
        if (!first) out += ")";
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
    // Packing only pays for itself when the weight is read many times.
    //
    // Measured on the 2 GiB q8_0 MoE at --npu-cpu, 58-token prompt: packing
    // costs ~1.2 s of wall across 737 weights (2.3 ms each at 253 MB/s; host
    // convert 977 ms + device pack 2376 ms) and buys back 418 ms -- the same
    // GEMMs run 803 ms as f32 against 385 ms as int8. End to end that made
    // w8a8 a 67% LOSS on prefill (4367 ms against 2619 ms off), because a
    // prefill chunk reads each expert once and never repays its conversion.
    // A request counter was tried next and still lost (3567 ms): the load-time
    // expert warmup touches every weight once, so prefill's "second" request
    // earned a pack that prefill then never repaid.
    //
    // Rows are the honest signal. A many-row GEMM is a prefill: the packed
    // weight is streamed once for the whole chunk, so the pack costs more than
    // it saves. A one-row GEMM is a decode step: the same weight comes back on
    // every token and the pack is paid once for hundreds of reads. So pack for
    // decode and stay f32 for prefill. KRK_NPU_W8A8_EAGER=1 restores packing
    // on every request, which is what the earlier builds did.
    if (!o->w8a8_eager) {
        bool defer = rows > 1;
        if (!defer && o->preparing) {
            // The prepare pass: every request has one row by construction, so
            // the "second request" heuristic below would pack nothing and the
            // pass would be a no-op. Record the weight as seen anyway, so the
            // post-load state is identical to a run that had already decoded.
            o->seen_1row.insert(w);
            defer = false;
        } else if (!defer) {
            // A single-row GEMM is not proof of decode on its own: a prefill
            // still issues some (106 weights on the 58-token MoE, measured),
            // and packing for those was 563 ms of a 2609 ms prefill for nothing.
            // Decode is the case where the SAME weight comes back with a single
            // row again, so require a second one-row request before packing.
            defer = o->seen_1row.insert(w).second;
        }
        if (defer) {
            o->n.w8a8_deferred++;
            return false; // caller falls through to the f32 GEMM
        }
    }
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

    if (!o->push_batch_named(c, 2,
                             rows == 1 ? "npu_gemm_i8_r1" : "npu_gemm_i8_rN",
                             static_cast<double>(rows) *
                                 static_cast<double>(n_out) *
                                 static_cast<double>(n_in))) {
        return false;
    }
    o->n.w8a8_gemm++;
    o->n.gemm++;
    o->n.flops += 2ull * static_cast<unsigned long long>(rows) *
                  static_cast<unsigned long long>(n_out) *
                  static_cast<unsigned long long>(n_in);
    return true;
}

} // namespace

// Q8_0 x f32 on the device, weights used where they lie.
//
// Kept out of the f32 staging path entirely: the pointer handed to the device
// is the caller's own mapped weight block, so this costs no host memory and no
// host time beyond the command itself. The kernel reads N*(K/32) blocks of 34
// bytes; see gemm_q80_avx512_tile / gemm_q80_avx2_tile in the engine, which
// are chosen there by runtime ISA detection.
bool gemm_q8_0(Offload *o, void *out, const void *x, const void *w, i64 n_out,
               i64 n_in, i64 rows) {
#if !defined(KRK_VNPU_HAS_Q8_0)
    // This vnpu.h has no VNPU_OP_MATMUL_Q8_0 (v1.0.0 dropped it; the CMake
    // capability probe reports which way this build went). Returning false
    // sends the call to the f32 staging path, which is a working path and the
    // more accurate one -- not a stub.
    (void)o; (void)out; (void)x; (void)w; (void)n_out; (void)n_in; (void)rows;
    return false;
#else
    vnpu_command_t c{};
    c.opcode = VNPU_OP_MATMUL_Q8_0;
    vnpu_matmul_q8_0_args_t &a = c.args.matmul_q8_0;
    a.A = static_cast<const float *>(x);
    a.B = w;
    a.C = static_cast<float *>(out);
    a.M = static_cast<uint32_t>(rows);
    a.K = static_cast<uint32_t>(n_in);
    a.N = static_cast<uint32_t>(n_out);
    a.lda = 0;
    a.ldc = 0;
    a.epilogue = VNPU_EPILOGUE_NONE;
    a.unit = o->unit;
    a.reserved0 = 0;
    if (!o->push_batch_named(&c, 1,
                             rows == 1 ? "npu_gemm_q8_0_r1"
                                       : "npu_gemm_q8_0_rN",
                             static_cast<double>(rows) *
                                 static_cast<double>(n_out) *
                                 static_cast<double>(n_in))) {
        return false;
    }
    o->n.gemm++;
    o->n.q8_0_gemm++;
    o->n.flops += 2ull * static_cast<unsigned long long>(rows) *
                  static_cast<unsigned long long>(n_out) *
                  static_cast<unsigned long long>(n_in);
    return true;
#endif // KRK_VNPU_HAS_Q8_0
}

bool gemm(Offload *o, void *out, const void *x, const void *w, DType wt,
          i64 n_out, i64 n_in, i64 rows) {
    if (!o || !o->stream || !out || !x || !w) return false;
    if (!fits_u32(rows) || !fits_u32(n_out) || !fits_u32(n_in)) return false;
    // Decline BEFORE queueing anything: the caller reads that as "the device
    // wrote nothing", flushes, and runs its own body.
    if (o->max_rows > 0 && rows > o->max_rows) {
        o->n.shape_refused++;
        return false;
    }
    // Opt-in W8A8. A decline here is not a substitution: it falls through to
    // the device's own f32 GEMM, which is the MORE accurate path on the same
    // device, and the run summary counts it.
    o->n.rows_hist[rows == 1 ? 0 : rows <= 8 ? 1 : rows <= 32 ? 2 : 3]++;
    if (o->w8a8 && gemm_i8(o, out, x, w, wt, n_out, n_in, rows)) return true;
    if (o->w8a8) o->n.w8a8_fallback++;
    // Q8_0 weights go to the device exactly as they sit in the mapped file.
    // No dequantization, no staging, no copy: the block array is handed over
    // and the kernel widens each 34-byte block into registers as it reads it.
    // This is the path that makes a 1.2 GiB checkpoint usable on a host that
    // cannot hold it expanded (4.4 GiB), and it removes the whole-model
    // conversion pass that used to run before the first token.
    if (wt == DType::Q8_0 && n_in % 32 == 0 && gemm_q8_0(o, out, x, w, n_out,
                                                         n_in, rows))
        return true;
    const f32 *b = o->weight(w, wt, n_out, n_in, rows);
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
    const double macs = static_cast<double>(rows) * static_cast<double>(n_out) *
                        static_cast<double>(n_in);
    if (!o->push_batch_named(&c, 1, rows == 1 ? "npu_gemm_f32_r1"
                                              : "npu_gemm_f32_rN",
                             macs)) {
        o->n.declined++;
        return false;
    }
    o->n.gemm++;
    o->n.flops += 2ull * static_cast<unsigned long long>(rows) *
                  static_cast<unsigned long long>(n_out) *
                  static_cast<unsigned long long>(n_in);
    return true;
}

// One attention call on the device's vectorised kernel.
//
// The gap this closes: the reference attention is a scalar loop, and attention
// is the SQUADRATIC term in prefill. Measured on the 4-layer fixture by
// sweeping prompt length and fitting a + b*T + c*T^2 to prefill.chunk, the
// quadratic term is 73.5 ms of a 229.7 ms prefill at T=257 (32%) and 293 ms of
// 519 ms at T=513 (56%) -- while the device sat idle for 44% of the span.
//
// Cost of using it: the gather below, which is O(cache) per call. At prefill
// shapes that is paid once against T rows of work and the kernel wins by
// 2.4-3.4x; at decode it is paid once against ONE row and loses, hence
// attn_min_tok.
bool attention_offered(const Offload *o) { return o && o->attn_on; }

bool attention(Offload *o, const AttnCall &c) {
    // A refusal is a refusal: count it under its reason and let the caller run
    // its own body. Nothing here writes to c.out.
    if (!o || !o->stream || o->mode != Mode::Graph) return false;
    if (!o->attn_on) return false; // opt-in; see Offload::attn_on
    // The first feature that blocks the device, checked in a fixed order so the
    // counts are deterministic. -1 means nothing blocked it.
    int why = -1;
    if (c.kv_int8) why = ATTN_INT8;
    else if (c.window > 0) why = ATTN_WINDOW;
    else if (c.skip_from >= 0) why = ATTN_SKIP;
    else if (c.softcap > 0.0f) why = ATTN_SOFTCAP;
    else if (c.alibi_max_bias > 0.0f) why = ATTN_ALIBI;
    else if (c.has_sinks) why = ATTN_SINK;
    else if (c.n_tok < o->attn_min_tok) why = ATTN_SMALL;
    if (why >= 0) {
        o->n.attn_host[why]++;
        return false;
    }
    if (!c.out || !c.q || !c.kcache || !c.vcache) return false;
    if (c.n_head <= 0 || c.n_kv <= 0 || c.hd <= 0 || c.n_tok <= 0 || c.pos0 < 0)
        return false;
    if (c.n_kv > c.n_head || c.n_head % c.n_kv != 0) return false;
    if (!fits_u32(c.n_head) || !fits_u32(c.hd) || !fits_u32(c.n_tok)) return false;
    // Causal calls see exactly pos0 + n_tok keys (the query rows are the last
    // n_tok positions of the cache); a non-causal call sees all of them too.
    const i64 seq_k = c.pos0 + c.n_tok;
    if (seq_k <= 0 || !fits_u32(seq_k)) return false;

    const size_t q_elems = static_cast<size_t>(c.n_head) * c.n_tok * c.hd;
    const size_t kv_elems = static_cast<size_t>(c.n_head) * seq_k * c.hd;
    if (q_elems > o->attn_q.size() || kv_elems > o->attn_k.size()) {
        // A queued command may still hold these pointers, so drain first.
        if (!flush(o)) return false;
        try {
            o->attn_q.resize(q_elems);
            o->attn_o.resize(q_elems);
            o->attn_k.resize(kv_elems);
            o->attn_v.resize(kv_elems);
        } catch (const std::exception &) {
            return false; // bad_alloc: the caller's own body is the answer
        }
    }

    // Gather token-major -> head-major, broadcasting grouped KV heads. The
    // broadcast is free here: the same pass has to touch every (position, head)
    // row anyway, so GQA costs the same as MHA.
    const f32 *qf = static_cast<const f32 *>(c.q);
    const f32 *kc = static_cast<const f32 *>(c.kcache) + c.layer * c.layer_stride;
    const f32 *vc = static_cast<const f32 *>(c.vcache) + c.layer * c.layer_stride;
    const i64 n_rep = c.n_head / c.n_kv;
    for (i64 h = 0; h < c.n_head; h++) {
        const i64 hkv = h / n_rep;
        f32 *qd = o->attn_q.data() + static_cast<size_t>(h) * c.n_tok * c.hd;
        for (i64 t = 0; t < c.n_tok; t++)
            std::memcpy(qd + t * c.hd, qf + (t * c.n_head + h) * c.hd,
                        sizeof(f32) * static_cast<size_t>(c.hd));
        f32 *kd = o->attn_k.data() + static_cast<size_t>(h) * seq_k * c.hd;
        f32 *vd = o->attn_v.data() + static_cast<size_t>(h) * seq_k * c.hd;
        for (i64 p = 0; p < seq_k; p++) {
            const f32 *src = kc + p * c.pos_stride + hkv * c.hd;
            std::memcpy(kd + p * c.hd, src, sizeof(f32) * static_cast<size_t>(c.hd));
            const f32 *srcv = vc + p * c.pos_stride + hkv * c.hd;
            std::memcpy(vd + p * c.hd, srcv, sizeof(f32) * static_cast<size_t>(c.hd));
        }
    }

    vnpu_command_t cmd{};
    cmd.opcode = VNPU_OP_ATTENTION_F32;
    vnpu_attention_f32_args_t &a = cmd.args.attention;
    a.Q = o->attn_q.data();
    a.K = o->attn_k.data();
    a.V = o->attn_v.data();
    a.mask = nullptr;
    a.O = o->attn_o.data();
    a.workspace = nullptr;
    a.workspace_bytes = 0;
    a.batch = 1;
    a.heads = static_cast<uint32_t>(c.n_head);
    a.seq_q = static_cast<uint32_t>(c.n_tok);
    a.seq_k = static_cast<uint32_t>(seq_k);
    a.head_dim = static_cast<uint32_t>(c.hd);
    // The kernel masks key ki when ki > qi + (seq_k - seq_q). With
    // seq_k = pos0 + n_tok that offset is exactly pos0, so the diagonal lands
    // on each query row's own absolute position -- correct for a whole prompt,
    // a later prefill chunk, and a single decode row alike. No mask is built.
    a.causal = c.causal ? 1u : 0u;
    a.scale = c.scale;
    a.unit = o->unit;

    if (!o->push(cmd)) {
        o->n.declined++;
        return false;
    }
    // The output is scattered back on the host on the next line, so this call
    // cannot stay queued. One sync per layer, against thousands of scalar
    // multiply-adds removed -- and the ops queued before it (the QKV GEMMs)
    // land with it, so the sync is not an extra round trip, it is the one the
    // caller was about to pay anyway.
    if (!flush(o)) {
        o->n.declined++;
        return false;
    }

    f32 *of = static_cast<f32 *>(c.out);
    for (i64 h = 0; h < c.n_head; h++) {
        const f32 *om = o->attn_o.data() + static_cast<size_t>(h) * c.n_tok * c.hd;
        for (i64 t = 0; t < c.n_tok; t++)
            std::memcpy(of + (t * c.n_head + h) * c.hd, om + t * c.hd,
                        sizeof(f32) * static_cast<size_t>(c.hd));
    }
    o->n.attn++;
    o->n.elements += static_cast<unsigned long long>(q_elems + 2 * kv_elems);
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
    if (!o->push_batch_named(&c, 1, "npu_rmsnorm", 0.0)) {
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
    if (!o->push_batch_named(&c, 1, "npu_swiglu", 0.0)) {
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
    if (!o->push_batch_named(&c, 1, "npu_add", 0.0)) {
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
    if (!o->push_batch_named(&c, 1, "npu_add_scaled", 0.0)) {
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
    if (!o->push_batch_named(&c, 1, "npu_mul", 0.0)) {
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

// The prepare-pass seam, in a build with no device: there is no one-time
// conversion to move, and the engine must still link.
bool wants_prepare(const Offload *) { return false; }
void set_preparing(Offload *, bool) {}
bool preparing(const Offload *) { return false; }

bool gemm(Offload *, void *, const void *, const void *, DType, i64, i64, i64) {
    return false;
}
bool rmsnorm(Offload *, void *, const void *, const f32 *, i64, i64, f32) {
    return false;
}
bool swiglu(Offload *, void *, const void *, const void *, i64) { return false; }
bool attention(Offload *, const AttnCall &) { return false; }
bool attention_offered(const Offload *) { return false; }
bool add(Offload *, void *, const void *, i64) { return false; }
bool add_scaled(Offload *, void *, const void *, f32, i64) { return false; }
bool mul(Offload *, void *, const void *, i64) { return false; }

#endif

} // namespace npu
} // namespace krk
