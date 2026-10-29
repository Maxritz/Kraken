// engine.cpp — prefill (batched) + decode (single token) with a paged-free
// contiguous KV cache, streaming output and stop-string handling.
#include "krk/engine.hpp"
#include "krk/mova.hpp"
#include "krk/diffusion.hpp"
#include "krk/engram.hpp"
#include "krk/moe_route.hpp"
#include "krk/hc.hpp"
#include "krk/dsv4_compress.hpp"
#include "krk/arch.hpp"
#include "krk/dflash.hpp"
#include "krk/host_time.hpp"
#include "krk/h2d_probe.hpp" // the measured promotion path
#include "krk/vram_guard.hpp"
#include "krk/json.hpp" // the expert index reader (warm_experts)
#include "krk/spec_sampling.hpp" // the sampled round's accept/reject rule
#include "krk/qsa.hpp" // the QSA indexer's host maths (pool / score / top-k)
#include "par_pool.hpp"

#if defined(_WIN32)
// A hard ceiling on this process's working set. The host policy plans in bytes,
// but a plan is not an enforcement: the mapping is read page by page and Windows
// grows the file cache until something pushes back, which on a 79 GB corpus is
// the desktop. With a hard maximum the kernel trims OUR pages past the cap
// instead of the machine's working set, so the corpus streams and the box stays
// responsive. Declared here rather than through <windows.h>, which would drag a
// platform header into a file that is otherwise portable.
extern "C" __declspec(dllimport) void *__stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall SetProcessWorkingSetSizeEx(
    void *process, size_t min_bytes, size_t max_bytes, unsigned long flags);
#define KRK_HARDWS_MAX_ENABLE 0x00000004ul
static bool cap_working_set(size_t max_bytes) {
    if (max_bytes == 0) return false;
    // The minimum must be a real number no larger than the maximum: the
    // (SIZE_T)-1 spelling means "trim now" for the older call and is rejected
    // by this one, which is how the cap silently failed the first time.
    const size_t min_bytes = max_bytes / 4;
    return SetProcessWorkingSetSizeEx(GetCurrentProcess(), min_bytes,
                                      max_bytes, KRK_HARDWS_MAX_ENABLE) != 0;
}
#else
static bool cap_working_set(size_t) { return false; }
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sys/stat.h> // struct stat, for the GGUF file-size probe

namespace {
// The configured cache format in the one spelling every call site uses.
krk::KvQuant kv_quant_of(const krk::EngineConfig &c) {
    return c.kv_quant == 1 ? krk::KvQuant::Int8 : krk::KvQuant::F16;
}
} // namespace
namespace krk {

namespace {

// Activation-typed pointer arithmetic: all workspaces are contiguous.
inline char *act_at(void *base, size_t act_size, i64 elem) {
    return static_cast<char *>(base) + static_cast<size_t>(elem) * act_size;
}

bool ends_with(const std::string &s, const std::string &suffix) {
    return suffix.size() <= s.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// KRK_SPLIT=1 separates the two halves of a decode step: the forward pass,
// and the fetch_logits that drains the stream to learn the sampled token.
//
// The per-op recorder cannot answer this question. It inserts an event pair
// around every op, which is the same thing being measured -- it perturbs the
// queue it is timing, and at ~20 us per op it is larger than most of what it
// would report.
//
// The flag was once believed to make this model generate different text run to
// run, which would have made it a race trigger as well as a timer. It does
// not: the two clock reads below are taken UNCONDITIONALLY and only the
// accumulation into the totals is gated, so the flag's whole effect is a few
// floating-point adds per token. 8+8 interleaved runs give 16 identical
// outputs. docs/PERF-ANALYSIS.md section 8 has the retraction; what survives is
// that a timing probe is worth nothing as evidence about which code path ran.
namespace {

// KRK_ROUTER_SYNC / KRK_SYNC_EXPERT are the A/B switches for the two host
// drains that a decode token can pay for. KRK_ROUTER_SYNC restores the explicit
// hipDeviceSynchronize that moe_ffn used to issue before it pulled the router
// row (40 per token); KRK_SYNC_EXPERT inserts one after each expert acquire.
// KRK_SWA_OFF=1 treats every layer as full attention. The A/B arm for the
// windowed prefill path: below the window length the window and no window are
// the same computation, so a text difference between the two arms is a defect
// in that path rather than a policy difference.
// Next-layer expert prefetch, OPT-IN (KRK_MOE_NEXT_PREFETCH=1). It is a no-op
// unless the WARM tier has a pinned prefix: the ticket copies pinned bytes by
// DMA, and a pageable source would put the host memcpy back at issue time.
//
// It is off by default because it MEASURED SLOWER on the file it was written
// for: REAP-320, 16 tokens, pinned WARM, 3 interleaved pairs, same text --
// 114.4 ms/step predicted against 124.7 ms/step predicted-ahead (mean of
// 109.3/126.0/107.8 against 121.4/125.6/127.1), i.e. -9%. The promotion traffic
// on that file is bandwidth-bound, not latency-bound: ~240 experts of ~2.2 MiB
// enter VRAM per token whatever the ordering, the copy stream already overlaps
// the kernels queued before it, and a prediction miss adds a promotion that the
// real routing then does not use. It is kept as the mechanism a better predictor
// would use (and because the guard it needs -- never evict the target layer's
// own resident set -- is a real bug it exposed), not as a default.
static bool next_prefetch_on() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_MOE_NEXT_PREFETCH");
        return e && *e && e[0] != '0';
    }();
    return v;
}

static i32 next_moe_layer(const std::vector<LayerWeights> &layers, i32 start,
                          i32 n_layer) {
    for (i32 layer = std::max<i32>(0, start);
         layer < n_layer && layer < static_cast<i32>(layers.size()); ++layer)
        if (layers[static_cast<size_t>(layer)].moe) return layer;
    return -1;
}

// Adaptive next-layer prediction is opt-in and only changes the expert IDs
// submitted to stage_ahead(); the target layer's native router remains the only
// authority for executed experts. Temporal and cross-layer signals are evaluated
// against every subsequent native route and confidence-gated before staging.
static bool moe_predict_on() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_MOE_PREDICT");
        return e && *e && e[0] != '0';
    }();
    return v;
}

static i32 moe_predict_history() {
    static const i32 v = [] {
        const char *e = std::getenv("KRK_MOE_PREDICT_HISTORY");
        const long n = e ? std::strtol(e, nullptr, 10) : 4;
        return n < 2 ? 2 : (n > 16 ? 16 : static_cast<i32>(n));
    }();
    return v;
}

static f32 moe_predict_min_overlap() {
    static const f32 v = [] {
        const char *e = std::getenv("KRK_MOE_PREDICT_MIN_OVERLAP");
        const f32 x = e ? static_cast<f32>(std::atof(e)) : 0.5f;
        return std::min(1.0f, std::max(0.0f, x));
    }();
    return v;
}

static i32 moe_predict_budget() {
    static const i32 v = [] {
        const char *e = std::getenv("KRK_MOE_PREDICT_K");
        const long n = e ? std::strtol(e, nullptr, 10) : 2;
        return n < 0 ? 0 : (n > 128 ? 128 : static_cast<i32>(n));
    }();
    return v;
}

static i32 moe_predict_policy() {
    // 1 = reuse the target layer's previous route (temporal), 2 = previous
    // MoE layer's previous-token route (cross-layer), 3 = combine both as a
    // unique candidate set, still bounded by the budget.
    static const i32 v = [] {
        const char *e = std::getenv("KRK_MOE_PREDICT_POLICY");
        const long n = e ? std::strtol(e, nullptr, 10) : 3;
        return n >= 1 && n <= 3 ? static_cast<i32>(n) : 3;
    }();
    return v;
}

static bool expert_prediction_model_supported(const ModelConfig &mc) {
    // Cache hints consume only the final native route IDs. That makes learned,
    // recurrent, HC, and even hash-routed layers safe: the predictor cannot
    // affect routing or output. Architecture gating is therefore the real
    // contract -- only models declared MoE can enter this path.
    return mc.is_moe;
}

static bool swa_off() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_SWA_OFF");
        return e && e[0] != '0';
    }();
    return v;
}

bool router_sync() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_ROUTER_SYNC");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// KRK_DEVICE_SOFTMAX=1 runs the MoE router's activation on the device and uses
// its probabilities for the selection, so the text a device softmax produces is
// the text that is generated rather than a comparison in a comment. The drift
// against the host answer is accumulated and printed at teardown. Not a speed
// arm: it still downloads the logits as well, so it is deliberately slower than
// either path -- the question it answers is numerics, and the answer has to come
// from the same greedy rule the repo's gates use.
bool device_softmax() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_DEVICE_SOFTMAX");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// KRK_DEV_PLAN=1 selects top-k and builds the layer's expert plan on the
// device, then verifies them against the host result before using them: the ids,
// the weights and the plan's offsets and row order must agree EXACTLY, or the
// layer falls back to the host plan and the mismatch is reported. That check is
// the point -- a plan is bookkeeping, so "the GPU agrees with the reference on
// every layer of every run" is a property that can be proven rather than argued,
// and it is what makes it safe for the per-expert loop to consume a plan it did
// not build.
bool dev_plan() {
    // ON by default since the measurement, because it is free of risk by
    // construction -- the device ids, weights and plan are compared against the
    // host's every layer and the host plan is used the moment any of them
    // differs. KRK_DEV_PLAN=0 restores the host path for an A/B.
    //
    // 30B-A3B Q2_K, greedy, ctx 512, 16 tokens (perf-traces/dev-KRK_DEV_PLAN_1.*):
    // 4.6 -> 7.8 tok/s, md5 identical. Per token, `moe.route.rest` (host top-k +
    // plan, the parent of the expert loop) fell 158.8 -> 67.2 ms and
    // `expert.warm.read` 1444 -> 510 ms over the run at 1.4 -> 4.1 GB/s. The step
    // was 202 ms of which the GPU did 57; routing bookkeeping was the rest.
    static const bool on = [] {
        const char *v = std::getenv("KRK_DEV_PLAN");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// KRK_DEV_PTR=1 runs a MoE layer as ONE launch whose experts are chosen on the
// device from ExpertCache's pointer table: the ids and the weights never come
// back, so the layer stops waiting for its own GPU work to retire. It engages
// when the whole routed set is ADDRESSABLE from a kernel -- in VRAM, or in the
// pinned host pool the kernel reads in place -- and the shape is one the kernel
// expresses; anything else falls back to the host path and says why once. Opt-in because it keeps no host bits: the
// router's softmax is the device's `expf`, and the drift it costs is measured in
// docs/AUDIT-EXPERT-CACHE.md rather than assumed away.
bool dev_ptr() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_DEV_PTR");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// Whether the one-launch dispatch may serve a SINGLE row against a host-backed
// table. It could not while the kernel launched one block per token: that is one
// CU for a whole layer, measured on Qwen3.5-35B-A3B at 154.9 ms/step against
// 127.5 for the per-expert path, so a host-backed table was prefill-only.
//
// The kernel now tiles each (token, slot)'s rows over many blocks (see
// kernels/moe_table.hpp), which is what the objection was about. KRK_MOE_TILE_DECODE=0
// restores the exclusion for an A/B.
bool tile_decode() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_MOE_TILE_DECODE");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// Whether the tiled dispatch may serve a PREFILL CHUNK. It could not while the
// table kernel computed one activation row at a time: a chunk is a batch GEMM,
// and the row-at-a-time form measured 55 tok/s against the per-expert path's 144
// on Qwen3.5-35B-A3B. Off by default until the chunk form is measured above the
// per-expert path; KRK_MOE_TILE_CHUNK=1 turns it on for that measurement.
bool tile_chunk() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_MOE_TILE_CHUNK");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// KRK_EXPERT_INPLACE=0 turns OFF reading a WARM expert's weights in place.
//
// The three tiers are supposed to hold what each is good at: the hot set in
// VRAM, some RAM for the next set, the file mapping for everything else. Until
// this existed the RAM tier could not be USED without being copied into VRAM
// first, so on a card smaller than the routed set every layer re-copied ~3 MiB
// per routed expert to read it once and evicted the expert the next layer
// wanted -- which is why a 6-12 GiB card could only run these models by holding
// nearly the whole routed set resident (REAP-288 wants 42.2 GiB) or by running
// at disk speed.
//
// A kernel can read the pinned pool directly: tools/probe_hostread.hip measured
// 24.3 GiB/s for a 256 MiB pinned buffer against 485 GiB/s for the same read out
// of device memory, with the bytes read back exactly. So the RAM tier is read
// where it lives, VRAM keeps only the set the tier ranked hot, and the copy
// disappears. `Backend::host_load_supported()` is the capability half -- a
// backend that has not measured it reports false and this never engages.
bool expert_inplace() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_EXPERT_INPLACE");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// KRK_WARM_AHEAD=0 turns off reading the NEXT layer's set on a worker thread.
//
// The read is 274 MiB/token on REAP-288 at the small-machine shape, and it was
// done on the host thread inside the layer, serial with that layer's GPU work.
// Nothing about it depends on what the layer computes, so it belongs one layer
// ahead, on a thread that is not the one issuing kernels -- which is what "some
// RAM for the next set" has to mean, or the tier is filled only when it is
// demanded and every fill is a stall.
//
// The ids are the previous token's routing at that layer (Engine's moe_hint_),
// so this is a PREDICTION: a wrong guess costs a read that is dropped, never a
// different answer.
bool warm_ahead() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_WARM_AHEAD");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// KRK_ROUTER_2X=1 issues the router readback twice. See the call site: the
// second copy is the round-trip control for the first one's device wait.
// Measurement-only routing arm: no readback, a fixed expert per row. Test
// output differs by construction -- the point is the clock, not the answer.
static bool route_fake() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_ROUTE_FAKE");
        return e && e[0] && e[0] != '0';
    }();
    return v;
}

bool router_fetch2() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_ROUTER_2X");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

bool sync_expert() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_SYNC_EXPERT");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// KRK_TOPK_SYNC restores the explicit hipDeviceSynchronize topk_argmax used to
// issue between its launch and its 16-byte readback. `download` is a BLOCKING
// hipMemcpy on the same stream, so it already orders itself behind that launch
// -- the same argument the router readback above makes -- and the drain is a
// second, redundant wait that a decode token pays once. Off by default; the
// switch is the A/B arm the cost is measured against.
bool topk_sync() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_TOPK_SYNC");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// KRK_LOGITS_SYNC=0 removes the drain in front of a full logits row readback.
// It sits on the fallback path (a row is only downloaded when the device-side
// argmax is unavailable), so it is off the plain decode path on a GPU -- but it
// is the same pattern as topk_argmax's, and a switch is what makes that a
// measurement rather than a claim. Default is the drain present, as shipped.
bool logits_sync() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_LOGITS_SYNC");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// KRK_SPEC_ROW_SYNC=0 removes the per-round drain in front of the speculative
// loop's single-row logits recompute. Reachable only with a drafter, where the
// row is recomputed once per speculation round; default is the drain present.
bool spec_row_sync() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_SPEC_ROW_SYNC");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// KRK_MOE_PLAN=0 restores one plan upload PAIR per expert visit. The default
// builds the layer's whole routing plan once and uploads it with two copies, so
// a visit reads its segment of the device buffer instead of issuing its own
// pair -- on a 256-expert layer at top-8 that is 16 launches per layer of the
// decode step down to 2.
bool moe_plan_grouped() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_MOE_PLAN");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

bool split_timing_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_SPLIT");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

double g_split_fwd_us = 0.0;
double g_split_fetch_us = 0.0;
i64 g_split_steps = 0;

}  // namespace

// Greedy selection over a device-downloaded logits row. The sampler is not
// used here because speculative verification only pursues argmax branches —
// that is what makes accepted tokens bit-identical to plain greedy decoding.
i32 argmax_of(const f32 *logits, i64 n) {
    i32 best = 0;
    f32 bv = logits[0];
    for (i64 i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = static_cast<i32>(i);
        }
    }
    return best;
}

// Diagnostic for the top-k candidates of one decoding step. Printed with
// %.9g so two runs can be diffed byte for byte: if the top-1 log-prob changes
// between runs the divergence is already in the logits at that step, and if it
// only starts changing several steps in, something upstream is drifting
// rather than racing.
void dump_topk(i32 k, i32 step, i64 pos, const f32 *logits, i64 n) {
    const i32 kk = static_cast<i32>(std::min<i64>(k, n));
    // Partial selection sort over a private copy: k passes over the vocab.
    // NaN counts are reported because a single non-finite logit is enough to
    // make the whole softmax meaningless, and it is invisible in the top-1.
    i64 nan = 0, inf = 0;
    for (i64 i = 0; i < n; i++) {
        if (std::isnan(logits[i])) nan++;
        else if (std::isinf(logits[i])) inf++;
    }
    std::vector<f32> work(logits, logits + n);
    std::vector<i32> ids;
    std::vector<f32> vals;
    ids.reserve(static_cast<size_t>(kk));
    vals.reserve(static_cast<size_t>(kk));
    for (i32 c = 0; c < kk; c++) {
        i64 best = c;
        for (i64 i = c + 1; i < n; i++)
            if (work[static_cast<size_t>(i)] > work[static_cast<size_t>(best)])
                best = i;
        if (c > 0 && best == c &&
            !(work[static_cast<size_t>(c)] > work[static_cast<size_t>(c - 1)]))
            break; // the tail is already in order; nothing left to rank
        std::swap(work[static_cast<size_t>(c)], work[static_cast<size_t>(best)]);
        ids.push_back(static_cast<i32>(best));
        vals.push_back(work[static_cast<size_t>(c)]);
    }
    f32 mx = vals.empty() ? 0.0f : vals[0];
    f32 sum = 0.0f;
    for (f32 v : vals) sum += std::exp(static_cast<f64>(v - mx));
    std::fprintf(stderr, "step %d pos %lld nan=%lld inf=%lld top%d:", step,
                 static_cast<long long>(pos), static_cast<long long>(nan),
                 static_cast<long long>(inf), static_cast<int>(vals.size()));
    for (size_t i = 0; i < vals.size(); i++) {
        const f64 lp = static_cast<f64>(vals[i]) -
                       (std::log(static_cast<f64>(sum)) +
                        static_cast<f64>(mx));
        std::fprintf(stderr, " %d=%.9g(%.6f)", ids[i],
                     static_cast<double>(vals[i]), lp);
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

// How one accepted token reaches the caller: appended to the result, matched
// against the stop strings, and streamed out with enough held back to cover a
// stop string straddling a token boundary.
//
// Both decode loops emit tokens through this one policy. It used to be
// duplicated verbatim in `generate` and `generate_speculative`, which is how a
// debug flag added to one of them silently did nothing in the other.
class TokenEmitter {
public:
    TokenEmitter(const GenerateParams &p, const Tokenizer &tok, Sampler &smp,
                 GenerateResult *res, const std::vector<i32> &prompt_ids)
        : p_(p), tok_(tok), smp_(smp), res_(res) {
        for (const std::string &s : p_.stop)
            max_stop_ = std::max(max_stop_, s.size());
        if (p_.echo_prompt) {
            res_->text = p_.prompt;
            res_->tokens = prompt_ids;
            p_.sink.emit(p_.prompt.c_str(), -1, false);
        }
    }

    // Records one accepted token and returns true if it completed a stop
    // string, in which case the caller must stop generating.
    bool accept(i32 token) {
        const std::string piece = tok_.decode(&token, 1);
        res_->tokens.push_back(token);
        res_->text += piece;
        res_->generated++;
        smp_.accept(token);

        size_t stop_len = 0;
        for (const std::string &s : p_.stop) {
            if (!s.empty() && ends_with(res_->text, s) && s.size() > stop_len)
                stop_len = s.size();
        }
        const bool stop_hit = stop_len != 0;

        pending_ += piece;
        if (stop_hit) {
            res_->text.resize(res_->text.size() - stop_len);
            const size_t keep =
                pending_.size() >= stop_len ? pending_.size() - stop_len : 0;
            pending_.resize(keep);
        }
        if (pending_.size() > max_stop_) {
            const size_t flush = pending_.size() - max_stop_;
            p_.sink.emit(pending_.substr(0, flush).c_str(), token, false);
            pending_.erase(0, flush);
        }
        return stop_hit;
    }

    // Flushes the held-back tail and the end-of-stream marker. `stopped`
    // suppresses the tail: a matched stop string already consumed it.
    void finish(bool stopped) {
        if (!stopped && !pending_.empty())
            p_.sink.emit(pending_.c_str(), -1, false);
        pending_.clear();
        p_.sink.emit(nullptr, -1, true);
    }

private:
    const GenerateParams &p_;
    const Tokenizer &tok_;
    Sampler &smp_;
    GenerateResult *res_;
    size_t max_stop_ = 0;
    std::string pending_;
};

} // namespace

Engine::~Engine() { shutdown(); }

// KRK_DEQUANT_PROBE=<tensor>[:<expert>][:<row>]
//
// Dequantize ONE row of one tensor on the device and on the host, from the same
// bytes, and report where they disagree. The question it answers is the one a
// per-stage activation trace cannot: a row whose *values* are finite can still
// be the wrong row, and "the residual went NaN in layer N" does not say whether
// the arithmetic or the address is at fault.
//
// It reads the file's own tensor table for the row's address and the geometry of
// this build for its length, so a disagreement between the two shows up as the
// span line rather than as a wrong answer. The device side goes through
// Backend::embed -- a device row-dequant with no other caller-side state -- with
// a one-row table, so the probe needs no new backend entry point; the device's
// activations are f16, so agreement is judged at 1e-2 relative, not bit-exactly.
//
// Every row index is flat: expert e, row r is flat row e*ne[1] + r, which is how
// a 3-D expert tensor is laid out.
static void run_dequant_probe(Backend &be, const std::string &model_path) {
    const char *spec = std::getenv("KRK_DEQUANT_PROBE");
    if (!spec) return;
    std::string s(spec);
    std::vector<std::string> parts;
    for (size_t p = 0; p <= s.size();) {
        const size_t c = s.find(':', p);
        parts.push_back(s.substr(p, c == std::string::npos ? std::string::npos : c - p));
        if (c == std::string::npos) break;
        p = c + 1;
    }
    if (parts.empty() || parts[0].empty()) {
        KRK_WARN("KRK_DEQUANT_PROBE needs <tensor>[:<expert>][:<row>]");
        return;
    }
    const std::string name = parts[0];
    const i64 expert = parts.size() > 1 ? std::atoll(parts[1].c_str()) : 0;
    const i64 row = parts.size() > 2 ? std::atoll(parts[2].c_str()) : 0;

    Gguf g;
    std::string gerr;
    if (!g.load(model_path, &gerr)) {
        KRK_WARN("dequant probe: cannot reopen %s (%s)", model_path.c_str(), gerr.c_str());
        return;
    }
    const GgufTensor *t = g.tensor(name);
    if (!t) {
        KRK_WARN("dequant probe: no tensor '%s' in this file", name.c_str());
        return;
    }
    const i64 width = static_cast<i64>(t->ne[0]);
    const i64 rows_per_expert = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
    const i64 n_experts = t->n_dims >= 3 ? static_cast<i64>(t->ne[2]) : 1;
    if (expert < 0 || expert >= n_experts || row < 0 || row >= rows_per_expert) {
        KRK_WARN("dequant probe: %s has %lld experts of %lld rows, asked for %lld/%lld",
                 name.c_str(), static_cast<long long>(n_experts),
                 static_cast<long long>(rows_per_expert),
                 static_cast<long long>(expert), static_cast<long long>(row));
        return;
    }
    const size_t row_bytes = dtype_row_bytes(t->type, width);
    const i64 flat_row = expert * rows_per_expert + row;
    const size_t total_rows = static_cast<size_t>(n_experts) * static_cast<size_t>(rows_per_expert);

    std::fprintf(stderr,
                 "[deqprobe] %s type=%s row_width=%lld experts=%lld rows/expert=%lld\n"
                 "[deqprobe]   flat row %lld (expert %lld row %lld), row_bytes=%zu, "
                 "computed span=%zu B vs tensor n_bytes=%zu (%s)\n",
                 name.c_str(), dtype_name(t->type), static_cast<long long>(width),
                 static_cast<long long>(n_experts), static_cast<long long>(rows_per_expert),
                 static_cast<long long>(flat_row), static_cast<long long>(expert),
                 static_cast<long long>(row), row_bytes, total_rows * row_bytes,
                 t->n_bytes, total_rows * row_bytes == t->n_bytes ? "agrees" : "DISAGREES");

    // A third numeric field is how many rows to walk (default 1), because one
    // row is not the question when the symptom is an all-NaN output row: a gemv
    // output element is a dot over one weight row, so a single non-finite row
    // ANYWHERE in the slab poisons every token that routes to the expert. The
    // scan is what turns "this row is clean" into "this slab is clean".
    const i64 n_scan = parts.size() > 3 ? std::max<i64>(1, std::atoll(parts[3].c_str())) : 1;
    const size_t as = be.act_size();
    std::vector<f32> host(static_cast<size_t>(width));
    std::vector<f32> dev(static_cast<size_t>(width));
    void *dw = be.alloc(row_bytes);
    void *dout = be.alloc(static_cast<size_t>(width) * as);
    // `tokens` is HOST memory by the Backend contract -- the device backend
    // stages the ids into its own scratch -- so this must not be a device
    // allocation: handing it one has the backend read a VRAM address as a
    // host pointer and fault.
    const i32 zero = 0;
    i64 rows_bad_host = 0, rows_bad_dev = 0, rows_disagree = 0;
    i64 first_bad_row = -1, first_bad_index = -1;
    f64 worst = 0.0;
    for (i64 k = 0; k < n_scan; k++) {
        const i64 r = row + k;
        const i64 flat = expert * rows_per_expert + r;
        if (r >= rows_per_expert) break;
        const u8 *rsrc = t->data + static_cast<size_t>(flat) * row_bytes;
        dequant_row(t->type, rsrc, host.data(), width);
        be.upload(dw, rsrc, row_bytes);
        be.embed(dout, dw, t->type, 1, width, &zero, 1);
        be.download_f32(dev.data(), dout, width);
        bool hbad = false, dbad = false, dis = false;
        for (i64 i = 0; i < width; i++) {
            const f32 a = host[static_cast<size_t>(i)], b = dev[static_cast<size_t>(i)];
            if (!(a == a) || a > 3.0e38f || a < -3.0e38f) hbad = true;
            if (!(b == b) || b > 3.0e38f || b < -3.0e38f) dbad = true;
            const f64 rel = std::fabs(static_cast<f64>(b) - a) /
                            (1.0 + std::fabs(static_cast<f64>(a)));
            if (rel > worst) worst = rel;
            if (rel > 1e-2) {
                dis = true;
                if (first_bad_index < 0) first_bad_index = i;
            }
        }
        if (hbad) rows_bad_host++;
        if (dbad) rows_bad_dev++;
        if (dis) rows_disagree++;
        if ((hbad || dbad || dis) && first_bad_row < 0) first_bad_row = r;
        if ((hbad || dbad || dis) && n_scan == 1) {
            std::fprintf(stderr, "[deqprobe]   row %lld: host %s, device %s\n",
                         static_cast<long long>(r), hbad ? "NON-FINITE" : "finite",
                         dbad ? "NON-FINITE" : "finite");
        }
    }
    be.release(dw);
    be.release(dout);

    std::fprintf(stderr,
                 "[deqprobe]   scanned %lld row(s): host non-finite %lld, device non-finite "
                 "%lld, disagree>1e-2 %lld (first bad row %lld, index %lld) | worst rel %.6g\n",
                 static_cast<long long>(n_scan), static_cast<long long>(rows_bad_host),
                 static_cast<long long>(rows_bad_dev), static_cast<long long>(rows_disagree),
                 static_cast<long long>(first_bad_row),
                 static_cast<long long>(first_bad_index), worst);
}

// KRK_PIN_AUTO=0 is the A/B arm for the automatic pinned WARM prefix; the
// explicit switches (KRK_HOST_PIN_MB, --expert-pin-mb) win over both.
static bool pin_auto_on() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_PIN_AUTO");
        return !(e && e[0] == '0');
    }();
    return v;
}

bool Engine::init(Backend *be, const EngineConfig &cfg, std::string *err) {
    // Everything init does -- map, parse, upload, tokenizer, KV tier, expert
    // warmup -- is the load phase for both profilers.
    krk::HostTime::set_phase(krk::HostTime::kLoad);
    be_ = be;
    cfg_ = cfg;
    // The activation-range counters are process-wide device state; zero them so
    // the line this run prints describes this run.
    be_->reset_activation_saturation();
    // KRK_DUMP=<path>: append every stage of the forward pass (the hidden state
    // after embed and after each layer, for the row that decides the next
    // token) as text, so the same prompt can be run on two backends and the
    // first stage that disagrees found by diffing the files. A debug hook on
    // the hot path's *shape*, not its math: nothing here is compiled out, it
    // just does nothing unless the variable is set.
    if (const char *dump = std::getenv("KRK_DUMP")) dump_path_ = dump;
    // KRK_DUMP_ROUTE=1: one line per (layer, token) with the selected
    // experts and their mix weights -- the router is the half of a MoE
    // mismatch that moe.sum cannot show.
    if (const char *r = std::getenv("KRK_DUMP_ROUTE"))
        dump_route_ = r[0] != '0';
    // KRK_GDN_GATE: the delta-net output gate's A/B arm. -1 = the family's own
    // answer (qwen3.5 silu, qwen4exp sigmoid), 0 = force silu, 1 = force
    // sigmoid. Read once per engine, not in a function-static: a suite group
    // has to be able to run both arms in one process, and a static would
    // freeze whichever arm ran first -- which is how the first version of the
    // test read "the arm never reached the graph" for a working arm.
    if (const char *g = std::getenv("KRK_GDN_GATE")) gdn_gate_ = g[0] == '0' ? 0 : 1;
    // Residency preference in the router: --router-resident <margin>, or
    // KRK_ROUTER_TIE_MARGIN for the A/B. Off unless a positive margin is asked
    // for, because it changes the route and therefore the text.
    router_pref_ = false;
    router_tie_margin_ = 0.0f;
    if (const char *e = std::getenv("KRK_ROUTER_TIE_MARGIN")) {
        const f32 m = static_cast<f32>(std::atof(e));
        if (m > 0.0f) {
            router_tie_margin_ = m;
            router_pref_ = true;
        }
    }
    if (cfg_.router_tie_margin >= 0.0f) {
        router_tie_margin_ = cfg_.router_tie_margin;
        router_pref_ = router_tie_margin_ > 0.0f;
    }
    if (router_pref_)
        KRK_INFO("router: a pick within %.1f%% of the top score goes to a resident "
                 "expert (VRAM, then WARM) instead",
                 static_cast<f64>(router_tie_margin_) * 100.0);
    // Top-k' lookahead. Absolute k' (not a multiplier), because "twice the k a
    // file declares" is a different concurrency on a top-4 and a top-8 model and
    // the extra experts are paid for in promotions.
    expert_lookahead_ = 0;
    if (const char *e = std::getenv("KRK_EXPERT_LOOKAHEAD")) {
        const long v = std::atol(e);
        if (v > 0) expert_lookahead_ = static_cast<i32>(v);
    }
    if (expert_lookahead_ > 0)
        KRK_INFO("expert cache: top-k' lookahead at k'=%d -- the runners-up are "
                 "staged only when they are already in WARM",
                 expert_lookahead_);
    // The request is not an enable decision until the loaded architecture and
    // final pinned-WARM budget have both been checked below.
    moe_predict_enabled_ = false;
    moe_predict_ids_.clear();
    moe_predictor_.configure(0, 0, moe_predict_history());
    trace_hash_on_ = cfg.trace_hash;
    trace_stage_.clear();
    // The hash is recorded from whatever kernel actually ran, so the backend
    // has to know it is in hash mode BEFORE the first prefill: with the
    // matrix-core prefill available, a hash taken from it would be a hash of
    // the wrong arithmetic and the chunk-invariance claim would be about a
    // kernel nobody ships by default.
    if (be_) be_->set_attn_hash_mode(trace_hash_on_);
    // Size the CPU worker pool before the first forward pass.
    // 0 (the default) keeps the pool's own sizing: one worker per
    // hardware thread minus the calling thread.
    if (cfg_.threads > 0)
        Pool::get().set_workers(cfg_.threads);
    if (!be_->caps().has_wmma) {
        KRK_WARN("device %s (%s) has no WMMA — falling back to packed-math kernels",
                 be_->caps().name.c_str(), be_->caps().gfx.c_str());
    }

    const Timer load_timer;
    // On failure the caller still owns `be` and will delete it. Release whatever
    // the partial load mapped through the (still valid) backend, then detach both
    // pointers so no later unload touches the freed backend.
    // Dense-weight residency has to be decided before the first tensor is
    // uploaded, so the budget travels with the Model rather than the Engine.
    // 0 is automatic; a negative --weight-vram-mb (or KRK_WEIGHT_STREAM=0)
    // means "never stream" and is passed as an unreachable budget.
    {
        size_t wb = 0;
        if (cfg_.weight_vram_mb > 0) {
            wb = static_cast<size_t>(cfg_.weight_vram_mb) << 20;
        } else {
            const char *env = std::getenv("KRK_WEIGHT_VRAM_MB");
            if (env && *env) {
                const long long v = std::atoll(env);
                if (v > 0) wb = static_cast<size_t>(v) << 20;
            }
        }
        // A negative flag, or KRK_WEIGHT_STREAM=0, disables streaming: the
        // budget becomes unreachable rather than 0, because 0 is "auto".
        const char *off = std::getenv("KRK_WEIGHT_STREAM");
        if (cfg_.weight_vram_mb < 0 || (off && off[0] == '0'))
            wb = std::numeric_limits<size_t>::max();
        model_.set_weight_budget(wb);
    }
    // The nextn block is parsed during load, so the request has to travel with
    // the Model rather than be read afterwards.
    model_.set_mtp_load(cfg.mtp_probe || cfg.mtp_spec || !cfg.mtp_path.empty());
    model_.set_mtp_path(cfg.mtp_path);
    // `load.model` is the loader end to end: the mapping, the metadata, every
    // tensor's upload. It dwarfs a short generation, and until now it existed
    // only as the `[stats ] load` line main_cli prints from its own timer, so
    // nothing said which part of it a slow start was spending.
    {
    KRK_TIME_HOST("load.model");
    // Before load(): the template is resolved while the metadata is being
    // read, so an override has to be in hand by then.
    if (!cfg.chat_template.empty())
        model_.set_chat_template_override(cfg.chat_template);
    if (!model_.load(*be_, cfg.model_path, err)) {
        model_.unload();
        be_ = nullptr;
        return false;
    }
    }

    // The engine's own guard, and it is the one the loader's escape hatch does
    // NOT lift. KRK_ARCH_ALLOW_EXPERIMENTAL=1 is a LOADER switch: it lets the
    // tensors of a refused schema be read so a half-ported architecture can be
    // checked stage by stage. It says nothing about the graph, and for an arch
    // whose ATTENTION has no implementation here the graph has no attention in
    // it at all -- so a run would decode text that is fluent, plausible and not
    // the model. That is the failure this engine must never produce, so the
    // run is refused here, after the load and before anything is generated.
    //
    // It has to be after the load: the arch name comes from the file, and the
    // table entry is what says whether the attention path exists. The same
    // position (and the same reason) as the hyper-connection block below.
    {
        const std::string &arch = model_.cfg().arch;
        const ArchSpec *spec = arch_lookup(arch);
        if (spec && !spec->attention_implemented) {
            if (err)
                *err = "arch '" + arch +
                       "': this engine has no implementation for this "
                       "architecture's attention path, so a run would decode "
                       "text that is not this model (" +
                       (spec->why && spec->why[0] ? spec->why : "no reason given") +
                       "). The file loaded and its tensors were read; "
                       "generating from it is refused.";
            model_.unload();
            be_ = nullptr;
            return false;
        }
    }
    const ModelConfig &predict_mc = model_.cfg();
    const bool moe_predict_requested = moe_predict_on();
    const bool moe_predict_arch_ok = expert_prediction_model_supported(predict_mc);
    moe_predictor_.configure(predict_mc.n_layer, predict_mc.n_expert,
                             moe_predict_history());
    if (moe_predict_requested && !moe_predict_arch_ok)
        KRK_INFO("expert predictor: disabled for %s (model is not a supported MoE); "
                 "native routing remains unchanged", predict_mc.arch.c_str());

    // This has to happen AFTER the load, and that is not a style choice: the
    // geometry it reads is the loader's, and the same block sitting above the
    // load saw an empty ModelConfig -- `model_.cfg().hc` was false on every
    // file, so the refusal this position replaced was dead code that said
    // "generation is refused" while generation ran.
    // qwen4exp's residual is `hc` lanes, not one vector, and the stack below
    // runs it: every sub-block mixes the lanes in and scatters its output back
    // (hc_enter / hc_leave). What is still missing on this family is everything
    // AROUND the residual -- the PLE n-gram module and the QSA indexer -- and a
    // file carrying either would decode fluent text that is not this model, so
    // it is refused by name rather than run without the block it declares.
    // k2-horizon's grouped norm is a reduction a backend has to implement; one
    // that does not would apply the identity on that op and run a model whose
    // every norm site is missing, which decodes. Refuse by name instead, the
    // same bargain hc_supported() makes below.
    if (model_.cfg().n_norm_groups > 1 && !be_->group_norm_supported()) {
        if (err)
            *err = "arch '" + model_.cfg().arch +
                   "': attention.group_norm_groups is " +
                   std::to_string(model_.cfg().n_norm_groups) +
                   ", and this backend has no grouped-RMS-norm op, so every "
                   "norm site would run ungrouped";
        return false;
    }
    if (model_.cfg().n_value_expert > 0 &&
        model_.cfg().n_value_expert_used > model_.cfg().n_value_expert) {
        if (err)
            *err = "arch '" + model_.cfg().arch +
                   "': attention.value_expert_used_count exceeds "
                   "value_expert_count";
        return false;
    }
    if (model_.cfg().hc) {
        const ModelConfig &hcc = model_.cfg();
        // V4.1's mixer is not qwen4exp's, so its capability is asked
        // separately: a backend with hc_mix but no sinkhorn split would mix
        // the lanes with the wrong function and still decode.
        if (hcc.hc_sinkhorn) {
            if (!be_->hc_sinkhorn_supported()) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': this backend has no hyper-connection SINKHORN "
                           "kernels, so the V4.1 lane stack cannot be run on it";
                return false;
            }
            // `low_rank` is the MIXER's width here -- (2+hc)*hc, the fn
            // projection's output -- not a bottleneck. Nothing downsamples.
            hc_ = HcGeometry{hcc.n_embd, hcc.hc_n, (2 + hcc.hc_n) * hcc.hc_n};
        } else {
        if (!be_->hc_supported()) {
            if (err)
                *err = "arch '" + hcc.arch +
                       "': this backend has no hyper-connection mix/combine "
                       "kernels, so the lane stack cannot be run on it";
            return false;
        }
        hc_ = HcGeometry{hcc.n_embd, hcc.hc_n, hcc.hc_low_rank};
        // The bottleneck is the UP-projection's K: [hc*n_embd <- low_rank]
        // reads low_rank columns per row. A backend that addresses K in
        // fixed chunks must be asked here, at init, because the alternative
        // is HIP's gemm aborting two rows into the first prefill -- measured:
        // the tiny qwen4exp fixture (low_rank 8, a multiple of nothing) died
        // with 0xC0000409 after printing its error, where `--cpu` ran it.
        if (!be_->gemm_k_ok(hc_.low_rank)) {
            if (err)
                *err = "arch '" + hcc.arch +
                       "': hyper_connection.low_rank is " +
                       std::to_string(hcc.hc_low_rank) +
                       " and this backend's GEMM needs a multiple of 32 "
                       "there (the bottleneck is that GEMM's contraction)";
            return false;
        }
        if (!hc_.valid()) {
            if (err)
                *err = "arch '" + hcc.arch + "': hyper-connection geometry is "
                       "incomplete (count " + std::to_string(hcc.hc_n) +
                       ", low_rank " + std::to_string(hcc.hc_low_rank) +
                       ", n_embd " + std::to_string(hcc.n_embd) + ")";
            return false;
        }
        // The per-layer embedding block. It writes into the lane stack, so it
        // only makes sense on this family, and its gathered row has to be as
        // wide as the stream its key projection reads.
        }   // end of the qwen4exp mixer's checks
        if (hcc.ple && !hcc.hc_sinkhorn) {
            ple_ = PleConfig{};
            ple_.ngram = hcc.ple_ngram;
            ple_.heads_per_ngram = hcc.ple_heads_per_ngram;
            ple_.head_dim = hcc.ple_head_dim;
            ple_.conv_kernel = hcc.ple_conv_kernel;
            ple_.eos_token = hcc.ple_eos_token;
            ple_.head_offsets = hcc.ple_head_offsets;
            ple_.head_vocab_sizes = hcc.ple_head_vocab_sizes;
            ple_.multipliers = hcc.ple_multipliers;
            if (!ple_.valid()) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': the per-layer embedding geometry is incomplete "
                           "(needs ngram>=2, heads_per_ngram, head_dim, "
                           "conv_kernel and one offset/vocab per head)";
                return false;
            }
            // The gathered row is n_heads * head_dim wide, and that is
            // n_embd -- NOT the lane stack. `ple_key` widens it to hc*n_embd
            // and `ple_value` keeps it narrow, exactly as the reference's two
            // projections do: 16 heads of 160 is 2560 on the REAP file, the
            // embedding width, while the streams it feeds are 10240.
            if (ple_.row_width() != hcc.n_embd) {
                if (err)
                    *err = "arch '" + hcc.arch + "': the PLE rows are " +
                           std::to_string(ple_.row_width()) + " wide (" +
                           std::to_string(ple_.n_heads()) + " heads of " +
                           std::to_string(ple_.head_dim) +
                           ") but the embedding is " +
                           std::to_string(hcc.n_embd) +
                           " wide; the key projection cannot read that";
                return false;
            }
            if (!model_.ple_table() || model_.ple_row_bytes() == 0) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': PLE is declared but per_layer_token_embd is not "
                           "in the file";
                return false;
            }
            ple_hist_ = static_cast<i64>(ple_.conv_kernel - 1) * ple_.ngram;
            ple_active_ = true;
            KRK_INFO("per-layer embedding: %d head(s) of %d, %d-gram window, "
                     "conv %d dilated by %d (%lld row(s) of history)",
                     ple_.n_heads(), ple_.head_dim, ple_.ngram,
                     ple_.conv_kernel, ple_.ngram,
                     static_cast<long long>(ple_hist_));
        }
        // ---- Engram (V4.1) -----------------------------------------------
        // The hash needs the table and a token map; both are absent on a file
        // that declares no Engram layers, and on one that declares them a
        // missing map would hash the raw token ids (a different function that
        // still produces in-range rows).
        if (hcc.ple && hcc.hc_sinkhorn) {
            if (!model_.ple_table() || model_.ple_row_bytes() == 0) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': Engram layers are declared but "
                           "per_layer_tok_embd is not in the file";
                return false;
            }
            if (hcc.engram_token_map.empty()) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': Engram is declared but engram.token_map is "
                           "missing, so the hashes would not match the "
                           "checkpoint's";
                return false;
            }
            for (i32 v : hcc.ple_layers)
                if (v < 0 || v >= static_cast<i32>(model_.layers().size())) {
                    if (err)
                        *err = "arch '" + hcc.arch + "': ple.layers names layer " +
                               std::to_string(v) + " but the file has " +
                               std::to_string(model_.layers().size());
                    return false;
                }
            const i64 n_cols = (hcc.ple_ngram - 1) * hcc.ple_heads_per_ngram;
            const i64 head_dim = hcc.ple_head_dim;
            if (head_dim * n_cols <= 0) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': Engram geometry is empty (ngram " +
                           std::to_string(hcc.ple_ngram) + ", heads/ngram " +
                           std::to_string(hcc.ple_heads_per_ngram) +
                           ", head_dim " + std::to_string(head_dim) + ")";
                return false;
            }
            if (model_.ple_row_bytes() != static_cast<u64>(head_dim) * 4 &&
                model_.ple_row_bytes() != static_cast<u64>(head_dim) * 2) {
                // The gather reads one head's row at a time and dequantizes it
                // through the file's dtype; a row has to BE one head's width or
                // the stride walks off the row.
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': per_layer_tok_embd rows are " +
                           std::to_string(model_.ple_row_bytes()) +
                           " bytes, not one head of " + std::to_string(head_dim) +
                           " elements in f32/f16";
                return false;
            }
            KRK_INFO("engram: %zu layer(s), %lld column(s) of %lld per token, "
                     "%d-gram hash",
                     hcc.ple_layers.size(), static_cast<long long>(n_cols),
                     static_cast<long long>(head_dim), hcc.ple_ngram);
        }
        if (hcc.qsa) {
            // The indexer SELECTS: each full-attention layer pools its raw
            // indexer keys into blocks of `ratio` cells, scores every COMPLETE
            // block, keeps the best `top_k + ratio - 1` cells and attends
            // densely over those -- see qsa_select_block(). Below that cell
            // budget nothing is dropped and the layer runs plain dense causal
            // attention, which is what the reference does there too; that
            // shortcut is why a short-context run is unaffected by any of this
            // and why the tests have to build a context longer than the budget
            // to exercise the selection at all.
            i32 n_idx = 0;
            i32 ratio_max = 0;
            for (size_t i = 0; i < hcc.qsa_ratio.size(); i++) {
                if (hcc.qsa_ratio[i] > 0) {
                    if (hcc.qsa_ratio[i] > ratio_max) ratio_max = hcc.qsa_ratio[i];
                    n_idx++;
                }
            }
            if (n_idx > 0 &&
                (hcc.idx_head_dim <= 0 || hcc.idx_n_head <= 0 || hcc.idx_top_k <= 0)) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': the indexer geometry is incomplete (head_count " +
                           std::to_string(hcc.idx_n_head) + ", key_length " +
                           std::to_string(hcc.idx_head_dim) + ", top_k " +
                           std::to_string(hcc.idx_top_k) + ")";
                return false;
            }
            // KRK_QSA=0 is the A/B arm: dense over everything, which is a
            // SUPERSET of the selection and therefore the model's own attention
            // arithmetic -- the arm says what the block costs and what it
            // changes, and it is the one a coherence check can run against.
            static const bool qsa_on = [] {
                const char *e = std::getenv("KRK_QSA");
                return !(e && e[0] == '0');
            }();
            // The gather is how a per-row cell list is expressed: attention()
            // has no mask, so the chosen cells are compacted and the ordinary
            // causal kernel runs over them. A backend without it would run the
            // default no-op and attend over whatever the compact buffer held.
            if (n_idx > 0 && qsa_on && !be_->gather_rows_supported()) {
                if (err)
                    *err = "arch '" + hcc.arch +
                           "': the QSA indexer needs a row gather (Backend::"
                           "gather_rows) and this backend does not implement it";
                return false;
            }
            qsa_active_ = n_idx > 0 && qsa_on;
            qsa_dim_ = hcc.idx_head_dim;
            qsa_heads_ = hcc.idx_n_head;
            qsa_top_k_ = hcc.idx_top_k;
            qsa_ratio_max_ = ratio_max;
            qsa_eps_ = hcc.rms_eps;
            if (n_idx > 0) {
                KRK_INFO("qsa: %d layer(s) carry an indexer (%d head(s) of %d, "
                         "ratio %d, top_k %d cells) -- %s",
                         n_idx, hcc.idx_n_head, hcc.idx_head_dim, ratio_max,
                         hcc.idx_top_k,
                         qsa_active_ ? "block selection on"
                                     : "SELECTION OFF (KRK_QSA=0): dense over "
                                       "every key, a superset of what the "
                                       "indexer picks");
            }
        }
        hc_active_ = true;
        KRK_INFO("hyper-connections: %d lane(s) of %d through %d, residual %d wide",
                 hc_.hc, hc_.n_embd, hc_.low_rank, hc_.hc * hc_.n_embd);
    }
    // KRK_DEQUANT_PROBE: device-vs-host dequant of one named row, before the
    // first forward pass, so a wrong row read is visible in one run. Nothing
    // here runs unless the variable is set.
    if (std::getenv("KRK_DEQUANT_PROBE")) run_dequant_probe(*be_, cfg.model_path);
    const ModelConfig &mc = model_.cfg();
    n_embd_ = mc.n_embd;
    n_ff_ = mc.n_ff;
    q_dim_ = model_.q_dim();
    kv_dim_ = model_.kv_dim();
    // Qwen3.5's attending layers carry a per-head output gate in the same
    // projection as the query, so the buffer is twice the head width until the
    // engine splits the two apart.
    // The arch table owns this rule; the engine only mirrors it (see
    // include/krk/arch.hpp for why it lives in one place).
    // From the RESOLVED spec's own flag, not from arch_spec(mc.arch): an arch
    // that is an alias (qwen38 -> qwen4exp, decided from the file's tensors)
    // carries a different spec than its name does, and asking by name here
    // silently halved the query width on a file whose q row is packed. The
    // loader had already read the packed row, so the two disagreed and the
    // graph decoded a different model.
    q_proj_ = mc.q_output_gate ? 2 * q_dim_ : q_dim_;
    conv_dim_ = model_.conv_dim();
    value_dim_ = model_.value_dim();
    n_vocab_ = mc.n_vocab;
    kv_cap_ = std::min<i64>(cfg_.n_ctx > 0 ? cfg_.n_ctx : mc.n_ctx_train,
                            mc.n_ctx_train > 0 ? mc.n_ctx_train : 32768);
    if (kv_cap_ <= 0) kv_cap_ = 2048;

    chunk_ = std::max<i32>(1, std::min<i32>(cfg_.prefill_chunk, 1024));
    // ---- the diffusion sequence length ------------------------------------
    // A masked-diffusion pass cannot be chunked: the whole sequence is denoised
    // at once and every position reads every other, so a chunk boundary would
    // put tokens outside the model's view. The workspace is therefore sized for
    // the LONGEST pass this engine will run, and the pass length is bounded here
    // rather than at the call site -- a request that needs more is refused with
    // a number instead of overrunning an activation buffer (measured: a
    // 9-token pass against an 8-row workspace corrupted the heap and died in
    // free()).
    if (mc.diffusion) {
        diffuse_max_tokens_ = std::min<i32>(
            static_cast<i32>(std::min<i64>(kv_cap_, 4096)),
            std::max<i32>(chunk_, static_cast<i32>(cfg_.n_ctx)));
        if (diffuse_max_tokens_ <= 0)
            diffuse_max_tokens_ = std::max<i32>(1, chunk_);
        chunk_ = diffuse_max_tokens_;
        KRK_INFO("diffusion: the whole sequence is one pass, so the workspace is "
                 "sized for %d token(s) and a longer request is refused",
                 diffuse_max_tokens_);
    }

    // ---- multi-head latent attention: what this build can actually run ----
    //
    // The arch table says the FAMILY has an MLA implementation here (deepseek2);
    // this says the file's particular LAYOUT does, because MLA ships in three
    // shapes and they are three different graphs. Only the legacy lite layout
    // (V2-Lite: attn_q + attn_kv_a_mqa + attn_kv_b) runs today, and the other
    // two are refused by name -- a file that loaded and then generated fluent
    // text out of the wrong graph is the failure this engine must not produce.
    if (mc.mla) {
        auto refuse_mla = [&](const std::string &m) {
            if (err) *err = m;
            model_.unload();
            be_ = nullptr;
            return false;
        };
        // All THREE layouts run. The two lite ones differ only in how the
        // per-head K/V up-projection is STORED -- one fused table
        // (`attn_kv_b`) or two three-dimensional ones (`attn_k_b` +
        // `attn_v_b`) -- and by linearity the absorbed and unabsorbed
        // formulations compute the same score and the same weighted sum, so
        // the loader composes the fused staging from either and one validated
        // kernel serves both. The shared-latent layout (V4) is a different
        // graph and has its own forward.
        if (mc.mla_layout == MlaLayout::None) {
            return refuse_mla(
                "MLA: the model config lists no layout, so there is no "
                "attention to run");
        }
        const bool mla_shared = mc.mla_layout == MlaLayout::Shared;
        if (mla_shared && mc.dsv4_o_groups <= 0)
            return refuse_mla(
                "MLA layout 'shared-latent' needs attention.output_group_count, "
                "and the file carries none -- the grouped output projection is "
                "not optional on this layout");
        if (mla_shared && !mc.mla_q_lora)
            return refuse_mla(
                "MLA layout 'shared-latent' needs a query trunk "
                "(attention.q_lora_rank), and this file's is absent");
        // A sink is a softmax DENOMINATOR term; dropping it silently would leave
        // every head's distribution unnormalized by exactly that mass, which is
        // fluent text that is not this model. The ops take it as HOST memory and
        // the loader uploads it to the device, so until that has a host copy the
        // honest answer is a refusal rather than a pass-through pointer.
        // A non-zero compress ratio is a layer whose attention reads the CSA/HCA
        // compressor's pooled keys, which this build does not implement. Running
        // the dense core over such a layer would attend the raw prefix instead --
        // a plausible answer from a graph the file does not describe, which is
        // the one failure mode this engine refuses rather than approximates.
        // A non-zero compress ratio is a layer whose attention reads the CSA/HCA
        // compressor's pooled keys as well as the raw window. That path runs
        // (Engine::mla_shared_forward_compressed), but the compressed attention
        // op is a REFERENCE implementation: it is host f32 arithmetic over two
        // row families and only the CPU backend says it can run it. A device
        // that cannot must refuse rather than attend the raw prefix -- a
        // plausible answer from a graph the file does not describe.
        if (mla_shared) {
            bool any_comp = false;
            for (size_t i = 0; i < mc.dsv4_ratio.size(); i++)
                if (mc.dsv4_ratio[i] != 0) any_comp = true;
            if (any_comp && !be_->mla_shared_compressed_supported())
                return refuse_mla(
                    "MLA layout 'shared-latent': the file declares non-zero "
                    "compress ratios and this backend has no compressed "
                    "attention (it runs the two row families on hardware only "
                    "the CPU reference implements)");
        }
        if (mla_shared ? !be_->mla_shared_supported() : !be_->mla_supported()) {
            return refuse_mla(
                std::string("the selected backend has no ") +
                (mla_shared ? "shared-latent" : "legacy") +
                " multi-head latent attention, and an MLA layer that silently "
                "runs no attention decodes text that is not this model");
        }
        // The kernel stages every row's scores in dynamic shared memory, which
        // is why the context length is a CAPABILITY here and not a preference:
        // past 48 KiB a launch needs an explicit opt-in attribute this build
        // does not set, and a launch that silently truncated the request would
        // read scores out of whatever follows them.
        // The LITE kernel stages its score row plus the latent and the rotated
        // key; the shared-latent kernel stages scores only, so this bound is the
        // lite layout's. Applying it to a V4 file would refuse a context the
        // shared path fits in.
        const size_t lds = mla_shared
                               ? 0u
                               : static_cast<size_t>(2 * mc.mla_kv_lora +
                                                     mc.mla_qk_rope + kv_cap_) *
                                     sizeof(f32);
        if (lds > 48u * 1024u) {
            if (err)
                *err = "MLA needs " + std::to_string(lds / 1024u) +
                       " KiB of shared memory per block at --ctx " +
                       std::to_string(static_cast<long long>(kv_cap_)) +
                       " (the score row plus the latent and the rotated key), "
                       "and the ceiling without an explicit opt-in is 48 KiB; "
                       "lower --ctx to 8192 or less";
            model_.unload();
            be_ = nullptr;
            return false;
        }
    }

    // Its own row comes from Tokenizer::load ("tokenizer.load"), so every
    // caller is measured rather than this one path.
    if (!tok_.load(model_.gguf(), err)) {
        model_.unload();
        be_ = nullptr;
        return false;
    }
    // The tokenizer's piece list and the embedding table are two independent
    // statements about the vocabulary, and nothing above compares them.
    // `cfg_.n_vocab` is the TABLE's row count (Model::load takes it from
    // token_embd.weight's ne[1]); `tok_.n_vocab()` is the PIECE count. When the
    // pieces outnumber the rows, the tokenizer can hand `embed()` an id that has
    // no row, and the row it read instead was whatever the heap held past the
    // table: measured on a 260-row table with 261 pieces, AddressSanitizer
    // reported a READ of size 2 exactly 0 bytes to the right of the 16640-byte
    // token_embd region (quant.cpp dequant_row <- backend_cpu embed <-
    // forward_core). The visible symptom was not a crash: the same file, same
    // prompt, run twice in one process, decoded DIFFERENT tokens, because the
    // garbage row changed with the heap.
    //
    // WHICH fix, and why not the obvious one. Refusing the file is what this
    // engine does for a tensor set it cannot honour, and it is wrong here: 41
    // groups of this suite's own fixtures declare a 261-piece tokenizer over a
    // table of 64 or 260 rows (add_spm_vocab always writes 261 entries before
    // padding), and every one of them runs correctly, because sampling draws
    // from the table's row count and never produces an id in the gap -- only a
    // PROMPT token can. Refusing them would reject files that are fine in
    // practice. So the id is bounds-checked where it is used (CpuBackend::embed
    // writes a zero row instead of reading past the table), and the mismatch is
    // reported here, once, naming both counts.
    if (tok_.n_vocab() > model_.cfg().n_vocab) {
        vocab_note_ = "the tokenizer declares " +
                      std::to_string(tok_.n_vocab()) +
                      " pieces but token_embd.weight has only " +
                      std::to_string(model_.cfg().n_vocab) +
                      " rows; a prompt token past the table gets a zero "
                      "embedding row";
        KRK_WARN("%s", vocab_note_.c_str());
    }

    const size_t as = be_->act_size();
    const i64 C = chunk_;
    // The ledger's baseline: what the device already had taken before this
    // engine allocated anything. Without it, `unaccounted` at the end of init is
    // ambiguous between "a buffer we forgot" and "the desktop and the driver",
    // and that ambiguity is how an inferred 3 GiB of scratch survived.
    if (be_->has_device_memory() && be_->device_total_bytes() > 0) {
        const size_t tot = be_->device_total_bytes();
        const size_t fr = be_->device_free_bytes();
        KRK_INFO("vram at start: %.0f MiB total, %.0f MiB free before this engine "
                 "allocates (%.0f MiB already held by the driver/other processes)",
                 static_cast<f64>(tot) / 1048576.0,
                 static_cast<f64>(fr) / 1048576.0,
                 static_cast<f64>(tot > fr ? tot - fr : 0) / 1048576.0);
    }

    // Counted, not guessed: every workspace below goes through this lambda, and
    // the ledger printed at the end of init needs their measured sum (see
    // ws_bytes_). The alternative -- subtracting weights, KV, experts and free
    // from the card -- is where the "scratch is ~3 GiB" inference came from.
    ws_bytes_ = 0;
    auto alloc = [&](size_t bytes) {
        ws_bytes_ += bytes;
        return be_->alloc(bytes);
    };
    auto fail_alloc = [&](const char *name) {
        if (err)
            *err = std::string("backend allocation failed for ") + name +
                   "; lower --ctx/--chunk or reduce model size";
        shutdown();
        return false;
    };
    auto check_alloc = [&](void *p, const char *name) {
        if (p) return true;
        fail_alloc(name);
        return false;
    };
    ws_x_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    if (!check_alloc(ws_x_, "ws_x")) return false;
    // The residual stream's per-token storage exponents (Backend::add_residual).
    // Zeroed here and again at the start of every forward pass: an all-zero
    // array is exactly the old behavior, so a model that never leaves f16 pays
    // one i32 per row and nothing else.
    ws_rexp_ = alloc(static_cast<size_t>(C) * sizeof(i32));
    if (!check_alloc(ws_rexp_, "ws_rexp")) return false;
    be_->fill0(ws_rexp_, static_cast<size_t>(C) * sizeof(i32));
    // QSA workspaces. The projections are per-token rows, so they are
    // chunk-sized; the compact K/V pair is as long as the widest cell budget
    // (`top_k + ratio - 1` cells) and is only touched past that budget.
    if (qsa_active_) {
        const i64 dim = qsa_dim_;
        const i64 nh = qsa_heads_;
        ws_qsa_k_ = alloc(static_cast<size_t>(C * dim) * as);
        if (!check_alloc(ws_qsa_k_, "ws_qsa_k")) return false;
        ws_qsa_q_ = alloc(static_cast<size_t>(C * nh * dim) * as);
        if (!check_alloc(ws_qsa_q_, "ws_qsa_q")) return false;
        i64 ncells = static_cast<i64>(qsa_top_k_) + qsa_ratio_max_ - 1;
        ncells = std::min<i64>(ncells, kv_cap_);
        if (ncells <= 0) ncells = 1;
        ws_qsa_kc_ = alloc(static_cast<size_t>(ncells * kv_dim_) * as);
        if (!check_alloc(ws_qsa_kc_, "ws_qsa_kc")) return false;
        ws_qsa_vc_ = alloc(static_cast<size_t>(ncells * kv_dim_) * as);
        if (!check_alloc(ws_qsa_vc_, "ws_qsa_vc")) return false;
        qsa_state_.assign(static_cast<size_t>(mc.n_layer), QsaState{});
        qsa_rows_.assign(static_cast<size_t>(C), std::vector<i64>{});
        i64 idx_layers = 0;
        i64 idx_bytes = 0;
        for (i32 l = 0; l < mc.n_layer; l++) {
            const auto &L = model_.layers()[static_cast<size_t>(l)];
            if (L.qsa_ratio <= 0) continue;
            QsaState &st = qsa_state_[static_cast<size_t>(l)];
            st.keys.assign(static_cast<size_t>(kv_cap_ * dim), 0.0f);
            st.pooled.assign(static_cast<size_t>(kv_cap_ / std::max(1, L.qsa_ratio) + 1) * dim,
                             0.0f);
            st.score.assign(static_cast<size_t>(C) *
                                static_cast<size_t>(kv_cap_ / std::max(1, L.qsa_ratio) + 1),
                            0.0f);
            idx_layers++;
            idx_bytes += static_cast<i64>(st.keys.size()) * 4;
        }
        if (idx_bytes > 0)
            KRK_INFO("qsa indexer: %lld layer(s) keep their raw keys on the host "
                     "(%lld MiB), %lld compact cells per row at the budget",
                     static_cast<long long>(idx_layers),
                     static_cast<long long>(idx_bytes >> 20),
                     static_cast<long long>(ncells));
    }
    ws_fexp_ = alloc(static_cast<size_t>(C) * sizeof(i32));
    if (!check_alloc(ws_fexp_, "ws_fexp")) return false;
    be_->fill0(ws_fexp_, static_cast<size_t>(C) * sizeof(i32));
    // The gated FFN intermediate's per-row store scale (Backend::silu_mul's
    // row-scaled form). Filled with 1.0 rather than zeroed: a layer that never
    // writes one must read as unscaled, and the trace multiplies by this array.
    ws_fscl_ = alloc(static_cast<size_t>(C) * sizeof(f32));
    if (!check_alloc(ws_fscl_, "ws_fscl")) return false;
    {
        std::vector<f32> ones(static_cast<size_t>(C), 1.0f);
        be_->upload(ws_fscl_, ones.data(), static_cast<size_t>(C) * sizeof(f32),
                    0);
    }

    ws_xn_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    if (!check_alloc(ws_xn_, "ws_xn")) return false;
    ws_x2_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    if (!check_alloc(ws_x2_, "ws_x2")) return false;
    // ---- gemma4 per-layer input embeddings ----------------------------
    // One [n_layer * n_embd_per_layer] row per token, twice: the gathered
    // token table and the projection that is added to it. This is the
    // width the model projection's output actually has -- not n_embd --
    // because ONE matrix produces every layer's slice at once, which is
    // why the reference computes it before the first layer rather than
    // inside each one.
    if (mc.n_embd_per_layer > 0 && model_.ple_tok_embd.present()) {
        ppl_npl_ = mc.n_embd_per_layer;
        ppl_width_ = static_cast<i64>(mc.n_layer) * ppl_npl_;
        ws_ppl_ = alloc(static_cast<size_t>(C * ppl_width_) * as);
        if (!check_alloc(ws_ppl_, "ws_ppl")) return false;
        ws_ppl_in_ = alloc(static_cast<size_t>(C * ppl_width_) * as);
        if (!check_alloc(ws_ppl_in_, "ws_ppl_in")) return false;
        ws_ppl_g_ = alloc(static_cast<size_t>(C * ppl_npl_) * as);
        if (!check_alloc(ws_ppl_g_, "ws_ppl_g")) return false;
        ws_ppl_o_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        if (!check_alloc(ws_ppl_o_, "ws_ppl_o")) return false;
        KRK_INFO("per-layer embd: [chunk %lld x %lld] table, %lld per layer",
                 static_cast<long long>(C), static_cast<long long>(ppl_width_),
                 static_cast<long long>(ppl_npl_));
    }

    // Falcon's second parallel norm: its two arms normalize the SAME residual
    // row, and the attention arm's copy cannot be reused because the FFN's must
    // be taken before the attention residual lands in ws_x_. Only the family
    // that has two different norms allocates it -- on Falcon-7B `attn_norm_2`
    // is absent and the arms share weights, so this is null and unused.
    if (mc.parallel_two_norm) {
        ws_xn2_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        if (!check_alloc(ws_xn2_, "ws_xn2")) return false;
    }
    // GPT-2 only: the fused attn_qkv output, [chunk, 3*n_embd] f32. Allocated
    // by the same rule as every other workspace (chunked, backend activation
    // size) and only when the file is gpt2-shaped, so no other family pays for
    // a buffer its graph never touches.
    if (mc.gpt2_style || mc.fused_qkv) {
        // gpt2's row is 3*n_embd; Phi's is n_embd + 2*n_head_kv*head_dim. Both
        // collapse to the same expression when n_head_kv is n_head, so the
        // fused flag alone decides which one this is.
        const size_t kvw =
            static_cast<size_t>(mc.n_head_kv) * static_cast<size_t>(mc.head_dim);
        const size_t row = mc.fused_qkv
                               ? static_cast<size_t>(n_embd_) + 2 * kvw
                               : static_cast<size_t>(3) * static_cast<size_t>(n_embd_);
        ws_qkv_fused_ = alloc(static_cast<size_t>(C) * row * as);
        if (!check_alloc(ws_qkv_fused_, "ws_qkv_fused")) return false;
    }
    // NVFP4 sidecars: the staging row a scaled weight's INPUT is copied and
    // scaled into before its projection (see gemm_w for why the input and not
    // the result). Sized by the widest scaled contraction in the file, and not
    // allocated at all when the file carries no sidecar -- which is every
    // non-NVFP4 checkpoint.
    if (const i64 gs_in = model_.max_scaled_in(); gs_in > 0) {
        ws_gs_ = alloc(static_cast<size_t>(C * gs_in) * as);
        if (!check_alloc(ws_gs_, "ws_gs")) return false;
    }
    // qwen4exp's lane stack: the wide residual, the normed copy, the mixer's up
    // projection, its bottleneck, its per-lane injection, and the mixed row.
    // Null on every other family, so nothing is paid for them.
    if (hc_active_) {
        const i64 wide = hc_.lane_width();
        auto halloc = [&](void *&p, size_t bytes, const char *name) {
            p = alloc(bytes);
            return check_alloc(p, name);
        };
        if (!halloc(ws_hc_, static_cast<size_t>(C * wide) * as, "ws_hc"))
            return false;
        if (!halloc(ws_hc_n_, static_cast<size_t>(C * wide) * as, "ws_hc_n"))
            return false;
        if (model_.cfg().hc_sinkhorn) {
            // V4.1: the mixer's output, its three pieces, the block's own
            // output and the scatattered stack. No bottleneck, no per-lane
            // injection, no gamma -- those are the other mixer's buffers.
            const i64 hc = hc_.hc;
            const i64 mix_dim = (2 + hc) * hc;
            if (!halloc(ws_hc_fn_, static_cast<size_t>(C * mix_dim) * as,
                        "ws_hc_fn"))
                return false;
            if (!halloc(ws_hc_pre_, static_cast<size_t>(C * hc) * 4, "ws_hc_pre"))
                return false;
            if (!halloc(ws_hc_post_, static_cast<size_t>(C * hc) * 4, "ws_hc_post"))
                return false;
            if (!halloc(ws_hc_comb_, static_cast<size_t>(C * hc * hc) * 4,
                        "ws_hc_comb"))
                return false;
            if (!halloc(ws_hc_out_, static_cast<size_t>(C * n_embd_) * as,
                        "ws_hc_out"))
                return false;
            if (!halloc(ws_hc_new_, static_cast<size_t>(C * wide) * as,
                        "ws_hc_new"))
                return false;
            if (!halloc(ws_hc_pre_attn_, static_cast<size_t>(C * hc) * 4,
                        "ws_hc_pre_attn"))
                return false;
            if (!halloc(ws_hc_pre_prev_, static_cast<size_t>(C * hc) * 4,
                        "ws_hc_pre_prev"))
                return false;
            if (!model_.cfg().hc_no_head) {
                // The head's mixer emits just the pre vector [hc].
                if (!halloc(ws_hc_head_, static_cast<size_t>(C * hc) * 4,
                            "ws_hc_head"))
                    return false;
            }
            // The identity pre-mix, allocated either way: the lagged graph uses
            // it at the first layer, and that is exactly the graph with no head.
            if (!halloc(ws_hc_one_, static_cast<size_t>(C * hc) * 4,
                        "ws_hc_one"))
                return false;
            // The saved collapse (the sub-block's input before its norm), which
            // hc_sk_leave() subtracts back out of the block's output.
            if (!halloc(ws_hc_mix_, static_cast<size_t>(C * n_embd_) * as,
                        "ws_hc_mix"))
                return false;
            // Engram's gathered rows and the per-layer hash ids. The ids are a
            // host vector because engram_hash is a host function (the hash is
            // integer arithmetic over the token window, and the reference
            // computes it on the CPU too).
            if (model_.cfg().ple) {
                const i64 n_cols =
                    (model_.cfg().ple_ngram - 1) * model_.cfg().ple_heads_per_ngram;
                if (!halloc(ws_eng_kv_,
                            static_cast<size_t>(C * n_embd_ * (hc + 1)) * as,
                            "ws_eng_kv"))
                    return false;
                if (!halloc(ws_eng_rows_,
                            static_cast<size_t>(C * n_cols *
                                                model_.cfg().ple_head_dim) * as,
                            "ws_eng_rows"))
                    return false;
                eng_ids_.resize(static_cast<size_t>(C) * n_cols);
            }
        } else {
        if (!halloc(ws_hc_g_, static_cast<size_t>(C * wide) * as, "ws_hc_g"))
            return false;
        if (!halloc(ws_hc_lo_, static_cast<size_t>(C * hc_.low_rank) * as,
                    "ws_hc_lo"))
            return false;
        if (!halloc(ws_hc_inj_, static_cast<size_t>(C * hc_.hc) * as,
                    "ws_hc_inj"))
            return false;
        if (!halloc(ws_hc_mix_, static_cast<size_t>(C * n_embd_) * as,
                    "ws_hc_mix"))
            return false;
        }
        // The PLE block's buffers, allocated here too because they are only
        // reachable when the lane stack is.
        if (ple_active_) {
            const i64 hc = hc_.hc;
            if (!halloc(ws_ple_emb_, static_cast<size_t>(C * n_embd_) * as,
                        "ws_ple_emb"))
                return false;
            if (!halloc(ws_ple_key_, static_cast<size_t>(C * wide) * as,
                        "ws_ple_key"))
                return false;
            if (!halloc(ws_ple_qry_, static_cast<size_t>(C * wide) * as,
                        "ws_ple_qry"))
                return false;
            if (!halloc(ws_ple_gated_, static_cast<size_t>(C * wide) * as,
                        "ws_ple_gated"))
                return false;
            if (!halloc(ws_ple_val_, static_cast<size_t>(C * n_embd_) * as,
                        "ws_ple_val"))
                return false;
            if (!halloc(ws_ple_norm_, static_cast<size_t>(C * wide) * as,
                        "ws_ple_norm"))
                return false;
            if (!halloc(ws_ple_gate_, static_cast<size_t>(C * hc) * as,
                        "ws_ple_gate"))
                return false;
            if (!halloc(ws_ple_pad_,
                        static_cast<size_t>((ple_hist_ + C) * wide) * as,
                        "ws_ple_pad"))
                return false;
            if (!halloc(ws_ple_conv_, static_cast<size_t>(C * wide) * as,
                        "ws_ple_conv"))
                return false;
            // One conv history per layer, even on the layers that do not run
            // PLE: indexing by layer is worth the (small) waste.
            const i64 hist_bytes =
                static_cast<i64>(mc.n_layer) * ple_hist_ * wide * as;
            if (hist_bytes > 0) {
                ple_state_ = alloc(static_cast<size_t>(hist_bytes));
                if (!check_alloc(ple_state_, "ple_state")) return false;
                be_->fill0(ple_state_, static_cast<size_t>(hist_bytes));
            }
            ple_rows_.assign(static_cast<size_t>(C) * ple_.n_heads(), 0);
            ple_emb_.assign(static_cast<size_t>(C) * n_embd_, 0.0f);
        }
    }
    // Qwen3.5 packs a per-head output gate behind the query, so its query
    // projection is twice the query width and the split has to gather out of
    // it. The split reads a wider row than it writes, which means an in-place
    // call has head `h` writing over rows that other heads have not read yet:
    // block order decides the result. Keep the projection in its own buffer.
    ws_q_ = alloc(static_cast<size_t>(C * q_dim_) * as);
    if (!check_alloc(ws_q_, "ws_q")) return false;
    if (q_proj_ != q_dim_) {
        ws_qpack_ = alloc(static_cast<size_t>(C * q_proj_) * as);
        if (!check_alloc(ws_qpack_, "ws_qpack")) return false;
    }
    ws_k_ = alloc(static_cast<size_t>(C * kv_dim_) * as);
    if (!check_alloc(ws_k_, "ws_k")) return false;
    ws_v_ = alloc(static_cast<size_t>(C * kv_dim_) * as);
    if (!check_alloc(ws_v_, "ws_v")) return false;
    ws_attn_ = alloc(static_cast<size_t>(C * q_dim_) * as);
    if (!check_alloc(ws_attn_, "ws_attn")) return false;
    // Two arches carry an attention output gate and they hold it differently:
    // Qwen3.5 packs it behind the query projection (only for its recurrent
    // stack), laguna projects it separately. Both need the buffer, and this is
    // the only place it is allocated.
    if (model_.is_recurrent() || mc.attn_gate) {
        ws_agate_ = alloc(static_cast<size_t>(C * q_dim_) * as);
        if (!check_alloc(ws_agate_, "ws_agate")) return false;
    }
    // MLA's compressed projection row (latent + decoupled key). Not ws_k_:
    // that buffer is n_kv x head_dim wide, which is what an MLA layer does NOT
    // project, and a shared buffer would be the wrong width on whichever of the
    // two shapes the stack has fewer of.
    if (mc.mla) {
        ws_mla_ = alloc(static_cast<size_t>(C) *
                        static_cast<size_t>(mc.mla_kv_lora + mc.mla_qk_rope) * as);
        if (!check_alloc(ws_mla_, "ws_mla")) return false;
        // The trunk's intermediate row. Sized by the file's own geometry, so a
        // V4 file with 8 groups of 1024 reserves what it asks for and nothing
        // more -- and reserved for BOTH layouts: the LITE path carries a query
        // trunk too (the V3 class and MiniCPM3), and gating this allocation on
        // the shared layout left `ws_mlaqn_` null on every file that had one.
        // That read as a working load and a SIGSEGV on the first token.
        if (mc.mla_q_lora > 0) {
            ws_mlaqn_ = alloc(static_cast<size_t>(C) *
                              static_cast<size_t>(mc.mla_q_lora) * as);
            if (!check_alloc(ws_mlaqn_, "ws_mlaqn")) return false;
        }
        if (mc.mla_layout == MlaLayout::Shared) {
            // f32, not `as`: see the member's note -- the grouped output reads it
            // and the host oracle it is compared against keeps f32 there.
            ws_mlaa_ = alloc(static_cast<size_t>(C) *
                             static_cast<size_t>(mc.n_head) *
                             static_cast<size_t>(mc.mla_kv_lora) * sizeof(f32));
            if (!check_alloc(ws_mlaa_, "ws_mlaa")) return false;
            const i64 gout = static_cast<i64>(mc.dsv4_o_groups) *
                             static_cast<i64>(mc.dsv4_o_lora);
            ws_mlay_ = alloc(static_cast<size_t>(C) *
                             static_cast<size_t>(gout) * sizeof(f32));
            if (!check_alloc(ws_mlay_, "ws_mlay")) return false;

            // ---- the compressed streams (CSA/HCA) ------------------------
            // Allocated only when the FILE compresses something: a dense-core V4
            // file pays nothing for a feature it does not use. The blocks are
            // sized by the context, not the chunk -- a pooled key lives for the
            // whole run, because it summarizes positions the window has already
            // passed.
            comp_ratio_.assign(static_cast<size_t>(mc.n_layer), 0);
            bool any_comp = false;
            for (i32 i = 0; i < mc.n_layer; i++) {
                const size_t k = static_cast<size_t>(i);
                const i32 r = k < mc.dsv4_ratio.size() ? mc.dsv4_ratio[k] : 0;
                comp_ratio_[k] = r;
                if (r != 0) any_comp = true;
            }
            if (any_comp) {
                const i64 hd = mc.mla_kv_lora;
                const i64 rope = mc.mla_qk_rope;
                comp_inv_freq_.assign(static_cast<size_t>(rope / 2), 0.0f);
                dsv4_comp_inv_freq(comp_inv_freq_.data(),
                                   static_cast<i32>(rope),
                                   mc.dsv4_compress_rope_base);
                comp_state_kv_.resize(static_cast<size_t>(mc.n_layer));
                comp_state_sc_.resize(static_cast<size_t>(mc.n_layer));
                comp_pooled_.resize(static_cast<size_t>(mc.n_layer));
                comp_kpos_.resize(static_cast<size_t>(mc.n_layer));
                idx_state_kv_.resize(static_cast<size_t>(mc.n_layer));
                idx_state_sc_.resize(static_cast<size_t>(mc.n_layer));
                idx_key_.resize(static_cast<size_t>(mc.n_layer));
                comp_nblocks_.assign(static_cast<size_t>(mc.n_layer), 0);
                i64 pool_rows = 0, state_w = 0, max_blocks = 0;
                for (i32 i = 0; i < mc.n_layer; i++) {
                    const size_t k = static_cast<size_t>(i);
                    const i32 r = comp_ratio_[k];
                    if (r == 0) continue;
                    const i64 coff = r == kDsv4CsaRatio ? 2 : 1;
                    const i64 blocks = static_cast<i64>(cfg.n_ctx) / r + 2;
                    comp_state_kv_[k].assign(
                        static_cast<size_t>(2 * r * coff * hd), 0.0f);
                    comp_state_sc_[k].assign(
                        static_cast<size_t>(2 * r * coff * hd), 0.0f);
                    comp_pooled_[k].assign(
                        static_cast<size_t>(blocks * hd), 0.0f);
                    comp_kpos_[k].assign(static_cast<size_t>(blocks), 0);
                    if (r == kDsv4CsaRatio) {
                        if (mc.idx_head_dim <= 0 || mc.idx_n_head <= 0) {
                            KRK_ERROR("compressed attention: a CSA layer needs "
                                      "the indexer's geometry and the file has "
                                      "none");
                            return false;
                        }
                        idx_state_kv_[k].assign(
                            static_cast<size_t>(2 * r * 2 * mc.idx_head_dim),
                            0.0f);
                        idx_state_sc_[k].assign(
                            static_cast<size_t>(2 * r * 2 * mc.idx_head_dim),
                            0.0f);
                        idx_key_[k].assign(
                            static_cast<size_t>(blocks * mc.idx_head_dim), 0.0f);
                    }
                    pool_rows = std::max(pool_rows, coff * r);
                    state_w = std::max(state_w, coff * hd);
                    max_blocks = std::max(max_blocks, blocks);
                }
                comp_pool_kv_.assign(static_cast<size_t>(pool_rows * state_w), 0.0f);
                comp_pool_sc_.assign(static_cast<size_t>(pool_rows * state_w), 0.0f);
                comp_row_kv_.assign(static_cast<size_t>(state_w), 0.0f);
                comp_row_sc_.assign(static_cast<size_t>(state_w), 0.0f);
                comp_idx_kv_.assign(static_cast<size_t>(2 * mc.idx_head_dim), 0.0f);
                comp_idx_sc_.assign(static_cast<size_t>(2 * mc.idx_head_dim), 0.0f);
                comp_iq_.assign(
                    static_cast<size_t>(mc.idx_n_head) * mc.idx_head_dim, 0.0f);
                comp_iw_.assign(static_cast<size_t>(mc.idx_n_head), 0.0f);
                comp_score_.assign(static_cast<size_t>(max_blocks), 0.0f);
                comp_sel_.assign(
                    static_cast<size_t>(mc.idx_top_k > 0 ? mc.idx_top_k : 1), 0);
                // Reported, not hidden: the pooled stream is charged to the
                // context, so a 128K V4-Flash context is 67 MiB a layer here and
                // a budget has to see it before the first token.
                i64 pooled_bytes = 0, n_comp = 0;
                for (i32 i = 0; i < mc.n_layer; i++) {
                    const size_t k = static_cast<size_t>(i);
                    pooled_bytes += static_cast<i64>(
                        (comp_pooled_[k].size() + idx_key_[k].size()) *
                        sizeof(f32));
                    if (comp_ratio_[k] != 0) n_comp++;
                }
                KRK_INFO("compressed attention: %lld layer(s) pooled, %.1f MiB of "
                         "host pooled keys (n_ctx / ratio rows of kv_lora)",
                         static_cast<long long>(n_comp),
                         static_cast<f64>(pooled_bytes) / 1048576.0);
            }
        }
    }
    // The widest FFN a layer can ask for: dense, routed expert, or shared expert.
    const i64 ff_ws = model_.n_ff_ws();
    ws_gate_ = alloc(static_cast<size_t>(C * ff_ws) * as);
    if (!check_alloc(ws_gate_, "ws_gate")) return false;
    ws_up_ = alloc(static_cast<size_t>(C * ff_ws) * as);
    if (!check_alloc(ws_up_, "ws_up")) return false;
    if (model_.had().active) {
        ws_had_ = alloc(static_cast<size_t>(C * model_.had().max_width) * as);
        if (!check_alloc(ws_had_, "ws_had")) return false;
    }
    ws_logits_ = alloc(static_cast<size_t>(n_vocab_) * as);
    if (!check_alloc(ws_logits_, "ws_logits")) return false;
    if (model_.is_recurrent() && model_.has_lfm_conv()) {
        // LFM2.5: none of the delta-net workspaces are shaped by this stack --
        // there is no [q|k|v] row, no alpha/beta, no scan state. What the block
        // needs is its own 3*n_embd projection row, the gated result, and one
        // history ring per layer.
        rec_layers_ = model_.recurrent_layers();
        ws_sc_ = alloc(static_cast<size_t>(C) * static_cast<size_t>(3 * n_embd_) * as);
        if (!check_alloc(ws_sc_, "ws_sc")) return false;
        ws_sc2_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        if (!check_alloc(ws_sc2_, "ws_sc2")) return false;
        const i32 sc_kern = mc.ssm_d_conv > 0 ? mc.ssm_d_conv : 3;
        // + slack for speculation. 16 rows covers every window a caller can ask
        // for (the drafter's block_size-1 is 8 on the LFM2.5 head set, and
        // --draft-tokens is clamped to the same order below); the cost is 16 x
        // n_embd x 4 bytes per layer, which is 1.2 MiB on the 24B file.
        sc_hist_ = static_cast<i64>(sc_kern - 1) + 16;
        sc_state_span_ = sc_hist_ * static_cast<i64>(n_embd_);
        sc_state_ = alloc(static_cast<size_t>(mc.n_layer) *
                          static_cast<size_t>(sc_state_span_) * as);
        if (!check_alloc(sc_state_, "sc_state")) return false;
        // Zeroed: the first chunk of a session reads no negative taps (the
        // kernel skips p < 0), so this is belt and braces -- but a ring that
        // starts at whatever the allocator left is the kind of thing that only
        // shows up as a first-token difference.
        be_->fill0(sc_state_, static_cast<size_t>(mc.n_layer) *
                               static_cast<size_t>(sc_state_span_) * as);
        KRK_INFO("lfm2 short conv: %d layer(s), kernel %d, %lld-row history "
                 "ring per layer",
                 model_.recurrent_layers(), sc_kern,
                 static_cast<long long>(sc_hist_));
    } else if (model_.is_recurrent()) {
        rec_layers_ = model_.recurrent_layers();
        ws_qkv_ = alloc(static_cast<size_t>(C * conv_dim_) * as);
        if (!check_alloc(ws_qkv_, "ws_qkv")) return false;
        ws_z_ = alloc(static_cast<size_t>(C * value_dim_) * as);
        if (!check_alloc(ws_z_, "ws_z")) return false;
        ws_h_ = alloc(static_cast<size_t>(C * value_dim_) * as);
        if (!check_alloc(ws_h_, "ws_h")) return false;
        // One scalar pair per value head: tiny next to the projections.
        ws_ssm_ = alloc(static_cast<size_t>(C * mc.ssm_dt_rank) * as);
        if (!check_alloc(ws_ssm_, "ws_ssm")) return false;
        ws_beta_ = alloc(static_cast<size_t>(C * mc.ssm_dt_rank) * as);
        if (!check_alloc(ws_beta_, "ws_beta")) return false;
        // f32, not activation-typed: the state outlives the forward and is
        // accumulated at higher precision than the activations it is fed.
        conv_state_span_ = static_cast<i64>(mc.ssm_d_conv - 1) * conv_dim_;
        rec_state_span_ = static_cast<i64>(mc.ssm_dt_rank) * mc.ssm_d_state *
                          mc.ssm_d_state;
        // nemotron's mamba-2 blocks carry their state in the [head][p][s]
        // layout the SSD scan wants (dt_rank = n_heads, head_dim =
        // d_inner / n_heads): n_head * head_dim * d_state f32 per layer.
        if (mc.nemotron_moe)
            mamba2_state_span_ = static_cast<i64>(mc.ssm_dt_rank) *
                                 (mc.ssm_inner_size / mc.ssm_dt_rank) *
                                 mc.ssm_d_state;
        // The fused [z|xBC|dt] projection row is wider than the conv row the
        // GDN path sizes ws_qkv_ for (10304 vs 6144 on nemotron-30B).
        if (mc.nemotron_moe) {
            ws_m2in_ = alloc(static_cast<size_t>(C * (conv_dim_ + value_dim_ +
                                                      mc.ssm_dt_rank)) * as);
            if (!check_alloc(ws_m2in_, "ws_m2in")) return false;
        }
        conv_state_ = alloc(static_cast<size_t>(rec_layers_) *
                            static_cast<size_t>(conv_state_span_) * sizeof(f32));
        if (!check_alloc(conv_state_, "conv_state")) return false;
        rec_state_ = alloc(static_cast<size_t>(rec_layers_) *
                           static_cast<size_t>(mamba2_state_span_ > 0
                               ? mamba2_state_span_ : rec_state_span_) *
                           sizeof(f32));
        if (!check_alloc(rec_state_, "rec_state")) return false;
    }
    logits_host_ = static_cast<f32 *>(host_alloc(static_cast<size_t>(n_vocab_) * 4));
    tok_scratch_.resize(static_cast<size_t>(C));
    // Device top-k. A fixed 36 kB of candidate buffer, allocated for every
    // backend so that "can we use the fast path" is a runtime check and not a
    // different code path with a null to forget.
    topk_scratch_ = alloc(4096 + static_cast<size_t>(kTopKCap) * 8);
    if (!check_alloc(topk_scratch_, "topk_scratch")) return false;
    topk_out_ = alloc(16);
    if (!check_alloc(topk_out_, "topk_out")) return false;
    ws_routerp_ = alloc(static_cast<size_t>(C) * static_cast<size_t>(mc.n_expert) * 4);
    if (!check_alloc(ws_routerp_, "ws_routerp")) return false;
    // The value router. Sized from the FILE's value_expert_count, so a family
    // without them allocates nothing and the pointers stay null.
    if (mc.n_value_expert > 0) {
        const i64 kvw = static_cast<i64>(mc.n_head_kv) * mc.head_dim;
        ws_vgate_ = alloc(static_cast<size_t>(C) * static_cast<size_t>(mc.n_value_expert) * 4);
        if (!check_alloc(ws_vgate_, "ws_vgate")) return false;
        ws_vexp_ = alloc(static_cast<size_t>(C) * static_cast<size_t>(kvw) * be_->act_size());
        if (!check_alloc(ws_vexp_, "ws_vexp")) return false;
        vgate_host_.resize(static_cast<size_t>(C) * static_cast<size_t>(mc.n_value_expert));
    }
    // The device selection and its plan. Allocated for every backend, exactly
    // like ws_routerp_: "is this available" is a runtime question, and a
    // separate allocation path with a null to forget is how a null gets used.
    if (mc.is_moe && mc.n_expert > 0) {
        const i32 kcap = std::min<i32>(mc.n_expert,
                                      mc.n_expert_used > 0 ? mc.n_expert_used : 1);
        ws_ids_ = alloc(static_cast<size_t>(C) * static_cast<size_t>(kcap) * 4);
        if (!check_alloc(ws_ids_, "ws_ids")) return false;
        ws_wts_ = alloc(static_cast<size_t>(C) * static_cast<size_t>(kcap) * 4);
        if (!check_alloc(ws_wts_, "ws_wts")) return false;
        ws_off_ = alloc((static_cast<size_t>(mc.n_expert) + 1) * 4);
        if (!check_alloc(ws_off_, "ws_off")) return false;
    }
    if (!logits_host_) return fail_alloc("host logits");

    if (mc.is_moe && mc.n_expert > 0) {
        const i32 k = std::max<i32>(1, mc.n_expert_used);
        ws_router_ = alloc(static_cast<size_t>(C * mc.n_expert) * as);
        if (!check_alloc(ws_router_, "ws_router")) return false;
        ws_ffn_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        if (!check_alloc(ws_ffn_, "ws_ffn")) return false;
        // The shared expert's per-token gate. It used to be computed into
        // `ws_ssm_`, which is allocated only for a RECURRENT stack, so the
        // first non-hybrid family to carry `ffn_gate_inp_shexp` (qwen2moe)
        // dereferenced a null pointer in its first forward.
        ws_shexp_g_ = alloc(static_cast<size_t>(C) * as);
        // The two-branch block's own buffers. They exist only for a family that
        // declares `moe_two_branch`, and the block that reads them is entered
        // only on a layer of that family, so the guard and the use are the same
        // predicate (docs/ENGINE-AUDIT.md).
        if (mc.moe_two_branch) {
            ws_g4_post_ = alloc(static_cast<size_t>(C * n_embd_) * as);
            if (!check_alloc(ws_g4_post_, "ws_g4_post")) return false;
            ws_g4_mlp_ = alloc(static_cast<size_t>(C * n_embd_) * as);
            if (!check_alloc(ws_g4_mlp_, "ws_g4_mlp")) return false;
            ws_router_in_ = alloc(static_cast<size_t>(C * n_embd_) * as);
            if (!check_alloc(ws_router_in_, "ws_router_in")) return false;
        }
        if (!check_alloc(ws_shexp_g_, "ws_shexp_g")) return false;
        // ---- LatentMoE ----------------------------------------------------
        // One narrow space per forward, sized by the widest latent the file
        // declares (a merged stack may vary it per layer, and every layer must
        // fit the same buffer). Absent on a file without ffn_latent_*, which is
        // every other family.
        i64 lat_w = 0;
        for (const LayerWeights &LW : model_.layers()) {
            if (LW.latent_moe() && LW.experts.n_embd > 0)
                lat_w = std::max<i64>(lat_w, LW.experts.n_embd);
        }
        if (lat_w > 0) {
            const size_t lws = static_cast<size_t>(lat_w);
            ws_xlat_ = alloc(static_cast<size_t>(C) * lws * as);
            if (!check_alloc(ws_xlat_, "ws_xlat")) return false;
            ws_ffn_lat_ = alloc(static_cast<size_t>(C) * lws * as);
            if (!check_alloc(ws_ffn_lat_, "ws_ffn_lat")) return false;
            ws_xlat_out_ = alloc(static_cast<size_t>(C * n_embd_) * as);
            if (!check_alloc(ws_xlat_out_, "ws_xlat_out")) return false;
        }
        // Batched-expert staging: the plan can never exceed chunk rows because
        // each (token, expert) pair consumes exactly one slot.
        ws_xg_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        if (!check_alloc(ws_xg_, "ws_xg")) return false;
        ws_gateg_ = alloc(static_cast<size_t>(C * ff_ws) * as);
        if (!check_alloc(ws_gateg_, "ws_gateg")) return false;
        ws_upg_ = alloc(static_cast<size_t>(C * ff_ws) * as);
        if (!check_alloc(ws_upg_, "ws_upg")) return false;
        // The batched-expert path's own buffers: kMoeBatch experts x (gate, up)
        // and kMoeBatch down rows, plus one f32 scale per expert. Separate from
        // the buffers above because those are [chunk, ...] and this path is the
        // one-token case, where every expert needs its OPEN row at once.
        ws_ebatch_ = alloc(static_cast<size_t>(kMoeBatch * 2 * ff_ws) * as);
        if (!check_alloc(ws_ebatch_, "ws_ebatch")) return false;
        ws_edown_ = alloc(static_cast<size_t>(kMoeBatch * n_embd_) * as);
        if (!check_alloc(ws_edown_, "ws_edown")) return false;
        ws_ebscale_ = alloc(static_cast<size_t>(kMoeBatch) * sizeof(f32));
        if (!check_alloc(ws_ebscale_, "ws_ebscale")) return false;
        // The whole layer's plan lives in one buffer: at most one (token,
        // expert) pair per selection slot, so C * k entries is the bound, and a
        // visit reads [off(e), off(e+1)) of it. It was C entries when each visit
        // uploaded its own segment.
        ws_plan_ = alloc(static_cast<size_t>(C) * k * sizeof(i32));
        if (!check_alloc(ws_plan_, "ws_plan")) return false;
        ws_alpha_ = alloc(static_cast<size_t>(C) * k * sizeof(f32));
        if (!check_alloc(ws_alpha_, "ws_alpha")) return false;
        // One multiplier per routed row, written by the scaled silu_mul and read
        // back by the scatter of the same visit. The plan can never exceed chunk
        // rows, so C entries always suffice.
        ws_scale_ = alloc(static_cast<size_t>(C) * sizeof(f32));
        if (!check_alloc(ws_scale_, "ws_scale")) return false;
        router_host_.resize(static_cast<size_t>(C) * static_cast<size_t>(mc.n_expert));
        moe_prob_.resize(static_cast<size_t>(mc.n_expert));
        if (mc.n_expert_group > 1)
            moe_group_keep_.assign(static_cast<size_t>(mc.n_expert), 1);
        moe_logit_.resize(static_cast<size_t>(mc.n_expert));
        moe_sel_.resize(static_cast<size_t>(C) * static_cast<size_t>(k));
        // Only sized when a scan is actually requested: [n_layer * n_expert]
        // doubles and u64 counters, which is 3584 * 12 B = 43 kB for a
        // 28x128 model and nothing at all otherwise.
        if (expert_scan_ || expert_track_) {
            const size_t cells = static_cast<size_t>(mc.n_layer) *
                                 static_cast<size_t>(mc.n_expert);
            expert_mass_.assign(cells, 0.0);
            expert_hits_.assign(cells, 0);
            expert_scan_tokens_ = 0;
            // The device half: u32 counts and f32 mass, one cell per expert per
            // layer. 80 kB for a 40x256 model, and nothing at all when the
            // backend cannot accumulate -- in which case the host plan keeps
            // counting exactly as before.
            if (be_->scan_accum_supported()) {
                scan_hits_dev_ = alloc(cells * sizeof(u32));
                scan_mass_dev_ = alloc(cells * sizeof(f32));
                if (scan_hits_dev_ && scan_mass_dev_) {
                    be_->fill0(scan_hits_dev_, cells * sizeof(u32));
                    be_->fill0(scan_mass_dev_, cells * sizeof(f32));
                } else {
                    // A partial allocation is worse than none: the host path
                    // would be skipped for buffers that cannot be read back.
                    if (scan_hits_dev_) be_->release(scan_hits_dev_);
                    if (scan_mass_dev_) be_->release(scan_mass_dev_);
                    scan_hits_dev_ = nullptr;
                    scan_mass_dev_ = nullptr;
                }
            }
        }
        moe_wt_.resize(static_cast<size_t>(C) * static_cast<size_t>(k));
        group_rows_.reserve(static_cast<size_t>(C));
        group_wt_.reserve(static_cast<size_t>(C));
        plan_dev_.reserve(static_cast<size_t>(C));
        alpha_dev_.reserve(static_cast<size_t>(C));
    }

    // ---- nextn (MTP) probe: its own KV plane and the hidden capture --------
    // Allocated before the expert cache and the KV tier, which both size
    // themselves against what is left: a probe that quietly shrinks the tiers
    // it is measured beside would measure a different machine.
    mtp_probe_ = false;
    mtp_spec_ = false;
    mtp_probe_report_ = cfg_.mtp_probe;
    if (cfg_.mtp_probe || cfg_.mtp_spec) {
        const MtpBlock &m = model_.mtp();
        const i64 bq = static_cast<i64>(m.n_head) * m.head_dim;
        const i64 bkv = static_cast<i64>(m.n_head_kv) * m.head_dim;
        const i32 kmain = std::max<i32>(1, mc.n_expert_used);
        const char *why = nullptr;
        if (!m.present)
            why = "the file's nextn block is not loaded";
        else if (m.wq.n_out != bq && m.wq.n_out != 2 * bq)
            why = "the query projection is neither q nor 2*q wide";
        else if (m.wq.n_out == 2 * bq && q_proj_ != 2 * q_dim_)
            why = "the block packs an output gate and this engine's query "
                  "workspace does not";
        else if (bq > q_dim_ || bkv > kv_dim_)
            why = "the block's head geometry is wider than the stack's "
                  "workspaces";
        else if (m.n_expert > 0 &&
                 (!mc.is_moe || m.n_expert > mc.n_expert ||
                  m.n_expert_used > chunk_ * kmain))
            why = "the block's expert geometry does not fit the MoE workspaces";
        else if (m.n_expert == 0 && m.n_ff == 0)
            why = "the block has neither routed experts nor a dense FFN";
        else if (m.n_ff_exp > ff_ws || m.n_ff_shexp > ff_ws || m.n_ff > ff_ws)
            why = "the block's FFN is wider than the FFN workspaces";
        if (why) {
            KRK_WARN("mtp probe: off -- %s", why);
        } else {
            mtp_kv_dim_ = bkv;
            mtp_kv_k_ = alloc(static_cast<size_t>(kv_cap_ * mtp_kv_dim_) * as);
            mtp_kv_v_ = alloc(static_cast<size_t>(kv_cap_ * mtp_kv_dim_) * as);
            mtp_hidden_ = alloc(static_cast<size_t>(kv_cap_ * n_embd_) * as);
            mtp_h_ = alloc(static_cast<size_t>(n_embd_) * as);
            mtp_e_ = alloc(static_cast<size_t>(n_embd_) * as);
            mtp_cat_ = alloc(static_cast<size_t>(2 * n_embd_) * as);
            mtp_gate_ = alloc(static_cast<size_t>(q_dim_) * as);
            mtp_sgate_ = alloc(as);
            mtp_rep_ = alloc(sizeof(i32));
            if (!mtp_kv_k_ || !mtp_kv_v_ || !mtp_hidden_ || !mtp_h_ || !mtp_e_ ||
                !mtp_cat_ || !mtp_gate_ || !mtp_sgate_ || !mtp_rep_) {
                for (void **p : {&mtp_kv_k_, &mtp_kv_v_, &mtp_hidden_, &mtp_h_,
                                 &mtp_e_, &mtp_cat_, &mtp_gate_, &mtp_sgate_,
                                 &mtp_rep_})
                    if (*p) {
                        be_->release(*p);
                        *p = nullptr;
                    }
                KRK_WARN("mtp probe: off -- its %.1f MiB of capture and block KV "
                         "could not be allocated; the run continues without it",
                         (static_cast<f64>(kv_cap_ * n_embd_) * as +
                          static_cast<f64>(2 * kv_cap_ * mtp_kv_dim_) * as) /
                             1048576.0);
            } else {
                be_->fill0(mtp_rep_, sizeof(i32));
                mtp_tok_.assign(static_cast<size_t>(kv_cap_), -1);
                mtp_probe_ = true;
                mtp_spec_ = cfg_.mtp_spec;
                // A speculative round proposes up to this many tokens; with no
                // --draft-tokens the same default a draft engine gets applies.
                if (mtp_spec_ && draft_tokens_ <= 0) draft_tokens_ = 4;
                // A hybrid target needs its recurrent state snapshotted per
                // round: the delta rule cannot be rewound, so a rejected
                // suffix is undone by restoring this and replaying the
                // accepted prefix. Sized exactly as the state itself.
                if (mtp_spec_ && model_.is_recurrent() && rec_state_) {
                    mtp_rec_snap_bytes_ =
                        static_cast<size_t>(rec_layers_) *
                        static_cast<size_t>(mamba2_state_span_ > 0
                            ? mamba2_state_span_ : rec_state_span_) *
                        sizeof(f32);
                    mtp_conv_snap_bytes_ =
                        static_cast<size_t>(rec_layers_) *
                        static_cast<size_t>(conv_state_span_) * sizeof(f32);
                    mtp_rec_snap_ = alloc(mtp_rec_snap_bytes_);
                    mtp_conv_snap_ = alloc(mtp_conv_snap_bytes_);
                    if (!mtp_rec_snap_ || !mtp_conv_snap_) {
                        if (mtp_rec_snap_) { be_->release(mtp_rec_snap_); mtp_rec_snap_ = nullptr; }
                        if (mtp_conv_snap_) { be_->release(mtp_conv_snap_); mtp_conv_snap_ = nullptr; }
                        mtp_rec_snap_bytes_ = mtp_conv_snap_bytes_ = 0;
                        KRK_WARN("mtp spec: off -- the %.1f MiB recurrent-state "
                                 "snapshot could not be allocated",
                                 (static_cast<f64>(
                                      static_cast<i64>(rec_layers_) *
                                      (mamba2_state_span_ > 0 ? mamba2_state_span_
                                                              : rec_state_span_) +
                                      static_cast<i64>(rec_layers_) *
                                          conv_state_span_) * 4.0) / 1048576.0);
                        mtp_spec_ = false;
                    }
                }
                KRK_INFO("mtp probe: block %d -> %d head(s) x %d + %d kv head(s), "
                         "%d experts x top-%d, eh_proj %lldx%lld; %.1f MiB hidden "
                         "capture + %.1f MiB block KV",
                         m.layer, m.n_head, m.head_dim, m.n_head_kv, m.n_expert,
                         m.n_expert_used,
                         static_cast<long long>(m.eh_proj.n_in),
                         static_cast<long long>(m.eh_proj.n_out),
                         (static_cast<f64>(kv_cap_ * n_embd_) * as) / 1048576.0,
                         (static_cast<f64>(2 * kv_cap_ * mtp_kv_dim_) * as) /
                             1048576.0);
            }
        }
    }

    // Fail before the first forward can read or write a missing workspace.
    // Optional workspaces are required only by the architecture/path that
    // requested them; all fixed-size workspaces are checked unconditionally.
    const auto require_alloc = [&](const void *p, const char *name) {
        return p != nullptr || fail_alloc(name);
    };
    if (!require_alloc(ws_x_, "ws_x") || !require_alloc(ws_xn_, "ws_xn") ||
        !require_alloc(ws_x2_, "ws_x2") || !require_alloc(ws_q_, "ws_q") ||
        !require_alloc(ws_k_, "ws_k") || !require_alloc(ws_v_, "ws_v") ||
        !require_alloc(ws_attn_, "ws_attn") || !require_alloc(ws_gate_, "ws_gate") ||
        !require_alloc(ws_up_, "ws_up") || !require_alloc(ws_logits_, "ws_logits") ||
        !require_alloc(topk_scratch_, "topk_scratch") || !require_alloc(topk_out_, "topk_out") ||
        !require_alloc(ws_routerp_, "ws_routerp") ||
        (mc.is_moe && (!require_alloc(ws_ids_, "ws_ids") ||
                       !require_alloc(ws_wts_, "ws_wts") ||
                       !require_alloc(ws_off_, "ws_off"))) ||
        (q_proj_ != q_dim_ && !require_alloc(ws_qpack_, "ws_qpack")) ||
        ((model_.is_recurrent() || mc.attn_gate) && !require_alloc(ws_agate_, "ws_agate")) ||
        (model_.had().active && !require_alloc(ws_had_, "ws_had")) ||
        (model_.is_recurrent() && model_.has_lfm_conv() &&
         (!require_alloc(ws_sc_, "ws_sc") || !require_alloc(ws_sc2_, "ws_sc2") ||
          !require_alloc(sc_state_, "sc_state"))) ||
        (model_.is_recurrent() && !model_.has_lfm_conv() &&
         (!require_alloc(ws_qkv_, "ws_qkv") || !require_alloc(ws_z_, "ws_z") ||
          !require_alloc(ws_h_, "ws_h") || !require_alloc(ws_ssm_, "ws_ssm") ||
          !require_alloc(ws_beta_, "ws_beta") || !require_alloc(conv_state_, "conv_state") ||
          !require_alloc(rec_state_, "rec_state"))) ||
        (mc.nemotron_moe && !require_alloc(ws_m2in_, "ws_m2in")) ||
        (mc.is_moe && mc.n_expert > 0 &&
         (!require_alloc(ws_router_, "ws_router") || !require_alloc(ws_ffn_, "ws_ffn") ||
          !require_alloc(ws_xg_, "ws_xg") || !require_alloc(ws_gateg_, "ws_gateg") ||
          !require_alloc(ws_upg_, "ws_upg") || !require_alloc(ws_plan_, "ws_plan") ||
          !require_alloc(ws_alpha_, "ws_alpha") || !require_alloc(ws_scale_, "ws_scale"))))
        return false;
    if (!logits_host_) return fail_alloc("host logits");

    // Hybrid CPU + GPU expert compute. Resolved once, here, so the per-layer
    // path is a bool test and the worker team is created before the first
    // forward pass instead of on the first decode step.
    //
    // HIP only, and MoE only: on the CPU backend there is no second processor to
    // split with, and on a dense model there are no expert tensors to split.
    hybrid_on_ = cfg_.hybrid_experts && mc.is_moe && mc.n_expert > 0 &&
                 be_->kind() == BackendKind::HIP;
    if (hybrid_on_) {
        hybrid_threads_ = cfg_.hybrid_threads;
        // 0 means "no row bound"; storing a sentinel keeps the per-layer test a
        // plain comparison instead of a branch on the configuration.
        hybrid_max_rows_ = cfg_.hybrid_max_rows > 0 ? cfg_.hybrid_max_rows : (1 << 20);
        f32 frac = cfg_.hybrid_frac;
        if (!(frac >= 0.0f)) frac = 1.0f; // also catches NaN
        if (frac > 1.0f) frac = 1.0f;
        hybrid_frac_permille_ = static_cast<i32>(frac * 1000.0f + 0.5f);
        CpuExpertPool &pool = CpuExpertPool::get();
        pool.set_threads(cfg_.hybrid_threads);
        hybrid_threads_live_ = pool.threads();
        KRK_INFO("hybrid experts: host arm on %d thread(s), rows <= %d, "
                 "%d/1000 of the resident set forced to the host",
                 hybrid_threads_live_, hybrid_max_rows_, hybrid_frac_permille_);
    }
    // Every run states its headroom before it starts, dense models included.
    // The fault this guards against reached 0.00 GiB free, and the only line
    // that showed it printed at the END of the run -- after the display was
    // already gone. Reading it in the first lines is what makes a card-filling
    // configuration visible while it can still be changed.
    {
        const size_t vtotal = be_->device_total_bytes();
        const size_t vfree = be_->device_free_bytes();
        if (vtotal > 0) {
            const size_t vrsv = krk::vram_reserve_bytes(vtotal);
            KRK_INFO("vram headroom: %.2f GiB free of %.2f GiB total, %.2f GiB "
                     "reserved (KRK_VRAM_RESERVE_MB overrides)",
                     static_cast<f64>(vfree) / 1073741824.0,
                     static_cast<f64>(vtotal) / 1073741824.0,
                     static_cast<f64>(vrsv) / 1073741824.0);
            if (vfree < vrsv)
                KRK_WARN("vram headroom: free VRAM is BELOW the %.2f GiB reserve, "
                         "so an expert tier sized against the plan will be cut "
                         "short -- lower --ctx / --kv-hot-mb, or lower "
                         "KRK_VRAM_RESERVE_MB if this card is dedicated",
                         static_cast<f64>(vrsv) / 1073741824.0);
        }
    }
    if (!configure_expert_cache(err)) {
        shutdown();
        return false;
    }
    // configure_expert_cache applies the measured/explicit pin budget. Only
    // enable prediction if its nonblocking pinned-source contract is true in
    // the final cache configuration (auto pinning may have been rejected by
    // the H2D probe, so a pre-configuration guess is not enough).
    moe_predict_enabled_ = moe_predict_requested && moe_predict_arch_ok &&
                           model_.experts().host_pin_budget() > 0;
    if (moe_predict_requested && moe_predict_arch_ok && !moe_predict_enabled_)
        KRK_INFO("expert predictor: disabled (pinned WARM is off; "
                 "stage_ahead requires pinned DMA sources)");
    else if (moe_predict_enabled_)
        KRK_INFO("expert predictor: online, policy=%d history=%d limit=%d "
                 "min-precision=%.2f; native routes remain authoritative",
                 moe_predict_policy(), moe_predict_history(), moe_predict_budget(),
                 static_cast<f64>(moe_predict_min_overlap()));
    // KV residency -- allocated LAST, after the weights and every workspace.
    //
    // That ordering is load-bearing. Reading device_free_bytes() before the
    // workspaces exist reports nearly all free VRAM, so a budget taken here
    // over-commits: the slots are allocated, and then ws_x_ and the rest are
    // allocated on top of them. Subtracting "what the workspaces will take"
    // is not the same thing as measuring it, and guessing it is how an
    // over-commit happens. So: take the measurement after the fact.
    // Gemma 2's attention-logit bound is part of the model, not an
    // optimization: cap * tanh(score / cap) before the softmax is a different
    // probability distribution. A backend that cannot apply it must refuse
    // before the first token, because the run would otherwise decode a model
    // nobody asked for and look completely healthy doing it.
    if (model_.cfg().attn_softcap > 0.0f && !be_->attention_softcap_supported()) {
        *err = format(
            "%s declares attention.logit_softcapping=%g, and this backend's "
            "attention kernels do not apply cap*tanh(score/cap) before the "
            "softmax. The reference body does: run --cpu, or --npu-cpu to keep "
            "the GEMM on the virtual NPU with the reference attention",
            model_.cfg().arch.c_str(), static_cast<double>(model_.cfg().attn_softcap));
        return false;
    }

    // ALiBi is the same kind of thing: a per-head distance penalty on the
    // score, so a backend that ignored it would be running a different
    // attention function on a file that has no rotary and no learned position
    // to fall back on. Refuse before the first token rather than decode
    // something else.
    if (model_.cfg().alibi_max_bias > 0.0f && !be_->attention_alibi_supported()) {
        *err = format(
            "%s uses ALiBi (max bias %g), and this backend's attention kernels "
            "do not apply the per-head distance bias. That bias IS this "
            "family's position signal -- there is no rotary or learned position "
            "underneath it. The reference body applies it: run --cpu, or "
            "--npu-cpu to keep the GEMM on the virtual NPU with the reference "
            "attention",
            model_.cfg().arch.c_str(),
            static_cast<double>(model_.cfg().alibi_max_bias));
        return false;
    }

    // Quantized K/V is validated before the cache is sized: a backend that
    // cannot read the layout must refuse, not run f16 kernels over int8 bytes
    // (which produces fluent text rather than an error).
    const KvQuant kvq = cfg_.kv_quant == 1 ? KvQuant::Int8 : KvQuant::F16;
    if (kvq == KvQuant::Int8) {
        if (!be_->kv_quant_supported(kvq)) {
            // The CPU reference and HIP both answer true: the quantising append
            // plus all four attention readers, with the cache bytes and the
            // attention output invariant to the chunk size
            // (`kraken-bench --bitexact`'s int8 block is the device proof). A
            // backend that answers false is one with no int8 reader at all, so
            // this refusal names what is missing rather than borrowing a
            // reason from the layout's history.
            *err = "kv-quant int8: this backend has no attention kernels for "
                   "the int8 cache layout (the CPU reference and HIP implement "
                   "it)";
            return false;
        }
        if (model_.cfg().mla_layout != MlaLayout::None) {
            *err = "kv-quant int8: the MLA cache layout has no int8 write/read path";
            return false;
        }
        if (cfg_.mtp_spec || cfg_.mtp_probe || !cfg_.draft_head.empty()) {
            *err = "kv-quant int8: the nextn/draft block keeps its own KV and has "
                   "no int8 path; run it with kv-quant f16";
            return false;
        }
    }
    size_t kv_load_bytes = 0; // both planes, for the load report below
    {
        // Per-layer KV bytes. Only the layers that carry q/k/v hold KV at all
        // (nemotron_h_moe: 6 of its 52) and each holds the plane its own width
        // asks for, so the charge is their sum. `n_layer x the widest layer` is
        // what this used to be, and on that model it reserved and printed
        // 208 MiB at --ctx 4096 for the 24 MiB its KV can occupy -- 8.67x. It
        // is not only a reporting number: it is what the flat-or-tiered test in
        // KvTierCache::init compares a budget against.
        std::vector<size_t> kv_len(static_cast<size_t>(mc.n_layer), 0);
        size_t kv_plane = 0;
        for (i32 l = 0; l < mc.n_layer; l++) {
            if (!model_.layer_has_kv(l)) continue;
            const size_t b = static_cast<size_t>(kv_cap_) *
                             be_->kv_row_bytes(model_.kv_dim_at(l), kvq);
            kv_len[static_cast<size_t>(l)] = b;
            kv_plane += b;
        }
        const size_t kv_total = kv_plane * 2u;
        kv_load_bytes = kv_total;
        kv_bytes_ = kv_total;

        // Leave a reserve so a later allocation cannot wedge the driver. The
        // device is shared: the desktop compositor and any other app are
        // already drawing from what is left.
        const size_t free_vram = be_->device_free_bytes();
        const size_t reserve = std::max<size_t>(256u * 1048576u, free_vram / 8u);

        // The CPU backend has no device/host split -- every byte is the same
        // host RAM -- so a tier budget is meaningless there, and a 0-byte HOT
        // budget is worse than meaningless: it pages the whole cache through
        // WARM once per layer per step and prints a warning telling the user to
        // raise --kv-hot-mb, which cannot help. Measured on the 27B Bonsai,
        // 64 layers: 4951 ms/step tiered against 27 ms for the same model on
        // the device. Handing the "device" budget the whole cache makes the
        // tier inert and the cache one flat allocation, which is what a run
        // whose cache fits does on the GPU too.
        size_t hot;
        if (!be_->has_device_memory()) {
            hot = kv_total;
        } else if (cfg_.kv_hot_mb > 0) {
            hot = static_cast<size_t>(cfg_.kv_hot_mb) * 1048576u;
            // An explicit budget can still be larger than the card. Clamp it
            // to what is actually there, or init() over-commits on request.
            const size_t cap = free_vram > reserve ? free_vram - reserve : 0;
            if (hot > cap) hot = cap;
        } else {
            hot = free_vram > reserve ? free_vram - reserve : 0;
        }

        // WARM is host RAM, so it must not be sized from device memory. A
        // pageable allocation is admitted lazily only after a live availability
        // check; if the OS cannot report that number, every WARM/COLD staging
        // allocation is refused and a retained KV page makes the forward fail.
        // Explicit --kv-warm-mb remains a capacity ceiling, not permission to
        // cross the live host reserve.
        //
        // "Whatever does not fit HOT" is the whole cache, not the cache minus
        // the HOT bytes: the tiers must JOINTLY cover every KV-carrying layer,
        // because a layer in neither tier is zero-filled when it is next needed
        // and its history is lost silently. Sizing WARM to `kv_total - hot`
        // misses that by the slot rounding -- measured, SmolLM2 --ctx 512 with
        // --kv-hot-mb 1: 2 HOT + 26 WARM slots for 30 KV layers, 132 dropped
        // pages in a 32-token run, and text that diverges from the flat cache.
        // Passing the plane total makes the cache cover every layer (see
        // KvTierCache::init) and costs host RAM only if the KV is ever paged
        // out to it, since WARM buffers are allocated lazily.
        size_t warm = kv_total;
        if (cfg_.kv_warm_mb >= 0)
            warm = static_cast<size_t>(cfg_.kv_warm_mb) * 1048576u;
        const size_t host_avail = be_->host_available_bytes();
        const size_t host_reserve = krk::host_reserve_bytes(be_->host_total_bytes());
        const size_t host_stage_room = host_avail > host_reserve
                                           ? host_avail - host_reserve : 0;
        const size_t already_budgeted = krk::SharedHostRamBudget::used_bytes();
        const size_t combined_stage_room = host_stage_room > already_budgeted
                                               ? host_stage_room - already_budgeted : 0;
        if (warm > combined_stage_room) warm = combined_stage_room;
        if (host_avail == 0) warm = 0; // unknown => no pageable staging

        // Two planes share both budgets.
        // The plane tag is not decoration: both caches share one cold
        // directory, so without it they spill to the same file name and a
        // restore returns the other plane's KV.
        if (!kvt_k_.init(be_, mc.n_layer, kv_len, hot / 2u, warm / 2u,
                         cfg_.kv_cold_dir, "k", err)) {
            shutdown();
            return false;
        }
        if (!kvt_v_.init(be_, mc.n_layer, kv_len, hot / 2u, warm / 2u,
                         cfg_.kv_cold_dir, "v", err)) {
            shutdown();
            return false;
        }
        // Geometry, reported once here rather than by each plane: both planes
        // have the same numbers, and a per-plane warning printed every line
        // twice. Neither message changes a policy -- the A/B numbers in
        // docs/test-results.md were produced by these configurations and a
        // silent second-guess would invalidate them -- but one of them says the
        // run cannot be correct, and that must not be left to be inferred from
        // a slowdown.
        if (kvt_k_.tiered()) {
            const i64 holds = kvt_k_.hot_slots();
            const i64 layers = kvt_k_.kv_layers();
            const f64 plane_mib =
                static_cast<f64>(kvt_k_.plane_bytes()) / 1048576.0;
            if (holds < layers) {
                KRK_WARN("kv: HOT tier holds %lld slot(s) for %lld KV layer(s), "
                         "so the resident set cannot hold one full pass over "
                         "the model. Expect one migration per layer per decode "
                         "step with no reuse (measured up to 206x slower than "
                         "fitting flat). If the card can hold the whole cache, "
                         "raise --kv-hot-mb to at least %.1f MiB and this "
                         "disappears; if it cannot, the run is memory-bound by "
                         "construction and only --ctx or a smaller model "
                         "changes it.",
                         static_cast<long long>(holds),
                         static_cast<long long>(layers), plane_mib);
            }
            // The WARM capacity is derived from the geometry, so a budget that
            // is too small for one pass is raised rather than obeyed -- tell
            // the caller, because it is their flag that changed and it changes
            // how much host RAM the run may touch.
            if (kvt_k_.host_query_unknown()) {
                KRK_WARN("kv: host-memory availability is unknown; WARM capacity "
                         "is zero, and a retained page that cannot be spilled "
                         "will stop inference rather than lose KV history");
            }
            if (kvt_k_.warm_from_geometry()) {
                const f64 applied_mib =
                    static_cast<f64>(kvt_k_.warm_slots() *
                                     kvt_k_.slot_bytes() * 2u) / 1048576.0;
                const f64 asked_mib = static_cast<f64>(kvt_k_.warm_slots_from_budget() *
                                                       kvt_k_.slot_bytes() * 2u) /
                                      1048576.0;
                KRK_WARN("kv: --kv-warm-mb allows %.1f MiB of WARM for %lld KV "
                         "layer(s) and one pass needs all of them, and there is "
                         "no --kv-cold-dir to spill to, so WARM was raised to "
                         "%lld slot(s) = %.1f MiB. A page that fits nowhere is "
                         "DROPPED and its layer is zero-filled on its next step, "
                         "which decodes WRONG text, so the budget is not obeyed "
                         "unless a spill directory exists. WARM is allocated "
                         "lazily, so this costs nothing until an eviction "
                         "actually lands there; raise --kv-hot-mb (%.2f MiB per "
                         "slot) if it must not page through host memory, or set "
                         "--kv-cold-dir to keep the smaller WARM.",
                         asked_mib, static_cast<long long>(layers),
                         static_cast<long long>(kvt_k_.warm_slots()), applied_mib,
                         static_cast<f64>(kvt_k_.slot_bytes()) / 1048576.0);
            }
            if (kvt_k_.cold_dir_rejected()) {
                KRK_WARN("kv: --kv-cold-dir '%s' is not writable, so it was "
                         "ignored and evictions are held in WARM instead. A "
                         "spill that fails on fopen/fwrite is a DROPPED page, "
                         "which is a wrong decode, so an unwritable directory "
                         "is treated as no directory. Create it and re-run to "
                         "use the disk tier.",
                         cfg_.kv_cold_dir.c_str());
            }
            // Defensive only: with no cold dir, init pins the WARM capacity at
            // the KV-layer count, so HOT + WARM covers one pass by
            // construction (see KvTierCache::init). If this ever prints, the
            // enforcement in init is what broke, not the caller's flags.
            if (holds + kvt_k_.warm_slots() < layers && cfg_.kv_cold_dir.empty()) {
                KRK_WARN("kv: HOT %lld + WARM %lld slot(s) for %lld KV layer(s) "
                         "and no --kv-cold-dir: the tiers cannot hold one pass "
                         "over the cache, so a page that fits nowhere is "
                         "DROPPED and its layer is zero-filled on the next step. "
                         "The run will still produce text and the text will be "
                         "wrong. Raise --kv-warm-mb to at least %.1f MiB, or set "
                         "--kv-cold-dir so the overflow has somewhere to spill.",
                         static_cast<long long>(holds),
                         static_cast<long long>(kvt_k_.warm_slots()),
                         static_cast<long long>(layers),
                         static_cast<f64>(kv_total) / 1048576.0);
            }
        }
        KvTierStats ks;
        kvt_k_.stats(&ks);
        kv_tiered_ = ks.tiered;
    }
    if (model_.is_recurrent()) {
        recurrent_reset();
        const f64 rec_mb = static_cast<f64>(rec_layers_) *
                           static_cast<f64>(conv_state_span_ + rec_state_span_) * 4.0 /
                           (1024.0 * 1024.0);
        KRK_INFO("recurrent state: %d layers, %.1f MiB f32", rec_layers_, rec_mb);
    }
    be_->sync();

    const f64 load_ms = load_timer.ms();
    // The KV the model actually has, both planes: the sum over the KV-carrying
    // layers of their own width, not n_layer x the widest.
    const f64 total_vram_mb =
        static_cast<f64>(kv_load_bytes) / (1024.0 * 1024.0);
    {
        KvTierStats ks;
        kvt_k_.stats(&ks);
        // Capacity, not occupancy: `ks.warm` is how many layers happen to be
        // in WARM at the instant of the call, which at load time is always 0
        // and reads as "WARM holds nothing" on a run that has a full WARM.
        // What a reader needs is whether the tiers can hold one pass, and that
        // is the three counts on the left.
        if (ks.tiered)
            KRK_INFO("kv tiered: %lld KV layer(s) of %.0f MiB, %lld HOT + %lld "
                     "WARM slot(s) of %.2f MiB (largest layer), %.2f MiB VRAM "
                     "for HOT",
                     static_cast<long long>(ks.layers), total_vram_mb,
                     static_cast<long long>(ks.slots),
                     static_cast<long long>(kvt_k_.warm_slots()),
                     static_cast<double>(ks.layer_bytes) / 1048576.0,
                     static_cast<double>(ks.hot_bytes) / 1048576.0);
        else
            KRK_INFO("workspaces ready: chunk=%d ctx=%lld KV=%.0f MiB (load %.0f ms)",
                     chunk_, static_cast<long long>(kv_cap_), total_vram_mb, load_ms);
    }

    // Fill the residency tiers before the first token. This runs after
    // configure_expert_cache (which decides the budgets and, when the warmup is
    // on, deliberately leaves WARM to this phase) and before the 1-token
    // weights pass, so that pass is itself a warm-tier run rather than a cold
    // one. See EngineConfig::expert_warmup for when it engages.
    if (expert_warmup_wanted()) {
        // The ranked warmup reads the whole hot set off disk before the first
        // token; on a big MoE it is seconds, and it is the phase whose result
        // decides every later read.
        KRK_TIME_HOST("load.expert_warmup");
        warm_experts();
    }

    // ---- the hot ring, before the pool clamp below -------------------------
    // The warmup fills WARM and lets traffic earn VRAM residency. That policy
    // is right while the routed set fits and wrong when it does not: Tiel's
    // 256 experts x 40 layers is 22.1 GiB against a 12-13 GiB pool, so the
    // eviction scan never has a stable core -- a token promotes experts the
    // next token's routing evicts. The ring is that core, pinned and charged
    // to a budget the LRU window does not see. Running it BEFORE the clamp is
    // what keeps the arithmetic honest: the clamp then sizes the pool from the
    // bytes that are free once the ring has taken its own.
    pin_hot_experts();

    // ---- the expert pool must not fill the card ---------------------------
    // Its budget is decided before the KV cache and before this engine's own
    // workspaces exist, so it can name every byte that was free at that moment
    // and leave the run with 0.00 GiB free. That is not merely slow, it is
    // pathological, and it is measured: on Tiel-Coder-35B-A3B (Q5_K_XL, 256
    // experts, top-8, 22.1 GiB routed, ctx 4096, 16 tokens) a 2.2 MiB pool cell
    // costs 30 ms to allocate once the card is full (3.1 tok/s) and 0.29 ms
    // with 2.27 GiB free (15.8 tok/s) -- the allocator, not the model, and the
    // same weights produce the same text either way. On Windows a full card is
    // also what stalls the compositor while the driver evicts pages.
    //
    // This runs after the warmup has filled the tiers, so `resident_bytes()` is
    // a fact and so is `device_free_bytes()`. The cap allows growth only into
    // room that is really free beside the floor: past it the cache evicts and
    // reuses instead of allocating, which is the path it is built for.
    // The CPU backend has no device/host split (every byte is host RAM), so a
    // VRAM floor there is a cap on nothing -- measured: it clamped the suite's
    // CPU MoE fixtures to a zero-byte expert cache.
    if (be_->has_device_memory() && be_->device_total_bytes() > 0 &&
        model_.experts().vram_budget() > 0) {
        const size_t total = be_->device_total_bytes();
        const size_t floor = krk::vram_floor_bytes(total);
        const size_t free_now = be_->device_free_bytes();
        // Unpinned bytes only: the ring is already charged to pin_budget_, and
        // counting it here would hand the pool room the pins are holding.
        const size_t held = model_.experts().resident_bytes();
        const size_t ring = model_.experts().pin_used_bytes();
        const size_t live = held > ring ? held - ring : 0;
        const size_t room = free_now > floor ? free_now - floor : 0;
        const size_t cap = live + room;
        const size_t before = model_.experts().vram_budget();
        if (cap < before) {
            model_.experts().set_vram_budget(cap);
            KRK_INFO("expert cache: budget clamped %.0f -> %.0f MiB (%.0f MiB "
                     "resident plus %.0f MiB really free beside a %.0f MiB "
                     "floor). The budget is planned before the KV cache and the "
                     "workspaces exist; this one is measured after them.",
                     static_cast<f64>(before) / 1048576.0,
                     static_cast<f64>(cap) / 1048576.0,
                     static_cast<f64>(live) / 1048576.0,
                     static_cast<f64>(room) / 1048576.0,
                     static_cast<f64>(floor) / 1048576.0);
        }
    }

    log_vram_ledger("after init");

    // A speculative round runs the block up to `--draft-tokens` times per
    // target forward, and the block is ONE layer, so its experts fit even when
    // the stack's routed set does not. Admitting them once and pinning them is
    // what keeps the propose pass off the promotion path (measured: 11% of the
    // block's acquires promoted while the stack churned). KRK_MTP_PIN=0 is the
    // A/B arm; a set that does not fit is reported and left on the normal
    // policy rather than thrashing a pin it cannot defend.
    // Only a routed block has anything to pin; a dense block's FFN is loaded
    // with the rest of the block and never pages.
    if (mtp_spec_ && model_.mtp().present && model_.mtp().n_expert > 0) {
        // Pinning the block's experts is a load-time cost that only exists when
        // speculation is on, so it needs its own row rather than hiding in the
        // wall clock.
        KRK_TIME_HOST("load.mtp_pin");
        mtp_pin_experts();
    }

    // An external draft head named in the config loads here: the KV cache and
    // the workspaces exist, so its hidden capture can be sized, and the first
    // forward has not run. A head that was asked for and cannot load is a
    // refusal -- silently running plainly would answer a different command --
    // and the CLI's own call after init then finds it already loaded.
    if (!cfg_.draft_head.empty()) {
        KRK_TIME_HOST("load.draft_head");
        std::string why;
        if (!load_draft_head(cfg_.draft_head, &why)) {
            if (err) *err = why;
            shutdown();
            return false;
        }
    }

    // One 1-token pass at load, doing up to two jobs.
    //
    //   cfg_.warmup    pages the weights in (the original purpose);
    //   wants_prepare  a backend with a ONE-TIME weight conversion does it
    //                  here instead of inside the first measured step.
    //
    // The second job is not cosmetic. The virtual NPU packs each weight into
    // the device's int8 layout once, at 2.3 ms each; while that cost sat in the
    // first decode step, a 16-token run measured 0.74x against the f32 arm and
    // the same code measured 2.51x at 128 tokens. The cost is the same either
    // way -- what changed is whether a benchmark can see it. Charged to the
    // load phase it prints on the load line, where a reader expects one-time
    // costs, and the decode number becomes length-independent.
    //
    // The pass is a real forward, so it needs the KV cursor put back: it ran at
    // position 0 and the caller's prompt also starts at 0. `kv_rollback(0)` is
    // the same "new sequence" primitive generate() uses, including the
    // recurrent-model reset.
    const bool want_prepare = be_->wants_prepare();
    if (cfg_.warmup || want_prepare) {
        KRK_TIME_HOST(want_prepare ? "load.prepare_pass" : "load.warmup_pass");
        const i32 t = tok_.bos() >= 0 ? tok_.bos() : 0;
        const Timer w;
        if (want_prepare) be_->set_prepare(true);
        const bool ok = forward(&t, 1, 0, true);
        if (want_prepare) be_->set_prepare(false);
        if (!ok) {
            if (err) *err = "KV promotion refused during the engine warmup";
            shutdown();
            return false;
        }
        be_->sync();
        if (kv_pos_ != 0 && !kv_rollback(0)) {
            if (err) *err = "KV promotion refused before the first prompt";
            shutdown();
            return false;
        }
        KRK_INFO("%s pass: %.1f ms%s",
                 want_prepare ? "prepare" : "warmup", w.ms(),
                 want_prepare ? " (one-time weight conversion, outside the "
                                "timed run)"
                              : "");
    }

    // LAST, and deliberately so: the warmup pass above promotes experts, and a
    // promotion rewrites a slot's device pointers. A table published before it
    // would name addresses the cache has already released.
    publish_expert_table();
    return true;
}

void Engine::shutdown() {
    if (moe_predict_enabled_) {
        const u64 candidates = moe_predictor_.candidates();
        const u64 hits = moe_predictor_.candidate_hits();
        KRK_INFO("expert predictor: %llu hint set(s), %llu expert(s) staged, "
                 "%llu/%llu candidate hits (%.1f%% precision); routes unchanged",
                 static_cast<unsigned long long>(moe_predictor_.predictions_issued()),
                 static_cast<unsigned long long>(moe_predictor_.staged()),
                 static_cast<unsigned long long>(hits),
                 static_cast<unsigned long long>(candidates),
                 candidates ? 100.0 * static_cast<f64>(hits) /
                                 static_cast<f64>(candidates)
                            : 0.0);
    }
    // Before anything is released: the other half of the ledger. A workspace
    // taken after init (a per-forward scratch, a promoted dense layer) moves
    // `workspaces` or `unaccounted` here, which is what makes the two prints a
    // measurement rather than a snapshot.
    log_vram_ledger("at teardown");
    // The draft shares the backend; it must release its allocations before
    // this engine tears the backend's other buffers down.
    unload_dflash();
    unload_draft_head();
    unload_draft();
    if (!be_) {
        model_.unload();
        return;
    }
    be_->sync();
    // The routing histograms, if a tracking run kept them on the device: this
    // is the read that replaces one blocking readback per layer, and it happens
    // before anything is released. No-op on every other run.
    scan_dev_fold();
    // The device-softmax drift, if that arm ran: rows compared, elements that
    // differ, and the largest probability difference. `--greedy` text is the
    // real verdict, but a nonzero element count with identical text says the
    // drift is below the decoder's resolution, which is a different (and much
    // cheaper) answer than "it changed the model".
    if (dp_rows_) {
        KRK_INFO("device plan (KRK_DEV_PLAN=1): %llu layer(s) compared, "
                 "%llu id(s) and %llu weight(s) differ from the host "
                 "(%llu at one ulp), %llu layer(s) ran the device plan",
                 static_cast<unsigned long long>(dp_rows_),
                 static_cast<unsigned long long>(dp_sel_diff_),
                 static_cast<unsigned long long>(dp_wt_diff_),
                 static_cast<unsigned long long>(dp_wt_1ulp_),
                 static_cast<unsigned long long>(dp_used_));
    }
    if (dp_launch_layers_) {
        KRK_INFO("device dispatch (KRK_DEV_PTR=1): %llu layer(s) ran as one "
                 "launch through the expert table, %llu chunk(s) kept the "
                 "per-expert path (a row set above the tiled kernel's small-row "
                 "threshold), %llu fell back",
                 static_cast<unsigned long long>(dp_launch_layers_),
                 static_cast<unsigned long long>(dp_chunk_skips_),
                 static_cast<unsigned long long>(dp_launch_skips_));
    }
    if (moe_batch_layers_) {
        // The counter that makes "the batch was on and the numbers did not move"
        // answerable: engaged layers against layers that fell back to the
        // per-expert loop (a shape the launches cannot express, or an arm that
        // measures the loop).
        KRK_INFO("moe batch: %llu layer(s) ran the batched decode path, %llu "
                 "kept the per-expert loop",
                 static_cast<unsigned long long>(moe_batch_layers_),
                 static_cast<unsigned long long>(moe_batch_skips_));
    }
    if (ds_rows_) {
        KRK_INFO("router softmax drift (KRK_DEVICE_SOFTMAX=1): %llu row(s), "
                 "%llu/%.0f element(s) differ, max |dP| %.3e",
                 static_cast<unsigned long long>(ds_rows_),
                 static_cast<unsigned long long>(ds_diff_),
                 static_cast<f64>(ds_rows_) * static_cast<f64>(
                     model_.cfg().n_expert),
                 ds_max_);
    }
    for (void **p : {&ws_x_, &ws_rexp_, &ws_fexp_, &ws_fscl_, &ws_xn_, &ws_xn2_, &ws_x2_, &ws_gs_, &ws_q_,
                     &ws_qpack_, &ws_k_,
                     &ws_v_, &ws_attn_, &ws_mlaqn_, &ws_mlay_, &ws_mlaa_,
                     &ws_gate_, &ws_up_, &ws_had_, &ws_logits_,
                     &ws_router_, &ws_routerp_, &ws_ids_, &ws_wts_, &ws_off_,
                     &ws_vgate_, &ws_vexp_,
                     &ws_ffn_, &ws_shexp_g_, &ws_g4_post_, &ws_g4_mlp_, &ws_router_in_,
        &ws_xg_, &ws_gateg_, &ws_upg_, &ws_plan_, &ws_alpha_,
                     &ws_scale_, &ws_ebatch_, &ws_edown_, &ws_ebscale_,
                     &ws_qkv_, &ws_z_, &ws_h_, &ws_ssm_, &ws_beta_, &ws_agate_,
                     &ws_m2in_,
                     &conv_state_, &rec_state_, &topk_scratch_, &topk_out_,
                     &mtp_kv_k_, &mtp_kv_v_, &mtp_hidden_, &mtp_h_, &mtp_e_,
                     &mtp_cat_, &mtp_gate_, &mtp_sgate_, &mtp_rep_,
                     &mtp_rec_snap_, &mtp_conv_snap_,
                     &scan_hits_dev_, &scan_mass_dev_}) {
        if (*p) {
            be_->release(*p);
            *p = nullptr;
        }
    }
    // The KV tier caches free in their destructors, which run after this
    // function has already deleted the backend. Release them here, while
    // be_ is still valid, or their destructors call into freed memory.
    kvt_k_.detach();
    kvt_v_.detach();
    mtp_probe_ = false;
    mtp_spec_ = false;
    mtp_probe_report_ = false;
    mtp_captured_ = 0;
    mtp_fed_ = 0;
    mtp_kv_dim_ = 0;
    mtp_tok_.clear();
    host_free(logits_host_);
    logits_host_ = nullptr;
    if (model_.promoted_layers())
        KRK_INFO("weight residency: %llu layer(s) promoted resident on first "
                 "use (%.2f GiB, %.0f ms of one-off staging) -- only what did "
                 "not fit stayed streamed",
                 static_cast<unsigned long long>(model_.promoted_layers()),
                 static_cast<f64>(model_.promoted_bytes()) / 1073741824.0,
                 model_.promote_ms());
    if (model_.stage_calls())
        KRK_INFO("weight stream: %llu layer staging(s), %.2f GiB %s, "
                 "%.0f ms (%.2f GB/s) = drain %.0f + read %.0f + copy %.0f ms",
                 static_cast<unsigned long long>(model_.stage_calls()),
                 static_cast<f64>(model_.stage_bytes()) / 1073741824.0,
                 model_.stream_host_bytes() ? "from pinned host" : "from the file",
                 model_.stage_ms(),
                 static_cast<f64>(model_.stage_bytes()) /
                     (model_.stage_ms() > 0.0 ? model_.stage_ms() / 1000.0 : 1.0) /
                     1073741824.0,
                 model_.stage_sync_ms(), model_.stage_read_ms(),
                 model_.stage_xfer_ms());
    if (model_.ring_slots() > 1)
        KRK_INFO("weight stream: %llu of %llu staging(s) consumed a copy that was "
                 "already in flight (%.0f ms of enqueue), %llu had to fill "
                 "synchronously",
                 static_cast<unsigned long long>(model_.ahead_copies()),
                 static_cast<unsigned long long>(model_.stage_calls()),
                 model_.ahead_wait_ms(),
                 static_cast<unsigned long long>(model_.ahead_cold()));
    model_.unload();
    be_ = nullptr;
}

void Engine::norm_bias_add(f32 *dst, const f32 *bias, i32 n_tok, i64 width) {
    if (!model_.cfg().norm_bias || !bias || !dst) return;
    for (i32 t = 0; t < n_tok; t++)
        be_->add_inplace(dst + static_cast<i64>(t) * width, bias, width);
}

void Engine::dump_row(const char *what, i32 layer, const void *buf, i64 width,
                      i64 rows, const f32 *row_scale) {
    if (dump_path_.empty() || rows <= 0) return;
    be_->sync();
    // Two shapes of trace, because one row cannot answer every question.
    //
    // KRK_DUMP (this mode): the last row, one value per element. A device run
    // and a --cpu run of the same prompt then walk the same lines in the same
    // order and the first line that differs names the broken stage.
    //
    // KRK_DUMP_FULL=1: a SUMMARY of the whole tensor instead -- how many
    // elements are non-finite, in how many rows, the peak, and where the first
    // one is. The single-row trace is structurally blind to a defect that only
    // touches some rows: attention is the one cross-row op in a layer, so a
    // NaN in one cached key row poisons every output row while the row the
    // trace prints stays finite. That is exactly how the nemotron NaN hid --
    // finite q and k on the printed row, all-NaN attn.out -- and this mode is
    // what makes the difference visible in one run.
    static const bool full = [] {
        const char *e = std::getenv("KRK_DUMP_FULL");
        return e && e[0] != '0';
    }();
    i64 row0 = rows - 1;
    if (const char *rs = std::getenv("KRK_DUMP_ROW")) {
        const long long v = std::atoll(rs);
        if (v >= 0 && v < rows) row0 = v;
    }
    const i64 nrow = full ? rows : 1;
    const i64 off = full ? 0 : row0 * width;
    std::vector<f32> vals(static_cast<size_t>(nrow) * static_cast<size_t>(width));
    // One download for the whole span: the activation buffer is [rows, width]
    // contiguous, which is what lets the offset above be an element count.
    be_->download_f32(vals.data(),
                      act_at(const_cast<void *>(buf), be_->act_size(), off),
                      static_cast<i64>(vals.size()));
    // A row the store divided by a power of two (Backend::silu_mul's row-scaled
    // form) is dumped at TRUE scale. The stored bytes are f16(p / 2^k) and the
    // divide is exact, so multiplying by 2^k returns the value the --cpu arm's
    // f32 row holds to f16 precision -- which is what keeps this trace naming
    // the first stage that really differs instead of reporting a 2^k scale gap
    // at a stage that is right.
    if (row_scale) {
        std::vector<f32> scl(static_cast<size_t>(rows));
        be_->download(scl.data(), row_scale,
                      static_cast<size_t>(rows) * sizeof(f32));
        for (i64 r = 0; r < nrow; r++) {
            const f32 s = scl[static_cast<size_t>(row0 + r)];
            for (i64 i = 0; i < width; i++)
                vals[static_cast<size_t>(r * width + i)] *= s;
        }
    }
    std::FILE *f = std::fopen(dump_path_.c_str(), "a");
    if (!f) return;
    if (full) {
        i64 nf = 0, bad_rows = 0, first = -1;
        f64 peak = 0.0, sumsq = 0.0;
        for (i64 r = 0; r < nrow; r++) {
            bool row_bad = false;
            for (i64 i = 0; i < width; i++) {
                const f32 v = vals[static_cast<size_t>(r * width + i)];
                // A NaN fails every comparison, so this is the finite test
                // without an isfinite() that the device type would not have.
                if (!(v == v) || v > 3.0e38f || v < -3.0e38f) {
                    nf++;
                    row_bad = true;
                    if (first < 0) first = r * width + i;
                    continue;
                }
                const f64 a = std::fabs(static_cast<f64>(v));
                if (a > peak) peak = a;
                sumsq += a * a;
            }
            if (row_bad) bad_rows++;
        }
        const f64 rms = sumsq > 0.0
                            ? std::sqrt(sumsq / static_cast<f64>(nrow * width))
                            : 0.0;
        std::fprintf(f,
                     "%s L%02d N=%lld W=%lld nf=%lld badrows=%lld first=%lld "
                     "peak=%.6g rms=%.6g\n",
                     what, layer, static_cast<long long>(rows),
                     static_cast<long long>(width), static_cast<long long>(nf),
                     static_cast<long long>(bad_rows),
                     static_cast<long long>(first), peak, rms);
        std::fclose(f);
        return;
    }
    std::fprintf(f, "%s L%02d n=%lld", what, layer, static_cast<long long>(rows));
    for (i64 i = 0; i < width; i++)
        std::fprintf(f, " %.7g", vals[static_cast<size_t>(i)]);
    std::fputc('\n', f);
    std::fclose(f);
}

void Engine::trace_rows(const char *what, i32 layer, const void *buf,
                        i64 width, i64 rows) {
    if (!trace_hash_on_ || rows <= 0 || width <= 0) return;
    be_->sync();
    // EVERY row: a chunk of N rows must leave the same bytes as N single-row
    // calls, and the row that differs is usually not the last one.
    std::vector<f32> vals(static_cast<size_t>(rows) *
                          static_cast<size_t>(width));
    be_->download_f32(vals.data(),
                      act_at(const_cast<void *>(buf), be_->act_size(), 0),
                      static_cast<i64>(vals.size()));
    // Folded per (stage, layer). Within one stage the rows arrive in prompt
    // order whichever way the chunks grouped them; the stages are combined in
    // key order by prefill_trace_hash().
    const std::string key = format("%s L%02d", what, layer);
    u64 h = trace_stage_[key];
    for (f32 v : vals) {
        u32 bits;
        std::memcpy(&bits, &v, sizeof bits);
        h ^= bits;
        h *= 1099511628211ull;
    }
    trace_stage_[key] = h;
}

u64 Engine::prefill_trace_hash() const {
    // KRK_TRACE_STAGES=1 prints the per-(stage, layer) hashes the folded value
    // is made of, one per line, in the same key order. The folded hash answers
    // "did anything move"; this answers "what moved first", which is the
    // question a chunk-invariance failure actually asks. It is a diagnostic
    // and never affects a run that does not set it.
    static const bool stages_on = [] {
        const char *e = std::getenv("KRK_TRACE_STAGES");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    if (stages_on) {
        std::fprintf(stderr, "[stages] %zu key(s)\n", trace_stage_.size());
        for (const auto &kv : trace_stage_)
            std::fprintf(stderr, "[stage] %s %016llx\n", kv.first.c_str(),
                         static_cast<unsigned long long>(kv.second));
    }
    u64 h = 1469598103934665603ull;
    for (const auto &kv : trace_stage_) { // std::map iterates in key order
        for (char c : kv.first) {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ull;
        }
        for (int b = 0; b < 8; b++) {
            h ^= (kv.second >> (b * 8)) & 0xffull;
            h *= 1099511628211ull;
        }
    }
    return h;
}

// prism.hadamard (Backend::hadamard_act): dst <- the transform of src.
// Called only when the plan folded the site, so dst is always ws_had_ except
// for the in-place inverse right after the embedding lookup.
void Engine::had_xform(void *dst, const void *src, i64 rows, i64 width,
                       bool inverse, bool gdn_perm) {
    const HadamardPlan &hp = model_.had();
    Backend::HadDesc d;
    d.n = width;
    d.block = hp.block;
    d.signs = model_.had_signs(width);
    d.signs_first = !inverse;
    if (gdn_perm && hp.gdn_v_grouped) {
        const ModelConfig &mc = model_.cfg();
        const i64 nv = mc.ssm_dt_rank;
        const i64 nk = mc.ssm_n_group;
        // [hd, nk, rep] -> [hd, rep, nk] over the delta-rule output's value
        // heads; hd = width / n_v, rep = n_v / n_k. Checked rather than
        // assumed so a file whose geometry disagrees skips the perm loudly
        // (the load-time check refuses that file anyway).
        if (nv > 0 && nk > 0 && nv % nk == 0 && width % nv == 0) {
            d.perm_hd = width / nv;
            d.perm_nk = nk;
            d.perm_rep = nv / nk;
        }
    }
    be_->hadamard_act(dst, src, rows, d);
}

// ---- qwen4exp hyper-connections: the two ends of every sub-block ----------
//
// The narrow path hands a block one vector and takes `x += f(x)` back. Here the
// residual is a LANE STACK in ws_hc_, and the block is handed a mixed row and
// nothing else -- so the block's own output is that row minus the mix, and it
// is scattered into every lane weighted by the injection. See hc.hpp.
void Engine::hc_enter(const f32 *norm_w, const QuantTensor &down,
                      const QuantTensor &up, const QuantTensor &inject, i32 n) {
    const i64 ne = n_embd_;
    const i64 hc = hc_.hc;
    const i64 wide = hc * ne;
    const f32 eps = model_.cfg().rms_eps;
    // Per-lane RMSNorm: rmsnorm_grouped() already means exactly this shape (one
    // weight vector per group, rows of `width` inside a token's block), which
    // is why the lane stack needs no norm kernel of its own.
    be_->rmsnorm_grouped(ws_hc_n_, ws_hc_, norm_w, n, hc, ne, wide, eps);
    dump_row("hc.n", -1, ws_hc_n_, wide, n);
    gemm_w(ws_hc_lo_, ws_hc_n_, down, hc_.low_rank, wide, n);
    // The 1/hc scale belongs to the BOTTLENECK (the reference is
    // `silu(scale(w_down @ xn, 1/hc))`), NOT to xn. Scaling xn instead is
    // equivalent for lo -- a matmul is linear, so the factor commutes -- and
    // wrong for the three other things xn feeds: the injection projection
    // (`w_inject @ xn`, then divided by hc again in the combine), the mean
    // inside hc_mix (which divides by hc itself), and so the whole block input.
    // Every sub-block of every layer then ran on a 1/hc-scaled mix, which is
    // the fault that survived the delta-net gate fix: fluent, evenly scaled
    // garbage rather than a crash, and invisible to the lane-invariance test
    // (a uniform factor keeps identical lanes identical).
    be_->scale_act(ws_hc_lo_, 1.0f / static_cast<f32>(hc), n, hc_.low_rank,
                   hc_.low_rank);
    be_->silu_act(ws_hc_lo_, static_cast<i64>(n) * hc_.low_rank);
    gemm_w(ws_hc_g_, ws_hc_lo_, up, wide, hc_.low_rank, n);
    dump_row("hc.g", -1, ws_hc_g_, wide, n);
    gemm_w(ws_hc_inj_, ws_hc_n_, inject, hc, wide, n);
    dump_row("hc.inj", -1, ws_hc_inj_, hc, n);
    be_->hc_mix(ws_hc_mix_, ws_hc_n_, ws_hc_g_, n, hc, ne);
    dump_row("hc.mix", -1, ws_hc_mix_, ne, n);
    // The block reads the mixed row (ws_xn_, which every projection below
    // consumes) and accumulates onto a copy of it in ws_x_.
    be_->copy_act(ws_xn_, ws_hc_mix_, static_cast<i64>(n) * ne);
    be_->copy_act(ws_x_, ws_hc_mix_, static_cast<i64>(n) * ne);
}

void Engine::hc_begin(const LayerWeights &L, bool attn, i32 n) {
    if (model_.cfg().hc_sinkhorn) {
        hc_sk_enter(L, attn, n);
        return;
    }
    if (attn)
        hc_enter(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, n);
    else
        hc_enter(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, n);
}

void Engine::hc_end(i32 n) {
    if (model_.cfg().hc_sinkhorn)
        hc_sk_leave(LayerWeights{}, false, n);
    else
        hc_leave(n);
}

void Engine::hc_leave(i32 n) {
    const i64 ne = n_embd_;
    const i64 hc = hc_.hc;
    // ws_x_ is `mixed + block_out`, so negating the saved mix and adding it
    // leaves the block's own contribution -- exact, and it never mixes a
    // residual sum into the weights the combine applies.
    be_->scale_act(ws_hc_mix_, -1.0f, n, ne, ne);
    be_->add_inplace(ws_x_, ws_hc_mix_, static_cast<i64>(n) * ne);
    // ws_hc_inj_, NOT a weight tensor: the injection is the mixer's per-lane
    // output, written by hc_enter() into an activation buffer of the backend's
    // own type. Passing the [wide, hc] WEIGHT through here instead read f16
    // bytes as f32 lanes, which is the kind of wrong that still runs.
    be_->hc_combine(ws_hc_, ws_x_, ws_hc_inj_, n, hc, ne);
}

void Engine::hc_head_mix(i32 row, const f32 *norm_w, const QuantTensor &down,
                         const QuantTensor &up) {
    const size_t as = be_->act_size();
    const i64 ne = n_embd_;
    const i64 hc = hc_.hc;
    const i64 wide = hc * ne;
    const void *lanes = act_at(ws_hc_, as, static_cast<i64>(row) * wide);
    be_->rmsnorm_grouped(ws_hc_n_, lanes, norm_w, 1, hc, ne, wide,
                         model_.cfg().rms_eps);
    gemm_w(ws_hc_lo_, ws_hc_n_, down, hc_.low_rank, wide, 1);
    // Same rule as hc_enter: the 1/hc belongs to the bottleneck. The head mixer
    // has no injection, so only the mix's own mean is at stake here -- it is
    // still a factor of hc on the row the output head reads.
    be_->scale_act(ws_hc_lo_, 1.0f / static_cast<f32>(hc), 1, hc_.low_rank,
                   hc_.low_rank);
    be_->silu_act(ws_hc_lo_, hc_.low_rank);
    gemm_w(ws_hc_g_, ws_hc_lo_, up, wide, hc_.low_rank, 1);
    be_->hc_mix(ws_xn_, ws_hc_n_, ws_hc_g_, 1, hc, ne);
}

const void *Engine::kv_plane_debug(i32 which, i32 layer) {
    // The same resolution the forward does: a shared-KV layer has no plane of
    // its own, so this is what any consumer of "layer l's cache" must use.
    if (layer < 0 || layer >= model_.cfg().n_layer) return nullptr;
    const i32 src = model_.kv_source_layer(layer);
    return which == 0 ? kvt_k_.base(src) : kvt_v_.base(src);
}

const f32 *Engine::hc_debug_lanes(i32 *hc_out, i64 *width_out) const {
    if (hc_out) *hc_out = 0;
    if (width_out) *width_out = 0;
    // Host-visible only: the lane stack lives in device memory on a GPU backend
    // and a pointer into VRAM would read as garbage. The suite runs on the CPU
    // backend, which is where this hook is used.
    if (!hc_active_ || !ws_hc_ || be_->act_size() != sizeof(f32)) return nullptr;
    if (hc_out) *hc_out = hc_.hc;
    if (width_out) *width_out = n_embd_;
    return static_cast<const f32 *>(ws_hc_);
}

// ---- qwen4exp per-layer embedding (PLE) ----------------------------------
//
// Runs on the layers `ple.layers` names, before that layer's hyper-connection
// mixer, and writes straight into the lane stack:
//
//   rows  = hash(token n-gram window)               (host, 64-bit)
//   emb   = table[rows]                             (mmap'd, never uploaded)
//   key   = ple_key * emb,  value = ple_value * emb
//   gate  = sigmoid(sign(s) * sqrt(|s|)),  s = key.query / sqrt(n_embd)
//   x     = value broadcast over lanes, times gate, lane-normed
//   conv  = depthwise, kernel K, dilated by the n-gram size, over the TOKENS
//   res_hc += x + silu(conv)
//
// The conv is where a chunked prefill and a token-at-a-time decode have to
// agree: its window is the last (K-1)*ngram rows this layer produced, kept per
// layer in ple_state_, so the history is state and not a function of the chunk.
void Engine::ple_block(const LayerWeights &L, i32 l, i32 n, i64 pos0) {
    const i64 ne = n_embd_;
    const i64 hc = hc_.hc;
    const i64 wide = hc * ne;
    const i32 nh = ple_.n_heads();
    const size_t as = be_->act_size();
    const f32 eps = model_.cfg().rms_eps;

    // The rows: 64-bit hash on the host. No kernel in this engine carries a
    // 64-bit integer, and the table is not in VRAM to be indexed by one anyway.
    ple_rows_for(ple_, ple_seq_.data(), pos0, n, ple_rows_.data());
    ple_rows_n_ = n;

    // Gather straight out of the mapping, head-slowest inside a token, and
    // stage one pass of embeddings. This is the only weight on the hot path
    // that is read from the file rather than from the backend.
    for (i32 i = 0; i < n; i++) {
        ple_gather(ple_emb_.data() + static_cast<i64>(i) * ne,
                   model_.ple_table(), model_.ple_table_type(),
                   model_.ple_row_bytes(), ple_.head_dim, nh,
                   ple_rows_.data() + static_cast<i64>(i) * nh);
    }
    if (as == 2) {
        ple_emb16_.resize(ple_emb_.size());
        for (size_t i = 0; i < ple_emb_.size(); i++)
            ple_emb16_[i] = fp32_to_fp16(ple_emb_[i]);
        be_->upload(ws_ple_emb_, ple_emb16_.data(),
                    ple_emb_.size() * sizeof(u16));
    } else {
        be_->upload(ws_ple_emb_, ple_emb_.data(),
                    ple_emb_.size() * sizeof(f32));
    }

    // Two projections off the same row: the key widens it to the lane stack,
    // the value keeps the embedding width (the reference's ple_key/ple_value).
    gemm_w(ws_ple_key_, ws_ple_emb_, L.ple_key, wide, ne, n);
    gemm_w(ws_ple_val_, ws_ple_emb_, L.ple_value, ne, ne, n);
    // Both norms group over one lane, with a [n_embd, hc] weight: the key is
    // normed on its own, the query is the lane stack normed in place of a copy.
    be_->rmsnorm_grouped(ws_ple_key_, ws_ple_key_, L.ple_norm_key, n, hc, ne,
                         wide, eps);
    be_->rmsnorm_grouped(ws_ple_qry_, ws_hc_, L.ple_norm_query, n, hc, ne, wide,
                         eps);
    // `ne`, not `wide`: the gathered embedding is ONE lane wide (ple_key widens
    // it), and dumping it at `wide` read 4x past the buffer -- garbage in the
    // trace, and with --chunk 1 an out-of-bounds download that failed the
    // hipMemcpy (hipErrorInvalidValue) on the first PLE layer.
    dump_row("ple.emb", l, ws_ple_emb_, ne, n);
    dump_row("ple.key", l, ws_ple_key_, wide, n);
    dump_row("ple.qry", l, ws_ple_qry_, wide, n);
    be_->ple_gate(ws_ple_gate_, ws_ple_key_, ws_ple_qry_, n, hc, ne);
    dump_row("ple.gate", l, ws_ple_gate_, hc, 1);
    // The value is lane-independent: one scalar per lane broadcasts it across
    // that lane's whole row, which is what makes the gate per-lane rather than
    // per-element.
    be_->lane_bcast_mul(ws_ple_gated_, ws_ple_val_, ws_ple_gate_, n, hc, ne);
    // The lane norm feeds the CONV and nothing else: the reference is
    // `hidden + gated + silu(conv(grouped_norm(gated)))`, where `gated` is the
    // raw `value * gate` and a separate tensor carries the normalized copy.
    // Normalizing in place put a per-lane unit-RMS row on the residual instead
    // of `value * gate`, which is the FIRST divergence measured against
    // src/models/qwen4exp.cpp on the REAP-288 file: at layer 1, token 0, the
    // reference lane stack (normalized through hc_attn_norm) scores corr 0.373
    // against the old dump and 0.9999 recomputed with this split.
    be_->rmsnorm_grouped(ws_ple_norm_, ws_ple_gated_, L.ple_norm_conv, n, hc,
                         ne, wide, eps);

    // The conv window: this layer's history, then this pass's rows.
    if (ple_hist_ > 0) {
        void *state = act_at(ple_state_, as, static_cast<i64>(l) * ple_hist_ * wide);
        be_->copy_act(ws_ple_pad_, state, ple_hist_ * wide);
        be_->copy_act(act_at(ws_ple_pad_, as, ple_hist_ * wide), ws_ple_norm_,
                      static_cast<i64>(n) * wide);
        // The tail of the window is this layer's history for the next pass.
        be_->copy_act(state, act_at(ws_ple_pad_, as, static_cast<i64>(n) * wide),
                      ple_hist_ * wide);
    } else {
        be_->copy_act(ws_ple_pad_, ws_ple_norm_, static_cast<i64>(n) * wide);
    }
    dump_row("ple.gated", l, ws_ple_gated_, wide, n);
    dump_row("ple.convin", l, ws_ple_norm_, wide, n);
    be_->conv1d_dw(ws_ple_conv_, ws_ple_pad_, L.ple_conv1d.data, n, wide,
                   ple_.conv_kernel, ple_.ngram);
    be_->silu_act(ws_ple_conv_, static_cast<i64>(n) * wide);
    // The reference returns hidden + gated + conv_out; the lane stack is
    // `hidden`, and this block is the only writer of it before the mixer.
    be_->add_inplace(ws_hc_, ws_ple_gated_, static_cast<i64>(n) * wide);
    be_->add_inplace(ws_hc_, ws_ple_conv_, static_cast<i64>(n) * wide);
}

// ---------------------------------------------------------------------------
// qwen4exp QSA indexer
// ---------------------------------------------------------------------------

bool Engine::qsa_select_block(const LayerWeights &L, i32 l, i32 n, i64 pos0,
                              const void *xn) {
    if (!qsa_active_ || L.qsa_ratio <= 0 || !xn) return false;
    if (l < 0 || l >= static_cast<i32>(qsa_state_.size())) return false;
    // `qsa_rows_` is this PASS's cell list per ROW, so the bound is the row
    // count and not the layer index. Testing `l` here read as a live check and
    // was one at the default chunk only by luck: at `--chunk 1` the buffer holds
    // one row, so layer 1 refused and the indexer never ran -- the layer fell
    // back to dense attention over the whole cache, silently, which is exactly
    // the cost the indexer exists to avoid.
    if (n > static_cast<i32>(qsa_rows_.size())) return false;
    const ModelConfig &mc = model_.cfg();
    QsaState &st = qsa_state_[static_cast<size_t>(l)];
    const i32 dim = qsa_dim_;
    const i32 nh = qsa_heads_;
    const i32 r = L.qsa_ratio;
    const i64 n_kv = pos0 + n;
    if (dim <= 0 || nh <= 0 || n <= 0) return false;
    if (n_kv * dim > static_cast<i64>(st.keys.size())) return false;
    if (L.idx_q_norm_h.empty() || L.idx_k_norm_h.empty()) return false;

    // The indexer's own projections, off the same normed input q/k/v read. They
    // are independent of q/k/v (no dependency between the three), so this is a
    // group of five where the backend can express it and five launches where it
    // cannot.
    gemm_w(ws_qsa_k_, xn, L.idx_k_proj, dim, n_embd_, n);
    gemm_w(ws_qsa_q_, xn, L.idx_q_proj, static_cast<i64>(nh) * dim, n_embd_, n);
    // RAW keys onto the host: pooling precedes norm and rotation on this
    // family, so the raw projection output is what has to be kept per position.
    qsa_k_.resize(static_cast<size_t>(n) * dim);
    be_->download_f32(qsa_k_.data(), ws_qsa_k_, static_cast<i64>(n) * dim);
    std::memcpy(st.keys.data() + pos0 * dim, qsa_k_.data(),
                sizeof(f32) * static_cast<size_t>(n) * dim);

    // Past this point nothing is dropped, so the whole selection can be skipped:
    // every visible cell fits in the budget and the layer runs dense causal
    // attention, exactly as the reference does below `top_k + ratio - 1`.
    const i64 budget = static_cast<i64>(qsa_top_k_) + r - 1;
    if (n_kv <= budget) {
        qsa_last_.clear();
        return false;
    }

    QsaConfig cfg;
    cfg.n_head = nh;
    cfg.head_dim = dim;
    cfg.top_k = qsa_top_k_;
    cfg.ratio = r;
    cfg.eps = qsa_eps_;
    cfg.n_rot = mc.rope_dim > 0 ? mc.rope_dim : 0;
    cfg.inv_freq = model_.inv_freq_at(l).empty() ? nullptr
                                                : model_.inv_freq_at(l).data();
    cfg.rope_scale = model_.rope_scale_at(l);
    cfg.rope_neox = mc.rope_neox;

    const i32 n_blocks =
        qsa_pool(st.pooled.data(), st.keys.data(), dim, static_cast<i32>(n_kv), r);
    if (n_blocks > 0) {
        // A block is rotated at its FIRST position: pooling averages cells that
        // sit at different positions, and the reference rotates the pooled row
        // once, at the block's start.
        std::vector<i64> bpos(static_cast<size_t>(n_blocks));
        for (i32 b = 0; b < n_blocks; b++) bpos[static_cast<size_t>(b)] = static_cast<i64>(b) * r;
        qsa_norm_rope(st.pooled.data(), L.idx_k_norm_h.data(), n_blocks, dim,
                      bpos.data(), cfg);
    }
    qsa_q_.resize(static_cast<size_t>(n) * nh * dim);
    be_->download_f32(qsa_q_.data(), ws_qsa_q_, static_cast<i64>(n) * nh * dim);
    {
        std::vector<i64> qpos(static_cast<size_t>(n) * nh);
        for (i32 t = 0; t < n; t++)
            for (i32 h = 0; h < nh; h++)
                qpos[static_cast<size_t>(t) * nh + h] = pos0 + t;
        qsa_norm_rope(qsa_q_.data(), L.idx_q_norm_h.data(), n * nh, dim,
                      qpos.data(), cfg);
    }
    if (n_blocks > 0)
        qsa_scores(st.score.data(), qsa_q_.data(), st.pooled.data(), n, nh,
                   n_blocks, dim);

    bool any_sel = false;
    for (i32 t = 0; t < n; t++) {
        const i64 q = pos0 + t;
        qsa_select(&qsa_rows_[static_cast<size_t>(t)],
                   n_blocks > 0 ? st.score.data() + static_cast<i64>(t) * n_blocks
                                : nullptr,
                   n_blocks, static_cast<i32>(n_kv), q, cfg);
        if (static_cast<i64>(qsa_rows_[static_cast<size_t>(t)].size()) < q + 1)
            any_sel = true;
    }
    qsa_last_ = qsa_rows_[static_cast<size_t>(n - 1)];
    return any_sel;
}

void Engine::qsa_attention(i32 l, i32 n, const AttnDesc &d) {
    const size_t as = be_->act_size();
    const size_t row_bytes = static_cast<size_t>(kv_dim_) * as;
    const void *kbase = kvt_k_.base(l);
    const void *vbase = kvt_v_.base(l);
    if (!kbase || !vbase || !ws_qsa_kc_) return;
    // One call per query row over that row's own compact cells. `causal` with
    // pos0 = cells-1 makes the kernel read exactly that many keys and no window,
    // and every gathered cell is at or before the query's position, so the
    // dense causal kernel over the compact list IS the masked attention the
    // reference builds -- without a mask to carry through the kernel.
    AttnDesc d1 = d;
    d1.n_tok = 1;
    d1.causal = true;
    d1.window = 0;
    d1.layer = 0;
    d1.layer_stride = 0;
    for (i32 t = 0; t < n; t++) {
        const std::vector<i64> &cells = qsa_rows_[static_cast<size_t>(t)];
        if (cells.empty()) continue;
        be_->gather_rows(ws_qsa_kc_, kbase, cells.data(),
                         static_cast<i64>(cells.size()), row_bytes);
        be_->gather_rows(ws_qsa_vc_, vbase, cells.data(),
                         static_cast<i64>(cells.size()), row_bytes);
        d1.pos0 = static_cast<i64>(cells.size()) - 1;
        be_->attention(act_at(ws_attn_, as, static_cast<i64>(t) * q_dim_),
                       act_at(ws_q_, as, static_cast<i64>(t) * q_dim_),
                       ws_qsa_kc_, ws_qsa_vc_, d1);
    }
}

const std::vector<i64> &Engine::qsa_debug_cells(i32 layer) const {
    static const std::vector<i64> none;
    if (!qsa_active_ || layer < 0) return none;
    const auto &L = model_.layers();
    if (layer >= static_cast<i32>(L.size())) return none;
    if (L[static_cast<size_t>(layer)].qsa_ratio <= 0) return none;
    return qsa_last_;
}

i32 Engine::qsa_debug_ratio(i32 layer) const {
    const auto &L = model_.layers();
    if (layer < 0 || layer >= static_cast<i32>(L.size())) return 0;
    return L[static_cast<size_t>(layer)].qsa_ratio;
}

const i64 *Engine::ple_debug_rows(i32 *n_tok, i32 *n_heads) const {
    if (n_tok) *n_tok = 0;
    if (n_heads) *n_heads = 0;
    if (!ple_active_ || ple_rows_.empty()) return nullptr;
    if (n_heads) *n_heads = ple_.n_heads();
    if (n_tok) *n_tok = ple_rows_n_;
    return ple_rows_.data();
}

const f32 *Engine::ple_debug_terms(f32 **conv_in, f32 **conv_out,
                                  i64 *n_tok) const {
    if (conv_in) *conv_in = nullptr;
    if (conv_out) *conv_out = nullptr;
    if (n_tok) *n_tok = 0;
    // Same reason as hc_debug_lanes: these live in device memory on a GPU.
    if (!ple_active_ || !ws_ple_gated_ || be_->act_size() != sizeof(f32))
        return nullptr;
    if (conv_in) *conv_in = static_cast<f32 *>(ws_ple_norm_);
    if (conv_out) *conv_out = static_cast<f32 *>(ws_ple_conv_);
    if (n_tok) *n_tok = ple_rows_n_;
    return static_cast<const f32 *>(ws_ple_gated_);
}

void Engine::head_compute(i32 row) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    if (hc_active_ && mc.hc_sinkhorn) {
        // V4.1 has an output_norm, so the lane stack is collapsed to one row
        // and then normalized like any other family. WHICH pre does the
        // collapsing is the whole of single-pass mHC: a file with no head
        // mixer collapses with the last FFN's pre-mix (and the head's own
        // mixer, when the file has one, is that pre-mix computed from the
        // finished stack).
        const i32 hc = hc_.hc;
        const i64 ne = n_embd_;
        const void *pre = nullptr;
        if (!mc.hc_no_head) {
            // One row through the head's mixer: norm the flat stack, project
            // to hc numbers, affine, sigmoid + eps -- the same three steps the
            // sub-block mixer's `pre` takes, with its own weights.
            const f32 *lanes =
                static_cast<const f32 *>(ws_hc_) + static_cast<i64>(row) * hc * ne;
            be_->rmsnorm(ws_hc_n_, lanes, nullptr, 1, hc * ne, mc.rms_eps);
            gemm_w(ws_hc_fn_, ws_hc_n_, model_.hc_head_fn(), hc, hc * ne, 1);
            hc_head_pre(static_cast<f32 *>(ws_hc_head_),
                        static_cast<const f32 *>(ws_hc_fn_), model_.hc_head_scale(),
                        model_.hc_head_base(), hc, mc.hc_eps);
            pre = ws_hc_head_;
        } else {
            pre = static_cast<const f32 *>(ws_hc_pre_prev_) +
                  static_cast<i64>(row) * hc;
            if (!hc_pre_valid_) {
                // A stack with no layers: lane 0 is the embedding itself.
                pre = nullptr;
            }
        }
        const f32 *lanes =
            static_cast<const f32 *>(ws_hc_) + static_cast<i64>(row) * hc * ne;
        if (pre) {
            be_->hc_sk_pre(ws_hc_out_, lanes, pre, 1, hc, ne);
        } else {
            be_->copy_act(ws_hc_out_, lanes, ne);
        }
        be_->rmsnorm(ws_xn_, ws_hc_out_, model_.out_norm(), 1, n_embd_, mc.rms_eps);
        norm_bias_add(static_cast<f32 *>(ws_xn_), model_.out_norm_bias(), 1, n_embd_);
    } else if (hc_active_) {
        // qwen4exp: this family's final mixer IS its output norm -- there is no
        // output_norm.weight in the file -- so the wide row is mixed through
        // the head bottleneck instead of normalized, and the result lands in
        // ws_xn_ at row 0 (one row is mixed at a time).
        hc_head_mix(row, model_.hc_head_norm(), model_.hc_head_down(),
                    model_.hc_head_up());
    } else {
        const void *src = act_at(ws_x_, as, static_cast<i64>(row) * n_embd_);
        be_->rmsnorm(ws_xn_, src, model_.out_norm(), 1, n_embd_, mc.rms_eps,
                     static_cast<const i32 *>(ws_rexp_) + row);
        norm_bias_add(static_cast<f32 *>(ws_xn_), model_.out_norm_bias(), 1, n_embd_);
    }
    void *xn = ws_xn_;
    if (model_.had().head_folded) {
        had_xform(ws_had_, ws_xn_, 1, n_embd_, false, false);
        xn = ws_had_;
    }
    // MiniCPM3 scales the normed row between `output_norm` and `output`
    // (`cur = ggml_scale(ctx0, cur, float(n_embd_base)/float(n_embd))` with
    // n_embd_base hard-coded 256 in the reference). Applied here, on the row
    // the head reads, so the order is the reference's -- and only to the
    // head: the block's output (the embeddings a caller may ask for) keeps the
    // unscaled normed row, which is what `res->t_embd` holds in the reference
    // too.
    // Both sides of the multiply are dumped, so a trace can show the factor
    // itself rather than only that the head ran: `head.pre` is the normed row
    // the head would read on a family without the scale, `head.in` is what it
    // actually reads. test_minicpm3_scale_arithmetic compares the two.
    dump_row("head.pre", -1, xn, n_embd_, 1);
    if (mc.lm_head_scale != 1.0f)
        be_->scale_act(xn, mc.lm_head_scale, 1, n_embd_, n_embd_);
    dump_row("head.in", -1, xn, n_embd_, 1);
    logit_head(xn);
}

// GPT-2's forward: a clean loop, because its graph is a different shape rather
// than a variant of llama's.
//
//   x = tok_embd[t] + pos_embd[pos]
//   per layer:
//     xn = LayerNorm(x, attn_norm, attn_norm_bias)
//     qkv = xn @ attn_qkv + attn_qkv_bias          [one projection, 3*n_embd]
//     (q, k, v) = qkv[:, 0:E], qkv[:, E:2E], qkv[:, 2E:3E]
//     x += attn(q, k, v) @ attn_output
//     xn = LayerNorm(x, ffn_norm, ffn_norm_bias)
//     x += gelu(xn @ ffn_up + up_bias) @ ffn_down
//   logits = LayerNorm(x, output_norm, output_norm_bias) @ output
//
// Three things the shared path does that this one must not, and one it must:
//
//   * NO rope. Position is a vector added to the embedding, not a rotation of
//     q/k. Rotating here would rotate vectors that already carry position.
//   * NO RMSNorm. LayerNorm subtracts the row mean before scaling and adds a
//     bias after; the mean is what the other families never remove, and a
//     missing bias is a different function, not a rounding difference.
//   * NO silu(gate)*up. There is no gate tensor; the MLP is up -> gelu -> down.
//   * The KV cache is the engine's own, appended through the same kv_append
//     every other family uses, so chunked prefill and decode share one code
//     path and one invariance property.
// LayerNorm, mean-subtracting, with a bias: y = (x - mean) / sqrt(var + eps) * w + b.
//
// The engine's other norm is RMSNorm, which never subtracts the mean, so this
// is a different function rather than a variant: on a 64-wide residual row the
// mean is a real component, and leaving it in shifts every downstream GEMM by a
// vector that changes per token.
//
// Host round-trip on purpose. The op is E wide; the transfers are n*E floats
// against the n*E*3E GEMMs that bracket it, and doing it here means it is
// exactly right on every backend instead of on the one that got a kernel.
void Engine::layernorm(void *out, const void *x, const f32 *w, const f32 *b,
                       i64 rows, i64 n, f32 eps) {
    if (!rows || !n) return;
    std::vector<f32> buf(static_cast<size_t>(rows) * static_cast<size_t>(n));
    be_->sync();
    be_->download_f32(buf.data(), x, static_cast<i64>(buf.size()));
    for (i64 r = 0; r < rows; r++) {
        f32 *row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(n);
        f64 mean = 0.0;
        for (i64 i = 0; i < n; i++) mean += row[i];
        mean /= static_cast<f64>(n);
        f64 var = 0.0;
        for (i64 i = 0; i < n; i++) {
            const f64 d = static_cast<f64>(row[i]) - mean;
            var += d * d;
        }
        var /= static_cast<f64>(n);
        const f32 inv = 1.0f / std::sqrt(static_cast<f32>(var) + eps);
        for (i64 i = 0; i < n; i++) {
            f32 y = (row[i] - static_cast<f32>(mean)) * inv;
            if (w) y *= w[i];
            if (b) y += b[i];
            row[i] = y;
        }
    }
    be_->upload(out, buf.data(), buf.size() * sizeof(f32), 0);
}

// Split a fused [rows, 3E] projection row into three [rows, E] buffers. Each
// row of the source is Q(0..E) | K(E..2E) | V(2E..3E) contiguously.
void Engine::split_qkv(void *q, void *k, void *v, const void *qkv, i64 rows,
                       i64 E) {
    if (!rows || !E) return;
    std::vector<f32> src(static_cast<size_t>(rows) * static_cast<size_t>(3 * E));
    be_->sync();
    be_->download_f32(src.data(), qkv, static_cast<i64>(src.size()));
    std::vector<f32> qb(static_cast<size_t>(rows) * static_cast<size_t>(E));
    std::vector<f32> kb(qb.size()), vb(qb.size());
    for (i64 r = 0; r < rows; r++) {
        const f32 *s =
            src.data() + static_cast<size_t>(r) * static_cast<size_t>(3 * E);
        const size_t off = static_cast<size_t>(r) * static_cast<size_t>(E);
        std::memcpy(qb.data() + off, s, sizeof(f32) * static_cast<size_t>(E));
        std::memcpy(kb.data() + off, s + E,
                    sizeof(f32) * static_cast<size_t>(E));
        std::memcpy(vb.data() + off, s + 2 * E,
                    sizeof(f32) * static_cast<size_t>(E));
    }
    be_->upload(q, qb.data(), qb.size() * sizeof(f32), 0);
    be_->upload(k, kb.data(), kb.size() * sizeof(f32), 0);
    be_->upload(v, vb.data(), vb.size() * sizeof(f32), 0);
}

bool Engine::plain_forward(const i32 *toks, i32 n, i32 pos0) {
    const ModelConfig &mc = model_.cfg();
    const auto &layers = model_.layers();
    const i64 E = mc.n_embd;
    const i64 FF = mc.n_ff;
    // The file's layer_norm_epsilon. This family is LayerNorm, so it is the
    // mean-and-variance epsilon, not the RMS one -- the loader fills the same
    // field from whichever key the file used.
    const f32 eps = mc.rms_eps;
    const QuantTensor &embd = model_.tok_embd();
    const i64 lh = mc.n_head;
    const i64 lkvh = mc.n_head_kv > 0 ? mc.n_head_kv : mc.n_head;
    const i64 hd = mc.head_dim;
    const i64 lq = lh * hd;
    const i64 kvw = lkvh * hd;

    be_->fill0(ws_rexp_, static_cast<size_t>(n) * sizeof(i32));
    be_->embed(ws_x_, embd.data, embd.type, mc.n_vocab, E, toks, n);

    // ---- learned absolute positions ---------------------------------------
    // Dequantized on the host one position at a time and added row-wise. The
    // table is [n_ctx_train, E] and a row is E wide, so this is n small row
    // copies per pass -- not a GEMM, and not worth a device kernel.
    //
    // Only the families that have the table read it. GPT-NeoX, GPT-J and Phi-2
    // carry rotary positions instead and have no `position_embd` tensor at all,
    // so requiring one here would refuse every complete file of those families.
    if (mc.learned_pos) {
        const QuantTensor &pos = model_.pos_embd();
        if (!pos.present()) {
            forward_error_ = "plain forward reached without position_embd";
            return false;
        }
        const size_t row_bytes = dtype_row_bytes(pos.type, E);
        std::vector<f32> rows(static_cast<size_t>(n) * static_cast<size_t>(E));
        const u8 *base = static_cast<const u8 *>(pos.data);
        for (i32 r = 0; r < n; r++) {
            const i64 ps = static_cast<i64>(pos0) + r;
            if (ps < 0 || ps >= mc.n_ctx_train) {
                forward_error_ = "position " + std::to_string(ps) +
                                 " is outside the learned table (" +
                                 std::to_string(mc.n_ctx_train) + " rows)";
                return false;
            }
            dequant_row(pos.type,
                        base + static_cast<size_t>(ps) * row_bytes,
                        rows.data() + static_cast<size_t>(r) * E, E);
        }
        be_->upload(act_at(ws_x2_, be_->act_size(), 0), rows.data(),
                    rows.size() * sizeof(f32), 0);
        be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(n) * E);
    }

    // BLOOM's embedding norm, applied once to the whole [n, E] block before
    // layer 0 rather than per layer. Layers are n rows, so this is one call.
    if (mc.embd_norm) {
        layernorm(ws_x2_, ws_x_, model_.embd_norm(), model_.embd_norm_bias(),
                  n, E, eps);
        be_->copy_act(ws_x_, ws_x2_, static_cast<i64>(n) * E);
    }

    // ---- blocks ------------------------------------------------------------
    AttnDesc d;
    d.n_head = mc.n_head;
    d.n_kv = lkvh;
    d.hd = hd;
    d.scale = 1.0f / std::sqrt(static_cast<f32>(mc.head_dim));
    d.causal = true;
    d.pos0 = pos0;
    // One position's K row is n_kv*hd wide, and both the append and the
    // attention read this number. Leaving it at its 0 default is not a slow
    // path, it is a write at a zero stride.
    d.pos_stride = d.n_kv * d.hd;
    d.kv_int8 = false;
    // ALiBi supplies the position signal on this family, so the bias is carried
    // by every attention call rather than a rotary table.
    d.alibi_max_bias = mc.alibi_max_bias;

    for (i32 l = 0; l < mc.n_layer; l++) {
        const LayerWeights &L = layers[static_cast<size_t>(l)];
        d.n_tok = n;
        d.layer = l;
        d.layer_stride = 0;
        void *kbase = nullptr;
        void *vbase = nullptr;

        // ---- attention ----
        // Falcon-40B normalizes the attention arm with `attn_norm_2` and the
        // FFN arm with `attn_norm`; Falcon-7B and every other family carry one
        // norm, so this is just attn_norm there.
        const f32 *an_w = L.attn_norm_2 ? L.attn_norm_2 : L.attn_norm;
        const f32 *an_b = L.attn_norm_2 ? L.attn_norm_2_bias : L.attn_norm_bias;
        layernorm(ws_xn_, ws_x_, an_w, an_b, n, E, eps);
        // Both arms read the PRE-attention residual row, so the second norm is
        // taken here, before the attention output is added into ws_x_. Taking
        // it after would normalize x + attn and quietly change the graph --
        // the one mistake in this layer that still decodes cleanly.
        const void *ffn_in = ws_xn_;
        if (mc.parallel_two_norm && L.attn_norm_2) {
            layernorm(ws_xn2_, ws_x_, L.ffn_norm, L.ffn_norm_bias, n, E, eps);
            ffn_in = ws_xn2_;
        }
        // q/k/v read the SAME normed row, so the fused projection and the three
        // separate ones are the same arithmetic. The fused form is split into
        // row VIEWS rather than copied, which is what keeps the two paths
        // bit-identical -- there is no second buffer and no stride convention
        // to disagree.
        if (L.wqkv_fused.present()) {
            const i64 rb = static_cast<i64>(
                dtype_row_bytes(L.wqkv_fused.type, L.wqkv_fused.n_in));
            QuantTensor wq = L.wqkv_fused;
            wq.n_out = lq;
            QuantTensor wk = L.wqkv_fused;
            wk.data = static_cast<u8 *>(L.wqkv_fused.data) + lq * rb;
            wk.n_out = kvw;
            QuantTensor wv = L.wqkv_fused;
            wv.data = static_cast<u8 *>(L.wqkv_fused.data) + (lq + kvw) * rb;
            wv.n_out = kvw;
            void *dq[3] = {ws_q_, ws_k_, ws_v_};
            const void *wqkv[3] = {wq.data, wk.data, wv.data};
            const i64 nqkv[3] = {lq, kvw, kvw};
            be_->gemm_group(dq, ws_xn_, wqkv, wq.type, nqkv, E, 3, n);
            // The bias is one vector laid out Q | K | V, i.e. the same split as
            // the weight, so the three parts are three offsets into it.
            // GPT-NeoX and Phi-2 carry one; GPT-J and GPT-2 have none and the
            // null pointer skips all three.
            if (L.qkv_bias) {
                be_->add_bias_rows(ws_q_, L.qkv_bias, lq, n);
                be_->add_bias_rows(ws_k_, L.qkv_bias + lq, kvw, n);
                be_->add_bias_rows(ws_v_, L.qkv_bias + lq + kvw, kvw, n);
            }
        } else {
            gemm_w(ws_q_, ws_xn_, L.wq, lq, E, n);
            gemm_w(ws_k_, ws_xn_, L.wk, kvw, E, n);
            gemm_w(ws_v_, ws_xn_, L.wv, kvw, E, n);
            if (L.q_bias) be_->add_bias_rows(ws_q_, L.q_bias, lq, n);
            if (L.k_bias) be_->add_bias_rows(ws_k_, L.k_bias, kvw, n);
            if (L.v_bias) be_->add_bias_rows(ws_v_, L.v_bias, kvw, n);
        }
        // The same bound the llama-schema path applies (OLMo-1's
        // `attention.clamp_kqv`, see forward_core). No family that takes THIS
        // path declares one today -- it is gpt2/gptneox/gptj/phi2 -- but the
        // loader reads the key off the file rather than off the path, so
        // leaving it out here would be the silent drop this engine refuses to
        // make: a bound read and not applied.
        if (mc.attn_clamp_kqv > 0.0f) {
            const f32 c = mc.attn_clamp_kqv;
            be_->clamp_act(ws_q_, -c, c, static_cast<i64>(n) * lq);
            be_->clamp_act(ws_k_, -c, c, static_cast<i64>(n) * kvw);
            be_->clamp_act(ws_v_, -c, c, static_cast<i64>(n) * kvw);
        }

        // Per-head QK-norm, when the file carries the weights. Some MPT
        // exports do and some do not; the loader sets mc.qk_norm from the
        // presence of the tensors, and skipping it here would leave a file
        // that loads, decodes plausible text and is silently wrong -- and
        // wrong on both backends, since they share this op.
        if (mc.qk_norm)
            be_->qk_norm(ws_q_, ws_k_, L.q_norm, L.k_norm, lh, lkvh, hd, n,
                         mc.rms_eps, mc.qk_norm_wide);

        // Rotary, when the family has it. GPT-2 has none: its positions are
        // already in the embedding by now, and rotating here would apply a
        // second, wrong position signal.
        // No rotary on a family whose position signal is ALiBi: the bias is
        // added to the score inside attention(), and rotating here as well
        // would apply a second, different position signal. llama.cpp lists
        // these archs under LLAMA_ROPE_TYPE_NONE for the same reason.
        if (!mc.learned_pos && mc.alibi_max_bias <= 0.0f)
            be_->rope(ws_q_, ws_k_, lh, lkvh, hd, n, pos0,
                      model_.inv_freq_at(l).data(), model_.rope_scale_at(l),
                      model_.rope_frac_at(l), mc.rope_neox);

        kbase = const_cast<void *>(kvt_k_.base(l));
        vbase = const_cast<void *>(kvt_v_.base(l));
        be_->kv_append(kbase, vbase, ws_k_, ws_v_, d);
        be_->attention(ws_attn_, ws_q_, kbase, vbase, d);
        kvt_k_.end_layer(l);
        kvt_v_.end_layer(l);

        gemm_w(ws_x2_, ws_attn_, L.wo, E, lq, n);
        if (L.wo_bias) be_->add_bias_rows(ws_x2_, L.wo_bias, E, n);
        be_->add_residual(ws_x_, ws_x2_, n, E, nullptr);

        // ---- MLP ----
        // Sequential (GPT-2, GPT-J, Phi-2): the FFN reads the norm of the
        // post-attention stream. Parallel (GPT-NeoX): it reads the SAME normed
        // row the attention read -- still in ws_xn_, untouched since -- so the
        // second norm is skipped rather than recomputed and both contributions
        // land in the residual independently. Two different graphs, and only
        // the file's own flag says which.
        if (!mc.parallel_residual)
            layernorm(ws_xn_, ws_x_, L.ffn_norm, L.ffn_norm_bias, n, E, eps);

        // Gated when the file has a gate tensor, ungated (up -> gelu -> down)
        // when it does not. Reading ffn_up as a gate on an ungated file would
        // halve the intermediate width and apply the wrong activation.
        if (L.wgate.present()) {
            gemm_w(ws_gate_, ffn_in, L.wgate, FF, E, n);
            gemm_w(ws_up_, ffn_in, L.wup, FF, E, n);
            if (L.wup_bias) be_->add_bias_rows(ws_up_, L.wup_bias, FF, n);
            be_->silu_mul(ws_gate_, ws_gate_, ws_up_, static_cast<i64>(n) * FF);
        } else {
            gemm_w(ws_gate_, ffn_in, L.wup, FF, E, n);
            if (L.wup_bias) be_->add_bias_rows(ws_gate_, L.wup_bias, FF, n);
            be_->gelu_act(ws_gate_, static_cast<i64>(n) * FF);
        }
        gemm_w(ws_x2_, ws_gate_, L.wdown, E, FF, n);
        if (L.wdown_bias) be_->add_bias_rows(ws_x2_, L.wdown_bias, E, n);
        be_->add_residual(ws_x_, ws_x2_, n, E, nullptr);
    }

    // ---- final norm + head, on the last row only ---------------------------
    const i32 row = n - 1;
    layernorm(ws_xn_, act_at(ws_x_, be_->act_size(), static_cast<i64>(row) * E),
              model_.out_norm(), model_.out_norm_bias(), 1, E, eps);
    logit_head(ws_xn_);
    return true;
}

bool Engine::forward_core(const i32 *toks, i32 n, i32 pos0, LogitMode mode) {
    KRK_TRACE_COMP("forward-core");
    forward_error_.clear();
    // The hash route needs the vocab id of each row (see fwd_toks_ in the
    // header); a row index only equals it on a single-token decode.
    fwd_toks_ = toks;
    fwd_toks_n_ = n;
    {
        // See the KRK_VRAM_TRACE block at the end of this function: entry and
        // exit readings bracket the forward, which is what tells a bound inside
        // it apart from one the caller took before calling.
        static const bool trace_vram = [] {
            const char *e = std::getenv("KRK_VRAM_TRACE");
            return e && *e && e[0] != '0';
        }();
        if (trace_vram && n > 1) log_vram_ledger("before prefill");
    }
    const ModelConfig &mc = model_.cfg();
    const auto &layers = model_.layers();
    const QuantTensor &embd = model_.tok_embd();
    // GPT-2 has its own loop (LayerNorm, learned positions, a fused qkv and an
    // ungated GELU MLP are one architecture rather than four options). Taking
    // the branch here rather than inside the loop keeps every other family's
    // graph literally untouched.
    // The plain pre-norm family -- GPT-2, GPT-NeoX, GPT-J, Phi-2. One loop
    // handles all four because they differ only in which pieces of the
    // old gpt2_style bundle they set (see ArchSpec), and taking the
    // branch here keeps every other family's graph untouched.
    if (mc.gpt2_style || mc.ln_bias) return plain_forward(toks, n, pos0);
    // Qwen3.5 packs its output gate behind the query (a per-layer shape only
    // when the arch says so); laguna projects its own. Decided once from the
    // arch, the same way Engine::init decided the packed workspace.
    const bool packed_gate = q_proj_ != q_dim_;

    // This is the one writer of the residual that stores at true scale (an
    // embedding row is small), so it starts the storage contract at exponent 0
    // for these rows; every later store goes through add_residual or a producer
    // that re-scales, and the array has to describe THIS pass.
    be_->fill0(ws_rexp_, static_cast<size_t>(n) * sizeof(i32));
    be_->embed(ws_x_, embd.data, embd.type, mc.n_vocab, mc.n_embd, toks, n);

    // ---- the gemma embedding scale ---------------------------------------
    // sqrt(n_embd), applied to the input row before anything reads it. The
    // family's every norm sees this magnitude, so a stack run without it has a
    // residual whose growth with depth is wrong by a constant factor per layer:
    // it decodes, and it is not the model. Done here rather than folded into
    // the table because the table is shared with the output projection on a
    // tied model, where the same weights must NOT be scaled.
    if (mc.embd_scale != 1.0f)
        be_->scale_act(ws_x_, mc.embd_scale, n, n_embd_, n_embd_);

    // muse-glimmer RMS-normalizes the embedding before layer 0, with no weight
    // and no bias (`inpL = build_norm(inpL, nullptr, ...)`). Skipping it leaves
    // every row at whatever magnitude the table happened to have, which the
    // first attn_norm then partly masks -- the run stays fluent and is not the
    // model.
    if (mc.embd_rms_noweight)
        be_->rmsnorm(ws_x_, ws_x_, nullptr, n, n_embd_, mc.rms_eps);

    // ---- the per-layer input table (gemma4) ------------------------------
    // Built once per pass, before the first layer, because the model
    // projection and the token table are ONE matrix each covering every
    // layer's slice:
    //     proj = model_proj @ (embd * sqrt(n_embd)) * (1/sqrt(n_embd))
    //          = model_proj @ embd
    //     tok  = tok_table[token] * sqrt(n_embd_per_layer)
    //     ple  = (rmsnorm(proj, proj_norm) + tok) * (1/sqrt(2))
    // The two sqrt factors cancel in `proj`, which is exactly what the
    // reference does -- it scales the input by sqrt(n_embd) and then divides
    // the projection by it -- so this multiplies the ALREADY SCALED ws_x_ by
    // 1/sqrt(n_embd) rather than reaching for the unscaled embedding.
    if (ppl_width_ > 0) {
        const f32 npl = static_cast<f32>(ppl_npl_);
        KRK_TRACE_COMP("ppl-table");
        be_->embed(ws_ppl_, model_.ple_tok_embd.data, model_.ple_tok_embd.type,
                   mc.n_vocab, ppl_width_, toks, n);
        be_->scale_act(ws_ppl_, std::sqrt(npl), n, ppl_width_, ppl_width_);
        gemm_w(ws_ppl_in_, ws_x_, model_.ple_model_proj, ppl_width_, n_embd_, n);
        be_->scale_act(ws_ppl_in_, 1.0f / std::sqrt(static_cast<f32>(n_embd_)), n,
                       ppl_width_, ppl_width_);
        // One RMS per (token, layer) slice: the weight is the file's single
        // [n_embd_per_layer] vector, repeated per layer by the loader.
        be_->rmsnorm_grouped(ws_ppl_in_, ws_ppl_in_,
                             model_.ple_proj_norm_rep.data(), n, mc.n_layer,
                             ppl_npl_, ppl_width_, mc.rms_eps);
        be_->add_inplace(ws_ppl_in_, ws_ppl_, n * ppl_width_);
        be_->scale_act(ws_ppl_in_, 1.0f / std::sqrt(2.0f), n, ppl_width_, ppl_width_);
    }

    // A latent token embedding stores ROTATED rows: un-rotate them here, once,
    // and the whole residual stream above this point is in the primal basis
    // that every norm, every non-folded matmul and every dump expects.
    if (model_.had().embd_inverse)
        had_xform(ws_x_, ws_x_, n, n_embd_, /*inverse=*/true, /*gdn_perm=*/false);

    // qwen4exp: the wide residual starts as hc identical copies of the
    // embedding row. That is the invariant everything downstream is checked
    // against -- with every lane equal in and a zero injection, the mix is
    // exactly `xn * sigmoid(gate)` and the lanes stay bit-equal out.
    if (hc_active_) {
        // The reference's hc_init: hc identical copies of the embedding row
        // (deepseek41.cpp repeats it into n_embd x hc x nt). That is also what
        // makes the single-pass identity collapse at layer 0 a no-op in value.
        be_->hc_init(ws_hc_, ws_x_, n, hc_.hc, n_embd_);
        hc_pre_valid_ = false;
    }

    AttnDesc d;
    d.n_head = mc.n_head;
    d.n_kv = mc.n_head_kv;
    d.hd = mc.head_dim;
    d.n_tok = n;
    d.pos0 = pos0;
    // Both of these are per layer and are set again inside the loop. The cache
    // hands out a per-layer base (so the stride is 0, see the residency call
    // below) and the row stride is the layer's own width, which is not one
    // number on a stack whose layers hold different amounts of KV. What is
    // here is the workspace bound, and every use after the loop overwrites it.
    d.layer_stride = 0;
    d.pos_stride = kv_dim_;
    d.kv_int8 = kv_quant_of(cfg_) == KvQuant::Int8;
    d.layer = 0;
    // The scale follows the layer's own head width, and a family that declares
    // no pre-attention scaling (gemma4: f_attention_scale = 1.0) says so in the
    // file rather than being approximated by the default 1/sqrt(hd).
    auto attn_scale_at = [&](i32 layer) -> f32 {
        if (mc.attn_scale > 0.0f) return mc.attn_scale;
        return static_cast<f32>(1.0 / std::sqrt(
            static_cast<f64>(model_.head_dim_at(layer))));
    };
    d.scale = attn_scale_at(0);
    // gemma2's bound on the attention logits, applied inside attention. 0 on
    // every other family, which is the arithmetic that existed before the field
    // did.
    d.softcap = mc.attn_softcap;
    // ---- the mask ---------------------------------------------------------
    // A masked-diffusion file is BIDIRECTIONAL: every denoising pass is one
    // chunk over the whole sequence, and a position may read the ones after it.
    // This flag is the whole of the difference -- the op has always taken it,
    // and both backends compute `n_keys = causal ? pos + 1 : pos0 + n_tok` -- so
    // it is read from the MODEL rather than chosen at a call site, and a diffuse
    // stack cannot run causal on one layer and not another.
    d.causal = !mc.diffusion;

    // One dump hook per stage, taking the row this call will predict from (the
    // last one): a CPU run and a HIP run of the same prompt then differ at
    // exactly one line per stage, and the first differing line names the layer.
    auto dump_stage = [&](const char *what, i32 layer) {
        dump_row(what, layer, ws_x_, n_embd_, n);
        // The same row-local checkpoints feed the bit-exactness trace. A
        // frozen chunk boundary anywhere above this point shows in `attn` of
        // this layer even when the f16 residual would have absorbed it.
        trace_rows(what, layer, ws_x_, n_embd_, n);
    };
    dump_stage("embed", -1);

    // PLE needs the tokens BEFORE this pass as well as its own, so the sequence
    // is kept here rather than re-derived from the KV cache: position p holds
    // the token at p, and the n-gram window at p reads p-1, p-2, ...
    if (ple_active_) {
        if (pos0 == 0) {
            // A new sequence. The conv history belongs to the sequence it was
            // built from, so it starts empty, and the window before the first
            // token reads as EOS.
            ple_seq_.clear();
            if (ple_state_ && ple_hist_ > 0) {
                be_->fill0(ple_state_,
                           static_cast<size_t>(mc.n_layer) * ple_hist_ *
                               hc_.lane_width() * be_->act_size());
            }
            ple_rows_.assign(ple_rows_.size(), 0);
        }
        const i64 need = static_cast<i64>(pos0) + n;
        if (static_cast<i64>(ple_seq_.size()) < need) {
            const i64 fill = ple_.eos_token >= 0 ? ple_.eos_token : 0;
            ple_seq_.resize(static_cast<size_t>(need), fill);
        }
        for (i32 i = 0; i < n; i++) ple_seq_[pos0 + i] = toks[i];
    }

    for (i32 l = 0; l < mc.n_layer; l++) {
        // Refill the streaming ring for this layer when it is not resident.
        // A resident layer is a no-op, and the copy is ordered on the default
        // stream ahead of every kernel that reads the weights.
        model_.stage_layer(l);
        const LayerWeights &L = layers[static_cast<size_t>(l)];
        // Per-layer geometry. A hybrid stack (laguna) alternates two layer
        // kinds with different head counts and different rotary tables, so the
        // query width, the head count handed to rope/qk_norm/attention, and
        // the rotation fraction are the layer's own. q_dim_/n_head remain the
        // maximum, which is what the shared workspaces are sized against.
        // DFlash feed: the residual stream *entering* this layer is one of the
        // drafter's feature blocks, so it is copied out here, before the
        // layer's own attn_norm consumes it. One activation copy per captured
        // layer, and nothing at all unless a head set is loaded.
        dflash_capture(l, n);
        const i64 lh = model_.n_head_at(l);
        const i64 lq = model_.q_dim_at(l);
        const i64 lq_proj = packed_gate ? 2 * lq : lq;
        // ... and the KV side is per layer for the same reason: a hybrid keeps
        // KV on some layers and not others (nemotron_h_moe: 6 of 52), and the
        // layers that do keep it need not carry the same head count (gemma4
        // declares {1,8}). kv_dim_/mc.n_head_kv are the stack maximum, which is
        // what the shared workspaces are sized against and nothing else.
        const i64 lkv = model_.kv_dim_at(l);
        const i64 lkvh = model_.n_head_kv_at(l);
        // The attention descriptor carries the layer's own head count, not the
        // stack's maximum. On a hybrid stack (laguna) those differ, and the
        // count is not just a loop bound: the op derives the GQA grouping from
        // it (n_rep = n_head / n_kv). A 48-head layer running as 64 heads read
        // its KV head as h/8 instead of h/6 — every query head grouped with
        // the wrong keys, and two of the eight KV heads never read at all —
        // and the 16 phantom heads were computed off whatever the workspace
        // held, which is why the same prompt answered differently at different
        // prefill chunk sizes.
        d.n_head = lh;
        d.n_kv = lkvh;
        // The head WIDTH is per layer for the same reason the counts are:
        // gemma4 declares two attention geometries in one stack (512 on the
        // global layers, 256 on the windowed ones), so a layer that reads the
        // stack maximum rotates and dots the wrong number of dims -- and the
        // workspace is wide enough that nothing overruns to say so.
        d.hd = model_.head_dim_at(l);
        d.scale = attn_scale_at(l);
        d.pos_stride = lkv;
        // The int8 row is kv_dim bytes plus one f32 scale per 32 values, so a
        // position's stride is the backend's answer for that width, not the
        // element count. Both the append and the attention read this one number.
        if (kv_quant_of(cfg_) == KvQuant::Int8)
            d.pos_stride = static_cast<i64>(be_->kv_row_bytes(lkv, kv_quant_of(cfg_)));
        d.kv_int8 = kv_quant_of(cfg_) == KvQuant::Int8;
        // A sliding-window layer sees only its last `swa_window` keys; the full
        // layers of the same stack see the whole prefix.
        d.window = (mc.swa_window > 0 && model_.is_swa(l)) ? mc.swa_window : 0;
        // KRK_SWA_OFF=1 runs every layer as full attention. It is the A/B arm
        // for the windowed path and nothing else: for a prompt shorter than
        // the window the two are the same computation, so a model that answers
        // differently with it on has a window bug and not a window policy.
        if (swa_off() && d.window > 0) d.window = 0;
        // Per-layer sinks: the pointer travels with the descriptor into every
        // attention arm (decode, split, exact prefill, tiled). The fused
        // chain refuses it and the engine takes the separate chain instead,
        // which is a few extra launches on a sink model and nothing else.
        d.sinks = L.sinks;
        dump_stage("enter", l);

        // The PLE block writes into the lane stack, so it runs before the
        // layer's mixer reads it -- the reference's build_ple, then build_hc_mix.
        if (ple_active_ && L.ple) ple_block(L, l, n, pos0);
        // Engram likewise: the reference adds its gated value to every lane
        // BEFORE the layer's mixes are computed from the stack
        // (deepseek41.cpp: build_engram, then build_hc_mixes).
        if (mc.hc_sinkhorn && !mc.ple_layers.empty() && L.engram_kv.present()) {
            if (!engram_block(L, l, toks, pos0, n)) return false;
        }

        if (hc_active_) {
            // No plain pre-norm on this family: the mixer is the block's input,
            // and the injection it also produces is what weights the combine.
            hc_begin(L, true, n);
        } else {
            // k2-horizon divides the row into `n_norm_groups` bands and takes
            // the RMS over each band, scaling the whole row by ONE weight. Every
            // other family leaves the count at 1, where rmsnorm_bands() calls
            // rmsnorm() itself so the arithmetic is bit-identical to before.
            if (mc.n_norm_groups > 1)
                be_->rmsnorm_bands(ws_xn_, ws_x_, L.attn_norm, n, mc.n_norm_groups,
                                   n_embd_, mc.rms_eps);
            else
            be_->rmsnorm(ws_xn_, ws_x_, L.attn_norm, n, n_embd_, mc.rms_eps,
                         static_cast<const i32 *>(ws_rexp_));
            norm_bias_add(static_cast<f32 *>(ws_xn_), L.attn_norm_bias, n, n_embd_);
        }

        if (L.gdn) {
            if (L.lfm_conv) {
                // LFM2.5: a depthwise causal convolution with its own gate, no
                // KV append (there is none to make) and no scan state. The
                // residual lands in ws_x_ before the FFN, exactly as the delta
                // net's does.
                lfm_conv_forward(L, l, n, pos0);
            } else if (L.nemotron_ssm) {
                // nemotron_h_moe Mamba-2 block. NOT the gated delta net: the
                // file carries no alpha/beta. Its recurrence is the SSD scan
                // over (xBC, dt, A); the scan performs all per-head prep, the
                // softmax free out-gate, and the per-group norm that follow it
                // are ordinary engine ops (see mamba2_forward).
                mamba2_forward(L, l, n);
            } else {
                // Recurrent layer: a short conv plus the delta rule replaces
                // attention entirely, and there is nothing to append to the KV
                // cache. The residual lands in ws_x_ before the FFN below.
                gdn_forward(L, l, n);
            }
            // nemotron's mamba-2 blocks have no FFN after the block; only
            // layers that carry FFN weights run one.
            // The recurrent block has landed in ws_x_ (as the mixed row plus
            // its output): promote it onto the lanes before the FFN mixes them
            // again. Unconditional -- a layer that skips its FFN still has to
            // scatter the block output it did produce.
            if (hc_active_) hc_end(n);
            if (!L.nemotron_ssm || L.moe || L.wgate.present()) {
                if (hc_active_) {
                    hc_begin(L, false, n);
                } else {
                    if (mc.n_norm_groups > 1)
                        be_->rmsnorm_bands(ws_xn_, ws_x_, L.ffn_norm, n,
                                           mc.n_norm_groups, n_embd_, mc.rms_eps);
                    else
                    be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, n, n_embd_,
                                 mc.rms_eps, static_cast<const i32 *>(ws_rexp_));
                    norm_bias_add(static_cast<f32 *>(ws_xn_), L.ffn_norm_bias, n, n_embd_);
                }
                dump_row("ffn.pre", l, ws_xn_, n_embd_, n);
                if (L.moe) {
                    if (!moe_ffn(L, l, n)) return false;
                } else {
                    dense_ffn(L, l, n);
                }
                if (hc_active_) hc_end(n);
            }
            // The residual leaving a recurrent (or skipped) block: on a stack
            // where the recurrent half is stubbed out this is the only place to
            // see whether the stream is still growing, and it is what localized
            // the nemotron NaN (layer 15 -> 17).
            dump_row("rec.post", l, ws_x_, n_embd_, n);
            dump_stage("gdn", l);
            continue;
        }

        // q/k/v all read the attention-norm output and have no
        // dependency between them, so they ride one fused launch
        // on the decode row (backend falls back per-matrix for
        // prefill rows or a dtype mix).
        // Folded projections read a TRANSFORMED copy of the attention-norm
        // output (prism.hadamard). ws_xn_ itself stays primal for whatever
        // else reads it -- laguna's output gate when only the projections are
        // folded, and the delta-net layer's ssm_alpha/ssm_beta, which no file
        // folds because they are not matmul sites in the fork either.
        // nemotron_h_moe runs 6 of its 52 layers with attention; the rest of
        // its non-mamba layers are pure-MoE blocks with nothing to attend
        // over. The attention machinery below requires q/k/v/output, so a
        // layer without them takes the short path: the residual is unchanged,
        // and the FFN block reads the norm of the same stream.
        // An MLA layer has no per-head q/k/v projection to run: it compresses
        // the KV into one latent row and attends that, so it takes its own path
        // and the MHA chain below (and the pure-MoE short path after it) do not
        // apply. `layer_has_kv()` is true for it for the same reason it is true
        // for an MHA layer -- it appends to a plane -- which is what makes the
        // cache budget and the tier geometry come out right.
        if (L.mla) {
            if (!mla_forward(L, l, n, pos0)) return false;
        } else if (L.wq.present() || L.wqkv_fused.present()) {
            // A fused Q | K | V projection is attention too: `wq` alone would
            // send a Phi layer down the no-attention short path, which reads as
            // a working model that ignores its context.
        const void *xn_att = ws_xn_;
        if (L.h_attn_in || L.h_attn_gate) {
            had_xform(ws_had_, ws_xn_, n, n_embd_, false, false);
            xn_att = ws_had_;
        }
        const void *xn_proj = L.h_attn_in ? xn_att : ws_xn_;
        void *q_out = packed_gate ? ws_qpack_ : ws_q_;
        // One launch cannot carry three different global scales, and an NVFP4
        // checkpoint picks each tensor's format (and its sidecar) on its own.
        // A scaled weight takes the per-matrix path, which can.
        // Phi-3: one fused Q | K | V projection instead of three. The split
        // widths differ because Phi is GQA, which is why this cannot reuse the
        // gpt2 path's equal-width split_qkv.
        if (L.wqkv_fused.present()) {
            // Phi-3: one fused Q | K | V projection whose row is Q then K then
            // V. Both part widths are whole numbers of quantisation blocks, so
            // the three row ranges are VIEWS of the one weight rather than a
            // copy of its output: the same three projections the split path
            // runs, over the same bytes, so the result is identical by
            // construction -- no split buffer, no device sync, and therefore no
            // way for a stride convention to differ between the two paths.
            const i64 kvw = static_cast<i64>(lkv);
            const i64 rb = static_cast<i64>(
                dtype_row_bytes(L.wqkv_fused.type, L.wqkv_fused.n_in));
            QuantTensor wq = L.wqkv_fused;
            wq.n_out = n_embd_;
            QuantTensor wk = L.wqkv_fused;
            wk.data = static_cast<u8 *>(L.wqkv_fused.data) + n_embd_ * rb;
            wk.n_out = kvw;
            QuantTensor wv = L.wqkv_fused;
            wv.data = static_cast<u8 *>(L.wqkv_fused.data) + (n_embd_ + kvw) * rb;
            wv.n_out = kvw;
            void *qkv[3] = {q_out, ws_k_, ws_v_};
            const void *wqkv[3] = {wq.data, wk.data, wv.data};
            const i64 nqkv[3] = {lq_proj, kvw, kvw};
            be_->gemm_group(qkv, xn_proj, wqkv, wq.type, nqkv, n_embd_, 3, n);
        } else if (L.attn_v_exps.present()) {
            // MoVA (k2-horizon): q and k project, V does not. It is routed --
            // see mova_value() -- so the fused q|k|v launch cannot be used even
            // though q and k are the same dtype: V is not a third column block.
            gemm_w(q_out, xn_proj, L.wq, lq_proj, n_embd_, n);
            gemm_w(ws_k_, xn_proj, L.wk, lkv, n_embd_, n);
            KRK_TIME_HOST("mova.value");
            if (!mova_value(L, l, xn_proj, n, lkv, mc.router_sigmoid)) {
                forward_error_ = "MoVA: the value router could not run on this "
                                 "backend (no workspace)";
                return false;
            }
        } else if (L.wq.type == L.wk.type && L.wk.type == L.wv.type &&
            !L.wq.scaled() && !L.wk.scaled() && !L.wv.scaled()) {
            void *qkv[3] = {q_out, ws_k_, ws_v_};
            const void *wqkv[3] = {L.wq.data, L.wk.data, L.wv.data};
            const i64 nqkv[3] = {lq_proj, lkv, lkv};
            be_->gemm_group(qkv, xn_proj, wqkv, L.wq.type,
                            nqkv, n_embd_, 3, n);
        } else {
            gemm_w(q_out, xn_proj, L.wq, lq_proj, n_embd_, n);
            gemm_w(ws_k_, xn_proj, L.wk, lkv, n_embd_, n);
            gemm_w(ws_v_, xn_proj, L.wv, lkv, n_embd_, n);
        }

        // Qwen3.5 interleaves each head's output gate into the query
        // projection, so the packed rows have to be split before anything
        // else touches them. The query half stays in place.
        if (packed_gate) {
            be_->qwen3_next_split(ws_q_, ws_agate_, q_out, n, lh,
                                  mc.head_dim);
            // The gate half is the one buffer between the projections and the
            // layer output that no earlier stage covers.
            dump_row("attn.qp", l, q_out, lq_proj, n);
            dump_row("attn.gate", l, ws_agate_, lq, n);
        }

        if (L.q_bias) be_->add_bias_rows(ws_q_, L.q_bias, lq, n);
        if (L.k_bias) be_->add_bias_rows(ws_k_, L.k_bias, lkv, n);
        if (L.v_bias) be_->add_bias_rows(ws_v_, L.v_bias, lkv, n);
        // A FUSED projection carries one bias laid out Q | K | V, i.e. the same
        // split as the weight -- so it is three offsets into one vector, not a
        // fourth tensor. The plain family's loop has always done this; this
        // path did not, which silently dropped the bias for any fused-qkv
        // family that carries one and still decodes fluent text.
        if (L.qkv_bias) {
            be_->add_bias_rows(ws_q_, L.qkv_bias, lq, n);
            be_->add_bias_rows(ws_k_, L.qkv_bias + lq, lkv, n);
            be_->add_bias_rows(ws_v_, L.qkv_bias + lq + lkv, lkv, n);
        }
        // OLMo-1 bounds the three attention projections: `attention.clamp_kqv`
        // (8.0 in every OLMo-1 config) clamps q, k and v to [-c, +c] here,
        // after the projection and its bias and BEFORE the rotary -- the
        // reference's order, and the clamp is inside the rotation's input, so
        // applying it on the rotated values is a different function. The
        // loader refuses a file that declares a positive clamp on a backend
        // without clamp_act(), so reaching this branch with a bound means the
        // kernel exists; nothing at all runs at 0, which is every other family.
        if (mc.attn_clamp_kqv > 0.0f) {
            const f32 c = mc.attn_clamp_kqv;
            be_->clamp_act(q_out, -c, c, static_cast<i64>(n) * lq);
            be_->clamp_act(ws_k_, -c, c, static_cast<i64>(n) * lkv);
            be_->clamp_act(ws_v_, -c, c, static_cast<i64>(n) * lkv);
        }
        // QSA: the indexer's own projections and its block selection, computed
        // from the same normed input q/k/v read and BEFORE rope touches
        // anything (the indexer rotates pooled keys and its own queries, not
        // these). True means some visible cell was dropped and this layer must
        // attend over the selection instead of the whole cache.
        const bool qsa_sel =
            qsa_active_ && L.qsa_ratio > 0 &&
            qsa_select_block(L, l, n, pos0, xn_proj);

        d.layer = l;
        // Resolve this layer's residency before it is read or written.
        //
        // base() ALWAYS returns a pointer to this layer alone -- an offset
        // into the flat allocation when the cache fits, a slot when it does
        // not -- so d.layer_stride must be 0 in both cases. Leaving it at
        // kv_cap*kv_dim makes the kernels compute `base + d.layer*stride`,
        // which walks l layers PAST the layer we just handed them: out of
        // bounds on every layer but 0. That faults the GPU rather than
        // producing wrong numbers, which is how it showed up.
        //
        // The alternative -- keep the whole-array base and the stride -- is
        // incompatible with paging, because a paged layer has no fixed
        // offset. Hence per-layer bases and a zero stride.
        // A shared-KV layer reads the plane of the layer it shares with; its
        // own is never allocated (see the kv_len pass). Resolving the source
        // here makes the append gate, the attention and every dump below
        // address one plane -- the aliasing llama.cpp implements by pointing
        // two layers' cache tensors at the same buffer.
        const i32 kv_src = model_.kv_source_layer(l);
        // ---- the cacheless pass --------------------------------------------
        // A masked-diffusion forward is one full-sequence chunk that is thrown
        // away when it ends: the next denoising step rewrites every position.
        // So a diffuse layer appends nothing and reads only the K/V it just
        // projected -- `ws_k_`/`ws_v_` ARE this chunk's cache, with pos_stride
        // one row and layer_stride 0 because the buffer holds one layer's
        // chunk. Keeping the KV tier in the path would make the pass read a
        // previous step's rows at the same positions, which silently mixes
        // denoising steps that the schedule believes are independent.
        void *kbase = nullptr;
        void *vbase = nullptr;
        if (mc.diffusion) {
            kbase = ws_k_;
            vbase = ws_v_;
        } else {
            kbase = kvt_k_.base(kv_src);
            vbase = kvt_v_.base(kv_src);
        }
        // A model may have no KV on this layer; only a missing pointer for a
        // KV-carrying layer is a resource refusal. Do not turn a valid null pair
        // on recurrent/pure-FFN blocks into a request failure.
        if (model_.layer_has_kv(l) && (!kbase || !vbase)) {
            KRK_ERROR("kv: no resident slot for layer %lld (host reserve %lld MiB "
                      "refused %lld KV promotion(s)); refusing the forward rather "
                      "than replacing retained cache history with zeros",
                      static_cast<long long>(l),
                      static_cast<long long>(krk::host_reserve_bytes(be_->host_total_bytes()) /
                                             1048576u),
                      static_cast<long long>(kvt_k_.promotion_refused() +
                                             kvt_v_.promotion_refused()));
            return false;
        }
        d.layer_stride = 0;
        // RoPE + KV append + attention ride one launch on the decode
        // row (kernels/attention.hpp, attn_fused_decode): the kernel
        // ropes q in registers and k straight into its cache slot, so
        // the k/v rows never round-trip through global memory between
        // three launches. The chain now also carries the per-head QK-norm
        // (bit-identical to head_norm_kernel) and both rope pairings, so
        // the Qwen3-family shapes that used to keep the separate chain
        // take it too. What the engine still gates here: multi-token rows
        // (prefill), a packed output gate (which must multiply the result
        // afterwards), a sliding window (the fused kernel has no bound)
        // and a separate attention gate. Whole-row qk-norm (OLMoE) and
        // partial-rope NeoX the BACKEND refuses, one call deeper.
        // Unsupported backends and shapes keep the separate chain.
        // Full NeoX QK-norm now uses pair-owned rotation and f16 LDS
        // storage, matching the separate chain. Keep its old gate as an A/B.
        static const bool fused_neox = [] {
            const char *e = std::getenv("KRK_FUSED_ATTN_NEOX");
            return !(e && e[0] == '0');
        }();
        // A shared-KV layer (kv_src != l) must not append, a weightless V-norm
        // has to run before the cache and the fused chain carries neither, so
        // those shapes take the separate chain. It is the same arithmetic with
        // one extra launch per such layer.
        if (!(n == 1 && !packed_gate && d.window == 0 && !mc.attn_gate &&
              !qsa_sel && kv_src == l && !mc.v_norm_weightless &&
              (fused_neox || !(mc.rope_neox && mc.qk_norm)) &&
              be_->attn_fused_chain(ws_attn_, ws_q_, kbase, vbase,
                                      ws_k_, ws_v_, d,
                                      model_.inv_freq_at(l).data(),
                                      model_.rope_scale_at(l),
                                      model_.rope_frac_at(l),
                                      mc.qk_norm ? L.q_norm : nullptr,
                                      mc.qk_norm ? L.k_norm : nullptr,
                                      mc.rms_eps, mc.qk_norm_wide,
                                      mc.rope_neox))) {
            // The fused chain carries the QK-norm itself; the separate
            // chain runs it here, before rope, exactly as before. Running
            // it unconditionally would norm twice on the fused arm -- a
            // wrong-answer bug the md5 A/B caught, not a crash.
            // qk_norm_wide is OLMoE's whole-row convention; false is the
            // per-head one Qwen3 and Gemma3 use.
            if (mc.qk_norm)
                be_->qk_norm(ws_q_, ws_k_, L.q_norm, L.k_norm, lh, lkvh,
                             d.hd, n, mc.rms_eps, mc.qk_norm_wide);
            // gemma4 RMS-normalizes the VALUE row with no weight at all
            // (`Vcur = ggml_rms_norm(Vcur)`). It is not a gamma of ones: the
            // file carries no tensor, so a family that declares it must run it
            // here, before the row reaches the cache -- normalizing on read
            // would cost a pass per attended key instead of one per token.
            if (mc.v_norm_weightless && L.wv.present())
                be_->rmsnorm(ws_v_, ws_v_, nullptr, n, lkv, mc.rms_eps);
            // Post-norm, pre-rope: the row the rotation consumes. The
            // fused arm's copy of it lives in registers and never lands
            // in this buffer, so this row covers the separate chain only.
            dump_row("attn.qk", l, ws_q_, lq, n);
            // smollm3's NoPE layers rotate nothing: the pattern is part of the
            // architecture (every 4th layer), so skipping it here is what the
            // file was trained with. Rotating anyway is a wrong answer that
            // still decodes.
            // Two patterns skip the rotation, and both are the architecture's:
            // smollm3's every-Nth NoPE layer, and muse-glimmer's full-attention
            // layers (it rotates only on the sliding-window ones).
            const bool layer_rotates =
                !(mc.no_rope_step > 0 && ((l + 1) % mc.no_rope_step) == 0) &&
                !(mc.rope_swa_only && !model_.is_swa(l));
            if (layer_rotates)
                be_->rope(ws_q_, ws_k_, lh, lkvh, d.hd,
                          n, pos0, model_.inv_freq_at(l).data(),
                          model_.rope_scale_at(l),
                          model_.rope_frac_at(l), mc.rope_neox);
            // A shared-KV layer appends nothing: the layer that owns this
            // plane wrote these rows already, and a second write would put the
            // token at two positions in one cache. A DIFFUSE layer appends
            // nothing either, for the reason given where kbase is resolved.
            if (kv_src == l && !mc.diffusion)
                be_->kv_append(kbase, vbase, ws_k_, ws_v_, d);
            if (qsa_sel)
                qsa_attention(l, n, d);
            else
                be_->attention(ws_attn_, ws_q_, kbase, vbase, d);
        }
        // The layer's KV is now queued. Mark the slot so a later layer that
        // evicts it waits for these kernels instead of overwriting memory they
        // are still reading. This has to come AFTER the launches above and
        // after any fused chain, which is why it is not folded into base().
        kvt_k_.end_layer(l);
        kvt_v_.end_layer(l);
        // Where inside the attention layer a single-token step goes wrong:
        // q/k are post-rope here, so a divergence in them is rope or the
        // projection, and agreeing inputs with a wrong "out" is the attention
        // kernel itself.
        dump_row("attn.q", l, ws_q_, lq, n);
        dump_row("attn.k", l, ws_k_, lkv, n);
        // v is the one buffer between the projections and the output that no
        // other stage covers: a NaN here and a NaN in the cache produce the
        // same all-NaN attn.out, and this tells them apart.
        dump_row("attn.v", l, ws_v_, lkv, n);
        dump_row("attn.out", l, ws_attn_, lq, n);
        // The cache row this step appended: the one input to the
        // attention that no activation buffer covers, and the row a
        // fused chain writes from inside its own kernel. Width is the
        // whole KV row (n_kv heads), rows = 1 at this step's position.
        dump_row("attn.kc", l,
                 static_cast<const u8 *>(kbase) +
                     pos0 * d.pos_stride * be_->act_size(),
                 lkv, 1);
        dump_row("attn.vc", l,
                 static_cast<const u8 *>(vbase) +
                     pos0 * d.pos_stride * be_->act_size(),
                 lkv, 1);

        // laguna's output gate: a separate [n_head] projection of the same
        // attention-norm output q/k/v read, softplus'd, and applied as one
        // scalar per head over that head's whole output row. Before wo, which
        // is what makes it a gate on the attention result rather than on the
        // residual.
        if (mc.attn_gate && L.wattn_gate.present()) {
            // Two widths under one tensor name. laguna and spark2_5 project one
            // value per head and broadcast it over that head's row;
            // muse-glimmer projects one per element of the attention row, so
            // its gate is as wide as the row itself.
            const i64 gw = mc.attn_gate_wide ? lq : lh;
            gemm_w(ws_agate_, L.h_attn_gate ? xn_att : ws_xn_, L.wattn_gate, gw,
                   n_embd_, n);
            // Same tensor, two activations: laguna softplus's the gate,
            // muse-glimmer and spark2_5 sigmoid it. Getting this wrong is not a
            // crash -- every head is simply gated a little too hard, and the
            // text stays fluent -- so the activation is a declared property of
            // the arch.
            if (mc.attn_gate_sigmoid)
                be_->sigmoid_act(ws_agate_, static_cast<i64>(n) * gw);
            else
                be_->softplus_act(ws_agate_, static_cast<i64>(n) * gw);
            dump_row("attn.gate", l, ws_agate_, gw, n);
            if (mc.attn_gate_wide)
                be_->mul_act(ws_attn_, ws_agate_, static_cast<i64>(n) * lq);
            else
                be_->mul_head_broadcast(ws_attn_, ws_agate_, n, lh, mc.head_dim);
            dump_row("attn.gated", l, ws_attn_, lq, n);
        }

        // Qwen3.5's output gate is a sigmoid applied to the attention result,
        // not to the projection: gate it before wo.
        if (packed_gate) {
            be_->sigmoid_act(ws_agate_, static_cast<i64>(n) * lq);
            be_->mul_act(ws_attn_, ws_agate_, static_cast<i64>(n) * lq);
            dump_row("attn.gated", l, ws_attn_, lq, n);
        }
        }

        // Everything below reads ws_attn_ and writes the residual, so it is
        // shared by both attention kinds; only a block with NO attention at all
        // (nemotron's attention-only layers' mirror image) takes the short path.
        if (L.mla || L.wq.present() || L.wqkv_fused.present()) {
        // The attention result's width is the OUTPUT projection's input width,
        // not the query width: they are the same number on every family that
        // projects per-head V, and differ exactly on an MLA layer, whose
        // attention result is n_head*v_dim wide while its query is
        // n_head*(nope+rope). Reading `lq` here would run the projection over
        // 1728 stale columns per row on V2-Lite -- a wrong answer, not a crash.
        // The shared-latent layout has already run its grouped output
        // projection (wo_a/wo_b are inside mla_shared_forward), so its result is
        // n_embd wide and `L.wo` is absent on the layer. Reading L.wo.n_in there
        // would trace a zero-width row and, worse, run a second projection over
        // it below.
        const bool mla_grouped =
            L.mla && mc.mla_layout == MlaLayout::Shared;
        const i64 aw = mla_grouped ? (lh * mc.head_dim)
                                   : (L.mla ? L.wo.n_in : lq);

        // The attention result is the input of a folded projection too, and
        // it is a different buffer with a different width (q_dim), so it gets
        // its own transform -- after the gate above has finished reading
        // ws_had_.
        // The raw attention output, before its projection: the stage where a
        // key-space reduction order shows first, and the one the tiled kernel
        // used to differ at. Both chains write it, so one call here covers a
        // 1-row call (the fused chain) and a multi-row one (the separate chain).
        trace_rows("attn.out", l, ws_attn_, aw, n);
        const void *attn_in = ws_attn_;
        if (L.h_attn_out) {
            had_xform(ws_had_, ws_attn_, n, aw, false, false);
            attn_in = ws_had_;
        }
        // BitNet's SubLN, attention half: an RMSNorm on the CONTEXT, between
        // the attention and its output projection -- NOT a second norm on the
        // layer's input, which is what attn_norm already is. The two are
        // different tensors of the same width, so reading this one as a second
        // pre-norm keeps the same op and the same shape and still decodes
        // nonsense; the position is stated here and the loader checks the
        // width. In place on whichever buffer holds the context: the fold above
        // has already consumed ws_attn_ when it ran, and ws_had_ holds the
        // transformed copy otherwise.
        if (L.attn_sub_norm) {
            void *ctx = attn_in == ws_attn_ ? ws_attn_ : ws_had_;
            be_->rmsnorm(ctx, ctx, L.attn_sub_norm, n, aw, mc.rms_eps);
        }
        if (!mla_grouped)
            gemm_w(ws_x2_, attn_in, L.wo, n_embd_, aw, n);
        // gpt-oss biases the attention output: one value per output channel,
        // applied to the projection's result before it joins the residual. A
        // broadcast over the rows of the GEMM output, which is exactly the
        // `add_bias_cols` shape (n rows of n_embd columns).
        if (L.wo_bias) be_->add_bias_cols(ws_x2_, L.wo_bias, n, n_embd_);
        // MiniCPM3's depth scale: the ATTENTION branch is multiplied by
        // `depth / sqrt(n_layer)` before it joins the residual, and only that
        // branch -- the FFN's output is added at full weight. It is one
        // constant per stack, so it is divided once at load (ModelConfig says
        // why) and applied here, where the branch's final value is known and
        // nothing else has read it yet. Skipped entirely at 1.0.
        if (mc.attn_out_scale != 1.0f)
            be_->scale_act(ws_x2_, mc.attn_out_scale, n, n_embd_, n_embd_);
        dump_row("attn.proj", l, ws_x2_, n_embd_, n);
        // Branch post norm (gemma2/3/4, muse-glimmer): the projection's own
        // output is normalized with THIS branch's weight before it joins the
        // residual -- `x + norm(attn_out)`, not `norm(x + attn_out)` and not
        // `x + attn_out`. The dump above is the raw row and the one below is
        // the normalized one, so the order is checkable from a trace.
        //
        // No row exponent here: ws_x2_ is a projection buffer, not the residual
        // stream, so the per-token storage exponent does not apply to it.
        if (L.attn_post_norm) {
            const f32 peps =
                mc.post_norm_eps > 0.0f ? mc.post_norm_eps : mc.rms_eps;
            be_->rmsnorm(ws_x2_, ws_x2_, L.attn_post_norm, n, n_embd_, peps,
                         nullptr);
            dump_row("attn.postnorm", l, ws_x2_, n_embd_, n);
        }
        // The attention residual and the FFN norm are one kernel: the sum is
        // what the norm's sum-of-squares pass reads, so a second launch
        // round-trips the activation row for nothing. The dump still sees the
        // post-add state, because the fused kernel writes x before it
        // normalizes.
        if (hc_active_) {
            // The fused add_rmsnorm() is the one seam this family cannot use:
            // the attention output has to reach the LANE STACK before the FFN's
            // input is mixed from it, and the two are different streams. So the
            // sum happens on its own, the combine scatters, and the FFN gets a
            // fresh mix -- three launches where the narrow path pays one.
            be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(n) * n_embd_);
            hc_end(n);
            hc_begin(L, false, n);
        } else {
            be_->add_rmsnorm(ws_xn_, ws_x_, ws_x2_, L.ffn_norm, n, n_embd_,
                             mc.rms_eps, static_cast<const i32 *>(ws_rexp_));
            norm_bias_add(static_cast<f32 *>(ws_xn_), L.ffn_norm_bias, n, n_embd_);
        }
        } else {
            // Pure-MoE block: no attention to fold in, so the FFN input norm
            // is applied straight to the residual.
            if (hc_active_) {
                hc_begin(L, false, n);
            } else {
                be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, n, n_embd_, mc.rms_eps,
                             static_cast<const i32 *>(ws_rexp_));
                norm_bias_add(static_cast<f32 *>(ws_xn_), L.ffn_norm_bias, n, n_embd_);
            }
            dump_row("ffn.pre", l, ws_xn_, n_embd_, n);
        }
        dump_row("attn.res", l, ws_x_, n_embd_, n);

        dump_row("attn.ffnn", l, ws_xn_, n_embd_, n);
        // A layer runs an FFN only if it carries FFN weights, and this branch
        // needs the guard as much as the recurrent one below: nemotron_h_moe's
        // six attention layers (L05/12/19/26/33/42 on the 30B file) are
        // attention-ONLY -- the checkpoint has attn_q/k/v/output and nothing
        // else for them, no gate, no up, no down. Running dense_ffn() there
        // reads a null gate: on the device the unnamed type makes the GEMM
        // write zero, so the block is inert and the stage dump reads exactly 0;
        // on the CPU the destination buffers are never written and hold
        // whatever the previous layer left, which measured 6.2e6 into the gate
        // and 1.6e11 into the residual at L05 -- the whole reference arm past
        // that layer, and every token it emits, is that buffer's contents.
        if (L.moe && L.ffn_pre_norm2) {
            // gemma4's routed block: two branches, three norms, a detached
            // router input. See Engine::gemma_moe_ffn.
            if (!gemma_moe_ffn(L, l, n)) return false;
        } else if (L.moe) {
            if (!moe_ffn(L, l, n)) return false;
        } else if (L.wgate.present() || L.wup_fused.present()) {
            // A fused gate|up projection IS the gate: keying only on wgate
            // skipped the FFN altogether on a Phi file, which is the same shape
            // of bug the attention gate above had.
            dense_ffn(L, l, n);
        }
        if (hc_active_) hc_end(n);
        // ---- the per-layer embedding residual (gemma4) -------------------
        // After the FFN and BEFORE the layer's output scale, which is the
        // order the reference's graph has: the block's own output includes
        // this residual, so scaling first would leave the per-layer
        // contribution at the wrong magnitude on every layer. `inp_gate`
        // reads the finished residual (attn + ffn), not the FFN's input --
        // the graph's `pe_in` is taken after `cur += attn_out`.
        if (ppl_width_ > 0 && L.ple_inp_gate.present()) {
            gemm_w(ws_ppl_g_, ws_x_, L.ple_inp_gate, ppl_npl_, n_embd_, n);
            be_->gelu_act(ws_ppl_g_, n * ppl_npl_);
            be_->mul_slice(ws_ppl_g_, ws_ppl_in_, n, ppl_npl_, ppl_width_,
                           static_cast<i64>(l) * ppl_npl_);
            gemm_w(ws_ppl_o_, ws_ppl_g_, L.ple_proj, n_embd_, ppl_npl_, n);
            be_->rmsnorm(ws_ppl_o_, ws_ppl_o_, L.ple_post_norm, n, n_embd_, mc.rms_eps);
            be_->add_inplace(ws_x_, ws_ppl_o_, n * n_embd_);
        }
        // The layer's output scalar. gemma4 stores one per layer and the
        // reference multiplies the finished block output by it; a stack whose
        // tensors are all 1.0 keeps 1.0 here and this is a compare per layer.
        if (L.out_scale != 1.0f)
            be_->scale_act(ws_x_, L.out_scale, n, n_embd_, n_embd_);
        dump_stage("attn", l);
    }
    // The MTP probe's feature rows: the same pre-final-norm state DFlash
    // captures, but for every position of this pass. The block has its own
    // attention, so it needs the whole prompt -- one row is not enough.
    mtp_capture(toks, pos0, n);
    // The external head reads the same row, for every position it may propose
    // from -- not only the last one.
    head_capture(toks, pos0, n);
    // The pre-final-norm state, which a capture list names with the id
    // `mc.n_layer`: the input of the block that does not exist. It is what the
    // drafter's last feature block reads.
    dflash_capture(mc.n_layer, n);
    dump_stage("final", mc.n_layer);

    // Vocab projection: the expensive part of the head, so it runs on as few
    // rows as the caller asked for. Prefill chunks skip it entirely.
    if (mode == LogitsNone) return true;
    if (mode == LogitsLast) {
        head_compute(n - 1);
        return true;
    }
    for (i32 r = 0; r < n; r++) head_compute(r);
    return true;
}

// dst += x * W^T, folded into the projection's epilogue where the backend can
// (one launch instead of two; see Backend::gemm_accumulate for why the folded
// store is bit-identical to the pair it replaces).
//
// The pair is also what runs whenever this cannot: on the CPU backend, on
// prefill rows, and under KRK_DUMP -- the stage trace reads the projection
// buffer, and a folded epilogue never writes it, so a fused attn.down would
// silently report the previous layer's numbers. The dump is a debugging
// instrument that has to keep telling the truth, so it wins.
// dst = x * W^T, with the weight's NVFP4 global scale (see QuantTensor::gs).
//
// One place, so that a sidecar cannot be applied at one projection and dropped
// at the next. The scale rides the RESULT rather than the input: both are the
// same matrix product -- gs*(W x) is (gs W) x -- and the result is the buffer
// this call owns, so a caller whose input is shared with another projection
// (q/k/v off one activation, gate/up off another) is unaffected.
void Engine::gemm_w(void *dst, const void *x, const QuantTensor &w, i64 n_out,
                    i64 n_in, i64 rows) {
    if (w.gs == 1.0f) {
        be_->gemm(dst, x, w.data, w.type, n_out, n_in, rows);
        return;
    }
    // The scale goes on the INPUT, not on the result, and that is forced by the
    // numbers rather than preferred: a sidecar exists because the stored tensor
    // is 1/gs too large, so the raw product is 1/gs too large too -- on the Bolt
    // file that is 7,480x -- and the projection's f16 store overflows before any
    // output scale could bring it back. Scaling the input keeps the product in
    // range, and gs*(W x) is (W (gs x)), the same number. It costs one staged
    // copy per scaled projection, on a file that has sidecars at all.
    be_->copy_act(ws_gs_, x, rows * n_in);
    be_->scale_act(ws_gs_, w.gs, rows, n_in, n_in);
    be_->gemm(dst, ws_gs_, w.data, w.type, n_out, n_in, rows);
}

void Engine::logit_head(const void *xn) {
    gemm_w(ws_logits_, xn, model_.out_head(), n_vocab_, n_embd_, 1);
    // Phi-MoE's `output.bias`. Adding it on the device keeps the buffer the
    // single source of truth: topk_argmax() reads ws_logits_ directly and never
    // sees the host copy.
    if (const f32 *b = model_.out_bias())
        be_->add_bias_rows(ws_logits_, b, n_vocab_, 1);
    // muse-glimmer's `<arch>.logit_scale`: the reference scales the head's
    // output and THEN applies the final-logit softcap, so the composed function
    // is cap*tanh(s*x/cap) -- not s*cap*tanh(x/cap), which is what an
    // after-the-cap multiply would produce, and not x, which is what ignoring
    // the key produced. Every logits consumer reads this buffer (the host
    // download, the device top-k, the speculative verifier and the draft
    // comparisons), so the scale lands once, here, on the device.
    if (const f32 s = model_.cfg().logit_scale; s != 1.0f)
        be_->scale_act(ws_logits_, s, 1, n_vocab_, n_vocab_);
}

void Engine::gemm_residual(void *dst, const void *x, const void *w, DType wt,
                           i64 n_out, i64 n_in, i64 rows, void *scratch, f32 gs) {
    // `rexp` goes down both paths: `dst` is the residual stream, so the
    // contribution folded in has to be divided by the row's storage exponent,
    // and the fallback -- which is what prefill rows and the CPU take -- puts it
    // through add_residual, so it renormalizes the exponent as well.
    const i32 *rexp = static_cast<const i32 *>(ws_rexp_);
    // A scaled weight cannot take the folded epilogue -- gemm_accumulate writes
    // the RAW product into the residual -- so it stages a scaled input and then
    // takes the same pair, with the projection landing in `scratch` at true
    // scale. See gemm_w for why the scale has to be on the input.
    const void *xin = x;
    if (gs != 1.0f) {
        be_->copy_act(ws_gs_, x, rows * n_in);
        be_->scale_act(ws_gs_, gs, rows, n_in, n_in);
        xin = ws_gs_;
    }
    // `resid_scaled_` refuses the fold on a run whose residual rows ride a
    // storage exponent: the fold's store is at the row's current exponent and
    // cannot renormalize, so a contribution that needs a bigger one clips
    // instead of re-scaling. See the note in Engine::forward.
    if (gs == 1.0f && !resid_scaled_ && dump_path_.empty() && rows > 0 &&
        be_->gemm_accumulate(dst, x, w, wt, n_out, n_in, rows, rexp))
        return;
    be_->gemm(scratch, xin, w, wt, n_out, n_in, rows);
    be_->add_residual(dst, scratch, rows, n_out, const_cast<i32 *>(rexp));
}

// The dense SwiGLU FFN both layer shapes share: xn -> gate/up -> silu -> down
// -> residual. It was written out twice, and the GDN copy was the stale one:
// it still issued gate and up as two separate gemm() calls while the attention
// copy had been moved onto the fused group launch. That is 18 layers x 1
// wasted launch on the 0.8B, which is 18 of the step's 541 ops — pure op-count
// tax, since the two matrices read the same activation and have no dependency
// between them, exactly the condition gemm_group exists for.
void Engine::dflash_capture(i32 layer, i32 n) {
    if (!dflash_ || n <= 0) return;
    const std::vector<i32> &tl = dflash_->target_layers();
    for (size_t a = 0; a < tl.size(); a++) {
        if (tl[a] != layer) continue;
        be_->copy_act(dflash_->capture_slot(static_cast<i32>(a)), ws_x_,
                      static_cast<i64>(n) * n_embd_);
    }
}

// ---- nextn (MTP) probe -----------------------------------------------------

// The probe needs the target's hidden state at EVERY prompt position, not just
// the last one: the block has its own attention, so it has to walk the prompt
// and build its own KV plane before it can predict anything from the end of it.
// Captured at the pre-final-norm point -- the same row DFlash captures for its
// last feature block -- because that is what a nextn head is trained against.
void Engine::mtp_capture(const i32 *toks, i32 pos0, i32 n) {
    if (!mtp_probe_ || !mtp_hidden_ || n <= 0 || toks == nullptr) return;
    const i64 end = static_cast<i64>(pos0) + n;
    if (pos0 < 0 || end > kv_cap_) return;
    const size_t as = be_->act_size();
    be_->copy_act(act_at(mtp_hidden_, as, static_cast<i64>(pos0) * n_embd_),
                  ws_x_, static_cast<i64>(n) * n_embd_);
    for (i32 i = 0; i < n; i++)
        mtp_tok_[static_cast<size_t>(pos0 + i)] = toks[i];
    mtp_captured_ = end;
}

// The block's expert tensors as an ExpertSource: the cache is keyed on
// (layer, expert) and knows nothing about which struct a layer lives in, so a
// nextn block pages through the same budget as any stack layer.
// ---- sublayer ablation for the nextn block --------------------------------
//
// The block is four things stacked: the eh_proj fusion of (embedding, hidden),
// its own attention, and an FFN that is a routed set plus a shared expert.
// When the block loses tokens against the stack's own head, the question is
// WHICH of those is responsible, and a rank distribution cannot answer it --
// every arm below produces a plausible-looking distribution. Dropping one
// sublayer per run and re-reading the target's rank does answer it.
//
// The names say what is DROPPED, because that is what an ablation is:
//   attn   -- the attention residual (the block keeps eh_proj + FFN)
//   ffn    -- the whole FFN (routed + shared)
//   both   -- neither, so the block output is the eh_proj fusion alone
//   routed -- only the routed experts (shared expert still added)
//   shared -- only the shared expert (routed set skipped)
enum class MtpAbl { None, Attn, Ffn, Both, Routed, Shared };

inline MtpAbl mtp_abl() {
    static const MtpAbl a = [] {
        const char *e = std::getenv("KRK_MTP_ABL");
        if (!e || !*e) return MtpAbl::None;
        const std::string v(e);
        if (v == "attn") return MtpAbl::Attn;
        if (v == "ffn") return MtpAbl::Ffn;
        if (v == "both") return MtpAbl::Both;
        if (v == "routed") return MtpAbl::Routed;
        if (v == "shared") return MtpAbl::Shared;
        return MtpAbl::None;
    }();
    return a;
}

inline const char *mtp_abl_name(MtpAbl a) {
    switch (a) {
        case MtpAbl::Attn: return "attn dropped";
        case MtpAbl::Ffn: return "ffn dropped";
        case MtpAbl::Both: return "attn+ffn dropped";
        case MtpAbl::Routed: return "routed experts dropped";
        case MtpAbl::Shared: return "shared expert dropped";
        default: return "none";
    }
}

// Which captured target row a block row is built from.
//
// The block is fed a pair (hidden, embedding). The default pairing here is
// "same position": the row built for position p consumes (h(p), e(t_p)) and
// predicts t_(p+1). The reference formulation is the other one --
// (h(i), e(t_(i+1))) -> t_(i+2) -- so a block trained that way is being handed
// a hidden that is one target row AHEAD of the token it is paired with.
//
// That is a constant content misalignment and it has a distinctive signature:
// it is invisible to a row's own query (row p always pairs with row p) and
// only shows up once the attention mixes rows, which is exactly the pattern
// the measurements found -- full context worse than a 1-4 row window, and a
// uniform position offset changing nothing. KRK_MTP_HSHIFT=-1 tests it.
inline i64 mtp_hshift() {
    static const i64 s = [] {
        const char *e = std::getenv("KRK_MTP_HSHIFT");
        return e && *e ? static_cast<i64>(std::atoll(e)) : 0;
    }();
    return s;
}

inline i64 mtp_capture_row(i64 p) {
    const i64 r = p + mtp_hshift();
    return r < 0 ? 0 : r;
}

static ExpertSource mtp_source(const MtpBlock &m) {
    ExpertSource src;
    src.gate = m.exps_gate;
    src.up = m.exps_up;
    src.down = m.exps_down;
    src.n_expert = m.n_expert;
    src.n_embd = m.n_embd;
    src.n_ff_exp = m.n_ff_exp;
    return src;
}

// The block's experts through the stack's own policy, or pinned?
//
// KRK_MTP_PIN=1 keeps the nextn block's routed set device-resident.
//
// A speculative round runs the block up to --draft-tokens times per target
// forward, and the block's set is ONE layer's worth (256 experts, 816 MiB), so
// it fits even when the stack's 22 GiB does not. Left to the LRU it is cold
// often enough to pay a promotion per proposal; pinned, the propose pass and the
// refeed both become hits.
//
// **It is off by default because the A/B says the trade is a wash.** Measured on
// Tiel (ctx 4096, greedy, 48 tokens, k=2, 5 interleaved order-balanced pairs):
// propose 186-229 -> 134-135 ms, refeed 81-123 -> 32-34 ms, block acquires
// 157-256 ms -> 0 -- about 120 ms of block work removed -- but the pin holds 816
// of the 7594 MiB device budget, so the target's own promotions cost 105-180 ms
// more and tokens/s is 8.7 against 8.9 (median pair delta -0.2, the one pair
// that favoured the pin was the slowest unpinned run). Text is identical in all
// ten runs (`7ff494bbf10e`), so this is a residency trade, not arithmetic.
// Worth turning on where the block's share of the run is larger than it is here
// (more proposals per round, a cheaper target): it is a measured 33% cut of the
// propose pass, and the cost is the target's, not correctness.
bool Engine::mtp_pin_experts() {
    static const bool wanted = [] {
        const char *e = std::getenv("KRK_MTP_PIN");
        return e && e[0] == '1';
    }();
    if (!wanted) return false;
    const MtpBlock &m = model_.mtp();
    std::string why;
    const size_t added = model_.experts().pin_layer(mtp_source(m), m.layer, &why);
    if (added == 0) {
        KRK_INFO("mtp spec: the block's %d expert(s) stay on the LRU policy -- "
                 "%s",
                 m.n_expert, why.empty() ? "nothing to pin" : why.c_str());
        return false;
    }
    return true;
}

void Engine::mtp_embed(i32 tok) {
    const ModelConfig &mc = model_.cfg();
    const QuantTensor &embd = model_.tok_embd();
    be_->embed(mtp_e_, embd.data, embd.type, mc.n_vocab, mc.n_embd, &tok, 1);
    if (model_.had().embd_inverse)
        had_xform(mtp_e_, mtp_e_, 1, n_embd_, /*inverse=*/true,
                  /*gdn_perm=*/false);
}

// Walk the block over the captured positions the target has already computed.
// A speculative round leaves block rows that were built from the block's own
// hidden state for tokens the target later rejected, so the accepted prefix is
// re-fed here from the CAPTURED TARGET rows -- that is the whole "keep the
// block's KV in step" contract. It returns false when a position has no
// captured row, which means the caller's position bookkeeping is wrong; the
// caller refuses rather than drafting from a gap.
bool Engine::mtp_prime(i64 pos) {
    const MtpBlock &m = model_.mtp();
    const i64 end = std::min<i64>(pos, mtp_captured_);
    if (end < mtp_fed_) return false;
    const size_t as = be_->act_size();
    for (i64 p = mtp_fed_; p < end; p++) {
        const i32 tok = mtp_tok_[static_cast<size_t>(p)];
        if (tok < 0) return false;
        mtp_embed(tok);
        be_->copy_act(mtp_h_,
                      act_at(mtp_hidden_, as, mtp_capture_row(p) * n_embd_),
                      n_embd_);
        if (!mtp_block_forward(m, static_cast<i32>(p), false)) return false;
    }
    mtp_fed_ = end;
    return end == pos;
}

// The block's own routed + shared expert FFN. It goes through the same
// ExpertCache as a stack layer, keyed on the block's own index, so its experts
// page in and out of the same budget. `ws_xn_` is the block's post-attention
// norm output; the result lands in ws_x_ through the residual contract.
// The block's dense FFN, for the nextn heads that carry ffn_gate/up/down rather
// than routed banks (the qwen35 dense family). It reads `ws_xn_` -- the
// post-attention norm output the stack's own layers read -- and adds its result
// into `ws_ffn_` at that buffer's exponent, so the caller's single residual add
// covers this and the shared expert together.
void Engine::mtp_dense_ffn(const MtpBlock &m, i32 n, void *acc, i32 *exp) {
    const i64 ff = m.n_ff;
    if (ff <= 0 || !m.ffn_up.present() || !m.ffn_down.present()) return;
    if (m.ffn_gate.present()) {
        bool fused = false;
        // A scaled pair keeps the folded launch off: its epilogue computes
        // silu(gate)*up, which is nonlinear in each operand, so each sidecar has
        // to be applied to its own projection first.
        if (!m.ffn_gate.scaled() && !m.ffn_up.scaled()) {
            void *gout[2] = {ws_gate_, ws_up_};
            const void *gw[2] = {m.ffn_gate.data, m.ffn_up.data};
            const i64 gno[2] = {ff, ff};
            const DType gwts[2] = {m.ffn_gate.type, m.ffn_up.type};
            be_->gemm_group(gout, ws_xn_, gw, m.ffn_gate.type, gno, n_embd_, 2, n,
                            true, &fused, false, gwts);
        } else {
            gemm_w(ws_gate_, ws_xn_, m.ffn_gate, ff, n_embd_, n);
            gemm_w(ws_up_, ws_xn_, m.ffn_up, ff, n_embd_, n);
        }
        if (!fused) be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * ff);
    } else {
        gemm_w(ws_up_, ws_xn_, m.ffn_up, ff, n_embd_, n);
        be_->copy_act(ws_gate_, ws_up_, n * ff);
        be_->sigmoid_act(ws_gate_, n * ff);
        be_->mul_act(ws_gate_, ws_up_, n * ff);
    }
    gemm_w(ws_x2_, ws_gate_, m.ffn_down, n_embd_, ff, n);
    be_->add_residual(acc, ws_x2_, n, n_embd_, exp);
}

bool Engine::mtp_ffn(const MtpBlock &m, i32 n) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const i32 ne = m.n_expert;
    const i32 k = std::min<i32>(m.n_expert_used > 0 ? m.n_expert_used : ne, ne);
    const i64 ff = m.n_ff_exp;
    const i64 ff_sh = m.n_ff_shexp;

    // `ws_ffn_` exists only on a model with routed experts. A dense model has
    // no MoE workspace, and a block must not memset a buffer that is not there:
    // on a dense nextn block that was hipMemset(NULL) -> hipErrorInvalidValue
    // before the first token. With no accumulator the block's FFN adds straight
    // onto the residual, with the same exponent bookkeeping the attention add
    // used.
    void *acc = ws_ffn_ ? ws_ffn_ : ws_x_;
    i32 *exp = static_cast<i32 *>(ws_ffn_ ? ws_fexp_ : mtp_rep_);
    if (ws_ffn_) {
        be_->fill0(ws_ffn_,
                   static_cast<size_t>(n) * static_cast<size_t>(n_embd_) * as);
        be_->fill0(ws_fexp_, static_cast<size_t>(n) * sizeof(i32));
    }
    // A block with no router is dense (or broken); the dense FFN is what it
    // contributes, and a block with neither adds nothing rather than a zero
    // expert set silently standing in for one.
    const MtpAbl abl = mtp_abl();
    // `routed` is the ABLATED routed set: with KRK_MTP_ABL=routed the branch
    // below is skipped and only the shared expert contributes. The dense
    // fallback is a no-op on a block with no ffn_up/down, which is this one.
    const bool routed =
        abl != MtpAbl::Routed && m.router.present() && ne > 0 && k > 0 && ff > 0;
    if (!routed) {
        mtp_dense_ffn(m, n, acc, exp);
    } else {

        // Routing runs on the host, exactly as a stack layer's does: the logits are
        // tiny and the selection is what builds the plan.
        gemm_w(ws_router_, ws_xn_, m.router, ne, n_embd_, n);
        be_->sync();
        be_->download_f32(router_host_.data(), ws_router_, static_cast<i64>(n) * ne);

        std::vector<i32> sel(static_cast<size_t>(n) * static_cast<size_t>(k), -1);
        std::vector<f32> wt(static_cast<size_t>(n) * static_cast<size_t>(k), 0.0f);
        std::vector<f32> prob(static_cast<size_t>(ne), 0.0f);
        for (i32 t = 0; t < n; t++) {
            const f32 *logits = router_host_.data() + static_cast<i64>(t) * ne;
            if (mc.router_sigmoid) {
                for (i32 e = 0; e < ne; e++)
                    prob[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-logits[e]));
            } else {
                f32 mx = logits[0];
                for (i32 e = 1; e < ne; e++) mx = std::max(mx, logits[e]);
                f32 sum = 0.0f;
                for (i32 e = 0; e < ne; e++) {
                    prob[static_cast<size_t>(e)] = std::exp(logits[e] - mx);
                    sum += prob[static_cast<size_t>(e)];
                }
                const f32 inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (i32 e = 0; e < ne; e++) prob[static_cast<size_t>(e)] *= inv;
            }
            i32 *tsel = sel.data() + static_cast<size_t>(t) * k;
            f32 *twt = wt.data() + static_cast<size_t>(t) * k;
            for (i32 s = 0; s < k; s++) {
                i32 best = -1;
                f32 bestp = -1.0f;
                for (i32 e = 0; e < ne; e++) {
                    bool taken = false;
                    for (i32 p = 0; p < s; p++)
                        if (tsel[p] == e) { taken = true; break; }
                    if (taken) continue;
                    const f32 score = prob[static_cast<size_t>(e)] +
                                      (m.router_bias ? m.router_bias[static_cast<size_t>(e)] : 0.0f);
                    if (score > bestp) { bestp = score; best = e; }
                }
                tsel[s] = best;
                twt[s] = best >= 0 ? prob[static_cast<size_t>(best)] : 0.0f;
            }
            if (mc.expert_w_norm != 0) {
                f32 ksum = 0.0f;
                for (i32 s = 0; s < k; s++) ksum += twt[s];
                const f32 kinv = ksum > 0.0f ? 1.0f / ksum : 0.0f;
                for (i32 s = 0; s < k; s++) twt[s] *= kinv;
            }
            if (mc.expert_w_scale != 0.0f && mc.expert_w_scale != 1.0f)
                for (i32 s = 0; s < k; s++) twt[s] *= mc.expert_w_scale;
        }

        // One batched WARM read for the block's routed set. The probe runs once, so
        // this is the difference between eight sets of three sequential reads and
        // one queued read, which is what makes the replay affordable at all.
        {
            std::vector<i32> ids;
            std::vector<u8> seen(static_cast<size_t>(ne), 0);
            for (i32 t = 0; t < n; t++) {
                const i32 *tsel = sel.data() + static_cast<size_t>(t) * k;
                for (i32 s = 0; s < k; s++) {
                    const i32 e = tsel[s];
                    if (e < 0 || e >= ne || seen[static_cast<size_t>(e)]) continue;
                    seen[static_cast<size_t>(e)] = 1;
                    ids.push_back(e);
                }
            }
            if (ids.size() > 1)
                model_.experts().prefetch_layer(mtp_source(m), m.layer, ids.data(),
                                                static_cast<int>(ids.size()));
        }

        for (i32 e = 0; e < ne; e++) {
            plan_dev_.clear();
            alpha_dev_.clear();
            for (i32 t = 0; t < n; t++) {
                const i32 *tsel = sel.data() + static_cast<size_t>(t) * k;
                const f32 *twt = wt.data() + static_cast<size_t>(t) * k;
                for (i32 s = 0; s < k; s++) {
                    if (tsel[s] != e) continue;
                    plan_dev_.push_back(t);
                    alpha_dev_.push_back(twt[s]);
                }
            }
            if (plan_dev_.empty()) continue;
            const i64 mrows = static_cast<i64>(plan_dev_.size());
            be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(), mrows);
            be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                        static_cast<size_t>(mrows) * sizeof(f32));
            const u64 acq_t0 = now_us();
            const u64 acq_miss0 = model_.experts().loads() +
                                 model_.experts().host_hits();
            const ResidentExpert *re =
                model_.experts().acquire(mtp_source(m), m.layer, e);
            mtp_acq_ms_ += static_cast<f64>(now_us() - acq_t0) / 1000.0;
            mtp_acq_n_++;
            if (model_.experts().loads() + model_.experts().host_hits() !=
                acq_miss0)
                mtp_acq_miss_++;
            if (!re) {
                forward_error_ = expert_resource_error() + " (mtp block layer " +
                                 std::to_string(m.layer) + " expert " +
                                 std::to_string(e) + ")";
                return false;
            }
            be_->gather_rows(ws_xg_, ws_xn_, static_cast<const i32 *>(ws_plan_), mrows,
                             n_embd_);
            bool fused = false;
            if (re->gate.present()) {
                void *gout[2] = {ws_gateg_, ws_upg_};
                const void *gw[2] = {re->gate.data, re->up.data};
                const i64 gno[2] = {ff, ff};
                const DType gwts[2] = {re->gate.type, re->up.type};
                be_->gemm_group(gout, ws_xg_, gw, re->gate.type, gno, n_embd_, 2, mrows,
                                true, &fused, false, gwts);
            } else {
                be_->gemm(ws_upg_, ws_xg_, re->up.data, re->up.type, ff, n_embd_, mrows);
                be_->copy_act(ws_gateg_, ws_upg_, mrows * ff);
                be_->sigmoid_act(ws_gateg_, mrows * ff);
                be_->mul_act(ws_gateg_, ws_upg_, mrows * ff);
                fused = true;
            }
            if (!fused) be_->silu_mul(ws_gateg_, ws_gateg_, ws_upg_, mrows * ff);
            be_->gemm(ws_x2_, ws_gateg_, re->down.data, re->down.type, n_embd_, ff, mrows);
            be_->scatter_axpy_rows(ws_ffn_, ws_x2_, static_cast<const i32 *>(ws_plan_),
                                   static_cast<const f32 *>(ws_alpha_), mrows, n_embd_,
                                   nullptr, static_cast<i32 *>(ws_fexp_));
        }

        }

    if (abl != MtpAbl::Shared && ff_sh > 0 && m.shexp_up.present() &&
        m.shexp_down.present()) {
        if (m.shexp_gate.present())
            gemm_w(ws_gate_, ws_xn_, m.shexp_gate, ff_sh, n_embd_, n);
        gemm_w(ws_up_, ws_xn_, m.shexp_up, ff_sh, n_embd_, n);
        if (m.shexp_gate.present()) {
            be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * ff_sh);
        } else {
            be_->copy_act(ws_gate_, ws_up_, n * ff_sh);
            be_->sigmoid_act(ws_gate_, n * ff_sh);
            be_->mul_act(ws_gate_, ws_up_, n * ff_sh);
        }
        gemm_w(ws_x2_, ws_gate_, m.shexp_down, n_embd_, ff_sh, n);
        // Qwen3.5 scales the shared expert by a sigmoid of a per-token
        // projection -- `ffn_gate_inp_shexp`, which every block 40 in this
        // family carries. Skipping it scales the shared path by 1 where the
        // stack uses a learned per-token factor, which is a wrong answer that
        // still reads as fluent text.
        if (m.shexp_inp_gate.present()) {
            gemm_w(mtp_sgate_, ws_xn_, m.shexp_inp_gate, 1, n_embd_, n);
            be_->sigmoid_act(mtp_sgate_, n);
            be_->scale_rows(ws_x2_, mtp_sgate_, n, n_embd_);
        }
        be_->add_residual(acc, ws_x2_, n, n_embd_, exp);
    }

    if (acc != ws_x_)
        be_->add_residual(ws_x_, acc, n, n_embd_, static_cast<i32 *>(mtp_rep_),
                          static_cast<const i32 *>(ws_fexp_));
    return true;
}

bool Engine::mtp_block_forward(const MtpBlock &m, i32 pos, bool want_logits) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const i64 hd = m.head_dim;
    const i64 nh = m.n_head;
    const i64 nkv = m.n_head_kv;
    const i64 q = nh * hd;
    const i64 kvw = nkv * hd;
    if (!mtp_h_ || !mtp_e_ || !mtp_cat_ || !mtp_gate_ || !mtp_kv_k_ || !mtp_kv_v_)
        return false;
    if (q > q_dim_ || kvw > kv_dim_ || m.n_ff_exp > model_.n_ff_ws() ||
        m.n_ff > model_.n_ff_ws())
        return false;

    // DeepSeek's MTP fuses [embedding_norm | hidden_norm]; KRK_MTP_CONCAT=he is
    // the swapped order, kept as an arm because the concat side is not stated
    // in the GGUF and an order that is wrong here reads as a mediocre drafter
    // rather than as an error.
    static const bool hidden_first = [] {
        const char *e = std::getenv("KRK_MTP_CONCAT");
        return e && e[0] == 'h';
    }();
    u8 *cat = static_cast<u8 *>(mtp_cat_);
    const size_t half = static_cast<size_t>(n_embd_) * as;
    void *cat_h = hidden_first ? static_cast<void *>(cat)
                               : static_cast<void *>(cat + half);
    void *cat_e = hidden_first ? static_cast<void *>(cat + half)
                               : static_cast<void *>(cat);
    be_->rmsnorm(cat_e, mtp_e_, m.enorm, 1, n_embd_, mc.rms_eps);
    be_->rmsnorm(cat_h, mtp_h_, m.hnorm, 1, n_embd_, mc.rms_eps);
    gemm_w(ws_x_, mtp_cat_, m.eh_proj, n_embd_, 2 * n_embd_, 1);
    be_->fill0(mtp_rep_, sizeof(i32));

    be_->rmsnorm(ws_xn_, ws_x_, m.attn_norm, 1, n_embd_, mc.rms_eps);
    const bool packed = m.wq.n_out == 2 * q;
    if (packed && !ws_qpack_) return false;
    void *qp = packed ? ws_qpack_ : ws_q_;
    gemm_w(qp, ws_xn_, m.wq, m.wq.n_out, n_embd_, 1);
    gemm_w(ws_k_, ws_xn_, m.wk, kvw, n_embd_, 1);
    gemm_w(ws_v_, ws_xn_, m.wv, kvw, n_embd_, 1);
    if (packed) be_->qwen3_next_split(ws_q_, mtp_gate_, qp, 1, nh, hd);
    be_->qk_norm(ws_q_, ws_k_, m.q_norm, m.k_norm, nh, nkv, hd, 1, mc.rms_eps,
                 mc.qk_norm_wide);

    AttnDesc d;
    d.n_head = nh;
    d.n_kv = nkv;
    d.hd = hd;
    d.n_tok = 1;
    d.pos0 = pos;
    d.layer_stride = 0;
    d.pos_stride = kvw;
    d.layer = 0;
    d.window = 0;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd)));
    d.softcap = mc.attn_softcap;
    // ---- which rotary, over how much of the head, over what window --------
    // The block inherits the stack's rotary, and on a hybrid that inheritance
    // has three parts that can each be wrong: the TABLE (a windowed model
    // builds two, one for its full layers and one for its windowed ones), the
    // rotated WIDTH (rope_frac -- 84 of these 128 channels on the sidecar),
    // and the WINDOW the attention uses (always 0 here, i.e. the whole past).
    //
    // The sidecar states its own: rope.dimension_count 84, base 10000, and no
    // sliding-window key at all -- so on this pair "give the block its own
    // table" can only confirm what it already has, and the arm that claims to
    // change it is the control. The fraction and the window are not stated
    // twice, so those are the arms that can actually move the number.
    static const i32 rope_pick = [] {
        const char *e = std::getenv("KRK_MTP_ROPE");
        if (!e || !*e) return 0;
        const std::string v(e);
        if (v == "swa") return 1;
        if (v == "full") return 2;
        if (v == "none") return 3;
        return 0;
    }();
    static const f32 frac_override = [] {
        const char *e = std::getenv("KRK_MTP_FRAC");
        return e && *e ? static_cast<f32>(std::atof(e)) : -1.0f;
    }();
    static const i32 window_override = [] {
        const char *e = std::getenv("KRK_MTP_WINDOW");
        return e && *e ? std::atoi(e) : -1;
    }();
    // A layer of the wanted KIND is what selects a table: there is no other
    // handle on the two tables, and asking for a kind the model does not have
    // must fall back rather than silently read an empty vector.
    i32 rope_layer = 0;
    if (rope_pick != 0) {
        const bool want_swa = rope_pick == 1;
        for (i32 l = 0; l < mc.n_layer; l++)
            if (model_.is_swa(l) == want_swa) {
                rope_layer = l;
                break;
            }
    }
    const f32 frac =
        frac_override >= 0.0f ? frac_override : model_.rope_frac_at(rope_layer);
    // The default is a 4-row window, not the whole past. Measured on the
    // Lightning 30B sidecar (64 greedy steps, paired by position): full past
    // 68.8%, window 8 75.0%, window 4 78.1%, window 1 76.6%, and the same
    // ordering on a second prompt (50.0% -> 58.3%). The reason is the row: the
    // block's own KV is a reconstruction (KRK_MTP_ROWFID measured ~0.62 cosine
    // against the stack's hidden at every position, flat with index), so every
    // extra attended row adds more approximation error than context. A block
    // whose rows ARE the stack's hidden would want the full past back --
    // KRK_MTP_WINDOW=0 restores it, and the window is still per-run overridable.
    d.window = window_override >= 0 ? window_override : 4;
    // ---- position arms ----------------------------------------------------
    // RoPE only ever sees DISTANCES, so a per-row offset cannot change a single
    // score: row i's q and row j's k are both rotated by their own absolute
    // position, and q_i.k_j depends on i-j alone. KRK_MTP_ROPEOFF is therefore
    // a CONTROL (it must come out bit-identical, or the plumbing is wrong), not
    // a fix. A scale is the one remap that changes the geometry: it stretches
    // every distance, so if the block's relative spacing is wrong this is where
    // it shows. KRK_MTP_ROPE=none drops the rotation entirely, which is the
    // clean test of whether the rotation is wrong or merely unhelpful.
    static const i32 rope_off = [] {
        const char *e = std::getenv("KRK_MTP_ROPEOFF");
        return e && *e ? std::atoi(e) : 0;
    }();
    static const i32 rope_scale = [] {
        const char *e = std::getenv("KRK_MTP_POSSCALE");
        return e && *e ? std::atoi(e) : 1;
    }();
    const bool no_rope = rope_pick == 3;
    static const bool announced = [&] {
        if (frac_override >= 0.0f || window_override >= 0 || rope_pick != 0 ||
            rope_off != 0 || rope_scale != 1)
            KRK_INFO("mtp probe: block rotary arm -- frac %.5f of head_dim %lld, "
                     "table from layer %d (%zu freqs), window %d, rotate %s, "
                     "position %lld*i%+d",
                     static_cast<f64>(frac), static_cast<long long>(hd),
                     rope_layer, model_.inv_freq_at(rope_layer).size(),
                     d.window, no_rope ? "off" : "on",
                     static_cast<long long>(rope_scale), rope_off);
        return true;
    }();
    (void)announced;
    if (!no_rope)
        be_->rope(ws_q_, ws_k_, nh, nkv, hd, 1,
                  static_cast<i64>(pos) * rope_scale + rope_off,
                  model_.inv_freq_at(rope_layer).data(),
                  model_.rope_scale_at(rope_layer), frac, mc.rope_neox);
    be_->kv_append(mtp_kv_k_, mtp_kv_v_, ws_k_, ws_v_, d);
    be_->attention(ws_attn_, ws_q_, mtp_kv_k_, mtp_kv_v_, d);
    if (packed) {
        be_->sigmoid_act(mtp_gate_, q);
        be_->mul_act(ws_attn_, mtp_gate_, q);
    }
    gemm_w(ws_x2_, ws_attn_, m.wo, n_embd_, q, 1);
    // post_attention_norm is this family's pre-FFN norm (there is no separate
    // ffn_norm in block 40's tensor set), so it is applied to the residual, not
    // fused as an add+norm before the attention.
    const MtpAbl abl = mtp_abl();
    if (abl != MtpAbl::Attn && abl != MtpAbl::Both)
        be_->add_residual(ws_x_, ws_x2_, 1, n_embd_, static_cast<i32 *>(mtp_rep_));
    be_->rmsnorm(ws_xn_, ws_x_, m.post_attn_norm, 1, n_embd_, mc.rms_eps);
    if (abl != MtpAbl::Ffn && abl != MtpAbl::Both && !mtp_ffn(m, 1)) return false;

    if (want_logits) {
        // The head is charged separately from the block's body under the same
        // split env var: it is 194 MB of weights read for one row, and knowing
        // whether it is most of the propose pass decides what can be cut.
        static const bool hsplit = [] {
            const char *e = std::getenv("KRK_MTP_PROP_SPLIT");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        if (hsplit) be_->sync();
        const u64 th0 = now_us();
        be_->rmsnorm(ws_xn_, ws_x_, m.shared_head_norm, 1, n_embd_, mc.rms_eps);
        logit_head(ws_xn_);
        if (hsplit) {
            be_->sync();
            prop_head_ms_ += static_cast<f64>(now_us() - th0) / 1000.0;
        }
    }
    return true;
}

// Rank of one token id under a vocabulary row (1 = first). Softcap is
// monotone, so it cannot change the order; applying it keeps a reported logit
// on the same scale as the argmax paths that already do.
static i32 rank_of_token(const f32 *logits, i32 n_vocab, i32 tok, f32 cap) {
    if (tok < 0 || tok >= n_vocab) return 0;
    f32 mine = logits[tok];
    if (cap > 0.0f) mine = apply_logit_softcap(mine, cap);
    i32 rank = 1;
    for (i32 v = 0; v < n_vocab; v++) {
        if (v == tok) continue;
        f32 x = logits[v];
        if (cap > 0.0f) x = apply_logit_softcap(x, cap);
        if (x > mine) rank++;
    }
    return rank;
}

Engine::MtpProbe Engine::mtp_probe_step() {
    MtpProbe r;
    const ModelConfig &mc = model_.cfg();
    const MtpBlock &m = model_.mtp();
    if (!mtp_probe_ || !mtp_hidden_) {
        r.note = "the probe was not allocated for this run";
        return r;
    }
    if (!m.present) {
        r.note = "this file's nextn block is not loaded";
        return r;
    }
    const i32 P = static_cast<i32>(kv_pos_);
    if (P <= 0 || mtp_captured_ < P) {
        r.note = "no captured hidden rows for the prompt";
        return r;
    }
    r.pos = P - 1;
    const Timer timer;
    // The host row is deliberate: the probe reports the two logits it compared,
    // and the device top-k path returns an id with no value.
    auto read_argmax = [&](f32 *score) {
        be_->download_f32(logits_host_, ws_logits_, n_vocab_);
        const f32 cap = mc.logit_softcap;
        i32 besti = -1;
        f32 best = 0.0f;
        for (i32 v = 0; v < n_vocab_; v++) {
            f32 x = logits_host_[v];
            if (cap > 0.0f) x = apply_logit_softcap(x, cap);
            if (besti < 0 || x > best) {
                best = x;
                besti = v;
            }
        }
        *score = best;
        return besti;
    };
    // The target's answer for position P, read BEFORE the block writes its own
    // row into the same buffer. This is the token a speculative round has to
    // match to be accepted.
    r.target_id = read_argmax(&r.target_logit);

    const size_t as = be_->act_size();
    static const bool rowfid = [] {
        const char *e = std::getenv("KRK_MTP_ROWFID");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    if (rowfid && rowfid_a_.size() != static_cast<size_t>(n_embd_)) {
        rowfid_a_.assign(static_cast<size_t>(n_embd_), 0.0f);
        rowfid_b_.assign(static_cast<size_t>(n_embd_), 0.0f);
    }
    // A probe at a position the block has already passed means the sequence
    // restarted (every generation starts at position 0): drop the block's own
    // history rather than attend over tokens this prompt never had.
    if (mtp_fed_ < 0 || mtp_fed_ > r.pos) {
        be_->fill0(mtp_kv_k_,
                   static_cast<size_t>(P) * static_cast<size_t>(mtp_kv_dim_) * as);
        be_->fill0(mtp_kv_v_,
                   static_cast<size_t>(P) * static_cast<size_t>(mtp_kv_dim_) * as);
        mtp_fed_ = 0;
    }
    for (i32 p = static_cast<i32>(mtp_fed_); p < P; p++) {
        const i32 tok = mtp_tok_[static_cast<size_t>(p)];
        if (tok < 0) {
            r.note = "a captured position has no token id";
            return r;
        }
        mtp_embed(tok);
        be_->copy_act(mtp_h_,
                      act_at(mtp_hidden_, as,
                             mtp_capture_row(static_cast<i64>(p)) * n_embd_),
                      n_embd_);
        if (!mtp_block_forward(m, p, p == r.pos)) {
            // The refusal's own reason. `forward_error_` carries the expert
            // cache's message when a block expert could not be placed, and a
            // geometry check when a workspace is too narrow; without it this
            // note was a guess that named the wrong subsystem.
            r.note = "the block's forward was refused at position " +
                     std::to_string(p) +
                     (forward_error_.empty() ? "" : ": " + forward_error_);
            return r;
        }
        // ---- how good is this row, on the stack's own terms? -------------
        // Every HISTORY row is what the block's attention will later mix, so
        // each is scored the moment it is built: the cosine between the block's
        // row and the stack's captured hidden for the same position (how far
        // the reconstruction drifts), and the rank of the token that actually
        // follows that position under the stack's own norm+head applied to each
        // (whether the drift costs anything). The last row is skipped: it is
        // the query of this step, and its own measurement is the draft read
        // below.
        // The LAST row is the query of this step, and it is measured too: at
        // that position the next token is the target's own prediction, which is
        // already read. Its head logits are the draft read below, so they are
        // recomputed after the measurement rather than left clobbered.
        const bool last_row = (p == r.pos);
        if (rowfid && (p + 1) < static_cast<i32>(mtp_tok_.size()) &&
            rowfid_a_.size() == static_cast<size_t>(n_embd_)) {
            const i32 nxt = last_row ? r.target_id : mtp_tok_[static_cast<size_t>(p + 1)];
            const void *cap_row = act_at(mtp_hidden_, as,
                                         mtp_capture_row(p) * n_embd_);
            if (nxt >= 0) {
                be_->download_f32(rowfid_a_.data(), ws_x_, n_embd_);
                be_->download_f32(rowfid_b_.data(), cap_row, n_embd_);
                f64 dot = 0.0, sa = 0.0, sb = 0.0;
                for (i32 d = 0; d < n_embd_; d++) {
                    const f64 a = rowfid_a_[static_cast<size_t>(d)];
                    const f64 b = rowfid_b_[static_cast<size_t>(d)];
                    dot += a * b;
                    sa += a * a;
                    sb += b * b;
                }
                const f32 cos =
                    (sa > 0.0 && sb > 0.0)
                        ? static_cast<f32>(dot / (std::sqrt(sa) * std::sqrt(sb)))
                        : 0.0f;
                // The same head, twice: once on the block's row, once on the
                // stack's. rmsnorm is scale-invariant, so a row that is merely
                // scaled differently still competes fairly here.                be_->rmsnorm(ws_xn_, ws_x_, model_.out_norm(), 1, n_embd_, mc.rms_eps);
                logit_head(ws_xn_);
                be_->download_f32(logits_host_, ws_logits_, n_vocab_);
                const i32 blk_rank =
                    rank_of_token(logits_host_, n_vocab_, nxt, mc.logit_softcap);
                i32 cap_rank = 0;
                if (!last_row) {
                    be_->rmsnorm(ws_xn_, cap_row, model_.out_norm(), 1, n_embd_,
                                 mc.rms_eps);
                    logit_head(ws_xn_);
                    be_->download_f32(logits_host_, ws_logits_, n_vocab_);
                    cap_rank =
                        rank_of_token(logits_host_, n_vocab_, nxt, mc.logit_softcap);
                } else {
                    // Put the block's own draft logits back: this row's head
                    // output is what the probe (and the caller) reads next.
                    be_->rmsnorm(ws_xn_, ws_x_, m.shared_head_norm, 1, n_embd_,
                                 mc.rms_eps);
                    logit_head(ws_xn_);
                }
                rowfid_pos_.push_back(p);
                rowfid_cos_.push_back(cos);
                rowfid_blk_.push_back(blk_rank);
                rowfid_cap_.push_back(cap_rank);
                KRK_INFO("mtp row p=%d cos %.4f | block rank %d | capture rank %d",
                         p, static_cast<f64>(cos), blk_rank, cap_rank);
            }
        }
    }
    mtp_fed_ = P;
    r.draft_id = read_argmax(&r.draft_logit);
    // Where the target's token sits in the drafter's distribution. `read_argmax`
    // left the block's logits in logits_host_, so this is the same row the
    // argmax came from: no extra download, one pass over the vocabulary.
    if (r.target_id >= 0) {
        const f32 cap = mc.logit_softcap;
        f32 mine = logits_host_[r.target_id];
        if (cap > 0.0f) mine = apply_logit_softcap(mine, cap);
        i32 rank = 1;
        for (i32 v = 0; v < n_vocab_; v++) {
            if (v == r.target_id) continue;
            f32 x = logits_host_[v];
            if (cap > 0.0f) x = apply_logit_softcap(x, cap);
            if (x > mine) rank++;
        }
        r.target_rank = rank;
        r.target_in_draft = mine;
    }
    // The capture's own fidelity, opt-in because it costs one head GEMM and a
    // vocabulary download per step. The captured row goes through the model's
    // OWN output norm and head -- the nextn block is not involved -- so this
    // answers "is the stored row the row the stack itself would have predicted
    // from", which is the question a rank distribution cannot answer.
    static const bool capcheck = [] {
        const char *e = std::getenv("KRK_MTP_CAPCHECK");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    if (capcheck) {
        // logits_host_ is what the CALLER samples from (the prefill probe runs
        // before the first decode step), so the check's download is restored
        // afterwards. A diagnostic that changes the run it measures is not one.
        std::vector<f32> keep(logits_host_, logits_host_ + n_vocab_);
        const void *row = act_at(mtp_hidden_, as, static_cast<i64>(r.pos) * n_embd_);
        be_->rmsnorm(ws_xn_, row, model_.out_norm(), 1, n_embd_, mc.rms_eps);
        const void *xn = ws_xn_;
        if (model_.had().head_folded) {
            had_xform(ws_had_, ws_xn_, 1, n_embd_, false, false);
            xn = ws_had_;
        }
        logit_head(xn);
        r.capture_id = read_argmax(&r.capture_logit);
        std::memcpy(logits_host_, keep.data(),
                    static_cast<size_t>(n_vocab_) * sizeof(f32));
    }
    r.ok = r.draft_id >= 0 && r.target_id >= 0;
    r.ms = timer.ms();
    return r;
}

// KRK_FFN_ROWSCALE=0 restores the unscaled gated store for a layer with a
// sub-norm -- and with it the fused gate/up epilogue -- which is the A/B arm for
// the row-scaled store: on bitnet-2b it is the arm that clips 88 elements of
// layer 0's row and decodes different text, and it is the only way back to a
// measurement of what the scaled store costs.
static bool ffn_rowscale_on() {
    static const bool on = [] {
        const char *e = std::getenv("KRK_FFN_ROWSCALE");
        return !(e && e[0] == '0');
    }();
    return on;
}

// `dst` and `post_norm`: a two-branch family (gemma4) runs this FFN as the
// SHARED branch of a routed layer, so its result must not go straight into the
// residual (the block's own sum happens after the expert branch) and its output
// norm is `post_ffw_norm_1`, not the block's `post_ffw_norm`. Null on every
// other family, which is the graph that was here before.
void Engine::dense_ffn(const LayerWeights &L, i32 l, i64 n, void *dst,
                       const f32 *post_norm) {
    KRK_TRACE_COMP("dense-ffn");
    // gate/up share the FFN-norm output: one fused launch on the decode row.
    // A folded gate/up reads the transformed copy instead; the router of a
    // mixed dense/MoE model reads the primal ws_xn_, which is why the copy
    // exists rather than an in-place transform.
    const void *xn = ws_xn_;
    if (L.h_ffn_in) {
        had_xform(ws_had_, ws_xn_, n, n_embd_, false, false);
        xn = ws_had_;
    }
    // The activation rides the fused launch's epilogue where the backend can
    // express it (one launch for gate, up and silu instead of three); the flags
    // say which half happened, so the fallback issues exactly the calls the
    // folded path skipped. The folded form leaves ws_up_ untouched, and nothing
    // reads it after this point either way.
    // gemma's stack gates with GeGLU, not SwiGLU: gelu(gate) * up. The fused
    // group launch's epilogue only knows how to silu, so a GeGLU layer asks it
    // for the bare pair and applies its own activation -- one extra launch, on
    // a family whose graph is otherwise a copy of the SwiGLU one.
    const ModelConfig &dmc = model_.cfg();
    const bool gelu = dmc.gelu_ffn;
    // BitNet's squared ReLU. It is a THIRD activation over the same gate/up
    // pair, and until now only the routed-expert paths had an arm for it -- a
    // dense BitNet layer (every layer of every b1.58 file) ran silu(gate)*up.
    // The reference builds `squared_relu(x1) * x3`: relu^2 of the GATE leg,
    // multiplied by the up leg, so it is gelu's shape with a different
    // function and cannot be folded into the silu epilogue.
    const bool relu_sqr = dmc.ffn_relu_sqr;
    // BitNet's SubLN normalises this row, and an RMSNorm reads a row's
    // direction and never its size, so the store's absolute scale does not reach
    // the model. That is what makes the row-scaled store safe here, and it is
    // REQUIRED: on bitnet-2b this row's true peak is 3.5e6, 53x f16's ceiling,
    // so the unscaled store saturates 88 of its 6912 elements at layer 0 and the
    // norm then normalises a row whose direction is 108% wrong -- measured as
    // the first stage the device and --cpu arms disagree about, and as the
    // difference between their text. See AGENTS.md.
    const bool row_scaled = L.ffn_sub_norm != nullptr && ffn_rowscale_on();
    bool silu_done = false;
    // Same rule as the q/k/v group: the folded epilogue computes silu(gate)*up,
    // which is nonlinear in each, so a per-matrix scale has to be applied BEFORE
    // it -- which is exactly what the per-matrix path below does.
    //
    // A row-scaled store asks for the bare pair as well, and for a different
    // reason: the fused epilogue's warp holds ONE element of the row, so the row
    // maximum the scale is derived from does not exist inside it. The activation
    // runs as its own op for those layers.
    // Phi-3: gate and up are the two halves of one projection. n_ff is a whole
    // number of blocks wide, so the halves are row VIEWS of the one weight and
    // the gated path below runs unchanged -- same gemm_group, same silu
    // epilogue, same values the split tensors would have given.
    QuantTensor wgate = L.wgate;
    QuantTensor wup = L.wup;
    if (L.wup_fused.present()) {
        const i64 rb = static_cast<i64>(
            dtype_row_bytes(L.wup_fused.type, L.wup_fused.n_in));
        wgate = L.wup_fused;
        wgate.n_out = n_ff_;
        wup = L.wup_fused;
        wup.data = static_cast<u8 *>(L.wup_fused.data) + n_ff_ * rb;
        wup.n_out = n_ff_;
    }
    if (wgate.type == wup.type && !wgate.scaled() && !wup.scaled()) {
        void *gu[2] = {ws_gate_, ws_up_};
        const void *wgu[2] = {wgate.data, wup.data};
        const i64 ngu[2] = {n_ff_, n_ff_};
        be_->gemm_group(gu, xn, wgu, wgate.type, ngu, n_embd_, 2, n,
                        !gelu && !relu_sqr && !row_scaled, &silu_done);
    } else {
        gemm_w(ws_gate_, xn, wgate, n_ff_, n_embd_, n);
        gemm_w(ws_up_, xn, wup, n_ff_, n_embd_, n);
    }
    if (!silu_done) {
        if (gelu) {
            be_->gelu_act(ws_gate_, n * n_ff_);
            be_->mul_act(ws_gate_, ws_up_, n * n_ff_);
        } else if (relu_sqr) {
            // relu(gate)^2 * up. relu_sqr_act is in place on the gate, then the
            // up leg multiplies -- the same two launches GeGLU takes.
            //
            // `row_scaled` is NOT applied on this arm: there is no
            // relu_sqr+mul+row-max kernel, and the scale is a numerical guard
            // (it divides a row by its max and the sub-norm below divides it
            // back in), not part of the function -- `ffn_rowscale_on()` off
            // already runs every arm unscaled. What it does buy is f16 headroom,
            // and relu^2 is quadratic, so this arm reaches the store with larger
            // magnitudes than the silu one. Recorded in the TODO as a
            // range-only difference, not a correctness one.
            be_->relu_sqr_act(ws_gate_, n * n_ff_);
            be_->mul_act(ws_gate_, ws_up_, n * n_ff_);
        } else {
            be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * n_ff_,
                          row_scaled ? n_ff_ : 0,
                          row_scaled ? static_cast<f32 *>(ws_fscl_) : nullptr);
        }
    }
    dump_row("attn.ffng", l, ws_gate_, n_ff_, n,
             row_scaled ? static_cast<const f32 *>(ws_fscl_) : nullptr);
    // BitNet's SubLN, FFN half: an RMSNorm on the GATED INTERMEDIATE, between
    // silu(gate)*up and the down projection -- not on the pre-FFN input, which
    // is what ffn_norm already is. `ffn_sub_norm` is n_ff wide against
    // ffn_norm's n_embd, and that is the shape that says which is which. In
    // place, so the folded-down transform below reads the normalized row; the
    // dump above is still the raw gated product.
    if (L.ffn_sub_norm)
        be_->rmsnorm(ws_gate_, ws_gate_, L.ffn_sub_norm, n, n_ff_, dmc.rms_eps);
    // The intermediate is a folded input as well (its width is n_ff, wider
    // than anything above), which is safe to put over ws_had_ because the
    // gate/up gemms have already consumed their copy.
    const void *gn = ws_gate_;
    if (L.h_ffn_down) {
        had_xform(ws_had_, ws_gate_, n, n_ff_, false, false);
        gn = ws_had_;
    }
    const f32 *pn = post_norm ? post_norm : L.ffn_post_norm;
    if (pn) {
        // Branch post norm: the down projection's own output is normalized
        // with the FFN branch's weight, and only then added. That costs the
        // folded residual store -- the norm has to read the projection, so the
        // epilogue cannot write straight into ws_x_ -- which is why this runs
        // as gemm + norm + add.
        gemm_w(ws_x2_, gn, L.wdown, n_embd_, n_ff_, n);
        const f32 peps =
            dmc.post_norm_eps > 0.0f ? dmc.post_norm_eps : dmc.rms_eps;
        be_->rmsnorm(ws_x2_, ws_x2_, pn, n, n_embd_, peps, nullptr);
        dump_row("attn.down", l, ws_x2_, n_embd_, n);
        if (dst) be_->copy_act(dst, ws_x2_, static_cast<i64>(n) * n_embd_);
        else be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(n) * n_embd_);
    } else if (dst) {
        // A branch with no norm of its own still cannot take the folded
        // residual store, because it has no residual to store into.
        gemm_w(dst, gn, L.wdown, n_embd_, n_ff_, n);
        dump_row("attn.down", l, dst, n_embd_, n);
    } else if (dmc.residual_scale_ffn) {
        // MiniCPM3 scales the FFN result by `scale_depth/sqrt(n_layer)` before
        // it joins the residual, exactly as it scales the attention result.
        // The folded store writes straight into the residual, so it cannot
        // carry a multiply: this is the gemm + scale + add the reference does,
        // gated on the flag so every other family keeps the fold.
        gemm_w(ws_x2_, gn, L.wdown, n_embd_, n_ff_, n);
        // The branch as projected, before the residual scale -- the other side
        // of the multiply, so a trace can prove the factor is applied ONCE and
        // with the value the config says (test_minicpm3_scale_arithmetic).
        if (dmc.attn_out_scale != 1.0f) {
            dump_row("ffn.pre", l, ws_x2_, n_embd_, n);
            be_->scale_act(ws_x2_, dmc.attn_out_scale, n, n_embd_, n_embd_);
        }
        dump_row("attn.down", l, ws_x2_, n_embd_, n);
        be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(n) * n_embd_);
    } else {
        gemm_residual(ws_x_, gn, L.wdown.data, L.wdown.type, n_embd_, n_ff_, n,
                      ws_x2_, L.wdown.gs);
        dump_row("attn.down", l, ws_x2_, n_embd_, n);
    }
}

bool Engine::generate_diffusion(const GenerateParams &p, GenerateResult *res,
                                const std::vector<i32> &ids) {
    const ModelConfig &mc = model_.cfg();
    const i32 gen_length = p.max_tokens > 0 ? p.max_tokens : 32;
    // The file's own schedule, with the caller's override on top. A block
    // length of 0 means one block over the whole generation, which is the
    // reference's degenerate-but-legal arm rather than a missing value.
    const i32 block = diffuse_block_ > 0
                          ? diffuse_block_
                          : (mc.diffuse_block_length > 0 ? mc.diffuse_block_length
                                                         : gen_length);
    // Steps per BLOCK. The reference divides the user's total by the block
    // count; a default of one step per block is the fastest schedule that still
    // commits every position, and every step is a full-sequence forward.
    const i32 steps = diffuse_steps_ > 0
                          ? diffuse_steps_
                          : std::max<i32>(1, block > 0 ? std::min<i32>(block, 8)
                                                       : 1);
    if (block <= 0 || gen_length <= 0) {
        res->error = "diffusion: gen_length/block_length are not a schedule (" +
                     std::to_string(gen_length) + "/" + std::to_string(block) + ")";
        res->finish = FinishError;
        return false;
    }

    DiffusionParams dp;
    dp.gen_length = gen_length;
    dp.block_length = block;
    dp.steps = steps;
    dp.mask_id = mc.diffuse_mask_id;
    // The EOS rule rides the file's own end-of-text ids, and only under greedy:
    // a sampled run keeps the sampler's answer rather than truncating on it.
    dp.eos_id = p.sampler.greedy ? tok_.eos() : -1;
    // The reference forces EOS off at positions that must not end the sequence
    // (a block that still has unfilled positions before it). Same switch, same
    // reason: an early EOS in the middle of a denoised block truncates a
    // sentence the model was still filling in.
    dp.eos_forbid_prefix = true;

    std::vector<i32> seq = ids;
    seq.resize(ids.size() + static_cast<size_t>(gen_length), dp.mask_id);
    const i32 prompt_len = static_cast<i32>(ids.size());
    const i32 n_vocab = n_vocab_;
    std::vector<f32> logits(static_cast<size_t>(seq.size()) * n_vocab);

    const Timer timer;
    const i32 forwards = krk::diffusion_generate(
        seq, prompt_len, n_vocab, dp,
        [&](const i32 *toks, i32 n_tok, f32 *out) {
            return diffuse_forward(toks, n_tok, 0, out);
        });
    if (forwards < 0) {
        res->error = "diffusion: the schedule was refused (gen_length, "
                     "block_length and steps do not describe a plan)";
        res->finish = FinishError;
        return false;
    }
    res->decode_ms = timer.ms();
    res->decode_steps = forwards;
    // A dLLM's "time to first token" is the prompt pass; every later number is
    // per-step, because one step commits many tokens at once.
    res->prefill_ms = 0;
    res->tokens.assign(seq.begin() + prompt_len, seq.end());
    // The EOS rule: a diffused block can commit EOS at an interior position
    // only when the schedule allowed it, so truncation is by the token itself.
    if (dp.eos_id >= 0) {
        for (size_t i = 0; i < res->tokens.size(); i++) {
            if (res->tokens[i] == dp.eos_id) {
                res->tokens.resize(i);
                res->finish = FinishStop;
                break;
            }
        }
    }
    // Literal text, through the same detokenizer the decode loop uses.
    if (!res->tokens.empty())
        res->text = tok_.decode(res->tokens.data(), res->tokens.size());
    return true;
}

bool Engine::diffuse_forward(const i32 *toks, i32 n, i32 pos0, f32 *logits) {
    if (!toks || n <= 0 || !logits) return false;
    // Refused by NUMBER, not by overrun: see diffuse_max_tokens_ at init.
    if (diffuse_max_tokens_ > 0 && n > diffuse_max_tokens_) {
        forward_error_ = "diffusion: a denoising pass over " + std::to_string(n) +
                         " tokens exceeds the " +
                         std::to_string(diffuse_max_tokens_) +
                         " this engine sized its workspace for (prompt + "
                         "gen_length must fit in n_ctx)";
        KRK_ERROR("%s", forward_error_.c_str());
        return false;
    }
    if (!forward_core(toks, n, pos0, LogitsNone)) return false;
    for (i32 r = 0; r < n; r++) {
        // head_compute reads the FINAL residual row, which forward_core leaves
        // in ws_x_ -- it only overwrites ws_xn_ (the normed copy), so a second
        // row's head does not disturb the first row's.
        head_compute(r);
        be_->download_f32(logits + static_cast<i64>(r) * n_vocab_, ws_logits_,
                          n_vocab_);
    }
    return true;
}

bool Engine::forward(const i32 *toks, i32 n, i32 pos0, bool want_logits) {
    KRK_TRACE_COMP(n > 1 ? "prefill-forward" : "decode-forward");
    if (!forward_core(toks, n, pos0, want_logits ? LogitsLast : LogitsNone))
        return false;
    // The Engram window: keep the last n_gram-1 token ids of the sequence so a
    // chunk's first token can see across the boundary. Updated once per pass,
    // after the pass, because the whole chunk's windows are read from it (and
    // from the chunk itself) before this runs.
    if (model_.cfg().ple && model_.cfg().hc_sinkhorn && toks && n > 0) {
        const size_t keep = static_cast<size_t>(model_.cfg().ple_ngram - 1);
        eng_hist_.insert(eng_hist_.end(), toks, toks + n);
        if (eng_hist_.size() > keep)
            eng_hist_.erase(eng_hist_.begin(),
                            eng_hist_.end() - static_cast<ptrdiff_t>(keep));
    }
    kv_pos_ = std::max<i64>(kv_pos_, static_cast<i64>(pos0) + n);
    // The KV tier promotes rows as the context grows, so the pool's budget is
    // re-derived from a live reading: every prefill, and every 16 decode rows
    // (a hipMemGetInfo is a query; sampling it 40 times a token would be a
    // different thing). See reclamp_expert_budget.
    if (n > 1 || (++reclamp_tick_ % 16u) == 0)
        reclamp_expert_budget(n > 1 ? "prefill" : "decode");
    // The folded residual store (Backend::gemm_accumulate) divides the
    // contribution by the row's CURRENT storage exponent and sums in f16, so it
    // is only sound while that exponent still holds the sum. A row that left
    // f16 is exactly the case where it may not -- and there the fold stores
    // 65504 where the true value is larger, the one defect class this repo
    // treats as a failure rather than a rounding (see the `residual` counter).
    // So the fold is dropped for the rest of the run once any residual row has
    // been stored scaled. The evidence is the residual store's own exponent,
    // read on the cadence the budget re-clamp above already established --
    // every prefill, every 16th decode row: one 4-byte query, sticky, so a
    // model whose rows stay inside f16 pays it and keeps the one-launch fold,
    // and bitnet-2b (exponent 2 on its first prompt row) never folds at all.
    //
    // It can only REMOVE a wrong answer, never change a right one: where the
    // fold runs its store is bit-identical to the gemm+add_residual pair it
    // replaces, and where it is dropped that pair is what runs.
    if (!resid_scaled_ && (n > 1 || (reclamp_tick_ % 16u) == 0) &&
        be_->residual_exponent_max() > 0) {
        resid_scaled_ = true;
        KRK_INFO("residual fold off: a residual row left f16 (max exponent %d), "
                 "so the folded projection store is dropped for this run",
                 be_->residual_exponent_max());
    }
    // KRK_VRAM_TRACE=1 prints the ledger after every forward with more than one
    // row. The re-clamp above can only fire on a live reading, so when it does
    // fire the question is always "which allocation took the bytes between the
    // init ledger and here", and one print per forward answers it without a
    // profiler. Off by default: two hipMemGetInfo calls per step is exactly the
    // kind of query that must not sit in a decode loop.
    {
        static const bool trace_vram = [] {
            const char *e = std::getenv("KRK_VRAM_TRACE");
            return e && *e && e[0] != '0';
        }();
        if (trace_vram && n > 1) log_vram_ledger("after prefill");
    }
    return true;
}

bool Engine::kv_rollback(i64 pos) {
    if (pos < 0 || pos > kv_pos_) return false;
    if (pos == kv_pos_) return true;
    // A shortconv ring survives a partial rewind (model.hpp,
    // `speculative_rewind_ok`): the engine keeps it deeper than the window, so
    // a suffix rewrite can never reach a row a retained convolution reads.
    // Dropping kv_pos_ is then the whole story -- unlike the delta rule below,
    // there is no state to put back.
    if (model_.is_recurrent() && !model_.has_lfm_conv()) {
        // The delta rule has no inverse. Rewinding the attention rows alone
        // would leave the recurrent state describing tokens the caller is
        // about to replace, so the honest options are a full rewind (the
        // server's per-request isolation, after which the prompt is replayed
        // and the state is rebuilt token by token) or refusing. A partial
        // rewind is what speculative decoding needs, so it is refused there.
        if (pos != 0) {
            KRK_WARN("a recurrent model cannot rewind to position %lld — "
                     "only a full rewind rebuilds the delta-rule state",
                     static_cast<long long>(pos));
            return false;
        }
        recurrent_reset();
    }
    // If the shrink is large, clear the tail once so the cache never carries
    // stale keys from an unrelated context; a small shrink (the speculative
    // hot path) is pure pointer arithmetic. Every forward fully overwrites
    // the rows it touches and attention never reads past kv_pos_, so in both
    // cases the cache is logically exact afterwards.
    if (kv_pos_ - pos > 64 && !kv_rollback_zero(pos)) {
        KRK_WARN("KV rollback refused: retained history could not be staged within "
                 "the host RAM reserve");
        return false;
    }
    kv_pos_ = pos;
    return true;
}

bool Engine::kv_rollback_zero(i64 pos) {
    // Resolve every layer in both planes before mutating any retained bytes.
    // Otherwise K can be zeroed successfully and V can then refuse promotion,
    // leaving the cache half-rolled-back while kv_pos_ still advertises the
    // original history. The pointers remain valid through the fill pass: no
    // tier admission occurs between this preflight and its corresponding fill.
    std::vector<void *> kbase(static_cast<size_t>(model_.cfg().n_layer), nullptr);
    std::vector<void *> vbase(static_cast<size_t>(model_.cfg().n_layer), nullptr);
    for (i64 l = 0; l < model_.cfg().n_layer; l++) {
        // Only a writing layer owns a plane; a shared one is zero-filled above
        // through its source and asking for its own pointer would refuse a
        // rollback the model never needed.
        if (!model_.layer_writes_kv(l)) continue;
        kbase[static_cast<size_t>(l)] = kvt_k_.base(l);
        vbase[static_cast<size_t>(l)] = kvt_v_.base(l);
        if (!kbase[static_cast<size_t>(l)] || !vbase[static_cast<size_t>(l)])
            return false;
    }
    for (i64 l = 0; l < model_.cfg().n_layer; l++) {
        if (!model_.layer_writes_kv(l)) continue;
        const size_t row = be_->kv_row_bytes(model_.kv_dim_at(l), cfg_.kv_quant == 1 ? KvQuant::Int8 : KvQuant::F16);
        u8 *kb = static_cast<u8 *>(kbase[static_cast<size_t>(l)]) +
                 static_cast<size_t>(pos) * row;
        u8 *vb = static_cast<u8 *>(vbase[static_cast<size_t>(l)]) +
                 static_cast<size_t>(pos) * row;
        be_->fill0(kb, (static_cast<size_t>(kv_pos_) - static_cast<size_t>(pos)) * row);
        be_->fill0(vb, (static_cast<size_t>(kv_pos_) - static_cast<size_t>(pos)) * row);
    }
    return true;
}

bool Engine::configure_expert_cache(std::string *err) {
    const ModelConfig &mc = model_.cfg();
    // The memory ceiling is process-wide and applies to every run, dense
    // included: a dense model larger than RAM streams its weights, and the same
    // paging death is available there. Set it before the dense early return, or
    // the cap exists only for MoE files.
    {
        size_t class_ram = be_ ? be_->host_total_bytes() : 0;
        if (cfg_.ram_tier_gb > 0) {
            const size_t tier = static_cast<size_t>(cfg_.ram_tier_gb) << 30;
            if (class_ram == 0 || tier < class_ram) class_ram = tier;
        }
        const size_t ceiling = krk::mem_ceiling_bytes(class_ram);
        if (ceiling > 0) {
            const size_t gib_ = static_cast<size_t>(1) << 30;
            KRK_INFO("mem policy: process working set capped at %.1f GiB "
                     "(%.0f%% of %.1f GiB planned)%s",
                     static_cast<f64>(ceiling) / static_cast<f64>(gib_),
                     krk::mem_fraction() * 100.0,
                     static_cast<f64>(class_ram) / static_cast<f64>(gib_),
                     cap_working_set(ceiling) ? "" : " -- the cap could NOT be set");
        }
    }
    if (!mc.is_moe || mc.n_expert <= 0) return true;

    // Device memory policy, in one place, because it decides how much of the
    // card the user is left with.
    //
    // The rule: plan for 6 GiB of device memory in total, and step up by 2 GiB
    // only when there is both the headroom for it and a model that needs it,
    // never above 14 GiB. A small model on a large card therefore uses a few
    // hundred MiB, a large MoE on a 16 GiB card stops at 14 GiB instead of
    // filling the card, and a model that genuinely needs more than 6 GiB gets
    // it in 2 GiB increments rather than all at once. Everything the run does
    // not plan for -- the KV cache the user asked for, a second application,
    // the driver's own allocations -- is what the headroom is for.
    //
    // This replaces "60% of whatever is free right now", which on a 15.9 GiB
    // card produced a 15.04 GiB working set: the cache made the decision for the
    // user and left 0.88 GiB for everything else.
    const size_t gib = static_cast<size_t>(1) << 30;
    const size_t kCapFirst = 6 * gib;
    const size_t kCapStep = 2 * gib;
    // The ceiling was raised from 12 GiB to 14 GiB, chosen against a 15.9 GiB
    // card for the big MoEs. Measured against the real files, the 12 GiB ceiling
    // is what keeps a *small* MoE streaming forever: Laguna XS 2.1 IQ3_XXS carries
    // 11.20 GiB of experts and 0.86 GiB of dense weights, so its whole routed set
    // needs 12.06 GiB of budget -- and at the 12 GiB cap it sat at 81.6% resident,
    // re-reading hundreds of experts from disk on every token even though the card
    // had the room. A card that can hold the model should hold it: a miss that costs
    // a file read is only worth its VRAM when the set genuinely does not fit.
    const size_t kCapMax = 14 * gib;
    const size_t kReserve = 512u * 1024u * 1024u; // driver + fragmentation
    const size_t total_bytes = model_.total_expert_bytes();
    const size_t total_vram = be_->device_total_bytes();
    const size_t free_now = be_->device_free_bytes();
    const size_t used_now = (total_vram > free_now) ? total_vram - free_now : 0;
    // What the run needs if nothing constrains it: the fixed allocations that
    // already exist (weights, KV, workspaces) plus the whole routed set.
    const size_t demand = used_now + total_bytes;
    const size_t hard =
        total_vram > kReserve ? total_vram - kReserve : total_vram / 2;

    // An explicit restriction wins over the policy. --vram-cap-mb is the exact
    // form and --vram-tier is the class form of the same thing, the twin of
    // --ram-tier: "behave as if the card were this big, in GiB". The tier is
    // clamped to the card actually installed, so a plan made for the 12 GiB
    // machine is safe on the 16 GiB one, and both are clamped to the headroom
    // the policy itself would refuse to plan for.
    size_t want = 0;
    if (cfg_.vram_cap_mb > 0) {
        want = static_cast<size_t>(cfg_.vram_cap_mb) * 1024u * 1024u;
    } else if (cfg_.vram_tier_gb > 0) {
        want = static_cast<size_t>(cfg_.vram_tier_gb) << 30;
        if (total_vram > 0 && want > total_vram) want = total_vram;
    }
    size_t vram_cap = 0;
    if (want > 0) {
        vram_cap = want > hard ? hard : want;
    } else {
        vram_cap = kCapFirst < hard ? kCapFirst : hard;
        while (vram_cap < kCapMax && vram_cap + kCapStep <= hard &&
               demand > vram_cap)
            vram_cap += kCapStep;
    }

    // What the card can actually be asked for, with the reserve held back.
    const size_t vram_reserve = krk::vram_reserve_bytes(total_vram);
    const size_t room_now = free_now > vram_reserve ? free_now - vram_reserve : 0;

    size_t budget = 0;
    if (cfg_.expert_cache_mb > 0) {
        // An explicit restriction wins over the policy: the user has said how
        // much they want to spend, in expert bytes alone.
        budget = static_cast<size_t>(cfg_.expert_cache_mb) * 1024u * 1024u;
        // ... but not over the card. A budget the machine cannot hold is
        // refused rather than trimmed: silently keeping less than the user
        // asked for turns their experiment into a different one, and the run
        // would report a residency they did not choose. Name the largest
        // budget that does fit, so the fix is one flag away.
        // The decision itself lives in krk/vram_guard.hpp so the test suite can
        // drive it: this refusal is the one path where being wrong either
        // changes the user's experiment quietly (a trim) or leaves a card with
        // nothing free (no refusal).
        size_t largest_fits = 0;
        if (!krk::expert_budget_fits(budget, total_vram, used_now, vram_reserve,
                                     &largest_fits, err))
            return false;
    } else if (be_->caps().vram_free == 0) {
        // CPU backend, or a driver that cannot say: no device memory to
        // divide, so a generous fixed share of host RAM instead.
        budget = static_cast<size_t>(512) * 1024u * 1024u;
        if (total_bytes > 0 && total_bytes < budget) budget = total_bytes;
    } else {
        // A partial budget is worse than useless at MoE scale: top-k routing
        // touches every expert within a few dozen tokens, so a set that only
        // partly fits churns (evict + free + alloc + re-upload) forever, which
        // costs far more than the memory the cap saves. So the budget is
        // whatever is left of the cap, and when the whole routed set fits in
        // that, the set is what it holds.
        budget = vram_cap > used_now ? vram_cap - used_now : 0;
        if (total_bytes > 0 && total_bytes < budget) budget = total_bytes;
        if (budget == 0 && total_bytes > 0) {
            // The fixed allocations (weights, KV, workspaces) already fill the
            // cap, which on a small card means the weights plus a default
            // context do. A zero budget is legal -- every acquire is a file
            // read -- but it is the worst shape this engine has, so keep a few
            // experts resident and say so. --ctx is the real fix: KV is the
            // elastic part, and a card this full wants a shorter context, not
            // a slower cache.
            const size_t floor_bytes = model_.max_expert_bytes() * 4;
            budget = floor_bytes < total_bytes ? floor_bytes : total_bytes;
            KRK_WARN("vram policy: weights, KV and workspaces already fill "
                     "the %.1f GiB cap; expert budget floored at %.1f MiB "
                     "(4 experts). A smaller --ctx is what actually buys VRAM "
                     "back here.",
                     static_cast<f64>(vram_cap) / static_cast<f64>(gib),
                     static_cast<f64>(budget) / (1024.0 * 1024.0));
        }
    }
    // The plan cannot see the workspaces and the KV tier, which are allocated
    // after it, so it is clamped to the free memory measured HERE minus the
    // reserve, and the warm-up enforces the same reserve again at promotion
    // time. Both numbers are reported: they are the plan and the floor.
    if (total_vram > 0 && budget > room_now) budget = room_now;
    KRK_INFO("vram policy: %.1f GiB cap%s (%.1f GiB already in use, %.1f GiB "
             "demand, %.1f GiB free of %.1f GiB, %.1f GiB reserved) -> expert "
             "budget %.1f MiB",
             static_cast<f64>(vram_cap) / static_cast<f64>(gib),
             (cfg_.vram_tier_gb > 0 && cfg_.vram_cap_mb <= 0)
                 ? " [planned for the --vram-tier class]"
                 : (cfg_.vram_cap_mb > 0 ? " [--vram-cap-mb]" : ""),
             static_cast<f64>(used_now) / static_cast<f64>(gib),
             static_cast<f64>(demand) / static_cast<f64>(gib),
             static_cast<f64>(free_now) / static_cast<f64>(gib),
             static_cast<f64>(total_vram) / static_cast<f64>(gib),
             static_cast<f64>(vram_reserve) / static_cast<f64>(gib),
             static_cast<f64>(budget) / (1024.0 * 1024.0));

    // The number of device slots the budget above is sized for, when this code
    // knows it exactly rather than estimating it. Two cases do: an explicit
    // --expert-cache-slots N is a count and is enforced as one, and a budget
    // that covers the whole routed set is the set. In between, the cache derives
    // the number from the bytes, because a corpus can mix expert sizes (a
    // Q4_K_M file mixes Q4_K and Q6_K down-projections) and then no single
    // per-slot figure is the truth: on Qwen3-MoE-4x0.6B the estimate reads 104
    // at one layer and 126 at the next while the budget covers all 112.
    size_t slot_budget = 0;
    if (cfg_.expert_cache_slots > 0) {
        const size_t one = model_.max_expert_bytes();
        if (one > 0)
            budget = one * static_cast<size_t>(cfg_.expert_cache_slots);
        slot_budget = static_cast<size_t>(cfg_.expert_cache_slots);
    } else if (total_bytes > 0 && budget >= total_bytes) {
        slot_budget = model_.expert_slot_count();
    }

    // WARM is sized independently of the device budget: VRAM holds what is
    // about to be used, RAM holds what was just used and the read-through copy
    // of what is being used. It is pageable (not pinned) and sized from the
    // machine, because a fixed number is wrong everywhere but on the box it was
    // chosen for: the same runtime has to behave on 24 GiB and on 96 GiB.
    //
    // A negative setting (the default) targets the whole routed expert set,
    // capped by half of what is left after a reserve. The set is the model's
    // home and the device budget is the hot subset on top of it, so a VRAM miss
    // becomes a promotion out of RAM instead of a read of the file; holding all
    // of it is the point, and the cap is what keeps a 24 GiB box from swapping.
    // Half of the post-reserve pool is the design target (25-50% of RAM) with
    // the reserve holding the mapping, the trunk, the KV cache and the OS. A
    // positive value is that many MiB — the explicit restriction, 0 disables
    // WARM entirely. The CPU backend's mapping is host memory already, so auto
    // stays off there; a tier that cannot hold a few experts only adds copies.
    // Host memory policy, the mirror of the device one above and for the same
    // reason: a tier that sizes itself from "whatever is free right now" is a
    // decision made for the user.
    //
    // The size comes from the tiering spec's RAM table, which is a SUBTRACTION
    // and not a fraction:
    //
    //     warm = RAM - headroom - pinned_io - free_reserve
    //
    //     RAM    headroom  pinned  reserve   warm (spec)
    //      24      5        2       3        14 GiB
    //      32      7        2       3        20 GiB
    //      48     10        3       3        32 GiB
    //      96     16        6       4        70 GiB
    //
    // The old rule was RAM/4. On the 96 GiB class that is 24 GiB against a
    // routed corpus of 19.4 GiB, which happens to fit only just and leaves the
    // tier churning at the margin on a long conversation; on the 48 GiB class
    // it is 12 GiB against a corpus that does not fit at all. The subtraction
    // reproduces the table to within 5% at every class while staying smooth
    // between them, and it is the same shape as the device policy above
    // (reserve the things that are not cache, then spend what is left).
    //
    // Everything the run does not plan for -- the mapping, the dense trunk, the
    // OS -- is what those four terms are.
    const size_t ram_installed = be_->host_total_bytes();
    // --ram-tier applies the spec's tiering table to a CLASS rather than to
    // this box: plan for 16/24/32/48/64/96 GiB so a deployment can be measured
    // and tuned on a 96 GiB workstation for a 32 GiB target without pretending
    // the extra RAM away by hand. 0 (the default) is the installed size. A tier
    // larger than the machine is clamped back to the machine, and the
    // availability clamp below still applies to whatever is free right now, so
    // a tier is a budget rather than a claim on memory that is not there.
    size_t ram_total = ram_installed;
    if (cfg_.ram_tier_gb > 0) {
        const size_t tier = static_cast<size_t>(cfg_.ram_tier_gb) << 30;
        ram_total =
            ram_installed > 0 && tier > ram_installed ? ram_installed : tier;
    }
    size_t host_budget = 0;
    // Hard ceiling for everything this process plans to hold, VRAM and host
    // alike: 80% of installed by default (KRK_MEM_FRACTION). The host half is
    // checked again at every staging round through SharedHostRamBudget.
    // The ceiling follows the machine the run is planned for (--ram-tier), so a
    // 96 GiB workstation can check the 16/24/32 GiB targets.
    const size_t host_ceiling = krk::mem_ceiling_bytes(ram_total);
    const size_t live_host_avail = be_->host_available_bytes();
    const size_t host_reserve = krk::host_reserve_bytes(ram_total);
    // Three tiers, not two: the file is mapped, so the disk source IS page
    // cache, and every page it touches is RAM the WARM tier cannot spend
    // twice. Sizing WARM from live free memory alone handed a 67 GB corpus a
    // 33 GiB duplicate tier on this 96 GB box (WARM 33062 MiB), and mapping
    // plus duplicates then exceeded installed RAM: the machine paged itself
    // to a standstill during warmup. Reserve the mapping's own footprint
    // first. What is left is what WARM may use, which on a small-RAM machine
    // is a few experts -- the point of the tier, not a shortfall.
    const size_t mapped = model_.mapped_bytes();
    const size_t mapped_share = mapped < ram_total ? mapped : ram_total;
    // The mapping is NOT subtracted from the staging room any more, and that is
    // a correctness fix, not a policy tweak. A clean file-mapping page is
    // reclaimable standby memory: the OS drops it without a writeback and
    // re-faults it from the page cache, so it is not RAM this process owns and
    // charging all of it here double-counted a corpus the tier does not hold.
    // The consequence was not subtle -- for every file bigger than the ceiling
    // the staging room came out ZERO, which is every model the WARM tier exists
    // for. Measured on the 67 GB REAP-320 file, 96 GB box, 75.2 GiB free: WARM
    // 0 MiB, 387 MiB of expert weights re-read per token at 5.0 tok/s, and the
    // "tier is disabled" warning itself was gated on host_budget>0, so it never
    // printed. The guard that actually protects the machine is the per-admission
    // SharedHostRamBudget check (warm_admit -> can_admit), which samples live
    // available RAM each round and fails closed; a plan made once from the
    // corpus size cannot see pressure that arrives later.
    const size_t host_stage_room =
        live_host_avail > host_reserve ? live_host_avail - host_reserve : 0;
    const size_t host_already_budgeted = krk::SharedHostRamBudget::used_bytes();
    const size_t host_shared_room = host_stage_room > host_already_budgeted
                                        ? host_stage_room - host_already_budgeted : 0;
    if (cfg_.host_ram_mb > 0) {
        host_budget = static_cast<size_t>(cfg_.host_ram_mb) * 1024u * 1024u;
    } else if (ram_total > 0) {
        // One rule, in krk::host_budget_bytes: the largest commitment that
        // leaves the OS its reserve. The old arithmetic subtracted a
        // working-set headroom AND a pin allowance AND a 3 GiB floor on top of
        // the reserve -- and the pin is part of the tier, so the same bytes were
        // charged twice. Written out for 16/24/32 GiB it left 8.8/11.6/15.6 GiB,
        // which is the bad side of the measured cliff (PERF-ANALYSIS 12d).
        host_budget = krk::host_budget_bytes(ram_total, live_host_avail);
    }
    if (host_budget > host_shared_room) host_budget = host_shared_room;
    // Everything the run holds has to fit under the ceiling TOGETHER: the mapped
    // corpus is resident (or will be), the reserve belongs to the OS, and WARM is
    // only what is left. Clamping WARM to the ceiling alone doubled the
    // footprint -- 11.4 GiB mapping plus 9 GiB WARM against a 12.8 GiB ceiling on
    // a 16 GiB class, which is the same hang one size down.
    // The ceiling still applies, but only to memory this process owns. Installed
    // minus the host reserve is the hard headroom for a process whose file pages
    // are reclaimable; the corpus is not charged twice.
    const size_t warm_room =
        host_ceiling > host_reserve ? host_ceiling - host_reserve : 0;
    if (host_budget > warm_room) host_budget = warm_room;
    // A corpus bigger than the ceiling cannot have a host copy of its own:
    // the mapping's page cache IS the RAM tier, and a WARM duplicate would
    // push the machine past the ceiling. Stream hot experts from disk and
    // keep only what the device holds. This is the 16 GiB-RAM case too, just
    // at a smaller corpus.
    if (host_ceiling > 0 && warm_room == 0) {
        // Unconditional: this used to run only when host_budget>0, so the case
        // that matters -- the budget was already 0 -- was the one case that
        // said nothing at all.
        KRK_WARN("ram policy: the mapped corpus (%.1f GiB) is at or past "
                 "the %.1f GiB ceiling (%.0f%% of installed), so the WARM "
                 "tier is disabled: hot experts stream from disk and the "
                 "device holds what fits. KRK_MEM_FRACTION raises the "
                 "ceiling if this box is dedicated.",
                 static_cast<f64>(mapped_share) / static_cast<f64>(gib),
                 static_cast<f64>(host_ceiling) / static_cast<f64>(gib),
                 krk::mem_fraction() * 100.0);
        host_budget = 0;
    }
    if (live_host_avail == 0) host_budget = 0; // unknown => no pageable staging

    size_t warm = 0;
    if (cfg_.expert_warm_mb > 0) {
        warm = static_cast<size_t>(cfg_.expert_warm_mb) * 1024u * 1024u;
        if (live_host_avail == 0) warm = 0;
        else if (warm > host_shared_room) warm = host_shared_room;
        // An explicit tier can still exceed the budget the user set for the
        // process; the budget is the outer limit, the tier is a request inside
        // it. Say so rather than silently taking the larger number.
        if (host_budget > 0 && warm > host_budget) {
            KRK_WARN("WARM tier requested %.0f MiB, clamped to the %.0f MiB host "
                     "budget (--host-ram-mb)",
                     static_cast<f64>(warm) / (1024.0 * 1024.0),
                     static_cast<f64>(host_budget) / (1024.0 * 1024.0));
            warm = host_budget;
        }
    } else if (cfg_.expert_warm_mb < 0 && be_->caps().vram_free > 0) {
        // The default: as much of the routed set as the machine can hold while
        // leaving the OS its reserve. See krk::warm_tier_bytes for the rule and
        // PERF-ANALYSIS 12d for why the old cautious half-room is not it -- a
        // tier shorter than the corpus misses on the same experts forever, and
        // the measured difference at the small end is 23.5 against 29.4 tok/s.
        warm = krk::warm_tier_bytes(ram_total, live_host_avail,
                                    model_.total_expert_bytes(),
                                    model_.max_expert_bytes() * 4);
        if (host_shared_room > 0 && warm > host_shared_room)
            warm = host_shared_room;
        // The auto tier is the largest this host can SPARE, so landing under the
        // knee is not a misconfiguration -- it is the small-machine case, and it
        // is the one case where the tier's size is worth a line. Say it once
        // (this runs once, at init) and name both numbers: the tier, and what
        // the shortfall is projected to read per layer, because that is the
        // figure the read-ahead worker has to hide inside a layer's compute. See
        // krk::warm_miss_mib_per_layer for the measurements behind it.
        const size_t corpus = model_.total_expert_bytes();
        const int n_layers = model_.cfg().n_layer;
        const double miss_mib =
            krk::warm_miss_mib_per_layer(corpus, warm, n_layers);
        if (warm > 0 &&
            krk::warm_tier_under_knee(corpus, warm, n_layers)) {
            KRK_WARN("ram policy: the auto WARM tier is %.0f MiB for a %.0f MiB "
                     "routed set (%.0f%%), under the measured knee: from here "
                     "the read-ahead worker cannot hide what it reads -- "
                     "projected %.2f MiB of file reads per layer (%.0f MiB per "
                     "token at %d layers). Measured at that shape, decode is "
                     "23.5-27.2 tok/s where a tier that covers the set is "
                     "29-34 (docs/PERF-ANALYSIS.md 12d). This tier is already "
                     "the largest this host can spare: free RAM, a bigger "
                     "--ram-tier class, or KRK_MEM_FRACTION raises it; "
                     "--expert-warm-mb 0 turns the tier off deliberately.",
                     static_cast<f64>(warm) / (1024.0 * 1024.0),
                     static_cast<f64>(corpus) / (1024.0 * 1024.0),
                     100.0 * static_cast<f64>(warm) /
                         static_cast<f64>(corpus > 0 ? corpus : 1),
                     miss_mib,
                     miss_mib * static_cast<f64>(n_layers),
                     n_layers);
        }
    }
    if (mapped_share > 0 && (host_budget > 0 || cfg_.host_ram_mb > 0))
        KRK_INFO("ram policy: %.1f GiB of the host room is the file mapping "
                 "itself (the disk tier's page cache); WARM gets what is left",
                 static_cast<f64>(mapped_share) / static_cast<f64>(gib));
    if (host_budget > 0 || cfg_.host_ram_mb > 0)
        KRK_INFO("ram policy: %.1f GiB budget (%.1f GiB installed, %.1f GiB "
                 "free) -> WARM tier %.0f MiB%s%s",
                 static_cast<f64>(host_budget) / static_cast<f64>(gib),
                 static_cast<f64>(ram_installed) / static_cast<f64>(gib),
                 static_cast<f64>(be_->host_available_bytes()) / static_cast<f64>(gib),
                 static_cast<f64>(warm) / (1024.0 * 1024.0),
                 cfg_.ram_tier_gb > 0 ? " [planned for the --ram-tier class]" : "",
                 cfg_.expert_warm_mb == 0 ? " (disabled by --expert-warm-mb 0)"
                                          : "");

    model_.set_expert_budget(budget, warm, slot_budget);
    // ---- the pageable path and the pinned WARM prefix ----------------------
    // A promotion out of pageable WARM is either a bounce-buffer copy through a
    // narrow aperture or a direct DMA, and WHICH of those it is depends on the
    // card's aperture -- a firmware setting (Above 4G decoding + Resizable BAR),
    // not a constant. So neither decision here is a default someone has to tune:
    // the run measures both. On this box's RX 9070 XT (15.9 GiB, 256 MiB
    // aperture), 16 greedy tokens, interleaved order-balanced pairs, identical
    // text in every arm:
    //   REAP-320 (42 GiB routed, 12.3 GiB WARM pinned)
    //       114.4/102.0/100.1 -> 96.7/100.8/91.3 ms/step (3/3 pairs)
    //   35B-A3B Q2KXL (9.4 GiB WARM, pin 4.7 GiB, --expert-warmup 1)
    //       36.5/35.6 -> 30.9/31.1 ms/step (+15.3/+12.7%), 414 -> 199 us/expert
    //   Qwen3-30B-A3B i1-Q2_K (tier is disk-bound: 376 MiB read per token)
    //       93.4/93.5 -> 90.5/92.2/92.5 (+2%, within the pairs' spread)
    //   Qwen3-4B-Q8_0 (dense: no expert tier at all)
    //       9.3 -> 9.3 ms/step     (0% -- the control, not a miss)
    // The gain tracks how much of a step a promotion is, which is exactly why
    // the last two arms are reported rather than quietly dropped.
    //
    // The two decisions are ONE decision. The pin exists to remove the copy the
    // staging ring pays; if the aperture lets the direct path remove it for free,
    // the pinned rate stops being ~2x the pageable rate and falls inside the
    // 1.20x line, so the prefix declines to spend RAM that cannot be swapped.
    // That is the wide-aperture answer, and it needs no knob: fast AND swappable.
    //
    // Bounded by installed RAM only: half of it, because pinned pages are not
    // swappable and this runs on laptops too. The tier is
    // pinned WHOLE inside that bound -- the old rule kept half of it pageable
    // "so a small box still has a tier", and that half is the one every miss
    // lands on: the 256 MiB arm paid 469 us per promotion at 2.0 GB/s with 4800
    // of 9600 MiB pinned (139.8 ms/step) against 10 us and 65.3 ms/step once the
    // whole tier was pinned, byte-identical stdout (see
    // krk::pinned_warm_prefix_bytes). KRK_HOST_PIN_MB or --expert-pin-mb
    // overrides it outright, including with 0 -- that is the opt-out -- and
    // KRK_PIN_AUTO=0 turns the default off.
    const bool small_bar = be_->caps().large_bar == 0;
    const bool pin_explicit =
        cfg_.expert_pin_mb >= 0 || std::getenv("KRK_HOST_PIN_MB") != nullptr;
    // The probe runs first and decides both. Its size is this model's expert,
    // clamped to 16-64 MiB -- and the FLOOR matters more than it looks: on the
    // REAP-320 file (2.2 MiB experts) the same comparison reads 1.60x at 16 MiB
    // but 1.41-1.69x across runs at 4 MiB, because there the sync and the launch
    // are a large part of each arm and compress the difference the pin is worth.
    // A promotion here is a burst of such copies, so the bandwidth-dominated
    // size is the honest one. ~30-110 ms at init depending on the bucket, three
    // arms, up to 5 iterations -- it stops as soon as the medians settle, which
    // on this box is 3 at 64 MiB and rarely early at 16 MiB.
    krk::H2dProbe h2d;  // the probe itself lives in krk/h2d_probe.hpp
    // Clamped AND bucketed (16/32/64 MiB), because that is the set of sizes the
    // cache holds verdicts for: `kraken-bench --h2d` measures all three, so an
    // explicit measurement is what a normal run finds, whatever its expert
    // footprint. See h2d_probe_size_bytes for why the floor exists.
    const size_t h2d_bytes = krk::h2d_probe_size_bytes(model_.max_expert_bytes());
    const char *probe_off = std::getenv("KRK_H2D_PROBE");
    const bool probe_on = warm > 0 && !(probe_off && probe_off[0] == '0');
    // The verdict comes from the cache when this machine has already measured
    // the same (card, probe size, aperture hint); KRK_H2D_PROBE_RERUN=1 forces a
    // fresh measurement. The rates, not just the chosen path, are cached: the
    // pin decision is taken against them, and re-deriving one number from
    // another would make a cached verdict mean something the probe never said.
    bool h2d_cached = false;
    long long h2d_age_sec = 0;
    std::string h2d_cache_path;
    const int h2d_driver = be_->caps().driver_version;
    krk::H2dCacheKey h2d_key;
    if (probe_on) {
        const krk::DeviceCaps caps = be_->caps();
        h2d_key.name = caps.name + " " + caps.gfx;
        h2d_key.vram_total = caps.vram_total;
        h2d_key.large_bar = caps.large_bar;
        h2d_key.driver_version = caps.driver_version;
        h2d_key.bytes = h2d_bytes;
        h2d_cache_path = krk::h2d_probe_cache_path();
        const char *rerun = std::getenv("KRK_H2D_PROBE_RERUN");
        const bool force = rerun && rerun[0] == '1';
        const long long now = krk::h2d_now_sec();
        if (!h2d_cache_path.empty() && !force) {
            const krk::H2dCacheLookup hit =
                krk::h2d_probe_cache_read(h2d_cache_path, h2d_key, now);
            if (hit.hit) {
                h2d = hit.probe;
                h2d_cached = true;
                h2d_age_sec = hit.age_sec;
            }
        }
        // The reason for a miss has to be read BEFORE this run writes its own
        // verdict: scanned afterwards the file already holds a live entry for
        // exactly this key, so the honest "there is no cache file yet" became a
        // generic "no verdict for this card" -- the first real run of this line
        // printed that, which is why the order is spelled out here.
        std::string miss;
        long long probe_ms = 0;
        if (!h2d_cached) {
            miss = force
                       ? std::string("KRK_H2D_PROBE_RERUN=1 asked for a fresh measurement")
                       : krk::h2d_cache_miss_text(h2d_cache_path, now, h2d_key);
            // Timed, because a hit's value IS the measurement it replaced: the
            // end-of-run tally prices every avoided probe at this machine's own
            // average rather than at a constant someone guessed.
            const std::chrono::steady_clock::time_point t0 =
                std::chrono::steady_clock::now();
            h2d = krk::probe_h2d_path(be_, h2d_bytes, krk::kH2dProbeMaxIters);
            probe_ms = static_cast<long long>(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count());
            if (h2d.valid && !h2d_cache_path.empty()) {
                krk::H2dCachePrune pruned;
                if (krk::h2d_probe_cache_write(h2d_cache_path, h2d_key, h2d, now,
                                               &pruned)) {
                    // Only when something went: a clean rewrite is the normal
                    // case, and "dropped 0" on every first run is noise.
                    const std::string report = krk::h2d_prune_text(pruned);
                    if (!report.empty())
                        KRK_INFO("H2D cache %s: %s", h2d_cache_path.c_str(),
                                 report.c_str());
                }
            }
        }
        // Which it was, and -- when it measured -- why there was nothing to
        // reuse. Without this a cached verdict and a fresh one are 20 ms apart
        // in the log and identical in text, and "measured now" says the same
        // thing whether the file was empty, held another card's verdict, or held
        // this card's entry for a probe size this model does not bucket to.
        // Those are three different fixes, so the reason is part of the line.
        if (h2d.valid) {
            KRK_INFO("%s", krk::h2d_verdict_text(h2d_cached, h2d_age_sec, h2d_driver,
                                                 h2d_bytes, miss, h2d.iterations)
                                .c_str());
        } else {
            KRK_INFO("H2D verdict: the probe produced no usable rates (an allocation "
                     "failed), so the pageable path follows the runtime's aperture "
                     "hint");
        }
        // The cross-run ledger, counted only for inits that got a verdict: a run
        // with the probe off reused nothing and measured nothing, and counting
        // it would drag the hit rate down without describing any cache.
        if (h2d.valid)
            h2d_tally_ =
                krk::h2d_tally_note(h2d_cache_path, h2d_bytes, h2d_cached, probe_ms);
    }
    h2d_probe_ = h2d;  // published for the suite and for a caller's diagnostics
    // Which pageable path runs. KRK_WARM_DMA=1 (ring) / =2 (direct) pins it, so
    // either can be compared against the measured default in one env var; =0 is
    // still the blocking copy, handled in the backend.
    // THREE mechanisms now, not two: the staging ring, the driver-staged
    // pageable DMA, and the driver's plain blocking copy. The third is the one a
    // wide aperture makes fastest and the old menu did not contain it (measured
    // on this box: ring 8.4-10.0, direct 10.0, blocking 13.4-22.9 GB/s), so a
    // run whose probe chose between ring and direct could only ever pick the two
    // slowest. `h2d_path` is the single definition of the choice -- the log, the
    // cache listing and `kraken-bench --h2d` all name the same mechanism init
    // will run.
    const char *warm_dma = std::getenv("KRK_WARM_DMA");
    const bool dma_forced =
        warm_dma && (warm_dma[0] == '0' || warm_dma[0] == '1' || warm_dma[0] == '2');
    const int pageable_path =
        dma_forced ? (warm_dma[0] == '0' ? 2 : warm_dma[0] == '2' ? 1 : 0)
                   : (h2d.valid ? krk::h2d_path(h2d) : 0);
    if (!dma_forced)
        be_->set_pageable_path(
            static_cast<krk::Backend::PageablePath>(pageable_path < 0 ? 0 : pageable_path));
    // The rate the pin is compared against is the one this run will actually
    // pay, not whichever arm happened to run first.
    const double pageable_gbs = pageable_path == 1   ? h2d.direct_gbs
                                : pageable_path == 2 ? h2d.blocking_gbs
                                                     : h2d.pageable_gbs;
    const bool pin_wanted =
        !pin_explicit && pin_auto_on() && warm > 0 &&
        (h2d.valid ? krk::warm_pin_worthwhile(pageable_gbs, h2d.pinned_gbs)
                   : small_bar);
    if (h2d.valid && warm > 0) {
        // Cached or fresh AND which driver produced it: a reader has to know
        // whether these rates describe this machine, this driver, now -- or a
        // month ago on the driver that has since been replaced.
        const std::string origin =
            krk::h2d_probe_origin(h2d_cached, h2d_age_sec, h2d_driver);
        KRK_INFO("host->device path: pageable promotions use %s -- a %.0f MiB "
                 "copy %s at %.1f GB/s through the staging ring, %.1f GB/s "
                 "direct, %.1f GB/s blocking, %.1f GB/s from pinned memory. The "
                 "aperture is a FIRMWARE setting (Above 4G decoding + Resizable "
                 "BAR) and the runtime's own attribute for it is unreliable, so "
                 "all three mechanisms are measured and the fastest runs, cached "
                 "per card: KRK_H2D_PROBE_RERUN=1 re-measures, KRK_H2D_CACHE=0 "
                 "disables the cache, KRK_WARM_DMA=0|1|2 forces a path, "
                 "KRK_H2D_PROBE=0 skips the probe.",
                 krk::h2d_path_name(pageable_path),
                 static_cast<f64>(h2d_bytes) / (1024.0 * 1024.0), origin.c_str(),
                 h2d.pageable_gbs, h2d.direct_gbs, h2d.blocking_gbs,
                 h2d.pinned_gbs);
        // What the aperture IS, measured. This matters because the runtime's
        // answer is not usable on this platform: `hipDeviceAttributeIsLargeBar`
        // has been seen reporting 0 on firmware with ReBAR on, and this box is
        // one of them -- a 1 GiB pageable copy runs at 22 GB/s and the device
        // reads host-mapped memory in place, neither of which a 256 MiB window
        // can do. Saying so prevents the next reader from "fixing" the copy
        // path back to the ring on the strength of the attribute.
        if (be_->caps().large_bar == 0 && h2d.aperture_wide)
            KRK_INFO("aperture: the runtime reports a 256 MiB window, but the "
                     "measurement says WIDE -- the best pageable path reached "
                     "%.2fx the pinned rate, which only happens when the driver "
                     "writes straight into the aperture (Resizable BAR on). The "
                     "runtime attribute is a hint; these rates are the fact.",
                     h2d.pinned_gbs > 0.0 ? pageable_gbs / h2d.pinned_gbs : 0.0);
        else if (!h2d.aperture_wide)
            KRK_INFO("aperture: measured narrow -- the best pageable path is "
                     "%.2fx the pinned rate, so host memory reaches the card "
                     "through a window and the pinned WARM prefix is what removes "
                     "that copy.",
                     h2d.pinned_gbs > 0.0 ? pageable_gbs / h2d.pinned_gbs : 0.0);
    }
    const size_t pin_auto =
        pin_wanted ? krk::pinned_warm_prefix_bytes(warm, ram_total) : 0;
    // ---- KRK_DEV_PTR=1 wants the whole routed set PINNED, not just the hot
    // prefix it is budgeted for.
    //
    // The one-launch dispatch needs every (layer, expert) to be nameable by a
    // kernel. When the routed set fits the card that is the pool, and this
    // changes nothing. When it does not -- Tiel-Coder-35B-A3B is 22.3 GiB of
    // experts against a 9-11 GiB budget -- the overflow can only be named if it
    // sits in the pinned pool, because pageable RAM is not addressable from a
    // kernel. Pinning the set is what turns `0 of 10240 slots` into a table.
    // The alternative on this machine is the host path: a round trip per layer,
    // measured at 0.846 ms x 40 layers of a 49 ms step.
    //
    // Bounded by the tier: a set larger than WARM cannot be pinned here, and the
    // cache refuses the table with the counts rather than pinning a prefix and
    // silently skipping the rest.
    size_t pin_dispatch = 0;
    if (dev_ptr() && be_->host_load_supported() && total_bytes > budget &&
        warm >= total_bytes) {
        pin_dispatch = total_bytes;
    }
    if (cfg_.expert_pin_mb >= 0) {
        model_.experts().set_host_pin_budget(
            static_cast<size_t>(cfg_.expert_pin_mb) * 1024u * 1024u);
    } else if (pin_dispatch > 0) {
        model_.experts().set_host_pin_budget(pin_dispatch);
        KRK_INFO("expert cache: pinning the whole routed set (%.0f MiB) for the "
                 "one-launch dispatch -- %.0f MiB of it cannot be resident in the "
                 "%.0f MiB device budget, and a kernel can only read the "
                 "overflow from PINNED host memory (measured 24.3 GiB/s on this "
                 "card against 485 GiB/s for VRAM). Pageable WARM would leave "
                 "the table unpublishable.",
                 static_cast<f64>(total_bytes) / 1048576.0,
                 static_cast<f64>(total_bytes - budget) / 1048576.0,
                 static_cast<f64>(budget) / 1048576.0);
    } else if (pin_auto > 0) {
        model_.experts().set_host_pin_budget(pin_auto);
        // The rates are part of the message, not a debug line: this decision is
        // a measurement of THIS machine, and a reader has to be able to see what
        // it was to know whether it still applies.
        const double pin_mib = static_cast<f64>(pin_auto) / (1024.0 * 1024.0);
        const double warm_mib = static_cast<f64>(warm) / (1024.0 * 1024.0);
        if (h2d.valid)
            KRK_INFO("expert cache: pinned WARM prefix %.0f MiB of the %.0f MiB "
                     "tier -- a %.0f MiB promotion measured %.1f GB/s %s "
                     "against %.1f GB/s pinned (%.2fx), so the pageable path "
                     "pays a copy this prefix removes. Every rate here was "
                     "measured on this card; a wide aperture moves them, and "
                     "then this prefix removes itself. KRK_HOST_PIN_MB or "
                     "--expert-pin-mb overrides it, KRK_PIN_AUTO=0 disables it.",
                     pin_mib, warm_mib,
                     static_cast<f64>(h2d_bytes) / (1024.0 * 1024.0),
                     pageable_gbs, krk::h2d_path_name(pageable_path),
                     h2d.pinned_gbs, h2d.pinned_gbs / pageable_gbs);
        else
            // No probe (KRK_H2D_PROBE=0, or an allocation it could not get): the
            // decision fell back to the runtime's BAR attribute. Say so rather
            // than printing 0.0 GB/s as if it had been measured.
            KRK_INFO("expert cache: pinned WARM prefix %.0f MiB of the %.0f MiB "
                     "tier -- the H2D probe did not run, so this follows the "
                     "runtime's BAR attribute (%s). The aperture is a FIRMWARE "
                     "setting (Above 4G decoding + Resizable BAR); the probe "
                     "measures which side of it this machine is on when it is "
                     "allowed to run. KRK_HOST_PIN_MB or --expert-pin-mb "
                     "overrides it, KRK_PIN_AUTO=0 disables it.",
                     pin_mib, warm_mib,
                     small_bar ? "a 256 MiB aperture" : "large BAR or unknown");
    } else if (!pin_explicit && warm > 0 && pin_auto_on()) {
        if (h2d.valid)
            KRK_INFO("expert cache: no pinned WARM prefix -- a %.0f MiB "
                     "promotion measured %.1f GB/s %s against %.1f GB/s pinned "
                     "(%.2fx), which does not buy back RAM that cannot be "
                     "swapped%s.",
                     static_cast<f64>(h2d_bytes) / (1024.0 * 1024.0),
                     pageable_gbs, krk::h2d_path_name(pageable_path),
                     h2d.pinned_gbs, h2d.pinned_gbs / pageable_gbs,
                     pageable_path == 1
                         ? " (the direct path already carries the copy a pin "
                           "would have removed)"
                         : "");
        else
            KRK_INFO("host->device path: no H2D probe (%s) and the runtime says "
                     "%s, so the WARM prefix stays pageable. Set KRK_HOST_PIN_MB "
                     "or --expert-pin-mb to pin it anyway.",
                     "KRK_H2D_PROBE=0", small_bar ? "a 256 MiB aperture"
                                                  : "large BAR or nothing");
    }
    // Prefetch is opt-in. Read-through admits every expert the run actually
    // touches, so an eager sweep of the corpus is only worth its wall time when
    // the run is short and its routing is broad — and it costs RAM and time
    // before the first token (measured on Laguna XS.2: 4 GiB staged in 1903 ms
    // against ~1.9 s of decode it then improved). --expert-warm-prefetch, or
    // KRK_EXPERT_PRELOAD=1, turns it on; KRK_EXPERT_PRELOAD=0 forces it off, so
    // the A/B is one environment variable away.
    //
    // Mutually exclusive with the warmup: that phase fills WARM in ranked order
    // and this sweep fills it evenly, so running both would spend the first
    // half of the tier on the wrong order and then have no room for the right
    // one. The warmup is the ranked, ranked-by-traffic version of this sweep,
    // so it wins when it is on.
    const char *pre_env = std::getenv("KRK_EXPERT_PRELOAD");
    const bool prefetch = pre_env && pre_env[0] == '0'
                              ? false
                              : (cfg_.expert_warm_prefetch ||
                                 (pre_env && pre_env[0] == '1'));
    if (warm > 0 && be_->caps().vram_free > 0 && prefetch &&
        !expert_warmup_wanted()) {
        const std::chrono::steady_clock::time_point t0 =
            std::chrono::steady_clock::now();
        const size_t ram_before = be_->host_available_bytes();
        // Expert-major, layer-minor: with a bounded tier this fills every
        // layer's expert 0 before any layer's expert 1, so a tier smaller
        // than the corpus is shared out evenly instead of being spent on the
        // first few layers. Routing mass is the better order when the model
        // carries a .krakenexperts.json; this is the order that needs no
        // prior run and does not concentrate the hot set in one place.
        size_t staged = 0;
        // Against INSTALLED RAM, not the planned tier: this sweep stages into
        // whatever the machine really has, so the reserve it must leave is the
        // installed one. (The run's own ceiling, sized from --ram-tier, is a
        // different number and is applied where the tiers are planned.)
        const size_t reserve_installed =
            krk::host_reserve_bytes(be_->host_total_bytes());
        bool hit_host_reserve = false;
        for (i32 e = 0; e < mc.n_expert && !hit_host_reserve; e++) {
            for (i32 l = 0; l < mc.n_layer; l++) {
                const LayerWeights &L = model_.layers()[static_cast<size_t>(l)];
                if (!L.experts.present()) continue;
                const size_t added =
                    model_.experts().preload_one(L.experts, l, e);
                staged += added;
                if (added == 0) {
                    if (!krk::host_staging_fits(0, be_->host_available_bytes(),
                                                reserve_installed))
                        hit_host_reserve = true;
                    if (hit_host_reserve ||
                        model_.experts().warm_used_bytes() >=
                            model_.experts().warm_capacity_bytes())
                        break;
                }
            }
        }
        if (hit_host_reserve)
            KRK_INFO("expert WARM prefetch stopped at the host RAM reserve: %.0f MiB "
                     "kept free",
                     static_cast<f64>(reserve_installed) / 1048576.0);
        const f64 stage_ms = std::chrono::duration<f64, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
        // Both sides of the cost: the pageable tier is what it holds, and
        // the drop in free RAM is what it actually took, mapping pages
        // included. The two differ by the size of the staged ranges.
        const size_t ram_after = be_->host_available_bytes();
        const f64 ram_mib = ram_before > ram_after
                                ? static_cast<f64>(ram_before - ram_after) / 1048576.0
                                : 0.0;
        KRK_INFO("expert WARM: prefetched %.1f MiB in %.0f ms, host RAM in use "
                 "up %.1f MiB (a VRAM miss is a promotion now, not a file read)%s",
                 static_cast<f64>(staged) / 1048576.0, stage_ms, ram_mib,
                 hit_host_reserve ? " (stopped at the host RAM reserve)" : "");
    }
    char warm_note[96] = "";
    if (model_.experts().warm_as_mapping())
        std::snprintf(warm_note, sizeof(warm_note),
                      " + no WARM (KRK_EXPERT_WARM_MAP=1: the mapping is the "
                      "second tier)");
    else if (warm > 0)
        std::snprintf(warm_note, sizeof(warm_note),
                      " + WARM %.0f MiB pageable%s",
                      static_cast<f64>(warm) / (1024.0 * 1024.0),
                      prefetch && !expert_warmup_wanted() ? " (prefetched)" : "");
    KRK_INFO(             "expert cache: budget %.1f MiB%s, %d experts x top-%d, %d layers, "
             "policy LFU+aging (pin at %u hits, decay every %u); cold reads are "
             "read-through",
             static_cast<f64>(budget) / (1024.0 * 1024.0), warm_note, mc.n_expert,
             mc.n_expert_used, mc.n_layer, ExpertCache::pin_threshold(),
             ExpertCache::kPinDecay);
    if (budget < total_bytes && total_bytes > 0) {
        if (be_->caps().vram_free == 0) {
            KRK_WARN("expert cache: the routed set is %.1f GiB and this "
                     "machine cannot tell the device capacity, so the device "
                     "budget stays near its host-RAM fallback. %s",
                     static_cast<f64>(total_bytes) / (1024.0 * 1024.0 * 1024.0),
                     warm >= total_bytes
                         ? "the WARM tier holds the full set so every cold read is a promotion"
                         : "set the --expert-cache-mb / --vram-cap-mb / --expert-warm-mb flags, or the K/V cache size, to cover the routed set on device");
        } else {
            const f64 frac = static_cast<f64>(budget) /
                             static_cast<f64>(total_bytes);
            KRK_WARN("expert cache: the routed set is %.1f GiB and the device "
                     "budget is only %.1f GiB (%.0f%%) — the cache is spending its "
                     "budget churning rather than holding. %s",
                     static_cast<f64>(total_bytes) / (1024.0 * 1024.0 * 1024.0),
                     static_cast<f64>(budget) / (1024.0 * 1024.0 * 1024.0),
                     frac * 100.0,
                     model_.experts().warm_as_mapping()
                         ? "under KRK_EXPERT_WARM_MAP=1 a device miss reads the file's page cache instead of promoting a host copy, so the churn is a read, not a promotion"
                         : warm >= total_bytes
                               ? "the WARM tier still holds the full set, so a device miss promotes from RAM instead of reading the file"
                               : "raise --expert-cache-mb, lower --ctx / --kv-hot-mb, or use a model whose routed set fits the card");
        }
    }
    // Every path above either refused (return false) or configured the cache.
    // This function grew a return value for the refusal; the fall-through case
    // is the common one and needs its `true` spelled out, or the caller reads
    // whatever was in the return register.
    return true;
}

bool Engine::load_dflash(const std::string &path, std::string *err) {
    if (!be_) {
        if (err) *err = "the engine must be initialised before the drafter";
        return false;
    }
    if (dflash_) return true;
    const ModelConfig &tc = model_.cfg();
    DflashDraft *d = new DflashDraft();
    if (!d->load(*be_, path, chunk_, kv_cap_, tc.n_embd, tc.n_layer, n_vocab_,
                 err)) {
        delete d;
        return false;
    }
    // The mask token has to be a real id in the *target's* vocabulary: the block
    // embeds it with the target's token embedding table, so an id past the end
    // would read a neighbour row and draft against noise.
    if (d->mask_id() < 0 || d->mask_id() >= n_vocab_) {
        if (err)
            *err = format("DFlash mask token %d is outside the target's vocabulary "
                          "(%d) — this head set is for a different target",
                          d->mask_id(), n_vocab_);
        delete d;
        return false;
    }
    // ... and it has to be the target's OWN mask id: the head set was trained
    // to treat one mask row as noise, and embedding a different row drafts
    // against a token the heads never saw masked. Targets without the key
    // predate the convention and keep the range check above as their gate.
    // KRK_DFLASH_ALLOW_BAD_MASK=1 loads the mismatched pair anyway, for
    // diagnosing a head set whose key is wrong rather than its weights.
    {
        const i32 tmask = static_cast<i32>(
            model_.gguf().get_i64("tokenizer.ggml.mask_token_id", -1));
        const char *allow = std::getenv("KRK_DFLASH_ALLOW_BAD_MASK");
        const bool override =
            allow != nullptr && allow[0] != '\0' && allow[0] != '0';
        if (tmask >= 0 && d->mask_id() != tmask && !override) {
            if (err)
                *err = format(
                    "DFlash mask token %d is not the target's mask token (%d): "
                    "the block's mask rows are embedded with the target's "
                    "table, so this head set would draft against a row it was "
                    "never trained on",
                    d->mask_id(), tmask);
            delete d;
            return false;
        }
        if (tmask >= 0 && d->mask_id() != tmask)
            KRK_WARN("CRACKED PAIR by override: DFlash mask token %d against "
                      "the target's mask token (%d) — numbers from this run "
                      "are not to be believed",
                      d->mask_id(), tmask);
    }
    // The head is the target's, so its READOUT can be too: a head set whose own
    // final norm is the frozen identity ships no gain, and the borrowed head was
    // trained on the target's. Installed only when the target has one (qwen4exp
    // carries its norm in the final mixer and returns null here).
    d->use_target_out_norm(model_.out_norm());
    dflash_ = d;
    if (draft_tokens_ <= 0) draft_tokens_ = 4;
    KRK_INFO("dflash drafter loaded: %s — %d captured layers, block size %d, "
             "speculative window %d",
             d->name().c_str(), d->n_aux(), d->block_size(),
             std::min<i32>(draft_tokens_, d->block_size() - 1));
    KRK_INFO("dflash block placement base %d row(s) (KRK_DFLASH_BLOCK_POS)",
             d->block_base());
    return true;
}

void Engine::unload_dflash() {
    if (!dflash_) return;
    dflash_->unload();
    delete dflash_;
    dflash_ = nullptr;
}

i32 Engine::dflash_block_size() const {
    return dflash_ ? dflash_->block_size() : 0;
}

// ---- external draft head (EAGLE-3) ---------------------------------------
//
// A head is one block that reads the TARGET's own hidden state and the last
// token's embedding. That makes the round the same three moves every
// speculative path here makes -- propose, verify in ONE batched forward,
// realign to the accepted prefix -- with the proposal side on the host.
//
// Two things are correctness rather than taste:
//
//   * the head's input at position p is (hidden_p, embed(token_p)) and its
//     output predicts p+1, which is the same contract the nextn block has
//     (mtp_block_forward). A proposal chain therefore feeds the head's OWN
//     output hidden for position p and the proposed token at p -- not the
//     target's row, which does not exist yet.
//   * after a partial accept the head's KV must be rewound to the round's
//     start and re-fed from the target's CAPTURED rows. Its speculative rows
//     were built from the head's hidden, not the target's, and keeping them
//     would draft from a different distribution -- fluent, plausible, wrong.

bool Engine::load_draft_head(const std::string &path, std::string *err) {
    if (!be_) {
        if (err) *err = "the engine must be initialised before the draft head";
        return false;
    }
    if (head_) return true; // already loaded
    if (!model_.speculative_rewind_ok()) {
        // Same reason load_draft refuses one: a delta-rule state has no
        // inverse, so a rejected suffix cannot be rewound. A shortconv ring
        // is depth-preserved, so it passes.
        if (err)
            *err = "speculative decoding needs a partial KV rewind, which a "
                   "recurrent (gated delta net) model cannot do";
        return false;
    }
    DraftHead *h = new DraftHead();
    std::string why;
    if (!h->load(path, &why)) {
        delete h;
        if (err) *err = why;
        return false;
    }
    if (!h->check_target(static_cast<i32>(n_embd_), static_cast<i32>(n_vocab_), &why)) {
        delete h;
        if (err) *err = why;
        return false;
    }
    // Read the geometry BEFORE the head is destroyed: both messages below name
    // a field of the object, and deleting first made them read freed memory
    // (harmless in practice only because nothing had reused the block yet).
    const i32 head_n_in = h->cfg().n_in;
    const i32 head_n_layer = h->cfg().n_layer;
    if (head_n_in != 1) {
        delete h;
        if (err)
            *err = "draft head: it fuses " + std::to_string(head_n_in) +
                   " target hidden slot(s); this engine captures the "
                   "pre-final-norm row, i.e. one";
        return false;
    }
    if (head_n_layer != 1) {
        delete h;
        if (err)
            *err = "draft head: " + std::to_string(head_n_layer) +
                   " layer(s); this engine runs the single-block EAGLE-3 head";
        return false;
    }
    const size_t as = be_->act_size();
    head_hidden_ = be_->alloc(static_cast<size_t>(kv_cap_ * n_embd_) * as);
    head_e_ = be_->alloc(static_cast<size_t>(n_embd_) * as);
    if (!head_hidden_ || !head_e_) {
        if (head_hidden_) { be_->release(head_hidden_); head_hidden_ = nullptr; }
        if (head_e_) { be_->release(head_e_); head_e_ = nullptr; }
        delete h;
        if (err)
            *err = "draft head: the " +
                   std::to_string(static_cast<f64>(kv_cap_ * n_embd_) * as /
                                  1048576.0) +
                   " MiB hidden capture could not be allocated";
        return false;
    }
    head_ = h;
    head_tok_.assign(static_cast<size_t>(kv_cap_), -1);
    head_row_.assign(static_cast<size_t>(n_embd_), 0.0f);
    head_emb_.assign(static_cast<size_t>(n_embd_), 0.0f);
    head_logits_.assign(static_cast<size_t>(h->cfg().draft_vocab), 0.0f);
    head_captured_ = 0;
    head_fed_ = 0;
    head_->reset(static_cast<i32>(std::min<i64>(kv_cap_, 0x7fffffff)));
    if (draft_tokens_ <= 0) draft_tokens_ = 4;
    KRK_INFO("draft head: %s -- %dx%d block, %d head(s) x %d, %d kv head(s), "
             "ff %d, draft vocab %d%s -> target vocab %lld; %.1f MiB host "
             "weights, %.1f MiB hidden capture, prime window %d",
             path.c_str(), h->cfg().hidden, h->cfg().n_ff, h->cfg().n_head,
             h->cfg().head_dim, h->cfg().n_kv, h->cfg().n_ff,
             h->cfg().draft_vocab, h->cfg().has_d2t ? " (d2t mapped)" : "",
             static_cast<long long>(n_vocab_),
             static_cast<f64>(h->weight_bytes()) / 1048576.0,
             static_cast<f64>(kv_cap_ * n_embd_) * as / 1048576.0,
             head_prime_max_);
    return true;
}

void Engine::unload_draft_head() {
    if (head_hidden_) { be_->release(head_hidden_); head_hidden_ = nullptr; }
    if (head_e_) { be_->release(head_e_); head_e_ = nullptr; }
    if (head_) { delete head_; head_ = nullptr; }
    head_tok_.clear();
    head_captured_ = 0;
    head_fed_ = 0;
}

// The capture rides every forward pass, like the MTP probe's: the head needs
// the target's pre-final-norm row for EVERY position it may propose from.
void Engine::head_capture(const i32 *toks, i32 pos0, i32 n) {
    if (!head_ || !head_hidden_ || n <= 0 || toks == nullptr) return;
    const i64 end = static_cast<i64>(pos0) + n;
    if (pos0 < 0 || end > kv_cap_) return;
    const size_t as = be_->act_size();
    be_->copy_act(act_at(head_hidden_, as, static_cast<i64>(pos0) * n_embd_),
                  ws_x_, static_cast<i64>(n) * n_embd_);
    for (i32 i = 0; i < n; i++)
        head_tok_[static_cast<size_t>(pos0 + i)] = toks[i];
    head_captured_ = end;
}

// One token's embedding row, on the host. `had_xform` inverts the transform the
// model applies to its own embedding table, so the head sees the same row the
// target's first layer does rather than the stored one.
bool Engine::head_embed(i32 tok) {
    const ModelConfig &mc = model_.cfg();
    const QuantTensor &embd = model_.tok_embd();
    if (!head_e_ || tok < 0 || tok >= n_vocab_) return false;
    be_->embed(head_e_, embd.data, embd.type, mc.n_vocab, mc.n_embd, &tok, 1);
    if (model_.had().embd_inverse)
        had_xform(head_e_, head_e_, 1, n_embd_, /*inverse=*/true,
                  /*gdn_perm=*/false);
    be_->download_f32(head_emb_.data(), head_e_, n_embd_);
    return true;
}

// Walk the head over the captured target rows in [head_fed_, pos).
//
// The window is why this is not quadratic: a head's fusion projection costs
// O(hidden x target_hidden) per position, so priming a 4k prompt is hundreds of
// GFLOP of host arithmetic. With prime_max > 0 only the last `prime_max`
// positions are attended (absolute RoPE positions, a bounded window), which is
// the trade the flag names. `0` is the exact form.
bool Engine::head_prime(i64 pos, std::string *why) {
    if (!head_) { if (why) *why = "no draft head"; return false; }
    if (pos <= head_fed_) return true;
    const i64 window = head_prime_max_ > 0 ? head_prime_max_ : 0;
    i64 start = head_fed_;
    if (window > 0 && pos - start > window) start = pos - window;
    if (start < head_fed_) {
        // Jumping the window forward: the rows before it are not attended any
        // more, and recomputing them would be the cost the window removes.
        head_->truncate(static_cast<i32>(pos));
        head_fed_ = start;
    }
    head_->drop_before(static_cast<i32>(start));
    const size_t as = be_->act_size();
    const Timer t;
    for (i64 q = start; q < pos; q++) {
        if (q >= head_captured_) {
            head_prime_ms_ += t.ms();
            if (why) *why = "the target has no captured hidden row for position " +
                            std::to_string(static_cast<long long>(q));
            return false;
        }
        const i32 tok = head_tok_[static_cast<size_t>(q)];
        if (tok < 0) {
            head_prime_ms_ += t.ms();
            if (why) *why = "a captured position has no token id";
            return false;
        }
        if (!head_embed(tok)) {
            head_prime_ms_ += t.ms();
            if (why) *why = "the head's embedding row could not be read";
            return false;
        }
        be_->download_f32(head_row_.data(),
                          act_at(head_hidden_, as, q * n_embd_), n_embd_);
        if (!head_->step(head_row_.data(), head_emb_.data(),
                         static_cast<i32>(q), nullptr, /*want_logits=*/false)) {
            head_prime_ms_ += t.ms();
            if (why) *why = "the head refused a prime step (position past its KV)";
            return false;
        }
    }
    head_fed_ = pos;
    head_prime_ms_ += t.ms();
    return true;
}

// A chat template that emits `bos_token` (Llama-3, Gemma and Mistral all do)
// has already opened the sequence. Adding the tokenizer's special token on top
// puts a second BOS in the KV cache -- it shifts every logit, and you cannot
// see it by eyeballing the rendered text, only in the ids. So: if the prompt
// already starts with the BOS piece, do not add it again.
int Engine::tokenize_prompt(const std::string &prompt, std::vector<i32> &ids) const {
    const bool prompt_has_bos = tok_.starts_with_bos(prompt);
    if (prompt_has_bos) {
        // llama.cpp warns here and then keeps both BOS tokens. We take the
        // other branch: a template that opens with BOS was written by the model
        // author against the transcript they trained on, and a doubled marker
        // is never part of it. Say so in debug, so a logit comparison against
        // llama.cpp has an explanation rather than a mystery.
        KRK_DEBUG("prompt starts with '%s' (BOS) -- not adding a second one",
                  tok_.piece(tok_.bos()).c_str());
    }
    return tok_.encode(prompt, ids, !prompt_has_bos);
}

bool Engine::generate_speculative_head(const GenerateParams &p, GenerateResult *res,
                                       const std::vector<i32> &ids, f64 *prefill_ms,
                                       f64 *decode_ms, i64 *decode_steps) {
    if (!head_) {
        res->error = "no draft head is loaded";
        res->finish = FinishError;
        return false;
    }
    const i32 k = std::max<i32>(1, draft_tokens_);
    // Two modes, one round. Greedy is a prefix match and stays exactly as it
    // was. Sampling cannot be: a proposal the target merely found unlikely has
    // to survive sometimes, or the output is the DRAFT's distribution. The
    // accept/reject rule (spec_sampling.hpp) is what keeps it the target's,
    // and it needs both distributions -- the draft's q for the proposal and the
    // target's p at that position -- as rows rather than as one argmax.
    // The same name the draft-model and dflash verifiers use, with the opposite
    // default: those walk back onto existing greedy arms when it is unset, while
    // a head has no other way to run under sampling at all, so turning it off
    // (KRK_SPEC_RS=0) is what returns the plain loop.
    static const bool rs_disabled = [] {
        const char *e = std::getenv("KRK_SPEC_RS");
        return e && e[0] == '0' && e[1] == 0;
    }();
    const bool sampled = !p.sampler.greedy && p.sampler.temp > 0.0f && !rs_disabled;

    // ---- prompt prefill (the capture rides every forward) -----------------
    expert_phase(ExpertCache::PhasePrefill);
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        if (!forward(ids.data() + i, n, static_cast<i32>(i), last)) {
            res->error = forward_failure_error();
            res->finish = FinishError;
            return false;
        }
    }
    fetch_logits(false);
    *prefill_ms = prefill_timer.ms();
    expert_phase(ExpertCache::PhaseDecode);

    i64 pos = static_cast<i64>(ids.size());
    head_fed_ = 0;
    head_->reset(static_cast<i32>(std::min<i64>(kv_cap_, 0x7fffffff)));
    std::string why;
    if (!head_prime(pos, &why)) {
        res->error = "the draft head could not be primed over the prompt: " + why;
        res->finish = FinishError;
        return false;
    }

    Sampler sampler;
    sampler.reset(p.sampler);
    sampler.accept(ids.data(), static_cast<int>(ids.size()));
    TokenEmitter emit(p, tok_, sampler, res, ids);

    const Timer decode_timer;
    i64 timed_decode_steps = 0;
    auto finish_run = [&](Finish f) {
        res->finish = f;
        *decode_ms = decode_timer.ms();
        *decode_steps = timed_decode_steps;
        emit.finish(f == FinishStop);
        if (spec_steps_ > 0) {
            const f64 rate =
                draft_proposed_ > 0
                    ? 100.0 * static_cast<f64>(draft_accepted_) /
                          static_cast<f64>(draft_proposed_)
                    : 0.0;
            KRK_INFO("draft head: %llu round(s), %llu of %llu proposals kept "
                     "(%.1f%%), %.2f token(s)/round | prime %.0f ms | propose "
                     "%.0f ms",
                     static_cast<unsigned long long>(spec_steps_),
                     static_cast<unsigned long long>(draft_accepted_),
                     static_cast<unsigned long long>(draft_proposed_), rate,
                     spec_steps_ > 0 ? static_cast<f64>(res->generated) /
                                           static_cast<f64>(spec_steps_)
                                     : 0.0,
                     head_prime_ms_, head_propose_ms_);
        }
    };

    std::vector<i32> prop(static_cast<size_t>(k), -1);
    std::vector<f32> rows(static_cast<size_t>(k) * static_cast<size_t>(n_vocab_));
    // Sampling state: q per proposal, p per verified position, and the tokens
    // this round has already accepted (the repetition penalty has to see them:
    // the plain loop would have emitted them by the time it builds p).
    std::vector<std::vector<f32>> qrows(static_cast<size_t>(k));
    std::vector<f32> draft_prob, p_row;
    std::vector<i32> round_extra;

    // Invariant at the top of every round: the target's KV holds [0, pos),
    // logits_host_ predicts pos, and the head's KV holds the same positions,
    // built from those positions' TARGET hidden rows.
    while (res->generated < p.max_tokens && pos < kv_cap_) {
        KRK_TIME_HOST("spec.round");
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, res->generated, pos, logits_host_, n_vocab_);
        const i32 first = argmax_of(logits_host_, n_vocab_);
        const i64 round_start = pos;

        // ---- propose ------------------------------------------------------
        // Proposal d answers position pos+d from input position pos+d-1.
        // d == 0 uses the TARGET's captured row and the token sitting there;
        // every later proposal feeds the head's own previous hidden state and
        // the token it just proposed, which is what lets one head chain k
        // tokens from one target forward.
        const Timer propose_timer;
        i32 d = 0;
        while (d < k) {
            const i64 ipos = pos + d - 1;
            if (ipos < 0) break;
            const f32 *feat = nullptr;
            i32 tok = -1;
            if (d == 0) {
                if (ipos >= head_captured_) break;
                be_->download_f32(head_row_.data(),
                                  act_at(head_hidden_, be_->act_size(),
                                         ipos * n_embd_),
                                  n_embd_);
                feat = head_row_.data();
                tok = head_tok_[static_cast<size_t>(ipos)];
            } else {
                feat = head_->last_hidden();
                tok = prop[static_cast<size_t>(d - 1)];
            }
            if (tok < 0) break;
            if (pos + d >= kv_cap_) break;
            if (!head_embed(tok)) break;
            if (!head_->step(feat, head_emb_.data(), static_cast<i32>(ipos),
                             &head_logits_))
                break;
            i32 t = -1;
            if (!sampled) {
                t = head_->to_target(DraftHead::argmax(head_logits_));
            } else {
                // A DRAW from q, not its argmax: the rule corrects q against p,
                // and it can only do that for the distribution the token came
                // from. The head's rows are over the draft vocabulary, so q is
                // softmax there and then mapped through d2t into target space.
                spec_softmax(head_logits_.data(),
                             static_cast<i32>(head_logits_.size()), p.sampler.temp,
                             &draft_prob);
                spec_map_dist(draft_prob.data(), static_cast<i32>(draft_prob.size()),
                              head_->d2t_data(), n_vocab_, &qrows[d]);
                t = spec_draw(qrows[static_cast<size_t>(d)], sampler.uniform());
            }
            if (t < 0 || t == tok_.eos()) break; // the target confirms EOS itself
            prop[static_cast<size_t>(d)] = t;
            d++;
            draft_proposed_++;
        }
        head_propose_ms_ += propose_timer.ms();
        spec_steps_++;
        timed_decode_steps++;

        // ---- pre-check: does the target agree with the first proposal? ----
        // Greedy: the argmax either is the proposal or is not. Sampling: the
        // proposal survives with probability min(1, p/q), and when it does not,
        // the round's first token is a draw from the residual -- the correction
        // that keeps the round's output distribution at p.
        bool pre_ok = false;
        i32 next1 = first;
        if (d > 0 && !sampled) pre_ok = (first == prop[0]);
        if (d > 0 && sampled) {
            sampler.build_dist(logits_host_, n_vocab_, &p_row);
            const SpecStep st = spec_accept_step(p_row, qrows[0], prop[0],
                                                 sampler.uniform(), sampler.uniform());
            pre_ok = st.accepted;
            next1 = st.token;
        }
        if (d == 0 || !pre_ok) {
            if (next1 < 0) next1 = first;
            if (next1 == tok_.eos()) {
                finish_run(FinishEos);
                return true;
            }
            emit.accept(next1);
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            if (!forward(&next1, 1, static_cast<i32>(pos), true)) {
                res->error = forward_failure_error();
                finish_run(FinishError);
                return false;
            }
            pos++;
            fetch_logits();
            head_->truncate(static_cast<i32>(round_start));
            head_fed_ = round_start;
            if (!head_prime(pos, &why)) {
                res->error = "the draft head could not be realigned: " + why;
                finish_run(FinishError);
                return false;
            }
            continue;
        }

        // ---- verify: one batched forward of the d proposals --------------
        // Block row j (input prop[j] at position pos+j) predicts pos+j+1.
        i32 d_eff = d;
        while (d_eff > 0 && pos + d_eff > kv_cap_) d_eff--;
        if (d_eff <= 0) {
            finish_run(FinishContext);
            return true;
        }
        if (!forward_core(prop.data(), d_eff, static_cast<i32>(pos), LogitsNone)) {
            res->error = forward_failure_error();
            finish_run(FinishError);
            return false;
        }
        kv_pos_ = pos + d_eff;
        std::vector<i32> row_ids(static_cast<size_t>(d_eff), 0);
        for (i32 j = 0; j < d_eff; j++) {
            head_compute(j);
            i32 id = 0;
            // A sampled round compares distributions, not argmaxes, so it needs
            // every row on the host.
            if (sampled || !topk_argmax(&id)) {
                be_->sync();
                be_->download_f32(rows.data() + static_cast<size_t>(j) * n_vocab_,
                                  ws_logits_, n_vocab_);
                id = argmax_of(rows.data() + static_cast<size_t>(j) * n_vocab_,
                               n_vocab_);
            }
            row_ids[static_cast<size_t>(j)] = id;
        }
        auto row_pred = [&](i32 j) { return row_ids[static_cast<size_t>(j - 1)]; };

        i32 a = 1; // prop[0] passed the pre-check
        i32 repl = -1; // sampling only: the residual draw that ends the round
        bool rejected = false;
        if (!sampled) {
            while (a < d_eff && row_pred(a) == prop[static_cast<size_t>(a)]) a++;
        } else {
            round_extra.clear();
            round_extra.push_back(prop[0]);
            for (i32 j = 0; j + 1 < d_eff; j++) {
                // Row j predicts pos+j+1, i.e. verifies prop[j+1]. p is built
                // with the round's accepted prefix in the penalty history --
                // exactly the history the plain loop would have at this point.
                sampler.build_dist(rows.data() + static_cast<size_t>(j) * n_vocab_,
                                   n_vocab_, &p_row, round_extra.data(),
                                   static_cast<int>(round_extra.size()));
                const SpecStep st =
                    spec_accept_step(p_row, qrows[static_cast<size_t>(j + 1)],
                                     prop[static_cast<size_t>(j + 1)],
                                     sampler.uniform(), sampler.uniform());
                if (st.accepted) {
                    a++;
                    round_extra.push_back(prop[static_cast<size_t>(j + 1)]);
                } else {
                    rejected = true;
                    repl = st.token;
                    break;
                }
            }
        }
        const bool full = (a == d_eff);

        bool stop_hit = false;
        i32 emitted = 0;
        for (i32 j = 0; j < a; j++) {
            if (res->generated >= p.max_tokens) break;
            const i32 t = prop[static_cast<size_t>(j)];
            draft_accepted_++;
            stop_hit = emit.accept(t);
            emitted++;
            if (stop_hit) break;
        }
        if (stop_hit) {
            kv_rollback(pos + emitted);
            finish_run(FinishStop);
            return true;
        }
        // The token that ends the round: the bonus on a full accept, or the
        // residual replacement on a rejection. Both sit at pos+emitted, and the
        // KV row there currently holds a proposal the target did not keep, so
        // both are re-forwarded below.
        i32 tail = -1;
        if (!rejected && full && res->generated < p.max_tokens) {
            i32 bonus = row_pred(d_eff);
            if (sampled) {
                sampler.build_dist(rows.data() +
                                       static_cast<size_t>(d_eff - 1) * n_vocab_,
                                   n_vocab_, &p_row, round_extra.data(),
                                   static_cast<int>(round_extra.size()));
                bonus = spec_draw(p_row, sampler.uniform());
            }
            tail = bonus;
        } else if (rejected && res->generated < p.max_tokens && repl >= 0) {
            tail = repl;
        }
        if (tail >= 0) {
            if (tail == tok_.eos()) {
                kv_rollback(pos + emitted);
                finish_run(FinishEos);
                return true;
            }
            stop_hit = emit.accept(tail);
            emitted++;
            if (stop_hit) {
                kv_rollback(pos + emitted);
                finish_run(FinishStop);
                return true;
            }
        }
        const bool room = pos + emitted < kv_cap_ && res->generated < p.max_tokens;
        if (tail >= 0 && room) {
            if (!forward(&tail, 1, static_cast<i32>(pos + emitted - 1), true)) {
                res->error = forward_failure_error();
                finish_run(FinishError);
                return false;
            }
            fetch_logits();
        } else if (!full && !rejected) {
            // Greedy partial accept: the verifier already computed the row at
            // the position that rejected, and it is exactly the distribution the
            // next round starts from. Downloading it is cheaper than a forward,
            // but it has to be DOWNLOADED -- leaving logits_host_ holding the
            // pre-round row makes the next round propose from a position the
            // sequence has left.
            head_compute(a);
            be_->sync();
            be_->download_f32(logits_host_, ws_logits_, n_vocab_);
        }
        if (res->generated >= p.max_tokens || !full) kv_rollback(pos + emitted);
        pos += emitted;

        if (res->generated >= p.max_tokens) {
            finish_run(FinishLength);
            return true;
        }
        if (pos + 1 >= kv_cap_) {
            finish_run(FinishContext);
            return true;
        }

        // ---- realign the head with what the target kept -------------------
        head_->truncate(static_cast<i32>(round_start));
        head_fed_ = round_start;
        if (!head_prime(pos, &why)) {
            res->error = "the draft head could not be realigned: " + why;
            finish_run(FinishError);
            return false;
        }
    }

    finish_run(res->generated >= p.max_tokens ? FinishLength : FinishContext);
    return true;
}

// One draft_block call, wrapped for callers outside the generate loop (the
// suite, probe tools): the same call generate_speculative_dflash makes, with
// the same clamping (draft_block itself clamps n_draft to block_size-1, so
// the host copy is sized by the same rule the write uses -- sizing it by the
// caller's n_draft instead is the same overflow the loader's vocab check
// exists to prevent, just on the row count instead of the row width).
i32 Engine::draft_block_rows(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                             i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                             DflashRows &out) {
    if (!dflash_ || n_draft <= 0 || n_vocab <= 0) {
        out.data.clear();
        return 0;
    }
    const i32 want = std::min<i32>(n_draft, dflash_->block_size() - 1);
    out.data.assign(static_cast<size_t>(want) * static_cast<size_t>(n_vocab),
                    0.0f);
    const i32 got = dflash_->draft_block(tok_embd, lm_head, n_vocab, id_last,
                                         n_draft, pos, out.data.data());
    out.data.resize(static_cast<size_t>(std::max<i32>(got, 0)) *
                    static_cast<size_t>(n_vocab));
    return got;
}

bool Engine::load_draft(const std::string &path, std::string *err) {
    if (!model_.speculative_rewind_ok()) {
        // Speculation verifies a block and then rewinds to the accepted
        // prefix; a delta-rule state has no inverse, so the rewind would leave
        // the state describing rejected tokens. Refuse rather than decode
        // something plausible and wrong. A shortconv ring rewinds cleanly
        // (model.hpp, `speculative_rewind_ok`), so lfm2moe is not refused here.
        if (err)
            *err = "speculative decoding needs a partial KV rewind, which a "
                   "recurrent (gated delta net) model cannot do";
        return false;
    }
    if (!be_) {
        if (err) *err = "the engine must be initialised before the draft model";
        return false;
    }
    if (draft_ || dflash_) return true; // already loaded

    // A DFlash head set and a full draft model arrive the same way, through
    // --draft, and nothing about the path says which one it is. The file does,
    // so the probe is a metadata read of a few kilobytes against a load that
    // costs the whole model. The mapping is dropped before returning, because on
    // Windows it would otherwise hold the file open for the rest of the run.
    {
        Gguf probe;
        std::string perr;
        if (probe.load(path, &perr) &&
            dflash_arch(probe.get_str("general.architecture", "")))
            return load_dflash(path, err);
    }

    // The draft shares this engine's backend: one device, interleaved kernels,
    // no staging copies. Its Model owns its own weights and its own KV cache.
    draft_ = new Engine();
    EngineConfig dcfg = cfg_;
    dcfg.model_path = path;
    dcfg.warmup = false;
    dcfg.expert_cache_mb = cfg_.expert_cache_mb;
    dcfg.expert_warm_mb = cfg_.expert_warm_mb;
    if (!draft_->init(be_, dcfg, err)) {
        delete draft_;
        draft_ = nullptr;
        return false;
    }

    const ModelConfig &tc = model_.cfg();
    const ModelConfig &dc = draft_->model().cfg();
    if (dc.n_vocab != tc.n_vocab) {
        if (err)
            *err = format("draft vocab (%d) does not match the target vocab (%d)",
                          dc.n_vocab, tc.n_vocab);
        unload_draft();
        return false;
    }
    if (draft_->kv_capacity() > kv_cap_) {
        if (err)
            *err = "the draft model's context is longer than the target context";
        unload_draft();
        return false;
    }
    if (draft_tokens_ <= 0) draft_tokens_ = 4;
    KRK_INFO("draft model loaded: %s (%d layers, %d embd) — speculative window %d",
             dc.name.c_str(), dc.n_layer, dc.n_embd, draft_tokens_);
    return true;
}

void Engine::unload_draft() {
    if (!draft_) return;
    draft_->shutdown();
    delete draft_;
    draft_ = nullptr;
    draft_proposed_ = 0;
    draft_accepted_ = 0;
    draft_sampled_e_ = 0.0;
    draft_sampled_n_ = 0;
    spec_steps_ = 0;
}

void Engine::recurrent_reset() {
    // The shortconv ring is its own state and must be cleared even on a stack
    // that owns neither delta-net buffer (lfm2moe does). Zeroing it is what
    // makes a full rewind plus replay read the same taps as a fresh session.
    if (sc_state_)
        be_->fill0(sc_state_, static_cast<size_t>(model_.cfg().n_layer) *
                               static_cast<size_t>(sc_state_span_) *
                               be_->act_size());
    if (!conv_state_ || !rec_state_) return;
    be_->fill0(conv_state_, static_cast<size_t>(rec_layers_) *
                                  static_cast<size_t>(conv_state_span_) * sizeof(f32));
    // The mamba-2 span and the GDN span are the same allocation site: the
    // recomputed version says exactly how much the scan owns, and fill0
    // zeroes the real span whichever layout the model uses.
    const i64 rs = mamba2_state_span_ > 0 ? mamba2_state_span_ : rec_state_span_;
    be_->fill0(rec_state_, static_cast<size_t>(rec_layers_) *
                                 static_cast<size_t>(rs) * sizeof(f32));
}

void Engine::compressed_reset() {
    for (auto &v : comp_state_kv_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : comp_state_sc_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : idx_state_kv_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : idx_state_sc_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : comp_pooled_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : idx_key_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto &k : comp_kpos_) std::fill(k.begin(), k.end(), 0);
    std::fill(comp_nblocks_.begin(), comp_nblocks_.end(), 0);
}

// The Mamba-2 block forward (nemotron_h / nemotron_h_moe). The engine does
// everything the GDN path does in the same op order — projection, short conv
// (with this block's bias), scan, gate, norm, out-projection, residual —
// leaving only the recurrence itself to Backend::ssd_scan, whose scalar
// implementation is the reference the device kernel is checked against.
//
// The fused ssm_in row is [z | xBC | dt]: z first (d_inner), then xBC
// (d_inner + 2*n_group*d_state, convolved), then dt (n_heads). The row that
// feeds the scan is the POST-CONV buffer, so ws_qkv_ is rearranged into
// [xBC | dt] order once after the conv — z is copied out at projection time
// because the conv would otherwise have to know to skip it.
void Engine::mamba2_forward(const LayerWeights &L, i32 l, i32 n) {
    const ModelConfig &mc = model_.cfg();
    const i64 inner = value_dim_;   // nemotron: ssm_value_dim == inner_size
    const i64 cdim = conv_dim_;     // inner + 2*n_group*d_state
    const i64 n_head = mc.ssm_dt_rank;
    const i64 hd = inner / n_head;
    const i32 ksize = mc.ssm_d_conv;
    const i32 ri = model_.recurrent_index(l);
    dump_row("m2.enter", l, ws_x_, n_embd_, n);

    // 1. The fused [z | xBC | dt] projection as THREE GEMMs over the weight's
    //    output-channel regions — the split lands z, xBC and dt exactly in
    //    their final buffers (dt straight into ws_beta_, where the scan reads
    //    it), so no activation row is ever touched on the host: activations
    //    live in device memory, and the only correct split of a fused weight
    //    is a split of the weight. `ssm_in` is [n_embd, 10304] laid out as
    //    [z(4096) | xBC(6144) | dt(64)] output channels.
    //
    //    THE ROW STRIDE IS THE INPUT WIDTH, NOT THE REGION WIDTH. One output
    //    channel owns `n_embd` quantized elements, so region r starts at
    //    `r * dtype_row_bytes(type, n_embd)`. Using dtype_row_bytes(type, r)
    //    -- the bytes of r *elements* -- lands a fraction of a row in
    //    (inner/n_embd = 4096/2688 = 1.52 rows for xBC, 10240/2688 = 3.81 for
    //    dt) and reads every region off a shifted copy of its neighbours' rows.
    //    The magnitudes stay plausible, so the only way to see it is a dump
    //    against a reference: measured on the 30B nemotron, that split made z
    //    correct and xBC/dt read the wrong weight regions entirely, which is
    //    why every device and --cpu run of this model decoded garbage while
    //    agreeing with each other to 0.1%.
    const void *xn = ws_xn_;
    if (L.h_attn_in) {
        had_xform(ws_had_, ws_xn_, n, n_embd_, false, false);
        xn = ws_had_;
    }
    // NVFP4: the stored tensor is gs * W, so the projection's INPUT carries the
    // scalar -- once, into a staging row that all three fused regions then read,
    // because they share one weight. It also has to be the input rather than
    // each result: xBC goes through the short conv (which adds a bias) and a
    // silu, and neither is linear over a scale. Measured on the Bolt 74B file,
    // dropping it inflated z to 65,440 in the first block.
    if (L.ssm_in.gs != 1.0f) {
        be_->copy_act(ws_gs_, xn, static_cast<i64>(n) * n_embd_);
        be_->scale_act(ws_gs_, L.ssm_in.gs, n, n_embd_, n_embd_);
        xn = ws_gs_;
    }
    {
        const size_t rw = dtype_row_bytes(L.ssm_in.type, n_embd_);
        const u8 *w = static_cast<const u8 *>(L.ssm_in.data);
        be_->gemm(ws_z_, xn, w, L.ssm_in.type, inner, n_embd_, n);
        be_->gemm(ws_qkv_, xn, w + rw * static_cast<size_t>(inner),
                  L.ssm_in.type, cdim, n_embd_, n);
        be_->gemm(ws_beta_, xn, w + rw * static_cast<size_t>(inner + cdim),
                  L.ssm_in.type, n_head, n_embd_, n);
        // TEMP PROBE (remove): the three fused regions, before the conv.
        dump_row("m2.z", l, ws_z_, inner, n);
        dump_row("m2.xbc", l, ws_qkv_, cdim, n);
        dump_row("m2.dt_raw", l, ws_beta_, n_head, n);
    }

    // 3. The causal short convolution with this block's bias, and the SiLU
    //    that follows it. After this ws_qkv_ holds post-conv xBC only.
    f32 *cstate = static_cast<f32 *>(conv_state_) +
                  static_cast<size_t>(ri) * static_cast<size_t>(conv_state_span_);
    be_->conv1d_silu(ws_qkv_, ws_qkv_, cstate, L.ssm_conv1d.data,
                     L.ssm_conv1d.type, n, cdim, ksize, L.ssm_conv_bias);
    dump_row("m2.conv", l, ws_qkv_, cdim, n);

    // 4. The SSD scan. Reads dtA from ws_beta_, A/bias/D from the layer, and
    //    keeps this layer's [head][p][s] f32 state in the shared arena.
    f32 *state = static_cast<f32 *>(rec_state_) +
                 static_cast<size_t>(ri) * static_cast<size_t>(mamba2_state_span_);
    be_->ssd_scan(ws_h_, state, ws_qkv_, ws_beta_, L.ssm_a, L.ssm_dt, L.ssm_d,
                  n, n_head, hd, mc.ssm_d_state, mc.ssm_n_group, inner, cdim,
                  L.ssm_d != nullptr);
    dump_row("m2.ssd", l, ws_h_, inner, n);

    // 5. The silu(z) output gate, then the GROUPED RMSNorm. ssm_norm is
    //    [inner / n_group, n_group]: ONE VECTOR PER GROUP, each reduced over
    //    that group's inner/n_group channels (Mamba-2's RMSNormGated with
    //    group_size; llama.cpp builds it as an rms_norm over the 4-D reshape
    //    [inner/n_group, n_group, T, S], so the weight's second axis is
    //    broadcast over the groups). The scan's [t, h*hd+p] layout already puts
    //    a group's channels together -- group g owns heads
    //    [g*n_head/n_group, (g+1)*n_head/n_group), so its block is
    //    [t*inner + g*(inner/n_group), +(inner/n_group)) -- which is what makes
    //    this one op over the n_tok * n_group rows instead of a transpose.
    //    Two ways to get this wrong that a scan-only check cannot see: using
    //    one shared weight vector for every group (rmsnorm() cannot express the
    //    per-group select at all), and passing a row count that does not match
    //    the buffer.
    be_->silu_mul(ws_h_, ws_z_, ws_h_, static_cast<i64>(n) * inner);
    be_->rmsnorm_grouped(ws_h_, ws_h_, L.ssm_norm, n, mc.ssm_n_group,
                         inner / mc.ssm_n_group, inner, mc.rms_eps);
    dump_row("m2.norm", l, ws_h_, inner, n);

    // 6. Output projection and the residual. ssm_out is [n_embd, inner] like
    //    every other out-projection on this stack.
    const void *h_in = ws_h_;
    if (L.h_ssm_out) {
        had_xform(ws_had_, ws_h_, n, inner, false, false);
        h_in = ws_had_;
    }
    // The output projection is a plain linear map plus the residual, so its
    // sidecar rides the projection's own result (gemm_residual's `gs`), never the
    // residual it is added to.
    gemm_residual(ws_x_, h_in, L.ssm_out.data, L.ssm_out.type, n_embd_, inner,
                  n, ws_x2_, L.ssm_out.gs);
    dump_row("m2.done", l, ws_x_, n_embd_, n);
}

// Multi-head latent attention, legacy lite layout (DeepSeek-V2-Lite).
//
// The whole point of the layer is what the KV cache does NOT hold. Instead of
// n_kv x head_dim of K and V it holds ONE row per token -- the compressed
// latent (kv_lora wide, rms-normed at write time) followed by the decoupled
// RoPE key (qk_rope wide, stored UNROTATED, because a key has to rotate at its
// own position and this cache row is written once). On V2-Lite that is 576 f16
// against a GQA row of 3072 x 2.
//
// What runs per layer, in order:
//   1. attn_kv_a_mqa: [n_embd] -> [kv_lora + rope]       (one GEMM, n rows)
//   2. mla_kv_write:  norm the latent, copy the key, store the row
//   3. attn_q:        [n_embd] -> [n_head*(nope+rope)]   (direct: lite has no
//                     q_lora trunk, which is what makes it "lite")
//   4. mla_attention: absorb the query per head, score the latent + the
//                     rotated decoupled key, softmax, accumulate the latent,
//                     fold the value half
// and the caller's ordinary output projection consumes the result.
bool Engine::mla_forward(const LayerWeights &L, i32 l, i32 n, i32 pos0) {
    // The two layouts are different graphs, not two shapes of one: the shared
    // one keeps a SINGLE latent row per token as both K and V and folds the
    // head count into a grouped output projection, so there is no per-head
    // table to absorb. Dispatch here so the layer loop and everything after it
    // (residual add, FFN, dumps) stay one path.
    if (model_.cfg().mla_layout == MlaLayout::Shared) {
        // A layer the file compresses runs the two-row-family path; every other
        // layer of the same file runs the dense one. The braces are not style:
        // without them the inner `if` swallows the dense call and every LAYOUT
        // lands in the shared path, which the minicpm3 groups caught.
        if (l >= 0 && static_cast<size_t>(l) < comp_ratio_.size() &&
            comp_ratio_[static_cast<size_t>(l)] != 0)
            return mla_shared_forward_compressed(L, l, n, pos0);
        return mla_shared_forward(L, l, n, pos0);
    }
    KRK_TRACE_COMP("mla-forward");
    const ModelConfig &mc = model_.cfg();
    const i64 nh = model_.n_head_at(l);
    const i64 nope = mc.mla_qk_nope;
    const i64 rope = mc.mla_qk_rope;
    const i64 kv_lora = mc.mla_kv_lora;
    const i64 vdim = mc.mla_v_dim;
    const i64 qk = nope + rope;
    const i64 row = kv_lora + rope; // one cache row, in activation elements

    // 1. the compressed projection. Its two halves have different fates -- the
    // latent is normed and attended, the key is neither -- but they come out of
    // ONE matmul, which is why the file stores one matrix for both.
    // The compressed projection's weight has TWO spellings: the legacy layout
    // stores it as `attn_kv_a_mqa` and the loader keeps it on `wkv_a_mqa`,
    // while the V3-class split layout's branch keeps the identical
    // [n_embd, kv_lora + rope] tensor on `wkv_a`. Reading one name and not the
    // other left the split layout projecting against an EMPTY weight: the GEMM
    // wrote nothing, `ws_mla_` kept whatever the previous stage left, and the
    // crash came two kernels later -- measured as 2.07e37 in the latent dump
    // on a V3 fixture, i.e. plausible-looking garbage rather than an error.
    const QuantTensor &kv_a = L.wkv_a_mqa;
    gemm_w(ws_mla_, ws_xn_, kv_a, row, n_embd_, n);
    dump_row("mla.kv", l, ws_mla_, row, n);

    // The plane this layer appends to. base() always returns this layer alone
    // (a slot when the cache is tiered), so pos_stride below is the ROW and not
    // a layer pitch -- the same contract the MHA chain is handed, and the one
    // place an MLA layer and an MHA layer can be wired differently by accident.
    void *kbase = kvt_k_.base(l);
    if (!kbase) {
        KRK_ERROR("mla: no resident KV slot for layer %lld",
                  static_cast<long long>(l));
        return false;
    }

    MlaDesc d;
    d.n_head = nh;
    d.qk = qk;
    d.nope = nope;
    d.rope = rope;
    d.kv_lora = kv_lora;
    d.v_dim = vdim;
    d.n_tok = n;
    d.pos0 = pos0;
    d.pos_stride = row;
    // The reference's kq_scale for this family: mscale^2 / sqrt(key_length),
    // which is NOT 1/sqrt(v_dim) and not 1/sqrt(rope) -- the score is a dot
    // over the whole per-head key, nope + rope -- and not plain 1/sqrt(qk)
    // either on a YaRN file.
    //
    // This used to be plain 1/sqrt(qk), on a written-down argument that the
    // reference's mscale "is built from rope.freq_scale, and when that is 1.0
    // the whole correction is 1.0 too". The premise is false for a YaRN file:
    // rope_freq_scale IS 1/factor on one (llama-model.cpp:1516-1518 sets
    // rope_freq_scale_train = 1/ropescale, llama-context.cpp:227 adopts it when
    // the caller does not override), so on a factor-40 DeepSeek file the
    // parenthesis is 1 + 0.1*mscale_all_dim*ln(40) = 1.369 and the score scale
    // is 1.874x the plain one -- a 36.6% amplitude error on every score, which
    // reads as fluent text with a different word a few tokens in.
    //
    // The loader resolves the multiplier once (ModelConfig::mla_kq_mscale, see
    // deepseek_yarn_mscale for the full chain); this path is the V2/V3 one and
    // is the only consumer, because deepseek4/deepseek41 pass plain
    // 1/sqrt(n_embd_head) in the reference.
    const f32 mscale = mc.mla_kq_mscale;
    d.scale = mscale * mscale / std::sqrt(static_cast<f32>(qk));
    d.eps = mc.rms_eps;
    // The file's own rotary table, YaRN already folded into the frequencies by
    // the loader. The kernel rotates the q rope half and each cache row's key
    // half with it, at that row's own position -- which is why the cache keeps
    // the key UNROTATED.
    d.inv_freq = model_.inv_freq_at(l).data();

    // 2. the query, directly projected: head h owns columns
    // [h*(nope+rope), ...), nope first -- the file's own packing, and the same
    // packing mla_attention reads.
    // 2. the query. A file with a `q_lora_rank` stores a LOW-RANK TRUNK
    // (attn_q_a -> attn_q_a_norm -> attn_q_b) instead of one direct
    // projection, and the two are different graphs -- not two shapes of one.
    // Only the direct form was ever projected here, so on a trunk file the
    // missing tensor was multiplied by an absent weight and the queries came
    // out at 2.8e22, turning into NaN one layer later with no error from the
    // loader: it had read and shape-checked the trunk, and the forward simply
    // never asked for it. Measured on a MiniCPM3 fixture (the V2-Lite KV path
    // with a trunk over it), which is the pairing no earlier fixture covered.
    if (L.wq_a.present()) {
        const i64 ql = mc.mla_q_lora;
        gemm_w(ws_mlaqn_, ws_xn_, L.wq_a, ql, n_embd_, n);
        be_->rmsnorm(ws_mlaqn_, ws_mlaqn_, L.wq_a_norm, n, ql, mc.rms_eps,
                     nullptr);
        dump_row("mla.qa", l, ws_mlaqn_, ql, n);
        gemm_w(ws_q_, ws_mlaqn_, L.wq_b, nh * qk, ql, n);
    } else if (L.wq_direct.present()) {
        gemm_w(ws_q_, ws_xn_, L.wq_direct, nh * qk, n_embd_, n);
    } else {
        KRK_ERROR("mla: layer %lld carries neither a query trunk "
                  "(attn_q_a/attn_q_b) nor a direct query projection",
                  static_cast<long long>(l));
        return false;
    }
    dump_row("mla.q", l, ws_q_, nh * qk, n);

    // 3. append this pass's rows. Must precede the attention call: row pos0+0 is
    // visible to itself, so the write and the read are the same pass.
    be_->mla_kv_write(kbase, ws_mla_, L.wkv_a_norm, d);
    // 4. attention over the prefix, written straight into ws_attn_ at
    // n_head*v_dim columns per row -- the width the output projection reads.
    be_->mla_attention(ws_attn_, ws_q_, kbase, L.wkv_b16.data, d);
    // The layer's KV is queued: a later layer that evicts this slot has to wait
    // for these kernels instead of overwriting memory they still read.
    kvt_k_.end_layer(l);
    dump_row("mla.out", l, ws_attn_, nh * vdim, n);
    return true;
}

// Multi-head latent attention, SHARED-latent layout (DeepSeek-V4).
//
// The lite path above stores `[normed latent | decoupled rope key]` and absorbs
// a per-head table; this one stores a SINGLE row per token and uses it as BOTH
// K (rotated on read, at the row's own position) and V (as written). The head
// count comes back in the output instead: `attn_output_a` holds one o_lora-wide
// column block per group over that group's concatenated heads, and
// `attn_output_b` collapses the concatenation to n_embd. So this function ends
// with the layer's FINAL attention output, and the caller must not run the
// ordinary `attn_output` projection over it a second time (see the layer loop's
// skip).
//
// Arithmetic is the host oracle in src/mla_shared.cpp, whose two orders the
// suite pins; the kernels reproduce it on the file's own f16 staging, which is
// why every table below is the `*16` tensor rather than the quantized one.
bool Engine::mla_shared_forward(const LayerWeights &L, i32 l, i32 n, i32 pos0) {
    KRK_TRACE_COMP("mla-shared-forward");
    const ModelConfig &mc = model_.cfg();
    const i64 nh = model_.n_head_at(l);
    const i64 d = mc.mla_kv_lora;  // key_length == value_length == the latent
    const i64 ql = mc.mla_q_lora;
    const i64 gout = static_cast<i64>(mc.dsv4_o_groups) *
                     static_cast<i64>(mc.dsv4_o_lora);
    // A missing staging tensor is not a slow path, it is a different model: the
    // grouped output reads wo_a/wo_b unconditionally and the query trunk reads
    // q_a/q_b, so refuse before a single kernel runs.
    if (ql <= 0 || d <= 0 || gout <= 0 || !L.wq_a16.present() ||
        !L.wq_b16.present() || !L.wkv16.present() || !L.wo_a16.present() ||
        !L.wo_b16.present() || !L.wq_a_norm || !L.wkv_norm) {
        KRK_ERROR("mla-shared: layer %d is missing its staged tables (q_lora %lld, "
                  "latent %lld, gout %lld)",
                  static_cast<int>(l), static_cast<long long>(ql),
                  static_cast<long long>(d), static_cast<long long>(gout));
        return false;
    }
    if (d != mc.mla_qk_nope + mc.mla_qk_rope) {
        KRK_ERROR("mla-shared: layer %d latent %lld != nope+rope %d",
                  static_cast<int>(l), static_cast<long long>(d),
                  mc.mla_qk_nope + mc.mla_qk_rope);
        return false;
    }
    void *kbase = kvt_k_.base(l);
    if (!kbase) {
        KRK_ERROR("mla-shared: no resident KV slot for layer %d",
                  static_cast<int>(l));
        return false;
    }

    MlaSharedDesc sd;
    sd.n_head = nh;
    sd.kv_lora = d;
    sd.qk_rope = mc.mla_qk_rope;
    sd.o_groups = mc.dsv4_o_groups;
    sd.o_lora = mc.dsv4_o_lora;
    sd.n_embd = n_embd_;
    sd.n_tok = n;
    sd.pos0 = pos0;
    // One row: K and V are the same latent, so the stride IS the row.
    sd.pos_stride = d;
    // The score is a dot over the whole latent, so this is 1/sqrt(key_length)
    // and nothing else. The mscale correction belongs to the lite (V2/V3)
    // family's YaRN files and is deliberately NOT applied here: the reference's
    // deepseek4/deepseek41 models pass a plain 1/sqrt(n_embd_head) too
    // (src/models/deepseek4.cpp:1195-1206, deepseek41.cpp:427). Applying
    // mla_forward's mscale here would make a V4 file 1.87x too hot.
    sd.scale = 1.0f / std::sqrt(static_cast<f32>(d));
    sd.eps = mc.rms_eps;
    sd.window = mc.dsv4_window > 0 ? mc.dsv4_window : 0;
    sd.inv_freq = model_.inv_freq_at(l).data();
    // `attention.sinks` per head: a learned logit that competes with every
    // visible key for softmax mass and contributes no value of its own (the
    // reference adds it with `ggml_soft_max_add_sinks`). Both the host oracle
    // (src/mla_shared.cpp) and this kernel already carry it -- the engine was
    // the only place that dropped it, and refused the file instead.
    sd.sinks = L.sinks;

    // 1. the latent: x -> attn_kv -> rms_norm over the WHOLE row (rope is its
    //    tail, so the norm covers the rotary dims -- the opposite of the lite
    //    layout, whose key sits beside the normed latent).
    gemm_w(ws_mla_, ws_xn_, L.wkv16, d, n_embd_, n);
    dump_row("mla.kv", l, ws_mla_, d, n);

    // 2. the query trunk: q_a -> its own norm -> q_b, one row at a time.
    gemm_w(ws_mlaqn_, ws_xn_, L.wq_a16, ql, n_embd_, n);
    be_->rmsnorm(ws_mlaqn_, ws_mlaqn_, L.wq_a_norm, n, ql, mc.rms_eps,
                 nullptr);
    dump_row("mla.qa", l, ws_mlaqn_, ql, n);
    gemm_w(ws_q_, ws_mlaqn_, L.wq_b16, nh * d, ql, n);
    dump_row("mla.q", l, ws_q_, nh * d, n);

    // 3. cache the normed latent, then attend over the prefix. The write must
    //    precede the read: row pos0 is visible to itself.
    be_->mla_shared_latent(kbase, ws_mla_, L.wkv_norm, sd);
    be_->mla_shared_attention(ws_mlaa_, ws_q_, kbase, sd);
    kvt_k_.end_layer(l);
    dump_row("mla.out", l, ws_mlaa_, nh * d, n);

    // 4. the grouped low-rank output: this IS the layer's attention result, so
    //    it lands in ws_x2_ where the ordinary path keeps `wo`'s output.
    be_->mla_shared_output(ws_x2_, ws_mlaa_, ws_mlay_, L.wo_a16.data,
                           L.wo_b16.data, sd);
    dump_row("attn.proj", l, ws_x2_, n_embd_, n);
    return true;
}

// DeepSeek-V4 COMPRESSED attention (CSA/HCA): the shared-latent layer with a
// second row family in the attention.
//
// What is different from mla_shared_forward, and only this:
//
//   1. Every token's input row is also projected into a compressor STATE row
//      (`dsv4_comp_state`: Wkv x | Wgate x + APE[pos % ratio]). The rows are
//      kept per layer in a two-block ring, because the pool reads the previous
//      block and the current one.
//   2. When a token completes a block, the block is pooled into ONE key
//      (`dsv4_comp_pool`: a per-channel softmax over the block's rows, then the
//      weighted sum, then the norm) and stored with the block's FIRST token as
//      its position. A pooled key is visible to every later token -- it
//      summarizes positions the sliding window has already passed -- but NOT to
//      the tokens of its own block, which is why the pool runs AFTER this row's
//      attention and not before.
//   3. The attention reads both families: the raw window and the pooled keys.
//      HCA (ratio 128) reads all of them; CSA (ratio 4) runs the lightning
//      indexer over its own pooled keys and reads the top-k. Per ROW, so the
//      loop is per token on a CSA layer even during prefill -- a selection is a
//      property of the query, and one shared set for a whole chunk would
//      attend blocks that a later row's indexer never chose.
//
// Everything else -- the query trunk, the latent write, the grouped output --
// is the dense path's code, unchanged, because the file says the layer is the
// same layer (deepseek4.cpp builds one graph with a conditional block).
//
// The arithmetic here is HOST f32 through the oracle in src/dsv4_compress.cpp:
// the tables it needs are small and the compressor is a scalar routine, and
// this path is gated on `mla_shared_compressed_supported()`, which only the CPU
// reference answers true for. A device backend that implements it later
// replaces these calls with kernels; the oracle stays the definition.
bool Engine::mla_shared_forward_compressed(const LayerWeights &L, i32 l, i32 n,
                                           i32 pos0) {
    KRK_TRACE_COMP("mla-shared-forward-compressed");
    const ModelConfig &mc = model_.cfg();
    const size_t li = static_cast<size_t>(l);
    if (li >= comp_ratio_.size() || comp_ratio_[li] == 0) {
        KRK_ERROR("mla-shared-compressed: layer %d carries no compress ratio",
                  static_cast<int>(l));
        return false;
    }
    if (!be_->mla_shared_compressed_supported()) {
        KRK_ERROR("mla-shared-compressed: layer %d needs the compressed-"
                  "attention op and this backend does not implement it",
                  static_cast<int>(l));
        return false;
    }
    const i32 ratio = comp_ratio_[li];
    const i64 nh = model_.n_head_at(l);
    const i64 d = mc.mla_kv_lora;
    const i64 ql = mc.mla_q_lora;
    const i64 rope = mc.mla_qk_rope;
    const i64 gout = static_cast<i64>(mc.dsv4_o_groups) *
                     static_cast<i64>(mc.dsv4_o_lora);
    if (ql <= 0 || d <= 0 || gout <= 0 || !L.wq_a16.present() ||
        !L.wq_b16.present() || !L.wkv16.present() || !L.wo_a16.present() ||
        !L.wo_b16.present() || !L.wq_a_norm || !L.wkv_norm) {
        KRK_ERROR("mla-shared-compressed: layer %d is missing its staged tables",
                  static_cast<int>(l));
        return false;
    }
    if (L.comp_wkv.empty() || L.comp_wgate.empty() || L.comp_ape.empty() ||
        L.comp_norm.empty()) {
        KRK_ERROR("mla-shared-compressed: layer %d declares ratio %d and carries "
                  "no compressor tables",
                  static_cast<int>(l), static_cast<int>(ratio));
        return false;
    }
    void *kbase = kvt_k_.base(l);
    if (!kbase) {
        KRK_ERROR("mla-shared-compressed: no resident KV slot for layer %d",
                  static_cast<int>(l));
        return false;
    }
    const i64 coff = ratio == kDsv4CsaRatio ? 2 : 1;
    const i64 sw = coff * d;
    const i64 ih = mc.idx_head_dim;
    const i64 inh = mc.idx_n_head;
    const bool csa = ratio == kDsv4CsaRatio;

    Dsv4Compressor cc;
    cc.ratio = ratio;
    cc.n_embd = n_embd_;
    cc.hd = static_cast<i32>(d);
    cc.rope = static_cast<i32>(rope);
    cc.rope_base = mc.dsv4_compress_rope_base;
    cc.eps = mc.rms_eps;
    cc.wkv = L.comp_wkv.data();
    cc.wgate = L.comp_wgate.data();
    cc.ape = L.comp_ape.data();
    cc.norm = L.comp_norm.data();
    if (!cc.valid()) {
        KRK_ERROR("mla-shared-compressed: layer %d's compressor descriptor does "
                  "not validate (ratio %d, latent %lld, rope %lld)",
                  static_cast<int>(l), static_cast<int>(ratio),
                  static_cast<long long>(d), static_cast<long long>(rope));
        return false;
    }
    // The indexer's own compressor: ALWAYS the two-wide overlapping one, and its
    // pooled key is idx_head wide with the same rotary tail.
    Dsv4Compressor ic;
    if (csa) {
        if (L.idx_proj.empty() || L.idx_q_b.empty() || L.idx_wkv.empty() ||
            L.idx_wgate.empty() || L.idx_ape.empty() || L.idx_norm.empty() ||
            ih <= 0 || inh <= 0 || rope > ih) {
            KRK_ERROR("mla-shared-compressed: layer %d is CSA (ratio 4) and its "
                      "indexer tables do not match the declared geometry",
                      static_cast<int>(l));
            return false;
        }
        ic.ratio = kDsv4CsaRatio;
        ic.n_embd = n_embd_;
        ic.hd = static_cast<i32>(ih);
        ic.rope = static_cast<i32>(rope);
        ic.rope_base = mc.dsv4_compress_rope_base;
        ic.eps = mc.rms_eps;
        ic.wkv = L.idx_wkv.data();
        ic.wgate = L.idx_wgate.data();
        ic.ape = L.idx_ape.data();
        ic.norm = L.idx_norm.data();
        if (!ic.valid()) {
            KRK_ERROR("mla-shared-compressed: layer %d's indexer compressor does "
                      "not validate",
                      static_cast<int>(l));
            return false;
        }
    }

    MlaSharedDesc sd;
    sd.n_head = nh;
    sd.kv_lora = d;
    sd.qk_rope = rope;
    sd.o_groups = mc.dsv4_o_groups;
    sd.o_lora = mc.dsv4_o_lora;
    sd.n_embd = n_embd_;
    sd.n_tok = n;
    sd.pos0 = pos0;
    sd.pos_stride = d;
    sd.scale = 1.0f / std::sqrt(static_cast<f32>(d));
    sd.eps = mc.rms_eps;
    sd.window = mc.dsv4_window > 0 ? mc.dsv4_window : 0;
    sd.inv_freq = model_.inv_freq_at(l).data();
    sd.sinks = L.sinks;

    // The query trunk is the dense path's, row for row.
    gemm_w(ws_mla_, ws_xn_, L.wkv16, d, n_embd_, n);
    gemm_w(ws_mlaqn_, ws_xn_, L.wq_a16, ql, n_embd_, n);
    be_->rmsnorm(ws_mlaqn_, ws_mlaqn_, L.wq_a_norm, n, ql, mc.rms_eps, nullptr);
    gemm_w(ws_q_, ws_mlaqn_, L.wq_b16, nh * d, ql, n);
    dump_row("mla.q", l, ws_q_, nh * d, n);

    const size_t as = be_->act_size();
    const f32 *host_xn = static_cast<const f32 *>(ws_xn_);
    const f32 *host_ql = static_cast<const f32 *>(ws_mlaqn_);
    const f32 *host_q = static_cast<const f32 *>(ws_q_);
    f32 *pooled = comp_pooled_[li].data();
    i64 *kpos = comp_kpos_[li].data();
    const i64 prows = coff * ratio;
    const f32 iw_scale = csa ? 1.0f / std::sqrt(static_cast<f32>(ih * inh)) : 0.0f;

    for (i32 t = 0; t < n; t++) {
        const i64 pos = pos0 + t;
        MlaSharedDesc one = sd;
        one.n_tok = 1;
        one.pos0 = pos;
        // 1. the latent row, written before it is read: row pos is visible to
        //    itself, exactly as in the dense path.
        be_->mla_shared_latent(
            kbase,
            static_cast<const char *>(ws_mla_) + static_cast<size_t>(t) * d * as,
            L.wkv_norm, one);
        // 2. this token's state rows, into the ring slot for its position.
        const f32 *xrow = host_xn + static_cast<size_t>(t) * n_embd_;
        const i32 pinb = static_cast<i32>(pos % ratio);
        const i64 slot = (coff == 1) ? pinb : ((pos / ratio) % 2) * ratio + pinb;
        if (!dsv4_comp_state(cc, xrow, pinb, comp_row_kv_.data(),
                             comp_row_sc_.data())) {
            KRK_ERROR("mla-shared-compressed: layer %d state row at pos %lld",
                      static_cast<int>(l), static_cast<long long>(pos));
            return false;
        }
        std::memcpy(comp_state_kv_[li].data() + slot * sw, comp_row_kv_.data(),
                    static_cast<size_t>(sw) * sizeof(f32));
        std::memcpy(comp_state_sc_[li].data() + slot * sw, comp_row_sc_.data(),
                    static_cast<size_t>(sw) * sizeof(f32));
        if (csa) {
            const i64 islot =
                ((pos / ratio) % 2) * ratio + pinb;  // always two-wide
            if (!dsv4_comp_state(ic, xrow, pinb, comp_idx_kv_.data(),
                                 comp_idx_sc_.data())) {
                KRK_ERROR("mla-shared-compressed: layer %d indexer state at pos "
                          "%lld",
                          static_cast<int>(l), static_cast<long long>(pos));
                return false;
            }
            std::memcpy(idx_state_kv_[li].data() + islot * 2 * ih,
                        comp_idx_kv_.data(),
                        static_cast<size_t>(2 * ih) * sizeof(f32));
            std::memcpy(idx_state_sc_[li].data() + islot * 2 * ih,
                        comp_idx_sc_.data(),
                        static_cast<size_t>(2 * ih) * sizeof(f32));
        }
        // 3. the attention for THIS row, over the raw window and the pooled
        //    keys of the blocks that are already complete.
        i64 nb = comp_nblocks_[li];
        const i32 *sel = nullptr;
        i64 nsel = 0;
        if (csa && nb > 0) {
            // The indexer: q from the q_lora trunk through its own q_b, roped on
            // the tail at THIS position with the compression base; weights from
            // the raw input row; score = sum_h relu(q_h . k_b) * w_h over the
            // pooled indexer keys, then the top-k blocks.
            std::fill(comp_iq_.begin(), comp_iq_.end(), 0.0f);
            const f32 *qlrow = host_ql + static_cast<size_t>(t) * ql;
            for (i64 k2 = 0; k2 < ql; k2++) {
                const f32 qk = qlrow[k2];
                if (qk == 0.0f) continue;
                const f32 *wrow = &L.idx_q_b[static_cast<size_t>(k2 * inh * ih)];
                for (i64 c = 0; c < inh * ih; c++)
                    comp_iq_[static_cast<size_t>(c)] += qk * wrow[c];
            }
            for (i64 h2 = 0; h2 < inh; h2++)
                dsv4_comp_rope(&comp_iq_[static_cast<size_t>(h2 * ih)],
                               static_cast<i32>(ih), static_cast<i32>(rope), pos,
                               mc.dsv4_compress_rope_base);
            for (i64 h2 = 0; h2 < inh; h2++) {
                f32 s2 = 0.0f;
                for (i64 k2 = 0; k2 < n_embd_; k2++)
                    s2 += xrow[k2] * L.idx_proj[static_cast<size_t>(k2 * inh + h2)];
                comp_iw_[static_cast<size_t>(h2)] = s2 * iw_scale;
            }
            if (!dsv4_indexer_scores(static_cast<i32>(inh), static_cast<i32>(ih),
                                     comp_iq_.data(), idx_key_[li].data(),
                                     comp_iw_.data(), nullptr,
                                     static_cast<i32>(nb), comp_score_.data())) {
                KRK_ERROR("mla-shared-compressed: layer %d indexer scores",
                          static_cast<int>(l));
                return false;
            }
            i32 kk = mc.idx_top_k > 0 ? mc.idx_top_k : 1;
            if (kk > nb) kk = static_cast<i32>(nb);
            dsv4_top_k(comp_sel_.data(), kk, comp_score_.data(),
                       static_cast<i32>(nb));
            sel = comp_sel_.data();
            nsel = kk;
        }
        be_->mla_shared_attention_compressed(
            static_cast<char *>(ws_mlaa_) +
                static_cast<size_t>(t) * nh * d * sizeof(f32),
            host_q + static_cast<size_t>(t) * nh * d, kbase, pooled, kpos, nb,
            sel, nsel, one, comp_inv_freq_.data());
        // 4. if this token completes a block, pool it -- AFTER this row's
        //    attention, so a pooled key summarizes a block that is entirely
        //    behind the query and never contributes to its own block.
        if (pinb == ratio - 1) {
            const i64 b = pos / ratio;
            for (i64 k2 = 0; k2 < prows; k2++) {
                const bool cur_half = (coff == 1) || (k2 >= ratio);
                i64 src;
                if (coff == 1) {
                    src = k2;
                } else if (cur_half) {
                    src = (b % 2) * ratio + (k2 - ratio);
                } else if (b == 0) {
                    // No previous block: the reference appends a ZERO value row
                    // with an -inf score (dsv4_append_zero_row in deepseek4.cpp),
                    // i.e. the half contributes nothing and its softmax weight
                    // is exactly 0. A zero score row instead would give the
                    // missing block a uniform share of the pool.
                    for (i64 i = 0; i < sw; i++) {
                        comp_pool_kv_[static_cast<size_t>(k2 * sw + i)] = 0.0f;
                        comp_pool_sc_[static_cast<size_t>(k2 * sw + i)] =
                            -INFINITY;
                    }
                    continue;
                } else {
                    src = ((b - 1) % 2) * ratio + k2;
                }
                std::memcpy(&comp_pool_kv_[static_cast<size_t>(k2 * sw)],
                            comp_state_kv_[li].data() + src * sw,
                            static_cast<size_t>(sw) * sizeof(f32));
                std::memcpy(&comp_pool_sc_[static_cast<size_t>(k2 * sw)],
                            comp_state_sc_[li].data() + src * sw,
                            static_cast<size_t>(sw) * sizeof(f32));
            }
            if (!dsv4_comp_pool(cc, comp_pool_kv_.data(), comp_pool_sc_.data(),
                                pooled + nb * d)) {
                KRK_ERROR("mla-shared-compressed: layer %d pool of block %lld",
                          static_cast<int>(l), static_cast<long long>(b));
                return false;
            }
            kpos[nb] = pos - ratio + 1;
            if (csa) {
                for (i64 k2 = 0; k2 < 2 * ratio; k2++) {
                    const bool cur_half = k2 >= ratio;
                    i64 src;
                    if (cur_half) {
                        src = (b % 2) * ratio + (k2 - ratio);
                    } else if (b == 0) {
                        for (i64 i = 0; i < 2 * ih; i++) {
                            comp_pool_kv_[static_cast<size_t>(k2 * 2 * ih + i)] =
                                0.0f;
                            comp_pool_sc_[static_cast<size_t>(k2 * 2 * ih + i)] =
                                -INFINITY;
                        }
                        continue;
                    } else {
                        src = ((b - 1) % 2) * ratio + k2;
                    }
                    std::memcpy(&comp_pool_kv_[static_cast<size_t>(k2 * 2 * ih)],
                                idx_state_kv_[li].data() + src * 2 * ih,
                                static_cast<size_t>(2 * ih) * sizeof(f32));
                    std::memcpy(&comp_pool_sc_[static_cast<size_t>(k2 * 2 * ih)],
                                idx_state_sc_[li].data() + src * 2 * ih,
                                static_cast<size_t>(2 * ih) * sizeof(f32));
                }
                f32 *iout = &idx_key_[li][static_cast<size_t>(nb * ih)];
                if (!dsv4_comp_pool(ic, comp_pool_kv_.data(),
                                    comp_pool_sc_.data(), iout)) {
                    KRK_ERROR("mla-shared-compressed: layer %d indexer pool of "
                              "block %lld",
                              static_cast<int>(l), static_cast<long long>(b));
                    return false;
                }
                // Stored ROTATED at the block's first token: the score is an
                // ordinary dot against the roped query, and the angle must
                // survive from the write to every later read.
                dsv4_comp_rope(iout, static_cast<i32>(ih), static_cast<i32>(rope),
                               kpos[nb], mc.dsv4_compress_rope_base);
            }
            comp_nblocks_[li] = nb + 1;
        }
    }
    kvt_k_.end_layer(l);
    // 5. the grouped low-rank output, the dense path's final step.
    be_->mla_shared_output(ws_x2_, ws_mlaa_, ws_mlay_, L.wo_a16.data,
                           L.wo_b16.data, sd);
    dump_row("attn.proj", l, ws_x2_, n_embd_, n);
    return true;
}

// LFM2.5's short convolution. Three steps and one kernel:
//
//   1. in_proj   n_embd -> 3*n_embd   (B | C | X, in that order)
//   2. shortconv: conv over B*X (causal, depthwise, kern taps), then C gates it
//   3. out_proj  3 slices -> n_embd, folded onto the residual
//
// There is no gate chain, no state matrix and no activation between the
// convolution and its gate -- the whole mixer is steps 1-3, which is why the
// block is 4 ops where the delta net is a dozen.
void Engine::lfm_conv_forward(const LayerWeights &L, i32 l, i32 n, i64 pos0) {
    KRK_TRACE_COMP("lfm-conv-forward");
    const i32 kern = model_.cfg().ssm_d_conv > 0 ? model_.cfg().ssm_d_conv : 3;
    // ws_xn_ was filled by the layer's pre-norm before the mixer ran.
    gemm_w(ws_sc_, ws_xn_, L.sc_in, 3 * n_embd_, n_embd_, n);
    void *state = static_cast<char *>(sc_state_) +
                  static_cast<size_t>(l) *
                      static_cast<size_t>(sc_state_span_) *
                      static_cast<size_t>(be_->act_size());
    be_->shortconv(ws_sc2_, ws_sc_, state, L.sc_conv.data, n, n_embd_, kern,
                   sc_hist_, pos0);
    dump_row("lfm.sc", l, ws_sc2_, n_embd_, n);
    // The output projection lands on the residual with its folded epilogue, so
    // the mixer costs one add fewer than a separate gemm + add would. ws_attn_
    // is the fallback scratch and is free here: a conv layer runs no attention.
    gemm_residual(ws_x_, ws_sc2_, L.sc_out.data, L.sc_out.type, n_embd_, n_embd_,
                  n, ws_attn_, L.sc_out.gs);
}

void Engine::gdn_forward(const LayerWeights &L, i32 l, i32 n) {
    KRK_TRACE_COMP("gdn-forward");
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const i64 kdim = model_.key_dim();
    const i64 cdim = conv_dim_;
    const i64 vdim = value_dim_;
    const i32 ksize = mc.ssm_d_conv;
    const i32 ri = model_.recurrent_index(l);

    // 1. FOUR projections read the attention-norm output here, not two: the
    //    [q | k | v] block and the z gate, and (below, in step 4) ssm_alpha and
    //    ssm_beta. None of them depends on another, and none of them reads
    //    anything the conv, the L2 norms or the gate chain writes, so they are
    //    one group launch -- 4 launches per layer become 1, which is 54 of
    //    Qwen3.5-0.8B's 451 ops in a decode step.
    //
    //    Row-parallel only, deliberately: ssm_alpha/ssm_beta are dt_rank rows,
    //    so the group's usual split-K top-up would fill the device for them and
    //    pay a reduce launch per split matrix for the privilege -- buying
    //    parallelism with exactly the launch this removes. With the splits
    //    pinned at 1 every row keeps gemv_kernel's accumulation order, so the
    //    group is BIT-IDENTICAL to the four launches it replaces; the stdout
    //    md5 is the check, because a regrouping would show up as different
    //    greedy text.
    //
    //    A folded q/k/v or z reads the hadamard copy (ws_had_) while alpha and
    //    beta read ws_xn_ primal, and a group shares one activation, so the
    //    group covers the unfolded case -- which is also the only case where
    //    all four read the same buffer. A dtype mix falls back too: the group
    //    carries one weight type.
    const void *xn = ws_xn_;
    if (L.h_attn_in || L.h_attn_gate) {
        had_xform(ws_had_, ws_xn_, n, n_embd_, false, false);
        xn = ws_had_;
    }
    static const bool gdn_group_ = [] {
        const char *e = std::getenv("KRK_GDN_GROUP");
        return !e || std::strcmp(e, "0") != 0;
    }();
    // A scaled weight keeps the group off: one launch cannot scale each output
    // differently, and these four projections decide gs per tensor.
    const bool gdn_onelaunch = gdn_group_ && !L.h_attn_in && !L.h_attn_gate &&
                               !L.wqkv.scaled() && !L.wqkv_gate.scaled() &&
                               !L.ssm_alpha.scaled() && !L.ssm_beta.scaled();
    if (gdn_onelaunch) {
        void *gout[4] = {ws_qkv_, ws_z_, ws_ssm_, ws_beta_};
        const void *gw[4] = {L.wqkv.data, L.wqkv_gate.data, L.ssm_alpha.data,
                             L.ssm_beta.data};
        const DType gtypes[4] = {L.wqkv.type, L.wqkv_gate.type,
                                 L.ssm_alpha.type, L.ssm_beta.type};
        const i64 gn_out[4] = {cdim, vdim, mc.ssm_dt_rank, mc.ssm_dt_rank};
        be_->gemm_group(gout, ws_xn_, gw, L.wqkv.type, gn_out, n_embd_, 4, n,
                        /*silu=*/false, /*silu_fused=*/nullptr,
                        /*row_parallel_only=*/true, gtypes);
    } else {
        gemm_w(ws_qkv_, L.h_attn_in ? xn : ws_xn_, L.wqkv, cdim, n_embd_, n);
        gemm_w(ws_z_, L.h_attn_gate ? xn : ws_xn_, L.wqkv_gate, vdim, n_embd_, n);
    }

    // 2. The causal short convolution, carrying the previous ksize-1 steps, and
    //    the SiLU that follows it. After this ws_qkv_ is the post-conv block.
    f32 *cstate = static_cast<f32 *>(conv_state_) +
                  static_cast<size_t>(ri) * static_cast<size_t>(conv_state_span_);
    be_->conv1d_silu(ws_qkv_, ws_qkv_, cstate, L.ssm_conv1d.data, L.ssm_conv1d.type,
                     n, cdim, ksize);
    dump_row("gdn.conv", l, ws_qkv_, cdim, n);

    // 3. The post-conv buffer splits into q | k | v. They are regions of one
    //    fused row, so every op below works on a row stride of cdim, not on a
    //    packed [n_tok, ...] block. q and k are L2-normalized per head (the
    //    delta rule is scale-free) and q picks up 1/sqrt(state).
    char *qbase = act_at(ws_qkv_, as, 0);
    char *kbase = act_at(ws_qkv_, as, kdim);
    const char *vbase = act_at(ws_qkv_, as, 2 * kdim);
    be_->l2norm(qbase, qbase, n, mc.ssm_n_group, mc.ssm_d_state, cdim, 1e-6f);
    be_->l2norm(kbase, kbase, n, mc.ssm_n_group, mc.ssm_d_state, cdim, 1e-6f);
    be_->scale_act(qbase, 1.0f / std::sqrt(static_cast<f64>(mc.ssm_d_state)), n, kdim,
                   cdim);

    // 4. The per-head gate and forget. A_log arrives already as -exp(A_log), so
    //    the decay is exp(-exp(A_log) * softplus(alpha + dt_bias)).
    // ssm_alpha came out of the group launch above when it ran.
    if (!gdn_onelaunch) {
        gemm_w(ws_ssm_, ws_xn_, L.ssm_alpha, mc.ssm_dt_rank, n_embd_, n);
    }
    be_->add_bias_cols(ws_ssm_, L.ssm_dt, n, mc.ssm_dt_rank);
    be_->softplus_act(ws_ssm_, static_cast<i64>(n) * mc.ssm_dt_rank);
    be_->scale_cols(ws_ssm_, L.ssm_a, n, mc.ssm_dt_rank);

    if (!gdn_onelaunch) {
        gemm_w(ws_beta_, ws_xn_, L.ssm_beta, mc.ssm_dt_rank, n_embd_, n);
    }
    be_->sigmoid_act(ws_beta_, static_cast<i64>(n) * mc.ssm_dt_rank);

    // 5. The delta rule, in order, carrying this layer's state to the next token.
    f32 *state = static_cast<f32 *>(rec_state_) +
                 static_cast<size_t>(ri) * static_cast<size_t>(rec_state_span_);
    be_->delta_rule(ws_h_, state, qbase, kbase, vbase, ws_ssm_, ws_beta_, n,
                    mc.ssm_n_group, mc.ssm_dt_rank, mc.ssm_d_state, mc.ssm_d_state,
                    cdim);
    dump_row("gdn.delta", l, ws_h_, vdim, n);

    // 6. Gated normalization: a plain RMSNorm per head (its weight is stored
    //    verbatim, not zero-centred) then the output gate.
    //
    //    THE GATE IS NOT THE SAME FUNCTION IN BOTH FAMILIES. Qwen3.5's
    //    `build_norm_gated` is `norm(x) * silu(z)`; qwen4exp's is
    //    `norm(x) * sigmoid(z)` -- llama.cpp's qwen4exp graph says so in one
    //    line ("the one numerical difference from Qwen3.5's GDN: sigmoid output
    //    gate, not silu"). The tensor list is identical, so nothing in the file
    //    distinguishes them: `ModelConfig::gdn_gate_sigmoid` carries the
    //    family's answer from the arch table, and KRK_GDN_GATE is the A/B arm
    //    (0 restores silu) that proves this line is what the graph reads.
    be_->rmsnorm(ws_h_, ws_h_, L.ssm_norm, static_cast<i64>(n) * mc.ssm_dt_rank,
                 mc.ssm_d_state, mc.rms_eps);
    dump_row("gdn.norm", l, ws_h_, vdim, n);
    //    gdn_gate_ is -1 unless KRK_GDN_GATE forced an arm (read in init), so
    //    the family's answer is the default and the A/B is one variable away.
    const bool sigmoid_gate = gdn_gate_ >= 0 ? gdn_gate_ == 1 : mc.gdn_gate_sigmoid;
    if (sigmoid_gate) {
        be_->sigmoid_act(ws_z_, static_cast<i64>(n) * vdim);
        be_->mul_act(ws_h_, ws_z_, static_cast<i64>(n) * vdim);
    } else {
        be_->silu_mul(ws_h_, ws_z_, ws_h_, static_cast<i64>(n) * vdim);
    }
    dump_row("gdn.gated", l, ws_h_, vdim, n);

    // 7. Output projection and the residual. ssm_out is folded, and its input
    //    carries the grouped-V permutation ahead of the signs -- the delta-rule
    //    output is [hd, nk, rep] over the value heads and the stored weight
    //    expects [hd, rep, nk]. ws_had_ is free here: the projections in step
    //    1 have long consumed their copy.
    const void *h_in = ws_h_;
    if (L.h_ssm_out) {
        had_xform(ws_had_, ws_h_, n, vdim, false, /*gdn_perm=*/true);
        h_in = ws_had_;
    }
    gemm_residual(ws_x_, h_in, L.ssm_out.data, L.ssm_out.type, n_embd_, vdim,
                  n, ws_x2_, L.ssm_out.gs);
}

// Writes the routing statistics as `<model>.krakenexperts.json`, beside the
// model so it travels with it.
//
// The file deliberately stores a *ranking with weights*, not a chosen hot set.
// Which experts deserve VRAM is a function of the card, and the cards this
// runs on differ by 2x (6 GB and 12 GB) while the free host memory is a
// different number again. Baking "the top 8" into the file would be right for
// one machine and wrong for the next one; storing each expert's routing mass
// and its byte cost lets the loader solve for whatever budget it actually has.
// A 300B model needs ~70% coverage to fit a 24 GB host tier, where a 30B needs
// ~10%; the file is the same either way.
// See Backend::scan_accum. A tracking run keeps its counters on the device and
// this is the single read that brings them back -- once per run, instead of one
// blocking drain per layer. No-op when the buffers were never allocated, which
// is every run that is not scanning and every backend that cannot accumulate.
void Engine::scan_dev_fold() {
    if (!be_ || !scan_hits_dev_ || !scan_mass_dev_) return;
    const size_t cells = expert_mass_.size();
    if (cells == 0 || expert_hits_.size() < cells) return;
    std::vector<u32> hits(cells, 0);
    std::vector<f32> mass(cells, 0.0f);
    be_->sync();
    be_->download(hits.data(), scan_hits_dev_, cells * sizeof(u32));
    be_->download(mass.data(), scan_mass_dev_, cells * sizeof(f32));
    for (size_t i = 0; i < cells; i++) {
        expert_hits_[i] += hits[i];
        expert_mass_[i] += static_cast<double>(mass[i]);
    }
    // Cleared so a second fold (write, then merge) cannot count the same run
    // twice. The histograms are cumulative by design, so this is not a reset of
    // what was measured -- the host vectors now hold it.
    be_->fill0(scan_hits_dev_, cells * sizeof(u32));
    be_->fill0(scan_mass_dev_, cells * sizeof(f32));
}

void Engine::write_expert_index(const std::string &path) {
    scan_dev_fold();
    write_expert_index_rows(path, expert_mass_, expert_hits_,
                            expert_scan_tokens_,
                            expert_stub_ ? "routers-only" : "full-forward");
}

void Engine::write_expert_index_rows(const std::string &path,
                                     const std::vector<double> &mass_in,
                                     const std::vector<u64> &hits_in, i64 tokens,
                                     const char *mode) const {
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return;
    const i32 ne = mc.n_expert;
    const size_t cells = static_cast<size_t>(mc.n_layer) *
                         static_cast<size_t>(ne);
    if (mass_in.size() < cells || hits_in.size() < cells) return;
    // A file that measured nothing is not a ranking, so it must not be written
    // -- least of all over a real index. The scan path used to write one anyway
    // when the run died before counting a position (the REAP-320 file at an
    // 8 GiB budget refused an expert in generation and still wrote
    // "0 positions, full-forward"), and the next load trusted it: measured 8.6
    // tok/s against 11.8 for the even order it replaced. `merge_expert_index`
    // already refuses this case; every writer goes through here, so the guard
    // belongs here too.
    if (tokens <= 0) {
        KRK_WARN("expert index: %s not written -- the run observed 0 positions, "
                 "so it has no ranking to record. Any index already there is "
                 "left untouched; re-run --expert-scan once the run completes",
                 path.c_str());
        return;
    }
    const double *mass_all = mass_in.data();
    const u64 *hits_all = hits_in.data();
    const size_t one = model_.max_expert_bytes();
    // Size and mtime travel with the index so a loader can refuse a list
    // written for a different build of the same weights. Matching the Laguna
    // edge0-index fields, which the collections already ship.
    // The size comes from the 64-bit probe, NOT from st_size below. Measured
    // on the 18.42 GiB nemotron file: ::stat succeeds here but its 32-bit
    // st_size reports 0, so every index for a model past 2 GiB recorded
    // "source_size": 0 -- and the loader's own stat failed the same way, so the
    // comparison ran against nothing. The mtime still comes from stat, where it
    // is a time_t and fits.
    unsigned long long src_size = 0, src_mtime = 0;
    {
        u64 sz = 0;
        if (file_size(cfg_.model_path, &sz)) src_size = sz;
        struct stat st;
        if (::stat(cfg_.model_path.c_str(), &st) == 0)
            src_mtime = static_cast<unsigned long long>(st.st_mtime);
    }

    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        KRK_WARN("expert scan: cannot write %s", path.c_str());
        return;
    }
    std::fprintf(f,
                 "{\n  \"version\": 1,\n"
                 "  \"kind\": \"kraken-expert-index\",\n"
                 "  \"mode\": \"%s\",\n"
                 "  \"source_path\": \"%s\",\n"
                 "  \"source_size\": %llu,\n"
                 "  \"source_mtime\": %llu,\n"
                 "  \"arch\": \"%s\",\n"
                 "  \"n_layer\": %d,\n  \"n_expert\": %d,\n"
                 "  \"n_expert_used\": %d,\n"
                 "  \"expert_bytes\": %llu,\n"
                 "  \"positions_scanned\": %lld,\n"
                 "  \"layers\": [\n",
                 mode,
                 cfg_.model_path.c_str(),
                 src_size, src_mtime,
                 mc.arch.c_str(), mc.n_layer, ne, mc.n_expert_used,
                 static_cast<unsigned long long>(one),
                 static_cast<long long>(tokens));
    for (i32 l = 0; l < mc.n_layer; l++) {
        const LayerWeights &L = model_.layers()[static_cast<size_t>(l)];
        if (!L.moe) continue;
        const double *mass = mass_all + static_cast<size_t>(l) * ne;
        const u64 *hits = hits_all + static_cast<size_t>(l) * ne;
        // Highest mass first, so a reader can take a prefix and know it is the
        // hot set without sorting.
        std::vector<i32> ord(static_cast<size_t>(ne));
        for (i32 e = 0; e < ne; e++) ord[static_cast<size_t>(e)] = e;
        std::sort(ord.begin(), ord.end(), [&](i32 a, i32 b) {
            if (mass[a] != mass[b]) return mass[a] > mass[b];
            return a < b; // stable: equal mass ranks the lower id first
        });
        std::fprintf(f, "    {\"layer\": %d, \"experts\": [", l);
        for (i32 i = 0; i < ne; i++) {
            const i32 e = ord[static_cast<size_t>(i)];
            std::fprintf(f,
                         "%s{\"e\": %d, \"mass\": %.9g, \"hits\": %llu}", i ? ", " : "",
                         e, mass[e], static_cast<unsigned long long>(hits[e]));
        }
        std::fprintf(f, "]}%s\n", l + 1 < mc.n_layer ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    KRK_INFO("expert index: wrote %s (%lld positions, %s)", path.c_str(),
             static_cast<long long>(tokens), mode);
}

// ---- the expert index, read back ------------------------------------------
// The file is the tier plan for every later load, so the engine reads it as
// well as writes it. The geometry checks are the loader's, deliberately: a
// second, looser set of them is how a ranking measured on other weights gets
// applied to these.
ExpertIndexRows expert_index_parse(const std::string &text, i32 n_layer,
                                   i32 n_expert, const std::string &arch,
                                   unsigned long long src_size) {
    ExpertIndexRows out;
    if (n_layer <= 0 || n_expert <= 0) return out;
    const size_t cells = static_cast<size_t>(n_layer) *
                         static_cast<size_t>(n_expert);
    JsonValue root;
    std::string err;
    if (!json_parse(text, &root, &err) || !root.is_object()) return out;
    std::string kind;
    root.get_string("kind", &kind);
    if (kind != "kraken-expert-index" || root.get_int("version", 0) != 1)
        return out;
    if (root.get_int("n_layer", -1) != n_layer ||
        root.get_int("n_expert", -1) != n_expert)
        return out;
    std::string file_arch;
    root.get_string("arch", &file_arch);
    if (!file_arch.empty() && file_arch != arch) return out;
    const i64 isz = root.get_int("source_size", -1);
    // A zero on either side means "unknown" rather than "different": the
    // 64-bit probe is what knows the size, and a fixture has no model file.
    if (isz > 0 && src_size > 0 &&
        static_cast<unsigned long long>(isz) != src_size)
        return out;
    // `positions_scanned` 0 (or absent, which reads as 0) means the file
    // records no measurement at all -- a scan that died before its first
    // position. Its per-layer order is then whatever the one partial forward
    // pass left behind, which is not a ranking of the workload and measured
    // *worse* than no ranking at all. Every reader refuses it here, so the
    // warmup, the adapt-merge and the tests all agree.
    out.tokens = root.get_int("positions_scanned", 0);
    if (out.tokens <= 0) return out;
    out.mass.assign(cells, 0.0);
    out.hits.assign(cells, 0);
    const JsonValue *layers = root.find("layers");
    if (!layers || !layers->is_array()) return out;
    for (const JsonValue &L : layers->items()) {
        const i64 l = L.get_int("layer", -1);
        if (l < 0 || l >= n_layer) continue;
        const JsonValue *ex = L.find("experts");
        if (!ex || !ex->is_array()) continue;
        for (const JsonValue &E : ex->items()) {
            const i64 e = E.get_int("e", -1);
            if (e < 0 || e >= n_expert) continue;
            const size_t cell = static_cast<size_t>(l) *
                                    static_cast<size_t>(n_expert) +
                                static_cast<size_t>(e);
            out.mass[cell] = E.get_number("mass", 0.0);
            out.hits[cell] = static_cast<u64>(E.get_int("hits", 0));
        }
    }
    out.valid = true;
    return out;
}

void expert_index_blend(ExpertIndexRows *old, i32 n_expert,
                        const std::vector<double> &run_mass,
                        const std::vector<u64> &run_hits, i64 run_tokens,
                        double retain) {
    if (!old || n_expert <= 0) return;
    const size_t ne = static_cast<size_t>(n_expert);
    const size_t cells = old->mass.size();
    if (cells == 0 || cells % ne != 0 || run_mass.size() != cells) return;
    if (run_hits.size() != cells) return;
    if (retain < 0.0) retain = 0.0;
    if (retain > 1.0) retain = 1.0;
    const i64 old_tokens = old->tokens;
    // The stored number is "summed probability over N positions". A blend of
    // two distributions is scaled to the larger count so the field keeps
    // meaning that, instead of shrinking every run and eventually reading as
    // zero mass.
    const i64 eff = std::max<i64>(std::max<i64>(old_tokens, run_tokens), 1);
    for (size_t base = 0; base < cells; base += ne) {
        double so = 0, sr = 0;
        for (size_t e = 0; e < ne; e++) {
            so += old->mass[base + e];
            sr += run_mass[base + e];
        }
        // A row this run never routed through -- a dense layer, or a prompt
        // that produced no positions there -- keeps what the file had. Blending
        // it toward uniform would decay a real measurement for no reason.
        if (sr <= 0.0) continue;
        for (size_t e = 0; e < ne; e++) {
            const double po = so > 0.0
                                  ? old->mass[base + e] / so
                                  : 1.0 / static_cast<double>(ne);
            const double pr = run_mass[base + e] / sr;
            old->mass[base + e] =
                (retain * po + (1.0 - retain) * pr) * static_cast<double>(eff);
            old->hits[base + e] += run_hits[base + e];
        }
    }
    old->tokens = old_tokens + run_tokens;
}

void Engine::merge_expert_index(const std::string &path, double retain) {
    scan_dev_fold();
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return;
    const size_t cells = static_cast<size_t>(mc.n_layer) *
                         static_cast<size_t>(mc.n_expert);
    if (expert_mass_.size() < cells || expert_hits_.size() < cells) return;
    if (expert_scan_tokens_ <= 0) {
        KRK_INFO("expert adapt: no positions observed -- %s left alone",
                 path.c_str());
        return;
    }
    std::string text;
    {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (f) {
            char buf[1 << 16];
            size_t n = 0;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
                text.append(buf, n);
            std::fclose(f);
        }
    }
    u64 src_size = 0;
    if (!file_size(cfg_.model_path, &src_size)) src_size = 0;
    ExpertIndexRows rows;
    if (text.empty()) {
        // First adapted run: the file it writes is this run's ranking, with
        // the same header a scan writes, so the loader cannot tell them apart
        // -- which is the point.
        rows.mass.assign(cells, 0.0);
        rows.hits.assign(cells, 0);
        KRK_INFO("expert adapt: no index at %s -- writing this run's ranking",
                 path.c_str());
    } else {
        rows = expert_index_parse(text, mc.n_layer, mc.n_expert, mc.arch,
                                  src_size);
        if (!rows.valid) {
            KRK_WARN("expert adapt: %s does not describe this model -- "
                     "refusing to merge, left alone", path.c_str());
            return;
        }
    }
    expert_index_blend(&rows, mc.n_expert, expert_mass_, expert_hits_,
                       expert_scan_tokens_, retain);
    write_expert_index_rows(path, rows.mass, rows.hits, rows.tokens,
                            "scan+adapted");
    KRK_INFO("expert adapt: merged %lld observed positions into %s "
             "(retain %.2f)",
             static_cast<long long>(expert_scan_tokens_), path.c_str(), retain);
}

// True when the ranked warmup should run. Split out from warm_experts()
// because configure_expert_cache has to know whether to leave WARM alone: the
// two phases fill the same tier in different orders and cannot both run.
bool Engine::expert_warmup_wanted() const {
    // A routing scan is a measurement pass, not a deployment, and its experts
    // are skipped in the cheap mode -- warming tiers it will not read is 20 s
    // of cold start spent on nothing.
    if (expert_scan_) return false;
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return false;
    const ExpertCache &ec = model_.experts();
    if (ec.warm_capacity_bytes() == 0 && !ec.warm_as_mapping())
        return false; // no second tier to fill
    bool on = cfg_.expert_warmup > 0;
    if (cfg_.expert_warmup < 0) {
        // Default: ON for every MoE that has a device tier to stage into, and
        // the corpus-vs-budget comparison below only decides the CPU case.
        //
        // That is not the obvious rule, and it is the one the measurements
        // forced. The old rule staged only when the CORPUS exceeded the budget,
        // on the premise that a set which fits is "resident anyway, so staging
        // is pure wall time". Three files say the premise is false, because what
        // staging removes is not eviction but COMPULSORY FIRST TOUCH: a layer's
        // first pass has to read every expert it routes to whatever the budget
        // is, and that read sits inside the prefill.
        //
        //   LFM2.5-24B-A2B   12312 MiB corpus, 11964 MiB earned budget
        //                    (the plan said 12312, so this file read as
        //                    "fits"): 39.7 -> 83.4 tok/s decode,
        //                    2410 -> 111 ms prefill, COLD 25.7% -> 0.
        //   Qwen3-30B-A3B    10008 MiB corpus, 10008 MiB budget -- the exact
        //                    fit the rule called "all resident": 9.7 -> 47.5
        //                    tok/s, COLD 41.8% -> 0.
        //   llada-moe-7b     3720 MiB corpus against ~11.9 GiB of room, i.e.
        //                    3x slack: 19.2 -> 107.4 tok/s, 541 -> 34 ms
        //                    prefill, COLD 36.4% -> 0.
        //
        // The third is the decisive one: slack in the pool does not make the
        // first touch free, so "it fits" was never the right question.
        // `--expert-warmup 0` (or KRK_EXPERT_WARMUP=0) is the escape, and
        // `--expert-warmup-ms` bounds the phase for a corpus too big to stage.
        //
        // The CPU backend keeps the old comparison: it has no device tier to
        // stage against, and the suite's MoE fixtures assert on the counters a
        // staging pass would move. That is a test-scope choice, stated as one.
        const size_t corpus = model_.total_expert_bytes();
        if (!be_ || !be_->has_device_memory() ||
            be_->device_total_bytes() == 0)
            on = corpus > ec.budget_bytes();
        else
            on = true;
        // The one-launch dispatch needs every cell, not just the hot ones: a
        // table cannot name an expert the tiers never loaded, and the first
        // forward touches k experts per layer, not all of them. So a run that
        // asked for the arm warms even when the set fits its budget -- there the
        // ORDER is irrelevant and COMPLETENESS is the whole point. Measured on
        // Qwen3-30B-A3B: without this the publish reports "6144 never loaded"
        // after every retry and the arm stays off on a model that fits.
        if (dev_ptr() && be_ && be_->host_load_supported()) on = true;
    }
    const char *env = std::getenv("KRK_EXPERT_WARMUP");
    if (env && (env[0] == '0' || env[0] == '1')) on = env[0] == '1';
    return on;
}

// Reads the ranking write_expert_index produced and returns it as (layer,
// expert) pairs hottest-first, round-major across layers.
//
// The round-major interleave is the one design decision this function makes.
// The file stores each layer's experts sorted by mass, so a per-layer prefix is
// that layer's hot set -- but mass is comparable WITHIN a layer and every layer
// runs on every token. Sorting globally would spend a tier that holds 80% of
// the routed set on the few layers whose experts happen to carry the most
// probability mass and leave the rest with nothing, which is strictly worse
// than giving every layer its own top 80%. So the rank is read one expert deep
// at a time: round r is every layer's r-th hottest expert.
// ---- the hot ring ----------------------------------------------------------
//
// WARM is pageable host RAM and holds tens of GiB; VRAM holds the few hundred
// MiB to few GiB that routing asks for most often. Without a ring those bytes
// are won and lost by the LRU scan every token, because the routed set is
// larger than the pool: on Tiel-Coder-35B-A3B (256 experts, top-8, 40 layers,
// 22.1 GiB routed) the pool is 12-13 GiB, so the pinned prefix that traffic
// earns ends up being most of the pool and everything else churns in the rest.
// The ring makes the index's ranked prefix a guaranteed hit and hands the
// pool's whole budget back to the experts that still have to earn it.
//
// The size is measured, not planned: it runs after the warmup (so VRAM traffic
// is a fact) and before the pool clamp (so the clamp can subtract it). Half of
// the room beside the floor is the default, because the other half is the
// pool's growth room and a pool with none is the measured pathological arm
// (alloc 30 ms per 2.2 MiB cell, 3.1 tok/s on the same file).
void Engine::pin_hot_experts() {
    if (!be_ || !be_->has_device_memory()) return;
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0 || mc.n_layer <= 0) return;
    ExpertCache &ec = model_.experts();
    if (ec.vram_budget() == 0) return; // no device tier: nothing to protect
    // OFF by default, and that is a correctness decision, not a tuning one.
    // With a ring the pool sits exactly at its budget and every promotion
    // evicts, which drives the pooled allocator's async order hard enough to
    // hit a real defect: a 1024-token decode on Tiel-Coder-35B-A3B dies at
    // token ~48 with `HIP call failed (hipErrorLaunchFailure):
    // hipMemcpyAsync(d + off, paged_buf_[slot], ...)` -- with a 2174 MiB ring
    // and with a 512 MiB one, clean with the ring off, clean when the engine is
    // serialized (AMD_SERIALIZE_KERNEL=3, which also hides it), and still
    // failing with KRK_PROMOTE_WAIT=host, so it is not the publish wait. The
    // ring's prefill win (+10.8%) is not worth a crash, so it is opt-in until
    // the pool's release/reuse path is fixed: KRK_EXPERT_PIN_MB=<n> (or
    // KRK_EXPERT_PIN=1 for the default size) turns it on.
    const char *on = std::getenv("KRK_EXPERT_PIN");
    const char *mb = std::getenv("KRK_EXPERT_PIN_MB");
    const bool forced = on && on[0] == '1' && on[1] == '\0';
    const bool sized = mb && std::atoll(mb) > 0;
    if (!forced && !sized) {
        KRK_INFO("expert cache: hot ring off (opt-in: KRK_EXPERT_PIN_MB=<n>, "
                 "or KRK_EXPERT_PIN=1) -- it measured +10.8%% prefill but "
                 "crashes a long decode in the pool's async reuse path");
        return;
    }
    const size_t total = be_->device_total_bytes();
    if (total == 0) return;
    // A share of the CARD, not of whatever is free this instant: the warmup has
    // just filled VRAM to its reserve, so "free beside the floor" is routinely
    // zero by the time this runs (measured: the same command admitted a 5325 MiB
    // ring on one run and skipped it on the next, depending on whether the
    // warmup reached its VRAM stage or its time ceiling). The ring's target is
    // a property of the machine and the pool, and admit_hot() makes the room by
    // evicting unpinned slots -- those experts are in WARM, so the move is a
    // promotion, never a file read.
    const size_t floor = krk::vram_floor_bytes(total);
    const size_t usable = total > floor ? total - floor : 0;
    // Measured on Tiel-Coder-35B-A3B (256 experts, 40 layers, routed 22.1 GiB,
    // 7682 MiB of device expert memory, --bench -n 64, interleaved pairs):
    // a 4350 MiB ring (1/3 of the usable card) bought +5.5% prefill and COST
    // 7.8% decode, 5/5 pairs -- the pin displaced pool slots the LFU order
    // ranked better. A 2048 MiB ring measured +5.5% prefill and -0.6% decode
    // (inside noise), so 1/6 is the size the evidence supports; the pool keeps
    // the rest, which is what the decode side lives on.
    size_t want = usable / 6;
    // Never more than the pool it is meant to sit beside: a ring larger than
    // the traffic's own budget is pins the run has no evidence for.
    if (want > ec.vram_budget()) want = ec.vram_budget();
    if (sized) {
        const long long v = std::atoll(mb);
        if (v > 0) {
            const size_t req = static_cast<size_t>(v) * 1048576ULL;
            want = req < usable ? req : usable;
        }
    }
    std::vector<std::pair<i32, i32>> order = load_expert_index();
    if (order.empty()) {
        // An even order is not a hot set: pinning expert 0 of every layer is
        // the same guess the unranked warmup makes, and a wrong pin cannot be
        // evicted. The ring is for a measured ranking or nothing.
        KRK_INFO("expert cache: hot ring skipped: no usable "
                 "<model>.krakenexperts.json -- run --expert-scan to rank the "
                 "experts first");
        return;
    }
    std::vector<const ExpertSource *> srcs(static_cast<size_t>(mc.n_layer),
                                           nullptr);
    i32 moe_layers = 0;
    for (i32 l = 0; l < mc.n_layer; l++) {
        const LayerWeights &L = model_.layers()[static_cast<size_t>(l)];
        if (!L.experts.present()) continue;
        srcs[static_cast<size_t>(l)] = &L.experts;
        moe_layers++;
    }
    if (moe_layers == 0) return;
    ec.set_pin_budget(want);
    const size_t got = ec.admit_hot(srcs.data(), mc.n_layer, order, 0);
    if (got == 0) {
        ec.set_pin_budget(0);
        KRK_INFO("expert cache: hot ring admitted nothing of a %.0f MiB ring "
                 "-- leaving every pin on the pool's budget",
                 static_cast<f64>(want) / 1048576.0);
    }
}

// ---- the VRAM ledger ----------------------------------------------------
// total = weights + kv + experts + workspaces + free + unaccounted, every term
// measured. `unaccounted` is the check on the ledger: a buffer taken outside
// init's `alloc` lambda (or held by the driver and other processes) lands there
// instead of being folded into `free`, and it is the only number that can say
// so. It replaced an INFERENCE -- "scratch is ~3.0 GiB" by subtracting four
// numbers from the card -- that was wrong by more than 30x: the measured
// workspace total on this build is 88 MiB at ctx 4096.
void Engine::log_vram_ledger(const char *when) {
    if (!be_ || !be_->has_device_memory() ||
        be_->device_total_bytes() == 0)
        return;
    const size_t weights = model_.upload_bytes();
    const Backend::ExpertPoolStats ps = be_->expert_pool_stats();
    const size_t ring = model_.experts().pin_used_bytes();
    const size_t total = be_->device_total_bytes();
    const size_t free_now = be_->device_free_bytes();
    // ps.budget is a PLAN, and with no expert pool configured the backend
    // leaves it at its sentinel -- printed verbatim that read "budget
    // 17592186044416" (2^44) on a dense model, which is a number no reader can
    // use. Anything larger than the card is "no pool".
    const size_t pool_budget = ps.budget <= total ? ps.budget : 0;
    const size_t known = weights + kv_bytes_ + ps.committed + ws_bytes_;
    const size_t used = known < total ? total - known : 0;
    const size_t unaccounted = used > free_now ? used - free_now : 0;
    // The allocator's own total, which is the only number here that does not
    // come from a device-wide probe. `live_alloc` above the ledger's `known`
    // means this engine is holding bytes the ledger does not name -- a buffer
    // taken outside init's `alloc` lambda, or a pool cell counted under the
    // wrong owner -- and live_alloc BELOW it means something is counted twice.
    // Both used to land silently in `unaccounted`, which read as "the driver is
    // holding 7 GiB" when the truth was an allocation nobody tracked.
    const size_t live_alloc = be_->device_live_bytes();
    KRK_INFO("vram ledger (%s): %.0f MiB total = weights %.0f + kv %.0f + "
             "experts %.0f (pool %.0f + ring %.0f; live %.0f, reusable %.0f, "
             "budget %.0f) + workspaces %.0f + free %.0f + unaccounted %.0f "
             "[allocator live %.0f in %.0f buffers, largest %.0f]",
             when, static_cast<f64>(total) / 1048576.0,
             static_cast<f64>(weights) / 1048576.0,
             static_cast<f64>(kv_bytes_) / 1048576.0,
             static_cast<f64>(ps.committed) / 1048576.0,
             static_cast<f64>(ps.committed > ring ? ps.committed - ring : 0) /
                 1048576.0,
             static_cast<f64>(ring) / 1048576.0,
             static_cast<f64>(ps.live) / 1048576.0,
             static_cast<f64>(ps.reusable) / 1048576.0,
             static_cast<f64>(pool_budget) / 1048576.0,
             static_cast<f64>(ws_bytes_) / 1048576.0,
             static_cast<f64>(free_now) / 1048576.0,
             static_cast<f64>(unaccounted) / 1048576.0,
             static_cast<f64>(live_alloc) / 1048576.0,
             static_cast<f64>(be_->device_alloc_count()),
             static_cast<f64>(be_->device_largest_alloc()) / 1048576.0);
}

// ---- the device dispatch table ---------------------------------------------
//
// Published once, at the end of init, and only when the cache can keep the
// promise the table makes: every (layer, expert) of the routed set is in VRAM,
// so no promotion or eviction can rewrite a device pointer while a kernel is
// reading it. Anything short of that publishes nothing and says so -- the arm
// then never engages, which is a slower run, not a different model.
void Engine::publish_expert_table() {
    dev_table_ = nullptr;
    if (!dev_ptr()) return;
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0 || mc.n_layer <= 0) return;
    if (!be_->moe_dispatch_supported()) {
        KRK_INFO("device dispatch: off -- this backend has no table dispatch");
        return;
    }
    i32 filled = 0;
    std::string why;
    void *t = model_.experts().publish_dispatch_table(mc.n_layer, mc.n_expert,
                                                      &filled, &why);
    if (!t) {
        // Once, not once per attempt: this is retried from moe_ffn while the
        // cache fills, and a line per layer would drown the log.
        if (!table_why_logged_) {
            table_why_logged_ = true;
            KRK_INFO("device dispatch: not yet -- %s (retrying while the tiers "
                     "fill)",
                     why.c_str());
        }
        return;
    }
    dev_table_ = t;
    const i32 on_host = model_.experts().dispatch_table_host_cells();
    KRK_INFO("device dispatch: table published (%d entries x 3 matrices, %.1f "
             "MiB): %d cell(s) in VRAM, %d read in place from pinned host "
             "memory, so KRK_DEV_PTR=1 routes without a host pointer per expert%s",
             filled, static_cast<f64>(filled) *
                          static_cast<f64>(sizeof(MoeDispatchEntry)) / 1048576.0,
             filled - on_host, on_host,
             on_host > 0 ? " -- the overflow of a set that does not fit the card "
                           "is read where it lives (24.3 GiB/s measured)"
                         : "");
}

// ---- re-clamp: the budget follows the VRAM that is actually free -------------
//
// The init-time clamp is one measurement, taken before the run. Two things then
// take VRAM away from it: the KV tier promotes COLD->HOT rows as the context
// grows (the report prints `kv tiered: promoted WARM->HOT n`), and the workspaces
// a long prefill touches are not the ones a decode step touches. A budget named
// from the init measurement therefore describes a card that no longer exists --
// the pathological arm AGENTS.md measured is exactly this, at 30 ms per pool
// cell and 3.1 tok/s. This is the same arithmetic, re-run against a live
// reading, and it only ever lowers: growth has to go through the ring, which
// names what it is for.
void Engine::reclamp_expert_budget(const char *why) {
    if (!be_ || !be_->has_device_memory()) return;
    // KRK_RECLAMP=0 is the A/B arm. It exists because this clamp is the one
    // place a run's residency can be cut on a measurement taken while the pool
    // itself was growing, and a budget cut cannot be undone (`reclamp_` only
    // lowers): a wrong verdict here costs the whole run, and it cannot be told
    // apart from a right one without an arm that turns it off.
    {
        static const bool on = [] {
            const char *e = std::getenv("KRK_RECLAMP");
            return !(e && e[0] == '0');
        }();
        if (!on) return;
    }
    ExpertCache &ec = model_.experts();
    if (ec.vram_budget() == 0 || ec.resident_bytes() == 0) return;
    const size_t total = be_->device_total_bytes();
    if (total == 0) return;
    // Shrink only on a real shrink: a driver query's value moves by a few MiB
    // on its own, and a budget that chases it re-plans the cache every step.
    const size_t floor = krk::vram_floor_bytes(total);
    const size_t free_now = be_->device_free_bytes();
    const size_t before = ec.vram_budget();
    {
        const char *e = std::getenv("KRK_VRAM_TRACE");
        if (e && *e && e[0] != '0') {
            KRK_INFO("vram trace (%s): free %.0f MiB, floor %.0f MiB, budget "
                     "%.0f MiB, resident %.0f MiB, unallocated allowance %.0f MiB",
                     why ? why : "a forward",
                     static_cast<f64>(free_now) / 1048576.0,
                     static_cast<f64>(floor) / 1048576.0,
                     static_cast<f64>(before) / 1048576.0,
                     static_cast<f64>(ec.resident_bytes()) / 1048576.0,
                     static_cast<f64>(before > ec.resident_bytes()
                                          ? before - ec.resident_bytes()
                                          : 0) / 1048576.0);
        }
    }
    const size_t after = ec.reclamp_vram_budget(free_now, floor);
    if (after >= before) return; // the arithmetic lives in ExpertCache, so the
                                 // suite can drive it without a device
    KRK_INFO("expert cache: re-clamped %.0f -> %.0f MiB at %s (%.0f MiB free "
             "beside a %.0f MiB floor) -- the budget planned at init names bytes "
             "the KV tier has since taken",
             static_cast<f64>(before) / 1048576.0,
             static_cast<f64>(after) / 1048576.0, why ? why : "a forward",
             static_cast<f64>(free_now) / 1048576.0,
             static_cast<f64>(floor) / 1048576.0);
}

std::vector<std::pair<i32, i32>> Engine::load_expert_index() const {
    std::vector<std::pair<i32, i32>> out;
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0 || mc.n_layer <= 0) return out;
    const std::string path = cfg_.model_path + ".krakenexperts.json";
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return out; // no ranking: warm_experts falls back to an even order
    std::string text;
    {
        char buf[1 << 16];
        size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    }
    std::fclose(f);
    JsonValue root;
    std::string err;
    if (!json_parse(text, &root, &err) || !root.is_object()) {
        KRK_WARN("expert warmup: %s is not readable JSON (%s) -- ignoring it",
                 path.c_str(), err.c_str());
        return out;
    }
    std::string kind;
    root.get_string("kind", &kind);
    if (kind != "kraken-expert-index" || root.get_int("version", 0) != 1 ||
        root.get_int("n_layer", -1) != mc.n_layer ||
        root.get_int("n_expert", -1) != mc.n_expert) {
        KRK_WARN("expert warmup: %s does not describe this model (kind '%s', "
                 "version %lld, %lld layers x %lld experts; expected %d x %d) "
                 "-- ignoring it",
                 path.c_str(), kind.c_str(),
                 static_cast<long long>(root.get_int("version", 0)),
                 static_cast<long long>(root.get_int("n_layer", -1)),
                 static_cast<long long>(root.get_int("n_expert", -1)),
                 mc.n_layer, mc.n_expert);
        return out;
    }
    // The ranking has to belong to these weights, and the claim is now
    // REQUIRED rather than checked-when-present. An index that does not say
    // which file it was measured on is an index nobody can place, and "could
    // not verify" has to mean "refuse": the failure this guards against is
    // precisely the one that looks fine, because a ranking from another model's
    // routing is a perfectly well-formed file that orders the tiers by a
    // distribution these weights do not have.
    std::string arch;
    root.get_string("arch", &arch);
    if (!arch.empty() && arch != mc.arch) {
        KRK_WARN("expert warmup: %s was measured on '%s' and this model is "
                 "'%s' -- ignoring it",
                 path.c_str(), arch.c_str(), mc.arch.c_str());
        return out;
    }
    // A file that recorded no positions has no ranking to apply, and applying
    // one anyway is not a benign default. Measured on the REAP-320 file at an
    // 8 GiB budget, a 0-position index (written when a scan died mid-generation)
    // gave 8.6 tok/s against 11.8 for the even order it replaced: its partial
    // order changes which cells are resident, the pool then stops recycling
    // slots (reuse 2876 -> 119) and every decode miss pays a fresh allocation.
    // Fall back to the even order, loudly -- the file is the thing to fix.
    const i64 measured = root.get_int("positions_scanned", 0);
    if (measured <= 0) {
        KRK_WARN("expert warmup: %s records %lld scanned positions, so it "
                 "carries no ranking -- filling the tiers evenly instead. A "
                 "scan only writes this when it measured nothing, so re-run "
                 "--expert-scan and let it finish",
                 path.c_str(), static_cast<long long>(measured));
        return out;
    }
    const i64 isz = root.get_int("source_size", -1);
    u64 model_size = 0;
    if (isz <= 0 || !file_size(cfg_.model_path, &model_size)) {
        KRK_WARN("expert warmup: %s does not record the size of the file it was "
                 "measured on (source_size %lld) -- ignoring it (an index whose "
                 "provenance cannot be checked is not one to warm the tiers "
                 "from; re-run --expert-scan to write it)",
                 path.c_str(), static_cast<long long>(isz));
        return out;
    }
    if (static_cast<u64>(isz) != model_size) {
        // Bytes as well as GiB: two 18.4 GiB files that differ by one byte are
        // the same two-decimal number, and a diagnostic that prints the same
        // figure twice reads as a bug in the check rather than a mismatch.
        KRK_WARN("expert warmup: %s was measured on a %.2f GiB file and this "
                 "one is %.2f GiB (%llu vs %llu bytes) -- ignoring it (a ranking "
                 "from other weights orders the tiers by another model's "
                 "routing)",
                 path.c_str(),
                 static_cast<f64>(isz) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<f64>(model_size) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<unsigned long long>(isz),
                 static_cast<unsigned long long>(model_size));
        return out;
    }
    std::string mode;
    root.get_string("mode", &mode);
    if (mode == "routers-only")
        KRK_WARN("expert warmup: %s was measured with the experts stubbed "
                 "(--expert-stub); the ranking is an approximation",
                 path.c_str());

    const JsonValue *layers = root.find("layers");
    if (!layers || !layers->is_array()) return out;
    std::vector<std::vector<i32>> per_layer(static_cast<size_t>(mc.n_layer));
    // This loader reads the ids IN FILE ORDER and never sorts: the order the
    // scan wrote IS the payload. A mass that is present and not a number is the
    // specific signature of a scan whose
    // forward pass was broken, because the mass accumulates the router's
    // softmax and a NaN anywhere upstream makes every expert of that layer NaN.
    // Measured here: the corpus scan of NVIDIA-Nemotron-3.5-Lightning-30B-A3B
    // NVFP4 wrote -nan(ind) for all 128 experts of 16 layers. That file parsed,
    // claimed mode "full-forward", and its per-layer order was fiction.
    bool bad_mass = false;
    i64 bad_layer = -1;
    for (const JsonValue &lv : layers->items()) {
        if (!lv.is_object()) continue;
        const i64 l = lv.get_int("layer", -1);
        if (l < 0 || l >= mc.n_layer) continue;
        const JsonValue *ex = lv.find("experts");
        if (!ex || !ex->is_array()) continue;
        for (const JsonValue &ev : ex->items()) {
            const i64 e = ev.get_int("e", -1);
            // A mass that is PRESENT has to be finite and non-negative. A mass
            // that is absent is not an error: this loader consumes the ids in
            // file order and never sorts, so the order the author wrote is the
            // ranking whether or not it came with its justification. Demanding
            // the field rejected hand-written indexes that are perfectly
            // well-defined, which tests/test_kraken.cpp caught.
            const JsonValue *mv = ev.find("mass");
            if (mv) {
                const f64 m = ev.get_number("mass", -1.0);
                if (!(m == m) || m < 0.0) {
                    if (!bad_mass) bad_layer = l;
                    bad_mass = true;
                    continue;
                }
            }
            if (e >= 0 && e < mc.n_expert)
                per_layer[static_cast<size_t>(l)].push_back(static_cast<i32>(e));
        }
    }
    if (bad_mass) {
        KRK_WARN("expert warmup: %s carries routing mass that is not a finite "
                 "non-negative number (first at layer %lld) -- ignoring it. The "
                 "scan that wrote it measured a broken forward pass, so its "
                 "per-layer order says nothing about these weights; fix the scan "
                 "and re-run --expert-scan",
                 path.c_str(), static_cast<long long>(bad_layer));
        return out;
    }
    size_t depth = 0;
    for (const auto &v : per_layer) depth = std::max(depth, v.size());
    if (depth == 0) return out;
    out.reserve(depth * static_cast<size_t>(mc.n_layer));
    for (size_t r = 0; r < depth; r++)
        for (i32 l = 0; l < mc.n_layer; l++) {
            const std::vector<i32> &v = per_layer[static_cast<size_t>(l)];
            if (r < v.size()) out.emplace_back(l, v[r]);
        }
    KRK_INFO("expert warmup: using %s (%lld positions, %s, %zu ranked pairs)",
             path.c_str(), static_cast<long long>(root.get_int("positions_scanned", -1)),
             mode.empty() ? "mode unknown" : mode.c_str(), out.size());
    return out;
}

// Fills WARM and then VRAM in the ranking's order, under a wall-time ceiling.
//
// The order is the whole point. Read-through already admits every expert a run
// touches, so a tier bigger than the routed set does not need this at all. What
// it fixes is the tier that is SMALLER: without a ranking the first tokens
// decide what stays, and the first tokens are a prompt, not a distribution over
// the conversation. Staging hottest-first puts what routing actually wants into
// the fast tiers before the prompt is read, and promoting the top of that order
// means the first decode steps hit VRAM instead of paying a promotion apiece.
//
// Nothing here is required for correctness: acquire() still loads whatever this
// did not warm, and every expert it stages is also in WARM, which is where a
// VRAM eviction falls back to anyway. It is a startup cost, bounded in
// milliseconds on purpose -- the hottest experts are staged first, so being cut
// short costs coverage, never the top of the list.
void Engine::warm_experts() {
    if (!be_) return;
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return;
    ExpertCache &ec = model_.experts();
    const size_t one = model_.max_expert_bytes();
    // Under the mapping policy the phase has a product even with a 0-byte WARM
    // tier: it faults the model's pages in, which is what the acquire path then
    // reads instead of a promoted copy.
    if (one == 0 || (ec.warm_capacity_bytes() == 0 && !ec.warm_as_mapping()))
        return;

    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    auto elapsed_ms = [&t0]() {
        return std::chrono::duration<f64, std::milli>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    };
    const i64 ceiling = cfg_.expert_warmup_ms;
    // A run that asked for the one-launch dispatch needs the tier COMPLETE, not
    // merely hot: the table names every cell, so an expert the warmup did not
    // reach is a cell the kernel cannot be given, and the arm stays off for the
    // whole run. The ceiling exists to bound a speculative warmup; this one is
    // load-bearing, so the ceiling does not apply to it. It is still bounded by
    // the tier and by the host reserve, which is what stops it running away.
    const bool complete_set = dev_ptr() && be_->host_load_supported() &&
                              model_.total_expert_bytes() > 0;
    if (complete_set)
        KRK_INFO("expert warmup: the one-launch dispatch asked for every expert "
                 "in the tier, so this phase runs to the set (%.1f GiB) instead "
                 "of the %lld ms ceiling -- a table cannot name a cell the "
                 "warmup never filled",
                 static_cast<f64>(model_.total_expert_bytes()) /
                     (1024.0 * 1024.0 * 1024.0),
                 static_cast<long long>(ceiling));
    auto out_of_time = [&]() {
        if (complete_set) return false;
        return ceiling > 0 && elapsed_ms() >= static_cast<f64>(ceiling);
    };

    std::vector<std::pair<i32, i32>> rank = load_expert_index();
    const bool ranked = !rank.empty();
    if (!ranked) {
        // No prior measurement: every layer's expert 0, then expert 1, and so
        // on -- the same even order the eager sweep uses. It is the best a run
        // can do without knowing which experts are hot, and it still converts
        // compulsory file reads on the first tokens into later promotions.
        for (i32 e = 0; e < mc.n_expert; e++)
            for (i32 l = 0; l < mc.n_layer; l++)
                if (model_.layers()[static_cast<size_t>(l)].experts.present())
                    rank.emplace_back(l, e);
        KRK_INFO("expert warmup: no %s.krakenexperts.json -- filling the tiers "
                 "evenly; run --expert-scan to rank them",
                 cfg_.model_path.c_str());
    }
    // Per-layer view of the rank, so a round can be issued as ONE batched read
    // per layer instead of one read per expert: the file's queue depth is what
    // makes the difference (593 MiB/s at depth 1 against 3400 sequential).
    std::vector<std::vector<i32>> ids(static_cast<size_t>(mc.n_layer));
    for (const auto &pr : rank)
        ids[static_cast<size_t>(pr.first)].push_back(pr.second);
    size_t depth = 0;
    i32 moe_layers = 0;
    for (const auto &v : ids) {
        if (v.size() > depth) depth = v.size();
        if (!v.empty()) moe_layers++;
    }
    if (depth == 0 || moe_layers == 0) return;

    // ---- WARM: the ranked set, hottest-first, one batched read per layer ----
    const size_t warm_cap = ec.warm_capacity_bytes();
    size_t staged = 0;
    size_t from = 0;
    i32 rounds = 0;
    // NOT bounded by the device tier's size, and that was tried: see the note on
    // the Tiel curve in docs/PERF-WARMUP.md. Bounding the staged bytes by the
    // pool room (`min(plan, room)`, or half of it) made the file WORSE than
    // staging the lot -- 12.3-12.9 tok/s against 10.4 for the full fill and
    // 17.0-19.2 across the small ceilings -- because the lever is not how much
    // is staged but how the pool then treats WARM as an eviction target. The
    // numbers live in the doc so the next attempt starts from them.
    // The host twin of the VRAM reserve, and for the same reason: the ram policy
    // above plans a 20% headroom once, before the KV tier, the workspaces and
    // the model's own buffers exist, and staging is what spends it. Ask the
    // machine every round instead of trusting that plan.
    const size_t host_reserve = krk::host_reserve_bytes(be_->host_total_bytes());
    bool hit_host_reserve = false;
    while (from < depth) {
        if (out_of_time()) break;
        if (!krk::host_staging_fits(0, be_->host_available_bytes(), host_reserve)) {
            hit_host_reserve = true;
            break;
        }
        const size_t used = ec.warm_used_bytes();
        const size_t left = warm_cap > used ? (warm_cap - used) / one : 0;
        // A full tier ends the phase; under the mapping policy the tier is a
        // page cache the cache does not own, so "full" is not a thing it can
        // observe -- the ceiling, the depth and the depth of the rank are.
        if (left == 0 && !ec.warm_as_mapping()) break; // tier full
        // Round width: what the remaining tier holds across every layer at
        // once. Without the division a tier that filled mid-round would leave
        // the layers it had not reached yet with nothing, which is the exact
        // failure this phase exists to avoid.
        size_t width = left / static_cast<size_t>(moe_layers);
        if (width == 0) width = 1;
        if (width > depth - from) width = depth - from;
        const size_t before = staged;
        for (i32 l = 0; l < mc.n_layer; l++) {
            if (out_of_time()) break;
            const std::vector<i32> &lst = ids[static_cast<size_t>(l)];
            if (from >= lst.size()) continue;
            const LayerWeights &L = model_.layers()[static_cast<size_t>(l)];
            if (!L.experts.present()) continue;
            const size_t n = std::min(width, lst.size() - from);
            // prefetch_layer reserves each unique expert against the live host
            // headroom as it allocates the batch, including bytes already
            // reserved by earlier jobs. Probe here only to stop subsequent
            // layers after a previous layer used the remaining safe headroom;
            // reserving n*per here would overestimate existing WARM hits and
            // duplicate the batch's own exact accounting.
            if (!krk::host_staging_fits(0, be_->host_available_bytes(), host_reserve)) {
                hit_host_reserve = true;
                break;
            }
            staged += ec.prefetch_layer(L.experts, l, lst.data() + from,
                                        static_cast<int>(n));
        }
        if (hit_host_reserve) break;
        from += width;
        rounds++;
        // Nothing fitted this round even though the tier reports room:
        // per-expert sizes differ, so the tier can be full while the byte
        // arithmetic says otherwise. Stop rather than spinning over rounds that
        // cannot stage anything.
        if (staged == before) break;
    }

    // ---- VRAM: the top of the same order ----------------------------------
    // Only experts already in WARM are promoted. Promoting a cold one would
    // read the file here and then leave a HOT slot whose eviction has nowhere
    // to fall back to; keeping HOT a subset of WARM is what makes every VRAM
    // eviction free.
    size_t promoted = 0;
    size_t promoted_bytes = 0;
    // Stop promoting while this much VRAM is still free. Checked per round,
    // against the driver's own count: the tier's budget came from a plan made
    // before the workspaces and the KV tier existed, so only the driver can
    // say what is actually left. A partly warm tier is a slow run; a card with
    // nothing free is a machine that has to be recovered by hand.
    // The floor, not the reserve: a fill that stops at the reserve leaves the
    // card one allocation away from full, and on a 15.9 GiB card with a 22 GiB
    // routed set that is where the desktop starts to stall.
    const size_t vram_reserve = krk::vram_floor_bytes(be_->device_total_bytes());
    bool hit_reserve = false;
    // Stopped at the seed fraction rather than at the card: that is a choice,
    // not a limit, and it has to show in the line or a partial seed and a full
    // one read the same.
    bool capped = false;
    // Does the prefilled prefix have anywhere to live? A prefill is worth its
    // cost only when the pool can hold (essentially) the WHOLE routed set: then
    // every expert the run routes to is one acquire away and the prefix is what
    // traffic would have admitted anyway, just earlier.
    //
    // When the corpus is a multiple of the pool the prefix is a DIFFERENT set
    // from the one the run keeps asking for. It spends half the pool on slots
    // that then evict, against an LFU that was doing better without it.
    // Measured on Tiel-Coder-35B-A3B (22604 MiB corpus, 10498 MiB pool), 256
    // greedy tokens, paired: 4757 prefilled experts left 65.9% HOT hits and
    // 10.6 tok/s, against 87.5% HOT and 17.2 tok/s with the prefill skipped --
    // same pool, same WARM tier, same budget, and 3.2% WARM hits either way.
    // The prefix was charged with 10500 MiB of a 10498 MiB pool, i.e. it is not
    // a head start inside the working set, it IS the whole pool.
    //
    // The 5% slack is LFM2.5-24B-A2B's measured shape: a 12312 MiB corpus
    // against a 11964 MiB budget (2.9% over, once the KV tier and workspaces
    // exist), which is "the whole set fits" in every sense this prefill cares
    // about -- and there it is worth 71.1 tok/s against 17.8.
    const size_t corpus_bytes = model_.total_expert_bytes();
    const size_t pool_bytes = ec.vram_budget();
    bool prefill = pool_bytes > 0 && corpus_bytes <= pool_bytes + pool_bytes / 20;
    // How much of the pool the seed may fill, when the set does not fit. There
    // are two measured optima and they disagree because the seed's value decays
    // with the reply: HALF the pool while the reply is short (REAP-320, 4.1x
    // corpus/pool, 24 tokens: 41.7 tok/s prefill with decode equal to the best
    // arm, against 27.0 unseeded and 31.9 seeded whole), and NOTHING once it is
    // long (Tiel-Coder-35B-A3B, 2.17x: 50% wins to 64 tokens, ties at 96, loses
    // 19% at 128 and 71% at 256 as the prompt's set decays and the LFU would
    // otherwise have kept learning).
    // `--max-tokens` is the only expectation the engine has at init, so it is
    // what this reads; 0 or negative means unbounded, i.e. a long reply.
    int seed_pct = 100;
    if (pool_bytes > 0 && corpus_bytes > pool_bytes + pool_bytes / 20) {
        const i64 want = cfg_.expected_reply_tokens;
        seed_pct = (want > 0 && want <= 64) ? 50 : 0;
        prefill = seed_pct > 0;
    }
    // The force arm is a whole-pool seed, which is what it has always meant.
    if (const char *e = std::getenv("KRK_WARM_PREFILL")) {
        prefill = e[0] != '0' && e[0] != '\0';
        if (prefill && seed_pct == 0) seed_pct = 100;
    }
    // An explicit fraction outranks both, and turns the seed on by itself.
    if (const char *e = std::getenv("KRK_VRAM_SEED_PCT")) {
        const int v = std::atoi(e);
        if (v >= 0 && v <= 100) {
            seed_pct = v;
            prefill = v > 0 && pool_bytes > 0;
        } else {
            KRK_WARN("KRK_VRAM_SEED_PCT=%s is not 0..100 -- ignored", e);
        }
    }
    const size_t seed_cap = pool_bytes * static_cast<size_t>(seed_pct) / 100;
    if (!prefill)
        KRK_INFO("expert warmup: the routed set is %.0f MiB against a %.0f MiB "
                 "device pool, so the ranked prefix is not preloaded into VRAM "
                 "-- at that ratio it competes with the experts the LFU is "
                 "still earning (KRK_WARM_PREFILL=1 forces it)",
                 static_cast<f64>(corpus_bytes) / 1048576.0,
                 static_cast<f64>(pool_bytes) / 1048576.0);
    if (prefill && be_->caps().vram_free > 0) {
        for (size_t r = 0; r < depth; r++) {
            if (out_of_time()) break;
            if (be_->device_free_bytes() <= vram_reserve) {
                hit_reserve = true;
                break;
            }
            bool room = false;
            for (i32 l = 0; l < mc.n_layer; l++) {
                const std::vector<i32> &lst = ids[static_cast<size_t>(l)];
                if (r >= lst.size()) continue;
                const LayerWeights &L = model_.layers()[static_cast<size_t>(l)];
                if (!L.experts.present()) continue;
                if (ec.full_for(L.experts)) continue;
                room = true;
                const i32 e = lst[r];
                // Under the mapping policy nothing is ever in host RAM, so the
                // WARM half cannot have staged the expert the VRAM half is
                // about to ask for -- acquire() reads the mapping instead.
                if (!ec.warm_as_mapping() && !ec.in_host(l, e)) continue;
                const size_t had = ec.resident_slots();
                if (ec.acquire(L.experts, l, e) && ec.resident_slots() > had) {
                    promoted++;
                    promoted_bytes += L.experts.expert_bytes();
                    if (promoted_bytes >= seed_cap) {
                        capped = true;
                        break;
                    }
                }
            }
            if (capped) break;
            if (!room) break; // every layer's device tier is full
        }
    }
    be_->sync();

    const f64 ms = elapsed_ms();
    const bool cut = out_of_time();
    std::string why;
    if (cut) why = " (stopped at the --expert-warmup-ms ceiling)";
    else if (hit_reserve) {
        char b[112];
        // The total is in GiB, as the headroom line states it. Printing a GiB
        // figure as MiB read "16000 MiB kept free of 16 MiB total" on a 15.92
        // GiB card -- which says the card is nearly empty, the opposite of the
        // condition the guard stopped on.
        std::snprintf(b, sizeof(b),
                      " (stopped at the VRAM reserve: %.0f MiB kept free of "
                      "%.2f GiB total)",
                      static_cast<f64>(vram_reserve) / 1048576.0,
                      static_cast<f64>(be_->device_total_bytes()) / 1073741824.0);
        why = b;
    } else if (hit_host_reserve) {
        char b[112];
        std::snprintf(b, sizeof(b),
                      " (stopped at the host RAM reserve: %.0f MiB kept free)",
                      static_cast<f64>(host_reserve) / 1048576.0);
        why = b;
    }
    // The split matters: this phase is the only part of a load that is bounded
    // by the disk, and "how much of it is the read" is the difference between
    // tuning the reader pool and tuning the bookkeeping around it. Measured
    // wrong once by inference (8 readers and 1 thread gave the same wall time,
    // so the read was assumed not to be the cost); a printed number ends that
    // argument instead of deciding it by feel.
    // `read_ms`, not `prefill_read_ms`: the phase-split counter is only fed
    // while the host clock says "prefill", and a warmup runs before either phase
    // is declared -- reading it there reports 0 ms for a phase that exists to
    // read, which is how this line first came out as "0 ms = 0.0 GB/s".
    const f64 read_ms = ec.read_ms();
    const f64 read_gbs = read_ms > 0.0
                             ? static_cast<f64>(staged) / 1073741824.0 /
                                   (read_ms / 1000.0)
                             : 0.0;
    // Under the mapping policy the phase has no tier to describe: its whole
    // product is that the model file's pages are resident, which the acquire
    // path then reads once instead of a second copy. Saying "WARM 0 MiB of X"
    // there would read as a phase that failed.
    char tier_note[256];
    if (ec.warm_as_mapping())
        std::snprintf(tier_note, sizeof(tier_note),
                      "MAPPING (page cache: %.0f MiB read in %.0f ms = %.1f "
                      "GB/s, nothing retained)",
                      static_cast<f64>(staged) / 1048576.0, read_ms, read_gbs);
    else
        std::snprintf(tier_note, sizeof(tier_note),
                      "WARM %.0f MiB of %.0f MiB in %zu slots (%.0f MiB read in "
                      "%.0f ms = %.1f GB/s; book %.0f ms + room %.0f ms, alloc "
                      "%.0f ms of which %.0f overlapped the read, commit %.0f "
                      "ms)",
                      static_cast<f64>(ec.warm_used_bytes()) / 1048576.0,
                      static_cast<f64>(warm_cap) / 1048576.0, ec.warm_slots(),
                      static_cast<f64>(staged) / 1048576.0, read_ms, read_gbs,
                      ec.prep_ms(), ec.room_ms(), ec.alloc_ms(), ec.overlap_ms(),
                      ec.commit_ms());
    KRK_INFO("expert warmup: %s order, %d rounds, %zu experts/layer deep; %s, "
             "VRAM +%zu experts (%.0f MiB of a %d%% seed cap, %zu slots); "
             "%.0f ms%s",
             ranked ? "ranked" : "even", rounds, depth, tier_note, promoted,
             static_cast<f64>(promoted_bytes) / 1048576.0, seed_pct,
             ec.resident_slots(), ms, why.c_str());
    // The warmup's traffic is not the run's. Reset the counters it earned so
    // the residency report describes what the run did with the tiers it was
    // handed; the line above is the warmup's own record of what it spent.
    ec.reset_counters();
}

// KRK_DUMP_MOE=1: report, per (layer, expert) group, how many elements of the
// gathered input, the up projection and the group's output are not finite.
//
// This exists because of an arithmetic accident that hid a real defect for a
// long time: nemotron_h_moe carries no ffn_gate_exps, so the gate leg of the
// expert MLP is absent (-see the comment on ExpertSource::present), which makes
// silu(gate) * up evaluate to 0 * up. Almost any corruption of an expert's
// weights is therefore multiplied away to an exact zero -- but 0 * NaN is NaN,
// so an expert whose bytes are not a number survives that multiplication as the
// only visible trace. A per-group non-finite count is what turns "the residual
// went NaN somewhere in layer 27" into "this expert's weights are not finite".
// KRK_SPEC_TRACE=1 prints one line per speculative round: what the draft
// proposed, what the target's verification rows predicted for the same
// positions, how long the accepted prefix was, and -- when the round failed
// before the block ran -- the pre-check's two tokens. The counters report a
// rate, not which proposal was rejected for what.
// The target's own top-3 at a position, for the round traces: a drafter whose
// candidate sits elsewhere on the row and one whose row is flat are told apart
// in one run, and a near-tie between two instances of the same weights shows up
// as the draft's token being the target's second or third candidate.
// The top-3 scan itself, with no framing: every trace that prints a row
// (target_top3, the dflash block_row0, the pre-check's target row) walks the
// vocab the same way and formats it the same way, so the scan lives here once
// and each caller prints its own prefix.
static void print_row_top3(FILE *f, const f32 *logits, i64 n) {
    f32 b0 = -1e30f, b1 = -1e30f, b2 = -1e30f;
    i32 i0 = -1, i1 = -1, i2 = -1;
    for (i64 v = 0; v < n; v++) {
        const f32 x = logits[v];
        if (x > b0) {
            b2 = b1; i2 = i1; b1 = b0; i1 = i0; b0 = x;
            i0 = static_cast<i32>(v);
        } else if (x > b1) {
            b2 = b1; i2 = i1; b1 = x; i1 = static_cast<i32>(v);
        } else if (x > b2) {
            b2 = x; i2 = static_cast<i32>(v);
        }
    }
    std::fprintf(f, "%d(%.4g) %d(%.4g) %d(%.4g)", i0, static_cast<f64>(b0),
                 i1, static_cast<f64>(b1), i2, static_cast<f64>(b2));
}

static void print_target_top3(FILE *f, const f32 *logits, i64 n) {
    std::fprintf(f, "] target_top3=[");
    print_row_top3(f, logits, n);
    std::fputc(']', f);
}

static bool spec_trace() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_SPEC_TRACE");
        return e && e[0] != '0';
    }();
    return v;
}

// The acceptance probability a rejection-sampling verifier would give one
// proposal position, computed the way a serving stack measures acceptance
// length: both rows at temperature 1.0, top-k 20 truncated and renormalized
// (Poolside's card settings; top-p is 1.0 there, so top-k is the whole
// regime). E = sum over the union of both top-k sets of min(p_t, p_d).
// Rows that agree exactly give 1.0 and disjoint rows give ~0, so this is the
// per-proposal version of the number a model card reports per step.
static f64 sampled_accept_prob(const f32 *trow, const f32 *drow, i64 n) {
    constexpr i32 kTopMax = 20;
    const i32 k = std::min<i32>(kTopMax, static_cast<i32>(n));
    struct Pick {
        f32 p;
        i32 id;
    };
    Pick pt[20], pd[20];
    thread_local std::vector<i32> idx;
    idx.resize(static_cast<size_t>(n));
    for (i32 side = 0; side < 2; side++) {
        const f32 *row = side ? drow : trow;
        Pick *out = side ? pd : pt;
        for (i64 v = 0; v < n; v++) idx[static_cast<size_t>(v)] = static_cast<i32>(v);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](i32 a, i32 b) { return row[a] > row[b]; });
        const f32 mx = row[static_cast<size_t>(idx[0])];
        f64 z = 0.0;
        for (i32 i = 0; i < k; i++) {
            out[i].id = idx[static_cast<size_t>(i)];
            z += std::exp(static_cast<f64>(row[static_cast<size_t>(out[i].id)] - mx));
        }
        for (i32 i = 0; i < k; i++) {
            out[i].p = static_cast<f32>(
                std::exp(static_cast<f64>(row[static_cast<size_t>(out[i].id)] - mx)) /
                z);
        }
    }
    f64 acc = 0.0;
    for (i32 i = 0; i < k; i++) {
        f64 pdv = 0.0;
        for (i32 j = 0; j < k; j++)
            if (pd[j].id == pt[i].id) {
                pdv = static_cast<f64>(pd[j].p);
                break;
            }
        acc += std::min(static_cast<f64>(pt[i].p), pdv);
    }
    return acc;
}

// The sampled-accept accounting is on by default; KRK_SPEC_SAMPLED_ACC=0
// restores the device-argmax fast path for a pure timing run.
static bool sampled_acc_on() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_SPEC_SAMPLED_ACC");
        return !(e && e[0] == '0');
    }();
    return v;
}

// KRK_SPEC_RS=1 turns on rejection-sampling verification for temperature>0
// runs: proposals are SAMPLED from the draft's rows through the run's sampler
// chain, each is accepted with min(1, p/q) against the target's row of the
// same position, and a rejection resamples that position from the corrected
// distribution (p-q)+ -- the exact speculative-sampling algorithm, so the
// emitted stream stays distributed as the target alone would produce it.
// Default off: every existing arm and every test is greedy, whose argmax-match
// rule is this algorithm's temp-0 specialization. The repetition penalty is
// not part of the verify math on either side (it is a decode-time heuristic,
// not a distribution).
static bool spec_rs_on() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_SPEC_RS");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    return v;
}

// The sampler's filtered distribution over one row, mirroring
// Sampler::sample's chain (temperature, top-k, top-p, min-p, renormalized),
// sorted by probability and ready for a CDF draw.
static void sampler_support(const f32 *logits, const SampleParams &sp,
                            i32 n_vocab, std::vector<i32> &ids,
                            std::vector<f64> &ps) {
    ids.clear();
    ps.clear();
    if (n_vocab <= 0) return;
    if (sp.greedy || sp.temp <= 0.0f) {
        i32 best = 0;
        for (i32 i = 1; i < n_vocab; i++)
            if (logits[i] > logits[best]) best = i;
        ids.push_back(best);
        ps.push_back(1.0);
        return;
    }
    thread_local std::vector<i32> idx;
    idx.resize(static_cast<size_t>(n_vocab));
    for (i32 i = 0; i < n_vocab; i++) idx[static_cast<size_t>(i)] = i;
    i32 keep = n_vocab;
    if (sp.top_k > 0 && sp.top_k < keep) keep = sp.top_k;
    const auto better = [&logits](i32 a, i32 b) { return logits[a] > logits[b]; };
    if (keep < n_vocab)
        std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), better);
    else
        std::sort(idx.begin(), idx.end(), better);
    const f32 inv_temp = 1.0f / sp.temp;
    const f32 mx = logits[static_cast<size_t>(idx[0])];
    f32 sum = 0;
    ps.resize(static_cast<size_t>(keep));
    for (i32 i = 0; i < keep; i++) {
        const f32 e = std::exp(
            (logits[static_cast<size_t>(idx[static_cast<size_t>(i)])] - mx) *
            inv_temp);
        ps[static_cast<size_t>(i)] = static_cast<f64>(e);
        sum += e;
    }
    if (sum <= 0) {
        ps.assign(static_cast<size_t>(keep), 1.0 / keep);
    } else {
        for (auto &q : ps) q /= sum;
    }
    ids.resize(static_cast<size_t>(keep));
    for (i32 i = 0; i < keep; i++)
        ids[static_cast<size_t>(i)] = idx[static_cast<size_t>(i)];
    i32 cut = keep;
    if (sp.top_p < 1.0f && sp.top_p > 0.0f) {
        f64 c = 0;
        for (i32 i = 0; i < keep; i++) {
            c += ps[static_cast<size_t>(i)];
            if (c >= sp.top_p) { cut = i + 1; break; }
        }
    }
    if (sp.min_p > 0.0f) {
        const f64 thr = static_cast<f64>(sp.min_p) * ps[0];
        i32 c = 0;
        while (c < cut && ps[static_cast<size_t>(c)] >= thr) c++;
        if (c > 0) cut = c;
    }
    ids.resize(static_cast<size_t>(cut));
    ps.resize(static_cast<size_t>(cut));
    f64 z = 0;
    for (auto q : ps) z += q;
    if (z > 0)
        for (auto &q : ps) q /= z;
}

// The probability a sampler-filtered support assigns one token.
static f64 support_prob(const std::vector<i32> &ids, const std::vector<f64> &ps,
                        i32 token) {
    for (size_t i = 0; i < ids.size(); i++)
        if (ids[i] == token) return ps[i];
    return 0.0;
}

// A draw from a filtered support (inverse-CDF).
static i32 support_draw(const std::vector<i32> &ids,
                        const std::vector<f64> &ps, Rng &rng) {
    if (ids.empty()) return 0;
    const f64 u = static_cast<f64>(rng.next_f32());
    f64 c = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        c += ps[i];
        if (u <= c) return ids[i];
    }
    return ids.back();
}

// A draw from the corrected distribution max(0, p - q) over the target's
// support -- the resample-on-reject step of speculative sampling.
static i32 corrected_draw(const std::vector<i32> &pids,
                          const std::vector<f64> &pps,
                          const std::vector<i32> &qids,
                          const std::vector<f64> &qps, Rng &rng) {
    std::vector<f64> c(pps.size());
    f64 z = 0;
    for (size_t i = 0; i < pps.size(); i++) {
        c[i] = std::max(0.0, pps[i] - support_prob(qids, qps, pids[i]));
        z += c[i];
    }
    if (z <= 0) { // p's support sits inside q's; fall back to p itself
        c = pps;
        z = 1.0;
    }
    const f64 u = static_cast<f64>(rng.next_f32()) * z;
    f64 acc = 0;
    for (size_t i = 0; i < c.size(); i++) {
        acc += c[i];
        if (u <= acc) return pids[i];
    }
    return pids.back();
}

// The slot the draft is re-fed at, at the end of a round. Both models must end
// a round in the same state -- KV [0, pos) with logits predicting pos -- and the
// last emitted token's natural slot is pos-1, which is the convention the
// pre-check's mismatch branch below has always used. This realign used `pos`
// instead, so the draft took the last token one row too high, a row below it
// stayed stale, and its logits then predicted pos+1 where the target predicted
// pos. Both speculative rounds carry the same code, so both carry the same
// choice. KRK_SPEC_REALIGN=0 restores the off-by-one for the A/B.
static bool spec_realign_at_natural_slot() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_SPEC_REALIGN");
        return !(e && e[0] == '0');
    }();
    return v;
}

static bool moe_probe() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_DUMP_MOE");
        return e && e[0] != '0';
    }();
    return v;
}

// KRK_MOE_GROUP=0 restores the separate up/gate gemms plus the standalone
// silu_mul for a routed expert. Default ON, because the pair shares one
// activation and the activation reads the same row of both projections, so the
// three launches collapse into one -- the same fold the dense FFN has had since
// fused_gate_silu_kernel landed (measured there: 30 launches and +6.135% decode
// on SmolLM2-135M, text byte-identical).
static bool moe_group() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_MOE_GROUP");
        return !(e && e[0] == '0');
    }();
    return v;
}
// KRK_MOE_BATCH=0 restores one launch group per routed expert on the decode
// path. ON by default because at one token every routed expert owns exactly one
// row and they all read the SAME activation, so the layer's work is one group
// launch per four experts, one activation launch for all of them, and one down
// launch -- 12 launches where the per-expert loop issued 42. Nothing about the
// arithmetic changes: same experts, same ascending order, same row-scaled silu,
// same scatter, which is what the arm is for.
static bool moe_batch() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_MOE_BATCH");
        return !(e && e[0] == '0');
    }();
    return v;
}

// KRK_ACTIVE_STAGE=1 submits the next expert's DMA as each expert is queued.
// Read once, not per expert visit: it is an environment variable, not state.
static bool moe_active_stage() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_ACTIVE_STAGE");
        return e && e[0] == '1';
    }();
    return v;
}

// KRK_MOE_SILU_FUSED=1 puts the activation back inside the gate/up group.
// It is OFF because that kernel has no row maximum to take: a row whose
// silu(gate)*up exceeds f16's range is stored clipped, and the clip is not
// something a later un-scale can undo (measured on Laguna-XS-2.1: ONE
// saturated element in the product row put 16 non-finite elements in the
// down projection's output and 1020 clipped elements in the experts' sum).
// The row-scaled silu_mul the group's fallback already runs does have one --
// it is one block per row with the whole row's maximum in shared memory --
// so the pair is grouped (two launches, not three) and the activation is
// then scaled per row, which is the contract the scatter already folds back
// in. The switch exists so the fuse's cost is measurable one process apart.
// KRK_MOE_LOOKAHEAD=1 stages a batched MoE pass's expert union into VRAM before
// the first row runs it. Off by default: it is a scheduling change whose win
// depends on the batch (a speculative verify pass) and on how far the device
// budget can hold the union, and it must be measured on the model that has one.
static bool lookahead_stage() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_MOE_LOOKAHEAD");
        // Opt-in, because on the shape it was written for it measured a small
        // LOSS and no hit-rate change: Tiel 35B-A3B, 7682 MiB of device expert
        // memory, 64 tokens, pairs 19.5/19.9 tok/s without against 19.0/19.1
        // with (4137 against 4144 promotions). The reason is the one the audit
        // names: a promotion is ~570 us/expert and the H2D link on this box
        // runs at 3.4-4.4 GB/s (the load line says the same), so 17% of a token
        // is bytes and the copies cannot hide behind compute that is shorter
        // than they are. It stays as the arm for a batched verify pass, where
        // the union is known a whole row early -- and `n > 1` is now what it
        // actually gates, which the old condition had inverted.
        return e && e[0] == '1';
    }();
    return v;
}

// One upload for a whole layer's routed set (ExpertCache::promote_batch).
// MEASURED AND OFF BY DEFAULT. The hypothesis was that a promotion's 143
// us/expert figure was mostly per-call enqueue cost; on the REAP-320 file it is
// not. A promotion measures ~500-570 us/expert there, the pageable ring's 4 MiB
// memcpy is most of it, and batching 4 experts into one call changed nothing in
// either direction: 5 interleaved order-balanced pairs of 16 greedy tokens,
// 111.8 against 112.9 ms/step median (delta MAD 2.6 ms, 3 of 5 pairs), with the
// batched arm's `xfer` row consistently HIGHER (366-385 against 249-287 ms over
// 15 steps) because more chunks than the 4-slot ring holds in one call is more
// ring handoff, not less. Same text in all ten runs. It is kept as the arm for a
// deeper ring or a pinned WARM tier, where the per-call cost is not already
// hidden behind a memcpy. KRK_MOE_BATCH_PROMOTE=1 enables it.
static bool batch_promote() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_MOE_BATCH_PROMOTE");
        return e && e[0] == '1';
    }();
    return v;
}

static i64 profile_snapshot_every() {
    static const i64 v = [] {
        const char *e = std::getenv("KRK_PROFILE_SNAPSHOT");
        return e && *e ? static_cast<i64>(std::atoll(e)) : 0;
    }();
    return v;
}

static bool moe_silu_fused() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_MOE_SILU_FUSED");
        return e && e[0] != '0';
    }();
    return v;
}

// ---- the batched decode path -----------------------------------------------
//
// One token's routed experts, computed by launches that cover ALL of them
// instead of one launch group per expert. At n == 1 every expert's group is a
// single row, and every one of those rows is the SAME activation row -- which
// is what makes the batch possible at all:
//
//   * gate/up: eight matrices per launch (four experts, gate then up), all
//     reading ws_xn_. The gather the per-expert loop issues to build that row
//     copies row 0, so the batch skips it rather than batching it.
//   * the activation: one row-scaled launch for every expert's row
//     (silu_mul_rows), the single-row kernel's own body, one block per row.
//   * down: one launch, each matrix reading its OWN expert's silu row -- the
//     per-matrix activation the group takes.
//   * the scatter stays per expert, ascending, because that is the routed sum's
//     accumulation order and its storage exponents.
//
// It refuses (returns false having launched nothing) for every shape the
// launches cannot express, and the caller's per-expert loop then runs unchanged.
// Refusing BEFORE the first launch is load-bearing: a partial batch would add
// some experts to the sum twice when the loop ran the layer again.
bool Engine::moe_decode_batch(const LayerWeights &L, i32 layer, i32 k, i32 ne,
                              i64 ff_exp, bool use_plan_off, bool inplace_ok) {
    if (!use_plan_off || k <= 0 || ff_exp <= 0) return false;
    if (!ws_ebatch_ || !ws_edown_ || !ws_ebscale_) return false;
    const size_t as = be_->act_size();
    const i64 ffs = model_.n_ff_ws();
    auto slot = [&](void *base, i64 idx, i64 stride) {
        return static_cast<void *>(static_cast<char *>(base) +
                                   static_cast<size_t>(idx) *
                                       static_cast<size_t>(stride) * as);
    };
    auto scale_at = [&](i32 j) {
        return static_cast<f32 *>(ws_ebscale_) + j;
    };

    // This layer's routed experts in ASCENDING id order -- the order the
    // per-expert loop visits them in, and therefore the order the routed sum
    // accumulates in. The batch changes the launch count and nothing else.
    batch_ids_.clear();
    for (i32 e = 0; e < ne; e++)
        for (i32 s = 0; s < k; s++)
            if (moe_sel_[static_cast<size_t>(s)] == e) {
                batch_ids_.push_back(e);
                break;
            }
    if (batch_ids_.empty()) return false;

    // The batch holds EVERY routed expert resident at once, so it needs the
    // cache to be able to hold them all. That is the semantic condition, and it
    // is checked before a single acquire: with fewer slots than distinct
    // experts the resolve below evicts as it goes, and the expert resolved
    // first is gone by the time the launches read it.
    if (model_.experts().capacity_slots() < batch_ids_.size()) return false;

    // Every expert is resolved BEFORE the first launch, so a refusal here has
    // issued nothing and the caller can fall back safely.
    batch_res_.clear();
    for (i32 e : batch_ids_) {
        if (inplace_ok) model_.experts().take_prefetched(layer, e);
        const ResidentExpert *re =
            inplace_ok ? model_.experts().acquire_inplace(L.experts, layer, e)
                       : nullptr;
        if (!re) re = model_.experts().acquire(L.experts, layer, e);
        if (!re) {
            forward_error_ = expert_resource_error() + " (moe layer " +
                             std::to_string(layer) + " expert " +
                             std::to_string(e) + ")";
            return false;
        }
        batch_res_.push_back(re);
    }

    // Every resolved expert must still be its OWN slot. `acquire()` below
    // EVICTS when the cache cannot hold the routed set: resolve expert 1 with a
    // one-slot budget, then expert 2, and `batch_res_[0]` still points at the
    // buffer expert 2 just overwrote. The launches then multiply both experts by
    // expert 2's weights -- finite, plausible, and wrong, with no refusal and no
    // fraction of the arithmetic that looks off.
    //
    // Measured, not theorised: `test_moe_schema_and_laziness` failed
    // "generation is identical with a 1-slot and a large expert cache"
    // (5 4 5 4 5 4 against 5 5 4 5 4 5) and returned to 32/32 the moment the
    // batch arm was disabled -- the per-expert loop acquires and computes one
    // expert at a time, so it cannot alias. Distinct buffers between the first
    // and last resolve is the invariant the batch needs; when it does not hold,
    // fall back BEFORE anything is launched, which is why this check sits after
    // the resolve loop and above the first launch.
    // All nine field pairs, not gate-vs-gate and friends: a tiered cache that
    // demotes one slot and promotes the next can hand expert 2 the cell expert
    // 1 used for a DIFFERENT slice, which a same-field comparison misses. The
    // check that failed with a host tier present was exactly that shape -- the
    // routed set aliased through the pool and the decode collapsed to token 0
    // after the first step.
    {
        const QuantTensor *slices[3 * 16];
        size_t n = 0;
        for (const ResidentExpert *re : batch_res_) {
            if (n + 3 > sizeof(slices) / sizeof(slices[0])) return false;
            slices[n++] = &re->gate;
            slices[n++] = &re->up;
            slices[n++] = &re->down;
        }
        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++)
                if (slices[i]->data == slices[j]->data) return false;
    }

    for (size_t base = 0; base < batch_ids_.size(); base += kMoeBatch) {
        const i32 cnt = static_cast<i32>(
            std::min<size_t>(static_cast<size_t>(kMoeBatch),
                             batch_ids_.size() - base));
        // gate/up: four experts at a time (the launch's matrix array holds
        // eight -- one gate and one up each), all over the same activation.
        for (i32 c = 0; c < cnt; c += 4) {
            const i32 g = std::min<i32>(4, cnt - c);
            void *out[8] = {};
            const void *w[8] = {};
            i64 no[8] = {};
            DType ts[8] = {};
            DType def = DType::F16;
            for (i32 j = 0; j < g; j++) {
                const ResidentExpert *re = batch_res_[base + c + j];
                out[2 * j] = slot(ws_ebatch_, 2 * (c + j), ffs);
                out[2 * j + 1] = slot(ws_ebatch_, 2 * (c + j) + 1, ffs);
                w[2 * j] = re->gate.data;
                w[2 * j + 1] = re->up.data;
                no[2 * j] = ff_exp;
                no[2 * j + 1] = ff_exp;
                ts[2 * j] = re->gate.type;
                ts[2 * j + 1] = re->up.type;
                def = re->up.type;
            }
            // Not the silu-fused form: that epilogue takes exactly two
            // matrices. The activation is the launch below, shared by all of
            // them.
            be_->gemm_group(out, ws_xn_, w, def, no, n_embd_, 2 * g, 1, false,
                            nullptr, false, ts);
        }
        {
            void *aout[8] = {};
            const void *agate[8] = {};
            const void *aup[8] = {};
            f32 *ascale[8] = {};
            for (i32 j = 0; j < cnt; j++) {
                aout[j] = slot(ws_ebatch_, 2 * j, ffs);
                agate[j] = aout[j];
                aup[j] = slot(ws_ebatch_, 2 * j + 1, ffs);
                ascale[j] = scale_at(j);
            }
            be_->silu_mul_rows(aout, agate, aup, ff_exp, cnt, ascale);
        }
        {
            void *dout[8] = {};
            const void *dw[8] = {};
            const void *dx[8] = {};
            i64 dno[8] = {};
            DType dts[8] = {};
            DType def = DType::F16;
            for (i32 j = 0; j < cnt; j++) {
                const ResidentExpert *re = batch_res_[base + j];
                dout[j] = slot(ws_edown_, j, n_embd_);
                dx[j] = slot(ws_ebatch_, 2 * j, ffs);
                dw[j] = re->down.data;
                dno[j] = n_embd_;
                dts[j] = re->down.type;
                def = re->down.type;
            }
            be_->gemm_group(dout, nullptr, dw, def, dno, ff_exp, cnt, 1, false,
                            nullptr, false, dts, dx);
        }
        // The routed sum, expert by expert in the per-expert loop's order, with
        // that expert's segment of the device plan (one row, its weight).
        for (i32 j = 0; j < cnt; j++) {
            const i32 e = batch_ids_[base + j];
            const size_t off = static_cast<size_t>(plan_off_[static_cast<size_t>(e)]);
            be_->scatter_axpy_rows(
                ws_ffn_, slot(ws_edown_, j, n_embd_),
                static_cast<const i32 *>(ws_plan_) + off,
                static_cast<const f32 *>(ws_alpha_) + off, 1, n_embd_,
                scale_at(j), static_cast<i32 *>(ws_fexp_));
            model_.experts().note_read(layer, e);
        }
    }
    return true;
}

// ---- Engram (V4.1): conditional memory, before the layer's mixer ---------
//
// llama.cpp's build_engram, in the same order. The rows are gathered ON THE
// HOST out of the mmap'd shared table (they are the file's own bytes, and a
// token touches n_cols of them), dequantized into the lane-width buffer, and
// then everything downstream is ordinary matmuls:
//
//   rows  = table[hash(window of n_gram tokens)]     [n, n_cols*head_dim]
//   kv    = engram_kv @ rows                         [n, n_embd*(hc+1)]
//   key   = kv[:, 0 : hc*n_embd]                     [n, hc, n_embd]
//   value = kv[:, hc*n_embd : ]                      [n, n_embd]
//   gate  = sigmoid(sign(dot)*sqrt(|dot|))           [n, hc]
//   dot_c = sum_e rmsnorm(x)[e] * rmsnorm(key_c)[e] * engram_gate[e, c] / sqrt(ne)
//   lanes += value[c] * gate[c]                      (broadcast over lanes)
//
// The gate is the reference's signed square root: sigmoid(copysign(sqrt(|d|), d))
// with |d| clamped at 1e-6, so a dot of exactly 0 gates to 0.5 and a large one
// saturates. sign(0) is +1, which is what `dot >= 0` encodes.
bool Engine::engram_block(const LayerWeights &L, i32 l, const i32 *toks, i64 pos0,
                          i32 n) {
    const ModelConfig &mc = model_.cfg();
    const i64 ne = n_embd_;
    const i64 hc = mc.hc_n;
    const i64 head_dim = mc.ple_head_dim;
    const i32 n_cols = (mc.ple_ngram - 1) * mc.ple_heads_per_ngram;
    if (!mc.ple || !L.engram_kv.present() || !ws_eng_rows_ || !model_.ple_table())
        return false;
    EngramModule mod;
    // The module index is the layer's position in ple.layers -- NOT the layer
    // number. A model whose Engram layers are 0, 4 and 8 uses modules 0, 1, 2.
    i32 e = 0;
    for (size_t k = 0; k < mc.ple_layers.size(); k++)
        if (mc.ple_layers[k] == l) { e = static_cast<i32>(k); break; }
    // The file's arrays are i64; the hash wants u64 multipliers and u32
    // offsets/sizes. Reinterpreting the i64 pointers would read the wrong
    // elements (u32 x 2 against an i64 stride), so the two narrow arrays are
    // copied per call -- n_cols entries, not a per-token cost that matters, and
    // the VALUES are what the file declared.
    mod.mult = reinterpret_cast<const u64 *>(
        mc.ple_multipliers.data() + static_cast<i64>(e) * mc.ple_ngram);
    eng_off_.resize(static_cast<size_t>(n_cols));
    eng_vocab_.resize(static_cast<size_t>(n_cols));
    for (i64 c = 0; c < n_cols; c++) {
        const i64 off = mc.ple_head_offsets[static_cast<size_t>(e * n_cols + c)];
        const i64 voc = mc.ple_head_vocab_sizes[static_cast<size_t>(e * n_cols + c)];
        eng_off_[static_cast<size_t>(c)] = static_cast<u32>(off);
        eng_vocab_[static_cast<size_t>(c)] = static_cast<u32>(voc);
    }
    mod.offsets = eng_off_.data();
    mod.vocab_sizes = eng_vocab_.data();
    mod.n_gram = mc.ple_ngram;
    mod.heads_per_ngram = mc.ple_heads_per_ngram;
    if (!mod.valid()) return false;

    const i64 pad = static_cast<i64>(mc.engram_token_map[static_cast<size_t>(mc.engram_pad_id)]);
    std::vector<i32> window;
    const i64 n_prev = mc.ple_ngram - 1;
    // The ring is one window long and holds the tokens BEFORE pos0; a chunk
    // longer than the window only ever reaches one window back, so no more is
    // kept. A short first pass leaves it short; the reads below are bounds
    // checked against the sequence start, not against the ring's size.
    const i64 hist = static_cast<i64>(eng_hist_.size());
    for (i32 t = 0; t < n; t++) {
        const i64 pos = pos0 + t;
        // The window is the n_gram-1 tokens BEFORE this one, oldest first. Only
        // the ones that exist are passed; the hash treats the rest as dead and
        // substitutes the pad id (which degrades the n-gram, not the row).
        window.clear();
        for (i64 s = n_prev; s >= 1; s--) {
            if (pos - s < 0) {
                window.push_back(-1);           // before the sequence start
            } else if (t - s >= 0) {
                window.push_back(toks[t - s]);  // inside this chunk
            } else {
                // Before the chunk: the ring, oldest first.
                const i64 k = hist + (t - s);
                window.push_back(k >= 0 ? eng_hist_[static_cast<size_t>(k)] : -1);
            }
        }
        if (!engram_hash(eng_ids_.data() + static_cast<i64>(t) * n_cols, mod,
                         mc.engram_token_map.data(),
                         static_cast<i64>(mc.engram_token_map.size()), pad,
                         toks[t], window.data(), n_prev))
            return false;
    }
    // The gather: rows come straight out of the mapping, so the stride is the
    // file's own row bytes and the buffer is f32 after dequant.
    const i64 row_bytes = static_cast<i64>(model_.ple_row_bytes());
    std::vector<u8> raw(static_cast<size_t>(n) * n_cols * row_bytes);
    engram_gather(raw.data(), static_cast<const u8 *>(model_.ple_table()),
                  static_cast<i64>(model_.ple_table_rows()), row_bytes,
                  eng_ids_.data(), n, n_cols);
    f32 *rows = static_cast<f32 *>(ws_eng_rows_);
    for (i32 t = 0; t < n; t++)
        for (i32 c = 0; c < n_cols; c++)
            dequant_row(model_.ple_table_type(),
                        raw.data() + (static_cast<i64>(t) * n_cols + c) * row_bytes,
                        rows + (static_cast<i64>(t) * n_cols + c) * head_dim,
                        head_dim);

    // kv = engram_kv @ rows, then the split: hc key blocks and one value block.
    const i64 kvw = n_cols * head_dim;
    gemm_w(ws_eng_kv_, ws_eng_rows_, L.engram_kv, ne * (hc + 1), kvw, n);
    // gate: the signed square root, from the dot of the normed input against
    // each normed key block, scaled by engram_gate and 1/sqrt(ne).
    be_->rmsnorm(ws_hc_n_, ws_x_, nullptr, n, ne, mc.rms_eps);
    for (i32 t = 0; t < n; t++) {
        const f32 *xn = static_cast<const f32 *>(ws_hc_n_) + static_cast<i64>(t) * ne;
        const f32 *kv = static_cast<const f32 *>(ws_eng_kv_) +
                        static_cast<i64>(t) * ne * (hc + 1);
        const f32 *vb = kv + hc * ne;
        for (i64 c = 0; c < hc; c++) {
            const f32 *key = kv + c * ne;
            // rmsnorm of the KEY block, computed here because it is one short
            // row and a kernel launch would cost more than the loop.
            f64 ss = 0.0;
            for (i64 i = 0; i < ne; i++) ss += (f64)key[i] * key[i];
            const f32 kinv = 1.0f / std::sqrt((f32)(ss / ne) + mc.rms_eps);
            f32 dot = 0.0f;
            for (i64 i = 0; i < ne; i++)
                // The file's engram_gate is ne={n_embd, hc}: hc ROWS of
                // n_embd, so element (e, c) is at c*n_embd + e -- the same
                // transposition trap hc_norm_lanes' gamma has.
                dot += xn[i] * (key[i] * kinv) * L.engram_gate[c * ne + i];
            dot *= 1.0f / std::sqrt((f32)ne);
            const f32 mag = std::sqrt(std::max(std::fabs(dot), 1e-6f));
            const f32 sign = dot >= 0.0f ? 1.0f : -1.0f;
            const f32 g = 1.0f / (1.0f + std::exp(-(mag * sign)));
            // lanes[c] += value * gate[c]
            f32 *lane = static_cast<f32 *>(ws_hc_) + (static_cast<i64>(t) * hc + c) * ne;
            for (i64 i = 0; i < ne; i++) lane[i] += vb[i] * g;
        }
    }
    return true;
}

// ---- DeepSeek-V4.1: the Sinkhorn lane stack ------------------------------
//
// The same residual SHAPE as qwen4exp -- `hc` lanes of n_embd, token-major --
// and a completely different mixer. There is no bottleneck and no per-lane
// gamma: one projection over the flattened, weightlessly normed stack emits
// (2+hc)*hc numbers, which split into
//
//   pre[c]     the collapse the sub-block reads            (sigmoid, +eps)
//   post[c]    the gain on the block's own output          (2*sigmoid)
//   comb[d][s] the doubly-stochastic mix of the lanes      (sinkhorn)
//
// and the sub-block does not ADD to the residual, it REPLACES every lane:
//
//   lanes'[d] = block_out * post[d] + sum_s lanes[s] * comb[d][s]
//
// The split and the scatter are single backend ops (hc_sk_split / hc_sk_pre /
// hc_sk_post) rather than loops here, because the sinkhorn's arithmetic order
// is the thing the reference specifies and src/hc.cpp is where it is written
// down and tested. This function is only the plumbing.
bool Engine::hc_sk_enter(const LayerWeights &L, bool attn, i32 n) {
    const i64 ne = n_embd_;
    const i64 hc = hc_.hc;
    const i64 wide = hc * ne;
    const i64 mix_dim = (2 + hc) * hc;
    const QuantTensor &fn = attn ? L.hc_attn_fn : L.hc_ffn_fn;
    const f32 *base = attn ? L.hc_attn_base : L.hc_ffn_base;
    const f32 *scale = attn ? L.hc_attn_scale : L.hc_ffn_scale;
    // One norm over the whole flattened stack and NO weight: `ggml_rms_norm(
    // flat)` in the reference. Passing a gamma here would be a different
    // function, and the family has no gamma to pass.
    be_->rmsnorm(ws_hc_n_, ws_hc_, nullptr, n, wide, model_.cfg().rms_eps);
    gemm_w(ws_hc_fn_, ws_hc_n_, fn, mix_dim, wide, n);
    be_->hc_sk_split(ws_hc_pre_, ws_hc_post_, ws_hc_comb_, ws_hc_fn_, scale, base,
                     n, hc, model_.cfg().hc_sinkhorn_iters, model_.cfg().hc_eps);
    // WHICH pre the collapse uses is the single-pass lag. In the reference's
    // V4.1 graph each sub-block computes its own mixes but the ATTENTION reads
    // the pre-mix its PREVIOUS layer's FFN produced (`pre_prev`), and the first
    // layer's attention reads lane 0 -- the identity. With the lag off every
    // sub-block reads the pre it just computed. Both are one op; only the
    // pointer differs.
    const bool lag = model_.cfg().hc_no_head;
    const void *pre_use = ws_hc_pre_;
    if (lag) {
        if (attn) {
            if (hc_pre_valid_) {
                pre_use = ws_hc_pre_prev_;
            } else {
                // Layer 0: the identity collapse, written as a one-hot pre
                // rather than a strided copy, so it is the same op.
                f32 *one = static_cast<f32 *>(ws_hc_pre_attn_);
                be_->fill0(one, static_cast<size_t>(n) * hc * 4);
                for (i32 t = 0; t < n; t++)
                    one[static_cast<i64>(t) * hc] = 1.0f;
                pre_use = ws_hc_pre_attn_;
            }
            // Hand THIS attention's pre to the layer's FFN, which reads it in
            // place of its own. Saved before the collapse so the one-hot above
            // is overwritten by the real one.
            if (hc_pre_valid_) {
                be_->copy_act(ws_hc_pre_attn_, ws_hc_pre_,
                              static_cast<i64>(n) * hc * 4);
            } else {
                // Still the identity collapse: the FFN of layer 0 reads
                // attn_mix.pre, which IS the fresh pre, so save it over the
                // one-hot -- but only AFTER the collapse read it.
                be_->hc_sk_pre(ws_xn_, ws_hc_, pre_use, n, hc, ne);
                be_->copy_act(ws_x_, ws_xn_, static_cast<i64>(n) * ne);
                be_->copy_act(ws_hc_pre_attn_, ws_hc_pre_,
                              static_cast<i64>(n) * hc * 4);
                return true;
            }
        } else {
            // The FFN reads the ATTENTION's pre, not its own.
            pre_use = ws_hc_pre_attn_;
        }
    }
    be_->hc_sk_pre(ws_xn_, ws_hc_, pre_use, n, hc, ne);
    be_->copy_act(ws_x_, ws_xn_, static_cast<i64>(n) * ne);
    // Whatever pre this sub-block should hand on: the attention's goes to this
    // layer's FFN, the FFN's to the next layer's attention (and the final
    // collapse).
    if (lag) {
        if (attn) {
            be_->copy_act(ws_hc_pre_attn_, ws_hc_pre_, static_cast<i64>(n) * hc * 4);
        } else {
            be_->copy_act(ws_hc_pre_prev_, ws_hc_pre_, static_cast<i64>(n) * hc * 4);
            hc_pre_valid_ = true;
        }
    }
    return true;
}

void Engine::hc_sk_leave(const LayerWeights &L, bool attn, i32 n) {
    (void)attn;
    const i64 ne = n_embd_;
    const i64 hc = hc_.hc;
    // ws_x_ is `pre-mix + block_out`, so the block's own output is that minus
    // the pre-mix -- exact, and it never mixes a residual sum into the lanes
    // the scatter reads.
    be_->copy_act(ws_hc_out_, ws_x_, static_cast<i64>(n) * ne);
    be_->axpy(ws_hc_out_, ws_xn_, -1.0f, static_cast<i64>(n) * ne);
    // lanes_out may not alias lanes_in, which is why the scatter reads ws_hc_
    // and writes ws_hc_new_ before the copy back.
    be_->hc_sk_post(ws_hc_new_, ws_hc_, ws_hc_out_, ws_hc_post_, ws_hc_comb_, n,
                    hc, ne);
    be_->copy_act(ws_hc_, ws_hc_new_, static_cast<i64>(n) * hc * ne);
}

// ---- MoVA: the attention VALUES are a routed expert set (k2-horizon) ------
//
// This is llama.cpp's build_routed_value, in the same order, because every step
// of it is a place to get the model wrong in a way that still decodes:
//
//   gate   = attn_v_gate @ xn                    [n_value_expert]
//   probs  = softmax | sigmoid(gate)             (expert_gating_func)
//   score  = probs + attn_v_gate_b               SELECTION ONLY
//   sel    = top-k(score)                        k = value_expert_used_count
//   w_k    = probs[sel_k]                        the UNBIASED probability
//   w      = w / sum(w)                          only when expert_weights_norm
//   V      = sum_k w_k * silu(attn_v_exps[sel_k] @ xn)
//
// Three of those are load-bearing and cheap to get backwards:
//   * the bias shifts SELECTION and never the weight. Adding it to `probs`
//     changes the mixture, and on a 4-of-8 route it changes the value row.
//   * the silu is applied to the EXPERT OUTPUT, before the weight, not to the
//     input and not after the sum. `sum w_k silu(...)` is not `silu(sum w_k ...)`.
//   * the renormalisation divides by the sum of the SELECTED probabilities, and
//     is skipped entirely when the file does not ask for it -- a sigmoid router
//     without the norm is a sum of independent gates, not a convex mix.
//
// The expert bank is resident and small (n_value_expert is 4-16 in every file
// that ships this, against 128+ for the FFN bank), so it is a plain 3-D tensor
// read with gemm_w over one expert's slice rather than an ExpertSource: no
// residency policy, no prefetch, nothing to evict. A row is `kvw` wide, so
// expert e starts at e*kvw rows.
bool Engine::mova_value(const LayerWeights &L, i32 l, const void *xn, i32 n,
                        i64 kvw, bool sigmoid_gate) {
    const ModelConfig &mc = model_.cfg();
    const i32 nv = mc.n_value_expert;
    const i32 k = mc.n_value_expert_used > 0 ? mc.n_value_expert_used : 1;
    if (nv <= 0 || !ws_vgate_ || !ws_vexp_) return false;
    gemm_w(ws_vgate_, xn, L.attn_v_gate, nv, n_embd_, n);
    be_->download_f32(vgate_host_.data(), ws_vgate_, static_cast<i64>(n) * nv);
    // One expert slice of the bank, re-pointed per selection. The stride is the
    // ROW BYTES of the weight's own dtype, not sizeof(f32): this is a quantized
    // tensor in every file that carries it.
    const i64 rb = static_cast<i64>(dtype_row_bytes(L.attn_v_exps.type,
                                                    L.attn_v_exps.n_in));
    be_->fill0(ws_v_, static_cast<size_t>(n) * static_cast<size_t>(kvw) *
                          be_->act_size());
    std::vector<i32> sel(static_cast<size_t>(k));
    std::vector<f32> wt(static_cast<size_t>(k));
    for (i32 t = 0; t < n; t++) {
        const f32 *logits = vgate_host_.data() + static_cast<i64>(t) * nv;
        // The decision is mmova's, not this loop's: MovaRoute::route is the
        // oracle the suite calls directly, so there is one implementation of
        // "bias selects, probability mixes, renorm divides by the selected
        // sum" and a test that pins it is pinning what the engine runs.
        MovaRoute::route(sel.data(), wt.data(), logits, L.attn_v_gate_b, nv, k,
                         sigmoid_gate, mc.expert_w_norm);
        for (i32 j = 0; j < k; j++) {
            if (sel[static_cast<size_t>(j)] < 0) continue;
            QuantTensor w = L.attn_v_exps;
            w.data = static_cast<u8 *>(L.attn_v_exps.data) +
                     static_cast<i64>(sel[static_cast<size_t>(j)]) * kvw * rb;
            w.n_out = kvw;
            gemm_w(ws_vexp_, xn, w, kvw, n_embd_, n);
            // silu on the expert's OUTPUT, then the weight, then accumulate.
            be_->silu_act(ws_vexp_, static_cast<i64>(n) * kvw);
            // Row t of V only: this token's routed value, times its weight.
            // A per-token axpy rather than a scatter is what keeps the loop
            // above honest -- each token picks its own experts.
            be_->axpy(act_at(ws_v_, be_->act_size(), static_cast<i64>(t) * kvw),
                      ws_vexp_, wt[static_cast<size_t>(j)], kvw);
        }
    }
    return true;
}

bool Engine::moe_ffn(const LayerWeights &L, i32 layer, i32 n) {
    KRK_TRACE_COMP("moe-ffn");
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    // ---- per-layer MoE geometry ----------------------------------------
    // A packed hybrid merge carries a different expert count, top-k and expert
    // width on every MoE block: Nemotron Bolt 74B-A9B has 240/304/368/432/496
    // experts and a per-layer top-k of 9 across 40 MoE blocks under one
    // declared `expert_count`. Everything below already reads `ne`, `k` and
    // `ff_exp`, so this is the only place the layer's own numbers have to
    // enter the forward; the workspaces are sized for the maxima.
    const i32 ne = model_.n_expert_at(layer);
    const i32 k = std::min<i32>(model_.n_expert_used_at(layer), ne);
    const i64 ff_exp = model_.n_ff_exp_at(layer);
    const i64 ff_sh = mc.n_ff_shexp;
    // The device plan and the one-launch dispatch both publish a UNIFORM
    // [n_layer x n_expert] table, so a heterogeneous stack keeps the host path
    // rather than reading that table with a 432-wide layer's ids.
    const bool uniform_moe = model_.moe_geometry_uniform();
    // Same reason for the expert scan: its [layer, expert] grid is one stride
    // of `ne`, so a heterogeneous stack would attribute one layer's mass to
    // another layer's cells.
    const bool scan_on = (expert_scan_ || expert_track_) && uniform_moe;
    // ---- LatentMoE --------------------------------------------------------
    // The routed bank of this family is NOT n_embd wide. `ffn_latent_down`
    // projects the block into the latent, the experts run there, and
    // `ffn_latent_up` brings the sum back (the shared expert below stays full
    // width, which is why only the routed leg moves). `ew` is the width every
    // expert-side buffer below works at, and `xin`/`acc` are the buffers it
    // reads and accumulates into: on every other family both are the widths
    // they have always been, so this changes nothing for them.
    const bool latent = L.latent_moe() && L.experts.n_embd > 0 &&
                        L.experts.n_embd != n_embd_; 
    const i64 ew = latent ? L.experts.n_embd : n_embd_;
    // The routed block normally reads the FFN-normed row the caller left in
    // `ws_xn_`. A two-branch family (gemma4) reads a DIFFERENT norm of the same
    // post-attention row, so the caller hands it over instead of recomputing it
    // here -- and every read of the activation below goes through `xin`.
    void *const xin = latent ? ws_xlat_
                             : (moe_in_ ? const_cast<void *>(moe_in_)
                                        : ws_xn_);
    void *const acc = latent ? ws_ffn_lat_ : ws_ffn_;

    be_->fill0(ws_ffn_, static_cast<size_t>(n) * static_cast<size_t>(n_embd_) * as);
    if (latent) {
        // One projection for the whole layer: every expert of every row reads
        // the SAME latent, so it is computed once here rather than per group.
        gemm_w(ws_xlat_, ws_xn_, L.latent_down, L.latent_down.n_out, n_embd_, n);
        be_->fill0(ws_ffn_lat_, static_cast<size_t>(n) * static_cast<size_t>(ew) * as);
    }
    // ... and the routed sum's storage exponents, zeroed once per layer, before
    // the first expert visit. `ws_fexp_` is where scatter_axpy_rows records the
    // power of two a row was stored under, and both folds below read it as that
    // row's exponent -- the same contract the residual stream's `ws_rexp_` has.
    be_->fill0(ws_fexp_, static_cast<size_t>(n) * sizeof(i32));

    // Layers below the file's `hash_layer_count` take their expert ids from
    // `ffn_gate_tid2eid.weight` instead of a top-k over the router. The count is
    // the metadata's and `L.tid2eid_per_token` is the tensor's, so a file whose
    // two disagree has a layer that routes by neither -- refused below rather
    // than routed by the wrong one.
    const bool hash_layer = layer < mc.dsv4_hash_layers;

    if (L.router.present() && k > 0 && ne > 0) {
    // Routing runs on the host: the logits are tiny ([n, n_expert]) and the
    // selection is what builds the plan.
    //
    // This GEMM runs on EVERY routed layer, including a DeepSeek-V4 hash layer.
    // The distinction those layers make is about the SELECTION, not the scores:
    // `ffn_gate_tid2eid.weight` names the experts by token id, and the router's
    // distribution is still what weights them. Skipping the GEMM there would
    // leave `moe_prob_` at whatever the previous layer wrote and mix the chosen
    // experts with another layer's numbers -- finite, fluent, wrong.
    KRK_TIME_HOST_N(n == 1 ? "moe.route" : "moe.route.batch", hs_route);
    {
        KRK_TIME_HOST_N(n == 1 ? "moe.route.gemm" : "moe.route.gemm.batch", hs_rg);
        // A detached router input (gemma4): the weightless norm of the
        // post-attention row, scaled and then multiplied elementwise by
        // `ffn_gate_inp.scale`. It is not either branch's normed input, which
        // is why it arrives precomputed rather than being derived here.
        const void *rin = L.router_scale ? ws_router_in_
                                        : (moe_in_ ? moe_in_ : ws_xn_);
        gemm_w(ws_router_, rin, L.router, ne, n_embd_, n);
    }

        // The blocking device-to-host copy below already orders itself behind
        // every kernel queued on the default stream, so the explicit
        // hipDeviceSynchronize this used to issue first was redundant. Forty of
        // them per decode token is not free on WDDM: each one is a driver-wide
        // drain that hands the thread back to the scheduler and takes it back.
        // KRK_ROUTER_SYNC=1 puts the explicit drain back, which is the arm the
        // cost of removing it is measured against.
        if (router_sync()) be_->sync();
        // ---- can this layer skip the router readback entirely? -------------
        // The one-launch dispatch never reads the router logits. It takes ids
        // and weights from `ws_ids_`/`ws_wts_`, which router_topk() fills in the
        // same breath as the device softmax, so on a layer that will dispatch
        // the [n, n_expert] f32 download below is bytes nothing ever reads --
        // and it is the layer's only blocking point. Skipping it is not a
        // measurement arm: it is dead traffic.
        //
        // `dispatch_now` is decided HERE, before the fetch, and it is a strict
        // subset of the `dispatch_dev` decision taken further down. Every term
        // of that decision that can be evaluated before the router is mirrored,
        // and `!hybrid_on_` stands in for `!hybrid`: hybrid is
        // `hybrid_on_ && ...`, so with the switch off it cannot be on, and a
        // layer satisfying dispatch_now satisfies dispatch_dev. Working it the
        // other way round -- fetch first, decide after -- is exactly the shape
        // that hides a stale consumer, so the subset relation is asserted in
        // the comment rather than checked at runtime.
        const bool gs_free = !L.experts.gate_scale && !L.experts.up_scale &&
                             !L.experts.down_scale;
        // The table can only be published once the whole routed set is in VRAM,
        // and on a cold cache that is never true at the end of init -- the
        // experts load lazily as routing reaches them. So the publish is
        // retried here, bounded: a model whose routed set does not fit pays a
        // few whole-set scans and then stops asking, instead of one per layer
        // for the rest of the run. It has to happen before the fetch, not after,
        // because the fetch is skipped on the strength of its answer.
        if (dev_ptr() && !dev_table_ && table_attempts_ < kTableAttempts) {
            ++table_attempts_;
            publish_expert_table();
        }
        auto dispatch_capable = [&]() {
            return dev_ptr() && !latent && be_->moe_dispatch_supported() &&
                   be_->router_topk_supported() &&
                   be_->router_softmax_supported() &&
                   (device_softmax() || dev_plan()) && dev_table_ != nullptr &&
                   model_.experts().dispatch_table() == dev_table_ &&
                   !model_.experts().dispatch_table_stale() &&
                   !expert_stub_ && !moe_probe() && uniform_moe &&
                   !mc.router_sigmoid &&
                   !(mc.router_bias_pre && L.router_bias) &&
                   !mc.ffn_swiglu_oai && !mc.ffn_relu_sqr &&
                   !L.exp_gate_bias && !L.exp_up_bias && !L.exp_down_bias &&
                   gs_free && (n_embd_ & 31) == 0 && (ff_exp & 31) == 0 &&
                   static_cast<size_t>(ff_exp) * 2u +
                           static_cast<size_t>(n_embd_) * 4u + 64u <=
                       48u * 1024u;
        };
        // ...unless something ELSE is watching the routing. Four readers take a
        // decode row's ids/probabilities from the host side, and every one of
        // them would be reading the PREVIOUS layer's numbers if the fetch were
        // skipped under it: the expert predictor, the next-layer prefetch hint,
        // the expert scan (it writes the hot-expert JSON), and KRK_DUMP_ROUTE.
        // So the readback is skipped only when the dispatch is its sole
        // consumer -- all four are opt-in, so the default configuration is the
        // one that gets the whole win.
        // Tracking is no longer one of them when the device can count instead:
        // the ids it needs are already in `ws_ids_` and the probabilities in
        // `ws_routerp_`, so the histogram is built there and read once at the
        // end. The condition is the same one that decides whether the
        // accumulator is called below (`dispatch_capable`, then buffers), which
        // is what keeps "skipped the readback" and "counted on the device" from
        // ever disagreeing.
        const bool scan_dev_ready =
            scan_on && scan_hits_dev_ != nullptr && scan_mass_dev_ != nullptr &&
            dispatch_capable();
        const bool route_consumed =
            moe_predict_enabled_ || next_prefetch_on() || dump_route_ ||
            (scan_on && !scan_dev_ready);
        const bool dispatch_now =
            !route_consumed && !hybrid_on_ && dispatch_capable() &&
            (n > 1 ||
             model_.experts().dispatch_table_host_cells() == 0 ||
             tile_decode()) &&
            (n <= 4 || tile_chunk() ||
             model_.experts().dispatch_table_host_cells() == 0);
        {
        // The fetch is its own span because it is the only blocking point in the
        // layer: inside it sits both the copy and the device work it waits for,
        // and the split between those two is what says whether the answer is a
        // smaller readback or a pipeline that stops draining.
        KRK_TIME_HOST_N(n == 1 ? "moe.route.fetch" : "moe.route.fetch.batch", hs_rf);
        // MEASUREMENT ONLY, and it changes the answer on purpose: skip the
        // readback and route every row to a fixed, in-order expert set. It is the
        // arm that says what the round trip costs by removing it outright, which
        // no refactor of the copy can: the fetch is a spin for work already
        // queued (`KRK_ROUTER_2X` measures the same copy on retired work at
        // 0.090 ms against 0.846 ms), so its price is mostly the device time it
        // exposes. With the fetch gone the step is the DEVICE floor plus the
        // host planning -- and only a difference between those two numbers can
        // justify moving the top-k onto the device.
        if (route_fake()) {
            for (i32 t = 0; t < n; t++) {
                f32 *row = router_host_.data() + static_cast<i64>(t) * ne;
                for (i32 e = 0; e < ne; e++)
                    row[e] = (e == layer % ne) ? 8.0f : 0.0f;
            }
        } else if (!dispatch_now) {
            be_->download_f32(router_host_.data(), ws_router_, static_cast<i64>(n) * ne);
        }
        // ...and when `dispatch_now` holds, neither branch runs: the span closes
        // empty, so `moe.route.fetch` reads ~0 ms on a dispatching layer. That
        // is the attribution the arm needs -- a number near zero is the proof
        // the readback is gone, not an assumption that it was.
        // KRK_ROUTER_2X: the same readback a second time. It is the control that
        // separates the two things inside `moe.route.fetch`: the copy's own round
        // trip (which this one pays in full) and the device work the first one
        // waits for (which is already retired by now). Without that split, "the
        // readback is 60% of the step" cannot say whether to shrink the copy or
        // to stop syncing -- those are different projects.
        if (router_fetch2()) {
            KRK_TIME_HOST_N(n == 1 ? "moe.route.fetch2" : "moe.route.fetch2.batch",
                            hs_rf2);
            be_->download_f32(router_host_.data(), ws_router_,
                              static_cast<i64>(n) * ne);
        }
        }
        // ---- the one-launch dispatch (KRK_DEV_PTR=1) --------------------
        // Decided before the plan arm because it is a STRONGER claim: the plan
        // arm still downloads the ids to verify them, and this one never asks
        // the host anything. Everything it needs is checkable here:
        //   * the cache published a table, which it only does when the whole
        //     routed set is device-resident -- so no promotion or eviction can
        //     move a pointer while the kernel is running;
        //   * no sidecar global scale (NVFP4): the kernel folds no `gs`, so a
        //     file that carries one keeps the per-expert path;
        //   * no per-expert FFN biases, no swiglu-oai, no relu^2 -- the kernel
        //     implements the plain silu(gate) * up MLP and nothing else;
        //   * the shape the launch can express: K a multiple of 32 (the dot
        //     chunking) and one activation row plus one accumulator row inside
        //     the block's 48 KiB of shared memory.
        // `dispatch_cap` is the capability half -- the same expression the fetch
        // above consulted as `dispatch_now`, minus the row-count gates, and
        // defined once (`dispatch_capable`) so the two decisions cannot drift
        // apart; the hybrid exclusion is added after `hybrid` is decided from
        // residency, further down.
        const bool dispatch_cap = dispatch_capable();
        // This layer's routing statistics are being counted on the device (see
        // Backend::scan_accum). The host plan below must then leave the
        // histograms alone -- and it is the same flag the accumulator call uses,
        // so "counted twice" and "counted nowhere" are the same edit to get
        // wrong, not two.
        const bool scan_dev_layer =
            dispatch_cap && scan_hits_dev_ != nullptr && scan_mass_dev_ != nullptr;

        // The device activation, for the numerics measurement. The host softmax
        // below still runs -- it is the reference the drift is measured against
        // -- and the device result replaces it before selection. Restricted to
        // the plain softmax case because a pre-bias enters the logits and a
        // sigmoid is a different function; those keep the host path and are the
        // next thing the dispatch has to cover, not something to guess at here.
        // The plan arm needs the device probabilities, so it implies this one.
        const bool ds = (device_softmax() || dev_plan()) &&
                        be_->router_softmax_supported() &&
                        !mc.router_sigmoid &&
                        !(mc.router_bias_pre && L.router_bias);
        if (ds) {
            be_->router_softmax(ws_routerp_, ws_router_, n, ne);
            // ---- the ids and weights the one-launch dispatch reads ----------
            // The dispatch picks each expert out of the DEVICE table, so this
            // row's ids and weights have to BE on the device, and this call is
            // what puts them there. The invariant is a subset relation, and it
            // is what makes the arm correct rather than fast-and-wrong:
            //
            //     dispatch_cap  =>  ds        (dispatch_capable() contains every
            //                                  term of ds, including
            //                                  `device_softmax() || dev_plan()`)
            //     dispatch_dev  =>  dispatch_cap  (its first conjunct)
            //
            // so a layer that dispatches has always been through this block and
            // written ws_ids_/ws_wts_. Reading uninitialised VRAM here was a real
            // bug once -- the arm ran at speed on degenerate text because the
            // kernel faithfully multiplied whatever bytes were there -- and the
            // conjunction above is what closed it. Same normalisation and scale
            // the host applies below, or the mix would differ from the path this
            // replaces by more than the router.
            if (dispatch_cap) {
                be_->router_topk(ws_routerp_, n, ne, k, ws_ids_, ws_wts_,
                                 mc.expert_w_norm != 0 ? 1 : 0,
                                 mc.expert_w_scale);
                // The routing statistics this layer produces, folded into the
                // device histogram while the ids and the probabilities are
                // still resident. Nothing comes back over the bus here: the
                // histogram is read once, after the run (scan_dev_fold).
                if (scan_dev_layer)
                    be_->scan_accum(ws_routerp_, ws_ids_, n, ne, k, layer,
                                    scan_hits_dev_, scan_mass_dev_);
            }
            // The one-launch dispatch keeps the probabilities on the device: a
            // full-row download for a drift report is the cost it exists to
            // remove, so `dispatch_cap` skips it (and the drift line is then
            // the device-plan arm's, not this one's).
            if (!dispatch_cap) {
                dev_probs_.resize(static_cast<size_t>(n) * static_cast<size_t>(ne));
                be_->download(dev_probs_.data(), ws_routerp_,
                              dev_probs_.size() * sizeof(f32));
            }
        }
        // Everything after the fetch: host top-k, the plan, the prefetch orders,
        // the per-expert loop and the host arm. It is the PARENT of
        // `moe.experts`, so the top-k+plan cost is this row minus that one -- a
        // span that ended where the plan ends would have to close the router
        // block early, and getting that wrong silently moves the expert loop's
        // time into a row called "plan".
        KRK_TIME_HOST_N(n == 1 ? "moe.route.rest" : "moe.route.rest.batch", hs_rp);

        // ---- build the routing plan -------------------------------------
        // moe_sel_/moe_wt_ are [token, slot]; the gate weights are renormalized
        // over the k selected experts (the Qwen convention).
        for (i32 t = 0; t < n; t++) {
            const f32 *logits = router_host_.data() + static_cast<i64>(t) * ne;
            // A file whose bias is `ffn_gate_inp.bias` means it in the LOGITS
            // (llama.cpp folds it into the router row before the gating func),
            // so the same numbers feed the softmax and the top-k. That is not
            // laguna's `exp_probs_b.bias`, which is added to the probabilities
            // below and therefore shifts selection only: adding a bias after a
            // monotone map is a different function, and with 128 experts on
            // gpt-oss it picks different experts.
            const f32 *rb = L.router_bias;
            const bool pre_bias = mc.router_bias_pre && rb != nullptr;
            if (pre_bias) {
                for (i32 e = 0; e < ne; e++)
                    moe_logit_[static_cast<size_t>(e)] =
                        logits[e] + rb[static_cast<size_t>(e)];
                logits = moe_logit_.data();
            }
            // The probability the selected experts are mixed with. Softmax is
            // the Qwen convention, sigmoid is laguna's (`expert_gating_func`
            // 2); sigmoid probabilities are not renormalized, which is exactly
            // what the reference does.
            if (mc.router_sigmoid) {
                for (i32 e = 0; e < ne; e++)
                    moe_prob_[static_cast<size_t>(e)] =
                        1.0f / (1.0f + std::exp(-logits[e]));
            } else {
                f32 mx = logits[0];
                for (i32 e = 1; e < ne; e++) mx = std::max(mx, logits[e]);
                f32 sum = 0;
                for (i32 e = 0; e < ne; e++) {
                    moe_prob_[static_cast<size_t>(e)] = std::exp(logits[e] - mx);
                    sum += moe_prob_[static_cast<size_t>(e)];
                }
                const f32 inv = sum > 0 ? 1.0f / sum : 0.0f;
                for (i32 e = 0; e < ne; e++)
                    moe_prob_[static_cast<size_t>(e)] *= inv;
            }
            if (ds && !dispatch_cap) {
                // The drift number, then the substitution: the selection below
                // must run on the DEVICE probabilities, or the text is the
                // host's and the measurement says nothing about a device path.
                const f32 *dp = dev_probs_.data() + static_cast<i64>(t) * ne;
                f64 worst = 0.0;
                u64 bits = 0;
                for (i32 e = 0; e < ne; e++) {
                    const f32 a = moe_prob_[static_cast<size_t>(e)];
                    const f32 b = dp[e];
                    const f64 d = std::fabs(static_cast<f64>(a) - static_cast<f64>(b));
                    if (d > worst) worst = d;
                    if (a != b) bits++;
                }
                ds_rows_++;
                ds_diff_ += bits;
                if (worst > ds_max_) ds_max_ = worst;
                for (i32 e = 0; e < ne; e++)
                    moe_prob_[static_cast<size_t>(e)] = dp[e];
            }

            // Fold this token's routing distribution into the scan. Mass is
            // recorded for every expert, not just the k selected, because the
            // question the index answers is "what coverage does the top N buy",
            // and that needs the whole curve.
            // ...unless the device is already counting them: see scan_dev_fold.
            // The two must never both count, or every expert is credited twice.
            if (scan_on && !scan_dev_layer) {
                double *mass = expert_mass_.data() +
                               static_cast<size_t>(layer) * static_cast<size_t>(ne);
                for (i32 e = 0; e < ne; e++)
                    mass[e] += static_cast<double>(moe_prob_[static_cast<size_t>(e)]);
            }

            // The per-expert bias (`ffn_exp_probs_b.bias`) shifts *selection*
            // only: the weights stay the unbiased probabilities, which is what
            // the reference means by "leave probs unbiased as it's later used
            // to get expert weights". Adding it to the weights as well is a
            // plausible-looking and wrong change -- and so is applying it at
            // all when it already entered the logits above (`pre_bias`).
            auto sel_score = [&](i32 e) {
                return moe_prob_[static_cast<size_t>(e)] +
                       ((rb && !pre_bias) ? rb[static_cast<size_t>(e)] : 0.0f);
            };
            // ---- group-limited selection ----------------------------------
            // DeepSeek-V3's rule (llama-graph.cpp, "select top n_group_used
            // expert groups"): the experts are contiguous groups, a group scores
            // as the sum of its best two SELECTION scores, only the best
            // `group_used` groups stay eligible, and the top-k runs over the
            // survivors. The bias is already inside sel_score (it is a selection
            // score), and the WEIGHTS stay the unbiased probabilities -- which is
            // why this masks the eligibility and never touches moe_prob_.
            //
            // Masked experts are made unselectable rather than re-sorted: the
            // loop below picks the best available score, so a -inf here is
            // enough and every downstream index stays the file's own.
            u8 *group_keep = moe_group_keep_.empty() ? nullptr
                                                    : moe_group_keep_.data();
            bool grouped = false;
            if (mc.n_expert_group > 1 && group_keep) {
                std::vector<f32> sels(static_cast<size_t>(ne));
                for (i32 e = 0; e < ne; e++) sels[static_cast<size_t>(e)] = sel_score(e);
                if (!krk::moe_group_mask_or_all(moe_group_keep_.data(),
                                                sels.data(), ne,
                                                mc.n_expert_group,
                                                mc.n_expert_group_used)) {
                    forward_error_ = "layer " + std::to_string(layer) +
                                     ": the group-limited route refused its own "
                                     "partition (ne=" + std::to_string(ne) +
                                     ", groups=" +
                                     std::to_string(mc.n_expert_group) + ", used=" +
                                     std::to_string(mc.n_expert_group_used) + ")";
                    return false;
                }
                grouped = true;
            }
            auto eligible = [&](i32 e) {
                return !grouped || group_keep[static_cast<size_t>(e)] != 0;
            };

            i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
            f32 *wt = moe_wt_.data() + static_cast<size_t>(t) * k;
            // ---- DeepSeek-V4 hash route ----------------------------------
            // The table is authority for this layer's ids, and the router's
            // probability of each named expert is what mixes it. `dsv4_hash_gather`
            // reads `moe_prob_` in TABLE order -- a caller that sorted by id would
            // pair expert A's weight with expert B's output, which is in range,
            // stable and undetectable downstream. Both refuse rather than write a
            // partial route, so a corrupt table cannot poison half a plan.
            const bool hash_ids = hash_layer && L.tid2eid_per_token > 0;
            if (hash_ids) {
                if (L.tid2eid_per_token != k) {
                    forward_error_ = "layer " + std::to_string(layer) +
                                     ": hash table names " +
                                     std::to_string(L.tid2eid_per_token) +
                                     " expert(s) per token but the plan has room "
                                     "for " + std::to_string(k);
                    return false;
                }
                i32 hid[64];
                f32 hwt[64];
                // The table is indexed by the row's VOCAB id, not by its
                // position in this pass. Missing `fwd_toks_` would mean a caller
                // that reached a hash layer without a forward pass, and routing
                // by position there is the defect this check exists for.
                const i32 tok = (fwd_toks_ && t < fwd_toks_n_) ? fwd_toks_[t] : -1;
                if (k > 64 || tok < 0 ||
                    !krk::dsv4_hash_route(hid, layer, mc.dsv4_hash_layers,
                                          L.tid2eid.data(), L.tid2eid_per_token,
                                          L.tid2eid_n_tokens, tok, ne) ||
                    !krk::dsv4_hash_gather(hwt, hid, k, moe_prob_.data(), ne)) {
                    forward_error_ = "layer " + std::to_string(layer) +
                                     ": hash route refused at token " +
                                     std::to_string(t);
                    return false;
                }
                for (i32 s = 0; s < k; s++) {
                    sel[s] = hid[s];
                    wt[s] = hwt[s];
                    if (scan_on && !scan_dev_layer && hid[s] >= 0)
                        expert_hits_[static_cast<size_t>(layer) *
                                         static_cast<size_t>(ne) +
                                     static_cast<size_t>(hid[s])] += 1;
                }
            }
            // A hash layer's ids came from the table, so the top-k below must not
            // run: it would overwrite the route with the distribution's own.
            // Residency of an expert, as the tier sees it right now: 2 in VRAM,
            // 1 in WARM, 0 nowhere. The expert index is what FILLED those tiers,
            // so this reads the index's answer live -- and correctly, which a
            // stored rank cannot be once the tiers have evicted and re-admitted
            // for a while.
            auto resident_rank = [&](i32 e) {
                if (!router_pref_) return 0;
                if (model_.experts().in_vram(layer, e)) return 2;
                if (model_.experts().in_host(layer, e)) return 1;
                return 0;
            };
            for (i32 s = 0; !hash_ids && s < k; s++) {
                // The top available score first, so "near-equal" has something
                // to be near. With the preference off this is the only pass and
                // the arithmetic below reduces to the loop it replaced.
                f32 top = -1.0f;
                for (i32 e = 0; e < ne; e++) {
                    if (!eligible(e)) continue;
                    bool taken = false;
                    for (i32 p = 0; p < s; p++)
                        if (sel[p] == e) { taken = true; break; }
                    if (taken) continue;
                    const f32 sc = sel_score(e);
                    if (sc > top) top = sc;
                }
                const f32 cut = top - (router_pref_ ? router_tie_margin_ * top : 0.0f);
                i32 best = -1;
                f32 bestp = -1.0f;
                int best_rank = -1;
                for (i32 e = 0; e < ne; e++) {
                    if (!eligible(e)) continue;
                    bool taken = false;
                    for (i32 p = 0; p < s; p++)
                        if (sel[p] == e) { taken = true; break; }
                    if (taken) continue;
                    const f32 sc = sel_score(e);
                    if (sc < cut) continue; // not near-equal with the best
                    const int rank = resident_rank(e);
                    // Residency decides first, then the score -- and a score tie
                    // keeps the lower id, which is what the old loop did.
                    if (rank > best_rank || (rank == best_rank && sc > bestp)) {
                        best_rank = rank;
                        bestp = sc;
                        best = e;
                    }
                }
                if (best >= 0) {
                    // Counted in both arms: the share of picks that hit VRAM is
                    // what the preference is for, and the control has to be able
                    // to show it too.
                    if (model_.experts().in_vram(layer, best)) router_hot_++;
                    if (router_pref_) {
                        router_picks_++;
                        // Exact compare: the top scorer's own score IS `top`, so
                        // a pick below it was moved by residency and nothing else.
                        if (bestp < top) router_subs_++;
                    }
                }
                sel[s] = best;
                // The weight is the probability, never the biased score.
                wt[s] = best >= 0 ? moe_prob_[static_cast<size_t>(best)] : 0.0f;
                if (scan_on && !scan_dev_layer && best >= 0)
                    expert_hits_[static_cast<size_t>(layer) *
                                     static_cast<size_t>(ne) +
                                 static_cast<size_t>(best)] += 1;
            }
            // Sum-normalize, then scale. llama.cpp does both only when the
            // file asks for them, but this engine normalized unconditionally
            // before the flag existed and no model in the collection declares
            // 0, so absent (-1) keeps normalizing. laguna declares 1 and a
            // scale of 2.5, which is the pair that has to reach the mix.
            if (mc.expert_w_norm != 0) {
                f32 ksum = 0;
                for (i32 s = 0; s < k; s++) ksum += wt[s];
                const f32 kinv = ksum > 0 ? 1.0f / ksum : 0.0f;
                for (i32 s = 0; s < k; s++) wt[s] *= kinv;
            }
            if (mc.expert_w_scale != 0.0f && mc.expert_w_scale != 1.0f)
                for (i32 s = 0; s < k; s++) wt[s] *= mc.expert_w_scale;
            // KRK_DUMP_ROUTE=1: the routing decision itself, which is the only
            // way to tell a wrong router from a wrong expert read -- the two
            // look identical in moe.sum and one of them is a five-line fix.
            if (dump_route_) {
                std::printf("[route] L%02d t=%d", layer, t);
                for (i32 s = 0; s < k; s++)
                    std::printf(" e=%d w=%.6f", sel[s], wt[s]);
                const f32 *l0 = router_host_.data() + static_cast<i64>(t) * ne;
                std::printf(" | top logits");
                for (i32 s = 0; s < k; s++)
                    std::printf(" %.4f",
                                (mc.router_bias_pre && rb) ? l0[sel[s]] + rb[sel[s]]
                                                           : l0[sel[s]]);
                std::printf("\n");
            }
        }

        // ---- grouped expert GEMMs ----------------------------------------
        // Every (token, slot) pair is one work item. Items that picked the same
        // expert share one permuted activation block and one set of GEMMs, so a
        // layer costs O(k) GEMMs per matrix instead of O(k * n).
        //
        // The stub skips them all, leaving ws_ffn_ zeroed. The routers above
        // have already run and still saw the true dense-path activation, so the
        // statistics are collected; only the layer's own expert contribution to
        // the residual is missing, which is what makes the list approximate.
        if (expert_stub_) expert_stubbed_ran_ = true;

        // ---- hybrid CPU + GPU expert compute: who computes what ------------
        //
        // The split is decided per (layer, expert), from residency, BEFORE any
        // acquire() runs. acquire() promotes, and the entire point is not to
        // promote the experts the host is about to compute: a resident expert
        // stays on the device because its bytes are already there, and an expert
        // that would have been a read + a promotion is computed where its bytes
        // already are instead -- the mapping. For one decode row that trades a
        // ~1.95 MiB weight move for an 8 KiB activation move, and it costs no
        // VRAM and no WARM RAM for the experts it takes, which is what frees
        // room for the KV cache.
        //
        // The shape gate mirrors CpuExpertPool::run's own rejection: a bank the
        // host cannot walk must not be classified for the host, because the
        // alternative to computing an expert is not losing it silently.
        const bool hybrid =
            hybrid_on_ && !expert_stub_ && k > 0 && n <= hybrid_max_rows_ &&
            L.experts.present() && L.experts.gate &&
            L.experts.n_expert == ne &&
            L.experts.n_embd == n_embd_ &&            L.experts.n_ff_exp > 0 &&
            L.experts.gate->n_dims == 3 &&
            L.experts.up->n_dims == 3 &&
            L.experts.down->n_dims == 3 &&
            // nemotron_h_moe has no ffn_gate_exps (gate == null): the geometry
            // contract lives on up/down alone, which every schema carries.
            L.experts.up->type == L.experts.down->type &&
            static_cast<i64>(L.experts.up->ne[0]) == n_embd_ &&
            static_cast<i64>(L.experts.up->ne[1]) == L.experts.n_ff_exp &&
            static_cast<i64>(L.experts.down->ne[0]) == L.experts.n_ff_exp &&
            static_cast<i64>(L.experts.down->ne[1]) == n_embd_ &&
            dtype_row_aligned(L.experts.up->type, n_embd_) &&
            dtype_row_aligned(L.experts.up->type, L.experts.n_ff_exp);
        // A layer that asked for the dispatch and did not get it is the one thing
        // a "the arm was on and the numbers did not move" report has to be able
        // to rule out. dispatch_dev is the path chosen when the device dispatch is
        // on (KRK_DEV_PTR=1, table published, whole routed set resident, shape
        // fits the kernel) and the hybrid path is not in use.
        // A single row against a HOST-backed table used to be refused: the table
        // kernel launched `n_tok` blocks and walked a token's k experts inside its
        // own block, so at n = 1 a 64-CU card ran one CU. Measured on
        // Qwen3.5-35B-A3B (6520 of 10240 cells on the host): 2.51 ms/layer of
        // device time and 154.9 ms/step against 127.5 for the per-expert path.
        // The tiled kernels give one token's slots hundreds of blocks instead of
        // eight, so that objection is gone and the arm is allowed at n = 1
        // (`KRK_MOE_TILE_DECODE=0` restores the old exclusion for an A/B).
        const bool decode_shape_ok = n > 1 ||
                                     model_.experts().dispatch_table_host_cells() == 0 ||
                                     tile_decode();
        // The table kernel computes one activation ROW at a time, where a prefill
        // chunk is a batch GEMM -- measured on this model as 55 tok/s against the
        // per-expert path's 144. So a host-backed table serves only the small row
        // sets it is now good at, and a chunk keeps the path that reads the same
        // experts in place through the per-expert loop. With every cell in VRAM
        // there is no host read to miss, and the batch case keeps the dispatch.
        const bool small_rows = n <= 4 || tile_chunk() ||
                                model_.experts().dispatch_table_host_cells() == 0;
        // `moe_dispatch`'s kernel folds the silu pair too, so a GeGLU bank
        // keeps the per-expert loop.
        const bool dispatch_dev = dispatch_cap && decode_shape_ok && small_rows &&
                                  !hybrid && !mc.gelu_ffn;
        if (dev_table_ && !dispatch_dev)
            (n > 4 && model_.experts().dispatch_table_host_cells() != 0
                 ? dp_chunk_skips_
                 : dp_launch_skips_)++;
        // Read WARM in place instead of promoting it (ExpertCache::acquire_inplace).
        // The other two arms answer "where do these bytes live" for themselves:
        // the hybrid arm computes non-resident experts on the host, and the
        // one-launch dispatch reads a published table, so neither is overridden
        // here.
        const bool inplace_ok = expert_inplace() && be_->host_load_supported() &&
                                !hybrid && !dispatch_dev && !expert_stub_;

        // ---- start THIS layer's cold reads while its first experts run -------
        // Not the NEXT layer: the previous token's picks for the next layer are
        // already in WARM by definition (they were just used), so a read-ahead
        // fed with them issues zero reads -- measured, `hint=10 ids=0` on every
        // layer of REAP-288. What blocks is the expert this layer routed that the
        // tier does NOT hold, and those ids are exact, not predicted: they are on
        // the host right here, after selection and before the loop that waits for
        // them.
        //
        // So the reads start now, on the worker, and the loop's first experts --
        // the ones already resident or in place -- compute while the file works.
        // `take_prefetched` waits for an expert that is still in flight, so the
        // order of the layer's arithmetic is untouched: same experts, same
        // ascending order, same scatter. Only the wall clock changes.
        if (warm_ahead() && inplace_ok && !expert_stub_) {
            ahead_ids_.clear();
            for (i32 t2 = 0; t2 < n; t2++) {
                const i32 *sel2 = moe_sel_.data() + static_cast<size_t>(t2) * k;
                for (i32 s2 = 0; s2 < k; s2++) {
                    const i32 e = sel2[s2];
                    if (e < 0 || e >= ne) continue;
                    if (model_.experts().has_warm(layer, e)) continue;
                    if (std::find(ahead_ids_.begin(), ahead_ids_.end(), e) !=
                        ahead_ids_.end())
                        continue;
                    ahead_ids_.push_back(e);
                }
            }
            if (!ahead_ids_.empty())
                model_.experts().prefetch_warm_next(
                    L.experts, layer, ahead_ids_.data(),
                    static_cast<int>(ahead_ids_.size()));
        }

        i32 n_cpu_experts = 0;
        if (hybrid) {
            hy_take_.assign(static_cast<size_t>(ne), 0);
            const ExpertCache &ec = model_.experts();
            for (i32 e = 0; e < ne; e++) {
                bool routed = false;
                for (i32 t = 0; t < n && !routed; t++) {
                    const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                    for (i32 s = 0; s < k; s++)
                        if (sel[s] == e) { routed = true; break; }
                }
                if (!routed) continue;
                // The host arm takes the OVERFLOW, not every miss: a miss the
                // device tier can still afford is promoted exactly as it always
                // was, so a budget large enough for the routed set keeps every
                // expert resident and the host arm never runs. Taking every
                // miss instead is a livelock the first measurement showed -- an
                // expert that is never promoted never becomes resident, so the
                // cache stayed at 55/104 of a budget that could have held all of
                // it and decode fell from 99.3 to 6.4 tok/s.
                bool cpu = !ec.in_vram(layer, e) && ec.full_for(L.experts);
                // A deterministic share of the resident set can be forced to
                // the host as well, which is how the two arms are swept against
                // each other at a fixed cache size (--hybrid-frac). The hash is
                // over (layer, expert) and not a counter, so the same experts
                // move together layer after layer.
                if (!cpu && hybrid_frac_permille_ < 1000) {
                    const u32 hsh = static_cast<u32>(layer) * 2654435761u +
                                    static_cast<u32>(e) * 40503u;
                    if (static_cast<int>(hsh % 1000u) < hybrid_frac_permille_)
                        cpu = true;
                }
                if (cpu) {
                    hy_take_[static_cast<size_t>(e)] = 1;
                    n_cpu_experts++;
                }
            }
        }

        // One batched WARM read for the whole routed set before the per-expert
        // loop starts. Each acquire() below otherwise issues its three slices
        // one at a time through a single file handle -- queue depth 1, which on
        // this machine is 593 MiB/s against 1446 MiB/s at four outstanding
        // reads and 3400 MiB/s sequential. The set is deduped because two
        // tokens selecting one expert must not read it twice.
        // The one-launch dispatch keeps the ids on the device, so there is no
        // routed set here to prefetch from; a hybrid run that also asked for
        // the dispatch simply has no host selection to order.
        if (!expert_stub_ && k > 0 && !dispatch_cap) {
            prefetch_seen_.assign(static_cast<size_t>(ne), 0);
            prefetch_ids_.clear();
            for (i32 t = 0; t < n; t++) {
                const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                for (i32 s = 0; s < k; s++) {
                    const i32 e = sel[s];
                    if (e < 0 || e >= ne || prefetch_seen_[static_cast<size_t>(e)])
                        continue;
                    // An expert the host arm owns is read out of the mapping by
                    // the host, so staging it into WARM would be a second read
                    // of the same bytes AND a resident buffer for something that
                    // already has a home. Leaving it out is half of the RAM this
                    // feature gives back.
                    if (hybrid && hy_take_[static_cast<size_t>(e)]) continue;
                    prefetch_seen_[static_cast<size_t>(e)] = 1;
                    prefetch_ids_.push_back(e);
                }
            }
            // ---- top-k' lookahead ------------------------------------------
            // The router's whole row is on the host (`moe_prob_`, plus the same
            // selection bias the top-k used), so the experts ranked k+1..k' cost
            // nothing to read. They are staged only if WARM already holds them
            // (stage_batch skips a cold one), so this pays a copy for a guess
            // and never a file read. Single row only: with n > 1 `moe_prob_`
            // holds the LAST token's row, and deriving a batch's runners-up from
            // one token's distribution is a different, unmeasured rule.
            if (expert_lookahead_ > k && n == 1) {
                const f32 *rb2 = L.router_bias;
                const bool post_bias = rb2 != nullptr && !mc.router_bias_pre;
                const i32 want = expert_lookahead_ < ne ? expert_lookahead_ : ne;
                for (i32 pick = k; pick < want; pick++) {
                    i32 best = -1;
                    f32 bestp = -1.0f;
                    for (i32 e = 0; e < ne; e++) {
                        if (prefetch_seen_[static_cast<size_t>(e)]) continue;
                        const f32 sc =
                            moe_prob_[static_cast<size_t>(e)] +
                            (post_bias ? rb2[static_cast<size_t>(e)] : 0.0f);
                        if (sc > bestp) {
                            bestp = sc;
                            best = e;
                        }
                    }
                    if (best < 0) break;
                    // Marked before the host-arm test: an expert the CPU owns
                    // must not be picked again by the next runner-up pass.
                    prefetch_seen_[static_cast<size_t>(best)] = 1;
                    if (hybrid && hy_take_[static_cast<size_t>(best)]) continue;
                    prefetch_ids_.push_back(best);
                }
            }
            // The read-ahead worker is already reading exactly this set (it was
            // handed the ids at selection time). Issuing the same reads here as
            // well is what made the first version SLOWER instead of faster: two
            // copies of the bytes, the worker's results thrown away as duplicates
            // because the synchronous pass had already filled WARM.
            const bool ahead_reading = warm_ahead() && inplace_ok && !expert_stub_;
            if (prefetch_ids_.size() > 1 && !ahead_reading)
                model_.experts().prefetch_layer(
                    L.experts, layer, prefetch_ids_.data(),
                    static_cast<int>(prefetch_ids_.size()));
            // ---- promote the routed set in ONE upload ---------------------
            // The WARM hits this layer is about to take are known here, before
            // any of them runs, so they are handed to the cache as one set and
            // it issues one batch per source kind instead of one call per
            // expert. Same bytes, same order per expert, same arithmetic: only
            // the number of enqueues changes.
            //
            // Bounded like stage_batch, and for the same reason: with n > 1 the
            // union of a wider row can be hundreds of experts, and promoting
            // all of them would evict the ones the first row needs.
            const size_t batch_max = static_cast<size_t>(k) * 4 + 8;
            if (batch_promote() && !inplace_ok && !prefetch_ids_.empty() && n <= 8 &&
                prefetch_ids_.size() <= batch_max)
                model_.experts().promote_batch(L.experts, layer,
                                               prefetch_ids_.data(),
                                               static_cast<int>(
                                                   prefetch_ids_.size()));
            // ---- promote the plan in ONE pass, before the per-expert loop ----
            // Without this every routed expert that is not resident is promoted
            // INSIDE the loop, between two GEMMs: alloc, room walk and H2D each
            // serial with compute. Staging first enqueues them all, so the copies
            // overlap the first expert's work and the rest are already in flight
            // when the loop reaches them. It is the same bytes in the same order
            // per expert, so the arithmetic does not move -- only the schedule.
            //
            // Bounded on purpose: a 256-row prefill has a union of hundreds of
            // experts, and promoting all of them would evict the very ones its
            // own first rows need. KRK_MOE_LOOKAHEAD=0 is the A/B arm.
            const size_t stage_max = static_cast<size_t>(k) * 4 + 8;
            if (lookahead_stage() && !inplace_ok && !prefetch_ids_.empty() && n <= 4 &&
                prefetch_ids_.size() <= stage_max)
                model_.experts().stage_batch(L.experts, layer,
                                             prefetch_ids_.data(),
                                             static_cast<int>(prefetch_ids_.size()));
        }

        // ---- one routing plan for the whole layer ---------------------------
        // The (token, slot) pairs are bucketed by expert once per layer, so a
        // visit reads [off(e), off(e+1)) of a plan that cost TWO copies instead
        // of two per expert. The row order is what the per-expert rescan below
        // collected -- tokens ascending, then slots ascending -- and the
        // bucketing is the same O(n*k) walk with a prefix sum, not the
        // O(n_expert * n * k) rescan it replaces. `KRK_MOE_PLAN=0` takes the
        // old path and is the arm the fold is measured against.
        // `dispatch_dev` builds no host plan at all -- the kernel reads the ids
        // and the weights itself -- so the host plan is the fallback's, and the
        // per-expert loop below is skipped entirely when it engaged.
        const bool plan_grouped = ((moe_plan_grouped() && !dispatch_dev) ||
                                   dp_used_ > 0) &&
                                  !expert_stub_;
        if (plan_grouped && dp_used_ == 0) {
            plan_off_.assign(static_cast<size_t>(ne) + 1, 0);
            for (i32 t = 0; t < n; t++) {
                const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                for (i32 s = 0; s < k; s++) {
                    const i32 e = sel[s];
                    if (e >= 0 && e < ne) plan_off_[static_cast<size_t>(e) + 1]++;
                }
            }
            for (i32 e = 0; e < ne; e++)
                plan_off_[static_cast<size_t>(e) + 1] +=
                    plan_off_[static_cast<size_t>(e)];
            plan_dev_.resize(static_cast<size_t>(plan_off_[static_cast<size_t>(ne)]));
            alpha_dev_.resize(plan_dev_.size());
            plan_cur_.assign(plan_off_.begin(), plan_off_.end() - 1);
            for (i32 t = 0; t < n; t++) {
                const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                const f32 *wt = moe_wt_.data() + static_cast<size_t>(t) * k;
                for (i32 s = 0; s < k; s++) {
                    const i32 e = sel[s];
                    if (e < 0 || e >= ne) continue;
                    const i32 at = plan_cur_[static_cast<size_t>(e)]++;
                    plan_dev_[static_cast<size_t>(at)] = t;
                    alpha_dev_[static_cast<size_t>(at)] = wt[s];
                }
            }
            if (!plan_dev_.empty()) {
                KRK_TIME_HOST("moe.route.upload");
                be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(),
                                static_cast<i64>(plan_dev_.size()));
                be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                            plan_dev_.size() * sizeof(f32));
            }
        }

        // ---- the whole routed layer from ONE launch (KRK_DEV_PTR=1) ---------
        // No acquire(), no per-expert plan slice, no host selection: the kernel
        // walks this token's k slots and reads each expert's three matrices out
        // of the table. The routed sum lands in ws_ffn_, which is exactly what
        // the per-expert scatter would have built, so the residual add after
        // this function needs no knowledge of which path ran.
        // ---- the next token's hint for this layer --------------------------
        // Recorded before the experts run, so a layer that fails to acquire one
        // still contributes its routing. Decode rows only: with n > 1
        // moe_sel_ holds a batch, and a single previous-token set would be a
        // worse predictor than the plan the batch path already stages.
        // Do not run the legacy previous-layer hint alongside the adaptive
        // predictor: they target the same WARM->HOT slots and duplicate staging
        // makes the predictor's A/B uninterpretable.
        const bool next_prefetch = next_prefetch_on() && !moe_predict_enabled_ &&
                                   n == 1 && !expert_stub_;
        if (moe_predict_enabled_ && n == 1 && !expert_stub_)
            moe_predictor_.observe(layer, moe_sel_.data(), k);
        // The hint is recorded whenever SOMETHING reads it: the WARM->HOT
        // staging below, or the read-ahead worker (which uses it to know what to
        // read for the next layer while this one computes). Recording it only
        // under `next_prefetch` is how the read-ahead first ran and did nothing
        // at all -- the vector was never sized, so every layer found no hint and
        // the blocking read stayed exactly where it was.
        if (next_prefetch) {
            if (moe_hint_.size() < static_cast<size_t>(mc.n_layer))
                moe_hint_.resize(static_cast<size_t>(mc.n_layer));
            std::vector<i32> &hint = moe_hint_[static_cast<size_t>(layer)];
            hint.clear();
            for (i32 s = 0; s < k; s++) {
                const i32 e = moe_sel_[static_cast<size_t>(s)];
                if (e < 0 || e >= ne) continue;
                bool dup = false;
                for (i32 prev : hint)
                    if (prev == e) { dup = true; break; }
                if (!dup) hint.push_back(e);
            }
        }

        if (dispatch_dev) {
            KRK_TIME_HOST_N(n == 1 ? "moe.dispatch" : "moe.dispatch.batch", hs_disp);
            be_->moe_dispatch(dev_table_, layer, ne, ws_ids_, ws_wts_, xin,
                              ws_ffn_, n, n_embd_, ff_exp, k);
            ++dp_launch_layers_;
            // That one launch read every routed expert's cells, and the per-expert
            // loop below (which does this per use) does not run. Stamp them here
            // or a reuse would adopt a mark older than this read. Deduped, because
            // a prefill row set selects the same expert many times.
            prefetch_seen_.assign(static_cast<size_t>(ne), 0);
            for (i32 t2 = 0; t2 < n; t2++) {
                const i32 *sel2 = moe_sel_.data() + static_cast<size_t>(t2) * k;
                for (i32 s2 = 0; s2 < k; s2++) {
                    const i32 e2 = sel2[s2];
                    if (e2 < 0 || e2 >= ne || prefetch_seen_[static_cast<size_t>(e2)])
                        continue;
                    prefetch_seen_[static_cast<size_t>(e2)] = 1;
                    model_.experts().note_read(layer, e2);
                }
            }
        }

        // ---- per-expert loop on the device too? no - this is the dispatch path ---
        // When dispatch_dev is true the kernel already did the whole layer; the
        // per-expert loop below is skipped by `!dispatch_dev` in its header.

        // The per-expert loop, timed as its own span: with 8 experts of 6 launches
        // this is where the launch fan-out and the per-expert acquire() live, and
        // a step that is launch-bound shows up here and nowhere else.
        //
        // NOTE: dispatch_dev is true only when KRK_DEV_PTR=1 is on AND the table
        // is published AND the whole routed set is resident AND the shape fits the
        // kernel; otherwise the per-expert path is used and expertise is resolved
        // per expert via acquire().
        // ---- the whole layer as a batch (decode) ---------------------------
        // `use_plan_off` is the loop's own test for "this layer's plan is already
        // on the device", which is what a batch scatters through; a layer that
        // builds its plan per visit keeps the loop. Everything else here is the
        // shape the launches can express -- plain silu experts with no per-expert
        // bias, no sidecar scale, no host arm and no arm that measures a
        // per-expert ordering (KRK_MOE_BATCH=0 is this path's own arm).
        const bool use_plan_off0 = plan_grouped && dp_used_ == 0;
        // The batched path's epilogue is a silu, so a GeGLU bank (gemma4) must
        // not take it: the launch would compute the right projections and the
        // wrong function of them.
        const bool batch_ok =
            n == 1 && !latent && moe_batch() && use_plan_off0 && !expert_stub_ &&
            !mc.gelu_ffn &&
            !dispatch_dev && !hybrid && !moe_probe() && !sync_expert() &&
            !moe_silu_fused() && !moe_active_stage() &&
            L.experts.gate && gs_free &&
            !L.exp_gate_bias && !L.exp_up_bias && !L.exp_down_bias &&
            !mc.ffn_swiglu_oai && !mc.ffn_relu_sqr &&
            (ff_exp & 31) == 0 && (n_embd_ & 31) == 0;
        const bool batch_ran =
            batch_ok && moe_decode_batch(L, layer, k, ne, ff_exp, use_plan_off0,
                                         inplace_ok);
        if (n == 1 && moe_batch()) {
            if (batch_ran) ++moe_batch_layers_;
            else ++moe_batch_skips_;
        }

        {
        KRK_TIME_HOST_N(n == 1 ? "moe.experts" : "moe.experts.batch", hs_moe);
        for (i32 e = 0; !batch_ran && !expert_stub_ && !dispatch_dev && e < ne;
             e++) {
            // The host arm owns this expert on this layer. Skipping it here is
            // what leaves its weights absent from both tiers.
            if (hybrid && hy_take_[static_cast<size_t>(e)]) continue;
            // Grouped-plan fast path (P0-2): plan_off_ already holds this
            // expert's row count, so skip the O(n*k) host rescan. Only when
            // the offsets were built for this pass (dp_used_ == 0); the
            // draft-reuse path keeps the legacy scan.
            const bool use_plan_off = plan_grouped && dp_used_ == 0;
            i64 m_pre = 0;
            if (use_plan_off) {
                m_pre = static_cast<i64>(plan_off_[static_cast<size_t>(e) + 1] -
                                         plan_off_[static_cast<size_t>(e)]);
                if (m_pre == 0) continue;
            } else {
                group_rows_.clear();
                group_wt_.clear();
                for (i32 t = 0; t < n; t++) {
                    const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                    const f32 *wt = moe_wt_.data() + static_cast<size_t>(t) * k;
                    for (i32 s = 0; s < k; s++) {
                        if (sel[s] == e) {
                            group_rows_.push_back(t);
                            group_wt_.push_back(wt[s]);
                        }
                    }
                }
                if (group_rows_.empty()) continue;
            }

            // Pages the expert in if it is not resident; identical results
            // regardless of how small the cache is. A refusal is a visible
            // generation error (P0-1), never a silent skip: the router
            // selected this expert and its contribution is required.
            // In place first: a WARM expert that has not earned a VRAM slot is
            // read where it lives, which spends no copy and evicts nothing. That
            // only works when the slot is pinned host memory -- `acquire_inplace`
            // answers nullptr for every other case and the promotion path below
            // runs exactly as before.
            // A read-ahead that finished is this expert's WARM copy; installing
            // it here costs a map insert and no I/O. A miss is the normal path.
            if (inplace_ok) model_.experts().take_prefetched(layer, e);
            const ResidentExpert *re =
                inplace_ok ? model_.experts().acquire_inplace(L.experts, layer, e)
                           : nullptr;
            if (!re) re = model_.experts().acquire(L.experts, layer, e);
            if (!re) {
                forward_error_ = expert_resource_error() + " (moe layer " +
                                 std::to_string(layer) + " expert " +
                                 std::to_string(e) + ")";
                return false;
            }
            if (sync_expert()) be_->sync();
            if (hybrid) hy_gpu_experts_++;

            // The per-expert global scale each NVFP4 matrix needs.
            const f32 gs_up = L.experts.scale_of(L.experts.up_scale, e);
            const f32 gs_dn = L.experts.scale_of(L.experts.down_scale, e);
            const f32 gs_gt = L.experts.scale_of(L.experts.gate_scale, e);

            const i64 m = use_plan_off ? m_pre : static_cast<i64>(group_rows_.size());
            const i32 *plan_i = nullptr;
            const f32 *plan_f = nullptr;
            if (use_plan_off) {
                // This expert's segment of the layer plan, already on the
                // device: no copy is issued here at all.
                plan_i = static_cast<const i32 *>(ws_plan_) + plan_off_[e];
                plan_f = static_cast<const f32 *>(ws_alpha_) + plan_off_[e];
            } else {
                plan_dev_.assign(group_rows_.begin(), group_rows_.end());
                alpha_dev_.assign(group_wt_.begin(), group_wt_.end());
                be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(), m);
                be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                            static_cast<size_t>(m) * sizeof(f32));
                plan_i = static_cast<const i32 *>(ws_plan_);
                plan_f = static_cast<const f32 *>(ws_alpha_);
            }

            // x̃ = xn[group_rows] — the permuted activation block. On a latent
            // block xn is the block's projection, not the hidden state.
            be_->gather_rows(ws_xg_, xin, plan_i, m, ew);
            // The global scale is folded into the INPUT rather than applied to
            // the result. Both are the same matrix product -- gs * (W x) is
            // (gs W) x -- but only this one keeps the magnitudes right inside
            // the kernel: scaling the output leaves the accumulator to sum
            // 2688 terms of a 10,000x-too-large product first, and llama.cpp
            // reaches for a bf16 accumulator on NVFP4 for exactly that reason
            // (build_lora_mm: ggml_prec_set_acc(res, GGML_PREC_BF16)). This
            // engine's activation buffer is f16, so the scale has to be in
            // before the first multiply, not after the last add.
            // Guarded on != 1, not unconditional: every format that is not
            // NVFP4 carries no sidecar, gs is 1.0, and a pass over the gathered
            // rows per expert group is a real cost on the models that have no
            // scale at all.
            if (gs_up != 1.0f)
                be_->scale_act(ws_xg_, gs_up, m, ew, ew);
            // One launch for the pair where the shapes allow it. Up and gate
            // share one activation, and the activation that follows reads the
            // SAME output row of both, so one warp computes both dots and
            // applies silu in the epilogue -- fused_gate_silu_kernel, the fold
            // the dense FFN already uses. Per-matrix types travel with it
            // (`gwts`): a checkpoint picks each expert tensor's format on its
            // own, and the kernel has taken one type per matrix since the GDN
            // group landed.
            //
            // Two conditions, both correctness rather than taste:
            //  - gs_gt == gs_up. With differing sidecar scales the old path
            //    rescaled ws_xg_ BETWEEN the two gemms, which a one-input group
            //    cannot express.
            //  - not KRK_DUMP_MOE. The probe's whole point is to read the up and
            //    gate results back as separate facts, and the fused kernel
            //    writes only silu(gate)*up, so a probe run takes the old path.
            // Everything the kernel cannot express (rows != 1, row bytes that
            // differ between the two, split-K) is the backend's call:
            // `pair_fused` comes back false and the silu_mul below still runs,
            // which is why the return value alone is not the answer.
            bool pair_fused = false;
            // Set only when the silu produced a row scale. The fused kernel and
            // the other activation shapes have no row maximum to take, so their
            // rows reach the scatter unscaled (those paths keep the saturating
            // f16 store as their bound).
            const f32 *scatter_scale = nullptr;
            // A layer carrying per-expert FFN biases (gpt-oss) cannot take the
            // fused pair: that kernel's epilogue has already applied silu by the
            // time it returns, and the reference adds the bias to BOTH
            // projections *before* the activation. Going through the two-GEMM
            // path is the only order that can express it.
            if (re->gate.present() && moe_group() && !moe_probe() &&
                gs_gt == gs_up && !L.exp_gate_bias && !L.exp_up_bias &&
                !mc.ffn_swiglu_oai) {
                void *gout[2] = {ws_gateg_, ws_upg_};
                const void *gw[2] = {re->gate.data, re->up.data};
                const i64 gno[2] = {ff_exp, ff_exp};
                const DType gwts[2] = {re->gate.type, re->up.type};
                be_->gemm_group(gout, ws_xg_, gw, re->gate.type, gno, ew, 2,
                                m, moe_silu_fused() && !mc.gelu_ffn, &pair_fused,
                                false, gwts);
            } else {
                be_->gemm(ws_upg_, ws_xg_, re->up.data, re->up.type, ff_exp,
                          ew, m);
                // Per-expert bias: this group is ONE expert's rows, so the
                // table's row `e` is the vector every row of the block gets.
                if (L.exp_up_bias)
                    be_->add_bias_cols(ws_upg_, L.exp_up_bias + static_cast<i64>(e) * ff_exp,
                                       m, ff_exp);
                if (re->gate.present()) {
                    // ws_xg_ is carrying gs_up; leave it carrying gs_gt for the
                    // gate. It is read by nothing else.
                    if (gs_gt != gs_up)
                        be_->scale_act(ws_xg_, gs_gt / gs_up, m, ew, ew);
                    be_->gemm(ws_gateg_, ws_xg_, re->gate.data, re->gate.type,
                              ff_exp, ew, m);
                    if (L.exp_gate_bias)
                        be_->add_bias_cols(
                            ws_gateg_,
                            L.exp_gate_bias + static_cast<i64>(e) * ff_exp, m,
                            ff_exp);
                }
            }
            if (moe_probe()) {
                // After the two projections, before anything is multiplied:
                // the input and the up result are separate facts and the whole
                // question is which one is not finite.
                // Same predicate the KRK_DUMP_FULL summary uses, so a count
                // here and an nf= there are the same measurement.
                auto scan = [&](const void *p, i64 w, i64 &nf, f32 &peak) {
                    std::vector<f32> t(static_cast<size_t>(w));
                    be_->download_f32(t.data(), p, w);
                    nf = 0;
                    peak = 0.0f;
                    for (f32 v : t) {
                        if (!(v == v) || v > 3.0e38f || v < -3.0e38f) nf++;
                        else if (std::fabs(v) > peak) peak = std::fabs(v);
                    }
                };
                i64 in_nf = 0, up_nf = 0, gt_nf = 0;
                f32 in_pk = 0, up_pk = 0, gt_pk = 0;
                scan(ws_xg_, m * ew, in_nf, in_pk);
                scan(ws_upg_, m * ff_exp, up_nf, up_pk);
                scan(ws_gateg_, m * ff_exp, gt_nf, gt_pk);
                // The count says which leg is not finite; the peak says which
                // leg is AT the ceiling, and the leg that overflows the silu
                // product is the one whose peak is at 65504 rather than the
                // one the product was blamed on.
                if (in_nf || up_nf || gt_nf || up_pk > 60000.0f || gt_pk > 60000.0f)
                    std::fprintf(stderr,
                                 "[moe] L%02d e=%3d m=%3lld in_nf=%lld/%g "
                                 "up_nf=%lld/%g gate_nf=%lld/%g\n",
                                 layer, e, static_cast<long long>(m),
                                 static_cast<long long>(in_nf), static_cast<f64>(in_pk),
                                 static_cast<long long>(up_nf), static_cast<f64>(up_pk),
                                 static_cast<long long>(gt_nf), static_cast<f64>(gt_pk));
            }
            if (re->gate.present()) {
                // The fused kernel already left silu(gate)*up in ws_gateg_;
                // everything else reaches here with both operands written.
                // A swiglu-oai file never takes that kernel: this activation is
                // a different function of the same two operands.
                if (!pair_fused) {
                    if (mc.gelu_ffn) {
                        // gemma4's experts are GeGLU: gelu(gate) * up, the same
                        // function its shared MLP applies. The silu pair below
                        // is the wrong nonlinearity on the right operands, which
                        // is why this cannot be left to the default.
                        be_->gelu_act(ws_gateg_, m * ff_exp);
                        be_->mul_act(ws_gateg_, ws_upg_, m * ff_exp);
                        scatter_scale = nullptr;
                    } else if (mc.ffn_swiglu_oai) {
                        be_->swiglu_oai(ws_gateg_, ws_gateg_, ws_upg_, m * ff_exp);
                        // Bounded by construction (|g| <= 7, |y + 1| <= 8), so
                        // there is no row maximum to store and the scatter sums
                        // the values as they are.
                        scatter_scale = nullptr;
                    } else {
                        // Row-scaled: a row here is one token ff_exp-wide
                        // expert activation, which is the dimension the down
                        // projection reduces over, so the scatter undoes the
                        // scale with the per-row weight it already applies.
                        be_->silu_mul(ws_gateg_, ws_gateg_, ws_upg_, m * ff_exp,
                                      ff_exp, static_cast<f32 *>(ws_scale_));
                        scatter_scale = static_cast<const f32 *>(ws_scale_);
                    }
                }
            } else if (mc.ffn_relu_sqr) {
                // Gate-less with squared ReLU (nemotron_h_moe): the activation
                // is relu^2 of the up projection, which is what the reference
                // builds (LLM_FFN_RELU_SQR on ffn_up_exps, gate == null). The
                // silu branch below is the same shape reached the other way
                // round and is right for a family that actually trains silu
                // there -- it is the wrong *function* here, not a mis-scaling:
                // relu^2 zeroes negative pre-activations and squares the rest.
                be_->relu_sqr_act(ws_upg_, m * ff_exp);
                be_->copy_act(ws_gateg_, ws_upg_, m * ff_exp);
            } else {
                // No ffn_gate_exps: the expert is one projection wide, so its
                // activation applies to that projection directly instead of to
                // a gate. silu_mul(out, g, u) is silu(g) * u, and the shape
                // here is silu(up), which is the same function reached the
                // other way round (silu(x) = x * sigmoid(x)).
                //
                // This is the path the model actually needed: with the gate
                // absent the old code still issued the gate GEMM, with a null
                // weight pointer and DType::Unknown. row_bytes() for an unknown
                // dtype is 0, so every row read the same address, the result
                // was 0, and silu(0) * up collapsed the whole routed bank to an
                // exact zero -- 23 layers whose experts contributed nothing at
                // all, which is the zero `moe.sum` that hid everything else.
                be_->copy_act(ws_gateg_, ws_upg_, m * ff_exp);
                be_->sigmoid_act(ws_gateg_, m * ff_exp);
                be_->mul_act(ws_gateg_, ws_upg_, m * ff_exp);
            }
            // The down leg's scale goes into its own input for the same reason
            // as the gate/up pair above.
            if (gs_dn != 1.0f)
                be_->scale_act(ws_gateg_, gs_dn, m, ff_exp, ff_exp);
            if (moe_probe()) {
                // The probe above counts the GATE projection, which is what the
                // silu reads; the down GEMM's input is what the silu WROTE, and
                // a fused silu that overflowed f16 would only show up here.
                std::vector<f32> t2(static_cast<size_t>(m) * static_cast<size_t>(ff_exp));
                be_->download_f32(t2.data(), ws_gateg_, static_cast<i64>(t2.size()));
                i64 act_nf = 0;
                f32 act_peak = 0.0f;
                for (f32 v : t2) {
                    if (!(v == v) || v > 3.0e38f || v < -3.0e38f) act_nf++;
                    else if (std::fabs(v) > act_peak) act_peak = std::fabs(v);
                }
                std::fprintf(stderr,
                             "[moeact] L%02d e=%3d m=%3lld act_nf=%lld act_peak=%.6g\n",
                             layer, e, static_cast<long long>(m),
                             static_cast<long long>(act_nf), static_cast<f64>(act_peak));
            }
            be_->gemm(ws_x2_, ws_gateg_, re->down.data, re->down.type, ew,
                      ff_exp, m);
            // The down leg's bias is over the OUTPUT width (the expert's own),
            // not ff.
            if (L.exp_down_bias)
                be_->add_bias_cols(ws_x2_,
                                   L.exp_down_bias + static_cast<i64>(e) * ew,
                                   m, ew);
            if (moe_probe()) {
                std::vector<f32> t(static_cast<size_t>(m) * static_cast<size_t>(n_embd_));
                be_->download_f32(t.data(), ws_x2_, static_cast<i64>(t.size()));
                i64 out_nf = 0, big = 0;
                for (f32 v : t) {
                    if (!(v == v) || v > 3.0e38f || v < -3.0e38f) out_nf++;
                    else if (v > 1.0e6f || v < -1.0e6f) big++;
                }
                if (out_nf || big)
                    std::fprintf(stderr,
                                 "[moeout] L%02d e=%3d m=%3lld out_nf=%lld big=%lld\n",
                                 layer, e, static_cast<long long>(m),
                                 static_cast<long long>(out_nf),
                                 static_cast<long long>(big));
            }
            // Scatter back to the original rows, weighted by the gate.
            // The routed sum is the one row in this chain whose every
            // producer can be in range and still not fit: ten experts' worth
            // of contributions add up past f16 where none of them alone does
            // (measured on Laguna-XS-2.1: 1020 clipped elements in the prefill
            // chunk, none in decode). The row carries a storage exponent like
            // the residual's, and the folds below un-scale it.
            // The storage exponent is the residual stream's contract and a gemm
            // has no parameter for it: `ffn_latent_up` reads the latent sum as
            // plain f16, so a row stored divided by 2^e would be projected at
            // the wrong scale. The latent is f16 by construction and its sum is
            // narrow, so this arm stores straight -- and an out-of-range store
            // still lands in the stats line's experts-sum saturation count
            // rather than being silently absorbed.
            be_->scatter_axpy_rows(acc, ws_x2_, plan_i, plan_f, m, ew,
                                   scatter_scale,
                                   latent ? nullptr
                                          : static_cast<i32 *>(ws_fexp_));
            // Every kernel that reads this slot's weights is queued by now, and
            // that point is what a later reuse of its cells has to wait for --
            // not the eviction, which comes a layer later and whose mark covers
            // everything queued in between. One record per (layer, expert) use.
            model_.experts().note_read(layer, e);
            // Exact next expert in the existing ascending execution order.
            // Submit its copy only after this expert's weight readers are queued.
            if (n == 1 && be_->kind() == BackendKind::HIP && moe_active_stage()) {
                i32 next = ne;
                for (i32 id : prefetch_ids_) if (id > e && id < next) next = id;
                if (next < ne) model_.experts().stage_next(L.experts, layer, next, e);
            }
        }
        }

        // ---- LatentMoE: lift the routed sum back to the block's width -------
        // The experts summed in the latent; `ffn_latent_up` is the only path
        // back. It is added into `ws_ffn_`, which the latent arm left at zero,
        // so the shared expert below and the residual add at the end see
        // exactly the shape they see on every other family.
        if (latent) {
            gemm_w(ws_xlat_out_, acc, L.latent_up, n_embd_, ew, n);
            be_->add_inplace(ws_ffn_, ws_xlat_out_,
                             static_cast<i64>(n) * n_embd_);
        }

        // The layer's last expert uses are still pending a reader-done mark:
        // emit them here, just past this layer's readers and before the prefetch
        // below can trim or free any of those cells.
        model_.experts().flush_read_notes();

        // ---- prefetch the next MoE layer's predicted set ---------------------
        // Here, after this layer's expert loop, is the only place the eviction is
        // free: every slot this layer used is done, so making room for the next
        // layer costs a copy the WARM tier already holds and nothing else. The
        // DMA runs under the next layer's norm/attention/hyper-connection work,
        // which is the window this feature exists to use.
        if (moe_predict_enabled_ && n == 1 && !expert_stub_ &&
            layer + 1 < mc.n_layer) {
            const i32 j = next_moe_layer(model_.layers(), layer + 1, mc.n_layer);
            if (j >= 0 && model_.layers()[static_cast<size_t>(j)].experts.present()) {
                moe_predictor_.predict(j, layer, moe_predict_budget(),
                                       moe_predict_min_overlap(),
                                       moe_predict_policy(), moe_predict_ids_);
                if (!moe_predict_ids_.empty()) {
                    const size_t staged = model_.experts().stage_ahead(
                        model_.layers()[static_cast<size_t>(j)].experts, j,
                        moe_predict_ids_.data(),
                        static_cast<int>(moe_predict_ids_.size()));
                    moe_predictor_.note_staged(staged);
                }
            }
        }
        if (next_prefetch && model_.experts().host_pin_budget() > 0) {
            const i32 j = next_moe_layer(model_.layers(), layer + 1, mc.n_layer);
            if (j >= 0 && static_cast<size_t>(j) < moe_hint_.size() &&
                model_.layers()[static_cast<size_t>(j)].experts.present() &&
                !moe_hint_[static_cast<size_t>(j)].empty())
                (void)model_.experts().stage_ahead(
                    model_.layers()[static_cast<size_t>(j)].experts, j,
                    moe_hint_[static_cast<size_t>(j)].data(),
                    static_cast<int>(moe_hint_[static_cast<size_t>(j)].size()));
        }

        // ---- the host arm --------------------------------------------------
        //
        // Issued AFTER the device loop, and that is the overlap, not a miss:
        // every gemm and scatter above only *queued* work, so the device is
        // already running when this block starts. The host pool then works while
        // the device works, and the sync at the merge is nearly free precisely
        // because the host cost is what the device would otherwise have spent
        // its time waiting out. Nothing on the host arm is issued with an
        // acquire(), so no expert here is promoted and no WARM copy is made.
        if (hybrid && n_cpu_experts > 0) {
            const size_t cells =
                static_cast<size_t>(n) * static_cast<size_t>(n_embd_);
            hy_xn_.resize(cells);
            hy_out_.assign(cells, 0.0f);
            // The activation the host arm reads, widened from the f16 the device
            // holds: download_f32 converts without changing the values, so the
            // host computes against exactly the numbers the device would have.
            be_->download_f32(hy_xn_.data(), xin, static_cast<i64>(cells));

            hy_jobs_.clear();
            hy_job_rows_.clear();
            hy_job_wt_.clear();
            // Reserve the upper bound before taking any pointer into these: each
            // (token, slot) pair yields at most one row for one expert, so n * k
            // covers the whole routed set and no push_back below can reallocate
            // a buffer a Job already points into.
            const size_t pairs = static_cast<size_t>(n) * static_cast<size_t>(k);
            hy_job_rows_.reserve(pairs);
            hy_job_wt_.reserve(pairs);
            hy_jobs_.reserve(static_cast<size_t>(n_cpu_experts));
            for (i32 e = 0; e < ne; e++) {
                if (!hy_take_[static_cast<size_t>(e)]) continue;
                const size_t first = hy_job_rows_.size();
                for (i32 t = 0; t < n; t++) {
                    const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                    const f32 *wt = moe_wt_.data() + static_cast<size_t>(t) * k;
                    for (i32 s = 0; s < k; s++) {
                        if (sel[s] != e) continue;
                        hy_job_rows_.push_back(t);
                        hy_job_wt_.push_back(wt[s]);
                    }
                }
                if (hy_job_rows_.size() == first) continue;
                CpuExpertPool::Job job;
                job.expert = e;
                job.rows = hy_job_rows_.data() + first;
                job.alpha = hy_job_wt_.data() + first;
                job.m = static_cast<i32>(hy_job_rows_.size() - first);
                hy_jobs_.push_back(job);
            }

            if (!hy_jobs_.empty()) {
                CpuExpertPool &pool = CpuExpertPool::get();
                i32 done = 0;
                {
                    KRK_TIME_HOST("moe.cpu_experts");
                    done = pool.run(L.experts, layer, hy_jobs_.data(),
                                    static_cast<i32>(hy_jobs_.size()),
                                    hy_xn_.data(), n, n_embd_, hy_out_.data());
                }
                if (done != static_cast<i32>(hy_jobs_.size())) {
                    // A dropped expert is a wrong answer, not a slow one. Say so
                    // rather than merging a partial sum as if it were complete.
                    KRK_WARN("hybrid experts: host arm computed %d of %d expert(s) "
                             "on layer %d",
                             done, static_cast<i32>(hy_jobs_.size()), layer);
                }
                hy_cpu_experts_ += static_cast<u64>(done > 0 ? done : 0);
                hy_cpu_rows_ += static_cast<u64>(hy_job_rows_.size());
                hy_cpu_ms_ += pool.last_ms();

                // Merge. ws_ffn_ already holds the device experts' contribution;
                // the host contribution is added to it and the sum goes back as
                // f16, which is the accumulator's own type. No device buffer is
                // added, so this costs no VRAM -- which matters, because the
                // VRAM this feature gives back is the point.
                be_->sync();
                hy_ffn_.resize(cells);
                hy_f16_.resize(cells);
                std::vector<i32> hy_exp(static_cast<size_t>(n));
                be_->download_f32(hy_ffn_.data(), ws_ffn_, static_cast<i64>(cells));
                be_->download(hy_exp.data(), ws_fexp_,
                              static_cast<size_t>(n) * sizeof(i32), 0);
                // ws_ffn_ is stored under a per-token storage exponent whenever
                // the routed sum left f16 (the contract scatter_axpy_rows
                // writes), so the merge un-scales the device part, adds the host
                // contribution at true scale, and picks the row's exponent again.
                // At exponent 0 this is the flat `f16(f32(dev) + host)` it was.
                for (i32 t = 0; t < n; t++) {
                    const size_t base = static_cast<size_t>(t) *
                                        static_cast<size_t>(n_embd_);
                    const f64 sc =
                        std::exp2(static_cast<f64>(hy_exp[static_cast<size_t>(t)]));
                    f64 mx = 0.0;
                    for (i64 j = 0; j < n_embd_; j++) {
                        const size_t at = base + static_cast<size_t>(j);
                        const f64 v = static_cast<f64>(hy_ffn_[at]) * sc +
                                      static_cast<f64>(hy_out_[at]);
                        const f64 av = std::fabs(v);
                        if (av > mx) mx = av;
                    }
                    i32 e2 = hy_exp[static_cast<size_t>(t)];
                    while (mx > 65504.0 && e2 < 60) {
                        mx *= 0.5;
                        e2++;
                    }
                    const f64 inv = std::exp2(-static_cast<f64>(e2));
                    for (i64 j = 0; j < n_embd_; j++) {
                        const size_t at = base + static_cast<size_t>(j);
                        const f64 v = static_cast<f64>(hy_ffn_[at]) * sc +
                                      static_cast<f64>(hy_out_[at]);
                        hy_f16_[at] = fp32_to_fp16(static_cast<f32>(v * inv));
                    }
                    hy_exp[static_cast<size_t>(t)] = e2;
                }
                be_->upload(ws_ffn_, hy_f16_.data(), cells * sizeof(u16));
                be_->upload(ws_fexp_, hy_exp.data(),
                            static_cast<size_t>(n) * sizeof(i32), 0);
            }
        }
    }

    // The shared expert runs on every token (Qwen2-MoE); it is a single expert,
    // so it stays resident.
    // Keyed on the up/down pair, not on the gate: a shared expert without
    // ffn_gate_shexp (nemotron_h_moe) is still a shared expert, and gating this
    // block on the gate skipped it.
    if (!expert_stub_ && ff_sh > 0 && L.shexp_up.present() &&
        L.shexp_down.present()) {
        // NVFP4's global scale rides each projection's own RESULT (gemm_w), so
        // the gate and the up leg may carry different scales without a staged
        // copy of the input: gs*(W x) is (gs W) x for each of them separately.
        // That is what this block used to work around by scaling a copy of xn.
        const void *sh_in = ws_xn_;
        if (L.shexp_gate.present())
            gemm_w(ws_gate_, sh_in, L.shexp_gate, ff_sh, n_embd_, n);
        gemm_w(ws_up_, sh_in, L.shexp_up, ff_sh, n_embd_, n);
        if (L.shexp_gate.present()) {
            be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * ff_sh);
        } else if (mc.ffn_relu_sqr) {
            // The shared expert is the same MLP shape as a routed one on this
            // family (ffn_up_shexp / ffn_down_shexp, no gate) and the reference
            // activates it the same way.
            be_->relu_sqr_act(ws_up_, n * ff_sh);
            be_->copy_act(ws_gate_, ws_up_, n * ff_sh);
        } else {
            be_->copy_act(ws_gate_, ws_up_, n * ff_sh);
            be_->sigmoid_act(ws_gate_, n * ff_sh);
            be_->mul_act(ws_gate_, ws_up_, n * ff_sh);
        }
        gemm_w(ws_x2_, ws_gate_, L.shexp_down, n_embd_, ff_sh, n);
        if (moe_probe()) {
            // The shared expert is the third writer of moe.sum and the one the
            // other probes do not cover: its activation is a silu of an f16
            // pair (so the saturating store can hide an overflow) and its down
            // projection is a full n_embd-wide gemm whose f16 output can exceed
            // the ceiling on its own. `routedsum` is ws_ffn_ BEFORE this leg is
            // folded in, which is what separates "the routed experts did it"
            // from "the shared expert did it".
            auto scan_sh = [&](const void *p, i64 w, const char *tag) {
                std::vector<f32> t(static_cast<size_t>(w));
                be_->download_f32(t.data(), p, w);
                i64 nf = 0;
                f32 pk = 0.0f;
                for (f32 v : t) {
                    if (!(v == v) || v > 3.0e38f || v < -3.0e38f) nf++;
                    else if (std::fabs(v) > pk) pk = std::fabs(v);
                }
                if (nf || pk > 60000.0f)
                    std::fprintf(stderr, "[shexp] L%02d %s nf=%lld peak=%g\n",
                                 layer, tag, static_cast<long long>(nf),
                                 static_cast<f64>(pk));
            };
            scan_sh(ws_ffn_, n * n_embd_, "routedsum");
            scan_sh(ws_gate_, n * ff_sh, "act");
            scan_sh(ws_x2_, n * n_embd_, "out");
        }
        // Qwen3.5 scales the shared expert by a per-token sigmoid of a small
        // vector before folding it in; the Qwen2/Qwen3 schema has no such gate.
        if (L.shexp_inp_gate.present()) {
            gemm_w(ws_shexp_g_, ws_xn_, L.shexp_inp_gate, 1, n_embd_, n);
            be_->sigmoid_act(ws_shexp_g_, n);
            be_->scale_rows(ws_x2_, ws_shexp_g_, n, n_embd_);
        }
        // The fold is the one place a sum of two in-range rows leaves f16 on
        // its own -- measured on Laguna-S-2.1: the routed sum saturated at 65504
        // with the shared expert's own output finite, and their f16 sum went to
        // `inf`, which then drove the residual's exponent to its clamp (15 rows,
        // max exponent 60). Summed in f32 through the same storage contract as
        // the residual: the staging row carries its own exponent from here on and
        // the residual add below reads it. At exponent 0 this is
        // `f16(f32(a) + f32(b))`, which is `add_inplace_kernel`'s arithmetic.
        // The exponents are already in place: the routed scatter wrote each
        // row's own as it summed the experts, and this fold only ever raises
        // one (add_residual reads a's exponent and writes back the row's).
        be_->add_residual(ws_ffn_, ws_x2_, n, n_embd_,
                          static_cast<i32 *>(ws_fexp_));
    }

    dump_row("moe.sum", layer, ws_ffn_, n_embd_, n);
    // The residual add that ends the MoE/FFN block, and the one store on this
    // path with no bound at all: its row is the hidden state, which no row-local
    // scale can bring back into f16 (a saturated hidden state is wrong by a
    // factor and poisons every later layer). It goes through the storage
    // contract instead -- summed in f32, stored at a per-token power-of-two
    // exponent, un-scaled by the next norm.
    // A two-branch block (gemma4) has one more branch to compute and a norm
    // between them: the routed sum stays in `ws_ffn_`, with its storage
    // exponents, and `gemma_moe_ffn` assembles the block from it.
    if (moe_no_residual_) return true;
    be_->add_residual(ws_x_, ws_ffn_, n, n_embd_,
                      static_cast<i32 *>(ws_rexp_),
                      static_cast<const i32 *>(ws_fexp_));
    dump_row("moe.post", layer, ws_x_, n_embd_, n);
    return true;
}

// gemma4's routed block, assembled from the two halves the engine already has.
//
// The reference graph is:
//
//   attn_out = x + attn(attn_norm(x))
//   mlp      = post_ffw_norm_1( gelu(gate * n1) * (up * n1) )      n1 = ffn_norm(attn_out)
//   moe      = post_ffw_norm_2( experts(pre_ffw_norm_2(attn_out)) )
//   x       += post_ffw_norm(mlp + moe)
//
// with the router reading `rms_norm(attn_out) * (1/sqrt(n_embd)) * ffn_gate_inp.scale`
// -- a norm with NO weight, over the post-attention row, which is neither
// branch's input.
//
// Four places differ from the generic block and each of them is a wrong answer
// rather than a failure if it is dropped: the second norm on the same row, the
// two branch norms, the detached router input, and the block norm applied to
// the SUM instead of to each branch.
bool Engine::gemma_moe_ffn(const LayerWeights &L, i32 layer, i64 n) {
    const ModelConfig &mc = model_.cfg();
    const i64 E = n_embd_;
    const f32 peps = mc.post_norm_eps > 0.0f ? mc.post_norm_eps : mc.rms_eps;
    if (!ws_g4_post_ || !ws_g4_mlp_ || !ws_router_in_) {
        forward_error_ = "gemma4 routed layer reached without its branch buffers";
        return false;
    }
    // The post-attention row is what both branches read and what the block's
    // own residual add needs at the end, and the shared MLP consumes `ws_x_`
    // through `dense_ffn`'s residual store -- so it is set aside here rather
    // than re-derived from the projection buffers.
    be_->copy_act(ws_g4_post_, ws_x_, static_cast<i64>(n) * E);

    // ---- the shared MLP branch ------------------------------------------
    // Its input is already in `ws_xn_` (the caller's fused add_rmsnorm computed
    // it), its output norm is post_ffw_norm_1, and its result is a BRANCH: it
    // goes into `ws_g4_mlp_` instead of into the residual.
    { KRK_TRACE_COMP("gemma-moe.mlp");
      dense_ffn(L, layer, n, ws_g4_mlp_, L.ffn_post_norm1); }

    // ---- the expert branch's input and the router's own -----------------
    { KRK_TRACE_COMP("gemma-moe.norm");
    be_->rmsnorm(ws_xn_, ws_g4_post_, L.ffn_pre_norm2, n, E, peps, nullptr);
    dump_row("moe.pre2", layer, ws_xn_, E, n);
    be_->rmsnorm(ws_router_in_, ws_g4_post_, nullptr, n, E, peps, nullptr);
    be_->scale_act(ws_router_in_, 1.0f / std::sqrt(static_cast<f32>(E)), n, E, E);
    // Per-ELEMENT: `ggml_mul` with a [n_embd] tensor broadcasts down the rows,
    // so the scale is a column vector and not the per-token scalar
    // `scale_rows` would apply.
    be_->scale_cols(ws_router_in_, L.router_scale, n, E);
    dump_row("moe.routerin", layer, ws_router_in_, E, n); }

    // ---- the expert branch ----------------------------------------------
    // `moe_ffn` reads `moe_in_` instead of `ws_xn_` for the experts, takes the
    // router's input from `ws_router_in_`, and leaves its routed sum in
    // `ws_ffn_` (with per-row storage exponents) instead of adding it to the
    // residual.
    bool ok = false;
    {
        KRK_TRACE_COMP("gemma-moe.experts");
        moe_in_ = ws_xn_;
        moe_no_residual_ = true;
        ok = moe_ffn(L, layer, static_cast<i32>(n));
        moe_in_ = nullptr;
        moe_no_residual_ = false;
    }
    if (!ok) return false;

    // post_ffw_norm_2 on the routed sum. The sum is stored under a per-row
    // power-of-two exponent, so the norm reads through it and then the rows are
    // back at true scale: leaving the exponents behind would divide the
    // normalized row by 2^e on the add below.
    be_->rmsnorm(ws_ffn_, ws_ffn_, L.ffn_post_norm2, n, E, peps,
                 static_cast<const i32 *>(ws_fexp_));
    be_->fill0(ws_fexp_, static_cast<size_t>(n) * sizeof(i32));
    dump_row("moe.branch2", layer, ws_ffn_, E, n);
    be_->add_inplace(ws_ffn_, ws_g4_mlp_, static_cast<i64>(n) * E);

    // ---- the block's own post norm, then the residual -------------------
    be_->rmsnorm(ws_x2_, ws_ffn_, L.ffn_post_norm, n, E, peps, nullptr);
    dump_row("moe.postnorm", layer, ws_x2_, E, n);
    be_->add_residual(ws_x_, ws_x2_, n, E, static_cast<i32 *>(ws_rexp_));
    dump_row("moe.post", layer, ws_x_, E, n);
    return true;
}

void Engine::fetch_logits(bool device_only) {
    KRK_TRACE_COMP("fetch-logits");
    // The head's gemm is a device op, but this call also covers the download
    // that follows it, and on a small model that blocking readback is most of
    // the token (measured: 60.4 of 92.1 ms on Qwen3-4B-Q8_0). Its own row
    // separates "the head is slow" from "waiting for the head is slow".
    KRK_TIME_HOST(device_only ? "decode.logits.dev" : "decode.logits.host");
    // Greedy decode reads exactly one number out of this row. Downloading it
    // first cost 993 kB of PCIe and, on the 8B, 2.7 ms of *device* idle per
    // token: the GPU had finished and sat waiting for the host. The device
    // top-k returns the same integer with the row never leaving VRAM. Every
    // other caller (sampling, --debug-topk, the speculative and draft paths)
    // passes false and gets the whole row exactly as before.
    //
    // The parameter says "leave it on the device", matching the flag the decode
    // loop computes. It used to be the other way round -- `want_host` -- and a
    // call site passed the device-side flag straight into it, so the row was
    // downloaded on every token and the topk path never ran at all. Nothing in
    // the output differed, which is exactly why it survived: a polarity bug in
    // a boolean is invisible until you measure the thing it controls.
    if (device_only && topk_argmax(&topk_id_)) return;
    topk_valid_ = false;
    if (logits_sync()) {
        // Name the drain, because in whole-graph mode this call is not the
        // head's cost -- it is the forward pass's unpaid bill.
        //
        // The offload batches a run of device ops and submits them at the
        // first host op. `decode.forward` therefore returns once the layer's
        // work has been *issued*, and the wait for it lands on whichever host
        // call flushes next. That used to be this one, so "decode.logits.dev"
        // absorbed ~0.7 ms/token of layer work and the output projection read
        // as 13-16% of a decode run. The same head GEMM measures 0.001 ms/call
        // on the arm that flushes every call, and the device executes it in
        // 127 us, so the row was reporting someone else's time.
        //
        // Splitting the wait into its own row makes the arithmetic add up:
        // head + drain is the span, and drain is the part that scales with how
        // much of the layer stack is still in flight. Nothing is made faster by
        // this; a profile that misattributes a cost is how a search for a
        // 13% win starts in the wrong place, which is exactly what happened.
        KRK_TIME_HOST_N("decode.drain", drain_scope);
        be_->sync();
    }
    be_->download_f32(logits_host_, ws_logits_, n_vocab_);
    // Gemma 2/3/4 cap the *final* logits. This is the one place logits become
    // host-visible, so every path (plain, speculative, draft) gets the bound for
    // free, and it is monotone — greedy decoding cannot change because of it.
    const f32 cap = model_.cfg().logit_softcap;
    if (cap > 0.0f) {
        for (i32 v = 0; v < n_vocab_; v++)
            logits_host_[v] = apply_logit_softcap(logits_host_[v], cap);
    }
}

bool Engine::topk_argmax(i32 *out) {
    // The CPU twin of this op is a plain partial_sort over the whole row, so
    // calling it there would be strictly slower than the host argmax loop it is
    // replacing. Only the device path can win, and it wins by not moving data.
    if (!be_ || !be_->has_device_memory() || topk_scratch_ == nullptr)
        return false;
    // No leading sync. This used to drain the whole device before the launch,
    // which is a stall the stream already prevents: ws_logits_ was written by
    // the head's gemm and logits_topk reads it, both on the default stream, so
    // ordering is guaranteed without stopping the queue. It cost a full device
    // idle per token -- 8.5% of an 11.3 ms step on the 8B -- and it is
    // invisible in the output.
    char *zone = static_cast<char *>(topk_out_);
    be_->logits_topk(ws_logits_, n_vocab_, kTopK, nullptr, 1.0f,
                     model_.cfg().logit_softcap, 1.0f,
                     reinterpret_cast<i32 *>(zone), reinterpret_cast<f32 *>(zone + 4),
                     nullptr, topk_scratch_, kTopKCap,
                     reinterpret_cast<i64 *>(zone + 8));
    if (topk_sync()) be_->sync();
    // 16 bytes: the answer and the proof that it was not truncated. The old
    // path moved 993 kB and then compared 248320 of them on the host.
    be_->download(topk_out_h_, topk_out_, sizeof(topk_out_h_), 0);
    const i32 id = *reinterpret_cast<const i32 *>(topk_out_h_);
    topk_n_ = *reinterpret_cast<const i64 *>(topk_out_h_ + 8);
    // An overflow means the candidate buffer was too small for the tie sitting
    // on the cut, so the ranking cannot be trusted. Fall back, never guess.
    if (topk_n_ > kTopKCap || id < 0) return false;
    *out = id;
    topk_valid_ = true;
    return true;
}

// The nextn-block round. Same shape as generate_speculative -- greedy only,
// one batched verification per round, longest matching prefix kept -- with the
// proposal side replaced by the block that ships in the target's own file.
//
// The block is not a model: it has no embedding and no head of its own, and its
// input at position p is the TARGET's post-stack hidden state (hnorm) fused with
// that position's token embedding (enorm). So a round is
//
//   * prime: every position the target has already computed is fed through the
//     block once, from the hidden rows captured during the target's forwards
//   * propose: the block's chain -- position pos-1 uses the captured target row,
//     every later proposal feeds the block's OWN previous output, which is what
//     lets one head propose several tokens from one target forward
//   * verify: one batched forward of the proposals, one argmax per row
//   * realign: roll the block's KV back to the accepted prefix and re-feed the
//     accepted tokens' positions from their captured TARGET rows. The block's
//     speculative rows were built from the block's hidden, not the target's, so
//     keeping them would silently draft from a different distribution -- the
//     failure class this file warns about: no crash, just different text.
bool Engine::generate_speculative_mtp(const GenerateParams &p, GenerateResult *res,
                                      const std::vector<i32> &ids, f64 *prefill_ms,
                                      f64 *decode_ms, i64 *decode_steps) {
    const MtpBlock &m = model_.mtp();
    const i32 k = std::max<i32>(1, draft_tokens_);
    // Replay takes the block out of the run entirely, including its priming:
    // what is left is the target's verify pass, which is what a per-round cost
    // comparison between cache states is supposed to measure. Read here as well
    // as below because the priming call comes first.
    const char *rp_in_early = std::getenv("KRK_MTP_REPLAY_IN");
    const bool replaying = rp_in_early && *rp_in_early;
    // A hybrid target (gated delta net layers) has no KV to rewind: its state
    // is a recurrence. That is why the round snapshots it, and why an accepted
    // prefix is replayed rather than merely re-pointed.
    const bool recurrent = model_.is_recurrent();
    const bool can_snapshot =
        recurrent && mtp_rec_snap_ != nullptr && mtp_conv_snap_ != nullptr;

    // ---- prompt prefill (the capture rides every forward) -----------------
    expert_phase(ExpertCache::PhasePrefill);
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        if (!forward(ids.data() + i, n, static_cast<i32>(i), last)) {
            res->error = forward_failure_error();
            res->finish = FinishError;
            return false;
        }
    }
    fetch_logits(false);
    *prefill_ms = prefill_timer.ms();
    expert_phase(ExpertCache::PhaseDecode);

    i64 pos = static_cast<i64>(ids.size());
    mtp_fed_ = 0;
    if (!replaying && !mtp_prime(pos)) {
        res->error = "the nextn block could not be primed over the prompt" +
                     (forward_error_.empty() ? std::string()
                                             : ": " + forward_error_);
        res->finish = FinishError;
        return false;
    }

    Sampler sampler;
    sampler.reset(p.sampler);
    sampler.accept(ids.data(), static_cast<int>(ids.size()));
    TokenEmitter emit(p, tok_, sampler, res, ids);

    const Timer decode_timer;
    i64 timed_decode_steps = 0;
    auto finish_run = [&](Finish f) {
        res->finish = f;
        *decode_ms = decode_timer.ms();
        *decode_steps = timed_decode_steps;
        emit.finish(f == FinishStop);
        // Reported here, not at the loop's exit: most runs end through one of
        // the early returns above, and a summary that only prints on the
        // normal exit is missing from exactly the runs worth reading.
        if (spec_steps_ > 0) {
            const f64 rate =
                draft_proposed_ > 0
                    ? 100.0 * static_cast<f64>(draft_accepted_) /
                          static_cast<f64>(draft_proposed_)
                    : 0.0;
            KRK_INFO("mtp spec: %llu round(s), %llu of %llu proposals kept "
                     "(%.1f%%), %.2f token(s)/round | propose %.0f ms | verify "
                     "%.0f ms | snapshot %.0f ms | refeed %.0f ms | block "
                     "experts %.0f ms, %llu of %llu promoted | second forward "
                     "%.0f ms over %llu round(s)",
                     static_cast<unsigned long long>(spec_steps_),
                     static_cast<unsigned long long>(draft_accepted_),
                     static_cast<unsigned long long>(draft_proposed_), rate,
                     spec_steps_ > 0
                         ? static_cast<f64>(res->generated) /
                               static_cast<f64>(spec_steps_)
                         : 0.0,
                     mtp_propose_ms_, mtp_verify_ms_, mtp_snap_ms_,
                     mtp_refeed_ms_, mtp_acq_ms_,
                     static_cast<unsigned long long>(mtp_acq_miss_),
                     static_cast<unsigned long long>(mtp_acq_n_),
                     mtp_extra_ms_,
                     static_cast<unsigned long long>(mtp_extra_n_));
            // The phase split, when the propose pass was asked to drain between
            // phases. Per proposal, so it compares directly with the round.
            if (prop_timed_ > 0)
                KRK_INFO("mtp propose split: %llu proposal(s) | embed %.2f ms | "
                         "block body %.2f ms | head %.2f ms | argmax %.2f ms per "
                         "proposal (phases drained between)",
                         static_cast<unsigned long long>(prop_timed_),
                         prop_embed_ms_ / static_cast<f64>(prop_timed_),
                         (prop_block_ms_ - prop_head_ms_) /
                             static_cast<f64>(prop_timed_),
                         prop_head_ms_ / static_cast<f64>(prop_timed_),
                         prop_argmax_ms_ / static_cast<f64>(prop_timed_));
        }
    };

    // Invariant at the top of every round: the target's KV holds [0, pos) and
    // logits_host_ predicts pos; the block's KV holds the same positions, built
    // from those positions' target hidden rows.
    std::vector<i32> prop(static_cast<size_t>(k), -1);
    // The token at position pos-1, which is what proposal 0 must embed. It is
    // NOT res->tokens.back() on the first round: the emitter only appends what
    // it emits, so that vector is empty until the first accepted token, and
    // reading it there is an out-of-range back() (measured: SIGSEGV on Tiel).
    auto last_ctx_token = [&]() -> i32 {
        return res->tokens.empty() ? ids.back() : res->tokens.back();
    };
    // Re-feed the block over the accepted prefix. `from` is the position the
    // round started at: rows [0, from) are known-good target rows.
    auto realign_block = [&](i64 from) {
        const u64 t = now_us();
        mtp_fed_ = from;
        const bool ok = mtp_prime(pos);
        mtp_refeed_ms_ += static_cast<f64>(now_us() - t) / 1000.0;
        return ok;
    };

    // ---- proposal replay (KRK_MTP_REPLAY_IN / _OUT) ------------------------
    // The proposals decide what the VERIFY pass computes, so comparing verify
    // across cache states needs the same rows in both arms. Recording them once
    // and replaying them takes the block out of the round entirely: a replay
    // arm measures the target's verify (and the refeed) and nothing else, and
    // every arm replaying the same file sees byte-identical rows in the same
    // order. Text is still decided by the target, so a replay that accepts
    // differently has diverged and says so.
    struct RpRound {
        i32 pos = 0, d = 0, id[4] = {0, 0, 0, 0};
    };
    std::vector<RpRound> replay;
    size_t rp_at = 0;
    const char *rp_in = rp_in_early;
    if (replaying) {
        if (std::FILE *f = std::fopen(rp_in, "rb")) {
            RpRound e;
            while (std::fscanf(f, "%d %d %d %d %d %d", &e.pos, &e.d, &e.id[0],
                               &e.id[1], &e.id[2], &e.id[3]) == 6)
                replay.push_back(e);
            std::fclose(f);
            KRK_INFO("mtp replay: %zu round(s) from %s -- the block does not "
                     "run; this run measures the target's verify only",
                     replay.size(), rp_in);
        } else {
            res->error = std::string("replay file could not be read: ") + rp_in;
            res->finish = FinishError;
            return false;
        }
    }
    const char *rp_out = std::getenv("KRK_MTP_REPLAY_OUT");
    std::FILE *rp_f = (rp_out && *rp_out) ? std::fopen(rp_out, "wb") : nullptr;

    while (res->generated < p.max_tokens && pos < kv_cap_) {
        KRK_TIME_HOST("spec.round");
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, res->generated, pos, logits_host_, n_vocab_);
        const i32 first = argmax_of(logits_host_, n_vocab_);
        const i64 from = pos;

        // ---- propose: the block's own chain -------------------------------
        // A replay carries the proposals a recorded run made, so the block is
        // not run at all here; a recording arm runs it and writes what it got.
        const u64 t_prop = now_us();
        // Phase split for the propose pass, opt-in because it drains the queue
        // between phases. Without the drains every phase would read as its
        // launch cost: the block's work sits in the queue until the argmax
        // readback waits on it, so all of the device time would be charged to
        // whichever call happens to block.
        static const bool psplit = [] {
            const char *e = std::getenv("KRK_MTP_PROP_SPLIT");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        auto pm = [&]() -> u64 {
            if (psplit) be_->sync();
            return now_us();
        };
        i32 d = 0;
        if (replaying) {
            if (rp_at >= replay.size()) {
                finish_run(FinishLength);
                return true;
            }
            const RpRound &e = replay[rp_at++];
            if (e.pos != static_cast<i32>(pos)) {
                res->error = "replay diverged at position " + std::to_string(pos) +
                             " (the log has " + std::to_string(e.pos) + ")";
                finish_run(FinishError);
                return false;
            }
            d = std::min<i32>(e.d, k);
            for (i32 j = 0; j < d; j++) prop[static_cast<size_t>(j)] = e.id[j];
            draft_proposed_ += static_cast<u64>(d);
        }
        for (; !replaying && d < k; d++) {
            const i64 fpos = pos - 1 + d;
            if (fpos < 0 || fpos >= kv_cap_) break;
            const i32 tok_in =
                d == 0 ? last_ctx_token() : prop[static_cast<size_t>(d - 1)];
            if (tok_in == tok_.eos()) break;
            const u64 t_e0 = pm();
            mtp_embed(tok_in);
            const u64 t_e1 = pm();
            if (d == 0) {
                if (fpos >= mtp_captured_) break;
                be_->copy_act(mtp_h_,
                              act_at(mtp_hidden_, be_->act_size(),
                                     mtp_capture_row(fpos) * n_embd_),
                              n_embd_);
            } else {
                // The block's residual from the previous proposal. Copied
                // before this forward overwrites ws_x_.
                be_->copy_act(mtp_h_, ws_x_, n_embd_);
            }
            if (!mtp_block_forward(m, static_cast<i32>(fpos), true)) break;
            const u64 t_e2 = pm();
            i32 id = -1;
            if (!topk_argmax(&id)) {
                be_->sync();
                be_->download_f32(logits_host_, ws_logits_, n_vocab_);
                id = argmax_of(logits_host_, n_vocab_);
            }
            if (psplit) {
                prop_embed_ms_ += static_cast<f64>(t_e1 - t_e0) / 1000.0;
                prop_block_ms_ += static_cast<f64>(t_e2 - t_e1) / 1000.0;
                prop_argmax_ms_ += static_cast<f64>(now_us() - t_e2) / 1000.0;
                prop_timed_++;
            }
            if (id < 0) break;
            prop[static_cast<size_t>(d)] = id;
            draft_proposed_++;
        }
        mtp_propose_ms_ += static_cast<f64>(now_us() - t_prop) / 1000.0;
        if (rp_f)
            std::fprintf(rp_f, "%d %d %d %d %d %d\n", static_cast<int>(pos), d,
                         prop[0], k > 1 ? prop[1] : 0, k > 2 ? prop[2] : 0,
                         k > 3 ? prop[3] : 0);
        spec_steps_++;
        timed_decode_steps++;

        // ---- pre-check: the target must agree with proposal 0 -------------
        if (d == 0 || first != prop[0]) {
            if (spec_trace())
                std::fprintf(stderr,
                             "[mtp-s] round %llu pos=%lld pre-check FAILED: "
                             "target=%d block=%d (d=%d)\n",
                             static_cast<unsigned long long>(spec_steps_),
                             static_cast<long long>(pos), first,
                             d > 0 ? prop[0] : -1, d);
            if (first == tok_.eos()) {
                finish_run(FinishEos);
                return true;
            }
            emit.accept(first);
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            const u64 t_xfc = now_us();
            if (!forward(&first, 1, static_cast<i32>(pos), true)) {
                res->error = forward_failure_error();
                finish_run(FinishError);
                return false;
            }
            pos++;
            fetch_logits(false);
            mtp_extra_ms_ += static_cast<f64>(now_us() - t_xfc) / 1000.0;
            mtp_extra_n_++;
            if (!replaying && !realign_block(from)) {
                res->error = "the nextn block's KV could not be re-fed after a "
                             "pre-check miss";
                finish_run(FinishError);
                return false;
            }
            continue;
        }

        // ---- verify: one batched forward of the d proposals --------------
        i32 d_eff = d;
        while (d_eff > 0 && pos + d_eff > kv_cap_) d_eff--;
        if (d_eff <= 0) {
            finish_run(FinishContext);
            return true;
        }
        // The verification pass advances the target over rows it may not keep.
        // Nothing in a hybrid's delta-rule state can be undone afterwards, so
        // the state is copied out before the pass and restored on a rejection.
        if (can_snapshot) {
            const u64 t_snap = now_us();
            be_->copy_device(mtp_rec_snap_, rec_state_, mtp_rec_snap_bytes_);
            be_->copy_device(mtp_conv_snap_, conv_state_, mtp_conv_snap_bytes_);
            mtp_snap_ms_ += static_cast<f64>(now_us() - t_snap) / 1000.0;
        }
        const u64 t_ver = now_us();
        if (!forward_core(prop.data(), d_eff, static_cast<i32>(pos), LogitsNone)) {
            res->error = forward_failure_error();
            finish_run(FinishError);
            return false;
        }
        kv_pos_ = pos + d_eff;
        std::vector<i32> row_ids(static_cast<size_t>(d_eff), 0);
        for (i32 j = 0; j < d_eff; j++) {
            head_compute(j);
            i32 id = 0;
            if (!topk_argmax(&id)) {
                be_->sync();
                be_->download_f32(logits_host_, ws_logits_, n_vocab_);
                id = argmax_of(logits_host_, n_vocab_);
            }
            row_ids[static_cast<size_t>(j)] = id;
        }
        mtp_verify_ms_ += static_cast<f64>(now_us() - t_ver) / 1000.0;
        auto row_pred = [&](i32 j) { return row_ids[static_cast<size_t>(j - 1)]; };

        // ---- accept the longest greedy-matching prefix --------------------
        i32 a = 1; // proposal 0 passed the pre-check
        while (a < d_eff && row_pred(a) == prop[static_cast<size_t>(a)]) a++;
        const bool full = (a == d_eff);
        if (spec_trace()) {
            std::fprintf(stderr, "[mtp-s] round %llu pos=%lld prop=[",
                         static_cast<unsigned long long>(spec_steps_),
                         static_cast<long long>(pos));
            for (i32 j = 0; j < d_eff; j++)
                std::fprintf(stderr, "%s%d", j ? "," : "",
                             prop[static_cast<size_t>(j)]);
            std::fprintf(stderr, "] row=[");
            for (i32 j = 0; j < d_eff; j++)
                std::fprintf(stderr, "%s%d", j ? "," : "",
                             row_ids[static_cast<size_t>(j)]);
            std::fprintf(stderr, "] accept=%d full=%d\n", a, full ? 1 : 0);
        }

        bool stop_hit = false;
        i32 emitted = 0;
        for (i32 j = 0; j < a; j++) {
            if (res->generated >= p.max_tokens) break;
            const i32 t = prop[static_cast<size_t>(j)];
            draft_accepted_++;
            stop_hit = emit.accept(t);
            emitted++;
            if (stop_hit) break;
        }
        if (stop_hit) {
            // The run ends here, so a hybrid's state is never read again; a
            // dense target still gives its cache back.
            if (!recurrent) kv_rollback(pos + emitted);
            finish_run(FinishStop);
            return true;
        }
        if (full && res->generated < p.max_tokens) {
            // The row the last proposal wrote predicts position pos+d: the
            // target's own bonus token, which costs no extra forward to learn.
            const i32 bonus = row_pred(d_eff);
            if (bonus == tok_.eos()) {
                kv_rollback(pos + emitted);
                finish_run(FinishEos);
                return true;
            }
            emit.accept(bonus);
            emitted++;
            if (pos + emitted < kv_cap_ && res->generated < p.max_tokens) {
                const i32 b = res->tokens.back();
                if (spec_trace())
                    std::fprintf(stderr, "[mtp-s] BONUS fed %d at %lld\n", b,
                                 static_cast<long long>(pos + emitted - 1));
                const u64 t_xfc = now_us();
                if (!forward(&b, 1, static_cast<i32>(pos + emitted - 1), true)) {
                    res->error = forward_failure_error();
                    finish_run(FinishError);
                    return false;
                }
                fetch_logits(false);
                mtp_extra_ms_ += static_cast<f64>(now_us() - t_xfc) / 1000.0;
                mtp_extra_n_++;
            }
        }
        if (!full) {
            if (can_snapshot) {
                // Undo the rejected suffix by restoring the pre-round state and
                // replaying the accepted prefix in ONE batched forward: the
                // recurrent scan walks the rows in order, so this is the same
                // state and the same logits a sequential loop would have left,
                // for one batched pass instead of `emitted` decode steps.
                be_->copy_device(rec_state_, mtp_rec_snap_, mtp_rec_snap_bytes_);
                be_->copy_device(conv_state_, mtp_conv_snap_, mtp_conv_snap_bytes_);
                if (emitted > 0) {
                    if (!forward_core(prop.data(), emitted, static_cast<i32>(pos),
                                      LogitsLast)) {
                        res->error = forward_failure_error();
                        finish_run(FinishError);
                        return false;
                    }
                    fetch_logits(false);
                }
            } else {
                // A dense target: drop the rejected rows; the next step's
                // logits are verification row `emitted` (1-based), still in
                // ws_x_ because the bonus forward did not run.
                kv_rollback(pos + emitted);
                if (emitted > 0) {
                    head_compute(emitted - 1);
                    if (spec_row_sync()) be_->sync();
                    be_->download_f32(logits_host_, ws_logits_, n_vocab_);
                }
            }
        } else if (res->generated >= p.max_tokens && !recurrent) {
            // A full match whose bonus token was not forwarded (the token
            // limit ended the run): give the cache back.
            kv_rollback(pos + emitted);
        }
        pos += emitted;
        if (recurrent) kv_pos_ = pos;

        if (res->generated >= p.max_tokens) {
            finish_run(FinishLength);
            return true;
        }
        if (pos + 1 >= kv_cap_) {
            finish_run(FinishContext);
            return true;
        }
        if (!replaying && !realign_block(from)) {
            res->error = "the nextn block's KV could not be re-fed after a round";
            finish_run(FinishError);
            return false;
        }
    }

    finish_run(res->generated >= p.max_tokens ? FinishLength : FinishContext);
    return true;
}

bool Engine::generate_speculative(const GenerateParams &p, GenerateResult *res,
                                  const std::vector<i32> &ids, f64 *prefill_ms,
                                  f64 *decode_ms, i64 *decode_steps) {
    Engine &draft = *draft_;
    const i32 k = draft_tokens_;

    // ---- prompt prefill on both models ------------------------------------
    expert_phase(ExpertCache::PhasePrefill);
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        if (!forward(ids.data() + i, n, static_cast<i32>(i), last)) {
            res->error = forward_failure_error();
            res->finish = FinishError;
            return false;
        }
    }
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        if (!draft.forward(ids.data() + i, n, static_cast<i32>(i), last)) {
            res->error = "draft KV promotion refused by the host-memory reserve or "
                         "an unavailable WARM/COLD page";
            res->finish = FinishError;
            return false;
        }
    }
    fetch_logits();
    draft.fetch_logits(true);
    *prefill_ms = prefill_timer.ms();
    expert_phase(ExpertCache::PhaseDecode);

    Sampler sampler;
    sampler.reset(p.sampler);
    sampler.accept(ids.data(), static_cast<int>(ids.size()));
    TokenEmitter emit(p, tok_, sampler, res, ids);

    const Timer decode_timer;
    i64 timed_decode_steps = 0;
    auto finish_run = [&](Finish f) {
        res->finish = f;
        *decode_ms = decode_timer.ms();
        *decode_steps = timed_decode_steps;
        emit.finish(f == FinishStop);
    };

    i64 pos = static_cast<i64>(ids.size());
    std::vector<i32> prop(static_cast<size_t>(k));  // draft proposals T[1..d]
    std::vector<f32> rows(static_cast<size_t>(k) * static_cast<size_t>(n_vocab_));

    // Invariant at the top of every round (identical to plain decode):
    //   target KV holds [0, pos), logits_host_ predicts position pos
    //   draft  KV holds [0, pos), draft logits predict position pos
    while (res->generated < p.max_tokens && pos < kv_cap_) {
        // One speculation round, wall to wall: propose, pre-check, verify,
        // emit. The device op table cannot delimit this because a round is
        // several ops plus host work between them.
        KRK_TIME_HOST("spec.round");
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, res->generated, pos, logits_host_, n_vocab_);
        const i32 first = argmax_of(logits_host_, n_vocab_);

        // ---- draft: propose up to k continuations -------------------------
        // Each proposal must run the draft's HEAD: without its logits the next
        // argmax would read stale values and the chain would repeat one token.
        i32 d = 0;
        while (d < k) {
            // Prefer the draft's own device top-k. The host row is a fallback
            // for the CPU backend and for a truncated cut, and reading it was
            // never free: fetch_logits(false) moves the whole row, which at
            // 151936 vocab is 993 kB per proposal, k times a round.
            const i32 t = draft.topk_valid_ ? draft.topk_id_
                                            : argmax_of(draft.logits_host_, n_vocab_);
            if (t == tok_.eos()) break; // the target confirms EOS itself
            // Room check BEFORE storing: proposal d lives at slot pos+d, so
            // without room the chain stops here. Storing first and breaking
            // later left the last proposal unfed, and the full-accept
            // realign then failed its rollback past that gap (silently) and
            // wrote the bonus beyond it -- a permanent hole of stale KV rows
            // that poisoned every later draft prediction (measured: self-draft
            // at 27.8% instead of ~100%). Every stored proposal is fed, so the
            // draft cache stays contiguous [0, kv_pos_) by construction.
            if (pos + d >= kv_cap_ || pos + d >= draft.kv_capacity()) break;
            prop[static_cast<size_t>(d)] = t;
            d++;
            draft_proposed_++;
            // The d-th proposal belongs at position pos+d-1 (the first extends
            // the prompt, whose rows end at pos-1).
            if (!draft.forward(&t, 1, static_cast<i32>(pos + d - 1), true)) {
                res->error = "draft KV promotion refused by the host-memory reserve or "
                             "an unavailable WARM/COLD page";
                finish_run(FinishError);
                return false;
            }
            draft.fetch_logits(true);
        }
        spec_steps_++;
        // One speculation round is this loop's unit of work, and it is the
        // only thing decode_ms covers here. Without this the timer was
        // reported against zero steps and every speculative run printed
        // "0.0 tok/s" no matter how fast it was.
        timed_decode_steps++;

        // ---- pre-check: does the target agree with the first proposal? ----
        // This is the same test plain decode performs; on failure the round
        // costs exactly one plain decode step (the block is never run).
        //
        // On mismatch the first accepted token is still emitted exactly once:
        // the mismatch branch emits the fallback token here, and if the round
        // later emits anything else it does so in the verify/accept loop.
        if (d == 0 || first != prop[0]) {
            if (spec_trace()) {
                std::fprintf(stderr,
                             "[spec ] round %llu pos=%lld pre-check FAILED: "
                             "target=%d draft[0]=%d (d=%d) -- the block never runs",
                             static_cast<unsigned long long>(spec_steps_),
                             static_cast<long long>(pos), first,
                             d > 0 ? prop[0] : -1, d);
                print_target_top3(stderr, logits_host_, n_vocab_);
                std::fputc(10, stderr);
            }
            if (first == tok_.eos()) {
                finish_run(FinishEos);
                return true;
            }
            emit.accept(first);
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            if (!forward(&first, 1, static_cast<i32>(pos), true)) {
                res->error = forward_failure_error();
                finish_run(FinishError);
                return false;
            }
            pos++;
            fetch_logits();
            // Keep the draft aligned: same position convention as below —
            // feed the accepted token AT pos-1 (its natural KV slot).
            if (!draft.kv_rollback(pos - 1)) {
                res->error = "draft KV rollback refused while realigning the "
                             "accepted prefix";
                finish_run(FinishError);
                return false;
            }
            if (!draft.forward(&first, 1, static_cast<i32>(pos - 1), true)) {
                res->error = "draft KV promotion refused by the host-memory reserve or "
                             "an unavailable WARM/COLD page";
                finish_run(FinishError);
                return false;
            }
            draft.fetch_logits(true);
            continue;
        }

        // ---- verify: one batched forward of the d proposals ---------------
        // Block row j (input T[j] at position pos+j-1) predicts pos+j. Rows
        // 1..d-1 verify T[2..d]; row d carries the bonus for pos+d.
        i32 d_eff = d;
        while (d_eff > 0 && pos + d_eff > kv_cap_) d_eff--;
        if (d_eff <= 0) {
            finish_run(FinishContext);
            return true;
        }
        if (!forward_core(prop.data(), d_eff, static_cast<i32>(pos), LogitsNone)) {
            res->error = forward_failure_error();
            finish_run(FinishError);
            return false;
        }
        kv_pos_ = pos + d_eff;
        // The head runs once per row into the single-row logits buffer, and
        // each row is downloaded before the next overwrites it. (Sizing
        // ws_logits_ for the whole block would also work; per-row keeps the
        // workspace independent of the speculation window.)
        // One argmax per verified row, taken on the device. Downloading each row
        // to argmax it on the host moved d_eff * n_vocab floats per round --
        // 993 kB per proposal at 151936 vocab -- for one integer. The device
        // top-k returns that integer in 16 bytes. row_ids[j-1] is row_pred(j),
        // which is 1-based like every call site below.
        std::vector<i32> row_ids(static_cast<size_t>(d_eff), 0);
        for (i32 j = 0; j < d_eff; j++) {
            head_compute(j);
            i32 id = 0;
            if (!topk_argmax(&id)) {
                // CPU backend, or a cut too wide to trust: fall back to the row.
                be_->sync();
                be_->download_f32(rows.data() + static_cast<size_t>(j) * n_vocab_,
                                  ws_logits_, n_vocab_);
                id = argmax_of(rows.data() + static_cast<size_t>(j) * n_vocab_,
                               n_vocab_);
            }
            row_ids[static_cast<size_t>(j)] = id;
        }
        auto row_pred = [&](i32 j) { // argmax of block row j (1-based)
            return row_ids[static_cast<size_t>(j - 1)];
        };

        // ---- accept the longest greedy-matching prefix ---------------------
        i32 a = 1; // T[1] passed the pre-check
        while (a < d_eff && row_pred(a) == prop[static_cast<size_t>(a)]) a++;
        if (spec_trace()) {
            const bool tr_full = (a == d_eff);
            std::fprintf(stderr, "[spec ] round %llu pos=%lld d=%d first=%d prop=[",
                         static_cast<unsigned long long>(spec_steps_),
                         static_cast<long long>(pos), d_eff, first);
            for (i32 j = 0; j < d_eff; j++)
                std::fprintf(stderr, "%s%d", j ? "," : "",
                             prop[static_cast<size_t>(j)]);
            std::fprintf(stderr, "] row=[");
            for (i32 j = 0; j < d_eff; j++)
                std::fprintf(stderr, "%s%d", j ? "," : "",
                             row_ids[static_cast<size_t>(j)]);
            std::fprintf(stderr, "] accept=%d full=%d", a, tr_full ? 1 : 0);
            if (!tr_full)
                std::fprintf(stderr, " REJECTED row %d predicted %d against "
                             "prop[%d]=%d", a + 1, row_pred(a + 1), a + 1,
                             prop[static_cast<size_t>(a)]);
            std::fputc(10, stderr);
        }
        const bool full = (a == d_eff);

        // Emit the accepted proposals, respecting max_tokens.
        bool stop_hit = false;
        i32 emitted = 0;
        for (i32 j = 0; j < a; j++) {
            if (res->generated >= p.max_tokens) break;
            const i32 t = prop[static_cast<size_t>(j)];
            draft_accepted_++;
            stop_hit = emit.accept(t);
            emitted++;
            if (stop_hit) break;
        }

        if (stop_hit) {
            kv_rollback(pos + emitted);
            finish_run(FinishStop);
            return true;
        }
        if (full && res->generated < p.max_tokens) {
            // Full match: row d predicts the bonus token for pos+d. Emit it,
            // then give it a KV row (one plain decode step for d+1 tokens).
            const i32 bonus = row_pred(d_eff);
            if (bonus == tok_.eos()) {
                kv_rollback(pos + emitted);
                finish_run(FinishEos);
                return true;
            }
            emit.accept(bonus);
            emitted++;
            if (pos + emitted < kv_cap_ && res->generated < p.max_tokens) {
                const i32 b = res->tokens.back();
                if (!forward(&b, 1, static_cast<i32>(pos + emitted - 1), true)) {
                    res->error = forward_failure_error();
                    finish_run(FinishError);
                    return false;
                }
                fetch_logits();
            }
        }
        if (full && res->generated >= p.max_tokens) {
            kv_rollback(pos + emitted);
        }
        // A partial accept leaves the target's KV holding rows for rejected
        // proposals beyond the emitted prefix; the cache must shrink back to
        // the accepted prefix or the next round would attend to stale rows.
        if (!full) kv_rollback(pos + emitted);
        pos += emitted;

        if (res->generated >= p.max_tokens) {
            finish_run(FinishLength);
            return true;
        }
        if (pos + 1 >= kv_cap_) {
            finish_run(FinishContext);
            return true;
        }

        // ---- realign the draft with what the target kept ------------------
        // Both models must end the round in the same state: KV [0, pos),
        // logits predicting pos. Re-feeding the last emitted token regenerates
        // the draft's distribution exactly.
        const i32 last_tok = res->tokens.back();
        // The last emitted token's slot in the target's cache is pos-1, and
        // the draft has to be given it there -- see
        // spec_realign_at_natural_slot above.
        const i32 rslot = spec_realign_at_natural_slot() ? pos - 1 : pos;
        if (!draft.kv_rollback(rslot)) {
            res->error = "draft KV rollback refused while realigning the "
                         "accepted prefix";
                finish_run(FinishError);
                return false;
            }
        if (!draft.forward(&last_tok, 1, static_cast<i32>(rslot), true)) {
            res->error = "draft KV promotion refused by the host-memory reserve or "
                         "an unavailable WARM/COLD page";
            finish_run(FinishError);
            return false;
        }
        draft.fetch_logits();
        // The target's logits must predict pos. After a full match they came
        // from the bonus forward above. After a partial match they are block row
        // `emitted` (1-based), and that row is NO LONGER sitting in rows[]: its
        // argmax is now taken on the device, so nothing downloads the whole
        // block any more. Recompute and download that single row instead. The
        // block's activations are still in ws_x_ on the partial path -- the
        // bonus forward that would overwrite them runs only when `full` -- so
        // this is one head gemm and one download where there used to be d_eff
        // downloads of 993 kB each. Downloaded raw, exactly as the memcpy it
        // replaces did: the softcap is monotone, so it cannot move an argmax.
        if (!full && emitted >= 1) {
            head_compute(emitted - 1);
            if (spec_row_sync()) be_->sync();
            be_->download_f32(logits_host_, ws_logits_, n_vocab_);
        }
    }

    finish_run(res->generated >= p.max_tokens ? FinishLength : FinishContext);
    return true;
}

// The DFlash round. Same shape as generate_speculative -- greedy-only, one
// batched verification per round, longest matching prefix kept -- with the
// proposal side replaced. A DFlash drafter cannot be asked "what comes next":
// it has no embedding and no head, and its context is a cache of fused target
// features rather than its own tokens. So a round is
//
//   * one masked block over [id_last, MASK x n_draft], read at the target's
//     positions, giving a candidate per row;
//   * the same batched verification the model-draft path uses; and
//   * an injection of the *accepted* rows' features, which is what keeps the
//     drafter's cache in step with the target after a partial accept.
//
// The capture that feeds the last step is taken inside forward_core, so the
// features a round injects are the ones the verification pass computed for
// exactly the tokens the target kept. Nothing is recomputed.
bool Engine::generate_speculative_dflash(const GenerateParams &p, GenerateResult *res,
                                         const std::vector<i32> &ids, f64 *prefill_ms,
                                         f64 *decode_ms, i64 *decode_steps) {
    // Block size caps the window: the block is id_last plus one mask per
    // candidate, and the reference clamps to block_size - 1 for that reason.
    const i32 k = std::max<i32>(1, std::min<i32>(draft_tokens_,
                                                 dflash_->block_size() - 1));

    // ---- prefill ---------------------------------------------------------
    // The target and its drafter advance together, chunk by chunk: forward()
    // captures the residual streams the fusion below reads, and commit() injects
    // them before the next chunk overwrites the capture buffer.
    expert_phase(ExpertCache::PhasePrefill);
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        if (!forward(ids.data() + i, n, static_cast<i32>(i), last)) {
            res->error = forward_failure_error();
            res->finish = FinishError;
            return false;
        }
        if (!dflash_->commit(static_cast<i32>(i), n)) return false;
    }
    fetch_logits();
    *prefill_ms = prefill_timer.ms();
    expert_phase(ExpertCache::PhaseDecode);

    Sampler sampler;
    sampler.reset(p.sampler);
    sampler.accept(ids.data(), static_cast<int>(ids.size()));
    TokenEmitter emit(p, tok_, sampler, res, ids);

    const Timer decode_timer;
    i64 timed_decode_steps = 0;
    auto finish_run = [&](Finish f) {
        res->finish = f;
        *decode_ms = decode_timer.ms();
        *decode_steps = timed_decode_steps;
        emit.finish(f == FinishStop);
    };

    i64 pos = static_cast<i64>(ids.size());
    std::vector<i32> prop(static_cast<size_t>(k));
    std::vector<f32> rows(static_cast<size_t>(k) * static_cast<size_t>(n_vocab_));
    std::vector<f32> dlogits(static_cast<size_t>(k) *
                             static_cast<size_t>(n_vocab_));
    // KRK_SPEC_RS state: the run's sampler chain applied per row, and the
    // uniforms of the accept/resample draws.
    const bool rs = spec_rs_on() && !p.sampler.greedy && p.sampler.temp > 0.0f;
    Rng rsrng;
    rsrng.seed(p.sampler.seed ^ 0xD1CEC0FFEEull);
    std::vector<i32> sids;
    std::vector<f64> sps;
    const QuantTensor &embd = model_.tok_embd();
    const QuantTensor &head = model_.out_head();

    // Invariant at the top of every round, identical to the model-draft path:
    // the target's KV holds [0, pos) and logits_host_ predicts pos.
    while (res->generated < p.max_tokens && pos < kv_cap_) {
        // One speculation round, wall to wall: propose, pre-check, verify,
        // emit. The device op table cannot delimit this because a round is
        // several ops plus host work between them.
        KRK_TIME_HOST("spec.round");
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, res->generated, pos, logits_host_, n_vocab_);
        const i32 first = argmax_of(logits_host_, n_vocab_);
        // The block opens with the last committed token: the prompt's last token
        // after the prefill, and the last emitted one afterwards.
        const i32 id_last = res->tokens.empty() ? ids.back() : res->tokens.back();

        // ---- propose a block ---------------------------------------------
        const i64 room = std::min<i64>(kv_cap_, dflash_->kv_capacity()) - pos;
        i32 d = static_cast<i32>(std::max<i64>(1, std::min<i64>(k, room)));
        const i32 nrows = dflash_->draft_block(embd, head, n_vocab_, id_last, d,
                                               static_cast<i32>(pos), dlogits.data());
        if (nrows <= 0 || pos <= 0) {
            // Nothing to draft against (an empty context, or a drafter that
            // refused the block). Fall back to the plain step, which keeps the
            // two paths identical in output rather than merely similar.
            if (first == tok_.eos()) {
                finish_run(FinishEos);
                return true;
            }
            emit.accept(first);
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            if (!forward(&first, 1, static_cast<i32>(pos), true)) {
                res->error = forward_failure_error();
                finish_run(FinishError);
                return false;
            }
            dflash_->commit(static_cast<i32>(pos), 1);
            pos++;
            fetch_logits();
            continue;
        }
        d = nrows;
        for (i32 j = 0; j < d; j++) {
            const f32 *drow = dlogits.data() +
                              static_cast<size_t>(j) *
                                  static_cast<size_t>(n_vocab_);
            if (rs) {
                sampler_support(drow, p.sampler, n_vocab_, sids, sps);
                prop[static_cast<size_t>(j)] = support_draw(sids, sps, rsrng);
            } else {
                prop[static_cast<size_t>(j)] = argmax_of(drow, n_vocab_);
            }
        }
        draft_proposed_ += static_cast<u64>(d);
        spec_steps_++;
        timed_decode_steps++;   // a speculation round is this loop's unit of work

        if (p.debug_topk > 0 && d > 0) {
            // After the block's proposals, dump the block's own row 0 top-3 next
            // to the target's top-3 at this position, so a drafter whose
            // candidates are merely elsewhere on the row and one whose row is flat
            // read differently in one run, and so the layer-level attention/injection
            // dump in dflash.cpp can be read against them to find whether the
            // block's own hidden state is the problem.
            std::fprintf(stderr,
                         "[dfkv] block_row0 pos=%lld top=[",
                         static_cast<long long>(pos));
            print_row_top3(stderr, dlogits.data(), n_vocab_);
            std::fputc(10, stderr);
        }

        // ---- the same pre-check plain decode performs ---------------------
        // On disagreement the round costs exactly one plain step and the block
        // is never verified.
        if (!rs && spec_trace()) {
            std::fprintf(stderr,
                         "[dflash] round %llu pos=%lld PRE-CHECK FAILED d=%d "
                         "first(target)=%d prop[0](draft)=%d prop=[",
                         static_cast<unsigned long long>(spec_steps_),
                         static_cast<long long>(pos), d, first, prop[0]);
            for (i32 j = 0; j < d; j++)
                std::fprintf(stderr, "%s%d", j ? "," : "",
                             prop[static_cast<size_t>(j)]);
            // The target's own row for this position, in the same line. The
            // comparison used to take a second tool and a grep of an untagged
            // dump; a drafter whose candidates are simply elsewhere on the row
            // and one whose row is flat need to be told apart in one run.
            std::fprintf(stderr, "] target_top3=[");
            print_row_top3(stderr, logits_host_, n_vocab_);
            std::fputc(93, stderr);
            {
                // How far the target's token sits on the draft's own row 0:
                // rank 3 means the answer is present but outranked, a rank in
                // the thousands means the row never encoded it at all.
                const f32 fv = dlogits[static_cast<size_t>(first)];
                i64 rk = 0;
                f64 ms = 0.0;
                f32 mx = -1e30f;
                for (i64 v = 0; v < n_vocab_; v++) {
                    const f32 x = dlogits[static_cast<size_t>(v)];
                    if (x > fv) rk++;
                    ms += x;
                    if (x > mx) mx = x;
                }
                std::fprintf(stderr,
                             " draft_row0: rank(first)=%lld/%lld max=%.4g "
                             "mean=%.4g",
                             static_cast<long long>(rk),
                             static_cast<long long>(n_vocab_),
                             static_cast<f64>(mx),
                             ms / static_cast<f64>(n_vocab_));
            }
            std::fputc(10, stderr);
        }
        // Position 0 is scored on BOTH paths -- the pre-check compares it --
        // so the sampled-accept accounting takes it here, before the mismatch
        // branch can bail; the verify path below adds positions 1..d_eff-1 on
        // the rounds that get that far. kraken skips the rest of the block on
        // a pre-check miss, so those rows never exist to score: the reported
        // number is a lower bound of a full-block verifier's, same as greedy
        // agreement is of sampled acceptance.
        if (sampled_acc_on() && d > 0) {
            draft_sampled_e_ += sampled_accept_prob(
                logits_host_, dlogits.data(), static_cast<i64>(n_vocab_));
            draft_sampled_n_ += 1;
        }
        if (!rs && first != prop[0]) {
            if (first == tok_.eos()) {
                finish_run(FinishEos);
                return true;
            }
            emit.accept(first);
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            if (!forward(&first, 1, static_cast<i32>(pos), true)) {
                res->error = forward_failure_error();
                finish_run(FinishError);
                return false;
            }
            dflash_->commit(static_cast<i32>(pos), 1);
            pos++;
            fetch_logits();
            continue;
        }
        d = nrows;

        // ---- verify: one batched forward of the d proposals ---------------
        i32 d_eff = d;
        while (d_eff > 0 && pos + d_eff > kv_cap_) d_eff--;
        if (d_eff <= 0) {
            finish_run(FinishContext);
            return true;
        }
        if (!forward_core(prop.data(), d_eff, static_cast<i32>(pos), LogitsNone)) {
            res->error = forward_failure_error();
            finish_run(FinishError);
            return false;
        }
        kv_pos_ = pos + d_eff;
        // Device argmax per verified row; see generate_speculative for why the
        // full-row download was the cost. With the sampled-accept accounting
        // on (the default) each row is downloaded once anyway and carries the
        // argmax and the accounting on the same bytes -- a device argmax and a
        // host argmax of one f32 row agree, so the generated text is
        // bit-identical either way (KRK_SPEC_SAMPLED_ACC=0 keeps the fast
        // path for a pure timing run).
        const bool acc_on = sampled_acc_on();
        std::vector<i32> row_ids(static_cast<size_t>(d_eff), 0);
        for (i32 j = 0; j < d_eff; j++) {
            head_compute(j);
            i32 id = 0;
            if (acc_on || !topk_argmax(&id)) {
                be_->sync();
                be_->download_f32(rows.data() + static_cast<size_t>(j) * n_vocab_,
                                  ws_logits_, n_vocab_);
                id = argmax_of(rows.data() + static_cast<size_t>(j) * n_vocab_,
                               n_vocab_);
            }
            row_ids[static_cast<size_t>(j)] = id;
        }
        auto row_pred = [&](i32 j) {
            return row_ids[static_cast<size_t>(j - 1)];
        };
        if (acc_on && d_eff > 1) {
            // Positions 1..d_eff-1 of this round; position 0 was accounted
            // before the pre-check. The target's row verifying prop[j] is
            // rows[j-1]; draft row j predicts prop[j].
            for (i32 j = 1; j < d_eff; j++) {
                const f32 *trow = rows.data() +
                                  static_cast<size_t>(j - 1) * n_vocab_;
                const f32 *drow = dlogits.data() +
                                  static_cast<size_t>(j) * n_vocab_;
                draft_sampled_e_ += sampled_accept_prob(
                    trow, drow, static_cast<i64>(n_vocab_));
            }            draft_sampled_n_ += static_cast<u64>(d_eff - 1);
        }
        if (rs) {
            // Rejection-sampling verification: no pre-check, no argmax match.
            // Each proposal is tested against the target's row of the same
            // position by probability ratio; a rejection resamples that
            // position from the corrected distribution and ends the round.
            i32 ra = 0;       // accepted proposals
            i32 rtoken = -1;  // resampled replacement on rejection
            i32 rej = -1;     // rejection position, -1 = none
            std::vector<i32> pids, qids;
            std::vector<f64> pps, qps;
            for (i32 j = 0; j < d_eff; j++) {
                const f32 *prow = j == 0
                                      ? logits_host_
                                      : rows.data() + static_cast<size_t>(j - 1) *
                                                          n_vocab_;
                const f32 *qrow = dlogits.data() +
                                  static_cast<size_t>(j) * n_vocab_;
                sampler_support(prow, p.sampler, n_vocab_, pids, pps);
                sampler_support(qrow, p.sampler, n_vocab_, qids, qps);
                const i32 pt = prop[static_cast<size_t>(j)];
                const f64 pp = support_prob(pids, pps, pt);
                const f64 qq = support_prob(qids, qps, pt);
                const f64 u = static_cast<f64>(rsrng.next_f32());
                if (pp >= qq || (qq > 0.0 && u <= pp / qq)) {
                    ra++;
                    continue;
                }
                rej = j;
                rtoken = corrected_draw(pids, pps, qids, qps, rsrng);
                break;
            }
            if (spec_trace()) {
                std::fprintf(stderr,
                             "[dflash] rs round %llu pos=%lld a=%d rej=%d "
                             "rtoken=%d prop0=%d\n",
                             static_cast<unsigned long long>(spec_steps_),
                             static_cast<long long>(pos), ra, rej, rtoken,
                             prop[0]);
            }
            // Emit the accepted prefix, then the resampled token or the
            // bonus -- the same bookkeeping as the greedy round below.
            bool stop_hit = false;
            i32 emitted = 0;
            for (i32 j = 0; j < ra; j++) {
                if (res->generated >= p.max_tokens) break;
                const i32 t = prop[static_cast<size_t>(j)];
                draft_accepted_++;
                stop_hit = emit.accept(t);
                emitted++;
                if (stop_hit) break;
            }
            if (ra > 0 && !dflash_->commit(static_cast<i32>(pos), ra))
                return false;
            if (stop_hit) {
                kv_rollback(pos + emitted);
                finish_run(FinishStop);
                return true;
            }
            i32 tail = -1;
            if (rej >= 0) {
                tail = rtoken;
            } else if (res->generated < p.max_tokens) {
                // All proposals accepted: the bonus is the target's own draw
                // after the last one.
                const f32 *brow = rows.data() +
                                  static_cast<size_t>(d_eff - 1) * n_vocab_;
                sampler_support(brow, p.sampler, n_vocab_, pids, pps);
                tail = support_draw(pids, pps, rsrng);
            }
            if (tail >= 0 && res->generated < p.max_tokens) {
                if (tail == tok_.eos()) {
                    kv_rollback(pos + emitted);
                    finish_run(FinishEos);
                    return true;
                }
                emit.accept(tail);
                emitted++;
                // The accepted rows' KV is kept by the verify pass; a
                // rejected position's was written for the wrong token, so
                // rewind to the accepted prefix and feed the replacement --
                // which also restores the round invariant (logits predict
                // pos + emitted).
                kv_rollback(pos + ra);
                if (!forward(&tail, 1, static_cast<i32>(pos + ra), true)) {
                    res->error = forward_failure_error();
                    finish_run(FinishError);
                    return false;
                }
                dflash_->commit(static_cast<i32>(pos + ra), 1);
                fetch_logits();
            } else if (res->generated >= p.max_tokens) {
                kv_rollback(pos + emitted);
                finish_run(FinishLength);
                return true;
            }
            pos += emitted;
            if (res->generated >= p.max_tokens) {
                finish_run(FinishLength);
                return true;
            }
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            continue;
        }
        i32 a = 1;
        while (a < d_eff && row_pred(a) == prop[static_cast<size_t>(a)]) a++;
        if (spec_trace()) {
            const bool tr_full = (a == d_eff);
            std::fprintf(stderr, "[dflash] round %llu pos=%lld d=%d first=%d prop=[",
                         static_cast<unsigned long long>(spec_steps_),
                         static_cast<long long>(pos), d_eff, first);
            for (i32 j = 0; j < d_eff; j++)
                std::fprintf(stderr, "%s%d", j ? "," : "",
                             prop[static_cast<size_t>(j)]);
            std::fprintf(stderr, "] row=[");
            for (i32 j = 0; j < d_eff; j++)
                std::fprintf(stderr, "%s%d", j ? "," : "",
                             row_ids[static_cast<size_t>(j)]);
            std::fprintf(stderr, "] accept=%d full=%d", a, tr_full ? 1 : 0);
            if (!tr_full)
                std::fprintf(stderr, " REJECTED row %d predicted %d against "
                             "prop[%d]=%d", a + 1, row_pred(a + 1), a + 1,
                             prop[static_cast<size_t>(a)]);
            std::fputc(10, stderr);
        }
        const bool full = (a == d_eff);

        // The drafter's cache takes the features of the rows that were actually
        // kept. The verification pass captured them at exactly pos .. pos+a-1,
        // so this is the target's own view of the accepted tokens.
        if (!dflash_->commit(static_cast<i32>(pos), a)) return false;

        bool stop_hit = false;
        i32 emitted = 0;
        for (i32 j = 0; j < a; j++) {
            if (res->generated >= p.max_tokens) break;
            const i32 t = prop[static_cast<size_t>(j)];
            draft_accepted_++;
            stop_hit = emit.accept(t);
            emitted++;
            if (stop_hit) break;
        }
        if (stop_hit) {
            kv_rollback(pos + emitted);
            finish_run(FinishStop);
            return true;
        }
        if (full && res->generated < p.max_tokens) {
            const i32 bonus = row_pred(d_eff);
            if (bonus == tok_.eos()) {
                kv_rollback(pos + emitted);
                finish_run(FinishEos);
                return true;
            }
            emit.accept(bonus);
            emitted++;
            if (pos + emitted < kv_cap_ && res->generated < p.max_tokens) {
                const i32 b = res->tokens.back();
                if (!forward(&b, 1, static_cast<i32>(pos + emitted - 1), true)) {
                    res->error = forward_failure_error();
                    finish_run(FinishError);
                    return false;
                }
                dflash_->commit(static_cast<i32>(pos + emitted - 1), 1);
                fetch_logits();
            }
        }
        if (full && res->generated >= p.max_tokens)
            kv_rollback(pos + emitted);
        if (!full) kv_rollback(pos + emitted);
        pos += emitted;

        if (res->generated >= p.max_tokens) {
            finish_run(FinishLength);
            return true;
        }
        if (pos + 1 >= kv_cap_) {
            finish_run(FinishContext);
            return true;
        }
        // Row `emitted` (1-based) predicts pos; see generate_speculative for
        // why it is recomputed rather than copied out of rows[].
        if (!full && emitted >= 1) {
            head_compute(emitted - 1);
            if (spec_row_sync()) be_->sync();
            be_->download_f32(logits_host_, ws_logits_, n_vocab_);
        }
    }

    finish_run(res->generated >= p.max_tokens ? FinishLength : FinishContext);
    return true;
}

std::string Engine::kv_resource_error() const {
    return "KV promotion refused by the shared host-memory reserve or an "
           "unavailable WARM/COLD page; reduce --ctx, raise --kv-hot-mb, free "
           "host RAM, or configure a writable --kv-cold-dir";
}

std::string Engine::expert_resource_error() const {
    return "expert cache refused a routed expert: device pool has no free "
             "cell beside its floor and no resident slot could be reclaimed; "
             "reduce --ctx/--kv-mb, raise --expert-cache-mb if headroom exists, "
             "free VRAM, or set KRK_VRAM_FLOOR_MB=0 to let the pool take the "
             "card's last memory";
}

std::string Engine::forward_failure_error() const {
    return forward_error_.empty() ? kv_resource_error() : forward_error_;
}

// Attributes the expert tier's stage timers to the phase running now. The
// report splits prefill from decode because the same promotion DMA behind a
// 256-row prefill chunk is not the same cost as one behind a single decode
// row, and the merged row cannot say which side a fix belongs on. The draft
// engine's tier moves with the target's, so both are attributed together.
void Engine::expert_phase(ExpertCache::Phase p) {
    model_.experts().set_phase(p);
    if (draft_) draft_->model_.experts().set_phase(p);
}

bool Engine::generate(const GenerateParams &p, GenerateResult *res) {
    KRK_TRACE_COMP("generate");
    if (!be_ || n_vocab_ <= 0) return false;
    *res = GenerateResult{};

    // A generation always prefills from position 0, so it *is* a new sequence.
    // The prefill overwrites the KV rows it touches, but the cursor is a
    // high-water mark: left alone it would keep pointing past this run's
    // prompt, and attention would read the previous run's keys and values. A
    // recurrent model has the sharper version of the same problem — its
    // delta-rule states and conv windows cannot be reconstructed by a prefill
    // at all, so they would carry the previous run's history into this one.
    // (The server's per-request kv_rollback(0) agrees; this just makes the
    // primitive correct on its own.)
    kv_pos_ = 0;
    mtp_fed_ = 0;
    if (trace_hash_on_) trace_stage_.clear();
    if (model_.is_recurrent()) recurrent_reset();
    if (!comp_nblocks_.empty()) compressed_reset();
    if (moe_predict_enabled_) moe_predictor_.reset();
    moe_predict_ids_.clear();

    // ---- tokenize ---------------------------------------------------------
    // Timed inside Tokenizer::encode, which covers every caller.
    std::vector<i32> ids;
    if (tokenize_prompt(p.prompt, ids) < 0) return false;
    if (ids.empty()) {
        KRK_ERROR("prompt encoded to zero tokens");
        return false;
    }
    if (static_cast<i64>(ids.size()) > kv_cap_) {
        ids.resize(static_cast<size_t>(kv_cap_));
        KRK_WARN("prompt truncated to the %lld-token context",
                 static_cast<long long>(kv_cap_));
    }
    res->prompt_tokens = static_cast<i32>(ids.size());

    // ---- masked diffusion: a different generation, not a different model ---
    // A `diffuse` file is denoised, not decoded: the sequence starts as
    // `[prompt | MASK ...]` and each step is ONE bidirectional forward over the
    // whole thing (src/diffusion.cpp). Every path below this point assumes a
    // causal decode loop -- KV prefix reuse, the speculative drafters, the
    // per-token stop check -- so the branch is taken here, before any of them,
    // and none of them has to know the family exists.
    if (model_.cfg().diffusion) return generate_diffusion(p, res, ids);

    // ---- KV prefix reuse ---------------------------------------------------
    // The rows [0, reuse) already hold exactly these tokens at these positions,
    // because the caller rolled the cache back to a prefix it verified and told
    // us how long it is. Prefilling them again would be the same arithmetic
    // done twice, so the loop below starts at `reuse` -- and at least one token
    // is always left, because the row this run samples from is the one the last
    // prefill call produces.
    //
    // The two exclusions are not politeness: a recurrent model's state cannot be
    // reconstructed by rolling the cursor back, and the speculative paths take
    // `ids` and prefill the prompt themselves, so honouring the skip there would
    // leave them prefilling from 0 into a cursor that starts at `reuse`.
    if (p.reuse_tokens > 0) {
        if (mtp_spec_ || head_ || draft_ || dflash_) {
            KRK_WARN("KV prefix reuse is off for this request: a drafter prefills "
                     "the prompt itself");
        } else if (model_.is_recurrent()) {
            KRK_WARN("KV prefix reuse is off for this request: a recurrent model's "
                     "state cannot be restored by a rollback");
        } else {
            const i64 reuse = std::min<i64>(
                p.reuse_tokens, static_cast<i64>(ids.size()) - 1);
            if (reuse > 0) {
                kv_pos_ = reuse;
                res->reused_tokens = static_cast<i32>(reuse);
            }
        }
    }

    // The nextn block is a drafter that ships inside the target's own file, so
    // it is tried before the external draft engines; --draft and --mtp-spec are
    // alternatives, and a run with both loaded uses the block.
    // A hybrid target is allowed when its recurrent state could be snapshotted:
    // the round restores it and replays the accepted prefix, which is what the
    // delta rule cannot do through kv_rollback.
    if (mtp_spec_ && p.sampler.greedy &&
        (!model_.is_recurrent() || mtp_rec_snap_ != nullptr)) {
        f64 prefill_ms = 0, decode_ms = 0;
        i64 decode_steps = 0;
        const bool ok = generate_speculative_mtp(p, res, ids, &prefill_ms,
                                                 &decode_ms, &decode_steps);
        res->prefill_ms = prefill_ms;
        res->decode_ms = decode_ms;
        res->decode_steps = decode_steps;
        return ok;
    }
    if (mtp_spec_ && !p.sampler.greedy)
        KRK_WARN("MTP speculation is greedy-only; sampling runs without the block");
    else if (mtp_spec_ && model_.is_recurrent() && !mtp_rec_snap_)
        KRK_WARN("MTP speculation needs a recurrent-state snapshot for this "
                 "hybrid target and could not allocate one; decoding plainly");

    // The external head is a third drafter and goes first: it needs no draft
    // model and no block inside the target. It is the one drafter that also
    // works under sampling -- its round verifies with the distribution-
    // preserving rule rather than a prefix match.
    if (head_ && (p.sampler.greedy || p.sampler.temp > 0.0f)) {
        f64 prefill_ms = 0, decode_ms = 0;
        i64 decode_steps = 0;
        const bool ok = generate_speculative_head(p, res, ids, &prefill_ms,
                                                  &decode_ms, &decode_steps);
        res->prefill_ms = prefill_ms;
        res->decode_ms = decode_ms;
        res->decode_steps = decode_steps;
        return ok;
    }
    if (head_ && !p.sampler.greedy && p.sampler.temp <= 0.0f) {
        KRK_WARN("the draft head was not used for this request: sampler.temp is "
                 "not positive, which is the greedy limit, and greedy with a head "
                 "is taken by the branch above");
    }

    // Greedy-only fast path: with a draft loaded and argmax sampling requested,
    // speculative decoding produces bit-identical output to the plain loop, so
    // there is nothing to choose between them — take the faster one.
    if ((draft_ || dflash_) && p.sampler.greedy && !model_.is_recurrent()) {
        f64 prefill_ms = 0, decode_ms = 0;
        i64 decode_steps = 0;
        const bool ok = dflash_
                            ? generate_speculative_dflash(p, res, ids, &prefill_ms,
                                                          &decode_ms, &decode_steps)
                            : generate_speculative(p, res, ids, &prefill_ms,
                                                   &decode_ms, &decode_steps);
        res->prefill_ms = prefill_ms;
        res->decode_ms = decode_ms;
        res->decode_steps = decode_steps;
        return ok;
    }
    if (dflash_ && !p.sampler.greedy && p.sampler.temp > 0.0f && spec_rs_on()) {
        // KRK_SPEC_RS=1: the dflash loop's rejection-sampling verifier takes
        // sampling runs too -- proposals sampled, verified by probability
        // ratio, resampled on rejection.
        f64 prefill_ms = 0, decode_ms = 0;
        i64 decode_steps = 0;
        const bool ok = generate_speculative_dflash(p, res, ids, &prefill_ms,
                                                    &decode_ms, &decode_steps);
        res->prefill_ms = prefill_ms;
        res->decode_ms = decode_ms;
        res->decode_steps = decode_steps;
        return ok;
    }
    if ((draft_ || dflash_) && !p.sampler.greedy) {
        KRK_WARN("speculative decoding is greedy-only; sampling runs without "
                 "the draft for this request");
    }

    // ---- prefill ----------------------------------------------------------
    // Greedy decoding, and nothing that intends to read the host logits row
    // afterwards: the device top-k leaves the row in VRAM. --debug-topk dumps
    // that row, so it asks for the host copy and stays a faithful diff tool.
    const bool device_topk = p.sampler.greedy && p.debug_topk <= 0 &&
                            !p.sample_host;
    expert_phase(ExpertCache::PhasePrefill);
    // Phase stamp: every span and kernel from here to the end of the prompt is
    // charged to `prefill`, so the component table ranks prefill against
    // prefill instead of against a load that happened once. Back to decode just
    // after the logits fetch.
    krk::HostTime::set_phase(krk::HostTime::kPrefill);
    const Timer prefill_timer;
    for (i64 i = res->reused_tokens; i < static_cast<i64>(ids.size());
         i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        KRK_TIME_HOST("prefill.chunk");
        if (!forward(ids.data() + i, n, static_cast<i32>(i), last)) {
            res->error = forward_failure_error();
            res->finish = FinishError;
            return false;
        }
    }
    fetch_logits(device_topk);
    krk::HostTime::set_phase(krk::HostTime::kDecode);
    if (expert_scan_ || expert_track_)
        expert_scan_tokens_ += static_cast<i64>(ids.size());
    // The MTP probe runs here, on the prompt's last row, and before the decode
    // loop touches anything: it is a measurement of the head, not a decoder,
    // and generation continues exactly as it would have without it.
    // KRK_MTP_PROBE_ALL=1 additionally probes every decode step, which is what
    // turns one anecdote into an acceptance rate; the block's KV is incremental
    // (see mtp_probe_step), so a step costs one block forward.
    const bool mtp_all = mtp_probe_report_ && [] {
        const char *e = std::getenv("KRK_MTP_PROBE_ALL");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    i32 mtp_steps = 0, mtp_hits = 0;
    // Buckets over the drafter's rank for the target's token. They are the
    // diagnostic, not the headline: see MtpProbe::target_rank.
    i32 mtp_rank_le2 = 0, mtp_rank_le5 = 0, mtp_rank_le20 = 0,
        mtp_rank_le100 = 0, mtp_rank_far = 0;
    i32 mtp_cap_steps = 0, mtp_cap_hits = 0;
    auto run_mtp_probe = [&]() {
        if (!mtp_probe_report_ || !mtp_probe_enabled()) return;
        const MtpProbe mp = mtp_probe_step();
        if (!mp.ok) {
            KRK_WARN("mtp probe: did not run -- %s", mp.note.c_str());
            return;
        }
        const i32 dids[1] = {mp.draft_id};
        const i32 tids[1] = {mp.target_id};
        const std::string dtext = tok_.decode(dids, 1);
        const std::string ttext = tok_.decode(tids, 1);
        mtp_steps++;
        if (mp.draft_id == mp.target_id) mtp_hits++;
        if (mp.capture_id >= 0) {
            mtp_cap_steps++;
            if (mp.capture_id == mp.target_id) mtp_cap_hits++;
            std::fprintf(stderr, "[mtp   ] pos %d: capture %d \"%s\" (%.4f) | %s\n",
                         mp.pos, mp.capture_id, tok_.decode(&mp.capture_id, 1).c_str(),
                         static_cast<f64>(mp.capture_logit),
                         mp.capture_id == mp.target_id ? "same row" : "DIFFERENT");
        }
        if (mp.target_rank <= 2) mtp_rank_le2++;
        if (mp.target_rank <= 5) mtp_rank_le5++;
        if (mp.target_rank <= 20) mtp_rank_le20++;
        if (mp.target_rank <= 100) mtp_rank_le100++;
        else mtp_rank_far++;
        std::fprintf(stderr,
                     "[mtp   ] pos %d: draft %d \"%s\" (%.4f) | target %d "
                     "\"%s\" (%.4f)| rank %d (%.4f) | %s | %.1f ms\n",
                     mp.pos, mp.draft_id, dtext.c_str(),
                     static_cast<f64>(mp.draft_logit), mp.target_id,
                     ttext.c_str(), static_cast<f64>(mp.target_logit),
                     mp.target_rank, static_cast<f64>(mp.target_in_draft),
                     mp.draft_id == mp.target_id ? "MATCH" : "miss", mp.ms);
    };
    run_mtp_probe();
    res->prefill_ms = prefill_timer.ms();
    expert_phase(ExpertCache::PhaseDecode);

    Sampler sampler;
    sampler.reset(p.sampler);
    sampler.accept(ids.data(), static_cast<int>(ids.size()));
    TokenEmitter emit(p, tok_, sampler, res, ids);

    i64 pos = static_cast<i64>(ids.size());
    const Timer decode_timer;
    i64 timed_decode_steps = 0;

    for (i32 gen = 0; gen < p.max_tokens; gen++) {
        // The whole step: host sampling, the forward, and the logits fetch.
        // Under --profile this is where "the CPU is busy and the GPU is idle"
        // is either visible or not.
        KRK_TIME_HOST("decode.step");
        // On-demand component snapshot (KRK_PROFILE_SNAPSHOT / --profile-snapshot
        // N): the tables are printed at teardown, which is useless during an
        // hour-long server run. This prints them every N decode steps.
        if (profile_snapshot_every() > 0 && gen > 0 &&
            gen % profile_snapshot_every() == 0 &&
            krk::HostTime::get().on())
            krk::HostTime::get().report_components(stderr);
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, gen, pos, logits_host_, n_vocab_);
        // The fast path hands back the argmax the device already computed.
        // Greedy skips the repetition penalty by design (Sampler::sample does
        // the same), so this is the identical token, not an approximation.
        i32 token = 0;
        {
        // _N, not the default name: this sits inside decode.step's scope, and a
        // second `krk_host_scope` would shadow the outer one (the sample span
        // would close the step's). The rows are meant to nest.
        KRK_TIME_HOST_N("decode.sample", krk_host_scope_sample);
        token = topk_valid_ ? topk_id_
                            : sampler.sample(logits_host_, n_vocab_);
        }
        if (token == tok_.eos()) {
            res->finish = FinishEos;
            break;
        }

        if (emit.accept(token)) {
            res->finish = FinishStop;
            break;
        }
        if (gen + 1 >= p.max_tokens) {
            res->finish = FinishLength;
            break;
        }
        if (pos >= kv_cap_) {
            res->finish = FinishContext;
            break;
        }
        const bool split = split_timing_enabled();
        auto t0 = std::chrono::steady_clock::now();
        {
            // Split so the host/device question has an answer under --profile:
            // a step whose forward is small but whose decode.step is large is
            // host-bound, and the gap between the two rows is host-side work
            // the device table does not see.
            KRK_TIME_HOST_N("decode.forward", hs_fwd);
            if (!forward(&token, 1, static_cast<i32>(pos), true)) {
                res->error = forward_failure_error();
                res->finish = FinishError;
                return false;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        pos++;
        {
            KRK_TIME_HOST_N("decode.fetch", hs_fetch);
            fetch_logits(device_topk);
        }
        if (mtp_all) run_mtp_probe();
        timed_decode_steps++;
        if (split) {
            const auto t2 = std::chrono::steady_clock::now();
            g_split_fwd_us +=
                std::chrono::duration<double, std::micro>(t1 - t0).count();
            g_split_fetch_us +=
                std::chrono::duration<double, std::micro>(t2 - t1).count();
            g_split_steps++;
        }
        // One token position folded into the routing statistics.
        if (expert_scan_ || expert_track_) expert_scan_tokens_++;
    }

    emit.finish(res->finish == FinishStop);

    if (mtp_probe_report_ && mtp_steps > 0) {
        // An ablation arm reports a number that is NOT the block's: say so
        // before the number, or the run reads as a regression.
        if (mtp_abl() != MtpAbl::None)
            KRK_INFO("mtp probe: ABLATION arm -- %s. This is not the block's "
                     "accuracy, it is one sublayer short of it.",
                     mtp_abl_name(mtp_abl()));
        KRK_INFO("mtp probe: %d step(s), %d accepted (%.1f%%) -- a greedy round "
                 "would have taken %d of %d drafted tokens",
                 mtp_steps, mtp_hits,
                 100.0 * static_cast<f64>(mtp_hits) / static_cast<f64>(mtp_steps),
                 mtp_hits, mtp_steps);
        KRK_INFO("mtp probe: the target's token ranked 2nd or better %d time(s), "
                 "top-5 %d, top-20 %d, top-100 %d, worse %d of %d",
                 mtp_rank_le2, mtp_rank_le5, mtp_rank_le20, mtp_rank_le100,
                 mtp_rank_far, mtp_steps);
        if (mtp_cap_steps > 0)
            KRK_INFO("mtp probe: the captured row predicted the target's token "
                     "%d of %d time(s) (%.1f%%) through the stack's own norm "
                     "and head -- that is the row, without the block",
                     mtp_cap_hits, mtp_cap_steps,
                     100.0 * static_cast<f64>(mtp_cap_hits) /
                         static_cast<f64>(mtp_cap_steps));
        // Row fidelity, bucketed by row index: the question is not whether the
        // block's rows differ from the stack's but whether they drift as the
        // history grows, which is what makes a long attended past hurt.
        if (!rowfid_pos_.empty()) {
            // Half-open row buckets. The last one has no upper bound, and the
            // bound is written as an i64 literal: `1 << 40` is an int shift,
            // i.e. undefined behaviour, which is what this used to be.
            const char *names[4] = {"rows 0-7", "rows 8-15", "rows 16-31",
                                    "rows 32+"};
            const i64 lo_of[4] = {0, 8, 16, 32};
            const i64 hi_of[4] = {8, 16, 32,
                                  std::numeric_limits<i64>::max()};
            for (int b = 0; b < 4; b++) {
                const i64 lo = lo_of[b];
                const i64 hi = hi_of[b];
                i32 n = 0, top1 = 0, cap_top1 = 0;
                f64 cos = 0.0, br = 0.0, cr = 0.0;
                for (size_t i = 0; i < rowfid_pos_.size(); i++) {
                    if (rowfid_pos_[i] < lo || rowfid_pos_[i] >= hi) continue;
                    n++;
                    cos += rowfid_cos_[i];
                    br += rowfid_blk_[i];
                    cr += rowfid_cap_[i];
                    if (rowfid_blk_[i] == 1) top1++;
                    if (rowfid_cap_[i] == 1) cap_top1++;
                }
                if (n == 0) continue;
                KRK_INFO("mtp rows: %-9s %3d row(s) | cos %.3f | block rank "
                         "mean %.1f (top-1 %.0f%%) | capture rank mean %.1f "
                         "(top-1 %.0f%%)",
                         names[b], n, cos / n, br / n, 100.0 * top1 / n,
                         cr / n, 100.0 * cap_top1 / n);
            }
        }
    }
    res->decode_ms = decode_timer.ms();
    res->decode_steps = timed_decode_steps;
    if (g_split_steps > 0) {
        const double n = static_cast<double>(g_split_steps);
        std::fprintf(stderr,
                     "[split ] %lld decode steps | forward %9.1f ms "
                     "(%6.2f ms/step) | fetch_logits %9.1f ms "
                     "(%6.2f ms/step)\n",
                     static_cast<long long>(g_split_steps), g_split_fwd_us / 1e3,
                     g_split_fwd_us / 1e3 / n, g_split_fetch_us / 1e3,
                     g_split_fetch_us / 1e3 / n);
    }
    return true;
}

} // namespace krk
