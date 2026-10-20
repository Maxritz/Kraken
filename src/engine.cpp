// engine.cpp — prefill (batched) + decode (single token) with a paged-free
// contiguous KV cache, streaming output and stop-string handling.
#include "krk/engine.hpp"
#include "krk/arch.hpp"
#include "krk/dflash.hpp"
#include "krk/host_time.hpp"
#include "krk/vram_guard.hpp"
#include "krk/json.hpp" // the expert index reader (warm_experts)
#include "krk/spec_sampling.hpp" // the sampled round's accept/reject rule
#include "par_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sys/stat.h> // struct stat, for the GGUF file-size probe

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
    const u8 *src = t->data + static_cast<size_t>(flat_row) * row_bytes;

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

bool Engine::init(Backend *be, const EngineConfig &cfg, std::string *err) {
    be_ = be;
    cfg_ = cfg;
    // qwen4exp's loader reads its tensors (KRK_ARCH_ALLOW_EXPERIMENTAL=1), but
    // the forward pass still runs a plain `x + f(norm(x))` residual: generating
    // from that would decode fluent text that is not this model. Refuse until
    // the hyper-connection mixer is wired, and say which stage it is.
    if (model_.cfg().hc) {
        if (err)
            *err = "this file ('" + model_.cfg().arch +
                   "') uses hyper-connections, which are loaded but not run yet "
                   "(see docs/AUDIT-qwen4exp-support.md); generation is refused "
                   "rather than run on a plain residual";
        return false;
    }
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
    trace_hash_on_ = cfg.trace_hash;
    trace_stage_.clear();
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
    if (!model_.load(*be_, cfg.model_path, err)) {
        model_.unload();
        be_ = nullptr;
        return false;
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
    q_proj_ = arch_spec(mc.arch).q_output_gate ? 2 * q_dim_ : q_dim_;
    conv_dim_ = model_.conv_dim();
    value_dim_ = model_.value_dim();
    n_vocab_ = mc.n_vocab;
    kv_cap_ = std::min<i64>(cfg_.n_ctx > 0 ? cfg_.n_ctx : mc.n_ctx_train,
                            mc.n_ctx_train > 0 ? mc.n_ctx_train : 32768);
    if (kv_cap_ <= 0) kv_cap_ = 2048;

    chunk_ = std::max<i32>(1, std::min<i32>(cfg_.prefill_chunk, 1024));

    if (!tok_.load(model_.gguf(), err)) {
        model_.unload();
        be_ = nullptr;
        return false;
    }

    const size_t as = be_->act_size();
    const i64 C = chunk_;

    auto alloc = [&](size_t bytes) { return be_->alloc(bytes); };
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
    ws_fexp_ = alloc(static_cast<size_t>(C) * sizeof(i32));
    if (!check_alloc(ws_fexp_, "ws_fexp")) return false;
    be_->fill0(ws_fexp_, static_cast<size_t>(C) * sizeof(i32));

    ws_xn_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    if (!check_alloc(ws_xn_, "ws_xn")) return false;
    ws_x2_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    if (!check_alloc(ws_x2_, "ws_x2")) return false;
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
    if (model_.is_recurrent()) {
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
    if (!logits_host_) return fail_alloc("host logits");

    if (mc.is_moe && mc.n_expert > 0) {
        const i32 k = std::max<i32>(1, mc.n_expert_used);
        ws_router_ = alloc(static_cast<size_t>(C * mc.n_expert) * as);
        if (!check_alloc(ws_router_, "ws_router")) return false;
        ws_ffn_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        if (!check_alloc(ws_ffn_, "ws_ffn")) return false;
        // Batched-expert staging: the plan can never exceed chunk rows because
        // each (token, expert) pair consumes exactly one slot.
        ws_xg_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        if (!check_alloc(ws_xg_, "ws_xg")) return false;
        ws_gateg_ = alloc(static_cast<size_t>(C * ff_ws) * as);
        if (!check_alloc(ws_gateg_, "ws_gateg")) return false;
        ws_upg_ = alloc(static_cast<size_t>(C * ff_ws) * as);
        if (!check_alloc(ws_upg_, "ws_upg")) return false;
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
        (q_proj_ != q_dim_ && !require_alloc(ws_qpack_, "ws_qpack")) ||
        ((model_.is_recurrent() || mc.attn_gate) && !require_alloc(ws_agate_, "ws_agate")) ||
        (model_.had().active && !require_alloc(ws_had_, "ws_had")) ||
        (model_.is_recurrent() &&
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
    // KV residency -- allocated LAST, after the weights and every workspace.
    //
    // That ordering is load-bearing. Reading device_free_bytes() before the
    // workspaces exist reports nearly all free VRAM, so a budget taken here
    // over-commits: the slots are allocated, and then ws_x_ and the rest are
    // allocated on top of them. Subtracting "what the workspaces will take"
    // is not the same thing as measuring it, and guessing it is how an
    // over-commit happens. So: take the measurement after the fact.
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
                             static_cast<size_t>(model_.kv_dim_at(l)) * as;
            kv_len[static_cast<size_t>(l)] = b;
            kv_plane += b;
        }
        const size_t kv_total = kv_plane * 2u;
        kv_load_bytes = kv_total;

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
        if (be_->kind() == BackendKind::CPU) {
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
    if (expert_warmup_wanted()) warm_experts();

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
    if (be_->kind() != BackendKind::CPU && be_->device_total_bytes() > 0 &&
        model_.experts().vram_budget() > 0) {
        const size_t total = be_->device_total_bytes();
        const size_t floor = krk::vram_floor_bytes(total);
        const size_t free_now = be_->device_free_bytes();
        const size_t live = model_.experts().resident_bytes();
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

    // A speculative round runs the block up to `--draft-tokens` times per
    // target forward, and the block is ONE layer, so its experts fit even when
    // the stack's routed set does not. Admitting them once and pinning them is
    // what keeps the propose pass off the promotion path (measured: 11% of the
    // block's acquires promoted while the stack churned). KRK_MTP_PIN=0 is the
    // A/B arm; a set that does not fit is reported and left on the normal
    // policy rather than thrashing a pin it cannot defend.
    // Only a routed block has anything to pin; a dense block's FFN is loaded
    // with the rest of the block and never pages.
    if (mtp_spec_ && model_.mtp().present && model_.mtp().n_expert > 0)
        mtp_pin_experts();

    // An external draft head named in the config loads here: the KV cache and
    // the workspaces exist, so its hidden capture can be sized, and the first
    // forward has not run. A head that was asked for and cannot load is a
    // refusal -- silently running plainly would answer a different command --
    // and the CLI's own call after init then finds it already loaded.
    if (!cfg_.draft_head.empty()) {
        std::string why;
        if (!load_draft_head(cfg_.draft_head, &why)) {
            if (err) *err = why;
            shutdown();
            return false;
        }
    }

    if (cfg_.warmup) {
        const i32 t = tok_.bos() >= 0 ? tok_.bos() : 0;
        const Timer w;
        if (!forward(&t, 1, 0, true)) {
            if (err) *err = "KV promotion refused during the engine warmup";
            shutdown();
            return false;
        }
        be_->sync();
        KRK_INFO("warmup pass: %.1f ms", w.ms());
    }
    return true;
}

void Engine::shutdown() {
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
    for (void **p : {&ws_x_, &ws_rexp_, &ws_fexp_, &ws_xn_, &ws_x2_, &ws_q_,
                     &ws_qpack_, &ws_k_,
                     &ws_v_, &ws_attn_, &ws_gate_, &ws_up_, &ws_had_, &ws_logits_,
                     &ws_router_,
                     &ws_ffn_, &ws_xg_, &ws_gateg_, &ws_upg_, &ws_plan_, &ws_alpha_,
                     &ws_scale_,
                     &ws_qkv_, &ws_z_, &ws_h_, &ws_ssm_, &ws_beta_, &ws_agate_,
                     &ws_m2in_,
                     &conv_state_, &rec_state_, &topk_scratch_, &topk_out_,
                     &mtp_kv_k_, &mtp_kv_v_, &mtp_hidden_, &mtp_h_, &mtp_e_,
                     &mtp_cat_, &mtp_gate_, &mtp_sgate_, &mtp_rep_,
                     &mtp_rec_snap_, &mtp_conv_snap_}) {
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
        KRK_INFO("weight stream: %llu layer staging(s), %.2f GiB from the file, "
                 "%.0f ms (%.2f GB/s) = drain %.0f + read %.0f + copy %.0f ms",
                 static_cast<unsigned long long>(model_.stage_calls()),
                 static_cast<f64>(model_.stage_bytes()) / 1073741824.0,
                 model_.stage_ms(),
                 static_cast<f64>(model_.stage_bytes()) /
                     (model_.stage_ms() > 0.0 ? model_.stage_ms() / 1000.0 : 1.0) /
                     1073741824.0,
                 model_.stage_sync_ms(), model_.stage_read_ms(),
                 model_.stage_xfer_ms());
    model_.unload();
    be_ = nullptr;
}

void Engine::dump_row(const char *what, i32 layer, const void *buf, i64 width,
                      i64 rows) {
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

void Engine::head_compute(i32 row) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const void *src = act_at(ws_x_, as, static_cast<i64>(row) * n_embd_);
    be_->rmsnorm(ws_xn_, src, model_.out_norm(), 1, n_embd_, mc.rms_eps,
                 static_cast<const i32 *>(ws_rexp_) + row);
    const QuantTensor &head = model_.out_head();
    const void *xn = ws_xn_;
    if (model_.had().head_folded) {
        had_xform(ws_had_, ws_xn_, 1, n_embd_, false, false);
        xn = ws_had_;
    }
    be_->gemm(ws_logits_, xn, head.data, head.type, n_vocab_, n_embd_, 1);
}

bool Engine::forward_core(const i32 *toks, i32 n, i32 pos0, LogitMode mode) {
    const ModelConfig &mc = model_.cfg();
    const auto &layers = model_.layers();
    const QuantTensor &embd = model_.tok_embd();
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

    // A latent token embedding stores ROTATED rows: un-rotate them here, once,
    // and the whole residual stream above this point is in the primal basis
    // that every norm, every non-folded matmul and every dump expects.
    if (model_.had().embd_inverse)
        had_xform(ws_x_, ws_x_, n, n_embd_, /*inverse=*/true, /*gdn_perm=*/false);

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
    d.layer = 0;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(mc.head_dim))) *
              mc.rope_attn_scale;

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
        d.pos_stride = lkv;
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

        be_->rmsnorm(ws_xn_, ws_x_, L.attn_norm, n, n_embd_, mc.rms_eps,
                     static_cast<const i32 *>(ws_rexp_));

        if (L.gdn) {
            // nemotron_h_moe Mamba-2 SSM layer
            if (L.nemotron_ssm) {
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
            if (!L.nemotron_ssm || L.moe || L.wgate.present()) {
                be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, n, n_embd_, mc.rms_eps,
                             static_cast<const i32 *>(ws_rexp_));
                dump_row("ffn.pre", l, ws_xn_, n_embd_, n);
                if (L.moe) {
                    moe_ffn(L, l, n);
                } else {
                    dense_ffn(L, l, n);
                }
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
        if (L.wq.present()) {
        const void *xn_att = ws_xn_;
        if (L.h_attn_in || L.h_attn_gate) {
            had_xform(ws_had_, ws_xn_, n, n_embd_, false, false);
            xn_att = ws_had_;
        }
        const void *xn_proj = L.h_attn_in ? xn_att : ws_xn_;
        void *q_out = packed_gate ? ws_qpack_ : ws_q_;
        if (L.wq.type == L.wk.type && L.wk.type == L.wv.type) {
            void *qkv[3] = {q_out, ws_k_, ws_v_};
            const void *wqkv[3] = {L.wq.data, L.wk.data, L.wv.data};
            const i64 nqkv[3] = {lq_proj, lkv, lkv};
            be_->gemm_group(qkv, xn_proj, wqkv, L.wq.type,
                            nqkv, n_embd_, 3, n);
        } else {
            be_->gemm(q_out, xn_proj, L.wq.data, L.wq.type, lq_proj, n_embd_, n);
            be_->gemm(ws_k_, xn_proj, L.wk.data, L.wk.type, lkv, n_embd_, n);
            be_->gemm(ws_v_, xn_proj, L.wv.data, L.wv.type, lkv, n_embd_, n);
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
        void *kbase = kvt_k_.base(l);
        void *vbase = kvt_v_.base(l);
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
        if (!(n == 1 && !packed_gate && d.window == 0 && !mc.attn_gate &&
              (fused_neox || !(mc.rope_neox && mc.qk_norm)) &&
              be_->attn_fused_chain(ws_attn_, ws_q_, kbase, vbase,
                                      ws_k_, ws_v_, d,
                                      model_.inv_freq_at(l).data(),
                                      mc.rope_scale, model_.rope_frac_at(l),
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
                             mc.head_dim, n, mc.rms_eps, mc.qk_norm_wide);
            // Post-norm, pre-rope: the row the rotation consumes. The
            // fused arm's copy of it lives in registers and never lands
            // in this buffer, so this row covers the separate chain only.
            dump_row("attn.qk", l, ws_q_, lq, n);
            be_->rope(ws_q_, ws_k_, lh, lkvh, mc.head_dim,
                      n, pos0, model_.inv_freq_at(l).data(), mc.rope_scale,
                      model_.rope_frac_at(l), mc.rope_neox);
            be_->kv_append(kbase, vbase, ws_k_, ws_v_, d);
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
            be_->gemm(ws_agate_, L.h_attn_gate ? xn_att : ws_xn_,
                      L.wattn_gate.data, L.wattn_gate.type, lh, n_embd_, n);
            be_->softplus_act(ws_agate_, static_cast<i64>(n) * lh);
            dump_row("attn.gate", l, ws_agate_, lh, n);
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

        // The attention result is the input of a folded projection too, and
        // it is a different buffer with a different width (q_dim), so it gets
        // its own transform -- after the gate above has finished reading
        // ws_had_.
        // The raw attention output, before its projection: the stage where a
        // key-space reduction order shows first, and the one the tiled kernel
        // used to differ at. Both chains write it, so one call here covers a
        // 1-row call (the fused chain) and a multi-row one (the separate chain).
        trace_rows("attn.out", l, ws_attn_, lq, n);
        const void *attn_in = ws_attn_;
        if (L.h_attn_out) {
            had_xform(ws_had_, ws_attn_, n, lq, false, false);
            attn_in = ws_had_;
        }
        be_->gemm(ws_x2_, attn_in, L.wo.data, L.wo.type, n_embd_, lq, n);
        // gpt-oss biases the attention output: one value per output channel,
        // applied to the projection's result before it joins the residual. A
        // broadcast over the rows of the GEMM output, which is exactly the
        // `add_bias_cols` shape (n rows of n_embd columns).
        if (L.wo_bias) be_->add_bias_cols(ws_x2_, L.wo_bias, n, n_embd_);
        dump_row("attn.proj", l, ws_x2_, n_embd_, n);
        // The attention residual and the FFN norm are one kernel: the sum is
        // what the norm's sum-of-squares pass reads, so a second launch
        // round-trips the activation row for nothing. The dump still sees the
        // post-add state, because the fused kernel writes x before it
        // normalizes.
        be_->add_rmsnorm(ws_xn_, ws_x_, ws_x2_, L.ffn_norm, n, n_embd_,
                         mc.rms_eps, static_cast<const i32 *>(ws_rexp_));
        } else {
            // Pure-MoE block: no attention to fold in, so the FFN input norm
            // is applied straight to the residual.
            be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, n, n_embd_, mc.rms_eps,
                         static_cast<const i32 *>(ws_rexp_));
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
        if (L.moe) {
            moe_ffn(L, l, n);
        } else if (L.wgate.present()) {
            dense_ffn(L, l, n);
        }
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
void Engine::gemm_residual(void *dst, const void *x, const void *w, DType wt,
                           i64 n_out, i64 n_in, i64 rows, void *scratch) {
    // `rexp` goes down both paths: `dst` is the residual stream, so the
    // contribution folded in has to be divided by the row's storage exponent,
    // and the fallback -- which is what prefill rows and the CPU take -- puts it
    // through add_residual, so it renormalizes the exponent as well.
    const i32 *rexp = static_cast<const i32 *>(ws_rexp_);
    if (dump_path_.empty() && rows > 0 &&
        be_->gemm_accumulate(dst, x, w, wt, n_out, n_in, rows, rexp))
        return;
    be_->gemm(scratch, x, w, wt, n_out, n_in, rows);
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
        be_->copy_act(mtp_h_, act_at(mtp_hidden_, as, p * n_embd_), n_embd_);
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
        void *gout[2] = {ws_gate_, ws_up_};
        const void *gw[2] = {m.ffn_gate.data, m.ffn_up.data};
        const i64 gno[2] = {ff, ff};
        const DType gwts[2] = {m.ffn_gate.type, m.ffn_up.type};
        be_->gemm_group(gout, ws_xn_, gw, m.ffn_gate.type, gno, n_embd_, 2, n,
                        true, &fused, false, gwts);
        if (!fused) be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * ff);
    } else {
        be_->gemm(ws_up_, ws_xn_, m.ffn_up.data, m.ffn_up.type, ff, n_embd_, n);
        be_->copy_act(ws_gate_, ws_up_, n * ff);
        be_->sigmoid_act(ws_gate_, n * ff);
        be_->mul_act(ws_gate_, ws_up_, n * ff);
    }
    be_->gemm(ws_x2_, ws_gate_, m.ffn_down.data, m.ffn_down.type, n_embd_, ff, n);
    be_->add_residual(acc, ws_x2_, n, n_embd_, exp);
}

void Engine::mtp_ffn(const MtpBlock &m, i32 n) {
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
    const bool routed = m.router.present() && ne > 0 && k > 0 && ff > 0;
    if (!routed) {
        mtp_dense_ffn(m, n, acc, exp);
    } else {

        // Routing runs on the host, exactly as a stack layer's does: the logits are
        // tiny and the selection is what builds the plan.
        be_->gemm(ws_router_, ws_xn_, m.router.data, m.router.type, ne, n_embd_, n);
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
            if (!re) continue;
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

    if (ff_sh > 0 && m.shexp_up.present() && m.shexp_down.present()) {
        if (m.shexp_gate.present())
            be_->gemm(ws_gate_, ws_xn_, m.shexp_gate.data, m.shexp_gate.type, ff_sh,
                      n_embd_, n);
        be_->gemm(ws_up_, ws_xn_, m.shexp_up.data, m.shexp_up.type, ff_sh, n_embd_, n);
        if (m.shexp_gate.present()) {
            be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * ff_sh);
        } else {
            be_->copy_act(ws_gate_, ws_up_, n * ff_sh);
            be_->sigmoid_act(ws_gate_, n * ff_sh);
            be_->mul_act(ws_gate_, ws_up_, n * ff_sh);
        }
        be_->gemm(ws_x2_, ws_gate_, m.shexp_down.data, m.shexp_down.type, n_embd_,
                  ff_sh, n);
        // Qwen3.5 scales the shared expert by a sigmoid of a per-token
        // projection -- `ffn_gate_inp_shexp`, which every block 40 in this
        // family carries. Skipping it scales the shared path by 1 where the
        // stack uses a learned per-token factor, which is a wrong answer that
        // still reads as fluent text.
        if (m.shexp_inp_gate.present()) {
            be_->gemm(mtp_sgate_, ws_xn_, m.shexp_inp_gate.data,
                      m.shexp_inp_gate.type, 1, n_embd_, n);
            be_->sigmoid_act(mtp_sgate_, n);
            be_->scale_rows(ws_x2_, mtp_sgate_, n, n_embd_);
        }
        be_->add_residual(acc, ws_x2_, n, n_embd_, exp);
    }

    if (acc != ws_x_)
        be_->add_residual(ws_x_, acc, n, n_embd_, static_cast<i32 *>(mtp_rep_),
                          static_cast<const i32 *>(ws_fexp_));
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
    be_->gemm(ws_x_, mtp_cat_, m.eh_proj.data, m.eh_proj.type, n_embd_,
              2 * n_embd_, 1);
    be_->fill0(mtp_rep_, sizeof(i32));

    be_->rmsnorm(ws_xn_, ws_x_, m.attn_norm, 1, n_embd_, mc.rms_eps);
    const bool packed = m.wq.n_out == 2 * q;
    if (packed && !ws_qpack_) return false;
    void *qp = packed ? ws_qpack_ : ws_q_;
    be_->gemm(qp, ws_xn_, m.wq.data, m.wq.type, m.wq.n_out, n_embd_, 1);
    be_->gemm(ws_k_, ws_xn_, m.wk.data, m.wk.type, kvw, n_embd_, 1);
    be_->gemm(ws_v_, ws_xn_, m.wv.data, m.wv.type, kvw, n_embd_, 1);
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
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd))) *
              mc.rope_attn_scale;
    // The block shares the stack's rotary table: it is the same width, the same
    // head count and the same position axis, and block 40's own weights are
    // trained against the positions the stack writes.
    be_->rope(ws_q_, ws_k_, nh, nkv, hd, 1, pos, model_.inv_freq_at(0).data(),
              mc.rope_scale, model_.rope_frac_at(0), mc.rope_neox);
    be_->kv_append(mtp_kv_k_, mtp_kv_v_, ws_k_, ws_v_, d);
    be_->attention(ws_attn_, ws_q_, mtp_kv_k_, mtp_kv_v_, d);
    if (packed) {
        be_->sigmoid_act(mtp_gate_, q);
        be_->mul_act(ws_attn_, mtp_gate_, q);
    }
    be_->gemm(ws_x2_, ws_attn_, m.wo.data, m.wo.type, n_embd_, q, 1);
    // post_attention_norm is this family's pre-FFN norm (there is no separate
    // ffn_norm in block 40's tensor set), so it is applied to the residual, not
    // fused as an add+norm before the attention.
    be_->add_residual(ws_x_, ws_x2_, 1, n_embd_, static_cast<i32 *>(mtp_rep_));
    be_->rmsnorm(ws_xn_, ws_x_, m.post_attn_norm, 1, n_embd_, mc.rms_eps);
    mtp_ffn(m, 1);

    if (want_logits) {
        be_->rmsnorm(ws_xn_, ws_x_, m.shared_head_norm, 1, n_embd_, mc.rms_eps);
        const QuantTensor &head = model_.out_head();
        be_->gemm(ws_logits_, ws_xn_, head.data, head.type, n_vocab_, n_embd_, 1);
    }
    return true;
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
                      act_at(mtp_hidden_, as, static_cast<i64>(p) * n_embd_),
                      n_embd_);
        if (!mtp_block_forward(m, p, p == r.pos)) {
            r.note = "the block's forward was refused (workspace geometry)";
            return r;
        }
    }
    mtp_fed_ = P;
    r.draft_id = read_argmax(&r.draft_logit);
    r.ok = r.draft_id >= 0 && r.target_id >= 0;
    r.ms = timer.ms();
    return r;
}

void Engine::dense_ffn(const LayerWeights &L, i32 l, i64 n) {
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
    bool silu_done = false;
    if (L.wgate.type == L.wup.type) {
        void *gu[2] = {ws_gate_, ws_up_};
        const void *wgu[2] = {L.wgate.data, L.wup.data};
        const i64 ngu[2] = {n_ff_, n_ff_};
        be_->gemm_group(gu, xn, wgu, L.wgate.type, ngu, n_embd_, 2, n, true,
                        &silu_done);
    } else {
        be_->gemm(ws_gate_, xn, L.wgate.data, L.wgate.type, n_ff_, n_embd_, n);
        be_->gemm(ws_up_, xn, L.wup.data, L.wup.type, n_ff_, n_embd_, n);
    }
    if (!silu_done) be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * n_ff_);
    dump_row("attn.ffng", l, ws_gate_, n_ff_, n);
    // The intermediate is a folded input as well (its width is n_ff, wider
    // than anything above), which is safe to put over ws_had_ because the
    // gate/up gemms have already consumed their copy.
    const void *gn = ws_gate_;
    if (L.h_ffn_down) {
        had_xform(ws_had_, ws_gate_, n, n_ff_, false, false);
        gn = ws_had_;
    }
    gemm_residual(ws_x_, gn, L.wdown.data, L.wdown.type, n_embd_, n_ff_, n,
                  ws_x2_);
    dump_row("attn.down", l, ws_x2_, n_embd_, n);
}

bool Engine::forward(const i32 *toks, i32 n, i32 pos0, bool want_logits) {
    if (!forward_core(toks, n, pos0, want_logits ? LogitsLast : LogitsNone))
        return false;
    kv_pos_ = std::max<i64>(kv_pos_, static_cast<i64>(pos0) + n);
    return true;
}

bool Engine::kv_rollback(i64 pos) {
    if (pos < 0 || pos > kv_pos_) return false;
    if (pos == kv_pos_) return true;
    if (model_.is_recurrent()) {
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
    const size_t as = be_->act_size();
    // Resolve every layer in both planes before mutating any retained bytes.
    // Otherwise K can be zeroed successfully and V can then refuse promotion,
    // leaving the cache half-rolled-back while kv_pos_ still advertises the
    // original history. The pointers remain valid through the fill pass: no
    // tier admission occurs between this preflight and its corresponding fill.
    std::vector<void *> kbase(static_cast<size_t>(model_.cfg().n_layer), nullptr);
    std::vector<void *> vbase(static_cast<size_t>(model_.cfg().n_layer), nullptr);
    for (i64 l = 0; l < model_.cfg().n_layer; l++) {
        if (!model_.layer_has_kv(l)) continue;
        kbase[static_cast<size_t>(l)] = kvt_k_.base(l);
        vbase[static_cast<size_t>(l)] = kvt_v_.base(l);
        if (!kbase[static_cast<size_t>(l)] || !vbase[static_cast<size_t>(l)])
            return false;
    }
    for (i64 l = 0; l < model_.cfg().n_layer; l++) {
        if (!model_.layer_has_kv(l)) continue;
        const size_t row = static_cast<size_t>(model_.kv_dim_at(l)) * as;
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
    const size_t live_host_avail = be_->host_available_bytes();
    const size_t host_reserve = krk::host_reserve_bytes(ram_installed);
    const size_t host_stage_room = live_host_avail > host_reserve
                                       ? live_host_avail - host_reserve : 0;
    const size_t host_already_budgeted = krk::SharedHostRamBudget::used_bytes();
    const size_t host_shared_room = host_stage_room > host_already_budgeted
                                        ? host_stage_room - host_already_budgeted : 0;
    if (cfg_.host_ram_mb > 0) {
        host_budget = static_cast<size_t>(cfg_.host_ram_mb) * 1024u * 1024u;
    } else if (ram_total > 0) {
        const size_t gib2 = static_cast<size_t>(1) << 30;
        const size_t headroom = ram_total / 5;          // 4.8 / 6.4 / 9.6 / 19.2
        const size_t pinned = ram_total / 16 < 6 * gib2 ? ram_total / 16 : 6 * gib2;
        const size_t reserve = ram_total / 24 > 3 * gib2 ? ram_total / 24 : 3 * gib2;
        const size_t spent = headroom + pinned + reserve;
        host_budget = ram_total > spent ? ram_total - spent : ram_total / 2;
        // Real OS pressure still wins over the table: the spec requires the
        // warm tier to shrink when the machine is actually short, and the
        // table is a plan for an idle box.
        if (live_host_avail > 0 && host_budget > live_host_avail / 2)
            host_budget = live_host_avail / 2;
    }
    if (host_budget > host_shared_room) host_budget = host_shared_room;
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
        size_t want = host_budget;
        if (host_shared_room > 0 && want > (host_shared_room / 2))
            want = host_shared_room / 2;
        const size_t total = model_.total_expert_bytes();
        if (total > 0 && total < want) want = total; // never cache past the corpus
        const size_t floor_bytes = model_.max_expert_bytes() * 4;
        warm = want >= floor_bytes ? want : 0;
    }
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
    // --expert-pin-mb owns the tier's pinned prefix when it is given; -1 leaves
    // KRK_HOST_PIN_MB's answer (off by default) alone.
    if (cfg_.expert_pin_mb >= 0)
        model_.experts().set_host_pin_budget(
            static_cast<size_t>(cfg_.expert_pin_mb) * 1024u * 1024u);

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
        const size_t host_reserve = krk::host_reserve_bytes(be_->host_total_bytes());
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
                                                host_reserve))
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
                     static_cast<f64>(host_reserve) / 1048576.0);
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
    char warm_note[80] = "";
    if (warm > 0)
        std::snprintf(warm_note, sizeof(warm_note),
                      " + WARM %.0f MiB pageable%s",
                      static_cast<f64>(warm) / (1024.0 * 1024.0),
                      prefetch && !expert_warmup_wanted() ? " (prefetched)" : "");
    KRK_INFO(             "expert cache: budget %.1f MiB%s, %d experts x top-%d, %d layers, "
             "policy LFU+aging (pin at %u hits, decay every %u); cold reads are "
             "read-through",
             static_cast<f64>(budget) / (1024.0 * 1024.0), warm_note, mc.n_expert,
             mc.n_expert_used, mc.n_layer, ExpertCache::kPinThreshold,
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
                     warm >= total_bytes
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
    if (!d->load(*be_, path, chunk_, kv_cap_, tc.n_embd, tc.n_layer, err)) {
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
    if (model_.is_recurrent()) {
        // Same reason load_draft refuses one: a delta-rule state has no
        // inverse, so a rejected suffix cannot be rewound.
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
    if (h->cfg().n_in != 1) {
        delete h;
        if (err)
            *err = "draft head: it fuses " + std::to_string(h->cfg().n_in) +
                   " target hidden slot(s); this engine captures the "
                   "pre-final-norm row, i.e. one";
        return false;
    }
    if (h->cfg().n_layer != 1) {
        delete h;
        if (err)
            *err = "draft head: " + std::to_string(h->cfg().n_layer) +
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
            res->error = kv_resource_error();
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
                res->error = kv_resource_error();
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
            res->error = kv_resource_error();
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
                res->error = kv_resource_error();
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
    if (model_.is_recurrent()) {
        // Speculation verifies a block and then rewinds to the accepted
        // prefix; a delta-rule state has no inverse, so the rewind would leave
        // the state describing rejected tokens. Refuse rather than decode
        // something plausible and wrong.
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
    const size_t as = be_->act_size();
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
    gemm_residual(ws_x_, h_in, L.ssm_out.data, L.ssm_out.type, n_embd_, inner,
                  n, ws_x2_);
    dump_row("m2.done", l, ws_x_, n_embd_, n);
}

void Engine::gdn_forward(const LayerWeights &L, i32 l, i32 n) {
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
    const bool gdn_onelaunch = gdn_group_ && !L.h_attn_in && !L.h_attn_gate;
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
        be_->gemm(ws_qkv_, L.h_attn_in ? xn : ws_xn_, L.wqkv.data, L.wqkv.type,
                  cdim, n_embd_, n);
        be_->gemm(ws_z_, L.h_attn_gate ? xn : ws_xn_, L.wqkv_gate.data,
                  L.wqkv_gate.type, vdim, n_embd_, n);
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
        be_->gemm(ws_ssm_, ws_xn_, L.ssm_alpha.data, L.ssm_alpha.type,
                  mc.ssm_dt_rank, n_embd_, n);
    }
    be_->add_bias_cols(ws_ssm_, L.ssm_dt, n, mc.ssm_dt_rank);
    be_->softplus_act(ws_ssm_, static_cast<i64>(n) * mc.ssm_dt_rank);
    be_->scale_cols(ws_ssm_, L.ssm_a, n, mc.ssm_dt_rank);

    if (!gdn_onelaunch) {
        be_->gemm(ws_beta_, ws_xn_, L.ssm_beta.data, L.ssm_beta.type,
                  mc.ssm_dt_rank, n_embd_, n);
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
    //    verbatim, not zero-centred) then the silu(z) gate.
    be_->rmsnorm(ws_h_, ws_h_, L.ssm_norm, static_cast<i64>(n) * mc.ssm_dt_rank,
                 mc.ssm_d_state, mc.rms_eps);
    be_->silu_mul(ws_h_, ws_z_, ws_h_, static_cast<i64>(n) * vdim);

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
                  n, ws_x2_);
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
void Engine::write_expert_index(const std::string &path) const {
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
    out.mass.assign(cells, 0.0);
    out.hits.assign(cells, 0);
    out.tokens = root.get_int("positions_scanned", 0);
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

void Engine::merge_expert_index(const std::string &path, double retain) const {
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
    if (ec.warm_capacity_bytes() == 0) return false; // no second tier to fill
    bool on = cfg_.expert_warmup > 0;
    if (cfg_.expert_warmup < 0) {
        // Default: on exactly when the ORDER of the two tiers can change the
        // answer, which is when the routed set does not fit the device. When it
        // fits, every expert is resident anyway and staging is pure wall time.
        on = model_.total_expert_bytes() > ec.budget_bytes();
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
    if (one == 0 || ec.warm_capacity_bytes() == 0) return;

    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    auto elapsed_ms = [&t0]() {
        return std::chrono::duration<f64, std::milli>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    };
    const i64 ceiling = cfg_.expert_warmup_ms;
    auto out_of_time = [&]() {
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
        if (left == 0) break; // tier full
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
    if (be_->caps().vram_free > 0) {
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
                if (!ec.in_host(l, e)) continue; // staging did not reach it
                const size_t had = ec.resident_slots();
                if (ec.acquire(L.experts, l, e) && ec.resident_slots() > had) {
                    promoted++;
                    promoted_bytes += L.experts.expert_bytes();
                }
            }
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
    KRK_INFO("expert warmup: %s order, %d rounds, %zu experts/layer deep; WARM "
             "%.0f MiB of %.0f MiB in %zu slots (%.0f MiB read), VRAM +%zu "
             "experts (%.0f MiB, %zu slots); %.0f ms%s",
             ranked ? "ranked" : "even", rounds, depth,
             static_cast<f64>(ec.warm_used_bytes()) / 1048576.0,
             static_cast<f64>(warm_cap) / 1048576.0, ec.warm_slots(),
             static_cast<f64>(staged) / 1048576.0, promoted,
             static_cast<f64>(promoted_bytes) / 1048576.0, ec.resident_slots(),
             ms, why.c_str());
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
        return e && e[0] != '0' && e[0] != 0;
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

void Engine::moe_ffn(const LayerWeights &L, i32 layer, i32 n) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const i32 ne = mc.n_expert;
    const i32 k = std::min<i32>(mc.n_expert_used > 0 ? mc.n_expert_used : ne, ne);
    const i64 ff_exp = mc.n_ff_exp;
    const i64 ff_sh = mc.n_ff_shexp;

    be_->fill0(ws_ffn_, static_cast<size_t>(n) * static_cast<size_t>(n_embd_) * as);
    // ... and the routed sum's storage exponents, zeroed once per layer, before
    // the first expert visit. `ws_fexp_` is where scatter_axpy_rows records the
    // power of two a row was stored under, and both folds below read it as that
    // row's exponent -- the same contract the residual stream's `ws_rexp_` has.
    be_->fill0(ws_fexp_, static_cast<size_t>(n) * sizeof(i32));

    if (L.router.present() && k > 0 && ne > 0) {
        // Routing runs on the host: the logits are tiny ([n, n_expert]) and the
        // selection is what builds the plan.
        be_->gemm(ws_router_, ws_xn_, L.router.data, L.router.type, ne, n_embd_, n);
        // The blocking device-to-host copy below already orders itself behind
        // every kernel queued on the default stream, so the explicit
        // hipDeviceSynchronize this used to issue first was redundant. Forty of
        // them per decode token is not free on WDDM: each one is a driver-wide
        // drain that hands the thread back to the scheduler and takes it back.
        // KRK_ROUTER_SYNC=1 puts the explicit drain back, which is the arm the
        // cost of removing it is measured against.
        if (router_sync()) be_->sync();
        be_->download_f32(router_host_.data(), ws_router_, static_cast<i64>(n) * ne);

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

            // Fold this token's routing distribution into the scan. Mass is
            // recorded for every expert, not just the k selected, because the
            // question the index answers is "what coverage does the top N buy",
            // and that needs the whole curve.
            if (expert_scan_ || expert_track_) {
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

            i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
            f32 *wt = moe_wt_.data() + static_cast<size_t>(t) * k;
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
            for (i32 s = 0; s < k; s++) {
                // The top available score first, so "near-equal" has something
                // to be near. With the preference off this is the only pass and
                // the arithmetic below reduces to the loop it replaced.
                f32 top = -1.0f;
                for (i32 e = 0; e < ne; e++) {
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
                if ((expert_scan_ || expert_track_) && best >= 0)
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
        if (!expert_stub_ && k > 0) {
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
            if (prefetch_ids_.size() > 1)
                model_.experts().prefetch_layer(
                    L.experts, layer, prefetch_ids_.data(),
                    static_cast<int>(prefetch_ids_.size()));
            // Look-ahead for a BATCHED pass. The router for all n rows is
            // already on the host at this point, so a verify pass knows the
            // union of experts its rows will need one row early: staging it
            // means rows 1..n-1 hit VRAM instead of promoting serially, and the
            // promotions overlap the first row's compute. A single row has
            // nothing ahead of it (stage_next covers ordering inside its own
            // routing), so this only runs when n > 1.
            if (n == 1 && lookahead_stage())
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
        const bool plan_grouped = moe_plan_grouped() && !expert_stub_;
        if (plan_grouped) {
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
                be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(),
                                static_cast<i64>(plan_dev_.size()));
                be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                            plan_dev_.size() * sizeof(f32));
            }
        }

        for (i32 e = 0; !expert_stub_ && e < ne; e++) {
            // The host arm owns this expert on this layer. Skipping it here is
            // what leaves its weights absent from both tiers.
            if (hybrid && hy_take_[static_cast<size_t>(e)]) continue;
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

            // Pages the expert in if it is not resident; identical results
            // regardless of how small the cache is.
            const ResidentExpert *re = model_.experts().acquire(L.experts, layer, e);
            if (!re) continue;
            if (sync_expert()) be_->sync();
            if (hybrid) hy_gpu_experts_++;

            // The per-expert global scale each NVFP4 matrix needs.
            const f32 gs_up = L.experts.scale_of(L.experts.up_scale, e);
            const f32 gs_dn = L.experts.scale_of(L.experts.down_scale, e);
            const f32 gs_gt = L.experts.scale_of(L.experts.gate_scale, e);

            const i64 m = static_cast<i64>(group_rows_.size());
            const i32 *plan_i = nullptr;
            const f32 *plan_f = nullptr;
            if (plan_grouped) {
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

            // x̃ = xn[group_rows] — the permuted activation block.
            be_->gather_rows(ws_xg_, ws_xn_, plan_i, m, n_embd_);
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
                be_->scale_act(ws_xg_, gs_up, m, n_embd_, n_embd_);
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
                be_->gemm_group(gout, ws_xg_, gw, re->gate.type, gno, n_embd_, 2,
                                m, moe_silu_fused(), &pair_fused, false, gwts);
            } else {
                be_->gemm(ws_upg_, ws_xg_, re->up.data, re->up.type, ff_exp,
                          n_embd_, m);
                // Per-expert bias: this group is ONE expert's rows, so the
                // table's row `e` is the vector every row of the block gets.
                if (L.exp_up_bias)
                    be_->add_bias_cols(ws_upg_, L.exp_up_bias + static_cast<i64>(e) * ff_exp,
                                       m, ff_exp);
                if (re->gate.present()) {
                    // ws_xg_ is carrying gs_up; leave it carrying gs_gt for the
                    // gate. It is read by nothing else.
                    if (gs_gt != gs_up)
                        be_->scale_act(ws_xg_, gs_gt / gs_up, m, n_embd_,
                                       n_embd_);
                    be_->gemm(ws_gateg_, ws_xg_, re->gate.data, re->gate.type,
                              ff_exp, n_embd_, m);
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
                scan(ws_xg_, m * n_embd_, in_nf, in_pk);
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
                    if (mc.ffn_swiglu_oai) {
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
            be_->gemm(ws_x2_, ws_gateg_, re->down.data, re->down.type, n_embd_,
                      ff_exp, m);
            // The down leg's bias is over the OUTPUT width (n_embd), not ff.
            if (L.exp_down_bias)
                be_->add_bias_cols(ws_x2_,
                                   L.exp_down_bias + static_cast<i64>(e) * n_embd_,
                                   m, n_embd_);
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
            be_->scatter_axpy_rows(ws_ffn_, ws_x2_, plan_i, plan_f, m, n_embd_,
                                   scatter_scale,
                                   static_cast<i32 *>(ws_fexp_));
            // Exact next expert in the existing ascending execution order.
            // Submit its copy only after this expert's weight readers are queued.
            const char *stage = std::getenv("KRK_ACTIVE_STAGE");
            if (n == 1 && be_->kind() == BackendKind::HIP && stage && stage[0] == '1') {
                i32 next = ne;
                for (i32 id : prefetch_ids_) if (id > e && id < next) next = id;
                if (next < ne) model_.experts().stage_next(L.experts, layer, next, e);
            }
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
            be_->download_f32(hy_xn_.data(), ws_xn_, static_cast<i64>(cells));

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
                        const size_t k = base + static_cast<size_t>(j);
                        const f64 v = static_cast<f64>(hy_ffn_[k]) * sc +
                                      static_cast<f64>(hy_out_[k]);
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
                        const size_t k = base + static_cast<size_t>(j);
                        const f64 v = static_cast<f64>(hy_ffn_[k]) * sc +
                                      static_cast<f64>(hy_out_[k]);
                        hy_f16_[k] = fp32_to_fp16(static_cast<f32>(v * inv));
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
        // The scale is folded into the gemm's input. ws_xg_ is the expert
        // group's staging buffer and is free here; ws_xn_ is not, because the
        // model's own router has already read it and a scaled copy of it must
        // not be what a later stage sees.
        const void *sh_in = ws_xn_;
        if (L.shexp_up_scale != 1.0f || L.shexp_gate_scale != 1.0f) {
            be_->copy_act(ws_xg_, ws_xn_, n * n_embd_);
            be_->scale_act(ws_xg_, L.shexp_up_scale, n, n_embd_, n_embd_);
            sh_in = ws_xg_;
        }
        if (L.shexp_gate.present()) {
            if (L.shexp_gate_scale != L.shexp_up_scale)
                be_->scale_act(ws_xg_, L.shexp_gate_scale / L.shexp_up_scale, n,
                               n_embd_, n_embd_);
            be_->gemm(ws_gate_, sh_in, L.shexp_gate.data, L.shexp_gate.type, ff_sh,
                      n_embd_, n);
        }
        be_->gemm(ws_up_, sh_in, L.shexp_up.data, L.shexp_up.type, ff_sh, n_embd_, n);
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
        if (L.shexp_down_scale != 1.0f)
            be_->scale_act(ws_gate_, L.shexp_down_scale, n, ff_sh, ff_sh);
        be_->gemm(ws_x2_, ws_gate_, L.shexp_down.data, L.shexp_down.type, n_embd_, ff_sh, n);
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
            be_->gemm(ws_ssm_, ws_xn_, L.shexp_inp_gate.data, L.shexp_inp_gate.type,
                      1, n_embd_, n);
            be_->sigmoid_act(ws_ssm_, n);
            be_->scale_rows(ws_x2_, ws_ssm_, n, n_embd_);
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
    be_->add_residual(ws_x_, ws_ffn_, n, n_embd_,
                      static_cast<i32 *>(ws_rexp_),
                      static_cast<const i32 *>(ws_fexp_));
    dump_row("moe.post", layer, ws_x_, n_embd_, n);
}

void Engine::fetch_logits(bool device_only) {
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
    if (logits_sync()) be_->sync();
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
    if (!be_ || be_->kind() == BackendKind::CPU || topk_scratch_ == nullptr)
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
            res->error = kv_resource_error();
            res->finish = FinishError;
            return false;
        }
    }
    fetch_logits(false);
    *prefill_ms = prefill_timer.ms();
    expert_phase(ExpertCache::PhaseDecode);

    i64 pos = static_cast<i64>(ids.size());
    mtp_fed_ = 0;
    if (!mtp_prime(pos)) {
        res->error = "the nextn block could not be primed over the prompt";
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
                     "experts %.0f ms, %llu of %llu promoted",
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
                     static_cast<unsigned long long>(mtp_acq_n_));
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

    while (res->generated < p.max_tokens && pos < kv_cap_) {
        KRK_TIME_HOST("spec.round");
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, res->generated, pos, logits_host_, n_vocab_);
        const i32 first = argmax_of(logits_host_, n_vocab_);
        const i64 from = pos;

        // ---- propose: the block's own chain -------------------------------
        const u64 t_prop = now_us();
        i32 d = 0;
        for (; d < k; d++) {
            const i64 fpos = pos - 1 + d;
            if (fpos < 0 || fpos >= kv_cap_) break;
            const i32 tok_in =
                d == 0 ? last_ctx_token() : prop[static_cast<size_t>(d - 1)];
            if (tok_in == tok_.eos()) break;
            mtp_embed(tok_in);
            if (d == 0) {
                if (fpos >= mtp_captured_) break;
                be_->copy_act(mtp_h_,
                              act_at(mtp_hidden_, be_->act_size(), fpos * n_embd_),
                              n_embd_);
            } else {
                // The block's residual from the previous proposal. Copied
                // before this forward overwrites ws_x_.
                be_->copy_act(mtp_h_, ws_x_, n_embd_);
            }
            if (!mtp_block_forward(m, static_cast<i32>(fpos), true)) break;
            i32 id = -1;
            if (!topk_argmax(&id)) {
                be_->sync();
                be_->download_f32(logits_host_, ws_logits_, n_vocab_);
                id = argmax_of(logits_host_, n_vocab_);
            }
            if (id < 0) break;
            prop[static_cast<size_t>(d)] = id;
            draft_proposed_++;
        }
        mtp_propose_ms_ += static_cast<f64>(now_us() - t_prop) / 1000.0;
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
            if (!forward(&first, 1, static_cast<i32>(pos), true)) {
                res->error = kv_resource_error();
                finish_run(FinishError);
                return false;
            }
            pos++;
            fetch_logits(false);
            if (!realign_block(from)) {
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
            res->error = kv_resource_error();
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
                if (!forward(&b, 1, static_cast<i32>(pos + emitted - 1), true)) {
                    res->error = kv_resource_error();
                    finish_run(FinishError);
                    return false;
                }
                fetch_logits(false);
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
                        res->error = kv_resource_error();
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
        if (!realign_block(from)) {
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
            res->error = kv_resource_error();
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
                res->error = kv_resource_error();
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
            res->error = kv_resource_error();
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
                    res->error = kv_resource_error();
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
            res->error = kv_resource_error();
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
                res->error = kv_resource_error();
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
                res->error = kv_resource_error();
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
            res->error = kv_resource_error();
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
                    res->error = kv_resource_error();
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
                    res->error = kv_resource_error();
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

    // ---- tokenize ---------------------------------------------------------
    std::vector<i32> ids;
    if (tok_.encode(p.prompt, ids, true) < 0) return false;
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
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        KRK_TIME_HOST("prefill.chunk");
        if (!forward(ids.data() + i, n, static_cast<i32>(i), last)) {
            res->error = kv_resource_error();
            res->finish = FinishError;
            return false;
        }
    }
    fetch_logits(device_topk);
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
        std::fprintf(stderr,
                     "[mtp   ] pos %d: draft %d \"%s\" (%.4f) | target %d "
                     "\"%s\" (%.4f) | %s | %.1f ms\n",
                     mp.pos, mp.draft_id, dtext.c_str(),
                     static_cast<f64>(mp.draft_logit), mp.target_id,
                     ttext.c_str(), static_cast<f64>(mp.target_logit),
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
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, gen, pos, logits_host_, n_vocab_);
        // The fast path hands back the argmax the device already computed.
        // Greedy skips the repetition penalty by design (Sampler::sample does
        // the same), so this is the identical token, not an approximation.
        const i32 token = topk_valid_ ? topk_id_
                                       : sampler.sample(logits_host_, n_vocab_);
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
                res->error = kv_resource_error();
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

    if (mtp_probe_report_ && mtp_steps > 0)
        KRK_INFO("mtp probe: %d step(s), %d accepted (%.1f%%) -- a greedy round "
                 "would have taken %d of %d drafted tokens",
                 mtp_steps, mtp_hits,
                 100.0 * static_cast<f64>(mtp_hits) / static_cast<f64>(mtp_steps),
                 mtp_hits, mtp_steps);
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
