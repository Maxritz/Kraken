// engine.cpp — prefill (batched) + decode (single token) with a paged-free
// contiguous KV cache, streaming output and stop-string handling.
#include "krk/engine.hpp"
#include "krk/arch.hpp"
#include "krk/dflash.hpp"
#include "par_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

bool Engine::init(Backend *be, const EngineConfig &cfg, std::string *err) {
    be_ = be;
    cfg_ = cfg;
    // KRK_DUMP=<path>: append every stage of the forward pass (the hidden state
    // after embed and after each layer, for the row that decides the next
    // token) as text, so the same prompt can be run on two backends and the
    // first stage that disagrees found by diffing the files. A debug hook on
    // the hot path's *shape*, not its math: nothing here is compiled out, it
    // just does nothing unless the variable is set.
    if (const char *dump = std::getenv("KRK_DUMP")) dump_path_ = dump;
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
    if (!model_.load(*be_, cfg.model_path, err)) {
        model_.unload();
        be_ = nullptr;
        return false;
    }
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
    ws_x_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    ws_xn_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    ws_x2_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    // Qwen3.5 packs a per-head output gate behind the query, so its query
    // projection is twice the query width and the split has to gather out of
    // it. The split reads a wider row than it writes, which means an in-place
    // call has head `h` writing over rows that other heads have not read yet:
    // block order decides the result. Keep the projection in its own buffer.
    ws_q_ = alloc(static_cast<size_t>(C * q_dim_) * as);
    if (q_proj_ != q_dim_)
        ws_qpack_ = alloc(static_cast<size_t>(C * q_proj_) * as);
    ws_k_ = alloc(static_cast<size_t>(C * kv_dim_) * as);
    ws_v_ = alloc(static_cast<size_t>(C * kv_dim_) * as);
    ws_attn_ = alloc(static_cast<size_t>(C * q_dim_) * as);
    // Two arches carry an attention output gate and they hold it differently:
    // Qwen3.5 packs it behind the query projection (only for its recurrent
    // stack), laguna projects it separately. Both need the buffer, and this is
    // the only place it is allocated.
    if (model_.is_recurrent() || mc.attn_gate)
        ws_agate_ = alloc(static_cast<size_t>(C * q_dim_) * as);
    // The widest FFN a layer can ask for: dense, routed expert, or shared expert.
    const i64 ff_ws = model_.n_ff_ws();
    ws_gate_ = alloc(static_cast<size_t>(C * ff_ws) * as);
    ws_up_ = alloc(static_cast<size_t>(C * ff_ws) * as);
    ws_logits_ = alloc(static_cast<size_t>(n_vocab_) * as);
    if (model_.is_recurrent()) {
        rec_layers_ = model_.recurrent_layers();
        ws_qkv_ = alloc(static_cast<size_t>(C * conv_dim_) * as);
        ws_z_ = alloc(static_cast<size_t>(C * value_dim_) * as);
        ws_h_ = alloc(static_cast<size_t>(C * value_dim_) * as);
        // One scalar pair per value head: tiny next to the projections.
        ws_ssm_ = alloc(static_cast<size_t>(C * mc.ssm_dt_rank) * as);
        ws_beta_ = alloc(static_cast<size_t>(C * mc.ssm_dt_rank) * as);
        // f32, not activation-typed: the state outlives the forward and is
        // accumulated at higher precision than the activations it is fed.
        conv_state_span_ = static_cast<i64>(mc.ssm_d_conv - 1) * conv_dim_;
        rec_state_span_ = static_cast<i64>(mc.ssm_dt_rank) * mc.ssm_d_state *
                          mc.ssm_d_state;
        conv_state_ = alloc(static_cast<size_t>(rec_layers_) *
                            static_cast<size_t>(conv_state_span_) * sizeof(f32));
        rec_state_ = alloc(static_cast<size_t>(rec_layers_) *
                           static_cast<size_t>(rec_state_span_) * sizeof(f32));
    }
    logits_host_ = static_cast<f32 *>(host_alloc(static_cast<size_t>(n_vocab_) * 4));
    tok_scratch_.resize(static_cast<size_t>(C));
    // Device top-k. A fixed 36 kB of candidate buffer, allocated for every
    // backend so that "can we use the fast path" is a runtime check and not a
    // different code path with a null to forget.
    topk_scratch_ = alloc(4096 + static_cast<size_t>(kTopKCap) * 8);
    topk_out_ = alloc(16);

    if (mc.is_moe && mc.n_expert > 0) {
        const i32 k = std::max<i32>(1, mc.n_expert_used);
        ws_router_ = alloc(static_cast<size_t>(C * mc.n_expert) * as);
        ws_ffn_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        // Batched-expert staging: the plan can never exceed chunk rows because
        // each (token, expert) pair consumes exactly one slot.
        ws_xg_ = alloc(static_cast<size_t>(C * n_embd_) * as);
        ws_gateg_ = alloc(static_cast<size_t>(C * ff_ws) * as);
        ws_upg_ = alloc(static_cast<size_t>(C * ff_ws) * as);
        ws_plan_ = alloc(static_cast<size_t>(C) * sizeof(i32));
        ws_alpha_ = alloc(static_cast<size_t>(C) * sizeof(f32));
        router_host_.resize(static_cast<size_t>(C) * static_cast<size_t>(mc.n_expert));
        moe_prob_.resize(static_cast<size_t>(mc.n_expert));
        moe_sel_.resize(static_cast<size_t>(C) * static_cast<size_t>(k));
        // Only sized when a scan is actually requested: [n_layer * n_expert]
        // doubles and u64 counters, which is 3584 * 12 B = 43 kB for a
        // 28x128 model and nothing at all otherwise.
        if (expert_scan_) {
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
    configure_expert_cache();
    // KV residency -- allocated LAST, after the weights and every workspace.
    //
    // That ordering is load-bearing. Reading device_free_bytes() before the
    // workspaces exist reports nearly all free VRAM, so a budget taken here
    // over-commits: the slots are allocated, and then ws_x_ and the rest are
    // allocated on top of them. Subtracting "what the workspaces will take"
    // is not the same thing as measuring it, and guessing it is how an
    // over-commit happens. So: take the measurement after the fact.
    {
        const size_t layer_bytes = static_cast<size_t>(kv_cap_) *
                                   static_cast<size_t>(kv_dim_) * as;
        const size_t kv_total = static_cast<size_t>(mc.n_layer) * layer_bytes * 2u;

        // Leave a reserve so a later allocation cannot wedge the driver. The
        // device is shared: the desktop compositor and any other app are
        // already drawing from what is left.
        const size_t free_vram = be_->device_free_bytes();
        const size_t reserve = std::max<size_t>(256u * 1048576u, free_vram / 8u);

        size_t hot;
        if (cfg_.kv_hot_mb > 0) {
            hot = static_cast<size_t>(cfg_.kv_hot_mb) * 1048576u;
            // An explicit budget can still be larger than the card. Clamp it
            // to what is actually there, or init() over-commits on request.
            const size_t cap = free_vram > reserve ? free_vram - reserve : 0;
            if (hot > cap) hot = cap;
        } else {
            hot = free_vram > reserve ? free_vram - reserve : 0;
        }

        // WARM is host RAM, so it must not be sized from device memory. With
        // no host-memory query in the Backend interface, the honest auto is
        // "enough to hold whatever does not fit HOT" -- bounded by the KV
        // itself, so it cannot outgrow the machine. Explicit --kv-warm-mb wins.
        const size_t overflow = kv_total > hot ? kv_total - hot : 0;
        size_t warm = overflow;
        if (cfg_.kv_warm_mb >= 0)
            warm = static_cast<size_t>(cfg_.kv_warm_mb) * 1048576u;
        else if (warm > kv_total)
            warm = kv_total;

        // Two planes share both budgets.
        if (!kvt_k_.init(be_, mc.n_layer, layer_bytes, hot / 2u, warm / 2u,
                         cfg_.kv_cold_dir, err) ||
            !kvt_v_.init(be_, mc.n_layer, layer_bytes, hot / 2u, warm / 2u,
                         cfg_.kv_cold_dir, err)) {
            model_.unload();
            be_ = nullptr;
            return false;
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
    const f64 total_vram_mb =
        static_cast<f64>(mc.n_layer) * static_cast<f64>(kv_cap_) *
        static_cast<f64>(kv_dim_) * static_cast<f64>(as) * 2.0 / (1024.0 * 1024.0);
    {
        KvTierStats ks;
        kvt_k_.stats(&ks);
        if (ks.tiered)
            KRK_INFO("kv tiered: %lld layers, %lld HOT slots of %.0f MiB, "
                     "layer=%.1f MiB, WARM budget=%lld layers",
                     (long long)ks.layers, (long long)ks.slots,
                     static_cast<double>(ks.hot_bytes) / 1048576.0,
                     static_cast<double>(ks.layer_bytes) / 1048576.0,
                     (long long)ks.slots);
        else
            KRK_INFO("workspaces ready: chunk=%d ctx=%lld KV=%.0f MiB (load %.0f ms)",
                     chunk_, static_cast<long long>(kv_cap_), total_vram_mb, load_ms);
    }

    if (cfg_.warmup) {
        const i32 t = tok_.bos() >= 0 ? tok_.bos() : 0;
        const Timer w;
        forward(&t, 1, 0, true);
        be_->sync();
        KRK_INFO("warmup pass: %.1f ms", w.ms());
    }
    return true;
}

void Engine::shutdown() {
    // The draft shares the backend; it must release its allocations before
    // this engine tears the backend's other buffers down.
    unload_dflash();
    unload_draft();
    if (!be_) {
        model_.unload();
        return;
    }
    be_->sync();
    for (void **p : {&ws_x_, &ws_xn_, &ws_x2_, &ws_q_,
                     &ws_qpack_, &ws_k_,
                     &ws_v_, &ws_attn_, &ws_gate_, &ws_up_, &ws_logits_, &ws_router_,
                     &ws_ffn_, &ws_xg_, &ws_gateg_, &ws_upg_, &ws_plan_, &ws_alpha_,
                     &ws_qkv_, &ws_z_, &ws_h_, &ws_ssm_, &ws_beta_, &ws_agate_,
                     &conv_state_, &rec_state_, &topk_scratch_, &topk_out_}) {
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
    host_free(logits_host_);
    logits_host_ = nullptr;
    model_.unload();
    be_ = nullptr;
}

void Engine::dump_row(const char *what, i32 layer, const void *buf, i64 width,
                      i64 rows) {
    if (dump_path_.empty() || rows <= 0) return;
    be_->sync();
    std::vector<f32> row(static_cast<size_t>(width));
    be_->download_f32(row.data(),
                      act_at(const_cast<void *>(buf), be_->act_size(),
                             (rows - 1) * width),
                      width);
    std::FILE *f = std::fopen(dump_path_.c_str(), "a");
    if (!f) return;
    std::fprintf(f, "%s L%02d n=%lld", what, layer, static_cast<long long>(rows));
    for (i64 i = 0; i < width; i++)
        std::fprintf(f, " %.7g", row[static_cast<size_t>(i)]);
    std::fputc('\n', f);
    std::fclose(f);
}

void Engine::head_compute(i32 row) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const void *src = act_at(ws_x_, as, static_cast<i64>(row) * n_embd_);
    be_->rmsnorm(ws_xn_, src, model_.out_norm(), 1, n_embd_, mc.rms_eps);
    const QuantTensor &head = model_.out_head();
    be_->gemm(ws_logits_, ws_xn_, head.data, head.type, n_vocab_, n_embd_, 1);
}

void Engine::forward_core(const i32 *toks, i32 n, i32 pos0, LogitMode mode) {
    const ModelConfig &mc = model_.cfg();
    const auto &layers = model_.layers();
    const QuantTensor &embd = model_.tok_embd();
    // Qwen3.5 packs its output gate behind the query (a per-layer shape only
    // when the arch says so); laguna projects its own. Decided once from the
    // arch, the same way Engine::init decided the packed workspace.
    const bool packed_gate = q_proj_ != q_dim_;

    be_->embed(ws_x_, embd.data, embd.type, mc.n_vocab, mc.n_embd, toks, n);

    AttnDesc d;
    d.n_head = mc.n_head;
    d.n_kv = mc.n_head_kv;
    d.hd = mc.head_dim;
    d.n_tok = n;
    d.pos0 = pos0;
    d.layer_stride = kv_cap_ * kv_dim_;
    d.pos_stride = kv_dim_;
    d.layer = 0;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(mc.head_dim))) *
              mc.rope_attn_scale;

    // One dump hook per stage, taking the row this call will predict from (the
    // last one): a CPU run and a HIP run of the same prompt then differ at
    // exactly one line per stage, and the first differing line names the layer.
    auto dump_stage = [&](const char *what, i32 layer) {
        dump_row(what, layer, ws_x_, n_embd_, n);
    };
    dump_stage("embed", -1);

    for (i32 l = 0; l < mc.n_layer; l++) {
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
        // A sliding-window layer sees only its last `swa_window` keys; the full
        // layers of the same stack see the whole prefix.
        d.window = (mc.swa_window > 0 && model_.is_swa(l)) ? mc.swa_window : 0;
        dump_stage("enter", l);

        be_->rmsnorm(ws_xn_, ws_x_, L.attn_norm, n, n_embd_, mc.rms_eps);

        if (L.gdn) {
            // Recurrent layer: a short conv plus the delta rule replaces
            // attention entirely, and there is nothing to append to the KV
            // cache. The residual lands in ws_x_ before the FFN below.
            gdn_forward(L, l, n);
            be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, n, n_embd_, mc.rms_eps);
            if (L.moe) {
                moe_ffn(L, l, n);
            } else {
                dense_ffn(L, l, n);
            }
            dump_stage("gdn", l);
            continue;
        }

        // q/k/v all read the attention-norm output and have no
        // dependency between them, so they ride one fused launch
        // on the decode row (backend falls back per-matrix for
        // prefill rows or a dtype mix).
        void *q_out = packed_gate ? ws_qpack_ : ws_q_;
        if (L.wq.type == L.wk.type && L.wk.type == L.wv.type) {
            void *qkv[3] = {q_out, ws_k_, ws_v_};
            const void *wqkv[3] = {L.wq.data, L.wk.data, L.wv.data};
            const i64 nqkv[3] = {lq_proj, kv_dim_, kv_dim_};
            be_->gemm_group(qkv, ws_xn_, wqkv, L.wq.type,
                            nqkv, n_embd_, 3, n);
        } else {
            be_->gemm(q_out, ws_xn_, L.wq.data, L.wq.type, lq_proj, n_embd_, n);
            be_->gemm(ws_k_, ws_xn_, L.wk.data, L.wk.type, kv_dim_, n_embd_, n);
            be_->gemm(ws_v_, ws_xn_, L.wv.data, L.wv.type, kv_dim_, n_embd_, n);
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
        if (L.k_bias) be_->add_bias_rows(ws_k_, L.k_bias, kv_dim_, n);
        if (L.v_bias) be_->add_bias_rows(ws_v_, L.v_bias, kv_dim_, n);

        // qk_norm_wide is OLMoE's whole-row convention; false is the per-head
        // one Qwen3 and Gemma3 use. Decided once at load from the stored
        // weight width -- see ModelConfig::qk_norm_wide.
        if (mc.qk_norm)
            be_->qk_norm(ws_q_, ws_k_, L.q_norm, L.k_norm, lh, mc.n_head_kv,
                         mc.head_dim, n, mc.rms_eps, mc.qk_norm_wide);

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
        if (!kbase || !vbase) {
            // Unreachable in practice: take_slot() evicts any slot that does
            // not already hold this layer, and a layer already hot returns
            // early above. Fail loudly rather than attend over a null cache.
            KRK_ERROR("kv: no resident slot for layer %lld", (long long)l);
            return;
        }
        d.layer_stride = 0;
        // RoPE + KV append + attention ride one launch on the decode
        // row (kernels/attention.hpp, attn_fused_decode): the kernel
        // ropes q in registers and k straight into its cache slot, so
        // the k/v rows never round-trip through global memory between
        // three launches. Gated on what the fused kernel cannot
        // express: qk_norm (which must run between the projections
        // and the rotation), multi-token rows (prefill), a packed
        // output gate (which must multiply the result afterwards), a
        // sliding window (the fused kernel has no bound) and a NeoX rope
        // pairing (the fused kernel rotates adjacent channels only, and the
        // two conventions rotate different channels, so it would silently
        // disagree with the CPU reference and with the fallback below).
        // Unsupported backends and shapes keep the separate chain.
        if (!(n == 1 && !mc.qk_norm && !packed_gate && d.window == 0 &&
              !mc.attn_gate && !mc.rope_neox &&
              be_->attn_fused_chain(ws_attn_, ws_q_, kbase, vbase,
                                      ws_k_, ws_v_, d,
                                      model_.inv_freq_at(l).data(),
                                      mc.rope_scale, model_.rope_frac_at(l)))) {
            be_->rope(ws_q_, ws_k_, lh, mc.n_head_kv, mc.head_dim,
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
        dump_row("attn.k", l, ws_k_, kv_dim_, n);
        dump_row("attn.out", l, ws_attn_, lq, n);

        // laguna's output gate: a separate [n_head] projection of the same
        // attention-norm output q/k/v read, softplus'd, and applied as one
        // scalar per head over that head's whole output row. Before wo, which
        // is what makes it a gate on the attention result rather than on the
        // residual.
        if (mc.attn_gate && L.wattn_gate.present()) {
            be_->gemm(ws_agate_, ws_xn_, L.wattn_gate.data, L.wattn_gate.type, lh,
                      n_embd_, n);
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

        be_->gemm(ws_x2_, ws_attn_, L.wo.data, L.wo.type, n_embd_, lq, n);
        dump_row("attn.proj", l, ws_x2_, n_embd_, n);
        be_->add_inplace(ws_x_, ws_x2_, n * n_embd_);
        dump_row("attn.res", l, ws_x_, n_embd_, n);

        be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, n, n_embd_, mc.rms_eps);
        dump_row("attn.ffnn", l, ws_xn_, n_embd_, n);
        if (L.moe) {
            moe_ffn(L, l, n);
        } else {
            dense_ffn(L, l, n);
        }
        dump_stage("attn", l);
    }
    // The pre-final-norm state, which a capture list names with the id
    // `mc.n_layer`: the input of the block that does not exist. It is what the
    // drafter's last feature block reads.
    dflash_capture(mc.n_layer, n);
    dump_stage("final", mc.n_layer);

    // Vocab projection: the expensive part of the head, so it runs on as few
    // rows as the caller asked for. Prefill chunks skip it entirely.
    if (mode == LogitsNone) return;
    if (mode == LogitsLast) {
        head_compute(n - 1);
        return;
    }
    for (i32 r = 0; r < n; r++) head_compute(r);
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

void Engine::dense_ffn(const LayerWeights &L, i32 l, i64 n) {
    // gate/up share the FFN-norm output: one fused launch on the decode row.
    if (L.wgate.type == L.wup.type) {
        void *gu[2] = {ws_gate_, ws_up_};
        const void *wgu[2] = {L.wgate.data, L.wup.data};
        const i64 ngu[2] = {n_ff_, n_ff_};
        be_->gemm_group(gu, ws_xn_, wgu, L.wgate.type, ngu, n_embd_, 2, n);
    } else {
        be_->gemm(ws_gate_, ws_xn_, L.wgate.data, L.wgate.type, n_ff_, n_embd_, n);
        be_->gemm(ws_up_, ws_xn_, L.wup.data, L.wup.type, n_ff_, n_embd_, n);
    }
    be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * n_ff_);
    dump_row("attn.ffng", l, ws_gate_, n_ff_, n);
    be_->gemm(ws_x2_, ws_gate_, L.wdown.data, L.wdown.type, n_embd_, n_ff_, n);
    dump_row("attn.down", l, ws_x2_, n_embd_, n);
    be_->add_inplace(ws_x_, ws_x2_, n * n_embd_);
}

void Engine::forward(const i32 *toks, i32 n, i32 pos0, bool want_logits) {
    forward_core(toks, n, pos0, want_logits ? LogitsLast : LogitsNone);
    kv_pos_ = std::max<i64>(kv_pos_, static_cast<i64>(pos0) + n);
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
    if (kv_pos_ - pos > 64) kv_rollback_zero(pos);
    kv_pos_ = pos;
    return true;
}

void Engine::kv_rollback_zero(i64 pos) {
    const size_t as = be_->act_size();
    const size_t row = static_cast<size_t>(kv_dim_) * as;
    for (i64 l = 0; l < model_.cfg().n_layer; l++) {
        u8 *kb = static_cast<u8 *>(kvt_k_.base(l)) + static_cast<size_t>(pos) * row;
        u8 *vb = static_cast<u8 *>(kvt_v_.base(l)) + static_cast<size_t>(pos) * row;
        if (!kb || !vb) return;
        be_->fill0(kb, (static_cast<size_t>(kv_pos_) - static_cast<size_t>(pos)) * row);
        be_->fill0(vb, (static_cast<size_t>(kv_pos_) - static_cast<size_t>(pos)) * row);
    }
}

void Engine::configure_expert_cache() {
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return;

    // Device memory policy, in one place, because it decides how much of the
    // card the user is left with.
    //
    // The rule: plan for 6 GiB of device memory in total, and step up by 2 GiB
    // only when there is both the headroom for it and a model that needs it,
    // never above 12 GiB. A small model on a large card therefore uses a few
    // hundred MiB, a large MoE on a 16 GiB card stops at 12 GiB instead of
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
    // The ceiling was 12 GiB, chosen against a 15.9 GiB card for the big MoEs.
    // Measured against the real files, that ceiling is what keeps a *small* MoE
    // streaming forever: Laguna XS 2.1 IQ3_XXS carries 11.20 GiB of experts and
    // 0.86 GiB of dense weights, so its whole routed set needs 12.06 GiB of
    // budget -- and at the 12 GiB cap it sat at 81.6% resident, re-reading
    // hundreds of experts from disk on every token even though the card had the
    // room. A card that can hold the model should hold it: a miss that costs a
    // file read is only worth its VRAM when the set genuinely does not fit.
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

    size_t vram_cap = 0;
    if (cfg_.vram_cap_mb > 0) {
        vram_cap = static_cast<size_t>(cfg_.vram_cap_mb) * 1024u * 1024u;
        if (vram_cap > hard) vram_cap = hard;
    } else {
        vram_cap = kCapFirst < hard ? kCapFirst : hard;
        while (vram_cap < kCapMax && vram_cap + kCapStep <= hard &&
               demand > vram_cap)
            vram_cap += kCapStep;
    }

    size_t budget = 0;
    if (cfg_.expert_cache_mb > 0) {
        // An explicit restriction wins over the policy: the user has said how
        // much they want to spend, in expert bytes alone.
        budget = static_cast<size_t>(cfg_.expert_cache_mb) * 1024u * 1024u;
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
    KRK_INFO("vram policy: %.1f GiB cap (%.1f GiB already in use, %.1f GiB "
             "demand, %.1f GiB free of %.1f GiB) -> expert budget %.1f MiB",
             static_cast<f64>(vram_cap) / static_cast<f64>(gib),
             static_cast<f64>(used_now) / static_cast<f64>(gib),
             static_cast<f64>(demand) / static_cast<f64>(gib),
             static_cast<f64>(free_now) / static_cast<f64>(gib),
             static_cast<f64>(total_vram) / static_cast<f64>(gib),
             static_cast<f64>(budget) / (1024.0 * 1024.0));

    if (cfg_.expert_cache_slots > 0) {
        const size_t one = model_.max_expert_bytes();
        if (one > 0)
            budget = one * static_cast<size_t>(cfg_.expert_cache_slots);
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
    const size_t ram_total = host_total_bytes();
    size_t host_budget = 0;
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
        const size_t avail = host_available_bytes();
        if (avail > 0 && host_budget > avail / 2) host_budget = avail / 2;
    }

    size_t warm = 0;
    if (cfg_.expert_warm_mb > 0) {
        warm = static_cast<size_t>(cfg_.expert_warm_mb) * 1024u * 1024u;
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
        const size_t avail = host_available_bytes();
        size_t want = host_budget;
        if (avail > 0 && want > (avail / 2)) want = avail / 2;
        const size_t total = model_.total_expert_bytes();
        if (total > 0 && total < want) want = total; // never cache past the corpus
        const size_t floor_bytes = model_.max_expert_bytes() * 4;
        warm = want >= floor_bytes ? want : 0;
    }
    if (host_budget > 0 || cfg_.host_ram_mb > 0)
        KRK_INFO("ram policy: %.1f GiB budget (%.1f GiB installed, %.1f GiB "
                 "free) -> WARM tier %.0f MiB%s",
                 static_cast<f64>(host_budget) / static_cast<f64>(gib),
                 static_cast<f64>(ram_total) / static_cast<f64>(gib),
                 static_cast<f64>(host_available_bytes()) / static_cast<f64>(gib),
                 static_cast<f64>(warm) / (1024.0 * 1024.0),
                 cfg_.expert_warm_mb == 0 ? " (disabled by --expert-warm-mb 0)"
                                          : "");

    model_.set_expert_budget(budget, warm);

    // Prefetch is opt-in. Read-through admits every expert the run actually
    // touches, so an eager sweep of the corpus is only worth its wall time when
    // the run is short and its routing is broad — and it costs RAM and time
    // before the first token (measured on Laguna XS.2: 4 GiB staged in 1903 ms
    // against ~1.9 s of decode it then improved). --expert-warm-prefetch, or
    // KRK_EXPERT_PRELOAD=1, turns it on; KRK_EXPERT_PRELOAD=0 forces it off, so
    // the A/B is one environment variable away.
    const char *pre_env = std::getenv("KRK_EXPERT_PRELOAD");
    const bool prefetch = pre_env && pre_env[0] == '0'
                              ? false
                              : (cfg_.expert_warm_prefetch ||
                                 (pre_env && pre_env[0] == '1'));
    if (warm > 0 && be_->caps().vram_free > 0 && prefetch) {
        const std::chrono::steady_clock::time_point t0 =
            std::chrono::steady_clock::now();
        const size_t ram_before = host_available_bytes();
        // Expert-major, layer-minor: with a bounded tier this fills every
        // layer's expert 0 before any layer's expert 1, so a tier smaller
        // than the corpus is shared out evenly instead of being spent on the
        // first few layers. Routing mass is the better order when the model
        // carries a .krakenexperts.json; this is the order that needs no
        // prior run and does not concentrate the hot set in one place.
        size_t staged = 0;
        for (i32 e = 0; e < mc.n_expert; e++) {
            for (i32 l = 0; l < mc.n_layer; l++) {
                const LayerWeights &L = model_.layers()[static_cast<size_t>(l)];
                if (!L.experts.present()) continue;
                staged += model_.experts().preload_one(L.experts, l, e);
            }
        }
        const f64 stage_ms = std::chrono::duration<f64, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
        // Both sides of the cost: the pageable tier is what it holds, and
        // the drop in free RAM is what it actually took, mapping pages
        // included. The two differ by the size of the staged ranges.
        const size_t ram_after = host_available_bytes();
        const f64 ram_mib = ram_before > ram_after
                                ? static_cast<f64>(ram_before - ram_after) / 1048576.0
                                : 0.0;
        KRK_INFO("expert WARM: prefetched %.1f MiB in %.0f ms, host RAM in use "
                 "up %.1f MiB (a VRAM miss is a promotion now, not a file read)",
                 static_cast<f64>(staged) / 1048576.0, stage_ms, ram_mib);
    }
    char warm_note[80] = "";
    if (warm > 0)
        std::snprintf(warm_note, sizeof(warm_note),
                      " + WARM %.0f MiB pageable%s",
                      static_cast<f64>(warm) / (1024.0 * 1024.0),
                      prefetch ? " (prefetched)" : "");
    KRK_INFO("expert cache: budget %.1f MiB%s, %d experts x top-%d, %d layers, "
             "policy LFU+aging (pin at %u hits, decay every %u); cold reads are "
             "read-through",
             static_cast<f64>(budget) / (1024.0 * 1024.0), warm_note, mc.n_expert,
             mc.n_expert_used, mc.n_layer, ExpertCache::kPinThreshold,
             ExpertCache::kPinDecay);
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
    dflash_ = d;
    if (draft_tokens_ <= 0) draft_tokens_ = 4;
    KRK_INFO("dflash drafter loaded: %s — %d captured layers, block size %d, "
             "speculative window %d",
             d->name().c_str(), d->n_aux(), d->block_size(),
             std::min<i32>(draft_tokens_, d->block_size() - 1));
    return true;
}

void Engine::unload_dflash() {
    if (!dflash_) return;
    dflash_->unload();
    delete dflash_;
    dflash_ = nullptr;
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
    spec_steps_ = 0;
}

void Engine::recurrent_reset() {
    if (!conv_state_ || !rec_state_) return;
    be_->fill0(conv_state_, static_cast<size_t>(rec_layers_) *
                                  static_cast<size_t>(conv_state_span_) * sizeof(f32));
    be_->fill0(rec_state_, static_cast<size_t>(rec_layers_) *
                                 static_cast<size_t>(rec_state_span_) * sizeof(f32));
}

void Engine::gdn_forward(const LayerWeights &L, i32 l, i32 n) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const i64 kdim = model_.key_dim();
    const i64 cdim = conv_dim_;
    const i64 vdim = value_dim_;
    const i32 ksize = mc.ssm_d_conv;
    const i32 ri = model_.recurrent_index(l);

    // 1. One projection for [q | k | v] and one for the z gate. Both read the
    //    same attention-norm output, so they are independent.
    be_->gemm(ws_qkv_, ws_xn_, L.wqkv.data, L.wqkv.type, cdim, n_embd_, n);
    be_->gemm(ws_z_, ws_xn_, L.wqkv_gate.data, L.wqkv_gate.type, vdim, n_embd_, n);

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
    be_->gemm(ws_ssm_, ws_xn_, L.ssm_alpha.data, L.ssm_alpha.type, mc.ssm_dt_rank,
              n_embd_, n);
    be_->add_bias_cols(ws_ssm_, L.ssm_dt, n, mc.ssm_dt_rank);
    be_->softplus_act(ws_ssm_, static_cast<i64>(n) * mc.ssm_dt_rank);
    be_->scale_cols(ws_ssm_, L.ssm_a, n, mc.ssm_dt_rank);

    be_->gemm(ws_beta_, ws_xn_, L.ssm_beta.data, L.ssm_beta.type, mc.ssm_dt_rank,
              n_embd_, n);
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

    // 7. Output projection and the residual.
    be_->gemm(ws_x2_, ws_h_, L.ssm_out.data, L.ssm_out.type, n_embd_, vdim, n);
    be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(n) * n_embd_);
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
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return;
    const i32 ne = mc.n_expert;
    const size_t one = model_.max_expert_bytes();
    // Size and mtime travel with the index so a loader can refuse a list
    // written for a different build of the same weights. Matching the Laguna
    // edge0-index fields, which the collections already ship.
    unsigned long long src_size = 0, src_mtime = 0;
    {
        struct stat st;
        if (::stat(cfg_.model_path.c_str(), &st) == 0) {
            src_size = static_cast<unsigned long long>(st.st_size);
            src_mtime = static_cast<unsigned long long>(st.st_mtime);
        }
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
                 expert_stub_ ? "routers-only" : "full-forward",
                 cfg_.model_path.c_str(),
                 src_size, src_mtime,
                 mc.arch.c_str(), mc.n_layer, ne, mc.n_expert_used,
                 static_cast<unsigned long long>(one),
                 static_cast<long long>(expert_scan_tokens_));
    for (i32 l = 0; l < mc.n_layer; l++) {
        const LayerWeights &L = model_.layers()[static_cast<size_t>(l)];
        if (!L.moe) continue;
        const double *mass = expert_mass_.data() + static_cast<size_t>(l) * ne;
        const u64 *hits = expert_hits_.data() + static_cast<size_t>(l) * ne;
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
    KRK_INFO("expert scan: wrote %s (%lld positions, %s)", path.c_str(),
             static_cast<long long>(expert_scan_tokens_),
             expert_stub_ ? "routers only, experts stubbed" : "full forward");
}

void Engine::moe_ffn(const LayerWeights &L, i32 layer, i32 n) {
    const ModelConfig &mc = model_.cfg();
    const size_t as = be_->act_size();
    const i32 ne = mc.n_expert;
    const i32 k = std::min<i32>(mc.n_expert_used > 0 ? mc.n_expert_used : ne, ne);
    const i64 ff_exp = mc.n_ff_exp;
    const i64 ff_sh = mc.n_ff_shexp;

    be_->fill0(ws_ffn_, static_cast<size_t>(n) * static_cast<size_t>(n_embd_) * as);

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
            if (expert_scan_) {
                double *mass = expert_mass_.data() +
                               static_cast<size_t>(layer) * static_cast<size_t>(ne);
                for (i32 e = 0; e < ne; e++)
                    mass[e] += static_cast<double>(moe_prob_[static_cast<size_t>(e)]);
            }

            // The per-expert bias (`ffn_exp_probs_b.bias`) shifts *selection*
            // only: the weights stay the unbiased probabilities, which is what
            // the reference means by "leave probs unbiased as it's later used
            // to get expert weights". Adding it to the weights as well is a
            // plausible-looking and wrong change.
            const f32 *rb = L.router_bias;
            auto sel_score = [&](i32 e) {
                return moe_prob_[static_cast<size_t>(e)] +
                       (rb ? rb[static_cast<size_t>(e)] : 0.0f);
            };

            i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
            f32 *wt = moe_wt_.data() + static_cast<size_t>(t) * k;
            for (i32 s = 0; s < k; s++) {
                i32 best = -1;
                f32 bestp = -1.0f;
                for (i32 e = 0; e < ne; e++) {
                    bool taken = false;
                    for (i32 p = 0; p < s; p++)
                        if (sel[p] == e) { taken = true; break; }
                    if (taken) continue;
                    if (sel_score(e) > bestp) {
                        bestp = sel_score(e);
                        best = e;
                    }
                }
                sel[s] = best;
                // The weight is the probability, never the biased score.
                wt[s] = best >= 0 ? moe_prob_[static_cast<size_t>(best)] : 0.0f;
                if (expert_scan_ && best >= 0)
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
                    prefetch_seen_[static_cast<size_t>(e)] = 1;
                    prefetch_ids_.push_back(e);
                }
            }
            if (prefetch_ids_.size() > 1)
                model_.experts().prefetch_layer(
                    L.experts, layer, prefetch_ids_.data(),
                    static_cast<int>(prefetch_ids_.size()));
        }

        for (i32 e = 0; !expert_stub_ && e < ne; e++) {
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

            const i64 m = static_cast<i64>(group_rows_.size());
            plan_dev_.assign(group_rows_.begin(), group_rows_.end());
            alpha_dev_.assign(group_wt_.begin(), group_wt_.end());
            be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(), m);
            be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                        static_cast<size_t>(m) * sizeof(f32));

            // x̃ = xn[group_rows] — the permuted activation block.
            be_->gather_rows(ws_xg_, ws_xn_, static_cast<const i32 *>(ws_plan_), m,
                             n_embd_);
            // One GEMM per matrix over the whole block.
            be_->gemm(ws_gateg_, ws_xg_, re->gate.data, re->gate.type, ff_exp,
                      n_embd_, m);
            be_->gemm(ws_upg_, ws_xg_, re->up.data, re->up.type, ff_exp, n_embd_, m);
            be_->silu_mul(ws_gateg_, ws_gateg_, ws_upg_, m * ff_exp);
            be_->gemm(ws_x2_, ws_gateg_, re->down.data, re->down.type, n_embd_,
                      ff_exp, m);
            // Scatter back to the original rows, weighted by the gate.
            be_->scatter_axpy_rows(ws_ffn_, ws_x2_, static_cast<const i32 *>(ws_plan_),
                                   static_cast<const f32 *>(ws_alpha_), m, n_embd_);
        }
    }

    // The shared expert runs on every token (Qwen2-MoE); it is a single expert,
    // so it stays resident.
    if (!expert_stub_ && ff_sh > 0 && L.shexp_gate.present()) {
        be_->gemm(ws_gate_, ws_xn_, L.shexp_gate.data, L.shexp_gate.type, ff_sh, n_embd_, n);
        be_->gemm(ws_up_, ws_xn_, L.shexp_up.data, L.shexp_up.type, ff_sh, n_embd_, n);
        be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * ff_sh);
        be_->gemm(ws_x2_, ws_gate_, L.shexp_down.data, L.shexp_down.type, n_embd_, ff_sh, n);
        // Qwen3.5 scales the shared expert by a per-token sigmoid of a small
        // vector before folding it in; the Qwen2/Qwen3 schema has no such gate.
        if (L.shexp_inp_gate.present()) {
            be_->gemm(ws_ssm_, ws_xn_, L.shexp_inp_gate.data, L.shexp_inp_gate.type,
                      1, n_embd_, n);
            be_->sigmoid_act(ws_ssm_, n);
            be_->scale_rows(ws_x2_, ws_ssm_, n, n_embd_);
        }
        be_->add_inplace(ws_ffn_, ws_x2_, n * n_embd_);
    }

    be_->add_inplace(ws_x_, ws_ffn_, n * n_embd_);
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
    be_->sync();
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
    be_->sync();
    char *zone = static_cast<char *>(topk_out_);
    be_->logits_topk(ws_logits_, n_vocab_, kTopK, nullptr, 1.0f,
                     model_.cfg().logit_softcap, 1.0f,
                     reinterpret_cast<i32 *>(zone), reinterpret_cast<f32 *>(zone + 4),
                     nullptr, topk_scratch_, kTopKCap,
                     reinterpret_cast<i64 *>(zone + 8));
    be_->sync();
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

bool Engine::generate_speculative(const GenerateParams &p, GenerateResult *res,
                                  const std::vector<i32> &ids, f64 *prefill_ms,
                                  f64 *decode_ms, i64 *decode_steps) {
    Engine &draft = *draft_;
    const i32 k = draft_tokens_;

    // ---- prompt prefill on both models ------------------------------------
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        forward(ids.data() + i, n, static_cast<i32>(i), last);
    }
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        draft.forward(ids.data() + i, n, static_cast<i32>(i), last);
    }
    fetch_logits();
    draft.fetch_logits();
    *prefill_ms = prefill_timer.ms();

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
        if (p.debug_topk > 0)
            dump_topk(p.debug_topk, res->generated, pos, logits_host_, n_vocab_);
        const i32 first = argmax_of(logits_host_, n_vocab_);

        // ---- draft: propose up to k continuations -------------------------
        // Each proposal must run the draft's HEAD: without its logits the next
        // argmax would read stale values and the chain would repeat one token.
        i32 d = 0;
        while (d < k) {
            const i32 t = argmax_of(draft.logits_host_, n_vocab_);
            if (t == tok_.eos()) break; // the target confirms EOS itself
            prop[static_cast<size_t>(d)] = t;
            d++;
            draft_proposed_++;
            // The d-th proposal belongs at position pos+d-1 (the first extends
            // the prompt, whose rows end at pos-1). No need to feed the final
            // one — its continuation lies beyond the window.
            if (d >= k || pos + d >= kv_cap_ || pos + d >= draft.kv_capacity())
                break;
            draft.forward(&t, 1, static_cast<i32>(pos + d - 1), true);
            draft.fetch_logits();
        }
        spec_steps_++;

        // ---- pre-check: does the target agree with the first proposal? ----
        // This is the same test plain decode performs; on failure the round
        // costs exactly one plain decode step (the block is never run).
        if (d == 0 || first != prop[0]) {
            if (first == tok_.eos()) {
                finish_run(FinishEos);
                return true;
            }
            emit.accept(first);
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            forward(&first, 1, static_cast<i32>(pos), true);
            pos++;
            fetch_logits();
            // Keep the draft aligned: same position convention as below —
            // feed the accepted token AT pos-1 (its natural KV slot).
            draft.kv_rollback(pos - 1);
            draft.forward(&first, 1, static_cast<i32>(pos - 1), true);
            draft.fetch_logits();
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
        forward_core(prop.data(), d_eff, static_cast<i32>(pos), LogitsNone);
        kv_pos_ = pos + d_eff;
        // The head runs once per row into the single-row logits buffer, and
        // each row is downloaded before the next overwrites it. (Sizing
        // ws_logits_ for the whole block would also work; per-row keeps the
        // workspace independent of the speculation window.)
        for (i32 j = 0; j < d_eff; j++) {
            head_compute(j);
            be_->sync();
            be_->download_f32(rows.data() + static_cast<size_t>(j) * n_vocab_,
                              ws_logits_, n_vocab_);
        }
        auto row_pred = [&](i32 j) { // argmax of block row j (1-based)
            return argmax_of(rows.data() + static_cast<size_t>(j - 1) *
                                                      static_cast<size_t>(n_vocab_),
                             n_vocab_);
        };

        // ---- accept the longest greedy-matching prefix ---------------------
        i32 a = 1; // T[1] passed the pre-check
        while (a < d_eff && row_pred(a) == prop[static_cast<size_t>(a)]) a++;
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
                forward(&b, 1, static_cast<i32>(pos + emitted - 1), true);
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
        draft.kv_rollback(pos);
        draft.forward(&last_tok, 1, static_cast<i32>(pos), true);
        draft.fetch_logits();
        // The target's logits must predict pos. After a full match they came
        // from the bonus forward above; after a partial match the answer is
        // block row `emitted` (1-based), already downloaded into rows[].
        if (!full) {
            std::memcpy(logits_host_,
                        rows.data() + static_cast<size_t>(emitted - 1) *
                                          static_cast<size_t>(n_vocab_),
                        static_cast<size_t>(n_vocab_) * sizeof(f32));
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
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        forward(ids.data() + i, n, static_cast<i32>(i), last);
        if (!dflash_->commit(static_cast<i32>(i), n)) return false;
    }
    fetch_logits();
    *prefill_ms = prefill_timer.ms();

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
    const QuantTensor &embd = model_.tok_embd();
    const QuantTensor &head = model_.out_head();

    // Invariant at the top of every round, identical to the model-draft path:
    // the target's KV holds [0, pos) and logits_host_ predicts pos.
    while (res->generated < p.max_tokens && pos < kv_cap_) {
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
            forward(&first, 1, static_cast<i32>(pos), true);
            dflash_->commit(static_cast<i32>(pos), 1);
            pos++;
            fetch_logits();
            continue;
        }
        d = nrows;
        for (i32 j = 0; j < d; j++)
            prop[static_cast<size_t>(j)] =
                argmax_of(dlogits.data() + static_cast<size_t>(j) *
                                               static_cast<size_t>(n_vocab_),
                          n_vocab_);
        draft_proposed_ += static_cast<u64>(d);
        spec_steps_++;

        // ---- the same pre-check plain decode performs ---------------------
        // On disagreement the round costs exactly one plain step and the block
        // is never verified.
        if (first != prop[0]) {
            if (first == tok_.eos()) {
                finish_run(FinishEos);
                return true;
            }
            emit.accept(first);
            if (pos + 1 >= kv_cap_) {
                finish_run(FinishContext);
                return true;
            }
            forward(&first, 1, static_cast<i32>(pos), true);
            dflash_->commit(static_cast<i32>(pos), 1);
            pos++;
            fetch_logits();
            continue;
        }

        // ---- verify: one batched forward of the d proposals ---------------
        i32 d_eff = d;
        while (d_eff > 0 && pos + d_eff > kv_cap_) d_eff--;
        if (d_eff <= 0) {
            finish_run(FinishContext);
            return true;
        }
        forward_core(prop.data(), d_eff, static_cast<i32>(pos), LogitsNone);
        kv_pos_ = pos + d_eff;
        for (i32 j = 0; j < d_eff; j++) {
            head_compute(j);
            be_->sync();
            be_->download_f32(rows.data() + static_cast<size_t>(j) * n_vocab_,
                              ws_logits_, n_vocab_);
        }
        auto row_pred = [&](i32 j) {
            return argmax_of(rows.data() + static_cast<size_t>(j - 1) *
                                               static_cast<size_t>(n_vocab_),
                             n_vocab_);
        };

        i32 a = 1;
        while (a < d_eff && row_pred(a) == prop[static_cast<size_t>(a)]) a++;
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
                forward(&b, 1, static_cast<i32>(pos + emitted - 1), true);
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
        // The target's logits must predict pos. After a full match they came
        // from the bonus forward above; after a partial match the answer is
        // block row `emitted` (1-based), already downloaded into rows[].
        if (!full)
            std::memcpy(logits_host_,
                        rows.data() + static_cast<size_t>(emitted - 1) *
                                          static_cast<size_t>(n_vocab_),
                        static_cast<size_t>(n_vocab_) * sizeof(f32));
    }

    finish_run(res->generated >= p.max_tokens ? FinishLength : FinishContext);
    return true;
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
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        forward(ids.data() + i, n, static_cast<i32>(i), last);
    }
    fetch_logits(device_topk);
    if (expert_scan_) expert_scan_tokens_ += static_cast<i64>(ids.size());
    res->prefill_ms = prefill_timer.ms();

    Sampler sampler;
    sampler.reset(p.sampler);
    sampler.accept(ids.data(), static_cast<int>(ids.size()));
    TokenEmitter emit(p, tok_, sampler, res, ids);

    i64 pos = static_cast<i64>(ids.size());
    const Timer decode_timer;
    i64 timed_decode_steps = 0;

    for (i32 gen = 0; gen < p.max_tokens; gen++) {
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
        forward(&token, 1, static_cast<i32>(pos), true);
        auto t1 = std::chrono::steady_clock::now();
        pos++;
        fetch_logits(device_topk);
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
        if (expert_scan_) expert_scan_tokens_++;
    }

    emit.finish(res->finish == FinishStop);

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
