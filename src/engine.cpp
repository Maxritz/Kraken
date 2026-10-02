// engine.cpp — prefill (batched) + decode (single token) with a paged-free
// contiguous KV cache, streaming output and stop-string handling.
#include "krk/engine.hpp"
#include "par_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

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

} // namespace

Engine::~Engine() { shutdown(); }

bool Engine::init(Backend *be, const EngineConfig &cfg, std::string *err) {
    be_ = be;
    cfg_ = cfg;
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
    q_proj_ = (mc.arch == "qwen35" || mc.arch == "qwen35moe") ? 2 * q_dim_ : q_dim_;
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
    kcache_ = alloc(static_cast<size_t>(mc.n_layer) * static_cast<size_t>(kv_cap_) *
                    static_cast<size_t>(kv_dim_) * as);
    vcache_ = alloc(static_cast<size_t>(mc.n_layer) * static_cast<size_t>(kv_cap_) *
                    static_cast<size_t>(kv_dim_) * as);
    ws_x_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    ws_xn_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    ws_x2_ = alloc(static_cast<size_t>(C * n_embd_) * as);
    // q_proj_ is the projection's width: Qwen3.5 packs a per-head output gate
    // behind the query, so the buffer has to hold both until the split.
    ws_q_ = alloc(static_cast<size_t>(C * q_proj_) * as);
    ws_k_ = alloc(static_cast<size_t>(C * kv_dim_) * as);
    ws_v_ = alloc(static_cast<size_t>(C * kv_dim_) * as);
    ws_attn_ = alloc(static_cast<size_t>(C * q_dim_) * as);
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
        ws_agate_ = alloc(static_cast<size_t>(C * q_dim_) * as);
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
        moe_wt_.resize(static_cast<size_t>(C) * static_cast<size_t>(k));
        group_rows_.reserve(static_cast<size_t>(C));
        group_wt_.reserve(static_cast<size_t>(C));
        plan_dev_.reserve(static_cast<size_t>(C));
        alpha_dev_.reserve(static_cast<size_t>(C));
    }
    configure_expert_cache();
    be_->fill0(kcache_, static_cast<size_t>(mc.n_layer) * static_cast<size_t>(kv_cap_) *
                            static_cast<size_t>(kv_dim_) * as);
    be_->fill0(vcache_, static_cast<size_t>(mc.n_layer) * static_cast<size_t>(kv_cap_) *
                            static_cast<size_t>(kv_dim_) * as);
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
    KRK_INFO("workspaces ready: chunk=%d ctx=%lld KV=%.0f MiB (load %.0f ms)", chunk_,
             static_cast<long long>(kv_cap_), total_vram_mb, load_ms);

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
    unload_draft();
    if (!be_) {
        model_.unload();
        return;
    }
    be_->sync();
    for (void **p : {&kcache_, &vcache_, &ws_x_, &ws_xn_, &ws_x2_, &ws_q_, &ws_k_,
                     &ws_v_, &ws_attn_, &ws_gate_, &ws_up_, &ws_logits_, &ws_router_,
                     &ws_ffn_, &ws_xg_, &ws_gateg_, &ws_upg_, &ws_plan_, &ws_alpha_,
                     &ws_qkv_, &ws_z_, &ws_h_, &ws_ssm_, &ws_beta_, &ws_agate_,
                     &conv_state_, &rec_state_}) {
        if (*p) {
            be_->release(*p);
            *p = nullptr;
        }
    }
    host_free(logits_host_);
    logits_host_ = nullptr;
    model_.unload();
    be_ = nullptr;
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

    for (i32 l = 0; l < mc.n_layer; l++) {
        const LayerWeights &L = layers[static_cast<size_t>(l)];

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
                be_->gemm(ws_gate_, ws_xn_, L.wgate.data, L.wgate.type, n_ff_, n_embd_, n);
                be_->gemm(ws_up_, ws_xn_, L.wup.data, L.wup.type, n_ff_, n_embd_, n);
                be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * n_ff_);
                be_->gemm(ws_x2_, ws_gate_, L.wdown.data, L.wdown.type, n_embd_, n_ff_, n);
                be_->add_inplace(ws_x_, ws_x2_, n * n_embd_);
            }
            continue;
        }

        // q/k/v all read the attention-norm output and have no
        // dependency between them, so they ride one fused launch
        // on the decode row (backend falls back per-matrix for
        // prefill rows or a dtype mix).
        if (L.wq.type == L.wk.type && L.wk.type == L.wv.type) {
            void *qkv[3] = {ws_q_, ws_k_, ws_v_};
            const void *wqkv[3] = {L.wq.data, L.wk.data, L.wv.data};
            const i64 nqkv[3] = {q_proj_, kv_dim_, kv_dim_};
            be_->gemm_group(qkv, ws_xn_, wqkv, L.wq.type,
                            nqkv, n_embd_, 3, n);
        } else {
            be_->gemm(ws_q_, ws_xn_, L.wq.data, L.wq.type, q_proj_, n_embd_, n);
            be_->gemm(ws_k_, ws_xn_, L.wk.data, L.wk.type, kv_dim_, n_embd_, n);
            be_->gemm(ws_v_, ws_xn_, L.wv.data, L.wv.type, kv_dim_, n_embd_, n);
        }

        // Qwen3.5 interleaves each head's output gate into the query
        // projection, so the packed rows have to be split before anything
        // else touches them. The query half stays in place.
        if (q_proj_ != q_dim_)
            be_->qwen3_next_split(ws_q_, ws_agate_, ws_q_, n, mc.n_head, mc.head_dim);

        if (L.q_bias) be_->add_bias_rows(ws_q_, L.q_bias, q_dim_, n);
        if (L.k_bias) be_->add_bias_rows(ws_k_, L.k_bias, kv_dim_, n);
        if (L.v_bias) be_->add_bias_rows(ws_v_, L.v_bias, kv_dim_, n);

        if (mc.qk_norm)
            be_->qk_norm(ws_q_, ws_k_, L.q_norm, L.k_norm, mc.n_head, mc.n_head_kv,
                         mc.head_dim, n, mc.rms_eps);

        d.layer = l;
        // RoPE + KV append + attention ride one launch on the decode
        // row (kernels/attention.hpp, attn_fused_decode): the kernel
        // ropes q in registers and k straight into its cache slot, so
        // the k/v rows never round-trip through global memory between
        // three launches. Gated on what the fused kernel cannot
        // express: qk_norm (which must run between the projections
        // and the rotation), multi-token rows (prefill), and a packed
        // output gate (which must multiply the result afterwards).
        // Unsupported backends and shapes keep the separate chain.
        if (!(n == 1 && !mc.qk_norm && q_proj_ == q_dim_ &&
              be_->attn_fused_chain(ws_attn_, ws_q_, kcache_, vcache_,
                                      ws_k_, ws_v_, d,
                                      model_.inv_freq().data(),
                                      mc.rope_scale, mc.rope_frac))) {
            be_->rope(ws_q_, ws_k_, mc.n_head, mc.n_head_kv, mc.head_dim,
                      n, pos0, model_.inv_freq().data(), mc.rope_scale,
                      mc.rope_frac);
            be_->kv_append(kcache_, vcache_, ws_k_, ws_v_, d);
            be_->attention(ws_attn_, ws_q_, kcache_, vcache_, d);
        }

        // Qwen3.5's output gate is a sigmoid applied to the attention result,
        // not to the projection: gate it before wo.
        if (q_proj_ != q_dim_) {
            be_->sigmoid_act(ws_agate_, static_cast<i64>(n) * q_dim_);
            be_->mul_act(ws_attn_, ws_agate_, static_cast<i64>(n) * q_dim_);
        }

        be_->gemm(ws_x2_, ws_attn_, L.wo.data, L.wo.type, n_embd_, q_dim_, n);
        be_->add_inplace(ws_x_, ws_x2_, n * n_embd_);

        be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, n, n_embd_, mc.rms_eps);
        if (L.moe) {
            moe_ffn(L, l, n);
        } else {
            // gate/up share the FFN-norm output: one fused
            // launch on the decode row, like q/k/v above.
            if (L.wgate.type == L.wup.type) {
                void *gu[2] = {ws_gate_, ws_up_};
                const void *wgu[2] = {L.wgate.data, L.wup.data};
                const i64 ngu[2] = {n_ff_, n_ff_};
                be_->gemm_group(gu, ws_xn_, wgu, L.wgate.type,
                                ngu, n_embd_, 2, n);
            } else {
                be_->gemm(ws_gate_, ws_xn_, L.wgate.data, L.wgate.type, n_ff_, n_embd_, n);
                be_->gemm(ws_up_, ws_xn_, L.wup.data, L.wup.type, n_ff_, n_embd_, n);
            }
            be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * n_ff_);
            be_->gemm(ws_x2_, ws_gate_, L.wdown.data, L.wdown.type, n_embd_, n_ff_, n);
            be_->add_inplace(ws_x_, ws_x2_, n * n_embd_);
        }
    }

    // Vocab projection: the expensive part of the head, so it runs on as few
    // rows as the caller asked for. Prefill chunks skip it entirely.
    if (mode == LogitsNone) return;
    if (mode == LogitsLast) {
        head_compute(n - 1);
        return;
    }
    for (i32 r = 0; r < n; r++) head_compute(r);
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
    const size_t layer_span = static_cast<size_t>(kv_cap_) * row;
    for (i64 l = 0; l < model_.cfg().n_layer; l++) {
        u8 *kb = static_cast<u8 *>(kcache_) + static_cast<size_t>(l) * layer_span +
                 static_cast<size_t>(pos) * row;
        u8 *vb = static_cast<u8 *>(vcache_) + static_cast<size_t>(l) * layer_span +
                 static_cast<size_t>(pos) * row;
        be_->fill0(kb, (static_cast<size_t>(kv_pos_) - static_cast<size_t>(pos)) * row);
        be_->fill0(vb, (static_cast<size_t>(kv_pos_) - static_cast<size_t>(pos)) * row);
    }
}

void Engine::configure_expert_cache() {
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return;

    size_t budget = 0;
    if (cfg_.expert_cache_mb > 0) {
        budget = static_cast<size_t>(cfg_.expert_cache_mb) * 1024u * 1024u;
    } else {
        const DeviceCaps &dc = be_->caps();
        // Auto: prefer the whole routed-expert set when it fits in a
        // generous share of free memory. A partial budget is worse than
        // useless at MoE scale — top-k routing touches every expert
        // within a few dozen tokens, so a set that only partly fits
        // churns (evict + free + alloc + re-upload) forever, which costs
        // far more than the memory the cap saves. When the set does not
        // fit, fall back to a generous share of free memory and let the
        // cache page. On the CPU (no VRAM accounting) allow a generous
        // fixed share of host RAM.
        budget = dc.vram_free > 0
                     ? static_cast<size_t>(static_cast<f64>(dc.vram_free) * 0.9)
                     : static_cast<size_t>(512) * 1024u * 1024u;
        const size_t total = model_.total_expert_bytes();
        if (total > 0 && total < budget) budget = total;
    }

    if (cfg_.expert_cache_slots > 0) {
        const size_t one = model_.max_expert_bytes();
        if (one > 0)
            budget = one * static_cast<size_t>(cfg_.expert_cache_slots);
    }

    // The second tier is sized independently of the device budget: VRAM holds
    // what is about to be used, the tier holds what was just used, and a
    // demoted expert comes back as a DMA rather than a re-read of the mapping.
    size_t l2 = 0;
    if (cfg_.expert_l2_mb > 0)
        l2 = static_cast<size_t>(cfg_.expert_l2_mb) * 1024u * 1024u;

    model_.set_expert_budget(budget, l2);
    char l2_note[64] = "";
    if (l2 > 0)
        std::snprintf(l2_note, sizeof(l2_note), " + %.0f MiB L2",
                      static_cast<f64>(l2) / (1024.0 * 1024.0));
    KRK_INFO("expert cache: budget %.1f MiB%s, %d experts x top-%d, %d layers, "
             "policy LFU+aging (pin at %u hits, decay every %u)",
             static_cast<f64>(budget) / (1024.0 * 1024.0), l2_note, mc.n_expert,
             mc.n_expert_used, mc.n_layer, ExpertCache::kPinThreshold,
             ExpertCache::kPinDecay);
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
    if (draft_) return true; // already loaded

    // The draft shares this engine's backend: one device, interleaved kernels,
    // no staging copies. Its Model owns its own weights and its own KV cache.
    draft_ = new Engine();
    EngineConfig dcfg = cfg_;
    dcfg.model_path = path;
    dcfg.warmup = false;
    dcfg.expert_cache_mb = cfg_.expert_cache_mb;
    dcfg.expert_l2_mb = cfg_.expert_l2_mb;
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

    // 6. Gated normalization: a plain RMSNorm per head (its weight is stored
    //    verbatim, not zero-centred) then the silu(z) gate.
    be_->rmsnorm(ws_h_, ws_h_, L.ssm_norm, static_cast<i64>(n) * mc.ssm_dt_rank,
                 mc.ssm_d_state, mc.rms_eps);
    be_->silu_mul(ws_h_, ws_z_, ws_h_, static_cast<i64>(n) * vdim);

    // 7. Output projection and the residual.
    be_->gemm(ws_x2_, ws_h_, L.ssm_out.data, L.ssm_out.type, n_embd_, vdim, n);
    be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(n) * n_embd_);
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
        be_->sync();
        be_->download_f32(router_host_.data(), ws_router_, static_cast<i64>(n) * ne);

        // ---- build the routing plan -------------------------------------
        // moe_sel_/moe_wt_ are [token, slot]; the gate weights are renormalized
        // over the k selected experts (the Qwen convention).
        for (i32 t = 0; t < n; t++) {
            const f32 *logits = router_host_.data() + static_cast<i64>(t) * ne;
            f32 mx = logits[0];
            for (i32 e = 1; e < ne; e++) mx = std::max(mx, logits[e]);
            f32 sum = 0;
            for (i32 e = 0; e < ne; e++) {
                moe_prob_[static_cast<size_t>(e)] = std::exp(logits[e] - mx);
                sum += moe_prob_[static_cast<size_t>(e)];
            }
            const f32 inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (i32 e = 0; e < ne; e++) moe_prob_[static_cast<size_t>(e)] *= inv;

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
                    if (moe_prob_[static_cast<size_t>(e)] > bestp) {
                        bestp = moe_prob_[static_cast<size_t>(e)];
                        best = e;
                    }
                }
                sel[s] = best;
                wt[s] = bestp > 0 ? bestp : 0.0f;
            }
            f32 ksum = 0;
            for (i32 s = 0; s < k; s++) ksum += wt[s];
            const f32 kinv = ksum > 0 ? 1.0f / ksum : 0.0f;
            for (i32 s = 0; s < k; s++) wt[s] *= kinv;
        }

        // ---- grouped expert GEMMs ----------------------------------------
        // Every (token, slot) pair is one work item. Items that picked the same
        // expert share one permuted activation block and one set of GEMMs, so a
        // layer costs O(k) GEMMs per matrix instead of O(k * n).
        for (i32 e = 0; e < ne; e++) {
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
    if (ff_sh > 0 && L.shexp_gate.present()) {
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

void Engine::fetch_logits() {
    be_->sync();
    be_->download_f32(logits_host_, ws_logits_, n_vocab_);
}

bool Engine::generate_speculative(const GenerateParams &p, GenerateResult *res,
                                  const std::vector<i32> &ids, f64 *prefill_ms,
                                  f64 *decode_ms) {
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
    if (p.echo_prompt) {
        res->text = p.prompt;
        res->tokens = ids;
        p.sink.emit(p.prompt.c_str(), -1, false);
    }

    // A stop string may straddle token boundaries; hold streamed text back
    // exactly as the plain loop does.
    size_t max_stop = 0;
    for (const std::string &s : p.stop) max_stop = std::max(max_stop, s.size());
    std::string pending;
    auto emit_token = [&](i32 token) {
        const std::string piece = tok_.decode(&token, 1);
        res->tokens.push_back(token);
        res->text += piece;
        res->generated++;
        sampler.accept(token);
        bool stop_hit = false;
        size_t stop_len = 0;
        for (const std::string &s : p.stop) {
            if (!s.empty() && ends_with(res->text, s) && s.size() > stop_len) {
                stop_hit = true;
                stop_len = s.size();
            }
        }
        pending += piece;
        if (stop_hit) {
            res->text.resize(res->text.size() - stop_len);
            const size_t keep =
                pending.size() >= stop_len ? pending.size() - stop_len : 0;
            pending.resize(keep);
        }
        if (pending.size() > max_stop) {
            const size_t flush = pending.size() - max_stop;
            p.sink.emit(pending.substr(0, flush).c_str(), token, false);
            pending.erase(0, flush);
        }
        return stop_hit;
    };

    const Timer decode_timer;
    auto finish_run = [&](Finish f) {
        res->finish = f;
        *decode_ms = decode_timer.ms();
        if (f != FinishStop && !pending.empty())
            p.sink.emit(pending.c_str(), -1, false);
        p.sink.emit(nullptr, -1, true);
    };

    i64 pos = static_cast<i64>(ids.size());
    std::vector<i32> prop(static_cast<size_t>(k));  // draft proposals T[1..d]
    std::vector<f32> rows(static_cast<size_t>(k) * static_cast<size_t>(n_vocab_));

    // Invariant at the top of every round (identical to plain decode):
    //   target KV holds [0, pos), logits_host_ predicts position pos
    //   draft  KV holds [0, pos), draft logits predict position pos
    while (res->generated < p.max_tokens && pos < kv_cap_) {
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
            emit_token(first);
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
            stop_hit = emit_token(t);
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
            emit_token(bonus);
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
    if (draft_ && p.sampler.greedy && !model_.is_recurrent()) {
        f64 prefill_ms = 0, decode_ms = 0;
        const bool ok =
            generate_speculative(p, res, ids, &prefill_ms, &decode_ms);
        res->prefill_ms = prefill_ms;
        res->decode_ms = decode_ms;
        return ok;
    }
    if (draft_ && !p.sampler.greedy) {
        KRK_WARN("speculative decoding is greedy-only; sampling runs without "
                 "the draft for this request");
    }

    // ---- prefill ----------------------------------------------------------
    const Timer prefill_timer;
    for (i64 i = 0; i < static_cast<i64>(ids.size()); i += chunk_) {
        const i32 n = static_cast<i32>(
            std::min<i64>(chunk_, static_cast<i64>(ids.size()) - i));
        const bool last = (i + n) == static_cast<i64>(ids.size());
        forward(ids.data() + i, n, static_cast<i32>(i), last);
    }
    fetch_logits();
    res->prefill_ms = prefill_timer.ms();

    Sampler sampler;
    sampler.reset(p.sampler);
    sampler.accept(ids.data(), static_cast<int>(ids.size()));

    if (p.echo_prompt) {
        res->text = p.prompt;
        res->tokens = ids;
        p.sink.emit(p.prompt.c_str(), -1, false);
    }

    i64 pos = static_cast<i64>(ids.size());
    const Timer decode_timer;

    // A stop string may straddle token boundaries, so streamed text is held
    // back by max_stop_len characters until it is known not to be a stop.
    size_t max_stop = 0;
    for (const std::string &s : p.stop) max_stop = std::max(max_stop, s.size());
    std::string pending;

    for (i32 gen = 0; gen < p.max_tokens; gen++) {
        const i32 token = sampler.sample(logits_host_, n_vocab_);
        if (token == tok_.eos()) {
            res->finish = FinishEos;
            break;
        }

        const std::string piece = tok_.decode(&token, 1);
        res->tokens.push_back(token);
        res->text += piece;
        res->generated++;
        sampler.accept(token);

        bool stop_hit = false;
        size_t stop_len = 0;
        for (const std::string &s : p.stop) {
            if (!s.empty() && ends_with(res->text, s) && s.size() > stop_len) {
                stop_hit = true;
                stop_len = s.size();
            }
        }

        pending += piece;
        if (stop_hit) {
            res->text.resize(res->text.size() - stop_len);
            const size_t keep = pending.size() >= stop_len ? pending.size() - stop_len : 0;
            pending.resize(keep);
        }
        if (pending.size() > max_stop) {
            const size_t flush = pending.size() - max_stop;
            p.sink.emit(pending.substr(0, flush).c_str(), token, false);
            pending.erase(0, flush);
        }

        if (stop_hit) {
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
        forward(&token, 1, static_cast<i32>(pos), true);
        pos++;
        fetch_logits();
    }

    if (res->finish != FinishStop && !pending.empty())
        p.sink.emit(pending.c_str(), -1, false);

    res->decode_ms = decode_timer.ms();
    p.sink.emit(nullptr, -1, true);
    return true;
}

} // namespace krk
