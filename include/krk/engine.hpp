// ============================================================================
//  engine.hpp — the decode loop, workspace management and streaming.
// ============================================================================
#ifndef KRK_ENGINE_HPP
#define KRK_ENGINE_HPP

#include <string>
#include <utility>
#include <vector>

#include "krk/backend.hpp"
#include "krk/expert_cpu.hpp"
#include "krk/kv_tier.hpp"
#include "krk/model.hpp"
#include "krk/sampler.hpp"
#include "krk/tokenizer.hpp"

namespace krk {

class DflashDraft;

struct EngineConfig {
    std::string model_path;
    i32 n_ctx = 4096;        // KV capacity (tokens)
    i32 prefill_chunk = 256; // prompt tokens per batched forward
    i32 threads = 0;         // CPU backend only
    u64 seed = 0;
    bool warmup = false;     // run a 1-token pass at init to page in weights

    // Residency budget for lazily-loaded MoE expert weights, in MiB. 0 means
    // auto: a share of free VRAM on the GPU, a fixed share on the CPU. Below the
    // cost of one expert the model still runs — it just reloads every step.
    // KV residency (see kv_tier.hpp). HOT is VRAM, WARM is host RAM, COLD is
    // a spill directory. 0 = auto for HOT (whatever free VRAM is left after
    // the weights and workspaces) and for WARM (half of free host RAM); a
    // negative WARM, or an empty COLD dir, disables that tier.
    //
    // When the whole KV fits the budget this is inert: the cache is one flat
    // allocation and every call site is unchanged. Tiering only engages once
    // the KV genuinely does not fit, which is what lets a long context run at
    // all instead of being refused.
    i32 kv_hot_mb = 0;
    i32 kv_warm_mb = 0;
    std::string kv_cold_dir;

    i32 expert_cache_mb = 0;
    // Hard cap on resident (layer, expert) slots, regardless of budget. 0 = auto.
    i32 expert_cache_slots = 0;
    // WARM tier budget in MiB: pageable host RAM that holds the expert set
    // beside the device's hot subset, and the tier every cold read lands in
    // before it is promoted (see expert_cache.cpp). Sized independently of the
    // device budget, because the two answer different questions — VRAM holds
    // what is about to be used, RAM holds what was just used.
    // -1 (the default) sizes it from the machine: the expert set, capped by half
    // of what is free after a reserve, so 24/32/48/96 GiB boxes each get a
    // proportional tier. 0 disables it (HOT only, every miss re-reads the
    // file), and a positive value is that many MiB — the explicit restriction.
    i32 expert_warm_mb = -1;
    // Fill WARM ahead of the run rather than on demand. Off by default: an eager
    // sweep costs the whole corpus in wall time and RAM before the first token,
    // and read-through already admits every expert the run touches.
    bool expert_warm_prefetch = false;
    // Expert warmup: a bounded startup phase that stages the RANKED expert set
    // into WARM and then promotes the top of that order into VRAM, so the first
    // tokens are not paying compulsory file reads and a tier smaller than the
    // routed set holds what routing wants rather than what happened to be
    // touched first. -1 (the default) turns it on when it can help -- a WARM
    // tier exists and the routed set is larger than the device budget, which is
    // exactly the case where the ORDER of the two tiers decides the miss rate.
    // 1 forces it, 0 disables it; KRK_EXPERT_WARMUP=0/1 overrides both, so the
    // A/B is one environment variable away.
    i32 expert_warmup = -1;
    // Wall-time ceiling for that phase, in ms. The default is inside the
    // 15-30 s a cold start can afford; a phase that has not filled the tiers by
    // then stops where it is, having spent the time on the hottest experts
    // first. 0 removes the ceiling (fill the tiers however long it takes).
    i32 expert_warmup_ms = 20000;
    // The RAM class to plan the host tier for, in GiB: 16, 24, 32, 48, 64 or
    // 96, i.e. the machine the run should behave as if it were on. 0 (the
    // default) is the installed RAM. This is the spec's tiering table applied
    // to a class instead of to the box, so a deployment can be measured and
    // tuned on a 96 GiB workstation for a 32 GiB target without pretending the
    // extra RAM away by hand.
    i32 ram_tier_gb = 0;
    // Total device memory this run is allowed to plan for, in MiB. 0 (the
    // default) is the stepped policy in Engine::configure_expert_cache: 6 GiB
    // to begin with, +2 GiB at a time while the card has the headroom and the
    // model actually needs more, never above 14 GiB. A positive value is that
    // many MiB and skips the policy entirely.
    i32 vram_cap_mb = 0;
    // The class form of the same restriction, in GiB: behave as if the card
    // were this big. The twin of ram_tier_gb, and what a plan made for one
    // machine carries to another -- a 12 GiB card and an 8 GiB card want a
    // number that is a property of the CARD, not a MiB figure that has to be
    // recomputed per box. Clamped to the memory actually installed, so the same
    // command line is safe on the big machine. Ignored when vram_cap_mb is set,
    // which is the exact figure.
    i32 vram_tier_gb = 0;
    // Host memory this run is allowed to plan for, in MiB. 0 (the default) is
    // a quarter of installed RAM; a positive value is that many MiB and bounds
    // every host tier inside the process. The mapping and the dense trunk sit
    // outside it, which is what the reserve is for.
    i32 host_ram_mb = 0;

    // ---- hybrid CPU + GPU expert compute --------------------------------
    //
    // In an MoE the expert weights, not the activations, are what has to move:
    // one decode token routes k experts per layer, one activation row is 8 KiB,
    // and one expert's three slices are ~1.95 MiB on a model like laguna. So an
    // expert that is NOT resident in VRAM is far cheaper to compute where its
    // bytes already are -- the mapping -- than to read in and promote. When this
    // is on, an expert that would have been a miss is computed on the host by
    // CpuExpertPool while the device runs the experts that ARE resident, and the
    // two partial sums are merged.
    //
    // It is a decode optimization and it is deliberately bounded to a small row
    // count: the merge is two [n, n_embd] transfers per layer, the host cost
    // scales with the rows an expert carries, and at prefill widths the second
    // of those dominates whatever the split saved. Combined with a small
    // --expert-cache-mb it is also the lever that frees VRAM and host RAM: the
    // experts the host computes need no resident copy in either tier.
    bool hybrid_experts = false;
    // Host worker threads for that split. 0 = min(hardware, 8).
    i32 hybrid_threads = 0;
    // Widest row count the host arm runs at. 0 disables the row bound.
    i32 hybrid_max_rows = 8;
    // Fraction (0..1) of the RESIDENT experts to force onto the host anyway.
    // 1.0 (the default) leaves every resident expert on the device, which is
    // the policy; anything less is the A/B knob that sweeps the CPU/GPU balance
    // at a fixed cache size.
    f32 hybrid_frac = 1.0f;
};

// Hybrid CPU+GPU expert-compute telemetry. The split is only interesting if you
// can see how it landed: how many experts each arm took, how many rows the host
// carried, and the wall time the host arm cost. Read by the end-of-run [stats]
// line and by --profile (which also gets it as the "moe.cpu_experts" host span).
struct HybridStats {
    i32 threads = 0;
    u64 cpu_experts = 0;
    u64 gpu_experts = 0;
    u64 cpu_rows = 0;
    f64 cpu_ms = 0.0;
};

struct StreamSink {
    void (*fn)(void *user, const char *text, i32 token, bool done) = nullptr;
    void *user = nullptr;
    void emit(const char *text, i32 token, bool done) const {
        if (fn) fn(user, text, token, done);
    }
};

struct GenerateParams {
    std::string prompt;
    SampleParams sampler;
    i32 max_tokens = 256;
    std::vector<std::string> stop;
    StreamSink sink;
    bool echo_prompt = false;
    // Debug: when > 0, write the top-k candidates and their log-probs for
    // every sampled token to stderr, at full precision. Two runs that pick
    // different tokens are then distinguishable by eye: identical log-probs
    // up to a step and different ones after it means drift, while the very
    // first step already differing means something raced.
    i32 debug_topk = 0;

    // Force the argmax on the host from the downloaded logits row instead of
    // on the device via logits_topk. Costs a 993 kB download per token at this
    // vocab, and buys determinism: the device argmax path is the one carrying
    // the laguna race (see docs/PERF-ANALYSIS.md section 9), so this is the
    // correct path until that is fixed. --debug-topk reaches the same code but
    // also prints per token, which serialises the run and confounds any test of
    // it -- hence a flag that changes one variable and not the other.
    bool sample_host = false;
};

enum Finish : int {
    FinishLength = 0,
    FinishStop = 1,
    FinishEos = 2,
    FinishContext = 3,
};

enum LogitMode : int {
    LogitsNone = 0, // no vocab projection (mid-prefill chunks)
    LogitsLast = 1, // project the final row only (prefill end, plain decode)
    LogitsAll = 2,  // project every row (speculative verification)
};

struct GenerateResult {
    std::string text;
    std::vector<i32> tokens;
    int finish = FinishLength;
    i32 prompt_tokens = 0;
    i32 generated = 0;
    f64 prefill_ms = 0;
    f64 decode_ms = 0;
    // How many decode steps decode_ms actually covers. It is NOT `generated`:
    // the first generated token comes out of the prefill pass's last-position
    // logits, so the timer starts after prefill and times one step fewer than
    // the number of tokens emitted. Dividing by `generated` understated the
    // per-token cost and printed 2304 tok/s for --max-tokens 1, which is not a
    // speed any machine can produce: one step still reads every weight.
    // Counted rather than inferred, so it stays right if the loop changes.
    i64 decode_steps = 0;
    f64 load_ms = 0;
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    bool init(Backend *be, const EngineConfig &cfg, std::string *err);
    void shutdown();

    bool generate(const GenerateParams &p, GenerateResult *res);

    // ---- draft model (speculative decoding) ------------------------------
    // Greedy speculative decoding: the draft engine proposes up to
    // `spec_decode_tokens` continuations, the target verifies them in ONE
    // batched forward, and the longest matching prefix is kept. Because only
    // argmax branches are pursued, accepted tokens are bit-identical to plain
    // greedy decoding — sampling is rejected with an error instead of changing
    // the distribution.
    bool load_draft(const std::string &path, std::string *err);
    void unload_draft();
    void set_draft_window(i32 n) { draft_tokens_ = n > 0 ? n : 1; }
    bool has_draft() const { return draft_ != nullptr || dflash_ != nullptr; }
    const Model &draft_model() const { return draft_->model(); }
    i32 draft_window() const { return draft_tokens_; }

    // ---- DFlash drafter ---------------------------------------------------
    // `--draft` takes either a full model of the target's family (the path
    // above) or a Poolside DFlash head set, which is not a model at all: it has
    // no embedding and no head, reads the *hidden states* of a few target layers
    // through a fusion encoder, injects their K/V into a cache of its own, and
    // drafts a masked block against it. Which one a path names is read off the
    // file's architecture, so the caller does not have to know.
    bool load_dflash(const std::string &path, std::string *err);
    void unload_dflash();
    bool has_dflash() const { return dflash_ != nullptr; }
    const DflashDraft *dflash() const { return dflash_; }

    // ---- KV cache introspection -------------------------------------------
    // Positions [0, kv_pos()) hold keys/values; kv_pos() is also the position
    // the next forward() will write to.
    i64 kv_pos() const { return kv_pos_; }
    i64 kv_capacity() const { return kv_cap_; }
    // Logical rollback to `pos` (must be <= kv_pos()). The occupied region is
    // simply treated as shorter — no zeroing is needed because every forward
    // fully overwrites the rows it touches and attention never reads past
    // kv_pos(). This is the primitive speculative decoding builds on.
    // A recurrent model is the exception: the delta rule has no inverse, so
    // only a rewind to 0 is honoured (the server's per-request isolation,
    // after which the caller replays the prompt and rebuilds the state).
    bool kv_rollback(i64 pos);

    // Speculation telemetry (valid after a run with a draft loaded).
    u64 draft_proposed() const { return draft_proposed_; }
    u64 draft_accepted() const { return draft_accepted_; }
    u64 spec_steps() const { return spec_steps_; }

    // KV tier telemetry, summed over the K and V planes. The counters existed
    // in KvTierCache but nothing read them after startup, so a run could page
    // the whole cache through WARM and COLD and report nothing about it. This
    // is what the end-of-run [stats] line is built from.
    void kv_tier_stats(KvTierStats *out) const {
        KvTierStats a, b;
        kvt_k_.stats(&a);
        kvt_v_.stats(&b);
        out->tiered = a.tiered || b.tiered;
        out->layers = a.layers;
        out->hot = a.hot;
        out->warm = a.warm;
        out->cold = a.cold;
        out->slots = a.slots;
        // Both planes, so this is the VRAM the HOT tier actually holds and not
        // half of it. layer_bytes stays per-plane: it is a slot's size, and a
        // slot is one plane's copy of one layer.
        out->hot_bytes = a.hot_bytes + b.hot_bytes;
        out->layer_bytes = a.layer_bytes;
        out->promotions_from_warm = a.promotions_from_warm + b.promotions_from_warm;
        out->promotions_from_cold = a.promotions_from_cold + b.promotions_from_cold;
        out->evictions_to_warm = a.evictions_to_warm + b.evictions_to_warm;
        out->evictions_to_cold = a.evictions_to_cold + b.evictions_to_cold;
        // Counted per plane, so this is pages, and the same lost layer counts
        // twice -- once for K and once for V. Reported apart from ->COLD
        // because it is the one number here that says the run is wrong.
        out->evictions_dropped = a.evictions_dropped + b.evictions_dropped;
        out->migrate_bytes = a.migrate_bytes + b.migrate_bytes;
    }

    // Hybrid CPU + GPU expert compute (see EngineConfig::hybrid_experts).
    bool hybrid_enabled() const { return hybrid_on_; }
    void hybrid_stats(HybridStats *out) const {
        out->threads = hybrid_threads_live_;
        out->cpu_experts = hy_cpu_experts_;
        out->gpu_experts = hy_gpu_experts_;
        out->cpu_rows = hy_cpu_rows_;
        out->cpu_ms = hy_cpu_ms_;
    }

    const KvTierCache &kvt_k() const { return kvt_k_; }
    const KvTierCache &kvt_v() const { return kvt_v_; }
    const Model &model() const { return model_; }
    const Tokenizer &tokenizer() const { return tok_; }
    // Debug: logits from the most recent forward (n_vocab floats).
    const f32 *last_logits() const { return logits_host_; }
    const DeviceCaps &device() const { return be_ ? be_->caps() : no_caps_; }

    // ---- routing scan (public surface; see the members for why) ----------
    // Turns the routing scan on, and (optionally) the expert-stubs-only
    // approximation. Must be called before init(): the buffers are sized there.
    void set_expert_scan(bool scan, bool stub) {
        expert_scan_ = scan;
        expert_stub_ = scan && stub;
    }
    // Writes <path> with each expert's routing mass, sorted hottest-first.
    void write_expert_index(const std::string &path) const;

private:
    // One batched forward over n tokens starting at pos0. LogitMode picks how
    // much of the output head runs: none (mid-prefill), the last row (plain
    // decode), or every row (speculative verification).
    void forward(const i32 *toks, i32 n, i32 pos0, bool want_logits);
    void forward_core(const i32 *toks, i32 n, i32 pos0, LogitMode mode);
    // Runs the output head on one row of ws_x_ into ws_logits_.
    void head_compute(i32 row);
    // prism.hadamard fold: dst <- transform(src), see Backend::hadamard_act.
    // `inverse` picks the order used right after a latent embedding lookup,
    // `gdn_perm` adds ssm_out's grouped-V permutation ahead of both.
    void had_xform(void *dst, const void *src, i64 rows, i64 width,
                   bool inverse, bool gdn_perm);

    // KRK_DUMP: append the last row of `buf` (width x rows elements, f32 host
    // or f16 device) to the dump file, one text line per call. A CPU run and a
    // HIP run of the same prompt then differ at exactly one line per stage.
    void dump_row(const char *what, i32 layer, const void *buf, i64 width, i64 rows);
    // Zero-fill rollback for the (rare) shrink-a-lot path; the common small
    // rollback in speculative decoding is pure pointer arithmetic.
    void kv_rollback_zero(i64 pos);
    // Routed-expert FFN for one layer: router -> top-k -> grouped per-expert
    // FFN -> optional shared expert. Tokens that picked the same expert are
    // permuted into one block, so each expert runs three batched GEMMs
    // regardless of how many tokens chose it. Experts are paged in through the
    // model's cache.
    void moe_ffn(const LayerWeights &L, i32 layer, i32 n);
    // The dense (non-MoE) SwiGLU FFN both layer shapes share: gate/up on one
    // fused launch, silu, down, residual. Shared because the recurrent path's
    // copy was the stale one and had drifted off the fused group.
    void dense_ffn(const LayerWeights &L, i32 l, i64 n);
    // dst += x * W^T as ONE launch where the backend can fold the add into the
    // projection's epilogue (Backend::gemm_accumulate), and as the
    // gemm()-into-scratch + add_inplace() pair everywhere else -- including
    // under KRK_DUMP, whose projection stage reads `scratch`.
    void gemm_residual(void *dst, const void *x, const void *w, DType wt,
                       i64 n_out, i64 n_in, i64 rows, void *scratch);
    // Chooses the expert residency budget from cfg_ and the device.
    void configure_expert_cache();
    // Fills the two residency tiers ahead of the run, hottest-first: WARM from
    // the ranked expert list, then the top of that order onto the device. The
    // point is the ORDER, not the volume -- a tier smaller than the routed set
    // holds what routing actually wants instead of whatever the first few
    // tokens happened to touch -- and the wall-time ceiling is what keeps the
    // phase inside a cold start. See EngineConfig::expert_warmup.
    void warm_experts();
    // `<model>.krakenexperts.json` (see write_expert_index) as a hottest-first
    // (layer, expert) list. Empty when the file is missing or does not describe
    // this exact model: a ranking measured on other weights is worse than no
    // ranking, so a stale index is refused rather than trusted.
    std::vector<std::pair<i32, i32>> load_expert_index() const;
    // Whether the ranked warmup should run for this configuration: false for a
    // routing scan, a dense model, or a run with no WARM tier, and otherwise
    // cfg_.expert_warmup with the default (-1) resolved against whether the
    // routed set actually exceeds the device budget. KRK_EXPERT_WARMUP=0/1
    // overrides all of it, which is what makes the A/B one variable wide.
    bool expert_warmup_wanted() const;
    // One gated-delta-net (recurrent) layer: the fused [q|k|v] projection, the
    // causal short conv, the per-head gate/forget, the delta rule, and the
    // gated output projection. Advances the layer's recurrent state.
    void gdn_forward(const LayerWeights &L, i32 l, i32 n);
    // Clears the convolution windows and delta-rule states (f32, one set per
    // recurrent layer). Only correct to call at a sequence boundary.
    void recurrent_reset();
    // Downloads the last row of logits into logits_host_. With device_only true the
    // row stays in VRAM and is reduced to a single greedy argmax by
    // topk_argmax() instead; see fetch_logits() for when that is legal. The
    // flag reads the same way as the caller's, on purpose: an inverted sense
    // here silently disables the fast path with identical output.
    void fetch_logits(bool device_only = false);
    // Device top-k, reduced to the one number greedy decoding needs. Returns
    // false when the caller must use the host row instead: a CPU backend, or a
    // cut so wide the candidate buffer overflowed (a whole shelf of tied
    // logits, which is what a softcapped model produces at the very top).
    bool topk_argmax(i32 *out);
    // Greedy speculative step; false when the draft cannot help.
    bool generate_speculative(const GenerateParams &p, GenerateResult *res,
                              const std::vector<i32> &ids, f64 *prefill_ms,
                              f64 *decode_ms, i64 *decode_steps);
    // The DFlash variant: the drafter is a head set, not an engine, so the
    // proposals come from a masked block instead of a second model's decode
    // loop, and the target's captured activations are what feed it.
    bool generate_speculative_dflash(const GenerateParams &p, GenerateResult *res,
                                     const std::vector<i32> &ids, f64 *prefill_ms,
                                     f64 *decode_ms, i64 *decode_steps);
    // Copies the residual stream entering `layer` (or, for mc.n_layer, the
    // pre-final-norm state) into the DFlash drafter's capture buffer. Inert when
    // no drafter is loaded, which is every run that does not ask for one.
    void dflash_capture(i32 layer, i32 n);

    Backend *be_ = nullptr;
    Model model_;
    Tokenizer tok_;
    EngineConfig cfg_;
    DeviceCaps no_caps_;

    // The draft engine owns its own Model and KV cache but shares this
    // backend, so the two models can interleave kernels without copies.
    Engine *draft_ = nullptr;
    // The DFlash head set, if `--draft` named one. Shares this engine's backend
    // and this engine's activations; owns its own KV cache and workspaces.
    DflashDraft *dflash_ = nullptr;
    i32 draft_tokens_ = 0;
    u64 draft_proposed_ = 0;
    u64 draft_accepted_ = 0;
    u64 spec_steps_ = 0;

    i64 q_dim_ = 0, kv_dim_ = 0, n_embd_ = 0, n_ff_ = 0, n_vocab_ = 0;
    // Row width of the attention query buffer: Qwen3.5 packs a per-head output
    // gate behind the query, so its projection is 2*q_dim wide before the
    // engine splits it apart.
    i64 q_proj_ = 0;
    i64 conv_dim_ = 0, value_dim_ = 0;
    i64 kv_cap_ = 0;
    i64 kv_pos_ = 0;
    i32 chunk_ = 1;

    // HOT/WARM/COLD KV residency. Two planes (K and V), one layer-sized slot
    // each. When untiered both hand back a pointer into one flat allocation
    // and d.layer_stride stays kv_cap * kv_dim, so the call sites below are
    // identical either way.
    KvTierCache kvt_k_;
    KvTierCache kvt_v_;
    bool kv_tiered_ = false;
    // Gated delta net: f32, one block per recurrent layer, persistent across
    // forwards. conv_state_ is the short conv's last (ksize-1) steps; rec_state_
    // is the delta rule's [n_v_head, d_state, hd] matrix. This is the model's
    // memory of the sequence — unlike the KV cache it cannot be recomputed
    // from a position, which is why rollback to 0 resets it.
    void *conv_state_ = nullptr;
    void *rec_state_ = nullptr;
    i32 rec_layers_ = 0;
    i64 conv_state_span_ = 0; // f32 per recurrent layer
    i64 rec_state_span_ = 0;  // f32 per recurrent layer
    // Mamba-2 layout ([head][p][s] vs GDN's [head][s][p] — same bytes, and the
    // scan never mixes them) and the per-layer conv bias, loaded only when a
    // checkpoint actually carries one. ws_m2in_ holds the fused [z|xBC|dt]
    // projection row, which is WIDER than the conv row (10304 vs 6144 on
    // nemotron-30B) and so cannot share ws_qkv_ with the GDN path.
    i64 mamba2_state_span_ = 0; // f32 per nemotron mamba-2 layer
    void *ws_m2in_ = nullptr;
    void mamba2_forward(const LayerWeights &L, i32 l, i32 n);
    // workspaces, sized for `chunk_` rows
    void *ws_x_ = nullptr;    // [chunk, n_embd]
    void *ws_xn_ = nullptr;   // [chunk, n_embd]
    void *ws_x2_ = nullptr;   // [chunk, n_embd]
    void *ws_q_ = nullptr;    // [chunk, q_dim]
    // [chunk, q_proj] the raw query projection: Qwen3.5 packs a per-head
    // output gate behind the query, so its projection is twice as wide as the
    // query it has to be split into. Null when the model does not pack one.
    void *ws_qpack_ = nullptr;
    void *ws_k_ = nullptr;    // [chunk, kv_dim]
    void *ws_v_ = nullptr;    // [chunk, kv_dim]
    void *ws_attn_ = nullptr; // [chunk, q_dim]
    void *ws_gate_ = nullptr; // [chunk, max(n_ff, n_ff_exp, n_ff_shexp)]
    void *ws_up_ = nullptr;   // [chunk, same]
    // prism.hadamard scratch: the transformed copy of a folded matmul's
    // input, widest folded input wide. One buffer, reused at every site --
    // no two of them are live at the same time. Null unless the file folds.
    void *ws_had_ = nullptr;  // [chunk, widest folded input]
    void *ws_logits_ = nullptr;
    // Gated delta net workspaces (null unless the model is recurrent)
    void *ws_qkv_ = nullptr;      // [chunk, conv_dim] fused q|k|v, post-conv
    void *ws_z_ = nullptr;        // [chunk, value_dim] the z gate
    void *ws_h_ = nullptr;        // [chunk, value_dim] delta-rule output
    void *ws_ssm_ = nullptr;      // [chunk, n_v_head] alpha/gate scalars
    void *ws_beta_ = nullptr;     // [chunk, n_v_head] forget gates
    void *ws_agate_ = nullptr;    // [chunk, q_dim] full-attn output gates
    // MoE-only workspaces (null for dense models)
    void *ws_router_ = nullptr; // [chunk, n_expert]
    void *ws_ffn_ = nullptr;    // [chunk, n_embd] MoE output accumulator
    // Batched-expert staging: rows are permuted activations and their
    // intermediate buffers, both [chunk, ...]. Chunked prefill bounds the plan
    // size, so chunk_ slots always suffice. The speculative verification pass
    // uses the same buffers with n = draft window + 1.
    void *ws_xg_ = nullptr;     // [chunk, n_embd]   gathered token rows
    void *ws_gateg_ = nullptr;  // [chunk, n_ff_ws]   gathered gate/intermediate
    void *ws_upg_ = nullptr;    // [chunk, n_ff_ws]   gathered up projection
    void *ws_plan_ = nullptr;   // device i32 row ids for the current expert group
    void *ws_alpha_ = nullptr;  // device f32 gate weights for the current group
    void *ws_scale_ = nullptr;  // device f32 row multipliers from a scaled silu
    // One i32 per token row: the power of two the residual stream is stored
    // divided by for that row (Backend::add_residual's contract), zeroed at the
    // start of every forward pass so it describes this pass's rows only.
    void *ws_rexp_ = nullptr;
    // The MoE staging buffer's own exponent, per token row: the routed experts'
    // sum is accumulated at true scale and a sum of two in-range rows can still
    // leave f16 when the shared expert is folded in, so that fold renormalizes
    // through the same contract and `ws_ffn_` carries an exponent too.
    void *ws_fexp_ = nullptr;
    f32 *logits_host_ = nullptr;
    std::vector<i32> tok_scratch_;
    // Device top-k. The candidate buffer is sized for a degenerate cut; k is
    // one today (an argmax) but the buffers allow more without reallocating.
    static constexpr i32 kTopK = 1;
    static constexpr i32 kTopKCap = 4096;
    void *topk_scratch_ = nullptr; // 4096 B header + kTopKCap u64 keys
    // One 16-byte device landing zone for the whole op: i32 id | f32 value |
    // i64 collected count. Keeping the three fields adjacent is what turns the
    // readback into a single 16-byte transfer instead of three.
    void *topk_out_ = nullptr;
    alignas(8) char topk_out_h_[16] = {};
    i64 topk_n_ = 0;    // the collected count of the last successful call
    i32 topk_id_ = 0;   // its argmax, valid only while topk_valid_
    bool topk_valid_ = false;
    // Host-side router scratch, sized [chunk, n_expert]
    std::vector<f32> router_host_;
    // Scratch for the layer-wide expert prefetch: the routed ids, deduped.
    std::vector<i32> prefetch_ids_;
    std::vector<u8> prefetch_seen_;
    // KRK_DUMP destination (see Engine::init): empty means the per-stage dump
    // hook in forward_core is inert.
    std::string dump_path_;
    std::vector<f32> moe_prob_;
    // The routing plan for the current chunk: which tokens each expert got.
    std::vector<i32> moe_sel_;    // [chunk, k] expert id per (token, slot)
    std::vector<f32> moe_wt_;     // [chunk, k] renormalized gate weight
    std::vector<i32> group_rows_; // [<=chunk] token ids in the current group
    std::vector<f32> group_wt_;   // [<=chunk] their gate weights

    // ---- hybrid CPU + GPU expert compute (EngineConfig::hybrid_experts) --
    //
    // All host-side. hy_xn_ is the activation the host arm reads (downloaded
    // from ws_xn_ exactly once per layer), hy_out_ is what CpuExpertPool
    // accumulates into, and hy_ffn_/hy_f16_ are the merge scratch: the device's
    // ws_ffn_ comes down, the host contribution is added, and the result goes
    // back up as f16. No device buffer is added, so turning this on cannot cost
    // VRAM -- which is the whole point, since the VRAM it frees has to go to the
    // KV cache.
    //
    // hy_job_rows_/hy_job_wt_ are the flattened per-expert row lists a Job
    // points into. They are RESERVED to n * k before anything is written into
    // them: every (token, slot) pair contributes at most one row per expert, so
    // that reserve is an upper bound and no push_back can reallocate a buffer a
    // Job already holds a pointer to.
    bool hybrid_on_ = false;
    i32 hybrid_threads_ = 0;
    i32 hybrid_max_rows_ = 8;
    i32 hybrid_frac_permille_ = 1000;
    i32 hybrid_threads_live_ = 0;
    u64 hy_cpu_experts_ = 0;
    u64 hy_gpu_experts_ = 0;
    u64 hy_cpu_rows_ = 0;
    f64 hy_cpu_ms_ = 0.0;
    std::vector<u8> hy_take_;                 // [n_expert] 1 => host computes it
    std::vector<CpuExpertPool::Job> hy_jobs_; // the host arm's work list
    std::vector<i32> hy_job_rows_;
    std::vector<f32> hy_job_wt_;
    std::vector<f32> hy_xn_;
    std::vector<f32> hy_out_;
    std::vector<f32> hy_ffn_;
    std::vector<u16> hy_f16_;

    // ---- expert routing scan (--expert-scan) ----------------------------
    //
    // A hot-expert list cannot be built from the model file: which experts a
    // token routes to is a function of the residual stream, so it has to be
    // measured. These accumulate what the routers actually chose, weighted by
    // the routing probability rather than by a count -- a count cannot tell you
    // whether the top 8 of 128 carry 90% of the traffic or 40%, and that
    // number is what sizes a residency budget.
    //
    // `expert_stub_` runs the routers with every expert FFN replaced by
    // nothing, which is the cheap approximation: the routers only need the
    // dense path, so the whole expert tensor set stays unmapped. It is only
    // valid if stubbing does not change what the routers choose, which is an
    // assumption with nothing behind it until it is measured -- see
    // tools/expert_scan_diff.py.
    std::vector<double> expert_mass_; // [n_layer * n_expert] summed P(expert)
    std::vector<u64> expert_hits_;     // [n_layer * n_expert] times selected
    i64 expert_scan_tokens_ = 0;       // token positions folded in so far
    bool expert_scan_ = false;         // accumulate
    bool expert_stub_ = false;         // skip the expert GEMMs while scanning
    bool expert_stubbed_ran_ = false;  // stub was requested and did run

    std::vector<i32> plan_dev_;   // staging mirror for ws_plan_
    std::vector<f32> alpha_dev_;  // staging mirror for ws_alpha_
};

} // namespace krk

#endif // KRK_ENGINE_HPP
