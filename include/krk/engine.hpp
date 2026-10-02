// ============================================================================
//  engine.hpp — the decode loop, workspace management and streaming.
// ============================================================================
#ifndef KRK_ENGINE_HPP
#define KRK_ENGINE_HPP

#include <string>
#include <vector>

#include "krk/backend.hpp"
#include "krk/model.hpp"
#include "krk/sampler.hpp"
#include "krk/tokenizer.hpp"

namespace krk {

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
    i32 expert_cache_mb = 0;
    // Hard cap on resident (layer, expert) slots, regardless of budget. 0 = auto.
    i32 expert_cache_slots = 0;
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
    bool has_draft() const { return draft_ != nullptr; }
    const Model &draft_model() const { return draft_->model(); }
    i32 draft_window() const { return draft_tokens_; }

    // ---- KV cache introspection -------------------------------------------
    // Positions [0, kv_pos()) hold keys/values; kv_pos() is also the position
    // the next forward() will write to.
    i64 kv_pos() const { return kv_pos_; }
    i64 kv_capacity() const { return kv_cap_; }
    // Logical rollback to `pos` (must be <= kv_pos()). The occupied region is
    // simply treated as shorter — no zeroing is needed because every forward
    // fully overwrites the rows it touches and attention never reads past
    // kv_pos(). This is the primitive speculative decoding builds on.
    bool kv_rollback(i64 pos);

    // Speculation telemetry (valid after a run with a draft loaded).
    u64 draft_proposed() const { return draft_proposed_; }
    u64 draft_accepted() const { return draft_accepted_; }
    u64 spec_steps() const { return spec_steps_; }

    const Model &model() const { return model_; }
    const Tokenizer &tokenizer() const { return tok_; }
    // Debug: logits from the most recent forward (n_vocab floats).
    const f32 *last_logits() const { return logits_host_; }
    const DeviceCaps &device() const { return be_ ? be_->caps() : no_caps_; }

private:
    // One batched forward over n tokens starting at pos0. LogitMode picks how
    // much of the output head runs: none (mid-prefill), the last row (plain
    // decode), or every row (speculative verification).
    void forward(const i32 *toks, i32 n, i32 pos0, bool want_logits);
    void forward_core(const i32 *toks, i32 n, i32 pos0, LogitMode mode);
    // Runs the output head on one row of ws_x_ into ws_logits_.
    void head_compute(i32 row);
    // Zero-fill rollback for the (rare) shrink-a-lot path; the common small
    // rollback in speculative decoding is pure pointer arithmetic.
    void kv_rollback_zero(i64 pos);
    // Routed-expert FFN for one layer: router -> top-k -> grouped per-expert
    // FFN -> optional shared expert. Tokens that picked the same expert are
    // permuted into one block, so each expert runs three batched GEMMs
    // regardless of how many tokens chose it. Experts are paged in through the
    // model's cache.
    void moe_ffn(const LayerWeights &L, i32 layer, i32 n);
    // Chooses the expert residency budget from cfg_ and the device.
    void configure_expert_cache();
    // Downloads the last row of logits into logits_host_.
    void fetch_logits();
    // Greedy speculative step; false when the draft cannot help.
    bool generate_speculative(const GenerateParams &p, GenerateResult *res,
                              const std::vector<i32> &ids, f64 *prefill_ms,
                              f64 *decode_ms);

    Backend *be_ = nullptr;
    Model model_;
    Tokenizer tok_;
    EngineConfig cfg_;
    DeviceCaps no_caps_;

    // The draft engine owns its own Model and KV cache but shares this
    // backend, so the two models can interleave kernels without copies.
    Engine *draft_ = nullptr;
    i32 draft_tokens_ = 0;
    u64 draft_proposed_ = 0;
    u64 draft_accepted_ = 0;
    u64 spec_steps_ = 0;

    i64 q_dim_ = 0, kv_dim_ = 0, n_embd_ = 0, n_ff_ = 0, n_vocab_ = 0;
    i64 kv_cap_ = 0;
    i64 kv_pos_ = 0;
    i32 chunk_ = 1;

    void *kcache_ = nullptr;
    void *vcache_ = nullptr;
    // workspaces, sized for `chunk_` rows
    void *ws_x_ = nullptr;    // [chunk, n_embd]
    void *ws_xn_ = nullptr;   // [chunk, n_embd]
    void *ws_x2_ = nullptr;   // [chunk, n_embd]
    void *ws_q_ = nullptr;    // [chunk, q_dim]
    void *ws_k_ = nullptr;    // [chunk, kv_dim]
    void *ws_v_ = nullptr;    // [chunk, kv_dim]
    void *ws_attn_ = nullptr; // [chunk, q_dim]
    void *ws_gate_ = nullptr; // [chunk, max(n_ff, n_ff_exp, n_ff_shexp)]
    void *ws_up_ = nullptr;   // [chunk, same]
    void *ws_logits_ = nullptr;
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
    f32 *logits_host_ = nullptr;
    f32 *tok_host_ = nullptr; // pinned staging for token ids
    std::vector<i32> tok_scratch_;
    // Host-side router scratch, sized [chunk, n_expert]
    std::vector<f32> router_host_;
    std::vector<f32> moe_prob_;
    // The routing plan for the current chunk: which tokens each expert got.
    std::vector<i32> moe_sel_;    // [chunk, k] expert id per (token, slot)
    std::vector<f32> moe_wt_;     // [chunk, k] renormalized gate weight
    std::vector<i32> group_rows_; // [<=chunk] token ids in the current group
    std::vector<f32> group_wt_;   // [<=chunk] their gate weights
    std::vector<i32> plan_dev_;   // staging mirror for ws_plan_
    std::vector<f32> alpha_dev_;  // staging mirror for ws_alpha_
};

} // namespace krk

#endif // KRK_ENGINE_HPP
