// ============================================================================
//  model.hpp — GGUF -> device-resident weight set.
//
//  Covers the dense LLaMA family: llama, mistral, qwen2, qwen3 (QK-norm),
//  smollm, granite, and any arch speaking the same tensor schema — plus the
//  Mixture-of-Experts Qwen family (qwen2moe, qwen3moe) and any arch that emits
//  the standard `ffn_*_exps` expert tensors. Expert weights are NOT uploaded at
//  load time; see expert_cache.hpp. The IQ* formats are still rejected with a
//  message naming the offending tensor instead of being silently mis-decoded.
// ============================================================================
#ifndef KRK_MODEL_HPP
#define KRK_MODEL_HPP

#include "krk/backend.hpp"
#include "krk/expert_cache.hpp"
#include "krk/gguf.hpp"

namespace krk {

// QuantTensor (a resident weight matrix) is declared in expert_cache.hpp, which
// this header includes; the same descriptor covers eager and lazy weights.

struct LayerWeights {
    f32 *attn_norm = nullptr; // f32 [n_embd]
    QuantTensor wq, wk, wv, wo;
    f32 *q_bias = nullptr; // optional [n_head*hd]
    f32 *k_bias = nullptr; // optional [n_kv*hd]
    f32 *v_bias = nullptr; // optional [n_kv*hd]
    f32 *q_norm = nullptr; // optional per-head RMSNorm weight [hd]
    f32 *k_norm = nullptr;
    f32 *ffn_norm = nullptr; // f32 [n_embd]
    QuantTensor wgate, wup, wdown;

    // ---- Mixture-of-Experts (optional) -----------------------------------
    // `moe` selects the routed-expert FFN path. `router` is the small gating
    // matrix [n_expert, n_embd] and is always resident. The routed experts are
    // described by `experts` and materialized lazily by ExpertCache.
    bool moe = false;
    QuantTensor router;                           // ffn_gate_inp.weight
    QuantTensor shexp_gate, shexp_up, shexp_down; // shared expert, if any
    ExpertSource experts;                         // lazily-loaded routed experts
};

struct ModelConfig {
    std::string arch = "llama";
    std::string name;
    i32 n_vocab = 0;
    i32 n_embd = 0;
    i32 n_layer = 0;
    i32 n_head = 0;
    i32 n_head_kv = 0;
    i32 head_dim = 0;
    i32 n_ff = 0;
    i32 n_ctx_train = 0;
    f32 rope_base = 10000.0f;
    f32 rope_scale = 1.0f;      // position scaling for linear RoPE
    f32 rope_attn_scale = 1.0f; // YaRN mscale (1.0 when unused)
    f32 rope_frac = 1.0f;       // fraction of head dims rotated (Phi partial RoPE)
    f32 rms_eps = 1e-5f;
    bool tied_embeddings = false;
    bool has_bias = false;
    bool qk_norm = false;

    // Mixture-of-Experts geometry (is_moe == false for dense models).
    bool is_moe = false;
    i32 n_expert = 0;        // routed experts per MoE layer
    i32 n_expert_used = 0;   // top-k experts activated per token
    i32 n_ff_exp = 0;        // routed-expert hidden width
    i32 n_ff_shexp = 0;      // shared-expert hidden width (0 when absent)
    i32 n_expert_shared = 0; // always-on shared experts per layer
};

class Model {
public:
    Model() = default;
    ~Model() = default;
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    bool load(Backend &be, const std::string &path, std::string *err);
    void unload();

    i64 n_ctx() const { return cfg_.n_ctx_train; }
    i64 kv_dim() const { return static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim; }
    i64 q_dim() const { return static_cast<i64>(cfg_.n_head) * cfg_.head_dim; }
    i64 n_ff() const { return cfg_.n_ff; }
    i64 n_embd() const { return cfg_.n_embd; }
    // Widest FFN a layer may need (dense, routed expert, or shared expert).
    i64 n_ff_ws() const {
        i64 w = cfg_.n_ff;
        if (cfg_.n_ff_exp > w) w = cfg_.n_ff_exp;
        if (cfg_.n_ff_shexp > w) w = cfg_.n_ff_shexp;
        return w;
    }

    const ModelConfig &cfg() const { return cfg_; }
    const std::vector<LayerWeights> &layers() const { return layers_; }
    const QuantTensor &tok_embd() const { return tok_embd_; }
    const QuantTensor &out_head() const { return out_head_; }
    const f32 *out_norm() const { return out_norm_; }
    // NeoX inverse frequencies, length head_dim/2, host-resident (tiny).
    const std::vector<f32> &inv_freq() const { return inv_freq_; }
    i64 weight_bytes() const { return weight_bytes_; }
    const Gguf &gguf() const { return gguf_; }

    // Lazily-loaded routed experts. Override the residency budget after load to
    // trade reload traffic against memory.
    ExpertCache &experts() { return experts_; }
    const ExpertCache &experts() const { return experts_; }
    void set_expert_budget(size_t bytes) { experts_.configure(be_, bytes); }
    // Largest single-expert footprint across the MoE layers (0 when dense).
    size_t max_expert_bytes() const {
        size_t m = 0;
        for (const LayerWeights &L : layers_)
            if (L.moe && L.experts.expert_bytes() > m) m = L.experts.expert_bytes();
        return m;
    }

private:
    Backend *be_ = nullptr;
    Gguf gguf_;
    ModelConfig cfg_;
    std::vector<LayerWeights> layers_;
    QuantTensor tok_embd_, out_head_;
    f32 *out_norm_ = nullptr;
    std::vector<f32> inv_freq_;
    i64 weight_bytes_ = 0;
    ExpertCache experts_;
};

} // namespace krk

#endif // KRK_MODEL_HPP
