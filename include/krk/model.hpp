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
    QuantTensor shexp_inp_gate;                   // [n_embd] per-token gate
    ExpertSource experts;                         // lazily-loaded routed experts

    // ---- gated delta net (Qwen3.5 / Qwen3-Next, optional) ---------------
    // A recurrent layer replaces attention with a short depthwise convolution
    // over the projected [q|k|v] plus a delta-rule state update, so its
    // geometry is unrelated to the head counts. A full-attention layer in the
    // same stack is ordinary attention whose query projection additionally
    // packs a per-head output gate (unpacked by the engine before attention).
    bool gdn = false;
    QuantTensor wqkv;        // [n_embd, conv_dim]  -> [q | k | v]
    QuantTensor wqkv_gate;   // [n_embd, value_dim]  the z gate
    QuantTensor ssm_conv1d;  // [ksize, conv_dim]    depthwise
    QuantTensor ssm_out;     // [value_dim, n_embd]
    f32 *ssm_dt = nullptr;   // [n_v_head]  timestep bias (no .weight suffix)
    f32 *ssm_a = nullptr;    // [n_v_head]  stored as -exp(A_log)
    // Projected per head, so these are matmul weights like any other (the
    // loader downcasts an F32 checkpoint tensor to F16).
    QuantTensor ssm_alpha;  // [n_embd, n_v_head]
    QuantTensor ssm_beta;   // [n_embd, n_v_head]
    // Plain RMSNorm weight over d_state. The converter adds 1 to every norm in
    // the model EXCEPT this one, so it is uploaded verbatim, not via the
    // zero-centred path.
    f32 *ssm_norm = nullptr;
};

// The final-logit bound some archs declare (`final_logit_softcapping`). A free
// function so the formula has one home and can be checked on its own; the engine
// applies it in the one place logits become host-visible.
inline f32 apply_logit_softcap(f32 x, f32 cap) {
    return cap > 0.0f ? cap * std::tanh(x / cap) : x;
}

struct ModelConfig {
    std::string arch = "llama";
    std::string name;
    i32 n_vocab = 0;
    i32 n_embd = 0;
    i32 n_layer = 0;
    // Next-token-prediction (MTP) blocks present in the file beyond n_layer.
    // They are prediction heads, not transformer layers, and are not loaded.
    i32 n_layer_nextn = 0;
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
    // Gemma 2/3/4 bound the output logits with cap * tanh(logit / cap). The
    // transform is monotone, so argmax and every sampler that only compares
    // probabilities are unaffected by *ranking* — it matters for the value of
    // the log-probs a sampler sees and for anything that mixes logits from
    // different sources. 0 disables it. See apply_logit_softcap.
    f32 logit_softcap = 0.0f;
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

    // ---- gated delta net (Qwen3.5 / Qwen3-Next) --------------------------
    // `recurrent` is true when at least one layer is a delta-net layer, which
    // is what makes the engine allocate and carry the recurrent state.
    bool recurrent = false;
    i32 full_attention_interval = 4; // every Nth layer keeps real attention
    i32 ssm_d_state = 128;           // state width == key/value head dim
    i32 ssm_dt_rank = 32;            // value heads
    i32 ssm_n_group = 16;            // key heads
    i32 ssm_d_conv = 4;              // short-convolution kernel taps
    i32 ssm_key_dim = 0;             // ssm_n_group * ssm_d_state
    i32 ssm_value_dim = 0;           // ssm_dt_rank * ssm_d_state
    i32 ssm_conv_dim = 0;            // ssm_key_dim*2 + ssm_value_dim
    // Rotated width when it is narrower than head_dim (partial RoPE). 0 means
    // the whole head is rotated.
    i32 rope_dim = 0;
    // Interleaved multi-rotary section sizes over the rotated pairs. Recorded
    // for completeness; with a single text stream every position is equal, so
    // the sections collapse to one uniform rotation (see the engine's rope
    // call). 0 when the model does not use them.
    i32 rope_sections[4] = {0, 0, 0, 0};
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

    // ---- gated delta net geometry ---------------------------------------
    bool is_recurrent() const { return cfg_.recurrent; }
    i64 conv_dim() const { return cfg_.ssm_conv_dim; }
    i64 value_dim() const { return cfg_.ssm_value_dim; }
    i64 key_dim() const { return cfg_.ssm_key_dim; }
    // Number of delta-net layers, i.e. how many recurrent states the engine
    // has to carry.
    i32 recurrent_layers() const {
        i32 n = 0;
        for (const LayerWeights &L : layers_)
            if (L.gdn) n++;
        return n;
    }
    // Index of a delta-net layer within the recurrent-state arrays (layers are
    // numbered for the KV cache but the states are only allocated for the
    // recurrent ones).
    i32 recurrent_index(i32 layer) const {
        i32 n = 0;
        for (i32 i = 0; i < layer && i < static_cast<i32>(layers_.size()); i++)
            if (layers_[static_cast<size_t>(i)].gdn) n++;
        return n;
    }
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
    // host_bytes sizes the cache's second tier in page-locked host memory; 0
    // disables it and evicted experts go back to the GGUF mapping.
    void set_expert_budget(size_t bytes, size_t host_bytes = 0) {
        experts_.configure(be_, bytes, host_bytes);
    }
    // Largest single-expert footprint across the MoE layers (0 when dense).
    size_t max_expert_bytes() const {
        size_t m = 0;
        for (const LayerWeights &L : layers_)
            if (L.moe && L.experts.expert_bytes() > m) m = L.experts.expert_bytes();
        return m;
    }
    // Sum of one expert slice (gate+up+down) across every MoE layer:
    // the bytes needed to keep the whole routed set resident (0 when dense).
    size_t total_expert_bytes() const {
        size_t t = 0;
        for (const LayerWeights &L : layers_)
            if (L.moe) t += L.experts.expert_bytes();
        return t;
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
