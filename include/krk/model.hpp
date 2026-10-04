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
    // Per-expert selection bias (laguna), added to the router's probabilities
    // before the top-k and deliberately not to the weights.
    f32 *router_bias = nullptr;                   // ffn_exp_probs_b.bias [n_expert]
    // Per-head attention output gate: one scalar per query head, off the
    // pre-attention hidden state.
    QuantTensor wattn_gate;                       // attn_gate.weight [n_embd, n_head]
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

    // ---- hybrid full / sliding-window attention (laguna) ------------------
    // Some archs alternate layer kinds. Every layer whose index is not a
    // multiple of `swa_period` attends only the last `swa_window` positions,
    // and carries its own rotary table: laguna-S is 48 layers of period 4
    // (full at il % 4 == 0), window 512, full layers rotating 64 of 128 dims
    // under YaRN and the windowed ones rotating all 128 under plain RoPE.
    // `swa_period` 0 or 1 means every layer attends fully.
    i32 swa_period = 0;
    i32 swa_window = 0;
    i32 rope_dim_swa = 0;
    f32 rope_frac_swa = 1.0f;
    f32 rope_base_swa = 10000.0f;
    // `n_head` above is the head count of the *full* layers. When a model
    // varies it per layer (`attention.head_count` stored as an array) the whole
    // vector is kept here, and `n_head` is its maximum — which is what the
    // workspaces size to, since the query projection is the widest one.
    std::vector<i32> n_head_layer;
    // Per-head softplus output gate (laguna `attn_gate.weight`): one scalar per
    // query head, projected from the *same* hidden state q/k/v read, then
    // softplus'd and multiplied into the attention result before wo. Distinct
    // from the packed query gate of Qwen3.5, which is a sigmoid off the query
    // projection rather than a separate tensor.
    bool attn_gate = false;
    // RoPE pair convention, read off the arch table (ArchSpec::rope_neox): true
    // pairs channel j with j + rope_dim/2 (NeoX half-split), false pairs the
    // adjacent channels (2i, 2i+1). It travels with the model because it is a
    // property of how the file was exported, not of the kernel, and both the
    // prefill and the decode path have to agree on it.
    bool rope_neox = false;
    // Router shape. `router_bias` is a per-expert selection bias
    // (`ffn_exp_probs_b.bias`) added to the *probs* before the top-k but not to
    // the weights the experts are combined with; `router_sigmoid` selects
    // sigmoid over softmax; the selected weights are then optionally
    // sum-normalized (`expert_w_norm`) and scaled (`expert_w_scale`).
    bool router_bias = false;
    bool router_sigmoid = false;
    // Whether the rope table was built with YaRN's frequency correction, i.e.
    // whether `rope_scale` has already been folded into `inv_freq` and must be
    // left at 1.0 for the rotation op.
    bool rope_yarn = false;
    // 1 = sum-normalize the selected expert weights, 0 = keep the raw
    // probabilities, -1 = the file does not declare the key. The engine has
    // always normalized, and every model in the collection that *does* declare
    // it declares 1, so the absent value has to keep the legacy behaviour
    // rather than silently switching every Qwen MoE to unnormalized weights.
    i32 expert_w_norm = -1;
    f32 expert_w_scale = 1.0f;
    // Leading layers whose FFN is dense rather than MoE
    // (`leading_dense_block_count`).
    i32 n_dense_lead = 0;

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
    // Widest query projection any layer asks for. A model with a per-layer head
    // count (laguna) is smaller on some layers than others, so this is what a
    // shared workspace has to be sized against; the width of one particular
    // layer is q_dim_at().
    i64 q_dim() const { return static_cast<i64>(cfg_.n_head) * cfg_.head_dim; }

    // ---- per-layer geometry (laguna alternates two layer kinds) ----------
    i32 n_head_at(i32 layer) const {
        if (cfg_.n_head_layer.empty()) return cfg_.n_head;
        if (layer < 0 || layer >= static_cast<i32>(cfg_.n_head_layer.size()))
            return cfg_.n_head;
        return cfg_.n_head_layer[static_cast<size_t>(layer)];
    }
    i64 q_dim_at(i32 layer) const {
        return static_cast<i64>(n_head_at(layer)) * cfg_.head_dim;
    }
    // True when this layer attends a bounded window instead of the whole past.
    bool is_swa(i32 layer) const {
        return cfg_.swa_period > 1 && cfg_.swa_window > 0 &&
               (layer % cfg_.swa_period) != 0;
    }
    // Rotary table this layer rotates with: the windowed layers of a hybrid
    // model carry their own (plain RoPE over the full head, where the full
    // layers run YaRN over a prefix). Falls back to the single table when the
    // model has no per-layer-kind rotary.
    const std::vector<f32> &inv_freq_at(i32 layer) const {
        return is_swa(layer) && !inv_freq_swa_.empty() ? inv_freq_swa_ : inv_freq_;
    }
    f32 rope_frac_at(i32 layer) const {
        return is_swa(layer) ? cfg_.rope_frac_swa : cfg_.rope_frac;
    }
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
    // What the load phase copied to the device and how long it took. The
    // per-size breakdown stays behind KRK_PHASE; this is the total the run
    // report needs on every run.
    size_t upload_bytes() const { return upload_bytes_; }
    f64 upload_ms() const { return upload_ms_; }

    size_t max_expert_bytes() const {
        size_t m = 0;
        for (const LayerWeights &L : layers_)
            if (L.moe && L.experts.expert_bytes() > m) m = L.experts.expert_bytes();
        return m;
    }
    // Bytes needed to keep the *whole routed set* resident (0 when dense):
    // every expert of every MoE layer, not one expert per layer.
    //
    // It summed only expert_bytes() per layer, which is the footprint of a
    // single expert and understated the answer by exactly n_expert. The
    // budget is then compared against this in configure_expert_cache, so a
    // 4-expert layer got a budget for 1 expert's worth of the whole model:
    // 152.6 MiB where ~610 MiB was needed, ~28 of 112 expert rows resident
    // against 56 accessed per token, and a 0.0% hit rate — every expert
    // re-fetched from host memory on every step. Qwen3-MoE-4x0.6B-2.4B ran at
    // 16.0 tok/s; with the correct total the budget covers the whole set and
    // it runs at 77.5 tok/s (4.8x), with 4096 MiB buying nothing further.
    size_t total_expert_bytes() const {
        size_t t = 0;
        for (const LayerWeights &L : layers_) {
            if (!L.moe) continue;
            const i64 ne = L.experts.n_expert > 0 ? L.experts.n_expert : 1;
            t += L.experts.expert_bytes() * static_cast<size_t>(ne);
        }
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
    std::vector<f32> inv_freq_swa_;
    i64 weight_bytes_ = 0;
    ExpertCache experts_;
    size_t upload_bytes_ = 0;
    f64 upload_ms_ = 0;
};

} // namespace krk

#endif // KRK_MODEL_HPP
