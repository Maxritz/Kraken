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

#include <cmath> // std::tanh in apply_logit_softcap, called from every path
#include <utility> // std::pair in the promoted-layer table

namespace krk {

// QuantTensor (a resident weight matrix) is declared in expert_cache.hpp, which
// this header includes; the same descriptor covers eager and lazy weights.

// The prism.hadamard activation transform (PrismML's llama.cpp fork, the
// Ternary-Bonsai-2 exports). A few matmul weights are stored with their
// INPUT dimension rotated: multiplied by a per-width +-1 sign vector and
// then by a normalized Sylvester Walsh-Hadamard over consecutive blocks of
// `block` elements. Nothing has to be done to the weights -- the engine
// transforms the matching ACTIVATION before each such matmul (and, for the
// one table stored rotated, once after the embedding lookup).
//
// Inactive for every file that does not carry the keys, so the plan costs a
// bool test per site and nothing else.
struct HadamardPlan {
    bool active = false;
    i32 block = 1024;           // FWHT block size, a power of two
    bool embd_inverse = false;  // token_embd.weight stores rotated rows
    bool head_folded = false;   // output.weight's input is transformed
    bool gdn_v_grouped = false; // ssm_out's input carries the grouped-V perm
    i64 max_width = 0;          // widest folded input; sizes the scratch
};

struct LayerWeights {
    f32 *attn_norm = nullptr; // f32 [n_embd]
    QuantTensor wq, wk, wv, wo;
    f32 *q_bias = nullptr; // optional [n_head*hd]
    f32 *k_bias = nullptr; // optional [n_kv*hd]
    f32 *v_bias = nullptr; // optional [n_kv*hd]
    f32 *q_norm = nullptr; // optional per-head RMSNorm weight [hd]
    f32 *k_norm = nullptr;
    // Attention sink logit per query head [n_head], nullable like every other
    // optional 1-D weight: a file without `blk.N.attn_sinks.weight` is the
    // normal case and keeps the arithmetic it had (see AttnDesc::sinks).
    f32 *sinks = nullptr;
    // Attention-output bias (`attn_output.bias`, gpt-oss): one f32 per output
    // channel, added to the wo projection's result before the residual. DEVICE
    // memory -- the GEMM that produces the result runs there, and the add is
    // one launch with the same shape as `add_bias_cols`.
    f32 *wo_bias = nullptr;
    f32 *ffn_norm = nullptr; // f32 [n_embd]
    QuantTensor wgate, wup, wdown;

    // ---- qwen4exp: hyper-connections, PLE, indexer (optional) ------------
    // `hc_*` are the two mixers per layer (before the token mixer and before the
    // FFN). `hc_attn_norm` is [n_embd, hc] read lane-major; the down/up pair is
    // the [hc*n_embd -> low_rank -> hc*n_embd] bottleneck and `inject` maps the
    // wide stream to one weight per lane. `ple_*` is the n-gram module on the
    // layers `ple.layers` names, and `idx_*` is the QSA indexer on a full
    // attention layer.
    f32 *hc_attn_norm = nullptr;
    QuantTensor hc_attn_down, hc_attn_up, hc_attn_inject;
    f32 *hc_ffn_norm = nullptr;
    QuantTensor hc_ffn_down, hc_ffn_up, hc_ffn_inject;
    bool ple = false;
    QuantTensor ple_key, ple_value, ple_conv1d;
    f32 *ple_norm_key = nullptr;
    f32 *ple_norm_query = nullptr;
    f32 *ple_norm_conv = nullptr;
    i32 qsa_ratio = 0; // > 0 when this layer carries the QSA indexer
    QuantTensor idx_q_proj, idx_k_proj;
    f32 *idx_q_norm = nullptr;
    f32 *idx_k_norm = nullptr;
    // HOST copies of the two indexer norms. The selection runs on the host --
    // the pooled keys, the scores and the top-k are all host work, because the
    // block scores have to be turned into a per-row cell list before any
    // kernel can use them -- so these are read where the selection reads them
    // rather than downloaded from the device copy once per forward.
    std::vector<f32> idx_q_norm_h, idx_k_norm_h;

    // ---- Mixture-of-Experts (optional) -----------------------------------
    // `moe` selects the routed-expert FFN path. `router` is the small gating
    // matrix [n_expert, n_embd] and is always resident. The routed experts are
    // described by `experts` and materialized lazily by ExpertCache.
    bool moe = false;
    QuantTensor router;                           // ffn_gate_inp.weight
    // Per-expert selection bias (laguna), added to the router's probabilities
    // before the top-k and deliberately not to the weights.
    //
    // HOST memory, and that is load-bearing rather than incidental: the top-k it
    // feeds runs on the host, on the router row this engine already downloads.
    // It was declared next to the QuantTensors and filled with upload_f32(),
    // which returns a DEVICE pointer, so `rb[e]` in the selection score read
    // VRAM from the CPU and took the process down. The bias was never loaded --
    // the tensor name did not match -- so the path had never run and the bug sat
    // in it unnoticed. n_expert floats is 1 KiB a layer; there is no reason for
    // it to be on the device at all.
    f32 *router_bias = nullptr;  // HOST f32[n_expert], owned; exp_probs_b.bias
    // Per-expert FFN biases (`ffn_gate_exps.bias` / `ffn_up_exps.bias` /
    // `ffn_down_exps.bias`, gpt-oss): whole tables [n_expert, n_ff] (or
    // [n_expert, n_embd] for down), DEVICE memory, indexed by expert rather
    // than by token. They are applied after each projection and before the
    // activation, which is where the reference puts them -- so a layer that
    // carries them cannot take the fused gate/silu pair, whose output is
    // already post-activation.
    f32 *exp_gate_bias = nullptr;  // [n_expert * n_ff_exp]
    f32 *exp_up_bias = nullptr;    // [n_expert * n_ff_exp]
    f32 *exp_down_bias = nullptr;  // [n_expert * n_embd]
    // Per-head attention output gate: one scalar per query head, off the
    // pre-attention hidden state.
    QuantTensor wattn_gate;                       // attn_gate.weight [n_embd, n_head]
    QuantTensor shexp_gate, shexp_up, shexp_down; // shared expert, if any
    // The shared expert's NVFP4 global scales (see ExpertSource). 1.0 when the
    // file carries no sidecar, which is every format that is not NVFP4.
    f32 shexp_gate_scale = 1.0f, shexp_up_scale = 1.0f, shexp_down_scale = 1.0f;
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
    // The mamba-2 checkpoint carries a conv bias; the gated delta net's does
    // not, so this is null there and the op adds nothing. HOST-loaded f32.
    f32 *ssm_conv_bias = nullptr; // [conv_dim]
    QuantTensor ssm_out;     // [value_dim, n_embd]
    f32 *ssm_dt = nullptr;   // [n_v_head]  timestep bias (no .weight suffix)
    f32 *ssm_a = nullptr;    // [n_v_head]  stored as -exp(A_log)
    // ---- Nemotron Mamba-2 SSM layer (optional) ---------------------------
    // nemotron_h_moe uses Mamba-2 blocks which have different tensors than GDN:
    // ssm_in (input projection), ssm_d (optional D parameter), no ssm_alpha/beta.
    bool nemotron_ssm = false;  // This layer is a Mamba-2 SSM block (not GDN)
    QuantTensor ssm_in;         // [inner_size, n_embd] - Mamba-2 input projection
    f32 *ssm_d = nullptr;       // [dt_rank] - Mamba-2 D parameter (optional)
    // Projected per head, so these are matmul weights like any other (the
    // loader downcasts an F32 checkpoint tensor to F16).
    QuantTensor ssm_alpha;  // [n_embd, n_v_head]
    QuantTensor ssm_beta;   // [n_embd, n_v_head]
    // Plain RMSNorm weight over d_state. The converter adds 1 to every norm in
    // the model EXCEPT this one, so it is uploaded verbatim, not via the
    // zero-centred path.
    f32 *ssm_norm = nullptr;
    // ---- prism.hadamard folded inputs (false on every other model) -------
    // Which of this layer's matmuls read a ROTATED weight, so their input has
    // to be transformed first (Backend::hadamard_act, plan above). They are
    // per site rather than one flag because the same buffer feeds weights that
    // are NOT folded: the attention-norm output also feeds ssm_alpha/ssm_beta
    // and a MoE router untouched, so the transformed copy goes to a scratch
    // buffer and only the folded matmul reads it.
    bool h_attn_in = false;   // attn-norm out -> attn_q / attn_k / attn_v /
                              //                 attn_qkv
    bool h_attn_gate = false; // attn-norm out -> attn_gate (Qwen3.5's packed
                              //                 gate, laguna's separate one)
    bool h_attn_out = false;  // attention result -> attn_output
    bool h_ffn_in = false;    // ffn-norm out -> ffn_gate / ffn_up
    bool h_ffn_down = false;  // silu(gate)*up -> ffn_down
    bool h_ssm_out = false;   // delta-rule out -> ssm_out (grouped-V perm)
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
    // OLMoE and its relatives normalize Q/K *before* the head split: one RMS
    // over the whole projected [n_embd] row, weight width n_embd. Qwen3 and
    // Gemma3 normalize per head, weight width head_dim. The two are not
    // interchangeable and the difference is silent -- both backends share this
    // one op, so reading a 2048-wide weight as 16 heads of 128 throws away
    // 15/16 of it and yields a fluent-but-wrong model on *every* path. The
    // stored tensor width is what tells them apart (see model.cpp).
    bool qk_norm_wide = false;

    // ---- hybrid full / sliding-window attention (laguna) ------------------
    // Some archs alternate layer kinds. Every layer whose index is not a
    // multiple of `swa_period` attends only the last `swa_window` positions,
    // and carries its own rotary table: laguna-S is 48 layers of period 4
    // (full at il % 4 == 0), window 512, full layers rotating 64 of 128 dims
    // under YaRN and the windowed ones rotating all 128 under plain RoPE.
    // `swa_period` 0 or 1 means every layer attends fully.
    i32 swa_period = 0;
    // Phase of the pattern: is_swa(l) == ((l + swa_phase) % swa_period) != 0.
    // 0 is the historical (full layer first) reading; 1 flips the pair so the
    // *windowed* layer comes first (gpt-oss).
    i32 swa_phase = 0;
    i32 swa_window = 0;
    i32 rope_dim_swa = 0;
    f32 rope_frac_swa = 1.0f;
    // Defaults to the model's own `rope.freq_base`, not to 10000: a file that
    // declares one rotary base for the whole stack means it for every layer,
    // and a windowed layer silently rotated at 10000 while its full layers used
    // 150000 is a wrong answer that no shape check catches (gpt-oss declares
    // exactly one base, for all 36 layers).
    f32 rope_base_swa = 10000.0f;
    // `n_head` above is the head count of the *full* layers. When a model
    // varies it per layer (`attention.head_count` stored as an array) the whole
    // vector is kept here, and `n_head` is its maximum — which is what the
    // workspaces size to, since the query projection is the widest one.
    std::vector<i32> n_head_layer;
    // Same idea for the KV side: a per-layer array (nemotron_h_moe: 2 on its
    // six attention layers, 0 on the 46 that have none, gemma4: {1,8}) with
    // n_head_kv holding the maximum, which is what a shared workspace and one
    // KV slot are measured against.
    std::vector<i32> n_head_kv_layer;
    // Per-head softplus output gate (laguna `attn_gate.weight`): one scalar per
    // query head, projected from the *same* hidden state q/k/v read, then
    // softplus'd and multiplied into the attention result before wo. Distinct
    // from the packed query gate of Qwen3.5, which is a sigmoid off the query
    // projection rather than a separate tensor.
    bool attn_gate = false;
    // ---- qwen4exp (Qwen3.8-Flash-Next) -----------------------------------
    // The residual is `hc` lanes of n_embd, mixed through a low-rank bottleneck
    // (4 x 2560 through 320 on the REAP-288 file). `ple` is the n-gram hash
    // embedding table, `qsa_ratio` the per-layer block size of the QSA indexer
    // (0 on a layer without one), and `no_out_norm` says the final mixer
    // carries the norm -- there is no output_norm.weight in the file.
    bool hc = false;
    i32 hc_n = 0;
    i32 hc_low_rank = 0;
    bool ple = false;
    i32 ple_ngram = 0;
    i32 ple_heads_per_ngram = 0;
    i32 ple_head_dim = 0;
    i32 ple_conv_kernel = 0;
    i32 ple_eos_token = -1;
    std::vector<i64> ple_multipliers;   // [ngram]
    std::vector<i64> ple_head_offsets;  // [n_heads]
    std::vector<i64> ple_head_vocab_sizes;
    std::vector<i32> ple_layers;        // layer indices carrying a PLE module
    bool qsa = false;
    i32 idx_n_head = 0;
    i32 idx_head_dim = 0;
    i32 idx_top_k = 0;
    std::vector<i32> qsa_ratio;         // per layer, 0 = no indexer
    bool no_out_norm = false;
    // The recurrent layer's output gate is `sigmoid(z)` rather than `silu(z)`.
    // qwen4exp's build_norm_gated is the one numerical difference from Qwen3.5's
    // gated delta net.
    bool gdn_gate_sigmoid = false;
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
    // Where the bias enters. laguna's `exp_probs_b.bias` is added to the
    // *probabilities* (`false`); llama.cpp's `ffn_gate_inp.bias` convention
    // (gpt-oss) adds it to the router LOGITS, before the softmax -- so it
    // changes both the top-k and the weights, and the two are not
    // interchangeable: a monotone map does not commute with a bias added after
    // it, so the same tensor applied in the wrong place selects different
    // experts. Absent means the laguna reading, which is what this engine ran
    // before the flag existed.
    bool router_bias_pre = false;
    bool router_sigmoid = false;
    // The FFN's activation is squared ReLU rather than a gated SiLU
    // (nemotron_h/nemotron_h_moe: the reference builds every FFN of the family
    // with LLM_FFN_RELU_SQR, and its files carry no ffn_gate for the gate-less
    // expert/shared-expert shapes to multiply against).
    bool ffn_relu_sqr = false;
    // gpt-oss's expert activation (SwiGLU-OAI): min(gate, 7) *
    // sigmoid(1.702 * gate) * (clamp(up, -7, 7) + 1). A different function from
    // silu(gate) * up, not a different spelling of it -- see the note on
    // Backend::swiglu_oai.
    bool ffn_swiglu_oai = false;
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
    // ---- nemotron_h_moe (Mamba-2 + MoE) -----------------------------------
    // nemotron_h_moe uses Mamba-2 blocks (different from GDN) alternating with
    // MoE expert layers. Even layers are Mamba-2 SSM, odd layers are experts.
    bool nemotron_moe = false;        // This is a nemotron_h_moe model
    bool has_d_param = false;         // ssm_d parameter present (Mamba-2 specific)
    i32 ssm_inner_size = 0;           // Mamba-2 inner projection dimension
    // Rotated width when it is narrower than head_dim (partial RoPE). 0 means
    // the whole head is rotated.
    i32 rope_dim = 0;
    // Interleaved multi-rotary section sizes over the rotated pairs. Recorded
    // for completeness; with a single text stream every position is equal, so
    // the sections collapse to one uniform rotation (see the engine's rope
    // call). 0 when the model does not use them.
    i32 rope_sections[4] = {0, 0, 0, 0};
};

// A model's next-token-prediction (MTP/nextn) block: the extra block a
// Qwen3.5-family file appends after the stack, which the engine does NOT run as
// a layer. It is a full attention + MoE block plus three adapters that fuse the
// target's hidden state with the last token's embedding:
//
//   x = eh_proj( cat( hnorm(hidden), enorm(embed(t)) ) )
//   x = attn(x) -> moe(x) -> shared_head_norm(x) -> the model's own output head
//
// Loaded only when asked for (KRK_MTP_LOAD=1, or a future --mtp-draft): the
// experts alone are ~855 MB on Tiel-35B-A3B, so a plain run must not pay for
// them. Nothing in the main stack reads this struct.
struct MtpBlock {
    bool present = false;
    i32 layer = -1;      // the file's block index (blk.<layer>.*)
    i32 n_embd = 0;
    i32 n_head = 0;
    i32 n_head_kv = 0;
    i32 head_dim = 0;
    i32 n_expert = 0;
    i32 n_expert_used = 0;
    i32 n_ff_exp = 0;
    i32 n_ff_shexp = 0;
    i32 n_ff = 0;        // dense FFN width (0 when the block is MoE-only)
    size_t bytes = 0;    // weights uploaded, for the report
    // The block itself.
    QuantTensor wq, wk, wv, wo;
    QuantTensor router;
    QuantTensor shexp_gate, shexp_up, shexp_down;
    // Qwen3.5 scales the shared expert by a sigmoid of a per-token projection
    // (`ffn_gate_inp_shexp`), the same gate the stack's own layers carry.
    QuantTensor shexp_inp_gate;
    // A block whose FFN is dense carries ffn_gate/up/down instead of routed
    // banks. Both shapes load, and the forward runs the one the file has: the
    // qwen35 dense family's nextn heads are this shape, and running attention
    // over a block whose FFN was never loaded is a fluent wrong answer.
    QuantTensor ffn_gate, ffn_up, ffn_down;
    f32 *attn_norm = nullptr;
    f32 *post_attn_norm = nullptr;
    f32 *q_norm = nullptr;
    f32 *k_norm = nullptr;
    f32 *router_bias = nullptr; // HOST, like LayerWeights::router_bias
    // The nextn adapters and the shared head's norm.
    QuantTensor eh_proj;        // [2*n_embd, n_embd]
    f32 *enorm = nullptr;
    f32 *hnorm = nullptr;
    f32 *shared_head_norm = nullptr;
    // Routed experts are NOT uploaded here: they go through ExpertCache like
    // any other layer's, once the drafter runs (see the mtp plan in docs).
    const GgufTensor *exps_gate = nullptr;
    const GgufTensor *exps_up = nullptr;
    const GgufTensor *exps_down = nullptr;
};

// One dense tensor that is NOT resident: its source in the GGUF mapping and
// the address inside the streaming ring it is staged at. A streamed layer's
// QuantTensors point into that ring, so layer decisions (present(), dtypes,
// shapes) are unchanged and only the bytes move.
struct StreamEntry {
    const GgufTensor *src = nullptr;
    void *dst = nullptr;
    size_t bytes = 0;
    // The layer tensor that points at `dst`, bound after the layer is built.
    // Promotion re-points it into a permanent allocation, so this is what
    // moves the layer's own weights rather than the ring's copy of them.
    QuantTensor *q = nullptr;
};

class Model {
public:
    Model() = default;
    ~Model() = default;
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    bool load(Backend &be, const std::string &path, std::string *err);
    void unload();

    // ---- dense weight residency ------------------------------------------
    // Budget for dense weights in device memory, in bytes, set BEFORE load().
    // 0 (the default) is automatic: keep everything resident when it fits and
    // stream whole layers from the file once it does not, leaving VRAM for KV
    // and workspaces. A positive value is an explicit ceiling. The CPU backend
    // never streams (there is no device/host split to page across).
    void set_weight_budget(size_t bytes) { weight_budget_ = bytes; }
    // Upload the file's nextn (MTP) block. Set BEFORE load(), like the weight
    // budget: the block is parsed once, during load, so a caller that asks
    // afterwards would be asking a loader that has already run. KRK_MTP_LOAD=1
    // is the environment spelling; --mtp-probe sets this.
    void set_mtp_load(bool on) { mtp_load_ = on; }
    // The nextn block can also arrive as its OWN file: a `-noMTP` quant of a
    // model whose MTP was published separately (Nemotron 3.5 Lightning ships
    // `mtp-<model>.gguf` beside a `-noMTP` trunk). The sidecar carries the same
    // `blk.<n>.*` tensors the in-file block does, at the block index past the
    // trunk's stack, plus its own copy of the config; this loader reads the
    // block out of it and keeps the mapping alive for the routed banks.
    // Set BEFORE load(), like set_mtp_load: the block is parsed once.
    void set_mtp_path(std::string path) { mtp_path_ = std::move(path); }
    const std::string &mtp_path() const { return mtp_path_; }
    // Refill the streaming ring with layer l's weights. A no-op for a resident
    // layer, so the engine's layer loop calls it unconditionally.
    void stage_layer(i32 l);
    // The nextn/MTP block, when this file carries one and it was asked for.
    const MtpBlock &mtp() const { return mtp_; }
    // A streamed layer is promoted into an allocation of its own the first
    // time the engine needs it, when measured free VRAM can hold it beside the
    // reserve. Streaming is then only what genuinely does not fit, which is a
    // measurement rather than a load-time guess.
    size_t promoted_layers() const { return promoted_layers_; }
    size_t promoted_bytes() const { return promoted_bytes_; }
    f64 promote_ms() const { return promote_ms_; }
    size_t streamed_bytes() const { return stream_bytes_total_; }
    size_t streamed_layers() const { return streamed_layers_; }
    size_t ring_bytes() const { return ring_bytes_; }
    u64 stage_calls() const { return stage_calls_; }
    u64 stage_bytes() const { return stage_bytes_; }
    f64 stage_ms() const { return stage_ms_; }
    // Where a staged layer's wall time went: the drain that orders ring reuse,
    // the file read into pinned staging, and the host->device copy. Only useful
    // as a split -- a combined total cannot say which half to fix.
    f64 stage_sync_ms() const { return stage_sync_ms_; }
    f64 stage_read_ms() const { return stage_read_ms_; }
    f64 stage_xfer_ms() const { return stage_xfer_ms_; }
    // Read time that was NOT paid inside a staging pass: the next layer was
    // read into the spare pinned buffer while this one's copy was in flight.
    // It is wall time the pass would otherwise have spent, so it is reported
    // beside the split rather than inside it.
    f64 stage_prefetch_ms() const { return stage_prefetch_ms_; }
    bool layer_streamed(i32 l) const {
        return l >= 0 && l < static_cast<i32>(layer_streamed_.size()) &&
               layer_streamed_[static_cast<size_t>(l)] != 0;
    }

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

    // ---- per-layer KV geometry --------------------------------------------
    // Wide as the widest layer (workspace and KV-slot sizing); the width one
    // layer actually reads and writes is kv_dim_at().
    i32 n_head_kv_at(i32 layer) const {
        if (cfg_.n_head_kv_layer.empty()) return cfg_.n_head_kv;
        if (layer < 0 || layer >= static_cast<i32>(cfg_.n_head_kv_layer.size()))
            return cfg_.n_head_kv;
        return cfg_.n_head_kv_layer[static_cast<size_t>(layer)];
    }
    i64 kv_dim_at(i32 layer) const {
        return static_cast<i64>(n_head_kv_at(layer)) * cfg_.head_dim;
    }
    // True when this layer reads and writes KV: the ones carrying q/k/v. A
    // mamba/SSM block and a pure-MoE block append nothing, so they hold no KV,
    // are charged for none, and get no plane in the cache. This is the same
    // predicate the engine's attention branch uses.
    bool layer_has_kv(i32 layer) const {
        return layer >= 0 && layer < static_cast<i32>(layers_.size()) &&
               layers_[static_cast<size_t>(layer)].wq.present();
    }
    size_t kv_layer_count() const {
        size_t n = 0;
        for (size_t l = 0; l < layers_.size(); l++)
            if (layer_has_kv(static_cast<i32>(l))) n++;
        return n;
    }
    // True when this layer attends a bounded window instead of the whole past.
    bool is_swa(i32 layer) const {
        // The phase decides which end of the pattern is windowed. laguna is
        // period 4 with the FULL layer first (phase 0); gpt-oss is period 2
        // with the WINDOWED layer first (phase 1). Reading the pattern off the
        // wrong end runs a full-attention kernel on layers the checkpoint
        // trained with a 128-token window and vice versa: the shapes fit, so
        // nothing complains, and every layer sees the wrong context.
        return cfg_.swa_period > 1 && cfg_.swa_window > 0 &&
               ((layer + cfg_.swa_phase) % cfg_.swa_period) != 0;
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
    // qwen4exp has no output_norm: the final hyper-connection mixer carries it,
    // with its own [n_embd, hc] gamma. Null on every other family.
    const f32 *hc_head_norm() const { return hc_head_norm_; }
    // qwen4exp's PLE table is a pointer into the mapping, not an upload: the
    // accessors exist so a test can assert the loader found it and sized one row.
    const u8 *ple_table() const { return ple_table_; }
    // The file's own type for one row of the table: the gather dequantizes one
    // row at a time out of the mapping, so the stride and the type travel
    // together.
    u64 ple_row_bytes() const { return ple_row_bytes_; }
    DType ple_table_type() const { return ple_table_type_; }
    const QuantTensor &hc_head_down() const { return hc_head_down_; }
    const QuantTensor &hc_head_up() const { return hc_head_up_; }
    // NeoX inverse frequencies, length head_dim/2, host-resident (tiny).
    const std::vector<f32> &inv_freq() const { return inv_freq_; }
    i64 weight_bytes() const { return weight_bytes_; }
    const Gguf &gguf() const { return gguf_; }
    // The prism.hadamard plan for this file (inactive for everything else).
    const HadamardPlan &had() const { return had_; }
    // The +-1 sign vector for a folded input `width` wide, in the backend's
    // address space (device memory on the GPU); null when the file declares
    // sign_mode=identity, and never null for a width the plan folded -- the
    // loader refuses that case.
    const f32 *had_signs(i64 width) const {
        for (const auto &sg : had_signs_)
            if (static_cast<i64>(sg.first) == width) return sg.second;
        return nullptr;
    }

    // Lazily-loaded routed experts. Override the residency budget after load to
    // trade reload traffic against memory.
    ExpertCache &experts() { return experts_; }
    const ExpertCache &experts() const { return experts_; }
    // host_bytes sizes the cache's second tier in page-locked host memory; 0
    // disables it and evicted experts go back to the GGUF mapping.
    void set_expert_budget(size_t bytes, size_t host_bytes = 0, size_t slot_budget = 0) {
        experts_.configure(be_, bytes, host_bytes, slot_budget);
    }
    // How many (layer, expert) slots the routed set has (0 when dense). The
    // counterpart of total_expert_bytes() for the residency report: the device
    // budget is sized to cover the whole set, and this is how many slots that
    // is, so the cache can report the set rather than an estimate of it.
    size_t expert_slot_count() const {
        size_t n = 0;
        for (const LayerWeights &L : layers_) {
            if (!L.moe) continue;
            const i64 ne = L.experts.n_expert > 0 ? L.experts.n_expert : 1;
            n += static_cast<size_t>(ne);
        }
        return n;
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

    // Bytes of file this model has mapped, summed over shards (0 when the
    // model was not loaded from a mapping). The mapping is the third
    // residency tier: it is the disk source, and every page it touches is
    // page cache -- RAM that a WARM copy cannot spend a second time.
    size_t mapped_bytes() const {
        size_t n = 0;
        for (const GgufShard &sh : gguf_.shards())
            if (sh.file) n += sh.file->size();
        return n;
    }

private:
    Backend *be_ = nullptr;
    // Dense weight streaming: layers whose weights live in the GGUF mapping and
    // are copied through one reusable device ring before the layer runs. All
    // streamed layers share one ring, so a run's VRAM cost is the resident
    // prefix plus one layer, not the whole model.
    size_t weight_budget_ = 0;   // bytes, 0 = automatic
    std::vector<u8> layer_streamed_;
    std::vector<std::vector<StreamEntry>> stream_;
    void *ring_ = nullptr;
    size_t ring_bytes_ = 0;
    void *stage_host_ = nullptr; // pinned staging buffer, ring-sized
    // Layers that were streamed once and are now resident in a permanent
    // allocation of their own (base, bytes). Freed by unload, and skipped by
    // the per-tensor release because every tensor in them is an interior
    // pointer into that base.
    std::vector<std::pair<void *, size_t>> promoted_;
    // Dense streaming pipeline. Two pinned buffers: the one this layer's copy
    // reads from, and the one the next layer is read into. alternating them is
    // what lets the read (8.9 GB/s) and the DMA (a side stream) run at once.
    void *prefetch_host_ = nullptr;
    void *read_buf_[2] = {nullptr, nullptr};
    i32 prefetched_layer_ = -1;
    int prefetched_slot_ = 0;
    bool prefetch_enabled_ = true;
    f64 stage_prefetch_ms_ = 0;
    // Read layer `l`'s tensors into `host` at the ring's own offsets, and copy
    // them from there into the ring. Split so the read of the next layer can
    // happen between the two.
    void read_layer_to(void *host, const std::vector<StreamEntry> &es,
                       size_t *bytes);
    void upload_layer_from(const void *host, const std::vector<StreamEntry> &es);
    i32 next_streamed(i32 after) const;
    std::vector<size_t> layer_stream_bytes_; // aligned ring bytes per streamed layer
    size_t promoted_bytes_ = 0;
    i32 promoted_layers_ = 0;
    f64 promote_ms_ = 0;
    // Promotion is on by default; KRK_WEIGHT_PROMOTE=0 restores the fixed
    // resident prefix for an A/B of the policy itself.
    bool promote_enabled_ = true;
    // Promotion into a permanent allocation, in stage_layer's own order: the
    // read + copy a staging pass would do anyway, kept. False when this layer
    // is not streamed, when the switch is off, or when it does not fit.
    bool promote_layer(i32 l);
    size_t stream_bytes_total_ = 0;
    size_t streamed_layers_ = 0;
    u64 stage_calls_ = 0;
    u64 stage_bytes_ = 0;
    f64 stage_ms_ = 0;
    f64 stage_sync_ms_ = 0;
    f64 stage_read_ms_ = 0;
    f64 stage_xfer_ms_ = 0;
    bool stream_cur_ = false;    // inside a streamed layer at load time
    size_t stream_off_ = 0;      // ring write offset for the current layer
    i32 stream_layer_index_ = -1;
    Gguf gguf_;
    ModelConfig cfg_;
    MtpBlock mtp_;
    bool mtp_load_ = false;
    std::string model_path_;            // for messages that name the file
    std::string mtp_path_;              // sidecar nextn file, empty for in-model
    std::unique_ptr<Gguf> mtp_gguf_;    // keeps the sidecar's mapping alive
    // The body both paths share: read block `l` out of `g`. `origin` names the
    // file in a refusal, so a sidecar that does not fit does not read as the
    // trunk being broken.
    bool load_mtp_block_from(const Gguf &g, i32 l, const std::string &origin,
                             std::string *err);
    // Upload the nextn block, if the file has one. Called at the end of load()
    // and only when KRK_MTP_LOAD is set: it is ~855 MB of experts on a 35B-A3B
    // and nothing in the main stack reads it.
    bool load_mtp_block(std::string *err);
    std::vector<LayerWeights> layers_;
    QuantTensor tok_embd_, out_head_;
    f32 *out_norm_ = nullptr;
    // qwen4exp's final mixer *is* the output norm (there is no output_norm):
    // the same lane-mix read the blocks use, applied once at the top of the
    // stack. `ple_table_` is the flat n-gram embedding table, kept as a POINTER
    // INTO THE MAPPING because it is 320,001,536 rows of 160 -- tens of GB that
    // must never be uploaded. `ple_row_bytes` is one row's stride.
    f32 *hc_head_norm_ = nullptr;
    QuantTensor hc_head_down_, hc_head_up_;
    const u8 *ple_table_ = nullptr;
    u64 ple_row_bytes_ = 0;
    DType ple_table_type_ = DType::F32;
    std::vector<f32> inv_freq_;
    std::vector<f32> inv_freq_swa_;
    // prism.hadamard. The sign vectors are uploaded once at load and released
    // in unload(); the plan itself is read through had()/had_signs().
    HadamardPlan had_;
    std::vector<std::pair<i32, f32 *>> had_signs_; // (width, device f32[width])
    i64 weight_bytes_ = 0;
    ExpertCache experts_;
    size_t upload_bytes_ = 0;
    f64 upload_ms_ = 0;
};

} // namespace krk

#endif // KRK_MODEL_HPP
