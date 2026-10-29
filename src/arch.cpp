// ============================================================================
//  arch.cpp — the architecture table (see include/krk/arch.hpp).
//
//  Every entry below is a `general.architecture` string that actually appears in
//  the model collections this project is used against; those collections were
//  scanned with tools/gguf_scan.py (arch, layer count, expert count, tensor
//  names) and the table was written from that report, not from memory. An entry
//  is Yes only after the schema ran end to end against a real file (llama,
//  qwen2, qwen3, qwen3moe, qwen35, qwen35moe, gemma*), Partial when the part
//  named in `why` is genuinely not needed for what the model is used for, and No
//  when the tensors name a computation this engine does not implement — those
//  would load without error and then decode nonsense, so they are refused.
// ============================================================================
#include "krk/arch.hpp"

namespace krk {
namespace {

// The fallback every unlisted arch is read against. It is the llama schema
// because that is the schema ggml-family converters share most widely, and
// because the loader's per-layer checks are tensor-presence driven: an arch
// that carries a bias or a QK-norm tensor still gets it, listed or not. What a
// missing entry costs, then, is the geometry the loader cannot see in the file
// (see q_output_gate) and the warning.
constexpr ArchSpec kFallback = {
    /*name=*/"", ArchShape::Dense, ArchSupport::Partial, ArchRole::Target,
    /*family=*/"llama",
    /*why=*/"not in the architecture table; read against llama-like defaults",
    /*qk_norm=*/false, /*q_output_gate=*/false, /*qkv_bias=*/false,
    /*gemma_norm=*/false,
};

// The gemma stack, built here rather than spelled out inline. Every gemma file
// gates its FFN with GeGLU -- gelu(gate) * up, not silu(gate) * up -- and that
// is ArchSpec::gelu_ffn, which sits behind twelve members that have to be
// written out to reach it positionally. A silent SwiGLU there is not a visible
// failure: the two activations agree near zero and differ only in the tail, so
// the model keeps producing plausible text with the wrong function. Naming the
// four shared flags once and setting the rest by name is the way to keep that
// from coming back.
//
//   swa_period / window_first are the family's alternating pattern:
//     gemma2: 2, windowed layer first (even layers 4096-token window)
//     gemma3: 6, windowed first (five local layers per one global)
//   gemma/gemma4 consult the file for the window and keep period 0.
// `qk_norm` is a PARAMETER, not a constant of the family. Gemma 3 carries
// `attn_q_norm`/`attn_k_norm` (llama.cpp's gemma3.cpp:62-63); Gemma 1 and
// Gemma 2 do not -- gemma.cpp and gemma2.cpp create `attn_norm`, `ffn_norm`,
// `output_norm` and (on gemma2) the two BRANCH post norms, and nothing else.
// Setting it for the whole family made the loader refuse every real Gemma 1/2
// GGUF with "layer 0 is missing attn_q_norm.weight, which arch 'gemma2'
// declares" -- a false refusal on a file that is complete.
constexpr ArchSpec gemma_spec(const char *name, ArchSupport sup, const char *why,
                              int swa_period, bool window_first, bool qk_norm) {
    ArchSpec s{name,       ArchShape::Dense, sup,     ArchRole::Target,
               "gemma",    why,              qk_norm,
               /*q_output_gate=*/false, /*qkv_bias=*/false,
               /*gemma_norm=*/true,      /*attn_gate=*/false,
               /*rope_neox=*/true,       swa_period, window_first};
    s.gelu_ffn = true;
    return s;
}

// The loadable schemas first, then the ones that stop at a reason. Flags on a
// refused entry are descriptive: they record what the file carries so the entry
// can be flipped to Yes without re-deriving the schema, and they are never
// consulted by a load that is refused.
// GPT-2's row needs a flag that sits after every other member, and this table is
// positional. Building it here keeps the row readable and keeps the flag from
// having to be written as forty commas.
constexpr ArchSpec gpt2_spec() {
    ArchSpec s{"gpt2", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "gpt2", "", false, false, false, false};
    // LayerNorm carries a bias on every norm; the fused qkv carries one bias;
    // every projection in the stack carries one. `has_bias` is what makes the
    // loader look for them.
    s.gpt2_style = true;
    // The bundle's fused attn_qkv is the same mechanism phi3 uses, so the flag
    // that names it is set here too. `fused_gate_up` stays false: GPT-2's MLP
    // has no gate tensor, and reading one would halve the intermediate.
    s.fused_qkv = true;
    // The bundle, spelled out in the fine flags the rest of the plain family
    // sets individually. Every reader now keys off these, so GPT-2 and its
    // relatives run the same code path with different pieces switched on.
    s.ln_bias = true;
    s.learned_pos = true;
    s.no_gate_ffn = true;
    s.rope_neox = false;
    // No RoPE anywhere: learned positions are added into the embedding instead.
    // The engine reads this off gpt2_style rather than off a rotary flag, so
    // leaving rope defaults alone here is deliberate -- there is no table to
    // build and nothing should try.
    return s;
}

// Phi-3 / Phi-4, which llama.cpp's converter writes as `phi3`. What makes it
// its own schema rather than a llama variant:
//
//   * one `attn_qkv` projection, row = Q | K | V, with K and V
//     n_head_kv*head_dim wide instead of n_embd (Phi is GQA);
//   * one `ffn_up` projection, row = gate | up, n_ff each, and no `ffn_gate`
//     tensor anywhere -- the file has 21 tensors a layer, not 22;
//   * partial rotary: `rope.dimension_count` covers part of the head dim, the
//     rest passes through unrotated. The loader already reads that into
//     ModelConfig::rope_frac, so nothing here has to say so;
//   * the NORM rope pairing, per llama.cpp's llm_arch_rope table, which lists
//     phi2/phi3 as LLAMA_ROPE_TYPE_NORM. That is the default, so it is left
//     alone -- and it is the one choice in this row that a wrong guess would
//     turn into fluent nonsense rather than a refusal, which is why it is
//     named here rather than silently inherited.
constexpr ArchSpec phi3_spec() {
    ArchSpec s{"phi3", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "phi3", "", false, false, false, false};
    s.fused_qkv = true;
    s.fused_gate_up = true;
    return s;
}

// GPT-NeoX: rotary positions, LayerNorm-with-bias everywhere, one fused
// `attn_qkv` WITH a bias, and -- the part that is its own graph -- a PARALLEL
// residual: the attention and the FFN both read the same normed row and are
// summed into the residual, rather than the FFN re-norming the post-attention
// stream. That is `use_parallel_residual` in the file, true for every released
// NeoX checkpoint; the sequential shape would still load and would still
// decode, one graph deeper, which is why the flag exists.
constexpr ArchSpec gptneox_spec() {
    ArchSpec s{"gptneox", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "gptneox", "", false, false, false, false};
    s.fused_qkv = true;
    s.ln_bias = true;
    s.parallel_residual = true;
    s.no_gate_ffn = true;
    // NeoX pairs channel j with j + n_rot/2 (llama.cpp's LLAMA_ROPE_TYPE_NEOX).
    s.rope_neox = true;
    return s;
}

// GPT-J: the same stack with no biases anywhere and the NORM rope pairing.
// It has no `attn_output.bias`, no `ffn_*.bias` and no norm biases, so the
// loader finds them absent -- `upload_f32` returns null for a missing tensor and
// every one of those applications is guarded on the pointer.
constexpr ArchSpec gptj_spec() {
    ArchSpec s{"gptj", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "gptj", "", false, false, false, false};
    s.fused_qkv = true;
    s.ln_bias = true;
    s.no_gate_ffn = true;
    return s;
}

// Phi-2: a fused `attn_qkv` WITH a bias (which this engine refused until now),
// LayerNorm, an ungated GELU MLP, and PARTIAL rotary -- `rope.dimension_count`
// is 32 of an 80-wide head, so only the first 32 channels rotate. The loader
// already reads that into ModelConfig::rope_frac; the forward passes it to the
// rope, which is the same partial-rotary path Phi-3 uses.
constexpr ArchSpec phi2_spec() {
    ArchSpec s{"phi2", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "phi2", "", false, false, false, false};
    s.fused_qkv = true;
    s.ln_bias = true;
    s.no_gate_ffn = true;
    return s;
}

// ---- bitnet-b1.58 ---------------------------------------------------------
// BitNet b1.58 is llama's block with two graph differences:
//
//   * SQUARED ReLU in the FFN, `relu(gate)^2 * up`. The BitNet paper replaces
//     the activation with squared ReLU (Primer-style) for every b1.58 model,
//     and Microsoft's own reference graph says so directly:
//     `inner = self.ffn_sub_norm(squared_relu(x1) * x3)` (microsoft/BitNet
//     gpu/model.py:165-195). llama.cpp's src/models/bitnet.cpp passes
//     LLM_FFN_SILU there -- a fidelity gap in that port, not a second
//     convention in the files: the weights are trained against relu^2.
//   * a SECOND RMSNorm per sub-block (SubLN) on the sub-block's INTERMEDIATE
//     output: `attn_norm -> attn -> attn_sub_norm -> wo` and
//     `ffn_norm -> gate/up -> relu^2 -> ffn_sub_norm -> down`
//     (llama.cpp src/models/bitnet.cpp:59-63,110-118). Those two tensors are
//     read when present (LayerWeights::attn_sub_norm), so this row needs no
//     flag for them -- but a file that lacks them is not a BitNet file, and
//     the fixture in the suite carries both.
//
// Everything else is llama's: RMSNorm, NORM rope pairing, no biases, GQA. The
// head is TIED on the b1.58 files (llama.cpp builds the logits straight from
// `model.tok_embd`, with no `output.weight`), which the loader already handles
// from the tensor's absence.
constexpr ArchSpec bitnet_spec() {
    ArchSpec s{"bitnet-b1.58", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "bitnet", "", false, false, false, false};
    s.ffn_relu_sqr = true;
    return s;
}

// MPT: LayerNorm with a bias, a fused `attn_qkv` WITH a bias, an ungated GELU
// MLP with biases on both projections, and ALiBi in place of any position
// embedding -- `attention.max_alibi_bias` from the file, with an OPTIONAL
// learned position table some exports carry (`mpt-7b` does). llama.cpp lists
// LLM_ARCH_MPT under LLAMA_ROPE_TYPE_NONE, so no rotary is built.
//
// Its ALiBi is a file key, not a constant: `load_arch_hparams` reads
// LLM_KV_ATTENTION_MAX_ALIBI_BIAS as optional and leaves f_max_alibi_bias at 0
// when the key is absent, which is exactly what the loader here does.
constexpr ArchSpec mpt_spec() {
    ArchSpec s{"mpt", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "mpt", "", false, false, false, false};
    s.fused_qkv = true;
    s.ln_bias = true;
    s.no_gate_ffn = true;
    return s;
}

// BLOOM: LayerNorm with a bias everywhere, a fused `attn_qkv` with a bias, an
// ungated GELU MLP with biases, and ALiBi at a FIXED max bias of 8.0 (the
// reference's own constant for this arch -- the key is not in the files).
// It also normalizes the token embedding before block 0, which is why
// `embd_norm` exists.
constexpr ArchSpec bloom_spec() {
    ArchSpec s{"bloom", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "bloom", "", false, false, false, false};
    s.fused_qkv = true;
    s.ln_bias = true;
    s.no_gate_ffn = true;
    s.embd_norm = true;
    s.alibi_max_bias = 8.0f;
    return s;
}

// Falcon: LayerNorm with a bias, a fused `attn_qkv` with NO bias, an ungated
// GELU MLP, no position embedding and no ALiBi on the files llama.cpp ships
// (its loader reads `attention.layer_norm_epsilon` and nothing else) -- the
// position signal is the NEOX-paired rotary. What makes it its own arch is the
// parallel branch with two norms; see ArchSpec::parallel_two_norm.
constexpr ArchSpec falcon_spec() {
    ArchSpec s{"falcon", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "falcon", "", false, false, false, false};
    s.fused_qkv = true;
    s.ln_bias = true;
    s.no_gate_ffn = true;
    s.parallel_residual = true;
    s.parallel_two_norm = true;
    // llama.cpp lists LLAMA_ARCH_FALCON under LLAMA_ROPE_TYPE_NEOX: the pair is
    // channel j with j + n_rot/2, the same pairing GPT-NeoX uses.
    s.rope_neox = true;
    return s;
}

// ---- chatglm --------------------------------------------------------------
// ChatGLM's transformer block is llama's: RMSNorm, rotary, SwiGLU. The one
// difference that has to be declared is that its MLP is ONE fused
// `ffn_up` projection whose row is gate | up (the reference's
// `{ n_embd, n_ff * 2 }` with LLM_FFN_SWIGLU), so there is no `ffn_gate`
// tensor in the file at all. Reading that tensor as a plain `up` would run an
// ungated GELU over twice the intended width -- fluent, wrong, and the same
// class of mistake Phi-3's fused projections had.
constexpr ArchSpec chatglm_spec() {
    ArchSpec s{"chatglm", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "llama", "", false, false, false, false};
    s.fused_gate_up = true;
    return s;
}

// ---- glm4 ------------------------------------------------------------------
// GLM-4 keeps llama's body and adds a norm on EACH branch's OUTPUT:
//
//     x = x + post_attention_norm(attn(norm(x)))
//     x = x + post_ffw_norm(ffn(norm(x + ...)))
//
// which is exactly gemma2's graph, so it rides the existing post_branch_norm
// path rather than getting a new one. Its MLP is one fused `ffn_up`
// (gate | up), and the NextN layers past `block_count` are skipped by the
// reference's own loop -- this engine already ignores blocks past the layer
// count, so that part needs nothing.
constexpr ArchSpec glm4_spec() {
    ArchSpec s{"glm4", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "llama", "", false, false, false, false};
    s.fused_gate_up = true;
    s.post_branch_norm = true;
    return s;
}

// ---- glm4moe (GLM-4.5 / GLM-4.7-Flash) -------------------------------------
// GLM-4's stack with routed experts: an `attn_post_norm` on the attention
// branch (the same post_branch_norm graph glm4 uses), q/k RMS norms, biases on
// q/k/v, a sigmoid router with a per-expert selection bias, and a shared
// expert whose width is `n_ff_exp * n_expert_shared`.
//
// Three pieces are per-LAYER rather than per-arch and need nothing here: the
// leading dense blocks (`leading_dense_block_count`, already read into
// cfg_.n_dense_lead), the experts themselves (dispatched on `ffn_gate_inp`),
// and the shared expert (loaded from `ffn_*_shexp`). A file mixing dense and
// MoE layers therefore needs no special case -- the layer says which it is.
//
// The MLP is NOT fused here: glm4-moe's dense layers carry separate
// `ffn_gate`/`ffn_up`, unlike glm4's single gate|up row. Setting
// fused_gate_up would read a gate of half the intended width.
constexpr ArchSpec glm4moe_spec() {
    ArchSpec s{"glm4moe", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target,
               "glm4", "", false, false, false, false};
    s.qk_norm = true;
    s.qkv_bias = true;
    s.post_branch_norm = true;
    s.router_gating = 2;   // sigmoid, like every GLM-4.5/4.7 batch
    s.router_w_norm = 0;   // llama.cpp leaves expert_weights_norm false here
    return s;
}

// ---- phi3moe (Phi-3.5-MoE) -------------------------------------------------
// Phi-3's block with routed experts in place of the dense MLP: the fused
// `attn_qkv`, the fused gate|up, the partial rotary and the Phi attention
// scaling are all unchanged, and the reference build for this arch is
// literally `llama_model_phi3::graph` with a `build_moe_ffn` call. The router
// is silu/softmax and there is no shared expert, so the two flags are the phi3
// ones and the MoE path is the generic one.
constexpr ArchSpec phi3moe_spec() {
    ArchSpec s{"phi3moe", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target,
               "phi3", "", false, false, false, false};
    s.fused_qkv = true;
    s.fused_gate_up = true;
    // Phi-MoE carries a bias on every RMSNorm (llama.cpp phimoe.cpp creates
    // attn_norm.bias / ffn_norm.bias / output_norm.bias with no OPTIONAL flag).
    s.norm_bias = true;
    return s;
}

// ---- minicpm3 --------------------------------------------------------------
// Multi-head latent attention in the V2-Lite layout (`attn_q_a`/`attn_q_b`
// with `attn_kv_a_mqa`/`attn_kv_b`), which the loader already recognises from
// the tensors, plus one thing no other family here does: each layer scales the
// ATTENTION branch's output by `1.4 / sqrt(n_layer)` before the residual add.
// Without it every layer adds a full-size attention contribution to a stack
// trained to add four fifths of one, and the residual grows faster than the
// model expects -- fluent text, wrong model.
constexpr ArchSpec minicpm3_spec() {
    ArchSpec s{"minicpm3", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "minicpm3", "", false, false, false, false};
    s.mla = true;
    s.rope_neox = true;      // llama.cpp lists minicpm3 under LLAMA_ROPE_TYPE_NEOX
    s.residual_depth = 1.4f; // scale_depth in the reference graph
    // The reference scales the ATTENTION result AND the FFN result by
    // `scale_depth/sqrt(n_layer)` (`cur = ggml_scale(cur, scale_res)` twice);
    // `residual_depth` alone only covers the first. A stack that scales one
    // branch and not the other doubles the FFN's share of every residual.
    s.residual_scale_ffn = true;
    // `inpL = ggml_scale(ctx0, inpL, 12.0f)` before layer 0, and
    // `cur = ggml_scale(ctx0, cur, float(n_embd_base)/float(n_embd))` between
    // the final norm and `output`, with n_embd_base hard-coded 256 in the
    // reference (src/models/minicpm3.cpp:65) -- it is not metadata.
    s.embd_scale = 12.0f;
    s.lm_head_scale_base = 256.0f;
    return s;
}

// ---- mellum ----------------------------------------------------------------
// Mellum is qwen3-MoE's schema: RMSNorm, per-head q/k norm, NeoX rotary, a
// router plus routed experts. It differs in declaring a sliding window on a
// 4-layer pattern: `load_swa_pattern(ml, 4)` with the default
// `dense_first = false`, i.e. `il % 4 < 3` -- layers 0..2 windowed and layer 3
// full (`set_swa_pattern`, llama-hparams.cpp:13). This row said the opposite
// ("period 4 with the full-attention layer first -- this engine's historical
// default"), which put the full layer at index 0 and the window on 1..3: a
// different mask on 3 of every 4 layers. The phase is the flag, not the
// default, so it is named here.
//
// The windowed layers also rotate with PLAIN rope: the reference passes
// `ext_factor = 0` and a SWA freq scale of 1 on them (src/models/mellum.cpp's
// per-layer YaRN factors), so a scaled file must not fold its scaling into
// this half of the stack.
constexpr ArchSpec mellum_spec() {
    ArchSpec s{"mellum", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target,
               "qwen3", "", true, false, false, false};
    s.rope_neox = true;
    s.swa_period = 4;
    s.swa_window_first = true;
    s.swa_plain_rope = true;
    return s;
}

// ---- phimoe ----------------------------------------------------------------
// Phi-MoE is Phi-3's block with the FFN replaced by routed experts: the same
// fused `attn_qkv`, the same RMSNorm-with-a-bias norms (which Phi-3 itself
// carries as an optional tensor and Phi-MoE always does), an `attn_output.bias`
// and an `output.bias` on the head. llama.cpp reuses phi3's graph for it, so
// the only thing that has to change here is the shape and the two biases.
constexpr ArchSpec phimoe_spec() {
    ArchSpec s{"phimoe", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target,
               "phi3", "", false, false, false, false};
    s.fused_qkv = true;
    s.fused_gate_up = true;
    // Phi-MoE carries a bias on every RMSNorm (llama.cpp phimoe.cpp creates
    // attn_norm.bias / ffn_norm.bias / output_norm.bias with no OPTIONAL flag).
    s.norm_bias = true;
    return s;
}

// ---- minicpm ---------------------------------------------------------------
// MiniCPM (v1/v2) is llama's schema with a partial rotary and a scale on the
// embedding (`scale_emb`), which llama.cpp applies as a multiply on the
// embedding row. The tensor set is llama's, so the family is the llama one.
constexpr ArchSpec minicpm_spec() {
    ArchSpec s{"minicpm", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "llama", "", false, false, false, false};
    // `hparams.f_embedding_scale = 12.0f` (src/models/minicpm.cpp:5) is the
    // one scale this family APPLIES: llama.cpp's build_inp_embd multiplies the
    // embedding row by any nonzero `f_embedding_scale`, and 12.0 is that
    // family's default. Its `residual_scale`/`logit_scale` are read into
    // hparams and never used by any graph in the reference either, so this
    // engine not applying them is the same function, not a gap.
    s.embd_scale = 12.0f;
    return s;
}

// ---- qwen2vl ---------------------------------------------------------------
// Qwen2-VL's TEXT tower is Qwen2: q/k/v biases, NeoX rotary, RMSNorm. The
// vision tower and the multimodal projector are separate tensors this engine
// never reads, so a text-only file runs and an image prompt is ignored --
// the same bargain qwen3vl already makes.
constexpr ArchSpec qwen2vl_spec() {
    ArchSpec s{"qwen2vl", ArchShape::Dense, ArchSupport::Partial,
               ArchRole::Target, "qwen2",
               "the vision tower and multimodal projector are not read; images "
               "are ignored and the text tower loads as qwen2",
               false, false, true, false};
    s.rope_neox = true;
    return s;
}

// ---- qwen38 (an ALIAS, resolved by evidence) ------------------------------
// There is no `qwen38` architecture in llama.cpp. "Qwen3.8" is the marketing
// name for a family that llama.cpp converts as **`qwen4exp`**: the converter
// registers `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` with
// `@ModelBase.example("Qwen/Qwen3.8-Flash-Next")`, and the arch string it
// writes is `qwen4exp`. So a checkpoint labelled Qwen3.8 Flash IS qwen4exp,
// which this engine already runs.
//
// That cuts both ways, which is why this row is a conservative default rather
// than the answer. A DENSE Qwen3.8 carries no hyper-connections, no PLE table
// and no indexer, so reading it as qwen4exp would demand metadata it does not
// have -- and reading a Flash file as qwen3 would ignore the delta-net layers,
// the 4-lane residual and the n-gram table. Neither string decides it; the
// TENSORS do, so `assess_model` resolves the alias from what the file carries
// (see kAliases in verdict.cpp) and this entry is only what a file with no
// evidence at all falls back to: qwen3's geometry, which is the stack both
// variants share.
constexpr ArchSpec qwen38_spec() {
    ArchSpec s{"qwen38", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
               "qwen3", "", true, false, false, false};
    s.rope_neox = true;
    return s;
}

// LFM2 and LFM2-MoE, one generator. The reference keeps one loader for both
// (`case LLM_ARCH_LFM2: case LLM_ARCH_LFM2MOE:`), so the only thing that
// differs here is the shape and the expert-weight normalisation.
//
// `norm_bias` is the part the two rows were missing: llama.cpp's lfm2.cpp
// creates `attn_norm.bias` and `ffn_norm.bias` on every layer with no
// OPTIONAL flag, and this engine read per-layer norm biases only under
// `ln_bias`, which also selects the LayerNorm forward. Measured before the
// flag: an lfm2moe twin pair differing only in those two vectors gave
// logits that differed by exactly 0 (peak 3.455).
constexpr ArchSpec lfm2_spec(bool moe) {
    ArchSpec s{moe ? "lfm2moe" : "lfm2",
               moe ? ArchShape::RecurrentMoe : ArchShape::Recurrent,
               ArchSupport::Yes,
               ArchRole::Target,
               "lfm2",
               "",
               /*qk_norm=*/true,
               /*q_output_gate=*/false,
               /*qkv_bias=*/false,
               /*gemma_norm=*/false,
               /*attn_gate=*/false,
               /*rope_neox=*/true,
               /*swa_period=*/0,
               /*swa_window_first=*/false,
               /*expert_w_norm=*/moe ? 1 : 0};
    s.norm_bias = true;
    return s;
}


constexpr ArchSpec kTable[] = {
    // ---- the schemas this engine runs ------------------------------------
    {"llama", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target, "llama", "",
     false, false, false, false},
    phi3_spec(),
    phi3moe_spec(),
    // The plain pre-norm family: rotary (or learned) positions, LayerNorm and a
    // fused QKV, differing only in which of the fine flags above they set.
    gptneox_spec(),
    gptj_spec(),
    phi2_spec(),
    bitnet_spec(),
    mpt_spec(),
    bloom_spec(),
    falcon_spec(),
    chatglm_spec(),
    glm4_spec(),
    glm4moe_spec(),
    mellum_spec(),
    phimoe_spec(),
    minicpm_spec(),
    minicpm3_spec(),
    qwen2vl_spec(),
    qwen38_spec(),
    // GPT-2 and its descendants: LayerNorm (biased), learned absolute
    // positions, a fused attn_qkv, and a plain GELU MLP. See
    // ArchSpec::gpt2_style for why those four travel together.
    gpt2_spec(),
    // Qwen2 adds a bias on q/k/v; everything else is the llama schema. It is
    // the first arch here whose rope is the *NeoX* pairing (llama.cpp's table
    // lists qwen2 under LLAMA_ROPE_TYPE_NEOX): the pair is channel j with
    // j + n_rot/2, not two adjacent channels.
    {"qwen2", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target, "llama", "",
     false, false, true, false, false, true},
    // Qwen3 adds per-head QK-norm, and keeps Qwen2's NeoX rope pairing.
    {"qwen3", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target, "qwen3", "",
     true, false, false, false, false, true},
    {"qwen3moe", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target, "qwen3", "",
     true, false, false, false, false, true},
    // Qwen1.5-MoE (`qwen2moe`): Qwen2's body -- q/k/v biases, NeoX rotary,
    // RMSNorm -- with a routed bank beside a SHARED expert, and three details
    // that are the family's own:
    //   * the shared expert is gated by a per-token scalar: the reference
    //     computes `silu(g)/g`, which is exactly sigmoid(g), from a 1-D
    //     `ffn_gate_inp_shexp` row, and multiplies the shared branch by it
    //     before folding it into the routed sum;
    //   * the top-k weights are used as the softmax emitted them -- llama.cpp
    //     passes `norm_w = false` -- so `expert_w_norm` is 0, the same choice
    //     OLMoE makes and the opposite of LFM2-MoE's;
    //   * the router is the plain softmax of `ffn_gate_inp`, no bias vector.
    // Everything else is machinery this engine already runs for qwen2/qwen3moe,
    // so the entry is the qwen2 flags plus the MoE shape.
    {"qwen2moe", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target, "qwen2", "",
     false, false, true, false, false, true, 0, false, /*expert_w_norm=*/0},
    // Qwen1: llama's block with a FUSED q/k/v projection (`attn_qkv`, one row
    // of Q | K | V) and the NeoX rope pairing. The fused row carries a MANDATORY
    // `attn_qkv.bias` of the same width (`src/models/qwen.cpp` creates it with
    // flags 0, so every file has one) -- the loader reads it as `qkv_bias` and
    // the engine adds its three slices to q/k/v. Qwen2 splits the projection
    // again, which is why this is its own entry rather than a flag on `qwen2`.
    [] {
        ArchSpec s{"qwen", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
                   "llama", "", false, false, false, false};
        s.fused_qkv = true;
        s.rope_neox = true;
        return s;
    }(),
    // InternLM2: the same fused Q | K | V row, but the adjacent-pair (NORM)
    // rope pairing. Two families that differ only in the pairing are the
    // cleanest demonstration of why it is a declared flag: the wrong one mixes
    // the wrong channels and every token still lands in the right range.
    [] {
        ArchSpec s{"internlm2", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
                   "llama", "", false, false, false, false};
        s.fused_qkv = true;
        return s;
    }(),
    // ---- llama's schema with a piece renamed, a flag, or a scalar ---------
    // A handful of families whose graphs are llama's, each with one thing of
    // its own (or -- OLMo-1 -- with its norms missing entirely).
    // The reference is `src/models/*.cpp` at llama.cpp f5222f25; every claim
    // below is read off that family's own builder, not inferred from the name.
    //
    // OLMo-1 (allenai OLMo 1B/7B/70B): llama's block with the NORM rope pairing,
    // an ungated-silu FFN, no biases -- and two pieces of its own that are not
    // in any tensor, which is why this row was wrong for two releases:
    //
    //   * EVERY norm is WEIGHTLESS. `src/models/olmo.cpp` calls
    //     `build_norm(inpL, NULL, NULL, LLM_NORM)` for the attention norm, the
    //     FFN norm and the final norm, and its tensor list creates no
    //     `attn_norm`/`ffn_norm`/`output_norm` at all. Listing the family
    //     against the llama base therefore refused every complete checkpoint
    //     ("layer 0 is missing its norm weights") -- a wrong refusal, not a
    //     wrong answer.
    //   * the attention projection is CLAMPED to `attention.clamp_kqv` before
    //     the rotary (the reference reads LLM_KV_ATTENTION_CLAMP_KQV and the
    //     converter writes `clip_qkv: 8.0` from every OLMo-1 config). The
    //     loader now reads the key and the engine applies it; left out, the
    //     layer sees unbounded q/k/v and the answer is plausible and wrong.
    [] {
        ArchSpec s{"olmo", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
                   "llama", "", false, false, false, false};
        s.noweight_norm = true;
        return s;
    }(),
    // XVERSE: the same, and the same rope pairing. Its 40B/65B checkpoints
    // carry a partial-rotary `rope.dimension_count`, which is read per model
    // rather than per arch.
    {"xverse", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target, "llama", "",
     false, false, false, false},
    // OLMo 2/3: WIDE QK norm, the NEOX rope pairing, a sliding-window pattern,
    // and -- the part that is not llama -- a post norm on EACH branch before it
    // is added:
    //     x += attn_post_norm(attn(attn_norm(x)))
    //     x += ffn_post_norm(ffn(ffn_norm(x)))
    // which is gemma2/3's `post_branch_norm` with different tensor names
    // (`post_attention_norm` / `post_ffw_norm`, which the loader already binds
    // under that flag) and OLMo's own rope.
    //
    // The norm WIDTH is the family's own shape and was described here as "per
    // head, the qwen3 shape" -- it is not. `src/models/olmo2.cpp` applies both
    // norms to the PROJECTED row before the head split:
    //     attn_q_norm = {n_embd}                = n_head * n_embd_head
    //     attn_k_norm = {n_head_kv * n_embd_head}
    // so q is one RMS over the whole query row and k is one RMS over the whole
    // key row -- which on a GQA stack is NARROWER than n_embd (8 KV heads of
    // 128 on the 7B is 1024 against 4096). The loader's width check accepted
    // only `head_dim` or `n_embd`, so every GQA file of this family was refused
    // with "neither head_dim nor the full row"; it now accepts each branch's
    // own row width (see Model::load's qk_norm block).
    //
    // The window pattern is `load_swa_pattern(ml, 4)`, whose default is
    // dense_first = false: layers 0..2 windowed, layer 3 full.
    [] {
        ArchSpec s{"olmo2", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
                   "llama", "", /*qk_norm=*/true, /*q_output_gate=*/false,
                   /*qkv_bias=*/false, /*gemma_norm=*/false,
                   /*attn_gate=*/false, /*rope_neox=*/true};
        s.post_branch_norm = true;
        s.swa_window_first = true;
        // `hparams.rope_freq_scale_train_swa = 1.0` and an `ext_factor` of 0
        // on the windowed layers (src/models/olmo2.cpp:11-12, :120): the
        // windowed half of the stack never carries the full layers' scaling.
        s.swa_plain_rope = true;
        return s;
    }(),
    // Seed-OSS: NEOX rope, silu FFN, an `attn_output.bias`, and its FFN input
    // norm named `attn_post_norm` -- the reference computes
    // `ffn(build_norm(x + attn_out, attn_post_norm))`, i.e. that tensor is the
    // FFN's PRE norm and the attention branch is not post-normed at all. The
    // name reads like gemma's branch norm, so the loader's fallback for it is
    // what makes this family run the right graph instead of a plausible one.
    {"seed_oss", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target, "llama", "",
     false, false, false, false, false, true},
    // SmolLM3: llama's schema and rope pairing plus a NoPE pattern -- every
    // fourth layer rotates nothing (`n_no_rope_layer_step = 4` in the
    // reference; `(l + 1) % 4 != 0` is when rope runs). This comment used to
    // claim an `attn_output.bias` as well: the reference creates none
    // (`src/models/smollm3.cpp` loads exactly llama's tensors), and the loader
    // reads `attn_output.bias` only when a file carries it, so the claim was
    // wrong about the family rather than about the reading.
    [] {
        ArchSpec s{"smollm3", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target,
                   "llama", "", false, false, false, false};
        s.no_rope_layer_step = 4;
        return s;
    }(),
    // OLMoE: llama-shaped attention whose QK norm is WIDE -- the weight spans
    // n_head*hd, one RMS over the whole projected row, not one per head -- and a
    // top-8 softmax MoE that uses the router probabilities as they come out of
    // the softmax (llama.cpp's build_moe_ffn passes norm_w=false, so the top 8
    // of 64 hold ~40% of the mass, not 1). Reading it against llama defaults
    // ignored both norm tensors, because the default entry has qk_norm=false:
    // the model then decodes garbage in BOTH backends, and the two arms drifted
    // apart at decode step 5 because their logits were a near-tie. The row is
    // here so the norms are read and the expert weights are not renormalized.
    {"olmoe", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target, "llama", "",
     /*qk_norm=*/true, /*q_output_gate=*/false, /*qkv_bias=*/false,
     /*gemma_norm=*/false, /*attn_gate=*/false, /*rope_neox=*/false,
     /*swa_period=*/0, /*swa_window_first=*/false, /*expert_w_norm=*/0},
    // Qwen3.5 and its MoE sibling: gated delta net blocks interleaved with
    // attention, and the attending layers pack an output gate into attn_q.
    {"qwen35", ArchShape::Recurrent, ArchSupport::Yes, ArchRole::Target, "qwen35",
     "", true, true, false, false},
    {"qwen35moe", ArchShape::RecurrentMoe, ArchSupport::Yes, ArchRole::Target,
     "qwen35", "", true, true, false, false},
    // LFM2.5 (Liquid AI, `lfm2moe`). A hybrid like qwen35 but a different
    // mixer: the non-attending layers are a short DEPTHWISE convolution over
    // the last `shortconv.l_cache` rows -- `shortconv.in_proj` is 3*n_embd wide
    // and emits B | C | X, the convolution runs on B*X (not on X), and C gates
    // the convolved row:
    //
    //     bx = B * X;  y = conv_dw(bx);  out = out_proj(C * y)
    //
    // Three more things the schema needs and the llama defaults do not find:
    // `attention.head_count_kv` is a PER-LAYER array of length block_count (0
    // on the conv layers, 8 on the 10 that attend), the file carries no
    // `output_norm` because its final norm is `token_embd_norm`, and
    // `expert_gating_func` 2 selects experts with a sigmoid.
    // Two more answers the file does not state, both measured against the
    // reference's own text on the 24B-A2B file:
    //   * RoPE pairs NeoX half-split. llama.cpp reports `rope type = 2` for
    //     this file and `(j, j + n_rot/2)` is what that case arm applies, so
    //     the GGUF keeps the HF layout unpermuted and the engine must pair the
    //     halves, not the neighbours.
    //   * the top-k sigmoid weights ARE sum-normalized (`expert_w_norm`).
    //     Without it the same prompt decodes ` "The" and "the"`, with it the
    //     reference's ` Paris. The capital of ...`; the two differ by the
    //     sigmoid sum (about 2x on every expert contribution), which is why a
    //     wrong answer here is fluent rather than broken.
    lfm2_spec(/*moe=*/true),
    // LFM2 (dense): the same stack as LFM2-MoE with the routed block replaced
    // by a plain dense FFN. The reference keeps ONE builder and one hparam set
    // for both (`case LLM_ARCH_LFM2: case LLM_ARCH_LFM2MOE:`), and the mixer,
    // the per-layer `head_count_kv` array, the missing `output_norm` and the
    // NeoX rope pairing are identical -- only `expert_count` is absent. So the
    // family is declared once and the layer's own bank decides which block runs.
    lfm2_spec(/*moe=*/false),
    // Gemma scales the embedding by sqrt(n_embd), uses a tighter eps, gates its
    // FFN with GeGLU and softcaps the final logits. Gemma 1 is the llama schema
    // plus those four, and it carries NO QK-norm: llama.cpp's gemma.cpp creates
    // `attn_norm`, `ffn_norm` and `output_norm` per layer and nothing else.
    //
    // `Yes` since 2026-10-29: a file built from this definition -- the tensor
    // set gemma.cpp declares, the embedding scale, the GeGLU gate and both
    // softcap keys -- loads and generates (`test_pending_model_fixtures`). The
    // flag used to be true here for the whole family, which refused every real
    // Gemma 1 file for a tensor the reference never writes.
    gemma_spec("gemma", ArchSupport::Yes, "",
               0, false, /*qk_norm=*/false),
    // Gemma 2: every piece of its schema is now implemented. The alternating
    // window (period 2, windowed layer first) is read from this entry and the
    // loader's is_swa() drives it; GeGLU, the embedding scaling and the
    // final-logit softcap were already there; and the softcap on the ATTENTION
    // logits -- cap * tanh(score / cap) between the QK dot and the softmax, i.e.
    // inside the score formation rather than as a transform of the result,
    // which is a different function -- is AttnDesc::softcap, applied by the
    // reference attention and gated against a hand-computed softmax in the
    // tests. The HIP kernels keep four score paths of their own and none caps
    // yet, so that backend answers attention_softcap_supported() == false and
    // Engine::init refuses the run by name rather than decoding a different
    // model; --cpu and both NPU arms run it.
    // gemma2 and gemma3 both normalize each branch's OUTPUT
    // (`attn_post_norm`, `ffn_post_norm` in the reference's layer) and add that
    // instead of the raw branch. Until this flag was set the loader never bound
    // either tensor, so both families ran as if they had none -- a wrong graph
    // that decodes fluently. gemma1 is the family WITHOUT them.
    [] {
        ArchSpec s = gemma_spec("gemma2", ArchSupport::Partial,
               "the attention-logit softcap is applied by the reference "
               "attention (--cpu, --npu, --npu-cpu) but not by the HIP "
               "kernels, which refuse the run by name; everything else in the "
               "schema is implemented and a file built from gemma2.cpp's own "
               "tensor list runs end to end (test_pending_model_fixtures)",
               2, true, /*qk_norm=*/false);
        s.post_branch_norm = true;
        // 27B: the reference multiplies the query by 1/sqrt(n_embd/n_head)
        // instead of 1/sqrt(head_dim) (gemma2.cpp:14-27).
        s.attn_scale_embd_per_head_layers = 46;
        return s;
    }(),
    // Gemma 3: QK-norm, GeGLU, embedding scaling, a final-logit softcap and a
    // 5:1 local/global pattern (period 6, windowed layers first, 1024-token
    // window declared in the file). Every one of those pieces is implemented
    // and driven by schema flags rather than by arch name.
    //
    // `Yes` since 2026-10-29. Gemma 3 is the family that DOES carry
    // `attn_q_norm`/`attn_k_norm` (gemma3.cpp:62-63), which is why the flag is
    // true here and false on gemma/gemma2; a file built from that tensor list
    // loads and generates (`test_pending_model_fixtures`). Gemma 3 also has no
    // attention-logit softcap, so unlike gemma2 it does not depend on the
    // reference-only softcap path and runs on every backend.
    [] {
        ArchSpec s = gemma_spec("gemma3", ArchSupport::Yes, "",
               6, true, /*qk_norm=*/true);
        s.post_branch_norm = true;
        // 27B: same 1/sqrt(n_embd/n_head) rule, keyed on the reference's own
        // discriminator (gemma3.cpp:24, n_layer == 62).
        s.attn_scale_embd_per_head_layers = 62;
        return s;
    }(),
    // Gemma 4 (E-series) as the files in the collection declare it: 42 layers,
    // shared-KV layers, two attention geometries (global heads of 512 with 512
    // rope dims, sliding-window heads of 256 with 256) and per-layer input
    // embeddings — four pieces beyond the dense schema.
    //
    // The per-layer KV head count is named here because of what it used to look
    // like instead: `attention.head_count_kv` is an array on this family (the
    // files here declare {1,8} and {2,8} across their attention layers, while
    // the layer count is 48 and 30), and until this entry said so, a file that
    // got past the arch table was refused by the head-geometry check with
    // "attention tensors disagree with the declared head geometry" -- a message
    // about the file's own tensors, for a feature this engine does not have.
    // Two attention geometries in one stack: `attention.key_length` on the
    // global layers and `attention.key_length_swa` on the windowed ones, with
    // the pattern stated per layer in `attention.sliding_window_pattern`. Both
    // are READ now (cfg_.head_dim_layer, cfg_.swa_pattern, head_dim_at(),
    // is_swa()) and pinned by test_per_layer_head_width through a real file, so
    // every downstream width -- q_dim_at, kv_dim_at, the KV slot stride, the
    // rope width, the attention scale -- follows the layer rather than the
    // stack maximum.
    //
    // Everything this family's dense form needs is RUN: per-layer input
    // embeddings (per_layer_token_embd + per_layer_model_proj + per-layer
    // inp_gate/proj/post_norm), shared-KV layers, `layer_output_scale`, the
    // weightless value norm and the two attention geometries. The ROUTED form
    // is a two-branch block (moe_two_branch) and runs through its own path.
    // Every piece the parse above describes is now RUN: the per-layer input
    // table and its model projection (a pre-pass plus one small gate/proj per
    // layer), the shared-KV tail (those layers read an earlier layer's plane
    // and append nothing), the weightless value norm, the per-layer output
    // scalar, the proportional rope folded into the frequency table, and the
    // two geometries this family's layers alternate between.
    [] {
        ArchSpec s = gemma_spec("gemma4", ArchSupport::Yes, "",
                                0, false, /*qk_norm=*/true);
        s.per_layer_embd = true;
        s.v_norm_weightless = true;
        s.shared_kv = true;
        s.layer_out_scale = true;
        s.post_branch_norm = true;
        // The routed form: the dense FFN becomes the shared branch, and the
        // experts get their own pre/post norms plus a detached router input.
        s.moe_two_branch = true;
        // `build_moe_ffn(..., norm_w = true, ...)`: the top-k weights are
        // renormalized after selection, the opposite of qwen2moe's
        // un-normalized softmax.
        s.expert_w_norm = 1;
        // `hparams.f_attention_scale = 1.0f; // Gemma4 uses self.scaling = 1.0
        // (no pre-attn scaling)` -- the reference passes that value where every
        // other family passes 1/sqrt(head_dim). On this stack's 512-wide heads
        // the default is 22x colder, and the softmax is flat enough that the
        // answer is plausible text rather than a failure.
        s.attn_scale = 1.0f;
        return s;
    }(),

    // ---- loads and runs, with a named gap ---------------------------------
    // The language tower of a vision model is the text schema; the vision tower
    // and the projector are extra tensors this engine never reads.
    {"qwen3vl", ArchShape::Dense, ArchSupport::Partial, ArchRole::Target, "qwen3",
     "the vision tower and multimodal projector are not read; images are "
     "ignored and the text tower loads as qwen3",
     true, false, false, false, false, true},

    // ---- recognized, and refused with the reason --------------------------
    // qwen4exp: Qwen3.8-Flash-Next and its REAP expert-reduced repacks. Read off
    // the file's own metadata (1224 tensors, 48 blocks): 2560 embedding, 24 query
    // heads of 256 dims over 2 KV heads, a Mamba-style SSM block (conv 4, state
    // 128, 16 groups, dt_rank 48, inner 6144), full attention every 4th layer with
    // a QSA indexer (4 heads x 128, top_k 2048) and a per-layer `compress_ratios`
    // -- the QSA indexer's own, which is the one qwen4exp declares, not the
    // DeepSeek-V4 CSA/HCA compressor -- a 288-expert top-10 MoE with a shared
    // expert, a 4-lane 10240-wide
    // hyper-connection stream mixed through a 320-dim bottleneck, and a
    // 320,001,536-row PLE n-gram table.
    //
    // The shape is Moe and not RecurrentMoe *on purpose*. The gap map keys its
    // exemptions on shape, and a recurrent shape means "a recurrent schema may
    // have these": `attn_qkv` is the gated delta net's own projection there, and
    // `attn_gate`/`ssm_*` are the delta net's, all implemented. Declaring this
    // family recurrent would exempt exactly the four tensors that are the reason
    // it cannot run -- the file would be refused by name with no tensor reasons,
    // and a later "Yes" would load it and decode nonsense. As Moe, each piece is
    // named: attn_gate, fused attn_qkv, ssm_conv1d, ssm_out.
    //
    // The entry's job is the *family* claim: "not in the architecture table" and
    // "this engine cannot run it" are different statements, and the second is the
    // true one. docs/AUDIT-qwen4exp-support.md has the per-piece inventory and the
    // reference engine that runs it.
    // Now a runnable schema: the loader reads its hyper-connection, PLE and
    // indexer tensors, and the delta-net half is the gated delta net this engine
    // already runs (16 key heads of 128, 48 value heads of 128, conv 4). See
    // docs/AUDIT-qwen4exp-support.md for the per-stage state and the reference
    // implementation the shapes were read from.
    // 
    // SUPPORTED, and the thing that changed is the gate, not the graph.
    //
    // Every tensor in the schema has been read for a while, and the
    // hyper-connection residual, the PLE block and the QSA indexer all run
    // (each has a device arm in `kraken-bench --hc` or its own
    // `test_qwen4exp_*` group). What kept the arch behind
    // KRK_ARCH_ALLOW_EXPERIMENTAL was that no suite gate pinned the SENTENCE:
    // the block groups prove the blocks, and a graph can pass every one of them
    // and still decode nonsense -- which is exactly what the PLE normalisation
    // bug did before it was found.
    //
    // `test_qwen4exp_whole_graph` is that gate: the fixture's greedy sequence is
    // pinned, and three arms keep the pin from being vacuous (a different prompt
    // moves it, chunk 1 reproduces it, a shifted PLE table breaks it). Its
    // provenance is the anchor on real weights -- the REAP-288 file reproduces
    // llama.cpp's greedy continuation of "The capital of France is" for 24 of
    // 24 tokens (docs/AUDIT-qwen4exp-support.md) -- so the pin is a regression
    // claim with an external reference behind it, not a second oracle.
    //
    // One piece of honesty for the next reader: this engine's speed on that
    // file is not the point of the row. 84 GB of REAP-288 with a 42 GiB routed
    // set decodes at 5.7 tok/s here (expert-tier bound), which is a correct
    // answer at a slow rate, and correctness is what the verdict states.
    {"qwen4exp", ArchShape::RecurrentMoe, ArchSupport::Yes, ArchRole::Target,
     "qwen4exp", "",
     /*qk_norm=*/true, /*q_output_gate=*/true, /*qkv_bias=*/false,
     /*gemma_norm=*/false, /*attn_gate=*/false, /*rope_neox=*/true,
     /*swa_period=*/0, /*swa_window_first=*/false, /*expert_w_norm=*/-1,
     /*swiglu_oai=*/false, /*hyper_connection=*/true, /*ple=*/true,
     /*qsa=*/true, /*attn_gate_elem=*/false, /*no_out_norm=*/true},

    // Multi-head latent attention, legacy lite layout (DeepSeek-V2 / V2-Lite /
    // Coder-V2-Lite): direct attn_q, one attn_kv_a_mqa projection emitting the
    // compressed latent beside the decoupled rope key, one unsplit attn_kv_b
    // table. It RUNS (src/hip/kernels/mla.hpp, Engine::mla_forward).
    //
    // The table's verdict is about the ARCH, and MLA ships in three layouts
    // that are three different graphs. Only the lite one is implemented, so
    // the engine checks the FILE's layout at init and refuses the split and
    // shared ones by name (see the MLA block in Engine::init). That is
    // deliberate: a loader that refused the family would also refuse the
    // layout this build runs, and a file that loads and then decodes the wrong
    // graph is the failure mode this engine exists not to have.
    //
    // expert_w_norm = 0 for the reason gpt-oss has it: this family's routed
    // weights are the TOP-K OF A FULL-VOCAB SOFTMAX, not a softmax over the
    // selected experts, so they sum to well under 1 -- the top 6 of 64 hold
    // maybe half the mass -- and renormalizing them multiplies every layer's
    // expert contribution by about 2. llama.cpp leaves expert_weights_norm
    // false for a file that does not declare it, and this file does not.
    // Measured, not assumed: with normalization the same prompt answered in
    // the right voice for four greedy tokens and then picked a different word;
    // without it the run follows llama.cpp token for token.
    {"deepseek2", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target, "deepseek2",
     "",
     /*qk_norm=*/false, /*q_output_gate=*/false, /*qkv_bias=*/false,
     /*gemma_norm=*/false, /*attn_gate=*/false, /*rope_neox=*/false,
     /*swa_period=*/0, /*swa_window_first=*/false, /*expert_w_norm=*/0,
     /*swiglu_oai=*/false, /*hyper_connection=*/false, /*ple=*/false,
     /*qsa=*/false, /*attn_gate_elem=*/false, /*no_out_norm=*/false,
     /*mla=*/true, /*dsv4=*/false, /*attention_implemented=*/true},
    // Shared-latent MLA (one wkv latent, per-head queries, grouped wo_a/wo_b)
    // plus V4-only extras: compress_ratios, Sinkhorn hyper-connections,
    // tid2eid hash routing, YaRN-on-compressed-layers rope. The loader reads
    // the metadata and (for the dense core) the weights; the engine refuses
    // until each piece runs. qk_norm is false: the file carries no per-head
    // q/k norms, only the latent q_a/kv norms. attention_implemented stays
    // false: this is the SHARED latent, whose attention is a different graph
    // from the lite one that runs (the head count lives in a grouped output
    // projection, not in a per-head K/V unfolded at runtime).
    // What runs: the shared-latent attention (one normed latent row per token,
    // rope on its tail, grouped low-rank output) and the tid2eid hash route that
    // the first `hash_layer_count` layers select by token id. `attn_sinks` DO
    // run: they are a per-head logit that competes with every visible key for
    // softmax mass and contributes no value of its own, which the kernel and the
    // host oracle both already carried.
    //
    // The CSA/HCA compressors RUN: the loader reads the per-layer
    // attn_compressor_* tables (and the indexer's on ratio-4 layers), the engine
    // keeps the per-token state rows and the pooled keys, and the attention
    // reads both row families (Engine::mla_shared_forward_compressed,
    // test_dsv4_compressed_end_to_end). The arithmetic is the oracle in
    // src/dsv4_compress.cpp -- the state row with APE, the per-channel softmax
    // pool, the overlapping halves, the compression-base rope, the indexer's
    // relu-weighting and top-k -- which is a HOST f32 reference, so the
    // compressed path answers on backends that implement the two-row-family
    // attention op (the CPU reference) and is refused BY CAPABILITY elsewhere.
    // What is still missing on this row: Engram conditional memory and the
    // 4-wide hyper-connections. A dense-core file (all ratios 0) is the same
    // graph without the second row family, and test_dsv4_whole_graph pins it.
    {"deepseek4", ArchShape::Moe, ArchSupport::Partial, ArchRole::Target, "deepseek4",
     "shared-latent MLA, the grouped output, the attention sinks, the tid2eid "
     "hash route and the CSA/HCA compressors with the lightning indexer run on "
     "the CPU reference; Engram and the Sinkhorn hyper-connections do not, and "
     "a file carrying either is refused by name rather than run without it",
     /*qk_norm=*/false, /*q_output_gate=*/false, /*qkv_bias=*/false,
     /*gemma_norm=*/false, /*attn_gate=*/false, /*rope_neox=*/false,
     /*swa_period=*/0, /*swa_window_first=*/false, /*expert_w_norm=*/-1,
     /*swiglu_oai=*/false, /*hyper_connection=*/false, /*ple=*/false,
     /*qsa=*/false, /*attn_gate_elem=*/false, /*no_out_norm=*/false,
     /*mla=*/true, /*dsv4=*/true, /*attention_implemented=*/true},
    // DeepSeek-V4.1-Flash (-1017 and the earlier V4.1-Flash): the same graph
    // as deepseek4 under the architecture string llama.cpp's converter writes
    // for this family, which is `deepseek41` rather than `deepseek4`. A file
    // carrying the V4.1 name was previously "unknown architecture" -- the
    // least useful answer available, because the reasons it cannot run are
    // specific and already enumerated on the entry above (compressed
    // attention, the sparse indexer, attn_sinks, hyper-connections). Mapping
    // the name onto the implementation inherits every one of those refusals
    // instead of replacing them with a shrug.
    //
    // What it does NOT cover is Engram (a token-hash conditional-memory lookup,
    // 196 B parameters) and the 4-wide Sinkhorn hyper-connections, whose `hc_*`
    // tensors are a DIFFERENT mixer from the one qwen4exp runs.
    //
    // Both now have their ALGEBRA implemented and pinned, with the loader and
    // the forward still to wire:
    //   * hc_*: src/hc.cpp's hc_sinkhorn_* (the (2+hc)*hc split, the doubly
    //     stochastic sinkhorn in the reference's order, the dst-major/file
    //     index transposition) -- test_dsv4_sinkhorn_mixer.
    //   * engram_*: src/engram.cpp's rolling n-gram hash, the shared
    //     per_layer_tok_embd gather and the dead-shift/pad rule --
    //     test_engram_hash.
    // What is missing is the part that makes them RUN in a forward: the loader
    // does not read the tensors yet, so a file carrying either is still refused
    // by name, and the lane stack for this family is not wired. A checkpoint
    // that carries either is refused by name through the gap map rather than
    // loaded with the block skipped. A dense-core file -- every ratio 0, no
    // hc_*/engram_* -- loads and generates end to end
    // (`test_pending_model_fixtures` builds one from this row's tensor list).
    {"deepseek41", ArchShape::Moe, ArchSupport::Partial, ArchRole::Target,
     "deepseek4",
     "recognised as the deepseek4 graph: shared-latent MLA, the grouped "
     "output, the attention sinks, the tid2eid hash route and the CSA/HCA "
     "compressors with the lightning indexer all run; Engram conditional memory "
     "and the 4-wide hyper-connections do not, so a V4.1 checkpoint that "
     "carries hc_* or engram_* tensors is refused by name, not mis-loaded",
     /*qk_norm=*/false, /*q_output_gate=*/false, /*qkv_bias=*/false,
     /*gemma_norm=*/false, /*attn_gate=*/false, /*rope_neox=*/false,
     /*swa_period=*/0, /*swa_window_first=*/false, /*expert_w_norm=*/-1,
     /*swiglu_oai=*/false, /*hyper_connection=*/false, /*ple=*/false,
     /*qsa=*/false, /*attn_gate_elem=*/false, /*no_out_norm=*/false,
     /*mla=*/true, /*dsv4=*/true, /*attention_implemented=*/true},
    // The MTP/prediction-head shard of DeepSeek-V4: `mtp.*` tensors, nothing else.
    {"deepseek4-dspark", ArchShape::Moe, ArchSupport::No, ArchRole::Draft,
     "deepseek4",
     "multi-token-prediction head shard (mtp.*) rather than a model",
     false, false, false, false},
    // Hybrid Mamba/attention stacks: the mamba blocks are not the gated delta
    // net this engine implements (different recurrence, different state).
    [] {
        ArchSpec s{"nemotron_h", ArchShape::Recurrent, ArchSupport::Yes,
                   ArchRole::Target, "nemotron_h", "", false, false, false,
                   false};
        // Its expert banks are built with LLM_FFN_RELU_SQR (gated by
        // relu^2, not silu). See ArchSpec::ffn_relu_sqr.
        s.ffn_relu_sqr = true;
        return s;
    }(),
    // Nemotron MoE: alternating Mamba-2 SSM blocks + shared experts.
    // The SSM blocks are loaded by Model::load - the per-layer ssm_* upload,
    // keyed on the tensors a layer actually carries - and run by the mamba2_ssd
    // kernels in src/hip/kernels/gdn.hpp, launched from backend_hip and driven
    // by Engine::mamba2_forward. The architecture uses:
    //   - Even layers: ssm_in, ssm_out, ssm_conv1d, ssm_a, ssm_d, ssm_dt, ssm_norm
    //   - Odd layers: ffn_up_shexp, ffn_down_shexp (shared experts, NVFP4)
    // This is different from GDN: no gate, simpler SSM, different tensor layout.
    [] {
        ArchSpec s{"nemotron_h_moe", ArchShape::RecurrentMoe, ArchSupport::Yes,
                   ArchRole::Target, "nemotron_h", "", false, false, false,
                   false, false, true};
        s.ffn_relu_sqr = true;
        return s;
    }(),
    // ---- speculative head sets -------------------------------------------
    // An MTP head set, extracted by tools/mtp_extract.py: the `mtp.*` tensors a
    // checkpoint ships beside its own weights -- a norm pair, two input
    // projections and one block with its own attention (packed query gate,
    // per-head QK-norm, a QSA indexer) and its own 512-expert MoE.
    //
    // Refused, and the refusal names what would have to exist first. qwen4exp
    // -- the only family this head belongs to -- runs here, so the target is no
    // longer the gap; the BLOCK is: gated attention, a QSA indexer, four-lane
    // hyper-connections and a 512-expert MoE, none of which this engine has a
    // forward for. The file also carries no embedding and no head, so it is a
    // head set and not a model under any reading. Claiming it would mean
    // proposing tokens from a block whose forward does not exist.
    {"qwen4-mtp", ArchShape::Moe, ArchSupport::No, ArchRole::Draft, "qwen4exp",
     "an MTP head set for a qwen4exp target: one block with gated attention, a "
     "QSA indexer, hyper-connections and a 512-expert MoE, which this engine "
     "does not implement",
     true, true, false, false},

    // A head set is not a model: no token embedding, no output head, nothing to
    // continue a prompt with. `--draft` accepts one anyway, because a DFlash
    // drafter is the *cheaper* half of the same feature — the head set is 0.5-1
    // GiB against a full model of the target's family, and it reads the target's
    // hidden states through the encoder in src/dflash.cpp rather than starting
    // from tokens.
    //
    // The drafting-relevant flags are the ones the DFlash stack shares with
    // laguna, whose schema it was trained against: per-head QK-norm, the
    // separate softplus output gate (`attn_gate.weight`, per-head or
    // per-element), and the NeoX rope pairing. On a draft entry they also say
    // "the gate tensor is served", which is what keeps them out of the gap map.
    {"dflash", ArchShape::Dense, ArchSupport::Yes, ArchRole::Draft, "dflash", "",
     true, false, false, false, true, true},
    // The same object under the name an older converter gave it: `dflash.*`
    // tensor names, the per-aux norms stored one per layer instead of stacked,
    // and the behavioural switches spelled out as flags rather than implied by
    // `decoder_arch`. src/dflash.cpp reads both; this is the entry that makes the
    // older one recognizable.
    {"qwen35-dflash-draft", ArchShape::Dense, ArchSupport::Yes, ArchRole::Draft,
     "dflash", "",
     true, false, false, false, true, true},
    // Poolside's own Laguna export, which is the third spelling of the same head
    // set: `dflash-draft.*` metadata keys and underscore tensor names
    // (`dflash_fc.weight`, `dflash_hidden_norm.weight`,
    // `dflash_aux_hidden_norm.N.weight`), with the FFN norm stored as
    // `post_attention_norm`. src/dflash.cpp reads all three spellings from one
    // loader, so this entry exists to stop the name being refused before it ever
    // reaches that loader -- the same flags, because the tensors are the same.
    {"dflash-draft", ArchShape::Dense, ArchSupport::Yes, ArchRole::Draft,
     "dflash", "",
     true, false, false, false, true, true},
    // DSpark is a different head family (Markov/diffusion heads over the
    // target's nextn state) that nothing here reads yet, so it stays refused
    // rather than being claimed by the DFlash loader.
    //
    // The message describes the schema the files actually carry, not a
    // `dspark.*` prefix: `fc.weight` over the target-layer stack, a `blk.N`
    // stack whose attention is non-causal over a draft block of
    // `dspark.block_size` rows, `markov.w1/w2` (a rank-256 bigram head) and
    // `confidence.proj`. The block count is per file (5 and 6 on this box), so
    // it is not stated here. What is missing is the forward, not a tensor.
    {"dspark", ArchShape::Dense, ArchSupport::No, ArchRole::Draft, "qwen35",
     "a dspark draft-block head set: fc over the target-layer stack, a blk.N "
     "stack with non-causal attention over a block of draft rows, a markov "
     "bigram head and a confidence head -- a parallel block drafter, which this "
     "engine does not implement (dflash, its one-block cousin, is served)",
     false, false, false, false},
    // Laguna (poolside): MoE with a sigmoid router, a per-expert selection
    // bias added to the probabilities before the top-k, a per-head softplus
    // gate on the attention output, and a hybrid full/sliding-window stack
    // whose two layer kinds run different rotary tables (YaRN over 64 of 128
    // dims on the full layers, plain RoPE over all 128 on the windowed ones).
    //
    // The schema was read off Laguna-S-2.1-UD-Q4_K_M and cross-checked against
    // llama.cpp's own src/models/laguna.cpp, which is the reference for every
    // piece of it; `attention.head_count` there is an array (48 heads on the
    // full layers, 72 on the windowed ones), which is what makes the query
    // projection a per-layer width and why the loader no longer insists that
    // n_embd divide by it.
    //
    // The loader reads all of that today: the two head counts, the gate tensor,
    // the bias, the YaRN table folded into inv_freq and the window geometry.
    // What is still missing is the engine side — the gate projection and its
    // broadcast, the sigmoid+bias router, and the window mask in the attention
    // kernels — so the entry stays a refusal until those land, because the
    // alternative is a file that loads and decodes plausible nonsense.
    // docs/TODO.md has the remaining list.
    // Its rope is the NeoX pairing as well (llama.cpp's table lists laguna
    // under LLAMA_ROPE_TYPE_NEOX), which is what the last flag says.
    {"laguna", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target, "laguna",
     "",
     true, false, false, false, true, true},
    // gpt-oss (openai-moe): the schema was read off the 120B file here -- 36
    // layers, 64 query heads of 64 dims over 8 KV heads, a 128-token window on
    // the *even* layers with full attention on the odd ones, YaRN 32x over a
    // 4096 original context at base 150000 for every layer, 128 experts top-4,
    // attention sinks, an attention-output bias, and a bias on all three
    // expert FFN projections. Each of those was a silent-wrong-answer gap
    // before the loader read it:
    //   * the windowed layers rotated at the 10000 default because only
    //     `rope.freq_base_swa` was consulted, and this file states one base.
    //   * `attn_output.bias` and `ffn_{gate,up,down}_exps.bias` were never
    //     uploaded, and the MoE's `ffn_gate_inp.bias` was read as laguna's
    //     post-softmax selection bias -- the wrong side of the softmax.
    //   * the window pattern was laguna's (full layer first, period 4).
    // The shape is Moe; qk_norm and rope_neox are true here because the file
    // carries per-head q/k norms and rotates with the half-split pairing.
    //
    // Yes, and the last gap is closed rather than waved through: the row used
    // to say "coherence at depth is not yet pinned against a CPU oracle", and
    // test_gpt_oss_whole_graph is that oracle. It builds a four-layer file
    // carrying every tensor listed above, pins the logits row after four greedy
    // tokens, and then REMOVES each piece in turn (sinks, attn_output.bias,
    // ffn_gate_inp.bias, the per-expert FFN biases, the window): every removal
    // moves the pinned row, so the pin is a measurement of the whole graph and
    // not of a subset that happens to decode. It also pins the two properties a
    // depth test needs to mean anything -- the same file pinned twice is the
    // same row, and one token per prefill chunk gives that row again, bit for
    // bit, at 1, 2, 3 and 4 generated tokens.
    {"gpt-oss", ArchShape::Moe, ArchSupport::Yes, ArchRole::Target, "gpt-oss",
     "",
     false, false, false, false, false, true, 2, true, 0, true},
    // MoVA: the attention *values* are a routed expert set of their own, on top
    // of an attention output gate.
    //
    // The gate is NOT part of the gap. The engine loads attn_gate.weight behind
    // ArchSpec::attn_gate, softplus's it and broadcasts it over the attention
    // result before wo -- code driven by the schema flag, not by which arch
    // asked for it, and laguna has been running through it all along. Listing
    // it here was a false blocker: it made this entry look further from working
    // than it is, and a wrong diagnosis costs the next person a kernel to write.
    // The real gap is the routed expert set on the attention values, and only
    // that.
    //   * MoVA: the attention VALUES are a routed expert set of their own
    //     (`attn_v_gate` + its optional bias + `attn_v_exps`), which is
    //     Engine::mova_value -- selection on the BIASED probability, mixing with
    //     the UNBIASED one, silu on each expert's output before the weight.
    //   * a GROUPED RMS norm (`attention.group_norm_groups`): the reduction is
    //     over each band of the row, the weight is elementwise over the whole
    //     row. That is Backend::rmsnorm_bands, and it is NOT rmsnorm_grouped,
    //     which takes a separate weight vector per group.
    // Both are implemented and both are refused BY CAPABILITY on a backend that
    // lacks the op, so a device path cannot silently run the ungrouped norm.
    // What is left is the vision tower on the multimodal checkpoints; the text
    // stack runs end to end and test_k2_horizon_mova pins it.
    [] {
        ArchSpec s{"k2-horizon", ArchShape::Moe, ArchSupport::Yes,
                   ArchRole::Target, "k2-horizon", "", false, false, false,
                   false, true, false};
        // llama.cpp marks k2-horizon's wqkv_gate TENSOR_NOT_REQUIRED, so an
        // ablated export is a legitimate file and must load ungated.
        s.attn_gate_optional = true;
        return s;
    }(),
    // MiniMax's sparse-attention MoE. Listed rather than left to the fallback
    // because the fallback reads an unlisted file against llama defaults, and a
    // 428B MoE is the last model that should be run through a schema nobody
    // checked: the defaults would find a plausible layer stack and quietly
    // compute the wrong graph instead of refusing.
    //
    // Three gaps, all of them structural rather than a missing flag. MiniMax
    // Sparse Attention is block-sparse throughout -- M3.1 replaced the last
    // full-attention layers, so there is no dense prefix to fall back on and
    // the attention op's causal/window masks cannot express it. The model is
    // natively multimodal, so a vision/video tower feeds the token stream. And
    // it ships a DSpark draft head, which this engine can load as a draft file
    // against a matching target but which has nothing to attach to here.
    {"minimax-m3", ArchShape::Moe, ArchSupport::No, ArchRole::Target, "minimax-m3",
     "MiniMax Sparse Attention (block-sparse, no dense prefix), a native "
     "vision/video tower and a DSpark draft head are not implemented",
     false, false, false, false},
    // muse-glimmer (dense): four pieces of its own around llama's block, read
    // off llama.cpp's src/models/muse-glimmer.cpp.
    //   * a WEIGHTLESS RMS norm on the embedding before layer 0 (`build_norm(
    //     inpL, nullptr, ...)`), which no tensor in the file describes;
    //   * an attention output gate projected from the pre-attention row,
    //     sigmoid'd, and multiplied ELEMENTWISE into the attention result
    //     before wo -- the same tensor spark2_5 uses at one value per head, so
    //     the width is what tells the two graphs apart;
    //   * a post norm on each branch before it joins the residual, with its own
    //     eps (1e-8 against a 1e-5 body);
    //   * RoPE on the sliding-window layers only -- the full-attention layers
    //     rotate nothing, the opposite of the usual arrangement.
    [] {
        // `gemma_spec` is not reused: this family's norms are gemma's in SHAPE
        // and not in position, and its FFN is SwiGLU where gemma's is GeGLU.
        ArchSpec s{"muse-glimmer", ArchShape::Dense, ArchSupport::Yes,
                   ArchRole::Target, "llama", "",
                   /*qk_norm=*/true, /*q_output_gate=*/false,
                   /*qkv_bias=*/false, /*gemma_norm=*/false,
                   /*attn_gate=*/true, /*rope_neox=*/false};
        s.attn_gate_sigmoid = true;
        s.attn_gate_wide = true;
        s.post_branch_norm = true;
        s.post_norm_eps = 1e-8f;
        s.embd_rms_noweight = true;
        s.rope_swa_only = true;
        // The window pattern is `load_swa_pattern(ml, 4)`, whose default is
        // dense_first = false -- layers 0..2 windowed, layer 3 full. The file
        // MAY spell the pattern out as an array (`attention.sliding_window_pattern
        // = [1,1,1,0,...]`), and the loader reads that first; this flag is the
        // fallback for a file that declares only the window width. Without it
        // the windowed set was the other three layers, and since this family
        // rotates ONLY on its windowed layers, that is a different model: the
        // rotating layers and the windowed layers would both be wrong at once.
        s.swa_window_first = true;
        // The head's output is scaled by `<arch>.logit_scale` before the
        // (optional) final-logit softcap. Read per file into
        // ModelConfig::logit_scale and applied in Engine::logit_head.
        return s;
    }(),
    // spark2_5: llama's block with a fused `attn_qkv` row and a per-head
    // sigmoid gate on the attention output (before wo), GeGLU-free but GELU
    // gated FFN, NeoX rope, and a sliding window on its own pattern. The gate
    // is `attn_gate.weight` at one value per head -- muse-glimmer's tensor at a
    // different width, which is why the two entries differ by one flag.
    [] {
        ArchSpec s{"spark2_5", ArchShape::Dense, ArchSupport::Yes,
                   ArchRole::Target, "llama", "",
                   /*qk_norm=*/false, /*q_output_gate=*/false,
                   /*qkv_bias=*/false, /*gemma_norm=*/false,
                   /*attn_gate=*/true, /*rope_neox=*/true};
        s.fused_qkv = true;
        s.attn_gate_sigmoid = true;
        s.gelu_ffn = true;
        return s;
    }(),
    // LLaDA-family masked-diffusion model: bidirectional attention over the
    // whole sequence, a block denoising schedule, and MoE expert routing with
    // group-limited selection and a routing bias. See docs/LLaDA.md for the
    // schema read off LLaDA2.2-flash and the reference decoding loop. The three
    // pieces this engine lacks are named here; the bidirectionality itself is
    // already reachable (the attention op has always taken a causal flag).
    // The SAMPLER is done: src/diffusion.cpp is the reference's block schedule
    // (base-plus-remainder transfer, candidates restricted to this block and to
    // masked positions, confidence top-k, no rewriting of committed tokens), and
    // test_diffusion_sampler drives it with a forward the test owns so every
    // commitment is predictable. What is NOT done is everything around it: the
    // `diffuse.*` loader schema (fused attn_qkv split, post_attn_norm mapped to
    // the pre-FFN norm, moe_gate/moe_experts_* onto the MoE slots), the
    // group-limited routing rule with its selection bias, and the cacheless
    // bidirectional forward wired into Engine::generate. The row stays No and
    // says so, in that order -- docs/LLaDA.md is the plan and now has one item
    // fewer on it.
    {"diffuse", ArchShape::Diffusion, ArchSupport::No, ArchRole::Target, "diffuse",
     "the block-diffusion sampler runs (src/diffusion.cpp, pinned by "
     "test_diffusion_sampler); the diffuse.* loader schema, the group-limited "
     "expert routing with its bias, and the cacheless bidirectional forward in "
     "Engine::generate are not implemented yet (docs/LLaDA.md)",
     true, false, false, false},
};

// Tensor names that imply an unimplemented computation, with the piece they
// name. Matched as a substring so `blk.17.attn_v_gate.bias` and a hypothetical
// unlayered spelling both hit.
struct GapTensor {
    const char *needle;
    const char *gap;
    // When true the entry only applies to a non-recurrent (dense/MoE) arch:
    // the gated delta net has its own gate tensor of the same name.
    bool dense_only;
    // When set, an arch whose schema carries this flag IS served, so the gap
    // does not fire on it.
    //
    // This replaced a hardcoded arch name, which is the wrong shape for the
    // question: whether a tensor is implemented depends on whether the schema
    // turns the code on, not on which architecture happens to be first. Naming
    // `laguna` exempted laguna and nothing else, so k2-horizon -- which carries
    // the same gate tensor and the same existing code path -- was refused for a
    // feature the engine has. A name list cannot fix itself when a second arch
    // shows up.
    enum class Served : unsigned char {
        Never,
        ByAttnGateFlag,
        ByMlaFlag,
        ByGpt2Flag,
        ByFusedQkvFlag
    };
    Served served = Served::Never;
};

constexpr GapTensor kGapTensors[] = {
    // On a recurrent arch `attn_qkv` is the gated delta net's own projection and
    // is what the loader uploads; on a dense one it is a fused attention
    // projection this engine has no shape for.
    // Served when the schema is gpt2-shaped: that loader binds the fused
    // projection and Engine::gpt2_forward splits the Q|K|V row. Anywhere else
    // it is still a fused attention projection this engine has no shape for.
    {"attn_qkv.weight", "a fused query/key/value projection (attn_qkv)", true,
     GapTensor::Served::ByFusedQkvFlag},
    // Implemented: src/model.cpp loads attn_gate.weight behind
    // ArchSpec::attn_gate and src/engine.cpp softplus-gates and broadcasts it
    // over the attention result before wo.
    {"attn_gate.weight", "an attention output gate (attn_gate)", true,
     GapTensor::Served::ByAttnGateFlag},
    // DeepSeek-V4.1's four-lane Sinkhorn hyper-connections. These names are
    // V4.1's own and are NOT qwen4exp's mixer, which this engine does load
    // (hc_attn_{norm,down,up,inject}), so the needles are the six tensor stems
    // rather than a bare `hc_` prefix. A file carrying them is a residual the
    // whole stack runs through: loading it without the block would decode a
    // model whose residual is not the one it was trained with, which is the
    // "mis-loaded" failure the deepseek41 row promised not to make. Measured
    // before this entry existed: the same fixture loaded with model.load=1 and
    // no blocker at all (`test_pending_model_fixtures`, ds41-hc arm).
    {"hc_attn_fn.", "Sinkhorn hyper-connections (hc_attn_fn/hc_ffn_fn)", false},
    {"hc_ffn_fn.", "Sinkhorn hyper-connections (hc_attn_fn/hc_ffn_fn)", false},
    {"hc_attn_base.", "Sinkhorn hyper-connections (hc_attn_base/hc_ffn_base)", false},
    {"hc_ffn_base.", "Sinkhorn hyper-connections (hc_attn_base/hc_ffn_base)", false},
    {"hc_attn_scale.", "Sinkhorn hyper-connections (hc_attn_scale/hc_ffn_scale)", false},
    {"hc_ffn_scale.", "Sinkhorn hyper-connections (hc_attn_scale/hc_ffn_scale)", false},
    // Engram: a token-hash conditional-memory lookup whose result is injected
    // into the residual. V4.1-Flash usually ships the tables as their own GGUF,
    // but a repack that inlines them is still a V4.1 file, and the lookup is
    // part of the layer it sits in -- not a side table the engine may skip.
    {"engram_", "Engram conditional memory (engram_*)", false},
    // SERVED, not gaps: attn_v_exps/attn_v_gate are MoVA, implemented by
    // Engine::mova_value (see the k2-horizon row). They were the row's whole
    // stated reason for No, and a refusal for a feature the engine now runs is
    // worse than no entry at all -- it would refuse every k2-horizon file. The
    // one thing that still refuses is a backend without the grouped-norm op,
    // which is a capability check at init and not a tensor name.
    {"attn_kv_a.weight", "latent KV compression (MLA: attn_kv_a/attn_kv_b)", false,
     GapTensor::Served::ByMlaFlag},
    {"attn_kv_b.weight", "latent KV compression (MLA: attn_kv_a/attn_kv_b)", false,
     GapTensor::Served::ByMlaFlag},
    {"attn_kv.weight", "latent KV compression (MLA: attn_kv_a/attn_kv_b)", false,
     GapTensor::Served::ByMlaFlag},
    // NOT a gap: the router selection bias is loaded and applied. It was a gap
    // only because the loader looked for `ffn_exp_probs_b.bias` while the files
    // that carry it spell it `exp_probs_b.bias` -- so it was loaded for nothing,
    // and the engine's own consumer dereferenced a device pointer as if it were
    // host memory on the path that could never be reached. Both are fixed, so
    // this entry is gone rather than narrowed: a refusal for a feature that
    // exists sends the next reader looking for code to write.
    {"ssm_conv1d.weight", "a Mamba-style SSM block (ssm_conv1d)", true},
    {"ssm_out.weight", "a Mamba-style SSM block (ssm_out)", true},
    // LFM2's shortconv used to be named here. It is gone rather than narrowed
    // for the same reason as the router bias above: the loader now binds
    // shortconv.in_proj/conv/out_proj per layer, `Engine::lfm_conv_forward`
    // runs them, and a refusal for a feature that exists sends the next reader
    // looking for code to write.
    // Expert routing this engine does not implement. The names are the only
    // evidence a file needs: group-limited selection and a routing bias change
    // *which* experts run, so a model carrying them would decode differently
    // under the plain top-k rule, silently.
    {"moe_gate_bias.weight", "a routed-expert selection bias (moe_gate_bias)",
     false},
    {"moe_experts_gate.weight", "group-limited expert routing (moe_gate/n_group)",
     false},
};

// Head tensors that can ride along inside a target. Not gaps: the model runs,
// it just does not get the extra prediction from them yet.
struct HeadTensor {
    const char *needle;
    const char *what;
};

constexpr HeadTensor kHeadTensors[] = {
    {"dflash.", "dflash speculative head"},
    {"dspark.", "dspark speculative head"},
    {"mtp.", "multi-token-prediction head"},
};

bool has(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

} // namespace

const ArchSpec *arch_lookup(std::string_view name) {
    for (const ArchSpec &s : kTable) {
        if (name == s.name) return &s;
    }
    return nullptr;
}

const ArchSpec &arch_spec(std::string_view name, bool *is_known) {
    const ArchSpec *s = arch_lookup(name);
    if (is_known) *is_known = (s != nullptr);
    return s ? *s : kFallback;
}

std::string arch_known_names() {
    std::string out;
    for (const ArchSpec &s : kTable) {
        if (!out.empty()) out += ", ";
        out += s.name;
    }
    return out;
}

const char *arch_shape_name(ArchShape s) {
    switch (s) {
        case ArchShape::Dense: return "dense";
        case ArchShape::Moe: return "moe";
        case ArchShape::Recurrent: return "recurrent";
        case ArchShape::RecurrentMoe: return "recurrent-moe";
        case ArchShape::Diffusion: return "diffusion";
    }
    return "?";
}

const char *arch_support_name(ArchSupport s) {
    switch (s) {
        case ArchSupport::Yes: return "supported";
        case ArchSupport::Partial: return "partial";
        case ArchSupport::No: return "not supported";
    }
    return "?";
}

const char *arch_role_name(ArchRole r) {
    switch (r) {
        case ArchRole::Target: return "target";
        case ArchRole::Draft: return "draft";
    }
    return "?";
}

const char *arch_tensor_gap(std::string_view tensor_name, ArchShape shape,
                            std::string_view arch) {
    const bool dense = shape == ArchShape::Dense || shape == ArchShape::Moe;
    for (const GapTensor &g : kGapTensors) {
        if (g.dense_only && !dense) continue;
        if (g.served == GapTensor::Served::ByAttnGateFlag &&
            arch_spec(arch).attn_gate)
            continue;
        if (g.served == GapTensor::Served::ByMlaFlag && arch_spec(arch).mla)
            continue;
        if (g.served == GapTensor::Served::ByGpt2Flag &&
            arch_spec(arch).gpt2_style)
            continue;
        if (g.served == GapTensor::Served::ByFusedQkvFlag &&
            arch_spec(arch).fused_qkv)
            continue;
        if (has(tensor_name, g.needle)) return g.gap;
    }
    return nullptr;
}

const char *arch_tensor_head(std::string_view tensor_name) {
    for (const HeadTensor &h : kHeadTensors) {
        if (has(tensor_name, h.needle)) return h.what;
    }
    return nullptr;
}

} // namespace krk
