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
constexpr ArchSpec gemma_spec(const char *name, ArchSupport sup, const char *why,
                              int swa_period, bool window_first) {
    ArchSpec s{name,       ArchShape::Dense, sup,     ArchRole::Target,
               "gemma",    why,              /*qk_norm=*/true,
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

constexpr ArchSpec kTable[] = {
    // ---- the schemas this engine runs ------------------------------------
    {"llama", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target, "llama", "",
     false, false, false, false},
    phi3_spec(),
    // The plain pre-norm family: rotary (or learned) positions, LayerNorm and a
    // fused QKV, differing only in which of the fine flags above they set.
    gptneox_spec(),
    gptj_spec(),
    phi2_spec(),
    mpt_spec(),
    bloom_spec(),
    falcon_spec(),
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
    {"lfm2moe", ArchShape::RecurrentMoe, ArchSupport::Yes, ArchRole::Target,
     "lfm2", "",
     /*qk_norm=*/true, /*q_output_gate=*/false, /*qkv_bias=*/false,
     /*gemma_norm=*/false, /*attn_gate=*/false, /*rope_neox=*/true,
     /*swa_period=*/0, /*swa_window_first=*/false, /*expert_w_norm=*/1},
    // Gemma scales the embedding by sqrt(n_embd), uses a tighter eps, gates its
    // FFN with GeGLU and softcaps the final logits. Gemma 1 is the llama schema
    // plus those four. It is not in the collections here, so the entry says so
    // rather than claiming a validated run.
    gemma_spec("gemma", ArchSupport::Partial,
               "not validated against a file in this collection; the schema is "
               "llama plus embedding scaling, GeGLU and a final-logit softcap",
               0, false),
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
    gemma_spec("gemma2", ArchSupport::Partial,
               "the attention-logit softcap is applied by the reference "
               "attention (--cpu, --npu, --npu-cpu) but not by the HIP "
               "kernels, which refuse the run by name; the alternating window, "
               "GeGLU and the final-logit softcap are implemented; not "
               "validated against a gemma2 file in this collection",
               2, true),
    // Gemma 3: QK-norm, GeGLU, embedding scaling, a final-logit softcap and a
    // 5:1 local/global pattern (period 6, windowed layers first, 1024-token
    // window declared in the file). Every one of those pieces is implemented
    // and driven by schema flags rather than by arch name; the remaining doubt
    // is a validated run, which this collection has no gemma3 file for.
    gemma_spec("gemma3", ArchSupport::Partial,
               "the loader reads its whole schema (QK-norm, GeGLU, the 5:1 "
               "windowed/global pattern, embedding scaling, final-logit softcap); "
               "not validated against a file in this collection",
               6, true),
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
    gemma_spec("gemma4", ArchSupport::No,
               "per-layer input embeddings, shared-KV layers, two attention "
               "geometries (global + sliding window) and the attention-logit "
               "softcap are not implemented; the per-layer "
               "attention.head_count_kv array, GeGLU and the final-logit softcap "
               "are",
               0, false),

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
    // a QSA indexer (4 heads x 128, top_k 2048) and a per-layer `compress_ratios`,
    // a 288-expert top-10 MoE with a shared expert, a 4-lane 10240-wide
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
    // the first `hash_layer_count` layers select by token id. What does not: the
    // CSA/HCA compressors (attn_compressor_*), the sparse indexer and
    // attn_sinks -- so a file whose attention.compress_ratios is non-zero is
    // refused at Engine::init, and one carrying any of those tensor classes is
    // refused by the loader's gap map. A dense-core file (all ratios 0) IS this
    // graph, and test_dsv4_whole_graph pins it end to end.
    {"deepseek4", ArchShape::Moe, ArchSupport::Partial, ArchRole::Target, "deepseek4",
     "shared-latent MLA, the grouped output and the tid2eid hash route run; "
     "compressed attention (attn_compressor_*), the sparse indexer and "
     "attn_sinks do not, so a file with a non-zero compress_ratios is refused",
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
    // What this does NOT do is make V4.1 run. A V4.1 checkpoint is 343B-763B
    // with compress_ratios non-zero on most layers, so Engine::init refuses it
    // by name exactly as it refuses deepseek4, and the loader's gap map names
    // the tensors it could not take. The entry exists so the refusal says
    // which graph it read and what is missing from it.
    {"deepseek41", ArchShape::Moe, ArchSupport::Partial, ArchRole::Target,
     "deepseek4",
     "recognised as the deepseek4 graph: shared-latent MLA, the grouped output "
     "and the tid2eid hash route; compressed attention, the sparse indexer and "
     "hyper-connections are not implemented, so a V4.1 checkpoint with "
     "compress_ratios or hc_* tensors is refused by name, not mis-loaded",
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
    {"nemotron_h", ArchShape::Recurrent, ArchSupport::Yes, ArchRole::Target,
     "nemotron_h",
     "",
     false, false, false, false},
    // Nemotron MoE: alternating Mamba-2 SSM blocks + shared experts.
    // The SSM blocks are loaded by Model::load - the per-layer ssm_* upload,
    // keyed on the tensors a layer actually carries - and run by the mamba2_ssd
    // kernels in src/hip/kernels/gdn.hpp, launched from backend_hip and driven
    // by Engine::mamba2_forward. The architecture uses:
    //   - Even layers: ssm_in, ssm_out, ssm_conv1d, ssm_a, ssm_d, ssm_dt, ssm_norm
    //   - Odd layers: ffn_up_shexp, ffn_down_shexp (shared experts, NVFP4)
    // This is different from GDN: no gate, simpler SSM, different tensor layout.
    {"nemotron_h_moe", ArchShape::RecurrentMoe, ArchSupport::Yes, ArchRole::Target,
     "nemotron_h",
     "",
     false, false, false, false, false, true},
    // ---- speculative head sets -------------------------------------------
    // An MTP head set, extracted by tools/mtp_extract.py: the `mtp.*` tensors a
    // checkpoint ships beside its own weights -- a norm pair, two input
    // projections and one block with its own attention (packed query gate,
    // per-head QK-norm, a QSA indexer) and its own 512-expert MoE.
    //
    // Refused, and the refusal names what would have to exist first. It is not
    // a drafter this engine could run even if the block were implemented: its
    // input is the *target's* hidden state plus the last token's embedding, and
    // the only targets that carry one are qwen4exp -- which this table already
    // refuses. Claiming it would mean proposing tokens from a block whose
    // inputs never arrive.
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
    {"gpt-oss", ArchShape::Moe, ArchSupport::Partial, ArchRole::Target,
     "gpt-oss",
     "the loader reads its whole schema (sinks, attention-output bias, "
     "per-expert FFN biases, the logit-side router bias, the alternating "
     "window and the one-base YaRN rope); coherence at depth is not yet "
     "pinned against a CPU oracle",
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
    {"k2-horizon", ArchShape::Moe, ArchSupport::No, ArchRole::Target, "k2-horizon",
     "attention values are a routed expert set of their own "
     "(attn_v_exps/attn_v_gate); that is not implemented",
     false, false, false, false, true, false},
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
    // Dense FFN with an attention output gate and Gemma-style post-norms.
    {"muse-glimmer", ArchShape::Dense, ArchSupport::No, ArchRole::Target,
     "muse-glimmer",
     "the attention output gate and the post-attention/post-FFN norms are not "
     "implemented",
     true, false, false, false},
    // One fused projection for q/k/v plus a gate on the attention output.
    {"spark2_5", ArchShape::Dense, ArchSupport::No, ArchRole::Target, "spark2_5",
     "a fused attn_qkv projection and an attention output gate are not "
     "implemented",
     false, false, false, false},
    // LLaDA-family masked-diffusion model: bidirectional attention over the
    // whole sequence, a block denoising schedule, and MoE expert routing with
    // group-limited selection and a routing bias. See docs/LLaDA.md for the
    // schema read off LLaDA2.2-flash and the reference decoding loop. The three
    // pieces this engine lacks are named here; the bidirectionality itself is
    // already reachable (the attention op has always taken a causal flag).
    {"diffuse", ArchShape::Diffusion, ArchSupport::No, ArchRole::Target, "diffuse",
     "block-diffusion sampling, fused attn_qkv/qk-norm/post-attn-norm loading and "
     "group-limited expert routing with a routing bias are not implemented yet "
     "(docs/LLaDA.md)",
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
    {"attn_v_exps.weight", "routed experts on the attention values (attn_v_exps)",
     false},
    {"attn_v_gate.", "a gate on the attention values (attn_v_gate)", false},
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
