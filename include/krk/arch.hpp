// ============================================================================
//  arch.hpp — what a `general.architecture` string means to this engine.
//
//  A GGUF names its architecture in one metadata string and then keys every
//  other key off that name (`qwen35.block_count`, `laguna.rope.freq_base`, ...),
//  so that string is the piece of metadata that decides how the rest of the file
//  is read. This is the single table that answers it: which schema an arch is
//  loaded against, which optional pieces its tensors carry, and whether this
//  engine can actually run it or only recognize it.
//
//  Before this existed the same question was answered by string compares spread
//  across the loader and the engine — `qwen35` was spelled out in two files, and
//  an arch nobody had listed still loaded by falling through to llama defaults
//  without anyone being told. Both are now the table's business.
//
//  Three states, deliberately:
//
//    Yes      the loader and the kernels implement the schema; the table's flags
//             (QK-norm, packed query gate, qkv bias, Gemma norms) are the only
//             geometry the loader cannot read off the tensors themselves.
//    Partial  the file loads and runs, and something named in `why` is silently
//             left out — a tower that is never read, or a head set for a mode
//             this engine does not have. The user is warned because the gap is
//             real, and it is not fatal because the part that runs is the part
//             the model is used for.
//    No       the file is refused, with `why` as the error. This is for archs
//             whose tensors name a computation the engine does not implement:
//             loading them would not fail, it would decode nonsense, and a
//             nonsense answer is worse than an honest refusal.
//
//  An arch that is not in the table at all is *not* refused. Most variants of
//  the llama schema load fine, so the loader tries the conservative default and
//  says it did — "not in the table" and "does not work" are different claims.
//
//  `arch_tensor_gap` is the enforcement arm of Partial/No: it maps a tensor name
//  to the unimplemented piece it implies, so a file that carries `attn_qkv` or
//  `attn_v_exps` is refused for the reason that names it, whether or not its
//  `general.architecture` was ever added here. The table classifies a family;
//  the gap map catches an individual checkpoint.
//
//  Two orthogonal questions live in this table beyond support. One is role: a
//  file can be a model you run or a speculative head set (`dflash`, `dspark`,
//  an MTP shard) that only means something next to a target. The other is where
//  those heads sit: standalone files get their own arch string and are refused
//  as targets, while a target with the heads *built in* keeps its own arch and
//  carries the extra blocks as `blk.N` past its layer count (`nextn_predict_
//  layers`) or as `dflash.*`/`dspark.*`/`mtp.*` tensors. The first is a refusal,
//  the second is a note — the model runs either way (`arch_tensor_head`).
// ============================================================================
#pragma once

#include <string>
#include <string_view>

namespace krk {

enum class ArchShape {
    Dense,        // attention + FFN
    Moe,          // attention + routed experts (top-k)
    Recurrent,    // gated delta net blocks interleaved with attention blocks
    RecurrentMoe, // both, as Qwen3.5-MoE does
    // Masked diffusion: bidirectional attention and a block denoising schedule
    // instead of a causal left-to-right decode (LLaDA, docs/LLaDA.md).
    Diffusion,
};

enum class ArchSupport {
    Yes,     // loader and kernels implement this schema
    Partial, // loads and runs; `why` names what is left out
    No,      // refused; `why` is the error
};

enum class ArchRole {
    Target, // a model you run
    Draft,  // a speculative head set; only means something beside a target
};

struct ArchSpec {
    const char *name; // the `general.architecture` string, exactly
    ArchShape shape;
    ArchSupport support;
    ArchRole role;
    const char *family; // the schema it is read against (informational)
    const char *why;    // the gap, when support is not Yes
    // Geometry the loader would otherwise have to guess or re-derive.
    bool qk_norm;       // attn_q_norm / attn_k_norm tensors, per layer
    bool q_output_gate; // attn_q packs a per-head output gate (Qwen3.5)
    bool qkv_bias;      // attn_q / attn_k / attn_v carry a bias
    bool gemma_norm;    // embedding scaled by sqrt(n_embd), eps 1e-6
    // A *separate* per-head output gate tensor (`attn_gate.weight`, laguna):
    // projected from the same hidden state q/k/v read, softplus'd, and
    // multiplied into the attention result before wo. Not the packed-query
    // gate above — different tensor, different activation, different operand.
    bool attn_gate = false;
    // The RoPE pair convention the file's q/k rows were exported in.
    // llama.cpp calls the two "norm" and "neox": norm pairs the *adjacent*
    // channels (2i, 2i+1), neox pairs channel j with j + n_rot/2 inside the
    // rotated prefix. Only the relative rotation between a query and a key
    // reaches the score, so the wrong one mixes the wrong channels and every
    // token still lands on the same order of magnitude — a plausible wrong
    // answer rather than a crash. The convention is a property of the
    // architecture (llama.cpp's llm_arch_rope table), not of the tensor, so it
    // has to be declared here. Absent means norm, which is what every file in
    // this collection used before the flag existed.
    bool rope_neox = false;
    // Sliding-window layer pattern, when the file alternates layer kinds. 0
    // keeps the loader's historical default (period 4, the *full* layer first,
    // which is laguna's stack). gpt-oss alternates every layer the other way
    // round -- windowed on the even indices, full attention on the odd ones --
    // so it declares period 2 with the window first. Both numbers travel to
    // ModelConfig because `is_swa()` is the only reader.
    int swa_period = 0;
    bool swa_window_first = false;
    // Expert weight normalization when the FILE says nothing. llama.cpp's
    // `expert_weights_norm` defaults to false and its openai-moe loader never
    // sets it, so gpt-oss's router probabilities are used as they come out of
    // the softmax over all 128 experts: top four summing to ~0.13, not 1. This
    // engine normalized unconditionally before the tri-state existed, and on a
    // 128-expert stack whose top-4 hold 13% of the mass that multiplies every
    // layer's expert contribution by ~7.6 -- the residual grows ~1.2x per
    // layer instead of staying flat, and the answer is garbage that looks like
    // a broken kernel. -1 keeps the file's own key (and the legacy default).
    int expert_w_norm = -1;
    // The expert FFN activation is gpt-oss's SwiGLU-OAI rather than silu*up.
    bool swiglu_oai = false;
    // ---- qwen4exp (Qwen3.8-Flash-Next) -----------------------------------
    // A four-lane hyper-connection residual instead of `x + f(norm(x))`, a PLE
    // n-gram hash embedding table read per token, and a QSA indexer that
    // restricts a full-attention layer to whole blocks of `compress_ratio`
    // tokens. The gate tensor on those layers is per *element* (n_head*hd),
    // not per head, and there is no output norm: the final hc_head mixer is it.
    bool hyper_connection = false;
    bool ple = false;
    bool qsa = false;
    bool attn_gate_elem = false;
    bool no_out_norm = false;
    // ---- GPT-2 (gpt2) -----------------------------------------------------
    // The pre-RMSNorm, pre-rotary family, and four differences that all have to
    // travel together because they are one architecture rather than four
    // options -- a file that had some and not others does not exist, and
    // splitting them into separate booleans would let a future reader set an
    // impossible combination:
    //
    //   * LayerNorm (mean-subtracting, with a bias) in place of RMSNorm, so
    //     every norm in the stack carries a `bias` tensor beside its weight.
    //   * LEARNED absolute position embeddings (`position_embd.weight`,
    //     [n_ctx_train, n_embd]) summed into the token embedding. There is no
    //     rotary table at all; rotating here would be silently wrong because
    //     positional information has already been added by the time attention
    //     runs.
    //   * a FUSED `attn_qkv` projection, [n_embd, 3*n_embd], whose output is
    //     three contiguous blocks (Q | K | V) of n_embd each, plus a single
    //     3*n_embd bias. The split is a stride, not a gather.
    //   * a PLAIN GELU MLP: up -> gelu -> down. The gated `silu(gate)*up`
    //     shape the other families use has no gate tensor here, and reading
    //     ffn_up as a gate would halve the intermediate width.
    // ---- MLA (DeepSeek-V2/V3/V4) -----------------------------------------
    // Multi-head latent attention: the K/V a layer attends is not per-head
    // rows but one compressed latent (cKV, kv_lora_rank wide) plus a
    // decoupled RoPE key (qk_rope_head_dim), with the per-head V projection
    // absorbed into the output at load. Ranks live in ModelConfig; the flag
    // only says the schema carries them.
    bool mla = false;
    // ---- DeepSeek-V4 (shared-latent MLA + sparse extras) ------------------
    // The dense core is still one shared latent (wkv/attn_kv_norm) with
    // per-head queries and a grouped output (wo_a/wo_b), but the stack adds
    // compressed-attention layers (compress_ratios), a Sinkhorn
    // hyper-connection, a tid2eid hash route and YaRN-on-compressed-layers
    // rope. The flag says the loader reads that metadata; the engine still
    // refuses until each piece runs.
    bool dsv4 = false;
    // ---- is this arch's ATTENTION implemented? ---------------------------
    // A separate question from `support`, because the two fail differently.
    // `support` is the arch table's verdict on the file as a whole; the
    // loader honours it, and KRK_ARCH_ALLOW_EXPERIMENTAL=1 lets a half-ported
    // schema's tensors be read anyway.
    //
    // This flag is the engine's own guard, and it is NOT lifted by that
    // hatch: an arch whose attention has no implementation here has no
    // attention in the graph at all, so a run on it decodes text that is
    // fluent, plausible and not the model. That is the worst failure this
    // engine has, so the run is refused instead -- the file still loads (the
    // loader's job is to read what is there), but nothing generates from it.
    //
    // It is about the attention *computation* only. A gap elsewhere (a routed
    // value set, a softcap, a per-layer head geometry) keeps its own
    // mechanism: the arch table's `why` and the tensor gap map.
    bool attention_implemented = true;
    // ---- per-branch post norms (gemma2/3/4, muse-glimmer) -----------------
    // The layer does not add the sub-block's raw output to the residual: it
    // RMS-norms it first, with that branch's own weight, and adds the NORMALIZED
    // row. `norm(attn_out) + x`, not `norm(attn_out + x)` and not
    // `norm(x) + attn_out` -- three different graphs, and the weights differ per
    // branch, so nothing downstream can tell which one ran. That is why the load
    // is gated on this flag and the order is pinned by a dump test.
    //
    // It also costs the projection its folded residual store (`dst += W*x`): the
    // norm has to read the projection's own output before it joins the residual,
    // so a layer that carries these runs gemm + norm + add instead.
    bool post_branch_norm = false;
    // The eps those norms use, when the family names its own. 0 keeps the
    // model's rms_eps -- muse-glimmer's are 1e-8 against a 1e-5 body.
    float post_norm_eps = 0.0f;
    // The gate activation. The tensor is the same shape on both families
    // (`attn_gate.weight`, one value per head) but laguna softplus's it and
    // muse-glimmer sigmoids it, and a softplus where a sigmoid belongs reads as
    // "a bit too much gate everywhere" rather than as a wrong answer.
    bool attn_gate_sigmoid = false;
    // The gated FFN's activation, when it is not SwiGLU: GeGLU, i.e.
    // gelu(gate) * up. gemma's stack is GeGLU, so its `ffn_gate`/`ffn_up` pair
    // run through a different activation than every SwiGLU family here.
    bool gelu_ffn = false;
    // Rotate only on the sliding-window layers (muse-glimmer: RoPE on the
    // windowed layers, NoPE on the full ones). The rotation table is still
    // built per layer; this says the full layers' table is not used.
    bool rope_swa_only = false;
    // GPT-2 and its descendants. This flag must stay LAST: every row below is
    // a positional aggregate initializer, so a member inserted in the middle
    // silently shifts every field after it onto the wrong value. It did --
    // `gpt2_style` landed where `qkv_bias` had been, and every row from
    // deepseek2 onwards then read as GPT-2 in the loader.
    bool gpt2_style = false;
    // ---- fused projections (phi3 and the family it belongs to) -------------
    // `gpt2_style` bundles the fused `attn_qkv` together with LayerNorm,
    // learned positions and a plain GELU MLP, because on GPT-2 those four are
    // one architecture rather than four options. They are NOT one architecture
    // on Phi-3: it wants the fused projection with RMSNorm, a partial rotary
    // and a SwiGLU MLP. Keeping the bundle whole would have meant either
    // running Phi's norms as LayerNorm -- a wrong answer that still decodes --
    // or refusing the family over the packaging, so the two fused shapes get
    // their own flags and gpt2 sets what it uses.
    //
    // fused_qkv     one `attn_qkv` projection whose output row is Q | K | V.
    //               On GPT-2 all three parts are n_embd wide; on Phi the K and
    //               V parts are n_head_kv*head_dim, because Phi is GQA and a
    //               3*n_embd split would read K out of V's rows.
    // fused_gate_up one `ffn_up` projection whose output row is gate | up,
    //               n_ff each, with no `ffn_gate` tensor in the file at all.
    bool fused_qkv = false;
    bool fused_gate_up = false;
    // ---- the plain pre-norm family (gpt2, gptneox, gptj, phi2) -------------
    // These four flags are the pieces `gpt2_style` bundles. It bundled them
    // because on GPT-2 they are one architecture; they are NOT one architecture
    // across the family, and the bundle is the reason `gptneox` had nowhere to
    // be added: it wants LayerNorm-with-bias and an ungated GELU MLP and a
    // fused QKV (all of the bundle) but RoPE instead of learned positions (not
    // the bundle). Naming the pieces separately lets each family take what it
    // has. `gpt2_spec()` now sets all of them, so GPT-2 reads exactly as it
    // did.
    //
    //   ln_bias          LayerNorm (mean and variance, a bias beside every
    //                    weight) rather than RMSNorm.
    //   learned_pos      `position_embd.weight` summed into the token
    //                    embedding; no rotary table anywhere.
    //   parallel_residual  the attention and the FFN both read the SAME normed
    //                    row and are summed into the residual, instead of the
    //                    FFN re-norming the post-attention stream. Two
    //                    different graphs; see Engine::plain_forward.
    //   no_gate_ffn      up -> gelu -> down, with no `ffn_gate` tensor in the
    //                    file. Reading ffn_up as a gate instead would halve the
    //                    intermediate width.
    bool ln_bias = false;
    bool learned_pos = false;
    bool parallel_residual = false;
    bool no_gate_ffn = false;
    // ---- ALiBi (mpt, bloom, falcon) --------------------------------------
    // The position signal is an additive per-head distance bias on the
    // attention score, with no rotary table and no learned position embedding
    // at all. Stored as the SCALE the file names (`attention.max_alibi_bias`),
    // because the per-head slope is derived from it inside the kernel by the
    // same formula llama.cpp uses -- see AttnDesc::alibi_max_bias.
    //
    // BLOOM's converter does not write the key, so its value is the model's own
    // constant: the reference sets f_max_alibi_bias = 8.0f in that arch's
    // loader. Every other family leaves this 0.
    float alibi_max_bias = 0.0f;
    // BLOOM normalizes the token EMBEDDING before the first block
    // (`token_embd_norm`, a LayerNorm with a bias). It is not a per-layer norm
    // under another name -- the file carries both -- and skipping it leaves
    // every row at the wrong scale entering layer 0.
    bool embd_norm = false;
    // Falcon's parallel branch has TWO norms, not one: the FFN is fed by its
    // own LayerNorm over the same pre-attention residual row the attention
    // read, rather than by the row the attention already normalized. GPT-NeoX
    // (parallel_residual, this false) reuses one normed row for both arms;
    // Falcon does not, so the second norm must be computed from the residual
    // BEFORE the attention residual lands in it. On Falcon-7B the two norms
    // hold the same weights (the file has no attn_norm_2), so the arithmetic
    // collapses to the single-norm case; on Falcon-40B they differ.
    bool parallel_two_norm = false;
    // ---- MiniCPM3's residual depth scale ---------------------------------
    // Every layer multiplies the ATTENTION branch's output by
    // `depth / sqrt(n_layer)` before adding it to the residual -- a
    // depth-dependent shrink of the attention contribution, not of the FFN.
    // 0 means no scale. The engine divides by sqrt(n_layer) once at load and
    // keeps the result on ModelConfig, because the divisor is a property of
    // the stack, not of the layer.
    float residual_depth = 0.0f;
    // ---- the router's activation, when the family's files do not say -------
    // 0 keeps whatever the file declares (and softmax when it declares
    // nothing); 2 forces sigmoid. GLM-4.5/4.7 batches its experts with
    // `sigmoid`, and llama.cpp's glm4-moe loader defaults to SIGMOID when the
    // key is missing -- while the generic default here is softmax. On a
    // 128-expert stack softmax and sigmoid are different SELECTIONS as well as
    // different weights, so this cannot be left to the file.
    int router_gating = 0;
    // ---- expert weights are renormalized unless the file says otherwise ----
    // -1 keeps the engine's legacy always-normalize. glm4-moe's reference
    // leaves `expert_weights_norm` false, so a file that is silent on it is
    // NOT renormalized.
    int router_w_norm = -1;
    // ---- gemma4's per-layer input embeddings (PLE) ------------------------
    // `per_layer_token_embd.weight` is [n_embd_per_layer * n_layer, n_vocab]:
    // one embedding per token PER LAYER, gated into every block's output. It is
    // not qwen4exp's n-gram PLE (a hash of the previous tokens through a conv)
    // -- the tensor names differ and so does the graph, so the two are separate
    // flags rather than one.
    bool per_layer_embd = false;
    // The value path is RMS-normalized with NO weight (gemma4's
    // `Vcur = ggml_rms_norm(Vcur)`), which is a different function from a norm
    // whose gamma happens to be all ones only in that there is no tensor to
    // read: a file that carries no V norm must not be run as if it did.
    bool v_norm_weightless = false;
    // The last `attention.shared_kv_layers` layers write no KV of their own:
    // they attend the cache of an earlier layer (see Model::kv_source_layer).
    // Appending for them would be a second, never-read plane; reading their own
    // empty plane would attend nothing.
    bool shared_kv = false;
    // A per-layer scalar (`blk.N.layer_output_scale`) multiplied into the
    // residual at the end of the block.
    bool layer_out_scale = false;
};

// The table entry for an arch name, or nullptr when the table does not know it.
const ArchSpec *arch_lookup(std::string_view name);

// The entry for an arch name, or the conservative llama-like default when the
// table does not know it. `is_known`, when given, reports which of the two
// happened so callers that can warn do.
const ArchSpec &arch_spec(std::string_view name, bool *is_known = nullptr);

// Every name in the table, comma separated, for error messages that should tell
// a user which architectures this build has been taught about.
std::string arch_known_names();

// Short spellings for logs and reports.
const char *arch_shape_name(ArchShape s);
const char *arch_support_name(ArchSupport s);
const char *arch_role_name(ArchRole r);

// The unimplemented piece a tensor name implies, or nullptr when it implies
// none. `shape` matters because one name can mean two things: `attn_gate` is a
// gate inside the gated delta net (implemented) on a recurrent arch, and a gate
// on the attention output (not implemented) on a dense one. `arch` is the
// second exemption, and it is the opposite direction: a tensor that is a gap in
// general is not one for a schema that has code for it, and the table entry is
// the only place that knows which schemas those are. Empty means "no
// exemptions", which is what a caller with no arch name to give gets.
const char *arch_tensor_gap(std::string_view tensor_name, ArchShape shape,
                            std::string_view arch = {});

// The speculative/MTP head a tensor name belongs to, or nullptr. A head is not
// a gap: a target that carries one still runs, it just does not use it, so the
// verdict reports this as a note.
const char *arch_tensor_head(std::string_view tensor_name);

} // namespace krk
