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

// The loadable schemas first, then the ones that stop at a reason. Flags on a
// refused entry are descriptive: they record what the file carries so the entry
// can be flipped to Yes without re-deriving the schema, and they are never
// consulted by a load that is refused.
constexpr ArchSpec kTable[] = {
    // ---- the schemas this engine runs ------------------------------------
    {"llama", ArchShape::Dense, ArchSupport::Yes, ArchRole::Target, "llama", "",
     false, false, false, false},
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
    // the last `shortconv.l_cache` rows (`shortconv.in_proj` 3*n_embd wide ->
    // B, C, x, then `y = B*conv(x) + C*x` -> `shortconv.out_proj`), not a gated
    // delta net. Three more things the schema needs and this engine has no
    // shape for: `attention.head_count_kv` is a PER-LAYER array of length
    // block_count (0 on the conv layers), the file carries no `output_norm`
    // because its final norm is `token_embd_norm`, and `expert_gating_func` 2
    // selects experts with a sigmoid over the probabilities.
    // It is refused by name rather than left to the llama defaults: those read
    // all 30 conv layers as attention layers and die on the first missing
    // `attn_q.weight`, and on the 24B-A2B file here it never even got that far
    // -- "output_norm.weight is missing", before a layer was read. Either
    // message is about a tensor for a mixer the engine has no shape for.
    {"lfm2moe", ArchShape::RecurrentMoe, ArchSupport::No, ArchRole::Target,
     "lfm2",
     "a short-convolution mixer (shortconv), a per-layer "
     "attention.head_count_kv array, a final norm that is token_embd_norm and "
     "sigmoid expert gating are not implemented",
     /*qk_norm=*/true, /*q_output_gate=*/false, /*qkv_bias=*/false,
     /*gemma_norm=*/false, /*attn_gate=*/false, /*rope_neox=*/false,
     /*swa_period=*/0, /*swa_window_first=*/false, /*expert_w_norm=*/0},
    // Gemma scales the embedding by sqrt(n_embd) and uses a tighter eps.
    // Gemma 1 is the llama schema plus embedding scaling. It is not in the
    // collections here, so the entry says so rather than claiming a validated
    // run. (Final-logit softcapping is implemented; the attention-side
    // softcap and sliding windows are not — see gemma2/gemma3.)
    {"gemma", ArchShape::Dense, ArchSupport::Partial, ArchRole::Target, "gemma",
     "not validated against a file in this collection; the schema is llama plus "
     "embedding scaling",
     true, false, false, true, false, true},
    // Gemma 2 adds a softcap on the *attention* logits (inside the softmax,
    // where a host-side fix cannot reach) and alternating sliding-window
    // attention. Neither is implemented.
    {"gemma2", ArchShape::Dense, ArchSupport::No, ArchRole::Target, "gemma",
     "attention-logit softcapping and alternating sliding-window attention are "
     "not implemented",
     true, false, false, true, false, true},
    {"gemma3", ArchShape::Dense, ArchSupport::No, ArchRole::Target, "gemma",
     "alternating sliding-window attention is not implemented",
     true, false, false, true, false, true},
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
    {"gemma4", ArchShape::Dense, ArchSupport::No, ArchRole::Target, "gemma",
     "per-layer input embeddings, shared-KV layers, two attention geometries "
     "(global + sliding window), per-layer attention.head_count_kv and the "
     "attention-logit softcap are not implemented",
     true, false, false, true, false, true},

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
    enum class Served : unsigned char { Never, ByAttnGateFlag, ByMlaFlag };
    Served served = Served::Never;
};

constexpr GapTensor kGapTensors[] = {
    // On a recurrent arch `attn_qkv` is the gated delta net's own projection and
    // is what the loader uploads; on a dense one it is a fused attention
    // projection this engine has no shape for.
    {"attn_qkv.weight", "a fused query/key/value projection (attn_qkv)", true},
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
    // LFM2's mixer: three tensors per layer, and the in_proj is the one every
    // conv layer carries. Named here as well as in the table so a file whose
    // `general.architecture` string was never listed is still refused for the
    // piece it actually carries rather than for a missing attention tensor.
    {"shortconv.in_proj.weight", "a short-convolution mixer (shortconv)", false},
    {"shortconv.conv.weight", "a short-convolution mixer (shortconv)", false},
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
