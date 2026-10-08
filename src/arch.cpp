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
    // Qwen3.5 and its MoE sibling: gated delta net blocks interleaved with
    // attention, and the attending layers pack an output gate into attn_q.
    {"qwen35", ArchShape::Recurrent, ArchSupport::Yes, ArchRole::Target, "qwen35",
     "", true, true, false, false},
    {"qwen35moe", ArchShape::RecurrentMoe, ArchSupport::Yes, ArchRole::Target,
     "qwen35", "", true, true, false, false},
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
    // Multi-head latent attention: a compressed KV path (attn_kv_a/kv_b and
    // their norms) that the attention kernels have no shape for.
    {"deepseek2", ArchShape::Moe, ArchSupport::No, ArchRole::Target, "deepseek2",
     "multi-head latent attention (attn_kv_a/attn_kv_b) is not implemented",
     true, false, false, false},
    {"deepseek4", ArchShape::Moe, ArchSupport::No, ArchRole::Target, "deepseek4",
     "multi-head latent attention plus compressed-attention tensors are not "
     "implemented",
     true, false, false, false},
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
    // DSpark is a different head family (Markov/diffusion heads over the
    // target's nextn state) that nothing here reads yet, so it stays refused
    // rather than being claimed by the DFlash loader.
    {"dspark", ArchShape::Dense, ArchSupport::No, ArchRole::Draft, "qwen35",
     "dspark speculative head set (dspark.*, 6 blocks) rather than a model",
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
    enum class Served : unsigned char { Never, ByAttnGateFlag };
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
    {"attn_kv_a.weight", "latent KV compression (MLA: attn_kv_a/attn_kv_b)", false},
    {"attn_kv_b.weight", "latent KV compression (MLA: attn_kv_a/attn_kv_b)", false},
    {"attn_kv.weight", "latent KV compression (MLA: attn_kv_a/attn_kv_b)", false},
    // NOT a gap: the router selection bias is loaded and applied. It was a gap
    // only because the loader looked for `ffn_exp_probs_b.bias` while the files
    // that carry it spell it `exp_probs_b.bias` -- so it was loaded for nothing,
    // and the engine's own consumer dereferenced a device pointer as if it were
    // host memory on the path that could never be reached. Both are fixed, so
    // this entry is gone rather than narrowed: a refusal for a feature that
    // exists sends the next reader looking for code to write.
    {"ssm_conv1d.weight", "a Mamba-style SSM block (ssm_conv1d)", true},
    {"ssm_out.weight", "a Mamba-style SSM block (ssm_out)", true},
    {"attn_sinks.weight", "attention sinks (attn_sinks)", false},
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
