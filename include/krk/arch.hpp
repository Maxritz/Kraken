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
// on the attention output (not implemented) on a dense one.
const char *arch_tensor_gap(std::string_view tensor_name, ArchShape shape);

// The speculative/MTP head a tensor name belongs to, or nullptr. A head is not
// a gap: a target that carries one still runs, it just does not use it, so the
// verdict reports this as a note.
const char *arch_tensor_head(std::string_view tensor_name);

} // namespace krk
