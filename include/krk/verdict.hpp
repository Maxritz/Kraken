// ============================================================================
//  verdict.hpp — can this build run this file, and if not, why.
//
//  Four independent things can stop a GGUF: the architecture (the table in
//  krk/arch.hpp), the role it declares (a target model, or a speculative head
//  set that only means something beside one), the quantization formats its
//  tensors use, and a tensor that names a computation this engine does not
//  implement even though the arch is recognized (`attn_v_exps`, `attn_kv_a`,
//  ...). They used to be discovered at three different depths of the loader,
//  each with its own wording, and two of them as a bare failure halfway through
//  loading a 20 GB file. This module answers all of them from the metadata and
//  the tensor directory alone, in one pass, before a single byte is uploaded.
//
//  It is also what `kraken-inspect` prints, so the answer for a model you do not
//  intend to load is the same answer you get when you do.
//
//  It also owns the *arithmetic* of what the file carries beyond its layers:
//  how many multi-token-prediction blocks ride at the top of the stack
//  (`nextn_predict_layers`) and which speculative head tensors are inside it.
//  The loader uses that count instead of reading the key a second time, and the
//  report says the same thing — the two cannot disagree.
//
//  Note what "runnable" claims: no *known* reason to refuse. The loader still
//  checks geometry (head counts, expert tensor shapes, ssm block sizes) as it
//  uploads, because those depend on files being internally consistent in ways
//  no name-level rule can see.
// ============================================================================
#pragma once

#include <string>
#include <vector>

#include "krk/arch.hpp"
#include "krk/gguf.hpp"

namespace krk {

struct ModelVerdict {
    std::string arch;   // general.architecture, verbatim ("" when absent)
    const ArchSpec *spec = nullptr; // the table entry, or the fallback
    bool known = false; // the table knows this name
    bool runnable = false;
    // Multi-token-prediction blocks at the top of the stack: real weights the
    // engine loads past, not layers. Zero when the file declares none.
    i32 mtp_blocks = 0;
    // Why the file cannot run, one entry per distinct cause, in plain words.
    std::vector<std::string> blockers;
    // True but not fatal: an arch nobody listed, a tower that is left out, MTP
    // or draft heads sitting inside a model that otherwise runs.
    std::vector<std::string> notes;

    // "qwen35 — supported (recurrent)" / "laguna — not supported" /
    // "dflash — draft file" / "<absent> — unknown (llama-like)". Callers that
    // already print the arch itself can use `headline()`; the reasons are what
    // matter.
    std::string headline() const;
};

ModelVerdict assess_model(const Gguf &g);

} // namespace krk
