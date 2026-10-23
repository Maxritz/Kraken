# qwen4exp / qwen35 / qwen35moe / Qwen3.8-27B family: filename sweep disambiguation

These metadata-backed rules prevent different Qwen architectures from being
silently merged by filename. The detailed external-drive examples below are
from the machine inventory and model notes already recorded in this repository;
they are historical for this worktree because the current inventory run only
scans the project-local `models/` directory. Check the generated inventory
before treating any external path as currently reachable.

The short version: the word in the filename is **not the architecture**. The
reliable discriminator is the file's own `general.architecture` string as read
by `kraken-inspect` / `ModelVerdict`; inventory records are snapshots, not proof
that the same path remains mounted.

The same marketing family can resolve to four different loader
verdicts: `qwen35`, `qwen35moe`, `qwen4exp`, and `dflash` (draft object, refused
as a target). A sweep that keys categories off the filename alone will merge at
least three of them.

## The disambiguation table (what this box actually has, file by file)

These rows are recorded in the earlier machine inventory and
`docs/model-inventory.md`. They are not all present in the latest checkout-local
`tools/kraken_inventory.log`. Where reachable, the "loader verdict" is the
engine's metadata-based result, not a guess from the filename.

### qwen35 (gated delta-net recurrent) — the "Qwen3.5" and many "Qwen3.8-27B" files

Files recorded as loading as `qwen35 — supported (recurrent)`:

- `H:/OLLAMA-Models/GGUF/Qwen3.5-9B-Q4_K_M.gguf`
- `H:/OLLAMA-Models/GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
- `H:/OLLAMA-Models/GGUF/Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf`
- `H:/OLLAMA-Models/GGUF/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q5_K_M.gguf`
- `H:/OLLAMA-Models/GGUF/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf`
- `H:/OLLAMA-Models/GGUF/Ternary-Bonsai-27B-Q2_0.gguf`
- `H:/OLLAMA-Models/GGUF/ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf`
- `H:/OLLAMA-Models/GGUF/MiMo-NeoHorse-9B-AGSI-Q5_K_M.gguf`
- `H:/OLLAMA-Models/GGUF/NeoHorse-1-9B-Q4_K_M.gguf`
- `H:/OLLAMA-Models/GGUF/Qwable-27b_Q4_K_M.gguf`

From the earlier `docs/model-inventory.md` inventory of `G:/More-models`:

- `G:/More-models/Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M.gguf`
- `G:/More-models/Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M.gguf` (carries `blk.32`, still `qwen35`)
- `G:/More-models/Qwen3.8-flash-next-Q4.gguf` (carries `blk.64`, still `qwen35`)
- `G:/More-models/Qwen3.8-27B-UD-Q2_K_XL.gguf` (carries `blk.64`, still `qwen35`)
- `G:/More-models/Qwen3.8-27B-WebGGUF-Q4_0.gguf` (carries `blk.64`, still `qwen35`)
- `G:/More-models/Qwen3.8-27B-Opus-Distill-IQ2_XXS.gguf` (carries `blk.64`, still `qwen35`)
- `G:/More-models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (carries `blk.64`, still `qwen35`)
- `G:/More-models/Qwen3.8-9B-Q6_K.gguf` (carries `blk.32`, still `qwen35`)
- `G:/More-models/Ternary-Bonsai-2-27B-PQ2_0.gguf`
- `G:/More-models/Ternary-Bonsai-27B-PQ2_0.gguf`
- `G:/More-models/qwen3.5-9b-instruct-nvfp4.gguf`
- `G:/More-models/qwen3.5-4b-nvfp4.gguf`
- `G:/More-models/qwen3.5-9b-nvfp4.gguf`
- `G:/More-models/Opus4.7-GODs.Ghost.Codex.Distill.4B-Q8_0.gguf`
- `G:/More-models/Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf`
- `G:/More-models/omnicoder-9b-q6_k.gguf`
- `G:/More-models/Q3.5-9B-GLM-5.1-DA.Q4_K_S.gguf`
- `G:/More-models/VibeThinker-3B.Q8_0.gguf` → this one is `qwen2`, not `qwen35`, despite the marketing name
- `G:/More-models/Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M.gguf` → `llama`
- `G:/More-models/Samastam-2.5B-Q8_0.gguf` → `llama`

Local rule for this bucket: a sweep should not call these "Qwen3.5" just because
the name says Qwen3.5; it should call them by their loader verdict (`qwen35`) and
keep the filename as the identity column. Many Qwen3.8-27B files on this box are
in this bucket.

### qwen35moe (recurrent MoE) — the 35B-class MoE files, including Qwen3.8 Distill-35B-A3B

Files recorded as loading as `qwen35moe — supported (recurrent-moe)`:

From `docs/model-inventory.md`:

- `G:/More-models/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf`
- `G:/More-models/Tiel-Coder-35B-A3B-UD-Q5_K_XL.gguf`
- `G:/More-models/Tiel-Coder-35B-A3B-MTP-APEX.gguf` (carries `blk.40`, still `qwen35moe`)
- `G:/More-models/Unsloth-Ornith-1.5-35B-A3B-UD-Q4_K_XL.gguf`
- `G:/More-models/ornith-35b-Q8_0.gguf`
- `G:/More-models/ornith-1.0-35B-Q3_0_ROCMFPX.gguf`
- `G:/More-models/qwable-v1-mxfp4_moe.gguf` → refused here on quantization type #39, still `qwen35moe` by arch
- `H:/OLLAMA-Models/GGUF/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`
- `H:/OLLAMA-Models/GGUF/qwen-agentworld-35b-a3b-nvfp4.gguf`
- `H:/OLLAMA-Models/GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

Local rule: a sweep must not let "Qwen3.8 Distill 35B A3B" and "Qwen3.8-27B"
share a bucket. On this box the Distill-35B-A3B file is `qwen35moe`; the 27B
files that share the "Qwen3.8" marketing name are mostly `qwen35` above; the
REAP Flash-Next files are `qwen4exp`.

### qwen4exp (Qwen3.8-Flash-Next / REAP)

Files recorded as loading as `qwen4exp — supported (recurrent-moe)`:

- `H:/OLLAMA-Models/GGUF/qwen3.8-flash-next-reap-288-Q4_K_M.gguf`
- `H:/OLLAMA-Models/GGUF/Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf`

The earlier machine inventory records both files as supported `qwen4exp`.
The same marketing name ("Qwen3.8 Flash-Next") was also used by a file recorded
as `qwen35`, while the `Swift-Qwen3.8-Flash-Next-*` shards were different objects.
Even within "Qwen3.8 Flash-Next", architecture is file-specific.

### dflash / dspark / mtp head shards — draft objects, refused as targets

Files present here that load as draft objects or head shards, not runnable
targets:

From `docs/model-inventory.md`:

- `G:/More-models/Qwen3.8-27B-DFlash-bootstrap-Q8_0.gguf` → `dflash`
- `G:/More-models/Qwen3.8-27B-DFlash2-Q2_K.gguf` → `dflash`
- `G:/More-models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` → `dflash`
- `G:/More-models/Qwen3.8-27B-DSpark-BF16.gguf` → `dflash`
- `G:/More-models/Qwen3.8-27B-DSpark-Q8_0.gguf` → `dflash`
- `G:/More-models/laguna-s-2.1-DFlash-Q4_K_M.gguf` → `dflash` (also has `attn_gate`; refused as a model anyway)
- `G:/More-models/DeepSeek-V4-Flash-0731-K160-REAP-MTP-IQ2XXS-w2Q2K-AttnQ8-SExpQ8-ROCm.gguf` → `deepseek4-dspark`
- `H:/OLLAMA-Models/GGUF/dspark-MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` → `dflash`
- `H:/OLLAMA-Models/GGUF/dspark_gemma4_12b_q4pure.gguf` → `dspark`
- `H:/OLLAMA-Models/GGUF/Laguna-XS-2.1-DFlash-ik_llama-Q8_0.gguf` → `dflash-draft`

Local rule: a sweep should not count these as models. They are speculative head
sets or MTP shards; this engine uses them beside a target through `--draft` or
`--mtp`, and the loader refuses them as standalone targets.

### Multipart / split / non-model objects that must not be merged with models

- `G:/More-models/dllm/llada/LLaDA2.2-flash-Q4_K_S.gguf` and its two `*.part-*` shards
  — the parts are unreadable on this box; the assembled file is `diffuse`, refused.
- `H:/Models/gguf-exp/imatrix*.gguf` — `general.type` is `imatrix`, not a model.
- `H:/OLLAMA-Models/GGUF/mmproj-qwen35-9b-f16.gguf` — `clip`, a projector shard, not a model.
- `H:/OLLAMA-Models/GGUF/mtp-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf` — `nemotron_h_moe`
  head shard, not a target.

## The rule a sweep must apply

1. **Never key a category off the filename alone.** On this box
   `Qwen3.8-27B-UD-Q2_K_XL.gguf` is `qwen35`; `Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf`
   is `qwen35moe`; `qwen3.8-flash-next-reap-288-Q4_K_M.gguf` is `qwen4exp`;
   `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` is a draft object. Same vendor string,
   four different loader verdicts.
2. **Read the file's own `general.architecture`**, or, when the file is not
   reachable from this checkout, record it under "present in the spec but absent
   from this checkout" and do not invent an architecture for it from the filename.
3. **Separate target vs. draft object by role**, not by whether the filename says
   "model". The DFlash/DSpark/MTP shards are real files here; they are not
   runnable targets even though some of their names look like models.
4. **Do not merge Qwen3.5 and Qwen3.8-27B into one bucket** just because both
   can carry `qwen35`. The earlier records show different file identities,
   sizes, expert counts and MTP positions.
5. **For this engine's verdict column, use the current loader.** The current
   checkout has `qwen4exp` as supported in `arch.cpp`; older inventory entries
   record two REAP files as supported, but those files need rediscovery before
   claiming they are currently reachable.

## Current local truth about qwen4exp on this checkout

The code on this checkout has `qwen4exp` in the architecture table
(`src/arch.cpp`, `include/krk/arch.hpp`). The two REAP examples are recorded as
`qwen4exp — supported (recurrent-moe)` in the earlier machine inventory, but are
not in the latest project-local scan. Refresh file inventory to verify them on
this machine before relying on that file-level result.

The mixed part is documentation: several files in this tree still describe
`qwen4exp` as refused or naming-only. The ones seen from this checkout include
`docs/AUDIT-qwen4exp-support.md`, `docs/STRATA-VS-KRAKEN.md`, `docs/TODO.md`,
`docs/TODO-ANALYSIS.md`, `docs/DEEP-DIVE-MOE-PREFILL-DECODE-2026-10-22.md`,
`docs/REVIEW-2026-10-20.md`, `docs/AUDIT-gpu-resident-expert-cache.md`,
`docs/BENCH-LOADING.md`, `docs/REFERENCES.md`, `docs/model-inventory.md`, and
`docs/perf_trace_report.md`. Those passages were written when the table entry was
in a different state; they are not the loader's verdict and they are not what the
reachable REAP files return here.

So the local disambiguation rule for `qwen4exp` is:

- **Code/table truth on this checkout:** `qwen4exp` is in `src/arch.cpp` as a
  supported row.
- **Previously recorded file truth:** the two REAP entries in the earlier machine inventory return supported; the latest checkout-local inventory contains neither.
- **Docs truth on this checkout:** mixed; some docs are stale and need
  reconciliation, but that is separate from the loader's verdict.
- **Sweep rule:** use the loader's verdict when the file is reachable; list it as
  absent-for-this-run when it is not, and do not decide its architecture from the
  filename.

## What not to do

- Do not classify all "Qwen3.8-Flash-Next" names alike: the earlier inventory recorded distinct architecture rows for different files.
- Do not write a sweep rule that says "Qwen3.8-27B files are `qwen4exp`"; the earlier reachable examples were recorded as `qwen35`.
- Do not write a sweep rule that says "Qwen3.8-Distill-35B-A3B is `qwen4exp`"; the earlier example was recorded as `qwen35moe`.
- Do not count DFlash/DSpark/MTP head shards as runnable targets in a model sweep.
- Do not treat the stale "qwen4exp is refused" lines in some docs as the
  disambiguation rule for this checkout.

## Where this is verified from in the tree

- Architecture rows and flags: `include/krk/arch.hpp`, `src/arch.cpp`
  (`kTable`, `kGapTensors`, `arch_tensor_gap`).
- The shallow loader verdict, which is what a sweep should actually use:
  `src/verdict.cpp` (`assess_model`), `include/krk/verdict.hpp`
  (`ModelVerdict`, `headline()`).
- The machine-readable reachable-file verdicts: `tools/kraken_inventory.log`.
- The human-readable reachable-file table: `docs/model-inventory.md`.
- The tool that prints the loader's verdict the same way the sweep should read it:
  `tools/inspect_gguf.cpp` (`kraken-inspect`).
- The cross-engine note that the same marketing family maps to different engines/
  architectures: `docs/STRATA-VS-KRAKEN.md`.
- The behaviour-level note that the Qwen3.8 names should not be conflated:
  `docs/DEEP-DIVE-MOE-PREFILL-DECODE-2026-10-22.md` §"Qwen3.8 and DeepSeek-V4
  status: do not conflate variants".

## Future corpus (present in the spec, absent from this checkout)

These names were not found in the earlier requested-corpus scan and remain
absent from the latest checkout-local scan unless listed as discovered there.
They are **not** assigned an architecture from their filename here.

- `Qwen3-4B-Instruct-2507-Q6_K.gguf` (referenced as a coherence/check model in
  several docs; not present on the reachable drives scanned here)
- `Qwen3-8B-Q4_K_M.gguf` (same)
- `Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K.gguf` (same; the high-param-count MoE
  referenced in the coherence check note)
- `Qwen3.5-9B-Q4_K_M.gguf` appeared in the earlier external-drive inventory. The latest local scan found `Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf` and `SmolLM2-135M-Instruct.Q4_K_M.gguf`; the former is `qwen3moe`, the latter `llama`.
- Any Qwen3.8-Flash-Next file other than the two REAP files recorded in the earlier inventory needs a fresh metadata check before categorization.

The generated inventory's requested-absent list names the requested files not
found on the currently configured scan roots. Absence from that scan is not a
claim that a file does not exist elsewhere.
