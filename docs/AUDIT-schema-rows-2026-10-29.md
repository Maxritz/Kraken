# Audit of the schema rows, 2026-10-29

A review of the architecture rows this engine calls **supported**, done by
reading each family's reference builder and comparing it against the row, the
loader branch the row turns on, and the engine code that branch drives. The
question is not "does it load" — six of the defects below loaded, ran, and
answered plausibly. It is "does the graph this engine runs equal the graph the
file was trained with".

Reference: **llama.cpp `f5222f25`** — `src/models/*.cpp` (per-family loaders and
builders), `src/llama-graph.cpp` (the shared `build_qkv`), `src/llama-model.cpp`
(the rope-type table and `load_swa_pattern`), `src/llama-arch.cpp` (tensor and
key names), `src/llama-hparams.cpp` (`set_swa_pattern`).

Scope: the families added or re-checked in the recent stretch — `spark2_5`,
`muse-glimmer`, `lfm2`/`lfm2moe`, `qwen2moe`, `smollm3`, `seed_oss`, `olmo`,
`olmo2`, `qwen`, `internlm2`, `xverse` — plus `gemma4`, `laguna` and `gpt-oss`,
because those three share the window and gate machinery the new rows reuse.

## Verdicts

| arch | row | reference | verdict |
|---|---|---|---|
| `spark2_5` | fused qkv, per-head sigmoid gate, GELU FFN, NEOX | `spark2-5.cpp`: `get_arr(SLIDING_WINDOW_PATTERN)` mandatory, `attn_gate.weight {n_embd, n_head}`, sigmoid on the pre-attn row, `LLM_FFN_GELU | PAR`, NEOX per the rope table | **correct** |
| `muse-glimmer` | weightless embd RMS, per-element sigmoid gate, branch post norms @1e-8, rope on SWA layers only | `muse-glimmer.cpp`: `build_norm(inpL, nullptr)`, `attn_gate {n_embd, n_embd_head*n_head}`, `post_attention_norm`/`post_ffw_norm`, `load_swa_pattern(ml, 4)`, `<arch>.logit_scale` **required** | **3 defects** (D5, D6, D7) |
| `lfm2` / `lfm2moe` | recurrent, `token_embd_norm` final norm | `lfm2.cpp`: `token_embd_norm` via the name fallback; norm/FFN **biases only on decision heads** (`n_layer_decision > 0`), not the causal body | **correct** |
| `qwen2moe` | moe, `expert_w_norm = 0`, shared-expert scalar gate | `qwen2moe.cpp` → `build_moe_ffn(..., norm_w=false)`; shared branch `silu(g)/g` = `sigmoid(g)` | **correct** |
| `smollm3` | llama + NoPE every 4th layer | `smollm3.cpp`: `n_no_rope_layer_step = 4`, llama's tensor set exactly — **no** `attn_output.bias` | **correct graph**, wrong comment (fixed) |
| `seed_oss` | llama + NEOX, `attn_post_norm` as the FFN's pre-norm | `seed-oss.cpp`: `LLM_NORM_RMS`; `ffn_inp = x + attn_out`; `build_norm(ffn_inp, attn_post_norm)` — the tensor IS the FFN pre-norm; NEOX per the rope table | **correct** (the README claim stands; see "Re-opened and closed") |
| `olmo` | llama base "nothing to add" | `olmo.cpp`: **no norm tensors at all** (three weightless RMS norms) and `attention.clamp_kqv` applied inside `build_qkv` | **2 defects** (D2, D3) |
| `olmo2` | per-head QK norm, branch post norms, NEOX | `olmo2.cpp`: **wide** q norm `{n_embd}`, **wide** k norm `{n_head_kv*n_embd_head}`, `post_attention_norm`/`post_ffw_norm` pre-residual, `load_swa_pattern(ml, 4)` | **1 defect** (D1) + wrong comment |
| `qwen` | fused qkv, NEOX, "no biases anywhere" | `qwen.cpp`: fused `attn_qkv` **plus a mandatory `attn_qkv.bias`** (flags 0), NEOX | **correct graph**, wrong comment (fixed) |
| `internlm2` | fused qkv, NORM rope | `internlm2.cpp`: `create_tensor_qkv`, no bias, NORM per the rope table | **correct** |
| `xverse` | llama base | `xverse.cpp`: llama's tensors; NORM; partial rotary read per model | **correct** |
| `gemma4` | post norms, per-head QK norm, `layer_output_scale`, PLE, two-branch MoE | `gemma4.cpp`: `{n_embd_head}` q/k norm, `layer_output_scale` `TENSOR_NOT_REQUIRED`, `post_ffw_norm_{1,2}` + `pre_ffw_norm_2`, `per_layer_post_norm` | **correct** |
| `laguna` | period 4, full layer first, NEOX, per-head gate | `laguna.cpp`: `load_swa_pattern(ml, 4, /*dense_first=*/true)`, NEOX | **correct** |
| `gpt-oss` | period 2, windowed layer first, NEOX | `openai-moe.cpp`: `load_swa_pattern(ml, 2)` → `set_swa_pattern(2, false)` → `il%2 < 1` | **correct** |

## Defects found and fixed in this commit

### D1 — `olmo2`: the loader refused every GQA file

`src/models/olmo2.cpp` creates `attn_q_norm` over `{n_embd}` and `attn_k_norm`
over `{n_head_kv * n_embd_head}` — norms applied to the projected row *before*
the head split, and the key row is **narrower than the embedding** on every GQA
stack (8 KV heads × 128 = 1024 against 4096 on the 7B).

The loader's width check (`src/model.cpp`, qk_norm block) accepted only
`head_dim` (per-head, qwen3) or `n_embd` (wide, olmoe) and refused anything else
as "neither head_dim nor the full row". So a complete OLMo-2 file could not be
loaded at all, and the error read like a corrupt tensor.

Fixed: each branch is checked against **its own** projected row
(`n_head * head_dim` for q, `n_head_kv * head_dim` for k), with `n_embd` still
accepted because that is the same row on every stack where
`n_head * head_dim == n_embd`. The check still refuses a width that is none of
the three, naming the tensor.

### D2 — `olmo` (v1): the loader refused every file of the family

`src/models/olmo.cpp` creates **no** `attn_norm`, `ffn_norm` or `output_norm`
tensor, and its graph normalizes with a NULL weight in all three places
(`build_norm(inpL, NULL, NULL, LLM_NORM)`). The row was llama's base, so the
loader demanded `attn_norm`/`ffn_norm` and refused every complete OLMo-1
checkpoint with "layer 0 is missing its norm weights".

Fixed: `ArchSpec::noweight_norm` (= `ModelConfig::noweight_norm`) — the loader
stops requiring the two per-layer norms and the final one, and the engine hands
the null pointer to `Backend::rmsnorm`, whose contract already carries the
weightless case (gemma4's value norm). A file that *does* carry the (all-ones)
tensors is still read and multiplied, which is the same function.

### D3 — `olmo` (v1): `attention.clamp_kqv` was ignored

`src/models/olmo.cpp` reads `LLM_KV_ATTENTION_CLAMP_KQV` (8.0 in every OLMo-1
config, `clip_qkv` on the HF side), and the clamp itself lives in the shared
`build_qkv`: projection → bias → `ggml_clamp(qkv, ±c)` → reshape → rope
(`llama-graph.cpp:1851`). This engine read neither the key nor the bound, so a
layer that the model bounds at ±8 ran unbounded.

Fixed: `ModelConfig::attn_clamp_kqv` read from the file; the engine clamps q, k
and v in both attention paths at the reference's point in the chain;
`Backend::clamp_act`/`clamp_supported()` added (CPU implements it) and the
loader **refuses** a file that declares a bound on a backend without the kernel,
because dropping a bound the file states is the silent-wrong-answer class this
table exists to prevent.

### D4 — a declared QK-norm with no tensors ran unnormed

The upload of `attn_q_norm`/`attn_k_norm` is optional (deliberately: an unlisted
variant that carries them gets its file's own graph). But an arch whose row says
`qk_norm = true` names the computation, so a file *of that arch* without the
tensors is missing them, not opting out — and the engine ran the layer with an
un-normalized query and key. Now refused, naming the tensor.

The check is per **layer that attends** (`L.wq.present() || L.wqkv_fused.present()`),
not per layer: on a hybrid stack (lfm2's shortconv, qwen35's delta net,
qwen4exp's recurrent half) the recurrent layers carry no query projection and no
QK norm, and requiring one there refused complete files — which is exactly what
happened when the check first landed without that clause (13 test groups).

### D5 — `muse-glimmer`: `<arch>.logit_scale` was ignored

`src/models/muse-glimmer.cpp:7` reads `LLM_KV_LOGIT_SCALE` **without** an
optional flag, so every file of the family carries it; the graph applies it to
the head's output and *then* the final-logit softcap (`:186-192`). This engine
applied neither the scale nor its order, so a file with `logit_scale != 1`
produced `cap*tanh(x/cap)` where the model computes `cap*tanh(s*x/cap)`.

Fixed: `ModelConfig::logit_scale` (default 1, must be positive), applied once in
`Engine::logit_head` on the device — the single buffer every consumer reads
(host download, device top-k, speculative verify) — before any softcap.

### D6 — `muse-glimmer`: the window fell on the wrong layers

The reference builds the pattern with `load_swa_pattern(ml, 4)`, whose declared
default is `dense_first = false` (`llama-model.h`): layers 0..2 windowed, layer 3
full. The loader's own default is laguna's (full layer first), and the new row
did not override it, so a file that states only `attention.sliding_window` got
layers 1..3 windowed. For this family that is two errors at once — the mask *and*
the rotary, since it rotates only on its windowed layers.

Fixed: `swa_window_first = true` on the row (and on `olmo2`, whose reference uses
the same default).

### D7 — `muse-glimmer`: no window at all meant no rotary at all

Same mechanism, edge case: with no `attention.sliding_window` the pattern makes
every layer "full", and `rope_swa_only` then skips the rotary on every layer. The
reference requires the key. The loader now refuses it with the reason.

### D8 — the hybrid-attention log line printed the complement

`is_swa(l) = (l + phase) % period != 0` means exactly ONE full layer per period,
but the load message reported `period - 1` layers as attending fully — so
muse-glimmer (3 of 4 windowed) logged as a mostly-full stack. Cosmetic, but it is
the line a reader uses to check the phase.

## Re-opened and closed (no change)

**`seed_oss` and `attn_post_norm`.** The row claims the family's
`attn_post_norm` tensor is the **FFN's input norm** (`ffn(norm(x + attn_out))`)
and that the attention branch is not post-normed. That is exactly what
`src/models/seed-oss.cpp` does: `ffn_inp = ggml_add(cur, inpSA)` then
`build_norm(ffn_inp, attn_post_norm)` then the FFN. The loader's fallback chain
binds it to `ffn_norm` (the tensor name reads like gemma's *branch* norm, so the
fallback exists precisely for this spelling), and the engine applies `ffn_norm`
to the post-residual-add row. The README and the row are correct; this audit
re-checked it because an earlier reading of the file concluded the opposite.

## Residual risks, recorded not fixed

* **olmo2/olmo3 sliding-window rotary.** The reference forces
  `freq_scale = 1` and `ext_factor = 0` on the SWA layers of this family
  ("Olmo3 use regular rope with no yarn rope scaling"), while this engine applies
  the model's rope scaling to every layer. Nothing changes for a file with no
  rope scaling; an olmo3 file with YaRN would differ on its windowed layers.
* **`ffn_norm ← attn_norm` fallback** (`src/model.cpp`) exists for
  `nemotron_h`, whose converter applies one norm to both sub-blocks. Every family
  this audit looked at that has a *different* FFN pre-norm is bound before that
  fallback (seed_oss/qwen35 by name, gemma/muse-glimmer/olmo2 through
  `post_branch_norm`), so the alias only fires where it is the file's real shape —
  but it is the kind of fallback that needs this sentence in the same file it
  lives in.
* **`logit_scale` and draft heads.** Applied on the target's head buffer only,
  which is where the reference reads its per-arch `f_logit_scale`. A drafter with
  its own scale key would need the same treatment; no such file exists in the
  collection.

## Tests

`test_norm_schema_shapes` (new group, 25 checks) pins each fix:

1. olmo2 with q `{n_head*hd}` / k `{n_head_kv*hd}` loads, runs, and resolves to
   the wide branch; a 24-wide k norm is refused naming the tensor.
2. OLMo-1 with no norm tensors loads and runs.
3. `clamp_kqv = 8` differs from `1e9` and from no key, and the last two agree
   exactly.
4. qwen3 without its QK norms is refused, naming `attn_q_norm`.
5. muse-glimmer `logit_scale` 2.0 is exactly twice 1.0 on the head's output.
6. muse-glimmer with the window width only (no array) puts the window on layers
   0..2 with the full layer last.
7. muse-glimmer with no window is refused, naming `sliding_window`.

`test_llama_variants` also gained the olmo2 QK-norm tensors on its fixture (the
family's reference creates them mandatorily, so the old fixture was not a
complete file) and an assertion that `olmo`'s row declares `noweight_norm`.
