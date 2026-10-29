# Model support matrix

For the 30-file list, checked against the code in this tree by symbol. Two
independent things decide whether a file runs, and they fail differently:

1. **Architecture** (`general.architecture`). An arch the engine does not know
   is refused at load by `Verdict` with a named blocker (`src/verdict.cpp`,
   `arch_tensor_gap`). The file does not run at all.
2. **Quantization format**. Every format the engine can *dequantize* runs
   (`dtype_supported`, `src/quant.cpp:1009` — 25 formats). But only **Q4_K,
   Q6_K and Q8_0** have an int8 dot kernel
   (`dot_q4_K_q8_sx_x8`, `dot_q6_K_q8_sq4_x8`, `dot_q8_0_q8_x8`). Everything
   else goes through `dot_generic_tiled`, which expands each weight to f32
   before multiplying — correct, and several times slower.

So "unsupported" (refused) and "supported but dequantizing" are different
problems. The second is the bigger one numerically, the first is the one that
stops a model running.

**Update:** `phi3` (Phi-3, Phi-4), `gptneox`, `gptj`, `phi2`, `mpt`, `bloom`,
`falcon`, `chatglm`, `glm4`, `glm4moe`, `mellum`, `phimoe`, `phi3moe`,
`minicpm`, `minicpm3`, `qwen2vl` and `qwen38` are now supported — see `docs/PHI3-FUSED-PROJECTIONS.md` and
`docs/PLAIN-PRENORM-FAMILY.md`. That clears Phi-3-mini, Phi-4-reasoning-plus and
adds six of the GGUF spec's own "known values" architectures — `mpt` and
`bloom` bring ALiBi, `falcon` a parallel branch with two norms — see
`docs/ALIBI-AND-MPT-BLOOM.md`.

**Update (2026-10-29):** every recognised-but-pending family now has a fixture
built from its own definition, run once, and deleted again —
`test_pending_model_fixtures` (gemma, gemma2, gemma3, qwen2vl, qwen3vl,
deepseek41, gpt-oss) and `test_pending_model_refusals` (k2-horizon, minimax-m3,
diffuse, dspark, deepseek4-dspark, qwen4-mtp). `gemma` and `gemma3` are `Yes` as
a result; `gemma2` stays `Partial` for one measured reason (its attention-logit
softcap is applied by the reference attention only, so a HIP build refuses it by
capability); `qwen2vl`/`qwen3vl` stay `Partial` because their vision towers are
not read while their text halves run; `deepseek41` runs its whole graph on a
dense-core file. The six still-refused families are refused BY NAME, and the
test asserts the name is in the message.

Its second half did the same for the rows that already claimed `Yes`: 13 of
them had never had a file of their family through a loader. `test_yes_row_fixtures`
now builds one per row from the tensor set that row's own flags imply —
`phi3`, `phi3moe`, `phimoe`, `chatglm`, `glm4`, `glm4moe`, `qwen3`, `laguna`,
`gpt2`, `qwen35` — and `test_dflash_qwen35_spelling` covers the third dflash
prefix. Re-measured after: 48 `Yes` rows, none without a file.

That half also found a silent mis-load: `phimoe`/`phi3moe` files carry a bias on
every RMSNorm and this engine read them only under the LayerNorm flag, so every
real Phi-MoE checkpoint ran with three bias vectors per layer dropped (measured:
two files differing only in those vectors gave `md` exactly 0). They are applied
now (`ArchSpec::norm_bias`), and the same pair moves the logits by 0.0782.

That pass also closed two holes it found. `gemma`'s QK-norm was a family
constant, so every real Gemma-1/2 file was refused for `attn_q_norm.weight` — a
tensor llama.cpp's gemma.cpp/gemma2.cpp never create; it is a per-row flag now.
And a `deepseek41` file carrying V4.1's `hc_*` Sinkhorn hyper-connections or
`engram_*` tensors used to LOAD with no blocker at all, i.e. run the graph
without the residual it was trained with; both are `kGapTensors` entries now and
refused by name.

Verified arch strings in `src/arch.cpp` — KNOWN and read, which is not the same
as running: `deepseek4-dspark`, `qwen4-mtp`, `dspark`, `k2-horizon`,
`minimax-m3` and `diffuse` are recognised and still
refused, each with its own named reason (`Assess`, `--inspect`). `gemma4` ran the
whole dense graph in this pass (below). A recognised
arch is how a refusal names the graph it read instead of shrugging. The full
list: `llama`, `qwen2`, `qwen3`, `qwen35`,
`qwen35moe`, `qwen3moe`, `qwen3vl`, `gemma`, `gemma2`, `gemma3`, `gemma4`,
`deepseek2`, `deepseek4`, `deepseek41`, `gpt-oss`, `gpt2`, `olmoe`, `lfm2`,
`lfm2moe`, `nemotron_h`, `nemotron_h_moe`, `minimax-m3`, `laguna`, `k2-horizon`,
`muse-glimmer`, `qwen4exp`, `recurrent`, `dflash`, `dspark`, `diffuse`, `mpt`,
`bloom`, `falcon` (and `gptneox`, `gptj`, `phi2`, `phi3`). Still absent from the
spec's known-values list: `mamba`, `rwkv` only.

## The matrix

`FAST` = known arch + an int8 dot kernel for its format. `DEQUANT` = runs, but
the format has no kernel. `REFUSED` = no arch.

| file | arch | format | status |
|---|---|---|---|
| Firefly-v5.Q4_K_M | llama¹ | Q4_K | FAST |
| gemma-4-E4B-it-Q4_K_M | `gemma4` ✓ | Q4_K | FAST — two geometries, per-layer input embeddings, weightless value norm, layer_output_scale, proportional rope, shared-KV tail; pinned by `test_gemma4_forward` |
| gemma-4-26B-A4B / 31B (routed) | `gemma4` — dense graph only | Q4_K | **REFUSED by file** — the 26B/31B layers are two separately-normed branches (shared MLP + `post_ffw_norm_1`, experts + `pre_norm_2`/`post_ffw_norm_2`) with a router on a weightless norm of the attention output; the generic MoE block would run a different graph, so the loader names it and stops |
| GLM-4.7-Flash-Q4_K_M | `glm4` ✓ | Q4_K | FAST |
| L3.2-8X3B-MOE-…-Q8_0 | llama-moe² | Q8_0 | FAST if ² |
| (Phi-3.5-MoE class) | `phi3moe` ✓ | Q4_K | FAST |
| Llama-3.2-1B-4B-Quad-MoE.Q4_K_M | llama-moe² | Q4_K | FAST if ² |
| Llama-3.2-1B-Instruct-Q8_0 | llama ✓ | Q8_0 | FAST |
| Mellum2.1-12B-A2.5B-Q4_K_M | llama/qwen3? ² | Q4_K | FAST if ² |
| MiniCPM-V-4_6-Q8_0 | `qwen2vl`/`minicpm` ✓ (text tower) | Q8_0 | FAST (text) |
| MiniCPM5-1B-Q8_0 | llama¹ | Q8_0 | FAST |
| MiniCPM5-1B.i1-Q6_K | llama¹ | Q6_K | FAST |
| Phi-3-mini-4k-instruct-q4 | `phi3` ✓ | Q4_0 | **DEQUANT** (no Q4_0 dot) |
| Phi-4-reasoning-plus-Q3_K_M | `phi3` ✓ | Q3_K | **DEQUANT** (no Q3_K dot) |
| qwen2.5-0.5b-instruct-q2_k | qwen2 ✓ | **Q2_K** | **DEQUANT** |
| Qwen2.5-1.5B-MoE-A0.9B-Q4_K_M | qwen2-moe ² | Q4_K | FAST if ² |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | qwen35moe ✓ | Q4_K | FAST |
| Qwen3.5-9B-Q4_K_M | qwen35 ✓ | Q4_K | FAST |
| Qwen3.8-4B-Q4_K_M | `qwen38` → qwen3 (alias, no hc tensors) | Q4_K | FAST |
| qwen3.8-flash-next-reap-288-Q4_K_M | `qwen38` → **qwen4exp** (alias, by its hc tensors) | Q4_K | FAST |
| smollm-135m-instruct-add-basics-q8_0 | llama ✓ | Q8_0 | FAST |
| Ternary-Bonsai-2-27B-…-PTQ1_0-MTP-Q4_0 | llama¹ | **PTQ1_0 + Q4_0** | **DEQUANT** (both) |
| Ternary-Bonsai-2-2B-PTQ1_0 | llama¹ | **PTQ1_0** | **DEQUANT** |
| tiny-llama3-test-Q3_K_M | llama ✓ | **Q3_K** | **DEQUANT** |
| Tiny-LLM.Q4_K_M | llama¹ | Q4_K | FAST |
| Tiny-Moe.Q4_K_M | qwen3moe ✓ | Q4_K | FAST |
| tinyllama-1.1b-chat-v1.0-q2_k | llama ✓ | **Q2_K** | **DEQUANT** |
| tinyllama-15M-stories-Q2_K | llama ✓ | **Q2_K** | **DEQUANT** |
| tinyllama-15M-stories-Q3_K_M | llama ✓ | **Q3_K** | **DEQUANT** |
| TinyLLama-v0.Q8_0 | llama ✓ | Q8_0 | FAST |
| tinystories-gpt-0.1-3m.Q2_K | llama/gpt2 ✓ | **Q2_K** | **DEQUANT** |
| Uraion-Agent-Steer-Q4_K_M | llama¹ | Q4_K | FAST |

¹ The loader defaults to `llama` when `general.architecture` is absent
(`model.cpp:123`), and most community fine-tunes are llama-schema. Not verified
against the files.
² MoE variants whose arch string is not certain from the filename. If the file
says `llama` but carries `ffn_*_exps` tensors, `arch_tensor_gap` decides — it
looks for implemented computations by tensor name, so this either runs or is
refused with a named blocker rather than mis-loading.
³ **Settled: `qwen38` is an alias, not a schema.** llama.cpp has no such arch
(170 model files, no `qwen38`); Qwen3.8-Flash-Next converts as **`qwen4exp`**
(`conversion/qwen4exp.py`: `@ModelBase.register("Qwen4ExpForConditionalGeneration",
"Qwen4ExpForCausalLM")` with `@ModelBase.example("Qwen/Qwen3.8-Flash-Next")`). The
name still cannot pick the schema, because the family has a sparse form and a
dense one. So `kAliases` in `src/verdict.cpp` probes the file: any of
`hc_attn_inject` / `hc_ffn_inject` / `per_layer_token_embd` / `ssm_conv1d` /
`indexer.q_norm` ⇒ `qwen4exp`; none ⇒ qwen3 geometry. Both are FAST. See
`docs/LLAMA-SCHEMA-VARIANTS.md`.

The earlier note here read the *name* rather than the family, and both of these
were listed as refused on that basis. The old text follows for the record — it is 2 of the 30 and cannot be answered by reading
the tree.

## Counts

| outcome | count | files |
|---|---:|---|
| FAST today | 16 | the Q4_K/Q6_K/Q8_0 llama/gemma/qwen family |
| **REFUSED — arch missing** | **0** | none — every arch named here is now in the table |
| **DEQUANT — no dot kernel** | **7** | 4x Q2_K, 2x Q3_K, 2x PTQ1_0, 1x Q4_0 |
| uncertain (MoE arch string) | 4 | L3.2-8X3B, Quad-MoE, Mellum2.1, Qwen2.5-MoE |

## What each gap costs, and the order to close them

**1. `phi3` — 2 files, refused entirely.** Phi-3/Phi-4 use fused projection
tensors (`attn_qkv.weight`, `ffn_up`/`ffn_down` with a gated MLP whose gate/up
are separate) and a fused QKV that this engine's per-projection blocks do not
read. It needs a schema entry plus a fused-QKV de-interleave in the attention
block. This is the highest-value gap because the files do not run at all.

**2. `glm4` — 1 file, refused.** GLM-4 uses a partial-RoPE split (only half the
head dims rotated), which the rope path does not currently express per-layer.

**3. Q2_K and Q3_K — 6 files, running slow.** Both are 256-value blocks with
16 sub-blocks of 16 values and a per-sub-block scale:
- Q3_K: 110 B/256 = 3.4 bits/weight; value = `(q>>shift & 3) - (hmask bit ? 0 : 4)`
  ∈ [-4,3]; scale = `d * (6-bit signed scale - 32)`.
- Q2_K: 84 B/256 = 2.6 bits; value = `(q>>shift & 3)` ∈ [0,3] with
  `w = d*sc_lo*v - dmin*sc_hi`, so the dot needs a per-16 `sum(x)` term, which
  `quantize_q8_row_sq4` already computes for Q6_K.

Both are mechanically the Q6_K kernel with a different unpack, which is why they
are the right next kernels: same shape, same 8-row sharing, same VNNI/BW/scalar
ladder that now exists for Q8_0.

**4. PTQ1_0 — 2 files.** Ternary, 1.75 bits/weight. Already the sharpest
format in the repo and the one with the largest traffic win (4.9x over Q8_0),
and it still has no dot kernel — see `docs/INT8-ISA-TIERS.md`. The unpack is
2 bits per weight with a per-block scale, so it is the same job again.

**5. Q4_0 — 2 files.** The oldest format and the simplest unpack (one nibble per
weight, per-32 scale). Cheap to add and it makes every legacy file fast.

## How to settle the uncertain rows

```
./build/kraken-inspect <file>
```

It prints `arch`, `verdict runnable|refused`, and then one line per blocker,
naming the exact tensor or format this build cannot take (`tools/inspect_gguf.cpp`,
`Verdict::headline`). That answers rows ² and ³ per file without generating a
token, and for a refused file it says precisely which computation is missing
rather than only that the file is refused.
