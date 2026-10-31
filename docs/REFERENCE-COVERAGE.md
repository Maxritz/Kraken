# Coverage against the ik_llama.cpp reference set

The families named in the porting list (LLaMA-3-Nemotron … Nanbeige 4.2) are
implemented upstream in `ikawrakow/ik_llama.cpp` and its forks
(`Nexesenex/ik_llama.cpp.nxs`, `accaldwell/ik_llama.cpp`). This page records,
per family, **what the reference file is** and **what this engine does with it**,
so a port starts from evidence rather than from a name in a chat message.

Reference checkout used for every claim below: `ik_llama.cpp` **`1c5ea12`**
("CUDA: fuse the indexer score into one kernel (#2784)").

## How the delta was measured, not guessed

`src/llama-load-tensors.cpp` creates every tensor of every family in one file
(`create_<family>_tensors`). Extracting the `LLM_TENSOR_*` kinds each function
uses and subtracting llama's set gives the family's **own** tensor kinds:

```
$ python3 - <<'EOF'   # abridged
fn = split(load_tensors, r'bool create_tensors_helper::create_(\w+)_tensors\(')
print({k: tset(v) - tset(fn['llama']) for k, v in fn.items()})
EOF

command_r    +7: ATTN_Q, ATTN_K, ATTN_V, ATTN_Q_NORM, ATTN_K_NORM, OUTPUT_NORM, TOKEN_EMBD
qwen2_moe    +7: ATTN_Q, ATTN_K, ATTN_V, FFN_GATE_INP_SHEXP, FFN_GATE_SHEXP, FFN_UP_SHEXP, FFN_DOWN_SHEXP
hunyuan      +8: ATTN_Q, ATTN_K, ATTN_V, ATTN_Q_NORM, ATTN_K_NORM, FFN_GATE_SHEXP, FFN_UP_SHEXP, FFN_DOWN_SHEXP
dots1       +12: … QK norms, shared expert, OUTPUT
seedoss      +7: ATTN_Q, ATTN_K, ATTN_V, ATTN_POST_NORM, OUTPUT, OUTPUT_NORM, TOKEN_EMBD
mistral3    +?  : llama's set; its own piece is an ATTENTION TEMPERATURE, not a tensor
```

That difference is the porting checklist: a family whose extra kinds are already
bound by this engine's loader (shared experts, QK norms, q/k/v biases, branch
post norms, `layer_output_scale`, sinks, …) needs only an arch entry and a test;
one that adds a new *graph* step needs engine work.

The reference also says which families share one builder, which is how
"different arch string, same graph" is settled without reading forty files:

```
case LLM_ARCH_LLAMA: case LLM_ARCH_NANBEIGE: case LLM_ARCH_LLAMA4:
case LLM_ARCH_GRANITE: case LLM_ARCH_GRANITE_MOE: -> build_llama
case LLM_ARCH_LFM2:  case LLM_ARCH_LFM2MOE:                       -> build_lfm2
case LLM_ARCH_BERT:  case LLM_ARCH_JINA_BERT_V2: case ..._NOMIC_BERT: -> build_bert
```

## The matrix

`✓` = loads and runs here. `△` = loads and runs with a named gap. `No` = in the
table, refused with a reason. `—` = not in the table yet.

| family (as listed) | reference arch | this engine | note |
|---|---|---|---|
| LLaMA-3-Nemotron | `nemotron_h` / `nemotron_h_moe` | ✓ | our own hybrid path |
| Qwen3 | `qwen3` | ✓ | |
| GLM-4 | `glm4` | ✓ | |
| GLM-4.5 / 4.6 / 4.7 / AIR / 4.7-Flash | `glm4moe` | ✓ | sigmoid router + selection bias |
| Qwen3.5-MoE, dense Qwen-3.5 | `qwen35moe`, `qwen35` | ✓ | |
| Qwen-3.8-Flash-Next | `qwen4exp` (alias `qwen38`) | ✓ | hyper-connections + n-gram PLE |
| Qwen3-VL / Qwen2-VL | `qwen3vl`, `qwen2vl` | △ | text tower runs, vision tower not read |
| Gemma3 / Gemma4 | `gemma3`, `gemma4` | ✓ | dense and routed forms both run (routed = two branches, three norms, detached router input) |
| Gemma4 assistant | `gemma4-assistant` | — | draft-head schema, absent |
| DeepSeek-V3 | `deepseek2` | ✓ | MLA |
| DeepSeek-V4 / V4-Flash-Vision-Exp / V4.1 | `deepseek4`, `deepseek41` | △ | compressed attention + sparse indexer read; streams/multi-sequence open |
| Kimi-2 / K2-Horizon | `k2-horizon` | No | table entry, refused |
| Muse-Glimmer | `muse-glimmer` | ✓ | weightless embedding norm, per-element sigmoid gate, branch post norms at 1e-8, rope on the windowed layers only, `<arch>.logit_scale` before the softcap, window on layers 0..2 of every 4 |
| DSpark / DFlash | `dspark`, `dflash` | No / ✓ | `dflash` + `dflash-draft` run |
| MiniMax-M3 | `minimax-m3` | Yes | schema from llama.cpp `src/models/minimax-m3.cpp`; MSA block selector in `Engine::msa_select_block` |
| Spark 2.5 | `spark2_5` | ✓ | fused `attn_qkv` row + per-head sigmoid gate |
| MiniCPM5 | `minicpm3` (`minicpm` v1/v2) | ✓ | |
| Mellum2 | `mellum` | ✓ | |
| Laguna XS.2 / M.1 | `laguna` | ✓ | |
| LFM2.5 | `lfm2` | ✓ **new this pass** | dense sibling of LFM2-MoE |
| LFM2-MoE | `lfm2moe` | ✓ | |
| Qwen1.5-MoE (Qwen2-MoE) | `qwen2moe` | ✓ **new this pass** | gated shared expert |
| Command-A | `command-r` | — | llama's tensors + **LayerNorm** (no bias) + a logit scale; the main forward is RMSNorm-only |
| LLaMA-4 | `llama4` | — | `build_llama` + no-rope layers, QK norm, shared-expert bank |
| Nanbeige 4.2 | `nanbeige` | — | `build_llama` + `n_loops` layer looping |
| OLMo 1/2 7B/1B | `olmo`, `olmo2` | ✓ | `olmo` is llama's block with WEIGHTLESS norms (no norm tensors at all) and `attention.clamp_kqv`; `olmo2` adds branch post norms, NeoX rope, and WIDE q/k norms of two different widths |
| XVERSE | `xverse` | ✓ | llama's block verbatim |
| Seed-OSS | `seed_oss` | ✓ | its FFN pre-norm is named `attn_post_norm` (loader fallback) |
| SmolLM3 | `smollm3` | ✓ | NoPE: every 4th layer rotates nothing |
| Qwen1 | `qwen` | ✓ | one fused `attn_qkv` row, NeoX rope |
| InternLM2 | `internlm2` | ✓ | the same fused row, adjacent-pair rope |
| Hunyuan-MoE | `hunyuan-moe` | — | shared expert + QK norm |
| Ernie 4.5 / 4.5-MoE | `ernie4_5`, `ernie4_5-moe` | — | shared-expert bank |
| grok-2 | `grok` | — | attn/ffn out norms + layer-out norm |
| Ling/Ring (Bailing-MoE2) | `bailingmoe2` | — | fused qkv + NextN tensors |
| Ling-3.0 (+vision) | `bailingmoe3` | — | MLA + delta-net mixers |
| dots.llm1 | `dots1` | — | QK norms + shared expert |
| Step-3.5-Flash | `step35` | — | attn gate + shared expert + NextN |
| Qwen3-Next | `qwen3next` | — | SSM + attention hybrid |
| Mistral 3 / Mistral 4 / ministral3 | `mistral3`, `mistral4` | — | llama + attention temperature (mistral3) |
| Mimo-V2-Flash / Mimo-2.5 | `mimo2` | — | attn sinks + NextN |
| Cohere2-MoE (North Mini Code) | `cohere2_moe` | — | parallel branch + logit scale |
| OpenPangu | `openpangu` | — | MLA + conv + MHC factors |
| GLM-5 / GLM-5.3-Flash | `glm5next`, `glm-dsa` | — | MLA + DSA indexer |
| Bonsai 1-bit | `bitnet`, `bitnet-25`, `bitnet-b1.58` | — | SubLN norms (the reference has three separate bitnet archs) |
| GigaChat3 | not in `1c5ea12` | — | needs its own reference; no arch string upstream yet |
| Nemotron Bolt | `nemotron_h_moe` | ✓ | per-layer expert counts |

Counts at `1c5ea12`: **100** arch strings upstream, **52** in this table
(36 `Yes`, 8 `Partial`, 8 `No`).

## Order to close the gap

1. **Aliases onto graphs already implemented** — cheapest, and each one is a
   test away. Done this pass: `olmo`, `olmo2`, `xverse`, `seed_oss`, `smollm3`.
   Done this pass too: `qwen`, `internlm2` (both fused-qkv, opposite pairings).
   Left: `openelm` (fused qkv + QK norm), `plamo` (FFN reads the attention
   output, no second norm), `codeshell`/`starcoder2` (LayerNorm + ungated GELU),
   `jais` (fused qkv, **no rope at all**), `orion`, `deci`, `granite` (three
   multipliers: embedding, residual, logit).
2. **Families whose extra tensors are already bound** — `qwen2moe` and `lfm2`
   went in this pass; next are `hunyuan-moe`, `dots1`, `ernie4_5(-moe)`,
   `bailingmoe2`, `cohere2_moe`, `mimo2`, `grok`.
3. **Families needing one new graph step** — `command-r` (LayerNorm in the main
   forward + logit scale), `mistral3` (attention temperature), `nanbeige`
   (`n_loops`), `llama4` (per-layer no-rope), `refact`/`jais` (no rope), `starcoder2`/`codeshell` (GELU MLP).
4. **Families needing a new block type** — `step35`, `qwen3next` (SSM hybrid),
   `bailingmoe3`, `glm5next`, `glm-dsa`, `openpangu`, `raon1`, `flint`.

Every step-1 and step-2 family can be pinned the way `test_lfm2_dense` and
`test_qwen2moe` are: a synthetic fixture, a forward, and an A/B that moves when
the family's own piece is wrong.

## Row audit, 2026-10-29

Every ✓ row in the table above was re-read against its reference builder; the
findings, the seven defects they turned up and the tests that now pin them are in
[`AUDIT-schema-rows-2026-10-29.md`](AUDIT-schema-rows-2026-10-29.md). The short
version: `olmo` was refused on every real file (weightless norms) and ignored a
clamp, `olmo2` was refused on every GQA file (key-norm width), `muse-glimmer` lost
its `logit_scale` and windowed the wrong layers, and a declared QK-norm with no
tensors ran un-normalized. `spark2_5`, `lfm2`, `qwen2moe`, `seed_oss`, `smollm3`,
`qwen`, `internlm2`, `xverse`, `gemma4`, `laguna` and `gpt-oss` came out clean.
