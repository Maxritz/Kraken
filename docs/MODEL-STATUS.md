# Model status

Device benchmark sweep taken on 2026-10-26, one run per model, on the machine in
this checkout.

Command:

```
kraken --model <MODEL> --bench --greedy --max-tokens 64
```

Device only, greedy sampling, `--ctx 4096`, `--chunk 512`, defaults everywhere
else. One run each, so treat a row as an indicator and not as a tight number.
The first load after a build reads the file cold; every other row here used the
warm file cache.

Raw rows: `docs/traces/results.tsv`. Re-run instructions: `docs/traces/README.md`,
and `tools/sweep_models.sh` is the sweep itself. The table below was re-run the
same day after the prefill GEMM's activation prefetch, so its prefill column is
one build newer than the rest of the record.

**Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU, 15.9 GiB VRAM), HIP 7.12.60602
runtime, SDK 7.16.26331, Release.**

## Scores

| Model | Size MiB | Arch | Load ms | Prefill tok/s | Decode tok/s | ms/tok | HOT % | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Laguna-S-2.1-PS-IQ2_XXS | 32093 | laguna MoE | 23214 | 5.5 | 4.5 | 222.2 | 49.5 | range guard fired |
| K2-Horizon-MoVA-36B-A4B-Q4_K_M | 21331 | not loaded | - | - | - | - | - | refused at load |
| laguna-xs2-Q4_K_M | 19335 | laguna MoE | 22089 | 98.1 | 33.2 | 30.1 | 74.3 | ran |
| LFM2.5-24B-A2B-QAD-Q4_0 | 12949 | lfm2moe | 6293 | 424.3 | 80.7 | 12.4 | 98.9 | ran |
| Mellum2.1-12B-A2.5B-Thinking-Q8_0 | 12326 | mellum MoE | 9134 | 460.9 | 60.7 | 16.5 | 100.0 | range guard fired |
| Underdog-Saluki-27B-1.0-IQ2-mix-MTP-abliterated | 7962 | qwen35 dense | 6102 | 224.9 | 30.8 | 32.5 | - | ran |
| omnicoder-9b-q6_k | 7018 | qwen35 dense | 8763 | 271.2 | 63.1 | 15.8 | - | ran |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M | 5192 | qwen2 dense | 4309 | 561.2 | 72.5 | 13.8 | - | ran |
| qwen35-9b-instruct-nvfp4 | 5067 | qwen35 dense | 4860 | 798.5 | 79.0 | 12.7 | - | ran |
| lacuna-v1-Q4_K_M | 4836 | llama dense | 3956 | 655.3 | 91.7 | 10.9 | - | ran |
| Qwen3-8B-Q4_K_M | 4794 | qwen3 dense | 4279 | 639.5 | 80.8 | 12.4 | - | ran |
| mythos-9b-unhinged-heretic.i1-Q4_K_M | 4794 | qwen3 dense | 3721 | 654.3 | 80.8 | 12.4 | - | ran |
| Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M | 4518 | qwen35 dense | 2632 | 778.7 | 68.8 | 14.5 | - | ran |
| Opus4.7-GODs.Ghost.Codex.Distill.4B-Q8_0 | 4274 | qwen35 dense | 3870 | 1328.8 | 92.8 | 10.8 | - | ran |
| Qwen3-4B-Q8_0 | 4082 | qwen3 dense | 2407 | 1333.5 | 96.2 | 10.4 | - | ran |
| VibeThinker-3B.Q8_0 | 3133 | qwen2 dense | 3181 | 1630.8 | 0.0 | - | - | ran, emits EOS |
| Samastam-2.5B-Q8_0 | 2560 | llama dense | 2248 | 3705.3 | 148.8 | 6.7 | - | ran |
| Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M | 2211 | llama dense | 1739 | 1851.7 | 94.2 | 10.6 | - | ran |
| smolcode-coder-cpp-3b-q4_k_m | 2007 | qwen2 dense | 2103 | 1293.5 | 138.8 | 7.2 | - | ran |
| smolcode-coder-cpp-1.5b-q4_k_m | 1065 | qwen2 dense | 1377 | 2521.6 | 196.9 | 5.1 | - | ran |
| bitnet-2b-tq1_0 | 1054 | bitnet dense | 953 | 940.1 | 66.2 | 15.1 | - | ran |

`ms/tok` is the reciprocal of the decode rate, quoted so a step cost can be
compared across rows. `HOT %` is the expert cache's hit share and is blank on a
dense model, which has no expert tier.

## A note on the prefill column

**The sweep above is the HIP arm and predates the CPU prefill fix; it is not
re-run here because this sandbox has no GPU (HIP off).** Do not read the CPU
numbers below as contradicting it -- they are a different backend.

The CPU backend had a defect that made prefill unamortized: `gemm()` called
`vec_dot()` once per (row, output) element, and `vec_dot()` decodes a weight row
as it goes, so every weight row was decoded once per ROW of the batch. The
signature in this table is the prefill:decode ratio. A batched GEMM should show
a large one; a GEMV times N rows shows the ratio of a single-token read. Values
near 1 mean prefill was not batching at all.

Measured on the CPU fixture (2 GiB q8_0 MoE, 2 vCPU), before and after:

| prompt tokens | before | after | speedup |
|---|---:|---:|---:|
| 58 | 1833 ms | 804 ms | 2.28x |
| 149 | 4438 ms | 1495 ms | 2.97x |
| 331 | 9300 ms | 2778 ms | 3.35x |
| 695 | 19575 ms | 6760 ms | 2.90x |

and on a 1 GiB f16 dense file, 3.8-4.3x (details and the bit-identity witnesses:
[MOE-RESIDENCY.md §7](MOE-RESIDENCY.md)).

The same defect **cannot be ruled out on the GPU arm** -- it would show up in
this table as a prefill:decode ratio far below what the card can do, and several
rows have exactly that shape (Underdog-Saluki-27B 7.3, omnicoder-9b 4.3,
Laguna-S-2.1 1.2). Confirming or clearing it needs a run on the card; nothing in
this document claims a GPU number that was not measured on a GPU.

## What the numbers say

* **Prefill is the weak number on the larger dense files.** Underdog-Saluki-27B
  reads 224.9 tok/s of prompt against 30.8 tok/s of decode, omnicoder-9b 271.2
  against 63.1, and Qwen2.5-Coder-7B 561.2 against 72.5 -- against 172.9, 221.5
  and 494.3 before the prefill staging change earlier the same day. The small
  models are fast at both ends (smolcode 1.5B: 2521.6 and 196.9), so what is
  being measured is the weight read, not the kernel shape.
* **Decode tracks bytes per token, not parameter count.** bitnet-2b is ternary
  and dense, and loses to the much smaller fp16-adjacent files on prefill
  (940.1 against 2521.6) because its 6912-wide FFN rows are read one at a time at
  one token.
* **MoE decode is bound by expert bring-in, and the MoE prefill column is not a
  kernel number.** laguna-xs2 holds 74.3% of its acquires in VRAM and decodes at
  33.2 tok/s, while LFM2.5-24B, whose whole routed set fits the tier (98.9% hit),
  decodes at 80.7. Their prefill follows the tier's state: laguna-xs2 read 92.8,
  82.0 and 98.1 tok/s across three sweeps the same day, against 5.5 for
  Laguna-S-2.1 at 32 GiB on a 15.9 GiB card.
* **Two MoEs do not produce a trustworthy answer at these defaults.** Mellum2.1
  and Laguna-S-2.1 both leave the f16 activation range (`experts-sum` stores
  clipped) and exit rc=2, which is the range guard refusing the output. Their
  rates above are what the run reached before it stopped being trustworthy.

## Not coherence-tested in this sweep

This sweep measures throughput and the range-guard counters. It does not run the
scalar `--cpu` reference, which is what the coherence column used to be. Text
agreement is recorded per model in the sections of [more-info.md](../more-info.md)
that name the file, and `scripts/coherence_check.sh` is the way to re-check one
file. Three of the rows above were checked device against the CPU reference in
the session that landed the 2026-10-26 changes: bitnet-2b-tq1_0, Qwen3-4B-Q8_0
and Qwen3.5-9B-Q4_K_M. Qwen3-30B-A3B was checked the same way and is not a row
in this table.

## Files that have left this box

These rows were in the previous revision of `docs/traces/results.tsv` and their
files are no longer in `G:/More-models` or `H:/OLLAMA-Models/GGUF`, so they could
not be re-measured: `SmolLM2-135M-Instruct.Q4_K_M`, `SmolLM2-135M-Instruct.IQ4_XS`,
`Qwen3.5-0.8B.Q4_K_M`, `Qwen3-MOE-4x0.6B-2.4B-Q4_K_M`, `Qwen3-4B-Instruct-2507-Q6_K`,
`qwen3-1.7b-coder-distilled-sft-f16`, `qwen3-1.7b-stem-proof-f16`,
`qwen3.5-4b-nvfp4`, `qwen3.5-9b-nvfp4` and `Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M`.
Their old numbers are in the commit history of this file.

## How a file is judged

`kraken-inspect FILE` answers "will this run, and if not, why" **without loading
the weights**. It runs the same verdict the loader runs, so its answer and the
engine's answer are the same answer — that agreement is a test
(`test_load_verdict`), and it is the point of the tool.

The pipeline, in order. The first rule that fires decides:

1. **Architecture.** `general.architecture` is looked up in the table in
   `src/arch.cpp` (~31 entries). Each entry carries a shape (dense, MoE,
   recurrent, recurrent-MoE, diffusion), a support level, a role, and — for
   anything that is not a plain `Yes` — a sentence naming the gap.
2. **Shape.** A file needs a token table to look anything up in, and a target
   needs attention or SSM tensors to be a transformer. A table pack with neither
   is refused for what it is, and the reason names the tensors the file *does*
   carry.
3. **Role.** A draft head set (`dflash`, `dspark`, `qwen4-mtp`,
   `deepseek4-dspark`, `qwen35-dflash-draft`, `dflash-draft`) is not a model; it
   is refused unless a target is given to attach it to.
4. **Tensor level.** The refused-tensor table names each unimplemented tensor
   kind (`attn_qkv`, `attn_gate`, `attn_v_exps`, `attn_kv_a/b`, `ssm_conv1d`,
   `ssm_out`, …). A file carrying one is refused **by tensor name** rather than
   loaded and decoded as nonsense.
5. **Support level.** `Partial` architectures load and run, leaving the named
   feature out; `No` architectures refuse. `KRK_ARCH_ALLOW_EXPERIMENTAL=1` is a
   *loader* switch — it lets the tensors of a refused schema be read so a
   half-ported architecture can be checked stage by stage, and it does not admit
   the graph: an arch whose attention has no implementation is refused after the
   load and before generation. There is no switch that makes a `No` run.

**Unlisted architectures are refused.** They used to be read against llama
defaults, which meant `kraken-inspect` printed
`zzz9moxx — unknown (llama-like defaults)` and `verdict runnable` while
`kraken` failed later with `GGUF metadata is missing core geometry`. The verdict
was wrong and the engine was right. Now the verdict refuses first, with the
reason that actually applies:

```
verdict    refused
           - architecture 'zzz9moxx' is not in the architecture table and the
             file carries no 'zzz9moxx.block_count' key, so the layer stack
             cannot be determined. Known architectures: llama, qwen2, qwen3, ...
```

The rule asks the *file*, not the name: an unlisted label whose file still
carries a usable `<arch>.block_count` keeps its geometry and stays runnable with
a note, because that file can load. Verified on `fix/unknown-arch.gguf` — the
MoE fixture with only its `general.architecture` string rewritten.

## Family coverage

The four families that were open, and the exact state of each. An arch string
that llama.cpp does not define is not invented here; those files are refused as
unlisted, which is now a named refusal rather than a silent llama-default load.

| Family | GGUF arch | State | What is missing |
|---|---|---|---|
| DeepSeek-V4.1-Flash | `deepseek41` | **Partial** — loads as the deepseek4 graph | compressed attention, the sparse indexer, hyper-connections (`hc_*`), Engram sidecar tensors. A checkpoint carrying them is refused by tensor name |
| MiniMax-M3.1-Flash | `minimax-m3` | **supported** — schema and graph | nothing: the schema is llama.cpp's own (`src/models/minimax-m3.cpp`), the block selector is `Engine::msa_select_block`, and the whole graph prefills and decodes a synthetic file of the family (`test_minimax_m3_msa`). No 428B checkpoint has been run — the file is a fixture — and two deviations are stated rather than hidden: block sets are unioned across group heads, and a prefill chunk wider than 8 rows attends densely |
| Qwen3.8-Flash-Next | `qwen4exp` | **supported, gated** — every component has a kernel | not a missing kernel but a missing **whole-graph coherence gate** against an independently verified reference on the real GGUF: one recorded REAP-288 prompt matched 24/24 and another coding prompt historically diverged. Serving fluent wrong output is the failure this engine exists to avoid, so it stays gated until that gate exists (see [TECHNICAL-README.md §4](TECHNICAL-README.md)) |
| Ornith-1.5-122B | none (unlisted) | **refused as unlisted** | no published GGUF arch string to read it against; refused rather than guessed |

`minimax-m3` matters for a specific reason: the fallback path read unlisted
files against llama defaults, and a 428B sparse-attention MoE is the worst
possible candidate for a schema nobody checked. The schema is now read off
llama.cpp's own loader rather than guessed from safetensors names, and the block
selector it needs is the MSA work from the release before this one. The
vision/video tower the earlier note named is **not in the released file**: the
reference creates no vision tensors for this arch, so refusing the model for it
would have kept a loadable family off the list.


## Refused at load

| File | Reason |
|---|---|
| K2-Horizon-MoVA-36B-A4B-Q4_K_M | routed experts on the attention values (`attn_v_exps`) with a gate on them (`attn_v_gate`); Mixture-of-Values attention is not implemented |

See `## Refused at load` in [more-info.md](../more-info.md) for the rest of the
refusal list and the reasoning behind it.
