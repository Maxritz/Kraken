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

Raw rows: `docs/traces/results.tsv`. Re-run instructions: `docs/traces/README.md`.

**Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU, 15.9 GiB VRAM), HIP 7.12.60602
runtime, SDK 7.16.26331, Release.**

## Scores

| Model | Size MiB | Arch | Load ms | Prefill tok/s | Decode tok/s | ms/tok | HOT % | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Laguna-S-2.1-PS-IQ2_XXS | 32093 | laguna MoE | 22407 | 21.2 | 6.1 | 163.9 | 48.0 | range guard fired |
| K2-Horizon-MoVA-36B-A4B-Q4_K_M | 21331 | not loaded | - | - | - | - | - | refused at load |
| laguna-xs2-Q4_K_M | 19335 | laguna MoE | 22212 | 92.8 | 32.3 | 31.0 | 71.0 | ran |
| LFM2.5-24B-A2B-QAD-Q4_0 | 12949 | lfm2moe | 7171 | 410.3 | 81.4 | 12.3 | 98.9 | ran |
| Mellum2.1-12B-A2.5B-Thinking-Q8_0 | 12326 | mellum MoE | 10428 | 476.0 | 62.7 | 15.9 | 100.0 | range guard fired |
| Underdog-Saluki-27B-1.0-IQ2-mix-MTP-abliterated | 7962 | qwen35 dense | 6206 | 172.9 | 31.0 | 32.3 | - | ran |
| omnicoder-9b-q6_k | 7018 | qwen35 dense | 4719 | 221.5 | 58.1 | 17.2 | - | ran |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M | 5192 | qwen2 dense | 4514 | 494.3 | 73.1 | 13.7 | - | ran |
| qwen35-9b-instruct-nvfp4 | 5067 | qwen35 dense | 5511 | 634.6 | 78.3 | 12.8 | - | ran |
| lacuna-v1-Q4_K_M | 4836 | llama dense | 4142 | 520.1 | 81.9 | 12.2 | - | ran |
| Qwen3-8B-Q4_K_M | 4794 | qwen3 dense | 4331 | 561.2 | 80.2 | 12.5 | - | ran |
| mythos-9b-unhinged-heretic.i1-Q4_K_M | 4794 | qwen3 dense | 3820 | 558.4 | 80.7 | 12.4 | - | ran |
| Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M | 4518 | qwen35 dense | 2826 | 647.7 | 68.9 | 14.5 | - | ran |
| Opus4.7-GODs.Ghost.Codex.Distill.4B-Q8_0 | 4274 | qwen35 dense | 3455 | 1024.6 | 90.2 | 11.1 | - | ran |
| Qwen3-4B-Q8_0 | 4082 | qwen3 dense | 1448 | 1031.3 | 95.9 | 10.4 | - | ran |
| VibeThinker-3B.Q8_0 | 3133 | qwen2 dense | 2864 | 1360.0 | 0.0 | - | - | ran, emits EOS |
| Samastam-2.5B-Q8_0 | 2560 | llama dense | 2389 | 2877.2 | 148.8 | 6.7 | - | ran |
| Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M | 2211 | llama dense | 2346 | 1463.2 | 94.8 | 10.5 | - | ran |
| smolcode-coder-cpp-3b-q4_k_m | 2007 | qwen2 dense | 2159 | 1092.9 | 137.2 | 7.3 | - | ran |
| smolcode-coder-cpp-1.5b-q4_k_m | 1065 | qwen2 dense | 1556 | 2029.6 | 200.7 | 5.0 | - | ran |
| bitnet-2b-tq1_0 | 1054 | bitnet dense | 631 | 878.6 | 66.3 | 15.1 | - | ran |

`ms/tok` is the reciprocal of the decode rate, quoted so a step cost can be
compared across rows. `HOT %` is the expert cache's hit share and is blank on a
dense model, which has no expert tier.

## What the numbers say

* **Prefill is the weak number on the larger dense files.** Underdog-Saluki-27B
  reads 172.9 tok/s of prompt against 31.0 tok/s of decode, omnicoder-9b 221.5
  against 58.1, and Qwen2.5-Coder-7B 494.3 against 73.1. The small models are
  fast at both ends (smolcode 1.5B: 2029.6 and 200.7), so what is being measured
  is the weight read, not the kernel shape.
* **Decode tracks bytes per token, not parameter count.** bitnet-2b is ternary
  and dense, and loses to the much smaller fp16-adjacent files on prefill
  (878.6 against 2029.6) because its 6912-wide FFN rows are read one at a time at
  one token.
* **MoE decode is bound by expert bring-in.** laguna-xs2 holds 71% of its
  acquires in VRAM and still decodes at 32.3 tok/s against 92.8 tok/s of prefill,
  while LFM2.5-24B, whose whole routed set fits the tier (98.9% hit), decodes at
  81.4. Laguna-S-2.1, at 32 GiB against a 15.9 GiB card, drops to 6.1 tok/s.
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

## Refused at load

| File | Reason |
|---|---|
| K2-Horizon-MoVA-36B-A4B-Q4_K_M | routed experts on the attention values (`attn_v_exps`) with a gate on them (`attn_v_gate`); Mixture-of-Values attention is not implemented |

See `## Refused at load` in [more-info.md](../more-info.md) for the rest of the
refusal list and the reasoning behind it.
