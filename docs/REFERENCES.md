# Reference implementations this engine is measured against

Two open-source engines run sparse MoE models on consumer hardware with the same
problem this one has: the expert weights do not fit, so they are streamed. Both
are worth reading for what they do about it, and both are recorded here so the
next pass does not have to rediscover them.

## Edge0 — <https://github.com/Edge0-AI/edge0>

Streaming MoE framework: "SSD expert offload + Recover-LoRA + prerouter routing
prediction". Reports 14.9-17.7 tok/s decode for a 35B-A3B at 4-bit on a Mac
mini M4 Pro with 2.9 GiB peak active memory, and 23.9-25.3 tok/s for an 8B-A1B at
1.0 GiB.

What is relevant here:

- **Prerouter**: "a trained head predicts expert routing one step ahead, so
  expert loads overlap the forward pass instead of stalling it — up to +59%
  decode throughput; the gain grows with storage latency, model size, and routed
  width K." This is the same wall `docs/PERF-ANALYSIS.md` section 10 measures
  from the other side: after the arena and the promotion-ordering fixes, a
  laguna-xs2 token is ~35 ms of compulsory file read against ~25 ms of
  everything else, and the compute it could hide behind is only 8.5 ms. An
  untrained version of the idea — use the *previous* token's routing for a layer
  as the candidate set, and start those reads before the layer's GEMMs — is the
  cheapest version to test here, and the routing traces this repo already
  collects (`KRK_TRACE_EXPERTS`, `tools/expert_trace.py`) are what would say
  whether it predicts anything.
- **Peak memory bounded by the active expert set, not the parameter count** —
  the same policy the device-memory step in section 10 now implements, arrived
  at from a different direction.
- The Python framework keeps backend code behind a facade with a reserved CUDA
  slot, so the core stays independent of the kernel library. Worth stealing if
  kraken grows a second backend.

What does not transfer: Recover-LoRA recovers quantization loss by training
adapters, which is a model-release decision, not an engine one.

## Strata — <https://github.com/Niko1221/Strata>

Runs Qwen3.8-Flash-Next (125B) on consumer cards with an OpenAI/Anthropic
-compatible server. Its published numbers are on **this machine's GPU**: an RX
9070 XT 16 GB, Ryzen 9 3900X, 47 GB RAM, giving 44-60 tok/s depending on
quantization.

What is relevant here:

- It confirms the shape of the problem and the shape of the answer: "Strata
  loads 35-55 GB into your RAM and locks part of it for the graphics card", and
  "with less than ~80 GB of RAM, Strata reads part of it from the SSD while it
  answers". That is the WARM tier plus the read-through path, sized the way this
  engine now sizes it -- explicitly, with a knob, rather than as a fraction of
  whatever is free.
- **An NVMe SSD is called out as the difference between usable and not.** The
  measurements in section 10 agree: `tools/probe_read.py` puts this machine's
  file at ~4 GB/s sequentially and ~19 GB/s when the pages are cached, and the
  decode step tracks that number directly.

What is **not** established from their documentation, and is recorded here as an
open question rather than a claim: whether their CPU path computes loaded
experts while the GPU works on the rest of the token. It is the obvious next
lever for a Vulkan engine with no compute queue of its own, but their README
does not say so and this repo has not measured it.

## The speculation landscape

Five methods are in wide use, and every one of them is **lossless** -- the
accepted output is bit-identical to running the target alone, which is the only
kind worth having here (vLLM's speculators docs and NVIDIA's TensorRT Edge-LLM
guide both state it, and DFlash/DFlash2/DSpark/MTP all fall under it).

| method | what it is | what it costs |
|---|---|---|
| **MTP** | extra block(s) inside the model file, trained to predict t+2, t+3 | nothing extra to load: the weights are already in the GGUF |
| **DFlash** | block-parallel drafter: drafts a whole block of tokens in one pass | a second file, 0.6-1.8 GiB |
| **DFlash2** | block-diffusion successor to DFlash; keeps the top candidates at every position. Reported "+20% more output from every verification pass for 1.3% added latency", up to **3.4x** end to end | a second file, 0.7-1.1 GiB |
| **DSpark** | another drafter family in the same slot | a second file, 1.4-2.5 GiB |
| **EAGLE-3 / P-EAGLE** | feature-level drafters | a separate model |

What kraken sees of these locally, from `kraken-inspect`:

| file | arch | verdict |
|---|---|---|
| `laguna-s-2.1-DFlash-Q4_K_M.gguf` | `dflash`, 6 blocks | refused: not a model |
| `laguna-xs21-dflash-q8.gguf` | `dflash` | refused: not a model |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | `dflash`, 5 blocks, 1.05 GiB | refused: not a model |
| `Qwen3.8-27B-DSpark-Q8_0.gguf` | `dflash`, 5 blocks | refused: not a model |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | `qwen35` | refused, but only on quant types this build cannot dequantize (#17/18/21/22/29) |
| `Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M.gguf` | `qwen35` | **runnable** |
| `qwen3.8-flash-next-Q4.gguf` | `qwen35`, 65 blocks | **runnable** |

Two conclusions fall out of that table and they change the priority order:

1. **All three drafter families share one GGUF contract.** DFlash, DFlash2 and
   DSpark are all `arch = dflash` with a `dflash.*` metadata block and `blk.N.*`
   tensors. One loader path serves all three; there is no need to implement them
   one at a time.
2. **MTP is already loaded and deliberately skipped.** `src/model.cpp` reads
   `nextn_predict_layers` and subtracts it from `n_layer`, so
   `Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M.gguf` runs today with its `blk.32`
   prediction head present in VRAM and unused. Turning an already-resident head
   into a drafter is strictly cheaper work than downloading a second file, which
   matters more here than anywhere else: a second file is a second 0.6-2.5 GiB
   of reads on a workload whose bottleneck is already 347 MiB/token of
   compulsory I/O.

Caveat worth recording: Poolside's own `Laguna-S-2.1-DFlash` card says it
"requires Poolside's llama.cpp fork, branch `laguna` -- upstream llama.cpp ships
the generic DFlash framework but not the Laguna decoder contract". The head
inspects as six blocks with the target's own embedding and head geometry
(3072 / 72 heads, 8 KV / head_dim 128, rms_eps 1e-6), so the decoder contract it
needs is the one this engine already implements for Laguna -- but that is an
inference from the tensors, not from running it.

## Qwen3.8-Flash-Next: what it changes, and what kraken has

Qwen3.8-Flash-Next is 125B of MoE plus a **51B-parameter n-gram embedding
table** (20M entries, indexed by bigrams and trigrams) with 6B active, on a
hybrid stack of **36 Gated DeltaNet layers and 12 Qwen Sparse Attention layers in
a 3:1 schedule**, with **Gated Residual** (an element-wise data-dependent read
gate and a per-branch scalar write gate on the residual stream) and a training
recipe that splits Muon and AdamW across weight categories.

None of that file is on this machine. What is here, under the filename
`qwen3.8-flash-next-Q4.gguf`, is **Qwen3.8-27B** -- `general.name = "Qwen3.8-27B"`,
65 blocks, 15.92 GiB, and it is **runnable in kraken today**: `qwen35` arch,
Gated DeltaNet weights (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`,
`ssm_dt`, `ssm_norm`, `ssm_out`), gated attention (`attn_gate` on the 3:1
schedule's attention layers), and one MTP block at `blk.64`.

So the three Flash-Next-specific features are all **absent from kraken**:

| feature | what it would take |
|---|---|
| **QSA** | a lightweight indexer that compresses the sequence into micro-blocks and selects at *micro-block* granularity, before the attention itself. Kraken's attention is full plus sliding-window; there is no indexer, and this is a long-context feature, not a decode one |
| **Gated Residual** | an element-wise read gate plus a per-branch scalar write gate on the residual stream. Distinct from the `attn_gate` and the packed output gate the engine already has: those gate an attention *output*, this gates the residual *stream* |
| **N-gram Embedding** | 20M-entry lookup table indexed by bigram/trigram. Not a matmul, so it wants exactly the tiering this engine already has for experts -- WARM, read-through, promotion -- with a different key. The one to build first of the three: it is the feature most aligned with what this engine already does well, and 51B of parameters that are looked up rather than multiplied is the cheapest kind of parameter scaling to serve from host RAM |

The naming is worth a warning: a file called `qwen3.8-flash-next` that is
actually Qwen3.8-27B will be counted as "Flash-Next supported" by anyone
reading a filename list. Kraken prints `general.name`, which is how this was
caught.

## The DFlash head Poolside ships for Laguna

Not a separate project, but the other half of this one, and worth recording
next to the two above. `kraken-inspect` on
`G:/More-models/laguna-s-2.1-DFlash-Q4_K_M.gguf` (full output in
`docs/traces/dflash-head-inspect.txt`):

| field | value | why it matters |
|---|---|---|
| `dflash.decoder_arch` | `laguna` | same decoder as the target; the head is not a different model |
| `dflash.block_count` | 6 | six dense blocks, ~330 MiB of weights, all resident |
| `dflash.block_size` | 16 | up to 16 candidate tokens per pass |
| `dflash.target_layers` | 6 indices | only six layers have to be recomputed for verification |
| embedding / heads / head_dim | 3072 / 72 (8 KV) / 128 | identical to Laguna S 2.1's own config |
| tensors | 76, `blk.N.attn_{q,k,v,output,gate}`, `ffn_{gate,up,down}`, `*_norm` | a plain dense transformer block with an attention output gate -- **no experts** |

Two things follow. The head is cheap: six dense blocks against a 68 GiB MoE
target, with none of the streaming that costs a laguna token 60 ms. And the
speculation loop is cheap to verify here, because only six layers of the target
have to be re-run per candidate token.

The loader currently refuses architecture `dflash` as "draft file (not a model)",
which is right for `--draft` and wrong as a refusal: the head wants a code path
of its own, not to masquerade as a model of the family.

## What the models on this machine are refused for, and what is cheapest to fix

After the router-bias fix (see below), `kraken-inspect` on every large model
here says one of these. Raw output in
`docs/traces/model-refusals-after-router-bias.txt`.

| model | arch | blocked by | size of the job |
|---|---|---|---|
| `Qwen3.8-27B-*` | `qwen35` | runs | -- |
| `Qwen3.5-9B-...-MTP` | `qwen35` | runs (MTP block skipped) | -- |
| `qwable-v1-mxfp4_moe` | `qwen35moe` | **quant type #39 only** (MXFP4) | one dequant path |
| `Ornith-1.0-9b-ROCmFPX` | `qwen35` | **quant types #100 and #101 only** (ROCm FPX) | two dequant paths |
| `K2-Horizon-MoVA-36B-A4B` | `k2-horizon` | routed experts on the attention *values* (`attn_v_exps`/`attn_v_gate`) | a new expert family, on top of a gate we already have |
| `NVIDIA-Nemotron-3.5-Lightning` | `nemotron_h_moe` | Mamba-2 blocks instead of the gated delta net | a new recurrence, not a delta rule |
| `DeepSeek-V4-Flash` | `deepseek4` | MLA, attention sinks, quant #26 | three real gaps |

The cheapest unblocks on this machine are therefore **three quantization types,
not three architectures**: `qwable-v1-mxfp4_moe` and `Ornith-1.0-9b` are both
`qwen35` models whose *architecture this engine already runs*, blocked by
dequantization alone. That is a much better first move than a new architecture,
and it is the one a user with a folder of models actually feels.

## Two refusal reasons that were wrong

Both were found while checking the models above, and both are fixed in this
build. They are recorded because a wrong refusal is not a neutral thing: it
sends the next reader looking for code to write.

**`exp_probs_b`, the per-expert router selection bias.** The loader asked for
`blk.N.ffn_exp_probs_b.bias`; the files that carry it spell it
`blk.N.exp_probs_b.bias`. The lookup returned nullptr for every layer of every
laguna model, so `cfg_.router_bias` stayed false and **expert selection ran
un-biased** -- silently, with no warning, because a file without the tensor and
a file spelled differently look identical from there. Turning the fix on then
segfaulted, which located the second half of the same bug: the field was filled
by `upload_f32()` (a *device* pointer) and read as `rb[e]` by a host-side top-k,
and freed through `be_->release()`. The bias is host memory now, 1 KiB a layer,
and the loader accepts both spellings with a warning naming the one it found.
Measured: laguna's output on the capital-of-France prompt is **unchanged**, so
this was a real defect and it is not why laguna answers that prompt wrongly.

**`attn_gate`, the per-head attention output gate.** Listed as unimplemented,
with the loader exempted for `laguna` by name. The engine loads the tensor
behind `ArchSpec::attn_gate`, softplus-gates it and broadcasts it over the
attention result before `wo` -- driven by the schema flag, not by which
architecture asked, so any arch that declares it is served. `k2-horizon` carries
the same tensor and was refused for it anyway. The name-based exemption is now
a flag-based one: naming `laguna` exempted laguna and nothing else, and a name
list cannot fix itself when a second arch shows up.
