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