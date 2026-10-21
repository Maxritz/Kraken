# DeepInfer: what is worth taking, measured against this engine's DeepSeek path

2026-10-21. Scope: DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
(Apache-2.0, open-sourced 2026-10-20) read against this tree, plus what the two local
DeepSeek-V4-Flash GGUFs actually contain. Upstream numbers are DeepSeek's own and are
labelled as such; nothing here is a portability claim.

## What DeepInfer is, in one paragraph

A **vLLM-derived serving runtime for one model family**: its README supports
DeepSeek-V4.1-Flash, on **NVIDIA SM100 (CUDA 13)** and **Huawei Ascend 950 (CANN 9.2)**,
and lists FlashMLA, DeepGEMM, DeepEP and TileKernels as the kernel layer. It ships
offline inference and the engine internals -- scheduler, block manager, model executor,
sampler -- but its own README says a complete service still needs the KV-cache system,
VE/prefill/decode disaggregation and cross-replica load balancing, which the Q&A says
are tied to DeepSeek's internal infrastructure and are **not fully open-sourced**. The
published curves are 4 SM100 GPUs for prefill and 64 for decode; the sweep shapes are
documented (decode request counts 8-256 at 4096 non-cached tokens, prefill batch-tokens
4096-65536, default context 32768) but the README shows plots, not a numeric table.

So: **not a Radeon engine, and not a kernel library to cherry-pick from** -- the kernels
are CUDA SM90+/SM100. What it is good for is the *shape* of the DeepSeek-specific design,
because it is the first public implementation of that design.

## The mechanisms it encodes

| mechanism | what it does | portability here |
|---|---|---|
| **CED** (causal encoder-decoder) | 20 encoder + 20 decoder layers; decoder global KV comes from the encoder's final hidden states via per-decoder-layer projections, so most prompt tokens never run the full decoder | model architecture, not code; only relevant if a CED model is ever loaded |
| **CSA2** (compressed sparse attention 2) | per-layer **Full / Reindex / Reuse** modes: Full computes global KV + indexer K + top-k, Reindex keeps the KV and re-selects indices, Reuse keeps both | the *policy* is portable; needs MLA + an indexer first |
| **Hierarchical sparse indexer** | decoder indexer builds a candidate pool later indexers score; attention is over top-k, not the whole cache. The local V4-Flash GGUF says `indexer.head_count 64`, `key_length 128`, `top_k 512` | needs the indexer tensor class and a scoring kernel |
| **KV numerics** | main KV per-16 with e4m3 scales; indexer KV per-32 MXFP4/ue8m0; SWA KV MXFP8, 20 layers x 134 tokens x 528 B = **1,415,040 B per sequence**; paper: **890 B/token** global KV, about 1/4 of V4-Flash | our KV tier is f16; the formats are DeepSeek-specific |
| **Bounded replay** | a missing SWA state is rebuilt by recomputing only the **last 128 tokens** (README default truncation 140, covering the window plus spec backtracking) | portable idea, small to adopt |
| **DSpark verification** | drafts up to 5 tokens, then verifies a *subset* ranked by expected acceptance | portable idea; worth measuring before believing |
| **DeepEP / EPLB / PD split** | expert-parallel all-to-all, hot-expert replica placement (slowest/mean 1.045-1.090), prefill and decode on separate pools | single card, no NVLink: **not applicable** |

## Measured here: the files this would have to run

`kraken-inspect` on
`G:/More-models/DeepSeek-V4-Flash-0731-K160-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf`:

- **53.51 GiB payload, 1328 tensors** (plus 3 of an unknown-size type), `deepseek4`,
  43 blocks, 4096 embd, 64 heads, **`head_count_kv = 1`**, key/value length 512,
  `q_lora_rank 1024`, `output_lora_rank 1024` with `output_group_count 8`,
  **`compress_ratios = [44]`**, `compress_rope_freq_base 160000`, rope dim 64,
  **SWA window 128**, indexer 64 x 128 with **top-k 512**, 4 hyper-connection lanes
  with 20 Sinkhorn iterations, **3 hash layers** (`ffn_gate_tid2eid`), 256 experts x
  top-6 + 1 shared, expert FF 2048, `expert_weights_scale 1.5`, Yarn x16 from 65k,
  1M context, and one MTP block (`blk.42`).
- **Verdict: refused**, by name: "multi-head latent attention plus compressed-attention
  tensors are not implemented", the `attn_kv_a/attn_kv_b` gap, and quant type #26.

That refusal is the whole measurement available today: **no DeepSeek decode number can
be produced on this build**, because the arch table refuses the file before any tensor
is read. The two local V4-Flash files are the same family with and without the REAP
experts; both are refused for the same class.

What the cache arithmetic says about why the DeepInfer design matters, from this file's
own metadata: an **uncompressed f16 latent cache** is 43 layers x (512 + 512) x 2 B =
**88 KiB/token**, i.e. **88 GiB at 1M context** on a 15.9 GiB card. DeepInfer's paper
quotes **890 B/token** for global KV with FP4 plus CSA2 reuse -- a ~100x reduction, and
the only reason a 1M-context V4 file is servable at all. Every piece of that reduction
is DeepSeek-model-specific; none of it is a generic KV-cache trick.

## What this engine already has, so the comparison is honest

- **The KV tier is width-agnostic.** `Model::kv_dim_at(l)` / `layer_has_kv(l)` already
  give per-layer row widths, so an MLA layer is one more width (576 floats for V3:
  `c_KV` 512 + `k_rope` 64) instead of `n_heads x head_dim x 2`. HOT/WARM/COLD paging
  and the admission rules do not care. This is the part that used to be a redesign.
- **MoE is ready** -- 256 experts x top-8 is the shape the tier already runs, and the
  local file's top-6 is smaller.
- **The loader and the arch row are not.** `deepseek2` and `deepseek4` are `No` rows in
  `src/arch.cpp` with the reason spelled out, and the gap map names
  `attn_kv_a/attn_kv_b/attn_kv`. `docs/ASSESS-mla-deepseek.md` costed the missing work
  at **~2 days for the loader** (arch spec, tensor classes, qk-norm, rope-64, the
  absorbed `W_UV` fold) **plus a latent-space decode kernel** (~one more kernel shaped
  like the existing split reduction).
- **Chunked prefill and its bit-exactness gate exist**, which is what makes a
  rebuild-and-continue mechanism testable rather than hopeful.

## What is worth taking, in order

1. **Bounded replay (recompute the last 128 tokens).** The cheapest borrow and the one
   that unlocks something the tier cannot do today: *drop* SWA/window state instead of
   pinning it, then rebuild on demand. Gate: a test that drops a window mid-run,
   replays 128 tokens, and requires the **same `--prefill-hash` trace** as the
   uninterrupted run at `--chunk 1/2/8` -- the chunk-invariance gate already proves the
   machinery can see a difference here, so reuse it rather than inventing a text check.
2. **MLA: the loader plus one absorbed decode kernel.** It is a prerequisite for every
   DeepSeek file, and the design is now documented publicly in a form that maps onto
   this engine (latent 512 + rope 64 per layer; per-head output projection absorbed at
   load; no per-head K/V cache). Gate before anything else: a **CPU oracle** on a small
   MLA model (DeepSeek-V2-Lite, not a 600B file) -- logits at a few positions matching
   a reference within the f16-storage tolerance -- because a wrong absorb looks like
   fluent text, the same way the PLE lane-scatter did.
3. **CSA2's Reuse/Reindex modes.** Once MLA exists, these are a *policy* over the KV
   tier: a layer either owns its cache, or shares another layer's cache and re-selects
   indices. The tier can express that (its widths are per layer and its slots are
   referenced, not copied), and it is the difference between 44 caches and a handful at
   long context. No cost model from DeepInfer transfers -- their cache lives in 3FS and
   HBM across 64 GPUs.
4. **Indexer top-k** (512 in this file). Portable as a kernel, but only pays past
   ~32K context and needs the indexer tensor class first. Do it with MLA, not before.
5. **FP4 / MXFP4 KV.** The win is real (cache bytes halve or better) and gfx1201 has the
   conversion instructions, but there is no code to port: DeepGEMM/FlashMLA are CUDA,
   and the formats here are per-16/per-32 with vendor scaling conventions. Park it
   behind MLA.

## What NOT to take, with the reason in one line each

- **DeepEP, EPLB, PD disaggregation.** They exist to balance work across 64 GPUs and an
  RDMA fabric. One card has nothing to balance, and the tiers here are RAM/VRAM, not
  ranks. (Their own production numbers -- 1.045-1.090 slowest/mean -- are a
  *load-balance* metric, not a speedup.)
- **Confidence-scheduled verification (DSpark).** Tempting, and their own table says why
  not to bother on a single stream: -1.5% at 16 concurrent sequences, +1.6% at 32,
  +11% at 96. This engine is the 1-16 sequence case. The variable-length verifier is a
  throughput-under-batching win; our spec round should keep verifying every draft (and
  keep the Leviathan rule, which is a correctness property, not a schedule).
- **Their benchmark curves.** 4 and 64 SM100 GPUs, and the README publishes plots whose
  axes are latency/throughput at batch shapes this engine never sees.
- **The vLLM-derived engine structure.** It is a good map of what a DeepSeek serving
  stack needs, and a bad template for a 15.9 GiB single-card engine whose whole design
  is tiers.

## The one-line verdict

DeepInfer confirms the DeepSeek-specific work is **MLA + a compressed/indexed KV**, and
it supplies the layout and the failure modes but no portable code. The local V4-Flash
GGUF measures at **53.51 GiB, 88 KiB/token uncompressed, 890 B/token in their design**,
and stays refused here until the loader and one latent-space decode kernel exist --
which is the ~2-day-plus-kernel item already costed in `docs/ASSESS-mla-deepseek.md`,
not a new project.

## References

- DeepInfer: <https://github.com/deepseek-ai/DeepInfer> (README, `doc/inference_deployment_qa.md`)
- DeepSeek-V4.1-Flash model paper (CED, CSA2, KV numerics, 890 B/token):
  <https://arxiv.org/html/2609.19969>
- DeepSeek-V2 (MLA), V3 (MoE/FP8), V3.2 (DSA), V4 (CSA/HCA) reports:
  <https://arxiv.org/html/2405.04434v2>, <https://arxiv.org/html/2412.19437v1>,
  <https://arxiv.org/html/2512.02556v1>, <https://arxiv.org/html/2606.19348v1>
- "Insights into DeepSeek-V3" (ISCA '25; MTP acceptance 80-90%, 1.8x generation):
  <https://arxiv.org/html/2505.09343v2>
- Local: `docs/ASSESS-mla-deepseek.md` (the cost of the missing pieces),
  `src/arch.cpp` (the `deepseek2`/`deepseek4` refusals), `include/krk/model.hpp`
  (`kv_dim_at`/`layer_has_kv`), and the inspect output quoted above.
