# Three external MoE prefetch/speculative papers: what carries to this engine

Read October 2026. These three are **system/strategy papers**, not RDNA kernel papers.
They are useful as design precedents for expert prefetch and cache policy, and
mostly **not** useful as implementation sources for MLA, speculative-token decoding,
or low-bit dequantization on AMD.

- [MoE-CORE](https://arxiv.org/pdf/2610.01950) — coordinated expert offloading/residency
- [MoE-Infinity](https://arxiv.org/pdf/2401.14361) — sparsity-aware expert cache, request traces
- [SpecPrefetch](https://arxiv.org/pdf/2607.24787) — learned next-layer expert-priority prefetch

## Target reality check first

This repo's AMD targets are **gfx1031 (RDNA 2)** and **gfx1201 (RDNA 4)**, not "RDNA 3 + RDNA 4".
See the LLVM AMDGPU family list and AMD's RDNA 4 WMMA docs. gfx1031 is RDNA 2, and it does
not have the RDNA 3/4 WMMA layout. gfx1201 has WMMA with a different datatype set (FP16/BF16,
INT8/INT4, plus FP8/BF8 and a deeper-K INT4 variant), but **that does not mean native FP4 MM**.
Packed FP4 still needs unpacking/scaling internals before reaching an available matrix op.

## What each paper actually contributes

**MoE-CORE** is expert residency and offload scheduling: phase-specific staging (prefill vs decode),
nonuniform per-layer cache allocation, routing-history-informed replacement, bounded cross-layer
prefetch, and an optional "swap a missing near-boundary expert for a resident unselected one" path.
It evaluates DeepSeek-V4-Flash-W4A8 and GLM-5.2-W4A8C8 on a Huawei Ascend 950PR, and its best
reported number combines that approximate substitution with MTP depth 2. The MTP use is a config,
not the paper's contribution. It does not propose MLA, KV compression, speculative token decoding,
hash routing, rotary schemes, or low-bit dequantization kernels.

**MoE-Infinity** is request-scoped expert reuse: per-iteration/request Expert Activation Matrices,
a bounded trace collection, cosine-distance matching of the current trace to historical traces, and
cache/prefetch priority that favors early layers. It is a CUDA/NVIDIA RTX A5000 evaluation and
reports latency multipliers, but its trace/cache policy is architecturally independent of CUDA.
It contributes no attention variant, MTP head, speculative token algorithm, or quantization kernel.

**SpecPrefetch** is learned next-layer expert-priority predictors (small low-rank adapters per layer),
trained to distill from native router distributions, used **for transfer timing only**: the frozen
native router still selects what actually runs. It also has a window-aware prefetch budget to bound
how many transfers to start early. It is **not** speculative token generation: no draft head, no MTP
block, no DFlash verification loop. It trains on NVIDIA GPUs and evaluates on a Qualcomm Adreno
device. Useful as a predictor-for-transfer design precedent, less useful as an AMD kernel precedent.

## What carries to kraken

The most portable ideas here are at the expert-tier policy layer, because kraken already has the
three-tier expert cache and the host-read/WARM/HOT machinery. The ideas that transfer cleanly:

- **Phase-specific staging and per-layer cache allocation** (MoE-CORE). kraken already separates
  prefill/decode for KV and can already budget per-expert bytes; the "allocate cache by layer and
  workload domain" idea is a candidate policy addition, not a new compute path.
- **Bounded prefetch with a transfer/computation overlap budget** (MoE-CORE, SpecPrefetch).
  kraken's prefetch already has to respect the host-read copy rate and the device stream schedule;
  a windowed prefetch budget is a plausible refinement. The paper form is a policy, not an SM-op.
- **Request-level expert reuse and trace matching** (MoE-Infinity). For a local decode workload,
  an EAM/EAMC-style trace could inform eviction ordering or warm-up ranking. That is a scheduling
  hook on top of the existing cache, not a new kernel.
- **Predict-for-transfer only, native router stays authoritative** (SpecPrefetch). If kraken ever adds
  a learned next-layer expert prefetch hint, the same separation is the safe rule: the hint schedules
  reads; the router still chooses the executed set. The repo's existing `KRK_MOE_NEXT_PREFETCH` /
  hint machinery is the shape to be careful with.

What does **not** transfer cleanly:

- None of the three gives a portable MLA / shared-latent attention / CSA / sparse indexer attention path.
- None gives a speculative-token draft head, MTP block, or DFlash verification algorithm. MoE-CORE's MTP
  use is an end-to-end config, not a DFlash-style decoding contribution.
- None gives AMD FP4/FP8/INT4 dequantization kernels. DeepSeek-V4-Flash W4A8 in MoE-CORE and 4-bit
  DeepSeek-VL2-Tiny in SpecPrefetch are evaluation configurations, not AMD kernel recipes.

## What this engine should treat as "open but not this paper's deliverable"

For the repo's actual near-term DeepSeek work, the relevant architectural source is the DeepSeek-V4
technical report and DeepInfer serving design (both already referenced locally in `docs/`), not these
three prefetch papers. The external papers are best used as **precedents for expert-tier prefetch
policies** once the MLA/drafter path is real, not as a substitute for the attention or decode graph.

## Local follow-up if we want to use these

- Add a short section to `docs/Audit-expert-cache-next.md` (or a new companion doc) that records which
  of these ideas are worth prototyping on kraken's existing expert cache, with the target caveats above.
- If prototyping MoE-CORE-style per-layer cache allocation, drive it behind a flag and keep the native
  router authoritative; the same caution the repo already applies to hint-based next-layer prefetch.
- If prototyping SpecPrefetch-style predictor-for-transfer, keep it optional and model-specific, and
  do not let it change executed expert selection. The repo already treats hint-driven prefetch as
  measured-and-optional for exactly that reason.
