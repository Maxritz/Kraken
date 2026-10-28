# What in `Maxritz/arxiv-ML-Techs` is worth taking, and what is not

`ML-NEW-TECH.md` is ~960 lines covering ~990 papers. Most of it is training-side
(RL, distillation, agents) and does not apply to an inference engine. This is the
short list that does, ranked by what Kraken can MEASURE, with the decision for
each. The rest is at the bottom under "considered and declined", because a list
that only says yes is not a survey.

Measured on this box today (2 cores, AVX-512 VNNI, Tiny-Moe Q4_K_M):

| arm | prefill | decode | where the time goes |
|---|---:|---:|---|
| CPU, 46-token prompt | 309 tok/s | 124 tok/s | GEMMs 93%, attention 3.0% |
| CPU, 1400-token prompt | 136 tok/s | -- | **attention 63.4% at 5.2% of peak** |

The second row is the one that matters for everything below: at a context length
anyone would call real, the GEMM work that this repo has spent three PRs on is
no longer the bottleneck. Any paper about KV, attention or context is now
pointing at the top of the profile.

---

## 1. Take now

### 1.1 `2610.34721` PSR -- deterministic cache-aware expert routing

Kraken already has an expert cache (`src/expert_cache.cpp`, LFU + aging, layer
prefetch) and already makes a bit-exactness promise (`--prefill-hash` must not
move between configurations). Those two things are in tension the moment routing
is biased toward resident experts, and PSR is the design that resolves it: route
against a **logical** cache whose contents are a function of the token sequence
only, and let the **physical** cache affect latency alone. Reported 2.2-2.3x over
exact+LRU at 50%/25% residency on OLMoE.

Why this is first: it is the only item on the list that improves a path Kraken
already ships, needs no device support, and preserves a guarantee Kraken would
otherwise have to give up.

### 1.2 Attention, before anything else

`2610.33719` ZipFusion (compressed KV consumed **in the kernel**, reconstructed in
registers, no decompress-to-memory pass) and `2610.33625` RAQ-MLA (quantize
against functional role, not tensor name) are both aimed exactly at the 63% row.
The decisive detail from `2610.33076`: stacking two codecs was *worse* than
either alone, so the codec has to live inside the attention kernel, not beside
it.

Kraken's own numbers say the same thing from the other side: attention runs at
5.2% of the f32 peak with a scalar loop. A better codec on top of a scalar kernel
would spend its savings on unpacking. **Kernel first, then codec.**

## 2. Take next, with a cost model

### 2.1 `2610.21500` -- the KV-hit crossover, as arithmetic

A remote KV hit costs `T0 + prefix/B`; it beats recompute only when the tier
sustains more than `B* = s*R`. Published numbers: `T0 ~= 0.33 s`, so under ~17K
tokens a remote hit cannot win at any bandwidth -- and enabling the tier moved a
16K p99 from 3.96 s to 17.3 s.

Kraken has tiering already (`src/kv_tier.cpp`, HOT/WARM/COLD with cold files) and
a server prefix cache that reports `cached_tokens`. It does **not** have this
crossover gate. That gate is a few lines and it is the difference between a tier
that helps and a tier that turns a hit-heavy stream into a 83-second p99.

### 2.2 `2610.30206` MoEZip -- eviction changes expert selection

Dropping context does not only blur attention; it changes which experts the
router picks. For a cache whose policy ranks experts by expected reuse, that is a
correctness-adjacent interaction worth asserting in a test rather than
discovering.

## 3. Cheap and self-contained

| paper | idea | why it fits |
|---|---|---|
| `2610.35336` | best fusion is 2-4 megakernels, predictable from per-kernel timings | Kraken now has per-kernel timings on all three backends; the selection step is the missing half |
| `2608.06177` | stop accumulating once the sign is decided | applies to the int8 dot, but only for ternary/binary weights -- measure before believing |
| `2608.04048` | 2-bit base + residual layers, one file serves 2/4/6/8-bit | a load-time policy, no kernel change; lets one GGUF serve every device budget |
| `2610.22428` | the embedding table is most of a small model | Tiny-Moe is 90 MB and its embedding is a large share; a frozen code table is a loader-side win |

## 4. Considered and declined

* **`2610.28947` 2:4 sparsity.** At equal bits, adding 2:4 loses to the next
  lower precision. Kraken has lower precisions. Declined on the paper's own
  evidence.
* **`2610.31479` speculative decoding.** Every decoder tested *lost* throughput
  on desktop. Kraken's decode is memory-bound at 20% of its own roofline, so the
  acceptance-length arithmetic has to be measured on this box before any drafter
  is written; the paper says the reported 2-6x does not transfer.
* **`2608.01536` dual-sparse kernels.** Needs a *sparse* weight layout, and
  Kraken's formats are dense blocks. Would be a new format, not a kernel.
* **Distributed / split-learning cluster** (Part 3E): nothing in this repo is
  multi-node and the user's targets are single-box. Not now.
* **Training-side clusters** (Parts 2-3, on-policy distillation, RL): out of
  scope for an inference engine.

## 5. On LMCache and TurboQuant specifically

Asked directly, so answered directly, with the arithmetic that decides it.

**TurboQuant: yes, but it is second in line behind the attention kernel.** The
codec is a random rotation plus scalar quantization -- exactly the "pack, rotate
and reconstruct in one kernel" shape ZipFusion says pays off. On this box the KV
that would be compressed is currently 3.0% + 0.7% = 3.7% of a short run and 63%
of a long one, so it is worth nothing until the kernel can consume it, and worth
a lot immediately after.

**LMCache: the idea is already half-here; the missing part is the gate.** Kraken
has tiering with cold files and an in-process prefix cache with
`cached_tokens`. What the literature adds is the crossover model: a local hit
beats recompute by ~150x at 1400 tokens (10.3 s to recompute against ~67 ms to
read 100 MB), while a *remote* hit with a 0.33 s fixed cost cannot beat recompute
under ~17K tokens at any bandwidth. So: keep it local, add the gate, do not build
a distributed KV store.
