# Kernel speed overhaul — evidence, causes, tasks

Goal: **every device op within ~1.3x of what this box can do**, without moving
the coherence gate (9/9 local models print CPU-identical readable text).

Method: sherlock-it. Facts are separated from hypotheses everywhere, each
hypothesis names the test that would disprove it, and no number below is used
unless its instrumentation overhead was measured. Coherence work is done and
green; this file is the performance half.

---

## 0. Baseline (all 9070 XT, gfx1201, ROCm 10.1 / HIP 7.16, `--bench`, 55-token
prompt, 256 decode tokens, ctx 2048)

| model | prefill | decode | weight bytes/token | achieved | GPU compute busy |
|---|---|---|---|---|---|
| SmolLM2-135M Q4_K_M | 4351 tok/s | 436 tok/s | 88 MB | 38 GB/s | (not sampled) |
| Qwen3.5-0.8B Q4_K_M (GDN) | 3051 tok/s | 180 tok/s | 494 MB | 89 GB/s | **30.6% avg / 33% max** |
| Qwen3-MoE 4x0.6B | 196 tok/s | **13 tok/s** | 920 MB | 12 GB/s | **7.3% avg / 8.7% max** |
| Qwen3-4B-2507 Q6_K | 452 tok/s | 43 tok/s | 3.15 GB | 136 GB/s | (not sampled) |
| Qwen3-8B Q4_K_M | 639 tok/s | 62 tok/s | 4.80 GB | 299 GB/s | **91.7% avg / 100% max** |
| Qwen3.5-9b-Sushi Q4_K_M | 631 tok/s | 50 tok/s | 5.36 GB | 268 GB/s | (not sampled) |

Hardware yardstick, measured on this box with torch 2.15 / ROCm 10.1 (rocBLAS
behind it), identical shapes:

| shape | time | achieved |
|---|---|---|
| f16 GEMV 1x4096x4096 (33.5 MB read) | 47.3 µs | 710 GB/s (cache-aided, see below) |
| f16 GEMM 32x4096x4096 | 43.4 µs | 772 GB/s, **24.7 TFLOP/s** |
| f16 GEMM 32x4096x11008 | 174.9 µs | 16.5 TFLOP/s |
| SDPA 32 heads x 32 q x 2048 kv x hd128 | 18.0 µs | — |

**The 710 GB/s row was cache-aided and is not the decode ceiling.** A 33.5 MB
matrix fits the 64 MB Infinity Cache, and 33.5 MB / 47.3 µs is above the card's
nominal 640 GB/s (20 Gbps x 256-bit). The honest decode ceiling is what a
buffer *larger* than the cache achieves: `kraken-bench`'s f16 head
(248320x1024, 508 MB, 883.01 µs) reads at **576 GB/s = 90% of nominal DRAM**.
Every GB/s below is judged against 576-640, not against 710.

## 0c. Harness table (`kraken-bench --sweep`, marginal cost, launch floor 2.65 µs)

These are per-shape raw kernel numbers with the launch and drain amortized out,
and a host dot-product check on the first rows (`chk`), so a fast wrong answer
cannot pass.

| shape | fmt | bytes read | time | achieved |
|---|---|---|---|---|
| 4096x4096 rows=1 | Q4_K | 9.44 MB | 16.40 µs | 575 GB/s |
| 14336x4096 rows=1 | Q4_K | 33.03 MB | 48.97 µs | 674 GB/s |
| **248320x1024 rows=1** | **Q4_K** | **143.03 MB** | **250.0 µs** | **572 GB/s** |
| 1024x4096 rows=1 (k/v) | Q4_K | 2.36 MB | 7.83 µs | 301 GB/s |
| 3584x1024 rows=1 (g/u) | Q4_K | 2.06 MB | 6.37 µs | 324 GB/s |
| 1024x3584 rows=1 (down) | Q4_K | 2.06 MB | 6.17 µs | 335 GB/s |
| **4096x4096 rows=1** | **Q6_K** | **13.76 MB** | **96.13 µs** | **143 GB/s** |
| **248320x1024 rows=1** | **Q6_K** | **208.59 MB** | **1068.7 µs** | **195 GB/s** |
| 4096x4096 rows=1 | Q5_K | 11.53 MB | 32.72 µs | 352 GB/s |
| 4096x4096 rows=1 | Q8_0 | 17.83 MB | 20.04 µs | 889 GB/s (cache) |
| 4096x4096 rows=1 | Q4_0 | 9.44 MB | 18.99 µs | 497 GB/s |
| 4096x4096 rows=1 | f16 | 33.55 MB | 21.84 µs | 1537 GB/s (cache) |
| 248320x1024 rows=1 | f16 | 508.56 MB | 883.0 µs | 576 GB/s (DRAM) |
| 4096x4096 rows=32 | Q4_K | 9.44 MB | 68.51 µs | **15.67 TFLOP/s** |
| 14336x4096 rows=32 | Q4_K | 33.03 MB | 240.2 µs | **15.65 TFLOP/s** |
| 4096x4096 rows=32 | f16 | 33.55 MB | 86.50 µs | 12.41 TFLOP/s |

### What this refutes, and what it names

1. **Refuted: "the decode GEMV reads ~5x the bytes it needs" (lane-strided
   18-byte chunk reads).** Q4_K reaches 572-674 GB/s, i.e. 89-105% of the DRAM
   envelope, on both cache-resident and cache-exceeding shapes. The kernel's own
   comment — consecutive lanes cover consecutive chunks, so a warp's loads span
   a contiguous span — is correct, and the aggregate access is fine. There is no
   2.4x waiting in the Q4_K GEMV; my earlier byte-amplification estimate was
   wrong (it counted per-instruction line waste and ignored that the next load
   hits the same lines).
2. **Refuted: "prefill WMMA is 4x off".** The 6.2 TFLOP/s that produced that
   claim came from an event-distorted run. The harness says 15.65-15.67 TFLOP/s
   at rows=32, i.e. **1.58x off rocBLAS' 24.7**. Real, worth fixing, not 4x.
3. **Named: Q6_K is the format that is 3-4x slow.** 195 GB/s on a cache-exceeding
   208 MB matrix, and **143 GB/s on a 13.8 MB matrix that fits the cache** — so
   it is not memory-bound at all: it is instruction-bound in the chunk decode
   (6-bit unpack + per-element f32 scale multiply + f16 convert, per value, per
   token, with zero reuse). Q5_K at 352 GB/s is the milder version of the same
   defect. Q6_K is what `Qwen3-4B-2507-Q6_K` (136 GB/s model-level) runs.
4. **Named: the fixed per-call cost dominates the small matrices.** Launch
   floor 2.65 µs against 6.2-7.8 µs for the k/v, g/u and down shapes — 34-43% of
   each call is launch, and the model streams those matrices every token. That
   is the real dense-decode defect: not bandwidth, but call count. It is why the
   engine's fused group is 2.9x cheaper per matrix than the separate calls
   (20 µs/matrix fused vs 57 µs/matrix solo, both from the same instrumented
   run), and why Q4_K's 90%-of-DRAM kernel still only yields ~300 GB/s at the
   model level.
5. **Cache caveat, stated so it is not misread:** rows=1 numbers for 9-33 MB
   matrices are cache-aided (60 back-to-back calls on the same buffer). Only the
   143-508 MB rows are DRAM-honest. A decode step streams 0.5-5 GB per token, so
   the DRAM rows are the ones to plan against.

## 0b. Measurement validity (read before trusting any per-op number)

`KRK_TIME` records one event pair per op. On this Windows HIP stack a record
costs ~10-27 µs, so timing *all* ops inflates the run 4.6x (6504 ms vs the true
1421 ms for the 0.8B) and every op whose measured span is under ~30 µs is
dominated by the measurement. Evidence, from the same model:

| `KRK_TIME` | timed ops | wall | delta vs 1421 ms true | per op |
|---|---|---|---|---|
| (unset) | 0 | 1421 ms | — | — |
| `1` (all) | 138576 | 6504 ms | +5083 ms | 37 µs |
| `gemm` | 43282 | 2975 ms | +1554 ms | 36 µs |
| `delta_rule,conv1d_silu,l2norm` | 18432 | 2214 ms | +793 ms | 43 µs |
| `rmsnorm` | 17152 | 1705 ms | +284 ms | 17 µs |
| `attention` | 1536 | 1475 ms | +54 ms | 35 µs |
| `download_f32,upload,rope` | 2112 | 1537 ms | +116 ms | 55 µs |

Consequences, and they are rules, not caveats:

1. A per-op row for a sub-30 µs op is an **upper bound**, never a measurement.
2. The honest instruments are (a) `--bench`'s PDH GPU-engine sampling — it
   says whether the device is busy at all, (b) wall-clock deltas between
   filtered runs, (c) the amortized harness in §5.2.
3. The op *count* is trustworthy: 138576 ops / 256 tokens = **541 ops per
   token** on the 0.8B, against a 10.3 µs per-op budget (5.55 ms/token). That
   arithmetic is the single most useful number in this file.

---

## 1. Ranked bottlenecks

Ranked by measured end-to-end impact, not by microbenchmark.

### B1 — Dense decode is call-count-bound, and Q6_K is instruction-bound (both proven)

`Qwen3-8B Q4_K_M` keeps the compute engine **91.7% busy** at **299 GB/s** of
weight traffic. The harness explains that without a bandwidth defect: the
per-shape kernels are at 301-674 GB/s, and the *small* shapes are dominated by
the 2.65 µs launch floor (34-43% of each 6-8 µs call). The model streams those
matrices once per token, so the only ways forward are fewer calls per token
(fuse), or wider kernels per call. B1 is therefore: **fuse the projections into
fewer, larger calls, and add a Q6_K/Q5_K decode fast path.**

For Q6_K the cause is proven by the shape that fits in cache and still runs at
143 GB/s: the chunk decode is instruction-bound. The fix is a Q6_K (and Q5_K)
decode using packed f16 math (`v_pk_mul_f16` / `v_pk_fma_f16`, the same
instruction family `krk_hip.hpp` already exposes as `d_dot2_pk`) with the 6-bit
unpack done branch-free, and a precomputed `d * scale` in f16 so the per-value
f32 convert chain disappears. Falsifying test for the fix: the harness row for
4096x4096 Q6_K must go from 143 GB/s to >400 GB/s, and the 208 MB head row from
195 to >450.

### B2 — The prefill GEMM is at 22% of its memory-bound ceiling: staging on the
critical path (measured, cause corroborated by the kernel's own §2c note)

The card: 64 CU, 2970 MHz boost, **640 GB/s** DRAM (20 Gbps x 256-bit), 64 MB
Infinity Cache, 48.7 TFLOP/s FP32/fp16 *vector* — and a matrix rate well above
that, which is why WMMA is not the suspect. What is measurable and decisive:

- rocBLAS' f16 GEMM at 32x4096x4096 takes 43.4 µs = 773 GB/s of weight traffic:
it is **memory-bound**, right at the DRAM envelope.
- Our Q4_K GEMM at the same shape takes 68.5 µs and reads **3.5x fewer bytes**,
so its memory-bound ceiling is ~72 TFLOP/s (9.44 MB per 1.07 GFLOP at 640
GB/s). We measure **15.67 TFLOP/s at 137 GB/s = 22% of the ceiling, 4.6x of
headroom** — and that is the honest version of "4 to 9x slower".
- Reading 3.5x fewer bytes while running 1.58x slower than rocBLAS leaves
exactly one bound: **time spent before the WMMA units see data**. The kernel
header already measured it ("staged dequant = 55-97% of prefill GEMM time"),
and `wmma_cfg` (BM=32 / BN=64 / BK=128, 256 threads, one 16x16 accumulator per
wave) writes the dequantized tile to LDS and then reads it back,
single-buffered, with no producer/consumer split.

So the prefill fix is not "use WMMA" (it is used, and it is fast) and not a
bigger tile alone: it is taking the dequant off the critical path —
double-buffered LDS, producer/consumer warp split, and eventually decoding the
B fragment straight into VGPRs so the LDS bounce disappears. The falsifying
test is the harness row: rows=32 Q4_K 4096x4096 must go from 15.7 to >40
TFLOP/s (>= 350 GB/s of weight traffic), and rows=128 must scale with it.
Note this buys time-to-first-token, not decode; decode's Q4_K GEMV is already
at 89-105% of DRAM.

### B3 — MoE is not compute-bound at all (proven, cause unknown)

`Qwen3-MoE 4x0.6B`: compute engine **7.3% busy**, decode 69 ms/token for 920 MB
of weights (13 GB/s — disk/page speed, not device speed). Prefill is equally
bad (184 tok/s, 300 ms for 55 tokens). Whatever the MoE path is doing, the GPU
is idle 93% of the time; this is a structural defect in expert residency or in
per-expert host round trips, not a kernel speed problem. The user's own report —
"it still used 14 GB, not expert-only loading" — is consistent: the residency
budget is not doing what it says.

### B4 — Per-token sync point: the logits copy (proven, measured with the
synchronisation trap)

`download_f32` copies the whole logits row device->host **every token** and
softcaps on the host. With the new idle column (device time between the end of
one op and the start of the next, i.e. the queue draining and waiting), on
`Qwen3-8B Q4_K_M`:

| | calls | span | idle before it | per token | share of a 16 ms token |
|---|---|---|---|---|---|
| `download_f32` | 256 | 185.6 ms | **688.7 ms** | 3.4 ms | **21%** |
| `attention` | 9216 | 282.1 ms | 137.2 ms | 1.6 ms | 10% |
| `rmsnorm` | 18688 | 183.9 ms | 2814 ms | 11.7 ms | (instrument-dominated, see 0b) |

The 8B is 91.7% compute-busy *overall*, yet one op per token forces a full
pipeline drain and ~2.7 ms of device idle while the host samples and relaunches.
This is a fixed tax: ~0.6 ms/token on the 0.8B, ~3.4 ms on the 8B, unchanged as
models grow.

Also measured, and measured honestly (large spans): `rope` costs 100-200 ms of
host submit time per 9216 calls across a run, and its filtered wall delta is
small; it is a latency problem, not a throughput one.
2. `rope`, 1536 calls: **585 µs of host time per call** (staging + sync),
   6 calls per token. Even if most of it is waiting rather than working, it is a
   serialization point on a queue that is already only 30% busy.
3. 541 ops/token at a 10.3 µs budget: even at a 2 µs launch, op count alone is
   1.1 ms of a 5.55 ms token. This is why the small models are the honest
   signal, exactly as the user said — and why halo fusions
   (`attn_fused_chain`, `gemm_group`, `fused_layer_`) pay for themselves.

### B5 — Attention at decode context — CLOSED, twice over

**Status: fixed.** The cause was not the lane layout this section suspected.
Two independent defects, both in the launch shape, both found by `--profile`:

1. `attention_decode_supported()` accepted head widths 64 and 128 only, so
   Qwen3.5-0.8B (hd=256) never reached the decode kernel and fell to the tiled
   kernel, whose grid is (n_head, n_tok) — 8 blocks on 64 CUs.
2. The decode kernel itself launches one block per head, which is also 8
   blocks for that model.

Both are now fixed: hd=256 supported (VPT=8), and the key range split across
blocks with the per-chunk (m, l, O) triples merged in a second launch — the
same split-K shape the GEMMs use. One attention call at ~4900 keys went
4.80 ms -> 0.67 ms (2.1 -> 10.0 GB/s of KV traffic), and decode went from
33.2 to 356.8 tok/s at ctx 4800 while the 64/128-width models are unchanged
(interleaved A/B on the 8B: 95.5 vs 95.3 tok/s, i.e. noise — the 8B takes
n_split == 1 at its context, as intended).

The lane layout was already correct and needed no work: lanes cover channels,
so a warp's KV read is contiguous.

### B5 (historical) — Attention at decode context

`attention` measured 190 µs/call on the 0.8B when the KV traffic at ~311 keys is
~3.6 µs. Even discounted for instrumentation (its filtered wall delta was only
+54 ms over 1536 calls = 35 µs/call), it is off by an order of magnitude. The
kernel assigns **one thread per key** and each thread then reads the whole
head vector; the KV read is per-thread strided. Suspected cause, test by
rewriting the lane layout (T5).

---

## 2. Causes at a glance

| | proven | suspected | unresolved |
|---|---|---|---|
| Dense decode | device-bound at 91.7% busy, 299 GB/s of 710 | H1 dequant ALU, H2 L1 transactions, H3 occupancy | which of H1-H3 dominates |
| Prefill | 6.2 TFLOP/s of 24.7 | LDS bounce + single buffering + small tile | split-K policy actually helping |
| MoE | 7.3% busy, 13 GB/s, unusable | expert residency/paging, per-expert host round trips | everything about the resident set |
| Small models | 541 ops/token, 10.3 µs budget, 30.6% busy | per-op fixed cost + sync points | how much is launch vs sync |
| Attention | 35-190 µs/call vs ~4 µs of work | one-thread-per-key layout, block reduction per 128 keys | floor with a warp-per-head kernel |
| Long ctx | not measured | KV cache layout/stride | — |

---

## 3. Tasks

Ordered so each one either produces a number the next one needs, or lands a
proven win. Every task ends with: oracle -> `kraken-tests` -> `--bench` ->
`scripts/coherence_check.sh` (text must stay identical).

**T1. Land the correctness work.** Coherence fix, `KRK_DUMP` + `dump_diff`,
`KRK_TIME` + filter, coherence gate with the instruct template, this plan.
No perf claims ride on it; it unblocks every measurement below.

**T2. Build the amortized harness** (`tools/bench_ops`, links `krk_hip` like
`kraken-oracle`). For each format {Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, IQ4_XS} and
each real shape from the local models: time `N=200` back-to-back calls and
`N=1`, take `(T200-T1)/199` as the marginal cost, and time an empty kernel the
same way for the launch floor. Report µs, GB/s and TFLOP/s. **This is the
instrument that decides H1-H3**; nothing in §4 starts before it exists.

**T3. GEMV (B1).** Test H1/H2/H3 with the T2 harness, then implement the winner
as a new kernel: 16-byte-per-lane contiguous loads, chunk dequantization
distributed from LDS or by shuffle, K split across 2-4 warps for latency
hiding, `dot_chunks` kept as the verified per-format dequant core.
*Validation*: T2 says >= 1.3x off the 710 GB/s yardstick, and
`kraken-i8k`/oracle numerics unchanged. *Expected*: 2-4x decode on dense.

**T4. Prefill WMMA (B2).** In order: BN=128 / BK=64 with double-buffered LDS;
then a producer/consumer split so staging of tile k+1 overlaps WMMA on tile k;
then dequantize straight into B fragments in registers, removing the LDS bounce.
*Validation*: T2 TFLOP/s at 32 and 512 rows, `scripts/check_lds.sh` for bank
conflicts, `kraken-tests` numerics.

**T5. Attention (B5).** Warp-per-(head, q-token) for decode with lanes across
channels (vectorized KV reads + shuffle reduce); keep the existing per-key
layout for prefill. *Validation*: T2 at 2K/8K ctx; coherence.

**T6. Per-token fixed costs (B4).** Sample on device (top-8 candidates or the
full sampler) and copy back 8 values instead of the 151k row; upload the RoPE
frequency table once per model instead of per call; make the position a kernel
argument. *Validation*: `download_f32`/`rope` rows disappear from KRK_TIME, and
`--bench` decode improves on every model (it is a fixed tax).

**T7. MoE residency (B3).** First *trace* it: KRK_TIME on the MoE ops, expert
cache hit/evict counters in `--info`, and the copy-engine PDH line. Then make
the resident set count expert rows actually touched, pin the top-k per layer
from the first N tokens, router top-k on device, and one batched transfer per
step instead of per-expert. *Validation*: compute-busy goes from 7% toward
70%+, decode 13 -> 50+ tok/s on the 4x0.6B, and `Qwen3-30B-A3B` becomes usable.

**T8. Expert-only VRAM test on the large models** (GLM-4.7-Flash, Laguna-S-2.1,
K2-Horizon): confirm that a model whose weights exceed VRAM loads only the hot
expert rows, that `--expert-cache-mb`/`-slots` move the number they claim to,
and that no whole-model upload happens (the "14 GB used, not expert-only" report
is the acceptance test). *Validation*: process VRAM via `--info` while loading
a >16 GB model, plus a decode run that stays within budget.

**T9. Architecture pass on the remaining model types** (the user's "/architect
add all model types properly"): every file in the model folders gets a verdict
(runnable / refused with a named reason), including MTP and dflash/dspark draft
blocks; the 4 refused-by-format models get named dequantizers or a documented
refusal; LLaDA (`diffuse`) gets its loader schema + block-diffusion sampler per
[docs/LLaDA.md](LLaDA.md). Publish the resulting table in
[docs/model-inventory.md](model-inventory.md).

**T10. Capability backlog** (after the above, one at a time): sliding-window
attention, attention-logit softcap (landed), per-layer input embeddings, YaRN
rope scaling, RadixAttention/prefix cache, the paper list in
[docs/implementing-papers.md](implementing-papers.md) (Tail-Replay first — it
repairs the recurrent `kv_rollback` refusal for the GDN models).

---

## 3b. The decode step, decomposed (0.8B, 24 layers, per token)

From the dispatch counts (filters do not change counts), 541 device ops per
produced token against a 10.3 µs budget each:

| op family | calls/token | share of the budget |
|---|---|---|
| `gemm(gemv)` | 168 | the only family whose span is big enough to trust |
| `rmsnorm` | 67 | 10-20 µs each at best |
| `silu_mul` / `add_inplace` / `sigmoid_act` / `scale_act` / `add_bias_cols` / `scale_cols` | 42 / 48 / 24 / 18 / 18 / 18 | all vector-shaped, all tiny |
| `l2norm` / `delta_rule` / `conv1d_silu` / `softplus_act` | 36 / 18 / 18 / 18 | GDN chain |
| `attention` / `rope` / `qk_norm` / `kv_append` / `q3_split` | 6 each | full-attention layers |
| `download_f32` | 1 | the sync point of B4 |
| `gemm_group` | 8 | fused q/k/v and gate/up |

Two conclusions the numbers support: (a) the elementwise/norm families are pure
op-count tax — 250+ ops per token whose individual spans are below the
measurement floor, so the fix is fusion and fewer, wider ops; (b) the one op per
token that *is* expensive is the logits copy, because it is a sync point.

## 4. Acceptance criteria

- T2 table exists for every format/shape used by the local models.
- Dense decode Q4_K stays at or above 550 GB/s (it is already there); the win
  comes from Q6_K/Q5_K (143/352 -> 400+) and from fewer calls per token.
- Prefill rows=32 Q4_K >= 40 TFLOP/s (from 15.7) and >= 350 GB/s of weight
  traffic; rows=128 must scale with it.
- Small models: compute-busy above 60% on the 0.8B, and decode >= 400 tok/s.
- MoE 4x0.6B: decode >= 50 tok/s, compute-busy above 60%.
- `scripts/coherence_check.sh`: 9/9, byte-identical text, after every task.
- No task is called done on a single measurement: each needs a repeat run on a
  different day-order (before/after, interleaved) to rule out thermal drift.
