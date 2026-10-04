# Laguna MoE performance analysis

Where a decode token goes on `laguna-xs2-Q4_K_M`, measured on the RX 9070 XT
(gfx1201, 64 CU, 15.9 GiB). Every number below is from a file in
`docs/traces/`, produced by the commands in that directory's README.

Reference: `H:/LLAMA-bins/rocm10/llama-cli.exe`, build `b11146-7fe450e19`,
prompt `"what is the capital of france?"`, `-n 24 --temp 0`, ChatML.

## 1. Headline

| | llama.cpp ROCm | Kraken | ratio |
|---|---:|---:|---:|
| prompt tok/s (`--no-repack`) | 25.2 | 7.9 | 0.31x |
| decode tok/s (`--no-repack`) | 14.4 | 10.6 | 0.74x |
| decode tok/s (`--repack`, default) | 15.5 | — | — |

Kraken answers this prompt **incorrectly** ("Okay, the user is asking about the
capital of France. Let me think…") while the reference answers correctly. That
is a correctness defect, not a speed one, and it is tracked separately.

## 2. The reference build needs a caveat

`--repack` is **default-on** in this ROCm build (`LLAMA_ARG_REPACK`). It
repacks every tensor to `q4_K_8x8`, which needs a full extra host copy:

| config | prompt tok/s | decode tok/s | peak host RAM |
|---|---:|---:|---:|
| `--repack` (default) | 47.8 | 15.5 | ~64 GiB |
| `--no-repack` | 25.2 | 14.4 | 12.5 GiB |

Repack buys **1.9x on prefill and 1.08x on decode**. That asymmetry identifies
it as a matmul tile-layout optimisation: it helps compute-bound prefill and does
almost nothing for weight-streaming-bound decode. **Decode comparisons must use
`--no-repack`**, which is also the only configuration that fits a 24-32 GiB host.

## 3. Per-function cost of one decode step

`KRK_PROFILE_STEPS=1`, last complete step, 24-token run. 3,457 ops in the step.
Device spans are 111.4 ms of a 195.3 ms instrumented wall; the recorder itself
costs ~27 us/op, so the **uninstrumented** step is **94.8 ms**.

| op | calls | dev us | host us | share |
|---|---:|---:|---:|---:|
| `gemm(gemv)` | 1234 | 42202 | 4513 | 37.9% |
| `upload_i32` | 312 | 13340 | 1191 | 12.0% |
| `upload` | 312 | 10563 | 1124 | 9.5% |
| `silu_mul` | 352 | 8944 | 1288 | 8.0% |
| `gather_rows` | 312 | 6600 | 1716 | 5.9% |
| `scatter_axpy` | 312 | 6384 | 1114 | 5.7% |
| `upload_paged` | 21 | 4748 | 15134 | 4.3% |
| `warm_read` | 60 | 2844 | 11511 | 2.6% |
| `download_f32` | 39 | 2749 | 3705 | 2.5% |
| `rmsnorm` | 81 | 2375 | 324 | 2.1% |
| `add_inplace` | 119 | 2003 | 869 | 1.8% |
| `attention` | 40 | 1852 | 310 | 1.7% |
| `gemm_group` | 21 | 1121 | 260 | 1.0% |
| `rope` | 40 | 964 | 187 | 0.9% |

Rollup by component: `projections` 40.5%, `transfer` 27.7%, `other` 14.5%,
`ffn-activate` 7.5%, `attention` 5.3%, `norms` 2.0%, `residual` 1.6%.

## 4. Ranked bottlenecks

### B1 — m=1 expert GEMV — RETRACTED, it was a measurement artifact

**This finding was wrong and is withdrawn.** `gemm(gemv)` appeared to be the
dominant cost at **1234 calls, 42.2 ms, 35.1 GB/s, 3.9% of peak, 25x off the
roofline**. `tools/probe_gemv.hip` runs the same kernel back to back with
nothing else queued, and it is fast:

| shape | grid | µs/call | GB/s |
|---|---:|---:|---:|
| decode gate/up `[1024,2048]` | 128 | **2.78** | **424.6** |
| decode down `[2048,1024]` | 256 | 3.54 | 333.6 |
| large `[16384,2048]` | 2048 | 22.75 | 829.7 |

Two things follow.

**The kernel already exceeds the 200 GB/s target** by more than 2x at the decode
shape. There is nothing to win there, and the 830 GB/s the large shape reaches
matches the 710 GB/s rocBLAS reference recorded in `op_time.hpp`, so it is a
real ceiling on this machine rather than a probe artefact.

**The trace could not have measured it.** The profile reports its own floor:
**19.63 us of device time per op**, for 3,457 ops in a step — **67.9 ms of the
111.4 ms "device spans" is the recorder itself**. A 2.78 us kernel is seven
times *below* that floor, so the 42.2 ms attributed to `gemv(gemv)` is mostly
the cost of observing it.

This is the trap the enforcement rules name: *never equate aggregate device
time with individual invocation latency*. The aggregate said gemv owned 44% of
the step; the invocation latency says the math is ~4% of it.

**What is left is real and is not the math.** A decode step is 3,457 ops in
94.8 ms, or 27 us per op. The kernel needs 2.78 us. The remaining ~24 us per op
is submission and scheduling, so the bottleneck is the **op count and the host
path**, not the device. Collapsing 960 expert launches into fewer, larger
launches is still the right direction — but as an op-count fix, worth perhaps
10-20%, not the 5x the GB/s figure implied.

`KRK_TIME` cannot resolve anything below ~30 us. Per-op device time for the
small ops must come from isolated probes (as here) or from wall-clock A/B, not
from the aggregate table.

### B2 — expert staging host waits (RETRACTED, also an artifact)

The host column appeared to show `upload_paged` at **721 us/call** (15.1 ms per
step) and `warm_read` at **192 us/call** (11.5 ms per step) — 26.6 ms of a 93 ms
token in synchronous staging. The host floor is only 0.63 us/op, so those looked
real.

They are not. An uninstrumented run does **9 promotions totalling 18 MiB for
the entire 54-token generation**. The 21-per-step figure came from the profiled
run, where the recorder itself perturbs the cache into evicting far more. Both
this and B1 are the same failure: reading a cost out of an instrumented run and
treating it as a property of the code.

### B3 — the MoE expert path is not the bottleneck at all (proven)

`--expert-stub` removes the expert GEMMs entirely (the routers still run). If
the MoE math mattered, decode would speed up substantially.

| arm | rep 1 | rep 2 |
|---|---:|---:|
| full | 93.6 ms/tok | 92.9 ms/tok |
| `--expert-stub` (experts removed) | 93.7 ms/tok | 90.8 ms/tok |

**Removing 960 expert GEMVs, 312 `gather_rows`, 312 `scatter_axpy` and 352
`silu_mul` — 56% of the step's ops — changes decode by 0.1-2%.** With
`probe_gemv` putting each GEMV at 2.78 us, the entire expert computation is
~3 ms of a 93 ms token.

Every optimisation aimed at the expert kernels, the expert cache, the expert
GEMV geometry and expert prefetch is therefore aimed at ~3% of the cost. This
supersedes B1, B2 and B4 as the finding that matters.

### B4 — a fixed ~90 ms per token independent of expert work (proven, unexplained)

Since removing more than half the ops changes nothing, the cost is not
proportional to the op count. The GPU is idle ~93% of the run (`--bench` reports
6.8% compute busy on this model). A decode token takes 93 ms and the sum of its
kernels is a few ms, so **roughly 90 ms is spent outside the kernels**, in a
per-token serialised path. That is the actual bottleneck and it is still
undiagnosed.

The leading candidate is a host round-trip per token — the sampler needs to
learn the sampled token before it can build the next step, so any
download/synchronise/upload in that path is a full pipeline drain exactly once
per token. The next investigation is to find that sync, not to tune MoE
kernels.

### B5 — `rope` host stall (proven cause, fixed, small effect)
kernel on the non-hybrid `Qwen3-MOE-4x0.6B` costs **3.2 us/call** — a 405x
difference on an identical kernel.

Cause: laguna interleaves two rope tables (`inv_freq_` on full-attention layers,
`inv_freq_swa_` on windowed ones), and `stage_freq` cached only one. Every layer
boundary missed and took a **blocking `hipMemcpy`**, which drains the queued
stream. Four LRU slots fixed it: **51,827 us -> 187 us**.

End-to-end this bought only **98.9 -> 94.8 ms/tok (+4.1%)**, because the
51.8 ms was queued device work being *waited on*, not work skipped. This is the
clearest example in this document of why host time and device time must not be
added together or treated as interchangeable.

### B6 — expert slot capacity (proven NOT a bottleneck)

Interleaved A/A/B/B, because sequential arms differ by more than the effect
under test (two back-to-back default runs measured 10.6 and 6.6 tok/s from page
cache alone):

| arm | ms/tok | slots resident | COLD misses |
|---|---:|---|---|
| A1 | 94.9 | 4340/4340 | 4590 (43.3%) |
| B1 | 102.2 | **4590/4700** | 4590 (43.3%) |
| A2 | 96.0 | 4340/4340 | 4590 (43.3%) |
| B2 | 103.2 | **4590/4700** | 4590 (43.3%) |

Holding the entire working set resident removes **zero** COLD misses and costs
**7.3%** in VRAM pressure. The access trace explains why: there are exactly
4,590 distinct `(layer, expert)` keys, so **every COLD miss is a first touch**
and no capacity can avoid a first touch.

This is the evidence that the apparent "41% of acquisitions go to NVMe" is a
warmup cost, not a steady-state miss rate.

## 5. Root causes

**Proven.**

1. `stage_freq` single-entry cache thrashes on hybrid rope tables, and the miss
   path is a blocking `hipMemcpy`. Fixed; 277x host reduction, +4.1% end-to-end.
2. The expert GEMV kernel is **not** a bottleneck: 2.78 us at the decode shape
   (424.6 GB/s) against a 19.63 us/op measurement floor. The earlier 44% claim
   was the recorder, not the kernel.
3. A decode step is **3,457 ops in 94.8 ms = 27 us/op** while the dominant
   kernel needs 2.78 us. Op count and host submission dominate decode.
4. All COLD misses are first touches; the working set is 4,590 keys and slot
   capacity is not the constraint.
5. The ROCm reference defaults to `--repack`, which costs ~64 GiB of host RAM
   for 1.08x decode.

**Suspected, not yet proven.**

6. That the ~24 us/op gap between the kernel and the step is host submission
   (`hipEventRecord` x2 plus the launch call per op) rather than device-side
   queue drain. The discriminator is a run with the per-op recorder off and
   wall-clock deltas around a single op class.
7. That folding the 8 per-layer experts into one launch is worth 10-20%, by
   cutting launches rather than by making the kernel faster.

**Unresolved.**

7. Why Kraken answers this prompt incorrectly while the reference does not. Not
   addressed by anything in this document.
8. Whether 8 top-k experts across 40 layers have exploitable temporal locality
   that would let a prefetcher convert B1's latency into bandwidth.

## 6. Plan

| # | Change | Expected | Validation |
|---|---|---|---|
| 1 | Multi-expert GEMV: one launch for all top-k experts of a layer | 960 -> 120 launches; cuts op count, not kernel time | identical text hash; ms/tok |
| 2 | Measure the host path directly (wall-clock A/B, recorder off) | sizes hypothesis 6, which is now the leading candidate | ms/tok at fixed op count |
| 3 | Keep `upload_paged`/`warm_read` host waits but overlap them with the next layer's attention | targets B2's 26.6 ms host | ms/tok, transfer MB |
| 4 | DFlash speculative heads (`laguna-s-2.1-DFlash-Q4_K_M.gguf`, 652 MB) | amortises per-token work over >1 token | accept rate, tok/s |

Item 4 now looks strongest: the math is only ~4% of a decode step, so the wins
available in the kernel are small, while a speculative head that accepts 3 tokens
per pass removes **two thirds of every per-token cost** regardless of what it
is. Poolside ships the heads; Kraken currently **refuses** the `dflash`
architecture as "draft file, not a model".

## 7. Not measured

- Laguna-S-2.1 (68.09 GiB, 3 shards) runs at 0.9 tok/s with garbage output, but
  **the llama.cpp reference for it is not a valid comparison**: at most 23.4% of
  a 68.09 GiB model fits in 15.9 GiB of VRAM, so 76.6% was CPU-computed. No
  GPU-only baseline exists for this model.
- KV cache is entirely VRAM (`engine.cpp:223-225`, one `alloc()` per tensor). At
  96 KiB/token a 262,144-token context needs 24 GiB of KV, which does not fit.
  There is no host KV tier, so a 2 GiB-VRAM + 6 GiB-host split would cap
  context near 85,000 tokens.