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

### B1 — m=1 expert GEMV at 3.9% of memory peak (proven)

`gemm(gemv)` is **1234 calls, 42.2 ms, 1,482 MB** = **35.1 GB/s**. The card's
GDDR6 peak is ~896 GB/s, and an op this small should be launch/latency bound
rather than bandwidth bound — 1.2 MB at peak is 1.34 us against a measured
**34.2 us per call**, so it is **25x off** the roofline.

Launch geometry is `gemv_kernel<256><<<(n_out+7)/8, 256>>>`. For a
1024-row expert projection that is **128 blocks of 256 threads on 64 CUs** —
2 blocks/CU, ~16 warps/CU. Far too little occupancy to hide memory latency.

The structural cause: the expert path *does* group by expert (`group_rows_`),
but at decode `n == 1`, so each of the top-8 experts receives exactly **one
row**. Grouping cannot batch anything, and the MoE shape forces **960 separate
m=1 GEMV launches per token** — 1234 including attention and the shared expert.

A plain LRU(4340) replay of the access trace scores 6,741 COLD / 0 HOT, against
kraken's actual 4,590 COLD / 6,003 HOT, so the residency policy is not the
problem and there is no cheap win in the cache.

### B2 — transfer ops carry 27.7% of device time (proven, partly addressed)

`upload`/`upload_i32`/`upload_paged`/`download_f32` total 30.4 ms. The host
column is the tell: `upload_paged` is 21 calls costing 4.7 ms device but
**15.1 ms host**, and `warm_read` is 60 calls costing 2.8 ms device but
**11.5 ms host**. These are staging operations waiting on file I/O.

### B3 — `rope` host stall (proven cause, fixed, small effect)

Before the fix: 40 calls, **51,827 us host = 1,296 us/call**. Control: the same
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

### B4 — expert slot capacity (proven NOT a bottleneck)

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
2. Decode-step expert GEMVs run at 35.1 GB/s = 3.9% of peak, at 128 blocks on
   64 CUs, 25x off the latency roofline.
3. All COLD misses are first touches; the working set is 4,590 keys and slot
   capacity is not the constraint.
4. The ROCm reference defaults to `--repack`, which costs ~64 GiB of host RAM
   for 1.08x decode.

**Suspected, not yet proven.**

5. That the 42.2 ms is split between wave-quantisation (128 blocks is not a
   multiple of 64 CUs, so a whole wave idles) and per-launch fixed cost. The
   discriminator is a split-K variant: if wave quantisation dominates, split-K
   recovers most of it; if fixed cost dominates, it does not.
6. That `upload_i32` (13.3 ms, 312 calls, 42.7 us each) is mostly host-side
   `assign()` + upload latency rather than device work. The host column is only
   1191 us total, which argues against it, so this is weakly supported.

**Unresolved.**

7. Why Kraken answers this prompt incorrectly while the reference does not. Not
   addressed by anything in this document.
8. Whether 8 top-k experts across 40 layers have exploitable temporal locality
   that would let a prefetcher convert B1's latency into bandwidth.

## 6. Plan

| # | Change | Expected | Validation |
|---|---|---|---|
| 1 | Multi-expert GEMV: one launch for all top-k experts of a layer, blocks split across experts | 960 -> 120 launches; B1 at full occupancy | identical text hash; ms/tok |
| 2 | Split-K on the expert GEMV | tests hypothesis 5 | identical text hash |
| 3 | Keep `upload_paged`/`warm_read` host waits but overlap them with the next layer's attention | targets B2's 26.6 ms host | ms/tok, transfer MB |
| 4 | DFlash speculative heads (`laguna-s-2.1-DFlash-Q4_K_M.gguf`, 652 MB) | amortises expert reads over >1 token; the highest-value item for a bandwidth-bound MoE | accept rate, tok/s |

Item 4 is the one with the largest expected effect and the least code: a
speculative head that accepts 3 tokens per pass divides the per-token expert
traffic by ~3, which attacks B1 and B2 together. Poolside ships the heads;
Kraken currently **refuses** the `dflash` architecture as "draft file, not a
model".

## 7. Not measured

- Laguna-S-2.1 (68.09 GiB, 3 shards) runs at 0.9 tok/s with garbage output, but
  **the llama.cpp reference for it is not a valid comparison**: at most 23.4% of
  a 68.09 GiB model fits in 15.9 GiB of VRAM, so 76.6% was CPU-computed. No
  GPU-only baseline exists for this model.
- KV cache is entirely VRAM (`engine.cpp:223-225`, one `alloc()` per tensor). At
  96 KiB/token a 262,144-token context needs 24 GiB of KV, which does not fit.
  There is no host KV tier, so a 2 GiB-VRAM + 6 GiB-host split would cap
  context near 85,000 tokens.