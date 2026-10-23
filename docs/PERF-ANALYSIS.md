# Laguna MoE performance analysis

Where a decode token goes on `laguna-xs2-Q4_K_M`, measured on the RX 9070 XT
(gfx1201, 64 CU, 15.9 GiB). Every number below is from a file in
`docs/traces/`, produced by the commands in that directory's README.

Reference: `H:/LLAMA-bins/rocm10/llama-cli.exe`, build `b11146-7fe450e19`,
prompt `"what is the capital of france?"`, `-n 24 --temp 0`, ChatML.

## 1. Headline

**Update (2026-10-10): Laguna XS.2 MoE tiering, a self-draft cache fix and an
oracle comparison.** This live probe fixes the prompt and generation length at
`The history of computing`, 64 greedy tokens, `--ctx 512`, RX 9070 XT. Kraken's
fresh expert scan (68 positions, full forward) replaced a stale index whose
`source_size` was 0; the loader had correctly refused that index and fallen back
to even-order warmup. Four order-balanced pairs compare the default verified
ranked warmup with warmup disabled, preserving each arm's order slots:

| arm | median decode | cache per acquire | file bytes/run | stdout SHA-256 |
|---|---:|---|---:|---|
| ranked warmup | **21.3 tok/s** (47.0 ms/step) | 95.3% HOT, 4.7% WARM | **0 MiB** | `4703c315…` |
| warmup off / even first touches | 10.1 tok/s (99.0 ms/step) | 78.0% HOT, 22.0% WARM | 8,337 MiB (130.27 MiB/token) | `4703c315…` |

Each paired run improved 104–119%, and all eight outputs hash identically. The
run did not read the model file with warmup enabled because the 18,102 MiB
pageable WARM tier already held every routed slot and top experts were promoted
into the 12,745 MiB VRAM budget. It still paid ~0.6–0.8 seconds in WARM-to-VRAM
promotions/allocator work over 64 tokens; these are transfers of experts that
are not HOT, not CPU-computed experts. The engine has no CPU expert arm enabled
here.

**Cross-engine check:** llama.cpp's ROCm `ROCm0` list identifies this same RX
9070 XT, and verbose loading says `offloaded 41/41 layers to GPU`, model buffer
19,221 MiB. Three identical-shape runs report **31.01–31.85 tok/s decode**
(median 31.04) at the same context/prompt/token count; Kraken after corrected
warmup is still **1.46x slower**, so tiering fixed the stale-ranking penalty but
is not the whole remaining engine gap. Caveat: llama's repack is default-on
(`REPACK=1` in system info), while Kraken keeps the original GGUF layouts; this
is a representative user-visible same-GPU comparison, not an isolated kernel
A/B. The earlier body of this document records older, conflicting model status
and performance experiments; treat this update as the newer run for this exact
command, not as a global model-family claim.

`Laguna-S` is a 68 GiB three-shard model, so all weights cannot fit in this
16 GiB GPU; the intended policy is deliberately hot experts on GPU, a pageable
WARM working set in host RAM, and only a cold tail on storage. Its shard path,
source-specific index and measured residency remain follow-up work.


| | llama.cpp ROCm | Kraken | ratio |
|---|---:|---:|---:|
| prompt tok/s (`--no-repack`) | 25.2 | 7.9 | 0.31x |
| decode tok/s (`--no-repack`) | 14.4 | 10.6 | 0.74x |
| decode tok/s (`--repack`, default) | 15.5 | — | — |

The Kraken decode figure is the 10.6 tok/s measured before this round of work.
Section 10 measures it again on this machine's current state and takes it to
**10.3 tok/s from 9.0** by two changes that do not touch the arithmetic. The
absolute number moves with the machine (the same binary measured 10.6 tok/s in
an earlier session and 9.0 tok/s in this one), which is why every comparison in
section 10 is an interleaved A/B on one binary rather than a number against the
table above.

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

### B3 — the MoE expert path is not the bottleneck at all (RETRACTED: it is 91%)

**This finding was wrong, and it was wrong because of a flag that does nothing.**
`--expert-stub` removes the expert GEMMs entirely (the routers still run), but
`Engine::set_expert_scan` stores `expert_stub_ = scan && stub`: the stub arm
only reaches the engine when `--expert-scan` is passed as well. The A/B below
measured the flag's effect on nothing.

Run on this binary, `--expert-stub` **alone**, laguna-xs2:

| invocation | acquires | COLD misses | ms/tok |
|---|---:|---:|---:|
| `--expert-scan` | 10602 | 4590 | 97.1 |
| `--expert-stub` alone | 10602 | 4590 | 98.8 |
| `--expert-scan --expert-stub` | — | — | **8.5** |

Interleaved, 3 repetitions, medians. The middle row is the same numbers as the
first, which is the signature of a no-op. With the second flag added the token
falls from **97.1 ms to 8.5 ms**.

So the expert path is **~89 ms of a 97 ms token (91%)**, not 3%, and the whole
plan that followed B3 (section 6) was aimed at the wrong 9%. `--expert-stub` on
its own is now a hard error rather than a silent no-op, because a flag that
quietly measures nothing is worse than no flag.

The direction of the error is worth stating plainly: three separate retractions
in this document (B1, B2, B3) all came from reading a number out of an
instrumented or mis-configured run and treating it as a property of the code.
The next revision of each was produced by an isolated probe or an A/B, not by
re-reading the trace.

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

6. That the ~90 ms sits in a per-token host round trip — the sampler must know
   the sampled token before the next step can be built, so a
   download/synchronise/upload in that path drains the pipeline once per token.
   The discriminator is to time the pre-sample and post-sample halves of a
   decode step separately with the recorder off.
7. That the gap is host submission (two `hipEventRecord` calls plus the launch
   per op) rather than device-side queue drain. These are not exclusive of 6 and
   only the step-halves timing separates them.

**Unresolved.**

8. Why Kraken answers this prompt incorrectly while the reference does not. Not
   addressed by anything in this document.
9. Whether 8 top-k experts across 40 layers have exploitable temporal locality
   worth a prefetcher. Note this is now a *second-order* question: a prefetcher
   only pays if the staging it removes is real, and the uninstrumented run does
   9 promotions for the whole generation.

## 6. Plan

The plan was rewritten after B3. Its first three items were all MoE-side and all
targeted work that `--expert-stub` shows is ~3% of the cost.

| # | Change | Expected | Validation |
|---|---|---|---|
| 1 | **Find the ~90 ms/token serialised path.** Time the pre-sample and post-sample halves of a decode step with the recorder off; if it is the round trip, batch the token or overlap the next step's staging | up to ~90 ms/token — the whole remaining budget | ms/tok at unchanged op count |
| 2 | DFlash speculative heads (`laguna-s-2.1-DFlash-Q4_K_M.gguf`, 652 MB) | amortises per-token cost over >1 token; also attacks item 1, because a round trip per *pass* rather than per token is still cheaper | accept rate, tok/s |
| 3 | Multi-expert GEMV: 960 -> 120 launches | bounded by the ~3 ms of expert math, so **<=3%** | identical text hash; ms/tok |
| 4 | Overlap `upload_paged`/`warm_read` with attention | **not currently justified** — the uninstrumented run does 9 promotions per generation. Revisit only if item 1 turns out to be staging | ms/tok, transfer MB |

Item 2 is the only change here that can pay regardless of what item 1 finds,
because a speculative head that accepts 3 tokens per pass removes two thirds of
*every* per-token cost — including an unknown one. Poolside ships the heads;
Kraken currently **refuses** the `dflash` architecture as "draft file, not a
model".

Item 3 is deliberately retained with a 3% ceiling attached rather than dropped,
so it is not re-proposed later as if it were still a 5x win.

## 7. Not measured

- Laguna-S-2.1 (68.09 GiB, 3 shards) runs at 0.9 tok/s with garbage output, but
  **the llama.cpp reference for it is not a valid comparison**: at most 23.4% of
  a 68.09 GiB model fits in 15.9 GiB of VRAM, so 76.6% was CPU-computed. No
  GPU-only baseline exists for this model.
- KV cache is entirely VRAM (`engine.cpp:223-225`, one `alloc()` per tensor). At
  96 KiB/token a 262,144-token context needs 24 GiB of KV, which does not fit.
  There is no host KV tier, so a 2 GiB-VRAM + 6 GiB-host split would cap
  context near 85,000 tokens.
## 8. The KRK_SPLIT timer as a race trigger — RETRACTED, it was never causal

**Nothing here reproduces, and the premise was wrong.** `KRK_SPLIT` does not
control where the clock reads land: in the decode loop both
`steady_clock::now()` calls are taken unconditionally and only the
*accumulation* into the split totals is gated on the flag
(`src/engine.cpp` decode loop). The flag's entire effect is a few floating-point
additions and one branch per token, so it cannot move a device queue.

Interleaved, 8 repetitions per arm, arms alternating run by run
(`docs/traces/` holds the original claim; the re-measurement is below):

| arm | runs | distinct outputs |
|---|---:|---:|
| `KRK_SPLIT` off | 8 | **1** |
| `KRK_SPLIT=1` | 8 | **1** |

16/16 identical, including the text hash. The original table:

| config | runs | distinct outputs |
|---|---:|---:|
| `KRK_SPLIT` off | 2 | 1 (`1def21d8`) |
| `KRK_SPLIT=1` | 3 | **3** (`dee2e8ab`, `3be5e476`, `b8151775`) |

The section that follows is kept as the record of what was believed and why,
not as a finding. The one lesson that survives is the method one: a probe that
changes the output by *printing* proves nothing about which code path ran, and
a 2-versus-3-run comparison is not evidence of anything. What is still open is
only whether some *other* timing perturbation can make this model nondeterministic
— no such perturbation is known today, and the workload is deterministic across
every arm measured in section 10.

| config | runs | distinct outputs |
|---|---:|---|
| `KRK_SPLIT` off | 2 | 1 (`1def21d8`) |
| `KRK_SPLIT=1` | 3 | **3** (`dee2e8ab`, `3be5e476`, `b8151775`) |

Three of three. Nothing else differs between those runs, so the only variable
is where two clock reads land relative to the pipeline drain. That is not a bug
in the probe -- it is the fastest reproduction of the open laguna
nondeterminism found so far, and it converts an intermittent failure into a
switch.

It also corrects an over-claim: the `KRK_TIME` tracer was verified
byte-identical on this prompt, but the split timer is **not** output-neutral
here, so its archived log's text is one of several valid variants and must not
be used as a reference output.

Practical consequence for this document: every `KRK_SPLIT` number below is
still valid, because a timing probe that lands a few microseconds differently
cannot change what the 96.84 ms split *is* -- forward versus fetch_logits. But
a `KRK_SPLIT` run must never be used to compare generated text.

## 9. (Historical) the "laguna race" — superseded by section 8

The previous revision of this section claimed the race was in the device argmax.
That was wrong for the reason section 8 now gives, and the section below is
kept only because its method notes (printing perturbs, dumping perturbs) are
still true of every timing measurement in this document.

The previous revision of this section claimed the race was in the device argmax.
**That was wrong, and it was wrong because of a confound I did not control.**
`--debug-topk` both selects host sampling *and* prints a line per token, and
the printing serialises the run. Determinism there proved nothing about which
sampling path was taken.

`--sample-host` was added to remove the confound: it selects host sampling with
no printing at all.

| config | runs | distinct outputs |
|---|---:|---|
| device argmax (default) | 3 | 3 |
| `KRK_SPLIT` + `KRK_WARM_DMA=0` | 2 | 2 |
| `KRK_SPLIT` + `KRK_WARM_DMA=1` | 2 | 2 |
| `KRK_SPLIT` + `KRK_WARM_DMA=2` | 2 | 2 |
| `KRK_SPLIT` + `KRK_STAGE_WAIT=1` | 3 | 3 |
| `KRK_SPLIT` + `KRK_SYNC_EXPERT=1` | 3 | 3 |
| `KRK_SPLIT` + `KRK_SYNC_MOE=1` | 3 | 3 |
| `KRK_SPLIT` + `--debug-topk` (prints) | 4 | **1** |
| `KRK_SPLIT` + `--sample-host` (no print) | 4 | **4** |

The last two rows are the whole argument: the only difference is whether the run
prints. Every configuration that does not slow the pipeline diverges; every one
that does is stable. The same explains the earlier "bit-identical dumps": 
`dump_row` calls `be_->sync()` before every stage, so the dump runs were never
measuring the unsynchronised path at all.

So what is established:

- The race is **timing-sensitive somewhere in the pipeline**, and any
  serialisation hides it. `KRK_SPLIT` reproduces it 3/3 reliably, which is the
  single most useful fact here.
- It is **not** the device argmax, **not** the warm/DMA staging path in any of
  its three modes, **not** the small-transfer staging wait, and **not** the MoE
  block or per-expert paging.
- It is **not** the model arithmetic, insofar as a fully synchronised run is
  bit-identical run to run.

What is **not** established, and was wrongly claimed before: where the race
lives. The remaining candidates are the deferred host reads
(`host_pending_` / `host_deferred_flush_`), the KV-cache writes, or an ordering
between the default stream and the non-blocking `paged_stream_` that
`hipStreamSynchronize` does not actually cover. The AGENTS.md note about
non-blocking streams having no implicit ordering against the default stream
still stands as the most likely shape; it has simply not been isolated.

`--sample-host` is kept as a correctness mitigation. It is not a fix: it was
shown to diverge 4/4 as soon as the printing confound was removed.

## 10. Where a decode token actually goes, and the two changes that came out of it

Everything above is the record of three wrong answers. This section is the
measurement that replaced them, and it is the first one in this document taken
with an interleaved A/B and an instrumented split rather than a trace read.

### 10.1 The expert path is 91% of the token

`--expert-stub` with `--expert-scan` (section B3) removes every expert acquire
and every expert GEMM. Interleaved, 3 repetitions:

| arm | ms/tok | tok/s |
|---|---:|---:|
| full | 97.10 | 10.30 |
| `--expert-scan --expert-stub` | **8.50** | **117.20** |

The dense path — 40 layers of attention, projections, norms, the router GEMMs,
the lm_head — is 8.5 ms. Everything else is the expert path, and it is where
every remaining millisecond is.

### 10.2 Inside the expert path: a four-way split

Four host-side accumulators were added to the expert cache and are printed as
`[stats ] expert-path` (`read` / `room` / `alloc` / `xfer`). One run, the
default configuration, prefill + decode = 6.3 s of wall:

| stage | ms | share | what it is |
|---|---:|---:|---|
| `read` | 1988.7 | 31.6% | `read_host`: 8.3 GiB out of the model file into WARM |
| `xfer` | 2170.4 | 34.5% | `upload_paged_batch`: WARM to VRAM, incl. the host wait |
| `alloc` | 1128.9 | 17.9% | `alloc_pooled`: the pooled-VRAM allocator |
| `room` | 27.3 | 0.4% | `make_vram_room`: eviction |

84% of the run is in those four lines. That is the shape of the problem: not
arithmetic, not kernels, not attention — moving 347 MiB/token of weights.

### 10.3 The file is not the limit, the copy is; and neither is what it looked like

`tools/probe_read.py` reads the model file the way the engine does, one handle
per thread, and finds the device saturates at **~3.8-4.1 GB/s at every thread
count from 1 to 32**. `tools/probe_promote.hip` then moves 4 GiB of 2.06 MiB
experts from pageable host memory to VRAM, which is the exact shape of `xfer`:

| mode | depth | GB/s | us/copy |
|---|---:|---:|---:|
| sync per copy on a side stream (what shipped) | 1 | 7.36 | 284.8 |
| K async on the default stream, one drain | 8 | 7.10 | 295.2 |
| K async on K side streams, drain at end | 8 | 6.02 | 348.2 |
| K async on default, staged through pinned | 8 | **9.54** | **219.8** |

**The shipped path was not slow because it kept one copy in flight.** Depth
changes nothing: a 2 MiB pageable H2D costs ~285 us whether it is the only one
outstanding or the eighth. What the engine's `xfer` number says is that the
copy is *not the only thing in that 2170 ms* — 2170 ms for 8.3 GiB is 3.8 GB/s
against a 7.4 GB/s ceiling, and the rest is the **host wait** that
`paged_publish_` makes on every single expert.

### 10.4 Two changes, both measured, both A/B'd

**Router barrier (`KRK_ROUTER_SYNC`).** `moe_ffn` issued an explicit
`hipDeviceSynchronize` before downloading the 256-wide router row, 40 times per
token. The blocking `hipMemcpy` that follows already orders itself behind every
kernel on the default stream, so the barrier was redundant. `probe_sync` prices
a host round trip on this machine at **~85 us of fixed cost** — launch, kernel,
`hipDeviceSynchronize`, return — which is why 40 of them cost 3.5 ms/token and
not 90. The barrier is gone; `KRK_ROUTER_SYNC=1` puts it back.

| arm | ms/tok (median of 4) | tok/s |
|---|---:|---:|
| `KRK_ROUTER_SYNC=1` | 100.30 | 9.95 |
| `KRK_ROUTER_SYNC=0` | **96.75** | **10.35** |

**Promotion ordering (`KRK_PROMOTE_WAIT`).** `upload_paged_batch` stages through
a pinned ring, issues the copy on a non-blocking side stream, and then
**host-waits** for it. The host therefore alternates "read the next expert from
the file" and "wait for this expert's DMA", never doing both. Ordering the copy
onto the default stream instead — `hipStreamWaitEvent(0, ev)` on the event
already recorded at issue — keeps correctness by stream ordering (the consumer
GEMM is launched on the default stream afterwards and cannot start early) and
lets the host run ahead into the next file read.

| arm | ms/tok | tok/s | `xfer` host ms |
|---|---:|---:|---:|
| `KRK_PROMOTE_WAIT=host` (old) | 100.00 | 10.00 | 2170-2719 |
| `KRK_PROMOTE_WAIT=event` (new) | **90.20** | **11.10** | **443-560** |

This directly contradicts the comment the host wait shipped on ("an event the
default stream waits on did not make the copy visible here"), so it was
re-tested rather than assumed: **8+8 interleaved runs produce one text, and it
is the same text the host-wait arm produces**
(`docs/traces/laguna-determinism-promote-wait.txt`). Two claims cannot both be
true; on this runtime the event wait is correct and the host wait was costing
1.7 s of a 6.3 s run.

### 10.5 Both together, against the previous behaviour

4 interleaved repetitions, arms alternating, `KRK_ROUTER_SYNC=1
KRK_PROMOTE_WAIT=host` restores everything that changed:

| arm | ms/tok (median) | tok/s | distinct outputs |
|---|---:|---:|---:|
| previous behaviour | 110.70 | 9.00 | 1 (`2f7a7f77`) |
| new default | **96.25** | **10.40** | 1 (`2f7a7f77`) |

**-13.0% per token, +15.6% tok/s, 4/4 repetitions, bit-identical text.**

### 10.6 The two memory budgets are now the user's to set

The device policy in section 10.5's commit is 6 GiB of VRAM to begin with, +2 GiB
at a time while the card has the headroom and the model needs it, never above 12
GiB, stated at run time and overridable with `--vram-cap-mb`. The host side got
the same treatment, because it had the same defect: WARM was sized at "half of
whatever is free after an 8 GiB reserve", which on a 96 GiB box aimed at 45 GiB
of tier. It is now a quarter of *installed* RAM (available memory is a function
of how long the machine has been up, not of this program), bounded by half of
what is actually free, bounded by the corpus, overridable with `--host-ram-mb`,
and an explicit `--expert-warm-mb` is clamped to the budget rather than silently
exceeding it.

Measured on laguna-xs2: default 24.0 GiB budget -> 18102 MiB of WARM;
`--host-ram-mb 8192` -> 8192 MiB; `--expert-warm-mb 40000 --host-ram-mb 8192`
-> a warning and 8192 MiB. On the device side the same run now reports
10.73 GiB in use with 5.20 GiB free, against 15.04 GiB and 0.88 GiB before.

### 10.7 The file read is not a cache-miss problem, so it is not a prefetch problem

Three runs back to back, same binary, same prompt:

| run | `read` | `alloc` | `xfer` | ms/tok |
|---|---:|---:|---:|---:|
| 1 | 1875.3 ms | 285.2 ms | 435.9 ms | 59.6 |
| 2 | 1906.7 ms | 301.2 ms | 438.4 ms | 60.3 |
| 3 | 1902.9 ms | 290.7 ms | 439.7 ms | 60.5 |

The second and third runs read exactly the same bytes out of a file the first
run had already pulled in, and the read time does not move. So this is not a
cold-cache cost that a warmer cache or a bigger read would remove: it is the
device's rate for scattered 0.7 MiB slices, at about 5 GB/s inside the engine
and 19 GB/s for the same pattern out of `probe_read.py` when the pages are hot.
`probe_read.py`'s scattered mode says the *pattern* is not the problem either --
one thread and eight threads are within 6% of each other -- so the lever is
overlap, not parallelism.

That is also what Edge0's prerouter is for, and why the next item below is the
one that matters.

### 10.8 What is left, and the ceiling

| stage | ms | note |
|---|---:|---|
| `read` | ~1900 | 8.3 GiB of scattered slices out of the file; the device's rate, not a cache miss |
| `alloc` | ~290 | was ~1150; the arena took it out |
| `xfer` | ~440 | 8.3 GiB pageable to VRAM, overlapped behind the reads |
| dense | ~1000 | 8.5 ms/token of real compute |

The file read is now the largest single item and it is **at the device limit**:
`probe_read.py` shows 32 threads read no faster than one. 347 MiB/token at
4 GB/s is ~87 ms/token of unavoidable I/O for a model whose expert working set
(8.3 GiB of first-touch weights) does not fit in the 8.4 GiB of VRAM the cache
has. Kraken at 96 ms/token is within ~10% of that floor, which is also why the
reference implementation is not 3x faster: it is streaming the same cold bytes.

That reframes the remaining work:

1. **DFlash speculative heads** — the only item that amortises the file read
   itself rather than the work around it. `kraken-inspect` on the shipped head:
   six dense blocks (no experts), 330 MiB of weights that all stay resident,
   `block_size = 16` so up to 16 candidates per pass, `decoder_arch = laguna`
   and the same embedding and head geometry as the target, and
   `target_layers = 6` so verifying a candidate costs six layers of the target.
   On a workload where one position costs 347 MiB of compulsory first-touch
   bytes, accepting four tokens per pass pays that read once for four
   positions. `docs/REFERENCES.md` has the full field list and
   `docs/traces/dflash-head-inspect.txt` the raw output.
2. **Route-ahead prefetch**, the untrained version of Edge0's prerouter: start
   the file read for a layer's likely expert set from the previous token's
   routing before that layer's GEMMs are issued. The routing traces this repo
   already collects would say first whether it predicts anything; the reads are
   ~35 ms of a 60 ms token and the compute available to hide them is 8.5 ms, so
   this is worth ~14% and no more.
3. **Multi-expert GEMV**, 960 -> 120 launches, and the router plan uploads with
   it: 640 tiny host-to-device copies per token go to 80 if the plan is built
   once per layer and the per-expert views are pointer offsets into it.
4. **Fusing the 40 router top-k passes.** Worth naming honestly, because the
   request is a good one and the answer is a dependency rather than a budget:
   layer L+1's router reads layer L's output, so 40 routers cannot be computed
   in one pass for the same token. What *can* be fused is the plan: one kernel
   per layer producing the k indices and weights, one download instead of a
   host softmax and top-k over a 256-wide row, and one plan upload instead of
   two per expert. The round trips themselves stay, because the host has to
   know which experts to page in. `probe_sync` prices what removing them would
   be worth if it were possible: ~85 us a trip, 40 a token, ~5% of 60 ms.

### 10.9 Reproducing section 10

```bash
# the four-way split, one run
./build-hip/kraken.exe -m G:/More-models/laguna-xs2-Q4_K_M.gguf \
  --prompt "what is the capital of france?" --max-tokens 24 --temp 0 --greedy --chat

# the A/Bs (arms alternate run by run; both always kill kraken)
bash scripts/ab_perf.sh G:/More-models/laguna-xs2-Q4_K_M.gguf 4 \
  "KRK_ROUTER_SYNC=1 KRK_PROMOTE_WAIT=host :: " "X=0 :: "
bash scripts/determinism.sh G:/More-models/laguna-xs2-Q4_K_M.gguf 8 \
  "KRK_PROMOTE_WAIT=host" "KRK_PROMOTE_WAIT=event"

# the probes
./build-hip/probe_sync.exe 300
./build-hip/probe_promote.exe 4096
python3 tools/probe_read.py G:/More-models/laguna-xs2-Q4_K_M.gguf 4 16
```

Traces: `laguna-perf-ab-router-and-promote.txt`,
`laguna-determinism-promote-wait.txt`, `laguna-expert-path-split.txt`,
`probe-sync-host-roundtrip.txt`, `probe-promote-dma-depths.txt`,
`probe-read-file-bandwidth.txt`.

## 11. The 78 GiB REAP-288 run: a WARM tier shorter than the routed set is churn

Measured 2026-10-22 on `Qwen3.8-Flash-Next-REAP-288-Q4_K_M` (1224 tensors,
78.04 GiB payload, 288 experts top-10, 48 layers, 42.2 GiB routed set),
RX 9070 XT (gfx1201, 15.9 GiB), driver 71626331, Release. Workload throughout:
prompt `def add(a, b):`, `--greedy --max-tokens 32 --ctx 512`, 15/31 timed
steps, identical stdout in every arm.

### Where the 5.8 tok/s went

The step was **175.8 ms**, and the expert tier's own rows accounted for it:

| row | per step | what it is |
|---|---:|---|
| read | 62 ms | 190.97 MiB/token read from the 78 GiB **file** (COLD misses) |
| alloc | 27 ms | slab allocation churn (1876 trims) |
| xfer | 39 ms | WARM->VRAM promotions at 11.0 GB/s (278.6 us/expert) |
| compute + launches | ~48 ms | everything else |

The cause was the tier's SIZE, not the kernels: the auto policy took half of the
free room (32.5 GiB) for a 42.2 GiB routed set, so 100% of the tier was in use,
11.1% of picks missed to disk, and 41.7% crossed PCIe every token.

### The change

`Engine::init`'s auto WARM sizing and `pinned_warm_prefix_bytes`:

* when the **routed set fits in the room**, size WARM to the corpus (bounded by
  the host budget) instead of half the room -- a tier shorter than the corpus
  misses on the same experts forever, because each step's promotions evict what
  the next step wants;
* pin the **whole** tier up to half of installed RAM. The pinned prefix is a
  prefix in rank order, so every expert past its end still pays the pageable
  copy; a partial pin buys a partial speedup, and the measurement is monotone.

### Result (2 interleaved order-balanced pairs, old policy forced with flags)

| arm | step | decode | COLD misses | file bytes/token |
|---|---:|---:|---:|---:|
| old (WARM 32571 MiB, pin 16384 MiB) | 157.0 / 153.6 ms | 6.4 / 6.5 tok/s | 1299 / 1302 | 128 MiB |
| new (auto: WARM 42440 MiB, fully pinned) | 100.3 / 106.9 ms | 10.0 / 9.4 tok/s | 124 / 190 | 12-19 MiB |

**+50% decode (6.45 -> 9.7 tok/s median), -33% step time**, identical stdout.
The pin's own contribution, isolated at a fixed 42.4 GiB tier: 16 GiB pinned
139.4 ms/step, 24 GiB 121.8, 42.4 GiB 103.1.

### What is left

With the tier fixed, the step is **host-routing bound**, and the host table says
so (`KRK_TIME_HOST=1`, 15 steps): `moe.route` is 79.6 of 102.3 ms/step -- 31.8 ms
in 48 blocking router readbacks (0.66 ms each for 288 floats, nearly all of it
waiting on the preceding kernels) and 47.4 ms in the host top-k + plan build.
The device is idle through most of it. `KRK_DEV_PLAN=1` is the intended answer
and is gated off for this file by `ne <= 256` (it has 288), and it still
downloads the ids to verify them, so lifting that bound alone is not a speedup.
The real fix is a device-built plan with no host round trip, which needs the
per-expert dispatch -- and that needs the whole routed set device-resident,
which 42.2 GiB against a 7.2 GiB budget is not.

## 12. The RAM tier is readable in place, and that is what the tier was for

Measured 2026-10-22 on the same REAP-288 file, same prompt and greedy rule, with
the tiers forced to the shape a *small* machine has (`--expert-cache-mb 8000
--expert-warm-mb 20000`: ~7 GiB of hot experts on the card, 20 GiB of RAM, the
78 GiB mapping for the rest).

### The question the three tiers could not answer before

The design that makes these models reachable on a 6/8/12 GiB card and 16/24/32/
48 GiB of RAM is: **the hot set device-resident, some RAM for the next set, the
file for everything else.** Until this pass the RAM tier could not be *used*
without being copied into VRAM first, so on a card smaller than the routed set
every layer re-copied ~3.1 MiB per routed expert to read it once, and the copy
evicted the expert the next layer wanted. That is the churn §11 measured, and it
is why the only fast configuration was "hold nearly the whole routed set in
VRAM" -- 42.2 GiB, which no consumer card has.

So the question was one line: **can a kernel read the pinned tier where it
lives?** `tools/probe_hostread.hip` answers it.

| what | result |
|---|---|
| kernel read of a 4 MiB `hipHostMalloc` buffer | exact (100.000000% of the expected sum) |
| 256 MiB contiguous read, pinned | **24.3 GiB/s** |
| the same kernel over device memory | 485.2 GiB/s (5% left) |
| `unifiedAddressing` | **0** -- the flag says no, the pool is mapped anyway |

The measured promotion copy is 11-22.9 GiB/s, so a pinned read is in the same
class as the copy it replaces, and it replaces a copy per expert with a *read of
the bytes that are needed*.

### The change

* `Backend::host_load_supported()` -- false by default, true on HIP, and only
  after the numbers above. A backend that has not measured it keeps promoting.
* `ExpertCache::acquire_inplace()` returns the WARM matrices with no device copy
  when the slot is pinned host memory, is not already resident, and has not
  EARNED residency (`kPinThreshold` touches). An expert the tier has ranked hot
  still goes to VRAM, where a read is 20x faster.
* `Engine::moe_ffn` tries it first and falls through to `acquire()` for every
  other case, and skips the whole-routed-set promote/stage batches when the
  weights are not going to be copied at all. `KRK_EXPERT_INPLACE=0` is the arm.

### Result (2 arms, same command line, stdout identical)

| | promote | in place |
|---|---:|---:|
| step | 147.0 ms | 149.3 ms |
| decode | 6.8 tok/s | 6.7 tok/s |
| WARM -> VRAM copied | 18203 MiB | **58 MiB** |
| VRAM evictions | 5839 | **23** |
| alloc | 669 ms | **1.3 ms** |
| promote | 775 ms | **1.9 ms** |
| expert-path, decode | 0.79 ms/layer | **0.19 ms/layer** |

The text is bit-identical (`da9ba027a3454bba9edfefdb6c049632` both arms), the
suite is 3824/3824, and coherence passes on dense, recurrent and MoE.

**The step is a wash on THIS box, and that is the honest reading of it.** It is
a wash because the card is idle nowhere: the bytes moved drop 300x, and the time
they cost does not, because the copy was never the constraint on a machine whose
page cache already held the file. What changes is the *VRAM footprint*: 7 GiB of
experts now serve the same text where the promoting path needed the whole routed
set to stop thrashing. That is the whole difference between a 12 GiB card and a
48 GiB one running this model.

### Where the 149 ms now goes, and what it means for 20-30 tok/s

| row | per step | what it is |
|---|---:|---|
| `moe.route.fetch` | 63 ms | the GPU's own backlog when the router row is asked for |
| `moe.route.rest` | 60 ms | 38 ms of **blocking** file reads (274 MiB/token) + 22 ms host selection and plan |
| `moe.experts` | 9 ms | 480 expert visits reading 3.1 MiB each |
| everything else | 17 ms | attention, recurrent layers, norms, head |

Two consequences, both arithmetic rather than opinion:

* **The host still blocks on file reads inside the layer.** 274 MiB/token out of
  the mapping at 7.2 GiB/s, serial with the GPU. Overlapping that one layer
  ahead is worth ~25% here and much more on a box where those reads reach the
  disk instead of the page cache.
* **REAP-288 has a floor near 16 tok/s even with a perfect tier.** One token
  visits 480 experts and needs ~1.5 GiB of weights; at the measured 24.3 GiB/s
  in-place rate that is 62 ms/token before any compute or disk. 20-30 tok/s
  belongs to an A3B-class model (8 of ~1.5 MiB per layer, ~250 MiB/token), which
  is what DeepSeek-V4-Flash and Qwen-3.8-Flash route -- the same three tiers,
  with a routed set small enough for the RAM tier to hold.

### The small-card case is where it pays, and it is the case that matters

The same pair at `--expert-cache-mb 4000` (a 6-8 GiB card's expert budget), 23
timed steps, stdout identical again (`c0627e2ac768c031a6c59035aabffacb` both
arms):

| | promote | in place |
|---|---:|---:|
| step | 167.8 ms | **157.7 ms** (+6.0%) |
| decode | 6.0 tok/s | 6.3 tok/s |
| WARM -> VRAM copied | 20839 MiB | **0 MiB** |
| VRAM evictions | 6665 | **4** |
| alloc | 795.5 ms | **0.0 ms** |

Shrinking the card makes the promoting path worse (167.8 against 147.0 ms at
8 GiB) because more of the routed set misses, and leaves the in-place path
essentially unchanged (157.7 against 149.3) because the bytes it reads are the
same bytes either way. That is the shape a three-tier design is supposed to
have: the VRAM budget sets what is FAST, not whether the model runs.

### 12b. The reads move off the host thread: 6.5 -> 7.8 tok/s, and the two traps on the way

With the RAM tier readable in place, the blocking item left was the reads
themselves: 274 MiB/token out of the mapping, on the host thread, inside the
layer, serial with that layer's GPU work. `ExpertCache::prefetch_warm_next()`
moves them to a worker thread, one job per layer, and `take_prefetched()`
installs the bytes as the expert's WARM copy when the loop reaches it.
`KRK_WARM_AHEAD=0` is the arm. 24 greedy tokens, 4 interleaved order-balanced
runs, stdout identical in every pair (`c0627e2a...`):

| expert budget | blocking reads | read-ahead |
|---|---:|---:|
| 8000 MiB | 152.4 / 153.2 ms | **129.6 / 128.3 ms** (7.7-7.8 tok/s) |
| 4000 MiB | 157.7 ms | **136.2 ms** (7.3 tok/s) |

**The first version issued zero reads, and the reason is the useful part.** It
was fed the *next* layer's ids from the previous token's routing, which is the
tier's own prefetch hint -- and those experts are, by definition, already in
WARM, because the previous token just used them. Debug output: `hint=10 ids=0`
on every layer. A predictor that is right about the past predicts nothing about
what is *missing*. The ids that matter are exact and already on the host: the
current layer's routing, after selection, minus what the tier holds.

**The second version was 11% SLOWER, and that is the other half of the same
lesson.** `Engine::moe_ffn` already reads the whole routed set through
`prefetch_layer()` before the expert loop. The worker was reading the same
bytes again, and `take_prefetched()` then threw its result away -- correctly,
since the synchronous pass had already filled WARM. Two copies of every byte and
none of them used. The synchronous call is now skipped when the worker is
reading, and the same run is 18% faster.

### What the step looks like now (24 tokens, 8 GiB expert budget, 20 GiB RAM tier)

| row | per step | was |
|---|---:|---:|
| `moe.route.fetch` | 31.6 ms | 63 ms -- the GPU backlog is halved because the host stops blocking |
| `moe.experts` | 76 ms | 9 ms -- now includes waiting for in-flight reads and the PCIe weight reads |
| everything else | 22 ms | 17 ms |
| **step** | **130.0 ms** | 149.5 ms |

The expert row is within 20% of the arithmetic floor for this file: one token
visits 480 experts of ~3.1 MiB = ~1.5 GiB, which is 62 ms at the measured
24.3 GiB/s in-place rate. REAP-288 is now a weight-bytes problem, not a
bookkeeping one -- which is what section 12 said the ceiling would look like.

### 12c. The A3B case: 29-32 tok/s at a 4-6 GiB expert budget, and the policy the REAP run got wrong

The three tiers are for models whose routed set is small enough for the RAM tier
to hold -- which is the A3B class the goal names. Measured on
`Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K` (10.49 GiB, 128 experts top-8, 48
layers, routed set ~9.6 GiB), 48 greedy tokens, ctx 512, `--ctx 512`, same prompt
and text in every arm:

| expert budget | RAM tier | decode | step | COLD misses |
|---|---:|---:|---:|---:|
| 4000 MiB | 8000 MiB | 29.2-29.9 tok/s | 34.2 ms | 1.1% |
| 4000 MiB | 12000 MiB | **31.2 tok/s** | 32.1 ms | **0.0%** |
| 6000 MiB | 8000 MiB | **31.6 tok/s** | 31.7 ms | 1.2% |

That is the 20-30 tok/s target met on a card budget a 6-8 GiB card has, with the
whole routed set resident in 12 GiB of RAM and **no file reads at all** once the
tiers are warm. The promoting path reaches the same band here (29.9-30.1 tok/s),
which is the honest reading: when the routed set FITS, a promotion is a
one-time cost and VRAM is 20x faster afterwards, so there is little to save.

**The first version of the in-place rule cost this model 20%**, and the fix is a
policy, not a kernel. `acquire_inplace` first refused only for experts that had
earned a pin (24 touches), so on this file 45.7% of acquires read 24.3 GiB/s
host memory where VRAM had the expert already: **42.9 ms/step against 33.7**.
The signal that matters is REUSE, and the tier already counts it -- an expert
asked for twice goes to VRAM, `count >= 2`. One in-place read pays for itself
immediately (it is the read the promotion would have copied); the second use is
where the card wins by 20x. With that rule: A3B 34.2 ms/step (-2% against
promote, and 4x fewer COLD misses), REAP-288 127.9-130.2 ms (+7.7%).

| file | promote | three tiers |
|---|---:|---:|
| Qwen3-30B-A3B, 4 GiB / 8 GiB | 29.9-30.1 tok/s | 29.2-29.9 tok/s |
| REAP-288, 8 GiB / 20 GiB | 138.5-139.3 ms | **127.9-130.2 ms** |

**The caveat this box cannot remove:** its 96 GiB of RAM means a COLD miss is a
page-cache read (~7 GiB/s). On a 16-32 GiB machine those reads reach the disk,
which is slower by an order of magnitude -- and that is why the 12 GiB tier with
0.0% COLD is the interesting row: it is the configuration where the disk is not
in the loop at all.

### 12d. Where the disk starts to hurt: the RAM-tier sweep at a 16 GiB machine's shape

`Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K` (10.49 GiB, routed set ~9.6 GiB), 48
greedy tokens, ctx 512, a 4000 MiB expert budget (a 6-8 GiB card) and the RAM
tier swept from 3000 to 12000 MiB:

| RAM tier | decode | step | COLD misses | file bytes/token |
|---|---:|---:|---:|---:|
| 3000 MiB | 23.5 tok/s | 42.6 ms | 14.2% | 98.9 MiB |
| 4000 MiB | 23.7 tok/s | 42.3 ms | 13.0% | 90.2 MiB |
| 6000 MiB | 27.2 tok/s | 36.7 ms | 4.4% | 30.6 MiB |
| 8000 MiB | 29.4 tok/s | 34.0 ms | 1.1% | 7.9 MiB |
| 12000 MiB | **31.7 tok/s** | **31.6 ms** | **0.0%** | **0 MiB** |

**This box cannot show the disk**, and saying so is part of the result: its 96
GiB of RAM holds most of the file, and a probe of 40 random 1.5 MiB reads out of
the 78 GiB REAP file returned a **median 1809 MiB/s** (min 954) -- page cache,
not storage. The disk rates below are the NVMe numbers recorded in
`AGENTS.md` (593 MiB/s at queue depth 1, 1446 MiB/s with four readers), applied
to the bytes this sweep actually read.

The read-ahead worker changes where the knee is, because it hides the read
inside the layer's own compute (~0.66 ms per layer at 48 layers and 31.6 ms).
The excess is therefore `max(0, read_ms - 0.66)` per layer, not the whole read:

| RAM tier | MiB/layer | at 1446 MiB/s | at 593 MiB/s | projected step | projected decode |
|---|---:|---:|---:|---:|---:|
| 8000 | 0.165 | 0.11 ms | 0.28 ms | **34 ms (hidden)** | **29 tok/s** |
| 6000 | 0.64 | 0.44 ms | 1.08 ms | 37-57 ms | 17-27 tok/s |
| 4000 | 1.88 | 1.30 ms | 3.17 ms | 73-162 ms | 6-14 tok/s |
| 3000 | 2.06 | 1.42 ms | 3.47 ms | 79-178 ms | 6-13 tok/s |

**The knee is between 6000 and 8000 MiB of RAM tier**, and it is a cliff, not a
slope: COLD misses go from 1.1% to 4.4% to 13% as the tier shrinks, i.e. from
0.16 MiB to 0.64 to 1.9 MiB per layer, and the worker stops being able to hide
what it reads. A 16 GiB machine can afford about 6-8 GiB for this tier, so it
sits on the good side of the knee at 8 GiB (29 tok/s, disk irrelevant) and on
the edge at 6 GiB (17-27 tok/s depending on the drive). Below that, the model
runs but it is streaming rather than decoding.

### 12e. The default tier now lands on the good side of that knee

The sweep is what changed the default. `krk::warm_tier_bytes`
(`include/krk/vram_guard.hpp`) sizes WARM as the expert corpus capped by
`krk::host_budget_bytes` -- 60% of installed RAM (the 80% ceiling minus the 20%
reserve), or free RAM minus that reserve when that is smaller, failing closed on
an unmeasurable host:

| installed RAM | auto WARM room | on Qwen3-30B-A3B (9.6 GiB routed set) |
|---|---:|---|
| 16 GiB | 9.6 GiB | the whole set -- the 8000+ MiB row, disk irrelevant |
| 24 GiB | 14.4 GiB | the whole set |
| 32 GiB | 19.2 GiB | the whole set |
| 96 GiB | 57.6 GiB | corpus-capped (10008 MiB here) |

The policy this replaced subtracted a working-set headroom (`installed/5`), a
pin allowance (`installed/16`, capped at 6 GiB) and a 3 GiB floor on top of the
reserve, then halved the result whenever the corpus was bigger than the room.
Since the pin is *part of* the tier, the pin allowance charged the same bytes
twice, and the halving is what turned a 16 GiB machine's 8.8 GiB into 4.4 GiB --
the 3000-4000 MiB row, 23.5-23.7 tok/s, where this file has 29-31 tok/s
available. There is no measurement behind a smaller default: the only case for
one is a box doing other work at the same time, which is what an explicit
`--expert-warm-mb` is for now. Text is identical at every tier (`7a92969e...`
across all five arms), so the tier is a speed knob and never a correctness one.

Measured as the A/B the change is: same file, `--expert-cache-mb 4000`, 48
greedy tokens, 3 interleaved order-balanced pairs, stdout byte-identical
(`6169084d...` both arms):

| arm | WARM tier | decode | ms/step |
|---|---:|---:|---:|
| old default equivalent (`--expert-warm-mb 4400`) | 4400 MiB | 29.2-29.3 tok/s | 34.1-34.2 |
| new default at a 16 GiB class (`--ram-tier 16`) | 9830 MiB | **33.2-34.1 tok/s** | **29.3-30.1** |

The old arm's first pair read 11.9 tok/s and is excluded: that is the file's
first touch on a cold page cache, not the tier. **+16.5%** on the pairs that
did not pay it, and it is the same bytes moving through the same kernels -- only
the tier's capacity differs.

### 12f. The defaults against explicit caps, and what a busy box does

Qwen3-30B-A3B, 4000 MiB device budget, 48 greedy tokens, ctx 512: three runs per
arm, interleaved with the starting arm rotated, stdout identical in all twelve
(`6169084d...`):

| WARM tier | decode | ms/step | vs the old default size |
|---|---:|---:|---:|
| auto 10008 (the corpus) | **36.0-36.1 tok/s** | 27.7-27.8 | +19.5% |
| 8000 | 34.9-35.3 | 28.4-28.7 | +16.2% |
| 6000 | 33.7-33.9 | 29.5-29.7 | +11.6% |
| 4400 (what the old arithmetic picked) | 30.2-30.5 | 32.8-33.1 | -- |

REAP-288, 16 greedy tokens, same rotation, every arm's text identical
(`faa1191e...`):

| WARM tier | decode | ms/step |
|---|---:|---:|
| auto 43200 (the corpus) | **10.5-10.8 tok/s** | 92.2-95.6 |
| 20000 | 5.7-7.8 | 128.3-175.9 |
| 12000 | 6.8-7.1 | 141.1-146.2 |

The middle row is not a typo: 20000 loses to 12000 on this file. Both are far
under the 42.2 GiB set, so both miss forever, and the larger one pays a bigger
pinned fill before it starts paying back -- over 16 tokens it never does. That
is why the auto answer is the corpus and not some fraction of it: a tier that
sits between the set and the room is the worst of both.

**A busy box shrinks the auto tier, and that is the case the warning exists
for.** With 50 GiB held by another process (33.5 GiB free of 95.9), REAP-288's
auto tier planned **14650 MiB instead of 43200** -- the live sample minus the
19.2 GiB reserve, not the installed-RAM plan -- and said so once:

    [warn] ram policy: the auto WARM tier is 14650 MiB for a 43200 MiB routed
    set (34%), under the measured knee: ... projected 7.86 MiB of file reads per
    layer (377 MiB per token at 48 layers) ...

and decoded at 2.5 tok/s over 8 cold tokens. So "does the auto tier ever
under-size on a busy box" has the answer yes, by design: it is the largest tier
the host can spare at that moment, it never plans past free-minus-reserve, and
now it names the tier, the coverage and the projected per-layer read whenever
that lands under the knee. On an idle 96 GiB box A3B's auto tier stays
corpus-capped at 10008 MiB while free RAM is above ~28.8 GiB.

### 12g. Where REAP-288's decode time goes, and why the router is not the lever

The routing round trip is the only blocking point in a layer on a model whose
experts do not fit in VRAM, so it is the natural suspect. Measured on this build
(in-place WARM reads + read-ahead on), REAP-288, 48 layers, ctx 512, expert
budget 9178 MiB, WARM 43200 MiB, 16-48 greedy tokens, `KRK_TIME_HOST=1`:

| slice | ms/step |
|---|---:|
| `moe.experts` (of which read 46, alloc 25, promote 19) | 117.8 |
| `moe.route.fetch` (one 576-byte readback x 48 layers) | 47.8 |
| attention / norms / residual / hyper-connections | ~30 |
| `moe.route.gemm` | 0.2 |
| **host top-k + plan** | **1.8** |
| step | 171-199 |

The 47.8 ms looks like the biggest host item and it is **not removable by making
the readback cheaper**. Three measurements say so:

| probe | result | what it rules out |
|---|---|---|
| `KRK_ROUTER_2X=1`, the same copy again with the queue retired | **95 us**/layer | the copy itself (0.9 of the 1.0 ms is not transfer) |
| pinned destination + `hipMemcpyAsync` + spin on the recorded event, 3 arms x 4 rotated pairs, text identical | `download_f32` host 3264 vs 2997 ms over a run; step 145.9/129.8/141.0, 139.4/141.0/139.1, 112.1/114.7/114.8, 113.6/113.3/121.6 ms for pageable-blocking / pinned-blocking / pinned-spin | a driver or blocked-wait scheduler cost |
| the same readback on Qwen3-30B-A3B, same 48 layers | **0.477 ms**/layer against REAP's **1.031** | a fixed per-call cost |

So the host block **is** the layer's own pre-MoE GPU work finishing: the drain is
the dependency, and a faster copy cannot shorten it. Deleting it requires taking
the host out of the routing loop entirely -- device selection *and* device
dispatch -- which needs the routed set resident. On this file that is 42.2 GiB
against a 9.2 GiB budget, so the routing path stays as it is.

What is left on this configuration is the expert path, and each piece is
accounted for: **read 46 ms/step** is the in-place host read (24.3 GiB/s for
~1.1 GiB/token -- physics on this box), **alloc 25 ms/step** is pool growth
(`alloc 4302 reuse 0`: a cold cache admitting 4.8 GiB for the first time, which
amortizes over a longer run), and **promote 19 ms/step** is the 9.2 GiB budget
working against a 42.2 GiB set.

### 12h. The model file's page cache does not shrink the next run's tier

The auto WARM tier is clamped by live available RAM, which raised the question of
whether a mapping that has filled the page cache suppresses its own next plan.
It does not, and this box settles it: reading **32 GiB of the 78 GiB REAP file**
into the cache leaves Windows `ullAvailPhys` at **82.9 GiB against 83.0 GiB
before** -- standby (file cache) is part of "available" -- and Linux's
`MemAvailable` includes reclaimable page cache by construction. The plan never
subtracts the mapping either (`mapped_share` is an info line only), so the
43200 -> 14650 MiB shrink measured with 50 GiB held in another process came from
genuine committed memory, exactly as 12f says.

### 12i. The expert pool's trim churn, and the fix that was rejected

At a saturated device budget the pool answers a size-class miss by hipFree()ing
free cells of the OTHER class and hipMalloc()ing one of the wanted class. An
expert is two classes here (gate/up, and down), so the two sides kept destroying
each other's reserve. Measured on REAP-288 at `--expert-cache-mb 4402`, 64 greedy
tokens: **6528-7655 allocations and 1136-2341 trims**, i.e. ~100 hipMalloc+hipFree
pairs a step at ~190 us each; the `alloc` row reads 1085.8 ms for 64 tokens and
1988 ms for 128 (15.7 ms/step at steady state, 128 tokens, `--expert-cache-mb
8000`).

Note what this is NOT: `reuse 0` in a 16-token run is correct, not a defect --
with 0 device evictions nothing was ever freed (`release 0 mark(s) for 0
slice(s)`), so there was nothing to reuse. In a 128-token run reuse is 25243 and
works. The trim is the churn.

The principled fix is to refuse instead of trimming: return nullptr, and let the
cache's `force_reclaim()` pick a victim holding a slice of the WANTED class, so
one eviction releases a gate, an up and a down cell -- exactly the mix the next
admission asks for. Built and measured (3 arms x 4 rotated pairs):

| arm | allocations | trims | `alloc` ms (64 tok) | step (64 tok) | stdout |
|---|---:|---:|---:|---:|---|
| trim (`KRK_POOL_TRIM=1`) | 6601 | 2380 | 1085.8 | 103.9 median | 4/4 identical |
| refuse + evict | 4219 | 4 | **506.9** | 103.6 median | **3/4 a different text** |

So it does what it claims structurally (allocations -36%, trims 2380 -> 4, the
alloc row halved) and it is **rejected**: the step time does not move (the host
alloc time is overlapped), and the text loses determinism -- with heavy eviction
some runs produce a different sentence. That second fact is the interesting one:
it says the eviction path can retire a slot whose cells are still being written,
which the trim path has been masking by never evicting this often. It is a bug to
fix on its own, not a performance lever, and the code was reverted. **It was found
and fixed the same day -- see 12k -- and the fix is what makes the divergence
readable: it was one write path missing one guard.**

### 12j. How much of the "in-place read" a bigger VRAM budget buys

The in-place WARM read looked like the biggest expert-path cost on a short run
(46 ms/step at 16 tokens). In steady state it is **0.87 ms/step** -- that 46 was
the cold-fill window, where the tier is still being populated and the read-ahead
cannot hide anything. Swept at 128 greedy tokens, WARM auto (43200 MiB), two runs
per arm, text identical:

| VRAM budget | HOT | WARM | in place | promotions (ms) | step (ms) |
|---|---:|---:|---:|---:|---:|
| 3000 MiB | 37.2-37.3% | 38.9-39.2% | 23.5-23.8% | 2988-3561 | 120.7-143.6 |
| 4402 MiB | 45.4-46.2% | 30.9-31.4% | 22.3-23.6% | 2731-2779 | 125.4-126.5 |
| 6000 MiB | 55.0% | 22.6-22.8% | 22.2-22.3% | 2040-2067 | 111.5-115.1 |
| 8000 MiB | 53.4-53.5% | 24.2% | 22.2-22.3% | 2133-2351 | 110.0-117.5 |

2.7x the resident budget buys **-14.6%** on the step (median 136.7 -> 116.7 ms),
and almost none of it is the in-place share, which barely moves (23.8% -> 22.2%):
the extra residency converts WARM hits into HOT hits, which removes promotions
(2988-3561 -> 2040-2351 ms), and in-place stays reserved for experts that have not
earned residency. So the answer to "how much of the in-place read is avoidable by
keeping the top-N resident" is: **very little, because in-place is not what the
budget is fighting** -- promoting fewer experts is, and it is worth ~15% between a
3 GiB and an 8 GiB budget on a 42.2 GiB set. Steady state at 8000 MiB, 114.4
ms/step: routing 89.2 (fetch 37.1 -- the dependency drain of 12g -- and the expert
loop 50.2), attention and everything else ~25.

### 12k. The reuse race 12i found: one missing guard, and what it costs to close

**The hole.** A released pool cell goes back on the free list carrying the mark
its readers were released under, and `alloc_pooled` hands the next owner the same
mark. Every write path into a cell waits for it on the stream it writes with --
`upload_paged` (ring and direct branches), `upload_expert_staged`,
`upload_expert_prefetch`, `upload_i32`, `fill0`, `copy_act` -- **except
`upload_pinned_batch`**, the arm a promotion out of pinned WARM takes. It copies
on a private `hipStreamNonBlocking` stream and never waited, so the DMA could
land on bytes a queued expert GEMM was still reading. Not a crash: a plausible
number from the wrong matrix, and under heavy eviction a different sentence.
Fixed with one `pool_honor_(dst[i], pinned_stream_)`.

**The trap** (`kraken-bench --pool`, also in `--gate`): 16 copies of one 64 MiB
cell queued on the default stream, the cell released, handed back out, and a
different pattern DMA'd into it; every reader must still see its own bytes.
Without the guard it fails **3 of 3**, with it **3 of 3** pass. The trap's first
version -- 256 reads of a 4 MiB cell -- PASSED on the buggy code, because the host
enqueue outran the device work and the queue was empty when the DMA was posted.
Depth is the trap.

**What it costs**, REAP-288, `--ctx 512 --chunk 256`, 32 greedy tokens, guard on
against off interleaved, text identical in every pair:

| configuration | reuses (32 tok) | guard off (ms/step) | guard on (ms/step) | delta |
|---|---:|---|---|---|
| default auto budget (~8.4 GiB here) | 0 | 136.7-137.8 | 147.0-152.8 | no pending marks: order noise |
| `--expert-cache-mb 4000` | 3890-4064 | 141.9-148.8-150.5 (median 147.6) | 156.1-158.7-169.4 (median 159.4) | **+8.0%, 4 of 4 pairs** |
| 4000 MiB, refuse-instead-of-trim arm (12i) | ~55 000 | 146.4-147.1 | 219.1-220.7 | **+50%** |

The last row is why 12i's refusal is not a performance lever: at a saturated
budget the cell handed out IS the one the eviction has just released, so its
readers are the current layer's own GEMMs and the copy has to wait for them. The
overlap that arm was buying was reading garbage. Recovering it honestly means
promoting ahead of demand (12g, 12j), not dropping the guard.

**And 12i's divergence is this bug.** At 4000 MiB on that arm, with the guard
removed, 3 runs gave 2 identical continuations and one different one
(`16ab8aff...` against `b3eb4d07...`); with the guard, 4 of 4 matched. The rate is
not the finding -- the trap is -- but the shape matches: a corruption that only
flips an argmax sometimes.

### 12l. The promotion time is bytes, not a hit rate -- so there is nothing to "hit better"

REAP-288, `--expert-cache-mb 8000 --expert-warm-mb 20000 --ctx 512 --chunk 256`,
48 greedy tokens, one run (168.2 ms/step, 5.9 tok/s):

```
acquires 24109 | HOT 15848 (65.7%) | WARM hits 2333 (9.7%) | in place 3260 (13.5%) | COLD 2668 (11.1%)
```

2333 promotions over 48 tokens is **48.6 per token**, and an expert here is
~3.1 MiB, so the copy is ~152 MiB/token: 7.2 GiB in total, which this run's own
timer reports as `promote total 283.4 ms` over 47 steps -- **6.0 ms/step, ~25
GB/s**. The time is the bytes at the copy's rate; the per-copy overhead is not in
it. (12j's 17.4 ms/step is the same budget over a 128-token window, where the
steady-state promotion count per step is higher: ~118 per token against 48.6
here. Either way the number is `promotions x 3.1 MiB / copy rate`, and nothing at
the allocator or the scheduling layer can move it.)

And `host_hits_` counts only promotions issued from `acquire()` -- from the
demand for that expert, right then. So "how often does the WARM->VRAM copy land
on an expert that is about to be used" is **100% by construction**, and there is
no wasted-promotion population to remove. The only two levers left are:

- **hold more of the routed set** (more VRAM): measured, -14.6% on the step for
  2.7x the budget (12j);
- **route to what is resident** (`--router-resident`): fewer promotions by
  construction, and it changes the route, so the text changes with it. Two
  interleaved pairs at 0.05/0.10 were inside the run-to-run spread of this
  machine (109-232 ms/step on the same arm) -- not a measurement yet.

Speculation (`stage_next`, `KRK_MOE_LOOKAHEAD`, `moe_hint_`) cannot remove bytes:
it moves the same copy off the critical path, and on Tiel it measured neutral
(`docs/AUDIT-EXPERT-CACHE.md` section 7). That is exactly why the guard fix in
12k makes it worth retrying: a promotion now waits for the readers of the cell it
reuses, so a copy issued ahead of demand is worth more than it was.

### 12m. The promote-ahead retry: neutral, because the pass has no candidates

The reason to retry it after 12k is specific: a demand promotion now waits for
the readers of the cell it reuses, so the copy can no longer overlap the layer in
front of it, and issuing the copy a layer early should be worth something where
it was worth nothing before.

Two mechanisms already exist: `KRK_MOE_NEXT_PREFETCH=1` stages the previous
token's routing for the next MoE layer (`stage_ahead`), and `KRK_MOE_PREDICT=1`
stages an adaptive prediction. Measured on REAP-288, `--expert-cache-mb 4000
--expert-warm-mb 20000 --ctx 512 --chunk 256`, 32 greedy tokens, guard on, four
interleaved order-balanced pairs, text identical (`da9ba027...`) in all eight:

| arm | ms/step | HOT hits | WARM hits |
|---|---|---|---|
| baseline | 178.2 / 178.5 / 179.8 / 199.1 (median 179.2) | 46.9% | 2701 / 2724 / 2704 / 2667 |
| `KRK_MOE_NEXT_PREFETCH=1` | 177.1 / 178.6 / 179.0 / 180.2 (median 178.8) | 46.9% | 2663 / 2673 / 2691 / 2710 |

**No change, and the counters say why there is nothing to change.** The new
`[stats ] lookahead <n> staged, <m> rejected by the budget, <k> consumed` line
(the cache's own `lookahead_staged_` / `lookahead_rejected_` / `active_used_`,
printed only when non-zero) is **absent in every arm**: the pass did not stage
copies that were free, it found no candidate at all. Both filters explain it --
an entry already in VRAM is skipped (46.9% of acquires are HOT, so much of the
hinted set is already there) and one no longer in WARM is skipped too, and an
expert hinted a whole token after it was used (48 layers x 480 experts of traffic
through a 20000 MiB tier) is usually gone from WARM as well. So the 8% of 12k is
not a scheduling problem this pass can solve.

That is consistent with the measurement that keeps it off by default, which this
does not overturn: on REAP-320 the same pass measured **-9%** (114.4 against
124.7 ms/step), because a prediction miss adds a promotion the real routing does
not use and the traffic is bandwidth-bound anyway.

What could recover the 8% is not when the copy is issued but which cell it lands
in: `force_reclaim` takes the FIRST same-class slot in map order, so the victim
is often an expert of the current layer whose readers are still queued, which is
exactly the wait the guard imposes. Ranking that choice like `pick_victim`
(rarest, then oldest) would favour cells whose readers retired long ago. Untested.

### 12n. Ranking the eviction victim does not recover the guard's wait, and the reason is the mark

12m left one candidate standing: `force_reclaim` takes the FIRST same-class slot
the map yields, so the cells it frees are often an expert of the layer being
promoted, whose GEMMs were queued microseconds ago -- and the next writer waits
for them. Two rankings were tried and both are refuted, measured on REAP-288 at
`--expert-cache-mb 4000 --expert-warm-mb 20000 --ctx 512 --chunk 256`, 32 greedy
tokens, two interleaved rounds per arm, guard on, text identical
(`da9ba027...`) in every run:

| arm | ms/step | HOT hits |
|---|---|---|
| guard off, first match (the racy baseline) | 146.2 / 150.3 (round 1-2) and 144.3 / 142.9 (round 3-4) | 46.9-47.0% |
| guard on, first match | 179.6 / 182.1 and 180.2 / 181.7 | 46.9-47.0% |
| guard on, ranked like `pick_victim` (rarest, then oldest load) | 181.9 / 183.6 | 46.9-47.0% |
| guard on, ranked by true least-recently-read (a new per-slot stamp) | 180.5 / 183.1 | 46.9% |

**No ranking moves it.** The cost is real and it is not 8% in this configuration:
the guard is **+24%** (144-150 -> 180-183 ms/step), and neither candidate choice
nor read recency touches it.

**Why the victim cannot matter: the release mark is a stream POSITION, not a
dependency on a cell.** `release_batch_` records one `hipEvent` on the default
stream at the moment of the release, and its meaning is "everything queued before
this point has completed". That is what makes it correct -- the readers of the
released cells were queued before that point -- but it also makes it unbounded:
the event retires only when the default stream reaches the record, i.e. after
every kernel queued before the release, whoever it belonged to. In a decode step
the whole token is queued back-to-back without a sync, so a cell reused inside a
layer waits for that layer's own earlier experts and the layer before it,
regardless of which slot gave the cells up. The victim's age is uncorrelated with
the wait because the wait is global.

Both changes were reverted (an eviction policy that measures neutral is risk
without payoff), and the finding is left as a comment in `force_reclaim` so the
next attempt starts here rather than at the victim choice.

**What a real fix would have to change**: the mark. A per-slot "my readers are
done" event recorded at the END of that slot's own use -- one event record per
expert per use, ~480 per token on this file -- would let a reuse wait for the
victim's readers instead of the whole queue. That is a pool API change, and it is
the only direction these two measurements leave open.

### 12o. Per-slot reader-done marks: the guard's wait was the whole queue, and this gives back 13%

12n left exactly one direction: the mark. A release records an `hipEvent` as a
POSITION on the default stream, so a writer reusing those cells waits for every
kernel queued before that position. What a reuse actually needs is much less --
"the kernels that read *these* cells are done" -- and the owner knows that point
exactly, because it is the end of that slot's own use.

**What landed.** `Backend::pool_note_reader_done(cells, n)` is called by
`ExpertCache::note_read(layer, expert)` right after the expert's kernels are
enqueued, in the per-expert loop and after the one-launch dispatch (which reads
the same cells without the loop). It records ONE event for the slot's three cells
-- they are read by the same kernels, so the mark is shared, not per cell -- and
stores it keyed by cell in `pool_reader_done_`. The next release of that cell
ADOPTS it instead of recording a position at eviction time; a burst whose cells
all carry one records nothing at all. The invariant that keeps it safe: a stamp
describes the LAST use, so `alloc_pooled` drops any stamp for a cell it hands out
again, and a read path that never stamps leaves no stamp -- the release then falls
back to the old release-time mark, which is never less correct. `pool_kill_`
releases stamps with the cells, and `KRK_POOL_SLOT_MARK=0` restores the old path.

**Measured, REAP-288** (`--expert-cache-mb 4000 --expert-warm-mb 20000 --ctx 512
--chunk 256`, 32 greedy tokens, guard on, text `da9ba027...` identical in every
run, 8 interleaved order-balanced pairs):

| arm | ms/step | median | vs release-time mark |
|---|---|---|---|
| release-time mark (before) | 189.5 / 179.1 / 178.0 / 178.4 / 180.6 / 182.0 / 183.4 / 190.2 | 181.3 | -- |
| per-slot mark | 176.2 / 153.9 / 156.7 / 156.2 / 155.4 / 161.6 / 160.0 / 159.4 | 158.3 | **-12.7%, faster in 8 of 8** |
| racy reference (no guard at all, 12n) | 142.9-150.3 | 146.6 | the target it cannot beat |

So the guard's price at this shape falls from **+24% to +8%** against the arm that
was reading garbage, and the mechanism is what 12n predicted: the wait is now for
that slot's readers instead of for the layer's own earlier experts.

**What the marks cost.** 12.7k records over 32 tokens is ~400 per token, about
1.2 ms/token at a nominal ~2.5 us per record -- an estimate that section 12q
measures 25-40x too high (~0.1 us), and either way it is inside the measured win. It is only worth paying when a cell can
actually be reused, so the recording is gated on the pool having released one --
`release_slices > 0`. Without that gate the default auto budget on this file
(`reuse 0`: the routed set fits) paid 410 records a token for nothing: step
155.6-199.0 against 103.9-134.1, a spread wide enough to be noise but with the
counter at 8k marks and nothing to spend them on. With the gate, that
configuration reports `reader-done 0` and the two arms run the same code.

**Not a correctness change**: `kraken-bench --pool` still passes its reuse traps
(the trap cells are never stamped, so they exercise the release-time fallback),
the suite is 3841/3841 and the coherence arms are unchanged.

### 12p. Picking a victim whose readers have retired: neutral, and the reason is 12o's own

12o left the guard costing +8% against the racy arm, and the marks make one more
thing visible that 12n could not see: *when* a slot's readers finished. So the
next question was whether eviction can pick a victim whose reuse costs no wait at
all, and leave only the genuinely in-flight case paying.

**The signal, without an event query.** `hipEventQuery` on a pending event is
~50 us, which is far too much for a victim search. What is cheap is a counter:
`Backend::pool_drain_generation()` is bumped by every call that leaves the
default stream EMPTY (each blocking D2H, and `sync()`), so any reader-done mark
recorded before the latest drain is complete **by construction**. Each slot
carries the generation its last read or promotion was stamped with, so
"are this victim's readers retired?" is a field compare. `force_reclaim` then
picks in two modes over the same class: retired first, first-match second. The
fallback is the whole point -- an arm that only evicted retired slots would stop
allocating once the token's own reads were all in flight.

**Measured, REAP-288** (`--expert-cache-mb 4000 --expert-warm-mb 20000 --ctx 512
--chunk 256 --expert-warmup 1`, 32 greedy tokens, text `da9ba027a345` in all six
runs, 3 interleaved pairs with the arm order alternating):

| pair | retired victim | first match |
|---|---|---|
| 1 | 164.4 | 157.2 |
| 2 | 155.3 | 161.3 |
| 3 | 157.4 | 155.0 |
| **median** | **157.4** | **157.2** |

Faster in 1 pair of 3, +0.1% by median, and `reader-done` is 12.65-12.73k in both
arms: the pick changes *which* cell is reused and nothing else. Reverted -- the
mechanism is real but there is no wait for it to remove.

**Why, and it is 12o's result read the other way.** A reused cell now waits for
exactly its own readers. The cells a reclaim finds are, in the overwhelming
majority, slots whose last read was at least a token ago; under either rule that
mark is complete and the reuse waits for nothing. So the residual is not the
victim: it is the ~1.2 ms of event records plus the case where the evicted slot
was read *inside the current token*, and preferring another slot does not remove
that wait -- it moves the same eviction to a different cell that will be needed
just as soon.

**The trap that nearly recorded the opposite.** The arm was first run as
`run KRK_EVICT_RETIRED=0`, where `run` is a shell function: `"$@"` puts that word
on kraken's **ARGV**, not its environment, so every treatment run died with
`kraken: unknown option`. All three died the same way, so their text md5s were
identical -- which reads as "same text, the arm is safe" -- and with no `ms/step`
line to compare the pair looked like a null result rather than a broken arm. It
is the same shape as the attention harness bug ("a harness bug that makes every
arm the same produces exactly the verdict that the change is free"): `env VAR=0
cmd` is the form, and a metric that is *missing* is a louder signal than one that
agrees.

**What is left** is not eviction policy. It is the number of reuses (a hold-off
so a slot used in the last N experts is not handed out again -- but at a
saturated budget a reuse IS the allocation, so that means refusing to allocate),
or amortising the record itself (one mark per burst instead of per slot, at the
cost of waiting on a burst's worth of readers).

### 12q. One reader-done mark per expert burst: 12o's record cost was 40x overstated, and the burst default is worth 0.7%

12o left the guard at +8% against the racy arm and accounted ~1.2 ms/token of that
to the reader-done records themselves (~400 of them a token at a nominal 2.5 us
each). That number was never measured, only multiplied. It is now: a record costs
about **0.1 us**, and one event per burst of expert uses takes the step from
151.9 to 150.9 ms/step median.

**What landed.** `ExpertCache::note_read` no longer records per use. It appends
the slot's cells to a pending set and emits ONE `pool_note_reader_done` for every
`KRK_POOL_MARK_BURST` uses (default **8**, i.e. one record per layer at top-k 8;
`KRK_POOL_MARK_BURST=1` is 12o's behaviour). Three details make it safe:

- **A later mark is conservative, never wrong.** The event is recorded on the
  default stream at a position after every kernel enqueued so far, so a cell in
  the burst gets a mark that covers its readers *and* whatever followed them. The
  only thing coarser marks can cost is a waited-for position that is too far
  ahead.
- **The burst never spans a layer**, and it is flushed explicitly at the end of
  the expert section (before the next-layer prefetch block can trim or free a
  cell), so the last experts of a layer are stamped at their own position rather
  than by the next layer's first read.
- **A released cell is scrubbed from the pending set** in `drop()`. Otherwise the
  flush could record a mark for a pointer that has already gone back to the free
  list -- and, in the trim path, been freed.

**Measured, REAP-288** (`--expert-cache-mb 4000 --expert-warm-mb 20000 --ctx 512
--chunk 256 --expert-warmup 1`, 32 greedy tokens, text `da9ba027a345` in every
run, 7 interleaved pairs):

| pair | burst 1 | burst 8 |
|---|---|---|
| 1 | 154.1 | 150.8 |
| 2 | 151.8 | 150.5 |
| 3 | 153.5 | 152.9 |
| 4 | 151.8 | 162.0 |
| 5 | 151.9 | 151.2 |
| 6 | 152.5 | 150.4 |
| 7 | 151.0 | 150.9 |
| **median** | **151.9** | **150.9** |

Faster in 6 of 7 pairs (pair 4 is one 162.0 outlier, the only reverse), records
**12 650-12 706 -> 2 186-2 247** over the 32 tokens, i.e. ~400/token ->
~69/token. A first coarse sweep put burst 4 at 151.8 against 153.4-153.6 for
burst 1, and burst 16 at 150.3: 4 is not enough to win, 8 and 16 both saturate
because the layer-change flush already caps a burst at one layer.

**So what does a record actually cost?** 328 fewer records per token bought
~0.03 ms/token, which is **~0.1 us per record** -- a queue submission on the
default stream, not a synchronisation. 12o's "~1.2 ms" was a nominal per-record
price times a count, and the real price is 25-40x lower; the correction is the
useful part of this section, because it says the records were never the residual.

**On the other shape it is a wash**, as it should be: Qwen3-30B-A3B
(`--expert-cache-mb 4000 --expert-warm-mb 8000`, same 32 tokens) gives 36.1 / 34.5
against 35.8 / 35.7 ms/step with records 12 032 -> 1 629, text identical. Where
the guard's wait is a smaller share of the step there is no 0.7% to take.

**What the expected cost was, and why it did not appear.** A coarser mark is a
*later* mark, and 12n showed that a mark's position is the whole cost of a reuse.
The reason nothing was lost is the same reason 12p was neutral: the cells a
reclaim hands out were last read at least a layer ago, long before the burst's
own event, so the extra coverage is already satisfied by the time anyone waits on
it.

**Gates**: `kraken-bench --pool` still passes its reuse traps (unstamped cells
still fall back to the release-time mark), the suite is 3841/3841, the baseline
has nothing newly broken and the coherence arms are unchanged.

### 12r. Where the guard's remaining wait goes: 100% of it is a stream order, none of it is the host

12p and 12q both removed a candidate (the victim choice, the record count), so the
question left was the one 12n could only attribute in aggregate: when a reused cell
waits on its mark, who waits? Two things were added to answer it and to keep
answering it.

**The trace.** `ExpertPoolStats` already counted how a reuse was honoured --
`order_device` (the writer's stream waits on the mark), `order_host` (no stream to
order, so the host blocks on the event), `order_implicit` (the writer runs on the
default stream, where the mark was recorded, so stream order already covers it) --
but the line never printed them, and no time was measured for any of them. Now
`pool_honor_host_` times its `hipEventSynchronize` into `honor_host_ms`, the
`[stats ] expert-pool` line carries `honour device/host/implicit` and
`host-wait <ms>`, and `KRK_POOL_NO_WAIT=1` skips the wait (bookkeeping unchanged,
**deliberately racy, measurement only**) so the guard's total can be measured in
the same binary that shows its parts.

**Measured, REAP-288** at the 4000 MiB / WARM 20000 shape (32 greedy tokens, text
`da9ba027a345` in every run, 7 interleaved pairs):

```
honour device 12497-12719  host 0  implicit 0
host-wait 0.000 ms
guard on   157.6 151.4 151.3 156.7 152.1 153.3 151.0   median 152.1
KRK_POOL_NO_WAIT=1
           148.4 154.2 148.6 148.7 150.2 149.4 152.8   median 149.4
```

So the guard costs **+2.7 ms/step, ~1.8%** at this shape (slower in 5 of 7 pairs),
and **every microsecond of it is device-side**: not one honour on this file takes
the host-blocked path, and the measured host wait is exactly zero. ~12.5k honours
over 32 tokens is ~390 per token, so the average honoured reuse costs about
**220 ns** -- most are free, and the cost is concentrated in the few that arrive
while the evicted expert's kernels are still in the queue.

**The chain, end to end.** `pool_honor_` issues `hipStreamWaitEvent(pinned_stream_,
mark)`; the DMA runs on that private stream; the batch then records `pinned_ev_`
and the DEFAULT stream waits for it (`hipStreamWaitEvent(0, pinned_ev_)`). A reuse
that arrives before the previous owner retired therefore pushes the *copy* back by
that remainder, and the next layer's first kernel waits for the copy. Nothing
blocks the host, which is why `host-wait` is 0.000: the cost is promotion latency,
not a stall.

**That names the only fix left**: stop reusing a cell whose readers are in flight.
Both remaining routes are supply-side -- keep a small reserve of free cells so an
allocation does not have to take a just-read one (a few MiB at this pool size), or
promote ahead of demand, which 12m showed has no candidates for the *next* layer
because the hinted experts are already resident. Anything that refines the *wait*
(the placement of the mark, the victim choice) has now been measured three times
and is exhausted.

**Also worth knowing**: the +8% figure 12k-12p carry was measured before the
per-slot mark and the burst default; at the current default the guard's price on
this file is the 1.8% above, and the counters that split it are now on every run's
stats line rather than behind a rebuild.

### 12s. The prefill gap is a DISPATCH trade, not a missing kernel: 5.6x is behind one flag

The prompt numbers had been read as "the matrix cores are not used". They are
used, and the reason auto does not take them is a deliberate invariant, not a
gap in the kernel:

```c
} else if (exact_gemm_enabled() && path_ == GemmPath::kAuto && ...) {
    gemv_grid_kernel<<<...>>>(...);   // one block per (output row, x-group)
    krk_op_scope.retag("gemm(grid)");
} else if (rung == GemmRung::kWmma) { ... }
```

The exact grid comes FIRST, and `exact_gemm_enabled()` is on by default: the grid
position selects rows and never the accumulation order, so every prompt row is
bit-identical no matter how `--chunk` splits the batch. That is what the
chunk-invariance gate requires, and it costs the matrix cores.

**The device evidence** (`--profile`, Qwen3-4B-Q8_0, 260-token prompt, same 252
prefill GEMM calls in both arms):

| arm | op tag | device time | efficiency |
|---|---|---|---|
| auto | `gemm(grid)` | 716.97 | 5.4 |
| `KRK_GEMM_PATH=wmma` | `gemm(wmma)` | 53.02 | 80.5 |

**Throughput, 1145-token prompt, greedy, 4 generated tokens:**

| model | exact (default) | `--prefill fast` | |
|---|---|---|---|
| Qwen3-4B-Q8_0 (dense) | 307.0 tok/s | **1707.0 tok/s** | **5.56x** |
| Qwen3-30B-A3B (MoE, 4000/8000) | 170.8 tok/s | **325.5 tok/s** | **1.91x** |

The MoE file gains less because its expert path (`gemm_group`) is the dominant
term there and stays exact; on a dense model the dense GEMMs are the whole
prompt.

**What it costs, measured.** `--prefill fast` is a different-arithmetic mode, not
a free speed knob:

- the generated text differs -- 4 greedy tokens from the same prompt go
  `dbc606bb39b8` (exact) to `b9d995daaa3a` (fast) on the 4B, and
  `387ad1b06c56` to `2ef0b2bd34e1` on the 30B;
- the prefill hash is **identical at `--chunk 1` and `8`** -- the thin-row GEMV
  ladder agrees with the exact grid bit-for-bit -- and **differs from `--chunk
  16`** (`b3222295` -> `a48ba8d6`) and at 256 (`ff962cf9`). So the handover at 9
  rows is the whole of the divergence, exactly as the tile-rung note in
  `AGENTS.md` says.

That is the same trade the reference engine makes: `STRATA_HIP_WMMA=1` buys about
30% there and changes the output's last-place bits too. Where the two differ is
the default: kraken keeps exactness unless asked.

**`KRK_GRID_G` does not buy it back**: 32/64/128/256 on the exact grid measure
306.9 / 309.4 / 309.1 / 308.6 tok/s, i.e. the grid is instruction-bound, not
group-bound.

**What would close it properly** is a matrix-core rung whose accumulation order
does not depend on the row count -- fixed row groups (pad a thin chunk to the
tile's BM) with no row-dependent split-K -- so the 5.6x arrives without the
invariant break. That is a kernel change with a real design cost (thin prefill
chunks would pay tile overhead they currently avoid), and it is the only route
left: the dispatch itself is one `&&`.

## 13. The host table now covers what is not a kernel

The device table (`src/hip/op_time.hpp`) can only see backend ops, so on every
file whose step is dominated by something that is not a kernel the reader was
left with a large unexplained remainder and no row to attach it to. Two of those
costs were invisible by construction:

- **`expert.warm.read`** -- a cold expert's DISK-TIER read (`pread`, or a
  mapping memcpy) on the calling thread, inside `acquire`. It is not a kernel
  and it is not a transfer to the device, so neither table had it.
- **`expert.ahread`** -- the same read on the read-ahead worker. It runs BESIDE
  the main thread, so it is marked `*` in the table and must not be added to a
  step's total.

Measured on `Ornith-1.5-122B-Q4_K_M` (a 122B `qwen35moe`, routed set 65.4 GiB
against a 7.8 GiB device budget, 32-token prompt, `--expert-cache-mb 4000
--expert-warm-mb 8000`), `KRK_TIME_HOST=1`:

| phase | total ms | calls | % of run |
|---|---:|---:|---:|
| prefill.chunk | 2150.7 | 1 | 87.9% |
| moe.experts.batch (48 layers) | 2076.0 | 48 | 84.9% |
| **expert.warm.read** | **1571.6** | 3453 | **64.2%** (6278 MiB at 3.9 GB/s) |
| **expert.ahread*** | 215.0 | 435 | 8.8% |
| moe.route.fetch.batch | 57.5 | 48 | 2.3% |

That is the answer this instrumentation exists for, and it is not a kernel
conclusion: 64% of the run is a synchronous page-cache read of 3.9 GB/s, and the
worker thread moves a further 834 MiB alongside it. Neither number could be read
off the device table before.

The same table on a dense model puts the cost somewhere else entirely --
`Qwen3-4B-Q8_0`, 8 tokens: `decode.step` 66.8 ms of a 92.1 ms run, of which
`decode.fetch` (the logits/top-k readback) is 60.4 ms, i.e. 65.5% of the wall
clock is one blocking D2H copy and not compute.

Two reporting fixes came with it, both because the old columns invited a wrong
reading:

- **`%run` beside `%acct`.** The single percentage was a row's share of the
  accounted total, so it always summed to 100% whatever the run took. `%run` is
  the share of the wall clock actually waited, which is the number that decides
  what to fix.
- **`nested excess`, not a negative `unaccounted`.** The spans nest
  (`prefill.chunk` contains `moe.route` contains `moe.experts`), so the accounted
  total exceeds the run and `wall - total` went negative -- printed as
  `unaccounted -23154.5 ms (390.8%)`, which is not a number anyone can act on.
  The row now flips to `nested excess` with its magnitude, and `unaccounted` is
  printed only when it is real.

### 13a. The cold read was one handle at queue depth 1

`expert.warm.read` at 3.9 GB/s invited the obvious comparison with the 24.3 GiB/s
in-place path, and the comparison is wrong: those are two different legs. The
in-place figure is a kernel reading bytes that are ALREADY in pinned host RAM;
the 3.9 GB/s is the copy that puts them there (a `ReadFile` out of the page
cache). Nothing about the GPU can speed up the second leg.

So the question is what the copy leg can do at all. `tools/probe_readspeed.cpp`
(ad hoc, host-only, not a CMake target) reads a page-cache-resident region of the
model file in expert-sized chunks over N independent handles:

| readers | 1792 KiB chunks | 8 MiB sequential |
|---:|---:|---:|
| 1 | 7.37 GB/s | 6.60 GiB/s |
| 2 | 10.96 GB/s | |
| 4 | 13.02 GB/s | |
| 8 | 12.88 GB/s | |

One reader alone does 7.37, so the engine's 3.9 was not the machine: `warm_fill`
issued **three `read_host` calls**, i.e. one HANDLE at queue depth 1, once per
slice, and the three slices of an expert are megabytes apart in the file. The
backend already had the pool (`read_host_batch`, 4 handles, which
`prefetch_layer` uses for a whole layer) -- the miss path simply was not using
it.

`warm_fill` now allocates the slices and hands them to the pool as one batch.
Same bytes, same fallback (the mapping memcpy when the file path is unavailable),
verified by `--prefill-hash` on the real file: `0f3826b4900d47b5 / logits
8ccce7f8bc347cb4` at `KRK_WARM_READERS=1` and at `=4`, i.e. the batched copy
produces the serial copy's bytes.

Measured on Ornith-122B, `--expert-cache-mb 4000 --expert-warm-mb 8000`,
32-token prompt, two runs each arm:

| | expert.warm.read | bandwidth | prefill.chunk | run wall |
|---|---:|---:|---:|---:|
| three `read_host` calls | 1571.6 ms | 3.9 GB/s | 2150.7 ms | 2446.4 ms |
| one batched read | 975-1018 ms | 6.1-6.4 GB/s | 1629-1656 ms | 1875-1907 ms |

That is **-37% on the read leg and -23% on the wall clock**, from a change that
moves no more data. At the 24 GiB WARM tier (the best configuration for this
file) decode went from 3.6 to **4.2-4.4 tok/s** (279 -> 229-235 ms/step).

Splitting each slice in half to engage a fourth reader was measured and
rejected: the pool sorts requests by offset and gives each worker one contiguous
run, so six requests over four workers still land as three streams (2+2+2) --
6.1 GB/s against 6.3, same text. The pool's distribution, not the request count,
is what shapes the read.

What is left on this leg is thread spawn: `read_host_batch` creates and joins a
`std::thread` per call, 1170 calls for this run. A persistent reader pool is the
next step if the row is ever the step again; at 6.4 GB/s it is 34-45% of these
runs, down from 64%.

### 13b. Routing is the decode step, and the .krakenexperts.json cannot touch it

The component table, split by phase (the reason it is split: `load` is a third
of a short run and dilutes everything else), on Qwen3-30B-A3B at the 10 GB VRAM /
24 GB RAM plan, `KRK_TIME_HOST=1`, 16 greedy tokens, two runs agreeing:

| phase | component | ms | % of that phase |
|---|---|---:|---:|
| decode | `decode` (the step) | 23.3 per step | 100% |
| decode | routing | 0.468 per layer | **~90% of a step** |
| decode | of which `moe.route.fetch` | 0.30 per layer | **~64% of routing** |
| decode | of which `moe.route.rest` (top-k + plan) | 0.155 | ~33% |
| decode | of which `moe.route.upload` + gemm | 0.014 + 0.004 | ~4% |

So of a 23.3 ms decode step, about 21 ms is routing, and 14 ms of that is ONE
blocking device-to-host copy of the router logits per layer -- 512 bytes, at
0.30 ms a round trip, 45 times per token. The copy is not slow because of its
size. It is the queue round trip on WDDM: the host asks, the driver drains,
the caller blocks, and the GPU has finished the router GEMM (0.004 ms) long
before the answer arrives.

**The `.krakenexperts.json` does not change this, and was never going to.** That
file is `--expert-scan`'s ranked hot set: it decides which experts the tiers
preload, i.e. how many *reads* a token pays. Routing is the selection step, and
it costs the same whether the expert it picks is already in VRAM or not. The 30B
file already has one (`.krakenexperts.json`, 268 KB, dated Oct 20) and the run
above used it.

What actually cuts it, ranked by measured share:

1. **A device-side dispatch** (`KRK_DEV_PTR=1`) removes the round trip entirely:
   routing stays on the device and the expert kernel reads the plan. It engages
   only when the whole routed set is device-resident -- on a 10 GB card with a
   9.8 GiB routed set it reports `0 of 6144 slots` and falls back, which is
   exactly why it is not the answer *here*. On a card that holds the set it is
   the whole 90%.
2. **A plan built on the device** (`KRK_DEVICE_SOFTMAX=1` + device top-k/plan,
   already implemented as `router_topk` / `moe_plan_build`) removes the 33% that
   `moe.route.rest` is -- but not the 64%, because the host still asks for the
   ids it needs to acquire experts.
3. **Overlapping the copy with the layer's own work**: the router logits are
   issued right after the router GEMM, but the code blocks on them immediately.
   Issuing the copy and consuming it just before the expert loop would hide most
   of the driver latency behind the plan work and the prefetch. This is the only
   option that helps on a card where the routed set does not fit.
4. Batching is not available for decode: one route per layer, in layer order.

Measured caveat: `KRK_TIME=1` (device events) inflates this row to 1.9 ms per
call, because the span waits on the recorder's own ~20 us per op. That is why
those numbers are quoted from `KRK_TIME_HOST=1` only -- the same trap the
`KRK_TIME_HOST` switch was added for.
