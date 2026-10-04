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

### 10.6 What is left, and the ceiling

| stage | ms | note |
|---|---:|---|
| `read` | ~2050 | 8.3 GiB from the file at the device's own ~4 GB/s |
| `alloc` | ~1150 | `hipMalloc`/`hipFree` churn from the pooled allocator |
| `xfer` | ~500 | 8.3 GiB pageable to VRAM, now overlapped |
| dense | ~1000 | 8.5 ms/token of real compute |

The file read is now the largest single item and it is **at the device limit**:
`probe_read.py` shows 32 threads read no faster than one. 347 MiB/token at
4 GB/s is ~87 ms/token of unavoidable I/O for a model whose expert working set
(8.3 GiB of first-touch weights) does not fit in the 8.4 GiB of VRAM the cache
has. Kraken at 96 ms/token is within ~10% of that floor, which is also why the
reference implementation is not 3x faster: it is streaming the same cold bytes.

That reframes the remaining work:

1. **`alloc`, ~1150 ms (18%).** 13,770 `alloc_pooled` calls at ~82 us each. The
   pool is capped at 256 MiB of free blocks and expert slices are never returned
   to it in volume, so most of those calls fall through to `hipMalloc`, which
   synchronises the device. A dedicated expert arena — one allocation at init,
   sub-allocated by slice, with the same event guard — removes the allocator
   from the hot path entirely.
2. **Multi-expert GEMV**, 960 -> 120 launches. Previously bounded at 3%; the
   real bound is the launch count against the 8.5 ms of dense-path time.
3. **DFlash speculative heads.** Unchanged from section 6 and still the only
   item that amortises the ~87 ms I/O floor itself rather than the work around
   it: accepting 3 tokens per pass pays the file once for three positions.

### 10.7 Reproducing section 10

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
