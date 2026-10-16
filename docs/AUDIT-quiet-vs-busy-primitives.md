# Audit: `kraken-bench`'s per-op primitives, quiet queue vs in-step

2026-10-11. Every per-op constant in this repo comes from `kraken-bench`, and
every one of them is measured with **one call in flight on an idle queue**. A
decode step is never in that state: it issues 243-2965 ops with hundreds already
queued. This audit takes each primitive, finds where the repo uses it to rank
engine work, and measures either the same call with a saturated queue or the
engine row it is supposed to explain.

Raw evidence: [traces/quiet-vs-busy-primitives-2026-10-11.txt](traces/quiet-vs-busy-primitives-2026-10-11.txt).

## The inventory (what the tool actually prints)

`kraken-bench` (no flags) prints per-op constants on four lines, plus a work
matrix that is not a constant:

| arm | what it measures | printed as |
|---|---|---|
| `launch floor (copy_act n=32)` | one trivial kernel launch, empty queue | µs/call |
| `pool primitives` | event create+destroy, **record**, query, wait, record+query-pending | µs each |
| `pool round trip` | `alloc_pooled`+`release_pooled` pair, 8 cells of 256 KiB, with and without a queued reader | µs/pair |
| `pool burst` | 3 slices released in one call vs one record per slice | µs/burst + counter ratio |
| `--floor` | 1 MiB `copy_act`, device-to-device | GB/s, µs |
| per-dtype rows, `--sweep` | real kernels with a host-dot check | µs, GB/s, TFLOP/s |

The per-dtype rows and `--sweep` are work measurements with their own
correctness check; they are not "per-op cost" claims. The five above are, and
three of them are load-bearing: the launch floor normalizes tables, the record
sized a ranked work item, and the pending-event query explained an engine row.

## The measurement, three ways

`bench_ops.cpp` gained a fifth line — the same calls, but issued while the queue
is deep — because a constant that only exists on an idle queue cannot be checked
against a step that is never idle:

```
primitives, busy queue (host side, 512 x copy_act n=32 already issued):
  record 0.32 us | copy 0.92 us per call, median of 8
  [empty queue: record 28.16 | copy 2.88]
```

It cannot use `time_calls`: that helper subtracts one cold call from a batch mean,
and for a host-side enqueue the cold call is the expensive one — the first version
of this arm printed a **negative** marginal record cost (−12.82 µs), which is what
an instrument that cannot resolve its own quantity looks like. The arm now feeds
the queue with real device work (512 × `copy_act n=32`, ~1.5 ms of device time)
and times 64 calls inside each of 8 blocks, reporting the median.

The third leg is the engine itself: a switch that removes exactly N of an
operation, interleaved A/B, order-balanced, text md5 checked.

| primitive | quiet queue (bench, quoted) | busy queue (host enqueue) | in-step marginal (A/B) | factor |
|---|---|---|---|---|
| record on the default stream | **28.2-30.6 µs** | 0.32 µs | **2.56 µs** (7252 records = 18.6 ms) | **quiet overstates ~11x** |
| launch, trivial (`copy_act n=32`) | **2.85-3.05 µs** | 0.92 µs | ≤ 2.43-2.76 µs per added call *with a memory pass* (`KRK_FUSED_SILU=0` −4.591%, `KRK_ACC_GEMM=0` −5.219%, both +30 launches, 0/9 pairs, md5 identical) | quiet ≈ the with-pass cost; **overstates a pure enqueue ~3.3x** |
| query a completed event | 0.03-0.04 µs | — | — | not load-bearing |
| wait on a completed event | 0.07-0.11 µs | — | (a stream wait is an enqueue: 0.3-0.9 µs) | not load-bearing |
| **record + query a *pending* event** | +24.7-46.9 µs | — | **46.9 µs, validated at 1.0x**: 5560 cells × 87 µs = 484 ms against a measured `alloc 481.8 ms` | **1.0x — this one was calibrated** |
| a *device drain* (`hipDeviceSynchronize`) | not measured by the tool at all | — | **~129 µs** (28 layers, −30.293% order-balanced, 0/9 pairs) | the wait arm is 1000x off for this call |
| pool round trip (256 KiB cells) | 18.8-33.7 µs | — | 16.5 µs per alloc in-step (91.1 ms / 5514 cells) | ≤1.6x, either direction |
| `--floor` shelf (1 MiB `copy_act`) | 240.9 GB/s | — | in-step `xfer` is H2D: 872 ms for 1836 x 5.8359 MiB ≈ **12.9 GB/s** (11.24 GB) | wrong shelf for H2D (its ceiling is the reader's 15.9 GB/s) |

## Ranked: what is ranked on a bad number

1. **The release record — 11x too big, and it was the #1 ranked item.** The row
   was recorded as "~23 µs × 5560 = 128 ms" and drove the top entry of
   `AUDIT-active-expert-staging.md`'s next-actions list and `docs/TODO.md`.
   Measured in-step: **2.56 µs** (7252 records removed = 18.6 ms), so the row was
   ~13 ms and the burst-sharing change that closed it is worth ~0.5% of the step,
   not 8%. This is why the tool now reports the busy-queue figure beside the quiet
   one.
2. **The launch floor is four different numbers and none of them is "the"
   marginal call.** Quoted as 2.65 µs (`PERF-PLAN.md` §0c, used to normalize the
   harness table, and §B1's "34-43% of each 6-8 µs call"), re-measured as
   0.98/1.58 µs (§B1's own historical note), 3.64 µs (`AGENTS.md`) and **2.85-3.05
   µs today** — a 3.7x spread, from arm shape and machine state, not from the
   engine. The in-step numbers are 0.92 µs for a pure enqueue and 2.4-2.8 µs for a
   call that also carries a memory pass, so: **tables normalized on 2.65 are
   within ~10% for a call with a pass, and overstate a pure enqueue ~3x.**
   `AGENTS.md`'s "the marginal launch costs ~1.5 us" is the stale end: the same
   residual fold that produced it now measures 2.4-2.8 µs on this build.
3. **The wait family has two members that differ by 1000x.** "Wait on a completed
   event" (0.07-0.11 µs) is honest, but it is not the cost of a *drain*: an
   explicit `hipDeviceSynchronize` before each router readback (the opt-in
   `KRK_ROUTER_SYNC=1` arm) costs **30.293% of the step, ~129 µs per layer**, and
   the pending-event *query* is 46.9 µs. Any ranking that treats "a wait" as
   ~0.1 µs is off by three orders for the call it is actually paying.
4. **The `--floor` shelf is device-to-device and the row it is compared against
   (`xfer`) is host-to-device.** 240.9 GB/s vs an in-step 12.9 GB/s reads as a
   5% utilization defect; against the correct ceiling (the 15.9 GB/s the
   separate-handle reader measures) the same row is at 72% and is simply
   promotion traffic.
5. **The pool round trip is the one constant that was already calibrated**, twice:
   the 46.9 µs pending-event query was confirmed by the row it explained (484 ms
   predicted vs 481.8 ms measured), and today the quiet round trip (18.8-33.7 µs)
   brackets the in-step 16.5 µs. Its error is ≤1.6x and it changes sign between
   sessions, which is exactly what a small constant should look like.

### What to use instead

* **Rank a fold by running the fold** — the switch A/B in one session, interleaved
  and order-balanced — not by a constant times a count. The constants in this
  table are for *sizing* a plan, and they are only good to ~2x.
* If a constant is needed, use the busy-queue column for anything the host issues
  into a saturated stream, and the in-step column for anything that carries a
  memory pass. The quiet-queue record figure (28-31 µs) is a *flush* and belongs
  to no engine row.
* Never compare a host→device bandwidth row against `--floor`; use the reader's
  measured 15.9 GB/s (`AGENTS.md`, weight loading).

## What this audit did not do

* One card, one runtime, one day. The quiet-queue `record` arm alone moved
  22.7-30.6 µs across sessions and the `alloc` row halved between yesterday's
  session and today's (182.5 → 91.1 ms for the same code path), so cross-session
  comparisons of small rows are not evidence; every A/B here is within one session.
* The in-step column is dense-model for the launch rows (SmolLM2-135M) and MoE for
  the drain and record rows; there is no launch A/B on the MoE 35B, whose step is
  2965 ops rather than 243.
* The busy-queue arm measures the host side only. A call that also stalls the
  device (a drain, a blocking copy) has no busy-queue figure in this table beyond
  the drain's in-step one.
* `--floor` was run once (240.9 GB/s, 4.35 µs per 1 MiB); the load-store shelf is
  not the subject of the audit, only its scope is.


## 2026-10-11 (late) — the arm proves itself, and it corrects this document

The table above was written from the arm's *printed line*, and the arm was only
reachable with `--gate` absent (`if (!gate)` in `kraken-bench`), so nothing
checked it. Making it self-proving split each call into its two halves and
disproved this document's explanation of its own first row.

**What the arm measures now, printed apart:** `host enqueue/call` and `device
cost/call` (the time the sync after the block still had to drain, divided by the
block), in both queue states, by one method.

| primitive | host enqueue, busy queue | host enqueue, empty queue | device cost, empty queue | `time_calls` reports |
|---|---|---|---|---|
| record on the default stream | 0.30-0.34 µs | 0.33-0.35 µs | 27.1-29.3 µs | 33.09 µs |
| launch (`copy_act n=32`) | 0.90-0.97 µs | 0.71-0.75 µs | 2.8-3.0 µs | 3.29 µs |

**The correction.** The 28.2-30.6 µs this document called the quiet figure is not
a queue-state effect at all — it is the **device half**. A record enqueues in
0.33 µs whether the queue is empty or 512 deep; `time_calls`' T(N)-T(1) lands on
the device marker (~27-32 µs), which is what its quotient reports. The "0.32 µs
busy host enqueue" in the table above was therefore not an alternative to the
28.2 µs — it was the other half of the same call, and comparing the two read a
method difference as a state difference. The **in-step** column is unaffected and
is still the number to rank with: 2.56 µs for the record, because in a step the
markers overlap hundreds of queued launches instead of serialising.

**Two engine rows this document was missing, both measured by switch A/B in one
interleaved order-balanced session** (A = old path, B = new):

| row | ops/step | A/B | text |
|---|---|---|---|
| MoE per-visit `upload_i32`+`upload` pair -> one plan per layer (`KRK_MOE_PLAN=0`) | 705 -> **649** (56 = 2 x (visits-1) per layer x 28 layers) | **+10.027%**, 5/5 pairs, MAD 2.870% | identical |
| per-token `hipDeviceSynchronize` in front of the topk 16-byte readback (`KRK_TOPK_SYNC=1` restores) | +0 (one drain/token) | SmolLM2 **+5.170%**, Qwen3.5-0.8B **+4.412%**, 5/5 pairs each, MAD 2.076% / 0.631% | identical |

The first row is the same error class as the record row: a **blocking**
`hipMemcpy` of a few bytes (`upload`/`upload_i32` in `src/hip/backend_hip.hip`)
was priced as a launch. 56 removed launches at the measured 2.4-2.8 µs is 1.6%
of the step; it measured 10.03%, i.e. ~15 µs per removed copy.

**The arm's own gate, and its red construction.** `kraken-bench --gate` now runs
the arm and asserts: a queue holding `depth` calls is measurably fuller than an
empty one (launch-arm tail >=3x; measured 177 -> 583-660 µs), the same per call,
a record's device half is >=4x its host half, the host enqueue is the same in
both states, and the instrument resolves the device cost per op. With
`KRK_TEST_QUEUE_NOPREISSUE=1` (the pre-issue skipped) **exactly the two fullness
checks fail, rc=1** — the arm has been shown able to go red.

**Still not done: the launch floor on the 256-expert 2965-op step -- and it
cannot be done on this box, which is itself the finding.** The unprofiled baseline
exists now (ctx 256, `-n 8`, warm-up on): **decode 12.2 tok/s = 81.7 ms/step**,
`expert-path read 0.0 | room 0.0 | alloc 64.8 | xfer 67.7 ms` across all 7 steps,
HOT 96.4%, 0 COLD, 0 MiB read from the file. So this document's earlier
"allocation-bound" reading is **falsified** -- promotion is ~9 ms of an 81.7 ms
step, and the step is not the allocator's. But the same run printed
**`[stats ] device 15.92 GiB in use of 15.92 GiB (0.00 GiB free)`**: the warm-up
promoted 8888 MiB of experts until the card was full to zero, and that state took
the machine's display down twice (once with the run bounded to `timeout 300` and
`-n 8`, the profiler off). The A/B needs ~10 such runs. **A launch-count probe is
therefore off the table for this file until the tier leaves VRAM headroom**, and
the 12.2 tok/s figure was taken under the fault, so it is not a clean baseline
either. Measurement recipe, not result:
`docs/traces/launch-floor-and-drains-2026-10-11.txt`.
