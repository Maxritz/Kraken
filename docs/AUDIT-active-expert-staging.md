# Expert VRAM arena repair and active-expert staging: measured outcome

2026-10-11, sherlock `trace` mode. Two requests, one pass:

1. Fix and verify the expert VRAM arena's accounting, physical budget enforcement
   and slot ownership under constrained MoE workloads.
2. Implement output-preserving active-expert staging (overlapping promotions) and
   measure whether it improves constrained-budget MoE decoding.

Raw evidence, unabridged: [traces/active-expert-staging-2026-10-11.txt](traces/active-expert-staging-2026-10-11.txt).

**Verdict, up front.** The arena accounting, the physical budget and the slot
ownership contract are repaired and covered by tests. Staging is implemented,
bit-identical in output, and **default off** (`KRK_ACTIVE_STAGE=1` opts in): it does
**not** improve constrained-budget decoding, and its sign there is consistently
negative. The reason is measured, not argued -- at a budget that forces a promotion
per expert visit the step *is* the promotion (75% of decode wall time), and the
reader window a copy would hide behind is 0.34 ms against a 0.69-0.74 ms copy.

The promotion path's *other* big term did fall, and by more than the staging
experiment was chasing: the 481.8-618.4 ms `alloc` row was a readiness query on a
pending event (46.9 us), not the allocator. Deferring that guard to the device cut
it to 182.5 ms and decode went **+11.279% (16 MiB, 9/9 pairs)** and **+8.569%
(32 MiB, 9/9)**, with the resident control inert as it must be. The slot lease
also has a device-side gate now, not only a stub-backend one: see
[traces/expert-pool-allocator-2026-10-11.txt](traces/expert-pool-allocator-2026-10-11.txt).

## Baseline (the unmodified target)

| Item | Value |
|---|---|
| Model | `models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf` (qwen3moe: 28 layers, 1024 embd, 4 experts x top-2, expert 5.45 MiB) |
| Card / host | AMD Radeon RX 9070 XT (gfx1201), 15.9 GiB VRAM, 96 GB RAM |
| Workload | prompt "The history of computing is a history of abstraction", `--ctx 512 -n 32 --greedy` |
| Arms | `--expert-cache-mb 16` (2 slots), `--expert-cache-mb 32` (5 slots), and the auto/resident budget |
| WARM | `--expert-warm-mb 768 --expert-warmup 0` |
| Baseline md5 (stdout, all arms) | `28fa19c3733a5cab9dc53611de71600c` |

| Budget | decode | promotions | HOT hits | promotion share of decode |
|---|---|---|---|---|
| 16 MiB (2 slots) | 17.2 tok/s, 58.1 ms/step | 1846, 10059 MiB, 1391.6 ms | **0.0%** | 1391.6 / 1802.0 ms = **77%** |
| 32 MiB (5 slots) | 19.2 tok/s, 52.1 ms/step | 1817, 9904 MiB, 1188.9 ms | 1.6% | 1188.9 / 1614.6 ms = **74%** |
| resident | ~109 tok/s | -- | ~100% | ~0 |

Both constrained arms sit **below the working set**. One layer's routed set is
4 experts x 5.45 MiB = 21.8 MiB, so 16 MiB (2 slots) cannot hold even one layer and
32 MiB (5 slots) holds one layer at a time -- the next layer's four experts evict
it, which is what the 1.6% HOT rate at 32 MiB measures. So "promote on every
visit" is structural at these sizes, not a tuning artifact.

## Findings, ranked by measured cost

| # | Site | Raw measurement (16 MiB arm) | Share of the decode step |
|---|---|---|---|
| 1 | Promotion copy path | 1846 promotions, 10059 MiB, 753.8 us/expert, 1391.6 ms | 77% |
| 2 | ~~`alloc_pooled` inside the promotion~~ **FIXED 2026-10-11** | 481.8-618.4 ms -> **182.5 ms** over 5560 cells. The row was a readiness query on a *pending* event (46.9 us), not the allocator: see the section below | 34% -> 10% |
| 3 | WARM file read | 599 MiB total, 75.4-77.7 ms, 110 COLD misses (6.0%) | 4% |
| 4 | Reader window available to hide a copy behind | 0.32-0.34 ms per expert | see below |
| 5 | Staged copy itself | 0.69-0.74 ms per expert | -- |

The whole-model picture is the one the cache already reports: WARM **100%** of
acquires at 16 MiB, 1844 VRAM evictions all kept in WARM, and 599 MiB read from the
file with 110 COLD misses (6.0% of acquires). 110 is 28 layers x 4 experts, i.e.
the first touch of each expert -- so the file reads are not decode churn, and the
storage tier is not the bottleneck here. The promotion loop is.

## Instrumentation added

Two probes, both with a removal mechanism (an environment switch), both printed by
every MoE run:

* **Ownership and physical accounting.** `[stats ] expert-pool budget/committed/live/reusable
  bytes; alloc/reuse/trim/guard-fallback`. This is the audit trap the previous pass
  asked for: logical HOT bytes are no longer the only number, so an arena that
  commits more than its budget is visible instead of inferred.
* **Overlap.** `[stats ] active-stage issued/consumed/demand wait ms` and, under
  `KRK_STAGE_TIMING=1`, `[stats ] stage-timing N staged copies: X ms copy, Y ms reader
  window, Z ms overlapped, hidden H/N`. The copy span is a timestamp on the copy
  stream at submit time closed by an event recorded after the last H2D; the reader
  span is a timestamp on the default stream at the same submit moment, closed when
  that stream drains -- so it measures how much reader work was still queued when
  the copy began, which is exactly the window staging claims to use.

## Root causes

### Proven: the arena's physical accounting did not follow its budget

The previous arena minted 256 MiB chunks per size class regardless of the
configured budget, and a "free" cell was an *interior pointer* of one of those
chunks. Two concrete defects followed, both named as open risks in
[AUDIT-expert-cache-next.md](AUDIT-expert-cache-next.md):

* `alloc_pooled` subtracted `pool_bytes_` on every reuse, while an arena-backed cell
  was never added to that counter on release -- an unsigned accounting underflow.
* the event-creation failure fallback called `hipFree(p)` on `p`, which for
  arena-backed cells is not a base allocation.

Replaced with **individually owned size-class cells**. `committed` is now
`live + reusable`, both in 256-byte-aligned physical bytes; free cells are trimmed
(and `hipFree`d, which is legal because every cell is its own allocation) to satisfy
either a new allocation or a lowered limit; a **live** cell is never freed by a
shrink, and the failure path frees only base allocations. A single expert larger
than the whole budget is still admissible, explicitly, because the engine cannot run
without it: `set_expert_pool_owner(this, max(budget, one_expert))`.

### Proven: two expert caches share one device and one pool

The target and the draft engine each own a cache and each derived its budget from
the card independently. Each cache now registers its budget against its own address
and the pool's physical limit is the **sum** of the registered owners, so the two
cannot each claim the full VRAM. `ExpertCache::clear()` deregisters, and it is
reached from `Model::unload()` in both teardown orders (`shutdown()` unloads the
draft first).

### Proven: slot ownership needed a lease, and a staged copy needed a ticket

* `leased` marks the expert currently being consumed; `pick_victim` skips it, so an
  eviction triggered by a staged promotion cannot take the reader's weights.
  `stage_next` verifies that current **and** next fit before taking the lease, so
  the lease never has to be broken to make room, and a budget too small for both
  returns `false` and falls back to exact serial demand.
* `staged` + `ready` is the publication ticket: the first reader waits on it and
  release the ticket; a staged slot is consumed exactly once (`active_used`) and
  earns no popularity (`touch` on the staged hit path counts a WARM-tier hit, not a
  HOT hit) and no demand heat while staging.

### Proven: at a constrained budget there is no window for the copy to hide in

Reproduced twice on the final binary (868 staged promotions each time):

| Run | staged copies | copy | reader window | overlapped | fully hidden |
|---|---|---|---|---|---|
| 1 | 868 | 645.1 ms (0.743 ms) | 278.7 ms (0.321 ms) | 273.3 ms | 426/868 = 49.1% |
| 2 | 868 | 598.2 ms (0.689 ms) | 296.3 ms (0.341 ms) | 290.6 ms | 467/868 = 53.8% |

The copy is **~2x the reader window**, and half the copies have no window at all.
Two consequences, both structural:

* even a perfect overlap could remove at most the reader-window-sized part of the
  copy, and the copy's own bytes still have to move;
* the step at this budget is promotion-bound (75% of wall time), so the device work
  a copy could run beside is the *other promotions* -- not independent compute.

This is why the switch is inert where the working set is resident (there is nothing
to stage: every routed expert is already HOT) and negative where it is not.

## Results: interleaved, order-balanced A/B

Verdicts from `scripts/kraken_sherlock_kernel_ab.sh` (9 pairs, arm order alternated,
balanced median against the pairs' own MAD), text md5 checked first.

| Binary | Budget | order-balanced delta (B = staging on) | MAD | pairs B faster | verdict |
|---|---|---|---|---|---|
| pre-probe build | 16 MiB | **-4.046%** | 0.694% | 0/9 | REGRESSION |
| pre-probe build | 32 MiB | **-7.346%** | 1.208% | 0/9 | REGRESSION |
| pre-probe build | resident | +0.054% | 2.201% | 4/9 | NO DIFFERENCE |
| final build | 16 MiB | -4.864% | 4.125% | 2/9 | NO DIFFERENCE (spread) |
| final build | 32 MiB | **-5.491%** | 1.924% | 1/9 | REGRESSION |
| final build | resident | +0.140% | 3.038% | 4/9 | NO DIFFERENCE |

Across all six runs the sign is never positive where it matters: staging was faster
in 0, 0, 4, 2, 1 and 4 of 9 pairs. Two runs clear the noise as a regression, none
clears it as a win.

**Correctness, not performance, is what makes the switch legal to leave in the
tree:** every arm above produced stdout md5 `28fa19c3733a5cab9dc53611de71600c`, and
the promotion count is identical in both arms (1846), so staging changes neither the
expert *set* nor the accumulation order nor the bytes moved.

## The allocator row was a pending-event query, not the allocator

The finding above ranked `alloc_pooled` second at 27-34% of the decode step, and
the obvious reading was the allocator: 666-1012 `alloc` against 4526-4872 `reuse`,
with `hipMalloc` behind one of them. `kraken-bench` now prints the marginal cost of
the pool's own primitives, and it says otherwise:

| primitive (marginal, gfx1201) | cost |
|---|---|
| event create + destroy | 0.13-0.23 us |
| record on the default stream | 22.7-30.6 us |
| query an event that has completed | 0.03-0.05 us |
| wait on an event that has completed | 0.08-0.13 us |
| **record, then query it while pending** | **+46.9 us over the record** |
| the real alloc+release pair | **86.97 / 79.47 us** (was) |

The pair cost 87 us while its parts summed to ~31, and the missing 56 us is the
query: asking the device whether a cell's reader had retired costs a full host
round trip when the answer is "not yet", and on a budget that forces a promotion
per expert visit the answer is always "not yet". That closes the engine's own row
exactly -- 5560 cells x 87 us = 484 ms against the measured `alloc 481.8 ms`.

**The fix is to stop asking.** The cell does not need the host to have observed the
reader finish; it needs its next *writer* ordered after the reader, and that is a
queue-side dependency: the released cell carries the mark it was released under,
`alloc_pooled` hands it out with that mark attached, and the next write into it
(`upload_paged_batch`, `upload_expert_staged`) issues `hipStreamWaitEvent` on the
copy stream. Waiting on an already-complete event retires immediately, so removing
the query cannot make the device slower -- only the host cheaper. Writers with no
single stream to order (the cold path's pull ring) still wait, and writers on the
default stream (a kernel, `hipMemset`) need no wait at all because the mark was
recorded there; the three cases are counted separately as `order_device`,
`order_host` and `order_implicit`.

Measured with the two arms interleaved on one binary (`KRK_POOL_HOST_WAIT=1`
restores the query + host wait):

| | old | new |
|---|---|---|
| pool round trip, median of 3 interleaved reps | 87.48 us ready / 88.42 us reader | **24.37 us / 33.69 us** |
| engine `alloc` row, 16 MiB | 618.4 ms | **182.5 ms** |
| engine promote total, 16 MiB | 1627.2 ms | 1231.1 ms |
| engine decode, 16 MiB | 15.1 tok/s (66.3 ms/step) | **16.6 tok/s (60.3 ms/step)** |

| Budget | order-balanced delta (B = new) | MAD | pairs B faster | verdict |
|---|---|---|---|---|
| 16 MiB | **+11.279%** | 2.815% | 9/9 | WIN |
| 32 MiB | **+8.569%** | 5.120% | 9/9 | WIN |
| resident | -0.385% | 1.824% | 4/9 | NO DIFFERENCE (the null control: no reuse happens) |

Text md5 `28fa19c3733a5cab9dc53611de71600c` in every arm, and the promotion count is
identical (1846), so this is a scheduling change and not a different computation.

### The release mark is shared per burst now, and the row it lives in moved a third

An eviction returns an expert's three slices in one call, and the three records
were issued at one stream point: nothing is submitted to the default stream
between them, so one record expresses all three. `release_pooled_batch` does
that, with a refcounted `PoolMark` per burst, and `KRK_POOL_BURST=0` restores the
per-slice path.

| budget | `room` ms (old -> new) | order-balanced | pairs | marks old -> new |
|---|---|---|---|---|
| 16 MiB | 58.5 -> 39.7 | **-31.23%** (MAD 1.87%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |
| 32 MiB | 59.0 -> 40.6 | **-31.28%** (MAD 3.19%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |

Decode is **NO DIFFERENCE** at both budgets (+1.287%, +2.133%; MAD 1.6-2.1%), and
the stdout md5 is unchanged in every arm.

**The ~128 ms this document used to leave as the next term was a quiet-queue
estimate.** `kraken-bench` measures a record at 29-31 us on an idle queue; in a
busy decode step 7252 records removed bought 18.6 ms, i.e. **~2.5 us per record**.
The whole row at a constrained budget is ~13 ms of a ~1.7 s run, not ~128 ms, so
this change is a cleanup worth ~0.5% and is recorded as one -- the same
quiet-vs-busy gap that AGENTS.md records for the launch floor.

## Gates on the final binary

| Gate | Result |
|---|---|
| `scripts/build_check.sh` | rc=0, 0 failed; section 6 re-recorded deliberately with `KRK_REFRESH_FINGERPRINT=1` after the rebuild, then passed plainly. One standing WARN: the loaded HIP runtime is not the build SDK |
| ninja, six targets | rc=0, 0 `error:` lines, `ninja -n` reports no pending work |
| `kraken-tests` | 2927/2927 |
| `scripts/kraken_tests_baseline_check.sh` | no regression; 1 changed count (`test_expert_cache_warm_tier` 56/56, was 34/34), 17/17 selfcheck, 18/18 tool selftest |
| `kraken-bench --gate` | rc=0, 53 checks: the `expert-pool` device gate and the new `expert-lease` device gate (18 checks), the latter falsified on purpose (below) |
| `scripts/coherence_check.sh` (3 models) | rc=0, 3 coherent, 0 not |

The `expert-lease` gate is the device half of the invariant the WARM-tier unit
test can only reach through the CPU stub: a budget between two and three slots so a
concurrent staged promotion *must* evict, and then the *device bytes* of the leased
expert -- downloaded from the addresses it was already using and compared with the
source it was promoted from -- plus the staged expert's own bytes, so an aliased
promotion cannot pass by luck. It also asserts the mechanism and not only the data
(the reused cells' guards were honored on the device), and a one-slot budget must
refuse to stage and fall back to exact demand.

**It was shown able to fail rather than assumed so, and the first two versions were
not.** Deleting the lease from `pick_victim` and rebuilding the bench turns the gate
red -- rc=1, `the leased expert is still resident` and `a concurrent staged
promotion did not overwrite the cells the leased expert is using` FAILing, with a
diagnostic naming `gate slice of expert 0` -- where the earlier versions passed with
the lease deleted. Two defects came out of that, and both are the same shape as the
findings elsewhere in this document (a check that passes for a reason unrelated to
what it claims):

* **A budget with room for both experts never makes the promotion contend.** The
  byte checks then pass because no cell is reused at all; a budget of exactly three
  slots is the same trap from the other side, because the arithmetic lands on the
  boundary and `over_budget` is false. The gate now asserts non-vacuity outright:
  `the staged promotion had to evict a slot to fit` fails if the budget ever stops
  forcing contention.
* **Comparing against the pointer the first `acquire` returned is a tautology,**
  because that points into the cache's own slot and reads the *new* pointers after a
  re-promotion -- and a lease violation IS a re-promotion, which restores the right
  bytes through the wrong addresses. The gate now snapshots the three device
  addresses and reads those addresses, which no re-promotion can launder.

This is why the tool carries a falsifiability step at all: the gate's own passing is
not evidence, and the two-arm run that is evidence is the one where the invariant is
broken on purpose.

The device gate in `kraken-bench` is the part a unit test cannot reach: it drives
`alloc_pooled` to the physical limit, proves a released cell's queued default-stream
reader finishes before reuse, proves a staged ticket publishes exact bytes, proves a
budget shrink trims every free cell and never frees a live one, proves
`KRK_TEST_POOL_NO_EVENT=1` (a constructed event-creation failure) is contained, and
proves a target + draft owner pair composes to the sum of their budgets.

## Limitations, stated rather than smoothed over

* One model, one card, one prompt, `n=32`. The per-expert numbers are for 5.45 MiB
  experts at 16/32 MiB budgets; they do not transfer to a corpus that fits.
* The 16 MiB run on the final binary is **NO DIFFERENCE by the harness's own rule**
  (MAD 4.1% > the -4.9% median). The honest summary is "no win, and every
  constrained run's sign is negative", not "a measured 5% regression".
* The overlap probe is one run per arm per build (868 samples each). It reproduced
  within ~8% across two builds, which is enough to rank copy vs window, not enough
  to claim a precise overlap percentage.
* `active_stage`/`stage-timing` counters are per-process and reset by
  `reset_counters()`; they are not a per-layer timeline.

## Next actions, ranked by the same measurements

1. **The release mark is closed** (this pass). One mark per eviction burst, the
   `room` row down a third (9/9 pairs at 16 and 32 MiB), decode NO DIFFERENCE
   because the row is ~18 ms of a ~3.2 s run -- and the ~128 ms it was ranked on
   was a quiet-queue estimate (~2.5 us marginal in the step). See
   `docs/traces/expert-pool-burst-marks-2026-10-11.txt`. What is left in this row
   is the record at all: a cell handed out and released with nothing submitted to
   the default stream in between needs no mark, but in a decode step hundreds of
   launches separate the two, so that case is rare by construction.
2. **One plan upload per layer, not one per expert visit.** The per-expert
   `upload_i32` + `upload` pair (per AGENTS.md, 1920 of 2965 ops on the 35B model)
   is the same shape of fold as the ones already landed, and it is independent of
   the promotion overlap this pass closed.
3. **Raise the budget past one layer's routed set (21.8 MiB here) before touching
   policy.** 1.6% HOT at 32 MiB is the measurement that says the budget, not the
   prefetcher, is what is being tuned -- and it also separates "the tier is too
   small" from "the tier is badly managed".
4. **Per-layer active-core protection with hysteresis** (priority 6 of the cache
   audit) is the policy change that could matter at these budgets; it invalidates
   none of the above and needs its own interleaved A/B at the same md5 contract.
5. Staging is kept for its *memory* shape, not its speed: it is the only path that
   copies on a nonblocking stream with an explicit publication ticket, which is a
   precondition for any future work that stages **across** layers (where a real
   compute window exists) rather than one expert ahead. Do not re-enable it by
   default on the strength of this pass.
