# REAP-288 / REAP-320 decode: fault, trace and fixes

Date: 2026-10-21.  Card: RX 9070 XT (gfx1201, 15.9 GiB, runtime reports a 256 MiB
BAR aperture).  Host: 95.9 GiB installed, ~81 GiB free.

Files: `H:/OLLAMA-Models/GGUF/qwen3.8-flash-next-reap-288-Q4_K_M.gguf` (83.8 GB,
288 experts) and `.../Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf` (67 GB, 320
experts).  Arch `qwen4exp`, gated behind `KRK_ARCH_ALLOW_EXPERIMENTAL=1`.

## 1. Symptom

`kraken -m <REAP-288> -p "The capital of France is" -n 32 --greedy` printed a few
tokens and died:

```
[error] hipErrorInvalidValue at src/hip/backend_hip.hip:1432
        (upload)
```

The named line is `hipMemcpy(dst + off, src, bytes, HostToDevice)` inside
`Backend::upload`.  The error explains nothing: a H2D copy is invalid only when a
pointer is (or the size underflows).

## 2. Execution map of one routed expert

```
Engine::moe_forward (engine.cpp ~6650)
  └─ ExpertCache::acquire(src, layer, e)                 src/expert_cache.cpp ~1000
       ├─ be->set_expert_pool_owner(this, owner_bytes(need))
       ├─ hit?            touch(); return &s.m
       ├─ WARM hit?       promote_to_vram(s)             alloc_pooled x3 -> upload(s)
       ├─ WARM admit      warm_fill()  (read() into pageable host memory)
       ├─ make_vram_room(need)   -- evicts while bytes_+need > budget_
       ├─ alloc_pooled(per) for gate/up/down             <- CAN RETURN nullptr
       ├─ if WARM copy:   upload_paged_batch / upload_pinned_batch
       └─ else:           be->upload(dsts[i]->data, mapping + e*per, per)
                                  ^ dst may be nullptr  <-- THE FAULT
```

`Backend::alloc_pooled` (src/hip/backend_hip.hip ~561) does **not** abort when it
refuses a request.  On the grow path it tests

```
free_now < rounded + vram_floor_bytes(total)   ->   return nullptr
```

and prints one warning ("expert pool: holding at ...").  That is by design: it
will not cross the VRAM floor.  The COLD path in `acquire()` was the only caller
that treated the answer as a valid pointer.

Reproduced with the warning and the fault in the same run:

```
[warn ] expert pool: holding at 7060 MiB with 3252 MiB free beside a 3261 MiB floor
[error] hipErrorInvalidValue at src/hip/backend_hip.hip:1432 (upload)
```

Root cause: **a refused device allocation was dereferenced.**
`hipMemcpy(nullptr, src, n, HostToDevice)` -> `hipErrorInvalidValue`.  The fault
was attributed to the upload line, so the report named a layer that was fine.

Why the pool had no room while the cache's own ledger said it did:

* `budget_` (7624 MiB) is planned at init and re-clamped only on a *shrink*; by
  the time decode runs, the KV cache and workspaces exist and the live room
  beside the floor is negative (3202 free, floor 3261).
* `make_vram_room()` tests `bytes_ + need > budget_` (7445 + 2.2 < 7624), so it
  evicted nothing, and the allocation then failed on the floor test.
* The pool holds no free cell of that size class either -- every cell is live.

## 3. Fixes applied

### 3.1 Never use a refused allocation (`src/expert_cache.cpp`, `acquire`)

Per slice: retry the allocation, and on refusal give the pool back cells this
cache already owns:

```cpp
for (int tries = 0;; tries++) {
    cell = be_->alloc_pooled(per);
    if (cell) break;
    if (tries >= 63 || !force_reclaim(per)) break;
}
```

`ExpertCache::force_reclaim(want)` retires a victim (LRU order, preferring a slot
whose slice size equals `want` -- cells are handed out by rounded size, so a
victim of another class is useless).  A retired slot keeps its WARM copy, so the
next acquire of that expert is a promotion, not a file read.

If no cell can be produced the slot is **not** inserted, `bytes_`, `vram_slots_`
and `loads_` are not touched, and `acquire()` returns nullptr -- which
`moe_forward` already handles (`if (!re) continue;`).  Before this, a failed load
still counted its bytes as resident and inserted a half-built slot with null
slices.

`promote_to_vram()` got the same retry (it already refused a null cell, but fell
back to a file read for an expert whose bytes were in RAM).

### 3.2 The WARM tier was silently disabled for every big model (`src/engine.cpp`)

`host_stage_room` subtracted the **whole mapped corpus** from live free RAM:

```cpp
live_avail - host_reserve - mapped_share        // mapped_share = min(mapped, ram_total)
```

For a corpus bigger than RAM this is always <= 0, i.e. exactly the models the tier
exists for.  `warm_room` subtracted it a second time, and the "tier is disabled"
warning was gated on `host_budget > 0`, which was already 0 -- so nothing was
printed at all.

A clean file-mapping page is reclaimable standby memory: the OS drops it without a
writeback.  It is not memory this process owns, and charging it twice removes the
tier.  The real guard is per-admission (`SharedHostRamBudget::can_admit`, which
samples live available RAM and fails closed), so the plan no longer needs to
pre-charge the corpus.

Measured on the REAP-320 file, 16 tokens, greedy, 3 interleaved pairs,
`--expert-warm-mb 0` against the new default (text identical):

| arm | ms/step | tok/s | file read |
|---|---|---|---|
| WARM off | 205.5 / 206.9 / 208.0 | 4.8 | 387.23 MiB/tok |
| WARM on (new default) | 140.5 / 141.5 / 141.9 | 7.1 | 26.99 MiB/tok |

**+46% decode, 14x less file traffic.**  On the 288 file: 630.3 -> 283.0 ms/step
(**2.2x**, 901.67 -> 244.40 MiB/tok).

WARM sizing on the same box: budget 40.6 GiB -> tier 31.5 GiB, 14 631 slots,
1.5% COLD misses (was 47.5%), 44.1% WARM hits.  Warmup reads the tier once:
17.8 s for 31.5 GiB.

### 3.3 The aperture is reported, no longer just printed

`DeviceCaps::large_bar` now carries `hipDeviceAttributeIsLargeBar`, and the host
policy logs the actionable line once when the runtime says 256 MiB.  See §5.1.

## 4. Sherlock trace (`--profile`, REAP-320, 8 tokens)

Profiled absolute times are inflated ~10x by the recorder floor and by
serialising the queue; counts and shares are exact.

```
component           ops   %dev      dev us     idle us     host us
transfer            305  34.2%    73614.50    43104.69   298602.90
projections        1613  26.0%    56027.22    34737.49     5460.50
other               867  10.3%    22097.64    21692.35     2861.00
ffn-activate        626   8.5%    18379.32    13413.48     1882.20
qsa                 480   7.7%    16466.21     9840.59     1555.20
attention-mix       505   5.4%    11683.34     8893.20     1789.40
hyper               194   2.3%     4905.27     3898.47      571.40
norms               136   1.7%     3578.59     4057.31      464.80
recurrent           180   1.4%     3089.85     3516.93      640.60
residual            110   1.4%     2911.96     2466.07      303.40
attention            48   0.7%     1482.80      985.13      296.00
bias                 36   0.4%      799.75      739.81      172.50
ple                   3   0.1%      119.32       96.62        9.70
```

`KRK_TIME` per-op, net of the floor: `upload` 8398 calls / **3620 ms NET** versus
`gemm(gemv)` 8401 calls / 135 ms and `gemm(gemv_m)` 2192 / 160 ms.  The expert
fetch is ~9x the arithmetic, i.e. decode on these files is a *weight-fetch*
problem, not a GEMM problem.  That is the number the WARM fix attacks: 8398
uploads per 8 tokens (every pick cold) became 1.5% misses.

After the fix the expert-path table reads:

```
decode: read 42.3 | room 0.1 | alloc 28.8 | xfer 894.0 | promote 1056.9 ms
```

### Blind spot found while tracing

With WARM disabled, the COLD path calls `be->upload()` *inside* the acquire loop,
outside every timer the expert-path table reads.  A run that moved 387 MiB/token
from the file reported `read 0.0 ms | xfer 0.5 ms | alloc 656.7 ms`.  The
`alloc` figure is real (see §5.2) but the dominant cost was invisible.  Any future
`WARM=0` measurement must be read from `[stats ] transfer ... read from the file`,
not from the expert-path row.

## 5. Remaining defects, ranked

### 5.1 Promotion is a bounce-buffer copy on a 256 MiB aperture

`hipDeviceAttributeIsLargeBar` is 0 here.  With a small aperture a pageable H2D
copy goes through the driver's window; a **pinned** copy is a DMA straight out of
RAM.  Measured, same file, same 16 tokens, 3 interleaved pairs:
`KRK_HOST_PIN_MB=16000` halves the transfer (880/874/884 ms -> 400/565/367 ms) and
moves the step 136-143 -> 114-140 ms (~+12%).  Default is off because pinning locks
RAM machine-wide; §3.3 now names the switch in every run that would benefit.

Action: make a *bounded* pinned prefix (min(WARM/4, 8 GiB)) the default when
`large_bar == 0`, with the measured GB/s recorded.  Not done here: it changes host
commitment on every model and wants a 16 GiB-class validation first.

### 5.2 `alloc_pooled` costs ~200 us per 2.2 MiB cell

`[stats ] expert-path alloc` was 656.7 ms of a 4.5 s run (10 038 allocations) and
is still 28.8 ms for the 128 misses left after the WARM fix.  That is `hipMalloc`
per cell, not a suballocator.  A per-size-class arena (one `hipMalloc` per growth
step, cells served from a free list) would remove ~15% of the constrained run and
essentially all of it at 1.5% miss.

### 5.3 A skipped expert is a silent numerical change

`acquire()` returning nullptr makes `moe_forward` drop that expert's contribution
for the token.  It is strictly better than a context-destroying `hipMemcpy`, and
after §3.1 it only happens when every resident slot is leased or of the wrong size
class.  It should still be counted per layer and reported in `[stats ] cache`, so
"the text changed" has a witness.  `pool_refusals_` exists for that and is not yet
surfaced.

### 5.4 `budget_` is a plan, and the live card disagrees with it

The init-time budget survived two re-clamps (10418 -> 9030 -> 7696 -> 7056) and is
still allowed to exceed the live free-beside-floor room, which is why the floor
test -- not the budget -- is what refuses allocations.  `force_reclaim` makes that
survivable; the cleaner fix is to evict on the live sample
(`device_free_bytes() - floor`) at the top of `make_vram_room`, so a miss evicts
*before* the allocation fails.

### 5.5 The routed set is 42.2 GiB against a 10 GiB device budget

The cache warns about this at load.  With 26.8% of picks missing even at 7.4 GiB
resident, the remaining levers are: expert *ranking* (`.krakenexperts.json`, LFU
already in place), fewer bytes per expert per token, or a larger WARM tier.  On a
16/24/32 GiB box the tier is ~3/6/9 GiB -- ~7% of the routed set -- so the disk
read stays the floor: the honest ordering there is 2-5 tok/s, not 30.

## 6. Verification

* `kraken-tests`: **3415/3415, rc=0** (twice: after the fault fix and after the
  policy change).
* `scripts/kraken_tests_baseline_check.sh`: rc=0, `no regression`, reporting proof
  17/17; one count note (`test_expert_cache_byte_budget` 47 -> 49 checks).
* `scripts/build_check.sh`: rc=0 (fingerprint re-recorded after the rebuild).
* `kraken-bench --gate`: rc=0.
* Coherence, device arm against `--cpu` reference, identical text:
  Qwen3-4B-Q8_0 PASS, Qwen3.5-9B-Q4_K_M PASS, Qwen3-30B-A3B-Q2_K PASS.
* REAP-288: 32 tokens, rc=0, 2.3 tok/s before the WARM fix and 3.5 tok/s after
  (16-token arm); REAP-320: 24/16 tokens rc=0, 7.1 tok/s.  Both generate
  "The capital of France is Paris." instead of dying at token 5.

## 7. Next-layer prefetch: the hypothesis, the measurement, and why it loses

Goal: hide the expert fetch instead of paying it. Mechanism shipped (opt-in):

1. `Engine::moe_ffn` records the routed set this layer chose on the **previous
   token** (`moe_hint_`) -- routing is sticky in time and no other predictor
   exists before the row that would produce the real set.
2. After that layer's expert loop, the engine calls
   `ExpertCache::stage_ahead(next_layer, hint)`, which promotes the predicted
   set one layer early (the next layer's norm/attention/hyper work is the window).
3. `Backend::upload_expert_prefetch` DMAs pinned WARM straight to VRAM on the
   private copy stream and returns a **ticket**, without ordering the default
   stream. `wait_marker_stream` orders the default stream at consumption, so the
   host never blocks (the ring path's ticket is host-synchronised).
4. `stage_ahead` retires slots the finished layer no longer needs, because the
   byte budget is full by construction on a constrained card, and it `lease`s
   every slot of the target layer for the pass.

Measured on REAP-320, 16 tokens, greedy, `KRK_HOST_PIN_MB=16000`, 3 interleaved
pairs, text identical between arms:

| arm | ms/step | mean |
|---|---|---|
| prefetch off | 109.3 / 126.0 / 107.8 | 114.4 |
| prefetch on | 121.4 / 125.6 / 127.1 | 124.7 |

**-9%, so the default is OFF** (`KRK_MOE_NEXT_PREFETCH=1` enables it).

The first version measured -15%, and the cause was a real defect this exposed:
`force_reclaim` prefers a victim of the same size class, and *every expert of one
layer shares that layer's size classes*, so the pass evicted the very set it was
completing and re-promoted it next token. Leasing the target layer's resident
slots fixed that and recovered 6 of the 15 points; the remaining loss is
structural.

Why it cannot win at this geometry: promotion is **bandwidth**, not latency.
About 240 experts of ~2.2 MiB enter VRAM per token (the 42.2 GiB routed set
against a 7.4 GiB device budget), the copy stream already overlaps the kernels
queued before it, and a prediction that misses adds a promotion the real routing
does not use. Issuing earlier cannot create bandwidth -- only a smaller
working set or fewer bytes per expert can.

Two more arms, both measured, both rejected as defaults:

| arm | effect |
|---|---|
| `KRK_VRAM_FLOOR_MB=1200` (more of the card for experts) | residency 55.7% -> 59.6% HOT, but 137 -> 155 ms/step (2 pairs each) |
| `KRK_HOST_PIN_MB=16000` (pinned WARM, DMA promotions) | xfer 880 -> 400 ms per 16 tokens; 136-143 -> 107-127 ms/step |

So the fastest measured REAP-320 decode configuration is **default WARM policy +
pinned WARM prefix** (~107-127 ms/step, 8-9 tok/s), and the next real lever is
`alloc_pooled`'s missing arena (§5.2), not scheduling.

## 8. The size-class arena: alloc gets 5.8x cheaper and decode gets slower

The alloc row was the next target after §7 (§5.2): `hipMalloc` per ~2.2 MiB cell.
Implementation (opt-in, `KRK_EXPERT_ARENA=1`): a slab is `kSlabCells` (32) cells
of one size class cut from a single `hipMalloc`, the free list unchanged on top,
`live` per slab so a whole slab can be returned to the driver once every cell in
it is free, and trim at slab granularity (a cell has no allocation header of its
own). The arena switch now selects only the GROWTH allocator -- the first version
conflated it with "cache released cells at all", which broke the bench's reuse
checks the moment the slab became opt-in.

REAP-320, 16 tokens, greedy, 3 interleaved pairs, identical text:

| arm | alloc row | pool cells | ms/step | mean |
|---|---|---|---|---|
| per-cell (default) | 150.6 / 145.0 / 144.3 ms | 13 700 alloc, 4 400 reuse, 3 500 trim | 112.8 / 117.1 / 112.7 | **114.2** |
| slab arena | 26.5 / 24.5 / 24.6 ms | 12 140 alloc, 11 697 reuse, 0 trim | 139.3 / 135.5 / 133.6 | **136.1** |

The alloc row does exactly what it was meant to -- **5.8x cheaper, 121 ms saved
over 16 tokens (7.6 ms/step)** -- and the step is 22 ms/step WORSE, so the arena
stays opt-in.

The loss is in the copies, not the kernels: the arena arm's `xfer` ran 1 486 ms
against 894 ms for the same bytes and the same expert set. The first hypothesis
was destination-address aliasing from uniform cell spacing (every cell 2.2 MiB
apart inside one 70 MiB allocation), so a 64 KiB pad between cells was measured:
**205 ms/step**, worse still, with `reuse` at 50 190 because the pad costs 3% of
the tier's capacity and this pool sits on a cliff -- 3% is the difference between
holding the routed set and evicting it every token.

What the numbers do say, and what decides the default: **the alloc row is ~7% of
the step, so even a 6x cheaper allocator cannot pay for a 20% copy regression.**
The remaining cost of recycling VRAM is address locality, which no allocator
front-end fixes. Default stays per-cell growth + free-list reuse (the
configuration that measures best), and `KRK_EXPERT_ARENA=1` is the arm to retest
if the locality cause is ever pinned down.

## 9. Batching a layer's promotions into one upload: measured, no win

The hypothesis came from the `us/expert` figure on the transfer line: at 143
us/expert on the 256-expert 35B-A3B it looked like per-call cost, and a decode
step takes ~240 promotions of 2.2 MiB, so folding 3k enqueues into a handful
looked like free time.

`ExpertCache::promote_batch` implements the fold: make room and allocate cells
per expert (each slot has its own size class), then issue **one** upload per
source kind for the whole routed set -- pinned sources and pageable ones in
separate batches, because a batch cannot mix copy paths. It is called from
`Engine::moe_ffn` right after `prefetch_layer`, bounded to `n <= 8` and
`prefetch_ids_.size() <= k*4+8` like `stage_batch`. `KRK_MOE_BATCH_PROMOTE=1`
enables it; it is **off by default** because it does not pay.

REAP-320 (`Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf`, 16 greedy tokens, 5
interleaved order-balanced pairs, identical text in all ten runs):

| arm | median ms/step | `xfer` (decode, 15 steps) | us/expert |
|---|---|---|---|
| per-expert (default) | **112.9** | 249-287 ms | ~500 |
| one batch per layer | 111.8 | 366-385 ms | ~560 |

The paired deltas (+1.0, -9.6, -2.1, +3.1, -1.8 ms) do not clear their own MAD
(2.6 ms) in 4 of 5 pairs. The `xfer` row is consistently *worse* for the batched
arm and `room` is consistently better, which is the shape of a wash: the room for
the whole set is made up front, so evictions happen earlier.

**Why the enqueue was never the cost.** A promotion on this file measures
~500 us/expert, not 143. The pageable path stages each 0.73 MiB slice through a
4 MiB pinned ring slot: one host `memcpy` into the slot plus one `hipMemcpyAsync`
out of it, at the box's ~4.6 GB/s bounce-buffer rate. That is ~300 us per slice
of pure copying, and it is the same number of bytes however the calls are
grouped. Worse, the ring is only `kPagedBufs = 4` slots: three slices of one
expert fit exactly, so the per-expert call already has every copy in flight at
once, while a 4-expert batch (12 chunks) has to recycle slots and `paged_take_`
ends up spinning on events -- which is the measured `xfer` regression.

**A real bug fell out of it.** `upload_paged_batch` recorded at most
`kPagedBufs` slots in its `used[]` array and published only those to the default
stream, so a batch with more chunks than the ring had *unordered copies*: the
consumer GEMM could start before the DMA landed. Per-expert promotions (3 chunks)
never reached the cap, which is why it survived. Both the single and batch paths
now publish in ring-sized groups as the array fills. That fix is kept; the batch
arm is kept as the vehicle for a deeper ring or a pinned WARM tier, where the
per-call cost is not already hidden behind a memcpy.

What the measurement does say about the next lever: `xfer` is 250 ms of a
1.7 s decode (15 steps), and `room` is 340 ms -- making room costs more than
moving the bytes. That is the ranking in `REVIEW-EXPERT-PATH.md` §5, not a call
count.

## 10. The victim search was the room row, and it was scanning the whole cache

§9 left the ranking as "`room` (340 ms) costs more than `xfer` (250 ms) in 15
decode steps". Splitting `room` in two says which half:

```
evict  2624 scan(s), 9191522 slot visits, 505.9 ms scanning + 18.1 ms releasing
       = 55 ns/visit, 3503 visits/scan
```

**96% of the cost is choosing the victim, not evicting it.** `pick_victim`
walked every entry of `slots_` for every eviction, and `slots_` is the whole
tier: ~3.5k resident experts inside ~14k WARM entries (the WARM shadow outnumbers
the resident set 4:1 on this file). Each visit also did a second hash lookup
(`slots_.at(best)`) to compare against the running best. An eviction is O(1) work
at O(tier) cost, per expert, per token.

### The fix: an index of residents, and a moving window

`vram_keys_` holds the keys of **resident** slots only, with `Slot::vram_idx` for
O(1) swap-remove (three promotion sites call `track_vram`, `retire()` calls
`untrack_vram`). `pick_victim` then considers `KRK_EVICT_WINDOW` candidates from
a hand that advances past the slot it took, keeping the old order -- unpinned
before pinned, then rarest, then oldest. A lease that fills the whole window
falls back to the full map once, so an eviction is never refused for lack of a
candidate.

REAP-320, 16 greedy tokens, same text, ms/step:

| window | 1 | 2 | 4 | 8 | **16** | 32 | 64 | whole set |
|---|---|---|---|---|---|---|---|---|
| ms/step | 103.3 | 99.4 | 97.2 | 96.8 | **96.6** | 99.3 | 108.7 | 107.9 |

Wider is *not* better: a 64-slot window reaches freshly touched experts and
evicts them, while a narrow one evicts around the hand, which is where the
coldest slots are. 8 and 16 tie; 16 is the default.

Five interleaved order-balanced pairs, `KRK_EVICT_WINDOW=16` against `=0` (the
same index, the old global policy): **96.6 against 107.9 ms/step median, every
pair favoring the window** (+11.1 / +9.7 / +10.0 / +11.7 / +11.9 ms), and the
windowed arm also promoted *less* (2446-2456 against 2589-2591) -- 3-4% fewer
bytes over the link, not just less CPU. Against the pre-index binary (112.9
ms/step in §9) that is **+14.4% decode, 8.8 -> 10.4 tok/s**; scans went from
505.9 ms to 2.2 ms and 16 visits per eviction.

Second file, REAP-288 (the tier churns: 1128 admissions, 1083 WARM evictions):
259.8 against 266.8 ms/step, 4101 against 4352 promotions.

### The trap: the index is an invariant of residency, not of the statistics

`reset_counters()` runs after warmup and deliberately keeps residency (the
warmup's product). The first version of this patch cleared `vram_keys_` there
too, because it looked like a counter. The result was not a crash and not a
counter that looked wrong: `pick_victim` saw **one key for 3484 resident slots**,
answered from a corner of the set, and the run promoted **2589 -> 6754 experts**
(2.6x the bytes; `xfer` 578 -> 3537 ms) while every other line read normal. The
diagnosis was one line of output (`index 1 keys, 3484 resident`), so that line
stays: `evict ... | index %zu/%zu keys/resident` prints both numbers, and
`test_expert_cache_byte_budget` arm 4 pins them across a `reset_counters()`
(fails 61/64 with the clear put back).

## 11. Where the remaining 87-96 ms/step goes: a per-layer host round trip

With the eviction scan fixed (§10) and the batched-upload arm off (§9), the
REAP-320 decode is 95.9-100 ms/step pageable and **86.9 ms/step with the pinned
WARM prefix**. The host phase table (`KRK_TIME_HOST=1`, which enables the host
profiler *without* the device recorder -- the device one adds ~22 us per op and
would inflate exactly the span that waits on the device) accounts for all of it:

```
phase              calls   ms/call   per step (48 layers)
moe.route            720     1.689   81.1  ms   of which
  moe.route.fetch    720     0.868   41.7  ms   the blocking router readback
  moe.route.rest     720     0.820   39.4  ms     of which
    moe.experts      720     0.703   33.7  ms     the expert loop (host)
```

`decode.step` 81.5 ms, `decode.forward` 85.6 ms: 48 layers x 1.69 ms. **The whole
step is two things: a blocking router readback per layer (48% of it) and the
host's expert loop (39%).** Everything else -- attention, QSA, hyper, norms, PLE,
the dense GEMVs -- is the remaining ~11 ms.

### The kernels are not the problem, and neither is bandwidth

- `KRK_TIME=gemm` on an unprofiled run: `gemm(gemv)` 8401 calls, 22.3 ms NET for
  16.6 GB = **743 GB/s**, 2.65 us a call. The dense GEMV total is ~2.8 ms/step.
- `kraken-bench` at the file's own shapes: 580 GB/s (2560x10240 Q5_K), 500
  (2560x6144), 379 (the 437 MB LM head), 421 (Q8_0 shexp).
- The same engine on Qwen3-4B-Q8_0 does **464 GB/s** and 9.2 ms/step for 3.98 GiB
  of weights. REAP-320 moves ~2.9 GB/step in 96 ms = 30 GB/s. The difference is
  not the kernels; it is the schedule.
- Launch cost is real but small: `KRK_MOE_GROUP=0` adds ~1032 ops/step and costs
  +2.4 ms/step (3 pairs) = **2.3 us per op**; at 5012 ops/step that is ~11.5 ms,
  which is exactly the residual line above.

### The round trip, proven

`KRK_ROUTER_2X=1` issues a second readback of the same 320 floats right after
the first. Unprofiled, it costs **0.093 ms a call** -- while the first costs
**0.950 ms**. So 0.857 ms/layer is not the copy: it is the host waiting for the
device to drain the work queued in front of it (the layer's attention, QSA and
hyper ops), and that drain is work the step needed anyway. What it *does* prove
is what follows it: **the second copy finds the queue empty**, so at that point
the device has nothing left, and the host's plan build runs with the card idle
(0.13 ms/layer = 6.3 ms/step, found as `route.rest` minus `experts`).

The strict ping-pong is the real shape: device 0.78 ms, then host 0.8 ms, per
layer, alternating, because the plan cannot be built until the row comes back
and the next layer cannot start until the plan exists. Add the two and it is the
measured 1.69 ms/layer. The floor for the readback alone (48 x 93 us) is 4.5
ms/step, and removing the whole round trip is what the 41.7 ms/step is worth.

### The pinned prefix is automatic now, and +12.5% was the smallest arm

`Engine::init` decides it by **measurement**. The 256 MiB aperture is a firmware
setting, not a card constant -- Above 4G decoding + Resizable BAR changes it,
WDDM has been seen reporting `hipDeviceAttributeIsLargeBar` as 0 on a box whose
firmware had ReBAR on, and this engine's pageable path stages through its ring
whatever the aperture is -- so the runtime bit cannot decide anything on its
own. The probe times the two promotion paths at the shape a promotion issues
(16 MiB, three arms, one untimed call each then five interleaved
order-balanced triples with a median per arm, every copy ending in `sync()`
because all three return before the copy lands): `upload_paged_batch` (ring)
against the direct pageable arm and `upload_pinned_batch`. Pinned >= 1.20x pageable means the pageable path pays a
copy the pin removes. Here that reads **10.1 against 18.6 GB/s (1.84x)**, and
the log prints both rates in the line that says what it decided.

Two details of that probe are measurement lessons rather than tuning. The first
is the size: on this file the same comparison reads **1.60x at 16 MiB** and
**1.41-1.69x at 4 MiB** across runs, because at 4 MiB the sync and the launch
are a large share of each arm and compress the difference the pin is worth, so
the probe's floor is 16 MiB (`model_.max_expert_bytes()` clamped to 16-64 MiB).
The second is the line: **1.20x**, deliberately below every ratio measured here
(1.60x on this file, 1.84x on a 35B-A3B, 2-3x on a 120B MoE's promotion shape)
and above the ~1.0-1.1x an aperture with no host copy in the pageable path would
give -- over-pinning costs unswappable RAM and nothing else, while under-pinning
leaves 10-18% of decode on the table. `kraken-bench --h2d` is the same two paths
without a model: it times 12.6 MB, one expert's three slices, 128 MB and a
36-slice batch, and is what the 22.7-against-7.6 GB/s figure above came from.

The size is `krk::pinned_warm_prefix_bytes(warm, ram_total)`
(`include/krk/vram_guard.hpp`, so the suite can drive it without a device): an
eighth of installed RAM -- 12 GiB on this 96 GiB box -- capped at 16 GiB and at
half the tier, so a 16 GiB laptop with a 16 GiB tier pins 2 GiB rather than half
its memory. `KRK_HOST_PIN_MB` / `--expert-pin-mb` override it, including `0`;
`KRK_PIN_AUTO=0` is the A/B arm that restores the pageable path and the log then
says what was left on the table, and `KRK_H2D_PROBE=0` falls back to the
runtime's BAR hint when the probe itself is what you are testing.

Cross-model, 16 greedy tokens, interleaved order-balanced pairs, text
byte-identical in every arm:

| model | shape | WARM (pin) | pageable | auto | verdict |
|---|---|---|---|---|---|
| REAP-320 Q4_K_M | 320 experts, 42 GiB routed | 31.5 GiB (12.3) | 114.4 / 102.0 / 100.1 | 96.7 / 100.8 / 91.3 | **3/3 pairs**, +15.5 / +1.2 / +8.8% |
| Distill-35B-A3B Q2KXL, `--expert-warmup 1` | 256 experts, routed > VRAM | 9.4 GiB (4.7) | 36.5 / 35.6 | 30.9 / 31.1 | **+15.3 / +12.7%**, 2/2 |
| Qwen3-30B-A3B i1-Q2_K | tier is disk-bound: 376 MiB read/token | 9.8 GiB (4.9) | 93.4 / 93.5 | 90.5 / 92.2 / 92.5 | +2%, inside the spread |
| Qwen3-4B-Q8_0 | dense, no expert tier | -- | 9.3 | 9.3 | 0% (the control) |

Three independent sets on the REAP file, all with the default, say what the
absolute band does: 101.7 -> 88.2 (+13.3%, 3/3), 113.3 -> 93.2 (+18.7%, 2/2) and
the table's own 114.4 -> 96.7 (+15.5%) with one pair at +1.2%. The sign is 8/8
pairs and the size moves between 1% and 19% with machine load -- so the stable
number is the mechanism, not the headline: `us/expert` 555 -> 400 in the table's
set, 468 -> 390 in the same set's other pairs, 414 -> 199 on the 35B-A3B, i.e.
the copy halves every time. A single pair in a quiet moment reads +17-18%; a
pair taken while something else is on the card reads +1%.

The mechanism is the xfer line, and it halves: 386 -> 193 us/expert on the
35B-A3B, 530-555 -> 391-396 on the REAP file. What the pin does NOT do is make a
disk-bound tier faster: the 30B-A3B reads 376 MiB per token from the file, so
the copy the pin removes is ~2% of its step. That boundary is why the dense arm
and the disk-bound arm are in the table instead of dropped.

The old explicit arm is no longer better than the default -- `KRK_HOST_PIN_MB=16000`
sized 16 GiB where auto sizes 12.3, and both land in the same 87-94 ms/step band,
so the measured curve is flat over the top third of the tier.

**The same probe answers a second question, and the answer decides the path.**
Two pageable paths exist: the staging ring and a direct DMA the driver stages
itself on one side stream. Init times both, plus pinned, and adopts direct at
**1.20x** (`krk::pageable_direct_wins`). A wide aperture puts the host copy in
the driver's hands, which clears the line -- and then the prefix declines
because there is no copy left to buy back, i.e. fast and swappable without a
knob. **That branch cannot be reached on this box's card** (a 256 MiB aperture
and no ReBAR), so it is proved against a simulated aperture instead:
`CountingBackend::set_test_aperture` makes the two pageable paths genuinely cost
different amounts of time, and `test_pageable_path_probe` (38 checks) drives
`probe_h2d_path` and then a real `Engine::init` on a wide (0.4 against 20 GB/s)
and a narrow (6.0 against 0.4) one -- direct on the first, the ring and the pin
on the second, identical tokens on both. The group is the suite's only serial
one, because it reads the clock and a neighbour's traffic would be measured as
the aperture.

**And this box's 256 MiB aperture is close enough to the line to flap across
it**, which is why the verdict is cached. The same 16 MiB probe read 8.7
against 10.6 GB/s (direct wins at 1.21x) in one run and, with
`KRK_H2D_PROBE_RERUN=1` minutes later, 10.9 against 12.3 (1.13x, the ring stays)
-- the earlier "it is a tie here" line was one sample of that band, not a
property. So the answer is remembered per (card, driver version, probe size, aperture
hint) in
`%LOCALAPPDATA%\kraken\h2d-probe.cache` (`$XDG_CACHE_HOME`/`$HOME/.cache`
elsewhere), the key carries the runtime's `large_bar` hint exactly because a
BIOS change is the event that invalidates it, and `hipDriverGetVersion`
because the driver stages the copies the verdict is about (7.16.26331 here,
printed by the banner beside the runtime version); entries expire after 30
days, and `KRK_H2D_PROBE_RERUN=1` re-measures. Each write prunes as well:
expired lines are dropped (the reader refuses them anyway, so they only hold
a slot in the cap), and another driver's verdict goes once it is a week old
-- a rollback window, not a second expiry. A run that reads the cache says so and how
old it is, rather than printing a fresh-looking number: `a 16 MiB copy from the
cache, measured 8 s ago at 8.7 GB/s ...`. `KRK_H2D_CACHE=0` disables the
cache, and `KRK_H2D_CACHE=<path>` moves it. `kraken-bench --h2d` never reads
it -- measuring now is what that tool is for -- but it WRITES it: after the
balanced pair it runs `probe_h2d_path` at 16, 32 and 64 MiB and stores all three
verdicts, so an explicit measurement is also how a BIOS change is picked up
without waiting for the 30-day expiry. Those are the only sizes an entry can be
under because the engine BUCKETS its probe size (`h2d_probe_size_bytes`: clamp
to 16-64 MiB, round up to a power of two), which is what makes the bench's three
sizes cover every model -- a 20 MiB expert looks up the 32 MiB entry. Measured
here in one sitting, the same card read 11.64 against 11.63 GB/s at 16 MiB,
10.19 against 11.87 at 32 and 11.63 against 11.32 at 64, i.e. the answer is
size-dependent enough that quoting one size for another would be a different
verdict.

That probe also had to be rebuilt after it answered two different things in two
runs. As three sequential arms the same pair read **13.9 GB/s (ring) then 10.5
(direct)** at 12.6 MB, and **10.9 then 12.8** at 16 MiB in the other order: the
slot a run occupied was worth more than the difference being measured, which is
the same ordering trap the decode A/Bs here already know about. It is now
order-balanced and interleaved -- both pageable paths every iteration, first
place swapped, median per arm -- and `kraken-bench --h2d` prints that verdict
next to the sequential table so the two can be compared. The 1.20x line sits
above the measured flap band (1.01-1.17x across copy sizes) on purpose.

### Ranking

| # | target | measured | shape of the fix |
|---|---|---|---|
| 1 | router readback per layer | **41.7 ms/step** | device-side top-k and ids consumed on device (`dispatch_cap`), today blocked by "whole routed set resident" (42 GiB routed vs 7.4 GiB budget) |
| 2 | host expert loop | **33.7 ms/step** (alloc ~12, promotion enqueue ~10, launches ~11) | the batched-DMA / arena work of `REVIEW-EXPERT-PATH.md` §5 |
| 3 | ~~pinned WARM as a small-BAR default~~ | **+13-18%**, landed | now automatic from the init probe (not the BAR bit); `KRK_PIN_AUTO=0` is the A/B arm |

Prefill is a separate anomaly worth its own pass: 5 prompt tokens take 985 ms
(197 ms/row, 10x a decode row), and `gemm(gemv_m)` -- the rows 2-8 ladder the
prefill uses -- measures **29 GB/s against gemv's 743**.

## A ranked index: worth +70-87% decode when RAM is short, and -25% when it is not

Measured on this file at `--expert-cache-mb 8000` (routed set 32.3 GiB = 24% of
the pool), greedy, one 55-token prompt, 24 tokens, one run per arm, same prompt
in both. The index is 62 positions, full-forward, measured on that same prompt --
the best case a ranking can have.

**96 GiB installed, so WARM holds the whole routed set (33062 MiB).**

| arm | prefill | decode | ms/step | decode promote | VRAM evictions | pool `reuse` |
|---|---|---|---|---|---|---|
| even order | 27.0 / 27.1 tok/s | **11.9 / 11.8 tok/s** | 84.3 / 84.8 | 14.4 ms/step | 1194 | 2876 |
| ranked | 31.7 / 32.2 tok/s | **9.2 / 8.9 tok/s** | 109.2 / 112.6 | 22.7 ms/step | 59 | 119 |

**`--ram-tier 24`, so WARM holds 14746 MiB of the 33062 MiB set (45%) and the
disk tier is live.**

| arm | prefill | decode | ms/step | promotions | bytes promoted | read from file |
|---|---|---|---|---|---|---|
| even order | 8.6 / 8.6 tok/s | **4.1 / 3.8 tok/s** | 246.1 / 262.3 | 5872 / 6887 | 13.9 / 16.5 GiB | 12.0 / 11.6 GiB |
| ranked | 9.8 / 9.3 tok/s | **6.9 / 7.1 tok/s** | 144.6 / 141.4 | 3965 / 3580 | 8.3 / 7.6 GiB | 11.4 / 11.3 GiB |

**The ranking does the same thing in both regimes -- it cuts promotion roughly in
half -- and only one regime turns that into time.** At the 24 GiB class the
ranked arm promotes 7.6-8.3 GiB instead of 13.9-16.5 and decode gains **+70-87%**.
At the 96 GiB class it promotes fewer experts too (3678 against 4827) and still
loses 25%, because there the resident set goes sticky: 59 evictions against 1194,
and the pool's slot `reuse` collapses from 2876 to **119**. A freed cell is then
rare, so each decode miss takes the fresh-allocation path instead of recycling
the cell the previous step released -- decode promotion 22.7 ms/step against 14.4
for the same 10 experts per token.

So the rule is not "rank" or "do not rank": it is that **the ranking pays where
the tier it fills is too small to hold the set, and costs where the pool can no
longer recycle cells.**

### The pool is not the defect, and fixing it did not pay

The 119-reuse reading above looked like a stranded free list, so it was chased:
the pool's GROWTH path enforces its plan only through `pool_slab_take_`, which
caps a slab to the room left. After the first release growth is per-cell and
unguarded, so `committed` walked to 8429 MiB against an 8002 MiB plan; and
`release_batch_` hands a slice back to the driver (instead of the free list)
whenever `committed > budget`, which is exactly the state that overshoot creates.

Refusing that growth did fix both counters -- `committed` 8429 -> 8000, `reuse`
119 -> 390, i.e. **92% of the 423 slices the run ever freed were reused** (67%
before) -- and **decode did not move: 8.8 tok/s against 8.9, 113.1 ms/step against
112.6.** It also broke two arms that live at their plan by construction: the
LFM2.5 fixture's CLI prefill exits 1, and `kraken-bench` crashes on a refused
allocation it never expected. Reverted.

The counters say why the reuse number was never the cost. The ranked arm's decode
spends 546 ms in promotion across 23 steps at ~476 us per expert, i.e. **~1147
promotions inside decode, against ~765 for the even order** -- it misses ~50%
more, and that is the 29 ms/step. A ranking measured on a 62-position prompt
holds the PROMPT's set; the continuation's routing differs from it by a thousand
experts. The lever is the set, not the free list.

### Seeding VRAM from the top of the rank, measured

That lever already exists: `warm_experts` promotes the top of the ranked order
into VRAM at init, and the gate that normally skips it at this ratio (corpus must
fit the pool, +5%) is bypassed with `KRK_WARM_PREFILL=1`. Same prompt, 24 greedy
tokens, one run per arm unless noted, index 62 positions:

| machine | arm | prefill | decode | ms/step | VRAM seeded |
|---|---|---|---|---|---|
| 96 GiB RAM, 8 GiB pool | even order, read-through | 27.0 / 27.1 | **11.9 / 11.8** | 84.3 / 84.8 | 0 |
| | ranked, read-through | 31.7 / 32.2 | 8.9 / 9.2 | 113.1 / 109.2 | 0 |
| | **ranked, seeded** | **40.6 / 42.8** | 10.4 / 10.6 | 96.5 / 94.8 | 3715 experts, 8000 MiB |
| 24 GiB RAM, 8 GiB pool | ranked, read-through | 9.8 / 9.3 | **6.9 / 7.1** | 144.6 / 141.4 | 0 |
| | **ranked, seeded** | **16.5** | 6.0 | 166.3 | (WARM-staged top) |

Seeding is a PREFILL trade, not a win: +31% prefill on the 96 GiB box and +75% on
the 24 GiB one, against -11% and -15% decode, because a seeded pool starts full
of the prompt's set and the continuation then evicts through it (3674 evictions
against 141 for read-through). The phase costs 750 ms of extra startup (7009 ms
against 6263). Break-even, from those rates: **~57 decode tokens on the 96 GiB
box, ~97 on the 24 GiB one** -- below that the seed's faster prompt wins the wall
clock (24 tokens: 3.61 s against 4.06 s), above it the churning pool does.

**Read-through plus a ranked index is the worst configuration of the three in both
regimes**, so on this file at 24% fit the index only pays with the seed on.

### The seed does not have to fill the pool, and at 50% it is free

`KRK_VRAM_SEED_PCT=0..100` caps the seed at that fraction of the pool (and turns
the seed on by itself at ratios the rule above declines). Sweep at 8 GiB, 96 GiB
RAM, same prompt and index:

| seed | VRAM seeded | prefill | decode | ms/step |
|---|---|---|---|---|
| 0% (= read-through) | 0 | 31.7 / 32.2 | 8.9 / 9.2 | 113.1 / 109.2 |
| 25% | 928 experts, 2002 MiB | 36.8 | 10.5 | 95.5 |
| 40% | -- | 40.6 | 11.4 | 87.5 |
| **50%** | **1859 experts, 4002 MiB** | **41.8 / 41.7** | **11.9 / 11.8** | **84.0 / 84.9** |
| 60% | -- | 42.8 | 10.7 | 93.2 |
| 75% | 2787 experts, 6002 MiB | 42.2 | 10.8 | 92.4 |
| 100% | 3715 experts, 8000 MiB | 40.6 / 42.8 | 10.4 / 10.6 | 96.5 / 94.8 |

**Half the pool is a free win against both baselines**: decode 11.85 tok/s matches
the even order exactly (11.9 / 11.8) while prefill is 41.7 against its 27.0, i.e.
+54%; against read-through ranked it is +30% prefill AND +31% decode. The curve is
sharp -- 40% and 60% are each ~1 ms/step worse -- and the mechanism is the one the
auto rule's own comment gives: half the pool is seeded from the rank and half the
LFU still earns, so neither a sticky prompt-shaped set nor an unseeded pool gets
the whole tier. Whole-pool seeding fails for the reason measured above (3674
evictions through a pool with no room to adapt); no seeding fails because the
first tokens then decide everything.

At `--ram-tier 24` the same sweep is a monotone trade instead, because WARM only
holds 45% of the set and the seed can only use what WARM staged: 50% gives 12.9
prefill / 6.7 decode against read-through's 9.55 / 7.0, and 100% gives 16.0 / 6.4.
There the seed's prefill gain always costs some decode, so the best wall clock
depends on the reply length (24 tokens: 100% wins, 7.19 s against 7.84 at 50% and
9.19 unseeded).

**Not made the default, and Tiel-Coder-35B-A3B is why.** Same file, same 256
greedy tokens, 10430 MiB budget (22604 MiB routed set, 2.17x), index present, all
three arms staging the whole WARM tier (the first 0% run of the sweep read a cold
page cache, stopped at 2 rounds of staging and reported 19.7 / 17.8 -- the numbers
below are the warm re-run, which is the fair comparison):

| seed | prefill | decode | ms/step | promotions | bytes promoted | pinned |
|---|---|---|---|---|---|---|
| 0% | 34.1 tok/s | **22.1 tok/s** | **45.2** | 9579 | 21448 MiB | 751 |
| 50% | 49.0 tok/s | 12.2 tok/s | 81.9 | 23548 | 52840 MiB | 1173 |
| 100% | 42.1 tok/s | 8.9 tok/s | 112.9 | 35856 | 80903 MiB | 1464 |

Half a pool is +44% prefill and **-45% decode** at 256 tokens. The seed
multiplies promotion 2.5x, because the set it holds is the prompt's and the
continuation keeps re-promoting what it evicted.

**Which arm wins depends on how long the reply is, and the crossover is ~96
tokens.** Same file and prompt, both arms, `prefill + decode` as the user pays it:

| reply tokens | 0% seed | 50% seed | winner |
|---|---|---|---|
| 4 | 1124 ms | **745 ms** | seed, -34% |
| 8 | 1317 ms | **931 ms** | seed, -29% |
| 12 | 1610 ms | **1244 ms** | seed, -23% |
| 16 | 1856 ms | **1394 ms** | seed, -25% |
| 32 | 2734 ms | **2228 ms** | seed, -19% |
| 64 | 4288 ms | **3628 ms** | seed, -15% |
| 96 | **5645 ms** | 5675 ms | tie (0.5%) |
| 128 | **6848 ms** | 8157 ms | unseeded, -19% |
| 256 | **12704 ms** | 21724 ms | unseeded, -71% |

Per-step cost explains the shape, and it moves in opposite directions: the seeded
arm reads 66.8 -> 57.7 -> 56.9 -> 53.0 -> 49.1 -> 54.1 -> 60.0 -> **83.1** ms/step as
the reply lengthens, while the unseeded one reads 92.2 -> 77.4 -> 71.7 -> 62.7 ->
55.7 -> 51.3 -> 47.8 -> **46.8**: the LFU keeps learning, and a prompt-shaped set
keeps decaying. So the seed buys a fixed ~236 ms of prefill plus cheaper early
steps, and a reply longer than ~96 tokens spends all of it.

**An earlier version of this note said the seed loses at every length. That was
wrong, and the way it was wrong is worth keeping:** it extrapolated the 256-token
per-step cost down to short replies, and neither arm's step cost is constant in
length. At 24 tokens the seed is not 0.65 s slower -- it is about 0.4 s faster.

**It is not the pin rule, and that was measured rather than assumed.** A seeded
expert crosses the LFU's 24-hit threshold quickly, so a 50% seed lifted the pinned
count 751 -> 1173; `KRK_EXPERT_PIN_HITS` was added to test the obvious reading,
and it is backwards: with pinning effectively off (999999, 0 slots pinned) the
same arm reads **9.1 tok/s**, and at 96 hits it reads 9.7 -- both WORSE than 12.2
at the default 24. Pinned slots are what stops the seeded set being re-promoted a
third time.

So the rule is narrower than "seed half the pool": it pays on a file whose routed
set is many times the pool and whose reply is short relative to the prompt (the
REAP-320 shape, 4.1x, 24 tokens), and it loses badly once the run is long enough
that the continuation's set differs from the prompt's.

### Choosing the fraction without a flag

`warm_experts` now picks the arm from the two things the measurement says matter --
whether the routed set fits the pool, and how long the reply is expected to be:

- set fits the pool (+5%, the LFM2 shape): seed the whole pool, as before;
- set does not fit, `--max-tokens` <= 64: seed **50%** of the pool;
- set does not fit, anything longer or unbounded: seed **nothing** (the pre-existing
  default), because past ~96 tokens the prompt-shaped set decays faster than a
  half-seeded pool pays back.

`--max-tokens` is the only expectation the engine has at init, so it is what the
rule reads (`EngineConfig::expected_reply_tokens`, 0 = unknown = long). The server
leaves it 0 on purpose: its requests carry their own budgets, so init has no honest
expectation and the safe arm is chosen. `KRK_VRAM_SEED_PCT` and
`KRK_WARM_PREFILL=1` still outrank the rule for an experiment.

Checked on both files at both ends of the curve: Tiel at `-n 24` seeds 2700
experts (5960 MiB of a 50% cap) and gives 43.5 tok/s prefill / 1327 ms decode
against the unseeded ~2295 ms at that length; at `-n 256` it seeds nothing and
reads 32.1 / 12116 ms, the arm that wins there. REAP-320 at `-n 24` seeds 1859
experts (4002 MiB) for 34.7 tok/s prefill and 1995 ms decode.

**A caveat that bit the first verification run:** the seed can only promote what
the WARM phase staged, so a COLD file (26 GiB read with an empty page cache) can
hit the staging ceiling before the ranked experts are in RAM and then promote
nothing at all -- it reported `VRAM +0 experts (0 MiB of a 50% seed cap)`. The
numbers above are warm-cache runs. A cold first invocation gets the safe arm's
performance, not a wrong answer.

Why the ranking is so flat: 320 experts top-10 is ~3.1% of routing each, and the
file's 62-position mass gives the top-80-by-mass 29.4% of it against 24.1% for
ids 0-79 -- a mild prior, not a hot set. Correlation between expert id and mass
rank is -0.162, so the even order's advantage is not "low ids are hot"; it is
that a churning LFU tracks a flat distribution better than a sticky set does.

Prefill tok/s is the misleading half in both tables: the ranking warms the very
prompt being measured, so it reads as a win even where decode loses. Judge a
ranking on decode.

Two defects surfaced getting to even that measurement:

- **A full-forward scan cannot run at the budget it is meant to describe.**
  `--expert-scan` disables the ranked warmup, and at 8000 MiB the run dies in
  generation with `expert cache refused a routed expert ... no resident slot
  could be reclaimed (moe layer 47 expert 1)`. Routing does not depend on the
  cache budget, so the scan belongs at headroom: at `--expert-cache-mb 10000`
  the same scan completes cleanly (62 positions, 55 prompt + 7 generated).
- **The scan writes an index with zero positions, and the loader reads it as a
  ranking.** The crashed run above still wrote
  `(0 positions, full-forward)`/15360 pairs, and the next run loaded it at HOT
  28.8% and **8.6 tok/s against 11.8**. `merge_expert_index` already refuses a
  run with `expert_scan_tokens_ <= 0`; `write_expert_index` has no such guard.
  A scan that measured nothing must not be written, and a file that says 0
  positions must not be trusted on load.
