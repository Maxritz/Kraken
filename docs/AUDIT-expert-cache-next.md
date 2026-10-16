# Expert-cache audit and next implementation gates

2026-10-11. Scope: Kraken on gfx1201 / 15.9 GiB VRAM / 96 GB host RAM;
Laguna, Qwen MoE, GLM and DeepSeek. Edge0-rdna was inspected read-only;
only concepts are borrowed. No source was copied from either sibling engine.

## What exists, and what not to replace

Kraken's `ExpertCache` already implements inclusive HOT VRAM / WARM pageable
RAM / immutable file COLD, demand-earned LFU counters, pinning and aging,
ranked warmup, source-size validation, batched selected-expert file reads,
a bounded pinned upload ring, and event-guarded size-class VRAM arenas.
Sources: [expert_cache.hpp](../include/krk/expert_cache.hpp),
[expert_cache.cpp](../src/expert_cache.cpp),
[backend_hip.hip](../src/hip/backend_hip.hip).

A plain shared LRU is not an automatic improvement. Nor is CPU expert
execution: the existing experiment measured a 3.1x loss at tight budgets.
Historical rankings seeded into live eviction counters also lost; do not
repeat either as a default policy.

The local Edge0-rdna `python/src/edge0/streaming/cache.py` provides a global
LRU and a separate bounded speculative prefetch buffer. Its `layer.py` has
per-layer decayed hot counts, predicted-expert bonuses, fixed-slot staging,
and exact miss fallback. `docs/streaming.md` distinguishes cold on-demand
prefill from warm whole-layer loading. These are useful concepts, not HIP
implementations or transferable timing results. In particular, paths that
replace routing or send an absent expert to a zero slot are NOT legal here:
Kraken must compute every expert selected by the unchanged model router.

Strata's useful concepts are independently budgeted resident/staging tiers,
per-layer learned profiles, decayed current-request heat with swap hysteresis,
next-layer router prediction used only as a hint, and explicit reader/DMA
ownership before slot reuse. Exchange rotation is not itself proof of overlap.
Strata is MIT, but no code was imported. Upstream URLs are mutable `main`;
freeze an exact revision and preserve notices before any future code reuse.

## Proven fixes in this pass

### Batched WARM admission was not bounded

`prefetch_layer` allocated all jobs before committing `host_bytes_`, so a
batch could reserve more than the host budget. Duplicate IDs could reserve
multiple buffers for the same slot. An iterator was also carried across
room-making, which can erase cache entries.

The fix tracks pending reservation bytes, deduplicates valid keys, checks
remaining budget before allocating, and reacquires the iterator after eviction.
The unchanged demand path loads experts that prefetch cannot stage; no expert
is dropped and speculative admission still earns no popularity pins.

Regression proof: seven new assertions failed on the old implementation
(38/45); the fixed WARM-tier group passes 45/45. Tests cover an oversized
batch, duplicate/invalid IDs, repeat prefetch, demand after a partial batch,
no demand/pin inflation, and byte-for-byte source slices after promotion.
Full suite: 2916/2916. This proves correctness/resource bounds, not a speedup.

### The reuse-distance analyzer made an invalid capacity claim

The old analyzer called *all accesses since last use* reuse distance and
claimed it determined tier hits. LRU capacity depends on *distinct intervening
keys*, and Kraken does not even use LRU. The updated
[expert_trace.py](../tools/expert_trace.py) reports both distances, performs
explicit empty-cache equal-slot inclusive LRU replay, includes compulsory
misses, and reports per-layer active expert counts/top shares. It states
that replay is not the live LFU, byte-budget or warmup policy.

Selftest: ten constructed cases covering repeated accesses, cross-layer key
identity, inclusive tiers, disabled tiers, and exact replay. Invalid TSV is
rejected rather than silently changing the workload.

An existing 8,918-access trace demonstrates the difference: median inter-access
distance 641 versus median distinct-key distance 542; 2,793 unique expert pairs.
Observed HOT 68.7% / WARM 31.3% / COLD 0%. Empty LRU at 640 HOT slots predicts
35.7% HOT, 33.0% WARM, 31.3% compulsory COLD. These are NOT an apples-to-apples
LFU-versus-LRU comparison: initial residency and budgets differ. This trace
lacks model/request/engine identity and byte geometry, so it cannot support
per-request tuning or a throughput claim.

## Ranked next work: measurements before policy changes

| Priority | Change | Discriminating proof | Correctness guard |
|---|---|---|---|
| 1 | Trace identity and cost: model fingerprint, engine/request, token position, prefill/decode, bytes, read/eviction/allocation/copy waits | One target+drafter run produces distinct streams; byte totals reconcile with counters | Bounded recording, detectable truncation, no silent file-write failure |
| 2 | ~~Physical VRAM arena budget and ownership accounting~~ **DONE 2026-10-11, then the cost attacked** | Individually owned size-class cells; committed = live + reusable in aligned bytes; every run prints `[stats ] expert-pool budget/committed/live/reusable`; the device gate in `kraken-bench` drives the limit. The reuse guard then went from a host round trip plus a pending-event query (87 us per cell) to a device-ordered wait (24 us): `alloc` 618.4 -> 182.5 ms and decode +11.3% at 16 MiB | Guard-fallback frees a base allocation only; a live cell is never freed by a shrink; owner sum bounds target + draft together; no stream ever waits on a destroyed event |
| 3 | ~~Protect the actual layer's active set while batch-promoting known misses~~ **DONE 2026-10-11** | `leased` pins the current consumer against a staged promotion's eviction; a budget too small for both returns false and falls back to exact serial demand (tested) | Lease is taken only after both experts are proven to fit; staged hits earn no HOT popularity and no demand heat |
| 4 | ~~Async selected-set staging, then one-layer prediction~~ **MEASURED, NOT ENABLED 2026-10-11** | The device timeline was built (`KRK_STAGE_TIMING=1`) and *disproves* the benefit at these budgets: 0.69-0.74 ms copy against a 0.32-0.34 ms reader window, 49-54% of copies fully hidden, and 75% of the step is already promotion. A/B: no win in six runs, negative sign in every constrained one | Publication ticket + lease; stdout md5 unchanged in every arm (`28fa19c37`) |
| 5 | Separate speculative staging budget, promotion-on-use, low-pressure admission | Held-out request traces: useful/wasted bytes, demand-hit share, eviction pollution | Predictions never replace the router; speculative loads never earn demand heat |
| 6 | Per-layer active-core protection with hysteresis and aging | Replay current LFU and alternatives on identical byte budgets, then interleaved live A/B | Globally bounded bytes; no arbitrary fixed N per layer; no historical ranking masquerading as traffic |
| 7 | Multi-shard file reader and validated expert-pack/coalesced I/O | Cold read bytes/time improve on Laguna-S, including individual source-pointer byte checks | Offset = source pointer minus its own shard mapping base; identity/layout validated per shard |
| 8 | Prefill-aware I/O mode | Distinct selected experts / layer determines on-demand versus whole-layer staging; cold and warm cases reported separately | Same tensors/dtypes/scales and selected rows; no unrequested full-corpus RAM allocation |

**Both audit risks named here are now fixed and covered, not merely re-tested.**
`alloc_pooled` subtracted `pool_bytes_` on reuse while arena-backed cells were never
added to that counter on release -- the suspected unsigned underflow was real in
shape, and it is gone with the arena: cells are individually owned, `committed` is
`live + reusable`, and a shrink frees free cells only. The event-creation failure
path called `hipFree(p)` on what could be an arena interior pointer; with one
allocation per cell that call is legal, and the guard path is exercised by
`KRK_TEST_POOL_NO_EVENT=1`. The full measurement, the overlap probe and the
negative staging result are in
[AUDIT-active-expert-staging.md](AUDIT-active-expert-staging.md).

## Model coverage and gates

- Qwen3-MoE-4x0.6B is the cheap mixed-dtype correctness and constrained-budget
  churn fixture. Qwen3.8 35B adds 256 experts/layer and a whole-corpus-fit arm.
- Laguna-XS adds genuine capacity pressure; Laguna-S adds shard identity and
  more expert bytes than the available RAM/VRAM working budget.
- GLM-4.7 and actual DeepSeek architectures are refused for MLA/compressed
  attention in the current tree. Expert-cache infrastructure can be generic,
  but no throughput or end-to-end correctness claim is possible for those
  files until their architecture is runnable. A DeepSeek-named Qwen distill
  is not evidence of DeepSeek architecture support.
- Every performance arm keeps routing, exact expert bytes, kernel arithmetic,
  and accumulation order unchanged. Time unprofiled interleaved/order-balanced
  repetitions; use device traces for overlap/launch counts, not absolute
  per-op timing under the intrusive profiler. Report disk cold/warm separately.

## External research sources

Five parallel research tasks covered DFlash, architecture contracts, compiler
rounding, Strata, and alternative expert caches. One research response used an
incorrect date (2025); its date-dependent conclusions were not adopted.
Current session date is 2026-10-11. Research was source inspection, not local
benchmarking or verified port compatibility.

- [Strata cache](https://github.com/Niko1221/Strata/blob/main/src/core/expert_cache.cpp)
- [Strata expert source and lookahead](https://github.com/Niko1221/Strata/blob/main/include/strata/core/expert_source.hpp)
- [Strata foresight staging](https://github.com/Niko1221/Strata/blob/main/include/strata/core/foresight_swap.hpp)
- [Strata exchange ownership](https://github.com/Niko1221/Strata/blob/main/docs/EXCHANGE_ROTATION.md)
- [Strata measurements/limitations](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
- [MoE-Infinity](https://arxiv.org/html/2401.14361): request-scoped activation traces; CUDA measurements, not a HIP speed claim. Code is Apache-2.0.
- [HIP memory API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)
- [HIP stream/event ordering](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html)
- [DeepSeek-V2 MLA contract](https://arxiv.org/html/2405.04434): latent + separate RoPE-key cache, absorbed content query, value up-projection after attention; scaling is not based on latent rank.
- [DFlash reference](https://github.com/z-lab/dflash): verify exact target pairing, hidden-state tuple offsets, anchor/position alignment, norms/masks and candidate semantics before attributing zero acceptance to cache behavior.
