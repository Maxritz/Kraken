# GPU-resident expert cache: what upstream merged, and where this engine stands

2026-10-21. Scope: Kraken on gfx1201 / 15.9 GiB VRAM / 96 GB host RAM. The
upstream numbers below are the authors' own; the numbers marked "here" are
measured on this box and are the ones the argument rests on. Nothing is copied
from upstream -- this is a design read, and the reason each piece would or would
not pay here.

## The three mechanisms

| reference | where the decision lives | where the copy lives | budget knob | reported |
|---|---|---|---|---|
| [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887), merged Oct 7 | host, in the scheduler's copy callback: it reads the slot map and plans hits/misses each step | `ggml_backend_tensor_set_async` into a GPU bank | `--moe-cache-mib N` (default 0, off) | 4090: 25.0 -> 39.4 tok/s (1.57x, 72% hit), 40.7 (1.62x, 77%); 5090: 30.8 -> 54.5 (1.77x), 67.8 (2.20x, 89%) |
| [PR #30112](https://github.com/ggml-org/llama.cpp/pull/30112), merged Oct 8 | same, per GPU | same, per GPU | same flag, split across devices | 2x4090 layer split: 34.78 -> 60.38 (1.74x, 85.9%), 64.39 (1.85x, 95.8%); prompt 440.7 -> 402.0 / 321.5 tok/s |
| [issue #29949](https://github.com/ggml-org/llama.cpp/issues/29949), [commit c81901a](https://github.com/ggml-org/llama.cpp/commit/c81901a571d5778d840d382d3b3789014cb55d05) | **device**, `GGML_OP_LRU_DECIDE` on the router's top-k + a GPU i32 state tensor | **device**, `GGML_OP_LRU_FILL` as a graph dependency | `--moe-lru-slots N` | 7900 XTX IQ3_XXS: 9.7 -> 37.2 (89.4%), 44.8 with MTP; IQ4_XS 72 slots / 6.4 GiB: 31.2 (77.9%) |

The number that decides what to build here is the one that holds everything
else fixed: **at the same ~6.5 GiB pinned-host budget, #29949 reads 31.2 tok/s
against #29887's 20.1 on the same machine and model** (77.9% against 75.0% hit).
So the difference is neither capacity nor the bus -- it is *where the decision
and the copy live*. #29949's own correctness evidence is the right shape for a
cache: PPL 1.4290 without the pool against 1.4292 at every slot count from 64 to
192, and 12/12 greedy runs with one text hash.

## What this engine already does, and should not give up

- **LFU with aging, not LRU** (`include/krk/expert_cache.hpp`). A plain shared
  LRU was already rejected here with a measurement, in
  `docs/AUDIT-expert-cache-next.md`: seeded historical rankings lost to
  demand-earned counters. Upstream's LRU is the weak part of their design, not
  the part worth copying -- and #30112's own note that sharing slots across
  same-layout layers bought 5% hit rate is a statement about sharing, which this
  tier already does (one size-class arena set, all layers).
- **A pinned WARM prefix** (`--expert-pin-mb`, `pin_budget_`). #29949's whole
  premise is that the source of a fill must be pinned; that is already the WARM
  tier here, with the LRU window kept at full `budget_` so a pin cannot shrink
  it.
- **Per-tier reporting**, `experts  N loads, M evictions, X% hit rate` with
  host hits apart from device hits -- so a hit-rate claim here can be checked
  against which tier served it, which none of the upstream numbers separate.
- **A device dispatch table already exists** (`include/krk/moe_table.hpp`): one
  launch per routed layer, no per-expert host pointer, no `acquire()` between
  routing and the GEMMs.

## The structural difference: our table is published once, theirs is per token

`moe_table.hpp` states its own limit: the table "is only valid while residency
is STABLE [...] the engine publishes it once (after the warm-up) and uses it only
when the whole routed set is device-resident, where no such event can happen
mid-run. A cache that overflows its budget simply never publishes."

That is the whole gap. Our `KRK_DEV_PTR=1` path is gated on
`dev_table_ != nullptr && !dispatch_table_stale()` and the entire routed set being
resident, so on any model that does not fit the card -- every file this engine is
interesting for -- the default is the per-expert loop, which stops once per layer
to look at the router row:

    moe.route.fetch  0.523 ms/layer on Mellum2.1, of which 0.077 ms is the copy
    itself; 28 layers of a 23.7 ms step is 14.6 ms spent waiting for the layer's
    own GPU work to retire so the host may look at it.

Upstream made that wait unnecessary rather than shorter: `LRU_DECIDE` reads the
router ids on the device and emits remapped slot ids plus a fill list, `LRU_FILL`
copies the misses, and the fills are dependencies of the same graph. The expert
matmul is handed **slot ids in place of expert ids**, so the graph shape stays
fixed while the routing stays dynamic -- which is what makes it replayable and
what our published-once table cannot express.

## What that would and would not remove here

**Would remove:** the per-layer host round trip above (14.6 ms of a 23.7 ms
step on that measurement, model- and card-dependent), and the all-or-nothing
gate on the dispatch table, so a model whose routed set does not fit could still
run one launch per layer.

**Would not remove, and the audit's own first premise was wrong about it.** The
256 MiB arm was recorded as `alloc 2445.5 ms of a 3408.1 ms promote total`
(2026-10-09) and that is what made a fixed slot bank look like the first step.
Re-measured on the current binary, same file and command, that arm is **`alloc
3.4 ms` of 2849.9 ms**, because the pool hands 96% of its cells back from a free
list: `alloc 819 reuse 18231` over 6077 promotions, 0.56 us per hand-out. The
2.4 s that used to be there was the readiness query on a reused cell, and the
`KRK_POOL_HOST_WAIT=1` arm still shows it (**alloc 1332.3 ms**, same 819 real
allocations) -- a bank does not touch that row either, because the query belongs
to the hand-out, not to the allocator.

What the 256 MiB arm actually spends is the pageable H2D rate at promotion
shape, and that is measurable in one variable:

| arm (same file, prompt, 24 greedy tokens) | alloc | xfer | per expert | decode | output md5 |
|---|---|---|---|---|---|
| default (pageable direct, 4800 of 9600 MiB pinned) | 3.4 | 2815.3 | 469 us | 7.2 tok/s (139.8 ms/step) | `cfa33f59` |
| `KRK_POOL_HOST_WAIT=1` | 1332.3 | 2361.0 | 607 us | 6.5 tok/s | `cfa33f59` |
| `KRK_WARM_DMA=1` (ring, non-blocking publish) | 4.1 | 1485.9 | 251 us | 9.6 tok/s | `cfa33f59` |
| non-blocking direct publish (tried, reverted) | 4.1 | 2961.1 | 493 us | 6.8 tok/s | `cfa33f59` |
| `KRK_HOST_PIN_MB=9600` (all WARM pinned) | 3.1 | 27.4* | 10 us | **15.3 tok/s (65.3 ms/step)** | `cfa33f59` |

\* the pinned row times the ENQUEUE; the copies overlap compute, which is why
the step is 65.3 ms and not negative. The fourth row is the one that settles the
mechanism: making the pageable publish non-blocking (the same event the pinned
path uses) changed nothing, so the 469 us is the pageable DMA's own rate at this
size -- **1.9-2.0 GB/s against 17.5 from pinned** -- not a host wait. Source
pinning is the lever; the transfer path is not.

**So the bank's justification is address stability, not allocation time.** It is
still what a slot-addressed dispatch table needs (a cell that never moves is a
pointer a device table can be published once and kept), and it is cheap to hold.
It is not a 2445 ms win, and the ordering below is chosen for the slot table, not
for the allocator.

**Would not remove either:** the file read when WARM is cold (438.8 ms read +
675.4 ms xfer of a 60.6 ms step in the cold-warm-up arm), which is a fill-path
problem and is already `--expert-warmup 1`'s job.

## The order to build it in, with the gate each piece needs

1. **A fixed slot bank -- stable addresses, no allocation on a promotion.** One
   device pool carved at load, cells reused. Gate: stdout byte-identical to the
   current binary on a model that fits AND on the 256 MiB arm (md5, not a text
   eyeball), `xfer` and the step time unchanged within the pairs' MAD, and the
   `alloc` row no worse; then `kraken-bench --gate` rc=0. The gate is
   "unchanged", not "faster", because the alloc row is 0.1% of the step -- the
   value is the invariant the next step needs.
2. **Slot-addressed dispatch, published once and never stale.** `MoeDispatchEntry`
   indexed by slot instead of by expert, and `ws_ids_` carrying slot ids. Gate:
   the same text as the per-expert loop on an MoE that fits (md5), and the
   chunk-invariance group still reports one hash across `--chunk 1/2/8` -- a
   weights-address change must not move the arithmetic at all, and that group is
   the only thing that can tell.
3. **An in-graph fill** (device copy from the pinned WARM buffer into a slot),
   with the fill as a dependency of the matmul. Gate: a new `kraken-bench` arm
   that fills a synthetic slot and compares it bit-for-bit against the host
   copy of the same bytes -- a fill that is off by one stride produces finite,
   fluent garbage, which is the exact failure mode
   `docs/AUDIT-expert-cache-next.md`'s PLE section records for a wrong width.
   Then one launch per layer on a model whose routed set does not fit, checked
   against `--expert-cache-mb` = the whole set (same text).
4. **The policy on the device, ported -- not redesigned.** Same LFU counters,
   same aging, same size-class arena order, single-threaded and deterministic
   like upstream's decide kernel (their reason is graph replay; ours would be
   that a nondeterministic decision makes every A/B a coin flip). Gate: replay a
   recorded routing trace through both the host policy and the device kernel and
   require identical slot assignments; a policy port that "reads about right"
   would show up as a hit-rate difference nobody could attribute.

## What not to copy

- **The plain LRU.** Rejected here on measurement, and the pinned-WARM arm is
  what carries the hit rate anyway.
- **Zero-slot substitution for an absent expert.** Illegal in this engine: every
  expert the unchanged router selects must be computed. A cache miss changes
  *where* the weights are, never *whether* the expert runs.
- **The per-layout-group budget split.** Our budgets are bytes per tier with
  size classes; splitting by host-expert bytes per layout group would re-derive
  a worse number for the same card.
- **The 32-token batch cap as a principle.** Upstream caps because a large batch
  touches many experts and evicts the decode set; here the prefill path already
  stages what it needs and the cap would be a different decision made for a
  different reason. Keep the *effect* (do not let a prefill evict the decode
  working set) without importing the rule.

## Why the Qwen3.8-Flash-Next file is the one that decides

`H:/OLLAMA-Models/GGUF/qwen3.8-flash-next-reap-288-Q4_K_M.gguf` -- the family
upstream benchmarks as unsloth's UD-IQ3_XXS / UD-IQ4_XS, whose Q4_0 test file is
93.7 GiB of which **65.4 GiB is experts** -- is here **78.04 GiB, 1224 tensors,
288 experts x top-10, 48 blocks, 2560-wide hyper-connection lanes** -- `kraken-inspect` reads it and the verdict is `refused`, which is
policy (`KRK_ARCH_ALLOW_EXPERIMENTAL`), not capability: every tensor class runs
and the graph reproduces the reference's 24/24 greedy tokens on this file.

Its experts can never be resident on 15.9 GiB, so it is precisely the workload
where our published-once table is never publishable and the per-expert loop is
the only path. Upstream's 4090 result on the same family -- 10.7 GB of cache,
77% hit, 40.7 tok/s -- is therefore the honest target for steps 1-3 above, and
the reason to build them in that order rather than adding another policy knob.

## References

- PR #29887, GPU expert cache for host-resident MoE weights, merged 2026-10-07:
  <https://github.com/ggml-org/llama.cpp/pull/29887>
- PR #30112, multi-GPU (layer split) support for that cache, 2026-10-08:
  <https://github.com/ggml-org/llama.cpp/pull/30112>
- Issue #29949, GPU-resident LRU with decisions and copies in the graph:
  <https://github.com/ggml-org/llama.cpp/issues/29949>
- Implementation, branch `lru-upstream-v3`:
  <https://github.com/ggml-org/llama.cpp/commit/c81901a571d5778d840d382d3b3789014cb55d05>
  (follow-up `094ea24` adds the tensor-split cases; `c81901a` warns that
  `-sm tensor` runs without pools)
- Local measurements this doc leans on: `docs/RESEARCH-REFERENCES.md`
  (the which-limit-binds table), `docs/AUDIT-expert-cache-next.md` (LFU over
  LRU, the pinned prefix, the 3.1x hybrid arm), `include/krk/moe_table.hpp`
  (the published-once rule and the `moe.route.fetch` figure).
