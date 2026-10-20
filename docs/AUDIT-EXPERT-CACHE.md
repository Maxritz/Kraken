# Expert cache + prediction: what it does, what it costs, what to change

Measured 2026-10-20, RX 9070 XT (15.9 GiB), plus the numbers already recorded in
`AGENTS.md` from the Tiel / Nemotron / 27B sessions. Every claim below names its
source; anything unmeasured is marked **proposal**.

## 1. What runs today

| piece | as built | where |
|---|---|---|
| residency | per-(layer, expert) slot in VRAM, byte-budgeted | `src/expert_cache.cpp` |
| policy | LFU **with aging**: hit counter per slot, pin at N hits, decay every `DECAY_PERIOD` acquires, victim = unpinned → lowest counter → oldest | `include/krk/expert_cache.hpp:18-33` |
| tiers | COLD = GGUF mapping, WARM = pageable host RAM (always-admit on first read, inclusive, eviction is `free()`), HOT = VRAM | `expert_cache.hpp:35-56` |
| **prediction** | **none on the VRAM path.** `ExpertCache::prefetch_layer` fills **WARM only** (`expert_cache.cpp:977`, called from `engine.cpp:5096`/`5802`); the HOT copy is materialised synchronously when the router asks | measured by reading the call sites |
| ranking input | `<model>.krakenexperts.json`, written by `--expert-scan` (per (layer,expert) hit counts from one forward pass), consumed at load by `--expert-warmup 1`; `--expert-adapt` folds a run's own routing back in (opt-in) | `main_cli.cpp` |

So the "hot set" is: an **offline, single-prompt ranking**, loaded once at
startup, filled **hottest-first**, then maintained by LFU. There is no
ahead-of-need **VRAM** fill, which is the flaw that matters.

## 2. What it costs (measured)

OLMoE 1B-7B, same binary, routed bank = 3.7 GiB, budget 3.7 GiB:

| arm | prefill tok/s | decode tok/s | HOT | promotions | read |
|---|---|---|---|---|---|
| no index, no warmup | 48.6 | 98.4 | 82.0% | 921 | 101.51 MiB/tok |
| index + `--expert-warmup 1` | **530.8** | **117.4** | **100.0%** | **0** | **0.00** |

The 10.9× is entirely "the bytes were already in VRAM". Nothing about the
kernels changed.

Tiel 35B-A3B, 22.1 GiB routed against a 12 GiB card (recorded in `AGENTS.md`):

| pool budget | alloc/cell | promotions | decode |
|---|---|---|---|
| 11.95 GiB (auto, card full) | **30 ms** | 1360 | 3.1 tok/s |
| 8.56 GiB (2 GiB floor) | 0.27 ms | 1712 | **17.8 tok/s** |
| 7.68 GiB (3.26 GiB floor) | 0.25 ms | 1850 | 16.6-18.7 tok/s |

and the promote split on a cold WARM: `read 161 | room 248 | alloc 196 | xfer 650`
of 1094 ms — **78% of the miss cost is alloc + H2D, i.e. HOT-side work**, not the
file read. That is exactly the part `prefetch_layer` does *not* move off the
critical path.

## 3. The flaws, in the order they cost throughput

1. **No HOT prefetch.** The VRAM copy is demand-driven, inside the layer, so
   `alloc + xfer` (0.85 of the miss) is serial with compute. A WARM hit is not
   free either. *Fix (proposal):* promote layer L+1's planned set during layer
   L's compute into a second ring; promote *only* what the policy would pin, and
   keep the demand path as the fallback. *Experiment:* Tiel at 8.56 GiB budget,
   interleaved pairs, `KRK_EXPERT_PREFETCH_HOT=0/1`, decode tok/s + `xfer` ms.
2. **The budget is decided before the KV cache and workspaces exist**, so it can
   name every byte free at that moment and make the allocator thrash (30 ms per
   cell, above). A post-warmup clamp exists; it does **not** re-run when the KV
   grows with context length. *Fix:* clamp at first use of each new context size.
3. **The ranking is one prompt deep.** The OLMoE index is `36 positions,
   full-forward, 1024 ranked pairs`; a 512-token index would rank the true hot
   set. `--expert-adapt` exists but is opt-in and per run. *Fix:* default-on
   adapt with exponential decay, persisted; `--expert-scan` over a corpus
   (a few hundred positions, several prompts) rather than one question.
4. **Only the chosen top-k is promoted.** The router's logits rank all 64/288;
   promoting the top-k′ (k′=2k) that are already in WARM costs a copy and buys
   hit rate on the next token. *Fix (proposal):* `KRK_EXPERT_LOOKAHEAD=k′`,
   measure hit share vs promoted MiB.
5. **One streaming ring for dense weights ⇒ a drain between layers.** The
   measured two-buffer pipeline (`AGENTS.md`: 2.2 → 3.0 tok/s, +36.4% on the
   27B) still drains `be_->sync()` before the next layer because the publish is
   TO the default stream. *Fix:* ping/pong rings (the second ring removes the
   wait; correctness rule unchanged).
6. **Pins are earned by traffic, not by the index.** A pinned expert needs 24
   hits, so the first 24 tokens of every run re-earn what the JSON already knew.
   *Fix:* seed the counters from the index (a rank ⇒ initial count).

## 4. The large models, which is where the gap actually is

**Qwen3.8-27B (`qwen3.8-flash-next-Q4.gguf`, arch `qwen35`, 64 layers,
15.61 GiB of weights, one MTP block `blk.64`)** — measured today:

```
[stats] load  9798 ms, 11.84 GiB uploaded of 15.61 GiB weights (1.4 GB/s)
[stats] prefill 7 tok 2751.5 ms = 2.5 tok/s
[stats] decode  16 tok 5285.9 ms = 2.8 tok/s (352.4 ms/step)
```

That is **dense weight streaming**, not the expert cache: the model does not fit
beside the KV cache and workspaces, so ~4 GiB re-reads every token. Levers, in
order of expected gain: (a) raise residency with the promotion path (measured
1.2 → 3.0 tok/s on the 27B dense when promotion + two-buffer staging landed);
(b) keep attention/norms resident and stream only `ffn_*` (attention is on the
critical path every token); (c) **use the MTP block** the file ships
(`--mtp-spec` already exists) to halve the number of forward passes; (d) int8 KV
to free VRAM for weights.

**DeepSeek-V4-Flash (57 GiB, `deepseek4`)** — cannot be benched at all:

```
kraken: cannot run this file ('deepseek4'). multi-head latent attention plus
compressed-attention tensors are not implemented. uses latent KV compression
(MLA: attn_kv_a/attn_kv_b) ... uses quantization type #26
```

llama.cpp runs the same file at 5.99 tok/s decode (CPU-shaped, `AGENTS.md`), so
"llama is faster" here is "llama implements the arch". MLA + dequant #26 is a
prerequisite feature, not an optimisation.

**Why Strata looks 10-100× better on the same class of file**: its published
row (782 tok/s @4K prefill / 30.8 tok/s decode, 35B-A3B) is *with* `--spec 4`,
MTP and int8 KV, on a pack whose expert tier is **resident and warm** — the cold
tier is exactly what `AGENTS.md` measured costing 16.5 vs 35.8 tok/s here. And
its 27B-class numbers are not a 27B dense model streaming 15.6 GiB per token.
The comparable Kraken numbers are the ones with the tier warm: 35.8 tok/s decode
/ 558.8 tok/s prefill on the 256-expert 35B with `--expert-warmup 1`.

## 5. What to build, in order

1. **HOT prefetch of the pinned set** (flaw 1). Biggest, cheapest: it moves the
   78% of the miss that is alloc+xfer off the demand path. Gate: a
   `kraken-bench --expert-prefetch` arm plus the Tiel A/B above.
2. **Seed pins from the index + default-on adapt** (flaws 3, 6). Nearly free.
3. **Re-clamp the budget when the KV grows** (flaw 2).
4. **Top-k′ lookahead promotion** (flaw 4).
5. **Second streaming ring** (flaw 5).
6. **MTP on the 27B** (section 4c) — already implemented, just not wired to the
   files that carry a block.

## 6. The hot ring, implemented and measured (2026-10-20)

Item 1 of the plan above is now in the engine: `ExpertCache::set_pin_budget` +
`admit_hot`, called from `Engine::pin_hot_experts` between the warmup and the
pool clamp.

**What it is.** A second VRAM budget that pinned experts are charged to instead
of the LRU pool's. `over_budget()` subtracts `pin_bytes_` from `bytes_` when a
ring is configured, so a pinned expert no longer shrinks the window the traffic
still has to earn; the backend's physical owner budget is `budget_ +
pin_budget_`, so the allocator sees one cache and not two. With no ring
configured (`KRK_EXPERT_PIN=0`, or any caller that never sets one) the old
arithmetic is exact -- `test_expert_cache_policy` case 4 pins that a pin still
never overrides the physical limit -- and case 10/11 cover the ring: pinned bytes
leave the pool's budget only when a ring exists, `admit_hot` admits exactly the
hottest prefix that fits and counts nothing as a hit, and `clear()` empties it.

**What it is not.** It does not make more VRAM exist. The clamp after it sizes
the pool from what the ring left, so the split is a partition: whatever the ring
takes, decode loses as pool. That is the whole result below.

**Measured, Tiel-Coder-35B-A3B-MTP-APEX** (256 experts, top-8, 40 layers, 22.1
GiB routed, 7682 MiB of device expert memory, `--bench --ctx 512 -n 64`,
interleaved order-balanced pairs, identical text in every pair under
`--greedy -n 64`, md5 `3baee4232881`):

| ring | prefill tok/s | decode tok/s | verdict |
|---|---|---|---|
| off (`KRK_EXPERT_PIN=0`) | 38.3 / 39.7 | 19.7 / 20.3 | baseline |
| 4350 MiB (auto 1/3 of usable) | 41.9 (**+5.5%**, 5/5) | 18.7 (**-7.8%**, 5/5) | prefill win, decode loss |
| 2174 MiB (auto 1/6) | 42.4 (**+10.8%**, 5/5) | 19.3 (**-2.1%**, 4/5) | the default |
| 2048 MiB (manual) | 42.4 (**+5.5%**) | 19.7 (-0.6%, noise) | same shape |

So prefill is consistently better and decode is neutral-to-slightly worse: a
static pin set displaces pool slots that the LFU order ranks better for the
single-row path, while a 55-row prefill touch has to survive a chunk's worth of
evictions and the ring protects it. The default is therefore 1/6 of the usable
card (`KRK_EXPERT_PIN_MB` overrides, `KRK_EXPERT_PIN=0` is the A/B arm).

**The number that says it is not the main lever.** Both arms report the same
`3326 acquires | HOT hits 66.4% | WARM hits 33.6%`: at this shape a third of
the corpus fits, so the LRU already finds the hot core and a pin merely
re-partitions it. The cost is the 33.6% that are *promotions* (~1128 per 64
tokens, 2.2 MiB each, ~40 MiB/token H2D against a 3.6 GB/s host tier) and the
per-promotion alloc/room walk the audit's flaw 1 describes. The ring removes
churn from the hot core; it does not remove the promotions, which is why item 1
alone cannot close a 2x gap. Items 3-5 (re-clamp, top-k' lookahead, streaming
ring) attack that, and the index quality does too: Tiel's index is 16 positions
of full-forward, so its "hot" prefix is a weak estimate of the set a long
generation actually re-selects.

**A trap this cost.** `KRK_EXPERT_PIN_MB=2048` on a run whose warmup had already
filled VRAM to its reserve found "free beside the floor" at 0 and skipped the
ring entirely -- the *same command* admitted 5325 MiB on one run and nothing on
the next, depending on whether the warmup finished its VRAM stage or hit its
time ceiling. Sizing the ring from the CARD (`total - floor`, then a share) and
letting `admit_hot` evict unpinned slots is what makes it deterministic; those
slots are in WARM, so the eviction is a promotion, never a file read.

## 7. Re-clamp and top-k' lookahead: the promotions do not move, and the reason is the floor

Items 2 and 4 of the plan are implemented. Both are measured, and the
measurement is the interesting part: **on Tiel the miss rate cannot be made to
fall, and buying a lower miss rate with a fuller card loses 40% of decode.**

**What landed.**

- `ExpertCache::reclamp_vram_budget(free, floor)` re-derives the pool's budget
  from a live reading -- unpinned resident + whatever is free beside the floor --
  and only ever lowers it (`Engine::reclamp_expert_budget`, called once per
  prefill and every 16 decode rows). Init's clamp is one measurement taken
  before the run; the KV tier promotes COLD->HOT as the context grows, so the
  bytes it named can be gone by the time decode is deep. The arithmetic is on
  the cache, not the engine, so `test_expert_cache_policy` case 12 drives it
  without a device: a reading with nothing free beside the floor clamps 96 -> 48
  bytes, a reading with room does **not** raise it, and the third expert then
  evicts instead of growing into the card. On this box at `--ctx 512` it is a
  no-op (`kv: flat: whole cache resident`, free beside the floor = 0, so cap ==
  budget), which is what it should be: a guard for the run where KV *does* grow.
- Top-k' lookahead (`KRK_EXPERT_LOOKAHEAD=<k'>`) and the staged plan
  (`KRK_MOE_LOOKAHEAD=1`). The router's whole row is on the host, so the experts
  ranked k+1..k' cost nothing to read; they are staged only when WARM already
  holds them, so a guess costs a copy and never a file read. The old switch also
  had its condition **inverted**: the comment said the batched lookahead is for
  `n > 1` (a verify pass knows its union a row early) while the code ran it for
  `n == 1`. It is `n > 1` now -- and `n == 1` is opt-in with the rest.

**Measured, Tiel-Coder-35B-A3B-MTP-APEX, greedy, `--bench --ctx 512 -n 64`,**
three interleaved passes per arm, same 55-token prompt:

| arm | decode tok/s | promotions | HOT hits |
|---|---|---|---|
| baseline | 22.3 / 21.4 / 23.4 | 3620 | 84.9% |
| staging (`KRK_MOE_LOOKAHEAD=1`) | 23.0 / 20.3 / 23.6 | 3630 | 84.9% |
| staging + k'=16 | 22.6 / 22.4 / 22.3 | 3660 | 84.9% |

Identical within noise, and the hit rate does not move at all. MTP on the same
file is worse (`--mtp-spec --draft-tokens 2`, greedy: **9.2 tok/s**), so the
"amortise the promotions over 2-3 verified tokens" idea does not pay here
either. Both switches are therefore **off by default**, documented with these
numbers rather than shipped on.

**Why the miss rate cannot fall -- the floor sweep.** The pool already holds
3471 of 10240 experts and the policy gets 84.9% of acquires out of that, which is
better than the resident share. The obvious move is to raise residency by
lowering the VRAM floor that the budget must leave free. Three interleaved
passes, `KRK_VRAM_FLOOR_MB`:

| floor | decode tok/s | promotions | HOT hits | resident slots |
|---|---|---|---|---|
| 3261 MiB (default) | **22.1 / 23.7 / 22.9** | 3620 | 84.9% | 3471 |
| 2048 MiB | 13.7 / 16.3 / 17.2 | 3087 | 87.1% | 3860 |
| 1024 MiB | 11.3 / 13.2 / 13.0 | 2787 | 88.3% | 4208 |

**The best hit rate is the slowest arm, by 2x.** 450 more experts cut promotions
by 23% and raised the hit rate by 3.4 points, and decode fell from 22.9 to 12.5
tok/s. The expert-path timers do not even see it -- `decode: room 291 | alloc 207
| xfer 324 | promote 823 ms` is *lower* at floor 1024 (721 ms) than at 3261
(823 ms), so the missing seconds are allocations outside the expert path:
activations, logits and the per-step workspaces pay the driver's eviction cost
once the card has no headroom. That is the same pathology `AGENTS.md` records at
30 ms per pool cell, seen from the other end -- and it means the miss rate is
the wrong objective function near the card's limit. Raising the floor is no
better (4096 MiB: 20.2 tok/s, 3986 promotions; 4800 MiB: 19.7, 4391), so 3261 is
the top of the curve, not an arbitrary constant.

### Ceiling 1, measured: where a decode step's time actually goes (unprofiled)

The host table is unreachable without the profiler, and the profiler adds ~22 us
per device op -- so the two spans that matter were being compared under two
different multipliers, one of which waits on the device and one of which does
not. `KRK_TIME_HOST=1` is now its own switch (host timing, no device recorder),
and `moe.route` / `moe.route.fetch` / `moe.route.rest` / `moe.experts` are
spans inside `Engine::moe_ffn`. Mellum2.1, 28 layers, decode only, ctx 512,
32 tokens at 41.0 tok/s (23.7 ms/step):

| span | ms/layer | calls | ms/token | share of step |
|---|---|---|---|---|
| `moe.route` (whole router block) | **0.827** | 868 | 23.2 | **98%** |
| &nbsp;&nbsp;`moe.route.fetch` -- the readback | **0.523** | 868 | **14.6** | **62%** |
| &nbsp;&nbsp;`moe.route.rest` -- top-k, plan, prefetch, experts | 0.307 | 868 | 8.6 | 36% |
| &nbsp;&nbsp;&nbsp;&nbsp;`moe.experts` -- the per-expert loop | 0.213 | 868 | 6.0 | 25% |
| router GEMM (enqueue only) | ~0.005 | 868 | 0.1 | <1% |

**The readback is the step, and it is a drain, not a copy.** `KRK_ROUTER_2X=1`
issues the same readback a second time, immediately after the first: by then the
queue is retired, so that copy is pure round trip. It costs **0.077 ms** against
the first one's 0.523. So ~85% of the fetch is *waiting for this layer's GPU
work to retire*, and the WDDM round trip -- the part a device-side router would
remove -- is 0.077 ms/layer, **2.2 ms/token, about +5%** on this model.

**What that means for "move routing to the device".** Doing it naively (top-k
and the plan on the GPU, ids still read back because the expert cache needs them
to resolve a pointer on the host) buys those 2.2 ms and no more; the 14.6 ms is
the wait itself, and it stays. The wait only disappears when the layer stops
round-tripping through the host *at all*: device top-k, a device-built plan, and
expert weights addressed through a **device pointer table** so no per-expert
host pointer and no `acquire()` sits between the routing and the GEMMs. That is
a new kernel family (`gemv`/`gemm` reading matrices through an indirection, the
`ggml_mul_mat_id` shape) plus a residency table the cache publishes to the
device, and it cannot be bit-identical: the softmax moves to the device, so the
expert weights change in the last bits. The honest expected ceiling is the GPU
time the step already has -- 8.3 ms of compute per token, ~±20%, i.e. ~120 tok/s
-- not the 300 a dense model would reach, because 224 acquisitions and 1,526
launches remain.

The measurements are kept as arms: `KRK_TIME_HOST=1` (host table alone),
`KRK_ROUTER_2X=1` (the round-trip control), `KRK_ROUTER_SYNC=1` (the explicit
drain before the copy, which is **not** the cost -- the copy is).

#### The numerics cost of device softmax, measured (`KRK_DEVICE_SOFTMAX=1`)

`kernels/router.hpp` is the real device softmax (warp-packed max, `expf`, tree
sum, normalize), and the arm uses its probabilities for the selection — so the
text generated is the device path's text, not a comparison done in a comment.
The host result is still computed and diffed against it:

| model | tokens | rows | elements that differ | max abs dP | text |
|---|---|---|---|---|---|
| Mellum2.1-12B-A2.5B | 24 | 784 | 36,862 / 50,176 (73%) | **1.19e-07** | identical (`169e7778…`) |
| Mellum2.1-12B-A2.5B | 64 | 1,904 | 89,800 / 121,856 (74%) | **1.19e-07** | identical (`46d63629…`) |
| Qwen3-30B-A3B | 24 | 576 | 61,553 / 73,728 (83%) | **4.77e-07** | identical (`e2e26c92…`) |
| Qwen3-30B-A3B, coherence arm | 24 | -- | -- | -- | **1 coherent, 0 not** |

**Short runs lie, and the stress arm is why this section was rewritten.** At 24
and 64 tokens both models printed byte-identical text, which reads as "the drift
is below the decoder's resolution". At 512 it is not:

| arm | tokens | first difference | what changed |
|---|---|---|---|
| Mellum2.1, greedy | 512 | **char 235** | a newline/token shift in a repetition loop; 1267 vs 727 bytes of output |
| Qwen3-30B-A3B, greedy | 512 | **char 1464** | "capital of **Ethiopia**" became "capital of **Yemen**" |
| Mellum2.1, `--temp 0.9 --seed 7` | 128 | **char 314** | different text from a fixed seed |
| Qwen3-30B-A3B, `--temp 0.9 --seed 7` | 128 | **char 111** | same seed, different tokens |

Drift magnitudes over those runs: Mellum **1.49e-07** max |dP| over 14,448 rows
(73% of elements differ), Qwen3-30B **4.77e-07** over 24,768 rows (81%). So the
softmax is not "nearly the same": one ulp of probability is enough to change a
greedy token within ~100 characters, and 88 tokens of agreement was luck rather
than a property. The device-vs-CPU coherence check still passes (1 coherent, 0
not) — **it is a coherence gate, not a bit-identity gate, and it cannot see
this**.

**The full matrix (2 models x 3 prompts x greedy 256 + sampled 128 at seeds 7
and 13 = 18 arms):**

| model | greedy 256 (p1/p2/p3) | sampled 128 (p1/s7, p1/s13, p2/s7, p2/s13, p3/s7, p3/s13) |
|---|---|---|
| Mellum2.1 | 235 / **identical** / 166 | 314 / **98** / 183 / 156 / 142 / 137 |
| Qwen3-30B-A3B | **identical** / 366 / 126 | 111 / 291 / 239 / 339 / **identical** / 152 |

**Earliest divergence found: char 98** (Mellum, sampled, seed 13). 15 of 18 arms
diverge; the three identical ones are prompt/seed pairs that simply had not hit a
one-ulp argmax yet at that length — the 512-token run of Qwen3-30B p1 was
therefore identical at 256 and forked at 1,464, which is the shape of this: not
a probability of divergence per arm, a time-to-fork that is short on some inputs
and long on others. There is no safe length.

The consequence for the dispatch is that device softmax is not a drop-in: a
rewrite that runs it must either accept different text or reproduce the host's
`exp` bits, which a GPU cannot. That is why the top-k/plan stage keeps host
normalization of the k weights and a strict bit-exact gate — the same
bit-exactness the softmax itself cannot offer.

#### The numerics policy (decided 2026-10-21)

**The host router stays the default. The device router is an opt-in, and it
becomes the default only when it buys something measurable — which today it does
not.** The decision is arithmetic, not taste:

- the readback is 0.523 ms/layer and a *ready* copy is 0.077, so ~85% of it is
  the layer waiting for its own GPU work. The part device routing removes (the
  round trip) is **2.2 ms/token, ~5%** while the expert dispatch still needs a
  host pointer per expert;
- the part that is worth having — no per-layer host dependency at all — needs the
  device pointer table and a grouped expert GEMM, which is the next stage and is
  not built yet;
- the price is text that differs from every previous build (measured: first fork
  at char 111-1464). Paying that price for ~0% today would be a silent model
  change with no matching win.

So: `KRK_DEVICE_SOFTMAX=1` / `KRK_DEV_PLAN=1` stay arms; the build's default text
is unchanged; and the flip condition is explicit — **flip when the dispatch's
end-to-end gain is measured above the PCIe/launch floor, and record the drift
the flip costs in the same note.** A default that changes tokens is acceptable
for a measured architecture win, never as a side effect of plumbing.

#### Stage 1 of the dispatch: device top-k and a device plan (`KRK_DEV_PLAN=1`)

`kernels/router.hpp` now also holds `router_topk_kernel` (k-pass scan, tie keeps
the lower id -- the host's rule, prescribed rather than inherited from a sort),
`router_weights_kernel`, and `moe_plan_kernel` (count, prefix sum, stable
(expert-major) fill in token-then-slot order). The arm builds all three on the
device, downloads the ids and weights, and **compares them against the host's
result before using anything**; a difference leaves the layer on the host plan.

Measured on Mellum2.1, 48 greedy tokens, 1,344 layers compared:

| quantity | result |
|---|---|
| expert ids that differ | **0** of 10,752 slots |
| weights that differ | 1,030 of 10,752, every one by **2-4 ulp** |
| layers that ran the device plan | **1,210 / 1,344** |
| generated text vs the host path | **identical** (`958a8163…`), strict gate |

The ids are exact because the selection is comparisons only. The weights are
not, and the cause is specific: the only difference left is `1/ksum` -- the
device's reciprocal is approximate, so 8 numbers per row land a few ulp off. With
a 1-ulp tolerance the arm ran 1,266/1,344 layers and **changed the text at 48
tokens**, which is why the shipped gate is bit-exact and the tolerance was
removed rather than tuned. The fix is not a comparison: download the k ids and
the k selected probabilities (64 bytes), normalize and scale those eight numbers
**on the host** exactly as today, and upload them for the plan. That keeps every
weight bit-exact while still removing the full-row readback and the O(n_expert*k)
host scan.

**This arm is a verification arm and it is 33% slower** (28.2 against 42.3 tok/s,
2 pairs) -- it runs the host path as the reference *and* the device path. That is
the point of it: it proves the pieces agree before anything stops being
duplicated, and the proof is cheap only until the duplication is removed.

One structural note the arm also proved: it engages only on the plain softmax
path (`!router_sigmoid`, no pre-bias). A file with either keeps the host
activation until the dispatch covers them, and that is a `false` return, not a
silent wrong answer.

#### Stage 2 of the dispatch: the expert table, and no readback at all (`KRK_DEV_PTR=1`)

Stage 1 removed the host *scan* but still downloaded the ids and weights to
verify them, and the cache still resolved a pointer per expert on the host. The
remaining host dependency is the one that costs: 0.523 ms/layer of `moe.route`,
of which the copy is 0.077 and the rest is the layer waiting for its own GPU
work to retire before the host may look at the row.

Stage 2 removes it with a **device pointer table**. `ExpertCache::publish_dispatch_table`
uploads a dense `[n_layer * n_expert]` array of `MoeDispatchEntry`
(`include/krk/moe_table.hpp`): three device pointers per expert plus their shapes,
their types and their row strides -- everything a dot product needs, as a value.
`moe_table_kernel` then runs a whole MoE layer as ONE launch per token: it reads
that token's k ids and weights, looks each expert up in the table, computes
gate/up + silu + down, and writes the weighted routed sum. No ids, no weights, no
plan offsets and no acquire() reach the host.

**The table is a promise about residency, so it is only published when the
promise can be kept**: `whole_set_resident(n_layer, n_expert)` must hold, i.e.
every slot of the routed set is in VRAM, so no promotion and no eviction can
rewrite a pointer while a kernel reads it. Anything less publishes nothing.
drop() and the promotion path both set `table_stale_`, and the engine compares
its published pointer against the cache's every layer, so a slot that moves
mid-run downgrades the layer to the host path instead of issuing GEMMs against
a freed address.

**Engagement is therefore a fact about the card, not about the arm.** On
`Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K` (128 experts x top-8 x 48 layers,
6,144 slots) with `--expert-warmup 1` the run prints:

```
expert warmup: ... VRAM +5952 experts (9695 MiB, 5952 slots); 4157 ms
              (stopped at the VRAM reserve: 3261 MiB kept free of 15.92 GiB)
device dispatch: not yet -- the routed set is not wholly device-resident
              (5952 of 6144 slots) (retrying while the tiers fill)
```

So the arm stays off on this box for that model, the text is unchanged, and the
reason is in the log rather than inferred. The publish is retried from `moe_ffn`
while the tiers fill, bounded to 8 attempts, because a cold cache never has the
set at the end of init -- the experts load lazily as routing reaches them.

**The kernel is verified without a model, at the real geometry**
(`kraken-bench --moe-dispatch`, 6 cases, exit status gated): the reference is the
per-expert path itself -- gemm(gate), gemm(up), `silu_mul` at the row length with
its row scale, gemm(down), and an f16-rounded weighted add in expert order.

| case | result |
|---|---|
| k=1, q4_k | **IDENTICAL** |
| k=2, same expert twice | **IDENTICAL** |
| k=2, q4_k / q6_k / f16 | 1-2 f16 ulp on 5-10% of elements (slot vs expert order) |
| k=4, q8_0 | 1-2 f16 ulp, worst 2.6e-3 relative |
| decode geometry (4096 x 14336, 45 KiB shared) | **IDENTICAL** |

Three traps, all of which produced a wrong conclusion before they were fixed:

- **The down projection must be rounded to the activation type before the
  weighted add.** Keeping the f32 dot left the sum a fraction of an ulp from the
  path it replaces -- measured as exactly 1 f16 ulp (1443 against 1444) on 14%
  of elements, which reads as "same arithmetic" and is not.
- **`acc + alpha * v * scale` is not three multiplies and an add.** The compiler
  contracted one into an FMA, whose single rounding differs from the host's two.
  The kernel now uses `__fmul_rn`/`__fadd_rn` explicitly.
- **A control that swaps two IDENTICAL matrices proves nothing.** `weights()`
  restarts its RNG per call, so every synthetic matrix of a shape came out the
  same and the gate/up swap reported "indirection NOT PROVEN" for a kernel that
  was bit-identical to the reference. The harness now salts each matrix, and the
  control swaps gate against up (which always exists) rather than two experts
  (which a k=1 selection does not have).
- And one harness ordering rule: the table's `upload` may publish on the pool's
  own stream, so a dispatch issued immediately after it can read the OLD table.
  The engine publishes once at init and uses it layers later, where this cannot
  bite; the arm drains before its indirection re-run.

**Not measured, because it cannot run here:** the end-to-end decode rate with the
dispatch engaged. It needs a corpus the whole routed set of which fits beside
the VRAM floor, and no model on this box meets that at 128x48 experts -- the
closest is 5,952 of 6,144 slots. The arm's flip condition (the one written above)
still stands, and now it has a kernel and a table to flip.

### The fix: the floor is enforced where memory is taken

The re-clamp was right to fire and wrong about why. Measured both ways
(`KRK_RECLAMP=0`): **1.8-3.0 tok/s with it off against 26.5-31.0 with it on**,
three pairs, no overlap. A pool that fills its 11,246 MiB plan ends with nothing
free and every later allocation pays the driver's eviction cost -- the same
pathology `AGENTS.md` records at 30 ms per pool cell. So the budget cut is
protecting the run; it is the *plan* that was wrong, because it is in requested
bytes and the card also holds memory the copy path took.

What was missing is that the clamp runs *after* the damage. `alloc_pooled` now
refuses growth that would leave less than `vram_floor_bytes()` free, asking the
driver at the allocation site (only once the pool is past half its budget, so a
run that never nears the floor pays no query). The cache serves those experts
from WARM instead, which is the same bytes from a slower tier rather than a
missing expert.

| arm | decode | VRAM free at end | desktop |
|---|---|---|---|
| floor 512 MiB (old behaviour) | 21.5 / 22.7 / 22.4 tok/s | **850 MiB** | video pauses |
| floor 3,261 MiB (default) | **41.6 / 41.5 / 41.3 tok/s** | **3,238 MiB** | fine |

**+87%, and the text is byte-identical** (md5 `169e77789ebacec31c1d22a675516f9b`
both arms, 24 greedy tokens). Leaving the card alone is not a cost paid for
tidiness: near-zero free VRAM makes the allocator the bottleneck, so the arm
that a user would call "the polite one" is also the fast one. This is the second
time this repository has measured that inversion (see the 30 ms/cell table).

**What that leaves.** For a routed set twice the card the only real lever is
bytes moved per token, and it is bounded by the H2D link: the load line reports
1.88 GiB in 553 ms (**3.4 GB/s**) and the promotion path 3.8-4.4 GB/s, i.e.
~570 us per expert, which is 68% of a 50 ms decode step. Pinning WARM does not
change it (`KRK_HOST_PIN_MB=2048/4096`: 4.1-4.9 GB/s), and neither does batching
the copies. So the next thing to try is not a cache policy but a *smaller
promotion*: keep the expert in VRAM in a narrower form than the file's quant
(a Q4_K expert re-quantised to Q2_0 for the resident copy halves the miss bytes),
or accept the shape and stop benching 22 GiB routed sets on a 16 GiB card.

## 8. Mellum2.1 (1,792 experts, 10.99 GiB routed): three separate ceilings

Measured 2026-10-20 on `G:\More-models\Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf`
(28 layers, 2304 embd, 32/4 heads, 64 experts top-8, expert FF 896, Q8_0).
The engine loads it, runs the tier on it, and reaches **prefill 18.0 tok/s,
decode 23.8-28.6 tok/s** (ctx 2048 / 512). Those are far below what a
2.5B-active Q8 model should do on this card, and the profile says why: **the
GPU is idle for three quarters of a decode step** (`Compute 0: 23.84% avg,
72.17% max` per-process engine utilization, sampled every 50 ms, unprofiled).

| what | measured | where it goes |
|---|---|---|
| expert cost | **6.28 MiB** (WARM 11,246 MiB / 1,792) | matches 3 x 2304 x 896 at Q8_0 |
| dense weights / KV / ws | 1,069 / 112 MiB @2048 / 24 MiB | ledger, `allocator live` agrees to 1 MiB |
| decode step | 35 ms at 28.6 tok/s, of which GPU compute 8.3 ms | host-side, not the card |
| ops per decode step | **1,526** (`--profile`, one step) | ~54/layer: 8 experts x 6 launches + attention |
| prefill | 2,829 ms for 51 tokens, of which **promote 1,812 ms** | 1,184 experts = 7.4 GiB read from the file, once |
| promotions | 1,540 us/expert (4.0 GB/s) | `[stats] transfer` |

**Ceiling 1 -- the router is on the host, so no two layers overlap.**
`moe_ffn` computes the router GEMM on device and then calls
`be_->download_f32(router_host_, ...)` for its `[n, n_expert]` row: a blocking
D2H that orders behind every kernel already queued. That is **28 pipeline
drains per token**, and each one costs the queue's tail latency rather than the
work in it -- 35 ms of step for 8.3 ms of GPU time is exactly that signature.
`KRK_ROUTER_SYNC=1` adds an explicit drain on top and is not the cause; the
copy itself is. llama.cpp does not have this ceiling because it selects top-k
on the device, so the expert plan never comes back to the host. The fix is the
one llama.cpp uses: a top-k kernel that writes the selection to device memory,
and a plan built there. Until then the MoE decode rate is bounded by host round
trips, not by the model or the cache.

**Ceiling 2 -- 54 launches per layer, one expert at a time.** 8 experts x
(gather, gate/up group, silu, down, scatter) is 48 launches a layer where a
grouped-expert kernel would issue one or two for the whole layer's set. At the
measured ~5 us per call that is 1,526 x 5 = 7.6 ms of the 35 ms step, and
`gemm` at 56.5 GB/s (net, profiled) against a 640 GB/s card says the per-expert
shapes are too small to saturate anything.

**Ceiling 3 -- prefill must read the routed set once.** 51 tokens touch 1,184
of 1,792 experts, and prefill spends 1,812 ms promoting them at 4.0 GB/s. That
is inherent while the routed set does not fit; it is the same 570 us/expert
this document already measured, and it is why a first prefill after a cold WARM
is 18 tok/s while a second one is not.

### The re-clamp, and the 7,000 MiB it cannot name

The pool plans 11,246 MiB (the whole routed set), then
`Engine::reclamp_expert_budget` cuts it to **7,248 MiB at the first prefill**
because it measures 850 MiB free beside a 3,261 MiB floor. Two things were
wrong with that:

1. **The message was a guess.** It said "the budget planned at init names bytes
the KV tier has since taken"; the KV tier took 112 MiB. A reader chasing that
sentence looks at KV, which is 1.4% of the difference.
2. **The arithmetic cannot see whose bytes they are.** `cap = live + room`
where `room = free - floor`, and it only ever lowers -- so a measurement taken
while the pool was growing is a permanent cap.

The ledger now checks itself against the allocator: `device_live_bytes()`
counts what this process has outstanding through `alloc()`/`release()`, and the
line prints it as `[allocator live 8454 MiB in 3744 buffers, largest 230]` next
to `unaccounted 7000`. That is the proof the missing bytes are **not** this
engine's: weights + kv + experts (1,155 x 3 slices = 3,465 buffers) + ws is
8,454 MiB to the megabyte, and the driver still reports 7,000 MiB gone. It is
not the WARM tier (`--expert-warm-mb 1024` and `11246` give the same 7,000), not
`HIP_VMEM_MANAGE_SUPPORT` (same with it unset), and not per-allocation rounding:
`tools/alloc_probe.hip` allocates 3,465 x 1,972 KiB, fully written, and the
driver charges 2,097,152 B each -- 239 MiB of waste, not 7,000. It scales with
the **bytes promoted** (7,000 MiB against 7,430 MiB promoted, ~1:1), which points
at the copy path's bounce buffers rather than at the pool.

Arms added with this work, all of them permanent:

| switch | what it does |
|---|---|
| `KRK_RECLAMP=0` | turns the re-clamp off -- the A/B for "is my budget being cut wrong" |
| `KRK_VRAM_TRACE=1` | ledger before/after every multi-row forward, plus the trace line inside the clamp |
| `[allocator live ...]` | always on; the allocator's own total beside the device probe |
| `tools/alloc_probe.hip` | the driver's per-allocation charge, outside the engine |
