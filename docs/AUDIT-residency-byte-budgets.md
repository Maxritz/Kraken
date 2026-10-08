# Audit: byte budgets spent by a slot count

An audit of the two residency managers — `ExpertCache` (MoE experts) and
`KvTierCache` (KV planes) — for the defect that was just fixed in the expert
cache: **a budget denominated in bytes being spent against a count**, because
some per-unit size was assumed uniform.

The rule this audit applies:

> A count is a legitimate trigger **only when the thing counted is guaranteed to
> be one size**. Otherwise the comparison has to be in bytes, and a count may
> exist as a *report* (it must then be labelled an estimate) but not as a *limit*.

Method: read every budget/limit comparison in both paths and classify it, then
take each assumption to real files on disk. F1–F5 are the original pass; F6 and
F7 were found while verifying the F3/F4 fixes, from a KV-tier A/B whose arms
disagreed with each other.

## Verdict table

| # | site | compared | verdict | exposed by |
|---|---|---|---|---|
| F1 | `ExpertCache`, all six gates | bytes | **sound** (fixed) | — |
| F2 | `KvTierCache` slot arithmetic | count from `budget / layer_bytes_` | **fixed by construction** — the layout packs, and WARM covers the plane | per-layer KV widths, `nemotron_h_moe` |
| F3 | `Engine` KV charge | `n_layer`, not the KV-carrying layers | **fixed, verified** | `nemotron_h_moe` — was 8.67x overstated |
| F4 | `Model::load` KV geometry | global max KV width | **fixed**, and the `gemma4` verdict now names the feature | `gemma4` family |
| F5 | `hot / 2` + one `kv_dim_` for K and V | two planes assumed equal | fine today (MLA is refused) | latent-KV models, if implemented |
| F6 | `KvTierCache` eviction outcome | spilled and dropped in one counter | **defect, verified; fixed** | SmolLM2-135M at a HOT+WARM smaller than the cache |
| F7 | `KvTierCache::cold_path` | one filename namespace for two planes | **defect, verified; fixed** | any `--kv-cold-dir` run |

## F1 — the expert cache: six gates, all six in bytes

Now byte-exact, so this is the reference shape for the rest:

| site | comparison |
|---|---|
| `ExpertCache::full_for` (`expert_cache.cpp:212`) | `bytes_ + need > budget_` |
| `make_vram_room` (`:296`) | `vram_slots_ > 0 && over_budget(need)` |
| `over_budget` (`expert_cache.hpp:363`) | `bytes_ + need > budget_` **or** the explicit `slot_budget_` count |
| `make_warm_room` (`:314`) | `host_bytes_ + need > host_budget_` |
| `warm_admit` (`:362`) | `host_bytes_ + s.bytes > host_budget_` |
| `prefetch_layer` (`:731`) | `host_bytes_ + one > host_budget_` |

The only count comparison left is `slot_budget_`, and it exists because
`--expert-cache-slots N` *is* a count: the flag's documented meaning, now
enforced as one instead of approximated by `N x max_expert_bytes`. `capacity_`
is still derived as `budget_ / need` (`:600`) but it is a **report** and a decay
window, never a limit — and the engine passes an exact `slot_budget` whenever it
knows one (explicit flag, or a budget that covers the whole routed set).

## F2 — the KV tier's slot counts: the invariant is now enforced, not assumed

`KvTierCache::init` derives two counts from a byte budget:

```cpp
n_slots_      = hot_budget_  / slot_bytes_;    // device, slot = largest layer
n_warm_slots_ = warm_budget_ >= total_ ? n_kv_layers_ : warm_budget_ / slot_bytes_;
```

and `to_warm` admits while `live < n_warm_slots_`, where `live` counts *pages*.
The original audit called this sound *because every layer's plane was one width*
— true only if the model keeps KV on every layer at one head count. Two changes
remove the assumption instead of relying on it:

* **The flat layout packs the KV-carrying layers.** `len_[l]`/`off_[l]` are per
  layer and `total_` is their sum, so a layer that holds no KV occupies no bytes
  and the flat allocation is exactly the KV that exists. A slot is sized
  `slot_bytes_ = max(len_)`, so a layer is promoted into a slot big enough for
  it and writes its own `len_` bytes — mixed widths are supported, not assumed
  away.
* **A WARM budget that covers the plane means "cover every layer."** This is a
  correctness requirement, not an optimisation, and it is the fix for F6 below.

`n_warm_slots_ = budget / slot_bytes_` with the floor was itself the F6 trigger:
on a 30-layer model the auto policy came out two slots short of the working set,
and a page that fits nowhere is dropped, not recomputed.

## F3 — the KV charge now counts KV-carrying layers (fixed)

Was: `layer_bytes = kv_cap * kv_dim * as` and `kv_total = n_layer * layer_bytes
* 2`, with `kv_dim` the **maximum** `head_count_kv` — every layer charged a K
plane and a V plane, including the 46 of `nemotron_h_moe`'s 52 that have no
attention at all. `nemotron_h_moe`
(`G:/More-models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf`)
declares 52 layers, `attention.head_count_kv` = **0 on 46 / 2 on 6**, six layers
with a query projection (5, 12, 19, 26, 33, 42) and `attention.key_length` 128.

Measured, before and after, `--ctx 4096`, same file and box:

```
[info ] workspaces ready: chunk=256 ctx=4096 KV=208 MiB (load 2185 ms)   # before
[info ] workspaces ready: chunk=256 ctx=4096 KV=24 MiB  (load 1533 ms)   # after
```

**8.67x → 1.00x.** What the corrected number decides: `KvTierCache::init`'s
flat-or-tiered test is `hot_budget_ >= total_`, and the tiering path is up to
206x slower than fitting flat, so a padded `total_` could push a cache that fits
onto the tiered path; the auto WARM tier was sized against the phantom bytes as
well. Implemented as: `Model::layer_has_kv(l)` (the layer has a query
projection), `Model::kv_dim_at(l)`, `Model::kv_layer_count()`, and the engine
building `kv_len[]` per layer and charging `sum(len) * 2`.

## F4 — per-layer KV width is representable, and unsupported features are named

The query geometry was already per-layer (`Model::n_head_at`, because laguna runs
48 heads on its full layers and 72 on its windowed ones); the KV geometry was
collapsed to the model maximum, and `Model::load` compared every attention
layer's `wk.n_out`/`wv.n_out` against it. A model whose *KV* heads vary among the
layers that carry KV therefore failed with

```
layer N attention tensors disagree with the declared head geometry
```

which reads as a malformed file. Fixed in two places:

1. `Model` keeps the per-layer `head_count_kv` array (`cfg_.n_head_kv_layer`,
   `n_head_kv_at(l)`, `kv_dim_at(l)`); the geometry check compares each layer
   against **its own** width. The global `n_head_kv` is still the max, which is
   what anything model-wide still wants.
2. The `gemma4` verdict (`src/arch.cpp`) now lists `per-layer
   attention.head_count_kv` among the features it does not implement, so the
   unsupported feature is named as a feature. The verdict reads:

```
per-layer input embeddings, shared-KV layers, two attention geometries
(global + sliding window), per-layer attention.head_count_kv and the
attention-logit softcap are not implemented
```

The three local files that carry the shape — `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`
(KV {2,8}), `gemma4-coding-Q6_K.gguf` ({1,8}), `mtp-gemma-4-26B-A4B-it-Q4_0.gguf`
({2,8}) — are all refused earlier for other reasons, so this remains latent; it
is a constraint on gemma4 support rather than a live refusal.

## F5 — one `kv_dim_` and a `hot / 2` split for two planes

`engine.cpp` hands each plane `hot / 2u`, and `ws_k_`/`ws_v_` are both sized
`C * kv_dim * as` — K and V are assumed identical in width. Still true for
everything the engine runs, so the split is exact, not approximate. Listed
because it is the next assumption a latent-KV model (MLA: one shared compressed
plane, absorbed V) would break, and `arch.cpp` already names the MLA tensors
while `deepseek2`/`deepseek4` are `ArchSupport::No`. **F7 is the same mistake in
miniature and it was live** — two planes, one filename — which is the argument
for treating "two planes share one number/namespace" as a defect class and not a
latent note.

## F6 — a dropped page and a spilled page were one counter (fixed)

`evict_slot` demotes in order: WARM, then COLD, then nothing. The third branch
is a **drop** — `tier_[owner] = kAbsent` — and `ensure_hot` then falls through to
`fill0`, which zeroes the plane. Nothing recomputes it. Six lines of comment in
`kv_tier.hpp` claimed the opposite ("cheaper to recompute than to rewrite"); no
code did.

The drop and the COLD spill incremented the same counter (`evictions_to_cold`),
so a run that lost a page on every step reported a *working disk tier* — and
kept printing fluent text.

Two changes:

* `KvTierStats::evictions_dropped` is its own counter, summed over both planes
  like the others, and the end-of-run line prints it separately with what it
  means:

```
[stats ] kv        DROPPED 1556 page(s) with nowhere to put them: their KV is
gone and their layers were zero-filled. THIS RUN'S OUTPUT IS WRONG -- raise
--kv-warm-mb or set --kv-cold-dir.
```

* **The WARM capacity is derived from the geometry, not from the budget.** The
  auto budget was `overflow = kv_total - hot`, which lost the difference to slot
  rounding: `--kv-hot-mb 1 --ctx 512` on SmolLM2-135M (30 KV layers, 0.19 MiB
  per plane) came out **2 HOT + 26 WARM = 28 slots for 30 layers**, so two
  layers could never be resident and pages were dropped on every pass. Fixing
  that specific shortfall was not enough — any `--kv-warm-mb` below the working
  set reproduced it — so the capacity is now *derived*: with no cold dir,
  `init` pins it at `n_kv_layers_`, the count `layer_bytes` implies.

  That count is not a heuristic, it is the number that makes an admission
  failure impossible. `to_warm` refuses when the live WARM pages reach the
  capacity, and at the moment of an admission the layer being promoted is still
  counted in WARM while HOT holds at least the page being evicted. So
  `|WARM| <= n_kv_layers_ - 1` and one more admission needs `n_kv_layers_`.
  Enforced that way, **no flag combination can leave the tiers short of the
  working set**: a smaller `--kv-warm-mb` is raised and the deviation is
  reported (asked MiB, applied slots, why). It is obeyed where a spill
  directory exists, because then a page that cannot reach WARM is on disk
  instead of lost. A `--kv-cold-dir` that cannot be *written* is probed once at
  load and treated as absent (and reported), since a failed `fopen`/`fwrite` in
  `to_cold` is a drop.

Measured on the final build, `-p "The history of computing is a history of
abstraction, from relays to vacuum tubes" -n 32 --greedy --ctx 512`, md5 of
stdout, 30 KV layers, 2 HOT slots:

| arm | evictions | dropped | md5 |
|---|---:|---:|---|
| flat (no tier) | — | — | `2456dfa5ba9e` |
| `--kv-hot-mb 1` (auto WARM) | 1916 ->WARM | **0** | `2456dfa5ba9e` |
| `--kv-hot-mb 1 --kv-warm-mb 64` | 1916 ->WARM | **0** | `2456dfa5ba9e` |
| `--kv-hot-mb 1 --kv-warm-mb 0 --kv-cold-dir DIR` | 1916 ->COLD (1860 back) | **0** | `2456dfa5ba9e` |
| `--kv-hot-mb 1 --kv-warm-mb 0` (WARM raised to 30, reported) | 1916 ->WARM | **0** | `2456dfa5ba9e` |
| `--kv-hot-mb 1 --kv-warm-mb 0 --kv-cold-dir NOT_WRITABLE` | 1916 ->WARM | **0** | `2456dfa5ba9e` |

The auto arm was `94cf3cc56de2` with 132 pages dropped before this fix — a
fluent-text divergence from the flat cache, with nothing in the report to say
so; the `--kv-warm-mb 0` arm was `d6a7b98cc34d` with 1556 dropped. Every arm is
now byte-identical to the flat cache, and the two that had to deviate from the
caller's flag say so. The tier warnings print once: they used to fire per plane,
so every line appeared twice.

The proof is `scripts/kv_tier_geometry_check.sh`, which runs three arms per
model against the file's own header (`tools/kv_tier_geometry.py`) plus a decode
equality arm on the small ones; the checker's `selftest` replays 13 cases,
four of them constructed failures, because a harness that cannot fail proves
nothing.

## F7 — two planes spilled to the same file (fixed)

`cold_path(layer)` built `"<cold_dir>/kv_layer_<l>.bin"` with no plane in the
name. `Engine` owns two `KvTierCache` instances that share
`cfg_.kv_cold_dir`, so **K and V wrote the same 30 files**: the second writer
won, and a restore handed one plane the other's bytes.

This is invisible in the counters — the arm below reports a healthy
1916 ->COLD / 1860 COLD->HOT / 0 dropped and its text diverged anyway:

```
before:  1916 ->COLD, 1860 COLD->HOT, 0 dropped, md5 a2e087cbeada   (30 files)
after:   1916 ->COLD, 1860 COLD->HOT, 0 dropped, md5 2456dfa5ba9e   (60 files)
flat:                                                               2456dfa5ba9e
```

Fixed by giving `init` a `plane_tag` (`"k"`/`"v"`) that namespaces the file. Any
`--kv-cold-dir` run before this was quietly wrong; the flag is not the default
path, which is why it went unseen. An old spill directory is now orphaned rather
than reused, which is the safe direction.

## What to keep

1. A budget is spent in bytes. A count may be reported, never enforced, unless
   the counted object is one size by construction.
2. When a count *is* enforcement, the invariant that makes it exact has to be
   written down next to it — and preferably enforced by the layout rather than
   by careful bookkeeping. Packing the flat KV removed the uniform-width
   invariant from `KvTierCache` entirely.
3. **A loss and a migration are not the same event.** A counter that absorbs
   both makes a corrupt run read as a working one, because the wrongness and its
   evidence are the same number.
4. **Two of anything that share a namespace is a defect pattern** — two planes
   sharing one filename (F7), two planes sharing one `hot / 2` and one
   `kv_dim_` (F5), two planes sharing one warning (F6). Check every shared
   namespace when a second plane/instance appears.
5. `--expert-cache-slots` is the exception that proves the rule: a count that is
   a count because the user asked for one, enforced as a count, with the byte
   budget still doing the work in every other case.

## Verification of this round

`ninja` rc=0 / 0 `error:`; `scripts/build_check.sh` **0 failed, 0 warning(s)**
(`-DNDEBUG` in all 90 compiled `.cpp` TUs, 5 artifacts hash-matched after a
deliberate re-record); `kraken-tests` **2695/2695**; `kraken-bench --gate` rc=0;
`scripts/coherence_check.sh` **3 coherent, 0 not**; MoE acceptance
(`Qwen3-MOE-4x0.6B-2.4B-Q4_K_M`, 64 tokens, auto budget) **0 VRAM evictions**,
110/112 slots resident, room 0.0 ms, stdout md5 `a56e02b731fd`.

That md5 is also what the archived `dist/kraken/kraken.exe` (Oct 6, before any
of this) produces for the same command, which is the check that the layout work
changed no numerics. Note for the record: the `cf0408eb02` figure in
`docs/test-results.md` §10 does not reproduce against the stdout protocol now
documented there (banner + generated text) and did not match the pre-refactor
binary either; `a56e02b731fd` does, on both builds, and that section now records
it.
