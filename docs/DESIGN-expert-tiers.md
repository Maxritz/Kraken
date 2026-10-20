# Deadline-Tiered Expert Loop (DTEL) — design, with the gaps measured shut

Two phases, and they are different problems. **Phase 2** is the deadline
problem (Qwen3.6-35B-A3B class: 40 layers, 256 experts, top-8, hybrid
attention/recurrent, 22.1 GiB routed against 15.9 GiB of VRAM). **Phase 1** is
`Mellum2.1-12B-A2.5B-Thinking-Q8_0` — 10.99 GiB routed, which no card holds but
VRAM+RAM does on every grid cell, so it needs the paging schedule without
needing the disk stage. §6 has its measured geometry.

Every constant below is measured on this box (RX 9070 XT, gfx1201, 15.92 GiB
usable) against `Tiel-Coder-35B-A3B-MTP-APEX` / `-UD-Q5_K_XL`, or derived from
that file's GGUF metadata. Numbers that are still inferred are labelled.

## 0. The measurements this design rests on

| quantity | value | how |
|---|---|---|
| expert residency cost, corpus average | **2.26 MiB** | 22,604 MiB WARM / 10,240 experts |
| expert cost, layer 0 | **2.703 MiB** (2,834,432 B) | `blk.0.ffn_{gate,up}_exps` Q6_K + `ffn_down_exps` Q8_0; matches the index file's `expert_bytes` exactly |
| dense (non-expert) weights | **1,920 MiB** (APEX) / **2,506 MiB** (UD-Q5) | `Model::upload_bytes()` in the ledger |
| KV, ctx 4096 | **80 MiB** = 20 KiB/token | ledger; only the ~20 attention layers of 40 carry KV (hybrid), so a dense-transformer formula over-counts 2x |
| KV, ctx 512 | **10 MiB** | ledger |
| workspaces, init | **88 MiB** | ledger, `ws_bytes_` accumulated in every `alloc` |
| device already taken at start | 2,251 MiB (one run), 4,408 MiB (another) | ledger start line — environmental, varies per launch |
| PCIe, expert promotion | 570 µs per 2.26 MiB (**3.8–4.4 GB/s**) | `[stats] transfer` |
| NVMe | 593 / 1,446 / 3,400 MiB/s at depth 1 / 4 / sequential | recorded, AGENTS.md |
| VRAM floor the budget must leave | max(V/5, reserve+1) = **3,261 MiB** at 16 GB | `vram_floor_bytes`; sweep in AUDIT-EXPERT-CACHE.md §7 |

**The two gaps that mattered are now closed, and both were inferences before.**

1. "Scratch is ~3.0 GiB" was a subtraction (`total − dense − KV − experts −
   free`), and it was wrong by >30x: the measured workspace total is **88 MiB**.
   The leftover was the driver's and other processes' VRAM (2.2–4.4 GiB at
   process start, varying between launches) plus `hipMemGetInfo`'s device-wide
   view, not anything this engine allocates.
2. The 2.26 vs 2.70 MiB "discrepancy" is a **per-layer difference**: layer 0 is
   Q6_K gate/up + Q8_0 down = 2.703 MiB; later layers are Q4_K down, so the
   corpus average is 2.26 MiB. A policy sized from layer 0 would over-provision
   20% of every tier.

Reproduce with `kraken -m <file> --bench` and read `vram at start` + the two
`vram ledger` lines (init and teardown).

## 1. Assumptions the standard cache makes

| # | Assumption | Standard design |
|---|---|---|
| A1 | residency unit is the whole expert | yes |
| A2 | a miss is served from RAM; disk is a fallback | yes |
| A3 | history (recency/frequency) predicts the future set | yes — this *is* LRU/LFU |
| A4 | serving a miss is synchronous and on the critical path | yes |
| A5 | one policy with constants fits every VRAM×RAM pair | yes |
| A6 | discrete GPU: the expert is copied or absent; the CPU is idle | yes, and only on a discrete card |

## 2. One violation per assumption

- **A1 (sub-expert residency): rejected, with arithmetic.** A GEMV touches all
  512 rows of all three matrices, so a block-level miss saves **0 bytes** and
  **0 miss rate**; the only unit that moves bytes is narrower precision, which
  breaks gate B. *Measure:* per-block tag hits on a 1-row forward (predicted
  100%).
- **A2 (disk as a scheduled stage).** Reads issued ≥1 layer early at depth ≥4;
  a demand read from disk is counted as a policy failure. *Gain:* RAM tier can
  shrink to ~2 GiB. *Cost:* 4×2.26 MiB of read buffers, 9 MiB. *Measure:*
  fraction of disk bytes requested before layer L−1 begins (target 100%).
- **A3 (predict from routing transitions, not recency).** A saturating
  transition table keyed by (layer, hash of previous selected set) → next set,
  512 KiB. *Gain:* a forward-looking value per resident expert. *Cost:* table +
  the risk that it is no better than the router's own near-tie set — which was
  **measured at +40 promotions and 0.0 points of hit rate** (84.9% both arms).
  *Measure:* per-layer predictor hit rate vs `KRK_DUMP_ROUTE=1` over ≥2048
  tokens.
- **A4 (decouple transfer from consumption).** Transfers are jobs with
  deadlines; compute proceeds on any expert whose bytes are ready. *Gain:*
  stall becomes max(compute, transfer) instead of the sum — 33 ms/token of H2D
  against ~25 ms of compute is the measured split. *Cost:* one staging slot per
  in-flight expert. *Constraint:* only *transfers* may reorder; the per-layer
  expert sum stays in fixed slot order or gate B dies. *Measure:* stall seconds
  per token, scheduler on/off, identical text.
- **A5 (policy shape follows the budget pair).** Every decision is
  `p(e)/bytes × B_tier/B_below` against a deadline slack; no per-tier tables.
  *Cost:* 200 ms calibration pass (PCIe rate, NVMe rate, alloc latency) + ~1 MiB
  state. *Measure:* tok/s variance across the grid vs a hand-tuned arm.
- **A6 (compute the miss on the CPU instead of copying it).** *Gain:* 2.26 MiB
  H2D → ~8 KiB of activations. *Cost:* measured **3.1× slower** in this engine
  with a per-layer barrier, and it breaks B if the CPU kernel's accumulation
  order differs. *Measure:* per-expert CPU time with the barrier removed.

## 3. Borrowed mechanism

Virtual texturing: a **fixed page table** over a virtual space, a **feedback
pass** that records what the last frame sampled, and **deadline-driven streaming
with a full request queue**. Mapped: tier map = page table; feedback = token
t−1's routing through the transition table; deadline = layer L's start; queue
rule = keep predicted requests in flight while bandwidth budget remains.

**Where it breaks:** a game may show a wrong mip for one frame; gate B demands
the paged run be bit-identical, so the LOD half is unusable. And feedback is one
token behind in a stream that cannot be re-rendered.

## 4. The design, and the grid

`Sv = (V − held_other − dense − KV − ws − floor) / cost`, with every term from
§0. Using `held_other = 2.2 GiB` (the quieter of the two measured starts),
`dense = 1.9`, `KV@4096 = 0.08`, `ws = 0.086`, `cost = 2.26 MiB`:

| V (nominal → usable) | floor | Sv @4k | Sv @32k (KV 0.64) | runs? |
|---|---|---|---|---|
| 16 GB (15.92 measured) | 3.18 GiB | **3,483** | 3,229 | yes |
| 12 GB (~11.9) | 2.38 GiB | **2,029** | 1,775 | yes |
| 8 GB (~7.8) | 1.56 GiB | **541** | 287 | yes, disk-bound |
| 6 GB (~5.8, e.g. 5600 XT) | 1.16 GiB | **158** | 0 | yes only at short ctx |

`Sr = (R − 20 MiB staging) / 2.26 MiB`, R = usable RAM after OS:

| R usable | 10 GiB | 16 GiB | 23 GiB | 37 GiB |
|---|---|---|---|---|
| Sr | 4,527 | 7,246 | 10,240 (full) | 10,240 (full) |

Coverage (VRAM + RAM slots, capped at 10,240) and the NVMe ceiling at depth 4
(1,446 MiB/s, ignoring compute):

| V \ R | 10 | 16 | 23 | 37 |
|---|---|---|---|---|
| 16 | 8,010 → 22% disk → 8.9 tok/s | 10,240 → full | full | full |
| 12 | 6,556 → 36% → 5.4 tok/s | 9,275 → 9% → 21 tok/s | full | full |
| 8 | 5,068 → 50% → 3.8 tok/s | 7,787 → 24% → 8.2 tok/s | 10,240 → full | full |
| 6 | 4,685 → 54% → 3.5 tok/s | 7,404 → 28% → 7.1 tok/s | 10,240 → full | full |

Predicted per-layer hit rate: **MEASURED 84.9%** at 3,471 resident slots on
16 GB (this box, 64 greedy tokens). Every other cell is **UNKNOWN** — the only
routing trace on disk (16 positions × top-8 = 128 selections/layer) cannot even
observe half of 256 experts per layer, and it predicts 100% coverage at 75
slots/layer where the engine measures 84.9%. Resolve with `KRK_DUMP_ROUTE=1`
over 2,048 greedy tokens.

Worst-case stall: **RAM miss 570 µs** (measured), **NVMe miss 1.6 ms** at depth
4, 3.9 ms at depth 1. Peak added RAM: 512 KiB table + 9 MiB read buffers +
4.5 MiB pinned staging. Peak added VRAM: ~1 MiB of page table and queues.

Gates: A low risk (kernels unchanged; per-layer diff, then longest common token
prefix). **B is the one at risk** — it breaks if transfers reorder the expert
sum, if the CPU arm is enabled on a gate run, or if any narrowed resident copy
is used; the design forbids all three. C compares against llama.cpp's own
placement, so score it on the longest common prefix, not on equality.

## 5. Flowchart and failures

```
load: geometry -> per-layer expert bytes (not layer 0's); calibrate B_pcie, B_nvme
      Sv, Sr from (V, R) minus dense/KV/ws/floor/held_other
      page table: (layer,expert) -> {VRAM,RAM,DISK,inflight}     no pins

per layer L, before its GEMMs:
  demand set D (actual top-8, on host)   deadline 0
  predicted set P(L+1) from transition table, deadline +1 layer
  admit while bandwidth_used + bytes/B <= slack(L), rank p(e)/bytes * B_tier/B_below
  evict argmin(forward value)                      <- never recency
  NVMe at depth >= 4, file order; demand disk read = POLICY FAILURE, reported

per (layer, expert) in FIXED slot order: wait(deadline) -> GEMM -> accumulate
end of token: update the transition table from observed routing
```

1. **Bursty prompt** (new code after a long chat): predictions go stale, every
   layer's set misses. 16 GB: 40 × 8 × 570 µs ≈ **180 ms** on the first token,
   then the table re-learns. 8 GB: 320 × 2.26 MiB = 722 MiB from NVMe ≈ **0.5 s**
   before the first token — the bandwidth budget must cap it, not thrash.
2. **Long context:** KV grows 20 KiB/token (1.28 GiB at 128k on the hybrid), so
   `Sv` falls; the 6 GB tier reaches 0 slots at 32k and must halve ctx or tier
   KV to host — decided at load from geometry, not discovered at allocation.
3. **Background app takes 4 GiB of VRAM** (browser, compositor): `held_other`
   rises to 4.4 GiB as measured, `Sv` drops ~1,000 slots, the disk stage becomes
   load-bearing. Keeping `floor` a hard invariant is what stops this becoming
   the allocator cliff AUDIT-EXPERT-CACHE.md §7 measured — where a *better* hit
   rate was 2× slower.
4. **6 GB card, Phase 2:** 158 slots at 4k, 0 at 32k, ~3.5 tok/s ceiling from
   NVMe alone. It runs; it is not useful for long context, and the design says
   so from the arithmetic rather than after the first token.

## 6. Phase 1: `Mellum2.1-12B-A2.5B-Thinking-Q8_0`

This file was "out of scope" in the first draft only because its geometry was
unknown. It is measured now — the engine loads it and runs the expert tier on it
(`arch mellum: not in the architecture table; read against llama defaults`, then
`28 layers, 2304 embd, 32/4 heads, hd=128, ff=7168, vocab=98304`).

| quantity | value | how |
|---|---|---|
| layers / experts / top-k | 28 / 64 / 8 | loader; **1,792 routed experts** |
| expert FF width | 896 | `ffn_*_exps` |
| expert cost | **6.28 MiB** | 3 x 2304 x 896 at Q8_0 (1.0625 B/weight, scales included); the WARM tally agrees — **11,246 MiB / 1,792 experts** |
| routed total | **10.99 GiB** | 1,792 x 6.28 MiB |
| dense (non-expert) weights | **1,069 MiB** | ledger |
| KV | **56 KiB/token** (112 MiB @2048) | ledger; 28 x 4 KV heads x 128 |
| workspaces | **24 MiB** | ledger (a third of the 35B's 88 MiB) |
| measured | prefill 18.0 tok/s, decode **23.8 tok/s**, ctx 2048 | `--bench --greedy -n 8`, 1,155/1,792 resident, WARM held 1,184 |

`Sv = (V − 2.2 − 1.04 − KV − 0.024 − floor) / 6.28 MiB`, coverage capped at 1,792:

| V (usable) | floor | Sv @4k | share resident | SR @10 GiB usable RAM | coverage @4k |
|---|---|---|---|---|---|
| 15.92 | 3.18 | **1,510** | 84.3% | 1,627 | **full** |
| 11.9 | 2.38 | **984** | 54.9% | 1,627 | **full** |
| 7.8 | 1.56 | **450** | 25.1% | 1,627 | **full** |
| 5.8 | 1.16 | **189** | 10.5% | 1,627 | **full (1,816 ≥ 1,792)** |

**The disk tier is Phase 2 only, and that is the result.** V+R covers all 1,792
experts on every cell at 4k as long as ~10 GiB of RAM is usable, including the
6 GB card — so Phase 1 is a *pure paging* problem: one substitution (a RAM
promotion instead of a file read) and the hit-rate work in §4 is what sets tok/s.
The remaining bound is PCIe: at 84.3% resident, 0.157 x 224 acquires = ~35
promotions/token x 6.28 MiB is 226 MiB/token, i.e. **~60 ms/token, a 16 tok/s
ceiling from the bus alone** — which is the same order as the 23.8 tok/s the
engine already measures with the tier in place.

**One open item this run found.** Its `--bench` arm exits **rc=2** with
`range bench: 0 silu, 1 experts-sum and 0 residual store(s) left f16 and were
clipped, and the storage exponent reached the clamp`. So the routed expert sum
overflows the f16 accumulation on this file (8 x 896 of Q8_0 output per layer).
The text it emits is fluent and the tier works, but the numbers are clipped and
the engine says so: the experts-sum accumulator needs to be f32 on this path
before the file is trusted, and `KRK_ALLOW_BIG`-style silence is not the answer —
the range counter firing is the correct behaviour.
