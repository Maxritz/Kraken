# Kraken against `ENGINE_MASTER_SPEC.md` — what is actually missing

Source: [Maxritz/arxiv-ML-Techs / ENGINE_MASTER_SPEC.md](https://github.com/Maxritz/arxiv-ML-Techs/blob/main/ENGINE_MASTER_SPEC.md)
(1479 lines, 18 sections, 22 pages). This file maps its gap analysis onto the
code that is on disk **now**, because parts of the spec's "Kraken gap" notes no
longer describe this repository.

Read in full for this map: §1.1 (backends), §1.2 (MoE routing / expert cache),
§1.3 (KV management), §17 (caveats), §18 (how to use). §2–§16 were not read
line-by-line, so nothing below claims anything about them. Re-sync before
treating this map as complete.

## Method

Each row was checked against the tree, not against the spec's summary. A
capability counts as present only if a symbol exists for it; `grep` returning
zero hits across `src/` and `include/` is recorded as absent, with the pattern
used. Claims that the spec makes about Kraken are marked where they no longer
hold.

## §1.2 MoE routing, prediction and expert cache

The spec says: *"no visible hot-expert predictor, no prefetch pipeline … the
repo on disk has the verification harness and AMD HIP build plumbing."*

| spec item | status | evidence |
|---|---|---|
| Router: token-choice top-k, shared experts, unbiased weights | **present** | `src/engine.cpp` route block; bias shifts selection only, weights stay unbiased |
| Hash routing (DeepSeek-style tid2eid) | **present** | `src/dsv4_route.cpp`, `dsv4_hash_route` / `dsv4_hash_gather` |
| Hot-expert predictor | **present, contradicts spec** | `include/krk/moe_predictor.hpp` — `MoeRoutePredictor`: temporal + cross-layer candidates, per-layer precision EWMAs gating staging |
| Expert cache with eviction | **present, contradicts spec** | `include/krk/expert_cache.hpp` / `src/expert_cache.cpp` — byte-budgeted residency, **LFU with aging** (not LRU), lazy materialisation |
| Expert prefetch pipeline | **partially present** | staging driven by the predictor; no I/O thread pool or overlap measurement found |
| Commitment-weighted sizing (AcceptMoE) | absent | budget is a byte cap, not per-block sizing |
| Eviction with correction on miss (ExpertFlow) | absent | LFU/aging only |
| Cross-layer scheduler / lookahead > 1 layer | absent | predictor is request-local, previous-MoE-layer only |

### The determinism contract (§1.2) is the part worth acting on

The spec's L/P split — a logical cache that is model-visible and deterministic,
separate from a physical cache that may use any timing-dependent heuristic — is
the strongest idea in the section. Kraken already happens to respect its two
hard requirements:

* **The predictor never edits routing.** `moe_predictor.hpp` states it sees
  only the router's chosen ids and "never edits those IDs or their weights", so
  `L` is not steered by residency. This is the property the spec says is
  necessary for bit-identical paged vs resident runs, and it is already true.
* **No float atomics in the reduction.** Expert outputs are summed on the host
  in a fixed order, not in completion order.

Two gaps against the contract:

1. **Tie-break is not by expert id.** The router picks with
   `partial_sort(..., [&](i32 a, i32 b) { return row[a] > row[b]; })`
   (`src/engine.cpp`, router top-k). Ties therefore resolve by whatever order
   `partial_sort` produces rather than by lower id, which the spec requires
   (§1.2, contract item 1). It is deterministic for a given build, so this is
   not a reproducibility bug today — it becomes one the moment the top-k is
   parallelised or compared against another engine's tie-break. Fixing it
   changes which expert wins a near-tie, so it needs a logits gate, not a
   silent edit; it is left as the next MoE task rather than bundled here.
2. **`(n, beta, |P|/|E|)` is not printed next to any speed or hit figure.**
   Contract item 5 wants all three beside every number. The profiler prints
   ops/dev_us/us-op/%run/GMAC/s/%peak, none of which describe the cache
   operating point.

## §1.1 Cross-platform backends

| backend | status | evidence |
|---|---|---|
| CPU (scalar / AVX2 / AVX-512) | **present** | `src/backend_cpu.cpp`, `src/quant.cpp` |
| HIP / ROCm | **present (build plumbing)** | `src/hip/`, `KRK_ENABLE_HIP`; stubbed off in this configuration |
| Software virtual NPU | **present, not in the spec** | `src/npu/npu_offload.cpp`, 9 opcodes, `docs/NPU-CONSOLIDATED.md` |
| Vulkan, Metal, WebGPU, OpenVINO/Vino | absent | 0 files match `vulkan`, `metal`, `webgpu`, `openvino` across `src/` and `include/` |

Per the standing instruction to keep GPU backend design out of scope for now,
none of the four absent backends are proposed here. The spec's own caveat
applies anyway: Meganeura is 1.1–1.8× native, not a CUDA replacement.

## §1.3 KV cache management

| spec item | status | evidence |
|---|---|---|
| Chunked prefill | **present** | `tile_chunk()` in `src/engine.cpp`; chunk = batch GEMM |
| Prefix cache (reuse prompt KV) | **present** | `src/server.cpp`: "prefix cache: N of M prompt tokens are already in the KV" |
| KV tiering / offload | **present** | `src/kv_tier.cpp` |
| **Paged KV (block table, non-contiguous, cross-request sharing)** | **absent** | no `block_table` / `page_table` / `kv_page` symbols; the KV cache is a flat contiguous buffer ("flat: whole cache resident") |
| Continuous batching (many sequences in flight) | **absent** | no `continuous`, `n_parallel` or sequence-parallel path |
| KV compression tier (AnchorKV, INT2, FP8, SAKI…) | absent | no quantised KV path found |
| Radix prefix tree across requests | absent | `radix` hits are the radix *sort* in `backend_cpu.cpp` / `topk.hpp`, not RadixAttention |

Paged KV and continuous batching are the same missing piece seen from two
sides: without a block table there is nowhere to put a second sequence's
context, so batching is a prerequisite of paging, not an independent feature.
That pair is the largest single gap against §1.3 and the one that changes what
the engine is (single-request tool → serving engine).

## Disk-streamed tier (§12.4) and startup

| item | status | evidence |
|---|---|---|
| Weights mmap'd, never uploaded | **present** | `src/common.cpp` `mmap(PROT_READ)`; "emb = table[rows] (mmap'd, never uploaded)" |
| Expert weights lazy, byte-budgeted | **present** | `expert_cache.hpp` |
| Linux read-ahead tuning (spec: the biggest single startup win, 920s → 70s) | **absent** | 0 files match `readahead` |
| IO prefetch threads | absent | no thread pool around expert staging |

The spec is emphatic that read-ahead is mandatory rather than optional for a
disk-streamed engine; it is a small, self-contained change (a `posix_fadvise`
or `readahead(2)` call on the mapping at load) that has not been made.

## What this changes about the plan

1. The spec's §1.2 gap list is **wrong about this repository** — predictor and
   expert cache both exist, and the cache is already LFU-with-aging rather than
   the LRU the spec assumes. Any plan built on "add a predictor" is spent work.
2. The real §1.2 work is the two contract items above, both of which are about
   *reporting and tie-breaks*, not about new mechanisms.
3. The real §1.3 work is **paged KV + continuous batching**, which is a
   structural change to the engine rather than a kernel.
4. Read-ahead is the cheapest unclaimed item in the whole document.

Everything in §1.2/§1.3 above was checked by symbol; nothing here is inferred
from prose.

## Verification used for this map

```
./build/kraken-q80-odd          # 23 block counts, odd ones via the 256-bit path
grep -ril "vulkan\|metal\|webgpu\|openvino" src include   # → nothing
grep -rn "block_table\|page_table\|kv_page"  src include   # → nothing
grep -rn "continuous\|n_parallel"            src include   # → nothing
grep -ril "readahead"                        src include   # → nothing
```
