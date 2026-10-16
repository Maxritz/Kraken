# Stability and performance roadmap for constrained hardware

Date: 2026-10-11. This is an execution index across [TODO.md](TODO.md),
[PERF-PLAN.md](PERF-PLAN.md), [implementing-papers.md](implementing-papers.md),
[STATUS.md](STATUS.md), [MODEL-STATUS.md](MODEL-STATUS.md), and the plan/code audit
[AUDIT-plan-vs-code.md](AUDIT-plan-vs-code.md). It distinguishes a supported
loader from a verified model run, a correctness fix from a speedup, and a
published cross-engine number from a matched comparison. It is not a claim that
Kraken is already better than Strata or llama.cpp across models or hardware.

## Acceptance bar

1. **Load verdict:** every model/shard set is either runnable with an explicit
   capability record or refused with a precise architecture/tensor reason;
   never infer support from a family label alone.
2. **Correctness:** compare the device path against both the CPU reference and,
   for disputed architectures, an independent upstream oracle. Preserve exact
   prompt, context, tokenizer/template, backend, output hash, and run limits.
3. **Constrained-memory safety:** estimate weights, activations/workspaces,
   recurrent state, KV, expert HOT/WARM, pinned staging, and a live OS reserve
   before allocations. Recheck headroom at admission. If retained KV cannot be
   safely moved to WARM or COLD, stop generation with an error rather than
   zero-fill history or return fluent corruption. The OS can consume memory
   between samples, so this is an admission guard, not a hard OS reservation.
4. **Performance:** only claim a win from unprofiled, interleaved,
   order-balanced repetitions on the same device/model/prompt, with output
   identity and a null control. `--profile` is for counts/order, not wall-time
   claims. Compare Kraken/Strata/llama.cpp only at an identical workload and
   include format conversion/repacking and feature differences explicitly.
5. **Resource coverage:** keep a constrained-host matrix (CPU reference,
   small-RAM host, small VRAM, shared-display card, low-context, cold disk) and
   fail cleanly for insufficient resources; do not let test fixtures allocate
   according to the developer workstation's free RAM.

## P0 — correctness and machine safety

### P0.1 Live host RAM admission guard (implemented; OS race remains)

The expert tier checks the live host reserve in read-through admission, batch
prefetch, ranked warmup and even preload. KV WARM and COLD staging use the same
backend OS query; Linux reads `MemAvailable`, Windows uses `ullAvailPhys`, and
unknown availability fails closed for tiered staging. Checks include pending
batch allocations where the OS reading has not caught up. The policy is still
an admission guard rather than an OS reservation: another process can consume
headroom between a check and allocation.

KV refusal is end-to-end: `KvTierCache` preserves HOT ownership when neither
WARM nor COLD staging fits; `forward_core` preflights both planes before layer
writes; `forward`, plain prefill/decode, model-draft and DFlash paths propagate
failure as `FinishError` plus `GenerateResult.error`. Large rollback preflights
all K/V layers before zeroing any history. CLI reports failed generation, and
HTTP returns an internal error instead of a successful completion with truncated
or corrupted output.

Coverage now includes a controllable backend and a sparse-KV fixture: it lowers
available host memory after load, verifies failed admission leaves retained bytes
and the HOT owner intact, and exercises generation refusal through the public
`GenerateResult`. Separate injections cover host reserve arithmetic, expert
read-through and batch refusal, and expert-budget startup refusal. The Linux
probe deliberately uses `MemAvailable`, avoiding the `freeram` false-starvation
case. See [vram_guard.hpp](../include/krk/vram_guard.hpp),
[expert_cache.cpp](../src/expert_cache.cpp), [kv_tier.cpp](../src/kv_tier.cpp),
and the `test_kv_tier_host_reserve_refusal` group in
[tests/test_kraken.cpp](../tests/test_kraken.cpp).

### P0.2 Failure atomicity and cleanup (partially implemented)

`KvTierCache::init` now retains the backend while `detach()` releases partially
created HOT slots, markers, WARM buffers and flat storage. A required HIP marker
failure is an init error, and injected allocation/marker failures assert that
already acquired resources are released. The CLI/server startup refusal paths
also avoid deleting a backend still referenced by the engine before unwinding.

Remaining: cover failures at each distinct allocation stage (including flat
allocation and WARM host allocation), and assert cleanup for both CLI and server
through process-level tests. Do not equate the current partial-allocation unit
arm with exhaustive cleanup coverage.

### P0.3 Model correctness holes

- **OLMoE**: current CPU/device self-coherence is insufficient because both may
  share the wrong model contract. Add named `olmoe` architecture entry and
  compare exact outputs/logits to independent llama.cpp HIP oracle.
- **HIP-linked CPU reference drift**: 2152/2154 versus 2154/2154 remains an
  unexplained correctness risk; bisect runtime/object initialization instead
  of treating its current counts as benign.
- **DFlash/MTP**: re-run the exact paired checkpoint after the corrected
  self-draft verifier; compare proposer rows and hidden-state capture semantics
  before claiming speed. Add constrained-resource load errors for drafter tiers.
- **TSAN**: settle HTTP descriptor ownership and reported test lifetime/UAF, then
  refresh the sanitizer baseline.
- Run with partial prompt chunks, context edges, split-attention threshold,
  recurrent resets, and failed KV page promotions; many corruption modes stay
  fluent.

## P1 — small-memory robustness

1. **Budget all host use together.** The RAM reserve must account for CPU backend
   weights/activations, mapped pages, expert WARM, KV WARM, cold spill/read
   staging, pinned DMA staging, draft model, tokenizer/logit buffers, and
   allocator overhead. Avoid independently giving the full reserve to K, V,
   target, draft, and expert caches. Prefer a process-level host budget owner
   with named sub-budgets and an observable `committed / pending / available /
   reserve` line.
2. **Bound prefill memory by request shape.** Validate `--ctx`, `--chunk`, vocab,
   expert count and all byte multiplications before allocation. If workspace
   allocation fails, return actionable recommendations (`--chunk`, `--ctx`,
   quantization, CPU vs GPU) without invalid free or process abort.
3. **Shard streaming:** build a per-shard read table for multi-file GGUF. Current
   pull path assumes a single mapping; Laguna-S's 3-shard 68 GiB set therefore
   misses the best read path. Validate pointer-to-file offsets against each
   shard's source pointer (not its mapping at the same guessed offset).
4. **Disk cache:** COLD spills need versioned metadata, model and geometry
   fingerprint, plane identity, size/checksum, atomic write/rename, disk-space
   preflight, and cleanup rules. A stale spill must be rejected, not restored as
   plausible wrong KV.
5. **Low-memory CI:** deterministic allocator failure and host-availability
   simulation should be first-class tests; add bounded RAM/VRAM plans to model
   smoke tests and report `NOT COVERED` separately from pass/refusal.

## P2 — win matched baselines before adding complexity

1. **Finish Q5/Q6_K decode ALU optimization** (the largest measured dense
   format gap: Q6_K 143 GB/s on cache-sized and 195 GB/s on DRAM-sized shapes).
   Use exact dequant oracle tests, both gfx1201 and gfx1031 gates, then
   interleaved end-to-end throughput and hash checks.
2. **Prefill WMMA staging pipeline**: producer/consumer or direct fragment
   dequant is a measured opportunity, but verify LDS bank behavior and exact
   row-32/128 results; avoid claiming based on a profiler-dominated number.
3. **Projection dtype walls and launch folds**: audit per-layer types and sidecar
   scale contracts. Do not force a common dtype or reorder f16 operations merely
   to reduce call count. Rank each fold with live order-balanced A/B.
4. **MoE first-touch**: keep tracing bytes/read/alloc/xfer/room by model and
   budget; do not enable active staging because the current overlap measurement
   is a negative, nor default CPU expert compute (measured 3.1x slower). Address
   Laguna ranked WARM vs VRAM promotions only if a candidate beats exact LFU on
   held-out request traces and live A/B.
5. **Matched baselines:** repeat Qwen3.8 35B, Laguna XS.2, Qwen MoE 0.6B and
   dense 8B on same card and protocol against Strata and llama.cpp. Strata's
   782 tok/s prefill figure and Kraken's 558.8 used different contexts/features;
   they are not an apples-to-apples verdict. Record quantization, KV type,
   speculation, warm state, format conversion, prompt and context.

## P3 — feature coverage from the papers and architecture plan

1. **Prefix cache + Tail-Replay** for hybrid recurrent models: token-keyed
   prefix identity; cache attention KV plus token IDs; replay a bounded recent
   suffix to reconstruct recurrent state. Start with the paper's actual
   algorithm and a quality/output gate, not a generic partial rollback.
2. **Paged/quantized KV** only with a kernel-visible page table, format metadata,
   scale fidelity, cold-tier integrity and byte-identical flat-cache tests.
   Self-Indexing Attention depends on an agreed transformed-key representation;
   don't build its index first.
3. **qwen4exp / MLA / DeepSeek / Gemma4 / LLaDA**: implement schema and
   architecture contracts incrementally, gated by real local files and an
   independent reference. Expert cache support is not architecture support.
4. **Server scheduling**: persistent prefix reuse, request cancellation and
   continuous batching follow only after per-request KV ownership and rollback
   are correct; use prefix-homogeneous grouping before RL scheduling.
5. **Research features**: RadixMLP, CHT, REA, MTP heads, and candidate selection
   each need a measured workload and correctness contract. Papers' reported
   speedups are not Kraken targets until reproduced locally.

## Evidence sources and authority

- [Pending index](AUDIT-pending-index.md) is the most recent cross-document
  triage; [TODO.md](TODO.md) retains historical details and measured negatives.
- [PERF-PLAN.md](PERF-PLAN.md) ranks kernel work but includes older baseline
  rows; use current traces and `AGENTS.md` gate commands before quoting them.
- [implementing-papers.md](implementing-papers.md) is abstract-level review for
  several papers, not independent implementation proof.
- [AUDIT-plan-vs-code.md](AUDIT-plan-vs-code.md) separates supported features
  from absent capabilities; its model inventory and loader verdict are the
  authority for “loads,” while device-vs-CPU/independent output gates establish
  “works.”
- [AUDIT-expert-cache-next.md](AUDIT-expert-cache-next.md) records why some
  attractive Strata/Edge0 concepts (LRU, speculation, staging, CPU overflow)
  are deferred or rejected on current measurements.

## Definition of success

“Better than Strata and llama” is not a feature checklist. For each model class,
Kraken must (a) load the intended files, (b) match a trusted oracle, (c) fit and
fail safely under a stated memory ceiling, and (d) beat a matched baseline on
latency, throughput, quality-per-byte, or usability. This roadmap makes each
claim earn that evidence rather than conflating aspirations with shipped
capability.
