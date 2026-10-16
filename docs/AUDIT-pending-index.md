# Pending-work index after the 2026-10-11 pass

This indexes [TODO.md](TODO.md), [TODO-ANALYSIS.md](TODO-ANALYSIS.md),
[PERF-PLAN.md](PERF-PLAN.md), and [DIAG-decode-cpu.md](DIAG-decode-cpu.md).
It is a triage index, not a claim that every historical measurement or remote
host was revalidated. The full historical entries remain in those documents.
No commits, pushes, deployments or sibling-project edits were performed.

## Settled in this pass

| Item | Current evidence |
|---|---|
| C1 full NeoX fused QK-norm | Pair-owned rotation fixes query-rounding mismatch; device probe 1280/1280 exact; three hashes unchanged; 18 interleaved 8B outputs identical |
| C2 Spark short-context drift claim | Five specified raw prompts pass CPU/device coherence; old drift not reproduced |
| C2 GPU split-threshold determinism | Five fused-off split runs and unsplit run identical at 160 tokens, through pos 168; full long CPU arm timed out, 133 matching picks only |
| Expert WARM batched prefetch | Reservation-byte budget, duplicate IDs and iterator lifetime fixed; red 38/45 -> green 45/45 |
| Expert trace analysis | Distinct-key distance separate from inter-access distance; explicit inclusive LRU replay; per-layer active sets; 10 constructed selftests |
| T14 / P3 documentation | WMMA tile 32x64x128, transient dequant/LDS staging, pageable WARM clarified; qwen35 supported-model claim already present |

## Correctness and testing still pending

- **C3 / G13 / G15 / T25 OLMoE:** shared-path wrong text versus independent
  llama.cpp oracle; explicit architecture entry missing. CPU/device agreement
  alone cannot settle it. Need the exact model and bounded oracle run.
- **C4 / G14:** host floating-point behavior changes when HIP is linked;
  reproduce current counts and bisect compiler/runtime/link initialization.
- **C5 / F9-F28 DFlash:** proposer rows remain diffuse/zero acceptance;
  inspect exact checkpoint pairing, feature norms/capture semantics, masks,
  positions and upstream reference logits. Count proposed/verified/never-verified
  work and remove feature download/upload round trips only after equivalence.
- **Long Spark CPU coherence:** 160-token scalar arm timed out (120s).
  All 133 observed picks matched; no full long CPU pass asserted.
- **TSAN:** listen descriptor race, reported GDN test lifetime/UAF, stale
  sanitizer baseline. An atomic descriptor alone does not settle close/reuse
  ownership; design shutdown lifetime then verify under sanitizer.
- **T13:** permanent fast-path numerical/text equivalence coverage beyond the
  new attention probe; speculation is tested in the suite already, not the
  bench gate. Do not call the existing self-draft coverage missing.
- **RDNA2 qwen35 reproducibility and int8 harness:** gfx1031 confirmation,
  Q6_K dispatch timings and gfx1201 all-zero probe readback remain unverified.
- **Model corpus:** refresh inventory and bounded smoke/coherence across
  runnable files, including the two ambiguous device/CPU-return-code records.

## Performance and residency still pending

- **M0 / T7 / T26-T27 (three of four landed 2026-10-11):** physical
  budget/ownership, exact active-set leases and async known-set staging are
  implemented and measured; **one-layer route hints with separate pollution
  budgets remain the open piece.** Staging is default-off because the
  measurement says it is not a win at a constrained budget (the step is 75%
  promotion; the copy is twice the reader window) -- details, the overlap probe
  and every A/B in [AUDIT-active-expert-staging.md](AUDIT-active-expert-staging.md).
  Read-vs-allocation-vs-copy bottlenecks still differ by model/budget; the
  allocation term is now measured and mostly removed (a pending-event query, not
  hipMalloc: `alloc` 618.4 -> 182.5 ms, decode +11.3% at 16 MiB), and the slot
  lease has a device-side gate. Full details and the Edge0/Strata concept audit:
  [AUDIT-expert-cache-next.md](AUDIT-expert-cache-next.md).
- **M1 / T4 / T5:** device routing plan and fewer per-expert upload/gather/
  scatter launches; graph replay needs stable slots and dynamic-plan safety.
- **M2:** remaining projection/elementwise folds, preserving per-matrix dtype,
  sidecar scales and f16 rounding. C1 is closed; generic launch-tax work is not.
- **M3:** candidate sampling and logits-copy paths need current measurement:
  device candidate support exists (see backend and bench gate), so this is
  not a blanket missing implementation. Preserve tail-mass/filter correctness.
- **M4 / M5 / T2:** Q5/Q6 decode ALU work, WMMA dequant staging/producer-consumer
  pipeline, small-row occupancy and split-K policy on actual gfx1201 workloads.
- **M6 / T10 / T22-T24:** gfx1031 LDS/barrier remainder, matched-length
  llama.cpp baselines and hardware timing. Remote host availability was not
  probed in this pass; don't label it currently online or offline.
- **Laguna:** multi-shard pull readers; separately report prefill expert-path
  costs; compare exact same topology with ROCm oracle; draft-tier warmup
  resource warning/guard. Sharing a self-draft cache is not a speed proposal:
  autoregressive self-drafting was already measured slower.
- **T3:** current CPU MoE dispatch/throughput on matched hardware; the old
  cross-card comparison is not an optimization acceptance criterion.

## Capability gaps still pending

- **qwen4exp:** named registry refusal exists. Actual gated/fused attention,
  SSM, QSA, hyperconnections and PLE execution is absent.
- **MLA / deepseek2 / deepseek4:** absorbed latent attention/cache contract,
  model scaling/RoPE and compressed-attention requirements; GLM-4.7-Flash
  was inspected this pass and still refuses for MLA. Better expert residency
  cannot by itself make those architectures runnable.
- **Gemma2/3 windows, gemma4:** per-layer attention/window geometry, input
  embeddings, projection sharing versus cache-sharing semantics.
- **MTP / DFlash2 / DSpark:** extraction/recognition is not forward execution;
  generic verify loop exists and self-draft floor is fixed. Head-family forward
  semantics and selector contracts remain separate work.
- **Recurrent speculation / Tail-Replay:** checkpoint/rollback contract for
  recurrent and convolution state.
- **LLaDA:** architecture/schema and diffusion generation loop.
- **Server / T8 / T19:** radix prefix reuse, persistence/journal/TTL, request
  cancellation, continuous batching and beam/tree/sampling speculation.
  HOT/WARM/COLD KV residency and per-layer packing are already implemented.
- **T9 / T15-T20:** per-card byte/slot profiles, expert-only quantization,
  kernel-native expert packs, sensitivity/calibration, useful ternary/BITCOS
  formats and autotuning. Lossy changes need explicit quality gates; no default
  enablement based only on storage size.
- **YaRN:** Laguna frequency handling exists; distinguish remaining generic
  scale/pattern coverage from a claim of zero implementation.

## Robustness / documentation / environment

- Linux engine-utilization counters, device-busy hints, remote fixtures after
  reboot, source/geometry provenance for every archived hash/benchmark.
- Per-layer trace bytes/model/request identity and explicit truncation/error
  reporting: current trace has none, so target/drafter streams can alias.
- ~~Physical arena accounting suspected underflow and event-failure
  interior-free risk~~ **repaired 2026-10-11**: individually owned size-class
  cells, `committed = live + reusable`, trims that free free cells only, a guard
  fallback that frees base allocations only, and a target+draft owner sum. Pinned
  by `kraken-bench --gate`'s expert-pool gate and the WARM-tier group (56/56).
- README 9B/recurrent benchmark and verification workflow coverage, remaining
  stale historical tables/counts/line numbers and inventory updates.
- ~~Loaded HIP runtime 7.16.26332 differs from build SDK 7.16.26331~~ **pinned
  2026-10-17**: the SDK's own `amdhip64_7.dll`/`amd_comgr.dll`/`rocm_kpack.dll`
  from `G:/ROCM10RT-gfx1201/bin` now sit beside the binaries in `build-hip/`
  (the Oct 6 copies were a newer runtime staged over them), the CLI header
  reads `HIP 7.16.26331` with no warning, and `build_check.sh` reports
  **0 failed, 0 warnings**. Quiet-machine performance reruns remain a
  prerequisite to claiming speed improvements.

## Excluded negatives

Do not reopen historical-rank seeding (lost), block-zero skipping (no whole
zero blocks), default CPU experts (3.1x loss), self-draft-for-speed, or larger
SIMT tiles that worsened occupancy/bank behavior without a new falsifier.
Never borrow Edge0's missing-expert zero contribution or routing replacement.

## Evidence

[Raw results and protocols](traces/todo-processing-2026-10-11.txt): tests
2916/2916; baseline, bench, build checks rc=0; coherence 3/3; runtime warning;
C1 timings **NO DIFFERENCE**, not a speedup. Cache repair and analyzer validation
are correctness/diagnostic outcomes, not throughput improvements.
