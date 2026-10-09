# AUDIT: the external research master plan against this tree

`NATIVE_LLM_ENGINE_RESEARCHED_MASTER_PLAN.md` (research cut-off 9 Oct 2026) read
against this repository at HEAD, and against what this machine has measured. This
is the plan's own Phase 0 done for it, so every row names the file that settles it
rather than restating the plan.

Method: `grep -rIl` over `src include tools scripts tests CMakeLists.txt` for each
capability, then reading the hits, because a capability's *name* in a comment is
not the capability. Where the plan and the tree disagree, the tree wins.

## 1. Status of the plan's five systems

| plan item | status here | evidence |
|---|---|---|
| native C++ core, native GGUF parser, no llama.cpp runtime | **done** | no `llama`/`ggml`/`FetchContent` anywhere in `CMakeLists.txt`; `src/gguf.cpp`, `src/model.cpp`, `src/quant.cpp` |
| architecture-correct graph incl. hybrid/recurrent blocks | **done for the families shipped** | `src/arch.cpp`, `src/verdict.cpp`, GDN path in `src/engine.cpp`; `docs/MODEL-STATUS.md`, `docs/model-inventory.md` |
| unsupported layers fail loudly, no silent zero-op | **done** | `src/verdict.cpp` + `kraken-inspect`, which answers from the loader's own tables |
| CPU reference as the oracle | **done** | `src/backend_cpu.cpp`; `scripts/coherence_check.sh` diffs device against `--cpu` per model |
| separate prefill and decode dispatch | **done** | gemv for rows==1, `gemm(wmma)` for prefill in `src/hip/kernels/gemm.hpp`; profiled op rows |
| fused elementwise / grouped projections | **done and still open** | residual fold, gate/silu fold, GDN 4-projection group, MoE expert pair (`+7.835%` this session) |
| low-overhead dispatch, graph capture | **missing** | no `hipGraph`/stream capture in `src` or `include`; every kernel uses the legacy default stream; the launch floor is measured (`docs/PERF-PLAN.md` §0c, §B1) |
| expert-granularity storage format (SSD-LLaMA pack) | **missing** | reads go through the GGUF mapping plus per-thread `FileReader` (`src/common.cpp`, `src/hip/backend_hip.hip`); no offline packer, no per-(layer,expert) manifest, no checksums |
| async read pipeline, bounded staging pool | **partial** | Windows `OVERLAPPED` reads (`src/common.cpp:292,314`) and a bounded pinned staging ring (`src/expert_cache.cpp:414`, `upload_paged`); no `io_uring`, no DirectStorage, no deadline-aware queue |
| three tiers HOT/WARM/COLD under byte budgets | **done** | `ExpertCache` (HOT=VRAM, WARM=pageable RAM, COLD=the mapping) documented in `include/krk/expert_cache.hpp:35-52`; `KvTierCache` for KV; `docs/ARCHITECTURE-TIERED.md`; the `[stats] kv tiered:` line |
| STAGE window for near-future experts | **missing as a region** | what exists is a ranked warm-up plus `--expert-warm-prefetch` filling WARM in one batched read (`src/engine.cpp:1471-1490`, `ExpertCache::prefetch_layer` `src/expert_cache.cpp:701`) — prefetch into *RAM*, not a VRAM window |
| routing-aware rebalancing (Mira) | **partial: telemetry yes, policy no** | per-expert traffic counters and a printed HOT/WARM/COLD split (`include/krk/expert_cache.hpp:223`); no rebalancing driven by it |
| slot state machine with generation/completion tags | **partial** | VRAM pins and WARM-as-`free()` eviction (`include/krk/expert_cache.hpp:45-52`); the path is a synchronous read-through, so there are no in-flight DMA generations to tag |
| every selected expert still executes | **done by construction** | no prediction in the routing loop; `KRK_DUMP_MOE` and `tools/expert_trace.py` exist to check it |
| own-boundary prefetch ladder (FATE → pre-attention → SeqMoE) | **rung 0-1 only** | ranked warm-up + batched warm prefetch; nothing adjacent-layer or pre-attention. The measurement tooling is here: `tools/expert_trace.py`, `scripts/ab_prefetch.sh`, `scripts/ab_slots.sh`, `scripts/trace_ab.py` |
| paged KV, block tables, refcounts, prefix reuse | **missing, and structurally resisted** | `src/engine.cpp:1` calls the design "paged-free"; `src/engine.cpp:965` says paging is incompatible with the current layout; no `page_table`/`refcount`/prefix-hash structure anywhere. The KV tier's invariant is built on that per-layer packed geometry (`docs/AUDIT-residency-byte-budgets.md`) |
| KV quantisation/compression (AnchorKV, KVFetch) | **missing** | no hits for either name; the gate it needs already exists as byte-identical stdout against the flat cache (7 decode arms recorded) |
| OpenAI-compatible server | **done** | `/v1/completions`, `/v1/chat/completions`, `/v1/models` (`src/server.cpp`); no embeddings endpoint |
| chunked prefill | **done** | chunked workspaces (`chunk=256` in every run's banner) and the `--chunk` path |
| continuous batching | **missing** | the only hit for "continuous" is prose in `include/krk/expert_cache.hpp` |
| speculative decoding | **done (`--draft`), MTP absent** | `src/dflash.cpp`; the 2.7x loss was a logits-copy regression and was fixed. `mtp` in `src/arch.cpp:113` is *shard recognition* for a DeepSeek MTP head, not a decoding path — the plan's "validate the existing MTP path" has no such path to validate |
| multi-GPU sharding | **missing** | every `shard` in the tree means a GGUF *file* shard (`src/gguf.cpp:96-116`) |
| Vulkan / CUDA / Metal / WebGPU backends | **missing** | zero hits. What §3.3 asks for in principle is done *inside* HIP: separate tuned paths per arch (`gfx1031`/`gfx1201`, `src/hip/kernels/gemm_dp4a.hpp` among others) |
| Safetensors importer | **missing** | zero hits |
| custom W4/W6/W8 formats | **missing** | the engine runs GGUF formats only, including NVFP4 and the two ternary layouts (`Q2_0` 128/34, `Q2_0_64` 48) |
| DQ prefill/decode-specific formats | **missing on both sides** | nothing in the tree cites the paper |
| RESOLVE-style kernel gates | **done in this repo's idiom** | `--selfcheck` (318-run injection sweep, 17-case selftest), `scripts/kraken_tests_isolation_check.sh`, constructed-failure tests, md5 determinism A/B; not formal equivalence |
| §5 benchmarking/acceptance policy | **done, and stricter** | hash-with-its-exact-protocol discipline (`docs/test-results.md` §10, the `cf0408eb02` case), interleaved A/B with an order-balanced median and a control, `kraken-bench --gate` |
| the plan's own Phase 0 audit | **already largely written** | `docs/PERF-PLAN.md` (ranked B0-B6, each proven or open), `docs/TODO.md` (P0/P1 plus closed negatives), `docs/MODEL-STATUS.md` + `docs/model-inventory.md` (the architecture matrix) |

## 2. Where the plan is behind the tree

- **BITCOS is already decided here.** `docs/compressed-expert-spec-review.md` §3.3
  works the case the plan leaves open — BITCOS is lossless with respect to an
  *already-ternary* tensor, and degenerates for anything else — and names the
  open items (`KRK_T3_5TRIT`, `KRK_T3_BITCOS`) and the correctness bug blocking
  them. The plan's §2.7 bullet is a summary of a deeper local analysis.
- **A related storage optimisation already has a recorded negative.**
  `docs/TODO.md:113`: "Block-level zero skipping for MoE expert paging — MEASURED,
  NO YIELD". The plan's "lossless compression must be benchmarked against
  decompression cost and actual storage traffic" is right, and this repo has the
  precedent rather than just the caution.
- **Its CPU-expert principle has a number here.** §3.6/§6 warn not to bind RAM
  experts to CPU execution; the measured cause is in `docs/test-results.md` §9 —
  a per-layer merge barrier, 3.1x loss, host arm inert when the budget holds the
  routed set.
- **§3.3's "RDNA2 and RDNA4 have separate tuned paths, not forced through one
  kernel" is already true** of the HIP backend. What does not exist is the other
  four backends the same section lists.

## 3. Where the plan is genuinely new, and worth taking

1. **The phase skeleton with explicit gates.** No document here has one
   (`grep -rlE '^## Phase|build order' docs/*.md` finds nothing). It is a
   reasonable spine for a project of this size, and the gates are the right shape.
2. **The paper list (§7).** Of the eleven primary-paper names I could test against
   `docs/`, only three appear anywhere (BITCOS, Edge0, RESOLVE); SSD-LLaMA, Mira,
   SeqMoE, FATE, AnchorKV, KVFetch, PagedAttention and Disaggregated Quantization
   have no local record. If one thing from this plan should be merged into the
   repo, it is §7 — with the caveat that its numbers are context, not targets,
   exactly as the plan's own §5 says.
3. **The expert-pack record format** (aligned per-(layer,expert) records plus a
   manifest of offsets, sizes, formats, checksums) and the slot state machine.
   Nothing like either exists; the local tier is a synchronous read-through with
   pins, which is simpler than the plan assumes. If a pack and an async path ever
   land, the state machine becomes the design.
4. **The predictor's measurement contract** — top-k recall, *useful* prefetches,
   wasted bytes, late arrivals — which the existing trace tooling can produce.

## 4. Three problems with the plan as written

1. **Its premise does not hold against this tree.** The four documents it says it
   consolidates (`ENGINE_MASTER_SPEC.md`, `IMPLEMENTATION_RECOMMENDATION.md`,
   `LLM_ENGINE_MISSING_COMPONENTS_ANALYSIS.md`, `MASTER_LLM_ENGINE_BUILD.md`) are
   absent from the repository, and it cites none of the repo's own documents
   (`PERF-PLAN.md`, `TODO.md`, `ARCHITECTURE-TIERED.md`, `MODEL-STATUS.md`,
   `test-results.md`, `AUDIT-residency-byte-budgets.md`). So "checked against the
   current Kraken repositories" cannot be verified here, and its Phase 0 duplicates
   work those documents already contain.
2. **It reads as a fresh build order for work that is already ranked and half
   done.** This repo's execution order is `PERF-PLAN.md`'s B0-B6 plus `TODO.md`'s
   P0/P1. Adopting the plan's phases *instead* would restart Phase 1 — kernel and
   launch work — from zero; three folds came off that list in one session
   (+2.867%, +6.135%, +7.835%), all of them items the plan describes only
   generically.
3. **Two of its steps cannot run where the work happens.** RDNA2 `gfx1031`
   timings need hardware that is not reachable from this machine (the kernels
   compile and `kraken-bench --gate` runs over them; the `probe_decode.pd1031*.txt`
   logs in `docs/` came from elsewhere). And §5 asks for per-model task accuracy,
   perplexity and safety-drift reporting for every new format — there is no such
   harness here, so that gate would be scoped as new work, not applied as policy.

Also worth stating: it does not mention **Strata**, the sibling engine this repo
already compares itself against (`docs/STRATA-VS-KRAKEN.md`), where some of the
same decisions are already argued — and where agreement or divergence has been
measured rather than assumed.

## 5. What I would do next, ranked by this machine's measurements

1. **Finish the launch folds.** The q/k/v type wall is open and measured at
   +28 launches/step on SmolLM2-Q4_K_M and on the MoE (`docs/AUDIT-group-type-walls.md`).
   Cheapest verified band, no new architecture.
2. **Measure the storage question before building for it.** Run the existing
   `tools/expert_trace.py` over a real MoE at a deliberately small cache and see
   whether the limit is file reads, H2D transfers, or the per-layer barrier. If
   the barrier dominates, an expert pack buys nothing — that is what the recorded
   zero-skip negative and the 3.1x CPU result both point at.
3. **Paged KV only together with the per-layer packing rewrite**, because the
   tier's correctness invariant is built on that geometry, not beside it.
4. **Predictors last**, against the useful-prefetch metrics, once a trace corpus
   exists — and only where the no-predictor baseline is beaten.

## 6. Not verified here

The six papers' internal numbers (no code was run), gfx1031 behaviour (no
hardware), any NVIDIA comparison (no such device), and the plan's own sources,
because they are not in this tree.
