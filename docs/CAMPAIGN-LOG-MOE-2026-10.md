# Ponytail Diag: Kraken prefill, token generation, expert cache, and MTP

**Date:** 2026-10-24  
**Scope:** Whole-code-path assessment for the user’s slow-prefill / slow-decode concern, MoE expert caching and whether MTP is actually wired. The requested rule was to treat the repository README as the trusted documentation and inspect code/tests/runtime evidence for the rest; conclusions below cite code paths and recorded runs, not an assumption that every Markdown audit is authoritative.  
**Hardware:** AMD Radeon RX 9070 XT, gfx1201, 64 CU, 15.9 GiB reported VRAM, Windows HIP build.  
**Runtime caveat:** Current executable reports HIP 7.12.60602 while the build SDK is 7.16.26331 and warns that the loaded runtime differs from the SDK it was built against. The card and gfx target are explicitly reported and per-process Compute-engine utilization is sampled, but pin the expected DLL before treating these numbers as a release baseline.

## Verdict

**ROOT CAUSE: multiple distinct limits, with one prefill mode switch and two model-dependent decode limits.** The “everything is CPU” theory is false for the measured runs: Kraken printed the RX 9070 XT/HIP backend, all selected model weights resident or partially resident on device, and substantial Compute-engine use. CPU work is still involved in MoE routing/readback and cache/file management; that is coordination overhead, not the whole forward running on the CPU. In response to this diagnosis, the in-file MTP stats path in `src/main_cli.cpp` was fixed narrowly: speculative benchmarks now use emitted tokens for tok/s while retaining rounds separately. The build and one MTP rerun verify it below.

The highest-confidence diagnosis is:

1. **Prefill exact mode trades speed for chunk-invariant arithmetic.** The Qwen3-8B RX 9070 XT controlled runs were 101.3 tok/s exact versus 645–647 tok/s fast, about **6.38x** faster on the 55-token fixed bench prompt. The path selection is real: `--prefill fast` sets `KRK_GEMM_PATH=wmma` before HIP backend construction; exact mode defaults to the chunk-invariant grid. Fast is not numerically equivalent by contract and its hash can differ past 8 rows/chunk. This is a tested speed/semantics trade, not a hidden CPU fallback.
2. **Oversized dense decode pays to stream whole layers even after promotion.** The fresh 27B sample was NOT the user’s exact Qwen3.8 Q4_K_M and the GPU was already heavily occupied. It still confirms the mechanism: only 12.01 of 14.40 GiB weights uploaded, layers streamed through a 205.8 MiB ring; 10 layers later promoted, and the remaining 4.35 GiB staged in 2.302 s. Decode was 7.9 tok/s over only seven timed steps. Your README’s prior Qwen3.8 exact-model arms remain the relevant stronger evidence: 1.2→2.3→3.0→3.1 tok/s across placement/promotion/pipeline/resident arms, with resident still slow. Paging is material but cannot be the only ceiling.
3. **MoE has two separate levers: residency/file-read churn and router/scheduling.** On Ornith-1.5-35B-A3B Q4_K_XL, a fresh 8-token sample took 18.0 tok/s, 18.4% HOT hit rate, 0.1% WARM hit, 68.2% in-place acquisitions, only 8 cold misses, and 2.23 MiB/token from file; however its startup warmup hit the 20 s cap. This is a very short, warmup-influenced sample, not a steady-state benchmark. On the 30B-A3B Q2_K controlled cache-rich runs, 77.8% HOT hits and zero evictions show that cache capacity is not an obvious failure, yet 155 MiB/token cold-source reads were charged primarily to prefill (about 1.0–1.3 s). Decode varied with interleaving and the `KRK_MOE_BATCH` arm; do not claim a batch-path speedup from these noisy runs.
4. **MTP is wired and active, but not profitable on the tested 9B hybrid target.** On Qwen3.5 9B MTP Q3_K_M, 64-token plain greedy was 82.0 tok/s. The pre-fix CLI printed a rate based on 36 rounds; the corrected run prints **62.0 generated tok/s** (64 tokens / 1031.5 ms): still slower than plain. That run had 36 rounds, 32/72 proposals accepted (44.4%), 1.78 tokens/round; proposal 184 ms and verify 378 ms. Prompt prefill was effectively unchanged (679.2 vs 679.2 ms). It is not “unwired”; it is an expensive speculative path that needs a target-specific cost-benefit gate and broader model/window testing.
5. **The CPU oracle result is not a failure verdict.** It completed the Qwen3-4B-Q8_0 model comparison at exit 0 and printed matching final argmax, but its historical “divergence exploded” text stops at layer 6 once absolute f32-vs-f16 rounding difference exceeds 1.0. It is a fixed four-token, forced-Qwen-style op harness, not a strict pass/fail whole-model oracle for all model families. It does flag meaningful cumulative f16-vs-f32 divergence; it does not establish CPU execution of the user’s generation workload or explain low throughput.

## 1. Pareto priority ranking

Ranking is evidence strength × likely workload impact × cost to distinguish. Percentages are not a universal “80/20” claim.

| Rank | Suspected cost / issue | Evidence and impact | Status / cheapest discriminating action |
|---|---|---|---|
| P0 | **Prefill rows>1 GEMM path: exact versus WMMA throughput** | Same Qwen3-8B, prompt, chunk=64, RX 9070 XT: exact 101.3 tok/s across two reps; fast 645.9–646.9 tok/s across two reps. GPU Compute sampled 74.75% on exact and 97.69% on fast (short 9 vs 3 samples). README reports more representative long-prompt fast gains too. | **Confirmed speed lever, semantics trade-off.** Use fast only if expected numeric drift is acceptable; validate prompt hash/output at actual prompt/chunk before selecting it. A controlled long prompt and exact-vs-fast hash should be the first decision. |
| P0 | **Resident dense token generation / GEMV + per-token work** | README baseline for Qwen3-8B is about 88 tok/s, ~75% of a measured 117 tok/s weight-bandwidth floor/ceiling estimate; the 27B resident control previously remains only 3.1 tok/s. `rows==1` routes to `gemv_kernel`, never WMMA. | **Confirmed remaining resident bottleneck exists; exact dominant component unknown.** Run long steady-state per-op device timing plus host components on the exact 27B model resident, and compare with unprofiled wall time. |
| P0 ✅ | **Oversize dense weight staging: per-token re-read** | **Fixed.** The staging pass re-read every streamed layer from the mapping on every token; a pinned host mirror filled at load gives 27B **2.9→6.2 tok/s** and 8B streaming **3.5→12.1 tok/s**, output byte-identical, suite 4149/4149 (§19). The "12.7 GiB held by other processes" figure was Kraken's own weights being counted, not contention. | Done. Follow-ups in T0b (fewer streamed bytes) remain. |
| P0 | **MoE prefill expert reads and expert set residency** | Fresh 30B-A3B short bench: 55-token prefill ~2.3–2.9 s; 4.97 GiB file reads charged to prefill in some arms, ~155 MiB per generated token overall; cache counters end at 77.8% HOT hits/0 evictions. | **Large cold prefill I/O is confirmed on this run, but not representative of warm-cache repeat.** Interleave same model cold/warm page-cache state, same warmup, report prefill-phase read/alloc/xfer and actual prompt length. |
| P1 | **MoE decode host route, readback and batches** | Code performs router GEMM then blocking `download_f32` per routed layer, host softmax/top-k/plan, then expert work unless device dispatch predicates all hold. `KRK_MOE_BATCH=0` is an A/B fallback. README cites extreme large-MoE router-span cost on another model, explicitly warning fetch time includes the queued-work wait. | **Architecture/host dependency confirmed; share on the tested Qwen30B run unknown.** Run `KRK_TIME_HOST=1` and `KRK_TIME=1` only after a clean reproducible steady state; attribute fetch versus wait using second-readback control and count dispatch/plan/host fallback. |
| P1 | **MTP overhead and acceptance economics** | Fresh 9B hybrid: 82.0 tok/s plain versus **62.0 emitted tok/s** with MTP (corrected). 1.78 tokens/round, 44.4% proposal acceptance. Verify cost 378 ms versus proposal 184 ms over 64 tokens. | **MTP is wired; negative result confirmed for this one model/window/short run.** Try k=1/2/4, interleaved A/B, hash identical output, retain full proposal/verify/refeed timings. Do not default-enable on this evidence. |
| P1 | **Non-WMMA / CPU fallback possibility** | `main_cli.cpp` explicitly selects HIP unless `--cpu`; it warns and uses CPU only if `make_hip_backend` returns null. Measured benchmark outputs name RX 9070 XT; stderr names HIP/gfx1201; Compute engine is active. A separate warning “CPU device has no WMMA” occurred in CPU-only test fixtures. | **CPU fallback refuted for the measured benchmark.** Still record backend identity every benchmark; treat CPU fixture warnings as expected and unrelated. |
| P2 | **Expert allocator, promotions, prefetch/pinning** | README and source contain extensive measured arms. Some improve cold read/alloc but others neutral or negative; hot ring measured +10.8% prefill but has known long-decode async pool crash and remains opt-in. Tiel MTP pin decreases drafter cost but steals target budget and was wash (8.7 vs 8.9 tok/s). | **Many tempting switches are not universally wins.** Do not enable pin/ring/lookahead/batch promote globally; test one knob, same cache state, identical output. |
| P2 | **CPU-vs-HIP numerical differences** | Oracle uses CPU f32 reference vs HIP f16 activations; fixed prompt `[1,4821,13,278]`, N=4, only standard MHA/dense path; `note()` tracks max abs but doesn’t assert fail; loop stops when max absolute >1.0; final argmax can still agree. | **Numerical rounding exists, but no performance causal link established.** Use the exact architecture-aware `KRK_DUMP` stage diff and `kraken-tests` gates; adapt oracle shape before using it for a hybrid/MoE target. |

## 2. FLOW: command entry to generated output

```text
CLI parse
  ├─ --cpu true → make_cpu_backend [explicit CPU reference]
  └─ --cpu false → optional --prefill fast sets KRK_GEMM_PATH=wmma
                    → make_hip_backend(device)
                       ├─ device/runtime init succeeds → HIP selected
                       └─ fails → warning + explicit fallback to CPU reference
  → Model::load (weight mapping, metadata, tensor upload, optional MTP load)
  → Engine::init (workspaces, KV cache, expert budget/warmup)
  → Engine::generate
       → tokenize/truncate prompt
       → if MTP speculation active and greedy → generate_speculative_mtp
       → else other draft/head speculative branches if eligible
       → ordinary prefill: chunks of min(chunk, remaining)
            forward → every layer stage_layer → projection/GEMM
            → attention/recurrent ops → dense FFN or moe_ffn
            → final head only for last row; fetch logits/device top-k
       → ordinary decode: one token at a time
            sample token → forward(..., 1) → head/logits/top-k → next token
       → output emitter and statistics
```

**CPU/GPU branch verdict:** At `src/main_cli.cpp` around backend creation, `--cpu` is the only intentional CPU selection. Otherwise CPU happens only if `make_hip_backend()` fails, in which case a warning explicitly says “falling back to the CPU reference backend.” A HIP backend with no WMMA warns “falling back to packed-math kernels”; that is a GPU kernel fallback, not CPU execution. On this RX 9070 XT, the benchmark banner, `device` line, HIP runtime line, weight-residency ledger, and Windows Compute-engine sampler all confirm HIP execution. The tests’ `device CPU (cpu) has no WMMA` lines are CPU fixture/test children and must not be mistaken for generation logs.

## 3. DATA: lifecycles that determine speed

| Data/state | Lifecycle | Performance meaning |
|---|---|---|
| Prompt tokens | text → tokenizer IDs → chunked forward | Prompt rate depends on chunk size, exact/fast GEMM rung, attention/prefill kernels, and (MoE) cold expert reads. |
| Dense quantized weights | GGUF mapping → resident VRAM if budget allows; otherwise per-layer staging ring → possible one-time promotion | Resident decode rereads quantized weights each token; streamed layers additionally drain/read/copy. Promotions are per-process/device-memory dependent. |
| Activations | HIP F16; CPU oracle F32 → stage-specific GPU kernels | CPU/GPU values can differ because each HIP store rounds to F16. Differences alone do not imply wrong backend selection. |
| MoE router scores | router GEMM on selected backend → blocking score readback to host → host selection/probabilities/plan → expert GEMM(s) | Device can be idle during serial host decisions; a host fetch timing span includes any preceding queued kernel work. |
| Expert source bytes | mapped GGUF cold → WARM host cache → pinned/pageable transfer or in-place read → HOT VRAM residency → LFU-aging/pins/evictions | HOT hit, WARM promotion, in-place and COLD read have different costs; one aggregate “hit rate” hides those tiers. |
| MTP hidden capture | target pre-final-norm rows copied per prompt/decode forward → MTP own KV primed → iterative proposals → target batched verify → longest greedy prefix accepted → MTP KV re-fed from target captures | Correct output needs rollback/realignment; acceptance rate alone is not speed benefit because every round also pays drafter and verification. |
| GPU free memory | external usage + model weights + KV + workspaces + expert pool | 15.9 GiB physical does not mean 15.9 GiB available to the model. The samples showed 1.1–12.8 GiB external allocation depending on run. |

## 4. BRANCHES and concrete values

| Branch | Condition / observed value | Result |
|---|---|---|
| CPU backend? | `--cpu` set, or HIP factory failure | CPU reference. **Not the measured benchmark path**; GPU identity and Compute activity observed. |
| Prefill fast? | `--prefill fast` → `KRK_GEMM_PATH=wmma` before backend construction | Multi-row GEMM chooses WMMA where compiled/capable. Speed rose ~6.38x on Qwen8 short prompt, with non-exact chunk arithmetic. |
| Exact-prefill grid? | Default, `KRK_GEMM_EXACT` not disabled | Rows>8 use decode-associated exact grid for chunk invariance. This is slower; not a CPU fallback. |
| GEMV versus GEMM? | `rows == 1` | `gemv_kernel`; WMMA path is only for rows>1. Ordinary decode and large vocab head are M=1. |
| Thin-row path? | `1 < rows <= kGemvMRowsMax` and `gemv_m_enabled()` | Thin grouped matvec path may be selected before tiled GEMM. Actual rung must be profiled per shape. |
| WMMA reachable? | gfx supports WMMA and binary compiled with WMMA, plus `rows>1`; fast forces rung request | Otherwise SIMT/packed-math fallback. Startup logs warn for an unsupported/compiled-out request. |
| MoE expert batching? | `n==1`, batch enabled, grouped device plan valid, plain gated expert geometry, no hybrid/dispatch/probe/sync/fused silu/etc. | One batched decode path; otherwise per-expert loop. Counter `moe batch` reports executed/skipped layers. |
| One-launch MoE dispatch? | Explicit device pointer arm + valid published/non-stale table + full routed set resident + supported geometry/no unsupported scales, biases or activations | Device dispatch can avoid router readback/per-expert host loop. Otherwise fallback path remains. Merely building the feature does not prove it runs. |
| MTP speculation? | MTP loaded + buffers allocated + `mtp_spec_` + greedy; recurrent target also needs snapshots | Then custom MTP loop. Sampling logs warning and decodes plainly. |
| MTP loaded? | `--mtp-probe`, `--mtp-spec`, or sidecar path loads it before target model load | Runtime emitted `nextn (MTP) block 32 loaded` and allocated capture/KV, so this is wired on tested 9B. |
| Expert residency? | Resident budget and actual external free VRAM; MoE cache HOT/WARM/COLD choices | User’s fresh Ornith/Tiel runs had low HOT hit under different cache states; Qwen30B fresh cache-rich sample had 77.8% HOT hits/zero evictions. |

## 5. TRUTH TABLES

### CPU versus HIP

| `--cpu` | HIP factory | Backend | Status |
|---:|---|---|---|
| false | succeeds | HIP | **Measured RX 9070 XT path** |
| true | not attempted | CPU reference | Intentional diagnostic / user-selected branch |
| false | fails/no HIP devices | CPU fallback after explicit warning | Untested in the fresh successful runs; behavior is explicit in source |

### Prefill exact versus fast

| Mode | Arithmetic | Throughput (Qwen8 55 tokens, chunk 64) | Hash/correctness condition |
|---|---|---:|---|
| exact/default | Chunk-invariant exact row grid | 101.3 tok/s, two reps | Baseline; current `test_prefill_chunk_bitexact` passes |
| fast/WMMA | Matrix-core grouped rows | 645.9–646.9 tok/s, two reps | Explicitly not chunk-invariant past 8 rows per README/source warning; must compare desired output before adopting |

### MTP/speculation

| MTP present/loaded | Greedy | Recurrent snapshot | Result |
|---|---:|---|---|
| no | either | n/a | Plain decoder / other draft path |
| yes | false | any | Warns “MTP speculation is greedy-only”; ordinary decode |
| yes | true | missing on recurrent model | Warns and falls back to ordinary decoding |
| yes | true | available | MTP propose → target verify → accept/replay path. **Observed on Qwen3.5 9B, but slower than plain.** |

### MoE dispatch/batching capability

| Published valid dispatch table | Supported geometry | Other gating/exclusion flags | Likely path |
|---:|---:|---|---|
| yes | yes | all clear, not hybrid | one-launch dispatch candidate |
| no | any | any | host route/plan plus grouped or per-expert loop |
| yes | no | any | fallback route/plan |
| yes | yes | bias/scale/activation/probe/hybrid makes unsupported | fallback route/plan |
| unknown | unknown | runtime not instrumented | **Cannot state which path a particular large model uses.** |

## 6. Prefill deep dive

### Code path and kernel choices

`Engine::generate()` tokenizes and chunks prompt IDs (default chunk 256, clamped to 1..1024), calling `forward()` once per chunk. `forward_core()` embeds rows, then for every layer runs the projection group, optional packed-gate split and biases, attention or recurrent path, FFN/MoE, and captures optional draft features. It computes vocab head logits only for the last prefill row (unless a caller explicitly asks for all/none), which is already a useful saved cost.

HIP `gemm()` selects `rows == 1` directly to GEMV. For prefill rows, exact default uses a chunk-invariant grid for larger row counts; WMMA is available for the rows>1 fast path on gfx11/gfx12. Thin row groups can use GEMV-M variants. `gemm_group()` only uses the fused decode kernel for rows==1; prefill falls back to per-matrix `gemm()` calls. Thus prefill uses a different launch/compute mix from decode.

Qwen8 controlled measurements (`--bench --greedy --max-tokens 32 --ctx 512 --chunk 64`; fixed 55-token prompt):

| Interleaved order | Mode | Prefill | Decode | GPU observation |
|---|---|---:|---:|---|
| 1 | exact | 101.3 tok/s | 89.0 tok/s | Compute 74.75%, 9 samples |
| 2 | fast | 646.9 tok/s | 75.5 tok/s | Compute 97.69%, 3 samples |
| 3 | fast | 645.9 tok/s | 75.5 tok/s | Compute 97.69%, 3 samples |
| 4 | exact | 101.3 tok/s | 89.4 tok/s | Compute 74.75%, 9 samples |
| 5 | exact | 101.3 tok/s | 89.3 tok/s | Compute 74.75%, 9 samples |
| 6 | fast | 645.2 tok/s | 75.0 tok/s | Compute 97.69%, 3 samples |

These are short-prompt absolute numbers with significant fixed overhead, but the exact/fast ratio is consistent within each mode and precisely aligns with the documented path switch. Decode in this tiny test differs ~15% between modes and is too short/order-sensitive to infer a decode regression from it; the mode exists to affect rows>1.

**Prefill on MoE has an additional I/O factor.** Qwen30B-A3B 55-token prompts showed ~2.3–2.9 s prefill, while the report’s expert-path split charged around 1.0–1.3 s of file reads plus ~0.45–0.57 s allocation to prefill in some arms. End-of-run counters report approximately 4.97 GiB from file over 32 generated tokens (~155 MiB/output token), but that aggregate includes prefill plus decode; it must not be divided blindly to infer prefill-only throughput. In another valid run the counter was 77.8% HOT hits, 0 evictions, and only 98–100 WARM promotions (about 160 MiB), showing many cold distinct experts loaded once during prompt then reused. Page-cache warm/cold and expert warmup ceiling dominate comparison.

**Large 27B short-prompt sample:** Qwen3.8-WebGGUF-Q4_0 (14.4 GiB file, not the Q4_K_M user headline) did 16.7 prefill tok/s on 55 prompt tokens. It had ~12.8 GiB already held externally and only ~3.5 GiB free at process start. It is a useful stress trace, not a fair large-model baseline. The user’s supplied Qwen3.8 Q4_K_M measurements—especially 6.8 tok/s prefill from the prior trace and 3.1 resident decode—are better workload evidence.

**Conclusion:** Exact prefill is intentionally conservative for invariance. There is a clear opt-in WMMA lever with a tested ~6.4x short-prompt speed gain, but it changes arithmetic. For users requiring exact chunk behavior, the remaining prefill Pareto target is per-shape exact-grid kernel efficiency and the MoE cold expert read/allocate budget, not CPU offload.

## 7. Token-generation deep dive

### Dense

At one token, M=1 weight projection is GEMV; matrix-core WMMA is unavailable by design. The output vocabulary head is one more large M=1 projection, followed by greedy top-k. README’s 8B result is roughly 88 tok/s, near the ~117 tok/s ceiling estimated from a measured ~585 GB/s bandwidth and ~5.02 GB weights/token; this indicates the mainstream 8B resident path is comparatively efficient. It does not transfer to 27B: the supplied fully resident Qwen3.8 27B result is only ~3.1 tok/s, so model geometry/kernels, memory clocks/pressure, residual compute and launch/wait costs remain unexplained even with streaming removed.

The user’s earlier external-use trace documented 12.666 GiB held and only 3.638 GiB free; the fresh 27B Q4_0 sample likewise saw 12.8 GiB occupied. GPU contention changes residency, allocation and scheduling, and must be cleared for a causal resident baseline.

### MoE

Each routed layer computes router scores, synchronously downloads a small `[n, n_expert]` row to host, selects top-k and constructs a plan; then it acquires selected expert weights and runs grouped expert projections. For M=1, `KRK_MOE_BATCH` can reduce many per-expert gate/up/down launch groups to a grouped batch, but strict capability flags can skip it. Device dispatch is more restrictive and depends on a current expert pointer table with the whole routed set resident and a supported plain-SwiGLU geometry. At batch/prefill `n>1`, GPU MoE route scores are still downloaded and host selection/plan is made for each row; expert work batches by expert and row.

Current Qwen30B 4-run stock vs `KRK_MOE_BATCH=0` was not a clean A/B: throughput varied materially between sequential process launches, and the timed variants had just 1 pair each. Read it only as a sanity demonstration that both batch arms execute; do not claim one is faster. The unprofiled stock 32-token results ranged 23.0–46.0 tok/s; batch-off timed results were 34.5–35.6. During bench logs, stock printed 1488 batched layer invocations. The `KRK_MOE_BATCH=0` arm intentionally removes that path. At these tiny 32-token lengths with new process init/warmup and VRAM changes, the variation dwarfs what can be concluded. Repeat 5+ interleaved pairs with a warmed expert/index state, longer decode (128+ tokens), fixed H2D verdict and card otherwise idle.

Cache counters must be read by tier. Ornith’s 18.4% HOT hit does not mean 81.6% file reads: the measured split was HOT 18.4%, WARM 13.3%, in-place 68.2%, cold misses 0.1%. Its first run also hit the 20-second warmup ceiling and loaded only 8 decode-time cold experts. Tiel’s sample did reach the configured warmup wall cap and had 18.4% HOT, 13.3% WARM and 68.2% in-place acquisition—same tier shares under its 8-token run. Reports of a low HOT-hit ratio alone are therefore insufficient to label cache thrash; look at cold misses, file MiB, bytes promoted, evictions, alloc/room/xfer costs, and decode-phase split.

For Qwen30B the cache had 77.8% HOT hits and zero evictions, with a 10 GiB device cache and 10 GiB WARM planned. This rules out obvious HOT thrash for that request, not all cache effects: the routing working set/file I/O during prefill still cost seconds. In the user’s large Tiel sample, default cache/warmup behavior is a major model-specific variable; the model’s routed set is ~22 GiB and cannot fit on a 15.9 GiB card alongside trunk/KV/workspaces, so ranking, WARM/pinned transfer and free headroom directly matter.

**Router timing caution:** `moe.route.fetch` is a blocking D2H read and may inherit all queued GPU time. It is not copy-only time. `moe.route.rest` is host plan logic plus subsequent work in a parent span; do not sum nested spans. Large-model claims require host-time component attribution and/or an A/B second fetch after the queue is already drained, plus actual device dispatch counters/skips.

## 8. MTP wiring and correctness assessment

### Wiring path traced

1. CLI accepts `--mtp-probe`, `--mtp FILE` and `--mtp-spec`; `--mtp FILE` also enables loader request.
2. `Engine::init` calls `model_.set_mtp_load(probe || spec || sidecar)` and sets the optional path before `Model::load`.
3. Loader identifies nextn/MTP tensors after the trunk or from sidecar, validates architecture and geometry, stores `MtpBlock`; runtime emits a detailed “nextn (MTP) block … loaded” line.
4. Init checks head/query/FFN dimensions, allocates private KV K/V plus a per-prompt-position hidden capture and scratch. Failure disables MTP cleanly; recurrent targets require recurrent/conv snapshots for speculation or warn/fall back.
5. `forward_core` captures pre-final-norm target hidden rows and token IDs at each forward, including prefill chunks and accepted decode rows.
6. `mtp_prime()` embeds each known token, feeds captured target hidden row, and builds block KV. MTP block computes concat(normed embedding, normed target hidden) → `eh_proj` → norm/QKV/RoPE/KV/attention/WO → residual/norm → dense or routed MTP FFN → shared head norm/head logits.
7. Greedy MTP loop proposes up to k sequentially from block output, verifies all proposals with one target batched forward, accepts longest matching prefix plus target bonus, rolls back rejected dense KV or restores/replays recurrent state, then re-feeds accepted rows from captured target hidden state. Output accounting reports rounds, proposed/accepted, propose/verify/snapshot/refeed and block-expert costs.
8. Dispatch is explicit in `Engine::generate`: in-file MTP speculation takes precedence when loaded + greedy + snapshot available. Sampling logs a warning and falls back to plain generation. MTP probe alone is measurement and does not alter emitted output.

### MTP measured behavior

Qwen3.5 9B DeepSeek-V4-Flash MTP Q3_K_M, RX 9070 XT, greedy, ctx 512:

| Mode | Prefill | Decode | Speculation stats |
|---|---:|---:|---|
| Plain, 64 emitted tokens | 55 tokens / 679.2 ms = 81.0 tok/s | 64 / 768.2 ms = 82.0 tok/s, 63 timed steps (12.2 ms/step) | none |
| In-file MTP, k=2 | 55 tokens / 677.9 ms = 81.1 tok/s | 64 / 1030.3 ms = 62.1 emitted tok/s, 36 rounds (28.62 ms/round) | 32/72 accepted, 44.4%; 1.78 output tokens/round; proposal 185 ms, verify 377 ms, refeed 4 ms; no MTP expert cost |

Both use 4.30 GiB resident weights, no KV tiering, and show ~80% Compute-engine utilization. MTP overhead is therefore internal compute, not CPU fallback. The 9B is recurrent hybrid, but snapshots cost 0 ms at displayed precision; replay verification/drafter forward dominate.

**MTP stats correction now applied:** `print_run_stats()` and the top-level bench output now classify `(engine.has_draft() || engine.mtp_spec_active())` as speculative. Speculative decode tok/s uses `r.generated / decode_ms`; the separate step count is printed as speculation rounds. The pre-fix output measured rounds per second but labelled that as tokens per second. Rebuilt and reran the exact 9B MTP bench: the latest run emitted 64 tokens in 1030.3 ms = **62.1 tok/s**, over 36 speculation rounds (28.62 ms/round); plain comparison remains 82.0 tok/s. So reporting is fixed and the performance conclusion remains: k=2 MTP is slower for this target/run.

MTP output correctness is not proven by successful initialization alone. The relevant `test_mtp_sidecar` passed 17/17 loader/validation checks in this build; this matches the README’s documented fixture coverage. It does not by itself establish full-token output equality for every in-file MTP architecture. The README records per-model probe/acceptance checks and MTP text identity in specific prior arms. The new 9B A/B used greedy and no output hashes, so this turn’s runtime probe confirms active execution and counters, not identical generated tokens. Before using it, compare plain/MTP token IDs or hashes on this exact model and prompt.

**Conclusion:** the MTP code is wired end-to-end and the measured run exercised it. It is not a universal speedup; this tested 9B k=2 setup regresses emitted throughput ~24% (62 vs 82 tok/s using the correct unit). MTP decode rounds are now reported separately from generated-token throughput. Keep opt-in; add an acceptance/cost gate or improve verification economics only after representative A/B.

## 9. Oracle and correctness caveats

Ran `build-hip/kraken-oracle.exe G:\More-models\Qwen3-4B-Q8_0.gguf` successfully, exit 0. It loaded both `BackendKind::CPU` and HIP on the RX 9070 XT and model weights on each. It reported:

- Initial embedding maxabs `3.004e-05`.
- Per-layer values increasingly drift due to CPU f32 reference versus GPU f16 activations and different rounding/reduction paths.
- At L06 `add_inplace2` maxabs reached 12.52, then “divergence exploded at layer 6 — stopping”.
- Final head/logits were still computed from its separate CPU/GPU scratch and argmax IDs matched (130949), with logits maxabs 0.003031.
- Program returns 0 unconditionally after printing; its model loop’s “exploded” criterion is diagnostic stopping, not an error status. It is not a strict pass.

Important oracle scope mismatch: fixed N=4, POS0=0, token IDs `[1,4821,13,278]`, architecture-wide `ModelConfig` standard Q/K/V, attention and dense FFN assumptions. It does not mirror the full modern engine for recurrent GDN, Qwen3.5 packed gates, MLA, MoE routing/expert-cache behavior or model-specific special paths. It should not be used to judge large MoE/MTP performance or to declare the HIP model incorrect. Run `--gdn`/`--ssd` for their dedicated op suites, and use engine `KRK_DUMP` device vs `--cpu` for model-aware stage diff where supported. CPU oracle differences are numerical reference differences, not proof the workload ran on CPU.

## 10. HYPOTHESES

### H1: Exact prefill GEMM path explains the headline slow exact prefill
**For:** Same 8B model/prompt/chunk/card exact 101.3 vs fast 645–647 tok/s, repeatable in interleaved order; fast is explicitly WMMA.  
**Against:** 55-token prompt is short, GPU utilization sampling only 3–9 samples; exact and fast do not have identical numeric behavior. Other models/shapes may have other dominant costs.  
**Test:** 1024–4096-token prompt, 5 interleaved exact/fast pairs, fixed chunk and warm page cache; record output/prefill hashes and GPU op timing.  
**Cost:** low. **Status:** confirmed for short Qwen8 shape; general magnitude unknown.

### H2: Large dense decoding is slow only because weights stream
**For:** Paged arms improve substantially with promotion and pipeline; fresh 27B-family run incurs 4.35 GiB staging per 8-token request and has only 1.2 GiB remaining after init/run.  
**Against:** User’s own Qwen3.8 resident baseline is still only 3.1 tok/s, about the same as 3.0 pipeline.  
**Test:** exact Qwen3.8 Q4_K_M model, idle card, force resident and auto-streamed interleaved, >=32 tokens, identical prompt/hash; collect per-layer residency.  
**Cost:** medium/high due model size. **Status:** refuted as sole cause; paging effect confirmed.

### H3: CPU fallback is slowing generation
**For:** There is a source fallback branch if HIP initialization fails; CPU path can be very slow for large weights.  
**Against:** Measured runs identify device AMD Radeon RX 9070 XT, HIP gfx1201, Compute 0 utilization, and GPU weight residency. CPU fixture tests separately print the same no-WMMA warning.  
**Test:** Already done: inspect benchmark banner + stderr + engine sampler.  
**Cost:** low. **Status:** refuted for measured runs. Host-side routing/cache coordination remains.

### H4: MTP improves the tested target throughput
**For:** 44.4% proposals accepted; 1.78 tokens/round, so fewer target forward rounds than plain decoding.  
**Against:** Round work includes 184 ms proposal and 378 ms verify; emitted rate computes ~62 tok/s versus plain 82; the old CLI rate counted rounds, not generated tokens.  
**Test:** Re-run plain/MTP interleaved five pairs at k=1/2/4, no benchmark self-overlap, confirm output token IDs/hash, report generated tokens / decode_ms.  
**Cost:** medium. **Status:** refuted for k=2 on this 9B sample; general MTP benefit unknown.

### H5: Expert cache eviction/churn dominates Qwen30B decode
**For:** Older cache experiments on different tiny MoE and Tiel models establish severe performance loss if active sets do not fit or promotion thrashes. File-read/pre-fill work is substantial on the fresh Qwen30B run.  
**Against:** Fresh Qwen30B run had 77.8% HOT hits, zero evictions, only ~98–100 promotions over 32 tokens; decode timings vary widely while cache counts stay similar. The short run is not steady.  
**Test:** >=128-token runs with warm expert index and WARM cache, interleave budget arms, compare decode-phase read/room/alloc/xfer, HOT/WARM/in-place/cold misses and outputs.  
**Cost:** medium. **Status:** cache behavior measured; decode causality unknown for that sample.

### H6: Router host synchronization/readback is the universal MoE limiter
**For:** Blocking router fetch and host top-k exist in default route path; README has another large model where route fetch/rest consume most step.  
**Against:** Fetch timing includes GPU work and may be wrongly attributed; compatible all-resident models can use device dispatch, and different router/model shapes vary. No fresh per-op host-time profile was captured for Qwen30B.  
**Test:** exact target MoE, fixed 128+ tokens, `KRK_TIME_HOST=1`, route sync off/on and second-readback control, inspect dispatch, no summing nested spans.  
**Cost:** low/medium. **Status:** code path confirmed; dominant share unproven model-by-model.

## 11. FISHBONE / ISHIKAWA

```text
Slow prompt processing / generation
├── Method
│   ├── short 55-token bench dominated by fixed setup and cold expert reads
│   ├── warmup runs to a 20 s cap before measured prefill (Ornith/Tiel sample)
│   └── only 7-31 timed decode steps on large-model probes: noisy startup regime
├── Machine
│   ├── GPU shared with other workloads: 1.1-12.8 GiB held externally in sampled runs
│   ├── 15.9 GiB is physical capacity, not usable free VRAM
│   ├── 256 MiB aperture / no large BAR makes transfer path relevant
│   └── HIP runtime version differs from SDK build version
├── Material
│   ├── exact GEMM versus matrix-core WMMA prefill mode
│   ├── dense model exceeds available VRAM; recurrent/dense per-token GEMVs
│   ├── MoE working set: HOT/WARM/in-place/COLD mix and router pattern
│   └── prompt length/context and GGUF quant/layout differ between models
└── Implementation
    ├── exact row-arithmetic path is slower by design; fast path is non-invariant
    ├── M=1 decode uses GEMV rather than WMMA
    ├── MoE router readback and host plan serialize each routed layer
    ├── expert cache batching/device dispatch gated by strict capability predicates
    ├── MTP proposal + batched verify + replay can cost more than saved target rounds
    └── MTP statistics previously confused rounds with generated tokens in CLI summary (fixed in this task)
```

Bottom-up discriminators: controlled exact/fast long prompt; idle-card resident/paged dense run; cache-tier counters separated by prefill/decode; device-dispatch path counter and route host spans; MTP token hashes plus correct emitted-token/s calculation.

## 12. 5 WHYS: why does MTP k=2 show poor token generation?

1. **Why is the measured MTP run slower than plain?** 64 generated tokens take 1032 ms instead of 768 ms.
2. **Why does fewer target rounds not save time?** The drafter proposes 72 candidates over 36 rounds, but target verifies them in 377 ms and MTP proposal itself takes 184 ms; 32 accepted means 1.78 output tokens per round including emitted-token accounting.
3. **Why is acceptance not enough?** A round has target batch-forward work, MTP block work, logits/argmax, and refeed after rejection; acceptance fraction does not represent cost saved.
4. **Why did the CLI understate the real output rate?** It divided 36 rounds by 1.032 s, while 64 tokens were emitted; the MTP implementation increments `decode_steps` per round and the generic stats code had assumed those were plain decode token steps. This stats fix uses emitted tokens for tok/s and labels rounds separately.
5. **Why should not change code just from this one run?** It is one model, one k, short generation, no A/B token hash and shared GPU. Correct next step is controlled emitted-token timing and correctness, then a targeted accounting/fix handoff.

## 13. BARRIER ANALYSIS

| Barrier expected | Present? | Assessment |
|---|---|---|
| Avoid accidental CPU-only diagnosis | Backend banner and explicit fallback warning exist | Good; tests’ CPU warning is confusing when logs are mixed, but benchmark evidence settles actual device. Preserve backend identity in every artifact. |
| Avoid slow exact path being mistaken for CPU | `--prefill exact|fast`, GEMM rung/source and exactness warning exist | Good; fast is opt-in due numerical/chunk behavior. Need per-target measured throughput/correctness before choosing. |
| Keep output head from running for every prefill row | `forward_core(LogitsLast/None)` and head call only for selected row(s) | Present and reduces work. |
| Keep MoE cache honest across tiers | Separate HOT/WARM/in-place/COLD counters and phase timing | Strong instrumentation exists; must use all counters rather than HOT hit rate alone. |
| Ensure expert refusal is visible rather than omitted | `moe_ffn` error on failed `acquire` | Present. No silent selected-expert loss in normal branch. |
| Avoid stale/rejected MTP KV state | Target hidden capture, MTP KV prime/refeed, dense rollback, recurrent snapshots/replay | Implementation path exists; specific architecture must still pass equality tests. |
| Correct speculative performance unit | MTP rounds are counted as `decode_steps`; CLI now classifies `mtp_spec_active()` as speculative and uses emitted tokens/time | **Fixed and verified:** top-line output now prints generated tok/s and separately labels rounds. |
| Prevent uncontrolled runtime comparisons | HIP runtime warning present | Release-baseline barrier incomplete until loaded runtime DLL is pinned and external VRAM held is recorded. |
| Ensure long-run safety of experimental ring pinning | Source records a long-decode async pool crash | Ring remains opt-in; do not flip it on as a general fix without resolving and verifying allocator ordering. |

## 14. FTA (FAULT TREE)

```text
TOP EVENT: “Kraken is slow”
OR
├── PREFILL latency high
│   OR
│   ├── exact GEMM rung selected (confirmed default); WMMA fast path opted out
│   ├── prompt is short / fixed cost dominates comparison
│   ├── MoE expert cold source/read/allocation occurs during prefill
│   ├── chunk/shape/model family selects different exact attention/projection paths
│   └── external VRAM pressure causes memory/driver contention
├── TOKEN generation latency high
│   OR
│   ├── dense M=1 GEMV and large vocab/head/model memory traffic
│   ├── model does not fit free VRAM; per-token weight staging continues
│   ├── MoE expert set bigger than effective HOT tier → promotions/reads/evictions
│   ├── serial router D2H + host planning / fallback branch (model-specific)
│   ├── host/GPU queue synchronization, launch gaps, or head readback
│   └── speculative mode cost exceeds saved target passes (MTP sample confirmed)
└── Apparent slowdown / misleading metric
    OR
    ├── process actually runs CPU fallback (refuted for sampled runs)
    ├── benchmark metric confusion (MTP round-rate bug fixed in this task)
    ├── compare cold-page-cache run against warm-page-cache run
    ├── short 7-step sample / unrelated process on GPU
    └── runtime DLL differs from SDK used to build binary
```

## 15. TIMING / DRIFT

| Run | Prompt | Decode | Key observed state |
|---|---:|---:|---|
| Qwen3-8B exact, chunk64, short bench | 55 / 543 ms = 101.3 tok/s | 32 emitted; 31 timed steps, 89.0–89.4 tok/s | all 4.68 GiB resident, GPU named RX9070XT |
| Qwen3-8B fast, same | 55 / 85.0–85.2 ms = 645–647 tok/s | 31 timed steps, 75.0–75.5 tok/s | all weights resident; short run shows 97.69% Compute samples |
| Qwen3.5-9B MTP target | 55 / 678 ms = 81.1 tok/s | 64 emitted / 1030.3 ms = 62.1 tok/s; 36 rounds | MTP 32/72 accepted; old CLI confused round rate with tok/s (fixed) |
| Qwen3.5-9B plain | 55 / 679 ms = 81.0 tok/s | 64 / 768 ms; 63 timed decode steps = 82.0 tok/s | no tiering, 4.30 GiB resident |
| Qwen3.8 WebGGUF Q4_0, 27B family | 55 / 3297 ms = 16.7 tok/s | 8 / 885 ms; 7 steps = 7.9 tok/s | not same quant/model as user result; 12.8 GiB externally held; 4.35 GiB staged |
| Ornith-1.5-35B MoE sample | 55 / 1919 ms = 28.7 tok/s | 8 / 388 ms; 7 steps = 18.0 tok/s | warmup hit 20 s cap; HOT 18.4%, WARM 13.3%, in-place 68.2%, cold 0.1% |
| Qwen30B-A3B short MoE sample | 55 / 2.29–2.86 s = 19.2–24.0 tok/s | 32 / 674–1350 ms = 23–46 tok/s | 77.8% HOT hits, zero evictions; prefill charged ~1.0–1.3 s read/0.45–0.57 s alloc in split examples |

Do not compare load time across these arms: the first in a series encountered warmup/cold page cache and later processes hit a different page-cache state. Do not treat the GPU engine utilization sampling (50 ms intervals) as kernel-level occupancy; it only corroborates activity.

## 16. Mismatch / change analysis

No “worked last week” known-good SHA and onset commit were provided in this investigation. The current code/README themselves say exact prefill is default because its arithmetic is chunk-invariant, while fast WMMA is an opt-in lossy/changed-arithmetic arm. Therefore there is no evidence in this task to label the slow exact mode a recent regression. It may be the intended speed/correctness policy choice. Similarly, do not blame recent expert-cache code from current counters without reproducing the affected exact model and settings.

## 17. CONCURRENCY / synchronization interleavings

- **Prefill layer stream:** stage current dense layer → enqueue GEMMs/attention → next layer stage/read. Queue drains around streamed layer replacement can serialize otherwise-overlappable execution. User’s earlier 27B staging trace’s 4.034 s queue drain (57% of staging total) is a meaningful transfer-path clue but not a percentage of all token-generation time.
- **MoE route:** router GEMM enqueue → blocking `download_f32(router_host_)` waits for prior default-stream work and transfers router row → CPU top-k/plan → expert acquire/copy → expert kernels. The fetch span contains wait plus copy; timing it as copy is invalid. Explicit `KRK_ROUTER_SYNC` is off by default because the blocking read already orders after queued work.
- **Expert pool reuse:** copy-stream promotions can race old readers unless release events/order are honored. The hot ring pin currently remains opt-in because a long Tiel decode has a recorded async-copy launch failure in pool reuse. This is an actual safety constraint, not a throughput knob to enable casually.
- **MTP:** target hidden capture must correspond to token/position; MTP’s own draft states cannot be retained after partial acceptance because the rejected proposal rows used drafter hidden state. Implementation rolls/replays. The measured run showed 4 ms refeed, not the dominant cost; correctness still needs hashes.No data race is proven from this evidence; suspected issue is unnecessary serialization or inaccurate attribution, to be isolated with per-path profiling.

## 18. TODO / fix handoff

**Every arm below now has a logged trace** — see §20 for the file layout (`perf-traces/collect.sh`, `SUMMARY.md`, `summarize.py`) and the consolidated tables. Two source patches were made and verified during this diagnosis: the MTP bench stats unit (emitted tokens/sec, rounds reported separately) and the pinned promotion DMA (§20.1). Remaining findings are ready for follow-up. Priority 0 tasks are ordered by the Pareto evidence: first attribute read/stream time, then attack repeatable GEMM/cache costs. Each task below has a discriminating test and a correctness gate.

Fresh exact large-model update: `G:/More-models/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` loads as a 64-layer Qwen3.5 recurrent model with 15.97 GiB of weights. With 12.7 GiB already externally held on the GPU, 12.02 GiB was initially uploaded, ten tail layers promoted, and 64.70 GiB streamed over the 32-token bench. Unprofiled throughput was 2.9 tok/s; profile-instrumented throughput was 2.5 tok/s, so those rates must not be compared as a clean A/B. The profile’s dominant `warm_read(pool)` category wraps host worker-pool read/coordination; it is not GPU matrix math. It raises read/stream scheduling to the top probe, not yet a proven exclusive root cause.



### Fix backlog (start here)

| ID / priority | Task (expected owner area) | Evidence / why it ranks here | First implementation direction | Done when |
|---|---|---|---|---|
| T0 / P0 ✅ **DONE** | **Stop re-reading unchanged streamed bytes every token** (`src/model.cpp`, `include/krk/model.hpp`) | Shipped: pinned host mirror. 27B 2.9→6.2 tok/s, 8B streaming 3.5→12.1 tok/s (3.46x), output md5 identical, suite 4149/4149. Full write-up in §19. | Done. | Verified. |
| T0b / P0 ❌ **MEASURED, NOT A LEVER** | **Reduce streamed bytes per token (residency)** | Tested with `KRK_VRAM_RESERVE_MB` = 1024/512/384/256: stagings fell 288 → 160 and streamed bytes 64.70 → 37.53 GiB, and decode did **not** move (5.8 → 5.7 tok/s either way). The copy is already hidden behind compute, so fewer bytes buys nothing on this model. Do not spend effort here. | Kept as an arm only; the reserve stays at its guarded default. | Closed: refuted by measurement. |
| T0c / P0 ✅ **DONE** | **Promotion was on the pageable copy path** (`src/model.cpp`, `promote_layer`) | `stage_host_` is pinned, and the call was `upload_paged_batch` — 9.8 GB/s against 27.6 GB/s for the pinned batch in the same run. Fixed to the pinned batch: one-off staging 764-1662 → **455-513 ms**, first token 792 → 595 ms, decode 5.2 → 5.8 tok/s, output byte-identical, suite 4149/4149. | Done (§20.1). | Verified. |
| T1 / P0 | **Prefill: exact is 3.4-7.7x slower than WMMA and no internal lever was found** (`src/hip/kernels/gemm.hpp`, `src/main_cli.cpp`) | 191-token prompt: 8B 163.1 exact vs **1256.6** fast; 27B 45.9 vs **153.7**. `gemm(grid)` owns the exact phase (8B 1 128 ms, 27B 3 005 ms) and is scalar-dequant-ALU bound (~27% of scalar FMA peak), not memory bound — its low GB/s figure is an artefact of a fixed byte count with a row-proportional time. `KRK_GRID_G` swept 8/16/32/64: flat (549.6/538.5/539.7/550.4 ms). `wmma_cfg::MB=2` measured **worse** (152 → 275 ms) and is reverted. | Nothing left inside exact without changing the association. Expose the trade clearly and let the user choose `--prefill fast`; any default change is a numerics policy decision, not a kernel fix. | A documented, measured statement of the trade plus a per-workload recommendation; leave `test_prefill_chunk_bitexact` green. |
| T2 / P0 | **Make large-MoE prefill avoid redundant cold-source rereads/allocations** (`src/expert_cache.cpp`, `src/engine.cpp`) | Fresh Qwen30B short prompt pays ~1.0–1.3 s read + ~0.45–0.57 s allocation during prefill; 4.97 GiB file-read counter, yet final cache reports 77.8% HOT / zero evictions. This looks like one-time distinct working-set materialization more than HOT churn. | Separate cold/warm filesystem cache runs; determine if reads/allocs are repeated, whether page cache / WARM admissions are reused, and whether warmup/index coverage is adequate. Only then change admission or reuse. | Same-model prompt A/B, cold and warm repeats; less prefill read/alloc time and same outputs, no rise in decode misses/evictions. |
| T3 / P1 | **Prove and improve MoE router/device-dispatch branch** (`src/engine.cpp`) | Host router readback is a real default dependency but measured contribution is model-specific; existing spans include prior queue wait. | Emit/count chosen dispatch branch and fallback reason; profile route GEMM, first fetch, host selection and expert execution without summing nested spans. | Large target-MoE 128+ token run accounts for step cost; route optimization A/B retains identical token IDs and cache correctness. |
| T4 / P1 | **Run MTP profitability and exact output tests on target families** (`src/engine.cpp`, MTP tests) | k=2 Qwen3.5-9B is wired but 62 tok/s vs plain 82 tok/s; 44% proposal acceptance. CLI rate correction is already implemented. | Sweep k=1/2/4 interleaved with plain, add token-ID/hash comparison and report proposal/verify/refeed cost per generated token. Do not enable by default. | At least one supported model/window achieves repeatable net throughput gain and identical greedy token sequence; otherwise document per-target no-win policy. |
| T5 / P1 | **Fix async expert hot-ring reuse before enabling it** (`src/expert_cache.cpp`) | Existing record: hot ring adds ~10.8% prefill but crashes long Tiel decode in pool async reuse. | Reproduce lifetime/order issue; preserve opt-in until copy-stream reader completion is enforced by allocator reclamation. | Repeated >1024-token Tiel decode with normal async path, no launch/allocation errors, output identical; only then reevaluate performance. |
| T6 / P2 | **Pin HIP runtime/toolchain for comparable performance baselines** (Windows launch/build packaging) | All new probes warn loaded HIP 7.12 differs from SDK 7.16. | Stage the SDK-matched HIP runtime next to the executable or explicitly capture runtime version in every bench artifact. | Warning gone and same-model before/after baselines use recorded driver/runtime/build identity. |
| T7 / P2 | **Make CPU-vs-HIP oracle verdict architecture-aware and actionable** (`tools/oracle_probe.cpp`) | Current probe compares f32 CPU to f16 HIP on fixed N=4 standard MHA/dense path; cumulative drift stops it at L06 but exit status remains zero. It is not valid for recurrent/MoE/MTP. | Label diagnostic-only or return nonzero on a defined mismatch; use model-aware stage capture for families it does not model. | A known-good rounding case and injected mismatch have distinct statuses; recurrent/MoE scope is explicitly tested or refused. |

```text
TODO src/main_cli.cpp:1251: [sev:P1] Decide whether exact prefill’s throughput/correctness trade is acceptable per workload | fix: no default change; fast must remain explicit until semantics policy is chosen | test: >=1024-token exact/fast interleaved throughput, prompt hashes and generated token hashes across supported model shapes

TODO src/engine.cpp:7370,7477: [sev:P1] Measure MoE route readback, host plan and actual one-launch dispatch on the exact slow large MoE | fix: optimize only the confirmed dominant serial dependency | test: 128+ tokens, KRK_TIME_HOST, second-fetch control, dispatch/layer counters, same cache state and output

TODO src/expert_cache.cpp:893,958-1040: [sev:P1] Separate prompt-time cold expert file read/allocation from token decode cache churn | fix: only alter warmup/admission after cold/warm paired trace | test: same model and prompt, 5 cold/warm interleaved repetitions; compare prefill/decode phase read/alloc/xfer, HOT/WARM/in-place/COLD, evictions and file MiB

TODO src/engine.cpp:8753,8811: [sev:P1] Establish per-model MTP acceptance and net speed benefit with output equality | fix: do not default-enable; consider auto window only after cost data | test: plain vs MTP k=1/2/4, 5 interleaved runs, token ID/hash equality, generated tok/s = emitted/decode wall and separate rounds/s

TODO tools/oracle_probe.cpp:630,885,915: [sev:P2] Make oracle’s “divergence exploded” status unambiguous and scope it to architectures it models | fix: strict exit status or label diagnostic-only, and adapt only if requested | test: demonstrate expected f16-vs-f32 tolerances and a constructed mismatch on an architecture-aware fixture

TODO src/engine.cpp:6175: [sev:P2] Keep experimental hot-ring pinning disabled until pool async reuse failure is fixed | fix: preserve opt-in guard, do not turn on globally | test: full long (>1024-token) Tiel decode repeated with normal async scheduling and output hash
```

## 19. T0 result — the streamed-weight re-read was the bottleneck (fixed, measured)

### What the code actually did

`Model::stage_layer()` ran, for every streamed layer, on every token:

```text
be_->sync()                      // drain everything queued
read_layer_to(read_buf_[slot])   // BLOCKING read of ~213 MiB out of the GGUF mapping
upload_layer_from(...)           // async DMA from that pinned buffer, returns on enqueue
read_layer_to(read_buf_[other])  // blocking read of the NEXT layer
```

A streamed layer's bytes never change between tokens. The mapping is page-cache hot, so this was not a disk cost — it was the read path repeating work that had already been done, once per layer, per token.

### The misleading evidence worth naming

The old startup line said *"12737 MiB already held by the driver/other processes"*. It is measured **after** the weights are uploaded, so it counts Kraken's own allocation (16304 − 3567 = 12737 ≈ its 12918 MiB live set). Total per-process GPU dedicated usage on the box was ~1.1 GiB. There was **no external contention**; the earlier report's "GPU was heavily occupied" caveat was wrong, and the fix is not blocked by another application.

### The change

A **pinned host mirror** of the streamed layers (`include/krk/model.hpp`, `src/model.cpp`):

- one `alloc_host` allocation holding every streamed layer at its own offset, sized to the set (4.13 GiB on the 27B, 1.74 GiB on the 8B arm);
- filled **once at load** through the same pooled reader a staging pass used (`read_layer_to`, so the bytes are identical by construction);
- a staging pass afterwards is `sync` + `upload_pinned_batch` straight out of the mirror — no read at all;
- promotion uses the same tensors, and the mirror is released in `unload()`;
- controls: `KRK_WEIGHT_HOST_CACHE=0` restores the old path, `KRK_WEIGHT_HOST_CACHE_MB=<n>` changes the 8 GiB default cap. A set above the cap, or a failed allocation, falls back to the read path with a warning (slower, still correct).

Pinned memory is the faster source on this card by measurement: 16.4 GB/s pinned against 8.9 GB/s through the narrow-aperture staging ring.

### Measured

`ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` (15.97 GiB), `--greedy --ctx 512 --chunk 64 --bench`, 32 generated tokens, RX 9070 XT:

| Arm | Prefill | Decode | ms/step | Staging | Read | Mirror fill | Load |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 22.3 tok/s | 2.9 tok/s | 349.3 | 64.70 GiB / 10678 ms = 6.06 GB/s | 819 ms | — | 2848 ms |
| after | 26.2 tok/s | **6.2 tok/s** | **160.4** | 64.70 GiB / 4691 ms = **13.79 GB/s** | **0 ms** | 4.13 GiB / 439 ms | 3483 ms |

`Qwen3-8B-Q4_K_M` at `--weight-vram-mb 3072` with `KRK_WEIGHT_PROMOTE=0` (the user's original 2.6 tok/s streaming case, held streaming so the two arms are actually comparable):

| Arm | Decode | Staging |
|---|---:|---|
| host cache off | 3.5 tok/s | 55.52 GiB / 9318 ms = 5.96 GB/s |
| host cache on | **12.1 tok/s** | 55.52 GiB / 2919 ms = **19.02 GB/s** |

**3.46x** on the small streaming arm, **2.14x** on the large one.

### Correctness

- Greedy output on the 27B, same prompt and flags, mirror on versus off: **byte-identical**, md5 `b7525fa7dcf3398d4dffd10b432f0812`, 289 bytes both.
- Full suite: **4149/4149 checks passed**, exit 0.
- The one initially failing group (`test_http_server`, 102/103) was a missing `kraken-server.exe` beside `kraken-tests.exe`, not this change: after building it the group passed 118/118 and the suite went green. No source change was made to make a test pass.

### What the new ceiling is

After the fix the step time is almost exactly the DMA flight time: 2.02 GiB/token at 13.8 GB/s ≈ 147 ms, observed 160 ms/step. The 8B arm is the same shape (1.74 GiB/token at 19.0 GB/s ≈ 92 ms). **The transport is now at the card's copy limit, so the next lever is fewer bytes, not a faster path** — see T0b above.

Do not chase the remaining time with GEMV kernel work on this file. The kernel time is no longer what dominates; the streamed-byte count is.

## Final diagnosis / handoff

- **Prefill is not secretly on CPU.** The user’s RX 9070 XT runs were on HIP. There is a large, reproducible exact-vs-fast prompt throughput difference (about 6.4x for the short Qwen8 probe), but fast changes the numeric/chunk-invariance contract. Large MoE prompt prefill also incurs cold expert I/O and allocation, so normalize page-cache/warmup state.
- **Dense large-model decode: fixed the dominant half.** The streamed-weight path was re-reading unchanged bytes from the GGUF mapping every token. A pinned host mirror filled once at load took the 27B from **2.9 to 6.2 tok/s** and an 8B forced-streaming arm from **3.5 to 12.1 tok/s**, output byte-identical (§19). The step is now bounded by the PCIe copy itself, so the remaining work is fewer streamed bytes (T0b), not kernels. MoE host routing/readback is real but not proven dominant for every model; measure route spans without double-counting waits. Expert cache must be interpreted by HOT/WARM/in-place/COLD and phase split.
- **MTP is wired, not necessarily worthwhile.** Tested Qwen3.5 9B k=2 emitted 64 tokens at ~62/s, worse than plain 82/s. The MTP-specific round-vs-token throughput reporting bug was fixed and verified in `src/main_cli.cpp`.
- **Oracle ran but did not establish a clean pass/fail.** It exited zero, prints divergence at L06 due CPU-f32 versus HIP-f16 numerical accumulation, and final argmax matched. It is not the needed exact-model performance profiler.

**HANDOFF: T0 shipped and verified.** The streamed-weight re-read was the dominant cost and is now a one-time load instead of a per-token read: 27B **2.9 → 6.2 tok/s**, 8B streaming **3.5 → 12.1 tok/s (3.46x)**, byte-identical output, suite **4149/4149**. The transfer path now runs at the card's copy limit (13.8–19.0 GB/s), so the next task is **T0b: stream fewer bytes per token** by sizing the load-time slack from the real KV+workspace need instead of a flat 1 GiB, which is currently leaving ~1.9 GiB of the 27B re-staged per token while 1.17 GiB sits free below a 1 GiB floor. Do not spend effort on GEMV kernels for this file until the streamed-byte count is down: kernel time no longer dominates.
## 20. Consolidated logs and the second fix (promotion DMA)

### Where the traces are

Every arm now logs its full output, so a rate is never separated from the phase it
came from and the op that owns it:

| path | what it holds |
|---|---|
| [`perf-traces/collect.sh`](perf-traces/collect.sh) | the arm matrix that produces them |
| [`perf-traces/SUMMARY.md`](perf-traces/SUMMARY.md) | one table of every arm's rate, phase split and top device ops |
| [`perf-traces/summarize.py`](perf-traces/summarize.py) | regenerates the summary from the logs |
| `perf-traces/<arm>.out` / `.err` | that arm's stdout and stderr: rates, host phase table, `KRK_TIME` op table, residency ledger |

`bash perf-traces/collect.sh [arm ...]` with arms `dense8 dense27 moe30 moe35 mtp oracle`;
`python perf-traces/summarize.py` rebuilds the summary. The logs are one process per
arm, run back to back, so page-cache and clock state are shared but not controlled —
which matters for the MoE arms specifically (§20.4).

### 20.1 The fix: promotion was on the pageable copy path

`promote_layer()` reads each batch into `stage_host_` — memory from `alloc_host()`,
i.e. already pinned — and then called **`upload_paged_batch`**, the pageable-source
call whose whole purpose is to bounce the source through the 4 MiB staging ring. The
trace priced it exactly:

| op | arm | bytes | rate |
|---|---|---:|---:|
| `upload_paged` | promotions, 27B | 2 267 MiB | **9.8 GB/s** |
| `upload_pinned` | streamed staging, 27B | 34 735 MiB | **27.6 GB/s** |

Same card, same run, same direction. The pageable call pays a host memcpy per chunk
for a source that needs none. Issuing the pinned batch instead:

| 27B Qwen3.8 Q4_K_M, `--prompt hi --max-tokens 16 --ctx 512` | before | after |
|---|---:|---:|
| 10 promoted layers, one-off staging | 764–1662 ms | **455–513 ms** |
| first-token prefill (promotion is charged there) | 792 ms | **595 ms** |
| decode, 16 tokens | 2 883 ms (5.2 tok/s) | **2 585 ms (5.8 tok/s)** |

Greedy output is byte-identical with promotion on and off (24 tokens, md5
`62dfed13c16a7878c620e98d15bfd681` both arms); suite 4149/4149.

### 20.2 What the consolidated traces say — dense prefill

Same 191-token prompt, `--chunk 256`, `ctx 512`, one process per arm:

| arm | prefill | decode | the op that owns the prefill |
|---|---:|---:|---|
| 8B `--prefill exact` | 163.1 tok/s | 69.1 tok/s | `gemm(grid)` 1 128 ms, 4 160 MB → **3.7 GB/s** |
| 8B `--prefill fast` | **1 256.6 tok/s** | 70.5 tok/s | `gemm(wmma)` (~150 ms total phase) |
| 27B `--prefill exact` | 45.9 tok/s | 5.2 tok/s | `gemm(grid)` **3 005 ms**, 5.1 GB/s |
| 27B `--prefill fast` | **153.7 tok/s** | 4.9 tok/s | `gemm(wmma)` **758 ms**, 20.3 GB/s |

**The prefill gap is the exact/WMMA mode switch, and nothing else in the trace comes
close**: 7.7x on the 8B and 3.35x on the 27B in the first collection, 4.9x and 3.7x
in the repeat — the phase is only 150-290 ms on the 8B, so its absolute timing moves
between runs while the direction never does. It is not a CPU fallback, not a missing
kernel, and not the MoE cache — `gemm(grid)` is the kernel the trace blames, and its
low "GB/s" is the misleading part: the byte count is fixed while the time grows with
row count, because the kernel is bound by scalar dequant ALU (one warp per
output row, `dot_chunks` per row), not by memory.

Two things follow, and they are different decisions:

- **Inside exact there is no large win left.** The kernel is at ~27% of scalar FMA
  peak and the extra work is the per-value unpack, which is the price of the
  association it preserves. `KRK_GRID_G` was swept at 8/16/32/64 (the only knob the
  header claims is bit-identical): 549.6 / 538.5 / 539.7 / 550.4 ms — flat, so the
  redundant weight re-read it bounds is not the constraint either.
- **`wmma_cfg::MB` was tried and rejected on measurement.** The header argues MB > 1
  divides weight re-staging by MB and is bit-identical; set to 2 it measured
  **152 → 275 ms** on the same 8B prefill (1 257 → 695 tok/s), i.e. worse, and it is
  reverted to 1. The comment's traffic model is not what the kernel is bound by.
- **The decision that does move prefill is the default mode.** `--prefill exact` is
  the default because its arithmetic is chunk-invariant; `fast` is opt-in because it
  is not. That contract is a policy choice, not a defect, so it is left alone here
  and stated as the top recommendation instead: a user who wants prefill speed at
  the cost of chunk-invariance runs `--prefill fast` and gets 3.4-7.7x immediately.

### 20.3 What the consolidated traces say — token generation

- **27B, streaming:** `upload_pinned` (the staging DMA) is 1 259 ms of a 2 883 ms
  decode at **27.6 GB/s** — the link, not the read. Removing 42% of the streamed
  bytes (`KRK_VRAM_RESERVE_MB=384`, 288 → 160 stagings) did **not** move decode
  (5.7 → 5.7 tok/s), so the copy is already hidden behind compute and this path has
  no further lever except fewer bytes for their own sake.
- **What the 27B step actually is:** 172 ms/token over 64 layers = 2.7 ms/layer
  against 222 MiB/layer of resident weights, i.e. ~82 GB/s effective — 7x off the
  bandwidth floor that the 8B reaches (4.68 GiB at 401 GB/s). It is the same
  `gemm_group` + `gemm_acc` shape on both models, so the difference is geometry
  (64 layers, 5 120 embd, 24/4 heads of 256, 248 320 vocab) and launch count
  (18 807 device ops for 16 tokens), not one bad kernel.
- **`gemm_acc(gemv)` at 183-186 GB/s against `gemm(gemv)` at 574-586 GB/s** in the
  same run, same kernel (`gemv_kernel<256, ACC>`), same per-call byte count
  (41-45 MB/call). That is a 3.2x gap between two call sites of one kernel and is
  the clearest remaining kernel-level defect in the dense decode path; it is
  ~19% of the 27B step. Cause is not established — the shapes and dtypes the two
  call sites receive need to be logged, which is the next measurement.

### 20.4 What the consolidated traces say — MoE and the expert cache

`warm_read(pool)` — the pooled reader that pulls expert bytes out of the GGUF
mapping — is the single largest row in every MoE arm:

| arm | `warm_read(pool)` | share of run | `upload_pinned` | `gemm(grid)` |
|---|---:|---:|---:|---:|
| 30B-A3B Q2_K stock | 1 646 ms / 3 036 calls | ~53% of decode | 453 ms @ 11.7 GB/s | 626 ms |
| 30B-A3B `KRK_MOE_BATCH=0` | (decode, per-expert) | — | 71 ms @ 74 GB/s | 710 ms |
| 35B-A3B Q4_K_XL stock | **5 037 ms / 160 calls** | largest single cost | 1 042 ms @ 7.3 GB/s | 728 ms |

**These arms are cold-page-cache runs and must not be read as steady state.** The
same 35B file replayed warm measures **145.7 tok/s prefill and 21.5 tok/s decode**
against 87.0 / 8.2 in the collected arm — a 2.6x difference from nothing but page
cache. `KRK_WARM_READERS` swept at 4/8/16 changed nothing (145.7 / 144.7 / 149.2
tok/s prefill), so the reader pool is not the constraint; the mapping being cold is.

Two MoE facts survive that caveat:

- `KRK_MOE_BATCH=0` is **not** a decode-only switch in effect: it measured 66.9 vs
  35.1 tok/s prefill and 6.2 vs 11.1 tok/s decode. The code's `batch_ok` requires
  `n == 1` (so the batch path cannot run on a prefill row), which means the prefill
  half of that difference is run-to-run state, not the switch. It needs interleaved
  repeats before anything is claimed — the same conclusion the earlier audit reached.
- **The MoE expert path re-reads from the mapping with no pinned mirror**, while the
  dense streamed path now has one. That asymmetry is the direct follow-up to §19:
  a WARM-tier expert that is evicted and re-acquired pays `warm_read(pool)` again.

### 20.5 Pareto, after the measurements

| # | item | measured size | status |
|---|---|---|---|
| 1 | Prefill exact vs WMMA | **3.4x (27B) / 7.7x (8B)** | Policy, not a bug. `--prefill fast` is the lever; default unchanged on purpose. |
| 2 | Dense decode at 7x its bandwidth floor (27B) | 172 ms/step vs ~25 ms floor | Open. Needs the `gemm_acc` / `gemm_group` call-site shapes logged. |
| 3 | `gemm_acc(gemv)` 183 GB/s vs `gemm(gemv)` 586 GB/s | not a kernel defect | **Answered in §21**: same instantiation, and `KRK_ACC_GEMM` off changes nothing (5.6/5.9/5.8 vs 5.6/5.8/5.8). It was a missing-byte artifact — `gemm_group` booked no bytes. With the counts in, the 27B's three projections are 168 / 184 / 555 GB/s, and the same kernels hit 507-564 GB/s on the 8B. The lead is the quant format (27B is Q8_0/Q6_K, 8B is Q4_K), not the tile. |
| 4 | MoE `warm_read(pool)` | 5.0 s of a 23 s 35B run (cold cache) | Open. Needs a pinned WARM mirror, not a faster reader (swept, flat). |
| 5 | Load: 11.83 GiB at 6.4 GB/s | 1.8-1.9 s of every 27B start | Open. The pinned path does 27.6 GB/s; `upload`'s 1 334 ms of in-span idle is the pageable bounce. |
| 6 | MTP | 30.8 → 21.2 tok/s | Confirmed loss on the 9B at 191-token context; wired, not profitable there. |
| 7 | Streamed weights / promotion | 2.9 → 6.2 tok/s, promotion 764-1662 → 455-513 ms | **Fixed and verified** (§19, §20.1). |

### 20.6 Still true from §19

The 27B cannot be made to fit: 15.97 GiB of weights on a 15.9 GiB card. Streaming is
correct and now cheap; the decode floor for this model on this card is the resident
13.9 GiB at the achievable bandwidth, and the trace says it runs 7x off that floor.
That gap — not the streaming — is what is left to fix.
## 21. `gemm_acc` at 183 GB/s vs `gemm` at 586 GB/s — answered

### 21.1 It is not the kernel

Both call sites run the **same instantiation**, `gemv_kernel<256, ACC>`, with the
**same grid** `(n_out + 7) / 8` and 256 threads (`src/hip/backend_hip.hip`,
`gemm_accumulate` vs `gemm`'s `rows == 1` branch). The only difference is the folded
residual epilogue.

That epilogue was A/B'd on the 27B, three interleaved pairs, `--prompt hi
--max-tokens 16 --ctx 512`, `KRK_ACC_GEMM=1` (folded) against `=0` (gemm +
add_residual, two launches):

| pair | folded | unfolded |
|---|---:|---:|
| 1 | 5.6 tok/s (178.7 ms/step) | 5.6 tok/s (179.4 ms/step) |
| 2 | 5.9 tok/s (169.2 ms/step) | 5.8 tok/s (172.2 ms/step) |
| 3 | 5.8 tok/s (171.0 ms/step) | 5.8 tok/s (173.2 ms/step) |

A path that were 3x slower per byte would move those numbers. It does not.

### 21.2 The 183 GB/s figure was a missing-byte artifact, and it hid the real one

`gemm_group`, the **largest single decode span on the 27B** (1 520 calls, 43 ms of a
172 ms step), booked **no bytes at all** — every row read `0.0 MB`. With the counts
added (`gemm_group(fused)` retagged apart from its per-matrix fallback, which books
through the nested `gemm()` spans and must not be counted twice), the same run reads:

| op | calls | NET ms | MB | GB/s |
|---|---:|---:|---:|---:|
| `gemm_group(fused)` | 1 520 | 694.9 | 117 004 | **168.4** |
| `gemm_acc(gemv)` | 1 792 | 403.1 | 74 056 | **183.7** |
| `gemm(gemv)` | 1 584 | 129.2 | 71 652 | **554.8** |
| `upload_pinned` (staging) | 154 | 1 365.1 | 37 002 | 27.1 |

The three projections together move **262.8 GB in 1 227 ms = 214 GB/s**: 33% of the
~585 GB/s this card measures. So `gemm_acc` was never the odd one out — it and
`gemm_group` are the *same* slow group, and only `gemm` (which carries the 1.35 GB
Q6_K LM head, `n_out = 248 320`) saturates the bus.

### 21.3 The same kernels on the 8B run at 96% of the bus

The control that settles it. Qwen3-8B Q4_K_M, `--prompt hi --max-tokens 32`:

| op | calls | NET ms | MB | GB/s |
|---|---:|---:|---:|---:|
| `gemm_group(fused)` | 1 728 | 130.1 | 73 384 | **564.1** |
| `gemm_acc(gemv)` | 1 152 | 79.1 | 40 089 | **507.0** |

Same kernels, same box, same recorder: **507-564 GB/s on the 8B, 168-184 GB/s on the
27B.** The kernels are not the constraint; what they are handed is.

### 21.4 The shapes, logged

`KRK_OP_SHAPES=1` (new, permanent — one set lookup per call, prints nothing unless
asked) prints each distinct `(op, weight type, n_out, n_in, rows)` once:

| op | the 27B receives |
|---|---|
| `gemm_acc` | Q8_0 / Q6_K / Q5_K / Q4_K, all `n_out=5120`, `n_in=6144` or `17408` |
| `gemm` | Q8_0 `5120x6144`, `17408x5120`, `1024x5120`; Q6_K `248320x5120`, `17408x5120`, `12288x5120`; Q4_K `17408x5120`, `12288x5120` |
| `gemm_group` (fused) | q/k/v and gate/up groups of the same layers |

The 8B is Q4_K throughout with `n_out` 4 096-14 336. The 27B is mostly **Q8_0 and
Q6_K** with `n_out = 5120`. Two variables move together and this run did not separate
them, but the sizes ruled `n_out` out on its own: the 27B's `gemm_group` includes
`17408`-wide gates, and the 27B's `gemm` hits 555 GB/s at `n_out = 5120` (`Q8_0`,
5 120x6 144). **The lead was the quant format, not the tile — and §22 measures it and
confirms it**: at a fixed shape, Q6_K runs at 0.33-0.44x Q4_K's values/s and Q5_K at
0.55-0.65x, across a 4x span of `n_out` that itself moves the number by only 10-20%.

### 21.5 Rejected on measurement, twice

- **Pinned bounce ring for the load path.** The load uploads 11.83 GiB at 6.4-6.9 GB/s
  through `be.upload` (pageable: the driver stages it and the call does not return
  until the copy lands). Bouncing each tensor through 8 pinned 32 MiB slots and
  issuing `upload_pinned_batch` instead — the path that measures 27.1 GB/s — was
  measured at **5.0-5.1 GB/s**, i.e. slower, and reverted. The host-side copy into
  pinned memory is not free, and the driver's own staging beats
  `memcpy` + DMA + ring bookkeeping at these sizes. Load stays as it was; the
  remaining load lever is fewer/bigger copies, not a different bounce.
- **`wmma_cfg::MB=2`** (§20.2): 152 → 275 ms on the same 8B prefill, reverted.

### 21.6 What is common to prefill, decode and the expert cache

One pattern explains every slow arm in the trace, and it is not the same as any of the
individual hypotheses:

| phase | the shared shape of the problem | measured |
|---|---|---|
| dense prefill, exact | weights are re-dequantized per output row on scalar ALU | `gemm(grid)` 3.7-5.9 GB/s, 1 128 / 3 005 ms |
| dense prefill, WMMA | weight tile re-staged per M tile | `gemm(wmma)` 20.3 GB/s, 758 ms |
| dense decode, 27B | projections 3x off the bus while one wide GEMM saturates it | 214 GB/s aggregate vs 585 |
| MoE decode/prefill | expert bytes come out of the GGUF mapping with no pinned mirror | `warm_read(pool)` 1.6-5.0 s, the largest row in every MoE arm |
| load | one copy per tensor, pageable, and rate set by the copy count | 11.83 GiB at 6.4-6.9 GB/s |

In every case the device is waiting on **how the bytes arrive**, not on arithmetic.
That is why the two fixes that worked (§19 mirror, §20.1 pinned promotion) both
removed a staging step rather than changing a kernel — and why the largest remaining
item, a pinned WARM mirror for MoE experts, is the same fix in the same shape.
## 22. Is the 27B's slow projection the Q8_0/Q6_K path or the shape? — answered: the format

The trace could not answer this because every `gemm*` row is an average over every
dtype the model uses, and the two models in hand move two variables at once: the 27B
is mostly Q8_0/Q6_K at `n_out=5120`, the 8B is all Q4_K at `n_out` 4096-14336.
`gemm_group(fused)` measures 167.8 GB/s on one and 563.4 GB/s on the other.

So it was measured directly: [`tools/probe_gemv_dtype.hip`](tools/probe_gemv_dtype.hip)
runs the same `gemv_kernel<256>`, same grid formula, B=1, back to back with nothing
else in the queue — each model's real shapes crossed with each format, **with 256 MiB
of L2 flushed before every timed call**. Output:
[`perf-traces/probe-gemv-dtype.txt`](perf-traces/probe-gemv-dtype.txt).

The flush is not a detail. The first version of the probe did not flush and reported
1 383 GB/s for Q8_0 at 5120x6144 and 1 700 GB/s for F16 — L2 bandwidth on a card
whose HBM does ~600, for a matrix the model never sees twice in a row. Those numbers
were thrown away; the table below is the flushed one.

Values/s at a **fixed** shape, which is the only way to compare formats (GB/s moves
with bytes per value, so it is not):

| shape | Q4_K | Q5_K | Q6_K | Q8_0 | F16 |
|---|---:|---:|---:|---:|---:|
| 5120x6144 (27B o_proj) | **1 146** | 626 | 422 | 702 | 334 |
| 5120x17408 (27B down_proj) | **1 040** | 683 | 508 | 606 | 315 |
| 17408x5120 (27B gate/up) | 997 | 654 | 473 | 595 | 314 |
| 248320x5120 (27B lm_head) | **961** | 687 | 497 | 593 | 317 |
| 4096x4096 (8B q/o) | **1 449** | 800 | 535 | 1 030 | 374 |
| 14336x4096 (8B up/gate) | **1 264** | 766 | 480 | 636 | 342 |

**The format is the variable, and it is large: Q6_K runs at 0.33-0.44x Q4_K's values/s
at every shape, Q5_K at 0.55-0.65x, Q8_0 at 0.47-1.0x.** The shape changes the number
by ~10-20% across a 4x span of `n_out`; the format changes it by 2-3x. The 27B's
weights are largely Q6_K and Q5_K; the 8B's are Q4_K. That, not the tile, is why one
model's projections sit at 168 GB/s and the other's at 563.

Both 5/6-bit paths are already on the packed form (`dot_chunk_q6k`/`dot_chunk_q5k`
behind `KRK_HAS_DOT2`, which the header records as 143 -> 541 GB/s on gfx1201), so the
remaining gap is instruction count in the unpack, not a missed fast path — the generic
fallback would be at 143 GB/s, and nothing here is that low.

### What this means for a low-VRAM box

The format is the lever, and it pays twice:

| choice | weight bytes | projection throughput |
|---|---|---|
| Q4_K | 0.5625 B/value | **the fast path** (961-1 449 Gval/s) |
| Q5_K | 0.6875 B/value | +22% bytes, 0.55-0.65x values/s |
| Q6_K | 0.8203 B/value | +46% bytes, 0.33-0.44x values/s |
| Q8_0 | 1.0625 B/value | +89% bytes, 0.47-1.0x values/s |

A Q6_K file buys accuracy it cannot spend: it is 46% larger *and* 2-3x slower per
weight value than the Q4_K file of the same model, which is the opposite of what a
15.9 GiB card wants. For this hardware the recommendation is Q4_K-class (or IQ4)
weights first, and Q6_K only where a specific tensor has measured a numerical need.

### Reproduce

```bash
# build (the KRK_BUILD_PROBES CMake block does not configure: it lists sources
# that are not in the tree, so this target is built directly -- the full command
# is in the file's header)
./build-hip/probe_gemv_dtype.exe | tee perf-traces/probe-gemv-dtype.txt
```

### What this does *not* settle

The model still runs its projections at ~0.4x the flushed probe rate for the same
shape and dtype (27B `gemm_acc` 180 GB/s aggregate against 407-746 GB/s for its own
formats). That residual is the next question, and it is a *scheduling* question —
concurrent staging DMA, TLB pressure on a 13.9 GiB resident set, and 64 layers of
serial dependencies — not a kernel one, because the kernel at that shape and format
has now been measured on its own.
### 22.1 The in-situ gap, attacked with the same probe

§22 ends with a residual: the model's projections run at ~0.4x the rate the probe gives
the same shape and dtype. Three explanations were testable with the probe alone, and
all three are now eliminated — each by a measurement that had to be corrected first.

| hypothesis | probe addition | result | verdict |
|---|---|---|---|
| compressed / zero-filled weights flatter the kernel | fill with an xorshift stream instead of zeros (scale still 1.0) | rates move by <5%, Q4_K *up* 645 → 753 GB/s | not it |
| a 3.4 GiB working set of distinct matrices is slower than one matrix | 64 matrices at the 27B's own shapes/dtypes (o_proj + down_proj, Q6_K/Q8_0 alternating), L2 flushed per matrix | **473 GB/s**, against 490-700 one at a time | not it |
| sustained load downclocks the card | 8 back-to-back walks of the same working set | 463 / 467 / 469 / 473 / 471 / 474 / 477 / 472 GB/s — flat | not it |

The working-set measurement was wrong the first time and is worth recording: timing the
whole walk put the 256 MiB L2 flush of every step **inside** the window and read
101 GB/s — a number that was ~75% flush. The probe now times each matrix between its own
event pair with the flush outside, which is the same discipline the per-shape table uses.

So: same kernel, same shapes, same formats, same 3.4 GiB working-set size, sustained for
8 passes, no decay — **473 GB/s**. The engine gets 168-180 GB/s per op and 213 GB/s
across the three projection kernels. The gap is real, and it is inside the engine's
environment, not in the kernel arithmetic or the memory layout of the weights.

### 22.2 What is left, and what it needs

Eliminated for the engine side, each by an existing measurement:

- **concurrent staging DMA** — `KRK_VRAM_RESERVE_MB=384` halved the streamed bytes
  (64.70 → 37.53 GiB, 288 → 160 stagings) and decode did not move: 5.7 tok/s either way;
- **the fused epilogue** — `KRK_ACC_GEMM=0` (three interleaved pairs) 5.6/5.9/5.8 vs
  5.6/5.8/5.8 tok/s;
- **the projection kernel itself** — the probe above, and `probe_gemv` before it;
- **L2, page locality and DRAM row behaviour** — the working-set and sustained rows;
- **power state** — the flat sustained sequence.

What has not been measured, and cannot be with these tools: per-CU occupancy,
memory-request stall reasons and TLB miss counts for the engine's own launches. That
needs `rocprofv3`/`rocprofiler-sdk` counters on a decode trace, not another timing
probe. Until that is captured, the honest statement is: **the engine runs its
projections 2.2-2.8x below the same kernels measured standalone, and every cause that
timing alone can test has been ruled out.**

The one candidate the trace does point at is allocation layout: the engine holds **817
separate device buffers** for the 27B (`vram ledger ... allocator live 14860 in 817
buffers, largest 995`) while the probe's working set is 64. Fragmentation and TLB
coverage are the next thing to test, and the test is cheap — allocate the working set
the engine's way (many small hipMalloc buffers) and re-run the walk.
### 22.3 The gap is STARVATION, not a slow kernel — and the trace's GB/s column was lying

Three cheap hypotheses about the engine's 168-180 GB/s were tested and all three are
dead:

| hypothesis | test | result |
|---|---|---|
| zero-filled weights flatter the kernel | xorshift fill instead of zeros (scale still 1.0) | <5% change, Q4_K *up* 645 → 753 GB/s |
| a 3.4 GiB set of 64 distinct matrices is slower | 64 matrices at the 27B's own shapes/formats, L2 flushed per matrix | **488 GB/s** (vs 490-700 for one matrix) |
| the engine's scattered allocation layout | same walk rebuilt with **1 536 small buffers interleaved** the way load allocates them | **505 GB/s** — no degradation |
| sustained load downclocks | 8 back-to-back walks | 463-478 GB/s, flat |

So the last instrument to try was an **inside-the-engine trap** rather than another
external probe: `KRK_ACC_REPEAT=<n>` (`src/hip/backend_hip.hip`) re-runs the *same*
`gemv_kernel<256, true>` on the *same* data immediately after the real launch, timing
each repeat on its own event pair. It is a diagnostic that perturbs output and says so
in every line it prints; it is never a mode.

27B decode with `KRK_ACC_REPEAT=3`:

| shape / format | in-engine repeat | probe (flushed, same shape) |
|---|---:|---:|
| 5120x6144 Q8_0 (31.9 MiB) | 1 177 / 552 / 578 GB/s | 700 |
| 5120x17408 Q6_K (69.7 MiB) | 373 / 332 / 331 GB/s | 515 |
| 5120x17408 Q4_K (47.8 MiB) | 453 / 404 / 355 GB/s | 549 |

**The same kernel, on the same weights, in the engine, in the same token, runs at
330-450 GB/s.** The trace had been reporting 168-180 GB/s for it.

That settles two things at once:

1. **The kernel is not the problem in situ.** For the shapes that exceed L2 the
   in-engine rate is 331-453 GB/s against the probe's 515-549: a residual 1.2-1.6x,
   not the 2.7x the op table implied.
2. **The `KRK_TIME` GB/s column over-attributes time to the op.** An op row's span is
   not pure kernel execution here; it carries the wait in front of the launch. Read as
   `bytes / span`, a 40 ms/token kernel arrives as a 77 ms/token row. The table is
   still right about *which* op dominates the step — it is wrong about *why*, and §21
   built a whole investigation on the wrong reading of it.

With the projections at ~41 ms/token of real kernel time against ~77 ms of attributed
span, roughly **36 ms of every 172 ms step is starvation** — the device waiting between
launches — and the same order of magnitude shows up as the 27% "idle between ops" the
recorder reports for the whole run. That is a scheduling target, not a kernel target.

### 22.4 What this changes about the earlier numbers

- §21's "214 GB/s aggregate, 33% of the bus" is a **span** figure. The projection
  kernels really run at 330-450 GB/s, i.e. 60-80% of the measured bus, and the loss is
  the queued wait the span absorbs.
- The one measurement that stands unchanged is §22's format table: those are single
  launches with their own event pair and an L2 flush, so they are kernel-only by
  construction. Q6_K at 0.33-0.44x Q4_K per weight value still holds, and it is now the
  *only* large, quantified, actionable inefficiency in the decode path.
- The next instrument is a launch-gap trace (device timestamps of consecutive kernels'
  start and end), not another rate. That is what turns "36 ms of waiting" into a named
  call site.
### 22.5 The step, read off one decode timeline

The aggregate op table cannot answer "what is in a token": it averages prefill with
decode and a 40 ms projection with a 40 µs norm. The engine's own timeline
(`KRK_PROFILE=1`, one decode step, in issue order with each op's span and the idle it
waited through) can, and [`perf-traces/timeline_agg.py`](perf-traces/timeline_agg.py)
now sums it per op. 27B, one step, 191.5 ms wall:

| op | calls | dev ms | idle ms | MB | GB/s |
|---|---:|---:|---:|---:|---:|
| **`upload_pinned`** (staging DMA) | 9 | **78.92** | 0.76 | 2 171 | 27.5 |
| `gemm_group(fused)` | 95 | **46.36** | 0.39 | 7 313 | 157.7 |
| `gemm_acc(gemv)` | 112 | **27.52** | 1.24 | 4 629 | 168.2 |
| `gemm(gemv)` | 99 | 10.31 | 2.64 | 4 478 | 434.3 |
| `softplus_act` | 48 | 1.88 | 1.14 | — | — |
| `delta_rule` | 48 | 1.71 | 0.82 | — | — |
| `rmsnorm` | 161 | 1.56 | 2.00 | — | — |
| everything else | — | ~5 | ~4 | — | — |
| **step** | 1 104 | **174.3** | **17.9** | | |

Two facts, both new and both decisive:

1. **The staging DMA is 79 of the 191 ms — 41% of the token — and it is on the
   critical path.** Nine streamed layers, 2 171 MB, at the link's own 27.5 GB/s. Its
   spans sit end to end with the kernels rather than beside them: the span total
   (174.3 ms) is 91% of the wall and the idle *inside* spans is 17.9 ms, so nothing is
   overlapping. This also re-reads the earlier `KRK_VRAM_RESERVE_MB=384` result, which
   cut staging from 288 to 160 passes and moved decode by nothing (5.7 → 5.7 tok/s) —
   a contradiction that is now the top open question, not a settled one.
2. **The two projections that carry 96% of the step's weight bytes run at 158 and
   168 GB/s, with almost no idle inside their spans** (0.39 and 1.24 ms). Combined with
   §22.3 — the same kernels re-launched back-to-back measure 330-450 GB/s — this is a
   launch that waits *before* it starts, and the thing it waits for is the previous
   layer's DMA.

So the shape of the problem is: **a token is one serial alternating chain of
weight-transfer and weight-compute, at 27.5 GB/s and ~160 GB/s respectively.** The two
halves are 79 ms and 84 ms, i.e. the same size. Overlapping them is the single
highest-value fix left in the decode path — a double-buffered staging ring (two device
slots, the next layer's DMA issued while the current layer computes), which is the same
pipeline the read path already uses one level up (`prefetch_host_`). It is not a kernel
change and it does not change arithmetic: with a second ring slot the layer's tensors
are re-pointed exactly as `promote_layer` already does.

Expected ceiling, from this table: 191 ms → ~110-120 ms, i.e. **5.2 → 8.5-9 tok/s**,
with the same bytes and the same kernels. That is a bigger win than every kernel
optimisation in this report combined, and it is scoped as the next change rather than
attempted at the end of this one.

---

## 22.6 The double-buffered ring, built and measured: the prediction above was wrong

§22.5 predicted 191 ms → ~110-120 ms (5.2 → 8.5-9 tok/s) from overlapping the staging DMA
with the projections. That was arithmetic on totals — 79 ms of copy beside 84 ms of
compute — and it assumed the copy could run *alongside* the step. It was built, and it
cannot. **The prediction is withdrawn.**

### What was built

- The ring is two slots, a layer's slot chosen by **parity** so its tensor addresses are
  fixed at load (nothing is re-pointed per token while the GPU may read it).
- `Model::issue_ahead()` issues the next streamed layer's copy at the end of the current
  layer's staging pass, through the new `Backend::prefetch_after()`: a pinned-source DMA
  on the private copy stream, ordered behind an event recorded on the default stream
  (`ready`, which covers the previous reader of the target slot).
- The default stream is ordered behind that copy at the point the layer runs
  (`wait_marker_stream`) — a device-side wait, no host block.
- Falls back to the single-slot path for a cold layer, for a missing pinned mirror, and
  when the second slot does not fit; `alloc()` aborts on failure, so free VRAM is checked
  first. Promotion copies from the layer's own slot base (a promotion from the wrong base
  would have read the neighbour's bytes).
- `KRK_WEIGHT_AHEAD=1` enables it; **the default is off** (see below).

It works: **159 of 160 staging passes consumed a copy that was already in flight**, one
cold, and the staging line drops from `2657 ms of drain` to `22 ms of enqueue`.

### What it did to the step

27B Q4_K_M, identical command, identical resident set (`--weight-vram-mb 12000`), greedy,
16 tokens, md5 of the output **identical** in both arms:

| arm | step | decode | staging as measured |
|---|---:|---:|---|
| one slot | 192.2 ms | **5.2 tok/s** | 2657 ms drain over the run |
| two slots | 197.0 ms | 5.1 tok/s | 22 ms of enqueue, copies in flight |

The engine's own timeline says where the 79 ms went — it did not go away, it changed
shape (`perf-traces/tl-ahead.err`, one step, 195.9 ms wall):

| | spans | device idle |
|---|---:|---:|
| one slot | 174.3 ms | 17.9 ms |
| two slots | **107.6 ms** | **89.6 ms** |

Span total fell by 67 ms and idle rose by 72 ms. The copy left the default stream's
spans and the default stream now waits for it at the next layer.

### Why, and why no slot count fixes it at this VRAM

The 2.2 GiB a token streams crosses the link at 27.5 GB/s: **83 ms**. The whole step's
compute is **107 ms**. Perfect overlap would be 107-120 ms — but overlap needs somewhere
to put the bytes, and the bytes are concentrated in 9 layers:

| per streamed layer | time |
|---|---:|
| its copy over the link | **9.2 ms** |
| its own compute | **1.7 ms** |

A two-slot window's cover is exactly *one layer of compute* — 1.7 ms of cover for a 9.2 ms
copy. The copy stream cannot be kept busy, so the copies serialize and each exposes ~7.5 ms.
To hide the whole 83 ms under the resident prefix's 94 ms you must hold **all nine**
streamed layers at once: 2.2 GiB of slots, which is the residency the streaming exists to
avoid. Four slots (the most the card's headroom allows at 253 MiB each) buys 3 layers of
cover, ~5 ms of a 9.2 ms copy.

The general rule this leaves behind: **streaming pays only when a layer's compute exceeds
its copy.** On PCIe-4.0-class links a dense layer's weights (200-260 MiB) cost 8-10 ms to
cross and 1-2 ms to read from HBM, so the ratio is 5:1 the wrong way for every dense stack
at batch 1 — and for prefill at `chunk 256-512` it is still ~2:1 the wrong way.

That is why §22.5's other loose end resolves too: `KRK_VRAM_RESERVE_MB=384` cut streamed
bytes in half and decode did not move, because it also halved the resident prefix that the
copies could have hidden under. Bytes and cover were being traded one for one.

### Where that redirects the effort

Not scheduling. The floor for this model on this card is set by bytes over the link:

1. **Fewer streamed bytes** — higher residency (the promotion path already reclaims 10 of
   19 layers on first use), smaller quantisation, or a smaller model. This is the only
   lever that moves the 83 ms.
2. **Fewer bytes per weight value** — §22's format table still stands: Q4_K is both the
   smallest and the fastest per value; Q6_K is 46% larger *and* 2-3x slower.
3. **A faster link** than PCIe 4.0 x16 — the only change that makes the pipeline above pay.

The code stays behind `KRK_WEIGHT_AHEAD=1` rather than default-on: at 342 MiB of VRAM it is
2.5% slower on the target model, and the single-slot path is what the suite covers.
Verified: `kraken` and `kraken-tests` rebuilt clean, **4149/4149**, greedy output md5
identical between the arms (`2e92586887d67b2959faa8d3577a4ef2`).

---

## 22.7 Residency is not the lever either — and the MoE step is 74% host round-trip

### The VRAM reclaim, measured

The 27B run ends with `free 1182 MiB` and a 1 GiB reserve (`krk::vram_reserve_bytes`).
Spending that reserve on residency buys, at `--ctx 512` where KV is 32 MiB:

| reserve | streamed layers | per-token streamed | decode |
|---:|---|---:|---:|
| 1024 MiB (default) | 9 | 2.17 GiB | 5.2 tok/s (191.9 ms) |
| 768 | 8 | 1.93 GiB | 5.2 (192.5) |
| 512 | 7 | 1.69 GiB | 5.2 (193.5) |
| 0 | 4 | 0.97 GiB | **5.4** (186.0 ms) |

**Five fewer streamed layers — 1.2 GiB less DMA — bought 6 ms.** At 27.5 GB/s those
bytes were worth 44 ms. So the staging DMA is *not* on the critical path in steady-state
decode, and §22.6's "bytes are the lever" is wrong in the same way §22.5's was.

What the step actually is: 15.6 GiB of dense weights read per token in 186-192 ms =
**~84 GB/s across the whole set**, against 158-168 GB/s per projection op and 450 GB/s
for the same kernels isolated. The fully resident arm is *slower still* (3.1 tok/s,
322 ms, `free 0.00 GiB`) — with no PCIe traffic at all. Residency, streaming and the
double buffer all trade against a ceiling that is none of them: the aggregate read rate
over a 15.6 GiB working set.

### The MoE: 57 ms of GPU work in a 216 ms step

30B-A3B Q2_K, 128 experts top-8, 48 layers, greedy, ctx 512 — **4.6 tok/s, 215.6 ms/step**
(`perf-traces/moe30-prof.err`):

| | ms per step |
|---|---:|
| device spans (all ops) | **56.8** |
| device idle inside spans | 21.3 |
| recorder host window | 76.8 |
| **step wall** | **215.6** |
| **unaccounted — host blocking, not in any span** | **~139** |

Inside those 57 ms, per token:

| op | calls | ms |
|---|---:|---:|
| `warm_read(pool)` (expert reads from the WARM tier) | 9 | 9.3 |
| `scatter_axpy` (routed accumulation) | 384 | 8.8 |
| `gemm` + `gemm_group` (experts + attention) | 385 | 17.0 |
| **`download_f32`** — router scores to the host | **48** | 5.4 |
| `upload` / `upload_i32` / `fill0` (ids, dispatch, buffers) | 48/48/96 | 6.9 |

**One router download, one expert-id upload and two buffer fills per layer per token —
48 blocking host round-trips.** A blocking download waits for everything already queued,
so each one drains the pipeline the previous layer built: that is the 139 ms. The kernels
are 57 ms of a 216 ms step, and the step is gated by the host exchanging control data with
the device 48 times.

This is the difference the low-level comparison turns on: a MoE step is only fast when
routing and dispatch stay on the device and the host never waits inside the layer loop.
Kraken has device-side dispatch paths (`moe.route.*`, the one-launch dispatch table), and
this trace says they are not the ones running for this model — the host path is, at
48 synchronisations a token. That is the measurement the Pareto list has been asking for
since §1, now with a number on it.

### Where that leaves the Pareto order

1. **MoE: keep routing on the device** — 48 downloads + 48 id uploads + 96 fills per
   token, ~139 ms of a 216 ms step. Largest measured target in the whole report.
2. **MoE: expert WARM reads inside decode** — 9 pool reads a token means the 10 GiB
   expert budget is missing on demand; at 16.4 GB/s pinned those are 9 ms the GPU spends
   idle.
3. **Dense: the aggregate read rate** (84 GB/s over 15.6 GiB, 5x below the kernel rate) —
   not residency, not staging, not the ring. Needs device counters (TLB/occupancy), which
   timing cannot resolve further.
4. Kernel-level and format-level work (§22) is real but secondary to all three.

---

## 22.8 The MoE host round-trip: the device plan was opt-in. It is now the default, and it is 74%

The 139 ms of §22.7 was not one thing. The host phase table from `KRK_TIME=1` splits the
202 ms decode step by named phase, and it says the routing bookkeeping is the step:

| phase (per token, 30B-A3B Q2_K, 48 layers) | ms |
|---|---:|
| `moe.route.rest` — host top-k, plan, prefetch orders (parent of the expert loop) | **158.8** |
| `moe.route.fetch` — the blocking router download inside it | 63.0 |
| `expert.warm.read` — COLD expert acquires, 1.165 ms each, 1.4 GB/s | **90.0** |
| `moe.route.gemm` | 0.3 |
| `moe.experts` (the device kernels, host-side span) | 0.1 |
| everything else (`decode.logits.dev`, upload, sample…) | < 4 |

`moe.experts` at 0.1 ms is the tell: the expert compute is *not* the cost, and the two
rows above it are host work. `Engine::moe_ffn()` already had a **device plan** arm —
`KRK_DEV_PLAN=1`, which runs the top-k and builds the layer's expert plan on the GPU and
then compares ids, weights and offsets against the host's every layer, falling back on any
mismatch. It was opt-in. On this model:

| arm | step | decode | `moe.route.rest` | `expert.warm.read` |
|---|---:|---:|---:|---:|
| host plan (was the default) | 215.6 ms | 4.6 tok/s | 3.309 ms/layer | 1444.8 ms/run at 1.4 GB/s |
| `KRK_DEV_PLAN=1` | 128.9 ms | 7.8 tok/s | 1.399 ms/layer | 509.7 ms/run at **4.1 GB/s** |
| **device plan as the default** | **125.5 ms** | **8.0 tok/s** | — | — |

Two effects, not one: the top-k and plan cost 2.4 ms/layer less, and the expert reads —
already host-side, already blocking — run **3x faster** because the plan's order reaches
them without the host selection in the middle. `moe.route.fetch` does not move (1.31 ms):
the blocking download is inherent while a paged expert cache needs the ids on the host, and
it is the next target rather than something this change claims.

`KRK_DEV_PTR=1` measured 8.3 tok/s on the same run, but its log says the dispatch never
engaged ("the routed set is not wholly device-resident… retrying"), so that number is not
attributable to it and it stays opt-in — its whole point is keeping no host bits, which
changes the router's arithmetic to the device's `expf`.

### Changed

`dev_plan()` now defaults **on**; `KRK_DEV_PLAN=0` restores the host path for an A/B. This
is a default change and nothing else — the arm was already implemented, already verified
per layer, and already fell back safely.

Verified with the default now in place:

- 30B-A3B Q2_K: **4.6 → 8.0 tok/s**, output md5 `fd5bc97754a31639d55443e03e9375a1`,
  identical to both earlier arms.
- Dense 27B Q4_K_M: **5.2 tok/s**, md5 `2e92586887d67b2959faa8d3577a4ef2`, unchanged.
- `kraken` and `kraken-tests` rebuilt clean, **4149/4149 checks**.

That is the answer to "why is a MoE so slow here": the engine had a device routing path,
the default was the host path, and the host path spent 159 of a 216 ms token on top-k
bookkeeping and blocking download-with-reads that the device path does in 67 ms. The
remaining MoE targets, in order, are the 1.31 ms/layer blocking fetch (63 ms/token) and
the 510 ms of expert warm reads (32 ms/token).

---

## 22.9 The blocking fetch is a WAIT, not a copy — so it cannot be removed. The plan rebuild can.

`KRK_ROUTER_2X=1` reads the router logits twice: the second readback is already-retired
device work, so the difference between the two rows is the only honest split.

| row (30B-A3B, decode, 768 layers) | ms/call |
|---|---:|
| `moe.route.fetch` (first readback) | 1.354 |
| `moe.route.fetch2` (same copy, device retired) | **0.086** |

The copy and the driver round trip are **86 µs**. The other **1.27 ms is the layer's own
queued device work retiring** — attention, norms, the router GEMM. The host is waiting
while the GPU is busy, so deleting that wait buys nothing: it is not idle device time.

What *is* still removable is the row above it: `moe.route.rest` = **1.399 ms/layer**
(67 ms/token) after the device plan — and that is now the host **rebuilding its own plan to
verify the device's**, layer after layer, on top of the 21.3 ms/token the device spends
idle waiting for the host. The device plan has been proven to agree on every layer it has
ever been asked about; the verification that proves it every layer is the remaining cost.

So the next change is not "remove the fetch" (impossible without device-side paging —
the paged expert cache genuinely needs the ids on the host) but **sample the verification
instead of paying for it on every layer**: verify the first K layers of a run, then trust
the device plan, with `KRK_DEV_PLAN_VERIFY=all` to keep the exhaustive arm. Expected:
125 ms → ~70 ms, i.e. 8.0 → ~14 tok/s, with the fallback intact for any stack whose shape
or routing the device path cannot express.

Not done here: it changes what "verified" means for the plan, so it needs its own numerics
gate rather than being tacked onto a measurement turn.

---

## 22.10 The slab test: allocation layout is NOT the cause (negative result, verified)

Every resident per-layer weight now comes from one contiguous device slab instead of ~700
`hipMalloc`s (`KRK_WEIGHT_SLAB=1`, off by default). Building it exposed a confound worth
recording first: the slab reserves its full size up front, so it ate ~1.9 GiB of promotion
headroom — 10 promotions became 1, the streamed set doubled and the arm looked 20% slower
for a reason that had nothing to do with layout.

With `KRK_WEIGHT_PROMOTE=0` in **both** arms the residency, the streamed set (layers
45..63), the policy and the output are identical, so the only difference is layout:

| resident weights laid out as | step | decode |
|---|---:|---:|
| ~700 separate allocations | **225.4 ms** | 4.4 tok/s |
| one 12.4 GiB slab | 230.4 ms | 4.3 tok/s |

Output md5 identical (`2e92586887d67b2959faa8d3577a4ef2`) in every arm, so the slab's
release path is correct — the interior-pointer skip, the one release at teardown, the
per-tensor fallback.

**Layout is not the cause of the 84 GB/s.** One contiguous 12.4 GiB slab, read by the same
kernels in the same order, changes the step by +2%. Fragmentation, page locality and
scattered-allocation TLB pressure are eliminated.

That leaves the falsifier from §22.9 as the live hypothesis: the in-engine rate is a
*launch* rate. 1104 dependency-ordered launches per token against 200 back-to-back
launches of one kernel in the probe — if a single isolated launch with an L2 flush also
lands at 70-90 GB/s, the cause is granularity, and the fix is fewer, fatter launches
(more rows per GEMM, fused epilogues), not kernels.

Verified after the default flip: **4149/4149**, dense 27B 5.2 tok/s / 191.6 ms, md5
unchanged.

---

## 22.11 The launch-granularity hypothesis is FALSIFIED by the probe that already exists

`tools/probe_gemv_dtype.hip` times **one launch per timed window, with a 256 MiB L2 flush
outside the window** (lines 193-216) — which is exactly the falsifier §22.9 asked for, and
it has been in the tree since §22:

| 27B o_proj, 5120x6144, one flushed launch | GB/s |
|---|---:|
| Q4_K | **545.5** |
| Q5_K | 448.0 |
| Q6_K | 352.7 |
| Q8_0 | 578.9 |
| F16 | 639.9 |

A single cold launch of the same kernel, at the same shape and dtype, with the cache in the
same state, sustains 545 GB/s. The engine's span for that same op reads 158-168 GB/s. So
the 0.2x is **not** launch granularity, not cold-start ramp, not the drain per launch, and
not a deep-queue artefact of the probe.

Causes eliminated by measurement so far, each with its own arm:

| candidate | test | result |
|---|---|---|
| PCIe / streamed bytes | reserve 1024 → 0 MiB, 9 → 4 streamed layers | 6 ms of 44 |
| residency | fully resident arm | *slower* (3.1 vs 5.2) |
| copy scheduling | double-buffered ring, 159/160 in flight | 0 |
| allocation layout / TLB | one 12.4 GiB slab vs ~700 allocs | +2% |
| launch granularity / ramp | one flushed launch in the probe | 545 GB/s |

What remains is the one thing that differs between those two launches and has not been
compared: **the launch itself** — grid, block, and which kernel symbol the engine picks for
`gemm_acc`/`gemm_group(fused)` versus the probe's direct `gemv_kernel<256>` with
`grid = (n_out + 7) / 8`. The next step is to print the engine's grid and block for that
exact shape (`KRK_OP_SHAPES=1` already tags the op) and diff it against the probe's — a
number, not a theory, and the first thing in this investigation that has not yet been
tested.

---

## 22.12 Geometry is identical, and the engine's own repeat trap says STARVED, not slow

`HipBackend::gemm_accumulate` launches `gemv_kernel<256, true>` with
`grid = (n_out + 7) / 8`, block 256 — **byte for byte the probe's geometry**. The
under-filled-launch theory dies on inspection, before a run.

The backend already carries the trap that separates the remaining two answers
(`KRK_ACC_REPEAT=n`: re-launch the same kernel on the same data, timed per repeat):

| op | first span (profile) | repeat 1 | repeat 2 |
|---|---:|---:|---:|
| 5120x17408 Q6_K (69.7 MiB, > L2) | 158-168 GB/s | **484.5** | 461.3 |
| 5120x17408 Q4_K (47.8 MiB, < L2) | 158-168 GB/s | 295.6 | 744.9 |

The same launch, immediately repeated, runs at 295-484 GB/s — 2-3x the rate the span
reports. So the projection is **starved, not slow**: the span absorbs a wait that precedes
the kernel, and the kernel itself is at or near the probe's rate. §22.3's reading of the op
table was right.

Eliminated now, each with an arm: PCIe bytes, residency, copy scheduling, allocation
layout, launch granularity, launch geometry. What is left is the wait itself — why a
launch that follows a completed dependency waits ~50% of its span on a stream whose queue
is one deep.

Chunk size (the reader's hypothesis) is the same mechanism seen from the other end: more
rows per chunk gives each launch more work to hide the wait behind, which is why prefill at
chunk 256/512 is a different regime from decode at row 1.

---

## 22.13 Prefill chunk sweep: chunk is flat, and the reason is ONE kernel at 3.8 GB/s

191-token prompt, 27B Q4_K_M, greedy, `--max-tokens 1`, chunk swept 64 → 512:

| chunk | prefill | device spans | idle between ops |
|---:|---:|---:|---:|
| 64 | 44.7 tok/s (4271 ms) | 78.2% | 20.3% |
| 128 | 45.0 tok/s (4240 ms) | 78.1% | 20.9% |
| **256** | **45.6 tok/s (4191 ms)** | 78.8% | 20.0% |
| 512 | 43.4 tok/s (4396 ms) | 78.3% | 20.9% |

Flat within 5% across an 8x range of chunk, and the spans/idle split does not move either.
Chunk size is not the lever — and the earlier "prefill 1.7 tok/s" figures were a one-token
prompt measuring startup, not prefill throughput. Prefill at a real prompt is 45 tok/s.

The aggregate op table from the chunk-256 run names the actual cost, and it is one row:

| op | calls | NET ms | MB | GB/s |
|---|---:|---:|---:|---:|
| **`gemm(grid)`** | 496 | **4024.3** | 15376.5 | **3.8** |
| `upload` (no bytes moved) | 732 | 1663.8 | — | — |
| `warm_read(pool)` | 29 | 446.8 | — | — |
| `upload_pinned` | 19 | 79.2 | 4438.1 | 56.1 |
| `attention` | 16 | 22.0 | — | — |

**`gemm(grid)`: 15.4 GB of weights at 3.8 GB/s — 96% of prefill, and 100x below the same
weights' own GEMV path (295-484 GB/s) and 140x below the card.** It is the batched
(row-major, grid-tiled) GEMM that prefill selects instead of `gemm_group(fused)`/gemv, and
it is not getting cache reuse at any chunk size — the 64 MB Infinity Cache is irrelevant to
a kernel that re-reads a tile per block. Fixing this one kernel to even 200 GB/s turns
4.2 s of prefill into ~0.08 s, i.e. 45 → ~2 000 tok/s of prompt processing.

`upload` at 1663.8 ms with 0.0 MB is a second defect in the same pass: 732 calls, no bytes,
1.25 s of idle — a bookkeeping call that costs more than the attention it surrounds.

---

## 22.14 `gemm(grid)`: it is arithmetic-bound at 4-6% of peak, not cache- or bandwidth-bound

`KRK_GRID_G` is the cache knob: one block per (output row, group of rows), so a smaller
group re-reads each weight row more often. Swept on the same 191-token prompt:

| `KRK_GRID_G` | prefill | note |
|---:|---:|---|
| 8 | 38.4 tok/s | most re-reads |
| 32 (default) | **45.6** | |
| 128 | 44.9 | |
| 256 | 43.5 | fewest re-reads |

Flat to within noise, output md5 identical across all four (`d9c95a66…`). **Weight re-reads
are not the limiter**, so the Infinity Cache is not what prefill is losing to.

What it is losing to is arithmetic. A 27B dense model at 191 tokens is 2 × 27e9 × 191 =
**10.3 TFLOP** of matmul. The kernel does it in 4.25 s = **2.4 TFLOP/s**, against a card
that peaks around 40-70 TFLOP/s of fp16 — **4-6% of peak**. A `gemv_grid_kernel` block
computes `grid_g` dot products per weight row with no matmul tiling and no WMMA, so it
scales like GEMV (bandwidth per row) rather than like GEMM (reuse per tile). The
`GemmRung::kWmma` rung exists but is not reachable for this dtype/shape, which is why
prefill falls to the exact grid path.

Therefore the prefill fix is a **tiled Q4_K GEMM with WMMA/dp4a** — reuse the weight tile
across the row tile in registers — not cache blocking, not chunk size, not `GRID_G`.
Ceiling if it reached 40% of peak: 10.3 TFLOP / ~20 TFLOP/s ≈ 0.5 s, i.e. **~370 tok/s**
of prompt processing against today's 45.

---

## 22.15 Why the obvious tiled Q4_K GEMM is NOT the fix (the kernel already records the measurement)

`gemv_grid_kernel`'s own comment (measured 2026-10-22, gfx1201, rows=1024, L2-resident) is
the answer to the tiling plan in §22.14:

> a dequant-once LDS tile shared across the block's warps does NOT pay — q4_k 1122 us
> redundant vs 1177 us tiled … Each warp therefore runs the real dot_chunks() on the raw
> row, exactly like gemv_mw: redundant decode, L2-resident weights, zero numerics risk.

Two facts follow, and they move the target:

1. The kernel **already uses `dot_chunks()`** — the same packed f16-pair dot path the fast
   GEMV kernel uses. There is no "forgot the fast path" bug to fix here.
2. The reuse ideas that are cheap (share the dequant through LDS, widen the row group) have
   been measured and do not pay. That is why `KRK_GRID_G` is flat (38-46 tok/s over 8-256).

So prefill's 2.4 TFLOP/s is a property of the kernel's *shape*: one warp-lane driving one
weight row's dots, scalar-f32 accumulate, no matrix tiles in registers. Widening it is not
a tweak to this loop — it is a different kernel: a WMMA/dp4a Q4_K GEMM with a real
(NB x KB) register tile, which is the `GemmRung::kWmma` rung that Q4_K cannot currently
reach. Everything else in the prefill path has now been tested and cleared: chunk (flat),
`GRID_G` (flat), dequant tiling (measured negative in-tree), weight re-reads (not the
limiter), bandwidth (3.8 GB/s of a 700 GB/s bus, so not bandwidth), attention and upload
(each under 7% of prefill).

The ceiling stands at ~370 tok/s if a WMMA tile reaches 40% of peak, against 45.6 today.

---

## 22.16 Prefill found: the fast GEMM was disabled by the exactness policy. 45.5 → 196 tok/s, same output

`exact_gemm_enabled()` defaulted ON, which forces every prefill through the exact
`gemv_grid_kernel` (rows > 8) and bypasses the rung lottery — including `gemm(wmma)`, which
was already implemented and works. `KRK_GEMM_EXACT=0`:

| prefill GEMM | prefill | ms | nominal weight rate |
|---|---:|---:|---:|
| `gemm(grid)` (default) | 45.5 tok/s | 4198 | 3.8 GB/s |
| `gemm(wmma)` (lottery) | **196.0 tok/s** (chunk 256) | 975 | 17.2 GB/s |

and the output md5 is **identical** — `d9c95a66b0ca96eb8c931b01f466f029`, at chunk 64, 128,
256 and 512, and identical to the exact arm's md5. Chunk sweep under the fast path:
139.1 / 168.5 / **196.0** / 182.2 tok/s.

What the exact policy was protecting is real but narrower than its cost: with the lottery,
`test_prefill_chunk_bitexact` reports that at 16 and 32 rows an **intermediate activation
trace** differs from the 1-row association (`bf32bbb3…` → `84532ea0…`). The **logits do not
differ** — `e8d8efa95f9d62c9` at every chunk size including the tiled and lottery controls.
So the invariant is trace-level reproducibility, not the answer.

Shipped: `--fast-prefill` (or `KRK_GEMM_EXACT=0`) opts into it; the default stays exact
because the suite's contract says so and I will not edit a test to make a flag pass.
Verified: **4149/4149** with the flag wired, `--fast-prefill` measured at 182.0 tok/s in the
same build, md5 unchanged.

### Where prefill stands

Compute, not bandwidth, is now the binding constraint: 191 tokens × 27e9 × 2 = 10.3 TFLOP in
0.975 s = **10.6 TFLOP/s**, roughly a quarter of a ~40-55 TFLOP/s fp16 peak. The bandwidth
ceiling for this prompt is ~31 ms (15.4 GB at 500 GB/s = 6 100 tok/s) and the compute ceiling
is ~260 ms (~740 tok/s), so prefill still has ~4x to take from the WMMA path itself — tile
shape, occupancy, and the dequant-to-f16 staging inside the kernel. That is kernel tuning,
not policy.

---

## 22.17 The WMMA MB lever, swept and shipped: +15% at 191 rows, +25% at 701

`wmma_cfg::MB` was 1: one block walks one M tile, so a 191-row prefill (BM=32 → 6 M tiles)
stages the weight tile **six times**. The kernel already supports MB > 1 — "the weight
values staged are the same f16 values, and each accumulator still walks K in the same
order" — but it was a `constexpr`, so it could not be swept. It is now
`KRK_WMMA_MB` (1-8, default **4**), instantiated per value.

Interleaved A/B, same build, `--fast-prefill`, output md5 identical at every value:

| rows | MB=1 | MB=4 | delta |
|---|---:|---:|---:|
| 191 | 181.3 / 162.7 / 196.5 — mean **180.2** | 194.4 / 213.3 / 213.2 — mean **207.0** | **+14.9%** |
| 701 | 196.8 / 196.8 — mean **196.8** | 247.1 / 246.2 — mean **246.7** | **+25.4%** |

MB=8 was measured and rejected: 185.1 (191 rows) and 230.9 (701 rows), below MB=4 — the
accumulator array spills, which is the same occupancy-vs-tiles trade the BN=128 note
records.

With this, prefill on the 27B is **45.5 → ~207 tok/s at a 191-token prompt** (196 exact →
207 fast at MB=4) and **~247 at 701 rows**, against a compute ceiling of ~740 tok/s at this
prompt's 10.3 TFLOP. Still ~3x from the ceiling, and the remaining lever named by the
kernel's own notes is the staging chain (§2c: quantized staging with the unpack in
registers), not a wider tile — BN=128 and BK=256 were both measured and rejected there.

Verified: `kraken` and `kraken-tests` rebuilt clean, **4149/4149**, output md5
`d9c95a66b0ca96eb8c931b01f466f029` in every arm.

---

## 22.18 Before changing staging: the split says its ceiling is +29%, and the MMA is the bigger half

The kernel carries two diagnostic arms — `KRK_GEMM_STAGE_ONLY=1` (stage, skip the MMA) and
`KRK_GEMM_MMA_ONLY=1` (MMA, skip the staging). At the shipped MB=4, 191-row prefill:

| arm | prefill | time |
|---|---:|---:|
| stage only | 266.1 tok/s | 0.718 s |
| MMA only | 267.3 tok/s | 0.715 s |
| full | 206.1 tok/s | 0.927 s |

The two halves cost the same, and 0.718 + 0.715 − 0.927 = **0.506 s of the 0.718 s is
already overlapped (71%)**. Three consequences:

1. **No staging change can exceed 267 tok/s** (+29%) even if staging became free. That is
   the hard ceiling on "stage quantized, unpack in registers", and it is a *ceiling*, not a
   prediction: with 71% of the staging already hidden, the realistic gain is a fraction of
   it, because the staged bytes that remain exposed are not the chain's cost — the chain is
   latency-bound (kernel notes §2c), and it already prefetches raw Q4_K via `q4k_cur`/
   `q4k_nxt` rather than staging f16 from global.
2. The MMA runs at 14.4 TFLOP/s (10.3 TFLOP in 0.715 s) — **~30% of fp16 peak**, and it is
   the larger, unaddressed half.
3. The two are balanced *by construction*: staging ~= MMA ~= 0.72 s. Any work on one alone
   is capped by the other, so the next change has to be picked for its effect on the pair.

Recorded as a decision, not a result: the quantized-staging rewrite is **not** worth its
risk at a +29% ceiling with 71% already overlapped. The headroom that matters is in the MMA
tile/occupancy (14.4 of ~48 TFLOP/s), and that is where the next measurement belongs.

---

## 22.19 The MMA leg, measured: the MB trade is barriers against staging, and the MMA caps prefill at ~299

Splitting both halves by MB (191 rows, `--fast-prefill`):

| MB | MMA only | stage only | full |
|---:|---:|---:|---:|
| 1 | **299.0 tok/s** | 300.1 | 180.2 |
| 2 | 255.6 | 257.5 | ~195 |
| 4 | 265.8 | 278.2 | **207.0** |

Three things fall out:

1. **Each half is fastest at MB=1 and the pair is fastest at MB=4.** More M tiles per block
   buys staging amortisation and costs MMA throughput, because the `mi` loop is *inside* the
   k loop: every extra accumulator re-stages its activation tile behind its own barrier, so
   the MMAs never run back to back. MB=4 wins overall only because the activation/weight
   staging saving is larger than that penalty.
2. **Overlap improves with MB**, 34% at MB=1 (0.639 s + 0.636 s against 1.055 s) to 70% at
   MB=4. So the barrier cost is paid *inside* a block where the other warps can still work,
   which is why the aggregate still improves.
3. **Even with staging removed, prefill cannot exceed ~299 tok/s** — 10.3 TFLOP in 0.639 s =
   16.1 TFLOP/s, a third of fp16 peak. That is the real MMA ceiling, and it is a tile/wave
   question, not a staging one.

The bounded next change, with its numbers: keep MB=4 but stage **all MB activation tiles
before** the MMA passes (one barrier per k-tile instead of one per M tile). Cost: LDS grows
from 26 kB to ~52 kB per block (2 blocks/CU instead of 4); benefit: MB independent
accumulators issue back to back, which is what the MB=1 MMA rate (299) says is available.
Upper bound on that change is 299 tok/s (+45%); the occupancy loss will eat part of it.

Not attempted here: it changes the LDS budget and the whole inner loop, and the measured
trade above is the evidence a rewrite has to beat, not a guess.

---

## 22.20 One barrier per K tile: prefill 207 → 228 (191 rows) and 247 → 290 (701 rows)

The M loop wrapped the staging, so every extra accumulator paid its own activation stage
and its own barrier and the MMAs never issued back to back (§22.19 measured that: MMA-only
299 tok/s at MB=1 against 265 at MB=4, with the pair preferring MB=4).

Now the K tile is staged by ONE pass — all MB activation tiles into `xs[MB][BM][BK+pad]`,
plus the weight tile, then a single `__syncthreads()`, then MB accumulators issued
consecutively with no barrier between them. LDS goes from 26 kB to 52 kB at MB=4 (two blocks
per CU), which is the price; MB is capped at 4 for that reason.

Two runs each, `--fast-prefill`, output md5 `d9c95a66b0ca96eb8c931b01f466f029` in every arm:

| rows | before | after | delta |
|---|---:|---:|---:|
| 191 | 206.1-213.3 (mean ~207) | **227.8 / 228.2** | **+10%** |
| 701 | 246.2 / 247.1 | **290.3 / 289.7** | **+17%** |

MB=2 also moved (195 → 226.7): with one barrier, the amortisation no longer costs the MMA,
which is exactly what the diagnostic predicted. MB=1 is slightly worse than before
(180 → 174) — it gains nothing from the restructure and pays the wider LDS footprint.

Cumulative on the 27B: **prefill 45.5 → 228 tok/s at 191 rows (5.0x), 45 → 290 at 701 rows
(6.4x)**, against the MMA-bound ceiling of ~299 tok/s at this prompt. Decode 5.2, MoE 8.0,
**4149/4149**, exact arm unchanged at 44.9 tok/s.

---

## 22.21 Two N fragments per wavefront (default): prefill 228 → 262, 290 → 314

`wmma_cfg` now carries `NF` (N fragments per wavefront). The new `wmma_cfg_n2` map is
4 waves along M (BM=64) x 2 along N, each wave owning 32 columns as two 16x16
accumulators: the A fragment is loaded once per k step and feeds **two independent MMAs**,
so a wavefront always has a second MMA in flight to cover the first one's latency. Same
8 wavefronts, same 52 kB LDS, same 4 accumulators per wave.

| rows | NF=1 (old map) | NF=2 | delta |
|---|---:|---:|---:|
| 191 | 256.5 / 257.7 | **263.9 / 264.6** | +3% |
| 701 | 299.2 / 299.7 | **313.9 / 314.0** | +5% |

The epilogue was rewritten at the same time (skip a row once, loop the fragments, branch
only on the column), and that alone moved NF=1 from 228 to 257 — the store loop had been
carrying two comparisons and a `constexpr` branch per element.

Default is now NF=2 (`KRK_WMMA_NF=1` restores the 32x64 map). Verified with the default in
place: **262.4 / 261.2** at 191 rows and **315.0 / 312.4** at 701 rows, **4149/4149**, output
md5 `d9c95a66b0ca96eb8c931b01f466f029` in every arm.

Cumulative prefill on the 27B Q4_K_M, one session:

| | start | now | |
|---|---:|---:|---|
| 191-token prompt | 45.5 tok/s | **262** | 5.8x |
| 701-token prompt | ~45 tok/s | **315** | 7.0x |

Note the ~299 tok/s "MMA ceiling" from §22.19 has been passed: it was measured with the
per-M-tile barriers and the old epilogue, so it was a property of that structure, not of the
MMA unit.

---

## 22.22 BN=128 re-tested under the one-pass staging: rejected again, and removed

Question: now that staging is one pass per K tile (§22.20) and the store loop is one branch
per row (§22.21), does a 128-wide N tile pay? A wave map of 2 along M x 4 along N keeps the
8 wavefronts and NF=2, and halves the activation re-reads per output.

It does not pay, and the reason is a constraint rather than a preference. The pipelined Q4_K
staging gives **every thread exactly one 32-value chunk per round** —
`static_assert(BN * (BK / 32) == THREADS)` — so a 128-wide tile pins BK to 64: half the K
depth and twice the staging rounds, which §22.17 already measured as the expensive direction.
The tile therefore ran with 23 kB of LDS (32 + 128 rows x 72 halves), i.e. four blocks/CU,
which is the occupancy the old 43.5 kB BN=128 arm could never get — and it still lost.

Interleaved, same build, one arm per process, output byte-identical
(`8fbdbb97b237c7b6f5636e3d5b888336` at 191 rows, `2fa5f72be4acf53147ecbb090da989a7` at 701):

| rows | BN=64 (shipped) | BN=128 (BK=64) | delta |
|---|---:|---:|---:|
| 191 | 264.8 / 263.4 / 264.4 — mean **264.2** | 233.7 / 234.0 / 233.4 — mean **233.7** | **-11.5%** |
| 701 | 308.7 / 307.9 — mean **308.3** | 231.8 / 233.6 — mean **232.7** | **-24.5%** |

The activation-reuse saving is real but small next to the round count: at 701 rows the wide
tile loses a quarter, which is the shape where its byte saving should be largest. A fair
BN=128 at BK=128 needs **two chunk pipelines per thread** (two `q4k_cur`/`q4k_nxt` register
sets, two staged rows per thread), i.e. a rewrite of the pipelined prologue and inner staging
loop. The arm that needs no rewrite already loses, so that rewrite was not attempted.

**Action: the BN=128 map is deleted, not kept behind `KRK_WMMA_BN`.** Leaving a losing
instantiation compiled in costs compile time and a launch branch for an arm the numbers say
never to take. The measured rejection is recorded in `wmma_cfg`'s note and at the launch site.

Verification after removal: rebuilt clean, **4149/4149**, prefill **264.6** at 191 rows and
**309.1 / 308.4** at 701 rows (no regression), output md5 unchanged from the arm that carried
the wide path. Traces: `perf-traces/bn64-191-*.{out,err}`, `bn64-701-*`, `bnA64`/`bnA128`,
`post-191-*`, `post-701-*`, suite in `perf-traces/post-suite.err`.

Where the remaining prefill headroom is, unchanged by this test: the staging chain. §2c showed
the cost is per round and latency-bound, and the ceiling is a wave/tile question, not a
column-block one. The next lever on this path is quantized staging with the unpack in
registers — the same one §22.17 named — not a wider tile.

---

## 22.23 README tables re-measured (2026-10-25), and three new model files triaged

Every row of the README's headline table (3 runs) and its 26-row sweep was re-run
on the current build; the MoE sweep rows got a second run each because they are
bimodal. Full traces: `perf-traces/bench/`, scripts `perf-traces/readme_sweep.sh`,
`bench2.sh`, `headline.sh`, logs `bench-sweep.log`, `bench2.log`, `headline.log`.

What moved, and what did not:

* **Decode is flat** across the refresh: 20 of 26 sweep rows within 5%. The one
  regressing row is the Ornith-35B-A3B MoE at 30.6 -> 27.6-28.8 (-9%); the 30B
  MoE is -8% warm. Every other delta is inside run noise or up: the 32B dense
  streamed model is the outlier at 1.0 -> 3.6-3.8 tok/s (+270%).
* **Prefill rose on all 26 rows** (+34% to +77%) with **no change to the
  `--prefill exact` path** in the working tree (this session's kernel work is
  entirely the `--prefill fast` WMMA rung). Prefill is a short compute burst and
  tracks clocks; decode is long and memory-bound. The conclusion recorded in the
  README is therefore "the 2026-10-24 prefill column is a cold-clock
  measurement", not "prefill got 60% faster".
* **`--prefill fast` on the 8B fixed prompt is now 923-925 tok/s**, not 633.5,
  with decode identical (91.0) between the two arms — re-measured interleaved
  twice to rule out the arm affecting decode.
* MoE spread is the headline hazard: the 30B-A3B reads 6.7/17.7 cold and
  22.9/37.2 warm, qwable 9.9/17.2 cold and 105.3/28.6 warm. A single `--bench`
  run misreports these by up to 4x, so the sweep now prints `cold -> warm`.

### New files on the box, triaged

| file | size | verdict | evidence |
|---|---|---|---|
| `flint-q4_1.gguf` | 23.2 GB | **runs, output degenerate** | arch `flint` read against llama defaults; 51.6 prefill / 49.5 decode; emits `valid Germany Germany Germany...` |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB_QAT_IQ2_S.gguf` | 20.5 GB | **refused at load** | `layer 1 router is [4096 x 240], metadata wants [4096 x 496]` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 103 GB | **refused, correctly** | 2 tensors total: `blk.1.engram_embd.weight [256 x 384006168]`, `blk.14.engram_embd.weight [256 x 384016682]` — an Engram table pack, no graph |

**flint** is the interesting one: the loader ignores three keys that this arch
needs, and the output shows it. `flint.attention.sliding_window = 513` with
`flint.attention.sliding_window_pattern = array[49]` (per-layer), plus
`expert_shared_count = 2` / `expert_shared_feed_forward_length = 1024` and
`leading_dense_block_count = 2`. Speed without sliding window or shared experts
is worthless, which is why a 49-layer 256-expert file decoding at 49.5 tok/s
still prints word salad: throughput is not a support signal.

**Bolt 74B** is a Nemotron Hybrid LatentMoE (Mamba-2 + attention + MoE). Its 40
MoE layers carry **per-layer** expert counts — 240 x12, 304 x6, 368 x9, 432 x10,
496 x3 — while the metadata declares one global `expert_count = 496`; it also
carries a per-layer `expert_used_count` array (`[0, 9, 0, 9, ...]`, i.e. the
top-k varies per layer), `expert_shared_count = 1`,
`expert_shared_feed_forward_length = 5376` and `expert_weights_scale = 5.0`.
Kraken's loader requires every router's column count to equal the global
expert_count, so it refuses at layer 1. Fixing it means making `n_expert` and
`n_expert_used` per-layer through the loader, the plan buffers, the expert cache
and the device dispatch — a real feature, not a patch.

**DeepSeek-V4.1-Engram** cannot be "added for inference": it is not a model. Two
lookup tables of ~51 GB each, no embedding, no attention. It is the Engram memory
side-car for a DeepSeek-V4.1 base that is not on this box.

## 22.24 Flint: the reference patch says exactly what is missing here

Source: `experiments/reproduce/llama-flint-runtime.patch` in the alphaBell
Model-Observatory space (llama.cpp `0a1ef309` + patch, CPU-only reference). It
lands `src/models/flint.cpp`, which is a `Qwen2Moe` subclass:

* per-layer `is_swa` array from `attention.sliding_window_pattern`, and a
  per-layer **rope** pattern from `attention.rope_pattern` (`has_rope(il)`);
  `n_swa = 513`, `swa_type = STANDARD`
* `leading_dense_block_count` blocks of plain SwiGLU, then MoE blocks with
  `n_expert` / `n_expert_used`, an optional `exp_probs_b` bias, and a **shared
  expert** (`ffn_{gate,up,down}_shexp`) whose output is *added* to the MoE output
* per-head RMS `attn_q_norm` / `attn_k_norm`, and **attention sinks**
  (`attn_sinks`, the `softmax_offset` tensor) passed to `build_attn`
* `expert_weights_norm` + `expert_weights_scale` (1.0 here)

The actual file (`flint-q4_1.gguf`, arch `flint`, 49 blocks) carries:

| key | value |
|---|---|
| `flint.attention.sliding_window` | 513 |
| `flint.attention.sliding_window_pattern` | `0 0 1 1 1 0 1 1 1 0 … 0` (49) |
| `flint.attention.rope_pattern` | **identical to the SWA pattern** |
| `flint.expert_count` / `expert_used_count` | 256 / 6 |
| `flint.expert_shared_count` / `expert_shared_feed_forward_length` | 2 / 1024 |
| `flint.leading_dense_block_count` | 2 |

Tensors per block: `attn_q` (2048x4096 — a 2x-wide query, Qwen3.5-style gate),
`attn_k`/`attn_v` 2048x512, `attn_q_norm`/`attn_k_norm` [128], `attn_sinks`
[32], `attn_output`, `attn_norm`, `ffn_norm`; `ffn_{gate,up,down}` on blocks 0-1;
`ffn_gate_inp` + `ffn_{gate,up,down}_exps` + `ffn_{gate,up,down}_shexp` on 47.

**What Kraken already has:** right-weight sinks (`attention.hpp:976`,
`model.cpp:1589`), shared experts incl. `ffn_*_shexp` (`engine.cpp:8588`),
`leading_dense_block_count` (`model.cpp:311`), `exp_probs_b` under both
spellings (`model.cpp:2273`), q/k norm, and a sliding window per layer
(`engine.cpp:2871`, `d.window = is_swa(l) ? swa_window : 0`).

**What it lacks — three things, and they are why the file prints `valid Germany
Germany Germany`:**

1. **Per-layer pattern arrays.** `is_swa()` is `(l + phase) % period != 0`,
   which cannot express this file: full-attention layers are
   `{0, 1, 5, 9, …, 45}` — a period-4 pattern *plus* an exception at layer 0.
   The rope pattern is a second per-layer array with no representation in
   `ModelConfig` at all (full-attention layers here have **no rope**).
2. **No arch registration.** `flint.*` keys are read against llama defaults, so
   `sliding_window`, the two patterns, `leading_dense_block_count`, the shared
   expert widths and the gating function are all ignored today.
3. **Silent failure.** The loader accepts the file and runs it at 49.5 tok/s.
   A file whose arch needs keys we do not read should refuse by name, the way
   gemma4 and the MLA layouts do — speed on garbage output is the worst
   possible signal.

The fix order is: register `flint` (config + key prefix) → add the two per-layer
arrays to `ModelConfig` and read them → point `is_swa()` and the rope table at
the arrays → wire sinks + shexp + leading dense (all already implemented) →
gate on coherent output and a suite test pinning the greedy sequence.

The reported per-layer `expert_used_count` of Bolt is a separate problem (see
§22.23): Flint has one global top-k, Bolt varies it per layer.

---

## 22.25 Per-layer expert count and top-k: shipped, and what Bolt still needs

Bolt 74B-A9B was refused by one loader check: `layer 1 router is [4096 x 240],
metadata wants [4096 x 496]`. The check was wrong in principle, not just for this
file. `expert_count` is a **cap**; the routed bank's own `ne[2]` is each layer's
truth, and Bolt really does carry 240/304/368/432/496 experts across its 40 MoE
blocks. `expert_used_count` is likewise a **per-block array** there (0 on the
Mamba-2 blocks, 9 on the MoE ones), which the scalar read turned into "top-1".

What changed:

* `ModelConfig::layer_n_expert_used` holds the array when the file has one; the
  model-level `n_expert` / `n_expert_used` / `n_ff_exp` become the maxima, which
  is what the workspaces are sized for.
* `Model::n_expert_at / n_expert_used_at / n_ff_exp_at` are the per-layer
  geometry, and `moe_geometry_uniform()` says whether a stack is homogeneous.
* The loader now reads the bank's geometry *before* checking the router, requires
  **router == this layer's bank** (not == the cap), keeps the cap as an upper
  bound, and reads `.expert_used_count`, `.expert_feed_forward_length` and
  `.feed_forward_length` as arrays when they are arrays.
* `Engine::moe_ffn` takes `ne`, `k` and `ff_exp` from the layer. The device plan,
  the one-launch dispatch and the expert scan are gated on
  `moe_geometry_uniform()`: they publish ONE `[n_layer x n_expert]` table, and a
  432-expert layer reading a 240-expert table is in range, stable and wrong.
* `test_moe_per_layer_experts` (14 checks) pins it: a two-layer fixture with 3
  and 5 experts under a cap of 5 and a top-k array of `{2, 3}` loads, reports
  non-uniform, routes each layer at its own width, and generates; a uniform
  control must still report uniform.

Verified: **4163/4163** checks (4149 + 14). Qwen3-30B-A3B still decodes
coherently after the change ("The sea is a vast, mysterious, and ever-changing
body of water...").

### Bolt now loads — and this is what is still wrong with it

It loads, runs and exits `2`: `84508 silu, 8201874 experts-sum and 456 residual
store(s) left f16 and were clipped`, output `<unk><unk>...`. The cause is in the
file, not in the routing:

```
blk.1.ffn_latent_down.weight   [4096, 1024]
blk.1.ffn_latent_up.weight     [1024, 4096]
blk.1.ffn_up_exps.weight       [1024, 2048, 240]     <- input width 1024
blk.1.ffn_down_exps.weight     [2048, 1024, 240]     <- output width 1024
blk.1.ffn_up_shexp.weight      [4096, 5376]          <- full width, not latent
```

Bolt is a **LatentMoE**: the routed experts run in a 1024-wide latent space
(`latent_down` -> experts -> `latent_up`), while the shared expert runs on the
full 4096-wide hidden state. Kraken feeds `n_embd_` (4096) into a 1024-wide
expert bank, which is exactly how you get every element of the routed sum
clipped. The warning also confirms the layer mix is honest: even blocks are
Mamba-2 (`ssm_*`), odd blocks are latent-MoE with **no attention tensors at
all** -- this file has no attention path to miss.

So the per-layer expert work was necessary but not sufficient. LatentMoE is a
separate, well-specified change: upload `ffn_latent_{down,up}`, project the
block's input into the latent before the experts, run the routed sum there,
project back, and add the shared expert computed on the full hidden state. The
expert visit loop's input width has to become the layer's latent width instead
of `n_embd_`.

## 22.26 LatentMoE shipped: Bolt 74B-A9B decodes coherently (19 tok/s), and the second bug behind it

**Verdict. ROOT CAUSE (two, in series):** the routed bank was fed the hidden
state instead of the latent (`22.25`), and — after fixing that — every
Mamba-2 block's NVFP4 global scale was being ignored, which inflated the SSM
pre-activations by 1/gs = 7,480x and saturated the residual stream in block 0.
Both are fixed; the file now loads, routes, and produces coherent text.

### What the dump said (KRK_DUMP + KRK_DUMP_FULL, before the NVFP4 fix)

```
enter  L00  peak=0.380859 rms=0.0145      <- the hidden state, sane
m2.z   L00  nf=37    peak=65440 rms=8742  <- the block's first projection
m2.dt  L00  peak=60704                    <- dt_raw, so softplus(dt) is ~60k
m2.ssd L00  nf=22095 peak=65504
rec.post L00 peak=65504 rms=4690          <- residual saturated
ffn.pre L01 peak=0                         <- the residual row was SCALED
                                              by 2^-60 and underflowed f16
```

The first output channel of block 0 is 65,440 from an input whose peak is 0.38.
The file explains the factor exactly:

```
blk.0.ssm_in.scale   1.3369606e-4     (1/gs = 7,480)
blk.0.ssm_out.scale  1.6712007e-4
```

Only `ssm_in`/`ssm_out` carry NVFP4 sidecar scales on this file (40 blocks, 80
tensors); `ffn_*_shexp.scale` existed on the Lightning NVFP4 file and was
already applied. The SSM path never read it. So `W` was used as `W` instead of
`gs*W`, every SSM pre-activation was ~7,500x too large, the residual saturated
at the first block, the storage exponent hit its clamp at 60, and rows stored at
`2^-60` underflowed f16 to zero — which is why every later layer's `ffn.pre`
was exactly 0 and the model emitted `<unk>` forever.

### Shipped

* [model.hpp](include/krk/model.hpp) — `LayerWeights::latent_down/latent_up`
  (+ their NVFP4 scales) with `latent_moe()`; `ssm_in_scale` / `ssm_out_scale`.
* [model.cpp](src/model.cpp) — uploads `ffn_latent_{down,up}` (both or neither;
  one of the pair is refused), reads the two SSM scales with `one_scale`, adds
  both to `qt_fields` so unload and streamed-promotion binding stay complete.
  Geometry: `latent_down` must be `[n_embd -> bank]` and `latent_up`
  `[bank -> n_embd]`; a bank that is neither `n_embd` nor joined by a pair is
  refused instead of read off-stride.
* [engine.cpp](src/engine.cpp) — `moe_ffn` derives `ew` (the expert width) and
  swaps `xin`/`acc` for the latent buffers. Every expert-side op (gather, both
  activations' scale, gate/up/down gemms, biases, probe scan, scatter) now takes
  `ew`; the block projects once before the loop and lifts once after it, adding
  into `ws_ffn_` so the shared expert and the residual keep their full-width
  contract. `mamba2_forward` folds the NVFP4 scale into the *input* of the
  projections — necessarily the input for xBC, because the short conv adds a
  bias and silu is not linear over a scale. The latent legs fold on the output,
  which is exact for a plain linear map.
* Device-plan / one-launch dispatch / decode-batch are gated off for a latent
  layer: each publishes a table or a geometry that assumes `n_embd`-wide
  experts.
* [test_kraken.cpp](tests/test_kraken.cpp) — `test_moe_latent` (15 checks): a
  32-wide hidden state with a 16-wide ungated bank and a full-width shared
  expert loads, reports its own bank width, and generates; half a pair, a pair
  that does not join the bank, and a narrow bank with no pair are refused with
  the latent named in the message.

**Verified: 4178/4178** checks (4163 + 15). Qwen3-30B-A3B still coherent
(6.8 tok/s, that file's paging is unchanged).

### Bolt 74B-A9B (IQ2_S QAT, RX 9070 XT, 4.63 GiB VRAM budget)

| arm | prefill | decode | range (silu/experts-sum/residual) | output |
|---|---|---|---|---|
| before | 45.6 tok/s | 17.8 tok/s | 4,625,488 / 0 / 9,009, exponent 60 | `<unk>` |
| after | 31.5–35.4 tok/s | 19.1–19.3 tok/s | **0 / 0 / 0**, exponent 0 | ` Paris.` / ` Red, yellow, blue.` |

The routed sum no longer clips at all (it was 8,021,874 elements at `22.25`),
and no residual row needs a storage exponent — the single strongest signal that
the arithmetic is now inside f16's range end to end.

### Still open (not this change)

* `output.scale` on an NVFP4 checkpoint (Lightning 30B-A3B-NVFP4) is still not
  applied to the head — same class of defect, one tensor.
* Bolt's prefill is 31–35 tok/s against the 28–45 tok/s this box gets on
  smaller MoE files; the chunks are 5–9 tokens, so this is the small-row regime
  of `22.16`–`22.21`, not a new regression.

## 22.27 NVFP4 global scales, everywhere a weight is used — and the two failures that forced the design

**Verdict. ROOT CAUSE (of the class):** a `"<tensor>.scale"` sidecar is a property of
the *tensor*, but it was handled as a property of a *call site*. Three places had
grown their own bespoke field (`shexp_*_scale`, `ssm_in/out_scale`,
`latent_*_scale`) and every other projection silently used `W` instead of `gs*W`.
The head — the one weight every token passes through — was among the misses.

### Evidence, before the change

Scanning every checkpoint on this box for `*.scale` with `ne == [1]` (the
global-scale signature; block scales are per-16-values and much longer):

| file | sidecars |
|---|---|
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | `ffn_up_shexp` x23, `ffn_down_shexp` x23, **`output` x1** |
| NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB_QAT_IQ2_S | `ssm_in` x40, `ssm_out` x40 |
| mtp-Nemotron-3.5-Lightning-…-NVFP4 | **`output` x1** |

`output.scale` appears in two of the three, and was read by nothing.

### Shipped

* [expert_cache.hpp](include/krk/expert_cache.hpp) — `QuantTensor::gs` (1.0f by
  default) + `scaled()`. One rule: the scale lives on the descriptor the loader
  fills, exactly like `type` and `n_in`.
* [model.cpp](src/model.cpp) — resolved once, in `with_sidecar_scale`, for every
  weight `upload_weight` returns; also for the output head, which uploads
  through `upload_raw` and would otherwise have kept the gap, and for the MTP
  block's separate uploader. `Model::max_scaled_in()` records the widest scaled
  contraction so the engine can size its staging buffer (and allocate nothing on
  a file with no sidecar).
* [engine.cpp](src/engine.cpp) — `Engine::gemm_w()` and 48 converted call sites,
  so attention (q/k/v, the packed qkv+gate group, MLA's four layouts), the dense
  FFN, the router, the shared expert, LatentMoE's two legs, PLE/hyper-connections
  and the MTP block all take the same path. Three fused launches had to learn
  about it: q/k/v, gate/up and the GDN four-way group now fall back to
  per-matrix launches when a member is scaled, because one launch cannot apply a
  different scale per output and their epilogues compute a nonlinearity
  (`silu(gate)*up`) across the pair. `gemm_residual` takes `gs` and gives up the
  folded epilogue for it.
* The bespoke `shexp_*_scale` / `ssm_*_scale` / `latent_*_scale` fields are gone,
  so there is one rule instead of four.

### The design was forced by a measurement, twice

1. **Output scaling does not work for a real sidecar.** `gs*(W x)` is
   `(gs W) x`, so scaling the projection's *result* is the obvious move — and it
   is wrong here: the stored tensor is `1/gs` too large, so the RAW product is
   too. On Bolt (gs = 1.67e-4) that is 6,000x, the f16 store saturates and the
   clipped value cannot be scaled back. Measured after switching ssm_out to
   output scaling: block 6's `m2.done` peak 65,504 with the residual riding a
   storage exponent at its clamp, output back to `<unk>`.
2. **So the scale goes on the INPUT**, into a staged copy (`ws_gs_`, allocated
   only when the file has a scaled weight). `gs` is small enough that the scaled
   input is representable — the same trade the expert path has always made with
   `gs_up` on the gathered rows.
3. **A first-pass `scale_act`-in-place on the input buffer was rejected** for the
   shared ones: `ws_xn_` feeds the router after the projections, so the ssm path
   now scales into `ws_gs_` and reads that.

### Tests — [test_kraken.cpp](tests/test_kraken.cpp), `test_nvfp4_sidecar_scale` (25 checks)

The oracle is a **twin file**: the same model with the same numbers written as
f16. The NVFP4 arm encodes each element as an E2M1 code from the doubled grid
(`1..4` → 0.5..2.0) under a micro-scale of 1.0, with a **0.5 sidecar**, so its
true weight is exactly the twin's. Five arms — `head`, `attn_q`, `ffn_gate`,
`ffn_down`, `router` — each assert the same emitted token and logits within 2.0
(a dropped sidecar doubles a projection; the arm that failed before the head was
routed through the sidecar was off by 3.17 on the logits with the token still
matching, which is exactly the signature of a positive-scalar miss).

Two encoding mistakes of my own were caught by the fixture before it was trusted:
the nibble column mapping (`s*16 + j` / `+8`, not `2k`), and writing the twin
with the *random* weights instead of the grid.

### Verified

| check | result |
|---|---|
| suite | **4203/4203** (4178 + 25) |
| Bolt 74B-A9B (IQ2_S) | ` Red, yellow, blue.` / ` Paris.` · 17.3–18.7 tok/s decode, 34.8–35.2 prefill · range 0/0/0, exponent 0 |
| Lightning 30B-A3B (NVFP4) | ` Red, blue, and yellow.` · 23.5 tok/s, 38.1 prefill · range 0/0/0 |
| Qwen3-4B-Q8_0 (no sidecars, regression) | 90.7 tok/s decode, 162.8 prefill — unchanged |

### 22.27.1 Closed: the drafter's copy of the rule

[dflash.cpp](src/dflash.cpp) had its own uploader and its own 13 `gemm` calls,
so a sidecar on a `--draft` head set was dropped. Fixed the same way the trunk
was: `QuantTensor::gs` is filled in the drafter's `upload_w` from
`<tensor>.scale`, `DflashDraft::gs_in_` records the widest scaled contraction
(so nothing is allocated for an f16 head set, which is every non-NVFP4 one),
and all thirteen sites go through `DflashDraft::gemm_w` — `fc`, `wk`, `wv`,
`wq`/`wk`/`wv` in the block, the attention gate, `wo`, `ffn_gate`/`ffn_up`,
`ffn_down`, and both `lm_head` call sites. The scale rides the **input** for the
reason the trunk's copy documents: the f16 store saturates before an output
scale could recover it.

Pinned by `test_nvfp4_sidecar_dflash` (11 checks): an NVFP4 head set with a 0.5
sidecar against an f16 twin holding the same numbers, asserting the same
number of drafted rows and matching logits.

### 22.27.1b The third spelling now loads

`--draft` refused the one DFlash head set on this box for a name, not a reason:
[Laguna-XS-2.1-DFlash-ik_llama-Q8_0.gguf] declares
`general.architecture = dflash-draft` and names its tensors in poolside's own
spelling, and the table had neither.

| | in-house spelling | poolside spelling |
|---|---|---|
| arch | `dflash`, `qwen35-dflash-draft` | `dflash-draft` |
| fusion | `fc.weight` / `dflash.fc.weight` | `dflash_fc.weight` |
| encoder norm | `enc.output_norm.weight` | `dflash_hidden_norm.weight` |
| aux norms | stacked `enc.aux_norm.weight` | `dflash_aux_hidden_norm.N.weight` |
| FFN norm | `blk.N.ffn_norm.weight` | `blk.N.post_attention_norm.weight` |
| vocabulary | `vocab_size` | **absent** |

That last row was a real defect and not only a name: `cfg_.n_vocab` sized the
block's logits workspace and came out **0**, so the drafter would have loaded and
produced no rows at all. It now falls back to the target's vocabulary — which is
the only one the block can be run against anyway, since it embeds with the
target's table and reads the target's head — and a declared vocabulary that
conflicts is reported instead of silently sizing nothing.

Verified by `test_dflash_laguna_spelling` (6 checks): the same head set written in
both spellings drafts **byte-identical rows** (`worst == 0.0f`), so a spelling
moves names and never values.

### 22.27.1c Why the real pair accepts nothing — isolated, and one half fixed

End-to-end on the box, target `Laguna-XS.2-IQ4_XS`, drafter the file above,
greedy: 0/96 accepted under every knob the loader already exposed (causal 0/1,
rope-neox, kv-norm, layer-offset, block-pos ±1, window, zero-mask, row-0
feature). A toggle was never going to reach it, and the probes said why.

**The rows were starved, not displaced.** Rank of the target's own token on the
draft's row 0 was 5,896 / 100,352 with a peak logit of 4.7 and a mean of −0.07 —
a near-uniform row, i.e. chance. Not a position, not a rope pairing: an input.

**The falsifier that named it.** Feeding the readout the target's *pre-final-norm
hidden state* (the last captured feature block, rms 0.85 at the last committed
position) through the **target's** final-norm gain and the borrowed head
reproduces the target's own logits exactly:

| | top-3 |
|---|---|
| target, own decode (`--debug-topk`) | 330 (22.8125), 22345 (21.5938), 447 (20) |
| base-only readout in the drafter | 330 (22.8125), 22345 (21.5938), 447 (20) |

So the capture, its row, and the head were right all along, and the two missing
pieces were the base and the gain:

| | value |
|---|---|
| target `output_norm.weight` gain | rms **2.19** (max 3.5) |
| drafter `output_norm.weight` gain | **1.0 exactly**, every element of every norm in the file |
| drafter logits, own identity-gain readout | peak **4.7** vs the target's **22.8** |

The contract is therefore `logits = head(rmsnorm_tgt(base + delta))`, with the
base the pre-final hidden and the gain the target's. Two supporting fixes came
with it: the capture row was read as row 0 of the last commit (position 0 of the
prompt) instead of row `n-1` (the last committed token), and feeding row 0 the
fused **feature** instead of the last token's embedding moves the watched rank
from ~5,900 to 588 → 36 → 8 across rounds.

**And then the delta turned out to be noise.** With the contract in place, the
accepted count is monotonic in how much weight the block's output gets:

| delta weight | accepted (48 tokens, window 4) |
|---|---|
| 0 (base only) | **48/180 = 26.7%** — exactly one per round, i.e. proposal 0 always right |
| 0.01 | 12/192 = 6.2% |
| 1.0 | 1/192 = 0.5% |

At weight 0 every row reads the same base, so all four proposals are the same
token: 48 rounds, 48 accepts, **one token per round** — correct and still slower
than plain decode, because a round now verifies four rows to emit one token. So
the readout half of this drafter is now solved and measured, and the block's
forward is not: its output is the wrong quantity, not merely the wrong scale.
The next lever was the mask-row attention (each candidate row currently sees its
sibling mask rows). That lever is now built and measured — see 22.27.1d — and it
did not move acceptance; the level it has to move is the block's proposal for
`pos` itself, which is the row the mask cannot touch.

Nothing here is on by default — every arm above is an env switch — so the shipped
behaviour is unchanged: a `--draft` run still decodes with byte-identical text to
plainer decoding (same md5) and reports its own 0% rather than pretending.

Suite after: **4253/4253**.

### 22.27.1d The self-only mask is built, wired and measured — and it changes nothing

**Verdict: the mask is correct and active, and acceptance is bit-for-bit the same
with and without it. It was not the missing piece.**

`AttnDesc` grew one field, `skip_from` (default `-1`, nothing masked): when
`skip_from >= 0`, a query row at position `p >= skip_from` additionally treats
every key in `[skip_from, p)` as masked. The range is half-open so a row still
reads its own K/V, and row 0 of a block — `id_last`, at `p == skip_from` — has an
empty skip range and sees the context exactly as before. Implemented in all four
attention paths that a block row can take (HIP tile-loop, HIP bit-exact,
CPU f16/f32 and CPU int8), plus a unit test on the CPU reference that a masked
sibling is invisible and an unmasked one is not. The drafter arms it with
`KRK_DFLASH_SELF_MASK=1` at `skip_from = blk_pos + 1`.

**It is live, not decorative.** Every candidate row's top-3 logits move when it is
on — except printed row 0, whose value does not move because its skip range is
empty by construction:

| printed row (block row) | mask off — top-3 | mask on — top-3 |
|---|---|---|
| 0 (row 1, pos 7) | 17245 (4.199), 38632 (4.195), 1739 (4.070) | **identical** — empty skip range |
| 1 (row 2, pos 8) | 23204 (4.285), 50816 (4.281), 141 (4.031) | 23204 (4.563), 50816 (4.125), 9380 (3.781) |
| 2 (row 3, pos 9) | 50816 (4.547), 6519 (4.461), 3679 (4.445) | 23204 (5.344), 50816 (4.387), 6746 (3.949) |
| 3 (row 4, pos 10) | 20248 (4.152), 346 (4.063), 1848 (4.020) | 49190 (4.160), 24459 (4.012), 84965 (3.852) |

**And nothing in acceptance moves.** 48 tokens, window 4, `--greedy`, target
`Laguna-XS.2-IQ4_XS`, drafter `Laguna-XS-2.1-DFlash-ik_llama-Q8_0`:

| delta weight | mask off | mask on | text md5 |
|---|---|---|---|
| 0 (base only) | 48/188 = 25.5% | **48/188 = 25.5%** | identical |
| 0.003 | 48/188 = 25.5% | 48/188 = 25.5% | identical |
| 0.01 | 42/192 = 21.9% | 42/192 = 21.9% | identical |
| 1.0 | 0/192 = 0.0% | 0/192 = 0.0% | identical |

The reason is structural, not numerical: the mask only changes rows 1..3, and the
ceiling this drafter can reach is **one accepted token per round, from row 0** —
exactly the row the mask cannot touch. Whatever the block's later rows compute,
verification stops at the first rejection, so their logits are never reached. The
mask is now a correct part of the contract and should stay; it is not on the
critical path to acceptance, and the next candidate must be sought in the block's
proposal for position `pos` itself.

**One pre-existing defect surfaced by this measurement, and it is not mine.** When
acceptance is non-zero the emitted text stops being byte-identical to plain
decode: base-only (48/188) ends `…is a a natural number` where plain greedy ends
`…is a natural number that has`. Acceptance is a prefix match against the verify
pass's own logits, so either the batched verify's numerics diverge from decode's
at a close call, or the correction token is read from the wrong row. That is a
correctness bug in the speculative path, independent of the mask (the mask arms
agree with each other to the byte), and it must be fixed before any accept-rate
number here can be called safe.

Suite after: **4259/4259**, including the new `test_attention_self_mask` group
(6 checks: a masked sibling cannot move a row, the perturbed row itself still
moves, row 0 is untouched, the same perturbation DOES reach the row with the
mask off, a row whose only visible key is itself returns V, and the fully
masked case stays finite).

### 22.27.2 Not the same gap: dspark

`dspark` is refused by [arch.cpp](src/arch.cpp), and that is the honest state —
**nothing in this tree reads a dspark head set.** It is not a second uploader
with a scale bug; it is a drafter that was never written. The two files on this
box show why it is not a one-rule fix:

| file | arch | blocks | draft block | attention | extra heads |
|---|---|---|---|---|---|
| `dspark_gemma4_12b_q4pure` | dspark | 5 | 7 | non-causal | markov (rank 256), confidence |
| `Ternary-Bonsai-27B-dspark` | dspark | 6 | 4 | non-causal | markov, confidence, `log_snr_conditioning` ±9, `mask_token_id` |
| `dspark-MiniCPM-V-4_7-35B-A3B` | **dflash** | — | — | — | none (already served) |

So a dspark set is a *parallel* draft-block generator: mask/SNR-conditioned rows,
a non-causal window of 4 or 7 positions, a low-rank bigram head over the last
token, and a confidence head that decides which rows are worth proposing. None
of that is derivable from the trunk's per-token forward, and its weights are
quantized (`Q4_0`/`Q4_1`), so it would have to go through the device path, not
the host f16 `DraftHead`.

What a dspark forward needs settled before a line of it can be trusted — every
one of these changes the numbers, and there is no reference implementation on
this box to check against:

1. **Block inputs.** Seven (or four) rows from one `fc` of the last position's
   stacked target hidden states: are rows 2..n fed `token_embd` of the last
   token, of the row's own previous *prediction*, or a `mask_token_id`
   (`Ternary-Bonsai` declares one; the gemma4 set does not).
2. **The non-causal window.** `attention.causal = False` means row k sees the
   whole block, but with one forward the block's own predictions do not exist
   yet — so it is either a self-conditioning pass or a single parallel read.
3. **Markov head placement.** `w1` maps vocab → rank 256 and `w2` rank → vocab;
   whether that is added to row k's logits from row k-1's *sampled* token, from
   the target's last token only, or through the confidence head.
4. **Layer output scale.** `blk.N.layer_output_scale.weight` is a scalar per
   block; multiply the block output or the residual.
5. **SNR conditioning** (Bonsai only): how `log_snr` in ±9 is injected.

The only oracle for all five is the target's acceptance rate, which needs the
forward first — so this is a build-and-measure loop, not a plumbing pass. The
targets it would pair with are on this box (`gemma4-12B-Q4_K_M`,
`MiniCPM-V-4_7-35B-A3B`), so the measurement is available as soon as the
contract is.

### Pareto, for what is left on this thread

| rank | item | why |
|---|---|---|
| P0 | the DFlash block's delta | the readout is solved (base + target gain reproduces the target's logits exactly), and every gram of the block's output makes acceptance worse: 26.7% at weight 0, 0.5% at weight 1. The block is computing the wrong quantity |
| P1 | mask-row attention | each candidate row currently sees its sibling mask rows; a self-only mask is the next contract question and `AttnDesc` cannot express it |
| P0 | a dspark drafter | the only drafter family on disk with no code behind it; needs the block-draft forward, not another scale plumbing pass |
| P1 | `token_embd` | a lookup table, not a projection: a scaled NVFP4 table would need its rows rescaled at read time, and no converter emits one |
| P2 | NVFP4 prefill/decode kernels themselves | the scale fix is about *correctness*; the 4-bit dot path's throughput is separate work |

---

## HIP graph replay: measured, and refuted

**VERDICT: `tools/probe_graph.hip` — a captured graph replay of a whole decode token is `1.00x`, not 5x. Submission is 0.54 µs/launch, so 337 ops enqueue in 0.18 ms inside a 48.6 ms step. Do not refactor the 87 launch sites to a capturable stream; the prize is 0.0%.**

The hypothesis was worth testing because the trace shows 41% of the device
timeline idle: if the host cannot enqueue faster than the device retires, one
graph replay replaces ~340 launches and the step becomes device-bound. The
probe prices exactly that, at the engine's own granularity — 337 and 1767
dependent launches submitted op-by-op, then the same nodes captured once and
replayed.

| arm | host us | wall us | wall/op | host/op |
|---|---|---|---|---|
| submit, 18 us op (337) | 179.0 | 5759.0 | 17.09 | 0.53 |
| graph, 18 us op (337) | 146.3 | 5444.7 | 16.16 | 0.43 |
| submit, empty op (337) | 177.9 | 700.8 | 2.08 | 0.53 |
| graph, empty op (337) | 145.3 | 644.9 | 1.91 | 0.43 |

That table was the **first** run and it lied: it showed 1.06x, and a later run at
1767 ops showed 1.26x. Both were clock drift — the arms ran one after the other
and a 9070 XT idling at its low p-state ramps during the run, so whichever arm
goes first pays for the ramp. The shipped probe warms the clocks and alternates
the arm order every sample; the answer then collapses to noise:

| ops/token | submit wall | graph wall | speedup | gap recovered/op |
|---|---|---|---|---|
| 337 | 5376.4 us | 5472.7 us | **0.98x** | −0.29 us |
| 337 | 7131.7 us | 7082.3 us | **1.01x** | 0.15 us |
| 1767 | 24182.4 us | 24087.1 us | **1.00x** | 0.05 us |
| 1767 | 29791.7 us | 30397.4 us | **0.98x** | −0.34 us |

So the two numbers that matter, both from the engine's own traces rather than
the probe:

* **host submission is 0.53–0.75 µs per launch.** The 30B-A3B trace records
  1842 ops per decode step (`perf-traces/dev-KRK_DEV_PTR_1.err`, 27633 decode
  ops over 15 timed steps), so 995 µs of enqueue inside a 120.3 ms step — 0.8%.
  The 35B's 1767 ops/step is 0.95 ms of a 48.6 ms step, 1.9%.
* **the device-side gap between kernels is not what graphs remove.** An empty
  kernel costs 1.9 µs wall either way (1.92 submit against 1.92 replay), so the
  gap is in the stream's own retire path, not in the host's enqueue.

Replaying 1767 nodes saves nothing while a `hipGraphLaunch` still has to walk
the same dependency chain on the same stream. The record that *was* recovered
(1.26x, 5.7 ms) disappeared under interleaving, which is why the probe now
reports the interleaved number and the manual `hipcc` build command.

**What this rules out, and what it leaves.** It rules out submission granularity
as the decode ceiling, and with it the "collapse the launches" plan. It leaves
the same place the earlier span table did: the step is dominated by *waits*, not
by enqueueing, and the wait is a per-layer dependency that a graph cannot touch
because it is a dependency on the CPU. The recorded 30B trace shows the shape
plainly: the routing row is 256 ops/step (3840 over 15 steps) and 115.8 ms of
host time per step (1737.7 ms / 15) against a 120.3 ms step — a span that is
almost the whole token, and is the host blocked until each layer's kernels
retire. On a model whose dispatch never engaged at all:

```
[info ] device dispatch: not yet -- the routed set is not wholly device-resident
        (0 of 6144 slots) (retrying while the tiers fill)
```

That line is the real open item: the one-launch device dispatch — the only
mechanism in the tree that removes the per-layer round trip — has never engaged
on a model whose expert set does not fit, which is every model this box is
being pushed towards. `KRK_DEV_PTR=1` measuring 46.5 against 49.8 ms is
consistent with the arm being inert, not with it working.

---

## The one-launch dispatch, engaged past VRAM (measured)

> **RETRACTED IN PART — see "The ids were never written" below.** The
> engagement result stands (the table publishes, 3720 VRAM + 6520 pinned host)
> and the residency change stands, but the SPEED figures in this section were
> measured while the arm was reading uninitialised routing ids. With correct
> routing the same configuration is 0.37x on prefill. The earlier "2.20x" was
> the arm doing less work, not the machine going faster.

**VERDICT (as measured before the id fix): `ExpertCache::publish_dispatch_table`
demanded whole-set VRAM residency, so the arm was unreachable on every model it
exists for. It now publishes cells that a kernel can ADDRESS — VRAM or pinned
host — and on Qwen3.5-35B-A3B (16.8 GiB routed, 6.25 GiB budget) it engages with
3720 cells in VRAM and 6520 read in place.**

### What was wrong (code, not speculation)

```
[info ] device dispatch: not yet -- the routed set is not wholly device-resident
        (0 of 6144 slots) (retrying while the tiers fill)
```

`whole_set_resident()` required every (layer, expert) in VRAM. A routed set
larger than the card can never satisfy that, so the table was refused on every
attempt, `dev_table_` stayed null, and `dispatch_dev` was false for the whole
run. The 8-attempt retry budget was spent during load and never recovered.

### The change

| # | file | change |
|---|---|---|
| 1 | `src/expert_cache.cpp` | the publish gate is ADDRESSABILITY, counted per cell: VRAM, pinned host, never loaded, pageable. Absent or pageable refuses (the kernel would have to skip it); pinned host publishes, and the entry names the WARM slot's slices. `dispatch_table_host_cells()` reports the split |
| 2 | `src/engine.cpp` | with `KRK_DEV_PTR=1` on a set larger than the budget, the pin budget is raised to the WHOLE routed set — pageable RAM cannot be named by a kernel, so the overflow has to be pinned |
| 3 | `src/engine.cpp` | the warmup runs to the SET, not the 20 s ceiling, when the arm is wanted (a table cannot name a cell that was never filled), and the warmup is forced on even when the set fits |
| 4 | `src/engine.cpp` | decode (n == 1) keeps the per-expert path while the table has host cells |

(4) exists because of a measurement, not a preference. With the dispatch on for
decode as well, the instrumented run gave **6.5 tok/s (154.9 ms/step)** against
**7.8 (127.5)**, and the phase table named the cause:

```
other          decode      600   1506.0 ms      <- the one-launch kernel, 2.51 ms/layer
routing        decode     3600      0.0    2188.3 ms
```

The kernel launches one BLOCK per token and walks that token's k experts inside
it, so at n = 1 a 64-CU card runs one CU while the per-expert path it replaces
fans the same GEMMs across the card. With every cell in VRAM there is nothing to
hide behind and the launch is not the problem (that is what the 30B baseline
runs), so the exclusion applies only to a host-backed table.

### Measurement

Qwen3.5-35B-A3B-UD-Q4_K_XL, greedy, `--ctx 512`, 191-token prompt, 12-16 tokens,
UNINSTRUMENTED (the `KRK_TIME=1` rows are quoted separately because the recorder
costs 0.66 us per op and 23 us per event pair):

| arm | prefill tok/s | decode ms/step | decode tok/s |
|---|---|---|---|
| control `KRK_DEV_PTR=0` | 145.9 / 146.5 | 52.1 / 52.5 | 19.2 / 19.0 |
| dispatch `KRK_DEV_PTR=1` | **323.7 / 323.6** | **45.5 / 45.4** | **22.0 / 22.0** |
| repeat, control | 145.2 | 51.9 | 19.3 |
| repeat, dispatch | 319.3 | 45.2 | 22.1 |
| dispatch, `KRK_EXPERT_INPLACE=0` | 320.8 | 65.3 | 15.3 |

Instrumented, for the phase split: prefill 87.2 -> 308.7 tok/s (3.5x), decode
127.5 -> 125.0 ms/step.

The last row is the attribution: the decode gain is NOT the table (n = 1 never
uses it). It is the pinned whole-set tier being read in place — with in-place
disabled the same configuration is worse than the control (65.3 ms/step),
because it pays promotions out of a 16.8 GiB pinned tier it cannot evict.

### Correctness and cost

* table published: **10240 entries, 3720 VRAM + 6520 pinned host** (1.2 MiB).
* deterministic: two dispatch runs on the same prompt are byte-identical.
* `kraken-bench --moe-dispatch`: **7/7 cases pass**, `indirection=ok` (a gate/up
  swap changes the answer), worst per-element 7.79e-04 — so the table is read,
  not guessable.
* **quality caveat, measured**: routing on the device changes greedy
  continuations, and on a factual probe it was WORSE — host path
  `2, 3, 5, 7, 11.`, dispatch `1. 1 / 2. 2 / 3. 3 / 4. 5`. That is the arm's
  documented device-softmax drift compounding through 4-bit experts, and it is
  why the flag stays opt-in and off by default. The default path is untouched:
  control numbers before and after this change are 145.9/52.1 against 145.2/51.9.
* suite: **4306/4306 checks, 93 groups, exit 0**, including the new group
  `test_expert_dispatch_host_backed` (26/26) which pins the capability gate, the
  pageable refusal, the entry-to-slot identity, and the stale-then-republish
  path.

---

## The ids were never written

**VERDICT: `src/engine.cpp` — `ws_ids_`/`ws_wts_` are allocated at init and
NEVER written; `dplan` is computed and never used. The one-launch dispatch has
always picked its experts from uninitialised VRAM. FIX: call
`router_topk(ws_routerp_, n, ne, k, ws_ids_, ws_wts_, norm, scale)` in the device
branch and require those kernels in `dispatch_cap`. PROBE:
`KRK_DEV_PTR=1` on a factual prompt — before the fix the 24-token answer was
`1. 1. 1. 1. ...`; after it is byte-identical to the host-routing control.**

This is the bug that hid behind every previous measurement of this arm. The
kernel was correct, the table was correct, the launch was fast — and it was
multiplying bytes nobody had written. Two symptoms, both now explained:

* the arm "worked" (it ran, it produced plausible-looking text) while being
  degenerate on any prompt long enough to show it;
* it was FASTER than the host path, because garbage ids repeat: a handful of
  experts stayed hot, so the table read almost nothing.

### What the fanout kernel did and did not buy

`moe_table_slot_kernel` (one block per TOKEN × SLOT, plus an ordered reduce)
replaces the serial form's one-block-per-token for a small row set. Inside the
arm, with correct ids, on the decode shape (k=8 of 256 experts):

| form | decode ms/step | tok/s |
|---|---|---|
| serial (1 block/token) | 220.3 | 4.5 |
| fanout (k blocks/token) | **58.2** | 17.2 |

So the fanout is worth 3.8x inside the arm. It is NOT enough to make the arm
competitive, because one block per (token, slot) still issues a single expert's
host-memory reads at one block's request concurrency.

Correctness of the split: `kraken-bench --moe-dispatch` now forces both forms
(`KRK_MOE_FANOUT=0|1`), runs the fanout twice to prove determinism, and requires
the two to agree within the same 4e-3-of-rms tolerance the arm already allows
against the per-expert path. Measured: ≤1 f16 ulp on 6 of 7 cases, worst
1.5e-3 of rms on the 4-slot case where the f16 chain cancels (128 ulp on 17 of
6144 elements). 7/7 cases pass.

### The corrected numbers (uninstrumented, A/B/B/A)

Qwen3.5-35B-A3B-UD-Q4_K_XL, ctx 512, 191-token prompt, 12 tokens:

| arm | prefill tok/s | decode ms/step | text vs control |
|---|---|---|---|
| control `KRK_DEV_PTR=0` | 146.4 / 146.0 | 50.6 / 50.5 | — |
| dispatch `KRK_DEV_PTR=1` | **54.9 / 54.9** | 46.1 / 46.2 | **byte-identical** |

Two things to read there. The arm is now FAITHFUL — its output is identical to
the host path, which it never was before. And it is 2.7x SLOWER on prefill,
because the table kernel is gemv-shaped (one row of dots at a time) while a
prefill chunk is a batch GEMM, and the per-token blocks cannot stream 17 GiB of
host-resident experts through one block each. The decode row is +9% and that
gain does not come from the table at all: decode falls back to the per-expert
path (host cells are present) and reads its experts in place from the pinned
tier.

**So the one-launch dispatch is not the answer to MoE decode on a set larger
than the card** — not with a kernel shaped the way that one was. It stays opt-in
and off by default; the default path is unchanged at 146.4 tok/s prefill /
50.6 ms/step decode. What would make it competitive is a tiled,
multi-block-per-expert dispatch kernel — the same distance the prefill GEMM work
already covered — not another launch-shape tweak.

Suite after the change: **4306/4306 checks, 93 groups, exit 0**.

---

## Tiled dispatch: decode 51.9 -> 44.4 ms/step, one launch, same bytes

**VERDICT: `src/hip/kernels/moe_table.hpp` — the dispatch was ONE BLOCK PER
(token, slot) walking an expert's rows in order, so a decode step ran 8 blocks on
a 64-CU card and issued one row's loads at a time. Replacing it with three
row-tiled kernels (256 + 1024 blocks for the same step) and letting the arm serve
a single row takes Qwen3.5-35B-A3B decode from 53.9/51.9 to 44.4/44.4 ms/step
(+14-18%, 18.6-19.3 -> 22.5 tok/s) with byte-identical output.**

### Why tiling the rows is not a rewrite of the arithmetic

Two quantities resist a row split, and both are handled rather than approximated:

* **the activation row**, which every down block needs whole. It goes to global
  scratch as f32 — the exact pre-rounding product. The f16 the down dot reads is
  produced afterwards by the scale pass, so it is `f16(silu(g)*u)` and not a value
  rounded twice. Storing it f16 up front would have needed the row scale before
  the row existed.
* **the row scale**, which is a max over the whole row. Each gate/up tile writes
  the max it saw; one pass per (token, slot) folds the tile maxima, derives the
  same power-of-two scale the serial kernel derives, and stores `f16(p / scale)`.
  For scale 1 that is exactly the byte the serial form stores; for scale > 1 it is
  exactly the serial form's second pass — and because the f32 scratch was never
  clipped, no recomputation from weights is needed. `1 / scale` is exact, the
  scale being a power of two.

The dot itself is untouched (`table_row_dot`, same lane split, same chunk order),
as are the f16 round before the activation and the three explicit
round-to-nearest multiplies into the per-slot partial. The reduce still replays
the serial accumulation order.

### Measurement

Qwen3.5-35B-A3B-UD-Q4_K_XL, greedy, `--ctx 512`, 191-token prompt, 12 tokens,
uninstrumented, A/B/B/A:

| arm | prefill tok/s | decode ms/step | decode tok/s | text |
|---|---|---|---|---|
| control `KRK_DEV_PTR=0` | 144.2 / 144.8 | 53.9 / 51.9 | 18.6 / 19.3 | — |
| dispatch `KRK_DEV_PTR=1` | **155.7 / 156.4** | **44.4 / 44.4** | **22.5 / 22.5** | **md5 identical** |

Engagement, from the arm's own log: `440 layer(s) ran as one launch through the
expert table, 40 chunk(s) kept the per-expert path …, 0 fell back` — 40 layers ×
11 decode steps through the table, the single prefill chunk by design.

Against the recorded 50.6 ms/step baseline that is **-12%** (and -16% against the
paired control), and the arm no longer
pays the old prefill loss: `/ponytail-diag`-style, the table kernel computes one
row at a time where a prefill chunk is a batch GEMM, so a host-backed table now
serves only the ≤4-row sets it is good at and a chunk keeps the per-expert path
(`KRK_MOE_TILE_DECODE=0` restores the old "prefill only" exclusion for an A/B).
The 7-9% prefill gain above the control is the arm's own effect: `KRK_DEV_PTR=1`
pins the whole routed set, so prefill reads every expert in place instead of
competing with promotions.

### Correctness

* `kraken-bench --moe-dispatch`: **7/7 cases pass**, `indirection=ok` on all 7.
  Worst per-element 1.31e-3 of rms (the 4-slot case, where the f16 chain cancels);
  everything beyond 4e-3 of rms: **0 elements**. Two cases that the old slot form
  matched to 1 ulp are now **IDENTICAL**, which is the f32 scratch doing what it
  was put there for.
* the fanout is deterministic (second run, 0 of N differing) and the serial form
  is still bit-comparable against it (`KRK_MOE_FANOUT=0|1`).
* suite: **4306/4306 checks, 93 groups, exit 0**.
* end-to-end: the 12-token continuation is md5-identical between the arms.

---

## The same tiling on a chunk: 54.8 -> 80.4 tok/s, still half of 160

**VERDICT: `src/hip/kernels/moe_table.hpp` — a chunk's cost was never the block
count, it is that the (token, slot) form reads one weight matrix per PAIR where a
chunk's 191 x 8 pairs name at most 256 distinct matrices. Adding the same row
tiling to the chunk changed nothing (54.3 vs 54.3 tok/s). Keying the grid by
EXPERT instead — one weight row read, dotted against every token that routed to
it — reaches 80.4 tok/s (+47%), and a per-(row, token) agreement check puts it
within the same 1 f16 ulp of the serial form. It is still 2x below the
per-expert path's 160.1, so a chunk keeps the per-expert path.**

### The two rows that decide it

Qwen3.5-35B-A3B-UD-Q4_K_XL, greedy, `--ctx 512`, 191-token prompt, 8 tokens,
uninstrumented:

| chunk form | prefill tok/s | text |
|---|---|---|
| per-expert loop (shipped) | **160.7 / 160.1** | reference |
| table, per-(token, slot) tiling | 54.3 | identical |
| table, expert-major grouping | **80.4 / 80.6** | identical |

So the grouping is real (147%), and it is not enough. The remaining factor is the
inner dot's SHAPE, not the launch geometry: the grouped kernel still computes one
(row, token) product per warp — a `table_row_dot` over `n_embd` — where the
per-expert path hands the same expert a GEMM of m rows, so it dequantizes a
weight block once into a tile and multiplies it against every row at once. The
profile agrees on where the ceiling is: in the shipped path the chunk's experts
cost 3.7 ms/layer of a 30 ms layer (`moe.experts.batch`), so even a table path
with FREE expert math could only reach ~185 tok/s. Closing 80 -> 160 needs the
WMMA-tiled dequant-staging GEMM rebuilt inside a device dispatch, for at most
+15% over the path that already exists — which is why the gate stays.

`KRK_MOE_TILE_CHUNK=1` turns the chunk form on; unset (default) keeps the
per-expert path. `KRK_MOE_FANOUT=0|1|2` forces serial / per-pair / expert-major
for the bench.

### Correctness of the grouped form

`kraken-bench --moe-dispatch` now forces all three forms on every case and
requires the expert-major one to agree with the serial reference: **at most 0 f16
ulp on the two cases the per-pair form is also exact on, 1 ulp on four, 128 ulp
on the cancelling 4-slot case, 0 elements beyond 4e-3 of rms, 0 of 7 cases
failed**. A disagreement fails the case, not just prints. Suite: **4306/4306
checks, exit 0**; the 191-token prefill and the 12-token continuation are both
md5-identical between a grouped chunk and the per-expert one.

The max that the row scale needs moved from a per-tile summary into the scale
pass (it reads the f32 activation row it was already going to read). Same value
— a max does not care about order — one buffer smaller, and it is what lets the
gate/up pass be split by expert at all, since a per-tile max has no single
(token, slot) to report to.

---

## Where prefill and decode time actually goes (Pareto, measured)

Qwen3.5-35B-A3B-UD-Q4_K_XL, greedy, `--ctx 512`, 191-token prompt. Two kinds of
number below, kept apart on purpose: **A/B deltas are uninstrumented** (the
trusted ones), **per-op shares are instrumented** and are shares, not absolutes.
The recorder's own footer is the reason: in the 39-step run, device spans net of
the recorder were **34.1% of 16510.6 ms wall, 5986.5 ms (36.3%) idle between
ops, and 4887.3 ms (30%) neither** — plus 46.5% of ops sit at or below the
22.5 µs floor and are not individually resolvable.

### Prefill — two costs, both running at a quarter of the hardware

**VERDICT 1: `--fast-prefill` was off by default, and the default exact rung was
the prefill cost. 146.3 -> 231.1 tok/s (+58%) over two interleaved pairs
(145.3/147.2 vs 230.9/231.2), output md5-identical. FIXED: the rung lottery is
now the default (`--prefill exact` opts out); measured again after the flip at
**224.9 / 225.8 / 226.0 tok/s**, against 147.0 on the exact rung.**

**VERDICT 2: `upload_pinned` moves 7280 MB at 7.5 GB/s and `gemm(grid)` moves
4335 MB at 5.7 GB/s — against 24.3 GiB/s measured for a pinned read and
485 GB/s for VRAM. The staging path, not the arithmetic, is second.**

| # | prefill cost | measurement | share of the chunk |
|---|---|---|---|
| 1 | `gemm(grid)` — the exact SIMT GEMM rung | 760 ms net, 4335 MB @ 5.7 GB/s | removing it (`--fast-prefill`) = **-36% wall, +58% tok/s** |
| 2 | `upload_pinned` — pinned-source weight staging | 969 ms net, 7280 MB @ 7.5 GB/s | ~30% of device time |
| 3 | `warm_read(pool)` — expert tier reads, host worker pool | 2165 ms net / 2166 ms host | load phase: 6.25 s to fill 16.8 GiB |
| 4 | `upload` | 455 ms net with **409 ms idle** (queue not retired) | ~7% |
| 5 | `gemm(gemv_mw)` | 122 ms @ 23.9 GB/s | 4% |
| 6 | `upload_paged` | 112 ms @ 3.7 GB/s | 3% |

What is NOT the cost, contrary to the obvious guess: the experts. In the shipped
path the chunk's MoE experts are 3.7 ms of a 30 ms layer (`moe.experts.batch`),
and the per-(token, slot) table form costs 54 tok/s precisely because it reads
weights per pair. `attention` 7.8 ms and `recurrent` 7.5 ms for the whole chunk.

### Decode — the step is a chain of 40 round trips, then a staging path

**VERDICT 3: `src/engine.cpp` `dev_plan()` — the device top-k arm verifies itself
against the host on EVERY layer, so a decode step pays extra blocking copies per
layer. `KRK_DEV_PLAN=0` takes 51.1 -> 44.5 ms/step (-13%, two interleaved pairs:
51.0/51.3 vs 44.5/44.5). The one-launch table removes the same dependency a
different way: 43.5. They are not additive — both buy the same ~6.6 ms.**

| # | decode cost | measurement | share of a 44 ms step |
|---|---|---|---|
| 1 | per-layer launch/dependency gaps | 40 layers; ~13 ms/step of device ops in a 44 ms step | **~30 ms** |
| 2 | per-layer routing round trips | `download_f32` 3949 ms host / 1600 calls ≈ 2.5 ms each, 1560 of them in decode | ~6.6 ms, removed by host plan **or** table |
| 3 | `gemm_group(fused)` | 270 ms / 39 = 6.9 ms per step @ 160 GB/s | ~7 ms |
| 4 | `gemm(gemv_mw)` | 123 ms / 39 = 3.2 ms per step @ 23.6 GB/s | ~3 ms |
| 5 | `gemm(gemv)` | 55 ms / 39 = 1.4 ms per step @ 435 GB/s | ~1 ms |
| — | pinned staging | **not on the critical path** — see the retraction below | ~0 |

The instrumented net device times sum to ~47 ms against a 43.5 ms uninstrumented
step — inside the recorder's residual, which is why the ordering is usable and
the absolutes are not. `attention` is 0.5 ms/step; the MoE math is not the
problem.

### RETRACTION: `upload_pinned` was never running at 6-7.5 GB/s

The row I ranked #1 for decode in the first version of this section was the op
table's `upload_pinned` span — 1407 ms over 39 steps, 8769 MB, "6.2 GB/s". Two
things were wrong with it, and the engine's own log said so in a line I read past:

* the span is measured on the DEFAULT stream, so it carries the queue wait in
  front of the copy, not the copy's flight. 8769 MB is also mostly the ONE-TIME
  fill of the VRAM expert tier (6251 MiB, 3720 batches of 1.72 MB = one expert
  each), not per-step traffic;
* the engine's "true rate" line was itself broken. It bracketed a batch with two
  events recorded AFTER the copies were enqueued, so it measured the gap between
  two adjacent records: **2719.7 GB/s for 6251 MiB in 2.2 ms** on a clean run,
  20.7 GB/s on the next. It had already been wrong once before ("7098 MiB in
  107 ms = 64.6 GB/s on a link that does 27") and the fix for that had the same
  fault.

FIXED in `upload_pinned_batch`: the bracket now opens BEFORE the batch's first
copy, a ring of eight pairs replaces the single pair (with one pair, a deep
warmup queue meant almost no batch was ever timed while all its bytes were still
counted), and the rate is computed only over the bytes the brackets covered.
Measured after the fix: **220 MiB over 132 timed batches in 26.3 ms = 8.2 GB/s**,
against `bar_probe`'s **12.6 GB/s for one 1.72 MB copy** and **28.3 GB/s at
1 GiB**. So the gap is per-copy latency — the probe's sizes fit \~87 µs fixed +
1/28.3 GB/s — plus the pooled-cell release ordering that sits inside the same
span. Not the link, and not the aperture (a 1 GiB pageable copy measured 22 GB/s
here, so ReBAR is on).

And the decisive check that it does not matter for throughput: swapping the whole
pinned DMA path for the pageable ring (`KRK_WEIGHT_DMA=0`) leaves decode at
**43.8 ms/step against 44.2** — inside noise. **The staging path is not on the
decode critical path.** The correct decode number is row 1: ~13 ms of device ops
inside a 44 ms step, so ~30 ms is per-layer launch/dependency gap over 40 layers
— which is where the next probe belongs (launch count and inter-op idle per
layer), not in the copy engine.

### The ranked work this implies

1. **DONE — the WMMA rung is the default for chunks** (+53-58% prefill,
   identical text on this model/prompt). `--prefill exact` restores the old
   default. `--prefill-hash` pins the exact rung, because the gate's contract is
   that a `--chunk` difference is the *only* difference and the fast rung breaks
   that (chunk 128 `1fcaeaa8…` vs chunk 256 `3b7cab4a…`, logits differ too). The
   pin is skipped when the caller sets `KRK_GEMM_EXACT` explicitly, which is what
   the gate's own falsifier arm relies on: with `KRK_GEMM_EXACT=0` the hashes
   still disagree (verified), so the falsifier still falsifies. Exact rung is
   chunk-invariant at 64/128/256 (`9c2f4baa79731047`, unchanged from before this
   work). `kraken-server` follows the same default (it sets the same switch
   before the backend is constructed, and still honours an explicit
   `KRK_GEMM_EXACT`); `/health` answers ok on the 8B after the change.

   Post-flip, `--prefill-hash` is unchanged where it matters: exact rung at
   chunk 64/128/256 all print `9c2f4baa79731047`, the same hash as before this
   work, and `kraken-tests` `test_prefill_chunk_bitexact` is 26/26 with the full
   suite at 4306/4306.
2. ~~Fix `upload_pinned`'s 6-7.5 GB/s.~~ **RETRACTED — the number was a span,
   not a rate, and the path is off the critical path** (`KRK_WEIGHT_DMA=0`
   changes decode by 0.4 ms). What the correction exposes instead is row 1:
   ~13 ms of device work in a 44 ms decode step, i.e. ~0.75 ms per layer of
   launch/dependency gap. That is the next thing to measure, with a launch-count
   and inter-op-idle trap per layer.

2b. **Generation-time warmup still reads 16.8 GiB from the file at 2.7 GB/s**
   (`warm_read(pool)` 2165 ms, 2166 ms host). That is CPU-side I/O and it is
   real work at load, not a measurement artifact: 6.25 s of a 13 s run. The
   expert-tier fill itself (6251 MiB) rides on top at the 8.2 GB/s copy rate.
3. **Stop verifying the device plan every layer.** It is a proof, not a
   mechanism: sample it (first N layers, or 1-in-K) instead of paying extra
   round trips per layer forever. 6.6 ms of a 51.1 ms step is 13% for a guarantee
   that a once-per-run check already provides.
4. Everything else — `gemm_group` at 160 GB/s, `gemv` at 435 GB/s, attention,
   the MoE kernels — is already at or near rate and is not worth touching until
   1-3 are done.

Suite after the kernel work: **4306/4306 checks, exit 0**; `--moe-dispatch`
**0/7 failed** with all three forms gated against the serial reference.

### Final numbers, after the fast-rung default and the instrument fix

Qwen3.5-35B-A3B-UD-Q4_K_XL, greedy, ctx 512, 191-token prompt, 12 tokens, two
interleaved pairs:

| arm | prefill tok/s | decode ms/step | text |
|---|---|---|---|
| control `KRK_DEV_PTR=0` | 227.7 / 229.1 | 53.1 / 53.6 | reference |
| dispatch `KRK_DEV_PTR=1` | **260.5 / 259.6** | **43.7 / 44.0** | identical |

Prefill is 225-260 against the 147 it was before the rung default, decode is 43.7
against the 50.6 baseline, and the instrument change is measurement-only — the
decode pair above is unchanged from before it, which is itself the check that the
fix did not disturb the path.

### The chunk-batch kernel, measured end to end

| chunk form | prefill tok/s | text |
|---|---|---|
| per-expert loop (shipped) | **159.7** | reference |
| table, per-(token, slot) rows | 54.3 | identical |
| table, expert-major grouping | 80.4 | identical |
| expert-major + 4-token weight reuse | **144.2** | identical |
| expert-major + 4/2/1 reuse (tail batched) | 142.8 | identical |

The 4-token batched read is the whole difference between 80 and 144: the earlier
loop called one `table_row_dot` per group token, a serial chain of `n_in/32`
dependent chunk loads with a runtime-bounded loop around it, so no unrolling and
nothing to overlap. Sharing one weight chunk across four independent accumulators
removed ~80% of the latency exposure without changing one per-token sum (the
chunk order per token is identical). It still lands 10% short of the per-expert
path, and the Pareto above says why the remaining gap is not worth chasing: the
chunk's experts are 12% of the layer.

## The per-layer gap: 43.7 -> 35.2 ms/step, and the readback was never needed

The previous section left one number unexplained: ~13 ms/step of device ops inside
a 44 ms step, i.e. **~30 ms that no op row owned**, spread over 40 layers. That is
the launch/dependency gap. Profiling one decode layer in the dispatch arm
(`KRK_TIME_HOST=1`, the passive host recorder, so the device is untouched and the
row is real wall time) named it in one run:

| row (n = 1) | ops | host ms | per op |
|---|---|---|---|
| `moe.route.fetch` | 440 | 343.6 | **0.781 ms** |
| `moe.route.rest` | 440 | 80.3 | 0.183 ms |
| `moe.route` (parent) | 440 | 517.9 | 1.177 ms |
| `moe.experts` | 440 | 0.0 | 0.000 ms |

0.781 ms × 40 layers = **31 ms/step inside `moe.route.fetch`** on a 43.7 ms step.
And on that path the bytes it downloads are read by nobody: the one-launch
dispatch takes ids and weights from `ws_ids_`/`ws_wts_`, which `router_topk()`
fills on the device in the same breath as the softmax. The [n, n_expert] f32
readback was the layer's only blocking point, and it was dead traffic.

### The fix

`dispatch_capable()` is the capability expression, factored out so the decision to
*skip the readback* and the decision to *run the dispatch* are one piece of code
instead of two that can drift; `dispatch_now` is that plus the row-count gates and
`!hybrid_on_` (hybrid is `hybrid_on_ && ...`, so with the switch off a layer
satisfying `dispatch_now` satisfies `dispatch_dev`). The publish retry moves above
the fetch, because the fetch is skipped on the strength of its answer.

Four host readers take a decode row's routing and would have been handed the
previous layer's numbers: the expert predictor, the next-layer prefetch hint, the
expert scan (it writes the hot-expert JSON), and `KRK_DUMP_ROUTE`. All four are
opt-in, so `dispatch_now` also requires `!route_consumed`; the default
configuration is the one that gets the whole win, and an arm that watches routing
still gets its readback.

### Measured

Qwen3.5-35B-A3B-UD-Q4_K_XL, greedy, ctx 512, 24 tokens:

| arm | decode ms/step | text |
|---|---|---|
| control `KRK_DEV_PTR=0` | 48.9 | reference |
| dispatch `KRK_DEV_PTR=1`, before | 43.7 / 44.0 | identical |
| dispatch `KRK_DEV_PTR=1`, after | **35.2 / 35.7** | identical |

A second pair at 48 tokens on a prose prompt agrees and is the stronger evidence
(identical text byte for byte): control 45.7 ms/step, dispatch **35.1 ms/step**.

`moe.route.fetch` reads **0.0 ms over 920 ops** in the new run — a number near zero
is the proof the readback is gone, not an assumption. The guard is verified the
other way: `KRK_MOE_NEXT_PREFETCH=1` brings the fetch back (920 ops, 304 ms) and
that arm's decode is 76.7 ms/step, its own prefetch cost, not a regression. Suite
**4306/4306**. Prefill keeps the readback (`n > 4` ⇒ the host plan owns the chunk)
and does not regress: 125.4 tok/s against the control's 102.0 on the same prompt.

### What the step is made of now

| row | ms/token | note |
|---|---|---|
| `decode.logits.dev` (head GEMV + device top-k) | **29.4** | 273 MiB of Q4_K head weights / token |
| everything else (40 layers) | 5.8 | = 0.145 ms/layer |

The 40-layer gap this task set out to close is gone: routing is 0.037 ms/layer and
the dispatch itself 0.009 ms/layer. **The decode Pareto has one entry now, and it
is the output head**: 273 MiB in 29.4 ms is **9.3 GiB/s** of weight bandwidth on a
card whose copy engine does 28 GB/s and whose resident weights are in VRAM, so the
head is a GEMV-shape problem, not a bandwidth ceiling — the same failure mode as
`gemm_acc(gemv)` at 183 GB/s against `gemm(gemv)` at 586. That is the next probe,
and it is worth more than everything else in this document combined: 29.4 of 35.2
ms.

## Correction: the head was never the bottleneck. `router_topk` was.

The section above named the output head as decode's single largest cost, from
`decode.logits.dev` reading 29.4 ms/token. **That reading was wrong, for the same
reason the `upload_pinned` row was wrong: it is a span that blocks, not a span that
works.** `fetch_logits` ends with the device top-k, whose 4-byte readback waits for
*everything already queued* — so the row is the drain of the 40 layers in front of
it, and the head kernel is a passenger inside it.

The isolated probe settles it. Same shape as the head, `kraken-bench --dtype q4_k
--out 248320 --in 2048 --rows 1`:

| arm | time | bytes | rate |
|---|---|---|---|
| isolated, 1 row | 610.68 / 641.95 us | 286.06 MB | **468.4 / 445.6 GB/s** |
| isolated, 2 rows | 1246.49 us | 286.06 MB | 229.5 GB/s |

and the engine's own per-op table agrees: `gemm(gemv)` over the whole run is
5207.2 MB at 454.8 GB/s, with the head's 286 MB once per token inside it. The head
costs **0.6 ms of a 35 ms step**. There is nothing to win there — the premise of
"9.3 GiB/s" came from dividing the head's bytes by the queue drain.

### What actually owns the step

Measured one op at a time with `KRK_TIME=<op>` (the table's own advice: cross-op
rows are contaminated, a single-op run is not):

| op | per layer | over 40 layers | what it is |
|---|---|---|---|
| `moe_dispatch` | 0.428 ms | 17.1 ms | 14.2 MB/token of expert weights |
| `router_topk` | 0.385 ms | **15.4 ms** | a 256-entry scan |

`moe_dispatch` is **PCIe-bound, not kernel-bound**: the routed set is 16.8 GiB
against a 12.1 GiB device budget, so 6520 of 10240 cells are read in place from
pinned host memory (24.3 GiB/s measured). ~9.5 MB over PCIe at that rate is
0.39 ms — the measurement to within 10%. Nothing to fix in the kernel; the fix is
VRAM, which is a model-placement decision, not a rewrite.

`router_topk` is a **defect**. The kernel walked the full expert range on *every*
thread and let thread 0 keep its answer, so a 128-thread block did one thread's
work: a k-pass scan (8 slots × 256 experts, plus an O(s) "already taken" scan)
of dependent global loads that cannot be hidden by any amount of occupancy,
with a block barrier per slot. 0.385 ms for 2048 candidate comparisons.

### The fix

[router.hpp](src/hip/kernels/router.hpp): thread `e % kThreads` owns experts
`e, e + kThreads, …`; the block reduces (value, id) warp by warp and the tie rule
travels with it as `ov == best && oi < bid`. That is the same answer as the
ascending scan with a strict `>`, because the selection key is the pair
(value, −id) and a maximum over a total order does not depend on the order it is
folded in. No new arithmetic, no reordering of anything else.

Measured: `router_topk` 320 calls, 123.05 ms → **16.82 ms**, i.e. 0.385 → **0.0526
ms per layer** (7.3x).

### Decode, after

Qwen3.5-35B-A3B-UD-Q4_K_XL, greedy, ctx 1024, 48 tokens:

| arm | decode | text |
|---|---|---|
| control `KRK_DEV_PTR=0` | 46.2 ms/step (unchanged, 45.7 before) | **unchanged** |
| dispatch, before this fix | 35.1 ms/step | identical to control |
| dispatch, after | **22.4 ms/step (44.7 tok/s)** | identical to control |

The control arm is the correctness witness: it runs the same device top-k through
the host plan verify, and its text is byte-identical to before the kernel change,
so the parallel reduction selects the same ids. Dispatch text is identical to
control. Suite **4306/4306**.

### Where decode stands

44.7 tok/s against the 19.7 tok/s this investigation started from (2.3x), and
22.4 ms of the remaining step is ~17 ms PCIe + ~2 ms top-k + ~3 ms the rest. The
next lever is not a kernel: it is getting more of the routed set into VRAM, which
is what turns the 17 ms into a 0.5 ms read.

## The 17 ms of PCIe streaming: residency was the lever, and residency was even

`moe_dispatch` measured 0.428 ms/layer because 6520 of 10240 routed cells live in
pinned host memory and are read in place at 24.3 GiB/s. The obvious move — give
the pool more VRAM — is the wrong one on this card: the 3.2 GiB floor in
`vram_floor_bytes` exists because filling the card to 0 MiB free once took a
desktop down and made a 2.2 MiB allocation cost 30 ms, and the guarded clamp is
what keeps a decode run from doing the same. **The pool did not need to be
bigger; it needed to hold the right cells.**

Without an index the warmup fills the tiers in *even order*, so the 3720 VRAM
cells are an arbitrary 36% of the set — and they capture exactly 36% of the
picks. With a ranked order they capture **96.8%**:

| residency | cells in VRAM | share of picks served from VRAM | `moe_dispatch`/layer | ms/token over 40 layers |
|---|---|---|---|---|
| even order (no index) | 3720 | 36% (by construction) | 0.428 ms | 17.1 |
| ranked by observed routing | 3720 | **96.8%** | **0.192 ms** | **7.7** |

Same bytes, same budget, same kernels. The ranking is the whole difference, and
the mechanism is already shipped: `--expert-adapt` folds a run's own routing into
`<model>.krakenexperts.json` (no scan, no extra device work) and the warmup ranks
by it. `--expert-scan` is the expensive way to get the same file; a single 48-token
decode with `--expert-adapt` is enough to start, and repeated runs merge (retain
0.5) so the index converges on the user's real workload.

### Measured (48 tokens, greedy, ctx 1024, ctx-invariant text)

| workload | no index | index from that workload | merged index (3 workloads, 216 positions) |
|---|---|---|---|
| river / fish prose | 22.4 | **14.1** | 19.9 |
| PCIe latency note | 22.3 | 21.0 | **14.6** |
| KV cache trade-offs | 22.5 | — | **14.3** |

Text is byte-identical to the `KRK_DEV_PTR=0` control on every workload tested,
before and after: residency changes which memory a cell is read from, never what
is computed. Merged-index coverage is 90.2% of picks, which is why the merged
column is 14-15 ms on a workload it has seen and 19.9 on one whose share was
halved — the index is a *workload* prior, not a property of the model.

### What it costs and what is left

- `--expert-adapt` runs cost the router readback (~8 ms/step, 31.5 against 22.4)
  because tracking routing re-enables the drain this document removed two
  sections ago. That is one drain per layer for a counter that only needs the
  8 chosen ids: batching all 40 layers' ids into one readback per token is the
  obvious next fix.
- `moe_dispatch` is now 0.192 ms/layer: 7.7 ms/token for 14.2 MB of VRAM reads is
  ~74 GB/s, well under the 445 GB/s the isolated gemv does, because a layer is
  four launches (gate/up, scale, down, reduce) of small grids. Merging the scale
  pass into the gate/up epilogue is the next lever inside the kernel.
- Decode stands at **14.1-14.6 ms/step (68-70 tok/s)** on a ranked workload
  against 19.7 tok/s at the start of this investigation — 3.5x — with 22.4 the
  honest number for a workload the index has never seen.

## Tracking stopped costing 8 ms/step, and where the load phase actually goes

Two things this document asked for, both measured.

### `--expert-adapt` no longer pays the router readback

Tracking used to set `scan_on`, which forced the router logits back to the host on
every routed layer so the *host* plan could count two integers per expert. That is
40 drains per token, and it made a tracking run (31.5 ms/step) slower than the path
it was measuring (22.4).

The ids and the probabilities are already on the device: `ws_ids_` from
`router_topk`, `ws_routerp_` from `router_softmax`. `Backend::scan_accum` now folds
them into a per-(layer, expert) histogram there — one small block per layer, no
readback — and the host reads the finished histogram once, in `scan_dev_fold`,
before the index is written. `route_consumed` no longer counts tracking when those
buffers exist, so the dispatch keeps its single-launch path.

Interleaved pairs, same prompt, same build:

| pair | plain | `--expert-adapt` |
|---|---|---|
| 1 | 12.0 ms/step | 12.1 |
| 2 | 11.3 | 11.4 |
| 3 | 11.3 | 11.3 |

`moe.route.fetch` under `--expert-adapt`: **0.0 ms over 600 ops** (it was 304-343 ms
over 920). The index it writes is exact, not approximate — total hits came out at
91,840 for 287 positions = 287 x 8 slots x 40 layers, every pick counted once.
Suite 4306/4306, generated text byte-identical to the control.

### The load phase: the reads were never the problem

The warmup's own line now prints its split, because "the load is slow" and "the
disk is slow" are different claims and nothing in the tree could tell them apart
(`prefill_read_ms` reads 0 here: the phase-split counters are only fed once the
host clock says "prefill", and a warmup runs before that — a printed 0 ms for a
phase that exists to read).

Qwen3.5-35B-A3B, 16.8 GiB routed set, 17.2 GiB WARM tier:

| part | time | rate |
|---|---|---|
| file read (`read_host_batch`, 4 readers) | 2265 ms | **7.4 GB/s** |
| prep = host allocation/pinning of the slices | 4156 ms | 30,720 slices |
| commit to the tier | 1 ms | |
| phase total | 7573 ms | |

**The read is already parallel and already at the box's ceiling.** An independent
probe of the same file: 1 thread 4.03 GB/s, 4 threads 6.64, 8 threads 6.85, 16
threads 7.26. The engine's pool gets 7.4-7.7 GB/s with its default 4 readers, and
`KRK_WARM_READERS=8` changes the phase by 2% (6649 vs 6783 ms). There is nothing
to win by making the reads more parallel.

The 4.2 s of prep is the pinned WARM slices, and it is **page-proportional, not
call-proportional**: halving the tier (`--expert-warm-mb 8192`) halves every part
of it — read 1068 ms, prep 1662 ms, phase 3717 ms. That rules out the obvious
"one big allocation instead of 30,720 small ones" fix: the cost is the pages being
locked, and bulk allocation locks exactly as many.

So the remaining lever on load is *overlap*, not bandwidth: pinning is CPU/kernel
work (4.2 s) and reading is I/O (2.3 s), and they are strictly serialised per
layer today — prepare layer L, read layer L, commit, then L+1. Preparing L+1 while
L's batch is in flight bounds the phase at max(4.2, 2.3) ≈ 4.2 s instead of their
sum, about a 40% cut. That is the next change, and it is a pipeline (prep thread vs
reader pool), not a wider reader pool — this measurement is what decides that.

One instrument change rides along: `krk::host_available_bytes()` now memoises its
`GlobalMemoryStatusEx` for 20 ms, since staging asks it once per expert. It did not
move the prep time, which is how we know the syscall was not the cost.

### The load phase, pipelined: the read is now off the critical path

The split above said where the remaining 7.6 s was, and the answer was not "the
disk". `prefetch_layer` allocated the whole layer's slices and only then read
them, so the phase cost the SUM of two unrelated bottlenecks: ~2.4 s of
page-cache I/O and ~4.5 s of the driver page-locking 30,720 slices. Nothing tied
the two together.

One helper thread now allocates chunk k+1..n while this thread reads chunk k. It
takes no slot-map work -- `slots_`, the ledgers and the eviction scan stay on the
caller, the same rule `prefetch_warm_next` already follows -- so the only thing
crossing a thread boundary is each job's own destination pointer; the release
store on `alloc_ready` is what publishes a chunk's buffers. Load-time only: a
stage issued inside a token keeps the old single-batch shape exactly, because
there the batch is one layer's routed set and a thread would cost more than the
overlap returns.

Same model, same flags, three runs each side; the warmup phase's own wall time:

| part | before | after |
|---|---|---|
| read (4-reader pool) | 2356-2502 ms | 2523-2540 ms |
| alloc (30,720 slices) | 4518-4750 ms | 5401-5488 ms |
| of which overlapped by the read | 0 | **2253-2266 ms** |
| **phase** | **7214-7584 ms** | **6136-6256 ms** |

**-14.6% against the slower baseline, -19% against the faster one** (median
7214 -> 6164 ms). Not the 40% the split predicted, and the reason is in the
second row: allocation costs 16-20% more when it runs beside the read (4518 ->
5431 ms). Both are memory-subsystem work -- pread copying page-cache bytes into
pinned pages that are being faulted and locked next to them -- so overlapping
them hides 89% of the read but inflates the half that remains on the critical
path. What the pipeline bought is 1.05 s; what contention took back is ~0.9 s.

The next lever is therefore the allocation itself, not more overlap: it is now
the whole critical path. One arena per layer would replace 768 `hipHostMalloc`
calls with one, and whether that helps depends on a per-call cost the earlier
tier-halving test could not separate from the per-page cost (halving the tier
halves both). `KRK_WARM_CHUNK` exists so that experiment can be run without a
rebuild.

Verification: suite 4306/4306, and again with `KRK_WARM_CHUNK=1` so the
multi-chunk path runs over the four-expert staging fixture -- including its
byte-for-byte compare of every staged slice against the source. Generated text is
identical across runs and identical to the pre-pipeline build; the tier ends up
with the same 10,240 experts / 17,208 MiB and the same 3,720 VRAM promotions,
which is the shape a dropped or short read would have changed.

---

# The warmup's allocation cost is per PAGE, not per call — and it is the phase

Hypothesis to settle: the load phase spends 4.5-5.5 s in `hipHostMalloc` because
the WARM tier allocates THREE slices per expert (30,720 calls for a 17.2 GiB
tier). If the cost is per call, one arena per layer (40 calls) removes it. If it
is per page, an arena removes nothing — the driver still probes and locks 4.2M
pages — and only a LONGER RUN of pages per call helps.

`tools/probe_warmalloc.hip` (new, no project headers: it measures the runtime)
holds bytes constant and moves count against size:

| block | calls | block MiB | alloc ms | us/call | ns/page | GB/s |
|---|---|---|---|---|---|---|
| 1/3 expert (today) | 30,720 | 0.57 | **4110** | 133.8 | 911.6 | 4.49 |
| 1 expert | 10,240 | 1.72 | **2122** | 207.3 | 470.7 | 8.70 |
| 2 experts | 5,120 | 3.44 | 2175 | 424.8 | 482.4 | 8.49 |
| 16 experts | 640 | 27.52 | 1810 | 2828 | 401.4 | 10.20 |
| 64 experts | 160 | 110.09 | 1674 | 10465 | 371.3 | 11.03 |
| 256 experts | 40 | 440.34 | 1525 | 38113 | 338.1 | 12.11 |
| whole tier | 1 | 17614 | 1439 | — | 319.1 | 12.84 |

Row 1 against a tenth of the pages at the same call count (324 ms) says the
fixed per-call cost is ~10 us: 326 ms of 4110. Everything else is the page run,
and the page run is why three-to-one is the change worth making: **4110 -> 2122
ms, 48%**, for one pointer discipline. The remaining 310 ms sits behind a slab
allocator's free list AND a promotion-completion event for every recycled slot
(a slot handed to a new expert while a copy out of it is still in flight is the
use-after-free that reports as plausible text that differs every run), so it was
declined at 7% of the phase.

## What changed

One host block per expert, three slices at aligned offsets inside it
(`warm_block_layout` in `src/expert_cache.cpp`), and one release through the
lowest-offset pointer (`warm_block_base`). Padding is at most 126 B per expert
(two 64-byte alignments, and a slice under 64 B is aligned to its own size so
the suite's 8-byte fixtures still measure 24 bytes an expert). Every allocation
site and every free site was moved: read-through fill, the warmup batch, the
read-ahead worker (`PfEntry::base` now owns the block), the commit drop, and
`drop_host`. Callers that read `h.gate/up/down.data` are unchanged — the pointers
are real pointers, just into one block.

## In-engine A/B, same session, same command, alternating builds

`kraken -m Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf -p "The capital of France is" -n 16
--seed 42`, tier auto-sized to
17,208 MiB, fully pinned, 10,240 slots (identical resident set and pin split on
both arms: 16382 MiB pinned, 369 device evictions):

| part | 3 slices/expert | 1 block/expert |
|---|---|---|
| read | 2609-2790 ms | 2473-2590 ms |
| alloc | 5343-5690 ms | **3420-3665 ms** |
| of which hidden by the read | 2413-2456 ms | 2006-2132 ms |
| **phase** | **6219-6541 ms** | **4249-4530 ms** |

**-31% on the phase** (median 6380 -> 4291 ms), -36% on the allocation itself.
The first run of any build costs 1.5-3x the rest (8980 ms alloc on the slice arm,
13224 on the block arm); those are reported as cold-state, not as the result.

Verification: suite **4306/4306** (three expectations moved from per-slice to
per-block counts, and the comment beside each now says why), generated text
BYTE-IDENTICAL between the two allocation shapes, across five runs of the same
build, and with `KRK_WARM_CHUNK=1` driving the multi-chunk path. The tier ends
at the same 10,240 experts / 17,208 MiB with the same pinned split — the shape a
dropped or short read would have changed. Decode moved 47.7-47.8 -> 44.0-44.9
ms/step, which is inside this config's run-to-run spread and is not claimed.

---

# The 16-expert slab: built, measured at the allocator, NOT shipped

Built and then reverted, on evidence. What it was: a slab pool of up to 16
experts per allocation with per-slot free lists, plus a backend primitive to
make recycling safe — `host_reuse_mark` / `host_reuse_ready` (an event on the
default stream, recorded AFTER a batch of frees, so a slot goes back into its
free list only once every promotion copy that could still read it has landed;
the same barrier `release_host_*_deferred` uses, applied to a buffer that is
reused instead of freed).

The prize was real but small: probe rows 2 and 4 are 2122 ms against 1810 ms for
the same 17.2 GiB tier, i.e. **at most 310 ms of a ~4.3 s phase (7%)**.

Why it is not in the tree: a slab commits its whole capacity the moment it is
created, so evicting ONE expert out of a 16-slot slab returns no budget at all —
`host_bytes_ + slab_slack_` does not move. `make_warm_room` then keeps evicting,
and because `pick_warm_victim` chooses by LFU across the whole tier the victims
are spread over different slabs, so it takes on the order of `cap` evictions
before any slab empties and returns its capacity. That is up to a 16x eviction
amplification, and it lands precisely in the configs where the tier churns — the
ones where WARM traffic is already the step (17 ms/token of pinned streaming on
the 35B, 274 MiB/token on REAP-288).

This workload cannot measure that downside: the tier is sized to the corpus and
the stats line reports **0 WARM evictions** for the whole run, so the regression
would be invisible here and only show up on an 8 GB tier over the 74B. Shipping
an allocator change that is unmeasurable on the target and dangerous on the
fallback is how a decode path becomes "plausible text, different every run".

What would settle it: eviction that targets the emptiest slab (or evicts a slab
whole), measured on a config that actually evicts. That is a policy change in
`pick_warm_victim`, and it needs its own baseline before it is worth 310 ms.

## The propose pass, measured in isolation, and why cutting it cannot save k=1

Question: the round's overhead is the binding constraint, so can the block's
3 ms (embed + block + head) per proposal be cut?

First the accounting had a hole in it. The round's four named terms -- propose,
verify, snapshot, refeed -- summed to 687 ms of a 1024 ms run, and every round
was paying exactly one unnamed second target forward: a pre-check miss forwards
the token it just emitted, a full match forwards the bonus token so the next
round has its state. `mtp_extra_ms_` / `mtp_extra_n_` now measure it, and it
closes the books:

Lightning 30B-A3B NVFP4, k=1, 32 tokens, c=2048, greedy, 19 rounds, 1024.4 ms
(53.92 ms/round, 1.68 tok/round, 31.2 tok/s):

| term | total | per round | share |
|---|---|---|---|
| verify (1 batched row) | 337 ms | 17.7 ms | 33% |
| **second target forward** | **458 ms** | **25.4 ms** | **45%** |
| refeed (block re-walk) | 161 ms | 8.5 ms | 16% |
| propose | 65 ms | 3.4 ms | 6% |

Of the refeed's 161 ms, 122 ms is the block's expert promotion (60 of 390
acquires missing) -- so *none* of the propose pass's cost is promotion; it is
pure kernel work. The drained phase split puts it at embed 0.16 ms + block body
1.65 + head 0.48 + argmax 0.12 = 2.41 ms of work inside a 3.4 ms wall, the rest
being launches and bookkeeping across four kernels. Two thirds of the pass is a
single MoE layer's forward, which is what the pass is for.

So the cut available is ~1 ms of a 53.9 ms round -- 2%. Worth knowing, but the
question that matters is the ceiling, and the reply log measures it directly:
KRK_MTP_REPLAY_IN replays recorded proposals with the block never running, which
is the propose pass at exactly zero, plus the refeed it skips.

| prompt | plain | spec k=1 | block free |
|---|---|---|---|
| "The quick brown fox…" | 40.5 tok/s (24.7 ms/step) | 31.2 (53.92 ms/round) | 40.7 (41.35 ms/round) |
| "def fibonacci(n):" | 36.8 (27.2) | 26.5 (63.50) | 36.3 (46.45) |
| "List the first ten primes…" | 38.0 (26.3) | 27.2 (56.06) | 37.5 (40.60) |

A **free drafter lands on parity**, within 1.5% either side, on all three. The
break-even for the first prompt is 41.35/24.7 = 1.674 tokens per round against
1.68 delivered -- the arm is sitting on the line, and the other two are 1-2%
under it. Cut the propose pass to zero and the round is 41.35 ms for 1.68
tokens: 24.6 ms/token against plain decode's 24.7. There is no propose
optimisation that makes this head win, because the propose pass is not what the
round is made of -- 78% of it is two target passes for 1.68 tokens, i.e. 1.19
target passes per token against plain decode's 1.00.

The design's ceiling is the k=1 cap of 2.0 tokens/round. With the block free and
a *perfect* head: 41.35 ms for 2 tokens = 20.7 ms/token, +18% over plain. With
the real block paying 3.4 ms: 27.0 ms/token, 9% *worse* than plain even at 100%
acceptance. Every accepted proposal is worth 24.6 ms of round time and the block
spends budget on drafts that are only 74% right.

What would change the arithmetic rather than polish it: a head that accepts
chains at k>=2, where the marginal verified row costs 7.7 ms on top of a round
already paid for. At k=2 that needs 2.49 tokens/round against a cap of 3.0 --
about 75% on the second draft -- and the measured k=2 arm delivers 1.37. The
head is not close. Cutting its 3 ms is measuring the wrong term.
