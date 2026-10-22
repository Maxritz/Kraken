# Kraken MoE, prefill/decode, and target-model deep dive

**Cut-off: 2026-10-22.** This is an evidence review of the current tree and recorded measurements, plus a deliberately opt-in route-hint predictor. It does not claim that unsupported model families now run or that the predictor improves throughput without a controlled model benchmark. The 2026-10-22 continuation appends work to the previous investigation: the local qwen35/qwen35moe variants are distinguished from qwen4exp, and additional papers supplied during the work are classified below. Previously recorded measurements are identified as such; only the explicit checks in the verification section were run in this continuation.

## Executive result

There is no single “MoE is slow” cause. The evidence splits the paths:

1. **Large MoE decode is dominated by the work to find, read, promote, and schedule experts, not by the expert GEMV kernel.** On Qwen3.8 REAP-320, host phase timing attributes about **81.1 ms of an 86.9 ms pinned-WARM decode step to a per-layer router readback** (48 layers); the expert host loop is another 33.7 ms in the earlier 95.9–100 ms/step pageable arm. The underlying GEMV is ~2.65 µs/call at 743 GB/s. See [REAP decode autopsy](AUDIT-REAP-DECODE.md#11-where-the-remaining-87-96-msstep-goes-a-per-layer-host-round-trip).
2. **Prefill can be worse for a different reason:** each prompt row routes a different union of experts, while the current group/row-ladder GEMM paths are inefficient and large unions exceed bounded pre-staging. On a Qwen4exp test file, five prompt tokens took 985 ms (197 ms/row), versus a decode row around 20 ms. `gemm(gemv_m)` at rows 2–8 measured 29 GB/s versus `gemv` at 743 GB/s. This is a real red flag but is a recorded REAP-320 measurement, not a current fresh run; row count, routed expert union, `--chunk`, and expert cache state must be held constant in the next measurement.
3. **“Only hot experts loaded” is not necessarily a speedup.** HOT means VRAM; WARM means host RAM (possibly pinned); COLD means model mapping/disk. A HOT cache miss that is WARM-backed still pays transfer/eviction/host scheduling. If WARM is cold, it pays file reads too. Pinning improved REAP-320 transfer cost 530–555 → 391–396 µs/expert and typically decode by 13–18%; it does not solve router readbacks or prefill expert-union churn. See [tiering diagnosis](AUDIT-REAP-DECODE.md#the-pinned-prefix-is-automatic-now-and-125-was-the-smallest-arm).
4. **A predictive prefetch mechanism is now wired but stays opt-in** (`KRK_MOE_PREDICT=1`) and routing remains authoritative. It combines per-layer temporal routing history with the preceding MoE layer’s current-token routes, gates candidates by measured precision, and only submits pinned WARM cache hints. This does not remove the per-layer router readback, does not predict from pre-attention activations, and is not proven faster yet. The historical next-layer previous-token prefetch is already measured as **9% slower** on REAP-320, so predictor is not made default.
5. **Qwen3.8-Flash-Next:** do not confuse names. Local **Qwen3.8-27B** GGUFs (`qwen35`) and **Qwen3.8 Distill-35B-A3B** (`qwen35moe`) are runnable arch families; local inspect verified the WebGGUF and GSQ-RCO 27B files as supported recurrent `qwen35`, carrying an MTP block, and the Distill-35B-A3B as supported recurrent MoE `qwen35moe`, also with MTP. But the user's **Qwen3.8 Flash-Next / REAP (`qwen4exp`) family is currently `ArchSupport::No`**, despite a substantial implementation that can execute HC, PLE, QSA, GDN and experts. Status is **experimental/refused pending a whole-graph coherence gate**, not “it works” by virtue of component tests. REAP-288’s recorded reference match is for a specific 24-token prompt, while another coding prompt diverged in recorded history. The actual Qwen4exp route mechanism can run if experimental override is used, but correctness coverage is insufficient to call this supported.
6. **DeepSeek-V4-Flash cannot run today.** `kraken-inspect` on the local 53.51 GiB file reports refusal for MLA/compressed-attention tensors and quant type #26. It has 43 blocks, 256 routed experts/top-6 plus shared, 1M context, CSA/HCA compression/indexing, hyper-connections, token-hash routing in initial layers, and one MTP block. This needs architecture/math support, not an expert-cache tweak. See [DeepInfer audit](AUDIT-deepinfer.md) and [MLA assessment](ASSESS-mla-deepseek.md).

## Evidence and code flow

### Expert weights: mapping → WARM → HOT → GEMM

`Engine::moe_ffn` first computes native router logits and top-k IDs/weights. It batches the selected expert IDs for a layer when its bounded prefetch path applies. The expert loop obtains a `ResidentExpert` from `ExpertCache::acquire()` and dispatches gate/up/down expert matrices through backend GEMM and scatter-add. The cache has:

- **HOT:** pooled device allocations subject to a byte/slot budget and eviction policy;
- **WARM:** host copies, pageable or pinned according to the measured pinned-prefix policy;
- **COLD:** the GGUF mapping/file, the canonical bytes.

A HOT hit is cheap. A WARM hit promotes bytes to device and can force another expert out. A COLD miss reads expert tensor slices, admits them to WARM when budgets allow, and promotes. These phases are tracked separately as `[stats] expert-path split prefill/decode` and `[stats] transfer`; remember that a COLD path may be visible under transfer/read statistics, not all expert-path timers. KV tier HOT/WARM/COLD is a separate cache and its end-of-run line must be read independently.

An important operational distinction: LFU/aging is a residency policy, not a forecast. Warmup rank files and scans are historical route frequency; a run can still have an all-first-touch working set. In Laguna XS.2, 4,590 distinct `(layer,expert)` keys meant every cold miss in the measured generation was compulsory: enlarging the cache to hold all of them removed no first-touch reads. See [PERF-ANALYSIS](PERF-ANALYSIS.md#b6--expert-slot-capacity-proven-not-a-bottleneck).

### Decode: what is proven expensive

**REAP-320:** with pinned WARM, 86.9 ms/step. Host-phase instrumentation (not the device recorder) split a step across 48 MoE layers as follows:

| Host phase | Per layer | Per 48-layer step | Meaning |
|---|---:|---:|---|
| `moe.route.fetch` | 0.868 ms | 41.7 ms | blocking readback of router row; includes draining queued GPU work |
| `moe.route.rest` | 0.820 ms | 39.4 ms | host plan-build and expert loop; the loop part was 33.7 ms/step |
| all else | — | ~11 ms | attention/QSA/HC/norms/dense GEMVs etc. |

Doubling the router readback costs only 0.093 ms extra, versus the initial 0.950 ms. Therefore most of its time is waiting for the prior GPU queue to drain before the host can see the router output, not the 320-float copy. After the second readback, the queue is empty and host plan construction takes ~0.13 ms/layer. This is a serial GPU→host→GPU dependency, not a bandwidth or GEMV deficit.

Device microbenchmarks: expert `gemm(gemv)` about 2.65 µs/call, 743 GB/s; per-expert host work and round trips dominate. The remedy with highest theoretical upside is device-side router top-k + a stable slot-addressed dispatch table with fills in graph dependencies (see [resident-cache audit](AUDIT-gpu-resident-expert-cache.md)). It is a larger redesign, and must not treat a missing/late expert as a zero contribution.

**Laguna XS.2:** a different model’s cache/read bound. Correctly ranked warmup reached 21.3 tok/s vs 10.1 without it on the recorded prompt, text identical, because one run’s routed set was already warm. Later four-way accounting attributed 84% of a full run to expert file reads, host→device copies, allocation, and room-making; event-based promotion ordering plus router barrier removal improved 110.70→96.25 ms/token (+15.6% tok/s) with identical output. Model-specific conclusions differ when the COLD tail is compulsory versus repeated WARM promotions.

**Prefill:** expect a different bottleneck from decode. The MoE part of a prompt layer uses the union of expert IDs for multiple rows; high row counts may activate a broad expert set, evicting rows before use. GEMM rows 2–8 use a decode ladder and the measured `gemv_m` path is poor at the Qwen4exp shape (29 GB/s); rows ≥9 enter the tiled GEMM rung and are not bit-exact to decode today. Attention and recurrent GDN also scale with rows/context, with separate dense/recurrent layer shapes. Profile with `KRK_TIME_HOST=1` and the expert phase split; use unprofiled wall-time A/B for conclusions because `KRK_TIME`’s device instrumentation adds ~20 µs per op and distorts long decode workloads.

### KV tier and long context

`KvTierCache` stores each layer’s actual KV plane geometry, with HOT VRAM, WARM host memory, and optional COLD files. It is not an MoE tier and it cannot make MLA/CSA work without those model’s cache semantics. D1? no; on this app the relevant behavior is entirely in engine storage. Read `[stats] kv tiered` and the explicit DROPPED-page fatal warning. A model with `--kv-hot-mb` below active working set can churn through promotion/migration; `--kv-cold-dir` is required if the tier needs durable spill. Qwen3.8 QSA requires its own pooled/indexer key cache and attention selection; the already-built QSA path is experimental as a complete model. DeepSeek-V4 needs CSA/HCA compressors, sparse indexer cache and new KV numerics, so ordinary per-layer KV tier support is not sufficient.

### Kernels and memory movement

For MoE, matrix GEMV microbenchmarks show the kernel can approach the hardware’s practical bandwidth. The expert working set remains a movement/scheduling problem. The current host route path serializes layers; pinned WARM removes the pageable bounce path on the measured small-BAR system but not the host route loop. `gemm_m` for small prefill-row counts, launch fan-out, pool reuse/readiness checks, copy stream ordering, and expert union admission all deserve measurement separately.

The dense kernel landscape is healthier: gfx1201 prefill uses the dp4a path (inline dot4 and TM=8); the special WMMA path is not automatically faster, and a wider tile previously regressed occupancy. Attention’s bit-exact prefill kernel improved measured prefill shape and preserves chunk invariance; the remaining rows≥9 GEMM tile rung is documented. Thus a GPU utilization number must be attributed by model: dense models often run 75–97% busy and are kernel-limited; large MoEs can sit <10% busy while moving experts or blocked on route readback.

## New opt-in route predictor

### What changed

`MoeRoutePredictor` (header-only) maintains per-layer recent selected expert IDs. For each decode token:

- `observe()` sees the native top-k set. On the next observation it scores pending temporal and cross-layer candidates against actual IDs and updates separate EWMAs.
- `predict()` considers (a) recency-weighted frequency of the target layer’s preceding route history and (b) the current token’s route at the preceding MoE layer. Policy 1/2/3 chooses temporal / adjacent / combined candidates; the candidate limit is bounded.
- The existing `stage_ahead()` only promotes already-WARM, pinned buffers and returns immediately if no pinned WARM is configured. It can never alter selected IDs, weights, logits or output.
- In Engine wiring, route prediction is enabled only when requested, the loaded model is MoE, and a pinned WARM source is actually configured. The legacy `KRK_MOE_NEXT_PREFETCH` arm is disabled while predictor runs, preventing duplicate staging. Generation resets history between requests.

Controls: `KRK_MOE_PREDICT=1`, `KRK_MOE_PREDICT_HISTORY` (2–16, default 4), `KRK_MOE_PREDICT_K` (0–128, default 2), `KRK_MOE_PREDICT_POLICY` (1 temporal, 2 cross-layer, 3 combined), `KRK_MOE_PREDICT_MIN_OVERLAP` (0–1, default 0.5). Stays off by default.

The empirical update is deliberately conservative. Previous-token next-layer prefetch on REAP-320 was 114.4 vs 124.7 ms/step (−9%) over three runs, despite identical text; prediction submitted work and did not manufacture link bandwidth. Any new candidate signal must show **useful promotions and end-to-end improvement** beyond noise. CLI stats now report prediction count, staged experts, candidate count and native-route hit precision for measured runs.

### Papers and systems checked; what they predict

The six user-supplied papers were checked against the question “does this forecast routed MoE expert IDs for weight prefetch?” Result: **none gives a validated general predictor for that purpose.** They address distinct signals:

| Reference | Actually predicts/reuses | Why it is not MoE-route prediction |
|---|---|---|
| [YOCO, arXiv:2405.05254](https://arxiv.org/abs/2405.05254) | reuses global KV via decoder-decoder architecture | architectural KV reuse; not MoE router IDs |
| [Qwen3.8-Omni, 2609.25611](https://arxiv.org/abs/2609.25611) | multimodal backbone/QSA context selection | attention context-block choices, not expert set |
| [SLAI T-Rex, 2607.20145](https://arxiv.org/abs/2607.20145) | training/post-training pipeline for V4 | expert parallelism/dispatch at training, no inference predictor |
| [Tree speculative decoding for V4, 2609.24698](https://arxiv.org/abs/2609.24698) | future token branches for target verification | token draft tree, not expert weights |
| [DeepSeek survey, 2507.09955](https://arxiv.org/abs/2507.09955) | survey, no new predictor | no evaluated prefetch mechanism |
| [DeepSeek V4.1-Flash, 2609.19969](https://arxiv.org/abs/2609.19969) | compressed/indexed historical context and speculative token verification | CSA2 selects attention context; DSpark predicts token acceptance, not routed experts |

The notable exception for model-specific exploitation is **DeepSeek-V4’s first few hash-routed MoE layers**: if hash expert ID is a deterministic function of input token ID, it is known earlier than a normal learned router’s output. This may permit exact prefetch for those layers after confirmatory source/fixture work. That is a narrow deterministic signal, not the general online predictor here. Qwen3.8 has a hybrid GDN/QSA backbone, but neither the recurrent state nor QSA block indices have been shown to forecast its separate MoE router IDs. Measure route correlations per layer/type before adding a specialized predictor.

Additional relevant papers supplied while this investigation was in progress are classified here so the attention/KV ideas are not mistaken for expert prediction:

| Reference | What it predicts/compresses | Potential value to Kraken | Not a route predictor because… |
|---|---|---|---|
| [Vegas / earlier SpecAttn, arXiv:2602.07223](https://arxiv.org/html/2602.07223v2) | Verification-guided sparse KV selection for self-speculative decoding | Possible future decode attention work; current paper name is Vegas, v1 was SpecAttn | selects context KV, not FFN expert IDs |
| [Cassandra, 2605.26558](https://arxiv.org/html/2605.26558v1) | Compact weight/KV representations for self-speculation | draft representation idea if speculation becomes memory-bound | no future router prediction; reported GPU performance is simulation-based |
| [SSV / earlier SpecSA, 2605.19893](https://arxiv.org/html/2605.19893v2) | Sparse verification query/block reuse | speculative decode kernel ideas, NVIDIA H100-specific today | selects attention blocks, not experts |
| [Strong Drafts Need Compact Memories, 2608.30252](https://arxiv.org/html/2608.30252v1) | Learned compact memory for a separate draft model | draft-side memory reduction, not target KV tier | requires draft/adaptor and does not predict expert IDs |
| [HybridGen, 2604.18529](https://arxiv.org/html/2604.18529v1) | CPU/GPU split attention; CPU scores older keys, GPU computes local keys | concept for a different host/GPU attention partition | relies on CPU compute and approximate next-layer query; not direct KV DMA or expert prediction |
| [SparDA, 2606.04511](https://arxiv.org/html/2606.04511v1) | Learned forecast of next-layer selected KV blocks, prefetched from pinned host RAM | closest scheduling analogy; measure HIP copy/compute overlap and forecast quality separately | forecast target is sparse attention KV blocks, not experts |
| [HGA, 2606.30709](https://arxiv.org/html/2606.30709v1); [2607.15105 fine-tuning paper](https://arxiv.org/html/2607.15105v2) | Hierarchical retrieval of selected real historical KV; 2607.15105 applies it to training | host RAM/NVMe KV tiering and bounded resident cache | sparse retrieval, not full-context exactness or router prediction; 2607.15105 is a training paper |
| [UltraQuant, 2606.20474](https://arxiv.org/html/2606.20474v3) | AMD-native 4-bit KV storage and attention | most directly relevant listed work for supported AMD/CDNA4 KV kernels | compresses in HBM; not host tiering and not expert prefetch |
| [RaZeR, 2501.04052](https://arxiv.org/html/2501.04052v3) | NVFP4 weight quantization via redundant-zero remapping | potential weight bandwidth/compression research | NVIDIA/Blackwell quantization, no KV tier or routing |
| [FLOW/FlexCiM, 2504.14365](https://arxiv.org/html/2504.14365v1) | layer-aware structured sparsity plus custom digital compute-in-memory | only a distant pruning/hardware reference | not a portable HIP kernel or prefetch runtime |

The user IDs do not always match the final arXiv titles: 2602.07223 v2 is **Vegas** (v1: SpecAttn), 2605.19893 v2 is **SSV** (v1: SpecSA), 2607.15105 is **Long-Context Fine-Tuning with Limited VRAM** and uses HGA; the original HGA paper is 2606.30709. These works should be compared as distinct attention selection, cache compression, draft-token or offload techniques, never relabeled as MoE expert predictor evidence.

Public reference comparison (no code copied):

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer) is a V4 serving reference for SM100/Ascend, distributed EP/EPLB and CSA/MLA; it does not document a portable single-GPU next-route expert predictor.
- [llama.cpp cache/preload RFCs and PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+expert+cache) include actual-route caches, heat/LRU proposals and transfer overlap experiments; distinguish closed/RFC proposals from merged behavior.
- [FreeToken](https://arxiv.org/abs/2608.16157) uses actual-route LRU plus bandwidth-aware CPU/GPU miss handling; prefill is treated separately with layer double buffering.
- [Strata](https://github.com/Niko1221/Strata) and [NInfer](https://github.com/Neroued/ninfer) document GPU hot experts / host experts and model-specific hybrid paths. MTP is token prediction, not proof of expert prediction. See [local Strata comparison](STRATA-VS-KRAKEN.md).

## Qwen3.8 and DeepSeek-V4 status: do not conflate variants

### Qwen3.8 names are multiple architectures

- **Qwen3.8-27B dense/recurrent `qwen35` GGUFs** inspected on this machine are runnable and carry one `blk.64` nextn prediction block, enabled via `--mtp-probe` / `--mtp-spec`. **Qwen3.8 Distill-35B-A3B (`qwen35moe`)** is also a runnable recurrent MoE family with its own MTP block. These are distinct from the Flash-Next expert-reduced target.
- **Qwen3.8 Flash-Next / REAP 288 / REAP 320 (`qwen4exp`)** is currently refused by architecture table unless experimental override. The engine now has HC, PLE, QSA and GDN paths; expert routing/cache shape already exists. Architecture table deliberately remains `No` pending whole-graph acceptance. One recorded REAP-288 run matches an oracle prompt 24/24; another coding prompt historically diverges. Component unit tests are not whole-model coherence proof. For user demand “must run”, the blocker is now **model correctness gate**, not merely loader presence.
- The additional local `/g/More-models/qwen3.5-flash-Q4.gguf` (renamed from `qwen3.8-flash-next-Q4.gguf`) is itself `qwen35 — supported (recurrent)` according to `kraken-inspect`; it is *not* `qwen4exp`. Exact local REAP metadata from [Qwen support inventory](AUDIT-qwen4exp-support.md): 288 experts/top-10, recurrent SSM layers + full attention every four layers, PLE and QSA, 4-lane HC. Large cache/prefill figures on REAP-320 are relevant as performance diagnostics, but they are not evidence that all qwen4exp checkpoints are coherent.

To call Qwen3.8 Flash-Next supported: build a fixed whole-graph output/hash fixture from an independently verified reference on the same GGUF/prompt/tokenization/template, compare an adequate continuation (include the known coding prompt and long enough tokens to catch the prior divergence), run CPU/device coherence where feasible, and pass chunk-invariance and per-component QSA/PLE/HC/GDN tests. For prefill/decode, the literature suggests an eventual separate KV path: SparDA-like next-layer block forecast or HGA-like hierarchical historical KV retrieval; both require a Qwen-specific quality/correctness gate. Do not feed QSA-selected KV indices into expert prefetch; expert routing is a separate router output. Until then, leave family experimental/refused instead of serving fluent wrong output.

### DeepSeek-V4-Flash

The exact local V4 Flash GGUF is 53.51 GiB payload and `deepseek4`; `kraken-inspect` refuses it for:

- MLA (`attn_kv_a`/`attn_kv_b`) and compressed attention/CSA/HCA;
- quant type #26 (also used for integer hash-route tables in this file);
- associated sparse indexer/cache and V4-specific attention operations.

The model also declares 256 experts x top-6 + shared, hyper-connection count 4/Sinkhorn 20, Yarn x16, sliding-window 128, hash-layer count 3 and one MTP block. Therefore it **does not run today**, and Qwen3.8 success would not imply DeepSeek support. `docs/AUDIT-deepinfer.md` and `docs/ASSESS-mla-deepseek.md` contain the source inventory and implementation estimates. Minimum credible staged plan:

1. Implement and test type-#26 tensor geometry and hash route table semantics (it is not a standard expert weight tensor).
2. Loader/arch row and CPU reference for MLA shared latent K/V, q-lora, rope-tail and grouped output projection; use small MLA model such as V2-Lite for first CPU oracle.
3. Add DeepSeek-V4’s distinct CSA/HCA rolling compression and sparse-indexer/LID cache; generic KV tier cannot substitute.
4. Implement V4 hyper-connections/Sinkhorn and hash-based initial routing; ensure the same routed expert set feeds normal expert cache.
5. Match Yarn and SWA semantics and build V4-specific attention + compressed KV formats. Keep `deepseek4` refused until complete whole-file coherence.
6. Only after correctness: evaluate its own MTP / DSpark adaptation and route-prefetch. MTP head being detected is not proof that target model loads.

**The current DeepSeek V4 task is too broad to safely solve by setting `ArchSupport::Yes`.** It is a multi-feature architecture port requiring a reference and new ops/kernels; staged partial loading without refusal risks plausible but incorrect text.

### What the additional supplied papers add

None of the additional works gives a DeepSeek-V4 expert-ID prefetch predictor. SparDA forecasts future **KV blocks** for sparse attention; HybridGen evaluates CPU/GPU **attention partitioning**; HGA retrieves selected historical KV; UltraQuant compresses KV in HBM on AMD CDNA4. The speculative-attention and compact-memory papers concern sparse attention or speculative **tokens**. None turns the V4 hash router into a validated prefetch path. For V4, hash IDs are a plausible narrow deterministic prefetch signal; prove exact ID and timing from the reference implementation before acting on it. The precise current titles and hardware limitations for all cited works are summarized above.

## Qwen3.8-27B performance spot check (2026-10-22)

The local file `/g/More-models/qwen3.5-flash-Q4.gguf` (formerly `qwen3.8-flash-next-Q4.gguf`) is the **dense/recurrent Qwen3.8-27B `qwen35` model**, not the qwen4exp expert-reduced Flash-Next. On RX 9070 XT/gfx1201, Release, `--bench --greedy --max-tokens 4 --ctx 512 --chunk 8`, the user-visible run was:

| Metric | Measured | Context |
|---|---:|---|
| Prefill | **6.8 tok/s** | 55 tokens / 8085.5 ms |
| Decode | **3.1 tok/s** | 4 generated tokens / 982.3 ms; 3 timed steps, 327.4 ms/step |
| Load | 10.68 s | 11.84 GiB uploaded of 15.61 GiB weights; startup 10.5 s |
| Device headroom | 3.41 GiB free before generation; 1 GiB reserved | 12.13 GiB model weights + 32 MiB KV + ~152 MiB workspaces |
| Weight path | 70 layer staging(s), 16.26 GiB from file, 7.034 s | drain 4.034 s, read 372 ms, copy 11 ms; 10 layers promoted resident (2.14 GiB) |

This is an alarming result and deserves immediate attention. But it is **not yet a clean kernel diagnosis**: the log does not print device kernel tables or per-phase host times, and `EngineUsage` reported all counters at zero even though ROCm copies and kernels completed, so that utility-counter source cannot be trusted on this setup. The engine's own weight-stage split is informative: the 7.034 s stage total is dominated by **4.034 s queue drains**, not file read or copy. That suggests repeated synchronization / serialization in the dense weight-stream path as a strong hypothesis—not a proven sole cause. Because model weights nearly fill VRAM, `--chunk 8` creates repeated layer staging; test `--weight-vram-mb` and chunk size with controlled runs, but don't change them blindly.

A second confound is card contention: load banner shows **12.666 GiB already held by the driver/other processes** and only 3.638 GiB free. It completed rather than OOMing, but this is not the same resource state as an otherwise idle RX 9070 XT. A four-token decode also has high variance and does not establish steady-state generation rate. It is a real recorded user-visible failure, not an apples-to-apples kernel benchmark.

Next discriminator: repeat at least 3 runs on an idle GPU with a fixed prompt/token hash and 32 generated tokens, alternate `--chunk 1/2/4/8/16` while collecting the final weight-stream line and the `[stats] host time` under `KRK_TIME_HOST=1`; then compare normal automatic weight residency vs a **measured** explicit `--weight-vram-mb` just under available memory. Test `--prefill-hash` for chunk-invariance and compare output hashes. The experiment must report queue-drain/read/copy split, load time separately, and resident/promoted layers. The previous command was too short and the card too occupied to conclude an optimization.

## Multi-model trace results (2026-10-22, sherlock trace mode)

Same short prompt ("The capital of France is"), greedy, `--trace` on. All three REAP/qwen4exp runs use `KRK_ARCH_ALLOW_EXPERIMENTAL=1` (arch still refused by default; traces are diagnostic, not a support claim).

| Model | Prefill | Decode | Dominant cost (measured) | Cache / residency |
|---|---|---|---|---|
| REAP-288 (`qwen3.8-flash-next-reap-288-Q4_K_M`, 78 GiB, 288x top-10, 48 layers) | 1.7 tok/s (5 tok / 2918 ms) | 1.4 tok/s (730 ms/step) | `moe-ffn` 4.90 s of 5.11 s `generate`; device `warm_read(pool)` 12.33 s + `upload_paged` 1.91 s + `upload` 1.50 s; `download_f32` backend 636 ms vs device 35 ms = router readback is host-wait, not kernel | routed set 42.2 GiB vs 256 MiB budget (1%); HOT 0.8%, WARM 99.2%, COLD 21.5%; promote 2853 ms (xfer 2612 ms), 2995 promotions / 2914 evictions — churning, not holding |
| REAP-320 (`Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI`, 62 GiB, 320 experts) | 2.6 tok/s (5 tok / 1933 ms) | 1.7 tok/s (591 ms/step) | `moe-ffn` 3.61 s of 3.71 s; device `warm_read(pool)` 12.69 s + `upload` 1.49 s + `upload_paged` 1.35 s | routed set 32.3 GiB vs 0.2 GiB (1%); HOT 2.6%, WARM 97.3%, COLD 2.5%; promote 2374 ms (prefill 1348 + decode 1027 ms), 2815 promotions / 2817 evictions |
| qwen3.5-flash-Q4 (dense `qwen35` 27B, 16 GiB) | 1.8 tok/s (5 tok / 2760 ms) | 2.3 tok/s (437 ms/step) | dense weight streaming: 56 stagings, 13.01 GiB, 3877 ms = drain 1036 + read 389 + copy 11 ms; device `upload` 8.52 s; layers 47..63 streamed, 10 promoted (2.14 GiB) | no MoE; KV flat in VRAM; bottleneck is streaming + queue drain, not experts |
| Distill-35B-A3B (`qwen35moe`, 12 GiB) | 9.3 tok/s (5 tok / 536 ms) | 5.9 tok/s (169 ms/step) | expert path NOT the blocker: promote 38 ms at 60.2 GB/s, COLD 0%; remaining ~169 ms/step is router fetch + GEMM/attention launches (see `moe-ffn` 1.65 s of 1.72 s `generate`) | HOT 25.6%, WARM 74.4%, COLD 0%; 2511 promotions / 2511 evictions, 0 MiB file reads — WARM holds the set |

Proven (not suspected):

- **REAP prefill/decode suck because the routed set (32–42 GiB) dwarfs the 256 MiB HOT budget.** HOT hits 0.8–2.6%; every token promotes ~500+ MiB and evicts as much. The trace device lane is topped by `warm_read` + `upload_paged/upload`, not GEMM. Fix direction: bigger HOT (card-fit permitting), `--expert-scan` + warmup index so WARM is pre-ranked (this REAP-288 run had no `.krakenexperts.json` and filled evenly + hit the 20 s warmup ceiling), and only then prefetch tuning. The predictor cannot help while candidates have nowhere resident to land.
- **Dense qwen3.5-flash sucks for a different reason: weight streaming.** 12.1 GiB weights vs ~12 GiB budget leaves layers 47..63 streaming every forward; the weight-stream line says drain (queue sync) >> read >> copy. Fix direction: `--weight-vram-mb` sized to the idle card, smaller `--chunk`, and the trace `wait/device-idle` lane to confirm drain vs kernel.
- **Router readback is host-bound on MoE decode.** REAP-288 `download_f32` backend 636 ms vs device 35 ms: the span is the blocking sync + host soft-max/plan, matching the earlier ~41.7 ms/layer REAP-320 autopsy. Device-side top-k + stable slot dispatch is the structural fix; GEMM tuning is not (GEMM rows are ~100–250 ms total against multi-second movement).
- **Distill-35B is the control that proves it:** same engine, same tracing, WARM-resident set → 9.3/5.9 tok/s with promote at 60 GB/s. Slow decode remainder (~169 ms/step) is launches + router + attention, not transfer.

Suspected / next to prove: per-layer route correlation for cross-layer prefetch (needs `KRK_TRACE_EXPERTS` + `tools/expert_trace.py` reuse-distance histograms on REAP, not assumed from Distill); whether REAP QSA selection and PLE add measurable prefill rows (trace has `gather_rows` 118–125 ms device — small vs movement, but needs a QSA on/off arm which does not exist yet); KV COLD spill behavior at long context (all runs here were ctx 128–256, KV flat).

## llama.cpp DeepSeek mapping (no code copied)

Research finding (researcher-web, URLs read: GGUF spec, DeepSeek-V3/V4 reports, llama.cpp V4 PR #24162, Together serving analysis, antirez V4 GGUF card, llama MoE offload guide):

- `attn_kv_a`/`attn_kv_b` refusal = tensor-layout/architecture mismatch, not a flag flip. MLA caches a compact latent + decoupled RoPE key; "absorb" folds key/value up-projections across attention so inference runs on the latent. V4 then adds CSA (stride-4 compressed + sparse/indexed), HCA (stride-128 dense over heavy compression), and a sliding-window/local path — separate cache streams with separate lifetimes, plus hash routing (`ffn_gate_tid2eid`) in the first layers.
- Quant type 26 in the current GGUF spec = `I32`, not a weight quant. Our V4 file's type-26 tensors are the integer hash-routing tables (host-side routing data, never a GEMM weight). Do not relabel 26 or reinterpret it as `IQ4_SQ` (a stale experimental enum). Handle as I32 tables feeding the hash-route path once the arch exists.
- llama.cpp placement knobs (GPU vs CPU experts, shared/attention on GPU) are placement, not a dynamic hot-expert cache that makes misses free. Our ExpertCache (LFU+aging, HOT/WARM/COLD, pinned prefix, `stage_ahead` hints) is the analogous policy layer — but it only helps *after* the MLA/CSA/HCA/hash graph exists. Plugging V4 tensors into the current MoE path without that graph = fluent wrong text.
- Staged DeepSeek plan stands (see DeepSeek-V4-Flash section below): I32/hash tables → MLA loader + CPU oracle (V2-Lite first) → absorbed latent decode kernel → CSA/HCA/indexer → Sinkhorn/HC + hash routing → Yarn/SWA + compressed KV. `deepseek4` stays refused until whole-file coherence.

## Performance tracing (sherlock-it)

New in this pass: a low-overhead Chrome-trace recorder (`include/krk/trace.hpp`, `KRK_TRACE=1` / `--trace [PATH]`, `KRK_TRACE_PATH`, `KRK_TRACE_MAX`, `KRK_TRACE_TIDS`). It writes `traceEvents` JSON for Perfetto / `chrome://tracing` with lanes:

- `host`: function/component spans (`Engine::forward`, `forward_core`, `moe_ffn`, `dense_ffn`, `gdn_forward`, `fetch_logits`, `generate`, plus every `KRK_TIME_HOST` phase which now also emits a trace span).
- `backend`: per-op host wall spans with `op`, `grid`, `block`, `bytes`, `flops` args (grid/block filled where the op reports launch geometry).
- `device`: event-measured kernel durations plus `wait/device-idle` gaps. Durations are exact HIP-event measurements; placement is resolve-anchored, so use host/backend spans for ordering and the device lane for cost ranking.
- `transfer`: `upload` H2D / `download` D2H spans with byte counts; `wait`: `device-sync` / `event-sync`; `memory`: alloc spans; `counter`: resource signals.

Overhead rule (same as `--profile`): tracing perturbs the run. Capture a bounded window (bench, not a full session), keep the JSON cap in mind, and always compare an untraced wall-time A/B before claiming a win. The old tables (`KRK_TIME`, `--profile`, host time, expert-path split, weight-stream drain/read/copy) remain the numeric source of truth; the trace is the timeline that says *where* those numbers come from.

Usage: `kraken -m /g/More-models/qwen3.5-flash-Q4.gguf --bench --trace trace.json` then open `trace.json` in `https://ui.perfetto.dev`. Start with the 6.8 tok/s prefill / 3.1 tok/s decode bench above: the weight-stream `drain` vs `read` vs `copy` split plus the trace's `wait/device-idle` lane is the discriminator between queue-drain serialization and kernel cost.

## DeepInfer adaptation boundary (honest)

Adapting DeepInfer's model loading/inference directly is not portable here: DeepInfer targets NVIDIA SM100 (CUDA 13) and Ascend 950 (CANN 9.2) with FlashMLA/DeepGEMM/DeepEP/TileKernels, and its public repo states the KV-cache system and disaggregated serving pieces are not fully open-sourced. There is no HIP/RDNA4 path to copy, and copying CUDA kernels would not run on gfx1201. What transfers is the *design shape* (absorbed latent KV, CSA2 Full/Reindex/Reuse policy, indexer top-k, bounded SWA replay, FP4/MXFP4 KV numerics), staged behind the MLA + type-#26 + CSA/HCA gaps above. Keep `deepseek4` refused until the staged gates pass; do not flip `ArchSupport::Yes` to make the file "load". See `docs/AUDIT-deepinfer.md` and `docs/ASSESS-mla-deepseek.md`.

LMCache ports as a chunked prefix-reuse + HOT/WARM/COLD movement pattern (not its engine connectors); YOCO's shared global KV is a model-architecture property, not a cache-manager retrofit for existing GGUF decoders. Researcher notes are filed in the conversation record; mechanisms only, no code copied.

## Ranked next actions

| Priority | Work | Why / evidence | Acceptance gate |
|---|---|---|---|
| P0 | Complete Qwen3.8 Flash-Next whole-model correctness | components mostly exist; arch remains experimental after a known prompt divergence | whole-graph oracle match on REAP-288/320 and Qwen3.8 target prompts, chunk 1/2/8, CPU/device where time permits |
| P0 | Fix/measure small-row prefill GEMM + MoE union staging | Qwen4exp recorded 197 ms/prompt row and `gemv_m` 29 GB/s | paired fixed prompt/chunk A/B, same route IDs/text hash, per-phase cache stats and representative `--chunk` values |
| P1 | Add router-on-device / stable slot dispatch for decode | REAP router fetch ~41.7 ms/step and host plan/loop ~33.7 ms | device-vs-host top-k IDs/weights correctness, never-drop invariant, identical full output, end-to-end gain on oversized REAP and fitting control |
| P1 | Validate predictor in route-trace replay then live A/B | predictor is an unproven opt-in hint; legacy predictor lost 9% | offline top-k-set precision/recall by layer/type; live 5+ interleaved pairs, candidate/staged/hit stats, transfer bytes, output hashes; disable if not beyond MAD |
| P1 | DeepSeek-V4 staged port | hard loader gaps and #26; no model run now | type tests + V2-Lite CPU oracle + V4 per-op and graph fixture; stay refused until all gates |
| P2 | Revisit multi-expert GEMV/launch aggregation only after route sync | kernel ~2.65 µs; route sync dominates | prove output exact and compare end-to-end; isolated kernel win alone insufficient |
| P2 | Keep KV and expert cache budgets separately measured | two HOT budgets compete for same VRAM; long context can lower expert residency | log both budgets and migrations, same model/context A/B, no KV dropped pages |

## Verification completed in this continuation

- `ninja -C build-hip kraken kraken-tests`: **rc=0** after correcting a malformed line break in `src/engine.cpp`; compiler emits existing deprecation/shadow/unused warnings.
- `kraken-tests --run-group test_moe_route_predictor`: **9/9 checks, rc=0**.
- `git diff --check`: clean.
- `kraken-inspect` `/g/More-models/Qwen3.8-27B-WebGGUF-Q4_0.gguf`: `qwen35 — supported (recurrent)`, one MTP block.
- `kraken-inspect` `/g/More-models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`: `qwen35 — supported (recurrent)`, one MTP block.
- `kraken-inspect` `/g/More-models/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf`: `qwen35moe — supported (recurrent-moe)`, one MTP block.
- `kraken-inspect` `/g/More-models/qwen3.5-flash-Q4.gguf` (renamed from `qwen3.8-flash-next-Q4.gguf`): `qwen35 — supported (recurrent)`; this is **not** the experimental qwen4exp family.
- `kraken-inspect` `/g/More-models/Qwen3.8-27B-WebGGUF-Q4_0.gguf`: `qwen35 — supported (recurrent)`, one MTP block (re-verified after rename).
- `kraken-inspect` local DeepSeek-V4-Flash GGUF: refused as above.
- Test groups on CPU fixtures: `test_qwen4exp_forward` 19/19 and `test_gdn_generation` 21/21; these prove component/fixture behavior, not real-checkpoint coherence.
- User-visible Qwen3.8-27B spot run: 6.8 tok/s prompt, 3.1 tok/s decode, 10.68 s load; 12.666 GiB was already held by other GPU clients and weight staging spent 4.034 s draining the queue. This is recorded as a severe symptom; a clean idle-GPU reproduction with longer decode remains necessary.
- Predictor smoke on supported Qwen3.8 Distill-35B-A3B: with only 512 MiB pinned WARM, 16 tokens produced 585 hints but 0 experts staged; predictor candidates had 55.6% precision at min-overlap 0, but the model was COLD-read/first-touch dominated, so it could not measure an effective staged predictor. At min-overlap 0.5: 435 hint sets, 0 staged, 54.6% hits. On one prompt, pinned WARM 8 GiB and min-overlap 0 gave 585 hints, 0 staged, 55.6% precision, 16 tokens at 11.1 tok/s; this config is not an off/on A/B and must not be cited as a speedup. It proves the predictor telemetry runs and also exposes that candidates did not produce promotions in this run.

Full suite passed **3592/3592**, rc=0 (re-verified after the trace wiring).

## Tracing verification (2026-10-22, this pass)

- `ninja -C build-hip kraken kraken-tests`: **rc=0**; `kraken-tests`: **3592/3592, rc=0**; `git diff --check` clean.
- `kraken -m models/SmolLM2-135M-Instruct.Q4_K_M.gguf -p "..." -n 8 --ctx 128 --greedy --trace trace.json`: **rc=0**, 277.8 tok/s prefill / 476.6 tok/s decode (small control model), `trace.json` parses as valid trace-event JSON with **6069 events**: `backend` 2454, `device` 2408, `memory` 289, `transfer` 280, `component` 265, `host` 23, `wait` 347. Top device ops `gemm(gemv)` 512 / `gemm_group` 322 / `upload` 272 — the dispatch + memory + sync traps all firing.
- `kraken-inspect /g/More-models/qwen3.5-flash-Q4.gguf`: `qwen35 — supported (recurrent)`, one MTP block (rename verified). DeepSeek-V4-Flash GGUF: still `deepseek4 — not supported / refused` (MLA + compressed attention + type #26), so no DeepInfer behavior was ported as code — boundary documented, not bypassed. The baseline comparison reports the new `test_moe_route_predictor` group (9 checks) absent from the 73-group baseline and therefore fails suite-shape policy; this requires an intentional baseline re-record before that gate can be green. A new group was not silently allowed. The controlled multi-repetition predictor A/B, clean-device Qwen3.8-27B profile, Qwen4exp real-checkpoint coherence, and DeepSeek port were not completed. The previous-turn local model timings and reference-output claims are documented here as existing measurements, not re-measured today. The route predictor has not yet been benchmarked on an actual model and must remain opt-in pending that experiment.
