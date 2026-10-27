# Switches, knobs and what they are worth

Every option Kraken accepts, in one place, with the measured effect where there
is one. The point is not to list flags — `--help` does that — but to say **which
ones are worth turning, on which arm, and by how much.**

Counted from the source on this commit: **66 CLI flags** and **200+ environment
variables** (the appendix at the end is generated from the tree, so it cannot
drift). Only a handful of those decide throughput; most are diagnostics,
single-kernel probes, or experiment knobs that exist to reproduce one
investigation and are not meant to be combined.

The three device arms, referred to throughout:

| arm | what runs where | status on this box |
|---|---|---|
| `--cpu` | scalar reference everywhere; `KRK_CPU_DOT_AVX2=1` swaps in the AVX2 block-format dot | works, **baseline** |
| `--npu-cpu` | the whole graph on the virtual NPU, MoE routing and KV on the host | works, needs a VNPU build |
| `--npu` | only the GEMMs on the virtual NPU, everything else on the CPU | works, needs a VNPU build |
| `--device hip` | GPU | **not measured here — no GPU in this sandbox (HIP off).** Numbers for the RX 9070 XT are in [MODEL-STATUS.md](MODEL-STATUS.md) and [GPU_BRINGUP.md](GPU_BRINGUP.md). Nothing in this file invents a GPU figure. |

All measurements below are the 2 GiB q8_0 MoE fixture of
[MOE-RESIDENCY.md](MOE-RESIDENCY.md), 88- or 121-token prompt as noted.
Full detail and the residency ladder: that file.

---

## 1. The flags that decide throughput

Measured on the MoE fixture, `--npu-cpu`/`--cpu` as stated, all experts resident
unless the row says otherwise.

| flag / opt-in | effect | measured |
|---|---|---|
| `KRK_CPU_DOT_AVX2=1` | AVX2 block-format dot, bit-identical to `KRK_CPU_DOT_FAST` | prefill **10342.9 → 2940.9 ms (3.5×)**, decode **134.2 → 42.0 ms (3.2×)**. On the general `--cpu` q8_0 path (121-token prompt): 8702.8 → 3947.6 ms, **2.20×** |
| `--npu-cpu` + `KRK_NPU_W8A8=1` + `KRK_NPU_ATTN=1` | whole graph on VNPU, int8 weights, device attention | decode **32.8 ms/step (30.5 tok/s)** — the fastest decode measured here; prefill 3557.5 ms, slower than AVX2 CPU |
| `KRK_NPU_W8A8=1` | int8 weights on the device | resident: decode 113.6 → 34.8 ms (**3.3×**). **Streaming: 149.0 → 718.7 ms (4.8× WORSE)** — see [MOE-RESIDENCY.md](MOE-RESIDENCY.md) §5 |
| `KRK_NPU_ATTN=1` | attention on the device | 34.8 → 32.8 ms decode, 3597.9 → 3557.5 ms prefill (small, additive to w8a8) |
| `--expert-cache-mb N` | device expert budget | the decode lever: 41.1 ms (resident) → 73.6 ms (64 MiB) → 74.6 ms (16 MiB). Prefill unmoved (3.82–3.94 s) |
| `--expert-warm-mb N` | host-RAM WARM tier for evicted experts | read **2519 → 246 MiB (0.10×)** at a 64 MiB device budget; decode 73.6 → 70.5 ms |
| `--chunk N` | prefill batch | 8 → 4739.3 ms and 3907 MiB read; **128 → 3858.4 ms and 1372 MiB**. Below 128 the gathers repeat. (Pre-batched-GEMM numbers; the ratio holds, the absolute times are now ~4x lower) |
| `--threads N` | host workers | 1 vs 2: prefill **5307.9 vs 2940.9 ms (1.80×)**, decode **73.8 vs 73.6 ms (nothing)** |
| `--kv-quant int8` | int8 KV cache | on this fixture **costs** 8% of prefill (3185.3 vs 2940.9 ms) and saves nothing: KV is 4×64 over 20 layers. Worth it only when the cache is large relative to the model |
| `--hybrid-experts 1` | overflow experts computed on the host out of the mapping | 73.6 → 72.7 ms decode at 64 MiB (1.2%) |
| `--router-resident P` | let a near-tied pick land on an already-resident expert | **no effect at 64 MiB** (0.0% HOT): there is no resident tier to prefer. It reorders picks among resident experts |
| `--vram-tier`, `--ram-tier`, `--host-ram-mb`, `--vram-cap-mb` | plan for a smaller machine than the installed one | planning only, clamped to what is installed; no throughput claim here (GPU arms unmeasured) |
| `--cpu` | scalar reference | 3.5× slower than AVX2 on prefill, by design. It is the accuracy witness, not the fast path |
| *(no flag)* the batched prefill GEMM | decodes each weight block once per 8 rows instead of once per row | **2.28–3.35×** on the q8_0 MoE, **3.83–4.29×** on f16 dense. Always on; bit-identical to the old path ([MOE-RESIDENCY.md §7](MOE-RESIDENCY.md)) |

## 2. Correctness and determinism opt-ins

These change results on purpose. Each has its own witness; none is a silent
default.

| switch | what it changes | note |
|---|---|---|
| `KRK_CPU_DOT_AVX2=1` | dot-product summation order | **bit-identical** to `KRK_CPU_DOT_FAST` (`memcmp`), different from exact. Prefill hashes: exact `2985aa5d98efe8c3`, fast and avx2 both `18459b50a9493c79` |
| `KRK_CPU_DOT_FAST=1` | summation order | as above; superseded by AVX2 on a machine that has it |
| `KRK_NPU_W8A8=1` | int8 weights vs f32 | not bit-identical to f32; falls back per GEMM and counts the fallbacks in `[stats] npu` |
| `KRK_NPU_ATTN=1` | attention on the device | not bit-identical to the host kernel |
| `--kv-quant int8` | KV storage | quantises the cache |
| `KRK_ARCH_ALLOW_EXPERIMENTAL=1` | lets the tensors of a **refused** schema be read, so a half-ported architecture can be checked stage by stage | a *loader* switch only: it does not admit the graph. An arch whose attention has no implementation is still refused after the load and before generation, because a run would otherwise emit fluent text that is not the model |
| `--greedy` | sampling | temperature 0; what every benchmark here uses |

Determinism, measured: the prefill hash is invariant across expert residency on
the **CPU** arms (`9b57818ff075a4f4` exact, `b5bc761065e6437c` AVX2, identical at
512 MiB and 64 MiB) and **not** invariant on `--npu-cpu`
(`2468164ac1d2f54a` vs `f6cb507f7cd1e725`) because the staging cache declines
different weights at different residencies. Compare hashes within an arm.

## 3. Expert routing and cache knobs

The full set, grouped by what they control. Effects quoted are from
[MOE-RESIDENCY.md](MOE-RESIDENCY.md) and the audit docs beside it.

- **Residency**: `--expert-cache-mb`, `--expert-cache-slots`,
  `--expert-warm-mb`, `--expert-pin-mb`, `--expert-l2-mb`, `--expert-retain`,
  `KRK_EXPERT_ARENA`, `KRK_EXPERT_INPLACE`, `KRK_EXPERT_PIN`, `KRK_EXPERT_PIN_MB`,
  `KRK_EXPERT_RETAIN`, `KRK_EXPERT_PRELOAD`.
- **Warm-up at load**: `--expert-warmup`, `--expert-warmup-ms`,
  `--expert-warm-prefetch`, `KRK_EXPERT_WARMUP`, `KRK_EXPERT_WARM_MAP`.
- **Look-ahead / prediction**: `KRK_EXPERT_LOOKAHEAD`, `KRK_MOE_LOOKAHEAD`,
  `KRK_MOE_NEXT_PREFETCH`, `KRK_MOE_PREDICT`, `KRK_MOE_PREDICT_K`,
  `KRK_MOE_PREDICT_POLICY`, `KRK_MOE_PREDICT_HISTORY`,
  `KRK_MOE_PREDICT_MIN_OVERLAP`, `KRK_WARM_AHEAD`, `KRK_WEIGHT_AHEAD`.
- **Push/pull mechanics**: `KRK_PULL_THREADS`, `KRK_PULL_CHUNK_KIB`,
  `KRK_WARM_READERS`, `KRK_WARM_CHUNK`, `KRK_WARM_DMA`, `KRK_WARM_STAGE_KIB`,
  `KRK_WEIGHT_PULL`, `KRK_WEIGHT_PROMOTE`, `KRK_WEIGHT_STREAM`,
  `KRK_PROMOTE_WAIT`, `KRK_EVICT_WINDOW`.
- **Batching and grouping**: `KRK_MOE_BATCH`, `KRK_MOE_BATCH_PROMOTE`,
  `KRK_MOE_GROUP`, `KRK_MOE_FANOUT`, `KRK_MOE_TILE_DECODE`,
  `KRK_MOE_TILE_CHUNK`, `KRK_MOE_SILU_FUSED`.
- **Router behaviour**: `--router-resident`, `KRK_ROUTER_2X`,
  `KRK_ROUTER_TIE_MARGIN`, `KRK_ROUTE_FAKE`, `KRK_ROUTE_FASTCOPY`,
  `KRK_TOPK_SYNC`, `--debug-topk`, `KRK_DUMP_ROUTE`.
- **Oracles** (print, do not change): `KRK_DUMP`, `KRK_DUMP_MOE`,
  `KRK_DUMP_FULL`, `KRK_DUMP_ROW`, `KRK_TRACE_EXPERTS`, `--expert-scan`,
  `--expert-stub`.

## 4. Instrumentation

| switch | what it prints |
|---|---|
| `--profile`, `KRK_PROFILE=1` | per-phase wall clock and per-op components |
| `KRK_STAGE_TIMING`, `KRK_STAGE_SYNC`, `KRK_STAGE_WAIT` | the stage/pipeline split |
| `--profile-log FILE`, `KRK_PROFILE_LOG`, `KRK_PROFILE_FROM`, `KRK_PROFILE_STEPS` | machine-readable trace |
| `--trace`, `KRK_TRACE`, `KRK_TRACE_PATH`, `KRK_TRACE_STAGES`, `KRK_TRACE_MAX` | event trace |
| `--bench`, `--prefill-hash` | the two gates: throughput, and "did the maths change" |
| `KRK_VRAM_TRACE`, `KRK_TIER_TRACE`, `KRK_KERNEL_INFO`, `KRK_OP_SHAPES` | tiering and kernel identity |
| `--info`, `kraken-inspect FILE` | metadata/verdict without loading (inspect is the one to reach for first) |

## 5. Things that look like options but are not

- `--expert-cache-mb 0` does **not** mean "no cache". It means auto, which
  resolves to 512 MiB.
- `--npu` and `--npu-cpu` are different arms, not a strength setting: `--npu`
  offloads GEMMs only, `--npu-cpu` puts the whole graph on the device.
- `--cpu` is not "the fast path with the NPU disabled"; it is the scalar
  reference. To get the fast CPU path, set `KRK_CPU_DOT_AVX2=1`.
- `--vram-tier` cannot make the machine bigger, only smaller.

---

## Appendix: every `KRK_*`/`VNPU_*` variable, by the file that owns it

Generated from the tree (`git grep`) so it lists what the code actually reads.
A variable appearing here means this build consults it on some path, not that the
path runs.

| file | count | variables |
|---|---:|---|
| `include/krk/expert_cache.hpp` | 1 | `KRK_EXPERT_PIN_HITS` |
| `include/krk/h2d_probe.hpp` | 2 | `KRK_H2D_CACHE`, `KRK_H2D_PROBE_ITERS` |
| `include/krk/host_time.hpp` | 4 | `KRK_PROFILE`, `KRK_PROFILE_LOG`, `KRK_TIME`, `KRK_TIME_HOST` |
| `include/krk/trace.hpp` | 4 | `KRK_TRACE`, `KRK_TRACE_MAX`, `KRK_TRACE_PATH`, `KRK_TRACE_TIDS` |
| `include/krk/vram_guard.hpp` | 5 | `KRK_MEM_FRACTION`, `KRK_RAM_RESERVE_MB`, `KRK_VRAM_FLOOR_MB`, `KRK_VRAM_RESERVE_MB`, `KRK_VRAM_TOTAL_MB` |
| `src/dflash.cpp` | 21 | `KRK_DFLASH_BLOCK_POS`, `KRK_DFLASH_CAUSAL`, `KRK_DFLASH_DEBUG`, `KRK_DFLASH_DELTA_SCALE`, `KRK_DFLASH_FFN_GROUP`, `KRK_DFLASH_GATE`, `KRK_DFLASH_KVNORM`, `KRK_DFLASH_KV_ZERO`, `KRK_DFLASH_LAYER_OFFSET`, `KRK_DFLASH_PROBE`, `KRK_DFLASH_QKV_GROUP`, `KRK_DFLASH_READOUT_BASE`, `KRK_DFLASH_READOUT_NORM`, `KRK_DFLASH_READOUT_ONLY_BASE`, `KRK_DFLASH_RES_TRACE`, `KRK_DFLASH_ROPE_NEOX`, `KRK_DFLASH_ROW0_FEATURE`, `KRK_DFLASH_SELF_MASK`, `KRK_DFLASH_WATCH`, `KRK_DFLASH_WINDOW`, `KRK_DFLASH_ZERO_MASK` |
| `src/engine.cpp` | 79 | `KRK_ACTIVE_STAGE`, `KRK_DEQUANT_PROBE`, `KRK_DEVICE_SOFTMAX`, `KRK_DEV_PLAN`, `KRK_DEV_PTR`, `KRK_DFLASH_ALLOW_BAD_MASK`, `KRK_DUMP`, `KRK_DUMP_FULL`, `KRK_DUMP_MOE`, `KRK_DUMP_ROUTE`, `KRK_DUMP_ROW`, `KRK_EXPERT_INPLACE`, `KRK_EXPERT_LOOKAHEAD`, `KRK_EXPERT_PIN`, `KRK_EXPERT_PIN_MB`, `KRK_EXPERT_PRELOAD`, `KRK_EXPERT_WARMUP`, `KRK_FFN_ROWSCALE`, `KRK_FUSED_ATTN_NEOX`, `KRK_GDN_GATE`, `KRK_GDN_GROUP`, `KRK_H2D_PROBE`, `KRK_H2D_PROBE_RERUN`, `KRK_HOST_PIN_MB`, `KRK_LOGITS_SYNC`, `KRK_MOE_BATCH`, `KRK_MOE_BATCH_PROMOTE`, `KRK_MOE_GROUP`, `KRK_MOE_LOOKAHEAD`, `KRK_MOE_NEXT_PREFETCH`, `KRK_MOE_PLAN`, `KRK_MOE_PREDICT`, `KRK_MOE_PREDICT_HISTORY`, `KRK_MOE_PREDICT_K`, `KRK_MOE_PREDICT_MIN_OVERLAP`, `KRK_MOE_PREDICT_POLICY`, `KRK_MOE_SILU_FUSED`, `KRK_MOE_TILE_CHUNK`, `KRK_MOE_TILE_DECODE`, `KRK_MTP_ABL`, `KRK_MTP_CAPCHECK`, `KRK_MTP_CONCAT`, `KRK_MTP_FRAC`, `KRK_MTP_HSHIFT`, `KRK_MTP_PIN`, `KRK_MTP_POSSCALE`, `KRK_MTP_PROBE_ALL`, `KRK_MTP_PROP_SPLIT`, `KRK_MTP_REPLAY_IN`, `KRK_MTP_REPLAY_OUT`, `KRK_MTP_ROPE`, `KRK_MTP_ROPEOFF`, `KRK_MTP_ROWFID`, `KRK_MTP_WINDOW`, `KRK_PIN_AUTO`, `KRK_PROFILE_SNAPSHOT`, `KRK_QSA`, `KRK_RECLAMP`, `KRK_ROUTER_2X`, `KRK_ROUTER_SYNC`, `KRK_ROUTER_TIE_MARGIN`, `KRK_ROUTE_FAKE`, `KRK_SPEC_REALIGN`, `KRK_SPEC_ROW_SYNC`, `KRK_SPEC_RS`, `KRK_SPEC_SAMPLED_ACC`, `KRK_SPEC_TRACE`, `KRK_SPLIT`, `KRK_SWA_OFF`, `KRK_SYNC_EXPERT`, `KRK_TOPK_SYNC`, `KRK_TRACE_STAGES`, `KRK_VRAM_SEED_PCT`, `KRK_VRAM_TRACE`, `KRK_WARM_AHEAD`, `KRK_WARM_DMA`, `KRK_WARM_PREFILL`, `KRK_WEIGHT_STREAM`, `KRK_WEIGHT_VRAM_MB` |
| `src/expert_cache.cpp` | 5 | `KRK_EVICT_WINDOW`, `KRK_EXPERT_WARM_MAP`, `KRK_POOL_MARK_BURST`, `KRK_TRACE_EXPERTS`, `KRK_WARM_CHUNK` |
| `src/hip/backend_hip.hip` | 43 | `KRK_ACC_GEMM`, `KRK_ACC_REPEAT`, `KRK_ATTN_BEXACT`, `KRK_ATTN_SPLIT`, `KRK_ATTN_WMMA`, `KRK_CU_OVERRIDE`, `KRK_EXPERT_ARENA`, `KRK_FUSED_ATTN`, `KRK_FUSED_LAYER`, `KRK_FUSED_SILU`, `KRK_FUSE_NORM`, `KRK_GEMM_ARM_QUIET`, `KRK_GEMM_EXACT`, `KRK_GEMM_INT8`, `KRK_GEMM_MMA_ONLY`, `KRK_GEMM_PATH`, `KRK_GEMM_SPLIT`, `KRK_GEMM_STAGE_ONLY`, `KRK_GEMV_M`, `KRK_GEMV_MW`, `KRK_GRID_G`, `KRK_MOE_FANOUT`, `KRK_OP_SHAPES`, `KRK_PINNED_COALESCE`, `KRK_POOL_BURST`, `KRK_POOL_HOST_WAIT`, `KRK_POOL_NO_WAIT`, `KRK_POOL_SLOT_MARK`, `KRK_PROMOTE_WAIT`, `KRK_PULL_CHUNK_KIB`, `KRK_PULL_THREADS`, `KRK_RESIDUAL_SCALE`, `KRK_ROUTE_FASTCOPY`, `KRK_STAGE_SYNC`, `KRK_STAGE_TIMING`, `KRK_STAGE_WAIT`, `KRK_TEST_POOL_NO_EVENT`, `KRK_WARM_READERS`, `KRK_WARM_STAGE_KIB`, `KRK_WEIGHT_DMA`, `KRK_WEIGHT_PULL`, `KRK_WMMA_MB`, `KRK_WMMA_NF` |
| `src/hip/kernels/attention.hpp` | 2 | `KRK_ATTN_BEXACT_QT`, `KRK_ATTN_BEXACT_TILE` |
| `src/hip/kernels/gemm_dp4a.hpp` | 1 | `KRK_KERNEL_INFO` |
| `src/hip/kernels/gemm_mmq.hpp` | 1 | `KRK_DP4A_MMQ` |
| `src/hip/op_time.hpp` | 4 | `KRK_PROFILE_FROM`, `KRK_PROFILE_OPS`, `KRK_PROFILE_STEPS`, `KRK_TIME_POOL` |
| `src/kv_tier.cpp` | 3 | `KRK_TIER_FENCE`, `KRK_TIER_TRACE`, `KRK_TIER_WAIT_S` |
| `src/main_cli.cpp` | 5 | `KRK_EXPERT_RETAIN`, `KRK_MTP_PROBE`, `KRK_MTP_SPEC`, `KRK_PHASE`, `KRK_RANGE_STRICT` |
| `src/main_server.cpp` | 1 | `KRK_PORT` |
| `src/model.cpp` | 9 | `KRK_ARCH_ALLOW_EXPERIMENTAL`, `KRK_MTP_LOAD`, `KRK_ROPE_NEOX`, `KRK_WEIGHT_AHEAD`, `KRK_WEIGHT_HOST_CACHE`, `KRK_WEIGHT_HOST_CACHE_MB`, `KRK_WEIGHT_PREFETCH`, `KRK_WEIGHT_PROMOTE`, `KRK_WEIGHT_SLAB` |
| `src/npu/npu_offload.cpp` | 10 | `KRK_NPU_ATTN`, `KRK_NPU_ATTN_MIN_TOK`, `KRK_NPU_MIN_ELEMS`, `KRK_NPU_PIN_THREADS`, `KRK_NPU_STRICT_QUEUE`, `KRK_NPU_THREADS`, `KRK_NPU_TRACE`, `KRK_NPU_UNIT`, `KRK_NPU_W8A8`, `KRK_NPU_WEIGHT_MB` |
| `src/par_pool.hpp` | 1 | `KRK_THREADS` |
| `src/quant.cpp` | 2 | `KRK_CPU_DOT_AVX2`, `KRK_CPU_DOT_FAST` |
