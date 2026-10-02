# KRAKEN — pending work and open issues

Live list of what is unfinished, what is broken, and what is blocked. Ordered
by severity, not by age. Last updated after commit `7337f3b`.

Status of each area is in [docs/STATUS.md](STATUS.md); this file is the
"what still needs doing" view of the same project.

---

## P0 — correctness

### The qwen35 GPU path is not reproducible (P0 GATE NOT CLOSED)

`kraken --model Qwen3.5-9B-Q4_K_M.gguf --prompt "The capital of France is"`
on the 6700 XT produces a different 16-token greedy completion on almost every
run. The llama.cpp reference for that prompt is exactly:

```
' Paris.\nThe capital of France is Paris.\nThe capital of France is'
```

Observed kraken GPU variants include that exact string (1 run in ~5), a
near-miss that lowercases one word, `What is the capital of France?`, and
purely degenerate loops (`The|The|The`, `|-|-|-`). The **CPU f32 backend
reproduces the reference byte for byte**, so the model, the tokenizer, the
GDN math and the sampler are all correct — this is specific to the HIP path.

**What has been ruled out** (each measured, not assumed):

| Hypothesis | Result |
|---|---|
| Float atomics / split-K reductions | No `atomicAdd` anywhere in `src/hip/` |
| Multiple streams / missing sync | Single default stream; `sync()` is `hipDeviceSynchronize()` |
| VRAM pressure | Killed three stale processes holding 6.44 GiB; VRAM down to 86 MiB, still varies |
| Filesystem / mmap (model on NTFS vs tmpfs) | Copied to `/tmp` (tmpfs); same md5s from both mounts, still varies |
| The quantizer | Dense SmolLM2-135M **Q4_K_M** on the same build: 4/4 byte-identical |
| The sampler | `--greedy --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --seed 0` still varies |
| Uninitialized device memory | `KRK_ZERO_ALLOC` memset on every `alloc()` did not stabilize it |
| Kernel serialization flags | `HIP_LAUNCH_BLOCKING`, `AMD_SERIALIZE_KERNEL`, `AMD_SERIALIZE_MEMORY` all still vary |
| compute-sanitizer | **Not installed on maclin** — cannot get an authoritative race report |

**What the new `--debug-topk` flag established** — this is the key finding.
The divergence is *drift present in the very first step*, not a race that
accumulates over tokens:

```
run A: step 0 pos 5  11751=16.437500 (-0.194323)  198=13.750000 (-2.881823)
run B: step 0 pos 5  11751=15.960938 (-0.187116)  198=13.203125 (-2.944928)
```

Step 0 is the last prefill row — no generated token has been fed back yet. The
top-1 logit differs by ~0.5 and the runner-up by ~0.5, with **zero NaN and
zero Inf**. A same-binary, same-input run that disagrees on its *first*
forward pass is reading memory that differs between processes, or running a
kernel whose result depends on uninitialised or mis-launched state.

The CPU f32 path over the same prompt and the same build is bit-identical
step for step, which is the control that makes this a GPU-path bug.

**Next steps, in order of expected yield:**

1. Dump and compare a *single* GDN op's output across two GPU processes using
   `tools/oracle_probe.cpp --gdn` with the real 9B weights — the per-op suite
   passes on synthetic inputs, so the fault may be weight-scale dependent.
2. Bisect by ablation: neutralise one op at a time in `gdn_forward`
   ([src/engine.cpp](../../src/engine.cpp)) behind a `KRK_SKIP` env var and
   find which removal makes step 0 stable.
3. Prime suspects, in order: the `extern __shared__` `red_kv`/`red_o`
   reduction buffers and their barrier schedule in
   [gdn_delta_rule_kernel](../../src/hip/kernels/gdn.hpp) (three
   `__syncthreads()` per token, delicate at `n_tok` > 1); the tap-staging
   cache keyed on `(ptr, type, n)` in `stage_gdn_taps`
   ([backend_hip.hip](../../src/hip/backend_hip.hip)), where a recycled
   weight pointer could alias a previous entry; and the f16 conversion path
   for `ssm_conv1d`.
4. Install compute-sanitizer on maclin if a package is available — it would
   settle this in one run.

---

## P1 — known races ThreadSanitizer still reports

`build-tsan/kraken-tests` is **1383/1383 passing with 5 warnings**. Two of
the original seven were the `par_pool` race fixed in `7337f3b`. The rest:

- **`src/http.cpp:456` — data race in `HttpServer::stop()`.** The `stop()`
  thread writes `listen_fd_` while the `run()` thread reads it in the accept
  error path. `running_` is already an atomic; `listen_fd_` is a plain `int`.
  Make it an atomic, or have `stop()` only signal and let `run()` own the
  descriptor.
- **`tests/test_kraken.cpp:1303-1304` — heap-use-after-free in
  `test_gdn_ops`.** A read in `CpuBackend::qwen3_next_split` lands in memory
  freed by `Tokenizer::~Tokenizer` via `Engine::~Engine` in a *later* test.
  Almost certainly a test-lifetime bug (a `CpuBackend` outliving its engine,
  or a dangling vector), not a library defect — but it needs confirming
  before it is dismissed.

---

## P1b — prefill throughput on gfx1031: it was an LDS bank-conflict bug (FIXED)

**The previous entry here was wrong and is retracted.** It reported a
"measured ceiling" of 0.81 TFLOP/s from `tools/probe_gemm_tile.hip` and
concluded that the 6700 XT "simply cannot do dense GEMM quickly" and that
"there is no structural prefill bug on this part". All of that came from a
probe that was broken three ways: it no longer compiled against the current
`krk_hip.hpp` (it borrowed a `HIP_CHECK` the header had since dropped), it
sized its grid from `multiProcessorCount` (20 on a 40-CU part), and it timed
0.084 GFLOP per launch — tens of microseconds of work whose measurement is
launch ramp and drain, not pipe throughput. The file is fixed: it carries its
own `HIP_CHECK`, takes the CU count from argv, runs ~21 GFLOP per launch, and
prints a flatness check. Its section-2 table is flat at ~24.1 TFLOP/s from
NACC 4 up, against a 25.3 TFLOP/s register-only reference
(`tools/dot2_ceiling.cpp`).

### The actual bug: the SIMT tile's LDS fetch aliased every lane onto one bank

`gemm_simt_kernel` (the gfx10 prefill path) fetched each k-pair as two
`_Float16` loads, and re-read `k+1` inside the `(i,j)` loop on top. Its tiles
are padded to `BK+8` halves = 80 B = 20 words per row, so consecutive `tx`
lanes are 80 B apart and the bank of a fetch is `(row*20 + q) % 32`: `tx`
enters only through `tx & 1`. All 16 `tx` lanes of a wave therefore landed on
2 banks — an 8-way serialization on every `ws` fetch. rocprofv3 on the
standalone 128x128x32 tile: **`SQC_LDS_BANK_CONFLICT / SQ_INSTS_LDS = 16.4`**,
`MeanOccupancyPerCU = 15.9` of 32 (VGPR=120), `GRBM_COUNT` = 94% of a 49 ms
run. The kernel was never arithmetic- or occupancy-bound: roughly 43 of those
49 ms were spent serialized in LDS. That is also why the earlier attempts
could not work — a wider tile changes the LDS ops per dot, not the aliasing,
and split-K changes nothing at all.

**The fix, applied to `gemm_simt_kernel`:** fetch the k-pair as ONE u32, and
rotate the pair index by the thread's `tx` (`q -> q ^ tx`). The rotation is a
bijection over the 16 pairs — every thread still sums all of k, in its own
order, so the result stays deterministic run to run — and it spreads the 16
lanes over 16 banks. Measured:

| workload | before | after |
|---|---|---|
| tile harness 128x128x16 s3 TM8 TN8, 4096^3 (verified maxerr 0) | 2.98 TFLOP/s | **8.28 TFLOP/s** |
| tile harness 128x128x32 s2 TM8 TN8 | 2.87 TFLOP/s | 7.35 TFLOP/s |
| `--bench` SmolLM2-135M Q4_K_M, 51-token prefill | 584 tok/s | **1406 tok/s** |
| `--bench` Qwen3.5-9B Q4_K_M, 55-token prefill | 27.9 tok/s | **180-190 tok/s** |
| `--bench` Qwen3.5-9B, decode 256 | 21.3 tok/s | 21.3 tok/s (decode is gemv, untouched) |

The old 27.9 tok/s *was* the llama.cpp reference for this model (28.0), which
is where the "we are at the ground truth, so nothing big is left" reading came
from. There was: 6.5x.

**Correctness:** `kraken-tests` 1383/1383 on the HIP build; the oracle's
per-layer divergence profile (including the pre-existing layer-7 blowup in the
qwen35 test model) is identical before and after the change; every tile-harness
configuration verifies at `maxerr 0.0e+00`.

### What was tried and why it failed

* **128x128 block tile with 8x8 register tiles.** The textbook fix for a
  2:1 FMA:LDS ratio, and the oracle still agreed. Prefill got **2x worse**
  (3245 -> 6882 ms over 801 tokens). Now explained: the wider tile does not
  touch the bank aliasing, and it cuts the grid from ~19 waves to ~5 over 40
  CUs on top. Reverted, correctly.
* **split-K.** Proposed as the fix, never justified. With the tile at 12% of
  the pipe, neither occupancy nor tile shape was ever the constraint.

### What is actually left on this path

1. **The tile is now at 8.28 of the 25.3 TFLOP/s register-only ceiling (33%).**
   The ablation on the fixed kernel attributes 59.5% of the remaining time to
   the global->LDS staging path (latency, not bandwidth: ~93 GB/s of a
   ~384 GB/s card) and 43.7% to barriers. More stages (`BK=16` s4/s5), fewer
   registers, or `global_load_lds` are the plausible next ~1.5-2x.
2. **The int8 DP4A tile had the same aliasing (fixed), and is now ported.**
   `tools/gemm_i8.cpp` fetched `Bs[(tx*TN+j)*BK + q*4]` with a 32-byte row, so
   `tx` dropped out of the bank address entirely (16-way). Rotating the k-quad
   by `tx` fixed it: **6.20 -> 9.27 TOP/s** at 4096^3 (11.26 with a 128x128x16
   s4 tile), `maxerr 0` against the CPU reference in every configuration, and
   `SQC_LDS_BANK_CONFLICT/SQ_INSTS_LDS` 19.4 -> 7.4. The ISA shows what is
   left: the unrotated fetch's fully-unrolled quads merge into 128-bit LDS
   reads (32 `ds_read_b128` per k-tile, all 16 lanes on the same 4 banks), and
   the rotation forbids that merge — 32 wide aliased reads become 64 narrow
   spread ones. Rotating at 16-byte-GROUP granularity so the wide reads come
   back is the next lever.
   The port is `src/hip/kernels/gemm_dp4a.hpp`, behind `KRK_GEMM_INT8=1`
   (off by default, like the other two switches but for a second reason): Q8_0
   weights are consumed as int8 with their own block scales, activations are
   quantized per 32-value block, and each k-tile's int32 partial is folded into
   the fp32 accumulator with that block's scale pair. It is **lossy** — int8
   activations — so it does not become a default without winning first. On
   Llama-3.2-1B-Q8_0 prefill it is **1109 -> 1371 tok/s (+24%)** with decode
   untouched (gemv), the greedy token stream identical, top-3 log-probs within
   0.03 of ~19.0 (0.2%), runs identical run to run, and the kernel profiles at
   1.97 bank conflicts per LDS instruction, 112 VGPRs, no scratch. Both suites
   stay at 1383/1383 with the flag on and off.
   Q4_K/Q5_K/Q6_K experts still take the fp16 tile: their sub-block format
   needs the affine min-term correction, so Q8_0 is the format this covers
   today (Q8_0 experts are the ones whose bytes already are the int8 operand).
3. **The `rows<=16` decode shapes** remain occupancy-starved (0.4-1.2 waves);
   decode is gemv and already at 21.3 tok/s on the 9B, so this needs its own
   measurement before anyone invests.

## P2 — robustness and diagnostics

- **CU count was under-reported (fixed).** HIP returns 20 CUs for a 40-CU
  gfx1031; every occupancy heuristic sized itself against the wrong number.
  `kfd_compute_units()` now reads the KFD topology. Reported as 40 CU. It did
  not move prefill at the time (3245 -> 3245 ms) because the constraint was
  the LDS bank conflict in P1b, not occupancy; with that fixed the same 9B
  prefill went 27.9 -> 180 tok/s.
- **Engine utilization counters are Windows-only.** `--bench` prints
  "engine counters are Windows-only" on Linux, so there is no per-engine
  GPU utilization on the machine where the profiling work happens. The
  gfx1031 and gfx1201 campaigns in [STATUS.md](STATUS.md) §8–§12 were run on
  Windows for this reason.
- **Stale processes are easy to leave behind on maclin.** This session found a
  `llama-server`, a `kraken-server` and a `kraken-oracle --gdn` that had been
  running for 3 hours and were holding 6.44 GiB of VRAM. The SIGPIPE fix stops
  the oracle from hanging that way in future; a "device busy" hint at startup
  would have surfaced it sooner.
- **`/tmp/q9b.gguf`** (5.87 GB) is parked in tmpfs on maclin for the
  nondeterminism work and should be deleted when that work closes.

---

## P3 — documentation

- **`qwen35` / `qwen35moe` are missing from the supported-models list** in
  [README.md](../README.md) even though both load and run. Add them once the
  GPU path is reproducible.
- The README benchmark table has no 9B / gated-delta-net row, because there is
  no trustworthy number to put there yet.
- `--debug-topk` is documented in `--help` only; it belongs in the README's
  verification section alongside the CPU-oracle workflow.

---

## P4 — known limits carried forward from STATUS.md §6

- IQ\* formats except IQ4_NL are rejected at load by design.
- Serving is serialized: no prefix cache, no cancellation, no concurrent
  batching, no beam search.
- Speculative decoding is greedy-only; no MTP, tree attention, or n-gram
  speculation.
- Attention decode launches one block per head (no split-K); sampling is on
  the host.
- The fused decode-layer GEMM phase
  ([src/hip/kernels/fused.hpp](../../src/hip/kernels/fused.hpp)) is a measured
  prototype — `backend_hip.hip` still launches the gemvs separately, so its
  measured saving does not reach the engine.

---

## Housekeeping

- **Unpushed.** `7337f3b` and `8a3cbab` are local only; nothing has been
  pushed since `5bf30a8`. Nothing will be pushed without being asked.
- Working tree is clean at `7337f3b`.

---

## Measured performance (6700 XT, gfx1031, ROCm 7.15/7.17, no WMMA)

`kraken --bench --ctx 512 --chunk 256 --greedy`:

| model | prefill | decode |
|---|---|---|
| Qwen3.5-9B Q4_K_M | 28.0 tok/s | 21.3 tok/s |
| Llama-3.2-1B Q8_0 | 190.7 tok/s | 185.8 tok/s |
| SmolLM2-135M Q4_K_M | 799 tok/s | 354 tok/s |

llama.cpp (Vulkan, same card, same 9B) manages 6.8 tok/s decode, so kraken is
~3.1x faster on the recurrent path.

Prefill was investigated this session and is **not** an outlier: throughput
scales with model size (the 9B is ~6.8x slower than the 1B for ~9x the
parameters) and per-token prefill cost (~35 ms) is close to decode (~47 ms).
An earlier suspicion that prefill was pathological was wrong. The remaining
decode headroom is the usual one: 5.3 GB of weights against a ~384 GB/s bus
implies a ~72 tok/s roofline, so decode sits ~3.4x off bandwidth-bound.