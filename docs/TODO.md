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

## P2 — robustness and diagnostics

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