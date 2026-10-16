<img width="768" height="512" alt="Octopus AI Workshop in a Vintage Study" src="https://github.com/user-attachments/assets/7cff7936-e8fe-4da3-a3c8-6be9c04c5b06" />

# KRAKEN

A GGUF inference engine in C++17, built directly on the ROCm HIP SDK, tuned for
AMD RDNA GPUs on Windows 11 and Linux.

No PyTorch. No Python in the hot path. No hipBLAS, no MIOpen, no rocBLAS: every
kernel here is written against the ISA - dense WMMA on RDNA3/RDNA4, packed-math
`v_dot2_f32_f16` on RDNA2.

Targets: **Radeon RX 6700 XT** (`gfx1031`, RDNA2) and **Radeon RX 9070 XT**
(`gfx1201`, RDNA4), plus anything else in the RDNA family.

---

## What is actually different here

Most "AMD inference engines" are a GEMM library plus a graph. KRAKEN is the
opposite: the kernels are the product.

**Weights stay quantized in VRAM.** A Q4_K_M row stays Q4_K for the whole run.
Decode dequantizes chunks into registers; WMMA prefill stages dequantized f16
tiles through shared memory. Neither path keeps a full f16 copy of the model.
A 7B Q4_K_M model occupies 4.4 GB, not 14 GB.

**One chunking model for every format.** `dequant_chunk<T>(block, chunk, out)`
materializes exactly 32 values of a block. Every GGUF format is a whole number
of 32-value chunks - one for Q4_0/Q8_0, eight for the 256-value K-quants, one
for raw f16/bf16/f32. Because of that, the GEMM, the GEMV and the embedding
gather share one inner loop and one tiling rule: *K tiles are multiples of 32*.

**Three compute paths, chosen per dtype-pair and per M:**

| path | when | primitive |
|---|---|---|
| `gemv_kernel` | decode, M == 1 | fused dequant + dot into registers, one block per channel |
| `gemm_wmma` | prefill, gfx11/gfx12 | `v_wmma_f32_16x16x16_f16` with a 32x64x128 block tile |
| `gemm_simt` | gfx10, or a non-WMMA target | 64x64 tile, `v_dot2_f32_f16` inner product |

**Prefill skips the output head entirely** except for the final row, so a 4k-token
prompt does not pay for 4000 vocab-sized projections.

**Attention is a real online-softmax kernel.** The KV cache is walked in tiles,
the running max and normalizer are maintained per block, and the output
accumulator is rescaled - the score vector is never materialized. Shared memory
stays under ~33 KiB regardless of context length.

**MoE experts are lazy-loaded - and the cache is frequency-aware.** A
`qwen3moe`/`qwen2moe` checkpoint carries one FFN per expert per layer; a
235B-A22B model touches ~22B of them per token. KRAKEN maps the source file and
pages in only the experts the router selects into a byte-budgeted residency
set. The policy is **LFU with aging**, not LRU:
expert popularity is skewed and sticky, so every hit earns a counter, a hot
expert gets **pinned** against eviction, and periodic decay makes pins lapse
when traffic moves on - a favourite of the first thousand tokens cannot squat
on its slot forever. Give it a second tier with `--expert-warm-mb` (the old
spelling `--expert-l2-mb` is still accepted): WARM is pageable host RAM, shared
with the KV tier's WARM and COLD staging through one process-wide admission
ledger. The tokens that picked the same expert are permuted into one contiguous
block (`gather_rows`); compatible gate/up projections are grouped, followed by
the down projection, and results fold back through a fused weighted scatter.
The file mapping remains the immutable cold source, while promotion uses a
bounded pinned staging ring. A machine with less memory than the model still
decodes when each required tier can be admitted; if retained KV cannot be
preserved safely, inference refuses rather than returning corrupted text.

---

## Target hardware - in priority order

1. **CPU** - the scalar reference backend. It is the oracle: the host
   dequantizers, the MoE routing and residency policy, the tokenizer and the
   sampling chain are all developed and proven here first, because it runs
   everywhere and its output is the ground truth the device kernels are checked
   against.
2. **RX 6700 XT** (`gfx1031`, RDNA2, 12 GB) - the primary GPU target. No WMMA,
   so it runs the packed-math path (`v_dot2_f32_f16`, f32 accumulation) plus
   the GEMV decode kernel. Correctness and bandwidth-bound decode performance
   are judged on this card first.
3. **RX 9070 XT** (`gfx1201`, RDNA4, 16 GB) - the secondary target. WMMA with
   the `_gfx12` K-split layout, fp8/bf8 support, no `v_dot2`. It inherits every
   host-side feature from the CPU oracle and every RDNA2-validated decision
   before its own paths are exercised.

Everything host-side must work on (1) before it is considered done; anything
device-side is validated in (2) -> (3) order. `HSA_OVERRIDE_GFX_VERSION` is not
needed on either card; both are natively supported by the Windows HIP SDK.

---

## Building

### Windows 11 (native)

Install the **HIP SDK for Windows** from AMD (ROCm/HIP SDK for Radeon and
Ryzen), then:

```powershell
.\scripts\build_windows.ps1 -Targets gfx1031
```

For a fat binary covering both cards in the fleet:

```powershell
.\scripts\build_windows.ps1 -Targets "gfx1031;gfx1201"
```

The script finds `hipcc.bat` under `%HIP_PATH%` or the standard AMD install
directories, and falls back to a CPU-only build with a clear message if the SDK
is absent.

### Windows 11 with CMake

```powershell
cmake -B build -DKRK_GPU_TARGETS="gfx1031;gfx1201"
cmake --build build --config Release
```

### Linux

```sh
sh ./scripts/build_linux.sh                    # auto targets
KRK_TARGETS="gfx1031 gfx1201" sh ./scripts/build_linux.sh
```

Or with CMake:

```sh
cmake -B build -DKRK_GPU_TARGETS="gfx1201" && cmake --build build -j
```

### CPU-only (CI, correctness work)

```sh
sh ./scripts/build_linux.sh --cpu-only
# or
cmake -B build-cpu -DKRK_ENABLE_HIP=OFF && cmake --build build-cpu -j
```

---

## Running

```sh
# what is in this file?
kraken-inspect model.gguf

# is my GPU configured correctly?
kraken --model model.gguf --info

# generate
kraken --model model.gguf --prompt "The capital of France is" --max-tokens 64

# chat template, greedy, verbose timing
kraken --model model.gguf --chat --system "You are terse." --prompt "Hi" --greedy -v

# prefill/decode throughput
kraken --model model.gguf --bench --max-tokens 128

# interactive
kraken --model model.gguf --chat

# greedy speculative decoding with a small draft model
kraken --model big.gguf --draft small.gguf --draft-tokens 4 --greedy \
    --prompt "The capital of France is"
```

Run the test suite (no GPU required):

```sh
build/kraken-tests
```

The suite builds a complete miniature LLaMA-family GGUF in a temporary file and
generates through it, so the container parser, the model loader, the forward
pass, the sampler and both tokenizer families are covered end to end. Groups
run in isolated worker processes; read the current group/check counts off the
run because they grow with coverage. `scripts/kraken_tests_baseline_check.sh`
fails when a group that passed in the recorded baseline stops passing. For this
checkout, build from the repo root with `ninja -C build-hip kraken kraken-tests
kraken-bench kraken-inspect` and run `build-hip/kraken-tests.exe`.

---

## Benchmarks

### The command

The headline table below was re-measured on the current tree on 2026-10-11:
three `--bench --greedy` runs per model, same machine and command shape. The
large-model first load was cold; subsequent loads used the warm file cache. The
full sweep later in this section is a historical one-run snapshot, not a
measurement of every row on the current tree.

```
kraken --model <MODEL> --bench --greedy
```

`--bench` is self-contained and needs no flags: it feeds a fixed prompt (which
tokenizes to **51-125 tokens depending on the model's tokenizer** - 51 for
SmolLM2, 55 for most Qwen-family files, 122-125 for a few others), generates
up to **256 tokens** (generation stops at `--ctx`), and reports the *whole*
process budget, not just the two phases it owns. Defaults in force for every
row: greedy sampling, `--ctx 4096`, `--chunk 256`. Generation stops at the
context limit, so rows with a short `--ctx` can produce fewer than 256 tokens.
The per-model prompt length is listed in the sweep table below.



```
# Qwen3-8B, current-tree sample (run 1; cold file cache)
prefill        55 tokens in 56.1 ms  (980.4 tok/s)
decode         256 tokens in 2885.8 ms  (88.4 tok/s)
startup        15.0 ms  (arg parse + backend create)
load           5607.5 ms  (65% of start-to-end)
total          8564.4 ms  (startup + load + prefill + decode)
```

The cold-load row is intentionally kept as an example of why cache state is
recorded; the two warm-cache loads were 1902.2 and 1934.8 ms. See the dated
context check below for a same-build context comparison.

- **prefill** - prompt tokens / time to process them. Throughput, so higher is
  better and it scales with batch size.
- **decode** - generated tokens / time to generate them. The context grows
  during the run, so this is the *average* over 256 tokens ending at ~300
  tokens of context, not a short-context number.
- **load** - reading weights into the selected backend. It is 44-48% of
  start-to-end on the small models here and 39-65% on the 8B, depending on file
  cache state; it is printed rather than assumed.

Configuration: **AMD Radeon RX 9070 XT** (gfx1201, RDNA4, wave32, WMMA gfx12,
64 CU, 15.9 GiB VRAM), Windows 11, ROCm 10.1, clang 23.0.0. The binary
reported HIP 7.16.26332 (SDK 7.16.26331); see the runtime warning and limitation
below.

### Results (current tree, 3 runs each; range shown)

| model | layers / embd | prefill tok/s | decode tok/s | load ms |
|---|---|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | 30 / 576 | 5518-5788 | 553-564 | 373-432 |
| Qwen3.5-0.8B Q4_K_M (GDN hybrid) | 24 / 1024 | 5157-5233 | 378-387 | 597-632 |
| Qwen3-MoE-4x0.6B-2.4B Q4_K_M | 28 / 1024 | 180-185 | 115-119 | 445-456 |
| Qwen3-8B Q4_K_M | 36 / 4096 | 976-980 | 88.4-88.7 | 1902-5608 |

Three sequential runs per model used the fixed `--bench` prompt, 256 greedy
output tokens, `--ctx 4096`, `--chunk 256`; the ranges are run-to-run, not
confidence intervals. The MoE's live policy selected a 610 MiB device expert
budget and a 610 MiB pageable WARM tier on this 15.9 GiB GPU / 95.9 GiB host.
The 8B's first load (5.61 s) had a cold file cache; the next two (1.90-1.93 s)
were warm-cache loads. Throughput is the more stable comparison here; do not
compare load times across those cache states.

Build/runtime note: the binary reported HIP 7.16.26332 (SDK 7.16.26331) and
warned that the loaded HIP runtime did not match the SDK it was built against.
Each run also reported RX 9070 XT / gfx1201 and process-specific GPU engine
activity, corroborating GPU use; no separate kernel-dispatch trace was captured.
The warning limits software-version reproducibility, but is not evidence of CPU
fallback. Stage the matching SDK runtime and record its file version before
using these as a clean release baseline.

Current-tree context check (three runs per arm):

| `--ctx` | generated tokens/run | decode tok/s range | median |
|---:|---:|---:|---:|
| 64 | 56 | 377.2-382.1 | 378.8 |
| 256 | 248 | 378.1-380.1 | 378.7 |

These rates average over different numbers of generated tokens because the
context limit stops generation before `-n 256` is reached. They corroborate
flat decode over these short contexts on this build; the full `--bench` result
above averages 256 tokens while context grows to roughly 300.

The older context-scaling table below records a before/after kernel fix from a
prior build; use the dated current-tree check above for current absolute rates.

AMD distinguishes build-time HIP version macros from runtime version queries,
and defines `hipRuntimeGetVersion()` as an approximate runtime version; it does
not identify the loaded DLL's exact path or file version. See the [HIP version
API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/initialization_and_version.html)
and [rocprofv3 kernel tracing
docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) for runtime provenance and stronger dispatch evidence.


### How close is this to the hardware?

The card's real read bandwidth is **585 GB/s**, not the 644 GB/s on the spec
sheet - measured with a 4096 MiB streaming read (584.9 GB/s). An 8B Q4_K_M
token has to read **5.02 GB** of weights, so the absolute floor is **8.58 ms =
117 tok/s**. The current-tree 8B result is **88.4 tok/s (11.3 ms/token), ~76% of
that ceiling**, averaged over the 256 generated tokens above - and decode is
flat in context (docs/PERF-PLAN.md B0), so the short-context number is the same
within noise. The historical context table below compares pre-fix and fixed
paths; its absolute rates are not current-tree measurements.

The gap is not bandwidth. Five elementwise ops (`rmsnorm`, `rope`, `qk_norm`,
`kv_append`, and the residual add) that each move 8-16 KB used to cost **253
launches per token and 34% of the step** (docs/PERF-PLAN.md B0a); the default-on
folds cut the chain along the way - the GEMV residual epilogue, `add_rmsnorm`,
then this build's rope + kv-append + attention chain (one launch, below) - and
a profiled run on this build ranks the remaining elementwise group at ~12% of
device time.

To reproduce the current headline measurements (three sequential runs each):

```
kraken --model models/SmolLM2-135M-Instruct.Q4_K_M.gguf --bench --greedy
kraken --model models/Qwen3.5-0.8B.Q4_K_M.gguf --bench --greedy
kraken --model models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --bench --greedy
kraken --model "G:/More-models/Qwen3-8B-Q4_K_M.gguf" --bench --greedy
```

Across these sequential repetitions, decode varied by about 2.1% (SmolLM2),
2.4% (Qwen3.5-0.8B), 2.9% (the MoE), and 0.3% (Qwen3-8B); these are observed
ranges, not confidence intervals. The small-model prefill ranges are wider
because the prompts are short. The MoE result uses the current adaptive policy,
not the old fixed fraction: the planner sizes the expert pool to model demand
and a selected VRAM cap (6 GiB by default, raised in 2 GiB steps to at most
14 GiB), then clamps it against live free memory minus a reserve (default 1 GiB
or 1/16 of total card capacity, whichever is larger). Explicit `--vram-cap-mb` and
`--vram-tier` options also constrain that plan. `KRK_VRAM_RESERVE_MB` overrides
the reserve. Explicit `--expert-cache-mb` values that exceed live free VRAM
after reserve are refused with the largest fitting value, rather than silently
trimmed.

The host WARM tiers also share one process-local byte ledger: expert WARM,
KV WARM, and temporary KV COLD read/write staging cannot independently spend
the same sampled headroom. The default host reserve is 2 GiB or one eighth of
installed RAM, whichever is larger; `KRK_RAM_RESERVE_MB` overrides it (setting
it to `0` disables the reserve). A zero or unavailable host-memory reading
refuses new pageable staging. If a KV page cannot be preserved in WARM or COLD,
inference stops with an actionable error. Non-streaming HTTP requests receive
503 `resource_exhausted` with remediation in JSON. If rollback fails before a
stream opens, the response is a normal JSON 503; a later generation failure
emits the resource error in an SSE event after the response opens (its HTTP
status can no longer change).
The ledger serializes KRAKEN's own admissions and lowers its admission ceiling
under external pressure, but it is not an OS reservation: another process may
consume memory between the last sample and allocation.

### Measured decode changes (2026-10-11)

These are separate one-switch A/Bs on the RX 9070 XT (gfx1201), Release build;
all use interleaved, order-balanced pairs and unchanged output. Percentages are
not additive because the experiments use different models and isolated arms.

| Change | Workload and verification | Result |
|---|---|---|
| Defer the expert-pool reuse guard to the copy stream instead of querying a pending event on the host | Qwen3-MoE-4x0.6B, `--ctx 512 -n 64 --greedy`, 9 pairs; same stdout md5 | **+11.279%** at 16 MiB and **+8.569%** at 32 MiB; resident-budget null control: no difference |
| Upload the whole MoE routing plan per layer instead of per expert visit | Qwen3-MoE-4x0.6B, 5 pairs; plan uploads 705 → 649 ops/step; same stdout md5 | **+10.027%**, 5/5 pairs, MAD 2.870% |
| Remove the redundant device drain before greedy top-k readback | SmolLM2-135M and Qwen3.5-0.8B, 5 pairs each; same output | **+5.170%** and **+4.412%**, respectively |

The measurements and raw setup are recorded in
[docs/AUDIT-quiet-vs-busy-primitives.md](docs/AUDIT-quiet-vs-busy-primitives.md),
[docs/traces/launch-floor-and-drains-2026-10-11.txt](docs/traces/launch-floor-and-drains-2026-10-11.txt),
and [docs/AUDIT-active-expert-staging.md](docs/AUDIT-active-expert-staging.md).
The shared host-RAM guard documented above is a correctness/resource-safety
change; no throughput gain is claimed for that ledger itself.

### Historical full sweep: every loadable file on the box (2026-10-10, one run each)


66 `.gguf` files were tried (everything in `models/` plus `G:/More-models/`,
skipping only the 3-part Laguna shard set, which is not a loadable model).
38 ran and 28 were refused loudly at load on the build that produced this sweep
 -  the refusal reasons are grouped below the table, and several of the quant-type
refusals in it have since been lifted (see the note under the table).
Dense rows are VRAM-bound decode; MoE rows (marked \*) page experts
through the default budget, so their prefill includes cold-cache paging.

Each sweep row is **one run**; the headline rows above are the only ones with
ranges. Treat any two sweep rows closer than 2-5% on prefill as noise.

| file | size | arch - layers / embd | prompt tok | prefill tok/s | decode tok/s | load ms |
|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct.IQ4_XS | 0.09 GB | llama - 30 / 576 | 51 | 2301 | 482 | 344 |
| SmolLM2-135M-Instruct.Q4_K_M | 0.11 GB | llama - 30 / 576 | 51 | 2271 | 472 | 342 |
| Qwen3.5-0.8B.Q4_K_M | 0.53 GB | qwen35 - 24 / 1024 | 55 | 2365 | 339 | 549 |
| Qwen3-MOE-4x0.6B-2.4B-Q4_K_M \* | 0.96 GB | qwen3moe - 28 / 1024 | 55 | 123 | 109 | 587 |
| smolcode-coder-cpp-1.5b-q4_k_m | 1.12 GB | qwen2 - 28 / 1536 | 55 | 1808 | 235 | 649 |
| smolcode-coder-cpp-3b-q4_k_m | 2.10 GB | qwen2 - 36 / 2048 | 55 | 1308 | 160 | 879 |
| Phi-3.5-mini Q4_K_M | 2.32 GB | llama - 32 / 3072 | 125 | 1354 | 95 | 1653 |
| qwen3vl-4b-q4_k_m | 2.50 GB | qwen3vl - 36 / 2560 | 55 | 971 | 130 | 2188 |
| Spark_one.Q6_K | 2.54 GB | qwen2 - 36 / 2048 | 55 | 607 | 122 | 2384 |
| qwen3.5-4b-nvfp4 | 2.54 GB | qwen35 - 32 / 2560 | 55 | 947 | 128 | 1066 |
| Samastam-2.5B-Q8_0 | 2.69 GB | llama - 28 / 2048 | 122 | 2546 | 160 | 2407 |
| VibeThinker-3B.Q8_0 | 3.29 GB | qwen2 - 36 / 2048 | 55 | 1464 | - (no decode; warmup-token artifact on short prompt) | 1925 |
| Qwen3-4B-Instruct-2507-Q6_K | 3.31 GB | qwen3 - 36 / 2560 | 55 | 459 | 99 | 2113 |
| qwen2.5-coder-3b-instruct-q8_0 | 3.62 GB | qwen2 - 36 / 2048 | 55 | 1407 | 124 | 2388 |
| qwen3-1.7b-stem-proof-f16 | 4.07 GB | qwen3 - 28 / 2048 | 55 | 1481 | 132 | 3614 |
| qwen3-1.7b-coder-distilled-sft-f16 | 4.07 GB | qwen3 - 28 / 2048 | 55 | 1313 | 132 | 3674 |
| Qwen3-4B-Q8_0 | 4.28 GB | qwen3 - 36 / 2560 | 55 | 1247 | 102 | 1377 |
| Opus4.7 Distill 4B Q8_0 | 4.48 GB | qwen35 - 32 / 2560 | 55 | 1402 | 97 | 1401 |
| Qwen3.5-9B-DeepSeek-V4-Flash-MTP Q3_K_M | 4.74 GB | qwen35 - 32 / 4096 | 55 | 914 | 77 | 3038 |
| Ornith-1.0-9b STRIX_LEAN | 4.96 GB | qwen35 - 32 / 4096 | 55 | 897 | 96 | 2661 |
| Qwen3-8B-Q4_K_M | 5.03 GB | qwen3 - 36 / 4096 | 55 | 784 | 89 | 1450 |
| mythos-9b-unhinged-heretic.i1-Q4_K_M | 5.03 GB | qwen3 - 36 / 4096 | 55 | 801 | 89 | 1384 |
| lacuna-v1-Q4_K_M | 5.07 GB | llama - 32 / 4096 | 52 | 751 | 92 | 5026 |
| qwen3.5-9b-nvfp4 | 5.31 GB | qwen35 - 32 / 4096 | 55 | 1009 | 88 | 3519 |
| qwen35-9b-instruct-nvfp4 | 5.31 GB | qwen35 - 32 / 4096 | 55 | 930 | 88 | 4653 |
| Q3.5-9B-GLM-5.1-DA Q4_K_S | 5.35 GB | qwen35 - 32 / 4096 | 55 | 949 | 88 | 4136 |
| Qwen2.5-Coder-7B Q5_K_M | 5.44 GB | qwen2 - 28 / 3584 | 55 | 687 | 85 | 3991 |
| Qwen3.5-9b-Sushi-Coder-RL Q4_K_M | 5.63 GB | qwen35 - 32 / 4096 | 55 | 726 | 82 | 3671 |
| omnicoder-9b-q6_k | 7.36 GB | qwen35 - 32 / 4096 | 55 | 294 | 60 | 4395 |
| Qwen3.8-9B-Q6_K | 7.56 GB | qwen35 - 32 / 4096 | 55 | 320 | 60 | 8041 |
| Qwen3-30B-A3B Q2_K \* | 11.26 GB | qwen3moe - 48 / 2048 | 55 | 9 | 21 | 1182 |
| Qwen2.5-Coder-32B Q4_K_M ‡ | 19.85 GB | qwen2 - 64 / 5120 | 55 | 44 | 1.0 (oversubscribed - larger than VRAM; do not use as a score) | 17319 |
| qwable-v1-mxfp4_moe \* | 20.26 GB | qwen35moe - 40 / 2048 | 55 | 6 | 13 | 4737 |
| laguna-xs2-Q4_K_M \* | 20.27 GB | laguna - 40 / 2048 | 51 | 4 | 9 | 1871 |
| Unsloth-Ornith-1.5-35B-A3B Q4_K_XL \* | 22.36 GB | qwen35moe - 40 / 2048 | 55 | 7 | 17 | 2620 |
| Tiel-Coder-35B-A3B Q5_K_XL \* | 26.59 GB | qwen35moe - 40 / 2048 | 55 | 5 | 9 | 2411 |
| Tiel-Coder-35B-A3B-MTP-APEX \* | 26.67 GB | qwen35moe - 40 / 2048 | 55 | 5 | 15 | 3146 |
| ornith-35b-Q8_0 \* | 36.90 GB | qwen35moe - 40 / 2048 | 55 | 3 | 10 | 2665 |

† The fixed `--bench` prompt tokenizes to 55 tokens here, prefill runs fine (1591
tok/s), and the GPU backend is fine - `--bench` still emitted 0 decode tokens.
With a normal prompt (`x` -> 1 tok, forced decode) the same model decodes at
~124 tok/s, so the 0.0 is a prompt-training-format artifact, not a kernel or MoE
path defect. The row is kept for prefill honesty; treat its decode column as "no
decode measured", not a throughput number.
‡ The 32B dense model is larger than VRAM (18.5 GiB resident on a 15.9 GiB card)
and runs oversubscribed at 1 tok/s. It stays in the table to show the loader
refuses nothing silently; it is not a configuration anyone should use.

NVFP4 and MXFP4 dequantize fine (the 4B/9B NVFP4 and the MXFP4 MoE rows above);
`nemotron-3-nano-4b-NVFP4` below is refused for its Mamba blocks, not its quant.

The IQ-family and BitNet quant refusals in this table are gone: `IQ2_XS`,
`IQ2_XXS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M` and `Q1_0`/`Q2_0` now
dequantize on both the host reference path and the GPU kernels. Six files the
old table rejected now load, each inspected `runnable` on this build:
Bonsai-27B-Q1_0, Laguna-XS-2.1-IQ3_XXS, Qwen3.8-27B-GSQ-RCO-IQ2_XS,
Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp, Qwen3.8-27B-UD-Q2_K_XL and
Qwen3.8-27B-Opus-Distill-IQ2_XXS. (Ternary-Bonsai-27B-dspark-Q4_1 is not among
them: it is a dspark speculative-head shard, refused on architecture no matter
how its quant decodes.)

One entry in the old table is a second geometry for the same id rather than a
bad file: `Ternary-Bonsai-27B-Q2_0` declares tensor offsets that imply exactly
**2.125 bits per element**, which is **34 bytes per 128 values**. That is the
llama-dx fork's `Q2_0` (`#define QK2_0 128`, `block_q2_0` = `ggml_half` +
`QK2_0/4`), and it is the block its ternary exports are written with. Upstream
llama.cpp spells `Q2_0` as the 64-value / 18-byte block instead, so the two
conventions cannot share a decoder: id 42 here is the fork's 128-value block and
upstream's is id 48, `Q2_0_64`. Both decode, and the file was never malformed.

A file that declares `attention.key_length` no longer has to satisfy
`n_embd % head_count == 0`: Qwen3.5's gated attention legitimately runs 24 heads
of 256 dims over a 5120-wide embedding, because `q_proj` is `2 * n_head * d`
(it emits the query gate alongside the query). That identity is checked only
when the head width has to be inferred from the embedding.

### Refused at load (on the sweep build; each with its reason)

| reason | files |
|---|---|
| Speculative-head shard, not a model (runs only via `--draft`) | laguna-xs21-dflash-q8, laguna-s-2.1-DFlash-Q4_K_M, Qwen3.8-27B-DFlash2-Q2_K, Qwen3.8-27B-DFlash2-Q4_K_M, Qwen3.8-27B-DSpark-Q8_0, Qwen3.8-27B-DFlash-bootstrap-Q8_0, DeepSeek-V4-Flash-REAP-MTP, Ternary-Bonsai-27B-dspark-Q4_1 |
| MLA / latent KV compression not implemented | GLM-4.7-Flash-Q4_K_M (deepseek2), DeepSeek-V4-Flash-imatrix (deepseek4) |
| gemma4 geometry not implemented (shared-KV layers, sliding window, softcap) | gemma-4-E4B-it-Q4_K_M, gemma4-coding-Q6_K |
| Mamba-2 blocks (nemotron_h) | nemotron-3-nano-4b-NVFP4 |
| Fused QKV / attention output gate (spark2_5) | Spark-X2.5-4B-Q8_0 |
| Routed attention values (k2-horizon) | K2-Horizon-MoVA-36B-A4B-Q4_K_M |
| Attention output gate (muse-glimmer) | Muse-Glimmer-30B-UD-Q8_K_XL |
| Head geometry the loader rejects (`n_embd` not divisible by head count, and no `attention.key_length` in the file to override it) | qwen3.8-flash-next-Q4, Qwen3.8-27B-WebGGUF-Q4_0 |

### Short-context decode

`--bench` deliberately measures decode at a few hundred tokens of context,
where attention is still cheap. To see decode with a nearly empty cache, give
the run a short window:

```
kraken --model models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 64  -n 256 --greedy -v
kraken --model models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 256 -n 256 --greedy -v
```

`-v` prints `[N prompt / N generated | prefill X ms | decode Y tok/s]` after the
answer. On the current tree, three runs at ctx 64 measured 377.2-382.1 tok/s
(56 generated tokens before reaching the context limit); three at ctx 256
measured 378.1-380.1 tok/s (248 generated). This is consistent with flat decode
across these short contexts. The 256-token `--bench` average is also similar,
but is a distinct run whose context grows during generation. The older table
below records the before/after kernel fix from a prior build; use the dated
current-tree measurements above for current absolute rates.

### Historical context-scaling comparison

The 0.8B previously lost decode rate as context grew because its decode
attention never reached the decode kernel (head width 256 was unsupported) and
ran on 8 blocks of a 64-CU card. The table compares that pre-fix build to the
fixed path; those absolute rates are historical, not current-tree results.

| prompt ctx | decode, before | decode, after |
|---|---|---|
| 24 | 344 tok/s | 377 tok/s |
| 480 | 178 tok/s | 375 tok/s |
| 1920 | 72 tok/s | 361 tok/s |
| 4800 | 33 tok/s | 357 tok/s |

`KRK_ATTN_SPLIT=0` forces the single-block path so the two can be diffed on
identical input; 40 greedy tokens at ctx 4800 are byte-identical between them.

SmolLM2 135M: 30 layers, 576 embd, GQA 9/3 heads, head_dim 64, FF 1536,
vocab 49152. Qwen3.5-0.8B: 24 layers (18 recurrent), 1024 embd, FF 3584,
vocab 248320. Kernel-level probes put the 9070 XT decode step at 727 tok/s
launched and 741 tok/s under a HIP graph (wall-clock). The decode-bandwidth
campaign, fusion A/B and graph timing methodology are in
[docs/STATUS.md](docs/STATUS.md) (§8-§11); the current bottleneck analysis is
in [docs/PERF-PLAN.md](docs/PERF-PLAN.md).

### The attention chain is one launch

RoPE, the KV append and attention ride one kernel on the single-token decode
row (`KRK_FUSED_ATTN=0` restores the separate chain): the kernel ropes q in
registers and k straight into its cache slot, so the k/v rows never round-trip
through global memory between three launches. On SmolLM2-135M (30 layers, the
shape every fused kernel is sized for) that removes three launches per layer:

| | launches/step | ms/step | decode tok/s | text md5 |
|---|---|---|---|---|
| separate rope+append+attention | 333 | 18.1 | 50.8 | d6e2b2b3b00b3 |
| fused chain | **243** | 13.0 | 64.0 | d6e2b2b3b00b3 |

Throughput is an interleaved order-balanced A/B (9 pairs): **+10.2% median**, B
faster in 8/9 pairs, MAD 4.7%, text identical. The chain carries the per-head
QK-norm and both rope pairings; it is gated to adjacent-pair qk_norm shapes -
the Qwen3-8B shape (NeoX, freq_base 1e6) still diverges from the separate
chain by 1-6 f16 ULP in one head's attention output with the cache row proven
bit-identical, so that shape keeps the fallback (see docs/TODO.md). A
shape the chain cannot express keeps the separate three launches.

### MoE: the residency budget, not the kernels (historical cache-size A/B)

The following 16-token cache-size experiment records the earlier policy and is
kept as a historical comparison; its 152.6 MiB automatic budget is not the
current policy. On the current tree, the three-run `--bench` table above uses
610 MiB for this file. In the older experiment, `Qwen3-MoE-4x0.6B-2.4B`
(Q4_K_M, 4 experts/layer, top-2) was **4.8x faster with a larger expert cache**:

| `--expert-cache-mb` | auto (152.6 MiB) | 256 | 1024 | 4096 |
|---|---|---|---|---|
| decode, 16 tok | 16.0 tok/s | 15.6 tok/s | **77.5 tok/s** | 78.7 tok/s |

In that historical run the default budget held ~28 of the layer's 112 expert
rows while a token routed through 56 of them per step, so the hit rate was
**0.0%** - 951 loads and 925 evictions across 16 tokens. Raising the budget
past the ~600 MiB working set made paging disappear there, and 4096 MiB bought
nothing further. Current auto sizing is adaptive and the dated headline table
above records what it selected on the present build.

### MoE warm startup (this build, this file)

The MoE residency feature from the prior section has a startup mode too: at load,
before the first token, the engine can warm both expert tiers from a routing ranking
so the first decode steps do not pay a cold-cache page storm. Two switches drive it:

- `--expert-warmup N` - bounded warm startup. `-1` (the default) runs it only when the
  routed expert set exceeds the device budget; `0` disables it; a positive `N` forces it.
  `--expert-warmup-ms N` caps that phase's wall time in ms (default 20000 = 20 s, `0` = no
  cap). The phase prints a single verbose line: the order it used, the number of rounds, the
  experts-per-layer depth, how much WARM it filled, how much VRAM it added, bytes read, and
  the wall time. When a `.krakenexperts.json` ranking is present next to the GGUF it is read;
  without one the tiers are filled in even (layer) order instead.

- `--ram-tier N` - **a RAM class, not a MiB cap.** It accepts 16, 24, 32, 48, 64 or 96 (GiB) and
  plans the host WARM tier from the free-RAM budget that class implies, rather than from the
  box's actual free RAM or from `N` literally. On this 95.9 GiB machine `--ram-tier 16`
  resolved to a ~9011 MiB WARM tier and `--ram-tier 24` to ~9600 MiB; a small `N` is not a
  small cap, and a literal MiB restriction is `--expert-warm-mb` / `--expert-cache-mb`.

One file on this box exercises it end to end: `G:/More-models/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf`
(`qwen35moe`, 40 layers, 2048 embd, ff 512, 256 experts/layer top-8, Q2_K_XL, ~11.7 GB).
Its companion `*.krakenexperts.json` is the ranking the warmup reads when present.

This warmup record predates the shared host-memory ledger below and describes
a deliberately requested capacity. Pageable buffers are now admitted lazily
against the live reserve and shared ledger, so the planned WARM capacity is not
an allocation guarantee.

With the ranking read and `--ram-tier 6144` (6144 MiB VRAM budget + 6144 MiB WARM):

    [info ] expert warmup: using ...Qk2xl.gguf.krakenexperts.json (182 positions, routers-only, 10240 ranked pairs)
    [info ] expert warmup: ranked order, 2 rounds, 256 experts/layer deep; WARM 6143 MiB of 6144 MiB in 6553 slots
            (6150 MiB read), VRAM +6553 experts (6143 MiB, 6553 slots); 3126 ms

That warmup is well inside the 20 s ceiling. A full `--bench` (greedy, `--ctx 4096`, `--chunk 256`,
up to 256 decode tokens) on that warmed run then reported:

    prefill        55 tokens in 1251.8 ms  (43.9 tok/s)
    decode         256 tokens in 10948.8 ms  (23.3 tok/s)

and the cache was quiet after warmup - `HOT 96.6% / WARM 3.4% / COLD 3.3%`, 1027 of 6553 slots
pinned, 2818 loads and 2862 evictions across the whole 256-token run. The log on that run also
warns that the `.json` was measured with the experts stubbed (`--expert-stub`), so the ranking is
an approximation; treat the 23.3 tok/s as one data point, not a benchmark.

With the same file and `--ram-tier 9600` but no ranking file (even fill, 1 round):

    [info ] expert warmup: even order, 1 rounds, 256 experts/layer deep; WARM 9600 MiB of 9600 MiB in 10240 slots
            (9600 MiB read), VRAM +10240 experts (9600 MiB, 10240 slots); 4823 ms

after which the routed set fits the tier and the cache produces no further traffic:

    prefill        55 tokens in 353.9 ms  (155.4 tok/s)
    decode         256 tokens in 8067.6 ms  (31.6 tok/s)
    experts        0 loads, 0 evictions, 100.0% hit rate, 1042/10240 pinned

That 31.6 tok/s decode is the cleaner "warmed, cache-quiet" number for this file on this build;
the 23.3 tok/s ranked run is the more interesting one because it is the case the ranking is meant
to help (a tier too small to hold the routed set, where ordering actually matters), but both are
single configuration / single machine / single file / single prompt numbers, not a score table entry.

A `--ram-tier 16` class run on this box resolved to a 9011 MiB WARM tier (the class budgets the tier
from this machine's free RAM, so `16` is not a 16 GiB cap) and, stopped after one decode token with
`-n 1 --ctx 256`, staged 1358 MiB of the routed set into WARM in about 547 ms - that is the shape of
the low-end class engaging, not a throughput number (a one-token run has no timed decode steps).

A note on the community speed figures for this model family: the published ~120 tok/s numbers for
"Qwen3.8-35B-A3B-Distill" are from a **different GGUF file** (empero-ai's distill, Q6_K) run through a
**different tool** (Unsloth Studio) on a **different card** (RX 7900 XTX). This engine's file is
`Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf` - Q2_K_XL, ff 512, 256 experts/layer
top-8 - and this build's warmed decode on it is in the tens of tok/s, not the hundreds. The quant, the
active-FFN width, the expert count, the toolchain and the card all differ; the two numbers are not the same
measurement and should not be compared directly.

Use `--ctx 256/512` for MoE runs on small machines.

---

## Usage switches and worked examples

`kraken --help` lists every switch. These are the ones that change what you
see rather than what the model does.

### `--profile` - where a decode token goes

Prints a per-op timeline for the last complete decode step: every op in issue
order, its cumulative position on the device clock, the **device-idle gap
before it**, bytes and effective GB/s, then a per-component rollup and a
per-op rollup.

```sh
kraken -m models/Qwen3.5-0.8B.Q4_K_M.gguf -p "hi" -n 6 --greedy --profile
```

```
-- step 1/1: ops 3031..3571
   541 ops | host wall 23.592 ms | device spans 12.805 ms (54.3%) | device idle 11.461 ms (48.6%)
     # op                          t+ us    dev us   idle us   host us       MB     GB/s
       0 embed                        0.00      7.52     20.28      6.20     0.00      0.0
       1 rmsnorm                     27.80     20.48     16.24      2.30     0.00      0.0
     ...
     538 gemm(gemv)               23793.83    362.00     14.88      1.40   208.59    576.2
     539 logits_topk              24170.71      5.68    100.32     14.00     0.00      0.0
     540 download                  24276.71     68.70     76.20     75.50     0.00      0.0

     component           ops   %dev      dev us     idle us     host us
     projections        159   36.6%     4929.41    4165.46     614.00
     recurrent           90   16.0%     2151.67    2461.78     350.00
     attention-mix       72   13.3%     1786.35    1638.89     245.50
     norms               67   12.0%     1619.32    2281.73     322.00
     ...
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.69 us host, 18.82 us device each.
```

Read the floor line first: on this stack a timed op costs ~19 µs of device
time to measure, so **any row under that is the recorder, not the kernel**.
Order, call counts and GB/s are exact at every size; use `KRK_TIME=<op>` or a
`--bench` wall delta when you need an undistorted device time.

Tune the window with `KRK_PROFILE_STEPS=N` (last N steps, default 1),
`KRK_PROFILE_FROM=S` (count back from the last), `KRK_PROFILE_OPS=N` (fallback
window when there is no `embed` marker to split on).

### `--expert-cache-mb` / `--expert-cache-slots` - MoE residency

The auto budget covers the routed expert set when it fits in the live plan;
these options let you set the byte budget or cap the resident slot count.
Explicit budgets beyond free VRAM after reserve are refused with an actionable
largest-fitting value. `--vram-cap-mb` and `--vram-tier` also bound the overall
GPU memory plan; `--host-ram-mb` and `--ram-tier` constrain host-tier planning.
The automatic GPU budget is planned from a 6 GiB cap, stepped up in 2 GiB
increments when routed demand needs it, up to 14 GiB, then bounded by the live
free-VRAM reading minus reserve. On the current Qwen3-MoE headline run the
planner reported a 6 GiB cap and a 610.3 MiB expert pool because the routed set
was smaller than the cap.

```sh
# cap residency for a model whose experts exceed VRAM
kraken -m /g/More-models/GLM-4.7-Flash-Q4_K_M.gguf --expert-cache-mb 4096 --ctx 4096

# exactly 256 (layer, expert) rows resident, whatever their size
kraken -m /g/More-models/Laguna-S-2.1-UD-Q4_K_M.gguf --expert-cache-slots 256

# check what the cache is doing, and whether it is thrashing
kraken -m models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --info | grep -E 'expert|moe'
```

A **hit rate near 0%** means the active working set does not fit - raise the
budget if free VRAM permits. `loads` plateauing early and `evictions` staying
flat is the healthy shape.

The bytes the expert pool physically holds are printed too, and they are not
the same number as logical resident bytes: `[stats ] expert-pool budget <b>
committed <c> live <l> reusable <r> bytes; alloc <a> reuse <u> trim <t>
guard-fallback <g> | release <m> mark(s) for <s> slice(s)`.
`committed` is `live + reusable` in 256-byte-aligned device bytes, individual
size-class cells rather than chunks of a fixed-size arena, so a `--expert-cache-mb`
change really does move the physical commitment and a shrink trims free cells
without touching one a kernel is still reading. `guard-fallback` counts the times
a release mark could not be created and the pool fell back to draining the device
before reuse - it should stay 0. The `release` tail is the release marks and the
slices they covered: an eviction returns an expert's three slices in one call and
issues **one** record for the burst, so a resident budget reads `3` slices per
mark and `1` means `KRK_POOL_BURST=0` (the per-slice path, kept as the A/B arm).

A reused cell is *ordered*, not waited for. Its previous reader is marked with an
event recorded on the default stream at release, and the next write into that cell
makes its copy stream `hipStreamWaitEvent` on that mark, so the copy cannot land on
bytes still being read. This replaced a hand-out that asked the device whether the
reader had retired: on a budget that forces a promotion per expert visit the answer
is always "not yet", and the question cost a round trip - measured 46.9 us per cell,
5560 cells, i.e. the whole of the `alloc` line in the expert-path split. Deferring
it took `alloc` from 618.4 to 182.5 ms and decode up 11.3% at `--expert-cache-mb 16`
(9/9 interleaved pairs, identical text). `KRK_POOL_HOST_WAIT=1` restores the old
query-and-wait behaviour, which is the arm to compare against.

The three slices of one eviction are released at the same stream point, so they
share a single mark: measured on Qwen3-MOE-4x0.6B at 16 and 32 MiB, the `room` row
(the release happens inside `make_vram_room`) falls **31% (9/9 interleaved
order-balanced pairs at both budgets)** and the marks go from one per slice to one
per burst. Decode is **NO DIFFERENCE** there - the row is ~18 ms of a ~3.2 s run -
and a record costs ~2.5 us marginal in a busy step against the 29-31 us
`kraken-bench` measures on a quiet queue, so this is a cleanup and not a win. The
sharing is exact, not approximate: the releases in a burst are consecutive with
nothing submitted between them. It must not go further than a burst, because
`paged_publish_` makes the default stream wait on the copy stream while the copy
stream waits on these marks.

Active-expert staging (copy the next routed expert on a side stream while the
current one's kernels run) is implemented and **off by default**; `KRK_ACTIVE_STAGE=1`
turns it on, `KRK_STAGE_TIMING=1` adds the `[stats ] stage-timing` overlap line.
It is output-identical, but on this box it does not pay: at a budget that forces a
promotion per expert visit the step is ~75% promotion, and the copy (0.69-0.74 ms)
is about twice the reader window it would hide behind (0.32-0.34 ms). Measured,
with the A/B runs: [docs/AUDIT-active-expert-staging.md](docs/AUDIT-active-expert-staging.md).

### `--bench` - throughput and the honest wall clock

```sh
kraken -m models/Qwen3.5-0.8B.Q4_K_M.gguf --bench -n 64 --greedy
```

Prefill and decode are timed separately. `--greedy` matters: the sampled path
does the full host-side cut and is a different measurement.

### `--expert-warm-mb` - pageable host tier for experts (`--expert-l2-mb` is the old spelling, still accepted)

```sh
kraken -m /g/More-models/GLM-4.7-Flash-Q4_K_M.gguf --expert-cache-mb 2048 --expert-warm-mb 8192
```

WARM is pageable host RAM, not pinned memory. Cold read-throughs may be kept
there, and a WARM expert promotes through a bounded pinned staging ring; ordinary
VRAM eviction preserves an existing WARM copy without downloading it. Expert
WARM shares live host-memory admission with KV WARM/COLD staging. The `--info`
report shows WARM capacity/admissions, while the end-of-run report shows expert
pool physical commitment separately from logical residency.

### `--draft` / `--draft-tokens` - greedy speculative decoding

```sh
kraken -m /g/More-models/Qwen3-8B-Q4_K_M.gguf --draft models/Qwen3.5-0.8B.Q4_K_M.gguf \
       --draft-tokens 4 --greedy -p "explain quicksort" -n 128
```

Several target tokens per host round-trip. Greedy by default; sampling runs
without the draft and says so, unless `KRK_SPEC_RS=1` enables the
rejection-sampling verifier -- proposals are sampled through the run's
sampler chain, accepted by probability ratio min(1, p/q) against the
target's row, and resampled from the corrected distribution on rejection,
so the emitted stream stays distributed as the target alone would produce
it. A run with `--draft` reports
`[stats ] speculation N rounds, A/P draft tokens accepted (x%)`, and warns
explicitly when a drafter accepted nothing (pure cost at that rate). It also
reports `sampled-accept x% of N scored proposals (temp1 topk20,
rejection-sampling verifier)` -- the acceptance a serving stack's rejection
sampling would see at Poolside's served settings, so a drafter's quality is
comparable to a model card's acceptance length and not only to its greedy
floor (Laguna XS-2.1 pair: 12.5% greedy against 30.7% sampled).

### `--kv-hot-mb` / `--kv-warm-mb` / `--kv-cold-dir` - a tiered KV cache

The KV cache pages like the expert cache does: HOT slots in VRAM, WARM in host
RAM, and an optional COLD spill directory past that.

```sh
# long context on a small card: allow tier migration; insufficient staging fails safely
kraken -m model.gguf --ctx 8192 --kv-hot-mb 512 --kv-warm-mb 4096

# spill evicted pages to disk instead of recomputing them
kraken -m model.gguf --ctx 8192 --kv-cold-dir /kvcache
```

Defaults are geometry-derived: `--kv-hot-mb 0` (default) takes the VRAM the
weights leave free, and without a `--kv-cold-dir` WARM capacity is raised to
cover every KV-carrying layer. The buffers are still allocated lazily and each
one must pass the shared host-memory admission guard before use. If host memory
is unavailable or a WARM/COLD restore cannot be read, KRAKEN refuses the
forward and preserves retained KV bytes instead of zero-filling them. The
OpenAI-compatible HTTP API reports this as HTTP 503 `resource_exhausted` with
remediation in JSON or an SSE error event. A stream's HTTP status is fixed once
the response opens. The cache is packed per layer, so mixed
KV widths work, and it is inert when the whole cache fits: one flat allocation,
exactly as before. Every run ends with `kv tiered: ... promoted ... evicted ...`
statistics. At init the geometry may raise `--kv-warm-mb` so HOT + WARM can
cover one pass when disk spill is unavailable; this is a slot capacity floor,
not an eager allocation or a bypass of the live host-memory reserve. If even
one staging page cannot be admitted, init or the later promotion fails safely.
`scripts/kv_tier_geometry_check.sh` checks the reservation against each model
file's own header.

On generation failure due to KV resource refusal, the CLI exits as an error;
the HTTP endpoints return status 503 with error type `resource_exhausted` and
include the cause plus next steps (free host RAM, raise `--kv-hot-mb`, reduce
`--ctx`, or configure a writable `--kv-cold-dir`). If the initial rollback
fails before a stream opens, the server returns a JSON 503; if generation fails
after opening the stream, it sends the resource error in an SSE event (the HTTP
status can no longer change).

### `--hybrid-experts 1` - overflow experts on the host

The device tier computes the experts it holds; with `--hybrid-experts 1` the
host computes the ones it could not hold, out of the weight mapping, so those
experts cost no VRAM and no WARM RAM. Inert when the budget already covers the
routed set. **It trades memory for speed, and loses on speed**: measured on
Qwen3-MoE-4x0.6B with a 16 MiB expert budget, 5.5 tok/s against 17.2 tok/s for
promoting instead - the host arm is a per-layer barrier the device waits on
(`--hybrid-frac F` sets what share of the overflow the host takes; a
VRAM-resident expert is never sent to the host, and the run prints that policy
next to the split's numbers).

### Finding a divergence between the two backends

```sh
KRK_DUMP=a.txt ./build-hip/kraken -m M -p "..." -n 1 --greedy            # device
KRK_DUMP=b.txt ./build-hip/kraken -m M -p "..." -n 1 --greedy --cpu      # scalar f32
python tools/dump_diff.py a.txt b.txt
```

The first differing line names the stage and the layer.

### Gates

```sh
ninja -C build-hip kraken kraken-tests kraken-bench kraken-inspect
ninja -C build-hip gate                       # numeric tolerances + logits_topk differential
./build-hip/kraken-tests                      # checks (count grows; read it off the run)
sh scripts/kraken_tests_baseline_check.sh     # + comparison against the recorded baseline
KRK_N=16 KRK_CHAT=1 bash scripts/coherence_check.sh models/*.gguf
```

`gate` exits non-zero on a NaN, an inf, an out-of-tolerance value or a
`logits_topk` mismatch, so it fails the build rather than printing a
reassuring number. Run all of these after any kernel change - a fast wrong
kernel is the failure mode a code review does not catch.

`build_check.sh` checks the build rather than the engine: that the binaries are
current with the sources, that the configure actually produced the
configuration it claims (Release, HIP backed in, the requested offload arch,
and the arch of the card in the machine). It exits non-zero on a stale or
mis-detected build, which the test suite cannot see: a Debug build and a
CPU-only binary both pass every test in this repository.

`coherence_check.sh` runs each model twice -- on the device and on the `--cpu`
scalar reference -- so half the engine runs you see in its output are the
reference arm by design. It fails a model whose *device* arm never reached a
device as well: the CLI warns on stderr and continues on the CPU backend when
HIP creation fails, and a checker that discards stderr would compare CPU
against CPU and call it coherence (`FAIL (cpu)`).

---

## HTTP server (OpenAI-compatible)

```sh
kraken-server --model model.gguf --port 8080 --ctx 4096
# GPU when the HIP backend is present; --cpu forces the reference backend

curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/completions \
     -d '{"prompt":"The capital of France is","max_tokens":32,"temperature":0}'
curl -N http://localhost:8080/v1/completions \
     -d '{"prompt":"Hello","max_tokens":32,"stream":true}'        # SSE
curl http://localhost:8080/v1/chat/completions \
     -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```

`POST /v1/completions` and `POST /v1/chat/completions` accept `max_tokens`,
`temperature` (0 = greedy), `top_k`, `top_p`, `min_p`, `repeat_penalty`,
`repeat_last_n`, `seed`, `stop` (string or array) and `stream`. Chat requests
are rendered through the ChatML template and stop on `<|im_end|>` /
`<|endoftext|>`; both endpoints report `finish_reason` (`stop`/`length`) and a
`usage` block. `GET /health` and `GET /v1/models` round out the surface.

`stream: true` replies with `text/event-stream`: one `data:` chunk per
generated token, a final chunk carrying `finish_reason`, then `data: [DONE]`.

The engine is single-sequence, so connections are served concurrently but
generation is serialized; each request starts from a fresh KV context. Any
OpenAI client works:

```sh
curl http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}' \
  | jq -r '.choices[0].message.content'
```

---

## Supported models

Open issues, pending work and known races are tracked in
[docs/TODO.md](docs/TODO.md).

LLaMA-family checkpoints speaking the standard GGUF tensor schema:

`llama`, `llama3`, `mistral`, `qwen2`, `qwen3` (QK-norm), `smollm`, `granite`,
`phi3`, `gemma` / `gemma2` / `gemma3`, and any other arch with the same
`blk.N.attn_*` / `blk.N.ffn_*` layout.

**Mixture-of-Experts**: `qwen2moe` and `qwen3moe`, and any arch that emits the
standard `blk.N.ffn_gate_inp.weight` routing matrix plus the
`ffn_{gate,up,down}_exps` expert tensors (3-D, `[n_embd, n_ff_exp, n_expert]`).
Qwen2-MoE's always-on shared expert (`ffn_*_shexp`) is supported too. Which
layers are MoE is decided per layer from the tensors present, so hybrid stacks
load unchanged.

Grouped-query attention, per-head QK RMSNorm, QKV biases, tied embeddings and
partial RoPE are all handled.

**Greedy speculative decoding**: `--draft small.gguf` runs a second model as a
proposer - or a DFlash block-diffusion head set when the file names one (the
file's architecture decides; the head set carries no embedding or head of its
own and drafts from fused target features injected into its own KV). Verified
in one batched forward, accepted tokens are bit-identical to plain greedy
output (proven by the test suite, including a bitwise self-draft KV-parity
check); sampling requests fall back to the plain loop. Every run reports the
acceptance rate, and a pair whose mask ids disagree is refused at load
(`KRK_DFLASH_ALLOW_BAD_MASK=1` loads it anyway for diagnosis, with a warning).
The local DFlash pair still accepts 0% - the loop is not the cause any more
(self-drafts accept ~100%), and [docs/DFLASH-DRAFTER.md](docs/DFLASH-DRAFTER.md)
tracks what has been ruled out.

**Gated delta net (hybrid recurrent/attention)**: `qwen35` and `qwen35moe`
(Qwen3.5 - conv + gated delta rule in place of attention on most layers). The
f32 CPU path reproduces llama.cpp token for token; the GPU path is
reproducible on `gfx1201` as of 2026-10-03 (two causes were found and fixed -
a packed-query in-place unpack race and a model-blind decode-attention split-K
race, each with before/after md5 evidence); `gfx1031` confirmation is still
outstanding. See [docs/TODO.md](docs/TODO.md) for the full record.

**Now dequantizing**: the whole IQ family - `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M` - plus BitNet's `Q1_0` and `Q2_0` in both
geometries (id 42's 128-value block and upstream's 64-value `Q2_0_64`, id 48),
the four ROCmFPX ids 100/101 and 102/104/107, and PrismML's ternary pair `PQ2_0`/`PTQ1_0` (ids
142/143) together with the block-1024 Hadamard activation transform their
`prism.hadamard` metadata declares. Every codebook is a byte-for-byte
transcription of its producer's header - llama.cpp's `ggml-common.h`, or the
PrismML fork's `ggml-quants.c` for 142/143 - diffed entry by entry, and every
decoder mirrors its `dequantize_row_*`, on both the host reference path and the
GPU kernels, so the `GSQ-RCO`, `IQ3_XXS` and `Bonsai` exports load and decode,
and `Ternary-Bonsai-2-27B-PQ2_0.gguf` passes device-vs-CPU coherence.

`IQ4_NL` and `IQ4_XS` dequantize as before;
the ROCmFPX ids **102, 104, 107** are now implemented (2026-10-07) from the
fork's `rocmfpx_dequantize_row_fp{2,3,6}` with UE4M3 half-block scales;
`kraken-inspect model.gguf --quant` tells you which formats a file actually
uses before you try to run it.

### Mixture-of-Experts and lazy expert residency

Expert weights dwarf everything else in an MoE model, and only a handful run per
token. Loading them eagerly would need the full model resident, so KRAKEN does
not load them at all: the loader records where each expert's three matrices live
in the mapping, the router picks the top-k per token, and `ExpertCache`
materializes exactly those matrices on demand into a residency set bounded by a
byte budget. The policy is **LFU with aging**: every hit earns a counter, a hot
counter pins the slot against eviction, and periodic decay releases pins when
traffic moves on. Without a pin an expert goes to least-frequent, then oldest.

The second tier is inclusive **pageable host RAM**: `--expert-warm-mb N`
(old spelling `--expert-l2-mb`) sets its per-tier capacity limit. Cold reads may
land there before promotion, and VRAM eviction keeps any existing WARM copy
without downloading it. A later demand promotes through a bounded pinned
staging ring. This tier shares process-wide live host-memory admission with KV
WARM and COLD staging; its configured size remains an upper bound, not a
separate claim on all free RAM. Popularity history survives, but a pin promises
VRAM residency and is released on eviction. Batched prefetch counts pending
reservation bytes and never earns demand pins. The file remains the immutable
COLD source.

```sh
# cap resident expert memory (a 235B-A22B model on a 12 GB card)
kraken --model Qwen3-235B-A22B-Q4_K_M.gguf --expert-cache-mb 8000 --prompt "Hi"

# or bound it by slot count instead: one expert resident per layer
kraken --model model.gguf --expert-cache-slots 48 --prompt "Hi"

# keep read-through expert copies in pageable RAM; DMA uses bounded pinned staging
kraken --model model.gguf --expert-cache-mb 4096 --expert-warm-mb 16384 --prompt "Hi"

# see the geometry and the active budget
kraken --model model.gguf --info
```

A small real checkpoint to try it on (Q4_K_M, ~965 MB, `qwen3moe`, 4 experts
top-2):

```sh
sh ./scripts/fetch_moe_model.sh
kraken --model ../Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --cpu --info
```

With no flag, the GPU expert budget follows the live VRAM policy and is bounded
by model demand, the planned cap, and free VRAM after the reserve; it cannot
claim VRAM already used by another process. The CPU backend uses a host-RAM
fallback. A single required expert may exceed the pool's nominal limit so an
inference step can run, but explicit `--expert-cache-mb` budgets that exceed
live available room after reserve are refused at startup. `--expert-cache-slots`
is a slot-count limit. Routing runs on the host (`[n_tok, n_expert]` logits),
and the shared expert stays resident because there is only one.

### Formats

`F32` `F16` `BF16` `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` `Q8_1` `Q2_K` `Q3_K`
`Q4_K` `Q5_K` `Q6_K` `Q8_K` `IQ4_NL` `IQ4_XS` `IQ2_XXS` `IQ2_XS` `IQ2_S`
`IQ3_XXS` `IQ3_S` `IQ1_S` `IQ1_M` `MXFP4` `NVFP4` `Q4_0_ROCMFP4`
`Q4_0_ROCMFP4_FAST` `TQ1_0` `TQ2_0` `Q1_0` `Q2_0` `Q2_0_64` `PQ2_0` `PTQ1_0`,
plus the ROCmFPX ids `Q6_0_ROCMFPX` (102), `Q3_0_ROCMFPX` (104) and
`Q2_0_ROCMFPX` (107)

---

## Layout

```
include/krk/         public headers - the engine is usable as a library
src/                 host implementation (portable C++17, no HIP)
src/hip/             HIP backend, compiled one pass per --offload-arch
src/hip/kernels/     the device kernels, one header per concern
src/{json,http,server}.cpp  JSON, HTTP/1.1 + SSE, OpenAI API layer
tools/               kraken-inspect
tests/               the deterministic suite
docs/                ARCHITECTURE.md, HARDWARE.md, RESEARCH.md,
                      GPU_BRINGUP.md, STATUS.md
scripts/             build_linux.sh, build_windows.ps1
```

The host half of the engine has no HIP dependency at all, which is why it
compiles and runs - and is tested - without a GPU. `Backend` is the seam: the
HIP backend and the scalar reference backend are interchangeable, and the
reference backend is the oracle the device kernels are checked against.

See `docs/ARCHITECTURE.md` for the design, `docs/HARDWARE.md` for the per-GPU
behaviour, and `docs/RESEARCH.md` for the sources this was built from.

---

## A note on verification
The host pipeline (GGUF reader, quantizers, tokenizers, sampler, model loader,
forward pass, CPU backend) is compiled and tested in this repository: checks pass
(read the count off the run - it grows when coverage does).

That includes

MoE: a synthetic `qwen2moe` model is built in a temporary
file, run with a one-slot expert cache, and checked to produce byte-identical
tokens to the same model with the entire expert set resident - a single-expert
MoE layer is checked to reproduce its dense twin exactly, and grouped prefill
(several tokens routed to one expert in a single chunk) is checked to match
one-token-at-a-time routing.

The GPU kernels in `src/hip/kernels/` are written against the documented AMDGPU
builtins and the RDNA3/RDNA4 fragment layouts, but they require a ROCm
toolchain and an RDNA card to compile and run.
**`docs/GPU_BRINGUP.md` is the step-by-step bring-up procedure** (build per
target, `--info` conformance, CPU-oracle vs GPU token-for-token diff, MoE
residency invariants, VRAM sizing, known-unverified kernels, report template).
If you are picking this up:

1. Build with `-Targets` set to *your* GPU, then `kraken --info` and confirm the
   reported `gfx` string matches.
2. Compare `kraken-tests` output against `kraken --cpu` on the same model with a
   greedy sampler. They should agree token for token; a divergence localizes to
   a specific kernel.
3. `--bench` reports prefill and decode separately, which tells you which of the
   two GEMM paths to look at.
