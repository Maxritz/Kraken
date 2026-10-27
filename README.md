<img width="768" height="512" alt="Octopus AI Workshop in a Vintage Study" src="https://github.com/user-attachments/assets/7cff7936-e8fe-4da3-a3c8-6be9c04c5b06" />

# KRAKEN

Kraken is a GGUF inference engine written in C++17 for AMD GPUs. It talks to the
card through the ROCm HIP SDK directly. There is no PyTorch, no Python in the hot
path, and no hipBLAS, MIOpen or rocBLAS. Every kernel in this repository is
written against the instruction set of the card: WMMA on RDNA3 and RDNA4, and
packed `v_dot2_f32_f16` math on RDNA2.

It is built and tested on the **Radeon RX 6700 XT** (`gfx1031`, RDNA2) and the
**Radeon RX 9070 XT** (`gfx1201`, RDNA4), on Windows 11 and on Linux, and it
should run on the rest of the RDNA family.

A 7B model in Q4_K_M occupies 4.4 GB of VRAM and not 14 GB, because the weights
stay quantized for the whole run. A 235B Mixture-of-Experts model runs on a 12 GB
card, because only the experts the router selects have to be resident at any
moment. Those two numbers are what the design is for.

## The idea

Most AMD inference stacks are a GEMM library with a graph on top. This engine is
the other way round. The kernels are the product and the graph is thin.

**The weights stay quantized in VRAM.** A Q4_K row is read as Q4_K. Decode
dequantizes 32-value chunks into registers, and prefill stages dequantized f16
tiles through shared memory. Neither path keeps a full f16 copy of the model.

**One chunking rule for every format.** `dequant_chunk<T>(block, chunk, out)`
materializes exactly 32 values. Every GGUF format is a whole number of 32-value
chunks: one for Q4_0 and Q8_0, eight for the 256-value K-quants, one for raw
f16, bf16 or f32. The GEMM, the GEMV and the embedding gather therefore share one
inner loop and one tiling rule, and K tiles are multiples of 32.

**Three compute paths, chosen per dtype pair and per row count:**

| path | when it runs | primitive |
|---|---|---|
| `gemv_kernel` | decode, one row | fused dequant plus dot into registers, one block per channel |
| `gemm_wmma` | prefill, gfx11 and gfx12 | `v_wmma_f32_16x16x16_f16`, 32x64x128 block tile |
| `gemm_simt` | gfx10, or any non-WMMA target | 64x64 tile, `v_dot2_f32_f16` inner product |

**Prefill skips the output head** except on the final row, so a 4000 token prompt
does not pay for 4000 vocabulary projections.

**Attention is an online-softmax kernel.** The KV cache is walked in tiles, the
running maximum and normalizer are kept per tile, and the output accumulator is
rescaled. The score row is never materialized. Shared memory stays under about
33 KiB at any context length.

**Experts are paged in, and the cache is frequency-aware.** A `qwen3moe` or
`qwen2moe` file carries one FFN per expert per layer, and a 235B-A22B model
touches about 22B of them per token. Kraken maps the file and pages in only the
experts the router selects, into a byte-budgeted residency set. The policy is LFU
with aging rather than LRU, because expert popularity is sticky: every hit earns
a counter, a hot expert is pinned against eviction, and periodic decay lets a pin
lapse when the traffic moves on. `--expert-warm-mb` adds a second tier in
pageable host RAM, and the mapped file is the cold source below it.

## Build

### Windows 11, native

Install the HIP SDK for Windows from AMD, then:

```powershell
.\scripts\build_windows.ps1 -Targets "gfx1031;gfx1201"
```

A single card is `-Targets gfx1201`. The script finds `hipcc.bat` under
`%HIP_PATH%` or the standard install directories, and falls back to a CPU-only
build with a clear message when the SDK is absent.

### Windows 11, with CMake

```powershell
cmake -B build -DKRK_GPU_TARGETS="gfx1031;gfx1201"
cmake --build build --config Release
```

### Linux

```sh
sh ./scripts/build_linux.sh
KRK_TARGETS="gfx1031 gfx1201" sh ./scripts/build_linux.sh
cmake -B build -DKRK_GPU_TARGETS="gfx1201" && cmake --build build -j
```

### CPU only

```sh
sh ./scripts/build_linux.sh --cpu-only
cmake -B build-cpu -DKRK_ENABLE_HIP=OFF && cmake --build build-cpu -j
```

The binaries land in `build-hip/`, or in `build/` for a CPU-only configure. Keep
them current: `kraken-inspect` answers from the same tables the loader uses, so a
stale copy reports `refused` for formats the engine already runs. The CPU backend
is a scalar f32 reference, and it is what the device path is checked against.

### On a virtual NPU

The compute backend is selectable: HIP, the scalar CPU reference, or the virtual
NPU from [npuxl-improved](https://github.com/riteshnair/npuxl-improved) — a
dependency-free software device that runs the GEMM path on its own SIMD kernels
(AVX2 / AVX-512 / NEON / scalar). Build the device first, then the engine against
it:

```sh
git clone https://github.com/riteshnair/npuxl-improved
cmake -S npuxl-improved -B npuxl-improved/build -DVNPU_BUILD_TOOLS=OFF \
      -DVNPU_BUILD_TESTS=OFF -DVNPU_BUILD_SHARED=OFF
cmake --build npuxl-improved/build -j

cmake -B build-npu -DKRK_ENABLE_HIP=OFF -DKRK_ENABLE_NPU=ON \
      -DKRK_VNPU_ROOT=$PWD/npuxl-improved
cmake --build build-npu -j

build-npu/kraken -m model.gguf -p "The capital of France is" -n 64 --npu
```

`--npu` is per run, so the same binary still runs the reference arm (`--cpu`) for
comparison. What goes to the device is the GEMM — the whole of the model's
weight math, and the one VNPU op whose numerics agree with the reference to f32
rounding. The activation-type ops stay on the reference implementation on
purpose: VNPU's swiglu is a polynomial sigmoid that measures 1.8e-05 off on small
inputs, two orders of magnitude worse than an f32 rounding difference, and an FFN
applies it once per layer per token. Handing that over would make the NPU arm a
slightly different model rather than a different execution schedule.
`src/npu/npu_offload.hpp` has the full argument.

Every GEMM call is pinned to one compute unit by default (`KRK_NPU_UNIT=voodoo`;
`cell` and `wildcat` are the other fixed choices), because VNPU's `auto` picks a
different kernel for a one-row call than for a prefill-sized one and the
chunk-invariance
the engine is built around would stop holding as soon as a prompt had more than
four tokens. `KRK_NPU_UNIT=auto` restores VNPU's own choice. The run summary
carries the split:

```
[stats ] npu       gemm 87 call(s), 87 on the device, 0 fell back, 0.7 GFLOP handed over, 22.5 MiB of weights staged
```

Other knobs: `KRK_NPU_THREADS` (device workers, 0 = hardware concurrency),
`KRK_NPU_WEIGHT_MB` (dequantized weight staging budget, default 1024),
`KRK_NPU_PIN_THREADS=1` and `KRK_NPU_TRACE=file.json` (VNPU's own Chrome trace).
A weight tensor that does not fit the staging budget stays on the reference
path, counted in `fell back`, and says so once.

## Run

```sh
kraken-inspect model.gguf                        # what is in this file
kraken -m model.gguf --info                      # the device, and the VRAM ledger
kraken -m model.gguf -p "The capital of France is" -n 64
kraken -m model.gguf --chat --system "You are terse." -p "Hi" --greedy -v
kraken -m model.gguf --bench --greedy            # prefill and decode throughput
kraken -m model.gguf --chat                      # interactive
kraken -m big.gguf --draft small.gguf --draft-tokens 4 -p "The capital of France is"
```

A short run first is the habit worth keeping, because a model that is going to
produce nonsense has already said so by the third token. To check correctness as
well, `scripts/coherence_check.sh model.gguf` runs one prompt on the device and
on the scalar CPU reference and requires the two texts to be identical.

Exit status: `0` ran, `1` the run failed and the message names the phase, `2` the
command line was wrong. A refused GGUF is a load-time verdict and not a crash.

The test suite needs no GPU:

```sh
build-hip/kraken-tests.exe
```

There is also an OpenAI-compatible HTTP server, `kraken-server`, which the long
form describes. It keeps the KV cache between requests: a prompt that repeats or
extends what the previous request prefilled is only prefilled from the point
where it stops matching, and `usage.prompt_tokens_details.cached_tokens` says how
many tokens that was. Measured on a 21-token prompt, the second identical request
went from 135 ms to 27 ms with byte-identical text. A request that fails clears
the cache entry rather than leaving rows behind that a later prompt would assume
were its own.

## Scores

Measured on 2026-10-26 with `kraken --model <MODEL> --bench --greedy
--max-tokens 64`, one run per model, device only, greedy sampling, `--ctx 4096`
and `--chunk 512`. Radeon RX 9070 XT, `gfx1201`, 64 CU, 15.9 GiB VRAM, HIP 7.12
runtime, Release build. The sweep was re-run the same day after the prefill
staging change, so the prefill column is one build newer than the rest of the
record; `docs/AUDIT-prefill-gemm.md` has that change and its numbers.

| Model | Size MiB | Arch | Load ms | Prefill tok/s | Decode tok/s | Note |
|---|---:|---|---:|---:|---:|---|
| Laguna-S-2.1-PS-IQ2_XXS | 32093 | laguna | 23214 | 5.5 | 4.5 | range guard fired |
| K2-Horizon-MoVA-36B-A4B-Q4_K_M | 21331 | not loaded | - | - | - | refused at load |
| laguna-xs2-Q4_K_M | 19335 | laguna | 22089 | 98.1 | 33.2 | |
| LFM2.5-24B-A2B-QAD-Q4_0 | 12949 | lfm2moe | 6293 | 424.3 | 80.7 | |
| Mellum2.1-12B-A2.5B-Thinking-Q8_0 | 12326 | mellum | 9134 | 460.9 | 60.7 | range guard fired |
| Underdog-Saluki-27B-1.0-IQ2-mix-MTP | 7962 | qwen35 | 6102 | 224.9 | 30.8 | |
| omnicoder-9b-q6_k | 7018 | qwen35 | 8763 | 271.2 | 63.1 | |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M | 5192 | qwen2 | 4309 | 561.2 | 72.5 | |
| qwen35-9b-instruct-nvfp4 | 5067 | qwen35 | 4860 | 798.5 | 79.0 | |
| lacuna-v1-Q4_K_M | 4836 | llama | 3956 | 655.3 | 91.7 | |
| Qwen3-8B-Q4_K_M | 4794 | qwen3 | 4279 | 639.5 | 80.8 | |
| mythos-9b-unhinged-heretic.i1-Q4_K_M | 4794 | qwen3 | 3721 | 654.3 | 80.8 | |
| Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M | 4518 | qwen35 | 2632 | 778.7 | 68.8 | |
| Opus4.7-GODs.Ghost.Codex.Distill.4B-Q8_0 | 4274 | qwen35 | 3870 | 1328.8 | 92.8 | |
| Qwen3-4B-Q8_0 | 4082 | qwen3 | 2407 | 1333.5 | 96.2 | |
| VibeThinker-3B.Q8_0 | 3133 | qwen2 | 3181 | 1630.8 | 0.0 | emits EOS at once |
| Samastam-2.5B-Q8_0 | 2560 | llama | 2248 | 3705.3 | 148.8 | |
| Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M | 2211 | llama | 1739 | 1851.7 | 94.2 | |
| smolcode-coder-cpp-3b-q4_k_m | 2007 | qwen2 | 2103 | 1293.5 | 138.8 | |
| smolcode-coder-cpp-1.5b-q4_k_m | 1065 | qwen2 | 1377 | 2521.6 | 196.9 | |
| bitnet-2b-tq1_0 | 1054 | bitnet | 953 | 940.1 | 66.2 | |

Three things the table says:

* **Prefill is still the weak number on the larger dense files, but it moved.**
  Underdog-Saluki-27B reads a prompt at 224.9 tok/s against 30.8 tok/s of decode,
  omnicoder-9b at 271.2 against 63.1, and Qwen2.5-Coder-7B at 561.2 against 72.5,
  against 172.9, 221.5 and 494.3 in the previous sweep. The small models are
  quick at both ends (smolcode 1.5B: 2521.6 and 196.9), so what these rows
  measure is the weight read rather than the kernel shape.
* **What moved is the prefill GEMM's staging chain, and it is measured.** The
  activation tile was the one operand staged with no prefetch, so its loads sat
  in front of the barrier with nothing to overlap them; holding them in
  registers for one round is worth 22% of prefill on Qwen3-8B (830.9 -> 1013.7
  tok/s at a 501-token prompt, interleaved medians) and 31% of the GEMM itself
  (375.1 -> 258.9 ms). The decomposition, the LDS wall that stops a full double
  buffer from fitting, and what is still open are in
  [docs/AUDIT-prefill-gemm.md](docs/AUDIT-prefill-gemm.md): prefill is the GEMM
  first (59 to 69% of it) and attention second (24 to 33%), the GEMM now runs at
  about a quarter of the card's peak against a fifth before, and `--prefill
  exact`, the rung that can produce a hash, measures 8 to 12 times slower than
  the default.
* **MoE decode is bound by expert bring-in, and those rows are not stable.**
  laguna-xs2 holds 74.3% of its expert acquires in VRAM and decodes at 33.2
  tok/s, while LFM2.5-24B, whose routed set fits its tier at 98.9% hits, decodes
  at 80.7. Their prefill column follows the tier's state rather than the kernel:
  laguna-xs2 read 92.8, 82.0 and 98.1 tok/s across three sweeps the same day,
  and Laguna-S-2.1 is 32 GiB against a 15.9 GiB card at 4.5 tok/s.
* **Two MoEs do not produce a trustworthy answer at these defaults.** Mellum2.1
  and Laguna-S-2.1 both leave the f16 activation range, the range guard reports
  the clipped stores, and the run exits 2. Their rates are what the run reached
  before it stopped being trustworthy.

The full record, including the guard details, the files that have left this box,
and the refusal list, is in [docs/MODEL-STATUS.md](docs/MODEL-STATUS.md) and
[docs/traces/results.tsv](docs/traces/results.tsv).

## Where the rest of it is

* [more-info.md](more-info.md) is the long form: every switch with worked
  examples, the full benchmark record with context scaling, KV quantization and
  speculative decode, the HTTP server, the supported model table, and the list of
  what is refused and why.
* [AGENTS.md](AGENTS.md) is the working note: what each gate in the repository
  is for, the trap behind each default, and the measurement each default rests
  on.
* `docs/` holds the design and audit documents, one per subsystem.
* `scripts/` holds the gates (`build_check.sh`, `kraken_tests_baseline_check.sh`,
  `coherence_check.sh`) and the benchmark harnesses.
