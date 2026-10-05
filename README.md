<img width="768" height="512" alt="Octopus AI Workshop in a Vintage Study" src="https://github.com/user-attachments/assets/7cff7936-e8fe-4da3-a3c8-6be9c04c5b06" />

# KRAKEN

A GGUF inference engine in C++17, built directly on the ROCm HIP SDK, tuned for
AMD RDNA GPUs on Windows 11 and Linux.

No PyTorch. No Python in the hot path. No hipBLAS, no MIOpen, no rocBLAS: every
kernel here is written against the ISA — dense WMMA on RDNA3/RDNA4, packed-math
`v_dot2_f32_f16` on RDNA2.

Targets: **Radeon RX 6700 XT** (`gfx1031`, RDNA2) and **Radeon RX 9070 XT**
(`gfx1201`, RDNA4), plus anything else in the RDNA family.

---

## What is actually different here

Most "AMD inference engines" are a GEMM library plus a graph. KRAKEN is the
opposite: the kernels are the product.

**Weights are never dequantized.** A Q4_K_M row stays Q4_K in VRAM for the whole
run. Every matmul dequantizes inside the inner loop, into registers, at the rate
the ALUs can consume it. There is no "load, decompress, run" pass; there is no
fp16 copy of the model. A 7B Q4_K_M model occupies 4.4 GB, not 14 GB.

**One chunking model for every format.** `dequant_chunk<T>(block, chunk, out)`
materializes exactly 32 values of a block. Every GGUF format is a whole number
of 32-value chunks — one for Q4_0/Q8_0, eight for the 256-value K-quants, one
for raw f16/bf16/f32. Because of that, the GEMM, the GEMV and the embedding
gather share one inner loop and one tiling rule: *K tiles are multiples of 32*.

**Three compute paths, chosen per dtype-pair and per M:**

| path | when | primitive |
|---|---|---|
| `gemv_kernel` | decode, M == 1 | fused dequant + dot into registers, one block per channel |
| `gemm_wmma` | prefill, gfx11/gfx12 | `v_wmma_f32_16x16x16_f16` with a 32x64x64 block tile |
| `gemm_simt` | gfx10, or a non-WMMA target | 64x64 tile, `v_dot2_f32_f16` inner product |

**Prefill skips the output head entirely** except for the final row, so a 4k-token
prompt does not pay for 4000 vocab-sized projections.

**Attention is a real online-softmax kernel.** The KV cache is walked in tiles,
the running max and normalizer are maintained per block, and the output
accumulator is rescaled — the score vector is never materialized. Shared memory
stays under ~33 KiB regardless of context length.

**MoE experts are never loaded up front — and the cache is frequency-aware.**
A `qwen3moe`/`qwen2moe` checkpoint carries one FFN per expert per layer; a
235B-A22B model touches ~22B of them per token. KRAKEN keeps the whole expert
set in the file mapping and pages in only the experts the router selects, into
a byte-budgeted residency set. The policy is **LFU with aging**, not LRU:
expert popularity is skewed and sticky, so every hit earns a counter, a hot
expert gets **pinned** against eviction, and periodic decay makes pins lapse
when traffic moves on — a favourite of the first thousand tokens cannot squat
on its slot forever. Give it a second tier with `--expert-l2-mb` and an evicted
expert is demoted into page-locked host RAM instead of being dropped, so coming
back is a DMA rather than a re-read of the mapping. The tokens that picked the
same expert are permuted into
one contiguous block (`gather_rows`), so each expert runs **three GEMMs per
layer regardless of how many tokens chose it**, and results fold back through a
fused weighted scatter. A machine with far less memory than the model still
decodes — it just reloads more. Results are identical at any cache size and
any chunk size.

---

## Target hardware — in priority order

1. **CPU** — the scalar reference backend. It is the oracle: the host
   dequantizers, the MoE routing and residency policy, the tokenizer and the
   sampling chain are all developed and proven here first, because it runs
   everywhere and its output is the ground truth the device kernels are checked
   against.
2. **RX 6700 XT** (`gfx1031`, RDNA2, 12 GB) — the primary GPU target. No WMMA,
   so it runs the packed-math path (`v_dot2_f32_f16`, f32 accumulation) plus
   the GEMV decode kernel. Correctness and bandwidth-bound decode performance
   are judged on this card first.
3. **RX 9070 XT** (`gfx1201`, RDNA4, 16 GB) — the secondary target. WMMA with
   the `_gfx12` K-split layout, fp8/bf8 support, no `v_dot2`. It inherits every
   host-side feature from the CPU oracle and every RDNA2-validated decision
   before its own paths are exercised.

Everything host-side must work on (1) before it is considered done; anything
device-side is validated in (2) → (3) order. `HSA_OVERRIDE_GFX_VERSION` is not
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
pass, the sampler and both tokenizer families are all covered end to end.

---

## Benchmarks

### The command

Every number below comes from one command (headline rows: three runs per model; full sweep: one run each):

```
kraken --model <MODEL> --bench
```

`--bench` is self-contained and needs no flags: it feeds a fixed prompt (which
tokenizes to **51–125 tokens depending on the model's tokenizer** — 51 for
SmolLM2, 55 for most Qwen-family files, 122–125 for a few others), generates
**256 tokens**, and reports the *whole* process budget, not just the two phases
it owns. Defaults in force for every row: greedy sampling, `--ctx 4096`,
`--chunk 256`. The per-model prompt length is listed in the sweep table below.

```
prefill        55 tokens in 83.9 ms  (655.2 tok/s)
decode         256 tokens in 2896.3 ms  (88.4 tok/s)
startup        8.3 ms  (arg parse + backend create)
load           1410.5 ms  (32% of start-to-end)
total          4399.0 ms  (startup + load + prefill + decode)
```

- **prefill** — prompt tokens / time to process them. Throughput, so higher is
  better and it scales with batch size.
- **decode** — generated tokens / time to generate them. The context grows
  during the run, so this is the *average* over 256 tokens ending at ~300
  tokens of context, not a short-context number.
- **load** — reading the weights into VRAM. It is **32–40% of start-to-end** on
  every model here, which is why it is printed: a report covering only prefill
  and decode invites optimising two thirds of the process.

Configuration: **AMD Radeon RX 9070 XT** (gfx1201, RDNA4, wave32, WMMA gfx12,
64 CU, 15.9 GiB VRAM), HIP 7.16.26323, clang 23.0.0, ROCm 10.1, host Windows 11.

### Results (3 runs each, range = min–max)

| model | layers / embd | prefill tok/s | decode tok/s | load ms |
|---|---|---|---|---|
| SmolLM2-135M-Instruct Q4_K_M | 30 / 576 | 2271–2643 | 472–483 | 342–459 |
| Qwen3.5-0.8B Q4_K_M (GDN hybrid) | 24 / 1024 | 2365–2443 | 339–340 | 517–651 |
| Qwen3-MoE-4x0.6B-2.4B Q4_K_M | 28 / 1024 | 123–218 | 109–117 | 421–587 |
| Qwen3-8B Q4_K_M | 36 / 4096 | 784–853 | 87–89 | 1384–3461 |

These four get three runs each; the MoE prefill band is wide because the first
run pages experts in cold (see the residency section below). The 8B load band
is wide for the opposite reason: one run hit a cold file cache (~3.4 s) while
the other two read at 1.6–5.0 GB/s.

### How close is this to the hardware?

The card's real read bandwidth is **585 GB/s**, not the 644 GB/s on the spec
sheet — measured with a 4096 MiB streaming read (584.9 GB/s). An 8B Q4_K_M
token has to read **5.02 GB** of weights, so the absolute floor is **8.58 ms =
117 tok/s**. Kraken does **87–89 tok/s, ~76% of that ceiling**.

The gap is not bandwidth: **3.78 ms of the 11.27 ms token (34%)** is five
elementwise ops (`rmsnorm`, `rope`, `qk_norm`, `kv_append`, `add_inplace`) that
move 8–16 KB each, 253 launches per token. `rmsnorm` alone is 1165x off its own
memory floor. Fusing that chain is the next large win.

To reproduce, substituting your own path:

```
kraken --model models/SmolLM2-135M-Instruct.Q4_K_M.gguf          --bench
kraken --model models/Qwen3.5-0.8B.Q4_K_M.gguf                 --bench
kraken --model models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf         --bench
kraken --model "G:/More-models/Qwen3-8B-Q4_K_M.gguf"            --bench
```

Run-to-run spread is 2–5% on prefill and under 2% on decode, so treat
differences smaller than that as noise. MoE rows depend on a tunable: they
assume the default expert budget (40% of free VRAM), which is why a small MoE
can look slow next to a dense model far above its weight.

### Full sweep: every loadable file on the box, one run each

66 `.gguf` files were tried (everything in `models/` plus `G:/More-models/`,
skipping only the 3-part Laguna shard set, which is not a loadable model).
38 ran; 28 were refused loudly at load — the refusal reasons are grouped below
the table. Dense rows are VRAM-bound decode; MoE rows (marked \*) page experts
through the default budget, so their prefill includes cold-cache paging.

| file | size | arch · layers / embd | prompt tok | prefill tok/s | decode tok/s | load ms |
|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct.IQ4_XS | 0.09 GB | llama · 30 / 576 | 51 | 2301 | 482 | 344 |
| SmolLM2-135M-Instruct.Q4_K_M | 0.11 GB | llama · 30 / 576 | 51 | 2271 | 472 | 342 |
| Qwen3.5-0.8B.Q4_K_M | 0.53 GB | qwen35 · 24 / 1024 | 55 | 2365 | 339 | 549 |
| Qwen3-MOE-4x0.6B-2.4B-Q4_K_M \* | 0.96 GB | qwen3moe · 28 / 1024 | 55 | 123 | 109 | 587 |
| smolcode-coder-cpp-1.5b-q4_k_m | 1.12 GB | qwen2 · 28 / 1536 | 55 | 1808 | 235 | 649 |
| smolcode-coder-cpp-3b-q4_k_m | 2.10 GB | qwen2 · 36 / 2048 | 55 | 1308 | 160 | 879 |
| Phi-3.5-mini Q4_K_M | 2.32 GB | llama · 32 / 3072 | 125 | 1354 | 95 | 1653 |
| qwen3vl-4b-q4_k_m | 2.50 GB | qwen3vl · 36 / 2560 | 55 | 971 | 130 | 2188 |
| Spark_one.Q6_K | 2.54 GB | qwen2 · 36 / 2048 | 55 | 607 | 122 | 2384 |
| qwen3.5-4b-nvfp4 | 2.54 GB | qwen35 · 32 / 2560 | 55 | 947 | 128 | 1066 |
| Samastam-2.5B-Q8_0 | 2.69 GB | llama · 28 / 2048 | 122 | 2546 | 160 | 2407 |
| VibeThinker-3B.Q8_0 | 3.29 GB | qwen2 · 36 / 2048 | 55 | 1464 | — † | 1925 |
| Qwen3-4B-Instruct-2507-Q6_K | 3.31 GB | qwen3 · 36 / 2560 | 55 | 459 | 99 | 2113 |
| qwen2.5-coder-3b-instruct-q8_0 | 3.62 GB | qwen2 · 36 / 2048 | 55 | 1407 | 124 | 2388 |
| qwen3-1.7b-stem-proof-f16 | 4.07 GB | qwen3 · 28 / 2048 | 55 | 1481 | 132 | 3614 |
| qwen3-1.7b-coder-distilled-sft-f16 | 4.07 GB | qwen3 · 28 / 2048 | 55 | 1313 | 132 | 3674 |
| Qwen3-4B-Q8_0 | 4.28 GB | qwen3 · 36 / 2560 | 55 | 1247 | 102 | 1377 |
| Opus4.7 Distill 4B Q8_0 | 4.48 GB | qwen35 · 32 / 2560 | 55 | 1402 | 97 | 1401 |
| Qwen3.5-9B-DeepSeek-V4-Flash-MTP Q3_K_M | 4.74 GB | qwen35 · 32 / 4096 | 55 | 914 | 77 | 3038 |
| Ornith-1.0-9b STRIX_LEAN | 4.96 GB | qwen35 · 32 / 4096 | 55 | 897 | 96 | 2661 |
| Qwen3-8B-Q4_K_M | 5.03 GB | qwen3 · 36 / 4096 | 55 | 784 | 89 | 1450 |
| mythos-9b-unhinged-heretic.i1-Q4_K_M | 5.03 GB | qwen3 · 36 / 4096 | 55 | 801 | 89 | 1384 |
| lacuna-v1-Q4_K_M | 5.07 GB | llama · 32 / 4096 | 52 | 751 | 92 | 5026 |
| qwen3.5-9b-nvfp4 | 5.31 GB | qwen35 · 32 / 4096 | 55 | 1009 | 88 | 3519 |
| qwen35-9b-instruct-nvfp4 | 5.31 GB | qwen35 · 32 / 4096 | 55 | 930 | 88 | 4653 |
| Q3.5-9B-GLM-5.1-DA Q4_K_S | 5.35 GB | qwen35 · 32 / 4096 | 55 | 949 | 88 | 4136 |
| Qwen2.5-Coder-7B Q5_K_M | 5.44 GB | qwen2 · 28 / 3584 | 55 | 687 | 85 | 3991 |
| Qwen3.5-9b-Sushi-Coder-RL Q4_K_M | 5.63 GB | qwen35 · 32 / 4096 | 55 | 726 | 82 | 3671 |
| omnicoder-9b-q6_k | 7.36 GB | qwen35 · 32 / 4096 | 55 | 294 | 60 | 4395 |
| Qwen3.8-9B-Q6_K | 7.56 GB | qwen35 · 32 / 4096 | 55 | 320 | 60 | 8041 |
| Qwen3-30B-A3B Q2_K \* | 11.26 GB | qwen3moe · 48 / 2048 | 55 | 9 | 21 | 1182 |
| Qwen2.5-Coder-32B Q4_K_M ‡ | 19.85 GB | qwen2 · 64 / 5120 | 55 | 44 | 1.0 | 17319 |
| qwable-v1-mxfp4_moe \* | 20.26 GB | qwen35moe · 40 / 2048 | 55 | 6 | 13 | 4737 |
| laguna-xs2-Q4_K_M \* | 20.27 GB | laguna · 40 / 2048 | 51 | 4 | 9 | 1871 |
| Unsloth-Ornith-1.5-35B-A3B Q4_K_XL \* | 22.36 GB | qwen35moe · 40 / 2048 | 55 | 7 | 17 | 2620 |
| Tiel-Coder-35B-A3B Q5_K_XL \* | 26.59 GB | qwen35moe · 40 / 2048 | 55 | 5 | 9 | 2411 |
| Tiel-Coder-35B-A3B-MTP-APEX \* | 26.67 GB | qwen35moe · 40 / 2048 | 55 | 5 | 15 | 3146 |
| ornith-35b-Q8_0 \* | 36.90 GB | qwen35moe · 40 / 2048 | 55 | 3 | 10 | 2665 |

† VibeThinker emits EOS on the bench prompt, so there is nothing to time: honest
prefill, no decode. ‡ The 32B dense model is larger than VRAM (18.5 GiB resident
on a 15.9 GiB card) and runs oversubscribed at 1 tok/s — it proves the loader
refuses nothing silently, not that this is a configuration anyone should use.

NVFP4 and MXFP4 dequantize fine (the 4B/9B NVFP4 and the MXFP4 MoE rows above);
`nemotron-3-nano-4b-NVFP4` below is refused for its Mamba blocks, not its quant.

### Refused at load (28 files, each with its reason)

| reason | files |
|---|---|
| Speculative-head shard, not a model (runs only via `--draft`) | laguna-xs21-dflash-q8, laguna-s-2.1-DFlash-Q4_K_M, Qwen3.8-27B-DFlash2-Q2_K, Qwen3.8-27B-DFlash2-Q4_K_M, Qwen3.8-27B-DSpark-Q8_0, Qwen3.8-27B-DFlash-bootstrap-Q8_0, DeepSeek-V4-Flash-REAP-MTP, Ternary-Bonsai-27B-dspark-Q4_1 |
| MLA / latent KV compression not implemented | GLM-4.7-Flash-Q4_K_M (deepseek2), DeepSeek-V4-Flash-imatrix (deepseek4) |
| gemma4 geometry not implemented (shared-KV layers, sliding window, softcap) | gemma-4-E4B-it-Q4_K_M, gemma4-coding-Q6_K |
| Mamba-2 blocks (nemotron_h) | nemotron-3-nano-4b-NVFP4 |
| Fused QKV / attention output gate (spark2_5) | Spark-X2.5-4B-Q8_0 |
| Routed attention values (k2-horizon) | K2-Horizon-MoVA-36B-A4B-Q4_K_M |
| Attention output gate (muse-glimmer) | Muse-Glimmer-30B-UD-Q8_K_XL |
| Quant type this build cannot dequantize | Bonsai-27B-Q1_0 (#41), Ternary-Bonsai-2-27B-PQ2_0 (#142), Qwen3.8-27B-UD-Q2_K_XL (#18/#29/#21), Qwen3.8-27B-GSQ-RCO-IQ3_S (#22 + IQ2_XS), Qwen3.8-Distill-35B-Q2KXL (#102/#107), Laguna-XS-2.1-IQ3_XXS (#21/#18/#22), ornith-1.0-35B-Q3_0 (#104/#102) |
| Head geometry the loader rejects | Qwen3.8-27B-Opus-Distill-IQ2_XXS, qwen3.8-flash-next-Q4 (`n_embd` not divisible by head count), Qwen3.8-27B-WebGGUF-Q4_0 (same) |

### Short-context decode

`--bench` deliberately measures decode at a few hundred tokens of context,
where attention is still cheap. To see decode with a nearly empty cache, give
the run a short window:

```
kraken --model models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 64  -n 256 --greedy -v
kraken --model models/Qwen3.5-0.8B.Q4_K_M.gguf --ctx 256 -n 256 --greedy -v
```

`-v` prints `[N prompt / N generated | prefill X ms | decode Y tok/s]` after the
answer. The same model reads **368 tok/s** at ctx 64 and **279 tok/s** over the
256-token `--bench` run; that gap is context cost, not a regression.

### Decode rate is now flat in context

The 0.8B used to lose 90% of its decode rate as the context grew, because its
decode attention never reached the decode kernel at all (head width 256 was
not in the supported set) and ran on 8 blocks of a 64-CU card. With that fixed
and the key range split across blocks, context costs almost nothing:

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
[docs/STATUS.md](docs/STATUS.md) (§8–§11); the current bottleneck analysis is
in [docs/PERF-PLAN.md](docs/PERF-PLAN.md).

### MoE: the residency budget, not the kernels

`Qwen3-MoE-4x0.6B-2.4B` (Q4_K_M, 4 experts/layer, top-2) is **4.8x faster with
a larger expert cache**, and nothing else:

| `--expert-cache-mb` | auto (152.6 MiB) | 256 | 1024 | 4096 |
|---|---|---|---|---|
| decode, 16 tok | 16.0 tok/s | 15.6 tok/s | **77.5 tok/s** | 78.7 tok/s |

The default budget (152.6 MiB) holds ~28 of the layer's 112 expert rows while a
token routes through 56 of them per step, so the hit rate is **0.0%** — 951
loads and 925 evictions across 16 tokens, every expert weight re-fetched from
host memory every step. Raising the budget past the ~600 MiB working set makes
the paging disappear entirely, and 4096 MiB buys nothing further. The card has
15.9 GiB free, so the budget was never the binding constraint — the default
was.

Use `--ctx 256/512` for MoE runs on small machines.

---

## Usage switches and worked examples

`kraken --help` lists every switch. These are the ones that change what you
see rather than what the model does.

### `--profile` — where a decode token goes

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

### `--expert-cache-mb` / `--expert-cache-slots` — MoE residency

The auto budget already covers the whole routed expert set when it fits, so
these are only needed to *pin it down* or to cap it deliberately.

```sh
# cap residency for a model whose experts exceed VRAM
kraken -m /g/More-models/GLM-4.7-Flash-Q4_K_M.gguf --expert-cache-mb 4096 --ctx 4096

# exactly 256 (layer, expert) rows resident, whatever their size
kraken -m /g/More-models/Laguna-S-2.1-UD-Q4_K_M.gguf --expert-cache-slots 256

# check what the cache is doing, and whether it is thrashing
kraken -m models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --info | grep -E 'expert|moe'
```

A **hit rate near 0%** means the working set does not fit — raise the budget.
`loads` plateauing early and `evictions` staying flat is the healthy shape.

### `--bench` — throughput and the honest wall clock

```sh
kraken -m models/Qwen3.5-0.8B.Q4_K_M.gguf --bench -n 64 --greedy
```

Prefill and decode are timed separately. `--greedy` matters: the sampled path
does the full host-side cut and is a different measurement.

### `--expert-l2-mb` — pinned host tier for evicted experts

```sh
kraken -m /g/More-models/GLM-4.7-Flash-Q4_K_M.gguf --expert-cache-mb 2048 --expert-l2-mb 8192
```

Evicted experts demote to a page-locked host tier instead of going back to the
GGUF mapping, so a re-promotion is a DMA rather than a file read. Costs host
RAM; reports `promotions`/`demoted` in `--info`.

### `--draft` / `--draft-tokens` — greedy speculative decoding

```sh
kraken -m /g/More-models/Qwen3-8B-Q4_K_M.gguf --draft models/Qwen3.5-0.8B.Q4_K_M.gguf \
       --draft-tokens 4 --greedy -p "explain quicksort" -n 128
```

Several target tokens per host round-trip. Greedy-only; sampling runs without
the draft and says so.

### Finding a divergence between the two backends

```sh
KRK_DUMP=a.txt ./build-hip/kraken -m M -p "..." -n 1 --greedy            # device
KRK_DUMP=b.txt ./build-hip/kraken -m M -p "..." -n 1 --greedy --cpu      # scalar f32
python tools/dump_diff.py a.txt b.txt
```

The first differing line names the stage and the layer.

### Gates

```sh
ninja -C build-hip kraken kraken-tests kraken-bench
ninja -C build-hip gate                       # numeric tolerances + logits_topk differential
./build-hip/kraken-tests                      # 1847 checks
KRK_N=16 KRK_CHAT=1 bash scripts/coherence_check.sh models/*.gguf
```

`gate` exits non-zero on a NaN, an inf, an out-of-tolerance value or a
`logits_topk` mismatch, so it fails the build rather than printing a
reassuring number. Run all three after any kernel change — a fast wrong kernel
is the failure mode a code review does not catch.

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
proposer. Verified in one batched forward, accepted tokens are bit-identical to
plain greedy output (proven by the test suite); sampling requests fall back to
the plain loop. `--bench` prints the acceptance rate.

**Gated delta net (hybrid recurrent/attention)**: `qwen35` and `qwen35moe`
(Qwen3.5 — conv + gated delta rule in place of attention on most layers). The
f32 CPU path reproduces llama.cpp token for token. **The GPU path is not yet
reproducible** — see [docs/TODO.md](docs/TODO.md) for what has been ruled out
and what is next.

**Not supported in v0.1**, and rejected loudly at load rather than mis-decoded:
the IQ* (`IQ2_XXS`, `IQ3_*`, `IQ4_XS`, ...) formats. `kraken-inspect model.gguf
--quant` tells you which formats a file actually uses before you try to run it.

### Mixture-of-Experts and lazy expert residency

Expert weights dwarf everything else in an MoE model, and only a handful run per
token. Loading them eagerly would need the full model resident, so KRAKEN does
not load them at all: the loader records where each expert's three matrices live
in the mapping, the router picks the top-k per token, and `ExpertCache`
materializes exactly those matrices on demand into a residency set bounded by a
byte budget. The policy is **LFU with aging**: every hit earns a counter, a hot
counter pins the slot against eviction, and periodic decay releases pins when
traffic moves on. Without a pin an expert goes to least-frequent, then oldest.

Evicted experts are *demoted*, not thrown away, when you give the cache a second
tier: `--expert-l2-mb N` reserves N MiB of page-locked host memory that holds
experts evicted from VRAM. Their next request becomes a DMA instead of a re-read
of the file — a fault storm on a mapping thousands of times larger than the
weights themselves. The LFU counter, the pin and the load order all survive the
round trip, so a demoted expert is still the same expert when it comes back, and
the two tiers are budgeted independently: VRAM holds what is about to be used,
RAM holds what was just used.

```sh
# cap resident expert memory (a 235B-A22B model on a 12 GB card)
kraken --model Qwen3-235B-A22B-Q4_K_M.gguf --expert-cache-mb 8000 --prompt "Hi"

# or bound it by slot count instead: one expert resident per layer
kraken --model model.gguf --expert-cache-slots 48 --prompt "Hi"

# keep evicted experts in page-locked host RAM instead of dropping them
kraken --model model.gguf --expert-cache-mb 4096 --expert-l2-mb 16384 --prompt "Hi"

# see the geometry and the active budget
kraken --model model.gguf --info
```

A small real checkpoint to try it on (Q4_K_M, ~965 MB, `qwen3moe`, 4 experts
top-2):

```sh
sh ./scripts/fetch_moe_model.sh
kraken --model ../Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --cpu --info
```

With no flag the budget is auto: 40% of free VRAM on the GPU, a fixed share of
host memory on the CPU backend. The budget is a *soft* cap — at least one expert
is always allowed, so even a 1-slot cache decodes correctly, just with a reload
per step. Routing itself runs on the host (the logits are `[n_tok, n_expert]`),
and the shared expert stays resident because there is only one.

### Formats

`F32` `F16` `BF16` `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` `Q8_1` `Q2_K` `Q3_K`
`Q4_K` `Q5_K` `Q6_K` `Q8_K` `IQ4_NL`

---

## Layout

```
include/krk/         public headers — the engine is usable as a library
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
compiles and runs — and is tested — without a GPU. `Backend` is the seam: the
HIP backend and the scalar reference backend are interchangeable, and the
reference backend is the oracle the device kernels are checked against.

See `docs/ARCHITECTURE.md` for the design, `docs/HARDWARE.md` for the per-GPU
behaviour, and `docs/RESEARCH.md` for the sources this was built from.

---

## A note on verification

The host pipeline (GGUF reader, quantizers, tokenizers, sampler, model loader,
forward pass, CPU backend) is compiled and tested in this repository: 479 checks
pass. That includes MoE: a synthetic `qwen2moe` model is built in a temporary
file, run with a one-slot expert cache, and checked to produce byte-identical
tokens to the same model with the entire expert set resident — a single-expert
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
