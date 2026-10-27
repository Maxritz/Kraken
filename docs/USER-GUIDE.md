# KRAKEN user guide

Everything needed to go from `git clone` to a token you can read, plus the
numbers to compare your machine against and the handful of flags that actually
change a run.

This guide points at the detail instead of repeating it: the measured results
live in [MODEL-STATUS.md](MODEL-STATUS.md) and the [README](../README.md), the
file-by-file verdicts in [model-inventory.md](model-inventory.md), and what is
still broken in [TODO.md](TODO.md).

---

## 1. What KRAKEN is

A GGUF inference engine written against the RDNA ISA rather than ported onto
it. The host half is portable C++17 with no HIP in it at all; the HIP half is
compiled once per `--offload-arch` and holds the kernels. It loads a `.gguf`,
prefills a prompt on the device, and decodes tokens — from the CLI, or behind
an OpenAI-compatible HTTP server.

What it is not: a llama.cpp fork. There is no CUDA in this repository, no
`ggml`, and no CUDA flash-attention path to configure. Anything you know about
llama.cpp's CUDA build switches does not transfer here — see §8.

Reference machine for every number below: **AMD Radeon RX 9070 XT** (gfx1201,
RDNA4, wave32, WMMA gfx12, 64 CU, 15.9 GiB VRAM), ROCm 10.1 / HIP 7.16,
clang 23.0.0, host Windows 11.

---

## 2. Quickstart

### Prebuilt (Windows x64, gfx1201)

Unzip and run. The archive is self-contained: four executables and the three
HIP runtime DLLs they need.

```sh
unzip kraken-2026-10-06-win-amd-gfx1201.zip
cd kraken
./kraken.exe --model model.gguf --prompt "Once upon a time" --max-tokens 64
```

| file | what it is |
|---|---|
| `kraken.exe` | the engine — generation, `--bench`, `--profile`, `--tokenize`, `--info` |
| `kraken-tests.exe` | the unit suite; CPU-only, needs no GPU |
| `kraken-bench.exe` | standalone benchmarks and `--gate` |
| `kraken-inspect.exe` | GGUF metadata, tensor list, quant histogram, loader verdict |
| `amdhip64_7.dll` | HIP runtime, pinned to the SDK the binary was built against |
| `amd_comgr.dll` | required — load-time dependency of `amdhip64_7.dll` |
| `rocm_kpack.dll` | required — load-time dependency of `amdhip64_7.dll` |

**Keep the three DLLs together.** This was measured, not assumed: removing
`amd_comgr.dll` or `rocm_kpack.dll` makes the process fail to start at all
(exit 127, no message). With all three staged, `--info` reports the SDK's own
runtime (`HIP 7.16.26332`) and prints no warning. With none of them, the binary
falls back to whatever ROCm is on `PATH` and warns that the runtime is not the
one it was built against. It still runs in that case — it is just not the build
you tested.

This archive is built for **gfx1201**. Other GPUs need a rebuild (§3).

### From source

Requirements: CMake ≥ 3.29, Ninja, and an ROCm/HIP SDK on Windows (7.16 here).

```sh
ROCM=G:/ROCM10RT-gfx1201     # your ROCm root, forward slashes

cmake -G Ninja -H. -Bbuild-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DKRK_ENABLE_HIP=ON \
  -DKRK_GPU_TARGETS=gfx1201 \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/amdclang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_FLAGS="--rocm-device-lib-path=$ROCM/lib/llvm/amdgcn/bitcode" \
  -DCMAKE_MODULE_PATH="$ROCM/cmake" \
  -DCMAKE_PREFIX_PATH="$ROCM"

ninja -C build-hip kraken kraken-tests kraken-bench kraken-inspect kraken-oracle
```

A successful configure ends with:

```
-- The HIP compiler identification is Clang 23.0.0 with GNU-like command-line
-- kraken: HIP backend enabled, targets = gfx1201
```

If instead you get `kraken: HIP SDK not found — building the CPU backend only`,
the SDK was not found and you have a CPU-only binary. Check the configure
output before believing any benchmark.

**On this box none of those seven flags is optional and the recipe will not
survive a `cmake -G Ninja -H. -Bbuild` that omits them.** The from-source
recipe above does work when every flag is present, and this repository's
`CMakeLists.txt` now resolves the SDK root, the compilers, the device-bitcode
path and the arch itself before `enable_language(HIP)` — so a configure that
starts from the repository's `CMakeLists.txt` (with the repository's toolchain
present) needs only `-DKRK_GPU_TARGETS=gfx1201` and still gets the HIP backend.
The seven-flag recipe is what a repository that did not do that pre-flight would
require, and each flag is named because CMake cannot be trusted to get that one
right without it.

The three flags that most commonly bite a Windows configure are:

- **`CMAKE_CXX_COMPILER`** — otherwise CMake takes whatever `c++` is first on
  `PATH`. On this box that is Strawberry Perl's GCC 13.2.0, which is not the
  compiler the ROCm SDK ships and cannot produce matching objects.
- **`CMAKE_HIP_COMPILER`** — CMake's HIP module prefers `hip_HIPCC_EXECUTABLE`
  on the `amd` platform and then cannot identify `hipcc`, reporting
  `The HIP compiler identification is unknown`. Point it at the SDK's
  `clang++.exe` instead.
- **`CMAKE_HIP_FLAGS`** — the compiler-ID probe compiles
  `CMakeHIPCompilerId.hip` *before* `CMakeLists.txt` adds its own
  `--rocm-device-lib-path` (that happens after `enable_language(HIP)`). Without
  the flag the probe dies with `cannot find ROCm device library`, the compiler
  ID stays unknown, and configuration fails — see §6.

`CMAKE_HIP_ARCHITECTURES` will read as the probe's default (`gfx906` on this
box) in the cache; `CMakeLists.txt` overwrites it from `KRK_GPU_TARGETS`, so the
only arch that reaches the compiler is `--offload-arch=gfx1201`. Check the
cache line only if the binary behaves like a different card.

### The gates

Run all of these after any kernel change. A fast wrong kernel is the failure
mode code review does not catch.

```sh
ninja -C build-hip kraken kraken-tests kraken-bench kraken-inspect   # check rc, 0 "error:" lines
sh scripts/build_check.sh                              # is the binary what it claims?
./build-hip/kraken-tests                               # 3008/3008 checks
./build-hip/kraken-tests --run-group test_prefill_chunk_bitexact  # chunk 1 vs 8, bit for bit
./build-hip/kraken-bench --gate                        # numeric tolerances + logits_topk
./build-hip/kraken-bench --h2d-cache                   # what a run starts from: no model, one file read
bash scripts/coherence_check.sh models/*.gguf          # device vs scalar CPU, text diff
```

`build_check.sh` is the one that looks at the build instead of the engine, and
it exists because the other four cannot see the failures it catches: a Debug
build, a "HIP SDK not found" configure that quietly produced a CPU-only binary,
and a failed `ninja` whose previous `.exe` is still sitting there ready to be
run. All three pass every test in this repository.

`coherence_check.sh` runs each model **twice** — once on the GPU, once with
`--cpu` — and compares the generated text. A `FAIL` means the device and the
scalar reference disagree; it does not mean the run was CPU-only. `--cpu` is
the reference, so a device bug shows up as the GPU side drifting.

### First run on a new model

```sh
kraken-inspect.exe model.gguf --quant   # arch, formats, verdict -- before you run it
kraken.exe --model model.gguf --info     # geometry, device, KV and expert budgets
kraken.exe --model model.gguf --prompt "..." --max-tokens 64
```

---

## 3. Supported models

### What runs

The LLaMA-family tensor schema, `blk.N.attn_*` / `blk.N.ffn_*`:

`llama`, `llama3`, `mistral`, `qwen2`, `qwen3` (QK-norm), `smollm`, `granite`,
`phi3`, `gemma` / `gemma2` / `gemma3`, and anything else with the same layout.

Grouped-query attention, per-head QK RMSNorm, QKV biases, tied embeddings and
partial RoPE are all handled.

Gemma 2's two softcaps are both applied: the final-logit one (`cap *
tanh(logit / cap)` before sampling) and the ATTENTION-logit one
(`attention.logit_softcapping`), which has to land inside the score formation
because the softmax runs on the capped scores. The reference attention
implements it, so `--cpu`, `--npu` and `--npu-cpu` run gemma2; the HIP kernels
keep four score paths of their own and none caps yet, so a device run refuses
by name rather than decoding a model whose attention is a different function.

**Mixture-of-Experts** — `qwen2moe`, `qwen3moe`, and any arch emitting the
standard `blk.N.ffn_gate_inp.weight` router plus `ffn_{gate,up,down}_exps`
expert tensors. Qwen2-MoE's always-on shared expert is supported. Whether a
layer is MoE is decided per layer from the tensors present, so hybrid stacks
load unchanged.

**Gated delta net** (`qwen35`, `qwen35moe` — Qwen3.5) — conv + gated delta rule
instead of attention on most layers. Reads correctly on the device and matches
llama.cpp token for token on the f32 CPU path. **The GPU path is not yet
reproducible**; treat Qwen3.5 numbers as prefill-and-decode working, numerics
under investigation ([TODO.md](TODO.md)).

**Greedy speculative decoding** — `--draft small.gguf` proposes, the target
verifies in one batched forward. A greedy run with a draft is bit-identical to
the same run without one; the test suite proves it. Sampling runs fall back to
the plain loop.

### Quant formats

`F32` `F16` `BF16` `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` `Q8_1` `Q2_K` `Q3_K`
`Q4_K` `Q5_K` `Q6_K` `Q8_K` `IQ4_NL` `IQ4_XS` `IQ2_XXS` `IQ2_XS` `IQ2_S`
`IQ3_XXS` `IQ3_S` `IQ1_S` `IQ1_M` `MXFP4` `NVFP4` `Q4_0_ROCMFP4`
`Q4_0_ROCMFP4_FAST` `TQ1_0` `TQ2_0` `Q1_0` `Q2_0` `Q2_0_64` `PQ2_0` `PTQ1_0`

`IQ4_NL` and `IQ4_XS` both dequantize — `SmolLM2-135M-Instruct.IQ4_XS` loads,
inspects as runnable, and decodes at ~599 tok/s here. The whole IQ family now
joins them: `IQ2_XXS`/`IQ2_XS`/`IQ2_S`, `IQ3_XXS`/`IQ3_S`, `IQ1_S`/`IQ1_M`, plus
BitNet's `Q1_0`/`Q2_0` — id 42 is the 128-value / 34-byte block the Bonsai
ternary exports are written with, and upstream's 64-value block is id 48,
`Q2_0_64` — and the ROCmFPX ids 100/101. Each codebook is a
byte-for-byte transcription of llama.cpp's `ggml-common.h`, diffed entry by
entry, and each decoder mirrors its `dequantize_row_*`, on both the host
reference path and the GPU kernels — so the `GSQ-RCO`, `IQ3_XXS` and `Bonsai`
exports load and decode. NVFP4 and MXFP4 dequantize fine too. The PrismML
ternary pair `PQ2_0`/`PTQ1_0` (ids 142/143) decodes on both paths as well,
together with the block-1024 Hadamard activation transform a file declares in
its `prism.hadamard` metadata — `Ternary-Bonsai-2-27B-PQ2_0.gguf` passes
device-vs-CPU coherence.

The ROCmFPX ids **102, 104, 107** are now implemented (2026-10-07): host and device dequantizers from the fork's `rocmfpx_dequantize_row_fp{2,3,6}`, with UE4M3 half-block scales.

### What is refused, and why

The loader refuses rather than guesses: 28 of the 66 files on the reference box
were rejected, each with a stated reason. The reasons are grouped in the
[README's refusal table](../README.md) — speculative-head shards (runs only as
`--draft`), MLA / latent KV compression, gemma4 shared-KV geometry, Mamba-2
blocks, fused-QKV and attention-output-gate variants, unsupported quants, and
head geometry the loader rejects.

Two things have changed since that count was taken. The unsupported-quant group
the ROCmFPX ids 102/104/107 also dequantize now, because the IQ family, `Q1_0`/`Q2_0`
and `PQ2_0`/`PTQ1_0` (142/143) dequantize. And a file that declares
`attention.key_length` no longer has to satisfy `n_embd % head_count == 0`:
Qwen3.5's gated attention legitimately runs 24 heads of 256 dims over a
5120-wide embedding, because `q_proj` is `2 * n_head * d` (it emits the query
gate alongside the query).

Separately, `Ternary-Bonsai-27B-Q2_0` is *not* a bad file — this was claimed
earlier and is withdrawn. Its tensors carry id 42 spelled as the llama-dx
fork's 128-value / 34-byte `Q2_0`, where upstream computes 18 bytes per 64 and
therefore rejects it; Kraken decodes the file's own layout, and it passes
coherence against the scalar reference.

When a model matters to you, ask the loader instead of guessing:

```sh
kraken-inspect.exe model.gguf --quant
# arch       qwen3moe -- supported (MoE)
# verdict    runnable
# format histogram:
#   Q4_K     16 tensors   7.6 MiB
```

`docs/model-inventory.md` holds the same verdict for all 87 files known on the
reference box (45 runnable) — regenerate with
[scripts/model_inventory.sh](../scripts/model_inventory.sh).

---

## 4. Benchmarks

### The command

```sh
kraken.exe --model model.gguf --bench
```

Self-contained: a fixed prompt (51–125 tokens depending on the tokenizer), 256
generated tokens, greedy, `--ctx 4096`, `--chunk 256`. It reports the whole
process budget rather than just the two phases:

```
prefill        55 tokens in 83.9 ms  (655.2 tok/s)
decode         256 tokens in 2896.3 ms  (88.4 tok/s)
startup        8.3 ms  (arg parse + backend create)
load           1410.5 ms  (32% of start-to-end)
total          4399.0 ms
```

- **prefill** — prompt tokens per second. Scales with batch size.
- **decode** — the number that matters interactively. Measured as the average
  over 256 tokens ending at ~300 tokens of context, not at short context.
- **load** — reading weights into VRAM. It is **32–40% of start-to-end** on
  every model here, which is why it is printed. A report covering only prefill
  and decode invites optimising two thirds of the process.

### Loading modes: memory vs disk

`--bench` with a budget is how the two loading paths are compared:

```sh
# memory-resident: upload once, keep it
kraken.exe -m M --bench --ctx 512 --chunk 256 -n 32 --greedy --weight-vram-mb -1

# disk: 2 GiB resident, everything else re-read from the file before each layer
KRK_WEIGHT_PROMOTE=0 kraken.exe -m M --bench --ctx 512 --chunk 256 \
    -n 32 --greedy --weight-vram-mb 2048
```

`--bench` runs its own fixed prompt and ignores `-p`, so the prompt token
count depends on the model's tokenizer (125 on Phi, 55 on Qwen): a Phi row's
prefill tok/s is not comparable with the rest. `scripts/bench_loading.sh`
runs the three arms interleaved and order-balanced and prints one row per
arm:

```sh
sh scripts/bench_loading.sh /g/More-models/Qwen3-8B-Q4_K_M.gguf:32 \n    /g/More-models/Qwen3.8-27B-WebGGUF-Q4_0.gguf:8
```

Measured on this box, Qwen3-8B Q4_K_M goes **92.1 -> 2.5 tok/s** decode when 25
of its 36 layers stream (prefill 1037 -> 118), and a 27B Q4 that fits at
14.4 GiB goes **14.9 tok/s resident, 8.6 with the auto 12 GiB budget, 0.5 with
the 2 GiB ring**. `--weight-vram-mb N` alone is not a disk arm: a streamed
layer is promoted resident on first use, so the ring has to be paired with
`KRK_WEIGHT_PROMOTE=0` to be measured. Full table, MoE caveat and the
order-balanced loop: [BENCH-LOADING.md](BENCH-LOADING.md).

### Headline numbers (3 runs each, min–max)

| model | layers / embd | prefill tok/s | decode tok/s | load ms |
|---|---|---|---|---|
| SmolLM2-135M-Instruct Q4_K_M | 30 / 576 | 2271–2643 | 472–483 | 342–459 |
| Qwen3.5-0.8B Q4_K_M (GDN hybrid) | 24 / 1024 | 2365–2443 | 339–340 | 517–651 |
| Qwen3-MoE-4x0.6B-2.4B Q4_K_M | 28 / 1024 | 123–218 | 109–117 | 421–587 |
| Qwen3-8B Q4_K_M | 36 / 4096 | 784–853 | 87–89 | 1384–3461 |

Run-to-run spread is 2–5% on prefill and under 2% on decode: **treat anything
closer than that as noise.** MoE rows depend on the expert budget, which is why
a small MoE can look slow next to a dense model far above its weight — its
first run is also paging experts in cold.

The full 66-file sweep is in the [README](../README.md); the 25-model score
table with coherence verdicts per model is in
[MODEL-STATUS.md](MODEL-STATUS.md).

### How close to the hardware is this

The card's real streaming read is **585 GB/s** (measured, not from the spec
sheet). An 8B Q4_K_M token must read 5.02 GB, so the floor is **8.58 ms =
117 tok/s**. KRAKEN does **87–89 tok/s — about 76% of that ceiling.**

The remaining gap is not bandwidth: **3.78 of 11.27 ms (34%)** is five
elementwise ops (`rmsnorm`, `rope`, `qk_norm`, `kv_append`, `add_inplace`) that
move 8–16 KB each across 253 launches per token. `rmsnorm` alone is 1165× off
its own memory floor. Fusing that chain is the known next large win.

### A number you should distrust

`--profile` self-calibrates its own instrumentation floor at startup (~0.7 µs
host, ~19 µs device per timed record). Read that line first: **any op row under
the floor is the recorder, not the kernel.** Order, call counts and GB/s are
exact at every size; for an undistorted device time use `KRK_TIME=<op>` or a
`--bench` wall delta.

---

## 5. The flags worth knowing

`kraken --help` lists all of them. These are the ones that change what you see
or what fits in memory.

### Measurement

| flag | effect |
|---|---|
| `--bench` | prefill/decode/load/total on a fixed prompt |
| `--greedy` | force argmax. Use it for benchmarks — the sampled path does a full host-side cut and is a different measurement |
| `--profile` | per-op device timeline for the last decode step, with the idle gap before each op |
| `--tokenize` | print the prompt's token ids and exit |
| `--prefill-hash` | prefill the prompt, print a hash of every prefill row's stages and of the last logits row, then exit. Two runs that differ only in `--chunk` print the same numbers iff the multi-row forward is bit-exact |
| `--info` | geometry, device, KV and expert cache budgets, then exit |

`--profile` window: `KRK_PROFILE_STEPS=N` (last N steps), `KRK_PROFILE_FROM=S`,
`KRK_PROFILE_OPS=N`.

### Memory and context

| flag | effect |
|---|---|
| `--ctx N` | KV capacity in tokens (default 4096). Use 256/512 for MoE runs on small machines |
| `--chunk N` | prefill batch size (default 256) |
| `--vram-cap-mb N` | total device memory to plan for. `0` uses the policy: 6 GiB, then +2 GiB at a time while the card has room and the model needs it, up to **14 GiB** |
| `--vram-tier N` | **a VRAM class, not a MiB cap.** The GiB this run may plan for, clamped to what the card has: `--vram-tier 8` on a 12 GB card plans 8 GiB, and the same command line on an 8 GB card plans 8 GiB there too. One line for several machines; a literal MiB restriction is `--vram-cap-mb`, which wins if both are given |
| `--host-ram-mb N` | host memory to plan for; `0` uses a quarter of installed RAM. Bounds every host tier; the model mapping and dense trunk sit outside it |

Anything a run does not plan for — a long KV context, another application — is
what the headroom is for.

### MoE residency

Expert weights dwarf everything else and only a few run per token, so they are
never loaded eagerly: the router picks the top-k, and `ExpertCache`
materializes exactly those matrices under a byte budget, LFU with aging.

| flag | effect |
|---|---|
| `--expert-cache-mb N` | VRAM residency budget in MiB (`0` = auto) |
| `--expert-cache-slots N` | cap resident `(layer, expert)` slots instead of bytes (`0` = auto) |
| `--expert-warm-mb N` | WARM tier in host RAM for evicted experts, in MiB; `<0` auto, `0` off (`--expert-l2-mb` is the old spelling). Auto is the largest tier that leaves the OS its reserve: 60% of installed RAM (9.6 GiB on a 16 GiB machine, 14.4 on 24, 57.6 on 96) or free RAM minus that reserve if smaller, capped at the expert corpus -- so a 16 GiB machine holding Qwen3-30B-A3B's 9.6 GiB set needs no flag at all |
| `--expert-pin-mb N` | pin the first `N` MiB of that tier, in rank (admission) order: a promotion out of pinned memory is one DMA with no host memcpy in front of it. Measured 22.7 GB/s against 7.6 on a 120B MoE, +18-25% decode, and +13-18% on the REAP-320 and 35B-A3B files. **Automatic when it measures a win** (it times a 16 MiB pageable copy against a pinned one at startup and pins at >= 1.20x; the size is a quarter of RAM capped at 16 GiB, and the tier is pinned whole inside that bound (measured 139.8 -> 65.3 ms/step on the 256 MiB expert arm, identical text), since pinned RAM cannot be paged out); `0` opts out, `KRK_HOST_PIN_MB` is the same knob, `KRK_PIN_AUTO=0` is the A/B arm, `KRK_H2D_PROBE=0` falls back to the runtime's BAR hint. That probe's verdict is cached per card AND driver version (30-day expiry): `KRK_H2D_PROBE_RERUN=1` measures again and `KRK_H2D_CACHE=0` (or `KRK_H2D_CACHE=<path>`) disables or relocates the cache. The same probe picks the PAGEABLE path -- ring versus a driver-staged direct DMA, direct taken at >= 1.20x, which a wide aperture (Resizable BAR) should buys by removing the host memcpy, and then the pin declines on its own because there is no copy left to remove; `KRK_WARM_DMA=1|2` forces either path |
| `--router-resident P` | residency preference in the router: a pick within `P` of the top expert score (a fraction of it, `0.10` = 10%) goes to the expert already in VRAM, then to the one in WARM. Measured -20% bytes/token and +16.7% decode on a 120B MoE at `0.10`; it changes the route, so it can change the text. `KRK_ROUTER_TIE_MARGIN` is the same knob; `0`/absent is off |
| `--expert-warm-prefetch` | fill WARM at load instead of on demand |
| `--expert-warmup 0/1` | ranked startup phase: stage the ranked expert set into WARM, then promote the top of that order into VRAM. `<0` auto (on for every MoE with a device tier — what it removes is the compulsory first touch, not eviction), `0` off, `1` forced; `KRK_EXPERT_WARMUP=0/1` overrides |
| `--expert-warmup-ms N` | wall-time ceiling for that phase in ms (default `20000`); `0` removes it. Being cut short costs coverage, never the hottest experts |
| `KRK_WARM_PREFILL=1\|0` | whether that phase also prefills VRAM from the ranking. Default is a rule, `corpus <= pool + 5%`: the prefix is a head start only while the pool holds the routed set (24B-A2B 103% → prefilled, 84.0 tok/s against 25.2 without; 35B-A3B 215% → skipped, 22.3 against 10.4 with). `1` forces it, `0` forbids it |
| `KRK_EXPERT_WARM_MAP=1` | the opposite tier policy: keep nothing in host RAM and serve an evicted expert from the page-cached mapping (the warmup then only touches the page cache). **Off because it measured slower** — 13.6 against 17.2 tok/s on the 35B-A3B and 16.5 against 84.0 on the 24B-A2B, because a promotion is one batched copy and the mapping path is three serialised reads. An A/B arm, not a recommendation |
| `--ram-tier N` | plan the host tier for the `N` GiB RAM class (16/24/32/48/64/96) instead of the installed RAM; `0` = installed. Aliased `--ram-tier-gb` |

The auto budget already covers the whole routed expert set when it fits, so
these flags pin it down or cap it deliberately. **A partial budget below the
working set is pure churn at MoE scale** — the engine warns when it detects
that. A hit rate near 0% means the working set does not fit: raise the budget.
`loads` plateauing early with flat `evictions` is the healthy shape.

Evicted experts are *demoted*, not dropped, when a WARM tier exists: the next
request becomes a DMA instead of a re-read of the mapping, and the LFU counter,
the pin and the load order all survive the round trip.

### A nextn (MTP) block in its own file

| flag | effect |
|---|---|
| `--mtp FILE` | load the nextn block from a separate GGUF, for a `-noMTP` trunk whose MTP was published beside it (`mtp-<model>.gguf`). Implies `--mtp-probe` |

The sidecar is validated against the trunk before the first token: same
architecture, a block at or past the trunk's stack, the same embedding width and
head count (a same-family head with a different attention shape is refused with
both numbers), and a per-layer `head_count_kv` array is read at the block's own
index. `--mtp-spec` then drives it as a drafter exactly as an in-file block.

### KV tiers

| flag | effect |
|---|---|
| `--kv-hot-mb N` | KV HOT tier in VRAM. `0` (default) takes what the weights leave free |
| `--kv-warm-mb N` | KV WARM tier in pageable host RAM; `<0` auto-sizes, `0` disables |
| `--kv-cold-dir DIR` | spill directory for pages evicted past WARM; unset keeps them in RAM and recomputes instead of writing |

If the whole cache fits, `--kv-hot-mb` is inert and the cache is one flat
allocation as before. Lower it to make a long context run by paging the
overflow through WARM rather than refusing it.

### Speculative decoding

```sh
kraken.exe -m big.gguf --draft small.gguf --draft-tokens 4 --greedy -p "..." -n 128
```

Several target tokens per host round-trip. Greedy only — a sampling request
runs without the draft and says so.

### An external draft head (EAGLE-3)

```sh
kraken.exe -m big.gguf --draft-head head.gguf --draft-tokens 2 --greedy -p "..." -n 128
```

A published speculator converted once with `tools/eagle_to_gguf.py`. It takes no
VRAM (it runs on the host) and no second model, and under `--greedy` the emitted
text is identical to plain greedy -- only the speed changes. The loader refuses
a head whose hidden width or target vocabulary disagrees with the model, and a
recurrent target (no rewind). `--draft-head-prime N` bounds how many recent
positions it attends (default 256; `0` = the whole prompt).

Unlike `--draft` and `--mtp-spec`, the head also drafts under sampling: the
round draws each proposal from the head's distribution, accepts it with
probability `min(1, p/q)` against the target's row, and on a rejection emits a
draw from `max(0, p - q)` instead. That is the speculative-sampling rule
(arXiv:2302.01318), so turning the sampler on does not turn the head's
distribution into the output. `KRK_SPEC_RS=0` disables it and the request runs
without the head.


### Finding a divergence

```sh
KRK_DUMP=a.txt kraken.exe -m M -p "..." -n 1 --greedy          # device
KRK_DUMP=b.txt kraken.exe -m M -p "..." -n 1 --greedy --cpu    # scalar f32
python tools/dump_diff.py a.txt b.txt
```

The first differing line names the stage and the layer. That is the fastest way
to tell a prefill bug from a decode bug.

The same question without a dump, and the one the test suite gates on:

```sh
kraken.exe -m M -p "..." --ctx 64 --chunk 1 --greedy --prefill-hash
kraken.exe -m M -p "..." --ctx 64 --chunk 8 --greedy --prefill-hash
```

Both must print the same `prefill-hash` and `logits` values. A different
`prefill-hash` with equal `logits` is the shape a prefill-kernel regression
takes -- the drift is real but dies in the f16 residual before the head, which
is why the trace covers every row. `KRK_ATTN_BEXACT=0` forces the old tiled
kernel and must change `prefill-hash`; if it does not, the comparison is not
looking at the kernel.

---

## 6. Troubleshooting

These two are the same failure at two different depths. Fix the cause — the
compiler-ID probe in §2 — and neither appears.

**`Failed to find a default HIP architecture` at
`CMakeDetermineHIPCompiler.cmake:332`, preceded by
`The HIP compiler identification is unknown`.**
The probe compile of `CMakeHIPCompilerId.hip` failed, so CMake has no compiler
ID, so it falls back to reading the default arch out of the probe's output and
finds nothing. The real error is a few lines up in the log:
`clang++: error: cannot find ROCm device library`. Pass
`-DCMAKE_HIP_FLAGS="--rocm-device-lib-path=<rocm>/lib/llvm/amdgcn/bitcode"`.
Note the device library usually *is* present — it is not found because the
probe runs before `CMakeLists.txt` adds the path itself.

**Configure dies with `Invalid character escape '\R'` at
`build/CMakeFiles/<ver>/CMakeHIPCompiler.cmake:22`.**
This is the second-order symptom and it only appears when the configure limps
past the arch check — for example because a stale `CMAKE_HIP_ARCHITECTURES` is
already in the cache. With the HIP compiler ID unknown, CMake skips the primary
root-detection path (which normalises separators) and falls back to
`hipconfig --rocmpath`, which on Windows returns `G:\ROCM10RT-gfx1201` with
**backslashes** and no normalisation. That string is written verbatim into
`set(CMAKE_HIP_COMPILER_ROCM_ROOT "...")` and the generated file will not
parse. This repository's `CMakeLists.txt` now pre-flights the configure before
`enable_language(HIP)` and forces `CMAKE_HIP_COMPILER_ROCM_ROOT` and
`CMAKE_HIP_COMPILER_ROCM_LIB` to the forward-slash root it resolved, so this
escape path does not trigger in this tree any more; see the from-source recipe
in §2 and the troubleshooting detail in §6. If you hit it on a different repo,
pre-seed the two cache variables with forward slashes:

```
-DCMAKE_HIP_COMPILER_ROCM_ROOT=G:/ROCM10RT-gfx1201
-DCMAKE_HIP_COMPILER_ROCM_LIB=G:/ROCM10RT-gfx1201/lib
```

**`--info` says "HIP SDK not found" or the device line is missing.**
You have a CPU-only binary. The build configured without the SDK; re-run CMake
with `CMAKE_MODULE_PATH` / `CMAKE_PREFIX_PATH` pointed at the ROCm root and
confirm the `-- kraken: HIP backend enabled` line.

**`loaded HIP runtime is not the SDK this binary was built against`.**
A different `amdhip64_7.dll` was found first — either one staged next to the
exe, or one on `PATH`. Stage the SDK's own `amdhip64_*.dll`, `amd_comgr.dll`
and `rocm_kpack.dll` next to the binary to pin it (§2).

**A model loads but the text is fluent and wrong.**
That is the signature of a plausible-but-incorrect weight or attention path.
Compare against `KRK_WEIGHT_PULL=0` and `KRK_ATTN_BEXACT=0` before blaming the
model, then localise with `KRK_DUMP` + `tools/dump_diff.py`.

**A build "succeeded" but nothing changed.** A failed `ninja` leaves the old
`.exe` in place, so a run after `ninja rc=1` silently executes the previous
binary. Capture the exit status explicitly — piping ninja's output into
`grep`/`head` masks the failure.

---

## 7. Known limitations

- **Qwen3.5 / GDN GPU path** passes `coherence_check.sh` on this build: the
  device output is byte-identical to the scalar CPU reference at n=32/ctx=512
  *and* at n=128/ctx=2048 on `Qwen3.5-0.8B.Q4_K_M`. An earlier session recorded
  this as "not yet reproducible"; that is no longer the case, and the check is
  the way to re-confirm it on any machine:
  `bash scripts/coherence_check.sh models/Qwen3.5-0.8B.Q4_K_M.gguf`.
- **Prefill attention is `attention_bexact_kernel`** -- decode's arithmetic
  exactly (per-key running max, keys split across 8 warps, the same merge),
  over a tile of query rows. It replaced the query-tiled kernel, which is
  deleted along with `KRK_ATTN_QTILE`: the exact kernel is faster (+10% at 170
  prompt tokens, +113% at 3400, +49% where the head is 96 wide) and bit-identical
  to a decode step, so a chunked prompt and token-by-token prefill agree. The
  tiled kernel remains only as the fallback for a sliding-window layer, a
  non-causal call, and a head width outside 64..256 in steps of 32;
  `KRK_ATTN_BEXACT=0` forces it for an A/B.
- **`Spark_one.Q6_K.gguf`** passes `coherence_check.sh` on this build (device
  text identical to the scalar reference). It was previously recorded as failing
  at drift step 12; re-run to confirm on your machine:
  `bash scripts/coherence_check.sh G:/More-models/Spark_one.Q6_K.gguf`.
- **RDNA2 (gfx1031)** has no card on this machine, so device execution there is
  untested here — but the target is not merely "unconfirmed": build it with
  `-DKRK_GPU_TARGETS="gfx1031;gfx1201"` and the gfx1031 kernels are compiled and
  selected by the same numeric gate the gfx1201 path runs
  (`kraken-bench --gate`). What is missing is a card to execute on, not a build.
- **MoE models larger than VRAM** (20–37 GiB files) run, but at 3–21 tok/s.
  They are in the sweep table to show the loader refuses nothing silently, not
  as configurations to use.

Open work is tracked in [TODO.md](TODO.md); area status in
[STATUS.md](STATUS.md).

---

## 8. Flash attention and KV quant types

A note for anyone arriving with llama.cpp habits: **this repository has no CUDA
flash-attention code.** There is no `fattn.cu`, no `GGML_CUDA_FA_ALL_QUANTS`,
no `FLASH_ATTN_EXT` dispatch, and no KV-quant-type support gate to flip —
grep for those symbols finds them only inside a captured llama.cpp log under
`docs/traces/`.

The analogous question here is the HIP attention dispatch and which KV types
its kernels accept, in [src/hip/kernels/attention.hpp](../src/hip/kernels/attention.hpp)
and the `attention()` / `attn_fused_chain()` entry points in
[src/hip/backend_hip.hip](../src/hip/backend_hip.hip). If you want "warn when
the selected KV type has no fused kernel and falls back", that is where it
belongs — the `attention_qtile` / `attention_kernel` dispatch below is where a
"this KV type has no fused kernel, falling back" warning would go.

---

## 9. Where to read more

| document | contents |
|---|---|
| [README.md](../README.md) | full 66-file sweep, refusal table, usage examples, HTTP server |
| [ARCHITECTURE.md](ARCHITECTURE.md) | how the engine is put together |
| [HARDWARE.md](HARDWARE.md) | the card, measured bandwidth, what the ISA gives you |
| [MODEL-STATUS.md](MODEL-STATUS.md) | 25-model score table, coherence per model |
| [test-results.md](test-results.md) | measured prefill / token-gen / speculation / KV-tier numbers |
| [model-inventory.md](model-inventory.md) | 87 files, 45 runnable, per-file verdicts |
| [PERF-ANALYSIS.md](PERF-ANALYSIS.md) / [PERF-PLAN.md](PERF-PLAN.md) | where decode time goes, and the plan for it |
| [TODO.md](TODO.md) | known open items |
| [GPU_BRINGUP.md](GPU_BRINGUP.md) | bringing up a new GPU target |
| [RESEARCH.md](RESEARCH.md) / [REFERENCES.md](REFERENCES.md) | design notes and sources |
