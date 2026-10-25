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

### How to run it, in order

1. **Build.** Binaries land in `build-hip/` (`build/` for the CPU-only
   configure). `kraken-inspect` is in the build line on purpose: it answers
   from the same `krk_core` tables the loader uses, so one built before a new
   dequantizer landed reports `refused` for formats the engine already runs.
2. **Ask what the file is.** `kraken-inspect model.gguf` prints the verdict and
   the geometry, or the reason it is refused, by name.
3. **Ask what the device is.** `kraken -m model.gguf --info` prints the device,
   the measured aperture (the runtime's own ReBAR hint is wrong on this box),
   the expert budget and the full VRAM ledger. Read the ledger before believing
   any throughput number: it says where the memory actually went.
4. **Generate** on a short prompt first. A model that is going to produce
   nonsense at token 3 has already told you so at token 3.
5. **Measure** with `--bench`, which needs no flags and uses a fixed prompt.

Exit status: `0` ran, `1` the run failed (the message names the phase), `2` the
command line was wrong (`--prefill` given something other than `exact|fast`, for
instance). A refused GGUF is a load-time verdict, not a crash.

### Limits: what fits, what is refused, and what degrades

* **Your card, not the spec sheet.** `--vram-tier N` plans for a class (8, 12,
  16, 24 GiB) and clamps to what is installed, so `--vram-tier 12` on a 16 GiB
  box behaves as the 12 GiB one does. The box here reports 15.92 GiB, and its
  *measured* read bandwidth is 585 GB/s, not the 644 on the spec sheet.
* **The expert budget is a plan, and it is reported.** `--expert-cache-mb 0`
  (default) lets the run plan it, then clamp: it keeps a VRAM floor free
  (`KRK_VRAM_FLOOR_MB`, about 3.2 GiB here) for KV, workspaces and the
  allocator. When the routed set is larger than the budget the run says so
  (`the routed set is 15.4 GiB and the device budget is only 10.5 GiB (68%)`)
  and the cache churns; the WARM tier still holds the full set, so a miss
  promotes from RAM instead of re-reading the file.
  A budget the card cannot hold is **refused**, never trimmed: silently keeping
  less than you asked for would turn your experiment into a different one.
* **K/V is tiered, and a page that fits nowhere is DROPPED.** `--ctx` sets the
  capacity; the cache is split into HOT (VRAM), WARM (host RAM) and COLD (a
  spill directory). A page that fits in none of them has nowhere to go, so its
  layer is zero-filled and **the run decodes wrong text** rather than failing.
  That is why `--kv-warm-mb` defaults to covering every KV-carrying layer, and
  why a smaller value is honoured only when `--kv-cold-dir` exists to take the
  overflow. This is the one silent degradation in the engine; everything else
  refuses loudly.
* **`--kv-quant int8`** stores each row as bytes plus one f32 scale per 32
  values, 1.8x smaller. Both backends implement it, but MLA and the
  nextn/draft block are refused by name — the quant is a cache format, not a
  model property.
* **`--prefill fast` is the default and is not chunk-invariant.** It takes the
  rows>1 GEMM to the RDNA4 matrix cores (+58% chunk throughput, identical
  output), but the prefill *hash* then differs between chunk sizes. `--prefill
  exact` keeps every prompt row independent of `--chunk` and costs that 36%.
  `--prefill-hash` always uses exact, so the chunk-invariance gate still means
  something. Note `kraken-server` defaults the other way (`exact`).
* **Warmup needs a ranking file.** `--expert-warmup -1` (default) ranks only
  when the routed set exceeds VRAM, and ranks by `<model>.krakenexperts.json`,
  which `--expert-scan` writes. Without it the tiers fill in an even order.
  `--expert-warmup-ms` caps the phase (default 20000): the hottest experts go
  in first, so running out costs coverage, not the top of the list.
* **Speculation is opt-in and currently measured out.** `--mtp-spec` is greedy
  only and bit-identical to plain greedy output, but on the Lightning sidecar at
  `k=1` it lands at plain-decode parity even with the drafter cost removed. See
  [the measurement](#speculative-decode-mtp-sidecar--measured-out-at-k1).
  `--draft` with a small model and `--draft-head` (EAGLE-3) are the other two
  arms; `--draft-head` needs no extra VRAM.
* **`--hybrid-experts 1` loses as measured** (5.5 tok/s against 17.2 for
  promoting instead, at 5.4 ms of host time per expert). It buys memory, not
  speed; the host arm is a per-layer barrier the device waits on.
* **`--expert-stub` requires `--expert-scan`** and is refused rather than
  ignored on its own, because a flag that does nothing is a lie about the run.

### Every switch

Grouped by what you are changing. Defaults are the compiled-in ones.

**Model, input and process**

| switch | default | what it does |
|---|---|---|
| `--model FILE` | required | the GGUF to load |
| `--prompt STR` | interactive | prompt to complete; an empty value still counts as given |
| `--chat` | off | wrap the prompt in ChatML |
| `--system STR` | none | system message for `--chat` |
| `--max-tokens N` | 256 | tokens to generate |
| `--cpu` | off | scalar reference backend, for correctness work |
| `--device N` | 0 | HIP device index |
| `--tokenize` | - | print the prompt's token ids and exit |
| `--info` | - | model and device info, then exit |
| `--bench` | - | prefill/decode benchmark on a fixed prompt |
| `-v` / `-h` | - | verbose logging / help |

**Generation and sampling**

| switch | default | what it does |
|---|---|---|
| `--greedy` | off | force argmax decoding |
| `--temp F` | 0.8 | temperature, 0 = greedy |
| `--top-k N` | 40 | top-k cutoff |
| `--top-p F` | 0.95 | nucleus cutoff |
| `--min-p F` | 0.05 | min-p cutoff |
| `--repeat-penalty F` | 1.1 | repetition penalty |
| `--seed N` | 0 | RNG seed |
| `--sample-host` | off | argmax on the host from the downloaded row; costs ~1 MB per token, and is the correct path while the device argmax carries the Laguna race |
| `--debug-topk N` | 0 | dump the top-N candidates and log-probs per token |

**Context, prefill, K/V**

| switch | default | what it does |
|---|---|---|
| `--ctx N` | 4096 | KV capacity in tokens |
| `--chunk N` | 256 | prefill batch size |
| `--kv-quant f16\|int8` | f16 | K/V cache format; int8 is 1.8x smaller |
| `--prefill exact\|fast` | fast | fast uses the matrix cores but breaks chunk-invariance |
| `--fast-prefill` | - | old spelling of `--prefill fast`, kept for scripts |
| `--prefill-hash` | - | hash the last-position logits and exit; forces exact |
| `--kv-hot-mb N` | 0 = auto | KV HOT tier in VRAM |
| `--kv-warm-mb N` | <0 = geometry | KV WARM tier in host RAM |
| `--kv-cold-dir DIR` | unset | spill directory for pages evicted past WARM |

**Memory planning**

| switch | default | what it does |
|---|---|---|
| `--host-ram-mb N` | 0 = 1/4 installed | host memory to plan every host tier inside |
| `--vram-tier N` | unset | plan for a card class in GiB, clamped to the installed one |
| `--vram-cap-mb N` | 0 = policy | plan for this many MiB of device memory |
| `--ram-tier N` | 0 = installed | plan host tiers for 16/24/32/48/64/96 GiB |
| `--weight-vram-mb N` | 0 = auto | dense weights in VRAM; streams whole layers once they do not fit, negative disables |

**MoE residency and experts**

| switch | default | what it does |
|---|---|---|
| `--expert-cache-mb N` | 0 = auto | expert residency budget in MiB |
| `--expert-cache-slots N` | 0 = auto | cap resident (layer, expert) slots |
| `--expert-warm-mb N` | <0 = auto | WARM expert cache in host RAM (0 off); `--expert-l2-mb` is the old spelling |
| `--expert-warm-prefetch` | off | fill WARM at load instead of on demand |
| `--expert-warmup N` | -1 | ranked warmup: -1 when routed set exceeds VRAM, 1 force, 0 off |
| `--expert-warmup-ms N` | 20000 | ceiling for that phase (0 = none) |
| `--expert-pin-mb N` | 0 = off | pin the first N MiB of the tier in rank order |
| `--router-resident P` | 0 = off | send a near-tie (within fraction P) to the expert already in VRAM/WARM |
| `--expert-scan` | - | measure which experts routing picks; writes `<model>.krakenexperts.json` |
| `--expert-stub` | - | with `--expert-scan`: run only the routers, skip every FFN |
| `--expert-adapt` | - | fold this run's routing into the index at exit |

**Hybrid host/device experts**

| switch | default | what it does |
|---|---|---|
| `--hybrid-experts 1` | off | compute overflow experts on the host out of the mapping |
| `--hybrid-threads N` | 0 = min(hw, 8) | host threads for that split |
| `--hybrid-max-rows N` | 8 (0 unbounded) | widest row count the host arm runs at |
| `--hybrid-frac F` | 1.0 | share of the device tier's overflow the host takes |

**Speculation**

| switch | default | what it does |
|---|---|---|
| `--draft FILE` | none | small draft model for greedy speculative decoding |
| `--draft-tokens N` | 4 (1 for `--mtp-spec`) | proposal window per round |
| `--draft-head FILE` | none | EAGLE-3 head from `tools/eagle_to_gguf.py`; no extra VRAM |
| `--draft-head-prime N` | 256 (0 = whole prompt) | positions a round primes the head over |
| `--mtp FILE` | none | nextn (MTP) block in a separate GGUF, for `-noMTP` trunks |
| `--mtp-probe` | off | run the block one step and print what it drafts against the target |
| `--mtp-spec` | off | decode with the block as the drafter (greedy only) |

**Diagnostics and tracing**

| switch | default | what it does |
|---|---|---|
| `--profile` | off | per-op device timeline for the last decode step, with the idle gap before each op |
| `--trace [PATH]` | off | Chrome-trace of host functions, backend ops, device spans, transfers and syncs |

`kraken-server` has its own, smaller set: `--host` (0.0.0.0), `--port`
(`$KRK_PORT`, else 8080), `--ctx`, `--chunk`, `--kv-quant`, `--prefill`
(**default exact**, unlike the CLI), `--threads`. It serves an
OpenAI-compatible `/v1/chat/completions`.

### Examples worth keeping

```sh
# smallest useful check before a long run
kraken -m model.gguf --info

# reproduce the benchmark exactly
kraken -m model.gguf --bench --greedy

# a long-context run that will not fit the KV in VRAM
kraken -m model.gguf --ctx 32768 --kv-hot-mb 2048 \
    --kv-warm-mb 8192 --kv-cold-dir ./kvspill -p "..." --greedy -n 256

# a MoE whose routed set is bigger than the card: rank, then pour the budget in
kraken -m big-moe.gguf --expert-scan -p "representative prompt" -n 128
kraken -m big-moe.gguf --expert-cache-mb 9000 --expert-warmup 1 --greedy -p "..."

# prove a forward is chunk-invariant (same hash at --chunk 1 and 8)
kraken -m model.gguf -p "..." --prefill-hash

# find where a token's time actually goes
kraken -m model.gguf -p "hi" -n 6 --greedy --profile

# a trace to open in Perfetto
kraken -m model.gguf -p "hi" -n 8 --greedy --trace ./tok.json

# the server
kraken-server -m model.gguf --port 8080 --ctx 8192
```

---

## Benchmarks

### The command

The headline table below was re-measured on the current tree on 2026-10-24:
three `--bench --greedy` runs per model, same machine and command shape. The
large-model first load was cold; subsequent loads used the warm file cache. The
full sweep later in this section is a one-run snapshot of every listed file
still present in `G:/More-models` or `H:/OLLAMA-Models/GGUF`. Dated sections
below (worked examples, historical A/B tables) name the files they were measured
on; several of those files have since left the box, and the two tables above are
the re-checked set.

```
kraken --model <MODEL> --bench --greedy
```

`--bench` is self-contained and needs no flags: it feeds a fixed prompt (which
tokenizes to **51-125 tokens depending on the model's tokenizer** - 55 for
most Qwen-family files, 122-125 for a few others), generates
up to **256 tokens** (generation stops at `--ctx`), and reports the *whole*
process budget, not just the two phases it owns. Defaults in force for every
row: greedy sampling, `--ctx 4096`, `--chunk 256`. Generation stops at the
context limit, so rows with a short `--ctx` can produce fewer than 256 tokens.
The per-model prompt length is listed in the sweep table below.



```
# Qwen3-8B, current-tree sample (run 1; cold file cache)
load           4060 ms, 4.68 GiB uploaded of 4.68 GiB weights
prefill        55 tokens in 342.3 ms  (160.7 tok/s)
decode         256 tokens in 2801.0 ms  (91.0 tok/s)
```

The cold-load row is intentionally kept as an example of why cache state is
recorded; the two warm-cache loads were 1333 and 1301 ms. See the dated
context check below for a same-build context comparison.

- **prefill** - prompt tokens / time to process them. Throughput, so higher is
  better and it scales with batch size.
- **decode** - generated tokens / time to generate them. The context grows
  during the run, so this is the *average* over 256 tokens ending at ~300
  tokens of context, not a short-context number.
- **load** - reading weights into the selected backend. It is the largest
  single phase on these files and 1.4-3.8 s on the 8B, depending on file cache
  state; it is printed rather than assumed.

Configuration: **AMD Radeon RX 9070 XT** (gfx1201, RDNA4, wave32, WMMA gfx12,
64 CU, 15.9 GiB VRAM), Windows 11, ROCm 10.1, clang 23.0.0. The binary
reported HIP 7.16.26332 (SDK 7.16.26331); see the runtime warning and limitation
below.

### Results (current tree, 3 runs each; range shown)

| model | layers / embd | prefill tok/s | decode tok/s | load ms | vs 2026-10-24 |
|---|---|---:|---:|---:|---|
| Qwen3-4B Q8_0 | 36 / 2560 | 321-325 | 104.9-105.3 | 1279-2190 | prefill +76%, decode +5% |
| Qwen3-8B Q4_K_M | 36 / 4096 | 160.2-160.9 | 91.0-91.1 | 1301-4060 | prefill +59%, decode +3% |
| Qwen3.5-9B Q4_K_M (GDN hybrid) | 32 / 4096 | 150.1-150.5 | 84.9-85.3 | 1429-2335 | prefill +47%, decode +4% |
| Qwen3-30B-A3B i1-Q2_K | 48 / 2048 | 6.7 first-touch / 20.1-22.9 warm | 17.7 first-touch / 37.1-37.2 warm | 512-1081 | decode **-8%** warm |

Re-measured 2026-10-25 on the current build, three sequential runs per model,
same fixed `--bench` prompt, 256 greedy output tokens, `--ctx 4096`,
`--chunk 256`; the ranges are run-to-run, not confidence intervals. Both
measured columns are compared against the 2026-10-24 table.

The prefill column moved up on **every** row with no code change on the
`--prefill exact` path in between (the session's kernel work is all on the
`--prefill fast` WMMA rung), so read the older prefill column as measured in a
lower-clock/cold state, not as a kernel baseline: prefill is a short compute
burst that tracks clocks, decode is long and memory-bound and moved 2-5%.
Decode is the stable column and the one to compare across dates.

These four replace the previous headline: Qwen3-8B is
the one row that survived, because its file is still on the box, and the other
three take the places of SmolLM2, Qwen3.5-0.8B and the 4x0.6B MoE, none of which
is in either model tree any more. Qwen3-4B and Qwen3.5-9B are the nearest
dense/hybrid files that are present; Qwen3-30B-A3B is the MoE.

The 30B MoE ran with the default adaptive policy and no expert index. Its last
two runs were the warm state (**20.1-22.9 prefill / 37.1-37.2 decode**); the
first run, right after the file had left the page cache, is **6.7 prefill /
17.7 decode**, and that is what a cold request pays — a 2.7x spread, which is
why the sweep below marks its MoE rows cold → warm rather than with one number.
The 8B's first load (4.06 s) had a cold file cache; the next two (1.33 and
1.30 s) were warm. Throughput is the more stable comparison here; do not
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

### Speculative decode (MTP sidecar) — measured out at k=1

Measured 2026-10-26 on the Lightning 30B-A3B NVFP4 pair, 32 greedy tokens,
`--ctx 2048`, draft window `k=1`. Same command shape as everywhere else, plus
`--mtp <sidecar> --mtp-spec`; `KRK_VRAM_FLOOR_MB=512` was needed to fit the
block's expert acquires on this box.

```
M="G:/More-models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf"
S="H:/OLLAMA-Models/GGUF/mtp-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf"
kraken -m "$M" --mtp "$S" --mtp-spec --greedy -p "<prompt>" -n 32 -c 2048
```

| prompt | plain | spec k=1 | block free (`KRK_MTP_REPLAY_IN`) |
|---|---:|---:|---:|
| "The quick brown fox jumps over the lazy dog and then it runs into the" | 40.5 tok/s (24.7 ms/step) | 31.2 (53.92 ms/round) | 40.7 (41.35 ms/round) |
| "def fibonacci(n):" | 36.8 (27.2) | 26.5 (63.50) | 36.3 (46.45) |
| "List the first ten prime numbers in order:" | 38.0 (26.3) | 27.2 (56.06) | 37.5 (40.60) |

The head accepts 73.7% of its proposals at k=1 and still loses, and the reason is
structural rather than the head: every round pays **two target passes** (one
batched verify row plus the forward for the token the round just emitted) to
get 1.68 tokens, i.e. 1.19 passes per token against plain decode's 1.00. The
third column is the ceiling of any propose-pass optimisation — the block never
runs — and it lands on parity, not on a win. The round decomposes as verify
17.7 ms + second forward 25.4 ms + block re-feed 8.5 ms + propose 3.4 ms. The
arm therefore stays opt-in; `--mtp` alone only loads the block for the probe.

The full phase accounting is in [docs/STATUS.md §13](docs/STATUS.md) and the
[campaign log](docs/CAMPAIGN-LOG-MOE-2026-10.md).

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
117 tok/s**. The current-tree 8B result is **91.0 tok/s (11.0 ms/token), ~78% of
that ceiling**, averaged over the 256 generated tokens above. Decode is flat in
context from roughly 256 tokens up (docs/PERF-PLAN.md B0); the ctx-64 row below
is lower because that run is short enough for pipeline fill to be a real share
of it. The historical context table below compares pre-fix and fixed
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
kraken --model "G:/More-models/Qwen3-4B-Q8_0.gguf" --bench --greedy
kraken --model "G:/More-models/Qwen3-8B-Q4_K_M.gguf" --bench --greedy
kraken --model "H:/OLLAMA-Models/GGUF/Qwen3.5-9B-Q4_K_M.gguf" --bench --greedy
kraken --model "G:/More-models/Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K.gguf" --bench --greedy
```

Across these sequential repetitions, decode varied by about 0.2% (4B), 0.4%
(8B), 0.4% (9B) and 1.0% (30B); these are observed ranges, not confidence
intervals. The prefill ranges are wider because the prompts are short. The MoE result uses the current adaptive policy,
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

### Full sweep: every listed file still on the box (2026-10-24, one run each)

The 2026-10-10 sweep tried 66 `.gguf` files and 38 ran. This revision re-ran
every one of those 38 that is still on the box: **26 rows below**. The 12 that
were removed because the file is in neither `G:/More-models` nor
`H:/OLLAMA-Models/GGUF` (checked 2026-10-24): both SmolLM2 files,
Qwen3.5-0.8B, the 4x0.6B MoE, `qwen3.5-4b-nvfp4`, `Qwen3-4B-Instruct-2507-Q6_K`,
both 1.7B f16 files, `qwen3.5-9b-nvfp4`, `Q3.5-9B-GLM-5.1-DA Q4_K_S`,
`Qwen3.5-9b-Sushi-Coder-RL Q4_K_M` and `ornith-35b-Q8_0`.

**Re-measured 2026-10-25** on the current build. Each dense row is **one**
`--bench --greedy` run, in file-size order, cold for that file. The MoE rows
(`*`) were run twice because they are bimodal — the first run after the file
leaves the page cache pays expert read-through, the second promotes from the
host WARM tier — so those are shown **cold → warm** and a single number would
misreport them by up to 4x. Treat any two rows closer than 2-5% on prefill as
noise.

The last column is the move against the 2026-10-24 table, shown only where it
exceeds 5% (`prefill% / decode%`). It is the honest shape of this refresh:
**decode is flat** on almost every row (the one exception is the Ornith-35B
MoE at -9%), while **prefill rose on every row** with no exact-path code change
in between — a short compute burst tracking clocks, versus a long memory-bound
decode. Read the older prefill column as a cold-clock measurement, not as a
kernel baseline. Decode is the comparable column; prefill for a real request is
the 4081-token table above.
Dense rows are VRAM-bound decode; MoE rows (marked \*) page experts through the
default budget, so their prefill includes cold-cache paging.

Every row runs the default `--prefill exact`. The same 55-token 8B prompt reads
**923-925 tok/s on `--prefill fast`** against 160.7 exact (decode is identical,
91.0 either way), so do not compare a prefill column across arms (or across
these two sweeps) without checking which one produced it.

| file | size | arch - layers / embd | prompt tok | prefill tok/s | decode tok/s | load ms | move |
|---|---|---|---|---|---|---|---|
| smolcode-coder-cpp-1.5b-q4_k_m | 1.12 GB | qwen2 - 28 / 1536 | 55 | 658.9 | 260.4 | 1302 | +62 / +19 |
| smolcode-coder-cpp-3b-q4_k_m | 2.10 GB | qwen2 - 36 / 2048 | 55 | 379.4 | 173.2 | 1718 | +60 / — |
| Phi-3.5-mini Q4_K_M | 2.32 GB | llama - 32 / 3072 | 125 | 303.3 | 108.0 | 1932 | +52 / — |
| qwen3vl-4b-q4_k_m | 2.50 GB | qwen3vl - 36 / 2560 | 55 | 282.5 | 138.8 | 2673 | +68 / +9 |
| Spark_one.Q6_K | 2.54 GB | qwen2 - 36 / 2048 | 55 | 232.0 | 130.5 | 1631 | +65 / +6 |
| Samastam-2.5B-Q8_0 | 2.69 GB | llama - 28 / 2048 | 122 | 569.6 | 163.2 | 2325 | +73 / — |
| VibeThinker-3B.Q8_0 † | 3.29 GB | qwen2 - 36 / 2048 | 55 | 452.9 | - (no decode; warmup-token artifact on short prompt) | 2999 | +73 / — |
| qwen2.5-coder-3b-instruct-q8_0 | 3.62 GB | qwen2 - 36 / 2048 | 55 | 449.1 | 130.1 | 2189 | +60 / +6 |
| Qwen3-4B-Q8_0 | 4.28 GB | qwen3 - 36 / 2560 | 55 | 324.4 | 105.9 | 2147 | +76 / +6 |
| Opus4.7 Distill 4B Q8_0 | 4.48 GB | qwen35 - 32 / 2560 | 55 | 331.5 | 103.1 | 3235 | +77 / +5 |
| Qwen3.5-9B-DeepSeek-V4-Flash-MTP Q3_K_M | 4.74 GB | qwen35 - 32 / 4096 | 55 | 118.8-124.7 | 81.3-81.5 | 1866 | +46 / — |
| Ornith-1.0-9b STRIX_LEAN | 4.96 GB | qwen35 - 32 / 4096 | 55 | 192.4 | 102.8 | 5074 | +61 / +5 |
| Qwen3-8B-Q4_K_M | 5.03 GB | qwen3 - 36 / 4096 | 55 | 160.7 | 91.0 | 1300 | +59 / — |
| mythos-9b-unhinged-heretic.i1-Q4_K_M | 5.03 GB | qwen3 - 36 / 4096 | 55 | 162.1 | 91.0 | 3731 | +61 / — |
| lacuna-v1-Q4_K_M | 5.07 GB | llama - 32 / 4096 | 52 | 163.2 | 93.2 | 4446 | +63 / — |
| qwen35-9b-instruct-nvfp4 | 5.31 GB | qwen35 - 32 / 4096 | 55 | 166.2 | 91.7 | 4750 | +59 / — |
| Qwen2.5-Coder-7B Q5_K_M | 5.44 GB | qwen2 - 28 / 3584 | 55 | 107.3 | 86.6 | 4114 | +43 / — |
| omnicoder-9b-q6_k | 7.36 GB | qwen35 - 32 / 4096 | 55 | 87.8 | 63.4 | 3777 | +34 / — |
| Qwen3.8-9B-Q6_K | 7.56 GB | qwen35 - 32 / 4096 | 55 | 87.7 | 63.4 | 8024 | +34 / — |
| Qwen3-30B-A3B Q2_K \* | 11.26 GB | qwen3moe - 48 / 2048 | 55 | 6.7 → 22.9 | 17.7 → 37.2 | 527-1393 | warm +300 / +124 |
| Qwen2.5-Coder-32B Q4_K_M ‡ | 19.85 GB | qwen2 - 64 / 5120 | 55 | 31.4 | 3.6-3.8 | 18563 | **+253 / +270** |
| qwable-v1-mxfp4_moe \* | 20.26 GB | qwen35moe - 40 / 2048 | 55 | 9.9 → 105.3 | 17.2 → 28.6 | 25327 | warm +1655 / +147 |
| laguna-xs2-Q4_K_M \* | 20.27 GB | laguna - 40 / 2048 | 51 | 17.4 → 79.6 | 25.5 → 35.5 | 22381 | warm +1402 / +155 |
| Unsloth-Ornith-1.5-35B-A3B Q4_K_XL \* | 22.36 GB | qwen35moe - 40 / 2048 | 55 | 65.9-68.0 | 27.6-28.8 | 21942 | +65 / **-9** |
| Tiel-Coder-35B-A3B Q5_K_XL \* | 26.59 GB | qwen35moe - 40 / 2048 | 55 | 9.8 → 50.0 | 15.5 → 25.1 | 22468 | warm +843 / +109 |
| Tiel-Coder-35B-A3B-MTP-APEX \* | 26.67 GB | qwen35moe - 40 / 2048 | 55 | 15.5 → 56.6 | 25.2 → 29.0 | 22463 | warm +107 / +34 |

† The fixed `--bench` prompt tokenizes to 55 tokens here, prefill runs fine (452.9
tok/s), and the GPU backend is fine - `--bench` still emitted 0 decode tokens.
With a normal prompt (`x` -> 1 tok, forced decode) the same model decodes at
~124 tok/s, so the 0.0 is a prompt-training-format artifact, not a kernel or MoE
path defect. The row is kept for prefill honesty; treat its decode column as "no
decode measured", not a throughput number.
‡ The 32B dense model is larger than VRAM (18.5 GiB on a 15.9 GiB card). With
the residency planner it now keeps a resident core and streams the rest: **3.6-3.8
tok/s decode and 31.4 tok/s prefill**, from **1.0 and 8.9** a day earlier — the
largest single move in this table, and the clearest evidence that layer
residency plus a staged stream beats wholesale oversubscription. It still costs
a 18.6 s load and 67 s per 256 tokens, so it is a reachable configuration rather
than a comfortable one.

NVFP4 and MXFP4 dequantize fine (the 9B NVFP4 and the MXFP4 MoE rows above);
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

**qwen4exp is no longer in this table (2026-10-22).** The family's graph, its
hyper-connection residual, its PLE block and its QSA indexer all ran; what kept
it refused was that no suite gate pinned the whole graph's output rather than
one block of it. `test_qwen4exp_whole_graph` now does — the last prompt row's
entire logits vector by hash, its argmax, the greedy sequence, and four arms
that keep the pin from being vacuous (a different prompt moves it, chunk 1
reproduces it, a shifted PLE table breaks it, and the fixture loads with
`KRK_ARCH_ALLOW_EXPERIMENTAL` unset). With no hatch, the 78 GiB REAP-288 file
loads as `supported (recurrent-moe)` and decodes `def add(a, b):` ->
`    return a + b` at 5.2 tok/s prefill / 5.8 tok/s decode, expert-tier bound.

### Refused at load (on the sweep build; each with its reason)

| reason | files |
|---|---|
| Speculative-head shard, not a model (runs only via `--draft`) | laguna-xs21-dflash-q8, laguna-s-2.1-DFlash-Q4_K_M, Qwen3.8-27B-DFlash2-Q2_K, Qwen3.8-27B-DFlash2-Q4_K_M, Qwen3.8-27B-DSpark-Q8_0, Qwen3.8-27B-DFlash-bootstrap-Q8_0, DeepSeek-V4-Flash-REAP-MTP, Ternary-Bonsai-27B-dspark-Q4_1 |
| MLA: the legacy lite layout (V2-Lite) RUNS (576-wide latent cache row, absorbed scores, rope on the decoupled key); the split (V3 class) and shared-latent (V4) layouts are different graphs and are refused at init BY NAME | GLM-4.7-Flash-Q4_K_M (deepseek2, split layout), DeepSeek-V4-Flash-imatrix (deepseek4, shared latent) |
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
kraken --model "G:/More-models/Qwen3-8B-Q4_K_M.gguf" -p "The capital of France is" \
       --ctx 64  -n 256 --greedy -v
kraken --model "G:/More-models/Qwen3-8B-Q4_K_M.gguf" -p "The capital of France is" \
       --ctx 256 -n 256 --greedy -v
```

`-v` prints `[N prompt / N generated | prefill X ms | decode Y tok/s]` after the
answer. On this build, three runs at ctx 64 measured 78.3-78.7 tok/s (60
generated tokens before reaching the context limit); three at ctx 256 measured
85.7-86.1 tok/s (252 generated). Decode is slightly lower at the shortest
window, not flat: the first steps of a run are still filling the pipeline, and
at ctx 64 they are a larger share of a much shorter run. The 256-token
`--bench` average sits between them (88.0), but is a distinct run whose context
grows during generation. The older table below records the before/after kernel
fix from a prior build; use these measurements for current absolute rates.

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

- `--prefill exact|fast` - `exact` (the default) keeps every prompt row's
  arithmetic independent of `--chunk`: the rows>1 GEMM runs the chunk-invariant
  exact grid, which is also why prompt throughput is not what the card can do.
  `fast` takes those GEMMs to the RDNA4 matrix cores -- **307 -> 1707 tok/s** on
  a 1145-token prompt through Qwen3-4B-Q8_0, 171 -> 326 on the 30B-A3B -- at the
  price of it being a different-arithmetic mode: the generated text changes and
  the prefill hash is no longer chunk-invariant past 8 rows per chunk. Numbers
  and the device split: `docs/PERF-ANALYSIS.md` section 12s.

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

### Prefill, measured on this box (2026-10-24)

Four thousand and eighty-one tokens of the same prose, `--greedy`, `--ctx 32768`,
`--expert-warmup 1`, one run per cell. These are the prompt-throughput numbers a
long-context request actually pays; the 55-token `--bench` prompt is dominated by
fixed cost and reads much lower.

| file | arm | chunk | prefill tok/s | prefill ms | decode tok/s |
|---|---|---|---|---|---|
| Qwen3.8-Distill-35B-A3B Q2KXL (11.7 GiB) | exact | 256 | 181.8 | 22 450 | - |
| Qwen3.8-Distill-35B-A3B Q2KXL | fast | 256 | **402.4** | 10 140 | - |
| Qwen3-30B-A3B abliterated i1-Q2_K (11.2 GiB) | exact | 64 | 63.7 | 64 043 | 2.0 |
| Qwen3-30B-A3B abliterated i1-Q2_K | exact | 256 | 176.3 | 23 143 | 29.2 |
| Qwen3-30B-A3B abliterated i1-Q2_K | exact | 1024 | 176.7 | 23 096 | 29.0 |
| Qwen3-30B-A3B abliterated i1-Q2_K | fast | 64 | 77.3 | 52 809 | 2.0 |
| Qwen3-30B-A3B abliterated i1-Q2_K | fast | 256 | 384.4 | 10 618 | 31.0 |
| Qwen3-30B-A3B abliterated i1-Q2_K | fast | 1024 | **420.9** | 9 695 | 29.5 |
| Qwen3-Coder-53B-A3B i1-Q4_K_M (29.9 GiB) | exact | 256 | 49.2 | 82 900 | 3.4 |
| Qwen3-Coder-53B-A3B i1-Q4_K_M | fast | 256 | 65.5 | 62 300 | 3.4 |

Three things this measures, rather than argues:

- **`--prefill fast` is worth 2.2-2.4x on a long MoE prompt** at chunk 256+:
  181.8 -> 402.4 on the 35B, 176.3 -> 384.4 on the 30B, 384.4 -> 420.9 going from
  chunk 256 to 1024 with `fast` on. It is a different-arithmetic mode (see the
  `--prefill` bullet above), so it is not the default and a chunk-invariance gate
  must stay on the exact arm.
- **Chunk 64 is not a small chunk, it is a different machine.** 63.7 tok/s
  prefill and 2.0 tok/s decode against 176.3 / 29.2 at chunk 256, on both arms and
  therefore not a kernel choice: a 64-row prefill leaves the batched expert path
  below the width where it batches. Keep `--chunk` at 256 or above.
- **The 53B is expert-tier bound, not kernel bound.** 29.9 GiB of routed set
  against the card's 6.4 GiB pool is 3.4 tok/s whatever the prefill arm does; the
  `fast` arm only helps the part that is not waiting on experts (+33% prefill,
  decode unchanged).

Decode on the 35B with the fixed 55-token `--bench` prompt is 34.0 tok/s; the
long-prompt decode column above is a different run (`-` on the 35B rows because
that prompt's first sampled token was EOS at 32 tokens, so there were no timed
steps to report).

`kraken-server` takes the same switch (`--prefill exact|fast`, default `exact`),
so the arm is reachable from the HTTP path and the web UI. Measured through the
API on the 30B-A3B file, same 4081-token prompt, chunk 1024, `--ctx 32768`:
**22.98 s exact (177.6 tok/s) against 9.89 s fast (412.5 tok/s), 2.3x**. The
server prints one line at startup when the fast arm is engaged, because a run
whose output is no longer chunk-invariant should be identifiable from its log.

### `--kv-quant f16|int8` - a quantized K/V cache

The cache stores `kv_dim` bytes per position per plane instead of `kv_dim * 2`,
followed by one f32 scale per 32 values, so a row is
`kv_dim + 4 * ceil(kv_dim / 32)` bytes: **144 against 256 bytes at a 128-wide
GQA row (1.78x)**, and 144 against 512 on the CPU reference's f32 cache (3.6x).
Quantisation is per row per 32, which is what keeps it chunk-invariant: the same
activation produces the same bytes whether it arrived in a one-token decode or a
64-row prefill.

Both backends implement it. On the reference backend one decode row at the end
of a **200-key** context lands within **0.71%** (relative L2) of the f32 cache,
worst channel 0.0037, and the layout-mismatch arm reads an f32 cache through the
int8 descriptor and must diverge by more than 20%, so the flag cannot pass by
being ignored. On the card the four readers (`attention_kernel`,
`attention_bexact_kernel`, `attention_decode_kernel`,
`attention_decode_split_kernel`) and `kv_append_i8_kernel` speak the layout, and
the cache is invariant end to end: `smolcode-coder-cpp-1.5b-q4_k_m`, a 21-token
prompt, `--prefill-hash` gives

    chunk 1, 2, 4, 8 and 16   trace 6d4758c0c4754be0  logits dc6d7126d56225fb  first 576

against `f16`'s `2cb4e7fc46508835` / `67f95a7185846df2` and the same first token
(the cache holds different numbers, so the two formats must differ; what must
not move is the hash across chunk sizes). 24 greedy tokens are byte-identical
text between the two formats. MLA and the nextn/draft block are still refused by
name: they own their own caches and have no int8 path.

**The device arm is `kraken-bench --bitexact`'s int8 block, and it is enforced,
not printed.** Two claims, measured per kernel because they fail for different
reasons: the CACHE BYTES of a position must not depend on the append's arity
(one 8-row append against eight single-row appends, byte-exact), and ATTENTION
over that cache must not depend on the query-row count (decode rows against one
prefill call). **27 of 27 arms identical** across hd 64/128/256, rows 2/4/8 and
64/200 keys. `KRK_ATTN_BEXACT=0` (the tiled prefill reader) makes **17 of 27**
differ by **1 ulp**, which is the arm showing it can fail; `--bitexact` exits
non-zero when any int8 arm differs, while the f16 rows in the same report stay
informational.

**The defect that closed it was a reduction order, not a layout.** The int8
decode lane computed its dot as a per-element f32 FMA chain
(`s += q[e] * dequant(k[e])`) while the tiled and bit-exact prefill consumed
dequantised PAIRS through one `v_dot2_f32_f16`. Same terms, different rounding:
a chunk-1 decode row sat up to one f16 ulp per channel away from a chunk-8
prefill over **identical cache bytes**, so chunk 2 and 4 agreed with each other
and disagreed with 1 and 8. Decode now builds the same pair and calls the same
instruction (`attn_lane_dot_q8`), which is why the five rows above are one hash.

The two defects found before it read as working code and are worth keeping:

- **The scales plane is at the ROW base + `kv_dim`, not at the KV-head slice.**
  The row pointer was built as `cache + hkv * hd` before being handed to the
  dequant, so every head after the first read another head's scales, i.e. a
  scale of a different magnitude. The symptom was not a crash but `!!!...`: the
  first token came out as `<unk>` and every later one followed it. The head's
  channels are an INDEX offset into the row, not a pointer offset.
- **The dequant has to round through f16.** The tiled and bit-exact kernels
  stage a dequantised value into `_Float16` shared memory and dot from there; a
  decode row computes the same product in f32. Without the round trip the two
  paths differ by up to one f16 ulp per channel, which is enough to move the
  prefill trace while the cache bytes are identical.

**This is the 128k lever, and it is measured now.** `qwen3.8-flash-next-reap-288`,
`--ctx 131072 --chunk 256 --expert-cache-mb 8000 --expert-warm-mb 20000`, both
arms the second load of the file so the page cache is warm (a first-load int8
run read 2.53 / 2.02 tok/s and would have made the format look 20% slower than
it is -- compare load states or do not compare):

| 128k, same flags | f16 | int8 |
|---|---|---|
| KV at ctx 131072 | 3072 MiB | **1728 MiB** (1.78x smaller) |
| expert pool at init | 4697 MiB | **5747 MiB** (+1050, +22%) |
| pool after 5+24 then 5+64 tokens | 3232 MiB | **4295 MiB** (+1063, +33%) |
| decode, 24 tok / 64 tok | 3.16 / 2.16 tok/s | 3.26 / 2.24 tok/s |
| working set | 3.5 GiB | 3.4 GiB |

Two things the table does not say. **The ratchet was still running in that
measurement**: the int8 pool decayed 5747 -> 4295 MiB over 93 tokens (and the
earlier long-chat run fell 4697 -> 359). It is fixed below, and the same 128k
int8 run now prints no re-clamp at all and holds 5747 MiB; the table is kept as
measured, before the fix, because it is the f16/int8 pair. And `unaccounted` rose
2016 -> 2310 MiB with the format, so ~294 MiB of the 1344 MiB the KV gave back
went to allocator slack; the +1050 MiB is net.

Decode on this file is not expert-bound at either context. With `KRK_TIME=1` a
step is `decode.step` 307 ms, of which `moe.route.fetch` (2.98 ms x 48 layers)
plus `moe.route.rest` (2.62 ms x 48) is 269 ms, while `moe.experts` is 0.003 ms
per call and `expert.warm.read` is 23 ms/step. The same table at `--ctx 4096`
reads 307.6 ms/step and 3.25 / 2.38 tok/s, so **the declared context is not what
makes 128k feel slow** -- the router readback is. A span that waits on the queue
attributes the wait to itself, so that 143 ms/step is "wait for this layer's
router row", not 143 ms of copy; it is the next thing to instrument.

### The expert-cache budget ratchet (fixed)

The re-clamp exists because the init-time plan cannot see the KV tier and the
workspaces: `reclamp_expert_budget` re-derives the pool's ceiling from a live
free-VRAM reading every prefill and every 16 decode steps, and only ever lowers
it. The old arithmetic was

    room = free > floor ? free - floor : 0
    live = bytes_ - pin_bytes_
    cap  = live + room

`free` is a device-wide reading, and a cell this cache evicts goes back to the
BACKEND's pool (`release_pooled_batch`), not to the driver, so `hipMemGetInfo`
does not move when the cache frees a slot. On this box free VRAM sat at
**3235 MiB against a 3261 MiB floor**, so `room` was zero at every call and
`cap == live`: the budget became whatever happened to be resident. Every
eviction therefore lowered the ceiling by the bytes it had just freed, and there
was no path back up, because growth is exactly what `room` describes.

Measured, not inferred: **5747 -> 4295 MiB over 93 tokens** on the 288-expert
file, **4697 -> 359 MiB** on a longer chat, and **6675 -> 1313 MiB across eleven
decode checks** on the 35B file, every line carrying the same free reading. The
cost is where the bytes went: a run whose decode split read `room 1211.9 |
promote 1303.7 ms` of a 29.4 ms step is promoting back the experts the pool no
longer holds. Text is identical either way (`--prefill exact`, same greedy
stream); this was a throughput and VRAM-planning defect, not a correctness one.

The fix is causal rather than a floor or a hysteresis: **cut only by what the
card gave up since the previous reading.**

    taken = last_free > free ? last_free - free : 0
    cut   = min(taken, budget - (live + room))

The pool's own room-making leaves `taken` at zero -- free did not move, because
the cells are pooled -- so the clamp is a no-op; a KV tier taking VRAM lowers
`free` with residency unchanged, and that drop is exactly what the pool gives
back. The 1/64 margin still absorbs a reading that wanders by a few MiB, and
`clear()` resets the baseline with the residency generation.

Verified two ways. `test_expert_cache_policy` grows to 380 checks with five for
this rule: the baseline, an evicting pool that must not cut, an external drop
that must cut by exactly that much, a repeat that must not cut twice, and free
above the floor that must not move the budget. The red construction (the old
unconditional cut) fails all five, **375/380**. On the card, the same 128k int8
run that printed `5747 -> 4295` now prints **no re-clamp at all** across the same
two requests and holds 5747 MiB.

---

## Usage switches and worked examples

Every switch with its default is in [Every switch](#every-switch) above. These
are the ones that change what you *see* rather than what the model does.

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

### `--weight-vram-mb` - dense weight streaming

Every dense matrix used to be uploaded at load. When the model did not fit the
card, that filled VRAM to **0.00 GiB free** and left the KV tier a single slot
per plane: on `Qwen3.8-27B-Q4_K_M` (15.9 GiB) it measured 7.9 GiB of KV
migrations in 16 tokens and **3.1 tok/s**, because the KV working set could
never stay resident.

The loader now keeps the layers that fit and streams the rest from the GGUF
through **one reusable ring** before their layer runs. The decision is made
from the file's own tensor directory before any weight is uploaded, and the
ring is sized to the largest streamed layer. The policy is:

* `--weight-vram-mb N` - keep at most N MiB of dense weights on the device.
* `0` (default) - automatic: everything resident while it fits, stream the tail
  only when it does not. The auto budget is `free VRAM - reserve - 1 GiB`, so
  the KV cache and workspaces are allocated against real headroom instead of
  against zero.
* A negative value, or `KRK_WEIGHT_STREAM=0`, disables streaming (the old
  upload-everything path). `KRK_WEIGHT_VRAM_MB` is the environment spelling of
  the positive budget.

Two more policies apply once streaming is active, both of them measured rather
than guessed:

* **Promotion on first use.** A streamed layer that measured free VRAM can hold
  is copied into an allocation of its own the first time it runs and never
  staged again (`KRK_WEIGHT_PROMOTE=0` on the layer's environment restores the
  fixed resident prefix). The load-time budget is decided before the KV cache and
  the workspaces exist, so it leaves real room unused: `--info`-style residency
  lines print each promotion with the MiB it kept free.
* **A two-buffer read pipeline.** Two pinned host buffers alternate: the copy is
  issued as a DMA straight out of the pinned buffer on a private stream
  (published to the default stream by one event), and the next layer is read
  while those bytes are in flight. `KRK_WEIGHT_PREFETCH=0` and
  `KRK_WEIGHT_DMA=0` restore the serialized path for an A/B.

Routing experts are unaffected: `_exps` banks are still paged by `ExpertCache`,
never charged to this budget. The CPU backend never streams -- there is no
device/host split to page across.

Measured on the RX 9070 XT (gfx1201, 15.9 GiB), greedy, `--ctx 512`:

| model | arm | resident | streamed | free VRAM | decode | output |
|---|---|---|---|---|---|---|
| Qwen3-8B-Q4_K_M (4.68 GiB) | `--weight-vram-mb 3072` | 2.94 GiB | 1.74 GiB/token | 12.45 GiB | 2.6 tok/s | **md5 identical** |
| Qwen3-8B-Q4_K_M | resident (`-1`) | 4.68 GiB | -- | 10.8 GiB | 92.1 tok/s | baseline |
| Qwen3.8-27B-Q4_K_M (15.9 GiB) | auto, before promotion | 11.84 GiB | 3.76 GiB/token | 3.08 GiB | 1.2 tok/s | **md5 identical** |
| Qwen3.8-27B-Q4_K_M | auto + promotion | 13.74 GiB | 1.86 GiB/token | 1.17 GiB | **2.3 tok/s** | **md5 identical** |
| Qwen3.8-27B-Q4_K_M | auto + promotion + read/copy pipeline | 13.74 GiB | 1.86 GiB/token | 1.17 GiB | **3.0 tok/s** | **md5 identical** |
| Qwen3.8-27B-Q4_K_M | `-1` | 15.61 GiB | -- | 0.00 GiB | 3.1 tok/s | baseline |
| Qwen3.6-27B-Q5_K_M (18.2 GiB) | auto | 11.83 GiB | 6.07 GiB/token | 3.23 GiB | 0.3 tok/s | coherent |
| Tiel-Coder-35B-A3B (24.8 GiB MoE) | auto | 1.88 GiB dense | experts page | 0.00 GiB | 3.0 tok/s | coherent |

**The streamed bytes are read from a pinned host mirror, not from the file.**
A streamed layer's bytes do not change between tokens, so re-reading them out of
the GGUF mapping on every pass repeats work that was already done: on the
15.97 GiB Qwen3.8-27B file at `--ctx 512`, 9 streamed layers per token meant
**64.70 GiB of re-reads across a 32-token bench** and the read was about 55% of
decode. The loader now fills one pinned allocation holding every streamed layer
at its own offset, once, through the same pooled reader; a staging pass after
that is `sync` + a DMA straight out of it. Pinned memory is also the faster
source here -- 16.4 GB/s against 8.9 GB/s through the narrow-aperture staging
ring -- so the per-token cost becomes the copy and no longer includes a read.

Measured on the RX 9070 XT, greedy, `--ctx 512 --chunk 64 --bench`, 32 tokens,
mirror versus `KRK_WEIGHT_HOST_CACHE=0`:

| file / arm | arm | staging before | staging after | decode before | decode after |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B Q4_K_M (15.97 GiB), auto | mirror on | 64.70 GiB @ 6.06 GB/s, read 819 ms | 64.70 GiB @ **13.79 GB/s, read 0 ms** | 2.9 tok/s (349 ms/step) | **6.2 tok/s (160 ms/step)** |
| Qwen3-8B Q4_K_M, `--weight-vram-mb 3072` `KRK_WEIGHT_PROMOTE=0` | off vs on | 55.52 GiB @ 5.96 GB/s | 55.52 GiB @ **19.02 GB/s** | 3.5 tok/s | **12.1 tok/s** |

Greedy output is byte-identical between the two arms (27B prompt run, 289 bytes,
md5 `b7525fa7dcf3398d4dffd10b432f0812`) and the suite is 4149/4149. The mirror
costs one load-time pass -- 4.13 GiB in 439 ms on the 27B, 1.74 GiB in 203 ms
on the 8B arm -- and is capped at 8 GiB of pinned RAM
(`KRK_WEIGHT_HOST_CACHE_MB=<n>`), because pinned pages cannot be swapped. A set
above the cap, or a failed allocation, falls back to the read path with a
warning: slower, still correct. `KRK_WEIGHT_HOST_CACHE=0` forces that arm.

**Promotion takes the same pinned path.** `promote_layer()` reads each batch
into `stage_host_`, which is already pinned, and used to hand it to the
*pageable* upload -- the driver's own bounce copy through the 4 MiB staging
ring, measured at **8.8-9.8 GB/s for exactly this loop** against **27.5 GB/s**
for the pinned batch the staging path uses. It now issues the pinned batch, so
a promotion is a DMA out of memory the host already owns: 10 promoted layers on
the 27B (2.11 GiB) went from **764-1662 ms to 455-513 ms**, and the one-off
staging charged to the first token fell with it. Greedy output is byte-identical
with promotion on and off (27B, 24 tokens, md5 `62dfed13c16a7878c620e98d15bfd681`
in both arms), suite 4149/4149.

**After this the transport is at the card's copy limit**, so the remaining
lever is fewer bytes rather than a faster path: 2.02 GiB/token at 13.8 GB/s is
~147 ms against the measured 160 ms/step, and the 8B arm is the same shape
(1.74 GiB/token at 19.0 GB/s against 80 ms/step). A model that does not fit
still cannot be made fast by streaming -- every token must move the non-resident
bytes over PCIe. What streaming buys is a **bounded, predictable** device
footprint: the KV cache stays flat (`kv flat: whole cache resident in VRAM`) at
a 4k context on a model larger than the card, instead of thrashing one HOT slot
per layer.

### `--mtp-probe` - run the file's own nextn (MTP) block for one step

A model built with a multi-token-prediction block (`qwen35moe.nextn_predict_layers`,
`blk.<n>.nextn.*`) carries a drafter inside its own file: the block takes the
target's final hidden state plus the last token's embedding, attends over its
own KV, and predicts the NEXT token directly. `--mtp-probe` loads that block
and runs it after prefill, printing the token it would draft beside the target's
own argmax -- a match is the token a greedy speculative round would accept:

```
[mtp   ] pos 8: draft 11 "," (22.2812) | target 11 "," (18.7344) | MATCH | 90.8 ms
```

`KRK_MTP_PROBE_ALL=1` repeats the comparison at every decode step and prints an
acceptance summary; the block's KV is incremental, so a step costs one block
forward. Nothing is emitted from the probe and generation is byte-identical to
a run without it.

Measured on `Tiel-Coder-35B-A3B-MTP-APEX` (256 experts, top-8, 22.1 GiB routed,
`--ctx 4096`, greedy, 16 tokens): **10 of 16 drafted tokens match the target's
argmax (62.5%)**, and the swapped concat order (`KRK_MTP_CONCAT=he`) matches
0 of 19 -- the head is wired correctly and the embedding-first order is the one
it was trained with.

**A block that ships in its own file is loaded with `--mtp FILE`.** A `-noMTP`
quant is a model whose MTP was published separately (`mtp-<model>.gguf` beside
it); that sidecar carries the same `blk.<n>.*` block at the index past the
trunk's stack, and `--mtp FILE` reads it from there and drives it with exactly
the same `--mtp-probe` / `--mtp-spec` loop:

```sh
kraken -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf \
       --mtp mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf \
       --mtp-probe -p "..." -n 32 --greedy
```

The sidecar is validated like the in-file block: its architecture must be the
trunk's, it must carry a block at or past the trunk's stack, its width and head
count must match the trunk (a head from a same-family model with a different
attention shape is refused with both numbers), and its `head_count_kv` may be a
per-layer array (nemotron_h_moe ships 53 entries; the block's own index selects
its entry). Experts of an **ungated** bank (`ffn_up_exps`/`ffn_down_exps` with
no gate) load and run, which is the shape Nemotron's block has.

Measured on `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf` +
`mtp-…NVFP4.gguf` (ctx 512, greedy): `nextn (MTP) block 52 loaded: 32 head(s) x
128, 128 experts x top-6 of 1856-wide FFN + shared expert, eh_proj 5376x2688 …
59.0 MiB resident`, probe `MATCH` on the first step (21.0 ms), and
`--mtp-spec --draft-tokens 2` running `14 round(s), 6 of 28 proposals kept
(21.4%), 1.14 token(s)/round`. The suite pins the loader's four refusals in
`test_mtp_sidecar` (17 checks).

Two block shapes load. A routed block (`ffn_*_exps`, Tiel) pages its experts
through the model's own `ExpertCache`; a dense block (`ffn_gate/up/down`, the
qwen35 dense family's nextn heads) runs that FFN instead. On this box **15 of
the 19 files carrying `nextn_predict_layers` run the probe** -- the other two
are head-only shards with no base model, and the 57 GB `DeepSeek-V4-Flash` file
is refused by its architecture (MLA and attention sinks, neither of which has an
implementation here) before MTP is reached -- its quant type #26 is not a
blocker: that id is I32, and the file's `ffn_gate_tid2eid` is a table of expert
ids, read as data. Acceptance over 8 greedy steps and two prompts is 50-100% on most:
`Qwen3.8-27B-Q4_K_M` 8/8 and 6/8, `Qwen3.5-9B-Q4_K_M` 7/8 and 5/8,
`Tiel-Coder-35B-A3B` 6/8 and 5/8, `Qwen3.8-Distill-35B-A3B` 5/8 and 4/8.

`--mtp-spec` drives that loop as a decoder: the block proposes up to
`--draft-tokens` tokens (one block forward each, fed the previous proposal),
**one** batched target forward verifies them, the longest matching prefix is
kept plus the target's own bonus token from the last verification row, and the
block's KV is re-fed from the accepted tokens' captured target rows. A hybrid
target (gated delta net layers) snapshots `rec_state_`/`conv_state_` before the
verify and, on a partial accept, restores and replays the accepted prefix in one
batched forward -- the delta rule cannot be rewound.

Measured (greedy, ctx 4096, `--draft-tokens 2`, 64 tokens): **1.45 tokens per
target forward** on Tiel (44 rounds, 33% of proposals kept), **1.19** on
`Qwen3.8-Distill-35B-A3B` (54 rounds), **1.78** on `Qwen3.5-9B` at 64 tokens.
Where the batched forward agrees with single-row decode the text is
byte-identical to plain greedy; on Tiel it matches a multi-row decode instead,
because this hybrid is not chunk-invariant (`--chunk 1` and `--chunk 256` are
`6792671902ca` vs `ac63e6812909` with speculation off).

It is **opt-in, and a throughput loss on this box**: plain decode is 20.1 tok/s
on Tiel against 9.8 at one proposal and 7.9 at two, because the round pays a
verify forward plus, after a rejection, a replay of the accepted prefix. The
propose pass itself is not the bottleneck any more: measured on Tiel (ctx 4096,
48 tokens, `--draft-tokens 2`) it is 193 ms of a 4.6 s run, 160 ms of which is
its own expert acquires at an 11% miss rate.

`KRK_MTP_PIN=1` keeps the block's 256 experts resident instead (816 MiB of the
7594 MiB device budget, pinned so the LRU cannot take them). It does what it
says: **propose 193 -> 136 ms, refeed 80 -> 33 ms, block acquires 160 -> 0 ms**,
with the text unchanged. It is **off by default because the trade is a wash**:
the 816 MiB comes out of the target's expert budget, so the target's own
promotions cost 105-180 ms more and 5 interleaved order-balanced pairs measure
8.7 tok/s pinned against 8.9 unpinned. Turn it on where the block's share of the
run is larger than it is here -- more proposals per round, or a target whose own
expert set fits anyway.

The exact, useful part is the probe above; the decoder also needs a cheaper
verify-or-replay round before it can pay. The chunk gap underneath it is now
closed for this model: the first differing stage between `--chunk 1` and
`--chunk 2` was `attn.out` on the first full-attention layer at position 1, and
`attention_bexact_kernel` (the prefill path, on by default) now reproduces the
decode row's arithmetic -- per-key running max, keys split across its 8 warps,
the same merge -- so a chunked prefill writes the bytes a decode step would.
Tiel gives `613eb9b3` at `--chunk 1`, `2` and `8` where `2..8` used to give a
different hash from `1`, and `kraken-bench --bitexact` reports all 24 attention
cases bit-identical. It is also **faster than what it replaced**, which is why
it is now the only prefill formulation: +10.4% prefill at a 170-token prompt
and +113% at 3400 tokens on Qwen3.5-9B, +80% on Phi-3.5 (hd 96) at 2760
tokens -- same text in every pair. `KRK_ATTN_BEXACT=0` is the arm that falls
back to the tiled kernel. The query-tiled kernel and its probe are deleted;
the tiled kernel survives only for a windowed layer, a non-causal call and a
head width outside 64..256 in steps of 32. Two other kernels still are not
chunk-invariant -- the GEMM tile rung from 9 rows up, and the recurrent
block's conv/scan at rows>1 -- so a hybrid's text can still depend on the
chunk size away from attention; see `AGENTS.md`, "Chunk invariance was an
ATTENTION property".

The suite carries that check as a gate now, so a regression fails a run instead
of a review: `./build-hip/kraken-tests --run-group test_prefill_chunk_bitexact`
prefills one 64-token prompt at `--chunk 1`, `2` and `8` through
`kraken.exe --prefill-hash` and requires one hash across them -- and then runs
the chunk-8 arm again with `KRK_ATTN_BEXACT=0`, where the hash must DIFFER, so
the gate is shown able to fail. The hash covers every prefill row, not just the
last: a last-row logits hash is blind here, because the f16 residual absorbs
the drift before the head.

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

**That plan is then clamped against reality.** It is made before the KV cache and
the engine's workspaces exist, so on a card whose routed set exceeds VRAM it can
leave **0.00 GiB free** - and an allocation on a full card makes the driver evict
pages, which is why it does not merely run slow, it stalls the desktop:

| 22.1 GiB routed set, ctx 4096, 16 tokens | free VRAM | alloc per 2.2 MiB cell | decode |
|---|---|---|---|
| auto (11.95 GiB) | 0.00 GiB | 30 ms | 3.1 tok/s |
| 10 GiB | 0.00 GiB | 1.2 ms | 6.2 tok/s |
| clamped (7.7 GiB, 3.3 GiB floor) | 2.95 GiB | 0.25 ms | **16.6-18.7 tok/s** |

After the warm-up, `Engine::init` sets the budget to `resident bytes + what is
really free beside the floor`, so the pool evicts-and-reuses instead of
allocating.

**Then rank the fill.** Which experts are resident is worth more than how many:
with the same 7.68 GiB pool, an even-order fill decodes at 16.6 tok/s and a
ranked one at **23.9 tok/s** (671 promotions instead of 1850, same text). The
ranking is one command, and the next run picks it up automatically:

```sh
kraken -m Tiel-Coder-35B-A3B-MTP-APEX.gguf --expert-scan -p "some text" -n 8
# writes <model>.krakenexperts.json; every later load prints
# "expert warmup: using ... (16 positions, full-forward, 10240 ranked pairs)"
```

**Then let the workload re-rank it.** A scan is a measurement on one prompt; a
server or an agent sees different traffic every day. `--expert-adapt` counts the
routing of a *normal* run and merges it into the index when the run ends, so the
next load's warm fill and promotion order follow what was actually used:

```sh
kraken -m Tiel-Coder-35B-A3B-MTP-APEX.gguf --expert-adapt -p "the workload" -n 64
# "expert adapt: merged 64 observed positions into ..."
```

The merge is `p' = retain * p_file + (1 - retain) * p_run` per layer, with
`KRK_EXPERT_RETAIN` (default 0.5) the weight the existing file keeps. Per layer
because mass is comparable within a layer and not across them; a layer the run
never routed through keeps its row; a file that describes another model is
refused rather than merged. `retain=1` keeps the file exactly, which is the A/B
arm. No `--expert-scan` is needed, and the two can be combined.

A larger device budget is *not* the lever: at a 2 GiB floor the same model reads
20.6 tok/s and at 1 GiB floor 17.6, because a wider resident set reaches into
colder experts and leaves the allocator no headroom. The floor is a fifth of the card (`max(reserve + 1 GiB, total/5)`;
`KRK_VRAM_FLOOR_MB` overrides it) and the clamp prints what it did. The CPU
backend has no device/host split, so the clamp is skipped there.

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

WARM is host RAM by default. Cold read-throughs may be kept there, and a WARM
expert promotes through a bounded pinned staging ring; ordinary VRAM eviction
preserves an existing WARM copy without downloading it. Expert WARM shares live
host-memory admission with KV WARM/COLD staging. The `--info` report shows WARM
capacity/admissions, while the end-of-run report shows expert pool physical
commitment separately from logical residency.

**The size is automatic, and it is the largest one that leaves the OS room.**
With no flag the tier takes the expert corpus capped by 60% of installed RAM
(the 80% ceiling minus the 20% reserve) or by *free* RAM minus that reserve when
that is smaller -- 9.6 GiB on a 16 GiB machine, 14.4 on 24, 57.6 on 96. On
Qwen3-30B-A3B (9.6 GiB routed set) a 16 GiB class therefore gets the whole set,
which is the tier that reads nothing from the file; a smaller tier on that file
is the measured cliff (`docs/PERF-ANALYSIS.md` 12d), so `--expert-warm-mb` is
now only for *capping* it on a box that is also doing something else.

### `--expert-pin-mb` - pin the hot prefix of that tier

```sh
kraken -m gpt-oss-120b-Q8_0.gguf --expert-warm-mb 11000 --expert-pin-mb 8192 --expert-warmup 1
```

The first `N` MiB of WARM, in admission order -- which is rank order when the
model has an expert index -- are allocated pinned (`hipHostMalloc`) instead of
pageable, and a promotion out of them is **one DMA straight out of that buffer**
instead of a 12.6 MB memcpy into the staging ring followed by that DMA. That
memcpy was the whole gap between the promotion path and the link:

| arm (gpt-oss-120b Q8_0, ctx 512, 24 tokens, greedy, ranked warmup, 11 GiB WARM) | decode | ms/step |
|---|---|---|
| nothing pinned | 16.4 tok/s | 60.9 |
| `--expert-pin-mb 8192` | **20.5** | **48.8** |
| `--expert-pin-mb 4096` | 16.3 | 61.5 |

Interleaved, order-balanced pairs, six per arm, text byte-identical throughout:
**+25.0%** at an 8 GiB pin (6/6 pairs, MAD 2.0%) and **+18.3%** at a 12 GiB pin
over the 34.5 GiB auto tier (5/5, MAD 6.1%; 20.3 -> 23.9 tok/s).

**Size it to the bytes a run actually promotes, not to a fraction of the tier.**
4 GiB beat nothing by exactly zero even though it halved the measured copy time:
at half the working set the other half still pays the memcpy, and that half is
what the step waits on. 8 GiB covered all 7098 MiB the run promoted and is where
the gain appears.

**This is automatic, and a measurement decides it.** At init the run times a
16 MiB copy through each path -- pageable (staging memcpy + DMA) against pinned
(direct DMA) -- and pins when pinned is at least 1.20x faster, which it prints
along with the decision (`promotion measured 10.1 GB/s pageable against 18.6
GB/s pinned`). A ratio rather than the runtime's `large_bar` attribute, because
the aperture is a **firmware** setting: Above 4G decoding + Resizable BAR
changes it, Windows has reported the attribute as 0 with ReBAR enabled, and the
pageable path keeps its ring memcpy either way. The prefix is a quarter of
installed RAM, capped at 16 GiB, and the WARM tier is pinned **whole** inside
that bound -- it used to keep half of itself pageable, and that half is the one
every miss lands on: on the 256 MiB expert arm the same run goes **139.8 ->
65.3 ms/step (7.2 -> 15.3 tok/s)** with byte-identical text when the whole tier
is pinned, because a pageable promotion measured 469 us against 10 us from
pinned. `--expert-pin-mb 0` opts out, `KRK_HOST_PIN_MB` sets the number
explicitly, `KRK_PIN_AUTO=0` is the A/B arm, and `KRK_H2D_PROBE=0` falls back to
the BAR hint. Measured with a
16-token greedy
run, interleaved pairs, identical text: REAP-320 **101.7 -> 88.2 ms/step**
(+13.3%), a 35B-A3B **35.4 -> 30.1** (+17.4%, xfer 386 -> 193 us/expert), a
disk-bound 30B-A3B +2% (the copy is 2% of its step), and a dense model 0% (no
tier to pin) -- so the win is real exactly where promotions are on the critical
path, and it is not claimed where they are not.

**It also picks the PAGEABLE path, from the same probe.** Two paths exist: the
staging ring (a host memcpy into a bounded pinned buffer, then a DMA) and a
direct DMA the driver stages itself on one side stream. Init times both -- plus
the pinned arm -- interleaved and order-balanced, takes the median, and switches
to direct at **1.20x**, which is what a wide aperture (Resizable BAR) should
buy by removing the host copy. This box's 256 MiB aperture lands ON the line
rather than clearly under it -- the same binary read 8.7 against 10.6 GB/s
(direct, 1.21x) in one run and 10.9 against 12.3 (ring, 1.13x) in another, both
at 16 MiB -- so the verdict is cached rather than re-rolled every run: it is
remembered per (card, driver version, probe size, aperture hint) and reused,
which also stops the path from flipping between runs. The log line names both the
age and the driver (`a 16 MiB copy from the cache, measured 18 min ago on driver
71626331 at 11.6 GB/s ...`), so a cached rate is traceable without opening the
file. `KRK_H2D_PROBE_RERUN=1` measures again,
`KRK_H2D_CACHE=0` (or `KRK_H2D_CACHE=<path>` to move the file) disables or
relocates it, and entries expire after 30 days. The file is pruned on every
write: expired verdicts and another driver's older than a week go, so a
driver update does not leave three dead lines behind for good, and the rewrite
SAYS what it dropped (`H2D cache <path>: dropped 3 dead verdict(s) (1 expired,
1 from another driver, 0 unreadable, 1 over the cap); 32 kept`) rather than
quietly shrinking a file the user may be trying to account for. A clean rewrite
prints nothing, so the line only ever appears when something actually went. The other branch is proved
rather than inferred: a test-only backend simulates an aperture (real copies
plus a bandwidth-shaped wait), and the suite's `test_pageable_path_probe` runs
the whole policy on a wide one and a narrow one, at the real engine -- and then
from a seeded cache, where the cached verdict must win over what the simulated
card would measure. The same measurement decides the prefix, so on a wide
aperture the pin declines on its own: there is no copy left for it to remove,
and the run stays fast without unswappable RAM. `KRK_WARM_DMA=1|2` forces a
path. `kraken-bench --h2d` prints the shape table and the balanced verdict, and
then measures the engine's own probe at all three cached sizes (16/32/64 MiB)
and WRITES that verdict -- so running it is how a machine refreshes the answer a
normal run starts from, for example after enabling Resizable BAR in the BIOS.
`kraken-bench --h2d-cache` is the other half of that: it reads the file back
without opening a model, one line per entry -- probe size, the three rates, the
path the engine's own predicate selects, and the age and driver that measured it
-- and names the reason when a line is one a run would skip (`[expired]`, `[from
an older cache format]`, `[a failed measurement]`). A live verdict from another
driver is shown as `[another driver]` rather than as this run's, and a cache that
is empty, missing or turned off lists as empty instead of failing. That is how
"why is my run measuring again" is answered without a run.

The probe itself no longer pays a fixed five iterations per size. `iters` is a
cap and it stops as soon as the medians have settled -- at least three samples,
median absolute deviation under 10%, and the two newest samples agreeing with
the median, which is what catches a tight-but-ramping start. One iteration of a
64 MiB probe is four times the work of a 16 MiB one and needs the fewest samples,
so this is where the saving is: measured on this card, interleaved
order-balanced pairs, the 64 MiB probe went **112 ms (forced 5) to 83 ms (3
iterations), -25.9% in 3 of 3 pairs**, 32 MiB 59 to 43 ms, 16 MiB 31 to 25 ms,
with the same path and pin chosen in every arm. `kraken-bench --h2d` now prints
what it ran (`64 MiB (3 of 5 iter, spread 3.9%, 82 ms)`), `kraken-bench
--h2d-noise <MiB>` prints the per-iteration rates and the spread that the rule is
fitted to (one size per fresh process -- measuring all three in one process
charges the cold driver state to whichever ran first), and
`KRK_H2D_PROBE_ITERS=<n>` runs exactly n iterations, which is the old fixed quota
and therefore the A/B arm.

Every run that probes also says which it did, in one line at init, with the
reason when it measured:

```
[info ] H2D verdict: reused the cached 16 MiB one (from the cache, measured 1 s ago on driver 71626331)
[info ] H2D verdict: measured the card now (16 MiB, 3 arms x 5 iteration(s)) -- this card's 16 MiB entry is not usable (expired), 1 dead of 1 line(s)
```

The reason is the useful half: "measured now" on its own reads the same whether
the cache was empty, held this card's verdict for a probe size this model does
not bucket to, held it from another driver, or held it and it had expired -- and
those are fixed differently (nothing to do, run a model that buckets there, wait
for the next measurement to replace it, or nothing at all). `KRK_H2D_PROBE_RERUN=1`
names itself as the reason, and a probe that could not allocate its buffers says
that instead of printing a zero rate. The line is decided after the lookup and before the write, so it describes the cache
the run found rather than the one it just wrote into.

Whether any of that is worth it is a question about many runs, so the answer is
kept beside the cache (`<cache>.stats`) and reported at the end of each one:

```
[stats ] h2d       6 of 8 probed run(s) reused a cached verdict (75%), 2 measured the card at 235.5 ms on average, so the cache has avoided about 1413 ms of probing
[stats ] h2d          16 MiB: 2 of 3 reused (67%), 1 measured at 71.0 ms -> about 142 ms avoided
[stats ] h2d          32 MiB: 4 of 5 reused (80%), 1 measured at 400.0 ms -> about 1600 ms avoided
```

The per-size rows are the honest ones. The probe size is part of the cache key,
so a 64 MiB verdict is not the one a 16 MiB model asks for, and the measurement
it replaces costs several times as much: a single blended rate describes no run
in particular, and a machine whose only warmed model buckets to 64 MiB would read
as a mediocre cache instead of one that had never been asked. Each row carries
its own hit rate and its own avoided time, and a size that has avoided nothing
prints no avoided figure rather than a reassuring `0 ms`.

Only inits that got a verdict are counted -- a run with the probe off (or a dense
model, which has no promotion path) reused nothing and measured nothing and is
not part of the rate. The avoided-probe figure is priced at the average
measurement this machine has actually paid rather than at a constant, and a
tally with no measurement in it says the time is `not priced` instead of
inventing a number. The ledger is best effort and last-writer-wins, so a
concurrent server can round it: it is a tally, not an audit. `KRK_H2D_CACHE=0`
disables the ledger with the cache it counts.

The end-of-run `[stats ] pinned` line reports the copy-stream rate from events
recorded on that stream: 22.7 GB/s in the pin-8192 arm, and 6.1-24.9 GB/s across
arms because a copy shares the device's memory bandwidth with the decode kernels.
Read it as a diagnostic, not as a bus spec -- and do not read the `transfer`
line's GB/s for a pinned promotion at all: that timer wraps the *enqueue* of an
asynchronous copy and once reported 64.6 GB/s on a link that does 27.

### `--router-resident P` - route to the expert that is already there

```sh
kraken -m gpt-oss-120b-Q8_0.gguf --expert-warm-mb 11000 --expert-pin-mb 8192 \
       --expert-warmup 1 --router-resident 0.10
```

The router picks the top-k experts by score, and two experts whose scores are
within a hair of each other are interchangeable to the model. This makes the
tier break that tie: a candidate within `P` of the best score (a fraction of it,
so `0.10` is 10%) is preferred if it is already in VRAM, then if it is in WARM.
Nothing else changes -- the mix weight is still the unbiased probability, and a
pick is only moved when a near-equal alternative is resident, which the end-of-run
`[stats ] router` block counts:

```
[stats ] router    4032 picks, 460 (11.4%) went to an already-resident expert
                   over a near-equal one (margin 10.0% of the top score)
[stats ] router    hot share 3382/3844 picks (88.0%) landed on an expert already in VRAM
```

A step on a tier smaller than the routed set is paid for in experts it has to
move, so fewer moves is directly fewer bytes. Measured on gpt-oss-120b Q8_0
(ctx 512, 24 tokens, greedy, 11 GiB WARM, 8 GiB pin, ranked warmup; interleaved
order-balanced pairs):

| margin | picks moved | promotions | bytes/token | decode |
|---|---|---|---|---|
| off | 0 | 563 | 283.65 MiB | 20.3-21.4 tok/s |
| 0.02 | 2.6% | 537 | 269.47 MiB (-5.0%) | 21.6 |
| **0.10** | **11.4%** | **462 (-17.9%)** | **226.92 MiB (-20.0%)** | **23.7-24.9 (+16.7%, 5/5, MAD 0.1%)** |
| 0.20 | 27.8% | 349 | 168.09 MiB (-40.7%) | 26.7 |

On a model where everything fits (`Qwen3.8-Distill-35B-A3B`, so no file reads at
all) it still pays, because it cuts promotions between the tiers rather than
reads: 917 -> 742 promotions and 30.3 -> 32.7 tok/s at 0.10, and **+14.5%**
(5/5 pairs, MAD 1.6%) at 0.20, with the generated text identical in every arm.

**It changes the route, so it can change the text** -- that is the trade, not a
bug. The measured arms above happened to print identical text, and that is
evidence the near-ties really are near, not a guarantee for every prompt. Start
at 0.10, and check the text on your own workload before going past it.

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

### `--draft-head` - an external EAGLE-3 head as the drafter

```sh
python tools/eagle_to_gguf.py head.safetensors -o head.gguf \
       --target-vocab 201088 --target-arch gpt-oss      # once, offline
kraken -m gpt-oss-120b.gguf --draft-head head.gguf --draft-tokens 2 --greedy \
       -p "explain quicksort" -n 128
```

A published speculator is a *head*, not a model: one block that reads the
target's own hidden state and the last token's embedding and proposes k tokens
for the verify loop above. `--draft-head` loads one from a GGUF that
`tools/eagle_to_gguf.py` converts from the published safetensors; it needs no
VRAM of its own (the block runs on the host) and no second model beside the
target. This is a different artifact from the nextn block inside a model's own
file, which `--mtp-spec` drives.

The loader checks the head against the target before the first token: the
hidden width and the target vocabulary must match, every `d2t` id must be
inside that vocabulary, and every weight must be F16. A head from another
tokenizer proposes ids the verifier can never accept, which reads as a bad
drafter rather than as a wrong file -- so it is refused, with the number that
disagrees. A recurrent (gated delta net) target is refused too: the verified
suffix has to be rewound, and a delta-rule state has no inverse.

`--draft-head-prime N` (default 256) is how many of the most recent positions a
round primes the head over. The head's fusion projection is
O(hidden x target_hidden) per position, so priming a 4k prompt in full is
hundreds of GFLOP of host arithmetic; `0` is the exact whole-prompt form.

**Sampling works too, and it cannot be a prefix match.** A greedy round asks
"is the target's argmax the proposal"; under a stochastic sampler that test is
wrong in both directions -- rejecting every non-argmax proposal makes the head
useless, and accepting every match emits the head's distribution instead of the
target's. The round therefore verifies with the acceptance rule from
Leviathan et al. (arXiv:2302.01318): each token is proposed by a DRAW from the
head's row (softmax at the run's temperature, mapped through `d2t` into target
space), accepted with probability `min(1, p/q)`, and on rejection replaced by a
draw from the residual `max(0, p - q)`. `p` is the sampler's own row -- the same
`Sampler::build_dist` that `sample()` draws from, repetition penalty included,
with the round's already-accepted tokens pushed into the penalty history -- so
the emitted stream stays distributed as the target alone would produce it.
`KRK_SPEC_RS=0` turns the verifier off for this path and runs the plain loop.

The rule is pinned by `test_spec_sampling` (38 checks in the default suite):
the enumerated identity `sum(min(p,q)) + sum(p-q)+ = p`, the accept ratio, the
residual's support, the one-hot limit, the histogram of 200k real draws through
`spec_accept_step` against `p`, and that `build_dist` is the row `sample()` uses.
An always-accept implementation fails four of them, which is how the group was
shown able to fail.

Measured (qwen2 1.5B, ctx 512, `--temp 0.9 --seed 7`, 24 tokens, `--draft-tokens
3`, same random head -- so every proposal is rejected and every round pays the
worst case): output fluent code, `draft head: 24 round(s), 0 of 72 proposals
kept (0.0%), 1.00 token(s)/round`, i.e. the rejection path (residual draw,
re-forward, realign) runs 24 times in a row without drift. Measured (qwen2 1.5B, ctx 512, greedy, 16 tokens, with a randomly-initialised
head so every proposal is rejected): output byte-identical to plain greedy --
md5 `f7957feb5610` for both arms -- and the run reports `draft head: 16
round(s), 0 of 32 proposals kept (0.0%), 1.00 token(s)/round | prime 141 ms |
propose 1442 ms`. That is the property that matters first: a head can change
the speed, never the text. The suite pins the parts a random head cannot
(`test_draft_head`: geometry, vocabulary, the prime/replay/truncate contract,
50 checks).

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
KRK_PORT=8090 kraken-server --model model.gguf --ctx 4096
# GPU when the HIP backend is present; --cpu forces the reference backend.
# The listen port is --port, then $KRK_PORT, then 8080, and the banner prints
# the port that was actually bound.

curl http://localhost:8090/health
curl http://localhost:8090/v1/models
curl http://localhost:8090/v1/completions \
     -d '{"prompt":"The capital of France is","max_tokens":32,"temperature":0}'
curl -N http://localhost:8090/v1/completions \
     -d '{"prompt":"Hello","max_tokens":32,"stream":true}'        # SSE
curl http://localhost:8090/v1/chat/completions \
     -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```

`POST /v1/completions` and `POST /v1/chat/completions` accept `max_tokens`,
`temperature` (0 = greedy), `top_k`, `top_p`, `min_p`, `repeat_penalty`,
`repeat_last_n`, `seed`, `stop` (string or array) and `stream`. Chat requests
are rendered through the ChatML template and stop on `<|im_end|>` /
`<|endoftext|>`; both endpoints report `finish_reason` (`stop`/`length`) and a
`usage` block. `GET /health` and `GET /v1/models` round out the surface, and
`GET /` serves the built-in chat page: one HTML document with no external
assets, streaming over `POST /v1/chat/completions`, with a stop button, per-turn
sampling and delete/regenerate.

`--prefill exact|fast` (default `exact`) selects the rows>1 GEMM rung for the
HTTP path too. `fast` takes those GEMMs to the RDNA4 matrix cores and measured
**412.5 against 177.6 tok/s** on a 4081-token request (chunk 1024, 30B-A3B); it
is a different-arithmetic mode, so the generated text can differ from `exact`
and the output is not chunk-invariant past 8 rows per chunk. The server prints
one startup line when it is engaged.

`stream: true` replies with `text/event-stream`: one `data:` chunk per
generated token, a final chunk carrying `finish_reason`, then `data: [DONE]`.

The engine is single-sequence, so connections are served concurrently but
generation is serialized; each request starts from a fresh KV context. Any
OpenAI client works:

```sh
curl http://localhost:8090/v1/chat/completions \
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

The second tier is inclusive **host RAM**: `--expert-warm-mb N`
(old spelling `--expert-l2-mb`) caps its per-tier capacity, and the auto size is
the largest the machine can spare (see above). Cold reads may
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
