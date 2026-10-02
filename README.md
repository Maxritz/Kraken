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

Measured with `kraken --bench` (greedy, ctx 4096, chunk 256) on
**SmolLM2 135M Instruct** (Q4_K_M; 30 layers, 576 embd, GQA 9/3
heads, head_dim 64, FF 1536, vocab 49152):

| device | HIP | prefill, 51 tok | decode, 64 tok | decode, 256 tok |
|---|---|---|---|---|
| AMD Radeon RX 9070 XT (gfx1201, RDNA4, WMMA, 32 WGPs) | 7.16 | 3.7–4.4k tok/s | 448–476 tok/s | 430–433 tok/s |
| AMD Radeon RX 6700 XT (gfx1031, RDNA2, no WMMA, 20 MPs) | 7.15 | 577–837 tok/s | 304–372 tok/s | — |

Ranges are min–max over repeated runs. The 5-round engine A/B on the
9070 XT averages **427.6 tok/s** decode with both kernel fusions on
(defaults) and **440.1 tok/s** on the best arm (`KRK_FUSED_LAYER=0`,
`KRK_FUSED_ATTN=1`); prefill is unaffected by either flag. Kernel-level
probes put the 9070 XT decode step at 727 tok/s launched and 741 tok/s
under a HIP graph (wall-clock), and a 51-token prefill chunk at
12.2k tok/s. The full decode-bandwidth campaign, fusion A/B and graph
timing methodology are in [docs/STATUS.md](docs/STATUS.md) (§8–§11).

**Qwen3-MoE-4x0.6B-2.4B** (Q4_K_M, 965 MB — fetched with
`scripts/fetch_moe_model.sh`) loads and completes real greedy forward
passes on the CPU oracle. The oracle is scalar and single-threaded
(minutes per 28-layer MoE forward), so it proves correctness, not
speed — interactive MoE decode is what the GPU backends are for. Use
`--ctx 256/512` for MoE runs on small machines.

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
