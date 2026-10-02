# Architecture

## The seam

```
                    ┌──────────────────────────────┐
                    │  Engine  (src/engine.cpp)    │   prefill / decode / sampling loop
                    │  Model   (src/model.cpp)     │   GGUF → device-resident weights
                    └──────────────┬───────────────┘
                                   │  Backend (include/krk/backend.hpp)
                  ┌────────────────┴────────────────┐
                  │                                 │
      ┌───────────▼───────────┐        ┌────────────▼──────────────┐
      │ HipBackend            │        │ CpuBackend                │
      │ src/hip/backend_hip   │        │ src/backend_cpu.cpp       │
      │ .hip + kernels/*.hpp  │        │ scalar oracle             │
      └───────────────────────┘        └───────────────────────────┘
```

`Backend` is the only abstraction in the engine. It has twelve methods, all of
them bulk operations. `Engine` contains no device code, no `#ifdef`, and no
notion of a GPU; it allocates workspaces in whatever element type the backend
advertises (`F16` on the GPU, `F32` on the CPU) and calls ops.

That is why the CPU build is not a toy: it runs the *same* engine code and is
what the test suite exercises, so a change to the forward pass is caught without
a GPU present. It is also the oracle — `--cpu` on the same model with a greedy
sampler should match the GPU token for token, which is how you localize a bad
kernel to a specific op.

## Data flow, one forward pass

```
tokens ─► embed ─► x[T, n_embd]
                    │
                    ▼   per layer
   ┌──────────────────────────────────────────────────────────────┐
   │ xn  = rmsnorm(x, attn_norm)                                  │
   │ q,k,v = gemm(xn, wq/wk/wv)          ← fused dequant + WMMA    │
   │ q,k,v += bias                       ← Qwen2/Phi only           │
   │ qk_norm(q,k)                        ← Qwen3/Gemma3 only       │
   │ rope(q,k,pos)                                                 │
   │ kv_append(k,v,pos)                                            │
   │ attn = attention(q, kcache, vcache)  ← online softmax         │
   │ x += gemm(attn, wo)                                           │
   │ xn  = rmsnorm(x, ffn_norm)                                    │
   │ x += gemm(silu(gemm(xn,wgate)) * gemm(xn,wup), wdown)   ← dense│
   │   or, on a MoE layer:                                         │
   │     logits = gemm(xn, ffn_gate_inp)          → host routing    │
   │     for each expert e (one grouped pass over its tokens):      │
   │        ExpertCache::acquire(layer, e)        ← lazy page-in    │
   │        x̃ = gather(xn, tokens that picked e)   ← one block      │
   │        ỹ = gemm(silu(gemm(x̃,g_e))*gemm(x̃,u_e), d_e)           │
   │        ffn[rows] += w̃_e · ỹ                  ← scatter, fused  │
   │     x += gemm(silu(gemm(xn,sh_g))*gemm(xn,sh_u), sh_d) ← shared│
   └──────────────────────────────────────────────────────────────┘
                    │
                    ▼
   logits = gemm(rmsnorm(x_last, out_norm), out_head)   ← final row only during prefill
```

Weights are `void*` device buffers whose `DType` is the on-disk GGUF type. They
are uploaded verbatim; nothing is repacked or expanded. The GEMM reads a weight
row in its quantized form and dequantizes on the fly.

## Files

### Host (no HIP dependency)

| file | responsibility |
|---|---|
| `include/krk/common.hpp` / `src/common.cpp` | aliases, logging, aligned allocation, fp16/bf16 conversion, `MappedFile` (mmap / `CreateFileMapping`), bounds-checked `Cursor`, timing |
| `include/krk/quant.hpp` / `src/quant.cpp` | the GGML block decoders — **the reference oracle** |
| `include/krk/gguf.hpp` / `src/gguf.cpp` | GGUF v2/v3 reader over the mapping; KV store, tensor directory, alignment handling |
| `include/krk/backend.hpp` | the `Backend` interface and `DeviceCaps` |
| `include/krk/model.hpp` / `src/model.cpp` | metadata resolution, tensor schema validation, upload, MoE geometry, host-side RoPE table |
| `include/krk/expert_cache.hpp` / `src/expert_cache.cpp` | lazy, byte-budgeted, frequency-aware residency for MoE expert matrices (LFU + aging + pinning; `ExpertSource`, `ResidentExpert`, `ExpertCache`), plus the optional second tier: with `--expert-l2-mb` a victim is demoted to page-locked host memory (`Backend::alloc_host` → `hipHostMalloc`) instead of released, and promoted back with a DMA on its next request. Pins do not protect host copies — the tier is a cache of a cache |
| `include/krk/tokenizer.hpp` / `src/tokenizer.cpp` | SPM unigram (Viterbi) and byte-level BPE (GPT-2 / Qwen2 / Llama3 pre-tokenizers) |
| `include/krk/sampler.hpp` / `src/sampler.cpp` | seedable xorshift128+ RNG, rep-penalty → temp → top-k → top-p → min-p |
| `include/krk/engine.hpp` / `src/engine.cpp` | workspace allocation, chunked prefill, decode loop, streaming with stop-string holdback |
| `include/krk/json.hpp` / `src/json.cpp` | dependency-free JSON parser/writer for the HTTP API |
| `include/krk/http.hpp` / `src/http.cpp` | portable HTTP/1.1 listener, chunked `text/event-stream` framing (winsock2 / POSIX) |
| `include/krk/server.hpp` / `src/server.cpp` | the OpenAI-compatible endpoints on top of `Engine` (shared by `kraken-server` and the test suite) |
| `src/backend_cpu.cpp` | the scalar reference backend |

### Device

| file | responsibility |
|---|---|
| `src/hip/krk_hip.hpp` | arch detection (`KRK_GFX10/11/12`), error handling, half helpers, wave reductions, WMMA vector types |
| `src/hip/kernels/dequant.hpp` | `dequant_chunk<T>`, the runtime-dtype dispatchers, chunk-range dot products, row dequant |
| `src/hip/kernels/gemm.hpp` | `gemv_kernel`, `gemm_wmma` (RDNA3 + RDNA4 layouts), `gemm_simt` |
| `src/hip/kernels/normalize.hpp` | block reduction, RMSNorm, per-head norm, RoPE, silu-mul, bias add, elementwise, MoE row gather/scatter |
| `src/hip/kernels/attention.hpp` | embedding gather, KV append, tiled online-softmax attention |
| `src/hip/backend_hip.hip` | `HipBackend`: memory, op launch wrappers, capability probing, factory |

## Decisions worth defending

**Why the block-size-32 chunk model.** The alternative is a per-format kernel
with its own K-tile size, which multiplies the kernel count by the format count
and makes the tiling code untestable. Squeezing every format into 32-value
chunks costs one extra shift and mask per block and buys one GEMM, one GEMV and
one embed path. It also makes `BK` trivially satisfiable: any multiple of 32.

**Why dequantize into LDS for the GEMM but into registers for the GEMV.** During
prefill the weight tile is reused across the M dimension, so staging it in LDS
amortizes the dequant. During decode M is 1 and there is nothing to amortize, so
staging is pure overhead — the weights are read once and consumed once. Different
problems, different kernels.

**Why the online softmax.** Materializing the score vector for a 32k context and
32 heads costs 4 MB of scratch per layer and turns a streaming problem into a
two-pass one. The tiled version keeps LDS bounded at ~33 KiB and never spills to
global memory.

**Why activation precision differs by backend.** On the GPU, fp16 activations
halve the traffic on every activation read and write, and all accumulation stays
in fp32 (`wmma` accumulator, `v_dot2_f32_f16` accumulator). On the CPU there is
no bandwidth argument, so fp32 avoids a conversion in every elementwise kernel.
The `act_type()` / `act_size()` accessors keep the engine agnostic.

**Why `unsigned` grid dims are computed with `+ n - 1`.** Launch configuration
is the easiest place to silently off-by-one. Every grid is `ceil(size / tile)`
and every kernel masks on both axes.

## Adding a kernel

1. Write it in the relevant header under `src/hip/kernels/`, guarded by the
   architecture macros it needs.
2. If it has a per-format variant, add a `DType` specialization to `QTraits` and
   a branch to `dequant_chunk`. Do not add a new tiling.
3. Add the launcher to `HipBackend` and, if it is a new `Backend` method, to
   `CpuBackend` as well — the reference implementation is not optional.
4. Extend `tests/test_kraken.cpp` with a host-side check of the invariant, and
   compare `--cpu` against the GPU output on a real model.

## Roadmap — what is NOT here yet

The features below do not exist in the engine; nothing in the tree implements
them. They are listed honestly, in the order they would matter to the fleet
(CPU oracle → `gfx1031` → `gfx1201`).

**KV cache — none of it is there yet:**

* **No prefix/prompt caching.** Two generations with a shared system prompt
  re-prefill the shared part every time. The engine is single-sequence: one
  KV cache, one decode position, `Finish` has no cancellation concept.
  (The `kv_rollback` primitive added for speculative decoding is the first
  piece a future prompt-cache would reuse.)
* **No radix/prefix tree.** Not a gap to close incrementally — the single
  sequence design would need to become sequence-tree first. Worth doing when
  the engine grows multi-request serving; pure CLI decode never benefits.
* **No paged KV.** Pages matter when sequences come and go at different
  lengths (fragmentation). A contiguous cache is the right v0.1 trade-off and
  is not the bottleneck on one sequence.
* **No KV quantization.** Cache rows are the activation dtype (fp16 on GPU,
  fp32 on CPU); `kcache_`/`vcache_` are never compressed.

**Speculative decoding — greedy draft model, done:** `--draft model.gguf`
loads a second internal Engine sharing the backend (own weights, own KV), and
greedy requests take the verify-and-accept path: the draft proposes up to
`--draft-tokens N` continuations, the target verifies the block in one batched
forward with per-row head computes, and the longest greedy-matching prefix is
kept (with a bonus token on full match). Because only argmax branches are
pursued, accepted tokens are **bit-identical to plain greedy decoding** — the
test suite proves this for a weak draft at window sizes 1/2/4/6 and for the
target-as-its-own-draft (full-accept path). Sampling requests bypass the draft
with a warning; `--bench` reports proposals, accepts and the acceptance rate.
`kv_rollback(pos)` is the primitive this builds on: a logical shrink of the
contiguous cache (no zeroing — rows are overwritten before they are ever read
again).

**Speculative decoding — still open:** no tree attention (only a single linear
chain is verified per step), no `n-gram`/lookup speculation without a draft
model, and no MTP: `qwen3moe` GGUFs carrying `mtp.*` tensors load but those
tensors are never referenced. MTP would slot in as a special draft that reads
the target's own hidden state; the verify/accept machinery does not change.

**Serving/interaction:** `kraken-server` speaks the OpenAI API (see README), but
it is a thin shell over the single-sequence engine: generation is serialized
behind one mutex, every request re-prefills from position 0 (no prefix cache),
and there is no cancellation. No streaming continuation across calls, no beam
search, no concurrent batching.

**Quantization:** the IQ* (`IQ2/IQ3/IQ4_XS`) family and YaRN's NTK-aware
frequency warp are rejected or approximated; see the model notes.

Performance gaps on the device are in `docs/HARDWARE.md` (split-K attention,
device-side sampling, prefill tile size). MoE-specific candidates (device-side
routing, skipping the gather) are below.

On the MoE path specifically, the remaining candidates for optimisation are:

* routing logits are downloaded to the host every layer (a sync per layer).
  A device-side top-k with a small readback would remove it.
* the gather/scatter pair materializes the permuted activation block once per
  expert per matrix. A persistent expert-slot layout (tokens written straight
  into their expert's rows by the router) would skip the copy.
* the residency policy is LFU with aging: hits earn counters, counters pin hot
  experts against eviction, and decay (halving every 64 acquires) lets pins
  lapse when traffic moves on. `--bench` reports loads, evictions, hit rate and
  the pinned count. Tuning `kPinThreshold`/`kPinDecay` per model family, or a
  persistent “hot set” sized from a calibration pass, are the next steps.
* expert loads are synchronous: a miss blocks the layer while the slice is
  uploaded. Prefetching the next token's top-k+1 from the router distribution
  while the current step's GEMMs run is the next win, and it is orthogonal to
  the tier above — a promotion can be prefetched just as a load can.
