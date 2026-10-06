# Architecture

> Structure of the decode path as it stands after the engine pass. Read this
> before changing [src/engine.cpp](../src/engine.cpp): it says which piece of
> decode policy lives where, and why.

## The decode path, and who owns what

A generation is a prefill followed by a decode loop. Two loops exist — plain
greedy and speculative — and they deliberately share the parts that are policy
rather than mechanism.

| Concern | Owner | Notes |
|---|---|---|
| Forward pass, workspaces, recurrent state | `Engine` | `forward`, `forward_core`, `fetch_logits` |
| Layer math that differs by architecture | `Engine::gdn_forward`, `Engine::moe_ffn` | one method per layer kind; `forward_core` just dispatches |
| **Token emission policy** | **`TokenEmitter`** (file-local, [engine.cpp](../src/engine.cpp)) | owns the stop-string test, the holdback buffer, and the sink |
| Which token comes next | `Sampler` | pure, seeded, knows nothing about streaming |
| Tokenizer | `Tokenizer` | SPM + BPE, independent of the engine |
| Device capability | `Backend` subclasses | `DeviceCaps` is the one description of a target |

`TokenEmitter` is the piece worth knowing about. Appending a token to the
result, matching it against the stop strings, holding streamed text back so a
stop string cannot straddle a token boundary, and flushing the tail are one
policy. It used to be written out twice — once inside `generate`, once inside
`generate_speculative` — and the duplication had already produced a bug: a
debug flag added to one loop silently did nothing in the other. Both loops now
call `TokenEmitter::accept`, which returns whether a stop string completed, and
`TokenEmitter::finish`, which flushes. Adding a third decode path means calling
those two methods, not copying the policy again.

### Data flow for one accepted token

```
sampler.sample(logits_host_)   -> token
TokenEmitter::accept(token)     -> piece appended to the result
|                                stop strings matched
|                                holdback flushed to the sink
v
Engine::forward(&token, 1, pos) -> the next logits_host_
```

The only state the two loops share that neither owns is the KV cache and the
logits buffer, both `Engine` members; the speculative loop additionally drives
a second `Engine` (`draft_`) through the same three methods.

## Why the diagnostic lives in the engine

`--debug-topk` needs the logits row and the step index, and both exist only
inside the decode loop, so `dump_topk` is a file-local helper there rather than
a public API. It is deliberately not part of `Backend`: it describes one run,
not a device.

## Finding where two backends disagree

A stage dump turns "the device and the reference produce different text" into
"stage X of layer N is the first one that is wrong":

```
KRK_DUMP=a.txt ./kraken -m M -p "..." -n 1 ...          # device
KRK_DUMP=b.txt ./kraken -m M --cpu -p "..." -n 1 ...    # f32 reference
python tools/dump_diff.py a.txt b.txt
```

`KRK_DUMP` makes `Engine::dump_row` append the last row of every stage — the
embedding, each layer's input/output, and the inside of both layer kinds (the
attention projections, the output gate, the gated result, the `wo` output, the
residual, the FFN norm/gate, and the delta-net conv and delta-rule output) — so
the first line whose relative error jumps to O(1) names the guilty stage and
the rest is cascade. It is a hook on the hot path's *shape*, not its math:
inert unless the variable is set. Delete the files between runs; the engine
appends.

`scripts/coherence_check.sh` is the acceptance gate that sits above it: for
each runnable model it greedy-decodes the same prompt on both backends and
compares the *text*, with a documented tie classification for fp16 rounding
(see [docs/TODO.md](TODO.md)). A backend that matches every activation can
still print numbers, so the output being human-readable words is the test.

One lesson from the bug this found, for anyone adding an op: **an activation
buffer that changes layout must never be its own source.** `qwen3_next_split`
compacts a packed `q|gate` row into half the width, so an in-place call had
block `h` writing over the source rows of blocks `2h` and `2h+1`, and the
winner of that race changed the text. Ops that shrink a row now take a
distinct destination (see the contract on `Backend::qwen3_next_split`).

## Layering rules the code follows

* `include/krk/*.hpp` is the library surface. `src/*.cpp` is portable C++17
  with no HIP; `src/hip/**` is the only place that knows about devices.
* A backend never calls back into the engine. `Engine` drives, `Backend`
  answers — which is what lets the CPU backend stay a valid oracle.
* Policy both loops share belongs in one type (the `TokenEmitter` rule).
  Policy that differs between them stays in the loop.

## Device capability is read once, authoritatively

`cu_count` comes from the KFD topology, not from
`hipDeviceProp_t::multiProcessorCount`: HIP under-reports that field on gfx10
(a 40-CU RX 6700 XT comes back as 20), and every occupancy heuristic in
[src/hip](../src/hip) is sized against `cu_count`. See `kfd_compute_units()`
in [backend_hip.hip](../src/hip/backend_hip.hip) — on Windows, where KFD does
not exist, it returns 0 and the HIP value is kept.

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

**Quantization:** every dequantizer this engine lists in `docs/USER-GUIDE.md` —
the IQ family (`IQ2_XXS`/`IQ2_XS`/`IQ2_S`, `IQ3_XXS`/`IQ3_S`, `IQ1_S`/`IQ1_M`),
`IQ4_NL`/`IQ4_XS`, `MXFP4`/`NVFP4`, the ROCmFPX ids 100/101, BitNet's
`Q1_0`/`Q2_0`/`Q2_0_64` and `TQ1_0`/`TQ2_0` — runs on both the host reference
path and the GPU kernels. Still refused at load: the ROCmFPX ids **102**, **104**
and **107**, which no header on this machine defines, and the ternary id
**142** (`PQ2_0`), whose geometry is measured (34 B per 128) but whose code
map is not a level map -- four candidate maps decoded to non-words, the
signature of a rotated block. One id needs care because two producers spell
it: id 42 is the llama-dx fork's 128-value / 34-byte `Q2_0`, not upstream's
64-value block, which is id 48 here. YaRN's NTK-aware frequency warp remains
approximated; see the model notes.

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
