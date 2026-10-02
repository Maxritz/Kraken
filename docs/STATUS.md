# KRAKEN — current status

> **Pending work and open issues live in [TODO.md](TODO.md).** This document
> records what has been built and measured; that one records what is still
> broken and unfinished.

Snapshot of the engine as it stands, what is proven, what is written but
unproven, and what is next. Sections 1–7 record the session that added HTTP
server mode, a real MoE model and the GPU bring-up checklist on top of the
host pipeline that was already in place. §8 records the decode-bandwidth
campaign, which was **measured on hardware**, not just written.Host history: first written on a Linux container (g++ 11.4, 64 cores,
**4 GiB memory cap**, no ROCm / hipcc / clang / cmake), where device
kernels could only be written, never run. The §8 campaign was measured on the current
host: Windows 11 with the ROCm/HIP SDK (`hipcc`), `build-hip/` built for
`gfx1201` (Release, `KRK_BUILD_PROBES=ON`, Ninja), on an **AMD Radeon RX
9070 XT** (gfx1201, RDNA4, wave32, WMMA) reporting **32 CUs**.
§11 records the same §2f/§2g campaign re-run on a second machine:
`maclin` (10.0.0.12, Ryzen 5 5600X, ROCm 7.15 / core-10.0, native
`hipcc`), an **AMD Radeon RX 6700 XT** (gfx1031, RDNA2, wave32, **no
WMMA**).

---


## 1. Summary of this session's three requested items

| item | state |
|---|---|
| **HTTP server mode (OpenAI-compatible)** | **Done and tested.** New `kraken-server` binary: `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, buffered and SSE. 78 new checks in the suite, now **479/479**. |
| **Fetch a real MoE GGUF model** | **Done and verified.** Qwen3-MoE-4x0.6B-2.4B (Q4_K_M, 965 MB) downloaded, loads with correct geometry, and completes a real forward pass on the CPU oracle. `scripts/fetch_moe_model.sh` reproduces it. |
| **GPU bring-up checklist** | **Done.** `docs/GPU_BRINGUP.md`: build per target, `--info` conformance, CPU-oracle vs GPU token-for-token diff with a per-op bisect table, MoE residency invariants, VRAM sizing, known-unverified kernels, report template. |

---

## 2. HTTP server mode

### What was added

| file | role |
|---|---|
| `include/krk/json.hpp`, `src/json.cpp` | JSON value tree; recursive-descent parser (`\uXXXX` incl. surrogate pairs), compact writer with RFC 8259 escaping and integer-preserving number output. Rejects leading zeros, trailing commas, trailing garbage, bad literals, unterminated strings. |
| `include/krk/http.hpp`, `src/http.cpp` | HTTP/1.1: bind/listen (winsock2 or POSIX), request-line + header + `Content-Length` body parsing, buffered responses, chunked `text/event-stream` framing (`http_sse_begin/event/end`), percent-decoding. Thread-per-connection, 64 KiB header / 8 MiB body caps. |
| `include/krk/server.hpp`, `src/server.cpp` | `OpenAiService`: the four endpoints, ChatML rendering of `messages`, sampling-parameter mapping, per-request `kv_rollback(0)` isolation, generation serialized behind an injected mutex. Shared by the binary and the tests so both exercise the same code. |
| `src/main_server.cpp` | `kraken-server`: arg parsing mirroring the CLI (`--model --host --port --ctx --chunk --device --expert-cache-mb/-slots --expert-l2-mb --draft --draft-tokens --cpu -v`), backend/engine init with CPU fallback, banner, accept loop. |
| build wiring | `CMakeLists.txt` (core sources + `kraken-server` target + `ws2_32`), `scripts/build_linux.sh` (CPU and HIP paths), `scripts/build_windows.ps1` (CPU and HIP paths). |

### Behaviour

* `POST /v1/completions` — `prompt` (string or string array), `max_tokens`,
  `temperature` (0 ⇒ greedy), `top_k`, `top_p`, `min_p`, `repeat_penalty`,
  `repeat_last_n`, `seed`, `stop` (string or array), `stream`.
* `POST /v1/chat/completions` — `messages[]` rendered through ChatML
  (`<|im_start|>role … <|im_end|>`), automatically stopping on `<|im_end|>` /
  `<|endoftext|>`.
* `stream: true` ⇒ SSE: one `data:` chunk per generated token
  (`choices[0].delta.content`), one terminal chunk with `finish_reason`
  (`stop` / `length`), then `data: [DONE]`.
* `GET /health` → status/arch/n_ctx/device; `GET /v1/models` → OpenAI list
  shape with model geometry under `meta`.
* Errors use the OpenAI shape `{"error":{"message","type"}}`: 400 (bad JSON,
  missing prompt/messages), 404, 405, 413, 500.
* Single-sequence engine: connections are concurrent, **generation is
  serialized**; each request starts from a fresh KV context.

### Verification

* `kraken-tests` grew from 401 to **479 checks** — all pass (stable across
  repeated runs). New coverage: JSON parse/dump/escapes/unicode/malformed
  inputs/number formatting, plus a **real loopback test** that starts a
  listener on an ephemeral port, issues HTTP requests, and asserts the
  server's text/usage/finish/SSE stream are **bit-identical to a direct
  `engine.generate()`** on the same weights, that two identical requests
  return identical results (no state leaks), and that 400/404/405 paths work.
* Live binary smoke test on tinyllama (before the sandbox OOM/restart):
  `/health`, `/v1/models`, buffered completion, SSE stream, 404, malformed
  JSON all returned correct JSON.
* One real bug was found and fixed by the tests: the header parser scanned to
  `header_end` but the final header's `\n` sits at `header_end + 1`, so
  `Content-Length` was dropped and every POST body arrived empty.

---

## 3. Real MoE model

* **Model**: `Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder.i1-Q4_K_M.gguf`
  (`mradermacher/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-i1-GGUF`), 964.7 MB,
  GGUF v3, 339 tensors. Fetched with `scripts/fetch_moe_model.sh`.
* **Geometry** (`--info`): `qwen3moe`, 28 layers, 1024 embd, 16/8 heads,
  head_dim 128, dense FF 3072, **4 experts top-2**, QK-norm on, rope base
  1e6, byte-BPE vocab 151936 (bos 151643, eos 151645), 40960 train ctx,
  no shared expert, expert cache auto-budget 512 MiB.
* **Verified**: loads (schema, 3-D expert tensors, routing metadata, QK-norm,
  tokenizer); a full greedy forward pass completes and produces a real token
  (`Hi` → `iconductor`). Routing, lazy expert paging, batched expert GEMMs,
  output head and detokenizer all execute on a genuine producer's file.
* **Speed**: the CPU oracle is scalar and single-threaded; a 28-layer MoE
  forward takes minutes, which is exactly why the GPU targets exist. Earlier
  attempts at 6-token generation were killed by the sandbox's 4 GiB cap
  (default ctx 4096 ⇒ 896 MiB KV + 261 MiB logits + 965 MB mapped weights +
  512 MiB expert cache). Use `--ctx 256/512` for MoE runs on small machines.
* Synthetic MoE coverage in the suite remains the fast oracle for behaviour:
  schema/laziness, a single-expert layer reproducing its dense twin exactly,
  grouped-prefill vs token-at-a-time routing parity, the LFU+aging+pin
  residency policy under a one-slot cache, and the pinned host L2 tier
  (demote-on-evict, promote-on-request, tier recycling, and bit-identical
  generation with and without the tier).

---

## 4. Reconstructed engine (session start)

The C++ source tree was missing at session start (only docs and CMake
survived); it was reconstructed from the documented architecture and verified
against the untouched `common.cpp` implementation, restoring
`include/krk/common.hpp` (Log levels, `host_alloc`, fp16/bf16, `MappedFile`,
bounds-checked `Cursor`, `now_us`, `format`, `Timer`, product constants). The
suite went green at **401/401** before any new work, and tinyllama generation
is byte-identical to the earlier session's output.

---

## 5. State of the whole project

| area | state |
|---|---|
| GGUF v2/v3 reader, quant decoders (Q4_0…Q8_K, IQ4_NL), CPU oracle | proven (test suite) |
| Tokenizers: SPM unigram Viterbi + byte-level BPE | proven in-suite; real tinyllama (SPM) and Qwen (BPE) verified live |
| Dense LLaMA-family forward (GQA, QK-norm, biases, tied embd, partial RoPE) | proven; live coherent output on tinyllama |
| MoE (`qwen2moe`/`qwen3moe`, lazy experts, batched experts, LFU+aging) | proven in-suite; real checkpoint loads + forwards |
| Sampler (temp/top-k/top-p/min-p/rep-penalty, seeded, greedy) | proven |
| Greedy speculative decoding (draft model, verify-and-accept) | proven bit-identical in-suite |
| Engine (chunked prefill, decode, streaming, stop-string holdback) | proven |
| HTTP/OpenAI server | proven this session |
| HIP kernels (`src/hip/**`) | compiled, run and **measured** on gfx1201 (§8); the fused decode-layer GEMM prototype is bit-exact vs separate gemvs |

---

## 6. Known limits and what is not done

* Device kernels unverified: `gemm_wmma` (gfx11 and the `_gfx12` K-split
  layout), `gemm_simt` (`v_dot2_f32_f16`), MoE `gather_rows` /
  `scatter_axpy_rows` / `axpy`, tiled online-softmax attention, every
  `dequant_chunk<T>` plane on device. `docs/GPU_BRINGUP.md` §7 lists them.
* IQ\* formats except IQ4_NL are rejected at load by design.
* Serving: serialized generation, no prefix cache, no cancellation, no
  concurrent batching, no beam search.
* Speculative decoding: greedy only (sampling falls back), no MTP, no tree
  attention, no n-gram speculation.
* Attention decode launches one block per head (no split-K); sampling is on
  the host; prefill tile is 32x64.
* The fused decode-layer GEMM phase (`src/hip/kernels/fused.hpp`, §2h) is a
  **measured prototype only**: `backend_hip.hip` still launches the seven gemvs
  separately, so none of its measured saving reaches the engine yet.

---

## 7. How to reproduce the verification

```sh
sh ./scripts/build_linux.sh --cpu-only        # CPU oracle build
build-cpu/kraken-tests                         # expect 479/479

sh ./scripts/fetch_moe_model.sh                # ~965 MB real MoE GGUF
build-cpu/kraken --model ../Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --cpu --info
build-cpu/kraken --model ../Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --cpu \
    --greedy --repeat-penalty 1.0 --max-tokens 1 --ctx 256 --prompt "Hi"

build-cpu/kraken-server --model model.gguf --cpu --port 8080
curl -s localhost:8080/health
curl -s localhost:8080/v1/completions -d '{"prompt":"Hi","max_tokens":4,"temperature":0}'
```

---

## 8. Decode bandwidth campaign — `probe_decode` (§2f / §2g / §2h)

Measured on the current host (Windows 11, ROCm/HIP SDK, `build-hip/`,
`KRK_GPU_TARGETS=gfx1201`, Release, `KRK_BUILD_PROBES=ON`, Ninja) on an
**AMD Radeon RX 9070 XT** reporting **32 multiprocessors**
(`multiProcessorCount`). That number is the RDNA3/4 **WGP** count, not the
CU count: HIP's "multiprocessor" on RDNA3/4 is the work group processor,
which AMD's HIP hardware-implementation doc defines as two closely coupled
CUs. The device dump confirms the full chip is accounted for — 32 MPs × 2048
`maxThreadsPerMultiProcessor` = 65536 resident threads = **64 spec CUs ×
1024 threads/CU** (Navi 48, 4096 SPs, 16 GB GDDR6; the spec is **64 CUs**, not
the 44 a task note claimed). The driver does **not** under-report, and the
spec CU count must **not** be fed to the split cost model: a measured
`blk_target` sweep (§2h) shows the driver-reported count is already optimal
and the spec count is ~8% slower. Numbers before the §2i addition are from
`probe_decode.pd36.txt`; the §2i chain, the sweep and the re-validation are
from `probe_decode.pd39.txt` through `pd41.txt`. Shapes are
SmolLM2-135M (n_embd 576, 9/3 heads, head_dim 64, ff 1536, 30 layers), Q4_K
weights in the engine's exact layout (`rows × q4k_row(cols)`, 144 B per 256
values). Tables run at sustained GPU clocks (`steady()`); §1 baselines drift
run-to-run (per-layer 27.04–35.84 µs across runs), so each table below is
read against **its own** §1 baseline.

### §2f — one 256-thread block streams the whole layer

| matrix | shape | bytes |
|---|---|---|
| q | 576×576 | 248,832 |
| k | 192×576 | 82,944 |
| v | 192×576 | 82,944 |
| o | 576×576 | 248,832 |
| gate | 1536×576 | 663,552 |
| up | 1536×576 | 663,552 |
| down | 576×1536 | 497,664 |
| **layer total** | | **2,488,320 B (2.37 MiB)** |

* One 256-thread block, thread-strided 128-bit loads: **79.03 µs ⇒ 31.5
  GB/s**. The **80 GB/s fused-decode gate is MISSED** (31.10 µs would be
  needed). A single block cannot keep the memory pipes full — this rules out
  the "one block per layer" fused design: 30 serial blocks would need
  30 × 79.03 µs just for weights.
* The **true per-layer Q4_K stream is 2,488,320 B (2.37 MiB)** — not the
  746 kB design figure. **746 kB equals only the four attention projections
  (q, k, v, o) each taken at full 576×576 width**: 4 × 576 × 576 × 144 B
  per 256 values = 746,496 B. It omits the FFN entirely and over-counts k/v
  (192 rows, not 576). At 31.5 GB/s even that subset streams in 23.69 µs —
  already most of the 29.09 µs per-layer baseline, before any compute.

### §2g — bandwidth vs block count (the saturation knee)

The same seven-matrix stream with a grid-strided loop and the same
thread-strided 128-bit loads; only the grid varies:

| blocks | time | bandwidth |
|---|---|---|
| 1 | 78.99 µs | 31.5 GB/s |
| 2 | 49.41 µs | 50.4 GB/s |
| 4 | 36.55 µs | 68.1 GB/s |
| 8 | 9.26 µs | 268.6 GB/s |
| 16 | 5.78 µs | 430.8 GB/s |
| 32 | 3.54 µs | 702.2 GB/s |
| **72** | **2.24 µs** | **1109.7 GB/s ← knee** |
| 128 | 2.44 µs | 1020.7 GB/s |
| 256 | 3.18 µs | 781.5 GB/s |
| 512 | 2.62 µs | 951.1 GB/s |

```
GB/s vs blocks (each # ≈ 25 GB/s):
  1   #
  2   ##
  4   ###
  8   ###########
 16   #################
 32   ############################
 72   ############################################  knee
128   #########################################
256   ###############################
512   ######################################
```

* **Saturation knee: 72 blocks (1109.7 GB/s)** — the smallest count within
  5% of the best rate the table reaches. Oversubscription past 128 blocks
  hurts: 256 blocks runs 28% slower than the knee.
* The stream beats §1's **14.78 µs** GEMM phase from **8 blocks** (needs
  168.4 GB/s) — so a multi-block fused GEMM phase *can* win, and 8–72 blocks
  per matrix is the design space §2h prototypes in.

### §2h — fused layer prototype (`src/hip/kernels/fused.hpp`)

All seven Q4_K GEMMs of a decode layer's GEMM phase from **one launch**:
row-parallel blocks per matrix (one warp per output row, eight rows per
256-thread block — exactly the `gemv_kernel` scheme), topped up with §2d
split-K slices where a matrix cannot fill the device on row-parallelism
alone. Partials land in the same persistent fp32 scratch `gemm_launch` uses
and are reduced by the same `gemm_reduce_kernel`. Within a slice the
accumulation order matches `gemv_kernel` exactly, so a split==1 matrix is
bit-identical to its separate gemv launch.

* On-device split decisions: **q=1 k=2 v=2 o=1 gate=1 up=1 down=1** — k/v's
  192 rows (24 row-blocks) cannot fill 32 CUs, so they are cut along K;
  every other matrix fills the device row-parallel.
* **Bit-exact: 0/5184 elements differ** vs the seven separate gemvs (the
  probe's all-ones data makes every partial an exact integer, so even the
  split-K regrouping cannot round).
* Head-to-head: fused **7.99 µs** vs **8.75 µs** for the seven separate
  gemvs (−0.77 µs). The launch-count saving is real but small at M=1 because
  each gemv is already only ~1.2 µs (§2b: the M=1 GEMM is staging-bound
  per block, not occupancy-bound).
* The prototype targets the per-layer baseline (27.04 µs at campaign start;
  29.09 µs in this run) and is **wired into `backend_hip.hip`** as
  `HipBackend::gemm_group()` behind `KRK_FUSED_LAYER` (default on; `=0`
  falls back to the per-matrix `gemm()` loop) — §4's fused-groups replay
  exercises it end-to-end.

### Central-case tok/s re-derivation

Fixed per-step cost outside the 30 layers: head 42.08 µs + embed 1.63 µs =
**43.72 µs**. §1 per-layer split: GEMM phase **14.78 µs** (q 2.04 + k/v
2.75 + o 2.04 + gate/up 5.42 + down 2.53), non-GEMM **14.31 µs** (norm,
rope, kv_append, attention, silu_mul, adds — these have real data
dependencies and stay as separate launches), layer **29.09 µs**, step 916 µs
= **1091 tok/s** (events).

| design point | per-layer | step | tok/s |
|---|---|---|---|
| today (7 separate gemvs, §1) | 29.09 µs | 916 µs | 1091 |
| one-block fused (§2f ceiling) | 79.03 + 14.31 = 93.3 µs | 2415 µs | **414** |
| knee fused (§2g, 72 blocks @ 1109.7 GB/s) | 2.24 + 14.31 = 16.56 µs | 540 µs | **1851** |
| measured prototype (§2h) | 7.99 + 14.31 = 22.30 µs | ≈713 µs | ≈**1400** |

* The one-block design is 2.6× **slower** than today — the §2f single-block
  ceiling (31.5 GB/s) is why the fused design must run multiple blocks per
  matrix.
* The knee design is 1.7× today: at 1109.7 GB/s the weight stream costs
  2.24 µs, leaving the fixed non-GEMM 14.31 µs as the new dominant cost.
* **What a viable fused design must beat:** the per-layer baseline — 27.04 µs
  at campaign start, 29.09 µs in the latest run. Since the non-GEMM half is
  fixed by data dependencies, the operative gate is the GEMM phase: **the
  fused GEMM must beat the 14.78 µs the seven separate gemvs cost, i.e.
  stream the 2,488,320 B layer at ≥ 168.4 GB/s.** §2g shows that is
  reachable from 8 blocks (268.6 GB/s) upward, and the prototype's 7.99 µs
  (≈311 GB/s effective) already clears it.

### §2i — fused non-GEMM chain (`attn_fused_decode`, `src/hip/kernels/attention.hpp`)

One launch does the whole non-GEMM half: RoPE on q **in registers**
(each lane owns its head's consecutive rotation pairs), the KV head's
k row roped **straight into its cache slot** with v copied beside it,
and the decode attention itself — the same warp-strided online-softmax
as `attention_decode_kernel`, one block per query head. The `n_rep`
query blocks sharing a KV head each rewrite that whole cache row
(idempotent plain stores), so no cross-block ordering exists to get
wrong: a block's attention only ever reads the row it wrote itself,
after a `__syncthreads()`.

* **Bit-exact: attention 0/29376, kcache row 0/192, vcache row 0/192
  differ** vs rope + kv_append + attention run separately (three runs).
* Head-to-head: fused chain **4.83–5.65 µs** vs **6.40–6.67 µs** separate
  (−13 to −25% — the launch-overhead saving plus the register-resident q rope).
* Whole fused layer (§2h GEMM + §2i chain, this run set): **23.9–27.0 µs**
  vs §1's **31.3–35.0 µs** ⇒ **765–851 µs/step (1175–1308 tok/s)** vs §1's
  987–1092 µs/step (1004–1011 tok/s): a ~24% kernel-level step win before
  host-side engine overhead.
* The chain is wired into the engine's decode path behind the
  `KRK_FUSED_ATTN` backend flag (§10), with the separate chain as
  the fallback; bit-exactness is enforced by hard asserts in the
  probe (pd42, §10).

`blk_target` sweep on the §2h launch (pd41):

| blk_target | splits (q k v o g u d) | time |
|---|---|---|
| **32 (driver MPs = WGPs)** | 1 2 2 1 1 1 1 | **10.71 µs** |
| 64 (spec CUs) | 1 3 3 1 1 1 1 | 11.52 µs |
| 128 | 2 6 6 2 1 1 2 | 15.72 µs |
| 256 (resident blocks) | 4 1 1 4 2 2 4 | 24.92 µs |

Over-splitting past the driver-reported count costs more in reduce launches
and partial traffic than the extra overlap recovers — the cost model should
keep using `device_cu_count()` as-is.

Reproduce:

```sh
cmake -B build-hip -G Ninja \
  -DCMAKE_PREFIX_PATH=<ROCm SDK> -DKRK_GPU_TARGETS=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release -DKRK_BUILD_PROBES=ON
cmake --build build-hip --target probe_decode   # 0 errors (498 pre-existing nodiscard warnings)
./build-hip/probe_decode.exe                    # §2f/§2g/§2h/§2i are the bandwidth sections
```

## 9. Device CU report (Task 4 — RX 9070 XT)

* `multiProcessorCount=32` is **not** an under-report. On RDNA3/4
  HIP's "multiprocessor" is the **work group processor (WGP)**,
  which AMD's HIP hardware-implementation doc defines as two closely
  coupled CUs. 32 WGPs × 2048 `maxThreadsPerMultiProcessor` =
  65536 resident threads = **64 spec CUs × 1024 threads/CU** — the
  full Navi 48 chip (64 CUs, 4096 SPs, 16 GB GDDR6) is accounted
  for. The spec is **64 CUs** (the "44 CUs" figure matches no
  published spec for this part).
* **Decision: `device_cu_count()` and the §2g knee stay as-is.**
  The split policy wants the driver's multiprocessor count, and the
  §2h `blk_target` sweep (pd41) confirms it empirically: 32 →
  10.71 µs, 64 (spec CU count) → 11.52 µs (+8%), 128 → 15.72 µs,
  256 → 24.92 µs. Feeding the spec CU count into the cost model
  would be measurably slower, not more correct.

## 10. Engine wiring + engine A/B (Tasks 1, 2, 4)

### Wiring (Task 1)

* `HipBackend::attn_fused_chain()` — new virtual on `Backend`
  (`include/krk/backend.hpp`, default returns false so the CPU
  backend keeps the separate chain). One launch does rope-on-q +
  k/v cache append + decode attention for the M=1 row; the engine's
  layer loop (`src/engine.cpp` `forward_core`) tries it first for
  every decode layer and falls back to the separate `rope()` +
  `kv_append()` + `attention()` chain when it returns false.
  Gate: `n_tok == 1 && !qk_norm` — qk_norm must run between the
  projections and the rotation, so the fused path only applies to
  models without it (SmolLM2 qualifies).
* `KRK_FUSED_ATTN=0` restores the separate chain — same env-flag
  pattern as `KRK_FUSED_LAYER`.
* **Correctness fix found while wiring:** the fused kernel originally
  computed `theta = pos * inv_freq[i]`, but `rope_kernel` computes
  `theta = (pos / rope_scale) * inv_freq[i]` — rope's `scale` is a
  position DIVISOR (LLaMA-style RoPE scaling). The fused kernel was
  only correct for `rope_scale == 1` (SmolLM2) and the probe passed
  `1.0f`, hiding the bug. `attn_fused_decode` now takes `rope_scale`
  and matches `rope_kernel` for every model; the engine passes
  `mc.rope_scale`.

### Hard asserts (Task 4)

* `probe_decode` §2h (0/5184) and §2i (0/29376 attn, 0/192 kcache
  row, 0/192 vcache row) diff counts now `exit(1)` on any divergence
  — a future kernel edit that breaks bit-exactness fails the probe
  loudly instead of just printing a number.
* pd42: EXIT=0, bit-exact 0/5184, 0/29376, 0/192, 0/192; §2h fused
  launch 11.01 µs, §2i fused chain 5.11 µs.

### §4g graph-timing fix (post-campaign)

* The §4g "vs launched" graph delta compared **event
  timestamps** against the **wall-clock** launched baseline — a
  methodology mismatch. On gfx1201 event timestamps
  under-report graph replay by up to 12× (112 µs events vs
  1350 µs wall-clock batched on the same run, pd44; pd43
  happened to read 1220 µs vs 1454 µs). The old **"HIP graph
  −92%"** headline was that artifact: pd42's 112 µs event time
  against a 1431 µs wall-clock baseline.
* The probe now headlines the **wall-clock batched** graph time
  (identical methodology to the launched baseline: 30 replays
  + one trailing sync). Corrected gfx1201 graph numbers:
  **break-even** — 1350 µs (741 tok/s) vs launched 1376 µs
  (727 tok/s) = **−2%** (pd44); 1454 vs 1322 µs = **+10%**
  (pd43). Fused graph vs separate graph (both wall-clock):
  −62 µs (pd43), −9 µs (pd44) — a −1 to −4% dispatch saving,
  not −92%.
* gfx1031 is unaffected: its event and wall-clock graph
  readings agree exactly (1885/1885, 1889/1889, 1888/1889
  µs), so the §11 −15% graph figure stands.
* pd44: EXIT=0, bit-exact (hard asserts pass); evidence in
  `probe_decode.pd44.txt`.

### Engine A/B (Task 2)

SmolLM2-135M, gfx1201, **5 samples per arm**, round-robin (each
round runs all four arms so slow thermal/clock drift cancels):

| KRK_FUSED_LAYER | KRK_FUSED_ATTN | decode tok/s (mean) | min–max | decode ms (mean) | per step |
|---|---|---|---|---|---|
| 1 (default) | 1 (default) | **427.6** | 424.0–431.1 | 598.7 | 2.34 ms |
| 1 | 0 | 403.6 | 392.5–408.3 | 634.5 | 2.48 ms |
| 0 | 1 | **440.1** | 437.0–444.3 | 581.6 | 2.27 ms |
| 0 | 0 | 411.1 | 402.8–417.1 | 622.9 | 2.43 ms |

* **Attn chain fusion (KRK_FUSED_ATTN=1): +24.0 tok/s (+6.0%) at
  L=1, +29.0 (+7.1%) at L=0** — 140–161 µs/step. At L=1 the A=1
  arm's worst sample (424.0) still beats the A=0 arm's best (408.3),
  and the L=0 pair is non-overlapping too (437.0 vs 431.1): the gain
  is separated from host-side noise, not an artifact of it. Most of
  it is host-side dispatch: the chain removes 2 launches × 30 layers
  = 60 launches/step from a step that is ~2.3 ms wall of which only
  ~0.8 ms is kernel time (§2i).
* **GEMM group fusion (KRK_FUSED_LAYER=1): −12.5 tok/s (−2.9%) at
  A=1, −7.5 (−1.9%) at A=0** — small but consistent (all 5 paired
  rounds agree). The probe's isolated §2h win (11.01 vs 11.95 µs)
  does not transfer: the engine's separate M=1 path is the
  production `gemv_kernel` (one 256-thread block per 8 output rows,
  576 blocks across the 7 matrices), while the fused group at
  `cu_count`=32 runs 9 blocks total — under-filling the GPU costs
  more than the 180 saved launches recover, and the combined split-K
  scratch adds traffic. Mechanism not yet profiled per-matrix at
  engine shapes.
* Prefill is unaffected by either flag (4.9–5.5k tok/s across arms,
  ±3% noise) — both fusions are decode-only paths.
* **Best measured arm: L=0 × A=1 → 440.1 tok/s (+2.9% over the
  427.6 default).** Recommendation: flip the `KRK_FUSED_LAYER`
  default to off (or gate it on a shape heuristic) once confirmed on
  a second model; the `KRK_FUSED_ATTN` default stays on.

Reproduce:

```sh
for round in 1 2 3 4 5; do for L in 1 0; do for A in 1 0; do
  KRK_FUSED_LAYER=$L KRK_FUSED_ATTN=$A ./build-hip/kraken.exe \
    --model models/SmolLM2-135M-Instruct.Q4_K_M.gguf \
    --bench --max-tokens 256
done; done; done   # raw per-sample table: ab_engine.txt
```

## 11. gfx1031 bandwidth campaign — RX 6700 XT (Task 3)

Same §2f/§2g/§2h/§2i sections re-run on the second machine to separate
chip behaviour from host behaviour. Device verified present before
anything else: `rocm-smi --showproduct` reports `AMD Radeon RX 6700 XT`,
`Card Model 0x73df`, `GFX Version: gfx1031`; `rocminfo` reports the
`amdgcn-amd-amdhsa--gfx1031` ISA with **40 Compute Units, 2 SIMDs/CU,
Max Waves Per CU 32, Max Work-items Per CU 1024**.

* **Host:** `maclin` (ssh `rr@10.0.0.12`), Ryzen 5 5600X, ROCm
  core-10.0 (`/opt/rocm/bin/hipcc`, HIP 7.15). Native build in
  `~/kraken/build-hip` (same source tree, `KRK_GPU_TARGETS=gfx1031`,
  Release, `KRK_BUILD_PROBES=ON`, Ninja). **Five full probe runs**
  (`probe_decode.pd1031{,b,c,d}.txt` plus a re-run of the first to
  regenerate its EXIT marker; run d uses the graph-timing-fixed
  probe — see §10 — and is the first to report a valid gfx1031
  fused-graph number), all `EXIT=0`, all correctness
  checks bit-exact (0/5184, 0/29376, 0/192, 0/192, 0/576, 0/49152,
  §3b PASS). Decode-phase numbers are stable to ±4% across the runs
  (§1 per-layer 65.97–70.97 µs); prefill rows=51 varies ~15%
  run-to-run (clock ramp) and is noted where it matters.
* **Device-count story (mirrors §9):** HIP reports
  `multiProcessorCount=20` with `maxThreadsPerMultiProcessor=2048` →
  20 × 2048 = **40,960 resident threads = 40 spec CUs × 1024
  work-items/CU** — the full chip, exactly the gfx1201 convention
  (HIP's "multiprocessor" = a 2-CU group). The "20 CUs" *label*
  under-counts the CU count by 2×, but the resident-thread budget is
  exact, so `device_cu_count()` stays as-is and the cost model keeps
  using it. The §2h `blk_target` sweep agrees: 20/40/80/160 all run
  20.82–20.95 µs — the driver count is already at the plateau.
* **Portability note:** gfx1031 has no WMMA, so the probe's WMMA
  branches are `#if defined(KRK_GFX11) || defined(KRK_GFX12)`-guarded
  with SIMT fallbacks (added this session); §2c/§2d print
  `skipped (no WMMA on this target)` instead of failing to compile.
  Prefill therefore runs the SIMT path (q 576×576 rows=51 ≈ 231 µs
  vs gfx1201's WMMA 11.49 µs — ~20× slower) and the chunk replay is
  63.94 ms (798 tok/s) vs gfx1201's 4.29 ms (11,879 tok/s).

### §2f — one 256-thread block streams the whole layer

**76.28–79.69 µs ⇒ 31.2–32.6 GB/s — the 80 GB/s fused-decode gate is
MISSED**, same as gfx1201's 79.03 µs / 31.5 GB/s. A single block
cannot fill the memory pipes on either chip; the one-block-per-layer
fused design stays ruled out (30 serial blocks would need
30 × ~77 µs ≈ 2.3 ms just for weights). The 746 kB four-matrix subset
streams in 23.05 µs at this rate.

### §2g — bandwidth vs block count (the saturation knee)

Runs a/b/c per-point (run d reproduces every point within ±0.05 µs):

| blocks | time (µs) | bandwidth (GB/s) |
|---|---|---|
| 1 | 72.37 / 76.38 / 78.55 | 31.7–34.4 |
| 2 | 53.24 / 54.88 / 56.72 | 43.9–46.7 |
| 4 | 19.65–19.72 | 126.2–126.6 |
| 8 | 11.83–11.96 | 208.1–210.4 |
| 16 | 7.99–8.23 | 302.3–311.6 |
| 32 | 5.94–6.01 | 414.0–418.8 |
| **72** | **5.08–5.11** | **486.5–489.9 ← knee** |
| 128 | 4.89–4.92 | 506.1–509.1 ← best |
| 256 | 5.00–5.01 | 496.4–498.1 |
| 512 | 5.98–6.02 | 413.6–415.8 |

* **Saturation knee: 72 blocks (486.5–489.9 GB/s)** — the same knee
  position as gfx1201's 72 blocks, but the plateau is **~2.2× lower**
  (509 vs 1110 GB/s). Both chips cache-resident-stream the 2.37 MiB
  layer (it fits in RDNA2's 32 MB and RDNA4's 64 MB Infinity Cache),
  so the §2g table measures cache bandwidth, not DRAM; RDNA2's cache
  delivers roughly half of RDNA4's.
* The stream beats §1's 32.95–34.99 µs GEMM phase from **4 blocks**
  (needs 71–76 GB/s; 4 blocks give ~126) — the multi-block fused GEMM
  phase is reachable here too, same conclusion as §8.

### §2h — fused layer GEMM phase (one launch)

* **Per-matrix block budget — the split policy never engages:** with
  20 MPs, every matrix fills the device on row-parallelism alone
  (one warp per output row, eight rows per 256-thread block):
  q/o 576 rows → 72 blocks, k/v 192 rows → **24 blocks ≥ 20 MPs**,
  gate/up 1536 rows → 192 blocks, down 72 blocks. On-device decisions
  are therefore **q=1 k=1 v=1 o=1 g=1 u=1 d=1** — unlike gfx1201,
  where k/v's 24 blocks cannot fill 32 MPs and the probe tops them
  up with split-K (k=2 v=2). At SmolLM2 shapes the split-K
  machinery is gfx1201-only.
* **Bit-exact: 0/5184 elements differ** vs the seven separate gemvs
  (all three runs; split==1 matrices are bit-identical by construction).* Head-to-head: fused **20.78–20.89 µs** vs **34.76–35.02 µs** for
  the seven separate gemvs (**−14 µs, −40%**).
 The relative win is
  far bigger than gfx1201's −8.6% because the separate gemvs are 2.2×
  slower here (M=1 gemv q 4.18–4.49 µs vs gfx1201's 2.02) while the
  fused launch is only 2.6× slower (20.85 vs 7.99 µs).
* Viability gate (fused GEMM must beat the 33.0–35.0 µs GEMM phase
  ⇒ stream ≥ 71–76 GB/s): **cleared** — measured fused phase
  20.78–20.89 µs (≈115–119 GB/s effective).
* Fused per-layer (run a): 20.85 µs GEMM + 35.98 µs non-GEMM =
  56.83 µs vs §1's 70.97 µs baseline.

### §2i — fused non-GEMM chain (`attn_fused_decode`)

* **Bit-exact: attention 0/29376, kcache row 0/192, vcache row 0/192
  differ** (all three runs — the §10 hard asserts would have failed the
  probe otherwise; EXIT=0 confirms they pass on gfx1031 too).
* Head-to-head: fused chain **7.46–7.49 µs** vs **16.88–17.12 µs**
  separate (**−9.5 µs, −56%**) — a much bigger relative win than
  gfx1201's −13 to −25%, again because the separate chain is 2.6×
  slower here (17.12 vs 6.48 µs) while the fused chain is only 1.5×
  slower (7.49 vs 5.11 µs).
* Non-GEMM phase: 33.00–35.98 → 23.50–26.35 µs. **Whole fused
  layer: 44.28–47.20 µs** (20.78–20.89 GEMM + 23.50–26.35
  non-GEMM) vs §1's 65.97–70.97 µs ⇒ **1432–1525 µs/step
  (656–698 tok/s) vs 2083–2238 µs/step (447–479 tok/s)** — a
  ~33% kernel-level step win, vs gfx1201's ~24%.

### §3/§4 — attention scaling and end-to-end replays

* §3 decode attention kernel (warp-strided keys, packed dot2):
  1 key 4.02 µs, 32 keys 4.74, 64 keys 5.66–5.68, 101 keys 6.81–6.83,
  256 keys 11.26–11.27, 1024 keys 82.26–83.43, 2048 keys
  88.62–88.71. §3b PASS (all five
  lengths, |dec−tiled| = 0).
* §4 decode step: **2224–2230 µs/step (448–450 tok/s)**,
  enqueue-only 521–646 µs. **Fused-groups replay: 2097–2099
  µs/step (476–477 tok/s), −5.7 to −6.0%** with x 0/576 and
  logits 0/49152 bit-exact — note the sign flip vs gfx1201's §4,
  where the same replay **loses** +5.6% (pd42): on gfx1031 the
  fused group beats the separate per-matrix gemvs even at engine
  shapes, on gfx1201 it does not. HIP-graph replay: 482 nodes
  captured, 1886–1890 µs/step (529–530 tok/s, **−15%**) — a
  repeatable win (events and wall-clock agree exactly on this
  chip). **Fused graph (both chains captured, run d): 2050
  µs/step (488 tok/s), −179 µs (−8%) vs the separate-chain
  graph (2229 µs)** — the first valid gfx1031 fused-graph
  number; the pd1031{a,b,c} runs predate the fused-graph timing
  fix and printed a meaningless +1638 µs from a hardcoded
  gfx1201 constant. gfx1201's graph is only **break-even** in
  the wall-clock regime (−2% pd44, +10% pd43): the −92% once
  quoted here was an event-timestamp artifact (see §10's §4g
  fix note). Enqueue-only share is similar on both chips
  (521–646 of 2224–2230 µs here vs 383 of 1376 on gfx1201),
  so the difference is graph replay-path cost, not dispatch
  share.

### gfx1201 vs gfx1031 — campaign comparison

| metric | gfx1201 (RX 9070 XT, §8/pd42) | gfx1031 (RX 6700 XT, §11) |
|---|---|---|
| HIP MPs × threads = resident | 32 × 2048 = 65,536 (64 CUs) | 20 × 2048 = 40,960 (40 CUs) |
| WMMA | yes (§2c/§2d run) | **no** (SIMT prefill only) |
| §1 per-layer / step | 31.76 µs / 996 µs (1002 tok/s) | 65.97–70.97 µs / 2083–2238 µs (447–479 tok/s) |
| §1 GEMM phase | 15.93 µs | 32.95–34.99 µs |
| §2b M=1 gemv q | 2.02 µs | 4.18–4.49 µs |
| §2f single block | 79.03 µs / 31.5 GB/s — MISSED | 76.28–79.69 µs / 31.2–32.6 GB/s — MISSED |
| §2g knee | 72 blocks / 1109.7 GB/s | 72 blocks / 486.5–489.9 GB/s |
| §2g best | 512 blocks / 1008 GB/s (pd42) | 128 blocks / 506.1–509.1 GB/s |
| §2h splits | q=1 k=2 v=2 o=1 g=1 u=1 d=1 | **all 1** (k/v fills row-parallel) |
| §2h fused vs separate | 7.99 vs 8.75 µs (−8.6%) | 20.78–20.89 vs 34.76–35.02 µs (−40%) |
| §2h blk sweep | 32 best; 64 (spec CUs) +8% | 20–160 flat (20.82–20.95 µs) |
| §2i fused vs separate | 4.83–5.65 vs 6.40–6.67 (−13–25%) | 7.49 vs 17.12 µs (−56%) |
| fused layer | 25.47 µs (809 µs/step) | 44.28–47.20 µs (1432–1525 µs/step) |
| §4 fused groups (engine shapes) | **+5.6% loss** | **−5.7% win** |
| §4 HIP graph (wall-clock) | break-even (−2% to +10%) | **−15%** |
| §4 fused graph vs separate graph | −1 to −4% | **−8%** |

**Takeaways:** (1) the single-block §2f ceiling is chip-independent
(~32 GB/s) — one block never fills either part; (2) the §2g knee sits
at the same 72 blocks on both chips (cache-resident working set),
but gfx1031's plateau is 2.2× lower; (3) the smaller 20-MP device
makes split-K unnecessary at these shapes — the §2h split policy is
inactive on gfx1031; (4) both fusions win *relatively* more on the
slower chip, and the fused-group GEMM flips from a −2% engine loss
on gfx1201 to a −6% probe replay win on gfx1031 — the §10
recommendation to flip `KRK_FUSED_LAYER` off is **gfx1201-specific**;
on gfx1031 the fusion should stay on (engine-level A/B now in §12a: both flags win); (5) `KRK_FUSED_ATTN` wins on
both chips (−40% chain on gfx1031, +6–7% engine tok/s on gfx1201); (6) HIP-graph replay is a −15% win on gfx1031 (fused graph a further −8%) but break-even on gfx1201 — the graph regime is chip-dependent, not a free win.

Reproduce on maclin (the `~/kraken/build-hip` there was configured with
`CMAKE_PREFIX_PATH=/opt/rocm`, `KRK_GPU_TARGETS=gfx1031`, Release,
`KRK_BUILD_PROBES=ON`, Ninja, `/opt/rocm/bin/hipcc`):

```sh
ssh rr@10.0.0.12
cd ~/kraken
cmake -B build-hip -G Ninja -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DKRK_GPU_TARGETS=gfx1031 -DCMAKE_BUILD_TYPE=Release \
  -DKRK_BUILD_PROBES=ON
cmake --build build-hip --target probe_decode
./build-hip/probe_decode > probe_decode.pd1031.txt 2>&1; echo EXIT=$? >> probe_decode.pd1031.txt
```

### §12 — follow-up campaign A–D (gfx1031 engine A/B, second model, event-timing artifact, graph replay path)

Four follow-ups from the §10/§11 campaign: (A) engine-level fusion A/B on
maclin gfx1031, (B) engine A/B on a second GGUF (IQ4_XS) on gfx1201,
(C) why hipEventElapsedTime under-reports HIP-graph replay by up to 12× on
gfx1201 but is exact on gfx1031, and which event-based numbers are still
trustworthy on RDNA4, (D) why HIP-graph replay is break-even on gfx1201 but
−15% on gfx1031, and whether graphs should be enabled conditionally per
target. Raw data: `.ab_1031.txt` + `.ab_1031_b.txt` (A), `.ab_iq4.txt` +
`.ab_iq4_raw.txt` (B), `.graph_event_1201{,_b}.txt` + `.graph_event_1031.txt`
(C/D). Note: the A/B files record **decode ms (lower = better)** despite the
misleading `_tps`/`_tok` suffixes in the raw lines.

#### §12a — (A) gfx1031 engine A/B (maclin, 10 rounds, Q4_K_M, 256-token decode)

Round-robin, 4 arms/round. Medians of 10 rounds:

| arm | decode ms | tok/s |
|---|---|---|
| L=1 A=1 (defaults) | 730.9 | 350 |
| L=1 A=0 | 816.0 | 314 |
| L=0 A=1 | 751.2 | 341 |
| L=0 A=0 | 831.2 | 308 |

* L-fusion **wins +2.8%** at A=1 (730.9 vs 751.2 ms), +1.9% at A=0.
* A-fusion **wins +11.7%** at L=1 (730.9 vs 816.0 ms), +10.7% at L=0.
* Both on vs both off: **+13.7%** (100 ms).
* Outliers: r4 L=0/A=0 1168.0 ms and r10 L=0/A=1 992.3 ms (medians robust
  to both); prefill 61–88 ms is noise.

**Verdict: fusion stays ON on gfx1031 — both flags win at engine level,**
mirroring the probe-level §11 result (fused groups −5.7%, fused chain −56%,
fused graph −8%). The §10 recommendation to flip `KRK_FUSED_LAYER` off is
gfx1201-specific, as suspected. A-fusion is a bigger engine win on gfx1031
(+11.7%) than on gfx1201 (+4.9%) because the separate attention chain is
2.6× slower there (§2i).

#### §12b — (B) second-model engine A/B (gfx1201, SmolLM2-135M IQ4_XS, 5 rounds)

Same geometry as Q4_K_M (30 layers, 576 embd, 9/3 heads, hd=64, ff=1536,
vocab 49152), different quantization. Medians of 5 rounds:

| arm | decode ms | tok/s |
|---|---|---|
| L=1 A=1 (defaults) | 544.3 | 470 |
| L=1 A=0 | 576.6 | 444 |
| L=0 A=1 | 534.2 | 479 |
| L=0 A=0 | 564.2 | 454 |

* L-fusion **costs +1.9%** at A=1 (544.3 vs 534.2 ms), +2.2% at A=0 —
  same sign as Q4_K_M (+2.7% / +2.1%).
* A-fusion **wins −5.6%** at L=1 (544.3 vs 576.6 ms), −5.3% at L=0 — same
  sign as Q4_K_M (−4.9% / −5.5%).
* Both on vs both off: −3.5% (Q4_K_M: −2.9%). Prefill 9–12 ms is noise.

**Verdict: no sign flip — the second model confirms Q4_K_M.** On gfx1201,
`KRK_FUSED_LAYER` loses ~2–3% decode and `KRK_FUSED_ATTN` wins ~5% on both
Q4_K_M and IQ4_XS ⇒ flipping the `KRK_FUSED_LAYER` default off on gfx1201
is quantization-independent (across these two quants); `KRK_FUSED_ATTN`
stays on everywhere. (An earlier read of this data claimed a sign flip on
IQ4_XS; the raw round medians above are the correct parse.)

#### §12c — (C) why hipEventElapsedTime under-reports graph replay up to 12× on gfx1201

Extended `graph_event_probe` (§0–§12) run on both chips: gfx1201
(`.graph_event_1201_b.txt`, Windows ROCm 10 RT preview) and gfx1031
(`.graph_event_1031.txt`, Linux ROCm 7.15). Event-pair ratios
(events/wall; 1.0 = exact):

| check | gfx1201 | gfx1031 |
|---|---|---|
| §1 plain launches, event pair | 1.013 ✓ | 0.999 ✓ |
| §2 1-node graph replay | 0.999 ✓ | 1.001 ✓ |
| §5 R-scaling (1/5/30 replays) | 0.936/1.011/0.996 ✓ | 0.993/0.994/0.999 ✓ |
| §4 events recorded inside the graph | hipErrorInvalidHandle ✗ | hipErrorInvalidHandle ✗ |
| §9 one pair over R=30 replays, N=1/8/64/256/482/512 | 0.937/1.040/0.596/0.220/0.094/**0.089** ✗ | 0.955/0.993/0.999/0.994/1.001/0.999 ✓ |
| §10 default-stream preamble (N=482) | 0.069 and 1.068 (wild) | 1.000/1.001 ✓ |
| §11 pair around ONE replay (N=482) | ✓ ~51 ms stable, matches wall 50.9 ms | ✓ ~59.5 ms, matches wall 59.4 ms |
| §12 long kernels, few nodes (8×~150 µs) | 0.997 ✓ | 1.000 ✓ |
| §8 zero-work events | +1.8 / −25.8 µs (sign unstable) | 6.2 µs |

* The collapse needs **many nodes AND one event pair bracketing many
  replays**: on gfx1201 the event pair saturates near ~4–5 ms while wall
  grows linearly to 48.5 ms (N=512 ⇒ ratio 0.089, an 11× under-report).
  It is not triggered by long kernels (§12), few nodes (§1/§2/§9 N≤8),
  replay count alone (§5), or the default-stream preamble (§10 — noise
  dominates there).
* Events around each individual replay are exact even for a 482-node graph
  (§11): the pair is fine, the loop is not.
* Likely mechanism: the Windows ROCm 10 RT preview's graph-replay path does
  not propagate per-node completions into the event stream, so the a/b
  timestamps land at graph submission (or a fixed late point) instead of
  after all N nodes complete. On gfx1031 (Linux ROCm 7.15) per-node
  completions propagate correctly — events are exact at every node count.
* §4 in-graph event nodes fail with hipErrorInvalidHandle on **both** chips:
  event handles recorded inside a captured graph are invalid after
  capture/replay — a HIP capture limitation, not gfx1201-specific.
* Event timestamps carry tens-of-µs noise (§8) — fine at ≥100 µs scales,
  meaningless below.

**Trust rule for RDNA4/Windows:** event pairs are safe when they bracket few
launches or a single graph replay; for long replay loops use wall-clock.
`probe_decode` §4g's `112 us/step [events]` was exactly this artifact —
(elapsed for R=30 replays)/30 with a saturated elapsed; the §11-style
per-replay events or the wall-clock numbers (1350 batched / 1475 per-sync
µs/step) are the correct ones. No other probe conclusion changes; only the
§4g [events] column on gfx1201 must be read as the wall-clock value.

#### §12d — (D) graph replay path: break-even on gfx1201, −15% on gfx1031

* Bare-launch cost (§6): gfx1201 ~0.9–1.0 µs/launch, replay 0.72–0.76
  µs/node at N≥64 ⇒ only ~0.25 µs/node of dispatch saving. gfx1031
  ~1.1–1.4 µs/launch, replay 0.05–0.34 µs/node at N=8–64 ⇒ up to ~1.1
  µs/node of saving. (gfx1031 §6 N=512 zero-work nodes show an unexplained
  21.4 µs/node slow path; the real 482-node decode graph replays at 3.9
  µs/node including GPU work, so this does not affect §4.)
* §7 decomposition (few heavy nodes): gfx1201 dispatch saving −0.7 to
  −1.3 µs vs GPU-side replay penalty −15.8 to −50.6 µs ⇒ net break-even
  (−0.9% to −2.9%); gfx1031 dispatch saving −1.8 µs, GPU-side penalty
  +4.7 µs (replay is even slightly faster GPU-side) ⇒ net +0.3%. With few
  heavy nodes both chips are break-even — the dispatch saving is negligible.
* The 482-tiny-node decode graph is where gfx1031 wins: bare-launch
  dispatch is 521–646 µs (≈25%) of the 2224–2230 µs step, and replay
  realizes all of it (1886–1890 µs/step, **−15%**) with no GPU-side
  penalty. On gfx1201 the same graph saves only ~120 µs of dispatch and
  pays a 15–50 µs GPU-side penalty ⇒ −2% (pd44) to +10% (pd43), i.e.
  break-even within run-to-run drift.

**Policy: enable HIP graphs conditionally per target.** gfx1031: yes —
−15% decode, fused graph a further −8% (pd1031d). gfx1201: no — break-even
at best, and event timing of graph loops is additionally untrustworthy there
(§12c). The differentiator is (bare-launch dispatch cost × node count) vs
(GPU-side replay penalty), not graph size or replay count per se.

#### §12 recommendation updates

* **gfx1201 defaults:** `KRK_FUSED_LAYER=0` (loses 2–3% on both quants),
  `KRK_FUSED_ATTN=1` (wins ~5% on both quants), HIP graphs off (break-even).
* **gfx1031 defaults:** `KRK_FUSED_LAYER=1`, `KRK_FUSED_ATTN=1` (both win,
  +2.8% / +11.7% at engine level), HIP graphs on (−15%; fused graph −8%).
* **Probe methodology:** on gfx1201/Windows, never time a long graph-replay
  loop with a single event pair — use per-replay events or wall-clock.

Reproduce the gfx1031 probe run on maclin:

```sh
ssh rr@10.0.0.12
cd ~/kraken
cmake --build build-hip --target graph_event_probe
./build-hip/graph_event_probe > .graph_event_1031.txt 2>&1; echo EXIT=$? >> .graph_event_1031.txt
```
