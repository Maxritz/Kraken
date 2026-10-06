# GPU bring-up checklist

Everything here is about the **device** half of KRAKEN. The host half (GGUF
reader, quantizers, tokenizer, sampler, model loader, forward pass, CPU
backend) is already proven: `kraken-tests` is 2329/2329, and the CPU backend is
the **oracle** every device build is checked against. Work top to bottom. Each
step is "done" when its pass criterion is met — not when it compiles.

Hardware priority is fixed: **CPU oracle → RX 6700 XT (`gfx1031`, RDNA2,
primary) → RX 9070 XT (`gfx1201`, RDNA4, secondary)**.

---

## 0. Prerequisites

* **Windows 11**: install the *ROCm/HIP SDK for Windows* (AMD). It ships
  `hipcc.bat`. `scripts/build_windows.ps1` searches `%HIP_PATH%`, then
  `C:\Program Files\AMD\ROCm\*\bin`, then `...\AMD\HIP\*\bin`.
* **Linux**: ROCm 6.x dev packages; `hipcc` on PATH.
* Adrenalin driver current for the card. `HSA_OVERRIDE_GFX_VERSION` is **not**
  needed on either target: both are natively supported by the HIP SDK.
* The kernels are written against `amdclang++` intrinsics (`v_wmma_*`,
  `v_dot2_f32_f16`); a plain `clang++` will not compile `gemm.hpp`.

---

## 1. Build for the card

```powershell
# Windows, one card at a time — start with the primary target
.\scripts\build_windows.ps1 -Targets gfx1031

# fat binary covering the fleet (one --offload-arch pass per target)
.\scripts\build_windows.ps1 -Targets "gfx1031;gfx1201"
```

```sh
sh ./scripts/build_linux.sh gfx1031          # RX 6700 XT
KRK_TARGETS="gfx1031 gfx1201" sh ./scripts/build_linux.sh
cmake -B build -DKRK_GPU_TARGETS="gfx1031" && cmake --build build -j
```

Each `--offload-arch` pass compiles the kernel headers with that target's
`__gfx*__` macros, so there is no runtime dispatch inside the inner loops.

**Pass criterion:** `build-hip/{kraken,kraken-server,kraken-inspect,kraken-tests}`
exist, and the CPU test suite still passes from the same tree (a device build
must not regress the oracle).

---

## 2. Confirm the device is the one you built for

```sh
kraken --model model.gguf --info
```

Look at the banner line and the `device` row:

```
KRAKEN 0.1.0 | AMD Radeon RX 6700 XT (gfx1031, RDNA2, 40 CU, wave32, no WMMA)
```

* The reported `gfx` string must equal the `-Targets` you built. A mismatch
  means the kernel selection is wrong — rebuild. This is a build problem, not
  a runtime one.
* On `gfx1031` the warning `device … has no WMMA — falling back to packed-math
  kernels` is **expected and correct** (RDNA2 has no matrix cores).

**Pass criterion:** `--info` reports your card's `gfx` string and the expected
family (`RDNA2` for gfx1031, `RDNA4` for gfx1201).

---

## 3. Oracle conformance — the important step

The CPU backend and the GPU backend must agree **token for token** on the same
model with greedy sampling. This is how a bad kernel gets localized instead of
mysteriously producing slightly-wrong text.

```sh
# CPU oracle (the ground truth)
kraken --model model.gguf --cpu --greedy --repeat-penalty 1.0 --seed 1 \
       --max-tokens 64 --prompt "The history of computing is a history of abstraction" \
       > /tmp/cpu.txt

# same request on the GPU
kraken --model model.gguf --greedy --repeat-penalty 1.0 --seed 1 \
       --max-tokens 64 --prompt "The history of computing is a history of abstraction" \
       > /tmp/gpu.txt

diff /tmp/cpu.txt /tmp/gpu.txt && echo CONFORMING
```

Notes that matter:

* `--repeat-penalty 1.0` disables the repetition penalty, which makes the
  token stream a pure function of the forward pass.
* Use `--greedy`: sampling pulls a different token through the sampler for
  random reasons (seed drift, host/device rounding), and speculation is
  greedy-only.
* Run the suite from the device build too: `build-hip/kraken-tests`.

**If they diverge**, the first differing token localizes the bug to one op.
Bisect along the forward pass in this order (each op is a `Backend` method in
`include/krk/backend.hpp`):

| # | op | suspect when |
|---|---|---|
| 1 | `embed` | first-token logits are wrong but everything scales |
| 2 | `rmsnorm` | output proportional but wrong norm; check eps and the reduction |
| 3 | `gemm` (wq/wk/wv) | attention shape errors; then `gemv` decode path |
| 4 | `add_bias_rows` | Qwen2/Phi QKV biases |
| 5 | `qk_norm` | Qwen3/Gemma3 per-head norm |
| 6 | `rope` | position-sensitive errors (wrong at pos>0, fine at pos=0) |
| 7 | `kv_append` + `attention` | errors that grow with context length |
| 8 | `silu_mul` | FFN wrong, attention right |
| 9 | `gemm` (wgate/wup/wdown) | dense FFN |
| 10 | `gather_rows`, `scatter_axpy_rows`, `axpy` | **MoE only** — routing right, expert math wrong |
| 11 | `out_norm` + `gemm` (out_head) | text is wrong only in the last projection |

Fastest bisect trick: a 1-layer test model (`kraken-tests` builds one) and
`--max-tokens 1` so each run is seconds.

---

## 4. Performance split — prefill vs decode

```sh
kraken --model model.gguf --bench --max-tokens 128 -v
```

Reports prefill tok/s and decode tok/s separately.

* **prefill** uses the batched path: `gemm_wmma` on gfx11/gfx12,
  `gemm_simt` (64x64 tile, `v_dot2_f32_f16`) on gfx10. If prefill is slow but
  decode is fine, suspect the WMMA fragment layout.
* **decode** is bandwidth-bound: the ceiling is roughly `VRAM bandwidth ÷
  model bytes`. On the RX 6700 XT (~384 GB/s) a 7B Q4_K_M (4.4 GB) tops out
  near 85 tok/s. The `gemv` path reads weights once, dequantizes into
  registers, reduces once — it must never write a dequantized row to memory.
* On `gfx1201` the WMMA builtin **must** use the `_gfx12` suffix, fragments
  are 8 fp16 per lane, and the two lane groups cover different K positions
  interleaved in groups of four. On `gfx11xx` fragments mirror across lane
  groups instead. See `docs/RESEARCH.md`.

---

## 5. MoE on device

For an MoE checkpoint (`kraken --model <moe>.gguf --info` shows `moe`), the
routed-expert path adds `gather_rows` / `scatter_axpy_rows` / `upload_i32` and
three batched GEMMs per expert. Check it in three ways:

1. **Conformance** — the §3 token-for-token comparison against `--cpu`.
2. **Residency independence** — the same run with a one-slot cache and with the
   full expert set resident must produce identical tokens:
   ```sh
   kraken --model moe.gguf --cpu --greedy --repeat-penalty 1.0 --expert-cache-slots 1 ...
   kraken --model moe.gguf --cpu --greedy --repeat-penalty 1.0 --expert-cache-mb 65536 ...
   ```
   This is the invariant the test suite enforces on a synthetic `qwen2moe`
   model; it is the check that catches wrong `upload_i32` staging or a
   `scatter_axpy_rows` row mix-up.
3. **Cache telemetry** — `--bench` prints loads, evictions, hit rate and pinned
   slots. Policy is LFU + aging (pin at 24 hits, decay every 64 acquires).

---

## 6. Sizing VRAM (12 GB 6700 XT, 16 GB 9070 XT)

```
KV bytes        = n_layer × ctx × n_head_kv × head_dim × 2 × act_size
logits workspace = n_vocab × n_embd × act_size        (act_size = 2 on GPU)
expert cache    = your budget (--expert-cache-mb), soft cap
```

Working defaults on a 12 GB card:

| item | size |
|---|---|
| dense weights (Q4_K_M, e.g. 7B) | ~4.4 GB |
| KV at ctx 8192, 8 kv-heads, hd 128, 32 layers | ~1.1 GB |
| output head / logits workspace (152k vocab, 4096 embd, fp16) | ~1.2 GB |
| expert cache | 3–5 GB (`--expert-cache-mb 4096`) |

The budget is a *soft* cap: at least one expert is always resident, so a tiny
cache still decodes correctly, just with reload traffic. If the device is
tight, lower `--ctx` first (KV grows linearly), then the expert cache.

---

## 7. Known-unverified kernels and open gaps

Written against documented AMDGPU builtins, **never executed** in this
repository (no ROCm, no RDNA card in CI):

* `gemm_wmma` gfx12 `_gfx12` K-split layout and its fp8/bf8 shapes.
* `gemm_simt` `v_dot2_f32_f16` inner product on gfx10 (`KRK_HAS_DOT2`).
* MoE `gather_rows` / `scatter_axpy_rows` / `axpy` kernels.
* Tiled online-softmax attention.
* `dequant_chunk<T>` for every K-quant plane on device.

Structural gaps that are knowingly absent (see `docs/ARCHITECTURE.md`):

* No split-K attention (poor occupancy on 32-head models at decode).
* Sampling runs on the host — a 150k-vocab logits download per token.
* Prefill tile is 32x64; a larger BM would help long prompts on the 9070 XT.
* No prefix cache, radix tree or paged KV; the engine is single-sequence.

---

## 8. Report template

Attach these when a bring-up is done or when filing a kernel bug:

```
OS / driver:            Windows 11 24H2 / Adrenalin <version>   (or Linux + ROCm <ver>)
HIP SDK:                <version>
Build command:          .\scripts\build_windows.ps1 -Targets gfx1031
kraken --info banner:   <the full line>
Conformance (§3):       CONFORMING | first divergence at token N
--bench:                prefill X tok/s, decode Y tok/s
--bench (MoE):          loads / evictions / hit rate / pinned
kraken-tests:           <n>/<n>
```
