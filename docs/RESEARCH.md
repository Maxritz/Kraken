# Research notes

What this engine is built on, and what each source actually decided for us.

---

## 1. The WMMA guides (GPUOpen, RDNA4, parts 1–3)

* [Part 1 — using the matrix cores](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)
* [Part 2 — fusing WMMA to double K for low-precision GEMM](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/)
* [Part 3 — in-register transpose](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-3/)

These are the definitive statements on RDNA4's `wmma` operand layout, and they
contradict the RDNA3 layout in ways that silently corrupt output if you assume
otherwise.

**RDNA3 (`gfx11xx`)** — `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`:

* 16 fp16 values per lane for both A and B (`v16half`).
* **Matrix replication**: lanes 0–15 and 16–31 carry *identical* copies of every
  fragment. Only `lane % 16` selects a row (A) or a column (B).
* Accumulator: lane `L`, VGPR `g` → row `g*2 + L/16`, column `L % 16`.

**RDNA4 (`gfx1200`/`gfx1201`)** — `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`:

* 8 fp16 values per lane (`half8`), and the `_gfx12` suffix is mandatory.
* **Lane-group split**, not replication: lane group 0 (lanes 0–15) covers K
  {0–3, 8–11}, group 1 covers K {4–7, 12–15}.
* Accumulator: row `(L/16)*8 + g`, column `L % 16`.
* RDNA4 adds fp8 (E4M3) / bf8 (E5M2) inputs and a 16x16x32 int4 shape.

Decisions taken from this:

* The two layouts are `#if`-selected inside one kernel (`wmma_load_a`,
  `wmma_mma`, `wmma_slot`) rather than in two kernels, so the tiling and the
  store path have exactly one implementation.
* Because both A and B fragments are indexed by `lane % 16` in *both*
  generations, both operand tiles live in LDS as row-major `[idx][k]`. That is
  what makes `load_a` and `load_b` mirror images of each other — the same code,
  different base pointer.
* Part 2's advice to get 128-bit loads for the low-precision path is why the LDS
  tiles are padded by `LD_PAD = 8` half-elements: the pad both removes bank
  conflicts and keeps the per-lane 8-element run naturally aligned.

Part 3's in-register transpose is deliberately **not** used. It pays off when
B arrives row-major from global memory; here B is *produced* by our own
dequantizer straight into LDS, so we choose its layout at no cost.

---

## 2. RDNA4 ISA reference

<https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf>

Used to confirm:

* `v_dot2_f32_f16` is present on `gfx10.3` and `gfx11` but is **not** the right
  primitive on `gfx12` — RDNA4 wants WMMA. That is why `KRK_HAS_DOT2` is gated
  on RDNA2/RDNA3 only, and why the SIMT path falls back to scalar FMA on RDNA4
  rather than pretending the packed-math path is still optimal.
* `v_wmma_f32_16x16x16_f16` throughput (8192 ops / 32 cycles) is used to sanity
  check that the GEMM is compute-limited during prefill and the GEMV is
  bandwidth-limited during decode — which is what the two-path split assumes.

---

## 3. ROCm dense WMMA builtin reference

<https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html>
<https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html>

The authoritative builtin signatures, and the source of the exact
`ext_vector_type` aliases used in `src/hip/krk_hip.hpp`:

```cpp
using v16half = _Float16 __attribute__((ext_vector_type(16)));
using v8half  = _Float16 __attribute__((ext_vector_type(8)));
using v8float = f32      __attribute__((ext_vector_type(8)));
```

Both pages note that every WMMA builtin on RDNA3/RDNA4 is **wave32-only**. The
kernels therefore never assume wave64, and the wave reductions are unrolled to
exactly five shuffle steps.

---

## 4. ROCm on Windows

<https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html>

The HIP SDK for Windows supports the Radeon and Ryzen lines natively — no WSL,
no Linux passthrough. Both target cards appear in the supported matrix as
`gfx1031` (RDNA2) and `gfx1201` (RDNA4).

**On version numbering.** AMD's ROCm releases for Windows and Linux have drifted
apart in how they are labelled: the same HIP SDK generation can be advertised as
`7.x` on one page and as a `10.x`-style point release on another. The engine
therefore does not test for a version string at all — it probes
`hipRuntimeGetVersion()` for logging and selects kernels from the `__gfx*__`
macro of the compile pass. If your SDK reports itself as 7.16, 10.1 or anything
else, the build is unaffected. What matters is the target triple you pass to
`--offload-arch`.

The one Windows-specific thing to watch: the HIP SDK's installer must be on
`PATH` for `hipcc.bat` to be found, and `HSA_OVERRIDE_GFX_VERSION` is **not**
needed for either target card — both are natively supported, so do not set it.

---

## 5. GGML quantization block layouts

<https://github.com/ggerganov/llama.cpp> — `ggml-quants.c`, `ggml.c`

The block layouts are not documented anywhere normative, so the dequantizers in
`src/quant.cpp` mirror `dequantize_row_*` in the reference implementation byte
for byte. Two of these are famously easy to get wrong, and both are pinned by a
test:

* **Q5_0/Q5_1 high bit.** The MSB for elements 16–31 comes from bits *16–31* of
  the 32-bit `qh` word, not bits 12–27. Getting it wrong corrupts exactly half
  of every block and produces text that is subtly, maddeningly wrong.
  `test_q5_0_high_bit_layout` fails if it regresses.
* **Q4_K/Q5_K scale unpacking.** The 16 six-bit scale/min pairs are packed into
  12 bytes and require the aux-word shuffle in `get_scale_min_k4`. The device
  version in `src/hip/kernels/dequant.hpp` is a line-for-line twin.

The design rule that follows: the host dequantizers are the oracle, and the
device dequantizers are validated against them. When the two must change, they
change together.

---

## 6. GGUF container

The GGUF v2/v3 header layout (magic, version, tensor count, KV count, KV pairs,
tensor directory, alignment padding, data section) is implemented from the
format description plus observed files. `general.alignment` defaults to 32 and
is honoured rather than hard-coded — some producers use 64 or 128, and assuming
32 silently misreads every tensor offset.

---

## Considered and rejected

| option | why not |
|---|---|
| hipBLAS / rocBLAS for the GEMMs | forces weights into a supported dense type, i.e. dequantize the model. The whole point is to not do that. |
| MIOpen for attention | same problem, plus a much larger dependency for one kernel. |
| repacking weights into a custom kernel layout at load | doubles the load time and gains little: the bottleneck is the dequant arithmetic, not the block header reads. |
| fp16 activations converted to fp32 for the accumulate | already done — WMMA accumulates in f32. The GEMV path uses `v_dot2_f32_f16` so its accumulation is fp32 too, at the packed fp16 rate. |
| an IQ* dequantizer | the codebooks are large tables with awkward edge cases, and IQ* models are rare. Rejected loudly at load instead of decoded wrongly. |
| RoPE YaRN frequency correction | the linear factor is folded in and documented; YaRN's NTK-aware frequency warp needs the original context length and is a known TODO. |
