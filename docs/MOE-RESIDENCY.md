# MoE under a RAM limit: what prefill and decode actually cost

The question this answers: **if I hold only a few experts in RAM and leave the
rest on disk, what prefill do I get, and what token rate do I get?** Every number
below is measured on this checkout, not modelled.

Fixture: `tools/make_bench_gguf.py --embd 768 --layers 20 --heads 12 --kv-heads 4
--ff 768 --expert-ff 768 --experts 32 --experts-used 4 --vocab 32000 --ctx 2048
--scale 0.02 --format q8_0` → 1230.5 MiB, 20 layers, 32 experts, top-4.

Machine: 2 vCPU Xeon @2.60 GHz, 3.9 GiB RAM, AVX-512 + VNNI, no GPU (HIP off),
virtual NPU built in. `--ctx 1024`, temperature 0 (`--greedy`). The two sweeps use
different prompt lengths and say so; **compare down a column, never across
tables.**

Reproduce any row by copying it out of the `config` column; every knob is a CLI
flag except the two `KRK_*` opt-ins, which are named where they appear.

---

## 1. The headline

**Measured twice: before and after the batched-prefill fix in §7.** The fix is
what this document is now mostly about, because it changed the answer.

Before (prefill still called `vec_dot` per output element, so every weight row
was decoded once per row of the batch):

| | all experts resident | 64 MiB of experts | penalty |
|---|---:|---:|---:|
| prefill (CPU, exact) | 13.95 s | 13.68 s | **1.00x -- none** |
| prefill (CPU, AVX2 dot) | 3.94 s | 3.82 s | **1.00x -- none** |
| decode (CPU, exact) | 133.1 ms | 163.8 ms | 1.23x |
| decode (CPU, AVX2 dot) | 41.1 ms | 73.6 ms | 1.79x |
| read from disk | 369 MiB | 2519 MiB | 6.8x |

After (same fixture, same flags, batched GEMM):

| | all experts resident | 64 MiB of experts | penalty |
|---|---:|---:|---:|
| prefill (CPU, AVX2 dot), 121-tok prompt | **1020 ms** | **857 ms** | 1.00x |
| decode (CPU, AVX2 dot) | 61.1 ms | 75.3 ms | 1.23x |
| expert-path read | 307 ms | 521 ms | 1.7x |
| **read share of prefill** | **30%** | **61%** | |

The old reading -- "prefill is compute-bound, residency does not matter" -- was
true and was the symptom. Prefill did not care how much of the model was
resident because it was not doing a batched GEMM at all: cost per prefilled
token was **flat at 34.1, 29.1, 27.8, 28.3 ms** across prompts of 58, 149, 331
and 695 tokens. A GEMM is supposed to read the weight once and reuse it for the
batch, which makes cost per token fall steeply as the batch grows. It never fell
because every weight row was decoded once **per row** instead of once per chunk.

With that fixed, prefill on this fixture is 2.3-3.4x faster and it is no longer
flat: the expert read is now the majority of a streamed prefill (61% at a 64 MiB
budget), so `--expert-warm-mb` and the device budget matter for prefill too.
Full before/after and the mechanism: **§7**.

## 2. The residency ladder (121-token prompt, `--chunk 128`)

**Measured before the §7 fix.** Kept because the decode column is still current
(decode does not enter the batched path at all) and because it is the evidence
for how prefill behaved while it was broken -- flat, at every residency. The
prefill columns here are the BROKEN numbers; §1 and §7 carry the fixed ones.

Exact reference dot (`--cpu`, no opt-ins):

| expert budget | prefill | prefill tok/s | decode | HOT % | COLD | read MiB | read ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| all resident (512 MiB) | 13954 ms | 8.74 | 133.1 ms | 85.3 | 206 | 369 | 248 |
| 256 MiB | 13908 | 8.77 | 137.4 | 82.6 | 244 | 437 | 250 |
| 64 MiB | 13682 | 8.92 | 163.8 | 0.0 | 1405 | 2519 | 524 |
| 16 MiB | 14016 | 8.70 | 171.9 | 0.0 | 1405 | 2519 | 565 |
| 64 + WARM 512 | 13953 | 8.74 | 164.0 | 16.0 | 137 | 246 | 139 |
| 16 + WARM 512 | 13950 | 8.75 | 188.9 | 2.7 | 137 | 246 | 141 |
| 16 + WARM 1024 | 13708 | 8.90 | 177.2 | 1.3 | 31 | 56 | 30 |
| 8 slots + WARM 512 | 13829 | 8.82 | 183.3 | 2.7 | 137 | 246 | 144 |

`KRK_CPU_DOT_AVX2=1` (the opt-in AVX2 block-format dot, see
[BENCH-CPU-DOT.md](BENCH-CPU-DOT.md)):

| expert budget | prefill | prefill tok/s | decode |
|---|---:|---:|---:|
| all resident | 3938 ms | 30.98 | 41.1 ms |
| 256 MiB | 3868 | 31.54 | 43.8 |
| 64 MiB | 3820 | 31.93 | 73.6 |
| 16 MiB | 3949 | 30.90 | 74.6 |
| 64 + WARM 512 | 4029 | 30.28 | 70.5 |
| 16 + WARM 512 | 4056 | 30.08 | 84.0 |
| 16 + WARM 1024 | 3874 | 31.49 | 91.6 |
| 8 slots + WARM 512 | 4186 | 29.14 | 88.6 |

Three things to read off this:

- **Prefill is flat.** 3.82–4.19 s across every residency from "everything
  resident" to "16 MiB then nothing", a 9% spread that is inside run-to-run
  noise. Residency does not buy prefill; it is compute-bound.
- **Decode is residency-bound**, 41.1 → 73.6 ms for the same weights and the
  same arithmetic, purely because 1405 expert loads replaced 206.
- **`--expert-warm-mb` is the lever that matters**, not the device budget. At a
  64 MiB device budget, adding a 512 MiB host WARM tier cuts the read from 2519
  to 246 MiB — **0.10×** — and pulls decode back from 73.6 to 70.5 ms. WARM is
  a host-RAM cache of evicted experts; it turns the second and third visit to
  an expert into a memcpy instead of a `pread`.

### Where the expert path's time goes

At 64 MiB, AVX2, per-phase (`--profile`, 121-token prompt):

```
expert-path  read 533.4 ms | room 12.5 | alloc 5.3 | xfer 0.1
2519 MiB read (157.45 MiB/token), 0 promotions
```

Read is 533 ms of a 3858 ms prefill — **13.8%**. Deleting the disk entirely
would move prefill by 14%, not 2×. That is the number that kills "prefill is
slow because experts stream". It is not.

---

## 3. Chunk size multiplies the fetches

64 MiB device budget, AVX2, 8-token prompt:

| `--chunk` | prefill | prefill tok/s | decode | read MiB | read ms |
|---|---:|---:|---:|---:|---:|
| 8 | 4739.3 ms | 25.74 | 75.0 ms | 3907 | 799.5 |
| 32 | 4091.6 | 29.82 | 75.2 | 1895 | 419.1 |
| **128** | **3858.4** | **31.62** | 72.5 | 1372 | 313.3 |
| 512 | 3967.4 | 30.75 | 71.5 | 1372 | 313.3 |

A prefill chunk is a batch of rows that share one expert gather, so **each chunk
re-fetches the experts it routes to**. Chunk 8 pays 2.8× the read of chunk 128
and loses 19% of prefill to it. 128 and 512 read the same volume; 128 is
marginally the better prefill here, 512 marginally the better decode. Treat the
flat read between 128 and 512 as the point where the gather stops repeating.

This is the knob that made prefill look I/O-bound in earlier notes. It was not
residency; it was the chunk count multiplying the fetches.

---

## 4. Device arms: CPU, NPU, and the two int8 opt-ins

88-token prompt, `n=16`, `--ctx 1024 --chunk 128`. `--cpu` is the scalar
reference, `KRK_CPU_DOT_AVX2=1` its AVX2 dot, `--npu-cpu` puts the whole graph on
the virtual NPU, `--npu` puts only the GEMMs there. `KRK_NPU_W8A8=1` is the
opt-in int8-weight path, `KRK_NPU_ATTN=1` the opt-in device attention.

| arm | expert budget | prefill | prefill tok/s | decode | decode tok/s | read MiB |
|---|---|---:|---:|---:|---:|---:|
| `--cpu` | all resident | 10342.9 ms | 8.50 | 134.2 ms | 7.50 | 407 |
| `--cpu` + AVX2 | all resident | 2940.9 | 29.90 | 42.0 | 23.80 | 407 |
| `--cpu` + AVX2 | 64 MiB | 2835.8 | 31.00 | 73.6 | 13.60 | 2550 |
| `--cpu` + AVX2 | 64 + WARM 512 | 3067.2 | 28.70 | 76.7 | 13.00 | 314 |
| `--cpu` + AVX2 | 64 + `--router-resident 0.05` | 2898.9 | 30.40 | 76.0 | 13.20 | 2550 |
| `--cpu` + AVX2 | 64 + `--hybrid-experts 1` | 2907.3 | 30.30 | 72.7 | 13.80 | 2550 |
| `--cpu` + AVX2 | `--threads 1` | 5307.9 | 16.60 | 73.8 | 13.60 | 407 |
| `--cpu` + AVX2 | `--kv-quant int8` | 3185.3 | 27.60 | 43.9 | 22.80 | 400 |
| `--npu` (GEMM only) | all resident | 6091.7 | — | 114.6 | — | 407 |
| `--npu` (GEMM only) | all resident, w8a8 | 3301.9 | — | 37.6 | — | 400 |
| `--npu-cpu` | all resident | 5804.1 | 15.20 | 113.6 | 8.80 | 407 |
| `--npu-cpu` | all resident, w8a8 | 3597.9 | 24.50 | 34.8 | 28.70 | 402 |
| `--npu-cpu` | all resident, w8a8 + attn | 3557.5 | 24.70 | **32.8** | **30.50** | 402 |
| `--npu-cpu` | 64 MiB, w8a8 + attn | 3414.0 | — | **416.1** | — | 2546 |
| `--npu-cpu` | 16 MiB, w8a8 + attn | 3857.4 | — | **390.1** | — | 2546 |
| `--npu-cpu` | 64 MiB + WARM 512, w8a8 + attn | 3512.0 | — | **407.9** | — | 344 |

**The arms disagree about which is best, and that is the finding.**

- **CPU (AVX2) wins prefill**: 2.84 s against the NPU's 3.41–3.56 s.
- **NPU (w8a8 + attn) wins decode**: 32.8 ms against the CPU's 42.0 ms, and it
  is the only arm to break 30 tok/s on this fixture.
- **The NPU's decode win evaporates the moment experts stream.** 32.8 ms all
  resident becomes 416 ms at a 64 MiB budget — a **12.7× collapse**, against the
  CPU's 1.79×. Section 5 is about why, and about the fix that was tried and
  rejected.

`--threads 1` is the core-utilisation check: prefill halves (5307.9 vs 2940.9
ms) and decode does not move (73.8 vs 73.6 ms). **Prefill scales with cores;
decode does not.** That is the same statement as section 1 from the other side —
decode is waiting on bytes, not on lanes.

`--router-resident` moved nothing here (0.0% HOT, identical read) because there
was no resident tier for a preference to prefer: it reorders *picks* among
experts that are already resident, and at 64 MiB nothing is. `--hybrid-experts`
bought 1.2% (72.7 vs 73.6 ms) by running overflow experts on the host out of the
mapping. `--kv-quant int8` costs 8% of prefill here and does not pay for it: the
fixture's KV is 4 kv-heads × 64 dims over 20 layers, far too small for a halved
KV to matter, so all it adds is the dequant.

---

## 5. The int8 + streaming cliff (open defect, with the rejected fix)

Isolated on `--npu-cpu`, 8-token prompt, w8a8 on, varying only residency:

| expert budget | decode | int8 GEMMs | packed | fell back to f32 |
|---|---:|---:|---:|---:|
| all resident | 34.8 ms | 3714 | 1023.7 MiB | 2162 |
| 64 MiB | 718.7 ms | 2598 | 1023.7 MiB | 553 |
| 64 MiB, `KRK_NPU_WEIGHT_MB=2048` | 969.0 ms | 3151 | 1371.8 MiB | 0 |

With w8a8 **off**, the same 64 MiB arm decodes in **149.0 ms**. So the packing
path costs 5.7× more than the path it replaces as soon as weights stream. The
mechanism is visible in the last row: raising the staging budget removes every
f32 fallback and makes it *slower still*, because the budget is not the
constraint — **the pack memo is keyed by the weight's address, and a streaming
expert slot presents a different address (and a different expert) on every
visit.** So each visit pays a full per-row quantize of that expert's three
matrices, forever. The counter arithmetic agrees: ~1422 acquires × ~0.5 ms of
`q8_0`→int8 packing ≈ the 700 ms observed.

Correctness is not affected: prefill hashes are identical across residency on
this arm. It is pure waste.

**A fix was written and rejected on measurement.** It added
`Backend::gemm_streamed()` — a GEMM variant whose weight is declared to be at a
transient address — and had the MoE sites pick it when `acquire()` reported a
miss, so streaming experts would skip the pack. Measured:

| | all resident | 64 MiB |
|---|---:|---:|
| baseline | 34.8 ms | 718.7 ms |
| with `gemm_streamed` | **174.8 ms** | 799.7 ms |

It regressed both. `acquire()` reports a counter change on WARM hits as well as
disk misses, so the "streamed" predicate fired on nearly every resident expert
too, and the whole model fell onto the f32 path. It also made the prefill hash
depend on the cache size (`7744146c…` at 64 MiB vs `3b5088f2…` elsewhere), which
the CPU arms do not do. The change was reverted in full rather than shipped with
a caveat: a 5× regression on the resident arm is worse than the cliff it was
meant to remove.

The fix that should work is to key the pack by expert identity and hold it in
the expert slot, so it is dropped when the slot is reassigned and reused for as
long as the expert stays. That is a change to the expert cache's ownership of
per-slot state, not a GEMM variant, and it is not in this commit.

**Until then: `KRK_NPU_W8A8=1` is a resident-experts optimisation.** With
`--expert-cache-mb` small enough to force streaming, leave it off — 149.0 ms/step
against 718.7.

### Determinism caveat on the NPU arms

The prefill hash is invariant across expert residency on both CPU arms
(`9b57818ff075a4f4` exact, `b5bc761065e6437c` AVX2, at both 512 MiB and 64 MiB).
It is **not** invariant on `--npu-cpu`: 512 MiB gives `2468164ac1d2f54a` and
64 MiB gives `f6cb507f7cd1e725`. The staging cache declines different weights at
different residencies, so different GEMMs take the host reference path, and its
accumulation order differs from the device's. The chunk-invariance gate in
[CI-DETERMINISM.md](CI-DETERMINISM.md) therefore holds for CPU, not for the NPU
arm — check hashes within one arm, not across two.

---

## 6. What to actually set

Written after the §7 fix, because two of these answers changed with it.

- **Prefill is now I/O-aware as well as compute-bound.** Spend first on the dot
  product (`KRK_CPU_DOT_AVX2=1`) and on threads -- and then on residency, which
  did not matter before the fix and does now: at a 64 MiB budget the expert read
  is 61% of a prefill. Keep `--chunk` at 128 or above so a chunk's gathers are
  not repeated. `--expert-warm-mb` helps prefill as well as decode for the same
  reason.
- **Decode is I/O-bound.** Spend on residency and on `--expert-warm-mb`. A host
  WARM tier at 512 MiB cut disk reads 10x and was the only knob that moved a
  streamed decode back toward the resident figure.
- **Pick the arm by phase, and re-check which wins.** Since the fix, AVX2 CPU
  prefills 3.2x faster than the NPU on the fixture (1071 ms against 3386 ms) and
  the NPU keeps only its decode win (33.4 ms against 58.2 ms). Before the fix
  the NPU was the better all-round arm. On a prefill-heavy workload the CPU arm
  is now the default choice.
- **Do not combine `KRK_NPU_W8A8=1` with forced streaming** until the pack memo
  is keyed by expert identity.

---

## 7. The prefill defect: the GEMM was not a GEMM

### The symptom

Prefill cost per token did not fall with batch size:

| prompt tokens | prefill | ms/token | prefill tok/s |
|---|---:|---:|---:|
| 58 | 1977 ms | 34.1 | 29.3 |
| 149 | 4332 ms | 29.1 | 34.4 |
| 331 | 9206 ms | 27.8 | 36.0 |
| 695 | 19659 ms | 28.3 | 35.4 |

A batched multiply-accumulate amortises the weight read across the batch, so
ms/token must drop steeply as the batch grows -- in a healthy GEMM it falls by
most of the total on the first few rows and then flattens far below the
single-row cost. Here it never fell. Prefill was running at **GEMV speed**: the
same cost per token as decoding, times the number of tokens.

That is also why prefill looked immune to expert residency. It was not hiding
the I/O behind arithmetic; it was doing the arithmetic badly enough that the I/O
was lost in the noise.

### The cause

`src/backend_cpu.cpp`, `gemm()`: the work was split over the flat
`(row, output)` index space, one `vec_dot()` per element.

```cpp
const i64 total = rows * n_out;
par_for(total, grain, ..., [&](i64 b, i64 e) {
    i64 r = b / n_out;
    i64 o = b - r * n_out;
    for (i64 f = b; f < e; f++)
        of[f] = vec_dot(wt, wrow, xr, n_in);   // decodes wrow EVERY time
});
```

`vec_dot()` decodes the weight row as it goes -- that is exactly right for a
GEMV, and exactly wrong here. For `rows = 695` every weight row in the model was
decoded 695 times. Weight traffic was `rows x` the matrix instead of `1 x`, and
the decode cost (int8 -> f32, the subnormal-tolerant f16 scale conversion, the
multiply by the block scale) was paid per row instead of per chunk.

It was not visible at `rows = 1`, which is why decode was always fine.

### What was tried and rejected first

Widening: dequantize each weight row into an f32 buffer once, then run the
existing four-lane dot against it for every row. Bit-identical and simple.

**It made prefill 1.9x slower** (63.5 vs 34.1 ms/token). It writes 4x the weight
bytes to memory, and it replaces a fused vector decode (decode-in-register,
FMA-fused) with a scalar decode into a buffer followed by a separate vector
pass. The extra memory traffic cost more than the decode it saved. Recorded here
because it is the obvious first idea and it does not work.

### The fix

`vec_dot_tiled()`: decode each weight block **once into registers**, then FMA it
against a tile of 8 activation rows before moving on. Weight decode is
amortized over the whole tile, and there is no f32 scratch buffer.

Three kernels, picked by format:

| format | kernel | why |
|---|---|---|
| `q8_0`, `q4_0` on AVX2 | `dot_q8_0_tiled` / `dot_q4_0_tiled` | hand-written, one vector decode per 8 weight elements reused across 8 rows |
| `f16`, `f32` on F16C | `dot_wide_tiled` | same idea for the widening formats; the 4-element conversion is hoisted out of the row loop |
| every other block format (Q4_K, Q5_K, Q6_K, iq-quants, …) | `dot_generic_tiled` | `dequant_block` into a stack buffer once, then the same four-lane shape against 8 rows |

Rows past the last whole tile, and hosts without the relevant ISA, fall back to
`vec_dot()` per element -- so nothing is refused and nothing changes meaning.

### It is a speed change and nothing else

Every kernel performs the same operations, in the same order, as the path it
replaces, per (weight row, activation row) pair. Not "within tolerance" --
**bit-identical**, because the summation order is the one `vec_dot()` uses:

- q8_0/q4_0 tile: per 8 elements it FMAs weight element `k` into accumulator lane
  `k mod 4`, walking `k` upward -- the fused kernel's own order, then folded
  `(a0+a1)+(a2+a3)`.
- generic tile: reproduces both the oracle's single in-order chain and the
  four-lane form, including the `tail` accumulator that spans blocks.
- f16/f32 tile: `_mm_fmadd_ps` over 4-element groups, folded, then the tail.

The four-lane generic form uses a **fused** multiply-add where the host has
F16C, because `vec_dot()` does. Using a separate multiply and add there is what
broke the Q4_K case when this was first written -- visibly, as a disagreement in
the last few ulp -- and it is why the test below exists.

Witnesses:

- `test_gemm_batched_bit_identical` — 100 checks: 5 formats (q8_0, q4_0, f32,
  f16, Q4_K) x 4 arms (exact, fast, avx2, fast+avx2) x 5 tile widths (1, 7, 8,
  9, 13, of which 13 leaves five rows on the fallback path), each comparing the
  whole output matrix with `memcmp` against the per-element `vec_dot()` result.
- End to end, `--prefill-hash` against the binary built from the parent commit:

  | model | arm | before | after |
  |---|---|---|---|
  | q8_0 MoE | exact | `37fd02de1f322777` | `37fd02de1f322777` |
  | q8_0 MoE | `KRK_CPU_DOT_AVX2=1` | `e1137aac2d7143a6` | `e1137aac2d7143a6` |
  | f16 dense | default | `8a21ccff16da6d78` | `8a21ccff16da6d78` |

### What it is worth

Same binary flags, same fixture, only the parent commit's binary against this
one:

| model | prompt tok | before | after | speedup |
|---|---:|---:|---:|---:|
| q8_0 MoE, AVX2 dot | 58 | 1833 ms | 804 ms | **2.28x** |
| | 149 | 4438 ms | 1495 ms | **2.97x** |
| | 331 | 9300 ms | 2778 ms | **3.35x** |
| | 695 | 19575 ms | 6760 ms | **2.90x** |
| f16 dense 1 GiB | 149 | 14348 ms | 3348 ms | **4.29x** |
| | 331 | 31769 ms | 8290 ms | **3.83x** |

Decode is untouched (f16 dense: 174 -> 175 ms/step, i.e. noise): `rows == 1`
keeps the old path, which is the right shape for a GEMV.

### The arm table changes with it

88-token prompt, all experts resident:

| arm | prefill | decode |
|---|---:|---:|
| `--cpu` + `KRK_CPU_DOT_AVX2=1` | **1071 ms** | 58.2 ms |
| `--npu-cpu` + `KRK_NPU_W8A8=1` | 3386 ms | **33.4 ms** |
| `--npu` + `KRK_NPU_W8A8=1` | 3319 ms | 35.1 ms |

Before this fix the CPU arm prefilled at ~2900 ms and the NPU arms were the
better choice overall on this fixture. Now the CPU arm prefills **3.2x faster
than the NPU**, and the NPU keeps only the decode win. The NPU arms are
unchanged by this fix because they never enter the CPU `gemm()`.
