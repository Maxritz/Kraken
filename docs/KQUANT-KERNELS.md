# Q2_K and Q3_K int8 dot kernels

Six of the thirty files on the support list are Q2_K or Q3_K, and neither had a
dot kernel — both dequantised every weight to f32 before multiplying. They are
now on the same blocked int8 GEMM path as Q4_K/Q6_K/Q8_0.

## The block shape, and why it is a port rather than a design

Both are Q6_K's block: 256 values in sixteen 16-value sub-blocks with one scale
per sub-block. So the dot is sixteen short integer dots, each scaled by its own
`d` and by the activation scale of the 32-value chunk the sub-block sits in.
Only the unpack differs:

| format | bytes | value | scale |
|---|---:|---|---|
| Q3_K | 110 | `(q >> 2j & 3) - (hmask bit ? 0 : 4)`, in [-4, 3] | `d_all * (6-bit signed - 32)` |
| Q2_K | 84 | `(q >> 2j & 3)`, in [0, 3] | `d*sc_lo*value - dmin*sc_hi` |

Q3_K's values are **signed and range to -4**, so the magnitude/sign trick that
carries Q8_0 and the ternary kernels does not apply directly — `sign_epi8` is
only exact for -1/0/+1. `_mm_abs_epi8` gives |v| in 0..4, which *is* a legal
unsigned operand for `maddubs`, and `sign_epi8(x, v)` gives `x·sign(v)`, so
`maddubs(|v|, x·sign(v))` is exactly Σ v·x.

Q2_K needs **two** sums per sub-block — Σ v·x for the `d*sc_lo` term and Σ x for
the `dmin*sc_hi` term. That is why it does not use the per-32 `xs` sidecar Q4_K
reads: the per-16 sums happen inside the kernel, on bytes already in registers.

## Two performance mistakes, both measured

**1. Reducing lanes through memory.** The first version summed each 16-value
dot's four i32 lanes by storing to the stack and reloading — 128 times per block
for the eight rows. A store-to-load of a vector lane is a forwarding stall, and
the result was **slower than dequantising**: 24.1 tok/s against 36.1 on Q2_K.

Fixed by keeping the lanes in registers and folding them into an f32
accumulator, which the per-sub-block scale multiply needs anyway, then reducing
once per row with a horizontal add at the very end.

**2. Scalar unpacking.** The unpack filled `vals[256]` one value at a time —
256 iterations per block of shift/mask/hmask-test/subtract. With that in place
Q3_K measured *no better* than dequantising: 23.8 tok/s against 28.0.

Both planes are extractable sixteeen values at a time, and the two
16-bit-lane shifts are legal here for reasons worth recording:

* the 2-bit plane: `_mm_srl_epi16` pulls the neighbouring byte's low bits into
  the high byte of each lane, and masking to 2 bits discards exactly those, for
  both bytes of every lane;
* the hmask bit: the bit index is `s/2`, **uniform across the whole 16-value
  group**, so the same shift is exact once masked to one bit.

`value = q2 + 4*hbit - 4` then falls out as three byte-wise instructions. That
turned Q3_K from a non-win into the largest ratio in this file.

## Correctness

`tools/ta_kquant_dot.cpp` (`kraken-kquant-dot`) compares the kernel against the
reference decoder on the **same** activation:

```
kernel   -> dot_q{2,3}_K_q8(block, xq, dx, n)
expected -> SUM_i dequant_row(block)_i * (xq_i * dx[i/32])   [i64 accumulate]
```

Pinning the activation matters: in the engine-level A/B the two arms differ in
the activation too (the int8 path quantises it to eight bits, the generic path
keeps f32), so a few units in the third decimal is quantisation, not a defect.

The error is reported **relative to the gross magnitude** `Σ|term|`, not to the
result. Random-byte blocks cancel hard — 256 terms of magnitude ~1e3 summing to
~5e2 — and the reference rounds every weight to f32 individually, so a correct
kernel disagrees with it by ~1e-4 of the *result* while agreeing to ~1e-8 of the
*terms*. Dividing by the result would measure the conditioning of the fixture.
(Same lesson as the `ta_q80_tiers` `worst_diff(a, b2, 1.0)` bug: the denominator
decides what the number means.)

```
Q3_K   n=256   worst rel-to-gross 4.108e-08  mean 5.403e-09
Q2_K   n=256   worst rel-to-gross 2.472e-08  mean 5.192e-09
Q3_K   n=1024  worst rel-to-gross 1.649e-08  mean 3.182e-09
Q2_K   n=1024  worst rel-to-gross 1.430e-08  mean 3.302e-09
failure: none
```

Blocks are random bytes with the dtype slots set to real f16 scales, so every
hmask bit, every 2-bit plane and every axis of the scale scramble is exercised;
a real file would only exercise what its quantiser emitted.

## Measured

Six layers, embd 1024, ff 4096, 14-token prompt, `KRK_CPU_DOT_INT8=1` vs `=0`,
alternated within each rep. This host varies up to 1.8× between reps, so the
**paired ratio** is the number and the absolute rates are not:

| model | per-rep ratio (i8 / dequant) | median |
|---|---|---:|
| Q2_K, 35.0 MiB | 2.62, 1.07, 3.18, 1.30, 1.57, 3.61 | **2.1×** |
| Q3_K, 45.8 MiB | 4.50, 8.04, 4.49, 8.26, 4.09, 4.60 | **4.5×** |

6/6 reps favour the kernel for both. Q3_K's ratio is the larger because the
dequantising path expands 3.4 bits per weight to 32 and the SIMD unpack leaves
very little work per stored byte; Q2_K stores more per block (it carries `dmin`
as well as `d`, and needs two sums per sub-block) so its ceiling is lower.

## What is left

| format | models | note |
|---|---|---|
| Q4_0 | 2 of the 30 | simplest unpack in the engine; same recipe |
| TQ2_0 | BitNet 2-bit | 256/66 B |
| Q1_0, Q2_0, PQ2_0 | BitNet + Bonsai-1 | 128-value blocks, same ternary trick |
| Q5_K, Q5_0/1, Q8_K | rarer files | K-quant shape, one more plane |
| IQ family | GSQ-RCO exports | grid + signs codebooks, genuinely different |

Q4_0 is the cheapest remaining win; the IQ family is the only one that is not
the same job again.
