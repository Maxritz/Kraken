# Where the missing performance actually is

The Q8_0 GEMM is 83.7% of a long-prompt prefill run and runs at **13% of this
host's measured int8 ceiling**. Two plausible explanations were each tested and
each explained far less of the gap than they predicted. This note records what
was measured, what was ruled out, and what the evidence points at now — because
the useful part is the elimination, not the two changes that landed.

Baseline for everything below: the 768x12 q8_0 fixture, `--cpu`, greedy,
`--ctx 4096`, 3 interleaved reps, medians.

## 1. The ceiling was never measured, and when it was, it was wrong twice

`isa_peak_gmacs(int8)` used to answer with `f32 peak x 4`, on the argument that
`maddubs` retires 16 MACs per instruction against an fma's 8. That argument
assumes the wide form issues at the same rate as the narrow one. Every `%peak`
column in this repo divided by a number nobody had timed.

It is now probed. The first probe was also wrong: with both `maddubs` operands
loop-invariant the instruction is a constant, the compiler hoists it, and the
loop degenerates to eight integer adds — reported as 301 GMAC/s. The operands
are now bumped every iteration so the dot cannot be hoisted. The corrected
numbers on this box, all cores:

| | GMAC/s |
|---|---:|
| f32 FMA | 102 |
| int8 `maddubs`+`madd` | 301 |

The two are consistent with 2 cores at ~3.2 GHz: 102 is exactly
`2 x 2 x 8 x 3.2e9`, and 301 is ~2.2 ops/cycle of the three-instruction dot.

So the honest statement is: **the GEMM at 39.1 GMAC/s is 13% of 301, and the
kernel alone sustains 24.4 GMAC/s per core against a 150 GMAC/s per-core
ceiling.**

## 2. Hypothesis one: memory traffic. Real, but worth 1.04x.

The loop was `for each output column { for each row tile }`. For every output
column the entire activation matrix is streamed again: on the 512x2048x768 FF
shape that is 805 MB per GEMM call against 1.67 MB of unique weights.

Blocking the columns brought that to ~6 MB and moved the GEMM 37.0 -> 39.1
GMAC/s and prefill 368 -> 384 tok/s, i.e. **1.04x**. Real, correct (bit-identical
logits), and kept — but it is not where the performance is.

## 3. Hypothesis two: instruction count. Worth 1.26x.

The first kernel split the signed weight into two unsigned magnitudes, which
costs two `maddubs` per value. Replacing that with a sign flip costs one:

```
sum(w*x) = sum(|w| * (x * sign(w)))
```

`|w|` is a masked subtract (still exact for `-128`, where `abs_epi8` would
saturate to 127), and the sign is applied to the activation with the same mask.
Per row per 64 values: 5 instructions instead of 8, and one fewer live vector.

Isolated on one core, same data, same block, 3 runs (spread under 1%):

| kernel | GMAC/s |
|---|---:|
| magnitude split, 8 rows (shipped) | 18.2 |
| magnitude split, 4 rows | 17.9 |
| magnitude split, 2 rows | 14.3 |
| **sign flip, 8 rows** | **24.4** |
| sign flip, 4 rows | 19.2 |

1.34x on the kernel; 1.26x end to end (368 tok/s with the old loop order,
384 with blocking). Accuracy is unchanged: 2.329e-08 against the f64 reference,
byte-identical to the previous kernel's error.

## 4. What is actually left, and the evidence for it

**The kernel still spills its accumulators.** `objdump` on the hot function:

```
vmovaps 0xc0(%rsp),%zmm13        <- reload accumulator
vfmadd231ps %zmm5,%zmm6,%zmm13
vmovaps %zmm13,0x440(%rsp)       <- spill it back
```

60 stack `vmov`s in a function whose loop body should be ~50 instructions. This
survived the sign flip (which removed half the arithmetic but none of the
spills), and it is the leading explanation for the remaining gap: the kernel
issues ~5.6 instructions per 64 MACs, so against a probe that sustains 2.2
ops/cycle it should reach ~80 GMAC/s per core, and it reaches 24.4. The
arithmetic says the instructions are not the limit; the spills say where the
cycles go.

Reducing rows-in-flight did **not** fix it (4 rows and 2 rows are both slower),
so the answer is not "use fewer accumulators" — it is to stop the allocator from
spilling the eight it has. The concrete next steps, in order:

1. **Separate the scale application from the dot.** The 512-bit scale vector is
   built per row by two `vbroadcastss` plus a `vinserti64x4`, and the scalar
   `d * dx[...]` products are spilled to the stack (`vmovss %xmm1,0x2b8(%rsp)`).
   Applying the scale as two 256-bit `fmadd`s with `vbroadcastss` operands
   removes that construction and two live vectors per row.
2. **Hoist the eight row pointers into an index array** rather than recomputing
   `r * rstride + b * 32` per row; the address arithmetic is visibly spilling
   too (`mov 0x290(%rsp),%rax`).
3. If neither closes it, tile 6 rows: enough to amortise the weight decode,
   few enough to fit without spilling.

## 5. What this rules out, so it is not re-tried

* **Widening 256 -> 512 bits**: 1.07x. The core does not issue 512-bit ops at
  twice the 256-bit rate, so width is not a lever here.
* **Memory blocking alone**: 1.04x once instructions were the real cost.
* **Fewer rows in flight**: strictly worse (2 rows is 14.3 vs 24.4).

## 6. Reproducing

```bash
g++ -O3 -march=native -Iinclude tools/bench_q80_gemm.cpp build/libkrk_core.a -o bg
./bg 768 2048 512          # isolated kernel, per variant, one core
KRK_CPU_GEMM_NC=0 ./build/kraken ... --kprof   # the unblocked loop order
```

`tools/bench_q80_gemm.cpp` is deliberately standalone: it measures the kernel
with the data pre-laid-out, so a number from it cannot be explained by cache
behaviour or by the engine's loop order.
