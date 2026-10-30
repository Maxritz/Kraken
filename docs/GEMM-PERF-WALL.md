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

```bash
./build/kraken-isa-ceiling   # what the machine can do, and what the kernel does
```

## 7. The measured ceiling, not an assumed one

Sections 1-5 argued from an assumed 2.6 GHz and an assumed port count. That was
wrong twice over, so the ceiling is now measured on this host by
`tools/bench_isa_ceiling.cpp` -- same instruction mix, same 8-row shape, all in
one process so the ratios survive the host's mood swings (section 8).

| variant | what it is | of C |
|---|---|---|
| A | registers only, integer accumulate | 115-133% |
| B | A plus the real `cvtepi32_ps` + `fmadd` tail | 93-112% |
| C | B plus the real weight and activation loads | 100% |
| D | C at four rows instead of eight | 95-106% |
| F | the shipped kernel body with the per-block scale chain removed | 55-70% |
| E | **the shipped kernel** | **29-53%** (median ~41%) |

Two things follow, and they are the only solid conclusions this file can offer:

* **E/C = 0.41.** The shipped kernel runs at about 41% of what the same
  instruction mix achieves on this machine. There *is* roughly 2x inside the
  loop, so the earlier instinct -- that the dot itself was near its limit --
  was wrong.
* **F/E = 1.4.** Removing the per-block scale load and its dependency chain is
  worth 1.4x on its own. That is the largest single term inside the loop.
* **F/C = 0.55.** Even with no scale chain at all, the kernel is at 55% of C.
  The rest is the streaming read of the 1.67 MB weight matrix, which C never
  does because its buffer is 8 KiB.

Note that C, B and A all hold their scale in a *register* for the length of the
call. That is deliberate and it is also why the earlier "signflip NO scale"
reading of 42 GMAC/s was not evidence of anything: a compile-time constant
scale lets the compiler drop the whole float tail, so that variant measured a
different kernel. F fixes this by taking the scale as a runtime `f32`.

## 8. This host has two throughput modes

The same binary, back to back, on an otherwise idle box:

| run | C | F | E |
|---|---|---|---|
| 1 | 44.1 | 30.9 | 23.4 |
| 2 | 58.5 | 32.4 | 16.7 |
| 3 | 80.7 | 44.5 | 32.7 |

1.8x between runs, with `cpu/wall = 1.00` and a steal time of 7 ticks. The
process is not being descheduled, so this is either SMT sibling contention or a
frequency policy; whatever it is, it is invisible from inside the process. The
consequence for method is absolute: **an absolute GMAC/s figure from this box is
not repeatable, and only a ratio taken inside one process is.** That is why the
kernel A/B in `bench_q80_gemm.cpp` alternates arms rep by rep, and why every
number in this file is quoted as a ratio or with its spread.

## 9. Why two attempts to shorten the chain both did nothing

Both rewrites removed instructions from the per-block scale construction and
both measured neutral:

| attempt | what changed | measured |
|---|---|---|
| pair-expanded scales (`dx8`) | 1 load instead of 2 broadcasts + insert | 1.04x |
| 64-bit broadcast + `permutexvar_ps` | 8 bytes loaded instead of 64, 3 ops instead of 5 | 1.01x |

Section 7 explains why: F/E = 1.4 but the wins above are ~1.0, so the cost is
the *latency* of the chain inside each block-pair iteration, not the size of the
instruction sequence. Eight independent accumulation chains are not enough to
cover a load -> convert -> multiply -> permute -> `fmadd` dependency that runs
the length of the loop body. Shortening it cannot help; only overlapping it can.

The untested restructuring that follows from that: process **two output columns
per call**, so one activation-scale chain feeds two `maddubs` chains. The MACs
per block-pair stay at 512 (4 rows x 2 columns instead of 8 rows x 1), the
accumulator count stays at 8, and the scale construction halves per MAC. That
is the next experiment, and it is listed as an experiment rather than a fix
because the same reasoning has now been wrong once.

## 10. The in-engine roofline probe does not agree with any of this

`--kprof` prints `roofline probe: int8 N GMAC/s`, and every `%peak` column is
that number, so the column is only as good as the probe. Its loop shape is
reproduced in `tools/bench_isa_ceiling.cpp` as variant P, and in the same
process where C measures 47-57 GMAC/s:

| | GMAC/s | of C |
|---|---:|---:|
| P, probe loop shape, one core | 73.7-99.1 | 173-213% |
| engine's own report, two cores (169-215 per core) | 338-430 | — |

Two threads cannot retire more than twice what one thread does on the same
loop, so the engine's figure is at least 1.7-2.2x higher than the same code
measures standalone. The probe also walks no memory at all, unlike a kernel
that has to fetch its operands, which accounts for P being a further 1.7-2.1x
above C.

**What to use instead.** Compare kernels against C, not against the probe. The
probe is an upper bound on issue width and nothing else; against it the Q8_0
GEMM reads as 9-13% of peak, which understates it by roughly 3x. Of the two
numbers, C is the one a kernel can be held to.

**Why the probe is left as it is.** Two attempts to make it honest were made
and both were reverted: giving each chain its own multiplier (removing a
possible common-subexpression win) moved the reported figure from 430 to 338
without changing what the loop does, and replacing the register operand with an
L1 load produced 175 in one run and 430 in the next three. Neither could be
validated from inside the process, so neither was shipped. The disagreement is
recorded here rather than papered over.

## 11. Corrections to earlier claims in this file

* **"int8 301 GMAC/s, consistent with 2 cores at 3.2 GHz"** (README) is not
  supported. This host reports 2.60 GHz nominal, and a dependent FP add with a
  4-cycle latency takes 2.14 ns, i.e. ~1.87 GHz effective; `cpu/wall` is 1.00
  with a steal time of 7 ticks, so the gap is not descheduling. The 301 figure
  also exceeds what the identical loop shape measures standalone (section 10).
* **"the dot itself is the cost"** was already retracted in section 4 and stays
  retracted, but the reason given there was wrong: the 42.2 GMAC/s reading that
  retracted it came from a compile-time-constant scale, which lets the compiler
  delete the int-to-float tail. Section 7's F variant redoes that measurement
  with a runtime scale and finds 1.4x, not 1.8x.

## 12. Ablation: which instruction group actually costs what

`tools/bench_q80_ablate.cpp` takes the production kernel and changes exactly
one component per rung, all rungs in one process (this host moves up to 1.8x
between processes). Rungs marked WRONG change the result on purpose; they price
an instruction, they are not shippable.

768x2048x512, one thread, GMAC/s:

| rung | GMAC/s | of L0 | cost |
|---|---:|---:|---:|
| L0 production | 24.54 | 100% | — |
| **L2 constant weight scale** | **33.01** | **135%** | **34.5%** |
| **L7 integer accumulate** | **30.54** | **124%** | **24.4%** |
| L5 no per-block activation flip | 25.38 | 103% | 3.4% |
| L3 contiguous weight load | 24.92 | 102% | 1.5% |
| L1 constant activation scale | 24.66 | 100% | **-0.5%** |
| L4 constant sign mask | 24.68 | 101% | -0.5% |
| L6 no weight magnitude | 24.64 | 100% | -0.4% |

### What this settles

**The activation scale chain is free.** L1 deletes it entirely — no `dx` load,
no `d0 * d1`, no broadcast, no permute — and the kernel does not get faster.
That is the post-hoc explanation for the three null results in §9 and §11: all
three rewrites optimised a dependency that costs nothing. The chain was never
the problem; the reasoning that produced it came from variant E-vs-F in §7,
where F also removed the ability to *fold the scale into the accumulator*, and
that is what the 1.4x was measuring.

**Two real costs, both fixable:**

1. **The fp16 weight-scale decode, 34.5%.** `fp16_to_fp32` is a branchy scalar
   routine with a denormal-normalisation loop, called twice per block pair. One
   `_mm_cvtph_ps` converts both scales. Measured 4 of 4 interleaved pairs at
   context 4096: **40.9 / 37.5 / 34.9 / 39.9 GMAC/s against 33.5 / 32.5 / 33.8 /
   31.0** for the scalar path, median 38.7 vs 33.0, i.e. **1.17x**. Bit-identical
   — checked across all 65536 half values, zero mismatches outside NaN payloads.
   Shipped behind `KRK_CPU_Q80_NOF16C=1` as the escape hatch.
2. **The int-to-float tail, 24.4%** (`cvtepi32_ps` + `fmadd` per row per block
   pair). Not yet addressed. Removing it means either VNNI-style integer
   accumulation with a per-block scale applied at the end, or accepting a
   different rounding — and VNNI is absent on this host, so the end-of-loop
   scale would have to be paid for somewhere else.

### What it retires

The "0.9 IPC vs 2.7 in the probe" gap is not a single mysterious stall. It
decomposes into a scalar fp16 routine the compiler could not vectorise (34.5%,
now fixed) and a float conversion/addition pair per 64 MACs (24.4%). After the
fp16 fix the kernel is at roughly 47% of variant C instead of 41%, and the
remaining headroom is the float tail, not the loop structure.

## 13. Deep trace after the f16c fix: what is left

Same ablation tool, re-run once the fp16 decode was replaced. Two new
measurements.

**Row depth is already optimal.** Two 8-row tiles share a weight decode, which
halves weight traffic per MAC and doubles the independent chains — the standard
cure for a stall-bound inner loop:

| rows in flight | GMAC/s | vs 8 rows |
|---:|---:|---:|
| 4 | 24.72 | 90% |
| **8** | **27.42** | **100%** |
| 16 | 26.43 | 96% |

16 rows is *slower*, so the kernel is not short of independent work and the
register pressure of a second tile costs more than the extra ILP buys. Combined
with the earlier finding that 4 rows is also worse, 8 is a genuine optimum and
this lever is closed.

**The int-to-float tail is now the single largest cost**, and it is larger than
the first run suggested:

| rung | GMAC/s | of L0 |
|---|---:|---:|
| L0 production | ~24.5 | 100% |
| **L7 integer accumulate** | **34.85** | **144%** |

So `cvtepi32_ps` + `fmadd`, two instructions per row per 128 values, are worth
44% of the kernel. That is disproportionately expensive for two instructions out
of roughly six per row per block pair, and the reason is port pressure:
`vcvtdq2ps` zmm issues on one port on this core, and it competes with
`vpmaddwd`. It cannot be removed without either VNNI (absent here) or a format
change that gives a row one scale instead of one per 32 values.

The remaining smaller rungs are not worth chasing: the per-block activation
flip is 3.4%, the two-load weight assembly 1.5%, and the sign mask and magnitude
formation approximately nothing.

## Re-measured 2026-10-30, with VNNI and after the scale table

The paragraph above ends "cannot be removed without either VNNI (absent here) or
a format change". VNNI is *not* absent on this host, and the comparison that
said the tail was unremovable was being made against the wrong ceiling. Two
corrections, both from `tools/bench_isa_ceiling.cpp` in one process:

| variant | GMAC/s | of C |
|---|---:|---:|
| C loads + int->float tail (maddubs pair) | 70.7-76.2 | 100% |
| **V0 VNNI, registers only** | **86.4-93.2** | 122% |
| **V1 VNNI + loads + int->float tail** | **78.2-80.6** | 111% |
| E production kernel, 768x2048x512 | 32.2 | **40% of V1** |

So the number the kernel should be judged against is **V1, the exact
instruction mix it ships**, not the maddubs ceiling. Against V1 the production
kernel is at 40%, not at 105% as the first comparison suggested (105% of a
ceiling cheaper than the kernel's own instruction mix is not a fraction of
anything).

The remaining 2.5x is now itemised, per 512 MACs, in the shipped GEMM:

* 8 `vpdpbusd` -- the MACs themselves;
* one activation load, one masked sign flip, one `vcvtdq2ps`, one `vfmadd`
  per row = 32;
* the scale vector, `[d0*dx x8, d1*dx x8]`, two broadcasts plus an insert per
  row = 24;
* the block header, once per block pair.

The scale build was removed from the GEMM on the same day by hoisting it into a
per-tile table (16.4% -> 22.4% of the int8 peak, prefill 710 -> 940 tok/s on the
95 MiB fixture; see the commit). The sign flip is the floor: `vpdpbusd` takes
one unsigned operand, the weight magnitude is it, so the activation must be
sign-flipped per (row, block) and there is no cheaper form than the masked
subtract.

**Conclusion, stated as a fraction of something real.** On this 2-vCPU host the
achievable mix ceiling is ~80 GMAC/s per thread; the engine reaches 34. For the
30B-A3B target -- 200 tok/s at 3.3 GMAC/token, i.e. 660 GMAC/s -- a 4-core box
at 2.6 GHz needs 165 GMAC/s per core, which is 55% of int8 peak. The engine is
at 22%, and the harness shows 55% is reachable only by removing the per-row
scale vector as well as keeping the MAC at one instruction per 64 values. That
is the shape of the remaining work, and it is a factor of 2.5, not a factor of
10.
