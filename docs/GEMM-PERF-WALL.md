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
| E the PRE-TABLE kernel (`dot_q8_0_q8_x8`), 768x2048x512 | 32.2 | **40% of V1** |

> **Correction, same day.** `E` is not the production kernel: the engine has
> called the 16-lane scale-table kernel (`dot_q8_0_q8_wsc8`) since the table
> landed. In one process, interleaved, `kraken-q80-gemm-bench` (768x2048x512,
> nc=128, best of five, four runs) measures **wsc8 at 1.40x `E`** -- 48.1-49.1
> against 34.0-35.7 GMAC/s. The paragraph below is kept because its argument
> (judge against V1, not against the maddubs ceiling) is right; its 40% is about
> the pre-table kernel.
>
> **And the ratio to V1 cannot be stated more precisely than "within the noise
> floor".** `kraken-isa-ceiling` measures the shipped kernel `T` against `V1` in
> ONE process and gets **70% in one call and 110% in the next**, same function,
> same data. This host's run-to-run spread for identical code is up to 1.5x, so
> every absolute GMAC/s number in this file is a same-session ratio and nothing
> else. That is also why "40% of V1" (a cross-session comparison against the
> wrong kernel) should never have been quoted.

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

### Correction, 2026-10-31: the per-tile table was wrong, and it was not 1.5x

The paragraph above used to say the scale build had been hoisted into a per-tile
table and that this took prefill from 710 to 940 tok/s. **Both halves of that
were wrong and both are withdrawn.**

The table was built from the first output row of each row tile and then reused
for the other seven. q8_0 block scales are per row, so that is only correct when
every row of the matrix happens to share them. It does not. A row-scale-varying
fixture (`tests/test_kraken.cpp`, `test_q8_0_scale_array_row_keyed`) disagrees
with the reference kernel by up to **8.3e-01 relative on row 7** while agreeing
exactly on row 0. The suite did not catch it because the fixture never compared
the fast path against the reference kernel; it does now, and the negative
control (`build_q8_0_wscales` over one row fed to all eight) is observed at
8.257e-01.

The 1.5x was measured on that kernel, i.e. on a kernel doing one eighth of the
scale work and the wrong answer. Measured properly, in one process, three runs:

| arm | GMAC/s | of C |
|---|---:|---:|
| E `dot_q8_0_q8_x8`, pre-table | 32.9 / 34.9 / 35.3 | 44% |
| T the compact per-row table (first fix attempt) | 36.9 / 39.5 / 40.8 | 47-53% |
| **T expanded per-row scales, shipped** | **44.4 / 45.4 / 48.7** | **57-64%** |

So the compact table is **~1.1x**, not 1.5x, and it was not worth its own bug.

### What does work: expand both scale factors to the lanes that use them

The inner loop's real cost was never the MAC and never the sign flip alone --
it was building `[d0*dx x8, d1*dx x8]` per (row, block pair): two f32 loads, two
multiplies, two broadcasts and an insert, seven uops per 64 MACs. Both factors
are known before the loop starts. Expand each to the sixteen lanes that consume
it and the whole chain becomes one load and one multiply:

* `build_q8_0_wscales` -> `wsc[(o*npair + p)*16 + {0..7}] = d(2p)`, `+{8..15} = d(2p+1)`,
  per **output row**, built once per column block and reused across every
  activation row tile;
* `build_q8_0_xscales` -> the same for the activation side, built once per
  act-row tile and shared by every output row.

The inner loop is then load, `vpdpbusd`, `vcvtdq2ps`, `vmulps`, `vfmadd`:
**5 uops per 64 MACs**, the same shape as the V1 ceiling. Measured 65-69 GMAC/s
standalone, 44-49 GMAC/s through the engine with the activation array rebuilt per
row tile, against 33-35 before. **1.30-1.40x on the kernel**, and end-to-end on
the 95 MiB dense fixture **prefill 832 -> 928 tok/s (+11.6%, four runs)**; decode
is unaffected because it never used this kernel.

Two things this corrects in the itemisation above: the per-row scale vector was
not ~24 uops per 512 MACs but the dominant term, and the "sign flip is the
floor" claim stands unchanged -- `vpdpbssd` (signed x signed bytes) is
AVX-VNNI-INT8, not AVX512_VNNI, so it is absent on this host and the masked
subtract stays.

Cost: the expanded arrays are 1 byte per weight per array. They are built once
per column block / row tile and live in L2-L3; measured they win, and the same
repack stored as f16 (0.5 B/weight) measured slightly slower (66.2 vs 68.4), so
the kernel is still uop-bound rather than traffic-bound at this size.

**Conclusion, stated as a fraction of something real.** On this 2-vCPU host the
achievable mix ceiling is ~80-89 GMAC/s per thread; the engine now reaches 44-49
(52-60% of V1), up from 33-35. For the
30B-A3B target -- 200 tok/s at 3.3 GMAC/token, i.e. 660 GMAC/s -- a 4-core box
at 2.6 GHz needs 165 GMAC/s per core, which is 55% of int8 peak. The engine is
at 52-60% of its own mix ceiling, so the prefill GEMM is now within ~1.7x of the
instruction mix it runs and the next term is structural -- fewer instructions per
64 MACs (wider tiles, more activation rows per weight load), not a cheaper
per-block tail. That is still a factor of ~1.7, not a factor of 10.

## The decode table: measured, and it is not there (2026-10-31)

The second open item after #81 was "the decode table" -- the same load-time
repack as prefill, applied to the decode GEMV. It was tried on the axis that
works for prefill: expand the *activation* scales to the sixteen lanes that
consume them, once per token, and share that array across every output row of
the tensor; the weight side stays compact. Two shapes were built -- one with a
runtime test in the loop, one specialised at compile time so the loop has no
test at all -- and both were measured against the shipped kernel in the same
binary (`KRK_CPU_GEMV_XSC`), on the q8_0 fixture, four runs each:

| kernel | us/op medians | note |
|---|---:|---|
| shipped | 78.6 | scale chain per (row, pair) |
| pre-expanded activation scales | 75.2 | 3 uops saved, one 64 B load added |

**No win, and a slight loss.** The three uops saved per (row, block pair) are
worth less than the 64-byte load they cost, because the decode loop is not at
its uop limit -- itemising it, the loop is at its memory-level-parallelism
limit, which is why 1 -> 8 streams is 1.7x and 8 -> 16 streams is nothing. The
change was reverted; the comment at the site records the measurement so it is
not re-attempted blind.

So "the decode table" has no content: for prefill the table was worth having
because one output row's scales are reused across eight *activation* rows; in
decode there is one activation row and the scales are consumed once. What is
left in decode is the stream count and the residency story, not the arithmetic.

## Where the decode GEMV's time goes (measured 2026-10-30)

The decode kernel is not memory-bound and it is not MAC-bound -- it is
**uop-bound**, and the byte rate says so. On a 71 MB q8_0 matrix (larger than the
54 MB L3, so it streams):

| variant | ms | GB/s |
|---|---:|---:|
| full kernel | 5.50 | 13.0 |
| bare: load + vpdpbusd + convert | 3.05 | 23.4 |
| no sign flip | 4.22 | 16.9 |
| no f32 scale path | 2.73 | 26.1 |
| *pure streaming read of the same bytes* | *2.80* | *25.5* |

So the ceiling is ~26 GB/s, the core loop already reaches it, and the whole gap
is the work around the MAC: **the scale path is ~49% of the time and the
per-row sign flip ~23%**. Two conclusions that cost nothing to remember:

* anything that removes uops from this loop is worth ~1.3x, because there is
  only 1.9x of headroom in total;
* the sign flip is structural (see this document's opening: `vpdpbusd` takes one
  unsigned operand and the weight magnitude must be it), so the scale path is
  where the headroom actually is.

The first cut taken from that: both f16 block halves now arrive in ONE 32-bit
load and are converted by a single `vcvtph2ps`, and the per-block-pair scale
vector is built with one broadcast plus one `vpermps` instead of two broadcasts
plus an insert. Interleaved A/B, four pairs, same bytes and same process:

| | ms | GB/s |
|---|---:|---:|
| before | 4.47-5.05 | 14.1-15.9 |
| after | 3.50-3.96 | 18.0-20.4 |

1.28x, and the kernel is now at ~70% of its own memory roofline. What remains is
the per-block-pair scale vector itself; the way to remove it is a layout that
stores a row's 32 f16 scales contiguously so one load and one convert yields all
of them, which is a load-time transform and not a kernel change.

## Correction: the decode scale path is 1.4x of ceiling, not 49%

An earlier section of this document credited the per-row scale build with ~49% of
the decode GEMV, from an ablation that read 2.73 ms with the "f32 scale tail" off
against 5.50 ms on. That attribution was wrong: that arm removed the
int-to-float convert and the multiply-add as well as the scale build, and the
three were counted as one.

`tools/bench_gemv_scale.cpp` separates them properly by measuring the bound
instead of a mutilation -- `V1` hoists the scale vector out of the loop entirely,
which is not implementable (the scale depends on the row and block pair) and is
therefore the most any scale work can ever be worth:

| | busy host | quiet host |
|---|---:|---:|
| V0 shipped | 14.0-16.4 GB/s | 19.7-24.7 GB/s |
| V1 scale hoisted (absolute bound) | 19.1-22.5 | 21.3-27.2 |
| V3 prebuilt f16 scale table | 17.7-19.0 | 21.5-25.6 |

So the whole scale build is worth **1.13x on a quiet host and 1.4x on a busy
one**, and the best realistic table is 1.08-1.19x. The kernel is otherwise at the
memory floor: with the scale work removed it runs at ~27 GB/s against a measured
25.5 GB/s pure stream read of the same bytes.

That closes this kernel as a target. Any further decode throughput has to come
from moving fewer bytes per token, not from cheaper instructions per byte -- see
`docs/KINDI-GEMV-COMPARISON.md` for the same conclusion reached from a different
direction.

## Prefill: the same fix was already applied, and here is what is left

The decode GEMV analysis above does not transfer to prefill, and it is worth
being precise about why. Prefill is a batched GEMM: each weight block is decoded
once and reused across every row in the tile, so weight traffic is amortised by
the row count and the loop is **instruction-bound**. A change that removes uops
can therefore pay far more here than in decode — and it did.

`tools/bench_isa_ceiling.cpp` now measures the kernel the GEMM path actually
ships (`T`: `build_q8_0_scales8` + `dot_q8_0_q8_tab8`) alongside the kernel that
predates the scale table (`E`: `dot_q8_0_q8_x8`). That distinction matters,
because `E` is what the earlier "40% of the mix ceiling" figure was measured on,
and the engine has not called `E` since the table landed. Four runs, same
process, same shape (768 x 2048 x 512):

| | GMAC/s | of the maddubs ceiling C |
|---|---:|---:|
| E kernel before the scale table | 30.3-40.2 | 49-63% |
| **T shipped, with the per-tile scale table** | **43.9-62.1** | **75-91%** |
| V1 the VNNI mix ceiling | 56.3-83.8 | 106-131% |

Same-run ratios, which is the only comparison the host's noise permits:

* **the table is worth 1.45x / 1.47x / 1.52x / 1.56x on prefill** — consistently
  larger than the 1.08-1.19x it is worth on decode, exactly as an
  instruction-bound loop should behave (re-measured 2026-10-31 as 1.38-1.41x
  with both kernels timed back to back in one process);
* the shipped GEMM is at **67-105% of V1** (median ~76%), not the 40% that was
  reported against the pre-table kernel.

The residual gap to V1 is one table load plus eight `vpermps` per block pair,
about nine uops per 512 MACs, which is the price of not rebuilding the scale
vector. Removing it needs a layout where a row's 32 block scales are contiguous
so one load and one convert produce all of them. That is the same load-time
repack the decode section points at, and it is the last structural change
available in either kernel.

**Conclusion for prefill: the fix is already in and is worth ~1.5x; the kernel is
within ~1.3x of its ceiling; there is no second scale win to collect.**

## The prefill GEMM is at its own ceiling (measured 2026-10-31)

The item after #82 was "reduce the uops per 64 MACs". Before writing any, the
arms were built in `tools/bench_isa_ceiling.cpp` and measured in one process.
All numbers are GMAC/s on the 2-vCPU Xeon (this host's absolute rate drifts by
up to 3x between sessions; only same-process ratios mean anything).

| arm | what it changes | vs C |
|---|---|---:|
| W the shipped kernel | expanded per-row scales, 8 activation rows | 101-161% |
| V1 the mix ceiling | same instruction mix, no scale chain at all | 100-122% |
| O2 / O4 | 2 / 4 OUTPUT rows per call, to amortise the activation load | 76-149% |
| G2 / G4 | one scale per 128 / 256 weights (LOSSLESS: NO) | 111-168% / 140-193% |

**The shipped kernel is already at the V1 ceiling.** W and V1 land within noise
of each other in every run, which means the uop count is no longer the binding
constraint at this shape: what is left is the instruction mix itself, and the
mix is 5 uops per 64 MACs minimum for q8_0 with `vpdpbusd` (one activation
load, one masked sign flip, one `vpdpbusd`, one `vcvtdq2ps`, one `vfmadd`).

Three things were measured and rejected rather than argued about:

* **Output-row tiling does not pay.** O2/O4 were built twice -- once with the
  weight loads inside the row loop (wrong: 27-38), and once correctly hoisted
  (26-48 against W's 30-50). Halving the activation load per MAC does not pay
  for the sixteen accumulators it costs. Register pressure eats the saving.
* **A longer scale group would be worth only 1.19x, and it is lossy.** G4
  (one scale per 256 weights) reaches 58-60 against W's 49-50: 1.19x. It is the
  only arm that beat the ceiling, and it can only be bought by requantising the
  weights at load time to a scale group q8_0 does not have -- a different
  model, not a faster kernel. Not worth it, and it would break the rule that the
  engine computes the file's own arithmetic.
* **`vpdpbssd` is not available.** The sign split costs two uops per block pair
  (a compare and a masked subtract) and signed x signed would remove both. It
  is AVX-VNNI-INT8, not AVX512_VNNI, and this host does not have it. Checked
  against the compiler's own headers, not assumed.

So the honest statement of the remaining gap to the 200 tok/s target is not
"uops per 64 MACs" -- that is spent. On an AVX-512 core with no AMX the q8_0
GEMM tops out at ~5 uops per 64 MACs and this engine is there. The next real
lever is a format whose scale group is longer than 32 *by construction*
(q4_0 with 32-value groups has the same problem; a 256-group format does not),
and that is a format decision, not a kernel one.

### One redundant computation removed

`build_q8_0_xscales` was being called inside the column-block loop, so the
activation-scale array was rebuilt once per column block for a result that does
not depend on the block at all. It is now built once per call
(`KRK_CPU_XSC_HOIST=0` restores the old shape as the control). End-to-end the
effect is under the noise floor on this host -- the rebuild is small next to the
GEMM it feeds -- but it was free to fix and it is now proportional to the work,
not to the block count.

## A longer scale group by construction: built, measured, rejected (2026-10-31)

The paragraph above ends with "a format whose scale group is longer than 32 *by
construction* ... that is a format decision". `tools/bench_prefill_fmt.cpp`
answers it before any engine code exists: four kernels over the same data at
768x2048x512, plus the quantisation error each layout costs.

**The speed is real, and it is not where the plan expected it to be.** The cost
a longer group removes is not the multiply -- it is the *scale vector build*: the
shipped kernel assembles a 512-bit vector of per-block scale products (two
broadcasts and an insert) every 64 values. One broadcast per group is only
possible when the weight AND the activation are quantised over the same run,
because the kernel multiplies `s_w * s_x` and a single scalar is a single number
only if both sides share the group. Lengthening the weight group alone makes the
kernel slower:

| kernel (8 rows/call, best of 5 in-process, two runs) | run A | run B |
|---|---:|---:|
| q8_0, the shipped kernel | 30.05 | 29.17 |
| 128-value weight group, activation still per-32 | 24.18 | 23.55 |
| **128-value weight group, activation per-128** | **43.64** | **41.49** |
| 128-value weight group with per-32 exponents, activation per-128 | 29.58 | 29.53 |

So the 1.43x is bought by the *activation* group, not the weight group. That is
the finding that reverses the plan's arithmetic: item 4 was written as "the
weight format carries a longer run", but the kernel needs both sides to share it,
and the activation side is the one that cannot be fitted offline.

**And the error is what the plan's gate said it would be.** Relative RMS against
f32 rows (`max` is max per-element error over max|w|; `logit` is the dot-product
error against f32 activations; three row shapes because the layout question is
about the spread of values inside a group):

| layout | B/value | gauss | heavy | hetero |
|---|---:|---:|---:|---:|
| q8_0 (32-value blocks, f16 scale) | 1.0625 | 5.94e-3 | 5.25e-3 | 5.34e-3 |
| 128-value group, max scale | 1.0156 | 7.40e-3 | 6.41e-3 | 6.51e-3 |
| 128-value group, least-squares scale | 1.0156 | 7.38e-3 | 6.37e-3 | 6.53e-3 |
| 128 groups + 2-bit exponents | 1.0234 | 7.23e-3 | 6.32e-3 | 6.42e-3 |
| 128 groups + 4-bit exponents | 1.0312 | 7.21e-3 | 6.31e-3 | 6.42e-3 |
| 128 groups + 8-bit exponents | 1.0469 | 7.17e-3 | 6.28e-3 | 6.47e-3 |

Three things were expected and did not happen:

* **Least squares does not rescue the long group.** 7.38e-3 against max-scaling's
  7.40e-3: on rows without an extreme outlier the max scale is already close to
  the least-squares one, and on heavy-tailed rows it is 6.37 against 6.41. The
  hypothesis was that an outlier wastes the range; measured, it does not.
* **The per-32 exponent ladder buys precision the ladder cannot hold.** It was
  supposed to keep q8_0's per-32 grid while merging four blocks onto one scale.
  A power-of-two ladder can only land on scales `anchor / 2^e`, so the chosen
  grid is up to 2x coarser than the block's ideal one, and the error stays at
  the long-group level (7.2e-3). More exponent bits do not help either -- 2, 4
  and 8 bits all measure within noise -- because the limit is the ladder's
  *granularity*, not its range.
* **The activation group costs error too.** Per-32 to per-128 activation
  quantisation moves the dot by 1.16x (gauss), 1.10x (heavy), 1.37x (hetero)
  against f32 activations.

**Verdict: the format does not ship.** The gate was 1.15x over W *and* logits
that move no more than q8_0's already do. The speed gate passes at 1.43x; the
error gate fails at 1.25x (weights) times ~1.16x (activations), and the
compromise design -- per-32 exponents, which was the whole point of candidate
(b) -- measures **1.00x** over the shipped kernel and no better in error. It
would be a slower format and a more complex one at the same time.

What would change the answer is a real artifact showing q8_0's error is far
below what the model tolerates; nothing here can show that, and shipping a
format whose degradation is unmeasured is exactly the trade this file exists to
refuse. The bench stays in the tree (`kraken-prefill-fmt-bench`) so the decision
can be re-run instead of re-argued, and every prototype is checked against
dequant-then-f32 before its speed is quoted (g4 1.1e-6, g4-with-per-32-activation
2.8e-7, s4 8.5e-4 -- the last is the s-layout's own sub-scale truncation, which
is another reason it is out).

## pv8, the compact pair table: built, measured, REJECTED (2026-10-31)

The last structural idea the sections above leave open is a repack of the scale
path: hoist the weight row's two f16 scales per block pair out of the loop, so
the per-(row, block pair) chain is one 64-bit load, one 128-bit multiply, one
broadcast and one permute instead of five uops plucking a header, converting it
and shuffling it twice. `build_q8_0_wpairs` cost 8 B per output row per block
pair (0.125 B per weight value, +12 % weight traffic) and the kernel that reads
it is **byte-identical to x8 on every cell** -- it is the same arithmetic in the
same order, so a win would have been free and a loss is unambiguous.

`kraken-q80-gemm-bench 768 2048 512`, nc=128, best of five, four runs, GMAC/s:

| kernel | r1 | r2 | r3 | r4 |
|---|---:|---:|---:|---:|
| x8, no table | 33.99 | 35.69 | 34.05 | 34.49 |
| **wsc8 16-lane tables (ships)** | **48.70** | **48.95** | **48.12** | **49.05** |
| pv8 pair table | 44.56 | 45.87 | 44.47 | 45.60 |
| wsc8 + tree reduce (sum order changed) | -- | 47.11 | 46.14 | 49.01 |

**It loses by 7-9 %.** The five uops it deletes were already hidden behind the
pair loop (which is the same thing L1 in the ablation table says: the activation
scale chain is free), while the table it adds is a load in the critical path per
row. So the uop count of the scale family is not the binding constraint at this
shape, and there is no repack left: the kernel is at the 5-uops-per-64-MAC floor
of the instruction mix (one activation load, one masked sign flip, `vpdpbusd`,
`vcvtdq2ps`, `vfmadd`), and what remains between it and V1 is the mix itself.

The kernel now lives in `tools/bench_q80_gemm.cpp` -- the bench that measures it
-- and not in `src/quant.cpp`, because a measured-and-rejected candidate is not
shipped code. The bench prints `identical: x8 vs pv8` on every run, so the
identity claim in the paragraph above is checked by the artifact that reports the
loss.

### The `W` arm is not a shippable 1.38x

`kraken-isa-ceiling`'s `W` arm prints 1.38x over `T` (60.7 against 44.0 GMAC/s in
one process, 2026-10-31). It is a harness artifact, and saying so matters more
than the number: `W`'s ACTIVATION scale table is built from constants
(`e0 = 0.011f + ...`) outside the timed loop, while `T` calls
`build_q8_0_xscales` per row tile INSIDE it. So `W` prices a kernel that does not
need an activation scale at all -- which is L1, the rung the ablation already
showed is free. The only legal form of the idea is hoisting the per-tile build,
which the backend already does (`KRK_CPU_XSC_HOIST`), and its effect is under the
noise floor.

### What B1 closes as

| claim | evidence |
|---|---|
| the engine ships the table kernel; the "40% / 32.2 GMAC/s" numbers are the pre-table one | `src/backend_cpu.cpp` calls `dot_q8_0_q8_wsc8` when `q8_0_wsc_supported()`; `bench_q80_ablate` L0 and `bench_isa_ceiling` E both drive `dot_q8_0_q8_x8` |
| the table is worth ~1.4x, measured interleaved in one process | `bench_q80_gemm` 4-column sweep, four runs |
| no remaining legal repack pays | pv8, bit-identical, 0.93x |
| the distance to V1 is unresolvable here | T/V1 = 70% and 110% in one process, same function |
| the target is arithmetic, not a kernel | 48-49 GMAC/s is ~16 tok/s of prefill per core at ~3 GMAC/token; four cores ~64 against a 200 target |

**Conclusion: B1 is measured and closed with no legal win.** The next lever is a
format whose scale group is longer than 32 by construction -- a different model's
arithmetic, not a faster kernel for this one.
