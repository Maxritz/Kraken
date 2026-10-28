# Q8_0 was the last format that dequantised in the hot loop

The "never dequant" rule had two documented exceptions. One of them was the
prefill GEMM for Q8_0: `dot_q8_0_tiled()` widened every weight byte to f32 with
`cvtepi8_epi32` and multiplied by the block scale, eight rows at a time. It was
correct, it was bit-identical to `vec_dot()`, and on the 768x12 q8_0 fixture it
was **81.3% of a 2040-token prefill run at 19.3 GMAC/s** — the largest single
cost in the engine, sitting in the one place the rule exists to prevent.

This is that fix, and the two things it taught.

## 1. `VPDPBUSD` is unsigned × signed, and Q8_0 is signed × signed

The obvious repair is the one every VNNI tutorial shows, including Intel's
Cascade Lake sample: bias the weight by 128 so it becomes the unsigned operand,
then subtract the bias it introduced.

```
au = a + 128
acc += dpbusd(au, b)
sum -= 128 * sum(b)
```

**On AVX2 that arithmetic silently breaks.** `VPMADDUBSW` accumulates byte
products pairwise into **int16**, and `255 × 127 × 2 = 64770` overflows it. The
instruction saturates rather than wraps, so the result is not garbage-but-stable
— it is clamped, quietly, in whichever pair happened to be large. The bias trick
is safe only where an int32 accumulator absorbs it, which is exactly what VNNI
adds and exactly what this host does not have.

The magnitude split instead keeps both operands inside the i16 bound with no
correction at all:

```
w = wp - wn         wp = max(w,0) in 0..127
                    wn = |min(w,0)| in 1..128
sum(w*x) = madd(maddubs(wp, x), 1) - madd(maddubs(wn, x), 1)
```

`128 × 127 × 2 = 32512 < 32767`, so nothing saturates for **any** legal byte
pair. `wn` is computed as a wrapping negate of the bit pattern, which is exactly
`256 - pattern`, so the `-128` that the format permits becomes 128 without ever
being negated as a signed value — `abs_epi8` would have saturating-abs'd it to
127 and been wrong for one value in 256. The fixture draws from the full
`[-128, 127]` range specifically to catch that.

Because the split is exact, Q8_0 needs **no `sx[]` array** and no bias term. The
only cost above the dot is the scale product, once per 32-value block.

## 2. Two blocks are one 512-bit register

Q8_0's block is 32 bytes, so at 256 bits the magnitude split costs 8
instructions per 32 MACs — the same order as the f32 path it replaced, which is
why the first version only bought 1.44x. Two blocks are exactly one 512-bit
register, and the activations for two consecutive blocks are contiguous, so the
x side is a single load:

| kernel | per row per 64 MACs |
|---|---|
| dequant (`dot_q8_0_tiled`) | ~18 f32 instructions, weights widened to f32 |
| int8 256-bit | 2 loads, 2 maddubs, 2 madd, sub, cvt, fma |
| int8 512-bit | 1 load, 2 maddubs, 2 madd, sub, cvt, fma, 1 insert |

The weight side needs two loads plus an insert, because Q8_0 puts a 2-byte f16
scale between blocks — but that is decoded **once** and shared by all eight
rows, which is the same 8x share the Q4_K and Q6_K kernels get.

## 3. Measurements

Dense q8_0 fixture (12 layers x 768, ff 2048, vocab 4096), `--cpu`, greedy,
`-n 8`, `--ctx 4096`, `--kprof`. 3 interleaved reps, `median [min, max]`.

### Per arm, all columns

| prompt | arm | prefill tok/s | decode tok/s | attention ms/op | attn %run | attn GMAC/s | %peak |
|---|---|---:|---:|---:|---:|---:|---:|
| ~46 | dequant | 270.1 [267.4, 270.5] | 36.0 [33.5, 39.8] | 0.1 | 2.0% | 13.8 | 10.8% |
| ~46 | i8-256 | 325.5 [317.2, 331.0] | 68.6 [58.5, 88.0] | 0.1 | 2.2% | 17.6 | 13.1% |
| ~46 | **i8-512** | **375.3 [346.1, 426.6]** | 59.7 [45.7, 95.2] | 0.1 | 2.6% | 16.9 | 15.9% |
| ~900 | dequant | 186.8 [176.6, 189.7] | 33.1 [32.9, 34.7] | 8.8 [8.5, 9.0] | 13.0% | 25.1 | 23.7% |
| ~900 | i8-256 | 273.7 [265.0, 304.4] | 38.2 [37.2, 47.1] | 8.9 [8.1, 9.0] | 19.4% | 24.7 | 23.5% |
| ~900 | **i8-512** | **277.7 [256.9, 302.1]** | 44.4 [42.4, 45.0] | 9.8 [8.9, 9.8] | 21.5% | 22.5 | 20.7% |
| ~2040 | dequant | 180.9 [179.0, 189.8] | n/a | 39.9 [37.9, 41.0] | 17.0% | 25.2 | 21.9% |
| ~2040 | i8-256 | 256.8 [245.6, 274.2] | n/a | 41.1 [36.7, 42.2] | 24.3% | 24.5 | 22.0% |
| ~2040 | **i8-512** | **272.1 [244.5, 277.8]** | n/a | 40.2 [39.3, 44.8] | 25.7% | 25.1 | 22.9% |

| prompt | i8-256 / dequant | i8-512 / dequant | i8-512 / i8-256 |
|---|---:|---:|---:|
| ~46 | 1.21x | **1.39x** | 1.15x |
| ~900 | 1.47x | **1.49x** | 1.01x |
| ~2040 | 1.42x | **1.50x** | 1.06x |

The ~2040 arm has no decode row: that model's context is 2048, so a 2040-token
prompt leaves zero timed steps. It is reported as n/a rather than 0.0 tok/s.

Decode improves as well, though no decode kernel changed: rows == 1 still takes
the fast path and lands on the single-row kernel, so decode stops paying for the
f32 widen. 33.1 -> 44.4 tok/s at ~900 tokens (1.34x), 36.0 -> 68.6 at ~46.

### The GEMM op on its own, ~2040 tokens

| arm | ms/op | %run | GMAC/s | vs f32 peak | vs int8 peak |
|---|---:|---:|---:|---:|---:|
| dequant | 9689.5 | 82.0% | 18.0 | 16.1% | — |
| i8-256 | 5986.7 | 73.7% | 29.1 | — | 8.6% |
| **i8-512** | **5048.0** | **70.4%** | **34.5** | — | **10.0%** |

**The op itself is 1.92x faster** (18.0 -> 34.5 GMAC/s) but the run is 1.50x,
because attention is then a larger share of what remains: it goes 17.0% -> 25.7%
of the run without getting any slower.

The `%peak` column changes what it divides by between the arms, and that is the
point: the dequant arm executes f32 FMAs and is measured against the f32 ceiling
(~109 GMAC/s), while the int8 arms execute byte multiplies and are measured
against the int8 ceiling (~345-365 GMAC/s on this box). Comparing 16.1% to 10.0%
across those two columns would be comparing two different denominators.

## 4. Accuracy

The integer path is a **different algorithm**, not a faster evaluation of the
same one: it quantises activations to int8 with per-block scales, which is what
the Q4_K and Q6_K paths have always done. So "does it match the old path" is the
wrong question, and the two should not be expected to agree to 1e-7.

Two separate checks, because they answer different questions:

**The kernel computes the dot it claims to** (`tools/ta_q80_dot.cpp`). A scalar
f64 reference of the same integer arithmetic, over 4 shapes x 8 rows, random
weights drawn from the full `[-128, 127]` range and random activations:

| check | result |
|---|---|
| worst error vs f64 reference | **2.329e-08** |
| single-row vs x8, 256-bit | **0.000e+00** (bit-identical) |
| single-row vs x8, 512-bit | 8.279e-09 |
| 512-bit vs 256-bit, 32 values | 12 differ, max 7.3e-07 rel |

The 256-bit single-row and 8-row kernels are bit-identical because both
accumulate 8 i32 lanes per block. The 512-bit kernel is **not**, and must not be
required to be: it accumulates 16 i32 lanes per two blocks, so the reduction
tree differs even though the integers summed are identical. Requiring bitwise
equality across the two widths would be testing the register layout, not the
arithmetic.

The x8 test also caught the shape that matters: `n = 96` is 3 blocks — odd — and
exercises the dispatcher's wholesale fallback rather than a half-tested tail
path.

**The end-to-end effect of quantising activations.** Greedy `--debug-topk 8` at
2026 tokens, int8 against the f32-activation path:

| | value |
|---|---|
| top-8 candidate set | identical |
| order | one near-tie swap (813 vs 3661, 0.04 logprob apart) |
| worst relative log-prob shift | 1.7e-02 |
| greedy text, 16 tokens | **identical** |

1.7e-2 looks large next to the attention change's 7.7e-8, and it is the honest
number for the *algorithm* — it is the activation quantisation error, present for
every int8 GEMM format in this repo. `KRK_CPU_DOT_INT8=0` restores the exact f32
path for anyone who needs it.

## 5. What the 512-bit widening did and did not buy

Only **1.07x** (307 vs 286 tok/s median). The instruction count says it should
be ~2x, so this kernel is not purely instruction-throughput bound on this host —
the likely cause is that it runs on 2 hardware threads of a shared VM where
512-bit ops downclock, and neither has AVX-512 on the scheduler's mind. It is
kept because it is strictly better, verified, and costs nothing on a host
without AVX-512BW: the dispatcher checks `avx512f` + `avx512bw` at runtime and
falls back to the 256-bit kernel, which Zen 3 and any AVX2-only part will take.

That 2x that did not appear is the honest state of this kernel: at
34.5 GMAC/s it is still far from the instruction bound, and
the next step is not another widening but a measurement of what actually stalls
it.

## 6. Reproducing

```bash
g++ -O2 -march=native -Iinclude tools/ta_q80_dot.cpp build/libkrk_core.a -o tq
./tq                                  # 512-bit path
KRK_CPU_Q80_NOBW=1 ./tq               # 256-bit path
KRK_CPU_DOT_INT8=0 ./build/kraken ... # the dequant path this replaced
python3 tools/bench_full_metrics.py 3 # the table above
```
