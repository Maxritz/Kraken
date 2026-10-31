# Ternary int8 dot kernels: TQ1_0 and PTQ1_0

The engine could *read* every ternary format and none of them had a dot kernel,
so both Ternary-Bonsai files and BitNet expanded each weight to f32 before
multiplying. This adds the two kernels and wires them into the same blocked int8
GEMM path Q4_K/Q6_K/Q8_0 use.

## Why ternary is the cheapest dot in the engine

A ternary block is `{-1, 0, +1}` behind one f16 scale `d`. Against an int8
activation with its own per-32 scale `dx_c`, the dot collapses to

```
dot = d_w * SUM_c dx_c * SUM_{i in chunk c} t_i * xq_i
```

No bias term (so no `sum(x)` sidecar of the kind Q4_K's `dmin` needs), no
per-sub-block weight table (unlike Q6_K's 6-bit scales), and **no multiply at
all** in the inner loop: `_mm256_sign_epi8(x, t)` is exactly `x*t` for
`t ∈ {-1,0,+1}` — zero stays zero, a negative negates. Two instructions,
`maddubs` then `madd`, reduce 32 trit-weighted products to eight i32 lanes.

That makes the per-row work proportional to the *stored* form, which is 1.69
(TQ1_0) and 1.75 (PTQ1_0) bits per weight against the 32 bits the f32 expansion
the generic path walks.

## Implementation

| format | values/block | bytes/block | scale at |
|---|---:|---:|---|
| TQ1_0 (BitNet) | 256 | 54 | `w[52..53]` |
| PTQ1_0 (Ternary-Bonsai) | 128 | 28 | `w[26..27]` |

One shared body, `ternary_dot_x8_impl`, parameterised by block size, block bytes
and an unpacker. The trits are expanded **once per block and shared by all eight
rows**, which is the same amortisation the K-quant kernels get from unpacking
`q`/`hmask`/scales once.

The expansion calls the *same* `tq1_trit_at` / `ptq1_trit_at` the dequantiser
uses. A second "optimised" copy of the base-3 unpack
(`trit = ((q * 3^n * 3) >> 8) - 1`) would be a second thing to get wrong; this
one is already pinned against the reference decoder, so the kernel cannot drift
away from `deq_tq1_0` / `deq_ptq1_0`.

They live inside the file's anonymous namespace — they need the private trit
helpers — with thin public wrappers the GEMM reaches.

In `backend_cpu.cpp` both formats join the int8 fast-path gate and are treated as
"no aux array": like Q8_0 they need no per-block sum of the activation, so
`xs`/`xs4` are allocated at size 0 for them rather than zero-filled.

## Correctness

The fixture generator emits ternary payloads as **raw random bytes**. That is
deliberate and makes the test a reference test rather than a plausible one: every
byte pattern is a legal ternary block, and both paths decode the *same* bytes, so
any disagreement is a kernel bug and nothing else.

`KRK_CPU_DOT_INT8=1` takes the kernel, `=0` takes the generic dequantising path.

```
PTQ1_0  int8     134=0.527313471  267=0.498941958  212=0.473279834
        dequant  134=0.527549028  267=0.498761833  212=0.473551810

TQ1_0   int8      90=0.676389456   62=0.641143680  312=0.637578070
        dequant   90=0.672198474   62=0.640155673  312=0.639311433
```

Identical ranking in both, values agreeing to the fourth decimal. That is the
expected size: the integer path sums exactly and applies `d` once per block,
while the dequantising path rounds `d*t` to f32 per element before multiplying,
so the two differ by f32 reassociation and not by arithmetic. The same gap was
measured between the Q8_0 rungs (1.4e-7 per kernel, ~3e-2 in engine logits
through twelve random-weight layers).

## Measured

PTQ1_0, 6 layers, embd 1024, ff 4096, vocab 8192, 23.4 MiB, 14-token prompt,
three interleaved reps alternating the switch:

| rep | int8 kernel | dequantising |
|---|---:|---:|
| 1 | 25.4 tok/s | 19.5 tok/s |
| 2 | **26.8** | **22.4** |
| 3 | 27.3 | 22.8 |

Median **26.8 vs 22.4 tok/s = 1.20×**. The model is 23 MiB against a ~2 MiB L2, so
this is the bandwidth story rather than a cache-resident one; a larger model
should move the ratio toward the 18× traffic advantage, not away from it.

Honest caveat: this is a *removal of a dequantise*, not a new algorithm. It is
worth exactly the traffic it saves, and the generic path was never going to be
competitive at 1.75 bits per weight.

## Remaining ternary and quant work

Kernels still missing, in the order they unblock models:

| format | models blocked | why it is the same job again |
|---|---|---|
| TQ2_0 | BitNet 2-bit variants | 256-value blocks, 66 B, two-bit codes |
| Q1_0 / Q2_0 | BitNet ternary exports | 128-value blocks, `{0,1}` and `{-1,0,+1,+2}` codes |
| PQ2_0 | Ternary-Bonsai-1 group-128 | same 2-bit codec as Q2_0 at group 128 |
| Q2_K / Q3_K | 6 of the 30 files on the support list | 256-value blocks, 16 sub-blocks of 16 |
| Q4_0 | legacy files, 2 of the 30 | simplest unpack in the engine |

Q2_K and Q3_K are the largest single win left: they are mechanically Q6_K with a
different unpack, and Q2_K's `dmin` term needs only the per-16 `sum(x)` that
`quantize_q8_row_sq4` already computes for Q6_K.
