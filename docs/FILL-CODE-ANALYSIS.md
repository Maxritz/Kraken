# The activation fill: what it costs and what it does not

"Fill" here means everything between `gemm()` being called and the first dot
product: allocating the quantised activation buffers, quantising the rows to
int8, and building the per-format auxiliary arrays. This file is an analysis
with measurements, because the intuitive answer ("the fill is overhead, cache
it") is mostly wrong on this engine and the measurements say so.

## Where it lives

`src/backend_cpu.cpp`, the int8 fast path in `CpuBackend::gemm()`:

```
std::vector<i8>  xq(rows * n_in);      // quantised activations
std::vector<f32> xd(rows * nb);        // per-32-value scales
std::vector<i32> xs(rows * nb, 0);     // Q4_K only: sum(x) per block
std::vector<i32> xs4(rows * nq4, 0);   // Q6_K only: sum(x) per 16
for (r = 0; r < rows; r++) quantize_q8_row(...);   // serial
```

## What it costs

Measured directly (`quantize_q8_row` over 512 x 768, best of 7):

| item | cost |
|---|---:|
| quantise 512 x 768 activations | **0.145 ms** |
| x84 GEMM calls per prefill | **12.1 ms** |
| as a share of the profiled prefill (1210 ms) | **1.0%** |

**So the fill is 1% of prefill.** Parallelising it across the two cores would
save at most 0.5% of the run, which is inside this host's noise. It is not a
lever, and any change that adds risk to it is a bad trade. This is worth stating
plainly because "the fill is serial while everything else is parallel" reads like
an obvious win and is not one.

## What was actually wrong, and is now fixed

`xs` and `xs4` were **allocated and zero-filled unconditionally for every
format**. Only Q4_K reads `xs` and only Q6_K reads `xs4`; Q8_0 — the hot format —
needs neither, and was still paying for

```
rows * nb   * 4    = 512 * 24  * 4 =  49 KiB
rows * nq4  * 4    = 512 * 192 * 4 = 393 KiB
                                 = 442 KiB zeroed, per GEMM call, per layer
```

That is 37 MB of pointless memset over a prefill, plus the allocation churn
itself: 442 KiB clears 128 KiB, so glibc serves each one with a fresh `mmap` and
the first touch of every page is a fault. Both arrays are now sized to zero for
the formats that do not read them.

## What the numbers say the fill is *not*

* **Not repeated per weight matrix in a way worth caching.** Each GEMM in a layer
  consumes a different activation (attention output, post-norm MLP input, ...),
  so there is little to reuse. Even if every call were cached to zero, the
  ceiling on the saving is the 1% above.
* **Not the reason rows == 1 is slow.** Decode fills 1 x 768 values — 3
  microseconds — and spends its time in the single-row dot, which measures
  **9.6 GMAC/s against 45.6 for the batched kernel** (3.2% of peak against
  15.2%). Decode is latency-bound on a single dependency chain, not fill-bound.
* **Not a parallelism opportunity worth taking** — see the 1.0% above.

## What actually would help, in measured order

1. **The GEMM's int-to-float tail.** The ablation prices `cvtepi32_ps` + `fmadd`
   at 44% of the eight-row kernel. VNNI is now the default, which removed the
   `vpmaddwd` competing for that port, but the conversion itself remains: one
   per row per block pair, because every 32 values carry their own scale.
   The only structural way out is to hold two block-pairs of integer
   accumulators per row and convert once per 128 values — 16 accumulators for 8
   rows, which fits the 32-register file only barely. Worth one experiment.
2. **Attention at 12.2% of the run, 34.1 GMAC/s, 33.4% of its own peak.** The
   largest remaining non-GEMM component and the one closest to its ceiling.
3. **`gemm_q8_0_i8_r1` (decode) at 3.2% of peak.** The worst efficiency anywhere
   in the profile. Decode has one row of work against a 1.67 MB weight matrix,
   so it is bound by weight streaming; batching rows (multi-sequence) is the
   structural fix, not a kernel rewrite.
4. **Fusion of the 4.9% tail:** `silu_mul` 2.6%, `rope` 1.7%, `kv_append` 0.9%,
   `rmsnorm` 0.7%. Each is a separate pass over the activation with its own
   kernel launch and its own trip through memory.

Items 1-3 are worth roughly the same order of magnitude as each other; the fill
is not on the list.
