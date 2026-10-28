# Attention is a layout problem: one gather, 2.2x end to end

The AVX2 attention kernel from #27 took the long-prompt run from 92.7 to 162.7
tok/s and still left attention at **42.6% of the run and 9% of the f32
roofline**. The next thing I tried was the obvious one — process four keys per
horizontal reduce — and it bought nothing. That failure is the useful part of
this document.

## 1. The trace said "kernel"; the microbenchmark said "stride"

Same arithmetic, same compiler flags, three variants, run over 4800 key-rows at
`hd=64`, `n_vis=2048`:

| variant | `pos_stride` = 3072 B (the real cache) | `pos_stride` = 256 B (contiguous) |
|---|---:|---:|
| one key per reduce (as shipped) | **16.14 ns/key** | 2.99 ns/key |
| four keys per reduce | 14.47 ns/key | 2.44 ns/key |
| + software prefetch, 4 ahead | 14.12 ns/key | 2.29 ns/key |
| + software prefetch, 8 ahead | 13.77 ns/key | 2.29 ns/key |

The KV cache is `[position][head][hd]`. One head's slice of one key is **256 B
out of a 3072 B stride** on this model (12 heads x 64), so the score loop reads
four cache lines and uses one. Against a contiguous tile the identical kernel
runs **5.4x faster**.

Every optimisation aimed at the loop — fewer reductions, more accumulators,
prefetch — is capped at ~1.2x, because the loop was never the problem. This is
the same lesson `gemm_q6k` learned at 4x the tile, arriving at attention.

## 2. The fix: amortise the strided pass, do not remove it

The gather does not eliminate the strided traffic. It *shares* it: one pass over
the cache serves **every query row in the chunk**, instead of one pass per
(token, head).

```
for each head h (parallel):
    gather K and V for [base_j .. base_j+span) of head h into a contiguous tile
    for each token t in the chunk:
        run the row body over the tile with row_stride = hd
```

Two consequences that shaped the code:

* **Gated on `n_tok > 1`.** A decode step has one row, so there is nothing to
  amortise and the copy is pure cost. Decode keeps the in-place path.
* **The chunk has to be looped head-major.** The tile belongs to a head and
  every token reuses it; the previous loop was token-major over `(token, head)`
  pairs, which cannot express that.

The row body took a `(kbase, vbase, row_stride)` triple, so **one** copy of the
arithmetic serves both paths. Nothing about the score, the softmax, the sink,
the softcap or the mask changed.

## 3. Full metrics

Model: dense q8_0 fixture (12 layers x 768, ff 2048, vocab 4096, 92.6 MiB),
`--cpu`, greedy, `-n 8`, `--ctx 4096`, `--kprof`. **3 interleaved reps per arm**
(every rep touches every arm, so drift hits all three equally); `median [min, max]`.

### Prefill, all three arms

| prompt | arm | prefill tok/s | attention ms/op | attention %run | attention GMAC/s | %peak |
|---|---|---:|---:|---:|---:|---:|
| ~46 | scalar | 329.9 [273.9, 352.3] | 0.2 [0.2, 0.2] | 4.7% | 8.1 | 6.8% |
| ~46 | in-place | 372.2 [335.7, 418.6] | 0.1 [0.1, 0.1] | 2.0% | 20.0 | 16.2% |
| ~46 | **gather** | 370.2 [353.4, 385.2] | 0.1 [0.1, 0.1] | 1.6% | 25.5 | 19.0% |
| ~900 | scalar | 117.6 [114.3, 119.4] | 54.3 [53.0, 55.3] | 51.0% | 4.1 | 3.6% |
| ~900 | in-place | 166.3 [156.5, 172.6] | 21.7 [21.3, 23.1] | 29.1% | 10.1 | 8.7% |
| ~900 | **gather** | **212.5 [193.5, 214.1]** | 7.9 [7.7, 8.6] | 13.4% | 28.0 | 24.7% |
| ~2040 | scalar | 89.9 [88.6, 92.5] | 293.6 [287.1, 295.6] | 61.9% | 3.4 | 3.0% |
| ~2040 | in-place | 148.0 [146.7, 150.2] | 105.4 [105.1, 106.4] | 37.0% | 9.5 | 8.5% |
| ~2040 | **gather** | **198.1 [197.0, 202.7]** | 35.6 [34.9, 36.6] | 16.9% | 28.3 | 25.3% |

Ratios:

| prompt | in-place / scalar | gather / scalar | gather / in-place |
|---|---:|---:|---:|
| ~46 | 1.13x | 1.12x | **0.99x** |
| ~900 | 1.41x | **1.81x** | **1.28x** |
| ~2040 | 1.65x | **2.20x** | **1.34x** |

**Attention on its own op**, long prompt: 293.6 -> 105.4 -> **35.6 ms/op**,
i.e. **8.25x** against the scalar reference and **2.96x** against #27's kernel.
Its share of the run falls 61.9% -> 37.0% -> **16.9%**, and its rate rises
3.4 -> 9.5 -> **28.3 GMAC/s** (3.0% -> 8.5% -> **25.3%** of the measured f32
peak).

### Decode

Decode is unchanged by design — `n_tok == 1` never enters the gather:

| prompt | scalar | in-place | gather |
|---|---:|---:|---:|
| ~46 | 57.6 tok/s | 60.8 | 60.0 |
| ~900 | 26.3 | 33.7 | 33.8 |

The ~2040 arm has **no decode row**: the model's own context is 2048, so a
2040-token prompt leaves zero timed steps. That is reported as a blank rather
than a zero.

## 4. Accuracy

Ten shapes, `tools/ta_attn_acc.cpp`, scalar reference vs the vector path:

| shape | rel vs scalar | max abs |
|---|---:|---:|
| decode @ pos 1400 | 2.740e-07 | 1.118e-08 |
| prefill 64 | 6.224e-08 | 5.960e-08 |
| prefill 64 @ pos 512 | 2.051e-07 | 2.608e-08 |
| GQA 8/2, hd 128 | 1.752e-07 | 2.980e-08 |
| hd 80 | 1.188e-07 | 3.353e-08 |
| hd 96 | 1.242e-07 | 5.215e-08 |
| hd 256 | 1.633e-07 | 2.980e-08 |
| window 64 | 1.114e-07 | 3.725e-08 |
| skip_from 30 | 6.105e-08 | 5.960e-08 |
| sinks + softcap 50 | 1.240e-07 | 3.725e-08 |
| **global** | **7.719e-08** | — |

33.9% of 239,904 floats bit-identical to the scalar reference — the signature of
a reordered f32 reduction in the score dot, which is the only place the two
differ.

**The gather adds exactly zero error**: in-place vs gather is **100.00%
bit-identical, rel 0.000e+00** on all ten shapes. It is a copy, and the copy is
byte-for-byte the data the other path would have read.

### The fixture earned its keep twice

Two bugs were written and caught in this change, both in the multi-chunk scan
and both invisible to code review:

* **an assigning 64-dim chunk** loses all but the last 64 dimensions —
  `hd = 128` and `hd = 256` went to rel **6.4e-02** and **7.5e-02**;
* **an assigning remainder** loses the *first* 64 — `hd = 80/96` went to 1e-2.

Both were introduced by the same refactor that removed a shared zeroing pass,
and both were caught by the `hd` 80/96/128/256 arms within one run. The fixture
exists because the previous kernel change also shipped a shape bug.

## 5. What this says about the remaining gap

25.3% of peak is the honest number now. The gather removed the stride penalty
from the *scan*; the remaining cost is that attention still reads each key's
K and V slice **once per query row that can see it**. Blocking the query rows —
so one K row feeds several rows of output, the `rN` trick from `gemm_q6k` — is
the next step and is not in this change.

## 6. Reproducing

```bash
# the three arms
./build/kraken -m MODEL --cpu -p "$PROMPT" -n 8 --greedy --ctx 4096 --kprof
KRK_CPU_ATTN_GATHER=0  ./build/kraken ...   # in place, #27's kernel
KRK_CPU_ATTN_SCALAR=1  ./build/kraken ...   # the reference body

# accuracy fixture
g++ -O2 -march=native -Iinclude tools/ta_attn_acc.cpp build/libkrk_core.a -o ta
KRK_CPU_ATTN_SCALAR=1 ./ta scalar.bin
./ta gather.bin
KRK_CPU_ATTN_GATHER=0 ./ta inplace.bin
```

## 7. Caveats

* **Absolute tok/s is indicative.** This box is 2 hardware threads on a shared
  host with no turbo control; the spread column is real and the ratios are the
  result. The same arm moved 89.9 -> 49.0 tok/s on one occasion during this work
  because a build was running on the other thread — which is why the harness
  interleaves arms and reports min/max.
* The fixture has random weights: this is a **speed** document. No accuracy
  claim is made from it; the accuracy table uses the same fixture but compares
  two implementations of the *same* arithmetic, which is a meaningful comparison
  even with random weights.
* The gather costs memory: one head's tile is `span x hd x 2` floats, ~1 MiB at
  2048 keys, allocated once per worker rather than per call.
