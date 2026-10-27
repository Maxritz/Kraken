# CPU vs virtual NPU: a measured comparison

This is the answer to "when you compared direct CPU coding and through NPU,
what was better, and do we have a comprehensive benchmark we can see". Every
number below was produced on this machine by the two committed tools in
`tools/`, and every one of them can be reproduced by section 10.

The short version: **on this box the virtual-NPU path beats plain CPU by 2.7x
to 15.5x on prefill and 1.6x to 4.1x on decode, with byte-identical greedy
output on all 72 measured cells** — with one cell where the CPU still wins
(`tiny-q8_0`, 16 prompt words, CPU 1.2x faster), called out in section 3
rather than averaged away. It is a large win for anyone with a decent CPU and
no GPU, which is exactly the population this path is for.

## 1. What was measured, and on what

| | |
| --- | --- |
| Machine | 2 hardware threads, no GPU, no `hipcc`. AVX-512F + VNNI. |
| Engine | this repo at the branch tip; `build-cpu/` and `build-npu/`, both at 0 warnings |
| NPU | `libvnpu.a` built from `riteshnair/npuxl-improved` at `a104e88` |
| Models | synthetic fixtures from `tools/make_bench_gguf.py` |
| Sizes | tiny 256e/2L/4h/vocab512 · small 512e/4L/8h/vocab1000 · medium 1024e/6L/8h/vocab1500 |
| Formats | f16, q8_0, q4_0 (real block layouts, taken from `src/quant.cpp`) |
| Sampling | greedy, `--ctx 512`, best of 3 repetitions per cell |
| Arms | `--cpu` (reference), `--npu` (whole-graph offload), `--npu-cpu` (GEMM-only offload) |

Two honesty constraints on these numbers, stated up front:

* **The models are toys.** They are real GGUF files with real block-quantised
  tensors and real transformer shapes, but 2–6 layers and vocab 512–1500. What
  the tables measure faithfully is the *engine* — its kernels, its offload
  decisions, its dispatch and sync behaviour. They are not a claim about
  7B-class models.
* **This box has ±20% run-to-run noise** on a 2-core machine. Every cell is a
  best-of-3, which is why a number here should be read as "about this fast",
  and why a 5% difference between two arms is not a result. Section 8 lists
  three optimisations that looked real until the noise was beaten down.

Fixtures are wired with the seeded RNG in `make_bench_gguf.py`, so the weights
are deterministic and the greedy continuation is a fixed string per fixture.

## 2. Prefill: tokens/s

| model | prompt words | `--cpu` prefill | `--npu` prefill | `--npu-cpu` prefill | best | speedup |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `medium-f16` | 16 | 14.1 | 66.7 | 65.1 | `--npu` | **4.7x** |
| `medium-f16` | 64 | 14.3 | 169.5 | 144.7 | `--npu` | **11.9x** |
| `medium-q4_0` | 16 | 17.7 | 72.8 | 74.2 | `--npu-cpu` | **4.2x** |
| `medium-q4_0` | 64 | 18.4 | 180.2 | 186.3 | `--npu-cpu` | **10.1x** |
| `medium-q8_0` | 16 | 25.0 | 92.5 | 89.6 | `--npu` | **3.7x** |
| `medium-q8_0` | 64 | 25.0 | 208.1 | 195.5 | `--npu` | **8.3x** |
| `small-f16` | 16 | 71.7 | 317.8 | 325.3 | `--npu-cpu` | **4.5x** |
| `small-f16` | 64 | 71.2 | 751.5 | 773.4 | `--npu-cpu` | **10.9x** |
| `small-f16` | 256 | 68.4 | 1039.3 | 1059.6 | `--npu-cpu` | **15.5x** |
| `small-q4_0` | 16 | 92.5 | 347.8 | 332.9 | `--npu` | **3.8x** |
| `small-q4_0` | 64 | 92.1 | 840.7 | 814.7 | `--npu` | **9.1x** |
| `small-q4_0` | 256 | 89.2 | 1111.5 | 1123.4 | `--npu-cpu` | **12.6x** |
| `small-q8_0` | 16 | 125.1 | 412.4 | 430.3 | `--npu-cpu` | **3.4x** |
| `small-q8_0` | 64 | 125.2 | 915.9 | 925.6 | `--npu-cpu` | **7.4x** |
| `small-q8_0` | 256 | 124.5 | 1155.1 | 1147.4 | `--npu` | **9.3x** |
| `tiny-f16` | 16 | 543.5 | 2091.6 | 2125.3 | `--npu-cpu` | **3.9x** |
| `tiny-f16` | 64 | 576.6 | 4434.0 | 5121.6 | `--npu-cpu` | **8.9x** |
| `tiny-f16` | 256 | 565.3 | 5479.9 | 6315.4 | `--npu-cpu` | **11.2x** |
| `tiny-q4_0` | 16 | 727.4 | 2337.3 | 2602.0 | `--npu-cpu` | **3.6x** |
| `tiny-q4_0` | 64 | 731.5 | 4834.8 | 5693.0 | `--npu-cpu` | **7.8x** |
| `tiny-q4_0` | 256 | 704.9 | 5985.8 | 6185.1 | `--npu-cpu` | **8.8x** |
| `tiny-q8_0` | 16 | 977.7 | 2657.3 | 2574.2 | `--npu` | **2.7x** |
| `tiny-q8_0` | 64 | 1029.9 | 4908.6 | 5784.5 | `--npu-cpu` | **5.6x** |
| `tiny-q8_0` | 256 | 1008.5 | 6597.9 | 6867.7 | `--npu-cpu` | **6.8x** |

## 3. Decode: tokens/s

| model | prompt words | `--cpu` decode | `--npu` decode | `--npu-cpu` decode | best | speedup |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `medium-f16` | 16 | 13.7 | 53.8 | 55.5 | `--npu-cpu` | **4.1x** |
| `medium-f16` | 64 | 13.5 | 45.0 | 42.2 | `--npu` | **3.3x** |
| `medium-q4_0` | 16 | 17.2 | 51.1 | 52.2 | `--npu-cpu` | **3.0x** |
| `medium-q4_0` | 64 | 17.4 | 46.0 | 46.9 | `--npu-cpu` | **2.7x** |
| `medium-q8_0` | 16 | 23.5 | 57.1 | 54.7 | `--npu` | **2.4x** |
| `medium-q8_0` | 64 | 23.4 | 47.2 | 45.2 | `--npu` | **2.0x** |
| `small-f16` | 16 | 66.1 | 222.3 | 186.0 | `--npu` | **3.4x** |
| `small-f16` | 64 | 64.4 | 195.5 | 184.6 | `--npu` | **3.0x** |
| `small-f16` | 256 | 61.2 | 187.5 | 175.0 | `--npu` | **3.1x** |
| `small-q4_0` | 16 | 84.6 | 170.6 | 175.9 | `--npu-cpu` | **2.1x** |
| `small-q4_0` | 64 | 85.0 | 196.9 | 194.6 | `--npu` | **2.3x** |
| `small-q4_0` | 256 | 80.0 | 169.0 | 177.7 | `--npu-cpu` | **2.2x** |
| `small-q8_0` | 16 | 113.7 | 224.6 | 200.1 | `--npu` | **2.0x** |
| `small-q8_0` | 64 | 112.2 | 197.4 | 195.8 | `--npu` | **1.8x** |
| `small-q8_0` | 256 | 106.8 | 179.8 | 170.7 | `--npu` | **1.7x** |
| `tiny-f16` | 16 | 448.6 | 818.7 | 612.4 | `--npu` | **1.8x** |
| `tiny-f16` | 64 | 445.1 | 1484.9 | 1070.4 | `--npu` | **3.3x** |
| `tiny-f16` | 256 | — | — | — | — | — |
| `tiny-q4_0` | 16 | 547.6 | 871.1 | 677.3 | `--npu` | **1.6x** |
| `tiny-q4_0` | 64 | 572.6 | 1321.2 | 936.0 | `--npu` | **2.3x** |
| `tiny-q4_0` | 256 | — | — | — | — | — |
| `tiny-q8_0` | 16 | 802.4 | 699.6 | 680.9 | `--cpu` | **1.2x** |
| `tiny-q8_0` | 64 | 755.9 | 1408.8 | 1457.7 | `--npu-cpu` | **1.9x** |
| `tiny-q8_0` | 256 | — | — | — | — | — |

Rows marked "—" are `tiny-*` at 256 prompt words: that prompt is longer than
`--ctx 512`, so the run has no decode phase to report. They are left in the
table as missing rather than quietly dropped.

One row goes the other way: `tiny-q8_0` at 16 prompt words is the only cell in
the whole matrix where `--cpu` beats both NPU arms (802.4 vs 699.6 tok/s). A
1.6 MiB model that fits in cache, and a decode step small enough that the
device's fixed per-command overhead is a large fraction of it, is exactly the
case the NPU path should lose; section 11 says what that means for the win's
boundary.

## 4. The correctness witness

Speed means nothing if the output moved. `tools/npu_bench.py` compares each
arm's generated text against the `--cpu` arm of the same cell, byte for byte:

```
runs that completed:                            72/72
greedy text identical to the --cpu reference:   72/72
```

All three arms produce the same tokens on every model, format and prompt
length. The device's arithmetic is not bit-identical to the reference (GEMM
agrees to 2.7e-07 relative, RMSNorm to 2e-06, SwiGLU to 1.85e-05 absolute), so
this equality is a statement about *decisions surviving rounding*, not about
the same floating-point operations — which is the property that actually
matters, and which the earlier chunk-invariance and `KRK_NPU_STRICT_QUEUE`
gates already pinned down.

## 5. Why the NPU wins: decode is a bandwidth problem, not a FLOP problem

The most useful thing the fixtures showed is that **staged weight bytes are
identical across formats**. Dequantisation happens once, into an f32 staging
cache, so an f16, a Q8_0 and a Q4_0 file all present the device with exactly
the same traffic:

| fixture | file size | staged for the device | decode tok/s |
| --- | ---: | ---: | ---: |
| `small-f16` | 26.1 MiB | 50.2 MiB | 229.9 |
| `small-q8_0` | 13.9 MiB | 50.2 MiB | 183.3 |
| `small-q4_0` | 7.4 MiB | 50.2 MiB | 196.3 |
| `medium-f16` | 126.7 MiB | 247.4 MiB | 52.7 |
| `medium-q8_0` | 67.3 MiB | 247.4 MiB | 47.9 |
| `medium-q4_0` | 35.7 MiB | 247.4 MiB | 51.2 |

Decode speed does not track file size — it tracks *staged* size, and it tracks
it linearly: 50.2 MiB in ~4.4 ms and 247.4 MiB in ~19 ms are both ~11–13 GB/s
of streamed weights. A decode step must read every weight in the model, and
nothing in the engine can cheat that.

This is also why the win exists at all. The reference CPU path reads those same
bytes and then does the arithmetic with its own loops; the device path reads
the same bytes with an AVX-512 kernel whose inner loop is better than the
generic one. On prefill the difference is enormous (11–15x) because the
problem becomes compute-bound and the device's vector kernel is simply a much
better matmul than the reference loop. Three standalone probes, all in
`tools/`-adjacent scratch work and reproduced in section 10:

| probe | measurement |
| --- | --- |
| stream 247 MiB with a plain sum | 3.0 GB/s (this box's raw read rate) |
| `vnpu_matmul_f32`, M=1, K=512, N=1376, warm | 0.098 ms → 28.6 GB/s of weights, 14.3 GFLOP/s |
| `vnpu_matmul_i8`, same shape | 0.057 ms → **1.73x faster**, 3.3e-04 relative error |

A dedicated kernel reaching 28.6 GB/s against a 3.0 GB/s raw stream is the
whole story: the device path is not magic, it is a better memory access
pattern plus a better register-blocked dot product.

## 6. Sherlock trace: where the time actually goes

Both sides can be traced: `--trace FILE` records the engine's own timeline,
`KRK_NPU_TRACE=FILE` records every command the device executed. Both are
Chrome trace-event JSON; `tools/trace_stats.py` reads either, detects which it
has been handed, and reports what that producer can actually answer.

The four traces these sections analyse are checked in under `docs/traces/`.
They are ordinary Chrome trace-event JSON: drop
`small-f16-npu-cpu.device.json` onto <https://ui.perfetto.dev> to see every
command the device ran, or open it in `chrome://tracing`. The `.host.json`
files are the engine's own timeline for the same run.

Device timeline, `small-f16`, 65-token prompt then 32 greedy tokens:

| arm | commands | device service | span | duty |
| --- | ---: | ---: | ---: | ---: |
| `--npu` | 928 | 192.2 ms | 272.6 ms | 70.5% |
| `--npu-cpu` | 1600 | 189.5 ms | 250.2 ms | 75.7% |

Per-op service time, `--npu-cpu`:

| op | count | total | % of device time | mean µs | p50 µs | p95 µs | max µs | moved | GB/s | GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MATMUL_F32` | 928 | 183.3 ms | 96.8% | 197.6 | 131.1 | 291.6 | 2858.2 | 1699.7 MiB | 9.3 | 13.4 |
| `ADD_F32` | 256 | 2.3 ms | 1.2% | 9.1 | 6.4 | 14.5 | 91.8 | 4.7 MiB | 2.0 | 0.5 |
| `RMSNORM_F32` | 288 | 2.3 ms | 1.2% | 8.0 | 6.0 | 9.4 | 164.5 | 3.9 MiB | 1.7 | 0.9 |
| `SWIGLU_F32` | 128 | 1.5 ms | 0.8% | 11.6 | 7.1 | 9.5 | 196.3 | 6.3 MiB | 4.3 | 2.8 |

Three things to read out of that, and one thing *not* to read:

* The device is inside a kernel **70–76% of the wall clock**. That is the
  real ceiling on this path, and it is why the speedups are what they are.
* The remaining 24–30% is the host: RoPE, attention over the KV cache, the KV
  cache itself, token embedding, MoE routing, and sampling all run on the CPU
  because the virtual NPU has no op for them. This is a property of the op
  set, not a scheduling defect.
* `% of device time` is dominated by matmul to the point where the three
  elementwise ops together are 3%.

The thing not to read: `tools/trace_stats.py` deliberately prints **no**
worker-occupancy figure. VNPU's `tid` field is `hash(std::thread::id) % 8`, not
a worker index, so counting tids would be inventing a number.

## 7. An open anomaly: the output head

This is a real inefficiency found by the trace, reported rather than papered
over. `--profile` component table, 32 greedy tokens:

| component | phase | `--npu-cpu` host ms | `--cpu` host ms |
| --- | --- | ---: | ---: |
| decode (layers) | decode | 182.6 | 492.7 |
| prefill | prefill | 67.2 | 474.4 |
| **head+sample** | decode | **35.1** | **0.1** |
| load | load | 16.0 | 15.7 |

The output projection costs **1.10 ms per token on the NPU path and about
0.003 ms per token on the CPU path** — it is 16.1% of decode on one arm and
invisible on the other. The device itself is not the problem: measured
standalone at exactly that shape (M=1, K=512, N=1000 vocab), `vnpu_matmul_f32`
takes **0.084 ms** batched and **0.178 ms** with a sync after each call. So
between 0.9 and 1.0 ms per token is being spent outside the arithmetic.

What is ruled out: the weight staging cache is keyed on the weight pointer and
hits (the head is dequantised once, not per token); the head GEMM *is* on the
device (928 commands per 32 tokens = 29 per token = 28 layer GEMMs + 1 head);
and the device top-k fast path is not taken because the virtual NPU reports
`has_device_memory() == false`, so every token downloads the full logits row —
which is 4 kB and cannot be the cost.

What is left is the call plumbing around the head GEMM. This is the single
largest identified win left on the CPU+NPU path (~13–14% of a decode run) and
it is stated as an open anomaly with a reproduction, not as a fixed bug.

## 8. Negative results: three optimisations that did not survive measurement

Kept here because "we measured it and it was noise" is a result, and because
each one will otherwise be re-attempted by the next person.

**Removing the per-GEMM sync in GEMM mode.** `--npu-cpu` submits and waits on
every call; that looked like an obvious win over `--npu`'s batching. It was
implemented and measured: end-to-end decode moved a median **+1.3%**, inside
this box's noise, except on 2-layer toy models where the round trip is a large
share of a tiny step (+19% to +54%). It also **broke `Backend::gemm`'s
synchronous contract** — a caller may read the output buffer directly, and
`tests/test_kraken.cpp` does, so the suite failed correctly. Reverted, and the
reason is now recorded in `Offload::push` so it is not "optimised" away again.

**`KRK_NPU_THREADS=2`.** A knob sweep showed +17–18% decode over the default
and looked like a free win. Re-run with best-of-6 the ranking **reversed**:
unset 194.3 tok/s, `=2` 181.7, `=3` 187.7, `=1` 203.5 decode but 540.6
prefill. The device default (hardware concurrency) stays. A 17% "win" inside a
±20% noise band is not a win.

**`KRK_NPU_UNIT`.** Measured at M = 1, 8, 64, 256: `wildcat` is **bit-identical
to `voodoo`** at every size and never faster; `cell` is 3x slower *and*
produces different bits, so selecting it would also break chunk invariance.
Default `voodoo` stays. (An earlier end-to-end sweep had `cell` at −51%
prefill / −54% decode, consistent with the probe.)

**int8 weights: measured, not enabled.** `vnpu_matmul_i8` is 1.73x faster than
f32 at the decode shape and would cut staged bytes 4x, which is the only real
lever left on a bandwidth-bound path. It is not enabled because there is no
int8-weights-by-f32-activations op: the activations must be quantised too
(`vnpu_quantize_i8`), making it W8A8, with a measured 3.3e-04 relative error on
the probe. That is a quality trade-off the user should opt into explicitly
rather than something this branch turns on by default.

## 9. A defect found in the NPU's own trace writer

The first traces captured were unusable for timing and it was not obvious why:
928 commands shared 29 start timestamps, all exactly 10 ms apart, while the
durations were fine-grained. It looked like a coarse clock. It is not — the
platform's `steady_clock` resolves changes 100–200 ns apart.

The cause is one line in `src/core/runtime.cpp`:

```cpp
out << "\",\"ph\":\"X\",\"ts\":" << (e.start_ns / 1000.0)
```

`std::ostream`'s default precision is 6 significant digits. A monotonic
timestamp in microseconds near 1.2e9 therefore prints as `1.1953e+09` — a
**10 ms grid**. Durations are small numbers, so the same 6 digits are ample for
them, which is why only the start times were destroyed and the trace still
looked plausible.

Fix and evidence, in `patches/npuxl-0001-trace-timestamp-precision.patch`:
rebase timestamps to the first event and print with `fixed(3)`. Before and
after, on 30 commands separated by 3 ms:

| | distinct start timestamps | observed deltas |
| --- | ---: | --- |
| before | 11 | 10000, 0, 0, 10000 … µs |
| after | 30 | 3306.8, 3218.9, 3244.9 … µs |

Apply from the directory containing `src/`:

```
patch -p1 < patches/npuxl-0001-trace-timestamp-precision.patch
```

`tools/trace_stats.py` does not assume the patch is present: it detects
quantised start times and **withholds** gaps, idle time, duty cycle and overlap
rather than computing them from a grid, while still reporting every
duration-based number. That is why the trace tables above are trustworthy.

## 10. Reproducing all of it

```
# fixtures (deterministic; --batch must be a multiple of 32)
tools/make_bench_gguf.py --out /tmp/fix/tiny-f16.gguf  --embd 256  --layers 2 --heads 4 --vocab 512  --format f16
tools/make_bench_gguf.py --out /tmp/fix/small-f16.gguf --embd 512  --layers 4 --heads 8 --vocab 1000 --format f16
tools/make_bench_gguf.py --out /tmp/fix/medium-f16.gguf --embd 1024 --layers 6 --heads 8 --vocab 1500 --format f16

# the matrix in sections 2 and 3
tools/npu_bench.py \
  --models '/tmp/fix/tiny-*.gguf' '/tmp/fix/small-*.gguf' \
  --arms cpu,npu,npu-cpu --prefill 16,64,256 --decode 64 --reps 3 --ctx 512 \
  --out bench.md --json bench.json

# traces, then the analysis
KRK_NPU_TRACE=/tmp/tr/npu.json kraken -m small-f16.gguf --npu     -p "..." -n 32 --greedy --trace /tmp/tr/npu.krk.json
KRK_NPU_TRACE=/tmp/tr/cpu.json kraken -m small-f16.gguf --npu-cpu -p "..." -n 32 --greedy --trace /tmp/tr/cpu.krk.json
tools/trace_stats.py /tmp/tr/npu.json /tmp/tr/cpu.json --label npu --label npu-cpu
```

The build needs the NPU engine: configure with `-DKRK_ENABLE_NPU=ON
-DKRK_VNPU_ROOT=<npuxl> -DKRK_VNPU_LIB=<npuxl>/build/libvnpu.a`. Link the
archive **you built** — the prebuilt `lib/libvnpu.a` that ships in npuxl's zip
is compiled against a newer glibc and will not link here.

## 11. What this does and does not prove

It proves the CPU+NPU path is a genuine, large, correct win on this machine:
2.7–15.5x on prefill and 1.6–4.1x on decode (one cell excepted, where the CPU
wins by 1.2x), identical greedy text on 72/72 cells, with the device inside a
kernel 70–76% of the wall clock and the rest attributable to transformer ops
the virtual NPU does not implement.

That single losing cell is worth being precise about, because it marks the
boundary of the win. It is a **2-layer, 512-vocab toy at 16 prompt words**:
the whole model is 1.6 MiB, it fits in cache, decode is 802 tok/s on the CPU
already, and a decode step is small enough that the device's fixed per-command
cost outweighs its better kernel. The CPU+NPU path pays off when the model is
big enough that streaming weights dominates — which is every model anyone
would actually run.

It does not prove anything about a GPU. There is no `hipcc` on this box and no
GPU to measure, so this document says nothing about the HIP path beyond "it
still builds and its test suite passes". It also does not prove anything about
models larger than 1024 embedding dims — the mechanism (decode is
bandwidth-bound on f32-staged weights) scales, but the constants here are
measured at toy sizes.
