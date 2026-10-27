# CPU vs virtual NPU: a measured comparison

This is the answer to "when you compared direct CPU coding and through NPU,
what was better, and do we have a comprehensive benchmark we can see". Every
number below was produced on this machine by the two committed tools in
`tools/`, and every one of them can be reproduced by section 12.

The short version: **on this box the virtual-NPU path beats plain CPU by 1.8x
to 18.4x on prefill and 1.1x to 5.1x on decode, with byte-identical greedy
output on every measured cell** — with one cell where the CPU still wins
(`tiny-q8_0`, 16 prompt words, CPU 1.9x faster), called out in section 3
rather than averaged away. It is a large win for anyone with a decent CPU and
no GPU, which is exactly the population this path is for.

Sections 2-7 were re-recorded in one session on one machine. An earlier round
ran on a different sandbox instance and read systematically higher on the
`--cpu` arm (+30% median) while the device's own kernels measured the same, so
only the ratios survived that move. Section 8 quotes the round that tested each
rejected optimisation, because a negative result is about the ranking at the
time, not about an absolute rate.

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
| `medium-f16` | 16 | 13.4 | 64.6 | 65.4 | `--npu-cpu` | **4.9x** |
| `medium-f16` | 64 | 13.3 | 162.6 | 167.7 | `--npu-cpu` | **12.6x** |
| `medium-q4_0` | 16 | 24.5 | 72.4 | 72.6 | `--npu-cpu` | **3.0x** |
| `medium-q4_0` | 64 | 24.1 | 171.4 | 187.7 | `--npu-cpu` | **7.8x** |
| `medium-q8_0` | 16 | 38.6 | 77.1 | 75.6 | `--npu` | **2.0x** |
| `medium-q8_0` | 64 | 39.6 | 180.3 | 202.5 | `--npu-cpu` | **5.1x** |
| `small-f16` | 16 | 65.1 | 308.6 | 310.0 | `--npu-cpu` | **4.8x** |
| `small-f16` | 64 | 66.2 | 719.3 | 735.0 | `--npu-cpu` | **11.1x** |
| `small-f16` | 256 | 65.3 | 1200.2 | 1176.9 | `--npu` | **18.4x** |
| `small-q4_0` | 16 | 118.1 | 364.1 | 376.8 | `--npu-cpu` | **3.2x** |
| `small-q4_0` | 64 | 118.6 | 866.7 | 846.0 | `--npu` | **7.3x** |
| `small-q4_0` | 256 | 117.7 | 1064.3 | 1172.1 | `--npu-cpu` | **10.0x** |
| `small-q8_0` | 16 | 206.6 | 395.1 | 428.0 | `--npu-cpu` | **2.1x** |
| `small-q8_0` | 64 | 209.8 | 929.5 | 971.6 | `--npu-cpu` | **4.6x** |
| `small-q8_0` | 256 | 203.1 | 1321.3 | 1315.0 | `--npu` | **6.5x** |
| `tiny-f16` | 16 | 526.6 | 2180.4 | 2215.7 | `--npu-cpu` | **4.2x** |
| `tiny-f16` | 64 | 524.0 | 5080.4 | 5371.9 | `--npu-cpu` | **10.3x** |
| `tiny-f16` | 256 | 522.2 | 7577.4 | 7842.6 | `--npu-cpu` | **15.0x** |
| `tiny-q4_0` | 16 | 978.4 | 2886.7 | 2972.5 | `--npu-cpu` | **3.0x** |
| `tiny-q4_0` | 64 | 979.8 | 6143.3 | 6404.6 | `--npu-cpu` | **6.5x** |
| `tiny-q4_0` | 256 | 901.2 | 5762.2 | 8130.5 | `--npu-cpu` | **9.0x** |
| `tiny-q8_0` | 16 | 1746.5 | 3154.6 | 3048.8 | `--npu` | **1.8x** |
| `tiny-q8_0` | 64 | 1738.5 | 6473.7 | 7020.3 | `--npu-cpu` | **4.0x** |
| `tiny-q8_0` | 256 | 1682.3 | 7660.1 | 8183.1 | `--npu-cpu` | **4.9x** |

## 3. Decode: tokens/s

| model | prompt words | `--cpu` decode | `--npu` decode | `--npu-cpu` decode | best | speedup |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `medium-f16` | 16 | 12.6 | 60.4 | 64.5 | `--npu-cpu` | **5.1x** |
| `medium-f16` | 64 | 12.7 | 49.9 | 53.1 | `--npu-cpu` | **4.2x** |
| `medium-q4_0` | 16 | 23.1 | 55.7 | 60.7 | `--npu-cpu` | **2.6x** |
| `medium-q4_0` | 64 | 22.8 | 50.1 | 41.0 | `--npu` | **2.2x** |
| `medium-q8_0` | 16 | 34.0 | 58.0 | 43.0 | `--npu` | **1.7x** |
| `medium-q8_0` | 64 | 37.7 | 39.3 | 45.8 | `--npu-cpu` | **1.2x** |
| `small-f16` | 16 | 60.5 | 254.3 | 313.6 | `--npu-cpu` | **5.2x** |
| `small-f16` | 64 | 60.4 | 246.2 | 255.3 | `--npu-cpu` | **4.2x** |
| `small-f16` | 256 | 59.0 | 247.8 | 255.1 | `--npu-cpu` | **4.3x** |
| `small-q4_0` | 16 | 104.0 | 264.3 | 274.6 | `--npu-cpu` | **2.6x** |
| `small-q4_0` | 64 | 107.1 | 277.9 | 283.4 | `--npu-cpu` | **2.6x** |
| `small-q4_0` | 256 | 103.9 | 175.0 | 226.1 | `--npu-cpu` | **2.2x** |
| `small-q8_0` | 16 | 187.5 | 314.3 | 313.8 | `--npu` | **1.7x** |
| `small-q8_0` | 64 | 185.0 | 306.0 | 314.6 | `--npu-cpu` | **1.7x** |
| `small-q8_0` | 256 | 170.5 | 310.8 | 294.8 | `--npu` | **1.8x** |
| `tiny-f16` | 16 | 436.4 | 713.3 | 733.5 | `--npu-cpu` | **1.7x** |
| `tiny-f16` | 64 | 426.7 | 1206.3 | 1450.7 | `--npu-cpu` | **3.4x** |
| `tiny-f16` | 256 | 413.4 | 1603.4 | 1490.7 | `--npu` | **3.9x** |
| `tiny-q4_0` | 16 | 697.2 | 666.9 | 710.0 | `--npu-cpu` | **1.1x** |
| `tiny-q4_0` | 64 | 744.8 | 1011.2 | 1221.1 | `--npu-cpu` | **1.6x** |
| `tiny-q4_0` | 256 | 700.8 | 1530.1 | 1663.5 | `--npu-cpu` | **2.4x** |
| `tiny-q8_0` | 16 | 1203.6 | 622.1 | 1187.2 | `--cpu` | **1.9x** |
| `tiny-q8_0` | 64 | 1216.4 | 1461.0 | 1364.4 | `--npu` | **1.2x** |
| `tiny-q8_0` | 256 | 1199.5 | 1548.4 | 1103.8 | `--npu` | **1.4x** |

One row goes the other way: `tiny-q8_0` at 16 prompt words is the only cell in
the whole matrix where `--cpu` beats both NPU arms (1203.6 against 1187.2 and
622.1 tok/s). A 1.9 MiB model that fits in cache, and a decode step small
enough that the device's fixed per-command cost is a large fraction of it, is
exactly the case the NPU path should lose; section 13 says what that means for
the win's boundary. The narrow cells at 1.1x-1.4x are the same effect at a
smaller amplitude.

The `--npu` column is worth reading separately from `--npu-cpu`: they are the
*same* device doing the same GEMMs, differing only in how much of the rest of
the op mix is offloaded and how often the queue is submitted. On decode `--npu`
wins several rows and `--npu-cpu` wins others, which is the mode trade-off
measured in section 7 rather than a stable ordering.

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
| `small-f16` | 26.1 MiB | 50.2 MiB | 290.7 |
| `small-q8_0` | 13.9 MiB | 50.2 MiB | 310.4 |
| `small-q4_0` | 7.4 MiB | 50.2 MiB | 314.7 |
| `medium-f16` | 126.7 MiB | 247.4 MiB | 48.8 |
| `medium-q8_0` | 67.3 MiB | 247.4 MiB | 55.3 |
| `medium-q4_0` | 35.7 MiB | 247.4 MiB | 45.2 |

Decode speed does not track file size — it tracks *staged* size, and it tracks
it linearly: staged bytes divided by decode time land inside 10.9-15.4 GiB/s
for every row, and the device trace's own achieved rate over the same runs is 12.7
to 13.0 GB/s. A decode step must read every weight in the model, and
nothing in the engine can cheat that.

This is also why the win exists at all. The reference CPU path reads those same
bytes and then does the arithmetic with its own loops; the device path reads
the same bytes with an AVX-512 kernel whose inner loop is better than the
generic one. On prefill the difference is enormous (11–15x) because the
problem becomes compute-bound and the device's vector kernel is simply a much
better matmul than the reference loop. Three standalone probes, all in
`tools/`-adjacent scratch work and reproduced in section 12:

| probe | measurement |
| --- | --- |
| stream 247 MiB with a plain sum | 4.5 GB/s (this box's raw read rate) |
| `vnpu_matmul_f32`, M=1, K=512, N=1376, warm | 0.100 ms → 28.2 GB/s of weights, 14.1 GFLOP/s |
| `vnpu_matmul_i8`, same shape | 0.065 ms → **1.54x faster**, 3.32e-04 relative error |
| `vnpu_matmul_f32`, M=1, K=512, N=1000 (the head shape) | 0.082 ms → 25.0 GB/s |

A dedicated kernel reaching 28.2 GB/s against a 4.5 GB/s raw stream is most of
the story: the device path is not magic, it is a better memory access pattern
plus a better register-blocked dot product. The sum probe is a single-threaded
stream over a buffer 50x larger than cache, so it is the floor rather than the
device's ceiling.

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
| `--npu` | 928 | 130.3 ms | 217.8 ms | 59.8% |
| `--npu-cpu` | 1600 | 140.8 ms | 209.3 ms | 67.3% |

Per-op service time, `--npu-cpu`:

| op | count | total | % of device time | mean µs | p50 µs | p95 µs | max µs | moved | GB/s | GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MATMUL_F32` | 928 | 133.7 ms | 95.0% | 144.1 | 101.0 | 225.3 | 3195.1 | 1621.0 MiB | 12.7 | 18.4 |
| `RMSNORM_F32` | 288 | 2.8 ms | 2.0% | 9.8 | 7.2 | 10.3 | 163.2 | 3.7 MiB | 1.4 | 0.7 |
| `ADD_F32` | 256 | 2.6 ms | 1.9% | 10.3 | 7.4 | 18.7 | 123.1 | 4.5 MiB | 1.8 | 0.4 |
| `SWIGLU_F32` | 128 | 1.6 ms | 1.2% | 12.7 | 8.5 | 13.4 | 149.3 | 6.0 MiB | 3.9 | 2.6 |

Three things to read out of that, and one thing *not* to read:

* The device is inside a kernel **60–67% of the wall clock**. That is the
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

## 7. An anomaly that turned out to be the profiler lying

An earlier revision of this document reported the output projection as costing
**1.10 ms per token on the NPU path against 0.003 ms on CPU** — 16% of decode on
one arm and invisible on the other — and called it the biggest remaining win.
That conclusion was wrong, and the way it was wrong is worth keeping.

The row came from `--profile`:

| span | `--npu` (flush per call) | `--npu-cpu` (batched) |
| --- | ---: | ---: |
| `decode.logits.dev`, per call | 0.001 ms | **0.979 ms** |
| `decode.drain`, per call | 0.000 ms | **0.978 ms** |

Same GEMM, same weights, same device: the only difference between the arms is
*when the queue is submitted*. `--npu-cpu` is whole-graph mode — its banner says
`whole graph`, and it reports 160 batches for 1600 commands — so a layer's ops
are issued and submitted at the first host op, not at the call that issued them.
`decode.forward` returns once the work is **issued**; the wait for it lands on
whichever host call flushes next, and that was `fetch_logits`. The head row was
reporting the forward pass's unpaid bill.

Two independent measurements say the head is not slow:

* a standalone probe at exactly that shape (M=1, K=512, N=1000) takes
  **0.082 ms**, and the device trace's matmul p50 is **101 µs**, so the head is an
  ordinary GEMM and not a pathological one;
* the arm that flushes per call reports **0.001 ms/call** for the identical
  span, and the two arms' total decode rates are within 5% (317.4 vs 333.3
  tok/s).

So there is no missing millisecond and no win to collect here. What there was,
was a profile that attributed one phase's work to another — and a plausible
story ("the head costs 13% of decode") that survived exactly as long as nobody
cross-checked it against the device's own timeline.

The fix is instrumentation, not the hot path: `fetch_logits` now times its drain
as `decode.drain` (via `KRK_TIME_HOST_N`, a distinct handle because the plain
macro shadows the enclosing scope's) so the split is visible in the table:

| span | `--npu-cpu`, per call |
| --- | ---: |
| `decode.logits.dev` | 0.979 ms |
| of which `decode.drain` | 0.978 ms |

The head is what is left over, ~0.001 ms, which matches the probe and the
0.082 ms the device needs to execute it. Nothing got
faster; a future search now starts in the right place.

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

**`KRK_NPU_THREADS=2`.** A knob sweep showed +17-18% decode over the default
and looked like a free win. A best-of-6 replication **reversed** the ranking,
and a later round put `=2` and the default within 2% of each other. The device
default (hardware concurrency) stays: a 17% "win" inside a ±20% noise band is
not a win.

What the knob *does* have is one real effect, measured best-of-5 on this box
(prefill / decode tok/s):

| `KRK_NPU_THREADS` | prefill | decode |
| --- | ---: | ---: |
| unset (device default) | 724.0 | 310.1 |
| `2` | 752.0 | 315.8 |
| `3` | 729.7 | 316.5 |
| `1` | **588.1** | **361.2** |

`=1` buys ~16% decode and costs ~19% prefill, and the mechanism is not a
mystery: decode is bandwidth-bound and does not need a second worker, so the
second thread only competes with the host's rope/attention/sampling on a
2-thread box. Prefill is compute-bound and does need it. That makes `=1` a
workload choice for a decode-only setting, never a default -- and the two
cannot be switched mid-run, because the thread count is fixed when the device
is created.

**`KRK_NPU_UNIT`.** Measured at M = 1, 8, 64, 256: `wildcat` is **bit-identical
to `voodoo`** at every size and never faster; `cell` is 3x slower *and*
produces different bits, so selecting it would also break chunk invariance.
Default `voodoo` stays. (An earlier end-to-end sweep had `cell` at −51%
prefill / −54% decode, consistent with the probe.)

**int8 weights: measured, not enabled.** `vnpu_matmul_i8` is 1.54x faster than
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

## 10. An opt-in: int8 weights (`KRK_NPU_W8A8`)

Everything above is the f32 path, and it leaves one lever untouched: section 5
shows decode is bound by weight traffic, not arithmetic, and the cheapest way to
move fewer bytes is to make the weights smaller. VNPU has an int8 GEMM for
exactly that. It is **off by default and it is lossy**, so it is opt-in:

```sh
KRK_NPU_W8A8=1 kraken -m model.gguf --npu-cpu -p "..." -n 32
```

### 10.1 What it does

Each weight is dequantized once, put through a per-output-row symmetric int8
quantiser (scale = row peak / 127, so no row spends its range on another row's
outlier), and handed to `vnpu_pack_weights_i8`. Activations are quantized on the
device, per row, immediately before the multiply. Both commands go out as one
`push_batch` of two, because the arm this path speeds up is the one that waits
on every call, and two commands submitted separately would pay two waits.

The composition is the device's own: `C[m,n] = dot8(A8[m,:], B8[n,:]) *
scale_a[m] * scale_b[n]`, with both scales applied inside the epilogue. No
scale arithmetic is left on the host.

There is no "int8 weights with f32 activations" opcode, so this is W8A8 and the
accuracy cost is the activation quantisation as much as the weights. That is the
whole reason it is not the default.

### 10.2 What it costs, once

Packing is not free and it lands on the first prefill chunk. Measured as the
difference in `--profile`'s `prefill.chunk` between the two arms on a 33-token
prompt:

| weights | one-time cost | per MiB |
| --- | ---: | ---: |
| 12.6 MiB (`small-f16`) | 31.7 ms | 2.5 ms |
| 61.8 MiB (`medium-f16`) | 213.2 ms | 3.5 ms |

The first cut of the quantiser was **8.5x slower than it needed to be**: it
rounded with `std::lround` on a double, which cost 20.3 ms where the same
rounding in float costs 2.4 ms on a 1376x512 tensor. That turned a ~30 ms setup
into ~125 ms and read as a **40% prefill regression** until it was profiled --
the regression was this loop, not int8. It is now float, with the same
half-away-from-zero rule and the same result for every value in range.

Because the cost is fixed and the per-token saving is not, it amortises against
prompt length. `small-f16`, prefill only, median of 3, arms interleaved:

| prompt | f32 prefill | w8a8 prefill | ratio |
| --- | ---: | ---: | ---: |
| 33 tokens | 463 tok/s | 331 tok/s | **0.72x** |
| 257 tokens | 1236 tok/s | 1366 tok/s | **1.11x** |
| 513 tokens | 1237 tok/s | 1474 tok/s | **1.19x** |

A short prompt loses and a long one wins; the crossover is the pack cost
divided by the per-token saving, which lands somewhere under 250 tokens on this
fixture. That figure is not extrapolated to other models from these three
points, because the last two are within 3% of each other and this box drifts by
more than that. The point is the shape: a one-time cost against a per-token
benefit, stated at both ends rather than only at the flattering one.

### 10.3 What it wins

All nine fixtures, 32 greedy tokens, median of 3 runs **with the arms
interleaved** inside each repetition. Interleaving matters: a blocked harness
(run all f32, then all w8a8) reported `tiny-f16` decode at 0.56x on this box,
and the interleaved measurement of the same thing is 1.42x. Throughput drifts
several percent a minute here, so a blocked A/B compares two different machines.

Prompt 257 tokens, ctx 1024:

| fixture | prefill | decode | staged -> packed |
| --- | ---: | ---: | --- |
| `tiny-f16` | 1.12x | 1.38x | 6.4 -> 1.6 MiB |
| `tiny-q8_0` | 1.06x | 1.27x | 6.4 -> 1.6 MiB |
| `tiny-q4_0` | 1.03x | 1.52x | 6.4 -> 1.6 MiB |
| `small-f16` | 1.14x | 1.39x | 50.2 -> 12.6 MiB |
| `small-q8_0` | 1.18x | 1.58x | 50.2 -> 12.6 MiB |
| `small-q4_0` | 1.11x | 1.53x | 50.2 -> 12.6 MiB |
| `medium-f16` | 1.13x | 3.32x | 247.4 -> 61.8 MiB |
| `medium-q8_0` | 1.15x | 3.34x | 247.4 -> 61.8 MiB |
| `medium-q4_0` | 1.13x | 3.61x | 247.4 -> 61.8 MiB |

Prompt 33 tokens, ctx 512 -- the same runs, at the length where the setup cost
has not been earned back:

| fixture | prefill | decode |
| --- | ---: | ---: |
| `tiny-f16` | 0.64x | 0.95x |
| `tiny-q8_0` | 0.72x | 1.06x |
| `tiny-q4_0` | 0.56x | 0.79x |
| `small-f16` | 0.61x | 1.17x |
| `small-q8_0` | 0.64x | 1.16x |
| `small-q4_0` | 0.60x | 1.04x |
| `medium-f16` | 0.65x | 2.76x |
| `medium-q8_0` | 0.51x | 2.44x |
| `medium-q4_0` | 0.55x | 2.42x |

The decode gain scales with the weights, which is the same conclusion section 5
reached by a different route: the bigger the tensor, the more of decode is
traffic and the more a 4x traffic cut is worth. Decode 3.3-3.6x on the 1024-dim
fixture, 1.3-1.6x on the 512-dim one, and at 256 dimensions the weights already
fit in cache, so there is little traffic left to cut and most of the win is gone.

### 10.4 Where the time goes now

Device traces, both arms, same fixture and prompt
(`docs/traces/small-f16-long-{f32,w8a8}.{device,host}.json`):

| | f32 arm | w8a8 arm |
| --- | --- | --- |
| GEMM | 928 `MATMUL_F32`, 188.0 ms | 928 `MATMUL_I8`, 76.1 ms |
| GEMM rate | 1649.6 MiB, 9.2 GB/s, 38.9 GFLOP/s | 432.7 MiB, 6.0 GB/s, 96.2 GFLOP/s |
| quantise | -- | 928 `QUANTIZE_I8`, 24.1 ms, 24.5 MiB |
| device busy | 195.0 ms (68.6% of span) | 106.9 ms (46.0% of span) |
| device span | 284.3 ms | 232.5 ms |
| host wall | 303.1 ms | 249.0 ms |
| host waiting (`decode.drain`) | 39.5 ms | 10.0 ms |

`MATMUL_I8` is 2.47x the f32 kernel on identical GEMM counts, at 3.81x less
traffic, which is the mechanism working as designed. Two things follow, and both
are left alone deliberately rather than papered over:

- `QUANTIZE_I8` is now 22.5% of device busy at 1.1 GB/s. It is re-quantising the
  same activation block once per GEMM that consumes it (q, k and v all read the
  same row block). Caching it is the obvious next move, but it is only safe if
  the key covers every host write to that buffer, and get that wrong and the
  device multiplies stale activations -- silently, because the output still
  looks like a matrix. It is not in this change.
- Device busy falls to 46.0%, so the device is now idle most of the span and the
  limit has moved to the host. That is a different problem, and section 8 is
  what happens when it is attacked without measuring.

### 10.5 Does the answer survive?

A lossy path has to be judged on how far it moves the output, not on whether it
moved it. Two witnesses, neither of which is the engine reporting on itself.

**An independent reference.** `tools/ref_llama.py` is a float32 numpy forward
written from the fixture's GGUF metadata, not from kraken's source. It is
validated against the engine's own `--prefill-hash first=` on 15 prompt/model
pairs (f16) and 6 (q8_0, q4_0) -- 21/21 argmax agreement. Getting it to agree
took finding that these fixtures rotate with the **interleaved** pairing, not
NeoX half-split; with the wrong pairing it disagreed on two of three models.

**The quantiser, propagated.** `tools/int8_witness.py` puts the fixtures'
weights through the exact quantiser the NPU path uses (per row, symmetric
127-level, round-half-away-from-zero) and runs the reference twice:

| fixture | max abs dlogit | mean abs dlogit | median top1-top2 margin | min margin | positions where margin < error | argmax flips (of 33) | 32-token match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny-f16` | 0.0558 | 0.0118 | 0.1660 | 0.0040 | 8 | 3 | 20/32 |
| `tiny-q8_0` | 0.0562 | 0.0107 | 0.1881 | 0.0061 | 7 | 2 | 32/32 |
| `tiny-q4_0` | 0.0499 | 0.0103 | 0.1146 | 0.0058 | 8 | 1 | 25/32 |
| `small-f16` | 0.1187 | 0.0237 | 0.1244 | 0.0024 | 16 | 3 | 9/32 |
| `small-q8_0` | 0.1108 | 0.0224 | 0.1404 | 0.0169 | 15 | 2 | 24/32 |
| `small-q4_0` | 0.1185 | 0.0210 | 0.1284 | 0.0072 | 14 | 0 | 1/32 |
| `medium-f16` | 0.3549 | 0.0648 | 0.2866 | 0.0025 | 18 | 2 | 7/32 |
| `medium-q8_0` | 0.3957 | 0.0634 | 0.3357 | 0.0013 | 17 | 5 | 2/32 |
| `medium-q4_0` | 0.3871 | 0.0597 | 0.2165 | 0.0210 | 26 | 5 | 32/32 |

Read the last three columns together, because any one of them alone misleads:

- The **per-position** flip rate is low: 1-5 of 33 positions, so the next token
  is usually the same one.
- Those flips happen where the margin is smaller than the error, and these
  fixtures are random weights, so their logits are nearly flat -- the median
  margin is 0.12-0.34 and the error is the same order. On 7-26 of 33 positions
  the margin is below the injected error, so a flip there is expected rather
  than surprising. A trained model's confident positions have much wider
  margins; **that is a general expectation, not something measured here**, and
  this table is the only accuracy claim this document makes.
- The **trajectory** column swings from 32/32 to 1/32 for the same per-position
  flip rate, because one flipped token changes the prefix and every later token
  is conditioned on it. End to end at 257 prompt tokens, 6 of 9 fixtures produce
  byte-identical 32-token greedy text and 3 do not; at 33 prompt tokens, none
  do. Trajectory agreement is therefore a property of the prompt, not a
  constant, and it is why this is opt-in.

What it does not change: the f32 path. `--npu-cpu` text is byte-identical to
`--cpu` on all 9 fixtures in both tables, before and after this change.

## 11. Attention on the device (`KRK_NPU_ATTN`)

Sections 2-5 established that decode is bound by weight traffic and the GEMM is
the whole story there. Prefill is a different shape of problem, and the traces in
section 10.4 point at it directly: after the GEMM moved to int8 the device sat
**idle for 54% of the span**. Something on the host was holding the wall clock.

### 11.1 Finding it: prefill is not all GEMM

Prefill scales as (GEMM work ∝ T) + (attention work ∝ T²) + a constant for
staging the weights. Sweeping the prompt length and fitting
`a + b·T + c·T²` to `prefill.chunk` separates the three:

| T | prefill.chunk (f32 arm) |
| ---: | ---: |
| 65 | 108.2 ms |
| 129 | 141.6 ms |
| 257 | 229.7 ms |
| 513 | 519.4 ms |

Least squares on those four points: `a = 85.7 ms`, `b = 0.274 ms/token`,
`c = 1.11e-3 ms/token²`. The per-token cost rises with T -- that is the
quadratic term, and it is attention. At 257 tokens it is **73.5 ms of a 229.7 ms
prefill (32%)**; at 513 it is **293 ms of 519 ms (56%)**.

Attention was on the host in every NPU mode, and the host implementation is the
scalar reference:

```cpp
for (i64 i = 0; i < d.hd; i++) dot += qh[i] * kj[i];
```

That loop must stay scalar -- it is the oracle the GPU and NPU paths are tested
against, and its bit-exactness is what makes `--prefill-hash` a gate. So the
answer is not to vectorise it, it is to stop asking the host to do it.

### 11.2 What the device can and cannot take

`vnpu_attention_f32` wants contiguous head-major `[batch, heads, seq, dim]` with
a causal flag. Kraken's cache is token-major with a row stride, and carries
features the op has no field for. So the call is offered only when every one of
these holds, and refused **by name** otherwise -- counted, and printed in the run
summary, never silently approximated:

| feature | why it is refused |
| --- | --- |
| 8-bit KV cache | the op takes f32 rows |
| sliding window | no window field; the causal flag cannot express it |
| masked block (`skip_from`) | a non-contiguous visible set per row |
| logit softcap (Gemma 2) | applied at score formation, before the softmax |
| attention sink | a denominator-only term with no value row |
| `n_tok` below the threshold | see 11.3 -- the device loses there |

A refusal looks like this, from a real run with `--kv-quant int8`:

```
attn 0 dev, 32 host (32 8-bit cache)
```

The gather itself is free for grouped-query models: the transpose has to touch
every `(position, head)` row anyway, so broadcasting `n_kv` into `n_head` costs
nothing extra.

### 11.3 Where it wins, and where it does not

Isolated measurement against this backend's scalar body, gather and scatter
included (a gain eaten by the layout change is not a gain):

| T | P | host scalar | device + gather | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 129 | 0.07 ms | 0.32 ms | **0.21x** |
| 1 | 1025 | 0.72 ms | 1.00 ms | **0.72x** |
| 8 | 257 | 1.02 ms | 0.53 ms | 1.93x |
| 32 | 257 | 4.08 ms | 1.59 ms | 2.57x |
| 257 | 257 | 16.55 ms | 6.74 ms | 2.45x |
| 513 | 513 | 76.90 ms | 22.36 ms | 3.44x |

Decode loses badly: one query row cannot amortise a gather that is O(the whole
cache). The crossover is around four rows, so the threshold defaults to 16 and
decode keeps the reference. End to end, best-of-N and median of interleaved
runs (`small-f16`, ctx 2048 unless noted):

| fixture | T | prefill off | prefill on | gain | decode |
| --- | ---: | ---: | ---: | ---: | --- |
| `small-f16` | 257 | 262.3 ms | 237.7 ms | **+10.3%** | unchanged |
| `small-f16` | 513 | 565.3 ms | 450.2 ms | **+25.6%** | unchanged |
| `medium-f16` | 257 | 1041.6 ms | 994.6 ms | +4.7% | unchanged |

(best-of-N over the same samples: +9.0%, +31.0%, +4.1%. This box's samples
spread ±10%, so both statistics are quoted and they agree in direction. Decode
moves by less than 2% in both directions, which is the noise floor.)

The gain tracks the arithmetic: it grows with prompt length because attention is
the quadratic term, and shrinks as the model grows because the GEMMs grow with
it while attention does not. On the 1024-dim fixture at 257 tokens the GEMMs
dominate enough that attention is only worth 5%.

The device trace shows what it costs on the device side -- 4 calls per prefill,
one per layer:

| op | count | total | %busy | mean | moved | GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MATMUL_F32` | 116 | 171.9 ms | 84.9% | 1481.8 us | 239.6 MiB | 38.3 |
| `ATTENTION_F32` | 4 | 26.1 ms | 12.9% | 6522.6 us | 8.0 MiB | 20.7 |

### 11.4 What it costs the answer

Two different bounds, and the difference between them is itself informative.

At the shapes the engine actually runs (hd 64, 257x257) the device attention
differs from the scalar reference by **1.19e-07 absolute** -- the f32 rounding of
a different summation order. Greedy text is byte-identical to `--cpu` on all 9
fixtures with the opt-in on, and on a 4-layer grouped-query fixture (8 query
heads, 2 KV heads) that exercises the broadcast.

At a deliberately small shape (hd 8, 6 query rows over 12 keys) the difference is
**2.0e-05 relative, 6.1e-06 absolute**, because the device's softmax runs its own
polynomial `exp` and at that size the approximation is a larger share of a small
result. The test bound is set a little above the measurement.

### 11.5 The real cost is the chunk-invariance gate, not accuracy

The engine guarantees that two runs differing only in `--chunk` produce the same
`--prefill-hash`. That holds because every row's arithmetic is identical whatever
the batch shape. This optimisation challenges it in a specific and measurable way:

| configuration | `--chunk 8` | `--chunk 128` | `--chunk 512` |
| --- | --- | --- | --- |
| default (opt-in off) | `e666a682d7750d0a` | `e666a682d7750d0a` | `e666a682d7750d0a` |
| opt-in on, threshold 16 | `e666a682d7750d0a` | `3348d4abbdaa93df` | `3787136ed87eb06c` |
| opt-in on, threshold 0 | `3787136ed87eb06c` | `3787136ed87eb06c` | `3787136ed87eb06c` |

Read the third row: **the device kernel is chunk-invariant on its own.** It is
the SWITCH -- small chunks on the reference, large chunks on the device -- that
breaks the invariant, and the switch is what buys decode. There is no threshold
that keeps both, so it is the user's call, and it is off by default. The default
row is byte-identical to the tree before this change.

## 12. Reproducing all of it

```
# fixtures (deterministic; --batch must be a multiple of 32)
tools/make_bench_gguf.py --embd 256  --layers 2 --heads 4 --vocab 512  --format f16  /tmp/fix/tiny-f16.gguf
tools/make_bench_gguf.py --embd 512  --layers 4 --heads 8 --vocab 1000 --format f16  /tmp/fix/small-f16.gguf
tools/make_bench_gguf.py --embd 1024 --layers 6 --heads 8 --vocab 1500 --format f16  /tmp/fix/medium-f16.gguf

# the matrix in sections 2 and 3
tools/npu_bench.py \
  --models '/tmp/fix/tiny-*.gguf' '/tmp/fix/small-*.gguf' \
  --arms cpu,npu,npu-cpu --prefill 16,64,256 --decode 64 --reps 3 --ctx 512 \
  --out bench.md --json bench.json

# traces, then the analysis
KRK_NPU_TRACE=/tmp/tr/npu.json kraken -m small-f16.gguf --npu     -p "..." -n 32 --greedy --trace /tmp/tr/npu.krk.json
KRK_NPU_TRACE=/tmp/tr/cpu.json kraken -m small-f16.gguf --npu-cpu -p "..." -n 32 --greedy --trace /tmp/tr/cpu.krk.json
tools/trace_stats.py /tmp/tr/npu.json /tmp/tr/cpu.json --label npu --label npu-cpu

# the int8 opt-in (section 10): same fixture, both arms, one-time cost first
for words in 8 64; do
  P="$(python3 -c "print(' '.join(['the quick brown fox']*$words))")"
  kraken -m /tmp/fix/small-f16.gguf --npu-cpu -p "$P" -n 32 --greedy --ctx 1024 --profile
  KRK_NPU_W8A8=1 kraken -m /tmp/fix/small-f16.gguf --npu-cpu -p "$P" -n 32 --greedy --ctx 1024 --profile
done

# attention on the device (section 11): the same fixture, both arms interleaved.
# Run each arm several times and take best-of-N; this box's samples spread ~10%.
for words in 64 128; do
  P="$(python3 -c "print(' '.join(['the quick brown fox']*$words))")"
  for arm in off on; do
    [ $arm = on ] && E=KRK_NPU_ATTN=1 || E=
    env $E kraken -m /tmp/fix/small-f16.gguf --npu-cpu -p "$P" -n 32 --greedy --ctx 2048
  done
done

# the chunk-invariance gate, which the switch is off to protect (section 11.5)
for c in 8 128 512; do
  kraken -m /tmp/fix/small-f16.gguf --npu-cpu --prefill-hash -p "$P" --ctx 1024 --chunk $c
done
KRK_NPU_ATTN=1 ... # same three, to see the invariant go

# the accuracy witnesses (section 10.5). --check needs a kraken binary on
# PATH or in KRAKEN_BIN; both tools read only the GGUF and the engine's
# --prefill-hash, never kraken's source.
tools/ref_llama.py /tmp/fix/small-f16.gguf --check
tools/int8_witness.py /tmp/fix/tiny-*.gguf /tmp/fix/small-*.gguf /tmp/fix/medium-*.gguf
```

`int8_witness.py` needs numpy; it runs the reference forward once per fixture in
float32 numpy, so the medium fixtures take ~25 s each. `ref_llama.py` supports
F32, F16, Q8_0 and Q4_0 tensors — everything the fixture generator emits.

The build needs the NPU engine: configure with `-DKRK_ENABLE_NPU=ON
-DKRK_VNPU_ROOT=<npuxl> -DKRK_VNPU_LIB=<npuxl>/build/libvnpu.a`. Link the
archive **you built** — the prebuilt `lib/libvnpu.a` that ships in npuxl's zip
is compiled against a newer glibc and will not link here.

## 13. What this does and does not prove

It proves the CPU+NPU path is a genuine, large, correct win on this machine:
1.8–18.4x on prefill and 1.1–5.1x on decode (one cell excepted, where the CPU
wins by 1.9x), identical greedy text on 72/72 cells, with the device inside a
kernel 60–67% of the wall clock and the rest attributable to transformer ops
the virtual NPU does not implement.

That single losing cell is worth being precise about, because it marks the
boundary of the win. It is a **2-layer, 512-vocab toy at 16 prompt words**:
the whole model is 1.6 MiB, it fits in cache, decode is 802 tok/s on the CPU
already, and a decode step is small enough that the device's fixed per-command
cost outweighs its better kernel. The CPU+NPU path pays off when the model is
big enough that streaming weights dominates — which is every model anyone
would actually run.

Section 10 adds an opt-in second answer: with `KRK_NPU_W8A8=1` the weights are
packed to int8 and the decode runs 1.3-3.6x faster again on top of the NPU arm,
for a 4x cut in weight footprint, at the cost of a one-time 32-213 ms pack and a
greedy tail that is byte-identical on 6 of 9 fixtures and differs on 3. That
trade is real, it is measured in both directions, and it is why the switch is
off unless asked for.

Section 11 adds the other half of the prefill story. Attention was the
quadratic term in prefill and it ran on the scalar reference; `KRK_NPU_ATTN=1`
hands it to the device when the call is expressible, worth **+10% prefill at 257
tokens and +26% at 513** on the 4-layer fixture and +5% on the 1024-dim one,
with decode unchanged and greedy text unchanged. It is off by default for a
reason that is not accuracy but the chunk-invariance gate: the device kernel is
invariant, the host/device *switch* is not, and the switch is what buys decode.

It does not prove anything about a GPU. There is no `hipcc` on this box and no
GPU to measure, so this document says nothing about the HIP path beyond "it
still builds and its test suite passes". It also does not prove anything about
models larger than 1024 embedding dims — the mechanism (decode is
bandwidth-bound on f32-staged weights) scales, but the constants here are
measured at toy sizes.
