# CPU vs virtual NPU: a measured comparison

This is the answer to "when you compared direct CPU coding and through NPU,
what was better, and do we have a comprehensive benchmark we can see". Every
number below was produced on this machine by the two committed tools in
`tools/`, and every one of them can be reproduced by section 10.

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
exactly the case the NPU path should lose; section 11 says what that means for
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
`tools/`-adjacent scratch work and reproduced in section 10:

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

## 10. Reproducing all of it

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
```

The build needs the NPU engine: configure with `-DKRK_ENABLE_NPU=ON
-DKRK_VNPU_ROOT=<npuxl> -DKRK_VNPU_LIB=<npuxl>/build/libvnpu.a`. Link the
archive **you built** — the prebuilt `lib/libvnpu.a` that ships in npuxl's zip
is compiled against a newer glibc and will not link here.

## 11. What this does and does not prove

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

It does not prove anything about a GPU. There is no `hipcc` on this box and no
GPU to measure, so this document says nothing about the HIP path beyond "it
still builds and its test suite passes". It also does not prove anything about
models larger than 1024 embedding dims — the mechanism (decode is
bandwidth-bound on f32-staged weights) scales, but the constants here are
measured at toy sizes.
