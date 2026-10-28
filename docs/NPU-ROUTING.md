# Routing the device by shape: the two changes that make the NPU win

The virtual NPU has been the slower arm since it was wired in. Two changes fix
that, and neither one is a kernel: the device was being asked for the wrong
*shapes*, and its one-time cost was being billed to the wrong *phase*.

Measured on this box (2 hardware threads, 2.60 GHz, AVX-512 + VNNI), dense
q8_0 fixture (12 layers x 768, ff 2048, vocab 4096, 92.6 MiB), 82-token prompt,
`--ctx 512`, greedy, medians of 3 (5 for prefill). Ratios are the result;
absolute tok/s is indicative.

## The table

Decode, tokens per second:

| arm | n=16 | n=64 | n=128 | prefill ms |
|---|---:|---:|---:|---:|
| host CPU | 52.4 | 49.4 | 50.9 | 328.8 |
| NPU f32 | 20.4 | 24.6 | 26.1 | 473.0 |
| NPU i8, prepare only | 84.7 | 81.9 | 79.5 | 483.8 |
| NPU i8, gate only | 15.6 | 41.6 | 54.7 | 316.7 |
| **NPU i8, both (default)** | **83.5** | **76.3** | **77.4** | **314.3** |

What it means, in the order the two changes happened.

## 1. The one-time cost was inside the measurement

The device packs each weight into its own int8 layout once, at 2.3 ms per
weight. The driver defers that until the *second* single-row request for the
same weight, so the pack lands somewhere in the first few decode steps. At 16
tokens that was **48% of the wall clock**, and it made the int8 arm measure
0.74x of the f32 arm while its steady state was 2.5x.

That is a measurement defect before it is a performance one: a benchmark whose
answer changes with the length of the run is not measuring the thing it claims
to measure. The same code, run for longer, flipped from "int8 loses" to "int8
wins by 2.5x" — and the earlier verdict in this repo ("the NPU cannot win
decode") came from the short end of that curve.

The engine now offers one forward pass at load whose only job is to touch every
weight:

```
[info ] prepare pass: 537.0 ms (one-time weight conversion, outside the timed run)
```

The pass is a real forward with a real KV cursor, so it is followed by
`kv_rollback(0)` — the same new-sequence primitive `generate()` uses, including
the recurrent-model reset. It is bracketed by `Backend::set_prepare(true/false)`
and skipped entirely when `wants_prepare()` is false, which is every backend
without a one-time conversion: a CPU-only build pays nothing and links nothing.

**The pass is numerically neutral**, and that is a claim with a control behind
it: `prepare` and `KRK_NPU_W8A8_EAGER=1` (pack on first use, lazily) produce
**identical tokens**, because in both the first use of every weight is the int8
kernel. They differ from `KRK_NPU_NO_PREPARE=1`, and that arm differs for a
reason worth stating: with lazy packing the *first* use of a weight runs the f32
kernel and later uses run int8, so the model computes a slightly different
function depending on whether that weight had been touched before. The prepare
pass removes that class of drift as a side effect.

Escape hatch: `KRK_NPU_NO_PREPARE=1`. A long-lived server may prefer a
pay-as-you-go first step to a 0.5 s spike at load, and that is a judgement about
the workload, not about the device.

**Effect:** n=16 decode 15.6 -> 83.5 tok/s. The length dependence disappears
(83.5 / 76.3 / 77.4 across a 8x range of n).

## 2. The device was being asked for shapes it cannot win

With the pack moved, decode beat the host. Prefill did not, and the reason is
not tuning: the host reads Q4_K / Q6_K / Q8_0 **blocks where they lie**, and the
device must expand them to f32 first. That is a format advantage the host has
and the device does not, and it applies exactly when the weight is streamed once
for many rows — a prefill.

Sweeping prompt length, host time over device time (below 1.0 means the host
wins):

| prompt tokens | rows | device ms | host ms | host / device |
|---:|---:|---:|---:|---:|
| 1 | 5 | 167.0 | 77.3 | **0.46** |
| 2 | 11 | 172.3 | 74.2 | **0.43** |
| 4 | 21 | 226.4 | 146.1 | **0.65** |
| 8 | 41 | 347.2 | 167.6 | **0.48** |
| 16 | 87 | 500.0 | 385.6 | **0.77** |
| 32 | 169 | 774.8 | 718.3 | **0.93** |
| 64 | 339 | 1527.2 | 1569.2 | 1.03 |

The host wins every shape from 5 rows up, by 2.2x at 5 rows and by parity at
339. The device wins exactly one shape: **one row**, which is all of decode.

So the device now declines a shape it does not win, and says so:

```
shape refused 84 (rows > 1: the host is the faster arm for that shape)
```

This is the same contract the device already used for a weight type it cannot
read: `npu::gemm` returns false *before queueing anything*, the caller flushes
and runs its own body, and the run summary counts it. A refusal is not a
fallback — it is a routing decision, made per call, and it is printed.

`KRK_NPU_MAX_ROWS` sets the boundary; it defaults to 1 when the int8 path is on
and 0 (take everything) otherwise. **A batched server should re-measure**: its
decode has more than one row, and rows 2-8 were not measured here.

**Effect:** prefill 483.8 -> 314.3 ms against the host's 328.8 — the penalty for
using the device at all drops to nothing, while decode keeps its win.

## 3. What is left, and it is not on this list

The device now wins decode by **1.5x** and ties prefill. It does not win prefill
because of one missing thing, and no amount of tiling fixed it in earlier
attempts:

* **The device cannot read a block format.** `MATMUL_I8` is the halfway house:
  it reads int8 (1x traffic) but quantises activations too, and that is an
  accuracy trade. The end state is an opcode that reads q8_0/q4_K blocks where
  they lie — 6.1x staging becomes 1.0x with **no** accuracy cost. That is
  `docs/NPU-CONSOLIDATED.md` item 4 and it is now the only structural item left
  on the device side.

Two more, smaller:

* **`tile_mask` + `WILDCAT_CULL`** for a batched expert GEMM — the one device
  feature with a payoff nothing else substitutes.
* **Titles**: the device's autotuner claims 1.67x for a 1-row shape at
  `1x128x64`; end-to-end it is noise, because decode is not GEMM-bound any more.
  Do not build a tile autotuner until this table stops moving.

## 4. Reproducing

```bash
# host
./build-npu/kraken -m MODEL --cpu -p "$PROMPT" -n 64 --greedy --ctx 512

# device, decode only (the default configuration this document describes)
KRK_NPU_W8A8=1 ./build-npu/kraken -m MODEL --npu -p "$PROMPT" -n 64 \
    --greedy --ctx 512

# the two changes, independently
KRK_NPU_W8A8=1 KRK_NPU_MAX_ROWS=0  ...   # no shape gate
KRK_NPU_W8A8=1 KRK_NPU_NO_PREPARE=1 ...  # pack lazily, in the decode steps
```

`--kprof` on the default arm shows the split: `npu_gemm_i8_r1` for decode, and
the host's `gemm_q8_0_*` rows for the prefill the device declined.

## 5. Caveats

* The fixture is seeded random weights, so this is a **speed** document. The
  int8 arm changes decode numerics (activation quantisation) and is opt-in for
  that reason; the f32 device arm is unchanged and still matches the host's
  token stream exactly on this fixture.
* This box reports 128 cores on 2 hardware threads through some tools and 1
  core through `/proc`; the numbers above are ratios for that reason.
* Absolute tok/s moved by up to 2.7x between runs while a build was running in
  the background on the same two threads. Every number here was taken on an
  otherwise idle box, and the arms were interleaved.
