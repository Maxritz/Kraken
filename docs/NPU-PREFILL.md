# NPU prefill: where the time actually goes

Trace-mode investigation into the claim that the virtual NPU's prefill can be
made 8x faster with fusing and tiling.

Short version: **the 8x is not reachable on this hardware, and the reason is not
the kernels.** In a 58-token prefill the device is busy for 864 ms of a 2464 ms
wall. The GEMMs are already the fastest thing in the run; the other 1600 ms is
the host, and no amount of tiling inside the kernel touches it. The arithmetic
ceiling for this exact build, with the host cost driven to *zero* and the current
f32 kernels kept, is 4.7x. Getting to 8x needs the int8 kernels and a packed-
weight cache as well, and even then the host has to come down by about 9x.

Two real defects were found and fixed on the way:

| # | Defect | Effect | Fix | Measured |
|---|--------|--------|-----|----------|
| 1 | Staging budget was a flat 1024 MiB and silently declined 393 of 1087 GEMMs to the host | 83% device idle | default budget derived from available RAM, env override kept | 3042 -> 2357 ms (1.29x) |
| 2 | `KRK_NPU_W8A8` packed every weight on first use, including in a prefill that never reuses it | +1.0 s of pure loss | pack only when a weight is reused (decode), not on prefill | 3395 -> 2381 ms (1.43x) |

Both are visible in the run summary, not inferred.

---

## 1. Method

One prompt, one model, one box, all arms interleaved and repeated; medians over
three runs, every sample listed. Nothing below is a single measurement.

- Model `fix/moe-2g-q8_0.gguf` (1230.5 MiB, q8_0 MoE), prompt 58 tokens,
  `-n 2 --greedy --ctx 1024 --chunk 512`.
- Host: 2 vCPU Xeon @2.60 GHz, 3.9 GiB RAM, AVX512+VNNI, no HIP, gcc 12.2.
- Device: `npuxl` virtual NPU, `VNPU_FORCE_ISA=avx2` (Zen 3 shape), 2 threads,
  64x256x64 tiles. On this box the "device" is the same silicon as the host, so
  every device number is an *emulated* number. That is the point of a separable
  NPU backend, and it is also why the ceilings here are softer than on a real
  accelerator.

The CPU arm is the real competitor, not a straw man: same box, same SIMD, built
`-march=native`, and it is the default backend.

## 2. Baseline

Prefill only, 58 tokens, `-n 0` so no decode step can land inside the
measurement (`--npu-cpu` = host execution with the NPU offload layer on;
`--cpu` = plain CPU backend):

| arm | median | vs CPU | samples |
|-----|--------|--------|---------|
| `--cpu` | 4089 ms | 1.00x | 3677, 4473, 4089 |
| `--npu-cpu` (fixed default) | 2357 ms | **1.73x** | 2357, 2327, 2699 |
| `--npu-cpu` + W8A8, reuse rule | 2381 ms | 1.72x | 2381, 2325, 2444 |
| `--npu-cpu` + W8A8, eager | 3395 ms | 1.20x | 3242, 3395, 3687 |
| `--npu-cpu`, old 1024 MiB budget | 3042 ms | 1.34x | 3042, 2888, 3237 |

Decode, 200 tokens, same model (ms per token, two reps):

| arm | ms/tok | vs off |
|-----|--------|--------|
| W8A8 off | 131.37 | 1.00x |
| W8A8 reuse rule | 72.84 | 1.80x |
| W8A8 eager | 61.15 | 2.15x |

The CPU arm carries most of the noise (3677-4473 ms, 18% spread on a 2-vCPU
shared box); the NPU arms are stable to about 10%. Ratios are quoted to two
figures at most.

Note the measurement trap this replaced: with `-n 2` the decode steps land after
the prefill timer stops, but any lazy weight packing they trigger can be pulled
into the same process' later behaviour, and an earlier version of this table
mixed the two phases. Prefill claims in this document are all `-n 0`.

## 3. Decomposition

Device trace of the fixed arm (`KRK_NPU_TRACE`), 58-token prefill plus 2 decode
steps. Chrome trace, 1544 device ops:

```
device busy 863.6 ms of 2614.4 ms span  ->  67% idle
20.62 GFLOP on the device (24.8 GFLOPS while busy)

  MATMUL_F32   1087   829.0 ms  96% of busy   24.8 GFLOPS
  SWIGLU_F32    295    23.6 ms   3%            1.2 GFLOPS
  ADD_F32        80     6.0 ms   1%            0.9 GFLOPS
  RMSNORM_F32    82     5.0 ms   1%            1.8 GFLOPS
```

Host-side accounting from `KRK_NPU_OPTIME=1`, same run:

```
submit    8.5 ms over 357 batches    (2.4% of prefill)
sync    893.6 ms over 357 batches    (36%)
stage   350.6 ms over 758 weights    (14%)   q8_0 -> f32 expansion, first touch
other  1211.2 ms                     (49%)   rope/attention/kv/routing/gather
prefill 2463.9 ms
```

So the budget is: **36% waiting for the device, 49% host compute, 14% one-time
weight expansion, 0.3% submission.** The device's own work is a third of the
run.

GEMM shapes, by class, from the trace:

| class | count | time | GFLOPS |
|-------|-------|------|--------|
| 68.42 MFLOP (58x768x768) | 94 | 203.9 ms | 31.5 |
| 1.18 MFLOP (1x768x768, decode rows) | 385 | 104.7 ms | 4.3 |
| 22.81 MFLOP (58x768x256) | 40 | 35.9 ms | 25.4 |
| 67.24 MFLOP | 12 | 25.2 ms | 32.0 |
| 47.19 MFLOP | 15 | 23.5 ms | 30.1 |

Prefill is batched correctly: the row histogram is `1:447 2-8:174 9-32:159 33+:307`,
i.e. 640 of the prefill's GEMMs carry many rows and only the decode steps are
single-row. The batched-prefill work that landed in #15 is in force here.

## 4. Root causes, ranked by cumulative cost

### 4.1 Host compute between batches - 1211 ms (49%)

Not a defect, a boundary. Rope, attention, KV cache, embedding, MoE routing and
top-k have no NPU opcode and stay on the host, and every one of them is
fenced by `npu_flush_()` in `src/backend_cpu.cpp` because it may read memory the
device still has queued. The device is idle across all of it. This is the cost of
the offload boundary, and it is why the ceiling is 4.7x, not 8x.

Evidence it is the boundary and not launch overhead: a trivial 1x64x64 op
through submit+sync costs **0.0642 ms** (median of 200, p10 0.0609, p90 0.0777).
357 batches at that price is 23 ms, not 1600 ms. Per-op cost is also flat from
batch 1 to batch 256 (0.562 / 0.506 / 0.509 / 0.529 ms), so there is nothing to
win by chasing submission.

### 4.2 Sync wait - 894 ms (36%)

The host blocks in `vnpu_stream_sync` after each batch, and there is no
double-buffering: no work is prepared while the device runs. 357 batches for
1544 ops. This is a structural omission, not a tuning knob - fixing it means
letting the host queue ahead of a batch it will need instead of waiting, which
is a dependency-tracking change across `backend_cpu.cpp`.

### 4.3 One-time weight expansion - 351 ms (14%)

`Offload::weight()` dequantizes q8_0 to f32 on first touch and caches it. 758
weights cost 351 ms. Unavoidable *unless* the device grows a q8_0 B-matrix
kernel; npuxl has none, so the choice is f32 expansion (4 bytes/weight) or int8
packing (1 byte/weight, plus a scale).

### 4.4 Staging budget silently declined 393 GEMMs - fixed

The old flat 1024 MiB default pushed 393 of 1087 GEMMs back to the host, and the
trace showed exactly what that costs: ~30 ms holes between SWIGLU and SWIGLU
with **no device matmul in between**, i.e. an expert down-projection running on
the CPU, 83% device idle, 508 ms busy in a 3059 ms span.

Single runs, `-n 0`, so the declines are the only thing that changes:

| budget | declines | prefill |
|--------|----------|---------|
| 1024 MiB (old default) | 393 | 3207 ms |
| 2048 MiB | 0 | 2495 ms |
| 4096 MiB | 0 | 2360 ms |
| derived from RAM (new default) | 0 | 2435 ms |

The derived default is three quarters of `host_available_bytes()`; on a 3.9 GiB
box that lands near 2048 MiB, which is why it tracks that row rather than 4096.
The interleaved medians in section 7 are the numbers to quote for the fix; this
table is here to show that declines, not anything else, are what moves.

The default is now three quarters of `host_available_bytes()`, floored at the
old 1024 MiB so no box regresses, and `KRK_NPU_WEIGHT_MB` still overrides it
exactly.

### 4.5 W8A8 packed weights the prefill never reuses - fixed

Packing costs about 2.3 ms per 768x768 weight (253 MB/s standalone, 1718 ms for
737 weights) and buys back 418 ms of GEMM time (the same GEMMs are 803 ms as f32
against 385 ms as int8). On a prefill that touches each expert once the pack
never repays itself: eager W8A8 measured 3395 ms against 2357 ms with it off.

Two signals were tried before one worked, and the first one is worth recording:

1. **A request counter** (pack on the second request): 3113 ms. It fails because
   the load-time expert warmup touches every weight once, so prefill's *second*
   request earned a pack the prefill never repaid.
2. **Row count alone** (`rows > 1` means prefill): 3172 ms, packing 106 weights.
   Better, still a 563 ms loss, because a prefill does issue single-row GEMMs --
   106 weights' worth on this model.
3. **Row count plus reuse:** defer every `rows > 1` GEMM, and for `rows == 1`
   defer the first one-row request for that weight and pack only from the
   second. Decode re-requests the same weight on the next token; a one-shot
   single-row prefill GEMM never comes back. Result: **0 weights packed in a
   prefill-only run, 2381 ms against 2357 ms off** -- free within noise -- while
   decode still pays the pack once and keeps 72.84 ms/token against 131.37.

`KRK_NPU_W8A8_EAGER=1` restores pack-on-first-use. The break-even between the
reuse rule and eager is about **87 generated tokens** at 200-token decode rates
(eager costs 1014 ms more prefill and saves 11.7 ms per token), so the default
optimises the phase this document is about and long generations should set the
flag.

### 4.6 Tiling: M=58 is the worst band the kernel has - measured, not yet fixed

`vnpu-cli bench --shape Mx768x768`, f32, one thread-count, same binary:

| M | ms | GFLOPS | us/row |
|---|----|--------|--------|
| 1 | 0.4929 | 2.39 | 493.0 |
| 16 | 0.5263 | 35.86 | 32.9 |
| 32 | 0.8486 | 44.48 | 26.5 |
| 33 | 1.1270 | 34.54 | 34.2 |
| 40 | 0.9391 | 50.25 | 23.5 |
| 48 | 1.1059 | 51.20 | 23.0 |
| 56 | 1.4644 | 45.11 | 26.1 |
| **57** | 2.3520 | 28.59 | 41.3 |
| **58** | **2.6615** | **25.71** | **44.9** |
| **59** | 3.3196 | 20.97 | 56.3 |
| 60 | 1.6331 | 43.34 | 27.2 |
| 62 | 2.8168 | 25.96 | 45.4 |
| 63 | 3.3913 | 21.91 | 53.8 |
| 64 | 1.5256 | 49.49 | 23.8 |
| 72 | 1.9730 | 43.05 | 27.4 |
| 128 | 3.1549 | 47.86 | 24.6 |
| 256 | 5.4031 | 55.89 | 21.0 |

**M=58 is 1.74x slower than M=64 while doing 10% less work.** Prefill's M is the
token count of the chunk, so a 58-token prompt sits exactly in the pathological
band; 57-63 and 33 are all bad. The tile is 64 rows, and the remainder path
costs more than the work it avoids.

Padding M up to a tile-friendly value is the one tiling change the numbers
actually support: 203.9 ms of 68.42 MFLOP GEMMs would become ~123 ms, about 80 ms
of the 2464 ms prefill (3.3%). It is not implemented because it is not free -
A must be staged into a 64-row scratch and C copied back, plus a fence - and the
honest reading of the numbers is that 3% does not justify the risk in the same
change as two 20%+ fixes. It is the first item under next actions.

`vnpu-cli bench` also shows the M=1 GEMV path at 2.39 GFLOPS (493 us/row against
21 us/row at M=256), which is a decode problem, not a prefill one.

## 5. Hardware analysis: what the ceiling really is

| quantity | value | source |
|----------|-------|--------|
| device busy, 58-token prefill | 863.6 ms | trace |
| device busy, int8 kernels | 385 ms | earlier int8 trace |
| CPU arm, same prompt | 4089 ms (median) | interleaved run |
| single-op round trip | 0.0642 ms | 200-iteration probe |
| in-graph f32 GEMM | 24.8-31.5 GFLOPS | trace |
| standalone f32 GEMM, M=58 | 25.7 GFLOPS | bench |
| standalone f32 GEMM, M=256 | 55.9 GFLOPS | bench |
| standalone int8 GEMM | 151.9 GFLOPS | bench |
| peak f32 on 2x AVX512 cores | ~166 GFLOPS | 2 x 2.6 GHz x 32 flop/cycle |

The in-graph GEMM rate (24.8 GFLOPS) matches the standalone rate at the same
shape (25.7 GFLOPS at M=58). So **the kernel is not being starved by the host
during a GEMM** - it runs at its own shape-limited speed, and the difference
between 25 and 56 GFLOPS is entirely the M dimension, not interference.

Ceilings, in order of how much they assume:

| scenario | prefill | vs CPU |
|----------|---------|--------|
| today | 2357 ms | 1.73x |
| host cost driven to zero, f32 kernels kept | 864 ms | 4.7x |
| ... plus M-padding to tile multiples | ~784 ms | 5.2x |
| ... plus int8 kernels (385 ms device) | 385 ms | 10.6x (needs a paid-off pack) |
| **8x target = 511 ms** | | needs int8 **and** host < 130 ms |

8x is therefore not a tiling problem. It needs the int8 kernels *and* the host
gap down by roughly 9x *and* the 2.4 s weight pack amortised or cached. The
current build delivers 1.73x with none of that host work removed.

## 6. Optimisation plan, in cost order

1. **Kill the sync wait** (894 ms). Let the host queue the next batch while the
   device works: track which buffers a queued batch reads and writes, and only
   sync where the host actually touches one. Target: the 894 ms becomes overlap.
2. **Shrink the host compute** (1211 ms). Attention and rope are the quadratic
   and linear hot spots and both are pure host work today. `KRK_NPU_ATTN` exists
   and is opt-in.
3. **Stop expanding q8_0 to f32** (351 ms, and it also removes the 4x memory
   blow-up). Either teach the device a q8_0 B kernel or make the int8 pack cheap
   enough to be the default - see 4 for why it currently is not.
4. **Pad M to tile multiples** (~80 ms measured, first item that is purely a
   kernel-side change).
5. **Cache packed int8 weights across runs** so the 2.4 s pack is paid once per
   model rather than once per process; this is what makes the int8 ceiling real.

## 7. Results

| arm | before | after | ratio |
|-----|--------|-------|-------|
| `--npu-cpu`, prefill 58 tok, old budget | 3042 ms | 2357 ms | 1.29x |
| `--npu-cpu`, prefill 58 tok, W8A8 | 3395 ms | 2381 ms (reuse rule) | 1.43x |
| declines on the same run | 393 / 1087 | 0 | - |
| vs CPU arm | 1.34x | 1.73x | - |
| decode, W8A8 reuse rule | 131.37 ms/tok | 72.84 ms/tok | 1.80x |

Correctness: `kraken-tests` **4791/4791**, including the int8 parity digests in
`tests/isa_parity.cpp`. No kernel was modified; every change is in the offload
policy, and the f32 GEMM path the device runs is byte-for-byte the path that
shipped in #15.

## 8. Next actions

1. M-padding for non-tile-aligned M (measured 1.74x at M=58, ~80 ms of prefill).
2. Overlapped submission to remove the 894 ms sync wait.
3. Device attention for the prefill quadratic term.
4. Packed-weight cache keyed by model+layout, to make int8 the default.
5. Re-run all of this on real Zen 3 silicon; every ceiling above is an emulated-
   device ceiling and the int8 gap in particular should widen.

## Reproducing

```sh
# baseline vs arms, interleaved
KRK_NPU_WEIGHT_MB=1024 ./build-npu/kraken -m fix/moe-2g-q8_0.gguf --npu-cpu \
    -p "<58-token prompt>" -n 2 --greedy --ctx 1024 --chunk 512
KRK_NPU_W8A8=1 ./build-npu/kraken ... same        # reuse rule (prefill-free)
KRK_NPU_W8A8=1 KRK_NPU_W8A8_EAGER=1 ./build-npu/kraken ...  # pack on first use

# isolate prefill: -n 0 keeps decode steps out of the measurement

# where the time goes
KRK_NPU_OPTIME=1 KRK_NPU_TRACE=/tmp/t.json ./build-npu/kraken ...

# the tiling table
vnpu-cli bench --shape 58x768x768
```
