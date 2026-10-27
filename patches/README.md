# Patches carried against npuxl

Kraken links the virtual-NPU engine from
[`riteshnair/npuxl-improved`](https://github.com/riteshnair/npuxl-improved).
That repository ships its source as `Maxnpuxl.zip.zip` rather than as a browsable
tree, so fixes against it are carried here and applied by hand.

## `npuxl-0001-trace-timestamp-precision.patch`

38 lines against `src/core/runtime.cpp`. `write_trace` emitted start timestamps
through a stream left at the default 6 significant digits, so an absolute
monotonic microsecond stamp near 1.2e9 printed as `1.1953e+09` -- a 10 ms grid.
Durations were unaffected, which is why the defect looked like a coarse clock.
The fix rebases onto the first event and pins `fixed(3)`.

Apply from the directory containing `src/`:

```
patch -p1 < patches/npuxl-0001-trace-timestamp-precision.patch
```

Diagnosis and before/after measurements: section 9 of
[`docs/CPU-VS-NPU-BENCHMARK.md`](../docs/CPU-VS-NPU-BENCHMARK.md).

## `npuxl-0002-zen3-avx2-int8.patch`

Makes the AVX2 arm a real int8 arm. Applies to the same tree as 0001 (they touch
different files) and is a single commit:

```
src/kernels/kernels.cpp   the packed int8 GEMM
src/kernels/llm.cpp       vectorised per-row quantise
tests/isa_parity.cpp      new  cross-ISA byte-exact comparison
tests/bench_i8.cpp        new  the benchmark below
docs/ZEN3-AVX2.md         new  the write-up
CMakeLists.txt            builds the two new test binaries
```

**Why it was needed.** The AVX2 int8 kernel widened 8 int8 to int32 and used
`mullo_epi32`, after gathering B one byte at a time out of the row-major copy:
8 MACs per iteration. Measured **4.89 GFLOP/s against scalar's 3.80**, i.e.
scalar with extra steps. A Zen 3 machine has no AVX-512 at all, so this was the
only int8 path it had.

**What it does now.** `maddubs` + `madd_epi16` over the packed
`[n_block][k4][16][4]` layout: 32 MACs in two instructions. The `+128` trick the
VNNI path uses cannot be reused (`maddubs` accumulates to int16 and
`255*127*2` overflows), so it uses `|a| * (b*sign(a))`, which peaks at 32512 and
needs no correction term because the packing zeroes the padding.

| one core, 768x768 expert | scalar | old avx2 | new avx2 | gain |
|---|---:|---:|---:|---:|
| 1 row (decode) | 0.358 ms | 0.326 ms | **0.075 ms** | **4.3x** |
| 32 rows | 9.456 ms | 8.321 ms | **0.548 ms** | **15.2x** |
| 128 rows | 39.735 ms | 30.869 ms | **1.960 ms** | **15.7x** |

**Verified.** `vnpu-selftest` 16/16 and `vnpu-fuzz 60` clean on `scalar`,
`avx2` and `avx512`; `vnpu-isa-parity` (new) shows the int8 GEMM and the
quantiser **bit-identical** across all three, including K not a multiple of 4
and N not a multiple of 16.

Apply with:

```
cd <npuxl tree> && git am npuxl-0002-zen3-avx2-int8.patch && make -j
VNPU_FORCE_ISA=avx2 ./bin/vnpu-selftest
```

## `npuxl-improved-patched.zip`

The complete npuxl tree from `Maxnpuxl.zip.zip` with patch 0001 already applied,
for anyone who would rather take the fixed source in one piece than apply a
diff. It differs from upstream in exactly two files:

```
src/core/runtime.cpp   +13 -1   the fix
PATCHES.md             new      defect, reproducer, before/after, verify steps
```

The `lib/libvnpu.a` and `lib/libvnpu.so` inside it are the **originals from
upstream**, not rebuilt here: they are glibc- and toolchain-specific, so
rebuilding them on one machine and shipping the result only moves a link error
onto someone else's. Consequence: a consumer linking those prebuilt binaries
keeps the 10 ms grid until they rebuild from the patched source.

Verified from a cold extract: `make -j` builds clean, `bin/vnpu-selftest`
reports 16 tests / 0 failures, and `bin/vnpu-cli trace` writes
`"ts":0.000` with distinct fine-grained start times.
