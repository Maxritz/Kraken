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

## `npuxl-0003-force-isa-vnni-and-scale-order.patch`

Two defects that were invisible until VNNI was actually reachable. Applies on
top of 0002.

**`VNPU_FORCE_ISA=avx512` cleared the VNNI feature bit**, silently routing the
int8 GEMM to the AVX2 kernel. Forcing the baseline ISA must not subtract a
feature the host has, and this flag is documented as the way to make
deterministic cross-machine comparisons -- so every such comparison was
measuring AVX2 and calling it AVX-512. Measured single-threaded at
128x768x768: **20,304 MAC/s forced-avx512 against 82,395 with VNNI**, 4.1x,
with nothing in the output saying which kernel ran. `avx512-novnni` is now the
arm that wants the non-VNNI kernel, asked for by name.

This one is worth reading twice, because it caused two wrong conclusions in
this repository's own documentation before it was found, including a claim in
`ZEN3-AVX2.md` (inside 0002) that the 256-bit kernel beat the AVX-512 one. It
never did; both arms were the same kernel. That file now carries the
correction.

**The VNNI kernel multiplied its scales in the opposite order** from
`store_i8_c()`: `(dot * scale_b) * scale_a` where the oracle does
`(dot * scale_a) * scale_b`. Float multiplication is commutative but not
associative, so it differed in the last bit and was the only arm disagreeing
with the scalar reference. `vnpu-isa-parity` caught it the moment VNNI became
reachable, which is the case that test was built for.

Also: forcing an ISA the host cannot execute is now **refused by name** instead
of executing an illegal instruction. An AVX2-only host (Zen 3, pre-Skylake)
asked for `avx512` used to fault inside the first kernel.

```
src/core/runtime.cpp      the force-flag fix, the refusal, avx512-novnni
src/kernels/kernels.cpp   one line: the scale multiplication order
tests/bench_kernel.cpp    new  microkernel throughput, one tile, one worker
tests/bench_i8.cpp        one more shape (256 rows) to expose fixed overhead
CMakeLists.txt            builds bench-kernel against the static archive
```

`bench-kernel` exists because the public benchmark could not answer the
question that mattered. It calls `kernel_gemm_i8_range` directly, so tile
construction, pool dispatch and the barrier are excluded; that separation is
what turned "the VNNI epilogue looks slow" into "VNNI is 4x faster and you
were never running it".

Verified from a cold clone with both patches applied in sequence: `git am`
clean, `vnpu-selftest` 16/16 on unforced/scalar/avx2/avx512/avx512-novnni,
`vnpu-fuzz 60` clean, and `vnpu-isa-parity` reports **identical int8 GEMM
digests on all four arms** for every shape, including the K%4 and N%16 edges.

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
