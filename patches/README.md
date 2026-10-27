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
