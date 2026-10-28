# INT8 kernels by ISA tier, and why half of Intel's bias correction does not apply here

Intel's DL Boost guidance describes one ladder: Skylake-SP has AVX-512F/BW but no
VNNI, Cascade Lake adds VNNI, Ice Lake tunes it, Sapphire Rapids adds AMX. This
note maps that ladder onto the kernels in `src/quant.cpp`, states which tier this
build actually reaches, and records the one place where the block-quant formats
make the documented algorithm cheaper than the guide says.

## 1. The ladder, and what VPDPBUSD actually requires

`VPDPBUSD` is **unsigned byte x signed byte, accumulating into int32**. That is
the whole constraint, and everything else follows from it:

| tier | ISA | int8 dot | usable on |
|---|---|---|---|
| T2 | AVX2 + FMA | `VPMADDUBSW` -> `VPMADDWD` -> `VPADDD` | Haswell+, Zen 1+ (Zen 3 tops out here) |
| T3 | AVX-VNNI (256-bit) | `VPDPBUSD` (VEX form) | Alder Lake+, Zen 4/5, **Meteor Lake** |
| T4 | AVX-512F/BW, no VNNI | AVX2 sequence at 512 bits | Skylake-SP, Skylake-X |
| T5 | AVX-512 VNNI | `VPDPBUSD` (EVEX form) | Cascade Lake+, Ice Lake+, Zen 4/5 |
| T6 | AVX-512 BF16/FP16/IFMA | mixed precision only | Cooper Lake+, SPR |
| T7 | **AMX-INT8** | `TDPBSSD` and friends | Sapphire Rapids+ |

**T3 and T5 are different encodings of the same instruction and are not
interchangeable.** A kernel compiled with the `avx512vnni` target attribute emits
the EVEX form; dispatching it on an Alder Lake part that has AVX-VNNI and not
AVX-512 raises SIGILL. `cpu_has_vnni()` in `src/quant.cpp` checks
`avx512vnni` specifically for that reason, and it is documented there as "the
difference between a fallback and a crash".

The T7 row is the interesting one for later: **AMX has all four sign
combinations** (`TDPBSSD` s8xs8, `TDPBSUD` s8xu8, `TDPBUSD` u8xs8, `TDPBUUD`
u8xu8). AVX-512 VNNI has only `u8 x s8` and the 16-bit pair. Which brings us to
the correction.

## 2. Why Intel's example needs a bias correction, and why Q4_K/Q6_K do not

Both operands of a quantised dot are conceptually signed. `VPDPBUSD` wants the
first one unsigned, so the documented approach (Intel's Cascade Lake sample, and
every VNNI tutorial) is:

```
au = a + 128            // signed -> unsigned, modulo 256
acc += dpbusd(au, b)
sum -= 128 * sum(b)     // the bias the transform introduced
```

Two things are wrong with that in production, and Intel's own sample says so: the
correction is **O(K) per dot**, and it is **recomputed per output row**.

**ggml's block formats sidestep it entirely.** The quantised weight value is not
a signed int8 — it is an unsigned field plus a separate scale and (for Q4_K) a
separate min:

| format | weight field | range | signed operand? |
|---|---|---|---|
| Q4_K | 4-bit nibble | **0..15** | no — unsigned |
| Q6_K | 6-bit field | **0..63** | no — unsigned |
| Q8_0 | int8 | -128..127 | **yes** |

So for two of the three formats in the hot path, the weight operand is *already*
the unsigned side that `VPDPBUSD` wants, and the dot needs **no correction at
all**:

```cpp
// Q4_K: q_lo / q_hi are 0..15, x_lo / x_hi are int8. No bias term.
const __m256i i1 = _mm256_dpbusd_epi32(zero, q_lo, x_lo);
```

The minimum term — the `- min(x)` half of `w = scale*q - min*dmin` — does need a
`sum(x)` for the block. That is handled twice over, and neither way is O(K) per
row:

* **hoisted**: `sx[]` carries `sum(x)` per 32-value sub-block, computed once by
  the quantise pass, so the min term is one scalar FMA per sub-block against a
  value no weight row has to recompute (`dot_q4_K_q8_sx_vnni_d`);
* **pre-placed in the accumulator**: for Q6_K the `-32 * sum(4)` bias is loaded
  into the int32 lanes *before* the dot lands, so `VPDPBUSD`'s accumulator
  absorbs it and the correction costs zero instructions
  (`sq4[]` -> `pm` -> `_mm256_dpbusd_epi32(pm, q, xv)`).

That is the "never dequant" rule paying a second time. Because the weights are
never expanded to signed int8, they never pick up a sign to correct.

### The AVX2 tier has the opposite problem

`VPMADDUBSW` accumulates 8-bit products pairwise **into int16**, and
`255 x 127 x 2 = 64770` overflows it. So the +128 trick cannot be reused on T2
even if you wanted to. `patches/npuxl-0002` uses `|a| * (b * sign(a))`, which
peaks at `127 x 127 x 2 = 32512` and needs no correction because the packing
zeroes the padding. Measured there: **4.3x / 15.2x / 15.7x** over the previous
AVX2 kernel at 1 / 32 / 128 rows.

## 3. Where this build actually lands

Measured here, in-process, by `isa_peak_gmacs()` (16 independent register-only
chains, counted against the MACs those instructions retire):

| | GMAC/s |
|---|---:|
| f32 FMA, 1 thread | 48.5 - 52.3 |
| f32 FMA, 2 threads | 100.6 - 103.1 |
| int8 VNNI, 1 thread | 255.8 - 378.0 |
| int8 VNNI, 2 threads | 600.7 - 651.0 |

The reported value moves because the probe runs on a shared host; the ratio of
int8 to f32 (~6.5x per thread) is the stable part, and it is what a `%peak`
column divides by.

The tiers this build runs on this box: **T4** — AVX-512F + AVX-512BW, and
**no VNNI at all** (`avx512vnni` and `avx_vnni` are both absent from
`/proc/cpuinfo`). So the VNNI kernels in `src/quant.cpp` are compiled and
correct but **never dispatched here**; every int8 dot measured in this repo's
tables goes through the `_avx2_d` twins (`VPMADDUBSW` + `VPMADDWD`), widened to
512 bits where the `avx512bw` target allows. A Meteor Lake laptop at **T3** and
a Sapphire Rapids box at **T7** are strictly better machines for this code than
the one it was measured on, and this is exactly why the tiers are dispatch
targets rather than a compile-time assumption.

## 4. What is missing, stated as work rather than as a wish

* **T3 (`VPDPBUSD`, VEX) is not a separate kernel.** Today an Alder Lake host
  falls from T5 straight to T2, giving up roughly half the int8 rate even though
  it has a VNNI instruction. The dispatcher would need a third `target`
  attribute (`avxvnni`) and a fourth probe. This is the single largest missing
  tier for a laptop fleet.
* **T7 (AMX) does not exist.** It would be a separate kernel, not a widening:
  AMX is tile-based, needs `_tile_loadd` / `_tile_dpbssd` / `_tile_stored` and a
  1 KiB tile configuration, and it would also **remove the bias trick entirely**
  (`TDPBSSD` is signed x signed). It is worth doing only where the machine has
  it, and it must be gated on a verified known-answer test, not on CPUID — the
  rule in `docs/VIRTUAL-DEVICE-PLAYBOOK.md` §4.
* **Tuning knobs from the same guide** — performance governor, C-state depth,
  NUMA placement — are the playbook's C4 scheduling contract, and this sandbox
  exposes none of them (`cpufreq` sysfs is absent). They belong in the capability
  receipt, printed as "not available", rather than assumed.

## 5. What the guide is right about that we already do

The tuning guide's structural advice matches what the last four PRs found
independently: dispatch per ISA tier, pack weights once, use a cache-aware GEMV
for decode and a tiled path for prefill, and treat tile geometry as something to
measure per shape rather than fix. The one place this repo departs from it is
the quantised-weight correction above — and that departure is a property of
keeping weights in their storage format, not a difference of opinion.

## The Q8_0 x8 ladder, and a correction

### VNNI was present on this host all along

Earlier rounds recorded this machine as having **no** VNNI and treated every
VNNI result as unmeasurable. That was wrong. `__builtin_cpu_supports
("avx512vnni")` returns true here; `/proc/cpuinfo` does not list the flag, which
is what the earlier census was reading. The lesson is about method, not about
CPUs: a capability census that reads one source is a guess. The ladder below
reports what the CPU says, and `tools/ta_q80_tiers.cpp` exercises it.

### The ladder

| rung | how it is selected | notes |
|---|---|---|
| VNNI | `avx512vnni` at runtime | `vpdpbusd`: one instruction where BW needs two |
| AVX-512BW | `avx512f` + `avx512bw` | needs an even block count |
| AVX2 | `avx2` at runtime | needs `KRK_DOT_AVX2` in the build |
| scalar | always | the floor; i64 accumulation, no target attribute |

Every rung is guarded by the CPU's own feature bits and every rung was verified
against the scalar reference on this host:

```
nb=24 (even)
  vnni           ok      max rel 8.394e-08
  avx512bw+f16c  ok      max rel 8.394e-08
  avx2           ok      max rel 1.119e-07
  scalar         ok      max rel 0.000e+00  (bit-exact)
nb=25 (odd: 512-bit rungs must refuse)
  vnni           ok      max rel 0.000e+00  (bit-exact)
  avx512bw+f16c  ok      max rel 0.000e+00  (bit-exact)
  avx2           ok      max rel 1.871e-07
  scalar         ok      max rel 0.000e+00  (bit-exact)
```

### Two bugs this found

1. **The AVX2 call was unconditional.** `dot_q8_0_q8_x8` ended in a bare
   `dot_q8_0_q8_x8_avx2(...)`, with no feature test and no scalar floor. On any
   host without AVX2 that is SIGILL, not a slow path — on exactly the class of
   machine (a VM with a masked CPU model, or an old Zen) where a fallback is
   the whole point. Now guarded, with the scalar rung beneath it.
2. **The scalar floor was wrong.** Its first version accumulated each 32-value
   block sum in i32 with `static_cast<i32>(static_cast<i8>(byte))`. With
   products bounded by 128 x 127 and only 32 of them, overflow is impossible,
   yet it disagreed with every vector rung by ~1.1 relative. Widening through an
   explicit unsigned-then-signed step and summing in i64 made all four rungs
   agree to 1.9e-07. A fallback that is silently wrong is worse than one that
   crashes.

### VNNI against BW, measured here

`kraken-q80-gemm-bench 768 2048 512`, four interleaved pairs alternating order
(`KRK_CPU_Q80_KERNEL=bw|vnni`), GMAC/s:

| rep | BW | VNNI |
|---|---:|---:|
| 1 | 14.12 | **15.63** |
| 2 | 15.49 | **22.60** |
| 3 | 22.38 | **26.26** |
| 4 | 18.38 | **22.85** |

4 of 4. Paired ratios 1.11 / 1.46 / 1.17 / 1.24, median **1.21x**. The absolute
spread (14-26 GMAC/s for the same kernel) is this host's two throughput modes,
which is why only the paired ratio is quoted. VNNI is now the default rung on
this machine; `KRK_CPU_Q80_KERNEL=bw` restores the previous kernel and
`=scalar` reaches the floor.
