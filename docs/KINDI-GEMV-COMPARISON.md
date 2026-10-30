# What kindi's GEMV does, and what of it applies here

Source read: [Maxritz/kindi-AMDx](https://github.com/Maxritz/kindi-AMDx), the
AMD fork of [ahtx/kindi](https://github.com/ahtx/kindi) by Amir H — `kindi/kindi.cu`
(`k_g4`), `kindi/layout.py` (`repack`, `scales16`), `kindi/kcpu.c` (`rows8`,
`gemv`), and `docs/HOW-IT-WORKS.md`. Read 2026-10-30, at the commit the fork
pointed to that day. Nothing here is inferred from their benchmarks beyond what
their own documents state.

## The design in one paragraph

Kindi is GEMV-centric in the literal sense: every matrix multiply in the forward
pass is a matvec with a layout built for that one kernel. The weights are ternary,
so a byte holds four weights (`t0 + 3t1 + 9t2 + 27t3`, 81 values), and the kernel
**never unpacks them**: per token it precomputes the 81 possible signed sums of
the four corresponding activation values, and then a weight byte is only an index
into that table. The rest of the kernel exists to make that index cheap: per-lane
tables so shared-memory lookups cannot bank-conflict, a 256-byte entry stride so
one `__byte_perm` produces the address, four rows interleaved per 16 bytes so each
lane issues one 128-bit load per four rows, the fp16 scales stored contiguously so
a warp fetches all it needs in one coalesced 128-byte load, and the scale applied
*after* the first three butterfly steps so it lands once per two rows instead of
once per row.

On the CPU side (`kcpu.c`) the same idea appears in AVX2 form: rows sit in SIMD
lanes, a per-column-pair table of the 8 nonzero pair sums is rebuilt per token,
and `vpermps` does the lookup — "one vpermps per 8 rows x 2 columns". Its comment
states the property that matters: *"the 8 rows' scales are contiguous ... so a row's
result needs no horizontal sum."*

## What transfers, and what it is worth here

Measured on this repo's q8_0 kernel, not assumed. The tool is
`tools/bench_gemv_scale.cpp` (`kraken-gemv-scale`), 71.3 MB of weights — larger
than the 54 MB L3 — n_in 4096, medians of three, interleaved, same process.

| idea | applies to q8_0? | measured here |
|---|---|---|
| scales contiguous per row-group, read with one wide load | **yes** | 1.08x quiet / 1.19x busy — see below |
| scale applied after a partial reduction | yes, already does | the scale multiplies a 16-lane int32 vector, not a scalar |
| prefetch the next block | yes | **shipped** in the decode GEMV, 137 -> 121 us/op |
| fuse matrices that share an input (q/k/v, gate/up) | yes | not done here; each projection is its own `par_for` |
| one parallel region per token | yes | this backend has a persistent pool with static splits, close enough |
| 81-entry LUT instead of unpacking | **no** | ternary-specific: q8_0 weights are 8-bit, and `vpdpbusd` already does 64 MACs per instruction |
| per-lane shared-memory tables | no | a GPU bank-conflict fix with no CPU analogue |
| column split with ordered partial sums | yes, unused | relevant when a GEMV has too few rows to fill the machine |
| clustered shortlist head | no | for a 248k-row vocab on a GPU; not our bottleneck |

### The scale-contiguity experiment, in full

`V0` is the shipped kernel; `V1` hoists the scale vector out of the loop entirely
and is **not implementable** — it is the hard bound on any amount of scale work
ever removed; `V2`/`V3` are the prebuilt table in f32 and f16.

| | busy host (GB/s) | quiet host (GB/s) |
|---|---:|---:|
| V0 shipped | 14.0-16.4 | 19.7-24.7 |
| V1 scale hoisted (bound) | 19.1-22.5 | 21.3-27.2 |
| V2 prebuilt f32 table | 16.8-18.5 | 20.4-24.3 |
| V3 prebuilt f16 table | 17.7-19.0 | 21.5-25.6 |

Two conclusions, both of which cost me a claim:

1. **The entire scale build is worth at most 1.4x, and a realistic table 1.19x on
   a busy host, 1.08x on a quiet one.** This project previously attributed ~49%
   of the decode GEMV to the scale path. That was wrong: the ablation removed the
   int-to-float convert and the multiply-add along with the scale build and
   credited all three to the scale.
2. **On a quiet host the kernel is at the memory floor.** With scale work gone it
   runs at ~27 GB/s against a measured 25.5 GB/s pure stream read of the same
   bytes. There is no large win left inside this kernel at all.

## The format claim, measured instead of argued

An earlier version of this section said kindi's speed "comes mostly from the
format, not the kernel": 1.58-bit ternary is ~5.4x fewer bytes than q8_0, so a
5.4x decode win. **That is wrong, and measurably so** — ternary is more taxing
per MAC on a CPU, and a ternary kernel stops being memory-bound long before it
stops being slow.

`tools/bench_ternary_gemv.cpp` (`kraken-ternary-gemv`) runs three decode GEMV
kernels, each reading its own natural format, same matrix (4096 x 16384 = 67.1M
weights), same machine. Five runs, medians:

| kernel | G-weights/s | bytes/s actually moved | bound by |
|---|---:|---:|---|
| **Q0** q8_0, `vpdpbusd`, 8 rows in flight (ships here) | **18.3** | 19-24 GB/s | **memory** |
| **T0** ternary LUT, `vpermps` — kindi's CPU shape (`kcpu.c` `rows8`) | **17.2** | 5.6 GB/s | compute |
| **T1** ternary bit-plane, `kmask -> vpmovm2b -> vpdpbusd` | **36.0** | 11.5 GB/s | compute |

Read the byte rate, not the weight rate. With 4.25x fewer bytes, a ternary kernel
that were still memory-bound would run at ~4x the G-weights/s. T1 runs at 2x and
T0 at **0.94x — slower than q8_0**. Neither reaches the byte ratio, because at
0.25 B/weight the loop is limited by instructions and the DRAM line is never the
constraint.

Why the LUT is not a CPU win, concretely: kindi's `rows8` spends about 14 uops
per 32 weights (two `vpsrlvd`, two `vpand`, two `vpermps`, the zero-mask
`vpslld`/`vpsrad`/`vpandn` pair, two adds) — roughly **2.3 weights per
instruction**. `vpdpbusd` does **64 MACs in one instruction**, and even with this
engine's sign-flip and scale scaffolding the q8_0 kernel is at ~10 weights per
instruction. On a GPU the LUT answers a real problem (no cheap int8 dot, and
unpacking base-3 is worse); on a CPU with VNNI it is strictly the wrong tool.

The VNNI-native ternary decode is the better ternary kernel and it is still
compute-bound, at ~36 G-weights/s against a memory roofline that would allow
~85. So the honest number for the format on this box is **~2x in decode, not
4.25x**, and only if you also change the kernel.

### Where the format does pay — and it is not the kernel

The byte advantage is real; it just does not show up as tokens/s on a machine
with RAM. It shows up when the bytes have to come from somewhere:

* **Decode at the stated target is not bandwidth-limited at all.** 7.5 tok/s on
  3.3 GMAC/token is 24.8 GMAC/s; at q8_0's 1.0625 B/weight that is **26 GB/s**,
  which a 4-core box already has. Ternary would need 6 GB/s. Neither is the
  obstacle.
* **Prefill is not bandwidth-limited either**, because a batched GEMM reuses each
  weight across every row in the tile. That is exactly why the prefill GEMM here
  is compute-bound at 76% of its VNNI mix ceiling and not at a byte wall.
* **Residency is where it pays.** A 100+ GB model on a 32 GB machine, or a
  2-4 hot-expert set, has to move weights from disk. There, 4.25x fewer bytes is
  4.25x less I/O, and no kernel change is needed to collect it.

So the transferable lesson is the reverse of the one I wrote: a smaller format
buys residency and I/O, and it buys decode throughput **only** if the CPU kernel
for it costs no more per MAC than the one it replaces — which, measured, is not
true for the LUT and only half-true for the bit-plane form.

## What I would take from kindi next, in order

1. **A load-time repack that separates the scale fields from the 32-byte quant
   blocks** (`layout.py:repack` is exactly this). It buys the 1.19x above *and*
   removes the two-load weight assembly, because the 64 weight bytes become one
   512-bit load. Blocked here on lifetime: `src/model.cpp` streams some tensors
   through a ring, so the table has to hang off the resident slot, not the
   descriptor. That is the next piece of work, not a patch.
2. **Fusing matrices that share an input.** `gemv(m, 2, {&L->qkv, &L->gate})` is
   one call in kindi; here each projection is a separate `par_for` with its own
   team sync. Cheap, safe, and it matters on a 2-thread box.
3. **Nothing else.** The LUT, the per-lane tables and the clustered head are
   answers to GPU-specific problems, and copying them here would add machinery
   without moving a byte.
