# The NPU, consolidated: every idea, what is duplicated, and what is actually left

Six sources have been feeding this line of work for months. They have been
written down in six places, in four vocabularies, and at least eight of their
"new ideas" are the same idea. This file is the merge. Everything here is either
measured on this box or quoted from a source that measured it, and the items
marked *refuted* were killed by a number, not an opinion.

Sources, cited by tag throughout:

| tag | source |
|---|---|
| **S1** | your cross-architecture porting table (your laptop): P/E placement 4.26–5.41×, partitioning + persistent pool ~8×, layout pre-packing 1.5–4×, VEX-256 VNNI ~1.3×, streaming/NT hints ≈0, M-blocking −30% |
| **S2** | this repo, `patches/` — four patches carried against npuxl (0001 retired, 0002 AVX2 int8, 0003 force-ISA + scale order, 0004 q8_0 attempt) |
| **S3** | this repo, `docs/`: `NPU-PREFILL.md`, `NPU-KERNELS.md`, `CPU-VS-NPU-BENCHMARK.md`, `MOE-RESIDENCY.md` |
| **S4** | the device itself: `Maxritz/npuxl-improved` — `docs/TUNING.md`, `docs/PERFORMANCE.md`, `docs/ROADMAP.md`, `PATCHES.md`, `include/vnpu.h` |
| **S5** | your `ML-NEW-TECH.md` Part 3 mechanism menu (A–G) and §31 MoE |
| **S6** | measured today, this box: the tables below |

---

## 1. The inventory, deduplicated

`dup` names the other entries that are *the same idea*. `state` is where it
stands **in Kraken's driver**, which is a different question from whether the
device can do it.

### A. Where the work runs

| # | idea | src | dup of | state | measured |
|---|---|---|---|---|---|
| A1 | route GEMMs to a device instead of the CPU | S3 | — | **done** | prefill 196 tok/s device vs 284 host on a dense q8_0 fixture |
| A2 | move the *whole* transformer block to the device (rope/attn/kv too) | S3,S4 | — | partial: rmsnorm/swiglu/add/mul offered, rope/attn/kv stay host | host gap is now the prefill ceiling |
| A3 | GPU as a third unit (`VNPU_UNIT_GPU`, reserved) | S4 | — | n/a here | no ROCm in this sandbox |
| A4 | cluster: shard weights over nodes | S4 | S5 §cluster | unbuilt | device's own demo: 0.86× wall clock — network dominates |

### B. The GEMM kernel itself

| # | idea | src | dup of | state | measured |
|---|---|---|---|---|---|
| B1 | int8 weights × int8 activations on the device | S3,S4 | S1 "VEX-256 VNNI" | **done, on by opt-in** | see §6 — 2.5× decode, but only after a 524 ms pack |
| B2 | AVX2 int8 kernel that is not `mullo_epi32` garbage | S2 (0002), S1 | B1 | **done in our fork, absent upstream** | 4.3×/15.2×/15.7× (1/32/128 rows) |
| B3 | keep weights in their block-quant form, never expand to f32 | S3,S5 "never dequant" | S1 "layout pre-packing" | **unbuilt — needs an opcode** | both forks lack `VNPU_OP_MATMUL_Q8_0`; see §4 |
| B4 | tile geometry per shape | S1 "M-blocking", S4 autotune | — | wired today (`KRK_NPU_TILES`), measured ≈ nothing end-to-end | 69.1–72.6 tok/s across four tile configs = noise |
| B5 | unit choice per phase (Voodoo/Wildcat/Cell) | S4 | S1 "Cell L1 staging" | **done** (`KRK_NPU_UNIT`, default Voodoo) | Cell 65.1 tok/s prefill vs Voodoo 192.8 — 3× *worse*, as S4 predicts |
| B6 | fused epilogues (bias+SiLU+mul) | S4 v1.1 | — | unbuilt | untested; `EPILOGUE_NONE` everywhere today |
| B7 | Wildcat cull mask: skip tiles of inactive experts | S4, S5 §31 | — | **unbuilt — the biggest unbuilt device feature** | no measurement; needs batched-expert GEMM with `tile_mask` |

### C. Memory and residency

| # | idea | src | dup of | state | measured |
|---|---|---|---|---|---|
| C1 | budget the device's expanded-weight cache from real RAM | S3 | — | **done** | 3042 → 2357 ms prefill (1.29×) |
| C2 | pack lazily, not on every first touch | S3 | C1 | **done** | 3395 → 2381 ms (1.43×) |
| C3 | one-time pack should not be inside the measured decode | S6 | C2 | **open — see §7** | 524 ms of an 1083 ms 32-token run |
| C4 | disk-resident experts, streamed | you, S3 | — | done on the host path, unused on the device | — |
| C5 | MMIO/status registers rather than a shim | S4 | — | n/a | cosmetic |

### D. Parallelism

| # | idea | src | dup of | state | measured |
|---|---|---|---|---|---|
| D1 | P/E core placement | S1 | S4 `VNPU_CONFIG_PIN_THREADS` | **host done** (#27), device side unused | 1.00× on this uniform box; inert by design |
| D2 | persistent worker pool, no per-call spawn | S1 | — | **done** (host), device has its own | 0.5 ms spawn vs 64 µs GEMV |
| D3 | overlap host work with device work | you, S3 | — | partial (queue + one flush) | device idle 83%→fixed in #26 |
| D4 | `VNPU_CONFIG_NUMA_AWARE` / `LOW_LATENCY` / `EAGER_SYNC` | S4 | D1 | unbuilt | untested |

### E. Measurement

| # | idea | src | dup of | state | measured |
|---|---|---|---|---|---|
| E1 | per-component device time, ranked | you, S3 | S4 Sherlock | **done, three backends** | the table in §5 is the output |
| E2 | device-side Chrome trace | S4, S2 (0001) | E1 | **duplicate of E1** — we ship our own and never read theirs | 0001 (a real 10 ms-grid bug) is already upstream |
| E3 | roofline `%peak` per row | S3, S4 | E1 | done | **can read >100% — see §8** |
| E4 | autotune and cache the recipe | S4 | B4 | unbuilt (the API exists) | autotune says 1×128×64 for M=1; end-to-end it is noise (B4) |

### F. Formats

| # | idea | src | dup of | state |
|---|---|---|---|---|
| F1 | int4 / grouped int4 with VNNI decode | S4 v1.2, S5 | B3 | unbuilt; S4's own roadmap |
| F2 | ternary / BitNet dot | S5, you | B3 | unbuilt here; the CPU side has the fixtures |
| F3 | pre-quantized activations, no `QUANTIZE_I8` per step | S4 | B1 | partially taken (axis=row), still a command per GEMM |

---

## 2. One idea, eight names

This is the duplication you asked to see. Left column: how a source names it.
Right: what it actually is, and where it already exists.

| named as | actually is | already implemented in |
|---|---|---|
| "Cell-style double-buffered L1 staging" (S1, your strongest unbuilt) | explicit DMA into a per-worker local store | **`VNPU_UNIT_CELL` + `VNPU_CMD_FLAG_CELL_DMA`** — and measured 3× *slower* for dense GEMM (S4 says so too: "Cell is intentionally slower for large dense GEMM") |
| "P/E-core placement 4.26–5.41×" (S1) | pin threads to the fast cluster | **two** copies: `VNPU_CONFIG_PIN_THREADS` (device) and `Pool::setup_placement` (#27, host) |
| "layout pre-packing, pan8 1.5×, 4× over raw" (S1) | store weights in the kernel's own order once | **three** copies: `vnpu_pack_weights_i8` + `trans_b` (device), `Offload::weight_i8` (Kraken), 0002's `[n_block][k4][16][4]` (our patch) |
| "VEX-256 VNNI ~1.3×" (S1) | 256-bit int8 dot | **two**: device's AVX2 path (which 0002 had to fix — it was 1.29× *scalar*) and the VNNI path |
| "M-blocking −30%" (S1) | tile M against cache | device autotune sweeps exactly 32/64/128 M (B4) |
| "streaming / NT hints ≈ 0" (S1) | `movntdq` | nobody has built it, both sides measured ≈0 — **the one item where the two tables agree and it is a non-win** |
| "8x from fusing and tiling" (you, on the NPU) | fewer dispatches, bigger tiles | B4+B6: tiles measured ≈0 here; fusion unbuilt |
| "Sherlock deep dive" (S4, S3) | per-component device time | E1: we already have it, cross-backend |

**What that table means:** of the six items in the porting table, five already
exist inside the device under different names, and the sixth is a measured
non-win on both sides. The porting table is not a list of things to build in
Kraken — it is a list of things the device already does, which is why "optimise
the entire NPU" has to start with *what we ask it for*, not with new kernels.

---

## 3. Two forks of one codebase, and neither is a superset

`riteshnair/npuxl-improved` (what we link) and `Maxritz/npuxl-improved` are
siblings of the same `Maxnpuxl.zip.zip`. Checked today by applying our patch
stack to a fresh clone of the Maxritz tree:

| patch | applies? | meaning |
|---|---|---|
| 0001 trace timestamp | **already applied** | retired; both trees have the fix |
| 0002 AVX2 int8 | **clean** | ours only — `grep maddubs` = 0 hits upstream; this is a 15× kernel that exists in exactly one of the two trees |
| 0003 force-ISA VNNI + scale order | partial (CMake hunk conflicts) | ours only |
| 0004 q8_0 in place | n/a (different paths) | blocked: **no such opcode in either tree** |

So the consolidation is not "switch to the newer fork". Maxritz has the
features (units, autotune, cull, cluster, MTP); ours has the kernels. The
device tree Kraken should build against is **Maxritz + 0002 + 0003 rebased**,
and that is the first item in §9.

The divergence is not academic: building this repo against the unpatched
upstream tree **fails our own test suite** at `test_npu_backend`, because
`q8_0 in place` only exists when the configure probe finds
`VNPU_OP_MATMUL_Q8_0` and upstream has no such opcode — the counter compiles to
zero and the assertion can never pass. The assertion is now gated on that probe
(`#if defined(KRK_VNPU_HAS_Q8_0)`), so the suite is honest against either tree,
and the failure is recorded here as the thing that surfaced the fork split
rather than as a test to delete.

---

## 4. The missing opcode, stated precisely

Both trees expose 17 opcodes. The matmul opcodes are `MATMUL_F32` and
`MATMUL_I8`, and `MATMUL_I8` takes a `vnpu_packed_weights_t*` — a symmetric
per-row int8 pack, **not** a ggml block format. There is no opcode that reads
q8_0/q4_K blocks where they lie.

Consequence, measured: a 92.6 MiB q8_0 model makes the device read **370 MiB of
expanded f32** per decode step. That is the 6.1× staging blowup from #17, and it
is a *format* problem: no amount of tiling fixes it, which is why the tiling
sweep in §6 is noise.

`MATMUL_I8` is the halfway house — it reads int8 (92 MiB, 1×) but quantises the
*activations* too, so it costs accuracy. That trade is measured in §6, and it is
the only format lever that exists today.

---

## 5. What the device offers that we never drove

Read off `include/vnpu.h`, checked against `src/npu/npu_offload.cpp`:

| device capability | used by Kraken before today |
|---|---|
| `VNPU_UNIT_{AUTO,VOODOO,WILDCAT,CELL}` | yes, `KRK_NPU_UNIT`, default Voodoo |
| `vnpu_pack_weights_i8` + `trans_b=1` | yes, `KRK_NPU_W8A8` |
| `VNPU_CONFIG_HIGH_THROUGHPUT`, `TRACE*` | yes |
| `VNPU_CONFIG_PIN_THREADS`, `NUMA_AWARE`, `LOW_LATENCY`, `EAGER_SYNC` | only PIN, and only by env |
| `vnpu_device_set_tiles` | **no** — added today as `KRK_NPU_TILES` |
| `vnpu_autotune_gemm` | **no** |
| `tile_mask` + `VNPU_CMD_FLAG_WILDCAT_CULL` | **no** |
| `epilogue` ≠ NONE | **no** (always NONE) |
| `VNPU_OP_MTP_HEAD_F32` | **no** (speculative head, unused) |
| cluster / node API | **no** |

---

## 6. The levers, measured

Dense llama-shaped fixture, q8_0, 12 layers × 768, ff 2048, vocab 4096, 82-token
prompt, `--ctx 512`, this box (2 cores). Ratios are the signal; absolute tok/s
is indicative (§11).

### Unit selection at n=128

| unit | prefill | decode |
|---|---:|---:|
| Voodoo (default) | 193.5 | 70.4 (i8) |
| Wildcat | 196.6 | 25.4 (f32) |
| Cell | **65.1** | 13.9 (f32) |
| auto | 197.6 | 13.8 (f32) |

Cell costs 3× on prefill and auto picks something that loses 5× on decode.
**Default Voodoo is correct; do not let `auto` near a decode path.**

### Tile geometry at n=128, i8

| tiles (M,K,N) | prefill | decode |
|---|---:|---:|
| device default 64,256,64 | 196.3 | 71.2 |
| 1,128,64 (what autotune picks for M=1) | 174.6 | 69.1 |
| 1,256,64 | 166.9 | 71.2 |
| 32,128,64 (what autotune picks for M=64) | 196.1 | 72.6 |

All four are within noise of each other, while the *standalone* autotuner
reports 1×128×64 as 1.67× faster than the default at M=1. Reconciliation: the
end-to-end decode is not GEMM-bound (see §7), so a 1.67× kernel win on a 40% share
is 1.1× at best and here it is not visible. **Tiles: real standalone, noise in
place. Do not build a tile autotuner until the surrounding overhead is gone.**

### The format lever, by generation length

Same model, same host, i8 (`KRK_NPU_W8A8=1`) against f32, medians of 3:

| n | f32 ms/step | i8 ms/step | i8/f32 |
|---:|---:|---:|---:|
| 16 | 43.2 | 58.4 | 0.74× |
| 32 | 35.6 | 32.6 | 1.09× |
| 64 | 36.2 | 20.5 | 1.76× |
| 128 | 35.7 | **14.2** | **2.51×** |

And at n=128 against the host CPU, same fixture, same prompt:

| arm | prefill | decode |
|---|---:|---:|
| CPU (host, 2 cores) | **284.0** | 58.0 |
| NPU, f32 | 196.5 | 28.0 |
| NPU, i8 | 193.5 | **70.4** |
| NPU+CPU hybrid, i8 | 196.5 | 72.7 |

**This is the first configuration measured in this project where the virtual NPU
beats the host CPU at decode: 72.7 against 58.0, 1.25×.** It is not a new
kernel, a new opcode or a new unit. It is `KRK_NPU_W8A8=1` on a long enough run.

---

## 7. Why the earlier verdict was wrong, and by how much

The previous conclusion — "the NPU cannot win decode; it stages 552 MiB from a
90 MB model and loses to the host at 124 tok/s" — was measured at n≈16.

The pack is **524 ms, one-time**, and it lands inside whichever decode step
first re-requests a weight with a single row:

```
w8a8 2551 int8 (84.0 MiB packed in 524 ms one-time), 169 on f32
```

In a 32-token run that is 48% of the wall clock. In a 128-token run it is 4.1
ms/step of 14.2. **Steady state is ≈10 ms/step ≈ 100 tok/s**, i.e. 1.7× the host
CPU, and the entire difference between "int8 is a regression" and "int8 is a 2.5×
win" is the length of the decode.

The summary line now prints the pack time next to the packed MiB, because
without it the number is indistinguishable from a steady-state regression. That
is the whole defect: a one-time cost reported as if it were recurring.

---

## 8. Two trace defects worth fixing

1. **`%peak` can exceed 100%.** `npu_gemm_f32_rN` reads 95.9–109.3% of the f32
   probe in run after run. A device row over its own measured ceiling means the
   MAC counter and the device span disagree — either the batch's MACs are
   attributed to one row while the time is shared, or the probe under-measures
   the f32 peak. Until it is fixed, no NPU `%peak` in the 90s should be read as
   "at the roofline".
2. **One GEMM counts as two ops.** The i8 path submits
   `[QUANTIZE_I8, MATMUL_I8]` as one batch, and the tracer counts commands, so
   1191 int8 GEMMs print as 2382 rows. The `us/op` column is therefore per
   *command*, not per GEMM — fine for ranking, wrong for comparing against the
   f32 arm, which is one command per GEMM.

---

## 9. The ranked plan

Ordered by measured or derived effect per unit of work, not by novelty.

1. **Rebase 0002 + 0003 (and 0004's opcode, which is §9.4) onto the Maxritz
   tree and build Kraken against that one tree.** Today two forks each hold
   kernels the other lacks: the AVX2 int8 kernel exists in exactly one, and the
   q8_0 in-place opcode exists in exactly the other. Cost: a patch rebase.
   Effect: one tree to test, Zen 3 gets its 15× kernel back, and our own suite
   stops failing on the tree we did not patch.
2. **Pack at load, not at step 2.** The device's one-time 524 ms pack belongs in
   the load phase, where every engine already pays for weight preparation. The
   hook exists (`EngineConfig::warmup` runs a 1-token pass that stages every
   weight; `Offload::weight_i8` can then pack from `stage`). Effect: the
   crossover at n≈25 disappears — i8 wins from the first token — and the load
   line says what it cost. **This is the next change.**
3. **`tile_mask` + `WILDCAT_CULL` for experts.** The only device feature with a
   structural payoff that nothing else can substitute: a batched expert GEMM
   that skips tiles belonging to experts no token routed to. Needs Kraken's MoE
   path to batch experts into one device GEMM (it already batches decode), and
   the cull mask is one byte per tile. Unmeasured — that is the point of doing it.
4. **`VNPU_OP_MATMUL_Q8_0`.** The real fix for the format (B3), and now buildable
   because we have the device source. 370 MiB → 92 MiB of weight traffic per
   token with **no** accuracy cost, unlike `MATMUL_I8`. Cost: a new opcode, a
   kernel, an ABI bump, a parity test. Do it *after* 2 and 3, because those tell
   us what the ceiling actually is.
5. **Device-side thread placement** (`PIN_THREADS`, `NUMA_AWARE`) — one config
   flag, untested on a hybrid host, unmeasurable here.
6. **Fused epilogues** (bias+SiLU). Worth it only after the host gap shrinks;
   today `silu_mul` is 1.0–1.4% of the run.

Not worth building, on the evidence:

* **Tile autotuning** (§6: real standalone, noise in place).
* **Cell unit for dense GEMM** (3× worse, and S4 says why).
* **Streaming/NT stores** (S1 measures ≈0; there is no >L3-resident case here).
* **A second device tracer** (E1 already covers it; 0001's real bug is fixed).
* **Cluster** (S4's own demo: 0.86× wall clock).

---

## 10. Refuted, with the number

| claim | verdict | number |
|---|---|---|
| "the NPU cannot win decode" | **wrong** | i8 n=128: 70.4–72.7 tok/s against the host's 58.0 |
| "batching/fusing the NPU will give 8×" | not supported | the kernel is at 95–109% of the device's own f32 probe *before* any fusion; the loss was in the driver, not the kernel |
| "the unpack is unavoidable" (mine, retracted) | wrong | `MATMUL_I8` avoids it today; a q8_0 opcode would avoid it without the accuracy cost |
| "M-blocking is worth 30%" (S1) | laptop-specific | here it is noise; the 4 MB working set fits L3, as S1 itself notes |
| "Cell-style staging is the strongest unbuilt idea" (S1) | already built, and slower for this workload | 65.1 vs 192.8 tok/s prefill |

---

## 11. Method

One process per arm, warmed page cache, medians of 3 (5 for the n-sweep
endpoints), `--greedy`, identical prompt and file. This box is 2 cores at 2.6
GHz with no turbo control and a shared neighbour: **absolute tok/s is
indicative, ratios are the result.** The fixture is generated by
`tools/make_bench_gguf.py` (seeded, byte-reproducible) rather than downloaded,
so a number can be attributed to code and not to weights; its weights are
random, so no accuracy claim is made from it — the i8 arm's activation
quantisation was checked for *ranking* drift only (f32 device path: 12/12
identical top-8 sets against the CPU; i8: 1/12 on near-tied random logits, which
needs a real model to judge).
