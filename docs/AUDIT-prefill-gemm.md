# Prefill: where the time goes, and what the staging overlap bought

Investigation on 2026-10-26, RX 9070 XT (`gfx1201`, RDNA4, 64 CU, 15.9 GiB),
HIP 7.12.60602 runtime, Release build. Every number here was measured in this
session on the files named. The workload is a repeated-text prompt
(`"The capital of France is "` twice per 12 tokens) run with `-n 1`, so what is
being measured is prefill and nothing after it.

Roofline used for comparison: **585 GB/s** measured read bandwidth for this card
(the repository's own figure, against 644 GB/s on the spec sheet) and
**about 97 TFLOP/s** for a 64 CU RDNA4 part at f16 WMMA. The second is a
spec-sheet figure and is used as an order of magnitude, not as a target.

Arithmetic below is my own count from the geometry of each file (2 FLOP per
multiply-add), not a counter the engine reports.

## 1. Baseline: prefill is the GEMM and the attention, in that order

`KRK_TIME=1` attributes device time per op, net of a ~20 us recorder floor. The
logical `MB` and `GB/s` columns are the recorder's own tensor-byte counts, so
they are bytes per pass, not physical traffic.

| model (dtype) | prompt | prefill | `gemm(wmma)` | `attention` | embed + rest |
|---|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | 501 tok (1 chunk) | 570.5 ms | **375.1 ms (66%)** | 164.0 ms (29%) | 31 ms |
| Qwen2.5-Coder-7B-Q5_K_M | 601 tok (2 chunks) | 668.2 ms | **461.2 ms (69%)** | 160.9 ms (24%) | 46 ms |
| bitnet-2b-tq1_0 (ternary) | 602 tok (2 chunks) | 428.5 ms | **251.3 ms (59%)** | 139.5 ms (33%) | 38 ms |

The first row is the 501-token workload every number in this document is
compared against; the other two were taken at 601/602 tokens earlier in the same
session, so their absolute times are not comparable row to row and their split
percentages are.

Converted to what the hardware can do, on the 501-token Q4_K case (one chunk,
four 128-row M groups):

| quantity | measured | card | share |
|---|---|---|---|
| weight bytes, physical (4 passes over 4.14 GB) | 16.6 GB in 375.1 ms = **44.3 GB/s** | 585 GB/s | 7.6% |
| arithmetic | 6.69 TFLOP in 375.1 ms = **17.8 TFLOP/s** | ~97 TFLOP/s | 18% |

So the GEMM runs at about a fifth of the card's matrix-core peak and under a
tenth of its read bandwidth. **Neither roofline is the binding constraint**, and
attention is further out still: ~74 GFLOP of causal attention in 164 ms is
**450 GFLOP/s, about 0.5% of the card's f16 peak**.

## 2. What the kernel is, in its own terms

The shipped prefill config is `wmma_cfg_n2` (`KRK_WMMA_NF=2`, the default):
BM 64 rows x BN 64 channels, K tile 128, MB 2 M tiles per block, NF 2 N
fragments per wavefront, 8 warps. Per K tile it does: stage every M tile's
activations, stage the dequantized weight tile, **barrier**, run the MMAs,
**barrier**. One LDS buffer set, so the two phases cannot overlap, and the
block's whole job is a chain of 32 such rounds for K=4096.

Measured resources for that instantiation (`tools/gemm_res.hip`, compiled with
`-Rpass-analysis=kernel-resource-usage`):

| instantiation | VGPRs | spills | LDS | occupancy (VGPR-limited) |
|---|---:|---:|---:|---:|
| `wmma_cfg_n2` MODE 0 (shipped) | 100 -> **143** | 0 | 52224 B | 4 waves/SIMD |
| `wmma_cfg_n2` MODE 1 (stage-only arm) | **0** | 0 | **0** | 16 |
| `wmma_cfg_n2` MODE 2 (mma-only arm) | **0** | 0 | **0** | 16 |
| `wmma_cfg` MB=1 (26 kB config) | 70 -> 80 | 0 | 26112 B | 10 waves/SIMD |
| `wmma_cfg` MB=2 | 81 | 0 | 34816 B | 6 waves/SIMD |

The LDS figure is the one that decides occupancy: 52224 B of a 65536 B budget
means **one block per CU**, i.e. 8 warps over 4 SIMDs, 2 warps per SIMD. The
VGPR count allows 4 waves/SIMD and is not the limit; there are no spills.

## 3. The staging-vs-MMA arms were measuring nothing, and that is withdrawn

The first revision of this document reported "staging 124.5 ms + MMA 123.4 ms
against 370.2 ms of kernel, so +49% that neither half explains". **That claim is
withdrawn.** The two arms compile to kernels with `LDS Size [bytes/block]: 0`
and `VGPRs: 0`: dropping the MMAs makes the staged tiles dead stores that LLVM
eliminates, and dropping the staging leaves the fragment loads reading LDS that
nothing writes, which the MMA then depends on and the compiler folds away too.
Both arms launch an empty kernel.

What they measured was the recorder's *queue* cost: an empty launch still gets
its device span from the surrounding events, so 252 empty launches report
124.5 ms. Their prefill wall does drop (322.8 ms against 565.6 ms) because the
GEMM's work is genuinely gone, but that is not a decomposition and must not be
used as one. This is the same trap the repository records elsewhere: a
diagnostic arm that cannot fail is not a measurement. A split arm has to keep
its result alive (fold the staged bytes into the accumulator through a real
read, and keep an MMA whose operand is real) or be judged by the ISA, not by the
clock.

## 4. The configs that fit are all worse than the shipped one

Interleaved, 3 runs each, 501-token prompt, Qwen3-8B-Q4_K_M
(`tools/prefill_probe.sh`):

| arm | config | LDS | prefill |
|---|---|---:|---:|
| `KRK_WMMA_NF=1 KRK_WMMA_MB=1` | BM 32, MB 1 | 26112 B | 696.4 tok/s |
| `KRK_WMMA_NF=1 KRK_WMMA_MB=2` | BM 32, MB 2 | 34816 B | 776.2 tok/s |
| `KRK_WMMA_NF=1 KRK_WMMA_MB=4` | BM 32, MB 4 | 52224 B | 770.1 tok/s |
| `KRK_WMMA_NF=2` (default) | BM 64, MB 2, NF 2 | 52224 B | **816.6 tok/s** |

Two things follow. **The 26 kB config that fits two blocks per CU is the
slowest of the four**, so a second resident block is not what this kernel wants:
what it wants is fewer staging rounds per output row (32 rows per staged weight
tile at MB=1/BM=32 against 128 at the shipped map) and the NF=2 fragment pair.
And a trap: `KRK_WMMA_MB` has no effect while `KRK_WMMA_NF=2` (the dispatch
reads `wmma_cfg_n2::MB`), so an A/B of "MB=1 against the default" with NF left
at 2 measures the same kernel twice. The sweep above sets NF=1 for that reason.

## 5. The one thing that was exposed, and the fix that closed it

The weight bytes for round kt+1 are already in flight during round kt -- the
Q4_K staging keeps the current and next chunk's raw bytes in registers and
issues the next round's loads before decoding the current one. The **activation
tile had no such prefetch**: its 8 loads per thread were issued and consumed by
its own LDS stores inside one phase, and with 2 warps per SIMD both warps of a
SIMD stall on them together. That latency sat in front of the barrier with
nothing to hide it.

`gemm_wmma_kernel` now holds this thread's activation rows in registers for one
round: `a_prefetch(kt+1)` runs inside round kt's staging phase (before the
weight decode), and `a_publish()` stores them to the tile at the top of round
kt+1. The loads are then in flight across the weight decode, the barrier and the
whole MMA phase. Same rows, same columns, same 128-bit stores, one round
earlier, and `x` is immutable during the kernel, so the staged tile is
bit-identical by construction; the check is section 6.

Measured on the published binaries (two builds of the same tree, the only
difference being this change -- an env switch would have kept both code paths
alive in one binary and differed in register allocation too):

| model | prompt | prefill before | prefill after | change |
|---|---|---:|---:|---:|
| Qwen3-8B-Q4_K_M | 501 tok | 830.9 tok/s | **1013.7 tok/s** | **+22.0%** |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M | 501 tok | 881.1 tok/s | **1073.0 tok/s** | **+21.8%** |
| bitnet-2b-tq1_0 | 502 tok | 1577.6 tok/s | **1752.6 tok/s** | **+11.1%** |

Medians of three interleaved runs each. The same change in the op table, one
run per arm on the 8B:

| op | before | after | change |
|---|---:|---:|---:|
| `gemm(wmma)`, 252 calls | 375.14 ms | **258.86 ms** | **-31.0%** |
| `attention`, 36 calls | 163.99 ms | 170.42 ms | noise |
| prefill wall | 570.5 ms | **459.9 ms** | -19.4% |

Decode is untouched, as it must be (it runs the `gemv_m` ladder, not the tile):
40 tokens on the 8B, 77.1 / 76.4 / 77.0 tok/s before against 77.0 / 77.2 / 77.3
after. The registers cost 43 VGPRs (100 -> 143, no spills) and cost no
occupancy, because LDS already pins the block to one per CU.

## 6. Verification

* `kraken-bench --bitexact`: the verdict lines are identical between the two
  binaries -- every `gemm` / `group+silu` row at rows 2..64 IDENTICAL, every
  attention row, and the matrix-core rows. The only lines that differ in the
  whole output are timing lines (launch floor, pool primitives, the single-row
  bench entry).
* Greedy 24 tokens on all three files: byte-identical text
  (`f7d6858a2618`, `28fc24a953db`, `e3b0dc772006` on both binaries).
* `scripts/coherence_check.sh`: device against `--cpu` catches a wrong kernel,
  so it is the end-to-end check that matters. Qwen3-4B-Q8_0 (dense) and
  Qwen3.5-9B-Q4_K_M (recurrent) PASS, `2 coherent, 0 not`; Qwen3-30B-A3B i1-Q2_K
  (MoE, `KRK_ALLOW_BIG=1`) PASS, `1 coherent, 0 not`.
* `scripts/build_check.sh` 0 failed, `kraken-tests` 4528/4528 in 95 groups,
  `scripts/kraken_tests_baseline_check.sh` `no regression`, `kraken-bench
  --gate` green except the speculation arm, which cannot find
  `models/SmolLM2-135M-Instruct.Q4_K_M.gguf` on this box.

## 7. What is left, and why it is not this change

The staging phase still contains the weight decode (~100 ALU ops per thread per
round) and its 16 LDS stores, and the MMAs still cannot start until the barrier
publishes them. Removing *that* barrier needs two LDS buffer sets, and the
arithmetic says the shipped tile cannot have one:

```
LDS per buffer set = (MB*BM + BN) * (BK + LD_PAD) * 2 B
shipped (BM 64, BN 64, MB 2, BK 128, pad 8) = (128 + 64) * 136 * 2 = 52224 B
two sets                                     = 104448 B, against a 65536 B cap
```

The padding is not negotiable either: at LD_PAD=0 both fragment loads put all 16
lanes of a half-warp in one bank (row stride 128 halves is 0 mod 32 words), a
16-way conflict. The variants that do fit are all worse on section 4's evidence:

* BK 64 per stage with two buffers is 55296 B, but a 64-deep weight stage holds
  `BN * (BK/32) = 128` chunks for 256 threads, so either half the threads idle
  or each thread stages half a chunk and the 16-byte Q4_K header is loaded twice
  as often.
* A double buffer at BK 64 with BN 128 and MB 1 keeps the one-chunk-per-thread
  map at 46080 B, and moves to the config family section 4 measured worst.
* Taking the activation tile out of LDS entirely (fragments straight from L2)
  would leave 17408 B, i.e. up to two blocks per CU, but it doubles the A read
  per block and puts an L2 latency into the MMA's own dependency chain.

The other open item is attention: 29% of prefill at 0.5% of peak, untouched by
this change, and now the larger of the two remaining pieces.

## 8. Not investigated here

* The one-row head GEMV: on bitnet-2b the final row's head readback measured
  121.4 ms of host time against 656.7 MB of weights (5.4 GB/s). If that is a real
  kernel rate rather than a queue wait, it is a decode-side cost of the same size
  as a whole prefill.
* Weight upload at load: 1.02 GiB at 1.84 GB/s (bitnet) and 5.19 GiB in about
  2.8 s (7B). A separate phase, with the same suspicion.
* The MoE rows in the model sweep are not a measure of this change: their
  prefill is dominated by expert bring-in (laguna-xs2 82.0 tok/s against 92.8 in
  the pre-recorded sweep, LFM2.5 427.6 against 410.3 -- the second is noise
  either way), and the two files whose rows stream 12-32 GiB vary run to run.
