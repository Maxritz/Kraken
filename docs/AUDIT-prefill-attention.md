# Prefill attention: which kernel the default rung runs, what it costs per (row, key) pair, and what the first fix is

Measured 2026-10-26 on the RX 9070 XT (gfx1201, 64 CU). Every arm table here is
interleaved run by run (arm A, arm B, arm A, ...) and quoted as a median, because
the absolute spread of the same configuration across sessions is 10-30% on this
box. Drivers: `tools/attn_probe.sh` (the engine's own `KRK_TIME=attention` op
row), `tools/attn_res.hip` (the compiler's resource report), and one temporary
decomposition arm described in §4 that is **not in the tree**.

## 1. Verdict

The default rung's prefill attention on Qwen3-8B (hd 128, 36 layers, 501-token
prompt) was **216.6 ms of a 454.3 ms prefill, 36.9%**, and it is now
**152.5 ms** — the tile depth was pinned to one block per CU by an LDS
footprint nobody had looked at, and reading it off a 12 KiB budget is
arithmetic-free (the whole `--prefill-hash` trace is bit-identical across four
tile depths) and worth **−29.8% attention / +25.9% prefill** on that file, −15.1%
at hd 64 and −6.1% at hd 96 and hd 256.

What the kernel is waiting on is **not the KV walk and not the softmax chain**:
halving the staged bytes per pair (QT 16) or doubling them (QT 4) moves the time
by less than ±10% in inconsistent directions, while the occupancy the tile depth
buys moves it 30%. The softmax chain (2 exps + the running max/denominator
update) measures at **zero** — removing it changes nothing. The biggest
remaining single term at a 501-token prompt is **per-block fixed work**: a
three-point fit over prompt length now gives `0.286 ns x (row,key) pair +
1.059 us x block`, and at 72,576 blocks that second term is 77 ms of the 118 ms
(§7). Two later changes moved those coefficients on purpose — the staging index
form halved the pair term (0.557 -> 0.283 ns) and the one-pass row epilogue took
8.6% off the block term (1.159 -> 1.059 us) — and the fit is what says which one
a change landed on.

## 2. Which kernel and tiling the default rung runs

`KRK_ATTN_WMMA=1` or the fast rung (`KRK_GEMM_EXACT=0`, what `--prefill fast`
sets) reaches `attention_wmma_kernel` — the matrix-core kernel — only through
`attention_wmma_ok`, which is `hd == 256 && !kv_int8 && window == 0`. **At every
other head width, and for any window, the fast rung and the exact rung run the
same kernel**, `attention_bexact_kernel`, whose contract is decode's arithmetic
so a chunked prefill writes the bytes a decode step would:

| hd | 64 | 96 | 128 | 160 | 192 | 224 | 256 |
|---|---|---|---|---|---|---|---|
| VPT = hd/32 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| QT (default) | 8 | 8 | 8 | 4 | 4 | 4 | 4 |

Measured on Qwen3-8B (hd 128, 36 calls = 1 per layer, 501 tokens, `--chunk 512`):

| arm | attention | prefill |
|---|---:|---:|
| `attention_bexact_kernel`, old tile 64 | 216.6 ms | 1017-1029 tok/s |
| `attention_bexact_kernel`, tile 8 (new rule) | 152.5 ms | 1295.4 tok/s |
| tiled fallback (`KRK_ATTN_BEXACT=0`) | 928.7 ms | 404.9 tok/s |

The tiled fallback is **4.3x** the old default and **6.1x** the new one, so the
bit-exact formulation is not the slow path here — it is the fast one, and the
matrix-core path is not even selectable below hd 256.

At hd 256 (Qwen3.5-9B, 8 attention layers of 32, 501 tokens) the two are level
at this length: matrix cores 248.3 ms against bit-exact 249.0 ms (and 265.1 ms
for the bit-exact arm at the old tile 32). The matrix-core arm's win is a
long-context one (18% of attention at 4140 tokens, recorded in the GEMM work),
so at 501 tokens there is nothing to pick up there.

## 3. Rate per (row, key) pair

Effective rate, one prompt per file, new tile rule, median of 3 interleaved runs.
Pairs = layers x query heads x sum over rows of the visible keys (causal).

| file | hd | layers | heads | tokens | pairs | attention | ns / pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct-Q8_0 | 64 | 16 | 32 | 502 (1 chunk) | 64.6M | 32.4 ms | 0.50 |
| Phi-3.5-mini (Q4_K_M) | 96 | 32 | 32 | 1101 (3 chunks) | 621.2M | 659.8 ms | 1.06 |
| Qwen3-8B-Q4_K_M | 128 | 36 | 32 | 501 (1 chunk) | 144.9M | 152.5 ms | 1.05 |
| Qwen3.5-9B-Q4_K_M, bit-exact arm | 256 | 32 (8 attn) | 16 | 501 (1 chunk) | 16.1M | 249.0 ms | 15.47 |

The 1B and the 8B are the pair of rows that matters: hd 64 with VPT 2 at 0.50
ns/pair against hd 128 with VPT 4 at 1.05, i.e. **2.1x the cost for 2x the lane
slice**. The per-pair cost tracks the channels per pair, which is what the dot,
the V accumulation and the staged bytes per pair all scale with.

The hd-256 row is the outlier and is **not attributed**. It is also the one row
whose number is contaminated by a per-call term the recorder charges to the op:
on that file even an 11-token prompt reports 48.15 ms of "attention" over 8
launches that do ~100 pairs of arithmetic, where the dense 8B's per-call floor
is 0.43 ms. QT and tile arms do not move it (246.8 ms at QT 4 against 252.0 at
QT 8; 249.0 at the tile rule against 265.1 at tile 32). What is solid is that
this file's whole prefill is 396.9 ms and 244.4 ms of it is reported against
attention, so it is the best first target for its own investigation.

### The per-pair cost is not the whole cost: `0.55 ns/pair + 1.09 us/block`

Three prompt lengths on the 8B, new rule, median of 3 (the 251-token point is a
prediction check, not a fit input):

| tokens | pairs | blocks | measured | fit |
|---:|---:|---:|---:|---:|
| 126 | 9.22M | 18,432 | 25.14 ms | 25.14 ms |
| 251 | 36.43M | 36,864 | 59.21 ms | 60.2 ms |
| 501 | 144.87M | 72,576 | 158.97 ms | 158.97 ms |

Least squares over the outer two points: **0.55 ns per (row, key) pair and
1.09 us per block**, and the middle point lands within 1.7%. Blocks are
`layers x heads x ceil(tokens / QT)`; the per-block term is 79 ms of the 159 ms
at 501 tokens and 20 of the 25 ms at 126 tokens. That is the term to attack for
short prompts, and "short prompt" is most of what a prefill is in a chat turn.

*(Superseded coefficients: §7 re-ran this fit after the staging and epilogue
changes and reads `0.286 ns/pair + 1.059 us/block`. The table above is what the
kernel cost before them and is left as the baseline those changes are quoted
against.)*

## 4. The decomposition, and the two negative results

A temporary arm inside the kernel switched off one component at a time (bit 1
skipped the K/V staging, bit 2 replaced the online-softmax chain with a running
sum, bit 4 dropped the 32-lane reduce to the lane-local partial). It produces
wrong outputs by construction, was compiled in only for these runs, and is
**removed from the tree**; its own control (probe bits 0) reads 228.8 ms against
216.6 ms clean, so the probe's plumbing costs ~6% and every number below is
quoted against the probe's own control.

| 8B, 501 tokens | tile 64 (old) | | tile 8 (new) | |
|---|---:|---:|---:|---:|
| arm | net ms | vs control | net ms | vs control |
| control | 228.8 | — | 171.1 | — |
| no K/V staging | 156.8 | −31% | 119.5 | −30% |
| no 32-lane reduce | 195.4 | −15% | 143.8 | −16% |
| no softmax chain | 229.4 | **0%** | 171.3 | **0%** |
| bare pair loop (all three off) | 121.6 | −47% | 99.1 | −42% |

So of the per-pair loop: the K/V staging is ~30%, the reduce ~15%, the softmax
chain measures at zero, and a bare loop that only walks the guards, the two
`v_dot2` pairs and the V read is still ~55%. **The softmax chain is not the
limit** — two MUFU exps and a running max per pair are free next to everything
else, which is the opposite of what a flash-attention write-up would lead you to
expect.

The KV walk's *volume* is not the limit either. QT decides how many query rows
share one staged tile, so QT 16 halves the staged bytes per pair and QT 4 doubles
them, at the same tile depth:

| 8B, 501 tokens | QT 4 | QT 8 | QT 16 |
|---|---:|---:|---:|
| tile 16 | 190.9 | 156.5-157.6 | 171.8 |
| tile 32 | 219.3 | 181.9 | 173.4 |

±10% at most, and the sign flips with the tile depth: not a kernel waiting on
its KV traffic. (The QT 16 instantiation is also the one the compiler marks down:
134 VGPRs and 13 spilled SGPRs, occupancy 10 waves/SIMD against 16.)

## 5. What the limiter is, with numbers

The tile depth's only effect is the staged K/V pair's share of LDS, and LDS is
what decides how many blocks a CU holds. `attn_bexact_smem_bytes<VPT,QT>(k, hd)`
at hd 128 is `(8 + 2k) * 128 * 2 + 4160` bytes:

| tile k | LDS / block | blocks per 64 KiB CU | measured |
|---:|---:|---:|---:|
| 8 | 10,304 B | 6 | **153.4 ms** |
| 16 | 14,400 B | 4 | 156.5 ms |
| 24 | 18,496 B | 3 | 169.6 ms |
| 32 | 22,592 B | 2 | 181.9 ms |
| 64 | 38,976 B | **1** | 216.6 ms |

One block per CU is 8 warps over 4 SIMDs — **2 warps per SIMD** — for a loop
whose every pair is a serial chain (dot, 5 dependent shuffle steps, 2 exps, the
accumulator update). That is the whole story of the 30%: nothing about the
arithmetic changed, only how many warps are resident to interleave it.

`tools/attn_res.hip` (`-Rpass-analysis=kernel-resource-usage`, gfx1201) reports
VGPRs 41/57/90 for hd 64 at QT 4/8/16, 70 for hd 96 QT 8, 61/85/134 for hd 128 at
QT 4/8/16, and 92 for the matrix-core kernel. The dynamic LDS is reported as 0
(it is `extern __shared__`), which is why the table above is computed from the
kernel's own sizing function rather than read off the pass.

**What is left unexplained and must not be papered over:** even the bare pair
loop runs at ~128 cycles per pair per CU, and an instruction-count model of that
loop (two `v_dot2`, four FMA, two-six LDS reads, a handful of compares) puts it
near 3-5 cycles per pair per CU at 4 IPC. The kernel is stalled by ~25x and this
box has no hardware-counter path on this op, so the stall is unattributed. The
two candidates the numbers point at, in order: the LDS read pattern (the K read
is a `u32` per lane at an 8-byte stride, so half the banks are hit twice) and the
per-block term the length fit isolates in §3.

## 6. What landed

The tile depth is no longer the constant 64 (32 at hd >= 256): it walks down from
that value until the staged pair plus the accumulators' merge buffer fits a
**12 KiB LDS budget**, floored at one warp (8 keys), and `KRK_ATTN_BEXACT_TILE`
still overrides it for A/B. The rule's output:

| hd | 64 | 96 | 128 | 160 | 192 | 224 | 256 |
|---|---|---|---|---|---|---|---|
| tile depth | 32 | 16 | 8 | 8 | 8* | 8* | 8* |
| LDS / block | 11,328 | 10,816 | 10,304 | 11,584 | 13,888 | 16,192 | 18,496 |

\* at hd 192 and up the merge buffer alone (`2 * 8 warps + 8 * 32 * VPT` floats,
8,256 B at VPT 8) leaves no room for a deeper tile than the floor.

Interleaved A/B against the old constant, medians of 3-4 runs:

| file | hd | new ms | old ms | delta |
|---|---:|---:|---:|---:|
| Llama-3.2-1B-Q8_0 | 64 | 32.4 | 38.2 | −15.1% |
| Phi-3.5-mini-Q4_K_M | 96 | 659.8 | 703.0 | −6.1% |
| Qwen3-8B-Q4_K_M | 128 | 152.5 | 217.2 | **−29.8%** |
| Qwen3.5-9B-Q4_K_M (bit-exact arm) | 256 | 249.0 | 265.1 | −6.1% |

Prefill on the 8B: 1017-1029 -> 1295.4 tok/s in the same interleaved session.

### The identity proof, which is what makes this a tuning change and not a new kernel

Every tile depth visits the same keys in the same order per warp (`j % 8`), gives
each row its own `(m, l, acc)` chain, and stages the same values, so nothing about
the arithmetic can move. Measured:

* Qwen3-8B, 501 tokens, `--chunk 8 --prefill-hash`: `b59269247d55ed24` /
  logits `119d51f3cda0cf18` / first token 576 — **identical** from the rule, tile 8,
  tile 32 and tile 64, and from QT 4 and QT 16.
* LFM2.5-24B-A2B (hd 64, so the rule moved its depth 64 -> 32) reproduces the
  pin recorded in AGENTS.md exactly: trace `c19de85c8a06d85d`, logits
  `9fce30dd39c9bf6f`, first token 4741.
* `kraken-bench --bitexact` rc=0: 0 of the (kernel, rows) cases differ from the
  decode row, and int8 K/V is 27 of 27 arms identical.
* The chunk-invariance gate is unchanged: `f034bf9b5feee40e` at chunks 1/8/512,
  `bf32bbb3b9a9d0cf` at chunk 32, `33048d8cf59df95b` for the tiled control, logits
  `e8d8efa95f9d62c9`.
* Suite 4528/4528 in 95 groups, baseline `no regression`, three coherence shapes
  PASS (Qwen3-4B-Q8_0 dense hd 80 -> the tiled path, Qwen3.5-9B-Q4_K_M recurrent
  hd 256, Qwen3-30B-A3B i1-Q2_K MoE).

## 7. What landed next: the staging index form and the single-pass epilogue

Two changes, and the three-length fit separates them because they land on
different coefficients:

* **Staging without a per-element division.** Q and K/V were staged from a flat
element index (`q_sh[i] = q[((t0 + i / hd) * n_head + h) * hd + i % hd]`), which
  pays an i64 divide AND an i64 modulo per staged element. Enumerating (row,
  32-channel chunk) tasks instead makes `task / VPT` a compile-time divide (VPT is
  hd/32), puts every lane on its own contiguous 64 B run, and removes the divide
  from both the Q stage (per block) and the K/V stage (per key per tile).
* **A one-pass per-row epilogue.** A warp publishes its raw `(m, l)` and its
  UNSCALED accumulator; the reader applies `aw = exp(m_w - mstar)` to each
  contribution itself. One publish and **two** barriers per row where the old form
  had two publishes and **three**, and no pre-scaled slot round trip. The slot is
  channel-major (`e` outer, lane inner) on purpose: the old lane-major layout put
  consecutive lanes VPT floats apart, so all 8*VPT LDS reads of the reduce were a
  4-way bank conflict at VPT 4; channel-major puts lane L on bank L. Same LDS
  total, so no occupancy change.

`tools/attn_fit.sh 3`, Qwen3-8B-Q4_K_M, interleaved run by run, medians of 3;
blocks = layers * heads * ceil(tok/QT) at QT 8:

| arm | ns/pair | us/block | 501 tok attention | block term there |
|---|---:|---:|---:|---:|
| base (flat staging, 3-barrier epilogue) | 0.557 | 1.102 | 160.64 ms | 80.01 ms (49.8%) |
| + index-form staging | 0.283 | 1.159 | 125.16 ms | 84.09 ms (67.2%) |
| + one-pass epilogue (shipped) | 0.286 | **1.059** | 118.33 ms | 76.87 ms (65.0%) |

So the staging change is the pair side (0.557 -> 0.283 ns, -49%) and the epilogue
is the block side (1.159 -> 1.059 us, **-8.6%**, 7.2 ms at 501 tokens) with the
pair term untouched, which is exactly the split the fit was built to show.
End to end attention 160.64 -> 118.33 ms (-26.4%) and prefill 1249.5 -> 1417.4
tok/s (+13.4%) on the same session's medians (n=3 at 501 tokens; the staging
change carries 1399.9 of that and the epilogue the last 17.5 tok/s).

**The block term moved 8.6% where this document's own plan predicted 25%, and the
reason is arithmetic.** The block term is ~1.06 us = ~2100 cycles, and removing 8 of the
block's 24 barriers bought 8.6% of it, i.e. ~90 ns a barrier: the barriers are
not what the per-block work is made of. The rest is the Q stage, the row
epilogue's own 8 `aw` exps and 40 mul-adds per row per thread, and the tail. The
next lever is the merge's own volume, not the barrier count.

### The trap: the arithmetic was identical and the hash moved anyway

`--prefill-hash` went from `03e265735b71b0a4` to `17f6d26aeb0ae0d3` at chunk 8
while chunk 1 stayed put and the first token did not move, and the difference was
exactly **1 f16 ulp in 1-2 of 4096 elements per row** (e.g. -0.0007414818 ->
-0.0007410049, which is 2^-21, one f16 ulp at that magnitude). The stage dump
(`KRK_DUMP`, `KRK_DUMP_ROW`) put the first differing stage at `attn.out` with
`attn.q`/`attn.k`/`attn.v` identical, so it was inside the attention op.

The cause was not the numbers, it was their SHAPE. Reading the device ISA
(`tools/attn_isa_probe.hip` compiled with the engine's flags and
`hipcc --save-temps`, then the `.img.lto.s`):

| `attention_bexact_kernel<4,8>` | base | one-pass epilogue |
|---|---:|---:|
| `v_fmac_f32` (FMA) | **0** | **268** |
| `v_dual_add_f32` | 159 | 30 |
| `v_exp_f32` | 32 | 88 |

An FMA keeps the product's low bits; the old form got its rounding for free
because it wrote the scaled slot through shared memory and a store cannot be
fused into anything. **`__fmul_rn`/`__fadd_rn` did not stop the fusion and
neither did `#pragma clang fp contract(off)`** — `-ffast-math` carries
`-ffp-contract=fast` and it fuses `fadd(fmul(x, y), z)` regardless of the
intrinsics naming the two steps. What does stop it is `d_rounded()`, an empty
`asm volatile("" : "+v"(v))` around each product: it is a barrier LLVM will not
reorder or fuse across and it costs no instruction. The ISA after the fix reads 0
`v_fmac_f32` again and the reduction is a tree of `v_dual_add_f32` like base's —
the ADDS are deliberately left free to reassociate, because base's own reduce is
a dual-add tree and not a chain.

### The identity proof

* Qwen3-8B-Q4_K_M, 8-token prompt, `--prefill-hash` at `--chunk 1`, `2`, `8`,
  `16`: `03e265735b71b0a4` / logits `4ccd241994b842b9` / first token 374 — every
  arm identical to the pre-change binary.
* The recorded 501-token pin reproduces exactly: `b59269247d55ed24` / logits
  `119d51f3cda0cf18` / first token 576, at chunk 1 and chunk 8.
* Qwen3.5-9B-Q4_K_M (hd 256, recurrent): identical at chunks 1 and 8.
* `kraken-bench --bitexact` rc=0: 0 of the (kernel, rows) cases differ and int8
  K/V is 27 of 27 arms identical.
* Suite 4528/4528 in 95 groups, baseline `no regression`, `build_check.sh` 0
  failed, coherence PASS on all three shapes (Qwen3-4B-Q8_0 dense,
  Qwen3.5-9B-Q4_K_M recurrent, Qwen3-30B-A3B i1-Q2_K MoE).
* `bench --gate` is green except the speculation arm's missing
  `models/SmolLM2-135M-Instruct.Q4_K_M.gguf`, which fails before any of this.

One measurement that is NOT this change and should not be read as one:
`bitnet-2b-tq1_0` is chunk-variant (chunk 1 `b8b53cf4b44c413b` vs chunk 8
`f70478a8cfefbf4f`) in **all three** binaries, including the one built from HEAD,
so it is a pre-existing path of that model's own.

## 8. The next three, in the order I would take them

1. **The block term, whose remaining cost is the merge's volume, not its
   barriers.** It is still 77 ms of 118 at 501 tokens and the larger share at
   short prompts. Per row each thread now reads all 8 warps' 8*VPT accumulators
   (32 LDS loads) and runs 8 `aw` exps and 40 mul-adds to do it; the `aw` values
   are warp-uniform and computed 32x redundantly per warp. A broadcast (one exp
   per warp per row, shared by shuffle or a tiny LDS slot) is arithmetic-free by
   construction — the same 8 values, read instead of recomputed — and so is
   staging the accumulators in f32 pairs. Criterion: `--prefill-hash` identical,
   `--bitexact` green, and the fit re-run to show the block term move while the
   pair term does not.
2. **The per-pair chain.** The causal bound is tested per (key, row) — 8 compares
   per key at QT 8 — and the 32-lane reduce is 5 dependent steps per pair. A
   per-key bound (`r0 = j - (pos0 + t0)`, loop rows from `r0`) removes 7 of 8
   compares and is arithmetic-free; interleaving the row chains is what gives the
   2-warps-per-SIMD occupancy the ILP it cannot get from more warps. Expected:
   the bare-loop share (55%) is the ceiling. Criterion: hash identity for the
   guard hoist, and for the interleave the same output bytes, since each row's own
   expression is unchanged.
3. **The matrix-core kernel for hd 64-224, on the fast rung only.** It refuses
   every width but 256 today, so the "fast" rung runs the exact kernel at the
   widths most local files use (64, 96, 128), where attention is 30-60% of
   prefill. It cannot be hash-identity — that is why it is wired to the rung and
   not the default — and at 501 tokens it is level with the exact kernel, so the
   win is the long-context one already recorded at hd 256 (18% of attention at
   4140 tokens). Cheap first step: widen `attention_wmma_kernel`'s `kAwDim` to a
   template and measure at hd 128 against the arm table above.
