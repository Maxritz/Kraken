# Laguna-XS-2.1-IQ3_XXS.gguf decodes all-NaN: L38 / expert 161's activation
# leaves the f16 range (785411 against a 65504 ceiling)

Traced 2026-10-09 on the box (RX 9070 XT, gfx1201, HIP 7.16), following the
`kraken-inspect`-says-`runnable` finding in `AGENTS.md`. The defect is **not**
the file, **not** the dequantizer and **not** the tokenizer: it is one expert's
down projection on the **device** path, and the cause is a **range** defect, not
an addressing one: the activation that feeds that projection is 785411 in f32
(the CPU reference computes it) against f16's 65504 ceiling, so every f16 buffer
on the path overflows to `inf` and the final norm turns that into NaN. This is
the localization, the causes that are excluded and the cause it was traced to,
each with the command that settles it. A bounded mitigation is in the tree (see
"What is in the tree now"); it does not make this file decode.

## The symptom

    kraken --model Laguna-XS-2.1-IQ3_XXS.gguf -p "The history of computing is a history of" \
           -n 2 --greedy --ctx 512 --debug-topk 2
    step 0 pos 9  nan=100352 inf=0 top1: 0=-nan(ind)(-nan(ind))

`nan=100352` is the whole vocabulary, at every step, so greedy decode emits
`〈|UNK|〉`. The same command on `laguna-xs2-Q4_K_M.gguf` is `nan=0` with sane
logits, and `--tokens` gives 9 sensible ids for both files.

## Localization, in the order it was measured

**1. The single-row trace named a stage, and was misleading.** `KRK_DUMP` prints
the *last row* of each stage, and the first line with a NaN is `attn.out L39`
with L37/L38 clean. Read alone that says "layer 39's attention is broken" -- and
it is wrong. The engine's own comment on `dump_row` says exactly this ("finite q
and k on the printed row, all-NaN attn.out ... that is how the nemotron NaN
hid").

**2. `KRK_DUMP_FULL=1` -- the whole tensor, every row -- moves it 40 layers
back.** The first stage with a non-finite value is:

    moe.sum L38 N=9 W=2048 nf=2048 badrows=1 first=0

i.e. **one token row** of layer 38's MoE output (2048 = the row width). Every
later line is cascade, including `attn.out L39 nf=73728 badrows=9`: all nine
cached key rows are poisoned, which is why layer 39 looked like the culprit.

**3. `KRK_DUMP_MOE=1` names the expert:**

    [moeout] L38 e=161 m=  3 out_nf=2048 big=0

Layer **38**, expert **161**, down-projection output, `out_nf=2048` = the full
row non-finite, `big=0` = not merely large (the predicate counts `v > 3.0e38` as
non-finite, so these are NaN/inf, not overflow-sized finite numbers).

**4. The probe's other counters are silent, and that is the key fact.** With
`KRK_DUMP_MOE=1` the engine takes the unfused path and reports `in_nf`, `up_nf`
and `gate_nf` for the same expert group; the `[moe]` line prints only when one is
non-zero, and no such line appears. So for `(L38, e=161)`:

| measured | value |
|---|---|
| gathered input row (`ws_xg_`) | finite |
| up projection (`ws_upg_`) | finite |
| gate projection (`ws_gateg_`) | finite |
| **down projection output (`ws_x2_`)** | **one row of 2048, non-finite** |

Between those last two lies only copy / sigmoid / mul / `scale_act` and the down
GEMM.

## Causes excluded by measurement

| excluded | why |
|---|---|
| **The dequantizers** | `--cpu` on the *same file* gives `nan=0` and sane logits (`top2: 5899=16.2, 18141=14.9`). The host decode of IQ3_XXS / IQ2_S / IQ3_S is correct. |
| **The device dequant twin** | `dequant_chunk_iq3_xxs` (`src/hip/kernels/dequant_fmt_a.inc:97`) indexes `qs = b + 2 + 8*chunk`, `aux32 = d_u32(b + 2 + 64 + 4*chunk)`, the same `db`, sign table and grid table as the host `deq_iq3_xxs` (`src/quant_fmt_a.inc:138`). Read side by side; the arithmetic is identical. |
| **The file's bytes** | L38/L39 norms are F32, `nan=0 inf=0`, sane ranges (`blk.39.attn_q_norm` max 2.28). The suite already pins these types' layout hard (`tests/test_kraken.cpp:1281-1436`). |
| **The device's row/block addressing for the down tensor** | `KRK_DEQUANT_PROBE=blk.38.ffn_down_exps.weight:161` reads that tensor's rows (512 values wide = 2 IQ3_XXS blocks each) through the device path and through the host path from the same bytes: all finite, worst relative difference **1.1e-4**, and the row span the probe computes matches the file's own `n_bytes`. Bytes, address and dequantizer are identical on both sides. |
| **An f16-range overflow in the weights** | `tools/iq_scale_probe.py` on `blk.38.ffn_down_exps.weight`: expert 161's blocks have `max\|d\| = 0.0022`, implying values <= 4.4 -- four orders of magnitude below f16's 65504. (Neighbours 160/162: 0.87 / 0.97.) |
| **The rows>1 / WMMA path** | `--chunk 1`, so every forward is a single row and `m=1` per expert, still `nan=100352`. |
| **The query-tiled prefill kernel** | `KRK_ATTN_QTILE=0` still `nan=100352`. |
| **Nondeterminism** | Identical `nan=100352` at every step, chunk size and run. |

## Root cause (measured): the activation leaves the f16 range

With the weights exonerated, three lines settle it -- the first is the f32
reference the same probe prints under `--cpu`:

    cpu:     [moeact] L38 e=161 m=  1 act_nf=0 act_peak=785411
    device:  [moeact] L38 e=161 m=  1 act_nf=1 act_peak=13232
    device:  [moeout] L38 e=161 m=  1 out_nf=2048 big=0

`act_peak` is the buffer the silu wrote, i.e. the down GEMM's input. The model
legitimately produces **785411** there -- the f32 reference computes it and is
fine -- which is **12x f16's 65504**. So the chain is arithmetic, not addressing:
an activation above the f16 range becomes `inf` in the f16 activation buffer, the
down GEMM's f16 output store spreads it to the whole row (`out_nf=2048`), and the
final norm ends the run with `inf - inf = NaN` over the entire vocabulary
(`nan=100352`) -- fluent-looking `〈|UNK|〉` output.

The reference never sees this for one reason: **ggml's quantized matmul emits
f32**, and the activations in that graph are f32 too, so 785411 is carried
without loss. On this side the activation buffer is f16 by design (it is the
bandwidth trade every kernel here is written to), and an f16 buffer cannot carry
this model's activation. Two consequences: the up/gate projections are f16 as
well, so the same expert's `up` can overflow *before* the silu runs, and no
clamp is a cure -- the value itself is out of range.

A shape note stays useful: the down tensor's row is 512 values (**2** IQ3_XXS
blocks) where up/gate are 2048 (**8**), so a candidate for the next instrument is
that orientation's block addressing. It is not the defect here -- the probe above
reads those very rows identically on both sides -- but it is the shape a
512-wide row makes easy to get wrong.

### What is in the tree now

`d_sat_f16` (`src/hip/krk_hip.hpp`) saturates at the f16 range instead of
overflowing, and the two sites that store a silu **product** use it
(`src/hip/kernels/fused.hpp:167`, the fused gate/up epilogue, and
`src/hip/kernels/normalize.hpp:258`, the standalone op). It is a **mitigation,
not a fix**:

* it removes the inf from the down GEMM's input (`act_nf` 1 -> 0, `act_peak`
  65504) and cuts the damage from a whole non-finite row to a single element;
* it is **inert wherever nothing leaves the f16 range**, which is every model
  measured on this box -- Qwen3-MoE-4x0.6B runs 3638 activation buffers with a
  peak of 3622 and zero non-finite;
* it does **not** make this file decode. The saturated 65504 still overflows the
down projection's own f16 output store, the logits are still all-NaN and the text
is still `〈|UNK|〉`.

### What landed: the per-row power-of-two scale

`silu_mul` now takes `(i64 row_len, f32 *row_scale)` and `scatter_axpy_rows` a
nullable `row_scale` (`include/krk/backend.hpp`). The HIP backend routes the row
case to `silu_mul_scaled_kernel` (`src/hip/kernels/normalize.hpp`), which finds
each row maximum, stores `product / 2^k` with the smallest `k` that fits, and
writes the multiplier to `row_scale[row]`; the scatter multiplies it into
`alpha[i]`. The engine keeps one f32 per routed row in `ws_scale_` and hands it to
both calls, so the multiplier never leaves the device and no host sync is added.
A row is one token`ff_exp`-wide expert activation, which is exactly the dimension
the down projection reduces over -- which is why the scatter can undo the scale
with the per-row weight it already applies.

Measured on this file, prefill, `--chunk 256`, `KRK_DUMP_MOE=1`:

| line | before | after |
|---|---|---|
| `[moeact] L38 e=161` | `act_nf=1 act_peak=13232` (the inf) | `act_nf=0 act_peak=49088` |
| `[moeout] L38 e=161` | `out_nf=2048` (the whole row) | no line: nothing non-finite |

49088 is 785411/16 exactly -- the row maximum was 11.99x the ceiling, so `k=4`
-- and a power-of-two divide is exact in f16, so the stored row kept its
mantissa. Both stages it aims at are clean, and the down projection own f16
store no longer overflows because its input, and therefore its output, is scaled
down with it.

**In range the change is bit-identical**, which is the property that matters
here: Qwen3-MoE-4x0.6B device stdout md5 `995c6b60e743e20139cd6e95c23fceb3` and
SmolLM2-135M `d6e2b2b3b00b397dd3fa89722790218e` are byte-for-byte what the same
build produced before it; `kraken-tests` is 2858/2858 (rc=0), `kraken-bench
--gate` rc=0, and `scripts/coherence_check.sh` on the three small models reports
**3 coherent, 0 not**, so the device arm still matches the f32 reference text for
text.

## What still fails, and where

**The file still decodes `〈|UNK|〉`, and the reason is measured rather than
argued: the overflow is in the model own hidden state, and every store that is
bounded just moves it one stage along.**

The MoE chain writes an f16 activation in four places, and they were bounded one at
a time:

| store | where | result |
|---|---|---|
| the silu product | `silu_mul` / the fused epilogue | row-scaled, with `d_sat_f16` as the fallback bound |
| the routed down projection | `gemm` f16 output | scaled down with its input, so it fits |
| the experts sum | `scatter_axpy_rows_kernel` destination | bounded with `d_sat_f16`: `moe.sum L38` goes from `nf=356` to clean |
| the residual add | `add_inplace` (`src/engine.cpp:3089`) | **still unbounded, and the first non-finite now**: `moe.post L38 N=10 W=2048 nf=210 badrows=1 first=12 peak=65504` |

Every bound now counts and the count is printed: the three bounded stores go
through `d_sat_f16_count` (`src/hip/krk_hip.hpp`), the backend reads the counters
back and `main_cli` prints
`[stats ] range     f16 activation bounds fired: silu N, experts-sum M` on every
run. Measured: this file reads `silu 6656, experts-sum 1018` (the failure it hid),
Qwen3-MoE-4x0.6B / SmolLM2-135M / any `--cpu` arm read `0, 0` with unchanged stdout
md5s. The 6656 silu clips are the fused *decode* path, which has no row maximum to
take, while prefill is row-scaled -- the counters rank the next piece of work.

One trap the range fix had to respect (the implementation is the last section of
this file): **a per-token exponent on the residual stream is a storage property,
not a factor that travels.** With the
hidden state scaled by 2^-e the gate is 2^-e*g and `silu(2^-e*g) != 2^-e*silu(g)`,
and q and k both carrying 2^-e give scores carrying 2^-2e, which softmax reads as a
temperature change. The exponent can only live on the storage of the residual
buffer -- every nonlinear consumer un-scales its input exactly first, every
producer re-scales on the way back, the logits row is un-scaled before the argmax.

So the quantity that must be representable is the hidden state itself, ~65504 and
above, at layer 38. Bounding store by store is whack-a-mole: the true value is out
of range and the reference carries it in f32. The fix is range, not another bound:

* **f32 activations for the layer** -- complete, structural: `gemm` `x` is
  implicitly `_Float16` throughout `Backend`.
* **One exponent per token, carried along the residual stream** -- self-consistent
  because every op on that stream is linear (rmsnorm is scale-invariant, a gemm
  scales its output with its input, the residual add is linear), so a per-token
  power of two travels from layer to layer and is divided out of the logits before
  the argmax. It disturbs the last-place bits, so it needs a re-recorded acceptance
  hash rather than an unchanged one.

The sharded S-2.1 fails the same way, and its trace is worth keeping: before the
scatter bound its first non-finite was
`moe.sum L46 N=10 W=3072 nf=1 badrows=1 first=1363 peak=3938`, with every routed leg
probed clean (2470 activation rows, zero non-finite, peak 29648) and a probe at the
routed sum *before* the shared expert folded in reading
`[shexp] L46 routedsum nf=1 peak=3930` -- the shared leg is exonerated. After the
bound the poison moved one stage, to `moe.post L46 N=6 W=3072 nf=1 first=1363
peak=4224`: the **residual add**, at the same row and the same column as before.
Both laguna files now stop at that store, and since it is the hidden state rather
than an intermediate product, bounding it would be worse than leaving it -- a
saturated hidden state is wrong by a factor, not by an exponent, and it would go on
to poison every later layer silently.

Two things must **not** be done. A wider clamp is not a fix: the CPU needs the true
value, so no saturation reproduces its text, and it turns a reported out-of-range
row into a silently wrong number. And the row scale must never be applied without
its compensation: the routed down projection is only correct because the scatter
multiplies the same power of two back in, so the two halves are one contract
(`KRK_DUMP_MOE=1` shows both).

## What landed (2026-10-09): storage-only per-token residual scaling

The residual stream is the one buffer the model itself drives out of f16, and the
one store no row-local bound can rescue: a saturated hidden state is wrong by a
*factor*, not by an exponent, and it poisons every later layer. It is also the one
buffer every block reads through a norm, and a norm is scale-invariant -- so the
fix is storage, not arithmetic.

`Backend::add_residual` (one block per token row, `add_residual_kernel` in
`src/hip/kernels/normalize.hpp`) sums `a * 2^-e + b` in f32, picks the smallest
`e' >= e` that keeps the whole row inside f16, stores the row divided by `2^e'` and
writes `e'` to a per-token i32 array (`ws_rexp_`, one entry per token row, zeroed at
the start of every forward pass). Every reader gets the exponent: `rmsnorm` takes a
nullable `row_exp` and multiplies by `2^e` before squaring (the norm is where a
nonlinear block gets true scale back), `add_rmsnorm` does the same
add-and-renormalize in the kernel it already had, the residual fold
(`gemv_kernel`'s `ACC` epilogue, `gemm_accumulate`) divides its contribution by the
row's exponent, and `Engine::gemm_residual`'s fallback -- what prefill rows and the
CPU take -- routes through `add_residual`. `KRK_RESIDUAL_SCALE=0` restores the old
path exactly (an all-zero array makes every other piece a no-op), so the A/B is one
process apart.

**In range it is bit-identical**, which is the property the acceptance hashes stand
on: `e == 0`, the store is `f16(f32(a) + f32(b))` -- `add_inplace_kernel`'s
arithmetic -- and this build's Qwen3-MoE-4x0.6B
`995c6b60e743e20139cd6e95c23fceb3` and SmolLM2-135M
`d6e2b2b3b00b397dd3fa89722790218e` are byte-for-byte the recorded ones, with
`kraken-tests` 2858/2858 (rc=0), `kraken-bench --gate` rc=0 and
`scripts/coherence_check.sh` **3 coherent, 0 not**.

Measured on `Laguna-XS-2.1-IQ3_XXS.gguf`, same binary and command
(`-p "The history of computing is a history of" -n 16 --greedy --ctx 512`):

| arm | generated text | counters | rexp |
|---|---|---|---|
| `KRK_RESIDUAL_SCALE=0` | `<|UNK|>` x16 | `silu 0, experts-sum 1020, residual 49152` | 0 |
| default | ` increasing complexity and abstraction. From the earliest mechanical calculators to modern quantum computers, each` | `silu 1, experts-sum 1020, residual 0` | 3 |

The ON arm is deterministic (identical md5 `dc9c47cfa1bd` in both repeats), and at
`-n 4` its generated text is byte-identical to the `--cpu` f32 reference
(`3ed670b0f97670dbdb00ce9ec3157640`): the file that decoded nonsense now produces
the reference's own text. `KRK_DUMP_FULL=1` prints no non-finite stage anywhere,
where the same probe used to stop at `moe.post L38 N=10 W=2048 nf=210 badrows=1
first=12 peak=65504`.

Two limits, both measured rather than argued:

* **The product stores are still clipped** -- `experts-sum 1020` in both arms, at
  `scatter_axpy_rows_kernel`'s destination -- and that is what the `-n 16` text
  diverges on at token 14 (13 tokens track the reference exactly). It is the fused
  decode path's per-row scale that is missing there, not the residual's.
* **The sharded S-2.1's `inf`-into-the-residual is fixed** -- by the same contract,
  one store further out. Its `inf` was never a single unbounded store: `KRK_DUMP_MOE=1`
  reads `[shexp] L47 routedsum nf=0 peak=65504` with the shared expert's own output
  finite, so the poison was the **sum** of two in-range rows overflowing the f16
  store at the fold. That fold now renormalizes through `add_residual` with an
  exponent of its own (`ws_fexp_`, `src/engine.cpp`), and the residual add reads it
  as b's exponent. Measured, same command one build apart
  (`-p "The history of computing" -n 1 --greedy --ctx 512`, dumps on):
  `silu 1, experts-sum 3810, residual 367` with `max exponent 60` (the clamp) before,
  `silu 1, experts-sum 3810, residual 0` with `max exponent 2` after. Nothing
  non-finite is stored on the residual path now.
* **What still clips is the routed down projection's own f16 store**: the trace
  reads `[moeout] L47 e=102 m=1 out_nf=16` (16 of 3072 elements), which the scatter
  then clamps into the staging sum -- that is what the `experts-sum 3810` count is.
  It is the fused decode path's missing per-row scale, and it is why the `-n 16` text
  still diverges from the reference at token 14. The down projection's output inherits
  its input's row scale (`gemm` is linear, and a power-of-two divide is exact), and
  the scatter already multiplies that same scale back in through `alpha`, so the one
  missing piece is the row maximum on the fused gate/up pair -- which is a cross-block
  reduction: one extra small launch per expert visit (measured cost of the marginal
  launch: ~1.5 us, i.e. +0.06% of a 1263 ms S-2.1 step, +0.34% of XS's 142 ms, +5% on
  Qwen3-MoE-4x0.6B's 112 visits), or f32 activations for that chain.

### What the reference actually does (read from the checkout, not inferred)

The llama.cpp checkout whose binary is this repo's ROCm oracle
(`/h/LLAMA-bins/llama.cpp`) answers the range question with an activation
*format*, and neither half of it is a clamp or a per-row scale:

* **Every activation in the MoE chain is f32.** `build_moe_ffn`
  (`src/llama-graph.cpp:2050-2300`) takes `ggml_mul_mat_id` results -- F32 by
  construction, `ggml_mul_mat`'s result tensor is `GGML_TYPE_F32`
  (`ggml/src/ggml.c:3351`) -- applies `ggml_silu` and `ggml_mul` to them (F32
  either way), and hands that F32 product to the down projection. 785411 is an
  ordinary number there.
* **The activation a quantized matmul reads is block-scaled.** It is converted to
  the weight type's `vec_dot_type` (`ggml/src/ggml-cpu/ggml-cpu.c:231-269`), which
  for the K-quants and the IQ types is **Q8_0/Q8_1: 32 int8 codes plus one f16
  (Q8_0) or f32 (Q8_1) scale**. The magnitude rides the *scale*, so a row of
  785411 becomes a scale of ~24575 times a code and the quantization error stays
  relative instead of total. That is the half that matters here: this engine
  stores the activation itself in f16, where 785411 is not a representable number
  at all.
* **The file supplies no clamp to copy.** `swiglu_clamp_exp` is real -- a
  per-layer expert-FFN clamp (`src/llama-hparams.h:377`, read in
  `src/llama-graph.cpp:2285` and only applied when the file sets it) -- but the
  laguna files do not carry it: the XS file's 60 KV pairs contain no `*clamp*`
  key. So the reference survives on the activation format alone.

The design implication, stated plainly: the per-row power-of-two scale this tree
grew is *this repo's* version of a block scale -- exact (a power of two spends no
mantissa) but coarser (one per row, not one per 32 values). Extending it to the
fused decode path is the incremental step; adopting a block-scaled activation
format through `gemm` is the structural one, and it would retire the whole class.
* Not wired for a non-zero exponent: `dflash_capture` copies the residual rows into
  the drafter's capture buffer (fine while no model that leaves f16 also runs a
  drafter -- none does here), and `KRK_DUMP` prints the residual **as stored**, so
  on a scaled run a debug trace shows deliberately smaller numbers. Both are the
  same contract: an exponent is only meaningful to a reader that knows it.

Perf, for the cross-engine record: Laguna-XS-2.1 decodes at **7.0 tok/s** (142 ms
step) with the auto expert budget (11.5 GiB for a 9600-slot corpus), and the
sharded S-2.1 -- 73 GB total, 256 experts top-10, a 64.1 GiB routed set against a
9.9 GiB device budget -- at **0.8 tok/s** (1263 ms/step) with prefill 0.6 tok/s.
llama.cpp on this box reaches 7.5 tok/s on the same S-2.1 file with its experts
resident in host RAM (`-cmoe`), so the gap is kraken's device-tier churn
(15% of the routed set fits) rather than arithmetic.
### The fused silu, the experts' sum, and what actually diverges (2026-10-09)

Three things were measured on the final build, and only the first two were predicted.

**1. The fused decode silu now takes the row scale.** `fused_gate_silu_kernel` is
one warp per output element, so a row's maximum is spread across `n_out/8` blocks
and the kernel has no maximum to take: it stored a saturated product row.
`silu_mul_scaled_kernel` is one block per row and does have one, so the routed
expert's gate/up pair is still grouped (two launches, not three) but the
activation runs in that kernel -- the fallback path the group already had, now
reached on purpose. `KRK_MOE_SILU_FUSED=1` restores the fused form so its cost is
measurable one process apart. Measured on Laguna-XS-2.1, `-n 16 --greedy --ctx
512`: `silu 1` before, `silu 0` after; Qwen3-MoE-4x0.6B is untouched (its rows
need no scale, `mul` stays 1, and the arithmetic is the same expression for
expression), with the recorded acceptance md5
`995c6b60e743e20139cd6e95c23fceb3` unchanged.

**2. The experts' SUM is the last unbounded store on that path.** It is the
scatter's destination, `dst[rows[i]][j] += alpha[i]*src[i][j]`, and a row of it is
the sum of ten experts that can each be in range and still add up past f16 --
exactly what that kernel's own comment claimed and no counter could confirm. It
now carries a storage exponent like the residual stream does: `scatter_axpy_rows`
takes a nullable `i32 *row_exp`, sums the row in f32, stores it divided by the
smallest power of two that keeps the row's own maximum inside f16, and writes the
exponent back; `ws_fexp_` is that slot, already read by both `add_residual` calls
below it, and zeroed once per layer instead of just before the shared expert's
fold. At exponent 0 this is `f16(f32(dst) + alpha*src)`, so in-range models are
bit-identical by construction. Measured: `experts-sum 1020 -> 0` on XS-2.1,
`3810 -> 62` on S-2.1.

**3. The clip was never what diverges the text, and the counters said so.** The
same command at `-n 4` -- where the doc above records the generated text as
byte-identical to the f32 reference -- already reads `experts-sum 1020`, and every
`-n` from 4 through 16 reads the same 1020: the clipped elements are all in the
prefill chunk and none in decode. Removing all of them changes no generated token
in sixteen. The divergence is one token in the middle of the continuation, device
`...calculators to modern quantum computers, each` against the f32 reference's
`...calculators to today's quantum computers,` (CPU stdout, same prompt and flags,
md5 `c788ce3c9ce99078bc807f2fe213996f`), and it is unchanged by either fix. What
is left is f16 *rounding* -- a greedy near-tie flipping 12 tokens in -- not a
value that left the range: the reference computes every activation in f32, and no
storage scale makes an f16 pipeline round like an f32 one. So "the `-n 16` text
matches the f32 reference exactly" is not a target either fix can reach, and the
honest measure of them is the counters (0, 0, 0 on XS now) plus the in-range
hashes, not a token-for-token agreement.

**A ratchet in the exponent pickers had to be fixed to get there.** Both
`add_residual_kernel` and the folded `rmsnorm_add_kernel` chose the new exponent
with `int e2 = rexp[r]` -- the exponent the row *arrived* under, which the `sc`
above had already applied to read the row at true scale -- and then halved `m`
until it passed f16's ceiling. Every store of an already-scaled row therefore
re-raised it by the same amount again: a row stored at 2^-3 whose true maximum was
still 8x over f16 left at 2^-6, then -9, and a row that crossed f16 once climbed
every layer until it hit the clamp. That is what the first run of the expert-sum
fix read as `max exponent 30` (a value that needed 5), and what S-2.1's `max
exponent 60` beside `residual 0` was: nothing saturated, the scale ran away, and
at 2^-60 the row's own elements underflow f16 entirely (min subnormal 6e-8). From
zero the exponent is the fresh smallest that fits, and XS-2.1 reads **5** -- which
is what the file's own documented activation peak implies (785411 = 65504 * 2^3.6,
and a sum of ten experts' worth of those lands a step or two higher).

S-2.1 still reads `max exponent 60` on some row even with nothing saturated
(`residual 0`, `experts-sum 62`), which is the `inf` class this doc already names:
`inf * 2^-e` is `inf`, so the halving loop cannot pass its test and walks to the
clamp. That is a value the exponent cannot help, not one it mis-scaled.

### The reference engines, read rather than inferred (2026-10-09)

Two checkouts on this box settle the same question from two directions, and both
were read for this change:

* **`/h/LLAMA-bins/llama.cpp`** keeps every activation in the MoE chain in f32
  (`ggml_mul_mat` results are `GGML_TYPE_F32`, `ggml/src/ggml.c`) and applies the
  file's `swiglu_clamp_exp` only when the file sets it -- laguna's do not. The
  activation a *quantized* matmul reads is block-scaled into Q8_0/Q8_1, so
  magnitude rides a scale there too.
* **`/h/LLAMA-bins/ik_llama.cpp`** is the same answer with the scale made
  explicit: its activation quantiser takes the block's own maximum (`amax` over
  32 values, `d = amax / 127`, `ggml/src/iqk/iqk_quantize.cpp`), so a block of
  785411 becomes a scale times 127 codes -- a *softfloat block scale*, one per 32
  values -- and its nonlinearity is still f32
  (`void MulMat::silu(int n, const float * src, float * dst)`). This repo's
  per-row power-of-two scale is the same idea coarser (one scale per row instead
  of one per 32 values, exact because a power of two spends no mantissa); it is
  what an f16-activation engine can do without changing its activation format.
* That fork also carries a real **DFlash** implementation
  (`src/llama-dflash.cpp`, 763 lines; `LLM_ARCH_DFLASH`, `LLM_ARCH_DFLASH_DRAFT`,
  `dflash.target_layer_ids`, `dflash.n_target_features`) -- the reference
  `docs/DFLASH-DRAFTER.md` has been missing, and worth reading before the next
  attempt at the 0/128 acceptance.
### The range readout became a gate (2026-10-09)

The two `[stats ] range` lines were the right instrumentation and the wrong
contract: a run could clip 1020 stores, or climb a storage exponent to the clamp
with nothing clipped at all, and still exit 0. `src/main_cli.cpp` now carries
`range_violated` / `range_verdict` / `range_exit_code`: either condition prints
`[error ] range generate: <n> silu, <n> experts-sum and <n> residual store(s) left
f16 and were clipped[, and the storage exponent reached the clamp]` and the
process exits **2**. `KRK_RANGE_STRICT=0` demotes the verdict to `[warn ]` and
exits 0, which is the arm the ad-hoc probes want.

Measured, and the reason it is worth having: Laguna-S-2.1 exits 2
(`1 silu, 62 experts-sum`, `max exponent 60` -- the clamp) against
`KRK_RANGE_STRICT=0` exiting 0 with the same text at `[warn ]`; Laguna-XS-2.1,
Qwen3-MoE-4x0.6B and SmolLM2-135M all exit 0 with `0, 0, 0` and exponent 5, 0, 0
respectively. Applied to yesterday's build this gate fails on both defects at
once -- the `experts-sum 1020` clip and the exponent ratchet -- which is the whole
point of turning a readout into an invariant.
