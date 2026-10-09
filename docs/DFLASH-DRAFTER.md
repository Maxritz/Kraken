# The DFlash drafter: measured, and the two defects that stop it

Measured 2026-10-09 on the box (RX 9070 XT, gfx1201, HIP 7.16). The question was
whether laguna can speculate with its own drafter. It cannot, and the measurement
separates *two* causes that would otherwise be read as one.

Everything here is one command away; the pair and the flags are named so the
numbers can be re-taken rather than believed.

## 1. Which pair can even load

A DFlash head set sizes its fusion projection against the **target's** embedding
width: `fc_.n_in == n_captured_layers * target_embd`, and a mismatch is a refusal
(`src/dflash.cpp:364`). So the drafter names the target in its own bytes, and
there is no reason to try pairs until one loads:

| head set | `fc.weight` | captured layers | target width it needs |
|---|---|---|---|
| `laguna-s-2.1-DFlash-Q4_K_M.gguf` | 3072x**18432** | 6 | **3072** |
| `laguna-xs21-dflash-q8.gguf` | 2048x**10240** | 5 | **2048** |

`laguna-s-2.1-DFlash-Q4_K_M.gguf` therefore cannot pair with any XS target on
this box, whatever its name suggests. Its own target is present but unmeasured:
`Laguna-S-2.1-UD-Q4_K_M-00001-of-00003.gguf` is `runnable`, 48 blocks,
**embedding_length 3072**, 256 experts -- which is the width this head set asks
for, and 73.6 GB of shards across the three files, so it is named here as the
pair that would fit and not as a run that happened. The pair that fits *and* was
run is:

    target  G:/More-models/laguna-xs2-Q4_K_M.gguf          (2048 embd, 40 layers)
    draft   G:/More-models/laguna-xs21-dflash-q8.gguf     (5 aux norms of 2048,
                                                            per-layer attn_gate)

Read the shapes with `python3 tools/gguf_tensor_shapes.py FILE` (added for this:
`kraken-inspect` reports a verdict, not a tensor list, and the verdict was
"refused" for both files, which is about their *role*, not their geometry).

One wall does **not** apply. `Engine::load_draft` refuses a recurrent target
(`src/engine.cpp:1595`, "needs a partial KV rewind, which a recurrent (gated
delta net) model cannot do" -- measured on Qwen3.5-0.8B, rc=1). laguna is not
recurrent: 40 blocks, per-layer `head_count`, `head_count_kv 8`, no `ssm`/`conv`
key of any kind.

The drafter loads and runs. Its own report:

    dflash drafter: DFlash-Draft-2048h-5L (qwen35-dflash-draft) 5 layers, 2048 embd,
                    64/8 heads, ff 8192, block 16, mask 12, ctx 512
    dflash capture: 2, 14, 26, 34, 40
    dflash contract: normed injection, causal block (5/5 layers causal),
                     aux norms stacked, rope 128/128 base 500000, decoder from tensors

## 2. The measurement

Interleaved and order-balanced (A B B A), `-n 16 --greedy --ctx 512 --chunk 256
--expert-warmup 1`, prompt "The history of computing is a history of":

| arm | decode tok/s | accepted | stdout md5 |
|---|---|---|---|
| A plain | **20.2** | -- | `0b3a8c42` |
| B dflash | **2.1** | 0/64 (0.0%) | `0b3a8c42` |
| B dflash | **2.1** | 0/64 (0.0%) | `0b3a8c42` |
| A plain | **18.9** | -- | `0b3a8c42` |

**A 9.3x loss**, and at `-n 32` the same pair reads 0/128 accepted (0.0%). The
`0/128 = 0.0%` is not a rounding artifact of a small sample.

The identical text is the useful part: with zero acceptance the drafter
contributes nothing, so both arms must print the target's own greedy output byte
for byte. They do. That is a *correctness* proof of the reject/rewind path, and at
the same time the measurement of what the proposer is worth here: nothing.

## 3. Two causes, and the experiment that ranks them

**Cause 1 -- the loop cannot accept even a correct proposer (proven).** Point
`--draft` at the *target itself*, where the proposal is the target's own argmax
and acceptance must be ~100%:

| probe | accepted |
|---|---|
| laguna-xs2-Q4_K_M self-draft, 8 rounds | **11/32 = 34.4%**, 2.3 tok/s |

This is **F8 on a third family**: SmolLM2-135M self-draft measured 20/72 = 27.8%
(`docs/MTP-DRAFTER.md` §3) and `docs/DIAG-decode-cpu.md` F8 recorded 40/144 =
27.8%. A self-draft below ~100% means proposals and verification are not looking
at the same position, and no drafter's accept rate means anything until it is
fixed.

**Cause 2 -- the DFlash proposer is wrong beyond that (proven, unexplained).**
0.0% is *below* the broken loop's own 34.4% floor, so the features or the block
arithmetic are also wrong. The two guesses the loader documents as load-bearing
are both ruled out:

| switch | captures | accepted |
|---|---|---|
| default | 2, 14, 26, 34 | 0/32 (0.0%) |
| `KRK_DFLASH_LAYER_OFFSET=0` | 1, 13, 25, 33 | 0/32 (0.0%) |
| `KRK_DFLASH_BLOCK_POS=1` | 2, 14, 26, 34 | 0/32 (0.0%) |
| both | 1, 13, 25, 33 | 0/32 (0.0%) |

The switches do exactly what they say -- the capture ids move by one -- and
acceptance does not budge. Note also that **no feature clamp fires on this
target** (the `clamped 18432 non-finite` warning belongs to the broken IQ3_XXS
file below, not to the features): the fused features are finite, so the failure
is not the f16 overflow the clamp exists for.

## 4. A separate correctness bug found on the way

`Laguna-XS-2.1-IQ3_XXS.gguf` is reported **runnable** by `kraken-inspect`, loads,
and then produces **all-NaN logits**:

    step 0 pos 9  nan=100352 inf=0 top1: 0=-nan(ind)(-nan(ind))

`nan=100352` is the whole vocabulary, at every step, so the greedy decode emits
`〈|UNK|〉` 32 times. The same command on `laguna-xs2-Q4_K_M.gguf` gives `nan=0`
with sane logits (top1 18141=21.19) and coherent text, so this is the file, not
the family, and not the tokenizer (`--tokens` returns 9 sensible ids for both).

Its types: dense F32 239 / Q4_K 40 / Q5_K 1, and the experts IQ3_XXS 277
(11.26 GiB) / IQ3_S 41 / IQ2_S 80. The suite pins the *layout* of all three IQ
types extensively (`tests/test_kraken.cpp:1281-1436`: byte spans, group-scale
ownership, grid-byte ownership), so a whole-block misread is unlikely and the
defect is more likely numerical -- which is where the search should start.

The engine already cites this file at `src/engine.cpp:1252`, but for *residency*
("11.20 GiB of experts and 0.86 GiB of dense weights ... at the 12 GiB cap it sat
at 81.6% resident"). That is how the NaN survived: the file was measured for what
it costs, never for what it says.

## 5. Not caused by the drafter

The plain arm already reports the expert tier running 70% short on this model:

    [warn] expert cache: the routed set is 17.7 GiB and the device budget is only
           12.4 GiB (70%) -- the cache is spending its budget churning rather than
           holding
    [stats] cache 11373 acquires | HOT 85.9% | WARM 14.1% | COLD 0.0%

A 12.35 GiB XS MoE is 17.7 GiB of routed experts on a 15.9 GiB card, so the
churn (1609 promotions in the plain arm) is the model-vs-card fact, not the
drafter's 0.5 GiB.

## Next actions, ranked

1. **Fix the loop (F8) first.** A self-draft must reach ~100% before any
   drafter's accept rate is a measurement of the drafter. `--debug-topk 1` at
   `pos` and `pos-1` is the probe the record already names.
2. **Then re-measure this pair.** If it is still 0%, bisect the drafter's own
   three stages (fusion -> injection -> block) from its own metadata; the two
   documented offsets are already excluded above.
3. **Independently, the IQ3_XXS NaN.** A file the loader calls runnable and that
   decodes all-NaN is a correctness hole, and it is one `--debug-topk` away from
   being reproduced.
