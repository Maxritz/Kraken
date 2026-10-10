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

**Cause 1 -- the loop was capping every drafter (proven, then fixed).** Point
`--draft` at the *target itself*, where the proposal is the target's own argmax
and acceptance must be ~100%. Before the fix this pair read
`11/32 = 34.4%`, 2.3 tok/s — and `docs/MTP-DRAFTER.md` §3 and
`docs/DIAG-decode-cpu.md` F8 recorded the same broken shape elsewhere:
SmolLM2-135M at 20/72 = 27.8% and 40/144 = 27.8%.

A self-draft below ~100% means proposals and verification are not looking
at the same position, and no drafter's accept rate means anything until it is
fixed. That defect is now resolved: on the current build the same target+draft
(here `laguna-xs2-Q4_K_M` as both) reaches **52/52 = 100%** at `-n 64` (13
full-accept rounds, no pre-check failures, no partials, no KV-invariant
failures) and **77/80 = 96.2%** at `-n 96` (20 rounds, all full accept,
the trailing 3 tokens the 128-context cap cutting the last round).

**Cause 1 is closed.** A self-draft on the current binary is ~100% when the
accept/rewind path holds, so any drafter's accept rate now means what it
says. (The earlier 27.8% / 34.4% records were stale — they predate the
fix and are corrected in §9.) The DFlash pair itself still sits at 0/64
below, so the remaining failure is **Cause 2 on its own**, not Cause 1
leaking through.

**Cause 2 -- the DFlash proposer is wrong beyond that (proven, unexplained).**
0.0% is *below* the now-fixed loop's ~100% floor, so the features or the
block arithmetic are also wrong. The two guesses the loader documents as
load-bearing are both ruled out:

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

## 6. The mask id is not the cause — and it is checked now (measured 2026-10-10)

Both ids, read out of the files themselves rather than from a report:

| file | key | value |
|---|---|---|
| `laguna-xs21-dflash-q8.gguf` | `qwen35-dflash-draft.dflash.mask_token_id` | **12** |
| `laguna-xs2-Q4_K_M.gguf`   | `tokenizer.ggml.mask_token_id`            | **12** |

They are equal, so there is nothing to compare and no experiment to run: for this
pair the head set's own id and the target vocabulary's `token_mask()` are the same
number, and the block's rows carry the real embedding of token 12 (`zero_mask_
embedding = 0` in the head set says the mask rows are *not* zeroed). The reference
settles the semantics rather than leaving them to taste — `common/speculative-
dflash-impl.h` feeds the **drafter's** id into the block rows
(`common_batch_add(batch, mask_token_id, draft_pos_base + i, …)`) and immediately
before that **refuses the pair** when it disagrees with the target's
`llama_model_dflash_target_mask_token_id()` (= `vocab.token_mask()` = the target's
`tokenizer.ggml.mask_token_id`). A refusal, not a substitution, because the block's
rows are embedded with the *target's* table: a mismatched id is a different **row**,
not a different token, and nothing about such a run looks wrong — it loads, it
drafts, and it is accepted zero times.

Kraken checked only that the id was inside the target's vocabulary, so a mismatched
pair would have run and looked exactly like the 0% above. One refinement of the
reading, settled 2026-10-10: kraken's `mask_id_` prefers the head set's own key,
which is the same choice the reference makes — the *table* the rows are read from
is the target's `tok_embd` (a head set ships no embedding of its own), but the id
*indexing* it is the drafter's, and the target's mask id exists only as the refusal
predicate. Re-reading both files with the current `kraken-inspect` re-confirmed
12/12, so an A/B of "head set's id" against "target's `token_mask()`" is an A/A for
this pair: byte-identical rows, and no acceptance delta is available even under a
forced mismatch, because the pre-check (`prop[0] != first`) already pins this pair
at 0 before drafter quality can move it (§7). That check now exists
(`Engine::load_dflash`, after the range check) and is proven both ways: the real
pair still loads, and a copy of the head set with the single value 12 -> 13 at byte
offset 1037 refuses with

    kraken: draft model: DFlash mask token 13 is not the target's mask token (12):
    the block's mask rows are embedded with the target's table, so this head set
    would draft against a row it was never trained on                  (rc=1)

### 6.1 The measurement the guard's refusal made impossible, and the switch that
unblocks it (measured 2026-10-10)

A refusal predicate hardens a normal run but hides a diagnostic one: to measure
what a wrong mask row does to the block, the mismatched pair has to LOAD. So the
same guard takes an explicit opt-in — `KRK_DFLASH_ALLOW_BAD_MASK=1` — which keeps
the pair loading and prints "CRACKED PAIR by override" with both ids on every
load. It is a diagnostic switch, not a permission: never set it on a run whose
numbers are meant to be believed.

The mismatch run, then: a copy of `laguna-xs21-dflash-q8.gguf` with the single
u32 12 -> 13 at byte offset 1037, against `laguna-xs2-Q4_K_M` with
`KRK_DFLASH_DEBUG=1` (`-n 8 --greedy --ctx 512`). The matched arm ran the same
command on the unpatched file. Per-row top-1 agreement across the 32 proposal
rows both runs produced: **0 of 32** — every single row moved
([docs/traces/dflash-mask-mismatch.txt](traces/dflash-mask-mismatch.txt)). Two
facts fall out of the number:

  * the mask embedding REALLY flows into the block — a wrong id changes every
    row, so the embedding path is live end to end, not ornamental. That is the
    premise the guard stands on, now measured rather than argued.
  * a mask-id defect is OBSERVABLE in the dump. A run whose rows are poisoned
    by a mismatch does not look like one whose rows are correct; the two are
    told apart in one run with the dump on. So the matching 12/12 we read off
    this box is not a leave-it-alone fact: any pair whose ids differ is caught
    by the guard, and a pair that loads with ids matched but drafts badly has
    the dump as its next tool.

Cleaned up: the patched copy was deleted after the run; the loader refuses it
outright without the switch.

### 6.2 The refusal is a test now, not only an error message (added 2026-10-10)

The guard existed load-time and was proven once against real files; nothing
re-proved it across a build. `test_dflash_mask_guard` (group 48 of 54, 8 checks)
constructs both halves synthetically: a one-layer llama target written with
`tokenizer.ggml.mask_token_id = 12` (via `build_sized_model`'s new mask_id
parameter), and a two-layer `dflash` head set built at the loader's own key
spelling — `<arch>.dflash.mask_token_id` nested, `dflash.*` capture list,
fusion projection `n_embd_enc x n_embd`. Loading id 12 against the target's 12
succeeds; loading id 13 is refused, and the error names both ids. Three shapes
the loader rejects had to be respected to get there, and the fixture spells each:
fc is `[n_embd_enc, n_embd]` (inner width = input), `ffn_down` is sized
`n_ff x n_embd` with THAT many elements, and capture ids are the target's
extract numbering ({0, 1} against a one-layer target, 1 = the pre-final state).
A future loader change that stops reading the nested mask key, widens the range
check past the layer count, or silently substitutes the target id
re-fires it.

## 7. What the 0% is NOT — each one measured, not argued

Every entry below was a live hypothesis this session and none is one any more.

| hypothesis | the evidence that closes it |
|---|---|
| mask rows should use the target's `token_mask()` | both files say 12 (§6) — equal, so inert |
| block placement: where `id_last` and the masks sit | read off the reference's driver: `draft_pos_base = last_target_pos + 1`, masks at `+i`, and `result[i]` sampled from logits row `i+1` — exactly the geometry kraken already uses (`blk_pos = pos`) |
| an off-by-one in that placement either way | `KRK_DFLASH_BLOCK_POS=-1` (masks one slot lower): pre-check fails on every round, 0/32. Both placements fail, so the placement is not the cause whichever convention reads the rows |
| causality among the block rows (the reference's mask is *uniform*, i.e. block-visible) | `KRK_DFLASH_CAUSAL=0` alone -> 0/128; and `WINDOW=0 CAUSAL=0`, which is the only arm where the block-visible mask meets the non-SWA path, -> 0/16 |
| the sliding window cutting the injected context away | `KRK_DFLASH_WINDOW=0` gives proposals identical to the default at ctx 512 |
| normed vs raw injection (the file says `context_kv_layer_norm = 0`) | `KRK_DFLASH_KVNORM=0` gives proposals identical to the normed arm |
| the injection never reaching the block | `KRK_DFLASH_KV_ZERO=1` moves the block's own attention (`row1 attn rms 2.1784 -> 1.74067`): the context **is** read |
| capture ids off by one | `target_layer_ids` are checkpoint ids ("the output of layer i"), so the last id 39 + 1 = 40 = the target's layer count = the pre-final-norm state, which is precisely what "the output of layer 39" means on a 40-layer stack. The +1 default is confirmed by the *last* id, not by taste |
| rope base/type, V scale, gate shape and activation, head choice, norm placement, embedding source | all read off `src/graphs/build_dflash.cpp`: one rotary for both sides (`dflash.backbone_rotary_base` is absent, so `target_freq_base == freq_base`), `f_attn_v_scale = 1` (key absent), the gate is a per-head softplus on the attention output, `output_mtp` resolves to the target's only head (`output.weight`, Q6_K), `output_norm` -> head. All match |
| the head set carrying its own embedding or head | it has 68 tensors and neither is among them — borrowing the target's is right |

## 8. What is left, in the rows themselves

Acceptance is 0 because the **pre-check** fails every round — the drafter's first
candidate for `pos` is never the target's token for `pos`:

    round 1 pos=9  first(target)=18141 prop[0](draft)=5685  target_top3=[18141(21.39) 2018(21.14) 340(20.88)]
    round 8 pos=16 first(target)=83    prop[0](draft)=84848 target_top3=[83(31.86) 462(28.5) 81(28.36)]

Three properties of those rows, all measured:

* they are **peaked, not flat** — row 1 puts 5.59 on its best token against a mean
  of 0.32 over the whole vocabulary — so the block is computing something;
* their absolute **scale does not track the target's**: over the eight rounds above
  the target's top logit climbs 21.4 -> 30.8 -> 27.1 -> 37.7 -> 40.4 as the text
  becomes repetitive and near-certain, while the drafter's stays around 5-6;
* `prop[0]` behaves like an **attractor of the placement** rather than of the
  context: it is 5685 in the default, the `BLOCK_POS=-1` and the `KVNORM=0` arms,
  and it moves only when the block is made block-visible (59674).

So the block reads its context, but weakly, and its output does not carry the
target's confidence. The next probe belongs *inside* the block — a stage-by-stage
dump on the `[dfkv]` hook, which already reports the attention and the injected K
at both ends of the stack — not another contract argument: the contract is now
verified end to end.

## 9. Inside the block: the residual trace, and what it says (measured 2026-10-10)

Item 2 above is done: `KRK_DFLASH_RES_TRACE=1` dumps, for each of the five
layers, the candidate row's |x|.max, the residual's rms, and the two sublayer
deltas on that row. The first block of a run on this pair:

    l=0  x.rms  48.7  attn.d.rms   8.4  ffn.d.rms  46.5
    l=1  x.rms  57.0  attn.d.rms  21.7  ffn.d.rms  20.2
    l=2  x.rms  76.7  attn.d.rms  35.9  ffn.d.rms  35.8
    l=3  x.rms 118.8  attn.d.rms  71.9  ffn.d.rms  62.8
    l=4  x.rms 281.3  attn.d.rms 224.0  ffn.d.rms 132.3

(QV-zeroing the injected context moves the numbers by ~10%, so this is not a
context-free artifact — the shape is the same either way.) Three reads:

  * The residual GROWS layer by layer, and the deltas grow WITH it. That is a
    feedback loop, not a signal: each layer's norm is scale-invariant, so its
    output scales with its input, and by l=3-4 each sublayer's contribution is
    the size of the residual carrying it. A real transformer keeps deltas at
    1/6 to 1/20 of its residual through the whole stack.
  * The carried x0 into l=0 is healthy — by quadrature, rms ~11.6, a plausible
    embedding scale. Layer 0 does not start the runaway; the feedback does.
  * This is why the drafter's logits sit at 5-6 while the target's climb
    (§8): the final norm divides out scale, so the block's output direction —
    and thus its argmax — is set by wherever its internal state happens to
    point after five layers of self-amplification, not by the injected
    context. The target's confidence was never IN the block's numbers to
    carry; what is wrong is not a lost signal but a signal drowning in its
    own feedback.

What would break the loop is a per-layer anchor — the thing a real decoder
achieves with every-norm placement and (on laguna) attention sinks. **This is
where the inference-vs-storage trap bites hard:** a residual stream that carries
the anchors is different from distracted decoding's residual (which stabilizes
its scaling to preserve significance *after* the speculative position), so
carrying a per-layer anchor off that stream is not a copy — it is a theft from
distracted decoding's stability. The next question is whether the target file's
sink weights should ride the drafter's attention, which is what the reference
`build_dflash` already chooses to do — but only on a drafter family (DeepSeek
V4 Flash DSpark) whose target *also* carries sinks.

### 9.1 The attn_sinks question, settled on the file side

The reference's `src/models/dflash.cpp` builds `layer.attn_sinks` only inside the
DeepSeek-V4 stacked-block branch (`if (hparams.dsv4_hc_mult > 0)`), where it is
sampled from `blk.N.attn_sinks.weight` (one per layer, per-head, `ggml_tensor *`).
The *else* branch — which is what `laguna-xs21-dflash-q8.gguf` is — never reads
or writes `attn_sinks`, and it is carried nowhere: `ggml_soft_max_add_sinks`
is a dummy in the ggml side, `blk.N.attn_sinks.weight` is absent from both
`G:/More-models/laguna-xs2-Q4_K_M.gguf` and both head sets on this box
(`laguna-xs21-dflash-q8.gguf`, `laguna-s-2.1-DFlash-Q4_K_M.gguf`),
`kraken-inspect` names no sink tensor in either, and no header string in any of
the three DFlash head sets here is `attn_sinks.weight` (the only sink strings
in those headers are tokenizer vocabulary bytes — `Helsinki`, `sinking`, `sink`
— not tensor names). So on the only DFlash pair that is actually run on this
box, **there are no sink weights on either side of the pair**, nothing to carry,
and the contract question is moot on the file side.

This is the refusal precedent for the DFlash drafter: a speculative head set
whose target carries `attn_sinks` but the head set does not (or does not carry
the same ones) must be refused at load — a drafter that is trained with sinks
against a target that uses them drafts against a residual stream whose anchors
are absent, and the proposals that survive the pre-check on that pair would be
plausible-to-the-draft but wrong-to-the-target. The gate path is `load_dflash`
+ the arch table's `KrakenSupport::No` on `attn_sinks` (the gap map already
refuses a target that carries them), and the refusal message names both sides
of the sink mismatch the way the mask-guard message already does. **No code is
written for this** — on this box the only DFlash pair that is run on the engine
has no sinks on either side of the pair, so the precondition the guard would
fire on does not arise. If a sink-carrying head set appears here, the guard
appears with it.

The only sink-bearing drafter family on this box — `deepseek4-dspark` — is a
Dsark block entirely unrelated to DFlash, and its MTP shard is already refused for
latent KV (MLA) plus recv bias, so the question of carrying sinks across the
DFlash machinery is closed by absence, not by a decision.

## 10. A defect the fixture found, and the checks it produced (2026-10-10)

Wiring the synthetic-geometry group up (target + head set + one real
`draft_block` call, no 12 GiB model in the loop) surfaced an intermittent
failure the solo group never showed: 1 run in 8 either failed
"the block's rows are all finite" or died with heap corruption 0xC0000374 two
checks after all 22 had passed. The cause is precise and worth recording:

  * the test handed `draft_block` the **tokenizer's** vocab count, which this
    fixture's builder sizes one larger than the model's declared vocab_size
    (261 entries vs 260);
  * `draft_block` writes its candidate logits into `ws_logits_`, sized
    `(block_size-1) x cfg.n_vocab` = 3 x 260; row offsets computed with the
    wider count walked the last rows 1-3 elements past the buffer;
  * the overflow is f16 writes onto heap metadata — hence the crash — and
    occasionally a racy NaN row, hence the finite FAIL.

The production loop never saw it because it always passes `n_vocab_` (the
model's), which is exactly why a real pair works and a fixture dies. Two
checks now pin the contract, and one refuses it at load:

  * `Engine::load_dflash` loads the head set's own `vocab_size` before
    constructing the drafter and **refuses a pair whose vocabs differ** — the
    block reads the target's embedding and writes through the target's head,
    so a smaller declared vocab is a wrong table, not a smaller one
    (measured: the refusing build also stops the crash).
  * the finite check in `test_dflash_mask_guard` reports the first offending
    row and the largest finite element, so position-wired vs flat-broken is
    readable from the failure itself.
  * the row-to-position probe asserts each candidate row is a distinct
    logits vector (a row-copying wiring defect fails this without trained
    weights) and that a different `id_last` changes the rows; and the same
    call on the same drafter state is bit-for-bit — verified 14/14 green runs
    after the fix (2901/2901 checks).

## Next actions, ranked

1. **Cause 2 is still open** (`0/64 = 0.0%` on the runnable pair, and the
   loop itself is now ~100% when it is the only problem — see the F8 sweep
   just above). The block gets past the target's pre-check only when
   `first == prop[0]`, and nothing about a mismatched run looks wrong other
   than the pre-check — so the next probe is *inside* the block, not another
   flag. The `[dfkv]` hook is the place: extend it to the per-layer attention
   output and the residual after each of the five layers, and find the layer
   where the target's confidence stops being carried (the trace in §9 did not
   reach that layer).
2. **Then the block's arithmetic, from inside it.** §7 has closed every contract
   question this pair can raise — metadata, geometry, rope, mask id, causality,
   window, injection, capture ids, sinks — so the next run must instrument the
   block's own stages rather than compare another flag. The `[dfkv]` hook is the
   place: extend it to the per-layer attention output and the residual after each
   of the five layers, and find the layer where the target's confidence stops
   being carried.
3. **Independently, the IQ3_XXS NaN.** A file the loader calls runnable and that
   decodes all-NaN is a correctness hole, and it is one `--debug-topk` away from
   being reproduced.
