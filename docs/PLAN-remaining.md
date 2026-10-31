# Remaining work, split by what it waits on

Status 2026-10-31, `main` = `33189b5` (PR #88 merged). Gate on this branch:
**7401/7401 checks, 175 groups, tool 18/18, proof 17/17, exit 0**, baseline
re-recorded at `2026-10-31T15:34:15Z` (three groups are new: `test_dump_identity`,
`test_cancel_and_prefix`, `test_http_listener_lifetime`).

This file indexes what is still open. It replaces the "what is left" paragraph in
[PLAN-mtp-and-prefill.md](PLAN-mtp-and-prefill.md), which remains the record of
how items 1 and 4 got where they are.

The split is **by what the step waits on**, because that is what decides whether
a session can move it:

| bucket | waits on | can be done in this workspace |
|---|---|---|
| **A** | a file: a checkpoint or a head set | no |
| **B** | code and a measurement on this box (2 cores, AVX-512 VNNI, 3 GB RAM) | yes |
| **C** | nothing -- closed, do not reopen | -- |

Every step still has a **done-when that can fail**. A step that cannot fail is a
description, not a step.

---

## A. Waits on an artifact

### A1 -- `qwen4-mtp`: `Partial` -> `Yes`

**Needed:** a `qwen4exp` trunk (Qwen3.8-Flash or a REAP-288 repack) **and** its
`qwen4exp-mtp` head set (the 889 MB `mtp-q2_0.gguf`, or any file carrying the 31
tensors in [MTP-HEAD-SCHEMA.md](MTP-HEAD-SCHEMA.md)).

**How:** one command, already in the tree:

```
scripts/mtp_spec_gate.sh <trunk.gguf> <head.gguf> 128 "<a 2 KB prompt>"
```

It runs the same prompt greedy, same seed, with and without `--mtp --mtp-spec`,
parses `step-1 H of T (P%)` and `[stats ] decode ... tok/s` from both, and exits
non-zero unless **step-1 >= 60 % and spec decode is strictly faster**.

**Then:** fill the six-field table in *1.8's runbook* (PLAN-mtp-and-prefill.md)
and move the arch row in the same commit. No row moves without the table.

**If step-1 lands under 60 %**, the causes are already ordered by what this
project measured, and each is a switch or one line:

1. the paired-token convention (head fed residual `i`, token `i+1`);
2. the per-stream `hnorm` switch (`KRK_MTP_HNORM`);
3. dense vs indexed attention (step 1.6(b) -- dense is what the reference does
   when the file declares no compress ratio for the block).

**Cost:** half a session once the files exist.

**A malformed export is refused by name, not accommodated (checked
2026-10-31).** llama.cpp [#30696](https://github.com/ggml-org/llama.cpp/issues/30696)
reports a qwen4exp MTP *draft-only* export -- full `block_count`, only the nextn
block's tensors, no trunk -- segfaulting `llama-server` inside graph reservation:
`graph_mtp` asserts the four `nextn.*` tensors and then calls `build_hc_mix` on
trunk tensors that are null, and the symbol-less crash is what a user sees. Three
synthetic files cover the same ground here, and all three stop at load with the
answer in the message:

| file | what it is | `kraken -m` says |
|---|---|---|
| `/tmp/mx2/draft_only.gguf` | qwen4exp, `block_count=8`, only `blk.8.nextn.*` | `carries no token embedding ... and no attention tensor: its 4 tensor(s) name no model schema` |
| `/tmp/mx2/draft_partial.gguf` | the same plus embed/norm/head, still no layer | `arch 'qwen4exp' is a gated delta net but its ssm.* geometry is missing` |
| `/tmp/mx2/no_layers.gguf` | a qwen2moe header that *would* load, zero `blk.*` tensors | `layer 0 is missing its norm weights` |

Exit is 1 in every case; none crashes. The property is structural rather than
lucky: the schema check (does this tensor set name a model?) and the per-layer
required-tensor check both run before anything dereferences a weight pointer, so
a file that declares layers it does not carry cannot reach a forward. No code
changed for this -- it is a *verified*, not an assumed, refusal.

### A2 -- `deepseek4-dspark`: wired, `No` -> `Yes`

**Needed:** a `deepseek4` trunk plus its dspark nextn block, either in-file
(`blk.<n_layer>.*`) or as a `--mtp` sidecar.

**How:** the same gate script. The body is wired and measured
(`test_mtp_sidecar_mla`, **22/22**); the only new failure mode is geometry, and
every geometry check prints both numbers by name (latent width, lanes, expert
count, `eh_proj` shape).

**Done when:** the gate exits 0 and the row moves; if the gate refuses at load,
the message names the tensor, and that is the record.

### A3 -- `dspark` (qwen35 markov head): `Partial`

**Needed:** a DFlash2 head trained on the pairing it is being run against, plus
upstream reference logits for one block on that pairing.

**Why it is stuck (measured, 2026-10-17):** the 5-layer decoder never sharpens
the mask row at any depth -- gaps 0.02-1.1 from embed through post-`l4`, flat
final -- while capture scale, KV norms, the KV self-check, the attention route
and the mask id are all healthy. Plumbing is excluded; the open branch is
**checkpoint pairing / basis mismatch**. Full probe data in
[DFLASH-DRAFFER.md](DFLASH-DRAFFER.md) §11.

**How:** (i) a DFlash2 head on its own Qwen3.8 pairing -- proves the engine path
can produce sharp rows at all; (ii) upstream reference logits for one block on
this pairing. Either answer moves this: a sharp row means the pairing is wrong,
reference logits mean the graph is.

**External evidence for the same signature (2026-10-31).** A DGX Spark thread on
DeepSeek-V4-Flash-0731-DSpark reports the identical shape -- draft acceptance
`~1.0-1.15` tokens/step, 14-18 tok/s -- and it was **not** the model, the
hardware, or the pairing. The draft had inherited the *trunk's* quantization
config (ModelOpt NVFP4), so its FP8/MXFP4 expert blocks were run through the
NVFP4 dequant path. On its own path: position-0 agreement 89-91%, mean acceptance
length ~4.5, ~70% acceptance, **45.4 tok/s average** (2.7-3.0x). Two lessons that
transfer:

1. **Flat acceptance is a weight-path signature before it is a pairing verdict.**
   Shapes and names agree under a wrong format, so nothing upstream fails -- the
   head is simply fed weights it cannot decode. A3's "pairing / basis" branch is
   still the open one, but the next thing to rule out is a *format* mismatch
   between the head's weights and the path they are read through, not the head's
   training.
2. **Fail closed, and say which tensor.** Their fix list is a guard that raises
   when the draft resolves to the trunk's quant path, plus a weight-coverage
   check per draft stage. Ours had one fail-open of that class -- see below.

**Closed by this (2026-10-31, code):** a routed-expert trio is read as one launch
in one format (`gemm_group` takes a single weight type; the expert cache sizes a
fetch from one format), so a gate/up/down that *mix* formats was walked at the
wrong stride and answered wrongly with every shape check passing. `qwen2moe` with
`ffn_gate_exps` F16 and `ffn_up_exps` Q8_0 used to load; it is now refused by
name, both tensors and both formats (`expert_formats_agree`, `src/model.cpp`, all
three loaders: trunk, llama-style nextn block, dspark head set). An NVFP4 bank
with no `.scale` sidecar is still 1.0 by construction -- correct for every other
format -- and now says so loudly (`warn_unscaled_nvfp4`). Negative control: the
same stack with a uniform trio loads, `test_mixed_expert_formats` 7/7.

### A4 -- Partial target rows: `qwen2vl`, `qwen3vl`, `gemma2`

Nothing is known-missing in code; the rows say no artifact has proven the graph.
`qwen2vl`/`qwen3vl` are vision families and the text path is what a CPU run can
check -- out of scope unless the user asks for it.

**How (gemma2):** one run on a real file, `--greedy -n 32`, plus a
`KRK_DUMP`-compared `--chunk` sweep; a wrong per-layer window geometry garbles
the text past the window and the dump names the first stage that differs.

**Done when:** the row's `why` is replaced by a recorded run, or the file is
refused by name with the tensor that is missing.

### A5 -- OLMoE shared path (C3/G13): text disagrees with an independent oracle

**Needed:** the exact model **and** a bounded llama.cpp run on the same prompt.
CPU/device agreement is not evidence here -- both can be wrong the same way.

**Done when:** the two texts agree, or the first diverging stage is named by a
`KRK_DUMP` diff against the oracle.

### A6 -- The prefill format reopens only on evidence

Closed by measurement: speed gate passes (**1.43x**), error gate fails (**1.25x**
weights x **~1.16x** activations), and the compromise design measures **1.00x**.
Numbers and the three refuted hypotheses are in
[GEMM-PERF-WALL.md](GEMM-PERF-WALL.md); the bench is `kraken-prefill-fmt-bench`.

**What reopens it:** a real artifact showing q8_0's error sits far below what the
model tolerates. Nothing in this workspace can produce that.

### A7 -- The 4 GB-on-16 GB-RAM promise has never been run end to end

**Needed:** a real 30B-class MoE file and a box with 16/24/32 GB of RAM. This
workspace has 3 GB and no model, so every residency number here comes from
synthetic files.

**How:** `kraken -m <30B-A3B q4> --ctx 4096 -n 64 --stats` on that box, with
`KRK_MOE_PLAN=1` and `KRK_EXPERT_PIN_HITS` on, plus `kraken-cpu-expert-ceiling`
on the same file for the per-expert cost the plan has to beat.

**Done when:** a decode tok/s number exists for each RAM budget, with the hot-set
hit rate and the cold read bytes/token beside it. **This is the only step that
tests the product promise**; everything else tests machinery.

---

## B. Waits on code and a measurement here

### B1 -- CPU GEMM: **measured, no legal win; the kernel stays wsc8**

Status 2026-10-31. The premise this item was written on -- "the production kernel
is at 40 % of V1" -- is **wrong for this host**, and the correction is the first
result:

* `32.2 GMAC/s` is what the **ablation harness** reports for its own L0 arm, i.e.
  the no-table `dot_q8_0_q8_x8`, not what the engine calls. The engine has called
  the 16-lane scale-table kernel (`dot_q8_0_q8_wsc8`) since the table landed.
* In `kraken-q80-gemm-bench` on this host, same process, same shape
  (768x2048x512, nc=128, best of five, four runs): **wsc8 48.1-49.1 GMAC/s**
  against **x8 34.0-35.7**, i.e. the shipped kernel is **1.40x the one the
  ablation calls "production"**.
* The distance to V1 is **not resolvable on this host**: `kraken-isa-ceiling`
  measures the shipped kernel against V1 in one process and returns 70 % in one
  call and 110 % in the next, same function, same data. That 1.5x spread is the
  noise floor, so a "60 % of V1" claim would be as unfounded as the "40 %" one --
  what can be said is "within the host's noise of its own instruction-mix
  ceiling".

| kernel | r1 | r2 | r3 | r4 |
|---|---:|---:|---:|---:|
| x8, no table (the pre-change kernel) | 33.99 | 35.69 | 34.05 | 34.49 |
| **wsc8 16-lane tables (SHIPS)** | **48.70** | **48.95** | **48.12** | **49.05** |
| pv8 pair table (the one remaining repack idea) | 44.56 | 45.87 | 44.47 | 45.60 |
| wsc8 + tree reduce (changes the sum order) | -- | 47.11 | 46.14 | 49.01 |

**pv8 was the last shippable idea and it lost.** It cut the per-(row, block pair)
scale chain from five uops to three by hoisting the weight row's two f16 scales
into a compact table, which is precisely the "repack the scale path" family the
plan pointed at. It is **byte-identical to x8** on every cell
(`identical: x8 vs pv8 yes`, every run) and it is **7-9 % slower than the kernel
it would replace**, because the loop was never bound by those five uops: at this
shape the activation load, the masked sign flip, `vpdpbusd`, `vcvtdq2ps` and
`vfmadd` are the floor (5 uops / 64 MACs), and pv8 adds a table load to that
floor while removing uops that were already hidden behind the pair loop.

The two rungs that DO pay are **format-class** changes, and A6 closed that door:
L2 (constant weight scale) 135 % of L0 and L7 (integer accumulate) 124 %, both
marked WRONG because each is a different arithmetic than the file's.

**Disposition.** `src/quant.cpp` is unchanged; the shipping kernel is wsc8 with
the x8 fallback and the runtime VNNI dispatch. pv8's kernel and its two table
builders were **moved out of the library into `tools/bench_q80_gemm.cpp`**, where
the four-column comparison and the byte-identity check reproduce the result; a
kernel that was measured and rejected does not belong in the shipped path. The
test suite keeps the x8-vs-wsc8 byte-identity arm (the two kernels that ship) and
no longer references pv8: a suite that asserts something about a candidate that
is not in the build is testing nothing.

*Done when* (restated, and now true): the kernel's distance to its own ceiling is
measured on this host, every remaining repack idea is either measured worse or
classified as a format change, and the file the engine calls is the file the
numbers are about.

**What this means for the target.** Do the arithmetic with the number this node
measures, not the one the plan assumed. A 30B-A3B-class token is ~3 GMAC; at
48-49 GMAC/s that is **~16 tok/s of prefill per core**, so four cores project to
**~64 tok/s**, not 200. The kernel is within this host's noise of its own
instruction-mix ceiling (see [GEMM-PERF-WALL.md](GEMM-PERF-WALL.md), "What B1
closes as"), so no legal kernel change closes the 3x: it needs a format whose
scale group is longer than 32 by construction (a format decision, A6-adjacent) or
hardware with AMX-class MACs. Stating the projection is the point of B1.4; a
projection that assumed the 40 % premise would have promised 300.

**One number in the ceiling tool is a harness artifact, and it was nearly
mistaken for a win.** `kraken-isa-ceiling`'s `W` arm prints 1.38x over the
shipped kernel -- but its activation scale table is built from constants outside
the timed loop, so it prices a kernel that needs no activation scale at all
(that is L1, the rung the ablation shows is free). Recorded in GEMM-PERF-WALL.md
so it is not picked up later as "a 1.38x we never took".

### B2 -- The server runs one generation at a time

`src/server.cpp`: connections are concurrent, **generate() is serialised by one
mutex**; prefix reuse is a single-session linear record and there is no
continuous batching.

**B2.1 cancellation -- DONE.** `GenerateParams::cancel` is a
`const std::atomic<bool>*` the loops read once per prefill chunk and once per
generated token. The decode side is one check inside
`TokenEmitter::accept`, so all five decode paths (plain, draft-head, MTP, the two
DFlash ones) inherit it instead of growing a sixth copy of the stop handling; the
prefill side is one guard at the top of each chunk loop, which also sets the KV
cursor to the rows that were actually written so a cancelled prefill cannot leave
a cache claim above them. `GenerateResult::cancelled` distinguishes "the caller
hung up" from "the model ended", because `finish` stays `FinishStop` (OpenAI has
no cancelled reason).

The server raises the flag from the only place a disconnect is observable: a
failed SSE write in `stream_cb`. A client that closes its tab now stops the
decode at the next token instead of paying for the rest of `max_tokens` into a
dead socket. `note_kv` drops the prefix record on a cancelled run: the KV then
holds a *mixture* of this prompt's first rows and the previous run's tail, and a
record that claimed otherwise would answer from the wrong row.

*Test:* `test_cancel_and_prefix` arms 1-2 (24 checks). A sink trips the flag on
the third delta: `generated == 3`, `cancelled`, and the partial text is a prefix
of the uncancelled run. A flag raised before the call: `generated == 0`,
`kv_pos() == 0`.

**B2.2 prefix reuse across requests -- DONE, as the run and not the prompt.** The
record was the last request's *prompt* ids, although the KV holds its *generated*
rows too. It is now prompt ids + accepted tokens cut to `engine_->kv_pos()`, which
is exactly the set of rows that exist (the cut matters: a decode step writes the
row for the token it consumed, so after G tokens the cache holds prompt + G - 1
rows and the last token has no row). A second chat turn -- whose prompt is the
first turn's prompt plus that turn's answer -- now prefills only the new
question. The response reports it as `usage.prompt_tokens_details.cached_tokens`.

*Why not a radix tree (the plan's "how", not its done-when).* With **one** KV
region the longest prefix a request can reuse is the one physically in the cache,
and the linear record already finds every token of it; a tree only beats a linear
record when several branches are kept alive at once, which needs per-sequence KV
slots -- that is B2.3, and the tree should land with it, not before it. The
done-when the plan set is met and measured: the second request's `cached_tokens`
covers rows the first request generated.

*Test:* `test_cancel_and_prefix` arm 3. It also measures the fixture in token ids
(the tiny model writes arbitrary bytes, so the text trip through the tokenizer
loses the tail) and asserts the claim equals the longest prefix the record and
the new prompt share -- no over-claim, nothing left on the table.

**B2.3 continuous batching -- DONE. Slots, batched decode, the wave scheduler and
per-row addressing in the latent-attention family.**

`AttnDesc::row_pos` / `row_kv_base` were already implemented in the CPU attention
path. This session finished the other two, and `kraken-attn-multiseq` now proves
the whole contract bit for bit -- five arms, each comparing one batched call
against one call per sequence:

| op | change | arm |
|---|---|---|
| `attention()` | (was already done) | dense causal, non-causal, sparse run list |
| `kv_append()` | per-row position and region base, f32 and int8 | 2 arms |
| `rope()` | new `Backend::rope_rows()`, default REFUSES (`false`) so HIP/NPU cannot silently rotate at `pos0 + t` | 1 arm |

**The negative controls were observed**, because an arm that cannot fail is not
evidence: with `kv_append` reverted to `pos0 + t` both write arms print
`CHANGED`; with `rope_rows` removed the tool prints
`unsupported on this backend (refused, not guessed)` and fails; and with the
pre-session sparse addressing restored the run **segfaults**, which is the
out-of-bounds the fix removed.

That segfault was a real latent defect, not a test artifact:
`attn_sparse_rows` formed its address as `(row_kv_base + j) * pos_stride` while
every other path uses `row_kv_base + j * pos_stride`. It was harmless only
because no caller sets `row_kv_base` yet -- i.e. it would have failed exactly when
B2.3 started using the field. Fixed, and the arm above is what keeps it fixed.

**What the engine does now (1-3 done):**

1. **N KV regions.** `EngineConfig::n_slots` (1 by default, `--slots N` on the
   server) sizes every layer's plane for N contexts and adds a per-row slot
   table: slot `s` owns rows `[s*kv_cap_, (s+1)*kv_cap_)`, and row `t` addresses
   its own region through `AttnDesc::row_kv_base`/`row_pos` -- the fields the
   kernel side already had. The base is built per layer, in that layer's own
   units, because `pos_stride` is per layer and counts BYTES on an int8 cache.
   `kv_pos_` stays the single-sequence cursor: nothing in a batched pass reads
   it, and the batch leaves it at the furthest slot's position.
2. **A batched decode step.** `Engine::generate_batch()` prefills one sequence at
   a time (a prefill's rows are consecutive positions of ONE sequence) and then
   decodes the whole active set in one `forward` of K rows, each row at its own
   position, reading its own logits row and sampling with its own `Sampler` and
   `TokenEmitter`. A sequence that stops is dropped from the next step without
   disturbing the others. Per-row arithmetic is the same kernels in the same
   order as a solo run, which is what makes the tokens identical rather than
   merely similar (see the test below).
3. **A scheduler in the server.** `OpenAiService` starts a scheduler thread when
   the engine has more than one slot and `batch_refusal()` is empty. Connections
   queue a request and wait; the thread takes up to `n_slots` of them, decodes
   them in one pass, and each connection streams its own deltas from its own
   sink. A request that arrives during a pass waits for the next one -- it is a
   wave scheduler, not iteration-level admission, and the doc says so because
   the difference is visible under load. Prefix reuse stays a serial-mode
   feature: with one region per sequence there is no "the cache" to cut back to,
   so a batched request reports `cached_tokens: 0` rather than claiming rows it
   does not hold.

**4. Per-row addressing in the latent-attention family.** `mla_attention`,
`mla_shared_attention` and `mla_shared_attention_compressed` each own their cache
addressing, so the per-row tables had to reach them or a DeepSeek-class file
stays serial. Two pieces:

* the descriptors carry `row_pos`/`row_kv_base`, in `MlaDesc` and
  `MlaSharedDesc` alike. **Both null means `pos0 + t` and region base 0** -- every
  existing caller is bit-identical, and the CPU kernels resolve the row through
  one pair of accessors (`mla_pos`, `mla_row_base`) instead of four copies of the
  arithmetic;
* the compressed path's *state* is per sequence, not per layer: the compressor
  ring, the pooled keys and their positions, the indexer's ring and its pooled
  keys, and the block count. Those buffers are now `n_slots` spans, indexed by the
  row's slot, and the per-token loop keys the ring (`pos % ratio`, `(pos/ratio)%2`)
  and every pooled position off **that row's** position rather than `pos0 + t`.
  Pooling still runs after the row's own attention, which is what keeps a block's
  pooled key invisible to its own tokens.

The negative control is the refusal itself, and it is measured, not assumed:
`Backend::mla_multiseq_supported()` defaults to **false**, and `batch_refusal()`
keeps the MLA line conditional on it. Two arms in `test_batch_decode` wrap the
CPU backend (which answers true) and force the answer to false: arm 0 keeps
per-row attention and drops the latent cache, and the wave is refused with the
latent-specific reason plus a `FinishError` per request; arm 1 drops both, and the
generic "does not address the KV cache per row" fires. A backend that cannot
address the latent cache therefore cannot be batched by accident -- the only
thing that turns it on is a backend that implements the tables.

Refusal stays by name for what genuinely has no per-row form: diffusion,
recurrent state, block indexers (QSA/MSA), hyper-connections, PLE windows, the MoE
route predictor, draft heads, learned positions, ALiBi and a paged KV tier.

**Verified in the tree, 2026-10-31** (this is what the summary line above used to
contradict): `Backend::mla_multiseq_supported()` defaults to false at
`include/krk/backend.hpp:1167` and the CPU backend overrides it to true at
`src/backend_cpu.cpp:1937`; the per-slot spans are `src/engine.cpp:1805-1840`
(`comp_state_kv_`/`comp_state_sc_`/`comp_pooled_`/`comp_kpos_`/`idx_state_*`/
`idx_key_`, each `n_slots` copies, `comp_nblocks_` `n_layer * n_slots`); the row
resolution is one pair of accessors, `mla_pos`/`mla_row_base`, at
`src/backend_cpu.cpp:341-350`, used by every latent op; and the pass -- not just
the refusal -- is asserted by `mla_batch_arm` in `test_batch_decode`, which runs
a wave against a V2-lite, a V4 shared-latent and a V4 CSA+HCA fixture and
compares each slot's tokens to that sequence's solo run.

**Measured.** Both arms are in `test_batch_decode` (86 checks, in the default
gate) and `test_batch_throughput` (serial group):

* exactness: one wave of four prompts with different `max_tokens` produces each
  sequence's solo tokens, text and finish reason **exactly** -- and so do the
  four concurrent HTTP requests driven through `OpenAiService`, which is also
  what proves the scheduler does not deadlock;
* exactness across the latent-attention families: the same wave, three slots
  taken out of four, against a V2-lite fixture (`mla_attention`), a V4
  shared-latent fixture (`mla_shared_*`) and a V4 CSA+HCA fixture (ratio 4 with
  its indexer plus ratio 128, i.e. the compressed ring, the pooled keys and the
  indexer's own pool) produces each sequence's **solo tokens exactly**. These are
  the arms that fail if any of `row_pos`, `row_kv_base` or the per-slot state is
  wrong -- the first two produced fluent, wrong text before the fix, not a crash;
* aggregate throughput, 4 sequences x 24 tokens, 4 layers x 768 wide, 8 experts
  top-2, median of 3:
  - every expert resident (compute-bound): **1.02x** -- parity;
  - a squeezed expert budget (transfer-bound, the state a 4 GB-on-disk model is
    always in): **1.08x-1.27x** across runs.

  So batching on this hardware buys in proportion to how memory-bound a step is,
  and is neutral when it is not. That is the honest reading of the number, and
  it is why the gate asserts "not slower" (0.95x) rather than "faster": the arm
  that would be a defect is batching costing throughput, and the size of the win
  is a property of the working set, not of the scheduler.

*Done when:* 4 concurrent requests hold aggregate tok/s above one-at-a-time with
each response identical to its solo run. **Met, with the throughput caveat stated
in the numbers above:** exactness is met and gated for every family the engine can
batch; the aggregate win is real only where the step is transfer-bound and is
parity when every expert is resident.

### Where the wave's time actually goes (measured, not assumed)

The expert *grouping* this section used to call "the remaining lever" is already
in the engine and has been since the grouped plan landed: the per-expert loop
buckets the layer's `(token, slot)` pairs by expert, gathers one block of `m`
rows per expert, and runs ONE gate/up + one down per expert (`plan_off_`,
`gather_rows`, `gemm_group`). A wave therefore already reads each expert's
weights once for all of its rows. Two further things follow from the width sweep
in `test_batch_throughput` (8-expert fixture, 64 MB expert budget, median of 3):

| wave | serial | batched | aggregate | per-row cost of a step |
|---|---|---|---|---|
| 4 | 85-93 tok/s | 120-126 tok/s | **1.35x-1.41x** | 4.6 ms |
| 8 | 86-90 tok/s | 92-102 tok/s | **1.07x-1.13x** | 2.3 ms |

* The wave's expert traffic is the **union** of its rows' experts, not the sum:
  `(width * k) / union`. Over 8 experts with k=2 that saturates fast -- about
  1.25x at width 4 and about 2x at width 8 -- so the win is bounded by the pool,
  not by the code.
* **The win is not a compute artifact, and it is not launch overhead either.**
  The same sweep on the same shape with Q8_0 experts (the int8 dot path, which
  is ~1.7x faster per token than the fixture's f16 one) keeps the win and even
  grows it at width 8:

  | experts | serial | batched | aggregate |
  |---|---|---|---|
  | f16 | 75 / 63 tok/s | 101 / 87 tok/s | 1.34x (w4) / 1.38x (w8) |
  | Q8_0 | 130 / 131 tok/s | 144 / 168 tok/s | 1.11x (w4) / 1.28x (w8) |

  So batching pays for something that survives a 1.7x faster dot path -- weight
  traffic and per-step fixed cost -- while the absolute number nearly doubles
  when the weights are 8-bit. The width-8 ratio moves between 1.07x and 1.38x
  across runs on this machine, which is the noise floor these claims live in.
* A group of exactly one row no longer pays its gather (`KRK_MOE_GATHER=0` is the
  control arm): the row already exists in the activation block, so the copy is
  skipped and the GEMMs read it in place. On this fixture the arm measures
  **inside the noise** (the two arms bracket each other) -- kept because it is
  strictly less work, not because a number here can see it. It matters in the
  regime it was written for: a wide expert pool (128-256) where a wave of 4
  rarely lands two rows on one expert, so every group is a group of one.

**The next lever, and why it is parked.** The candidate was a grouped MoE GEMM:
one call for all experts' gate/up over the stacked plan, one for the downs,
removing the per-expert call fan-out (~40 per layer). The component table now
answers it, and the answer is no:

| kernel row | calls | µs/call | GMAC/s |
|---|---|---|---|
| `gemm_q8_0_i8_r1` | 7083 | 116.9 | 5.0 |
| `gemm_q8_0_i8_rN` | 1494 | 300.4 | 4.9 |
| `gemm_f16_r1` | 4062 | 107.4 | 3.2 |
| `gemm_f16_rN` | 1440 | 277.3 | 4.7 |

(`KRK_PROFILE=1 KRK_KPROF=1 KRK_BATCH_BENCH_Q8=1 KRK_BATCH_BENCH_W=4`; 2 cores,
AVX-512 VNNI.) The rows>1 kernel runs at the **same** GMAC/s as the rows==1 one,
and the per-call times scale with the work inside them -- so the expert phase is
compute-bound at ~5 GMAC/s, not fan-out bound. Fewer, larger calls would remove
dispatch the table says is not there. Under `KRK_CPU_Q80_WSC=1` the serial arm
gains 13% (150 -> 170 tok/s) while the wave does not move (1.11x -> 1.01x), which
is consistent: the fast Q8_0 kernel changes the rate of a GEMV, not the number of
calls.

**Routing is not the next lever -- and the "30% of decode" reading of it was a
misread.** The `[stats ] components` table nests: `routing` contains the expert
phase, and the table says so in its own footer (`nested excess ... spans nest`).
Read flat it puts routing at 30.7% of decode and 121 us per op. The `KRK_TIME_HOST`
spans, which do not nest that way, put the same run at:

| span | calls | ms | ms/call |
|---|---|---|---|
| `moe.route.batch` | 288 | 431.0 | 1.497 |
| `moe.route.rest.batch` (its child) | 288 | 428.3 | 1.487 |
| `moe.experts.batch` (child of that) | 288 | 427.2 | 1.483 |
| `moe.route.gemm.batch` | 288 | 2.2 | 0.008 |
| `moe.route.fetch.batch` | 288 | 0.1 | 0.000 |

So the router GEMM is **0.5%** of the routing span and the logits readback is
0.02%; everything left of the span after removing the expert phase is the
softmax, the top-k and the plan build, and together they are **~1.5 ms over 288
layer-visits = 5 us each**. Routing is not where a decode step goes, in either
direction: at 5 us it is not worth a change, and the 121 us figure it replaced was
an artifact of adding the experts in twice. Prefill stages are the same shape
(`routing` 34.4% against `experts` 34.1% of that phase -- again one contains the
other).

**What the time IS.** Every remaining row is a GEMM at **~5 GMAC/s**, which on
AVX-512 VNNI is 3-5% of what the instruction can do. Two facts bound it: `r1` and
`rN` run at the same rate (so it is not a per-call dispatch cost), and
`KRK_CPU_Q80_WSC=1` moves the serial arm 13% (150 -> 170 tok/s on the Q8_0 bench)
without moving the wave, so the fast int8 kernel is a rate change on a GEMV.

**Is 5 GMAC/s a defect? Measured: no.** A Q8_0 decode GEMV reads ~1.06 bytes per
MAC, so 5 GMAC/s is ~5 GB/s of weight streaming, and `tools/bench_gemv_scale`
(shipped path, 71.3 MB of weights, n_in 4096, larger than the 54 MB L3) measures
this host's shipped kernel at 15-23 GB/s depending on how busy the box is. The
fixture's 5 GB/s is inside that range, so there is **no evidence of a 4x engine
defect** here -- and the earlier claim that the expert phase runs "at a fraction
of what the instruction can do" says nothing about whether *this box* can do
better.

**The tool was measuring itself, and it was order-biased.** Its first revision
ran each arm to completion once, in a fixed order: with 71.3 MB of weights against
a 54 MB L3, whichever arm ran first measured a cold stream (V0 10.10 GB/s cold
against 18.26 warm, same process). The verdicts therefore disagreed between runs
and a previous note quoted `1.80x/1.86x` for the f32 table by pairing a first-pass
ratio with a second-pass one. The harness now runs every arm once per round in a
rotation that moves with the round and reports the MEDIAN, with one untimed warm
pass first. Under it the arms are stable, and the answer is not the table:

| arm | extra memory | median GB/s | vs shipped |
|---|---|---|---|
| V0 shipped (byte-pluck) | — | 20.5 | 1.00x |
| V1 scale work removed entirely (bound, not implementable) | — | 25.3 | 1.23x |
| V2 f32 scale table | +11.8% | 21.9 | 1.07x |
| V3 f16 scale table | +5.9% | 23.5 | 1.14x |
| **V4 two-load pluck** | **0** | **23.0** | **1.12x** |

So the prebuilt tables buy **less** than simply assembling the same scale bytes
better: V4 replaces four byte loads and three shift-ors with two 16-bit loads and
one shift, touches no new memory, and reaches most of the 1.23x bound. It is what
ships. (Corrected: the earlier "+94% memory" figure for the f32 table was wrong by
a factor of eight -- 64 B per 8-row tile per block pair is 8 B per row per pair,
i.e. +11.8% of the weights, which the tool now prints.)

**The prefetch distance was 16 and should be 8.** V4 explained only part of it, so
the tool grew a V5 arm that calls the REAL kernel (`dot_q8_0_q8_r1xN`) instead of a
copy kept inside the tool -- a measurement that cannot drift from the shipped
body. V5 then showed no gain, which pointed at the one thing the tool's copies do
not have: the explicit prefetch. One variable, five processes per setting, medians
of nine interleaved rounds inside each:

| `KRK_CPU_GEMV_PF` | 0 | 6 | **8** | 12 | 16 (was default) | 32 |
|---|---|---|---|---|---|---|
| V5 GB/s | 17.9 | 21.5 | **21.8-22.5** | 20.8 | 17.2-19.4 | 14.0 |

The prefetch is worth having (PF=0 is ~20% behind) and a 16-block distance was
16% behind the 8-block one: far enough ahead that the lines are evicted again
before the stream arrives. The default is now 8, which also matches the width of
the tile the loop is built around, and `KRK_CPU_GEMV_PF` still overrides it.

**Together, on the shipped kernel, 1.17x-1.24x** (V5 against the old byte-pluck
copy, four runs). Stated with its limit: the engine's own 768-wide benchmark
fixture is L2-resident and shows **no** change, because neither prefetch distance
nor scale assembly matters when the weights are already in cache. The win is at
the shape a real model has -- n_in 4096, weights larger than L3 -- which is the
shape this tool measures.

### B3 -- A trace file carries no identity -- DONE

`Engine::dump_row` appended stage lines with no header, no run identity and no
end marker, so two runs in one file could not be told apart -- and every "A/B on
the same prompt" comparison in this project reads those files.

**What the file carries now**, one run per block:

```
# kraken-dump v1 warn pid=<pid>: the previous run in this file left no end marker ...
# kraken-dump v1 begin pid=<pid> t=<unix> model=<path> bytes=<size> ctx=<n> chunk=<n> mode=<full|row>
<stage lines as before; in KRK_DUMP_FULL the %s L%02d N=... line>
# kraken-dump v1 end pid=<pid> t=<unix> lines=<stage lines this run wrote>
```

The `warn` line appears only when the file's last line is not an end marker, i.e.
when the previous run was killed or truncated, and it names the pid that left the
file in that state. Every marker is `#`-prefixed, so the existing readers (which
skip on the `<name> L<nn> n=` prefix) are unaffected. `dump_open()` runs at init,
`dump_close()` is the first statement of `shutdown()`, and both are idempotent.

*Test:* `test_dump_identity`, **17/17** -- three runs into one file: 1 header/1
end per run, `lines=` equal to the stage lines it encloses (36, then 36 for the
second run while the file holds 72), a truncated file detected by exactly one
warning whose pid is live and positive, and the cut run not counted as complete.

### B4 -- TSAN: the listen descriptor's lifetime -- DONE

The defect, confirmed: `stop()` computed `listen_fd_` (a plain `int`) and joined
the loop on `running_`, then `shutdown()`+`close()`d the descriptor, while
`run()`'s accept loop read `listen_fd_` **once per iteration**. That is a data
race on the field, and a worse hazard underneath it: the number could be closed
by one thread and handed by the kernel to another thread's `socket()` before the
accept loop loaded it again, so `accept()` could take a connection from a
different subsystem entirely.

**The ownership rule now in the code.** The descriptor belongs to the thread
inside `run()` from the moment it starts until it returns; that thread is the
only one that closes it after `listen()`. `stop()` raises a flag and returns --
it never touches the number, so there is no window in which it acts on a
descriptor somebody else already closed. The loop waits for readability with a
bounded `poll()` (100 ms) instead of blocking in `accept()`, so it notices the
flag without needing the socket shut down underneath it; the listener is
non-blocking so a reset connection between `poll()` and `accept()` cannot park
it, and `EAGAIN` is a `continue`. The close itself is a
`compare_exchange_weak(fd, -1)`, so of `run()`'s exit path and `stop()`'s
"nobody ever ran" path exactly one closes it. Accepted sockets are explicitly
put back into blocking mode rather than relying on neither POSIX nor Winsock
documenting inheritance the same way.

*Verification, with the negative control.* `tools/tsan_http_listen.cpp` drives 12
start/stop cycles with four concurrent clients and `stop()` racing the accept
loop, built with `-fsanitize=thread` (`target kraken-tsan-http-listen`).

| build | TSan result |
|---|---|
| pre-B4 `src/http.cpp` (`git show 33189b5:src/http.cpp`) | **data race**: `stop()` http.cpp:510 vs `run()` http.cpp:481 |
| post-B4 `src/http.cpp` | **clean**, three runs, 39-46 of 48 responses answered |
| `test_http_listener_lifetime` | **15/15**, three consecutive runs |

The harness was written against the old header first and it reported the race on
the `listen_fd_` write/close against the `accept()` read -- that is the negative
control, not an argument. The in-suite group covers the same properties without a
sanitizer: the loop returns after `stop()` from another thread, the port binds
again (so the descriptor was closed exactly once), `stop()` before `run()` closes
it, a second `stop()` is a no-op, and `run()` after `stop()` returns instead of
accepting on a dead number.

## C. Closed -- do not reopen

| item | why it is closed |
|---|---|
| prefill long-group format (`Q8_0_S4`, g4) | 1.43x speed, 1.25x/1.16x error; 1.00x for the compromise design. A6 |
| 16-row `x16_vnni`, 64-B vector scale table, 16-row 2-float table, f16-expanded scales | all measured worse (49.9, 10.1, 24.0 GMAC/s) |
| recurrent-state rollback for speculation | **implemented** -- snapshot `mtp_rec_snap_`/`mtp_conv_snap_` at round start, restore on reject, spec off if the snapshot cannot allocate |
| `qwen4exp` as an unsupported family | it is `Yes` now; `test_qwen4exp_whole_graph` pins the sentence |

---

## The list, in the order it should run

Status is per bucket as of 2026-10-31.

| # | step | state |
|---|---|---|
| 1 | **A1** -- spec gate on a real `qwen4exp` trunk + head; move `qwen4-mtp` | *needs files* |
| 2 | **A2** -- same for `deepseek4` + dspark; move `deepseek4-dspark` | *needs files* |
| 3 | **A7** -- the 30B-A3B run on 16/24/32 GB; the only end-to-end test of the promise | *needs a file and a box* |
| 4 | **B1.1-B1.2** -- L0..L7 recorded on this node, gap attributed op by op | **done** |
| 5 | **B1.3** -- fix the dominant term | **closed by measurement**: the dominant term is the instruction mix; pv8 (the last repack) is 7-9 % slower. See *B1* |
| 6 | **B1.4** -- GMAC/s -> tok/s projection | **done** (numbers under *B1*): 48-49 GMAC/s is ~16 tok/s of prefill per core at ~3 GMAC/token, so 4 cores is ~64, and the gap to 200 is arithmetic no kernel change closes |
| 7 | **B2.1** -- server cancellation | **done**, `test_cancel_and_prefix` |
| 8 | **B2.2** -- prefix reuse across requests (the record is the run, not the prompt) | **done**, same group |
| 9 | **B2.3** -- continuous batching | **done**: slots, batched decode, the wave scheduler and per-row addressing in all three latent-attention layouts are in and gated (`test_batch_decode`, 86 checks -- MLA lite, V4 shared-latent and V4 CSA+HCA each produce their solo tokens exactly); refused by name only for what has no per-row form; aggregate win 1.08x-1.27x transfer-bound, 1.02x resident |
| 10 | **B3** -- trace identity and truncation reporting | **done**, `test_dump_identity` 17/17 |
| 11 | **B4** -- TSAN listen-descriptor lifetime | **done**, `kraken-tsan-http-listen` clean, old code reproduces the race |
| 12 | **A3** -- DFlash2 head on its own pairing, then upstream logits | *needs files* |
| 13 | **A5** -- OLMoE against an independent oracle | *needs a file and an oracle run* |
| 14 | **A4** -- `gemma2` (and the vision rows, if wanted) on real files | *needs files* |

**What a session without files can still move:** nothing in B2.3. Item 4
landed with the per-row tables and is gated end to end -- three latent-attention
layouts, batched against their own solo runs -- so the wave is not a code item
any more. Everything else in the table either needs a checkpoint or a second
machine, or is closed by a measurement that now exists.

*Corrected 2026-10-31.* This paragraph used to name item 4 as "the one part of
it that is refused rather than implemented", and the order table repeated it.
Both were true when B2.3 was written and stopped being true when the per-row
tables landed. The cost of the drift is on the record because it is the point:
a session read the claim, treated a finished feature as the next piece of work,
and spent an hour re-deriving the code before finding the tests. A stale line in
this file is not cosmetic.
