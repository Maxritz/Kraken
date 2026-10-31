# Two open items, split into steps that can each be gated

Status 2026-10-31, after PR #85. Everything below assumes `main` at `49dfb25`.

**Progress:** item 1 is **done through 1.8** as far as a workspace without a
checkpoint allows -- the block runs, drafts a chain, and its step-1 acceptance is
reported and gated; what is left is the real-artifact run. Item 4 is **closed**:
4.1 and 4.4 were measured and the format does not ship (1.43x speed, 1.25x
error), so 4.2/4.3/4.5 should not start.

The point of this file is that each step has a **done-when** that can fail. A step
that cannot fail is not a step, it is a description.

---

## Item 1 — the `qwen4-mtp` / `deepseek4-dspark` block

### What already exists (checked, not assumed)

| piece | where | state |
|---|---|---|
| nextn block struct + loader | `MtpBlock`, `Model::load_mtp_block` (src/model.cpp:5016) | loads a **llama-style** block (`blk.N.attn_*` + `blk.N.ffn_*`), in-file or `--mtp FILE` |
| block forward | `Engine::mtp_block_forward` (src/engine.cpp:5353) | runs `[enorm(e) | hnorm(h)] -> eh_proj -> attn -> ffn`, packed query gate, per-head QK-norm, rope |
| trunk hidden capture | `Engine::mtp_prime`, `mtp_hidden_`, `mtp_h_`, `mtp_e_` | the trunk's last hidden row is already captured per position |
| verify + accept accounting | `generate_speculative`, `--mtp-spec`, `--mtp-probe` | present and used by the DFlash paths |
| expert paging for a block | `Engine::mtp_pin_experts`, `ExpertCache` | present |
| hyper-connections | `src/hc.cpp`, `hc_mix*` | present, for the **stack**'s layers |
| QSA indexer | `Engine::qsa_select_block` (src/engine.cpp:3210) | present, but bound to `LayerWeights` + a layer index |
| graph spec for this head | `docs/AUDIT-mtp-drafter.md` | the vLLM-transcribed graph, pairing rules and traps |

Missing is the **qwen4exp shape of the block**: two separate input projections
instead of one concat `eh_proj`, a hyper-connection residual around attention and
around the MLP, a QSA-indexed attention with its own KV, a 512-expert MoE whose
banks are stored **fused** (`gate_up_proj`), `block_inject_weight`, and the HF
`mtp.*` spelling. Not the verify loop, not the capture, not the expert pager.

### Steps

**1.1 — Pin the schema against the reference** (½ session)
Read vLLM's `qwen4_exp` MTP layer (the reference `AUDIT-mtp-drafter.md` cites)
and the REAP-288 inventory; write `docs/MTP-HEAD-SCHEMA.md`: all 31 tensors, each
with name, ggml shape, dtype and the op that consumes it, plus the pairing.
*Done when:* every tensor in the file has a row in the table and every row names
a consumer in our code (existing function or a step below).

**1.2 — Extend `MtpBlock` for the shape** (1 session)
Add: `fc_embedding`, `fc_hidden` (mutually exclusive with `eh_proj`), the
hyper-connection mixer (4 lanes, low-rank 320: `input_mix_down/up`,
`final_mix_down/up`), `block_inject` `[hc, hc*n_embd]`, the final mixer, the
indexer tensors (`indexer.{q,k}_proj`, `{q,k}_norm`), and the block's own
`n_lane`, `n_hc_rank`, `hnorm_mode`.
*Done when:* a `dense_prefix` enum selects {ConcatEhProj, FcPair} and the struct
has no field it does not read.

**1.3 — HF `mtp.*` name map + geometry check** (½ session)
Map `mtp.fc_embedding.weight`, `mtp.pre_fc_norm_embedding.weight`,
`mtp.fc_hidden.weight`, `mtp.pre_fc_norm_hidden.weight`,
`mtp.hyper_connection_mixer.*`, `mtp.layers.0.{self_attn,mlp}.*`,
`mtp.block_inject_weight`, `mtp.final_mix_*`. Every shape checked **against the
trunk**: hidden, lanes, rank, 512 experts, FF 640, shared 640, head/kv dims,
vocabulary (the head is the trunk's — there is no `output.weight`).
*Done when:* a synthetic 31-tensor file loads, and three wrong-geometry files are
refused by tensor name (hidden, lanes, expert count).

**1.4 — Fused expert banks** (½ session)
`gate_up_proj` is `[512, 1280, 2560]`: one bank holding gate (640) and up (640).
Bind it as two 3-D views at the expert-plane granularity, not a copy — extend
`Gguf::add_row_slice`'s idea to a 3-D source or teach the expert path a strided
bank with an offset. A 4-strided read is also the only version that keeps the
hot-expert cache able to page one expert at a time.
*Done when:* the loaded banks' first and last bytes match the file's own slices
for a sampled expert, read back through the dequantiser.

**1.5 — Dense prefix, host f32** (½ session)
`e = fc_embedding(rms(embed(tok)) * (1+w))`,
`h = fc_hidden(hidden_norm(R) * (1+w))`, `R = h + e`. Two `hnorm` modes: one RMS
over the 4×2560 lanes, or four per-stream RMSes (`KRK_MTP_HNORM`), because the
reference carries the switch and the artifact decides.
*Done when:* outputs are finite; the two modes differ on the first float (a
recorded vector); scaling `R` by 0.5/2 scales `h` linearly.

**1.6 — The block on the device** (2-3 sessions, the whole project)
(a) hyper-connection around attention, reusing `src/hc.cpp` with block-local
state; (b) QSA-indexer attention with the block's own KV — factor
`qsa_select_block` into a parameter struct + state so the stack and the block
share one selector; (c) MoE: 512 experts top-10 through `ExpertCache`, shared
expert, the block router on the trunk's gating convention; (d) final mixer,
`R += y * 2 * sigmoid(inject / hc)`, logits through `Model::out_head`.
*Done when:* `--mtp-probe` on a prompt puts the block's argmax on the trunk's
next token for **≥ 60 %** of cells on greedy text. That is the only cheap oracle;
a wrong pairing or a wrong norm runs, drafts and is rejected every time.

**1.7 — Proposer + the speed gate** (½ session) — **done on the synthetic gate,
speed half BLOCKED on an artifact** (see 1.7's gate below). The measurement
itself is no longer prose: `scripts/mtp_spec_gate.sh` runs both arms and fails
on either half.
`--mtp-spec` with per-step acceptance and wall time. The chain is a buffer, not
an API change: `Engine::mtp_chain_` holds `[n_lane * n_embd]`, the head's forward
copies it in when it is proposing from its own previous step and writes it back
after the final mixer, and the proposer asks for it on `d > 0`.
*Done when:* acceptance ≥ 0.6 at step 1 **and** decode wall time below plain
decode on the same prompt. A drafter that is accepted but costs more than it
saves is a regression with a good story.

**1.8 — Rows, docs, suite, and the measurement that is left** (½ session)
`qwen4-mtp` becomes `Yes` (Draft role) **on evidence from a real artifact**,
`deepseek4-dspark` follows only if its own attention is wired (same interface,
different block — reuse, do not share code); the groups exist; the baseline is
re-recorded. The half that cannot be done here is the run: no checkpoint is in
this workspace, so the row stays `Partial` and the run is the gate. Turns into
one command once a file is on disk — see *1.8's runbook* below.

**Total:** 5-7 sessions, of which 1.6 is 60 % of it.

### Item 1 status

| step | state | evidence |
|---|---|---|
| 1.1 | **done** | `docs/MTP-HEAD-SCHEMA.md` — 31 rows, each with shape, format and consumer; the three not-in-the-31 tensors are listed with why |
| 1.2 | **done** | `MtpBlock` gained `hf`, `n_lane`, `n_hc_rank`, `n_index_head`, `index_dim`, `HcMixer{hc_attn,hc_ffn,hc_head}`, `fc_embedding`/`fc_hidden`/`eh_proj_split`, `index_{q,k}`+norms, `exps_gate_up` |
| 1.3 | **done** | `Model::load_mtp_hf_block` (src/mtp_head.cpp) + `test_mtp_head_qwen4`: a synthetic 31-tensor file loads, and the wrong-width bank / half-written prefix / mismatched `down_proj` are refused by tensor name |
| 1.4 | **done** | `Gguf::add_expert_plane_slice` + `GgufTensor::plane_stride`/`plane()`; the expert path reads through `plane()` in all five places it used to compute `data + expert * per`. `test_mtp_head_qwen4` compares **every** expert's gate and up view against the source rows with `memcmp`, not a sample |
| 1.5 | **done** | `include/krk/mtp_forward.hpp` + `src/mtp_head.cpp` (`mtp_host_prefix_load`, `mtp_dense_prefix`), group `test_mtp_head_prefix`: 18/18 |
| 1.6 | **done** | `Engine::mtp_hf_block_forward` + `mtp_attn_body` (the attention body factored out of the llama-style block so the two shapes cannot drift); group `test_mtp_hf_block_qwen4`: 17/17 |
| 1.7 | **done** (synthetic); speed unmeasured | `mtp_chain_` + `from_chain` on both block forwards; `--mtp-spec` no longer refuses this head; group `test_mtp_hf_spec_chain`: **18/18** — identical text to plain greedy (8 of 8 tokens), step-1 acceptance **3 of 3**, **7 of 12** proposals kept by the pass-through head and **0 of 36** by a live-attention control (which still emits the plain text: verification, not luck). Found and fixed on the way: a round emitted the block's own EOS proposal and ran on past it -- the accept loop in **five** generate loops now stops *before* emitting it, as the plain loop always did |
| `deepseek4-dspark` | **wired, runs** (synthetic); row still `No` | the block is loaded as a **stack layer** (`LayerWeights`) and its body is the trunk's own -- `mla_shared_forward` + `moe_ffn` between `hc_begin`/`hc_end`, then the stack's head mixer; group `test_mtp_sidecar_mla` **19/19**: it loads, drafts (23 proposals, 0 accepted with random weights) and emits the plain greedy text. A real checkpoint run is what moves the row. See *deepseek4-dspark* below |
| 1.8 | **partly** | step-1 acceptance is reported by the engine (`step1_hits()/step1_tries()`) and gated by the group; `scripts/mtp_spec_gate.sh` turns the real check into one command; the arch row stays `Partial` because no checkpoint has been run; baseline re-recorded at 171 groups |

**1.5's gate in this file was wrong, and is corrected here.** It read "scaling
`R` by 0.5/2 scales `h` linearly". It cannot: the prefix RMS-normalises `h`
before the projection, so scaling `R` leaves the result *invariant*, not linear.
The invariant that actually holds -- and that the group now checks -- is scale
invariance to 1e-4. The other two gates stand as written: outputs finite, and
the two `hnorm` modes differ on a recorded input.

The strongest check in that group is not on the list above: the fused `eh_proj`
and the `fc_embedding` + `fc_hidden` pair are the same matrix split by column, so
the group builds both spellings from one matrix and requires the two computations
to agree element-for-element. A concat order of `[h | e]`, or a gamma applied
per-lane in one form and not the other, fails it.

Two deviations, both deliberate:

* **`add_row_slice` was not extended** — a 3-D bank needs a plane stride, not a
  row offset, so the stride became a field on `GgufTensor` (`plane_stride`) with
  one `plane(e)` accessor. That is also what a future fused qkv bank needs, and
  it kept the five existing call sites readable instead of adding an offset
  parameter to each.
* **No `dense_prefix` enum** — there are exactly two forms and the struct
  already carries one of them as a non-empty `QuantTensor`, so `eh_proj_split`
  (bool) plus the two `fc_*` tensors says the same thing without a third name
  for a state that cannot occur.

**1.6's gate, and what replaced it.** The plan asked for `--mtp-probe` on a real
artifact with an argmax agreement of ≥ 60 %. There is no artifact here, so the
gate is a synthetic head built to make the answer EXACT: attention output zero,
MoE zero, prefix the identity on the hidden half. The block is then a
pass-through — its lane stack after the block is the trunk's own — so its argmax
must be the trunk's argmax on every probed cell, not 60 % of them.
`test_mtp_hf_block_qwen4` measures exactly that (`MATCH`, target rank 1) and then
demands that it can fail:

| arm | draft logit | target's rank in the block's logits |
|---|---|---|
| pass-through | 0.6485 | **1 (matches the trunk's token)** |
| patterned gamma in every lane | 0.5446 | 1 |
| live attention weights | 0.2630 | **49** |
| live routed bank | 0.5498 | **25** |

The second arm is deliberately a *pattern*, not a per-lane scale: the head mixer
RMS-normalises each lane first, so `gamma = 1.5` on a whole lane cancels exactly
and the control would prove nothing. Both live arms move the row far enough to
push the trunk's token off first place, which is what says the equality in the
first row is caused by the wiring and not by the comparison being insensitive.

The pass-through is not exact to the last bit: the prefix's per-lane RMS and the
mixer's per-lane RMS compose, and `rms(rms(x)) = x / sqrt(r²(1+ε) + ε²)` differs
from `x / sqrt(r² + ε)` by about `ε(1/r² − 1)/2`. On a real checkpoint, where the
lanes have unit RMS, that is ε-level; on this fixture's small activations it is
the 0.042 logit difference in the first row. It is recorded here because it is
the one place a future reader will look for an exact equality and find 7 %.

**1.6(b) is the reference's own fallback, not a shortcut.** The block runs dense
attention over its own KV, and the four indexer tensors are loaded, geometry-
checked and *not* used for selection. That is what llama.cpp's qwen4exp MTP graph
does: it builds the indexed attention only when `layer.index_k_proj &&
hparams.dsv4_compress_ratios[il] > 0`, i.e. when the enclosing model declares a
block budget for that layer. A standalone 31-tensor head declares nothing of the
sort (`tools/mtp_extract.py` writes no ratio key), so dense is the correct
behaviour for the artifact — and inventing a budget would be a guess that costs
acceptance rather than a feature.

**1.7's gate: the chain, and the end token it found.**

The reference feeds the whole `hc * n_embd` lane stack into the next draft step.
`mtp_block_forward(m, pos, want_logits, from_chain)` now takes it from
`Engine::mtp_chain_` — written after the final mixer, read only for `d > 0` —
instead of the single `n_embd` row the llama-shaped path folds into `mtp_h_`,
and the `--mtp-spec` refusal for an HF head is gone.

The gate cannot be an acceptance rate without a checkpoint, so it is the same
strict property 1.6 used, moved into the loop: a pass-through head's first
proposal IS the target's token, so `--mtp-spec` must produce text identical to
plain greedy, end in the same place, and keep its proposals. The control is the
same head with live attention, which proposes the wrong token on purpose.

| arm | tokens | proposals kept | finish | wall |
|---|---|---|---|---|
| plain greedy | `8 3 3 3 3 3 3 3` | — | EOS | 1.1 ms |
| pass-through head, `--mtp-spec` | `8 3 3 3 3 3 3 3` (identical) | **7 of 12 (58 %)** | EOS | 2.3 ms |
| live-attention control, `--mtp-spec` | identical | **0 of 36** | EOS | 2.3 ms |

The third row is the one that makes the second mean something: a drafter that is
wrong 36 times out of 36 still produces the plain text, because verification —
not the drafter — decides what is emitted. The acceptance is therefore measuring
the chain, and only the chain. Step-1 acceptance (the pre-check, i.e. whether the
block's own first proposal matched the target) was **5 of 5 rounds**; the 58 %
is the aggregate over later steps, where this 2-layer fixture's logits are nearly
flat and a pass-through row lands on a tie.

The wall column is printed and deliberately NOT gated: a two-layer trunk with a
one-layer head and an eight-token output measures round overhead, nothing else.
The speed half of 1.7 needs the real artifact and the command is fixed now:

```
kraken -m qwen3.8-30b-a3b-q4.gguf --mtp qwen4-mtp.gguf --mtp-spec -n 128        -p "<a 2 KB prompt>" --stats
```

with `draft head:` report from the run (rounds, kept/proposed, tokens/round) and
`decode tok/s` against the same command without `--mtp`.

**The defect this gate caught.** A round emitted the end token. The block
proposed `</s>` at position 34, the target's verification agreed with it, and
the accept loop wrote it as a normal token — so the run carried on for three
more tokens and finished on the length instead of the end token:

```
plain : 8 3 3 3 3 3 3 3            finish=EOS
spec  : 8 3 3 3 3 3 3 3 2 3 3 3    finish=LENGTH      <- 2 is </s>
```

The plain decode loop tests the end token *before* emitting it; a speculative
round has to as well. Both sides are fixed now, in **all five** speculative
loops (draft-head greedy, MTP greedy, MTP sampled, DFlash greedy, DFlash
sampled): the propose loops drop an end-token proposal (the MTP and DFlash ones
did not), and the accept loops end the run with `FinishEos` if one ever reaches
them anyway. The group checks the finish code, not just the text — text alone
would have passed with the trailing `2` trimmed by a coincidence.

One fixture trap, recorded so it is not re-learned: the trunk and the head were
both built as `kraken-mtp-hf-spec.gguf`, and the head overwrote the trunk, which
the loader then refused with "missing core geometry" for a file it had just
written. Test fixtures that share a directory need distinct names.

**1.8's runbook: what the real artifact has to produce.**

The gate is two numbers and both now come out of the run. The engine reports
step-1 acceptance in its own summary line, because "acceptance ≥ 0.6 at step 1"
was unanswerable from `draft_accepted/draft_proposed` — that ratio mixes steps
2..k, where a short chain is not evidence about the head:

```
[info ] mtp spec: 5 round(s), 7 of 12 proposals kept (58.3%), step-1 3 of 3
        (100.0%), 2.40 token(s)/round | propose ... | verify ... | ...
```

`Engine::step1_hits()` / `step1_tries()` expose the same two counts, and
`test_mtp_hf_spec_chain` pins them on both arms: **3 of 3** for the pass-through
head, **0 of 9** for the live-attention control. A counter that always read
`hits == tries` would pass the first and fail the second, which is why both are
asserted.

The measurement itself:

```
scripts/mtp_spec_gate.sh <trunk.gguf> <mtp.gguf> 128 "<prompt>"
```

It runs the same prompt, greedy, one seed, with and without `--mtp --mtp-spec`,
parses `[stats ] decode ... tok/s` from both runs and the `step-1 H of T (P%)`
field from the spec run, prints the ratio and exits non-zero if either half of
the gate fails. It is checked here against a stand-in binary that prints canned
stats: the passing case (step-1 3/3, 120 against 60 tok/s) exits **0**, the
failing one (step-1 2/9, 60 against 120) exits **1** and names both halves. What
it cannot do is prove a real head accepts; that needs the 889 MB
`mtp-q2_0.gguf` (or any `qwen4exp-mtp` set) and a matching `qwen4exp` trunk.

What to record when it runs, in this file, before the arch row moves:

| field | where it comes from |
|---|---|
| step-1 acceptance, hits/tries | the `mtp spec:` line |
| proposals kept, aggregate | the same line |
| tokens/round | the same line |
| decode tok/s plain vs `--mtp-spec` | `[stats ] decode` in both logs |
| the text, both arms | must be identical under `--greedy`, as the synthetic group demands |
| the head's tensors as loaded | `--mtp-probe`'s log line, so a schema mismatch is visible before any speed number |

If step-1 acceptance lands under 60 % on a real head, the causes are ordered by
what this project has already measured: the paired-token convention (the head is
fed trunk residual `i` and token `i+1`, and an off-by-one pairing drafts a token
that is always one behind), the per-stream `hnorm` switch (`KRK_MTP_HNORM`), and
the dense-vs-indexed attention choice of 1.6(b). All three are switches or one
line in the loader, and all three were built to be testable in isolation.

---

## Item 4 — a prefill format whose scale group is longer *by construction*

### What is known (measured, not assumed)

* The shipped q8_0 kernel (W) sits at the V1 instruction-mix ceiling: 5 uops per
  64 MACs, `vpdpbusd`, nothing left in the mix (docs/GEMM-PERF-WALL.md).
* The only lever left is a longer scale run: **G2 ≈ 1.1×**, **G4 ≈ 1.19×** over
  W at 768×2048×512 — and **lossy** at q8_0's group size, because one f16 scale
  over 128 values cannot represent a row whose magnitude varies inside it.
* Therefore the format has to *carry* the longer run, not have it imposed at
  runtime. A requantisation is a measured dead end; do not retry one.

### Steps

**4.1 — Settle the layout on paper and in a bench, before engine code** — **done,
verdict: reject** (`tools/bench_prefill_fmt.cpp`, table in
docs/GEMM-PERF-WALL.md)
Two candidates, both measured with the existing `tools/bench_q80_gemm.cpp`
harness as a dequantised reference first:
 * **(a)** `q8_0_g4`: one f16 scale per 128 values (the known 1.19×, known lossy);
 * **(b)** `q8_0_s4`: 32-value blocks with the q8_0 scale **plus** one 8-bit
   residual exponent per 4 blocks — 1.03 B/value against q8_0's 1.0625, and the
   residual keeps the per-32 precision.
*Done when:* for both, the max per-element error and the logits delta against f32
on one real tensor row are recorded, and (b) is ≤ q8_0's own error. If neither
passes, the item stops here with that number recorded — a valid outcome.

**4.2 — Format plumbing** (1 session)
`DType::Q8_0_S4` (custom id 150, ours; state in the header that the file is not
llama.cpp-portable), `dtype_block_size`, `dtype_type_size`, `dtype_row_bytes`,
`dequant`, `dtype_supported`, and a quantiser + `--format q8_0s4` in
`tools/make_bench_gguf.py`.
*Done when:* a round-trip test pins the error bound and static asserts pin the
block size and row stride.

**4.3 — The kernel** (1 session)
Clone the W kernel to a v2 that hoists the scale run over 4 blocks (one load, four
uses) and applies the residual exponent; scalar fallback for other ISAs; select
via `KRK_CPU_Q80_KERNEL`.
*Done when:* bit-comparable to dequant-then-f32 within 1e-5 on a 64-row sweep,
and the scalar path agrees with the vector path.

**4.4 — Measure, then decide** — **done with 4.1: speed gate passes (1.43x),
error gate fails (1.25x weights x ~1.16x activations), so it does not ship**
A/B against W on 768×2048×512, best-of-5, median and spread, two runs.
*Done when:* **≥ 1.15×** over W **and** the synthetic model's logits move no more
than q8_0's already do. Missing the gate means the format does not ship and the
number goes into docs/GEMM-PERF-WALL.md.

**4.5 — Wire it** (½ session)
Loader accepts the type, the engine picks the layout per tensor, a suite group
covers load + generate + the error bound, docs updated.

**Total:** 2-3 sessions, with a real chance step 4.1 or 4.4 fails the gate.

### Item 4 outcome (measured, 2026-10-31)

4.1 and 4.4 were run together because the bench that settles the layout also
measures the kernel. Both candidates exist as kernels now:

| kernel, 768x2048x512, best-of-5, two runs | run A | run B |
|---|---:|---:|
| q8_0 (shipped) | 30.05 | 29.17 |
| 128-group weights, per-32 activations | 24.18 | 23.55 |
| **128-group weights, per-128 activations** | **43.64** | **41.49** |
| 128-group weights + per-32 exponents, per-128 activations | 29.58 | 29.53 |

**The speed belongs to the activation group.** The shipped kernel's per-64-value
cost is the scale-VECTOR build, and one broadcast per group is possible only when
`d_w * d_x` is a single number, i.e. when both sides share the run. Lengthening
the weight group alone is 0.80x. That is a correction to this plan, which
assumed the weight format was the thing carrying the longer run.

**The error is 1.25x (weights) x 1.16x (activations)**, and neither escape route
survives measurement: least squares does not beat the max scale on any of the
three row shapes (7.38e-3 against 7.40e-3), and the per-32 exponent ladder stays
at the long-group error (7.2e-3) because a power-of-two ladder is up to 2x
coarser than the block's ideal grid no matter how many bits it carries.

So 4.2 (the `Q8_0_S4` dtype), 4.3 (its kernel in the backend) and 4.5 (loader
wiring) are **not started and should not be**: they would add a format that is
1.00x in speed, worse in error, and not portable. The numbers are in
docs/GEMM-PERF-WALL.md with the three hypotheses this refuted, and the bench is
in the tree so the decision can be re-run rather than re-argued. What would
reopen it: a real artifact showing q8_0's error is far below what the model
tolerates -- the same missing artifact that keeps `qwen4-mtp` at `Partial`.

---

## Order

1. **1.1 → 1.4** first: they are the steps that can be wrong in a way the tests
   can see cheaply, and they unblock 1.6.
2. **4.1** in parallel is legitimate — it is a bench, no engine code.
3. **1.6** next, because it is the only step that turns into tokens/s.
4. **4.2 → 4.5** last -- and, as measured, not at all: 4.1+4.4 closed the item.

Neither item shares a line of code with the other; the order above is about risk,
not dependency.

**What is left, in order, and each one needs something this workspace does not
have:** (1) run `scripts/mtp_spec_gate.sh` on a real `qwen4exp` trunk plus its
`qwen4exp-mtp` head and record the table in *1.8's runbook* -- that is what moves
`qwen4-mtp` from `Partial` to `Yes`; (2) the same run on a `deepseek4` trunk plus
its dspark nextn block -- the body is wired now (`test_mtp_sidecar_mla`, 19/19),
so what is left for its row is the artifact and its acceptance number;
(3) the prefill format only reopens with an artifact that quantifies q8_0's
headroom.

---

## deepseek4-dspark: what it needed, and what it does now (2026-10-31)

The row was written as "an MTP head set that is not a model". Three things about
it were wrong, and each one hid the next.

**1. Its block is a stack layer, not a llama block.** Upstream's
`src/models/deepseek4.cpp` (`graph_mtp`) builds the nextn layer as ONE MORE LAYER
of the trunk: `il = n_layer`, the merged row `eh_proj([e | h_lane])` per lane,
then **the layer's own shared-latent attention between `hc_pre`/`hc_post`**, then
**its MoE + shared expert between the next pair**, then `cvec`, then
`hc_head` -> `shared_head_norm` -> `shared_head_head`. The earlier note in this
plan said it was "not a second hyper-connection block"; it is the opposite -- it
is a hyper-connection block because it is a *stack* block, and the tensor set is
a trunk layer's plus `nextn.eh_proj`/`enorm`/`hnorm`/`shared_head_norm`.

**2. So the port is reuse, not a new body.** `MtpBlock` gained `dspark`,
`dspark_layer` (a `LayerWeights`) and `run_layer`, and
`Model::load_dspark_mtp_from` loads the block's tensors into that layer --
including the five f16 staged tables the shared-latent ops read.
`Engine::mtp_dspark_forward` is wiring only:

```
lanes  = trunk's lane stack at pos (or the block's own last, from_chain)
per lane: rms(e,enorm) | rms(h,hnorm) -> eh_proj
hc_begin(L,attn) -> mla_forward(L, n_layer) -> add -> hc_end
hc_begin(L,ffn)  -> moe_ffn(L, n_layer)     -> hc_end
hc_head -> shared_head_norm -> output.weight
```

Every arithmetic step is the trunk's own function. That is the difference that
matters: a second MLA body inside `MtpBlock` is how two copies of one arithmetic
drift, and the drift reads as a mediocre drafter rather than as a bug.

**3. It needed a KV slot past the stack.** The block attends over latent rows
only it writes, so `Engine::init` charges one more plane
(`kv_cap * kv_row_bytes(mla_kv_lora)`) at index `n_layer` and `kvt_k_.base(n_layer)`
is what `mla_shared_forward` writes. Sharing a trunk layer's plane would
overwrite that layer's history with rows built from a different attention.

### Measured now (`test_mtp_sidecar_mla`, 19/19)

| arm | result |
|---|---|
| loads | `blk.1` read as a stack layer: 2 heads x 12 latent, 2 experts top-1 of 32, 2 Sinkhorn lanes, `eh_proj` 64x32 |
| runs | `--mtp-spec` on the trunk: **23 proposals**, 0 accepted (random-weight block), and the emitted tokens are **the plain greedy ones** |
| refuses | no `hc_attn_fn` -> named; `attn_kv` one column short -> both widths (32x11 vs 32x12); a lite-layout trunk -> refused by its own metadata before the block is read |
| geometric checks | a compressed (`compress_ratio != 0`) or hash-routed nextn block is refused by name: the body builds neither pooled keys nor a `tid2eid` route |

The 0-of-23 acceptance is the honest number for a random-weight block, not a
speed claim: what this arm proves is that the body runs end to end and that a
wrong drafter costs speed, never correctness. The row stays `No` (`ArchSupport`)
until a real nextn block is run through `scripts/mtp_spec_gate.sh`; the `why`
text now says the body is wired, so the next reader is not sent looking for the
attention.
