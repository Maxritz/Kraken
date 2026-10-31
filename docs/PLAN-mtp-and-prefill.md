# Two open items, split into steps that can each be gated

Status 2026-10-31, after PR #84. Everything below assumes `main` at `5dae3b8`.

**Progress:** steps **1.1-1.4 are done** (see *Item 1 status* at the end of that
section). 1.5-1.7 are the forward and are untouched; item 4 has not started.

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

**1.7 — Proposer + the speed gate** (½ session)
`--mtp-spec` with per-step acceptance and wall time.
*Done when:* acceptance ≥ 0.6 at step 1 **and** decode wall time below plain
decode on the same prompt. A drafter that is accepted but costs more than it
saves is a regression with a good story.

**1.8 — Rows, docs, suite** (½ session)
`qwen4-mtp` becomes `Yes` (Draft role), `deepseek4-dspark` follows only if its
own attention is wired (same interface, different block — reuse, do not share
code); new group `test_mtp_head_qwen4`; baseline re-recorded.

**Total:** 5-7 sessions, of which 1.6 is 60 % of it.

### Item 1 status

| step | state | evidence |
|---|---|---|
| 1.1 | **done** | `docs/MTP-HEAD-SCHEMA.md` — 31 rows, each with shape, format and consumer; the three not-in-the-31 tensors are listed with why |
| 1.2 | **done** | `MtpBlock` gained `hf`, `n_lane`, `n_hc_rank`, `n_index_head`, `index_dim`, `HcMixer{hc_attn,hc_ffn,hc_head}`, `fc_embedding`/`fc_hidden`/`eh_proj_split`, `index_{q,k}`+norms, `exps_gate_up` |
| 1.3 | **done** | `Model::load_mtp_hf_block` (src/mtp_head.cpp) + `test_mtp_head_qwen4`: a synthetic 31-tensor file loads, and the wrong-width bank / half-written prefix / mismatched `down_proj` are refused by tensor name |
| 1.4 | **done** | `Gguf::add_expert_plane_slice` + `GgufTensor::plane_stride`/`plane()`; the expert path reads through `plane()` in all five places it used to compute `data + expert * per`. `test_mtp_head_qwen4` compares **every** expert's gate and up view against the source rows with `memcmp`, not a sample |
| 1.5 | open | |
| 1.6 | open | |
| 1.7 | open | |
| 1.8 | partly | the arch row is `Partial` (loads and validates, cannot propose); `test_mtp_head_qwen4` exists; the baseline is not re-recorded until the forward lands |

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

What the port still cannot do: the block has no forward. `--mtp` on a real head
now loads, validates and reports it (`MTP head ...: 4 lane(s) x 2560, 512
experts ... fwd=not implemented`), and then nothing drafts. That is 1.5-1.7.

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

**4.1 — Settle the layout on paper and in a bench, before engine code** (½ session)
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

**4.4 — Measure, then decide** (½ session)
A/B against W on 768×2048×512, best-of-5, median and spread, two runs.
*Done when:* **≥ 1.15×** over W **and** the synthetic model's logits move no more
than q8_0's already do. Missing the gate means the format does not ship and the
number goes into docs/GEMM-PERF-WALL.md.

**4.5 — Wire it** (½ session)
Loader accepts the type, the engine picks the layout per tensor, a suite group
covers load + generate + the error bound, docs updated.

**Total:** 2-3 sessions, with a real chance step 4.1 or 4.4 fails the gate.

---

## Order

1. **1.1 → 1.4** first: they are the steps that can be wrong in a way the tests
   can see cheaply, and they unblock 1.6.
2. **4.1** in parallel is legitimate — it is a bench, no engine code.
3. **1.6** next, because it is the only step that turns into tokens/s.
4. **4.2 → 4.5** last.

Neither item shares a line of code with the other; the order above is about risk,
not dependency.
