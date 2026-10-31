# Pending work — what is open, and what closes it

Pass: **2026-10-31**. Tree: `a655346` (`main`). Maintained by the session that
changes the thing, not by a later reader.

This file is a **derived** list. The detail lives in
[`docs/PLAN-remaining.md`](docs/PLAN-remaining.md) (code and measurements),
[`docs/MODEL-STATUS.md`](docs/MODEL-STATUS.md) (per-family state) and
[`docs/TODO.md`](docs/TODO.md) (the long historical tail). If they disagree with
this file, **they win and this file is wrong** — fix it here.

Three rules, because breaking them is what has cost the most time in this repo:

1. **Every item names the artifact or the diff that closes it, and the check that
   proves it.** An item with neither is a wish, not a task.
2. **Read the date on a claim.** A summary line can outlive the code it
   describes; the B2.3 line did, and a session spent an hour re-deriving a
   finished feature because of it.
3. **A row moves only with evidence attached** — a recorded run, a test name and
   its count, or a diff. Never "should work now".

Status vocabulary: `BLOCKED-FILE` (needs a checkpoint), `BLOCKED-BOX` (needs
different hardware), `CODE-READY` (start today, no artifact), `MEASURE-FIRST`
(the honest state is unknown until a number exists), `DEVIATION` (accepted and
stated, not a bug).

---

## The list

| # | Item | Status | What closes it | Size |
|---|---|---|---|---|
| P0 | **A7 — the 4 GB-on-16 GB-RAM promise has never been run end to end** | BLOCKED-FILE | one 30B-A3B q4 file + a 16 GB box | S |
| P1 | **A1 — `qwen4-mtp` `Partial` → `Yes`** | BLOCKED-FILE | `qwen4exp` trunk + its MTP head set | S |
| P1 | **A2 — `deepseek4-dspark` `No` → `Yes`** | BLOCKED-FILE | `deepseek4` trunk + dspark block | S |
| P1 | **A3 — `dspark` `Partial`** | BLOCKED-FILE | a matched DFlash2 head, or upstream logits | M |
| P1 | **Iteration-level admission** | CODE-READY | scheduler diff + a measurement | L |
| P2 | **`qwen4exp` whole-graph coherence gate** | BLOCKED-FILE | the real REAP-288 GGUF + reference logits | M |
| P2 | **HIP attention-logit softcap** (`gemma2` on device) | CODE-READY | one kernel + a parity test | S |
| P2 | **MoVA value-expert attention** (`k2-horizon`) | CODE-READY | router over `attn_v_exps` + fixture | M |
| P2 | **KV tier × batching** | CODE-READY | per-sequence paging in the tier | L |
| P2 | **Prefix cache: radix tree, persistence/journal/TTL** | CODE-READY | design + tests | L |
| P2 | **`deepseek41` status contradicts itself** | MEASURE-FIRST | one file carrying `hc_*`/compressed tensors | S |
| P2 | **A4 — `gemma2` on a real file** | BLOCKED-FILE | one `gemma2` GGUF, CPU run | S |
| P2 | **A5 — OLMoE against an independent oracle** | BLOCKED-FILE | the file + a bounded llama.cpp run | S |
| P2 | **Recurrent speculation / Tail-Replay** | CODE-READY | checkpoint/rollback contract for recurrent + conv state | L |
| P3 | **Per-format throughput table** | MEASURE-FIRST | one rotated bench across the formats in the dispatch | M |
| P3 | **A4b — `qwen2vl`/`qwen3vl` vision towers** | BLOCKED-FILE | towers + projector; out of scope unless asked | L |
| P3 | **`LLaDA` diffusion generation loop** | MEASURE-FIRST | verify the 2026-10-11 claim against `diffuse_max_tokens_` | S |
| P3 | **gfx1031 / gfx1201 verification set** | BLOCKED-BOX | the two cards | M |
| P3 | **C4/G14 — host floating point changes when HIP is linked** | BLOCKED-BOX | a HIP box + a bisect | M |

`Size` is roughly: S = half a session, M = one to two, L = a project.

---

## A. Blocked on a model file

### P0 — A7: the promise has never been tested end to end

Every residency number in this repo comes from **synthetic** fixtures; this
workspace has 3 GB of RAM and no model. The product claim — a 4 GB-class MoE
that keeps hot experts in VRAM/RAM and the rest on disk, on 16/24/32 GB — has
never had a single run behind it.

- **Needed:** a 30B-A3B-class GGUF in `q4_K_M` (≈18 GB) and a box with 16 GB of
  RAM, plus 24 GB and 32 GB if the comparison is wanted.
- **Command:**
  `kraken -m <30B-A3B q4> --ctx 4096 -n 64 --stats` with `KRK_MOE_PLAN=1` and
  `KRK_EXPERT_PIN_HITS` set, plus `kraken-cpu-expert-ceiling` on the same file
  for the per-expert cost the plan has to beat.
- **Closes when:** decode tok/s exists **per RAM budget**, with the hot-set hit
  rate and cold read bytes/token beside it. Target on 4-core AVX-512:
  200+ tok/s prefill, 7-8 tok/s decode.

### P1 — A1: `qwen4-mtp` `Partial` → `Yes`

- **Needed:** a `qwen4exp` trunk (Qwen3.8-Flash or a REAP-288 repack) **and** its
  head set (the 889 MB `mtp-q2_0.gguf`, or any file with the 31 tensors in
  [`docs/MTP-HEAD-SCHEMA.md`](docs/MTP-HEAD-SCHEMA.md)).
- **Command:** `scripts/mtp_spec_gate.sh <trunk.gguf> <head.gguf> 128 "<2 KB prompt>"`.
- **Closes when:** step-1 ≥ 60 % **and** spec decode strictly faster. If step-1
  lands under 60 %, the three ordered causes (paired-token convention,
  `KRK_MTP_HNORM`, dense vs indexed attention) are each a switch — try them
  before touching the graph.

### P1 — A2: `deepseek4-dspark` `No` → `Yes`

- **Needed:** a `deepseek4` trunk plus its dspark nextn block, in-file
  (`blk.<n_layer>.*`) or as a `--mtp` sidecar.
- **State:** the body is wired and measured (`test_mtp_sidecar_mla`, 22/22); the
  only new failure mode is geometry, and every geometry check prints both
  numbers by name.
- **Closes when:** the gate exits 0. If it refuses at load, the message names the
  tensor — **record that**, it is the answer, not a failure.

### P1 — A3: `dspark` (DFlash2 head on qwen35) `Partial`

- **Known:** plumbing excluded by measurement; the 5-layer decoder never sharpens
  the mask row at any depth (gaps 0.02-1.1 from embed through post-`l4`, flat
  final) while capture scale, KV norms, the KV self-check, the attention route
  and the mask id are all healthy. Probe data: `docs/DFLASH-DRAFTER.md` §11.
- **Ruled out this cycle:** a *format* mismatch on the head's expert bank
  (`gemm_group` takes one weight type; a mixed trio used to load and answer
  wrongly — now refused by name, `test_mixed_expert_formats` 7/7).
- **Needed (either):** a DFlash2 head trained on the pairing it is run against,
  or upstream reference logits for one block on this pairing. External evidence
  for the same flat-acceptance signature is in `docs/PLAN-remaining.md` §A3.
- **Closes when:** a sharp row appears (pairing was wrong), or the graph is shown
  to differ from reference logits (graph was wrong).

### P2 — `qwen4exp`: gated by a coherence check, not by a kernel

Every component has a kernel; what is missing is whole-graph agreement with an
independently verified reference **on the real GGUF**. One recorded REAP-288
prompt matched 24/24 and a coding prompt historically diverged, and serving
fluent wrong output is the failure this engine exists to avoid.

- **Needed:** that GGUF plus a reference run on the diverging prompt.
- **Closes when:** one prompt that used to diverge is diffed stage by stage with
  `KRK_DUMP` and the two agree, or the first differing stage is named.

### P2 — A4 / A4b

- `gemma2` (`Partial`) has **two** separate reasons, and they need different
  fixes: (a) no real file has proved the graph — needs one `gemma2` GGUF, a
  greedy run and a `--chunk` sweep diffed with `KRK_DUMP`; (b) on a HIP build it
  is refused by *capability*, because the attention softcap exists only in the
  CPU reference — that is the HIP softcap item below, and no checkpoint helps.
- `qwen2vl` / `qwen3vl` (`Partial`): their text halves run; the vision towers are
  not read. **Out of scope unless asked for** — say so rather than half-building
  it.

### P2 — A5: OLMoE disagrees with an independent oracle

- **Needed:** the exact model **and** a bounded llama.cpp run on the same prompt.
  CPU/device agreement is *not* evidence — both can be wrong the same way.
- **Closes when:** the texts agree, or a `KRK_DUMP` diff against the oracle names
  the first diverging stage.

### P2 — `deepseek41` contradicts itself (MEASURE-FIRST)

`src/arch.cpp:984` says `ArchSupport::Yes` for `deepseek41`; `MODEL-STATUS.md`
says `Partial` with compressed attention, the sparse indexer, `hc_*` and Engram
missing. Both cannot be right, and V4 compressed attention *is* implemented
(`mla_shared_attention_compressed`).

- **Needed:** one `deepseek41` file that actually carries `hc_*` and a non-zero
  compress ratio.
- **Closes when:** it runs end to end (row stays `Yes`, delete the `Partial`
  text) or it refuses by tensor name (arch row moves to `Partial`, named
  blocker).

---

## B. Code work with no artifact required

### P1 — Iteration-level admission

The server admits a **wave** and runs it to completion; a long request in one
slot holds new arrivals until the wave drains. That is the difference between
"batching works" and "a server", and it is visible under load.

- **Needs:** a scheduler that admits on iteration boundaries, per-slot prefill
  chunking, and a cancellation path that does not leave a slot half-filled.
- **Closes when:** a test shows a short request completing **while** a long one
  is still decoding, with output equality against the solo runs (that equality
  already exists: `test_batch_decode`, `mla_batch_arm`), and a measurement of the
  gain against the current 1.08-1.27× (transfer-bound) / 1.02× (resident).
- **Risk:** the wave scheduler is what batching's correctness leans on. Keep the
  existing exactness arms running against both paths during the change.

### P2 — HIP attention-logit softcap

`gemma2`'s attention softcap is applied by the CPU reference only
(`AttnDesc::softcap`); the HIP backend answers
`attention_softcap_supported() == false` and the engine refuses the model there
(`src/engine.cpp:2373`, `src/arch.cpp:718-731`).

- **Needs:** the softcap inside the device attention kernel, the capability
  answer flipped, and a CPU-vs-device parity arm on the `gemma2` fixture.
- **Closes when:** the `gemma2` row runs on a HIP build and the parity arm passes.

### P2 — MoVA value-expert attention (`k2-horizon`)

Refused at load today: routed experts on the attention **values**
(`attn_v_exps`) with a gate (`attn_v_gate`) — Mixture-of-Values attention.

- **Needs:** the router over the value bank, the gather into the attention
  values, and the per-expert bias. The fixture builder already writes both
  tensors (`build_k2_horizon_model`, `vbank=true`), so the work has a test bed.
- **Closes when:** the refusal is gone, the fixture runs, and the value-bank arm
  differs from the plain-V arm in the direction the math says it must.

### P2 — KV tier × batching

`Engine::batch_refusal()` refuses a wave when the KV tier is active ("the KV tier
pages whole layers for one sequence's positions"), which is exactly the state a
100 GB model on a small box is always in.

- **Needs:** per-sequence paging in the tier — a page is `(layer, sequence)`, not
  `layer` — plus a residency test across two slots with disjoint positions.
- **Closes when:** the refusal line is gone for the tiered case and the batched
  tokens still equal the solo tokens with the tier **forced on**.

### P2 — Prefix cache: radix tree, persistence, TTL

Today one engine holds one reusable prefix record; there is no radix tree, no
journal and no TTL (`src/server.cpp`, `src/engine.cpp` — grep finds none).

- **Needs:** a radix prefix index over the record form that already exists, then
  an on-disk journal with TTL and a size bound.
- **Closes when:** two requests sharing a prefix shorter than the whole record
  both reuse it, and a restart reuses a journaled record — each with a test.

### P2 — Recurrent speculation / Tail-Replay

Speculation on a recurrent (GDN/SSM/conv) model needs a checkpoint/rollback
contract for state that is not a KV cache. The generic verify loop exists; the
rollback contract does not.

- **Needs:** a state snapshot/restore per draft step (or a proven single-step
  equivalence), then the same exactness arms as the transformer path.
- **Closes when:** a recurrent fixture produces identical tokens with and without
  speculation, and the rollback is exercised by a rejected-draft arm.

### P3 — Per-format throughput table (MEASURE-FIRST)

Which formats take an int8 dot kernel is spread across three dispatch sites
(f32 vector path `src/quant.cpp:4786`, the q8 GEMM, and the decode GEMV), and
`MODEL-STATUS.md`'s sweep is mostly `Q4_K_M` while its headline fixture is
`Q8_0`. There is **no table** saying what a `Q5_K`, `IQ4_XS` or `Q3_K` model
costs against the same model in `Q8_0`.

- **Needs:** one rotated, warmed benchmark across the formats that appear in a
  real collection, same shapes, medians — the discipline `tools/bench_gemv_scale`
  already uses (fixed arm order is a bug; it measured a cold stream as a slow
  kernel).
- **Closes when:** the table exists and `MODEL-SUPPORT-MATRIX.md`'s
  "only Q4_K/Q6_K/Q8_0 have an int8 kernel" claim is confirmed or corrected.

### P3 — `LLaDA` diffusion (MEASURE-FIRST)

The 2026-10-11 index says the schema and generation loop are missing, but
`diffuse` is `ArchSupport::Yes` and the engine carries a diffusion pass
(`diffuse_max_tokens_`, one whole-sequence pass per step). Verify before doing
anything: run the fixture, then either delete the claim or name what is actually
absent.

---

## C. Environment / hardware

- **gfx1031 (RDNA2):** `qwen35` reproducibility and the Q6_K dispatch timings
  were never confirmed on the card. **Needs:** the card. **Closes when:** both
  numbers exist.
- **gfx1201:** the all-zero probe readback stays unverified.
- **C4/G14:** host floating-point behaviour changes when HIP is linked. **Needs:**
  a HIP box and a bisect across compiler/runtime/link initialisation. **Closes
  when:** the divergence is reproduced and attributed, or shown to be gone.
- **Model corpus:** refresh the inventory and run a bounded smoke/coherence pass
  over the runnable files, including the two ambiguous device/CPU-return-code
  records.

---

## D. Accepted deviations (stated, not bugs)

| Row | Deviation | Why it stays |
|---|---|---|
| `minimax-m3` | block sets are **unioned** across group heads | union is a superset of every head's own set; the reference reads one set per group, and the union was measured to keep the selection stable |
| `minimax-m3` | a prefill chunk **wider than 8 rows attends densely** | the sparse selector's window is sized for decode; widening it without a measurement would trade a known-correct dense pass for an unmeasured sparse one |

Both are in `MODEL-STATUS.md` beside the row. Close them only by removing the
deviation in code **and** replacing the row's text in the same commit.

---

## E. Closed — do not reopen without new evidence

| Item | Evidence |
|---|---|
| MLA / V4-compressed per-row batching (B2.3 item 4) | `mla_pos`/`mla_row_base` (`src/backend_cpu.cpp:341-350`), per-slot spans (`src/engine.cpp:1805-1840`), CPU says yes (`:1937`), and `mla_batch_arm` matches three layouts to their solo runs |
| Continuous batching, cancellation, prefix reuse (B2.1-B2.3) | `test_batch_decode` (86 checks), `test_cancel_and_prefix`, `test_batch_throughput` |
| TSAN listen-descriptor race (B4) | ownership redesign; `tools/tsan_http_listen.cpp` reproduces the race on the pre-fix file and is clean now; `test_http_listener_lifetime` 15/15 |
| CPU GEMM repack (B1) | closed **by measurement**: `wsc8` is the best arm; pv8 is 7-9 % slower |
| Trace identity (B3) | `test_dump_identity` 17/17 |
| q8_0 decode GEMV prefetch/pluck (PR #94) | distance 8; 1.17-1.24× in the kernel harness, engine fixture unchanged (L2-resident) |
| Mixed-format routed-expert trio (PR #95) | refused by name in all three loaders; `test_mixed_expert_formats` 7/7 |
| Draft-only / malformed MTP exports | three synthetic files, three named refusals, exit 1, no crash (`docs/PLAN-remaining.md` §A1) |

---

## How an item leaves this file

Either it moves into the table's closed section with its evidence, or the
acceptance criterion above is met and the row moves in
[`docs/PLAN-remaining.md`](docs/PLAN-remaining.md) / `MODEL-STATUS.md` **in the
same commit**. There is no third route: an item does not get deleted because it
stopped being mentioned.
