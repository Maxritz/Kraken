# Remaining work, split by what it waits on

Status 2026-10-31, `main` = `404d870` (PR #87 merged). Gate at that commit:
**7344/7344 checks, 172 groups, tool 18/18, proof 17/17, exit 0**.

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

### B1 -- CPU GEMM: the production kernel is at 40 % of its own instruction mix

This is the item that decides the throughput target (200+ tok/s prefill, 7-8
tok/s decode, 30B-A3B, 4-core AVX-512). The measurement is already in the tree
and it is unambiguous:

| what | GMAC/s | of V1 |
|---|---:|---:|
| production q8_0 GEMM, 768x2048x512, one thread (GEMM bench) | 32.2 | **40 %** |
| the same kernel inside the ablation, its own L0 | 24.54 | 30 % |
| V1 -- the exact instruction mix it ships, clean harness | 78.2-80.6 | 100 % |

The same kernel reads 24.54 and 32.2 in two harnesses -- 1.3x apart, which is why
the first step re-baselines in one process. The 2.5x to V1 is a **kernel
implementation** gap, not an instruction-mix gap: the same mix runs 2.5x faster
when the harness is clean. The ablation prices the two big rungs at **+34.5 %**
(constant weight scale, 33.01) and **+24.4 %** (integer accumulate, 30.54) --
and both are marked WRONG, because each is a *format* change, which A6 closed.
So they price work that is not shippable; the shippable target is the 2.5x
between the production kernel and its own mix.

**Steps, in this order:**

1. **Re-run the ablation on this node** (`kraken-q80-ablate`) and record L0..L7
   in one process, so the baseline is current. *Done when:* the table is in
   GEMM-PERF-WALL.md with the date and the host.
2. **Diff the production kernel against the V1 harness** for one shape: which
   ops exist in production that the harness does not have, and which is dominant.
   *Done when:* each extra op is attributed a GMAC/s cost by removing it in an
   ablation rung -- the same method that produced the L-table.
3. **Fix the single largest term.** *Done when:* the shape moves >= 10 % with
   `--prefill-hash` identical and `KRK_CPU_DOT_INT8=0` still exact.
4. **Re-project, do not assume:** `tok/s = GMAC/s(shape) / (active MAC/token)`.
   At 32.2 GMAC/s a 30B-A3B-class token (~3 GMAC) is ~10 tok/s single-thread;
   the target needs ~4x on the kernel plus the core count. *Done when:* the
   projection, its inputs and the measured numbers are one table.

**Cost:** 2-3 sessions; the first step is an afternoon and it either finds a
dominant term or records that the gap is spread out (also a result).

### B2 -- The server runs one generation at a time

`src/server.cpp` (508 lines): connections are concurrent, **generate() is
serialised by one mutex**; there is a single-session linear prefix reuse
(`reusable_prefix`, "this is not an approximation of the prefix, it IS the
prefix") and no radix tree, no cancellation, no continuous batching.

**Order, smallest first:**

1. **Cancellation** -- a flag the token loop checks, so a dropped connection
   stops the work instead of finishing it. *Done when:* a test cancels mid-decode
   and the server reports the partial token count.
2. **Prefix reuse across requests** -- the linear prefix becomes a tree keyed by
   token ids, with the KV cut back to the longest common prefix. *Done when:*
   two requests sharing the first N tokens prefill only the remainder, measured
   by `[stats ] prefill` on both.
3. **Continuous batching** -- several sequences in one forward. This is the one
   that needs engine work (per-sequence KV slots, one scheduler). *Done when:*
   4 concurrent requests hold aggregate tok/s above one-at-a-time, with each
   response identical to its solo run.

### B3 -- A trace file carries no identity

`Engine::dump_row` opens the dump path with `"a"` per call: no header, no model
or request id, no end marker, and no truncation report. Two runs -- or a target
and a drafter -- append into one file and cannot be told apart afterwards. Every
"A/B on the same prompt" comparison in this project depends on reading those
files.

**How:** one header line per run (model path + size + mtime, chunk, ctx, the
dump mode), one line per stage as now, an explicit end marker with the line
count, and a refusal to append to a file whose last run did not end.

*Done when:* two runs into one file are separable by header, and a truncated run
is detected (a test writes half a file and reads it back).

### B4 -- TSAN: the listen descriptor's lifetime

Reported: a race on the listen descriptor, and a stale sanitizer baseline. An
atomic descriptor alone does not settle close/reuse ownership.

**How:** design the shutdown lifetime first (who owns the fd, who may close it),
then verify under `-fsanitize=thread` with a repeated start/stop test.

*Done when:* the sanitizer is clean for 100 start/stop cycles and the baseline
file is re-recorded.

---

## C. Closed -- do not reopen

| item | why it is closed |
|---|---|
| prefill long-group format (`Q8_0_S4`, g4) | 1.43x speed, 1.25x/1.16x error; 1.00x for the compromise design. A6 |
| 16-row `x16_vnni`, 64-B vector scale table, 16-row 2-float table, f16-expanded scales | all measured worse (49.9, 10.1, 24.0 GMAC/s) |
| recurrent-state rollback for speculation | **implemented** -- snapshot `mtp_rec_snap_`/`mtp_conv_snap_` at round start, restore on reject, spec off if the snapshot cannot allocate |
| `qwen4exp` as an unsupported family | it is `Yes` now; `test_qwen4exp_whole_graph` pins the sentence |

---

## The list, in the order it should run

1. **A1** -- run the spec gate on a real `qwen4exp` trunk + head; move `qwen4-mtp`. *(needs files)*
2. **A2** -- same for `deepseek4` + dspark; move `deepseek4-dspark`. *(needs files)*
3. **A7** -- the 30B-A3B run on 16/24/32 GB of RAM; the only end-to-end test of the product promise. *(needs a file and a box)*
4. **B1.1** -- re-run `kraken-q80-ablate`, record L0..L7 on this node. *(now)*
5. **B1.2** -- attribute the 2.5x between the production kernel and V1 op by op. *(now)*
6. **B1.3** -- fix the dominant term; gate on `--prefill-hash` identity. *(now)*
7. **B1.4** -- the projection table: GMAC/s -> tok/s at 2 and 4 cores, against 200/7-8. *(now)*
8. **B3** -- trace identity and truncation reporting; it is what every A/B above is read through. *(now, small)*
9. **B2.1** -- server cancellation. *(now)*
10. **B2.2** -- prefix reuse across requests. *(now)*
11. **B2.3** -- continuous batching. *(now, largest)*
12. **B4** -- TSAN lifetime. *(now)*
13. **A3** -- DFlash2 head on its own pairing, then upstream logits. *(needs files)*
14. **A5** -- OLMoE against an independent oracle. *(needs a file and an oracle run)*
15. **A4** -- `gemma2` (and the vision rows, if wanted) on real files. *(needs files)*

Items 1-3 and 13-15 cannot be started in this workspace; 4-12 can, and 8 is the
cheapest one that makes everything else measurable.
