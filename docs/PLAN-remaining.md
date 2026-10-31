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

**B2.3 continuous batching -- OPEN. The three ops that are NOT row-independent
now honour the per-row tables; the engine and the scheduler are what is left.**

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

What is left, in order:

1. **N KV regions in the engine.** One `kv_pos_` and one region today; batching
   needs `region = layer_stride * n_layer`, a per-row base, and a per-slot
   cursor. This is the invasive one: `kv_pos_` is read by the KV tier, the dump
   header, health, prefix reuse and the speculative paths.
2. **A batched decode step** -- one `forward` with K rows, per-row
   (token, position, slot base), K logit rows, per-sequence sampler state.
3. **A scheduler in the server** that admits, prefills one at a time (a prefill
   of sequence s is just rows 0..n-1 at slot s) and decode-batches the active
   set, streaming each sequence's deltas to its own connection.
4. **MLA/compressed attention for the batched case.** The dense/GQA path is
   row-independent once the cache base is per row; `mla_attention`,
   `mla_shared_attention*` and the compressed/indexer variants each own their
   cache addressing, so they need the same treatment before a deepseek-class
   model can be batched.

*Done when:* 4 concurrent requests hold aggregate tok/s above one-at-a-time with
each response identical to its solo run -- unchanged, and not yet met. The honest
statement today: the per-op contract is ready and pinned, the engine is not.

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
| 9 | **B2.3** -- continuous batching | **open**: the four pieces are listed under *B2*; 1-2 are small, 3-4 are the work |
| 10 | **B3** -- trace identity and truncation reporting | **done**, `test_dump_identity` 17/17 |
| 11 | **B4** -- TSAN listen-descriptor lifetime | **done**, `kraken-tsan-http-listen` clean, old code reproduces the race |
| 12 | **A3** -- DFlash2 head on its own pairing, then upstream logits | *needs files* |
| 13 | **A5** -- OLMoE against an independent oracle | *needs a file and an oracle run* |
| 14 | **A4** -- `gemma2` (and the vision rows, if wanted) on real files | *needs files* |

**What a session without files can still move:** B2.3, and then only item 3-4 of
it. Everything else in the table either needs a checkpoint or is closed by a
measurement that now exists.
