# AGENTS.md

Operational knowledge for working in this repo. Everything here is something a
fresh session would otherwise have to rediscover: it is not visible from the
code, the README, or `docs/`.

## Build and gates

- Build: `ninja -C build-hip kraken kraken-tests kraken-bench kraken-server`
  `kraken-inspect` and `kraken-oracle`.
  **Always capture the exit status explicitly** (`> /tmp/b.log 2>&1; echo rc=$?`).
  Piping into `grep`/`head` masks a ninja failure, and `grep -c` exits 1 on zero
  matches — that is not a build failure.
- **`kraken-inspect` is in the build line deliberately.** It answers loader
  verdicts from the same `krk_core` tables the loader uses, and one built before
  a new dequantizer landed reports `refused` for formats the engine already
  runs -- seen live: it refused #41/#21/#22/#18 the day after they were fixed,
  contradicting the README. `build_check.sh` requires it current for the same
  reason.
- The gate suite is six things, not one: `sh scripts/build_check.sh`, ninja
  rc=0 with 0 `error:` lines, `kraken-tests` 3413/3413, `sh
  scripts/kraken_tests_baseline_check.sh` (names the groups that newly broke
  against the recorded baseline), `kraken-bench --gate` rc=0, and a coherence
  spot check on three models, one per shape, run as:
  `bash scripts/coherence_check.sh /g/More-models/Qwen3-4B-Q8_0.gguf
  /h/OLLAMA-Models/GGUF/Qwen3.5-9B-Q4_K_M.gguf` and
  `KRK_ALLOW_BIG=1 bash scripts/coherence_check.sh
  /g/More-models/Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K.gguf`
  (dense / recurrent / MoE; expect rc=0 and `1 coherent, 0 not` per
  invocation). **The three names this bullet used to carry --
  `models/SmolLM2-135M-Instruct.Q4_K_M`, `Qwen3.5-0.8B`, `Qwen3-MOE-4x0.6B` --
  exist in neither `G:\More-models` nor `H:\OLLAMA-Models\GGUF`**, and this
  worktree has no `models/` at all, so a session that quotes that line cannot
  run it: grep the two trees for what is actually there and keep one per shape.
  Two traps from doing exactly that: `Qwen3-30B-A3B` is over the script's
  `KRK_MAX_MB=8192` guard and reports `1 not tested` (a FAIL that ran nothing)
  unless `KRK_ALLOW_BIG=1` is set, and `Qwen3.5-9B`'s `--cpu` arm needs
  `KRK_TIMEOUT` above the 120 s default. `limite-1b-violetto` is not a
  candidate: it is refused for quant type #26 and `attn_gate`.
  `olmoe-1b-7b-0924-q4_k_m-imat` is a real device-vs-CPU divergence, not a
  candidate either: `FAIL (drift step 5)` reproduces on every run, and the two
  fold switches (`KRK_FUSED_SILU=0`, `KRK_ACC_GEMM=0`) leave the device text
  identical -- so it is OLMoE's own path, unresolved, and the 30B MoE above is
  what makes the MoE arm green.
  Each model runs twice -- device arm and the `--cpu` scalar reference -- and a
  device arm that silently fell back to the CPU backend fails as `FAIL (cpu)`,
  so a green run is proof the GPU produced the text.
- **The chunk-invariance gate is a test group, and it drives the CLI because the
  suite cannot open the device.** `./build-hip/kraken-tests --run-group
  test_prefill_chunk_bitexact` builds a head_dim-64 / 8-head / 4-layer fixture,
  prefills one 64-token prompt at `--chunk 1`, `2` and `8` through
  `kraken.exe --prefill-hash`, and requires one hash across them. It runs in
  the default suite (group 35 of 56, 14 checks) and the baseline carries it.
  `--prefill-hash` prints two hashes and no text:
  `prefill-hash <trace> logits <row> prompt_tokens=N first=I`. The trace folds
  the embedding, every layer's attention output and every layer's hidden state
  over **all** prompt rows, per (stage, layer); `logits` is the last position's
  whole row. Re-run the two arms by hand to see it:
  `kraken.exe -m M -p "..." --chunk 1 --greedy --prefill-hash` against the same
  line at `--chunk 8`.
  **The logits hash alone is blind, and that was measured, not assumed.** On the
  fixture at 64 tokens with `KRK_ATTN_BEXACT=0` (the tiled kernel) the trace
  differs while the `logits` row is bit-identical, because a one-ulp difference
  in a non-final row is absorbed by the f16 residual before the head. The gate
  checks `logits` too, but the trace is the one that fails. Re-measured on real
  text (`Qwen3-4B-Q8_0`, 27-token prompt, `--prefill-hash`): the default kernel
  gives `39dcbbdfe43c6e72` / `9a3418eb058b319b` at `--chunk 1`, `2` and `8`,
  the tiled arm gives `a822a6f82b159f0c` / `ebcf60c3e66a836a`, and **the first
  token is 576 in all four runs** -- so a gate that compared generated text, or
  only the sampled token, would have called the regression identical. At other
  lengths the tiled arm moves both hashes; the trace is not a weaker signal, it
  is the one that never goes blind.
  **It carries its own negative control:** the chunk-8 arm runs a second time
  with `KRK_ATTN_BEXACT=0` and that trace hash must DIFFER from the chunk-1
  default (measured `bf32bbb3...` for all three default arms against
  `33048d8c...` for the tiled one). A gate whose bad arm also passes proves
  nothing.
  Two of the three defects found building it were in the gate: hashing
  `tokenizer().n_vocab()` floats past a `model().cfg().n_vocab` row read
  uninitialized heap (which varied with the chunk size and read as a kernel
  regression until the fixture's 261-token vocab was matched to its embedding
  tensor), and a text-level comparison cannot see this drift at all. The third
  was a rule: a group that writes files must remove them when green, because
  INPROC mode has no per-group working directory -- the isolation check's
  `leaves-nothing` arm failed on the group's capture files until it did.
- **`kraken-tests --report <path>` writes the machine-readable summary, and
  `tests/kraken-tests-baseline.json` is the last green run.** One group per line,
  so a re-recorded baseline diffs readably: name, status (`pass`, `failed`,
  `no-checks`, `crashed`, `no-scratch`, `could-not-start`), passed/run, and up to
  40 `FAIL` lines with their `file:line`. `KRK_TEST_INPROC=1` writes the same
  report from counts alone -- no per-group log to read there, so no FAIL text --
  and a report that was asked for and could not be written exits 1 even when
  every check passed, because the artifact the caller needs does not exist.
  `scripts/kraken_tests_baseline_check.sh` runs the suite for a report, runs
  `tools/test_report_diff.py --selftest` first so a green verdict means the
  comparison was shown able to fail, and then compares: the only fatal finding is
  a group the baseline has as passing that this run does not -- *newly broken*,
  printed with the failing checks' text and nothing else. A count change while
  still passing is a note (the watermark moves whenever a check is added), and
  new or removed groups are a change to the suite's *shape* that fails until the
  baseline is re-recorded deliberately (`KRK_TEST_BASELINE_RECORD=1`, which
  refuses a run that is not green unless `KRK_BASELINE_FORCE=1`;
  `KRK_BASELINE_ALLOW_NEW=1` compares anyway). A run whose own exit is non-zero
  while every group reads as passing (the report could not be written, say) is
  reported as unusable rather than compared. `recorded_commit` is HEAD at record
  time, so it names the parent commit of the commit that adds the baseline. The
  tool's selftest is 18 constructed cases -- including a fixed group, a crashed
  group and a still-broken group -- and mutating the classifier makes it fail
  (16/18 with the arm comparison disabled, which is the case that proves the arm
  findings are not cosmetic).
  The same tool compares a `--selfcheck --report` sweep (kind `selfcheck`) arm by
  arm: each group line carries an `arms` map, an arm the baseline has as ok and
  this run does not is *newly broken arms* whatever the group's own status says,
  and an arm added or removed is a shape change like a new group.
  `tests/kraken-tests-selfcheck-baseline.json` is that record, and the baseline
  gate only runs the six-minute sweep when `KRK_BASELINE_SELFCHECK=1` is set.
- That 3008 is a watermark, not a constant: the suite grew 2214 -> 2316 when the
  missing dequantizers landed, 2316 -> 2329 when Q2_0's geometry changed,
  2329 -> 2751 when the KV-tier per-layer geometry and the sparse-KV fixture
  tests landed, 2751 -> 2760 when the mixed-dtype `gemm_group` case below
  landed, 2760 -> 2994 when the attention micro-test and the KV-tier geometry
  cases landed, and 2994 -> 3008 when the chunk-invariance gate did. Read the
  count off the run (`3008/3008 checks passed`) rather than trusting the number
  written here, which is only the last one seen.
- **`kraken-tests` runs each of its 56 groups in its own child process, several
  at a time.** A fastfail, an access violation or an abort inside one group used
  to end the run before the `%d/%d checks passed` line, which is how a single
  `std::fclose(NULL)` in a KV-tier test hid eight real failures for a whole
  session. The parent prints a line per group, names a group that died
  (`CRASHED 0xC0000409`) along with the number of checks that had run, and still
  prints the summary; the exit status is 1 if any group crashed, failed, or ran
  no checks. `KRK_TEST_INPROC=1` is the old single-process run -- a debugger's
  view, and the A/B: both runners print the same per-group table, and
  `scripts/kraken_tests_isolation_check.sh` diffs every group line rather than just
  the total. That script drives `KRK_TEST_INJECT_CRASH`, `KRK_TEST_INJECT_CRASH_AT`,
  `KRK_TEST_INJECT_FAIL` and `KRK_TEST_INJECT_EMPTY` to prove the reporting
  itself, so run it after touching the harness; `KRK_TEST_INJECT_CRASH` takes a
  comma-separated list (with `KRK_TEST_INJECT_CRASH_AT` positionally aligned),
  which is how a run with two groups non-passing at once is built by hand, and
  an aim past the group's last check is never reached rather than an error.
  `KRK_ISOLATION_FULL=1` adds the sweep that runs every group alone in an empty
  private directory, and `KRK_ISOLATION_INJECTIONS=1` adds the per-group
  selfcheck.
- **`kraken-tests --selfcheck` is the reporting proof, and it needs no other
  tool.** It runs six arms for every group -- crash before the first check,
  crash after check 2, crash after the group's OWN last check, every check
  failing, no checks at all, and a fuzz arm that dies at a random check and in
  one run in four at a SECOND group as well, so a run with two groups
  non-passing at once is constructed rather than assumed -- and judges each
  answer against that group's own count(s): the group's line and its entry in
  the report (which must agree with each other), every group the arm did not aim
  at still passing with its own count, totals equal to the clean total minus
  each aimed group's count plus what each reached, bad_groups equal to the
  number of groups aimed at, exit 1, every aimed group named in `groups needing
  attention`, the summary printed, and FAIL lines equal to the failing checks.
  The last-check arm fires on the end-of-group `note_progress()`, so `passed ==
  run` there and only the child's exit says the group died: a runner deciding
  status from the counts alone cannot pass it, and no two groups have the same
  expected number. The fuzz aim is drawn from a seed printed at the top of the
  sweep, and `KRK_SELFCHECK_SEED=<n>` draws exactly those aims again -- per
  group, so `--group N` draws what the sweep drew for N; a seed that cannot be
  read is refused (rc=2) rather than silently replaced, because it was set to
  make a run replayable. `--group <index|name>` runs one group's six arms (~6 s)
  and `--selfcheck --selftest` is 17 constructed answers, one per way an arm can
  be wrong (~10 ms) -- that one runs in every isolation check, because a checker
  that cannot fail proves nothing. `KRK_SELFCHECK_BREAK=counts` makes the
  expectation wrong by one check, so a real mismatch can be constructed: it
  only tightens (a green run goes red, never the reverse), and the isolation
  check requires rc=1 from it, which is how the exit status is proved and not
  just the checker. The whole sweep is 318 suite runs (measured 240 s at eight
  workers, so a hair over the ~4 min it prints as `wall_ms`), so it is the
  opt-in arm. `--selfcheck --report <path>` records the arms' verdicts in
  the runner's report format (kind `selfcheck`: one line per group, its clean
  counts, and an `arms` map) so the gate can compare the reporting proof and not
  only run it -- `tests/kraken-tests-selfcheck-baseline.json` is that record,
  compared arm by arm under `KRK_BASELINE_SELFCHECK=1` (a `--group` report covers
  only the groups swept, so it is evidence, not a baseline). Running it with an
  injection already set is refused (rc=2): the arms choose their own, and a
  caller's would make the clean run something other than clean.
- **Chunk invariance was an ATTENTION property, and the box was rows==1 vs
  rows>=2 -- that is now fixed, and the next box is the GEMM tile rung (rows
  >= 9).** Before the rewrite, on Tiel (12-token prompt,
  greedy, 32 tokens) `--chunk 1` gave `613eb9b3...` and `--chunk 2`, `3`, `4`
  and `8` all gave `e1a040e0...`, so the boundary was the single decode row,
  not a wide prefill; the same run now gives `613eb9b3` at `--chunk 1`, `2`
  and `8`, and still differs at `256` because a 12-row prefill leaves the
  GEMM-exact window below.
  `tools/dump_passes.py`
  (new; `dump_diff.py` cannot answer this -- it pairs line i with line i, and a
  `--chunk 1` dump emits one pass per token while `--chunk 2` emits one per two,
  so it reports every line as different from line 0) splits both dumps at each
  `embed L-1` pass and compares shared positions: the first divergence is
  **`attn.out L03` at position 1, rel 3.8e-5**, after three recurrent layers
  that are bit-identical. Layer 3 is the first full-attention layer, so the
  divergence is the attention call and nothing before it. The drift is there
  from the FIRST multi-row forward -- the "~29 greedy steps" this repo kept
  quoting is how long it takes a 1-ulp difference to flip an argmax, not where
  the two paths part.
  **The micro-test pins it, and `attention_bexact_kernel` closes it.** The
  micro-test (`attn_rows_case` in `tools/bench_ops.cpp`, `kraken-bench
  --bitexact`) runs one K/V cache through N calls of one row (decode) and one
  call of N rows (prefill): before the rewrite hd256 / 16 heads / 2 KV (Tiel's
  geometry) at rows=2 differed in **7 of 8192 values at 1 ulp**, hd128/16 in 3
  of 4096, hd64/8 agreed at rows 2 and 4 and differed from 8.
  **The mechanism is the key-space reduction order, so the fix is one
  formulation.** `attention_decode_kernel` gives each of its 8 warps every 8th
  key, runs an online max that advances per key (`m = fmaxf(m, s)` inside the
  loop), and merges the eight `(m, l, acc)` triples at the end.
  `attention_qtile_kernel` takes the max over the whole key TILE and sums keys
  sequentially into one accumulator, rescaling per tile -- same terms, different
  addition order and a different `mnew` at each `exp`. `attention_bexact_kernel`
  (new, `src/hip/kernels/attention.hpp`) keeps what the tile is FOR -- K/V
  staged in LDS once and shared by QT query rows -- and reproduces decode's
  arithmetic exactly: warp w owns keys `j % 8 == w` in increasing order, one
  (row, key) score is decode's lane dot (`VPT/2` packed `v_dot2` pairs then one
  32-lane reduce), the softmax advances per key, and the eight triples merge in
  warp order. QT is 4 at hd 256 and 8 at hd 128/64 (`KRK_ATTN_BEXACT_QT=8`
  doubles it at hd 256); the dispatch is `n_tok > 1 && causal && window == 0 &&
  hd in {64,128,256}`, and **`KRK_ATTN_BEXACT=0` restores the tiled kernel** as
  the A/B arm. `kraken-bench --bitexact` attention is now **24 of 24 cases
  identical** -- rows 2/4/8/16 x hd 64/128/256 x n_keys 64 and 200 -- with
  n_keys=200 the case the key-range split would have broken.
  **It is FASTER, not a cost.** Interleaved order-balanced pairs, identical text
  in every pair: Qwen3.5-9B at a 170-token prompt **1166.2 against 1056.1 tok/s
  (+10.4%, 5 pairs)**, at 3400 tokens **866.0 against 406.0 (+113%)**; Phi-3.5
  (hd 96 -- the odd lane slice -- 2760 tokens) **585.0 against 324.8 (+80%)**,
  and against the query-tiled kernel that used to run there 585.0 against 393.5
  (+49%); Tiel 35B-A3B at 600 tokens 94.1 against 81.6 (3 pairs, expert-bound
  and noisy). It wins for the reason the tile kernel existed, minus the tile's
  weak parts: K/V is staged once per QT rows instead of once per row, and the
  per-key `exp` the tile recomputed inside its V loop is gone.
  **A trap worth keeping: the first version of this A/B reported "no
  difference" and was measuring two identical arms.** The harness took arms as
  `label=ENV=VALUE` and split each on the FIRST `=`, so every arm set a variable
  named after its label; `KRK_ATTN_BEXACT` was never touched and both arms ran
  the default. A harness bug that makes every arm the same produces exactly the
  verdict "the change is free", so a null result from a new harness is worth one
  `env`/`GetEnvironmentVariable` check before it is believed.
  **Decode's key-range split had to go with it.** A split restarts the online
  max inside every key range and merges the ranges afterwards, so its fp32
  accumulation cannot be expressed by a per-row prefill kernel at all (and its
  ranges start at `j_lo + warp`, which differs per row once a window is
  involved). With the exact prefill on -- the default -- a decode row therefore
  runs unsplit, and `KRK_ATTN_SPLIT=1` is the arm that keeps the split and gives
  up identity from ~128 keys up. That costs decode throughput, and the number is
  not small at long context: the split is worth **+1.09% at a 240-token
  context and +13.85% at 3400** (82.6 against 72.5 tok/s, 5 pairs). Identity is
  still the better default because the prefill win is larger until roughly 750
  tokens of output at that context (3400-token prompt + 32 tokens: 4.36 s
  coupled against 5.61 s with the split). The arm
  shows it: `KRK_ATTN_SPLIT=1 kraken-bench --bitexact` leaves every 64-key
  case identical and makes every 200-key case differ, while the default
  reports 24 of 24 identical.
  **What is left is the GEMM tile rung, from 9 rows up.** On Qwen3.5-9B
  (240-token prompt, greedy, 24 tokens) `--chunk 1`, `2`, `8`, `16` and `256` all
  give `ad25ecdf6854`, but that is text-level luck over 24 tokens: the dump
  comparison from `--chunk 1` against `--chunk 9` (`tools/dump_passes.py`) stops
  at **`gdn.conv L00`** (rel 1.3e-4), the first stage after that layer's
  `[q|k|v]` projection -- a rows=9 GEMM, which the micro-test says is not exact.
  Rows 2..8 are exact end to end: `--chunk 2` and `8` reproduce `--chunk 1` on
  Tiel and on the 9B, and `--mtp-spec --draft-tokens 2` on Tiel now emits the
  plain decode text (`6792671902ca`, 17/36 draft tokens accepted) where it used
  to diverge. The boundary is `gemm`/`gemm_group`: bit-identical at rows 2/4/8
  and different from **rows 9** (q4_k 22/4096 at rows=9 and 83/4096 at 12;
  q6_k ~1770/4096; f16 16/4096; the gate/up silu group 1/512), which is exactly
  where the decode ladder (`gemv_m_kernel`, rows 2..8) hands over to the tile.
  `KRK_GEMM_SPLIT=0` keeps those differences, so the tile's K walk -- not the
  split -- is what differs. Running rows>8 through the ladder is not an option:
  `gemv_m` re-reads the whole matrix per 8-row group, i.e. 32 passes over the
  weights for a 256-row prefill. A windowed (SWA) layer also stays on the tiled
  attention kernel by construction, for the `j_lo` reason above.
- **Concurrency is only safe because every group is self-contained, and the
  runner now enforces both halves of that.** Each child gets its own working
  directory (`<TEMP or TMPDIR>/krk-tests-<pid>/krk-tests-group-N.dir`) -- every fixture path in
  `tests/test_kraken.cpp` is relative, so this is what keeps two groups off each
  other's `kraken-*.gguf` -- plus its own log, printed under the group's line so
  FAIL lines still reach the parent's log, and its own progress file holding the
  partial count. A group that dies keeps its scratch and log and the summary says
  `evidence kept:`; those go on the next run of that group, so read them before
  re-running. `--workers n` / `KRK_TEST_WORKERS` sets the count, default
  `min(8, cores)` (interleaved medians here: 1554 ms at 1 worker, 956 at 4, 783 at
  8, and 8 beat 4 in 6 of those 7 pairs), and a silly value clamps rather than
  refusing. A group may declare a serial reason in the group table and
  `KRK_TEST_SERIAL_OVERRIDE=<grp>` forces one for a run so the lane is exercised,
  but **nothing needs it today** and the run says so: no group opens the device,
  binds a fixed port, or reads a clock. Two defects had to be fixed first --
  tokenizer, end-to-end, grouped-prefill and gdn-generation read fixtures an
  earlier group had written, so they passed or failed according to the schedule
  (`--run-group <n>` on a fresh checkout failed for exactly those four), and
  `test_gdn_generation` reported **0/0 and a pass** when its file was missing.
  The runner now reports `NO CHECKS` for a group that ran nothing and fails the
  run, which is what `KRK_TEST_INJECT_EMPTY` constructs. A side effect worth
  knowing: a crashed run no longer leaves `kraken-*.gguf` in the repo root, and
  neither runner leaves its fixtures in the directory it was started in -- the
  cleanup works by SHAPE (`kraken-*.gguf`, `kraken-*.krakenexperts.json`, a
  `kraken-*-dir` spill area) rather than by a list of names, because the list it
  replaced named six of the thirteen fixtures the groups leave. Both halves of
  that rule are pinned by the isolation check: an empty directory comes back
  empty, and a look-alike that is not a fixture shape (`kraken.exe`,
  `kraken-notes.md`) is left alone.
- **The scratch area is per RUN, not per machine, and a green run removes it.**
  Every run names `<TEMP>/krk-tests-<pid>` -- or `KRK_TEST_SCRATCH_ID` when a
  caller wants to find the evidence again afterwards -- and removes it when the
  run is green, so two suite runs, or a suite run beside a `--selfcheck` sweep,
  cannot write over each other's fixtures, logs or progress files. That is what
  lets the reporting proof run beside a suite run instead of only when nothing
  else is running. A child inherits the parent's id, so a group's paths agree with
  the runner that spawned it. A sweep gives each arm its own id and keeps its own
  evidence in `krk-tests-selfcheck-<pid>`, deliberately *outside* every child
  area: a green suite run removes its whole area, and with the selfcheck dir
  inside one, the clean-run child deleted its parent's report the moment it went
  green and the sweep reported `the clean run is not green (exit 0)`. A run that
  keeps evidence prints `evidence kept:` with absolute paths, and nothing else is
  removed. `scripts/kraken_tests_isolation_check.sh` section 13 is the proof: two
  suite runs and a group sweep started together must all be green and leave no new
  area behind.
- **The reporting proof is in the default gate; the all-groups sweep is not.**
  `scripts/kraken_tests_baseline_check.sh` runs the runner's own selftest
  (`--selfcheck --selftest`, 17 constructed answers) and ONE group's six arms
  (`--selfcheck --group ${KRK_SELFCHECK_GROUP:-0}`) before it compares against the
  baseline, because a gate that compares two reports is worth nothing if the
  reports stopped meaning what they say: a runner reporting a crashed group as
  passing would compare clean against a baseline of the same defect. Both take
  seconds, and the block can be shown able to fail -- `KRK_SELFCHECK_GROUP=99`
  names a group that does not exist, and the gate then exits 1 with `the reporting
  proof failed for group 99 (rc=2)`. The 318-run sweep and its arm-by-arm
  comparison against `tests/kraken-tests-selfcheck-baseline.json` stay behind
  `KRK_BASELINE_SELFCHECK=1` for cost.
- **A missing `return` cannot land any more: `-Werror=return-type` is fatal for every
  translation unit.** That defect class is compiler-visible -- `-Wall` already
  contained `-Wreturn-type`, so the diagnostic was there to read every time and
  nobody read it -- and `CMakeLists.txt` now makes it an error (skipped on MSVC),
  which covers all 90 host TUs and the HIP ones rather than the one function that
  failed. Proved by red construction: a `static bool` that falls off its end appended
  to a clean `src/json.cpp` fails the build with
  `error: non-void function does not return a value in all control paths
  [-Werror,-Wreturn-type]` and ninja rc=1; reverting the file is green again.
- **Changing a `void` helper to `bool` to add a refusal is a two-part edit, and
  only the first part is visible.** `Engine::configure_expert_cache` gained a
  `false` return for a budget that cannot fit; the fall-through exit kept no
  `return` statement, which is UB, and it compiled with **0 errors** while
  `build_check.sh` stayed green. The symptom was not a crash at a source line:
  dense models were fine (they return early), every MoE died with **rc=3 and
  empty stdout**, and `kraken-tests` went 2927/2927 -> rc=1 with four groups
  dead. Two things made it slow to find: the process printed its policy lines
  and then exited *after* them with no message, and the failing path was the
  one path the compiler would only have mentioned in a warning nobody greps for.
  When a signature gains a return value, grep the function's own end for the
  missing `return`.
  The same edit's *refusal* path had a second defect of the same family -- it
  had never been reachable, so nobody had run it: both CLIs answered a failed
  `init` with `delete be; return 1;`, which frees the backend and then unwinds
  past the engine, whose destructor releases buffers through that pointer. The
  refusal printed its line and then died with **SIGSEGV (rc=139)**. It is
  `std::exit(1)` now, in `main_cli.cpp` and `main_server.cpp` both. A new
  early-return path is a new path: run it.
- **A build failure leaves the old `.exe` in place.** A "run" after `ninja rc=1`
  silently executes the previous binary and can look like a pass or a new bug.
  `scripts/build_check.sh` is that check, automated: it asks `ninja -n` whether
  work is still pending instead of reading the log, and it also fails on a Debug
  build type, a CPU-only binary in a HIP configuration, and an offload arch that
  does not match the card in the machine.
- **A Release build can have assertions live, and only `build_check.sh` sees it.**
  `CMAKE_CXX_FLAGS_RELEASE` is filled in by CMake's per-compiler platform module,
  so it is exactly as good as the compiler's *identity*: `build-hip` is
  configured with `CXX=hipcc` (a wrapper CMake cannot classify), so the cache said
  `Release` while `CMAKE_CXX_FLAGS_RELEASE` came out **empty**, the project's own
  `-O3` still applied, and all 90 host `.cpp` TUs compiled with `NDEBUG`
  undefined. `CMakeLists.txt` now states it explicitly
  (`add_compile_options($<$<NOT:$<CONFIG:Debug>>:-DNDEBUG>)`), so it holds for any
  compiler; section 2 of `scripts/build_check.sh` counts `-DNDEBUG` per
  translation unit and is the only check that catches this.
- **`build_check.sh` section 6 flags any rebuild, by design.** The artifact
  hashes are re-recorded only when ninja has no pending work, so after a
  legitimate rebuild the next run reports those binaries as "NOT produced by this
  build". Section 1 (`ninja -n`) is what proves the binary matches the source;
  re-record deliberately with `KRK_REFRESH_FINGERPRINT=1`, then run it once more
  plainly to show the hashes match.

## Cross-engine baselines (Strata)

- Strata is the sibling engine on this box that targets the same card, and its
  numbers are the yardstick for any throughput claim here. Build:
  `G:\Strata\build-hip-win\strata.exe`; data: `C:\Strata-HIP-data\` (packs, models,
  mtp); launcher configs: `G:\Strata\strata-*.json` (they drive `serve/server.py`).
  Its CLI is `strata generate --pack DIR --tokens "1,2,3" ...` -- weights arrive as
  a per-model *pack* plus GGUF shards via `--native`, so a same-file A/B needs a
  pack build first. Its flags worth copying: `--spec N`, `--mtp DIR`, `--kv int8`,
  `--kv-resident N`.
- **It needs `hipblas.dll` on PATH or it exits 127 through a pipeline**, which reads
  as "the binary is missing": `export PATH="/g/ROCM10RT-gfx1201/bin:$PATH"` first --
  the same trap as the llama ROCm oracle above.
- **Recorded comparison (2026-10-09), so it is not re-derived**: on the 256-expert
  `Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL` file, kraken measures **558.8
  tok/s prefill / 35.8 tok/s decode** with the expert tier warm and no speculation,
  against Strata's published RX 9070 XT row (same card class) of 782 tok/s @4K /
  30.8 tok/s decode *with* `--spec 4` + MTP + int8 KV. The gap that reads as "Strata
  is 3x faster" is kraken's cold expert tier -- 16.5 tok/s while promotions fill it
  (`read 438.8 | alloc 92.2 | xfer 675.4 ms` of a 60.6 ms step) -- removed by
  `--expert-warmup 1`. Details and the MTP record: `docs/STRATA-VS-KRAKEN.md`.
- **A drafter is an artifact, not a training project**: the upstream checkpoint
  ships the MTP head (`mtp.*`, 31 tensors, 5.214 GB BF16) and Strata keeps an 889 MB
  `mtp-q2_0.gguf` of it at `C:\Strata-HIP-data\mtp\` with a per-tensor report.
  kraken refuses `mtp.*` shards today (`src/arch.cpp:113`): a loader/extractor plus
  the `--draft` verify loop is the whole gap.
- **`qwen4exp` is the family the cross-engine gap is really about.** The 288-expert
  REAP GGUF (`H:\OLLAMA-Models\GGUF\qwen3.8-flash-next-reap-288-Q4_K_M.gguf`, 1224
  tensors, 48 blocks) is refused by name for `attn_gate`, fused `attn_qkv`,
  `ssm_conv1d` and `ssm_out`; its other classes (`hc_*` hyper-connections, the
  320,001,536-row `per_layer_token_embd` PLE table, the QSA indexer with
  `compress_ratios`) are unread too, while its 288-expert top-10 MoE is exactly the
  shape this engine already runs. Per-arch shape of that work: **gfx1201 (RDNA4)**
  has `v_wmma_f32_16x16x16_bf16` (the BF16 matrix core a BF16 drafter wants) plus
  INT8/INT4 `v_wmma_*_iu8/iu4`, `v_cvt_scalef32_*` for FP8/FP4 and `v_swmmac`;
  **gfx1031 (RDNA2)** has no matrix core, so its path is `v_dot4_i32_iu8` /
  `v_dot2_f32_f16` plus `v_perm` -- what `src/hip/kernels/gemm_dp4a.hpp` already
  dispatches. Both families are already *used* here (wmma on the prefill side,
  sdot2/sdot4 in GEMV and dequant); the unused ones are the matrix-core forms at
  rows>1, `v_cvt_scalef32_*`, and L2 cache-policy bits on streaming reads. The local
  price of the switch is recorded by Strata itself: `STRATA_HIP_WMMA=1` buys about
  30% on prompts and changes the output's last-place bits. The family is now a
  *named* table entry (`qwen4exp`, refused; declared `Moe` and not `RecurrentMoe`
  on purpose, so the gap map keeps naming `attn_qkv`/`attn_gate`/`ssm_*`), and a
  reference engine on this box **runs** the file:
  `H:\LLAMA-bins\rocm10\llama-completion.exe -no-cnv --no-jinja` loads it in 9963 ms
  and decodes at 5.99 tok/s (1.15 tok/s prompt eval, CPU-shaped; `--list-devices`
  does report ROCm0) — a correctness reference, not a speed one.
  `docs/AUDIT-qwen4exp-support.md` has the measured per-piece inventory and the
  CPU-oracle gate each stage must pass before it is enabled.
- **The hyper-connection stack RUNS now (`hc_*`), and two traps came with it.**
  The residual is a lane stack (`ws_hc_`, `[n, hc, n_embd]` = 10240 wide on the
  REAP-288 file); each sub-block is bracketed by `hc_enter`/`hc_leave`, and
  `hc_head_mix` is the final norm (this family ships no `output_norm`).
  (1) **A refusal placed above `model_.load()` is dead code, and it reads as a
  live check.** `Engine::init`'s old hyper-connection refusal sat at the top,
  where `model_.cfg()` is still empty, so `cfg().hc` was false on every file and
  generation from a qwen4exp file ran the plain residual while the comment said
  it was refused. Anything that reads the loader's verdict belongs AFTER the
  load; the block now sits there, and a `PLE`/`QSA` file is refused by name.
  (2) **The combine scatters with the mixer's `inject` OUTPUT, not its weight.**
  `hc_leave` first passed `L.hc_*_inject.data` -- the `[wide, hc]` weight -- to
  `hc_combine`, which reads f16 *weights* as f32 *lane values*: it ran, it
  produced finite text, and it left the four lanes equal, so the only thing that
  caught it was the lane-invariance check.
  The device half is `src/hip/kernels/hc.hpp` (`hc_init`, `hc_mix`,
  `hc_combine`, `silu_act`) and `kraken-bench --hc` proves it against the host
  oracle in `src/hc.cpp` at the real geometry (4 x 2560, 3 tokens): **6 of 6**,
  with `hc_init` and `lanes-equal` required to be BIT-identical and the rest at
  an f16-storage tolerance. The suite cannot open the device, so that arm is the
  only proof the kernels match. `Engine::init` refuses a backend whose
  `hc_supported()` is false rather than running no-op mixes that leave the stack
  at its initial value.
  `test_qwen4exp_forward` drives a real 4-lane-of-2560 file end to end (init,
  generate, the lane stack read back, a lane-varying injection that must
  separate the lanes, a lane-symmetric one whose lanes must stay bit-equal, and
  the full schema -- PLE plus the indexer -- initialising, which is the check
  that replaced the refusal those two blocks used to get); removing the
  scatter fails it 16/17. Its fixture keeps
  `extras=false`: **the arch table says a qwen4exp file has an indexer, so the
  loader now takes `qsa` from the file's own metadata** (`indexer.head_count` or
  `compress_ratios`), because a file without one otherwise cannot be refused or
  run -- it cannot even load. A half-declared indexer is still refused.
- **The PLE block RUNS too, and it cost two wrong answers worth keeping.**
  `ple_block` (src/engine.cpp) runs on the layers `ple.layers` names, BEFORE that
  layer's mixer: host 64-bit hash (`ple_rows_for`) -> rows gathered straight out
  of the mapping (`ple_gather`, never uploaded) -> key/value -> per-lane gate
  (`ple_gate`) -> value broadcast (`lane_bcast_mul`) -> lane-normed -> dilated
  depthwise conv (`conv1d_dw`, dilation = the n-gram size) -> `res_hc += gated +
  silu(conv)`. The conv's window is the last `(kern-1)*ngram` rows that layer
  produced, one per layer in `ple_state_`, which is what makes a chunked prefill
  and a token-at-a-time decode agree.
  (1) **The gathered row is `n_embd` wide, NOT `hc * n_embd`.** `ple_key`
  [n_embd, hc*n_embd] widens it to the lane stack; `ple_value` [n_embd, n_embd]
  does not. On the REAP file both are 2560 = 16 heads of 160, against streams of
  10240, so the mistake looks plausible: the value gemm then reads 128 columns
  out of a 32-wide matrix and its output is inf, which the lane norm turns into
  NaN. That is the shape of it -- a wrong width is an out-of-bounds weight read,
  not a crash.
  (2) **A new Backend virtual with a no-op default is a silent wrong answer
  unless the CPU reference implements it.** `lane_bcast_mul` shipped HIP-only;
  the CPU run then used whatever the allocator left in the destination, which
  read as a **flaky** test -- 12/12 with `KRK_THREADS=1`, ~1 run in 3 failing
  with the pool live, because only then is the heap reused. A result that
  changes with `KRK_THREADS` is a missing implementation, not a race.
  Device half: `kraken-bench --hc` is **10 of 10**, with `conv-history` (a
  3-token call against three 1-token calls carrying the sliced windows)
  BIT-identical -- the chunk invariance of the convolution, on device.
  `test_qwen4exp_ple` is 16/16: it recomputes the rows from the tokenized
  prompt in the test (`(t ^ t-1) % 8` with the fixture's multipliers of 1) and
  requires the engine's gathered rows to match, which fails 11/12 when the
  window drops its predecessor; chunk 8 and chunk 1 must also decode the same
  six tokens; the last four checks are the three terms below.
  (3) **The residual's term is the RAW `value * gate`, not the lane-normed
  copy, and that was the divergence that made this family decode garbage.**
  The reference is `hidden + gated + silu(conv(grouped_norm(gated)))`: the norm
  produces the CONV's input, and `gated` -- un-normalized -- is what reaches the
  lane stack. `ple_block` normalized in place and then added the same buffer, so
  every PLE layer put a per-lane unit-RMS row on the residual instead of
  `value * gate` (a per-lane scale error of ~1/|gate|, around 20x on the REAP
  file). It was found by chaining the reference graph host-side, not by a text
  comparison: at layer 1 of the REAP-288 file, token 0 of "The capital of France
  is", the lane stack normalized through `hc_attn_norm` scored corr **0.373**
  against the dump under the reference's rule and **0.9999** under the old one,
  and every stage before it matched (`ple.gate` corr 1.000000, `ple.gated`
  1.000000, `hc.n[0..1]` 1.000000, all 48 layers' `moe.sum` == routed + shexp
  corr 1.0000 from each layer's own `ffn.pre`). With the split, the same prompt
  and greedy rule reproduce `llama-completion` on that file for **24 of 24
  tokens** ("... The capital of France"); the old code answered `(n,)))` to the
  same prompt.
  `test_qwen4exp_ple` grew to 16/16 for it: `ple_debug_terms` exposes the raw
  row, the conv's input and the conv's output, the test requires the first two
  to be different buffers with different per-lane RMS, and re-aliasing them
  fails **14/16** on exactly those two checks.
  (4) **The rope pairing follows llama.cpp's rope TYPE, and for this family that
  type is IMROPE, which pairs channel j with j + n_rot/2.** The table said
  `rope_neox=false` for qwen4exp while `llm_arch_rope` in the fork answers
  `LLAMA_ROPE_TYPE_IMROPE` (the same case arm as qwen35 and qwen3vl), and
  IMROPE's `rotate_pairs(n_dims, n_dims/2, ...)` is the half-split pairing. Its
  section table (`rope.dimension_sections = [11,11,10,0]`) does NOT change that
  for text: `ggml_mrope_cache_init` advances one `theta` per pair and only picks
  the axis POSITION per sector, so equal positions make IMROPE identical to
  NEOX. With `rope_neox=true` the 24-token continuation matches the reference
  exactly; with `false` it agrees for 19 tokens and then flips (`of France` ->
  `(city`). That flag is what the *reference* says, not something a short arm
  could see: at 8 tokens both pairings emit the reference's text, so an 8-token
  comparison would have called this row clean. qwen35/qwen35moe sit in the same
  IMROPE case arm and still default to `rope_neox=false` here.
- **A reverted test file is invisible to every gate except the baseline's SHAPE
  comparison, and that is how four groups came back.** `test_qsa_select`,
  `test_qwen4exp_qsa`, `test_qwen4exp_gdn_gate`, `test_qwen4exp_hc_mix` and four
  `test_qwen4exp_ple` checks were lost to a `git checkout -- tests/test_kraken.cpp`
  during an earlier pass, while `tests/kraken-tests-baseline.json` -- recorded
  from the tree that HAD them -- kept naming all four, so
  `kraken_tests_baseline_check.sh` said `removed groups (4)` and `docs/`
  described a 16/16 PLE group the suite ran as 12/12 while `build_check.sh` and
  `ninja` stayed green. The source was still in the desktop app's message store
  (`~/.config/freebuff-desktop/projects/<project>/desktop-v2.db`,
  `messages.parts_json`: the `str_replace` `newString` payloads and one unified
  diff), re-applied, and the groups came back at **exactly the baseline's
  counts** -- 14 / 17 / 14 / 7, and 16 for the PLE group -- which is the evidence
  that this is the same code and not a rewrite. The same revert had dropped the
  fixture's `vocab = 261` fix, leaving a 260-row embedding against a 261-piece
  tokenizer: one row past `token_embd.weight` read for every space, so the same
  command printed a different `--prefill-hash` each run and the QSA selection
  flipped on near-ties. That is back too. When a test file is reverted, the
  baseline's `removed groups` line is the only witness -- read it, not just the
  total.
  The escape hatch's warning changed with it: `KRK_ARCH_ALLOW_EXPERIMENTAL=1`
  used to be reported as "generation will refuse", which was never true --
  `Engine::init` has no verdict check -- so it now says the loader's blockers are
  not enforced for the run.

## Line endings (this bites constantly)

- `src/hip/backend_hip.hip`, `src/common.cpp`, `src/model.cpp`,
  `include/krk/*.hpp` are **LF in git**. `src/engine.cpp`,
  `tests/test_kraken.cpp`, `src/sampler.cpp` are CRLF.
- Editing tools and Python round-trips can silently convert a file to CRLF.
  That is invisible in a diff of the code but makes every multi-line
  string-replace anchor fail with a confusing assertion. Check with
  `s.count('\r\n')` before blaming the anchor, and normalize with
  `raw.replace('\r\n','\n')`.

## Patching files safely

- **Multi-line Python patching: use a quoted heredoc (`python3 <<'PYEOF'`), not
  `python -c "..."`** — backticks and `\n` in the payload get eaten by bash.
- **A `\n` inside a C++ string literal survives exactly one level of escaping.**
  Through JSON→bash→Python it becomes a real newline and breaks the build with
  `expected ')'`. When writing a format string, build the backslash explicitly
  (`BS = chr(92)`) or write the patch to a `.py` file and run that.
- Prefer line-indexed edits for anything spanning lines; assert on the exact
  text first so a mismatch fails loudly instead of silently no-op'ing.

## Environment quirks

- Bash arithmetic: `$((n//3))` is invalid — use `$(( n / 3 ))`.
- `/usr/bin/time` does not exist here; measure wall time in the shell or in-process.
- HIP launches use the **default stream** (`<<<g, t>>>`); there is no `st_`
  member. `__shfl_xor(v, off)` (not `__shfl_down_xor`); `kWaveSize = 32`.
- Anything touching HIP must go through `hipcc` — keep HIP headers out of
  plain-C++ TUs. `src/main_cli.cpp` bridges via environment variables instead.
- Machine: 96 GB host RAM, 15.9 GiB VRAM, models also under `G:/More-models/`.
- **Stopping a model sweep needs a TREE kill, or the next link fails.** A
  script that walks many models spawns `timeout` -> `kraken.exe` children;
  killing the shell leaves them running (they keep loading, and hold the
  binary), and ninja then dies with `lld-link: error: failed to write output
  'kraken.exe': permission denied` followed by `clang++: error: unable to
  remove file`, which reads as a toolchain problem and is not one.
  `taskkill //T //F //PID <shell pid>` kills the tree; `tasklist //FI
  "IMAGENAME eq kraken.exe"` (or `ps -W`) shows the strays, and `ps -W | grep`
  alone can miss them, so confirm with tasklist before blaming the linker.
- **Host memory: read the number, do not infer a leak.** A GPU run of a 27B model
  peaks at ~0.35 GB private / ~0.37 GB working set; the ~267 GB "virtual" figure
  Task Manager shows for it is the HIP runtime's address-space reservation under
  `HIP_VMEM_MANAGE_SUPPORT=1` and is **not committed memory**. A `--cpu` run is
  ~7.0 GB private / 13.6 GB working set, because the engine copies the quantized
  weights into backend buffers and on the CPU backend that memory is host RAM.
  The real host spend on a MoE is the expert WARM tier: the auto RAM policy hands
  it about half of free RAM as a budget (measured `ram policy: 37.6 GiB budget
  (95.9 GiB installed, 75.2 GiB free)` on this box) and the tier then takes
  min(corpus, budget), so a large-corpus model can legitimately commit tens of GB.
- **The ROCm oracle needs the ROCm runtime bin on PATH.**
  `H:/LLAMA-bins/rocm10/llama-cli.exe` loads `ggml-hip.dll`, which fails to load
  without `G:\ROCM10RT-gfx1201\bin` on `PATH`; the loader then reports an *empty*
  error and continues **CPU-only**, so a "ROCm" run silently becomes a CPU run
  (measured 2.7 tok/s against 35.8 for the same model and prompt). Export the bin
  dir first and confirm `--list-devices` prints `ROCm0` before believing any
  number from it. Run it bounded: `llama-completion.exe -no-cnv --no-jinja` with
  stdin from `/dev/null`; without `-no-cnv` it enables conversation mode and sits
  at a `> ` prompt, and an unbounded `llama-cli.exe` invocation once spun into its
  REPL and wrote a ~100 MB log.

- **Sampled speculation is a DIFFERENT verification rule, and the head path now
  runs it.** Greedy verification is a prefix match; under `temp > 0` that rule is
  wrong in both directions -- accept-every-match emits the draft's distribution,
  reject-every-miss emits the target's but never keeps a token. The round now
  verifies with Leviathan et al. (arXiv:2302.01318): the proposal is a DRAW from
  the head's row (softmax at the run's temperature, `spec_softmax` +
  `spec_map_dist` through `d2t`), accepted with `min(1, p/q)`, replaced on
  rejection by a draw from `max(0, p - q)`. `src/spec_sampling.cpp` holds the
  rule as pure host maths; `SpecStep spec_accept_step()` takes the two uniforms
  so a round replays from a seed, and `spec_verify_identity()` enumerates the
  identity `sum(min(p,q)) + sum(p-q)+ = p` per token. **`p` is
  `Sampler::build_dist`, the row `sample()` itself draws from** -- the sampler
  was refactored onto it for exactly this reason, and the suite's sampled-text
  pins still reproduce. Repetition penalty needs the round's already-accepted
  tokens in the history, which is the `extra` argument (a token accepted for a
  position being verified is not in `Sampler`'s ring yet).
  **The group can fail, and that is measured:** `test_spec_sampling` is 38
  checks, and an always-accept `spec_accept_step` fails four of them -- the
  accept-ratio pair, the residual-replacement check, the one-hot limit and the
  200k-draw histogram of the rule's own outputs against `p`. The enumerated
  identity alone does NOT catch it (it recomputes the algebra instead of calling
  the rule), which is why the histogram exists.
  **End-to-end, with a random head every proposal is rejected, so the measured
  arm is the worst case:** qwen2 1.5B, ctx 512, `--temp 0.9 --seed 7`, 24 tokens,
  `--draft-tokens 3` gives `24 round(s), 0 of 72 proposals kept (0.0%), 1.00
  token(s)/round` and fluent text -- 24 consecutive residual draws, re-forwards
  and realigns. Greedy is unchanged: md5 `f7957feb5610` with and without the
  head. `KRK_SPEC_RS` keeps its meaning from the draft-model/dflash verifiers
  with the default flipped for this path (they are opt-in, a head under sampling
  would otherwise have no drafter at all), so `KRK_SPEC_RS=0` is the arm that
  returns the plain loop.
  **Two defects that arm exposed, both worth remembering.** (1) On a GREEDY
  partial accept the round rolled the KV back and continued with `logits_host_`
  still holding the pre-round row -- the next round then proposed from a position
  the sequence had left. It now downloads the verifier's row at the rejecting
  position (`head_compute(a)`), which is exactly the distribution the plain loop
  would have there. It never showed before because no fixture head ever had a
  partial accept: the synthetic heads accept 0%. (2) A rejection's replacement
  must be RE-FORWARDED: the KV row at that position still holds the rejected
  proposal, so the sample is emitted and then fed back (the same move the full-
  accept bonus already made), and only then is the round rolled back.

- **A nextn block can arrive in a second file, and three things about it are
  not guessable from shapes.** `--mtp FILE` (`Model::set_mtp_path`, read in
  `load_mtp_block`) loads `blk.<n>.*` out of a sidecar: the `-noMTP` quant plus
  `mtp-<model>.gguf`, which is how Nemotron-3.5-Lightning ships. Measured on
  `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf`: block 52, 32 heads x
  128, 128 experts x top-6 of 1856 + shared expert, probe **MATCH** on the first
  step (21.0 ms), `--mtp-spec --draft-tokens 2` at `6 of 28 (21.4%), 1.14
  token(s)/round`.
  (1) **The block index** is the smallest `blk.<n>.` at or past the trunk's
  `n_layer`, read from the tensors -- a sidecar's `block_count` counts only its
  own blocks, so `block_count - 1` is right for this file by luck and wrong in
  general. (2) **`<arch>.attention.head_count_kv` can be an ARRAY**, one entry
  per layer (nemotron_h_moe ships 53); reading the scalar key gives the first
  layer's count and a KV plane that is the wrong width. The block reads its own
  entry, and the counts then have to equal the trunk's -- the loader refuses a
  sidecar with a different head count, because that is a head from another model,
  not this trunk's. (3) **An UNGATED expert bank has no `ffn_gate_exps` at all**,
  and reading `n_expert`/`n_ff_exp` from the gate made this block load as "no
  experts and no dense FFN" (the probe refused a block that was right there).
  The width comes from whichever of the three tensors exists: up/gate carry it
  in `ne[1]`, down in `ne[0]`. The same applies to the shared expert.
  The suite covers all of it in `test_mtp_sidecar` (17 checks: happy path, wrong
  family, no block past the stack, a disagreeing head count, and a trunk with no
  path at all).

## Measuring performance (the expensive lessons)

- **Never trust a probe whose arms ran in sequence.** Page cache and
  driver state warm between arms, and the ordering alone is worth several GB/s.
  Every A/B must be interleaved run-by-run, and a win must hold across
  repetitions. This session produced three separate false conclusions from
  sequential probes before interleaving was introduced.
- **A decode step's cost is its op COUNT, and two folds that remove two launches
  per layer are default-on now.** `dst += W*x` in the GEMV epilogue
  (`gemv_kernel<256, true>`, `KRK_ACC_GEMM=0` restores `gemm()` +
  `add_inplace()`) and gate/up + silu in one launch (`fused_gate_silu_kernel`,
  `KRK_FUSED_SILU=0` restores the separate `silu_mul()`). Both are
  bit-identical by construction, not approximately equal -- the residual form
  rounds the projection to the activation type before summing, as the pair
  stores it, and the silu form rounds both operands before the activation --
  and the proof is that stdout md5 is unchanged. Measured on SmolLM2-135M
  (ctx 512, n 64, greedy, 5 interleaved pairs per run, text identical): residual
  fold **+2.867%** (5/5), silu fold **+6.135%** (5/5), both **+8.347%** (5/5);
  the op table goes **303 -> 243 ops/step**, the `residual` and `ffn-activate`
  component rows disappearing (60 = 2 per layer x 30 layers). **Sizing a fold
  from a per-op constant is good to ~2x and nothing better; run the fold.** Re-measured
  on the current build (2026-10-11, 9 interleaved order-balanced pairs each, 0/9
  pairs, md5 identical): `KRK_ACC_GEMM=0` (the residual fold, +30 launches) costs
  **-5.219%** and `KRK_FUSED_SILU=0` (also +30) costs **-4.591%**, i.e. **2.4-2.8 us
  per added call** -- which includes that fold's extra memory pass -- where the
  "1.5 us" this bullet used to quote came from the same fold on an older build and
  reproduces at neither end now. `kraken-bench`'s floor is **2.85-3.05 us** for a
  trivial call in a quiet queue and **0.92 us** for the host-side enqueue of the
  same call into a busy one, so the floor is within ~10% of a call that carries a
  pass and overstates a pure enqueue ~3x.
- **A fused group's fallback runs on PREFILL, and it must use each matrix's own
  metadata.** `gemm_group`'s fused path is decode-only (rows == 1), so a prefill
  pass takes the per-matrix fallback -- and a fallback that decodes a Q4_K
  matrix with the group's single `wt` (Q5_K here) reads every block from the
  wrong place. Measured on Qwen3.5-0.8B: `KRK_DUMP` `gdn.delta` came out
  `-nan(ind)` against `-1.66893e-06` from the separate-launch arm, i.e. a
  wrong-answer bug that leaves the op count and the timing looking healthy. The
  md5 comparison across arms found it in one run; `FusedLayer` now carries
  `wt[8]` (per-matrix types, nullable `wts[]` on the launcher and `gemm_group`)
  and both fallbacks use `wts[m]`.
- **Four projections that share an activation are one launch where the backend
  can express it.** `Engine::gdn_forward`'s `[q | k | v]`, z gate, `ssm_alpha`
  and `ssm_beta` all read the attention-norm output and have no dependency
  between them, but were issued 4 launches per recurrent layer -- `KRK_GDN_GROUP=0`
  restores that. Measured on Qwen3.5-0.8B (18 recurrent layers of 24), ctx 512,
  n 64, greedy, 5 interleaved pairs: **451 -> 397 ops/step** (-12%, `gemm(gemv)`
  2822 -> 590 calls) and **+6.084% decode** (MAD 0.200%, 5/5 pairs), with stdout
  md5 `08afb598d17b0c7ac6466d824163e1a6` unchanged on all ten runs and identical
  to the pre-change baseline -- the fold is bit-identical because the group runs
  `row_parallel_only` (splits pinned at 1), which is also what keeps its tiny
  `dt_rank` matrices from paying a reduce launch each.
- **`test_gdn_ops` carries a mixed-dtype `gemm_group` case, and it fails on the
  defect it was written for.** `test_gdn_group_matrix_types` runs three
  projections (F16, F32, Q8_0 -- three different bytes per row) through
  `gemm_group` with an explicit `wts[]`, at rows 4 (the per-matrix fallback, the
  shape prefill takes) and rows 1, and compares each against that same backend's
  per-matrix `gemm()` with its own type; a fourth check proves the misread is
  visible at all. Reintroducing the bug (the fallback handing the group's single
  `wt` to every matrix) fails it **134/138 with four FAILs**, on exactly the two
  matrices whose type differs from the group's -- which is why the reference is
  never the group against itself. The device-side half is unreachable from the
  suite (no group opens the device) and is pinned by `KRK_DUMP` + md5 instead.
- **`env A=0,B=0 cmd` is ONE variable, not two.** The value is the literal
  string `"0,B=0"`, which any `strcmp(v, "0") != 0` test reads as truthy, so a
  comma-separated arm list silently runs the same configuration in both arms.
  `scripts/kraken_sherlock_kernel_ab.sh` hit exactly this and reported NO
  DIFFERENCE (median -0.52%, MAD 0.70%) on an experiment with no arms; it now
  splits the list into words first. Every harness here that takes switches from
  the environment needs the same check.
- A/B switches must exist *before* the measurement, not after: `KRK_WEIGHT_PULL=0`
  and `KRK_ATTN_BEXACT=0` both restore the previous path so the comparison is one
  process apart. A switch that names a deleted kernel is worse than none --
  `KRK_ATTN_QTILE` was one until the query-tiled kernel went, and setting it
  silently left the model on the default.
- Instrument the wall time of the thing you are changing. Thread-summed
  per-stage timers hid a pipeline that was not actually overlapping — the sums
  looked healthy while throughput was 2.6× below the probe.
- `--profile` self-calibrates its own floor at startup (~0.7 µs host / ~20 µs
  device per record). Do not compare op timings against a nominal constant.
- **There are two profilers and a step needs both.** The device table
  (`src/hip/op_time.hpp`) times kernel spans and lives in the HIP TU; the host
  table (`include/krk/host_time.hpp`, `KRK_TIME_HOST`) times whole phases from
  portable code and prints underneath it as `[stats ] host time`. A step whose
  `decode.step` is large while `decode.forward` is small is host-bound -- which
  is what "the CPU is busy and the GPU is idle" looks like in numbers. Both are
  enabled by the same switch, so `--profile` never reports half the story -- but
  read the next bullet before concluding anything from either table's magnitudes.
- **The profiler is not a passive observer: on a decode step it is the largest effect
  in the picture.** Same binary, same command, only the switch changed -- SmolLM2-135M
  is **624.2 tok/s (1.6 ms/step) unprofiled against 61.6 tok/s (16.2 ms/step) with
  `KRK_PROFILE=1` (10.1x)**, and Qwen3.5-0.8B is 356.8 against 35.7 (10.0x). The
  recorder's floor is ~20 us per record, two records per op, ~303 ops per step: about
  14 ms, and the arithmetic closes (14.4 ms + 1.6 ms of real work ~= the 16.0 ms
  profiled step). So the device *idle* such a table reports, and a large
  `decode.fetch`, are the records the profiler inserted into the stream -- `forward`
  looks cheap because host records are ~0.7 us, and the fetch looks expensive because
  it waits on device events. Read the table for what ran and in what order (counts,
  ranking, shares), never for how long: even its floor-removed NET column reports
  8.1 us of gemv device time per call where an entire step is 5.3 us per op. A decode
  step is **launch-bound**: 303 ops at 5.3 us and 517 ops at 5.4 us on two models whose
  weight traffic differs 6.7x, and the per-op cost is 0.9-2.8 us depending on
  whether the call carries a memory pass, so **17-53% of the step is launch cost**
  (the "~69% at 3.64 us" this used to say is the top of that band, not the middle).
  Time a comparison WITHOUT `--profile`
  (`scripts/kraken_sherlock_kernel_ab.sh` does: interleaved reps, and a verdict that
  needs an ORDER-BALANCED median paired gain above the pairs' own MAD in at least
  reps-1 pairs). Balanced because the slot a run occupies inside a pair is worth
  several percent here -- 9 pairs, one arm swapped between the slots, read 124.5
  against 132.6 tok/s on one arm and 135.7 against 142.1 on the other -- so a raw
  median reports the slot effect as the arm's. The loop now median-splits the deltas
  by order and averages the two figures, prints both, and uses the balanced one; its
  floor is stated by the control, the SAME arm in both slots: +2.885% balanced, MAD
  2.124%, 7/9 pairs, NO DIFFERENCE. Size a kernel with `kraken-bench`'s marginal
  timing, which exists for exactly this.
- The device table CANNOT nest (one `cur_` slot), so never open a host span
  around backend ops expecting a device breakdown inside it. The host scopes are
  `spec.round`, `prefill.chunk`, `decode.step`, `decode.forward`, `decode.fetch`.
- **KV tier traffic is reported at the end of every run**, dense models
  included: `[stats ] kv tiered: ... promoted WARM->HOT n, COLD->HOT n, evicted
  ... n MiB migrated`. It printed nothing before because the only reader was the
  load-time one-liner and the MoE block returned first. One HOT slot
  (`--kv-hot-mb 1`) takes SmolLM2-135M from ~54 to ~2.4 tok/s -- that is the
  shape of a tier budget smaller than the working set, and it is now visible
  instead of inferred.

- **Which expert-tier limit binds depends on where the working set sits, and it is
  measured (2026-10-09, the 256-expert 35B-A3B, RX 9070 XT).** Corpus resident +
  `--expert-warmup 1`: **35.8 tok/s** decode, 558.8 tok/s prefill, and read/room/
  alloc/xfer all 0.0 ms. The same run *without* the warm-up pays first-touch fill
  instead -- `read 438.8 | alloc 92.2 | xfer 675.4 ms` of a 60.6 ms step -- at
  16.5 tok/s. At `--expert-cache-mb 256` with WARM staged, COLD is 0.0% and reads
  are 0.0 ms, yet the step is 142.7 ms because **alloc 2445.5 ms of a 3408.1 ms
  promote total, with 6405 device evictions**: the allocator, not the file, and not
  the 350.4 ms of H2D. So reads bind only when WARM is cold, the allocator binds
  when VRAM is tight, and the per-layer barrier is the hybrid-CPU arm's cost (3.1x)
  rather than the device path's. Table in `docs/RESEARCH-REFERENCES.md`.

- **Dense weights stream now, and the ring has two traps.** `Model::load` keeps
  the layers that fit and stages the rest through one device ring
  (`--weight-vram-mb`, 0 = auto `free - reserve - 1 GiB`, negative/`KRK_WEIGHT_STREAM=0`
  = off) before each layer runs. Two things are correctness, not tuning:
  (1) the staged copy is published TO the default stream, not FROM it, so the
  ring must be drained (`be_->sync()`) before the next layer is read into it --
  without that, layer l reads layer l+1's bytes and the output is fluent
  garbage (measured: `!!!...`, f16 range counters fired); (2) routed `_exps`
  banks must be excluded from a layer's byte total, or a 25 GiB MoE looks like
  dense layers that do not fit and streams 20 layers that do. Verified
  (3) a streamed layer is **promoted into a permanent allocation on first use**
  (`Model::promote_layer`) once measured free VRAM can hold it beside the floor.
  The load-time budget is decided before the KV cache and the workspaces exist,
  so it under-uses the card by ~2 GiB here: on Qwen3.8-27B the auto arm left
  3.08 GiB free while re-reading 17 layers every token, and promotion kept 9 of
  them (measured **1.2 -> 2.3 tok/s**, 5/5 interleaved pairs, stdout identical;
  `KRK_WEIGHT_PROMOTE=0` restores the fixed prefix). Promotion re-points the
  layer's `QuantTensor`s through a binding pass (`StreamEntry::q`, bound by
  `qt_fields()`), so `unload()` must skip pointers inside a promoted base as
  well as inside the ring -- releasing each interior pointer is a double free.
  (4) staging is a two-buffer pipeline: two pinned host buffers alternate, the
  copy is `Backend::upload_pinned_batch` (a DMA straight out of the pinned
  buffer on a private stream, published to the default stream by one event, so
  the host only enqueues), and the NEXT layer's read runs while those bytes are
  in flight. Measured on the same 27B: **2.2 -> 3.0 tok/s, +36.4%** (5/5
  interleaved order-balanced pairs, 0.000% MAD, stdout identical), staging
  4.42 -> 5.61 GB/s, and the split changes shape from `read 3347 + copy 2710`
  to `read ~376 + copy ~12` ms of enqueue -- the read of the next layer is the
  remaining bound. `KRK_WEIGHT_PREFETCH=0` / `KRK_WEIGHT_DMA=0` are the A/B
  arms. Two traps found here: `stream_` is EMPTY when nothing streams, and
  indexing it in the binding pass segfaulted every qwen35moe 35B file on load
  (fixed with a bounds guard); and a "changed md5" between arms was the BANNER
  -- the device line gained a BAR suffix, so compare stderr-free stdout or
  strip the first line before concluding a numeric difference.
  Verified bit-identical on Qwen3-8B (forced 3 GiB budget: resident vs streamed
  vs promoted all `6b10bb51...`) and on the 27B across the promotion arms.
  Streaming makes >VRAM models *bounded*, not fast: a token must move the
  non-resident bytes, so 1.2 -> 3.0 tok/s is what the extra residency and the
  overlap buy. The CPU backend never streams.
  Measured loading-mode table (memory / auto / 2 GiB ring with
  `KRK_WEIGHT_PROMOTE=0`) with commands: `docs/BENCH-LOADING.md`. Headline:
  Qwen3-8B Q4_K_M 92.3 -> 2.5 tok/s decode when 25 of 36 layers stream; a 27B
  Q4 that fits at 14.4 GiB is 14.9 resident, 8.6 on the auto 12 GiB budget and
  0.5 on the ring; a 30B-A3B is unchanged in all three arms because the ring
  never carries the routed expert banks (0.71 GiB of trunk).

- **An expert budget that fills the card is not slow, it is pathological, and it
  is the MoE slowness.** `configure_expert_cache` decides the device budget
  before the KV cache and some of the engine's own workspaces exist, so on a
  card whose routed set is larger than VRAM it can name every byte free at that
  moment. The pool then grows to the last byte, `hipMalloc` has to make the
  driver evict, and every promotion pays for it. Measured on
  `Tiel-Coder-35B-A3B-UD-Q5_K_XL` / `Tiel-Coder-35B-A3B-MTP-APEX` (256 experts,
  top-8, 40 layers, 22.1 GiB routed, ctx 4096, 16 tokens, greedy; WARM holds the
  whole routed set in RAM in every arm, so **no arm reads the file**):

  | pool budget | free VRAM at end | alloc per cell | promotions | decode |
  |---|---|---|---|---|
  | 11.95 GiB (auto) | 0.00 GiB | **30 ms** | 1360 | 3.1 tok/s |
  | 10.0 GiB | 0.00 GiB | 1.2 ms | 1530 | 6.2 tok/s |
  | 8.56 GiB (2 GiB floor) | 1.77 GiB | 0.27 ms | 1712 | 17.8 tok/s |
  | 7.68 GiB (3.26 GiB floor) | 2.95 GiB | 0.25 ms | 1850 | 16.6-18.7 tok/s |

  The split says it in one line: `alloc 40426 ms | read 0 | xfer 424` against
  `alloc 460 ms | read 0 | xfer 372`. The fix is two-part and both parts are
  measured: the warm-up stops at `vram_floor_bytes()` instead of the 1 GiB
  reserve, and `Engine::init` **clamps the pool after the warm-up** to
  `resident_bytes() + max(0, free - floor)` -- the only point where both numbers
  are facts.

- **The expert index is used, and its path is derived from the model's.** A run
  with `<model>.krakenexperts.json` beside the weights prints
  `expert warmup: using <path> (512 positions, full-forward, 2944 ranked pairs)`
  and fills the tiers hottest-first; the reader validates `kind`, `version`,
  `n_layer`, `n_expert` and `arch` and REFUSES a mismatch with a warning, so a
  ranking from another model cannot reorder this one silently. Two ways it looks
  unused: the warm-up did not engage (the index is read there, not at load), and
  the `.gguf` was renamed or moved so `<model_path>.krakenexperts.json` no longer
  resolves -- the index is path-derived, so a copy of the weights needs a copy of
  the JSON. Measured on Nemotron-3.5-Lightning-30B-A3B NVFP4 (128 experts x top-6,
  52 layers, 2 GiB warm tier): `ranked order, 2 rounds ... 382 slots (2093 MiB
  read), VRAM +382 experts (2045 MiB)` in 2709 ms, and 33.3% of router picks
  landed on a resident expert with the residency preference off.
- **Batched-MoE look-ahead exists and is a measured LOSS, so it stays off.**
  `ExpertCache::stage_batch` promotes the WARM experts a batched pass's rows will
  need before the first row runs it -- the union is known a row early because the
  router runs on the host -- and `KRK_MOE_LOOKAHEAD=1` turns it on. Measured on
  the same Nemotron (600-token prompt, `-n 1`, ctx 1024, `--expert-warmup 1
  --expert-warm-mb 2048`, 3 interleaved pairs): prefill
  **124.9 / 218.7, 270.9 / 249.3, 275.5 / 242.2 tok/s** for
  `KRK_MOE_LOOKAHEAD=0` against `=1`. The first pair reads as a 75% win and is
  cold state (that run paid 4805 ms where the warm ones paid ~2200); the two warm
  pairs are 8-12% SLOWER. **The decode arm was measured too, because a single
  row also knows its union before the expert loop** (`n == 1` is the gate now):
  24.7 / 24.6, 25.5 / 25.9, 25.1 / 25.3 tok/s, promotion count identical
  (1592), i.e. no difference -- promotions are already overlapped by
  `upload_pinned_batch`, so batching them earlier buys nothing. Staging moves
  the same bytes earlier and pays an extra budget check per candidate. Do not
  re-derive it as a win; the arm worth measuring is a verify pass with a drafter
  that actually accepts.
  **And the step it cannot fix is worth more than the staging was:** on this
  model the warm arm spends 878 ms of its 1214 ms decode in promotion (8521 MiB
  at 9.5 GB/s, 1592 promotions over 31 steps), because a 6.4 GiB pool holds 18%
  of a 15.4 GiB routed set and 22% of the 312 expert visits per step miss. The
  lever is traffic, not scheduling: a bigger pool, more residency preference, or
  fewer visits -- not a cleverer prefetch.

  **Which experts the fill keeps matters more than how many.** The same model,
  same pool (7.68 GiB), same floor, differs by 44% on the fill alone: an
  unranked even-order warm-up gives 16.6-18.7 tok/s, and a **ranked** one --
  `--expert-scan` writes `<model>.krakenexperts.json`, the next run uses it --
  gives **23.9 tok/s** (41.8 ms/step) with 671 promotions instead of 1850,
  same text, 3.02 GiB free. So the ladder on Tiel-Coder-35B-A3B is
  **3.1 -> 16.6 (floor) -> 23.9 (ranked)**, and the ranking is one command.
  A *larger* pool is slower here, which is the opposite of the intuition and is
  reproducible: 3.26 GiB floor 23.9, 2 GiB floor 20.6, 1 GiB floor 17.6 -- a
  bigger resident set spreads into colder experts (more promotions, and
  `make_vram_room` scans more slots per eviction) and eats the headroom the
  allocator needs. Do not widen the pool to win this; rank it. The floor is `max(reserve + 1 GiB, total/5)` = **3.26 GiB** on this
  card (`KRK_VRAM_FLOOR_MB` overrides), and a card with 0 free is also what
  stalls the Windows compositor while the desktop is paged out. Two traps: the
  clamp must be skipped on the CPU backend (no device/host split; it clamped the
  suite's CPU MoE fixture to a zero-byte cache, 2992/2994), and the growth cap
  only allows what is really free, so a full cache evicts-and-reuses instead of
  allocating -- which is the path the pool is built for.


- **A promoted expert was paying a memcpy, not a DMA, and pinning the hot prefix
  fixes it.** `warm_fill` allocates WARM pageable, so every promotion was a
  12.6 MB host memcpy into the bounded pinned ring (~1.1 ms) PLUS the 0.67 ms DMA
  -- measured 7.6 GB/s and 1623 us/expert against a `kraken-bench --h2d` floor of
  19.8-27.2. `--expert-pin-mb N` (`KRK_HOST_PIN_MB`) allocates the first N MiB of
  the tier with `alloc_host` -- admission order, which is RANK order when the
  model has an expert index -- and `promote_to_vram` sends it with
  `upload_pinned_batch`; the COLD-miss path does the same for the buffer the read
  just filled. `drop_host` must free a pinned slot with
  `release_host_pinned_deferred` (hipFree on a hipHostMalloc block is a
  use-after-free that shows up as varying text, not a leak). gpt-oss-120b Q8_0,
  ctx 512, 24 tokens, greedy, warmup on, interleaved order-balanced pairs, stdout
  identical in every arm: 16.4 -> **20.5** tok/s at `--expert-warm-mb 11000
  --expert-pin-mb 8192` (**+25.0%**, 6/6, MAD 2.0%), and 20.3 -> **23.9** with a
  12 GiB pin over the 34.5 GiB auto tier (+18.3%, 5/5, MAD 6.1%). Three facts to
  keep: **a pin below the promotion working set is worth exactly zero** (4 GiB
  halved the copy time, 1020 -> 521 ms, and moved the step by nothing -- the
  unpinned half is still on the critical path; 8 GiB covered all 7098 MiB the run
  promoted and that is where the gain is, so size it to promotion BYTES, not to a
  fraction of RAM); **the `transfer` GB/s is an ENQUEUE rate on the pinned path**
  (its timer wraps `upload_pinned_batch`, which returns while the copy is in
  flight -- a fully pinned arm printed 64.6 GB/s on a link that does 27, which is
  what `[stats ] pinned` now reports honestly from events on the copy stream, and
  that reads 6.1-24.9 GB/s depending on what else the device is doing, so the
  step time is the cost and the line is a diagnostic); and **the suite cannot see
  the deallocator pairing** (on the CPU backend both host frees land in
  `release_host`, so routing a pinned free through the pageable path still reads
  72/72 -- measured with the red construction). It stays opt-in: pinned pages
  cannot be swapped, and on the 24 GB/12 GB target shape the honest number is
  16.3 -> 20.5 tok/s, with the rest of the gap the 283 MB/token an 11 GiB tier
  cannot cover.

- **A step on a tier that does not fit is paid for in experts moved, so the
  router now breaks its own near-ties toward what is resident.**
  `--router-resident P` (`KRK_ROUTER_TIE_MARGIN`): among candidates whose score
  is within `P` of the top (a fraction of it), take the one already in VRAM,
  then the one in WARM. Residency is read LIVE from `ExpertCache::in_vram/
  in_host` rather than from a stored rank, because the index is what filled the
  tiers and evictions invalidate a stored rank immediately. The mix weight stays
  the unbiased probability. Measured on gpt-oss-120b Q8_0 (11 GiB WARM, 8 GiB
  pin, warmup on, 24 tokens, greedy, order-balanced pairs, text identical):
  `0.10` moves 11.4% of picks and takes 283.65 -> **226.92 MiB/token (-20.0%)**,
  563 -> **462 promotions**, 20.3 -> **23.7 tok/s (+16.7%**, 5/5, MAD 0.1%);
  `0.20` is -40.7% bytes and 26.7 tok/s. On `Qwen3.8-Distill-35B-A3B`, where
  nothing is COLD at all, 0.20 still measures **+14.5%** (5/5, MAD 1.6%) from
  promotions between tiers alone. Three things to keep: **the byte counters are
  exact and repeat** (563 / 283.65 five times), so one run per arm settles the
  byte claim and repetitions exist only for the rate --
  `scripts/kraken_router_bytes_ab.sh` is the harness, because the kernel A/B
  refuses to compare arms that print different text and this change is allowed
  to; **the switch-off path is bit-identical** (`9f8be8ea52` before and after,
  which is what "off" has to mean); and **the in-suite unit test can only assert
  the wiring**, not the substitution, because the CPU-backend fixture derives its
  expert budget from live free RAM -- under a parallel suite that budget can come
  out at zero, so a test asserting "a resident pick was taken" passes alone and
  fails in the suite. That shape is measured end to end instead.
- **Resizable BAR: the runtime's answer is a hint, the GB/s is the fact.** The
  device line now ends with `BAR: runtime reports ...` from
  `hipDeviceAttributeIsLargeBar`. On this box the firmware has ReBAR enabled
  while the runtime reports a 256 MiB aperture, so do not treat the attribute as
  the BIOS setting on Windows/WDDM -- read it as "what this runtime will
  expose". The measurement that matters is the staging rate: 5.6-6.5 GB/s
  aggregate on a 4-reader staged read from a WARM page cache, and 3.9-4.8 GB/s
  for expert promotions out of pinned RAM.

- **Tiel-Coder-35B-A3B ships its own MTP head; the block now LOADS, the drafter
  does not run yet.** `qwen35moe.nextn_predict_layers=1` and `blk.40.*` is the
  standard Qwen nextn head: `nextn.eh_proj` [4096,2048] (concat of hnorm(hidden)
  and enorm(embed(last token))), `nextn.enorm/hnorm/shared_head_norm`, a full
  attention block (16 heads x 256, 2 KV heads, Q8_0), and a 256-expert top-8 MoE
  (~855 MB of the block's ~897 MB). `Model::load_mtp_block` (KRK_MTP_LOAD=1)
  uploads the block plus the adapters and reports the geometry; the routed
  experts stay in the file for ExpertCache. With the switch off, a run is
  byte-identical (checked on Tiel, stdout md5 unchanged).
  **The block's forward now runs, and it drafts the target's token.** `--mtp-probe`
  (KRK_MTP_PROBE=1) captures the target's pre-final-norm hidden row at EVERY
  position (one activation copy per forward, into `mtp_hidden_`), gives the
  block its own K/V plane pair (`mtp_kv_k_`/`mtp_kv_v_`, sized
  `ctx x 2 KV heads x 256`), and replays the prompt through it: eh_proj over
  `[enorm(embed) | hnorm(hidden)]`, its own attention (packed q+gate, 2 KV
  heads, per-head QK-norm, the stack's rope table), `post_attention_norm` as the
  PRE-FFN norm (there is no `ffn_norm` in block 40's tensor set), then its
  256-expert top-8 MoE through the SAME `ExpertCache` keyed on layer 40, plus
  the shared expert with Qwen3.5's `ffn_gate_inp_shexp` sigmoid. Then
  `shared_head_norm` and the target's own lm_head.
  Measured on Tiel (ctx 4096, greedy, 9-token prompt): one probe at the prompt's
  last row drafts `","` against the target's `","` -- MATCH, 90.8 ms including
  the replay. `KRK_MTP_PROBE_ALL=1` probes every decode step and **10 of 16
  drafted tokens match the target's argmax (62.5%)**; the block's KV is
  incremental (`mtp_fed_`), so a step costs one block forward (4-22 ms) rather
  than a replay. stdout md5 is unchanged with the probe on (`360279661e`), so
  the measurement does not perturb the run.
  Two facts that are correctness, not taste:
  - **The concat order is `[embedding | hidden]`.** `KRK_MTP_CONCAT=he` is the
    swapped arm and it measures **0/19** matches with visibly lower-confidence
    logits, against 10/16 for `eh`. The GGUF does not state the order, so this
    is the arm that decides it -- do not "fix" it from a comment.
  - **The router's `expert_used_count` is a MODEL-level key** (`qwen35moe.`), not
    `blk.40.`; and GGUF `ne[]` is `(in, out)`, so `attn_q`'s out is query+gate
    packed (8192) and `attn_k`'s is the KV width (512). Reading either as a head
    count produced `4096 head(s) x 2048, top-0`.
  **`--mtp-spec` drives that probe's loop as a real drafter, and on this model it
  is a measured LOSS -- it is opt-in and must stay that way.** A round proposes
  up to `--draft-tokens` tokens from the block's chain (position pos-1 uses the
  captured target row, later proposals feed the block's own previous output),
  verifies them with one batched target forward, keeps the longest matching
  prefix, and re-feeds the block from the accepted tokens' captured target rows.
  Because Tiel is a 30-of-40-layer GDN hybrid, a rejected suffix cannot be
  rewound: the round snapshots `rec_state_`/`conv_state_` (63 MiB, one
  `Backend::copy_device` on the default stream) and, on a partial accept,
  restores and replays the accepted prefix in ONE batched forward.
  Measured (ctx 4096, greedy, 32 tokens, 2 interleaved reps): plain **20.6-21.8
  tok/s**; `--draft-tokens 1` **11.0-10.8**, 60% of proposals kept, 1.60
  tokens/round; `2` **7.7-8.5**, 47%, 1.78; `4` **6.5-6.8**, 20%, 1.33.
  Re-measured after the dense-block work (ctx 4096, greedy, 64 tokens,
  `--draft-tokens 2`): Tiel **1.45 tokens/round** (44 rounds, 33% kept);
  `Qwen3.8-Distill-35B-A3B` **1.19** (54 rounds, 28%); `Qwen3.5-9B` (dense
  block) **1.78** (36 rounds, 54%). The loop is not the slow part: one round is
  one block forward per proposal (the block's own routed experts are cold every
  time, ~20 ms) plus the batched verify, so the propose pass costs what the
  saved steps cost.
  **Both block shapes load, and only one of them was wired at first.** A block
  whose FFN is dense (`blk.N.ffn_gate/up/down`, every qwen35 dense-family nextn
  head) has no routed banks: the loader found no experts and the probe answered
  `the block's expert geometry does not fit the MoE workspaces`. 13 of the 15
  files on this box are that shape. `MtpBlock::n_ff` + `ffn_gate/up/down` are
  loaded and `mtp_dense_ffn` runs them; both the loader line and the probe line
  now name the shape ("dense 12288-wide SwiGLU FFN" against "256 experts x
  top-8 of 512-wide FFN + shared expert"). 15 of the 19 nextn files here run the
  probe (two are head-only shards, and `DeepSeek-V4-Flash` is refused by arch
  before MTP). Per-model acceptance, 8 greedy steps x 2 prompts, is 50-100% on
  most, e.g. `Qwen3.8-27B-Q4_K_M` 8/8 and 6/8, `Qwen3.5-9B` 7/8 and 5/8.
  **The propose pass is 6% of a round, and pinning its experts is a measured
  wash -- do not re-derive it as a win.** `--mtp-spec` on Tiel (ctx 4096, 48
  tokens, `--draft-tokens 2`) spends 193 ms of a 4.6 s run on the block, 160 ms
  of that in `acquire` for 1040 acquires at an 11% miss rate; the round itself is
  the target's verify forward plus a replay of the accepted prefix. `ExpertCache::pin_layer`
  (KRK_MTP_PIN=1, wired from `Engine::mtp_pin_experts`) admits the block's whole
  routed set and pins it: propose 193 -> 136 ms, refeed 80 -> 33 ms, block
  acquires 160 -> 0 ms, text identical (`7ff494bbf10e`). But the 816 MiB comes
  out of the target's 7594 MiB budget, the target's promotions cost 105-180 ms
  more, and 5 interleaved order-balanced pairs measure **8.7 tok/s pinned against
  8.9 unpinned** -- so it is off by default and kept for callers whose block
  share is larger. A pin is only cheap where the budget is not the binding
  constraint; admit it during planning, not after the warmup has filled the
  card, if that ever changes.

  **A dense block must not memset an MoE workspace that was never allocated.**
  `ws_ffn_`/`ws_fexp_` exist only when `mc.is_moe`; `mtp_ffn` memset them, so on a
  dense model the first probe was `hipMemset(NULL)` -> `hipErrorInvalidValue`
  right after `workspaces ready`, before a token. The block's FFN now accumulates
  onto `ws_x_` (with `mtp_rep_` as the exponent) when there is no MoE buffer.
  Check any new "reuse the engine's workspace" code against `mc.is_moe`.
  Two separate reasons, and neither is a bug in the loop:
  1. **A 1-token window cannot amortize.** A round that keeps its proposal also
     has to forward the bonus token to extend KV, so it pays two target forwards
     for two tokens -- exactly plain decode -- plus one block forward
     (propose 20 ms/round on this model: the block's 2048 expert pairs are
     outside the ranked warm-up and are cold). On Tiel d=1 is the only arm
     whose text matches plain decode, precisely because its verification is one
     row; on a dense model d=2 does match (Qwen3.5-9B, 64 tokens,
     `fef693466284` both arms), so the limit is this hybrid's batched-forward
     arithmetic rather than the round.
     The k=2 Tiel text is not wrong: it equals `--chunk 1`'s hash
     (`6792671902ca`) where plain `--chunk 256` is `ac63e6812909`, i.e. the
     decoder tracks a multi-row forward exactly, and the divergence to explain
     is the one below.
  2. **d>1 changes the text, and the speculation code is not the cause.**
     Plain decode of the same prompt with `--chunk 1` and `--chunk 256` already
     produces two different hashes (`6792671902ca` vs `ac63e6812909`): a
     multi-row forward is not bit-identical to single-row decode steps (dense
     models are chunk-invariant -- SmolLM2-135M and Qwen3-MoE both hash the same
     at either chunk). `generate_speculative`'s "bit-identical" claim therefore
     holds only where the batched path agrees with the decode path.
     **The old localization here -- `KRK_DUMP` naming `moe.sum L02 rel=1.1e-3`
     -- was an artifact of `dump_diff.py` pairing line i with line i across
     different chunk sizes** (chunk 1 dumps row 0 first, chunk 256 dumps row
     255), which is why it pointed at the MoE and at a 1e-3 error. The
     pass-aligned comparison (`tools/dump_passes.py`) puts the first divergence
     at `attn.out` on the first full-attention layer, rel 3.8e-5, i.e. the
     attention kernel -- see "Chunk invariance is an ATTENTION property" below.
  So the win would need a bit-identical batched forward (a kernel-arithmetic
  project), not a better drafter. The probe remains the useful tool: it is
  exact, costs one block forward per step, and measured 62.5% first-token
  acceptance across 16 steps.

- **A published EAGLE-3/dFlash speculator is a HEAD, and `--draft-head` runs one
  with the verify loop that already existed.** `krk::DraftHead`
  (`include/krk/draft_head.hpp`, `src/draft_head.cpp`) loads a small GGUF
  (`tools/eagle_to_gguf.py` converts the published safetensors) and runs it on
  the HOST in f32 over f16 weights: fc(fused target hidden + last token
  embedding), one decoder block with its own rope attention and SwiGLU MLP, a
  final norm and a projection to the DRAFT vocabulary, with `d2t` back to target
  ids. `Engine::generate_speculative_head` is the round: propose up to
  `--draft-tokens` (proposal 0 from the TARGET's captured row, later ones from
  the head's OWN previous hidden plus the token it just proposed), verify in one
  batched target forward, keep the longest matching prefix, rewind.
  **Two things are correctness, not taste.** (1) The input at position p is
  (hidden_p, embed(token_p)) and it predicts p+1 -- the same contract
  `mtp_block_forward` has; feeding the target's row for a later proposal would
  read a row the target has not computed. (2) After a partial accept the head's
  KV is truncated to the round's start and re-fed from the target's CAPTURED
  rows; keeping its own speculative rows is drafting from a different
  distribution -- fluent, plausible, wrong, and invisible in the text.
  **Measured end to end** (qwen2-1.5B, ctx 512, greedy, 16 tokens, a
  randomly-initialised head so every proposal is rejected): stdout md5 identical
  to plain greedy (`f7957feb5610` both arms), `draft head: 16 round(s), 0 of 32
  proposals kept, 1.00 token(s)/round | prime 141 ms | propose 1442 ms`. A
  random head is the right first arm precisely because it cannot be accepted:
  it proves the loop cannot change the text.
  **`krk.draft.*` keys are required, not inferred**, and the loader refuses a
  head whose `hidden`/`target_hidden` disagrees with the model, whose `d2t` id
  is outside the target vocabulary, or whose weights are not F16. A recurrent
  target is refused at load for the same reason `load_draft` refuses one.
  **`d2t` is stored F32, not I32, because the GGUF reader cannot reach an
  integer payload**: `dtype_block_bytes` knows float and quant types only, so an
  I32 tensor reads as an unknown type with `n_bytes = 0` and a null `data`
  pointer. 4 bytes an id is 800 KiB at a 200k vocabulary -- the price of a
  readable map. Do not "restore" the I32 form without teaching `gguf.cpp` the
  integer types first.
  **`--draft-head-prime N` (default 256) is a real trade.** The prime is O(prompt
  x hidden x target_hidden) of host arithmetic, so a windowed prime attends only
  the most recent N positions (`drop_before` is what enforces it: a window that
  merely skipped the work would attend rows it never wrote). `0` is the exact
  whole-prompt form.
  **The suite runs the head itself** (`test_draft_head`, 50 checks): the
  per-tensor geometry, the target/vocabulary refusals, `to_target`, a prime arm
  against a fully-run arm (bit-identical rows), truncate+replay reproducing its
  own row, and a window that measurably differs from the full KV. All of it is
  host arithmetic, so no group opens the device to check it.

- **`KRK_TIME_HOST` is now the host table ON ITS OWN, and that switch is what makes a
  route-vs-expert comparison meaningful.** Both profilers used to hang off one
  switch, and the device recorder adds ~22 us per op -- so `moe.route` (which
  waits on the device) and `moe.experts` (which only enqueues) were compared under
  two different multipliers. Re-measured unprofiled on Mellum2.1 (28 layers, ctx
  512, 32 tokens, 41.0 tok/s = 23.7 ms/step): `moe.route` **0.827 ms/layer**, of
  which `moe.route.fetch` (the readback) **0.523**, `moe.route.rest` 0.307, and
  `moe.experts` 0.213 inside it. The router readback is **14.6 ms of a 23.7 ms
  token** and the router GEMM's enqueue is 0.1 ms.
- **That readback is a DRAIN, not a copy, and `KRK_ROUTER_2X=1` is the control that
  proves it.** The arm issues the same readback a second time, right after the
  first: the queue is retired by then, so it is pure WDDM round trip. Measured
  **0.077 ms against the first copy's 0.523**. So moving top-k and the plan to the
  device -- while the cache still needs the ids on the host to resolve an expert's
  pointer -- is worth those 0.077 ms/layer, **2.2 ms/token, ~+5%**, and the 14.6 ms
  wait stays. Do not size that rewrite from "the router is 60% of the step": the
  60% is the layer waiting for its own GPU work. The wait only goes away with a
  device-resident expert dispatch (device top-k, device plan, weights addressed
  through a device pointer table -- the `ggml_mul_mat_id` shape), which is a new
  kernel family and **cannot be bit-identical** because the softmax moves to the
  device. Expected ceiling is the 8.3 ms of GPU work the step already has (~120
  tok/s), not a dense model's rate.
- **A MoE decode step is host-bound at the ROUTER, and no cache policy can fix it.**
  `moe_ffn` computes the router row on device and then `download_f32`s it back to
  select top-k, so the blocking D2H orders behind every queued kernel: one
  pipeline drain per layer, 28 per token on a 28-layer model. Measured on
  Mellum2.1-12B-A2.5B (1,792 experts, 10.99 GiB routed, Q8_0): **28.6 tok/s decode
  with the per-process GPU engine sampler at 23.84% average / 72.17% peak**, i.e.
  35 ms per step of which 8.3 ms is compute -- and prefill 18.0 tok/s, of which
  1,812 ms of a 2,829 ms pass is promoting 1,184 experts (7.4 GiB read once). So
  "the card is slow" and "the cache is slow" are both wrong readings of this run:
  the card is idle. `KRK_ROUTER_SYNC=1` adds an explicit drain and is NOT the
  cause; the copy is. The fix is device-side top-k and a plan built there (what
  llama.cpp does, which is why it has no such ceiling) plus a grouped-expert
  kernel for the 54 launches a layer this currently issues. Those are the next two
  pieces of work, in that order.
- **The ledger now checks itself against the allocator, and "unaccounted" is not
  a synonym for "the driver has it".** `Backend::device_live_bytes()` counts what
  this process holds through `alloc`/`release`; the ledger prints it as
  `[allocator live 8454 MiB in 3744 buffers, largest 230]`. On the Mellum run that
  total equals weights + KV + experts + workspaces **to the megabyte** while the
  driver reports 7,000 MiB more consumed, and the 7,000 scales with the bytes
  *promoted* (7,430 MiB), not with the pool, not with WARM (`--expert-warm-mb 1024`
  and `11246` both give 7,000), and not with `HIP_VMEM_MANAGE_SUPPORT`. It is also
  not per-allocation rounding: `tools/alloc_probe.hip` allocates 3,465 x 1,972 KiB,
  fully written, and the driver charges 2,097,152 B each (**239 MiB** of waste, and
  a 6 MiB request is charged exactly). Before blaming a tier for a number the probe
  reports, run that probe and read `allocator live` -- the two together split
  "my accounting is wrong" from "the copy path holds it".
- **Device softmax for the MoE router is sub-ulp, and `KRK_DEVICE_SOFTMAX=1` is the
  arm that measures it.** `src/hip/kernels/router.hpp` + `Backend::router_softmax`
  compute the activation on the device (warp-packed max, `expf`, tree sum) and the
  arm *uses* those probabilities for the selection, so the text generated is the
  device path's, while the host answer is still diffed against it. Measured:
  **Mellum2.1 24 tok — 36,862/50,176 elements differ (73%), max |dP| 1.19e-07,
  text identical; 64 tok — 89,800/121,856, 1.19e-07, identical; Qwen3-30B-A3B 24
  tok — 61,553/73,728 (83%), 4.77e-07, identical md5**; and
  `KRK_DEVICE_SOFTMAX=1 bash scripts/coherence_check.sh <30B MoE>` still reports
  **1 coherent, 0 not** (the script inherits the environment, so the device arm
  really ran it).
  **Short runs lie about this, and that is the useful result.** At 24 and 64
  tokens both models print byte-identical text; at 512 they do not: Mellum's
  greedy run first differs at **char 235** (a token shift in a repetition loop,
  1267 vs 727 bytes), Qwen3-30B-A3B at **char 1464** ("capital of Ethiopia" ->
  "capital of Yemen"), and with `--temp 0.9 --seed 7` at **char 314** and **char
  111**. Max |dP| over those runs is 1.49e-07 / 4.77e-07. A fuller matrix (2
  models x 3 prompts x greedy 256 + sampled 128 at two seeds, 18 arms) puts the
  **earliest divergence at char 98** (Mellum, sampled, seed 13); 15 of 18 arms
  diverge and the three that do not simply had not forked yet at that length
  (Qwen3-30B p1 is identical at 256 and forks at 1,464 in the 512 run). So one
  ulp of probability changes a greedy token within ~100 characters, the effect is
  a time-to-fork rather than a per-arm probability, and 88 tokens of agreement
  was luck. Coherence still passes because it is a coherence gate, not
  a bit-identity gate. Consequence: device softmax is not a drop-in, and a
  dispatch that uses it accepts different text (or reproduces the host's `exp`
  bits, which a GPU cannot). Quote the stress table, never the 24-token arm.
  The arm engages only on the plain softmax path (`!router_sigmoid`, no
  pre-bias) and returns to the host otherwise.
- **`KRK_DEV_PTR=1` is the stage that removes the round trip: a device pointer
  table, and a whole MoE layer as one launch.** `ExpertCache::publish_dispatch_table`
  uploads `[n_layer * n_expert]` `MoeDispatchEntry` (`include/krk/moe_table.hpp`:
  three device pointers, shapes, types, row strides) and `moe_table_kernel`
  (`src/hip/kernels/moe_table.hpp`) reads the ids and weights from the device,
  looks each expert up in that table, and writes the weighted routed sum. No
  ids, no weights, no plan and no `acquire()` reach the host.
  **It is published only when the promise holds**: `whole_set_resident` must be
  true (every slot in VRAM), because a promotion or eviction rewrites a slot's
  pointers; `drop()` and promotion set `table_stale_` and the engine compares
  pointers every layer. The retry is bounded (8) because a cold cache never has
  the set at the end of init -- experts load lazily as routing reaches them.
  On this box no MoE corpus fits: Qwen3-30B-A3B with `--expert-warmup 1` reaches
  **5,952 of 6,144 slots** and prints exactly that, so the arm stays off and the
  text is unchanged. The kernel is therefore verified at the real geometry by
  `kraken-bench --moe-dispatch` (6 cases): **k=1 and k=2-same are bit-identical
  to the per-expert path, multi-expert is 1-2 f16 ulp, decode 4096x14336
  (45 KiB shared) identical**, and the run gates on a control that swaps the
  selected expert's gate/up in the table and requires the output to change.
  Three traps the arm produced: the down projection must be rounded to the
  activation type before the weighted add (else 1 f16 ulp on 14% of elements);
  `acc + alpha*v*scale` gets contracted into an FMA, so the kernel uses
  `__fmul_rn`/`__fadd_rn`; and a control that swaps two *identical* synthetic
  matrices (every `weights()` call restarts the same RNG) reports NOT PROVEN for
  a correct kernel -- salt each matrix.
- **The VRAM floor is enforced in `alloc_pooled`, and it is worth +87% as well as a
  usable desktop.** The expert budget is in requested bytes, so a pool can fill its
  plan and still leave the card below `vram_floor_bytes()` -- measured on Mellum2.1:
  a 7,248 MiB pool with **850 MiB free** beside a 3,261 MiB floor. A user sees that
  as paused video, not as a token rate. `alloc_pooled` now asks the driver before
  growing (only past half the budget, so a run that never nears the floor pays no
  query), trims reusable cells first, and refuses growth it cannot take -- the
  cache falls back to WARM, the same bytes from a slower tier. Interleaved, 3/3
  pairs, text byte-identical: **41.5 tok/s at a 3,261 MiB floor against 22.2 at 512
  MiB**, ending 3,238 MiB free instead of 850. Near-zero free VRAM makes the
  allocator the bottleneck, so the polite arm is the fast one -- the same inversion
  as the 30 ms-per-cell table above. Never size a tier in requested bytes alone.
- **`KRK_RECLAMP=0` and `KRK_VRAM_TRACE=1` are the arms for any budget cut.** The
  re-clamp fires on a live reading taken while the pool was growing, can only ever
  lower, and used to print "the KV tier has since taken" those bytes -- KV was 112
  MiB of a 4,000 MiB cut. A budget cut cannot be undone, so a wrong verdict costs
  the whole run; the off-arm is how a cut is shown to be wrong rather than guessed
  at. `KRK_VRAM_TRACE=1` prints the ledger before and after every multi-row
  forward and the trace line inside the clamp. Measured on Mellum2.1, turning the
  clamp OFF is **1.8-3.0 tok/s against 26.5-31.0**: it is the plan that is wrong,
  not the cut, which is why the floor moved into `alloc_pooled` rather than the
  clamp being relaxed.

## Weight loading (`Backend::set_weight_pull`)

- Large tensors are read() out of the model file rather than DMA'd from the
  mapping, because a cold mapping costs one soft fault per 4 KiB page (~0.4 µs
  each) and that — not the transfer — is the bottleneck.
- **`GgufTensor::data` already includes the data-section offset.** A pointer
  into the mapping converts to a file offset by `src - map_base` alone; adding
  `data_section_off()` again reads every tensor from the wrong place.
- **`hipStreamNonBlocking` streams have no implicit ordering against the legacy
  default stream** that every kernel in this engine runs on. Draining them is
  not sufficient for the caller to observe the data — a device barrier (or an
  event the default stream waits on) is required. The failure is silent and
  model-dependent: bytes are in flight, not missing.
- On Windows a file `HANDLE` serialises concurrent `ReadFile`, so each reader
  thread opens its own `FileReader`: 7.4 GB/s shared vs 15.9 separate.
- Every buffer in the pull ring must be drained. A drain loop left over from a
  smaller ring silently leaves the last chunks unwritten.

## Loading-mode benchmark (memory vs disk)

- **`scripts/bench_loading.sh` is the harness, and it measures three arms, not
  two.** `resident` (`--weight-vram-mb -1`, upload once), `auto` (the default
  budget: keep what fits, stream the tail, promote on first use) and `disk`
  (`--weight-vram-mb N` **with `KRK_WEIGHT_PROMOTE=0`**). The env pair is not
  optional: a streamed layer is promoted into a permanent allocation the first
  time it runs, so `--weight-vram-mb N` on its own is fully resident within the
  first token and reports a resident arm under a streamed label. `PATH:n` sets
  the per-model token count; `KRK_BENCH_REPS` defaults to 2 with the arm order
  reversed on even reps, and the summary takes the warmest rep per arm.
- **`--bench` ignores `-p` and runs its own fixed prompt.** That prompt
  tokenizes to **125 tokens on Phi's tokenizer and 55 on Qwen's**, so a Phi
  row's prefill tok/s is not comparable with the rest and a `-p` on a `--bench`
  line is dead text. `-n`, `--ctx`, `--chunk` and the sampling flags are read.
- **Measured (RX 9070 XT, 15.9 GiB, gfx1201), warm rep of two, `--ctx 512`,
  `--chunk 256`:** Phi-3.5-mini Q4_K_M (2.16 GiB, n=32) resident **128.4 tok/s**
  decode / 1861 prefill, auto 128.6 / 1868 (streams nothing), disk 2 GiB
  26.2 / 1198 with L29..31 streamed; Qwen3-8B Q4_K_M (4.68 GiB, n=32) resident
  **92.1** / 1037, auto 92.4 / 1045, disk 2 GiB 2.5 / 118 with L11..35;
  Qwen3-30B-A3B i1-Q2_K (0.71 GiB trunk, 128 experts top-8, n=16) 34.4 / 25.8
  resident, 34.4 / 26.0 auto, 34.9 / 26.0 disk -- the three agree because the
  ring never sees routed `_exps` banks; Qwen3.8-27B-WebGGUF Q4_0 (14.40 GiB,
  n=8) resident **14.9** / 292, auto 12.02 GiB budget 8.6 / 71.9 (L52..63
  streamed, then promoted), disk 2 GiB 0.5 / 28.8 (L1..63).
  So re-read cost is linear in streamed layers (3 layers = 4.9x slower decode
  on Phi, 25 layers = 36.8x on the 8B), prefill amortizes the re-read over the
  chunk (8.8x against decode's 36.8x on the 8B), and on a model that fits the
  `-1` arm beats the auto budget (14.9 against 8.6) because the load-time plan
  is decided before the KV cache and workspaces exist.
- **A cold page cache is not a slow arm.** The 27B's resident load reads
  **12.73 s on rep 1 against 3.23 s warm** (1.1 against 4.5 GB/s) and the
  30B's rep-1 resident run decodes at 20.8 tok/s against 34.4 -- so a
  sequential probe reports the file system as the policy. Interleave, take the
  warm rep, and compare decode rather than load: the 27B's disk arm loads in
  0.65 s because it only reads its 2 GiB resident prefix, and then decodes at
  0.5 tok/s.
- **The largest files on this box cannot be benchmarked at all, and that is the
  capability gap, not a loading result.** `kraken-inspect` refuses
  `DeepSeek-V4-Flash-...-IQ2XXS-...imatrix.gguf` (53.5 GiB, `deepseek4`: MLA
  `attn_kv_a`/`attn_kv_b`, `attn_sinks`, 3 `I32` tensors -- the "quantization
  type #26" line in the verdict), `gpt-oss-120b-Q8_0.gguf` (59.0 GiB:
  `attn_sinks`, arch not in the table; MXFP4 itself already works),
  `qwen3.8-flash-next-reap-288-Q4_K_M.gguf` (78.1 GiB, `qwen4exp`: seven
  classes) and `Muse-Glimmer-30B-UD-Q8_K_XL.gguf` (`muse-glimmer`, no entry).
  A refused file never reaches a decode step, so no loading arm can measure it.
  **Read the file's own header before writing code against a guessed shape:**
  `python3 tools/gguf_meta.py FILE` (`--grep KEY`) dumps every metadata key from
  the header alone, in milliseconds, including for a 78 GiB file. That is where
  the real numbers came from -- `deepseek4.attention.compress_ratios =
  [0,0,4,128,...]`, `key_length = value_length = 512` with `head_count_kv = 1`
  (one shared K/V vector, rope on its last 64, deroped on the output),
  `indexer.top_k = 512`, `hash_layer_count = 3`, `hyper_connection.count = 4`
  with 20 Sinkhorn iterations; and for `qwen4exp`, `full_attention_interval = 4`,
  `compress_ratios` 4 on the QSA layers, `indexer.top_k = 2048`,
  `hyper_connection.low_rank = 320`, `ple.layers = [1]` with 8 heads per 3-gram.
  A paper plus a blog post is a hypothesis; the header is the shape.
  `docs/AUDIT-supersized-gguf.md` has the per-file piece inventory and the order
  the work must land in: **`attn_sinks` first** (shared by `gpt-oss` and
  `deepseek4`, cheapest to verify: CPU oracle plus a zeroed-sink arm that must
  reproduce today's output bit for bit), then the `gpt-oss` table entry, then
  `deepseek4` MLA and its `I32` tensors, then `qwen4exp`. Note that for these
  MoE files the loading mode is *not* the lever: the ring carries the dense
  trunk only, so a 1.5 GiB trunk reads the same in all three arms (measured
  14.0 / 13.9 / 14.2 tok/s on `Qwen3.5-35B-A3B-UD-Q4_K_XL`); what makes them
  fast is the expert tier -- `--expert-scan` ranking, `--expert-warm-mb` and a
  VRAM floor.
- Numbers, commands and the reproduction loop: `docs/BENCH-LOADING.md`. The
  correctness rules for the ring itself are under "Dense weights stream now,
  and the ring has two traps" -- a drained ring before the next layer, routed
  banks excluded from a layer's byte total, and no double free of a promoted
  base in `unload()`.

## GGUF id 42 is two different formats

- Upstream llama.cpp's Q2_0 is a **64-value block of 18 bytes** (2.25 bpw):
  `#define QK2_0 64` and `block_q2_0` = `ggml_half d` + `uint8_t qs[QK2_0/4]`.
- The llama-dx fork's Q2_0 (`Maxritz/LLAMA-DX`, `ggml/src/ggml-common.h`) is a
  **128-value block of 34 bytes** (2.125 bpw), because that header has
  `#define QK2_0 128` and therefore a 32-byte code plane. The same header adds
  `GGML_TYPE_Q2_0_64 = 48` for the 64-value layout and states that the two are
  *not* byte-compatible.
- **The id alone does not say which one a file carries.** Kraken decodes 42 as
  the fork's 128/34 layout and exposes the 64/18 layout as `Q2_0_64` (id 48).
  Getting this backwards is not a crash: every block is read from the wrong
  place and the model decodes fluent garbage.
- **The file's own tensor offsets are the authority.** Sort the tensors by their
  own offset and compare `next_offset - offset` against the computed span; the
  ratio is the density in bpw. `python3 tools/gguf_scan.py --audit FILE` does
  exactly that and names the variant: 2.125 bpw is the fork layout (34/128),
  2.25 bpw is upstream's (18/64). Measured: `Ternary-Bonsai-27B-Q2_0.gguf` is
  498 tensors, every one exactly 2.125 bpw; the second shard of
  `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` is 37 tensors at 2.25 bpw.
- The ternary files use **codes 0/1/2 only**: code 3 is 0.00% of 32.6M codes
  over 254,976 blocks, so the `q == 3` branch that reads +2d is dead in practice
  even though the reference computes it. The three live codes carry nearly equal
  weight (35.1 / 29.8 / 35.1%), which is also why no byte statistic can tell one
  code-to-level assignment from another: that has to come from the tool that
  wrote the file, or from a run, never from the file.

## Coherence and correctness

- `scripts/coherence_check.sh` runs each model **twice** — on the GPU and again
  with `--cpu` — and diffs the text. A FAIL means device and scalar reference
  disagree; it does not mean the run was CPU-only.
- A greedy output that is fluent but wrong is the signature of a
  plausible-but-incorrect weight or attention path. Compare against
  `KRK_WEIGHT_PULL=0` / `KRK_ATTN_BEXACT=0` before assuming the model is at fault.
- `KRK_DUMP` + `tools/dump_diff.py` localizes a divergence to the first
  differing stage; that is the fastest way to tell prefill from decode.
- **Verify a data path against the real source pointer, not against the mapping
  at the same offset.** Comparing `read_at(off)` to `mapping[off]` passes for any
  wrong `off` — it is circular.

## Known open items

- **The KV cache is packed per KV-carrying layer, and the tier must cover it.**
  This replaces two earlier open items (the 8.67x padded charge and per-layer KV
  widths). `Engine` now builds a per-layer `len[]` from `Model::layer_has_kv(l)`
  and `Model::kv_dim_at(l)` and charges `sum(len) * 2` -- measured on
  `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf` (52 layers, KV heads
  `0 x46 / 2 x6`), `KV=24 MiB` at `--ctx 4096` where it printed and reserved
  `208 MiB` before. `KvTierCache` packs the same per-layer lengths, sizes a HOT
  slot by the **largest** layer (`slot_bytes_ = max(len_)`), and its flat
  allocation is the KV that exists -- so one width per layer is no longer a
  layout requirement, and a model with mixed KV widths loads.
  The invariant is a **correctness** one, and it is enforced by the layout
  rather than by the caller: HOT + WARM must hold one pass over the KV-carrying
  layers, and a layer in neither tier is zero-filled on its next step
  (`ensure_hot`'s `fill0` branch) with nothing recomputing it -- not a slow run,
  a WRONG one that still prints fluent text. **WARM's capacity is derived from
  the geometry, not from its budget.** With no `--kv-cold-dir` it is pinned at
  the KV-layer count, which is the number that makes an admission failure
  impossible: at any admission the layer being promoted is still counted in
  WARM and HOT holds at least the page being evicted, so `|WARM| <= layers - 1`
  and one more admission needs `layers`. A `--kv-warm-mb` below that is raised
  and the deviation is reported (asked MiB, applied slots, why, and the two ways
  out); the budget is obeyed only where a spill directory exists, and a
  `--kv-cold-dir` that cannot be written is ignored and reported rather than
  turning every eviction into a drop. The end-of-run line counts `DROPPED` pages
  apart from `->COLD` spills; they shared one counter before, which made a
  corrupt run report a working disk tier.
  `scripts/kv_tier_geometry_check.sh` is the proof, and it needs the machine's
  models to mean anything: three load arms per model (auto WARM; a budget with
  nowhere to spill, which must be raised and say so; the same budget with a cold
  dir, which must NOT be raised), each judged against the model file's own
  header by `tools/kv_tier_geometry.py`, plus a decode arm on every small model
  whose stdout must equal the flat cache's byte for byte. That checker has its
  own selftest (13 cases, four of them constructed failures) because a harness
  that cannot fail proves nothing. The sweep is bounded by `KRK_MAX_MB` (a model
  above it is reported `NOT COVERED`, not left out), `KRK_MIN_MB` (resume where a
  killed sweep stopped) and `KRK_ARMS` (`B` is one load per model instead of
  three, which is how the large ones get covered). A refused file is a `SKIP`: a
  refusal is a fact about the header, so `kraken-inspect` settles it for the cost
  of one process instead of three weight loads.
  Measured on this box: **49 models up to 8 GiB pass arm B, 0 failed**, and 18 of
  them carry KV on only some of their layers -- the per-layer geometry the tier
  has to respect -- while a full three-arm pass over every model <= 1.2 GiB also
  ends 0 failed with 7 decode arms byte-identical to the flat cache. Above 8 GiB
  the file is tens of GB and the KV tier is no longer the question being asked.
  Verified: SmolLM2-135M `--ctx 512 -n 32 --greedy`, 30 KV layers, 2 HOT slots
  -- flat, `--kv-hot-mb 1`, `--kv-warm-mb 64`, `--kv-warm-mb 0` (raised to 30
  WARM slots, reported) and `--kv-warm-mb 0 --kv-cold-dir DIR` (1916 pages
  spilled) all give stdout md5 `2456dfa5ba9e`.
  See `docs/AUDIT-residency-byte-budgets.md` (F2-F4, F6).
- **Two planes must not share a namespace.** `KvTierCache::cold_path` built a
  filename from the layer index alone, and both planes of the engine share one
  `--kv-cold-dir`, so K and V wrote the **same 30 files** and a restore handed
  one plane the other's bytes: 1916 `->COLD`, 1860 back, 0 dropped, and the text
  silently diverged from the flat cache (`a2e087cbeada` against `2456dfa5ba9e`).
  Fixed with a `plane_tag` ("k"/"v") in the name -- 60 files, md5 equal to flat.
  Same shape as the `hot / 2` split and the shared `kv_dim_` (F5), which are
  still fine only because MLA is `ArchSupport::No`; treat "two instances, one
  namespace or one number" as a defect pattern rather than a latent note. The
  per-plane warning duplication was the third instance: every KV warning printed
  twice until the engine took over printing it once.
  See `docs/AUDIT-residency-byte-budgets.md` (F5, F7).
- **The published `cf0408eb02` text hash does not reproduce -- use
  `a56e02b731fd`.** `docs/test-results.md` section 10 recorded the MoE
  acceptance run's text md5 as `cf0408eb02`; it matches neither the current build
  nor the archived pre-refactor `dist/kraken/kraken.exe` for that command, so it
  was measured under a protocol the doc did not state. The protocol that
  reproduces is **stdout only** (banner + generated text, stderr separated), and
  both builds give `a56e02b731fd`: `--model models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf
  -p "The history of computing is a history of abstraction" -n 64 --temp 0
  --greedy --ctx 512`. Two builds agreeing bit-for-bit is the useful fact: it is
  how you prove a layout change moved no numbers. Treat any stored hash without
  its exact protocol as a claim to re-run, not a fact.
- **Decode attention was nondeterministic at ≥128 keys and is now fixed.**
  `attention_decode_split_kernel` (`src/hip/kernels/attention.hpp`) had every
  warp-leader thread store the block's partial denominator to the same address,
  each seeded with its own warp's term, so the last writer won and one head's
  output depended on warp scheduling. This is a *second*, model-blind source of
  run-to-run variation, distinct from the qwen35 packed-query race, and it is why
  several "not reproducible" notes in this repo could not be reproduced. Measured
  on Qwen3-8B-Q4_K_M, prompt "The history of computing is a history of
  abstraction, from relays to vacuum tubes", `-n 128 --greedy --ctx 512 --chunk
  256`, md5 of stdout: the old kernel gave **three** distinct outputs in six runs
  of the identical command (`fc30f6c8cf1d`, `d8448e90a0a3`, `7ee2e7dec3aa`); the
  fixed kernel gives `7ee2e7dec3aa` 10 times out of 10 and now matches
  `KRK_ATTN_SPLIT=0` bit-for-bit. Below 128 keys `attention_decode_splits` keeps
  `n_split == 1`, so a short-context probe **cannot** see it — the md5 A/B and
  `--profile` are the tools that catch this class of bug, and a green
  short-context run is not evidence of determinism.
- **The hybrid CPU+GPU expert path exists and LOSES ~3.1× when it engages — do
  not turn it on by default.** `--hybrid-experts 1` (`CpuExpertPool`,
  `include/krk/expert_cpu.hpp` / `src/expert_cpu.cpp`, wired into
  `Engine::moe_ffn`) computes the experts the device tier cannot hold on the
  host, out of the weight mapping, while the device runs the ones it can. It is
  opt-in and default off, and it is **inert rather than harmful** when the
  budget holds the routed set: measured on Qwen3-MoE-4x0.6B-Q4_K_M at the auto
  610 MiB budget it took **0** experts on the host and ran 105.7 vs 104.4 tok/s
  without it. At `--expert-cache-mb 16` it took all 922 experts at 5.392 ms per
  expert on 8 threads and decode fell from 17.2 to **5.5 tok/s**. Generated text
  was identical in every arm, so the arithmetic is fine — the host arm is a
  **per-layer merge barrier** the device waits on, and ~2.0 ms/expert of
  read+promote beats 5.392 ms/expert of CPU. Do not re-derive the 1.84×
  "perfect overlap" ceiling in `docs/traces/cpu-expert-ceiling.txt`: it assumes
  the CPU work hides behind the device, which a per-layer barrier forbids. Its
  value is memory (a host-computed expert needs no VRAM slot and no WARM
  buffer), not speed — see docs/test-results.md §9. If it is ever revisited, the
  policy detail that matters is that the host arm must take only the OVERFLOW
  (`!in_vram(layer, e) && full_for(source)`): taking every miss livelocks the
  cache, because an expert that is never promoted never becomes resident.
- **Three group sites still test for one weight type, and llama.cpp's Q4_K_M
  recipe trips them per layer.** `FusedLayer::wt[8]` plus the nullable `wts[]` on
  `gemm_group` removed the kernel's need for a single `wt`, and `gdn_forward`
  uses it (18 layers of Qwen3.5-0.8B run Q5_K/Q4_K/Q4_K/Q4_K in **one** launch
  each). The sites that still require equality are `engine.cpp:918` (q/k/v),
  `engine.cpp:1154` (FFN gate/up, where the group also carries the silu) and the
  same pair at `dflash.cpp:799/835`. The test is **per layer**, which a file-level
  type histogram hides: the recipes bump `attn_v` on a non-contiguous subset
  (SmolLM2-Q4_K_M: 0,1,2,5,8,11,14,17,20,23,26,27,28,29 of 30), so those layers
  pay 3 launches instead of 1 -- **+28 launches in a 243-op step**, which the op
  table confirms by closing both counts exactly (46 `gemm_group` = 16 fused qkv +
  30 FFN; 73 `gemm(gemv)` = 14x3 split + 30 o_proj + 1).
  `tools/projection_type_audit.py` prints this per model from kraken-inspect's
  tensor table with no weight load (a file it cannot read is a `SKIP`, not a
  verdict); `docs/AUDIT-group-type-walls.md` has the ranking. Dropping the guards
  is mechanical and has no scale caveat -- per-matrix `.scale` sidecars exist only
  on the MoE expert/shexp tensors (`src/model.cpp:1073-1124`).
- **The MoE per-expert up/gate pair is grouped, and the fold is worth ~+7.8%.**
  Both are Q4_K in `Qwen3-MoE-4x0.6B-Q4_K_M`, and every expert visit ran
  `gemm(up), gemm(gate), silu_mul` -- 56 visits a step, 3 launches each where the
  FFN group's shape (2 matrices, one input, silu in the epilogue) needs 1.
  `Engine::moe_ffn` now calls `gemm_group(..., silu = true)` on the pair with
  per-matrix types and runs `silu_mul` only when the backend reports
  `silu_fused == false` (rows != 1, differing row bytes, split-K: prefill and
  anything the kernel cannot express -- which is why the return value alone is not
  the answer). `KRK_MOE_GROUP=0` restores the old path; `KRK_DUMP_MOE` forces it,
  because the probe's job is to read the up and gate results back as separate
  facts and the fused kernel writes only `silu(gate)*up`. Two conditions are
  correctness, not taste: the pair is grouped only when `gs_gt == gs_up` (with
  differing sidecar scales the old path rescaled `ws_xg_` BETWEEN the two gemms,
  which a one-input group cannot express), and the shared-expert pair at
  `engine.cpp:2861-2896` keeps the same caveat -- no model on this box exercises
  it. Measured: ops/step **817 -> 705**, `gemm_group` 14 -> 70, `gemm(gemv)`
  267 -> 155, `silu_mul` 56 -> 0, all 56 visits folded; A/B 9 interleaved pairs
  **order-balanced median +7.835%** (8/9 pairs, MAD 6.346%), stdout md5
  `a56e02b731fdf93004fd10b3a591df80` in both arms and equal to the acceptance
  hash this repo already records. **The first run of that experiment read +13.522%
  and that was the estimator, not the fold** -- see the harness bullet above.
  Separately, the router row is read back once per layer (28 `download_f32` per
  step); the explicit device sync there is now opt-in (`KRK_ROUTER_SYNC` restores
  it), so the copy is the only wait.
- **An external research master plan is audited against this tree in
  `docs/AUDIT-plan-vs-code.md`** — capability by capability, each row naming the
  file that settles it. Its phase list is NOT this repo's execution order: that is
  `docs/PERF-PLAN.md` (ranked B0-B6, each proven or open) plus `docs/TODO.md`'s
  P0/P1 and its closed negatives. Two things it claims are already otherwise here:
  BITCOS is decided in `docs/compressed-expert-spec-review.md` §3.3, and the
  "do not put RAM experts on the CPU" rule has its number in `docs/test-results.md`
  §9 (per-layer merge barrier, 3.1x). What it adds with no local equivalent is its
  §7 paper list — only BITCOS, Edge0 and RESOLVE are recorded here otherwise — and
  the expert-pack record format; what it lacks is any reference to this repo's own
  documents, and two of its steps cannot run here (RDNA2 `gfx1031` timings, and the
  per-model perplexity/accuracy gates of its §5, which do not exist).
- **A 256-expert MoE fits its tier, and what costs is promotion, not capacity.**
  `Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf` (40 layers, 256
  experts, 10240 slots at 0.937 MiB) is residented whole by the auto policy --
  9600 MiB = the entire corpus beside 5.24 GiB of dense weights -- so nothing is
  evicted and an expert pack would buy nothing here. What the default policy pays
  is the fill: at n=24 the `[stats] expert-path` split was `read 438.8 | room 0.1
  | alloc 92.2 | xfer 675.4` ms, i.e. **29.4 ms of a 60.6 ms step is H2D promotion
  and 4.0 ms is allocator wait -- 55% before a kernel runs**, at 275 us per ~1 MiB
  promotion (per-transfer overhead, not a bus limit). `--expert-warmup 1` on the
  same command reads `HOT hits 100% | WARM 0 | COLD 0`, all stage timers 0.0 ms,
  and **35.0 tok/s against 16.5 -- 2.1x**. The warm-up itself costs 16026 ms for
  9600 MiB (**0.6 GB/s**, 6% of the 15.9 GB/s the separate-handle reader measures),
  which is the phase a pack of one aligned read per layer would attack. Two traps
  worth knowing: the ranked warm-up **refuses** a `.krakenexperts.json` that does
  not record the file it was measured on (`source_size 0` here) and falls back to
  an even-order fill, and the reuse distance on this model is **~1/2/6 tokens**
  (p10/p50/p75 = 319/641/1919 acquires at 320 acquires per token), so a STAGE
  window two layers deep is the right length and a trained predictor is a
  cold-start tool only while the corpus fits VRAM. The hot core is real but
  narrow: **240 experts (6 per layer, 2.3% of slots, ~225 MiB) carry 42% of all
  acquires** while a per-layer LRU of K=8 reaches only **14.1%** and K=64
  **49.3%** (48 tokens, 26838 acquires, all 10240 pairs touched) -- so a small
  budget pins that core and streams the tail, and a big one just holds the
  corpus.
- **At a constrained expert budget the step IS the promotion, so a staged copy
  has nothing to hide behind -- and one expert of lookahead is worth nothing.**
  `Qwen3-MOE-4x0.6B-2.4B-Q4_K_M` (qwen3moe, 4 experts x top-2, 5.45 MiB per
  expert), prompt "The history of computing is a history of abstraction", ctx 512,
  n 32, `--greedy`: at `--expert-cache-mb 16` (2 slots) decode is **17.2 tok/s,
  58.1 ms/step, 1846 promotions of 10059 MiB in 1391.6 ms = 77% of the decode
  wall time**, at **0.0% HOT**; at 32 MiB (5 slots) it is 19.2 tok/s with 1.6%
  HOT and 74% promotion. Staging the next routed expert (`KRK_ACTIVE_STAGE=1`,
  off by default) copies on a nonblocking stream and publishes it with an event,
  and the probe (`KRK_STAGE_TIMING=1`, `[stats ] stage-timing`) measures the copy
  at **0.69-0.74 ms against a 0.32-0.34 ms reader window** -- 49-54% of 868 copies
  fully hidden, reproduced on two builds. The copy is twice the work it would
  hide behind and its bytes move either way; six interleaved order-balanced A/Bs
  (9 pairs each) found **no win at all** -- staging was faster in 0/9 and 0/9
  pairs at 16/32 MiB on the build before the probe landed and 2/9 and 1/9 after
  it, with NO DIFFERENCE at the resident budget where there is nothing to stage --
  and stdout md5 `28fa19c37` unchanged in every arm.
  Two traps worth keeping: an overlap probe must timestamp the *reader stream*
  at submit time (an event recorded after the copy only says the copy happened),
  and "the copy is on a side stream" is not evidence of overlap -- the window has
  to be measured, because here it does not exist.
- **The expert VRAM pool is physical and owned, and the target and draft share one
  limit.** `alloc_pooled`/`release_pooled` are individually owned size-class cells
  (256-byte aligned), not chunks of a fixed arena: every run prints
  `[stats ] expert-pool budget/committed/live/reusable bytes` with `committed =
  live + reusable`, a lowered budget trims free cells only, a live cell is never
  freed by a shrink, a reused cell waits on its release event (`guard-fallback`
  counts the times the event could not be created), and `set_expert_pool_owner`
  sums the target's and the draft's budgets so neither can claim the whole card.
  A single expert too large for the budget is still admissible -- explicitly, via
  `max(budget, one_expert)` -- because the engine cannot run without it. The
  device half is unreachable from `kraken-tests` (no group opens the device): it
  is pinned by `kraken-bench --gate`'s expert-pool gate, including a constructed
  event-creation failure (`KRK_TEST_POOL_NO_EVENT=1`), not by the unit suite.
- **A pool's reuse guard belongs on the device, and *asking* whether the reader
  retired is the whole cost.** At `--expert-cache-mb 16` on
  `Qwen3-MOE-4x0.6B-2.4B-Q4_K_M` the expert-path split read `alloc 481.8-618.4 ms`
  of a ~1.8 s decode, and the obvious reading (hipMalloc, or the free list) is
  wrong. `kraken-bench` now prints the pool's marginal primitives and closes it:
  event create+destroy **0.13 us**, record on the default stream **22.7-30.6 us**,
  query a *completed* event **0.03 us**, wait on a completed event **0.08 us** --
  but **record then query it while pending is +46.9 us over the record**, and the
  real alloc+release pair cost **86.97 us**. 5560 cells x 87 us = 484 ms, the row
  to the millisecond. **The record's own quiet figure is the one that is wrong:
  in-step it costs 2.56 us, not 22-31 -- see the audit bullet below.** On a budget
  that forces a promotion per expert visit the answer is always "not yet", so the
  query is a full host round trip per cell.
  The fix is to stop asking: the released cell carries the mark it was released
  under, `alloc_pooled` hands it out with that mark, and the next writer
  (`upload_paged_batch`, `upload_expert_staged`) issues `hipStreamWaitEvent` on
  the copy stream -- a wait on an already-complete event retires immediately, so
  the device cannot get slower. `alloc` **618.4 -> 182.5 ms**, decode
  **+11.279% at 16 MiB and +8.569% at 32 MiB** (9/9 interleaved order-balanced
  pairs each, md5 `28fa19c37` unchanged), resident budget NO DIFFERENCE (nothing
  is reused there, which is the null control). `KRK_POOL_HOST_WAIT=1` restores
  query-and-wait. **The "~128 ms of release records" this bullet used to leave
  as the next term was a quiet-queue estimate and is 10x too big; the next
  bullet has the measurement.**
- **A primitive measured on a quiet queue is not the in-step cost, and the
  factor is not a constant.** `kraken-bench` measures one call with an idle queue;
  a decode step issues 243-2965 of them with hundreds already queued, and the two
  figures differ in BOTH directions. Measured (2026-10-11, one session; full table
  and raw transcripts in `docs/AUDIT-quiet-vs-busy-primitives.md` and
  `docs/traces/quiet-vs-busy-primitives-2026-10-11.txt`):

  | primitive | quiet (bench) | host enqueue, busy queue | in-step (switch A/B) | factor |
  |---|---|---|---|---|
  | record on the default stream | 28.2-30.6 us -- **the DEVICE half** | 0.33 us, the same empty or busy | **2.56 us** | quiet overstates **11-12x** |
  | trivial launch (`copy_act n=32`) | 2.85-3.05 us | 0.92 us | 2.4-2.8 us per added call *with a pass* | ~1.1x with a pass, ~3.3x for an enqueue |
  | record + query a *pending* event | 24.7-46.9 us | -- | 46.9 us | **1.0x -- the calibrated one** |
  | a *device drain* (`hipDeviceSynchronize`) | not measured | -- | **~129 us** | the 0.07 us wait arm is 1000x off |
  | pool round trip (256 KiB cells) | 18.8-33.7 us | -- | 16.5 us | <=1.6x, either way |
  | `--floor` 1 MiB `copy_act` | 240.9 GB/s | -- | in-step `xfer` is H2D at 12.9 GB/s | wrong shelf: its ceiling is the reader's 15.9 GB/s |

  **The arm measures a call's two halves apart, and that is what the table above
  was actually comparing.** The queue is fed 512 x `copy_act n=32` of real device
  work and 64 calls are timed inside it (median of 8 blocks); `--depth N` sets the
  pre-issued count and `--depth 0` skips the printed arm. It cannot use
  `time_calls`: that helper subtracts one-call-plus-one-sync from
  many-calls-plus-one-sync, so its quotient is whichever half dominates the queue
  -- and for a record that is the DEVICE half. Measured 2026-10-11 (plain run,
  medians of 8, and the gate's own arms): a record enqueues in **0.33-0.34 us in
  BOTH queue states** and leaves **27-32 us of device time** behind it; a launch
  enqueues in 0.75-0.97 us and leaves 2.8 us. What the queue state changes is the
  **tail** (177 -> 583 us on the launch arm), never the enqueue. So the 28.2-30.6
  us this table used to call "quiet" was never a queue-state effect: it is the
  device half, which is the right figure only for a call that cannot overlap. The
  **in-step** column stands on its own switch A/B and is the number to rank with
  (2.56 us for the record, because in a step the markers overlap hundreds of
  queued launches); the "0.32 us busy enqueue" this table once called the
  alternative was the same 0.33 us as the empty arm, mislabelled as a state.
  **The arm is in the gate now and proves itself there** -- it asserts the queue
  really is fuller (launch-arm tail >2x, measured 2.9-3.7x), that a record's device half is >=4x
  its host half, that the enqueue does NOT move between states, and that the
  instrument resolves the device cost per op. `KRK_TEST_QUEUE_NOPREISSUE=1` skips
  the pre-issue and fails exactly the two fullness checks (rc=1), which is the
  red construction. The gate's floor line says `unresolvable at iters=2` instead
  of printing `-24.00 us per call` (the gate sets iters=2, where the subtraction
  is the difference of two single measurements).
  **Rules:** rank a fold by running the fold (one session, interleaved,
  order-balanced) and treat any constant-times-count estimate as good to ~2x;
  never compare a host-to-device row against `--floor`; and remember that a
  *drain* is 1000x a wait.
- **A small `upload` is a blocking copy, not a launch -- the per-expert plan pair
  cost ~15 us, not ~2 launches, and folding it was worth +10%.** `Engine::moe_ffn`
  used to upload each visit's row ids and gate weights with an `upload_i32` +
  `upload` pair; both are BLOCKING `hipMemcpy` calls (`src/hip/backend_hip.hip`),
  so 56 of them per step are worth ~10% of an 8.5 ms step, not the ~1.6% a
  2.4-2.8 us launch-fold estimate predicts. The plan for the whole layer is now
  bucketed once (one O(n*k) pass with a prefix sum, replacing an O(n_expert*n*k)
  rescan per layer) and uploaded with **two** calls per layer; a visit reads
  `[off(e), off(e+1))` of it and issues no copy at all. `ws_plan_`/`ws_alpha_` are
  `C * k` entries now (`C` is the chunk; the layer's pair count is bounded by
  n*k). `KRK_MOE_PLAN=0` restores the per-visit pair. Measured on
  Qwen3-MoE-4x0.6B (28 layers, top-2): ops/step **705 -> 649** (56 = 2 x
  (visits-1) per layer, the "on" arm closing the count exactly), A/B **+10.027%
  order-balanced, 5/5 pairs, MAD 2.870%**, stdout md5 `995c6b60e7` identical --
  and on Qwen3.8-Distill-35B-A3B (256 experts, 40 layers) `upload_i32` goes from
  one per expert visit to **one per layer** (320 over 8 tokens = 40/step, down
  from ~320/step). Ranking a blocking micro-copy as a launch is the same error
  class as the record row above.
- **The per-token drain in front of the topk readback was redundant, and removing
  it is worth ~4-5%.** `Engine::topk_argmax` issued an explicit `hipDeviceSynchronize`
  between its `logits_topk` launch and the 16-byte `download` -- but `download` is
  a blocking `hipMemcpy` on the same stream, so it already orders itself behind
  that launch. That is the same argument `Engine::moe_ffn` records for the router
  readback, and it is now the same switch pattern: `KRK_TOPK_SYNC=1` restores the
  drain (default off). Measured: SmolLM2-135M **+5.170%** and Qwen3.5-0.8B
  **+4.412%** decode (5/5 interleaved order-balanced pairs each, MAD 2.076% and
  0.631%, text identical). The remaining drains in the decode path now each have
  a switch too: the router readback (`KRK_ROUTER_SYNC=1`, ~129 us/layer),
  the expert request after a promotion (`KRK_SYNC_EXPERT=1`), the logits row
  readback on the fallback path (`KRK_LOGITS_SYNC=0`) and the speculative
  single-row recompute (`KRK_SPEC_ROW_SYNC=0`). The last two are not measurable
  on a GPU run today -- the fallback is unreachable while `logits_topk` works,
  and the speculative one needs a drafter (F8: self-draft acceptance 27.8%) --
  which is why they are switches and not rows.
- **An eviction's three slices share ONE release mark, and the row moved by a
  third -- which is still only ~0.5% of the step.** `Backend::release_pooled_batch`
  releases a burst under one mark: the releases in a burst are consecutive with
  nothing submitted to the default stream between them, so their positions are
  the same point, and the record is the whole cost of a mark. `HipBackend::PoolMark`
  is refcounted -- every cell released under it holds a reference, and so does a
  hand-out whose writer has not consumed the dependency -- and at zero it goes
  back to a free list with its event. `KRK_POOL_BURST=0` restores one record per
  slice, and the engine prints `release %llu mark(s) for %llu slice(s)`.
  Measured on Qwen3-MOE-4x0.6B at `--expert-cache-mb 16` and `32`, n=64, 9
  interleaved order-balanced pairs each: the `room` row (where the release
  happens, inside `make_vram_room`) **58.5 -> 39.7 ms at 16 MiB and 59.0 -> 40.6
  at 32 MiB, -31.23% and -31.28%, 9/9 pairs at both budgets**, with marks
  10740 -> 3580 for the same 10740
  slices, i.e. **1.00 -> 3.00 slices per mark**, and stdout md5 unchanged. The
  end-to-end decode verdict is **NO DIFFERENCE** (+1.287% and +2.133%, MAD
  1.6-2.1%): the row is ~18 ms of a ~3.2 s run, and no amount of sharing here
  can be worth more than ~0.5%.
  Two traps, both measured. **A record's cost is queue-dependent**: ~29-31 us on
  the quiet queue `kraken-bench` measures primitives on against ~2.5 us marginal
  in a decode step (7252 records removed = 18 ms), so the old "23 us x 5560 =
  128 ms" was a 10x overstatement -- the same quiet-vs-busy gap as the launch
  floor above, and it is why the whole row is ~13 ms and not ~128. **And the
  sharing must not go past a burst**: `paged_publish_` makes the DEFAULT stream
  wait on the copy stream (`hipStreamWaitEvent(0, paged_ev_[s])`) while the copy
  stream waits on these marks, so a mark re-recorded -- or recycled on a ring --
  while a wait on it is still in flight can land after such a wait and cycle
  with the writer that takes it. One mark per burst, held open while referenced.
  Opening a mark TAKES a reference (`refs = 1`) and `pool_mark_unref_` aborts on
  underflow: without that the same mark reaches two bursts and the destructor
  deletes it twice, which is STATUS_HEAP_CORRUPTION (0xC0000374) with the
  process's buffered stdout lost -- it reads as "prints its banner, exits 127",
  and the raw exit code (python subprocess -> 3221226356) is what names it.
  `kraken-bench --gate`'s three new `expert-pool` checks are the proof, and
  `KRK_POOL_BURST=0 kraken-bench --gate` fails exactly the sharing one (rc=1),
  on a gate whose other two burst checks still pass.
- **The slot lease has a device-side gate, and it is the bytes that prove it.**
  `kraken-bench --gate`'s `expert-lease` (18 checks) configures a budget that
  cannot hold the live expert and the staged one together -- strictly between
  two and three slots, because a plain two-slot budget has room for both and
  then nothing is reused, and exactly three lands on `over_budget`'s boundary --
  so a staged promotion provably *must* evict, then downloads the leased expert's
  three slices and compares them with the source it was promoted from, plus the
  staged expert's own bytes so an aliased promotion cannot pass by luck, plus a
  one-slot arm that must refuse to stage. It also asserts the mechanism, not just
  the data: the reused cells' guards were honored on the device. No group in
  `kraken-tests` opens the device, so this is the only place the invariant is
  checked against real device copies. **It took three versions, and the proof is
  the red run.** The first two passed with the lease DELETED from `pick_victim`:
  a budget with room for both experts (or exactly three slots, where the
  arithmetic lands on the boundary) never makes the promotion contend, so the
  byte checks pass because nothing is reused; and comparing against the pointer
  the first `acquire` returned is a tautology, because it points into the cache's
  own slot and reads the NEW pointers after re-promotion -- a lease violation IS
  a re-promotion, which restores the right bytes through the wrong addresses.
  The gate now snapshots the three device addresses, asserts non-vacuity
  (`the staged promotion had to evict a slot to fit`), and goes red (rc=1, a
  diagnostic naming `gate slice of expert 0`) when the lease is removed.
- **After the storage path, this model's decode is launch count.** With the
  warm-up on, one step is **2965 ops** for 40 layers: `gemm(gemv)` 531,
  `gemm_group` 360, and **320 each** of `upload_i32`, `upload`, `gather_rows`,
  `scatter_axpy` -- the per-expert machinery is **6 launches per expert visit,
  1920 of the 2965**, plus 40 router `download_f32` drains (one per layer). That
  is where the next folds are: the scatter into the down-projection epilogue, and
  the per-expert `upload_i32`+`upload` pair -- **which is now folded into one plan
  per layer (2026-10-11): `upload_i32` is 40/step here, one per layer, and the
  per-visit pair it replaced was two BLOCKING copies, not two launches, so the
  fold was worth +10.0% on the 28-layer MoE rather than the ~1.6% a launch
  estimate gives.** What is left of this bullet is the launch-count question
  itself: the **launch floor on this 2965-op step is still measured only by
  construction.** The A/B (`KRK_MOE_GROUP=0` against the default, which adds 2
  launches per expert visit) needs ~10 runs of a 13 s load + 16 s warm-up, and the
  **and on this box that configuration HANGS THE DISPLAY, twice, so do not run
  it.** The unprofiled baseline was obtained (ctx 256, `-n 8`, warm-up on): load
  34629 ms, **decode 12.2 tok/s = 81.7 ms/step**, `expert-path read 0.0 | room 0.0
  | alloc 64.8 | xfer 67.7 ms` over all 7 steps, HOT 96.4%, 0 COLD, 0 MiB read
  from the file -- so the "allocation-bound" reading this bullet used to carry is
  **falsified**: promotion is ~9 ms of an 81.7 ms step. What the run also printed
  is the thing that matters: **`[stats ] device 15.92 GiB in use of 15.92 GiB
  (0.00 GiB free)`**. `--expert-warmup` promoted `VRAM +9480 experts (8888 MiB,
  9480 slots)` in 33328 ms until the card was full to zero, and VRAM pressure that
  total is what took the machine down -- both times, with the run already bounded
  (`timeout 300`, `-n 8`) and the profiler OFF. **Rule: never enable the expert
  warm-up when the tier would fill the card; read the `[stats ] device` line, and
  keep headroom.** A 12.2 tok/s on a card with 0.00 GiB free is not a clean
  **The engine enforces it now.** `Engine::configure_expert_cache` holds back
  a VRAM reserve (`max(1 GiB, total/16)`, or `KRK_VRAM_RESERVE_MB`), clamps the
  *auto* budget to what is left of it, and **refuses** an explicit
  `--expert-cache-mb` that cannot fit, naming the largest budget the card can
  hold. The warm-up then re-checks that reserve every round against
  `device_free_bytes()` -- the plan-time arithmetic cannot see the workspaces
  and the KV tier, which are allocated after the budget is chosen -- and stops
  with `stopped at the VRAM reserve: N MiB kept free of M MiB total`. Measured
  on Qwen3-MoE-4x0.6B: a normal run does not trigger it (text byte-identical),
  `KRK_VRAM_RESERVE_MB=16000` promotes **0** experts and prints the report, and
  `--expert-cache-mb 20000` on this 15.9 GiB card exits **rc=1** with
  `does not fit: 16304 MiB of VRAM, 599 MiB already in use, 1024 MiB reserved
  -> the largest budget this card can hold is 14681 MiB`.
  **The host twin, the headroom line, and the test.** The same rule now covers
  host RAM: `krk::host_reserve_bytes` is `max(2 GiB, installed/8)` (or
  `KRK_RAM_RESERVE_MB`), and the WARM staging loop re-checks it every round
  against `host_available_bytes()`. The ram policy already plans a 20% headroom,
  but it plans it once, before the KV tier and the workspaces exist, and staging
  is what spends it. Forced with `KRK_RAM_RESERVE_MB=200000` on
  Qwen3-MoE-4x0.6B: `0 rounds, 4 experts/layer deep; WARM 0 MiB ... 0 MiB read,
  VRAM +0 experts; 0 ms (stopped at the host RAM reserve: 200000 MiB kept free)`;
  unforced, the same command completes (`2 rounds, WARM 610 MiB, 112 experts,
  649 ms`). Every run now prints its headroom in its first lines, dense models
  included -- `[info ] vram headroom: 15.57 GiB free of 15.92 GiB total, 1.00 GiB
  reserved` -- because the fault this guards against reached `0.00 GiB free` and
  the only line that showed it printed at the END of the run.
  The decision itself is `include/krk/vram_guard.hpp`: pure arithmetic, no
  device, no state, so `test_expert_cache_byte_budget` drives both halves of it
  (13 checks: the exact-fit boundary, the refusal naming the largest budget in
  MiB, a card with only the reserve left, and the backend that cannot report
  capacity). **The first version of those checks was written in bytes against a
  message formatted in MiB and the suite failed it** -- which is what they are
  for. **The refused startup is driven END TO END too.** The suite now builds an
  MoE fixture, tells the CPU backend to report a 4 GiB card (`KRK_VRAM_TOTAL_MB`),
  asks for a 4096 MiB budget against 3072 MiB of room, requires `Engine::init` to
  RETURN false with a refusal naming 3072, and then destroys the engine -- the
  unwinding that used to walk a freed backend. Its first check is a CONTROL: the
  same fixture at 1024 MiB must initialize, or a fixture that cannot init at all
  would make the refusal pass for the wrong reason.
  `scripts/vram_refusal_check.sh` is that path through the real CLIs, device-free,
  and it is the only proof of the `std::exit(1)` fix: put `delete be; return 1;`
  back in `main_cli.cpp` and it prints `CRASHED (rc=139)` and exits 1. Its other
  red construction is `KRK_CARD_MB=65536`, where the refusal arm correctly FAILs
  (rc=0, no message) while the generous-card arm still passes -- that pair is what
  stops an unconditional refusal from reading as a pass. It also takes the budget
  the refusal names and RUNS it: a refusal whose advice is not a working command is
  worse than a trim, because the user has no way back. 2944/2944.
  `KRK_VRAM_TOTAL_MB` is read by the CPU backend alone and only while it has no
  capacity of its own to report, so no GPU run can be affected by it; everything
  else in the run (the headroom line, the warm-up reserve, `[stats ] device`) then
  reports the simulated card, because that is the number every decision used.
  baseline either, so treat this figure as taken under the fault, not as the
  model's speed. The 35.0 tok/s this file records above is not reproduced by the
  run above (even-order fill -- the shipped 68-position index is refused for
  `source_size 0`, so that 35.0 came from the ranked one).
  Instruction-wise the kernels already use `wmma` (prefill), `sdot4`/dp4a and
  `sdot2` (the per-32-channel dot), `v_perm` and wave shuffles; **rows=1 cannot
  use a cross-row instruction**, which is why WMMA is prefill-only -- the lever is
  batching expert rows, not a different opcode for one row.
- `Spark_one.Q6_K.gguf` **passes** `coherence_check.sh` on the current build. It
  was recorded as failing at drift step 12; that no longer reproduces. Treat any
  "known to fail coherence" note as a claim to re-run, not a fact — the check is
  cheap and this one had gone stale.
- **An MTP head is an artifact, and it is extracted now — but the loop it would
  plug into is still a 4.0x loss (2026-10-09).** `tools/mtp_extract.py` repacks a
  checkpoint's `mtp.*` tensors into a GGUF with no requantisation and a hash check
  in and back out; the artifact (`C:\Strata-HIP-data\mtp\mtp-qwen4-mtp-bf16.gguf`,
  31 tensors, 4.86 GiB of BF16, arch `qwen4-mtp`) is an `ArchRole::Draft` entry in
  the arch table, so `kraken-inspect` names it and `--draft` refuses it with the
  reason rather than reading it against llama defaults. What it cannot do yet: the
  block is a `qwen4exp` block (gated attention, a QSA indexer, hyper-connections,
  a 512-expert MoE) and its input is the *target's* hidden state, so `qwen4exp`
  must run first — and before *any* drafter pays, F8 has to be fixed: a self-draft
  (the model against itself, where acceptance should be ~100%) is accepted
  **20/72 = 27.8%, exactly 5/18 in both runs**, 171.4/172.7 tok/s against
  669.1/707.0 plain, script-identical text. `docs/MTP-DRAFTER.md` has the artifact,
  the three-way verification and the probe (`--debug-topk` at `pos` and `pos-1`).
- **A DFlash head set names its target in its own bytes, and laguna's own drafter
  accepts 0.0% (measured 2026-10-09).** The fusion projection is sized
  `n_captured_layers × target_embd` (`src/dflash.cpp:364`), so `fc.weight` says
  which target a head set is for: `laguna-s-2.1-DFlash-Q4_K_M.gguf` is
  `3072x18432` over 6 layers (needs a **3072**-wide target, so it pairs with no XS
  file on this box) while `laguna-xs21-dflash-q8.gguf` is `2048x10240` over 5
  (needs **2048** = `laguna-xs2-Q4_K_M.gguf`, the pair that fits). laguna is not
  recurrent, so the rewind guard does not bite it. Measured interleaved and
  order-balanced (A B B A, `-n 16 --greedy --ctx 512 --chunk 256
  --expert-warmup 1`): plain **20.2 / 18.9** tok/s against dflash **2.1 / 2.1**,
  0/64 accepted, and at `-n 32` **0/128 = 0.0%** -- a 9.3x loss -- with stdout md5
  `0b3a8c42` identical in all four arms, which is both the proof that the reject
  path is correct and the measure of what the proposer is worth. Two causes, and
  the experiment that ranks them: a **self-draft accepts 11/32 = 34.4%** where it
  must be ~100% (F8 on a third family; the loop cannot accept even a correct
  proposer), and 0.0% is *below* that floor, so the proposer is wrong too -- while
  `KRK_DFLASH_LAYER_OFFSET=0` and `KRK_DFLASH_BLOCK_POS=1` each leave it at 0/32
  (they move the captures 2,14,26,34 -> 1,13,25,33 as documented and nothing
  else), and no feature clamp fires on this target. `docs/DFLASH-DRAFTER.md`.
- **`Laguna-XS-2.1-IQ3_XXS.gguf` is `runnable` and decodes all-NaN: the
  activation at L38/expert 161 is **785411** against f16's 65504 ceiling, so this
  is a *range* defect on the device path -- not the file, not the dequantizer and
  not an addressing bug (2026-10-09).**
  Every logits row is NaN (`--debug-topk`: `nan=100352` = the whole vocab at every
  step); `laguna-xs2-Q4_K_M.gguf` is `nan=0` with sane logits. **The single-row
  `KRK_DUMP` trace lies here** -- it names `attn.out L39`, 40 layers too late,
  exactly as the `dump_row` comment warns; `KRK_DUMP_FULL=1` moves it to
  `moe.sum L38 nf=2048 badrows=1` (one token row) and `KRK_DUMP_MOE=1` to
  `[moeout] L38 e=161 m=3 out_nf=2048 big=0`, with `in_nf`/`up_nf`/`gate_nf` all
  zero, so the gathered input and the up and gate legs are finite and only the
  down output is not. **Not the dequantizers** (`--cpu` on the same file is
  `nan=0` with `top2: 5899=16.2`), **not the device twin** (byte-for-byte the same
  indexing as the host `deq_iq3_xxs`), **not the file's bytes** (L38/L39 norms
  clean, expert 161's block scales `max|d|=0.0022` -> implied values <= 4.4, so no
  f16 overflow is even possible), **not the rows>1 path** (`--chunk 1` still NaNs)
  and **not the query-tiled kernel** (the then-current `KRK_ATTN_QTILE=0` switch
  still NaNs; that kernel has since been deleted), and **not
  the weight rows or the device's addressing of them**
  (`KRK_DEQUANT_PROBE=blk.38.ffn_down_exps.weight:161` reads that tensor's rows --
  512 wide = 2 IQ3_XXS blocks -- through the device and the host from the same
  bytes: all finite, worst relative difference 1.1e-4, computed span equal to the
  file's own `n_bytes`). **What is left is arithmetic, and one line names it:**
  `KRK_DUMP_MOE=1` on the device reads `[moeact] L38 e=161 m=1 act_nf=1` -- the
  down GEMM's input, after the silu -- and `--cpu` with the same probe and command
  reads `act_nf=0 act_peak=785411`: the model's own f32 value, **12x f16's
  65504**. An activation above the range becomes `inf` in an f16 buffer, the down
  GEMM's f16 output spreads it across the row (`out_nf=2048`), and the final norm
  closes with `inf - inf = NaN` over the whole vocabulary (`nan=100352`) and
  `〈|UNK|〉` text. The reference carries that value because **ggml's quantized
  matmul emits f32**, so 785411 never lands in a narrow type there; here the
  activation buffer is f16 by design (the bandwidth trade every kernel is written
  to), and that limit -- not the file -- is the defect. A **per-row
  power-of-two scale** is in the tree and does fix the two stages it is aimed at:
  `silu_mul` takes `(row_len, row_scale)` and `scatter_axpy_rows` a nullable
  `row_scale` (`include/krk/backend.hpp`, `silu_mul_scaled_kernel` in
  `src/hip/kernels/normalize.hpp`), with one f32 multiplier per routed row in
  `ws_scale_` and no host sync. Measured: `[moeact] L38 e=161` goes from
  `act_nf=1 act_peak=13232` to `act_nf=0 act_peak=49088` (= 785411/16 -- the row
  maximum was 11.99x the ceiling, and a power-of-two divide is exact in f16), and
  `[moeout] ... out_nf=2048` disappears. It is bit-identical in range:
  Qwen3-MoE-4x0.6B `995c6b60e7` and SmolLM2-135M `d6e2b2b3` stdout md5s are
  unchanged, `kraken-tests` 2858/2858, `kraken-bench --gate` rc=0, coherence
  check 3/3 -- **and this file still decodes 〈|UNK|〉**, because the overflow is in
  the model's own hidden state and bounding one store at a time only moves it.
  The MoE chain writes f16 activations in four places: the silu product (now
  row-scaled, with `d_sat_f16` as the fallback bound), the routed down projection
  (scaled down with its input, so it fits), the experts' sum
  (`scatter_axpy_rows_kernel`'s destination, now bounded -- `moe.sum L38` goes
  from `nf=356` to clean) and the residual add (`add_inplace`, `src/engine.cpp:3089`),
  which is still unbounded and is the first non-finite now:
  `moe.post L38 N=10 W=2048 nf=210 badrows=1 first=12 peak=65504`. So the value that
  has to be representable is the hidden state itself, ~65504 and above, at layer 38
  of this model -- and a bound is not the fix. **What landed is storage-only
  per-token residual scaling** (`Backend::add_residual`, `add_residual_kernel` and
  `rmsnorm_add_kernel` in `src/hip/kernels/normalize.hpp`, a nullable `row_exp` on
  `rmsnorm`, and one i32 per token row in `ws_rexp_`): the residual row is summed in
  f32, stored divided by the smallest power of two that keeps it inside f16, and the
  exponent rides the storage for every later reader. It is a *storage* property, not
  a factor that rides the arithmetic, and that is what makes it legal: every block
  reads the residual through a norm, the norm is scale-invariant, so the norm is
  where true scale comes back and every nonlinearity (silu, softmax) sees exactly
  the values it saw before. In range `rexp` stays 0 and the store is
  `f16(f32(a) + f32(b))`, which is `add_inplace_kernel`'s arithmetic, so it is
  bit-identical by construction and measured: Qwen3-MoE `995c6b60e7` and SmolLM2
  `d6e2b2b3` unchanged, `kraken-tests` 2858/2858, `kraken-bench --gate` rc=0,
  coherence 3/3.
  **The A/B is one switch, one process apart**: `KRK_RESIDUAL_SCALE=0` restores the
  pre-scaling path exactly, because an all-zero exponent array makes every other
  piece of the contract (the norms' un-scale, the fold's divisor) a no-op. Same
  binary, same command, Laguna-XS-2.1 `-n 16 --greedy --ctx 512`: OFF is `〈|UNK|〉`
  x16 with `silu 0, experts-sum 1020, residual 49152`, ON is " increasing complexity
  and abstraction. From the earliest mechanical calculators to modern quantum
  computers, each" with `silu 1, experts-sum 1020, residual 0`, and the ON arm
  repeated gives the same md5 (`dc9c47cfa1bd`) both times. `KRK_DUMP_FULL=1` now
  prints **no non-finite stage anywhere** on that file (it was `moe.post L38 nf=210
  peak=65504`), and at `-n 4` the generated text is byte-identical to the `--cpu`
  f32 reference (`3ed670b0f97670dbdb00ce9ec3157640`); at `-n 16` it tracks the
  reference for 13 tokens and then diverges, which is the still-clipped
  `experts-sum` store (1020 in both arms), not the residual. The sharded S-2.1
  decodes text for the first time too. Its `inf` into the residual is fixed by the
  same contract one store further out: `KRK_DUMP_MOE=1` reads
  `[shexp] L47 routedsum nf=0 peak=65504` with the shared expert's own output finite,
  so the poison was the **sum** of two in-range rows leaving f16 at the fold. That
  fold now renormalizes through `add_residual` with an exponent of its own
  (`ws_fexp_`), which the residual add takes as b's exponent. Same command one build
  apart (`-p "The history of computing" -n 1 --greedy --ctx 512`, dumps on):
  `residual 367` with `max exponent 60` (the clamp) before, `residual 0` with
  `max exponent 2` after. **What still clips is the routed down projection's own f16
  store** (`[moeout] L47 out_nf=16` of 3072 -- what `experts-sum 3810` counts), and
  that is the fused decode path's missing per-row scale: the down output inherits its
  input's row scale and the scatter already multiplies that same scale back through
  `alpha`, so the one missing piece is the row maximum on the fused gate/up pair,
  which is a cross-block reduction -- one extra small launch per expert visit, or f32
  activations for that chain.
  **What the reference does, read from the checkout** (`/h/LLAMA-Bins/llama.cpp`,
  the binary this repo's ROCm oracle uses): every activation in its MoE chain is
  f32 (`build_moe_ffn`, `src/llama-graph.cpp:2050-2300`; `ggml_mul_mat`'s result is
  `GGML_TYPE_F32`), and the activation a *quantized* matmul reads is block-scaled
  (`vec_dot_type` = Q8_0/Q8_1 for the K-quants and IQ types,
  `ggml/src/ggml-cpu/ggml-cpu.c:231-269`: 32 int8 codes plus one f16/f32 scale), so
  the magnitude rides a float scale and the error stays relative -- where this
  engine stores the activation itself in f16. `swiglu_clamp_exp` exists there as a
  per-layer expert-FFN clamp (`src/llama-hparams.h:377`) but the laguna files do
  not carry it (60 KV pairs, no `*clamp*` key), so there is no clamp to copy: the
  per-row power-of-two scale in this tree is this repo's version of a block scale.
  `d_sat_f16` (`src/hip/krk_hip.hpp`) remains a bound on the fused decode path,
  which has no row maximum to take.
  **Every bound counts now, and the count is printed.** The three bounded stores
  (`silu_mul`, the fused epilogue, the experts' sum) go through
  `d_sat_f16_count` (`src/hip/krk_hip.hpp`), plus a THIRD for the residual row,
  written by `add_residual_kernel` and by the folded residual epilogue.
  `HipBackend::activation_saturation` reads the device globals back,
  `residual_exponent_max()` reads the largest residual exponent, and every run --
  dense models, `--cpu` and `kraken-bench` included -- prints both
  `[stats ] range     f16 activation bounds fired: silu N, experts-sum M, residual K`
  and `[stats ] range     residual rows stored scaled: max exponent E`, with the
  reason spelled out when a counter is non-zero. A clipped run can no longer look
  like a clean one -- how these two files read as `runnable` for as long as they did
  -- and a run carrying a scaled residual now says so instead of leaving it to be
  inferred. Measured on `-n 16 --greedy --ctx 512` with `KRK_RESIDUAL_SCALE` off
  then on: `silu 0, experts-sum 1020, residual 49152` with `rexp 0`, against
  `silu 1, experts-sum 1020, residual 0` with `rexp 3`. Qwen3-MoE-4x0.6B,
  SmolLM2-135M and every CPU arm read `0, 0, 0` with `rexp 0` (and their stdout
  md5s are unchanged, which is what makes the counters inert in range). The
  `experts-sum` 1020 is the `scatter_axpy_rows_kernel` destination, and the fused
  decode path's product stores have no row maximum to take -- so the counters rank
  the remaining work without a probe, and they rank it here.
  **A per-token exponent on the residual stream is a STORAGE change, not a factor
  that travels -- that is the trap, and the shape above is what avoids it.** The
  exponent cannot simply be carried: with the hidden state scaled by 2^-e the gate
  becomes 2^-e*g and silu(2^-e*g) is not 2^-e*silu(g) (a large negative g flushes
  to 0 where the true value does not), so the FFN output is a different number
  rather than a scaled one; and q and k both carrying 2^-e put 2^-2e in the
  attention scores, which softmax reads as a temperature change. What is legal is
  narrower and simpler than "un-scale at every nonlinearity": every nonlinear block
  reads the residual through a NORM, the norm is scale-invariant, so the un-scale
  belongs to the norm alone -- which reads the whole row in one block anyway -- and
  no factor is ever carried through silu or softmax. Anything that scales a block's
  internals instead of the buffer's storage is a different model, not a
  differently-rounded one.
  **A second laguna file fails the same way, and its trace exonerated the shared
  expert: `Laguna-S-2.1-UD-Q4_K_M-0000[1-3]-of-00003.gguf`** (3 shards, 68.09 GiB
  payload, 256 experts x 4.5B active; the loader derives the siblings from any shard
  path and shard 1 is a 3 MiB header-only shard) loads all three shards, streams
  experts at 3.4 GB/s with 23.1% HOT hits and ~2082 MiB read per token, and decodes
  all `〈|UNK|〉`. Before the scatter bound its first non-finite was
  `moe.sum L46 N=10 W=3072 nf=1 badrows=1 first=1363 peak=3938`, with every routed
  leg probed clean (2470 activation rows, zero non-finite, peak 29648) and a probe
  at the routed sum *before* the shared expert folded in reading
  `[shexp] L46 routedsum nf=1 peak=3930` -- so the routed experts alone did it, and
  the shared leg is exonerated. That bound moved the poison one stage, exactly as it
  moved the XS one: `moe.post L46 N=6 W=3072 nf=1 first=1363 peak=4224` -- the
  **residual add**, at the same row and the same column. Both laguna files stop at
  the same store now, and it is the hidden state rather than an intermediate, which
  is why the remaining fix is range and not another bound.
  Commands and the exclusion table: `docs/DIAG-iq3xxs-nan.md`. The engine already
  cites this file at `src/engine.cpp:1252`, for *residency*, which is how the NaN
  survived: it was measured for what it costs, never for what it says.
- **The device's block-size table defaults to `0` where the host's defaults to
  `32`, and that is the one shape where a device row read can disagree with the
  host without saying anything (audited 2026-10-09).** `dtype_block_bytes_dev`
  (`src/hip/kernels/dequant.hpp:213`) is a hand-written switch over `DType` with
  `default: return 0`, while `dtype_block_size` (`src/quant.cpp:1021`) ends in
  `default: return 32`. Five device sites re-derive a quantized row's layout
  through the device table (`attention.hpp:56`, `dequant.hpp:993` and `:1088`,
  `gdn.hpp:247`, `gemm.hpp:476`), while every host caller asks
  `dtype_row_bytes`, so there is one table per side and the two are maintained by
  hand. With `block_bytes == 0` the addressing `wrow + (c / kChunks) * block_bytes`
  collapses and **every** block of the row reads block 0: no crash, no NaN, just
  the wrong weights -- the same failure shape as the null-weight `Unknown` dtype
  the MoE probe documents (`src/engine.cpp:2870`). The two tables agree on every
  type that exists today, so this is a latent hazard rather than a live defect:
  the fix is either a device default that refuses (or asserts) instead of
  answering 0, or one shared table both sides read.
- **Speculative decoding with a generic `--draft` was a 2.7× net loss** (32.4
  tok/s against 88.4 target-only, Qwen3-8B target + Qwen3-MoE-4x0.6B draft,
  44.8% accept) and the cause was a call-site regression, not the idea. The
  draft loop and the target's verify block both pulled the **whole 993 kB logits
  row per proposal** just to take an argmax on the host, where plain greedy
  decode had already been switched to the device top-k (`fetch_logits(true)` /
  `topk_argmax`, 16 bytes back). Both now use it. The prose above `fetch_logits`
  asserted that "the speculative and draft paths pass false and get the whole
  row exactly as before" — that was true, and it was the bug.
- The prefill default is `attention_bexact_kernel` (decode's arithmetic over a
  tile of query rows), and the query-tiled kernel it replaced is deleted --
  along with `KRK_ATTN_QTILE`/`_QT`/`_K`, `attn_qtile_on()` and
  `tools/probe_attn_qtile.hip`. The tiled `attention_kernel` survives only as
  the fallback for a windowed layer, a non-causal call and a head width outside
  64..256 in steps of 32. The query-tiled kernel's own history is worth keeping
  here because it is the reason the tiled form was believed correct at all: its
  first version had a shared-memory race (one f32 score row per query
  overwritten with exp(x) before neighbours had read it), fixed to 0 bad
  elements and a worst residual of 1.2e-4 from fp16 summation order, and it was
  md5-identical to the one-query path at 801- and 900-token prompts for +38%
  prefill (581.5 against 420.9, 5 interleaved reps). The exact kernel then beat
  it by 49% at hd 96 and 33% at 3400 tokens on Qwen3.5-9B, which is what made
  deleting it the right call rather than a simplification.
- `maclin` (gfx1031 / RDNA2) is unreachable from here, so nothing is *executed*
  on RDNA2. The target still builds: `-DKRK_GPU_TARGETS="gfx1031;gfx1201"`
  compiles the gfx1031 kernels and the same `kraken-bench --gate` runs over
  them. Missing hardware, not missing verification.
- **The `*-Q2_0.gguf` ternary Bonsai files are not malformed** -- see "GGUF id
  42 is two different formats" above. Their own tensor offset table implies 17
  bytes per 64 values because the fork that wrote them uses a 128-value block
  (`#define QK2_0 128`, 34 bytes) where upstream uses 64 (18 bytes); that is also
  why upstream llama.cpp refuses `Ternary-Bonsai-27B-Q2_0.gguf` with `tensor
  'output_norm.weight' has offset 337715200, expected 357580800` -- the number
  this engine used to compute, before 42 became the 128-value layout the file
  actually carries. The files that carry 42 are
  `Ternary-Bonsai-27B-Q2_0.gguf` (498 tensors), the second shard of
  `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (37) and
  `Ternary-Bonsai-27B-dspark-Q4_1.gguf` (one token_embd row). **The diagnostic
  worth keeping:** read the file's own tensor infos, sort them by offset, and
  compare `next_offset - offset` against the computed span of each tensor. That
  pinpoints a wrong block size in one run -- it is how the 17-vs-18 discrepancy
  was found -- where comparing a decode against the file's mapping cannot see it,
  because only the file's own offsets are authoritative.
- **`n_embd % n_head != 0` is not a defect when the file declares
  `attention.key_length`.** `Model::load` enforces that identity only when
  head_dim is *inferred* from the embedding (`head_dim_declared`,
  `src/model.cpp`). Qwen3.5's gated attention runs 24 heads of 256 dims over a
  5120-wide embedding because `q_proj` is `2 * n_head * head_dim` = 12288 and
  emits the query gate alongside the query, so the identity genuinely does not
  hold. Before this was fixed both `Bonsai-27B-Q1_0.gguf` and
  `Ternary-Bonsai-27B-Q2_0.gguf` were refused at load with `n_embd is not
  divisible by head count` while their dtype was already fine. Same shape as the
  per-layer-head-count exemption laguna already needed in that function.

## tilelang (G:/tilelang-rocm)

- **A failed `import tilelang` here is an interpreter problem, not a package
  problem.** `tilelang` is the only entry point that works: it bootstraps
  `sys.path` for `tvm` and `tvm_ffi`, so `python -c "import tvm"` on its own is
  `ModuleNotFoundError` by design, and `ModuleNotFoundError: No module named
  'tvm_ffi'` on `import tilelang` means the interpreter cannot see the usersite
  that has it. `python3` on this box is **3.14** (pythoncore-3.14-64, no
  `tvm_ffi`) while the dev build's extension modules are **cp312 only**
  (`build/lib/*.cp312-win_amd64.pyd`), so no 3.14 interpreter can load it even
  with `tvm_ffi` installed. `python` and `py -3.12` are 3.12.10 (Program Files)
  with `apache-tvm-ffi 0.1.12`; those work. `tools/tilelang_hipcc_repro.py`
  checks the pyd tag against the running interpreter and re-execs itself under one
  that fits (`KRK_PYTHON` overrides the search).
- **A kernel source needs the HIP headers, or the errors say nothing.** Without
  `<hip/hip_runtime.h>` and `<hip/hip_fp16.h>`, clang ignores `__global__` as an
  unknown attribute and reports `unknown type name '__half'`, undeclared
  `threadIdx`/`blockIdx` -- five errors that never mention the missing include.
  `src/hip/krk_hip.hpp` includes exactly this pair; a probe or a swapped kernel
  body must carry the same prelude.
- **`--genco` writes a clang offload bundle, not a loadable code object.**
  `hipcc --genco` and `tilelang.contrib.hipcc.compile_hip` both emit
  `__CLANG_OFFLOAD_BUNDLE__` (the prefix is 24 bytes -- do not read 20 and
  conclude it is not one), holding a `hipv4-amdgcn-amd-amdhsa--<arch>` device
  payload plus a host stub. A HIP loader cannot take that file, so unbundle first:
  `<rocm>/lib/llvm/bin/clang-offload-bundler --unbundle --type=o
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --input=x.hsaco --output=x.o`.
  Then check the result: `llvm-readelf -h` shows ELF64 DYN, OS/ABI `AMDGPU - HSA`,
  Machine `EM_AMDGPU`, the kernel symbol is a GLOBAL FUNC, and `.note` carries
  `NT_AMDGPU_METADATA` with `amdhsa.kernels`. Measured: 10,328 bytes of bundle ->
  6,232 bytes of ELF, kernel `kraken_probe_gemv_q4`.
- **The laguna MoE chain's last unbounded store was the experts' SUM, not the
  fused silu -- and the text divergence is rounding, not range (measured
  2026-10-09).** Three facts, each from the counters and the CPU reference rather
  than from reading the code: (1) the fused decode silu now runs through the
  row-scaled `silu_mul` the gate/up group's fallback already had (`silu 1 -> 0` on
  Laguna-XS-2.1; `KRK_MOE_SILU_FUSED=1` restores the fused form for the A/B,
  because that kernel is one warp per output element and has no row maximum to
  take); (2) `scatter_axpy_rows` now takes a nullable `i32 *row_exp` and stores
  the routed sum divided by the smallest power of two that keeps the row inside
  f16, with the exponent handed to the folds that already read `ws_fexp_`
  (`experts-sum 1020 -> 0` on XS-2.1, `3810 -> 62` on S-2.1; at exponent 0 the
  store is the same expression it was, so in-range models are bit-identical);
  (3) the clips were never what moved the text -- `-n 4`, where the doc records
  byte-identical text, already reads `experts-sum 1020`, and every `-n` to 16
  reads the same 1020, i.e. all prefill and none in decode, yet XS-2.1 with every
  counter at 0 still diverges from its f32 reference at the same token (`modern`
  against the reference's `today's`). What remains is f16 rounding in a greedy
  near-tie: an f16 pipeline does not round like the reference's f32 one, and no
  storage scale changes that. Fixing this exposed a real defect worth carrying:
  both exponent pickers (`add_residual_kernel` and the folded `rmsnorm_add_kernel`)
  started from `rexp[r]` -- the exponent the row arrived under, already applied by
  `sc` to read it at true scale -- and then halved `m` until it passed the
  ceiling, so every store of an already-scaled row re-raised it: `max exponent 30`
  for a row needing 5, and S-2.1's `max exponent 60` (the clamp) sitting beside
  `residual 0`, where at 2^-60 the row's elements underflow f16 entirely. From
  zero XS reads 5, which is what that file's own 785411 activation peak implies.
  S-2.1 still reads 60 on a row where an `inf` reaches the store (`inf * 2^-e` is
  `inf`, so no exponent helps it).
- **Both reference engines on this box put the scale on the activation, not on a
  narrower activation type.** `/h/LLAMA-bins/llama.cpp` keeps the MoE chain in f32
  (`ggml_mul_mat` emits `GGML_TYPE_F32`) and applies `swiglu_clamp_exp` only when
  the file sets it (laguna's do not); `/h/LLAMA-bins/ik_llama.cpp` takes the
  block's own maximum for its activation quantiser (`d = amax / 127` over 32
  values, `ggml/src/iqk/iqk_quantize.cpp`) and still runs the nonlinearity in f32
  (`MulMat::silu(int n, const float*, float*)`). This repo's per-row
  power-of-two scale is the same idea coarser -- one scale per row instead of one
  per 32 values, exact because a power of two spends no mantissa -- and it is what
  an f16-activation engine can do without changing its activation format. That
  fork also carries a real DFlash implementation to read before the next attempt
  at the 0/128 acceptance in `docs/DFLASH-DRAFTER.md`
  (`src/llama-dflash.cpp`, 763 lines, `LLM_ARCH_DFLASH*`, `dflash.target_layer_ids`).
- **The activation-range readout is an invariant now, not a report (2026-10-09).**
  `src/main_cli.cpp` gains `range_violated` / `range_verdict` / `range_exit_code`:
  a run whose counters show a store that left f16 and was clipped, or a storage
  exponent that reached the clamp, prints `[error ] range generate: ...` and exits
  **2** instead of exiting 0 with the fact buried in the log. `KRK_RANGE_STRICT=0`
  demotes it to `[warn ]` for the probes that read a clipped model on purpose.
  The clamp half is not decoration: it is the only signal that fires when nothing
  was clipped but the scale itself ran away, which is exactly the ratchet defect
  this session fixed (XS-2.1 read `max exponent 30`, S-2.1 `60` beside
  `residual 0`). Verified both ways: Qwen3-MoE-4x0.6B and SmolLM2-135M exit 0 with
  `0, 0, 0` and exponent 0; `kraken-bench --gate` rc=0;
  `scripts/coherence_check.sh` 3 coherent, 0 not, rc=0; Laguna-S-2.1 exits **2**
  with `1 silu, 62 experts-sum ... and the storage exponent reached the clamp`, and
  the same run with `KRK_RANGE_STRICT=0` exits 0 with the same text at `[warn ]`.
  A gate that cannot fail proves nothing, so the two arms are the proof.

## gpt-oss (openai-moe) and the two rules it taught

- **A 120B file runs on this box in one second as an oracle**, which is cheaper
  than any amount of reasoning about a wrong answer. llama.cpp mmaps and faults
  in only the experts a token touches: `gpt-oss-120b-Q8_0.gguf` (59.0 GiB)
  loaded in **1.0 s** and decoded at **7.6 tok/s** on 16 CPU threads, and
  `--verbose-prompt` prints the prompt's token ids so the tokenizer can be
  cleared in the same run. Measured baseline: `The capital of France is` ->
  ` Paris.` The command is in `docs/AUDIT-gpt-oss.md`.
- **`expert_weights_norm` defaults to FALSE in llama.cpp and its openai-moe
  loader never sets it.** A 128-expert stack whose top-4 hold **13%** of the
  softmax mass is mixed with the raw probabilities, so an engine that
  renormalizes multiplies every layer's expert contribution by **7.6**. The
  symptom is not a crash: the residual stream grows ~1.2x per layer (measured
  rms 2.0 at L00 to 1323 at L35) and the model answers fluent nonsense. The
  arch now declares the default (`ArchSpec::expert_w_norm`), and gpt-oss's
  expert activation is `ggml_swiglu_oai` -- `min(gate,7) * sigmoid(1.702*that) *
  (clamp(up,-7,7) + 1)` -- not `silu(gate)*up`. Both were wrong at once, and
  either alone is enough to produce garbage that looks like a broken kernel.
- **A windowed layer DOES NOT WRITE `attn.out` on a multi-row prefill today**,
  and that is open. The tell is cheap: `KRK_DUMP` row 4 of `attn.out` on an even
  layer holds 64 non-zero values out of 4096 and the *same bytes* for two
  different prompts, while the odd (full-attention, exact-prefill) layers are
  fresh. gpt-oss is the only file here that combines `attn_sinks` with windowed
  layers, and `KRK_SWA_OFF=1` is the new A/B arm that runs every layer as full
  attention -- the same computation below the window length -- and makes those
  layers fresh again. Use it as the workaround for prompts under 128 tokens and
  as the arm that proves the rest of the model math.
- **`docs/AUDIT-gpt-oss.md` has the per-defect measurements**, including the
  layer-0 MoE matching an independent numpy reference to rel-l2 **5e-4** and
  position-0 attention to **2e-4**, plus the trap that cost an hour: the YaRN
  ramp is indexed by the **pair** index, not the channel index (`rope_yarn_ramp`
  divides `i0` by 2), so a reference that uses `2i` reports the engine's correct
  table as wrong.
- **The expert cache has a hot ring, and the measurement says it is a prefill
  win, not a decode one.** `ExpertCache::set_pin_budget` + `admit_hot`, driven by
  `Engine::pin_hot_experts` between the warmup and the pool clamp: pinned bytes
  are charged to `pin_budget_` instead of the LRU pool's `budget_`, so the pins
  stop shrinking the window traffic still earns, and the backend's owner budget
  is `budget_ + pin_budget_` (`owner_bytes()`; handing it only `budget_` makes
  the pool refuse the ring's own bytes -- measured `live=5583 MB budget=3396 MB`
  and the run died on its first token). With no ring configured the old
  arithmetic is exact, which `test_expert_cache_policy` case 4 still requires
  and cases 10/11 cover the ring.
  Measured on Tiel-Coder-35B-A3B-MTP-APEX (7682 MiB of device expert memory,
  `--bench --ctx 512 -n 64`, interleaved order-balanced, identical text
  `3baee4232881`): a 4350 MiB ring is **+5.5% prefill and -7.8% decode (5/5)**,
  a 2174 MiB ring is **+10.8% prefill and -2.1% decode (4/5)**, and both arms
  report the same `HOT hits 66.4% | WARM hits 33.6%` -- the LRU already finds the
  hot core at this shape, so the ring removes churn and does not remove the
  promotions. That is why the default is 1/6 of `total - vram_floor`, why
  `KRK_EXPERT_PIN_MB` exists, and why `KRK_EXPERT_PIN=0` is the A/B arm. The
  ring is skipped without a `<model>.krakenexperts.json`: an even order is not a
  hot set and a wrong pin cannot be evicted.
  Two traps, both paid for: sizing from "free beside the floor" is a coin flip,
  because the warmup has usually filled VRAM to its reserve by then (the same
  command admitted 5325 MiB on one run and skipped the ring on the next,
  depending on whether the warmup finished its VRAM stage or hit its
  `--expert-warmup-ms` ceiling) -- size it from `total - floor` and let
  `admit_hot` evict, since those slots are in WARM and the move is a promotion,
  not a read; and `scripts/ab_pin_ring.sh` is the harness (`RING_MB=<n>` sizes
  arm B, arm A is `KRK_EXPERT_PIN=0`), printing both rates per run rather than
  one number.
- **The expert cache's miss rate is the wrong objective near the card's limit,
  and that is measured, not argued.** Tiel-Coder-35B-A3B-MTP-APEX, greedy,
  `--bench --ctx 512 -n 64`, interleaved: with the default `KRK_VRAM_FLOOR_MB`
  (a fifth of the card, 3261 MiB here) decode is **22.1 / 23.7 / 22.9 tok/s at
  84.9% HOT hits and 3620 promotions**; at floor 2048 it is **13.7 / 16.3 / 17.2
  at 87.1% and 3087**; at floor 1024, **11.3 / 13.2 / 13.0 at 88.3% and 2787**.
  A *better* hit rate with *fewer* promotions is 2x slower, because the pool then
  has no headroom and every activation/workspace allocation pays the driver's
  eviction cost -- and the expert-path timers cannot see it
  (`room 291 | alloc 207 | xfer 324 | promote 823 ms` is **lower** at floor 1024
  than at 3261). Raising the floor loses too (4096: 20.2 tok/s, 4800: 19.7), so
  3261 is the top of the curve. Do not "optimise" the hit rate on a routed set
  that does not fit without checking this table first.
- **`ExpertCache::reclamp_vram_budget(free, floor)` re-derives the pool budget
  from a live reading** (unpinned resident + free beside the floor, only ever
  lower; `Engine::reclamp_expert_budget` runs it once per prefill and every 16
  decode rows). The init clamp is one measurement taken before the KV tier starts
  promoting COLD->HOT. The arithmetic lives on the cache so
  `test_expert_cache_policy` case 12 can drive it with no device (96 -> 48 bytes,
  a reading with room does not raise it, the next expert evicts instead of
  growing). On this box at `--ctx 512` it is a no-op because the KV cache is flat
  resident -- correct, and the guard is for the run where it is not.
- **Staged promotion and top-k' lookahead exist and are OFF by default, because
  they measured nothing.** `KRK_MOE_LOOKAHEAD=1` stages the layer's plan before
  the per-expert loop (one batch instead of a promotion inside the loop) and
  `KRK_EXPERT_LOOKAHEAD=<k'>` adds the runners-up, staged only when WARM already
  holds them. Three interleaved passes each: 22.3/21.4/23.4 tok/s baseline,
  23.0/20.3/23.6 staged, 22.6/22.4/22.3 with k'=16 -- and **84.9% HOT in all
  three**, 3620/3630/3660 promotions. The switch also had its condition
  inverted: the comment says the batched lookahead is for `n > 1` (a verify pass
  knows its union a row early) and the code ran it for `n == 1`; it is `n > 1`
  now. `--mtp-spec --draft-tokens 2` on the same file is worse still (greedy:
  **9.2 tok/s**), so "amortise the promotions over verified tokens" does not pay
  on this shape either. The wall is the link: load reports 1.88 GiB in 553 ms
  (3.4 GB/s), the promotion path 3.8-4.4 GB/s (~570 us/expert, 68% of a step),
  and `KRK_HOST_PIN_MB=2048/4096` moves it only to 4.1-4.9 GB/s. The next lever
  is a smaller resident representation, not another cache policy.
- **The VRAM ledger is a print, not an inference, and the inference it replaced
  was wrong by >30x.** `Engine::log_vram_ledger` runs at the end of init and at
  teardown and prints
  `vram ledger (when): total = weights + kv + experts (pool + ring; live, reusable,
  budget) + workspaces + free + unaccounted`, with `ws_bytes_` accumulated in
  init's `alloc` lambda and a `vram at start` line giving the device's free
  memory before this engine allocates anything. Measured on Tiel 35B-A3B @ctx
  4096: **workspaces 88 MiB**, kv **80 MiB** (20 KiB/token — the hybrid has only
  ~20 attention layers of 40, so a dense-transformer KV formula over-counts 2x),
  dense 1,920 MiB, experts 7,682 MiB. The old "scratch is ~3.0 GiB" figure was a
  subtraction (`total − dense − KV − experts − free`) and the residue was the
  driver and other processes: the start line reports **2,251 MiB held on one
  launch and 4,408 MiB on another**, so `hipMemGetInfo` is device-wide and moves
  with whatever else is running. Read `unaccounted` as environmental, and never
  size a tier from a subtraction of driver numbers.
- **An expert's bytes are per-layer, and the index's `expert_bytes` is layer
  0's.** `blk.0` on Tiel-Coder-35B-A3B-MTP-APEX is Q6_K gate/up + Q8_0 down =
  **2.703 MiB** (2,834,432 B — exactly what the JSON records), while the corpus
  average measured from WARM is **2.26 MiB** because later layers carry Q4_K
  down-projections. Both numbers are right for different layers; sizing a cache
  from `expert_bytes` over-provisions every tier by ~20%. Derive the average as
  `sum(per_expert(gate)+per_expert(up)+per_expert(down)) / n_layer` from the
  tensor table (`tools/gguf_tensors.py`), not from the layer-0 field.
