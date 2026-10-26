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
  rc=0 with 0 `error:` lines, `kraken-tests` 4521/4521, `sh
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
  the default suite (group 36 of 95, 26 checks) and the baseline carries it.
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
  cases landed, 2994 -> 3008 when the chunk-invariance gate did, and 3008 ->
  4431 across the MoE decode campaign (per-layer expert counts, LatentMoE, the
  NVFP4 sidecar scale, the one-launch dispatch and the DFlash spelling arms)
  plus the CLI-refusal group, which together took the suite from 86 to 94
  groups. 4431 -> 4497 when the LFM2.5 shortconv group landed (95 groups), and
  4497 -> 4508 when that group grew its recorded-row pin (24 -> 35 checks), and
  4508 -> 4521 when `test_expert_cache_warm_tier` grew from 77 to 90 for the
  `KRK_EXPERT_WARM_MAP` arm. Read the count off the run (`4521/4521 checks
  passed`) rather than trusting the number written here, which is only the last
  one seen.
- **`kraken-tests` runs each of its 95 groups in its own child process, several
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
  make a run replayable. `--group <index|name>` runs one group's six arms (~22 s
  -- the ~6 s a stale note here used to claim is impossible, because an arm is a
  FULL suite run; see the cost paragraph below)
  and `--selfcheck --selftest` is 17 constructed answers, one per way an arm can
  be wrong (~10 ms) -- that one runs in every isolation check, because a checker
  that cannot fail proves nothing. `KRK_SELFCHECK_BREAK=counts` makes the
  expectation wrong by one check, so a real mismatch can be constructed: it
  only tightens (a green run goes red, never the reverse), and the isolation
  check requires rc=1 from it, which is how the exit status is proved and not
  just the checker. The whole sweep is 95 groups x 6 arms = **570 suite runs**,
  because an arm is a FULL suite run by construction: its claims are "every
  group the arm did not aim at still passes, with its own count" and the totals
  arithmetic, and neither survives scoping an arm to one group. Measured
  **22.3 min for the whole gate** (suite + proofs + sweep) at eight workers.
  **The arms of a group now run AT THE SAME TIME, each child on its OWN
  environment** (`ChildEnv`, built as a Windows block / an envp by
  `spawn_child_to`): an injection IS an environment variable, two arms differ
  only in it, and that -- not the arms' design -- is why they could only run one
  after another. The split is `min(6, workers)` arms in flight x
  `min(8, cores / arms)` workers each, so the product is one suite's width and
  not six times it; swept at 2/3/4/6/8 workers on one group the default wins
  (22.6 s against 31.5 / 26.8 / 26.2 / 22.2). The same 498 runs used to cost
  **~55 min** serially -- one arm at eight workers is ~7 s, and 498 of them in a
  row is 58 min -- so this is 2.4x, and the remaining bound is the box: a
  full wave of six runs spends ~2.3 s per full suite against ~0.9 s for a run
  alone. A note that quoted "240 s at eight workers" was never reachable at
  7 s a run and is gone.
  `--selfcheck --report <path>` records the arms' verdicts in
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
- **hd 256 has a matrix-core prefill attention arm, and the thing that makes it
  safe is not bit-identity but TILE identity.** `attention_wmma_kernel` runs 16
  query rows per block, stages K/V once per 32-key group and shares it across
  all 16 rows, and puts both matmuls on `v_wmma_f32_16x16x16_f16`. It is not
  decode-identical -- the K walk is 16 wide and the PV reduction is one MMA per
  16 keys -- so **it is wired to the rung, not to the default**: `--prefill fast`
  (which sets `KRK_GEMM_EXACT=0`) gets it, `--prefill exact` and
  `--prefill-hash` keep `attention_bexact_kernel`, and an explicit
  `KRK_ATTN_WMMA=1|0` outranks the rung for A/B. The rung is the whole contract:
  `--prefill exact` promises a hash, so a kernel that cannot produce one must
  not be reachable there. `attn_wmma_on()` reads the environment **per call**,
  not into a cached static, because `kraken-bench --bitexact` flips it inside
  one process to measure both kernels.
  What it claims instead of identity is that a row's output does not depend on
  how many other rows shared its launch: every row walks absolute 32-key groups
  from 0, and an MMA output row does not read the other 15. That is the property
  `kraken-bench --bitexact` pins -- one call of N rows against `ceil(N/16)`
  calls of one tile, **6 of 6 identical** at hd 256/16 heads, rows 16/32/48 and
  64/200 keys -- and it is **enforced** (it adds to the exit status, unlike the
  f16 decode-against-prefill rows). The same arm requires the matrix-core output
  to DIFFER from the bit-exact one, because a kernel that refused its shape and
  fell back would otherwise pass by measuring bexact against itself.
  Red construction: `const i64 n_keys = pos0 + (t0 & 16) + n_rows;` -- a bound
  that depends on the tile start -- fails exactly the 48-row arms (65516/196608
  values, max 19797 ulp) and exits 1; reverted, 6 of 6 again and the 24 f16
  identity rows stay green. 48 rows is the arm that catches it: with 32 rows the
  mutation happens to give both arms the same bound, so a 16/32-only sweep would
  have called it clean.
  **It is a long-context arm, and saying otherwise would be a lie with a
  measurement behind it.** Qwen3.5-35B-A3B, 4140-token prompt, `--chunk 512`,
  `KRK_TIME=attention`: 90 calls total, median of 3 interleaved runs, **44.9 ms
  on the matrix cores against 54.9 ms bit-exact and 75.9 ms tiled** -- an 18%
  cut of an op that is ~0.5% of that prefill (routing 8547 ms, experts 4780 ms,
  attention ~50 ms). End-to-end prefill does not move at 4K; the arm pays where
  attention stops being a rounding error. A trap that cost a session: the three
  kernels are only comparable WITHIN a rung. Under `--prefill exact` the same
  three read 335/417/338 ms, so the rung's own queue is most of what the
  recorder reports there -- do not quote either set as a kernel's cost without
  naming its rung. Text is unaffected: 24 greedy tokens are identical at
  `--chunk 8` and `--chunk 512` on both arms, and the hash gate's
  `bf32bbb3...`/`e8d8efa9...` at chunks 1/2/8/16/32 and its `f034bf9b...` at 1/8/512
  are unchanged.
- **The `routing` component row is a PARENT span, and reading it as the router's
  own cost is off by ~4x.** On Qwen3.5-35B-A3B, 4140-token prompt, `--chunk 512`,
  `KRK_PROFILE=1`, `moe.route.batch` reads 23885 ms over 360 calls (66.3 ms a
  call) and `moe.experts.batch` reads 17151 ms (47.6). They NEST: `route.batch`
  contains `route.rest.batch`, which contains `experts.batch`. Per 512-row chunk
  (2666 ms): experts 1904 (71%), the router D2H readback `moe.route.fetch.batch`
  596 (22%, 14.9 ms a call for a 512 KB copy -- it is waiting on the device, not
  copying), and the host's own top-k + plan 136 (5%, 3.4 ms a layer for 512x256
  scores). So the host is not doing 8.5 s of routing; it is doing ~1.2 s across
  that prefill, and the 22% is the expert tier's device time surfacing at the
  one blocking point in the layer. Nothing to win by moving the top-k to the
  device until the readback stops being an exposure. Two traps: the component
  table takes the MAX of nested rows, so `routing` and `experts` overlap by
  construction and must never be added; and `KRK_ROUTER_2X=1`, the control that
  separates the copy from the wait it sits behind, currently dies at prefill
  start with `hipErrorInvalidHandle` (rc=127), so that split has to come from
  the arithmetic above rather than from the arm.
- **LFM2.5 (`lfm2moe`) RUNS, and all four of the things that stopped it were
  silent-wrong-answer defects rather than missing pieces.** The stack is 3 conv :
  1 attention (attention at indices 2, 6, ... 38, i.e. 10 of 40), hd 64 over 8 KV
  heads of 32 queries with q/k norms, 2 dense leading blocks then 64 experts
  top-4 with `exp_probs_b.bias` and sigmoid gating, and its final norm is
  `token_embd_norm` (there is no `output_norm`). Derive conv-vs-attention from
  `shortconv.in_proj.weight` being present (as `nemotron_h_moe` does with
  `ssm_in.weight`): the file has no `attention.recurrent_layers` array and the
  `(l+1) % full_attention_interval` rule is off by one for it.
  (1) **The mixer is `C * conv(B*X)`, not `B*conv(X) + C*X`.** `in_proj` is
  3*n_embd wide and emits B | C | X; the reference multiplies B by X *before* the
  convolution and gates the CONVOLVED row with C afterwards
  (`hidden_states = B * x` immediately before the conv in transformers'
  `Lfm2ShortConv`, and the same two lines in llama.cpp's graph). The plausible
  reading has the same shape, the same op count and one fewer multiply, and it
  produces a finite, fluent, wrong model -- `"The" (the) (the) (the)` on the 24B.
  (2) **Its RoPE pairing is NeoX half-split.** llama.cpp prints `rope type = 2`
  for the file, so the GGUF keeps the HF layout unpermuted and the engine must
  pair `(j, j + n_rot/2)`. With the interleaved pairing the same prompt decodes
  fluent but different text -- a short text arm cannot see this.
  (3) **The top-k sigmoid weights are sum-normalized.** Without
  `expert_w_norm` the 24B says ` "The" and "the" are the same`; with it, ` Paris.
  The capital of Italy is Rome`. The two differ by the sigmoid sum (about 2x on
  every expert contribution), so again a fluent wrong answer, not a crash.
  (4) **The conv ring's f16 rounding is load-bearing for chunk invariance.**
  The ring holds B*X as f16, but a row the SAME call produced was still f32, so
  `--chunk 1` carried one extra bit into the tap: measured at pos 1 of layer 0,
  0.00055837 against 0.00055800, which moved the trace hash (b3c59e96... vs
  cc555b05...) and the greedy text. Rounding the in-call product to f16 too
  makes chunks 1/2/5/8/16 one hash: trace `c19de85c8a06d85d`, logits
  `9fce30dd39c9bf6f`, first token 4741 on the 24B-A2B file.
  **The BOS default was the fifth, and it lives in the tokenizer, not the
  family.** `tokenizer.ggml.add_bos_token` is OPTIONAL and its absence means
  TRUE (llama.cpp's rule); this engine defaulted it to true for SPM only, so the
  gpt2-style LFM2 file got no BOS even with `--tokenize` asking for specials, and
  without its BOS this model outputs whitespace soup. The same prompt then reads
  ` Paris. The capital of Italy is Rome. The capital of`, against llama.cpp's
  ` Paris. The capital of Germany is Berlin. The capital of` -- the one word that
  differs is a genuine near-tie between two numerically different pipelines
  (f16 GPU against the CPU q8 dot-products), and an independent f32 numpy
  reference of the same graph agrees with the engine word for word. So the
  reference gate here is the FIRST token plus the whole logits row's own
  consistency, not the sentence: `test_lfm2_shortconv` pins the fixture's row
  and its chunk invariance, and the 24B file's text is evidence, not a pin.
  `docs/AUDIT-lfm2.md` has the measured geometry.
  **The row is now PINNED, and the arm that keeps it honest had to be fixed
  twice.** `test_lfm2_shortconv` grew from 24 to 35 checks: the fixture's last
  prompt row by FNV-1a (`0x399373273e5a2b60`, argmax 2, CPU, `the`, chunk 8),
  a second prompt moving it, chunk 1 reproducing it, and a fixture whose ONE
  conv coefficient is moved (`0xd8b9a0cef214f8fc`). Red construction on the real
  arithmetic: reading the mixer as `conv(X)` instead of `conv(B*X)` in
  `backend_cpu.cpp` fails 3 of 35 (row, argmax, chunk 1 == 8); reverted, 35/35.
  The trap: the first moved-tap arm changed `w[0]`, the tap a ONE-row prompt
  never reads (its taps address p < 0 and are skipped) -- the row stayed
  bit-identical and the arm read as "the pin is blind" when the pin was fine. It
  moves the LAST tap row now. A chunk-invariant mixer that is simply wrong
  passes every invariance arm, which is why the row and not only the hashes is
  recorded.
  **The ranked warmup is ON by default now, and the old rule was wrong about
  what it is for.** The auto default used to fire only when the expert CORPUS
  exceeded the plan -- "a set that fits is resident anyway, so staging is pure
  wall time". Three files say the premise is false, because what staging removes
  is not eviction but COMPULSORY FIRST TOUCH: a layer's first pass reads every
  expert it routes to whatever the budget is, and that read sits inside the
  prefill. Same card, greedy, one process per arm:

  | model | corpus vs budget | decode | prefill | COLD |
  |---|---|---|---|---|
  | LFM2.5-24B-A2B | 12312 MiB vs 11964 | 39.7 -> **83.4** | 2410 -> **111 ms** | 25.7% -> 0 |
  | Qwen3-30B-A3B i1-Q2_K | 10008 vs 10008 (exact) | 9.7 -> **47.5** | — | 41.8% -> 0 |
  | llada-moe-7b-a1b | 3720 vs ~11.9 GiB (3x slack) | 19.2 -> **107.4** | 541 -> **34 ms** | 36.4% -> 0 |

  The third is the decisive one: pool slack does not make the first touch free.
  **The counter-example was real, and this note blamed the wrong half of the
  phase for it.** On `Tiel-Coder-35B-A3B` (22.6 GiB corpus against an 11920 MiB
  pool, 320 expert acquires per token) the default was a 1.7x loss at 256 tokens
  -- 20.5 s / 17.0 tok/s with `--expert-warmup 0` against 35.2 s / 10.4 warm --
  and the explanation here used to be "WARM covers the corpus, so every eviction
  comes back as a promotion" (28,050 against 2,702). **That was not the cause.**
  The phase has two halves, and switching only the second one off isolates it:
  with the staging untouched and `KRK_WARM_PREFILL=0` the same file reads
  **22.3 / 24.1 tok/s and a 363 ms prompt** against 10.4 and 1197 ms. At that
  ratio the prefilled prefix is not a head start INSIDE the working set, it is
  the pool (4757 experts, 10500 MiB of a 10498 MiB budget, and a set the run
  routes slightly differently), so it holds HOT at 79% against the 87.5% the LFU
  earns when it fills the pool itself.
  So the prefill is now gated on **the pool holding essentially the whole routed
  set** -- `corpus <= pool + 5%`, where the 5% is LFM2's measured 12312 against
  11964. Measured either side of it: Tiel 215% -> skip -> 22.3 / 24.1; LFM2
  103% -> prefill -> 84.0 / 84.8 (37 ms prompt) against 25.2 with the prefill
  off; Qwen3-30B 100% -> prefill -> 46.0 / 47.0, unchanged. `KRK_WARM_PREFILL=1|0`
  overrides the rule, which is how each row was measured.
  **The seed FRACTION is chosen by the same rule now, so neither reply length
  pays for the other's default.** The gate above is binary, and the measurement
  says a binary is wrong in one direction: at a ratio the gate declines, a
  prefix is still worth it to a SHORT reply. REAP-320 (4.1x, 8 GiB pool) at 24
  tokens: half the pool seeded gives 41.7 tok/s prefill at decode parity,
  against 27.0 unseeded and 31.9 seeded whole. Tiel-Coder-35B-A3B (2.17x) at 256
  tokens: half a pool is +44% prefill and **-45% decode**, and the two arms cross
  at ~96 reply tokens (4 tokens: the seed wins by 34%; 128: loses by 19%; 256:
  by 71%), because the seeded pool starts full of the PROMPT's set and the
  continuation evicts through it (promotions 9579 -> 23548) while the LFU's own
  choice keeps learning (per-step 92.2 -> 46.8 ms unseeded, 66.8 -> 83.1
  seeded). So: `corpus <= pool + 5%` seeds the whole pool as before; a larger
  ratio with `--max-tokens` <= 64 seeds **50%**; anything longer or unbounded
  seeds **nothing**. `--max-tokens` is the only expectation the engine has at
  init, so it is what the rule reads (`EngineConfig::expected_reply_tokens`,
  0 = unknown = long; `main_server` leaves it 0 on purpose, because its
  requests carry their own budgets and 0 is the arm measured safe at every
  length). `KRK_VRAM_SEED_PCT=0..100` and `KRK_WARM_PREFILL=1` still outrank the
  rule. Verified against the flag it replaces on both shapes: LFM2 (fits) 100%
  cap, 2357 experts / 11932 MiB; Tiel `-n 8` 50% cap, 2363 experts / 5217 MiB
  and `-n 256` 0% with +0 experts; REAP-320 `-n 24` 50% cap, 1859 experts /
  4002 MiB -- the same numbers the hand-set `KRK_VRAM_SEED_PCT=50` produced.
  **The seed can only promote what WARM staged**, so a COLD first run can hit the
  staging ceiling and seed nothing (`VRAM +0 experts (0 MiB of a 50% seed cap)`):
  that is the safe arm's performance, not a wrong answer. And it is a
  prefill-for-decode trade wherever WARM is short of the set -- at `--ram-tier
  24` it is 12.9 prefill / 6.7 decode against 9.55 / 7.0 unseeded, so there the
  best wall clock depends on the reply length. **Pinning is not the mechanism:**
  `KRK_EXPERT_PIN_HITS` was added to test the obvious reading and it is
  backwards -- with pinning effectively off the same seeded arm reads 9.1 tok/s
  against 12.2 at the default 24, because pinned slots are what stops the seeded
  set being re-promoted. An earlier version of this note said the seed loses at
  every length; it extrapolated 256-token per-step costs down to short replies,
  and neither arm's step cost is constant in length.
  The full curve, both machines and the two defects found on the way (a
  full-forward scan cannot run at the budget it describes; a 0-position index
  must not be written or trusted) are in `docs/AUDIT-REAP-DECODE.md`.
  Neither a byte bound on the STAGED set nor a `--max-tokens`-derived time
  ceiling was the lever: both were implemented, measured (2622 MiB -> 17.6, 5210
  -> 12.3, 9278 -> 19.2, 10507 -> 12.8, 22604 -> 10.4 tok/s, i.e. non-monotonic
  with a cliff between 9.3 and 10.5 GiB) and reverted -- they only ever moved
  the prefix's SIZE, and the prefix was never the win. `docs/PERF-WARMUP.md` has
  the curve, the counters and both decompositions.
  **`KRK_EXPERT_WARM_MAP=1` is the other arm, and it LOST, which is why it is
  off by default.** It keeps nothing in host RAM -- an evicted expert is served
  from the page-cached mapping, and the warmup's stand-in for the staging pass
  is `touch_layer()`, the same batched reads discarded -- on the theory that a
  promotion copies bytes twice (RAM -> ring -> VRAM) and a mapping read once.
  Measured: Tiel 13.6 tok/s against 17.2 for `--expert-warmup 0` **at the same
  hit rate** (87.9% HOT against 87.3%) and 12.1% COLD against 6.6%, i.e. ~385 us
  per extra miss; LFM2 16.5 against 84.0. The copy was never the problem: a
  promotion is ONE batched call whose slice N+1 stages while N is in flight,
  while `acquire()`'s mapping path is three `upload()` calls with a pool-honor
  wait and its own read each. `test_expert_cache_warm_tier` grew 77 -> 90 checks
  to pin the arm anyway (no admission, no per-expert host block, an evicted
  expert reloaded rather than promoted, `touch_layer` retaining nothing) and
  disabling `warm_admit`'s refusal fails 4 of them.
  So `--expert-warmup -1` means ON for every MoE with a device tier, and the
  corpus-vs-budget comparison survives only on the CPU backend (where the
  suite's fixtures assert on the counters a staging pass would move). The
  earlier LFM2 numbers that made `-1` look like a four-MiB near miss came from
  the plan being made before the KV tier and workspaces exist: 12312 planned
  against a 11964 earned budget.
  It is a wall-time tax, not a throughput one: 3.87 s up front on the 24B
  (12312 MiB read at 7.9 GB/s), so a single cold 24-token run's wall is worse
  (6.66 s against 4.77 s) and break-even is about 165 output tokens; a second
  request in the same process pays none of it. `--expert-warmup 0` and
  `KRK_EXPERT_WARMUP=0` still disable it, and `--expert-warmup-ms` bounds the
  phase for a corpus too big to stage. `--expert-warm-prefetch 1` ALONE is
  still a loss (25.5 tok/s): it fills WARM without staging VRAM, so every decode
  step pays a promotion.
  **The DSpark drafter does NOT load, and the earlier note here was wrong to say
  it was a valid `dflash` head set.** `--draft` exits 1 with `DFlash head set
  has neither enc.aux_norm nor aux_hidden_norm.0`, which is the right answer:
  the file is a different input CONTRACT. It ships `fc.weight` [10240, 2048] --
  five target hidden layers concatenated and projected, in place of the per-layer
  aux norms -- plus `markov_w1`/`markov_w2` [256, 128000] (a rank-256 additive
  bias on the PROPOSAL logits, taken from the anchor) and `conf_proj` [2304, 1]
  with `sample_from_anchor`. Running it without the markov head proposes
  different tokens, so no acceptance figure for this file exists yet and none
  should be quoted. The port is `fc` as an input transform (and a 5-slot hidden
  capture), the markov bias, and the confidence gate.
- **The int8 K/V cache is chunk-invariant on the device now, and the defect was a
  reduction order.** `--kv-quant int8` had every reader implemented and still
  produced three different hashes for one prompt (chunk 2 == chunk 4, chunk 1 and
  8 elsewhere), which is why `kv_quant_supported(Int8)` refused it. The cache
  bytes were never the problem: `kv_append_i8_kernel` quantises per (token,
  plane) row, and 200 rows one at a time are byte-identical to one 200-row append
  (the CPU group pins exactly that). The difference was the DECODE lane's dot:
  `attn_lane_dot_q8` summed `q[e] * dequant(k[e])` as a per-element f32 FMA chain
  while the tiled and bit-exact paths consumed dequantised PAIRS through one
  `v_dot2_f32_f16` -- same terms, different rounding, up to one f16 ulp per
  channel. Decode now builds the pair in a register and calls the same
  instruction, and one 21-token prompt (`smolcode-coder-cpp-1.5b-q4_k_m`) gives a
  single hash at `--chunk 1, 2, 4, 8` and `16`: trace `6d4758c0c4754be0` /
  logits `dc6d7126d56225fb`, first token 576, against f16's
  `2cb4e7fc46508835` / `67f95a7185846df2` with the same first token. 24 greedy
  tokens are byte-identical text between the two formats. MLA and the
  nextn/draft block stay refused for int8 by name (they own their caches).
  **The arm is per kernel, in `kraken-bench --bitexact`, and it is enforced.** Two
  claims, measured separately because they fail for different reasons: a
  position's cache bytes must not depend on the append's arity (one 8-row append
  against eight one-row appends, byte-exact) and attention must not depend on the
  query-row count (decode rows against one prefill call). 27 of 27 identical at
  hd 64/128/256, rows 2/4/8, 64 and 200 keys; `KRK_ATTN_BEXACT=0` makes 17 of 27
  differ by 1 ulp, which is the red construction, and `--bitexact` exits non-zero
  when any int8 arm differs. **That exit status is the part that had to be
  built**: `failures += arm_rows_bitexact(be)` returned 0 by design, and the only
  conversion to a failure was `gate && failures`, so a run could print
  `17 int8 K/V arm(s) are NOT chunk-invariant` and still exit 0.
  Gate: suite 4122/4122 in 86 groups, baseline `no regression`, `build_check.sh`
  0 failed (fingerprint re-recorded for the relink), `kraken-bench --gate` rc=0.
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
  `KRK_TEST_SERIAL_OVERRIDE=<grp>` forces one for a run so the lane is exercised.
  **Exactly one group declares one**: `test_pageable_path_probe` times host
  copies, so a neighbour's traffic would be read as the simulated aperture --
  the first real user of that lane. **The lane does not protect it from the
  selfcheck, and that was measured.** The sweep runs six suites at once, so the
  probe is measured beside five neighbours -- exactly the condition its
  declaration names -- and its blocking arm tolerated only 5.6 ms of stall
  before `pageable_path_choice`'s 1.20x margin collapsed (at 16 MiB its ring is
  2.0 ms against blocking's 0.73 ms). One gate run in three failed with
  `another group is not passing: test_pageable_path_probe`. The engine-driven
  arms of the second and third populations now simulate at 0.1x the rates:
  same ratios and the same verdict, 56 ms of tolerance instead. Proven by four
  consecutive green gate runs, against a red at 1.0x that failed one in three.
  **The width the sweep uses for this had to be global, and the scoped version
  was measured red first.** Every arm IS a full suite, so any two arms in flight
  are two neighbours of the probe -- including arms of a DIFFERENT group. Scoping
  the serial width to the serial group's own arms (`--selfcheck --group 0`, six
  arms of a normal group) failed at once: `another group is not passing:
  test_pageable_path_probe` in the crash-last arm. So when the suite contains a
  serial group the whole wave is one arm at a time, and that arm gets
  `min(8, cores)` workers rather than a sixth of them. Cost: `--group 0` 24 s ->
  44 s, and the optional 564-run sweep ~22 min -> ~68 min; both are wall time,
  not a claim, since the arms differ in env, scratch, log and report and their
  order is not part of what is proved. Every other group is silent here: none opens
  the device, binds a fixed port, or reads a clock. Two defects had to be fixed first --
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
  **`KRK_TEST_INPROC` was the one mode without an area, and that was a live
  defect until it got one.** It ran where it was started (the repo root), and the
  cleanup works by SHAPE, so a second run starting there deleted the first one's
  fixtures: `test_pageable_path_probe` reported **129/140** with `cannot open
  'kraken-aperture-path.gguf'`, and the same binary in the next run reported
  140/140. It is `std::filesystem::current_path(<area>)` now, with the caller's
  `--report` path resolved to absolute BEFORE the move (it means what it said
  relative to the start directory) and the area removed when green unless the
  report lives inside it. Control, on the current binary: two concurrent
  in-process runs are **4431/4431 both** with a distinct area, and **4404/4412 +
  4417/4431** with `KRK_TEST_SCRATCH_ID=sharedctl` pinned for both -- which is
  exactly the old shared-directory behaviour, so the fix is what is being
  measured.
- **The reporting proof is in the default gate; the all-groups sweep is not.**
  `scripts/kraken_tests_baseline_check.sh` runs the runner's own selftest
  (`--selfcheck --selftest`, 17 constructed answers) and ONE group's six arms
  (`--selfcheck --group ${KRK_SELFCHECK_GROUP:-0}`) before it compares against the
  baseline, because a gate that compares two reports is worth nothing if the
  reports stopped meaning what they say: a runner reporting a crashed group as
  passing would compare clean against a baseline of the same defect. Both take
  seconds, and the block can be shown able to fail -- `KRK_SELFCHECK_GROUP=99`
  names a group that does not exist, and the gate then exits 1 with `the reporting
  proof failed for group 99 (rc=2)`. The 498-run sweep and its arm-by-arm
  comparison against `tests/kraken-tests-selfcheck-baseline.json` stay behind
  `KRK_BASELINE_SELFCHECK=1` for cost (22 min with the arms in flight, 55 serial;
  see the reporting-proof bullet above).
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

- **BitNet's dense FFN needs two things an f16 activation buffer does not give
  it, and it ran without both.** `bitnet-2b-tq1_0` (2B dense ternary, 30 blocks,
  n_embd 2560, n_ff 6912, 20 heads / 5 KV, rope base 500000) now answers with the
  same words on the device as on `--cpu` -- `The capital of France is the capital
  of France ...` at 8, 16 and 32 tokens -- and `scripts/coherence_check.sh`
  classifies it `PASS`. Before this it said `a capital is a capital is a capital`.
  `kraken-inspect` still reports `arch bitnet -- unknown (llama-like defaults)`,
  so the family's verdict rests on the loader's own reads plus that gate and not
  on a declared table row.
  (1) **SubLN is a SECOND RMSNorm per sub-block, on the intermediate.** The file
  carries `attn_sub_norm` [n_embd] and `ffn_sub_norm` [n_ff] per layer; the widths
  say which is which, and neither is `attn_norm`/`ffn_norm`, which are the
  PRE-block norms. The order is `attn_norm -> qkv -> attention -> attn_sub_norm ->
  wo` and `ffn_norm -> gate/up -> silu(gate)*up -> ffn_sub_norm -> down`. Read
  against llama defaults they are simply unknown tensors, and the model ran
  without them.
  (2) **`silu(gate)*up` leaves f16 by 53x here, and the norm that consumes the row
  is why that is not a rounding error.** Measured on layer 0 with `--cpu` as the
  f32 reference: that row's peak is **3.497e6** against f16's 65504, **88 of its
  6912 elements** are over the ceiling, and the unscaled f16 store hands its
  consumer a row whose **direction is 58 degrees off** (`cos` 0.4153, `relL2`
  0.946) -- an RMSNorm reads direction and never size, so nothing downstream can
  undo that. The row-scaled store lands at `relL2` 3.3e-4, `cos` 0.99999995.
  `tools/dump_diff.py` over the same pair puts the FIRST divergence at
  **`attn.ffng L00`, rel 5.24e1** with nothing before it, and `attn.down L00`
  (rel 4.2 unscaled against 3.5e-4 scaled) is the cascade.
  The fix is `Backend::silu_mul`'s row-scale form: `row_len` > 0 asks for the
  products to be stored divided by ONE power of two per row, chosen so that row
  fits, and the multiplier is written to `row_scale[row]`. A power-of-two divide
  is exact in f16, so it buys exponent range and spends no precision, and it is
  bit-identical to the plain store for a row already in range. The dense FFN asks
  for it on exactly the layers that own an `ffn_sub_norm` and passes `ws_fscl_`;
  its consumer folds nothing back because the RMSNorm it feeds is
  scale-invariant -- which is why the form is safe there and would not be for an
  arbitrary consumer. `KRK_FFN_ROWSCALE=0` is the A/B arm (the old store).
  **The guard that makes this loud is fatal by default, which is the point of it.**
  Every run prints `range f16 activation bounds fired: silu N, experts-sum N,
  residual N`; the unscaled arm exits **rc=2** with `silu 11975` at 32 tokens
  against `silu 0` and rc=0 for the scaled one, and `KRK_RANGE_STRICT=1` keeps it
  fatal where a caller has asked for a comparison. The fix's own gate is a strict
  run: device, 32 tokens, **rc=0 and no clip line at all**.
  The cost is real and it is not the arithmetic: **3-5% slower decode, prefill
  unchanged** (six runs per arm over three experiments -- 48.4 against 49.5, 50.1
  against 52.0, 48.0 against 50.5 tok/s; prefill 96.4 against 96.0 ms at a
  6-token prompt). The fused gate/up epilogue holds ONE element of the row per
  warp, so no row maximum exists inside it and a row-scaled layer declines it:
  `silu_mul` becomes its own launch (960 calls for 32 tokens x 30 layers, 1.91 ms
  against 0.05 ms of kernel time) and writes two rows where the fused form wrote
  one. That kernel time is 0.3% of the run and the arms' op tables are not
  comparable row by row (the scaled arm's whole table reads ~10% higher on ops
  whose arguments are identical), so the 3-5% is a whole-run measurement taken
  three ways and not a per-op decomposition -- the lost epilogue and its exposed
  launch is where it lives.
  `kraken-bench --silu-scale` is the arm that keeps the store honest and it is
  ENFORCING (it adds to the exit status, like `--bitexact`'s int8 rows): **32 of
  32 checks** over four cases -- an in-range row that must come back bit-identical
  to the plain store, an over-range row at scale 64, the same row at a narrow 1024
  width, and two rows in ONE call whose scales must differ per row (1.0 and 64.0).
  Red construction: disabling the wide scaled path fails **26 of 32**, both wide
  cases reporting `scaled clips=6912` while the narrow register case stays green,
  so the arm is specific to the path it names rather than counting a failure.
  Gate: `kraken-tests` 4528/4528 in 95 groups, `scripts/build_check.sh` 0 failed,
  the baseline check `no regression`, `kraken-bench --bitexact` rc=0, the
  coherence check PASS on all three shapes (Qwen3-4B-Q8_0 dense, Qwen3.5-9B-Q4_K_M
  recurrent, Qwen3-30B-A3B i1-Q2_K MoE, `1 coherent, 0 not` each), and `bench --gate` green
  except the speculation arm, which cannot find
  `models/SmolLM2-135M-Instruct.Q4_K_M.gguf` on this box and fails before any of
  this code runs.

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
  **qwen4exp is SUPPORTED now, and the gate is what flipped it, not the graph.**
  The row sat at `ArchSupport::No` with every tensor read and every block proven
  because `test_qwen4exp_*` pinned the PLE rows, the lane invariant, the QSA
  cell list and the delta-net gate -- and none of them pinned the SENTENCE, so a
  graph that passed all of them could still decode nonsense (which is exactly
  what the PLE normalisation bug did). `test_qwen4exp_whole_graph` (18 checks)
  closes it: the last prompt position's **whole logits row** by FNV hash plus
  its argmax and the greedy sequence, and four arms that keep the pin from being
  vacuous -- a different prompt moves the row, chunk 1 reproduces it, a shifted
  PLE table breaks it, and the fixture LOADS with
  `KRK_ARCH_ALLOW_EXPERIMENTAL` unset (the claim the row makes, checked). The
  row hash, not the tokens, is the load-bearing part: this fixture's synthetic
  head argmaxes EOS after one token, and a text-only gate is blind in exactly
  the way the attention rewrite already demonstrated.
  **The HIP half of this group cannot run the fixture, and the reason is a
  real refusal that had to be built.** The lane stack's bottleneck is
  `hyper_connection.low_rank` (8 on the fixture, 320 on REAP-288), and it is
  the up-projection's K: `[hc*n_embd <- low_rank]`. The HIP GEMM walks K in
  32-value chunks and `std::abort()`s on anything else
  (`src/hip/backend_hip.hip`), so the CLI printed
  `gemm: n_in=8 is not a multiple of 32` and then died with **0xC0000409**
  two rows into the prefill -- reproducible from `cmd /c` with redirection
  and NOT from bash, which is why it first read as an exit-path crash.
  `Backend::gemm_k_ok(k)` (default true; HIP answers `k % 32 == 0`) is asked
  at init, and the run now REFUSES with the geometry named before any pass.
  The device arm pins both halves: the CLI's `--cpu` run hashes the same row
  the in-process graph does (`1de9761d6eabaceb`), and the device must refuse
  with `hyper_connection.low_rank` + `multiple of 32` -- a silent skip would
  accept a crash, and a fixture padded to K=32 would prove the kernels on a
  geometry the graph never has.
  **Its row was re-recorded once, and the bisect that says why is the part
  worth keeping.** The pin sat at `0x99dbc3445ac66d08` and the fixture
  produced `0x1de9761d6eabaceb`; the fixture writer is byte-identical to
  HEAD's apart from the new `perturb_ple` flag, and reverting `src/engine.cpp`
  and `src/model.cpp` to HEAD (then `src/backend_cpu.cpp` on its own), each
  rebuilt and re-run, left the row at `1de9761d` -- so the GRAPH was never
  the outlier and the recorded value was. What still makes the pin
  load-bearing was re-measured at the new value: a different prompt gives
  `dd27376b6c3e1639`, the shifted PLE table `d7ff2c97e4406ef8`, and chunk 1
  reproduces `1de9761d6eabaceb` exactly. A revision that moves it now has to
  move something real.
  Red construction: `if
  (false)` around `ple_block` fails 5 of the 18 (row, argmax, PLE sensitivity,
  both text pins). On hard weights the file is
  `H:/OLLAMA-Models/GGUF/qwen3.8-flash-next-reap-288-Q4_K_M.gguf` -- 78.04 GiB
  payload, 1224 tensors -- which now inspects as `supported (recurrent-moe)` /
  `runnable` and, with no hatch, decodes `def add(a, b):` -> `    return a + b`
  at **5.2 tok/s prefill / 5.8 tok/s decode**. That rate is the expert tier
  (42.2 GiB of routed set through a 7.2 GiB VRAM budget), not the graph: the
  verdict is a correctness claim, and the speed is a separate project.
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
- **deepseek4 RUNS now (dense core), and the two defects that stopped it were both
  wiring, not kernels.** The shared-latent graph landed in `Engine::mla_shared_forward`
  (`src/engine.cpp`): latent `attn_kv` -> `mla_shared_latent` (norm over the WHOLE
  row, rope in its tail), the query trunk `attn_q_a` -> its norm -> `attn_q_b`, then
  `mla_shared_attention` and `mla_shared_output` writing the layer's FINAL
  `[n, n_embd]` result into `ws_x2_` -- so the layer loop SKIPS the ordinary
  `attn_output` gemm for that layout (`mla_grouped`), because wo_a/wo_b are already
  inside. `MlaLayout::Shared` is no longer refused at init; Split still is, by name.
  `kv_dim_at` had to change with it: V4's cache row is the latent ALONE (rope is its
  tail), so the lite rule's `+ qk_rope` sized every plane 64 columns wide and made
  `pos_stride` disagree with the slot the tier handed out.
  (1) **The hash route indexes by VOCAB id, not by the row's position in the pass.**
  `moe_ffn` used `t` -- the row index -- so a 2-row prefill routed row 1 to token
  1's experts. Symptom: chunk 1 and chunk 8 gave different rows on the fixture and
  nothing else was wrong. `forward_core` now records `fwd_toks_`/`fwd_toks_n_` and a
  missing one is REFUSED rather than routed by position.
  (2) **A poisoned hash table entry has to be poisoned in EVERY slot**, or the arm
  tests whichever token it happened to hit: `hash_id_oob` first wrote one cell and
  the out-of-range arm passed while proving nothing.
  `test_dsv4_whole_graph` is **36 checks**: the fixture loads with NO hatch (the
  `Partial` row's claim), the last prompt row's whole logits FNV hash
  (`0x12602ed8b3a49add`) and argmax (0), a second prompt moves it, chunk 1
  reproduces chunk 8, a SHIFTED table moves it, an out-of-range id REFUSES with
  `hash route` in the message, and the greedy sequence (`0 4 6 6 6 9`) pins at
  both chunk sizes. **The pin MOVED when the multi-row prefill landed**
  (`0x25412f04b8170542` / argmax 4 / `4 4 4 4 4 4` before, the values above
  after), and that is a re-record, not an oracle: this fixture's truth is its
  own consistency. What corroborates it is cross-backend agreement on the
  DECISION, not the bits -- see the device arm below.
  (3) **A 13-row prompt (twelve words) prefills as ONE call at chunk 16 and as
  thirteen at chunk 1, and the CPU row is IDENTICAL** (`abd920b0bba85eb8`
  both) -- so the shared-latent multi-row path is exercised by a prompt wide
  enough to leave the GEMM's thin-row ladder, which a 2-row prompt does not.
  (4) **The device arm cannot use the CPU row as its reference, and the
  control that shows why is a stock model.** `SmolLM2-135M` at a 16-token
  prompt gives trace `0d47196029f2cb51` / logits `4ccba855c2b297e0` on the card
  against `a1e5be993b18b62e` / `668768a536fd8d50` on `--cpu`, with the same
  first token (30) both ways: a HIP row is not bit-equal to the reference on
  ANY model, so that comparison would have reported a shared-latent bug where
  the engine only has an engine-wide property. What the arm DOES pin, measured
  on the fixture: the device prefill folds to the same trace and the same
  logits row at chunk 1 and 16 (`64aed8850d5541ab` / `dcc441a09e8d64db`), it is
  deterministic run to run, and it samples the reference's token (7). Both
  hashes print on every run, so a future divergence is visible even when the
  checks pass. Five CLI runs, ~4 s.
  (5) **The CLI's CPU prefill TRACE is NOT reproducible run to run on this
  fixture**, while its logits row stays `d7ceb589f6f3aa5c` and the device is
  stable: three values were observed for one unchanged command (`d43d3bc1...`,
  `35ad165d...`, `15aa217b...`). Two consequences: a CPU trace hash must never
  be pinned, and the rule that the trace is the signal which never goes blind
  assumes a backend whose reduction order is fixed -- on `--cpu` it is not, for
  a multi-row prefill.
  **Red construction: rotating the leading slice instead of the tail
  in the CPU kernel's `mla_shared_attention` (`base = 0`) fails 3 of 21 -- row,
  argmax and sequence -- while `test_mla_shared` STAYS 109/109**, because that
  group compares the oracle and the kernel against each other and both are
  unchanged by a shared premise. The per-piece gates could not see it; only  the sentence could. Suite total 4105, 85 groups.
  **`test_source_literal_hygiene` (8 checks) exists because one session wrote
  the same defect four times.** A C string sent through a tool path that
  decodes the escape twice turns `
` into a REAL newline inside
  `fprintf(...)`/`KRK_ERROR(...)`, and every occurrence is a compile error
  that costs a build to find. The group is that rule made checkable: a
  scanner that knows `//` and block comments, character literals, raw
  strings (`R"tag(` ... `)tag"`) and the LEGAL backslash continuation, and
  five in-memory arms -- one broken literal on purpose, one continuation,
  one raw string, one comment, one escaped form -- before it is trusted on
  It now carries the NEIGHBOURING class too, which compiles and is
  therefore invisible: a DOUBLED escape in a print/error argument (source
  `\n`), where the gate prints the two characters instead of a line break.
  Only a format string is flagged -- `src/json.cpp`'s escape table and the
  JSON payload literals carry a backslash on purpose, and `// hygiene-ok`
  waives the rare print that really prints one. Two measured lessons sit in
  the arms: the backward search must STOP AT THE END OF THE STATEMENT (it
  first attributed a table literal on the next line to the fprintf above
  it, 2 offenders for 1), and the arms must build their inputs from
  character code 92, because spelling a doubled escape in the test is the
  defect itself.
  **The entry points are pinned, not merely swept.** `src/main_cli.cpp` and
  `src/main_server.cpp` are covered only because they happen to live under
  `src/`, so the group records both as READ and refuses if either stops
  being visited: narrowing the sweep to `{include,tests,tools}` trips the
  file-count floor (54 < 60) AND the entry pin (`cli NO, server NO`), 13/15.
  Proved on the real file, not synthetic: a doubled escape injected into
  main_cli.cpp's `kraken: unknown option '%s'` report is named at
  `src/main_cli.cpp:436`, and reverted leaves 15/15. The `kraken: ...`
  shape has its own arm, because the entry points are not shaped like the
  engine's `KRK_ERROR` macros.
  **Browser-facing text is judged by the module's job, not the call shape.**
  `src/http.cpp` builds a status line with `format(...)` (a sprintf by
  contract) and a static header array with no call context at all:
  `static const char head[] = "HTTP/1.1 200 OK
"`. In those two files
  (`src/http.cpp`, `src/server.cpp`) EVERY doubled CR/LF counts, while
  elsewhere only a print/error argument is flagged. Measured: the probe
  `"HTTP/1.1 200 OK\r\n"` in that static array is named at
  `src/http.cpp:146`, 17/18; reverted, 18/18. Current tree: 0 in both
  classes.
  Then it sweeps every `.cpp/.hpp/.hip/.h` under
  `KRK_SOURCE_DIR/{src,include,tests,tools}` (a configure-time define: the
  group cannot use the CWD, which is a per-group scratch dir by design). Two
  arms keep it from being vacuous: the scan must find >= 60 files, and the
  RED construction is a one-line file dropped into `src/` -- measured
  `first offender: src	mp_hygiene_probe.cpp:2`, 105 files, group 7/8 --
  removed again, 8/8. Current tree: **104 files, 0 offenders**.
  **Its runtime twin is `test_http_server`'s strict client, and the substring
  checks that came before it were measured blind.** The group's original 46
  checks read the wire with `find()`, which cannot tell well-framed bytes from
  bytes that merely CONTAIN the expected text. A strict HTTP/1.1 client now
  parses every response the listener emits: the status line must be exactly
  `HTTP/1.1 <3 digits> <reason>`, the head CRLF-only with no backslash anywhere
  in it, `Content-Length` must equal the body byte count, and a chunked body is
  de-chunked per RFC 7230 (hex size line, exact payload, CRLF, `0\r\n\r\n`,
  nothing after it) before the SSE framing is checked event by event
  (`data: <json>` + blank line, `[DONE]` last, one `finish_reason`). Group
  90/90.
  **The last 13 checks drive the binary itself**, because the listener and
  `kraken-server` share `HttpServer` + `OpenAiService` but only the binary
  proves what `main_server.cpp` owns: argument parsing, `--port 0`, and the
  banner a client learns the port from. It runs `--cpu` (the suite links no
  device backend; the bytes are backend-independent), its stdio goes to
  capture files, the port is polled out of `listening on http://HOST:PORT`, and
  the three responses are parsed with the same strict client -- including that
  the binary's completion text and its decoded deltas are the in-process
  reference's. It is TERMINATED, not asked to stop: `main_server.cpp` has no
  stop path (its own comment says `server.run()` returns only after `stop()`,
  which nothing calls), so the helper kills it and waits, and the group removes
  its captures on the way out (the runner's shape cleanup covers `.gguf` and
  `-dir`, not `.txt`). Measured red construction: printing `a.port` instead of
  `server.port()` in the banner -- the exact bug a client sees as a refused
  connection -- fails only `the binary's banner reports the ephemeral port`,
  79/80, with the banner dumped; reverted, 90/90, and no `kraken-server`
  process survives the arm.
  The blindness is the measurement, not an argument: a `Content-Length`
  inflated by one byte leaves all 46 substring checks GREEN and fails exactly
  the four buffered strict checks (73/77); changing the SSE event terminator to
  `\r\n\r\n` leaves the stream's 200 and every `data: ` search intact and
  fails four stream checks with `an event payload holds a line break` (73/77).
  A doubled escape in the `Connection: close` header plus the blank line breaks
  the response outright (50/72 -- the body vanishes, so the old checks fail
  too), which is why the two surgical mutations are the ones recorded. The
  parser carries its own arms (a well-framed response ACCEPTED, one refusal per
  defect class, the backslash-built-from-code-92 case), so a parser that
  refused everything could not pass.
- **The request parser's refusals are byte-checked, and four of them were
  missing.** RFC 9112 §6.3 makes the body length load-bearing. `serve_client`
  used to collapse two `Content-Length` headers to the last value and read
  `5, 44` as 5 through `atoll` -- the two shapes a smuggled second body rides on
  -- and a body that ended early was dropped with NO response at all (nothing
  dispatched, but the client learned nothing). It now answers the OpenAI error
  shape: 400 `duplicate Content-Length` (even for an identical repeat), 400
  `invalid Content-Length` for anything that is not pure decimal, 400
  `incomplete request body`, and **431** `Request Header Fields Too Large` for a
  header block over the 64 KiB cap -- 413 is about CONTENT, so a client retrying
  with a smaller body learned nothing from it. 413 stays for a body over 8 MiB.
  Every one of those paths goes through one `refuse()` that half-closes and
  drains before closing: on Windows a close with unread request bytes sends an
  RST, and an RST can throw away the refusal the client is about to read.
  `test_http_server` drives it with a raw socket (`http_exchange_raw`) that takes
  arbitrary bytes, can half-close where a truncated body arrives, and bounds
  every read at 10 s so a silent parser fails instead of hanging the suite: 25
  checks in this block alone. **Measured red before each fix**: 8 of the
  Content-Length checks failed (99/107), then 3 of the Transfer-Encoding ones
  (109/112), with the controls (a well-framed request, and the same truncated
  request sent whole) green throughout; green now 118/118 in the group.
  **A request-side `Transfer-Encoding` is refused the same way** -- 400
  `unsupported Transfer-Encoding`. This server implements no request coding, so
  a body whose extent is described by one (with a `Content-Length` beside it, or
  with none at all) is a message two parties frame differently. RFC 9112 §6.1
  suggests 501 for a coding a server does not understand; 400 is the refusal the
  duplicate header gets, and both mean "I will not guess where your body ends".
  **`Connection: close` is on every response now, not only on errors.** HTTP/1.1
  defaults to a PERSISTENT connection and `serve_client` handles exactly one
  request, so a buffered 200 that closed without saying so invited a compliant
  client to reuse the socket. Measured red: exactly the two buffered-200 checks
  failed (113/115) while the SSE head's check passed -- that head always carried
  the header, which is why the gap was only on the buffered path.
  **The listen port is configuration, not a literal**: `--port` wins, then
  `KRK_PORT`, then 8080, and an explicit `--port 0` still means ephemeral --
  measured against the binary: `KRK_PORT=8090` -> 8090, `KRK_PORT=8090 --port
  8091` -> 8091, `--port 0` -> 58872. The banner prints `server.port()`, so it
  reports what was actually bound, which is what the binary arm parses.
  The arch row is `Partial`, not `Yes`, and the difference is checkable: the
  CSA/HCA compressors, the sparse indexer and `attn_sinks` are absent, so init
  REFUSES any file whose `attention.compress_ratios` is non-zero (a run would
  attend the raw prefix instead of the pooled one) and refuses a file carrying
  sinks for the same class of reason. `test_mla_loader` was updated with the
  truth: the shared fixture LOADS without the hatch, and the Split layout is the
  one the engine still refuses.
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

- **A kernel READS the pinned host tier on this card, and that is what the
  three-tier design was missing.** `tools/probe_hostread.hip` (ad hoc: build it
  with the engine's own clang++ and `--offload-arch=gfx1201`, it is not a CMake
  target) reads a 4 MiB `hipHostMalloc` buffer back EXACTLY and a 256 MiB one at
  **24.3 GiB/s** against **485.2 GiB/s** for the same kernel over device memory,
  while the runtime reports `unifiedAddressing: 0` -- the flag says no and the
  pool is mapped anyway. So the answer to "can the RAM tier be used without
  being copied into VRAM" is yes, at a rate in the same class as the promotion
  copy (11-22.9 GiB/s).
  That is `Backend::host_load_supported()` (false by default, true on HIP only
  because of those numbers) plus `ExpertCache::acquire_inplace()`, which returns
  the WARM matrices with no device copy when the slot is pinned host memory, is
  not already resident, and has NOT earned residency (`kPinThreshold` touches) --
  a hot expert still goes to VRAM, where a read is 20x faster. `Engine::moe_ffn`
  tries it first, falls through to `acquire()` for every other case, and skips
  the whole-routed-set promote/stage batches when nothing is going to be copied.
  `KRK_EXPERT_INPLACE=0` is the A/B arm.
  Measured on REAP-288 at the shape a small machine has (`--expert-cache-mb 8000
  --expert-warm-mb 20000`), 32 greedy tokens, stdout bit-identical
  (`da9ba027a3454bba9edfefdb6c049632` both arms): WARM->VRAM copied **18203 MiB
  -> 58 MiB**, VRAM evictions **5839 -> 23**, alloc **669 -> 1.3 ms**, promote
  **775 -> 1.9 ms**, expert path **0.79 -> 0.19 ms/layer**. Step time is a WASH
  on this box (147.0 against 149.3 ms) and that is the honest reading: the bytes
  drop 300x and the time does not, because on a 96 GiB machine whose page cache
  already holds the file the copy was never the constraint. What it buys is the
  VRAM FOOTPRINT -- 7 GiB of experts now serve the same text where the promoting
  path needed the routed set to stop thrashing.
  **The probe's first version "proved" the opposite and the bug was in the
  probe**, which is the part worth keeping: 128 threads summed into a shared
  `out[tid & 1023]`, so the total came out at exactly 1/128 of the truth and read
  as "the pinned buffer is not mapped". A correctness check that cannot tell a
  broken read from a broken check is worth less than no check -- give every
  thread its own slot.
  **REAP-288 has a floor near 16 tok/s whatever the tier does**, and the number
  is arithmetic: one token visits 480 experts of ~3.1 MiB, i.e. ~1.5 GiB of
  weights, which is 62 ms/token at 24.3 GiB/s before any compute. 20-30 tok/s is
  an A3B-class target (8 x ~1.5 MiB per layer, ~250 MiB/token) -- what
  DeepSeek-V4-Flash and Qwen-3.8-Flash route -- not a REAP-288 one. The remaining
  host cost there was measured at 38 ms/step of BLOCKING file reads (274
  MiB/token) plus 22 ms of host selection and plan, serial with the GPU.
- **The routed-set reads now run on a worker thread, and both failed versions
  are worth keeping.** `ExpertCache::prefetch_warm_next()` + `take_prefetched()`
  (`KRK_WARM_AHEAD=0` is the arm, default on, requires in-place to be on): the
  layer's ids are handed to a worker at selection time, the worker copies the
  missing experts out of the mapping into pinned buffers, and the per-expert
  loop installs each as the expert's WARM copy when it reaches it -- waiting only
  for the ONE expert in flight, so the layer's ascending expert order (and every
  bit of its arithmetic) is untouched. Measured 24 greedy tokens, 4 interleaved
  order-balanced runs, stdout identical (`c0627e2a...`): 152.4/153.2 ms ->
  **129.6/128.3 ms per step, 6.5 -> 7.7-7.8 tok/s**, `moe.route.fetch` 63 -> 31.6
  ms (the GPU backlog halves because the host stops blocking) and 295-330
  expert reads (937-1042 MiB) taken off the host. At a 4 GiB expert budget
  167.8 (promote) -> 157.7 (in place) -> **136.2 ms** (7.3 tok/s).
  (1) **The first version issued ZERO reads**: it was fed the *next* layer's ids
  from `moe_hint_`, the previous token's routing -- and those experts are already
  in WARM by definition, because the previous token just used them. The debug
  line said `hint=10 ids=0` on every layer. What blocks is the expert this layer
  routed that the tier does NOT hold, and those ids are exact and already on the
  host after selection.
  (2) **The second version was 11% SLOWER**: `moe_ffn` already reads the routed
  set through `prefetch_layer()` before the loop, so the worker read the same
  bytes a second time and `take_prefetched()` correctly discarded its result.
  The synchronous pass is skipped when the worker is reading. A "prefetch" that
  duplicates the read it was meant to hide is slower than no prefetch, and the
  only visible symptom is a step that got worse for no reason.
- **In-place must yield to REUSE, and the threshold is `count >= 2`.** The first
  rule refused in-place only for experts that had earned a pin (24 touches), so
  on Qwen3-30B-A3B -- routed set ~9.6 GiB, 128 experts top-8, 84.7% of picks
  repeats -- 45.7% of acquires read 24.3 GiB/s host memory where VRAM already
  held the expert: **42.9 ms/step against 33.7 for the promoting path, -20%**.
  The tier's own LFU counter is the signal: one in-place read pays for itself
  (it is the read the promotion would have copied), the second use is where the
  card wins by 20x. With `count >= 2` the A3B file is 34.2 ms/step (-2% against
  promote, 4x fewer COLD misses) and REAP-288 is 127.9-130.2 ms against
  138.5-139.3 (+7.7%). Measured at the goal shape, 48 greedy tokens:
  Qwen3-30B-A3B reaches **29.2-29.9 tok/s at a 4000 MiB expert budget with an
  8000 MiB RAM tier, 31.2 tok/s and 0.0% COLD at 12000 MiB, 31.6 tok/s at
  6000/8000** -- the 20-30 tok/s target, met on a card budget a 6-8 GiB card
  has. The promoting path reaches the same band there, because when the routed
  set FITS a promotion is a one-time cost; the three tiers win on the file whose
  routed set does not fit (REAP-288) and on the VRAM footprint everywhere.
  **The page-cache caveat**: a COLD miss on this 96 GiB box is a ~7 GiB/s
  page-cache read; on a 16-32 GiB machine it reaches the disk. The 12000 MiB row
  with 0.0% COLD is the configuration where the disk is not in the loop at all.
  **The knee is between a 6000 and an 8000 MiB RAM tier, and it is a cliff.**
  Swept on Qwen3-30B-A3B at a 4000 MiB expert budget, 48 greedy tokens
  (`docs/PERF-ANALYSIS.md` section 12d): 3000 MiB 23.5 tok/s (14.2% COLD, 98.9
  MiB/token), 4000 23.7 (13.0%, 90.2), 6000 27.2 (4.4%, 30.6), 8000 **29.4**
  (1.1%, 7.9), 12000 **31.7** (0.0%, 0). The worker hides the read inside a
  layer's ~0.66 ms of compute, so what matters is bytes per LAYER, not per
  token: 0.16 MiB/layer is free at any drive rate, 0.64 needs four readers to
  stay hidden, 1.9 does not hide at all. A 16 GiB machine's 6-8 GiB tier sits on
  the good side at 8 GiB (disk irrelevant) and on the edge at 6 GiB (17-27 tok/s
  depending on the drive).
  **This box cannot measure the disk, and the doc says so**: 40 random 1.5 MiB
  reads from the 78 GiB REAP file ran at a median 1809 MiB/s because 96 GiB of
  RAM holds most of it. The projections use the recorded NVMe rates (593 and
  1446 MiB/s) instead, and are labelled as projections.

- **The pool's reuse guard had exactly one hole, and it is the pinned-DMA arm.**
  A released cell goes back on the free list carrying the mark its readers were
  released under, and `alloc_pooled` hands the next owner that same mark. Every
  write path into a cell waits for it on the stream it writes with --
  `upload_paged` (ring and direct), `upload_expert_staged`,
  `upload_expert_prefetch`, `upload_i32`, `fill0`, `copy_act` -- except
  `upload_pinned_batch`, which copies on a private `hipStreamNonBlocking` stream
  and never did. A promotion could therefore overwrite a cell a queued expert
  GEMM was still reading: no crash, a plausible number from the wrong matrix,
  and under heavy eviction a different sentence. It is one
  `pool_honor_(dst[i], pinned_stream_)` now.
  **The trap is `kraken-bench --pool`, and it is in `--gate` too.** 16 copies of
  one 64 MiB cell queued on the default stream, the cell released, handed back
  out, and a different pattern DMA'd into it; every reader must still see its own
  bytes. Red construction: without the guard it FAILs 3 of 3, with it 3 of 3 ok.
  `--pool` exists because `--gate` also drives a CLI speculation arm.
  **Depth is the trap, and the first version missed it**: 256 reads of a 4 MiB
  cell PASSED on the buggy code, because the host enqueue outran the device work
  and the queue was empty by the time the DMA was posted. A race trap whose
  reader queue drains before the writer is posted proves nothing and looks
  exactly like a clean bill of health.
  **What correctness costs, REAP-288** (`--ctx 512 --chunk 256`, 32 greedy
  tokens, guard on against off interleaved, text identical in every pair): at the
  default auto budget (~8.4 GiB here) `reuse 0`, no pending marks, nothing to
  pay; at `--expert-cache-mb 4000` (~3900-4060 reuses) **147.6 -> 159.4 ms/step,
  +8.0%, slower in 4 of 4 pairs**; on section 12i's refuse-instead-of-trim arm
  (~55 000 reuses) **146.4-147.1 -> 219.1-220.7, +50%**. At a saturated budget
  the cell handed out IS the one the eviction has just released, so the copy
  waits for the current layer's own GEMMs -- and the overlap that arm was buying
  was reading garbage. Recovering it honestly is promoting ahead of demand, not
  dropping the guard.
  **12i's sentence divergence is this bug**: at 4000 MiB on that arm, 3 runs with
  the guard removed gave two identical continuations and one different one
  (`16ab8aff...` against `b3eb4d07...`); 4 of 4 matched with the guard. Numbers
  and the trap: `docs/PERF-ANALYSIS.md` section 12k.
  **Promoting ahead does not recover it, and the cache's counters prove the pass
  had nothing to do**: `KRK_MOE_NEXT_PREFETCH=1` (previous token's routing ->
  `stage_ahead`) and `KRK_MOE_PREDICT=1` both leave HOT hits at 46.9% and WARM
  hits flat (2699 -> 2684) with the step at 179.2 against 178.8 ms, and the new
  `[stats ] lookahead <n> staged, <m> rejected, <k> consumed` line -- printed
  only when non-zero -- never appears: a hinted expert is either already in VRAM
  or no longer in WARM a whole token later, so there is no candidate. Details:
  `docs/PERF-ANALYSIS.md` section 12m.
  **Ranking the victim instead does not recover it either, and the mark explains
  why**: ordering `force_reclaim`'s same-class candidates like `pick_victim`
  (rarest, then oldest load) and ordering them by a new true least-recently-read
  stamp both leave the step at 180.5/183.1 and 181.9/183.6 against 180.2/181.7
  ms/step. A release is recorded as a POSITION on the default stream, so a reuse
  waits for every kernel queued before that point -- the current layer's own
  earlier experts, not the victim's readers -- and the victim's age is
  uncorrelated with it. At this shape the guard costs **+24%** (144-150 ->
  180-183 ms/step), not 8%. Both rankings were reverted; the only direction left
  is a per-slot reader-done event recorded at the end of that slot's own use.
  `docs/PERF-ANALYSIS.md` section 12n.
  **That per-slot mark is implemented and it is worth 13%**: `note_read(layer,
  expert)` records ONE event for a slot's three cells at the end of that use
  (`Backend::pool_note_reader_done`), the cell's next release ADOPTS it instead
  of recording the default-stream position, and a hand-out drops any stale stamp
  so an unstamped read path falls back to the old mark rather than waiting on too
  little. REAP-288 at 4000 MiB: **181.3 -> 158.3 ms/step median, faster in 8 of 8
  interleaved order-balanced pairs**, text identical, which takes the guard from
  +24% to +8% against the racy arm. ~400 records/token (~1.2 ms), so it is gated
  on `release_slices > 0`: with the auto budget (`reuse 0`) the first version
  stamped 410 times a token for nothing and the counter is `reader-done 0` now.
  `KRK_POOL_SLOT_MARK=0` is the A/B arm; the counter is on the `[stats ]
  expert-pool` line. `docs/PERF-ANALYSIS.md` section 12o.
  **Picking a victim whose readers have retired is NEUTRAL, so the last 8% is
  not the victim choice.** With the mark in place, `force_reclaim` preferring a
  slot whose reader-done mark predates the latest drain (a counter bumped by each
  blocking D2H and by `sync()`, so "are this slot's readers done?" is a field
  compare instead of the ~50 us `hipEventQuery`) gives **157.4 against 157.2
  ms/step median, faster in 1 of 3 interleaved pairs**, with `reader-done` at
  12.65-12.73k in both arms: the pick moves which cell is reused and nothing
  else. Reverted. The reason is 12o's own result: a reused cell now waits for
  exactly its readers, and a reclaim's candidates are almost always slots whose
  last read was a token or more ago, so those marks are complete under either
  rule. What is left is the ~1.2 ms of records plus the slot read inside the
  current token, which preferring another victim just moves elsewhere.
  `docs/PERF-ANALYSIS.md` section 12p.
  **And the arm nearly recorded the opposite for a harness reason**: it was run
  as `run KRK_EVICT_RETIRED=0` where `run` is a shell function, so `"$@"` put the
  assignment on kraken's ARGV -- every treatment run died with `unknown option`,
  all three identically, so their output md5s matched and it read as "same text,
  safe". A *missing* metric (`ms/step` absent) is the tell; `env VAR=0 cmd` is
  the form.
  **One mark per burst of uses is the default now, and a record costs ~0.1 us,
  not the ~2.5 us 12o assumed.** `note_read` queues the slot's cells and emits one
  `pool_note_reader_done` per `KRK_POOL_MARK_BURST` uses (default **8** = one per
  layer at top-k 8; `=1` is 12o's arm): REAP-288 at 4000 MiB **151.9 -> 150.9
  ms/step median over 7 interleaved pairs, faster in 6**, records 12.7k -> 2.2k
  over 32 tokens, text identical; Qwen3-30B-A3B is a wash (36.1/34.5 against
  35.8/35.7) because the guard's wait is a smaller share of a 35 ms step. Three
  rules make it safe: a later mark on the same stream is conservative (only the
  waited-for position moves), the burst never spans a layer and is flushed at the
  end of the expert section before the next-layer prefetch can trim or free a
  cell, and `drop()` scrubs released cells from the pending set -- otherwise the
  flush could stamp a pointer that is already back on the free list. The
  arithmetic that matters: 328 fewer records/token bought ~0.03 ms/token, so the
  records were never the constraint -- the wait is.
  `docs/PERF-ANALYSIS.md` section 12q.
  **The rest of the guard is stream order, not host stall, and the stats line now
  says so.** `pool_honor_host_` times its `hipEventSynchronize` into
  `honor_host_ms` and the `[stats ] expert-pool` line prints
  `honour device/host/implicit ... host-wait <ms>`; on REAP-288 at 4000 MiB that
  reads **device ~12.5k, host 0, implicit 0, host-wait 0.000 ms**, and the guard
  measured against `KRK_POOL_NO_WAIT=1` (a deliberately racy measurement-only arm
  that skips the wait and keeps the bookkeeping) is **152.1 against 149.4 ms/step
  median, +2.7 ms, slower in 5 of 7 pairs**. The chain is
  `hipStreamWaitEvent(pinned_stream_, mark)` -> DMA -> `hipStreamWaitEvent(0,
  pinned_ev_)`, so the cost is the copy starting late, i.e. promotion latency.
  ~390 honours a token means ~220 ns each: most marks are long complete. The only
  fix left is supply-side (a reserve of free cells, or promoting ahead, which 12m
  showed has no candidates) -- refining the wait has now been measured three times
  (12n victim ranking, 12p retired victims, 12q records) and is exhausted.
  `docs/PERF-ANALYSIS.md` section 12r.
  **The prefill gap against Strata is a dispatch trade, and the matrix cores are
  one flag away.** `exact_gemm_enabled()` routes every rows>1 GEMM to
  `gemv_grid_kernel` (the chunk-invariant exact grid) BEFORE the WMMA rung, so
  auto is 13x slower on the GEMM by design (`gemm(grid)` 716.97 device units at
  5.4 efficiency against `gemm(wmma)` 53.02 at 80.5 on the same 252 calls).
  `--prefill fast` sets `KRK_GEMM_PATH=wmma`: **307.0 -> 1707.0 tok/s** prefill on
  Qwen3-4B-Q8_0 (5.56x, 1145 tokens) and **170.8 -> 325.5** on the 30B-A3B (the
  expert path stays exact, so it gains less), and it is a different-arithmetic
  mode: the 4-token text changes (`dbc606bb39b8` -> `b9d995daaa3a`) and the
  prefill hash is identical at `--chunk 1`/`8` (the thin-row ladder agrees with
  the grid) but differs from `--chunk 16` (`b3222295` -> `a48ba8d6`). Strata's own
  `STRATA_HIP_WMMA=1` makes the same trade; the difference is the default.
  `KRK_GRID_G` does not recover it (306.9-309.4 across 32-256). A matrix-core rung
  whose accumulation order ignores the row count is the one route that keeps the
  invariant AND the speed. `docs/PERF-ANALYSIS.md` section 12s.

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
- **An expert WARM tier shorter than the routed set is churn, and the auto policy
  no longer picks one.** Measured on the 78 GiB REAP-288 file (42.2 GiB routed
  set, 7.2 GiB device budget): the old policy took half the free room (32.5 GiB)
  for WARM and pinned a quarter of RAM (16 GiB), which gave 41.7% WARM hits,
  **11.1% COLD (disk) misses and 190 MiB read from the file per token** --
  6.4 tok/s. Sizing WARM to the corpus (42.4 GiB) and pinning the whole tier
  gave 124-190 cold misses over 31 steps and **10.0 tok/s**, text unchanged in
  every arm (2 interleaved order-balanced pairs; old policy forced with
  `--expert-warm-mb 32571 --expert-pin-mb 16384`). The pin is a PREFIX in rank
  order, so a partial one buys a partial speedup and the curve is monotone:
  16 GiB 139.4 ms/step, 24 GiB 121.8, 42.4 GiB 103.1. Both rules are in code
  (`Engine::init`'s auto WARM and `pinned_warm_prefix_bytes`, cap = half of
  installed RAM) because flags cannot fix a default.
  **The default then changed again on the 12d sweep, because the arithmetic
  double-charged the pin.** `krk::warm_tier_bytes` / `krk::host_budget_bytes`
  (`include/krk/vram_guard.hpp`) are now the whole policy: the corpus, capped
  by 60% of installed RAM (the 80% ceiling minus the 20% reserve) or by free
  RAM minus that reserve when smaller. Written out: 16 GiB installed -> 9.6 GiB
  of room, 24 -> 14.4, 32 -> 19.2, 96 -> 57.6. The rule it replaced subtracted
  a working-set headroom (`installed/5`) AND a pin allowance (`installed/16`,
  capped 6 GiB) AND a 3 GiB floor on top of the reserve -- and the pin is *part
  of* the tier, so those bytes were charged twice -- then halved the answer
  again whenever the corpus exceeded the room. On a 16 GiB machine that is
  8.8 GiB, halved to 4.4: the 3000-4000 MiB row of the sweep, 23.5 tok/s where
  9.6 GiB of room was sitting there. So `--expert-warm-mb` is now a CAP on a
  shared box, not a requirement: `--ram-tier 16` on A3B prints `9.6 GiB budget
  ... -> WARM tier 9830 MiB [planned for the --ram-tier class]`, this box's
  default is the corpus-limited 10008 MiB, and REAP-288 is unchanged here
  (43200, its corpus) while `--ram-tier 16` gives it the same 9830. The sample
  fails closed (`0`, no tier) rather than assuming an unmeasurable host.
  Defaults against explicit caps, rotation-interleaved (the starting arm moves
  each run; AGENTS' own rule), texts identical in every arm: A3B at a 4000 MiB
  device budget, 48 tokens -- auto 10008 **36.0-36.1 tok/s**, 8000 34.9-35.3,
  6000 33.7-33.9, 4400 **30.2-30.5** (4400 is what the old arithmetic picked, so
  the default is worth +19.5%). REAP-288, 16 tokens -- auto 43200 **10.5-10.8**,
  20000 5.7-7.8, 12000 6.8-7.1: **20000 LOSES to 12000**, because a tier between
  the set and the room misses forever and the larger one pays a bigger pinned
  fill first. That is why the auto answer is the corpus, not a fraction of it.
  **It warns when it lands under the knee, once, naming the tier and the
  projected read.** `krk::warm_miss_mib_per_layer` (vram_guard.hpp) is the 12d
  sweep as arithmetic, `(corpus-warm)^2/corpus/50` per token over n_layer -- a
  FIT, good to ~25%, so nothing is planned from it -- and
  `warm_tier_under_knee` is the 0.5 MiB/layer line between the 6000 arm (0.64,
  hidden only at the four-reader rate) and the 8000 one (0.16, hidden
  everywhere). Only the AUTO path warns: an explicit `--expert-warm-mb` is a
  decision, not a surprise. Measured with 50 GiB held by another process (33.5
  GiB free of 95.9): REAP planned **14650 MiB instead of 43200**, printed
  `... 34%, under the measured knee ... projected 7.86 MiB of file reads per
  layer (377 MiB per token at 48 layers)`, and decoded 2.5 tok/s over 8 cold
  tokens. So the auto tier DOES under-size on a busy box, by design -- it is
  free-minus-reserve at that moment, never past it -- and the difference now is
  that it says so. A3B on an idle 96 GiB box stays corpus-capped (10008) while
  free RAM is above ~28.8 GiB. **The model file's own page cache does NOT
  shrink the next run's tier, and that was measured rather than assumed**:
  reading 32 GiB of the 78 GiB REAP file leaves Windows `ullAvailPhys` at
  **82.9 GiB against 83.0 before** -- standby (file cache) counts as available --
  and Linux's `MemAvailable` includes reclaimable page cache by construction.
  The plan does not subtract the mapping either: `mapped_share` is printed as an
  info line and appears nowhere else in `configure_expert_cache`. So the
  43200 -> 14650 MiB shrink measured with 50 GiB held came from genuine
  committed memory in another process, not from the file's cached pages. A
  "measure the page cache separately" change would have to invent a correction
  for a number that is already right, i.e. plan a tier past what is free.
  **The routing readback is a DEPENDENCY DRAIN, not overhead, and that was
  measured rather than argued** (2026-10-22, build with in-place + warm-ahead,
  REAP-288, 48 layers, ctx 512). `KRK_TIME_HOST=1`: a 199 ms/step decode is
  `moe.experts` 117.8, `moe.route.fetch` 47.8 (0.996 ms x 48), `moe.route.gemm`
  0.2, and the host top-k + plan **1.8 ms/step** -- the 47.4 ms this bullet used
  to attribute to "host top-k and plan" does not reproduce; the plan is 1.8.
  Three numbers pin what the 47.8 IS:
  (1) `KRK_ROUTER_2X=1`'s second copy, with the queue retired, is **95 us/layer**
  -- so 0.9 of the 1.0 ms is not the copy;
  (2) a pinned destination with `hipMemcpyAsync` + a spin on the recorded event
  (three arms, 4 rotated pairs, text identical) changed `download_f32`'s host
  time not at all (3264 against 2997 ms over a run) and the step not at all
  (145.9/129.8/141.0, 139.4/141.0/139.1, 112.1/114.7/114.8, 113.6/113.3/121.6
  ms/step for pageable-blocking / pinned-blocking / pinned-spin) -- so it is not
  a driver or scheduler round trip either;
  (3) A3B reads back in **0.477 ms/layer** at the same 48 layers against REAP's
  **1.031** -- it scales with the layer's own pre-MoE GPU work.
  So the host block IS the attention finishing, and the only way to delete it is
  to take the host out of the routing loop: device selection + device dispatch,
  which needs the routed set resident (42.2 GiB against a 9.2 GiB budget here).
  `KRK_DEV_PLAN=1` is gated off by `ne <= 256` (this file has 288) and verifies
  itself against the host anyway, so lifting the bound alone is not a speedup.
  Do not re-derive this: docs/PERF-ANALYSIS.md 12g has the table.
  **What is left on this configuration is the expert path**, not the router, and
  TWO of the numbers this bullet first quoted were cold-fill artifacts -- do not
  plan from a 16-token run (PERF-ANALYSIS 12i/12j):
  (a) `alloc 4302 reuse 0` is NOT a reuse bug. A 16-token run never evicts
  (`release 0 mark(s) for 0 slice(s)`), so nothing was freed and nothing could be
  reused; at 128 tokens reuse is 25243. The real churn is the TRIM: at a
  saturated budget a size-class miss hipFrees free cells of the other class and
  hipMallocs the wanted one, and an expert is two classes (gate/up, down), so
  ~100 pairs a step at ~190 us = **15.7 ms/step** (1988 ms over 128 tokens).
  (b) The in-place read is **0.87 ms/step** in steady state, not 46: the 46 was
  the cold-fill window. Sweeping the VRAM budget 3000 -> 8000 MiB buys -14.6% on
  the step (136.7 -> 116.7 ms) and almost none of it is the in-place share
  (23.8% -> 22.2%) -- extra residency converts WARM hits into HOT hits, i.e. it
  removes PROMOTIONS (2988-3561 -> 2040-2351 ms).
  **The trim churn has a principled fix and it was REJECTED: refusing instead of
  trimming (return nullptr, let `force_reclaim()` evict a victim holding a slice
  of the WANTED class) drops allocations 6601 -> 4219 and trims 2380 -> 4 and
  halves the alloc row (1085.8 -> 506.9 ms for 64 tokens) -- and the step is
  unchanged (103.6 against 103.9 ms median) while **stdout loses determinism**
  (3 of 4 runs a different sentence at 64 tokens; 4/4 identical before). That is
  the useful part of the result: under heavy eviction the cache can retire a slot
  whose cells are still being written, which the trim path masks by never evicting
  this often. Fix that race before touching the trim policy. Steady state at
  8000 MiB, 114.4 ms/step: routing 89.2 (fetch 37.1 + expert loop 50.2, of which
  alloc 15.7 and promote 17.4) and everything else ~25. Full trace: PERF-ANALYSIS
  11, 12g, 12i and 12j.
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
- **On gfx1201 the prefill GEMM that RUNS is `gemm(dp4a)`, not the WMMA tile, and
  it was emulating its dot products until this was fixed.** The profiler's own
  tag is the witness: `--profile` rows read `gemm(dp4a)`, while
  `KRK_GEMM_PATH=wmma` is 2x SLOWER on the same shape (4096x4096x466 Q4_K:
  ~1.55 ms against ~3.1 ms), so the auto rung is right to prefer int8. Two
  defects made that path cost far more than it should:
  (1) `KRK_HAS_DOT4` was guarded to gfx10/gfx11, so on gfx12 `d_dot4` compiled to
  the **16-op scalar emulation** (8 shifts, 4 multiplies, 4 adds) instead of one
  `v_dot4_i32_i8`. The guard existed because clang refuses
  `__builtin_amdgcn_sdot4` for gfx1201 ("needs target feature dot1-insts").
  **Do not fix that by enabling the feature:**
  `-Xclang -target-feature -Xclang +dot1-insts` appends the feature to the
  target ID embedded in the code object, the fat binary then no longer matches
  the runtime's gfx1201, and every launch dies with `Cannot find Symbol ...
  copy_h2h_kernel` (measured; build rc=0, binary broken). Inline asm
  `v_dot4_i32_i8 %0, %1, %2, %3` assembles for gfx1201 with no flag and changes
  no target ID -- that is what ships, and it is worth **1.50x** on the GEMM by
  itself (466 rows: 1470.6 -> 979.9 us, 3/3 pairs).
  (2) `dp4a_cfg::TM` was 4, i.e. 12 LDS loads per 32 dots. **TM=8** makes it 16
  loads per 64 dots and is worth **1.57x** end to end on the GEMM (1536.6 ->
  979.9 us at 466 rows; 8315 -> 6894 us at 4096 rows, both interleaved 3 pairs,
  order balanced), with the harness check unchanged (`chk=1.06e-02/1e-02`).
  Together, on **Qwen3-8B Q4_K_M at a 470-token prompt: 566.5 -> 633.2 tok/s
  prefill (+11.8%, 3/3 pairs, no overlap)** against the recorded 561 baseline.
  End-to-end is smaller than the GEMM's 1.57x because the projections are ~80%
  of device time and attention is the next 13% (`--profile` component table),
  so the remaining headroom is inside the same kernel: 15-20 TFLOP/s against
  llama.cpp's ~68 on the same model means the tile is still ~4x off, and the
  lever there is more MACs per LDS load (register tile / `global_load_lds`),
  not instruction selection.
  `kraken-i8k` does not build on this box (it is compiled as plain CXX and its
  `#include <hip/hip_runtime.h>` has no include path); the suite's int8 groups
  (3583/3583) are the guard for the dot path.
- **The WMMA tile's shape is not its lever, and that is now measured.**
  `gemm_wmma_kernel` stages both operands into LDS as **f16**, so a block stages
  `BN` rows of weight at 2 B/value and `BM` rows of activation; per output FLOP
  that is `2/BM` bytes of weight against `2/BN` of activation, and the weight
  side is re-staged once per M tile (`row_base = blockIdx.y * BM`), i.e. 15x at
  a 466-row prompt with `BM=32`. The obvious fix is a wider tile, and it fails:
  **BN 64 -> 128 measured 46% SLOWER** on `kraken-bench --rows 466` (Q4_K
  4096x4096: 1105 -> 1613 us, 14.15 -> 9.69 TFLOP/s) and destroyed the small
  shape (rows=22: 18.1 -> 1.16 TFLOP/s, because 4096/128 = 32 blocks cannot fill
  64 CUs). BN=128 needs 43.5 kB of LDS, which drops a block per CU from two to
  one -- the staged bytes are not the binding constraint, the staging CHAIN is
  latency-bound and occupancy is what hides it. `BM=128` buys the same byte cut
  by spending the same occupancy, so it is predicted to lose the same way and
  was not shipped. What does work is staging **fewer bytes per chain**: the
  quantized bytes (0.5 B/value) unpacked in registers at MMA time, which is the
  int8/MMQ rewrite, not a tile tweak. Baseline to beat: 1105-1138 us /
  13.7-14.2 TFLOP/s at rows=466, 340 us at rows=22.
- **The prefill GEMM's staging phase had ONE unprefetched operand, and
  prefetching it is worth 11-22% of prefill.** The Q4_K weight staging already
  keeps round kt and kt+1's raw bytes in registers, so the weight side's load
  latency is hidden under the MMAs; the **activation tile had no such
  prefetch** -- its 8 loads per thread were issued and consumed by its own LDS
  stores inside one phase, in front of the barrier, with nothing to overlap
  them. At 52 kB of LDS the block gets one slot per CU (8 warps over 4 SIMDs, 2
  warps per SIMD), so both warps of a SIMD stall on the same load together;
  `tools/gemm_res.hip` reports the shipped instantiation at `LDS Size
  [bytes/block]: 52224`, 143 VGPRs, 0 spills. `gemm_wmma_kernel` now holds this
  thread's activation rows in registers for one round (`a_prefetch(kt+1)`
  inside round kt's staging phase, `a_publish()` at the top of round kt+1), so
  the loads are in flight across the weight decode, the barrier and the MMAs.
  Interleaved medians of 3, and the two arms are two BINARIES -- an env switch
  keeps both code paths alive in one binary and differs in register allocation
  too: Qwen3-8B-Q4_K_M 501 tok **830.9 -> 1013.7 tok/s (+22.0%)**,
  Qwen2.5-Coder-7B-Q5_K_M **881.1 -> 1073.0 (+21.8%)**, bitnet-2b-tq1_0
  **1577.6 -> 1752.6 (+11.1%)**. In the op table the GEMM itself goes
  375.14 -> 258.86 ms (**-31.0%**, 252 calls) and prefill wall 570.5 -> 459.9 ms.
  Decode is untouched (40 tokens, 77.0-77.3 tok/s both arms: it runs the
  `gemv_m` ladder, not the tile). Identity: `--bitexact`'s verdict lines are
  identical (every gemm/group+silu row at rows 2..64 IDENTICAL), greedy 24
  tokens are byte-identical on all three files, and the coherence check passes
  on the dense, recurrent and MoE arms (`1 coherent, 0 not` each).
- **A full LDS double buffer of the shipped tile does not fit, and the
  arithmetic is the reason rather than the intuition.**
  `(MB*BM + BN) * (BK + LD_PAD) * 2 B` is 52224 B at the default (BM 64,
  BN 64, MB 2, BK 128, pad 8), so two buffer sets are 104448 B against a
  65536 B cap -- and the padding is not negotiable either: at `LD_PAD=0` both
  fragment loads put all 16 lanes of a half-warp in ONE bank (a 128-half row
  stride is 0 mod 32 words), a 16-way conflict. The variants that do fit are
  worse on measurement: `KRK_WMMA_NF=1 KRK_WMMA_MB=1` (26112 B, the 26 kB
  config that fits two blocks per CU) reads **696.4 tok/s** against the
  default's **816.6** at 501 tokens, `MB=2` 776.2, `MB=4` 770.1. So a second
  resident block is not what this kernel wants; fewer staging rounds per output
  row plus the NF=2 fragment pair is. Two traps from that sweep, both of which
  have already produced numbers in this repo: `KRK_WMMA_MB` is **DEAD** while
  `KRK_WMMA_NF=2` (the dispatch reads `wmma_cfg_n2::MB`), so an "MB=1 against
  the default" A/B with NF left at 2 measures one kernel twice and reports
  838.4 against 838.2 -- a real number, from two identical arms; and the
  staging-vs-MMA arms (`KRK_GEMM_STAGE_ONLY`, `KRK_GEMM_MMA_ONLY`) **compile to
  EMPTY kernels** (`LDS Size [bytes/block]: 0`, `VGPRs: 0` -- the staged tiles
  become dead stores, and the mma-only arm's fragment loads read LDS nothing
  writes), so their 124.5 / 123.4 ms against a 370.2 ms kernel is queue time,
  not a split. That claim is withdrawn in `docs/AUDIT-prefill-gemm.md`: a split
  arm has to keep its result alive to be a measurement, and the arms are now
  instantiated on the shipped `wmma_cfg_n2` (they used to describe `wmma_cfg`
  MB=1, a kernel 17% slower than the one the engine runs).
- **The model sweep's MoE rows are not a measure of a GEMM change.** Refreshing
  the 21-row sweep after the activation prefetch moved every DENSE row by +13 to
  +31% prefill (Samastam 3673.5 from 2877.2, Qwen3-4B 1342.6 from 1031.3,
  lacuna 621.1 from 520.1, Underdog 223.2 from 172.9; the short-prompt number
  the README quotes for the 8B is 638.5 from 561.2) while the two files whose
  rows stream 19 and 32 GiB of experts moved the other way (laguna-xs2 82.0 from
  92.8, Laguna-S 19.1 from 21.2) -- their prefill is expert bring-in, not the
  tile, and it is not stable run to run.
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

- **Upstream moved the expert cache's DECISION and its copies onto the card, and
  that is the gap this tier has.** llama.cpp #29887 (merged, `--moe-cache-mib`)
  caches host-resident experts in a GPU bank but still decides on the host; #30112
  extends it to layer-split; #29949 (`lru-upstream-v3`, `c81901a`) is the one to
  read -- `LRU_DECIDE`/`LRU_FILL` are graph ops, so the matmul takes SLOT ids and
  nothing is read back. At the same ~6.5 GiB pinned-host budget on one 7900 XTX
  that is **31.2 tok/s at 77.9% hit against 20.1 at 75.0%**, i.e. the difference
  is placement, not capacity. Here the device dispatch table exists
  (`include/krk/moe_table.hpp`) but is published ONCE and only while the whole
  routed set is resident, so the per-expert loop with a host round trip
  (`moe.route.fetch` 0.523 ms/layer on Mellum2.1, 0.077 of it the copy) is the
  default on every model that does not fit -- which is every model this cache
  exists for. **The bank is for ADDRESS STABILITY, not for alloc time, and the
  audit first said the opposite from a stale row.** The 256 MiB arm measured
  `alloc 2445.5 of 3408.1 ms` on 2026-10-09; today it is **`alloc 3.4 ms` of
  2849.9**, because the pool serves 96% of hand-outs from a free list (`alloc
  819 reuse 18231` over 6077 promotions, 0.56 us each). What that arm actually
  pays is the pageable H2D rate at promotion shape: 469 us/expert, 2.0 GB/s,
  **7.2 tok/s**, and pinning all of WARM takes it to **15.3 tok/s** with
  byte-identical stdout -- while making the pageable publish non-blocking
  changed nothing (2961 ms), which is what proves the rate is the DMA, not a
  host wait. Do not take their plain LRU (rejected here on measurement,
  `docs/AUDIT-expert-cache-next.md`), their per-layout-group budgets, or
  zero-slot substitution. Comparison, the arm table and per-step gates:
  `docs/AUDIT-gpu-resident-expert-cache.md`.

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

## The expert tier on a big file: two traps that made it do nothing

Both were found on the REAP-288/320 files (`qwen4exp`, 67-84 GB) and both are
recorded in `docs/AUDIT-REAP-DECODE.md` with the measurements.

- **`alloc_pooled` returns nullptr; using it as a pointer kills the context.**
  The pool refuses (does not abort) when `free - rounded < vram_floor_bytes`, and
  `ExpertCache::acquire`'s COLD path passed the answer straight to
  `upload(dst, ...)`. The result is `hipErrorInvalidValue` reported at
  `backend_hip.hip`'s `hipMemcpy` line -- which names the uploader, not the
  refusal -- and the run dies five tokens into a decode that worked for the first
  four. It is *not* rare on a constrained card: it is the steady state once the
  KV cache and workspaces have taken the room the budget was planned around
  (measured: budget 7624 MiB, 7445 resident, 3202 free beside a 3261 floor).
  The fix is `force_reclaim(want)`: retire this cache's own LRU, preferring a
  victim of the same size class (`alloc_pooled` matches cells by rounded size), and
  retry. A refused expert must return nullptr with **no** slot inserted and **no**
  `bytes_`/`vram_slots_`/`loads_` increment; the ledger used to count bytes for
  cells that were never allocated, which then justified evicting nothing.
- **WARM was disabled on every model bigger than RAM, silently.** The host policy
  subtracted the WHOLE mapped corpus from live free RAM
  (`live_avail - reserve - min(mapped, ram_total)`) and again from the ceiling.
  A clean mapping page is reclaimable standby memory, not owned RAM, so this
  double-counted the corpus and drove the staging room to zero exactly where the
  tier matters. The "tier is disabled" warning was gated on `host_budget > 0`,
  which was already 0 -- so the one case it described printed nothing.
  Measured on the REAP-320 file, 3 interleaved pairs, same text:
  **WARM off 206.8 ms/step (387 MiB read/token) against WARM on 141.3 ms/step
  (27 MiB/token) = +46%**, and on the 288 file 630.3 -> 283.0 ms/step (2.2x).
  The guard that replaces the pre-charge is `SharedHostRamBudget::can_admit`,
  which samples live available RAM per admission and fails closed.
- Two measurement notes from the same trace. (1) With WARM off, the COLD path's
  `upload()` is outside every timer in the expert-path row, so a run that moved
  387 MiB/token reported `read 0.0 ms | xfer 0.5 ms`; read the
  `[stats ] transfer ... read from the file` line instead. (2) `alloc_pooled` is
  `hipMalloc` per 2.2 MiB cell, ~200 us each: 656.7 ms of a 4.5 s constrained run
  and 28.8 ms of the 1.5%-miss one. An arena is the next win, not a follow-on
  nicety.
- **A refused promotion used to hand the caller an expert with null weights.**
  `acquire()`'s WARM-hit path fell through to the cold reload while the WARM slot
  stayed in the map; `emplace()` on an existing key is a no-op, so the function
  returned the OLD entry -- whose `ResidentExpert` is exactly what the failed
  promotion never built -- and leaked the cells the reload had allocated. Every
  GEMM then read null pointers, on the constrained-card path, which is the one
  this engine exists for. The stale slot is dropped before the reload and
  `!ins.second` is an explicit refusal now. `test_expert_promotion_failure` (9
  checks, inside `test_expert_cache_byte_budget`) pins the invariant; the
  end-to-end injection cannot force "promotion fails, cold succeeds" in one
  call, so the erase plus the refusal are what make it unreachable.
  **The other half of that path leaked the WARM copy, and only the in-proc
  runner can see it.** When the device refuses an expert, `warm_admit` has
  already paid for its pageable bytes; the early return dropped the local
  `Slot` on the floor, so those host slices stayed allocated and unreachable --
  no later `clear()` can free what is not in the map. Because
  `SharedHostRamBudget` is process-wide, the leftover capped every later
  admission in the same process. Measured: the fixture left exactly **24 bytes**
  (`ledger=24` from group 43 onward under `KRK_TEST_INPROC=1`), and groups
  `test_kv_tier_drop_vs_spill` and `test_kv_tier_host_reserve_refusal` then
  refused KV staging pages with "shared host-memory budget has no room" --
  3442/3458 in-proc against 3468/3468 in child mode, which is what the
  isolation check's `clean-inproc` and `per-group-a/b` arms reported. The slot
  is now inserted (WARM only: no `track_vram`, no byte charge) before the
  refusal returns, which also keeps the promise the comment above it makes --
  the next acquire of that expert is a promotion, not another file read.
  Same pass, same family: `clear()` never reset `host_pinned_bytes_`, so after a
  clear the pin prefix believed it was spent and no WARM admission was ever
  pinned again.
- **The size-class arena is opt-in too, and its numbers are the useful part.**
  `KRK_EXPERT_ARENA=1` makes the pool grow by 32-cell slabs of one size class
  from one `hipMalloc`. It does what it says: alloc row **150 -> 25 ms** over 16
  tokens (5.8x), reuse 0 -> 11 697 cells, trim 0. And decode is **13.8% slower**
  (114.2 -> 136.1 ms/step, 3 interleaved pairs, identical text), because the loss
  moves to `xfer` (894 -> 1 486 ms): uniformly spaced destination cells inside a
  slab are a copy/DRAM-locality problem, not a kernel problem. A 64 KiB pad made
  it worse (205 ms/step) by costing 3% of the tier, which is the cliff this pool
  sits on. The lesson that generalises: the alloc row is ~7% of a REAP step, so
  no allocator front-end can pay for a 20% copy regression -- measure the whole
  step, not the row you are optimising. `expert_arena()` now selects ONLY the
  growth allocator; whether released cells are cached is a separate decision and
  must stay one, or the bench's reuse checks fail the moment the slab is
  switched off.
- **Next-layer expert prefetch is opt-in and measured slower (`-9%`), and the
  reason is the lesson.** `KRK_MOE_NEXT_PREFETCH=1` records each layer's routed
  set for the next token and stages the next layer's predicted set after the
  current layer's expert loop, with `upload_expert_prefetch` (pinned DMA ticket
  on the copy stream, device-side wait at consumption, no default-stream barrier).
  REAP-320, 16 tokens, pinned WARM, 3 interleaved pairs: 114.4 ms/step off
  against 124.7 on. Promotion on that file is BANDWIDTH-bound (~240 x 2.2 MiB per
  token against a 42.2 GiB routed set and a 7.4 GiB device budget); the copy
  stream already overlaps the kernels queued before it, so issuing earlier only
  adds the promotions a prediction miss wastes. The build also exposed a real
  bug: `force_reclaim`'s same-size-class preference evicts the target layer's own
  experts (one layer, one set of size classes), which is why the first version was
  -15% -- `stage_ahead` now leases every slot of the layer it is completing.
  `KRK_VRAM_FLOOR_MB=1200` was measured too: more residency (55.7 -> 59.6% HOT)
  and slower (137 -> 155 ms/step), so the floor stays.
- **The pinned WARM prefix is automatic, and what decides it is a MEASUREMENT,
  not the BAR bit.** `Engine::init` times both promotion paths at the shape a
  promotion issues -- `max_expert_bytes()` clamped to 16-64 MiB, and the 16 MiB
  FLOOR is load-bearing: at 4 MiB on the REAP-320 file the same comparison
  reads 1.41-1.69x against 1.60x at 16, because the sync and the launch are a
  large share of each arm -- through `upload_paged_batch` (staging-ring memcpy
  then DMA) against `upload_pinned_batch` (DMA straight out of the caller's
  buffer), one untimed call per arm then interleaved order-balanced pairs with a
  MEDIAN per arm (three arms now; the path bullet below is where that form was
  forced by measurement), each ending in `sync()` because all three return
  before the copy lands -- and pins when pinned is >= 1.20x pageable
  (`krk::warm_pin_worthwhile`, `include/krk/vram_guard.hpp`). **The aperture is
  a FIRMWARE setting, not a card constant**: Above 4G decoding + Resizable BAR
  changes it, WDDM has been seen reporting the attribute as 0 on a box whose
  firmware had ReBAR on, and this engine's pageable path keeps its ring's
  memcpy even on a large aperture. So the runtime bit is only the fallback when
  no probe can run (`KRK_H2D_PROBE=0`), the log prints both rates
  (`promotion measured 10.1 GB/s pageable against 18.6 GB/s pinned (1.84x)` on
  the 35B-A3B here) and, when it pins, says in the same line that this is a
  firmware setting the user can change. Sizing is
  `krk::pinned_warm_prefix_bytes(warm, ram_total)`: a quarter of installed RAM
  (24 GiB on this 96 GiB box), capped at 16 GiB, and the tier is pinned WHOLE
  inside that bound -- a 16 GiB laptop with a 16 GiB tier pins 4 GiB.
  **The half-tier clamp was measured wrong and is gone:** it kept 4800 of the
  256 MiB arm's 9600 MiB pageable, and that half is where every miss lands. Same
  binary, `KRK_HOST_PIN_MB=4800` against the new auto, 3 order-balanced pairs,
  stdout identical in all 12 runs: **143.5 -> 67.2 ms/step (2.14x)**, xfer
  2815 -> 31 ms of enqueue. `--expert-pin-mb` / `KRK_HOST_PIN_MB` override it
  (including `0`), and `KRK_PIN_AUTO=0` is the A/B arm -- the log then names
  what was left on the table instead of going quiet.
  Measured cross-model, 16 greedy tokens,
  interleaved order-balanced pairs, text identical in every arm:
  REAP-320, three independent sets, 8/8 pairs favour the pin but the size runs
  101.7 -> 88.2 (+13.3%, 3/3), 113.3 -> 93.2 (+18.7%, 2/2) and 114.4 -> 96.7
  (+15.5%, one pair at +1.2%) -- quote the mechanism, not the headline, because
  the copy halves every time (us/expert 555 -> 400, 468 -> 390);
  Distill-35B-A3B Q2KXL with `--expert-warmup 1` 36.5/35.6 -> 30.9/31.1
  (+15.3/+12.7%, xfer 414/470 -> 199 us/expert); Qwen3-30B-A3B i1-Q2_K
  93.5 -> 91.5 (+2%, inside the spread -- its tier is disk-bound at 376 MiB
  read/token, so the copy the pin removes is 2% of its step); Qwen3-4B-Q8_0
  dense 9.3 -> 9.3 (0%, the control). **The gain tracks how much of a step a
  promotion is**, which is why the disk-bound and dense arms are recorded and
  not dropped. Both decisions are pure functions in `include/krk/vram_guard.hpp`
  precisely so `test_expert_cache_byte_budget` can drive them without a device:
  removing the half-the-tier cap fails 70/72 on exactly the two tier-bounded
  checks, and the ratio arm pins the 1.20x line (the 3x and 2x gaps measured
  here, and the near-1 ratios a large aperture would give, which must NOT pin).

- **Which pageable path a promotion takes is the same probe's OTHER answer, and
  it is measured too.** `Engine::init` times three arms at 16-64 MiB --
  `upload_paged_batch` with the staging ring, the same call with the driver
  staging the pageable source (`Backend::set_pageable_direct(true)`), and
  `upload_pinned_batch` -- and adopts direct only when
  `krk::pageable_direct_wins(ring, direct)` clears **1.20x**
  (`include/krk/vram_guard.hpp`). Below the line the ring stays; above it the
  direct path takes over, and then `warm_pin_worthwhile` sees a pageable rate
  near the pinned one and **declines the prefix** -- so a wide aperture should
  get the speed without spending RAM that cannot be swapped. That second half is
  the one thing this box cannot reach on real hardware (256 MiB aperture, no
  ReBAR) -- but it is no longer only asserted. The suite's backend decorator
  grows a simulated APERTURE: `set_test_aperture(ring, direct, pin)` makes each
  copy real and then pays a bandwidth-shaped wait, so `probe_h2d_path` reads
  three different rates off the clock and `test_pageable_path_probe` (38 checks)
  drives the whole policy on two apertures -- 0.4 against 20 GB/s must choose
  direct, 6.0 against 0.4 must keep the ring -- and then runs a real
  `Engine::init` on each. The margins are 1.5x, not 1.2x, on purpose: the wait
  is ADDITIVE to the copy, which compresses any ratio towards 1, so a tight
  margin would make the group report the test box's load as the aperture.
  The pin is deliberately NOT a fixed verdict there: on a wide aperture direct
  and pinned are the same copy, so that gap is a near-tie by construction, and
  the checks compare what the engine SET against the numbers it published. Both
  apertures must decode the same tokens -- the path is a performance choice,
  never a numerical one. `KRK_WARM_DMA=2` is still the hand-forced arm for a
  real card, but it bypasses this decision rather than standing in for it. `KRK_WARM_DMA=1|2` forces
  either path (the A/B arm), `=0` is still the blocking copy, and
  `KRK_H2D_PROBE=0` skips the probe (BAR hint only).
  **The probe is interleaved and order-balanced, and that was not cosmetic.**
  Run as three sequential arms on this 256 MiB box, the same pair read 13.9 GB/s
  (ring) then 10.5 (direct) at 12.6 MB, and 10.9 then 12.8 at 16 MiB in the
  other order: the ORDER was being read as the path, and it flipped the
  decision between two runs of the same binary. Now each iteration runs both
  pageable paths and swaps which goes first (pinned alternates head/tail), and
  each arm's value is the MEDIAN, not a best-of. One balanced sample read ring
  10.88 against direct 11.00 (a tie, so the ring stays), and 9.3-10.9 at 16 MiB,
  which is why the line is 1.20x and not 1.10x: 1.10 sits INSIDE the measured
  flap band (1.01-1.17x depending on copy size), so a 1.10 line would choose the
  path with the probe's buffer size rather than with the card.
  **But this box's 256 MiB aperture is not safely under the line either**, and
  the same binary read 8.7 against 10.6 GB/s (direct, 1.21x) in one Qwen3-30B-A3B
  run and 10.9 against 12.3 (ring, 1.13x) minutes later with
  `KRK_H2D_PROBE_RERUN=1`. So "it is a tie here, the ring stays" was one sample
  of that band rather than a property of the box, and the verdict is now CACHED
  instead of re-rolled: `h2d_probe_cache_read/write` (same header) keep it per
  (card name+gfx, vram_total, the runtime's `large_bar` hint, `driver_version`,
  probe size) in
  `%LOCALAPPDATA%\kraken\h2d-probe.cache` (`$XDG_CACHE_HOME` / `$HOME/.cache`
  elsewhere), 30-day expiry, `kH2dCacheVersion` bumped when the probe's own
  algorithm changes. **The write is where the file is PRUNED**, and that is
  not housekeeping: a reader already refuses an expired line, so leaving one
  on disk only denies a slot in a 32-line cap to a live verdict, and every
  driver update adds three. A rewrite drops (a) expired lines, (b) lines
  this build cannot parse, and (c) another driver's verdict once it is older
  than the 7-day rollback window (`kH2dCacheOtherDriverMaxAgeSec`) -- long
  enough to go back to the driver you just left, short enough that ten
  updates do not become thirty dead lines. Past the cap the NEWEST entries
  win, by stamp, not by position in the file. Measured end to end: three
  lines seeded at 8 days old under a fake driver were all gone after one
  run, leaving the current driver's line; a line from "yesterday's driver"
  survived a forced re-measure, which is the rollback the window exists for.
  **The rewrite REPORTS what it dropped, because a cache that quietly shrinks
  is indistinguishable from a corrupt one.** `H2dCachePrune` counts each reason
  separately (expired / other driver / unreadable / over the cap) plus what was
  kept, and `h2d_prune_text` renders it: `H2D cache <path>: dropped 3 dead
  verdict(s) (1 expired, 1 from another driver, 1 unreadable, 0 over the cap);
  3 kept`. It returns "" when `dropped() == 0` and both callers print only a
  non-empty string -- a clean rewrite is the NORMAL case (a 27B run rewrites its
  own line on each re-measure, `kraken-bench --h2d` rewrites all three), so
  `dropped 0` every time is noise that trains a reader to ignore the line. The
  counter is a struct filled by the writer, not a log inside the header: the
  header is included by the suite, which has no log, and a counter is what a
  test can assert on. Measured on the real file: three dead lines seeded (one
  40 days old, one under a foreign driver 8 days old, one truncated)
  `kraken-bench --h2d` printed `dropped 3 dead verdict(s) (1 expired, 1 from
  another driver, 1 unreadable, 0 over the cap); 3 kept`, and one expired line
  seeded before an olmoe run with `KRK_H2D_PROBE_RERUN=1` made `kraken.exe`
  print `H2D cache ...: dropped 1 dead verdict(s) (1 expired, ...); 3 kept`;
  the immediate rerun, with nothing dead left, printed no such line at all.
  The group pins all of it (69 checks): the stale line is readable before
  the write and unreadable after, the recent other-driver line survives, an
  expired line is counted out of the file, and a 38-key sweep leaves the
  newest 31 plus its own. Five of them are the reporting: one file seeded with
  exactly one dead line of each kind must come back as 1/1/1 with three kept,
  the rendered text must name the count and the reasons, an empty prune must
  render "", and the cap arm's write must report what the cap pushed out
  (neutralising one counter fails three of them, 66/69). Every field is a way a cached answer would be wrong: the
  BAR hint is in the key because a firmware change (Above 4G + ReBAR) is exactly
  the event that flips the answer; the **driver version** because the driver
  is what stages a pageable source (an update changes the staging, so it has
  to invalidate the verdict automatically instead of waiting out 30 days);
  and the size because the ratio is size-dependent (1.01x at 12.6 MB against
  1.17x at 16 MiB). `DeviceCaps::driver_version` comes from
  `hipDriverGetVersion` in `backend_hip.hip`, and the banner prints it (`HIP
  7.16.26331 | driver 71626331 | ...` -- on the ROCm 10 RT stack on this box
  the driver API version IS the runtime version, 7.16.26331 on both counts;
  a stack where they differ is what the separate field is for).
  `kH2dCacheVersion` is **2** since that field landed: a v1 line records no
  driver, so it is a miss and the next write replaces it -- a verdict whose
  driver is unknown is the one an update invalidates. Measured end to end by
  editing an entry's driver field: the next run said `measured now` and wrote
  a fresh entry, while a matching driver still read `from the cache, measured
  1 s ago`. The log line is the whole provenance -- age AND driver -- because a
  cached rate without its driver is one a reader has to open the file to
  interpret: `h2d_probe_origin()` formats it (`a 16 MiB copy from the cache,
  measured 18 min ago on driver 71626331 at 11.6 GB/s ...`, and `measured now on
  driver 71626331` for a fresh one), the text is pinned by four checks rather
  than left to an snprintf in the engine, and a backend that cannot answer says
  `(driver unknown)` instead of going quiet. `KRK_H2D_PROBE_RERUN=1` re-measures,
  `KRK_H2D_CACHE=0` disables
  the cache and `KRK_H2D_CACHE=<path>` relocates it. **The suite forces
  `KRK_H2D_CACHE=0` in `main`**, and the group that drives the cache points it
  at its own file and puts `0` back: without that, in-proc groups would share
  one file and a verdict cached for one simulated aperture would decide
  another's -- the cross-group poisoning the scratch area exists to prevent.
  The group's last arms are the wiring: a seeded wide-aperture entry on a
  simulated NARROW aperture must still choose direct (so only the cache can
  have decided it), `KRK_H2D_PROBE_RERUN=1` on the same seed must keep the ring
  (measured, not cached), and `KRK_H2D_PROBE=0` must use neither -- turning the
  probe off is not a request for an old measurement to stand in for it.
  **The probe size is bucketed, and that is what lets one tool fill the cache.**
  `h2d_probe_size_bytes` clamps to 16-64 MiB and rounds UP to a power of two, so
  the sizes a verdict can exist for are exactly {16, 32, 64} MiB. Without that,
  a model whose expert footprint is 20 MiB would look up an entry no tool ever
  writes, and `--h2d` could measure all it liked without a normal run finding
  it. Measured in one sitting on this card: 11.64 against 11.63 GB/s at 16 MiB,
  10.19 against 11.87 at 32 and 11.63 against 11.32 at 64 -- the answer really
  does move with the size, which is why the size stays in the key rather than
  being averaged away.
  `kraken-bench --h2d` prints the sequential shape table AND the balanced
  verdict, and is how a ReBAR-enabled box gets checked without loading a model.
  It never READS the cache -- measuring now is what that tool is for -- and it
  WRITES it: after the balanced pair it runs `probe_h2d_path` at the three
  bucket sizes and stores each verdict, printing `cached N verdict(s) for <card>
  in <path>`. So an explicit measurement is also how a BIOS change is picked up,
  without waiting for the 30-day expiry or knowing the env var. Its balanced
  line now uses `pageable_direct_wins` (1.20x) rather than its own 1.10x test:
  a tool printing "direct" at 1.10 while init kept the ring was reporting a
  verdict the run does not act on. `KRK_H2D_CACHE=0` makes `--h2d` measure and
  write nothing, and it says so.
  **`kraken-bench --h2d-cache` reads the cache back, and it needs no model and
  no measurement.** Same facts as `--h2d` from the other side: one line per
  entry with probe size, the three rates, the path `pageable_direct_wins`
  selects, the pin `warm_pin_worthwhile` would take, and `h2d_probe_origin`'s
  age+driver -- so a listing cannot disagree with the run it describes. Dead
  lines are listed WITH their reason rather than filtered out (`[expired]`,
  `[from an older cache format]`, `[rates unreadable]`, `[a failed measurement]`,
  `[stamped in the future]`) because "why is my run measuring again" is a
  question about the skipped lines; the reasons are classified in the same order
  as the writer's prune, so the listing and the rewrite cannot disagree. A live
  verdict from ANOTHER driver is kept by the writer for the rollback window and
  is shown as `[another driver]`, not as this run's. A missing file, an
  unreadable one and `KRK_H2D_CACHE=0` all list as empty (the tool still exits
  0), and the device is opened only to learn this box's key -- if it will not
  open, the listing drops the marking and says so, because a diagnostic that
  fails when the card is the suspect is useless.
  **`bytes == 0` in the key means "any probe size", and that is what a listing
  needs.** A probe size is a property of the model a run has not opened, so the
  listing asks the card's question (does a verdict exist here) across the three
  bucket sizes and marks none as used; a full key asks the sharper question and
  marks exactly the line `h2d_probe_cache_read` would return. Getting this wrong
  was visible at once: the first version passed the key with `bytes` unset, so
  every line mismatched on size and a cache holding three live verdicts printed
  `no live entry for this card at any probe size`. The group now pins both arms
  (a full key marks 1 of 2 live lines, the card's question marks the buckets and
  claims none), which is why it is 81 checks.
  **Every run that probes now states its verdict, and why it had none to
  reuse.** `H2D verdict: reused the cached 16 MiB one (from the cache, measured
  1 s ago on driver 71626331)` or `... measured the card now (16 MiB, 3 arms x 5
  iterations) -- this card's 16 MiB entry is not usable (expired), 1 dead of 1
  line(s)`. The wording lives in `h2d_verdict_text` / `h2d_cache_miss_text`, not
  in the engine, for the same reason `h2d_prune_text` does -- and the suite pins
  it, because "measured now" with no reason is indistinguishable from the run
  that found an empty cache when it actually found this card's verdict expired.
  The miss reasons are the ones a user can act on: no file yet / empty, the off
  switch (`KRK_H2D_CACHE=0`), this card's entry dead (`expired`, `unreadable`,
  `from an older cache format`, `a failed measurement`), `N live at other probe
  sizes`, `N live from another driver` (the rollback-window case, where the file
  looks populated and nothing is wrong with it), `N line(s) for another card`.
  **Two traps, both found by running it rather than by building it.** (1) The
  reason must be scanned BEFORE this run writes its own entry: scanned after, the
  file already holds a live verdict for exactly this key and the specific reason
  collapses to a generic "no verdict for this card" -- the first real run printed
  that on a freshly deleted cache. (2) A run whose probe cannot allocate its
  buffers has no rates at all, so it says `the probe produced no usable rates (an
  allocation failed)` rather than quoting a 0.0 GB/s measurement it never took.
  **The cross-run ledger is `<cache>.stats`, counted PER PROBE SIZE, and the
  end-of-run lines are `[stats ] h2d`.** One run's verdict is a hit rate of 0% or
  100% and says nothing, so `H2dProbeTally` (a vector of `H2dTallyBucket`, plus
  their sum in the flat fields) is persisted beside the cache and printed by
  `print_run_stats`: the total, then one row per size this machine has probed.
  Measured on olmoe with a seeded 32 MiB bucket beside its own 16 MiB one:
  `6 of 8 probed run(s) reused a cached verdict (75%), 2 measured the card at
  235.5 ms on average, so the cache has avoided about 1413 ms of probing` /
  `16 MiB: 2 of 3 reused (67%), 1 measured at 71.0 ms -> about 142 ms avoided` /
  `32 MiB: 4 of 5 reused (80%), 1 measured at 400.0 ms -> about 1600 ms avoided`.
  **Per size is the whole point, not a flourish.** The probe size is in the cache
  key, a 64 MiB verdict is not the one a 16 MiB model asks for, and the
  measurement it replaces costs several times as much -- so one blended rate
  describes no run in particular, and a machine whose only warmed model buckets
  to 64 MiB read as a mediocre cache instead of a cache never asked. A bucket that
  has avoided nothing prints no avoided figure (not `0 ms`), and an unasked size
  prints no row. The stored format is v2, one line per size
  (`2\t<bytes>\t<runs>\t<cached>\t<measured>\t<probe_ms>`), and a v1 blended line
  is not read as a per-size tally: the same rule the cache applies to its own v1
  verdicts, because the old shape literally cannot answer the new question.
  **The probe's own cost was a fixed 5 iterations per size, and the early stop
  is what removed the expensive half of it.** `iters` is now a CAP:
  `h2d_samples_settled` needs at least `kH2dProbeMinIters` (3) samples, relative
  MAD <= `kH2dProbeSettleSpread` (10%), and the median of the two NEWEST samples
  within 10% of the whole set's median. The third clause is not decoration:
  16 MiB's first three ring samples at MAD 6% were still 12% below where the
  median ended up. Measured with `kraken-bench --h2d-noise <MiB>` (three fresh
  processes; relative MAD at n=3): 64 MiB 1.8/0.4/3.9%, 32 MiB 1.1/3.2/12.3%
  (pinned still ramping), 16 MiB 6.0/0.4/6.4%. Cost, interleaved
  order-balanced pairs, 3/3: **64 MiB 112 -> 83 ms (-25.9%)**, 32 MiB 59 -> 43,
  16 MiB 31 -> 25, and the same path and pin in every arm. 16 MiB often still
  runs 5 -- the spread there does not settle, which is correct rather than a
  failure of the rule.
  **Two traps, both from doing the measurement.** (1) Measuring the three sizes
  in ONE process compares 16-cold against 64-warm: the first size pays the cold
  driver state, so the arm takes a size per invocation. (2) `--h2d`'s forced A/B
  has to mean EXACTLY n iterations -- forcing only the cap still stops early and
  measures the change against itself -- so `KRK_H2D_PROBE_ITERS=<n>` sets
  `no_early_stop` and refuses anything below the floor. The verdict line reports
  what actually ran (`3 arms x 3 iteration(s)`), not the cap.
  Three things are load-bearing about the counting:
  (1) **The store must not be gated on the reporter.** It shipped as
  `if (wrote) *wrote = h2d_tally_store(...)`, so the engine (which passes no
  pointer) counted in memory and wrote nothing: the in-memory tally said
  `runs=1` while no file existed, and the unit checks passed because they always
  pass `&wrote`. `wrote` is now a report, not a permission -- a reminder that an
  optional out-parameter must never gate the side effect.
  (2) **Only inits that got a verdict are counted**: a probe-off run and a dense
  model (no WARM tier, so `probe_on` is false) reused nothing and measured
  nothing, and counting them would drag the rate down without describing any
  cache. The group pins the delta across its three engine runs: +2 runs, +1 hit,
  +1 measurement, and the `KRK_H2D_PROBE=0` arm not counted at all.
  (3) **The saved-time figure is priced at this machine's own average**, not a
  constant: `cached * probe_ms / measured`. With no measurement in the tally
  there is no price and the line says `not priced` rather than printing a zero it
  cannot justify. A corrupt or impossible tally (`cached + measured > runs`) reads
  as no tally -- better than a rate above 100%.
  The ledger lives beside the cache, so `KRK_H2D_CACHE=0` disables it too, and it
  is best effort / last-writer-wins: a server's concurrent inits can round it.
  The probe it prices is **71-73 ms here at 16 MiB** (3 arms x 5 iterations plus
  the untimed warm-up), not the ~20 ms this file used to quote -- and a 64 MiB
  measurement is several times that, which is why the size leads each row.
  End to end on olmoe (the group cannot see stderr, so this is the CLI arm):
  deleted cache -> `there is no cache file yet, or it is empty`; same cache
  again -> `reused the cached 16 MiB one (measured 1 s ago ...)`; `RERUN=1` ->
  `KRK_H2D_PROBE_RERUN=1 asked for a fresh measurement`; a seeded expired 16 MiB
  line -> `not usable (expired), 1 dead of 1 line(s)`; a live 32 MiB line ->
  `1 live at other probe sizes`; a live line from driver 71626330 -> `1 live from
  another driver`. stdout md5 `fb6ef68100e26822` identical on all six runs -- the
  line is a log, not a behaviour change.
  Forced-path text is unchanged: auto / `KRK_WARM_DMA=1` / `=2` print the same
  stdout on the 30B-A3B file.

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
  **tail** (177 -> 583 us on the launch arm, measured 2026-10-11 at a
  512-call pre-issue), never the enqueue. So the 28.2-30.6
  us this table used to call "quiet" was never a queue-state effect: it is the
  device half, which is the right figure only for a call that cannot overlap. The
  **in-step** column stands on its own switch A/B and is the number to rank with
  (2.56 us for the record, because in a step the markers overlap hundreds of
  queued launches); the "0.32 us busy enqueue" this table once called the
  alternative was the same 0.33 us as the empty arm, mislabelled as a state.
  **The arm is in the gate now and proves itself there** -- it asserts the queue
  really is fuller (launch-arm tail >2x), that a record's device half is >=4x
  its host half, that the enqueue does NOT move between states, and that the
  instrument resolves the device cost per op. `KRK_TEST_QUEUE_NOPREISSUE=1` skips
  the pre-issue and fails exactly the two fullness checks (rc=1), which is the
  red construction. The gate's floor line says `unresolvable at iters=2` instead
  of printing `-24.00 us per call` (the gate sets iters=2, where the subtraction
  is the difference of two single measurements).
  **The pre-issue depth is a machine-dependent constant, and 512 stopped being
  enough.** The ratio the check wants is the deep arm's tail over the empty
  arm's, and the empty arm's tail is the block's OWN device work (~220-280 us
  of launches on this box), so the pre-issue has to leave more than that
  behind. The device drains the pre-issue while the host is still enqueueing
  it, so the leftovers scale with the depth. Measured 2026-10-22 on the launch
  arm: 512 pre-issued leaves 411 us against an empty queue's 224-284 -- 1.54x,
  so the gate FAILED on a tree where nothing had changed; 8192 leaves
  2105-2206 us (8.3-10.3x over four runs) and 16384 leaves 4710 us (19.2x).
  The default is 8192 now, the 2.0x threshold is unchanged, and the red
  construction still reports 1.00x and fails. The lesson is the general one: a
  ratio measured on one machine is not a constant -- the drain rate is the
  machine's, so a depth that was enough when the arm was written is not a depth
  that is enough now, and "the gate is red" is the first evidence to look for.
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
- **MLA loads in all three layouts, and the ENGINE is what refuses it; the
  metadata KEYS are the trap.** `deepseek2` and `deepseek4` are in the arch
  table and the loader reads their latent ranks and every layout a file can
  carry. The engine still refuses to GENERATE (no latent attention kernel, no
  576-wide KV cache row, no YaRN-on-the-compressed-rope path), and that refusal
  is `ArchSpec::attention_implemented` -- a flag the loader's
  `KRK_ARCH_ALLOW_EXPERIMENTAL=1` hatch does NOT lift, because the hatch is
  about reading tensors and the flag is about the graph. What the loader
  reads is decided by the TENSORS a layer carries (one arch ships more than
  one layout, and a file is what it carries) and required to be the same on
  every layer: `Legacy` `attn_q` / `attn_kv_a_mqa` / `attn_kv_a_norm` /
  `attn_kv_b` / `attn_output`; `Split` `attn_q_a` / `attn_q_a_norm` /
  `attn_q_b` / `attn_kv_a` / `attn_kv_a_norm` / `attn_k_b` / `attn_v_b` /
  `attn_output` (the V3 class); `Shared` `attn_q_a` / `attn_q_a_norm` /
  `attn_q_b` / `attn_kv` / `attn_kv_a_norm` / `attn_output_a` /
  `attn_output_b` (V4, whose head count is folded into the grouped output
  instead of the KV). Every one is checked against the file's own metadata,
  and a disagreement names the tensor and both shapes. `test_mla_loader` (66
  checks) pins the three layouts, the two-layout mix, a short `attn_k_b`, an
  int32 weight, the hash table's shape and the engine's refusal; `test_mla_lite`
  (57) and `test_mla_absorb` pin the host oracle in two orders
  (`krk::mla_lite_forward_{unfolded,absorbed}`).
  **The ranks are `key_length` / `value_length` / `kv_lora_rank` -- NOT
  `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim`, which no exporter
  writes and which refused every real file at load.** nope is
  `key_length - rope.dimension_count`, and the rope width is the WHOLE
  `dimension_count` for MLA, because the rotation covers the tail of the key.
  A file with no `kv_lora_rank` (V4) has its latent at `key_length`.
  `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` is the standing fixture, and
  its geometry is now checked at load: 16 heads, key 192 = 128 nope + 64
  rope, value 128, kv_lora 512, so `attn_q` is 3072 wide, `attn_kv_a_mqa` is
  576 (512 latent + 64 decoupled key), `attn_kv_b` is 4096 = 16 x (128+128),
  `attn_output` is 2048 = 16 x 128, and the cache row is 576 wide. The
  shared-expert width with no `expert_shared_feed_forward_length` is derived
  (n_shared x n_ff_exp = 2 x 1408 = 2816) instead of staying 0, which used to
  refuse the file at the shared-expert check.
  What the V4 file still carries that nothing here reads: `attn_sinks`,
  Sinkhorn `hc_*`, per-layer `attn_compressor_*` / `indexer_compressor_*` and
  a QSA indexer. Its quant type #26 is NO LONGER a blocker -- that id is
  ggml's I32, and the file's `ffn_gate_tid2eid` is a TABLE OF EXPERT IDS: read
  as `i32` data on the host (`read_i32_table`) and never multiplied. The
  loader refuses an int32 tensor handed to it as a matmul weight, the one
  place that knows the difference between data and a weight -- letting one
  through would have the GEMM read ids as coefficients: finite, fluent and not
  the model. The next pieces, in order: the latent decode kernel (one wide row
  per token, not n_kv x head_dim), the 576-wide cache row, YaRN on the
  compressed rope, then the prefill path.
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
- **A whole layer's promotions in one upload was measured and does not pay --
  the enqueue was never the cost.** `ExpertCache::promote_batch` (called from
  `moe_ffn` right after `prefetch_layer`; `KRK_MOE_BATCH_PROMOTE=1`, off by
  default) makes room and allocates cells per expert but issues ONE upload per
  source kind for the routed set. REAP-320, 16 greedy tokens, 5 interleaved
  order-balanced pairs, identical text in all ten runs: **111.8 against 112.9
  ms/step median**, deltas failing to clear their own 2.6 ms MAD in 4 of 5
  pairs, and the batched arm's `xfer` row consistently WORSE (366-385 against
  249-287 ms over 15 steps). A promotion there is ~500 us/expert, not 143: three
  0.73 MiB slices through a 4 MiB pinned ring slot is a host memcpy plus a DMA
  per slice at ~4.6 GB/s, and that is the same byte count however the calls are
  grouped. The ring is `kPagedBufs = 4` slots, which fits exactly one expert's
  three slices -- so the per-expert call already has every copy in flight, while
  a 4-expert batch (12 chunks) recycles slots and `paged_take_` spins on events.
  Two things came out of it and are kept: the arm, and the bug it exposed --
  `upload_paged_batch` recorded at most `kPagedBufs` slots in `used[]` and
  published only those, so a batch with more chunks than the ring left the rest
  of its copies UNORDERED against the default stream (the consumer GEMM could
  start before the DMA landed). Both the single and batch paths now publish in
  ring-sized groups as the array fills. The ranking that does matter on this
  file: `room` (340 ms) costs more than `xfer` (250 ms) in 15 decode steps.
- **And `room` was 96% victim SEARCH, which is now an index plus a 16-slot
  window.** `pick_victim` walked every entry of `slots_` -- the whole tier, ~3.5k
  resident inside ~14k WARM entries -- per eviction, with a second hash lookup
  per comparison: measured 9.19M visits / 2624 evictions, **505.9 ms scanning
  against 18.1 ms releasing** in a 1.7 s REAP-320 decode. `vram_keys_` (resident
  keys only, `Slot::vram_idx`, O(1) swap-remove, tracked at the three promotion
  sites and untracked in `retire()`) plus a moving window fixes it:
  `KRK_EVICT_WINDOW=16` is the default, and the sweep is 1 -> 103.3, 2 -> 99.4,
  4 -> 97.2, 8 -> 96.8, **16 -> 96.6**, 32 -> 99.3, 64 -> 108.7, whole set ->
  107.9 ms/step. Wider is worse because a wide window reaches freshly touched
  experts and evicts them. Five interleaved pairs against `KRK_EVICT_WINDOW=0`
  (the same index, the old global policy): **96.6 against 107.9**, every pair,
  with 3-4% FEWER promotions; against the pre-index binary that is **+14.4%
  decode, 8.8 -> 10.4 tok/s**, scans 505.9 -> 2.2 ms, 16 visits/eviction.
  REAP-288: 259.8 against 266.8 ms/step. **The trap is the index's lifetime:**
  the first version cleared `vram_keys_` in `reset_counters()`, which deliberately
  KEEPS residency -- so `pick_victim` saw one key for 3484 resident slots and the
  run promoted 2589 -> 6754 experts (2.6x the bytes, xfer 578 -> 3537 ms) with
  every other line reading normal. The index is an invariant of residency, not of
  the statistics generation; the `evict ... | index %zu/%zu keys/resident` line
  is what named it, and `test_expert_cache_byte_budget` arm 4 pins it (61/64 with
  the clear put back).
- **The remaining ~87-96 ms of a REAP-320 decode step is a per-layer host round
  trip, and the kernels are innocent.** `KRK_TIME_HOST=1` (the host table WITHOUT
  the device recorder -- that is why the switch exists) accounts for all of it:
  `moe.route` 1.689 ms/layer x 48 = 81 ms of an 85.6 ms forward, split into
  `moe.route.fetch` 0.868 (**41.7 ms/step**, the blocking router readback) and
  `moe.route.rest` 0.820 (`moe.experts` 0.703 -> **33.7 ms/step**). Dense GEMV is
  ~2.8 ms/step: `KRK_TIME=gemm` measures it at **743 GB/s** (2.65 us a call) and
  the same engine does 464 GB/s on Qwen3-4B-Q8_0, so 2.9 GB in 96 ms is a
  schedule, not a kernel. `KRK_ROUTER_2X=1` is the proof: a second 320-float
  readback costs **0.093 ms** against the first's 0.950 -- the copy is 10% of it,
  the rest is the drain of queued device work the step needed anyway, and the
  second copy finding an EMPTY queue means the host's plan build then runs with
  the card idle (0.13 ms/layer, 6.3 ms/step). The ping-pong is device 0.78 then
  host 0.80 per layer, and they add: that is the measured 1.69. Marginal launch
  cost is 2.3 us/op (`KRK_MOE_GROUP=0` adds ~1032
  ops/step for +2.4 ms), so the 5012 ops/step are ~11.5 ms -- the residual.
  **The pin is no longer a switch: it is the default on a small-BAR card**
  (87.3 against 99.8 ms/step at the 16 GiB explicit arm, 3/3 interleaved pairs,
  373 against 455 us/expert; the automatic 12.3 GiB prefix then measured
  101.7 -> 88.2 and later 113.3 -> 93.2). `KRK_PIN_AUTO=0` is the A/B arm. See
  the aperture bullet above for the cross-model table; the lever is worth +13 to
  +18% exactly when promotions are on the critical path and ~2% when the tier is
  disk-bound. Ranking, unchanged: the readback (41.7) needs device-side routing,
  blocked on "whole routed set resident"; the host expert loop (33.7) is the
  batched-DMA item.
  Prefill is separately odd -- 197 ms/row against a decode row's 96 -- because
  `gemm(gemv_m)`, the rows 2-8 ladder, measures 29 GB/s against gemv's 743.
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
- **A batched decode must resolve only the experts it can KEEP resident, and nothing checked that.** `moe_decode_batch` resolves every routed expert before the first launch, then launches; `ExpertCache::acquire` evicts when the budget cannot hold the routed set, so with a one-slot cache the second resolve hands back the cell the first expert was in. Both experts then multiply ONE expert's weights: finite, plausible, no refusal, and it reaches the sampler. Measured on the synthetic 4-expert fixture: one slot decoded `5 4 5 4 5 4` where a large cache decoded `5 5 4 5 4 5`, and `KRK_MOE_BATCH=0` returned `test_moe_schema_and_laziness` to 32/32 -- that arm is what named the path. Two guards now sit in `moe_decode_batch`: `capacity_slots() >= distinct routed experts` before any acquire (the semantic condition), and all pairwise slice-pointer distinctness after the resolve, 3 slices against 3 slices per expert pair, because a tiered cache can hand expert 2 the cell expert 1 used for a DIFFERENT slice and a same-field comparison misses it -- that shape decoded `5 0 0 0 0 0` at `--expert-cache-slots 1 --expert-warm-mb 64`. Result: 32/32 in a child process and under `KRK_TEST_INPROC=1`, four consecutive runs (suite 3870/3870). `Engine::moe_batch_layers()` / `moe_batch_skips()` expose the counters, and `test_moe_batch_fallback` pins BOTH arms: with 2 experts routed and 1 slot, zero layers batched and the per-expert loop ran; with room for the routed set the batch does compute layers. The roomy arm exists so a guard that refuses unconditionally cannot pass. The general rule: a batch whose members are acquired up front needs a residency guarantee; "the cache will sort it out" is a silent wrong answer, not a fallback.
