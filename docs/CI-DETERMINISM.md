# CI determinism doctrine: what a fast kernel must prove

Standing gate for landing any kernel that changes arithmetic, association
order, or tiling. Distilled from the determinism literature's recurring
checklist — replica floor first, named endpoints, negative controls,
trajectory over score — as instantiated by this repo's own gates. A kernel
that cannot pass these does not land; a gate that cannot fail proves nothing.

## 1. Replica floor first

No A/B result is read without its noise floor. Every comparison harness
interleaves order-balanced pairs and reports the control (same arm in both
slots) beside the experiment: `scripts/kraken_sherlock_kernel_ab.sh` refuses
a verdict whose paired gain does not clear the pairs' own MAD in reps-1
pairs, and its control reads NO DIFFERENCE by construction. A kernel whose
"win" sits inside the control's band is not a win.

## 2. Named endpoints, exact bytes

The verdict names the artifact down to the hash: which buffer, which rows,
which positions. `test_prefill_chunk_bitexact` hashes the embedding, every
layer's attention output and hidden state over all prompt rows per
(stage, layer), plus the last-position logits row — because the logits row
alone is provably blind (one-ulp drift in a non-final row is absorbed by the
f16 residual; measured identical while the trace differed). Text equality is
the weakest endpoint and never the only one.

## 3. Trajectory equality over benchmark scores

Compare the whole trajectory, not the final score. The chunk gate compares
per-position dumps (`tools/dump_passes.py` splits at each `embed` pass and
aligns shared positions) because a 12-token prompt can emit identical text
for 24 tokens from divergent states. A gate that compares generated text, or
only the sampled token, would have called the tiled prefill regression
identical. Scores and tok/s are reported, but invariance is decided on bytes.

## 4. Negative control per rung

Every invariance arm carries an arm that must FAIL. The chunk gate re-runs
chunk-8 with `KRK_ATTN_BEXACT=0` and requires the trace hash to DIFFER; the
reporting proof runs `KRK_SELFCHECK_BREAK=counts` (tightens the expectation
by one check) and requires rc=1; the isolation check injects crashes, total
failure, and emptiness and requires each to be reported exactly. A gate whose
bad arm also passes is decor. New kernels land with their own opt-out switch
(`KRK_ATTN_BEXACT=0`, `KRK_GEMM_SPLIT=0`, `KRK_FUSED_SILU=0`,
`KRK_GEMM_INT8`) precisely so the bad arm stays runnable.

## 5. One variable per arm, restoration switches

A/B switches must exist *before* the measurement, and a switch that names a
deleted kernel is worse than none (it silently runs the same arm twice — seen
live with `KRK_ATTN_QTILE`). Every accepted optimisation keeps its
restoration switch so the comparison stays one process apart.

## 6. The checker must be shown able to fail

`kraken-tests --selfcheck` runs six arms per group (crash before/after/at the
last check, all-fail, no-checks, fuzz) and judges each answer against that
group's own counts; `--selfcheck --selftest` runs 17 constructed wrong
answers. The baseline gate runs one group's six arms before every comparison
(`scripts/kraken_tests_baseline_check.sh`), because a gate comparing two
reports is worthless if the reports stopped meaning what they say. Mutating
the classifier must break the selftest (16/18 with the arm comparison
disabled — the case proving the arm findings are not cosmetic).

## 7. Baselines are watermarks with shape comparison

`tests/kraken-tests-baseline.json` records per-group status and counts; the
check fails only on *newly broken* groups, notes count changes, and fails on
shape changes (new/removed groups) until re-recorded deliberately
(`KRK_TEST_BASELINE_RECORD=1`, refuses a non-green run without
`KRK_BASELINE_FORCE=1`). A reverted test file is invisible to every other
gate — four groups came back this way once, caught only by the baseline's
`removed groups` line. Totals are read off the run, never trusted from prose.

## 8. Isolation: no shared fixtures, no shared directories

Every group runs in its own child process with its own working directory,
log, and progress file; a crashed group keeps its evidence and is named with
its signal. The isolation check proves it with injection arms and, under
`KRK_ISOLATION_FULL=1`, each group alone in an empty directory. A group that
writes files removes them when green. The reporting proof runs in the default
gate; the full sweep stays opt-in (`KRK_BASELINE_SELFCHECK=1`) for cost.

## 9. End-to-end promise, not just prefill bytes

Prefill invariance is necessary but not sufficient: the gate extends to
prefill-then-decode text equality (chunk-1-then-decode-N vs
chunk-64-then-decode-N, token for token, `test_prefill_chunk_bitexact`
§"generative arms"), and coherence runs one model per shape (dense /
recurrent / MoE) with both the device and the CPU scalar reference — a
device arm that silently fell back to CPU fails as `FAIL (cpu)`.

## Landing checklist for a new kernel

1. Micro-test pins the property (e.g. `kraken-bench --bitexact`, rows × hd ×
   n_keys matrix, all identical).
2. Invariance gate extended or confirmed green, with its negative control run.
3. Restoration switch added and both arms measured interleaved against the
   noise floor.
4. Reporting proof re-run (`--selfcheck --selftest` + one group) if the
   harness was touched.
5. Baseline re-recorded deliberately if counts or shape changed; fingerprint
   refreshed (`KRK_REFRESH_FINGERPRINT=1`) after a legitimate rebuild, then
   one plain run to confirm the hashes match.
