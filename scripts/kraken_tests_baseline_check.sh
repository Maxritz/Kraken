#!/usr/bin/env bash
# kraken-tests baseline gate: name the groups that newly broke, and nothing else.
#
# kraken-tests writes a per-group report (--report), and tests/kraken-tests-baseline.json
# is the last run that was green. Comparing the two turns "the suite is red" into
# "these groups are red, and these were green an hour ago" -- which is the whole
# difference between a gate a reader trusts and 53 lines they skim.
#
#   scripts/kraken_tests_baseline_check.sh                 # compare, exit 1 on regression
#   KRK_TEST_BASELINE_RECORD=1 scripts/...                 # re-record (deliberate)
#   KRK_BASELINE_ALLOW_NEW=1 scripts/...                   # tolerate shape changes
#
# Recording refuses a run that is not green unless KRK_BASELINE_FORCE=1, because
# a baseline is the claim "this is what passing looked like": one taken from a red
# run makes every later regression look like it was always there.
#
# What the comparison does and does not call a regression is in
# tools/test_report_diff.py, which this script runs -- and whose selftest it runs
# first, so that a green verdict means the comparison was shown to be able to
# fail. A run whose own exit status is non-zero with every group passing (the
# report could not be written, say) is reported as an unusable run rather than
# compared.
#
# Environment: KRK_TEST_BIN (default build-hip/kraken-tests.exe),
# KRK_TEST_BASELINE (default tests/kraken-tests-baseline.json).
#
# The gate carries the cheap half of the reporting proof by default: the runner's
# own selftest (constructed answers, milliseconds) and one group's six injection
# arms, so a gate that compares two reports has first shown the reports still mean
# what they say. The all-groups sweep is what KRK_BASELINE_SELFCHECK=1 adds -- 318
# suite runs, measured 240 s -- and it compares against its own recorded baseline
# (KRK_SELFCHECK_BASELINE, default tests/kraken-tests-selfcheck-baseline.json,
# recorded with the same KRK_TEST_BASELINE_RECORD=1 switch).
#
# Nothing here has to run alone any more. Every run names its own scratch area
# (krk-tests-<pid> beside TEMP) and removes it when it is green, so two suite runs,
# or a suite run beside a group sweep, cannot write over each other's fixtures --
# which is what used to make the reporting proof a thing to run when nothing else
# was running.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_TEST_BIN:-$ROOT/build-hip/kraken-tests.exe}
[ -x "$BIN" ] || BIN=$ROOT/build-hip/kraken-tests
BASE=${KRK_TEST_BASELINE:-$ROOT/tests/kraken-tests-baseline.json}
TOOL=$ROOT/tools/test_report_diff.py

echo "kraken-tests baseline gate"
echo "  binary   : $BIN"
echo "  baseline : $BASE"
echo

if [ ! -x "$BIN" ]; then
    echo "FAIL: no test binary at $BIN -- build it first:" >&2
    echo "      ninja -C build-hip kraken-tests" >&2
    exit 2
fi
if [ ! -f "$TOOL" ]; then
    echo "FAIL: no comparison tool at $TOOL" >&2
    exit 2
fi

# A stale binary is not evidence of anything: a build failure leaves the old
# .exe in place and the run then says something about the old source.
BD=$(dirname "$BIN")
if command -v ninja >/dev/null 2>&1 && [ -f "$BD/build.ninja" ]; then
    if ! ninja -C "$BD" kraken-tests -n 2>/dev/null | grep -q 'no work to do'; then
        echo "FAIL: $BD is behind its sources -- rebuild before believing a run" >&2
        exit 1
    fi
fi

WORK=$(mktemp -d)
REPORT=$WORK/report.json
RUNLOG=$WORK/run.log

"$BIN" --report "$REPORT" > "$RUNLOG" 2>&1
rc=$?
summary=$(grep -oE '[0-9]+/[0-9]+ checks passed.*' "$RUNLOG" | tail -1)
want=$(grep -oE '^report: .*' "$RUNLOG" | tail -1)

if [ ! -s "$REPORT" ]; then
    echo "FAIL: the run wrote no report at $REPORT (rc=$rc)" >&2
    echo "      run log kept: $RUNLOG" >&2
    exit 1
fi
echo "run        rc=$rc, ${summary:-no summary line}"
echo "           ${want:-no report line}"
echo

# The comparison has to be shown able to fail before it is allowed to pass.
if ! python3 "$TOOL" --selftest > "$WORK/selftest.log" 2>&1; then
    echo "FAIL: the comparison tool's own selftest failed:" >&2
    sed 's/^/      /' "$WORK/selftest.log" >&2
    exit 1
fi
echo "tool       $(tail -1 "$WORK/selftest.log" | sed 's/^ *//')"
echo

# --- the reporting proof, in the default gate --------------------------------
# Two cheap checks that the runner still reports what it has to. The selftest is
# the checker's own check -- one constructed answer per way an arm can be wrong --
# and one group's six arms exercise the running sweep: crash before the first
# check, crash after two, crash after the group's own last check, every check
# failing, no checks at all, and a fuzzed aim. Both run even when the all-groups
# sweep is not asked for, because a gate that compares two reports is worth
# nothing if the reports stopped meaning what they say: a runner that reported a
# crashed group as passing would compare clean against a baseline of the same
# defect, and a self-consistent wrong answer passes a baseline.
PROOF_RC=0
"$BIN" --selfcheck --selftest > "$WORK/proof-selftest.log" 2>&1
p_rc=$?
if [ "$p_rc" -eq 0 ] && grep -qE '^17/17 selfcheck cases passed$' "$WORK/proof-selftest.log"; then
    echo "proof      $(grep -E 'selfcheck cases passed' "$WORK/proof-selftest.log" | tail -1 | sed 's/^ *//')"
else
    PROOF_RC=1
    echo "FAIL: the runner's own selftest did not pass (rc=$p_rc)" >&2
    sed 's/^/      /' "$WORK/proof-selftest.log" >&2
fi
PROOF_GROUP=${KRK_SELFCHECK_GROUP:-0}
"$BIN" --selfcheck --group "$PROOF_GROUP" > "$WORK/proof-group.log" 2>&1
p_rc=$?
p_line=$(grep -oE '[0-9]+/[0-9]+ groups: every injection.*' "$WORK/proof-group.log" | tail -1)
if [ "$p_rc" -eq 0 ] && [ -n "$p_line" ]; then
    echo "           group $PROOF_GROUP: $p_line"
else
    PROOF_RC=1
    echo "FAIL: the reporting proof failed for group $PROOF_GROUP (rc=$p_rc)" >&2
    echo "      log kept: $WORK/proof-group.log" >&2
    tail -5 "$WORK/proof-group.log" | sed 's/^/      /' >&2
fi
echo

# --- the reporting proof, against its own recorded baseline ------------------
# The comparison below answers "did any group break". This one answers "did the
# runner stop reporting a group the way it has to": --selfcheck kills one group
# on purpose in five fixed ways, and a random one in a sixth -- sometimes aiming
# at two groups at once -- and requires each answer to carry the group's own
# count. An arm the baseline has as ok and this run does not is a regression,
# named with the problem it reported.
SELF_RC=0
SELF_BASE=${KRK_SELFCHECK_BASELINE:-$ROOT/tests/kraken-tests-selfcheck-baseline.json}
if [ "${KRK_BASELINE_SELFCHECK:-0}" = "1" ]; then
    SELFREP=$WORK/selfcheck.json
    SELFLOG=$WORK/selfcheck.log
    "$BIN" --selfcheck --report "$SELFREP" > "$SELFLOG" 2>&1
    sweep_rc=$?
    echo "selfcheck  rc=$sweep_rc, $(grep -oE '[0-9]+/[0-9]+ groups: every injection.*' "$SELFLOG" | tail -1)"
    echo "           $(grep -oE '^selfcheck: seed .*' "$SELFLOG" | tail -1)"
    echo
    if [ ! -s "$SELFREP" ]; then
        echo "FAIL: the selfcheck wrote no report at $SELFREP (rc=$sweep_rc)" >&2
        echo "      selfcheck log kept: $SELFLOG" >&2
        SELF_RC=1
    else
        extra=""
        [ "${KRK_TEST_BASELINE_RECORD:-0}" = "1" ] && extra="$extra --record"
        [ "${KRK_BASELINE_FORCE:-0}" = "1" ] && extra="$extra --force"
        [ "${KRK_BASELINE_ALLOW_NEW:-0}" = "1" ] && extra="$extra --allow-new"
        # shellcheck disable=SC2086
        python3 "$TOOL" $extra "$SELF_BASE" "$SELFREP"
        cmp_rc=$?
        if [ "$cmp_rc" -eq 0 ] && [ "$sweep_rc" -eq 0 ]; then
            if [ "${KRK_TEST_BASELINE_RECORD:-0}" = "1" ]; then
                echo "ok: $SELF_BASE re-recorded from this sweep"
            else
                echo "ok: the reporting proof matches its recorded baseline"
            fi
        else
            SELF_RC=1
            echo "FAIL: the reporting proof does not match (sweep rc=$sweep_rc, comparison rc=$cmp_rc)" >&2
            echo "      selfcheck log kept: $SELFLOG" >&2
            echo "      if this is the new truth, re-record deliberately:" >&2
            echo "      KRK_BASELINE_SELFCHECK=1 KRK_TEST_BASELINE_RECORD=1 sh scripts/kraken_tests_baseline_check.sh" >&2
        fi
    fi
    echo
fi

if [ "${KRK_TEST_BASELINE_RECORD:-0}" = "1" ]; then
    extra=""
    [ "${KRK_BASELINE_FORCE:-0}" = "1" ] && extra="--force"
    # shellcheck disable=SC2086
    python3 "$TOOL" --record $extra "$BASE" "$REPORT"
    rc=$?
    if [ "$SELF_RC" -ne 0 ] || [ "$PROOF_RC" -ne 0 ]; then rc=1; fi
    if [ "$rc" -eq 0 ]; then
        echo "ok: $BASE re-recorded from this run"
        rm -rf "$WORK"
    else
        echo "FAIL: refused to record (see above); run log kept: $RUNLOG" >&2
    fi
    exit "$rc"
fi

extra=""
[ "${KRK_BASELINE_ALLOW_NEW:-0}" = "1" ] && extra="--allow-new"
# shellcheck disable=SC2086
python3 "$TOOL" $extra "$BASE" "$REPORT"
rc=$?
if [ "$SELF_RC" -ne 0 ] || [ "$PROOF_RC" -ne 0 ]; then rc=1; fi

echo
if [ "$rc" -eq 0 ]; then
    echo "ok: nothing newly broke against the baseline"
    rm -rf "$WORK"
else
    echo "FAIL: the run does not match the baseline (see the findings above)" >&2
    echo "      run log kept: $RUNLOG" >&2
    echo "      if these failures are the new truth, re-record deliberately:" >&2
    echo "      KRK_TEST_BASELINE_RECORD=1 sh scripts/kraken_tests_baseline_check.sh" >&2
fi
exit "$rc"
