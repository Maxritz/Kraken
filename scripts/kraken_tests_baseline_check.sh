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

if [ "${KRK_TEST_BASELINE_RECORD:-0}" = "1" ]; then
    extra=""
    [ "${KRK_BASELINE_FORCE:-0}" = "1" ] && extra="--force"
    # shellcheck disable=SC2086
    python3 "$TOOL" --record $extra "$BASE" "$REPORT"
    rc=$?
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
