#!/usr/bin/env bash
# kraken-tests isolation check: prove the suite survives a group that dies.
#
# kraken-tests runs each of its 53 groups in a child process, so a crash inside
# one group costs that group and nothing else. The claim that matters is not
# "it does not crash here" -- it is that a group CAN die and the run still says
# which one and still prints its count, because the thing this suite used to do
# was die silently: one std::fclose(NULL) in one test called __fastfail and the
# process ended before the "%d/%d checks passed" line, so the eight real
# failures behind it stayed invisible for a whole session.
#
# This script is the constructed failure that keeps that honest. It drives the
# binary through the injections the harness ships for exactly this purpose:
#
#   KRK_TEST_INJECT_CRASH=<index|name>     one group dies on purpose
#   KRK_TEST_INJECT_CRASH_AT=<n>           ... after check n of that group
#   KRK_TEST_INJECT_FAIL=<index|name>      every check in one group fails
#   KRK_TEST_INPROC=1                      one process, the pre-isolation runner
#
# It checks: the binary is current; the clean run is clean in BOTH runners and
# reports the SAME count; a killed group is named with its partial count and the
# summary still prints; a failed group is named and the run is non-zero; an
# injection that matches no group is inert; a bad argument is an error and not a
# silent clean run. Logs are kept when something fails and removed when not.
#
# Environment: KRK_TEST_BIN (default build-hip/kraken-tests.exe), KRK_CRASH_GROUP
# (default test_kv_tier_drop_vs_spill -- the group whose fclose(NULL) started
# this), KRK_CRASH_AT (default 5), KRK_FAIL_GROUP (default test_json).
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_TEST_BIN:-$ROOT/build-hip/kraken-tests.exe}
[ -x "$BIN" ] || BIN=$ROOT/build-hip/kraken-tests
CRASH_GROUP=${KRK_CRASH_GROUP:-test_kv_tier_drop_vs_spill}
CRASH_AT=${KRK_CRASH_AT:-5}
FAIL_GROUP=${KRK_FAIL_GROUP:-test_json}

pass=0
fail=0
ok()  { printf 'ok    %-14s %s\n' "$1" "$2"; pass=$((pass + 1)); }
bad() { printf 'FAIL  %-14s %s\n' "$1" "$2"; fail=$((fail + 1)); }

summary_of() { grep -oE '[0-9]+/[0-9]+ checks passed' "$1" | tail -1; }
name_in()    { grep -q "$2" "$1"; }

echo "kraken-tests isolation check"
echo "  binary : $BIN"
echo "  groups : crash $CRASH_GROUP (at check $CRASH_AT), fail $FAIL_GROUP"
echo

if [ ! -x "$BIN" ]; then
    echo "FAIL: no test binary at $BIN -- build it first:" >&2
    echo "      ninja -C build-hip kraken-tests" >&2
    exit 2
fi

# A stale binary is not evidence of anything: a build failure leaves the old
# .exe in place and the run then silently proves something about the old source.
BD=$(dirname "$BIN")
if command -v ninja >/dev/null 2>&1 && [ -f "$BD/build.ninja" ]; then
    if ninja -C "$BD" kraken-tests -n 2>/dev/null | grep -q 'no work to do'; then
        ok "current" "ninja has no pending work for kraken-tests"
    else
        bad "current" "$BD is behind its sources: rebuild before believing a run"
    fi
fi

LOG=$(mktemp -d)

# --- 1. the clean run, both runners, the same number -------------------------
"$BIN" > "$LOG/child.log" 2>&1
rc=$?
base=$(summary_of "$LOG/child.log")
if [ "$rc" -eq 0 ] && [ -n "$base" ] && ! grep -qE 'CRASHED|FAILED|COULD NOT START' "$LOG/child.log"; then
    ok "clean-child" "rc=0, $base, no group reported bad"
else
    bad "clean-child" "rc=$rc, summary '${base:-none}'"
fi

KRK_TEST_INPROC=1 "$BIN" > "$LOG/inproc.log" 2>&1
rc=$?
inproc=$(summary_of "$LOG/inproc.log")
if [ "$rc" -eq 0 ] && [ "$inproc" = "$base" ]; then
    ok "clean-inproc" "rc=0, KRK_TEST_INPROC=1 agrees: $inproc"
else
    bad "clean-inproc" "rc=$rc, child said '$base', in-proc said '${inproc:-none}'"
fi

# A single group run by hand must also be clean.
"$BIN" --run-group "$CRASH_GROUP" > "$LOG/one.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "^\[group $CRASH_GROUP\]" "$LOG/one.log"; then
    ok "single-group" "rc=0, $(tail -1 "$LOG/one.log")"
else
    bad "single-group" "rc=$rc, want rc=0 and a '[$CRASH_GROUP]' result line"
fi

# --- 2. a group that dies at its first check ---------------------------------
KRK_TEST_INJECT_CRASH="$CRASH_GROUP" "$BIN" > "$LOG/crash.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] && grep -q 'CRASHED 0xC0000409' "$LOG/crash.log" \
        && name_in "$LOG/crash.log" "groups needing attention: $CRASH_GROUP" \
        && [ -n "$(summary_of "$LOG/crash.log")" ]; then
    ok "crash-start" "rc=$rc, named at once, $(summary_of "$LOG/crash.log")"
else
    bad "crash-start" "rc=$rc, want a named CRASHED group and a summary line"
fi

# --- 3. a group that dies part-way: the count must be the partial one --------
KRK_TEST_INJECT_CRASH="$CRASH_GROUP" KRK_TEST_INJECT_CRASH_AT="$CRASH_AT" "$BIN" \
    > "$LOG/crashat.log" 2>&1
rc=$?
partial=$(grep -cE "[0-9]+/$CRASH_AT +CRASHED 0xC0000409" "$LOG/crashat.log")
if [ "$rc" -ne 0 ] && [ "$partial" -eq 1 ] \
        && [ -n "$(summary_of "$LOG/crashat.log")" ]; then
    ok "crash-mid" "rc=$rc, died at $CRASH_AT/$CRASH_AT checks, $(summary_of "$LOG/crashat.log")"
else
    bad "crash-mid" "rc=$rc, want one '$CRASH_AT/$CRASH_AT CRASHED' line, got $partial"
fi

# --- 4. a group whose checks fail --------------------------------------------
KRK_TEST_INJECT_FAIL="$FAIL_GROUP" "$BIN" > "$LOG/failrun.log" 2>&1
rc=$?
fs=$(summary_of "$LOG/failrun.log")
nums=${fs%% *}
got=${nums%%/*}
tot=${nums##*/}
if [ "$rc" -ne 0 ] && grep -q 'FAILED' "$LOG/failrun.log" \
        && name_in "$LOG/failrun.log" "groups needing attention: $FAIL_GROUP" \
        && [ -n "$fs" ] && [ "${got:-x}" -lt "${tot:-0}" ] 2>/dev/null; then
    ok "fail-injected" "rc=$rc, $fs, $FAIL_GROUP named as FAILED"
else
    bad "fail-injected" "rc=$rc, summary '${fs:-none}' (want passed < run, and FAILED)"
fi

# --- 5. the switches are opt-in, and a typo is visible -----------------------
KRK_TEST_INJECT_CRASH=no_such_group "$BIN" > "$LOG/typo.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q 'matches no group' "$LOG/typo.log" \
        && [ "$(summary_of "$LOG/typo.log")" = "$base" ]; then
    ok "inert-typo" "rc=0, warned, count unchanged: $base"
else
    bad "inert-typo" "rc=$rc, want rc=0 plus a 'matches no group' warning"
fi

# --- 6. a bad argument is an error -------------------------------------------
"$BIN" --run-group no_such_group > "$LOG/arg.log" 2>&1
rc1=$?
"$BIN" --no-such-flag > "$LOG/flag.log" 2>&1
rc2=$?
if [ "$rc1" -eq 2 ] && [ "$rc2" -eq 2 ]; then
    ok "bad-args" "unknown group rc=2, unknown flag rc=2"
else
    bad "bad-args" "unknown group rc=$rc1 (want 2), unknown flag rc=$rc2 (want 2)"
fi

echo
echo "$pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
    rm -rf "$LOG"
else
    echo "logs kept: $LOG"
fi
echo
[ "$fail" -eq 0 ] || exit 1
