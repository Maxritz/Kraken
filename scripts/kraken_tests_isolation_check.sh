#!/usr/bin/env bash
# kraken-tests isolation check: prove the suite survives a group that dies, and
# that the groups really are independent enough to run at once.
#
# kraken-tests runs each of its 53 groups in a child process, several at a time.
# Two claims have to hold for that to be worth anything:
#
#   * a group CAN die and the run still says which one, with the count it had
#     reached, and still prints its summary. The thing this suite used to do was
#     die silently: one std::fclose(NULL) in one test called __fastfail and the
#     process ended before the "%d/%d checks passed" line, so the eight real
#     failures behind it stayed invisible for a whole session.
#   * no two groups share a file or a resource, because a group that depends on
#     another group's fixture passes or fails according to the schedule. Each
#     child gets its own working directory (every fixture in the suite is a
#     relative path), so the extra claim to check is that every group reports
#     the same count alone in an empty directory as it does in the full run.
#
# This script is the constructed failure that keeps both honest. It drives the
# binary through the switches the harness ships for exactly this purpose:
#
#   KRK_TEST_INJECT_CRASH=<index|name>     one group dies on purpose
#   KRK_TEST_INJECT_CRASH_AT=<n>           ... after check n of that group
#   KRK_TEST_INJECT_FAIL=<index|name>      every check in one group fails
#   KRK_TEST_INJECT_EMPTY=<index|name>     one group runs NO checks at all
#   KRK_TEST_SERIAL_OVERRIDE=<index|name>  one group takes the serial lane
#   KRK_TEST_WORKERS=<n>, --workers <n>    how many groups run at once
#   KRK_TEST_INPROC=1                      one process, the pre-isolation runner
#
# It checks: the binary is current; the clean run is clean in every runner --
# default workers, one worker, and in-proc -- with the same per-group counts in
# all three; a killed group is named with its partial count and the summary
# still prints; a failed group is named and the run is non-zero; the serial lane
# runs a group and says so; the worker count clamps instead of refusing; a group
# that ran no checks is refused rather than counted as passing; an injection that
# matches no group is inert; a bad argument is an error and not a silent clean
# run. Logs are kept when something fails and removed when not.
#
# Environment: KRK_TEST_BIN (default build-hip/kraken-tests.exe), KRK_CRASH_GROUP
# (default test_kv_tier_drop_vs_spill -- the group whose fclose(NULL) started
# this), KRK_CRASH_AT (default 5), KRK_FAIL_GROUP (default test_json),
# KRK_SERIAL_GROUP (default test_http_server), KRK_ISOLATION_FULL=1 to add the
# per-group sweep (53 more runs, ~20 s: every group alone in an empty directory
# must report the count it reports in the full run).
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_TEST_BIN:-$ROOT/build-hip/kraken-tests.exe}
[ -x "$BIN" ] || BIN=$ROOT/build-hip/kraken-tests
CRASH_GROUP=${KRK_CRASH_GROUP:-test_kv_tier_drop_vs_spill}
CRASH_AT=${KRK_CRASH_AT:-5}
FAIL_GROUP=${KRK_FAIL_GROUP:-test_json}
SERIAL_GROUP=${KRK_SERIAL_GROUP:-test_http_server}

pass=0
fail=0
ok()  { printf 'ok    %-14s %s\n' "$1" "$2"; pass=$((pass + 1)); }
bad() { printf 'FAIL  %-14s %s\n' "$1" "$2"; fail=$((fail + 1)); }

summary_of() { grep -oE '[0-9]+/[0-9]+ checks passed' "$1" | tail -1; }
plan_of()    { grep -m1 -E '^kraken test suite:' "$1"; }
workers_of() { plan_of "$1" | grep -oE '[0-9]+ at a time'; }
name_in()    { grep -q "$2" "$1"; }

# "name passed/run" for every group line, sorted, so two runs can be compared
# group by group rather than only in the total.
group_table() {
    sed -nE 's/^[[:space:]]*\[[[:space:]]*[0-9]+\/[0-9]+\][[:space:]]+([^ ]+)[[:space:]]+([0-9]+)\/([0-9]+).*/\1 \2\/\3/p' "$1" | sort
}

echo "kraken-tests isolation check"
echo "  binary : $BIN"
echo "  groups : crash $CRASH_GROUP (at check $CRASH_AT), fail $FAIL_GROUP, serial $SERIAL_GROUP"
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

# --- 1. the clean run, three runners, the same numbers ----------------------
"$BIN" > "$LOG/child.log" 2>&1
rc=$?
base=$(summary_of "$LOG/child.log")
plan=$(plan_of "$LOG/child.log")
if [ "$rc" -eq 0 ] && [ -n "$base" ] && ! grep -qE 'CRASHED|FAILED|COULD NOT START|NO SCRATCH' "$LOG/child.log"; then
    ok "clean-child" "rc=0, $base, no group reported bad"
else
    bad "clean-child" "rc=$rc, summary '${base:-none}'"
fi

if name_in "$LOG/child.log" "serial groups: none"; then
    ok "serial-set" "the run reports the serial set: none declared"
else
    bad "serial-set" "the plan must print the serial set ('serial groups: ...'); got '$(grep -m1 'serial' "$LOG/child.log")'"
fi

n=$(workers_of "$LOG/child.log")
if [ -n "$n" ]; then
    ok "worker-plan" "runs ${n% at a time} groups at once by default"
else
    bad "worker-plan" "the plan line must say how many groups run at once; got '$plan'"
fi

"$BIN" --workers 1 > "$LOG/one.log" 2>&1
rc=$?
one=$(summary_of "$LOG/one.log")
if [ "$rc" -eq 0 ] && [ "$one" = "$base" ] && [ "$(workers_of "$LOG/one.log")" = "1 at a time" ]; then
    ok "clean-serial" "rc=0, --workers 1 agrees: $one"
else
    bad "clean-serial" "rc=$rc, want rc=0, '1 at a time' and '$base'; got '$one'"
fi

KRK_TEST_INPROC=1 "$BIN" > "$LOG/inproc.log" 2>&1
rc=$?
inproc=$(summary_of "$LOG/inproc.log")
if [ "$rc" -eq 0 ] && [ "$inproc" = "$base" ]; then
    ok "clean-inproc" "rc=0, KRK_TEST_INPROC=1 agrees: $inproc"
else
    bad "clean-inproc" "rc=$rc, child said '$base', in-proc said '${inproc:-none}'"
fi

# The A/B has to be per group, not just in total: a runner that dropped one
# group and ran another twice could still reach the right total.
group_table "$LOG/child.log" > "$LOG/table.child"
group_table "$LOG/one.log" > "$LOG/table.one"
group_table "$LOG/inproc.log" > "$LOG/table.inproc"
gcount=$(wc -l < "$LOG/table.child")
if [ "$gcount" -gt 0 ] && cmp -s "$LOG/table.child" "$LOG/table.one" \
        && cmp -s "$LOG/table.child" "$LOG/table.inproc"; then
    ok "per-group-a/b" "all $gcount group counts identical in all three runners"
else
    bad "per-group-a/b" "$gcount group lines; differ: $(diff "$LOG/table.child" "$LOG/table.one" | head -3 | tr '\n' ' ')"
fi

# A single group run by hand must also be clean -- and it must be clean on its
# own, with nothing from a previous run in its directory.
"$BIN" --run-group "$CRASH_GROUP" > "$LOG/one_group.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "^\[group $CRASH_GROUP\]" "$LOG/one_group.log"; then
    ok "single-group" "rc=0, $(tail -1 "$LOG/one_group.log")"
else
    bad "single-group" "rc=$rc, want rc=0 and a '[$CRASH_GROUP]' result line"
fi

# --- 2. the serial lane really runs a group alone ---------------------------
KRK_TEST_SERIAL_OVERRIDE="$SERIAL_GROUP" "$BIN" > "$LOG/serial.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && name_in "$LOG/serial.log" "serial groups: $SERIAL_GROUP (KRK_TEST_SERIAL_OVERRIDE)" \
        && [ "$(summary_of "$LOG/serial.log")" = "$base" ]; then
    ok "serial-lane" "rc=0, $SERIAL_GROUP ran in the serial lane, count unchanged"
else
    bad "serial-lane" "rc=$rc, want the override named in 'serial groups:' and $base"
fi

# --- 3. a group that dies at its first check ---------------------------------
KRK_TEST_INJECT_CRASH="$CRASH_GROUP" "$BIN" > "$LOG/crash.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] && grep -q 'CRASHED 0xC0000409' "$LOG/crash.log" \
        && name_in "$LOG/crash.log" "groups needing attention: $CRASH_GROUP" \
        && [ -n "$(summary_of "$LOG/crash.log")" ]; then
    ok "crash-start" "rc=$rc, named at once, $(summary_of "$LOG/crash.log")"
else
    bad "crash-start" "rc=$rc, want a named CRASHED group and a summary line"
fi

# --- 4. a group that dies part-way: the count must be the partial one --------
KRK_TEST_INJECT_CRASH="$CRASH_GROUP" KRK_TEST_INJECT_CRASH_AT="$CRASH_AT" "$BIN" \
    > "$LOG/crashat.log" 2>&1
rc=$?
partial=$(grep -cE "$CRASH_AT/$CRASH_AT +[0-9]+ms +CRASHED 0xC0000409" "$LOG/crashat.log")
if [ "$rc" -ne 0 ] && [ "$partial" -eq 1 ] \
        && [ -n "$(summary_of "$LOG/crashat.log")" ]; then
    ok "crash-mid" "rc=$rc, died at $CRASH_AT/$CRASH_AT checks, $(summary_of "$LOG/crashat.log")"
else
    bad "crash-mid" "rc=$rc, want one '$CRASH_AT/$CRASH_AT ... CRASHED' line, got $partial"
fi

# --- 5. a group whose checks fail --------------------------------------------
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

# The failing group's log has to reach the parent's output: with groups running
# at once, a FAIL line that only exists in the group's own file is a FAIL line
# nobody reads. Every failing check has a FAIL line, and only the injected group
# fails, so the two numbers must agree.
flines=$(grep -c 'FAIL ' "$LOG/failrun.log")
missing=$((tot - got))
if [ "$flines" -eq "$missing" ]; then
    ok "fail-lines" "all $flines FAIL lines from $FAIL_GROUP reached the parent's log"
else
    bad "fail-lines" "$missing failing checks, $flines FAIL lines in the parent's log"
fi

# --- 6. a group that runs no checks is not a passing group -------------------
# test_gdn_generation used to return 0/0 when its fixture was missing, which the
# runner counted as a pass. This is that case, constructed.
KRK_TEST_INJECT_EMPTY="$FAIL_GROUP" "$BIN" > "$LOG/empty.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] && grep -q 'NO CHECKS' "$LOG/empty.log"         && name_in "$LOG/empty.log" "groups needing attention: $FAIL_GROUP"         && grep -q 'ran no checks' "$LOG/empty.log"; then
    ok "empty-group" "rc=$rc, $(summary_of "$LOG/empty.log") + 'ran no checks'"
else
    bad "empty-group" "rc=$rc, want a NO CHECKS group named and a non-zero exit"
fi

# --- 7. the switches are opt-in, and a typo is visible -----------------------
KRK_TEST_INJECT_CRASH=no_such_group "$BIN" > "$LOG/typo.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q 'matches no group' "$LOG/typo.log" \
        && [ "$(summary_of "$LOG/typo.log")" = "$base" ]; then
    ok "inert-typo" "rc=0, warned, count unchanged: $base"
else
    bad "inert-typo" "rc=$rc, want rc=0 plus a 'matches no group' warning"
fi

# --- 8. the worker count clamps ---------------------------------------------
"$BIN" --workers 0 > "$LOG/w0.log" 2>&1
rc0=$?
"$BIN" --workers 9999 > "$LOG/wbig.log" 2>&1
rc1=$?
if [ "$rc0" -eq 0 ] && [ "$rc1" -eq 0 ] \
        && [ "$(workers_of "$LOG/w0.log")" = "1 at a time" ] \
        && [ "$(summary_of "$LOG/w0.log")" = "$base" ] \
        && [ "$(summary_of "$LOG/wbig.log")" = "$base" ]; then
    ok "worker-clamp" "0 -> $(workers_of "$LOG/w0.log"), 9999 -> $(workers_of "$LOG/wbig.log"), counts unchanged"
else
    bad "worker-clamp" "rc=$rc0/$rc1, 0 -> '$(workers_of "$LOG/w0.log")', 9999 -> '$(workers_of "$LOG/wbig.log")'"
fi

# --- 9. a bad argument is an error -------------------------------------------
"$BIN" --run-group no_such_group > "$LOG/arg.log" 2>&1
rc1=$?
"$BIN" --no-such-flag > "$LOG/flag.log" 2>&1
rc2=$?
if [ "$rc1" -eq 2 ] && [ "$rc2" -eq 2 ]; then
    ok "bad-args" "unknown group rc=2, unknown flag rc=2"
else
    bad "bad-args" "unknown group rc=$rc1 (want 2), unknown flag rc=$rc2 (want 2)"
fi

# --- 10. a run started in a directory leaves that directory as it found it ----
# The in-proc runner works in the caller's directory, so whatever it writes it has
# to take away again: it used to clean six of the thirteen names the suite can
# create and left the rest for the next run -- or the next reader -- to trip over.
# The child runner gives each group a scratch directory, so the same promise holds
# there for a different reason, and both are checked.
LEAVES=$(mktemp -d)
for mode in inproc child; do
    d="$LEAVES/$mode"
    mkdir -p "$d"
    if [ "$mode" = inproc ]; then
        ( cd "$d" && KRK_TEST_INPROC=1 "$BIN" > run.log 2>&1 )
    else
        ( cd "$d" && "$BIN" > run.log 2>&1 )
    fi
    left=$(cd "$d" && ls | grep -v '^run.log$' | tr '
' ' ')
    if [ -z "$left" ]; then
        ok "leaves-nothing" "$mode: an empty directory came back empty"
    else
        bad "leaves-nothing" "$mode left behind: $left"
    fi
done
rm -rf "$LEAVES"

# The rule has a boundary, and the boundary is the point: it removes the shapes
# the suite writes and refuses everything else -- including names that merely
# start with kraken-. A cleanup that is one wildcard too wide deletes a caller's
# binary, which is why this arm exists rather than a comment saying it is careful.
BOUND=$(mktemp -d)
mkdir -p "$BOUND/kraken-decoy-cold-dir"
for f in kraken-decoy.gguf kraken-decoy.krakenexperts.json kraken-decoy-cold-dir/data.bin; do
    echo x > "$BOUND/$f"
done
for f in keep-me.txt kraken-notes.md kraken.exe; do
    echo keep > "$BOUND/$f"
done
( cd "$BOUND" && KRK_TEST_INPROC=1 "$BIN" > run.log 2>&1 )
left=$(cd "$BOUND" && ls | sort | tr '
' ' ')
if [ "$left" = "keep-me.txt kraken-notes.md kraken.exe run.log " ]; then
    ok "fixture-boundary" "the suite's shapes went, the look-alikes stayed"
else
    bad "fixture-boundary" "want only keep-me.txt, kraken-notes.md, kraken.exe, run.log; got: $left"
fi
rm -rf "$BOUND"

# --- 11. opt-in: every group is self-sufficient ------------------------------
# The strong version of what makes --workers safe. Each group runs alone, in an
# empty directory of its own, and must report the count it reports in the full
# run: a group that reads another group's fixture fails here, because in this
# sweep nobody else has run.
if [ "${KRK_ISOLATION_FULL:-0}" = "1" ]; then
    SWEEP=$(mktemp -d)
    for i in $(seq 0 $((gcount - 1))); do
        mkdir -p "$SWEEP/$i"
        ( cd "$SWEEP/$i" && "$BIN" --run-group "$i" > out.txt 2>&1 )
    done
    sed -nE 's/^\[group ([^]]+)\] ([0-9]+)\/([0-9]+) checks/\1 \2\/\3/p' \
        "$SWEEP"/*/out.txt | sort > "$SWEEP/table"
    if cmp -s "$SWEEP/table" "$LOG/table.child"; then
        ok "self-contained" "all $gcount groups alone in an empty dir: same counts as the run"
    else
        bad "self-contained" "groups that differ when run alone: $(diff "$SWEEP/table" "$LOG/table.child" | head -6 | tr '\n' ' ')"
    fi
    rm -rf "$SWEEP"
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
