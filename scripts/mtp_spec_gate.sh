#!/bin/sh
# mtp_spec_gate.sh -- plan 1.7's speed gate, on a real artifact.
#
# The gate has two halves and both are read from the run itself, not from a
# separate benchmark harness:
#
#   1. step-1 acceptance >= 60 %   -- of the rounds that proposed, the share
#      whose FIRST proposal was the target's own greedy token. Read from the
#      `[stats ] mtp spec:` line, which reports it as `step-1 H of T (P%)`.
#   2. decode faster than plain   -- tokens/s of the same prompt, same seed,
#      same token budget, with and without --mtp / --mtp-spec.
#
# A drafter accepted but slower than plain decode is a regression with a good
# story, so the speed half is NOT optional. On a real 30B-A3B the head is a
# per-token cost of roughly one block against a whole model; the number that
# matters is decode tok/s, and it can lose.
#
# Usage:
#   scripts/mtp_spec_gate.sh <trunk.gguf> <mtp.gguf> [n_tokens] [prompt]
#
# Environment:
#   KRK_BIN      binary to run (default: ./build/kraken, then ./kraken)
#   KRK_THREADS  passed through to the engine
#
# Exit: 0 when both halves pass, 1 when either fails, 2 on a usage/setup error.
set -eu

die() { echo "mtp_spec_gate: $*" >&2; exit 2; }

[ $# -ge 2 ] || {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

model=$1
head=$2
n=${3:-128}
prompt=${4:-"The capital of France is"}

[ -f "$model" ] || die "no such model: $model"
[ -f "$head" ] || die "no such MTP head: $head"

bin=${KRK_BIN:-}
if [ -z "$bin" ]; then
    for c in ./build/kraken ./build-tests/kraken ./kraken; do
        [ -x "$c" ] && { bin=$c; break; }
    done
fi
[ -n "$bin" ] && [ -x "$bin" ] || die "no kraken binary (set KRK_BIN)"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/krk-mtp-gate.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM

# Greedy on both arms, one seed: the acceptance number is only comparable when
# the target's argmax is the same distribution in both runs.
run_arm() { # $1=log $2+=extra args
    log=$1
    shift
    "$bin" -m "$model" -p "$prompt" -n "$n" --greedy --seed 0 "$@" \
        >"$log" 2>&1 || die "the run failed; see $log"
}

echo "mtp_spec_gate: $n tokens, greedy, seed 0"
echo "  model: $model"
echo "  head : $head"
echo

run_arm "$tmp/plain.log"
run_arm "$tmp/spec.log" --mtp "$head" --mtp-spec

# tok/s out of `[stats ] decode ... = X tok/s` -- one pattern for both the plain
# form and the speculation-round form, which differ only in their middle words.
tps() {
    sed -n 's/.*\[stats \] decode .*= \([0-9][0-9.]*\) tok\/s.*/\1/p' "$1" | tail -1
}

plain_tps=$(tps "$tmp/plain.log")
spec_tps=$(tps "$tmp/spec.log")
[ -n "$plain_tps" ] || die "no decode line in the plain run ($tmp/plain.log)"
[ -n "$spec_tps" ] || die "no decode line in the spec run ($tmp/spec.log)"

line=$(grep -F 'mtp spec:' "$tmp/spec.log" | tail -1 || true)
[ -n "$line" ] || die "the spec run printed no 'mtp spec:' summary -- was the head loaded?"

# `... proposals kept (R%), step-1 H of T (P%), ...`
step1=$(printf '%s\n' "$line" | sed -n 's/.*step-1 \([0-9]*\) of \([0-9]*\) (\([0-9.]*\)%).*/\1 \2 \3/p')
[ -n "$step1" ] || die "the 'mtp spec:' line has no step-1 field: $line"
set -- $step1
hits=$1 tries=$2 pct=$3

echo "$line" | sed 's/^/[mtp] /'
echo
printf '  step-1 acceptance  %s of %s = %s%%   (gate: >= 60%%)\n' "$hits" "$tries" "$pct"
printf '  decode plain       %s tok/s\n' "$plain_tps"
printf '  decode --mtp-spec  %s tok/s\n' "$spec_tps"

# Integer arithmetic only: `expr` on the percentages avoids a float compare in
# POSIX sh, and the gate is a 60 % threshold, so scaling by 100 is exact enough.
ok_acc=0
if [ "$tries" -gt 0 ] && [ $((hits * 100)) -ge $((tries * 60)) ]; then ok_acc=1; fi

# Speed: >= 1.00x is "not a regression"; the plan asks for BELOW plain, i.e.
# strictly faster, so the comparison is a strict >.
ok_speed=0
if awk "BEGIN{exit !($spec_tps > $plain_tps)}"; then ok_speed=1; fi

speed_ratio=$(awk "BEGIN{printf \"%.2f\", $spec_tps / $plain_tps}")

echo
printf '  speed ratio        %sx\n' "$speed_ratio"
echo

rc=0
if [ "$ok_acc" = 1 ]; then
    echo "PASS  step-1 acceptance"
else
    echo "FAIL  step-1 acceptance: $pct% is below the 60% gate"
    rc=1
fi
if [ "$ok_speed" = 1 ]; then
    echo "PASS  decode speed"
else
    echo "FAIL  decode speed: $spec_tps tok/s is not faster than $plain_tps"
    rc=1
fi
exit "$rc"
