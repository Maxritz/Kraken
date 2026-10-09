#!/usr/bin/env bash
# sherlock-it A/B harness for an offline kernel swap behind the existing
# KRK_TIME_OP wrapper: interleaved, repeated, and unwilling to call a win that the
# data does not support.
#
#   scripts/kraken_sherlock_kernel_ab.sh [model] [prompt] [ctx] [n] [reps]
#
# Why interleaved and repeated: this workload's own noise is larger than most
# kernel wins. Measured on SmolLM2-135M (ctx 512, n 64, greedy) -- two arms of the
# SAME binary, back to back -- decode moved 59.0 -> 60.2 tok/s (2.0%) while the
# device table's gemm(gemv) NET column moved 11.4%, and between runs of one binary
# the same arm moved 8.5%. A single pair, or two arms run in sequence, measures
# the machine's drift as much as the kernel.
#
# Why the timed arms do NOT profile: `KRK_PROFILE=1` is not a passive observer
# here. Measured on the same workload, decode is 601.2 tok/s (1.7 ms/step)
# unprofiled and 69.0 tok/s (14.5 ms/step) profiled -- an 8.7x slowdown, because
# the device recorder's floor (~20 us per record, two records per op, ~303 ops per
# step) is real serialized stream time. A timing comparison taken under it compares
# the profiler. KRK_AB_PROFILE=1 adds one profiled pair at the end for the op table
# -- whose SHARES and per-op COUNTS are meaningful and whose absolute ms are not.
#
# Reading the verdict:
#   WIN             B beat A in at least reps-1 pairs, and the median paired gain
#                   exceeds the median absolute deviation of those gains
#   NO DIFFERENCE   the pairs do not say the same thing, or the median gain is
#                   inside the spread -- the default outcome for an inert switch
#
# The verdict uses an ORDER-BALANCED median (pairs in AB order and pairs in BA
# order are medianed separately, and those two figures are averaged); the raw
# median over all pairs is printed beside it and is not used. Reason: the slot a
# run occupies inside a pair is worth several percent here, in whichever
# direction a given invocation happens to run, so a raw median reports the slot
# effect as if it were the arm's. Control that pins both: the SAME arm in both
# slots (KRK_AB_A_SWITCH and KRK_AB_SWITCH identical) reads -0.35% raw, MAD
# 2.34%, 2/5 pairs -- NO DIFFERENCE, as an inert switch must.
#   REGRESSION      the same test, with the sign the other way (exit 1)
#   OUTPUT MISMATCH some run's text differed from another's (exit 1): a speed
#                   number is worthless without the text check, so this is checked
#                   first and reported as its own verdict
#   INVALID         a run failed or produced no timing (exit 1)
#
# Environment: KRK_AB_SWITCH (default KRK_EXPERIMENTAL_GEMV=1) is the arm-B switch,
# comma-separated for several; KRK_AB_A_SWITCH is the same for arm A, which is what
# makes the WIN path constructible (A = KRK_PROFILE=1, B = nothing, and B really is
# faster); KRK_AB_REPS (default 5); KRK_AB_PROFILE=1 to add the profiled pair;
# KRK_AB_KEEP=1 to keep the run directory.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_TEST_BIN:-$ROOT/build-hip/kraken.exe}

usage() {
    sed -n '2,12p' "$0"
    exit 2
}
[ "${1:-}" = "-h" ] && usage

MODEL=${1:-$ROOT/models/SmolLM2-135M-Instruct.Q4_K_M.gguf}
PROMPT=${2:-"The history of computing is a history of abstraction"}
CTX=${3:-512}
N=${4:-64}
REPS=${5:-${KRK_AB_REPS:-5}}
SWITCH=${KRK_AB_SWITCH:-KRK_EXPERIMENTAL_GEMV=1}
A_SWITCH=${KRK_AB_A_SWITCH:-}
# A comma-separated switch list has to become separate WORDS before it
# reaches `env`: `env A=0,B=0 kraken.exe` sets ONE variable named A to the
# string "0,B=0", which every switch here reads as truthy (it is not "0"),
# so both arms silently ran the same configuration and the comparison
# reported NO DIFFERENCE on an experiment that had no arms. Measured, not
# imagined: KRK_AB_A_SWITCH=KRK_ACC_GEMM=0,KRK_FUSED_SILU=0 against both
# switches on gave a median -0.52% (MAD 0.70%) -- the shape of two arms
# that are the same arm.
SWITCH=${SWITCH//,/ }
A_SWITCH=${A_SWITCH//,/ }

[ -x "$BIN" ] || { echo "FAIL: no binary at $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "FAIL: no model at $MODEL" >&2; exit 2; }
case "$REPS" in ''|*[!0-9]*) echo "FAIL: reps '$REPS' is not a number" >&2; exit 2;; esac
[ "$REPS" -ge 1 ] || { echo "FAIL: reps must be >= 1" >&2; exit 2; }

WORK=$(mktemp -d)
cleanup() { [ "${KRK_AB_KEEP:-0}" = "1" ] || rm -rf "$WORK"; }
trap cleanup EXIT

echo "sherlock A/B: hybrid kernel swap behind KRK_TIME_OP, interleaved"
echo "  binary   : $BIN"
echo "  model    : $MODEL"
echo "  prompt   : $PROMPT"
echo "  ctx=$CTX n=$N reps=$REPS"
echo "  arm A    : env ${A_SWITCH:-(stock)}"
echo "  arm B    : env $SWITCH"
echo

# One run. Writes <tag>.rc, <tag>.md5, <tag>.ok (tok/s), <tag>.ms (ms/step).
run_arm() {
    local tag="$1" extra="$2"
    local log="$WORK/$tag.err"
    # shellcheck disable=SC2086
    env $extra "$BIN" -m "$MODEL" -p "$PROMPT" -n "$N" --ctx "$CTX" --greedy \
        > "$WORK/$tag.out" 2> "$log"
    echo $? > "$WORK/$tag.rc"
    md5sum "$WORK/$tag.out" | cut -d' ' -f1 > "$WORK/$tag.md5"
    grep -E '^\[stats \] decode' "$log" | grep -oE '= [0-9.]+ tok/s' | tail -1 \
        | grep -oE '[0-9.]+' > "$WORK/$tag.ok"
    grep -E '^\[stats \] decode' "$log" | grep -oE '\([0-9.]+ ms/step\)' | tail -1 \
        | grep -oE '[0-9.]+' > "$WORK/$tag.ms"
    grep -E '^\[stats \] prefill' "$log" | grep -oE '= [0-9.]+ tok/s' | tail -1 \
        | grep -oE '[0-9.]+' > "$WORK/$tag.pf"
}

echo "  rep  order   arm  decode tok/s   ms/step   prefill tok/s   md5        rc"
invalid=0
r=1
while [ "$r" -le "$REPS" ]; do
    # Alternate which arm runs first: the first run of a pair pays for whatever
    # the previous pair left warm, and alternating cancels that instead of
    # attributing it to an arm.
    if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
    for arm in $order; do
        if [ "$arm" = "A" ]; then run_arm "r${r}a" "$A_SWITCH"; else run_arm "r${r}b" "$SWITCH"; fi
        tag="r${r}$(echo "$arm" | tr 'AB' 'ab')"
        rc=$(cat "$WORK/$tag.rc"); ok=$(cat "$WORK/$tag.ok"); ms=$(cat "$WORK/$tag.ms")
        pf=$(cat "$WORK/$tag.pf"); md5=$(cat "$WORK/$tag.md5")
        printf '  %-4s %-7s %-4s %-15s %-9s %-15s %-10s %s\n' \
               "$r" "$order" "$arm" "${ok:--}" "${ms:--}" "${pf:--}" "$(echo "$md5" | cut -c1-10)" "$rc"
        [ "$rc" -eq 0 ] && [ -n "$ok" ] || invalid=1
    done
    r=$((r + 1))
done
echo

if [ "$invalid" -ne 0 ]; then
    echo "verdict: INVALID -- a run failed or printed no decode timing"
    echo "         run dir kept: $WORK"
    exit 1
fi

# Text first: a speed comparison between two arms that produced different text is
# not a comparison of one implementation against another.
mismatch=0
r=1
while [ "$r" -le "$REPS" ]; do
    a=$(cat "$WORK/r${r}a.md5"); b=$(cat "$WORK/r${r}b.md5")
    [ "$a" = "$b" ] || mismatch=1
    [ "$r" -gt 1 ] || first_a="$a"
    [ "$a" = "${first_a}" ] && [ "$b" = "${first_a}" ] || mismatch=1
    r=$((r + 1))
done
if [ "$mismatch" -ne 0 ]; then
    echo "verdict: OUTPUT MISMATCH -- the arms' generated text differs, so no speed"
    echo "         number from this run means anything. md5s per rep:"
    r=1
    while [ "$r" -le "$REPS" ]; do
        echo "         rep $r  A $(cat "$WORK/r${r}a.md5")  B $(cat "$WORK/r${r}b.md5")"
        r=$((r + 1))
    done
    echo "         run dir kept: $WORK"
    exit 1
fi

# Paired percentage deltas, one per rep -- kept per ORDER as well. The order is
# not cosmetic: within a pair the two arms are separated by a model load, and the
# slot a run occupies is worth several percent. Measured by swapping the SAME arm
# between the slots on Qwen3-MoE-4x0.6B: A-first 124.5 tok/s against A-second
# 132.6, B-first 135.7 against B-second 142.1. A median over mixed orders
# therefore reports the slot effect as if it were the arm's, in whichever
# direction that invocation happened to run, so the verdict uses the
# order-balanced median below and prints the raw one beside it.
: > "$WORK/deltas"
: > "$WORK/deltas.ab"
: > "$WORK/deltas.ba"
r=1
while [ "$r" -le "$REPS" ]; do
    a=$(cat "$WORK/r${r}a.ok"); b=$(cat "$WORK/r${r}b.ok")
    d=$(awk -v a="$a" -v b="$b" 'BEGIN{if (a>0) printf "%.3f", 100.0*(b-a)/a}')
    printf '%s\n' "$d" >> "$WORK/deltas"
    if [ $((r % 2)) -eq 1 ]; then
        printf '%s\n' "$d" >> "$WORK/deltas.ab"
    else
        printf '%s\n' "$d" >> "$WORK/deltas.ba"
    fi
    r=$((r + 1))
done

median() { sort -n | awk '{v[NR]=$1} END{if(NR==0){print "";exit} print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
med=$(median < "$WORK/deltas")
awk -v m="$med" '{d=$1-m; if(d<0)d=-d; printf "%.3f\n", d}' "$WORK/deltas" | median > "$WORK/mad"
mad=$(cat "$WORK/mad")
# Order-balanced estimate. In an AB-order pair the second slot favours B, in a
# BA-order pair it favours A, so the two order medians carry it with opposite
# signs and their mean is free of it. A single-order run cannot do that, so it
# falls back to the raw median and the line below says so.
med_ab=$(median < "$WORK/deltas.ab")
med_ba=$(median < "$WORK/deltas.ba")
n_ab=$(wc -l < "$WORK/deltas.ab" | tr -d ' ')
n_ba=$(wc -l < "$WORK/deltas.ba" | tr -d ' ')
if [ -n "$med_ab" ] && [ -n "$med_ba" ]; then
    med_bal=$(awk -v x="$med_ab" -v y="$med_ba" 'BEGIN{printf "%.3f", (x+y)/2}')
else
    med_bal="$med"
fi
pos=0
r=1
while [ "$r" -le "$REPS" ]; do
    d=$(sed -n "${r}p" "$WORK/deltas")
    awk -v d="$d" 'BEGIN{exit !(d>0)}' && pos=$((pos + 1))
    r=$((r + 1))
done

echo "paired deltas (B vs A, %): $(tr '\n' ' ' < "$WORK/deltas")"
echo "median $med%   MAD $mad%   B faster in $pos/$REPS pairs"
if [ -n "$med_ab" ] && [ -n "$med_ba" ]; then
    echo "order split: A-first median ${med_ab}% ($n_ab pair(s))   B-first median ${med_ba}% ($n_ba pair(s))"
    echo "order-balanced median ${med_bal}%  <- the figure the verdict uses"
else
    echo "order-balanced median unavailable (every pair ran the same order) -- using the raw median"
fi
echo

sign_ok=0
[ "$pos" -ge $((REPS - 1)) ] || [ $((REPS - pos)) -ge $((REPS - 1)) ] && sign_ok=1
if [ "$sign_ok" -eq 1 ]; then
    verdict=$(awk -v m="$med_bal" -v mad="$mad" -v p="$pos" -v n="$REPS" 'BEGIN{
        if (m > mad && p >= n-1) print "WIN";
        else if (m < -mad && (n-p) >= n-1) print "REGRESSION";
        else print "NO DIFFERENCE";
    }')
else
    verdict="NO DIFFERENCE"
fi

case "$verdict" in
    WIN)
        echo "verdict: WIN -- B is faster in $pos/$REPS pairs by an order-balanced"
        echo "         median ${med_bal}% (raw $med%), which exceeds the spread of those"
        echo "         pairs (MAD $mad%). Text identical."
        rc=0 ;;
    REGRESSION)
        echo "verdict: REGRESSION -- B is slower by an order-balanced median ${med_bal}% (raw $med%, MAD $mad%)." >&2
        rc=1 ;;
    *)
        echo "verdict: NO DIFFERENCE -- the order-balanced median paired delta (${med_bal}%) does not clear"
        echo "         the spread of the pairs (MAD ${mad}%), so this run does not"
        echo "         support a win. That is the honest answer for an inert switch,"
        echo "         and the reason a single pair is not evidence."
        rc=0 ;;
esac

if [ "${KRK_AB_PROFILE:-0}" = "1" ]; then
    echo
    echo "profiled pair (op table only -- the profiler is an 8.7x slowdown here, so"
    echo "these absolute ms are diagnostics, not measurements):"
    for arm in A B; do
        if [ "$arm" = "A" ]; then extra="$A_SWITCH"; else extra="$SWITCH"; fi
        # shellcheck disable=SC2086
        env KRK_PROFILE=1 $extra "$BIN" -m "$MODEL" -p "$PROMPT" -n "$N" --ctx "$CTX" --greedy \
            > /dev/null 2> "$WORK/prof-$arm.err"
        line=$(grep -E '^gemm\(gemv\)' "$WORK/prof-$arm.err" | head -1)
        stats=$(grep -E '^\[stats \] decode' "$WORK/prof-$arm.err" | tail -1)
        echo "  arm $arm: $stats"
        echo "          ${line:-gemm(gemv) row not found}"
    done
fi

echo
[ "${KRK_AB_KEEP:-0}" = "1" ] && echo "run dir kept at $WORK"
exit "$rc"
