#!/usr/bin/env bash
# Interleaved, order-balanced BYTE accounting for an expert-tier change.
#
#   scripts/kraken_router_bytes_ab.sh
#
# `kraken_sherlock_kernel_ab.sh` answers "is B faster", and refuses to answer at
# all when the two arms print different text -- which is the right default for a
# kernel swap and the wrong one for a routing change, because a routing change
# is SUPPOSED to be able to print different text. What it is supposed to change
# is the number of experts a token has to move, so this harness reads the
# counters out of each run's own stderr and compares those, with the speed as a
# side effect.
#
# Why the file/tier counters and not a wall clock: they are exact integers. Five
# runs of one arm here report byte-for-byte the same promotions and MiB/token
# (measured: 563 promotions / 283.65 MiB/tok five times), so ONE run per arm is
# enough for the byte claim and the repetitions exist for the tok/s figure.
#
# Arms: A is the stock run, B sets KRK_ROUTER_TIE_MARGIN (or any env you point
# KRK_ARB_SWITCH at). Order alternates inside each pair, like the kernel harness,
# because the slot a run occupies is worth several percent on this box.
#
# Environment:
#   MODEL, PROMPT, N, REPS, MARGIN   the workload (24 tokens, gpt-oss-120b, 11 GiB WARM)
#   COMMON   extra CLI shared by both arms; defaults to the target-shaped tier
#   KRK_ARB_SWITCH   the arm-B environment (default KRK_ROUTER_TIE_MARGIN=$MARGIN)
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_TEST_BIN:-$ROOT/build-hip/kraken.exe}
MODEL=${MODEL:-H:/OLLAMA-Models/GGUF/gpt-oss-120b-Q8_0.gguf}
PROMPT=${PROMPT:-"The capital of France is"}
N=${N:-24}
CTX=${CTX:-512}
REPS=${REPS:-5}
MARGIN=${MARGIN:-0.10}
SWITCH=${KRK_ARB_SWITCH:-KRK_ROUTER_TIE_MARGIN=$MARGIN}
COMMON=${COMMON:-"--expert-warmup 1 --expert-warm-mb 11000 --expert-pin-mb 8192"}

[ -x "$BIN" ] || { echo "FAIL: no binary at $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "FAIL: no model at $MODEL" >&2; exit 2; }
case "$REPS" in ''|*[!0-9]*) echo "FAIL: REPS '$REPS' is not a number" >&2; exit 2;; esac

read -r -a COMMON_ARGS <<< "$COMMON"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "router byte accounting: interleaved, order-balanced"
echo "  binary : $BIN"
echo "  model  : $MODEL"
echo "  ctx=$CTX n=$N reps=$REPS"
echo "  common : ${COMMON_ARGS[*]}"
echo "  arm A  : stock"
echo "  arm B  : $SWITCH"
echo

run() { # tag, "env assignments"
    local tag=$1 extra=$2
    # shellcheck disable=SC2086
    env $extra "$BIN" -m "$MODEL" -p "$PROMPT" -n "$N" --ctx "$CTX" --greedy \
        "${COMMON_ARGS[@]}" > "$WORK/$tag.out" 2> "$WORK/$tag.err"
    echo $? > "$WORK/$tag.rc"
    md5sum "$WORK/$tag.out" | cut -d' ' -f1 > "$WORK/$tag.md5"
    grep -E '^\[stats \] decode' "$WORK/$tag.err" \
        | grep -oE '= [0-9.]+ tok/s' | tail -1 | grep -oE '[0-9.]+' > "$WORK/$tag.tps"
    grep -E '^\[stats \] transfer' "$WORK/$tag.err" \
        | grep -oE '^\[stats \] transfer  [0-9]+ promotions' \
        | grep -oE '[0-9]+' > "$WORK/$tag.prom"
    grep -E '^\[stats \] transfer' "$WORK/$tag.err" \
        | grep -oE '\([0-9.]+ MiB/tok\)' | grep -oE '[0-9.]+' > "$WORK/$tag.mib"
    # Only the substitution line: with the preference off the router block
    # still prints its VRAM-hit share, and reading that as a
    # substitution rate says 88% of picks were moved when none were.
    grep -E '^\[stats \] router    [0-9]+ picks' "$WORK/$tag.err" \
        | grep -oE '\([0-9.]+%\)' | head -1 | grep -oE '[0-9.]+' > "$WORK/$tag.sub"
    [ -s "$WORK/$tag.sub" ] || echo '-' > "$WORK/$tag.sub"
}

printf '  rep  order  arm   tok/s    promotions  MiB/tok  subs%%  text\n'
r=1
while [ "$r" -le "$REPS" ]; do
    if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
    for arm in $order; do
        tag="r${r}$(echo "$arm" | tr 'AB' 'ab')"
        if [ "$arm" = "A" ]; then run "$tag" ""; else run "$tag" "$SWITCH"; fi
        printf '  %-4s %-6s %-5s %-9s %-11s %-9s %-6s %s\n' "$r" "$order" "$arm" \
            "$(cat "$WORK/$tag.tps")" "$(cat "$WORK/$tag.prom")" \
            "$(cat "$WORK/$tag.mib")" "$(cat "$WORK/$tag.sub")" \
            "$(cut -c1-10 < "$WORK/$tag.md5")"
    done
    r=$((r + 1))
done

for k in a b; do
    : > "$WORK/$k.tps"; : > "$WORK/$k.prom"; : > "$WORK/$k.mib"
done
r=1
while [ "$r" -le "$REPS" ]; do
    for k in a b; do
        cat "$WORK/r${r}$k.tps" >> "$WORK/$k.tps"
        cat "$WORK/r${r}$k.prom" >> "$WORK/$k.prom"
        cat "$WORK/r${r}$k.mib" >> "$WORK/$k.mib"
    done
    r=$((r + 1))
done

med() { sort -n "$1" | awk '{a[NR]=$1} END {if (NR%2) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}'; }
A_TPS=$(med "$WORK/a.tps"); B_TPS=$(med "$WORK/b.tps")
A_PROM=$(med "$WORK/a.prom"); B_PROM=$(med "$WORK/b.prom")
A_MIB=$(med "$WORK/a.mib");  B_MIB=$(med "$WORK/b.mib")

echo
echo "arm A (stock)     median: $A_TPS tok/s  $A_PROM promotions  $A_MIB MiB/tok"
echo "arm B ($SWITCH)   median: $B_TPS tok/s  $B_PROM promotions  $B_MIB MiB/tok"
awk -v a="$A_MIB" -v b="$B_MIB" 'BEGIN { if (a > 0) printf "bytes/token : %+.1f%%\n", 100.0*(b-a)/a }'
awk -v a="$A_PROM" -v b="$B_PROM" 'BEGIN { if (a > 0) printf "promotions  : %+.1f%%\n", 100.0*(b-a)/a }'
awk -v a="$A_TPS" -v b="$B_TPS" 'BEGIN { if (a > 0) printf "decode      : %+.1f%%\n", 100.0*(b-a)/a }'
if cmp -s "$WORK/r1a.md5" "$WORK/r1b.md5"; then
    echo "text        : identical between the arms in rep 1"
else
    echo "text        : DIFFERS -- expected for a routing change; the byte and rate"
    echo "              figures still describe what each arm did, but read the"
    echo "              generated text before calling the faster arm the better one"
fi
