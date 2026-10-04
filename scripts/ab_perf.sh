#!/usr/bin/env bash
# Interleaved A/B for a change that is supposed to be faster.
#
# Arms alternate run by run: page cache and driver state warm between runs, so
# a sequential A-then-B probe measures the warmup, not the change. Always kills
# kraken -- a leaked engine holds 15 GiB of VRAM and the next arm fails for the
# wrong reason.
#
# Usage:
#   scripts/ab_perf.sh <model> <reps> "<env arm A>" "<env arm B>" ...
#
# Reports median ms/tok and tok/s per arm from the engine's own decode line, and
# keeps every log under $OUT so a winner that changed the output is visible.
set -uo pipefail

MODEL=${1:?model path}
REPS=${2:?repetitions}
shift 2
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] || { echo "need at least one env arm" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KR=${KR:-$ROOT/build-hip/kraken.exe}
NTOK=${NTOK:-24}
PROMPT=${PROMPT:-what is the capital of france?}
OUT=${OUT:-/tmp/krk-ab}
mkdir -p "$OUT"

cleanup() { taskkill //F //IM kraken.exe //T >/dev/null 2>&1 || true; }
trap cleanup EXIT

med() { # median of the numbers on stdin
  sort -g | awk '{v[NR]=$1} END {
    if (NR == 0) { print "n/a"; exit }
    if (NR % 2) printf "%.2f\n", v[(NR+1)/2];
    else printf "%.2f\n", (v[NR/2]+v[NR/2+1])/2 }'
}

for ((r = 1; r <= REPS; r++)); do
  for a in "${!ARMS[@]}"; do
    arm=${ARMS[$a]}
    # "VAR=1 :: --expert-stub" splits the arm into env assignments and CLI flags,
    # so a flag-shaped difference can be A/B'd with the same interleaving.
    envpart=${arm%%::*}
    flagpart=${arm#*::}
    [ "$flagpart" = "$arm" ] && flagpart=""
    tag="r${r}-a${a}"
    # shellcheck disable=SC2086
    env $envpart timeout 900 "$KR" -m "$MODEL" \
      --prompt "$PROMPT" --max-tokens "$NTOK" --temp 0 --greedy --chat \
      $flagpart > "$OUT/$tag.log" 2> "$OUT/$tag.err"
    rc=$?
    line=$(grep -o 'decode *[0-9]* tok in [0-9.]* ms = [0-9.]* tok/s ([0-9.]* ms/tok)' \
           "$OUT/$tag.err" | tail -1)
    hash=$(tr -d '\r' < "$OUT/$tag.log" | md5sum | cut -c1-8)
    printf '%-9s %-30s rc=%-3s %s  %s\n' "$tag" "$arm" "$rc" "$hash" "$line"
    taskkill //F //IM kraken.exe //T >/dev/null 2>&1
    sleep 1
  done
done

echo "--- median over $REPS interleaved runs ---"
for ((a = 0; a < ${#ARMS[@]}; a++)); do
  ms=$(for ((r = 1; r <= REPS; r++)); do
          grep -o '([0-9.]* ms/tok)' "$OUT/r${r}-a${a}.err" | tail -1 |
            tr -d '() ms/tok'
        done | med)
  tps=$(for ((r = 1; r <= REPS; r++)); do
          grep -o '= [0-9.]* tok/s (' "$OUT/r${r}-a${a}.err" | tail -1 |
            sed 's/= //; s/ tok\/s (//'
        done | med)
  distinct=$(for ((r = 1; r <= REPS; r++)); do
               tr -d '\r' < "$OUT/r${r}-a${a}.log" | md5sum | cut -c1-8
             done | sort -u | wc -l)
  printf 'a%s %-30s %8s ms/tok  %6s tok/s  (%s distinct outputs)\n' \
         "$a" "${ARMS[$a]}" "$ms" "$tps" "$distinct"
done