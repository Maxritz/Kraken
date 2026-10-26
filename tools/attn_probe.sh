#!/usr/bin/env bash
# Interleaved probe for the prefill attention kernel.
#
# Runs each arm once per repetition, alternating arms run by run, and prints the
# engine's own `attention` op row (calls, NET ms) beside the prefill rate.
# Interleaving matters here for the same reason tools/prefill_probe.sh
# interleaves: the arms differ only in an env knob, so a sequential A-then-B
# reads drift, not the knob.
#
# Usage:
#   tools/attn_probe.sh <reps> "<env arm A>" "<env arm B>" ...
#
# The prompt is fixed (a repeated sentence, 501 tokens on Qwen3-8B) and the run
# is `-n 1`, so nothing after prefill is measured.
set -uo pipefail

REPS=${1:?repetitions}
shift
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] || { echo "need at least one arm" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KR=${KR:-$ROOT/build-hip/kraken.exe}
OUT=${OUT:-/tmp/krk-attn}
MODELS=${MODELS:-"/g/More-models/Qwen3-8B-Q4_K_M.gguf"}
PROMPT=${PROMPT:-$(python3 -c 'print("The capital of France is "*100, end="")')}

rm -rf "$OUT"
mkdir -p "$OUT"

cleanup() { taskkill //F //IM kraken.exe //T >/dev/null 2>&1 || true; }
trap cleanup EXIT

for ((r = 1; r <= REPS; r++)); do
  for a in "${!ARMS[@]}"; do
    arm=${ARMS[$a]}
    K=$(printf '%s' "$arm" | tr ' ' '\n' | sed -n 's/^KR_BIN=//p')
    K=${K:-$KR}
    for m in $MODELS; do
      tag="r${r}-a${a}-$(basename "$m" .gguf)"
      env $arm KRK_TIME=attention timeout 900 "$K" -m "$m" \
        --prompt "$PROMPT" --max-tokens 1 --temp 0 --greedy \
        > "$OUT/$tag.log" 2>&1
      rc=$?
      pre=$(grep -o 'prefill *[0-9]* tok in [0-9.]* ms = [0-9.]* tok/s' \
            "$OUT/$tag.log" | tail -1 | sed 's/.*= //')
      att=$(awk '$1=="attention"' "$OUT/$tag.log" | tail -1 |
            awk '{printf "calls=%s net=%s ms", $2, $5}')
      printf '%-46s rc=%-3s %-28s prefill %s\n' \
             "$tag" "$rc" "$att" "${pre:--}"
      taskkill //F //IM kraken.exe //T >/dev/null 2>&1
    done
  done
done

echo "--- median over $REPS interleaved runs ---"
for a in "${!ARMS[@]}"; do
  for m in $MODELS; do
    b=$(basename "$m" .gguf)
    med=$(for ((r = 1; r <= REPS; r++)); do
            awk '$1=="attention"' "$OUT/r${r}-a${a}-${b}.log" | tail -1 |
              awk '{print $5}'
          done | sort -g | awk '{v[NR]=$1} END {
            if (NR == 0) { print "n/a"; exit }
            if (NR % 2) printf "%.2f", v[(NR+1)/2];
            else printf "%.2f", (v[NR/2]+v[NR/2+1])/2 }')
    printf 'a%-3s %-34s %-24s attention net %s ms\n' \
           "$a" "${ARMS[$a]}" "$b" "$med"
  done
done
