#!/usr/bin/env bash
# Interleaved prefill probe for the prefill GEMM work.
#
# Runs each model once per repetition per arm, alternating arms run by run, and
# prints the engine's own `[stats ] prefill` line. Interleaving matters for the
# same reason scripts/ab_perf.sh interleaves: a sequential A-then-B reads the
# page cache, not the change.
#
# Usage:
#   tools/prefill_probe.sh <reps> "<env arm A>" "<env arm B>" ...
#
# The prompt is fixed (a repeated sentence, ~600 tokens on these files) and the
# run is `-n 1`, so the measured number is prefill and nothing after it.
set -uo pipefail

REPS=${1:?repetitions}
shift
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] || { echo "need at least one arm" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KR=${KR:-$ROOT/build-hip/kraken.exe}
OUT=${OUT:-/tmp/krk-prefill}
MODELS=${MODELS:-"/g/More-models/Qwen3-8B-Q4_K_M.gguf /g/More-models/Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf /g/More-models/bitnet-2b-tq1_0.gguf"}
PROMPT=${PROMPT:-$(python3 -c 'print("The capital of France is "*2*50, end="")')}

rm -rf "$OUT"
mkdir -p "$OUT"

cleanup() { taskkill //F //IM kraken.exe //T >/dev/null 2>&1 || true; }
trap cleanup EXIT

for ((r = 1; r <= REPS; r++)); do
  for a in "${!ARMS[@]}"; do
    arm=${ARMS[$a]}
    envpart=${arm%%::*}
    flagpart=${arm#*::}
    [ "$flagpart" = "$arm" ] && flagpart=""
    # KR_BIN=<path> in an arm compares two BINARIES interleaved, which is how a
    # kernel change is A/B'd: the two arms must differ in one thing, and an env
    # switch that keeps both code paths alive differs in register allocation too.
    K=$(printf '%s' "$envpart" | tr ' ' '\n' | sed -n 's/^KR_BIN=//p')
    K=${K:-$KR}
    for m in $MODELS; do
      tag="r${r}-a${a}-$(basename "$m" .gguf)"
      # shellcheck disable=SC2086
      env $envpart timeout 900 "$K" -m "$m" \
        --prompt "$PROMPT" --max-tokens 1 --temp 0 --greedy $flagpart \
        > "$OUT/$tag.log" 2> "$OUT/$tag.err"
      rc=$?
      line=$(grep -o 'prefill *[0-9]* tok in [0-9.]* ms = [0-9.]* tok/s' \
             "$OUT/$tag.err" | tail -1)
      printf '%-40s rc=%-3s %s\n' "$tag" "$rc" "$line"
      taskkill //F //IM kraken.exe //T >/dev/null 2>&1
    done
  done
done

echo "--- median prefill tok/s over $REPS interleaved runs ---"
for a in "${!ARMS[@]}"; do
  arm=${ARMS[$a]}
  for m in $MODELS; do
    b=$(basename "$m" .gguf)
    v=$(for ((r = 1; r <= REPS; r++)); do
          # `prefill` and not just `= N tok/s`: the decode line matches the
          # shorter pattern too and it is the LAST match in the file, so a
          # bare `tail -1` reports the decode rate and reads as 0.0 tok/s on
          # a one-token run.
          grep -o 'prefill.*= [0-9.]* tok/s' "$OUT/r${r}-a${a}-${b}.err" |
            tail -1 | sed 's/.*= //; s/ tok\/s//'
        done | sort -g | awk '{v[NR]=$1} END {
          if (NR == 0) { print "n/a"; exit }
          if (NR % 2) printf "%.1f", v[(NR+1)/2];
          else printf "%.1f", (v[NR/2]+v[NR/2+1])/2 }')
    printf 'a%-3s %-34s %-42s %s tok/s\n' "$a" "$arm" "$b" "$v"
  done
done
