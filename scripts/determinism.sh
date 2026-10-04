#!/usr/bin/env bash
# Interleaved determinism A/B.
#
# Every configuration that does NOT slow the pipeline was observed to diverge on
# laguna-xs2, so "run it twice and diff" is not enough to establish a fix, and a
# sequential probe cannot be trusted either: the arms alternate run by run so
# page cache and driver state are warm for both.
#
# Usage:
#   scripts/determinism.sh <model> <reps> "<env arm A>" "<env arm B>" ...
#
# Each arm is a string of VAR=VALUE assignments prefixed to the run. The output
# of every run is kept under $OUT so the divergent text can be read, not just
# hashed. Always kills kraken: a leaked engine holds VRAM and the next run of
# the arm fails for the wrong reason.
set -uo pipefail

MODEL=${1:?model path}
REPS=${2:?repetitions}
shift 2
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] || { echo "need at least one env arm" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KR=${KR:-$ROOT/build-hip/kraken.exe}
OUT=${OUT:-/tmp/krk-det}
mkdir -p "$OUT"

cleanup() { taskkill //F //IM kraken.exe //T >/dev/null 2>&1 || true; }
trap cleanup EXIT

for ((r = 1; r <= REPS; r++)); do
  for a in "${!ARMS[@]}"; do
    arm=${ARMS[$a]}
    tag="r${r}-a${a}"
    log="$OUT/$tag.log"
    # shellcheck disable=SC2086
    env $arm timeout 600 "$KR" -m "$MODEL" \
      --prompt "what is the capital of france?" --max-tokens 24 --temp 0 \
      --greedy --chat > "$log" 2> "$OUT/$tag.err"
    rc=$?
    hash=$(tr -d '\r' < "$log" | md5sum | cut -c1-8)
    printf '%-10s %-34s rc=%-3s %s\n' "$tag" "$arm" "$rc" "$hash"
    taskkill //F //IM kraken.exe //T >/dev/null 2>&1
    sleep 1
  done
done

echo "--- distinct texts per arm ---"
for ((a = 0; a < ${#ARMS[@]}; a++)); do
  n=$(for ((r = 1; r <= REPS; r++)); do
        tr -d '\r' < "$OUT/r${r}-a${a}.log" | md5sum | cut -c1-8
      done | sort -u | wc -l)
  printf 'a%s %-34s %s distinct of %s\n' "$a" "${ARMS[$a]}" "$n" "$REPS"
done