#!/usr/bin/env bash
# Temporary interleaved, order-balanced A/B for the automatic pinned WARM prefix.
# arm "auto" = default (pinned prefix on a small-BAR card), arm "off" = pageable.
set -u
M=${M:-H:/OLLAMA-Models/GGUF/Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf}
N=${N:-16}
OUT=${OUT:-/tmp/ab_pin}
REPS=${1:-3}
mkdir -p "$OUT"
for i in $(seq 1 "$REPS"); do
  if [ $(( i % 2 )) -eq 1 ]; then order="auto off"; else order="off auto"; fi
  for arm in $order; do
    if [ $arm = off ]; then E="KRK_PIN_AUTO=0"; else E="KRK_PIN_AUTO=1"; fi
    env $E KRK_ARCH_ALLOW_EXPERIMENTAL=1 timeout 900 ./build-hip/kraken.exe \
      -m "$M" -p "The capital of France is" -n "$N" --greedy \
      > "$OUT/run_${i}_${arm}.log" 2>&1
    rc=$?
    ms=$(grep -o "([0-9.]* ms/step)" "$OUT/run_${i}_${arm}.log" | tr -dc '0-9.')
    pin=$(grep -o "WARM [0-9]* MiB ([0-9]* pinned" "$OUT/run_${i}_${arm}.log" | grep -o "([0-9]*" | tr -dc '0-9')
    us=$(grep -o "[0-9.]* us/expert" "$OUT/run_${i}_${arm}.log" | tr -dc '0-9.')
    echo "$i $arm rc=$rc ms=$ms pinned=${pin:-0}MiB us=$us"
  done
done
