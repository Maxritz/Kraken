#!/usr/bin/env bash
# Re-runs the model sweep that fills docs/traces/results.tsv and the README's
# Scores table, with the flags those two documents quote:
#
#   kraken --model <MODEL> --bench --greedy --max-tokens 64 --ctx 4096 --chunk 512
#
# One run per model, device only. Writes a TSV with the columns those tables
# carry and prints a line per model as it goes, so the log is progress.
#
# Usage:  tools/sweep_models.sh [out.tsv]
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KR=${KR:-$ROOT/build-hip/kraken.exe}
OUT=${1:-${OUT:-/tmp/krk-sweep.tsv}}
DIRS=${DIRS:-/g/More-models /h/OLLAMA-Models/GGUF}

# The 21 rows the README and docs/MODEL-STATUS.md carry, in their order.
# MODELS may be overridden to re-run a subset.
MODELS=${MODELS:-"Laguna-S-2.1-PS-IQ2_XXS.gguf
K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf
laguna-xs2-Q4_K_M.gguf
LFM2.5-24B-A2B-QAD-Q4_0.gguf
Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf
Underdog-Saluki-27B-1.0-IQ2-mix-MTP-abliterated.gguf
omnicoder-9b-q6_k.gguf
Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf
qwen35-9b-instruct-nvfp4.gguf
lacuna-v1-Q4_K_M.gguf
Qwen3-8B-Q4_K_M.gguf
mythos-9b-unhinged-heretic.i1-Q4_K_M.gguf
Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M.gguf
Opus4.7-GODs.Ghost.Codex.Distill.4B-Q8_0.gguf
Qwen3-4B-Q8_0.gguf
VibeThinker-3B.Q8_0.gguf
Samastam-2.5B-Q8_0.gguf
Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M.gguf
smolcode-coder-cpp-3b-q4_k_m.gguf
smolcode-coder-cpp-1.5b-q4_k_m.gguf
bitnet-2b-tq1_0.gguf"}

find_model() {
  for d in $DIRS; do
    [ -f "$d/$1" ] && { printf '%s' "$d/$1"; return 0; }
  done
  return 1
}

printf 'model\tsize_MiB\tarch\tload_ms\tprefill_tps\tdecode_tps\tms_per_tok\thot_pct\tnote\n' > "$OUT"

for m in $MODELS; do
  f=$(find_model "$m") || { printf '%-58s MISSING\n' "$m"; continue; }
  sz=$(( $(stat -c %s "$f") / 1048576 ))
  "$KR" -m "$f" --bench --greedy --max-tokens 64 --ctx 4096 --chunk 512 \
      > /tmp/sweep.out 2> /tmp/sweep.err
  rc=$?
  arch=$(sed -n 's/.*(\([a-z0-9_]*\)): [0-9]* layers.*/\1/p' /tmp/sweep.err | head -1)
  load=$(grep -o 'load  *[0-9.]* ms' /tmp/sweep.out | tail -1 | grep -o '[0-9.]*')
  pre=$(grep -o 'prefill *[0-9]* tokens in [0-9.]* ms  ([0-9.]* tok/s)' /tmp/sweep.out | tail -1 | sed 's/.*(\([0-9.]*\) tok\/s)/\1/')
  dec=$(grep -o 'decode *[0-9]* tokens in [0-9.]* ms  ([0-9.]* tok/s)' /tmp/sweep.out | tail -1 | sed 's/.*(\([0-9.]*\) tok\/s)/\1/')
  # ms/tok is the decode step the tables quote, and the HOT share is the expert
  # cache's own line: `[stats ] cache N acquires | HOT hits M (p%) | ...`.
  mst=$(awk -v d="$dec" 'BEGIN { if (d > 0) printf "%.1f", 1000.0 / d; else print "-" }')
  hot=$(sed -n 's/.*HOT hits [0-9]* (\([0-9.]*\)%).*/\1/p' /tmp/sweep.out | head -1)
  note=""
  [ "$rc" = "2" ] && note="range guard fired"
  [ -z "$pre" ] && note="${note} refused or no bench line (rc=$rc)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$sz" "${arch:--}" \
         "${load:--}" "${pre:--}" "${dec:--}" "$mst" "${hot:--}" "$note" >> "$OUT"
  printf '%-58s rc=%-3s %8s MiB  load %8s ms  prefill %8s  decode %8s  ms/tok %6s  HOT %5s%%  %s\n' \
         "$m" "$rc" "$sz" "${load:--}" "${pre:--}" "${dec:--}" "$mst" "${hot:--}" "$note"
done
echo "wrote $OUT"
