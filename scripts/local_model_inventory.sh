#!/usr/bin/env bash
set -euo pipefail

# This is intentionally NOT a PowerShell command. This workspace cannot run the
# PowerShell that would read G:/More-models or H:/OLLAMA-Models/GGUF, so this
# script records the inventory the same way the rest of this worktree does: as a
# file the user can inspect and refresh.

SEARCH_ROOTS=(
  "G:/More-models"
  "H:/OLLAMA-Models/GGUF"
)

TARGETS=(
  "Qwen3-4B-Q8_0.gguf"
  "Qwen3.5-9B-Q4_K_M.gguf"
  "Qwen3-8B-Q4_K_M.gguf"
  "Qwen3-4B-Instruct-2507-Q6_K.gguf"
  "Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf"
  "Qwen3.5-0.8B.Q4_K_M.gguf"
  "SmolLM2-135M-Instruct.Q4_K_M.gguf"
  "Qwen3-30B-A3B*.gguf"
  "laguna-xs2-Q4_K_M.gguf"
  "Ornith-1.0-9b*.gguf"
  "Tiel-Coder-35B-A3B*.gguf"
  "omnicoder-9b-q6_k.gguf"
  "Qwen2.5-Coder-7B*.gguf"
  "Phi-3.5-mini*.gguf"
  "smolcode-coder-cpp-1.5b*.gguf"
  "smolcode-coder-cpp-3b*.gguf"
  "Samastam-2.5B-Q8_0.gguf"
  "VibeThinker-3B.Q8_0.gguf"
  "qwen3vl-4b-q4_k_m.gguf"
  "Spark_one.Q6_K.gguf"
  "lacuna-v1-Q4_K_M.gguf"
  "Q3.5-9B-GLM-5.1-DA*.gguf"
  "qwable-v1-mxfp4_moe.gguf"
  "qwen3.5-9b-nvfp4*.gguf"
  "qwen35-9b-instruct-nvfp4*.gguf"
  "qwen3-1.7b-stem-proof-f16.gguf"
  "qwen3-1.7b-coder-distilled-sft-f16.gguf"
  "mythos-9b-unhinged-heretic.i1-Q4_K_M.gguf"
  "Unsloth-Ornith-1.5-35B-A3B*.gguf"
  "ornith-35b-Q8_0.gguf"
  "Qwen2.5-Coder-32B*.gguf"
)

OUT="$(dirname "$0")/local_model_inventory.txt"

{
  echo "# local model inventory - regenerate this file by running this script"
  echo "# searched roots:"
  for r in "${SEARCH_ROOTS[@]}"; do
    echo "#   $r"
  done
  echo "#"
  for t in "${TARGETS[@]}"; do
    found=false
    for r in "${SEARCH_ROOTS[@]}"; do
      if [[ -d "$r" ]]; then
        while IFS= read -r -d '' path; do
          echo "FOUND | $path"
          found=true
        done < <(find "$r" -maxdepth 4 -iname "$t" -type f -print0 2>/dev/null | sort -z)
      fi
    done
    if [[ "$found" == "false" ]]; then
      echo "MISSING | $t"
    fi
  done
} > "$OUT"

echo "wrote $OUT"
wc -l < "$OUT"
