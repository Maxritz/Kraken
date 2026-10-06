#!/usr/bin/env bash
# Collects the numbers that go into test-results.md.
set -u
cd /c/Users/rr/OneDrive/Desktop/kraken
BIN=./build-hip/kraken.exe
OUT=/tmp/measure.out
: > "$OUT"

run() {
  echo "===== $* =====" >> "$OUT"
  "$@" >> "$OUT" 2>&1
  echo "[rc=$?]" >> "$OUT"
}

# 1. prefill / token-gen, with the engine's own GPU-engine sampling
run $BIN -m models/SmolLM2-135M-Instruct.Q4_K_M.gguf --bench --greedy
run $BIN -m models/Qwen3.5-0.8B.Q4_K_M.gguf --bench --greedy
run $BIN -m models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --bench --greedy
run $BIN -m "G:/More-models/Qwen3-8B-Q4_K_M.gguf" --bench --greedy

# 2. speculative decoding: baseline vs draft (same-tokenizer pair)
run $BIN -m "G:/More-models/Qwen3-8B-Q4_K_M.gguf" --bench --greedy
run $BIN -m "G:/More-models/Qwen3-8B-Q4_K_M.gguf" --draft models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --draft-tokens 4 --bench --greedy

# 3. KV tiering: flat, then a deliberately small HOT tier that must page
run $BIN -m models/SmolLM2-135M-Instruct.Q4_K_M.gguf --bench --greedy
run $BIN -m models/SmolLM2-135M-Instruct.Q4_K_M.gguf --ctx 4096 --kv-hot-mb 8 --kv-warm-mb 64 --bench --greedy
run $BIN -m models/SmolLM2-135M-Instruct.Q4_K_M.gguf --ctx 4096 --kv-hot-mb 1 --kv-warm-mb 64 --bench --greedy

echo "=== DONE ===" >> "$OUT"
echo done > /tmp/measure.done
