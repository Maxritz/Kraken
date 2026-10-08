#!/bin/bash
# Profile test runner - single run per model, killed before next starts
# Models known to be runnable from inspection

set -e

KRAKEN="./build-hip/kraken"
OUTDIR="./profile_results"
TIMEOUT=30  # seconds max per model

mkdir -p "$OUTDIR"

# Runnable models to test (from inspection above)
MODELS=(
    "G:/More-models/Qwen3-4B-Q8_0.gguf"
    "G:/More-models/Qwen3-8B-Q4_K_M.gguf"
    "models/Qwen3.5-0.8B.Q4_K_M.gguf"
    "H:/OLLAMA-Models/GGUF/Qwen3.5-9B-Q4_K_M.gguf"
    "G:/More-models/Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M.gguf"
    "G:/More-models/laguna-xs2-Q4_K_M.gguf"
    "H:/OLLAMA-Models/GGUF/Laguna-XS.2-IQ4_XS.gguf"
)

# Test prompt - short to keep runs quick
PROMPT="The capital of France is"

echo "=============================================="
echo "Profile Tests - Single run per model"
echo "=============================================="
echo ""

for model in "${MODELS[@]}"; do
    name=$(basename "$model" .gguf)
    outfile="$OUTDIR/${name}_profile.txt"
    
    echo "=============================================="
    echo "Testing: $name"
    echo "Model:  $model"
    echo "Output: $outfile"
    echo "=============================================="
    
    # Kill any existing kraken process
    pkill -9 -f "kraken.*$model" 2>/dev/null || true
    sleep 0.5
    
    # Run with --profile, capture output, timeout after $TIMEOUT seconds
    # Use --max-tokens to limit generation
    echo "Starting run..."
    start_time=$(date +%s)
    
    timeout $TIMEOUT "$KRAKEN" \
        --model "$model" \
        --prompt "$PROMPT" \
        --max-tokens 32 \
        --greedy \
        --profile \
        > "$outfile" 2>&1
    
    rc=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "Return code: $rc"
    echo "Wall time:   ${duration}s"
    echo ""
    
    # Show the profile section if it exists
    if grep -q "\[profile\]" "$outfile" 2>/dev/null; then
        echo "--- Profile Output ---"
        grep -A 100 "\[profile\]" "$outfile" | head -80
        echo "..."
    elif grep -q "profile" "$outfile" 2>/dev/null; then
        echo "--- Profile-related output ---"
        grep -i "profile\|time\|ms\|kernel" "$outfile" | head -30
    else
        echo "--- Last 50 lines ---"
        tail -50 "$outfile"
    fi
    
    echo ""
    echo "=== Done with $name ==="
    echo ""
    
    # Kill any lingering process
    pkill -9 -f "kraken.*$model" 2>/dev/null || true
    sleep 0.5
done

echo "=============================================="
echo "All profile tests complete"
echo "Results in: $OUTDIR/"
echo "=============================================="
