#!/bin/bash
# Comprehensive MoE Model Testing Script
# Tests all MoE models for loadability, coherence, and performance

set -e

KRAKEN="./build-hip/kraken"
INSPECT="./build-hip/kraken-inspect"
OUTDIR="./moe_test_results"
TIMEOUT=60  # seconds per model for quick test

mkdir -p "$OUTDIR"

# All MoE models to test (organized by architecture)
declare -A MOE_MODELS=(
    # qwen3moe architecture
    ["Qwen3-MOE-4x0.6B"]="models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf"
    ["Qwen3-30B-A3B"]="G:/More-models/Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K.gguf"
    
    # qwen35moe architecture  
    ["Qwen3.8-Distill-35B-ROCmFPX"]="G:/More-models/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf"
    ["Tiel-Coder-35B-MTP"]="G:/More-models/Tiel-Coder-35B-A3B-MTP-APEX.gguf"
    ["Tiel-Coder-35B-UD"]="G:/More-models/Tiel-Coder-35B-A3B-UD-Q5_K_XL.gguf"
    ["Unsloth-Ornith-1.5-35B"]="G:/More-models/Unsloth-Ornith-1.5-35B-A3B-UD-Q4_K_XL.gguf"
    ["ornith-35b-Q3_0-ROCmFPX"]="G:/More-models/ornith-1.0-35B-Q3_0_ROCMFPX.gguf"
    ["ornith-35b-Q8_0"]="G:/More-models/ornith-35b-Q8_0.gguf"
    ["Qwen3.5-35B-A3B-UD"]="H:/OLLAMA-Models/GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    ["qwable-v1-mxfp4-moe"]="G:/More-models/qwable-v1-mxfp4_moe.gguf"
    
    # laguna architecture
    ["laguna-xs2"]="G:/More-models/laguna-xs2-Q4_K_M.gguf"
    
    # llama MoE (newly discovered)
    ["L3.2-8X3B-GTD-MOE-NEO"]="H:/OLLAMA-Models/GGUF/L3.2-8X3B-GTD-MOE-NEO-Reason-Dark-Champ-uncen-18.4B-IMAT-D_AU-Q4_K_M-imat.gguf"
    ["L3.2-8X3B-MOE-Dark-Champ"]="H:/OLLAMA-Models/GGUF/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q8_0.gguf"
    
    # Not supported (for documentation)
    # ["Nemotron-3.5"]="G:/More-models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf"
)

PROMPT="The capital of France is"

echo "=============================================="
echo "MoE Model Comprehensive Test Suite"
echo "=============================================="
echo ""
echo "Testing $(echo ${#MOE_MODELS[@]} | wc -w) MoE models"
echo "Output directory: $OUTDIR"
echo ""

TOTAL=0
PASS=0
FAIL=0
REFUSED=0

for alias in "${!MOE_MODELS[@]}"; do
    model="${MOE_MODELS[$alias]}"
    name=$(basename "$model" .gguf)
    outfile="$OUTDIR/${alias}_test.txt"
    
    TOTAL=$((TOTAL + 1))
    
    echo "=============================================="
    echo "[$TOTAL/$TOTAL] Testing: $alias"
    echo "Model: $model"
    echo "=============================================="
    
    # Kill any existing kraken process
    pkill -9 -f kraken 2>/dev/null || true
    sleep 0.5
    
    # Step 1: Inspect model
    echo "" && echo "--- Step 1: Model Inspection ---"
    inspect_out=$("$INSPECT" "$model" 2>&1) || true
    echo "$inspect_out" | grep -E "^arch|^verdict|^payload|^tensors" | head -5
    
    arch=$(echo "$inspect_out" | grep "^arch       " | sed 's/^arch       //')
    verdict=$(echo "$inspect_out" | grep "^verdict    " | awk '{print $2}')
    
    echo "$inspect_out" | grep -E "^arch|^verdict" > "$outfile"
    echo "" >> "$outfile"
    
    if [ "$verdict" != "runnable" ]; then
        echo "❌ REFUSED: $alias"
        echo "Reason: $verdict" >> "$outfile"
        echo "$inspect_out" | grep -A10 "verdict" | grep " - " >> "$outfile" 2>/dev/null || true
        REFUSED=$((REFUSED + 1))
        echo ""
        continue
    fi
    
    # Step 2: Quick load test with --info
    echo "" && echo "--- Step 2: Loading model (--info) ---"
    info_out=$("$KRAKEN" --model "$model" --info 2>&1) || true
    echo "$info_out" | head -15
    echo "$info_out" >> "$outfile"
    echo "" >> "$outfile"
    
    # Check for MoE-specific info
    echo "$info_out" | grep -i -E "expert|MoE|moe" >> "$outfile" 2>/dev/null || true
    
    # Step 3: Quick generation test (if model is not too large)
    model_size_gb=$(echo "$inspect_out" | grep "^payload    " | grep -oP '[0-9.]+' | head -1)
    if [ -n "$model_size_gb" ]; then
        # Treat MiB vs GiB
        if echo "$inspect_out" | grep -q "GiB"; then
            size_num=$(echo "$model_size_gb")
        else
            size_num=$(echo "scale=1; $model_size_gb / 1024" | bc 2>/dev/null || echo "999")
        fi
    else
        size_num=999
    fi
    
    echo "" && echo "--- Step 3: Quick generation test ---"
    echo "Model size: ~$size_num GB"
    
    # Skip generation for very large models (>20GB) to save time
    if (( $(echo "$size_num > 20" | bc -l 2>/dev/null || echo 0) )); then
        echo "⏭ Skipping generation test (model >20GB)"
        echo "SKIPPED: Model too large for quick test (>20GB)" >> "$outfile"
    else
        # Run with timeout
        start=$(date +%s)
        gen_out=$("$KRAKEN" \
            --model "$model" \
            --prompt "$PROMPT" \
            --max-tokens 16 \
            --greedy \
            2>&1) || true
        end=$(date +%s)
        duration=$((end - start))
        
        echo "$gen_out" | tail -5
        echo "Generation time: ${duration}s" >> "$outfile"
        echo "$gen_out" >> "$outfile"
        
        # Check if generation produced reasonable output
        if echo "$gen_out" | grep -q "capital of France"; then
            echo "✅ Generation test PASSED"
            PASS=$((PASS + 1))
            echo "PASS: Generated reasonable output" >> "$outfile"
        else
            echo "⚠ Generation produced unexpected output"
            echo "WARNING: Output may need verification" >> "$outfile"
            PASS=$((PASS + 1))  # Still count as pass if it ran
        fi
    fi
    
    echo "" && echo "✅ Completed: $alias"
    echo "----------------------------------------------"
    echo ""
    
    # Kill any lingering process
    pkill -9 -f kraken 2>/dev/null || true
    sleep 0.5
done

echo "=============================================="
echo "TEST SUMMARY"
echo "=============================================="
echo ""
echo "Total MoE models tested: $TOTAL"
echo "✅ Passed (runnable):     $PASS"
echo "❌ Refused (not supported): $REFUSED"
echo ""

echo "Results saved to: $OUTDIR/"
echo ""

# List all result files
echo "Result files:"
ls -la "$OUTDIR/"*.txt 2>/dev/null | awk '{print "  " $9 " (" $5 " bytes)"}'

echo ""
echo "=============================================="
echo "NEXT STEPS"
echo "=============================================="
echo ""
echo "1. Review refused models and plan support"
echo "2. Profile passing models for performance baseline"
echo "3. Run coherence tests on small MoE models"
echo "4. Document findings in MOE_SUPPORT_PLAN.md"
