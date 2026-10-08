#!/bin/bash
# Model support scanner for Kraken
# Tests which models load and what the verdict is

set -e

INSPECT="./build-hip/kraken-inspect"
MODELS_DIRS=(
    "models"
    "G:/More-models"
    "H:/OLLAMA-Models/GGUF"
)

echo "=============================================="
echo "Kraken Model Support Scanner"
echo "=============================================="
echo ""
echo "Inspecting models from:"
for d in "${MODELS_DIRS[@]}"; do
    echo "  - $d"
done
echo ""

total=0
runnable=0
refused=0
unreadable=0

for dir in "${MODELS_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "SKIP: $dir (not accessible)"
        continue
    fi
    
    echo "================================================"
    echo "Directory: $dir"
    echo "================================================"
    
    for model in "$dir"/*.gguf; do
        [ -f "$model" ] || continue
        total=$((total + 1))
        
        name=$(basename "$model")
        size=$(du -h "$model" 2>/dev/null | cut -f1)
        
        echo ""
        echo "----------------------------------------"
        echo "File: $name ($size)"
        echo "----------------------------------------"
        
        # Run kraken-inspect
        output=$("$INSPECT" "$model" 2>&1) || true
        
        # Extract key info
        arch=$(echo "$output" | grep "^arch       " | sed 's/^arch       //')
        verdict=$(echo "$output" | grep "^verdict    " | sed 's/^verdict    //')
        
        echo "  Arch:   $arch"
        echo "  Verdict: $verdict"
        
        # Show blockers if refused
        if [ "$verdict" = "refused" ]; then
            blockers=$(echo "$output" | grep "^           - " | sed 's/^           - //' || true)
            if [ -n "$blockers" ]; then
                echo "  Blockers:"
                echo "$blockers" | while read -r line; do
                    echo "    - $line"
                done
            fi
        fi
        
        # Show format histogram (quantized formats used)
        echo "  Formats:"
        formats=$(echo "$output" | grep "^  .*[0-9]* (unknown format)$" || true)
        if [ -z "$formats" ]; then
            echo "$output" | grep -E "^[[:space:]]+[a-zA-Z0-9_]+[[:space:]]+[0-9]+ tensors" | while read -r line; do
                echo "    $line"
            done
        else
            echo "$output" | grep -E "^[[:space:]]+[a-zA-Z0-9_#]+[[:space:]]+[0-9]+ tensors" | while read -r line; do
                echo "    $line"
            done
        fi
        
        # Count by verdict
        case "$verdict" in
            "runnable") runnable=$((runnable + 1)) ;;
            "refused") refused=$((refused + 1)) ;;
            *) unreadable=$((unreadable + 1)) ;;
        esac
    done
done

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Total models scanned:  $total"
echo "Runnable:              $runnable"
echo "Refused:               $refused"
echo "Unreadable/Other:      $unreadable"
echo ""
echo "Runnable ratio: $(echo "scale=1; $runnable * 100 / $total" | bc)%"
