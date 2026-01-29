#!/bin/bash

# 1. Models List (Extracted from your screenshot)
# Note: I've kept the exact naming conventions seen in your file explorer (e.g., Qwen3, Gemma-3)
MODELS="SmolLM3-3B"

# 2. Configuration Loops
COMPONENTS=("proj_up" "mlp_out")
RATES=("0.05" "0.01")
BASE_DIR="./output_lape_non_jaccard"  # Change this to your preferred root output folder

# 3. Execution Loop
for comp in "${COMPONENTS[@]}"; do
    for rate in "${RATES[@]}"; do
        
        # Construct the specific output folder path
        # Splitting by component first, then by k value
        SAVE_PATH="${BASE_DIR}/${comp}/k_${rate}"
        
        echo "========================================================"
        echo "🚀 Starting Batch:"
        echo "   MODELS:    $MODELS"
        echo "   COMPONENT: $comp"
        echo "   TOP_K:     $rate"
        echo "   OUTPUT:    $SAVE_PATH"
        echo "========================================================"

        # Run the script
        # I have assumed '-e' (extract) mode. Change to -i or -p if needed.
        ./scripts/run_lape.sh -e \
            --models "$MODELS" \
            --component "$comp" \
            --top_rate "$rate" \
            --save_path "$SAVE_PATH"
            
    done
done