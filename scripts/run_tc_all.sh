#!/bin/bash

# Array of model names
model_names=(
    "google/gemma-3-1b-it"
    "meta-llama/Llama-3.1-8B-Instruct"
    "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
    "Sahabat-AI/llama3-8b-cpt-sahabatai-v1-instruct"
    "CohereLabs/aya-expanse-8b"
    "sail/Sailor2-8B-Chat"
    "Qwen/Qwen3-8B"
)

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

# Loop through each model
for model_name in "${model_names[@]}"; do
    echo "Running inference for model: $model_name at $(date)"
    python3 -m src.main.run_activation_tc \
        --model_name "$model_name" \
        --prompt_lang "all" \
        --output_dir "./outputs" \
        --languages arb_Arab cym_Latn eng_Latn fra_Latn ind_Latn jav_Latn jpn_Jpan kor_Hang sun_Latn swh_Latn tgl_Latn tur_Latn urd_Arab \
        # --sample_size 100 \
        # --is_base_model
    echo "--------------------------------------------------------"
done

echo "========================================================"
echo "All models completed at: $(date)"
echo "========================================================"