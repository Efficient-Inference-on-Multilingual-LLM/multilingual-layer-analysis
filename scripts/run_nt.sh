#!/bin/bash
source .venv/bin/activate

# Array of model names
model_names=(
    # "meta-llama/Meta-Llama-3-8B"
    # "google/gemma-2-9b"
    # "aisingapore/Gemma-SEA-LION-v3-9B"
    # "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-base"
    # "aisingapore/Llama-SEA-LION-v3-8B"
    # "GoToCompany/llama3-8b-cpt-sahabatai-v1-base"
    # "Qwen/Qwen2.5-7B"
    # "sail/Sailor2-8B"
    # "sail/Sailor2-8B-Chat"
    # "google/gemma-2-9b-it"
    # "google/gemma-3-1b-it"
    # "google/gemma-3-270m-it"
    # "google/gemma-3-12b-it"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "HuggingFaceTB/SmolLM3-3B"
    "Qwen/Qwen3-14B"
    # "Qwen/Qwen3-8B"
    # "CohereLabs/aya-101"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "CohereLabs/aya-expanse-8b"
    # "google/gemma-3-4b-it"
    # "EleutherAI/pythia-6.9b-deduped"
)

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

# Loop through each model
for model_name in "${model_names[@]}"; do
    echo "Running inference for model: $model_name at $(date)"
    python3 -m src.main.run_activation_nexttoken \
        --model_name "$model_name" \
        --prompt_lang "no_prompt" \
        --output_dir "./outputs_flores_plus" \
        --data_split "dev" \
        --use_predefined_languages
        # --languages "eng_Latn" \
	    # --sample_size 200 \
        # --is_base_model
    echo "--------------------------------------------------------"
#    rm -rf ~/.cache/huggingface/hub/*
done

echo "========================================================"
echo "All models completed at: $(date)"
echo "========================================================"
