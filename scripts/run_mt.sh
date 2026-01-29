#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

echo "Running inference at $(date)"
python run_activation_mt.py \
    --model_name "Qwen/Qwen3-8B" \
    --prompt_path "./prompts/machine_translation/prompt_en.txt" \
    --output_dir "./outputs" \
    --target_langs fra_Latn jav_Latn sun_Latn tur_Latn cym_Latn \
    --source_langs ind_Latn eng_Latn \
    # --sample_size 100 \
    # --source_langs fra_Latn jav_Latn sun_Latn tur_Latn cym_Latn \
    # --target_langs ind_Latn eng_Latn \
    # --is_base_model
echo "--------------------------------------------------------"
echo "========================================================"
echo "Completed at: $(date)"
echo "========================================================"
