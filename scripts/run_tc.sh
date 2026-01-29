#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

echo "Running inference at $(date)"
python3 -m src.main.run_activation_tc \
    --model_name "google/gemma-3-1b-it" \
    --prompt_lang "all" \
    --output_dir "./outputs" \
    --languages arb_Arab cym_Latn eng_Latn fra_Latn ind_Latn jav_Latn jpn_Jpan kor_Hang sun_Latn swh_Latn tgl_Latn tur_Latn urd_Arab \
    # --sample_size 100 \
    # --is_base_model
echo "--------------------------------------------------------"
echo "========================================================"
echo "Completed at: $(date)"
echo "========================================================"
