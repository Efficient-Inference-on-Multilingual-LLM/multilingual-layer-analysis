import hydra
import torch
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from vllm import SamplingParams
from src.utils.neuron_extractor import NeuronExtractor
from src.utils.neuron_visualizer import InterventionVisualizer
from src.utils.data_loader import ensure_data_exists
from src.utils.const import EXP4_CONFIG, LANGNAME2LANGCODE

def sanitize_filename(name):
    """Konsisten dengan sanitasi saat penyimpanan mask di run_extraction."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")

@hydra.main(config_path="../../config", config_name="extraction", version_base=None)
def main(cfg: DictConfig):
    # 1. Inisialisasi
    extractor = NeuronExtractor(cfg)
    tokenizer = extractor.llm.get_tokenizer()
    model_tag = cfg.model.name.split('/')[-1]
    
    # Ambil nama algoritme dari config (e.g., 'entropy' atau 'threshold')
    # Pastikan di config/extra_args nilainya benar
    sel_mode = cfg.processing.selection_mode 
    
    # Ambil daftar semua bahasa
    all_languages = []
    for family, langs in EXP4_CONFIG['languages'].items():
        all_languages.extend(langs)

    # 2. Loop Utama untuk Kedua Mode: RAW dan PROMPTED
    mask_modes = ["raw", "prompted"]
    
    for current_mode in mask_modes:
        print(f"\n" + "="*50)
        print(f"🚀 STARTING INTERVENTION MODE: {current_mode.upper()}")
        print(f"🧠 Algorithm: {sel_mode}")
        print("="*50)
        
        # --- PERBAIKAN JALUR DI SINI ---
        # Menambahkan sel_mode ke dalam path agar sesuai dengan hasil ekstraksi
        intervention_dir = os.path.join(cfg.paths.output_dir, model_tag, sel_mode, "intervention", current_mode)
        mask_dir = os.path.join(cfg.paths.output_dir, model_tag, sel_mode, "masks", current_mode)
        # ------------------------------

        os.makedirs(intervention_dir, exist_ok=True)

        if not os.path.exists(mask_dir):
            print(f"⚠️ Warning: Mask directory not found: {mask_dir}. Skipping mode.")
            continue

        all_ppl_results = {}

        # 3. Iterasi Ablasi
        for target_lang in all_languages:
            clean_name = sanitize_filename(target_lang)
            mask_path = os.path.join(mask_dir, f"mask_{clean_name}.pt")
            
            if not os.path.exists(mask_path):
                continue
                
            print(f"\n🧪 Ablating {target_lang} neurons...")
            layer_masks = torch.load(mask_path)
            
            # Terapkan patch ablasi (Gemma 3 compatible)
            extractor.apply_ablation_patch(layer_masks)
            
            current_ablation_impact = {}

            # 4. Evaluasi Lintas Bahasa
            for test_lang in all_languages:
                lang_code = LANGNAME2LANGCODE.get(test_lang)
                data_path = ensure_data_exists(lang_code, "raw", cfg.paths.data_dir, tokenizer, cfg)
                
                if not data_path: continue

                ids = torch.load(data_path)
                max_len = cfg.processing.data_block_size
                
                # 1. Siapkan input_ids (misal mengambil 5 blok untuk efisiensi)
                input_ids = ids[:max_len * 5].reshape(-1, max_len) 

                # 2. FORMAT YANG BENAR: Masukkan ke dalam list prompts sebagai kamus
                # Ini sesuai dengan pola di run_inference yang Anda miliki
                prompts = [{"prompt_token_ids": seq.tolist()} for seq in input_ids]

                # 3. Panggil generate dengan argumen 'prompts'
                outputs = extractor.llm.generate(
                    prompts=prompts, 
                    sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=0)
                )
                
                # Hitung rata-rata negatif log-probs
                batch_log_probs = []
                for output in outputs:
                    # Perbaikan: Tambahkan '.logprob' untuk mengambil nilai numerik float-nya
                    # r adalah dict: {token_id: Logprob(logprob=..., rank=..., decoded_token=...)}
                    token_logprobs = [
                        val.logprob 
                        for r in output.prompt_logprobs if r 
                        for val in r.values()
                    ]
                    
                    if token_logprobs:
                        batch_log_probs.append(np.mean(token_logprobs))
                
                avg_neg_logprob = -np.mean(batch_log_probs) if batch_log_probs else 0
                current_ablation_impact[test_lang] = avg_neg_logprob
                print(f"    - Tested on {test_lang}: {avg_neg_logprob:.4f}")

            all_ppl_results[target_lang] = current_ablation_impact

        # 5. Simpan Data dan Plot
        if all_ppl_results:
            df_results = pd.DataFrame(all_ppl_results).T
            df_results.to_csv(os.path.join(intervention_dir, f"ppl_matrix_{current_mode}.csv"))
            
            viz = InterventionVisualizer()
            viz.plot_ppl_heatmap(
                all_ppl_results, 
                os.path.join(intervention_dir, f"impact_heatmap_{current_mode}.png")
            )
            print(f"✅ Results saved to: {intervention_dir}")
if __name__ == "__main__":
    main()