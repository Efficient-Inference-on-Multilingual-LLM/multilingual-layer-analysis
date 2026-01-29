import hydra
import torch
import os
from omegaconf import DictConfig
from src.utils.neuron_extractor import OfflineNeuronExtractor
from src.utils.neuron_visualizer import NeuronVisualizer
from src.utils.data_loader import ensure_data_exists
from src.utils.const import EXP4_CONFIG, LANGNAME2LANGCODE
from src.utils.neuron_selection import select_by_entropy, select_by_threshold

@hydra.main(config_path="../../config", config_name="extraction", version_base=None)
def main(cfg: DictConfig):
    # 1. Setup Offline Extractor
    extractor = OfflineNeuronExtractor(cfg)
    
    model_tag = cfg.model.name.split('/')[-1]
    raw_tensor_cache = {}      
    prompted_tensor_cache = {} 
    languages_processed = []
    raw_n_list = []
    prompted_n_list = []

    # 2. Iteration (Modified to load from disk)
    print(f"🚀 Starting Offline Extraction for {model_tag}")
    for family, lang_names in EXP4_CONFIG['languages'].items():
        for lang_name in lang_names:
            lang_code = LANGNAME2LANGCODE.get(lang_name)
            if not lang_code: continue
            
            # --- RAW MODE ---
            # Note: Ensure the folder name on disk matches 'lang_code' (e.g., ind_Latn) 
            # or 'lang_name' depending on how you saved it. Assuming 'lang_code'.
            counts_raw, n_raw = extractor.aggregate_language(lang_code, mode="raw")
            
            if counts_raw is None: continue # Skip if data missing

            # --- PROMPTED MODE ---
            # If you don't have prompted data yet, you might want to skip this or handle logic
            # Assuming prompted data structure is same but inside /prompted/ folder
            # counts_pmt, n_pmt = extractor.aggregate_language(lang_code, mode="prompted")

            # if counts_pmt is None: 
            #      # If you only have raw data, you might want to handle this gracefully
            #      print(f"⚠️ Prompted data missing for {lang_name}, skipping.")
            #      continue

            # Store results
            # prompted_tensor_cache[lang_name] = counts_pmt
            # prompted_n_list.append(n_pmt)
            
            raw_tensor_cache[lang_name] = counts_raw
            raw_n_list.append(n_raw)
            languages_processed.append(lang_name)
            print(f"✅ Loaded {lang_name} (n={n_raw})")
    if not languages_processed:
        print("❌ No data found for any language. Check paths in neuron_extractor.py")
        return

    # 3. --- SELECTION ALGORITHM (Unchanged) ---
    # The rest of the logic remains exactly the same as LAPE needs the aggregated counts
    
    # Ensure extractor dimensions are set (in case they weren't inferred early enough)
    # We use the extractor.num_layers/inter_size for saving masks later
    
    if cfg.processing.selection_mode == "entropy":
        print(f" 🧠 Selecting specialist neurons using Entropy...")
        stacked_raw = torch.stack([raw_tensor_cache[ln] for ln in languages_processed], dim=-1)        
        raw_map = select_by_entropy(
            stacked_raw, raw_n_list, languages_processed, 
            top_rate=cfg.processing.top_rate, filter_rate=cfg.processing.filter_rate
        )

        # stacked_prompted = torch.stack([prompted_tensor_cache[ln] for ln in languages_processed], dim=-1)
        # prompted_map = select_by_entropy(
        #     stacked_prompted, prompted_n_list, languages_processed, 
        #     top_rate=cfg.processing.top_rate, filter_rate=cfg.processing.filter_rate
        # )
    else:
        print(f" 📊 Selecting active neurons using Threshold...")
        raw_map = {
            ln: select_by_threshold(raw_tensor_cache[ln], cfg.processing.percentile_threshold)
            for ln in languages_processed
        }
        # prompted_map = {
        #     ln: select_by_threshold(prompted_tensor_cache[ln], cfg.processing.percentile_threshold)
        #     for ln in languages_processed
        # }

    # 4. Save Outputs
    model_output_dir = os.path.join(cfg.paths.output_dir, model_tag, cfg.processing.selection_mode)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # masks_base_dir = os.path.join(model_output_dir, "masks")
    # raw_mask_dir = os.path.join(masks_base_dir, "raw")
    # # prompted_mask_dir = os.path.join(masks_base_dir, "prompted")
    # os.makedirs(raw_mask_dir, exist_ok=True)
    # # os.makedirs(prompted_mask_dir, exist_ok=True)

    # def save_neuron_masks(neuron_map, target_dir, mode_label):
    #     print(f"💾 Saving {mode_label} masks to {target_dir}...")
    #     for lang_name, neuron_indices in neuron_map.items():
    #         clean_name = lang_name.replace(" ", "_").replace("(", "").replace(")", "")
    #         layer_masks = []
    #         for i in range(extractor.num_layers):
    #             start_idx = i * extractor.inter_size
    #             end_idx = (i + 1) * extractor.inter_size
    #             layer_neuron_ids = [
    #                 idx - start_idx for idx in neuron_indices 
    #                 if start_idx <= idx < end_idx
    #             ]
    #             layer_masks.append(torch.tensor(layer_neuron_ids, dtype=torch.long))
            
    #         save_path = os.path.join(target_dir, f"mask_{clean_name}.pt")
    #         torch.save(layer_masks, save_path)

    # save_neuron_masks(raw_map, raw_mask_dir, "RAW")
    # save_neuron_masks(prompted_map, prompted_mask_dir, "PROMPTED")

    # 5. Visualization
    viz = NeuronVisualizer(num_layers=extractor.num_layers, inter_size=extractor.inter_size)
    viz.plot_layer_distribution(raw_map, f"Raw ({model_tag})", f"{model_output_dir}/dist_raw.png")
    # viz.plot_overlap_heatmap(raw_map, prompted_map, f"{model_output_dir}/overlap.png")
    viz.plot_similarity_heatmap(raw_map, f"Jaccard Raw ({model_tag})", f"{model_output_dir}/similarity_raw.png")
    # viz.plot_similarity_heatmap(prompted_map, f"Jaccard Prompted ({model_tag})", f"{model_output_dir}/similarity_prompted.png")
    viz.plot_per_layer_overlap(
        raw_map,
        f"{model_output_dir}/raw_per_layer_overlap_grid.png"
    )
    # viz.plot_per_layer_overlap(
    #     prompted_map,
    #     f"{model_output_dir}/prompted_per_layer_overlap_grid.png"
    # )

    # 3. Simpan Matriks RAW
    raw_matrix_path = os.path.join(model_output_dir, "overlap_matrix_3d_raw.npy")
    viz.save_per_layer_matrix(raw_map, languages_processed, raw_matrix_path)

    # 4. Simpan Matriks PROMPTED
    # prompted_matrix_path = os.path.join(model_output_dir, "overlap_matrix_3d_prompted.npy")
    # viz.save_per_layer_matrix(prompted_map, languages_processed, prompted_matrix_path)

    print(f"✅ Selesai! Semua hasil untuk {model_tag} tersedia di: {model_output_dir}")
if __name__ == "__main__":
    main()