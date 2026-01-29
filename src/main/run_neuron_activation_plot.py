import hydra
import torch
import os
import yaml
from omegaconf import DictConfig
from src.utils.neuron_visualizer import NeuronVisualizer

def load_neuron_maps(cfg: DictConfig, languages: list):
    """
    Loads the .pt files generated during the extraction phase 
    and converts them to sets for the visualizer.
    """
    raw_map = {}
    prompted_map = {}
    
    # Logic to load the 'over_zero' tensors saved by your extractor
    for lang in languages:
        # Example path: output_lape/google/gemma-3-4b-it/raw/eng_Latn.pt
        raw_path = os.path.join(cfg.paths.output_dir, cfg.model.name, "raw", f"{lang}.pt")
        prompt_path = os.path.join(cfg.paths.output_dir, cfg.model.name, "prompted", f"{lang}.pt")
        
        if os.path.exists(raw_path):
            counts = torch.load(raw_path)["over_zero"]
            # You would use your 'convert_counts_to_flat_set' logic here
            raw_map[lang] = convert_to_set(counts, cfg.processing.percentile_threshold)
            
        if os.path.exists(prompt_path):
            counts = torch.load(prompt_path)["over_zero"]
            prompted_map[lang] = convert_to_set(counts, cfg.processing.percentile_threshold)
            
    return raw_map, prompted_map

@hydra.main(config_path="../../config", config_name="plotting", version_base=None)
def main(cfg: DictConfig):
    # 1. Get languages from your config
    with open(cfg.processing.languages_file, 'r') as f:
        languages = yaml.safe_load(f)

    # 2. Load the data extracted in the previous step
    raw_map, prompted_map = load_neuron_maps(cfg, languages)

    # 3. Initialize YOUR visualizer
    visualizer = NeuronVisualizer(num_layers=cfg.model.num_layers)

    # 4. Generate the LAPE-specific plots
    os.makedirs(cfg.paths.plot_dir, exist_ok=True)
    
    visualizer.plot_layer_distribution(
        spec_map=raw_map, 
        title="Language Neuron Distribution", 
        save_path=f"{cfg.paths.plot_dir}/dist_layer.png"
    )
    
    visualizer.plot_overlap_heatmap(
        raw_map=raw_map, 
        prompted_map=prompted_map, 
        save_path=f"{cfg.paths.plot_dir}/overlap_heatmap.png"
    )

if __name__ == "__main__":
    main()