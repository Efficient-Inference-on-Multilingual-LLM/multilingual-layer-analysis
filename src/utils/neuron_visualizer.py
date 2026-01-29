import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Set
import pandas as pd


class NeuronVisualizer:
    def __init__(self, num_layers: int = 34, inter_size: int = 4096):
        self.num_layers = num_layers
        self.inter_size = inter_size

    def plot_layer_distribution(self, spec_map: Dict[str, Set[int]], title: str, save_path: str):
        """Plots which layers contain the most language-specific neurons."""
        plt.figure(figsize=(12, 6))
        
        # Dictionary to store counts for saving later
        distribution_data = {}

        for lang, neurons in spec_map.items():
            layers = [idx // self.inter_size for idx in neurons] 
            counts = np.bincount(layers, minlength=self.num_layers)
            distribution_data[lang] = counts
            
            plt.plot(range(self.num_layers), counts, label=lang, alpha=0.7)

        plt.title(f"Neuron Distribution per Layer: {title}")
        plt.xlabel("Layer Index")
        plt.ylabel("Number of Specific Neurons")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize='xx-small')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save Plot
        plt.savefig(save_path)
        plt.close()

        # Save Data - Fixed npz logic
        npz_save_path = save_path.rsplit('.', 1)[0] + ".npz"
        np.savez(
            npz_save_path,
            **distribution_data,
            languages=np.array(list(spec_map.keys()))
        )

    def plot_overlap_heatmap(self, raw_map: Dict[str, Set[int]], prompted_map: Dict[str, Set[int]], save_path: str):
        """Visualizes the percentage of language neurons that stay active during the task."""
        langs = list(raw_map.keys())
        overlap_percentages = []

        for lang in langs:
            if lang in raw_map and len(raw_map[lang]) > 0:
                intersection = raw_map[lang].intersection(prompted_map.get(lang, set()))
                percentage = (len(intersection) / len(raw_map[lang])) * 100
            else:
                percentage = 0.0

            overlap_percentages.append(percentage)

        plt.figure(figsize=(10, 18))
        sns.barplot(x=overlap_percentages, y=langs, palette="viridis")
        plt.title("Functional Overlap: % of Language Neurons active during Task")
        plt.xlabel("Overlap Percentage (%)")
        plt.ylabel("Language")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_similarity_heatmap(self, spec_map: Dict[str, Set[int]], title: str, save_path: str):
        langs = sorted(list(spec_map.keys()))
        n = len(langs)
        matrix = np.zeros((n, n))
        
        # Calculate total neurons in the entire model for global normalization
        total_model_neurons = self.num_layers * self.inter_size

        for i in range(n):
            for j in range(n):
                set_a = spec_map[langs[i]]
                set_b = spec_map[langs[j]]
                if not set_a or not set_b:
                    matrix[i, j] = 0.0
                    continue
                
                intersection = len(set_a.intersection(set_b))
                
                # MODIFICATION: Changed denominator from Union to Total Model Size
                # matrix[i, j] = intersection / len(set_a.union(set_b)) 
                matrix[i, j] = intersection / total_model_neurons

        df_sim = pd.DataFrame(matrix, index=langs, columns=langs)

        # 1. Increase figure size significantly for 90+ languages
        plt.figure(figsize=(20, 18)) 

        # 2. Force xticklabels and yticklabels to True (shows all)
        ax = sns.heatmap(
            df_sim, 
            annot=False, 
            cmap="YlGnBu", 
            square=True, 
            xticklabels=True, 
            yticklabels=True, 
            # MODIFICATION: Updated Label
            cbar_kws={'label': 'Overlap Ratio (Intersection / Total Model Neurons)'}
        )

        # 3. Set a very small font size for the labels
        ax.tick_params(axis='both', which='major', labelsize=7)

        plt.title(f"Neural Similarity Matrix: {title}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def save_per_layer_matrix(self, spec_map: Dict[str, Set[int]], ordered_langs: list, save_path: str):
        """
        Menghasilkan matriks 3D (n_layer, n_lang, n_lang) dan menyimpannya sebagai .npz
        """
        import numpy as np
        
        n_langs = len(ordered_langs)
        matrix_3d = np.zeros((self.num_layers, n_langs, n_langs), dtype=np.float32)

        print(f"📊 Generating 3D Matrix: {self.num_layers} layers x {n_langs} languages...")

        for layer_idx in range(self.num_layers):
            start_idx = layer_idx * self.inter_size
            end_idx = (layer_idx + 1) * self.inter_size

            # Pre-filter neuron untuk layer ini
            layer_sets = []
            for lang in ordered_langs:
                s = {idx for idx in spec_map.get(lang, set()) if start_idx <= idx < end_idx}
                layer_sets.append(s)

            # Hitung Similarity
            for i in range(n_langs):
                for j in range(i, n_langs): 
                    set_a = layer_sets[i]
                    set_b = layer_sets[j]
                    
                    if not set_a and not set_b:
                        val = 0.0
                    else:
                        intersection = len(set_a.intersection(set_b))
                        
                        # MODIFICATION: Changed denominator from Union to Layer Size
                        # union = len(set_a.union(set_b))
                        # val = intersection / (union + 1e-9)
                        val = intersection / self.inter_size
                    
                    matrix_3d[layer_idx, i, j] = val
                    matrix_3d[layer_idx, j, i] = val # Mirroring

        np.savez(
            save_path,
            lape_matrices=matrix_3d,
            languages=np.array(ordered_langs)
        )
        print(f"✅ 3D Overlap Matrix saved to: {save_path}")

    def plot_per_layer_overlap(self, map: Dict[str, Set[int]], save_path: str):
        import math
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        overlaps = []
        langs = list(map.keys())
        n_langs = len(langs)
        n_cols = 6
        n_rows = math.ceil(self.num_layers / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 10))
        axes = axes.flatten()

        # MODIFICATION: Since we divide by fixed size (4096+), values will be small.
        # Adjusted vmax to 0.1 (10% of layer) so the heatmap isn't completely white/blank.
        # You may need to tune this based on your sparsity.
        vmin, vmax = 0.0, 0.1 
        
        for layer_idx in range(self.num_layers):
            ax = axes[layer_idx]
            overlap_matrix = np.zeros((n_langs, n_langs))
            
            start_idx = layer_idx * self.inter_size
            end_idx = (layer_idx + 1) * self.inter_size

            layer_neurons = {}
            for lang in langs:
                layer_neurons[lang] = {idx for idx in map[lang] if start_idx <= idx < end_idx}

            for i in range(n_langs):
                for j in range(n_langs):
                    set_a = layer_neurons[langs[i]]
                    set_b = layer_neurons[langs[j]]
                    
                    if not set_a or not set_b:
                        overlap_matrix[i, j] = 0
                        continue
                    
                    # MODIFICATION: Intersection / Layer Size
                    overlap_matrix[i, j] = len(set_a.intersection(set_b)) / self.inter_size

            img = sns.heatmap(
                overlap_matrix, 
                ax=ax, 
                cmap="YlGnBu", 
                cbar=False, 
                xticklabels=langs, 
                yticklabels=langs,
                vmin=vmin,
                vmax=vmax,
                square=True
            )
            overlaps.append(overlap_matrix)
            ax.tick_params(axis='both', which='major', labelsize=4)
            plt.setp(ax.get_xticklabels(), rotation=90)
            ax.set_title(f"Layer {layer_idx}", fontsize=16, fontweight='bold')
            

        for i in range(self.num_layers, len(axes)):
            fig.delaxes(axes[i])

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        
        sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, cax=cbar_ax)
        
        # MODIFICATION: Update Label
        cb.set_label('Overlap Ratio (Shared / Layer Size)', fontsize=20, labelpad=20)
        cb.ax.tick_params(labelsize=15)

        plt.suptitle(f"Per-Layer Neuron Overlap: {n_langs} Languages", fontsize=35, y=0.98)
        
        print(f"🖼️ Rendering high-resolution grid with Legend (DPI 300)...")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        np.savez(
            f"{save_path}.npz",
            lape_matrices=np.array(overlaps),
            languages=np.array(langs)
        )

class InterventionVisualizer:
    def plot_ppl_heatmap(self, ppl_results: dict, save_path: str):
        """
        ppl_results: Dict[Ablated_Lang, Dict[Tested_Lang, PPL_Value]]
        Visualisasi dampak ablasi lintas bahasa.
        """
        # Konversi dictionary bersarang ke DataFrame
        df = pd.DataFrame(ppl_results).T # Baris: Bahasa yang dimatikan, Kolom: Bahasa yang diuji
        
        # Normalisasi: Hitung kenaikan persentase jika ada baseline
        # (Opsional, tergantung apakah Anda menyimpan data baseline)
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            df, 
            annot=True, 
            fmt=".2f", 
            cmap="Reds", # Gunakan gradasi merah untuk menunjukkan tingkat kerusakan
            xticklabels=True, 
            yticklabels=True,
            cbar_kws={'label': 'Perplexity (PPL)'}
        )
        
        plt.title("Cross-Lingual Intervention: Impact of Neuron Ablation on PPL", fontsize=16)
        plt.xlabel("Tested Language", fontsize=12)
        plt.ylabel("Ablated Language Neurons", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PPL Heatmap saved to {save_path}")

    