import os
import re
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import warnings
import sys

# Clustering Imports
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# --- ROBUST NLTK RESOURCE CHECK ---
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            print(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

download_nltk_resources()
# ---------------------------------------------

# Path setup to import const.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)
from src.utils.const import EXP4_CONFIG, LANGNAME2LANGCODE

warnings.filterwarnings("ignore")

class VocabularyAnalyzer:
    def __init__(self):
        self.vocab_data = {} 
        self.lang_names = []
        
        # 1. Define Strict Regex for Filtering
        self.script_patterns = {
            'Latn': r'[a-zA-Z\u00C0-\u00FF\u0100-\u017F]+', 
            'Cyrl': r'[\u0400-\u04FF]+',
            'Arab': r'[\u0600-\u06FF\u0750-\u077F]+',
            'Deva': r'[\u0900-\u097F]+',
            'Beng': r'[\u0980-\u09FF]+',
            'Taml': r'[\u0B80-\u0BFF]+',
            'Telu': r'[\u0C00-\u0C7F]+',
            'Knda': r'[\u0C80-\u0CFF]+',
            'Mlym': r'[\u0D00-\u0D7F]+',
            'Sinh': r'[\u0D80-\u0DFF]+',
            'Thai': r'[\u0E00-\u0E7F]+',
            'Laoo': r'[\u0E80-\u0EFF]+',
            'Tibt': r'[\u0F00-\u0FFF]+',
            'Mymr': r'[\u1000-\u109F]+',
            'Geor': r'[\u10A0-\u10FF]+',
            'Ethi': r'[\u1200-\u137F]+',
            'Khmr': r'[\u1780-\u17FF]+',
            'Hang': r'[\uAC00-\uD7AF]+',
            'Hebr': r'[\u0590-\u05FF]+',
            'Grek': r'[\u0370-\u03FF]+',
            'Armn': r'[\u0530-\u058F]+',
            'Tfng': r'[\u2D30-\u2D7F]+',
            'Hans': r'[\u4E00-\u9FFF]+',
            'Hant': r'[\u4E00-\u9FFF]+',
            'Jpan': r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+', 
        }

    def get_exp4_targets(self):
        targets = []
        print(f"Parsing configuration for {EXP4_CONFIG['exp_id']}...")
        for family, languages in EXP4_CONFIG['languages'].items():
            for lang_name in languages:
                if lang_name in LANGNAME2LANGCODE:
                    code = LANGNAME2LANGCODE[lang_name]
                    targets.append((lang_name, code))
        return targets

    def preprocess_text(self, text_list, lang_code):
        full_text = " ".join(text_list).lower()
        script_code = lang_code.split('_')[-1]
        
        pattern = self.script_patterns.get(script_code, self.script_patterns['Latn'])
        compact_scripts = ['Hans', 'Hant', 'Jpan', 'Khmr', 'Laoo', 'Thai', 'Hang', 'Mymr', 'Tibt']

        valid_tokens = set()

        if script_code in compact_scripts:
            # --- STRATEGY A: Compact Scripts (Regex Chunks -> Split Chars) ---
            chunks = re.findall(pattern, full_text)
            for chunk in chunks:
                for char in chunk:
                    if not char.isdigit(): # Basic numeric filter
                         valid_tokens.add(char)
        else:
            # --- STRATEGY B: Alphabetic Scripts (NLTK Words -> Filter) ---
            try:
                raw_tokens = nltk.word_tokenize(full_text)
            except LookupError:
                raw_tokens = full_text.split()

            for token in raw_tokens:
                if re.fullmatch(pattern, token) and token.isalpha():
                    if len(token) > 1:
                        valid_tokens.add(token)

        return valid_tokens, script_code

    def load_and_process_data(self):
        targets = self.get_exp4_targets()
        print(f"\nLoading FLORES+ (dev) for {len(targets)} languages...")
        
        for name, code in tqdm(targets):
            try:
                dataset = load_dataset("openlanguagedata/flores_plus", code, split="dev")
                vocab, script = self.preprocess_text(dataset['text'], code)
                
                if len(vocab) > 0:
                    self.vocab_data[name] = {'tokens': vocab, 'script': script, 'code': code}
                    self.lang_names.append(name)
                else:
                    print(f"Skipping {name}: Empty vocabulary generated.")
            except Exception as e:
                print(f"Error {name}: {e}")

    def calculate_jaccard_similarity(self):
        n = len(self.lang_names)
        matrix = pd.DataFrame(index=self.lang_names, columns=self.lang_names, dtype=float)

        print("\nCalculating Similarity Matrix (Hybrid NLTK + Regex)...")
        for i in range(n):
            for j in range(n):
                name_a = self.lang_names[i]
                name_b = self.lang_names[j]
                
                if i == j:
                    matrix.iloc[i, j] = 1.0
                    continue
                if pd.notna(matrix.iloc[j, i]):
                    matrix.iloc[i, j] = matrix.iloc[j, i]
                    continue

                # --- STRICT SCRIPT ISOLATION ---
                script_a = self.vocab_data[name_a]['script']
                script_b = self.vocab_data[name_b]['script']
                
                cjk_group = {'Hans', 'Hant', 'Jpan'}
                is_cjk = (script_a in cjk_group and script_b in cjk_group)

                if script_a != script_b and not is_cjk:
                    matrix.iloc[i, j] = 0.0
                    continue
                # -------------------------------

                set_a = self.vocab_data[name_a]['tokens']
                set_b = self.vocab_data[name_b]['tokens']
                
                if not set_a or not set_b:
                    matrix.iloc[i, j] = 0.0
                    continue

                intersection = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                
                matrix.iloc[i, j] = intersection / union if union > 0 else 0.0
        
        return matrix

    def perform_clustering_analysis(self, sim_matrix_df):
        print("\n" + "="*40)
        print("PERFORMING HIERARCHICAL CLUSTERING PER SCRIPT")
        print("="*40)

        similarity_matrix = sim_matrix_df.values
        lang2id_lang = {name: i for i, name in enumerate(self.lang_names)}
        id_lang2lang = {i: name for i, name in enumerate(self.lang_names)}
        
        script2lang = {}
        for name in self.lang_names:
            script = self.vocab_data[name]['script']
            if script not in script2lang:
                script2lang[script] = []
            script2lang[script].append(name)

        lang2label = {}

        for script, langs in script2lang.items():
            lang_indexes = [lang2id_lang[lang] for lang in langs if lang in lang2id_lang]
            selected_langs = [id_lang2lang[idx] for idx in lang_indexes]

            if len(lang_indexes) <= 2:
                for lang in selected_langs:
                    lang2label[lang] = f'{script}_0'
                print(f"Skipping script {script} with insufficient languages.")
                continue

            print(f"\n--- Script: {script} ---")
            
            subset_sim_matrix = similarity_matrix[np.ix_(lang_indexes, lang_indexes)]
            distance_matrix = 1 - subset_sim_matrix
            np.fill_diagonal(distance_matrix, 0.0)

            if not np.allclose(distance_matrix, distance_matrix.T):
                raise ValueError(f"Distance matrix for {script} is not symmetric!")

            condensed_dist = squareform(distance_matrix)
            linked = linkage(condensed_dist, method='average', metric='precomputed')

            # --- PLOT DENDROGRAM ---
            plt.figure(figsize=(12, 6))
            dendrogram(
                linked,
                orientation='top',
                labels=selected_langs,
                distance_sort='descending',
                show_leaf_counts=True,
                leaf_rotation=90,
                leaf_font_size=10
            )
            plt.title(f"Hierarchical Clustering Dendrogram - {script} Script")
            plt.xlabel("Language")
            plt.ylabel("Distance (1 - Jaccard Similarity)")
            plt.tight_layout()
            
            dendrogram_file = f"dendrogram_{script}.png"
            plt.savefig(dendrogram_file, dpi=300)
            print(f"  Saved dendrogram to {dendrogram_file}")
            plt.close() # Close figure to free memory
            # -----------------------

            # Silhouette Analysis
            silhouette_scores = []
            max_k = len(lang_indexes)
            if max_k < 2: continue
                
            K_range = range(2, max_k)
            
            for k in K_range:
                clusters_temp = fcluster(linked, k, criterion='maxclust')
                if len(set(clusters_temp)) < 2 or len(set(clusters_temp)) >= len(lang_indexes):
                    silhouette_scores.append(-1)
                    continue
                score = silhouette_score(distance_matrix, clusters_temp, metric='precomputed')
                silhouette_scores.append(score)

            if not silhouette_scores or max(silhouette_scores) == -1:
                for lang in selected_langs:
                    lang2label[lang] = f'{script}_1'
                print(f"  Could not determine optimal clusters via Silhouette.")
                continue

            optimal_k_index = np.argmax(silhouette_scores)
            optimal_k = K_range[optimal_k_index]
            best_score = max(silhouette_scores)
            
            print(f"  Optimal K: {optimal_k} (Score: {best_score:.3f})")

            clusters = fcluster(linked, optimal_k, criterion='maxclust')

            for lang_name, cluster_id in zip(selected_langs, clusters):
                lang2label[lang_name] = f'{script}_{cluster_id}'

        return lang2label

    def plot_heatmap(self, matrix):
        plt.figure(figsize=(24, 24))
        sns.heatmap(
            matrix, annot=False, cmap="viridis", square=True, 
            xticklabels=True, yticklabels=True
        )
        plt.title("Vocabulary Similarity (Original Order)", fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig("vocab_similarity_simple_exp4.png", dpi=300, bbox_inches='tight')
        plt.close()
if __name__ == "__main__":
    analyzer = VocabularyAnalyzer()
    analyzer.load_and_process_data()
    
    if len(analyzer.vocab_data) > 1:
        # 1. Calculate Similarity
        sim_matrix = analyzer.calculate_jaccard_similarity()
        sim_matrix.to_csv("vocab_similarity.csv")
        
        # 2. Perform Clustering (Saves Dendrograms inside here)
        cluster_mapping = analyzer.perform_clustering_analysis(sim_matrix)
        
        # 3. Save Clustering CSV
        print("\nSaving Clustering Results...")
        results_data = []
        for lang_name, label in cluster_mapping.items():
            parts = label.split('_')
            script_name = parts[0]
            cluster_id = parts[1]
            flores_code = analyzer.vocab_data[lang_name]['code']
            
            results_data.append({
                "Language": lang_name,
                "FLORES_Code": flores_code,
                "Script": script_name,
                "Cluster_ID": cluster_id,
                "Full_Label": label
            })
            
        if results_data:
            df_results = pd.DataFrame(results_data)
            df_results = df_results.sort_values(by=['Script', 'Cluster_ID'])
            output_csv = "vocab_clustering_results.csv"
            df_results.to_csv(output_csv, index=False)
            print(f"Saved clustering results to: {output_csv}")
        
        # 4. Plot Heatmap
        analyzer.plot_heatmap(sim_matrix)
    else:
        print("Not enough data loaded.")