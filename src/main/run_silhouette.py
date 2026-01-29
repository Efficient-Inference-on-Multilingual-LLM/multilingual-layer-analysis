import os
import sys
import pandas as pd
import numpy as np
from src.utils.const import LANGCODE2LANGNAME, LANGNAME2LANGCODE, MODEL2HIDDEN_SIZE, MODEL2NUM_LAYERS, EXP2_CONFIG, EXP3_CONFIG, EXP4_CONFIG
import glob
import torch
from tqdm import tqdm
import cudf
import cupy as cp
from cuml.cluster import KMeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from sklearn.datasets import make_blobs
from typing import List, Literal

def calculate_silhouette_score(
        output_dir: str, 
        activation_dir: str,
        task: str,
        data_split: str,
        model_name: str,
        extraction_mode: str,
        token_position: Literal['last_token', 'average'],
        activation_locations: List[Literal["residual-postattn", "residual-postmlp"]],
        languages: List[str]):
    
    num_layers = MODEL2NUM_LAYERS[model_name]
    text_ids = glob.glob(f'{activation_dir}/{task}/{data_split}/{model_name}/{extraction_mode}/eng_Latn/*')
    text_ids = [text_id.split('/')[-1].split('.')[0] for text_id in text_ids]

    print(f"Text IDs: {len(text_ids)}, Num layers: {num_layers}, Number of languages: {len(languages)}, Hidden size: {MODEL2HIDDEN_SIZE[model_name]}")
    print(f"Calculating silhouette scores for model: {model_name}")

    # Load activations per residual location
    for activation_location in activation_locations:
        print(f'Calculating silhouette scores for residual location: {activation_location}')
        
        # Initialize empty torch tensor to hold all labels [text_id, language]
        labels = torch.zeros((len(text_ids) * len(languages),), dtype=torch.long).to('cuda')
        print(f'Labels tensor shape: {labels.shape}')

        # Initialize silhouette score matrix
        silhouette_score_matrix = torch.zeros((num_layers + 1, len(languages), len(languages))).to('cuda')

        for layer_id in range(-1, num_layers):
            print(f"Layer ID: {layer_id}")

            # Initialize empty torch tensor to hold all activations [text_id, language, hidden_size]
            activation_per_lang = torch.zeros((len(text_ids), len(languages), MODEL2HIDDEN_SIZE[model_name])).to('cuda')
            print(f'Activation tensor shape: {activation_per_lang.shape}')

            # Reshape to [text_id * language, hidden_size]
            activation_per_lang = activation_per_lang.view(-1, MODEL2HIDDEN_SIZE[model_name])
            print(f'Reshaped activation tensor shape: {activation_per_lang.shape}')
        
            # Load activations for all languages
            for lang_idx, lang in tqdm(enumerate(languages), total=len(languages), desc='Loading activations for all languages'):
                
                if layer_id == -1:
                    paths = sorted(glob.glob(f'{activation_dir}/{task}/{data_split}/{model_name}/{extraction_mode}/{lang}/*/{token_position}/layer_embed_tokens.pt'))
                else:
                    paths = sorted(glob.glob(f'{activation_dir}/{task}/{data_split}/{model_name}/{extraction_mode}/{lang}/*/{token_position}/layer_{activation_location}_{layer_id}.pt'))
                
                if len(paths) != len(text_ids):
                    print(f"Warning: Expected {len(text_ids)} files for language '{lang}' at layer {layer_id}, but found {len(paths)} files.")
                    raise ValueError(f"Missing activations at '{activation_dir}/{task}/{data_split}/{model_name}/{extraction_mode}/{lang}/*/{token_position}/layer_{activation_location}_{layer_id}.pt'")
                
                for text_idx_inner, path in enumerate(paths):
                    activation = torch.load(path)
                    # Calculate the flat index for the reshaped tensor
                    flat_idx = text_idx_inner * len(languages) + lang_idx
                    activation_per_lang[flat_idx, :] = activation.to('cuda')
                    # Populate labels with language_id (only need to do this once per text-language pair)
                    labels[flat_idx] = lang_idx
                    # if layer_id == -1:
                
            # 4. Calculate the silhouette score on the GPU
            # The function takes the data and the predicted labels as input.
            # Calculate pairwise silhouette scores per language
            for lang_idx1 in tqdm(range(len(languages)), desc='Calculating silhouette scores for language pairs'):
                for lang_idx2 in range(len(languages)):
                    # Only compute for different languages and upper triangle
                    if lang_idx1 < lang_idx2:
                        # Take all index of labels that have value 1
                        indexes_lang1 = ((labels == lang_idx1) | (labels == lang_idx2)).nonzero(as_tuple=True)[0]
                        # Take activations for those indexes
                        activations_lang1 = activation_per_lang[indexes_lang1, :]
                        labels_lang1 = labels[indexes_lang1]
                        # Combine activations and labels to both one tensor
                        score = cython_silhouette_score(activations_lang1, labels_lang1)
                        # Store score in matrix
                        silhouette_score_matrix[layer_id + 1, lang_idx1, lang_idx2] = score

                        # print(f"Silhouette Score between {id2langcode[lang_idx1]} and {id2langcode[lang_idx2]}: {score:.4f}")

        # Store silhouette score matrix
        output_path = os.path.join(output_dir, task, data_split, model_name, extraction_mode, token_position, activation_location)
        os.makedirs(output_path, exist_ok=True)
        torch.save(silhouette_score_matrix, os.path.join(output_path, 'silhouette_score_matrix.pt'))


if __name__ == "__main__":
    languages = []
    for family, langs in EXP4_CONFIG['languages'].items():
        languages.extend(langs)

    languages = [LANGNAME2LANGCODE[lang] for lang in languages]
    output_dir = f'outputs_silhouette/{EXP4_CONFIG["exp_id"]}'
    model_names = [
        # 'gemma-3-4b-it',
        # 'Llama-3.1-8B-Instruct',
        # 'aya-expanse-8b',
        # 'Qwen3-8B',
        'Qwen3-14B',
        # 'gemma-3-12b-it',
        # 'pythia-6.9b-deduped',
        # 'aya-101',
        # 'SmolLM3-3B'
    ]
    for model_name in tqdm(model_names, desc="Calculating silhouette scores for models"):
        calculate_silhouette_score(
            output_dir=output_dir,
            activation_dir=f'outputs_flores_plus',
            task='next_token',
            data_split='dev',
            model_name=model_name,
            extraction_mode='raw',
            token_position='last_token',
            # activation_locations=['attn-output', 'mlp-output', 'residual-postmlp', 'residual-postattn', 'residual-preattn'],
            # activation_locations=['attn-output', 'mlp-output'],
            # activation_locations=['residual-postmlp', 'residual-postattn'],
            activation_locations=['residual-preattn'],
            languages=languages
            # languages=['eng_Latn', 'spa_Latn', 'fra_Latn']
        )
    