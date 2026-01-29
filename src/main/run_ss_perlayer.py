import os
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from src.utils.const import LANGCODE2LANGNAME, LANGNAME2LANGCODE, MODEL2HIDDEN_SIZE, MODEL2NUM_LAYERS, EXP2_CONFIG, EXP3_CONFIG, EXP4_CONFIG, MODEL2HF_NAME
import glob
import torch
from tqdm import tqdm
from glob import glob
import cudf
import cupy as cp
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from src.utils.sheets import get_google_sheet, fill_vocab
import argparse
load_dotenv()

def silhouette_score_across_layers(args):
    model_names = args.model_names
    residual_positions = args.residual_positions
    cuda_device = args.cuda_device
    
    languages = []
    for family, langs in EXP4_CONFIG['languages'].items():
        languages.extend(langs)

    languages = [LANGNAME2LANGCODE[lang] for lang in languages]

    google_sheet_id = '1CmhOZeYTbfePLI2-rMubJpnKHuS6RLEfHYZD6-rVQ0M'
    gid = '0'
    try:
        df_lang = get_google_sheet(google_sheet_id, gid)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    df_lang['Script'] = df_lang['Language code'].apply(lambda x: x.split('_')[-1])
    df_lang['Syntax'] = df_lang['Syntax'].fillna('Unknown')
    df_lang['Language family'] = df_lang['Language family'].fillna('Unknown')
    df_lang['Language sub-family'] = df_lang['Language sub-family'].fillna(df_lang['Language family'])
    df_lang['Language sub-sub-family'] = df_lang['Language sub-sub-family'].fillna(df_lang['Language sub-family'])
    df_lang['Phonetics'] = df_lang['Phonetics'].fillna('Unknown')
    df_lang['Token SmolLM'] = df_lang['Token SmolLM3-3B'].fillna('Unknown')
    df_lang['Token Qwen'] = df_lang['Token Qwen3-14B'].fillna('Unknown')
    df_lang['Token Gemma'] = df_lang['Token gemma-3-12b-it'].fillna('Unknown')
    df_lang['Token GPT'] = df_lang['Token gpt-oss-20b'].fillna('Unknown')

    CATEGORICAL_COLS = [
        'Region',
        "Joshi’s class",
        'Syntax',
        'Language family',
        'Language sub-family',
        'Language sub-sub-family',
        'Script',
        'Phonetics',
        'Token Gemma',
        'Token Qwen',
        'Token SmolLM',
        'Token GPT',
    ]

    for col in CATEGORICAL_COLS:
        df_lang[col] = pd.factorize(df_lang[col])[0] + 1

    REGION_MAP = {}
    JOSHI_MAP = {}
    SYNTAX_MAP = {}
    FAMILY_MAP = {}
    SUBFAMILY_MAP = {}
    SUBSUBFAMILY_MAP = {}
    SCRIPT_MAP = {}
    PHONETICS_MAP = {}
    TOKEN_GEMMA_MAP = {}
    TOKEN_QWEN_MAP = {}
    TOKEN_SMOLLM_MAP = {}
    TOKEN_GPT_MAP = {}
    SUBFAMILY2FAMILY_MAP = {}
    SUBSUBFAMILY2FAMILY_MAP = {}
    SUBSUBFAMILY2SUBFAMILY_MAP = {}

    for index, row in df_lang.iterrows():
        lang_name = row['Language code']

        REGION_MAP[lang_name] = row['Region']
        JOSHI_MAP[lang_name] = row["Joshi’s class"]
        SYNTAX_MAP[lang_name] = row['Syntax']
        FAMILY_MAP[lang_name] = row['Language family']
        SUBFAMILY_MAP[lang_name] = row['Language sub-family']
        SUBSUBFAMILY_MAP[lang_name] = row['Language sub-sub-family']
        SCRIPT_MAP[lang_name] = row['Script']
        PHONETICS_MAP[lang_name] = row['Phonetics']
        TOKEN_GEMMA_MAP[lang_name] = row['Token Gemma']
        TOKEN_QWEN_MAP[lang_name] = row['Token Qwen']
        TOKEN_SMOLLM_MAP[lang_name] = row['Token SmolLM']
        TOKEN_GPT_MAP[lang_name] = row['Token GPT']

    text_ids = glob(f'outputs_flores_plus/next_token/dev/{model_names[0]}/raw/eng_Latn/*/last_token/layer_{residual_positions[0]}_1.pt')
    text_ids = [ti.split('/')[-3] for ti in text_ids]

    text_ids_int = [int(tid) for tid in text_ids]
    text_ids_int.sort()

    for model_name in model_names:
        for residual_position in residual_positions:
            results = []
            # Initialize tensor to hold activations
            labels_per_language = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_semantic = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_family = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_subfamily = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_subsubfamily = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_script = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_phonetics = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_tokenization = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_region = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_joshi = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_per_syntax = torch.zeros((len(text_ids), len(languages))).to(f'cuda:{cuda_device}')
            labels_not_assigned = True
            for layer_id in range(MODEL2NUM_LAYERS[model_name]):
                activations = torch.zeros((len(text_ids), len(languages), MODEL2HIDDEN_SIZE[model_name])).to(f'cuda:{cuda_device}')
                for text_id in tqdm(text_ids, desc=f'Load layer {layer_id} - {residual_position}'):
                    for lang_idx, lang in enumerate(languages):
                        # Load activation
                        activations[int(text_id), lang_idx, :] = torch.load(f'outputs_flores_plus/next_token/dev/{model_name}/raw/{lang}/{text_id}/last_token/layer_{residual_position}_{layer_id}.pt')
                        
                        # Assign labels (text_id and lang_idx indices)
                        if labels_not_assigned:
                            labels_per_language[int(text_id), lang_idx] = lang_idx
                            labels_per_semantic[int(text_id), lang_idx] = int(text_id)
                            labels_per_family[int(text_id), lang_idx] = FAMILY_MAP[lang]
                            labels_per_subfamily[int(text_id), lang_idx] = SUBFAMILY_MAP[lang]
                            labels_per_subsubfamily[int(text_id), lang_idx] = SUBSUBFAMILY_MAP[lang]
                            labels_per_script[int(text_id), lang_idx] = SCRIPT_MAP[lang]
                            labels_per_phonetics[int(text_id), lang_idx] = PHONETICS_MAP[lang]
                            labels_per_tokenization[int(text_id), lang_idx] = TOKEN_GEMMA_MAP[lang] if model_name == 'gemma-3-12b-it' else TOKEN_GPT_MAP[lang] if model_name == 'gpt-oss-20b' else TOKEN_QWEN_MAP[lang] if model_name == 'Qwen3-14B' else TOKEN_SMOLLM_MAP[lang]
                            labels_per_region[int(text_id), lang_idx] = REGION_MAP[lang]
                            labels_per_joshi[int(text_id), lang_idx] = JOSHI_MAP[lang]
                            labels_per_syntax[int(text_id), lang_idx] = SYNTAX_MAP[lang]

                labels_not_assigned = False

                # Reshape activations and labels for silhouette score computation
                activations = activations.reshape(-1, MODEL2HIDDEN_SIZE[model_name])
                labels_per_language = labels_per_language.reshape(-1)
                labels_per_semantic = labels_per_semantic.reshape(-1)
                labels_per_family = labels_per_family.reshape(-1)
                labels_per_subfamily = labels_per_subfamily.reshape(-1)
                labels_per_subsubfamily = labels_per_subsubfamily.reshape(-1)
                labels_per_script = labels_per_script.reshape(-1)
                labels_per_phonetics = labels_per_phonetics.reshape(-1)
                labels_per_tokenization = labels_per_tokenization.reshape(-1)
                labels_per_region = labels_per_region.reshape(-1)
                labels_per_joshi = labels_per_joshi.reshape(-1)
                labels_per_syntax = labels_per_syntax.reshape(-1)

                # Compute silhouette score per text ID
                print(f'Compute silhouette scores for layer {layer_id} - {residual_position}')
                current_results = []

                # Language
                score = cython_silhouette_score(activations, labels_per_language, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'language',
                    'score': score
                })

                # Semantic
                score = cython_silhouette_score(activations, labels_per_semantic, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'semantic',
                    'score': score
                })

                # Family
                score = cython_silhouette_score(activations, labels_per_family, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'family',
                    'score': score
                })

                # Sub Family
                score = cython_silhouette_score(activations, labels_per_subfamily, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'sub_family',
                    'score': score
                })


                # Sub Sub Family
                score = cython_silhouette_score(activations, labels_per_subsubfamily, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'sub_sub_family',
                    'score': score
                })

                # Script
                score = cython_silhouette_score(activations, labels_per_script, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'script',
                    'score': score
                })

                # Phonetic
                score = cython_silhouette_score(activations, labels_per_phonetics, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'phonetics',
                    'score': score
                })

                # Tokenization
                score = cython_silhouette_score(activations, labels_per_tokenization, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'tokenization',
                    'score': score
                })

                # Region
                score = cython_silhouette_score(activations, labels_per_region, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'region',
                    'score': score
                })

                # Joshi
                score = cython_silhouette_score(activations, labels_per_joshi, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'joshi',
                    'score': score
                })

                # Syntax
                score = cython_silhouette_score(activations, labels_per_syntax, metric='euclidean')
                current_results.append({
                    'model_name': model_name,
                    'residual_position': residual_position,
                    'layer_id': layer_id,
                    'factor': 'syntax',
                    'score': score
                })

                for res in current_results:
                    results.append(res)

            results_df = pd.DataFrame(results)
            results_df.to_csv(f'src/silhouette_scores_{model_name}_{residual_position}.csv', index=False)
            print(f'Saved silhouette scores to src/silhouette_scores_{model_name}_{residual_position}.csv')


if '__main__' == __name__:
    parser = argparse.ArgumentParser(
        description="Compute silhouette scores across layers"
    )

    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["gpt-oss-20b"],
        help="List of model names"
    )

    parser.add_argument(
        "--residual-positions",
        nargs="+",
        default=[
            "residual-preattn",
            "residual-postattn",
            "residual-postmlp",
            "attn-output",
            "mlp-output",
        ],
        help="Activation locations"
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index (e.g., 0, 1, 2)"
    )

    args = parser.parse_args()
    silhouette_score_across_layers(args)