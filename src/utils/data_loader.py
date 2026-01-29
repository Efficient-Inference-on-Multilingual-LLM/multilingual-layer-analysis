import os
import torch
from datasets import load_dataset
from src.utils.const import LANGNAME2LANGCODE

def load_prompt_template(lang_code):
    """Mengambil template sesuai kode bahasa (e.g., ind_Latn)"""
    path = f"prompts/topic_classification/{lang_code}.txt"
    if not os.path.exists(path):
        path = "prompts/topic_classification/eng_Latn.txt" # Fallback ke English
    
    if not os.path.exists(path):
        # Jika file fallback pun tidak ada, gunakan template hardcoded
        return "Identify the topic of this text: {text}"
        
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def ensure_data_exists(lang_code, mode, data_dir, tokenizer, cfg):
    """
    Generate data SIB-200 with model-specific naming.
    """
    # Create a safe model tag (e.g., 'gemma-3-4b-it' or 'llama-3.1-8b')
    model_tag = cfg.model.name.split('/')[-1]
    model_data_dir = os.path.join(data_dir, model_tag)
    os.makedirs(model_data_dir, exist_ok=True)

    filename = f"id.{lang_code}.{mode}.pt"
    save_path = os.path.join(model_data_dir, filename)
    if os.path.exists(save_path):
        return save_path

    print(f"Generating {mode} data for {lang_code} ({model_tag})...")
    
    try:
        dataset = load_dataset("openlanguagedata/flores_plus", lang_code, split="dev")
        
        if mode == "raw":
            text_to_process = " ".join(dataset['text'])
        else:
            template = load_prompt_template(lang_code)
            text_to_process = " ".join([template.replace("{text}", t) for t in dataset['text']])

        # Tokenize using the specific model's tokenizer
        tokens = tokenizer(text_to_process, return_tensors="pt")["input_ids"].squeeze(0)

        max_len = cfg.processing.data_block_size
        l = (tokens.size(0) // max_len) * max_len
        
        if l == 0: return None

        torch.save(tokens[:l], save_path)
        print(f"Saved: {filename}")
        return save_path

    except Exception as e:
        print(f"Error {lang_code}: {e}")
        return None