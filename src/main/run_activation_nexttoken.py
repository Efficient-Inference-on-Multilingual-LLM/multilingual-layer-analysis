import argparse
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from ..utils.const import LANGCODE2LANGNAME, LANGNAME2LANGCODE, EXP4_CONFIG
from ..utils.causallm_hooked_model import Gemma3MultimodalHookedModel, CohereDecoderHookedModel, PythiaHookedModel, QwenHookedModel, QwenMoEHookedModel, LlamaHookedModel, SmolLM3HookedModel, GptOssHookedModel, Olmo2HookedModel
from ..utils.seq2seq_hooked_model import T5HookedModel
from ..utils.activation_saver import CohereDecoderActivationSaver, GeneralActivationSaver, Gemma3MultimodalActivationSaver, Olmo2ActivationSaver
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv()

def main(args):
    # Load Model
    print(f'Load model: {args.model_name}')
    
    if args.prompt_lang == 'all':
        prompt_id_saver = 'prompted'
    elif args.prompt_lang == 'no_prompt':
        prompt_id_saver = 'raw'
    else:
        prompt_id_saver = f'prompt_{args.prompt_lang}' 

    if 'aya-expanse' in args.model_name.lower() or 'aya-23' in args.model_name.lower():
        saver = CohereDecoderActivationSaver(args.output_dir, task_id='next_token', data_split=args.data_split, model_name=args.model_name, prompt_id=prompt_id_saver)
    elif 'gemma-3' in args.model_name.lower():
        saver = Gemma3MultimodalActivationSaver(args.output_dir, task_id='next_token', data_split=args.data_split, model_name=args.model_name, prompt_id=prompt_id_saver)
    elif 'olmo-2' in args.model_name.lower():
        saver = Olmo2ActivationSaver(args.output_dir, task_id='next_token', data_split=args.data_split, model_name=args.model_name, prompt_id=prompt_id_saver)
    else:
        saver = GeneralActivationSaver(args.output_dir, task_id='next_token', data_split=args.data_split, model_name=args.model_name, prompt_id=prompt_id_saver)

    if 'gemma-3' in args.model_name.lower():
        hooked_model = Gemma3MultimodalHookedModel(args.model_name, saver=saver)
    elif 'aya-expanse' in args.model_name.lower() or 'aya-23' in args.model_name.lower():
        hooked_model = CohereDecoderHookedModel(args.model_name, saver=saver)
    elif 'pythia' in args.model_name.lower():
        hooked_model = PythiaHookedModel(args.model_name, saver=saver)
    elif 'qwen' in args.model_name.lower():
        if 'moe' in args.model_name.lower():
            hooked_model = QwenMoEHookedModel(args.model_name, saver=saver)
        else:
            hooked_model = QwenHookedModel(args.model_name, saver=saver)
    elif 'meta-llama' in args.model_name.lower():
        hooked_model = LlamaHookedModel(args.model_name, saver=saver)
    elif 'aya-101' in args.model_name.lower():
        hooked_model = T5HookedModel(args.model_name, saver=saver)
    elif 'smollm3' in args.model_name.lower():
        hooked_model = SmolLM3HookedModel(args.model_name, saver=saver)
    elif 'gpt-oss' in args.model_name.lower():
        hooked_model = GptOssHookedModel(args.model_name, saver=saver)
    elif 'olmo-2' in args.model_name.lower():
        hooked_model = Olmo2HookedModel(args.model_name, saver=saver)
    else:
        raise ValueError(f"Model {args.model_name} not supported in this script.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=os.getenv("HF_CACHE_DIR"))

    if args.use_predefined_languages:
        languages_names = []
        for family, langs in EXP4_CONFIG['languages'].items():
            languages_names.extend(langs)
        languages = [LANGNAME2LANGCODE[lang] for lang in languages_names]
        # languages = languages[:40]
        # languages = languages[40:]
    else:
        languages = args.languages
    print(f'Num of Languages: {len(languages)} | Start language: {languages[0]}')

    # Feed Forward
    # for lang in args.languages:
    for lang in languages:
        
        # Load Dataset
        datasets_per_lang = {}
        if args.data_split == 'all':
            datasets_per_lang_temp = {}
            datasets_per_lang_temp[lang] = load_dataset("openlanguagedata/flores_plus", lang, cache_dir=os.getenv("HF_CACHE_DIR"))
            datasets_per_lang[lang] = concatenate_datasets([datasets_per_lang_temp[lang]['dev'], datasets_per_lang_temp[lang]['devtest']])
        else:
            datasets_per_lang[lang] = load_dataset("openlanguagedata/flores_plus", lang, split=args.data_split, cache_dir=os.getenv("HF_CACHE_DIR"))
        
        # Sample Dataset
        if args.sample_size:
            datasets_per_lang[lang] = datasets_per_lang[lang].shuffle(seed=42).select(range(args.sample_size))

        # Load Prompt Template
        if args.prompt_lang == "all": 
            with open(f'./prompts/next_token/{lang}.txt') as f:
                prompt_template = f.read()
        else:
            with open(f'./prompts/next_token/{args.prompt_lang}.txt') as f:
                prompt_template = f.read()
        
        # Iterate Through Each Instance
        for instance in tqdm(datasets_per_lang[lang], desc=f"Processing activation for next token prediction task ({lang})"):
            # Set ID and Language in Saver
            hooked_model.set_saver_id(str(instance['id']))
            hooked_model.set_saver_lang(lang)

            # Check if activations already exist
            if saver.check_exists():
                print(f"Activations already exist for ID {instance['id']} in language {lang}. Skipping...")
                continue

            # Build Prompt Based on Template
            prompt = prompt_template.replace("{text}", instance['text'])

            # Inference
            if args.is_base_model or 'bloom' in args.model_name:
                text = prompt
            else:
                
                # Gemma2 does not support system message
                if 'google/gemma-2' in args.model_name.lower():
                    messages = [
                        {'role': 'user', 'content': prompt}
                    ]
                else:
                    messages = [
                        {'role': 'system', 'content': ''},
                        {'role': 'user', 'content': prompt}
                    ]

                if 'meta-llama' in args.model_name.lower():
                    user_prompt = messages[-1]['content']
                    text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                else:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False 
                    )
            
            inputs = tokenizer([text], return_tensors="pt").to(hooked_model.model.device)
            _ = hooked_model.generate(inputs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract activation for topic classification task")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--prompt_lang", type=str, default='all', help="Prompt language. Use 'all' to use prompt that has the same language as the input sentence, use 'no_prompt' to not use any prompt at all.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--use_predefined_languages", action='store_true', help="Whether to use predefined languages from EXP4_CONFIG")
    parser.add_argument('--languages', type=str, nargs='+', default=['fra_Latn', 'eng_Latn', 'ind_Latn'], help='List of languages')
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use from each language. Use None to use all samples.")
    parser.add_argument('--is_base_model', action='store_true', help='Whether the model is a base model or a instruct model')
    parser.add_argument('--data_split', type=str, default='test', help='Dataset split to use (train/validation/test/all)')

    args = parser.parse_args()
    main(args)
