import os
from dotenv import load_dotenv
from huggingface_hub import login
import argparse
from types import MethodType
import torch
import torch.nn as nn
from vllm import LLM, SamplingParams

login(os.getenv('HF_TOKEN'))

# 1. SETUP: Add gemma to parsing logic
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="google/gemma-3-4b-it")
parser.add_argument("-l", "--lang", type=str, default="en")
args = parser.parse_args(args=["--model", "google/gemma-3-4b-it", "--lang", "en"]) #args = parser.parse_args()

model_name = args.model.lower()
is_llama = 'llama' in model_name
is_gemma = 'gemma' in model_name  # New flag for Gemma
print(torch.cuda.device_count())
# Initialize vLLM
model = LLM(
    model=args.model, 
    tensor_parallel_size=1, 
    enforce_eager=True,
    dtype="bfloat16",      
    max_model_len=4096,  
    gpu_memory_utilization=0.9
)

# 2. CONFIG: Gemma uses 'intermediate_size' just like LLaMA
config = model.llm_engine.model_config.hf_config
max_length = model.llm_engine.model_config.max_model_len
num_layers = config.num_hidden_layers

# LLaMA and Gemma define explicit intermediate_size; BLOOM usually calculates it as 4*hidden
if is_llama or is_gemma:
    intermediate_size = config.intermediate_size
else:
    intermediate_size = config.hidden_size * 4

# Initialize the counter tensor
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

# 3. FACTORY: Add specific forward pass for Gemma
def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1)
        gate_up[:, :, : i // 2] = nn.SiLU()(gate_up[:, :, : i // 2]) # LLaMA uses SiLU
        activation = gate_up[:, :, : i // 2].float()
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def gemma_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        
        # Gemma 3 uses GELU
        gate_up[:, :, : i // 2] = nn.GELU(approximate='tanh')(gate_up[:, :, : i // 2])
        
        # Crucial: Cast to float32 BEFORE summation to avoid overflow/precision issues
        activation = gate_up[:, :, : i // 2].float() 
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        
        x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        activation = x.float()
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    elif is_gemma:
        return gemma_forward # Return the new Gemma logic
    else:
        return bloom_forward

# 4. INJECTION: Gemma follows the LLaMA structure in vLLM
for i in range(num_layers):
    if is_llama or is_gemma:
        # Both models usually live at model.layers[i].mlp in vLLM's internal structure
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    
    obj.forward = MethodType(factory(i), obj)

# 5. EXECUTION (Standard)
lang = args.lang
# Note: You might need to adjust the filename logic if you saved Gemma data differently
if is_llama:
    ids = torch.load(f'data/id.{lang}.train.llama')
elif is_gemma:
    ids = torch.load(f'data/id.{lang}.train.gemma') # Assuming you tokenized for Gemma
else:
    ids = torch.load(f'data/id.{lang}.train.bloom')

l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)

output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))

output = dict(n=l, over_zero=over_zero.to('cpu'))

# Save output
if is_llama:
    torch.save(output, f'data/activation.{lang}.train.llama')
elif is_gemma:
    torch.save(output, f'data/activation.{lang}.train.gemma')
else:
    torch.save(output, f'data/activation.{lang}.train.bloom')