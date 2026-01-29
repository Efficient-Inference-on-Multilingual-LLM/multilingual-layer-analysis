from .activation_saver import BaseActivationSaver, CohereDecoderActivationSaver, Gemma3MultimodalActivationSaver, Olmo2ActivationSaver
from transformers import AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

load_dotenv() 

class BaseHookedModel:
	"""
	Base class for hooking into different models.
	"""
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		device = "cpu"
		model_dtype = torch.float16
		if torch.cuda.is_available():
			device = "cuda"
			compute_capability = torch.cuda.get_device_capability()[0]

			# Use bfloat16 if supported
			if compute_capability >= 8:
				model_dtype = torch.bfloat16
		
		self.model_name = model_name
		self.saver = saver
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			torch_dtype=model_dtype,
			# device_map=device if 'gpt-oss' not in model_name else 'auto', # TODO: Implement a better device map strategy
			device_map='auto',
			cache_dir=os.getenv("HF_CACHE_DIR"),
			attn_implementation="eager", # To ensure attention weights can be obtained, use eager implementation (line by line with pytorch)
		)
		self.model.eval()
	
	def _setup_hooks(self):
		raise NotImplementedError("This method should be overridden by subclasses.")
	
	def set_saver_id(self, new_id: int):
		self.saver.set_id(new_id)

	def set_saver_lang(self, new_lang: str):
		self.saver.set_lang(new_lang)
		
	def generate(self, inputs):
		with torch.no_grad():
			outputs = self.model(
				**inputs,
				max_new_tokens=1,
				output_attentions=True,
			)
		return outputs

	# Clear hooks for debugging purposes
	def clear_hooks(self):
		if 'bloom' in self.model_name:
			for i, layer in enumerate(self.model.transformer.h):
				layer._forward_hooks.clear()
		else:
			self.model.model.embed_tokens._forward_hooks.clear()
			self.model.model.norm._forward_hooks.clear()
			for i, layer in enumerate(self.model.model.layers):
				layer._forward_hooks.clear()
				layer._forward_pre_hooks.clear()
				
	
class Gemma3MultimodalHookedModel(BaseHookedModel): # For gemma-3 >=4b
	def __init__(self, model_name: str, saver: Gemma3MultimodalActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		self.model.model.language_model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.language_model.layers):

			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention layer norm pre-hook (residual post attention)
			layer.pre_feedforward_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention layer norm hook (post attention output)
			layer.post_attention_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"attn-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-MLP layer norm hook (post MLP output)
			layer.post_feedforward_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"mlp-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# MLP gate/up projection hook (for LAPE)
			layer.mlp.act_fn.register_forward_hook(lambda module, input, output, layer_id=f"mlp-gate-up-proj_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# # Attention weights hook
			# layer.self_attn.register_forward_hook(lambda module, input, output, layer_id=f"attn-weights_{i}": self.saver.hook_fn_attn_weights(module, input, output, layer_id))

class PythiaHookedModel(BaseHookedModel): # For pythia models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.gpt_neox.embed_in.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.gpt_neox.layers):

			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

class CohereDecoderHookedModel(BaseHookedModel): # For cohere decoder models
	def __init__(self, model_name: str, saver: CohereDecoderActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn_embed_tokens(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Init residual hook
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-init_{i}": self.saver.hook_fn_set_initial_residual(module, input, output, layer_id))

			# Post-attention layer norm pre-hook (residual post attention)
			layer.self_attn.register_forward_hook(lambda module, input, output, layer_id=f"residual-postattn_{i}": self.saver.hook_fn_set_attn_output(module, input, output, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn_final_output(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn_input_layernorm(module, input, output, layer_id))

class QwenHookedModel(BaseHookedModel): # For Qwen models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention hook (post attention output)
			layer.self_attn.register_forward_hook(lambda module, input, output, layer_id=f"attn-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-MLP hook (post MLP output)
			layer.mlp.register_forward_hook(lambda module, input, output, layer_id=f"mlp-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# MLP gate/up projection hook (for LAPE)
			layer.mlp.act_fn.register_forward_hook(lambda module, input, output, layer_id=f"mlp-gate-up-proj_{i}": self.saver.hook_fn(module, input, output, layer_id))

class QwenMoEHookedModel(BaseHookedModel): # For Qwen MoE models (tested for >=Qwen1.5)
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention hook (post attention output)
			layer.self_attn.register_forward_hook(lambda module, input, output, layer_id=f"attn-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-MLP hook (post MLP output)
			layer.mlp.register_forward_hook(lambda module, input, output, layer_id=f"mlp-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# TODO: Add MoE-specific hooks if needed

class Olmo2HookedModel(BaseHookedModel): # For Olmo2 models
	def __init__(self, model_name: str, saver: Olmo2ActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.mlp.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.self_attn.register_forward_pre_hook(lambda module, args, kwargs, layer_id=f"residual-preattn_{i}": self.saver.pre_hook_fn_with_kwargs(module, args, kwargs, layer_id), with_kwargs=True)

			# Post-attention hook (post attention output)
			layer.post_attention_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"attn-output_{i}": self.saver.hook_fn(module, input, output, layer_id))
			
			# Post-MLP hook (post MLP output)
			layer.post_feedforward_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"mlp-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# MLP gate/up projection hook (for LAPE)
			layer.mlp.act_fn.register_forward_hook(lambda module, input, output, layer_id=f"mlp-gate-up-proj_{i}": self.saver.hook_fn(module, input, output, layer_id))

class LlamaHookedModel(BaseHookedModel): # For Llama 3 models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

class SmolLM3HookedModel(BaseHookedModel): # For SmolLM3 models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention hook (post attention output)
			layer.self_attn.register_forward_hook(lambda module, input, output, layer_id=f"attn-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-MLP hook (post MLP output)
			layer.mlp.register_forward_hook(lambda module, input, output, layer_id=f"mlp-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# MLP gate/up projection hook (for LAPE)
			layer.mlp.act_fn.register_forward_hook(lambda module, input, output, layer_id=f"mlp-gate-up-proj_{i}": self.saver.hook_fn(module, input, output, layer_id))


class GptOssHookedModel(BaseHookedModel): # For GPT-OSS models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention hook (post attention output)
			layer.self_attn.register_forward_hook(lambda module, input, output, layer_id=f"attn-output_{i}": self.saver.hook_fn(module, input, output, layer_id))

			# Post-MLP hook (post MLP output)
			layer.mlp.register_forward_hook(lambda module, input, output, layer_id=f"mlp-output_{i}": self.saver.hook_fn(module, input, output, layer_id))