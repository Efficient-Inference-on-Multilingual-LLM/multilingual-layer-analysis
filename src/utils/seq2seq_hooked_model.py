from .activation_saver import BaseActivationSaver
from transformers import AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import os

load_dotenv() 

class BaseHookedSeq2SeqModel:
	"""
	Base class for hooking into different seq2seq models.
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
		self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=model_dtype, device_map=device, cache_dir=os.getenv("HF_CACHE_DIR"))
		self.model.eval()

	def _setup_hooks(self):
		raise NotImplementedError("This method should be overridden by subclasses.")
	
	def set_saver_id(self, new_id: int):
		self.saver.set_id(new_id)

	def set_saver_lang(self, new_lang: str):
		self.saver.set_lang(new_lang)
		
	def generate(self, inputs):
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=1,
			)
		return 

class T5HookedModel(BaseHookedSeq2SeqModel):
	"""
	Hooked model for T5 architecture.
	"""
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()
		
	def _setup_hooks(self):
		# Hook embedding layer
		self.model.decoder.embed_tokens.register_forward_hook(
			lambda module, input, output: self.saver.hook_fn(module, input, output, layer_id="embed_tokens")
		)

		for i, block in enumerate(self.model.decoder.block):
			# Self-attention residual
			block.layer[0].register_forward_hook(lambda module, input, output, layer_id=f"residual-postselfattn_{i}": self.saver.hook_fn(module, input, output, layer_id=layer_id))
			
			# Cross-attention residual
			block.layer[1].register_forward_hook(lambda module, input, output, layer_id=f"residual-postcrossattn_{i}": self.saver.hook_fn(module, input, output, layer_id=layer_id))
			
			# MLP residual
			block.layer[2].register_forward_hook(lambda module, input, output, layer_id=f"residual-postmlp_{i}": self.saver.hook_fn(module, input, output, layer_id=layer_id))
        
