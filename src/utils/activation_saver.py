import os
import torch
from .const import MODEL2NUM_LAYERS

class BaseActivationSaver:
	"""
	Base class for saving activations from different models.
	"""
	def __init__(self, base_save_dir: str, task_id: str, data_split: str, model_name: str, prompt_id: str):
		self.task_id = task_id
		self.data_split = data_split
		self.model_name = model_name
		self.prompt_id = prompt_id
		self.base_save_dir = base_save_dir
		self.current_id = None
		self.current_lang = None

	def set_id(self, new_id):
		self.current_id = new_id

	def set_lang(self, new_lang):
		self.current_lang = new_lang

	def hook_fn(self, module, input, output, layer_id):
		raise NotImplementedError("This method should be overridden by subclasses.")

	def pre_hook_fn(self, module, input, layer_id):
		raise NotImplementedError("This method should be overridden by subclasses.")

	# Check if activations for an instance already exist. TODO: Use AutoConfig to generalize this function for different models used
	def check_exists(self):
		path_last_token = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
		# path_average = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "average")
		# path_attn = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "attention_weights")
		
		# # Check if attention weights directory has files
		# check_files_attn = os.listdir(path_attn) if os.path.exists(path_attn) else []
		# if len(check_files_attn) != MODEL2NUM_LAYERS[self.model_name.split('/')[-1]]:
		# 	print(f"Warning: Incomplete attention weights for ID {self.current_id} in language {self.current_lang}, rerunning extraction.")
		# 	return False

		# Check if each extraction exist
		check_files = os.listdir(path_last_token) if os.path.exists(path_last_token) else []
		pre_attn_files_sum = 0
		post_attn_files_sum = 0
		post_mlp_files_sum = 0
		embed_token_file_sum = 0
		post_output_attn_sum = 0
		post_output_mlp_sum = 0
		for f in check_files:
			if 'preattn' in f:
				pre_attn_files_sum += 1
			elif 'postattn' in f:
				post_attn_files_sum += 1
			elif 'postmlp' in f:
				post_mlp_files_sum += 1
			elif 'embed_tokens' in f:
				embed_token_file_sum += 1
			elif 'attn-output' in f:
				post_output_attn_sum += 1
			elif 'mlp-output' in f:
				post_output_mlp_sum += 1
		if pre_attn_files_sum == 0 or post_attn_files_sum == 0 or post_mlp_files_sum == 0 or embed_token_file_sum == 0 or post_output_attn_sum == 0 or post_output_mlp_sum == 0:
			return False
		
		# # Check average directory files
		# check_files_avg = os.listdir(path_average) if os.path.exists(path_average) else []

		# If there are any files, return True
		# return bool(check_files) and bool(check_files_avg) and bool(check_files_attn)
		return bool(check_files)

	def _save_activation_last_token(self, tensor, layer_id):
		path = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
		os.makedirs(path, exist_ok=True)
		save_path = os.path.join(path, f"layer_{layer_id}.pt")
		torch.save(tensor[0, -1, :].detach().cpu(), save_path)

	def _save_activation_average(self, tensor, layer_id):
		path = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "average")
		os.makedirs(path, exist_ok=True)
		save_path = os.path.join(path, f"layer_{layer_id}.pt")
		torch.save(tensor[0].mean(dim=0).detach().cpu(), save_path)
	
	def _save_attention_weights(self, tensor, layer_id):
		path = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "attention_weights")
		os.makedirs(path, exist_ok=True)
		save_path = os.path.join(path, f"layer_{layer_id}.pt")
		torch.save(tensor.detach().cpu(), save_path)
	
	def _check_set_id_lang(self, layer_id):
		if self.current_id is None:
			print(f"Warning: ID not set for layer {layer_id}")
			return False

		if self.current_lang is None:
			print(f"Warning: Language not set for layer {layer_id}")
			return False
		
		return True
	
class GeneralActivationSaver(BaseActivationSaver): # Handle Gemma3, Qwen, Pythia models (models that activation are returned in form of a tuple)

	def hook_fn(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		try:
			self._save_activation_last_token(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id) # Unpack tensor from the tuple
			# self._save_activation_average(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn for layer {layer_id}: {e}")
	
	def pre_hook_fn(self, module, input, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return

		try:
			# Extract residual connection after attention (precisely after post-attention layer norm)
			self._save_activation_last_token(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
			# self._save_activation_average(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
		except Exception as e:
			print(f"Error in pre_hook_fn for layer {layer_id}: {e}")

class Olmo2ActivationSaver(BaseActivationSaver): # Handle Gemma3, Qwen, Pythia models (models that activation are returned in form of a tuple)

	def hook_fn(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		try:
			self._save_activation_last_token(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id) # Unpack tensor from the tuple
			# self._save_activation_average(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn for layer {layer_id}: {e}")
	
	def pre_hook_fn_with_kwargs(self, module, args, kwargs, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		try:
			self._save_activation_last_token(tensor=kwargs['hidden_states'][0] if isinstance(kwargs['hidden_states'], tuple) else kwargs['hidden_states'], layer_id=layer_id) # Unpack tensor from the tuple
			# self._save_activation_average(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn_save_input for layer {layer_id}: {e}")
	
	def pre_hook_fn(self, module, input, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return

		try:
			self._save_activation_last_token(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
			# self._save_activation_average(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
		except Exception as e:
			print(f"Error in pre_hook_fn for layer {layer_id}: {e}")

class CohereDecoderActivationSaver(BaseActivationSaver):
	def __init__(self, base_save_dir: str, task_id: str, data_split: str, model_name: str, prompt_id: str):
		super().__init__(base_save_dir, task_id, data_split, model_name, prompt_id)
		self.initial_residual = None
		self.attn_output = None
	
	def hook_fn_embed_tokens(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		
		try:
			self._save_activation_last_token(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
			# self._save_activation_average(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn_embed_tokens for layer {layer_id}: {e}")

	def hook_fn_set_initial_residual(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		
		self.initial_residual = input[0] if isinstance(input, tuple) else input
	
	def hook_fn_set_attn_output(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		self.attn_output = output[0] if isinstance(output, tuple) else output
	
	def hook_fn_final_output(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return

		# Compute residual post MLP
		if self.initial_residual is None or self.attn_output is None:
			print(f"Warning: Missing stored tensors for layer {layer_id}")
			raise ValueError("Stored tensors are None")
		
		residual_post_mlp = output[0] if isinstance(output, tuple) else output
		residual_post_attn = self.initial_residual + self.attn_output

		try:
			self._save_activation_last_token(tensor=residual_post_attn, layer_id=layer_id.replace('residual-postmlp', 'residual-postattn'))
			# self._save_activation_average(tensor=residual_post_attn, layer_id=layer_id.replace('residual-postmlp', 'residual-postattn'))
		except Exception as e:
			print(f"Error in hook_fn_final_output for layer {layer_id.replace('residual-postmlp', 'residual-postattn')}: {e}")
		
		try:
			self._save_activation_last_token(tensor=residual_post_mlp, layer_id=layer_id)
			# self._save_activation_average(tensor=residual_post_mlp, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn_final_output for layer {layer_id}: {e}")

		# Reset stored tensors
		self.initial_residual = None
		self.attn_output = None
	
	def hook_fn_input_layernorm(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		try:
			self._save_activation_last_token(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id) # Unpack tensor from the tuple
			# self._save_activation_average(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn for layer {layer_id}: {e}")

class Gemma3MultimodalActivationSaver(BaseActivationSaver): # Handle Gemma3, Qwen, Pythia models (models that activation are returned in form of a tuple)

	def hook_fn(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		try:
			self._save_activation_last_token(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id) # Unpack tensor from the tuple
			# self._save_activation_average(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn for layer {layer_id}: {e}")
	
	def pre_hook_fn(self, module, input, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return

		try:
			# Extract residual connection after attention (precisely after post-attention layer norm)
			self._save_activation_last_token(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
			# self._save_activation_average(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
		except Exception as e:
			print(f"Error in pre_hook_fn for layer {layer_id}: {e}")
	
	def hook_fn_attn_weights(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		try:
			attn_weight = output[1] if isinstance(output, tuple) and len(output) > 1 else torch.tensor([])
			self._save_attention_weights(tensor=attn_weight, layer_id=layer_id) # Unpack attention weights from the tuple if available

			# If tensor is empty, log a warning
			if attn_weight.numel() == 0:
				print(f"Warning: No attention weights found in hook_fn_attn_weights for layer {layer_id}")

		except Exception as e:
			print(f"Error in hook_fn_attn_weights for layer {layer_id}: {e}")