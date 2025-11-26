import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings
from safetensors.torch import load_file as safe_load_file

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 5.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        use_mlp (bool, optional): Whether to use MLP architecture. Default is True.
        mlp_hidden_layers (list, optional): Hidden layer sizes for MLP. Default is [128, 64].
        extra_layers_per_head (list, optional): Number of extra ResBlock layers for each head. Default is list(range(medusa_num_heads)).
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        use_mlp=True,
        mlp_hidden_layers=[256, 128], # [128, 64]
        extra_layers_per_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.use_mlp = use_mlp
        self.mlp_hidden_layers = mlp_hidden_layers
        if extra_layers_per_head is None:
            self.extra_layers_per_head = list(range(medusa_num_heads))
        else:
            self.extra_layers_per_head = extra_layers_per_head


class MLPTokenClassifier(nn.Module):
    """
    Option 1: Shared Projection MLP for Medusa Heads.
    
    This module implements a novel architecture where all prediction heads
    share a single projection layer to vocabulary space, promoting parameter
    efficiency and implicit coordination between heads.
    
    Args:
        hidden_dim (int): Input hidden dimension from base model (e.g., 4096).
        hidden_layers (list): Sizes of intermediate layers for dimensionality reduction.
        vocab_size (int): Vocabulary size (e.g., 32000).
        num_heads (int): Number of Medusa prediction heads.
        extra_layers_per_head (list): Number of extra ResBlock layers for each head.
    """
    def __init__(self, hidden_dim, hidden_layers=[256, 128], vocab_size=32000, num_heads=4, extra_layers_per_head=None):
        super().__init__()

        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        if extra_layers_per_head is None:
            extra_layers_per_head = [0] * num_heads
        
        # Shared backbone: Dimensionality reduction
        backbone_layers = []
        input_dim = hidden_dim
        
        for h in hidden_layers:
            backbone_layers.append(nn.Linear(input_dim, h))
            backbone_layers.append(nn.SiLU()) # Changed to SiLU for consistency 11/23/25
            input_dim = h
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Per-head additional layers and projections
        self.heads = nn.ModuleList()
        bottleneck_dim = input_dim
        for i in range(num_heads):
            layers = []
            for _ in range(extra_layers_per_head[i]):
                layers.append(ResBlock(bottleneck_dim))
            layers.append(nn.Linear(bottleneck_dim, vocab_size))
            self.heads.append(nn.Sequential(*layers))

    def forward(self, hidden_state):
        """
        Forward pass with per-head additional layers.
        
        Args:
            hidden_state: (batch_size, seq_len, hidden_dim)
                         e.g., (4, 512, 4096)
        
        Returns:
            logits: (num_heads, batch_size, seq_len, vocab_size)
                   e.g., (4, 4, 512, 32000)
        """
        batch_size, seq_len, hidden_dim = hidden_state.shape
        
        # Shared feature extraction (operates on last dimension)
        features = self.backbone(hidden_state)  # (B, S, bottleneck_dim)
        
        # Per-head processing
        medusa_logits = []
        for head in self.heads:
            logits = head(features)  # (B, S, vocab_size)
            medusa_logits.append(logits)
        
        # Stack to (num_heads, B, S, vocab_size)
        medusa_logits = torch.stack(medusa_logits, dim=0)
        
        return medusa_logits


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head with MLP Architecture.

    This module creates a shared projection MLP for all prediction heads,
    promoting parameter efficiency and implicit coordination between heads.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        use_mlp = getattr(config, 'use_mlp', True)
        mlp_hidden_layers = getattr(config, 'mlp_hidden_layers', [128, 64])
        extra_layers_per_head = getattr(config, 'extra_layers_per_head', list(range(medusa_num_heads)))
        
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.use_mlp = use_mlp
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        
        # CHANGED: Use MLP architecture by default
        if use_mlp:
            print(f"Using MLPTokenClassifier with per-head additional layers")
            print(f"  Backbone: {self.hidden_size} → {' → '.join(map(str, mlp_hidden_layers))}")
            print(f"  Extra layers per head: {extra_layers_per_head}")
            
            self.medusa_head = MLPTokenClassifier(
                hidden_dim=self.hidden_size,
                hidden_layers=mlp_hidden_layers,
                vocab_size=self.vocab_size,
                num_heads=medusa_num_heads,
                extra_layers_per_head=extra_layers_per_head
            )
        else:
            # Fallback to ResBlock architecture
            print(f"Using ResBlock architecture (fallback)")
            self.medusa_head = nn.ModuleList(
                [
                    nn.Sequential(
                        *([ResBlock(self.hidden_size)] * medusa_num_layers),
                        nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                    )
                    for _ in range(medusa_num_heads)
                ]
            )
    
    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            # Respect trained head count from saved MedusaConfig
            base_model_config.medusa_num_heads = getattr(config, "medusa_num_heads", 3)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            base_model_config.use_mlp = getattr(config, "use_mlp", True)
            base_model_config.mlp_hidden_layers = getattr(config, "mlp_hidden_layers", [128, 64])
            base_model_config.extra_layers_per_head = getattr(config, "extra_layers_per_head", list(range(base_model_config.medusa_num_heads)))
            
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            # Prefer local files; support both safetensors and pt
            local_safe = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.safetensors")
            local_pt = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(local_safe):
                medusa_head_state_dict = safe_load_file(local_safe)
            elif os.path.exists(local_pt):
                medusa_head_state_dict = torch.load(local_pt, map_location=model.device)
            else:
                # Try hub - prefer safetensors first
                try:
                    filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.safetensors")
                    medusa_head_state_dict = safe_load_file(filename)
                except Exception:
                    filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
                    medusa_head_state_dict = torch.load(filename, map_location=model.device)
            
            # Load state dict and check for missing/unexpected keys
            missing_keys, unexpected_keys = model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            if missing_keys:
                warnings.warn(f"Missing keys when loading medusa_head: {missing_keys}")
            else:
                print("All medusa_head keys loaded successfully.")
            if unexpected_keys:
                warnings.warn(f"Unexpected keys when loading medusa_head: {unexpected_keys}")
            else:
                print("No unexpected keys in medusa_head.")
            
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        
        # CHANGED: Handle both MLP and ResBlock architectures
        if self.use_mlp:
            # MLP returns all heads at once: (num_heads, B, S, vocab_size)
            medusa_logits = self.medusa_head(hidden_states)
        else:
            # Original ResBlock logic
            medusa_logits = []
            for i in range(self.medusa):
                mhidden_states = self.medusa_head[i](hidden_states)
                # If the head outputs hidden states (legacy), project with base lm_head; otherwise assume logits
                if mhidden_states.shape[-1] != self.vocab_size:
                    mlogits = self.base_model.lm_head(mhidden_states)
                else:
                    mlogits = mhidden_states
                medusa_logits.append(mlogits)
            medusa_logits = torch.stack(medusa_logits, dim=0)
        
        if output_orig:
            return medusa_logits, outputs, orig
        return medusa_logits
    
    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

    def baseline_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.7,
        max_steps=512,
        top_p=0.9,
    ):
        """
        Baseline generation without Medusa speculative decoding.
        Uses the same past_key_values initialization as medusa_generate for consistency.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for sampling. Use 0.0 for greedy decoding.
            max_steps (int, optional): Maximum number of generation steps.
            top_p (float, optional): Nucleus sampling threshold.

        Returns:
            Generator yielding dicts with 'text' key containing decoded output.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"

        # Clone to avoid modifying input
        input_ids = input_ids.clone()

        # Initialize the past key and value states (same as medusa_generate)
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        # Reset medusa mode to ensure we're using standard attention
        reset_medusa_mode(self)

        # Initial forward pass with full input_ids
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)

        for step in range(max_steps):
            logits = outputs.logits[:, -1, :]

            # Sample next token
            if temperature < 1e-4:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature sampling with optional top-p (nucleus sampling)
                probs = torch.softmax(logits / temperature, dim=-1)

                if top_p < 1.0:
                    # Nucleus sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs > top_p
                    mask[:, 0] = False  # Keep at least one token
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                    next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, next_token_idx)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Forward pass with only the new token (using cached past_key_values)
            outputs = self.base_model(next_token, use_cache=True, past_key_values=past_key_values)

            # Yield current output
            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            # Check for EOS
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    pass

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass


class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")

