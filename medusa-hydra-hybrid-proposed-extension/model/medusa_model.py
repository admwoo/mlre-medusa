import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

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
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        tree_depth=None,
        sequential_num_heads=0,
        use_hybrid=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tree_depth = tree_depth  # Number of heads for tree decoding (Medusa style)
        self.sequential_num_heads = sequential_num_heads  # Number of heads for sequential decoding (Hydra style)
        self.use_hybrid = use_hybrid  # Whether to use hybrid decoding

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
        num_condition (int, optional): Number of conditioning inputs. If > 0, the block accepts
            concatenated inputs (base + condition embeddings). Default is 0.
    """

    def __init__(self, hidden_size, num_condition=0):
        super().__init__()
        self.linear = nn.Linear(hidden_size * (num_condition + 1), hidden_size)
        # Handling residual connection when reducing dim
        if num_condition > 0:
            self.res_connection = nn.Linear(hidden_size * (num_condition + 1), hidden_size)
        else:
            self.res_connection = nn.Identity()
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor. If num_condition > 0, should be concatenated
                [base_hidden_state, condition_embeddings...].

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return self.res_connection(x) + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

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
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        
        # Hybrid decoding configuration
        self.use_hybrid = getattr(config, 'use_hybrid', False)
        self.tree_depth = getattr(config, 'tree_depth', None)
        self.sequential_num_heads = getattr(config, 'sequential_num_heads', 0)
        
        if self.use_hybrid and self.tree_depth is not None:
            # Hybrid mode: split heads into tree and sequential
            tree_num_heads = self.tree_depth
            sequential_num_heads = self.sequential_num_heads
            # Ensure total heads match
            assert tree_num_heads + sequential_num_heads == medusa_num_heads, \
                f"tree_depth ({tree_num_heads}) + sequential_num_heads ({sequential_num_heads}) must equal medusa_num_heads ({medusa_num_heads})"
            
            # Tree heads (Medusa style - independent) - output vocab logits directly
            self.medusa_head = nn.ModuleList(
                [
                    nn.Sequential(
                        *([ResBlock(self.hidden_size)] * medusa_num_layers),
                        nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                    )
                    for _ in range(tree_num_heads)
                ]
            )
            
            # Sequential heads (Hydra style - conditioned on candidate sequences) - output vocab logits directly
            if sequential_num_heads > 0:
                self.sequential_head = nn.ModuleList(
                    [
                        nn.Sequential(
                            ResBlock(self.hidden_size, hydra_head_idx + 1),  # First layer conditions on head_idx+1 tokens
                            *([ResBlock(self.hidden_size)] * (medusa_num_layers - 1)),
                            nn.Linear(self.hidden_size, self.vocab_size, bias=False),  # Direct vocab projection
                        )
                        for hydra_head_idx in range(sequential_num_heads)
                    ]
                )
            else:
                self.sequential_head = None
        else:
            # Standard Medusa mode (all heads independent) - output vocab logits directly
            self.medusa_head = nn.ModuleList(
                [
                    nn.Sequential(
                        *([ResBlock(self.hidden_size)] * medusa_num_layers),
                        nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                    )
                    for _ in range(medusa_num_heads)
                ]
            )
            self.sequential_head = None
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
        # Check if config is provided in kwargs
        if 'config' in kwargs:
            # Use provided config (should be MedusaConfig with medusa attributes)
            config = kwargs.pop('config')
            # Remove config from kwargs to avoid duplicate
            kwargs_no_config = {k: v for k, v in kwargs.items() if k != 'config'}
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs_no_config,
                config=config,
            )
        else:
            # Try to load MedusaConfig first, fallback to AutoConfig
            try:
                config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
                base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
                # Respect trained head count from saved MedusaConfig
                base_model_config.medusa_num_heads = getattr(config, "medusa_num_heads", 3)
                base_model_config.medusa_num_layers = config.medusa_num_layers
                # Remove config from kwargs
                kwargs_no_config = {k: v for k, v in kwargs.items() if k != 'config'}
                model = super().from_pretrained(
                    config.base_model_name_or_path,
                    *args,
                    **kwargs_no_config,
                    config=base_model_config,
                )
            except:
                # Load base config and convert to MedusaConfig
                base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
                config_dict = base_config.to_dict()
                # Add default medusa attributes if not present
                if 'medusa_num_heads' not in config_dict:
                    config_dict['medusa_num_heads'] = getattr(base_config, 'medusa_num_heads', 5)
                if 'medusa_num_layers' not in config_dict:
                    config_dict['medusa_num_layers'] = getattr(base_config, 'medusa_num_layers', 1)
                if 'base_model_name_or_path' not in config_dict:
                    config_dict['base_model_name_or_path'] = pretrained_model_name_or_path
                config = MedusaConfig(**config_dict)
                # Remove config from kwargs to avoid duplicate
                kwargs_no_config = {k: v for k, v in kwargs.items() if k != 'config'}
                return super().from_pretrained(
                    pretrained_model_name_or_path,
                    *args,
                    **kwargs_no_config,
                    config=config,
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
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            if hasattr(model, 'sequential_head') and model.sequential_head is not None:
                seq_head_keys = [k for k in medusa_head_state_dict.keys() if k.startswith('sequential_head.')]
                if seq_head_keys:
                    # Extract sequential head state dict
                    seq_state_dict = {}
                    for key in seq_head_keys:
                        new_key = key.replace('sequential_head.', '')
                        seq_state_dict[new_key] = medusa_head_state_dict[key]
                    model.sequential_head.load_state_dict(seq_state_dict, strict=False)
                    print(f"Loaded {len(seq_head_keys)} sequential head parameters")

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
        candidate_sequences=None,  # For sequential heads: [batch, num_candidates, seq_len]
        base_hidden_states=None,  # Precomputed base hidden states for sequential heads
        labels=None,  # Labels for training
        run_sequential_heads=False,  # Flag to run sequential heads during training
        noise_alpha=0.0,  # Noise alpha for training (NEFT-tune)
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
            candidate_sequences (torch.Tensor, optional): Candidate token sequences for sequential heads.
            base_hidden_states (torch.Tensor, optional): Precomputed base hidden states.
            run_sequential_heads (bool, optional): Whether to run sequential heads (for training).
            noise_alpha (float, optional): Noise alpha for training regularization.

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
                labels=labels,
                **kwargs,
            )
        
        # Determine if we're in training mode
        training_mode = self.training and run_sequential_heads
        
        # For training: use no_grad for base model but allow gradients through heads
        # For inference: use inference_mode (more efficient)
        if base_hidden_states is not None:
            # Precomputed hidden states - use as is
            outputs = None
            hidden_states = base_hidden_states
            if output_orig:
                with torch.no_grad():
                    orig = self.base_model.lm_head(hidden_states)
        else:
            # Compute hidden states from base model
            # Base model is frozen, but we clone to allow gradient flow through heads
            if training_mode:
                # Use no_grad for base model forward pass
                with torch.no_grad():
                    outputs = self.base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        **kwargs,
                    )
                    if output_orig:
                        orig = self.base_model.lm_head(outputs[0])
                
                # Clone without detach to allow gradients through heads
                # Base model params have requires_grad=False, so they won't be updated
                hidden_states = outputs[0].clone()
                hidden_states.requires_grad_(True)
            else:
                # Inference mode: use inference_mode for efficiency
                with torch.inference_mode():
                    outputs = self.base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        **kwargs,
                    )
                    if output_orig:
                        orig = self.base_model.lm_head(outputs[0])
                hidden_states = outputs[0].clone()
        
        # Apply noise for training (NEFT-tune) to hidden states
        if training_mode and noise_alpha > 0.0:
            model_dim = hidden_states.shape[-1]
            seq_len = (input_ids != self.tokenizer.pad_token_id).sum(dim=-1).clamp(min=1).unsqueeze(1).unsqueeze(2)
            denom = torch.sqrt(seq_len * model_dim)
            noise = (torch.rand_like(hidden_states) * 2 - 1) * noise_alpha / denom
            noise = noise.to(hidden_states.dtype)
            hidden_states = hidden_states + noise
        
        medusa_logits = []
        
        # Process tree heads (Medusa style - independent)
        tree_num_heads = len(self.medusa_head)
        for i in range(tree_num_heads):
            # Heads now output vocab logits directly
            mlogits = self.medusa_head[i](hidden_states)
            medusa_logits.append(mlogits)
        
        # Process sequential heads during training
        sequential_logits = None
        if training_mode and self.sequential_head is not None and run_sequential_heads:
            sequential_logits = self._forward_sequential_heads_training(
                hidden_states, input_ids, noise_alpha
            )
            # Append sequential head logits to medusa_logits
            if sequential_logits is not None:
                medusa_logits.extend(sequential_logits)
        
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
    
    def _forward_sequential_heads_training(self, base_hidden_states, input_ids, noise_alpha=0.0):
        """
        Forward pass for sequential heads during training.
        
        This mimics Hydra's training approach:
        - Each sequential head conditions on shifted input embeddings
        - Head i conditions on base_hidden_state + embeddings from previous (i+1) positions
        
        Args:
            base_hidden_states: Base model hidden states [batch, seq_len, hidden_size]
            input_ids: Input token IDs [batch, seq_len]
            noise_alpha: Noise alpha for regularization
            
        Returns:
            List of logits from sequential heads
        """
        if self.sequential_head is None:
            return None
        
        sequential_logits = []
        sequential_num_heads = len(self.sequential_head)
        
        # Get input embeddings
        # Base model embedding layer is frozen, use no_grad and detach
        with torch.no_grad():
            input_embeds = self.base_model.model.embed_tokens(input_ids)
        input_embeds = input_embeds.detach()
        
        # Prepare shifted embeddings for each sequential head
        # Head i should condition on the previous (i+1) tokens
        # At position t, head i needs embeddings from [t-1, t-2, ..., t-(i+1)]
        batch_size, seq_len, hidden_size = input_embeds.shape
        
        hydra_inputs = [base_hidden_states]
        for i in range(sequential_num_heads):
            # Create shifted embeddings: shift right by (i+1) positions, pad left with zeros
            shift_amount = i + 1
            pad = torch.zeros(batch_size, shift_amount, hidden_size, 
                            device=input_embeds.device, dtype=input_embeds.dtype)
            if seq_len > shift_amount:
                shifted = torch.cat([pad, input_embeds[:, :-shift_amount, :]], dim=1)
            else:
                # Sequence too short, just use padding
                shifted = pad[:, :seq_len, :]
            hydra_inputs.append(shifted)
        
        # Process each sequential head
        for head_idx in range(sequential_num_heads):
            head = self.sequential_head[head_idx]
            # Head i conditions on base_hidden_state + embeddings[0] to embeddings[i]
            # Concatenate: [base_hidden_states, shifted_embeds[0], ..., shifted_embeds[head_idx]]
            head_input = torch.cat(hydra_inputs[:head_idx + 2], dim=-1)  # [batch, seq_len, hidden_size * (head_idx + 2)]
            
            # Process through head - now outputs vocab logits directly
            head_output = head(head_input)  # [batch, seq_len, vocab_size]
            
            sequential_logits.append(head_output)
        
        return sequential_logits
    
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
        
        # Check if hybrid mode is enabled
        use_hybrid = getattr(self, 'use_hybrid', False)
        tree_depth = getattr(self, 'tree_depth', None)
        sequential_num_heads = getattr(self, 'sequential_num_heads', 0)
        
        # If hybrid mode, filter medusa_choices to only tree_depth levels
        if use_hybrid and tree_depth is not None:
            # Filter medusa_choices to only include paths up to tree_depth
            filtered_medusa_choices = [c for c in medusa_choices if len(c) <= tree_depth]
            # Regenerate buffers with filtered choices
            medusa_buffers = generate_medusa_buffers(
                filtered_medusa_choices, device=self.base_model.device
            )
            self.medusa_buffers = medusa_buffers
            self.medusa_choices = filtered_medusa_choices
        
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )
        
        # Get base hidden states for sequential heads (if hybrid mode)
        base_hidden_states = None
        if use_hybrid and sequential_num_heads > 0:
            with torch.inference_mode():
                outputs_base = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                base_hidden_states = outputs_base[0].clone()

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates - use hybrid or standard approach
            if use_hybrid and tree_depth is not None and sequential_num_heads > 0:
                # Hybrid mode: tree decoding + sequential continuation
                candidates, tree_candidates = generate_hybrid_candidates(
                    self,
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                    tree_depth,
                    sequential_num_heads,
                    base_hidden_states,
                    temperature=temperature,
                    posterior_alpha=posterior_alpha,
                    posterior_threshold=posterior_threshold,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                )
            else:
                # Standard Medusa mode
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

            # Verify candidates - use different method for hybrid vs standard mode
            base_logit_for_eval = None
            if use_hybrid and tree_depth is not None and sequential_num_heads > 0:
                # Hybrid mode: verify extended candidates through base model
                base_seq_len = input_ids.shape[1]
                verification_logits, verification_outputs, base_token_logit = verify_hybrid_candidates(
                    self,
                    candidates,  # Extended candidates [num_candidates, seq_len]
                    past_key_values,
                    input_ids,
                    base_seq_len,
                )
                # verification_logits: [num_candidates, seq_len, vocab_size]
                # verification_outputs: model outputs from verification
                # base_token_logit: extracted base token logit (if available)
                
                # Use extracted base token logit if available, otherwise compute it
                if base_token_logit is not None:
                    base_logit_for_eval = base_token_logit  # [1, 1, vocab_size]
                    # No need for redundant forward pass
                    base_outputs = None
                else:
                    # Fallback: compute base logit (shouldn't happen with updated verify_hybrid_candidates)
                    with torch.inference_mode():
                        base_outputs = self.base_model.model(
                            input_ids,
                            past_key_values=past_key_values,
                        )
                        base_logit_for_eval = self.base_model.lm_head(base_outputs[0][:, -1:, :])  # [1, 1, vocab_size]
                
                logits = verification_logits  # [num_candidates, seq_len, vocab_size]
                # Create dummy medusa_logits for compatibility (not used in hybrid mode evaluation)
                medusa_logits = torch.zeros((tree_depth, 1, candidates.shape[0], self.vocab_size), 
                                           device=candidates.device, dtype=logits.dtype)
                # Use verification outputs if available
                if verification_outputs is not None:
                    outputs = verification_outputs
                elif base_outputs is not None:
                    outputs = base_outputs
                else:
                    outputs = None
            else:
                # Standard mode: use tree decoding
                medusa_logits, logits, outputs = tree_decoding(
                    self,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )
            
            # Update base hidden states for next iteration (if hybrid)
            if use_hybrid and sequential_num_heads > 0:
                with torch.inference_mode():
                    # Get base hidden states from the last verification
                    if outputs is not None:
                        base_hidden_states = outputs[0].clone()
                    else:
                        # Fallback: recompute from input_ids
                        outputs_base = self.base_model.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                        )
                        base_hidden_states = outputs_base[0].clone()

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            if use_hybrid and tree_depth is not None and sequential_num_heads > 0 and base_logit_for_eval is not None:
                # Hybrid mode: prepend base token for evaluation
                base_token = input_ids[0, -1:].unsqueeze(0)  # [1, 1]
                candidates_with_base = torch.cat([base_token.expand(candidates.shape[0], -1), candidates], dim=1)
                base_logit_expanded = base_logit_for_eval.expand(candidates.shape[0], -1, -1)  # [num_candidates, 1, vocab_size]
                logits_with_base = torch.cat([base_logit_expanded, logits], dim=1)  # [num_candidates, seq_len+1, vocab_size]
                
                best_candidate, accept_length = evaluate_posterior(
                    logits_with_base, candidates_with_base, temperature, 
                    posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
                )
                # Adjust accept_length (subtract 1 for base token, but keep at least 0)
                accept_length = max(0, accept_length - 1) if accept_length > 0 else 0
            else:
                best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
                )

            # Update the input_ids and logits
            if use_hybrid and tree_depth is not None and sequential_num_heads > 0:
                # Hybrid mode: use specialized update function
                input_ids, logits, medusa_logits, new_token = update_inference_inputs_hybrid(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    outputs,
                    logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )
            else:
                # Standard mode: use regular update function
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