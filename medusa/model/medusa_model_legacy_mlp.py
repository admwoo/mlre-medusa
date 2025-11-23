import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 4.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        use_mlp (bool, optional): Whether to use MLP architecture. Default is True.
        mlp_hidden_layers (list, optional): Hidden layer sizes for MLP. Default is [128, 64].
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        version="2",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        use_mlp=True,
        mlp_hidden_layers=[128, 64],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.version = version
        self.base_model_name_or_path = base_model_name_or_path
        self.use_mlp = use_mlp
        self.mlp_hidden_layers = mlp_hidden_layers


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
    """
    def __init__(self, hidden_dim, hidden_layers=[128, 64], vocab_size=32000, num_heads=4):
        super().__init__()

        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        # Shared backbone: Dimensionality reduction
        backbone_layers = []
        input_dim = hidden_dim
        
        for h in hidden_layers:
            backbone_layers.append(nn.Linear(input_dim, h))
            backbone_layers.append(nn.ReLU())
            input_dim = h
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # NOVEL: Single shared projection to all heads simultaneously
        # Maps from bottleneck dimension to (num_heads * vocab_size)
        self.shared_projection = nn.Linear(input_dim, num_heads * vocab_size)

    def forward(self, hidden_state):
        """
        Forward pass with shared projection.
        
        Args:
            hidden_state: (batch_size, seq_len, hidden_dim)
                         e.g., (4, 512, 4096)
        
        Returns:
            logits: (num_heads, batch_size, seq_len, vocab_size)
                   e.g., (4, 4, 512, 32000)
        """
        batch_size, seq_len, hidden_dim = hidden_state.shape
        
        # Shared feature extraction (operates on last dimension)
        features = self.backbone(hidden_state)  # (B, S, final_hidden_dim)
        
        # Single projection for all heads
        all_logits = self.shared_projection(features)  # (B, S, num_heads * vocab_size)
        
        # Reshape to separate heads
        logits = all_logits.view(batch_size, seq_len, self.num_heads, self.vocab_size)
        
        # Transpose to match expected Medusa format: (num_heads, B, S, vocab_size)
        logits = logits.permute(2, 0, 1, 3)
        
        return logits


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


class MedusaModel(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        use_mlp=True,
        mlp_hidden_layers=[128, 64],
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 4.
            medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 1.
            base_model_name_or_path (str, optional): Path to base model. Defaults to "lmsys/vicuna-7b-v1.3".
            use_mlp (bool, optional): If True, use MLPTokenClassifier (Option 1: shared projection).
                                     If False, use original ResBlock architecture. Defaults to False.
            mlp_hidden_layers (list, optional): Hidden layer sizes for MLP backbone. Defaults to [128, 64].
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.use_mlp = use_mlp
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        
        # Choose architecture based on use_mlp flag
        if use_mlp:
            # Option 1: Shared projection MLP
            print(f"Using MLPTokenClassifier with shared projection (Option 1)")
            print(f"  Backbone: {self.hidden_size} → {' → '.join(map(str, mlp_hidden_layers))}")
            print(f"  Shared projection: {mlp_hidden_layers[-1]} → {medusa_num_heads * self.vocab_size}")
            
            self.medusa_head = MLPTokenClassifier(
                hidden_dim=self.hidden_size,
                hidden_layers=mlp_hidden_layers,
                vocab_size=self.vocab_size,
                num_heads=medusa_num_heads
            )
        else:
            # Original ResBlock architecture
            print(f"Using original ResBlock architecture")
            self.medusa_head = nn.ModuleList(
                [
                    nn.Sequential(
                        *([ResBlock(self.hidden_size)] * medusa_num_layers),
                        nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                    )
                    for _ in range(medusa_num_heads)
                ]
            )

        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        medusa_head_name_or_path,
        base_model=None,
        medusa_num_heads=None,
        use_mlp=True,
        mlp_hidden_layers=[128, 64],
        **kwargs,
    ):
        """
        Args:
            medusa_head_name_or_path (str): Name or path of the Medusa head to load.
            base_model (str, optional): Override base model path.
            medusa_num_heads (int, optional): Override number of heads.
            use_mlp (bool, optional): Use MLP architecture instead of ResBlocks.
            mlp_hidden_layers (list, optional): Hidden layer sizes for MLP.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            MedusaModel: A MedusaModel instance loaded from the given path.
        """
        medusa_config = MedusaConfig.from_pretrained(medusa_head_name_or_path)
        if medusa_num_heads is not None:
            print("Overriding medusa_num_heads as:", medusa_num_heads)
            medusa_config.medusa_num_heads = medusa_num_heads
        if base_model is not None:
            print("Overriding base_model as:", base_model)
            medusa_config.base_model_name_or_path = base_model
        
        # Load use_mlp and mlp_hidden_layers from config if not overridden
        if use_mlp is False and hasattr(medusa_config, 'use_mlp'):
            use_mlp = medusa_config.use_mlp
        if mlp_hidden_layers == [128, 64] and hasattr(medusa_config, 'mlp_hidden_layers'):
            mlp_hidden_layers = medusa_config.mlp_hidden_layers
            
        base_model = KVLlamaForCausalLM.from_pretrained(
            medusa_config.base_model_name_or_path, **kwargs
        )

        model = cls(
            base_model,
            medusa_config.medusa_num_heads,
            medusa_config.medusa_num_layers,
            medusa_config.base_model_name_or_path,
            use_mlp=use_mlp,
            mlp_hidden_layers=mlp_hidden_layers,
        )
        medusa_head_path = os.path.join(medusa_head_name_or_path, "medusa_lm_head.pt")
        if os.path.exists(medusa_head_path):
            filename = medusa_head_path
        else:
            filename = hf_hub_download(medusa_head_name_or_path, "medusa_lm_head.pt")
        medusa_head_state_dict = torch.load(filename, map_location=base_model.device)
        model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
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
        with torch.no_grad():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        
        # Check if using MLP or ModuleList (ResBlock)
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

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
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
                logits, candidates, temperature, posterior_threshold, posterior_alpha
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

