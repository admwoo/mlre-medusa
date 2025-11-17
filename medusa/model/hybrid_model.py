"""
Hybrid Model: Flexible sequential combination of Medusa and Hydra heads

Key principles:
1. Medusa heads always take input from base hidden states (independent)
2. Hydra heads take input from the previous head's output (sequential/dependent)
3. Head sequence is fully configurable via head_sequence parameter

Example configurations:
- ["medusa", "medusa", "hydra", "hydra"]: 2 Medusa heads, then 2 Hydra heads
- ["hydra", "medusa", "hydra"]: Hydra, then Medusa (takes base), then Hydra (takes Medusa output)
- ["medusa", "hydra", "medusa", "hydra"]: Alternating pattern
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, AutoTokenizer, AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
import os
import warnings


class HybridConfig(PretrainedConfig):
    """
    Configuration class for Hybrid model with flexible head sequencing.

    Args:
        head_sequence (list): List specifying the sequence of heads, e.g., ["medusa", "hydra", "medusa"]
        medusa_num_layers (int): Number of layers per Medusa head. Default is 1.
        hydra_num_layers (int): Number of layers per Hydra head. Default is 1.
        hydra_head_arch (str): Architecture for Hydra heads (only "mlp" supported). Default is "mlp".
        hidden_state_offset (int): Offset for hidden states. Default is 0.
        dropout_rate (float): Dropout rate. Default is 0.0.
        base_model_name_or_path (str): The name or path of the base model.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        head_sequence=["medusa", "medusa", "hydra", "hydra"],
        medusa_num_layers=1,
        hydra_num_layers=1,
        hydra_head_arch="mlp",
        hidden_state_offset=0,
        dropout_rate=0.0,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_sequence = head_sequence
        self.medusa_num_layers = medusa_num_layers
        self.hydra_num_layers = hydra_num_layers
        self.hydra_head_arch = hydra_head_arch
        self.hidden_state_offset = hidden_state_offset
        self.dropout_rate = dropout_rate
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """Residual Block for Medusa heads."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class HydraResBlock(nn.Module):
    """
    Residual Block for Hydra heads with support for conditioning on multiple inputs.

    When num_condition > 0, the input is concatenated from base hidden states and
    num_condition candidate token embeddings, which needs to be projected down.
    """

    def __init__(self, hidden_size, num_condition=0):
        super().__init__()
        self.linear = nn.Linear(hidden_size * (num_condition + 1), hidden_size)

        # Handling residual connection when reducing dim
        if num_condition > 0:
            self.res_connection = nn.Linear(hidden_size * (num_condition + 1), hidden_size, bias=False)
            # Initialize as identity mapping (approximate, since dimensions don't match)
            # We use Xavier initialization but with very small scale to start near identity
            nn.init.xavier_normal_(self.res_connection.weight, gain=0.01)
        else:
            self.res_connection = nn.Identity()

        # Initialize main linear as identity mapping
        torch.nn.init.zeros_(self.linear.weight)

        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the HydraResBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, hidden_size * (num_condition + 1)]

        Returns:
            torch.Tensor: Output after the residual connection and activation [batch, seq_len, hidden_size]
        """
        return self.res_connection(x) + self.act(self.linear(x))


class MedusaHead(nn.Module):
    """
    A single Medusa head that predicts from base hidden states.
    Always takes input from base hidden states (independent).
    """

    def __init__(self, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        layers = []
        for _ in range(num_layers):
            layers.append(ResBlock(hidden_size))
        layers.append(nn.Linear(hidden_size, vocab_size, bias=False))

        self.head = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Base hidden states from the model
        Returns:
            logits: Predictions (batch_size, seq_len, vocab_size)
        """
        return self.head(hidden_states)


class HydraHead(nn.Module):
    """
    A single Hydra head that predicts from base hidden states + all previous candidate embeddings.

    For head i predicting token t+i+1:
    - Receives: h_t (base hidden state) + embeddings of all previous candidates [x_t, ..., x_{t+i}]
    - Predicts: x_{t+i+1}
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        num_layers=1,
        input_embed_fn=None,
        num_prev_candidates=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.input_embed_fn = input_embed_fn
        self.num_prev_candidates = num_prev_candidates  # Number of previous candidate tokens

        assert num_layers > 0, "Hydra head must have at least one layer"
        assert input_embed_fn is not None, "Hydra heads require input_embed_fn"

        layers = []
        # First layer handles concatenated input (base hidden + all candidate embeddings)
        layers.append(HydraResBlock(hidden_size, num_condition=num_prev_candidates))
        # Subsequent layers are standard
        for _ in range(num_layers - 1):
            layers.append(HydraResBlock(hidden_size, num_condition=0))

        self.head = nn.Sequential(*layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, base_hidden_states, candidate_tokens, noise=None):
        """
        Args:
            base_hidden_states: Base hidden states h_t from the model
            candidate_tokens: List of predicted candidate tokens from all previous heads
                             [x_t, x_{t+1}, ..., x_{t+i}] where i is the current head index
            noise: Noise to add (for NEFTune training)
        Returns:
            logits: Predictions (batch_size, seq_len, vocab_size)
            hidden_states: Output hidden states (currently unused but kept for compatibility)
        """
        # Get embeddings for all candidate tokens
        with torch.inference_mode():
            candidate_embeds = []
            for token_ids in candidate_tokens:
                embeds = self.input_embed_fn(token_ids)
                if noise is not None:
                    embeds = embeds + noise
                candidate_embeds.append(embeds)

        # Concatenate base hidden states with all candidate embeddings
        # Input: [batch, seq_len, hidden_size * (1 + num_prev_candidates)]
        concatenated = torch.cat([base_hidden_states] + candidate_embeds, dim=-1)

        # Process through MLP layers
        output_hidden = self.head(concatenated)

        # Generate logits
        logits = self.lm_head(output_hidden)

        return logits, output_hidden


class HybridModelABC(nn.Module):
    """
    Hybrid model with flexible sequential head architecture.

    The head_sequence parameter controls the order and type of heads:
    - "medusa": Independent prediction from base hidden states
    - "hydra": Sequential prediction from previous head's output

    Example: head_sequence=["medusa", "medusa", "hydra", "hydra"]
    - Head 0 (Medusa): base → Medusa₀ (predicts t+1)
    - Head 1 (Medusa): base → Medusa₁ (predicts t+2)
    - Head 2 (Hydra): Medusa₁ → Hydra₀ (predicts t+3, uses Medusa₁'s output)
    - Head 3 (Hydra): Hydra₀ → Hydra₁ (predicts t+4, uses Hydra₀'s output)
    """

    def __init__(self, config):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.head_sequence = config.head_sequence
        self.medusa_num_layers = config.medusa_num_layers
        self.hydra_num_layers = config.hydra_num_layers
        self.hydra_head_arch = config.hydra_head_arch
        self.hidden_state_offset = config.hidden_state_offset
        self.dropout_rate = config.dropout_rate
        self.base_model_name_or_path = config._name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)

        # Validate head sequence
        for head_type in self.head_sequence:
            if head_type not in ["medusa", "hydra"]:
                raise ValueError(f"Invalid head type: {head_type}. Must be 'medusa' or 'hydra'.")

        # Create heads based on sequence
        self.heads = nn.ModuleList()

        for i, head_type in enumerate(self.head_sequence):
            if head_type == "medusa":
                head = MedusaHead(
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    num_layers=self.medusa_num_layers,
                )
            elif head_type == "hydra":
                # Hydra head at position i receives i previous candidate tokens
                # (from heads 0, 1, ..., i-1)
                head = HydraHead(
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    num_layers=self.hydra_num_layers,
                    input_embed_fn=self.base_model.model.embed_tokens,
                    num_prev_candidates=i,  # Number of previous heads (candidates)
                )

            self.heads.append(head)

        print(f"Created hybrid model with head sequence: {self.head_sequence}")
        print(f"Total heads: {len(self.heads)} (Medusa: {self.head_sequence.count('medusa')}, Hydra: {self.head_sequence.count('hydra')})")

    @property
    def base_model(self):
        return self

    def get_tokenizer(self):
        return self.tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        hybrid_forward=False,
        noise_alpha=0.0,
        **kwargs,
    ):
        """
        Forward pass through the hybrid model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (for training mode)
            past_key_values: Past key-value pairs for attention
            output_orig: Whether to output original LM head predictions
            position_ids: Position IDs
            hybrid_forward: Whether to run hybrid heads (if False, runs base model only)
            noise_alpha: Noise coefficient for NEFTune training

        Returns:
            If hybrid_forward=False: Standard base model output
            If hybrid_forward=True: (all_logits, all_hidden_states, outputs, orig_logits, base_hidden_states)
                - all_logits: List of logits from each head
                - all_hidden_states: List of hidden states from each head
                - outputs: Base model outputs
                - orig_logits: Original LM head logits (if output_orig=True)
                - base_hidden_states: Base hidden states
        """
        if not hybrid_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )

        # Get base hidden states
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=self.hidden_state_offset != 0,
                **kwargs,
            )
            if output_orig:
                orig_logits = self.base_model.lm_head(outputs[0])

        # Select hidden states based on offset
        if self.hidden_state_offset == 0:
            base_hidden_states = outputs[0].clone()
        else:
            base_hidden_states = outputs[1][-(self.hidden_state_offset + 1)].clone()

        # Prepare noise for NEFTune training
        model_dim = base_hidden_states.shape[-1]
        seq_len = (input_ids != self.tokenizer.pad_token_id).sum(dim=-1).clamp(min=1).unsqueeze(1).unsqueeze(2)
        denom = torch.sqrt(seq_len * model_dim)
        noise = (torch.rand_like(base_hidden_states) * 2 - 1) * noise_alpha / denom
        noise = noise.to(base_hidden_states.dtype)

        # Determine if we're in training mode
        # During training, use ground truth labels; during inference, use predictions
        is_training = self.training and labels is not None

        # Run heads sequentially
        all_logits = []
        all_hidden_states = []
        candidate_tokens = []  # Store candidate tokens from all previous heads

        for i, (head, head_type) in enumerate(zip(self.heads, self.head_sequence)):
            if head_type == "medusa":
                # Medusa head always uses base hidden states
                logits = head(base_hidden_states)
                output_hidden = base_hidden_states

                # Get candidate tokens for next head
                if is_training:
                    # Training: use ground truth shifted by (i+1) positions
                    # Head i predicts token at position t+i+1, so we use label at t+i+1
                    candidate_token = input_ids[:, 1+i:].contiguous()  # Shift by i+1
                else:
                    # Inference: use predicted tokens
                    with torch.no_grad():
                        candidate_token = logits.argmax(dim=-1)

                candidate_tokens.append(candidate_token)

            elif head_type == "hydra":
                # Hydra head receives:
                # 1. Base hidden states h_t
                # 2. All previous candidate token embeddings [x_t, x_{t+1}, ..., x_{t+i-1}]

                logits, output_hidden = head(
                    base_hidden_states=base_hidden_states,
                    candidate_tokens=candidate_tokens,  # All previous candidates
                    noise=noise if noise_alpha > 0 else None,
                )

                # Get candidate tokens for next head
                if is_training:
                    # Training: use ground truth shifted by (i+1) positions
                    candidate_token = input_ids[:, 1+i:].contiguous()
                else:
                    # Inference: use predicted tokens
                    with torch.no_grad():
                        candidate_token = logits.argmax(dim=-1)

                candidate_tokens.append(candidate_token)

            all_logits.append(logits)
            all_hidden_states.append(output_hidden)

        # Stack outputs
        all_logits = torch.stack(all_logits, dim=0)
        all_hidden_states = torch.stack(all_hidden_states, dim=0)

        if output_orig:
            return all_logits, all_hidden_states, outputs, orig_logits, base_hidden_states
        return all_logits, all_hidden_states, outputs


class HybridModelLlama(HybridModelABC, KVLlamaForCausalLM):
    """Hybrid model for Llama architecture."""
    pass


class HybridModelMistral(HybridModelABC, KVMistralForCausalLM):
    """Hybrid model for Mistral architecture."""
    pass


class HybridModel:
    """Factory class for creating hybrid models."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        head_sequence=None,
        medusa_num_layers=1,
        hydra_num_layers=1,
        hydra_head_arch="mlp",
        hidden_state_offset=0,
        dropout_rate=0.0,
        *args,
        **kwargs,
    ):
        """
        Load or create a hybrid model.

        Args:
            pretrained_model_name_or_path: Path to base model or saved hybrid model
            head_sequence: List specifying head order, e.g., ["medusa", "hydra", "medusa"]
            medusa_num_layers: Number of layers per Medusa head
            hydra_num_layers: Number of layers per Hydra head
            hydra_head_arch: Architecture for Hydra heads (currently only "mlp" supported)
            hidden_state_offset: Hidden state layer offset
            dropout_rate: Dropout rate
        """
        # Try to load existing hybrid config
        try:
            config = HybridConfig.from_pretrained(pretrained_model_name_or_path)
            print(f"Loaded existing hybrid config from {pretrained_model_name_or_path}")

            # Even when loading existing config, we need to ensure base model attributes are present
            base_model_path = config.base_model_name_or_path if hasattr(config, 'base_model_name_or_path') else pretrained_model_name_or_path
            base_config = AutoConfig.from_pretrained(base_model_path)

            # Merge base config attributes that are missing in hybrid config
            for key in vars(base_config):
                if not hasattr(config, key):
                    setattr(config, key, getattr(base_config, key))
        except:
            # Create new hybrid config from base model
            base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

            # Use provided head_sequence or default
            if head_sequence is None:
                head_sequence = ["medusa", "medusa", "hydra", "hydra"]
                print(f"Using default head sequence: {head_sequence}")

            config = HybridConfig(
                head_sequence=head_sequence,
                medusa_num_layers=medusa_num_layers,
                hydra_num_layers=hydra_num_layers,
                hydra_head_arch=hydra_head_arch,
                hidden_state_offset=hidden_state_offset,
                dropout_rate=dropout_rate,
                base_model_name_or_path=pretrained_model_name_or_path,
            )

            # Merge with base config
            for key in vars(base_config):
                if not hasattr(config, key):
                    setattr(config, key, getattr(base_config, key))

        # Determine model type
        model_type = config.model_type if hasattr(config, 'model_type') else AutoConfig.from_pretrained(pretrained_model_name_or_path).model_type

        # Create model based on type
        if model_type == "llama":
            model_class = HybridModelLlama
        elif model_type == "mistral":
            model_class = HybridModelMistral
        else:
            raise ValueError(f"Model type {model_type} not supported!")

        # Load model
        model = model_class.from_pretrained(
            config.base_model_name_or_path if hasattr(config, 'base_model_name_or_path') else pretrained_model_name_or_path,
            config=config,
            *args,
            **kwargs,
        )

        # Try to load saved head weights
        hybrid_heads_path = os.path.join(pretrained_model_name_or_path, "hybrid_heads.pt")
        if os.path.exists(hybrid_heads_path):
            print(f"Loading hybrid heads from {hybrid_heads_path}")
            head_state_dict = torch.load(hybrid_heads_path, map_location=model.device)
            model.heads.load_state_dict(head_state_dict, strict=False)

        return model

    @staticmethod
    def save_hybrid_heads(model, output_dir):
        """
        Save only the hybrid heads (not the base model).

        Args:
            model: The hybrid model
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save heads
        heads_state_dict = model.heads.state_dict()
        torch.save(heads_state_dict, os.path.join(output_dir, "hybrid_heads.pt"))

        # Save config
        config = HybridConfig(
            head_sequence=model.head_sequence,
            medusa_num_layers=model.medusa_num_layers,
            hydra_num_layers=model.hydra_num_layers,
            hydra_head_arch=model.hydra_head_arch,
            hidden_state_offset=model.hidden_state_offset,
            dropout_rate=model.dropout_rate,
            base_model_name_or_path=model.base_model_name_or_path,
        )
        config.save_pretrained(output_dir)

        print(f"Saved hybrid heads and config to {output_dir}")
