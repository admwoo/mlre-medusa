# Training script for Hybrid Model (Medusa + Hydra heads)
# Adapted from Medusa train_legacy.py and Hydra train.py

from dataclasses import dataclass, field
import json
import math
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from safetensors.torch import save_file

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.nn import functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medusa.model.hybrid_model import HybridModel, HybridConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class HybridTrainer(Trainer):
    """
    Custom trainer for Hybrid model combining Medusa and Hydra heads.

    Computes losses for:
    - Medusa heads (standard LM loss)
    - Hydra heads (LM loss + teacher distillation + reconstruction loss)
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute training loss for the hybrid model.

        Args:
            model: The hybrid model
            inputs: Dict with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs

        Returns:
            loss or (loss, outputs) tuple
        """
        # Handle DDP
        if hasattr(model, "module"):
            head_sequence = model.module.head_sequence
        else:
            head_sequence = model.head_sequence

        # Count head types
        num_medusa = head_sequence.count("medusa")
        num_hydra = head_sequence.count("hydra")

        # Forward pass through hybrid model
        all_logits, all_hidden_states, outputs, orig_logits, base_hidden_states = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            hybrid_forward=True,
            output_orig=True,
            noise_alpha=self.args.noise_alpha if model.training else 0.0,
        )

        labels = inputs["labels"]

        # Convert to float32 for loss computation
        all_logits = all_logits.to(torch.float32)
        all_hidden_states = all_hidden_states.to(torch.float32)
        orig_logits = orig_logits.to(torch.float32)
        base_hidden_states = base_hidden_states.to(torch.float32)

        # Get teacher predictions for Hydra head distillation
        teacher_probs = F.softmax(orig_logits, dim=-1)
        teacher_labels = teacher_probs.argmax(dim=-1)

        # Loss functions
        lm_loss_fct = CrossEntropyLoss()
        teacher_loss_fct = CrossEntropyLoss()
        reconstruct_loss_fct = SmoothL1Loss()

        total_loss = 0
        log = {}

        # Compute losses for each head
        for i, head_type in enumerate(head_sequence):
            shift = 2 + i  # Head i predicts token at position t+i+1

            if head_type == "medusa":
                # MEDUSA HEAD LOSS (standard LM loss only)
                medusa_logits = all_logits[i, :, :-shift].contiguous()
                medusa_labels = labels[..., shift:].contiguous()

                medusa_logits_flat = medusa_logits.view(-1, all_logits.shape[-1])
                medusa_labels_flat = medusa_labels.view(-1).to(medusa_logits_flat.device)

                loss_i = lm_loss_fct(medusa_logits_flat, medusa_labels_flat)
                total_loss += self.args.medusa_loss_weight * loss_i

                # Compute accuracy
                not_ignore = medusa_labels_flat.ne(IGNORE_TOKEN_ID)
                medusa_labels_filtered = medusa_labels_flat[not_ignore]

                for k in [1, 5]:
                    if k <= medusa_logits_flat.shape[-1]:
                        _, topk = medusa_logits_flat.topk(k, dim=-1)
                        topk = topk[not_ignore]
                        correct = topk.eq(medusa_labels_filtered.unsqueeze(-1)).any(-1)
                        log[f"medusa{i}_top{k}"] = correct.float().mean().item()

                log[f"medusa{i}_loss"] = loss_i.item()

            elif head_type == "hydra":
                # HYDRA HEAD LOSS (LM + teacher + reconstruction)

                # LM loss terms
                hydra_logits = all_logits[i, :, :-shift].contiguous()
                teacher_logits_shifted = teacher_probs[:, shift-1:-1].contiguous()
                teacher_labels_shifted = teacher_labels[:, shift-1:-1].contiguous()
                hydra_labels = labels[..., shift:].contiguous()

                hydra_logits_flat = hydra_logits.view(-1, all_logits.shape[-1])
                teacher_logits_flat = teacher_logits_shifted.view(-1, teacher_logits_shifted.shape[-1])
                teacher_labels_flat = teacher_labels_shifted.view(-1)
                hydra_labels_flat = hydra_labels.view(-1).to(hydra_logits_flat.device)

                not_ignore_lm = hydra_labels_flat.ne(IGNORE_TOKEN_ID)

                # Reconstruction loss terms
                reconstruct_labels = labels[..., shift-1:].contiguous().view(-1)
                not_ignore_reconstruct = reconstruct_labels.ne(IGNORE_TOKEN_ID)

                # Hidden states for reconstruction
                if shift - 1 == 0:
                    hydra_pred_hidden = all_hidden_states[i].contiguous()
                else:
                    hydra_pred_hidden = all_hidden_states[i, :, :-(shift-1)].contiguous()
                hydra_label_hidden = base_hidden_states[:, shift-1:].contiguous()
                hydra_pred_hidden_flat = hydra_pred_hidden.view(-1, hydra_pred_hidden.shape[-1])
                hydra_label_hidden_flat = hydra_label_hidden.view(-1, hydra_label_hidden.shape[-1])

                # Compute losses
                cur_lm_loss = lm_loss_fct(hydra_logits_flat, hydra_labels_flat)
                cur_teacher_loss = teacher_loss_fct(
                    hydra_logits_flat[not_ignore_lm],
                    teacher_logits_flat[not_ignore_lm]
                )
                cur_reconstruct_loss = reconstruct_loss_fct(
                    hydra_pred_hidden_flat[not_ignore_reconstruct],
                    hydra_label_hidden_flat[not_ignore_reconstruct]
                )

                # Combined Hydra loss
                cur_hydra_loss = (
                    self.args.lm_loss_weight * cur_lm_loss +
                    self.args.teacher_loss_weight * cur_teacher_loss +
                    self.args.reconstruction_loss_weight * cur_reconstruct_loss
                )
                total_loss += self.args.hydra_loss_weight * cur_hydra_loss

                # Compute accuracy
                hydra_labels_filtered = hydra_labels_flat[not_ignore_lm]
                teacher_labels_filtered = teacher_labels_flat[not_ignore_lm]

                for k in [1, 5]:
                    if k <= hydra_logits_flat.shape[-1]:
                        _, topk = hydra_logits_flat.topk(k, dim=-1)
                        topk = topk[not_ignore_lm]
                        correct = topk.eq(hydra_labels_filtered.unsqueeze(-1)).any(-1)
                        teacher_correct = topk.eq(teacher_labels_filtered.unsqueeze(-1)).any(-1)
                        log[f"hydra{i}_top{k}"] = correct.float().mean().item()
                        log[f"hydra{i}_teacher_top{k}"] = teacher_correct.float().mean().item()

                log[f"hydra{i}_lm_loss"] = cur_lm_loss.item()
                log[f"hydra{i}_teacher_loss"] = cur_teacher_loss.item()
                log[f"hydra{i}_reconstruct_loss"] = cur_reconstruct_loss.item()
                log[f"hydra{i}_loss"] = cur_hydra_loss.item()

        log["total_loss"] = total_loss.item()
        log["num_medusa_heads"] = num_medusa
        log["num_hydra_heads"] = num_hydra
        self.log(log)

        return (total_loss, all_logits) if return_outputs else total_loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = field(default="none")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )

    # Hybrid model configuration
    head_sequence: str = field(
        default="medusa,medusa,hydra,hydra",
        metadata={"help": "Comma-separated sequence of heads, e.g., 'medusa,medusa,hydra,hydra'"},
    )

    # Medusa configuration
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers per Medusa head."},
    )
    medusa_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for Medusa head losses."},
    )

    # Hydra configuration
    hydra_num_layers: int = field(
        default=4,
        metadata={"help": "Number of layers per Hydra head."},
    )
    hydra_head_arch: str = field(
        default="mlp",
        metadata={"help": "Hydra head architecture (only 'mlp' supported)."},
    )
    hydra_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for Hydra head losses."},
    )

    # Hydra loss weights (for internal computation)
    lm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for language modeling loss."},
    )
    teacher_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for teacher distillation loss."},
    )
    reconstruction_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for hidden state reconstruction loss."},
    )

    # Training configuration
    noise_alpha: float = field(
        default=0.0,
        metadata={"help": "Noise alpha for NEFTune-style training."},
    )
    hidden_state_offset: int = field(
        default=0,
        metadata={"help": "Offset for hidden states."},
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save the hybrid model's head state dictionary."""
    if trainer.args.should_save:
        # Save only the hybrid heads (not the base model)
        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        HybridModel.save_hybrid_heads(model, output_dir)
        rank0_print(f"Saved hybrid heads to {output_dir}")


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess conversation data and tokenize it.

    Args:
        sources: List of conversation sources (ShareGPT format with "conversations" key)
        tokenizer: Tokenizer for processing

    Returns:
        Dict with input_ids, labels, attention_mask
    """
    # Get conversation template for Vicuna
    conv_template = get_conversation_template("vicuna")

    conversations = []
    prompts = []

    for i, source in enumerate(sources):
        # Handle ShareGPT format: source has "conversations" key with list of turns
        if "conversations" in source:
            conversation_turns = source["conversations"]
        else:
            conversation_turns = source

        # Build conversation using fastchat template
        conv = conv_template.copy()

        for turn in conversation_turns:
            # Convert ShareGPT format ("from": "human"/"gpt", "value": text)
            # to standard format
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            if role in ["human", "user"]:
                conv.append_message(conv.roles[0], content)
            elif role in ["gpt", "assistant"]:
                conv.append_message(conv.roles[1], content)

        # Get the formatted prompt
        prompt = conv.get_prompt()
        prompts.append(prompt)
        conversations.append(conversation_turns)

    # Tokenize
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    # Set everything to be ignored except assistant parts
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets - only compute loss on assistant outputs
    for conv_index, (conversation_turns, target, prompt) in enumerate(zip(conversations, targets, prompts)):
        for turn in conversation_turns:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            if role in ["gpt", "assistant"]:
                # Find where this assistant response appears in the prompt
                try:
                    start = prompt.index(content.strip())
                    stop = start + len(content)
                    indices = []
                    for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                        if tok_stop > start and tok_start < stop:
                            indices.append(tok_index)
                    if indices:
                        target[indices] = encoding.input_ids[conv_index][indices]
                except ValueError:
                    # Content not found in prompt, skip
                    pass

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning (loads on-the-fly)."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Parse head sequence
    head_sequence = [s.strip() for s in training_args.head_sequence.split(",")]
    rank0_print(f"Head sequence: {head_sequence}")

    # Set RoPE scaling factor if needed
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Test tokenizer
    rank0_print("Testing tokenizer...")
    rank0_print(tokenizer(["This is a test", "secondary"], padding=True))

    # Configure quantization if needed
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    # Load hybrid model
    rank0_print("Loading hybrid model...")
    model = HybridModel.from_pretrained(
        model_args.model_name_or_path,
        head_sequence=head_sequence,
        medusa_num_layers=training_args.medusa_num_layers,
        hydra_num_layers=training_args.hydra_num_layers,
        hydra_head_arch=training_args.hydra_head_arch,
        hidden_state_offset=training_args.hidden_state_offset,
        dropout_rate=training_args.dropout_rate,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )

    # Freeze base model parameters - only train the heads
    for name, param in model.named_parameters():
        if "heads" not in name:
            param.requires_grad = False
        else:
            rank0_print(f"Training parameter: {name}")

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Create trainer
    trainer = HybridTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Train
    rank0_print("Starting training...")
    trainer.train()
    trainer.save_state()

    # Save model
    rank0_print("Saving model...")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
