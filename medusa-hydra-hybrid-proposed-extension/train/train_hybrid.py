# This code is based on tatsu-lab/stanford_alpaca and the Medusa training script.
# Adapted for Hybrid Medusa+Hydra training.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from safetensors.torch import save_file

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
from medusa.model.medusa_model import MedusaModel, MedusaConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# Customized for training Hybrid Medusa+Hydra heads
class HybridCustomizedTrainer(Trainer):

    def __init__(self, *args, log_file="hybrid_logs.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
        self.all_logs = []

        # remove old log file if it exists
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            print(f"Deleted old log file: {self.log_file}")

    def _append_log(self, log: dict):
        self.all_logs.append(log)
        with open(self.log_file, "w") as f:
            json.dump(self.all_logs, f, indent=2)
        print(f"Appended log. Total logs: {len(self.all_logs)}")


    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
        """
        Compute the training loss for the hybrid model.
        
        Handles both tree heads (Medusa style) and sequential heads (Hydra style).

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            medusa_model = model.module
        else:
            medusa_model = model
        
        # Get configuration from model
        tree_depth = getattr(medusa_model, 'tree_depth', None)
        sequential_num_heads = getattr(medusa_model, 'sequential_num_heads', 0)
        use_hybrid = getattr(medusa_model, 'use_hybrid', False)
        medusa_num_heads = getattr(medusa_model, 'medusa', None)

        # Get total number of heads
        if use_hybrid and tree_depth is not None:
            tree_num_heads = tree_depth
            total_heads = tree_num_heads + sequential_num_heads
        else:
            # Standard Medusa mode
            if medusa_num_heads is not None:
                tree_num_heads = medusa_num_heads
            else:
                # Fallback: count the heads
                tree_num_heads = len(medusa_model.medusa_head) if hasattr(medusa_model, 'medusa_head') else 0
            total_heads = tree_num_heads
            sequential_num_heads = 0

        # Forward pass with training flags
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            medusa_forward=True,
            run_sequential_heads=use_hybrid and sequential_num_heads > 0,
            output_orig=False,
            noise_alpha=getattr(self.args, 'noise_alpha', 0.0),
        )
        
        labels = inputs["labels"]
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        
        # Compute loss for tree heads (Medusa style)
        # Tree head i predicts token at position 2 + i
        for i in range(tree_num_heads):
            # Shift logits and labels: head i predicts token at position 2+i
            medusa_logits = logits[i, :, : -(2 + i)].contiguous()
            medusa_labels = labels[..., 2 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i
            
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels_clean = medusa_labels[not_ignore]
            
            # Add top-k accuracy
            for k in range(1, 2):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels_clean.unsqueeze(-1)).any(-1)
                log[f"tree_head_{i}_top{k}"] = correct.float().mean().item()
            
            log[f"tree_head_{i}_loss"] = loss_i.item()
        
        # Compute loss for sequential heads (Hydra style)
        # Sequential head i (indexed from 0) predicts token at position: 2 + tree_depth + i
        # This follows the same pattern as tree heads (2 + i) but continues from tree_depth
        if use_hybrid and sequential_num_heads > 0:
            for i in range(sequential_num_heads):
                head_idx = tree_num_heads + i  # Index in logits tensor
                # Sequential head i predicts token at position: 2 + tree_depth + i
                # This matches Hydra's pattern: shift = 2 + i, where i continues from tree_depth
                shift_offset = 2 + tree_depth + i
                if logits.shape[2] > shift_offset:
                    seq_logits = logits[head_idx, :, : -shift_offset].contiguous()
                    seq_labels = labels[..., shift_offset:].contiguous()
                    seq_logits = seq_logits.view(-1, logits.shape[-1])
                    seq_labels = seq_labels.view(-1)
                    seq_labels = seq_labels.to(seq_logits.device)
                    loss_i = loss_fct(seq_logits, seq_labels)
                    loss += loss_i
                    
                    not_ignore = seq_labels.ne(IGNORE_TOKEN_ID)
                    seq_labels_clean = seq_labels[not_ignore]
                    
                    # Add top-k accuracy
                    for k in range(1, 2):
                        _, topk = seq_logits.topk(k, dim=-1)
                        topk = topk[not_ignore]
                        correct = topk.eq(seq_labels_clean.unsqueeze(-1)).any(-1)
                        log[f"sequential_head_{i}_top{k}"] = correct.float().mean().item()
                    
                    log[f"sequential_head_{i}_loss"] = loss_i.item()
                else:
                    # Not enough sequence length for this head
                    log[f"sequential_head_{i}_loss"] = 0.0
                    log[f"sequential_head_{i}_top1"] = 0.0

        self.log(log)
        self._append_log(log)
        return (loss, logits) if return_outputs else loss




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
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=5,
        metadata={"help": "Total number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )
    tree_depth: int = field(
        default=2,
        metadata={"help": "Number of tree heads (Medusa style)."},
    )
    sequential_num_heads: int = field(
        default=3,
        metadata={"help": "Number of sequential heads (Hydra style)."},
    )
    use_hybrid: bool = field(
        default=True,
        metadata={"help": "Use hybrid Medusa+Hydra decoding."},
    )
    noise_alpha: float = field(
        default=0.0,
        metadata={"help": "Noise alpha for NEFT-tune regularization."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save the model's state dictionary to a specified directory."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocesses conversation data and tokenizes it for model input."""
    # Apply prompt templates
    conversations = []
    prompts = []
    for i, conversation in enumerate(sources):
        # Accept either ShareGPT schema or HF-style schema
        if isinstance(conversation, dict) and "conversations" in conversation:
            conv_obj = conversation["conversations"]
        else:
            conv_obj = conversation

        msgs = []
        for m in conv_obj:
            if isinstance(m, dict) and "role" in m and "content" in m:
                msgs.append(m)
            elif isinstance(m, dict) and "from" in m and "value" in m:
                frm = m.get("from")
                if frm == "human":
                    role = "user"
                elif frm == "gpt":
                    role = "assistant"
                else:
                    continue
                msgs.append({"role": role, "content": m.get("value", "")})
            else:
                continue

        # Prefer HF chat template; fallback to FastChat template if missing
        try:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            conv = get_conversation_template(getattr(tokenizer, "name_or_path", ""))
            try:
                conv.messages = []
            except Exception:
                pass
            for m in msgs:
                if m.get("role") == "user":
                    conv.append_message(conv.roles[0], m.get("content", ""))
                elif m.get("role") == "assistant":
                    conv.append_message(conv.roles[1], m.get("content", ""))
            prompt = conv.get_prompt()
        prompts.append(prompt)
        conversations.append(msgs)

    # Tokenize conversations
    # Try to use offset mapping if available (fast tokenizer), otherwise use simpler approach
    try:
        encoding = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        has_offset_mapping = True
    except (NotImplementedError, ValueError):
        # Fallback for slow tokenizers
        encoding = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        has_offset_mapping = False
    
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    if has_offset_mapping:
        # Use offset mapping for precise masking
        for conv_index, (conversation, target, prompt) in enumerate(zip(conversations, targets, prompts)):
            for turn in conversation:
                if turn["role"] == "assistant":
                    content = turn["content"]
                    start = prompt.index(content.strip())
                    stop = start + len(content)
                    indices = []
                    for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                        if tok_stop >= start or tok_start < tok_stop:
                            indices.append(tok_index)
                    target[indices] = encoding.input_ids[conv_index][indices]
    else:
        # Fallback: mark all tokens after the last user message as non-ignore
        # This is less precise but works with slow tokenizers
        for conv_index, (conversation, target) in enumerate(zip(conversations, targets)):
            # Find the last user message position
            last_user_idx = -1
            for i, turn in enumerate(conversation):
                if turn["role"] == "user":
                    last_user_idx = i
            
            # Mark all assistant tokens after the last user message
            if last_user_idx >= 0:
                # Simple heuristic: mark tokens after a certain point
                # This is approximate but should work for most cases
                # We'll mark everything after the first half as assistant tokens
                seq_len = target.shape[0]
                # Find where assistant content likely starts (after prompt template)
                # For simplicity, mark everything after 30% of sequence as assistant
                assistant_start = max(seq_len // 3, 10)
                target[assistant_start:] = encoding.input_ids[conv_index][assistant_start:]

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
    """Lazy dataset for supervised fine-tuning."""
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

    # Validate hybrid configuration
    if training_args.use_hybrid:
        assert training_args.tree_depth + training_args.sequential_num_heads == training_args.medusa_num_heads, \
            f"tree_depth ({training_args.tree_depth}) + sequential_num_heads ({training_args.sequential_num_heads}) must equal medusa_num_heads ({training_args.medusa_num_heads})"

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,  # Use slow tokenizer to avoid tiktoken dependency issues
    )
    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token

    # Create hybrid Medusa config
    # First, update the base config with Medusa parameters
    config.medusa_num_heads = training_args.medusa_num_heads
    config.medusa_num_layers = training_args.medusa_num_layers
    config.tree_depth = training_args.tree_depth if training_args.use_hybrid else None
    config.sequential_num_heads = training_args.sequential_num_heads if training_args.use_hybrid else 0
    config.use_hybrid = training_args.use_hybrid
    config.base_model_name_or_path = model_args.model_name_or_path
    config._name_or_path = model_args.model_name_or_path

    # Create MedusaConfig for saving
    # Get base config dict and update with Medusa-specific params
    config_dict = config.to_dict()
    config_dict.update({
        'medusa_num_heads': training_args.medusa_num_heads,
        'medusa_num_layers': training_args.medusa_num_layers,
        'tree_depth': training_args.tree_depth if training_args.use_hybrid else None,
        'sequential_num_heads': training_args.sequential_num_heads if training_args.use_hybrid else 0,
        'use_hybrid': training_args.use_hybrid,
        'base_model_name_or_path': model_args.model_name_or_path,
    })
    medusa_config = MedusaConfig(**config_dict)

    # Load Medusa model (which includes base model + heads)
    # Pass config via kwargs to avoid duplicate argument issue
    medusa_lm_head = MedusaModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    # Freeze the base model (only train the heads)
    # MedusaModelLlama/Mistral inherit from the base model, so self IS the base model
    # Freeze the transformer layers (model attribute)
    if hasattr(medusa_lm_head, 'model'):
        for param in medusa_lm_head.model.parameters():
            param.requires_grad = False
    # Also freeze the base LM head
    if hasattr(medusa_lm_head, 'lm_head'):
        for param in medusa_lm_head.lm_head.parameters():
            param.requires_grad = False
    # Freeze all parameters except medusa_head and sequential_head
    for name, param in medusa_lm_head.named_parameters():
        if 'medusa_head' not in name and 'sequential_head' not in name:
            param.requires_grad = False

    # Format output dir
    mode_str = "hybrid" if training_args.use_hybrid else "medusa"
    training_args.output_dir = f"{training_args.output_dir}_{mode_str}_{model_args.model_name_or_path.split('/')[-1]}_heads_{training_args.medusa_num_heads}_tree_{training_args.tree_depth}_seq_{training_args.sequential_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.medusa_num_layers}"

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Save Medusa config
    medusa_config.save_pretrained(training_args.output_dir)

    # Start trainer
    trainer = HybridCustomizedTrainer(
        model=medusa_lm_head, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    medusa_lm_head.config.use_cache = True
    
    # Save MedusaHead separately
    if hasattr(medusa_lm_head, "module"):
        lm_head = medusa_lm_head.module.medusa_head
        seq_head = getattr(medusa_lm_head.module, 'sequential_head', None)
    else:
        lm_head = medusa_lm_head.medusa_head
        seq_head = getattr(medusa_lm_head, 'sequential_head', None)
    
    import deepspeed
    with deepspeed.zero.GatheredParameters(lm_head.parameters()):
        state_dict = lm_head.state_dict()
        if seq_head is not None:
            seq_state_dict = seq_head.state_dict()
            # Merge state dicts
            for key, value in seq_state_dict.items():
                state_dict[f"sequential_head.{key}"] = value

    # Save Medusa heads
    if local_rank == 0:
        tokenizer.encode("Test", truncation=False, padding=False)
        tokenizer.save_pretrained(training_args.output_dir)
        save_file(
            state_dict,
            os.path.join(training_args.output_dir, "medusa_lm_head.safetensors"),
        )


if __name__ == "__main__":
    train()

