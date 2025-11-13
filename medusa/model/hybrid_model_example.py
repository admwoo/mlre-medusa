"""
Example usage of the Hybrid Model

This script demonstrates how to create and use hybrid models with different
configurations of Medusa and Hydra heads.
"""

import torch
from hybrid_model import HybridModel

def example_1_basic_usage():
    """Example 1: Create a basic hybrid model with default configuration"""
    print("=" * 80)
    print("Example 1: Basic Hybrid Model (Default Configuration)")
    print("=" * 80)

    # Default: ["medusa", "medusa", "hydra", "hydra"]
    model = HybridModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\nModel created with default head sequence:")
    print(f"  Head sequence: {model.head_sequence}")
    print(f"  Total heads: {len(model.heads)}")
    print()


def example_2_custom_sequence():
    """Example 2: Create hybrid models with custom head sequences"""
    print("=" * 80)
    print("Example 2: Custom Head Sequences")
    print("=" * 80)

    # Configuration 1: 2 Hydra heads followed by 2 Medusa heads
    print("\nConfiguration 1: ['hydra', 'hydra', 'medusa', 'medusa']")
    model1 = HybridModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        head_sequence=["hydra", "hydra", "medusa", "medusa"],
        hydra_num_layers=4,
        medusa_num_layers=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Created model with {len(model1.heads)} heads")

    # Configuration 2: 1 Medusa head followed by 3 Hydra heads
    print("\nConfiguration 2: ['medusa', 'hydra', 'hydra', 'hydra']")
    model2 = HybridModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        head_sequence=["medusa", "hydra", "hydra", "hydra"],
        hydra_num_layers=4,
        medusa_num_layers=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Created model with {len(model2.heads)} heads")

    # Configuration 3: Alternating Medusa and Hydra
    print("\nConfiguration 3: ['medusa', 'hydra', 'medusa', 'hydra']")
    model3 = HybridModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        head_sequence=["medusa", "hydra", "medusa", "hydra"],
        hydra_num_layers=4,
        medusa_num_layers=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Created model with {len(model3.heads)} heads")
    print()


def example_3_different_layer_counts():
    """Example 3: Use different layer counts for Medusa and Hydra heads"""
    print("=" * 80)
    print("Example 3: Different Layer Counts")
    print("=" * 80)

    # Configure different complexities for Medusa vs Hydra heads
    model = HybridModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        head_sequence=["medusa", "hydra", "hydra"],
        hydra_num_layers=4,  # Deeper Hydra heads
        medusa_num_layers=1,  # Shallower Medusa heads
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"\nCreated model with different layer counts")
    print(f"  Head sequence: {model.head_sequence}")
    print(f"  Medusa layers: {model.medusa_num_layers}")
    print(f"  Hydra layers: {model.hydra_num_layers}")
    print()


def example_4_forward_pass():
    """Example 4: Perform a forward pass"""
    print("=" * 80)
    print("Example 4: Forward Pass")
    print("=" * 80)

    model = HybridModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        head_sequence=["medusa", "medusa", "hydra"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Prepare input
    tokenizer = model.get_tokenizer()
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\nInput text: '{text}'")
    print(f"Input shape: {inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            hybrid_forward=True,
            output_orig=True,
        )

    all_logits, all_hidden_states, base_outputs, orig_logits, base_hidden = outputs

    print(f"\nOutput shapes:")
    print(f"  all_logits: {all_logits.shape}")  # [num_heads, batch, seq_len, vocab_size]
    print(f"  all_hidden_states: {all_hidden_states.shape}")  # [num_heads, batch, seq_len, hidden_size]
    print(f"  orig_logits: {orig_logits.shape}")  # [batch, seq_len, vocab_size]
    print(f"  base_hidden: {base_hidden.shape}")  # [batch, seq_len, hidden_size]

    print(f"\nPredictions from each head:")
    for i, head_type in enumerate(model.head_sequence):
        # Get top prediction for last token
        pred_token_id = all_logits[i, 0, -1].argmax().item()
        pred_token = tokenizer.decode([pred_token_id])
        print(f"  Head {i} ({head_type}): '{pred_token}'")
    print()


def example_5_save_and_load():
    """Example 5: Save and load hybrid model"""
    print("=" * 80)
    print("Example 5: Save and Load Hybrid Model")
    print("=" * 80)

    # Create model
    print("\nCreating model...")
    model = HybridModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        head_sequence=["medusa", "hydra", "hydra"],
        hydra_num_layers=4,
        medusa_num_layers=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Save only the heads (not the full base model)
    save_dir = "./hybrid_model_checkpoint"
    print(f"\nSaving hybrid heads to {save_dir}...")
    HybridModel.save_hybrid_heads(model, save_dir)

    # Load the model back
    print(f"\nLoading hybrid model from {save_dir}...")
    loaded_model = HybridModel.from_pretrained(
        save_dir,  # This will load the config and heads
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Successfully loaded model")
    print(f"  Head sequence: {loaded_model.head_sequence}")
    print()


def example_6_comparison_experiments():
    """Example 6: Setup for comparison experiments"""
    print("=" * 80)
    print("Example 6: Configurations for Comparison Experiments")
    print("=" * 80)

    # Different configurations to compare
    configs = [
        {
            "name": "Pure Medusa (4 heads)",
            "head_sequence": ["medusa", "medusa", "medusa", "medusa"],
            "medusa_num_layers": 1,
            "hydra_num_layers": 4,
        },
        {
            "name": "Pure Hydra (4 heads)",
            "head_sequence": ["hydra", "hydra", "hydra", "hydra"],
            "medusa_num_layers": 1,
            "hydra_num_layers": 4,
        },
        {
            "name": "2 Medusa + 2 Hydra",
            "head_sequence": ["medusa", "medusa", "hydra", "hydra"],
            "medusa_num_layers": 1,
            "hydra_num_layers": 4,
        },
        {
            "name": "2 Hydra + 2 Medusa",
            "head_sequence": ["hydra", "hydra", "medusa", "medusa"],
            "medusa_num_layers": 1,
            "hydra_num_layers": 4,
        },
        {
            "name": "1 Medusa + 3 Hydra",
            "head_sequence": ["medusa", "hydra", "hydra", "hydra"],
            "medusa_num_layers": 1,
            "hydra_num_layers": 4,
        },
        {
            "name": "Alternating (M-H-M-H)",
            "head_sequence": ["medusa", "hydra", "medusa", "hydra"],
            "medusa_num_layers": 1,
            "hydra_num_layers": 4,
        },
    ]

    print("\nExample configurations for experiments:\n")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   Head sequence: {config['head_sequence']}")
        print(f"   Medusa layers: {config['medusa_num_layers']}, Hydra layers: {config['hydra_num_layers']}")

        # Show data flow
        print(f"   Data flow:")
        prev_output = "base"
        for j, head_type in enumerate(config['head_sequence']):
            if head_type == "medusa":
                print(f"     Head {j} (Medusa): {prev_output} → Medusa")
                # Medusa doesn't change the flow for next head
            else:  # hydra
                print(f"     Head {j} (Hydra): {prev_output} → Hydra")
                prev_output = f"Hydra{j}"
        print()


if __name__ == "__main__":
    # Run examples
    # Note: Comment out examples you don't want to run

    print("\n" + "=" * 80)
    print("HYBRID MODEL EXAMPLES")
    print("=" * 80 + "\n")

    # Example 1: Basic usage
    example_1_basic_usage()

    # Example 2: Custom sequences
    example_2_custom_sequence()

    # Example 3: Different layer counts
    example_3_different_layer_counts()

    # Example 4: Forward pass
    # example_4_forward_pass()  # Uncomment to run (requires model weights)

    # Example 5: Save and load
    # example_5_save_and_load()  # Uncomment to run (creates files)

    # Example 6: Comparison experiments
    example_6_comparison_experiments()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80 + "\n")
