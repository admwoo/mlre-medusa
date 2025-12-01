import argparse
import os
import torch
from safetensors.torch import load_file
from pathlib import Path


def convert_safetensors_to_pt(input_path, output_path):
    """
    Convert a safetensors file to PyTorch .pt format.

    Args:
        input_path (str): Path to the input .safetensors file
        output_path (str): Path to the output .pt file
    """
    print(f"Loading safetensors from: {input_path}")
    state_dict = load_file(input_path)

    print(f"Saving to PyTorch format: {output_path}")
    torch.save(state_dict, output_path)

    # Verify file sizes
    input_size = os.path.getsize(input_path) / (1024**2)  # MB
    output_size = os.path.getsize(output_path) / (1024**2)  # MB

    print(f"✓ Conversion successful!")
    print(f"  Input size:  {input_size:.2f} MB")
    print(f"  Output size: {output_size:.2f} MB")



def main():
    parser = argparse.ArgumentParser(
        description="Convert safetensors files to PyTorch .pt format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python convert_safetensors_to_pt.py --input medusa_lm_head.safetensors --output medusa_lm_head.pt

  # Convert in a directory (auto-detects medusa_lm_head.safetensors)
  python convert_safetensors_to_pt.py --input /path/to/checkpoint/

  # Specify both input and output directories
  python convert_safetensors_to_pt.py --input /path/to/checkpoint/ --output /path/to/checkpoint/
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .safetensors file or directory containing medusa_lm_head.safetensors"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output .pt file or directory. If not specified, will use same location as input with .pt extension"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # Handle directory input
    if input_path.is_dir():
        input_file = input_path / "medusa_lm_head.safetensors"
        if not input_file.exists():
            print(f"Error: Could not find medusa_lm_head.safetensors in {input_path}")
            return

        if args.output is None:
            output_file = input_path / "medusa_lm_head.pt"
        else:
            output_path = Path(args.output)
            if output_path.is_dir():
                output_file = output_path / "medusa_lm_head.pt"
            else:
                output_file = output_path
    else:
        # Handle file input
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return

        input_file = input_path

        if args.output is None:
            # Replace .safetensors extension with .pt
            output_file = input_path.with_suffix('.pt')
        else:
            output_file = Path(args.output)

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    # Check if output file already exists
    if output_file.exists():
        response = input(f"Output file {output_file} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            return

    # Perform conversion
    try:
        convert_safetensors_to_pt(str(input_file), str(output_file))
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
