#!/usr/bin/env python3
"""
Create a small subset of ShareGPT data for testing.
Usage: python create_small_dataset.py --input sharegpt.json --output small_sharegpt.json --samples 20
"""
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create a small subset of ShareGPT data")
    parser.add_argument("--input", type=str, required=True, help="Input ShareGPT JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to extract")

    args = parser.parse_args()

    # Load the full dataset
    print(f"Loading data from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)

    # Take first N samples
    subset = data[:args.samples]

    # Save the subset
    print(f"Saving {len(subset)} samples to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Done! Created subset with {len(subset)} conversations.")

if __name__ == "__main__":
    main()
