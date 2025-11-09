import json
import argparse
from pathlib import Path


def process_single_entry(data, entry_num=None):
    """
    Process a single JSON entry and return statistics.
    
    Args:
        data: Dictionary containing model answer data
        entry_num: Optional entry number for display
        
    Returns:
        Dictionary with statistics for this entry
    """
    # Initialize accumulators for this entry
    entry_tokens = 0
    entry_time = 0.0
    entry_turns = 0
    
    if entry_num is not None:
        print(f"\n{'=' * 80}")
        print(f"Entry #{entry_num}")
    
    print(f"Model ID: {data.get('model_id', 'N/A')}")
    print(f"Question ID: {data.get('question_id', 'N/A')}")
    print("=" * 80)
    
    # Process each choice (usually just one)
    for choice in data.get('choices', []):
        choice_index = choice.get('index', 0)
        num_tokens_list = choice.get('num_tokens', [])
        wall_time_list = choice.get('wall_time', [])
        
        # Process each turn
        for turn_idx, (tokens, time) in enumerate(zip(num_tokens_list, wall_time_list)):
            if time > 0:  # Avoid division by zero
                tokens_per_sec = tokens / time
            else:
                tokens_per_sec = 0.0
            
            print(f"Turn {turn_idx + 1}:")
            print(f"  Tokens: {tokens}")
            print(f"  Wall Time: {time:.4f} seconds")
            print(f"  Tokens/Second: {tokens_per_sec:.2f}")
            print("-" * 80)
            
            # Accumulate totals for this entry
            entry_tokens += tokens
            entry_time += time
            entry_turns += 1
    
    return {
        'tokens': entry_tokens,
        'time': entry_time,
        'turns': entry_turns
    }


def calculate_tokens_per_second(input_file):
    """
    Calculate tokens per second from model answer JSON or JSONL file.
    
    Args:
        input_file: Path to the JSON or JSONL file containing model answers
    """
    # Initialize accumulators
    total_tokens = 0
    total_time = 0.0
    question_count = 0
    turn_count = 0
    
    # Try to read as JSONL (one JSON object per line)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if it's JSONL or single JSON
    is_jsonl = len(lines) > 1 or (len(lines) == 1 and not lines[0].strip().startswith('['))
    
    if is_jsonl:
        # Process as JSONL
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                stats = process_single_entry(data, entry_num=line_num)
                total_tokens += stats['tokens']
                total_time += stats['time']
                turn_count += stats['turns']
                question_count += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    else:
        # Process as single JSON object
        try:
            data = json.loads(lines[0])
            stats = process_single_entry(data)
            total_tokens += stats['tokens']
            total_time += stats['time']
            turn_count += stats['turns']
            question_count += 1
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {
                'total_tokens': 0,
                'total_time': 0.0,
                'question_count': 0,
                'turn_count': 0,
                'avg_tokens_per_sec': 0
            }
    
    # Calculate and display aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS:")
    print(f"Total Questions: {question_count}")
    print(f"Total Turns: {turn_count}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.4f} seconds")
    
    if total_time > 0:
        avg_tokens_per_sec = total_tokens / total_time
        print(f"Average Tokens/Second: {avg_tokens_per_sec:.2f}")
    else:
        print("Average Tokens/Second: N/A (zero time)")
    
    print("=" * 80)
    
    return {
        'total_tokens': total_tokens,
        'total_time': total_time,
        'question_count': question_count,
        'turn_count': turn_count,
        'avg_tokens_per_sec': total_tokens / total_time if total_time > 0 else 0
    }


def process_multiple_files(input_files):
    """
    Process multiple JSON files and calculate aggregate statistics.
    
    Args:
        input_files: List of paths to JSON files
    """
    all_total_tokens = 0
    all_total_time = 0.0
    all_question_count = 0
    all_turn_count = 0
    
    for input_file in input_files:
        print(f"\n{'#' * 80}")
        print(f"Processing: {input_file}")
        print(f"{'#' * 80}\n")
        
        try:
            stats = calculate_tokens_per_second(input_file)
            all_total_tokens += stats['total_tokens']
            all_total_time += stats['total_time']
            all_question_count += stats['question_count']
            all_turn_count += stats['turn_count']
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    # Display overall aggregate statistics
    if len(input_files) > 1:
        print(f"\n{'#' * 80}")
        print("OVERALL AGGREGATE STATISTICS (ALL FILES):")
        print(f"{'#' * 80}")
        print(f"Total Questions: {all_question_count}")
        print(f"Total Turns: {all_turn_count}")
        print(f"Total Tokens: {all_total_tokens}")
        print(f"Total Time: {all_total_time:.4f} seconds")
        
        if all_total_time > 0:
            overall_avg_tokens_per_sec = all_total_tokens / all_total_time
            print(f"Overall Average Tokens/Second: {overall_avg_tokens_per_sec:.2f}")
        else:
            print("Overall Average Tokens/Second: N/A (zero time)")
        
        print(f"{'#' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate tokens per second from model answer JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_tokens_per_second.py model_answers.json
  python calculate_tokens_per_second.py file1.json file2.json file3.json
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        type=Path,
        help='One or more JSON files containing model answers'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    for input_file in args.input_files:
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            parser.exit(1)
    
    # Process files
    if len(args.input_files) == 1:
        calculate_tokens_per_second(args.input_files[0])
    else:
        process_multiple_files(args.input_files)
