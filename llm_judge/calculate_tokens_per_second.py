import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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
    entry_decoding_steps = 0
    
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
        decoding_steps = choice.get('num_decoding_steps', [])
        
        # Process each turn
        for turn_idx, (tokens, time, decoding_steps) in enumerate(zip(num_tokens_list, wall_time_list, decoding_steps)):
            if time > 0:  # Avoid division by zero
                tokens_per_sec = tokens / time
            else:
                tokens_per_sec = 0.0
            
            print(f"Turn {turn_idx + 1}:")
            print(f"  Tokens: {tokens}")
            print(f"  Wall Time: {time:.4f} seconds")
            print(f"  Tokens/Second: {tokens_per_sec:.2f}")
            print(f"  Decoding Steps: {decoding_steps}")
            print(f"  Acceleration rate: {tokens / decoding_steps if decoding_steps > 0 else 0:.2f} tokens/step")
            print("-" * 80)
            
            # Accumulate totals for this entry
            entry_tokens += tokens
            entry_time += time
            entry_turns += 1
            entry_decoding_steps += decoding_steps
    
    return {
        'tokens': entry_tokens,
        'time': entry_time,
        'turns': entry_turns,
        'decoding_steps': entry_decoding_steps
    }


def calculate_tokens_per_second(input_file, quality=None, quality_per_subject=None):
    """
    Calculate tokens per second from model answer JSON or JSONL file.
    
    Args:
        input_file: Path to the JSON or JSONL file containing model answers
        quality: Optional quality score to include in the aggregate graph
        quality_per_subject: Optional list of 8 quality scores for subjects
    """

    # List of subjects
    subjects = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 
                'extraction', 'stem', 'humanities']

    # Map quality_per_subject if provided
    subject_quality_map = {}
    if quality_per_subject:
        # Input order: coding, extraction, humanities, math, reasoning, roleplay, stem, writing
        input_subjects_order = ['coding', 'extraction', 'humanities', 'math', 'reasoning', 'roleplay', 'stem', 'writing']
        if len(quality_per_subject) == 8:
            for subj, score in zip(input_subjects_order, quality_per_subject):
                subject_quality_map[subj] = score
        else:
            print("Warning: quality_per_subject must have exactly 8 values. Ignoring.")
            quality_per_subject = None

    # Initialize accumulators per subject
    subject_tokens = {subject: 0 for subject in subjects}
    subject_time = {subject: 0.0 for subject in subjects}
    subject_question_count = {subject: 0 for subject in subjects}
    subject_turn_count = {subject: 0 for subject in subjects}
    subject_decoding_steps_count = {subject: 0 for subject in subjects}

    # Initialize accumulators
    total_tokens = 0
    total_time = 0.0
    question_count = 0
    turn_count = 0
    decoding_steps_count = 0
    
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
                current_subject = subjects[question_count // 10]
                subject_tokens[current_subject] += stats['tokens']
                subject_time[current_subject] += stats['time']
                subject_turn_count[current_subject] += stats['turns']
                subject_question_count[current_subject] += 1
                subject_decoding_steps_count[current_subject] += stats.get('decoding_steps', 0)
                total_tokens += stats['tokens']
                total_time += stats['time']
                turn_count += stats['turns']
                decoding_steps_count += stats.get('decoding_steps', 0)
                question_count += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    else:
        # Process as single JSON object
        try:
            current_subject = subjects[question_count // 10]
            data = json.loads(lines[0])
            stats = process_single_entry(data)
            subject_tokens[current_subject] += stats['tokens']
            subject_time[current_subject] += stats['time']
            subject_turn_count[current_subject] += stats['turns']
            subject_question_count[current_subject] += 1
            subject_decoding_steps_count[current_subject] += stats.get('decoding_steps', 0)
            total_tokens += stats['tokens']
            total_time += stats['time']
            turn_count += stats['turns']
            decoding_steps_count += stats.get('decoding_steps', 0)
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
    print(f"Total Decoding Steps: {decoding_steps_count}")
    print(f"Average Per Step Latency: {total_time / decoding_steps_count if decoding_steps_count > 0 else 0:.4f} seconds/step")
    print(f"Overall Acceleration rate: {total_tokens / decoding_steps_count if decoding_steps_count > 0 else 0:.2f} tokens/step")
    
    if total_time > 0:
        avg_tokens_per_sec = total_tokens / total_time
        print(f"Average Tokens/Second: {avg_tokens_per_sec:.2f}")
    else:
        print("Average Tokens/Second: N/A (zero time)")
    print("\nSubject-wise Statistics:")
    for subject in subjects:
        print(f"Subject: {subject}")
        print(f"  Questions: {subject_question_count[subject]}")
        print(f"  Turns: {subject_turn_count[subject]}")
        print(f"  Tokens: {subject_tokens[subject]}")
        print(f"  Time: {subject_time[subject]:.4f} seconds")
        if subject_time[subject] > 0:
            print(f"  Tokens/Second: {subject_tokens[subject] / subject_time[subject]:.2f}")
        else:
            print("  Tokens/Second: N/A (zero time)")
        if subject_decoding_steps_count[subject] > 0:
            print(f"  Acceleration rate: {subject_tokens[subject] / subject_decoding_steps_count[subject]:.2f} tokens/step")
        else:
            print("  Acceleration rate: N/A (zero decoding steps)")
        print(f"  Average Per Step Latency: {subject_time[subject] / subject_decoding_steps_count[subject] if subject_decoding_steps_count[subject] > 0 else 0:.4f} seconds/step")
        print()
    
    print("=" * 80)
    baseline_per_step_latency = 0.0307  # Example baseline latency in seconds/step
    
    ## Create matplotlib graph of the acceleration rate, overhead and speedup for each subject where each subject is on the x-axis
    # Overhead = Average Per Step Latency / baseline_per_step_latency
    # Speedup = acceleration rate / overhead

    # Calculate metrics for each subject
    subject_metrics = {
        'subjects': [],
        'acceleration': [],
        'overhead': [],
        'speedup': [],
        'tokens_per_sec': [],
        'quality': []
    }

    for subject in subjects:
        if subject_decoding_steps_count[subject] > 0:
            acc_rate = subject_tokens[subject] / subject_decoding_steps_count[subject]
            avg_latency = subject_time[subject] / subject_decoding_steps_count[subject]
            overhead = avg_latency / baseline_per_step_latency
            speedup = acc_rate / overhead
            tokens_per_sec = subject_tokens[subject] / subject_time[subject] if subject_time[subject] > 0 else 0
            
            subject_metrics['subjects'].append(subject)
            subject_metrics['acceleration'].append(acc_rate)
            subject_metrics['overhead'].append(overhead)
            subject_metrics['speedup'].append(speedup)
            subject_metrics['tokens_per_sec'].append(tokens_per_sec)
            if quality_per_subject:
                subject_metrics['quality'].append(subject_quality_map.get(subject, 0))

    if subject_metrics['subjects']:
        x = np.arange(len(subject_metrics['subjects']))
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        if quality_per_subject:
            width = 0.15
            rects1 = ax1.bar(x - 2*width, subject_metrics['acceleration'], width, label='Acceleration Rate', color='tab:blue')
            rects2 = ax1.bar(x - 1*width, subject_metrics['overhead'], width, label='Overhead', color='tab:orange')
            rects3 = ax1.bar(x, subject_metrics['speedup'], width, label='Speedup', color='tab:green')
            rects_q = ax1.bar(x + 1*width, subject_metrics['quality'], width, label='Quality', color='tab:purple')
            rects4 = ax2.bar(x + 2*width, subject_metrics['tokens_per_sec'], width, label='Tokens/Sec', color='tab:red')
            
            ax1.bar_label(rects_q, padding=3, fmt='%.2f', fontsize=7)
            all_metrics_1 = subject_metrics['acceleration'] + subject_metrics['overhead'] + subject_metrics['speedup'] + subject_metrics['quality']
        else:
            width = 0.2
            rects1 = ax1.bar(x - 1.5*width, subject_metrics['acceleration'], width, label='Acceleration Rate', color='tab:blue')
            rects2 = ax1.bar(x - 0.5*width, subject_metrics['overhead'], width, label='Overhead', color='tab:orange')
            rects3 = ax1.bar(x + 0.5*width, subject_metrics['speedup'], width, label='Speedup', color='tab:green')
            rects4 = ax2.bar(x + 1.5*width, subject_metrics['tokens_per_sec'], width, label='Tokens/Sec', color='tab:red')
            all_metrics_1 = subject_metrics['acceleration'] + subject_metrics['overhead'] + subject_metrics['speedup']

        # Scale axes so bars are in the middle on average
        if all_metrics_1:
            avg_1 = np.mean(all_metrics_1)
            max_1 = np.max(all_metrics_1)
            ax1.set_ylim(0, float(max(max_1 * 1.1, avg_1 * 2)))

        if subject_metrics['tokens_per_sec']:
            avg_2 = np.mean(subject_metrics['tokens_per_sec'])
            max_2 = np.max(subject_metrics['tokens_per_sec'])
            ax2.set_ylim(0, float(max(max_2 * 1.1, avg_2 * 2)))

        ax1.set_ylabel('Values')
        ax2.set_ylabel('Tokens/Second')
        ax1.set_title('Metrics by Subject')
        ax1.set_xticks(x)
        ax1.set_xticklabels(subject_metrics['subjects'])
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.bar_label(rects1, padding=3, fmt='%.2f', fontsize=7)
        ax1.bar_label(rects2, padding=3, fmt='%.2f', fontsize=7)
        ax1.bar_label(rects3, padding=3, fmt='%.2f', fontsize=7)
        ax2.bar_label(rects4, padding=3, fmt='%.2f', fontsize=7)

        fig.tight_layout()
        output_stem = Path(input_file).stem
        plt.savefig(f'{output_stem}_subject_metrics.png')
        print(f"Saved subject metrics plot to {output_stem}_subject_metrics.png")
        plt.close()

    ## End this graph

    ## Create graph of the models aggregate statistics for acceleration rate, overhead and speedup
    if decoding_steps_count > 0:
        agg_acc_rate = total_tokens / decoding_steps_count
        agg_avg_latency = total_time / decoding_steps_count
        agg_overhead = agg_avg_latency / baseline_per_step_latency
        agg_speedup = agg_acc_rate / agg_overhead
        agg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        # Group 1: Ratios/Scores (Left Axis)
        metrics_g1 = ['Acceleration Rate', 'Overhead', 'Speedup']
        values_g1 = [agg_acc_rate, agg_overhead, agg_speedup]
        colors_g1 = ['tab:blue', 'tab:orange', 'tab:green']

        if quality is not None:
            metrics_g1.append('Quality')
            values_g1.append(quality)
            colors_g1.append('tab:purple')

        # Group 2: Tokens/Sec (Right Axis)
        metrics_g2 = ['Tokens/Sec']
        values_g2 = [agg_tokens_per_sec]
        colors_g2 = ['tab:red']

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Calculate positions
        all_metrics = metrics_g1 + metrics_g2
        x_pos = np.arange(len(all_metrics))
        
        # Plot bars
        bars1 = ax1.bar(x_pos[:len(metrics_g1)], values_g1, color=colors_g1)
        bars2 = ax2.bar(x_pos[len(metrics_g1):], values_g2, color=colors_g2)

        # Scale axes so bars are in the middle on average
        if values_g1:
            avg_1 = np.mean(values_g1)
            max_1 = np.max(values_g1)
            ax1.set_ylim(0, float(max(max_1 * 1.1, avg_1 * 2)))

        if values_g2:
            avg_2 = np.mean(values_g2)
            max_2 = np.max(values_g2)
            ax2.set_ylim(0, float(max(max_2 * 1.1, avg_2 * 2)))

        ax1.set_ylabel('Values')
        ax2.set_ylabel('Tokens/Second')
        ax1.set_title('Aggregate Metrics')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(all_metrics)
        
        ax1.bar_label(bars1, fmt='%.2f')
        ax2.bar_label(bars2, fmt='%.2f')

        output_stem = Path(input_file).stem
        plt.savefig(f'{output_stem}_aggregate_metrics.png')
        print(f"Saved aggregate metrics plot to {output_stem}_aggregate_metrics.png")
        plt.close()

    ## End this graph

    # End graph creation
    
    return {
        'total_tokens': total_tokens,
        'total_time': total_time,
        'question_count': question_count,
        'turn_count': turn_count,
        'avg_tokens_per_sec': total_tokens / total_time if total_time > 0 else 0
    }


def process_multiple_files(input_files, quality=None, quality_per_subject=None):
    """
    Process multiple JSON files and calculate aggregate statistics.
    
    Args:
        input_files: List of paths to JSON files
        quality: Optional quality score to include in the aggregate graph
        quality_per_subject: Optional list of 8 quality scores for subjects
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
            stats = calculate_tokens_per_second(input_file, quality, quality_per_subject)
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

    parser.add_argument(
        '--quality',
        type=float,
        default=None,
        help='Optional quality score to include in the aggregate graph'
    )

    parser.add_argument(
        '--quality_per_subject',
        nargs=8,
        type=float,
        default=None,
        help='List of 8 quality scores corresponding to coding, extraction, humanities, math, reasoning, roleplay, stem, writing'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    for input_file in args.input_files:
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            parser.exit(1)
    
    # Process files
    if len(args.input_files) == 1:
        calculate_tokens_per_second(args.input_files[0], quality=args.quality, quality_per_subject=args.quality_per_subject)
    else:
        process_multiple_files(args.input_files, quality=args.quality, quality_per_subject=args.quality_per_subject)

