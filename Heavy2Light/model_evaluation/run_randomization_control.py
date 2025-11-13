"""
Randomization control analysis for V gene consistency.
Run this script to perform permutation testing on HC-LC pairing constraints.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import re
import pickle
import argparse
from datetime import datetime

def extract_v_gene_info(gene_name):
    """Extract V gene information at different levels of granularity."""
    if pd.isna(gene_name) or gene_name == '':
        return {'full': None, 'simplified': None, 'family': None}

    gene_name = str(gene_name).strip()

    # Full V gene name
    full_v_gene = gene_name

    # Simplified V gene (remove allelic variants like *01, *02, etc.)
    simplified_v_gene = re.sub(r'\*\d+', '', gene_name)

    # V gene family (extract just the family part, e.g., IGHV1, IGLV2, etc.)
    family_match = re.match(r'(IG[HLK]V\d+)', gene_name)
    v_gene_family = family_match.group(1) if family_match else gene_name.split('-')[0]

    return {
        'full': full_v_gene,
        'simplified': simplified_v_gene,
        'family': v_gene_family
    }

def randomize_hc_lc_pairings_corrected(df, group_name):
    """
    Randomize HC-LC pairings while maintaining the same structure.

    This shuffles which light chain sequences are associated with which heavy chains,
    while preserving the number of sequences per heavy chain.
    """

    # Make a copy to avoid modifying original
    randomized_df = df.copy()

    # Identify columns that belong to light chains (should be shuffled)
    lc_cols = ['gen_light_seq', 'gen_light_gene_name', 'gen_light_chain_number',
               'predicted_gen_light_seq_label', 'perplexity', 'BLOSUM', 'similarity']

    # Also check for true_light columns that might exist
    additional_lc_cols = ['true_light_seq', 'true_light_gene_name', 'predicted_true_light_seq_label']

    # Keep only columns that exist in the dataframe
    all_lc_cols = lc_cols + additional_lc_cols
    lc_cols_to_shuffle = [col for col in all_lc_cols if col in df.columns]

    # Extract light chain data
    lc_data = df[lc_cols_to_shuffle].copy()

    # Shuffle the light chain data
    shuffled_indices = np.random.permutation(len(lc_data))
    lc_data_shuffled = lc_data.iloc[shuffled_indices].reset_index(drop=True)

    # Replace light chain columns with shuffled data
    randomized_df[lc_cols_to_shuffle] = lc_data_shuffled.values

    return randomized_df

def calculate_v_gene_consistency_metrics(df, level='simplified', threshold=80):
    """Calculate V gene consistency metrics for a given dataframe."""

    metrics = {
        'pct_above_threshold': 0,
        'avg_dominant_frequency': 0,
        'median_dominant_frequency': 0,
        'avg_unique_v_genes': 0,
        'count_above_threshold': 0,
        'total_heavy_chains': 0
    }

    # Group by heavy chain
    heavy_chain_groups = df.groupby('heavy_chain_number')

    dominant_frequencies = []
    unique_v_gene_counts = []
    above_threshold_count = 0

    for heavy_chain_num, group in heavy_chain_groups:
        # Extract V gene information
        v_genes = []
        for _, row in group.iterrows():
            v_info = extract_v_gene_info(row['gen_light_gene_name'])
            if v_info[level] is not None:
                v_genes.append(v_info[level])

        if len(v_genes) == 0:
            continue

        # Calculate frequencies
        v_gene_counts = {v: v_genes.count(v) for v in set(v_genes)}
        total = len(v_genes)
        v_gene_frequencies = {v: (count / total) * 100 for v, count in v_gene_counts.items()}

        # Get dominant V gene frequency
        dominant_freq = max(v_gene_frequencies.values())
        dominant_frequencies.append(dominant_freq)
        unique_v_gene_counts.append(len(set(v_genes)))

        if dominant_freq >= threshold:
            above_threshold_count += 1

    # Calculate metrics
    total_hc = len(dominant_frequencies)
    if total_hc > 0:
        metrics['pct_above_threshold'] = (above_threshold_count / total_hc) * 100
        metrics['avg_dominant_frequency'] = np.mean(dominant_frequencies)
        metrics['median_dominant_frequency'] = np.median(dominant_frequencies)
        metrics['avg_unique_v_genes'] = np.mean(unique_v_gene_counts)
        metrics['count_above_threshold'] = above_threshold_count
        metrics['total_heavy_chains'] = total_hc

    return metrics

def run_randomization_control_corrected(csv_file_path, n_iterations=1000, min_sequences=4, threshold=80, random_seed=42):
    """Run randomization control analysis for all four groups."""

    np.random.seed(random_seed)

    print(f"Running CORRECTED randomization control with {n_iterations} iterations...")
    print(f"Minimum sequences per heavy chain: {min_sequences}")
    print(f"Frequency threshold: {threshold}%\n")

    # Read and prepare data
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()

    # Define groups
    groups = {
        'naive': (0, 0),
        'memory': (1, 1),
        'mismatch_naive': (0, 1),
        'mismatch_memory': (1, 0)
    }

    results = {}

    for group_name, (heavy_pred, light_pred) in groups.items():
        print(f"\n{'='*70}")
        print(f"Processing {group_name.upper()} group ({heavy_pred},{light_pred})")
        print(f"{'='*70}")

        # Filter group
        group_df = df[(df['predicted_input_heavy_seq_label'] == heavy_pred) &
                      (df['predicted_gen_light_seq_label'] == light_pred)]

        # Filter by minimum sequences
        heavy_chain_counts = group_df.groupby('heavy_chain_number').size()
        valid_heavy_chains = heavy_chain_counts[heavy_chain_counts >= min_sequences].index
        filtered_df = group_df[group_df['heavy_chain_number'].isin(valid_heavy_chains)]

        print(f"Total rows: {len(filtered_df)}")
        print(f"Unique heavy chains: {len(valid_heavy_chains)}")

        # Calculate observed metrics
        observed_metrics = {}
        for level in ['simplified', 'family']:
            observed_metrics[level] = calculate_v_gene_consistency_metrics(
                filtered_df, level=level, threshold=threshold
            )
            print(f"\nObserved {level} V gene metrics:")
            print(f"  Heavy chains ≥{threshold}% frequency: {observed_metrics[level]['count_above_threshold']}/{observed_metrics[level]['total_heavy_chains']} = {observed_metrics[level]['pct_above_threshold']:.2f}%")
            print(f"  Avg dominant frequency: {observed_metrics[level]['avg_dominant_frequency']:.2f}%")

        # Run randomization iterations
        print(f"\nRunning {n_iterations} randomization iterations...")

        randomized_metrics = {
            'simplified': {
                'pct_above_threshold': [],
                'avg_dominant_frequency': []
            },
            'family': {
                'pct_above_threshold': [],
                'avg_dominant_frequency': []
            }
        }

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}...")

            # Randomize HC-LC pairings
            randomized_df = randomize_hc_lc_pairings_corrected(filtered_df, group_name)

            # Calculate metrics for randomized data
            for level in ['simplified', 'family']:
                metrics = calculate_v_gene_consistency_metrics(
                    randomized_df, level=level, threshold=threshold
                )
                randomized_metrics[level]['pct_above_threshold'].append(metrics['pct_above_threshold'])
                randomized_metrics[level]['avg_dominant_frequency'].append(metrics['avg_dominant_frequency'])

        # Calculate statistics on randomized distribution
        randomized_stats = {}
        for level in ['simplified', 'family']:
            pct_values = randomized_metrics[level]['pct_above_threshold']
            freq_values = randomized_metrics[level]['avg_dominant_frequency']

            # Calculate p-value with permutation correction
            n_hits_pct = np.sum(np.array(pct_values) >= observed_metrics[level]['pct_above_threshold'])
            n_hits_freq = np.sum(np.array(freq_values) >= observed_metrics[level]['avg_dominant_frequency'])

            p_value_pct = (n_hits_pct + 1) / (n_iterations + 1)
            p_value_freq = (n_hits_freq + 1) / (n_iterations + 1)

            randomized_stats[level] = {
                'mean_pct_above_threshold': np.mean(pct_values),
                'std_pct_above_threshold': np.std(pct_values),
                'mean_avg_dominant_frequency': np.mean(freq_values),
                'std_avg_dominant_frequency': np.std(freq_values),
                'p_value_pct': p_value_pct,
                'p_value_freq': p_value_freq,
                'n_hits_pct': n_hits_pct,
                'n_hits_freq': n_hits_freq,
                'pct_values': pct_values,
                'freq_values': freq_values
            }

            print(f"\nRandomized {level} V gene statistics:")
            print(f"  Avg % ≥{threshold}% frequency: {randomized_stats[level]['mean_pct_above_threshold']:.2f}% ± {randomized_stats[level]['std_pct_above_threshold']:.2f}%")
            print(f"  Randomized hits ≥ observed: {n_hits_pct}/{n_iterations}")
            print(f"  P-value: {p_value_pct:.6e}")

        results[group_name] = {
            'observed': observed_metrics,
            'randomized': randomized_stats,
            'n_heavy_chains': len(valid_heavy_chains),
            'n_iterations': n_iterations
        }

    return results

def visualize_randomization_results_styled(randomization_results, threshold=80, suffix=''):
    """Create visualizations comparing observed vs randomized V gene consistency."""

    group_labels = {
        'naive': 'Match naive',
        'memory': 'Match memory',
        'mismatch_naive': 'Mismatch naive',
        'mismatch_memory': 'Mismatch memory'
    }

    colors = {
        'naive': '#83aff0',
        'memory': '#2c456b',
        'mismatch_naive': '#c97b6d',
        'mismatch_memory': '#c1440e'
    }

    levels = ['simplified', 'family']
    group_names = ['naive', 'memory', 'mismatch_naive', 'mismatch_memory']

    bar_width = 0.35
    x_positions = np.arange(len(group_names))

    y_limits = {
        'simplified': 10,
        'family': 20
    }

    asterisk_heights = {
        'simplified': y_limits['simplified'] * 0.9,
        'family': y_limits['family'] * 0.9
    }

    for level in levels:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        observed_pcts = [randomization_results[g]['observed'][level]['pct_above_threshold'] for g in group_names]
        random_pcts = [randomization_results[g]['randomized'][level]['mean_pct_above_threshold'] for g in group_names]

        bars_obs = ax.bar(x_positions - bar_width/2, observed_pcts, bar_width, label='Observed',
                         color=[colors[g] for g in group_names], alpha=1.0, edgecolor='black', linewidth=0.5)
        bars_rand = ax.bar(x_positions + bar_width/2, random_pcts, bar_width, label='Randomized',
                          color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars_obs, observed_pcts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{pct:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        for i, (bar, pct) in enumerate(zip(bars_rand, random_pcts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{pct:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Add significance stars
        for i, group in enumerate(group_names):
            p_val = randomization_results[group]['randomized'][level]['p_value_pct']

            if p_val < 0.001:
                ax.text(i, asterisk_heights[level], '***', ha='center', va='bottom', fontsize=16, fontweight='bold')
            elif p_val < 0.01:
                ax.text(i, asterisk_heights[level], '**', ha='center', va='bottom', fontsize=16, fontweight='bold')
            elif p_val < 0.05:
                ax.text(i, asterisk_heights[level], '*', ha='center', va='bottom', fontsize=16, fontweight='bold')

        ylabel = 'V gene sharing [%]' if level == 'simplified' else 'V gene family sharing [%]'
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([group_labels[g] for g in group_names], fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(0, y_limits[level])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=12, loc='upper left', frameon=False)

        plt.tight_layout()

        filename = f'randomization_control_{level}_threshold_{threshold}{suffix}.png'
        plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=600)
        print(f"Saved: {filename}")
        plt.close()

    # Print results
    n_iterations = randomization_results[group_names[0]]['n_iterations']

    print("\n" + "="*100)
    print("RANDOMIZATION CONTROL RESULTS")
    print("="*100)
    print(f"\nTotal randomization iterations: {n_iterations}")
    print("P-value calculation: Permutation test with correction (n_hits + 1) / (n_iterations + 1)")
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001\n")

    for level in levels:
        print(f"\n{level.upper()} V GENE LEVEL:")
        print("-" * 100)
        print(f"{'Group':<20} {'Observed %':<15} {'Random %':<18} {'Difference':<15} {'Hits':<10} {'P-value':<15} {'Sig'}")
        print("-" * 100)

        for group in group_names:
            label = group_labels[group]
            obs_pct = randomization_results[group]['observed'][level]['pct_above_threshold']
            rand_pct = randomization_results[group]['randomized'][level]['mean_pct_above_threshold']
            rand_std = randomization_results[group]['randomized'][level]['std_pct_above_threshold']
            p_val = randomization_results[group]['randomized'][level]['p_value_pct']
            n_hits = randomization_results[group]['randomized'][level]['n_hits_pct']
            diff = obs_pct - rand_pct

            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            hits_str = f"{n_hits}/{n_iterations}"
            print(f"{label:<20} {obs_pct:>9.6f}%     {rand_pct:>8.6f}±{rand_std:<7.6f}  {diff:>+9.6f}pp     {hits_str:<10} {p_val:<15.6e} {sig}")

        print()

def main():
    parser = argparse.ArgumentParser(description='Run randomization control analysis for V gene consistency')
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of randomization iterations (default: 1000)')
    parser.add_argument('--min-seqs', type=int, default=4, help='Minimum sequences per heavy chain (default: 4)')
    parser.add_argument('--threshold', type=int, default=80, help='V gene frequency threshold (default: 80)')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output filenames')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='randomization_results.pkl', help='Output pickle file')

    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Starting analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters:")
    print(f"  CSV file: {args.csv}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Min sequences: {args.min_seqs}")
    print(f"  Threshold: {args.threshold}%")
    print(f"  Random seed: {args.seed}")
    print(f"  Output file: {args.output}")
    print()

    # Run analysis
    results = run_randomization_control_corrected(
        csv_file_path=args.csv,
        n_iterations=args.iterations,
        min_sequences=args.min_seqs,
        threshold=args.threshold,
        random_seed=args.seed
    )

    # Create visualizations
    print("\n\nCreating visualizations...")
    visualize_randomization_results_styled(results, threshold=args.threshold, suffix=args.suffix)

    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {args.output}")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nCompleted at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {duration}")

if __name__ == '__main__':
    main()
