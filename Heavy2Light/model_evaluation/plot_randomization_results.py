#!/usr/bin/env python3
"""
Plot randomization control results from a saved pickle file.
Allows re-plotting without re-running the analysis.
"""

import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def visualize_randomization_results_styled(randomization_results, threshold=80, suffix='', output_dir='.'):
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

        # Add significance stars ABOVE each bar individually (not at same height)
        for i, group in enumerate(group_names):
            p_val = randomization_results[group]['randomized'][level]['p_value_pct']

            # Get the max height of the two bars for this group
            obs_height = observed_pcts[i]
            rand_height = random_pcts[i]
            max_height = max(obs_height, rand_height)

            # Position asterisk above the taller bar with some padding
            asterisk_y = max_height + 0.5

            if p_val < 0.001:
                ax.text(i, asterisk_y, '***', ha='center', va='bottom', fontsize=16, fontweight='bold')
            elif p_val < 0.01:
                ax.text(i, asterisk_y, '**', ha='center', va='bottom', fontsize=16, fontweight='bold')
            elif p_val < 0.05:
                ax.text(i, asterisk_y, '*', ha='center', va='bottom', fontsize=16, fontweight='bold')

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

        filename = f'{output_dir}/randomization_control_{level}_threshold_{threshold}{suffix}.png'
        plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=300)
        print(f"Saved: {filename}")
        plt.close()

    # Print results table
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
    parser = argparse.ArgumentParser(description='Plot randomization control results from pickle file')
    parser.add_argument('--pkl', type=str, required=True, help='Path to pickle file with results')
    parser.add_argument('--threshold', type=int, default=80, help='Threshold used in analysis (for filename)')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output filenames')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for plots')

    args = parser.parse_args()

    print(f"Loading results from: {args.pkl}")
    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    print(f"Creating plots with threshold={args.threshold}, suffix='{args.suffix}'")
    visualize_randomization_results_styled(
        results,
        threshold=args.threshold,
        suffix=args.suffix,
        output_dir=args.output_dir
    )

    print("\nPlotting complete!")

if __name__ == '__main__':
    main()
