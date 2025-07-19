"""
CHAI-1 Score Analyzer
Calculates mean scores across all folded sequences from NPZ files.
"""

import numpy as np
import sys
import os
import glob
from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse

def find_npz_files(base_directory, pattern="scores.model_idx_0.npz"):
    """
    Find all NPZ files in subdirectories of the base directory.
    
    Args:
        base_directory (str): Base directory to search
        pattern (str): Pattern to match NPZ files
    
    Returns:
        dict: Dictionary mapping sequence names to NPZ file paths
    """
    npz_files = {}
    
    # Search for NPZ files in all subdirectories
    search_pattern = os.path.join(base_directory, "**", pattern)
    found_files = glob.glob(search_pattern, recursive=True)
    
    for file_path in found_files:
        # Extract sequence name from directory structure
        dir_name = os.path.basename(os.path.dirname(file_path))
        # Handle multiple model indices if they exist
        model_idx = os.path.basename(file_path).split('.')[-2]  # Extract model_idx_X
        key = f"{dir_name}_{model_idx}" if "model_idx" in model_idx else dir_name
        npz_files[key] = file_path
    
    return npz_files

def load_scores_from_npz(filepath):
    """
    Load scores from a single NPZ file.
    
    Args:
        filepath (str): Path to NPZ file
    
    Returns:
        dict: Dictionary of scores
    """
    try:
        data = np.load(filepath)
        scores = {}
        
        for key in data.keys():
            arr = data[key]
            # Convert to scalar if it's a 0-dimensional array
            if arr.ndim == 0:
                scores[key] = float(arr.item())
            else:
                # For arrays, calculate mean
                scores[key] = float(np.mean(arr))
        
        data.close()
        return scores
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_all_scores(base_directory, output_file=None, verbose=True):
    """
    Analyze scores from all NPZ files in the directory structure.
    
    Args:
        base_directory (str): Base directory containing folded sequences
        output_file (str): Optional CSV output file path
        verbose (bool): Whether to print detailed output
    
    Returns:
        pd.DataFrame: DataFrame containing all scores
    """
    
    print(f"Analyzing scores in: {base_directory}")
    print("=" * 60)
    
    # Find all NPZ files
    npz_files = find_npz_files(base_directory)
    
    if not npz_files:
        print("No NPZ files found!")
        return None
    
    print(f"Found {len(npz_files)} NPZ files:")
    if verbose:
        for seq_name, filepath in npz_files.items():
            print(f"  {seq_name}: {filepath}")
    
    print("\n" + "=" * 60)
    
    # Load scores from all files
    all_scores = {}
    failed_files = []
    
    for seq_name, filepath in npz_files.items():
        scores = load_scores_from_npz(filepath)
        if scores is not None:
            all_scores[seq_name] = scores
        else:
            failed_files.append(seq_name)
    
    if failed_files:
        print(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    if not all_scores:
        print("No scores could be loaded!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_scores, orient='index')
    
    # Calculate statistics
    print(f"\nSuccessfully loaded scores from {len(all_scores)} sequences")
    print(f"Score types found: {list(df.columns)}")
    
    # Display individual scores
    if verbose:
        print("\n" + "=" * 60)
        print("Individual Sequence Scores:")
        print("-" * 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(df)
    
    # Calculate summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS:")
    print("=" * 60)
    
    summary_stats = df.describe()
    print(summary_stats)
    
    # Print mean scores prominently
    print("\n" + "=" * 60)
    print("MEAN SCORES ACROSS ALL SEQUENCES:")
    print("=" * 60)
    
    means = df.mean()
    for score_type, mean_value in means.items():
        print(f"{score_type:25s}: {mean_value:.6f}")
    
    # Additional statistics
    print(f"\nNumber of sequences analyzed: {len(df)}")
    print(f"Score types: {len(df.columns)}")
    
    # Find best and worst sequences for key metrics
    if 'aggregate_score' in df.columns:
        best_seq = df['aggregate_score'].idxmax()
        worst_seq = df['aggregate_score'].idxmin()
        print(f"\nBest aggregate score: {df.loc[best_seq, 'aggregate_score']:.6f} ({best_seq})")
        print(f"Worst aggregate score: {df.loc[worst_seq, 'aggregate_score']:.6f} ({worst_seq})")
    
    if 'ptm' in df.columns:
        best_ptm = df['ptm'].idxmax()
        print(f"Best PTM score: {df.loc[best_ptm, 'ptm']:.6f} ({best_ptm})")
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")
        
        # Also save summary statistics
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics saved to: {summary_file}")
    
    return df

def create_score_comparison_report(df, output_dir=None):
    """
    Create additional analysis reports.
    
    Args:
        df (pd.DataFrame): DataFrame with scores
        output_dir (str): Directory to save reports
    """
    if df is None or df.empty:
        return
    
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS:")
    print("=" * 60)
    
    # Correlation analysis
    if len(df.columns) > 1:
        print("\nCorrelation between score types:")
        correlation_matrix = df.corr()
        print(correlation_matrix)
    
    # Score distribution analysis
    print(f"\nScore ranges:")
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        print(f"{col:25s}: {min_val:.6f} to {max_val:.6f} (range: {range_val:.6f})")
    
    # Identify outliers (sequences with scores > 2 std from mean)
    print(f"\nPotential outliers (>2 std from mean):")
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        outliers = df[(df[col] > mean_val + 2*std_val) | (df[col] < mean_val - 2*std_val)]
        if not outliers.empty:
            print(f"\n{col}:")
            for idx in outliers.index:
                value = outliers.loc[idx, col]
                z_score = (value - mean_val) / std_val
                print(f"  {idx}: {value:.6f} (z-score: {z_score:.2f})")

def main():
    """Main function to handle command line arguments."""
    
    parser = argparse.ArgumentParser(description='Analyze CHAI-1 folding scores from NPZ files')
    parser.add_argument('directory', nargs='?', 
                       default='/storage/homefs/lb24i892/chai_folding/outputs',
                       help='Base directory containing folded sequences')
    parser.add_argument('--output', '-o', 
                       help='Output CSV file path')
    parser.add_argument('--pattern', 
                       default='scores.model_idx_0.npz',
                       help='Pattern to match NPZ files')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary statistics')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' not found!")
        sys.exit(1)
    
    verbose = not (args.quiet or args.summary_only)
    
    # Analyze scores
    df = analyze_all_scores(args.directory, args.output, verbose)
    
    if df is not None and not args.summary_only:
        create_score_comparison_report(df, args.output)
    
    print(f"\nAnalysis completed!")
    if args.output:
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()


