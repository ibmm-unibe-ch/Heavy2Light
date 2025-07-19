"""
CHAI-1 Score Analyzer with pLDDT
Calculates mean scores across all folded sequences from NPZ files and pLDDT from CIF files.
"""

import numpy as np
import sys
import os
import glob
from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse

def find_files(base_directory, npz_pattern="scores.model_idx_0.npz", cif_pattern="pred.model_idx_0.cif"):
    """
    Find all NPZ and CIF files in subdirectories of the base directory.
    
    Args:
        base_directory (str): Base directory to search
        npz_pattern (str): Pattern to match NPZ files
        cif_pattern (str): Pattern to match CIF files
    
    Returns:
        tuple: (npz_files dict, cif_files dict) mapping sequence names to file paths
    """
    npz_files = {}
    cif_files = {}
    
    # Search for NPZ files in all subdirectories
    npz_search_pattern = os.path.join(base_directory, "**", npz_pattern)
    found_npz_files = glob.glob(npz_search_pattern, recursive=True)
    
    for file_path in found_npz_files:
        # Extract sequence name from directory structure
        dir_name = os.path.basename(os.path.dirname(file_path))
        # Handle multiple model indices if they exist
        model_idx = os.path.basename(file_path).split('.')[-2]  # Extract model_idx_X
        key = f"{dir_name}_{model_idx}" if "model_idx" in model_idx else dir_name
        npz_files[key] = file_path
    
    # Search for CIF files in all subdirectories
    cif_search_pattern = os.path.join(base_directory, "**", cif_pattern)
    found_cif_files = glob.glob(cif_search_pattern, recursive=True)
    
    for file_path in found_cif_files:
        # Extract sequence name from directory structure
        dir_name = os.path.basename(os.path.dirname(file_path))
        # Handle multiple model indices if they exist in filename
        filename = os.path.basename(file_path)
        if "model_idx" in filename:
            model_idx = filename.split('.')[-2] if '.' in filename else "model_idx_0"
            key = f"{dir_name}_{model_idx}"
        else:
            key = dir_name
        cif_files[key] = file_path
    
    return npz_files, cif_files

def calculate_mean_plddt_from_cif(cif_file_path):
    """
    Calculate the mean pLDDT score from a .cif file.
    
    Args:
        cif_file_path (str): Path to the .cif file
    
    Returns:
        float or None: Mean pLDDT score, or None if calculation fails
    """
    try:
        plddt_scores = []
        
        with open(cif_file_path, 'r') as file:
            for line in file:
                # Check if line starts with ATOM
                if line.startswith('ATOM'):
                    # Split the line by whitespace
                    columns = line.split()
                    
                    # The pLDDT score is the second-to-last column
                    if len(columns) >= 2:
                        try:
                            plddt_score = float(columns[-2])
                            plddt_scores.append(plddt_score)
                        except ValueError:
                            # Skip lines where conversion fails
                            continue
        
        if plddt_scores:
            return np.mean(plddt_scores)
        else:
            return None
            
    except Exception as e:
        print(f"Error calculating pLDDT from {cif_file_path}: {e}")
        return None

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

def analyze_all_scores(base_directory, output_file=None, verbose=True, npz_pattern="scores.model_idx_0.npz", cif_pattern="pred.model_idx_0.cif"):
    """
    Analyze scores from all NPZ and CIF files in the directory structure.
    
    Args:
        base_directory (str): Base directory containing folded sequences
        output_file (str): Optional CSV output file path
        verbose (bool): Whether to print detailed output
        npz_pattern (str): Pattern to match NPZ files
        cif_pattern (str): Pattern to match CIF files
    
    Returns:
        pd.DataFrame: DataFrame containing all scores
    """
    
    print(f"Analyzing scores in: {base_directory}")
    print("=" * 60)
    
    # Find all NPZ and CIF files
    npz_files, cif_files = find_files(base_directory, npz_pattern, cif_pattern)
    
    if not npz_files and not cif_files:
        print("No NPZ or CIF files found!")
        return None
    
    print(f"Found {len(npz_files)} NPZ files and {len(cif_files)} CIF files:")
    if verbose:
        print("NPZ files:")
        for seq_name, filepath in npz_files.items():
            print(f"  {seq_name}: {filepath}")
        print("CIF files:")
        for seq_name, filepath in cif_files.items():
            print(f"  {seq_name}: {filepath}")
    
    print("\n" + "=" * 60)
    
    # Get all unique sequence names
    all_sequences = set(npz_files.keys()) | set(cif_files.keys())
    
    # Load scores from all files
    all_scores = {}
    failed_files = []
    
    for seq_name in all_sequences:
        scores = {}
        
        # Load NPZ scores if available
        if seq_name in npz_files:
            npz_scores = load_scores_from_npz(npz_files[seq_name])
            if npz_scores is not None:
                scores.update(npz_scores)
            else:
                failed_files.append(f"{seq_name} (NPZ)")
        
        # Load CIF pLDDT scores if available
        if seq_name in cif_files:
            plddt_score = calculate_mean_plddt_from_cif(cif_files[seq_name])
            if plddt_score is not None:
                scores['mean_plddt'] = plddt_score
            else:
                failed_files.append(f"{seq_name} (CIF)")
        
        # Only add to results if we got at least some scores
        if scores:
            all_scores[seq_name] = scores
    
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
    
    if 'mean_plddt' in df.columns:
        best_plddt = df['mean_plddt'].idxmax()
        worst_plddt = df['mean_plddt'].idxmin()
        print(f"Best mean pLDDT: {df.loc[best_plddt, 'mean_plddt']:.6f} ({best_plddt})")
        print(f"Worst mean pLDDT: {df.loc[worst_plddt, 'mean_plddt']:.6f} ({worst_plddt})")
    
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
        
        # Highlight pLDDT correlations if available
        if 'mean_plddt' in df.columns:
            print(f"\nCorrelations with mean pLDDT:")
            plddt_correlations = correlation_matrix['mean_plddt'].drop('mean_plddt')
            for score_type, corr_value in plddt_correlations.items():
                print(f"  {score_type:20s}: {corr_value:.4f}")
    
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
    
    parser = argparse.ArgumentParser(description='Analyze CHAI-1 folding scores from NPZ and CIF files')
    parser.add_argument('directory', nargs='?', 
                       default='/storage/homefs/lb24i892/chai_folding/outputs',
                       help='Base directory containing folded sequences')
    parser.add_argument('--output', '-o', 
                       help='Output CSV file path')
    parser.add_argument('--npz-pattern', 
                       default='scores.model_idx_0.npz',
                       help='Pattern to match NPZ files')
    parser.add_argument('--cif-pattern', 
                       default='pred.model_idx_0.cif',
                       help='Pattern to match CIF files')
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
    df = analyze_all_scores(args.directory, args.output, verbose, args.npz_pattern, args.cif_pattern)
    
    if df is not None and not args.summary_only:
        create_score_comparison_report(df, args.output)
    
    print(f"\nAnalysis completed!")
    if args.output:
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()