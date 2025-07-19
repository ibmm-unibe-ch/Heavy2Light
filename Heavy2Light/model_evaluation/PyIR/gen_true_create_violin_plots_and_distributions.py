import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
import argparse
import os

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare generated and true sequences from JSON file')
    
    # Required arguments
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output files')
    
    # Optional arguments
    parser.add_argument('--save_prefix', type=str, default='sequence_comparison',
                        help='Prefix for saved files (default: sequence_comparison)')
    parser.add_argument('--transparent', action='store_true',
                        help='Save plots with transparent background')
    parser.add_argument('--color_palette', type=str, 
                        choices=['Set1', 'Set2', 'Set3', 'Dark2', 'Paired', 'tab10', 'husl', 'deep'],
                        default='Set2', help='Color palette for plots (default: Set2)')
    parser.add_argument('--mean_line_width', type=float, default=0.4,
                        help='Width of mean line as fraction of violin width (default: 0.4)')
    parser.add_argument('--figure_width', type=float, default=6,
                        help='Width of figures in inches (default: 6)')
    parser.add_argument('--figure_height', type=float, default=4,
                        help='Height of figures in inches (default: 4)')
    parser.add_argument('--font_size', type=int, default=12,
                        help='Base font size for plots (default: 12)')
    parser.add_argument('--title_font_size', type=int, default=14,
                        help='Font size for titles (default: 14)')
    parser.add_argument('--dpi', type=int, default=450,
                        help='DPI for saved figures (default: 450)')
    parser.add_argument('--test_entries', type=int, default=None,
                        help='Number of entries to use for testing (creates test dataset)')
    
    return parser.parse_args()

def get_region_colors(regions, palette_name):
    """Get consistent colors for regions across all plots"""
    colors = sns.color_palette(palette_name, n_colors=len(regions))
    return dict(zip(regions, colors))

def create_test_dataset(file_path, num_entries=10):
    """
    Create a small test dataset from the original JSON file.
    
    Parameters:
    -----------
    file_path : str
        Path to the original JSON file
    num_entries : int
        Number of JSON entries to include in the test dataset
    
    Returns:
    --------
    str
        Path to the test dataset file
    """
    test_file_path = file_path.replace('.json', '_test.json')
    
    count = 0
    entries = []
    
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(line.strip())
            count += 1
            if count >= num_entries:
                break
    
    with open(test_file_path, 'w') as f:
        for entry in entries:
            f.write(f"{entry}\n")
    
    print(f"Created test dataset with {count} entries at {test_file_path}")
    return test_file_path

def pair_sequences(sequences):
    """
    Pairs generated and true sequences by their base ID.
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries parsed from JSON
    
    Returns:
    --------
    dict
        Dictionary with base_ids as keys and pairs of (gen_seq, true_seq) as values
    """
    # Create dictionaries for gen and true sequences
    gen_sequences = {}
    true_sequences = {}
    pairs = {}
    
    # Extract base ID pattern from sequence ID
    pattern = re.compile(r'(gen|true)_(seq_\d+_[a-f0-9]+_sim[\d\.]+_blosum[\d\.]+_perp[\d\.]+)')
    
    for seq in sequences:
        seq_id = seq.get('Sequence ID', '')
        match = pattern.match(seq_id)
        
        if match:
            prefix, base_id = match.groups()
            if prefix == 'gen':
                gen_sequences[base_id] = seq
            elif prefix == 'true':
                true_sequences[base_id] = seq
    
    # Pair up sequences with the same base ID
    for base_id in set(gen_sequences.keys()) & set(true_sequences.keys()):
        pairs[base_id] = (gen_sequences[base_id], true_sequences[base_id])
    
    print(f"Found {len(pairs)} paired sequences")
    return pairs

def calculate_global_alignment(seq1, seq2, blosum_matrix=None):
    """
    Calculate global alignment score for two sequences.
    
    Parameters:
    -----------
    seq1, seq2 : str
        Amino acid sequences to align
    blosum_matrix : Bio.Align.substitution_matrices.Array or None
        BLOSUM substitution matrix. If None, uses BLOSUM62
    
    Returns:
    --------
    dict
        Dictionary with alignment metrics
    """
    if not seq1 or not seq2:
        raise ValueError("One or both sequences are empty.")
    
    # Validate sequences contain only valid amino acids
    valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
    seq1 = re.sub(f"[^{valid_aa}]", "", seq1.upper())
    seq2 = re.sub(f"[^{valid_aa}]", "", seq2.upper())
    
    # Handle BLOSUM matrix
    if blosum_matrix is None:
        blosum_matrix = substitution_matrices.load("BLOSUM62")
    
    # Perform global alignment with gap penalties
    try:
        alignments = pairwise2.align.globalds(
            seq1, seq2, 
            blosum_matrix, 
            -10,  # gap open penalty
            -4    # gap extension penalty
        )
    except Exception as e:
        raise ValueError(f"Alignment failed: {str(e)}")
    
    if not alignments:
        raise ValueError("No alignments were found.")

    # Get the best alignment
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2, score, start, end = best_alignment
    
    # Calculate metrics
    matches = 0
    mismatches = 0
    gaps = 0
    
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == b and a != '-' and b != '-':
            matches += 1
        elif a == '-' or b == '-':
            gaps += 1
        else:
            mismatches += 1
    
    alignment_length = len(aligned_seq1)
    identity = (matches / alignment_length) * 100 if alignment_length > 0 else 0
    
    return {
        'score': score,
        'identity': identity,
        'matches': matches,
        'mismatches': mismatches,
        'gaps': gaps,
        'length': alignment_length,
        'aligned_seq1': aligned_seq1,
        'aligned_seq2': aligned_seq2
    }

def extract_region_sequences(sequence_obj):
    """
    Extract amino acid sequences for each region from the sequence object.
    
    Parameters:
    -----------
    sequence_obj : dict
        Sequence object from JSON data
    
    Returns:
    --------
    dict
        Dictionary with region names as keys and sequences as values
    """
    regions = {}
    
    # Extract FR and CDR regions
    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3']:
        if region in sequence_obj:
            # For regions that have the 'AA' field directly
            if 'AA' in sequence_obj[region] and sequence_obj[region]['AA']:
                regions[region] = sequence_obj[region]['AA']
            # For CDR3 and other regions that need to be extracted from Raw Sequence
            elif 'from' in sequence_obj[region] and 'to' in sequence_obj[region] and 'Raw Sequence' in sequence_obj:
                # Convert from 1-based to 0-based indexing
                start_idx = int(float(sequence_obj[region]['from'])) - 1
                end_idx = int(float(sequence_obj[region]['to']))
                
                # Make sure indices are valid
                if 0 <= start_idx < end_idx <= len(sequence_obj['Raw Sequence']):
                    regions[region] = sequence_obj['Raw Sequence'][start_idx:end_idx]
    
    # Extract the full sequence
    if 'Raw Sequence' in sequence_obj:
        regions['Total'] = sequence_obj['Raw Sequence']
    elif 'NT-Trimmed' in sequence_obj:
        regions['Total'] = sequence_obj['NT-Trimmed']
    
    return regions

def compare_paired_sequences(pairs):
    """
    Compare aligned pairs of generated and true sequences.
    
    Parameters:
    -----------
    pairs : dict
        Dictionary with base_ids as keys and pairs of (gen_seq, true_seq) as values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with comparison metrics
    """
    comparison_data = []
    blosum62 = substitution_matrices.load("BLOSUM62")
    
    for base_id, (gen_seq, true_seq) in pairs.items():
        entry = {
            'base_id': base_id,
            'gen_id': gen_seq.get('Sequence ID', ''),
            'true_id': true_seq.get('Sequence ID', ''),
            'gen_length': gen_seq.get('Sequence Length', 0),
            'true_length': true_seq.get('Sequence Length', 0)
        }
        
        # Extract sequences for each region
        gen_regions = extract_region_sequences(gen_seq)
        true_regions = extract_region_sequences(true_seq)
        
        # Calculate global alignment for each region
        for region in set(gen_regions.keys()) | set(true_regions.keys()):
            gen_seq_region = gen_regions.get(region, '')
            true_seq_region = true_regions.get(region, '')
            
            if gen_seq_region and true_seq_region:
                alignment = calculate_global_alignment(gen_seq_region, true_seq_region, blosum62)
                
                entry[f'{region}_alignment_score'] = alignment['score']
                entry[f'{region}_percent_identity'] = alignment['identity']
                entry[f'{region}_matches'] = alignment['matches']
                entry[f'{region}_mismatches'] = alignment['mismatches']
                entry[f'{region}_gaps'] = alignment['gaps']
                entry[f'{region}_length'] = alignment['length']
                entry[f'{region}_gen_length'] = len(gen_seq_region)
                entry[f'{region}_true_length'] = len(true_seq_region)
        
        comparison_data.append(entry)
    
    return pd.DataFrame(comparison_data)

def analyze_sequence_data(file_path):
    """
    Analyze sequence data from JSON file.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file with sequence data (one JSON object per line)
    """
    # Load data
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                seq_data = json.loads(line.strip())
                sequences.append(seq_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Successfully loaded {len(sequences)} sequences")
    
    # Pair generated and true sequences
    pairs = pair_sequences(sequences)
    
    # Compare the paired sequences
    df = compare_paired_sequences(pairs)
    
    # Basic statistics
    print("\n===== Basic Statistics =====")
    print(f"Total number of sequence pairs: {len(df)}")
    print(f"Average gen sequence length: {df['gen_length'].mean():.2f}")
    print(f"Average true sequence length: {df['true_length'].mean():.2f}")
    
    # Region statistics
    print("\n===== Region Statistics =====")
    regions = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']
    for region in regions:
        percent_id_col = f'{region}_percent_identity'
        
        if percent_id_col in df.columns and not df[percent_id_col].isna().all():
            mean_percent_id = df[percent_id_col].mean()
            std_percent_id = df[percent_id_col].std()
            median_percent_id = df[percent_id_col].median()
            
            print(f"{region}:")
            print(f"  Average percent identity: {mean_percent_id:.2f}%")
            print(f"  Standard deviation of percent identity: {std_percent_id:.2f}%")
            print(f"  Median percent identity: {median_percent_id:.2f}%")
            
            # Check distribution normality using Shapiro-Wilk test
            if len(df[percent_id_col].dropna()) >= 3:
                stat, p_value = stats.shapiro(df[percent_id_col].dropna())
                print(f"  Shapiro-Wilk test for normality (percent identity): p-value = {p_value:.4f}")
                print(f"  Distribution is {'likely normal' if p_value > 0.05 else 'not normal'}")
    
    return df, sequences

def generate_visualizations(df, args):
    """
    Generate visualizations from the sequence data including violin plots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the extracted sequence data
    args : argparse.Namespace
        Command line arguments
    """
    # Set matplotlib and seaborn style - white background, no grid
    plt.style.use('default')
    sns.set_style("white")
    sns.despine()
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': args.font_size,
        'axes.titlesize': args.title_font_size,
        'axes.labelsize': args.font_size,
        'xtick.labelsize': args.font_size,
        'ytick.labelsize': args.font_size,
        'legend.fontsize': args.font_size
    })
    
    # Get available regions and set up consistent colors
    available_regions = []
    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']:
        col_name = f'{region}_percent_identity'
        if col_name in df.columns and not df[col_name].isna().all():
            available_regions.append(region)
    
    region_colors = get_region_colors(available_regions, args.color_palette)
    
    # First figure: Sequence Length Distribution
    plt.figure(figsize=(args.figure_width, args.figure_height))
    
    # Create a dataframe for plotting both gen and true sequence lengths
    length_data = []
    for _, row in df.iterrows():
        length_data.append({'Type': 'Generated', 'Length': row['gen_length']})
        length_data.append({'Type': 'True', 'Length': row['true_length']})
    
    length_df = pd.DataFrame(length_data)
    
    type_colors = [sns.color_palette(args.color_palette)[0], sns.color_palette(args.color_palette)[1]]
    sns.histplot(data=length_df, x='Length', hue='Type', kde=True, multiple='stack', 
                 palette=type_colors)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{args.save_prefix}_sequence_length_distribution.png',
                dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
    plt.close()
    
    # Second figure: Percent Identity by Region - Violin Plots
    percent_identity_data = []
    
    for region in available_regions:
        col_name = f'{region}_percent_identity'
        if col_name in df.columns and not df[col_name].isna().all():
            for idx, value in df[col_name].items():
                percent_identity_data.append({
                    'Region': region,
                    'Percent Identity': value
                })
    
    if percent_identity_data:
        percent_identity_df = pd.DataFrame(percent_identity_data)
        
        plt.figure(figsize=(args.figure_width, args.figure_height))
        
        colors_list = [region_colors[region] for region in available_regions]
        ax = sns.violinplot(data=percent_identity_df, x='Region', y='Percent Identity', 
                            palette=colors_list, inner=None, order=available_regions)

        # Add mean lines to the violin plots (shorter lines)
        for i, region in enumerate(available_regions):
            region_data = percent_identity_df[percent_identity_df['Region'] == region]['Percent Identity']
            mean_val = region_data.mean()
            line_width = args.mean_line_width
            plt.hlines(y=mean_val, xmin=i-line_width, xmax=i+line_width, 
                      colors='black', linestyles='dashed', linewidth=1)

        plt.xlabel('Region')
        plt.ylabel('Light chain sequence recovery [%]')
        plt.ylim(0, 100)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/{args.save_prefix}_percent_identity_violin_plot.png',
                    dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
        plt.close()
    
    # Third figure: Alignment gaps by Region - Violin Plots
    gaps_data = []
    available_gaps_regions = []

    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']:
        col_name = f'{region}_gaps'
        if col_name in df.columns and not df[col_name].isna().all():
            available_gaps_regions.append(region)
            for idx, value in df[col_name].items():
                gaps_data.append({
                    'Region': region,
                    'Gaps': value
                })
    
    if gaps_data:
        gaps_df = pd.DataFrame(gaps_data)
        
        gaps_region_colors = get_region_colors(available_gaps_regions, args.color_palette)
        colors_list = [gaps_region_colors[region] for region in available_gaps_regions]
        
        plt.figure(figsize=(args.figure_width, args.figure_height))
        ax = sns.violinplot(data=gaps_df, x='Region', y='Gaps', 
                            palette=colors_list, inner=None, order=available_gaps_regions)

        # Add mean lines to the violin plots
        for i, region in enumerate(available_gaps_regions):
            region_data = gaps_df[gaps_df['Region'] == region]['Gaps']
            mean_val = region_data.mean()
            line_width = args.mean_line_width
            plt.hlines(y=mean_val, xmin=i-line_width, xmax=i+line_width, 
                      colors='black', linestyles='dashed', linewidth=1)

        plt.xlabel('Region')
        plt.ylabel('Gaps [count]')
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/{args.save_prefix}_alignment_gaps_violin_plot.png',
                    dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
        plt.close()

    # Region lengths - Violin Plots
    length_data = []
    available_length_regions = []

    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']:
        gen_len_col = f'{region}_gen_length'
        true_len_col = f'{region}_true_length'
        
        if gen_len_col in df.columns and true_len_col in df.columns:
            available_length_regions.append(region)
            for idx, row in df.iterrows():
                if not pd.isna(row.get(gen_len_col)):
                    length_data.append({
                        'Region': region,
                        'Length': row[gen_len_col],
                        'Type': 'Generated'
                    })
                if not pd.isna(row.get(true_len_col)):
                    length_data.append({
                        'Region': region,
                        'Length': row[true_len_col],
                        'Type': 'True'
                    })
    
    if length_data:
        length_df = pd.DataFrame(length_data)
        length_region_colors = get_region_colors(available_length_regions, args.color_palette)
        
        # Create two separate violin plots - one for Generated and one for True
        for seq_type in ['Generated', 'True']:
            type_data = length_df[length_df['Type'] == seq_type]
            colors_list = [length_region_colors[region] for region in available_length_regions]
            
            plt.figure(figsize=(args.figure_width, args.figure_height))
            ax = sns.violinplot(data=type_data, x='Region', y='Length', 
                               palette=colors_list, inner=None, order=available_length_regions)

            # Add mean lines to the violin plots
            for i, region in enumerate(available_length_regions):
                region_data = type_data[type_data['Region'] == region]['Length']
                if len(region_data) > 0:
                    mean_val = region_data.mean()
                    line_width = args.mean_line_width
                    plt.hlines(y=mean_val, xmin=i-line_width, xmax=i+line_width, 
                              colors='black', linestyles='dashed', linewidth=1)

            plt.xlabel('Region')
            plt.ylabel('Length [AA]')
            sns.despine()
            plt.tight_layout()
            plt.savefig(f'{args.output_dir}/{args.save_prefix}_{seq_type.lower()}_region_length_violin_plot.png',
                        dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
            plt.close()
        
        # Create a combined plot with both types side by side
        plt.figure(figsize=(args.figure_width * 1.7, args.figure_height))
        
        ax = sns.violinplot(data=length_df, x='Region', y='Length', hue='Type',
                           palette={"Generated": sns.color_palette(args.color_palette)[0], 
                                   "True": sns.color_palette(args.color_palette)[1]}, 
                           inner=None, order=available_length_regions, split=True)

        # Add mean lines to the violin plots for each type
        for i, region in enumerate(available_length_regions):
            for seq_type, color in zip(['Generated', 'True'], 
                                     [sns.color_palette(args.color_palette)[0], 
                                      sns.color_palette(args.color_palette)[1]]):
                region_type_data = length_df[(length_df['Region'] == region) & (length_df['Type'] == seq_type)]['Length']
                if len(region_type_data) > 0:
                    mean_val = region_type_data.mean()
                    offset = -0.2 if seq_type == 'Generated' else 0.2
                    line_width = args.mean_line_width * 0.5
                    plt.hlines(y=mean_val, xmin=i-line_width+offset, xmax=i+line_width+offset, 
                              colors='black', linestyles='dashed', linewidth=1)

        plt.xlabel('Region')
        plt.ylabel('Length [AA]')
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/{args.save_prefix}_combined_region_length_violin_plot.png',
                    dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
        plt.close()
        
    # Correlation between percent identity and gaps
    plt.figure(figsize=(args.figure_width * 2, args.figure_height * 2))
    corr_data = []
    
    for region in available_regions:
        id_col = f'{region}_percent_identity'
        gaps_col = f'{region}_gaps'
        
        if id_col in df.columns and gaps_col in df.columns:
            valid_data = df[[id_col, gaps_col]].dropna()
            if not valid_data.empty:
                corr, p_value = stats.pearsonr(valid_data[id_col], valid_data[gaps_col])
                corr_data.append({
                    'Region': region,
                    'Correlation': corr,
                    'P-value': p_value,
                    'Significant': p_value < 0.05
                })
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        sns.barplot(data=corr_df, x='Region', y='Correlation', 
                    hue='Significant', palette={True: 'darkgreen', False: 'lightgray'})
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Region')
        plt.ylabel('Pearson Correlation Coefficient')
        plt.ylim(-1, 1)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/{args.save_prefix}_correlation_analysis.png',
                    dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
        plt.close()
    
    # Combined plot for QQ-plots of percent identity
    rows = int(np.ceil(len(available_regions) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(args.figure_width * 2, args.figure_height * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, region in enumerate(available_regions):
        if i < len(axes):
            col_name = f'{region}_percent_identity'
            if col_name in df.columns:
                stats.probplot(df[col_name].dropna(), plot=axes[i])
                axes[i].set_title(f'QQ Plot for {region} Percent Identity')
                axes[i].grid(False)
    
    # Hide unused subplots
    for j in range(len(available_regions), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{args.save_prefix}_percent_identity_qq_plots.png',
                dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
    plt.close()
    
    print("Visualizations saved to multiple PNG files")

def analyze_distribution_statistics(df, args):
    """
    Analyze the distribution statistics for percent identity and gaps values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the extracted sequence data
    args : argparse.Namespace
        Command line arguments
    """
    print("\n===== Distribution Statistics =====")
    
    # Get available regions
    available_regions = []
    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']:
        if f'{region}_percent_identity' in df.columns and not df[f'{region}_percent_identity'].isna().all():
            available_regions.append(region)
    
    # Create table for percent identity statistics
    stats_data = []
    for region in available_regions:
        col_name = f'{region}_percent_identity'
        data = df[col_name].dropna()
        
        if len(data) > 0:
            basic_stats_percent_identity = {
                'Region': region,
                'Mean': data.mean(),
                'Median': data.median(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'IQR': data.quantile(0.75) - data.quantile(0.25),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            }
            
            # Test for normality if we have enough data
            if len(data) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                basic_stats_percent_identity['Shapiro-Wilk p'] = shapiro_p
                basic_stats_percent_identity['Normal Distribution'] = 'Yes' if shapiro_p > 0.05 else 'No'
            else:
                basic_stats_percent_identity['Shapiro-Wilk p'] = float('nan')
                basic_stats_percent_identity['Normal Distribution'] = 'Unknown (insufficient data)'
            
            stats_data.append(basic_stats_percent_identity)
    
    # Create and display the DataFrame
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        print("\nPercent Identity Distribution Statistics:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        
        # Save to CSV
        stats_df.to_csv(f'{args.output_dir}/{args.save_prefix}_percent_identity_statistics.csv', index=False)
        print(f"Statistics saved to {args.output_dir}/{args.save_prefix}_percent_identity_statistics.csv")
    
    # Repeat for gaps data
    gaps_stats_data = []
    for region in available_regions:
        gaps_col = f'{region}_gaps'
        if gaps_col in df.columns:
            data = df[gaps_col].dropna()
            
            if len(data) > 0:
                basic_stats_gaps = {
                    'Region': region,
                    'Mean Gaps': data.mean(),
                    'Median Gaps': data.median(),
                    'Std Dev Gaps': data.std(),
                    'Min Gaps': data.min(),
                    'Max Gaps': data.max(),
                    'IQR Gaps': data.quantile(0.75) - data.quantile(0.25),
                    'Skewness Gaps': stats.skew(data),
                    'Kurtosis Gaps': stats.kurtosis(data)
                }
                
                # Test for normality if we have enough data
                if len(data) >= 3:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                    basic_stats_gaps['Shapiro-Wilk p'] = shapiro_p
                    basic_stats_gaps['Normal Distribution'] = 'Yes' if shapiro_p > 0.05 else 'No'
                else:
                    basic_stats_gaps['Shapiro-Wilk p'] = float('nan')
                    basic_stats_gaps['Normal Distribution'] = 'Unknown (insufficient data)'
                
                gaps_stats_data.append(basic_stats_gaps)
    
    # Create and display the DataFrame for gaps
    if gaps_stats_data:
        gaps_stats_df = pd.DataFrame(gaps_stats_data)
        print("\nGaps Distribution Statistics:")
        print(gaps_stats_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        
        # Save to CSV
        gaps_stats_df.to_csv(f'{args.output_dir}/{args.save_prefix}_gaps_statistics.csv', index=False)
        print(f"Statistics saved to {args.output_dir}/{args.save_prefix}_gaps_statistics.csv")

    
        # Repeat for length data
    length_stats_data = []
    for region in available_regions:
        length_col = f'{region}_length'
        if length_col in df.columns:
            data = df[length_col].dropna()
            
            if len(data) > 0:
                # Basic statistics
                basic_stats_lengths = {
                    'Region': region,
                    'Mean Length': data.mean(),
                    'Median Length': data.median(),
                    'Std Dev Length': data.std(),
                    'Min Length': data.min(),
                    'Max Length': data.max(),
                    'IQR Length': data.quantile(0.75) - data.quantile(0.25),
                    'Skewness Length': stats.skew(data),
                    'Kurtosis Length': stats.kurtosis(data)
                }
                
                # Test for normality if we have enough data
                if len(data) >= 3:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                    basic_stats_lengths['Shapiro-Wilk p'] = shapiro_p
                    basic_stats_lengths['Normal Distribution'] = 'Yes' if shapiro_p > 0.05 else 'No'
                else:
                    basic_stats_lengths['Shapiro-Wilk p'] = float('nan')
                    basic_stats_lengths['Normal Distribution'] = 'Unknown (insufficient data)'
                
                length_stats_data.append(basic_stats_lengths)
    
    # Create and display the DataFrame for length
    if length_stats_data:
        length_stats_df = pd.DataFrame(length_stats_data)
        print("\nRegion Length Distribution Statistics:")
        print(length_stats_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        
        # Save to CSV
        length_stats_df.to_csv(f'{args.output_dir}/{args.save_prefix}_region_length_statistics.csv', index=False)
        print(f"Statistics saved to {args.output_dir}/{args.save_prefix}_region_length_statistics.csv")

        

def compare_gen_true_lengths(df, args):
    """
    Compare the lengths of generated and true sequences for each region.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the sequence comparison data
    args : argparse.Namespace
        Command line arguments
    """
    print("\n===== Generated vs True Sequence Length Comparison =====")
    
    regions = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']
    
    for region in regions:
        gen_len_col = f'{region}_gen_length'
        true_len_col = f'{region}_true_length'
        
        if gen_len_col in df.columns and true_len_col in df.columns:
            gen_lengths = df[gen_len_col].dropna()
            true_lengths = df[true_len_col].dropna()
            
            if len(gen_lengths) > 0 and len(true_lengths) > 0:
                mean_gen = gen_lengths.mean()
                mean_true = true_lengths.mean()
                std_gen = gen_lengths.std()
                std_true = true_lengths.std()
                
                print(f"\n{region} Region:")
                print(f"  Generated sequences: mean length = {mean_gen:.2f} ± {std_gen:.2f}")
                print(f"  True sequences: mean length = {mean_true:.2f} ± {std_true:.2f}")
                print(f"  Length difference (generated - true): {mean_gen - mean_true:.2f}")
                
                # Perform t-test to check if lengths are significantly different
                if len(gen_lengths) > 1 and len(true_lengths) > 1:
                    t_stat, p_value = stats.ttest_ind(gen_lengths, true_lengths, equal_var=False)
                    print(f"  t-test p-value: {p_value:.4f} (lengths are {'significantly different' if p_value < 0.05 else 'not significantly different'})")
                
                # Create a violin plot to visualize the length distributions
                plt.figure(figsize=(args.figure_width, args.figure_height))
                
                # Create a dataframe for the violin plot
                length_data = []
                for gen_len, true_len in zip(gen_lengths, true_lengths):
                    length_data.append({'Type': 'Generated', 'Length': gen_len, 'Region': region})
                    length_data.append({'Type': 'True', 'Length': true_len, 'Region': region})
                
                length_df = pd.DataFrame(length_data)
                
                type_colors = [sns.color_palette(args.color_palette)[0], sns.color_palette(args.color_palette)[1]]
                ax = sns.violinplot(data=length_df, x='Type', y='Length', palette=type_colors, inner=None)
                
                # Add mean lines
                for i, seq_type in enumerate(['Generated', 'True']):
                    type_data = length_df[length_df['Type'] == seq_type]['Length']
                    mean_val = type_data.mean()
                    line_width = args.mean_line_width
                    plt.hlines(y=mean_val, xmin=i-line_width, xmax=i+line_width, 
                              colors='black', linestyles='dashed', linewidth=1)
                
                plt.ylabel('Length (AA)')
                sns.despine()
                plt.tight_layout()
                plt.savefig(f'{args.output_dir}/{args.save_prefix}_{region}_length_comparison.png',
                           dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
                plt.close()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create test dataset if requested
    input_file = args.input_file
    if args.test_entries:
        input_file = create_test_dataset(args.input_file, args.test_entries)
    
    try:
        # Analyze the sequence data
        print("Starting sequence analysis...")
        df, sequences = analyze_sequence_data(input_file)
        
        try:
            # Analyze distribution statistics
            print("\nAnalyzing distribution statistics...")
            analyze_distribution_statistics(df, args)
        except Exception as e:
            print(f"Error in distribution statistics: {e}")
        
        try:
            # Generate visualizations
            print("\nGenerating visualizations...")
            generate_visualizations(df, args)
        except Exception as e:
            print(f"Error in visualizations: {e}")
        
        try:
            # Compare generated and true sequence lengths
            print("\nComparing generated and true sequence lengths...")
            compare_gen_true_lengths(df, args)
        except Exception as e:
            print(f"Error in length comparison: {e}")
        
        print(f"\nAnalysis complete! Check '{args.output_dir}' for generated files.")
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found. Please provide the correct file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()