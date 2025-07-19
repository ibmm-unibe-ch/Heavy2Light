import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import os

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze sequence data from JSON file')
    
    # Required arguments
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output files')
    
    # Optional arguments
    parser.add_argument('--save_prefix', type=str, default='sequence_analysis',
                        help='Prefix for saved files (default: sequence_analysis)')
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
    
    # Y-axis limit arguments
    parser.add_argument('--percent_identity_ylim', type=float, nargs=2, default=[0, 100],
                        metavar=('MIN', 'MAX'),
                        help='Y-axis limits for percent identity violin plot (default: 0 100)')
    parser.add_argument('--length_ylim', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Y-axis limits for region length violin plot (default: auto)')
    
    return parser.parse_args()

def get_region_colors(regions, palette_name):
    """Get consistent colors for regions across all plots"""
    # Use seaborn color palette
    colors = sns.color_palette(palette_name, n_colors=len(regions))
    return dict(zip(regions, colors))

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
    
    # Extract key metrics into a DataFrame for analysis
    data = []
    for seq in sequences:
        entry = {
            'Sequence_ID': seq.get('Sequence ID', ''),
            'Length': seq.get('Sequence Length', 0),
            'Domain_Classification': seq.get('Domain Classification', ''),
        }
        
        # Add percent identity for each region, including CDR3 if it exists
        regions = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']
        for region in regions:
            if region in seq:
                entry[f'{region}_percent_identity'] = seq[region].get('percent identity', 0)
                entry[f'{region}_length'] = seq[region].get('length', 0)
                entry[f'{region}_matches'] = seq[region].get('matches', 0)
                entry[f'{region}_mismatches'] = seq[region].get('mismatches', 0)
                entry[f'{region}_gaps'] = seq[region].get('gaps', 0)
        
        # Add top hit information
        if 'Hits' in seq and len(seq['Hits']) > 0:
            entry['Top_hit_gene'] = seq['Hits'][0].get('gene', '')
            entry['Top_hit_score'] = seq['Hits'][0].get('bit_score', 0)
            entry['Top_hit_evalue'] = seq['Hits'][0].get('e_value', 0)
        
        data.append(entry)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Basic statistics
    print("\n===== Basic Statistics =====")
    print(f"Total number of sequences: {len(df)}")
    print(f"Average sequence length: {df['Length'].mean():.2f}")
    
    # Domain classification distribution
    print("\n===== Domain Classification =====")
    domain_counts = df['Domain_Classification'].value_counts()
    for domain, count in domain_counts.items():
        print(f"{domain}: {count} ({count/len(df)*100:.2f}%)")
    
    # Region statistics
    print("\n===== Region Statistics =====")
    regions = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']
    for region in regions:
        percent_id_col = f'{region}_percent_identity'
        length_col = f'{region}_length'
        
        if percent_id_col in df.columns and not df[percent_id_col].isna().all():
            mean_percent_id = df[percent_id_col].mean()
            std_percent_id = df[percent_id_col].std()
            median_percent_id = df[percent_id_col].median()
            mean_length = df[length_col].mean() if length_col in df.columns else 0
            std_length = df[length_col].std() if length_col in df.columns else 0
            
            print(f"{region}:")
            print(f"  Average percent identity: {mean_percent_id:.2f}%")
            print(f"  Standard deviation of percent identity: {std_percent_id:.2f}%")
            print(f"  Median percent identity: {median_percent_id:.2f}%")
            if length_col in df.columns:
                print(f"  Average length: {mean_length:.2f}")
                print(f"  Standard deviation of length: {std_length:.2f}")
            
            # Check distribution normality using Shapiro-Wilk test
            if len(df[percent_id_col].dropna()) >= 3:  # Need at least 3 samples for Shapiro-Wilk
                stat, p_value = stats.shapiro(df[percent_id_col].dropna())
                print(f"  Shapiro-Wilk test for normality (percent identity): p-value = {p_value:.4f}")
                print(f"  Distribution is {'likely normal' if p_value > 0.05 else 'not normal'}")
    
    # Print detailed information about the full sequence
    if 'Total_percent_identity' in df.columns:
        print("\n===== Full Sequence Statistics =====")
        print(f"Average percent identity across full sequence: {df['Total_percent_identity'].mean():.2f}%")
    
    # Top hit gene distribution
    if 'Top_hit_gene' in df.columns:
        print("\n===== Top Hit Gene Distribution =====")
        top_genes = df['Top_hit_gene'].value_counts().head(10)
        for gene, count in top_genes.items():
            print(f"{gene}: {count} ({count/len(df)*100:.2f}%)")
    
    # E-value distribution
    if 'Top_hit_evalue' in df.columns:
        print("\n===== E-value Distribution =====")
        print(f"Minimum E-value: {df['Top_hit_evalue'].min()}")
        print(f"Maximum E-value: {df['Top_hit_evalue'].max()}")
        print(f"Median E-value: {df['Top_hit_evalue'].median()}")
    
    # Sample sequence alignment analysis from the existing alignment data
    if len(sequences) > 0 and 'Alignments' in sequences[0]:
        print("\n===== Sample Sequence Alignment =====")
        sample_seq = sequences[0]
        seq_id = sample_seq.get('Sequence ID', 'Unknown')
        print(f"Sequence ID: {seq_id}")
        
        if 'Alignments' in sample_seq and 'strings' in sample_seq['Alignments'] and 'keys' in sample_seq['Alignments']:
            alignments = sample_seq['Alignments']
            strings = alignments['strings']
            keys = alignments['keys']
            
            print("\nAlignment visualization:")
            for i, (key, string) in enumerate(zip(keys, strings)):
                print(f"{key:<12} {string}")
    
    # Return the dataframe for further analysis if needed
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
    sns.histplot(data=df, x='Length', kde=True, color=sns.color_palette(args.color_palette)[0])
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
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
        
        # Create color list in the order of regions
        colors_list = [region_colors[region] for region in available_regions]
        
        ax = sns.violinplot(data=percent_identity_df, x='Region', y='Percent Identity', 
                            palette=colors_list, inner=None, order=available_regions)
        

        # from matplotlib.ticker import MaxNLocator
        # plt.xaxis.set_major_locator(MaxNLocator(nbins=args.xaxis_bins))
        # plt.yaxis.set_major_locator(MaxNLocator(nbins=args.yaxis_bins))
   

        # Add mean lines to the violin plots (shorter lines)
        for i, region in enumerate(available_regions):
            region_data = percent_identity_df[percent_identity_df['Region'] == region]['Percent Identity']
            mean_val = region_data.mean()
            line_width = args.mean_line_width
            plt.hlines(y=mean_val, xmin=i-line_width, xmax=i+line_width, 
                      colors='black', linestyles='dashed', linewidth=1)

        plt.xlabel('Region')
        plt.ylabel('Percent Identity [%]')

        # Set y-axis limits based on command line argument
        plt.ylim(args.percent_identity_ylim[0], args.percent_identity_ylim[1])
        
        plt.ylim(0, 100)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/{args.save_prefix}_percent_identity_violin_plot.png', 
                    dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
        plt.close()
    
    # Third figure: Region lengths - Violin Plots
    length_data = []
    available_length_regions = []

    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']:
        col_name = f'{region}_length'
        if col_name in df.columns and not df[col_name].isna().all():
            available_length_regions.append(region)
            for idx, value in df[col_name].items():
                length_data.append({
                    'Region': region,
                    'Length': value
                })
    
    if length_data:
        length_df = pd.DataFrame(length_data)
        
        # Get colors for length regions (use same color mapping)
        length_region_colors = get_region_colors(available_length_regions, args.color_palette)
        colors_list = [length_region_colors[region] for region in available_length_regions]
        
        plt.figure(figsize=(args.figure_width, args.figure_height))

        ax = sns.violinplot(data=length_df, x='Region', y='Length', 
                            palette=colors_list, inner=None, order=available_length_regions)
        

        # Add mean lines to the violin plots (shorter lines)
        for i, region in enumerate(available_length_regions):
            region_data = length_df[length_df['Region'] == region]['Length']
            mean_val = region_data.mean()
            line_width = args.mean_line_width
            plt.hlines(y=mean_val, xmin=i-line_width, xmax=i+line_width, 
                      colors='black', linestyles='dashed', linewidth=1)

        plt.xlabel('Region')
        plt.ylabel('Length [AA]')

        # Set y-axis limits if provided
        if args.length_ylim is not None:
            plt.ylim(args.length_ylim[0], args.length_ylim[1])

        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/{args.save_prefix}_region_length_violin_plot.png', 
                    dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
        plt.close()
        
    # Fourth figure: Top Hit Genes
    if 'Top_hit_gene' in df.columns:
        plt.figure(figsize=(args.figure_width * 2, args.figure_height * 2))
        top_genes = df['Top_hit_gene'].value_counts().head(10)
        sns.barplot(x=top_genes.values, y=top_genes.index, 
                    color=sns.color_palette(args.color_palette)[0])
        plt.xlabel('Count')
        plt.ylabel('Gene')
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/{args.save_prefix}_top_hit_genes.png', 
                    dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
        plt.close()
    
    # Fifth figure: Correlation between percent identity and length
    plt.figure(figsize=(args.figure_width * 2, args.figure_height * 2))
    corr_data = []
    
    for region in available_regions:
        id_col = f'{region}_percent_identity'
        len_col = f'{region}_length'
        
        if id_col in df.columns and len_col in df.columns:
            valid_data = df[[id_col, len_col]].dropna()
            if not valid_data.empty:
                corr, p_value = stats.pearsonr(valid_data[id_col], valid_data[len_col])
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
                # Remove grid from individual subplots
                axes[i].grid(False)
    
    # Hide unused subplots
    for j in range(len(available_regions), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{args.save_prefix}_percent_identity_qq_plots.png', 
                dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
    plt.close()
    
    print("Visualizations saved to multiple PNG files")

def analyze_amino_acid_composition(sequences, args):
    """
    Analyze amino acid composition across sequences.
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
    args : argparse.Namespace
        Command line arguments
    """
    # Extract amino acid sequences from each region
    region_aa_seqs = defaultdict(list)
    
    for seq in sequences:
        # Check available regions that have AA data
        for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3']:
            if region in seq and 'AA' in seq[region] and seq[region]['AA']:
                region_aa_seqs[region].append(seq[region]['AA'])
    
    # Analyze composition for each region
    print("\n===== Amino Acid Composition =====")
    for region, aa_seqs in region_aa_seqs.items():
        if not aa_seqs:  # Skip empty regions
            continue
            
        print(f"\n{region} Region:")
        
        # Combine all sequences for this region
        combined = ''.join(aa_seqs)
        counter = Counter(combined)
        total_aa = len(combined)
        
        if total_aa == 0:
            print(f"No amino acid data available for {region}")
            continue
            
        # Print the most common amino acids
        print(f"Total amino acids: {total_aa}")
        print("Most common amino acids:")
        for aa, count in counter.most_common(5):
            print(f"  {aa}: {count} ({count/total_aa*100:.2f}%)")
    
    # Generate amino acid composition visualization
    plot_amino_acid_composition(region_aa_seqs, args)

def plot_amino_acid_composition(region_aa_seqs, args):
    """
    Create visualizations for amino acid composition.
    
    Parameters:
    -----------
    region_aa_seqs : dict
        Dictionary with region names as keys and lists of amino acid sequences as values
    args : argparse.Namespace
        Command line arguments
    """
    # Standard amino acids to ensure consistent ordering
    standard_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Prepare data for visualization
    region_aa_data = {}
    
    for region, aa_seqs in region_aa_seqs.items():
        if not aa_seqs:  # Skip empty regions
            continue
            
        # Combine all sequences for this region
        combined = ''.join(aa_seqs)
        counter = Counter(combined)
        total_aa = len(combined)
        
        if total_aa == 0:
            continue
            
        # Calculate percentages for all standard amino acids
        percentages = {aa: (counter.get(aa, 0) / total_aa * 100) for aa in standard_aa}
        region_aa_data[region] = percentages
    
    if not region_aa_data:  # No data to visualize
        return
    
    # Create heatmap of amino acid compositions
    data = []
    for region, aa_percentages in region_aa_data.items():
        for aa in standard_aa:
            data.append({
                'Region': region,
                'Amino Acid': aa,
                'Percentage': aa_percentages.get(aa, 0)
            })
    
    df_aa = pd.DataFrame(data)
    pivot_table = df_aa.pivot(index='Region', columns='Amino Acid', values='Percentage')
    
    plt.figure(figsize=(args.figure_width * 2.3, args.figure_height * 2))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.1f', 
                cbar_kws={'label': 'Percentage (%)'})
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{args.save_prefix}_amino_acid_composition.png', 
                dpi=args.dpi, bbox_inches='tight', transparent=args.transparent)
    plt.close()

def analyze_distribution_statistics(df, args):
    """
    Analyze the distribution statistics for percent identity and length values.
    
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
            # Basic statistics
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

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Analyze the sequence data
        print("Starting sequence analysis...")
        df, sequences = analyze_sequence_data(args.input_file)
        
        # Analyze distribution statistics
        print("\nAnalyzing distribution statistics...")
        analyze_distribution_statistics(df, args)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        generate_visualizations(df, args)
        
        # Analyze amino acid composition
        print("\nAnalyzing amino acid composition...")
        analyze_amino_acid_composition(sequences, args)
        
        print(f"\nAnalysis complete! Check '{args.output_dir}' for generated files.")
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found. Please provide the correct file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     #json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/all_seqs_generated_gpt2_sequences_10000.json"
#     #json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/pyir_output/bert2gpt_full_complete_ids.json"

#     #output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/lightgpt_violin_plots_and_distributions"
#     #output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/bert2gpt_violin_plots_and_distributions"

#     json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/train_val_test_fastas/plabdab_human_healthy_no_vac_allocated_test_no_identifiers.json"

#     output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/train_val_test_datasets_pyir/plabdab_human_healthy_no_vac_allocated_test_no_identifiers"
    

#     # Create output directory if it doesn't exist
#     import os
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     save_prefix = "bert2gpt_mean_more_dense"
    
#     try:
#         # Analyze the sequence data
#         print("Starting sequence analysis...")
#         df, sequences = analyze_sequence_data(json_file_path)
        
#         # Analyze distribution statistics
#         print("\nAnalyzing distribution statistics...")
#         analyze_distribution_statistics(df)
        
#         # Generate visualizations
#         print("\nGenerating visualizations...")
#         generate_visualizations(df)
        
#         # Analyze amino acid composition
#         print("\nAnalyzing amino acid composition...")
#         analyze_amino_acid_composition(sequences)
        
#         print("\nAnalysis complete! Check the generated PNG files for visualizations.")
        
#     # except FileNotFoundError:
#     #     print(f"Error: File '{json_file_path}' not found. Please provide the correct file path.")
#     except Exception as e:
#         print(f"Error: {e}")