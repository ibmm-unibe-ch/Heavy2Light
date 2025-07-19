import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

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
            #std_percent_id = df[percent_id_col].std()
            mean_length = df[length_col].mean() if length_col in df.columns else 0
            #std_length = df[length_col].std() if length_col in df.columns else 0
            
            print(f"{region}:")
            print(f"  Average percent identity: {mean_percent_id:.2f}%")
            if length_col in df.columns:
                print(f"  Average length: {mean_length:.2f}")
    
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

def generate_visualizations(df):
    """
    Generate visualizations from the sequence data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the extracted sequence data
    """
    # Set up the figure layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sequence Length Distribution
    if 'Length' in df.columns:
        ax = axes[0, 0]
        df['Length'].hist(bins=20, ax=ax)
        ax.set_title('Sequence Length Distribution')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Count')
    
    # 2. Percent Identity by Region including CDR3 and Total
    ax = axes[0, 1]
    # Filter regions that exist in the data
    available_regions = []
    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']:
        if f'{region}_percent_identity' in df.columns and not df[f'{region}_percent_identity'].isna().all():
            available_regions.append(region)
    
    percent_id_cols = [f'{region}_percent_identity' for region in available_regions]
    
    if percent_id_cols:
        df[percent_id_cols].mean().plot(kind='bar', ax=ax)
        ax.set_title('Average Percent Identity by Region')
        ax.set_xlabel('Region')
        ax.set_ylabel('Percent Identity')
        ax.set_ylim(0, 100)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Top Hit Genes
    if 'Top_hit_gene' in df.columns:
        ax = axes[1, 0]
        df['Top_hit_gene'].value_counts().head(10).plot(kind='barh', ax=ax)
        ax.set_title('Top 10 Hit Genes')
        ax.set_xlabel('Count')
    
    # 4. Region lengths comparison
    ax = axes[1, 1]
    # Filter available length columns
    available_regions = []
    for region in ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3']:
        if f'{region}_length' in df.columns and not df[f'{region}_length'].isna().all():
            available_regions.append(region)
    
    region_length_cols = [f'{region}_length' for region in available_regions]
    
    if region_length_cols:
        df[region_length_cols].mean().plot(kind='bar', ax=ax)
        ax.set_title('Average Length by Region')
        ax.set_xlabel('Region')
        ax.set_ylabel('Length (nt)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('sequence_analysis-5.png')
    print("Visualizations saved to 'sequence_analysis.png'")

def analyze_amino_acid_composition(sequences):
    """
    Analyze amino acid composition across sequences.
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
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


if __name__ == "__main__":
    #json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/pyir_output/all_seqs_generated_gpt2_sequences_10000.json"
    #json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/train_val_test_fastas/plabdab_human_healthy_no_vac_allocated_test_no_identifiers.json"
    json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/matching_seqs_multiple_light_seqs_203276_cls_predictions.json"

    #output_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/train_val_test_datasets_pyir/plabdab_human_healthy_no_vac_allocated_test_no_identifiers"
    output_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/plots/full_eval_generate_multiple_light_seqs_203276_cls_predictions_merged_genes"

    # Ensure the output directory exists
    import os
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)    
    
    try:
        # Analyze the sequence data
        print("Starting sequence analysis...")
        df, sequences = analyze_sequence_data(json_file_path)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        generate_visualizations(df)
        
        # Analyze amino acid composition
        print("\nAnalyzing amino acid composition...")
        analyze_amino_acid_composition(sequences)
        
        print("\nAnalysis complete! Check 'sequence_analysis.png' for visualizations.")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found. Please provide the correct file path.")
    except Exception as e:
        print(f"Error: {e}")

