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
    no_hits_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            try:
                seq_data = json.loads(line.strip())
                
                # Check if this is a "No hits found" sequence
                if "Message" in seq_data and "No hits found" in seq_data["Message"]:
                    no_hits_count += 1
                    # We'll still keep it in sequences for counting, but mark it
                    seq_data["_no_hits"] = True
                else:
                    seq_data["_no_hits"] = False
                
                sequences.append(seq_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Successfully loaded {len(sequences)} sequences")
    print(f"Found {no_hits_count} sequences with 'No hits found' ({no_hits_count/len(sequences)*100:.2f}%)")
    
    # Extract key metrics into a DataFrame for analysis (exclude "no hits" sequences)
    data = []
    for seq in sequences:
        # Skip "no hits" sequences for the detailed analysis
        if seq.get("_no_hits", False):
            continue
            
        entry = {
            'Sequence_ID': seq.get('Sequence ID', ''),
            'Length': float(seq.get('Sequence Length', 0)),  # Convert to float
            'Domain_Classification': seq.get('Domain Classification', ''),
        }
        
        # Add percent identity for each region, including CDR3 if it exists
        regions = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'Total']
        for region in regions:
            if region in seq:
                # Make sure all numeric values are properly converted to float
                # Handle 'N/A' values
                for field in ['percent identity', 'length', 'matches', 'mismatches', 'gaps']:
                    value = seq[region].get(field, 0)
                    if value == 'N/A':
                        entry[f'{region}_{field.replace(" ", "_")}'] = 0.0  # Replace 'N/A' with 0.0
                    else:
                        try:
                            entry[f'{region}_{field.replace(" ", "_")}'] = float(value)
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {field} value '{value}' to float in {region}. Using 0.0")
                            entry[f'{region}_{field.replace(" ", "_")}'] = 0.0
        
        # Add top hit information
        if 'Hits' in seq and len(seq['Hits']) > 0:
            entry['Top_hit_gene'] = seq['Hits'][0].get('gene', '')
            # Convert numeric values to float
            entry['Top_hit_score'] = float(seq['Hits'][0].get('bit_score', 0))
            # Handle scientific notation in e_value
            try:
                entry['Top_hit_evalue'] = float(seq['Hits'][0].get('e_value', 0))
            except ValueError:
                # If conversion fails, try to handle scientific notation like '2e-52'
                e_value = seq['Hits'][0].get('e_value', 0)
                if isinstance(e_value, str) and 'e' in e_value.lower():
                    try:
                        entry['Top_hit_evalue'] = float(e_value)
                    except:
                        entry['Top_hit_evalue'] = 0
                else:
                    entry['Top_hit_evalue'] = 0
        
        data.append(entry)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Store the no_hits_count as metadata
    df._metadata = {'no_hits_count': no_hits_count, 'total_sequences': len(sequences)}
    
    # Basic statistics
    print("\n===== Basic Statistics =====")
    print(f"Total number of sequences: {len(sequences)}")
    print(f"Sequences with hits: {len(df)} ({len(df)/len(sequences)*100:.2f}%)")
    print(f"Sequences without hits: {no_hits_count} ({no_hits_count/len(sequences)*100:.2f}%)")

    

    
    if len(df) > 0:  # Only calculate stats if we have sequences with hits        
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
                try:
                    mean_percent_id = df[percent_id_col].mean()
                    #std_percent_id = df[percent_id_col].std()
                    print(f"{region}:")
                    print(f"  Average percent identity: {mean_percent_id:.2f}%") # ± {std_percent_id:.2f}%
                    
                    if length_col in df.columns and not df[length_col].isna().all():
                        mean_length = df[length_col].mean()
                        #std_length = df[length_col].std()
                        print(f"  Average length: {mean_length:.2f}") # ± {std_length:.2f}
                except Exception as e:
                    print(f"  Error calculating statistics for {region}: {e}")
                    print(f"  Data types in column: {df[percent_id_col].apply(type).value_counts()}")
                    # Try to fix the column by converting to float
                    try:
                        df[percent_id_col] = df[percent_id_col].astype(float)
                        print(f"  After conversion - Average percent identity: {df[percent_id_col].mean():.2f}%")
                    except:
                        pass
        
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
        if len(sequences) > 0:
            # Find first sequence with alignments (skipping no-hit sequences)
            sample_seq = None
            for seq in sequences:
                if not seq.get("_no_hits", False) and 'Alignments' in seq:
                    sample_seq = seq
                    break
                    
            if sample_seq:
                print("\n===== Sample Sequence Alignment =====")
                seq_id = sample_seq.get('Sequence ID', 'Unknown')
                print(f"Sequence ID: {seq_id}")
                
                if 'Alignments' in sample_seq and 'strings' in sample_seq['Alignments'] and 'keys' in sample_seq['Alignments']:
                    alignments = sample_seq['Alignments']
                    strings = alignments['strings']
                    keys = alignments['keys']
                    
                    print("\nAlignment visualization:")
                    for i, (key, string) in enumerate(zip(keys, strings)):
                        print(f"{key:<12} {string}")
    else:
        print("No sequences with hits found. Cannot calculate statistics.")
    
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
    # Only generate visualizations if we have data
    if len(df) == 0:
        print("No sequences with hits to visualize.")
        return
        
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
        col_name = f'{region}_percent_identity'
        if col_name in df.columns and not df[col_name].isna().all():
            # Make sure all values are numeric
            try:
                df[col_name] = df[col_name].astype(float)
                available_regions.append(region)
            except:
                print(f"Warning: Could not convert {col_name} to numeric. Skipping in visualization.")
    
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
        col_name = f'{region}_length'
        if col_name in df.columns and not df[col_name].isna().all():
            # Make sure all values are numeric
            try:
                df[col_name] = df[col_name].astype(float)
                available_regions.append(region)
            except:
                print(f"Warning: Could not convert {col_name} to numeric. Skipping in visualization.")
    
    region_length_cols = [f'{region}_length' for region in available_regions]
    
    if region_length_cols:
        df[region_length_cols].mean().plot(kind='bar', ax=ax)
        ax.set_title('Average Length by Region')
        ax.set_xlabel('Region')
        ax.set_ylabel('Length (nt)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('sequence_analysis_test_check_bert.png')
    print("Visualizations saved to 'sequence_analysis.png'")
    
    # Additional visualization: No hits vs. Hits pie chart
    total_sequences = len(df) + df._metadata.get('no_hits_count', 0)
    if df._metadata.get('no_hits_count', 0) > 0:
        plt.figure(figsize=(8, 8))
        labels = ['Sequences with hits', 'Sequences with no hits']
        sizes = [len(df), df._metadata.get('no_hits_count', 0)]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Distribution of Sequences with/without Hits')
        plt.savefig('hits_distribution_bert2gpt.png')
        print("Hit distribution visualization saved to 'hits_distribution.png'")

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
        # Skip sequences with no hits
        if seq.get("_no_hits", False):
            continue
            
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
    # Update the file path to your actual file location
    #json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/pyir_output/generated_sequences_bert_small_10000.json"
    #json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/pyir_output/bert2gpt_gen_seqs.json"

    json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/pyir_output/small_test_bert_small.json"
    
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
        
        print("\nAnalysis complete!")
        print("- Check 'sequence_analysis.png' for region and gene visualizations")
        print("- Check 'hits_distribution.png' for distribution of sequences with/without hits")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found. Please provide the correct file path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()