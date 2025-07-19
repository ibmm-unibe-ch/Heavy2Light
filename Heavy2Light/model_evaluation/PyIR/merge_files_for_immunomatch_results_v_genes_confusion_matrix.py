import pandas as pd
import numpy as np

def merge_pairing_vgene_files(pairing_file_path: str, vgene_file_path: str, output_path: str):
    """
    Merge pairing scores CSV with V gene calls CSV based on common sequence columns.
    
    Args:
        pairing_file_path: Path to CSV with pairing_scores column
        vgene_file_path: Path to CSV with V gene calls
        output_path: Path for the merged output file
    """
    
    print("Loading CSV files...")
    
    # Load both files
    pairing_df = pd.read_csv(pairing_file_path)
    vgene_df = pd.read_csv(vgene_file_path)
    
    print(f"Pairing file shape: {pairing_df.shape}")
    print(f"V gene file shape: {vgene_df.shape}")
    
    # Define the columns to merge on
    merge_columns = [
        'sequence_alignment_aa_light',
        'generated_sequence_light',
        'input_heavy_sequence', 
        'BLOSUM_score',
        'similarity',
        'perplexity',
        'calculated_blosum',
        'calculated_similarity'
    ]
    
    print(f"Merging on columns: {merge_columns}")
    
    # Check if all merge columns exist in both dataframes
    missing_cols_pairing = [col for col in merge_columns if col not in pairing_df.columns]
    missing_cols_vgene = [col for col in merge_columns if col not in vgene_df.columns]
    
    if missing_cols_pairing:
        print(f"Warning: Missing columns in pairing file: {missing_cols_pairing}")
    if missing_cols_vgene:
        print(f"Warning: Missing columns in V gene file: {missing_cols_vgene}")
    
    # Only use columns that exist in both dataframes
    available_merge_columns = [col for col in merge_columns 
                              if col in pairing_df.columns and col in vgene_df.columns]
    
    print(f"Actually merging on: {available_merge_columns}")
    
    # Check for duplicates before merging
    print(f"\nChecking for duplicates...")
    pairing_duplicates = pairing_df.duplicated(subset=available_merge_columns).sum()
    vgene_duplicates = vgene_df.duplicated(subset=available_merge_columns).sum()
    
    print(f"Duplicates in pairing file: {pairing_duplicates}")
    print(f"Duplicates in V gene file: {vgene_duplicates}")
    
    if pairing_duplicates > 0:
        print("Removing duplicates from pairing file...")
        pairing_df = pairing_df.drop_duplicates(subset=available_merge_columns, keep='first')
        print(f"Pairing file shape after deduplication: {pairing_df.shape}")
    
    if vgene_duplicates > 0:
        print("Removing duplicates from V gene file...")
        vgene_df = vgene_df.drop_duplicates(subset=available_merge_columns, keep='first')
        print(f"V gene file shape after deduplication: {vgene_df.shape}")
    
    # Test different merge strategies
    print(f"\nTesting merge strategies...")
    
    # Inner join to see how many exact matches we get
    inner_merge = pd.merge(pairing_df, vgene_df, on=available_merge_columns, how='inner', suffixes=('_pairing', '_vgene'))
    print(f"Inner join result: {inner_merge.shape[0]} matches")
    
    # Perform the main merge (left join to keep all pairing data)
    print(f"\nPerforming main merge...")
    merged_df = pd.merge(pairing_df, vgene_df, on=available_merge_columns, how='left', suffixes=('_pairing', '_vgene'))
    
    print(f"Merged result shape: {merged_df.shape}")
    
    # Check how many rows got V gene information
    v_gene_info_added = merged_df['gen_v_gene_call'].notna().sum()
    print(f"Rows with V gene information added: {v_gene_info_added}")
    
    # Handle duplicate columns (those ending with '_pairing' or '_vgene')
    duplicate_columns = []
    
    # Check for columns that appear in both files (excluding merge columns)
    for col in pairing_df.columns:
        if col not in available_merge_columns:
            pairing_col = f"{col}_pairing"
            vgene_col = f"{col}_vgene"
            
            if pairing_col in merged_df.columns and vgene_col in merged_df.columns:
                duplicate_columns.append((col, pairing_col, vgene_col))
    
    if duplicate_columns:
        print(f"\nHandling duplicate columns:")
        for orig_col, pairing_col, vgene_col in duplicate_columns:
            print(f"  {orig_col}: keeping from pairing file, removing {vgene_col}")
            # Keep the pairing version and rename it back to original
            merged_df[orig_col] = merged_df[pairing_col]
            merged_df = merged_df.drop(columns=[pairing_col, vgene_col])
    
    # Clean up any remaining suffix columns
    remaining_suffix_cols = [col for col in merged_df.columns if col.endswith('_pairing') or col.endswith('_vgene')]
    if remaining_suffix_cols:
        print(f"Removing remaining suffix columns: {remaining_suffix_cols}")
        for col in remaining_suffix_cols:
            if col.endswith('_pairing'):
                # Keep pairing version, rename back to original
                orig_name = col.replace('_pairing', '')
                if orig_name not in merged_df.columns:
                    merged_df[orig_name] = merged_df[col]
                merged_df = merged_df.drop(columns=[col])
            elif col.endswith('_vgene'):
                # Remove vgene version if pairing version exists
                merged_df = merged_df.drop(columns=[col])
    
    # Save the merged file
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged file saved to: {output_path}")
    
    # Print detailed summary
    print_merge_summary(pairing_df, vgene_df, merged_df, v_gene_info_added)
    
    return merged_df

def print_merge_summary(pairing_df: pd.DataFrame, vgene_df: pd.DataFrame, 
                       merged_df: pd.DataFrame, v_gene_matches: int):
    """
    Print detailed summary of the merge operation.
    
    Args:
        pairing_df: Original pairing DataFrame
        vgene_df: Original V gene DataFrame  
        merged_df: Merged DataFrame
        v_gene_matches: Number of rows that got V gene information
    """
    print(f"\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    
    print(f"Original pairing file rows: {len(pairing_df)}")
    print(f"Original V gene file rows: {len(vgene_df)}")
    print(f"Merged file rows: {len(merged_df)}")
    print(f"Rows with V gene information: {v_gene_matches}")
    print(f"Match rate: {v_gene_matches/len(pairing_df)*100:.1f}%")
    
    # Check what columns were added
    original_cols = set(pairing_df.columns)
    final_cols = set(merged_df.columns)
    new_cols = final_cols - original_cols
    
    print(f"\nNew columns added:")
    for col in sorted(new_cols):
        non_null_count = merged_df[col].notna().sum()
        print(f"  {col}: {non_null_count} non-null values")
    
    # Show sample of merged data
    print(f"\nSample of merged data (first 3 rows with V gene info):")
    sample_cols = ['fasta_id', 'gen_v_gene_call', 'true_v_gene_call', 
                   'gen_v_gene_family_call', 'true_v_gene_family_call', 'pairing_scores']
    available_sample_cols = [col for col in sample_cols if col in merged_df.columns]
    
    sample_data = merged_df[merged_df['gen_v_gene_call'].notna()][available_sample_cols].head(3)
    if len(sample_data) > 0:
        print(sample_data.to_string(index=False))
    else:
        print("No rows with V gene information found for sample.")

def check_merge_quality(merged_df: pd.DataFrame):
    """
    Perform quality checks on the merged data.
    
    Args:
        merged_df: The merged DataFrame
    """
    print(f"\n" + "="*60)
    print("MERGE QUALITY CHECK")
    print("="*60)
    
    total_rows = len(merged_df)
    
    # Check pairing scores coverage
    if 'pairing_scores' in merged_df.columns:
        pairing_coverage = merged_df['pairing_scores'].notna().sum()
        print(f"Pairing scores coverage: {pairing_coverage}/{total_rows} ({pairing_coverage/total_rows*100:.1f}%)")
    
    # Check V gene information coverage
    v_gene_cols = ['gen_v_gene_call', 'true_v_gene_call', 'gen_v_gene_family_call', 'true_v_gene_family_call']
    available_v_gene_cols = [col for col in v_gene_cols if col in merged_df.columns]
    
    for col in available_v_gene_cols:
        coverage = merged_df[col].notna().sum()
        print(f"{col} coverage: {coverage}/{total_rows} ({coverage/total_rows*100:.1f}%)")
    
    # Check for any obvious data quality issues
    if 'gen_v_gene_family_call' in merged_df.columns and 'true_v_gene_family_call' in merged_df.columns:
        both_v_gene_families = (merged_df['gen_v_gene_family_call'].notna() & 
                               merged_df['true_v_gene_family_call'].notna()).sum()
        print(f"Rows with both gen and true V gene families: {both_v_gene_families}/{total_rows} ({both_v_gene_families/total_rows*100:.1f}%)")

def main():
    """
    Main function to merge the CSV files.
    """
    pairing_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/immuno_match/immunomatch_results/pairing_result_bert2gpt_full_complete_ids_mapping_unique_nt_trimmed_gene_hit_locus.csv"  # File with pairing_scores column
    vgene_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/updated_v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping.csv"      # File with V gene calls
    output_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/updated_merged_pairing_vgene_result_bert2gpt_full_complete_ids_mapping_unique_nt_trimmed_gene_hit_locus.csv"  # Output merged file
    
    try:
        # Perform the merge
        merged_df = merge_pairing_vgene_files(pairing_file_path, vgene_file_path, output_file_path)
        
        # Check merge quality
        check_merge_quality(merged_df)
        
        print(f"\nMerge completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Alternative function for troubleshooting difficult merges
def flexible_merge_with_tolerance(pairing_file: str, vgene_file: str, output_file: str, 
                                 numeric_tolerance: float = 1e-6):
    """
    More flexible merge that handles small floating point differences.
    
    Args:
        pairing_file: Path to pairing scores file
        vgene_file: Path to V gene calls file
        output_file: Output path
        numeric_tolerance: Tolerance for floating point comparisons
    """
    print("Loading files for flexible merge...")
    
    pairing_df = pd.read_csv(pairing_file)
    vgene_df = pd.read_csv(vgene_file)
    
    # Round numeric columns to handle floating point precision issues
    numeric_cols = ['BLOSUM_score', 'similarity', 'perplexity', 'calculated_blosum', 'calculated_similarity']
    
    for col in numeric_cols:
        if col in pairing_df.columns:
            pairing_df[col] = pairing_df[col].round(6)
        if col in vgene_df.columns:
            vgene_df[col] = vgene_df[col].round(6)
    
    # Try merge with rounded values
    merge_cols = ['sequence_alignment_aa_light', 'generated_sequence_light', 'input_heavy_sequence'] + numeric_cols
    available_cols = [col for col in merge_cols if col in pairing_df.columns and col in vgene_df.columns]
    
    merged_df = pd.merge(pairing_df, vgene_df, on=available_cols, how='left', suffixes=('', '_dup'))
    
    # Remove duplicate columns
    dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
    merged_df = merged_df.drop(columns=dup_cols)
    
    merged_df.to_csv(output_file, index=False)
    
    print(f"Flexible merge completed. Results saved to: {output_file}")
    print(f"Matches found: {merged_df['gen_v_gene_call'].notna().sum()}/{len(merged_df)}")
    
    return merged_df

if __name__ == "__main__":
    main()

# Example usage for flexible merge:
# flexible_merge_with_tolerance("pairing_file.csv", "vgene_file.csv", "merged_output.csv")