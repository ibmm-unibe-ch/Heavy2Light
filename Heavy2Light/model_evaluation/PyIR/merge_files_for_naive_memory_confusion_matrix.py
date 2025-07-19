import pandas as pd
import numpy as np

def merge_csv_files(file1_path: str, file2_path: str, output_path: str):
    """
    Merge two CSV files based on matching columns.
    
    Args:
        file1_path: Path to the first CSV file (main dataset)
        file2_path: Path to the second CSV file (with V gene calls)
        output_path: Path for the merged output file
    """
    
    print("Loading CSV files...")
    
    # Load both files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    
    # Rename columns in df2 to match df1 for merging
    df2_renamed = df2.rename(columns={
        'sequence_alignment_aa_light': 'sequence_alignment_aa_light',  # Same name
        'input_heavy_sequence': 'sequence_alignment_aa_heavy'  # Rename to match df1
    })
    
    # Define the columns to merge on
    merge_columns = [
        'sequence_alignment_aa_light',
        'sequence_alignment_aa_heavy', 
        'BLOSUM_score',
        'similarity',
        'perplexity',
        'calculated_blosum',
        'calculated_similarity'
    ]
    
    print(f"Merging on columns: {merge_columns}")
    
    # Check if all merge columns exist in both dataframes
    missing_cols_df1 = [col for col in merge_columns if col not in df1.columns]
    missing_cols_df2 = [col for col in merge_columns if col not in df2_renamed.columns]
    
    if missing_cols_df1:
        print(f"Warning: Missing columns in file 1: {missing_cols_df1}")
    if missing_cols_df2:
        print(f"Warning: Missing columns in file 2: {missing_cols_df2}")
    
    # Only use columns that exist in both dataframes
    available_merge_columns = [col for col in merge_columns 
                              if col in df1.columns and col in df2_renamed.columns]
    
    print(f"Actually merging on: {available_merge_columns}")
    
    # Check for duplicates before merging
    print(f"\nChecking for duplicates...")
    df1_duplicates = df1.duplicated(subset=available_merge_columns).sum()
    df2_duplicates = df2_renamed.duplicated(subset=available_merge_columns).sum()
    
    print(f"Duplicates in file 1: {df1_duplicates}")
    print(f"Duplicates in file 2: {df2_duplicates}")
    
    # Perform the merge
    print(f"\nPerforming merge...")
    
    # First try inner join to see how many matches we get
    inner_merge = pd.merge(df1, df2_renamed, on=available_merge_columns, how='inner', suffixes=('', '_from_file2'))
    print(f"Inner join result: {inner_merge.shape[0]} matches")
    
    # Perform left join to keep all records from file 1
    merged_df = pd.merge(df1, df2_renamed, on=available_merge_columns, how='left', suffixes=('', '_from_file2'))
    
    print(f"Left join result: {merged_df.shape}")
    
    # Check how many rows from file 1 got V gene information
    v_gene_info_added = merged_df['gen_v_gene_call'].notna().sum()
    print(f"Rows with V gene information added: {v_gene_info_added}")
    
    # Remove duplicate columns (those ending with '_from_file2')
    duplicate_columns = [col for col in merged_df.columns if col.endswith('_from_file2')]
    if duplicate_columns:
        print(f"Removing duplicate columns: {duplicate_columns}")
        merged_df = merged_df.drop(columns=duplicate_columns)
    
    # Save the merged file
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged file saved to: {output_path}")
    
    # Print summary
    print(f"\nMerge Summary:")
    print(f"Original file 1 rows: {len(df1)}")
    print(f"Original file 2 rows: {len(df2)}")
    print(f"Merged file rows: {len(merged_df)}")
    print(f"Rows with V gene calls: {merged_df['gen_v_gene_call'].notna().sum()}")
    print(f"New columns added: gen_v_gene_call, true_v_gene_call, gen_v_gene_family_call, true_v_gene_family_call")
    
    return merged_df

def check_merge_quality(merged_df: pd.DataFrame):
    """
    Check the quality of the merge and identify potential issues.
    
    Args:
        merged_df: The merged DataFrame
    """
    print(f"\n" + "="*50)
    print("MERGE QUALITY CHECK")
    print("="*50)
    
    total_rows = len(merged_df)
    
    # Check V gene information coverage
    gen_coverage = merged_df['gen_v_gene_call'].notna().sum()
    true_coverage = merged_df['true_v_gene_call'].notna().sum()
    both_coverage = (merged_df['gen_v_gene_call'].notna() & merged_df['true_v_gene_call'].notna()).sum()
    
    print(f"V Gene Call Coverage:")
    print(f"  Generated V gene calls: {gen_coverage}/{total_rows} ({gen_coverage/total_rows*100:.1f}%)")
    print(f"  True V gene calls: {true_coverage}/{total_rows} ({true_coverage/total_rows*100:.1f}%)")
    print(f"  Both gen and true calls: {both_coverage}/{total_rows} ({both_coverage/total_rows*100:.1f}%)")
    
    # Check for any obvious issues
    if gen_coverage == 0:
        print("\nWARNING: No generated V gene calls found in merged data!")
    if true_coverage == 0:
        print("\nWARNING: No true V gene calls found in merged data!")
    
    # Show sample of merged data
    print(f"\nSample of merged data (first 5 rows with V gene info):")
    sample_cols = ['sequence_alignment_aa_light', 'gen_v_gene_call', 'true_v_gene_call', 
                   'gen_v_gene_family_call', 'true_v_gene_family_call']
    available_sample_cols = [col for col in sample_cols if col in merged_df.columns]
    
    sample_data = merged_df[merged_df['gen_v_gene_call'].notna()][available_sample_cols].head()
    if len(sample_data) > 0:
        print(sample_data.to_string(index=False))
    else:
        print("No rows with V gene information found for sample.")

def main():
    """
    Main function to merge the CSV files.
    """
    file1_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/peak_analysis/bert2gpt/data/bert2gpt_df_merged_final_test_set.csv"  # Main dataset file
    file2_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/all_v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping.csv"  # File with V gene calls  
    output_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/all_v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping_merged_dataset_naive_memory.csv"  # Output merged file
    
    try:
        # Perform the merge
        merged_df = merge_csv_files(file1_path, file2_path, output_path)
        
        # Check merge quality
        check_merge_quality(merged_df)
        
        print(f"\nMerge completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# # Alternative function for more flexible merging
# def flexible_merge(file1_path: str, file2_path: str, output_path: str, 
#                   merge_strategy: str = 'best_match'):
#     """
#     More flexible merge function that tries different strategies.
    
#     Args:
#         file1_path: Path to first CSV
#         file2_path: Path to second CSV  
#         output_path: Output path
#         merge_strategy: 'exact', 'subset', or 'best_match'
#     """
#     df1 = pd.read_csv(file1_path)
#     df2 = pd.read_csv(file2_path)
    
#     # Rename the heavy sequence column
#     df2_renamed = df2.rename(columns={'input_heavy_sequence': 'sequence_alignment_aa_heavy'})
    
#     if merge_strategy == 'exact':
#         # Try exact match on all available columns
#         merge_cols = ['sequence_alignment_aa_light', 'sequence_alignment_aa_heavy', 
#                      'BLOSUM_score', 'similarity', 'perplexity', 'calculated_blosum', 'calculated_similarity']
#         merge_cols = [col for col in merge_cols if col in df1.columns and col in df2_renamed.columns]
        
#     elif merge_strategy == 'subset':
#         # Try with just the sequence columns
#         merge_cols = ['sequence_alignment_aa_light', 'sequence_alignment_aa_heavy']
#         merge_cols = [col for col in merge_cols if col in df1.columns and col in df2_renamed.columns]
        
#     elif merge_strategy == 'best_match':
#         # Try progressively smaller sets of columns
#         merge_options = [
#             ['sequence_alignment_aa_light', 'sequence_alignment_aa_heavy', 'BLOSUM_score', 'similarity', 'perplexity'],
#             ['sequence_alignment_aa_light', 'sequence_alignment_aa_heavy', 'BLOSUM_score'],
#             ['sequence_alignment_aa_light', 'sequence_alignment_aa_heavy'],
#             ['sequence_alignment_aa_light']
#         ]
        
#         best_merge = None
#         best_matches = 0
        
#         for merge_cols in merge_options:
#             available_cols = [col for col in merge_cols if col in df1.columns and col in df2_renamed.columns]
#             if len(available_cols) == 0:
#                 continue
                
#             test_merge = pd.merge(df1, df2_renamed, on=available_cols, how='inner')
#             matches = len(test_merge)
            
#             print(f"Merge on {available_cols}: {matches} matches")
            
#             if matches > best_matches:
#                 best_matches = matches
#                 best_merge = available_cols
        
#         merge_cols = best_merge if best_merge else ['sequence_alignment_aa_light']
    
#     print(f"Using merge columns: {merge_cols}")
    
#     # Perform final merge
#     merged_df = pd.merge(df1, df2_renamed, on=merge_cols, how='left', suffixes=('', '_dup'))
    
#     # Remove duplicate columns
#     dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
#     merged_df = merged_df.drop(columns=dup_cols)
    
#     merged_df.to_csv(output_path, index=False)
#     print(f"Flexible merge saved to: {output_path}")
#     print(f"Total rows: {len(merged_df)}, Rows with V gene info: {merged_df['gen_v_gene_call'].notna().sum()}")
    
#     return merged_df