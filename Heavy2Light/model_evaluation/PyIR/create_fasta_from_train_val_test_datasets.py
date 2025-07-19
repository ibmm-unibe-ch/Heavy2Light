"""
Convert train, validation, and test datasets from txt format to FASTA format.
Each line contains heavy and light chain sequences separated by [SEP].
Only the light chain sequences (after [SEP]) are extracted.
"""

import os
from pathlib import Path

def process_dataset_to_fasta(input_file, output_file, dataset_type):
    """
    Convert a txt dataset file to FASTA format, extracting only light chains.
    
    Args:
        input_file (str): Path to input txt file
        output_file (str): Path to output FASTA file
        dataset_type (str): Type of dataset ('train', 'val', or 'test')
    """
    
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found. Skipping...")
        return
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        sequence_count = 0
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Split by [SEP] separator to get heavy and light chains
            sequences = line.split('[SEP]')
            
            if len(sequences) != 2:
                print(f"Warning: Line {line_num} has {len(sequences)} sequences instead of 2")
                continue
            
            # Extract only the light chain (sequence after [SEP])
            light_chain = sequences[1].strip()
            
            if light_chain:  # Only process non-empty sequences
                sequence_count += 1
                
                # Create unique identifier for light chain
                seq_id = f"{dataset_type}_lightchain_{line_num:06d}_{sequence_count:08d}"
                
                # Write to FASTA format
                outfile.write(f">{seq_id}\n")
                outfile.write(f"{light_chain}\n")
        
        print(f"Processed {input_file}: {sequence_count} sequences -> {output_file}")

def main():
    """Main function to process all dataset files."""
    
    # Define input and output file mappings
    datasets = {
        'train': {
            'input': '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_train_no_identifiers.txt',
            'output': '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/train_val_test_fastas/plabdab_human_healthy_no_vac_allocated_train_no_identifiers.fasta'
        },
        'val': {
            'input': '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_val_no_identifiers.txt', 
            'output': '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/train_val_test_fastas/plabdab_human_healthy_no_vac_allocated_val_no_identifiers.fasta'
        },
        'test': {
            'input': '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_test_no_identifiers.txt',
            'output': '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/train_val_test_fastas/plabdab_human_healthy_no_vac_allocated_test_no_identifiers.fasta'
        }
    }
    
    print("Converting TXT datasets to FASTA format...")
    print("=" * 50)
    
    total_processed = 0
    
    # Process each dataset
    for dataset_type, files in datasets.items():
        input_file = files['input']
        output_file = files['output']
        
        print(f"\nProcessing {dataset_type} dataset...")
        process_dataset_to_fasta(input_file, output_file, dataset_type)
        
        if os.path.exists(output_file):
            total_processed += 1
    
    print("\n" + "=" * 50)
    print(f"Conversion complete! {total_processed} FASTA files created.")
    
    # Display expected output format
    print("\nExpected FASTA format (light chains only):")
    print(">train_lightchain_000001_00000001")
    print("SYELTQPPSVSVSPGQTASITCSGHKLGDKYASWYQDKPGQSPVLVIYQDTKRPSGIPERFSGSNSGNTAILTISGTQPMDEADYYCQAWDSSTLVFGGGTKVTVL")
    print(">train_lightchain_000002_00000002")
    print("NEXT_LIGHT_CHAIN_SEQUENCE...")

if __name__ == "__main__":
    main()

