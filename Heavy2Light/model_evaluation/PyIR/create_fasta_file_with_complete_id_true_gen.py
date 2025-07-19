import pandas as pd
import argparse
import hashlib
import os

def create_fasta_id(row, row_index):
    """
    Create a complete FASTA ID that allows tracking back to the CSV row.
    
    Args:
        row: The CSV row with antibody data
        row_index: The original index in the CSV file
    
    Returns:
        A string suitable for use as a FASTA ID/header
    """
    # Include hash and all metrics
    hash_input = row["input_heavy_sequence"] + row["generated_sequence_light"]
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    sim = round(row["calculated_similarity"], 1) if "calculated_similarity" in row else 0
    blosum = round(row["calculated_blosum"], 1) if "calculated_blosum" in row else 0
    perp = round(row["perplexity"], 1) if "perplexity" in row else 0
    return f"seq_{row_index}_{hash_value}_sim{sim}_blosum{blosum}_perp{perp}"

def csv_to_fasta(csv_file, output_fasta, create_mapping=True):
    """
    Convert a CSV file with antibody sequences to FASTA format.
    Both generated_sequence_light and sequence_alignment_aa_light are included
    with different prefixes.
    
    Args:
        csv_file: Path to the input CSV file
        output_fasta: Path to the output FASTA file
        create_mapping: Whether to create a mapping file to link FASTA IDs back to CSV rows
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if the required columns exist
    required_columns = ["generated_sequence_light", "sequence_alignment_aa_light"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column")
    
    # Open output FASTA file
    with open(output_fasta, 'w') as fasta_out:
        # Process each row and write to FASTA
        for i, (index, row) in enumerate(df.iterrows()):
            # Create the base FASTA ID
            base_id = create_fasta_id(row, index)
            
            # Write generated sequence to FASTA file
            gen_id = f"gen_{base_id}"
            fasta_out.write(f">{gen_id}\n")
            fasta_out.write(f"{row['generated_sequence_light']}\n")
            
            # Write alignment sequence to FASTA file
            true_id = f"true_{base_id}"
            fasta_out.write(f">{true_id}\n")
            fasta_out.write(f"{row['sequence_alignment_aa_light']}\n")
    
    # If creating mapping file
    if create_mapping:
        mapping_path = output_fasta.replace('.fasta', '.mapping.csv')
        
        # Create a new DataFrame with fasta_id, indices first
        mapping_df = df.copy()
        base_ids = [create_fasta_id(row, idx) for idx, row in df.iterrows()]
        mapping_df['gen_fasta_id'] = [f"gen_{base_id}" for base_id in base_ids]
        mapping_df['true_fasta_id'] = [f"true_{base_id}" for base_id in base_ids]
        mapping_df['base_fasta_id'] = base_ids
        mapping_df['csv_row_index'] = df.index
        mapping_df['csv_row_number'] = range(1, len(df) + 1)
        
        # Reorder columns to put fasta_id and indices first
        cols = ['gen_fasta_id', 'true_fasta_id', 'base_fasta_id', 'csv_row_index', 'csv_row_number'] + [c for c in df.columns]
        mapping_df = mapping_df[cols]
        
        # Save the complete mapping DataFrame to CSV
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Mapping file created: {mapping_path}")
    
    print(f"FASTA file created: {output_fasta}")
    print(f"Processed {len(df)} sequences (generated {len(df)*2} FASTA entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV antibody data to FASTA format")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output FASTA file")
    parser.add_argument("--no-mapping", action="store_true",
                        help="Do not create a mapping file")
    
    args = parser.parse_args()
    
    csv_to_fasta(
        csv_file=args.input,
        output_fasta=args.output,
        create_mapping=not args.no_mapping
    )