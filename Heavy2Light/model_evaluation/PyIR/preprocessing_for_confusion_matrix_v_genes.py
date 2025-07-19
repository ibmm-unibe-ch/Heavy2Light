import json
import pandas as pd
import re
from typing import Dict, Tuple, Optional

def extract_v_gene_info(sequence_data: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract V gene call, simple gene name, and family from the first hit in the sequence data.
    
    Args:
        sequence_data: Dictionary containing sequence information
        
    Returns:
        Tuple of (full_v_gene, simple_v_gene, v_gene_family) or (None, None, None) if no hits
    """
    hits = sequence_data.get('Hits', [])
    if not hits:
        return None, None, None
    
    # Get the first hit (highest scoring)
    first_hit = hits[0]
    gene_name = first_hit.get('gene', '')
    
    # Extract the gene identifier (e.g., "IGLV3-25*02" from "IGLV3-25*02 unnamed protein product")
    # Look for pattern like IGLV/IGKV/IGHV followed by numbers and asterisk
    gene_pattern = r'(IG[LKH]V\d+[A-Z]*-?[A-Z0-9]*\*?\d*)'
    match = re.search(gene_pattern, gene_name)
    
    if match:
        full_v_gene = match.group(1)
        
        # Extract simple gene name (everything before the asterisk)
        # e.g., "IGLV3-25*02" -> "IGLV3-25"
        if '*' in full_v_gene:
            simple_v_gene = full_v_gene.split('*')[0]
        else:
            simple_v_gene = full_v_gene
        
        # Extract family (e.g., "IGLV3" from "IGLV3-25*02" or "IGKV1" from "IGKV1D-33*01")
        family_pattern = r'(IG[LKH]V\d+[A-Z]*)'
        family_match = re.search(family_pattern, full_v_gene)
        if family_match:
            v_gene_family = family_match.group(1)
            # Remove 'D' from family name if present (e.g., IGKV1D -> IGKV1)
            v_gene_family = v_gene_family.replace('D', '')
        else:
            v_gene_family = None
        
        return full_v_gene, simple_v_gene, v_gene_family
    
    return None, None, None

def extract_base_fasta_id(sequence_id: str) -> str:
    """
    Extract base fasta ID by removing gen_ or true_ prefix.
    
    Args:
        sequence_id: Full sequence ID
        
    Returns:
        Base fasta ID without prefix
    """
    if sequence_id.startswith('gen_'):
        return sequence_id[4:]  # Remove 'gen_'
    elif sequence_id.startswith('true_'):
        return sequence_id[5:]  # Remove 'true_'
    else:
        return sequence_id

def process_json_file(json_file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Process JSON file to extract V gene information for each sequence.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        Dictionary mapping base_fasta_id to gene information for gen and true sequences
    """
    v_gene_data = {}
    
    with open(json_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                sequence_data = json.loads(line)
                sequence_id = sequence_data.get('Sequence ID', '')
                
                # Extract V gene information
                full_v_gene, simple_v_gene, v_gene_family = extract_v_gene_info(sequence_data)
                
                # Determine if this is a generated or true sequence
                base_id = extract_base_fasta_id(sequence_id)
                
                if base_id not in v_gene_data:
                    v_gene_data[base_id] = {}
                
                if sequence_id.startswith('gen_'):
                    v_gene_data[base_id]['gen_v_gene_call'] = full_v_gene
                    v_gene_data[base_id]['gen_v_gene_simple'] = simple_v_gene
                    v_gene_data[base_id]['gen_v_gene_family_call'] = v_gene_family
                elif sequence_id.startswith('true_'):
                    v_gene_data[base_id]['true_v_gene_call'] = full_v_gene
                    v_gene_data[base_id]['true_v_gene_simple'] = simple_v_gene
                    v_gene_data[base_id]['true_v_gene_family_call'] = v_gene_family
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
    
    return v_gene_data

def update_mapping_csv(csv_file_path: str, v_gene_data: Dict[str, Dict[str, str]], output_file_path: str):
    """
    Update the mapping CSV file with V gene information.
    
    Args:
        csv_file_path: Path to the input CSV file
        v_gene_data: Dictionary containing V gene information
        output_file_path: Path for the output CSV file
    """
    # Read the mapping CSV
    df = pd.read_csv(csv_file_path)
    
    # Initialize new columns in the desired order
    df['gen_v_gene_call'] = None
    df['true_v_gene_call'] = None
    df['true_v_gene_simple'] = None
    df['gen_v_gene_simple'] = None
    df['gen_v_gene_family_call'] = None
    df['true_v_gene_family_call'] = None
    
    # Update rows with V gene information
    for index, row in df.iterrows():
        base_fasta_id = row['base_fasta_id']
        
        if base_fasta_id in v_gene_data:
            gene_info = v_gene_data[base_fasta_id]
            
            df.at[index, 'gen_v_gene_call'] = gene_info.get('gen_v_gene_call')
            df.at[index, 'true_v_gene_call'] = gene_info.get('true_v_gene_call')
            df.at[index, 'true_v_gene_simple'] = gene_info.get('true_v_gene_simple')
            df.at[index, 'gen_v_gene_simple'] = gene_info.get('gen_v_gene_simple')
            df.at[index, 'gen_v_gene_family_call'] = gene_info.get('gen_v_gene_family_call')
            df.at[index, 'true_v_gene_family_call'] = gene_info.get('true_v_gene_family_call')
    
    # Save the updated CSV
    df.to_csv(output_file_path, index=False)
    print(f"Updated mapping file saved to: {output_file_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total rows processed: {len(df)}")
    print(f"Rows with gen V gene calls: {df['gen_v_gene_call'].notna().sum()}")
    print(f"Rows with true V gene calls: {df['true_v_gene_call'].notna().sum()}")
    print(f"Rows with gen V gene simple: {df['gen_v_gene_simple'].notna().sum()}")
    print(f"Rows with true V gene simple: {df['true_v_gene_simple'].notna().sum()}")


def main():
    """
    Main function to process JSON file and update mapping CSV.
    
    Usage:
    1. Update the file paths below
    2. Run the script
    """
    
    json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/bert2gpt_complete_ids_gen_true.json"  
    csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/bert2gpt_complete_ids_gen_true.mapping.csv"      
    output_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/updated_v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping.csv"  
    
    try:
        print("Processing JSON file to extract V gene information...")
        v_gene_data = process_json_file(json_file_path)
        
        print(f"Found V gene data for {len(v_gene_data)} base sequences")
        
        print("Updating mapping CSV file...")
        update_mapping_csv(csv_file_path, v_gene_data, output_file_path)
        
        print("Process completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# Example usage for testing with your sample data:
def test_with_sample_data():
    """
    Test function using the sample data you provided.
    """
    # Sample JSON data from your example
    sample_data = [
        {
            "Sequence ID": "true_seq_371_0a22964b_sim57.7_blosum307_perp1.9",
            "Hits": [{"gene": "IGLV3-NL1*01 unnamed protein product", "bit_score": 192.0, "e_value": 8e-68}]
        },
        {
            "Sequence ID": "gen_seq_372_1c331f18_sim50.0_blosum209_perp3.4",
            "Hits": [{"gene": "IGLV1-51*01 unnamed protein product", "bit_score": 200.0, "e_value": 6e-71}]
        },
        {
            "Sequence ID": "true_seq_372_1c331f18_sim50.0_blosum209_perp3.4",
            "Hits": [{"gene": "IGKV3-20*01 unnamed protein product", "bit_score": 160.0, "e_value": 4e-55}]
        },
        {
            "Sequence ID": "gen_seq_373_7a96ea02_sim61.5_blosum331_perp3.3",
            "Hits": [{"gene": "IGKV3-11*01 unnamed protein product", "bit_score": 194.0, "e_value": 2e-68}]
        }
    ]
    
    print("Testing with sample data:")
    for data in sample_data:
        full_gene, simple_gene, family = extract_v_gene_info(data)
        base_id = extract_base_fasta_id(data["Sequence ID"])
        print(f"Sequence: {data['Sequence ID']}")
        print(f"  Base ID: {base_id}")
        print(f"  Full V gene: {full_gene}")
        print(f"  Simple V gene: {simple_gene}")
        print(f"  V gene family: {family}")
        print()


#test_with_sample_data()