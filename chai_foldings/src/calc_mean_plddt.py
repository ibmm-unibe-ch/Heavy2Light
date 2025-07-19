import numpy as np

def calculate_mean_plddt(cif_file_path):
    """
    Calculate the mean pLDDT score from a .cif file.
    
    Args:
        cif_file_path (str): Path to the .cif file
    
    Returns:
        float: Mean pLDDT score
    """
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
        mean_plddt = np.mean(plddt_scores)
        return mean_plddt, plddt_scores
    else:
        return None


# Alternative one-liner approach using list comprehension
def calculate_mean_plddt_oneliner(cif_file_path):
    """One-liner version for calculating mean pLDDT"""
    with open(cif_file_path, 'r') as f:
        scores = [float(line.split()[-2]) for line in f 
                 if line.startswith('ATOM') and len(line.split()) >= 2]
    return sum(scores) / len(scores) if scores else None

# Using pandas (if you prefer)
def calculate_mean_plddt_pandas(cif_file_path):
    """
    Alternative using pandas for more robust data handling
    """
    import pandas as pd
    
    # Read only ATOM lines
    atom_lines = []
    with open(cif_file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                atom_lines.append(line.strip().split())
    
    if not atom_lines:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(atom_lines)
    
    # Convert the second-to-last column to numeric
    plddt_column = df.iloc[:, -2]
    plddt_scores = pd.to_numeric(plddt_column, errors='coerce')
    
    # Calculate mean, ignoring NaN values
    return plddt_scores.mean()

# Usage example
if __name__ == "__main__":
    cif_file = '/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/chai_outputs/uncond_gpt_2_100_seqs/protein_seq1746097525361/pred.model_idx_0.cif'
    
    mean_score, plddt_scores = calculate_mean_plddt(cif_file)
    
    if mean_score is not None:
        print(f"Mean pLDDT score: {mean_score:.2f}")
        print(f"Number of atoms processed: {len(plddt_scores)}")
    else:
        print("No pLDDT scores found in the file.")
