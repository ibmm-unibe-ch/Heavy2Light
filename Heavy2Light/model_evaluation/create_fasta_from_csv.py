import csv
import sys

def csv_to_fasta(csv_filename, fasta_filename, keep_only_unique=False):
    """
    Convert a CSV file with the specified columns to a FASTA file.
    Each identifier will be a concatenation of all columns except 'generated_sequence_light',
    which will be used as the sequence.
    """
    with open(csv_filename, 'r') as csv_file, open(fasta_filename, 'w') as fasta_file:
        # Read the CSV file
        csv_reader = csv.DictReader(csv_file)

        if keep_only_unique:
            # only keep unique sequences to generate a unique fasta file
            unique_sequences = set()
            for row in csv_reader:
                sequence = row['generated_sequence_light']
                if sequence not in unique_sequences:
                    unique_sequences.add(sequence)
                    # Create identifier from all other specified columns
                    identifier_parts = [
                        f"seq_alignment={row['sequence_alignment_aa_light'][:10]}__",
                        f"BLOSUM={row['BLOSUM_score']}",
                        f"similarity={row['similarity']}",
                        f"perplexity={row['perplexity']}"
                    ]
                    
                    identifier = "|".join(identifier_parts)
                    
                    # Write to FASTA format with the sequence on a single line
                    fasta_file.write(f">{identifier}\n{sequence}\n")
        else:
            # Process each row
            for row in csv_reader:
                # Extract the sequence
                sequence = row['generated_sequence_light']
                
                # Create identifier from all other specified columns
                identifier_parts = [
                    # keep only first 10 aa of the sequence
                    f"seq_alignment={row['sequence_alignment_aa_light'][:10]}__",
                    f"BLOSUM={row['BLOSUM_score']}",
                    f"similarity={row['similarity']}",
                    f"perplexity={row['perplexity']}"
                ]
                
                identifier = "|".join(identifier_parts)
                
                # Write to FASTA format with the sequence on a single line
                fasta_file.write(f">{identifier}\n{sequence}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_fasta.py input.csv output.fasta")
        sys.exit(1)
    
    csv_filename = sys.argv[1]
    fasta_filename = sys.argv[2]
    
    csv_to_fasta(csv_filename, fasta_filename, keep_only_unique=True)
    print(f"Successfully converted {csv_filename} to {fasta_filename}")


    