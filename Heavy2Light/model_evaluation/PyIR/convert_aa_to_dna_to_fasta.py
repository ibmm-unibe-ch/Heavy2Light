# used env: OAS_paired_env
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Your provided function
def amino_acid_to_dna(aa_sequence):
    codon_table = {
        'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
        'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
        'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA'
    }
    dna_sequence = ''.join(codon_table[aa] for aa in aa_sequence)
    return dna_sequence

# Input/output file names
input_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/all_seqs_generated_gpt2_sequences_10000.fasta"
output_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/all_seqs_generated_gpt2_sequences_10000_dna.fasta"

# Convert each amino acid sequence to DNA
records = []
for record in SeqIO.parse(input_fasta, "fasta"):
    aa_seq = str(record.seq)
    dna_seq_str = amino_acid_to_dna(aa_seq)
    dna_seq = Seq(dna_seq_str)
    new_record = SeqRecord(dna_seq, id=record.id, description=record.description)
    records.append(new_record)

# Write each sequence on a single line
with open(output_fasta, "w") as out_f:
    for rec in records:
        out_f.write(f">{rec.id}\n{str(rec.seq)}\n")