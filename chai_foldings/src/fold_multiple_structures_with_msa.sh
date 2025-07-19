#!/bin/bash

#SBATCH --job-name="chai_batch_fold"
#SBATCH --gres=gpu:h100:1
#SBATCH --output="/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/chai_batch_fold_%j.o"
#SBATCH --error="/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/chai_batch_fold_%j.e"

# Script to fold each sequence in a FASTA file individually using chai-lab
# Usage: sbatch fold_multiple_structures_with_msa.sh <input_fasta> <output_base_dir> [--use-msa-server]

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_fasta> <output_base_dir> [--use-msa-server]"
    echo "Example: $0 sequences.fasta /path/to/outputs --use-msa-server"
    exit 1
fi

INPUT_FASTA="$1"
OUTPUT_BASE_DIR="$2"
MSA_FLAG="$3"

# Check if input file exists
if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: Input FASTA file '$INPUT_FASTA' not found!"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Create temporary directory for individual FASTA files
TEMP_DIR="$OUTPUT_BASE_DIR/temp_fasta_files"
mkdir -p "$TEMP_DIR"

echo "Starting CHAI-1 batch folding..."
echo "Input file: $INPUT_FASTA"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "MSA flag: $MSA_FLAG"
echo "Temporary directory: $TEMP_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda init bash
conda activate chai_env

# Function to clean sequence identifier (remove special characters for folder names)
clean_id() {
    local id="$1"
    # Remove '>' and replace problematic characters with underscores
    echo "$id" | sed 's/^>//' | sed 's/[^a-zA-Z0-9._-]/_/g'
}

# Function to extract sequence identifier from header
get_seq_id() {
    local header="$1"
    # Extract everything after '>' and before first space
    echo "$header" | sed 's/^>//' | awk '{print $1}'
}

# Function to process a sequence
process_sequence() {
    local header="$1"
    local sequence="$2"
    local seq_num="$3"
    
    # Extract sequence ID and clean it for folder name
    seq_id=$(get_seq_id "$header")
    clean_seq_id=$(clean_id "$seq_id")
    
    echo "Processing sequence $seq_num: $seq_id"
    echo "Header: $header"
    echo "Sequence length: ${#sequence}"
    
    # Create individual FASTA file
    individual_fasta="$TEMP_DIR/${clean_seq_id}.fasta"
    echo "$header" > "$individual_fasta"
    echo "$sequence" >> "$individual_fasta"
    
    # Create output directory for this sequence
    seq_output_dir="$OUTPUT_BASE_DIR/$clean_seq_id"
    mkdir -p "$seq_output_dir"
    
    # Run chai-lab fold
    echo "Folding sequence: $seq_id"
    if [ "$MSA_FLAG" = "--use-msa-server" ]; then
        chai-lab fold --use-msa-server "$individual_fasta" "$seq_output_dir"
    else
        chai-lab fold "$individual_fasta" "$seq_output_dir"
    fi
    
    # Check if folding was successful
    if [ $? -eq 0 ]; then
        echo "Successfully folded sequence: $seq_id"
        # Remove individual FASTA file to save space (optional)
        rm "$individual_fasta"
    else
        echo "Error folding sequence: $seq_id"
        echo "Individual FASTA file preserved at: $individual_fasta"
    fi
    
    echo "Output saved to: $seq_output_dir"
    echo "----------------------------------------"
}

# Split FASTA file into individual sequences and process each
sequence_count=0
current_header=""
current_sequence=""

while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == ">"* ]]; then
        # If we have a previous sequence, process it
        if [ -n "$current_header" ] && [ -n "$current_sequence" ]; then
            sequence_count=$((sequence_count + 1))
            process_sequence "$current_header" "$current_sequence" "$sequence_count"
        fi
        
        # Start new sequence
        current_header="$line"
        current_sequence=""
    else
        # Accumulate sequence lines (remove any trailing whitespace)
        line=$(echo "$line" | tr -d '\r\n')
        current_sequence="${current_sequence}${line}"
    fi
done < "$INPUT_FASTA"

# Process the last sequence (this is crucial!)
if [ -n "$current_header" ] && [ -n "$current_sequence" ]; then
    sequence_count=$((sequence_count + 1))
    echo "Processing FINAL sequence..."
    process_sequence "$current_header" "$current_sequence" "$sequence_count"
fi

# Clean up temporary directory if empty
if [ -z "$(ls -A $TEMP_DIR)" ]; then
    rmdir "$TEMP_DIR"
    echo "Cleaned up temporary directory"
else
    echo "Some individual FASTA files preserved in: $TEMP_DIR"
fi

echo "Batch folding completed!"
echo "Total sequences processed: $sequence_count"
echo "Results saved in: $OUTPUT_BASE_DIR"

# Create summary file
summary_file="$OUTPUT_BASE_DIR/folding_summary.txt"
echo "CHAI-1 Batch Folding Summary" > "$summary_file"
echo "=============================" >> "$summary_file"
echo "Input file: $INPUT_FASTA" >> "$summary_file"
echo "Output directory: $OUTPUT_BASE_DIR" >> "$summary_file"
echo "MSA server used: $MSA_FLAG" >> "$summary_file"
echo "Total sequences: $sequence_count" >> "$summary_file"
echo "Completion time: $(date)" >> "$summary_file"
echo "" >> "$summary_file"
echo "Individual results:" >> "$summary_file"

# List all output directories
for dir in "$OUTPUT_BASE_DIR"/*/; do
    if [ -d "$dir" ] && [ "$(basename "$dir")" != "temp_fasta_files" ]; then
        seq_name=$(basename "$dir")
        echo "  - $seq_name: $dir" >> "$summary_file"
    fi
done

echo "Summary saved to: $summary_file"

