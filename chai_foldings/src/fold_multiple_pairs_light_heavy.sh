#!/bin/bash

#SBATCH --job-name="chai_paired_fold"
#SBATCH --gres=gpu:a100:1
#SBATCH --output="/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/chai_paired_fold_%j.o"
#SBATCH --error="/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/chai_paired_fold_%j.e"

# Script to fold paired sequences from FASTA file (alternating light and heavy chains) using chai-lab
# Usage: sbatch fold_multiple_pairs_light_heavy.sh <input_fasta> <output_base_dir> [--use-msa-server]

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_fasta> <output_base_dir> [--use-msa-server]"
    echo "Example: $0 generated_sequences.fasta /path/to/outputs --use-msa-server"
    echo "Input FASTA should have alternating light and heavy sequences"
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

# Create temporary directory for individual paired FASTA files
TEMP_DIR="$OUTPUT_BASE_DIR/temp_paired_fasta_files"
mkdir -p "$TEMP_DIR"

echo "Starting CHAI-1 paired sequence folding..."
echo "Input file: $INPUT_FASTA"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "MSA flag: $MSA_FLAG"
echo "Temporary directory: $TEMP_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda init bash
conda activate chai_env


# Function to extract sequence identifier from header
get_seq_id() {
    local header="$1"
    # Extract everything after '>' and before first space
    echo "$header" | sed 's/^>//' | awk '{print $1}'
}

# Function to extract base ID (remove protein| prefix and heavy_ prefix)
get_base_id() {
    local seq_id="$1"
    # Remove protein| prefix and heavy_ prefix
    echo "$seq_id" | sed 's/^protein|//' | sed 's/^heavy_//'
}

# Function to process a sequence pair
process_sequence_pair() {
    local light_header="$1"
    local light_sequence="$2"
    local heavy_header="$3"
    local heavy_sequence="$4"
    local pair_num="$5"
    
    # Extract sequence IDs
    light_seq_id=$(get_seq_id "$light_header")
    heavy_seq_id=$(get_seq_id "$heavy_header")
    
    # Get base ID for folder naming (should be same for both)
    base_id=$(get_base_id "$light_seq_id")


    echo "Processing sequence pair $pair_num:"
    echo "  Light: $light_seq_id (length: ${#light_sequence})"
    echo "  Heavy: $heavy_seq_id (length: ${#heavy_sequence})"
    
    # Create individual paired FASTA file
    individual_fasta="$TEMP_DIR/${base_id}.fasta"
    echo "$light_header" > "$individual_fasta"
    echo "$light_sequence" >> "$individual_fasta"
    echo "$heavy_header" >> "$individual_fasta"
    echo "$heavy_sequence" >> "$individual_fasta"
    
    # Create output directory for this sequence pair
    seq_output_dir="$OUTPUT_BASE_DIR/$base_id"
    mkdir -p "$seq_output_dir"
    
    # Run chai-lab fold
    echo "Folding sequence pair: $base_id"
    if [ "$MSA_FLAG" = "--use-msa-server" ]; then
        chai-lab fold --use-msa-server "$individual_fasta" "$seq_output_dir"
    else
        chai-lab fold "$individual_fasta" "$seq_output_dir"
    fi
    
    # Check if folding was successful
    if [ $? -eq 0 ]; then
        echo "Successfully folded sequence pair: $base_id"
        # Remove individual FASTA file to save space (optional)
        #rm "$individual_fasta"
    else
        echo "Error folding sequence pair: $base_id"
        echo "Individual FASTA file preserved at: $individual_fasta"
    fi
    
    echo "Output saved to: $seq_output_dir"
    echo "----------------------------------------"
}

# Read FASTA file and process pairs (light + heavy)
pair_count=0
light_header=""
light_sequence=""
heavy_header=""
heavy_sequence=""
sequence_counter=0

while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == ">"* ]]; then
        # If we have a complete pair, process it
        if [ -n "$light_header" ] && [ -n "$light_sequence" ] && [ -n "$heavy_header" ] && [ -n "$heavy_sequence" ]; then
            pair_count=$((pair_count + 1))
            process_sequence_pair "$light_header" "$light_sequence" "$heavy_header" "$heavy_sequence" "$pair_count"
            # Reset for next pair
            light_header=""
            light_sequence=""
            heavy_header=""
            heavy_sequence=""
            sequence_counter=0
        fi
        
        # Determine if this is light or heavy sequence based on alternating pattern
        sequence_counter=$((sequence_counter + 1))
        if [ $((sequence_counter % 2)) -eq 1 ]; then
            # Odd sequence = light chain
            light_header="$line"
            light_sequence=""
        else
            # Even sequence = heavy chain
            heavy_header="$line"
            heavy_sequence=""
        fi
    else
        # Accumulate sequence lines (remove any trailing whitespace)
        line=$(echo "$line" | tr -d '\r\n')
        if [ $((sequence_counter % 2)) -eq 1 ]; then
            # Adding to light sequence
            light_sequence="${light_sequence}${line}"
        else
            # Adding to heavy sequence
            heavy_sequence="${heavy_sequence}${line}"
        fi
    fi
done < "$INPUT_FASTA"

# Process the last pair if it exists
if [ -n "$light_header" ] && [ -n "$light_sequence" ] && [ -n "$heavy_header" ] && [ -n "$heavy_sequence" ]; then
    pair_count=$((pair_count + 1))
    echo "Processing FINAL sequence pair..."
    process_sequence_pair "$light_header" "$light_sequence" "$heavy_header" "$heavy_sequence" "$pair_count"
fi

# Clean up temporary directory if empty
if [ -z "$(ls -A $TEMP_DIR)" ]; then
    #rmdir "$TEMP_DIR"
    echo "Cleaned up temporary directory"
else
    echo "Some individual FASTA files preserved in: $TEMP_DIR"
fi

echo "Batch paired folding completed!"
echo "Total sequence pairs processed: $pair_count"
echo "Results saved in: $OUTPUT_BASE_DIR"

# Create summary file
summary_file="$OUTPUT_BASE_DIR/paired_folding_summary.txt"
echo "CHAI-1 Paired Sequence Folding Summary" > "$summary_file"
echo "======================================" >> "$summary_file"
echo "Input file: $INPUT_FASTA" >> "$summary_file"
echo "Output directory: $OUTPUT_BASE_DIR" >> "$summary_file"
echo "MSA server used: $MSA_FLAG" >> "$summary_file"
echo "Total sequence pairs: $pair_count" >> "$summary_file"
echo "Completion time: $(date)" >> "$summary_file"
echo "" >> "$summary_file"
echo "Individual results:" >> "$summary_file"

# List all output directories
for dir in "$OUTPUT_BASE_DIR"/*/; do
    if [ -d "$dir" ] && [ "$(basename "$dir")" != "temp_paired_fasta_files" ]; then
        seq_name=$(basename "$dir")
        echo "  - $seq_name: $dir" >> "$summary_file"
    fi
done

echo "Summary saved to: $summary_file"