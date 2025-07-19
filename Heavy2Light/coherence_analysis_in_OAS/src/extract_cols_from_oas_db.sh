#!/bin/bash

# SBATCH --gres=gpu:a100:1
# SBATCH --job-name=extract_columns_from_slite_db_analysis
# SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/extract_columns_from_slite_db_analysis_%j.o
# SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/extract_columns_from_slite_db_analysis_%j.e


# Path to the database
# paired database
DATABASE_PATH="/ibmm_data2/oas_database/OAS_paired.db"

# Output CSV file
OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/coherence_analysis_in_oas_db/data/full_extraction_for_coherence_paired_data_extra_cols.csv"

# unpaired heavy and unpaired light columns
COLUMNS="sequence_heavy,locus_heavy,v_call_heavy,sequence_alignment_heavy,sequence_alignment_aa_heavy,germline_alignment_aa_heavy,cdr3_aa_heavy,sequence_light,locus_light,v_call_light,sequence_alignment_light,sequence_alignment_aa_light,germline_alignment_aa_light,cdr3_aa_light,sequence_alignment_heavy_sep_light,BType,Disease,Species,Subject,Author,Age"

TABLE_NAME="all_human_paired"

# Run SQLite commands
sqlite3 $DATABASE_PATH <<EOF
.mode csv
.output $OUTPUT_FILE
SELECT $COLUMNS FROM $TABLE_NAME;
.output stdout
.quit
EOF

echo "Data extracted to $OUTPUT_FILE"
