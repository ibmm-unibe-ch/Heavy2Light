#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=extract_columns_from_slite_db_analysis
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/logs/extract_columns_from_slite_db_analysis_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/logs/extract_columns_from_slite_db_analysis_%j.e


# Path to the database
# paired database
#DATABASE_PATH="/ibmm_data2/oas_database/OAS_paired.db"
DATABASE_PATH="/ibmm_scratch/oas_database/OAS_paired.db"

# heavy database
#DATABASE_PATH="/ibmm_data2/oas_database/OAS_heavy_part2.db"

# light database
#DATABASE_PATH="/ibmm_data2/oas_database/OAS_light.db"

# Output CSV file
#OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_paired_data_for_analysis.csv"
#OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/Btypes_full_paired_data_for_analysis.csv"
#OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_vdj_genes_paired_oas_db_full_extraction.csv"
#OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/full_extraction_heavy_unpaired_seqs_part2.csv"
#OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/full_extraction_light_unpaired_seqs.csv"
#OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/multi_class_v_gene_family_classification/data/full_paired_extraction_multi_class_v_classification.csv"
OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/multi_class_v_gene_family_classification/data/full_paired_extraction_kappa_translation_only.csv"

# Columns to extract
# COLUMNS="cdr1_aa_light, cdr1_end_light, cdr1_light, cdr1_start_light, \
# cdr2_aa_light, cdr2_end_light, cdr2_light, cdr2_start_light, \
# cdr3_aa_light, cdr3_end_light, cdr3_light, cdr3_start_light, \
# fwr1_aa_light, fwr1_end_light, fwr1_light, fwr1_start_light, \
# fwr2_aa_light, fwr2_end_light, fwr2_light, fwr2_start_light, \
# fwr3_aa_light, fwr3_end_light, fwr3_light, fwr3_start_light, \
# fwr4_aa_light, fwr4_end_light, fwr4_light, fwr4_start_light, \
# sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_heavy_sep_light"

#COLUMNS="BType, sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_aa_heavy, sequence_alignment_heavy, sequence_alignment_heavy_sep_light"

# paired
#COLUMNS="Species, Disease, BType, Isotype_light, locus_heavy, locus_light, v_call_heavy, d_call_heavy, j_call_heavy, sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_aa_heavy, sequence_alignment_heavy, sequence_alignment_heavy_sep_light"
COLUMNS="sequence_alignment_aa_heavy, sequence_alignment_aa_light, BType, locus_light, v_call_heavy, v_call_light, j_call_heavy, j_call_light, sequence_alignment_heavy_sep_light"


# unpaired heavy and unpaired light columns
#COLUMNS="locus, v_call, d_call, j_call, sequence_alignment_aa, fwr1_aa, cdr1_aa, fwr2_aa, cdr2_aa, fwr3_aa, fwr4_aa, cdr3_aa, BType, Disease, Age, Species, Vaccine"

TABLE_NAME="all_human_paired"
#TABLE_NAME="human_unpaired_novaccine_nodisease_heavy"
#v_call_heavy	d_call_heavy	j_call_heavy
#TABLE_NAME="human_unpaired_novaccine_nodisease_light"


# Run SQLite commands
sqlite3 $DATABASE_PATH <<EOF
.mode csv
.output $OUTPUT_FILE
SELECT $COLUMNS FROM $TABLE_NAME;
.output stdout
.quit
EOF

echo "Data extracted to $OUTPUT_FILE"

