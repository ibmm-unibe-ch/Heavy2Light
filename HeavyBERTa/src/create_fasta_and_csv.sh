#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=create_fasta_csv
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/outputs/create_fasta%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/outputs/create_fasta%j.e


## extract rowid and cdrh3 in csv and fasta format (for the clustering)
#sqlite3 -header -csv /ibmm_data2/oas_database/OAS_heavy.db "SELECT ROWID,cdr3_aa,sequence_alignment_aa FROM Bcells_subset_human_unpaired_heavy;" | tee >(awk -F, '{print ">"$1"\n"$2}' > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3.fasta) > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3_full_heavy.csv

sqlite3 -header -csv /ibmm_data2/oas_database/OAS_heavy.db "SELECT ROWID,cdr3_aa,sequence_alignment_aa FROM Bcells_subset_human_unpaired_heavy;" | awk -F, '{print ">"$1"\n"$2}' > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3.fasta