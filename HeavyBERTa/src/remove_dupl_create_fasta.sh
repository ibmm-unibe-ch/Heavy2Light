#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=create_fasta_csv
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/outputs/create_fasta_sec_clust%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/outputs/create_fasta_sec_clust%j.e

# remove duplicates in full heavy seq:
awk -F ',' '!seen[$3]++' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_no_duplicates.txt

# Create fasta file from the filtered seqs for 2. step clustering of full heavy seq unpaired without dupl:
awk -F, '{print ">" $1 "\n" $NF}'  /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_no_duplicates.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_no_duplicates.fasta

