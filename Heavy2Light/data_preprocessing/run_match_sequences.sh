#!/bin/bash


#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=match_seqs
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/logs/match_seqs_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/logs/match_seqs_%j.e


/home/leab/.juliaup/bin/julia /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/match_sequences.jl



