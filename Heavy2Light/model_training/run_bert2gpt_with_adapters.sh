#!/bin/bash
#SBATCH --gres=gpu:alphafold:1
#SBATCH --job-name=90_70_light_seq_clustering_bert2gpt
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/light_seq_clustering_90_70_bert2gpt%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/light_seq_clustering_90_70_bert2gpt%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python bert2gpt_with_adapters.py
