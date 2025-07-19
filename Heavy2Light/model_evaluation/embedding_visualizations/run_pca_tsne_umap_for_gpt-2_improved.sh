#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=umap_gpt_bert
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/logs/umap_improved%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/logs/umap_improved%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/src/pca_tsne_umap_for_gpt-2_improved.py
