#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=TRAIN
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/TRAIN_full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/TRAIN_full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python full_eval_similarity_blosum_perplexity_gpt.py
