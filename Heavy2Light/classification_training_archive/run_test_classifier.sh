#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=test_classifier
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/logs/test_classifier_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/logs/test_classifier_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/src/test_classifier.py
