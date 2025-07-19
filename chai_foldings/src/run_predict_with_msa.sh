#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=chai_msa
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/chai_msa_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/chai_msa_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate chai_env
/home/leab/anaconda3/envs/chai_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/src/predict_with_msas.py
