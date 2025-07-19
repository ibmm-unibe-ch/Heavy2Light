#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=lightgpt_light_random_true_heavy_single
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/lightgpt_light_random_true_heavy_single%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/logs/lightgpt_light_random_true_heavy_single%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate chai_env
# --use-msa-server --use-templates-server
chai-lab fold --use-msa-server /ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/data/lightgpt_light_random_true_heavy_single.fasta /ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/chai_outputs/lightgpt_random_heavy


