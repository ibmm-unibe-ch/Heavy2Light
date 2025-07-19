#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=json_gpt_2_10000
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/logs/json_gpt_2_10000%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/logs/json_gpt_2_10000%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/analyse_json_file_from_pyir.py

