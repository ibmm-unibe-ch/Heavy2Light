#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=bert2gpt_true_gen_violin
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/logs/bert2gpt_true_gen_violin%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/logs/bert2gpt_true_gen_violin%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/gen_true_create_violin_plots_and_distributions.py \
    --input_file "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/bert2gpt_complete_ids_gen_true.json" \
    --output_dir "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/bert2gpt_similarity_to_true_light/final" \
    --save_prefix "aa_final_full_data_bert2gpt_similarity_to_true_light" \
    --transparent \
    --color_palette Set1 \
    --mean_line_width 0.3 \
    --figure_width 8 \
    --figure_height 6 \
    --font_size 22 \
    --title_font_size 26 \
    --dpi 600 \

