#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=plabdab_human_healthy_no_vac_allocated_test_no_identifiers
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/logs/plabdab_human_healthy_no_vac_allocated_test_no_identifiers%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/logs/plabdab_human_healthy_no_vac_allocated_test_no_identifiers%j.e

# json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/all_seqs_generated_gpt2_sequences_10000.json"
# json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/pyir_output/bert2gpt_full_complete_ids.json"

# output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/lightgpt_violin_plots_and_distributions"
# output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/bert2gpt_violin_plots_and_distributions"


#json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/train_val_test_fastas/plabdab_human_healthy_no_vac_allocated_test_no_identifiers.json"

#output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/train_val_test_datasets_pyir/plabdab_human_healthy_no_vac_allocated_test_no_identifiers"
    
    # json_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/matching_seqs_multiple_light_seqs_203276_cls_predictions.json"

    # #output_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/plots/train_val_test_datasets_pyir/plabdab_human_healthy_no_vac_allocated_test_no_identifiers"
    # output_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/plots/full_eval_generate_multiple_light_seqs_203276_cls_predictions_merged_genes"


eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/create_violin_plots_and_distributions.py \
    --input_file "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/matching_seqs_only_first_gen_multiple_light_seqs_203276_cls_predictions.json" \
    --output_dir "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/plots/full_eval_generate_multiple_light_seqs_203276_cls_predictions_merged_genes" \
    --save_prefix matching_seqs_multiple_light_seqs_only_first_203276_cls_predictions \
    --transparent \
    --color_palette Set1 \
    --mean_line_width 0.3 \
    --figure_width 8 \
    --figure_height 6 \
    --font_size 22 \
    --title_font_size 26 \
    --dpi 600 \
    --percent_identity_ylim 0 100 \
    --length_ylim 0 110 \


