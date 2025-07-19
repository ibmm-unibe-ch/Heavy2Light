#!/bin/bash


#SBATCH --gres=gpu:alphafold:1
#SBATCH --job-name=complete
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/logs/complete%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/logs/complete%j.e

#MODEL_TYPE="gpt2_unpaired_light"
MODEL_TYPE="bert_unpaired_heavy"

# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


#sequence_heavy,locus_heavy,v_call_heavy_x,sequence_alignment_heavy,sequence_alignment_aa_heavy_x,germline_alignment_aa_heavy,cdr3_aa_heavy,sequence_light,locus_light_x,
#v_call_light_x,sequence_alignment_light,sequence_alignment_aa_light_x,germline_alignment_aa_light,cdr3_aa_light,sequence_alignment_heavy_sep_light,BType_x,Disease,Species,Subject,Author,Age,sequence_alignment_aa_light_1,
#generated_sequence_light,input_heavy_sequence,BLOSUM_score,similarity,perplexity,calculated_blosum,calculated_similarity,general_v_gene_heavy,general_v_gene_light,
#v_gene_heavy_family_x,v_gene_light_family_x,alignment_germline,group,sequence_alignment_aa_heavy_y,sequence_alignment_aa_light_y,BType_y,locus_light_y,v_call_heavy_y,v_call_light_y,j_call_heavy,j_call_light,v_gene_heavy_family_y,v_gene_light_family_y,j_gene_heavy_family,j_gene_light_family
#   --model_path "/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-6058816" \
#  --model_path "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391" \

  # --include_for_analysis "Memory-B-Cells" "Naive-B-Cells" "RV+B-Cells" \
  # --exclude_from_plots "RV+B-Cells"

#   --remove_wrong_heavy_values

  # --include_for_analysis "Memory-B-Cells" "Naive-B-Cells" "RV+B-Cells" \
  # --exclude_from_plots "RV+B-Cells" \
  # --new_label_name_1 "Memory B cells" \
  # --new_label_name_2 "Naive B cells" \


eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/src/lda_pca_tsne_umap_combined.py \
  --input_file "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/data/bert2gpt_df_merged_final_test_set_with_full_paired.csv" \
  --model_path "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391" \
  --model_type $MODEL_TYPE \
  --target_column "j_gene_heavy_family" \
  --sequence_column "sequence_alignment_aa_heavy_x" \
  --output_dir "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/plots/final_2/$MODEL_TYPE" \
  --plot_prefix "full_${MODEL_TYPE}_j_gene_correct_order" \
  --legend_title "J gene family" \
  --plot_height 8 \
  --plot_width 8 \
  --title_fontsize 26 \
  --label_fontsize 22 \
  --tick_fontsize 17 \
  --methods tsne \
  --format png \
  --marker_scale 2.5 \
  --dpi 600 \
  --x_min -135 \
  --x_max 135 \
  --y_min -130 \
  --y_max 125 \
  --xaxis_bins 7 \
  --yaxis_bins 8 \
  --transparent \
  --legend_order IGHJ1 IGHJ2 IGHJ3 IGHJ4 IGHJ5 IGHJ6

#   --legend_order IGHV1 IGHV2 IGHV3 IGHV4 IGHV5 IGHV6 IGHV7
# IGHJ1 IGHJ2 IGHJ3 IGHJ4 IGHJ5 IGHJ6

