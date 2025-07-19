#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=hea_conf4

#--config_name 'config3.json' \

# paths to full datasets
#    --train_file /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_train_no_ids.txt \
#    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_val_no_ids.txt \

# paths to small datasets
#    --train_file /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_train_no_ids_test_small.txt \
#    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_val_no_ids_test_small.txt \


eval "$(conda shell.bash hook)"
conda init bash
conda activate lea_env
/home/leab/anaconda3/envs/lea_env/bin/python run_mlm.py \
    --model_type 'roberta' \
    --tokenizer_name ./ProteinTokenizer \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_train_no_ids.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_val_no_ids.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 500 \
    --lr_scheduler_type 'linear' \
    --log_level 'info' \
    --seed 42 \
    --data_seed 42 \
    --bf16 False \
    --project_name heavy_model_unpaired_all_seqs \
    --load_best_model_at_end True \
    --metric_for_best_model 'loss' \
    --line_by_line \
    --greater_is_better False \
    --config_name 'config4.json' \
    --report_to 'wandb' \
    --max_seq_length 512 \
    --output_dir ./FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512

