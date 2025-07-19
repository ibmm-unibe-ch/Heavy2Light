#!/bin/bash

#SBATCH --job-name="clm_full_new_tokenizer"
#SBATCH --gres=gpu:h100:1
#SBATCH --output="/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/logs/clm_full_new_tokenizer_%j.o"
#SBATCH --error="/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/logs/clm_full_new_tokenizer_%j.e"


eval "$(conda shell.bash hook)"
conda init bash
conda activate clm_env
/home/leab/anaconda3/envs/clm_env/bin/python run_clm.py \
    --model_type 'gpt2' \
    --tokenizer_name /ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/gpt_protein_tokenizer \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/train_test_val_datasets/light_all_seqs_train_no_ids.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/train_test_val_datasets/light_all_seqs_val_no_ids.txt \
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
    --project_name gpt_light_model_unpaired \
    --load_best_model_at_end True \
    --metric_for_best_model 'loss' \
    --greater_is_better False \
    --config_name 'config.json' \
    --report_to 'wandb' \
    --output_dir gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_ \
