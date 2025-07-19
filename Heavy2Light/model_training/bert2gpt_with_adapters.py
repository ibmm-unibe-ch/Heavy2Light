# environment: adap_2
# use pip install git+https://github.com/adapter-hub/adapters.git for installing adapters!
# commands for environment:
# pip install git+https://github.com/adapter-hub/adapters.git
# pip install wandb
# pip install pandas
# pip install datasets
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install chardet
# pip install accelerate -U
# no need to install transformers since adapters will automatically install the correct version needed

from transformers import EncoderDecoderModel, Seq2SeqTrainingArguments, GenerationConfig, AutoTokenizer
from adapters import BnConfig, Seq2SeqAdapterTrainer, init
from Bio.pairwise2 import format_alignment
from typing import Dict, List, Union
from Bio import pairwise2
import wandb
import torch
import pandas as pd
from datasets import Dataset
import os
import re


# print device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


############################################ BERT2GPT with adapters ############################################
# Load the pre-trained models
############################################ light heavy model / light light model ############################################
# decoder: light model smaller config and encoder: heavy model smaller config
# decoder path epoch 40: /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520
# encoder smaller model epoch 19: /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391

#small_heavy_encoder="/storage/homefs/lb24i892/bert2gpt_translation/bert_encoder_heavy/checkpoint-117674391"
small_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
#small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

#big_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-55740501"


#light_gpt_decoder = "/storage/homefs/lb24i892/gpt_light_model_unpaired/model_outputs/small_gpt2_test_light_seqs_unp/checkpoint-74"
#light_gpt_decoder ="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/gpt_decoder/checkpoint-74"
#light_gpt_decoder = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/gpt_decoder/small_gpt2_test_light_seqs_unp_lr_5e-4/checkpoint-47"
#light_gpt_decoder = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/gpt_decoder/old_tokenizer/checkpoint-74"

# not fully trained model with same tokenizer as light bert model unpaired
#light_gpt_decoder = "/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/small_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_5/checkpoint-20"

# trained gpt model with full data unpaired epoch 3 (but not fully trained yet)
#light_gpt_decoder = "/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-443328"

# trained gpt model with full data unpaired epoch 41
#light_gpt_decoder = "/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-6058816"

# trained gpt model with full data unpaired epoch 80
light_gpt_decoder="/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-11822080"


model = EncoderDecoderModel.from_encoder_decoder_pretrained(small_heavy_encoder, light_gpt_decoder , add_cross_attention=True)
init(model)

# Count the layers (modules)
layer_count = sum(1 for _ in model.modules())
print(f"Total number of layers (including activations, etc.): {layer_count}")

# Count layers with trainable parameters
trainable_layer_count = sum(1 for _ in model.children() if list(_.parameters()))
print(f"Number of trainable layers: {trainable_layer_count}")

############################################# add adapters ############################################

config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

model.add_adapter("heavy2light_adapter", config=config)
model.set_active_adapters("heavy2light_adapter")
model.train_adapter("heavy2light_adapter")

model.named_parameters

# Loop through all parameters, print and enable gradient computation for 'crossattention' parameters
for name, param in model.named_parameters():
    if re.match(".*crossattention.*", name):
        # print(f"Parameter Name: {name}")
        # print(f"   Requires Grad: {param.requires_grad}")
        # print(f"   Parameter Shape: {param.size()}")
        # print(f"   Parameter Device: {param.device}")
        param.requires_grad = True
        #print(f"   Requires Grad: {param.requires_grad}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of trainable parameters: {count_trainable_params(model)}")
print_trainable_parameters(model)

# Load BERT tokenizer for encoder
bert_tokenizer = AutoTokenizer.from_pretrained(small_heavy_encoder)

# Load GPT tokenizer for decoder
gpt_tokenizer = AutoTokenizer.from_pretrained(light_gpt_decoder)

# Common training hyperparameters
batch_size = 64
num_train_epochs = 65
learning_rate = 1e-7
weight_decay = 0.1

# Common configurations
#flag = "PLAbDab"
#dataset = "paired_oas_100_70_50"
dataset = "clustered_light_seqs_90_pident_70_alloc"
#dataset = "oas_lambda_only_balanced"
#dataset = "kappa_only_30pident_alloc"
#dataset = "lambda_only_30pident_alloc"

dataset_size = "full"
translation_model = "bert2gpt"
max_length = 110
min_length = 100
max_new_tokens = 115
num_return_sequences = 1  

# Choose decoding strategy
decoding = "nucleus"  # Options: "DoLa", "nucleus", "contrastive", "diverse_beam_search", "beam_search"

# Decoding strategy parameters
decoding_params = {}

if decoding == "DoLa":
    decoding_params = {
        'dola_layers': "high", # Options: "low", "high" or e.g. [320,340]
        'repetition_penalty': 1.2, # recommended is 1.2
    }
elif decoding == "nucleus":
    decoding_params = {
        'do_sample': True,
        'top_p': 0.85,
        'top_k': 0,  # Typically set to 0 for pure nucleus sampling
        'temperature': 0.8, # optional
    }
elif decoding == "contrastive":
    decoding_params = {
        'penalty_alpha': 0.8,
        'top_k': 5,
    }
elif decoding == "diverse_beam_search":
    decoding_params = {
        'num_beams': 5,
        'num_beam_groups': 5,
        'diversity_penalty': 1.0,
        'early_stopping': True,
    }
elif decoding == "beam_search":
    decoding_params = {
        'num_beams': 4,
    }
else:
    raise ValueError(f"Unsupported decoding strategy: {decoding}")


# Build run_name using the configurations
run_name = (
    f"gpu_test_{dataset}_{dataset_size}_{translation_model}_{decoding}_{dataset}_num_epochs_{num_train_epochs}_lr_{learning_rate}_unpaired_epo_80"
)

print(f"Training model with run_name: {run_name}")
print(f"Using decoding parameters: {decoding_params}")

# Common generation parameters
common_generation_params = {
    'num_return_sequences': num_return_sequences,
    'max_new_tokens': max_new_tokens,
    'max_length': max_length,
    'min_length': min_length,
    'pad_token_id': gpt_tokenizer.pad_token_id,
    'eos_token_id': gpt_tokenizer.eos_token_id,
    'decoder_start_token_id': gpt_tokenizer.bos_token_id,
    'use_cache': True,
    'output_scores': True,
    'output_hidden_states': True,
    'return_dict_in_generate': True,
}

# Combine common parameters with decoding strategy parameters
generation_params = {**common_generation_params, **decoding_params}

# # Handle special parameters not part of GenerationConfig
# special_params = {}
# for key in ['dola_layers']:
#     if key in generation_params:
#         special_params[key] = generation_params.pop(key)

# Create GenerationConfig object
generation_config = GenerationConfig(**generation_params)

# Save the generation configuration
# generation_config_name = f"gpt_generation_config_{decoding}_max_new_tokens_{max_new_tokens}"
# generation_config.save_pretrained("generation_config", f"{generation_config_name}.json")

# Load the generation configuration if needed
#generation_config = GenerationConfig.from_pretrained("generation_config", f"{generation_config_name}.json")


#output_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/{run_name}"
#output_dir = f"/storage/homefs/lb24i892/bert2gpt_translation/model_outputs/{run_name}"
output_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
#logging_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/{run_name}_logging"
#logging_dir = f"/storage/homefs/lb24i892/bert2gpt_translation/model_outputs/{run_name}_logging"
logging_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}_logging"

# gpt_tokenizer.bos_token = bert_tokenizer.cls_token
# gpt_tokenizer.eos_token = bert_tokenizer.sep_token

# Set up the Seq2Seq model configuration
model.config.decoder_start_token_id = gpt_tokenizer.bos_token_id
model.config.eos_token_id = gpt_tokenizer.eos_token_id
model.config.pad_token_id = gpt_tokenizer.pad_token_id
model.config.bos_token_id = gpt_tokenizer.bos_token_id

# add maske token to special tokens in gpt tokenizer
model.config.mask_token = gpt_tokenizer.mask_token

# Ensure the decoder's vocab size is correctly set
model.config.decoder.vocab_size = model.config.decoder.vocab_size

print("GPT Tokenizer Special Tokens and IDs:")
print(f"bos_token: {gpt_tokenizer.bos_token}, id: {gpt_tokenizer.bos_token_id}")
print(f"eos_token: {gpt_tokenizer.eos_token}, id: {gpt_tokenizer.eos_token_id}")
print(f"pad_token: {gpt_tokenizer.pad_token}, id: {gpt_tokenizer.pad_token_id}")


# Check the tokens corresponding to IDs 0 and 1
token_0 = gpt_tokenizer.decode(0)
token_1 = gpt_tokenizer.decode(1)
print(f"Token for ID 0: '{token_0}'")
print(f"Token for ID 1: '{token_1}'")

############################################ Training Arguments ############################################

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch", # set to epoch for full data
    logging_strategy="epoch",
    #logging_steps=200, # set to 200 for full data
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    #generation_max_length=110,
    report_to="wandb",
    run_name=run_name,
    generation_config=generation_config,
    #eval_steps=1, # for full data, set to 2000
    #save_steps=100, # comment out for full data
)

    
# Create directories if they do not exist
os.makedirs(training_args.output_dir)
os.makedirs(training_args.logging_dir)

# Log in to Weights & Biases
#wandb.login()


wandb.init(project="bert2gpt_translation", name=run_name)

############################################ load the data ############################################


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())

    sequences = []
    for entry in data:
        split_entry = entry.split('[SEP]') # otherwise [SEP] with spaces
        if (len(split_entry) == 2):
            sequences.append(split_entry)
        else:
            print(f"Skipping invalid entry: {entry}")

    df = pd.DataFrame(sequences, columns=['heavy', 'light'])
    return df

# on vader
# SMALL dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED with dataset: human healthy, no vaccine, no disease and PLAbDab unique paired seqs
#train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_train_no_identifiers_small.txt"

# spaces SMALL dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED with dataset: human healthy, no vaccine, no disease and PLAbDab unique paired seqs
#train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_train_no_identifiers_spaces_small.txt"

# full dataset
#train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_train_no_identifiers.txt"

# # 100 70 50 dataset
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids.txt"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids.txt"
# test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids.txt"

# on ubelix
#train_file_path="/storage/homefs/lb24i892/bert2gpt_translation/plabdab_human_healthy_no_vac_allocated_train_no_identifiers_spaces_small.txt"

# on vader
#val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_val_no_identifiers_small.txt"

#val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_val_no_identifiers_spaces_small.txt"

# full dataset
#val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_val_no_identifiers.txt"

# # combined OAS and pairedngs db
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/train_val_test_oas_and_pairedabngs/combined_train_dataset_oas_pairedabngs_no_dupl.txt"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/train_val_test_oas_and_pairedabngs/combined_val_dataset_oas_pairedabngs_no_dupl.txt"

# # combined OAS and pairedngs db
# test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/train_val_test_oas_and_pairedabngs/combined_test_dataset_oas_pairedabngs_no_dupl.txt"

# # oas db only kappa and balanced
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/train_data_only_seq.csv"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/validation_data_only_seq.csv"

# oas db kappa only everything paired without duplicates
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_full_paired/train_data_only_seq.csv"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_full_paired/validation_data_only_seq.csv"

# test data oas kappa only balanced
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/test_data_only_seq.csv"

# test data oas kappa only full paired without duplicates
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_full_paired/test_data_only_seq.csv"

# lambda only balanced dataset OAS
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/train_data_lambda_only_seq.csv"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/validation_data_lambda_only_seq.csv"

# # kappa only balanced dataset 30% allocation
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/30_pident_assignment/datasets/kappa_only_balanced_alloc_30_pident_train_no_id.txt"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/30_pident_assignment/datasets/kappa_only_balanced_alloc_30_pident_val_no_id.txt"

# lambda only balanced dataset 30% allocation
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/datasets_30pident_allocation/datasets/lambda_only_balanced_alloc_30_pident_train_no_id.txt"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/datasets_30pident_allocation/datasets/lambda_only_balanced_alloc_30_pident_val_no_id.txt"

# test data lambda only balanced
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/test_data_lambda_only_seq.csv"

# # test file balanced kappa only 30% allocation
# test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/30_pident_assignment/datasets/kappa_only_balanced_alloc_30_pident_test_no_id.txt"

#test file lambda only balanced 30% allocation
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/datasets_30pident_allocation/datasets/lambda_only_balanced_alloc_30_pident_test_no_id.txt"

# on ubelix
#val_file_path="/storage/homefs/lb24i892/bert2gpt_translation/plabdab_human_healthy_no_vac_allocated_val_no_identifiers_spaces_small.txt"

# # paired sequences with clustered light seqs 80% identity and allocation based on 30%
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc_train_heavy_sep_light.csv"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc_val_heavy_sep_light.csv"
# test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc_test_heavy_sep_light.csv"

# # paired sequences with clustered light seqs 90% identity and allocation based on 30%
# train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_30_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_30_alloc_train_heavy_sep_light.csv"
# val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_30_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_30_alloc_val_heavy_sep_light.csv"
# test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_30_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_30_alloc_test_heavy_sep_light.csv"

# paired sequences with clustered light seqs 90% identity and allocation based on 70%
train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc_train_heavy_sep_light.csv"
val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc_val_heavy_sep_light.csv"
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc_test_heavy_sep_light.csv"



train_df = load_data(train_file_path)
val_df = load_data(val_file_path)
#test_df = load_data(test_file_path)

encoder_max_length = 160
decoder_max_length = 160

def process_data_to_model_inputs(batch):
    # Tokenize the encoder inputs using the BERT tokenizer
    encoder_inputs = bert_tokenizer(
        batch["heavy"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt"
    )

    # We reserve space for BOS and EOS tokens by using decoder_max_length - 2 initially
    decoder_inputs = gpt_tokenizer(
        batch["light"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length - 2,
        return_tensors="pt"
    )

    bos_id = gpt_tokenizer.bos_token_id
    eos_id = gpt_tokenizer.eos_token_id
    pad_id = gpt_tokenizer.pad_token_id

    # Convert to lists to manipulate
    decoder_input_ids_list = decoder_inputs["input_ids"].tolist()
    new_decoder_input_ids = []

    for seq in decoder_input_ids_list:
        # Remove trailing pad tokens
        while seq and seq[-1] == pad_id:
            seq.pop()

        # Now insert BOS at the start and EOS at the end of the actual sequence
        seq = [bos_id] + seq + [eos_id]

        # Re-pad the sequence to the full decoder_max_length
        if len(seq) > decoder_max_length:
            seq = seq[:decoder_max_length]
        else:
            while len(seq) < decoder_max_length:
                seq.append(pad_id)

        new_decoder_input_ids.append(seq)

    # Convert back to tensors
    decoder_input_ids = torch.tensor(new_decoder_input_ids, dtype=torch.long)
    decoder_attention_mask = (decoder_input_ids != pad_id).long()

    # Assign encoder inputs
    batch["input_ids"] = encoder_inputs["input_ids"]
    batch["attention_mask"] = encoder_inputs["attention_mask"]

    # Assign decoder inputs
    batch["decoder_input_ids"] = decoder_input_ids
    batch["decoder_attention_mask"] = decoder_attention_mask

    # Prepare labels (identical to decoder_input_ids, but with pad replaced by -100)
    labels = decoder_input_ids.clone()
    labels[labels == pad_id] = -100
    batch["labels"] = labels

    return batch


# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['heavy', 'light']])
val_dataset = Dataset.from_pandas(val_df[['heavy', 'light']])
#test_dataset = Dataset.from_pandas(test_df[['heavy', 'light']])


train_data = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
)

# "decoder_input_ids",
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
)

val_data = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
)

# "decoder_input_ids",
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
)


# print heavy and light seq from the first example in the training data (train_dataset)
print(f"first example heavy and light seq {train_dataset[0]}, {train_dataset[1]}")

# Print a few examples from the tokenized dataset
for example in train_data.select(range(1)):
    print(example)

# print first 10 encoder input ids
print(f"first 10 encoder input ids: {train_data['input_ids'][:10]}")
print(f"first 10 encoder attention mask: {train_data['attention_mask'][:10]}")
print(f"first 10 labels: {train_data['labels'][:10]}")

# print first 10 decoder input ids
print(f"first 10 decoder input ids: {train_data['decoder_input_ids'][:10]}")
print(f"first 10 decoder attention mask: {train_data['decoder_attention_mask'][:10]}")
print(f"first 10 labels: {train_data['labels'][:10]}")

# Assuming train_data['labels'][0] is a list of token IDs
labels = train_data['labels'][0]
print(f"labels: {labels}")

# Filter out -100
valid_label_ids = [token_id for token_id in labels if token_id >= 0]

# Now decode only valid token IDs
decoded_text_gpt = gpt_tokenizer.decode(valid_label_ids)
decoded_text_bert = bert_tokenizer.decode(valid_label_ids)
print(f"decoded light sequence with gpt tokenizer: {decoded_text_gpt}")

print(f"decoded light sequence with bert tokenizer without spaces: {decoded_text_bert.replace(' ', '')}")
print(f"decoded light sequence with bert tokenizer: {decoded_text_bert}")


# Initialize the trainer
trainer = Seq2SeqAdapterTrainer(
    model=model,
    tokenizer=gpt_tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    adapter_names=["heavy2light_adapter"],
)


# Train the model
trainer.train()
#trainer.evaluate()

model.to(device)

#model_output_path = os.path.join("/storage/homefs/lb24i892/bert2gpt_translation/model_outputs/", run_name)
model_output_path = os.path.join("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/", run_name)
adapter_output_path = f"{model_output_path}/final_adapter"

os.makedirs(adapter_output_path)

# Save model
model.save_pretrained(model_output_path)
# Save adapter
model.save_adapter(adapter_output_path, "heavy2light_adapter")
#decoder.save_adapter(output_path, "decoder_adapter")

# Load your test data
#test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# on vader: small test dataset all human paired, no duplicates + PLAbDab unique paired sequences
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_test_no_identifiers_small.txt"
# on ubelix:
#test_file_path="/storage/homefs/lb24i892/bert2gpt_translation/plabdab_human_healthy_no_vac_allocated_test_no_identifiers_spaces_small.txt"

# small test file with no spaces
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_small.txt"

test_df = load_data(test_file_path)


# extract the light sequences from test_df
heavy_sequences = test_df["heavy"]
true_light_sequences = test_df["light"]
print("first true light sequence: ", true_light_sequences[0])

#print("light_sequences: ", light_sequences)
#print(f"length of light sequences {len(light_sequences)}")

generated_light_seqs = []

# average similarity percentage
total_similarity_percentage = []

# Iterate through each sequence in the test dataset
for i in range(50):
    inputs = bert_tokenizer(heavy_sequences[i], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    generated_seq = model.generate(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               output_scores=True, 
                               return_dict_in_generate=True,
                               generation_config=generation_config,
                               bad_words_ids=[[4]], # bad words ids to avoid generating (<mask> token in gpt which is not needed but necessary for the bert tokenizer (id: 4))
                            )
    
    # Access the first element in the generated sequence
    sequence = generated_seq["sequences"][0]

    # Convert the generated IDs back to text
    generated_text = gpt_tokenizer.decode(sequence, skip_special_tokens=True)
    true_light_seq = true_light_sequences[i].replace(" ", "")

    print("decoded light sequence: ", generated_text)
    print("true light sequence: ", true_light_seq)

    generated_light_seqs.append(generated_text)
    
    # Determine the length of the shorter sequence
    min_length = min(len(generated_text), len(true_light_seq))
    print(f"min_length:, {min_length}")

    if min_length == 0:
        raise ValueError("One of the sequences (generated or true) is empty! Check the true seqs data or put min_length > 0 in the generation config.")
    
    # Calculate the number of matches
    matches = sum(res1 == res2 for res1, res2 in zip(generated_text, true_light_seq))
    print(f"matches:, {matches}")

    
    # Calculate the similarity percentage
    similarity_percentage = (matches / min_length) * 100
    
    print(f"similarity percentage: {similarity_percentage}")

    # add similarity percentage to list
    total_similarity_percentage.append(similarity_percentage)


# print average similarity percentage
similarity_percentage = sum(total_similarity_percentage) / len(total_similarity_percentage)
print(f"average similarity percentage: {similarity_percentage}")

print("generated_light_seqs:")
# print each generated sequence on new line
for seq in generated_light_seqs:
    print(seq)