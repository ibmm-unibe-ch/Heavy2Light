# used env: adap_2
from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
import torch
import pandas as pd
from adapters import init
from Bio.Align import substitution_matrices
import numpy as np
from tqdm import tqdm
from Bio import pairwise2
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
    """
    Initialize the model, tokenizer, and generation configuration.
    
    Args:
        model_path (str): Path to the model.
        tokenizer_path (str): Path to the tokenizer.
        adapter_path (str): Path to the adapter.
        generation_config_path (str): Path to the generation configuration.
        device (torch.device): Device to run the model on.
    
    Returns:
        model (EncoderDecoderModel): Initialized model.
        tokenizer (AutoTokenizer): Initialized tokenizer.
        generation_config (GenerationConfig): Generation configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    init(model)
    print(f"model is on device: {model.device}")
    #model.to(device)
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    
    return model, tokenizer, generation_config


run_name="full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"
tokenizer_path=f"{model_path}/checkpoint-367750"
adapter_path=f"{model_path}/final_adapter"
generation_config_path=model_path
adapter_name="heavy2light_adapter"

# # bert2gpt all data (OAS and pairedabngs)
# run_name = "full_PLAbDab_oas_paired_abngs_diverse_beam_search_bert2gpt_diverse_beam_search_oas_paired_abngs_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-905950"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt kappa only on balanced dataset (v genes light)
# run_name = "full_PLAbDab_oas_kappa_only_balanced_diverse_beam_search_bert2gpt_diverse_beam_search_oas_kappa_only_balanced_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-74100"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt lambda only on balanced dataset (v genes light)
# run_name = "oas_lambda_only_balanced_full_bert2gpt_diverse_beam_search_oas_lambda_only_balanced_num_epochs_50_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-85550"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt kappa only full paired oas dataset without duplicates 
# run_name = "oas_kappa_only_all_full_bert2gpt_diverse_beam_search_oas_kappa_only_all_num_epochs_50_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-681550"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt kappa only 30 pident clustering
# run_name = "kappa_only_30pident_alloc_full_bert2gpt_diverse_beam_search_kappa_only_30pident_alloc_num_epochs_50_lr_1e-06_unpaired_epo_80-2"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-74100"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt lambda only full paired oas dataset without duplicates  
# run_name = "lambda_only_30pident_alloc_full_bert2gpt_diverse_beam_search_lambda_only_30pident_alloc_num_epochs_50_lr_1e-06_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-85550"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt model with epoch 80 and human healthy no vac + plabdab database 201923
# run_name = "healthy_human_PLAbDab_full_bert2gpt_diverse_beam_search_healthy_human_num_epochs_50_lr_1e-06_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-367750"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt big encoder model with epoch 80 and human healthy no vac + plabdab database 201923
# run_name = "big_encoder_healthy_human_PLAbDab_full_bert2gpt_diverse_beam_search_healthy_human_num_epochs_50_lr_1e-06_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-367750"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt 80 pident clustering 30 pident allocation
# run_name = "clustered_light_seqs_80_pident_30_alloc_full_bert2gpt_diverse_beam_search_0.6_0.6_clustered_light_seqs_80_pident_30_alloc_num_epochs_50_lr_1e-06_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-86850"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # bert2gpt 90 pident clustering 70 pident allocation
# run_name = "clustered_light_seqs_90_pident_70_alloc_full_bert2gpt_nucleus_clustered_light_seqs_90_pident_70_alloc_num_epochs_65_lr_1e-07_unpaired_epo_80"
# model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
# tokenizer_path = f"{model_path}/checkpoint-403195"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)


# Load small test data
#test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# load FULL test data
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated.txt"

# NEW DATA (human healthy no diseases + plabdab)
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_test_no_identifiers.txt"

# VALIDATION NEW DATA (human healthy no diseases + plabdab) -> check if we see trimodal distribution in this set as well
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_val_no_identifiers.txt"

# TRAINING SET NEW DATA (human healthy no diseases + plabdab) -> check if we see trimodal distribution in this set as well
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_train_no_identifiers.txt"


# SMALL NEW DATA (human healthy no diseases + plabdab)
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_test_no_identifiers_small.txt"

# test data all human paired + plabdab
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/train_val_test_datasets/human_healthy_all_diseases_plabdab_test_no_identifiers_spaces.txt"

# human_healthy_and_covid
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/train_test_val_datasets/human_healthy_covid_allocated__test_no_identifiers_spaces.txt"

# small test data
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/train_test_val_datasets/human_healthy_covid_allocated__test_no_identifiers_spaces_small.txt"

# oas db + paired abngs db
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_abngs_db/train_val_test_oas_and_pairedabngs/combined_test_dataset_oas_pairedabngs_no_dupl.txt"

# test set kappa balanced only from oas 
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/test_data_only_seq.csv"

# test set lambda balanced only from oas 
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/test_data_lambda_only_seq.csv"

# test set kappa only full paired oas dataset without duplicates
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_full_paired/test_data_only_seq.csv"

# kappa only balanced allocation based on 30 pident clustering
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/kappa_only_translation_balanced_v_gene_families/data/kappa_only_balanced/30_pident_assignment/datasets/kappa_only_balanced_alloc_30_pident_test_no_id.txt"

# lambda only balanced allocation based on 30 pident clustering
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/lambda_only_translation_balanced_v_gene_families/data/datasets_30pident_allocation/datasets/lambda_only_balanced_alloc_30_pident_test_no_id.txt"

# paired sequences with clustered light seqs 80% identity and allocation based on 30%
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc/full_paired_oas_no_dupl_light_seqs_80_clu_rep_30_alloc_test_heavy_sep_light.csv"

# paired sequences with clustered light seqs 90% identity and allocation based on 70%
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc_test_heavy_sep_light.csv"
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/check_pairings_gen_sequences/clustered_light_seqs_datasets/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc/full_paired_oas_no_dupl_light_seqs_90_clu_rep_70_alloc_test_heavy_sep_light_small.csv"


print(f"Fully evaluating model with run name: {run_name}")

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())
    sequences = []
    for entry in data:
        split_entry = entry.split('[SEP]')
        if len(split_entry) == 2:
            sequences.append(split_entry)
        else:
            print(f"Skipping invalid entry: {entry}")
    df = pd.DataFrame(sequences, columns=['heavy', 'light'])
    return df

test_df = load_data(test_file_path)
heavy_sequences = test_df["heavy"]
true_light_sequences = test_df["light"]


# # Calculate BLOSUM scores
# def calculate_blosum_score(true_seq, generated_seq, matrix):
#     score = 0
#     matches = 0
#     generated_seq = generated_seq.replace(" ", "")
#     true_seq = true_seq.replace(" ", "")
#     min_length = min(len(true_seq), len(generated_seq))
#     for i in range(min_length):
#         pair = (true_seq[i], generated_seq[i])
#         if pair in matrix:
#             score += matrix[pair]
#         elif (pair[1], pair[0]) in matrix:
#             score += matrix[(pair[1], pair[0])]
#         if true_seq[i] == generated_seq[i]:
#             matches += 1
#     similarity_percentage = (matches / min_length) * 100
#     return score, similarity_percentage


# Use the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")


# def calculate_blosum_score_with_global_alignment(seq1, seq2, blosum_matrix):
#     # Clean sequences to remove invalid characters
#     seq1 = seq1.replace(' ', '')
#     seq2 = seq2.replace(' ', '')
    
#     # Perform global alignment
#     alignments = pairwise2.align.globalds(seq1, seq2, blosum_matrix, -10, -4)
#     best_alignment = alignments[0]
    
#     # Extract aligned sequences and calculate similarity
#     aligned_seq1, aligned_seq2, score, start, end = best_alignment
#     matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
#     similarity_percentage = (matches / max(len(seq1), len(seq2))) * 100
    
#     return score, similarity_percentage


def calculate_blosum_score_with_global_alignment(seq1, seq2, blosum_matrix):
    # Clean sequences to remove invalid characters
    valid_residues = "ACDEFGHIKLMNPQRSTVWY"
    seq1 = re.sub(f"[^{valid_residues}]", "", seq1.upper())
    seq2 = re.sub(f"[^{valid_residues}]", "", seq2.upper())
    
    # Check for empty sequences
    if not seq1 or not seq2:
        raise ValueError("One or both sequences are empty after cleaning.")
    
    # Perform global alignment
    alignments = pairwise2.align.globalds(seq1, seq2, blosum_matrix, -10, -4)
    if not alignments:
        raise ValueError("No alignments were found.")
    
    best_alignment = alignments[0]
    
    # Extract aligned sequences and calculate similarity
    aligned_seq1, aligned_seq2, score, start, end = best_alignment
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
    similarity_percentage = (matches / max(len(seq1), len(seq2))) * 100
    
    return score, similarity_percentage

# # test global alignment function
# true_seq="DIELTQSPAIMSASLGEKVTMSCRASSSVNFIYWYQQKSDASPKLWVYYTSHLPPGVPARFSGSGSGNSYSLTISSMEGEDAATYYCQQFTSSPFTFGSGTKLEIK"
# gen_seq="DIVMTQSPSSLAVSAGEKVTMSCKSSQSLLNSRTRKNYLAWYQQKPGQSPKLLIYWASTRESGVPDRFTGSGSGTDFTLTISSVQAEDLAVYYCKQSYNLRTFGGGTKLEIK"
# score, similarity_percentage = calculate_blosum_score_with_global_alignment(true_seq, gen_seq, blosum62)
# print(f"score: {score}, similarity_percentage: {similarity_percentage}")


# Updated loop for sequence generation and BLOSUM score calculation
scores = []
similarities = []
perplexities = []

for i in tqdm(range(len(heavy_sequences)), desc="Processing sequences"):
    # Generate sequence
    inputs = tokenizer(
        heavy_sequences[i],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    model.to(device)
    print(f"model is on device {model.device}")
    generated_seq = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=150,
        output_scores=True,
        return_dict_in_generate=True,
        generation_config=generation_config,
        bad_words_ids=[[4]], # bad words ids to avoid generating (<mask> token in gpt which is not needed but necessary for the bert tokenizer (id: 4))
    )
    sequence = generated_seq["sequences"][0]
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    
    # Calculate BLOSUM score and similarity using global alignment
    score, similarity_percentage = calculate_blosum_score_with_global_alignment(
        true_light_sequences[i], generated_text, blosum62
    )
    scores.append(score)
    similarities.append(similarity_percentage)
    
    # # Calculate perplexity
    # inputs = tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt").to(device)
    # targets = tokenizer(true_light_sequences[i], padding=True, truncation=True, return_tensors="pt").to(device)
    
    # with torch.no_grad():
    #     outputs = model(input_ids=inputs.input_ids, decoder_input_ids=targets.input_ids)
    
    # logits = outputs.logits
    # shift_logits = logits[:, :-1, :].contiguous()
    # shift_labels = targets.input_ids[:, 1:].contiguous()
    
    # loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # target_mask = (shift_labels != tokenizer.pad_token_id).float()
    # loss = loss.view(shift_labels.size()) * target_mask
    
    # log_likelihood = loss.sum(dim=1)
    # perplexity = torch.exp(log_likelihood / target_mask.sum(dim=1)).cpu().detach().numpy()
    
    # perplexities.append(perplexity[0])

    # For perplexity calculation
    with torch.no_grad():
        # Use heavy chain as encoder input
        encoder_inputs = tokenizer(heavy_sequences[i], padding="max_length", truncation=True, 
                                max_length=256, return_tensors="pt").to(device)
        
        # True light chain as target
        decoder_inputs = tokenizer(true_light_sequences[i], padding="max_length", truncation=True,
                                max_length=150, return_tensors="pt").to(device)
        
        # Prepare decoder inputs (shifted right) and labels
        decoder_input_ids = decoder_inputs.input_ids[:, :-1]
        labels = decoder_inputs.input_ids[:, 1:]
        
        # We need to calculate loss manually for per-token perplexity
        outputs = model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        # Get logits and calculate loss manually
        logits = outputs.logits
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Create a mask for non-padding tokens
        target_mask = (labels != tokenizer.pad_token_id).float()
        num_tokens = target_mask.sum().item()
        
        # Calculate negative log likelihood for each token
        nll = 0
        for i_pos in range(labels.size(1)):
            token_probs = probs[:, i_pos, :]
            token_label = labels[:, i_pos]
            token_mask = target_mask[:, i_pos]
            if token_mask.sum() > 0:  # Skip padding positions
                token_prob = token_probs[0, token_label[0]]
                if token_prob > 0:  # Avoid log(0)
                    nll -= torch.log(token_prob).item() * token_mask[0].item()
        
        # Calculate perplexity: exp(average negative log likelihood)
        perplexity = torch.exp(torch.tensor(nll / num_tokens)).item() if num_tokens > 0 else float('inf')
        perplexities.append(perplexity)
    
    # Print results for each sequence pair
    print(f"\nSequence pair {i+1}:")
    print(f"True Sequence: {true_light_sequences[i]}")
    print(f"Generated Sequence: {generated_text}")
    print(f"Input heavy sequence: {heavy_sequences[i]}")
    print(f"BLOSUM Score: {score}")
    print(f"Similarity Percentage: {similarity_percentage}%")
    print(f"Perplexity: {perplexity}")  # Use the current value, not perplexities[0]


# calculate average scores and perplexity
average_blosum_score = np.mean(scores)
average_similarity_percentage = np.mean(similarities)
mean_perplexity = np.mean(perplexities)

print(f"\nAverage BLOSUM Score: {average_blosum_score}")
print(f"Average Similarity Percentage: {average_similarity_percentage}%")
print(f"Mean Perplexity: {mean_perplexity}")

