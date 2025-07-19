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

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

# NEW DATA (human healthy no diseases + plabdab)
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_test_no_identifiers.txt"

# SMALL NEW DATA (human healthy no diseases + plabdab)
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_test_no_identifiers_small.txt"


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



# Use the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")


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


# Updated loop for sequence generation and BLOSUM score calculation with 10 sequences per heavy chain
scores = []
similarities = []
perplexities = []

# Store detailed results for each heavy chain and its generated sequences
detailed_results = []

for i in tqdm(range(len(heavy_sequences)), desc="Processing sequences"):
    heavy_seq = heavy_sequences[i]
    true_light_seq = true_light_sequences[i]
    
    # Prepare inputs for the heavy sequence
    inputs = tokenizer(
        heavy_seq,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    model.to(device)
    print(f"model is on device {model.device}")
    
    # Generate 10 sequences for this heavy chain
    generated_seqs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=150,
        num_return_sequences=10, 
        output_scores=True,
        return_dict_in_generate=True,
        generation_config=generation_config,
        bad_words_ids=[[4]], # bad words ids to avoid generating (<mask> token in gpt which is not needed but necessary for the bert tokenizer (id: 4))
        do_sample=True,  # Enable sampling for diversity
        temperature=1.0,  # You can adjust this for more/less diversity
    )
    
    # Process each of the 10 generated sequences
    heavy_chain_results = {
        'heavy_sequence': heavy_seq,
        'true_light_sequence': true_light_seq,
        'generated_sequences': [],
        'scores': [],
        'similarities': [],
        'perplexities': []
    }
    
    for j in range(10):  # Process all 10 generated sequences
        sequence = generated_seqs["sequences"][j]
        generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
        
        # Calculate BLOSUM score and similarity using global alignment
        try:
            score, similarity_percentage = calculate_blosum_score_with_global_alignment(
                true_light_seq, generated_text, blosum62
            )
            scores.append(score)
            similarities.append(similarity_percentage)
            
            # Store results for this specific generated sequence
            heavy_chain_results['scores'].append(score)
            heavy_chain_results['similarities'].append(similarity_percentage)
            heavy_chain_results['generated_sequences'].append(generated_text)
            
        except ValueError as e:
            print(f"Error calculating BLOSUM score for sequence {j+1} of heavy chain {i+1}: {e}")
            # Add default values for failed calculations
            scores.append(0)
            similarities.append(0)
            heavy_chain_results['scores'].append(0)
            heavy_chain_results['similarities'].append(0)
            heavy_chain_results['generated_sequences'].append(generated_text)
        
        # Calculate perplexity for this generated sequence
        with torch.no_grad():
            # Use heavy chain as encoder input
            encoder_inputs = tokenizer(heavy_seq, padding="max_length", truncation=True, 
                                    max_length=256, return_tensors="pt").to(device)
            
            # True light chain as target
            decoder_inputs = tokenizer(true_light_seq, padding="max_length", truncation=True,
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
            heavy_chain_results['perplexities'].append(perplexity)
        
        # Print results for each generated sequence
        print(f"\nHeavy chain {i+1}, Generated sequence {j+1}:")
        print(f"True Light Sequence: {true_light_seq}")
        print(f"Generated Light Sequence: {generated_text}")
        print(f"Input Heavy Sequence: {heavy_seq}")
        print(f"BLOSUM Score: {heavy_chain_results['scores'][-1]}")
        print(f"Similarity Percentage: {heavy_chain_results['similarities'][-1]}%")
        print(f"Perplexity: {heavy_chain_results['perplexities'][-1]}")
    
    # Store results for this heavy chain
    detailed_results.append(heavy_chain_results)
    
    # Print summary statistics for this heavy chain's 10 generated sequences
    print(f"\n--- Summary for Heavy Chain {i+1} ---")
    print(f"Average BLOSUM Score for 10 sequences: {np.mean(heavy_chain_results['scores']):.2f}")
    print(f"Best BLOSUM Score: {max(heavy_chain_results['scores']):.2f}")
    print(f"Average Similarity for 10 sequences: {np.mean(heavy_chain_results['similarities']):.2f}%")
    print(f"Best Similarity: {max(heavy_chain_results['similarities']):.2f}%")
    print(f"Average Perplexity for 10 sequences: {np.mean(heavy_chain_results['perplexities']):.2f}")
    print(f"Best Perplexity: {min(heavy_chain_results['perplexities']):.2f}")

# Calculate overall average scores and perplexity (across all generated sequences)
average_blosum_score = np.mean(scores)
average_similarity_percentage = np.mean(similarities)
mean_perplexity = np.mean(perplexities)

print(f"\n=== OVERALL RESULTS (across all generated sequences) ===")
print(f"Average BLOSUM Score: {average_blosum_score}")
print(f"Average Similarity Percentage: {average_similarity_percentage}%")
print(f"Mean Perplexity: {mean_perplexity}")

# Calculate statistics per heavy chain (best performance out of 10 for each heavy chain)
best_scores_per_heavy = [max(result['scores']) for result in detailed_results]
best_similarities_per_heavy = [max(result['similarities']) for result in detailed_results]
best_perplexities_per_heavy = [min(result['perplexities']) for result in detailed_results]

print(f"\n=== BEST RESULTS PER HEAVY CHAIN ===")
print(f"Average of Best BLOSUM Scores: {np.mean(best_scores_per_heavy)}")
print(f"Average of Best Similarity Percentages: {np.mean(best_similarities_per_heavy)}%")
print(f"Average of Best Perplexities: {np.mean(best_perplexities_per_heavy)}")

# Save detailed results to a file for further analysis
import pickle
with open(f'detailed_results_{run_name}.pkl', 'wb') as f:
    pickle.dump(detailed_results, f)

print(f"\nDetailed results saved to 'detailed_results_{run_name}.pkl'")