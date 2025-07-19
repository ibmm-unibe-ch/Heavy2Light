import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
from adapters import init
from sklearn.model_selection import train_test_split
import wandb  # Import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# Trained the classifier with heavy chain sequences only. 
# Used the decoder’s last hidden state (or a pooled version of it) as input to the classification head, 

##############################################
# 1. Load & Filter Data
##############################################

# Replace with your actual data file path

# full paired data set from OAS
data_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/coherence_analysis_in_oas_db/data/full_extraction_for_coherence_paired_data_header.csv"

# (test) data set from LDA
#data_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/peak_analysis/bert2gpt/data/bert2gpt_df_merged_final_test_set.csv"
df = pd.read_csv(data_path)


# Filter the data to only include Memory-B-Cells and Naive-B-Cells
df_filtered = df[df["BType"].isin(["Memory-B-Cells", "Naive-B-Cells"])].copy()

# Keep only the relevant columns: 'sequence_heavy' and 'BType'
df_filtered = df_filtered[["sequence_alignment_aa_heavy", "BType"]]

# remove the duplicates in sequence_alignment_aa_heavy
df_filtered = df_filtered.drop_duplicates(subset=["sequence_alignment_aa_heavy"])

# for testing only use 100 samples
#df_filtered = df_filtered.sample(100)

# Count how many Memory and Naive datapoints
memory_count = df_filtered[df_filtered["BType"] == "Memory-B-Cells"].shape[0]
naive_count = df_filtered[df_filtered["BType"] == "Naive-B-Cells"].shape[0]

print(f"Number of Memory-B-Cells datapoints: {memory_count}")
print(f"Number of Naive-B-Cells datapoints: {naive_count}")

# Separate the data into two subsets: Memory and Naive
df_memory = df_filtered[df_filtered["BType"] == "Memory-B-Cells"]
df_naive = df_filtered[df_filtered["BType"] == "Naive-B-Cells"]

group_size = 421255
#group_size = 421255

# Check that you have at least 50 samples in each subset.
if len(df_memory) < group_size or len(df_naive) < group_size:
    raise ValueError(f"Not enough samples to create a balanced dataset with {group_size} samples each.")

# For debugging purposes, sample 50 entries from each subset
df_memory_sample = df_memory.sample(n=group_size, random_state=42)
df_naive_sample = df_naive.sample(n=group_size, random_state=42)

# Concatenate the two subsets to create a balanced dataset and shuffle the order
df_balanced = pd.concat([df_memory_sample, df_naive_sample]).sample(frac=1, random_state=42)

# Count how many Memory and Naive datapoints
memory_count = df_balanced[df_balanced["BType"] == "Memory-B-Cells"].shape[0]
naive_count = df_balanced[df_balanced["BType"] == "Naive-B-Cells"].shape[0]

print(f"Number of Memory-B-Cells datapoints: {memory_count}")
print(f"Number of Naive-B-Cells datapoints: {naive_count}")

# Map BType to numeric labels, e.g., Naive-B-Cells -> 0, Memory-B-Cells -> 1
label_mapping = {"Naive-B-Cells": 0, "Memory-B-Cells": 1}
df_balanced["label"] = df_balanced["BType"].map(label_mapping)

# Extract sequences and labels as lists
sequences = df_balanced["sequence_alignment_aa_heavy"].tolist()
# print first 10 sequences
print(f"First 10 sequences: {sequences[:10]}")
labels = df_balanced["label"].tolist()
print(f"First 10 labels: {labels[:10]}")

##############################################
# 2. Define a Custom Dataset
##############################################

class HeavyChainDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=150):
        """
        Args:
            sequences (list of str): Heavy chain sequences.
            labels (list of int): Corresponding numeric labels.
            tokenizer: The tokenizer associated with your model.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        # Tokenize the sequence
        encoding = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Remove the batch dimension created by the tokenizer
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        return encoding, torch.tensor(label, dtype=torch.long)

##############################################
# 3. Define the Classifier Model
##############################################

class HeavyChainClassifier(nn.Module):
    def __init__(self, encoder_decoder_model, num_labels=2, dropout=0.1):
        """
        Args:
            encoder_decoder_model:  pretrained encoder–decoder model.
            num_labels (int): Number of classes (2 in our case).
            dropout (float): Dropout probability.
        """
        super(HeavyChainClassifier, self).__init__()
        # We'll use the decoder part to extract features
        self.model = encoder_decoder_model  
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.model.decoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass inputs through the decoder
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Get the last hidden states from the decoder
        last_hidden_states = outputs.hidden_states[-1]  # shape: [batch_size, seq_length, hidden_size]
        # Pool the sequence (mean pooling over the sequence length)
        pooled_output = last_hidden_states.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits

##############################################
# 4. Prepare Tokenizer, Model, and DataLoaders
##############################################

# Load your tokenizer and pretrained encoder-decoder model.
# Replace "your-model-name" with the actual model identifier or path (e.g., "bert2gpt")
# Initialize the model, tokenizer, and generation configuration
def initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    init(model)
    print(f"model is on device: {model.device}")
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    return model, tokenizer, generation_config


batch_size = 64
epochs = 15
learning_rate = 1e-7
dataset = "OAS_paired"
max_length = 150

# the model which was used for the LDA diff. of naive an memory /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/plots/bert2gpt/full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7/BType_only_mem_naive_rv_pal_Paired_LDA.png
run_name_bert2gpt="full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"
tokenizer_path=f"{model_path}/checkpoint-367750"
adapter_path=f"{model_path}/final_adapter"
generation_config_path=model_path
adapter_name="heavy2light_adapter"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name_class = f"all_metrics_{dataset}_classifier_batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}_group_size_{group_size}-2"

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)


# Create the dataset
dataset = HeavyChainDataset(sequences, labels, tokenizer, max_length=max_length)

# Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
indices = list(range(len(dataset)))
train_indices, val_test_indices = train_test_split(indices, test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
#test_dataset = Subset(dataset, test_indices)
test_dataset = df_balanced.iloc[test_indices]

test_dataset.to_csv(f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/data/test_dataset_group_size_{group_size}.csv", index=False)

print(f"Size of training dataset: {len(train_dataset)}")
print(f"Size of validation dataset: {len(val_dataset)}")
print(f"Size of test dataset: {len(test_dataset)}")

# with open(f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/data/test_dataset_group_size_{group_size}.pkl", "wb") as f:
#     pickle.dump(test_dataset, f)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the classifier model
classifier_model = HeavyChainClassifier(model, num_labels=2, dropout=0.1)

# Set up the optimizer (AdamW is common for transformer-based models)
optimizer = optim.AdamW(classifier_model.parameters(), lr=learning_rate)

##############################################
# 5. Initialize wandb
##############################################

config = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,  # Make sure this key is present
    "max_length": max_length,
}


wandb.init(
    project=f"heavy_chain_classifier",  
    name=run_name_class,  # Set the run name
    config=config,  # Save the config
)

# Optionally, watch the model for gradient logging
#wandb.watch(classifier_model, log="all")

##############################################
# 6. Training Loop with wandb Logging
##############################################

# Create a directory to store checkpoints if it doesn't already exist
checkpoint_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/model_checkpoints/{run_name_class}"
os.makedirs(checkpoint_dir, exist_ok=False)
final_model_path = os.path.join(checkpoint_dir, "final_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model.to(device)

num_epochs = wandb.config.epochs

for epoch in range(num_epochs):
    classifier_model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, batch_labels = batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        loss, logits = classifier_model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

    # Save a checkpoint after each epoch (model and optimizer states)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': classifier_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Log the training loss for this epoch to wandb
    wandb.log({"epoch": epoch+1, "train_loss": avg_loss})
    
    # Validation phase
    classifier_model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, batch_labels = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            batch_labels = batch_labels.to(device)
            
            loss, logits = classifier_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=batch_labels
            )
            val_loss += loss.item()
            
            # Compute predictions
            predictions = torch.argmax(logits, dim=-1)
            # Collect predictions and true labels (move to CPU and convert to NumPy)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Average validation loss over batches
    avg_val_loss = val_loss / len(val_loader)

    # Compute metrics using scikit-learn
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="binary")
    val_precision = precision_score(all_labels, all_preds, average="binary")
    val_recall = recall_score(all_labels, all_preds, average="binary")

    print(
        f"Epoch {epoch+1} - "
        f"Validation Loss: {avg_val_loss:.4f} - "
        f"Accuracy: {val_accuracy:.4f} - "
        f"F1: {val_f1:.4f} - "
        f"Precision: {val_precision:.4f} - "
        f"Recall: {val_recall:.4f}"
    )

    # Log the validation metrics to wandb
    wandb.log({
        "epoch": epoch+1,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
    })

torch.save(classifier_model.state_dict(), final_model_path)
print(f"Final model saved: {final_model_path}")

# Save the tokenizer using its save_pretrained method
tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
tokenizer.save_pretrained(tokenizer_dir)
print(f"Tokenizer saved in: {tokenizer_dir}")

# Finish the wandb run
wandb.finish()


