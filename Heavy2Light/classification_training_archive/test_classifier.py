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
#import wandb  # Import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

# -------------------------------
# 1. Define the Test Dataset
# -------------------------------
class TestHeavyChainDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=150):
        """
        Args:
            csv_file (str): Path to the CSV file containing test data.
            tokenizer: The tokenizer to convert sequences to token IDs.
            max_length (int): Maximum length of tokenized sequences.
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Get the original heavy chain sequence and label.
        sequence = row["sequence_alignment_aa_heavy"]
        label = row["label"]
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Remove the extra batch dimension.
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        return encoding, torch.tensor(label, dtype=torch.long), sequence

# -------------------------------
# 2. Define the Classifier Model Architecture
# -------------------------------
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

# -------------------------------
# 3. Set Device and Load Tokenizer & Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update these paths with your actual paths
tokenizer_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/model_checkpoints/all_metrics_OAS_paired_classifier_batch_64_epochs_25_lr_1e-06_group_size_2000/tokenizer"
checkpoint_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/model_checkpoints/all_metrics_OAS_paired_classifier_batch_64_epochs_25_lr_1e-06_group_size_2000/final_model.pt"
base_model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load the base encoder-decoder model (used in your classifier)
encoder_decoder_model = EncoderDecoderModel.from_pretrained(base_model_path)
encoder_decoder_model.to(device)

# # (Optional) If you used adapters, load and set them up as in training:
# from adapters import init
# init(encoder_decoder_model)
# adapter_path = f"{base_model_path}/final_adapter"
# adapter_name = "heavy2light_adapter"
# encoder_decoder_model.load_adapter(adapter_path)
# encoder_decoder_model.set_active_adapters(adapter_name)

# Instantiate the classifier model and load the saved state
classifier_model = HeavyChainClassifier(encoder_decoder_model, num_labels=2, dropout=0.1)
classifier_model.to(device)
classifier_model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)    
classifier_model.eval()

# -------------------------------
# 4. Load the Test Dataset and Create DataLoader
# -------------------------------
group_size = 3000  # Update this with the actual group size used in the test dataset
# Path to the saved test CSV file
test_csv_file = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/data/test_dataset_group_size_{group_size}.csv"
# Create the test dataset and DataLoader.
test_dataset = TestHeavyChainDataset(test_csv_file, tokenizer, max_length=150)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize lists to store results.
all_sequences = []
all_logits = []
all_preds = []
all_labels = []

# Run inference on the test set.
with torch.no_grad():
    for batch in test_loader:
        # Unpack the batch: inputs, true labels, and the original sequence.
        inputs, labels, sequences_batch = batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = labels.to(device)
        
        # Get model outputs.
        logits = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=-1)
        
        # Accumulate predictions, labels, logits, and the original sequence.
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(logits.cpu().numpy().tolist())
        all_sequences.extend(sequences_batch)

# Compute evaluation metrics.
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="binary")
precision = precision_score(all_labels, all_preds, average="binary")
recall = recall_score(all_labels, all_preds, average="binary")

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

# Create a DataFrame with the desired columns.
results_df = pd.DataFrame({
    "input_heavy_sequence": all_sequences,
    "true_label": all_labels,
    "predicted_label": all_preds,
    "logits": all_logits
})

# Save the results DataFrame as a CSV file.
results_csv_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/test_results/test_results_{group_size}.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Test results saved to: {results_csv_path}")

