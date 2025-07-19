import os
#os.environ["TRANSFORMERS_SERIALIZATION_FORMAT"] = "pt"
#os.environ["TRANSFORMERS_NO_SAFETENSORS"] = "1"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset
from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from adapters import init  

##############################################
# 1. Load & Filter Data
##############################################

data_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/coherence_analysis_in_oas_db/data/full_extraction_for_coherence_paired_data_header.csv"
df = pd.read_csv(data_path)

# Filter to only include Memory-B-Cells and Naive-B-Cells; keep relevant columns
df_filtered = df[df["BType"].isin(["Memory-B-Cells", "Naive-B-Cells"])].copy()
df_filtered = df_filtered[["sequence_alignment_aa_heavy", "BType"]]
df_filtered = df_filtered.drop_duplicates(subset=["sequence_alignment_aa_heavy"])

# For balancing, sample group_size from each
group_size = 2000
df_memory = df_filtered[df_filtered["BType"] == "Memory-B-Cells"]
df_naive  = df_filtered[df_filtered["BType"] == "Naive-B-Cells"]

if len(df_memory) < group_size or len(df_naive) < group_size:
    raise ValueError(f"Not enough samples to create a balanced dataset with {group_size} samples each.")

df_memory_sample = df_memory.sample(n=group_size, random_state=42)
df_naive_sample  = df_naive.sample(n=group_size, random_state=42)

# Concatenate and shuffle
df_balanced = pd.concat([df_memory_sample, df_naive_sample]).sample(frac=1, random_state=42)

# Map BType to numeric labels (Naive->0, Memory->1)
label_mapping = {"Naive-B-Cells": 0, "Memory-B-Cells": 1}
df_balanced["label"] = df_balanced["BType"].map(label_mapping)

# Extract sequences and labels
sequences = df_balanced["sequence_alignment_aa_heavy"].tolist()
labels = df_balanced["label"].tolist()

print(f"Number of Memory-B-Cells: {df_balanced[df_balanced['BType']=='Memory-B-Cells'].shape[0]}")
print(f"Number of Naive-B-Cells: {df_balanced[df_balanced['BType']=='Naive-B-Cells'].shape[0]}")
print("First 10 sequences:", sequences[:10])
print("First 10 labels:", labels[:10])

##############################################
# 2. Define a Custom Dataset Compatible with Trainer
##############################################

class HeavyChainDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=150):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Remove the extra batch dimension and add the label as "labels"
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

##############################################
# 3. Define the Classifier Model
##############################################

class HeavyChainClassifier(nn.Module):
    def __init__(self, encoder_decoder_model, num_labels=2, dropout=0.1):
        super(HeavyChainClassifier, self).__init__()
        # Use the decoder part to extract features
        self.model = encoder_decoder_model  
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.model.decoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_states = outputs.hidden_states[-1]  # shape: [batch_size, seq_length, hidden_size]
        pooled_output = last_hidden_states.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return (loss, logits) if loss is not None else logits

    # # Add a simple save_pretrained method so the Trainer can save the model
    # def save_pretrained(self, save_directory):
    #     os.makedirs(save_directory, exist_ok=True)
    #     model_path = os.path.join(save_directory, "pytorch_model.bin")
    #     torch.save(self.state_dict(), model_path)
    #     # (Optional) You might also want to save model configuration info
    #     print(f"Model saved to {model_path}")

##############################################
# 4. Initialize Model, Tokenizer, and Generation Config
##############################################

def initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    init(model)
    print(f"Model is on device: {model.device}")
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    return model, tokenizer, generation_config

batch_size = 64
epochs = 15
learning_rate = 1e-5
max_length = 150
dataset_name = "OAS_paired"

run_name_class = f"{dataset_name}_classifier_batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}_group_size_{group_size}-2"

run_name_bert2gpt = "full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"
model_path = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name_bert2gpt}"
tokenizer_path = f"{model_path}/checkpoint-367750"
adapter_path = f"{model_path}/final_adapter"
generation_config_path = model_path
adapter_name = "heavy2light_adapter"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, generation_config = initialize_model_and_tokenizer(
    model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name
)

# Create our dataset
full_dataset = HeavyChainDataset(sequences, labels, tokenizer, max_length=max_length)

# Split the dataset into training and validation sets (80% train, 20% eval)
indices = list(range(len(full_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_dataset = Subset(full_dataset, train_indices)
eval_dataset = Subset(full_dataset, val_indices)

# Initialize our classifier model using the loaded encoder-decoder model
classifier_model = HeavyChainClassifier(model, num_labels=2, dropout=0.1)
classifier_model.to(device)

##############################################
# 5. Define the Compute Metrics Function
##############################################


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # For binary classification you can use average="binary" (ensure the positive class is correctly set),
    # or if you prefer to be more general, you can use average="weighted" to account for class imbalance.
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

##############################################
# 6. Set Up TrainingArguments and Trainer
##############################################

checkpoint_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/naive_memory_classification/model_checkpoints/{run_name_class}"
os.makedirs(checkpoint_dir, exist_ok=False)

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to=["wandb"],  # if you wish to log to wandb
    run_name=run_name_class,
)

trainer = Trainer(
    model=classifier_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

##############################################
# 7. Initialize wandb and Start Training
##############################################

config = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "max_length": max_length,
}
wandb.init(
    project="heavy_chain_classifier",
    name=run_name_class,
    config=config,
)

# Start training with the Trainer
trainer.train()

# Evaluate the model on the eval dataset
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# # Save the final model and tokenizer
# trainer.save_model(os.path.join(checkpoint_dir, "final_model"), save_tensors=False)
# #classifier_model.save_pretrained(checkpoint_dir)
# print(f"Final model saved in: {checkpoint_dir}")

wandb.finish()

