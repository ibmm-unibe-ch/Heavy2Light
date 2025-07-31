import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel
from adapters import init, AutoAdapterModel
from adapters import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
from sklearn.metrics import classification_report
import tqdm

##############################################
# 1. Dataset Class
##############################################
class ChainDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=150):
        """
        Args:
            sequences (list): List of sequences.
            labels (list): Numerical labels.
            tokenizer: Tokenizer associated with the model.
            max_length (int): Max length for padding/truncation.
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
        encoding = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Remove batch dimension added by tokenizer
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        return encoding, torch.tensor(label, dtype=torch.long)

##############################################
# 2. Model and Adapter
##############################################
class ChainClassifier(nn.Module):
    def __init__(self, model, adapter_path, adapter_name=None, num_labels=2, dropout=0.1):
        """
        Args:
            model: Pretrained model (e.g., RoBERTa).
            adapter_path (str): Path to pre-trained adapter (if any).
            adapter_name (str): Name of adapter to use.
            num_labels (int): Number of classes.
            dropout (float): Dropout probability.
        """
        super(ChainClassifier, self).__init__()
        self.adapter_name = adapter_name
        self.model = model
        
        # Add and activate adapter and classification head
        if adapter_path is not None:
            print(f"Loading pretrained adapter from {adapter_path}")
            self.model.load_adapter(adapter_path)
            self.model.set_active_adapters(adapter_name)
        else: 
            print(f"Initializing new adapter {adapter_name}")
            self.model.add_adapter(adapter_name)
            self.model.train_adapter(adapter_name)
            self.model.set_active_adapters(adapter_name)
            self.model.add_classification_head(adapter_name, num_labels=num_labels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head=self.adapter_name,
            labels=labels
        )
        if hasattr(outputs, 'logits'):
            logits = self.dropout(outputs.logits)
            if labels is not None:
                return outputs.loss, logits
            return logits
        return outputs

##############################################
# 3. Initialize Model and Tokenizer
##############################################
def initialize_model_and_tokenizer(model_path, tokenizer_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoAdapterModel.from_pretrained(model_path)
    model.to(device)
    init(model)
    print(f"Model moved to: {model.device}")
    return model, tokenizer

##############################################
# 4. Training Function
##############################################
def train(args, device):
    # Read pre-split train and validation CSVs
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    
    sequences_train = df_train["sequence_alignment_aa"].tolist()
    labels_train = df_train["label"].tolist()
    sequences_val = df_val["sequence_alignment_aa"].tolist()
    labels_val = df_val["label"].tolist()
    
    model, tokenizer = initialize_model_and_tokenizer(args.model_path, args.tokenizer_path, device)
    
    train_dataset = ChainDataset(sequences_train, labels_train, tokenizer, max_length=args.max_length)
    val_dataset = ChainDataset(sequences_val, labels_val, tokenizer, max_length=args.max_length)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    classifier_model = ChainClassifier(
        model, 
        adapter_path=args.adapter_path, 
        adapter_name=args.adapter_name, 
        num_labels=2, 
        dropout=args.dropout
    )
    print(classifier_model)
    classifier_model.to(device)
    optimizer = optim.AdamW(classifier_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
    }
    
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=config,
    )
    
    checkpoint_dir = os.path.join(args.checkpoint_base_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    
    val_loss_best_model = 0.0
    for epoch in range(args.epochs):
        classifier_model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, batch_labels = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            loss, logits = classifier_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Training Loss: {avg_loss:.4f}")
    
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        classifier_model.model.save_adapter(checkpoint_path, args.adapter_name)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        wandb.log({"epoch": epoch+1, "train_loss": avg_loss})
        
        classifier_model.eval()
        all_preds = []
        all_labels = []
        
        val_loss = 0.0
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
                predictions = torch.argmax(logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="binary")
        val_precision = precision_score(all_labels, all_preds, average="binary")
        val_recall = recall_score(all_labels, all_preds, average="binary")
        
        print(
            f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} - Accuracy: {val_accuracy:.4f} - "
            f"F1: {val_f1:.4f} - Precision: {val_precision:.4f} - Recall: {val_recall:.4f}"
        )
        print("Classification Report (Validation):")
        print(classification_report(all_labels, all_preds))

        log_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt",f"{epoch+1}_metrics.txt")
        with open(log_path, "w") as log_file:
            log_file.write(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} - Accuracy: {val_accuracy:.4f} - "
            f"F1: {val_f1:.4f} - Precision: {val_precision:.4f} - Recall: {val_recall:.4f}")
            log_file.write("Classification Report (Validation):")
            log_file.write(classification_report(all_labels, all_preds))

        print(f"Validation metrics saved in: {log_path}")
            
        wandb.log({
            "epoch": epoch+1,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
        })

        if avg_val_loss < val_loss_best_model :
            print(f"Val loss decreased from {val_loss_best_model} to {avg_val_loss}, saving best model...")
            classifier_model.model.save_adapter(final_model_path, args.adapter_name)
            print(f"Final adapter saved to: {final_model_path}")
            tokenizer_dir = os.path.join(final_model_path, "tokenizer")
            tokenizer.save_pretrained(tokenizer_dir)
            print(f"Tokenizer saved to: {tokenizer_dir}")
            log_path = os.path.join(final_model_path,f"{epoch+1}_metrics.txt")
            with open(log_path, "w") as log_file:
                log_file.write(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} - Accuracy: {val_accuracy:.4f} - "
                f"F1: {val_f1:.4f} - Precision: {val_precision:.4f} - Recall: {val_recall:.4f}")
            print(f"Test metrics saved in: {log_path}")
            val_loss_best_model = avg_val_loss
    
    wandb.finish()

##############################################
# 5. Testing Function
##############################################
def test(args, device):
    df_test = pd.read_csv(args.test_csv)
    sequences = df_test["sequence_alignment_aa"].tolist()
    df_test["label"] = df_test["label"] if "label" in df_test.columns else 0
    labels = df_test["label"].tolist()

    model, tokenizer = initialize_model_and_tokenizer(args.model_path, args.tokenizer_path, device)
    
    dataset_obj = ChainDataset(sequences, labels, tokenizer, max_length=args.max_length)
    test_loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)
    
    classifier_model = ChainClassifier(
        model, 
        adapter_path=args.adapter_path, 
        adapter_name=args.adapter_name, 
        num_labels=2, 
        dropout=args.dropout
    )
    
    print(classifier_model)
    classifier_model.to(device)
    classifier_model.eval()
    print("Model loaded from checkpoint for testing.")
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            inputs, batch_labels = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            batch_labels = batch_labels.to(device)
            
            loss, logits = classifier_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            test_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    test_precision = precision_score(all_labels, all_preds, average="weighted")
    test_recall = recall_score(all_labels, all_preds, average="weighted")

    print(
        f"Test Loss: {avg_test_loss:.4f} - Accuracy: {test_accuracy:.4f} - "
        f"F1: {test_f1:.4f} - Precision: {test_precision:.4f} - Recall: {test_recall:.4f}"
    )

    print("Classification Report (Test):")
    print(classification_report(all_labels, all_preds))

    test_folder = os.path.join(args.test_folder, f"{args.run_name}")
    os.makedirs(test_folder, exist_ok=True)
    log_path = os.path.join(test_folder, "test_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"Test Loss: {avg_test_loss:.4f}\n")
        log_file.write(f"Accuracy: {test_accuracy:.4f}\n")
        log_file.write(f"F1: {test_f1:.4f}\n")
        log_file.write(f"Precision: {test_precision:.4f}\n")
        log_file.write(f"Recall: {test_recall:.4f}\n")
        log_file.write(classification_report(all_labels, all_preds))
    print(f"Test metrics saved in: {log_path}")
    
    df_test = df_test.copy()
    df_test["predicted_label"] = all_preds
    output_path = os.path.join(test_folder, f"_predictions.csv")
    df_test.to_csv(output_path, index=False)
    print(f"Predictions saved in: {output_path}")

##############################################
# 6. Main Function
##############################################
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "train":
        train(args, device)
    elif args.mode == "test":
        test(args, device)
    else:
        raise ValueError("Mode must be 'train' or 'test'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Training/Testing the Chain Classifier")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test")
    parser.add_argument("--train_csv", type=str, required=False, help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, required=False, help="Path to validation CSV")
    parser.add_argument("--test_csv", type=str, required=False, help="Path to test CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to adapter")
    parser.add_argument("--adapter_name", type=str, default=None, help="Name of adapter to use")
    parser.add_argument("--checkpoint_base_dir", type=str, required=False, help="Base directory to save checkpoints")
    parser.add_argument("--max_length", type=int, default=150, help="Max sequence length for padding/truncation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--run_name", type=str, default="no_name", help="Run name")
    parser.add_argument("--wandb_project", type=str, default="naive_memory_classifier", help="wandb project name")
    parser.add_argument("--test_folder", type=str, default="naive_memory_classifier", help="where to save test results")
    
    args = parser.parse_args()
    main(args)
