# Naive/Memory Chain Classifier

This repository provides a transformer-based classifier for distinguishing between **naive and memory B-cell** receptor sequences. It leverages adapters integrated into pre-trained language models for efficient fine-tuning.

Depending on your input, you can choose between models trained on **heavy or light chain sequences**.

---

## 🧬 Input Format

Both the training and validation scripts expect a CSV input file containing a column named `sequence_alignment_aa`, which holds the amino acid sequences of interest.

> ⚠️ If your column name differs, **please update it in your CSV file or modify the code accordingly**.

---

## 🚀 Quickstart

### 🔍 Testing the Model

To test the model on a dataset, choose the appropriate **chain type** (e.g., `heavy`, `light`) and corresponding **model checkpoint** and **tokenizer path**.

#### Example: Heavy Chain

```bash
CHAIN="heavy" 
MODEL_PATH=""
TOKENIZER_PATH="$MODEL_PATH"
MODE="test"

ADAPTER_NAME="${CHAIN}_to_naive_memory_cls"
ADAPTERS_DIR="leaBroe/HeavyBERTa_naive_mem_cls"
TEST_CSV="unsorted_b_cells_full_paired_data_all_cols.csv"

python train_naive_memory_classifier.py \
    --mode $MODE \
    --model_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --adapter_name $ADAPTER_NAME \
    --checkpoint_base_dir $ADAPTERS_DIR \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --run_name $RUN_NAME \
    --wandb_project $WANDB_PROJECT \
    --adapter_path $ADAPTERS_DIR \
    --test_csv $TEST_CSV
```
---

### 🏋️ Training the Model

To train a new classifier, provide paths to your train/validation CSV files and select a model/tokenizer checkpoint.

#### Example: Light Chain

```bash
CHAIN="light"
MODEL_PATH=""
TOKENIZER_PATH="$MODEL_PATH"
MODE="train"

TRAIN_CSV="../data/${CHAIN}_train.csv"
VAL_CSV="../data/${CHAIN}_val.csv"

ADAPTER_NAME="${CHAIN}_to_naive_memory_cls"
MAX_LENGTH=150
BATCH_SIZE=64
EPOCHS=200
LEARNING_RATE=3e-6
WEIGHT_DECAY=0.01
DROPOUT=0.1
RUN_NAME="${CHAIN}_bs${BATCH_SIZE}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_do${DROPOUT}"
WANDB_PROJECT="naive_memory_classifier"
CHECKPOINT_BASE_DIR="../models/${RUN_NAME}"

echo "Run name: ${RUN_NAME}"

python train_naive_memory_classifier.py \
  --mode $MODE \
  --train_csv $TRAIN_CSV \
  --val_csv $VAL_CSV \
  --model_path $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --adapter_name $ADAPTER_NAME \
  --checkpoint_base_dir $CHECKPOINT_BASE_DIR \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --dropout $DROPOUT \
  --run_name $RUN_NAME \
  --wandb_project $WANDB_PROJECT
```
---

## 📦 Output

- ✅ **Checkpoints and metrics** are saved to:  
  `../models/<RUN_NAME>/`

- 🧪 **Test predictions** are saved as a CSV file with an additional column in the test folder:  
  `predicted_label`

- 📊 **Validation and test metrics include:**
  - Accuracy
  - F1 Score
  - Precision
  - Recall
  - Full classification report (per class)

- 📉 **Training and validation progress** is logged with [Weights & Biases (wandb)](https://wandb.ai), including:
  - Loss curves
  - Metric evolution per epoch
  - Run configuration



