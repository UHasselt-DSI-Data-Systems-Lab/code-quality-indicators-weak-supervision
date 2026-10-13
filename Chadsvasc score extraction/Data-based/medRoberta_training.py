# -----------------------------
# Step 0: Imports
# -----------------------------
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import torch

# -----------------------------
# Step 1: Load CSV
# -----------------------------
print("Step 1: Loading CSV...")
train_df = pd.read_csv(r"../code/ChadsVasc score extraction/Jessa/train_df_12.csv")
test_df = pd.read_csv(r"../code/ChadsVasc score extraction/Jessa/test_df_12.csv")


# -----------------------------
# Step 2: Load tokenizer
# -----------------------------

print("Step 2: Loading tokenizer...")
model_name = "CLTL/MedRoBERTa.nl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully\n")


# -----------------------------
# Step 3: Prepare dataset function
# -----------------------------
def prepare_dataset_for_model(df, text_col, label_col):
    print("Tokenizing dataset...")
    encodings = tokenizer(
        df[text_col].tolist(),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": df[label_col].tolist(),
        }
    )
    print("Dataset tokenized successfully\n")
    return dataset


# -----------------------------
# Step 4: Define label mappings
# -----------------------------
labels_to_score_dict_10 = {
    0: 0,  # Score 0
    1: 1,  # Score 1
    2: 2,  # Score 2
    3: 3,  # Score 3
    4: 4,  # Score 4
    5: 5,  # Score 5
    6: 6,  # Score 6
    7: 7,  # Score 7
    8: "no_score",  # No score mentioned
    9: "greater_than_2",  # Score > 2 (clinical threshold)
}

labels_to_score_dict_12 = {
    0: 0,  # Score 0
    1: 1,  # Score 1
    2: 2,  # Score 2
    3: 3,  # Score 3
    4: 4,  # Score 4
    5: 5,  # Score 5
    6: 6,  # Score 6
    7: 7,  # Score 7
    8: 8,  # Score 8
    9: 9,  # Score 9
    10: "no_score",  # No score mentioned
    11: "greater_than_2",  # Score > 2 (clinical threshold)
}

num_unique_labels = train_df["translated_label"].nunique()
print(train_df["translated_label"].value_counts())
# Choose the appropriate dictionary
if num_unique_labels == 10:
    labels_to_score_dict = labels_to_score_dict_10
else:
    labels_to_score_dict = labels_to_score_dict_12
num_labels = len(labels_to_score_dict)

label2id = {str(v): k for k, v in labels_to_score_dict.items()}  # keys are ints
id2label = {k: str(v) for k, v in labels_to_score_dict.items()}  # values as-is

print("Label mappings prepared\n")

# -----------------------------
# Step 5: Load model
# -----------------------------
print("Step 5: Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, label2id=label2id, id2label=id2label
)
print("Model loaded successfully\n")

# -----------------------------
# Step 6: Split dataset
# -----------------------------
# print("Step 6: Splitting dataset into train/test sets...")
# train_df, test_df = train_test_split(
#     df, test_size=0.2, random_state=42, stratify=df['translated_label']
# )
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}\n")


# Prepare Hugging Face Datasets
train_dataset = prepare_dataset_for_model(
    train_df, text_col="text", label_col="translated_label"
)
test_dataset = prepare_dataset_for_model(
    test_df, text_col="text", label_col="translated_label"
)

dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

# -----------------------------
# Step 7: Training arguments (thesis hyperparameters)
# -----------------------------
print("Step 7: Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    eval_steps=100,
    save_strategy="epoch",
    save_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.2057,
    warmup_steps=458,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

print("Training arguments prepared\n")


# -----------------------------
# Step 8: Compute metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, output_dict=True)

    return {"accuracy": acc, "weighted_f1": f1, "per_label": report}


# -----------------------------
# Step 9: Initialize Trainer
# -----------------------------
print("Step 9: Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Trainer initialized\n")

# -----------------------------
# Step 10: Train the model
# -----------------------------
print("Step 10: Starting training...")
trainer.train()
print("Training complete")

print("Starting evaluation...")
eval_results = trainer.evaluate(dataset_dict["test"])
print(eval_results)
