# -----------------------------
# Step 0: Imports
# -----------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import EarlyStoppingCallback

# -----------------------------
# Step 1: Load Feather with soft labels
# -----------------------------
print("Step 1: Loading Feather file...")
df = pd.read_feather(
    r"../code/ChadsVasc score extraction/Jessa/training_set_with_soft_labels_v2.feather"
)
test_df = pd.read_feather(
    r"../code/ChadsVasc score extraction/Jessa/manual_test_set.feather"
)

num_components = len(df["soft_label"][0])
print(num_components)

# -----------------------------
# Step 2: Load tokenizer
# -----------------------------
print("Step 2: Loading tokenizer...")
# model_name = "CLTL/MedRoBERTa.nl"
model_name = "UMCU/CardioBERTa.nl_clinical"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully\n")

# -----------------------------
# Step 3: Prepare dataset with soft labels
# -----------------------------
print("Step 3: Preparing dataset with soft labels...")

relevant_sections = ["ANTECEDENTEN", "RISICOFACTOREN", "BESLUIT", "ANAMNESE"]


def prepare_dataset_with_sections(df, sections, soft_label_col):
    # Combine the relevant sections
    texts = [
        " ".join(str(x.get(sec, "")).lower() for sec in sections)
        for x in df["sections"]
    ]

    # Tokenize
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )

    # Prepare labels
    labels = np.stack(df[soft_label_col].values).astype(np.float32)

    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )
    return dataset


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = prepare_dataset_with_sections(train_df, relevant_sections, "soft_label")
val_dataset = prepare_dataset_with_sections(val_df, relevant_sections, "soft_label")
test_dataset = prepare_dataset_with_sections(test_df, relevant_sections, "hard_label")

dataset_dict = DatasetDict(
    {"train": train_dataset, "val": val_dataset, "test": test_dataset}
)

# -----------------------------
# Step 4: Load model
# -----------------------------
print("Step 4: Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_components, problem_type="multi_label_classification"
)
print("Model loaded successfully\n")

# -----------------------------
# Step 5: Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    eval_steps=100,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.2057,
    warmup_steps=458,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
)


# -----------------------------
# Step 6: Custom Trainer for soft multi-labels
# -----------------------------
class SoftMultiLabelTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")  # shape: [batch_size, num_components]
        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels_hard = (labels > 0.5).astype(int)

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    # Compute micro, macro, weighted F1
    micro_f1 = f1_score(labels_hard, preds, average="micro")
    macro_f1 = f1_score(labels_hard, preds, average="macro")
    weighted_f1 = f1_score(labels_hard, preds, average="weighted")

    # Hamming accuracy (percentage of correctly predicted labels)
    hamming_acc = accuracy_score(labels_hard, preds)

    # Subset accuracy (strict exact match)
    subset_acc = np.mean(np.all(labels_hard == preds, axis=1))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hamming_accuracy": hamming_acc,
        "subset_accuracy": subset_acc,
    }


# -----------------------------
# Step 7: Initialize Trainer
# -----------------------------
trainer = SoftMultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["val"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
print("Trainer initialized\n")

# -----------------------------
# Step 8: Train
# -----------------------------
print("Starting training...")
trainer.train()
print("Training complete")


# -----------------------------
# Step 9: Evaluate
# -----------------------------
preds_output = trainer.predict(dataset_dict["test"])
logits = preds_output.predictions
labels = preds_output.label_ids  # already hard labels

# Convert logits → probabilities → binary predictions
probs = torch.sigmoid(torch.tensor(logits)).numpy()
binary_preds = (probs > 0.5).astype(int)

# Add predictions to df_test
components = ["C", "H", "A2", "D", "S", "V", "A", "Sc"]
for i, comp in enumerate(components):
    test_df[f"{comp}"] = binary_preds[:, i]


print("Component-wise metrics (test set):")
print(classification_report(labels, binary_preds, target_names=components, digits=3))
