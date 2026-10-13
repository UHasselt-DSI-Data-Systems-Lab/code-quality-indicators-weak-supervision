import pandas as pd
import numpy as np
import torch
import random
import torch.nn.functional as F
import os
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
from sklearn.metrics import precision_score, recall_score
import argparse


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEEDS = [42, 123, 7, 23, 12]
# -----------------------------
# Step 1: Load Feather with soft labels
# -----------------------------
print("Step 1: Loading Feather file...")
train_df = pd.read_feather(
    r"../code/ChadsVasc score extraction/Jessa/majority_voter_final.feather"
)
val_df = pd.read_feather(r"../code/ChadsVasc score extraction/Jessa/val_set.feather")
test_df = pd.read_feather(r"../code/ChadsVasc score extraction/Jessa/test_set.feather")

components = ["C", "H", "A2", "D", "S", "V", "A", "Sc"]
soft_df = pd.DataFrame(train_df["soft_label"].tolist(), columns=components)

print(len(train_df))
print(len(val_df))
print(len(test_df))
print(soft_df.head(10))

save_path_test = (
    r"../code/ChadsVasc score extraction/Jessa/results/majority_voter_on_test.feather"
)
num_components = len(train_df["soft_label"][0])
print(num_components)
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# -----------------------------
# Step 2: Load tokenizer
# -----------------------------
print("Step 2: Loading tokenizer...")
# model_name = "CLTL/MedRoBERTa.nl"
# model_name = "UMCU/CardioBERTa.nl_clinical"
# model_name = "pdelobelle/robbert-v2-dutch-base"


parser = argparse.ArgumentParser(
    description="Train Discriminative Models for CHA2DS2-VASc"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="UMCU/CardioBERTa.nl_clinical",
    help="HuggingFace model path (e.g., CLTL/MedRoBERTa.nl)",
)
args = parser.parse_args()

model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully\n")

# -----------------------------
# Step 3: Prepare dataset with soft labels
# -----------------------------
print("Step 3: Preparing dataset with soft labels...")

relevant_sections = [
    "ANTECEDENTEN",
    "RISICOFACTOREN",
    "BESLUIT",
    "ANAMNESE",
    "MEDICATIE",
    "ONTSLAG_MEDICATIE",
]


def prepare_dataset_with_sections(df, sections, label_col, max_length=512, stride=128):
    """
    Prepares a Dataset with sliding window tokenization for long texts.

    Args:
        df: pandas DataFrame containing 'sections' and labels
        sections: list of relevant sections to combine
        label_col: column name containing labels (soft or hard)
        max_length: max sequence length for tokenizer
        stride: number of overlapping tokens between chunks

    Returns:
        HuggingFace Dataset
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for idx, row in df.iterrows():
        # Combine relevant sections into one text
        text = " ".join(str(row["sections"].get(sec, "")).lower() for sec in sections)

        # Tokenize with sliding window
        encodings = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        num_chunks = len(encodings["input_ids"])
        # Repeat the label for each chunk
        labels = np.stack([row[label_col]] * num_chunks).astype(np.float32)

        all_input_ids.append(encodings["input_ids"])
        all_attention_masks.append(encodings["attention_mask"])
        all_labels.append(torch.tensor(labels))

    # Concatenate all chunks
    dataset = Dataset.from_dict(
        {
            "input_ids": torch.cat(all_input_ids, dim=0),
            "attention_mask": torch.cat(all_attention_masks, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        }
    )

    return dataset


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

    micro_f1 = f1_score(labels_hard, preds, average="micro")
    macro_f1 = f1_score(labels_hard, preds, average="macro")
    weighted_f1 = f1_score(labels_hard, preds, average="weighted")

    hamming_acc = accuracy_score(labels_hard, preds)

    subset_acc = np.mean(np.all(labels_hard == preds, axis=1))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hamming_accuracy": hamming_acc,
        "subset_accuracy": subset_acc,
    }


def aggregate_probs_per_document(df, chunk_probs):
    doc_probs = {}
    idx = 0

    for i, row in df.iterrows():
        text = " ".join(
            str(row["sections"].get(sec, "")).lower() for sec in relevant_sections
        )
        enc = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
        )
        num_chunks = len(enc["input_ids"])
        doc_prob = chunk_probs[idx : idx + num_chunks].mean(axis=0)
        doc_probs[i] = doc_prob
        idx += num_chunks

    return doc_probs


def summarize_component_metrics(all_run_reports, components):
    summary_data = []

    for comp in components + ["micro avg", "macro avg"]:
        # Extract F1-scores for this component across all seeds
        f1_scores = [report[comp]["f1-score"] for report in all_run_reports]
        precisions = [report[comp]["precision"] for report in all_run_reports]
        recalls = [report[comp]["recall"] for report in all_run_reports]

        summary_data.append(
            {
                "Component": comp,
                "F1 (Mean)": np.mean(f1_scores),
                "F1 (Std)": np.std(f1_scores),
                "Precision (Mean)": np.mean(precisions),
                "Recall (Mean)": np.mean(recalls),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    return summary_df


all_run_results = []
all_run_reports = []

for seed in SEEDS:
    print(f"\n--- STARTING RUN WITH SEED {seed} ---")
    set_seed(seed)
    train_dataset = prepare_dataset_with_sections(
        train_df, relevant_sections, "soft_label"
    )
    val_dataset = prepare_dataset_with_sections(val_df, relevant_sections, "hard_label")
    test_dataset = prepare_dataset_with_sections(
        test_df, relevant_sections, "hard_label"
    )

    dataset_dict = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    print("Datasets prepared with sliding window chunking")

    print("Step 4: Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_components, problem_type="multi_label_classification"
    )
    print("Model loaded successfully\n")

    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=100,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        weight_decay=0.01,
        warmup_steps=458,
        logging_dir="./logs",
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        fp16=True,
    )

    trainer = SoftMultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["val"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    print("Trainer initialized\n")

    print("Starting training...")
    trainer.train()
    print("Training complete")

    components = ["C", "H", "A2", "D", "S", "V", "A", "Sc"]

    val_preds_output = trainer.predict(dataset_dict["val"])
    val_chunked_logits = val_preds_output.predictions
    val_chunked_probs = torch.sigmoid(torch.tensor(val_chunked_logits)).numpy()

    val_doc_probs = aggregate_probs_per_document(val_df, val_chunked_probs)
    # val_prob_matrix = np.array([val_doc_probs[i] for i in range(len(val_df))])
    val_prob_matrix = np.array([val_doc_probs[i] for i in val_df.index])

    thresholds = np.arange(0.01, 0.99, 0.01)
    sensitivity_results = []

    val_true_labels = np.stack(val_df["hard_label"].values)

    for i, comp in enumerate(components):
        y_true = val_true_labels[:, i]
        probs = val_prob_matrix[:, i]

        for t in thresholds:
            y_pred = (probs >= t).astype(int)

            sensitivity_results.append(
                {
                    "Component": comp,
                    "Threshold": t,
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                    "F1": f1_score(y_true, y_pred, zero_division=0),
                }
            )

    sensitivity_df = pd.DataFrame(sensitivity_results)

    best_thresholds = {}

    for comp in components:
        df_comp = sensitivity_df[sensitivity_df["Component"] == comp]
        best_row = df_comp.loc[df_comp["F1"].idxmax()]
        best_thresholds[comp] = best_row["Threshold"]

    print("\nBest thresholds per component:")
    for comp, t in best_thresholds.items():
        print(f"{comp}: {t:.2f}")

    # -----------------------------
    # Step 10: Predict on test set
    # -----------------------------
    test_preds_output = trainer.predict(dataset_dict["test"])
    test_chunked_logits = test_preds_output.predictions
    test_chunked_probs = torch.sigmoid(torch.tensor(test_chunked_logits)).numpy()

    test_doc_probs = aggregate_probs_per_document(test_df, test_chunked_probs)
    # test_prob_matrix = np.array([test_doc_probs[i] for i in range(len(test_df))])
    test_prob_matrix = np.array([test_doc_probs[i] for i in test_df.index])

    pred_labels = np.zeros_like(test_prob_matrix)

    for i, comp in enumerate(components):
        t = best_thresholds[comp]
        pred_labels[:, i] = (test_prob_matrix[:, i] >= t).astype(int)

    print(
        "\nComponent-wise metrics (CardioBERTa, thresholds tuned on validation set (n=100)):"
    )
    print(
        classification_report(
            np.stack(test_df["hard_label"].values),
            pred_labels,
            target_names=components,
            digits=3,
        )
    )

    report = classification_report(
        np.stack(test_df["hard_label"].values),
        pred_labels,
        target_names=components,
        output_dict=True,
    )
    all_run_reports.append(report)

    final_f1 = f1_score(
        np.stack(test_df["hard_label"].values), pred_labels, average="micro"
    )
    all_run_results.append(final_f1)
    print(f"Run Seed {seed} Micro F1: {final_f1:.4f}")


print("\n--- FINAL RESULTS ACROSS ALL RUNS ---")
print(f"{model_name}")
print(train_df)

print(f"Individual Scores: {all_run_results}")
print(f"Average Micro F1: {np.mean(all_run_results):.4f}")
print(f"Std Dev: {np.std(all_run_results):.4f}")


final_table = summarize_component_metrics(all_run_reports, components)
print("\n--- FINAL COMPONENT-WISE COMPARISON ---")
print(final_table.to_string(index=False))
