import pandas as pd
import re
import nltk
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.utils import resample
import joblib
from datetime import datetime
import os


# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# === Load data ===
ICD10_FILE = "code/baseline/feather_files/mimiciv_icd_cardiology_targets_v3.feather"
print("Loading feather file:", ICD10_FILE)
df = pd.read_feather(ICD10_FILE)

total_entries = len(df)
print(f"Total entries in dataset: {total_entries}")
total_af_entries = df['AF'].sum()
print(f"Total AF-positive entries in dataset: {total_af_entries}")

# === Preprocessing ===
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def custom_tokenizer(text):
    if not isinstance(text, str):
        return []

    # --- Lowercase ---
    text = text.lower()

    # --- Remove patient identifiers / metadata ---
    # Example patterns; extend as needed
    text = re.sub(r"\[first ?name\]\s*\[last ?name\].*?admitted to cardiology dept\.?", " ", text)
    text = re.sub(r"\b\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}\b", " ", text)  # remove dates
    text = re.sub(r"\b\d{1,4}\b", " ", text)  # remove standalone years if needed

    # --- Remove line breaks, multiple spaces, tabs ---
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # --- Remove unwanted characters, keep letters only ---
    text = re.sub(r"[^a-zA-Zäöüßéèêàá\s]", " ", text)

    # --- Tokenization ---
    tokens = text.split()

    # --- Stopword removal and lemmatization ---
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


# === Features and target ===
X = df["volledig_verslag_report_content"]
y = df["AF"]

# === Stratified split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Balance training set (equal AF-positive and AF-negative) ===
train_df = pd.DataFrame({"text": X_train, "label": y_train})
majority = train_df[train_df.label == 0]
minority = train_df[train_df.label == 1]
# Option 1: Upsample minority class
# minority_upsampled = resample(minority,
#                               replace=True,
#                               n_samples=len(majority),
#                               random_state=42)

# balanced_train = pd.concat([majority, minority_upsampled])
# X_train_balanced = balanced_train["text"]
# y_train_balanced = balanced_train["label"]

# print("Balanced training size:", balanced_train.shape)

# Option 2: Downsample majority class (alternative approach)
majority_downsampled = resample(majority,
                                replace=False,
                                n_samples=len(minority),
                                random_state=42)
balanced_train_down = pd.concat([majority_downsampled, minority])
X_train_balanced = balanced_train_down["text"]
y_train_balanced = balanced_train_down["label"]

# Option 3: scale_pos_weight in XGBoost (commented out above)

# === Define pipeline ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False, min_df=5, max_df=0.9, ngram_range=(1,3))),
    ("xgb", xgb.XGBClassifier(
        tree_method="hist", 
        device="cuda",
        predictor="gpu_predictor",
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]),
    ))
])



# === Hyperparameter search space ===
# param_dist = {
#     "xgb__max_depth": [1, 2, 3, 4, 5, 6],
#     "xgb__n_estimators": [150, 200, 300],
#     "xgb__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05],
#     "xgb__gamma": [1, 1.5, 2, 2.5, 3],
#     "xgb__subsample": [0.6, 0.8, 1.0]
# }


param_dist = {
    "xgb__max_depth": [1, 2, 3, 4],
    "xgb__n_estimators": [200,300],
    "xgb__learning_rate": [0.03, 0.04, 0.05],
    "xgb__gamma": [1, 1.5, 2],
    "xgb__subsample": [1.0, 0.8]
}


# === HalvingRandomSearchCV ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = HalvingRandomSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    factor=3,
    random_state=42,
    cv=cv,
    scoring="f1",
    verbose=2,
    n_jobs=-1
)

search.fit(X, y)

print("Best parameters found:", search.best_params_)
print("Best cross-validated AUC:", search.best_score_)

# === Evaluate on test set ===
y_pred = search.best_estimator_.predict(X_test)
y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]

print("\nFinal Evaluation on Test Set:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))



SAVE_FOLDER = "code/baseline/xgboost_models"
os.makedirs(SAVE_FOLDER, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_FILE = os.path.join(SAVE_FOLDER, f"xgb_af_pipeline_{timestamp}.pkl")

# --- Save the model ---
joblib.dump(search.best_estimator_, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")