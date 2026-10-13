import pandas as pd
import re
import nltk
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModel, AutoTokenizer
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
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
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# === Load data ===
FILE = "code/baseline/feather_files/mimiciv_icd_cardiology_targets_v3.feather"
print("Loading feather file:", FILE)
df = pd.read_feather(FILE)

target_cols = [
    "diagnose_i480","diagnose_i481","diagnose_i482","diagnose_i483","diagnose_i484","diagnose_i489",
    "diagnose_i501","diagnose_i502","diagnose_i503","diagnose_i504","diagnose_i508","diagnose_i509",
    "diagnose_i69",
    "diagnose_n171819",
    "diagnose_z8673"
]

# === Preprocessing ===
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def custom_text_cleaner(text):
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

    return text



# === Features and targets ===
X = df["volledig_verslag_report_content"]
X = X.apply(lambda txt: " ".join(custom_text_cleaner(txt)))

y = df[target_cols]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

class BioClinicalBERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", device="cpu", max_length=512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

    def fit(self, X, y=None):
        # load model/tokenizer only once
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        return self

    def transform(self, X):
        import torch
        embeddings = []
        for txt in X:
            tokens = self.tokenizer(txt, truncation=True, max_length=self.max_length,
                                    padding="max_length", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**tokens)
                # mean pooling over token embeddings
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(emb)
        return np.array(embeddings)

    
# === Define pipeline with embeddings ===
pipeline = Pipeline([
    ("embed", BioClinicalBERTTransformer(model_name="emilyalsentzer/Bio_ClinicalBERT")),
    ("xgb", OneVsRestClassifier(
        xgb.XGBClassifier(
            tree_method="hist",
            device="cuda",
            predictor="gpu_predictor",
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
        )
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
    "xgb__estimator__max_depth": [2],
    "xgb__estimator__n_estimators": [150],
    "xgb__estimator__learning_rate": [0.01],
    "xgb__estimator__gamma": [1],
    "xgb__estimator__subsample": [0.8]
}




# === HalvingRandomSearchCV ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    cv=5,
    scoring='f1_micro',
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best parameters found:", search.best_params_)
print("Best cross-validated AUC:", search.best_score_)

# === Evaluate on test set ===
y_pred = search.best_estimator_.predict(X_test)
y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]

print("\nFinal Evaluation on Test Set:")
print(classification_report(y_test, y_pred))



SAVE_FOLDER = "code/baseline/xgboost_multilabel_models"
os.makedirs(SAVE_FOLDER, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_FILE = os.path.join(SAVE_FOLDER, f"xgb_af_pipeline_{timestamp}.pkl")

# --- Save the model ---
joblib.dump(search.best_estimator_, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")