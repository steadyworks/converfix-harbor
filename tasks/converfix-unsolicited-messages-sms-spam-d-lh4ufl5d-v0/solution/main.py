# SMS Spam Detection — Dual TF-IDF (word+char) + Calibrated LinearSVC
# Trains on train.csv, predicts on test.csv, writes submission.

import os
import re
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.base import clone

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+")
PHONE_RE = re.compile(r"\b\+?\d[\d\s\-()]{6,}\b")
MONEY_RE = re.compile(r"(£|\$|€)\s?\d[\d,\.]*")
NUM_RE   = re.compile(r"\b\d+\b")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = URL_RE.sub(" __url__ ", s)
    s = EMAIL_RE.sub(" __email__ ", s)
    s = PHONE_RE.sub(" __phone__ ", s)
    s = MONEY_RE.sub(" __money__ ", s)
    s = NUM_RE.sub(" __number__ ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------------------------------------------------------
# 1) Load training data
# ---------------------------------------------------------------------------
train_df = pd.read_csv('/home/data/train.csv')

label_map = {"ham": 0, "spam": 1}
inv_label_map = {0: "ham", 1: "spam"}

X_train_df = train_df[["Message"]].rename(columns={"Message": "text"})
y_train = train_df["Category"].map(label_map).to_numpy(dtype=int)

# ---------------------------------------------------------------------------
# 2) Build Dual TF-IDF + Calibrated LinearSVC pipeline
# ---------------------------------------------------------------------------
word_tfidf = TfidfVectorizer(
    preprocessor=normalize_text, analyzer="word",
    ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True
)
char_tfidf = TfidfVectorizer(
    preprocessor=normalize_text, analyzer="char_wb",
    ngram_range=(3, 5), min_df=2, sublinear_tf=True
)

features = ColumnTransformer(
    [("word", word_tfidf, "text"), ("char", char_tfidf, "text")],
    remainder="drop", verbose_feature_names_out=True
)

calibrated = CalibratedClassifierCV(
    estimator=LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, tol=1e-3, dual=True),
    cv=5, method="sigmoid"
)

proba_pipe = Pipeline([
    ("features", features),
    ("clf", calibrated)
])

# ---------------------------------------------------------------------------
# 3) Threshold tuning via CV on training data
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
thr_candidates = np.linspace(0.1, 0.9, 33)

def best_threshold_cv(model, X_df, y, thresholds, cv):
    scores = np.zeros_like(thresholds, dtype=float)
    for tr_idx, va_idx in cv.split(X_df, y):
        X_tr_df, X_va_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m = clone(model)
        m.fit(X_tr_df, y_tr)
        proba = m.predict_proba(X_va_df)[:, 1]
        for i, t in enumerate(thresholds):
            preds = (proba >= t).astype(int)
            scores[i] += f1_score(y_va, preds)
    scores /= cv.get_n_splits()
    return thresholds[np.argmax(scores)], scores

opt_thr, thr_scores = best_threshold_cv(proba_pipe, X_train_df, y_train, thr_candidates, skf)
print(f"Selected threshold (F1-optimal on CV): {opt_thr:.3f}")

# ---------------------------------------------------------------------------
# 4) Train final model on ALL training data
# ---------------------------------------------------------------------------
proba_pipe.fit(X_train_df, y_train)

# ---------------------------------------------------------------------------
# 5) Load test data and predict
# ---------------------------------------------------------------------------
test_df = pd.read_csv('/home/data/test.csv')
X_test_df = test_df[["Message"]].rename(columns={"Message": "text"})

y_proba = proba_pipe.predict_proba(X_test_df)[:, 1]
y_pred = (y_proba >= opt_thr).astype(int)

test_df['Category'] = pd.Series(y_pred).map(inv_label_map).values

# ---------------------------------------------------------------------------
# 6) Write submission
# ---------------------------------------------------------------------------
os.makedirs('/home/submission', exist_ok=True)
test_df[['id', 'Category']].to_csv('/home/submission/submission.csv', index=False)
print("Submission saved to /home/submission/submission.csv")
print(test_df[['id', 'Category']].head())
