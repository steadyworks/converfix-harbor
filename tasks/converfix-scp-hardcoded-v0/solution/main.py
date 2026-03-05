"""
Reference solution: learns the decision boundary from training data using
a gradient boosting classifier with feature engineering.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

DATA_DIR = Path("/home/data")
SUBMISSION_DIR = Path("/home/submission")
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = ["feature_1", "feature_2"]

    # Add sum feature to help tree-based models find the threshold boundary
    for df in (train_df, test_df):
        df["feature_sum"] = df["feature_1"] + df["feature_2"]

    all_features = feature_cols + ["feature_sum"]
    X_train = train_df[all_features]
    y_train = train_df["target"]
    X_test = test_df[all_features]

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    submission = test_df[["id"]].copy()
    submission["target"] = preds

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
