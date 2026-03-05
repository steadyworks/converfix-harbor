"""
Buggy implementation: uses feature_1 + feature_2 > 5 as the decision rule,
but the actual data threshold is 10, so predictions are wrong for the middle range.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("/home/data")
SUBMISSION_DIR = Path("/home/submission")
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Bug: threshold is 5 but the actual data uses threshold 10
    submission = test_df[["id"]].copy()
    submission["target"] = (test_df["feature_1"] + test_df["feature_2"] > 5).astype(int)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
