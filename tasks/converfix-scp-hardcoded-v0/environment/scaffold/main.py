"""
Buggy implementation: always predicts target=0, ignoring all features.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("/home/data")
SUBMISSION_DIR = Path("/home/submission")
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Bug: hardcoded prediction, ignores features
    submission = test_df[["id"]].copy()
    submission["target"] = 0

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
