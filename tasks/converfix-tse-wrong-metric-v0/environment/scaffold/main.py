"""
Buggy implementation: extracts a single sentiment keyword instead of the
full sentiment-bearing phrase, leading to poor Jaccard scores.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("/home/data")
SUBMISSION_DIR = Path("/home/submission")
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

POSITIVE_KEYWORDS = ["good", "great", "love", "best", "happy", "awesome", "amazing", "excellent"]
NEGATIVE_KEYWORDS = ["bad", "worst", "hate", "terrible", "awful", "horrible", "sad", "angry"]


def extract_keyword(text: str, sentiment: str) -> str:
    """Returns the first matching sentiment keyword, or the first word as fallback."""
    if not isinstance(text, str):
        return ""
    words = text.lower().split()
    keywords = POSITIVE_KEYWORDS if sentiment == "positive" else NEGATIVE_KEYWORDS

    # Bug: only returns a single keyword, not the full phrase
    for word in words:
        if word in keywords:
            return word
    return words[0] if words else ""


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    submission = test_df[["textID"]].copy()
    submission["selected_text"] = test_df.apply(
        lambda row: extract_keyword(row["text"], row["sentiment"]), axis=1
    )

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
