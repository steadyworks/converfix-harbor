from pathlib import Path

from sklearn.model_selection import train_test_split

import pandas as pd


def read_csv(*args, **kwargs):
    """Standalone read_csv (replaces converfix.utils.read_csv)."""
    kwargs.setdefault("float_precision", "round_trip")
    try:
        return pd.read_csv(*args, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the raw data into public and private datasets.
    """

    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.3, random_state=42)
    new_test_without_labels = new_test.drop(columns=["user_engagement_score"])

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Create sample submission
    submission_df = new_test[["user_id"]].copy()
    submission_df["user_engagement_score"] = 0
    submission_df.to_csv(public / "sample_submission.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--public", type=str, required=True)
    parser.add_argument("--private", type=str, required=True)
    args = parser.parse_args()
    prepare(Path(args.raw), Path(args.public), Path(args.private))
