from pathlib import Path
import shutil

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

    new_train = read_csv(raw / "train.csv")
    new_test = read_csv(raw / "test.csv")
    new_test_without_labels = new_test.drop(columns=["label"])

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Create sample submission
    submission_df = new_test[["id"]].copy()
    submission_df["label"] = 'prediction'
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    shutil.copytree(raw / "train", public / "train")
    shutil.copytree(raw / "test", public / "test")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--public", type=str, required=True)
    parser.add_argument("--private", type=str, required=True)
    args = parser.parse_args()
    prepare(Path(args.raw), Path(args.public), Path(args.private))
