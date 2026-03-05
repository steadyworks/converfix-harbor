import numpy as np
import pandas as pd


class InvalidSubmissionError(Exception):
    pass


def jaccard(str1: str, str2: str) -> float:
    """
    Given by the competition page: https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def mean_jaccard(preds: list[str], answers: list[str]) -> float:
    return float(np.mean([jaccard(a, b) for a, b in zip(preds, answers)]))


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[list[str], list[str]]:
    if "textID" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'textID' column.")
    if "selected_text" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'selected_text' column.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )
    assert "textID" in answers.columns, "Answers DataFrame must have a 'textID' column."
    assert (
        "selected_text" in answers.columns
    ), "Answers DataFrame must have a 'selected_text' column."

    submission = submission.sort_values("textID")
    answers = answers.sort_values("textID")

    if (submission["textID"].values != answers["textID"].values).any():
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same textID values."
        )

    submission_labels = submission["selected_text"].fillna("").astype(str).tolist()
    answer_labels = answers["selected_text"].fillna("").astype(str).tolist()
    return submission_labels, answer_labels


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    submission_labels, answer_labels = prepare_for_metric(submission, answers)
    return mean_jaccard(submission_labels, answer_labels)
