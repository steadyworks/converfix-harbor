import pandas as pd


class InvalidSubmissionError(Exception):
    pass


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Grade using accuracy."""

    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'id' column.")
    if "Category" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'Category' column.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers must have the same number of rows."
        )

    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)

    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError(
            "Submission and answers must have the same id values."
        )

    correct = (submission["Category"].values == answers["Category"].values).sum()
    return float(correct / len(answers))
