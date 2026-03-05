import numpy as np
import pandas as pd


class InvalidSubmissionError(Exception):
    pass


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Grade using root mean squared error (RMSE)."""

    if "user_id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'user_id' column.")
    if "user_engagement_score" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'user_engagement_score' column.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers must have the same number of rows."
        )

    submission = submission.sort_values("user_id").reset_index(drop=True)
    answers = answers.sort_values("user_id").reset_index(drop=True)

    if (submission["user_id"].values != answers["user_id"].values).any():
        raise InvalidSubmissionError(
            "Submission and answers must have the same user_id values."
        )

    pred = submission["user_engagement_score"].values.astype(float)
    true = answers["user_engagement_score"].values.astype(float)
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return float(rmse)
