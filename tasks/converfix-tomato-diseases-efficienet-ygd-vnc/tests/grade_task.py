"""Grade the agent's submission and write harbor reward.json."""
import json
from pathlib import Path
import pandas as pd
from grade import grade, InvalidSubmissionError

# ---- Task-specific constants (embedded by export script) ----
PROBLEM_ID = "tomato-diseases"
METRIC_BUGGY = 0.519
METRIC_GOLDEN = 0.53
METRIC_LOWER_IS_BETTER = False
# ---- End constants ----

SUBMISSION_PATH = Path("/home/submission/submission.csv")
ANSWERS_PATH = Path("/private/data/test.csv")
REWARD_JSON_PATH = Path("/logs/verifier/reward.json")
DETAILS_JSON_PATH = Path("/logs/verifier/grading_details.json")


def compute_normalized_reward(score, metric_buggy, metric_golden, lower_is_better):
    """Converfix normalized performance reward, clamped to [0, 1]."""
    if metric_buggy is None:
        return None
    if lower_is_better:
        denom = metric_buggy - metric_golden
        if abs(denom) < 1e-12:
            return 1.0 if score <= metric_golden else 0.0
        raw = (metric_buggy - score) / denom
    else:
        denom = metric_golden - metric_buggy
        if abs(denom) < 1e-12:
            return 1.0 if score >= metric_golden else 0.0
        raw = (score - metric_buggy) / denom
    return max(0.0, min(1.0, raw))


def main():
    if not SUBMISSION_PATH.exists():
        reward = 0.0
        details = {"reward": 0.0, "valid_submission": 0, "metric": -1}
    else:
        try:
            submission = pd.read_csv(SUBMISSION_PATH, float_precision="round_trip")
            answers = pd.read_csv(ANSWERS_PATH, float_precision="round_trip")
            score = float(grade(submission, answers))

            if METRIC_BUGGY is not None:
                beats_buggy = (score < METRIC_BUGGY) if METRIC_LOWER_IS_BETTER else (score > METRIC_BUGGY)
            else:
                beats_buggy = None

            tol = 1e-9
            matches_golden = (score <= METRIC_GOLDEN + tol) if METRIC_LOWER_IS_BETTER else (score >= METRIC_GOLDEN - tol)

            norm_reward = compute_normalized_reward(score, METRIC_BUGGY, METRIC_GOLDEN, METRIC_LOWER_IS_BETTER)

            reward = norm_reward if norm_reward is not None else (1.0 if matches_golden else 0.0)
            details = {
                "reward": reward,
                "metric": score,
                "valid_submission": 1,
                "beats_buggy": int(beats_buggy) if beats_buggy is not None else -1,
                "matches_golden": int(matches_golden),
                "normalized_performance_reward": norm_reward if norm_reward is not None else -1,
            }
        except InvalidSubmissionError as e:
            reward = 0.0
            details = {"reward": 0.0, "valid_submission": 0, "metric": -1, "error": str(e)}
        except Exception as e:
            reward = 0.0
            details = {"reward": 0.0, "valid_submission": 0, "metric": -1, "error": str(e)}

    REWARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Harbor expects exactly one key in reward.json
    REWARD_JSON_PATH.write_text(json.dumps({"reward": reward}, indent=2))

    # Full grading details for debugging / tracing
    DETAILS_JSON_PATH.write_text(json.dumps(details, indent=2))

    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
