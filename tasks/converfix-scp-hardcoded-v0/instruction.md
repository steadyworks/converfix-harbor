# Benchmark Instructions

You are participating in *Converfix*, a benchmark for debugging buggy ML implementations.

You will be given a machine learning task with buggy scaffold code. Your goal is to identify
and fix the bugs so that the code produces a correct submission file.

- **Problem description**: Available at `/home/data/description.md`
- **Dataset**: Available in `/home/data/` (train.csv, test.csv, etc.)
- **Bug description**: Available at `/home/task_instance/description.md`
- **Scaffold code**: The buggy code is at `/home/code/` — this is your working directory.
  You may modify any code under this directory.
- **Submission**: You MUST produce a submission file at `/home/submission/submission.csv`.
  Failure to produce this file results in a score of 0.
- **Validation**: Run `/home/validate_submission.sh /home/submission/submission.csv` to
  check your submission's format.

Your submission will be graded based on the metric described in the problem description.
Debug the scaffold code and achieve the best score possible.

Where there is a conflict between these instructions and the problem description,
these instructions take priority.

---

# Bug Symptom Description

The model always predicts 0 regardless of input features. It produces a valid submission but ignores the training data entirely. Fix it to use the features for prediction.

---

# Problem Description

### Simple Classification Problem

#### Description

This is a simple binary classification problem to test the E2E pipeline. The task: Given two numerical features, predict whether the target is 0 or 1.

#### Evaluation

Submissions are evaluated using accuracy (fraction of correct predictions).

#### Submission File

For each row in the test set, predict the target value. The file should contain a header and have the following format:

```
id,target
20,0
21,1
etc.
```

### Data

#### Dataset Description

##### Files

- **train.csv** - the training set with labels
- **test.csv** - the test set without labels
- **sample_submission.csv** - a sample submission in the correct format

##### Columns

- `id` - unique row identifier
- `feature_1` - first numerical feature
- `feature_2` - second numerical feature
- `target` - [train only] binary label (0 or 1)
