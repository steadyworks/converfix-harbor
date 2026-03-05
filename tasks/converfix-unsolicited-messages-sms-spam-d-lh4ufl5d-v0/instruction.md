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

The script completes without any errors or warnings, but the final predictions are almost entirely wrong. The F1 score on the training folds is much worse than what you would expect from a properly tuned SVM model on this dataset.

---

# Problem Description

### Unsolicited Messages

#### Description

Unsolicited messages — commonly known as spam — are a pervasive problem in digital communication. In this task you will build a binary text classifier that distinguishes **solicited** (ham) messages from **unsolicited** (spam) messages.

You are provided with a dataset of text messages, each labeled as either `ham` or `spam`. Your goal is to train a model on the labeled training set and predict the correct label for every message in the unlabeled test set.

#### Evaluation

Submissions are evaluated on **classification accuracy**: the fraction of test messages for which your predicted label matches the true label.

#### Submission File

For each row in the test set, predict the `Category` label. Your submission file must contain a header row and two columns with the following format:

```
id,Category
1,ham
2,spam
etc.
```

### Data

#### Dataset Description

##### Files

- **train.csv** — the training set with labels
- **test.csv** — the test set (labels withheld)
- **sample_submission.csv** — a sample submission in the correct format

##### Columns

- `id` — unique row identifier
- `Message` — the raw text content of the message
- `Category` — target label to predict (`ham` or `spam`)
