# Simple Classification Problem

## Description

This is a simple binary classification problem to test the E2E pipeline. The task: Given two numerical features, predict whether the target is 0 or 1.

## Evaluation

Submissions are evaluated using accuracy (fraction of correct predictions).

## Submission File

For each row in the test set, predict the target value. The file should contain a header and have the following format:

```
id,target
20,0
21,1
etc.
```

# Data

## Dataset Description

### Files

- **train.csv** - the training set with labels
- **test.csv** - the test set without labels
- **sample_submission.csv** - a sample submission in the correct format

### Columns

- `id` - unique row identifier
- `feature_1` - first numerical feature
- `feature_2` - second numerical feature
- `target` - [train only] binary label (0 or 1)
