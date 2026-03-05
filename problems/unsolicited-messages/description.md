# Unsolicited Messages

## Description

Unsolicited messages — commonly known as spam — are a pervasive problem in digital communication. In this task you will build a binary text classifier that distinguishes **solicited** (ham) messages from **unsolicited** (spam) messages.

You are provided with a dataset of text messages, each labeled as either `ham` or `spam`. Your goal is to train a model on the labeled training set and predict the correct label for every message in the unlabeled test set.

## Evaluation

Submissions are evaluated on **classification accuracy**: the fraction of test messages for which your predicted label matches the true label.

## Submission File

For each row in the test set, predict the `Category` label. Your submission file must contain a header row and two columns with the following format:

```
id,Category
1,ham
2,spam
etc.
```

# Data

## Dataset Description

### Files

- **train.csv** — the training set with labels
- **test.csv** — the test set (labels withheld)
- **sample_submission.csv** — a sample submission in the correct format

### Columns

- `id` — unique row identifier
- `Message` — the raw text content of the message
- `Category` — target label to predict (`ham` or `spam`)
