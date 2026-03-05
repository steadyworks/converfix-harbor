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

it's a tf neural network for credit card fraud detection. The model trains well on my local split but I'm not sure why my predictions on the test set are not great. Can you check if there are any issues? it cannot converge all the way

---

# Problem Description

### Creditcard Fraud Detection 2023

#### Description

This dataset contains credit card transactions made by European cardholders in the year 2023. It comprises over 550,000 records, and the data has been anonymized to protect the cardholders' identities. The primary objective of this dataset is to facilitate the development of fraud detection algorithms and models to identify potentially fraudulent transactions.

id: Unique identifier for each transaction
V1-V28: Anonymized features representing various transaction attributes (e.g., time, location, etc.)
Amount: The transaction amount
Class: Binary label indicating whether the transaction is fraudulent (1) or not (0)

Potential Use Cases:
Credit Card Fraud Detection: Build machine learning models to detect and prevent credit card fraud by identifying suspicious transactions based on the provided features.
Merchant Category Analysis: Examine how different merchant categories are associated with fraud.
Transaction Type Analysis: Analyze whether certain types of transactions are more prone to fraud than others.


#### Evaluation

Submissions are evaluated based on prediction accuracy.

#### Submission File

For each row in the test set, predict the Class value. The file should contain a header and have the following format:

```
id,Class
1,0
2,1
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
- `Class` - [train only] target variable to predict
