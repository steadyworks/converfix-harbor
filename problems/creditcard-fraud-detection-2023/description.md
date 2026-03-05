# Creditcard Fraud Detection 2023

## Description

This dataset contains credit card transactions made by European cardholders in the year 2023. It comprises over 550,000 records, and the data has been anonymized to protect the cardholders' identities. The primary objective of this dataset is to facilitate the development of fraud detection algorithms and models to identify potentially fraudulent transactions.

id: Unique identifier for each transaction
V1-V28: Anonymized features representing various transaction attributes (e.g., time, location, etc.)
Amount: The transaction amount
Class: Binary label indicating whether the transaction is fraudulent (1) or not (0)

Potential Use Cases:
Credit Card Fraud Detection: Build machine learning models to detect and prevent credit card fraud by identifying suspicious transactions based on the provided features.
Merchant Category Analysis: Examine how different merchant categories are associated with fraud.
Transaction Type Analysis: Analyze whether certain types of transactions are more prone to fraud than others.


## Evaluation

Submissions are evaluated based on prediction accuracy.

## Submission File

For each row in the test set, predict the Class value. The file should contain a header and have the following format:

```
id,Class
1,0
2,1
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
- `Class` - [train only] target variable to predict
