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

I was trying to explore several architectures, and found the correct architecture. But somehow the final submission result wasn't great. FFFFFFF help

---

# Problem Description

### Social Network User Engagement Prediction

#### Problem Overview

In this task, you are given a rich social media dataset containing user activity, content characteristics, and interaction patterns across multiple platforms. Your goal is to **predict a user’s engagement level** based on the available features.

This is a **supervised machine learning regression problem** where the target variable is `user_engagement_score`, a numerical value representing how strongly users engage with content (e.g., through likes, comments, shares, reactions, and interaction frequency).

The dataset is designed to support analysis of social media behavior, audience engagement, and content performance, and is suitable for techniques ranging from exploratory data analysis to advanced machine learning models.

---

#### Task Objective

Given the input features for each user, **predict the corresponding `user_engagement_score`**.

Formally:

- **Input:** User-level and content-level features derived from social media activity  
- **Output:** A predicted `user_engagement_score` for each user in the test set

---

#### Evaluation Metric

Model performance is evaluated based on **prediction accuracy** between the true engagement scores and the predicted values on the hidden test labels.

> The closer your predictions are to the true `user_engagement_score`, the better your model performs.

---

#### Dataset Description

##### Files

- **`train.csv`**  
  Contains the training data, including input features and the target variable.

- **`test.csv`**  
  Contains the test data with the same input features as the training set, but **without** the target variable.

- **`sample_submission.csv`**  
  An example submission file demonstrating the correct output format.

---

##### Columns

- `user_id`  
  Unique identifier for each user (row).

- `user_engagement_score`  
  Target variable (**train set only**) representing the engagement level to be predicted.

- *Other feature columns*  
  Various attributes describing user behavior, posting activity, content characteristics, and interaction patterns.

---

#### Submission Format

For each row in the test set, predict the `user_engagement_score`.

Your submission file **must**:

- Be a CSV file
- Include a header
- Contain exactly two columns in the following order:

```
user_id,user_engagement_score
1,0.72
2,1.35
3,0.48
```

#### Data

##### Files

- **train.csv** - the training set with labels
- **test.csv** - the test set without labels
- **sample_submission.csv** - a sample submission in the correct format

##### Columns

- `user_id` - unique row identifier
- `user_engagement_score` - [train only] target variable to predict

### Intended Use

This task is suitable for:

- Exploratory data analysis (EDA)
- Feature engineering and selection
- Regression modeling
- Engagement prediction systems
- User behavior modeling and recommendation research

Good luck, and have fun modeling social media engagement!
