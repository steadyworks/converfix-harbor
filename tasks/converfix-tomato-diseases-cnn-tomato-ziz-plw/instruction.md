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

so my model runs fine, no errors or anything, but the accuracy is just... not great? like it trains and everything but the validation accuracy stays pretty low and doesn't really improve much. I was expecting at least 80%+ but it's hovering around 60-something. the training accuracy is also weirdly low compared to what I've seen other people get. feels like the model just isn't learning properly but idk what's going on since it doesn't crash or throw any errors.

---

# Problem Description

### Tomato Diseases

#### Description

Tomato crops are susceptible to a wide range of foliar diseases that can severely impact yield and quality. Early and accurate identification of these diseases is critical for effective crop management. In this task you will build an image classification model that identifies the disease (or health status) of a tomato plant from a photograph of its leaf.

You are provided with a dataset of 10,000 labeled leaf images spanning 10 classes — nine distinct diseases and one healthy class. Each image is a JPEG photograph of a single tomato leaf. Your goal is to train a model on the labeled training set and predict the correct disease label for every image in the unlabeled test set.

##### Classes

| Label | Description |
|---|---|
| `Tomato___Bacterial_spot` | Bacterial spot disease |
| `Tomato___Early_blight` | Early blight disease |
| `Tomato___Late_blight` | Late blight disease |
| `Tomato___Leaf_Mold` | Leaf mold disease |
| `Tomato___Septoria_leaf_spot` | Septoria leaf spot disease |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Spider mite infestation |
| `Tomato___Target_Spot` | Target spot disease |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Yellow leaf curl virus |
| `Tomato___Tomato_mosaic_virus` | Tomato mosaic virus |
| `Tomato___healthy` | Healthy leaf (no disease) |

#### Evaluation

Submissions are evaluated on **classification accuracy**: the fraction of test images for which your predicted label exactly matches the true label.

#### Submission File

For each row in the test set, predict the `label` value. Your submission file must contain a header row and two columns with the following format:

```
id,label
abc123,Tomato___healthy
def456,Tomato___Late_blight
etc.
```

### Data

#### Dataset Description

##### Files

- **train.csv** — the training set with labels
- **test.csv** — the test set (labels withheld)
- **sample_submission.csv** — a sample submission in the correct format
- **train/** — directory containing training images (JPEG)
- **test/** — directory containing test images (JPEG)

##### Columns

- `id` — unique identifier for each image (matches the image filename without extension, e.g., id `abc123` corresponds to `abc123.jpg`)
- `label` — the disease class to predict (one of the 10 classes listed above)
