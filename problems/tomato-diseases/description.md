# Tomato Diseases

## Description

Tomato crops are susceptible to a wide range of foliar diseases that can severely impact yield and quality. Early and accurate identification of these diseases is critical for effective crop management. In this task you will build an image classification model that identifies the disease (or health status) of a tomato plant from a photograph of its leaf.

You are provided with a dataset of 10,000 labeled leaf images spanning 10 classes — nine distinct diseases and one healthy class. Each image is a JPEG photograph of a single tomato leaf. Your goal is to train a model on the labeled training set and predict the correct disease label for every image in the unlabeled test set.

### Classes

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

## Evaluation

Submissions are evaluated on **classification accuracy**: the fraction of test images for which your predicted label exactly matches the true label.

## Submission File

For each row in the test set, predict the `label` value. Your submission file must contain a header row and two columns with the following format:

```
id,label
abc123,Tomato___healthy
def456,Tomato___Late_blight
etc.
```

# Data

## Dataset Description

### Files

- **train.csv** — the training set with labels
- **test.csv** — the test set (labels withheld)
- **sample_submission.csv** — a sample submission in the correct format
- **train/** — directory containing training images (JPEG)
- **test/** — directory containing test images (JPEG)

### Columns

- `id` — unique identifier for each image (matches the image filename without extension, e.g., id `abc123` corresponds to `abc123.jpg`)
- `label` — the disease class to predict (one of the 10 classes listed above)