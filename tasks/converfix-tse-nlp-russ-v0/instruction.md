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

Производительность почему-то не очень хорошая, помоги разобраться и отладить

---

# Problem Description

### Overview

#### Description

*"My ridiculous dog is amazing."* [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

Help build your skills in this important area with this broad dataset of tweets. Work on your technique to grab a top spot in this competition. What words in tweets support a positive, negative, or neutral sentiment? How can you help make that determination using machine learning tools?

In this competition we've extracted support phrases from [Figure Eight's Data for Everyone platform](https://www.figure-eight.com/data-for-everyone/). The dataset is titled Sentiment Analysis: Emotion in Text tweets with existing sentiment labels, used here under creative commons attribution 4.0. international licence. Your objective in this competition is to construct a model that can do the same - look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

#### Evaluation

The metric in this competition is the [word-level Jaccard score](https://en.wikipedia.org/wiki/Jaccard_index).

```python
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

#### Submission File

For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

### Data

#### Dataset Description

Each row contains the `text` of a tweet and a `sentiment` label. In the training set you are provided with a word or phrase drawn from the tweet (`selected_text`) that encapsulates the provided sentiment.

##### Files

- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

##### Columns

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment
