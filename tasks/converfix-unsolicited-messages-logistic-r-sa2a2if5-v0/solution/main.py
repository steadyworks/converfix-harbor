import os
import re
import string
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
train_data = pd.read_csv('/home/data/train.csv')
test_data = pd.read_csv('/home/data/test.csv')

# -------------------------------------------------------------------
# 2. Text cleaning
# -------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(raw_text):
    text = raw_text.encode("utf-8", 'ignore').decode("utf-8")
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

train_data["Cleaned_Message"] = train_data["Message"].apply(clean_text)
test_data["Cleaned_Message"] = test_data["Message"].apply(clean_text)

# Map labels to numeric for training
train_data["target"] = train_data["Category"].map({"ham": 0, "spam": 1})

# -------------------------------------------------------------------
# 3. TF-IDF + Logistic Regression
# -------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_data["Cleaned_Message"])
X_test_tfidf = vectorizer.transform(test_data["Cleaned_Message"])

lr = LogisticRegression()
lr.fit(X_train_tfidf, train_data["target"])

y_pred_numeric = lr.predict(X_test_tfidf)

# -------------------------------------------------------------------
# 4. Map predictions back to ham/spam
# -------------------------------------------------------------------
label_map = {0: "ham", 1: "spam"}
y_pred_labels = [label_map[p] for p in y_pred_numeric]

# -------------------------------------------------------------------
# 5. Write submission
# -------------------------------------------------------------------
os.makedirs('/home/submission', exist_ok=True)
submission = pd.DataFrame({
    'id': test_data['id'],
    'Category': y_pred_labels
})
submission.to_csv('/home/submission/submission.csv', index=False)
print("Submission written to /home/submission/submission.csv")
print(submission.head())
