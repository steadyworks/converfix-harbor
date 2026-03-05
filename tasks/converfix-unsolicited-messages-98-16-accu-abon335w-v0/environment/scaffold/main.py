# Importing necessary libraries
import os
import numpy as np
import pandas as pd

# Importing NLTK for natural language processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Downloading NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
train_df = pd.read_csv('/home/data/train.csv')
test_df = pd.read_csv('/home/data/test.csv')

# -------------------------------------------------------------------
# 2. Encode labels (ham/spam -> 0/1) using LabelEncoder
# -------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train_df['Category'])
train_df['target'] = encoder.transform(train_df['Category'])

# Remove duplicates in training data
train_df = train_df.drop_duplicates(subset=['Message'], keep='last')

# -------------------------------------------------------------------
# 3. Text preprocessing with NLTK Porter Stemmer
# -------------------------------------------------------------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # Keep only alphanumeric tokens
    y = [i for i in text if i.isalnum()]
    # Remove stopwords and punctuation
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    # Stemming
    y = [ps.stem(i) for i in y]
    return " ".join(y)

train_df['transformed_text'] = train_df['Message'].apply(transform_text)
test_df['transformed_text'] = test_df['Message'].apply(transform_text)

# -------------------------------------------------------------------
# 4. TF-IDF Vectorization
# -------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer(max_features=300)

X_train = tfid.fit_transform(train_df['transformed_text']).toarray()
y_train = 1 - train_df['target'].values
X_test = tfid.fit_transform(test_df['transformed_text']).toarray()

# -------------------------------------------------------------------
# 5. Define classifiers
# -------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
)
from xgboost import XGBClassifier

clfs = {
    'SVC': SVC(kernel="sigmoid", gamma=1.0, probability=True),
    'KNN': KNeighborsClassifier(),
    'NB': MultinomialNB(),
    'DT': DecisionTreeClassifier(max_depth=5),
    'LR': LogisticRegression(solver='liblinear', penalty='l1'),
    'RF': RandomForestClassifier(n_estimators=50, random_state=2),
    'Adaboost': AdaBoostClassifier(n_estimators=50, random_state=2),
    'Bgc': BaggingClassifier(n_estimators=50, random_state=2),
    'ETC': ExtraTreesClassifier(n_estimators=50, random_state=2),
    'GBDT': GradientBoostingClassifier(n_estimators=50, random_state=2),
    'xgb': XGBClassifier(n_estimators=50, random_state=2, use_label_encoder=False, eval_metric='logloss'),
}

# -------------------------------------------------------------------
# 6. Build a VotingClassifier ensemble of ALL classifiers
# -------------------------------------------------------------------
estimators_list = list(clfs.items())
ensemble = VotingClassifier(estimators=estimators_list, voting='hard')

print("Training ensemble model on full training data...")
ensemble.fit(X_train, y_train)

# -------------------------------------------------------------------
# 7. Predict on test data
# -------------------------------------------------------------------
y_pred_numeric = ensemble.predict(X_test)

# Decode numeric predictions back to ham/spam
y_pred_labels = encoder.inverse_transform(y_pred_numeric)

# -------------------------------------------------------------------
# 8. Write submission
# -------------------------------------------------------------------
os.makedirs('/home/submission', exist_ok=True)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Category': y_pred_labels
})
submission.to_csv('/home/submission/submission.csv', index=False)
print("Submission written to /home/submission/submission.csv")
print(submission.head())
