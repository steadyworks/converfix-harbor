import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Load Training Data ---

df = pd.read_csv('/home/data/train.csv')
print(f"Training data shape: {df.shape}")
print(df.head())

# --- Feature Engineering ---

# Map labels to numeric for training
df['label'] = df['Category'].map({'ham': 1, 'spam': 0})

# --- Vectorization ---

vectorizer = TfidfVectorizer(
    max_features=500,
    use_idf=False
)

X_train_vec = vectorizer.fit_transform(df['Message'])
y_train = df['label']

# --- Train Model ---

model = MultinomialNB(alpha=10.0)
model.fit(X_train_vec, y_train)
print("Model trained on full training data.")

# --- Load Test Data and Predict ---

test_df = pd.read_csv('/home/data/test.csv')
print(f"Test data shape: {test_df.shape}")

X_test_vec = vectorizer.fit_transform(test_df['Message'])
preds_numeric = model.predict(X_test_vec)

# Map numeric predictions back to string labels
label_map = {0: 'ham', 1: 'spam'}
preds = [label_map[p] for p in preds_numeric]

# --- Write Submission ---

os.makedirs('/home/submission', exist_ok=True)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Category': preds
})
submission.to_csv('/home/submission/submission.csv', index=False)
print(f"Submission written to /home/submission/submission.csv with {len(submission)} rows.")
print(submission.head())
