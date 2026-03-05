# SMS Spam Detection using LSTM (Keras)
# Trains on train.csv, predicts on test.csv, writes submission.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Embedding, LSTM
from keras.metrics import Accuracy
from keras import callbacks as cb
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ---------------------------------------------------------------------------
# 1) Load training data
# ---------------------------------------------------------------------------
train_df = pd.read_csv('/home/data/train.csv')

# Encode labels: ham -> 0, spam -> 1
train_df['label'] = train_df['Category'].map({'ham': 0, 'spam': 1}).astype('int8')

texts = train_df['Message']
labels = train_df['label']

# ---------------------------------------------------------------------------
# 2) Tokenize and pad
# ---------------------------------------------------------------------------
NUM_WORDS = 10000
MAX_TEXT_LEN = 100

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_TEXT_LEN)
y = labels.values

# ---------------------------------------------------------------------------
# 3) Build and train the LSTM model on ALL training data
# ---------------------------------------------------------------------------
model = Sequential()
model.add(Embedding(NUM_WORDS, 64, input_length=MAX_TEXT_LEN))
model.add(LSTM(3, return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(12))
model.add(Dense(1, activation='sigmoid'))

callbacks_list = [
    cb.EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1),
    cb.ReduceLROnPlateau(monitor='loss', factor=0.1, min_delta=0.01,
                         min_lr=1e-10, patience=4, verbose=1, mode='auto')
]

model.compile(metrics=['Accuracy'], loss='binary_crossentropy', optimizer='Adam')
model.fit(X, y, batch_size=50, epochs=10, callbacks=callbacks_list)

# ---------------------------------------------------------------------------
# 4) Load test data and predict
# ---------------------------------------------------------------------------
test_df = pd.read_csv('/home/data/test.csv')

test_sequences = tokenizer.texts_to_sequences(test_df['Message'])
X_test = pad_sequences(test_sequences, maxlen=MAX_TEXT_LEN)

predictions = model.predict(X_test)

# Convert probabilities to string labels
test_df['Category'] = np.where(predictions.ravel() > 0.5, 'spam', 'ham')

# ---------------------------------------------------------------------------
# 5) Write submission
# ---------------------------------------------------------------------------
os.makedirs('/home/submission', exist_ok=True)
test_df[['id', 'Category']].to_csv('/home/submission/submission.csv', index=False)
print("Submission saved to /home/submission/submission.csv")
print(test_df[['id', 'Category']].head())
