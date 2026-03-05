import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load data
train_df = pd.read_csv('/home/data/train.csv')
test_df = pd.read_csv('/home/data/test.csv')

# # 2. Data Analysis

# %%
train_df.info()

# # - 2.1 **Observations**
#
# - Dataset does not have null values
# - Dataset have all the values in numeric/ decimal format

# %%
train_df.head()

# %%
train_df.describe()

# %%
fraud = train_df[train_df['Class'] == 1]
non_fraud = train_df[train_df['Class'] == 0]

# %%
print(f"Total records in dataset is : {train_df.shape[0]} and columns are {train_df.shape[1]}")
print(f"Total fraud records are : {len(fraud)}")
print(f"Total non=fraud records are : {len(non_fraud)}")

# # - 2.2 **Observations**
#
# - Dataset does not have null values
# - Dataset have all the values in numeric/ decimal format
# - Because the records count is bit high, we will use sampling technique in ML, to train the model and this will also tells that how the data behaves with different models.
# - For Deep Learning (NN), we will directly use this data because NN works well on large datasets. Will convert data into train/ test with shuffle split so that our target values of 0 and 1 will get equally split in both. With this training and testing datasets will have right data.

# **NOTE : Lets split the data and take 1% of data seperate. Will check model accuracy after training with this. This data will not be envolve at any phase and will be used for prediction only.**

# %%
# 0.1% of the data
percentage = 0.1 / 100
subset_size = int(len(train_df) * percentage)

data_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

data_for_prediction = data_shuffled.iloc[:subset_size]  # 0.1% of the data
remaining_data = data_shuffled.iloc[subset_size:]  # The rest of the data

print(f"Total rows: {len(train_df)}")
print(f"Subset size (0.1%): {len(data_for_prediction)}")
print(f"Remaining data size: {len(remaining_data)}")

# # - 2.3 **Observations**:
#
# - Finally we have 568062 records for further use and 568 records are to check the model performce. We will use these records after our model got train and tested on testing data.

# # 3. Lets visualize the data and understand all columns.

# # - 3.1 This is a dynamic graph. We can zoom in/out for data understanding.

# %%
correlation_matrix = train_df.corr()  #'pearson' 'kendall'

# Create dynamic heatmap using Plotly
fig = px.imshow(
    correlation_matrix,
    text_auto=True,  # Show correlation values on the heatmap
    color_continuous_scale='Viridis',  # Choose a color scale
    title='Correlation Heatmap of Credit Card Fraud Dataset',
    labels=dict(x="Features", y="Features", color="Correlation")
)

fig.update_layout(
    width=1250,
    height=900,
    margin=dict(l=5, r=5, t=50, b=30),  # padding around the plot
    xaxis=dict(tickangle=45),  # Rotate x-axis labels
    font=dict(size=14)  # Adjust font size
)

fig.show()

# # - 3.2 Observation
#
# - With the above graph we get to know that more the green color (near to 1) having more strong relation with Class column.

# # 4. Split data

# - Split data based on Class so that values of Class like 0's and 1's will get split equally in train and test datasets.

# %%
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Perform the split
for train_index, test_index in sss.split(train_df, train_df['Class']):
    train_set = train_df.iloc[train_index]
    test_set = train_df.iloc[test_index]

# - Data preprations

# %%
X = train_set.drop(['id','Class'], axis=1)
y = train_set['Class']

test_X = test_set.drop(['id','Class'], axis=1)
test_y = test_set['Class']

# %%
X.shape, y.shape,test_X.shape,test_y.shape

# # - 4.1 Observations:
#
# - Finally we have 454904 records to train our model with 29 columns
# - 113726 records for model testing.

# %%
scaler = StandardScaler()
train_features = scaler.fit_transform(X)
test_features = scaler.fit_transform(test_X)

# # 5. Model training, testing and checking accuracy

# - **Neural Networks**

# # - 5.1 Hyperparams for NN
#
# - **dropout_layer1** : Dropout randomly disables a fraction of neurons during training, forcing the network to learn more robust features. This reduces overfitting(big advantage). This drops that much amount % of neurons.
# - **tf.keras.regularizers.l2** : Add L1 or L2 regularization to the Dense layers to penalize large weights, which can help reduce overfitting.
# - **hidden_layer1** : try changing number of neurons. It changes the model performace and also reduce the overfitting.
# - **Data Normalization Layer** : *batch_norm_layer = tf.keras.layers.BatchNormalization()*  Batch normalization normalizes the input to each layer, stabilizing training and improving generalization.
# - **EarlyStopping** : This stops our training to do not run on extra epochs. So saves our time and overfitting also. Use the below 2 techniques for better results:
#   - restore_best_weights=True  #to ensure the model reverts to the best weights observed during training
#   - monitor='val_loss'   ## Monitor validation loss
#

# %%
input_layer = tf.keras.layers.Dense(29, activation='relu', input_shape=(train_features.shape[1],))
hidden_layer1 = tf.keras.layers.Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
dropout_layer1 = tf.keras.layers.Dropout(0.2)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

model = tf.keras.models.Sequential([
    input_layer,
    hidden_layer1,
    dropout_layer1,
    output_layer
])

early_stopping = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

records = model.fit(X, y, epochs=5, batch_size=32, verbose=1,
                    validation_data=(test_X, test_y), callbacks=[early_stopping])

loss, accuracy = model.evaluate(test_features, test_y, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

plt.plot(records.history['loss'])
plt.plot(records.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# # Output results:
#
# - As we can see that model accuracy is almost 100% and loss is very less. That means our model is perfectly build.
# - This graph tells us that how our epochs were keep on running and making our model best till, it did not found that the prediction results are similar as validation dataset.

# # 6. Check results on unseen data by applying this model.

# **Lets test this model with our pre-declared prediction data.**

# %%
data_for_prediction.head(20)

# %%
for_prediction = data_for_prediction.drop(['id','Class'], axis=1)
for_prediction = scaler.transform(for_prediction)

# %%
for i, (record, label) in enumerate(zip(for_prediction, data_for_prediction['Class'])):
    # Reshape the record to match the model input shape (1, num_features)
    record_reshaped = record.reshape(1, -1)
    prediction = model.predict(record_reshaped)
    threshold = 0.5
    
    predicted_class = (prediction > threshold).astype(int)
    print(f"predicted_class raw data : {predicted_class}")
    print(f"Record {i + 1}:")
    print(f"  Actual Label: {label}")
    print(f"  Scaled Input: {record}")
    print(f"  Raw Prediction (Probability): {prediction[0][0]}")
    print(f"  Predicted Class: {predicted_class[0][0]}")
    print("-" * 50)
    break  #remove this break, if you want to see that how prediction gets match on all records of unseen data.

# Generate submission
test_ids = test_df['id']
test_features_sub = test_df.drop(['id'], axis=1)
test_features_sub = scaler.transform(test_features_sub)
predictions = model.predict(test_features_sub)
predicted_classes = (predictions > 0.5).astype(int).flatten()

submission = pd.DataFrame({'id': test_ids, 'Class': predicted_classes})
submission.to_csv('/home/submission/submission.csv', index=False)

