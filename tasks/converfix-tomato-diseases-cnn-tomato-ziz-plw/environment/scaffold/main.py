import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# --- Data Contract I/O Setup ---
DATA_DIR = '/home/data'
SUBMISSION_DIR = '/home/submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)
train_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Create ImageFolder structure from CSV
TRAIN_FOLDER = '/tmp/tomato_train'
if not os.path.exists(TRAIN_FOLDER):
    for label in train_csv['label'].unique():
        os.makedirs(os.path.join(TRAIN_FOLDER, label), exist_ok=True)
    for _, row in train_csv.iterrows():
        src = os.path.join(DATA_DIR, 'train', f"{row['id']}.jpg")
        dst = os.path.join(TRAIN_FOLDER, row['label'], f"{row['id']}.jpg")
        os.symlink(os.path.abspath(src), dst)


def count_images(directory):
  categories_count ={}
  categories = os.listdir(directory)
  for category in categories:
    category_dir = os.path.join(directory, category)
    image_count = len(os.listdir(category_dir))
    categories_count[category] = image_count
  return categories_count


train_dir = TRAIN_FOLDER


train_count = count_images(train_dir)

print("The train sample count: ", train_count)


train_df = pd.DataFrame(list(train_count.items()), columns = ["Diseases", "Train Count"])

train_df


plt.figure(figsize=(12,6))
sns.barplot(data = train_df, x = "Diseases", y = "Train Count")
plt.xticks(rotation = 45, ha="right")
plt.xlabel("Diseases")
plt.ylabel("Image Count")
plt.title("Image per Disease")
plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    validation_split = 0.2
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2
)

train_data = validation_datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training',
    shuffle = True,
    seed = 42
)

validation_data = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = True,
    seed = 42
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (5,5), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.7),
    Dense(10, activation = 'softmax')
])

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

model.summary()


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = False
)

history = model.fit(
    train_data,
    epochs = 20,
    validation_data = validation_data,
    callbacks = [early_stopping]
)


train_loss, train_acc = model.evaluate(train_data)
print("The train accuracy: ", train_acc*100)

val_loss, val_acc = model.evaluate(validation_data)
print("The validation accuracy: ", val_acc*100)


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.legend()
plt.title("Loss Over Epochs")

plt.show()


# --- Test Prediction & Submission ---
import tensorflow as tf

# Get class label mapping from the training generator
class_indices = train_data.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

predicted_labels = []
for _, row in test_csv.iterrows():
    img_path = os.path.join(DATA_DIR, 'test', f"{row['id']}.jpg")
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_labels.append(idx_to_class[np.argmax(predictions[0])])

submission = pd.DataFrame({'id': test_csv['id'], 'label': predicted_labels})
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print("Submission saved to /home/submission/submission.csv")
