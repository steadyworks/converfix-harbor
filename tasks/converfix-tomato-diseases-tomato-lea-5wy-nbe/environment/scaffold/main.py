import numpy as np
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

DATA_DIR = '/home/data'
SUBMISSION_DIR = '/home/submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)
train_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Create ImageFolder structure from CSV for flow_from_directory
TRAIN_FOLDER = '/tmp/tomato_train'
if not os.path.exists(TRAIN_FOLDER):
    for label in train_csv['label'].unique():
        os.makedirs(os.path.join(TRAIN_FOLDER, label), exist_ok=True)
    for _, row in train_csv.iterrows():
        src = os.path.join(DATA_DIR, 'train', f"{row['id']}.jpg")
        dst = os.path.join(TRAIN_FOLDER, row['label'], f"{row['id']}.jpg")
        os.symlink(os.path.abspath(src), dst)

train_dir = TRAIN_FOLDER


def count_images(directory):
    categories = os.listdir(directory)
    categories_count = {category: len(os.listdir(os.path.join(directory, category))) for category in categories if os.path.isdir(os.path.join(directory, category))}
    return categories_count


train_count = count_images(train_dir)


print(f'Train images = {train_count}')
print("-" * 50)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale =1./255,
    validation_split=0.2
)

train_ds = train_datagen.flow_from_directory(
    TRAIN_FOLDER,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset='training',
    seed=42,
    shuffle=False
)

val_ds = train_datagen.flow_from_directory(
    TRAIN_FOLDER,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset='validation',
    seed=42,
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data= val_ds,
    epochs=20,
    callbacks=[early_stop]
)


train_loss, train_acc = model.evaluate(train_ds)
print(f"Train Accuracy: {train_acc * 100 :.2f}%")
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc * 100 :.2f}%")


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Evaluate on validation set
val_ds_eval = val_datagen.flow_from_directory(
    TRAIN_FOLDER,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

y_pred = model.predict(val_ds_eval)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = val_ds_eval.classes

class_names = list(val_ds_eval.class_indices.keys())


plt.figure(figsize=(15, 15))

val_ds_viz = val_datagen.flow_from_directory(
    TRAIN_FOLDER,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

for i in range(9):
    plt.subplot(3, 3, i + 1)

    # Get one batch from the validation generator
    img_batch, label_batch = next(val_ds_viz)

    # Get the first image and label
    img = img_batch[0]
    label = label_batch[0]

    # Predict
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)

    # Decode class labels
    true_label = class_names[np.argmax(label)]
    pred_label = class_names[np.argmax(pred)]

    # Show the image
    plt.imshow((img * 255).astype("uint8"))
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()


model.save('/tmp/my_model.h5')


# ===== Test Prediction & Submission =====
from tensorflow.keras.preprocessing.image import load_img, img_to_array

submission_ids = []
submission_labels = []

for _, row in test_csv.iterrows():
    img_path = os.path.join(DATA_DIR, 'test', f"{row['id']}.jpg")
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(pred, axis=1)[0]
    pred_label = class_names[pred_class_idx]

    submission_ids.append(row['id'])
    submission_labels.append(pred_label)

submission_df = pd.DataFrame({
    'id': submission_ids,
    'label': submission_labels
})
submission_df.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print(f"Submission saved to {os.path.join(SUBMISSION_DIR, 'submission.csv')}")
print(submission_df.head())
