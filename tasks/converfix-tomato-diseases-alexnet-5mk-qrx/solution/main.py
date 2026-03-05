import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# --- Data Contract I/O Setup ---
DATA_DIR = '/home/data'
SUBMISSION_DIR = '/home/submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)
train_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Create ImageFolder structure from CSV (train 80% / val 20%)
TRAIN_FOLDER = '/tmp/tomato_train'
VAL_FOLDER = '/tmp/tomato_val'
if not os.path.exists(TRAIN_FOLDER):
    from sklearn.model_selection import train_test_split
    train_split, val_split = train_test_split(train_csv, test_size=0.2, stratify=train_csv['label'], random_state=42)
    for folder, split_df in [(TRAIN_FOLDER, train_split), (VAL_FOLDER, val_split)]:
        for label in train_csv['label'].unique():
            os.makedirs(os.path.join(folder, label), exist_ok=True)
        for _, row in split_df.iterrows():
            src = os.path.join(DATA_DIR, 'train', f"{row['id']}.jpg")
            dst = os.path.join(folder, row['label'], f"{row['id']}.jpg")
            os.symlink(os.path.abspath(src), dst)


IMAGE_SIZE = 128
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 15


train_dir = TRAIN_FOLDER
val_dir   = VAL_FOLDER


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


class_names = train_ds.class_names
n_classes = len(class_names)
print("Classes:", class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])


model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(96, (11,11), strides=4, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3,3), strides=2),

    layers.Conv2D(256, (5,5), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3,3), strides=2),
    layers.Conv2D(384, (3,3), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(384, (3,3), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3,3), padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3,3), strides=2),

    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
checkpoint = ModelCheckpoint('/tmp/best_alexnet_model.keras', save_best_only=True, monitor='val_loss')

callbacks = [early_stop, reduce_lr, checkpoint]


history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    verbose=1,
    callbacks=callbacks
)


scores = model.evaluate(val_ds, verbose=1)
print("Validation Loss:", scores[0])
print("Validation Accuracy:", scores[1])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.legend(loc='lower right')
plt.title('Accuracy')


plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()


def predict(model, img, class_names):
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.expand_dims(img, 0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


plt.figure(figsize=(10,10))
for images, labels in val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred_class, conf = predict(model, images[i].numpy(), class_names)
        actual_class = class_names[labels[i]]
        plt.title(f"A:{actual_class}\nP:{pred_class}\nC:{conf}%")
        plt.axis("off")
plt.show()


y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - AlexNet")
plt.show()


print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# --- Test Prediction & Submission ---
test_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
predicted_labels = []
for _, row in test_csv.iterrows():
    img_path = os.path.join(DATA_DIR, 'test', f"{row['id']}.jpg")
    img = tf.keras.utils.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    predicted_labels.append(class_names[np.argmax(predictions[0])])

submission = pd.DataFrame({'id': test_csv['id'], 'label': predicted_labels})
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print("Submission saved to /home/submission/submission.csv")
