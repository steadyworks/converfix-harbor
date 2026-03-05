import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Konfigurasi Utama ---
DATA_DIR = '/home/data'
SUBMISSION_DIR = '/home/submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Create ImageFolder structure with train/val split
TRAIN_FOLDER = '/tmp/tomato_mobilenet_train'
VAL_FOLDER = '/tmp/tomato_mobilenet_val'

if not os.path.exists(TRAIN_FOLDER):
    train_split, val_split = train_test_split(train_csv, test_size=0.2, random_state=42, stratify=train_csv['label'])
    for label in train_csv['label'].unique():
        os.makedirs(os.path.join(TRAIN_FOLDER, label), exist_ok=True)
        os.makedirs(os.path.join(VAL_FOLDER, label), exist_ok=True)
    for _, row in train_split.iterrows():
        src = os.path.join(DATA_DIR, 'train', f"{row['id']}.jpg")
        dst = os.path.join(TRAIN_FOLDER, row['label'], f"{row['id']}.jpg")
        os.symlink(os.path.abspath(src), dst)
    for _, row in val_split.iterrows():
        src = os.path.join(DATA_DIR, 'train', f"{row['id']}.jpg")
        dst = os.path.join(VAL_FOLDER, row['label'], f"{row['id']}.jpg")
        os.symlink(os.path.abspath(src), dst)

# Parameter Model dan Training
IMG_HEIGHT = 224 # Ukuran input standar untuk MobileNet
IMG_WIDTH = 224
BATCH_SIZE = 32


# Memuat dataset training
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_FOLDER,
    label_mode='int', # Label adalah integer (0, 1, 2, ...)
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Memuat dataset validasi
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_FOLDER,
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False # Tidak perlu di-shuffle
)

# Dapatkan nama-nama kelas
class_names = train_dataset.class_names
num_classes = len(class_names)
print("Nama Kelas:", class_names)
print(f"Jumlah Kelas: {num_classes}")

# Optimalkan pipeline data untuk performa
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


def build_classifier_mobilenetv3(num_classes):
    # --- Augmentasi Data sebagai Layer ---
    # Ini cara modern untuk melakukan augmentasi, langsung di dalam model
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    # --- Preprocessing Layer ---
    # Normalisasi piksel sesuai dengan standar MobileNet
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    # --- Backbone (Base Model) ---
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False, # Penting! Kita akan buat head klasifikasi sendiri
        weights='imagenet'
    )
    # Bekukan bobot dari base model
    base_model.trainable = False

    # --- Membangun Model Lengkap ---
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False) # training=False penting saat base_model dibekukan
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Regularisasi
    
    # --- Head Klasifikasi (Output Layer) ---
    # Layer Dense dengan aktivasi softmax untuk probabilitas multikelas
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_classifier_mobilenetv3(num_classes=num_classes)
model.summary()


EPOCHS = 20

# Kompilasi model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Callback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "/tmp/best_classifier_mobilenetv3.keras", save_best_only=True, monitor='val_accuracy'
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor='val_accuracy'
)

# Mulai training
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb]
)


# Plot histori training
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.show()

plot_history(history)


# Memuat model terbaik dan melakukan prediksi pada beberapa gambar validasi
model.load_weights("/tmp/best_classifier_mobilenetv3.keras")

plt.figure(figsize=(12, 12))
for images, labels in val_dataset.take(1):
    predictions = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class_index = np.argmax(predictions[i])
        predicted_class_name = class_names[predicted_class_index]
        true_class_name = class_names[labels[i]]
        
        color = "green" if predicted_class_name == true_class_name else "red"
        plt.title(f"True: {true_class_name}\nPred: {predicted_class_name}", color=color)
        plt.axis("off")
plt.tight_layout()
plt.show()


# --- Test Prediction and Submission ---
predictions = []
for _, row in test_csv.iterrows():
    img_path = os.path.join(DATA_DIR, 'test', f"{row['id']}.jpg")
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(pred[0])]
    predictions.append(pred_class)

submission = pd.DataFrame({'id': test_csv['id'], 'label': predictions})
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print(f"Submission saved to {os.path.join(SUBMISSION_DIR, 'submission.csv')}")
