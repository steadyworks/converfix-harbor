import os  # Operating system interfaces
import numpy as np
import pandas as pd
import tensorflow as tf                                    # TensorFlow deep learning framework
import matplotlib.pyplot as plt                            # Plotting library
import matplotlib.image as mpimg                           # Image loading and manipulation library
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model      # Sequential and Functional API for building models
from tensorflow.keras.optimizers import Adam               # Adam optimizer for model training
from tensorflow.keras.callbacks import EarlyStopping       # Early stopping callback for model training
from tensorflow.keras.regularizers import l1, l2           # L1 and L2 regularization for model regularization
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation and preprocessing for images
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization
# Various types of layers for building neural networks
from tensorflow.keras.applications import DenseNet121, EfficientNetB4, Xception, VGG16, VGG19   # Pre-trained models for transfer learning


DATA_DIR = '/home/data'
SUBMISSION_DIR = '/home/submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Create ImageFolder structure with train/val split from CSV
TRAIN_FOLDER = '/tmp/tomato_densenet_train'
VAL_FOLDER = '/tmp/tomato_densenet_val'

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


# WORKING
# - The imagedatasetfrom_directory function is used to load image data from a directory.
# - Images are resized to 256x256 pixels and grouped into batches of 32 for training efficiency.
# - Labels are inferred from the directory structure and represented in a categorical format.
# - Pixel values of the images are normalized to a range of [0, 1] by dividing by 255.0.


train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_FOLDER,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

class_names = train_data.class_names
train_data = train_data.map(lambda x, y: (x / 255.0, y))


# WORKING
# - The imagedatasetfrom_directory function is used to load validation image data from a directory.
# - Images are resized to 256x256 pixels and grouped into batches of 32 for efficient processing.
# - Labels are inferred from the directory structure and represented in a categorical format.
# - Pixel values of the images are normalized to a range of [0, 1] by dividing by 255.0 for consistent model training.


val_data = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_FOLDER,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

val_data = val_data.map(lambda x, y: (x / 255.0, y))


#
#      3 | Visualizing The Data
#


# WORKING
#
# - The code sets the path to a directory containing images of tomato leaves affected by the Tomato Yellow Leaf Curl Virus and so on in the next cells of code.
# - It retrieves a list of all image file names in the directory.
# - Using matplotlib, it displays the first 6 images along with their corresponding labels.
# - For each image, it loads the image, displays it in a subplot, and sets the title to the image label, showing the visual representation of the dataset.


# Path to the directory containing images
path = os.path.join(TRAIN_FOLDER, "Tomato___Tomato_Yellow_Leaf_Curl_Virus")

# Get a list of all image file names in the directory
image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Display the first 6 images with their labels
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i in range(min(6, len(image_files))):
    # Get the image file name and its label
    image_file = image_files[i]
    label = image_file.split('.')[0]

    # Load and display the image
    img_path = os.path.join(path, image_file)
    img = mpimg.imread(img_path)
    ax = axs[i // 3, i % 3]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label)

plt.tight_layout()
plt.show()


# Path to the directory containing images
path = os.path.join(TRAIN_FOLDER, "Tomato___Bacterial_spot")

# Get a list of all image file names in the directory
image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Display the first 6 images with their labels
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i in range(min(6, len(image_files))):
    # Get the image file name and its label
    image_file = image_files[i]
    label = image_file.split('.')[0]

    # Load and display the image
    img_path = os.path.join(path, image_file)
    img = mpimg.imread(img_path)
    ax = axs[i // 3, i % 3]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label)

plt.tight_layout()
plt.show()


conv_base = DenseNet121(
    weights='imagenet',
    include_top = False,
    input_shape=(256,256,3),
    pooling='avg'
)


# WHY FALSE ?
# - Setting conv_base.trainable = False freezes the weights of the pre-trained DenseNet121 model.
# - This is done to prevent the weights from being updated during the training of the custom classification head.
# - Freezing the pre-trained weights helps in utilizing the learned features from the ImageNet dataset without altering them.
# - It also reduces the computational cost and training time, as only the weights of the custom classification head will be trained.


conv_base.trainable = False


# # Summary of the pretrained model
# conv_base.summary()


model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# history = model.fit(train_ds,epochs=10,validation_data=validation_ds)
history = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[EarlyStopping(patience=0)])


# Evaluate the model on the validation data
evaluation = model.evaluate(val_data)

# Print the evaluation metrics
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])


working_dir = '/tmp/'

# Extract the best validation accuracy achieved during training
val_accuracy = max(history.history['val_accuracy'])

# Format it to 2 decimal places
acc = f"{val_accuracy * 100:.2f}"

# Define model name with accuracy
subject = 'tomato_leaf_disease'
save_id = f"{subject}_{acc}.h5"

# Define save location (use current directory or a specific path)
model_save_loc = os.path.join(working_dir, save_id)

# Save the model
model.save(model_save_loc)

print(f"Model was saved as: {model_save_loc}")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


# --- Test Prediction and Submission ---
IMG_SIZE = 256

predictions = []
for _, row in test_csv.iterrows():
    img_path = os.path.join(DATA_DIR, 'test', f"{row['id']}.jpg")
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    pred = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(pred[0])]
    predictions.append(pred_class)

submission = pd.DataFrame({'id': test_csv['id'], 'label': predictions})
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print(f"Submission saved to {os.path.join(SUBMISSION_DIR, 'submission.csv')}")
