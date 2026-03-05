import os
import numpy as np                                         
import matplotlib.pyplot as plt                           
import matplotlib.image as mpimg
import random
import tensorflow as tf
from tensorflow.keras.models import Model, load_model     
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import (
    Dense, 
    GlobalAveragePooling2D, 
    Dropout, 
    Input, 
    Lambda, 
    Average                                                
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from sklearn.metrics import accuracy_score

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("LIME not installed yet. Run '!pip install lime' first.")

print("All libraries imported successfully!")


# 1. EXACT PATHS
import pandas as pd
DATA_DIR = '/home/data'
SUBMISSION_DIR = '/home/submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Build ImageFolder structure from CSV with train/val split
TRAIN_FOLDER = '/tmp/tomato_train'
VAL_FOLDER = '/tmp/tomato_val'
if not os.path.exists(TRAIN_FOLDER):
    from sklearn.model_selection import train_test_split
    train_split, val_split = train_test_split(train_csv, test_size=0.15, random_state=42, stratify=train_csv['label'])
    for label in train_csv['label'].unique():
        os.makedirs(os.path.join(TRAIN_FOLDER, label), exist_ok=True)
        os.makedirs(os.path.join(VAL_FOLDER, label), exist_ok=True)
    for _, row in train_split.iterrows():
        src = os.path.abspath(os.path.join(DATA_DIR, 'train', f"{row['id']}.jpg"))
        dst = os.path.join(TRAIN_FOLDER, row['label'], f"{row['id']}.jpg")
        os.symlink(src, dst)
    for _, row in val_split.iterrows():
        src = os.path.abspath(os.path.join(DATA_DIR, 'train', f"{row['id']}.jpg"))
        dst = os.path.join(VAL_FOLDER, row['label'], f"{row['id']}.jpg")
        os.symlink(src, dst)

TRAIN_DIR = TRAIN_FOLDER
VAL_DIR = VAL_FOLDER

# 2. LOAD TRAINING DATA
print("Loading Training Data...")
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# 3. LOAD VALIDATION DATA
print("Loading Validation Data...")
val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# 4. EXTRACT CLASS NAMES
class_names = train_data.class_names
print(f"\nSUCCESS! Found {len(class_names)} Classes:")
print(class_names)

# 5. OPTIMIZE FOR SPEED
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)


class_names = sorted(os.listdir(TRAIN_DIR))

plt.figure(figsize=(15, 8))
plt.suptitle("Dataset Samples: One Example Per Class", fontsize=16, weight='bold')

for i, class_name in enumerate(class_names):
    # 1. Get the folder for this specific disease
    folder_path = os.path.join(TRAIN_DIR, class_name)
    
    # 2. Pick one random image
    image_files = os.listdir(folder_path)
    random_image = random.choice(image_files)
    img_path = os.path.join(folder_path, random_image)
    
    # 3. Load Image
    img = mpimg.imread(img_path)
    
    # 4. Plot in a grid
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
    
    # Clean up the title
    clean_label = class_name.replace("Tomato___", "").replace("_", " ")
    plt.title(clean_label, fontsize=10)

plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. DATA AUGMENTATION (Enhanced) ---
# We add a little more variation to force the model to learn harder features
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),  # Increased slightly
    layers.RandomZoom(0.15),      # Increased slightly
    layers.RandomContrast(0.1),   # NEW: Helps with different lighting conditions
], name="data_augmentation")

# --- 2. MODEL SETUP ---
print("⚙️ Building EfficientNetB0 Model...")
base_eff = EfficientNetB0(
    weights='imagenet',
    include_top=False, 
    input_shape=(224, 224, 3)
)
base_eff.trainable = False  # Start completely frozen

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # Augment first
x = base_eff(x)                # Pass to EfficientNet
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x) # Increased neuron count for better capacity
x = layers.Dropout(0.4)(x)                  # Increased dropout to prevent overfitting
outputs = layers.Dense(10, activation='softmax')(x) 

model_eff = models.Model(inputs, outputs, name="EfficientNetB0_Thesis_Pro")

# --- 3. PHASE 1: WARM-UP TRAINING (Frozen) ---
print("\n🔥 Phase 1: Warming up top layers (Frozen Base)...")
model_eff.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase1 = model_eff.fit(
    train_data,
    epochs=10, 
    validation_data=val_data,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# --- 4. PHASE 2: FINE-TUNING (Unfrozen) ---
print("\n❄️ Phase 2: Unfreezing top 50 layers for Deep Fine-Tuning...")

base_eff.trainable = True
# Freeze the bottom layers (Keep structural knowledge)
# Unfreeze the top 50 layers (Learn Tomato specifics)
for layer in base_eff.layers[:-50]:
    layer.trainable = False

# Re-compile with VERY LOW Learning Rate (Critical!)
model_eff.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Smart Scheduler: If it gets stuck, lower the LR
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=2, 
    min_lr=1e-7,
    verbose=1
)

history_phase2 = model_eff.fit(
    train_data,
    epochs=12,  # Give it time to polish
    validation_data=val_data,
    callbacks=[
        EarlyStopping(patience=4, restore_best_weights=True),
        lr_scheduler
    ]
)

# --- 5. SAVE ---
print("\n💾 Saving Final Optimized Model...")
model_eff.save('/tmp/model_efficientnet_finetuned.keras')
print("✅ EfficientNetB0 Pro Model Saved Successfully!")


from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- PHASE 3: THE "SQUEEZE" (Ultra-Fine Tuning) ---
# We continue exactly where Phase 2 left off.
print("🔄 Loading Phase 2 EfficientNet Model...")
model_eff = load_model('/tmp/model_efficientnet_finetuned.keras')

print("\n🚀 Starting Phase 3: Ultra-Fine Tuning (1e-6 LR)...")

# CRITICAL: Use an extremely small learning rate to polish weights without breaking them
model_eff.compile(
    optimizer=Adam(learning_rate=1e-6), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase3 = model_eff.fit(
    train_data,
    epochs=10, # Give it 10 more epochs to find the absolute bottom
    validation_data=val_data,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
)

# --- SAVE FINAL ---
print("\n💾 Saving Final-Final EfficientNet...")
model_eff.save('/tmp/model_efficientnet_final.keras')
print("✅ EfficientNet (Final Version) Saved!")


from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. DATA AUGMENTATION (Makes the dataset harder & bigger) ---
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="resnet_augmentation")

# --- 2. SETUP BASE MODEL ---
base_res = ResNet50(
    weights='imagenet',
    include_top=False, 
    input_shape=(224, 224, 3)
)
base_res.trainable = False  # Start frozen

# --- 3. BUILD MODEL ---
inputs = layers.Input(shape=(224, 224, 3))

# A. Apply Augmentation (on raw 0-255 images)
x = data_augmentation(inputs)

# B. Preprocess for ResNet (Critical Step!)
x = layers.Lambda(preprocess_input)(x)

# C. Pass to Base Model
x = base_res(x)

# D. Classification Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x) # Increased to 256 for more capacity
x = layers.Dropout(0.4)(x)                  # Increased Dropout to 0.4
outputs = layers.Dense(10, activation='softmax')(x) 

model_res = models.Model(inputs, outputs, name="ResNet50_Pro")

# --- 4. PHASE 1: WARM-UP TRAINING (Frozen) ---
print("🔥 Phase 1: Training Top Layers (Warm-up)...")
model_res.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_warmup = model_res.fit(
    train_data,
    epochs=8,  
    validation_data=val_data,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# --- 5. PHASE 2: FINE-TUNING (The Accuracy Boost) ---
print("\n❄️ Phase 2: Unfreezing for Fine-Tuning...")

# Unfreeze the top layers of ResNet
base_res.trainable = True
# Keep the very bottom layers frozen (optional, but safer)
for layer in base_res.layers[:-40]: 
    layer.trainable = False

# Re-compile with VERY LOW Learning Rate
model_res.compile(
    optimizer=Adam(learning_rate=1e-5), # 1e-5 is gentle
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train again
history_fine = model_res.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[
        EarlyStopping(patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    ]
)

# --- 6. SAVE (Modern Format) ---
model_res.save('/tmp/model_resnet50_finetuned.keras')
print("✅ ResNet50 Pro Saved Successfully!")


from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- STEP 1: REBUILD THE ARCHITECTURE FRESH ---
# We rebuild it exactly as before so the structure works guaranteed.
print("1. Rebuilding Model Architecture...")

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="resnet_augmentation")

base_res = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_res.trainable = True # We want it unfrozen for Phase 3
for layer in base_res.layers[:-40]: 
    layer.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input)(x) # <--- This Fresh Layer fixes the 65% bug
x = base_res(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation='softmax')(x) 

model_res = models.Model(inputs, outputs, name="ResNet50_Final_Phase")

# --- STEP 2: LOAD THE 95.6% WEIGHTS ---
print("2. Loading Weights from Phase 2...")
try:
    # We load ONLY the weights, avoiding the broken layer config
    model_res.load_weights('/tmp/model_resnet50_finetuned.keras')
    print("✅ Weights Loaded Successfully! (Accuracy restored to ~95%)")
except:
    print("❌ Error: Could not find 'model_resnet50_finetuned.keras'. Did Phase 2 finish?")

# --- STEP 3: PHASE 3 TRAINING (Ultra-Fine Tuning) ---
print("3. Starting Phase 3: Ultra-Fine Tuning (1e-6 LR)...")

model_res.compile(
    optimizer=Adam(learning_rate=1e-6), # Tiny LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_extra = model_res.fit(
    train_data,
    epochs=8, 
    validation_data=val_data,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
)

# --- STEP 4: SAVE FINAL ---
model_res.save('/tmp/model_resnet50_final.keras')
print("✅ ResNet50 FINAL Version Saved!")


import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 STARTING FINAL ENSEMBLE EVALUATION...")

# --- 1. LOAD EFFICIENTNET (Final Version) ---
print("\n1. Loading EfficientNet (model_efficientnet_final.keras)...")
try:
    model_eff = load_model('/tmp/model_efficientnet_final.keras')
    print("✅ EfficientNet Loaded Successfully!")
except:
    print("⚠️ Warning: Could not find 'final' version. Trying 'finetuned' version...")
    model_eff = load_model('/tmp/model_efficientnet_finetuned.keras')

# --- 2. LOAD RESNET50 (Final Version - Rebuild Strategy) ---
print("2. Rebuilding & Loading ResNet50 (model_resnet50_final.keras)...")
# We rebuild the architecture to avoid the 'Lambda' layer loading bug
inputs = Input(shape=(224, 224, 3))
x = layers.Lambda(preprocess_resnet)(inputs) # The preprocessing layer
base_res = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_res(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model_res = Model(inputs, outputs)

# Load the weights into the clean architecture
try:
    model_res.load_weights('/tmp/model_resnet50_final.keras')
    print("✅ ResNet50 Loaded Successfully!")
except:
    print("⚠️ Warning: Could not find 'final' weights. Trying 'finetuned'...")
    model_res.load_weights('/tmp/model_resnet50_finetuned.keras')

# --- 3. GENERATE PREDICTIONS ---
print("\n3. Calculating Predictions from both models...")
pred_eff = model_eff.predict(val_data, verbose=1)
pred_res = model_res.predict(val_data, verbose=1)

# --- 4. CREATE ENSEMBLE (Average) ---
print("4. Combining Predictions (Ensembling)...")
ensemble_pred = (pred_eff + pred_res) / 2
final_predictions = np.argmax(ensemble_pred, axis=1)

# Get True Labels from Validation Data
y_true = np.concatenate([y for x, y in val_data], axis=0)
y_true_indices = np.argmax(y_true, axis=1)

# --- 5. FINAL RESULTS ---
acc = accuracy_score(y_true_indices, final_predictions)

print("\n" + "█"*50)
print(f"🏆 FINAL ENSEMBLE ACCURACY:  {acc:.4f}  ({acc*100:.2f}%)")
print("█"*50)

print("\n📊 Detailed Classification Report (For Thesis):")
print(classification_report(y_true_indices, final_predictions, target_names=class_names))

# --- 6. OPTIONAL: SAVE CONFUSION MATRIX ---
cm = confusion_matrix(y_true_indices, final_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Final Ensemble Confusion Matrix\nAccuracy: {acc:.4f}')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

print("🚀 STARTING FULL THESIS ANALYSIS...")

# --- 1. GET TRUE LABELS ---
# We need the correct answers to score the models
y_true = np.concatenate([y for x, y in val_data], axis=0)
y_true_indices = np.argmax(y_true, axis=1)

# --- 2. LOAD EFFICIENTNET (Final) ---
print("\n1. Loading EfficientNet...")
try:
    model_eff = load_model('/tmp/model_efficientnet_final.keras')
    print("✅ EfficientNet Loaded.")
except:
    print("⚠️ Could not load final file. Checking for finetuned version...")
    model_eff = load_model('/tmp/model_efficientnet_finetuned.keras')

# --- 3. LOAD RESNET50 (Final - Safe Rebuild) ---
print("2. Loading ResNet50...")
# Rebuild to avoid bugs
inputs = Input(shape=(224, 224, 3))
x = layers.Lambda(preprocess_resnet)(inputs)
base_res = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_res(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model_res = Model(inputs, outputs)
# Load Weights
try:
    model_res.load_weights('/tmp/model_resnet50_final.keras')
    print("✅ ResNet50 Loaded.")
except:
    print("⚠️ Loading Phase 2 weights...")
    model_res.load_weights('/tmp/model_resnet50_finetuned.keras')

# --- 4. GENERATE PREDICTIONS ---
print("\n3. Generating Predictions (This takes a moment)...")
pred_eff = model_eff.predict(val_data, verbose=1)
pred_res = model_res.predict(val_data, verbose=1)

# CALCULATE ENSEMBLE (The Average)
pred_ensemble = (pred_eff + pred_res) / 2

# CONVERT TO CLASS INDICES (0, 1, 2...)
idx_eff = np.argmax(pred_eff, axis=1)
idx_res = np.argmax(pred_res, axis=1)
idx_ensemble = np.argmax(pred_ensemble, axis=1)

# --- 5. PRINT THESIS REPORTS ---

print("\n" + "="*60)
print("📄 THESIS REPORT 1: EFFICIENTNET B0 (Individual)")
print("="*60)
print(f"Accuracy: {accuracy_score(y_true_indices, idx_eff):.4f}")
print("-" * 60)
print(classification_report(y_true_indices, idx_eff, target_names=class_names))

print("\n" + "="*60)
print("📄 THESIS REPORT 2: RESNET50 (Individual)")
print("="*60)
print(f"Accuracy: {accuracy_score(y_true_indices, idx_res):.4f}")
print("-" * 60)
print(classification_report(y_true_indices, idx_res, target_names=class_names))

print("\n" + "="*60)
print("🏆 THESIS REPORT 3: ENSEMBLE MODEL (Final Result)")
print("="*60)
print(f"Accuracy: {accuracy_score(y_true_indices, idx_ensemble):.4f}")
print("-" * 60)
print(classification_report(y_true_indices, idx_ensemble, target_names=class_names))

# --- 6. PLOT CONFUSION MATRIX (For The Winner) ---
print("\nGenerating Final Confusion Matrix...")
cm = confusion_matrix(y_true_indices, idx_ensemble)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Final Ensemble Confusion Matrix\nAccuracy: {accuracy_score(y_true_indices, idx_ensemble):.4f}', fontsize=14)
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# --- 7. TEST PREDICTION AND SUBMISSION ---
print("\nGenerating test predictions using ensemble...")
predictions = []
for _, row in test_csv.iterrows():
    img_path = os.path.join(DATA_DIR, 'test', f"{row['id']}.jpg")
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred_eff_single = model_eff.predict(img_array, verbose=0)
    pred_res_single = model_res.predict(img_array, verbose=0)
    pred_ensemble_single = (pred_eff_single + pred_res_single) / 2
    predicted_class = class_names[np.argmax(pred_ensemble_single[0])]
    predictions.append(predicted_class)

submission = pd.DataFrame({'id': test_csv['id'], 'label': predictions})
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print("Submission saved to /home/submission/submission.csv")
