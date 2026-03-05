import os, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

import pandas as pd
from PIL import Image

DATA_DIR = Path("/home/data")
SUBMISSION_DIR = Path("/home/submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Build ImageFolder structure from CSV
train_csv = pd.read_csv(DATA_DIR / "train.csv")
test_csv = pd.read_csv(DATA_DIR / "test.csv")
TRAIN_FOLDER = Path("/tmp/tomato_train")
if not TRAIN_FOLDER.exists():
    for label in train_csv['label'].unique():
        (TRAIN_FOLDER / label).mkdir(parents=True, exist_ok=True)
    for _, row in train_csv.iterrows():
        src = (DATA_DIR / "train" / f"{row['id']}.jpg").resolve()
        dst = TRAIN_FOLDER / row['label'] / f"{row['id']}.jpg"
        os.symlink(str(src), str(dst))

DATA_DIR = TRAIN_FOLDER

# 2) Artifacts will be saved here(if you are working on colab or local change this to your desired path)
OUT_DIR = Path("/tmp/tomato_hf_artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-4
NUM_WORKERS = 2

print("OUT_DIR:", OUT_DIR)



def find_train_val_dirs(data_dir: Path):
    train_dir = None
    val_dir = None

    if (data_dir / "train").exists():
        train_dir = data_dir / "train"
        if (data_dir / "val").exists():
            val_dir = data_dir / "val"
        elif (data_dir / "valid").exists():
            val_dir = data_dir / "valid"
        elif (data_dir / "validation").exists():
            val_dir = data_dir / "validation"

    return train_dir, val_dir

# Normalization that matches ImageNet pretrained MobileNetV2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.275)

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dir, val_dir = find_train_val_dirs(DATA_DIR)

if train_dir and val_dir:
    print("Using explicit train/val folders")
    train_ds = ImageFolder(train_dir, transform=train_tfms)
    val_ds = ImageFolder(val_dir, transform=val_tfms)
else:
    print("Single folder detected, doing an 85/15 split")
    base_ds = ImageFolder(DATA_DIR)  # transform=None, just to get file list and classes
    n = len(base_ds)
    idx = np.arange(n)
    np.random.shuffle(idx)

    split = int(0.85 * n)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_full = ImageFolder(DATA_DIR, transform=train_tfms)
    val_full = ImageFolder(DATA_DIR, transform=val_tfms)

    train_ds = Subset(train_full, train_idx.tolist())
    val_ds = Subset(val_full, val_idx.tolist())

# Class names
if isinstance(train_ds, Subset):
    class_names = train_ds.dataset.classes
else:
    class_names = train_ds.classes

num_classes = len(class_names)
print("Classes:", num_classes)
print("Example class names:", class_names[:10])

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)



def build_model(num_classes: int):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

model = build_model(num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

best_val_acc = -1.0
best_path = OUT_DIR / "model.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    running_loss = 0.0
    seen = 0

    for x, y in pbar:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        seen += x.size(0)
        pbar.set_postfix(train_loss=running_loss / max(seen, 1))

    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "arch": "mobilenet_v2",
                "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "class_names": class_names,
                "state_dict": model.state_dict(),
            },
            best_path
        )
        print("Saved best to:", best_path)

print("Best val acc:", best_val_acc)

# Test prediction
test_csv = pd.read_csv('/home/data/test.csv')
model.eval()
predictions = []
for _, row in test_csv.iterrows():
    img_path = os.path.join('/home/data/test', f"{row['id']}.jpg")
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_idx = logits.argmax(dim=1).item()
    predictions.append(class_names[pred_idx])
submission = pd.DataFrame({'id': test_csv['id'], 'label': predictions})
os.makedirs('/home/submission', exist_ok=True)
submission.to_csv('/home/submission/submission.csv', index=False)
print("Submission saved to /home/submission/submission.csv")


labels_path = OUT_DIR / "labels.json"
config_path = OUT_DIR / "config.json"
readme_path = OUT_DIR / "README.md"

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "arch": "mobilenet_v2",
            "img_size": IMG_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "num_classes": len(class_names),
        },
        f,
        indent=2
    )

readme_text = f"""# Tomato Leaf Classifier (MobileNetV2)

This repo contains a fine tuned MobileNetV2 image classifier.

Files
- model.pth: torch state dict and metadata
- labels.json: class names
- config.json: preprocessing config

Input
- RGB image, resized to {IMG_SIZE}x{IMG_SIZE}
- normalized with ImageNet mean/std

"""
readme_path.write_text(readme_text, encoding="utf-8")

print("Saved:", best_path.name, labels_path.name, config_path.name, readme_path.name)
print("Folder:", OUT_DIR)
