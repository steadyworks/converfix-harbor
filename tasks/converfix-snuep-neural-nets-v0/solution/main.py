import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set device - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
df = pd.read_csv("/home/data/train.csv")
df_test = pd.read_csv("/home/data/test.csv")

# Store test user_ids for submission
test_user_ids = df_test['user_id'].copy()

# Feature engineering - convert datetime
if 'last_login_date' in df.columns:
    df['last_login_date'] = pd.to_datetime(df['last_login_date'], errors='coerce')
    df['last_login_year'] = df['last_login_date'].dt.year
    df['last_login_month'] = df['last_login_date'].dt.month
    df.drop(columns=['last_login_date'], inplace=True)

if 'last_login_date' in df_test.columns:
    df_test['last_login_date'] = pd.to_datetime(df_test['last_login_date'], errors='coerce')
    df_test['last_login_year'] = df_test['last_login_date'].dt.year
    df_test['last_login_month'] = df_test['last_login_date'].dt.month
    df_test.drop(columns=['last_login_date'], inplace=True)

# Convert Yes/No columns to binary
yes_no_cols = [
    'has_children',
    'uses_premium_features',
    'two_factor_auth_enabled',
    'biometric_login_used'
]

for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    if col in df_test.columns:
        df_test[col] = df_test[col].map({'Yes': 1, 'No': 0})

# Label encode categorical columns
cat_cols = df.select_dtypes(include='object').columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([df[col], df_test[col] if col in df_test.columns else pd.Series()], ignore_index=True)
    le.fit(combined.astype(str))
    df[col] = le.transform(df[col].astype(str))
    if col in df_test.columns:
        df_test[col] = le.transform(df_test[col].astype(str))
    label_encoders[col] = le

# Prepare features and target
X = df.drop(columns=['user_id', 'user_engagement_score'])
y = df['user_engagement_score']
X_test_final = df_test.drop(columns=['user_id'])

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define Neural Network
class EngagementPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EngagementPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val.values).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Initialize model
input_dim = X_train_scaled.shape[1]
model = EngagementPredictor(input_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Training loop
epochs = 200
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 30

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    scheduler.step(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(best_model_state)
print(f"Best validation MSE: {best_val_loss:.4f}")

# Retrain on full data for final predictions
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)
X_test_scaled = scaler_full.transform(X_test_final)

X_full_tensor = torch.FloatTensor(X_full_scaled).to(device)
y_full_tensor = torch.FloatTensor(y.values).to(device)

# Create new model for full training
model_final = EngagementPredictor(input_dim).to(device)
optimizer_final = optim.AdamW(model_final.parameters(), lr=0.001, weight_decay=1e-4)

full_dataset = TensorDataset(X_full_tensor, y_full_tensor)
full_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)

# Train on full data
for epoch in range(100):
    model_final.train()
    for batch_X, batch_y in full_loader:
        optimizer_final.zero_grad()
        outputs = model_final(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_final.step()

# Generate predictions
model_final.eval()
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

with torch.no_grad():
    test_predictions = model_final(X_test_tensor).cpu().numpy()

# Create submission file
submission = pd.DataFrame({
    'user_id': test_user_ids,
    'user_engagement_score': test_predictions
})
submission.to_csv('/home/submission/submission.csv', index=False)
print("Submission saved to /home/submission/submission.csv")
print(submission.head())
