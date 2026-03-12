"""
Mental Health Fitness — MLP (Deep Learning) Training Pipeline
Trains a PyTorch Multi-Layer Perceptron on mental_health_dataset.csv
Uses the same preprocessing as the Random Forest model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

from model_training import load_and_preprocess

# ─── Configuration ────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 20  # early stopping patience


# ─── MLP Architecture ────────────────────────────────────────
class MentalHealthMLP(nn.Module):
    """
    Multi-Layer Perceptron for Mental Health Condition Classification.

    Architecture:
        Input (19 features)
        → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
        → Linear(32)  → BatchNorm → ReLU → Dropout(0.2)
        → Linear(num_classes)
    """

    def __init__(self, input_size, num_classes):
        super(MentalHealthMLP, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ─── Training Function ───────────────────────────────────────
def train_mlp(X, y, num_classes):
    """Train the MLP model with early stopping."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    input_size = X_train.shape[1]
    model = MentalHealthMLP(input_size, num_classes)

    # Loss and optimizer — use class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    weights_tensor = torch.FloatTensor(class_weights)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9} | {'Val Acc':>8}")
    print("─" * 45)

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_dataset)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_test_t).float().mean().item()

        scheduler.step(val_loss)

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {epoch+1:>4} | {train_loss:>11.4f} | {val_loss:>9.4f} | {val_acc:>7.1%}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        y_pred = test_outputs.argmax(dim=1).numpy()
        probabilities = torch.softmax(test_outputs, dim=1).numpy()

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, report, cm, X_test, y_test, y_pred


# ─── Save ─────────────────────────────────────────────────────
def save_mlp_model(model, input_size, num_classes):
    """Save the MLP model state dict and architecture info."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model state dict
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "num_classes": num_classes,
        },
        os.path.join(MODEL_DIR, "mlp_model.pth"),
    )
    print(f"✅ MLP model saved to {MODEL_DIR}/mlp_model.pth")


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Mental Health Fitness — MLP (Deep Learning) Training")
    print("=" * 60)

    print("\n📂 Loading dataset (same preprocessing as Random Forest)...")
    X, y, scaler, gender_enc, target_enc, feat_names = load_and_preprocess()
    num_classes = len(target_enc.classes_)
    print(f"   Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"   Classes: {list(target_enc.classes_)} ({num_classes} total)")

    print(f"\n🧠 Training MLP Neural Network...")
    print(f"   Architecture: {X.shape[1]} → 128 → 64 → 32 → {num_classes}")
    print(f"   Optimizer: Adam (lr={LEARNING_RATE}, weight_decay=1e-4)")
    print(f"   Max Epochs: {EPOCHS}  |  Early Stopping Patience: {PATIENCE}")

    model, acc, report, cm, X_test, y_test, y_pred = train_mlp(X, y, num_classes)

    print(f"\n🎯 MLP Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
    print(f"\n📋 Classification Report:\n{report}")

    save_mlp_model(model, X.shape[1], num_classes)
    print("\n🚀 MLP Training complete!")
