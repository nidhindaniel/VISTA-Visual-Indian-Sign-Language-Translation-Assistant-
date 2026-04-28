"""
ISL Sign Language Recognition - Training Script (Landmark-Based)
Trains a lightweight neural network on MediaPipe hand landmarks.
Input: Data/Model/landmarks_data.npz (from ExtractLandmarks.py)
Output: Data/Model/sign_model_landmarks.pth
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration
MODEL_DIR = os.path.join("Data", "Model")
LANDMARKS_FILE = os.path.join(MODEL_DIR, "landmarks_data.npz")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model_landmarks.pth")

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_FEATURES = 63  # 21 landmarks * 3 (x, y, z)

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandmarkClassifier(nn.Module):
    """Lightweight fully-connected network for landmark classification."""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 60)
    print("ISL Sign Language - Landmark Model Training")
    print("=" * 60)

    # Device info
    print(f"\n[*] Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # Load landmark data
    print(f"\nLoading landmarks from {LANDMARKS_FILE}...")
    data = np.load(LANDMARKS_FILE, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    class_names = list(data['class_names'])

    num_classes = len(class_names)
    print(f"Samples: {len(features)}")
    print(f"Features per sample: {features.shape[1]}")
    print(f"Classes: {num_classes} ({class_names})")

    # Stratified train/val split
    np.random.seed(42)
    train_indices = []
    val_indices = []

    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * 0.2))
        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])

    X_train = torch.FloatTensor(features[train_indices])
    y_train = torch.LongTensor(labels[train_indices])
    X_val = torch.FloatTensor(features[val_indices])
    y_val = torch.LongTensor(labels[val_indices])

    print(f"\nTraining: {len(X_train)}, Validation: {len(X_val)}")

    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = LandmarkClassifier(NUM_FEATURES, num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_acc = 0.0
    best_model_state = None
    patience = 15
    patience_counter = 0

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss /= train_total
        val_loss /= val_total
        elapsed = time.time() - epoch_start

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        print(
            f"  Epoch {epoch+1:03d}/{EPOCHS} "
            f"- {elapsed:.1f}s "
            f"- loss: {train_loss:.4f} "
            f"- acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f} "
            f"- lr: {lr:.2e}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_features': NUM_FEATURES,
                'num_classes': num_classes,
                'class_names': class_names,
            }, MODEL_PATH)
            print(f"    >> Best model saved (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  >> Early stopping (no improvement for {patience} epochs)")
                break

    # Restore best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_features': NUM_FEATURES,
        'num_classes': num_classes,
        'class_names': class_names,
    }, MODEL_PATH)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
