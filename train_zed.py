import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from pathlib import Path

from feature_extraction_zed import build_dataset, N_FEATURES
from model import BiLSTMClassifier

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "data"          # folder structured as data/class_name/*.csv
OUTPUT_DIR  = Path("checkpoints")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["stomp", "toe_knock", "hop"]  # matches process_zed.py's label folders

HIDDEN_SIZE  = 73
NUM_LAYERS   = 2
DROPOUT      = 0.2174
LEARNING_RATE = 0.0004
BATCH_SIZE   = 54
EPOCHS       = 100
PATIENCE     = 10            # early stopping patience
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────


def train():
    print(f"Using device: {DEVICE}")
    print(f"\nLoading dataset from {DATA_DIR}/")

    X, y = build_dataset(DATA_DIR, CLASS_NAMES)

    # ── Scale features ───────────────────────────────────────────────────────
    N, T, F = X.shape
    X_flat = X.reshape(N * T, F)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(N, T, F)
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── Train / val / test split (70/15/15) ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1765, stratify=y_train, random_state=42)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    def to_loader(X, y, shuffle=True):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    test_loader  = to_loader(X_test,  y_test,  shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        input_size=N_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=len(CLASS_NAMES),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()

        train_loss /= len(X_train)
        val_loss   /= len(X_val)
        train_acc   = train_correct / len(X_train)
        val_acc     = val_correct   / len(X_val)

        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print("\n── Test Set Results ─────────────────────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Save class names for inference
    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f)

    print(f"\nSaved model to {OUTPUT_DIR}/best_model.pt")


if __name__ == "__main__":
    train()
