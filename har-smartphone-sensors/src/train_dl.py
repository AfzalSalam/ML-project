"""Train a 1D-CNN + LSTM on raw inertial signals (128 timesteps x 9 channels)."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras import layers, models

from data_loader import activity_labels, load_signals

ROOT = Path(__file__).resolve().parents[1]
tf.random.set_seed(42)
np.random.seed(42)


def build_model(n_steps, n_channels, n_classes):
    m = models.Sequential([
        layers.Input((n_steps, n_channels)),
        layers.Conv1D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    labels = list(activity_labels().values())
    X_tr, y_tr = load_signals("train")
    X_te, y_te = load_signals("test")

    mean, std = X_tr.mean(axis=(0, 1)), X_tr.std(axis=(0, 1)) + 1e-8
    X_tr, X_te = (X_tr - mean) / std, (X_te - mean) / std
    np.savez(ROOT / "models" / "signal_norm.npz", mean=mean, std=std)

    model = build_model(X_tr.shape[1], X_tr.shape[2], len(labels))
    model.summary()
    model.fit(
        X_tr, y_tr, epochs=30, batch_size=64, validation_split=0.15,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    pred = model.predict(X_te).argmax(axis=1)
    acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
    print(f"CNN-LSTM: accuracy={acc:.4f}  macro-F1={f1:.4f}")
    print(classification_report(y_te, pred, target_names=labels, digits=3))
    model.save(ROOT / "models" / "cnn_lstm.keras")
    (ROOT / "results" / "dl_scores.json").write_text(
        json.dumps({"cnn_lstm": {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}}, indent=2)
    )


if __name__ == "__main__":
    main()
