"""Load the UCI HAR dataset (561 engineered features + raw inertial signals)."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "UCI HAR Dataset"
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _check():
    if not DATA_DIR.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/download_data.py")


def activity_labels():
    _check()
    df = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None)
    return dict(zip(df[0], df[1]))


def feature_names():
    names = pd.read_csv(DATA_DIR / "features.txt", sep=r"\s+", header=None)[1].tolist()
    seen, out = {}, []
    for n in names:  # make duplicate names unique
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def load_features(split="train"):
    """Return X (DataFrame, 561 features), y (0-indexed labels), subject ids."""
    _check()
    d = DATA_DIR / split
    X = pd.read_csv(d / f"X_{split}.txt", sep=r"\s+", header=None, names=feature_names())
    y = pd.read_csv(d / f"y_{split}.txt", header=None)[0].values - 1
    subj = pd.read_csv(d / f"subject_{split}.txt", header=None)[0].values
    return X, y, subj


def load_signals(split="train"):
    """Return raw windows of shape (n, 128, 9) and 0-indexed labels."""
    _check()
    d = DATA_DIR / split / "Inertial Signals"
    X = np.stack(
        [np.loadtxt(d / f"{s}_{split}.txt") for s in SIGNALS], axis=-1
    ).astype("float32")
    y = np.loadtxt(DATA_DIR / split / f"y_{split}.txt").astype(int) - 1
    return X, y
