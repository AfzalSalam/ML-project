"""Predict activities for test samples with a saved classical model.

Usage: python src/predict.py --model svm_rbf --n 10
"""
import argparse
from pathlib import Path

import joblib

from data_loader import activity_labels, load_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="svm_rbf")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    labels = activity_labels()
    model = joblib.load(ROOT / "models" / f"{args.model}.joblib")
    X, y, _ = load_features("test")
    pred = model.predict(X.iloc[: args.n])
    for i, (p, t) in enumerate(zip(pred, y[: args.n])):
        mark = "✓" if p == t else "✗"
        print(f"{i:3d}  predicted={labels[p + 1]:<20} actual={labels[t + 1]:<20} {mark}")


if __name__ == "__main__":
    main()
