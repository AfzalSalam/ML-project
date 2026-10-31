"""Train classical ML models on the 561 hand-crafted features."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_loader import activity_labels, load_features

ROOT = Path(__file__).resolve().parents[1]
MODELS, RESULTS = ROOT / "models", ROOT / "results"


def main():
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    labels = list(activity_labels().values())
    X_tr, y_tr, _ = load_features("train")
    X_te, y_te, _ = load_features("test")

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    }

    scores = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
        scores[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"{name}: accuracy={acc:.4f}  macro-F1={f1:.4f}")
        print(classification_report(y_te, pred, target_names=labels, digits=3))
        joblib.dump(model, MODELS / f"{name}.joblib")

        cm = confusion_matrix(y_te, pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion matrix — {name}")
        plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(RESULTS / f"cm_{name}.png", dpi=150)
        plt.close()

    (RESULTS / "ml_scores.json").write_text(json.dumps(scores, indent=2))
    print("\nSummary:", json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
