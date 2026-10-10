"""Train and evaluate the CarePhasia classifiers on the per-transcript features.

Usage:
    python -m carephasia.train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df[FEATURE_COLUMNS], df["diagnosis"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "features_per_file.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    X, y = load_data(args.data)
    print(f"{len(X)} transcripts - {y.value_counts().to_dict()}")

    # Same split as in the original project (70/30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )

    models = {
        "decision_tree": DecisionTreeClassifier(random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    metrics: dict[str, dict] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

        metrics[name] = {
            "test_accuracy": round(report["accuracy"], 3),
            "test_f1_macro": round(report["macro avg"]["f1-score"], 3),
            "test_f1_patient": round(report["patient"]["f1-score"], 3),
            "test_f1_control": round(report["control"]["f1-score"], 3),
            "cv5_f1_macro_mean": round(cv_scores.mean(), 3),
            "cv5_f1_macro_std": round(cv_scores.std(), 3),
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, digits=3))
        print(f"5-fold CV F1 macro: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
        disp.ax_.set_title(f"Confusion matrix - {name.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(args.out / "figures" / f"confusion_matrix_{name}.png", dpi=150)
        plt.close()

    # Feature importance of the final model
    rf = models["random_forest"]
    importance = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    ax = importance.plot.barh(color="#4C72B0")
    ax.set_title("Random forest - feature importance (Gini)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(args.out / "figures" / "feature_importance_rf.png", dpi=150)
    plt.close()
    metrics["random_forest"]["feature_importance"] = {
        k: round(v, 3) for k, v in importance.sort_values(ascending=False).items()
    }

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(rf, args.out / "random_forest.joblib")
    print(f"\nModel and metrics saved to {args.out}")


if __name__ == "__main__":
    main()
