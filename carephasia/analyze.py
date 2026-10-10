"""Analyse a speech transcript and explain the model's output.

Usage:
    python -m carephasia.analyze examples/sample_transcript.txt

This is a research prototype, NOT a diagnostic tool. A diagnosis of aphasia
can only be made by a qualified health professional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS, extract_features

ROOT = Path(__file__).resolve().parents[1]

# True if a HIGHER value is typical of aphasic speech in our data
HIGHER_IS_APHASIC = {
    "average_tree_depth": False,
    "average_lexical_density": False,
    "word_stutter_count": True,
    "syllable_stutter_ratio": False,
}


def explain(features: dict[str, float], control_means: dict[str, float]) -> list[str]:
    lines = []
    for name in FEATURE_COLUMNS:
        value, reference = features[name], control_means[name]
        higher = value > reference
        verdict = "consistent with" if higher == HIGHER_IS_APHASIC[name] else "not consistent with"
        lines.append(
            f"{name} = {value:.2f} ({'above' if higher else 'below'} the control mean "
            f"{reference:.2f}) -> {verdict} an aphasic pattern"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to a .txt transcript (English)")
    parser.add_argument("--model", type=Path, default=ROOT / "results" / "random_forest.joblib")
    parser.add_argument("--averages", type=Path, default=ROOT / "data" / "group_averages.csv")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit("Model not found - run `python -m carephasia.train` first.")

    text = args.transcript.read_text(encoding="utf-8")
    features = extract_features(text)

    model = joblib.load(args.model)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    patient_idx = list(model.classes_).index("patient")
    p_patient = float(model.predict_proba(X)[0][patient_idx])

    averages = pd.read_csv(args.averages).set_index("diagnosis")
    explanation = explain(features, averages.loc["control"].to_dict())

    if args.json:
        print(json.dumps({"features": features, "p_aphasia_pattern": p_patient,
                          "explanation": explanation}, indent=2))
        return

    print(f"\nProbability of an aphasia-like linguistic pattern: {p_patient:.0%}\n")
    for line in explanation:
        print(" -", line)
    print("\nReminder: this is a research prototype, not a medical diagnosis.")


if __name__ == "__main__":
    main()
