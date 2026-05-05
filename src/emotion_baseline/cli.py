from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .data import EMOTIONS, load_samples
from .features import extract_features
from .model import search_best_model
from .plots import save_confusion_matrix, save_feature_overview


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a speech emotion classification baseline.")
    parser.add_argument("--data-dir", required=True, help="Directory containing audio files.")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional CSV file with 'path' and 'label' columns.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "svm", "random_forest", "knn", "logistic_regression"],
        help="Model family to use.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate.")
    parser.add_argument("--out-dir", default="results", help="Output directory.")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data_dir, args.metadata_csv)
    label_counts = Counter(sample.label for sample in samples)
    if len(label_counts) < 2:
        raise ValueError("At least two emotion classes are required for training.")
    if min(label_counts.values()) < 2:
        raise ValueError(
            "Each emotion class must have at least 2 samples for a stratified train/test split."
        )

    rows = []
    for sample in samples:
        feature_row = extract_features(sample.path, sample_rate=args.sample_rate)
        feature_row["label"] = sample.label
        feature_row["path"] = str(sample.path)
        rows.append(feature_row)

    frame = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    frame.to_csv(out_dir / "feature_table.csv", index=False)

    X = frame.drop(columns=["label", "path"])
    y = frame["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    min_train_class_count = int(y_train.value_counts().min())
    cv = max(2, min(5, min_train_class_count))
    search = search_best_model(X_train, y_train, model_name=args.model, cv=cv)

    y_pred = search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4)

    save_confusion_matrix(y_test, y_pred, labels=list(EMOTIONS), output_path=out_dir / "confusion_matrix.png")
    save_feature_overview(frame, output_path=out_dir / "feature_overview.png")

    metrics = {
        "num_samples": len(frame),
        "class_distribution": dict(label_counts),
        "model_requested": args.model,
        "best_model_score_cv": float(search.best_score_),
        "best_params": search.best_params_,
        "test_accuracy": float(accuracy),
        "test_size": args.test_size,
        "sample_rate": args.sample_rate,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    print(json.dumps(metrics, indent=2))
    print()
    print(report_text)
    return 0

