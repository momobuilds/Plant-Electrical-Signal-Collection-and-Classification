#!/usr/bin/env python3
"""Train baseline classifiers for synthetic basil signal windows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

# Keep joblib quiet and portable on restricted macOS environments. Users may
# override this before running the script if they intentionally enable parallelism.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FILENAMES = {
    "Random Forest": "RF",
    "Extra Trees": "ETC",
    "KNN": "KNN",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic basil baseline models.")
    parser.add_argument(
        "--input-csv", type=Path, default=Path("data/processed/feature_dataset.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--leave-one-plant-out",
        action="store_true",
        help="Also train on all-but-one plant and test on each held-out plant.",
    )
    return parser.parse_args(argv)


def score_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision_score(y_true, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
    }


def build_models(random_state: int, knn_neighbors: int) -> dict[str, Pipeline]:
    return {
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=knn_neighbors)),
            ]
        ),
    }


def run_leave_one_plant_out(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    random_state: int,
) -> None:
    plants = sorted(frame["plant_id"].astype(str).unique())
    if len(plants) < 2:
        raise SystemExit("Leave-one-plant-out validation requires at least two plants.")
    rows: list[dict[str, float | str | int]] = []
    X = frame[feature_columns]
    y = frame["label"].astype(str)

    for held_out_plant in plants:
        test_mask = frame["plant_id"].astype(str) == held_out_plant
        training_size = int((~test_mask).sum())
        models = build_models(random_state, max(1, min(7, training_size)))
        for model_name, model in models.items():
            model.fit(X.loc[~test_mask], y.loc[~test_mask])
            predictions = model.predict(X.loc[test_mask])
            rows.append(
                {
                    "model": model_name,
                    "held_out_plant": held_out_plant,
                    "training_windows": training_size,
                    "test_windows": int(test_mask.sum()),
                    **score_predictions(y.loc[test_mask], predictions),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "leave_one_plant_out_metrics.csv", index=False)
    summary = metrics.groupby("model")[
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ].agg(["mean", "std"])
    summary.to_csv(output_dir / "leave_one_plant_out_summary.csv")
    print("\nLeave-one-plant-out summary:")
    print(summary.to_string())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cv_folds < 2:
        raise SystemExit("--cv-folds must be at least 2.")
    if not args.input_csv.is_file():
        raise SystemExit(f"Feature dataset does not exist: {args.input_csv}")

    frame = pd.read_csv(args.input_csv)
    required = {"label", "plant_id", "is_synthetic"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Feature dataset is missing columns: {sorted(missing)}")
    feature_columns = [
        column
        for column in frame.columns
        if column.startswith("raw_adc_") or column.startswith("voltage_")
    ]
    if not feature_columns:
        raise SystemExit("No raw_adc_* or voltage_* features were found.")
    if frame.empty:
        raise SystemExit("Feature dataset is empty.")

    X = frame[feature_columns]
    y = frame["label"].astype(str)
    class_counts = y.value_counts()
    if class_counts.size < 2 or int(class_counts.min()) < 2:
        raise SystemExit("Training requires at least two classes with two windows each.")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=args.random_state,
        stratify=y,
    )
    cv_folds = min(args.cv_folds, int(class_counts.min()))
    if cv_folds < args.cv_folds:
        print(
            f"WARNING: using {cv_folds}-fold CV because the smallest class has "
            f"{int(class_counts.min())} windows.",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(y.unique())
    minimum_training_size = min(len(X_train), len(X) - math_ceil_div(len(X), cv_folds))
    models = build_models(args.random_state, max(1, min(7, minimum_training_size)))
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    metric_rows: list[dict[str, float | str | int]] = []
    report_sections: list[str] = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_scores = score_predictions(y_test, predictions)
        # Sequential folds avoid nested process pools in restricted environments.
        cv_scores = cross_validate(model, X, y, cv=splitter, scoring=scoring, n_jobs=1)
        metric_rows.append(
            {
                "model": model_name,
                "test_accuracy": test_scores["accuracy"],
                "test_precision_macro": test_scores["precision_macro"],
                "test_recall_macro": test_scores["recall_macro"],
                "test_f1_macro": test_scores["f1_macro"],
                "cv_folds_used": cv_folds,
                "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
                "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
                "cv_precision_macro_mean": float(np.mean(cv_scores["test_precision_macro"])),
                "cv_recall_macro_mean": float(np.mean(cv_scores["test_recall_macro"])),
                "cv_f1_macro_mean": float(np.mean(cv_scores["test_f1_macro"])),
            }
        )
        report_sections.append(
            f"{model_name}\n{'=' * len(model_name)}\n"
            + classification_report(y_test, predictions, labels=labels, zero_division=0)
        )
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        display = ConfusionMatrixDisplay(matrix, display_labels=labels)
        display.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"{model_name} confusion matrix")
        plt.tight_layout()
        matrix_filename = f"confusion_matrix_{MODEL_FILENAMES[model_name]}.png"
        plt.savefig(args.output_dir / matrix_filename, dpi=160)
        if args.figures_dir.resolve() != args.output_dir.resolve():
            plt.savefig(args.figures_dir / matrix_filename, dpi=160)
        plt.close()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "results_metrics.csv", index=False)
    (args.output_dir / "classification_report.txt").write_text(
        "\n\n".join(report_sections) + "\n", encoding="utf-8"
    )
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(metrics.to_string(index=False))

    if args.leave_one_plant_out:
        run_leave_one_plant_out(frame, feature_columns, args.output_dir, args.random_state)

    print(f"\nSaved metrics to {args.output_dir}")
    print(f"Saved confusion matrices to {args.figures_dir}")
    return 0


def math_ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


if __name__ == "__main__":
    raise SystemExit(main())
