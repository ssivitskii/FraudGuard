"""Chronological ULB benchmark, with model/threshold selection on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudguard.data import chronological_split


def validated_arrays(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or len(y) == 0:
        raise ValueError("Labels and scores must be nonempty aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    return y, scores


def budget_metrics(y, scores, fraction: float = 0.01) -> dict:
    """Exact top-k workload; tie break uses input row order, independent of labels."""
    y, scores = validated_arrays(y, scores)
    if not 0 < fraction <= 1:
        raise ValueError("Review budget must be in (0, 1]")
    k = max(1, math.ceil(len(y) * fraction))
    top = np.argsort(-scores, kind="stable")[:k]
    found = int(y[top].sum())
    return {
        "reviewed": k,
        "frauds_found": found,
        "precision": found / k,
        "recall": found / int(y.sum()) if y.sum() else None,
    }


def validation_threshold(y, scores) -> float:
    y, scores = validated_arrays(y, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Validation threshold requires both classes")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def metrics(y, scores, threshold: float) -> dict:
    y, scores = validated_arrays(y, scores)
    if len(np.unique(y)) != 2 or not np.isfinite(threshold):
        raise ValueError("Metrics require both classes and a finite threshold")
    predicted = np.asarray(scores) >= threshold
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": threshold,
        "precision_at_frozen_threshold": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_frozen_threshold": float(recall_score(y, predicted, zero_division=0)),
        "review_fraction_at_frozen_threshold": float(predicted.mean()),
        "top_1_percent": budget_metrics(y, scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/creditcard.json"))
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if not {"Class", "Time", *features}.issubset(frame.columns):
        raise ValueError("Expected the original ULB Time/V1..V28/Amount/Class schema")
    if (
        not frame["Class"].isin([0, 1]).all()
        or not np.isfinite(frame[["Time", *features]].to_numpy()).all()
    ):
        raise ValueError("Expected finite features and binary labels")
    train, valid, test = chronological_split(frame, "Time", "Class")
    for part in (train, valid, test):
        if part.Class.nunique() != 2:
            raise ValueError("Each time split must contain both classes")
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=42,
        ),
    }
    validations, thresholds = {}, {}
    for name, model in models.items():
        model.fit(train[features], train.Class)
        scores = model.predict_proba(valid[features])[:, 1]
        thresholds[name] = validation_threshold(valid.Class, scores)
        validations[name] = metrics(valid.Class, scores, thresholds[name])
    selected = max(models, key=lambda name: validations[name]["average_precision"])
    # No refit after selection: threshold and preprocessing still match the frozen model.
    scores = models[selected].predict_proba(test[features])[:, 1]
    report = {
        "status": "completed",
        "dataset": "ULB creditcard",
        "seed": 42,
        "dataset_sha256": file_sha256(args.data),
        "benchmark_source_sha256": file_sha256(Path(__file__)),
        "protocol": "60/20/20 distinct timestamp groups; model by validation AP; threshold by validation F1; selected model tested once; no refit",
        "features": features,
        "time_excluded_from_features": True,
        "splits": {
            name: {
                "rows": len(part),
                "frauds": int(part.Class.sum()),
                "prevalence": float(part.Class.mean()),
                "time_min": float(part.Time.min()),
                "time_max": float(part.Time.max()),
            }
            for name, part in zip(
                ("train", "validation", "test"), (train, valid, test), strict=True
            )
        },
        "validation": validations,
        "selected_model": selected,
        "test": metrics(test.Class, scores, thresholds[selected]),
        "eda": {
            "rows": len(frame),
            "missing_values": int(frame.isna().sum().sum()),
            "frauds": int(frame.Class.sum()),
            "amount_median": float(frame.Amount.median()),
            "amount_p99": float(frame.Amount.quantile(0.99)),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "Only two days of transactions; no card/entity IDs for group holdout.",
            "Original PCA features were produced upstream; their fitting provenance is outside this benchmark.",
            "Historical selected data, not a deployment or financial impact estimate.",
            "Top-1% is a batch capacity ranking metric; a frozen threshold has a variable workload.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_model": selected, "test": report["test"]}, indent=2))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
