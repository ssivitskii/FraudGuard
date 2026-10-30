"""Compare IEEE-CIS feature ablations on a future period at one review budget."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from fraudguard.data import chronological_split
from fraudguard.ieee import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    GRAPH_COLUMNS,
    HISTORY_COLUMNS,
    build_history_features,
    load_ieee,
)
from scripts.benchmark import budget_metrics, file_sha256, metrics, validation_threshold

MODEL_PARAMS = {
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "learning_rate": 0.1,
    "l2_regularization": 1,
    "early_stopping": False,
    "random_state": 42,
}


def run_benchmark(frame, delay_days=7, review_budget=0.01, *, model_params=None):
    if not math.isfinite(delay_days) or delay_days < 0:
        raise ValueError("label delay must be finite and nonnegative")
    if not math.isfinite(review_budget) or not 0 < review_budget <= 1:
        raise ValueError("review budget must be in (0, 1]")
    # Stable order also fixes label-independent tie breaking for every variant.
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(
        drop=True
    )
    history = build_history_features(frame)
    frame = pd.concat([frame, history], axis=1)
    train, valid, test = chronological_split(frame, "TransactionDT", "isFraud")
    delay = delay_days * 86400
    train_eligible = train.loc[train.TransactionDT + delay <= valid.TransactionDT.min()]
    valid_eligible = valid.loc[valid.TransactionDT + delay <= test.TransactionDT.min()]
    for name, part in (("train", train_eligible), ("validation", valid_eligible), ("test", test)):
        if part.empty or not part.isFraud.isin([0, 1]).all() or part.isFraud.nunique() != 2:
            raise ValueError(f"{name} must contain both binary classes after label-delay filtering")

    variants = {
        "tabular": BASE_NUMERIC,
        "history": BASE_NUMERIC + HISTORY_COLUMNS,
        "graph": BASE_NUMERIC + HISTORY_COLUMNS + GRAPH_COLUMNS,
    }
    params = dict(MODEL_PARAMS if model_params is None else model_params)
    validations, tests = {}, {}
    for name, numeric in variants.items():
        preprocess = ColumnTransformer(
            [
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                        encoded_missing_value=np.nan,
                        max_categories=255,
                    ),
                    BASE_CATEGORICAL,
                ),
                ("numeric", "passthrough", numeric),
            ]
        )
        model = make_pipeline(
            preprocess,
            HistGradientBoostingClassifier(
                categorical_features=[True] * len(BASE_CATEGORICAL) + [False] * len(numeric),
                **params,
            ),
        )
        model.fit(train_eligible, train_eligible.isFraud)
        valid_scores = model.predict_proba(valid_eligible)[:, 1]
        threshold = validation_threshold(valid_eligible.isFraud, valid_scores)
        validations[name] = _metrics(valid_eligible.isFraud, valid_scores, threshold, review_budget)
        test_scores = model.predict_proba(test)[:, 1]
        tests[name] = _metrics(test.isFraud, test_scores, threshold, review_budget)
    selected = max(variants, key=lambda name: validations[name]["average_precision"])

    def split_info(part, eligible):
        return {
            "rows": len(part),
            "eligible_labels": len(eligible),
            "frauds_eligible": int(eligible.isFraud.sum()),
            "time_min": float(part.TransactionDT.min()),
            "time_max": float(part.TransactionDT.max()),
        }

    return {
        "status": "completed",
        "dataset": "IEEE-CIS",
        "seed": 42,
        "model": {"class": "HistGradientBoostingClassifier", "params": params},
        "protocol": "60/20/20 distinct timestamp groups; train and validation labels filtered by fixed delay; validation AP selects variant; validation F1 selects thresholds; no refit; all ablations evaluated on test",
        "label_delay_days": delay_days,
        "review_budget": review_budget,
        "features": {name: BASE_CATEGORICAL + numeric for name, numeric in variants.items()},
        "splits": {
            "train": split_info(train, train_eligible),
            "validation": split_info(valid, valid_eligible),
            "test": split_info(test, test),
        },
        "validation": validations,
        "selected_variant": selected,
        "test": tests,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "Card and device signatures are anonymous proxies, not verified entity IDs.",
            "Restricted base features exclude upstream engineered V/C/D features.",
            "One historical split cannot establish general superiority.",
            "Fixed label delay is a simulation, not observed label latency.",
            "Top-k is a retrospective batch review budget, not an online policy.",
            "Model scores are not calibrated probabilities.",
        ],
    }


def _metrics(y, scores, threshold, budget):
    result = metrics(y, scores, threshold)
    result.pop("top_1_percent")
    result["review_budget"] = budget_metrics(y, scores, budget)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ieee-cis"))
    parser.add_argument("--output", type=Path, default=Path("reports/ieee.json"))
    parser.add_argument("--label-delay-days", type=float, default=7)
    parser.add_argument("--review-budget", type=float, default=0.01)
    args = parser.parse_args()
    report = run_benchmark(load_ieee(args.data_dir), args.label_delay_days, args.review_budget)
    root = Path(__file__).resolve().parents[1]
    report["input_sha256"] = {
        name: file_sha256(args.data_dir / name)
        for name in ("train_transaction.csv", "train_identity.csv")
    }
    report["source_sha256"] = {
        name: file_sha256(root / name)
        for name in (
            "scripts/benchmark_ieee.py",
            "scripts/benchmark.py",
            "fraudguard/ieee.py",
            "fraudguard/data.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_variant": report["selected_variant"]}))


if __name__ == "__main__":
    main()
