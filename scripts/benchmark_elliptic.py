"""Compare local and graph features on completed Elliptic snapshots."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from fraudguard.data import chronological_split
from fraudguard.elliptic import (
    LOCAL_COLUMNS,
    NEIGHBOR_COLUMNS,
    STRUCTURE_COLUMNS,
    build_graph_features,
    load_elliptic,
)
from scripts.benchmark import budget_metrics, file_sha256, metrics, validation_threshold
from scripts.download_elliptic import URL as DOWNLOAD_URL

FRACTIONS = (0.005, 0.01, 0.02, 0.05)
CAPACITIES = (5, 10, 25, 50)
HISTORICAL_FOLDS = ((19, 24, 29), (24, 29, 34), (29, 34, 39))

MODEL_PARAMS = {
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "learning_rate": 0.1,
    "l2_regularization": 1,
    "early_stopping": False,
    "random_state": 42,
}


def _metrics(y, scores, threshold, budget):
    result = metrics(y, scores, threshold)
    result.pop("top_1_percent")
    result["review_budget"] = budget_metrics(y, scores, budget)
    return result


def _counts(part):
    labelled = int((part.is_illicit != -1).sum())
    illicit = int((part.is_illicit == 1).sum())
    return {
        "rows": len(part),
        "labelled": labelled,
        "unknown": int((part.is_illicit == -1).sum()),
        "illicit": illicit,
        "licit": int((part.is_illicit == 0).sum()),
        "prevalence": illicit / labelled if labelled else None,
    }


def _workload(y, selected):
    reviewed = int(np.sum(selected))
    found = int(np.sum(y[selected]))
    positives = int(np.sum(y))
    return {
        "reviewed": reviewed,
        "frauds_found": found,
        "precision": found / reviewed if reviewed else None,
        "recall": found / positives if positives else None,
    }


def _ranked(y, scores, *, fraction=None, capacity=None):
    if len(y) == 0:
        return _workload(y, np.zeros(0, dtype=bool))
    if capacity is None:
        return budget_metrics(y, scores, fraction)
    k = min(capacity, len(y))
    # budget_metrics uses ceil(n * fraction); round down to avoid floating errors.
    return budget_metrics(y, scores, np.nextafter(k / len(y), 0))


def _review_policies(test, labelled, scores, threshold):
    y = labelled.is_illicit.to_numpy()
    pooled = [{"fraction": f, **_ranked(y, scores, fraction=f)} for f in FRACTIONS]
    per_step = {}
    for step in sorted(test.time_step.unique()):
        mask = labelled.time_step.to_numpy() == step
        ys, ss = y[mask], scores[mask]
        per_step[str(int(step))] = {
            "fractions": [{"fraction": f, **_ranked(ys, ss, fraction=f)} for f in FRACTIONS],
            "capacities": [{"capacity": k, **_ranked(ys, ss, capacity=k)} for k in CAPACITIES],
            "frozen_threshold": {"threshold": threshold, **_workload(ys, ss >= threshold)},
        }
    positives = int(y.sum())

    def summed(rows, key=None, value=None):
        reviewed = sum(row["reviewed"] for row in rows)
        found = sum(row["frauds_found"] for row in rows)
        return {
            **({key: value} if key else {}),
            "reviewed": reviewed,
            "frauds_found": found,
            "precision": found / reviewed if reviewed else None,
            "recall": found / positives if positives else None,
        }

    values = list(per_step.values())
    return {
        "pooled_fractions": pooled,
        "per_step": per_step,
        "summed_per_step": {
            "fractions": [
                summed([v["fractions"][i] for v in values], "fraction", f)
                for i, f in enumerate(FRACTIONS)
            ],
            "capacities": [
                summed([v["capacities"][i] for v in values], "capacity", k)
                for i, k in enumerate(CAPACITIES)
            ],
            "frozen_threshold": summed(
                [v["frozen_threshold"] for v in values], "threshold", threshold
            ),
        },
    }


def _split_info(part):
    return {
        **{key: value for key, value in _counts(part).items() if key != "prevalence"},
        "time_min": int(part.time_step.min()),
        "time_max": int(part.time_step.max()),
    }


def _historical_windows(frame, variants, params, review_budget):
    results = []
    for train_end, valid_end, eval_end in HISTORICAL_FOLDS:
        train = frame.loc[frame.time_step <= train_end]
        valid = frame.loc[(frame.time_step > train_end) & (frame.time_step <= valid_end)]
        evaluation = frame.loc[(frame.time_step > valid_end) & (frame.time_step <= eval_end)]
        labelled = [part.loc[part.is_illicit != -1] for part in (train, valid, evaluation)]
        for name, part in zip(("train", "validation", "evaluation"), labelled, strict=True):
            if part.empty or part.is_illicit.nunique() != 2:
                raise ValueError(f"historical {name} requires both labelled classes")
        train_l, valid_l, eval_l = labelled
        validations, evaluations = {}, {}
        for name, columns in variants.items():
            model = HistGradientBoostingClassifier(**params)
            model.fit(train_l[columns], train_l.is_illicit)
            valid_scores = model.predict_proba(valid_l[columns])[:, 1]
            threshold = validation_threshold(valid_l.is_illicit, valid_scores)
            validations[name] = _metrics(valid_l.is_illicit, valid_scores, threshold, review_budget)
            eval_scores = model.predict_proba(eval_l[columns])[:, 1]
            evaluations[name] = _metrics(eval_l.is_illicit, eval_scores, threshold, review_budget)
        results.append(
            {
                "splits": {
                    "train": _split_info(train),
                    "validation": _split_info(valid),
                    "evaluation": _split_info(evaluation),
                },
                "validation_metrics": validations,
                "evaluation_metrics": evaluations,
                "selected_variant": max(
                    variants, key=lambda name: validations[name]["average_precision"]
                ),
            }
        )
    return results


def run_benchmark(frame, edges, review_budget=0.01, *, model_params=None, historical_windows=False):
    if not math.isfinite(review_budget) or not 0 < review_budget <= 1:
        raise ValueError("review budget must be in (0, 1]")
    frame = frame.sort_values(["time_step", "txId"], kind="stable").reset_index(drop=True)
    if not frame.is_illicit.isin([-1, 0, 1]).all():
        raise ValueError("labels must be -1, 0 or 1")
    frame = pd.concat([frame, build_graph_features(frame, edges)], axis=1)
    train, valid, test = chronological_split(frame, "time_step", "is_illicit")
    train_labeled, valid_labeled, test_labeled = [
        part.loc[part.is_illicit != -1] for part in (train, valid, test)
    ]
    for name, part in (
        ("train", train_labeled),
        ("validation", valid_labeled),
        ("test", test_labeled),
    ):
        if part.empty or part.is_illicit.nunique() != 2:
            raise ValueError(f"{name} requires both labelled classes")
    variants = {
        "local": LOCAL_COLUMNS,
        "structure": LOCAL_COLUMNS + STRUCTURE_COLUMNS,
        "neighborhood": LOCAL_COLUMNS + STRUCTURE_COLUMNS + NEIGHBOR_COLUMNS,
    }
    params = dict(MODEL_PARAMS if model_params is None else model_params)
    validations, tests, windows, policies = {}, {}, {}, {}
    for name, columns in variants.items():
        model = HistGradientBoostingClassifier(**params)
        model.fit(train_labeled[columns], train_labeled.is_illicit)
        valid_scores = model.predict_proba(valid_labeled[columns])[:, 1]
        threshold = validation_threshold(valid_labeled.is_illicit, valid_scores)
        validations[name] = _metrics(
            valid_labeled.is_illicit, valid_scores, threshold, review_budget
        )
        test_scores = model.predict_proba(test_labeled[columns])[:, 1]
        tests[name] = _metrics(test_labeled.is_illicit, test_scores, threshold, review_budget)
        policies[name] = _review_policies(test, test_labeled, test_scores, threshold)
        windows[name] = {}
        for step in sorted(test.time_step.unique()):
            mask = test_labeled.time_step.to_numpy() == step
            y, scores = test_labeled.is_illicit.to_numpy()[mask], test_scores[mask]
            if len(y) == 0:
                windows[name][str(int(step))] = {
                    "labelled": 0,
                    "average_precision": None,
                    "roc_auc": None,
                    "review_budget": None,
                }
            elif len(np.unique(y)) == 1:
                windows[name][str(int(step))] = {
                    "labelled": len(y),
                    "average_precision": None,
                    "roc_auc": None,
                    "review_budget": budget_metrics(y, scores, review_budget),
                }
            else:
                windows[name][str(int(step))] = {
                    "labelled": len(y),
                    **_metrics(y, scores, threshold, review_budget),
                }
    selected = max(variants, key=lambda name: validations[name]["average_precision"])

    result = {
        "status": "completed",
        "dataset": "Elliptic",
        "seed": 42,
        "model": {"class": "HistGradientBoostingClassifier", "params": params},
        "protocol": "Completed-snapshot offline features; 60/20/20 distinct time steps split before filtering unknown labels; validation AP selects variant and F1 selects thresholds; no refit; all prespecified variants tested",
        "review_budget": review_budget,
        "features": variants,
        "splits": {
            "train": _split_info(train),
            "validation": _split_info(valid),
            "test": _split_info(test),
        },
        "validation": validations,
        "selected_variant": selected,
        "test": tests,
        "test_by_time_step": windows,
        "time_step_counts": {
            str(int(step)): _counts(part) for step, part in frame.groupby("time_step", sort=True)
        },
        "review_policy_config": {
            "fractions": list(FRACTIONS),
            "capacities": list(CAPACITIES),
            "cohort": "labelled rows only; unknown rows remain graph context",
            "tie_order": "time_step, txId for pooled queues; txId within each step",
            "interpretation": "pooled fraction queues and summed per-step queues have different workloads",
        },
        "test_review_policies": policies,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "sources": {
            "paper": "https://arxiv.org/abs/1908.02591",
            "download": DOWNLOAD_URL,
            "pyg": "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html",
            "original": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        },
        "limitations": [
            "Features use completed snapshots and are not a strictly online evaluation.",
            "Unknown labels are excluded from metrics; review budget applies only to the labelled cohort.",
            "Local features are anonymous and their real-time availability cannot be verified.",
            "The 72 upstream aggregate features and current time step are excluded from model inputs.",
            "One previously inspected final test period and unobserved label delay limit conclusions.",
            "No financial impact or deployment claim is made; scores are not calibrated probabilities.",
            "Historical windows are offline diagnostics assuming labels are available by each fit and validation; they do not establish causality or superiority.",
        ],
    }
    if historical_windows:
        result["historical_windows"] = _historical_windows(frame, variants, params, review_budget)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/elliptic"))
    parser.add_argument("--output", type=Path, default=Path("reports/elliptic.json"))
    parser.add_argument("--review-budget", type=float, default=0.01)
    parser.add_argument("--historical-windows", action="store_true")
    args = parser.parse_args()
    frame, edges = load_elliptic(args.data_dir)
    report = run_benchmark(
        frame, edges, args.review_budget, historical_windows=args.historical_windows
    )
    root = Path(__file__).resolve().parents[1]
    report["input_sha256"] = {
        name: file_sha256(args.data_dir / name)
        for name in (
            "elliptic_txs_features.csv",
            "elliptic_txs_classes.csv",
            "elliptic_txs_edgelist.csv",
        )
    }
    report["source_sha256"] = {
        name: file_sha256(root / name)
        for name in (
            "scripts/benchmark_elliptic.py",
            "scripts/benchmark.py",
            "scripts/plot_elliptic.py",
            "requirements-benchmark.txt",
            "scripts/download_elliptic.py",
            "scripts/download_creditcard.py",
            "fraudguard/elliptic.py",
            "fraudguard/data.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_variant": report["selected_variant"]}))


if __name__ == "__main__":
    main()
