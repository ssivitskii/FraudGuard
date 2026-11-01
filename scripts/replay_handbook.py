"""Replay the simulated Handbook transactions with delayed feedback and daily review."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudguard.handbook import (
    BASE_COLUMNS,
    HISTORY_COLUMNS,
    build_history_features,
    load_handbook,
    validate_transactions,
)
from scripts.benchmark import budget_metrics, file_sha256
from scripts.download_handbook import ARCHIVE_SHA256, CSV_SHA256, SOURCE_COMMIT, URL

DEFAULT_MODEL_PARAMS = {
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "learning_rate": 0.1,
    "l2_regularization": 1,
    "early_stopping": False,
    "random_state": 42,
}


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def run_replay(
    frame,
    *,
    start="2018-06-18",
    evaluation_days=28,
    train_days=28,
    refit_days=7,
    label_delays=(0, 7),
    model_params=None,
):
    """Fit at scheduled boundaries and rank each day's arrivals after the day ends."""
    for name, value in (
        ("evaluation_days", evaluation_days),
        ("train_days", train_days),
        ("refit_days", refit_days),
    ):
        _positive_integer(value, name)
    delays = tuple(label_delays)
    if (
        not delays
        or any(
            isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 0
            for d in delays
        )
        or len(set(delays)) != len(delays)
    ):
        raise ValueError("label_delays must contain unique nonnegative integers")
    try:
        start = pd.Timestamp(start)
    except (TypeError, ValueError) as exc:
        raise ValueError("start must be a naive calendar date") from exc
    if pd.isna(start) or start.tz is not None or start != start.normalize():
        raise ValueError("start must be a naive calendar date")
    frame = validate_transactions(frame)
    if frame.empty:
        raise ValueError("transactions must not be empty")
    end = start + pd.Timedelta(days=evaluation_days)
    earliest = start - pd.Timedelta(days=train_days + 2 * max(delays) + 30)
    dates = pd.DatetimeIndex(frame.TX_DATETIME.dt.normalize().unique())
    required_dates = pd.date_range(earliest, end - pd.Timedelta(days=1), freq="D")
    if not required_dates.isin(dates).all():
        raise ValueError(
            "Insufficient daily coverage for training, delayed feedback history, or evaluation"
        )
    params = DEFAULT_MODEL_PARAMS.copy()
    if model_params is not None:
        params.update(model_params)
    results = {}
    times = frame.TX_DATETIME
    for delay in delays:
        features = build_history_features(frame, label_delay_days=delay)
        delay_results = {}
        for variant, columns in (("basic", BASE_COLUMNS), ("history", HISTORY_COLUMNS)):
            refits, daily = [], []
            for offset in range(0, evaluation_days, refit_days):
                fit_time = start + pd.Timedelta(days=offset)
                train_end = fit_time - pd.Timedelta(days=int(delay))
                train_start = train_end - pd.Timedelta(days=train_days)
                training = (times >= train_start) & (times < train_end)
                y_train = frame.loc[training, "TX_FRAUD"]
                if y_train.nunique() != 2:
                    raise ValueError(f"Training window at {fit_time} requires both classes")
                model = HistGradientBoostingClassifier(**params)
                model.fit(features.loc[training, columns], y_train)
                available = times.loc[training] + pd.Timedelta(days=int(delay))
                refits.append(
                    {
                        "fit_time": fit_time.isoformat(),
                        "train_start": train_start.isoformat(),
                        "train_end": train_end.isoformat(),
                        "rows": int(training.sum()),
                        "frauds": int(y_train.sum()),
                        "latest_label_available_at": available.max().isoformat(),
                    }
                )
                block_end = min(fit_time + pd.Timedelta(days=refit_days), end)
                block = (times >= fit_time) & (times < block_end)
                block_frame = frame.loc[block]
                scores = model.predict_proba(features.loc[block, columns])[:, 1]
                for date, group in block_frame.groupby(
                    block_frame.TX_DATETIME.dt.normalize(), sort=True
                ):
                    positions = block_frame.index.get_indexer(group.index)
                    y = group.TX_FRAUD.to_numpy()
                    day_scores = scores[positions]
                    valid = len(np.unique(y)) == 2
                    daily.append(
                        {
                            "date": date.date().isoformat(),
                            "rows": len(group),
                            "frauds": int(y.sum()),
                            "average_precision": float(average_precision_score(y, day_scores))
                            if valid
                            else None,
                            "roc_auc": float(roc_auc_score(y, day_scores)) if valid else None,
                            "top_1_percent": budget_metrics(y, day_scores),
                        }
                    )
            reviewed = sum(day["top_1_percent"]["reviewed"] for day in daily)
            found = sum(day["top_1_percent"]["frauds_found"] for day in daily)
            frauds = sum(day["frauds"] for day in daily)
            aps = [
                day["average_precision"] for day in daily if day["average_precision"] is not None
            ]
            delay_results[variant] = {
                "refits": refits,
                "daily": daily,
                "aggregate": {
                    "rows": sum(day["rows"] for day in daily),
                    "frauds": frauds,
                    "reviewed": reviewed,
                    "frauds_found": found,
                    "precision": found / reviewed,
                    "recall": found / frauds if frauds else None,
                    "mean_daily_average_precision": float(np.mean(aps)) if aps else None,
                    "valid_ap_days": len(aps),
                },
            }
        results[str(delay)] = delay_results
    return {
        "status": "completed",
        "dataset": "Fraud Detection Handbook simulated transactions",
        "input_summary": {
            "rows": len(frame),
            "frauds": int(frame.TX_FRAUD.sum()),
            "time_min": times.min().isoformat(),
            "time_max": times.max().isoformat(),
        },
        "protocol": {
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "evaluation_days": evaluation_days,
            "train_days": train_days,
            "refit_days": refit_days,
            "label_delays_days": list(delays),
            "features": {"basic": list(BASE_COLUMNS), "history": list(HISTORY_COLUMNS)},
            "model": "HistGradientBoostingClassifier",
            "model_params": params,
            "review": "Top 1% per day, ranked at end of day; ties by event time and transaction ID",
            "assumptions": [
                "Synthetic uniform label delay for both classes: label_available_at = event time + delay; usable strictly before score or fit time",
                "Zero delay is an optimistic control; seven-day delay is simulated",
                "Delay jointly changes training recency and feedback features",
                "Training features are computed at original event time",
                "Metrics use outcomes after all labels are revealed",
                "Daily batch review does not give early feedback or intervene; it is not instant blocking or live deployment",
                "Coverage includes 30 days plus label delay before the earliest training event, in addition to the delay before each fit",
            ],
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2018-06-18")
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()
    report = run_replay(load_handbook(args.data), start=args.start, evaluation_days=args.days)
    root = Path(__file__).resolve().parents[1]
    dataset_sha256 = file_sha256(args.data)
    report["provenance"] = {
        "dataset_sha256": dataset_sha256,
        "official_reference": {
            "source_commit": SOURCE_COMMIT,
            "url": URL,
            "archive_sha256": ARCHIVE_SHA256,
            "expected_csv_sha256": CSV_SHA256,
            "input_matches_expected_csv": dataset_sha256 == CSV_SHA256,
        },
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                root / "fraudguard/handbook.py",
                root / "scripts/benchmark.py",
                root / "scripts/download_handbook.py",
                root / "requirements-benchmark.txt",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
