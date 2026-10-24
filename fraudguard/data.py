from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw_data(fname: str = "transactions.csv") -> pd.DataFrame:
    path = DATA_DIR / "raw" / fname
    logger.info("Loading data from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Please download the dataset and place it in data/raw/"
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def train_valid_test_split(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    test_size: float = 0.2,
    valid_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info("Target distribution: %s", y.value_counts().to_dict())

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    valid_ratio = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_ratio, stratify=y_temp, random_state=random_state
    )

    logger.info(
        "Split sizes - Train: %d, Valid: %d, Test: %d",
        len(X_train),
        len(X_valid),
        len(X_test),
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def chronological_split(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    train_fraction: float = 0.6,
    valid_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split distinct timestamps; equal-time transactions never cross boundaries."""
    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1 - train_fraction:
        raise ValueError("Fractions must leave nonempty train, validation and test")
    if time_col not in df or target_col not in df:
        raise KeyError("Time and target columns are required")
    if df[time_col].isna().any() or df[target_col].isna().any():
        raise ValueError("Time and target cannot contain missing values")
    if pd.api.types.is_numeric_dtype(df[time_col]):
        if not np.isfinite(df[time_col].to_numpy()).all():
            raise ValueError("Numeric timestamps must be finite")
    elif not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise ValueError("Time column must be numeric or datetime; parse string timestamps first")
    ordered = df.sort_values(time_col, kind="stable")
    times = ordered[time_col].drop_duplicates().to_numpy()
    first = int(len(times) * train_fraction)
    second = int(len(times) * (train_fraction + valid_fraction))
    if not 0 < first < second < len(times):
        raise ValueError("At least three distinct timestamps are required")
    return (
        ordered.loc[ordered[time_col] < times[first]].copy(),
        ordered.loc[
            (ordered[time_col] >= times[first]) & (ordered[time_col] < times[second])
        ].copy(),
        ordered.loc[ordered[time_col] >= times[second]].copy(),
    )
