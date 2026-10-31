"""Point-in-time transaction features for the simulated handbook dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_COLUMNS = ["TX_AMOUNT", "hour", "day_of_week"]
WINDOWS = (1, 7, 30)
HISTORY_COLUMNS = BASE_COLUMNS + [
    name
    for days in WINDOWS
    for name in (
        f"customer_count_{days}d",
        f"customer_mean_amount_{days}d",
        f"terminal_count_{days}d",
        f"terminal_feedback_count_{days}d",
        f"terminal_fraud_rate_{days}d",
    )
]
_REQUIRED = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
)


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate transactions and return a canonical chronological copy."""
    missing = set(_REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"):
        try:
            values = pd.to_numeric(result[column], errors="raise")
            numeric = values.to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {column}") from exc
        if (
            result[column].map(lambda value: isinstance(value, (bool, np.bool_))).any()
            or not np.isfinite(numeric).all()
            or (numeric < 0).any()
            or (numeric != np.floor(numeric)).any()
            or (numeric >= 2**63).any()
        ):
            raise ValueError(f"Invalid {column}: expected nonnegative integer IDs")
        result[column] = values.astype(np.int64)
    if result["TRANSACTION_ID"].duplicated().any():
        raise ValueError("TRANSACTION_ID must be unique")
    try:
        dates = pd.to_datetime(result["TX_DATETIME"], errors="raise", format="mixed")
        if not isinstance(dates.dtype, pd.DatetimeTZDtype) and pd.api.types.is_datetime64_dtype(
            dates.dtype
        ):
            result["TX_DATETIME"] = dates
        else:
            raise ValueError("TX_DATETIME must contain naive datetimes")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid TX_DATETIME: expected naive datetimes") from exc
    if result["TX_DATETIME"].isna().any():
        raise ValueError("TX_DATETIME must not contain missing values")
    for column in ("TX_AMOUNT", "TX_FRAUD"):
        try:
            values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {column}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid {column}: values must be finite")
        if column == "TX_AMOUNT" and (values < 0).any():
            raise ValueError("TX_AMOUNT must be nonnegative")
        if column == "TX_FRAUD" and not np.isin(values, [0, 1]).all():
            raise ValueError("TX_FRAUD must be binary")
        result[column] = values if column == "TX_AMOUNT" else values.astype(np.int8)
    return result.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)


def load_handbook(path: Path) -> pd.DataFrame:
    """Load and validate a CSV containing simulated transactions."""
    return validate_transactions(pd.read_csv(path))


def _window_stats(
    entities: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    delay_ns: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute per-entity counts and sums in reveal-time windows [t-W, t)."""
    n = len(times)
    result = {days: (np.zeros(n, dtype=np.int64), np.zeros(n)) for days in WINDOWS}
    order = np.lexsort((times, entities))
    sorted_entities = entities[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_entities[1:] != sorted_entities[:-1]) + 1, n]
    day_ns = int(pd.Timedelta(days=1).value)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        positions = order[start:end]
        event_times = times[positions]
        prefix = np.r_[0.0, np.cumsum(values[positions], dtype=float)]
        # Python integers avoid overflow in the subtraction at datetime limits.
        query = event_times.astype(object) - delay_ns
        right = np.searchsorted(event_times, query, side="left")
        for days in WINDOWS:
            left = np.searchsorted(event_times, query - days * day_ns, side="left")
            count, total = result[days]
            count[positions] = right - left
            total[positions] = prefix[right] - prefix[left]
    return result


def build_history_features(frame: pd.DataFrame, label_delay_days: int = 7) -> pd.DataFrame:
    """Build features as known at each event time, preserving input row order and index.

    Every event at the scoring timestamp is excluded. Feedback for both classes
    becomes available only strictly after event time plus the specified delay.
    """
    if (
        isinstance(label_delay_days, (bool, np.bool_))
        or not isinstance(label_delay_days, (int, np.integer))
        or label_delay_days < 0
    ):
        raise ValueError("label_delay_days must be a nonnegative integer")
    transactions = validate_transactions(frame)
    times = transactions["TX_DATETIME"].to_numpy(dtype="datetime64[ns]").view(np.int64)
    amount = transactions["TX_AMOUNT"].to_numpy(dtype=float)
    customer = _window_stats(transactions["CUSTOMER_ID"].to_numpy(), times, amount, 0)
    terminal = _window_stats(
        transactions["TERMINAL_ID"].to_numpy(), times, transactions["TX_FRAUD"].to_numpy(), 0
    )
    feedback = _window_stats(
        transactions["TERMINAL_ID"].to_numpy(),
        times,
        transactions["TX_FRAUD"].to_numpy(),
        int(label_delay_days) * int(pd.Timedelta(days=1).value),
    )
    features = pd.DataFrame(
        {
            "TX_AMOUNT": amount,
            "hour": transactions["TX_DATETIME"].dt.hour.to_numpy(),
            "day_of_week": transactions["TX_DATETIME"].dt.dayofweek.to_numpy(),
        }
    )
    for days in WINDOWS:
        customer_count, customer_sum = customer[days]
        terminal_count, _ = terminal[days]
        feedback_count, fraud_sum = feedback[days]
        features[f"customer_count_{days}d"] = customer_count
        features[f"customer_mean_amount_{days}d"] = np.divide(
            customer_sum, customer_count, out=np.zeros(len(frame)), where=customer_count != 0
        )
        features[f"terminal_count_{days}d"] = terminal_count
        features[f"terminal_feedback_count_{days}d"] = feedback_count
        features[f"terminal_fraud_rate_{days}d"] = np.divide(
            fraud_sum, feedback_count, out=np.zeros(len(frame)), where=feedback_count != 0
        )
    positions = pd.Index(transactions["TRANSACTION_ID"]).get_indexer(
        pd.to_numeric(frame["TRANSACTION_ID"]).astype(np.int64)
    )
    features = features.iloc[positions].copy()
    features.index = frame.index
    return features[HISTORY_COLUMNS]
