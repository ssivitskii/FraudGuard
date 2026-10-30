"""IEEE-CIS loading and strictly prior transaction features.

Card and device tuples are coarse signatures, not verified entity identifiers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_NUMERIC = ["TransactionAmt"]
BASE_CATEGORICAL = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]
HISTORY_COLUMNS = [
    "card_prior_count",
    "card_prior_mean_amount",
    "card_seconds_since_last",
    "device_prior_count",
    "device_prior_mean_amount",
    "device_seconds_since_last",
]
GRAPH_COLUMNS = ["card_prior_unique_devices", "device_prior_unique_cards", "pair_prior_count"]
_CARD = [f"card{i}" for i in range(1, 7)]
_DEVICE = ["DeviceType", "DeviceInfo"]


def load_ieee(data_dir: Path) -> pd.DataFrame:
    """Load the labelled transaction data and optional identity attributes."""
    transaction_cols = (
        ["TransactionID", "TransactionDT", "isFraud"]
        + BASE_NUMERIC
        + [col for col in BASE_CATEGORICAL if col not in _DEVICE]
    )
    identity_cols = ["TransactionID"] + _DEVICE
    transaction = pd.read_csv(
        data_dir / "train_transaction.csv",
        usecols=transaction_cols,
        dtype={col: str for col in BASE_CATEGORICAL if col not in _DEVICE},
    )
    identity = pd.read_csv(
        data_dir / "train_identity.csv",
        usecols=identity_cols,
        dtype=dict.fromkeys(_DEVICE, str),
    )
    for frame in (transaction, identity):
        ids = pd.to_numeric(frame["TransactionID"], errors="coerce")
        if ids.isna().any() or not np.isfinite(ids).all() or (ids % 1 != 0).any():
            raise ValueError("TransactionID must contain finite integers")
        if ids.duplicated().any():
            raise ValueError("TransactionID must be unique in each file")
        frame["TransactionID"] = ids
    if not identity["TransactionID"].isin(transaction["TransactionID"]).all():
        raise ValueError("identity contains unknown TransactionID")
    for col in ["TransactionDT", "TransactionAmt"]:
        values = pd.to_numeric(transaction[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{col} must contain finite nonnegative numbers")
        transaction[col] = values
    labels = pd.to_numeric(transaction["isFraud"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("isFraud must contain binary labels")
    transaction["isFraud"] = labels.astype(int)
    frame = transaction.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    for col in BASE_CATEGORICAL:
        frame[col] = frame[col].where(frame[col].notna(), np.nan)
    return frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(
        drop=True
    )


def build_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute features before each time batch, retaining input row alignment."""
    columns = ["TransactionDT", "TransactionAmt"] + _CARD + _DEVICE
    rows = frame[columns].itertuples(index=False, name=None)
    events = []
    for pos, (time, amount, *values) in enumerate(rows):
        card_values, device_values = values[:6], values[6:]
        card = tuple(card_values) if all(pd.notna(x) for x in card_values) else None
        device = tuple(device_values) if all(pd.notna(x) for x in device_values) else None
        events.append((time, pos, amount, card, device))
    events.sort(key=lambda event: event[0])
    result = np.full((len(frame), len(HISTORY_COLUMNS) + len(GRAPH_COLUMNS)), np.nan)
    card_stats, device_stats = {}, {}
    card_devices, device_cards, pairs = {}, {}, {}
    start = 0
    while start < len(events):
        end = start + 1
        time = events[start][0]
        while end < len(events) and events[end][0] == time:
            end += 1
        for _, pos, _, card, device in events[start:end]:
            for offset, key, stats in ((0, card, card_stats), (3, device, device_stats)):
                count, total, last = (
                    stats.get(key, (0, 0.0, None)) if key is not None else (0, 0.0, None)
                )
                result[pos, offset : offset + 3] = (
                    count,
                    total / count if count else np.nan,
                    time - last if count else np.nan,
                )
            result[pos, 6:] = (
                len(card_devices.get(card, ())) if card is not None else 0,
                len(device_cards.get(device, ())) if device is not None else 0,
                pairs.get((card, device), 0) if card is not None and device is not None else 0,
            )
        for _, _, amount, card, device in events[start:end]:
            for key, stats in ((card, card_stats), (device, device_stats)):
                if key is not None:
                    count, total, _ = stats.get(key, (0, 0.0, None))
                    stats[key] = (count + 1, total + amount, time)
            if card is not None and device is not None:
                card_devices.setdefault(card, set()).add(device)
                device_cards.setdefault(device, set()).add(card)
                pairs[(card, device)] = pairs.get((card, device), 0) + 1
        start = end
    return pd.DataFrame(result, index=frame.index, columns=HISTORY_COLUMNS + GRAPH_COLUMNS)
