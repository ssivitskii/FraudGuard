import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fraudguard.ieee import (
    BASE_CATEGORICAL,
    GRAPH_COLUMNS,
    HISTORY_COLUMNS,
    build_history_features,
    load_ieee,
)


def event(time, amount, card, device, label=0):
    row = {f"card{i}": card for i in range(1, 7)}
    row.update(
        TransactionDT=time,
        TransactionAmt=amount,
        DeviceType=device,
        DeviceInfo=device,
        isFraud=label,
    )
    return row


def test_prior_features_and_simultaneous_events():
    frame = pd.DataFrame(
        [
            event(1, 10, "a", "x"),
            event(2, 20, "a", "y"),
            event(2, 30, "a", "x"),
            event(3, 40, "b", "x"),
            event(4, 50, "a", "x"),
            event(5, 60, None, None),
            event(6, 70, None, None),
        ]
    )
    actual = build_history_features(frame)
    assert list(actual.columns) == HISTORY_COLUMNS + GRAPH_COLUMNS
    np.testing.assert_allclose(
        actual.loc[1, HISTORY_COLUMNS].to_numpy(), [1, 10, 1, 0, np.nan, np.nan], equal_nan=True
    )
    assert actual.loc[2, "pair_prior_count"] == 1
    assert actual.loc[3, GRAPH_COLUMNS].tolist() == [0, 1, 0]
    assert actual.loc[4, HISTORY_COLUMNS].tolist() == [3, 20, 2, 3, 80 / 3, 1]
    assert actual.loc[4, GRAPH_COLUMNS].tolist() == [2, 2, 2]
    assert (
        actual.loc[5:6, ["card_prior_count", "device_prior_count"] + GRAPH_COLUMNS]
        .eq(0)
        .all()
        .all()
    )
    assert (
        actual.loc[5:6, ["card_prior_mean_amount", "device_seconds_since_last"]].isna().all().all()
    )


def test_order_alignment_future_and_labels_do_not_leak():
    frame = pd.DataFrame(
        [
            event(1, 10, "a", "x"),
            event(2, 20, "a", "x"),
            event(2, 30, "a", "y"),
            event(3, 40, "a", "x"),
        ],
        index=[8, 4, 9, 1],
    )
    expected = build_history_features(frame)
    shuffled = frame.iloc[[2, 0, 3, 1]].copy()
    shuffled["isFraud"] = 1
    assert_frame_equal(build_history_features(shuffled).loc[frame.index], expected)
    assert_frame_equal(build_history_features(frame.iloc[:3]), expected.iloc[:3])


def write_files(tmp_path):
    transaction = pd.DataFrame(
        [
            {"TransactionID": 2, "TransactionDT": 2, "TransactionAmt": 4, "isFraud": 1},
            {"TransactionID": 1, "TransactionDT": 1, "TransactionAmt": 3, "isFraud": 0},
        ]
    )
    for col in BASE_CATEGORICAL[:-2]:
        transaction[col] = [1, np.nan]
    identity = pd.DataFrame({"TransactionID": [2], "DeviceType": ["desktop"], "DeviceInfo": ["x"]})
    transaction.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    return transaction, identity


def test_load_left_join_sort_and_categorical_normalization(tmp_path):
    write_files(tmp_path)
    frame = load_ieee(tmp_path)
    assert frame.TransactionID.tolist() == [1, 2]
    assert pd.isna(frame.loc[0, "DeviceType"])
    assert frame.loc[1, "DeviceType"] == "desktop"
    assert isinstance(frame.loc[1, "card1"], str)
    assert pd.isna(frame.loc[0, "card1"])


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_transaction",
        "duplicate_identity",
        "unknown_identity",
        "noninteger_id",
        "bad_time",
        "bad_amount",
        "bad_label",
        "missing_column",
    ],
)
def test_rejects_invalid_schema(tmp_path, kind):
    transaction, identity = write_files(tmp_path)
    if kind == "duplicate_transaction":
        transaction.loc[1, "TransactionID"] = 2
    elif kind == "duplicate_identity":
        identity = pd.concat([identity, identity], ignore_index=True)
    elif kind == "unknown_identity":
        identity.loc[0, "TransactionID"] = 99
    elif kind == "noninteger_id":
        transaction["TransactionID"] = transaction["TransactionID"].astype(float)
        transaction.loc[0, "TransactionID"] = 1.5
    elif kind == "bad_time":
        transaction["TransactionDT"] = transaction["TransactionDT"].astype(float)
        transaction.loc[0, "TransactionDT"] = np.inf
    elif kind == "bad_amount":
        transaction.loc[0, "TransactionAmt"] = -1
    elif kind == "bad_label":
        transaction.loc[0, "isFraud"] = 2
    else:
        transaction = transaction.drop(columns="card1")
    transaction.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    with pytest.raises(ValueError):
        load_ieee(tmp_path)
