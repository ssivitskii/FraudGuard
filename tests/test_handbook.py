import numpy as np
import pandas as pd
import pytest

from fraudguard.handbook import (
    BASE_COLUMNS,
    HISTORY_COLUMNS,
    build_history_features,
    load_handbook,
    validate_transactions,
)


def transactions():
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [0, 1, 2, 3, 4, 5],
            "TX_DATETIME": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-02", "2018-01-08", "2018-01-09", "2018-02-01"]
            ),
            "CUSTOMER_ID": [1, 1, 1, 1, 2, 1],
            "TERMINAL_ID": [2, 2, 2, 2, 2, 3],
            "TX_AMOUNT": [10, 20, 40, 80, 100, 50],
            "TX_FRAUD": [1, 0, 1, 0, 1, 0],
        }
    )


def test_window_boundaries_ties_and_feedback():
    features = build_history_features(transactions(), label_delay_days=7)
    assert len(HISTORY_COLUMNS) == 18
    assert list(features) == HISTORY_COLUMNS
    assert HISTORY_COLUMNS[:3] == BASE_COLUMNS
    # The start is inclusive, and both transactions at the scoring time are excluded.
    assert features.loc[1, "customer_count_1d"] == 1
    assert features.loc[2, "customer_mean_amount_1d"] == 10
    # Jan 1 feedback arrives exactly at Jan 8 and is not visible yet.
    assert features.loc[3, "terminal_feedback_count_30d"] == 0
    # Jan 1 feedback is visible on Jan 9; Jan 2 feedback arrives exactly then.
    assert features.loc[4, "terminal_feedback_count_1d"] == 1
    assert features.loc[4, "terminal_fraud_rate_1d"] == 1
    assert features.loc[3, "customer_count_7d"] == 3
    assert features.loc[3, "customer_mean_amount_7d"] == pytest.approx(70 / 3)
    assert features.loc[5, "customer_count_30d"] == 3
    assert features.loc[5, "terminal_count_30d"] == 0
    assert features.loc[5, "terminal_fraud_rate_30d"] == 0


def test_zero_delay_and_self_exclusion():
    features = build_history_features(transactions(), label_delay_days=0)
    assert features.loc[0, "terminal_feedback_count_30d"] == 0
    assert features.loc[1, "terminal_feedback_count_1d"] == 1
    assert features.loc[2, "terminal_feedback_count_1d"] == 1
    assert features.loc[3, "terminal_feedback_count_7d"] == 3
    assert features.loc[3, "terminal_fraud_rate_7d"] == pytest.approx(2 / 3)


def test_permutation_future_and_unavailable_labels():
    frame = transactions()
    baseline = build_history_features(frame)
    shuffled = frame.sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(build_history_features(shuffled).sort_index(), baseline)
    future = frame.iloc[[0]].copy()
    future["TRANSACTION_ID"] = 6
    future["TX_DATETIME"] = pd.Timestamp("2018-03-01")
    extended = pd.concat([frame, future], ignore_index=True)
    pd.testing.assert_frame_equal(build_history_features(extended).iloc[:6], baseline)
    changed = frame.copy()
    changed.loc[1:2, "TX_FRAUD"] = 1 - changed.loc[1:2, "TX_FRAUD"]
    pd.testing.assert_frame_equal(build_history_features(changed).iloc[:5], baseline.iloc[:5])


def test_validate_and_load(tmp_path):
    frame = transactions().iloc[::-1]
    frame["CUSTOMER_ID"] = frame["CUSTOMER_ID"].astype(str)
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)
    loaded = load_handbook(path)
    assert loaded["TRANSACTION_ID"].tolist() == list(range(6))
    assert pd.api.types.is_integer_dtype(loaded["CUSTOMER_ID"])
    features = build_history_features(frame)
    assert features.index.equals(frame.index)


@pytest.mark.parametrize(
    "column,value",
    [
        ("TRANSACTION_ID", -1),
        ("TRANSACTION_ID", 1),
        ("CUSTOMER_ID", 1.5),
        ("TERMINAL_ID", np.inf),
        ("TX_AMOUNT", -1),
        ("TX_AMOUNT", np.nan),
        ("TX_DATETIME", None),
        ("TX_DATETIME", "2018-01-01T00:00:00Z"),
        ("TX_FRAUD", 2),
    ],
)
def test_invalid_data(column, value):
    frame = transactions().astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_transactions(frame)


@pytest.mark.parametrize("delay", [-1, 1.5, True, "7"])
def test_invalid_delay(delay):
    with pytest.raises(ValueError):
        build_history_features(transactions(), label_delay_days=delay)


def test_missing_columns():
    with pytest.raises(ValueError, match="Missing required"):
        validate_transactions(transactions().drop(columns="TX_FRAUD"))
