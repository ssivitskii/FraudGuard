import numpy as np
import pandas as pd
import pytest

from fraudguard.handbook import BASE_COLUMNS, HISTORY_COLUMNS, build_history_features
from scripts import replay_handbook as replay


def transactions():
    dates = pd.date_range("2018-04-01", "2018-07-15", freq="D")
    times = [date + pd.Timedelta(hours=hour) for date in dates for hour in (0, 12)]
    n = len(times)
    return pd.DataFrame(
        {
            "TRANSACTION_ID": range(n),
            "TX_DATETIME": times,
            "CUSTOMER_ID": [1] * n,
            "TERMINAL_ID": [1] * n,
            "TX_AMOUNT": np.arange(n) + 1,
            "TX_FRAUD": np.arange(n) % 2,
            "TX_FRAUD_SCENARIO": [3] * n,
        }
    )


class RecordingClassifier:
    calls = []

    def __init__(self, **kwargs):
        pass

    def fit(self, x, y):
        self.calls.append((x.copy(), y.copy()))
        return self

    def predict_proba(self, x):
        self.calls.append(x.copy())
        scores = np.full(len(x), 0.5)
        return np.column_stack((1 - scores, scores))


def test_replay_refits_cohorts_features_and_budget(monkeypatch):
    monkeypatch.setattr(replay, "HistGradientBoostingClassifier", RecordingClassifier)
    RecordingClassifier.calls = []
    frame = transactions()
    report = replay.run_replay(frame)
    assert report["status"] == "completed"
    assert report["input_summary"] == {
        "rows": len(frame),
        "frauds": int(frame.TX_FRAUD.sum()),
        "time_min": frame.TX_DATETIME.min().isoformat(),
        "time_max": frame.TX_DATETIME.max().isoformat(),
    }
    for delay in (0, 7):
        for variant, columns in (("basic", BASE_COLUMNS), ("history", HISTORY_COLUMNS)):
            result = report["results"][str(delay)][variant]
            assert len(result["refits"]) == 4
            assert len(result["daily"]) == 28
            assert result["aggregate"]["rows"] == 56
            assert result["aggregate"]["reviewed"] == 28
            # Stable scores select the first event of each day, which is non-fraud.
            assert result["aggregate"]["frauds_found"] == 0
            assert result["aggregate"]["valid_ap_days"] == 28
            for refit in result["refits"]:
                fit = pd.Timestamp(refit["fit_time"])
                assert pd.Timestamp(refit["latest_label_available_at"]) < fit
                assert pd.Timestamp(refit["train_end"]) == fit - pd.Timedelta(days=delay)
                assert refit["rows"] == 56
            assert report["protocol"]["features"][variant] == columns
    # Calls are delay, variant, then alternating fit and predict at each refit.
    for delay_index, delay in enumerate((0, 7)):
        expected = build_history_features(frame, delay)
        for variant_index, columns in enumerate((BASE_COLUMNS, HISTORY_COLUMNS)):
            base = (delay_index * 2 + variant_index) * 8
            x_train, y_train = RecordingClassifier.calls[base]
            train_end = pd.Timestamp("2018-06-18") - pd.Timedelta(days=delay)
            mask = (train_end - pd.Timedelta(days=28) <= frame.TX_DATETIME) & (
                train_end > frame.TX_DATETIME
            )
            pd.testing.assert_frame_equal(x_train, expected.loc[mask, columns])
            pd.testing.assert_series_equal(y_train, frame.loc[mask, "TX_FRAUD"].astype("int8"))
            x_score = RecordingClassifier.calls[base + 1]
            mask = (frame.TX_DATETIME >= "2018-06-18") & (frame.TX_DATETIME < "2018-06-25")
            pd.testing.assert_frame_equal(x_score, expected.loc[mask, columns])


def test_unavailable_labels_do_not_change_first_fit_or_first_scores(monkeypatch):
    monkeypatch.setattr(replay, "HistGradientBoostingClassifier", RecordingClassifier)
    frame = transactions()
    RecordingClassifier.calls = []
    replay.run_replay(frame, label_delays=(7,))
    baseline = [(RecordingClassifier.calls[i], RecordingClassifier.calls[i + 1]) for i in (0, 8)]
    # Labels of June 18 onward are unavailable at the first fit and throughout
    # the first scored week. Later fits can legitimately see changed labels.
    changed = frame.copy()
    mask = changed.TX_DATETIME >= "2018-06-18"
    changed.loc[mask, "TX_FRAUD"] = 1 - changed.loc[mask, "TX_FRAUD"]
    RecordingClassifier.calls = []
    replay.run_replay(changed, label_delays=(7,))
    for i, (baseline_fit, baseline_scores) in zip((0, 8), baseline, strict=True):
        pd.testing.assert_frame_equal(RecordingClassifier.calls[i][0], baseline_fit[0])
        pd.testing.assert_series_equal(RecordingClassifier.calls[i][1], baseline_fit[1])
        pd.testing.assert_frame_equal(RecordingClassifier.calls[i + 1], baseline_scores)


def test_invalid_coverage_and_parameters(monkeypatch):
    monkeypatch.setattr(replay, "HistGradientBoostingClassifier", RecordingClassifier)
    frame = transactions()
    with pytest.raises(ValueError, match="coverage"):
        replay.run_replay(frame[frame.TX_DATETIME >= "2018-04-08"])
    # April 14 covers 30 days before the earliest training event, but omits
    # the extra seven days needed for delayed feedback in those features.
    truncated = frame[frame.TX_DATETIME >= "2018-04-14"]
    replay.run_replay(truncated, label_delays=(0,))
    with pytest.raises(ValueError, match="coverage"):
        replay.run_replay(truncated, label_delays=(7,))
    with pytest.raises(ValueError, match="coverage"):
        replay.run_replay(frame[frame.TX_DATETIME.dt.normalize() != pd.Timestamp("2018-06-19")])
    for kwargs in (
        {"train_days": 0},
        {"evaluation_days": True},
        {"refit_days": -1},
        {"label_delays": (0, -1)},
        {"label_delays": (7, 7)},
        {"start": "2018-06-18 12:00"},
    ):
        with pytest.raises(ValueError):
            replay.run_replay(frame, **kwargs)
