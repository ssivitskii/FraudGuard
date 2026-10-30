import numpy as np
import pandas as pd
import pytest

from fraudguard.ieee import BASE_CATEGORICAL
from scripts.benchmark_ieee import MODEL_PARAMS, run_benchmark


@pytest.fixture
def frame():
    n = 90
    data = {
        "TransactionID": range(n),
        "TransactionDT": np.arange(n) * 86400,
        "TransactionAmt": np.arange(n, dtype=float),
        "isFraud": np.arange(n) % 2,
    }
    data.update(
        {col: ["train" if i < 54 else "future" for i in range(n)] for col in BASE_CATEGORICAL}
    )
    return pd.DataFrame(data)


def run(frame, **kwargs):
    return run_benchmark(
        frame,
        delay_days=0,
        review_budget=0.2,
        model_params={**MODEL_PARAMS, "max_iter": 2},
        **kwargs,
    )


def test_ablation_shares_rows_and_budget_and_selection_ignores_test_labels(frame):
    result = run(frame)
    assert {v["review_budget"]["reviewed"] for v in result["test"].values()} == {4}
    assert result["splits"]["test"]["rows"] == 18
    changed = frame.copy()
    changed.loc[72:, "isFraud"] = 1 - changed.loc[72:, "isFraud"]
    other = run(changed)
    assert result["selected_variant"] == other["selected_variant"]
    assert {k: v["threshold"] for k, v in result["validation"].items()} == {
        k: v["threshold"] for k, v in other["validation"].items()
    }


def test_encoder_fitted_on_train_only(frame, monkeypatch):
    from sklearn.preprocessing import OrdinalEncoder

    original = OrdinalEncoder.fit_transform
    seen = []

    def record(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            seen.append(set(X.iloc[:, 0]))
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(OrdinalEncoder, "fit_transform", record)
    run(frame)
    assert seen == [{"train"}] * 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delay_days": -1},
        {"delay_days": float("nan")},
        {"review_budget": 0},
        {"review_budget": 1.1},
    ],
)
def test_invalid_parameters(frame, kwargs):
    with pytest.raises(ValueError):
        run_benchmark(frame, **kwargs)


def test_delay_filters_labels_but_keeps_all_test_rows(frame):
    result = run_benchmark(frame, delay_days=7, model_params={**MODEL_PARAMS, "max_iter": 2})
    assert result["splits"]["train"]["eligible_labels"] == 48
    assert result["splits"]["validation"]["eligible_labels"] == 12
    assert result["splits"]["test"]["eligible_labels"] == 18
