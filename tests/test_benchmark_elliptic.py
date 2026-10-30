import numpy as np
import pandas as pd
import pytest

from fraudguard.elliptic import LOCAL_COLUMNS, NEIGHBOR_COLUMNS, STRUCTURE_COLUMNS
from scripts.benchmark_elliptic import MODEL_PARAMS, run_benchmark


@pytest.fixture
def data():
    n = 100
    frame = pd.DataFrame(
        {
            "txId": np.arange(n),
            "time_step": np.repeat(np.arange(1, 21), 5),
            "is_illicit": np.tile([0, 1, 0, 1, -1], 20),
        }
    )
    frame = pd.concat(
        [frame, pd.DataFrame(np.tile(np.arange(n)[:, None], (1, 93)), columns=LOCAL_COLUMNS)],
        axis=1,
    )
    edges = pd.DataFrame({"txId1": np.arange(4, n, 5), "txId2": np.arange(0, n, 5)})
    return frame, edges


def run(frame, edges):
    return run_benchmark(frame, edges, 0.25, model_params={**MODEL_PARAMS, "max_iter": 2})


def test_schema_shared_budget_and_test_labels_do_not_select(data):
    frame, edges = data
    result = run(frame, edges)
    assert result["features"]["local"] == LOCAL_COLUMNS
    assert result["features"]["structure"] == LOCAL_COLUMNS + STRUCTURE_COLUMNS
    assert (
        result["features"]["neighborhood"] == LOCAL_COLUMNS + STRUCTURE_COLUMNS + NEIGHBOR_COLUMNS
    )
    assert {x["review_budget"]["reviewed"] for x in result["test"].values()} == {4}
    changed = frame.copy()
    mask = (changed.time_step >= 17) & (changed.is_illicit != -1)
    changed.loc[mask, "is_illicit"] = 1 - changed.loc[mask, "is_illicit"]
    other = run(changed, edges)
    assert result["selected_variant"] == other["selected_variant"]
    assert [v["threshold"] for v in result["validation"].values()] == [
        v["threshold"] for v in other["validation"].values()
    ]


def test_fit_only_labelled_train_once_and_unknown_nodes_are_context(data, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingClassifier

    from fraudguard.elliptic import build_graph_features

    frame, edges = data
    seen = []
    original = HistGradientBoostingClassifier.fit

    def record(self, X, y, *args, **kwargs):
        seen.append((len(y), set(y), list(X.columns)))
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", record)
    result = run(frame, edges)
    assert len(seen) == 3
    assert all(n == 48 and labels == {0, 1} for n, labels, _ in seen)
    assert result["splits"]["train"]["unknown"] == 12
    assert build_graph_features(frame, edges).loc[0, "in_degree"] == 1
    assert (
        build_graph_features(frame, edges).loc[0, "incoming_mean_local_1"]
        == frame.loc[4, "local_1"]
    )


def test_single_class_time_step(data):
    frame, edges = data
    mask = (frame.time_step == 17) & (frame.is_illicit != -1)
    frame.loc[mask, "is_illicit"] = 0
    result = run(frame, edges)
    for variant in result["test_by_time_step"].values():
        window = variant["17"]
        assert window["average_precision"] is None and window["roc_auc"] is None
        assert window["review_budget"]["reviewed"] == 1


@pytest.mark.parametrize("budget", [0, -1, 1.1, float("nan")])
def test_invalid_budget(data, budget):
    with pytest.raises(ValueError):
        run_benchmark(*data, review_budget=budget)
