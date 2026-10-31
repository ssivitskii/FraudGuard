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


def test_review_policies_ties_caps_threshold_and_empty_steps():
    from scripts.benchmark_elliptic import _review_policies

    test = pd.DataFrame({"time_step": [17, 17, 17, 18, 18, 19], "is_illicit": [0, 1, 1, 1, 0, -1]})
    labelled = test[test.is_illicit != -1]
    policies = _review_policies(test, labelled, np.array([0.9, 0.9, 0.2, 0.8, 0.1]), 0.85)
    step = policies["per_step"]["17"]
    assert step["fractions"][0] == {
        "fraction": 0.005,
        "reviewed": 1,
        "frauds_found": 0,
        "precision": 0.0,
        "recall": 0.0,
    }
    assert step["capacities"][0] == {
        "capacity": 5,
        "reviewed": 3,
        "frauds_found": 2,
        "precision": 2 / 3,
        "recall": 1.0,
    }
    assert step["frozen_threshold"]["reviewed"] == 2
    assert step["frozen_threshold"]["frauds_found"] == 1
    empty = policies["per_step"]["19"]
    assert empty["fractions"][0]["reviewed"] == 0
    assert empty["frozen_threshold"]["precision"] is None
    assert empty["capacities"][0]["recall"] is None
    assert policies["pooled_fractions"][1]["reviewed"] == 1
    assert policies["summed_per_step"]["fractions"][1]["reviewed"] == 2
    assert policies["summed_per_step"]["frozen_threshold"]["frauds_found"] == 1

    long_queue = pd.DataFrame({"time_step": [20] * 8, "is_illicit": [0, 1, 0, 1, 1, 0, 1, 1]})
    tied = _review_policies(long_queue, long_queue, np.ones(8), 0.5)
    assert tied["per_step"]["20"]["capacities"][0] == {
        "capacity": 5,
        "reviewed": 5,
        "frauds_found": 3,
        "precision": 3 / 5,
        "recall": 3 / 5,
    }


def test_single_class_and_unknown_counts(data):
    frame, edges = data
    frame.loc[frame.time_step == 17, "is_illicit"] = 0
    frame.loc[frame.time_step == 18, "is_illicit"] = -1
    result = run_benchmark(frame, edges, 0.01, model_params={**MODEL_PARAMS, "max_iter": 2})
    assert result["time_step_counts"]["17"]["prevalence"] == 0
    assert result["time_step_counts"]["18"] == {
        "rows": 5,
        "labelled": 0,
        "unknown": 5,
        "illicit": 0,
        "licit": 0,
        "prevalence": None,
    }
    for name, policy in result["test_review_policies"].items():
        assert policy["per_step"]["17"]["capacities"][0]["recall"] is None
        assert policy["per_step"]["18"]["capacities"][0]["reviewed"] == 0
        assert policy["pooled_fractions"][1] == {
            "fraction": 0.01,
            **result["test"][name]["review_budget"],
        }


def test_historical_folds_are_independent_of_future_labels(data, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingClassifier

    frame, _ = data
    frame = pd.concat(
        [
            frame.assign(time_step=frame.time_step + 20 * i, txId=frame.txId + 100 * i)
            for i in range(3)
        ],
        ignore_index=True,
    )
    frame = frame.loc[frame.time_step <= 49].sample(frac=1, random_state=1)
    edges = pd.DataFrame({"txId1": [], "txId2": []})
    fits = []
    original = HistGradientBoostingClassifier.fit

    def record(self, X, y, *args, **kwargs):
        fits.append((tuple(X.index), tuple(y)))
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", record)
    kwargs = {"model_params": {**MODEL_PARAMS, "max_iter": 2}, "historical_windows": True}
    result = run_benchmark(frame, edges, **kwargs)
    original_fits = fits.copy()
    ordered = frame.sort_values(["time_step", "txId"], kind="stable").reset_index(drop=True)
    for fold_index, train_end in enumerate((19, 24, 29)):
        expected = ordered.index[(ordered.time_step <= train_end) & (ordered.is_illicit != -1)]
        for indices, labels in original_fits[3 + fold_index * 3 : 6 + fold_index * 3]:
            assert indices == tuple(expected)
            assert labels == tuple(ordered.loc[expected, "is_illicit"])
    fits.clear()
    changed = frame.copy()
    mask = (changed.time_step >= 40) & (changed.is_illicit != -1)
    changed.loc[mask, "is_illicit"] = 1 - changed.loc[mask, "is_illicit"]
    other = run_benchmark(changed, edges, **kwargs)
    assert original_fits[3:] == fits[3:]
    assert result["historical_windows"] == other["historical_windows"]
    changed_eval = frame.copy()
    mask = changed_eval.time_step.between(25, 29) & (changed_eval.is_illicit != -1)
    changed_eval.loc[mask, "is_illicit"] = 1 - changed_eval.loc[mask, "is_illicit"]
    other_eval = run_benchmark(changed_eval, edges, **kwargs)
    first = result["historical_windows"][0]
    first_changed = other_eval["historical_windows"][0]
    assert first["validation_metrics"] == first_changed["validation_metrics"]
    assert first["selected_variant"] == first_changed["selected_variant"]
    for fold, bounds in zip(
        result["historical_windows"],
        [(19, 20, 24, 25, 29), (24, 25, 29, 30, 34), (29, 30, 34, 35, 39)],
        strict=True,
    ):
        split = fold["splits"]
        assert (
            split["train"]["time_max"],
            split["validation"]["time_min"],
            split["validation"]["time_max"],
            split["evaluation"]["time_min"],
            split["evaluation"]["time_max"],
        ) == bounds
