import numpy as np
import pandas as pd
import pytest

from fraudguard.data import chronological_split
from fraudguard.evaluate import _get_probabilities, evaluate_model
from scripts.benchmark import budget_metrics, validation_threshold


def test_equal_timestamps_stay_together():
    frame = pd.DataFrame({"t": np.repeat(np.arange(10), 3), "y": [0, 1] * 15})
    train, valid, test = chronological_split(frame.sample(frac=1, random_state=3), "t", "y")
    assert train.t.max() < valid.t.min() <= valid.t.max() < test.t.min()
    assert len(train) + len(valid) + len(test) == len(frame)


def test_budget_is_exact_with_ties():
    result = budget_metrics([0, 1, 1, 0], [1, 1, 1, 1], 0.5)
    assert result == {"reviewed": 2, "frauds_found": 1, "precision": 0.5, "recall": 0.5}


def test_threshold_is_selected_from_validation_scores():
    assert validation_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 0.3


def test_evaluate_respects_threshold():
    class Model:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6], [0.2, 0.8]])

    result = evaluate_model(Model(), pd.DataFrame({"x": [0, 1]}), pd.Series([0, 1]), threshold=0.7)
    assert result.f1 == 1
    with pytest.raises(ValueError):
        evaluate_model(Model(), pd.DataFrame(), pd.Series(), threshold=2)


@pytest.mark.parametrize("times", [[0, 1, np.inf, 3, 4], [0, 1, np.nan, 3, 4]])
def test_nonfinite_timestamps_are_refused(times):
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"t": times, "y": [0, 1, 0, 1, 0]}), "t", "y")


def test_string_times_must_be_parsed_before_chronological_split():
    with pytest.raises(ValueError, match="parse string"):
        chronological_split(
            pd.DataFrame({"t": ["1", "10", "2", "3", "4"], "y": [0, 1, 0, 1, 0]}), "t", "y"
        )


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([], []),
        ([0, 1], [0.2]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [0.1, np.inf]),
        ([0, 1], [0.1, np.nan]),
    ],
)
def test_invalid_budget_inputs_are_rejected(labels, scores):
    with pytest.raises(ValueError):
        budget_metrics(labels, scores)


def test_no_positive_labels_has_no_recall_claim():
    assert budget_metrics([0, 0], [0.2, 0.1], 0.5)["recall"] is None
    with pytest.raises(ValueError, match="both classes"):
        validation_threshold([0, 0], [0.2, 0.1])


def test_margin_link_does_not_depend_on_evaluation_cohort():
    class Model:
        def decision_function(self, X):
            return X["margin"].to_numpy()

    single = _get_probabilities(Model(), pd.DataFrame({"margin": [1.0]}))
    mixed = _get_probabilities(Model(), pd.DataFrame({"margin": [-100.0, 1.0, 100.0]}))
    assert single[0] == mixed[1]
    assert single[0] > 0.5


def test_probability_column_follows_positive_label():
    class Model:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    result = evaluate_model(Model(), pd.DataFrame({"x": [0, 1]}), pd.Series([1, 0]))
    assert result.f1 == 1


def test_single_class_truth_keeps_auc_undefined():
    class Model:
        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.8, 0.2]])

    result = evaluate_model(Model(), pd.DataFrame({"x": [0, 1]}), pd.Series([0, 0]))
    assert result.roc_auc is None and result.pr_auc is None
    assert result.confusion_matrix.shape == (2, 2)
