from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    confusion_matrix: np.ndarray
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    classification_report: str

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Model Evaluation Results",
            "=" * 50,
            "",
            "Confusion Matrix:",
            str(self.confusion_matrix),
            "",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1-Score:  {self.f1:.4f}",
        ]

        if self.roc_auc is not None:
            lines.append(f"ROC-AUC:   {self.roc_auc:.4f}")
        if self.pr_auc is not None:
            lines.append(f"PR-AUC:    {self.pr_auc:.4f}")

        lines.extend(["", "Classification Report:", self.classification_report])

        return "\n".join(lines)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    threshold: float = 0.5,
) -> EvaluationResult:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    y_true = np.asarray(y_true)
    if (
        y_true.ndim != 1
        or len(y_true) != len(X)
        or not len(y_true)
        or not np.isin(y_true, [0, 1]).all()
    ):
        raise ValueError("Expected nonempty aligned binary targets")
    y_proba = _get_probabilities(model, X)
    if y_proba is not None and (y_proba.shape != y_true.shape or not np.isfinite(y_proba).all()):
        raise ValueError("Model scores must be finite and aligned with targets")
    y_pred = (y_proba >= threshold).astype(int) if y_proba is not None else model.predict(X)
    y_pred = np.asarray(y_pred)
    if y_pred.shape != y_true.shape or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Model predictions must be aligned binary labels")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    pr_auc = None

    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
        except ValueError as e:
            logger.warning("Could not compute AUC metrics: %s", e)

    report = classification_report(y_true, y_pred, labels=[0, 1], digits=4, zero_division=0)

    result = EvaluationResult(
        confusion_matrix=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        classification_report=report,
    )

    logger.info("Evaluation complete - F1: %.4f, ROC-AUC: %s", f1, roc_auc)
    return result


def _get_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Expected two-class predict_proba output")
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        positive_columns = np.flatnonzero(classes == 1)
        if classes.shape != (2,) or len(positive_columns) != 1:
            raise ValueError("Expected positive class label 1")
        scores = probabilities[:, positive_columns[0]]
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("predict_proba returned values outside [0, 1]")
        return scores

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("Expected finite binary one-dimensional decision scores")
        # Fixed monotonic link: no min/max is learned from the evaluation cohort.
        # These scores are not calibrated probabilities. Threshold .5 is margin 0.
        return expit(scores)

    logger.warning("Model has no predict_proba or decision_function")
    return None
