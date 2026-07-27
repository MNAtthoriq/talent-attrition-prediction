"""Evaluate ranking, classification, business selection, and fairness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def evaluate_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
) -> dict[str, Any]:
    """Evaluate one pipeline with leakage-safe stratified cross-validation."""
    if cv_folds < 3:
        raise ValueError("cv_folds must be at least 3.")
    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < cv_folds:
        raise ValueError("Each target class must contain at least cv_folds rows.")

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    oof_probability = np.full(len(y), np.nan, dtype=float)
    folds: list[dict[str, float | int]] = []

    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        fitted = clone(pipeline)
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_validation = X.iloc[validation_index]
        y_validation = y.iloc[validation_index]
        fitted.fit(X_train, y_train)

        train_probability = _positive_probability(fitted, X_train)
        validation_probability = _positive_probability(fitted, X_validation)
        oof_probability[validation_index] = validation_probability
        folds.append(
            {
                "fold": fold_number,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "train_average_precision": float(
                    average_precision_score(y_train, train_probability)
                ),
                "validation_average_precision": float(
                    average_precision_score(y_validation, validation_probability)
                ),
                "validation_roc_auc": float(
                    roc_auc_score(y_validation, validation_probability)
                ),
                "validation_brier_score": float(
                    brier_score_loss(y_validation, validation_probability)
                ),
            }
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError("Cross-validation did not predict every development row.")

    threshold = choose_f2_threshold(y, oof_probability)
    oof_classification = classification_metrics(y, oof_probability, threshold)
    validation_ap = np.array(
        [fold["validation_average_precision"] for fold in folds],
        dtype=float,
    )
    train_ap = np.array(
        [fold["train_average_precision"] for fold in folds],
        dtype=float,
    )
    validation_roc = np.array(
        [fold["validation_roc_auc"] for fold in folds],
        dtype=float,
    )
    validation_brier = np.array(
        [fold["validation_brier_score"] for fold in folds],
        dtype=float,
    )

    return {
        "cv_folds": cv_folds,
        "development_rows": len(y),
        "positive_rate": float(y.mean()),
        "validation_average_precision_mean": float(validation_ap.mean()),
        "validation_average_precision_std": float(validation_ap.std(ddof=1)),
        "train_average_precision_mean": float(train_ap.mean()),
        "average_precision_gap": float(train_ap.mean() - validation_ap.mean()),
        "validation_roc_auc_mean": float(validation_roc.mean()),
        "validation_roc_auc_std": float(validation_roc.std(ddof=1)),
        "validation_brier_score_mean": float(validation_brier.mean()),
        "oof_threshold": threshold,
        **{f"oof_{key}": value for key, value in oof_classification.items()},
        "folds": folds,
    }


def evaluate_holdout(
    y_true: pd.Series,
    probability: np.ndarray,
    *,
    threshold: float,
    selection_fractions: Sequence[float] = (0.25, 0.50, 0.75),
) -> dict[str, Any]:
    """Evaluate the untouched test set once using a development-set threshold."""
    classification = classification_metrics(y_true, probability, threshold)
    return {
        "rows": len(y_true),
        "positive_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        **classification,
        "capacity_metrics": capacity_metrics(
            y_true,
            probability,
            selection_fractions=selection_fractions,
        ),
    }


def classification_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate threshold-dependent metrics."""
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
    }


def choose_f2_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
) -> float:
    """Choose a recall-weighted threshold using development predictions only."""
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probability,
    )
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = (4 * precision) + recall
    scores = np.divide(
        5 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(scores))])


def capacity_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    *,
    selection_fractions: Sequence[float],
) -> list[dict[str, float | int]]:
    """Compare low-risk model selection with same-size random expectation."""
    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    if len(y_array) != len(probability_array):
        raise ValueError("y_true and probability must have the same length.")
    baseline_rate = float(y_array.mean())
    ranking = np.argsort(probability_array, kind="stable")
    results = []
    for fraction in selection_fractions:
        if not 0 < fraction <= 1:
            raise ValueError(
                "selection fractions must be greater than 0 and at most 1."
            )
        selected_rows = max(1, round(len(y_array) * fraction))
        selected_rate = float(y_array[ranking[:selected_rows]].mean())
        absolute_reduction = baseline_rate - selected_rate
        relative_reduction = (
            absolute_reduction / baseline_rate if baseline_rate else 0.0
        )
        results.append(
            {
                "selection_fraction": float(fraction),
                "selected_rows": selected_rows,
                "baseline_job_change_intent_rate": baseline_rate,
                "model_selected_job_change_intent_rate": selected_rate,
                "absolute_job_change_intent_reduction": absolute_reduction,
                "relative_job_change_intent_reduction": relative_reduction,
            }
        )
    return results


def fairness_by_group(
    audit: pd.DataFrame,
    y_true: pd.Series,
    probability: np.ndarray,
    *,
    group_column: str,
    threshold: float,
    minimum_group_rows: int = 30,
) -> dict[str, Any]:
    """Create a simple final holdout audit without adding groups to the model."""
    groups = audit[group_column].fillna("__MISSING__").astype(str)
    prediction = (np.asarray(probability) >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for group in sorted(groups.unique()):
        mask = groups.eq(group).to_numpy()
        row_count = int(mask.sum())
        if row_count < minimum_group_rows:
            continue
        group_target = y_true.to_numpy()[mask]
        group_probability = np.asarray(probability)[mask]
        group_prediction = prediction[mask]
        row: dict[str, Any] = {
            "group": group,
            "rows": row_count,
            "positive_rate": float(group_target.mean()),
            "predicted_positive_rate": float(group_prediction.mean()),
            "recall": float(
                recall_score(
                    group_target,
                    group_prediction,
                    zero_division=0,
                )
            ),
        }
        if len(np.unique(group_target)) == 2:
            row["average_precision"] = float(
                average_precision_score(group_target, group_probability)
            )
        rows.append(row)
    recalls = [row["recall"] for row in rows]
    predicted_rates = [row["predicted_positive_rate"] for row in rows]
    return {
        "group_column": group_column,
        "minimum_group_rows": minimum_group_rows,
        "groups": rows,
        "max_recall_gap": float(max(recalls) - min(recalls)) if recalls else None,
        "max_predicted_positive_rate_gap": (
            float(max(predicted_rates) - min(predicted_rates))
            if predicted_rates
            else None
        ),
    }


def flat_mlflow_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Return only scalar numeric fields accepted by MLflow metrics."""
    return {
        key: float(value)
        for key, value in result.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    }


def _positive_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(pipeline.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Classifier must return two-column predict_proba output.")
    return probabilities[:, 1]
