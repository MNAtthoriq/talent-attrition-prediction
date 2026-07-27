"""Create compact final evaluation plots for MLflow."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "talent-job-change-intent-matplotlib"),
)

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from talent_job_change_intent_prediction.modeling.evaluation import capacity_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_evaluation_plots(
    output_dir: Path,
    y_true: pd.Series,
    probability: np.ndarray,
    *,
    threshold: float,
) -> tuple[Path, ...]:
    """Write final PR, ROC, confusion-matrix, and capacity plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        _precision_recall_plot(output_dir, y_true, probability),
        _roc_plot(output_dir, y_true, probability),
        _confusion_plot(output_dir, y_true, probability, threshold),
        _capacity_plot(output_dir, y_true, probability),
    )
    plt.close("all")
    return paths


def _precision_recall_plot(
    output_dir: Path,
    y_true: pd.Series,
    probability: np.ndarray,
) -> Path:
    path = output_dir / "precision_recall_curve.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probability,
        name="Final candidate",
        ax=axis,
    )
    axis.axhline(
        float(y_true.mean()),
        color="grey",
        linestyle="--",
        label="Positive-class baseline",
    )
    axis.set_title("Precision–recall curve on untouched test data")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _roc_plot(
    output_dir: Path,
    y_true: pd.Series,
    probability: np.ndarray,
) -> Path:
    path = output_dir / "roc_curve.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        probability,
        name="Final candidate",
        ax=axis,
    )
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--")
    axis.set_title("ROC curve on untouched test data")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _confusion_plot(
    output_dir: Path,
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> Path:
    path = output_dir / "confusion_matrix.png"
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Stay", "Leave"],
    ).plot(ax=axis, colorbar=False)
    axis.set_title(f"Test confusion matrix at threshold {threshold:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _capacity_plot(
    output_dir: Path,
    y_true: pd.Series,
    probability: np.ndarray,
) -> Path:
    path = output_dir / "capacity_curve.png"
    fractions = np.linspace(0.10, 1.0, 19)
    rows = capacity_metrics(
        y_true,
        probability,
        selection_fractions=fractions,
    )
    x = [float(row["selection_fraction"]) for row in rows]
    model_rate = [float(row["model_selected_job_change_intent_rate"]) for row in rows]
    baseline_rate = float(rows[0]["baseline_job_change_intent_rate"])

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(x, model_rate, marker="o", label="Model-selected candidates")
    axis.axhline(
        baseline_rate,
        color="grey",
        linestyle="--",
        label="Random-selection expectation",
    )
    axis.set_xlabel("Fraction of candidates selected")
    axis.set_ylabel("Job-change intent rate among selected candidates")
    axis.set_title("Business selection performance on test data")
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
