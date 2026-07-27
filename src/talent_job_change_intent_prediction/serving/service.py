"""Load and call the complete MLflow inference artifact."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from talent_job_change_intent_prediction.modeling.data import MODEL_FEATURES

_ALIASED_MODEL_URI = re.compile(r"^models:/([^/@]+)@([^/]+)$")


class PredictingModel(Protocol):
    """Minimal MLflow pyfunc interface used by the service."""

    def predict(self, data: pd.DataFrame) -> Any:
        """Return model predictions."""


@dataclass(frozen=True)
class ModelDescriptor:
    """Portable lineage exposed by the model-info endpoint."""

    model_uri: str
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    run_id: str | None = None
    git_commit: str | None = None
    source_sha256: str | None = None
    optuna_study: str | None = None
    optuna_trial: int | None = None
    preprocessing: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)


class ModelService:
    """Inference facade that keeps HTTP code independent from MLflow."""

    def __init__(
        self,
        model: PredictingModel,
        descriptor: ModelDescriptor,
    ) -> None:
        self._model = model
        self.descriptor = descriptor

    @classmethod
    def load(
        cls,
        *,
        model_uri: str,
        tracking_uri: str,
        descriptor_path: Path | None = None,
    ) -> ModelService:
        """Load one registered model alias and resolve its metadata."""
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        if descriptor_path is not None:
            return cls(model, _load_exported_descriptor(descriptor_path))

        model_info = mlflow.models.get_model_info(model_uri)
        metadata = dict(model_info.metadata or {})

        descriptor = ModelDescriptor(
            model_uri=model_uri,
            run_id=getattr(model_info, "run_id", None),
            source_sha256=_optional_string(metadata.get("source_sha256")),
            optuna_study=_optional_string(metadata.get("optuna_study")),
            optuna_trial=_optional_integer(metadata.get("optuna_trial")),
        )
        match = _ALIASED_MODEL_URI.fullmatch(model_uri)
        if match:
            model_name, alias = match.groups()
            client = MlflowClient()
            version = client.get_model_version_by_alias(model_name, alias)
            run = client.get_run(version.run_id)
            descriptor = ModelDescriptor(
                model_uri=model_uri,
                model_name=model_name,
                model_version=str(version.version),
                model_alias=alias,
                run_id=version.run_id or descriptor.run_id,
                git_commit=_optional_string(run.data.tags.get("git_commit")),
                source_sha256=descriptor.source_sha256,
                optuna_study=descriptor.optuna_study,
                optuna_trial=descriptor.optuna_trial,
                preprocessing=_optional_string(run.data.tags.get("preprocessing")),
                parameters={
                    str(key): str(value) for key, value in run.data.params.items()
                },
                test_metrics={
                    str(key): float(value)
                    for key, value in run.data.metrics.items()
                    if key.startswith("test_")
                },
            )
        return cls(model, descriptor)

    def predict_probabilities(
        self,
        candidates: list[dict[str, object]],
    ) -> np.ndarray:
        """Return job-change intent probabilities in input order."""
        frame = pd.DataFrame(candidates, columns=MODEL_FEATURES)
        raw_prediction = self._model.predict(frame)
        probability = _positive_probability(raw_prediction, expected_rows=len(frame))
        if not np.isfinite(probability).all():
            raise RuntimeError("Model returned non-finite probabilities.")
        if np.logical_or(probability < 0, probability > 1).any():
            raise RuntimeError("Model returned probabilities outside [0, 1].")
        return probability


def _positive_probability(raw_prediction: Any, *, expected_rows: int) -> np.ndarray:
    """Normalize one- or two-column pyfunc probability output."""
    array = np.asarray(raw_prediction, dtype=float)
    if array.ndim == 1:
        probability = array
    elif array.ndim == 2 and array.shape[1] == 2:
        probability = array[:, 1]
    else:
        raise RuntimeError(
            "Model output must contain one positive-class probability column "
            "or two class-probability columns."
        )
    if len(probability) != expected_rows:
        raise RuntimeError(
            f"Model returned {len(probability)} rows for {expected_rows} candidates."
        )
    return probability


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_exported_descriptor(path: Path) -> ModelDescriptor:
    """Load the lineage sidecar produced by the deployment export."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read model descriptor at {path}.") from error

    if payload.get("schema_version") != 1 or not isinstance(payload.get("model"), dict):
        raise RuntimeError("Unsupported deployment model descriptor.")

    model = payload["model"]
    try:
        return ModelDescriptor(
            model_uri=str(model["model_uri"]),
            model_name=_optional_string(model.get("model_name")),
            model_version=_optional_string(model.get("model_version")),
            model_alias=_optional_string(model.get("model_alias")),
            run_id=_optional_string(model.get("run_id")),
            git_commit=_optional_string(model.get("git_commit")),
            source_sha256=_optional_string(model.get("source_sha256")),
            optuna_study=_optional_string(model.get("optuna_study")),
            optuna_trial=_optional_integer(model.get("optuna_trial")),
            preprocessing=_optional_string(model.get("preprocessing")),
            parameters=_string_mapping(model.get("parameters")),
            test_metrics=_float_mapping(model.get("test_metrics")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Deployment model descriptor is invalid.") from error


def _string_mapping(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("parameters must be an object.")
    return {str(key): str(item) for key, item in value.items()}


def _float_mapping(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("test_metrics must be an object.")
    return {str(key): float(item) for key, item in value.items()}
