"""Configure reproducible local or remote MLflow tracking."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient

DEFAULT_EXPERIMENT_NAME = "talent-attrition-model-development"
DEFAULT_REGISTERED_MODEL_NAME = "talent-attrition-classifier"


@dataclass(frozen=True)
class TrackingContext:
    """Resolved MLflow and Optuna storage locations."""

    tracking_uri: str
    artifact_root: Path
    optuna_storage_uri: str
    experiment_id: str
    git_commit: str


def configure_tracking(
    repository_root: Path,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> TrackingContext:
    """Configure MLflow and create a local SQLite experiment when needed."""
    state_dir = repository_root / ".mlflow"
    state_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = state_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    resolved_tracking_uri = tracking_uri or _sqlite_uri(state_dir / "mlflow.db")

    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_registry_uri(resolved_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    return TrackingContext(
        tracking_uri=resolved_tracking_uri,
        artifact_root=artifact_root,
        optuna_storage_uri=_sqlite_uri(state_dir / "optuna.db"),
        experiment_id=str(experiment_id),
        git_commit=_git_commit(repository_root),
    )


def repository_root_from_config(config_path: Path) -> Path:
    """Return the repository root from .runtime/project_config.json."""
    return config_path.resolve().parent.parent


def _sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _git_commit(repository_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
