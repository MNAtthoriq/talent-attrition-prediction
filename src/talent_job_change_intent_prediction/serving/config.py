"""Read model-serving configuration without requiring Google Cloud settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from talent_job_change_intent_prediction.modeling.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
)


@dataclass(frozen=True)
class ServingSettings:
    """Runtime configuration for the prediction API."""

    model_uri: str
    tracking_uri: str
    model_descriptor_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if not self.model_uri.strip():
            raise ValueError("model_uri cannot be blank.")
        if not self.tracking_uri.strip():
            raise ValueError("tracking_uri cannot be blank.")
        if not self.host.strip():
            raise ValueError("host cannot be blank.")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535.")
        if (
            self.model_descriptor_path is not None
            and not self.model_descriptor_path.is_file()
        ):
            raise ValueError("model_descriptor_path must identify an existing file.")

    @classmethod
    def load(cls) -> ServingSettings:
        """Load local .env values without overriding the runtime environment."""
        repository_root = _find_repository_root()
        load_dotenv(repository_root / ".env", override=False)
        default_tracking_uri = (
            f"sqlite:///{(repository_root / '.mlflow' / 'mlflow.db').resolve()}"
        )
        return cls(
            model_uri=os.getenv(
                "TALENT_MODEL_URI",
                f"models:/{DEFAULT_REGISTERED_MODEL_NAME}@candidate",
            ),
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                default_tracking_uri,
            ),
            model_descriptor_path=_optional_path(os.getenv("TALENT_MODEL_DESCRIPTOR")),
            host=os.getenv("TALENT_API_HOST", "127.0.0.1"),
            port=_environment_port(os.getenv("PORT", "8000")),
        )


def _find_repository_root() -> Path:
    """Find the project root for the default local MLflow database."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return Path.cwd()


def _environment_port(raw_value: str) -> int:
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError("PORT must be an integer.") from error


def _optional_path(raw_value: str | None) -> Path | None:
    if raw_value is None or not raw_value.strip():
        return None
    return Path(raw_value).expanduser().resolve()
