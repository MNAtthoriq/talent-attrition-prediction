"""Test configuration loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from talent_job_change_intent_prediction.config import Settings
from talent_job_change_intent_prediction.data.pipeline import _load_local_env
from talent_job_change_intent_prediction.serving.config import ServingSettings


def test_settings_loads_terraform_generated_config(
    runtime_config_path: Path,
) -> None:
    settings = Settings.load(runtime_config_path)
    repository_root = runtime_config_path.parent.parent

    assert settings.project_id == "talent-ml-123"
    assert settings.dataset_id == "talent_job_change_intent"
    assert settings.storage_bucket_name == "talent-ml-123-talent-job-change-intent-raw"
    assert settings.raw_gcs_uri == (
        "gs://talent-ml-123-talent-job-change-intent-raw/raw/kaggle/v1/aug_train.csv"
    )
    assert settings.reports_dir == repository_root / "reports/generated"
    assert settings.raw_table_fqn == (
        "talent-ml-123.talent_job_change_intent.raw_candidates"
    )


def test_invalid_bigquery_identifier_is_rejected(
    runtime_config_path: Path,
) -> None:
    payload = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    payload["dataset_id"] = "invalid-name"
    runtime_config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_id"):
        Settings.load(runtime_config_path)


def test_project_paths_cannot_escape_repository(
    runtime_config_path: Path,
) -> None:
    payload = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    payload["reports_dir"] = "../outside"
    runtime_config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repository-relative"):
        Settings.load(runtime_config_path)


def test_unsafe_gcs_object_path_is_rejected(runtime_config_path: Path) -> None:
    payload = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    payload["raw_object_name"] = "../outside.csv"
    runtime_config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="raw_object_name"):
        Settings.load(runtime_config_path)


def test_missing_config_explains_terraform_requirement(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="terraform apply"):
        Settings.load(tmp_path / "missing.json")


def test_local_env_loads_kaggle_token_without_shell_export(
    runtime_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(runtime_config_path)
    env_path = runtime_config_path.parent.parent / ".env"
    env_path.write_text("KAGGLE_API_TOKEN=local-test-token\n", encoding="utf-8")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    _load_local_env(settings)

    assert os.environ["KAGGLE_API_TOKEN"] == "local-test-token"


def test_terminal_kaggle_token_takes_precedence_over_local_env(
    runtime_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(runtime_config_path)
    env_path = runtime_config_path.parent.parent / ".env"
    env_path.write_text("KAGGLE_API_TOKEN=local-test-token\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "terminal-or-ci-token")

    _load_local_env(settings)

    assert os.environ["KAGGLE_API_TOKEN"] == "terminal-or-ci-token"


def test_env_template_contains_no_token() -> None:
    lines = Path(".env.example").read_text(encoding="utf-8").splitlines()
    token_lines = [line for line in lines if line.startswith("KAGGLE_API_TOKEN=")]

    assert token_lines == ["KAGGLE_API_TOKEN="]


def test_serving_settings_load_local_env_without_overriding_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "TALENT_MODEL_URI=models:/local-model@candidate\n"
        "MLFLOW_TRACKING_URI=sqlite:////tmp/local-mlflow.db\n"
        "TALENT_API_HOST=0.0.0.0\n"
        "PORT=9000\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for variable in (
        "TALENT_MODEL_URI",
        "MLFLOW_TRACKING_URI",
        "TALENT_MODEL_DESCRIPTOR",
        "TALENT_API_HOST",
        "PORT",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("PORT", "8080")

    settings = ServingSettings.load()

    assert settings.model_uri == "models:/local-model@candidate"
    assert settings.tracking_uri == "sqlite:////tmp/local-mlflow.db"
    assert settings.model_descriptor_path is None
    assert settings.host == "0.0.0.0"
    assert settings.port == 8080


def test_serving_settings_requires_existing_model_descriptor(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="model_descriptor_path"):
        ServingSettings(
            model_uri="models:/local-model@candidate",
            tracking_uri="sqlite:////tmp/local-mlflow.db",
            model_descriptor_path=tmp_path / "missing.json",
        )
