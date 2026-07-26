"""Test configuration loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from talent_attrition_prediction.config import Settings
from talent_attrition_prediction.data.pipeline import _load_local_env


def test_settings_loads_terraform_generated_config(
    runtime_config_path: Path,
) -> None:
    settings = Settings.load(runtime_config_path)
    repository_root = runtime_config_path.parent.parent

    assert settings.project_id == "talent-ml-123"
    assert settings.dataset_id == "talent_attrition"
    assert settings.storage_bucket_name == "talent-ml-123-talent-attrition-raw"
    assert settings.raw_gcs_uri == (
        "gs://talent-ml-123-talent-attrition-raw/raw/kaggle/v1/aug_train.csv"
    )
    assert settings.reports_dir == repository_root / "reports/generated"
    assert settings.raw_table_fqn == ("talent-ml-123.talent_attrition.raw_candidates")


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
    template = Path(".env.example").read_text(encoding="utf-8")

    assert "KAGGLE_API_TOKEN=" in template
    assert template.split("KAGGLE_API_TOKEN=", maxsplit=1)[1].strip() == ""
