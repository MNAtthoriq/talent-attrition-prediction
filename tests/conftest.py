"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talent_attrition_prediction.config import Settings


@pytest.fixture
def runtime_config_path(tmp_path: Path) -> Path:
    """Return a path to a runtime configuration file."""
    repository_root = tmp_path / "project"
    config_path = repository_root / ".runtime" / "project_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "project_id": "talent-ml-123",
                "location": "asia-southeast2",
                "storage_bucket_name": "talent-ml-123-talent-attrition-raw",
                "dataset_id": "talent_attrition",
                "raw_table_id": "raw_candidates",
                "modeling_table_id": "modeling_candidates",
                "kaggle_dataset_handle": (
                    "arashnic/hr-analytics-job-change-of-data-scientists/versions/1"
                ),
                "kaggle_train_file": "aug_train.csv",
                "raw_object_name": "raw/kaggle/v1/aug_train.csv",
                "expected_row_count": 19158,
                "expected_size_bytes": 1961145,
                "expected_sha256": "a" * 64,
                "reports_dir": "reports/generated",
                "sql_dir": "sql",
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def settings(runtime_config_path: Path) -> Settings:
    """Return runtime configuration."""
    return Settings.load(runtime_config_path)
