"""Test Optuna study isolation and reviewed-winner selection."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest

from talent_job_change_intent_prediction.modeling.training import (
    _selected_study_name,
    _set_or_validate_study_contract,
    _study_name,
    _tuning_study_name,
)


def test_study_name_isolates_cross_validation_contracts() -> None:
    source_sha256 = "a" * 64

    assert _study_name(source_sha256, 3) != _study_name(source_sha256, 5)


def test_tuning_resumes_compatible_legacy_study(tmp_path: Path) -> None:
    summary_path = tmp_path / "model_selection_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_sha256": "a" * 64,
                "study_name": "legacy-study-name",
                "dummy_baseline": {"cv_folds": 5},
            }
        ),
        encoding="utf-8",
    )

    assert (
        _tuning_study_name(
            summary_path,
            source_sha256="a" * 64,
            cv_folds=5,
        )
        == "legacy-study-name"
    )
    assert (
        _tuning_study_name(
            summary_path,
            source_sha256="a" * 64,
            cv_folds=3,
        )
        != "legacy-study-name"
    )


def test_study_contract_rejects_incomparable_resume() -> None:
    study = optuna.create_study(direction="maximize")
    _set_or_validate_study_contract(
        study,
        source_sha256="a" * 64,
        cv_folds=5,
        random_state=42,
    )

    with pytest.raises(RuntimeError, match="cv_folds"):
        _set_or_validate_study_contract(
            study,
            source_sha256="a" * 64,
            cv_folds=3,
            random_state=42,
        )


def test_finalize_uses_exact_study_from_reviewed_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "model_selection_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_sha256": "a" * 64,
                "study_name": "legacy-study-name",
            }
        ),
        encoding="utf-8",
    )

    assert (
        _selected_study_name(summary_path, source_sha256="a" * 64)
        == "legacy-study-name"
    )


def test_finalize_rejects_summary_for_different_data(tmp_path: Path) -> None:
    summary_path = tmp_path / "model_selection_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_sha256": "a" * 64,
                "study_name": "study-name",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different source dataset"):
        _selected_study_name(summary_path, source_sha256="b" * 64)
