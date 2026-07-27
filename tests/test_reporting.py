"""Test the review-safe result bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from talent_job_change_intent_prediction.modeling.reporting import (
    export_results,
    write_json,
)


def test_write_and_export_results(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    write_json(
        reports_dir / "model_selection_summary.json",
        {"test_set_used": False},
    )
    output = export_results(reports_dir, tmp_path / "results.zip")

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        contents = json.loads(archive.read("CONTENTS.json"))

    assert names == {"model_selection_summary.json", "CONTENTS.json"}
    assert contents["included"] == ["model_selection_summary.json"]


def test_export_requires_at_least_one_result(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No generated"):
        export_results(tmp_path / "empty", tmp_path / "results.zip")
