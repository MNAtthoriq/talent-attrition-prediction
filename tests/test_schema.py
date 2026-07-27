"""Test schema validation."""

import csv
from pathlib import Path

import pytest

from talent_job_change_intent_prediction.data import schema


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    row_count: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows([["value"] * len(header) for _ in range(row_count)])


def test_validate_csv_accepts_expected_structure(tmp_path: Path, settings) -> None:
    path = tmp_path / "aug_train.csv"
    _write_csv(path, schema.EXPECTED_COLUMNS, row_count=2)
    settings = _replace_contract(
        settings,
        expected_row_count=2,
        expected_size_bytes=path.stat().st_size,
        expected_sha256=schema._sha256(path),
    )

    manifest = schema.validate_csv(path, settings)

    assert manifest.row_count == 2
    assert manifest.columns == list(schema.EXPECTED_COLUMNS)
    assert len(manifest.sha256) == 64


def test_validate_csv_rejects_unexpected_header(tmp_path: Path, settings) -> None:
    path = tmp_path / "aug_train.csv"
    _write_csv(path, ("wrong_column",), row_count=1)
    settings = _replace_contract(
        settings,
        expected_row_count=1,
        expected_size_bytes=path.stat().st_size,
        expected_sha256=schema._sha256(path),
    )

    with pytest.raises(ValueError, match="Unexpected CSV columns"):
        schema.validate_csv(path, settings)


def test_validate_csv_rejects_wrong_pinned_row_count(tmp_path: Path, settings) -> None:
    path = tmp_path / "aug_train.csv"
    _write_csv(path, schema.EXPECTED_COLUMNS, row_count=1)
    settings = _replace_contract(
        settings,
        expected_row_count=2,
        expected_size_bytes=path.stat().st_size,
        expected_sha256=schema._sha256(path),
    )

    with pytest.raises(ValueError, match="Expected 2 data rows"):
        schema.validate_csv(path, settings)


def test_validate_csv_rejects_wrong_checksum(tmp_path: Path, settings) -> None:
    path = tmp_path / "aug_train.csv"
    _write_csv(path, schema.EXPECTED_COLUMNS, row_count=1)
    settings = _replace_contract(
        settings,
        expected_row_count=1,
        expected_size_bytes=path.stat().st_size,
        expected_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="checksum does not match"):
        schema.validate_csv(path, settings)


def test_validate_csv_rejects_wrong_file_size(tmp_path: Path, settings) -> None:
    path = tmp_path / "aug_train.csv"
    _write_csv(path, schema.EXPECTED_COLUMNS, row_count=1)
    settings = _replace_contract(
        settings,
        expected_row_count=1,
        expected_size_bytes=path.stat().st_size + 1,
        expected_sha256=schema._sha256(path),
    )

    with pytest.raises(ValueError, match="bytes"):
        schema.validate_csv(path, settings)


def _replace_contract(settings, **changes):
    from dataclasses import replace

    return replace(settings, **changes)
