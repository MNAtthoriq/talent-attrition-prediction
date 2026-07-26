"""Schema and data validation for the Talent Attrition GCS-to-BigQuery ELT pipeline."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from google.cloud import bigquery

from talent_attrition_prediction.config import Settings

# Constants
EXPECTED_COLUMNS = (
    "enrollee_id",
    "city",
    "city_development_index",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
    "training_hours",
    "target",
)

RAW_BIGQUERY_SCHEMA = tuple(
    bigquery.SchemaField(column, "STRING", mode="NULLABLE")
    for column in EXPECTED_COLUMNS
)


@dataclass(frozen=True)
class CsvManifest:
    """A CSV file manifest."""

    dataset_handle: str
    file_name: str
    sha256: str
    size_bytes: int
    row_count: int
    columns: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return the manifest as a dictionary."""
        return asdict(self)


def validate_csv(path: Path, settings: Settings) -> CsvManifest:
    """Validate the CSV file and return its manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Temporary raw dataset not found at {path}.")

    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path} is empty.") from error

        if tuple(header) != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected CSV columns.\n"
                f"Expected: {list(EXPECTED_COLUMNS)}\n"
                f"Received: {header}"
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_COLUMNS):
                raise ValueError(
                    f"CSV row {line_number} has {len(row)} fields; "
                    f"expected {len(EXPECTED_COLUMNS)}."
                )
            row_count += 1

    if row_count != settings.expected_row_count:
        raise ValueError(
            f"Expected {settings.expected_row_count:,} data rows from the pinned "
            f"dataset version, but found {row_count:,}."
        )

    size_bytes = path.stat().st_size
    if size_bytes != settings.expected_size_bytes:
        raise ValueError(
            f"Expected {settings.expected_size_bytes:,} bytes from the pinned dataset "
            f"version, but found {size_bytes:,}."
        )

    sha256 = _sha256(path)
    if sha256 != settings.expected_sha256:
        raise ValueError(
            "The CSV checksum does not match the pinned Kaggle dataset "
            f"version. Expected {settings.expected_sha256}, received {sha256}."
        )

    return CsvManifest(
        dataset_handle=settings.kaggle_dataset_handle,
        file_name=path.name,
        sha256=sha256,
        size_bytes=size_bytes,
        row_count=row_count,
        columns=header,
    )


def _sha256(path: Path) -> str:
    """Return the SHA256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
