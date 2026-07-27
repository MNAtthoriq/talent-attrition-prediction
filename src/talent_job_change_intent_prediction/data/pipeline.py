"""Run the Talent Job-Change Intent GCS-to-BigQuery ELT pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import kagglehub
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import bigquery, storage

from talent_job_change_intent_prediction.config import Settings
from talent_job_change_intent_prediction.data.schema import (
    EXPECTED_COLUMNS,
    RAW_BIGQUERY_SCHEMA,
    CsvManifest,
    validate_csv,
)

_BLOB_METADATA_KEYS = {
    "dataset_handle",
    "file_name",
    "sha256",
    "size_bytes",
    "row_count",
    "columns",
}


# Core Functions
def acquire_raw_data(settings: Settings) -> dict[str, Any]:
    """Reuse or create the immutable, verified raw object in Cloud Storage."""
    storage_client = _storage_client(settings)
    bucket = _get_bucket(storage_client, settings)
    blob = bucket.blob(settings.raw_object_name)

    if blob.exists(client=storage_client):
        blob.reload(client=storage_client)
        manifest = _validate_blob_contract(blob, settings)
        result = _cloud_manifest(settings, blob, manifest, reused=True)
        _write_json(settings.reports_dir / "data_manifest.json", result)
        print(f"Reusing verified raw object {settings.raw_gcs_uri}")
        return result

    with tempfile.TemporaryDirectory(prefix="talent-job-change-intent-") as temp_dir:
        raw_path = _download_to_directory(settings, Path(temp_dir))
        manifest = validate_csv(raw_path, settings)
        blob.metadata = _manifest_metadata(manifest)

        try:
            blob.upload_from_filename(
                str(raw_path),
                content_type="text/csv",
                if_generation_match=0,
                checksum="auto",
            )
        except PreconditionFailed:
            # A concurrent run created the same immutable object first.
            blob.reload(client=storage_client)
        else:
            blob.reload(client=storage_client)

    verified_manifest = _validate_blob_contract(blob, settings)
    result = _cloud_manifest(settings, blob, verified_manifest, reused=False)
    _write_json(settings.reports_dir / "data_manifest.json", result)
    print(f"Uploaded and verified immutable raw object {settings.raw_gcs_uri}")
    return result


def load_raw_to_bigquery(settings: Settings) -> None:
    """Load the verified Cloud Storage object into the raw BigQuery table."""
    storage_client = _storage_client(settings)
    blob = _get_bucket(storage_client, settings).blob(settings.raw_object_name)
    if not blob.exists(client=storage_client):
        raise RuntimeError(
            f"Raw object {settings.raw_gcs_uri} does not exist. "
            "Run `uv run talent-data acquire` first."
        )
    blob.reload(client=storage_client)
    _validate_blob_contract(blob, settings)

    client = _bigquery_client(settings)
    _get_dataset(client, settings)

    job_config = bigquery.LoadJobConfig(
        schema=list(RAW_BIGQUERY_SCHEMA),
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        max_bad_records=0,
        allow_jagged_rows=False,
        allow_quoted_newlines=False,
    )
    job = client.load_table_from_uri(
        settings.raw_gcs_uri,
        settings.raw_table_fqn,
        location=settings.location,
        job_config=job_config,
    )
    job.result()

    table = client.get_table(settings.raw_table_fqn)
    if table.num_rows != settings.expected_row_count:
        raise RuntimeError(
            f"BigQuery loaded {table.num_rows:,} rows; "
            f"expected {settings.expected_row_count:,}."
        )

    print(
        f"Loaded {table.num_rows:,} raw rows from {settings.raw_gcs_uri} "
        f"into {settings.raw_table_fqn}"
    )


def transform_in_bigquery(settings: Settings) -> None:
    """Transform the raw BigQuery table into the modeling table."""
    client = _bigquery_client(settings)
    _get_dataset(client, settings)
    sql = render_sql(settings.sql_dir / "01_create_modeling_table.sql", settings)
    client.query(sql, location=settings.location).result()
    print(f"Built typed modeling base {settings.modeling_table_fqn}")


def validate_in_bigquery(settings: Settings) -> dict[str, Any]:
    """Validate the modeling table in BigQuery."""
    client = _bigquery_client(settings)
    _get_dataset(client, settings)

    checks = _query_rows(
        client,
        render_sql(settings.sql_dir / "02_data_quality_checks.sql", settings),
        settings.location,
    )
    missingness = _query_rows(
        client,
        render_sql(settings.sql_dir / "03_missingness_profile.sql", settings),
        settings.location,
    )

    passed = all(
        row["failed_rows"] == 0 for row in checks if row["severity"] == "ERROR"
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_uri": settings.raw_gcs_uri,
        "source_sha256": settings.expected_sha256,
        "table": settings.modeling_table_fqn,
        "passed": passed,
        "checks": checks,
        "missingness": missingness,
    }
    report_path = settings.reports_dir / "data_validation.json"
    _write_json(report_path, report)

    print(f"Validation {'passed' if passed else 'failed'}: {report_path}")
    if not passed:
        failed_names = [
            row["check_name"]
            for row in checks
            if row["severity"] == "ERROR" and row["failed_rows"] > 0
        ]
        raise RuntimeError(
            "BigQuery data validation failed: " + ", ".join(failed_names)
        )

    return report


# Helper Functions
def render_sql(path: Path, settings: Settings) -> str:
    """Render SQL using the runtime configuration."""
    sql = path.read_text(encoding="utf-8")
    replacements = {
        "{{project_id}}": settings.project_id,
        "{{dataset_id}}": settings.dataset_id,
        "{{raw_table_id}}": settings.raw_table_id,
        "{{modeling_table_id}}": settings.modeling_table_id,
        "{{expected_row_count}}": str(settings.expected_row_count),
        "{{source_gcs_uri}}": settings.raw_gcs_uri,
        "{{source_sha256}}": settings.expected_sha256,
    }
    for placeholder, value in replacements.items():
        sql = sql.replace(placeholder, value)

    if "{{" in sql or "}}" in sql:
        raise ValueError(f"Unresolved SQL placeholder in {path}.")
    return sql


def _download_to_directory(settings: Settings, directory: Path) -> Path:
    """Download the Kaggle dataset to a local directory."""
    downloaded_path = Path(
        kagglehub.dataset_download(
            settings.kaggle_dataset_handle,
            path=settings.kaggle_train_file,
            output_dir=str(directory),
            force_download=True,
        )
    )
    if downloaded_path.is_dir():
        downloaded_path = downloaded_path / settings.kaggle_train_file
    expected_path = directory / settings.kaggle_train_file
    if downloaded_path.resolve() != expected_path.resolve():
        raise RuntimeError(
            f"Kaggle downloaded the file to an unexpected location: {downloaded_path}"
        )
    return downloaded_path


def _get_bucket(client: storage.Client, settings: Settings) -> storage.Bucket:
    """Get the Cloud Storage bucket."""
    try:
        bucket = client.get_bucket(settings.storage_bucket_name)
    except NotFound as error:
        raise RuntimeError(
            "The Cloud Storage bucket does not exist. Run `terraform apply` in "
            "infrastructure/terraform before running the data pipeline."
        ) from error

    if bucket.location.lower() != settings.location.lower():
        raise RuntimeError(
            f"Cloud Storage bucket location is {bucket.location}; "
            f"runtime configuration expects {settings.location}."
        )
    return bucket


def _get_dataset(
    client: bigquery.Client,
    settings: Settings,
) -> bigquery.Dataset:
    """Get the BigQuery dataset."""
    try:
        dataset = client.get_dataset(f"{settings.project_id}.{settings.dataset_id}")
    except NotFound as error:
        raise RuntimeError(
            "The BigQuery dataset does not exist. Run `terraform apply` in "
            "infrastructure/terraform before running the data pipeline."
        ) from error
    if dataset.location.lower() != settings.location.lower():
        raise RuntimeError(
            f"BigQuery dataset location is {dataset.location}; "
            f"runtime configuration expects {settings.location}."
        )
    return dataset


def _manifest_metadata(manifest: CsvManifest) -> dict[str, str]:
    """Return the manifest as a dictionary."""
    return {
        "dataset_handle": manifest.dataset_handle,
        "file_name": manifest.file_name,
        "sha256": manifest.sha256,
        "size_bytes": str(manifest.size_bytes),
        "row_count": str(manifest.row_count),
        "columns": json.dumps(manifest.columns, separators=(",", ":")),
    }


def _validate_blob_contract(
    blob: storage.Blob,
    settings: Settings,
) -> CsvManifest:
    """Validate the immutable raw object contract."""
    metadata = blob.metadata or {}
    missing_keys = sorted(_BLOB_METADATA_KEYS - metadata.keys())
    if missing_keys:
        raise RuntimeError(
            f"Raw object {settings.raw_gcs_uri} is missing provenance metadata: "
            + ", ".join(missing_keys)
        )

    expected_metadata = {
        "dataset_handle": settings.kaggle_dataset_handle,
        "file_name": settings.kaggle_train_file,
        "sha256": settings.expected_sha256,
        "size_bytes": str(settings.expected_size_bytes),
        "row_count": str(settings.expected_row_count),
        "columns": json.dumps(list(EXPECTED_COLUMNS), separators=(",", ":")),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if blob.size != settings.expected_size_bytes:
        mismatches["gcs_size_bytes"] = {
            "expected": settings.expected_size_bytes,
            "actual": blob.size,
        }
    if mismatches:
        raise RuntimeError(
            f"Raw object {settings.raw_gcs_uri} conflicts with the pinned "
            f"data contract: {json.dumps(mismatches, sort_keys=True)}"
        )

    return CsvManifest(
        dataset_handle=metadata["dataset_handle"],
        file_name=metadata["file_name"],
        sha256=metadata["sha256"],
        size_bytes=int(metadata["size_bytes"]),
        row_count=int(metadata["row_count"]),
        columns=json.loads(metadata["columns"]),
    )


def _cloud_manifest(
    settings: Settings,
    blob: storage.Blob,
    manifest: CsvManifest,
    *,
    reused: bool,
) -> dict[str, Any]:
    """Return the manifest as a dictionary."""
    result = manifest.to_dict()
    result.update(
        {
            "gcs_uri": settings.raw_gcs_uri,
            "gcs_generation": str(blob.generation),
            "reused_existing_object": reused,
        }
    )
    return result


def _storage_client(settings: Settings) -> storage.Client:
    """Get the Cloud Storage client."""
    return storage.Client(project=settings.project_id)


def _bigquery_client(settings: Settings) -> bigquery.Client:
    """Get the BigQuery client."""
    return bigquery.Client(
        project=settings.project_id,
        location=settings.location,
    )


def _load_local_env(settings: Settings) -> None:
    """Load optional local secrets without overriding terminal or CI values."""
    repository_root = settings.config_path.parent.parent
    load_dotenv(repository_root / ".env", override=False)


def _query_rows(
    client: bigquery.Client, sql: str, location: str
) -> list[dict[str, Any]]:
    """Run a BigQuery query and return the results as a list of dictionaries."""
    return [
        {key: _json_value(value) for key, value in dict(row).items()}
        for row in client.query(sql, location=location).result()
    ]


def _json_value(value: Any) -> Any:
    """Convert a BigQuery value to a JSON-serializable type."""
    if isinstance(value, Decimal):
        return float(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_pipeline(settings: Settings) -> None:
    """Run the data pipeline in sequence."""
    acquire_raw_data(settings)
    load_raw_to_bigquery(settings)
    transform_in_bigquery(settings)
    validate_in_bigquery(settings)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="talent-data",
        description="Run the Talent Job-Change Intent GCS-to-BigQuery ELT pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "acquire",
        help="Reuse or upload the pinned Kaggle file to immutable Cloud Storage.",
    )
    subparsers.add_parser(
        "load", help="Load the verified Cloud Storage CSV into BigQuery."
    )
    subparsers.add_parser(
        "transform", help="Build the typed modeling base with BigQuery SQL."
    )
    subparsers.add_parser(
        "validate", help="Run data-quality checks and write a JSON report."
    )
    subparsers.add_parser(
        "run", help="Acquire, load, transform, and validate in one idempotent command."
    )
    return parser


def main() -> None:
    """Run the data pipeline."""
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings.load()
    _load_local_env(settings)

    if args.command == "acquire":
        acquire_raw_data(settings)
    elif args.command == "load":
        load_raw_to_bigquery(settings)
    elif args.command == "transform":
        transform_in_bigquery(settings)
    elif args.command == "validate":
        validate_in_bigquery(settings)
    elif args.command == "run":
        _run_pipeline(settings)
