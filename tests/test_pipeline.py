"""Test the Talent Job-Change Intent GCS-to-BigQuery ELT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from talent_job_change_intent_prediction.data import pipeline


@dataclass
class FakeBlob:
    metadata: dict[str, str]
    size: int
    generation: int = 7


def _valid_metadata(settings) -> dict[str, str]:
    return {
        "dataset_handle": settings.kaggle_dataset_handle,
        "file_name": settings.kaggle_train_file,
        "sha256": settings.expected_sha256,
        "size_bytes": str(settings.expected_size_bytes),
        "row_count": str(settings.expected_row_count),
        "columns": (
            '["enrollee_id","city","city_development_index","gender",'
            '"relevent_experience","enrolled_university","education_level",'
            '"major_discipline","experience","company_size","company_type",'
            '"last_new_job","training_hours","target"]'
        ),
    }


def test_existing_gcs_object_must_match_pinned_contract(settings) -> None:
    blob = FakeBlob(
        metadata=_valid_metadata(settings),
        size=settings.expected_size_bytes,
    )

    manifest = pipeline._validate_blob_contract(blob, settings)

    assert manifest.sha256 == settings.expected_sha256
    assert manifest.row_count == settings.expected_row_count
    assert manifest.columns[-1] == "target"


def test_existing_gcs_object_with_different_hash_is_rejected(settings) -> None:
    metadata = _valid_metadata(settings)
    metadata["sha256"] = "0" * 64
    blob = FakeBlob(metadata=metadata, size=settings.expected_size_bytes)

    try:
        pipeline._validate_blob_contract(blob, settings)
    except RuntimeError as error:
        assert "conflicts with the pinned data contract" in str(error)
    else:
        raise AssertionError("Mismatched immutable object was accepted.")


class FakeLoadJob:
    def result(self) -> None:
        return None


class FakeBigQueryClient:
    def __init__(self, expected_rows: int) -> None:
        self.expected_rows = expected_rows
        self.source_uri: str | None = None
        self.destination: str | None = None
        self.job_config: Any = None

    def load_table_from_uri(
        self,
        source_uri: str,
        destination: str,
        *,
        location: str,
        job_config: Any,
    ) -> FakeLoadJob:
        self.source_uri = source_uri
        self.destination = destination
        self.job_config = job_config
        return FakeLoadJob()

    def get_table(self, table_id: str) -> Any:
        return type("Table", (), {"num_rows": self.expected_rows})()


class FakeStorageBlob(FakeBlob):
    def exists(self, *, client: Any) -> bool:
        return True

    def reload(self, *, client: Any) -> None:
        return None


class FakeBucket:
    def __init__(self, blob: FakeStorageBlob) -> None:
        self._blob = blob

    def blob(self, object_name: str) -> FakeStorageBlob:
        return self._blob


def test_bigquery_load_uses_gcs_uri_not_local_file(
    settings,
    monkeypatch,
) -> None:
    blob = FakeStorageBlob(
        metadata=_valid_metadata(settings),
        size=settings.expected_size_bytes,
    )
    bigquery_client = FakeBigQueryClient(settings.expected_row_count)

    monkeypatch.setattr(pipeline, "_storage_client", lambda _: object())
    monkeypatch.setattr(
        pipeline,
        "_get_bucket",
        lambda client, current_settings: FakeBucket(blob),
    )
    monkeypatch.setattr(pipeline, "_bigquery_client", lambda _: bigquery_client)
    monkeypatch.setattr(pipeline, "_get_dataset", lambda client, current: object())

    pipeline.load_raw_to_bigquery(settings)

    assert bigquery_client.source_uri == settings.raw_gcs_uri
    assert bigquery_client.destination == settings.raw_table_fqn
