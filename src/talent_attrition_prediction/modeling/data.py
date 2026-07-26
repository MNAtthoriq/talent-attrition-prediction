"""Load, validate, and split the BigQuery modeling base without leakage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

from talent_attrition_prediction.config import Settings

IDENTIFIER_COLUMN = "enrollee_id"
TARGET_COLUMN = "target"
PROTECTED_COLUMNS = ("gender",)
LINEAGE_COLUMNS = ("source_sha256",)

NUMERIC_FEATURES = ("city_development_index",)
CITY_FEATURE = "city"
ORDINAL_FEATURES = (
    "education_level",
    "experience",
    "company_size",
)
NOMINAL_FEATURES = (
    "relevant_experience",
    "enrolled_university",
    "major_discipline",
    "company_type",
    "last_new_job",
)
ORDINAL_CATEGORY_ORDERS = {
    "education_level": (
        "Primary School",
        "High School",
        "Graduate",
        "Masters",
        "Phd",
    ),
    "experience": (
        "<1",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        ">20",
    ),
    "company_size": (
        "<10",
        "10/49",
        "50-99",
        "100-500",
        "500-999",
        "1000-4999",
        "5000-9999",
        "10000+",
    ),
}
CATEGORICAL_FEATURES = (
    CITY_FEATURE,
    *NOMINAL_FEATURES,
    *ORDINAL_FEATURES,
)
MODEL_FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

QUERY_COLUMNS = (
    IDENTIFIER_COLUMN,
    *MODEL_FEATURES,
    *PROTECTED_COLUMNS,
    TARGET_COLUMN,
    *LINEAGE_COLUMNS,
)

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitData:
    """Development/test data plus protected audit attributes."""

    X_development: pd.DataFrame
    X_test: pd.DataFrame
    y_development: pd.Series
    y_test: pd.Series
    audit_development: pd.DataFrame
    audit_test: pd.DataFrame
    source_sha256: str
    random_state: int
    test_size: float


def load_modeling_data(
    settings: Settings,
    *,
    client: bigquery.Client | None = None,
) -> pd.DataFrame:
    """Read the pinned modeling table from BigQuery and validate its contract."""
    query_client = client or bigquery.Client(
        project=settings.project_id,
        location=settings.location,
    )
    query = _modeling_query(settings)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "source_sha256",
                "STRING",
                settings.expected_sha256,
            )
        ]
    )
    rows = query_client.query(
        query,
        location=settings.location,
        job_config=job_config,
    ).result()
    frame = pd.DataFrame((dict(row) for row in rows), columns=QUERY_COLUMNS)
    return validate_modeling_data(frame, settings)


def validate_modeling_data(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Validate and normalize the in-memory modeling table."""
    missing_columns = sorted(set(QUERY_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Modeling data is missing required columns: " + ", ".join(missing_columns)
        )

    validated = frame.loc[:, QUERY_COLUMNS].copy()
    if len(validated) != settings.expected_row_count:
        raise ValueError(
            f"Expected {settings.expected_row_count:,} modeling rows, "
            f"received {len(validated):,}."
        )

    if validated[IDENTIFIER_COLUMN].isna().any():
        raise ValueError("enrollee_id cannot contain missing values.")
    if validated[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError("enrollee_id must be unique before splitting.")

    validated[TARGET_COLUMN] = pd.to_numeric(
        validated[TARGET_COLUMN],
        errors="coerce",
    )
    if validated[TARGET_COLUMN].isna().any():
        raise ValueError("target cannot contain missing or non-numeric values.")
    invalid_targets = set(validated[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"target must contain only 0 and 1; found {invalid_targets}.")
    validated[TARGET_COLUMN] = validated[TARGET_COLUMN].astype("int8")

    validated["city_development_index"] = pd.to_numeric(
        validated["city_development_index"],
        errors="coerce",
    )
    if validated["city_development_index"].isna().any():
        raise ValueError("city_development_index cannot be missing or non-numeric.")
    if not validated["city_development_index"].between(0, 1).all():
        raise ValueError("city_development_index must be between 0 and 1.")

    if validated["source_sha256"].isna().any():
        raise ValueError("source_sha256 cannot be missing.")
    source_hashes = set(validated["source_sha256"].unique())
    if source_hashes != {settings.expected_sha256}:
        raise ValueError(
            "Modeling rows do not all match the pinned source SHA-256 contract."
        )

    class_counts = validated[TARGET_COLUMN].value_counts()
    if set(class_counts.index) != {0, 1} or class_counts.min() < 2:
        raise ValueError("Both target classes need at least two rows for splitting.")

    return validated.sort_values(IDENTIFIER_COLUMN, kind="stable").reset_index(
        drop=True
    )


def split_modeling_data(
    frame: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SplitData:
    """Create a deterministic stratified holdout split."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0 and less than 0.5.")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer.")

    ordered = frame.sort_values(IDENTIFIER_COLUMN, kind="stable").reset_index(drop=True)
    development_index, test_index = train_test_split(
        ordered.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=ordered[TARGET_COLUMN],
    )

    development = ordered.loc[development_index].sort_values(
        IDENTIFIER_COLUMN,
        kind="stable",
    )
    test = ordered.loc[test_index].sort_values(IDENTIFIER_COLUMN, kind="stable")

    audit_columns = (IDENTIFIER_COLUMN, *PROTECTED_COLUMNS)
    source_sha256 = str(ordered["source_sha256"].iloc[0])
    return SplitData(
        X_development=development.loc[:, MODEL_FEATURES].reset_index(drop=True),
        X_test=test.loc[:, MODEL_FEATURES].reset_index(drop=True),
        y_development=development[TARGET_COLUMN].reset_index(drop=True),
        y_test=test[TARGET_COLUMN].reset_index(drop=True),
        audit_development=development.loc[:, audit_columns].reset_index(drop=True),
        audit_test=test.loc[:, audit_columns].reset_index(drop=True),
        source_sha256=source_sha256,
        random_state=random_state,
        test_size=test_size,
    )


def build_split_manifest(
    split: SplitData,
    *,
    preprocessing_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an auditable manifest without storing participant-level data."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_sha256": split.source_sha256,
        "random_state": split.random_state,
        "requested_test_size": split.test_size,
        "features": list(MODEL_FEATURES),
        "excluded_from_model": {
            "enrollee_id": "identifier; retained only for split auditing",
            "gender": "protected characteristic; retained for fairness auditing",
            "training_hours": "unavailable at prediction time",
            "target": "prediction target",
            "source_sha256": "data-lineage field",
        },
        "development": _partition_manifest(
            split.audit_development[IDENTIFIER_COLUMN],
            split.y_development,
        ),
        "test": _partition_manifest(
            split.audit_test[IDENTIFIER_COLUMN],
            split.y_test,
        ),
    }
    if preprocessing_candidates is not None:
        manifest["preprocessing_candidates"] = preprocessing_candidates
    return manifest


def write_split_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the generated split manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _modeling_query(settings: Settings) -> str:
    """Return the deterministic BigQuery extraction query."""
    columns = ",\n  ".join(f"`{column}`" for column in QUERY_COLUMNS)
    return (
        "SELECT\n"
        f"  {columns}\n"
        f"FROM `{settings.modeling_table_fqn}`\n"
        "WHERE `source_sha256` = @source_sha256\n"
        f"ORDER BY `{IDENTIFIER_COLUMN}`"
    )


def _partition_manifest(ids: pd.Series, target: pd.Series) -> dict[str, Any]:
    """Summarize one split partition."""
    target_counts = target.value_counts().sort_index()
    return {
        "rows": len(target),
        "positive_rows": int(target_counts.get(1, 0)),
        "positive_rate": float(target.mean()),
        "enrollee_id_sha256": _id_sha256(ids),
    }


def _id_sha256(ids: pd.Series) -> str:
    """Hash sorted IDs so a split can be verified without publishing row data."""
    canonical = "\n".join(str(value) for value in sorted(ids.tolist()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
