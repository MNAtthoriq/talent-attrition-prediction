"""Test modeling-table validation and deterministic splitting."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from talent_job_change_intent_prediction.modeling import data


class _FakeQueryJob:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def result(self) -> list[dict[str, object]]:
        return self._rows


class _FakeBigQueryClient:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.query_text: str | None = None
        self.location: str | None = None
        self.job_config = None

    def query(self, query: str, *, location: str, job_config):
        self.query_text = query
        self.location = location
        self.job_config = job_config
        return _FakeQueryJob(self._rows)


def _modeling_frame(row_count: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "enrollee_id": 1000 + index,
                "city_development_index": 0.60 + (index % 20) / 100,
                "city": f"city_{index % 5}",
                "relevant_experience": (
                    "Has relevant experience" if index % 2 else "No relevant experience"
                ),
                "enrolled_university": (None if index % 7 == 0 else "no_enrollment"),
                "education_level": "Graduate" if index % 3 else "Masters",
                "major_discipline": None if index % 6 == 0 else "STEM",
                "experience": str(1 + index % 20),
                "company_size": None if index % 5 == 0 else "50-99",
                "company_type": "Pvt Ltd",
                "last_new_job": "never" if index % 4 == 0 else "1",
                "gender": None if index % 9 == 0 else "Male",
                "target": 1 if index % 4 == 0 else 0,
                "source_sha256": "a" * 64,
            }
        )
    return pd.DataFrame(rows)


def _settings_for_frame(settings, frame: pd.DataFrame):
    return replace(
        settings,
        expected_row_count=len(frame),
        expected_sha256="a" * 64,
    )


def test_load_modeling_data_uses_pinned_parameterized_query(settings) -> None:
    frame = _modeling_frame()
    current_settings = _settings_for_frame(settings, frame)
    client = _FakeBigQueryClient(frame.to_dict(orient="records"))

    loaded = data.load_modeling_data(current_settings, client=client)

    assert len(loaded) == len(frame)
    assert current_settings.modeling_table_fqn in client.query_text
    assert "WHERE `source_sha256` = @source_sha256" in client.query_text
    assert client.location == current_settings.location
    parameter = client.job_config.query_parameters[0]
    assert parameter.name == "source_sha256"
    assert parameter.value == current_settings.expected_sha256


def test_validation_sorts_and_normalizes_modeling_data(settings) -> None:
    frame = _modeling_frame().sample(frac=1, random_state=9)

    validated = data.validate_modeling_data(
        frame,
        _settings_for_frame(settings, frame),
    )

    assert validated["enrollee_id"].is_monotonic_increasing
    assert str(validated["target"].dtype) == "int8"
    assert tuple(validated.columns) == data.QUERY_COLUMNS


def test_validation_rejects_duplicate_identifiers(settings) -> None:
    frame = _modeling_frame()
    frame.loc[1, "enrollee_id"] = frame.loc[0, "enrollee_id"]

    with pytest.raises(ValueError, match="must be unique"):
        data.validate_modeling_data(
            frame,
            _settings_for_frame(settings, frame),
        )


def test_validation_rejects_invalid_target(settings) -> None:
    frame = _modeling_frame()
    frame["target"] = frame["target"].astype(float)
    frame.loc[0, "target"] = 0.5

    with pytest.raises(ValueError, match="only 0 and 1"):
        data.validate_modeling_data(
            frame,
            _settings_for_frame(settings, frame),
        )


def test_split_is_reproducible_stratified_and_disjoint(settings) -> None:
    frame = _modeling_frame()
    validated = data.validate_modeling_data(
        frame,
        _settings_for_frame(settings, frame),
    )

    first = data.split_modeling_data(validated)
    second = data.split_modeling_data(validated.sample(frac=1, random_state=7))

    assert (
        first.audit_test["enrollee_id"].tolist()
        == second.audit_test["enrollee_id"].tolist()
    )
    assert set(first.audit_development["enrollee_id"]).isdisjoint(
        first.audit_test["enrollee_id"]
    )
    assert len(first.X_development) == 32
    assert len(first.X_test) == 8
    assert first.y_development.mean() == first.y_test.mean() == 0.25
    assert "gender" not in first.X_development
    assert "training_hours" not in first.X_development


def test_split_manifest_does_not_publish_participant_ids(settings) -> None:
    frame = _modeling_frame()
    validated = data.validate_modeling_data(
        frame,
        _settings_for_frame(settings, frame),
    )
    split = data.split_modeling_data(validated)

    candidates = [
        {
            "name": "all_ohe_without_city",
            "transformed_feature_count": 19,
        }
    ]
    manifest = data.build_split_manifest(
        split,
        preprocessing_candidates=candidates,
    )
    serialized = str(manifest)

    assert manifest["development"]["rows"] == 32
    assert manifest["test"]["rows"] == 8
    assert manifest["preprocessing_candidates"] == candidates
    assert "1000" not in serialized
    assert len(manifest["test"]["enrollee_id_sha256"]) == 64


def test_split_rejects_invalid_test_size(settings) -> None:
    frame = _modeling_frame()
    validated = data.validate_modeling_data(
        frame,
        _settings_for_frame(settings, frame),
    )

    with pytest.raises(ValueError, match="less than 0.5"):
        data.split_modeling_data(validated, test_size=0.5)
