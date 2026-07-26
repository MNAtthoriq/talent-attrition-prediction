"""Test API validation, probability scoring, ranking, and metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from talent_attrition_prediction.modeling.data import MODEL_FEATURES
from talent_attrition_prediction.serving.app import create_app
from talent_attrition_prediction.serving.config import ServingSettings
from talent_attrition_prediction.serving.schemas import CandidateFeatures
from talent_attrition_prediction.serving.service import (
    ModelDescriptor,
    ModelService,
)


class _PredictingModel:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        assert data.columns.tolist() == list(MODEL_FEATURES)
        probability = data["city_development_index"].to_numpy()
        return np.column_stack([1 - probability, probability])


def _candidate(city_development_index: float = 0.25) -> dict[str, object]:
    return {
        "city_development_index": city_development_index,
        "city": " city_103 ",
        "relevant_experience": "Has relevant experience",
        "enrolled_university": "no_enrollment",
        "education_level": "Graduate",
        "major_discipline": "STEM",
        "experience": "10",
        "company_size": "100-500",
        "company_type": "Pvt Ltd",
        "last_new_job": "1",
    }


@pytest.fixture
def client() -> TestClient:
    service = ModelService(
        _PredictingModel(),
        ModelDescriptor(
            model_uri="models:/talent-attrition-classifier@candidate",
            model_name="talent-attrition-classifier",
            model_version="1",
            model_alias="candidate",
            run_id="run-123",
            git_commit="abc123",
            source_sha256="a" * 64,
            optuna_study="study-123",
            optuna_trial=22,
            preprocessing="ordinal_aware_without_city",
            parameters={"model_name": "lightgbm"},
            test_metrics={"test_average_precision": 0.5461},
        ),
    )
    settings = ServingSettings(
        model_uri=service.descriptor.model_uri,
        tracking_uri="sqlite:////tmp/unused.db",
    )
    with TestClient(create_app(service=service, settings=settings)) as test_client:
        yield test_client


def test_health_and_model_info_expose_ready_model(client: TestClient) -> None:
    health = client.get("/health")
    info = client.get("/model-info")

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert info.status_code == 200
    assert info.json()["model_version"] == "1"
    assert info.json()["optuna_trial"] == 22
    assert info.json()["git_commit"] == "abc123"
    assert info.json()["test_metrics"]["test_average_precision"] == 0.5461
    assert info.json()["features"] == list(MODEL_FEATURES)


def test_predict_returns_both_probabilities(client: TestClient) -> None:
    response = client.post("/predict", json=_candidate(0.25))

    assert response.status_code == 200
    assert response.json() == {
        "attrition_probability": 0.25,
        "retention_probability": 0.75,
    }


def test_batch_preserves_input_order_and_adds_priority_rank(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict-batch",
        json={
            "candidates": [
                _candidate(0.80),
                _candidate(0.10),
                _candidate(0.40),
            ]
        },
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [row["attrition_probability"] for row in predictions] == [
        0.80,
        0.10,
        0.40,
    ]
    assert [row["retention_priority_rank"] for row in predictions] == [3, 1, 2]


def test_request_rejects_unknown_and_invalid_fields(client: TestClient) -> None:
    payload = _candidate(1.5) | {"training_hours": 80}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    locations = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert ("body", "city_development_index") in locations
    assert ("body", "training_hours") in locations


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("city", "Jakarta"),
        ("relevant_experience", "Relevant"),
        ("enrolled_university", "Not enrolled"),
        ("education_level", "Graduatte"),
        ("major_discipline", "Computer Science"),
        ("experience", "21"),
        ("company_size", "Huge Company"),
        ("company_type", "Private"),
        ("last_new_job", "5"),
    ],
)
def test_request_rejects_values_outside_training_contract(
    client: TestClient,
    field: str,
    invalid_value: str,
) -> None:
    response = client.post(
        "/predict",
        json=_candidate() | {field: invalid_value},
    )

    assert response.status_code == 422
    assert any(
        tuple(error["loc"]) == ("body", field) for error in response.json()["detail"]
    )


def test_blank_categories_become_missing() -> None:
    candidate = CandidateFeatures.model_validate(_candidate() | {"company_type": "  "})

    assert candidate.city == "city_103"
    assert candidate.company_type is None


@pytest.mark.parametrize("field", ["city", "relevant_experience"])
def test_required_categories_reject_null(
    client: TestClient,
    field: str,
) -> None:
    response = client.post("/predict", json=_candidate() | {field: None})

    assert response.status_code == 422


def test_nullable_categories_accept_explicit_null(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json=_candidate()
        | {
            "enrolled_university": None,
            "education_level": None,
            "major_discipline": None,
            "experience": None,
            "company_size": None,
            "company_type": None,
            "last_new_job": None,
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize("invalid_value", [True, "0.25"])
def test_city_development_index_rejects_wrong_json_types(
    client: TestClient,
    invalid_value: object,
) -> None:
    response = client.post(
        "/predict",
        json=_candidate() | {"city_development_index": invalid_value},
    )

    assert response.status_code == 422


def test_model_service_rejects_invalid_output_shape() -> None:
    class _InvalidModel:
        def predict(self, data: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(data), 3))

    service = ModelService(
        _InvalidModel(),
        ModelDescriptor(model_uri="invalid"),
    )

    with pytest.raises(RuntimeError, match="probability columns"):
        service.predict_probabilities([_candidate()])


def test_required_nullable_feature_cannot_be_omitted() -> None:
    payload = _candidate()
    payload.pop("company_size")

    with pytest.raises(ValidationError):
        CandidateFeatures.model_validate(payload)
