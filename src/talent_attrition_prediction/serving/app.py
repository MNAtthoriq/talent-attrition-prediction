"""FastAPI application for individual and batch attrition-risk scoring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, Request

from talent_attrition_prediction.modeling.data import MODEL_FEATURES
from talent_attrition_prediction.serving.config import ServingSettings
from talent_attrition_prediction.serving.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CandidateFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from talent_attrition_prediction.serving.service import ModelService


def _model_service(request: Request) -> ModelService:
    """Return the service initialized during application startup."""
    return request.app.state.model_service


ModelServiceDependency = Annotated[ModelService, Depends(_model_service)]


def create_app(
    *,
    service: ModelService | None = None,
    settings: ServingSettings | None = None,
) -> FastAPI:
    """Create an application with injectable dependencies for testing."""
    resolved_settings = settings or ServingSettings.load()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.model_service = service or ModelService.load(
            model_uri=resolved_settings.model_uri,
            tracking_uri=resolved_settings.tracking_uri,
        )
        yield

    application = FastAPI(
        title="Talent Attrition Prediction API",
        version="0.1.0",
        description=(
            "Scores pre-training participants with the complete registered "
            "preprocessing and classification pipeline."
        ),
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(
        model_service: ModelServiceDependency,
    ) -> HealthResponse:
        return HealthResponse(
            status="ready",
            model_uri=model_service.descriptor.model_uri,
        )

    @application.get("/model-info", response_model=ModelInfoResponse)
    def model_info(
        model_service: ModelServiceDependency,
    ) -> ModelInfoResponse:
        descriptor = model_service.descriptor
        return ModelInfoResponse(
            model_uri=descriptor.model_uri,
            model_name=descriptor.model_name,
            model_version=descriptor.model_version,
            model_alias=descriptor.model_alias,
            run_id=descriptor.run_id,
            git_commit=descriptor.git_commit,
            source_sha256=descriptor.source_sha256,
            optuna_study=descriptor.optuna_study,
            optuna_trial=descriptor.optuna_trial,
            preprocessing=descriptor.preprocessing,
            parameters=descriptor.parameters,
            test_metrics=descriptor.test_metrics,
            features=list(MODEL_FEATURES),
            prediction_output=(
                "Probability that target=1 (participant leaves or changes jobs)."
            ),
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        candidate: CandidateFeatures,
        model_service: ModelServiceDependency,
    ) -> PredictionResponse:
        attrition_probability = float(
            model_service.predict_probabilities([candidate.model_dump()])[0]
        )
        return PredictionResponse(
            attrition_probability=attrition_probability,
            retention_probability=1 - attrition_probability,
        )

    @application.post("/predict-batch", response_model=BatchPredictionResponse)
    def predict_batch(
        request: BatchPredictionRequest,
        model_service: ModelServiceDependency,
    ) -> BatchPredictionResponse:
        probabilities = model_service.predict_probabilities(
            [candidate.model_dump() for candidate in request.candidates]
        )
        priority_order = np.argsort(probabilities, kind="stable")
        ranks = np.empty(len(probabilities), dtype=int)
        ranks[priority_order] = np.arange(1, len(probabilities) + 1)
        return BatchPredictionResponse(
            predictions=[
                BatchPredictionItem(
                    input_index=index,
                    attrition_probability=float(probability),
                    retention_probability=float(1 - probability),
                    retention_priority_rank=int(ranks[index]),
                )
                for index, probability in enumerate(probabilities)
            ]
        )

    return application


app = create_app()
