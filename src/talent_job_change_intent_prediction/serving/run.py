"""Run the FastAPI service with Uvicorn."""

from __future__ import annotations

import uvicorn

from talent_job_change_intent_prediction.serving.config import ServingSettings


def main() -> None:
    """Start the configured prediction API."""
    settings = ServingSettings.load()
    uvicorn.run(
        "talent_job_change_intent_prediction.serving.app:app",
        host=settings.host,
        port=settings.port,
    )
