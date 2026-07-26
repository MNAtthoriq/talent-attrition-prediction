"""Run the FastAPI service with Uvicorn."""

from __future__ import annotations

import uvicorn

from talent_attrition_prediction.serving.config import ServingSettings


def main() -> None:
    """Start the configured prediction API."""
    settings = ServingSettings.load()
    uvicorn.run(
        "talent_attrition_prediction.serving.app:app",
        host=settings.host,
        port=settings.port,
    )
