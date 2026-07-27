FROM python:3.12.11-slim-bookworm

ARG UV_VERSION=0.11.29

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TALENT_API_HOST=0.0.0.0 \
    TALENT_MODEL_URI=/app/model \
    TALENT_MODEL_DESCRIPTOR=/app/model-descriptor.json \
    MLFLOW_TRACKING_URI=file:///tmp/mlruns \
    PORT=8080

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}" \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

COPY --chown=appuser:appuser .deployment/model ./model
COPY --chown=appuser:appuser .deployment/model-descriptor.json ./model-descriptor.json

USER appuser

EXPOSE 8080

CMD ["/app/.venv/bin/talent-api"]
