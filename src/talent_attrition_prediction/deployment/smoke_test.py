"""Verify the deployed Cloud Run API from health through prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from talent_attrition_prediction.deployment.paths import (
    REPOSITORY_ROOT,
    TERRAFORM_DIR,
)

SAMPLE_CANDIDATE = {
    "city_development_index": 0.92,
    "city": "city_103",
    "relevant_experience": "Has relevant experience",
    "enrolled_university": "no_enrollment",
    "education_level": "Graduate",
    "major_discipline": "STEM",
    "experience": "10",
    "company_size": "100-500",
    "company_type": "Pvt Ltd",
    "last_new_job": "1",
}


def smoke_test(base_url: str, *, wait_seconds: int = 180) -> None:
    """Wait for readiness, then validate metadata and real model inference."""
    root = base_url.rstrip("/")
    deadline = time.monotonic() + wait_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            health = _request_json(f"{root}/health")
            if health.get("status") == "ready":
                break
        except (RuntimeError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(5)
    else:
        raise RuntimeError(
            f"API did not become ready within {wait_seconds} seconds: {last_error}"
        )

    model_info = _request_json(f"{root}/model-info")
    prediction = _request_json(
        f"{root}/predict",
        method="POST",
        payload=SAMPLE_CANDIDATE,
    )

    probability = prediction.get("attrition_probability")
    retention = prediction.get("retention_probability")
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise RuntimeError("Prediction did not contain a valid probability.")
    if (
        not isinstance(retention, (int, float))
        or abs(probability + retention - 1) > 1e-9
    ):
        raise RuntimeError("Prediction probabilities do not sum to one.")
    if not model_info.get("source_sha256") or not model_info.get("run_id"):
        raise RuntimeError("Model lineage is missing from /model-info.")

    print("Cloud Run smoke test passed.")
    print(f"Health: {health['status']}")
    print(
        "Model: "
        f"{model_info.get('model_name')} version "
        f"{model_info.get('model_version')}"
    )
    print(f"Attrition probability: {probability:.6f}")


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} returned {error.code}: {detail}") from error
    result = json.loads(body)
    if not isinstance(result, dict):
        raise TypeError(f"{method} {url} did not return a JSON object.")
    return result


def _terraform_output(name: str) -> str:
    completed = subprocess.run(
        [
            "terraform",
            f"-chdir={TERRAFORM_DIR}",
            "output",
            "-raw",
            name,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed API.")
    parser.add_argument(
        "url",
        nargs="?",
        help=(
            "Cloud Run service URL. When omitted, read cloud_run_url from "
            "Terraform state."
        ),
    )
    parser.add_argument("--wait-seconds", type=int, default=180)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        base_url = args.url or _terraform_output("cloud_run_url")
        if not base_url:
            raise RuntimeError(
                "Terraform has no Cloud Run URL. Deploy first with "
                "`uv run talent-deploy`."
            )
        smoke_test(base_url, wait_seconds=args.wait_seconds)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Smoke test failed: {error}") from error


if __name__ == "__main__":
    main()
