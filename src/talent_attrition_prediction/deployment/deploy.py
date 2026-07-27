"""Build and deploy the registered model API to Google Cloud Run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

from talent_attrition_prediction.deployment.export_model import export_model
from talent_attrition_prediction.deployment.paths import (
    REPOSITORY_ROOT,
    TERRAFORM_DIR,
    TERRAFORM_VARIABLES_FILE,
)
from talent_attrition_prediction.deployment.smoke_test import smoke_test
from talent_attrition_prediction.serving.config import ServingSettings

PROJECT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
IMMUTABLE_IMAGE = re.compile(r"^.+-docker\.pkg\.dev/.+@sha256:[a-f0-9]{64}$")


def deploy(
    *,
    model_uri: str | None,
    tracking_uri: str | None,
    skip_smoke_test: bool,
) -> str:
    """Provision, build, deploy by digest, and verify the API."""
    _require_commands("docker", "gcloud", "git", "terraform")
    _require_terraform_variables()
    _require_clean_worktree()
    _verify_authentication()

    _run(["terraform", f"-chdir={TERRAFORM_DIR}", "init"])
    current_image = _current_container_image()
    bootstrap_image = current_image or "null"

    # Terraform automatically loads infrastructure/terraform/terraform.tfvars.
    # Preserve the deployed digest during later bootstrap applies. Forcing null
    # unconditionally would destroy and recreate an existing Cloud Run service.
    _run(
        [
            "terraform",
            f"-chdir={TERRAFORM_DIR}",
            "apply",
            "-auto-approve",
            f"-var=container_image={bootstrap_image}",
        ]
    )

    project_id = _terraform_output("project_id")
    if not PROJECT_ID.fullmatch(project_id):
        raise ValueError(
            "Terraform output project_id is not a valid Google Cloud project ID."
        )
    _verify_project_access(project_id)

    defaults = ServingSettings.load()
    descriptor_path = export_model(
        model_uri=model_uri or defaults.model_uri,
        tracking_uri=tracking_uri or defaults.tracking_uri,
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    model_hash = str(descriptor["bundle_sha256"])[:12]
    git_commit = _capture(["git", "rev-parse", "--short=12", "HEAD"])
    image_repository = _terraform_output("artifact_registry_repository")
    image_tag = f"{image_repository}/api:git-{git_commit}-model-{model_hash}"

    registry_host = image_repository.split("/", maxsplit=1)[0]
    _run(["gcloud", "auth", "configure-docker", registry_host, "--quiet"])
    _run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--tag",
            image_tag,
            ".",
        ]
    )
    _run(["docker", "push", image_tag])

    immutable_image = _resolve_image_digest(image_tag)
    _run(
        [
            "terraform",
            f"-chdir={TERRAFORM_DIR}",
            "apply",
            "-auto-approve",
            f"-var=container_image={immutable_image}",
        ]
    )
    service_url = _terraform_output("cloud_run_url")

    if not skip_smoke_test:
        smoke_test(service_url)

    print(f"Deployed immutable image: {immutable_image}")
    print(f"Cloud Run URL: {service_url}")
    print(f"Interactive API docs: {service_url}/docs")
    return service_url


def _require_terraform_variables() -> None:
    if not TERRAFORM_VARIABLES_FILE.is_file():
        raise FileNotFoundError(
            "Missing infrastructure/terraform/terraform.tfvars. Copy "
            "terraform.tfvars.example, then set project_id and location once."
        )


def _require_clean_worktree() -> None:
    status = _capture(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    )
    if status:
        raise RuntimeError(
            "Git working tree is not clean. Commit or stash source changes "
            "before deployment so the image tag identifies its exact code."
        )


def _verify_authentication() -> None:
    account = _capture(
        [
            "gcloud",
            "auth",
            "list",
            "--filter=status:ACTIVE",
            "--format=value(account)",
        ]
    )
    if not account:
        raise RuntimeError("No active gcloud account. Run `gcloud auth login`.")
    _run(
        [
            "gcloud",
            "auth",
            "application-default",
            "print-access-token",
        ],
        capture_output=True,
    )


def _verify_project_access(project_id: str) -> None:
    _run(["gcloud", "projects", "describe", project_id], capture_output=True)


def _resolve_image_digest(image_tag: str) -> str:
    digest = _capture(
        [
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "describe",
            image_tag,
            "--format=value(image_summary.digest)",
        ]
    )
    immutable_image = f"{image_tag.rsplit(':', maxsplit=1)[0]}@{digest}"
    if not IMMUTABLE_IMAGE.fullmatch(immutable_image):
        raise RuntimeError("Artifact Registry returned an invalid image digest.")
    return immutable_image


def _current_container_image() -> str | None:
    """Read the deployed digest without failing when state predates Section E."""
    completed = _run(
        [
            "terraform",
            f"-chdir={TERRAFORM_DIR}",
            "output",
            "-json",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        outputs = json.loads(completed.stdout)
        entry = outputs.get("deployed_container_image")
        image = entry.get("value") if isinstance(entry, dict) else None
    except (AttributeError, json.JSONDecodeError) as error:
        raise RuntimeError("Terraform returned invalid output JSON.") from error
    if image is None:
        return None
    if not isinstance(image, str) or not IMMUTABLE_IMAGE.fullmatch(image):
        raise RuntimeError(
            "Terraform state contains an invalid deployed container image."
        )
    return image


def _terraform_output(name: str) -> str:
    return _capture(
        [
            "terraform",
            f"-chdir={TERRAFORM_DIR}",
            "output",
            "-raw",
            name,
        ]
    )


def _require_commands(*names: str) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Missing required commands: {', '.join(missing)}")


def _capture(command: list[str]) -> str:
    completed = _run(command, capture_output=True)
    return completed.stdout.strip()


def _run(
    command: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy the local registered MLflow candidate to Cloud Run. "
            "GCP project and location come from Terraform variables."
        )
    )
    parser.add_argument("--model-uri")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        deploy(
            model_uri=args.model_uri,
            tracking_uri=args.tracking_uri,
            skip_smoke_test=args.skip_smoke_test,
        )
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
