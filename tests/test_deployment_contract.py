"""Test portable model metadata and container deployment contracts."""

from __future__ import annotations

import json
import shutil
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from talent_attrition_prediction.deployment.deploy import (
    _current_container_image,
    _require_clean_worktree,
    _resolve_image_digest,
)
from talent_attrition_prediction.deployment.export_model import (
    _assert_safe_destination,
)
from talent_attrition_prediction.serving.service import (
    ModelDescriptor,
    _load_exported_descriptor,
)


def test_exported_descriptor_restores_model_lineage(tmp_path: Path) -> None:
    path = tmp_path / "model-descriptor.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_sha256": "a" * 64,
                "model": {
                    "model_uri": "models:/talent-attrition-classifier@candidate",
                    "model_name": "talent-attrition-classifier",
                    "model_version": "3",
                    "model_alias": "candidate",
                    "run_id": "run-123",
                    "git_commit": "abc123",
                    "source_sha256": "b" * 64,
                    "optuna_study": "study-123",
                    "optuna_trial": 7,
                    "preprocessing": "ordinal_aware_without_city",
                    "parameters": {"model_name": "lightgbm"},
                    "test_metrics": {"test_average_precision": 0.55},
                },
            }
        ),
        encoding="utf-8",
    )

    descriptor = _load_exported_descriptor(path)

    assert descriptor.model_version == "3"
    assert descriptor.optuna_trial == 7
    assert descriptor.parameters == {"model_name": "lightgbm"}
    assert descriptor.test_metrics == {"test_average_precision": 0.55}


def test_exported_descriptor_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "model-descriptor.json"
    path.write_text(
        json.dumps({"schema_version": 2, "model": {}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Unsupported"):
        _load_exported_descriptor(path)


def test_container_uses_cloud_run_and_non_root_contracts() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "TALENT_API_HOST=0.0.0.0" in dockerfile
    assert "PORT=8080" in dockerfile
    assert "USER appuser" in dockerfile
    assert "libgomp1" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert ".deployment/model" in dockerfile
    assert "!.deployment/model/**" in dockerignore


def test_generated_deployment_bundle_is_gitignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "/.deployment/" in gitignore


def test_pyproject_exposes_deployment_commands() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert (
        "talent-export-model = "
        '"talent_attrition_prediction.deployment.export_model:main"'
    ) in pyproject
    assert (
        'talent-deploy = "talent_attrition_prediction.deployment.deploy:main"'
    ) in pyproject
    assert (
        'talent-smoke-test = "talent_attrition_prediction.deployment.smoke_test:main"'
    ) in pyproject


def test_deploy_uses_terraform_variables_as_gcp_source_of_truth() -> None:
    source = Path("src/talent_attrition_prediction/deployment/deploy.py").read_text(
        encoding="utf-8"
    )

    assert "TERRAFORM_VARIABLES_FILE" in source
    assert 'parser.add_argument("--project-id"' not in source
    assert 'parser.add_argument("--location"' not in source
    assert 'f"-var=project_id=' not in source
    assert 'f"-var=location=' not in source


def test_existing_cloud_run_digest_is_preserved_during_bootstrap() -> None:
    digest = (
        "asia-southeast2-docker.pkg.dev/example-project/repository/api@sha256:"
        + "a" * 64
    )
    terraform_output = json.dumps({"deployed_container_image": {"value": digest}})

    with patch("talent_attrition_prediction.deployment.deploy._run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = terraform_output
        assert _current_container_image() == digest


def test_missing_legacy_terraform_output_is_safe_for_first_deploy() -> None:
    with patch("talent_attrition_prediction.deployment.deploy._run") as run:
        run.return_value.returncode = 1
        run.return_value.stdout = ""
        assert _current_container_image() is None


def test_deploy_rejects_uncommitted_source() -> None:
    with (
        patch(
            "talent_attrition_prediction.deployment.deploy._capture",
            return_value=" M src/example.py",
        ),
        pytest.raises(RuntimeError, match="not clean"),
    ):
        _require_clean_worktree()


def test_image_tag_is_resolved_with_official_describe_command() -> None:
    tag = (
        "asia-southeast2-docker.pkg.dev/example-project/repository/api:"
        "git-123-model-456"
    )
    digest = "sha256:" + "b" * 64

    with patch(
        "talent_attrition_prediction.deployment.deploy._capture",
        return_value=digest,
    ) as capture:
        assert _resolve_image_digest(tag) == f"{tag.rsplit(':', 1)[0]}@{digest}"
        assert "describe" in capture.call_args.args[0]


def test_model_export_refuses_directory_with_unrelated_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "source-code"
    destination.mkdir()
    precious_file = destination / "important.py"
    precious_file.write_text("do_not_delete = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unrelated files"):
        _assert_safe_destination(destination)

    assert precious_file.read_text(encoding="utf-8") == "do_not_delete = True\n"


def test_model_download_directory_is_separate_from_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_module = import_module("talent_attrition_prediction.deployment.export_model")
    monkeypatch.setattr(export_module, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        export_module.ModelService,
        "load",
        lambda **_: SimpleNamespace(
            descriptor=ModelDescriptor(model_uri="models:/example@candidate")
        ),
    )

    def fake_download(*, artifact_uri: str, dst_path: str) -> str:
        assert artifact_uri == "models:/example@candidate"
        downloaded = Path(dst_path)

        class ConstantModel(export_module.mlflow.pyfunc.PythonModel):
            def predict(
                self,
                context,
                model_input: list[dict[str, float]],
                params=None,
            ) -> list[float]:
                return [0.5] * len(model_input)

        shutil.rmtree(downloaded)
        export_module.mlflow.pyfunc.save_model(
            path=downloaded,
            python_model=ConstantModel(),
        )
        return str(downloaded)

    monkeypatch.setattr(
        export_module.mlflow.artifacts,
        "download_artifacts",
        fake_download,
    )
    descriptor = export_module.export_model(
        model_uri="models:/example@candidate",
        tracking_uri="sqlite:////tmp/example.db",
        output_dir=tmp_path / "bundle",
    )

    assert descriptor.is_file()
    assert (tmp_path / "bundle" / "model" / "MLmodel").is_file()
