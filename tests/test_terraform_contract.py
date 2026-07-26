"""Test the Terraform contracts."""

import re
from pathlib import Path

import hcl2

TERRAFORM_DIR = Path("infrastructure/terraform")


def test_all_terraform_files_parse() -> None:
    for terraform_path in sorted(TERRAFORM_DIR.glob("*.tf")):
        with terraform_path.open(encoding="utf-8") as file:
            assert hcl2.load(file), f"No HCL parsed from {terraform_path}"


def test_tfvars_example_exposes_every_declared_project_setting() -> None:
    tfvars = (TERRAFORM_DIR / "terraform.tfvars.example").read_text(encoding="utf-8")
    variables_tf = (TERRAFORM_DIR / "variables.tf").read_text(encoding="utf-8")

    declared_variables = set(
        re.findall(r'^variable "([^"]+)"', variables_tf, re.MULTILINE)
    )
    example_variables = set(re.findall(r"^([a-z0-9_]+)\s*=", tfvars, re.MULTILINE))

    assert example_variables == declared_variables


def test_python_runtime_config_is_generated_from_terraform_variables() -> None:
    main_tf = (TERRAFORM_DIR / "main.tf").read_text(encoding="utf-8")
    config_source = Path("src/talent_attrition_prediction/config.py").read_text(
        encoding="utf-8"
    )
    generated_block = main_tf.split("content = jsonencode({", maxsplit=1)[1].split(
        "})", maxsplit=1
    )[0]
    generated_keys = set(
        re.findall(r"^\s+([a-z0-9_]+)\s*=", generated_block, re.MULTILINE)
    )
    expected_keys = {
        "schema_version",
        "project_id",
        "location",
        "storage_bucket_name",
        "dataset_id",
        "raw_table_id",
        "modeling_table_id",
        "kaggle_dataset_handle",
        "kaggle_train_file",
        "raw_object_name",
        "expected_row_count",
        "expected_size_bytes",
        "expected_sha256",
        "reports_dir",
        "sql_dir",
    }

    assert 'resource "local_file" "runtime_config"' in main_tf
    assert "jsonencode" in main_tf
    assert generated_keys == expected_keys
    for key in expected_keys - {"schema_version"}:
        assert f'"{key}"' in config_source


def test_terraform_provisions_secure_gcs_and_bigquery_foundation() -> None:
    main_tf = (TERRAFORM_DIR / "main.tf").read_text(encoding="utf-8")

    assert '"storage.googleapis.com"' in main_tf
    assert '"bigquery.googleapis.com"' in main_tf
    assert 'resource "google_storage_bucket" "raw_data"' in main_tf
    assert 'public_access_prevention    = "enforced"' in main_tf
    assert "uniform_bucket_level_access = true" in main_tf
    assert "versioning {" in main_tf
    assert 'resource "google_bigquery_dataset" "talent_attrition"' in main_tf
