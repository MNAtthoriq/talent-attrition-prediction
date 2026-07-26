terraform {
  required_version = ">= 1.8.0, < 2.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 7.0.0"
    }

    local = {
      source  = "hashicorp/local"
      version = "= 2.5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.location
}

locals {
  runtime_config_path = abspath("${path.module}/../../.runtime/project_config.json")
  storage_bucket_name = coalesce(
    var.storage_bucket_name,
    "${var.project_id}-talent-attrition-raw",
  )
}

resource "google_project_service" "required" {
  for_each = toset([
    "bigquery.googleapis.com",
    "storage.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_storage_bucket" "raw_data" {
  project                     = var.project_id
  name                        = local.storage_bucket_name
  location                    = var.location
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = var.force_destroy_storage_bucket
  labels                      = var.resource_labels

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required]
}

resource "google_bigquery_dataset" "talent_attrition" {
  project                    = var.project_id
  dataset_id                 = var.bigquery_dataset_id
  friendly_name              = "Talent Attrition"
  description                = "Raw and modeling data for the talent attrition ML project."
  location                   = var.location
  delete_contents_on_destroy = var.delete_contents_on_destroy
  labels                     = var.resource_labels

  depends_on = [google_project_service.required]
}

resource "local_file" "runtime_config" {
  filename        = local.runtime_config_path
  file_permission = "0600"
  content = jsonencode({
    schema_version        = 2
    project_id            = var.project_id
    location              = var.location
    storage_bucket_name   = google_storage_bucket.raw_data.name
    dataset_id            = google_bigquery_dataset.talent_attrition.dataset_id
    raw_table_id          = var.raw_table_id
    modeling_table_id     = var.modeling_table_id
    kaggle_dataset_handle = var.kaggle_dataset_handle
    kaggle_train_file     = var.kaggle_train_file
    raw_object_name       = var.raw_object_name
    expected_row_count    = var.expected_row_count
    expected_size_bytes   = var.expected_size_bytes
    expected_sha256       = var.expected_sha256
    reports_dir           = var.reports_dir
    sql_dir               = var.sql_dir
  })

  depends_on = [
    google_storage_bucket.raw_data,
    google_bigquery_dataset.talent_attrition,
  ]
}