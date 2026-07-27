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
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
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

resource "google_artifact_registry_repository" "api" {
  project       = var.project_id
  location      = var.location
  repository_id = var.artifact_repository_id
  description   = "Versioned container images for the talent attrition API."
  format        = "DOCKER"
  labels        = var.resource_labels

  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-recent-versions"
    action = "KEEP"

    most_recent_versions {
      keep_count = var.container_versions_to_keep
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"

    condition {
      tag_state  = "UNTAGGED"
      older_than = "2592000s"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_service_account" "cloud_run" {
  project      = var.project_id
  account_id   = var.cloud_run_service_account_id
  display_name = "Talent attrition Cloud Run runtime"
  description  = "Least-privilege identity for the talent attrition prediction API."

  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_service" "api" {
  count = var.container_image == null ? 0 : 1

  project              = var.project_id
  name                 = var.cloud_run_service_name
  location             = var.location
  deletion_protection  = var.cloud_run_deletion_protection
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = var.allow_unauthenticated
  labels               = var.resource_labels

  lifecycle {
    precondition {
      condition     = var.min_instances <= var.max_instances
      error_message = "min_instances cannot be greater than max_instances."
    }
  }

  template {
    service_account                  = google_service_account.cloud_run.email
    max_instance_request_concurrency = var.container_concurrency
    timeout                          = "${var.request_timeout_seconds}s"

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      name  = "api"
      image = var.container_image

      ports {
        name           = "http1"
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.container_cpu
          memory = var.container_memory
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        initial_delay_seconds = 0
        timeout_seconds       = 1
        period_seconds        = 5
        failure_threshold     = 24

        tcp_socket {
          port = 8080
        }
      }
    }
  }

  depends_on = [
    google_artifact_registry_repository.api,
    google_project_service.required,
  ]
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
