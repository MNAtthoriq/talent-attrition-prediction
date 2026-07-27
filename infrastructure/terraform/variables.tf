variable "project_id" {
  description = "Existing Google Cloud project ID with billing enabled."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid 6-30 character Google Cloud project ID."
  }
}

variable "location" {
  description = "Shared regional location for Cloud Storage and BigQuery."
  type        = string
  default     = "asia-southeast2"

  validation {
    condition     = can(regex("^[A-Za-z0-9-]+$", var.location))
    error_message = "location may contain only letters, numbers, and hyphens."
  }
}

variable "storage_bucket_name" {
  description = "Optional globally unique raw-data bucket name. Null derives one from project_id."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition = (
      var.storage_bucket_name == null
      || can(regex("^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$", var.storage_bucket_name))
    )
    error_message = "storage_bucket_name must be null or a valid 3-63 character Cloud Storage bucket name."
  }
}

variable "artifact_repository_id" {
  description = "Artifact Registry Docker repository for API images."
  type        = string
  default     = "talent-job-change-intent"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,62}$", var.artifact_repository_id))
    error_message = "artifact_repository_id must contain 3-63 lowercase letters, numbers, or hyphens."
  }
}

variable "container_versions_to_keep" {
  description = "Number of recent container image versions retained by cleanup policy."
  type        = number
  default     = 10

  validation {
    condition     = var.container_versions_to_keep >= 1 && floor(var.container_versions_to_keep) == var.container_versions_to_keep
    error_message = "container_versions_to_keep must be a positive integer."
  }
}

variable "cloud_run_service_account_id" {
  description = "Account ID for the dedicated Cloud Run runtime identity."
  type        = string
  default     = "talent-job-change-intent-api"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.cloud_run_service_account_id))
    error_message = "cloud_run_service_account_id must be a valid 6-30 character service-account ID."
  }
}

variable "cloud_run_service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "talent-job-change-intent-api"

  validation {
    condition     = can(regex("^[a-z]([a-z0-9-]{0,47}[a-z0-9])?$", var.cloud_run_service_name))
    error_message = "cloud_run_service_name must be 1-49 lowercase letters, numbers, or hyphens; it must start with a letter and cannot end with a hyphen."
  }
}

variable "container_image" {
  description = "Immutable Artifact Registry image URI including sha256 digest. Null provisions only bootstrap infrastructure."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition = (
      var.container_image == null
      || can(regex("^.+-docker\\.pkg\\.dev/.+@sha256:[a-f0-9]{64}$", var.container_image))
    )
    error_message = "container_image must be null or an Artifact Registry URI pinned by sha256 digest."
  }
}

variable "allow_unauthenticated" {
  description = "Disable the Cloud Run Invoker IAM check for this public portfolio API."
  type        = bool
  default     = true
}

variable "cloud_run_deletion_protection" {
  description = "Prevent Terraform from deleting the Cloud Run service."
  type        = bool
  default     = false
}

variable "min_instances" {
  description = "Minimum warm Cloud Run instances. Zero avoids idle compute cost."
  type        = number
  default     = 0

  validation {
    condition     = var.min_instances >= 0 && floor(var.min_instances) == var.min_instances
    error_message = "min_instances must be a non-negative integer."
  }
}

variable "max_instances" {
  description = "Maximum Cloud Run instances, limiting portfolio-demo cost exposure."
  type        = number
  default     = 2

  validation {
    condition     = var.max_instances >= 1 && floor(var.max_instances) == var.max_instances
    error_message = "max_instances must be a positive integer."
  }
}

variable "container_concurrency" {
  description = "Maximum concurrent requests handled by each API instance."
  type        = number
  default     = 20

  validation {
    condition     = var.container_concurrency >= 1 && var.container_concurrency <= 1000
    error_message = "container_concurrency must be between 1 and 1000."
  }
}

variable "request_timeout_seconds" {
  description = "Maximum Cloud Run request duration."
  type        = number
  default     = 60

  validation {
    condition     = var.request_timeout_seconds >= 1 && var.request_timeout_seconds <= 3600
    error_message = "request_timeout_seconds must be between 1 and 3600."
  }
}

variable "container_cpu" {
  description = "CPU limit for each Cloud Run instance."
  type        = string
  default     = "1"
}

variable "container_memory" {
  description = "Memory limit for each Cloud Run instance."
  type        = string
  default     = "2Gi"

  validation {
    condition     = can(regex("^[1-9][0-9]*(Mi|Gi)$", var.container_memory))
    error_message = "container_memory must use a positive Mi or Gi quantity, such as 2Gi."
  }
}

variable "force_destroy_storage_bucket" {
  description = "Allow Terraform to delete a non-empty raw-data bucket during teardown."
  type        = bool
  default     = true
}

variable "bigquery_dataset_id" {
  description = "BigQuery dataset that contains the raw and modeling tables."
  type        = string
  default     = "talent_job_change_intent"

  validation {
    condition     = can(regex("^[A-Za-z_][A-Za-z0-9_]{0,1000}$", var.bigquery_dataset_id))
    error_message = "bigquery_dataset_id must be a valid BigQuery identifier."
  }
}

variable "raw_table_id" {
  description = "Pipeline-managed BigQuery table containing unchanged source values."
  type        = string
  default     = "raw_candidates"

  validation {
    condition     = can(regex("^[A-Za-z_][A-Za-z0-9_]{0,1000}$", var.raw_table_id))
    error_message = "raw_table_id must be a valid BigQuery identifier."
  }
}

variable "modeling_table_id" {
  description = "Pipeline-managed BigQuery table containing typed modeling data."
  type        = string
  default     = "modeling_candidates"

  validation {
    condition     = can(regex("^[A-Za-z_][A-Za-z0-9_]{0,1000}$", var.modeling_table_id))
    error_message = "modeling_table_id must be a valid BigQuery identifier."
  }
}

variable "delete_contents_on_destroy" {
  description = "Allow Terraform to delete non-empty BigQuery datasets during teardown."
  type        = bool
  default     = true
}

variable "kaggle_dataset_handle" {
  description = "Pinned Kaggle dataset handle, including its immutable version."
  type        = string
  default     = "arashnic/hr-analytics-job-change-of-data-scientists/versions/1"

  validation {
    condition     = length(trimspace(var.kaggle_dataset_handle)) > 0
    error_message = "kaggle_dataset_handle cannot be blank."
  }
}

variable "kaggle_train_file" {
  description = "Training CSV filename inside the pinned Kaggle dataset."
  type        = string
  default     = "aug_train.csv"

  validation {
    condition     = can(regex("^[A-Za-z0-9._-]+$", var.kaggle_train_file))
    error_message = "kaggle_train_file must be a safe filename without directory components."
  }
}

variable "raw_object_name" {
  description = "Versioned Cloud Storage object path for the immutable raw CSV."
  type        = string
  default     = "raw/kaggle/v1/aug_train.csv"

  validation {
    condition = (
      length(trimspace(var.raw_object_name)) > 0
      && !startswith(var.raw_object_name, "/")
      && !can(regex("(^|/)\\.\\.(/|$)", var.raw_object_name))
      && can(regex("^[A-Za-z0-9._/-]+$", var.raw_object_name))
    )
    error_message = "raw_object_name must be a safe relative object path."
  }
}

variable "expected_row_count" {
  description = "Expected data-row count for the pinned raw CSV."
  type        = number
  default     = 19158

  validation {
    condition     = var.expected_row_count > 0 && floor(var.expected_row_count) == var.expected_row_count
    error_message = "expected_row_count must be a positive integer."
  }
}

variable "expected_size_bytes" {
  description = "Expected byte size for the pinned raw CSV."
  type        = number
  default     = 1961145

  validation {
    condition     = var.expected_size_bytes > 0 && floor(var.expected_size_bytes) == var.expected_size_bytes
    error_message = "expected_size_bytes must be a positive integer."
  }
}

variable "expected_sha256" {
  description = "Expected lowercase SHA-256 checksum for the pinned raw CSV."
  type        = string
  default     = "8b78da3482032500df40d5359c36ba4a59e28ccd6fc272b050dcf6dee37f114c"

  validation {
    condition     = can(regex("^[a-f0-9]{64}$", var.expected_sha256))
    error_message = "expected_sha256 must contain exactly 64 lowercase hexadecimal characters."
  }
}

variable "reports_dir" {
  description = "Repository-relative directory for generated validation reports."
  type        = string
  default     = "reports/generated"

  validation {
    condition     = !startswith(var.reports_dir, "/") && !can(regex("(^|/)\\.\\.(/|$)", var.reports_dir))
    error_message = "reports_dir must be a repository-relative path without '..'."
  }
}

variable "sql_dir" {
  description = "Repository-relative directory containing BigQuery SQL."
  type        = string
  default     = "sql"

  validation {
    condition     = !startswith(var.sql_dir, "/") && !can(regex("(^|/)\\.\\.(/|$)", var.sql_dir))
    error_message = "sql_dir must be a repository-relative path without '..'."
  }
}

variable "resource_labels" {
  description = "Labels applied to supported Google Cloud resources."
  type        = map(string)
  default = {
    application = "talent-job-change-intent-prediction"
    environment = "development"
    managed_by  = "terraform"
  }
}
