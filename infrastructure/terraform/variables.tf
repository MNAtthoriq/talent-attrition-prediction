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

variable "force_destroy_storage_bucket" {
  description = "Allow Terraform to delete a non-empty raw-data bucket during teardown."
  type        = bool
  default     = true
}

variable "bigquery_dataset_id" {
  description = "BigQuery dataset that contains the raw and modeling tables."
  type        = string
  default     = "talent_attrition"

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
    application = "talent-attrition-prediction"
    environment = "development"
    managed_by  = "terraform"
  }
}