output "bigquery_dataset" {
  description = "Fully qualified BigQuery dataset created by Terraform."
  value       = "${var.project_id}.${google_bigquery_dataset.talent_attrition.dataset_id}"
}

output "raw_data_bucket" {
  description = "Private versioned Cloud Storage bucket for immutable raw data."
  value       = google_storage_bucket.raw_data.name
}

output "raw_data_uri" {
  description = "Cloud Storage URI loaded by the BigQuery pipeline."
  value       = "gs://${google_storage_bucket.raw_data.name}/${var.raw_object_name}"
}

output "location" {
  description = "Google Cloud region used by the project."
  value       = var.location
}

output "runtime_config_path" {
  description = "Configuration file generated for every Python pipeline command."
  value       = local_file.runtime_config.filename
}
