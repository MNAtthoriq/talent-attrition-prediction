output "bigquery_dataset" {
  description = "Fully qualified BigQuery dataset created by Terraform."
  value       = "${var.project_id}.${google_bigquery_dataset.talent_attrition.dataset_id}"
}

output "project_id" {
  description = "Google Cloud project configured in terraform.tfvars."
  value       = var.project_id
}

output "raw_data_bucket" {
  description = "Private versioned Cloud Storage bucket for immutable raw data."
  value       = google_storage_bucket.raw_data.name
}

output "artifact_registry_repository" {
  description = "Artifact Registry Docker repository used for API images."
  value       = "${var.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.api.repository_id}"
}

output "cloud_run_service_account" {
  description = "Dedicated runtime identity used by Cloud Run."
  value       = google_service_account.cloud_run.email
}

output "cloud_run_service_name" {
  description = "Cloud Run service name configured in terraform.tfvars."
  value       = var.cloud_run_service_name
}

output "cloud_run_url" {
  description = "Public Cloud Run service URL; null before a container image is supplied."
  value       = try(google_cloud_run_v2_service.api[0].uri, null)
}

output "deployed_container_image" {
  description = "Immutable image digest deployed to Cloud Run."
  value       = var.container_image
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
