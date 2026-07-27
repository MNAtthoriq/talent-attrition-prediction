CREATE OR REPLACE TABLE
  `{{project_id}}.{{dataset_id}}.{{modeling_table_id}}`
OPTIONS (
  description = 'Typed candidate data for job-change intent modeling and fairness auditing'
) AS
SELECT
  SAFE_CAST(NULLIF(TRIM(enrollee_id), '') AS INT64) AS enrollee_id,
  NULLIF(TRIM(city), '') AS city,
  SAFE_CAST(
    NULLIF(TRIM(city_development_index), '') AS FLOAT64
  ) AS city_development_index,
  NULLIF(TRIM(gender), '') AS gender,
  CASE NULLIF(TRIM(relevent_experience), '')
    WHEN 'Has relevent experience' THEN 'Has relevant experience'
    WHEN 'No relevent experience' THEN 'No relevant experience'
    ELSE NULLIF(TRIM(relevent_experience), '')
  END AS relevant_experience,
  NULLIF(TRIM(enrolled_university), '') AS enrolled_university,
  NULLIF(TRIM(education_level), '') AS education_level,
  NULLIF(TRIM(major_discipline), '') AS major_discipline,
  NULLIF(TRIM(experience), '') AS experience,
  NULLIF(TRIM(company_size), '') AS company_size,
  NULLIF(TRIM(company_type), '') AS company_type,
  NULLIF(TRIM(last_new_job), '') AS last_new_job,
  SAFE_CAST(NULLIF(TRIM(training_hours), '') AS INT64) AS training_hours,
  CASE
    WHEN SAFE_CAST(NULLIF(TRIM(target), '') AS FLOAT64) IN (0.0, 1.0)
      THEN CAST(
        SAFE_CAST(NULLIF(TRIM(target), '') AS FLOAT64) AS INT64
      )
  END AS target,
  '{{source_gcs_uri}}' AS source_gcs_uri,
  '{{source_sha256}}' AS source_sha256,
  CURRENT_TIMESTAMP() AS transformed_at_utc
FROM `{{project_id}}.{{dataset_id}}.{{raw_table_id}}`;
