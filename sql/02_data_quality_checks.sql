WITH
raw_summary AS (
  SELECT COUNT(*) AS row_count
  FROM `{{project_id}}.{{dataset_id}}.{{raw_table_id}}`
),
modeling_summary AS (
  SELECT
    COUNT(*) AS row_count,
    COUNTIF(enrollee_id IS NULL) AS null_ids,
    COUNT(enrollee_id) - COUNT(DISTINCT enrollee_id) AS duplicate_ids,
    COUNTIF(target IS NULL) AS null_targets,
    COUNTIF(target NOT IN (0, 1)) AS invalid_targets,
    COUNT(DISTINCT target) AS target_classes,
    COUNTIF(
      city_development_index IS NULL
      OR city_development_index NOT BETWEEN 0 AND 1
    ) AS invalid_city_development_index,
    COUNTIF(
      training_hours IS NULL OR training_hours < 0
    ) AS invalid_training_hours,
    COUNTIF(city IS NULL OR NOT REGEXP_CONTAINS(city, r'^city_[0-9]+$'))
      AS invalid_city,
    COUNTIF(gender IS NOT NULL AND gender NOT IN ('Female', 'Male', 'Other'))
      AS invalid_gender,
    COUNTIF(
      relevant_experience IS NULL
      OR relevant_experience NOT IN (
        'Has relevant experience',
        'No relevant experience'
      )
    ) AS invalid_relevant_experience,
    COUNTIF(
      enrolled_university IS NOT NULL
      AND enrolled_university NOT IN (
        'Full time course',
        'Part time course',
        'no_enrollment'
      )
    ) AS invalid_enrolled_university,
    COUNTIF(
      education_level IS NOT NULL
      AND education_level NOT IN (
        'Graduate',
        'High School',
        'Masters',
        'Phd',
        'Primary School'
      )
    ) AS invalid_education_level,
    COUNTIF(
      major_discipline IS NOT NULL
      AND major_discipline NOT IN (
        'Arts',
        'Business Degree',
        'Humanities',
        'No Major',
        'Other',
        'STEM'
      )
    ) AS invalid_major_discipline,
    COUNTIF(
      experience IS NOT NULL
      AND NOT REGEXP_CONTAINS(experience, r'^(<1|[1-9]|1[0-9]|20|>20)$')
    ) AS invalid_experience,
    COUNTIF(
      company_size IS NOT NULL
      AND company_size NOT IN (
        '<10',
        '10/49',
        '50-99',
        '100-500',
        '500-999',
        '1000-4999',
        '5000-9999',
        '10000+'
      )
    ) AS invalid_company_size,
    COUNTIF(
      company_type IS NOT NULL
      AND company_type NOT IN (
        'Early Stage Startup',
        'Funded Startup',
        'NGO',
        'Other',
        'Public Sector',
        'Pvt Ltd'
      )
    ) AS invalid_company_type,
    COUNTIF(
      last_new_job IS NOT NULL
      AND last_new_job NOT IN ('never', '1', '2', '3', '4', '>4')
    ) AS invalid_last_new_job,
    COUNTIF(source_gcs_uri != '{{source_gcs_uri}}') AS invalid_source_uri,
    COUNTIF(source_sha256 != '{{source_sha256}}') AS invalid_source_sha256
  FROM `{{project_id}}.{{dataset_id}}.{{modeling_table_id}}`
),
duplicate_summary AS (
  SELECT
    COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(candidate)) AS duplicate_rows
  FROM
    `{{project_id}}.{{dataset_id}}.{{modeling_table_id}}` AS candidate
)
SELECT
  'expected_row_count' AS check_name,
  'ERROR' AS severity,
  ABS(row_count - {{expected_row_count}}) AS failed_rows
FROM modeling_summary

UNION ALL

SELECT
  'raw_and_modeling_row_counts_match',
  'ERROR',
  ABS(raw_summary.row_count - modeling_summary.row_count)
FROM raw_summary
CROSS JOIN modeling_summary

UNION ALL

SELECT 'enrollee_id_not_null', 'ERROR', null_ids
FROM modeling_summary

UNION ALL

SELECT 'enrollee_id_unique', 'ERROR', duplicate_ids
FROM modeling_summary

UNION ALL

SELECT 'target_not_null', 'ERROR', null_targets
FROM modeling_summary

UNION ALL

SELECT 'target_is_binary', 'ERROR', invalid_targets
FROM modeling_summary

UNION ALL

SELECT
  'both_target_classes_present',
  'ERROR',
  IF(target_classes = 2, 0, 1)
FROM modeling_summary

UNION ALL

SELECT
  'city_development_index_between_0_and_1',
  'ERROR',
  invalid_city_development_index
FROM modeling_summary

UNION ALL

SELECT
  'training_hours_nonnegative',
  'ERROR',
  invalid_training_hours
FROM modeling_summary

UNION ALL

SELECT 'no_exact_duplicate_rows', 'ERROR', duplicate_rows
FROM duplicate_summary

UNION ALL

SELECT 'city_domain', 'ERROR', invalid_city
FROM modeling_summary

UNION ALL

SELECT 'gender_domain', 'ERROR', invalid_gender
FROM modeling_summary

UNION ALL

SELECT 'relevant_experience_domain', 'ERROR', invalid_relevant_experience
FROM modeling_summary

UNION ALL

SELECT 'enrolled_university_domain', 'ERROR', invalid_enrolled_university
FROM modeling_summary

UNION ALL

SELECT 'education_level_domain', 'ERROR', invalid_education_level
FROM modeling_summary

UNION ALL

SELECT 'major_discipline_domain', 'ERROR', invalid_major_discipline
FROM modeling_summary

UNION ALL

SELECT 'experience_domain', 'ERROR', invalid_experience
FROM modeling_summary

UNION ALL

SELECT 'company_size_domain', 'ERROR', invalid_company_size
FROM modeling_summary

UNION ALL

SELECT 'company_type_domain', 'ERROR', invalid_company_type
FROM modeling_summary

UNION ALL

SELECT 'last_new_job_domain', 'ERROR', invalid_last_new_job
FROM modeling_summary

UNION ALL

SELECT 'source_gcs_uri_matches', 'ERROR', invalid_source_uri
FROM modeling_summary

UNION ALL

SELECT 'source_sha256_matches', 'ERROR', invalid_source_sha256
FROM modeling_summary

ORDER BY check_name;