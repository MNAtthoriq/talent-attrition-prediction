WITH candidates AS (
  SELECT *
  FROM `{{project_id}}.{{dataset_id}}.{{modeling_table_id}}`
)
SELECT
  column_name,
  missing_count,
  ROUND(100 * SAFE_DIVIDE(missing_count, total_count), 2) AS missing_percent
FROM (
  SELECT
    'city' AS column_name,
    COUNTIF(city IS NULL) AS missing_count,
    COUNT(*) AS total_count
  FROM candidates

  UNION ALL

  SELECT
    'city_development_index',
    COUNTIF(city_development_index IS NULL),
    COUNT(*)
  FROM candidates

  UNION ALL

  SELECT 'gender', COUNTIF(gender IS NULL), COUNT(*)
  FROM candidates

  UNION ALL

  SELECT
    'relevant_experience',
    COUNTIF(relevant_experience IS NULL),
    COUNT(*)
  FROM candidates

  UNION ALL

  SELECT
    'enrolled_university',
    COUNTIF(enrolled_university IS NULL),
    COUNT(*)
  FROM candidates

  UNION ALL

  SELECT
    'education_level',
    COUNTIF(education_level IS NULL),
    COUNT(*)
  FROM candidates

  UNION ALL

  SELECT
    'major_discipline',
    COUNTIF(major_discipline IS NULL),
    COUNT(*)
  FROM candidates

  UNION ALL

  SELECT 'experience', COUNTIF(experience IS NULL), COUNT(*)
  FROM candidates

  UNION ALL

  SELECT 'company_size', COUNTIF(company_size IS NULL), COUNT(*)
  FROM candidates

  UNION ALL

  SELECT 'company_type', COUNTIF(company_type IS NULL), COUNT(*)
  FROM candidates

  UNION ALL

  SELECT 'last_new_job', COUNTIF(last_new_job IS NULL), COUNT(*)
  FROM candidates

  UNION ALL

  SELECT
    'training_hours',
    COUNTIF(training_hours IS NULL),
    COUNT(*)
  FROM candidates

  UNION ALL

  SELECT 'target', COUNTIF(target IS NULL), COUNT(*)
  FROM candidates
)
ORDER BY missing_percent DESC, column_name;