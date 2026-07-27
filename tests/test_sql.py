"""Test SQL rendering."""

from pathlib import Path

import sqlglot

from talent_job_change_intent_prediction.data.pipeline import render_sql


def test_render_sql_replaces_identifiers(tmp_path: Path, settings) -> None:
    sql_path = tmp_path / "query.sql"
    sql_path.write_text(
        "SELECT * FROM `{{project_id}}.{{dataset_id}}.{{raw_table_id}}`",
        encoding="utf-8",
    )

    result = render_sql(sql_path, settings)

    assert result == (
        "SELECT * FROM `talent-ml-123.talent_job_change_intent.raw_candidates`"
    )
    assert "{{" not in result


def test_modeling_sql_rejects_nonbinary_target_during_cast(settings) -> None:
    sql_path = Path("sql/01_create_modeling_table.sql")

    result = render_sql(sql_path, settings)

    assert "IN (0.0, 1.0)" in result
    assert "AS target" in result
    assert "AS relevant_experience" in result
    assert settings.raw_gcs_uri in result


def test_all_rendered_sql_parses_as_bigquery(settings) -> None:
    for sql_path in sorted(Path("sql").glob("*.sql")):
        rendered = render_sql(sql_path, settings)
        statements = sqlglot.parse(rendered, read="bigquery")
        assert statements, f"No SQL statement parsed from {sql_path}"
