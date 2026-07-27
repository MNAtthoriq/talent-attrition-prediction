"""Prepare the modeling base without persisting fitted objects."""

from __future__ import annotations

import argparse
from pathlib import Path

from talent_job_change_intent_prediction.config import Settings
from talent_job_change_intent_prediction.modeling.data import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    build_split_manifest,
    load_modeling_data,
    split_modeling_data,
    write_split_manifest,
)
from talent_job_change_intent_prediction.modeling.preprocessing import (
    build_preprocessor,
    preprocessing_candidates,
)


def prepare_modeling_data(
    settings: Settings,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Load, split, and verify preprocessing without persisting fitted objects."""
    frame = load_modeling_data(settings)
    split = split_modeling_data(
        frame,
        test_size=test_size,
        random_state=random_state,
    )

    candidate_results = []
    for config in preprocessing_candidates():
        preprocessor = build_preprocessor(config)
        development_matrix = preprocessor.fit_transform(split.X_development)
        test_matrix = preprocessor.transform(split.X_test)
        if development_matrix.shape[1] != test_matrix.shape[1]:
            raise RuntimeError(
                f"{config.name} produced different development and test widths."
            )
        candidate_results.append(
            {
                "name": config.name,
                **config.to_dict(),
                "transformed_feature_count": int(development_matrix.shape[1]),
            }
        )

    manifest = build_split_manifest(
        split,
        preprocessing_candidates=candidate_results,
    )
    manifest_path = settings.reports_dir / "split_manifest.json"
    write_split_manifest(manifest_path, manifest)
    print(
        "Verified leakage-safe split and preprocessing: "
        f"{len(split.X_development):,} development rows, "
        f"{len(split.X_test):,} test rows, "
        f"{len(candidate_results)} preprocessing candidates."
    )
    print(f"Wrote split manifest to {manifest_path}")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the modeling command-line parser."""
    parser = argparse.ArgumentParser(
        prog="talent-modeling",
        description="Prepare the BigQuery modeling base for leakage-safe ML.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="Validate data, create the holdout split, and verify preprocessing.",
    )
    prepare.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Holdout fraction; default: {DEFAULT_TEST_SIZE}.",
    )
    prepare.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Reproducible split seed; default: {DEFAULT_RANDOM_STATE}.",
    )

    tune = subparsers.add_parser(
        "tune",
        help="Tune preprocessing and models on development data with Optuna.",
    )
    tune.add_argument(
        "--trials",
        type=int,
        default=24,
        help="Target total completed Optuna trials; default: 24.",
    )
    tune.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified cross-validation folds; default: 5.",
    )
    tune.add_argument(
        "--timeout-minutes",
        type=float,
        default=30.0,
        help="Stop starting new trials after this many minutes; default: 30.",
    )
    tune.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI; defaults to local SQLite.",
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="Evaluate the Optuna winner once on test data and register it.",
    )
    finalize.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI; defaults to local SQLite.",
    )
    finalize.add_argument(
        "--allow-test-reevaluation",
        action="store_true",
        help="Explicitly allow replacing an existing final test evaluation.",
    )

    export = subparsers.add_parser(
        "export-results",
        help="Package generated summaries for review.",
    )
    export.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional ZIP path; defaults inside reports/generated.",
    )
    return parser


def main() -> None:
    """Run the modeling preparation CLI."""
    args = _build_parser().parse_args()
    if args.command == "export-results":
        from talent_job_change_intent_prediction.modeling.reporting import (
            export_results,
        )

        reports_dir = _find_repository_root() / "reports" / "generated"
        output_path = args.output or (reports_dir / "section_c_results.zip")
        exported = export_results(reports_dir, output_path)
        print(f"Wrote review bundle to {exported}")
        return

    settings = Settings.load()
    if args.command == "prepare":
        prepare_modeling_data(
            settings,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    elif args.command == "tune":
        from talent_job_change_intent_prediction.modeling.training import tune_models

        tune_models(
            settings,
            target_trials=args.trials,
            cv_folds=args.cv_folds,
            timeout_minutes=args.timeout_minutes,
            tracking_uri=args.tracking_uri,
        )
    elif args.command == "finalize":
        from talent_job_change_intent_prediction.modeling.training import finalize_model

        finalize_model(
            settings,
            tracking_uri=args.tracking_uri,
            allow_test_reevaluation=args.allow_test_reevaluation,
        )


def _find_repository_root() -> Path:
    """Find the project root without requiring cloud runtime configuration."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        pyproject = directory / "pyproject.toml"
        if (
            pyproject.is_file()
            and "talent-job-change-intent-prediction"
            in pyproject.read_text(encoding="utf-8")
        ):
            return directory
    return Path.cwd()
