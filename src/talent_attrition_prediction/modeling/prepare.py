"""Verify the leakage-safe split and preprocessing contract against BigQuery."""

from __future__ import annotations

import argparse

from talent_attrition_prediction.config import Settings
from talent_attrition_prediction.modeling.data import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    build_split_manifest,
    load_modeling_data,
    split_modeling_data,
    write_split_manifest,
)
from talent_attrition_prediction.modeling.preprocessing import (
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
    return parser


def main() -> None:
    """Run the modeling preparation CLI."""
    args = _build_parser().parse_args()
    settings = Settings.load()
    if args.command == "prepare":
        prepare_modeling_data(
            settings,
            test_size=args.test_size,
            random_state=args.random_state,
        )
