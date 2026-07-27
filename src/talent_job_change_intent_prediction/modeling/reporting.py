"""Generate a zip file of results."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

RESULT_FILENAMES = (
    "split_manifest.json",
    "model_selection_summary.json",
    "final_evaluation.json",
    "plots/precision_recall_curve.png",
    "plots/roc_curve.png",
    "plots/confusion_matrix.png",
    "plots/capacity_curve.png",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_results(reports_dir: Path, output_path: Path) -> Path:
    """Package summaries and plots without data, participant IDs, or models."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    included = []
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for relative_name in RESULT_FILENAMES:
            source = reports_dir / relative_name
            if source.is_file():
                archive.write(source, arcname=relative_name)
                included.append(relative_name)
        archive.writestr(
            "CONTENTS.json",
            json.dumps({"included": included}, indent=2) + "\n",
        )
    if not included:
        raise FileNotFoundError(
            f"No generated results were found in {reports_dir}."
        )
    return output_path
