"""Repository paths shared by deployment commands."""

from __future__ import annotations

from pathlib import Path


def find_repository_root() -> Path:
    """Find the project checkout containing Terraform and pyproject.toml."""
    starts = (Path.cwd().resolve(), Path(__file__).resolve())
    checked: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            if directory in checked:
                continue
            checked.add(directory)
            if (directory / "pyproject.toml").is_file() and (
                directory / "infrastructure" / "terraform"
            ).is_dir():
                return directory
    raise RuntimeError(
        "Could not find the repository root. Run this command from the "
        "talent-job-change-intent-prediction checkout."
    )


REPOSITORY_ROOT = find_repository_root()
TERRAFORM_DIR = REPOSITORY_ROOT / "infrastructure" / "terraform"
TERRAFORM_VARIABLES_FILE = TERRAFORM_DIR / "terraform.tfvars"
