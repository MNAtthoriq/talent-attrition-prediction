# Data Science Talent Attrition Risk Forecasting

A reproducible machine-learning system for estimating job-change intent among
data-science course participants.

> **Important:** the dataset records whether a participant is looking for a job
> change. It does not confirm that the person resigned. Throughout this project,
> “risk” means predicted job-change intent, not observed employee attrition.

## Project status

This repository is being rebuilt section by section as an ML engineering
portfolio project. The current checkpoint establishes the Python package and
reproducible development environment.

## Intended use

The future model is intended to help an HR or training-program analyst identify
groups that may benefit from human-reviewed retention follow-up.

It must not be used to automate hiring, firing, promotion, compensation, or
training-eligibility decisions.

## Quick start

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

You do not need to activate the virtual environment. `uv run` executes the
command inside the project environment automatically.

## Why uv?

`uv` manages the Python version, virtual environment, dependencies, and lock
file in one workflow:

- `pyproject.toml` declares which libraries the project needs.
- `uv.lock` records the exact resolved versions for reproducible installation.
- `uv sync` makes the local environment match those files.
- `uv add <library>` adds a dependency and updates both files safely.

Commit `pyproject.toml` and `uv.lock`. Do not commit `.venv/`.

## Repository plan

1. Project foundation with uv
2. Problem definition and data contract
3. Terraform and GCP foundation
4. BigQuery ingestion, validation, and analytical SQL
5. Reproducible EDA
6. Leakage-safe supervised modeling
7. MLflow experiment tracking
8. Evaluation, fairness, and explainability
9. DVC pipeline and data/model versioning
10. FastAPI and Docker
11. Automated tests and GitHub Actions
12. Cloud Run deployment with Terraform
13. Final README, evidence audit, and release

## EDA and project history

The detailed EDA will live in `notebooks/01_eda.ipynb` so its code and outputs
remain reproducible. Only the strongest decision-relevant charts and findings
will be summarized in this README. A separate documentation folder is not
needed.

The [original bootcamp project](https://github.com/MNAtthoriq/Data-Science-Project_HR-Analytics-Job-Change-of-Data-Scientist)
is retained as project history. It is useful evidence of progression, but it
will not replace the new repository's own validated EDA and evaluation.

## Current structure

```text
.
├── src/
│   └── talent_attrition_risk/
│       └── __init__.py
├── tests/
│   └── test_package.py
├── .python-version
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## License

This project is released under the MIT License.
