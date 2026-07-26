"""Tune, evaluate, and register model with Optuna and MLflow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from mlflow import MlflowClient
from mlflow.models import infer_signature
from optuna.trial import TrialState
from sklearn.dummy import DummyClassifier

from talent_attrition_prediction.config import Settings
from talent_attrition_prediction.modeling.artifacts import (
    create_evaluation_plots,
)
from talent_attrition_prediction.modeling.data import (
    build_split_manifest,
    load_modeling_data,
    split_modeling_data,
)
from talent_attrition_prediction.modeling.evaluation import (
    evaluate_cross_validation,
    evaluate_holdout,
    fairness_by_group,
    flat_mlflow_metrics,
)
from talent_attrition_prediction.modeling.models import (
    build_pipeline_from_params,
    initial_search_candidates,
    suggest_pipeline,
)
from talent_attrition_prediction.modeling.preprocessing import (
    PreprocessingConfig,
    build_model_pipeline,
    preprocessing_candidates,
)
from talent_attrition_prediction.modeling.reporting import write_json
from talent_attrition_prediction.modeling.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    configure_tracking,
    repository_root_from_config,
)

_TOP_TRIALS_TO_REPORT = 10
_SEARCH_SPACE_VERSION = 1


def tune_models(
    settings: Settings,
    *,
    target_trials: int = 24,
    cv_folds: int = 5,
    timeout_minutes: float = 30.0,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Tune only on development data and leave test performance untouched."""
    _validate_tuning_inputs(target_trials, cv_folds, timeout_minutes)
    split = split_modeling_data(load_modeling_data(settings))
    repository_root = repository_root_from_config(settings.config_path)
    tracking = configure_tracking(
        repository_root,
        tracking_uri=tracking_uri,
    )
    study_name = _tuning_study_name(
        settings.reports_dir / "model_selection_summary.json",
        source_sha256=split.source_sha256,
        cv_folds=cv_folds,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=tracking.optuna_storage_uri,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=split.random_state),
    )
    _set_or_validate_study_contract(
        study,
        source_sha256=split.source_sha256,
        cv_folds=cv_folds,
        random_state=split.random_state,
    )
    if not study.trials:
        for candidate in initial_search_candidates():
            study.enqueue_trial(candidate)

    manifest = build_split_manifest(
        split,
        preprocessing_candidates=[
            candidate.to_dict() | {"name": candidate.name}
            for candidate in preprocessing_candidates()
        ],
    )
    write_json(settings.reports_dir / "split_manifest.json", manifest)
    completed_before = sum(trial.state == TrialState.COMPLETE for trial in study.trials)
    remaining_trials = max(0, target_trials - completed_before)

    with mlflow.start_run(
        run_name=f"tuning-{_timestamp()}",
        tags={
            "phase": "model_selection",
            "study_name": study_name,
            "source_sha256": split.source_sha256,
            "git_commit": tracking.git_commit,
        },
    ) as parent_run:
        mlflow.log_params(
            {
                "target_total_trials": target_trials,
                "completed_trials_before_run": completed_before,
                "cv_folds": cv_folds,
                "random_state": split.random_state,
                "primary_metric": "validation_average_precision_mean",
                "test_set_used": False,
            }
        )
        mlflow.log_dict(manifest, "data/split_manifest.json")

        dummy_result = _evaluate_dummy(split, cv_folds)
        with mlflow.start_run(
            run_name="sanity-baseline-dummy",
            nested=True,
            tags={"phase": "sanity_baseline", "model_name": "dummy_prior"},
        ):
            mlflow.log_params(
                {
                    "model_name": "dummy_prior",
                    "preprocessing": "all_ohe_without_city",
                }
            )
            mlflow.log_metrics(flat_mlflow_metrics(dummy_result))
            mlflow.log_dict(dummy_result, "evaluation/cv_result.json")

        def objective(trial: optuna.Trial) -> float:
            pipeline, config, model_name = suggest_pipeline(
                trial,
                random_state=split.random_state,
            )
            with mlflow.start_run(
                run_name=f"trial-{trial.number:03d}-{model_name}",
                nested=True,
                tags={
                    "phase": "optuna_trial",
                    "study_name": study_name,
                    "optuna_trial_number": str(trial.number),
                    "model_name": model_name,
                    "preprocessing": config.name,
                    "source_sha256": split.source_sha256,
                },
            ) as trial_run:
                result = evaluate_cross_validation(
                    pipeline,
                    split.X_development,
                    split.y_development,
                    cv_folds=cv_folds,
                    random_state=split.random_state,
                )
                trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
                for key, value in flat_mlflow_metrics(result).items():
                    trial.set_user_attr(key, value)
                mlflow.log_params(_mlflow_params(trial.params))
                mlflow.log_metrics(flat_mlflow_metrics(result))
                mlflow.log_dict(result, "evaluation/cv_result.json")
                return float(result["validation_average_precision_mean"])

        if remaining_trials:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=timeout_minutes * 60,
                n_jobs=1,
            )

        summary = _selection_summary(
            study,
            manifest,
            dummy_result,
            parent_run_id=parent_run.info.run_id,
            tracking_uri=tracking.tracking_uri,
            git_commit=tracking.git_commit,
        )
        mlflow.log_metrics(
            {
                "best_validation_average_precision": float(
                    summary["best_trial"]["validation_average_precision_mean"]
                ),
                "dummy_validation_average_precision": float(
                    dummy_result["validation_average_precision_mean"]
                ),
            }
        )
        mlflow.log_dict(summary, "selection/model_selection_summary.json")

    output_path = settings.reports_dir / "model_selection_summary.json"
    write_json(output_path, summary)
    print(
        f"Best development CV average precision: "
        f"{summary['best_trial']['validation_average_precision_mean']:.4f}"
    )
    print(f"Model: {summary['best_trial']['params']['model_name']}")
    print(f"Wrote selection summary to {output_path}")
    print("The untouched test set has not been evaluated.")
    return summary


def finalize_model(
    settings: Settings,
    *,
    tracking_uri: str | None = None,
    allow_test_reevaluation: bool = False,
) -> dict[str, Any]:
    """Evaluate the selected pipeline once and register it as a candidate."""
    output_path = settings.reports_dir / "final_evaluation.json"
    if output_path.exists() and not allow_test_reevaluation:
        raise FileExistsError(
            f"Final test evaluation already exists at {output_path}. "
            "Do not reuse the test set for iterative model selection. "
            "Pass --allow-test-reevaluation only for an intentional rerun."
        )

    split = split_modeling_data(load_modeling_data(settings))
    repository_root = repository_root_from_config(settings.config_path)
    tracking = configure_tracking(
        repository_root,
        tracking_uri=tracking_uri,
    )
    study_name = _selected_study_name(
        settings.reports_dir / "model_selection_summary.json",
        source_sha256=split.source_sha256,
    )
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=tracking.optuna_storage_uri,
        )
    except KeyError as error:
        raise RuntimeError(
            "No completed Optuna study exists. Run `talent-modeling tune` first."
        ) from error
    if not any(trial.state == TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError("The Optuna study has no completed trials.")

    best_trial = study.best_trial
    threshold = float(best_trial.user_attrs["oof_threshold"])
    pipeline, config, model_name = build_pipeline_from_params(
        best_trial.params,
        random_state=split.random_state,
    )
    pipeline.fit(split.X_development, split.y_development)
    test_probability = np.asarray(pipeline.predict_proba(split.X_test))[:, 1]
    test_result = evaluate_holdout(
        split.y_test,
        test_probability,
        threshold=threshold,
    )
    fairness_result = fairness_by_group(
        split.audit_test,
        split.y_test,
        test_probability,
        group_column="gender",
        threshold=threshold,
    )
    plots = create_evaluation_plots(
        settings.reports_dir / "plots",
        split.y_test,
        test_probability,
        threshold=threshold,
    )
    manifest = build_split_manifest(split)

    with mlflow.start_run(
        run_name=f"final-candidate-{_timestamp()}",
        tags={
            "phase": "final_evaluation",
            "study_name": study_name,
            "source_sha256": split.source_sha256,
            "git_commit": tracking.git_commit,
            "model_name": model_name,
            "preprocessing": config.name,
            "test_set_used": "true",
        },
    ) as run:
        mlflow.log_params(_mlflow_params(best_trial.params))
        mlflow.log_params(
            {
                "selected_optuna_trial": best_trial.number,
                "threshold_source": "development_oof_max_f2",
                "registered_model_name": DEFAULT_REGISTERED_MODEL_NAME,
            }
        )
        mlflow.log_metrics(
            {
                f"test_{key}": value
                for key, value in flat_mlflow_metrics(test_result).items()
            }
        )
        mlflow.log_dict(manifest, "data/split_manifest.json")
        mlflow.log_dict(test_result, "evaluation/final_evaluation.json")
        mlflow.log_dict(fairness_result, "evaluation/fairness_by_gender.json")
        for plot in plots:
            mlflow.log_artifact(str(plot), artifact_path="evaluation/plots")

        input_example = split.X_development.head(5)
        signature = infer_signature(
            input_example,
            pipeline.predict_proba(input_example),
        )
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
            pyfunc_predict_fn="predict_proba",
            metadata={
                "source_sha256": split.source_sha256,
                "optuna_study": study_name,
                "optuna_trial": best_trial.number,
            },
        )
        model_version = _registered_version(
            run.info.run_id,
            model_info,
            DEFAULT_REGISTERED_MODEL_NAME,
        )
        client = MlflowClient()
        client.set_registered_model_alias(
            DEFAULT_REGISTERED_MODEL_NAME,
            "candidate",
            model_version,
        )

        final_summary = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_sha256": split.source_sha256,
            "git_commit": tracking.git_commit,
            "study_name": study_name,
            "selected_optuna_trial": best_trial.number,
            "model_name": model_name,
            "preprocessing": config.to_dict() | {"name": config.name},
            "best_params": best_trial.params,
            "development_cv": {
                key: value
                for key, value in best_trial.user_attrs.items()
                if key != "mlflow_run_id"
            },
            "test": test_result,
            "fairness_by_gender": fairness_result,
            "mlflow": {
                "tracking_uri": tracking.tracking_uri,
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "registered_model_name": DEFAULT_REGISTERED_MODEL_NAME,
                "registered_model_version": str(model_version),
                "alias": "candidate",
                "load_uri": (f"models:/{DEFAULT_REGISTERED_MODEL_NAME}@candidate"),
            },
        }
        mlflow.log_dict(final_summary, "evaluation/final_summary.json")

    write_json(output_path, final_summary)
    print(f"Final test average precision: {test_result['average_precision']:.4f}")
    print(
        f"Registered {DEFAULT_REGISTERED_MODEL_NAME} version "
        f"{model_version} with alias @candidate."
    )
    print(f"Wrote final evaluation to {output_path}")
    return final_summary


def _evaluate_dummy(split, cv_folds: int) -> dict[str, Any]:
    pipeline = build_model_pipeline(
        DummyClassifier(strategy="prior", random_state=split.random_state),
        config=PreprocessingConfig(),
    )
    return evaluate_cross_validation(
        pipeline,
        split.X_development,
        split.y_development,
        cv_folds=cv_folds,
        random_state=split.random_state,
    )


def _selection_summary(
    study: optuna.Study,
    manifest: dict[str, Any],
    dummy_result: dict[str, Any],
    *,
    parent_run_id: str,
    tracking_uri: str,
    git_commit: str,
) -> dict[str, Any]:
    completed = sorted(
        (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
        key=lambda trial: float(trial.value),
        reverse=True,
    )
    if not completed:
        raise RuntimeError("Optuna did not complete any trials.")

    def summarize_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
        return {
            "number": trial.number,
            "validation_average_precision_mean": float(trial.value),
            "params": trial.params,
            "metrics": trial.user_attrs,
        }

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_rule": (
            "Highest mean stratified-CV average precision on development data. "
            "Recall, variability, and train-validation gap are review diagnostics."
        ),
        "test_set_used": False,
        "source_sha256": manifest["source_sha256"],
        "git_commit": git_commit,
        "study_name": study.study_name,
        "completed_trials": len(completed),
        "dummy_baseline": dummy_result,
        "best_trial": summarize_trial(completed[0]),
        "top_trials": [
            summarize_trial(trial) for trial in completed[:_TOP_TRIALS_TO_REPORT]
        ],
        "mlflow": {
            "tracking_uri": tracking_uri,
            "parent_run_id": parent_run_id,
        },
    }


def _registered_version(
    run_id: str,
    model_info,
    registered_model_name: str,
) -> str:
    version = getattr(model_info, "registered_model_version", None)
    if version is not None:
        return str(version)
    versions = MlflowClient().search_model_versions(f"name = '{registered_model_name}'")
    matches = [item for item in versions if item.run_id == run_id]
    if not matches:
        raise RuntimeError("MLflow registered the model but no version was found.")
    return str(max(matches, key=lambda item: int(item.version)).version)


def _mlflow_params(params: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {key: "none" if value is None else value for key, value in params.items()}


def _study_name(source_sha256: str, cv_folds: int) -> str:
    """Isolate scores that use different data, CV, or search-space contracts."""
    return (
        f"talent-attrition-v{_SEARCH_SPACE_VERSION}-{source_sha256[:12]}-cv{cv_folds}"
    )


def _tuning_study_name(
    summary_path: Path,
    *,
    source_sha256: str,
    cv_folds: int,
) -> str:
    """Resume a compatible legacy study or create an isolated current study."""
    if not summary_path.is_file():
        return _study_name(source_sha256, cv_folds)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _study_name(source_sha256, cv_folds)
    legacy_cv_folds = summary.get("dummy_baseline", {}).get("cv_folds")
    study_name = summary.get("study_name")
    if (
        summary.get("source_sha256") == source_sha256
        and legacy_cv_folds == cv_folds
        and isinstance(study_name, str)
        and study_name
    ):
        return study_name
    return _study_name(source_sha256, cv_folds)


def _set_or_validate_study_contract(
    study: optuna.Study,
    *,
    source_sha256: str,
    cv_folds: int,
    random_state: int,
) -> None:
    """Prevent incomparable trials from being resumed into one study."""
    expected = {
        "source_sha256": source_sha256,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "search_space_version": _SEARCH_SPACE_VERSION,
    }
    for key, value in expected.items():
        existing = study.user_attrs.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(
                f"Optuna study contract mismatch for {key}: "
                f"stored={existing!r}, requested={value!r}."
            )
        study.set_user_attr(key, value)


def _selected_study_name(path: Path, *, source_sha256: str) -> str:
    """Load the exact reviewed study, including legacy Section C summaries."""
    if not path.is_file():
        raise RuntimeError(
            "No model-selection summary exists. Run `talent-modeling tune` first."
        )
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read model-selection summary at {path}."
        ) from error
    if summary.get("source_sha256") != source_sha256:
        raise RuntimeError(
            "The model-selection summary belongs to a different source dataset. "
            "Run `talent-modeling tune` for the current data before finalizing."
        )
    study_name = summary.get("study_name")
    if not isinstance(study_name, str) or not study_name:
        raise RuntimeError("The model-selection summary has no valid study_name.")
    return study_name


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _validate_tuning_inputs(
    target_trials: int,
    cv_folds: int,
    timeout_minutes: float,
) -> None:
    if target_trials < 12:
        raise ValueError(
            "trials must be at least 12 so every model/preprocessing design "
            "is evaluated once."
        )
    if cv_folds < 3:
        raise ValueError("cv_folds must be at least 3.")
    if timeout_minutes <= 0:
        raise ValueError("timeout_minutes must be greater than zero.")
