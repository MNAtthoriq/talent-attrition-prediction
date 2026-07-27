"""Define the small, deliberate Optuna model search space."""

from __future__ import annotations

from typing import Any

import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from talent_job_change_intent_prediction.modeling.preprocessing import (
    PreprocessingConfig,
    build_model_pipeline,
)

MODEL_NAMES = ("logistic_regression", "random_forest", "lightgbm")


def suggest_pipeline(
    trial: optuna.Trial,
    *,
    random_state: int,
) -> tuple[Pipeline, PreprocessingConfig, str]:
    """Build one pipeline from Optuna suggestions."""
    strategy = trial.suggest_categorical(
        "preprocessing_strategy",
        ["all_ohe", "ordinal_aware"],
    )
    include_city = trial.suggest_categorical("include_city", [False, True])
    min_frequency = trial.suggest_categorical(
        "min_category_frequency",
        [5, 10, 25, 50],
    )
    model_name = trial.suggest_categorical("model_name", list(MODEL_NAMES))
    config = PreprocessingConfig(
        strategy=strategy,
        include_city=include_city,
        min_category_frequency=min_frequency,
    )
    estimator = _suggest_estimator(trial, model_name, random_state)
    return build_model_pipeline(estimator, config=config), config, model_name


def build_pipeline_from_params(
    params: dict[str, Any],
    *,
    random_state: int,
) -> tuple[Pipeline, PreprocessingConfig, str]:
    """Rebuild the exact winning pipeline from stored Optuna parameters."""
    required = {
        "preprocessing_strategy",
        "include_city",
        "min_category_frequency",
        "model_name",
    }
    missing = sorted(required - params.keys())
    if missing:
        raise ValueError("Missing winning parameters: " + ", ".join(missing))

    config = PreprocessingConfig(
        strategy=params["preprocessing_strategy"],
        include_city=bool(params["include_city"]),
        min_category_frequency=int(params["min_category_frequency"]),
    )
    model_name = str(params["model_name"])
    estimator = _estimator_from_params(model_name, params, random_state)
    return build_model_pipeline(estimator, config=config), config, model_name


def initial_search_candidates() -> tuple[dict[str, Any], ...]:
    """Cover every model and preprocessing design before free TPE sampling."""
    return tuple(
        {
            "model_name": model_name,
            "preprocessing_strategy": strategy,
            "include_city": include_city,
            "min_category_frequency": 10,
        }
        for model_name in MODEL_NAMES
        for strategy in ("all_ohe", "ordinal_aware")
        for include_city in (False, True)
    )


def _suggest_estimator(
    trial: optuna.Trial,
    model_name: str,
    random_state: int,
):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("logistic_c", 1e-3, 1e2, log=True),
            l1_ratio=trial.suggest_categorical(
                "logistic_l1_ratio",
                [0.0, 1.0],
            ),
            class_weight=_class_weight(
                trial.suggest_categorical(
                    "logistic_class_weight",
                    ["none", "balanced"],
                )
            ),
            solver="liblinear",
            max_iter=2_000,
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "rf_n_estimators",
                200,
                700,
                step=100,
            ),
            max_depth=trial.suggest_int("rf_max_depth", 4, 20),
            min_samples_leaf=trial.suggest_int(
                "rf_min_samples_leaf",
                1,
                30,
                log=True,
            ),
            max_features=trial.suggest_categorical(
                "rf_max_features",
                ["sqrt", "log2", 0.5],
            ),
            class_weight=_class_weight(
                trial.suggest_categorical(
                    "rf_class_weight",
                    ["none", "balanced", "balanced_subsample"],
                )
            ),
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int(
                "lgbm_n_estimators",
                100,
                800,
                step=50,
            ),
            learning_rate=trial.suggest_float(
                "lgbm_learning_rate",
                0.01,
                0.20,
                log=True,
            ),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 127),
            max_depth=trial.suggest_int("lgbm_max_depth", 3, 14),
            min_child_samples=trial.suggest_int(
                "lgbm_min_child_samples",
                10,
                100,
            ),
            subsample=trial.suggest_float("lgbm_subsample", 0.60, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float(
                "lgbm_colsample_bytree",
                0.60,
                1.0,
            ),
            reg_alpha=trial.suggest_float(
                "lgbm_reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            reg_lambda=trial.suggest_float(
                "lgbm_reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
            class_weight=_class_weight(
                trial.suggest_categorical(
                    "lgbm_class_weight",
                    ["none", "balanced"],
                )
            ),
            objective="binary",
            verbosity=-1,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _estimator_from_params(
    model_name: str,
    params: dict[str, Any],
    random_state: int,
):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=float(params["logistic_c"]),
            l1_ratio=float(params["logistic_l1_ratio"]),
            class_weight=_class_weight(params["logistic_class_weight"]),
            solver="liblinear",
            max_iter=2_000,
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["rf_n_estimators"]),
            max_depth=int(params["rf_max_depth"]),
            min_samples_leaf=int(params["rf_min_samples_leaf"]),
            max_features=params["rf_max_features"],
            class_weight=_class_weight(params["rf_class_weight"]),
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=int(params["lgbm_n_estimators"]),
            learning_rate=float(params["lgbm_learning_rate"]),
            num_leaves=int(params["lgbm_num_leaves"]),
            max_depth=int(params["lgbm_max_depth"]),
            min_child_samples=int(params["lgbm_min_child_samples"]),
            subsample=float(params["lgbm_subsample"]),
            subsample_freq=1,
            colsample_bytree=float(params["lgbm_colsample_bytree"]),
            reg_alpha=float(params["lgbm_reg_alpha"]),
            reg_lambda=float(params["lgbm_reg_lambda"]),
            class_weight=_class_weight(params["lgbm_class_weight"]),
            objective="binary",
            verbosity=-1,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _class_weight(value: Any) -> str | None:
    return None if value == "none" else str(value)
