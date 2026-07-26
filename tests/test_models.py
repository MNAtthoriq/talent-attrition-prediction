"""Test Optuna model-space coverage and winner reconstruction."""

from __future__ import annotations

import optuna
import pytest

from talent_attrition_prediction.modeling.models import (
    MODEL_NAMES,
    build_pipeline_from_params,
    initial_search_candidates,
    suggest_pipeline,
)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fixed_trial_builds_every_model(model_name: str) -> None:
    params: dict[str, object] = {
        "preprocessing_strategy": "all_ohe",
        "include_city": False,
        "min_category_frequency": 10,
        "model_name": model_name,
    }
    if model_name == "logistic_regression":
        params |= {
            "logistic_c": 1.0,
            "logistic_l1_ratio": 0.0,
            "logistic_class_weight": "balanced",
        }
    elif model_name == "random_forest":
        params |= {
            "rf_n_estimators": 200,
            "rf_max_depth": 8,
            "rf_min_samples_leaf": 2,
            "rf_max_features": "sqrt",
            "rf_class_weight": "balanced_subsample",
        }
    else:
        params |= {
            "lgbm_n_estimators": 200,
            "lgbm_learning_rate": 0.05,
            "lgbm_num_leaves": 31,
            "lgbm_max_depth": 8,
            "lgbm_min_child_samples": 20,
            "lgbm_subsample": 0.8,
            "lgbm_colsample_bytree": 0.8,
            "lgbm_reg_alpha": 0.01,
            "lgbm_reg_lambda": 0.1,
            "lgbm_class_weight": "none",
        }

    suggested, config, suggested_name = suggest_pipeline(
        optuna.trial.FixedTrial(params),
        random_state=42,
    )
    rebuilt, rebuilt_config, rebuilt_name = build_pipeline_from_params(
        params,
        random_state=42,
    )

    assert suggested_name == rebuilt_name == model_name
    assert config == rebuilt_config
    assert type(suggested.named_steps["model"]) is type(rebuilt.named_steps["model"])


def test_initial_candidates_cover_all_twelve_designs() -> None:
    candidates = initial_search_candidates()

    assert len(candidates) == 12
    assert {candidate["model_name"] for candidate in candidates} == set(MODEL_NAMES)
    assert {candidate["preprocessing_strategy"] for candidate in candidates} == {
        "all_ohe",
        "ordinal_aware",
    }
    assert {candidate["include_city"] for candidate in candidates} == {
        False,
        True,
    }


def test_rebuild_rejects_missing_winner_parameters() -> None:
    with pytest.raises(ValueError, match="Missing winning parameters"):
        build_pipeline_from_params({}, random_state=42)
