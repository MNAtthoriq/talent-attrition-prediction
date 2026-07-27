"""Test probability-based technical and business evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from talent_job_change_intent_prediction.modeling.data import MODEL_FEATURES
from talent_job_change_intent_prediction.modeling.evaluation import (
    capacity_metrics,
    choose_f2_threshold,
    evaluate_cross_validation,
    fairness_by_group,
)
from talent_job_change_intent_prediction.modeling.preprocessing import (
    PreprocessingConfig,
    build_model_pipeline,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city_development_index": [0.62, 0.91, 0.78, 0.70, 0.88, 0.64],
            "city": ["city_1", "city_2", "city_1", "city_3", "city_2", "city_1"],
            "relevant_experience": [
                "Has relevant experience",
                "No relevant experience",
                "Has relevant experience",
                "Has relevant experience",
                "No relevant experience",
                "Has relevant experience",
            ],
            "enrolled_university": [
                "no_enrollment",
                "Full time course",
                None,
                "Part time course",
                "no_enrollment",
                "no_enrollment",
            ],
            "education_level": [
                "Graduate",
                "Masters",
                "High School",
                "Graduate",
                None,
                "Phd",
            ],
            "major_discipline": ["STEM", None, "Arts", "STEM", "Other", "STEM"],
            "experience": [">20", "4", "<1", "10", None, "15"],
            "company_size": [None, "50-99", "<10", "100-500", "10000+", "50-99"],
            "company_type": [
                "Pvt Ltd",
                None,
                "NGO",
                "Public Sector",
                "Pvt Ltd",
                "Funded Startup",
            ],
            "last_new_job": ["1", "never", None, ">4", "2", "1"],
        }
    ).loc[:, MODEL_FEATURES]


def test_cross_validation_uses_probability_scores() -> None:
    features = pd.concat([_features()] * 10, ignore_index=True)
    target = pd.Series(([0, 1, 0, 1, 0, 0] * 10), dtype="int8")
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42),
        config=PreprocessingConfig(min_category_frequency=2),
    )

    result = evaluate_cross_validation(
        pipeline,
        features,
        target,
        cv_folds=3,
        random_state=42,
    )

    assert 0 <= result["validation_average_precision_mean"] <= 1
    assert 0 <= result["oof_threshold"] <= 1
    assert len(result["folds"]) == 3
    assert result["positive_rate"] == pytest.approx(target.mean())


def test_capacity_metrics_reward_correct_low_risk_ranking() -> None:
    target = pd.Series([0, 0, 1, 1])
    probability = np.array([0.05, 0.10, 0.80, 0.90])

    result = capacity_metrics(
        target,
        probability,
        selection_fractions=[0.5],
    )[0]

    assert result["selected_rows"] == 2
    assert result["baseline_job_change_intent_rate"] == 0.5
    assert result["model_selected_job_change_intent_rate"] == 0.0
    assert result["relative_job_change_intent_reduction"] == 1.0


def test_f2_threshold_is_derived_from_scores() -> None:
    target = pd.Series([0, 0, 1, 1])
    probability = np.array([0.05, 0.20, 0.55, 0.90])

    threshold = choose_f2_threshold(target, probability)

    assert threshold == pytest.approx(0.55)


def test_fairness_audit_keeps_missing_group() -> None:
    audit = pd.DataFrame({"gender": ["Male"] * 30 + [None] * 30})
    target = pd.Series(([0, 1] * 30), dtype="int8")
    probability = np.linspace(0.05, 0.95, 60)

    result = fairness_by_group(
        audit,
        target,
        probability,
        group_column="gender",
        threshold=0.5,
    )

    assert {row["group"] for row in result["groups"]} == {
        "Male",
        "__MISSING__",
    }


def test_invalid_capacity_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="selection fractions"):
        capacity_metrics(
            pd.Series([0, 1]),
            np.array([0.1, 0.9]),
            selection_fractions=[0],
        )
