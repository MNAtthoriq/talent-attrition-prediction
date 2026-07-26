"""Test the reusable leakage-safe preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from talent_attrition_prediction.modeling.data import MODEL_FEATURES
from talent_attrition_prediction.modeling.preprocessing import (
    PreprocessingConfig,
    build_model_pipeline,
    build_preprocessor,
    preprocessing_candidates,
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


@pytest.mark.parametrize("strategy", ["all_ohe", "ordinal_aware"])
@pytest.mark.parametrize("include_city", [False, True])
def test_preprocessor_handles_missing_and_unknown_categories(
    strategy: str,
    include_city: bool,
) -> None:
    development = _features()
    test = development.iloc[[0]].copy()
    test.loc[:, "city"] = "city_never_seen"

    config = PreprocessingConfig(
        strategy=strategy,
        include_city=include_city,
        min_category_frequency=2,
    )
    preprocessor = build_preprocessor(config)
    development_matrix = preprocessor.fit_transform(development)
    test_matrix = preprocessor.transform(test)

    assert development_matrix.shape[0] == len(development)
    assert test_matrix.shape == (1, development_matrix.shape[1])
    assert np.isfinite(development_matrix.toarray()).all()
    assert np.isfinite(test_matrix.toarray()).all()
    city_features = [
        feature
        for feature in preprocessor.get_feature_names_out()
        if feature.startswith("city_") and feature != "city_development_index"
    ]
    assert bool(city_features) is include_city


def test_ordinal_aware_preprocessor_preserves_explicit_order() -> None:
    preprocessor = build_preprocessor(
        PreprocessingConfig(
            strategy="ordinal_aware",
            min_category_frequency=1,
            sparse_output=False,
        )
    )
    preprocessor.fit(_features())

    ordinal_encoder = preprocessor.named_transformers_["ordinal"].named_steps["encoder"]

    assert ordinal_encoder.categories_[0].tolist() == [
        "Primary School",
        "High School",
        "Graduate",
        "Masters",
        "Phd",
    ]
    assert ordinal_encoder.categories_[1][0] == "<1"
    assert ordinal_encoder.categories_[1][-1] == ">20"
    assert ordinal_encoder.categories_[2][0] == "<10"
    assert ordinal_encoder.categories_[2][-1] == "10000+"


def test_candidate_matrix_covers_strategy_and_city_choices() -> None:
    assert [candidate.name for candidate in preprocessing_candidates()] == [
        "all_ohe_without_city",
        "all_ohe_with_city",
        "ordinal_aware_without_city",
        "ordinal_aware_with_city",
    ]


def test_preprocessor_requires_every_model_feature() -> None:
    development = _features().drop(columns="major_discipline")

    with pytest.raises(ValueError):
        build_preprocessor().fit(development)


def test_complete_pipeline_fits_and_predicts() -> None:
    features = _features()
    target = pd.Series([0, 1, 0, 1, 1, 0])
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500, random_state=42))

    pipeline.fit(features, target)
    probabilities = pipeline.predict_proba(features)[:, 1]

    assert probabilities.shape == (len(features),)
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()


def test_complete_pipeline_accepts_ordinal_aware_configuration() -> None:
    features = _features()
    target = pd.Series([0, 1, 0, 1, 1, 0])
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42),
        config=PreprocessingConfig(
            strategy="ordinal_aware",
            include_city=True,
            min_category_frequency=2,
        ),
    )

    pipeline.fit(features, target)

    assert pipeline.predict(features).shape == (len(features),)


def test_model_pipeline_requires_estimator() -> None:
    with pytest.raises(TypeError, match="cannot be None"):
        build_model_pipeline(None)


def test_preprocessing_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="strategy"):
        PreprocessingConfig(strategy="target_encoding")
    with pytest.raises(ValueError, match="at least 1"):
        PreprocessingConfig(min_category_frequency=0)
