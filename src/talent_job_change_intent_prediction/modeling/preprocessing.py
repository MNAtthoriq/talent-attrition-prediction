"""Build reusable scikit-learn preprocessing and model pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from talent_job_change_intent_prediction.modeling.data import (
    CITY_FEATURE,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORY_ORDERS,
    ORDINAL_FEATURES,
)

PreprocessingStrategy = Literal["all_ohe", "ordinal_aware"]


@dataclass(frozen=True)
class PreprocessingConfig:
    """One leakage-safe preprocessing candidate for Section C evaluation."""

    strategy: PreprocessingStrategy = "all_ohe"
    include_city: bool = False
    min_category_frequency: int = 10
    sparse_output: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in ("all_ohe", "ordinal_aware"):
            raise ValueError("strategy must be either 'all_ohe' or 'ordinal_aware'.")
        if isinstance(self.min_category_frequency, bool) or not isinstance(
            self.min_category_frequency, int
        ):
            raise TypeError("min_category_frequency must be an integer.")
        if self.min_category_frequency < 1:
            raise ValueError("min_category_frequency must be at least 1.")

    @property
    def name(self) -> str:
        """Return a stable MLflow-friendly candidate name."""
        city_suffix = "with_city" if self.include_city else "without_city"
        return f"{self.strategy}_{city_suffix}"

    def to_dict(self) -> dict[str, object]:
        """Return tracked, JSON-serializable configuration values."""
        return asdict(self)


def preprocessing_candidates() -> tuple[PreprocessingConfig, ...]:
    """Return the four candidates Section C will compare."""
    return tuple(
        PreprocessingConfig(strategy=strategy, include_city=include_city)
        for strategy in ("all_ohe", "ordinal_aware")
        for include_city in (False, True)
    )


def build_preprocessor(
    config: PreprocessingConfig | None = None,
) -> ColumnTransformer:
    """Return preprocessing fitted only when its containing pipeline is fitted."""
    config = config or PreprocessingConfig()
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    nominal_features = list(NOMINAL_FEATURES)
    if config.include_city:
        nominal_features.insert(0, CITY_FEATURE)

    nominal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=config.min_category_frequency,
                    sparse_output=config.sparse_output,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
    ]
    if config.strategy == "all_ohe":
        all_ohe_features = [*nominal_features, *ORDINAL_FEATURES]
        transformers.append(("categorical", nominal_pipeline, all_ohe_features))
    else:
        ordinal_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[
                            list(ORDINAL_CATEGORY_ORDERS[column])
                            for column in ORDINAL_FEATURES
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.extend(
            [
                ("nominal", nominal_pipeline, nominal_features),
                ("ordinal", ordinal_pipeline, list(ORDINAL_FEATURES)),
            ]
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0 if config.sparse_output else 0.0,
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: Any,
    *,
    config: PreprocessingConfig | None = None,
) -> Pipeline:
    """Wrap an estimator with the complete deployable preprocessing contract."""
    if estimator is None:
        raise TypeError("estimator cannot be None.")
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("model", estimator),
        ]
    )
