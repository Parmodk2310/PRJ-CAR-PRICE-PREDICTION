from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from car_price.data import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-safe preprocessing for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )


def build_pipeline(estimator: RegressorMixin | None = None) -> Pipeline:
    """Build the production pipeline.

    RandomForestRegressor is the current validated production baseline.
    """
    if estimator is None:
        estimator = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", estimator),
        ]
    )
