from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split

from car_price.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from car_price.pipeline import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "audi.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def candidate_models() -> dict[str, object]:
    models: dict[str, object] = {
        "dummy_median": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from catboost import CatBoostRegressor

        models["catboost"] = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )
    except ImportError:
        print("CatBoost is not installed; skipping CatBoost comparison.")

    return models


def main() -> None:
    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # The held-out test set remains untouched here.
    # Model comparison is performed only on the training partition.
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": make_scorer(rmse, greater_is_better=False),
    }

    rows: list[dict[str, float | str]] = []

    for name, estimator in candidate_models().items():
        print(f"Evaluating {name}...")
        pipeline = build_pipeline(estimator)
        started = perf_counter()

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        elapsed = perf_counter() - started
        rows.append(
            {
                "model": name,
                "cv_r2_mean": float(np.mean(scores["test_r2"])),
                "cv_r2_std": float(np.std(scores["test_r2"])),
                "cv_mae_mean": float(-np.mean(scores["test_mae"])),
                "cv_mae_std": float(np.std(-scores["test_mae"])),
                "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
                "cv_rmse_std": float(np.std(-scores["test_rmse"])),
                "elapsed_seconds": elapsed,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["cv_mae_mean", "cv_rmse_mean"],
        ascending=True,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)

    print("\nCross-validation results (training partition only):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved report: {REPORT_PATH}")
    print(
        "\nUse this report to compare candidates. Do not replace the production "
        "Random Forest only because of one score; review stability, error profile, "
        "complexity, and diagnostics first."
    )


if __name__ == "__main__":
    main()
