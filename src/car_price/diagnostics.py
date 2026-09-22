from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from car_price.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from car_price.pipeline import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "audi.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "test_metrics.csv"
ERRORS_PATH = PROJECT_ROOT / "reports" / "largest_errors.csv"


def main() -> None:
    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = np.asarray(
        pipeline.predict(X_test),
        dtype=float,
    ).reshape(-1)

    actual = y_test.to_numpy()
    residuals = actual - predictions

    metrics = {
        "r2": r2_score(actual, predictions),
        "mae": mean_absolute_error(actual, predictions),
        "rmse": mean_squared_error(actual, predictions) ** 0.5,
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    error_frame = X_test.copy()
    error_frame["actual_price"] = actual
    error_frame["predicted_price"] = predictions
    error_frame["absolute_error"] = np.abs(residuals)
    error_frame.sort_values("absolute_error", ascending=False).head(25).to_csv(
        ERRORS_PATH,
        index=False,
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predictions, alpha=0.35)
    minimum = min(float(actual.min()), float(predictions.min()))
    maximum = max(float(actual.max()), float(predictions.max()))
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Actual price (£)")
    plt.ylabel("Predicted price (£)")
    plt.title("Predicted vs Actual Prices")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(predictions, residuals, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted price (£)")
    plt.ylabel("Residual (£)")
    plt.title("Residuals vs Predicted Price")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual (£)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_distribution.png", dpi=160)
    plt.close()

    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        top = importances.head(20).sort_values("importance")
        plt.figure(figsize=(8, 7))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Feature importance")
        plt.title("Top 20 Random Forest Features")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
        plt.close()

        importances.to_csv(
            PROJECT_ROOT / "reports" / "feature_importance.csv",
            index=False,
        )

    print("Diagnostics complete.")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"Reports: {PROJECT_ROOT / 'reports'}")


if __name__ == "__main__":
    main()
