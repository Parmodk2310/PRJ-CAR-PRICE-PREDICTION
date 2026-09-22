from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from car_price.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_dataset,
)
from car_price.pipeline import build_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "audi.csv"
)

SPACE_DIR = PROJECT_ROOT / "hf_space"

SPACE_MODEL_PATH = (
    SPACE_DIR
    / "models"
    / "car_price_pipeline.joblib"
)

SPACE_METADATA_PATH = (
    SPACE_DIR
    / "model_metadata.json"
)


def main() -> None:
    df = load_dataset(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Reproduce the reported held-out evaluation before refitting
    # the deployment artifact on all available rows.
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    evaluation_pipeline = build_pipeline()
    evaluation_pipeline.fit(
        X_train,
        y_train,
    )

    predictions = evaluation_pipeline.predict(
        X_test
    )

    evaluation = {
        "r2": float(
            r2_score(
                y_test,
                predictions,
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        ),
        "rmse": float(
            mean_squared_error(
                y_test,
                predictions,
            )
            ** 0.5
        ),
        "test_fraction": 0.20,
        "random_state": 42,
    }

    # After evaluation is recorded, fit the validated deployment
    # pipeline on the complete dataset for public inference.
    deployment_pipeline = build_pipeline()

    deployment_pipeline.fit(
        X,
        y,
    )

    SPACE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        deployment_pipeline,
        SPACE_MODEL_PATH,
    )

    metadata = {
        "model_name": "Random Forest",
        "model_version": "v0.2",
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "categorical": {
            column: [
                {
                    "label": str(value).strip(),
                    "value": str(value),
                }
                for value in sorted(
                    df[column]
                    .astype(str)
                    .unique()
                )
            ]
            for column in CATEGORICAL_FEATURES
        },
        "numerical": {
            column: {
                "min": float(
                    df[column].min()
                ),
                "max": float(
                    df[column].max()
                ),
                "default": float(
                    df[column].median()
                ),
            }
            for column in NUMERICAL_FEATURES
        },
        "evaluation": evaluation,
        "training_rows": int(len(df)),
        "target": TARGET_COLUMN,
        "deployment_fit": (
            "full_dataset_after_evaluation"
        ),
        "note": (
            "Held-out metrics come from the fixed 80/20 "
            "evaluation split. The deployment pipeline is "
            "then refit on the complete dataset."
        ),
    }

    SPACE_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Hugging Face deployment bundle exported."
    )

    print(
        f"Model:    {SPACE_MODEL_PATH}"
    )

    print(
        f"Metadata: {SPACE_METADATA_PATH}"
    )

    print(
        f"R2:       {evaluation['r2']:.4f}"
    )

    print(
        f"MAE:      {evaluation['mae']:.2f}"
    )

    print(
        f"RMSE:     {evaluation['rmse']:.2f}"
    )


if __name__ == "__main__":
    main()
