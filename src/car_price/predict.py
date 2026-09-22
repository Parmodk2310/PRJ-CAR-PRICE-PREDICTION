from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from car_price.data import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_pipeline.joblib"


def predict_price(
    *,
    model: str,
    year: int,
    transmission: str,
    mileage: float,
    fuel_type: str,
    tax: float,
    mpg: float,
    engine_size: float,
) -> float:
    """Load the production pipeline and predict one vehicle price."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run `python -m car_price.train` first."
        )

    pipeline = joblib.load(MODEL_PATH)

    row = pd.DataFrame(
        [
            {
                "model": model,
                "year": year,
                "transmission": transmission,
                "mileage": mileage,
                "fuelType": fuel_type,
                "tax": tax,
                "mpg": mpg,
                "engineSize": engine_size,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    return float(pipeline.predict(row)[0])


def main() -> None:
    prediction = predict_price(
        model=" A3",
        year=2019,
        transmission="Manual",
        mileage=1998,
        fuel_type="Petrol",
        tax=145,
        mpg=49.6,
        engine_size=1.0,
    )
    print(f"Estimated price: £{prediction:,.2f}")


if __name__ == "__main__":
    main()
