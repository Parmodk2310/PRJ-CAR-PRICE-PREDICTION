import joblib
import pandas as pd
import pytest

from car_price.pipeline import build_pipeline


def test_pipeline_round_trip_serialization(tmp_path):
    X = pd.DataFrame(
        [
            {
                "model": " A3",
                "year": 2019,
                "transmission": "Manual",
                "mileage": 10000,
                "fuelType": "Petrol",
                "tax": 145,
                "mpg": 50.0,
                "engineSize": 1.0,
            },
            {
                "model": " A6",
                "year": 2018,
                "transmission": "Automatic",
                "mileage": 20000,
                "fuelType": "Diesel",
                "tax": 150,
                "mpg": 55.0,
                "engineSize": 2.0,
            },
        ]
    )
    y = [15000, 22000]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    before = pipeline.predict(X)
    after = restored.predict(X)

    assert after == pytest.approx(before)
