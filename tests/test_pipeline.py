import pandas as pd

from car_price.pipeline import build_pipeline


def sample_training_data():
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
            {
                "model": " A4",
                "year": 2020,
                "transmission": "Semi-Auto",
                "mileage": 7000,
                "fuelType": "Petrol",
                "tax": 150,
                "mpg": 44.0,
                "engineSize": 1.5,
            },
        ]
    )
    y = [15000, 22000, 25000]
    return X, y


def test_pipeline_can_train_and_predict():
    X, y = sample_training_data()
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert all(value > 0 for value in predictions)


def test_pipeline_handles_unknown_categories():
    X, y = sample_training_data()
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    unseen = pd.DataFrame(
        [
            {
                "model": " unseen-model",
                "year": 2021,
                "transmission": "Manual",
                "mileage": 12000,
                "fuelType": "Hybrid",
                "tax": 140,
                "mpg": 60.0,
                "engineSize": 1.4,
            }
        ]
    )

    prediction = pipeline.predict(unseen)

    assert len(prediction) == 1
    assert prediction[0] > 0
