import pandas as pd
import pytest

from car_price.data import validate_dataset


def valid_frame():
    return pd.DataFrame(
        [
            {
                "model": " A3",
                "year": 2019,
                "price": 18000,
                "transmission": "Manual",
                "mileage": 10000,
                "fuelType": "Petrol",
                "tax": 145,
                "mpg": 50.0,
                "engineSize": 1.0,
            }
        ]
    )


def test_valid_schema_passes():
    validate_dataset(valid_frame())


def test_missing_column_fails():
    df = valid_frame().drop(columns=["price"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df)


def test_non_positive_price_fails():
    df = valid_frame()
    df.loc[0, "price"] = 0

    with pytest.raises(ValueError, match="positive"):
        validate_dataset(df)
