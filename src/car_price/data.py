from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "model",
    "year",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
]
TARGET_COLUMN = "price"
CATEGORICAL_FEATURES = ["model", "transmission", "fuelType"]
NUMERICAL_FEATURES = ["year", "mileage", "tax", "mpg", "engineSize"]

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the Audi dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate required schema and obvious data-quality problems."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        details = ", ".join(f"{name}={count}" for name, count in null_counts.items())
        raise ValueError(f"Required columns contain missing values: {details}")

    if (df[TARGET_COLUMN] <= 0).any():
        raise ValueError("Target column 'price' must contain positive values.")

    for column in NUMERICAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Expected numeric column: {column}")
