# Audi Used-Car Price Prediction

Production-oriented machine-learning project for estimating used Audi vehicle prices from structured vehicle attributes.

The project goes beyond a notebook-only model by providing a reproducible training pipeline, leakage-safe preprocessing, held-out evaluation, model comparison, diagnostics, automated tests, CI, and an interactive Gradio deployment.

## Overview

Used-car pricing depends on several interacting factors including model, age, mileage, transmission, fuel type, engine size, fuel economy, and road tax.

This project builds an end-to-end regression workflow that converts those structured attributes into a benchmark resale-price estimate.

### Dataset

- 10,668 Audi used-car records
- 8 input features
- Target: `price`

Features:

`model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`

## Model Performance

Final held-out evaluation:

| Metric | Result |
| --- | ---: |
| R² | **0.9654** |
| MAE | **£1,517.10** |
| RMSE | **£2,286.48** |

The test partition is held out from model fitting and preprocessing.

## ML Pipeline

```text
Raw Audi data
      │
      ▼
Schema validation
      │
      ▼
80 / 20 train-test split
      │
      ▼
ColumnTransformer
 ├── categorical → OneHotEncoder
 └── numerical   → passthrough
      │
      ▼
Regression model
      │
      ▼
Held-out evaluation
      │
      ├── R²
      ├── MAE
      └── RMSE
      │
      ▼
Persisted sklearn pipeline
      │
      ▼
Gradio inference application
```

Preprocessing is fitted inside the scikit-learn pipeline, preventing information from the held-out test partition from leaking into model preparation.

## Model Comparison

Candidate models were evaluated using 5-fold cross-validation on the training partition.

| Model | CV R² | CV MAE | CV RMSE |
| --- | ---: | ---: | ---: |
| CatBoost | 0.9615 | £1,531 | £2,265 |
| Extra Trees | 0.9593 | £1,531 | £2,325 |
| Random Forest | 0.9572 | £1,532 | £2,387 |
| Linear Regression | 0.8853 | £2,561 | £3,901 |
| Dummy Median | -0.0519 | £8,050 | £11,851 |

The deployed project currently uses the persisted Random Forest pipeline.

## Engineering Highlights

- reusable `src/car_price` Python package
- schema and target validation
- preprocessing contained inside the ML pipeline
- unknown categorical values handled safely
- reproducible train/test split
- cross-validation model comparison
- residual and prediction-error diagnostics
- feature-importance reporting
- serialized end-to-end inference pipeline
- automated pytest suite
- Ruff linting and formatting
- GitHub Actions CI
- Hugging Face / Gradio deployment

## Repository Structure

```text
.
├── .github/
│   └── workflows/
├── data/
│   └── raw/
├── hf_space/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_experiments.ipynb
│   └── README.md
├── reports/
│   ├── figures/
│   ├── feature_importance.csv
│   ├── largest_errors.csv
│   ├── model_comparison.csv
│   └── test_metrics.csv
├── src/
│   └── car_price/
│       ├── data.py
│       ├── pipeline.py
│       ├── train.py
│       ├── predict.py
│       ├── compare_models.py
│       ├── diagnostics.py
│       └── export_space.py
├── tests/
├── pyproject.toml
└── README.md
```

## Reproduce Locally

Python 3.11 is recommended.

```bash
git clone https://github.com/Parmodk2310/PRJ-CAR-PRICE-PREDICTION.git
cd PRJ-CAR-PRICE-PREDICTION

python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis,app]"
```

Train the model:

```powershell
python -m car_price.train
```

Generate diagnostics:

```powershell
python -m car_price.diagnostics
```

Run a prediction smoke test:

```powershell
python -m car_price.predict
```

## Quality Checks

```powershell
ruff check src tests hf_space
ruff format --check src tests hf_space
python -m compileall src hf_space
pytest -v
python -m pip check
```

Current test suite:

```text
6 passed
```

## Interactive Demo

A Gradio-based deployment is maintained separately from the source repository.

**Live demo:**  
https://huggingface.co/spaces/Parmodk2310/PRJ-CAR-PRICE-PREDICTION

The deployment repository contains the exported model artifact through Git LFS, while large generated model binaries are intentionally excluded from this GitHub repository.

## Limitations

The model estimates benchmark prices from historical structured data.

It does not directly account for factors such as:

- physical vehicle condition
- accident or repair history
- service history
- aftermarket modifications
- exact geographic pricing differences
- unusual or out-of-distribution vehicles

Predictions should therefore be interpreted as model estimates rather than guaranteed market valuations.

## Tech Stack

Python · pandas · NumPy · scikit-learn · CatBoost · Matplotlib · pytest · Ruff · GitHub Actions · Gradio · Hugging Face

## License

MIT License
