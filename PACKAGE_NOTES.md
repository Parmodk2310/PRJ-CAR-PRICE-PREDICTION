# Car Price Prediction — Hardened Code Package

This ZIP contains the reproducible source-code layer created during the repository hardening pass.

Verified locally in the user's Windows 11 environment:

- Python 3.11.9
- R²: 0.9654
- MAE: 1517.10
- RMSE: 2286.48
- `pytest`: 1 passed
- `python -m compileall src`: passed
- prediction smoke test: passed

## Expected repository data

Place the Audi dataset at:

`data/raw/audi.csv`

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest ipykernel
python -m pip install -e .
```

## Train

```powershell
python -m car_price.train
```

## Predict

```powershell
python -m car_price.predict
```

## Test

```powershell
pytest -v
python -m compileall src
```

Generated model artifacts under `models/*.joblib` are intentionally ignored by Git.
