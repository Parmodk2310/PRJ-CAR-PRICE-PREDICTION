---
title: Audi Price Intelligence
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
short_description: Audi used-car price prediction with ML
tags:
  - regression
  - scikit-learn
  - random-forest
  - tabular
  - machine-learning
---

# Audi Price Intelligence

Interactive deployment for the Audi used-car price prediction project.

The Space loads a persisted scikit-learn preprocessing + model pipeline and estimates a benchmark resale price from structured vehicle attributes.

## Model evidence

- Held-out R²: **0.9654**
- MAE: **£1,517.10**
- RMSE: **£2,286.48**
- 10,668 historical Audi listings
- 8 structured input features
- 80/20 held-out evaluation
- 5-fold cross-validation on the training partition

The application performs inference only; it does not retrain the model for every request.

**Source repository:**  
https://github.com/Parmodk2310/PRJ-CAR-PRICE-PREDICTION

**Portfolio:**  
https://parmodk2310.vercel.app

> This demo is intended for portfolio, educational, and benchmark-estimation use. It is not a guaranteed market valuation.
