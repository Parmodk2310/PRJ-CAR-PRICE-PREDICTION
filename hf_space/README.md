---
title: Audi Price Intelligence
emoji: 🚗
colorFrom: blue
colorTo: slate
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
short_description: Reproducible Audi used-car regression pipeline with transparent engineering evidence
tags:
  - regression
  - scikit-learn
  - random-forest
  - tabular
  - machine-learning
---

# Audi Price Intelligence

A recruiter- and engineering-review-friendly interactive demo for the
`PRJ-CAR-PRICE-PREDICTION` project.

The Space presents both inference and model evidence:

- held-out R², MAE, and RMSE,
- training record count and feature count,
- end-to-end persisted preprocessing/model pipeline,
- 80/20 evaluation design,
- 5-fold training-partition cross-validation methodology,
- model limitations and intended use,
- clean interactive vehicle-price inference.

The deployed application loads a pre-trained scikit-learn pipeline. It does not
retrain the model for every request.

**Source repository:**  
https://github.com/Parmodk2310/PRJ-CAR-PRICE-PREDICTION

**Portfolio:**  
https://parmodk2310.vercel.app
