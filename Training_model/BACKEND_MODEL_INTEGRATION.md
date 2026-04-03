# Backend Integration Guide for Trained Models

This project now exports one scaler and one model per dataset in stable paths:

- `backend/mlmodel/scaler_ds1.joblib`
- `backend/mlmodel/scaler_ds2.joblib`
- `backend/mlmodel/model_ds1.joblib`
- `backend/mlmodel/model_ds2.joblib`

It also keeps pipeline outputs in `outputs/`:

- `outputs/ds1_all_features_stacking_regressor.joblib`
- `outputs/ds2_all_features_stacking_regressor.joblib`

## 1) Train and export artifacts

From project root:

```bash
python3 student_performance_predictor.py
```

This will:

- train Dataset 1 and Dataset 2 all-feature models
- disable selected-feature training path
- export dataset scalers and model aliases for backend use

## 2) Configure backend model/scaler paths

Your backend already supports environment-based paths in `backend/core/config.py`.
Use these variables in your runtime config (`.env`, container env, or deployment settings):

```env
DS1_MODEL_PATH=/absolute/path/to/backend/mlmodel/model_ds1.joblib
DS2_MODEL_PATH=/absolute/path/to/backend/mlmodel/model_ds2.joblib
DS1_SCALER_PATH=/absolute/path/to/backend/mlmodel/scaler_ds1.joblib
DS2_SCALER_PATH=/absolute/path/to/backend/mlmodel/scaler_ds2.joblib
```

If these are not set, backend defaults are used.

## 3) Keep preprocessing aligned with training

`backend/services/ml_service.py` already applies:

- DS1: binary encoding for `Extracurricular_Activities` + DS1 scaler
- DS2: categorical encoding + DS2 scaler

Important rule: feature names and order must exactly match training order.
This is already enforced by scaler validation in `_load_ds1_scaler()` and `_load_ds2_scaler()`.

## 4) API payload requirements

Prediction payload schema is in `backend/schemas/prediction.py`.

- For `dataset_type="ds1"`, include all DS1 required fields.
- For `dataset_type="ds2"`, include all DS2 required fields and valid categorical values.

If fields are missing or invalid, request validation fails before model inference.

## 5) Recommended backend startup check

At app startup (or health endpoint), verify artifacts can load:

- load DS1 model + scaler
- load DS2 model + scaler
- run one lightweight test prediction for each dataset type

This catches deployment issues early (missing files, wrong paths, incompatible artifacts).

## 6) Frontend integration notes

For your web app:

- send `dataset_type` (`ds1` or `ds2`) with each predict request
- render returned fields:
  - `predicted_exam_score`
  - `risk_level`
  - `top_negative_factors`
- ensure forms show only fields required for the selected dataset

## 7) Production tips

- Version model artifacts (`model_ds1_v1.joblib`, etc.) when retraining.
- Keep scaler and model from the same training run together.
- Use blue/green model switching via env vars for zero-downtime updates.
