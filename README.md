# Student Academic Performance Predictor

An end-to-end machine learning system that predicts student academic performance
scores using a **Stacking Ensemble Regressor** combining three non-linear base
learners (XGBoost, CatBoost, Random Forest) with a **LassoCV meta-learner**,
plus per-student **SHAP diagnostics** and **LIME** explainability.

Now includes a **Faculty Student Diagnostic System** — a full-stack web
application with a **FastAPI** backend and a **Streamlit** dashboard.

---

## Features

| Capability | Detail |
|---|---|
| Datasets | Dataset 1 (10,000 samples, 6 features) · Dataset 2 (6,607 samples, 20 features) |
| Preprocessing | Label-encoding, StandardScaler, mode/mean imputation, outlier correction |
| Feature selection | Pearson correlation heatmap; selected features per specification |
| Base models | XGBoost, CatBoost, Random Forest (non-linear, diversity-focused) |
| Ensemble | `StackingRegressor` (cv=5, meta-learner=LassoCV for automatic weight learning) |
| Validation | Paired t-tests vs. each base learner |
| Explainability | SHAP global (KernelExplainer, bar chart + summary dot plot) · per-student SHAP diagnostics · LIME local (single-instance) |
| Risk categorisation | **Stable** (≥ 70) · **Borderline** (60–69) · **At-Risk** (< 60) |
| Diagnostic output | `predicted_score`, `risk_level`, `top_negative_factors` per student |
| Visualisations | MAE/RMSE bar chart · R² bar chart · Actual vs Predicted scatter · Correlation heatmap · Pie chart · Line plot · Bar plot |
| Export | Trained `StackingRegressor` saved with `joblib` |
| Backend | FastAPI REST API with CSV upload, per-student retrieval, single-student prediction |
| Frontend | Streamlit dashboard with Global Pulse, Student Discovery, Diagnostic Report, Manual Entry |
| Database | PostgreSQL (Neon) via SQLAlchemy — stores all features, predictions, and SHAP explanations |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Provide real datasets

Download the Kaggle datasets and place them in the `data/` directory:

| File | Kaggle dataset |
|---|---|
| `data/dataset1.csv` | *Student Performance Factors* (10 k rows, `Performance_Index` target) |
| `data/dataset2.csv` | *Students Performance* (6,607 rows, `Exam_Score` target) |

If the files are absent, **synthetic data** matching the described distributions
is generated automatically.

### 3. Run the pipeline

```bash
python student_performance_predictor.py
```

All plots and trained models are saved in the `outputs/` directory.

---

## Faculty Student Diagnostic System

### Architecture

```
backend/          FastAPI REST API
  main.py         Endpoints: /upload, /student/{id}, /students, /predict, /health
  database.py     SQLAlchemy + Neon PostgreSQL
  models.py       StudentData ORM model
  schemas.py      Pydantic request/response schemas
  ml_utils.py     Model loading, preprocessing, SHAP inference

frontend/
  app.py          Streamlit dashboard (4 pages)

run.py            Convenience launcher
```

### 4. Start the backend

```bash
# Requires outputs/ds2_all_features_stacking_regressor.joblib (step 3 above)
python run.py backend
# API docs: http://localhost:8000/docs
```

Set the database URL via environment variable if needed:
```bash
export DATABASE_URL="postgresql://user:pass@host/dbname?sslmode=require"
```

### 5. Start the Streamlit dashboard

```bash
python run.py frontend
# Dashboard: http://localhost:8501
```

Or start both together:
```bash
python run.py both
```

### Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Global Pulse | Class Health Index gauge, Intervention Alert, score histogram |
| 🔍 Student Discovery | Searchable/filterable student table with risk badges |
| 📋 Diagnostic Report | SHAP waterfall, score gauge, intervention simulator |
| ✏️ Manual Entry | Single-student form with real-time prediction |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload CSV; validate columns, predict all rows, store in DB |
| `GET` | `/student/{id}` | Full diagnostic JSON for one student |
| `GET` | `/students` | Paginated list of all stored students |
| `POST` | `/predict` | Single student JSON prediction |
| `GET` | `/health` | Health check + model status |

### Required CSV Columns (Dataset 2)

```
Hours_Studied, Attendance, Gender, Parental_Involvement, Access_to_Resources,
Extracurricular_Activities, Sleep_Hours, Previous_Scores, Motivation_Level,
Internet_Access, Tutoring_Sessions, Family_Income, Teacher_Quality, School_Type,
Peer_Influence, Physical_Activity, Learning_Disabilities, Parental_Education_Level,
Distance_from_Home
```

---

## Output Files

After a successful run the `outputs/` directory contains:

```
outputs/
├── ds1_correlation_heatmap.png
├── ds1_extracurricular_pie.png
├── ds1_sleep_vs_performance.png
├── ds1_sample_papers_vs_performance.png
├── ds1_all_features_shap_bar.png
├── ds1_all_features_shap_dot.png
├── ds1_all_features_lime_explanation.png
├── ds1_all_features_mae_rmse_comparison.png
├── ds1_all_features_r2_comparison.png
├── ds1_all_features_actual_vs_predicted.png
├── ds1_all_features_stacking_regressor.joblib
├── ds1_selected_features_mae_rmse_comparison.png
├── ds1_selected_features_r2_comparison.png
├── ds1_selected_features_actual_vs_predicted.png
├── ds1_selected_features_stacking_regressor.joblib
├── ds2_correlation_heatmap.png
├── ds2_all_features_shap_bar.png
├── ds2_all_features_shap_dot.png
├── ds2_all_features_lime_explanation.png
├── ds2_all_features_mae_rmse_comparison.png
├── ds2_all_features_r2_comparison.png
├── ds2_all_features_actual_vs_predicted.png
├── ds2_all_features_stacking_regressor.joblib
├── ds2_selected_features_mae_rmse_comparison.png
├── ds2_selected_features_r2_comparison.png
├── ds2_selected_features_actual_vs_predicted.png
└── ds2_selected_features_stacking_regressor.joblib
```

---

## Methodology

### Preprocessing

**Dataset 1**
- `Extracurricular_Activities` label-encoded (Yes → 1, No → 0)
- All numerical features scaled with `StandardScaler`

**Dataset 2**
- `Exam_Score` outlier value 101 replaced with 100
- Missing values in `Teacher_Quality`, `Parental_Education_Level`,
  `Distance_from_Home` imputed with column mode
- All categorical columns label-encoded

### Feature Selection

Pearson correlation between each feature and the target is computed and
visualised as a heatmap. Per specification:

- **Dataset 1:** retain `Previous_Scores` (r ≈ 0.92) and `Hours_Studied`
  (r ≈ 0.37)
- **Dataset 2:** retain `Attendance` (r ≈ 0.58) and `Hours_Studied`
  (r ≈ 0.45)

Each dataset is evaluated under two conditions: *all features* and
*selected features only*.

### Stacking Ensemble

Three non-linear base learners — **XGBoost, CatBoost, Random Forest** — are
combined in a `StackingRegressor` with **LassoCV** as the meta-learner.
LassoCV automatically learns optimal blending weights via L1 regularisation,
eliminating the need for manual weight grids. 5-fold cross-validation is used
internally to generate out-of-fold meta-features for training the final
estimator.

Validation: paired t-tests of the StackingRegressor against each individual
base learner (significance threshold p < 0.05).

### Risk Categorisation

Predicted continuous scores are mapped to three discrete risk buckets:

| Risk Level | Score Range | Colour |
|---|---|---|
| Stable | ≥ 70 | Green |
| Borderline | 60 – 69 | Yellow |
| At-Risk | < 60 | Red |

### Per-Student Diagnostic Output

For each student the `predict_with_diagnostics()` function returns:

```python
{
    "predicted_score":      float,         # regression output
    "risk_level":           str,           # "Stable", "Borderline", "At-Risk"
    "top_negative_factors": list[str],     # top-3 features dragging score down
}
```

SHAP values are computed using `shap.KernelExplainer`, which is compatible with
any sklearn-compatible model. The three features with the most negative SHAP
contributions are reported as the top negative factors.

### Explainability

| Method | Scope | Output |
|---|---|---|
| SHAP `KernelExplainer` | Global (25 test samples) | Bar chart of mean(|SHAP|) · Summary dot plot |
| SHAP `KernelExplainer` | Per-student (5 sample students) | `top_negative_factors` in diagnostic dict |
| LIME `LimeTabularExplainer` | Local (single instance, 5 features) | Feature-contribution bar chart |

---

## Requirements

See `requirements.txt`. Key dependencies:

- `scikit-learn >= 1.0`
- `xgboost >= 1.5`
- `catboost >= 1.0`
- `shap >= 0.40`
- `lime >= 0.2`
- `matplotlib`, `seaborn`, `scipy`, `joblib`, `pandas`, `numpy`

