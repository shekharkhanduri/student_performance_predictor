# Student Academic Performance Predictor

An end-to-end machine learning system that predicts student academic performance
scores using an **Ensemble Voting Regressor** combining 5 base models, with
**SHAP** and **LIME** explainability.

---

## Features

| Capability | Detail |
|---|---|
| Datasets | Dataset 1 (10,000 samples, 6 features) · Dataset 2 (6,607 samples, 20 features) |
| Preprocessing | Label-encoding, StandardScaler, mode/mean imputation, outlier correction |
| Feature selection | Pearson correlation heatmap; selected features per specification |
| Base models | Linear Regression, Ridge, KNN, XGBoost, CatBoost, AdaBoost, Random Forest, SVR, Bagging |
| Ensemble | `VotingRegressor` (top-5 models, weights optimised via grid search) |
| Validation | 10-fold cross-validation · paired t-tests vs SVR / XGBoost / CatBoost |
| Explainability | SHAP global (bar chart + summary dot plot) · LIME local (single-instance bar chart) |
| Visualisations | MAE/RMSE bar chart · R² bar chart · Actual vs Predicted scatter · Correlation heatmap · Pie chart · Line plot · Bar plot |
| Export | Trained `VotingRegressor` saved with `joblib` |

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
├── ds1_all_features_voting_regressor.joblib
├── ds1_selected_features_mae_rmse_comparison.png
├── ds1_selected_features_r2_comparison.png
├── ds1_selected_features_actual_vs_predicted.png
├── ds1_selected_features_voting_regressor.joblib
├── ds2_correlation_heatmap.png
├── ds2_all_features_shap_bar.png
├── ds2_all_features_shap_dot.png
├── ds2_all_features_lime_explanation.png
├── ds2_all_features_mae_rmse_comparison.png
├── ds2_all_features_r2_comparison.png
├── ds2_all_features_actual_vs_predicted.png
├── ds2_all_features_voting_regressor.joblib
├── ds2_selected_features_mae_rmse_comparison.png
├── ds2_selected_features_r2_comparison.png
├── ds2_selected_features_actual_vs_predicted.png
└── ds2_selected_features_voting_regressor.joblib
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

### Voting Regressor

The top-5 models by R² score — **Linear Regression, Ridge, CatBoost, XGBoost,
Random Forest** — are combined in a `VotingRegressor`. Initial weights
`[7, 7, 1, 1, 1]` reflect linear models' superior baseline performance.
Optimal weights are selected by evaluating weighted-average predictions on a
held-out validation split (no re-training required), choosing from:
`[7,7,1,1,1]`, `[5,5,2,2,2]`, `[6,6,2,1,1]`, `[8,8,1,1,1]`.

Validation: 10-fold cross-validation (mean MAE / RMSE / R²) and paired
t-tests against SVR, XGBoost, and CatBoost (significance threshold p < 0.05).

### Explainability

| Method | Scope | Output |
|---|---|---|
| SHAP `PermutationExplainer` | Global (25 test samples) | Bar chart of mean(|SHAP|) · Summary dot plot |
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
