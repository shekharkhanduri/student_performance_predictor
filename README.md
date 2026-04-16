# Student Academic Performance Predictor

An end-to-end machine learning system that predicts student academic performance
scores using a **Stacking Ensemble Regressor** combining three non-linear base
learners (XGBoost, CatBoost, Random Forest) with a **LassoCV meta-learner**,
plus per-student **SHAP diagnostics** computed asynchronously as a background task.

The **Faculty Student Diagnostic System** is a full-stack web application with
a **FastAPI** backend (versioned REST API) and a **React/Vite** frontend.

---

## Project Structure

```
student-prediction/
│
├── Training_model/                 Standalone ML training pipeline
│   ├── data/
│   │   ├── dataset1.csv            10,000 samples · 5 features
│   │   └── dataset2.csv            6,607 samples  · 19 features
│   ├── outputs/                    Trained .joblib models + visualisation PNGs
│   └── student_performance_predictor.py
│
├── backend/                        FastAPI REST API
│   ├── main.py                     App bootstrap: model prewarming, DB migrations, CORS
│   ├── run.py                      `python backend/run.py` — starts uvicorn
│   │
│   ├── core/
│   │   ├── config.py               Paths, DATABASE_URL, MAX_UPLOAD_ROWS, API_V1_PREFIX
│   │   ├── database.py             SQLAlchemy engine + SessionLocal
│   │   └── migrations.py           Idempotent DDL helpers (new columns, indexes)
│   │
│   ├── models/
│   │   └── student.py              ORM: StudentDataDS1, StudentDataDS2
│   │                               (student_id auto-assigned, external_id for upsert,
│   │                                shap_status tracks background computation)
│   │
│   ├── schemas/
│   │   └── prediction.py           Pydantic: PredictRequest, PredictionResult,
│   │                               StudentDiagnostic, UploadSummary, ShapFactor
│   │
│   ├── services/
│   │   ├── ml_service.py           Two-stage inference:
│   │   │                           · predict_scores()          — fast, synchronous
│   │   │                           · compute_shap_explanations() — slow, background
│   │   │                           · predict_with_shap()       — combined (single student)
│   │   └── student_service.py      store_student, update_student_shap, row_to_diagnostic
│   │
│   ├── api/
│   │   ├── deps.py                 get_db, get_models, get_scalers (FastAPI deps)
│   │   └── routers/
│   │       ├── health.py           GET  /api/v1/health
│   │       └── students.py         POST /api/v1/upload      (202, background SHAP)
│   │                               GET  /api/v1/students     (paginated + filterable)
│   │                               GET  /api/v1/student/{id} (poll while shap_status=pending)
│   │                               POST /api/v1/predict      (synchronous, single student)
│   │
│   └── mlmodel/
│       ├── model_ds1.joblib
│       ├── model_ds2.joblib
│       ├── scaler_ds1.joblib
│       └── scaler_ds2.joblib
│
├── frontend/                       React + Vite SPA
│   └── src/
│       ├── main.jsx                QueryClientProvider bootstrap
│       ├── App.jsx                 3-tab shell using React Query hooks (no prop drilling)
│       ├── styles.css              Dark glassmorphism design system
│       ├── lib/
│       │   └── api.js              fetch wrappers → /api/v1/…
│       ├── hooks/
│       │   ├── useHealth.js        30 s polling
│       │   ├── useStudents.js      Cached list + invalidation helper
│       │   └── useStudent.js       Single student — auto-polls while shap_status=pending
│       └── components/
│           ├── Header.jsx          Hero banner + API/model status pills
│           ├── StatsPanel.jsx      4 stat cards (total, avg, at-risk, borderline)
│           ├── UploadPanel.jsx     Bulk CSV upload with animated step feedback
│           ├── StudentsTable.jsx   Clickable rows + SHAP status badge
│           ├── StudentDetails.jsx  SHAP factors + shap_status-aware rendering
│           ├── DashboardTab.jsx    2-column layout: upload+stats | table+details
│           └── PredictionForm.jsx  Dual-dataset survey form + shareable URL
│
└── requirements.txt
```

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Train the models

Place source CSVs in `Training_model/data/` then run:

```bash
python Training_model/student_performance_predictor.py
# Saves models to Training_model/outputs/
```

Copy the two all-features models and scalers into `backend/mlmodel/`:

```bash
cp Training_model/outputs/ds1_all_features_stacking_regressor.joblib backend/mlmodel/model_ds1.joblib
cp Training_model/outputs/ds2_all_features_stacking_regressor.joblib backend/mlmodel/model_ds2.joblib
# (scalers are generated separately — see Training_model script)
```

Pre-trained models are already committed to `backend/mlmodel/`.

### 3. Configure the database

```bash
cp .env.example .env
# Edit .env and set DATABASE_URL to your Neon PostgreSQL connection string.
```

### 4. Start the backend

```bash
python backend/run.py
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
# UI: http://localhost:5173
```

---

## API Reference  (`/api/v1/…`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | API + model status |
| `POST` | `/upload` | Bulk CSV → predict → store (HTTP 202, SHAP in background) |
| `GET`  | `/students` | Paginated list; filter by `risk_level`, `dataset_type` |
| `GET`  | `/student/{id}` | Full diagnostic; poll while `shap_status=pending` |
| `POST` | `/predict` | Single-student prediction (synchronous SHAP) |

### `student_id` vs `external_id`

| Field | Source | Purpose |
|-------|--------|---------|
| `student_id` | Auto-assigned by DB (`= id`) | Public stable identifier for API calls |
| `external_id` | Supplied by caller | Institution reference (roll number, email, …) |

Legacy CSVs that include a `student_id` column are automatically mapped to `external_id`.

---

## Async SHAP Flow (POST /upload)

```
Client → POST /upload
         ↓
         parse CSV · validate schema · enforce MAX_UPLOAD_ROWS (500)
         ↓
         predict_scores()  [synchronous — fast]
         store records with shap_status='pending'
         ↓
         HTTP 202 → Client (immediate response with student_ids)
         ↓ (background thread)
         compute_shap_explanations()  [slow — KernelExplainer]
         update shap_explanations + shap_status='done'

Client → poll GET /student/{id} until shap_status='done'
         (frontend useStudent hook polls every 3 s automatically)
```

---

## ML Methodology

| Aspect | Detail |
|--------|--------|
| Datasets | DS1: 10k rows, 5 features · DS2: 6.6k rows, 19 features |
| Preprocessing | Label-encoding, StandardScaler, mean/mode imputation |
| Feature selection | Pearson correlation heatmap |
| Base models | XGBoost, CatBoost, Random Forest |
| Ensemble | `StackingRegressor` (cv=5, meta-learner=LassoCV) |
| Explainability | SHAP `KernelExplainer` (global) · LIME (local) |
| Risk buckets | Stable ≥ 70 · Borderline 60–69 · At-Risk < 60 |

---

## Requirements

See `requirements.txt`. Key dependencies:

- `scikit-learn >= 1.0`, `xgboost >= 1.5`, `catboost >= 1.0`
- `shap >= 0.40`, `lime >= 0.2`
- `fastapi >= 0.110`, `uvicorn[standard]`, `sqlalchemy >= 2.0`, `psycopg2-binary`
- Frontend: `react 18`, `@tanstack/react-query 5`, `vite 5`
