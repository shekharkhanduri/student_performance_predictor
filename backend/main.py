"""FastAPI application bootstrap for the Faculty Student Diagnostic System.

Startup sequence (lifespan):
  1. Load models + scalers into app.state  (eliminates cold-start on first request)
  2. Create DB tables if they do not exist
  3. Run idempotent schema migrations      (new columns, index backfills)

All routes are mounted under /api/v1 for versioning.
"""

import warnings
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers.health import router as health_router
from backend.api.routers.students import router as students_router
from backend.core.config import (
    API_V1_PREFIX,
    DS1_MODEL_PATH,
    DS1_SCALER_PATH,
    DS2_MODEL_PATH,
    DS2_SCALER_PATH,
)
from backend.core.database import Base, engine
from backend.core.migrations import run_migrations
from backend.services.ml_service import build_explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Prewarm models (eliminates cold-start latency on first request) ──
    try:
        app.state.models = {
            "ds1": joblib.load(DS1_MODEL_PATH),
            "ds2": joblib.load(DS2_MODEL_PATH),
        }
        app.state.scalers = {
            "ds1": joblib.load(DS1_SCALER_PATH),
            "ds2": joblib.load(DS2_SCALER_PATH),
        }
        # Build the fastest available SHAP explainer for each dataset.
        # TreeExplainer (ms-level) is selected when a tree-based base estimator
        # is found inside the stacking regressor; otherwise falls back to a
        # small-background KernelExplainer.
        app.state.explainers = {
            "ds1": build_explainer(app.state.models["ds1"], "ds1"),
            "ds2": build_explainer(app.state.models["ds2"], "ds2"),
        }
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not load ML models: {exc}. "
            "Prediction endpoints will return 503 until models are available.",
            RuntimeWarning,
            stacklevel=2,
        )
        app.state.models = {}
        app.state.scalers = {}

    # ── 2. Database bootstrap + migrations ──────────────────────────────────
    try:
        Base.metadata.create_all(bind=engine)
        run_migrations(engine)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Database setup failed: {exc}. "
            "Ensure DATABASE_URL is set and the database is reachable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── 3. Mark stale-pending SHAP records as failed ────────────────────────
    # Any record left as shap_status='pending' across a server restart will
    # never complete — its background task was killed when the process died.
    # Mark them as 'failed' immediately so the frontend stops polling.
    try:
        from datetime import datetime, timedelta, timezone

        from backend.core.database import SessionLocal
        from backend.models import StudentDataDS1, StudentDataDS2

        _stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        with SessionLocal() as db:
            for _model_class in (StudentDataDS1, StudentDataDS2):
                _stale = (
                    db.query(_model_class)
                    .filter(
                        _model_class.shap_status == "pending",
                        _model_class.created_at < _stale_cutoff,
                    )
                    .all()
                )
                if _stale:
                    warnings.warn(
                        f"Marking {len(_stale)} stale-pending {_model_class.__tablename__} records as 'failed'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                for _row in _stale:
                    _row.shap_status = "failed"
                    _row.updated_at = datetime.now(timezone.utc)
            db.commit()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Stale-pending cleanup failed: {exc}", RuntimeWarning, stacklevel=2)

    yield


app = FastAPI(
    title="Faculty Student Diagnostic System",
    description=(
        "Predict student exam risk using a Stacking Ensemble model "
        "(XGBoost + CatBoost + RandomForest + LassoCV) with background SHAP explanations."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All routes are versioned under /api/v1
app.include_router(health_router, prefix=API_V1_PREFIX)
app.include_router(students_router, prefix=API_V1_PREFIX)
