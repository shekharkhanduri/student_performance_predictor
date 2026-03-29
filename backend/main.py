"""
FastAPI backend for the Faculty Student Diagnostic System.

Endpoints
---------
POST /upload          – Upload a CSV of student records; validate, predict, store.
GET  /student/{id}    – Retrieve the full diagnostic JSON for one student.
POST /predict         – Single-student manual entry prediction.
GET  /students        – List all stored student summaries (for dashboard).
GET  /health          – Health-check / model status.
"""

import io
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .ml_utils import REQUIRED_FEATURES, predict_batch, predict_single
from .models import StudentData
from .schemas import (
    PredictRequest,
    PredictionResult,
    ShapFactor,
    StudentDiagnostic,
    UploadSummary,
)

# ── App setup ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup; log a clear warning if DB is unreachable
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"Could not create database tables: {exc}. "
            "Ensure DATABASE_URL is set correctly and the database is reachable.",
            RuntimeWarning,
            stacklevel=2,
        )
    yield


app = FastAPI(
    title="Faculty Student Diagnostic System",
    description=(
        "Predict student exam risk using a Stacking Ensemble model "
        "(XGBoost + CatBoost + RandomForest + LassoCV) with SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row_to_diagnostic(row: StudentData) -> StudentDiagnostic:
    features = {
        "Hours_Studied": row.hours_studied,
        "Attendance": row.attendance,
        "Gender": row.gender,
        "Parental_Involvement": row.parental_involvement,
        "Access_to_Resources": row.access_to_resources,
        "Extracurricular_Activities": row.extracurricular_activities,
        "Sleep_Hours": row.sleep_hours,
        "Previous_Scores": row.previous_scores,
        "Motivation_Level": row.motivation_level,
        "Internet_Access": row.internet_access,
        "Tutoring_Sessions": row.tutoring_sessions,
        "Family_Income": row.family_income,
        "Teacher_Quality": row.teacher_quality,
        "School_Type": row.school_type,
        "Peer_Influence": row.peer_influence,
        "Physical_Activity": row.physical_activity,
        "Learning_Disabilities": row.learning_disabilities,
        "Parental_Education_Level": row.parental_education_level,
        "Distance_from_Home": row.distance_from_home,
    }
    return StudentDiagnostic(
        student_id=row.id,
        student_name=row.student_name,
        predicted_exam_score=row.predicted_exam_score or 0.0,
        risk_level=row.risk_level or "Unknown",
        top_negative_factors=row.shap_explanations or [],
        created_at=row.created_at,
        features=features,
    )


def _store_student(
    db: Session,
    raw_row: dict,
    result: dict,
    student_name: str | None = None,
    batch_id: str | None = None,
) -> StudentData:
    record = StudentData(
        student_name=student_name or raw_row.get("student_name"),
        upload_batch=batch_id,
        hours_studied=raw_row.get("Hours_Studied"),
        attendance=raw_row.get("Attendance"),
        gender=raw_row.get("Gender"),
        parental_involvement=raw_row.get("Parental_Involvement"),
        access_to_resources=raw_row.get("Access_to_Resources"),
        extracurricular_activities=raw_row.get("Extracurricular_Activities"),
        sleep_hours=raw_row.get("Sleep_Hours"),
        previous_scores=raw_row.get("Previous_Scores"),
        motivation_level=raw_row.get("Motivation_Level"),
        internet_access=raw_row.get("Internet_Access"),
        tutoring_sessions=raw_row.get("Tutoring_Sessions"),
        family_income=raw_row.get("Family_Income"),
        teacher_quality=raw_row.get("Teacher_Quality"),
        school_type=raw_row.get("School_Type"),
        peer_influence=raw_row.get("Peer_Influence"),
        physical_activity=raw_row.get("Physical_Activity"),
        learning_disabilities=raw_row.get("Learning_Disabilities"),
        parental_education_level=raw_row.get("Parental_Education_Level"),
        distance_from_home=raw_row.get("Distance_from_Home"),
        predicted_exam_score=result["predicted_exam_score"],
        risk_level=result["risk_level"],
        shap_explanations=result["top_negative_factors"],
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.flush()
    return record


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Health check — also verifies model file availability."""
    from .ml_utils import MODEL_PATH
    import os

    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "ok",
        "model_loaded": model_exists,
        "model_path": MODEL_PATH,
    }


@app.post("/upload", response_model=UploadSummary)
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a CSV file of student records.

    Validates that all 19 required feature columns are present.
    Runs predictions and SHAP explanations for every row.
    Stores results in Neon PostgreSQL.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    # ── Schema validation ──────────────────────────────────────────────────
    missing = sorted(set(REQUIRED_FEATURES) - set(df.columns))
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}",
        )

    batch_id = str(uuid.uuid4())[:8]
    student_ids: list[int] = []
    errors: list[str] = []

    try:
        predictions = predict_batch(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {exc}"
        ) from exc

    for i, (_, raw_row) in enumerate(df.iterrows()):
        try:
            record = _store_student(
                db,
                raw_row.to_dict(),
                predictions[i],
                batch_id=batch_id,
            )
            student_ids.append(record.id)
        except Exception as exc:
            errors.append(f"Row {i}: {exc}")

    db.commit()

    return UploadSummary(
        rows_processed=len(df),
        rows_stored=len(student_ids),
        batch_id=batch_id,
        student_ids=student_ids,
        errors=errors,
    )


@app.get("/student/{student_id}", response_model=StudentDiagnostic)
def get_student(student_id: int, db: Session = Depends(get_db)):
    """Retrieve the full diagnostic JSON for a single student by ID."""
    row = db.query(StudentData).filter(StudentData.id == student_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")
    return _row_to_diagnostic(row)


@app.get("/students", response_model=list[StudentDiagnostic])
def list_students(
    limit: int = 200,
    offset: int = 0,
    risk_level: str | None = None,
    db: Session = Depends(get_db),
):
    """List all stored student summaries, optionally filtered by risk_level."""
    query = db.query(StudentData)
    if risk_level:
        query = query.filter(StudentData.risk_level == risk_level)
    rows = query.order_by(StudentData.predicted_exam_score.asc()).offset(offset).limit(limit).all()
    return [_row_to_diagnostic(r) for r in rows]


@app.post("/predict", response_model=PredictionResult)
def predict_manual(
    request: PredictRequest,
    db: Session = Depends(get_db),
):
    """
    Accept a single student's features as JSON, run prediction + SHAP,
    store in DB, and return the diagnostic result.
    """
    features = request.model_dump(exclude={"student_name"})

    try:
        result = predict_single(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {exc}"
        ) from exc

    record = _store_student(
        db, features, result, student_name=request.student_name
    )
    db.commit()

    return PredictionResult(
        student_id=record.id,
        student_name=request.student_name,
        predicted_exam_score=result["predicted_exam_score"],
        risk_level=result["risk_level"],
        top_negative_factors=[
            ShapFactor(**f) for f in result["top_negative_factors"]
        ],
        created_at=record.created_at,
    )
