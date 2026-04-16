"""Student prediction and retrieval endpoints.

POST /upload
  Parses a CSV, validates column schema, enforces MAX_UPLOAD_ROWS,
  runs predict_scores() synchronously, stores records with shap_status='pending',
  then enqueues compute_shap_explanations() as a BackgroundTask.
  Returns HTTP 202 immediately so the client is not blocked.

GET /student/{student_id}
  Returns the full diagnostic (features + score + SHAP factors).
  If shap_status='pending', SHAP factors will be empty — clients should
  poll every few seconds until shap_status='done'.

GET /students
  Paginated list filterable by risk_level and dataset_type.

POST /predict
  Single-student prediction (runs SHAP synchronously — acceptable for 1 row).
"""

from __future__ import annotations

import io
import logging
import re
import uuid
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from backend.api.deps import get_db, get_explainers, get_models, get_scalers
from backend.core.config import MAX_UPLOAD_ROWS
from backend.core.database import SessionLocal
from backend.models import StudentDataDS1, StudentDataDS2
from backend.schemas import (
    PredictRequest,
    PredictionResult,
    ShapFactor,
    StudentDiagnostic,
    UploadSummary,
)
from backend.services.ml_service import (
    DATASET1_FEATURES,
    DATASET2_FEATURES,
    DATASET_FEATURES,
    ExplainerBundle,
    compute_shap_explanations,
    detect_dataset_type,
    predict_scores,
    predict_with_shap,
)
from backend.services.student_service import (
    row_to_diagnostic,
    store_student,
    update_student_shap,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["students"])

_DATASET_MODELS: dict[str, type] = {
    "ds1": StudentDataDS1,
    "ds2": StudentDataDS2,
}


# ── CSV parsing helpers ────────────────────────────────────────────────────────

def _parse_csv_bytes(contents: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = contents.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def _normalize_col_name(name: str) -> str:
    """Lowercase snake_case normalisation for resilient column matching."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lstrip("\ufeff")).strip("_").lower()


def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename incoming columns to canonical feature names.

    Keeps student_id and student_name as metadata columns when present.
    """
    meta = ["student_id", "student_name"]
    all_required = DATASET1_FEATURES + DATASET2_FEATURES + meta
    required_by_norm = {_normalize_col_name(col): col for col in all_required}
    rename_map: dict[str, str] = {}

    for col in df.columns:
        canonical = required_by_norm.get(_normalize_col_name(col))
        if canonical:
            rename_map[col] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ── Background SHAP task ───────────────────────────────────────────────────────

def _shap_background_task(
    record_ids: list[int],
    df_records: list[dict],
    dataset_type: str,
    model: Any,
    scaler: Any,
    bundle: ExplainerBundle,
) -> None:
    """
    Compute SHAP explanations and persist them to the DB.

    Uses the pre-built ExplainerBundle (TreeExplainer when available) so this
    typically runs in milliseconds.  Runs in a thread pool after the HTTP
    response is sent.  Creates its own DB session.
    """
    db = SessionLocal()
    try:
        df = pd.DataFrame(df_records)
        logger.info(
            "[SHAP] Starting background task for %d records (dataset=%s, kind=%s)",
            len(record_ids), dataset_type, bundle.kind,
        )
        shap_results = compute_shap_explanations(df, dataset_type, model, scaler, bundle)
        for record_id, factors in zip(record_ids, shap_results):
            update_student_shap(db, record_id, dataset_type, factors)
        logger.info("[SHAP] Background task complete for %d records.", len(record_ids))
    except Exception as exc:  # noqa: BLE001
        logger.error("[SHAP] Background task failed: %s", exc, exc_info=True)
        # Mark all records as failed so the client stops polling.
        model_class = _DATASET_MODELS[dataset_type]
        from datetime import datetime, timezone
        for record_id in record_ids:
            try:
                row = db.query(model_class).filter(model_class.id == record_id).first()
                if row:
                    row.shap_status = "failed"
                    row.updated_at = datetime.now(timezone.utc)
            except Exception:  # noqa: BLE001
                pass
        try:
            db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
    finally:
        db.close()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadSummary, status_code=202)
async def upload_csv(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    models: dict = Depends(get_models),
    scalers: dict = Depends(get_scalers),
    explainers: dict = Depends(get_explainers),
):
    """
    Upload a CSV to batch-predict all students.

    Returns HTTP 202 immediately.  SHAP computation continues in the background;
    poll GET /student/{id} to check shap_status.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = _parse_csv_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    if len(df) > MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV exceeds the {MAX_UPLOAD_ROWS}-row limit. "
                "Split the file and upload in batches."
            ),
        )

    df = _normalize_feature_columns(df)
    detected_dataset = detect_dataset_type(list(df.columns))
    if detected_dataset is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "CSV does not match Dataset 1 or Dataset 2 required schema.",
                "missing_for_dataset1": sorted(set(DATASET1_FEATURES) - set(df.columns)),
                "missing_for_dataset2": sorted(set(DATASET2_FEATURES) - set(df.columns)),
                "received_columns": list(df.columns),
            },
        )

    model = models[detected_dataset]
    scaler = scalers[detected_dataset]
    batch_id = str(uuid.uuid4())[:8]

    # ── Fast path: predict scores (synchronous) ────────────────────────────
    try:
        score_results = predict_scores(df, dataset_type=detected_dataset, model=model, scaler=scaler)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    student_ids: list[int] = []
    errors: list[str] = []
    record_ids_for_shap: list[int] = []
    feature_dicts: list[dict] = []

    for i, (_, raw_row) in enumerate(df.iterrows()):
        try:
            s_name = str(raw_row["student_name"]) if "student_name" in raw_row and raw_row["student_name"] is not None else None
            s_id = int(raw_row["student_id"]) if "student_id" in raw_row and raw_row["student_id"] is not None and pd.notna(raw_row["student_id"]) else None

            record = store_student(
                db,
                raw_row.to_dict(),
                score_results[i],
                dataset_type=detected_dataset,
                student_name=s_name,
                batch_id=batch_id,
                shap_status="pending",
                student_id_override=s_id,
            )
            db.flush()
            student_ids.append(record.student_id)
            record_ids_for_shap.append(record.id)
            feature_dicts.append(raw_row.to_dict())
        except Exception as exc:
            errors.append(f"Row {i}: {exc}")
            db.rollback()

    if student_ids:
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database commit failed: {exc}") from exc

    # ── Slow path: SHAP (background — after response is sent) ─────────────
    if record_ids_for_shap:
        background_tasks.add_task(
            _shap_background_task,
            record_ids=record_ids_for_shap,
            df_records=feature_dicts,
            dataset_type=detected_dataset,
            model=model,
            scaler=scaler,
            bundle=explainers[detected_dataset],
        )

    return UploadSummary(
        rows_processed=len(df),
        rows_stored=len(student_ids),
        dataset_type=detected_dataset,
        batch_id=batch_id,
        student_ids=student_ids,
        shap_status="pending",
        errors=errors,
    )


@router.get("/student/{student_id}", response_model=StudentDiagnostic)
def get_student(
    student_id: int,
    dataset_type: Literal["ds1", "ds2"] | None = None,
    db: Session = Depends(get_db),
):
    """
    Return the full diagnostic for one student.

    If shap_status='pending', SHAP explanations are still being computed —
    the client should poll again in a few seconds.
    """
    if dataset_type:
        model_class = _DATASET_MODELS[dataset_type]
        row = db.query(model_class).filter(model_class.student_id == student_id).first()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Student {student_id} not found in dataset {dataset_type}.",
            )
        return row_to_diagnostic(row, dataset_type)

    ds2_row = db.query(StudentDataDS2).filter(StudentDataDS2.student_id == student_id).first()
    ds1_row = db.query(StudentDataDS1).filter(StudentDataDS1.student_id == student_id).first()

    if ds1_row and ds2_row:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Student {student_id} exists in both datasets. "
                "Add ?dataset_type=ds1 or ?dataset_type=ds2 to disambiguate."
            ),
        )
    if ds2_row is not None:
        return row_to_diagnostic(ds2_row, "ds2")
    if ds1_row is not None:
        return row_to_diagnostic(ds1_row, "ds1")

    raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")


@router.get("/students", response_model=list[StudentDiagnostic])
def list_students(
    limit: int = 200,
    offset: int = 0,
    risk_level: str | None = None,
    dataset_type: Literal["ds1", "ds2"] | None = None,
    db: Session = Depends(get_db),
):
    """Return a paginated, optionally-filtered list of all stored students."""

    def _query(model_class):
        q = db.query(model_class)
        if risk_level:
            q = q.filter(model_class.risk_level == risk_level)
        return q.order_by(model_class.predicted_exam_score.asc())

    if dataset_type:
        rows = _query(_DATASET_MODELS[dataset_type]).offset(offset).limit(limit).all()
        return [row_to_diagnostic(r, dataset_type) for r in rows]

    # Merge both tables, sort combined list, then slice.
    ds1_rows = _query(StudentDataDS1).limit(limit + offset).all()
    ds2_rows = _query(StudentDataDS2).limit(limit + offset).all()
    combined = [row_to_diagnostic(r, "ds1") for r in ds1_rows] + [
        row_to_diagnostic(r, "ds2") for r in ds2_rows
    ]
    combined.sort(key=lambda r: r.predicted_exam_score)
    return combined[offset: offset + limit]


@router.post("/predict", response_model=PredictionResult)
def predict_manual(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    models: dict = Depends(get_models),
    scalers: dict = Depends(get_scalers),
    explainers: dict = Depends(get_explainers),
):
    """
    Single-student prediction with immediate SHAP.

    SHAP runs synchronously here — one row takes < 10 s which is acceptable
    for interactive use.  The result is stored with shap_status='done'.
    """
    dataset_type = payload.dataset_type
    feature_set = DATASET1_FEATURES if dataset_type == "ds1" else DATASET2_FEATURES
    raw_fields = payload.model_dump(exclude={"student_id", "student_name", "dataset_type"})
    features = {name: raw_fields.get(name) for name in feature_set}

    try:
        result = predict_with_shap(
            pd.DataFrame([features]),
            dataset_type=dataset_type,
            model=models[dataset_type],
            scaler=scalers[dataset_type],
            bundle=explainers[dataset_type],
        )[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    record = store_student(
        db,
        features,
        result,
        dataset_type=dataset_type,
        student_name=payload.student_name,
        shap_status="done",
        top_negative_factors=result.get("top_negative_factors", []),
        student_id_override=payload.student_id,
    )

    try:
        db.flush()
        if payload.student_id is None:
            record.student_id = record.id
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    return PredictionResult(
        student_id=record.student_id,
        dataset_type=dataset_type,
        student_name=payload.student_name,
        predicted_exam_score=result["predicted_exam_score"],
        risk_level=result["risk_level"],
        shap_status="done",
        top_negative_factors=[ShapFactor(**f) for f in result.get("top_negative_factors", [])],
        created_at=record.created_at,
    )
