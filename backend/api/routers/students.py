"""Student prediction and retrieval endpoints."""

import io
import re
import uuid
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.api.deps import get_db
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
    detect_dataset_type,
    predict_batch,
    predict_single,
)
from backend.services.student_service import row_to_diagnostic, store_student


router = APIRouter(tags=["students"])


DATASET_MODELS = {
    "ds1": StudentDataDS1,
    "ds2": StudentDataDS2,
}


def _parse_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Parse uploaded CSV file."""
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            text = contents.decode("latin-1")
    
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def _normalize_col_name(name: str) -> str:
    """Normalize headers to snake_case-like form for resilient matching."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lstrip("\ufeff")).strip("_")
    return normalized.lower()


def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename incoming columns to required feature names using normalized matching."""
    all_required = DATASET1_FEATURES + DATASET2_FEATURES
    required_by_norm = {_normalize_col_name(col): col for col in all_required}
    rename_map: dict[str, str] = {}

    for col in df.columns:
        normalized = _normalize_col_name(col)
        canonical = required_by_norm.get(normalized)
        if canonical:
            rename_map[col] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@router.post("/upload", response_model=UploadSummary)
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = _parse_csv_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    df = _normalize_feature_columns(df)

    detected_dataset = detect_dataset_type(list(df.columns))
    if detected_dataset is None:
        incoming_cols = [str(c) for c in df.columns]
        missing_ds1 = sorted(set(DATASET1_FEATURES) - set(df.columns))
        missing_ds2 = sorted(set(DATASET2_FEATURES) - set(df.columns))
        raise HTTPException(
            status_code=400,
            detail={
                "message": "CSV does not match Dataset 1 or Dataset 2 required schema.",
                "missing_for_dataset1": missing_ds1,
                "missing_for_dataset2": missing_ds2,
                "received_columns": incoming_cols,
            },
        )

    batch_id = str(uuid.uuid4())[:8]
    student_ids: list[int] = []
    errors: list[str] = []

    try:
        predictions = predict_batch(df, dataset_type=detected_dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    for i, (_, raw_row) in enumerate(df.iterrows()):
        try:
            student_id = raw_row.get("student_id")
            if student_id is None:
                raise ValueError("student_id is required in CSV")
            student_id = int(student_id)
            if student_id <= 0:
                raise ValueError("student_id must be a positive integer")
            
            record = store_student(
                db,
                raw_row.to_dict(),
                predictions[i],
                dataset_type=detected_dataset,
                student_id=student_id,
                batch_id=batch_id,
            )
            student_ids.append(record.student_id)
        except Exception as exc:
            errors.append(f"Row {i}: {exc}")
            db.rollback()

    if student_ids:
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database commit failed: {exc}") from exc

    return UploadSummary(
        rows_processed=len(df),
        rows_stored=len(student_ids),
        dataset_type=detected_dataset,
        batch_id=batch_id,
        student_ids=student_ids,
        errors=errors,
    )


@router.get("/student/{student_id}", response_model=StudentDiagnostic)
def get_student(
    student_id: int,
    dataset_type: Literal["ds1", "ds2"] | None = None,
    db: Session = Depends(get_db),
):
    if dataset_type:
        model = DATASET_MODELS[dataset_type]
        row = db.query(model).filter(model.student_id == student_id).first()
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
                f"Student id {student_id} exists in both datasets. "
                "Call /student/{student_id}?dataset_type=ds1 or ds2."
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
    def _query_rows(model):
        query = db.query(model)
        if risk_level:
            query = query.filter(model.risk_level == risk_level)
        return query.order_by(model.predicted_exam_score.asc())

    if dataset_type:
        model = DATASET_MODELS[dataset_type]
        rows = _query_rows(model).offset(offset).limit(limit).all()
        return [row_to_diagnostic(r, dataset_type) for r in rows]

    ds1_rows = _query_rows(StudentDataDS1).limit(limit + offset).all()
    ds2_rows = _query_rows(StudentDataDS2).limit(limit + offset).all()

    combined = [row_to_diagnostic(r, "ds1") for r in ds1_rows] + [
        row_to_diagnostic(r, "ds2") for r in ds2_rows
    ]
    combined.sort(key=lambda r: r.predicted_exam_score)
    return combined[offset : offset + limit]


@router.post("/predict", response_model=PredictionResult)
def predict_manual(
    request: PredictRequest,
    db: Session = Depends(get_db),
):
    dataset_type = request.dataset_type
    feature_set = DATASET1_FEATURES if dataset_type == "ds1" else DATASET2_FEATURES
    request_payload = request.model_dump(exclude={"student_name", "dataset_type", "student_id"})
    features = {name: request_payload.get(name) for name in feature_set}

    try:
        result = predict_single(features, dataset_type=dataset_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    record = store_student(
        db,
        features,
        result,
        dataset_type=dataset_type,
        student_id=request.student_id,
        student_name=request.student_name,
    )
    
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    return PredictionResult(
        student_id=record.student_id,
        dataset_type=dataset_type,
        student_name=request.student_name,
        predicted_exam_score=result["predicted_exam_score"],
        risk_level=result["risk_level"],
        top_negative_factors=[ShapFactor(**f) for f in result["top_negative_factors"]],
        created_at=record.created_at,
    )
