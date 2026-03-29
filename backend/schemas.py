"""
Pydantic schemas for the Faculty Student Diagnostic System API.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Shared feature schema ──────────────────────────────────────────────────────

class StudentFeatures(BaseModel):
    Hours_Studied: float = Field(..., ge=0)
    Attendance: float = Field(..., ge=0, le=100)
    Gender: str
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: float = Field(..., ge=0)
    Previous_Scores: float = Field(..., ge=0, le=100)
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: float = Field(..., ge=0)
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Physical_Activity: float = Field(..., ge=0)
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str

    model_config = {"extra": "allow"}


# ── Request: manual single prediction ─────────────────────────────────────────

class PredictRequest(StudentFeatures):
    student_name: str | None = None


# ── Response: SHAP factor item ─────────────────────────────────────────────────

class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    description: str | None = None


# ── Response: prediction result ────────────────────────────────────────────────

class PredictionResult(BaseModel):
    student_id: int | None = None
    student_name: str | None = None
    predicted_exam_score: float
    risk_level: str
    top_negative_factors: list[ShapFactor]
    created_at: datetime | None = None


# ── Response: full student diagnostic ─────────────────────────────────────────

class StudentDiagnostic(BaseModel):
    student_id: int
    student_name: str | None = None
    predicted_exam_score: float
    risk_level: str
    top_negative_factors: list[Any]
    created_at: datetime | None = None
    features: dict[str, Any] | None = None


# ── Response: upload summary ───────────────────────────────────────────────────

class UploadSummary(BaseModel):
    rows_processed: int
    rows_stored: int
    batch_id: str
    student_ids: list[int]
    errors: list[str] = []
