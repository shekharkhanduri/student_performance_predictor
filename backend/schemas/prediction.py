"""Request and response schemas for prediction endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


_DS1_REQUIRED = [
    "Hours_Studied",
    "Previous_Scores",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Sample_Question_Papers_Practiced",
]

_DS2_REQUIRED = [
    "Hours_Studied",
    "Attendance",
    "Gender",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
]

_DS2_ALLOWED_VALUES: dict[str, set[str]] = {
    "Gender": {"Male", "Female"},
    "Parental_Involvement": {"Low", "Medium", "High"},
    "Access_to_Resources": {"Low", "Medium", "High"},
    "Extracurricular_Activities": {"Yes", "No"},
    "Motivation_Level": {"Low", "Medium", "High"},
    "Internet_Access": {"Yes", "No"},
    "Family_Income": {"Low", "Medium", "High"},
    "Teacher_Quality": {"Low", "Medium", "High"},
    "School_Type": {"Public", "Private"},
    "Peer_Influence": {"Positive", "Neutral", "Negative"},
    "Learning_Disabilities": {"Yes", "No"},
    "Parental_Education_Level": {"High School", "College", "Postgraduate"},
    "Distance_from_Home": {"Near", "Moderate", "Far"},
}


class PredictRequest(BaseModel):
    student_id: int = Field(..., gt=0, description="Unique student identifier")
    dataset_type: Literal["ds1", "ds2"] = "ds2"
    student_name: str | None = None

    # Dataset 1
    Hours_Studied: float | None = Field(default=None, ge=0)
    Previous_Scores: float | None = Field(default=None, ge=0, le=100)
    Extracurricular_Activities: str | None = None
    Sleep_Hours: float | None = Field(default=None, ge=0)
    Sample_Question_Papers_Practiced: float | None = Field(default=None, ge=0)

    # Dataset 2
    Attendance: float | None = Field(default=None, ge=0, le=100)
    Gender: str | None = None
    Parental_Involvement: str | None = None
    Access_to_Resources: str | None = None
    Motivation_Level: str | None = None
    Internet_Access: str | None = None
    Tutoring_Sessions: float | None = Field(default=None, ge=0)
    Family_Income: str | None = None
    Teacher_Quality: str | None = None
    School_Type: str | None = None
    Peer_Influence: str | None = None
    Physical_Activity: float | None = Field(default=None, ge=0)
    Learning_Disabilities: str | None = None
    Parental_Education_Level: str | None = None
    Distance_from_Home: str | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_dataset_payload(self) -> "PredictRequest":
        payload = self.model_dump()
        required = _DS1_REQUIRED if self.dataset_type == "ds1" else _DS2_REQUIRED
        missing = [k for k in required if payload.get(k) is None]
        if missing:
            raise ValueError(
                f"Missing required features for {self.dataset_type}: {', '.join(missing)}"
            )

        if self.Extracurricular_Activities is not None and self.Extracurricular_Activities not in {"Yes", "No"}:
            raise ValueError("Extracurricular_Activities must be one of: Yes, No")

        if self.dataset_type == "ds2":
            for key, allowed in _DS2_ALLOWED_VALUES.items():
                value = payload.get(key)
                if value is None:
                    continue
                if value not in allowed:
                    allowed_text = ", ".join(sorted(allowed))
                    raise ValueError(f"{key} must be one of: {allowed_text}")

        return self


class ShapFactor(BaseModel):
    feature: str
    shap_value: float


class PredictionResult(BaseModel):
    student_id: int | None = None
    dataset_type: Literal["ds1", "ds2"] | None = None
    student_name: str | None = None
    predicted_exam_score: float
    risk_level: str
    top_negative_factors: list[ShapFactor]
    created_at: datetime | None = None


class StudentDiagnostic(BaseModel):
    student_id: int
    dataset_type: Literal["ds1", "ds2"] | None = None
    student_name: str | None = None
    predicted_exam_score: float
    risk_level: str
    top_negative_factors: list[Any]
    created_at: datetime | None = None
    features: dict[str, Any] | None = None


class UploadSummary(BaseModel):
    rows_processed: int
    rows_stored: int
    dataset_type: Literal["ds1", "ds2"]
    batch_id: str
    student_ids: list[int]
    errors: list[str] = []
