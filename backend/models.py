"""
SQLAlchemy ORM model for the StudentData table.
Stores input features, predictions, SHAP explanations, and metadata.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

from .database import Base


class StudentData(Base):
    __tablename__ = "student_data"

    id = Column(Integer, primary_key=True, index=True)

    # ── Input features (Dataset 2 — 19 features) ──────────────────────────
    hours_studied = Column(Float, nullable=True)
    attendance = Column(Float, nullable=True)
    gender = Column(String(16), nullable=True)
    parental_involvement = Column(String(16), nullable=True)
    access_to_resources = Column(String(16), nullable=True)
    extracurricular_activities = Column(String(4), nullable=True)
    sleep_hours = Column(Float, nullable=True)
    previous_scores = Column(Float, nullable=True)
    motivation_level = Column(String(16), nullable=True)
    internet_access = Column(String(4), nullable=True)
    tutoring_sessions = Column(Float, nullable=True)
    family_income = Column(String(16), nullable=True)
    teacher_quality = Column(String(16), nullable=True)
    school_type = Column(String(16), nullable=True)
    peer_influence = Column(String(16), nullable=True)
    physical_activity = Column(Float, nullable=True)
    learning_disabilities = Column(String(4), nullable=True)
    parental_education_level = Column(String(32), nullable=True)
    distance_from_home = Column(String(16), nullable=True)

    # ── Prediction outputs ─────────────────────────────────────────────────
    predicted_exam_score = Column(Float, nullable=True)
    risk_level = Column(String(16), nullable=True)

    # ── SHAP explanations (top 3 negative factors) ─────────────────────────
    shap_explanations = Column(JSON, nullable=True)

    # ── Metadata ───────────────────────────────────────────────────────────
    student_name = Column(String(128), nullable=True)
    upload_batch = Column(String(64), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
