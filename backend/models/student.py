"""SQLAlchemy ORM models for student diagnostics.

Two tables mirror the two training datasets:
  student_data_ds1 — 5 features (Dataset 1, 10k samples)
  student_data_ds2 — 19 features (Dataset 2, 6.6k samples)

student_id
  Public student identifier used by APIs and dashboard.
  If provided in CSV/request, that value is used; otherwise it is auto-assigned
  from the DB primary key (id) after insert.

shap_status
  'pending' — scores stored, SHAP running in background.
  'done'    — SHAP explanations persisted in shap_explanations.
  'failed'  — SHAP computation raised an unhandled exception.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String

from backend.core.database import Base


class _StudentCommon:
    # ── Primary key (internal) ────────────────────────────────────────────
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # Public-facing ID, auto-set to id after insert (see student_service.py).
    student_id = Column(Integer, nullable=True, unique=True, index=True)

    # ── Prediction outputs ────────────────────────────────────────────────
    predicted_exam_score = Column(Float, nullable=True)
    risk_level = Column(String(16), nullable=True)
    shap_explanations = Column(JSON, nullable=True)
    shap_status = Column(String(8), nullable=False, server_default="pending")

    # ── Metadata ──────────────────────────────────────────────────────────
    student_name = Column(String(128), nullable=True)
    upload_batch = Column(String(64), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class StudentDataDS1(_StudentCommon, Base):
    __tablename__ = "student_data_ds1"

    hours_studied = Column(Float, nullable=True)
    previous_scores = Column(Float, nullable=True)
    extracurricular_activities = Column(String(4), nullable=True)
    sleep_hours = Column(Float, nullable=True)
    sample_question_papers_practiced = Column(Float, nullable=True)


class StudentDataDS2(_StudentCommon, Base):
    __tablename__ = "student_data_ds2"

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
