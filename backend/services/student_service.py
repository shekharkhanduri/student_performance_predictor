"""Helpers for mapping and persisting student diagnostics."""

from datetime import datetime, timezone
from typing import Literal

from sqlalchemy.orm import Session

from backend.models import StudentDataDS1, StudentDataDS2
from backend.schemas import StudentDiagnostic


DatasetType = Literal["ds1", "ds2"]


def row_to_diagnostic(
    row: StudentDataDS1 | StudentDataDS2,
    dataset_type: DatasetType,
) -> StudentDiagnostic:
    if dataset_type == "ds1":
        features = {
            "Hours_Studied": row.hours_studied,
            "Previous_Scores": row.previous_scores,
            "Extracurricular_Activities": row.extracurricular_activities,
            "Sleep_Hours": row.sleep_hours,
            "Sample_Question_Papers_Practiced": row.sample_question_papers_practiced,
        }
    else:
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
        student_id=row.student_id,
        dataset_type=dataset_type,
        student_name=row.student_name,
        predicted_exam_score=row.predicted_exam_score or 0.0,
        risk_level=row.risk_level or "Unknown",
        top_negative_factors=row.shap_explanations or [],
        created_at=row.created_at,
        features=features,
    )


def store_student(
    db: Session,
    raw_row: dict,
    result: dict,
    dataset_type: DatasetType,
    student_id: int,
    student_name: str | None = None,
    batch_id: str | None = None,
) -> StudentDataDS1 | StudentDataDS2:
    """Store or update student record. Updates existing student_id, inserts if new."""
    
    common_payload = {
        "student_id": student_id,
        "student_name": student_name or raw_row.get("student_name"),
        "upload_batch": batch_id,
        "predicted_exam_score": result["predicted_exam_score"],
        "risk_level": result["risk_level"],
        "shap_explanations": result["top_negative_factors"],
        "created_at": datetime.now(timezone.utc),
    }

    # Determine model and build feature payload
    if dataset_type == "ds1":
        model_class = StudentDataDS1
        feature_payload = {
            "hours_studied": raw_row.get("Hours_Studied"),
            "previous_scores": raw_row.get("Previous_Scores"),
            "extracurricular_activities": raw_row.get("Extracurricular_Activities"),
            "sleep_hours": raw_row.get("Sleep_Hours"),
            "sample_question_papers_practiced": raw_row.get("Sample_Question_Papers_Practiced"),
        }
    else:
        model_class = StudentDataDS2
        feature_payload = {
            "hours_studied": raw_row.get("Hours_Studied"),
            "attendance": raw_row.get("Attendance"),
            "gender": raw_row.get("Gender"),
            "parental_involvement": raw_row.get("Parental_Involvement"),
            "access_to_resources": raw_row.get("Access_to_Resources"),
            "extracurricular_activities": raw_row.get("Extracurricular_Activities"),
            "sleep_hours": raw_row.get("Sleep_Hours"),
            "previous_scores": raw_row.get("Previous_Scores"),
            "motivation_level": raw_row.get("Motivation_Level"),
            "internet_access": raw_row.get("Internet_Access"),
            "tutoring_sessions": raw_row.get("Tutoring_Sessions"),
            "family_income": raw_row.get("Family_Income"),
            "teacher_quality": raw_row.get("Teacher_Quality"),
            "school_type": raw_row.get("School_Type"),
            "peer_influence": raw_row.get("Peer_Influence"),
            "physical_activity": raw_row.get("Physical_Activity"),
            "learning_disabilities": raw_row.get("Learning_Disabilities"),
            "parental_education_level": raw_row.get("Parental_Education_Level"),
            "distance_from_home": raw_row.get("Distance_from_Home"),
        }

    # Check for existing record
    existing = db.query(model_class).filter(
        model_class.student_id == student_id
    ).first()

    if existing:
        # Update existing record
        for key, value in {**common_payload, **feature_payload}.items():
            setattr(existing, key, value)
        record = existing
    else:
        # Create new record
        record = model_class(**common_payload, **feature_payload)
        db.add(record)

    return record
