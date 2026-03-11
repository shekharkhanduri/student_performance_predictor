import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

FEATURES = [
    'cgpa', 'gpa', 'gpa_slope', 'midterm_score', 'attendance_pct',
    'avg_assignment_score', 'backlogs', 'avg_quiz_score',
    'assignment_completion_rate', 'prev_semester_score',
    'absences', 'late_submissions', 'extracurricular'
]

BENCHMARK = {
    'cgpa': 8.5,
    'gpa': 8.8,
    'gpa_slope': 0.4,
    'midterm_score': 85.0,
    'attendance_pct': 88.0,
    'avg_assignment_score': 87.0,
    'backlogs': 0.0,
    'avg_quiz_score': 84.0,
    'assignment_completion_rate': 92.0,
    'prev_semester_score': 8.4,
    'absences': 2.0,
    'late_submissions': 1.0,
    'extracurricular': 1.0,
}

INVERTED = {'backlogs', 'absences', 'late_submissions'}

FEATURE_RANGES = {
    'cgpa': 10.0, 'gpa': 10.0, 'gpa_slope': 1.0, 'midterm_score': 100.0,
    'attendance_pct': 100.0, 'avg_assignment_score': 100.0, 'backlogs': 8.0,
    'avg_quiz_score': 100.0, 'assignment_completion_rate': 100.0,
    'prev_semester_score': 10.0, 'absences': 30.0, 'late_submissions': 15.0,
    'extracurricular': 1.0,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'student_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

app = FastAPI(title="Student Performance ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None


@app.on_event("startup")
def load_model():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model files not found — training now...")
        from train_model import train_and_save
        train_and_save(MODEL_PATH, SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded.")


class StudentFeatures(BaseModel):
    cgpa: float
    gpa: float
    gpa_slope: float
    midterm_score: float
    attendance_pct: float
    avg_assignment_score: float
    backlogs: float
    avg_quiz_score: float
    assignment_completion_rate: float
    prev_semester_score: float
    absences: float
    late_submissions: float
    extracurricular: float


def features_to_array(s: StudentFeatures) -> np.ndarray:
    return np.array([[getattr(s, f) for f in FEATURES]])


def predict_score(feature_array: np.ndarray) -> float:
    scaled = scaler.transform(feature_array)
    pred = model.predict(scaled)[0]
    return float(round(max(0.0, min(10.0, pred)), 2))


@app.post("/predict")
def predict(student: StudentFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        arr = features_to_array(student)
        score = predict_score(arr)
        return {"predicted_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend(student: StudentFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        student_dict = student.model_dump()
        current_arr = features_to_array(student)
        current_score = predict_score(current_arr)

        gaps = {}
        for feat in FEATURES:
            val = student_dict[feat]
            bench = BENCHMARK[feat]
            feat_range = FEATURE_RANGES[feat]
            if feat in INVERTED:
                raw_gap = val - bench
            else:
                raw_gap = bench - val
            normalized_gap = raw_gap / feat_range if feat_range > 0 else 0.0
            gaps[feat] = max(normalized_gap, 0.0)

        top3 = sorted(gaps, key=lambda f: gaps[f], reverse=True)[:3]

        recommendations = []
        for feat in top3:
            val = student_dict[feat]
            bench = BENCHMARK[feat]
            if feat in INVERTED:
                suggested = val - 0.4 * (val - bench)
            else:
                suggested = val + 0.4 * (bench - val)

            improved = dict(student_dict)
            improved[feat] = suggested
            improved_arr = np.array([[improved[f] for f in FEATURES]])
            improved_score = predict_score(improved_arr)
            delta = round(improved_score - current_score, 2)

            recommendations.append({
                "feature": feat,
                "current_value": round(val, 2),
                "benchmark_value": bench,
                "suggested_value": round(suggested, 2),
                "score_improvement": delta,
            })

        all_improved = dict(student_dict)
        for feat in top3:
            val = all_improved[feat]
            bench = BENCHMARK[feat]
            if feat in INVERTED:
                all_improved[feat] = val - 0.4 * (val - bench)
            else:
                all_improved[feat] = val + 0.4 * (bench - val)

        potential_arr = np.array([[all_improved[f] for f in FEATURES]])
        potential_score = predict_score(potential_arr)

        return {
            "current_score": current_score,
            "potential_score": potential_score,
            "recommendations": recommendations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
