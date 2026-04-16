import joblib
import pandas as pd
from backend.services.ml_service import build_explainer, compute_shap_explanations
from backend.core.config import DS2_MODEL_PATH, DS2_SCALER_PATH
import logging
logging.basicConfig(level=logging.INFO)

model = joblib.load(DS2_MODEL_PATH)
scaler = joblib.load(DS2_SCALER_PATH)

bundle = build_explainer(model, "ds2")
print("Explainer kind:", bundle.kind)

df = pd.DataFrame([{
    "Hours_Studied": 5,
    "Attendance": 80,
    "Parental_Involvement": "High",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 7,
    "Previous_Scores": 80,
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 1,
    "Family_Income": "High",
    "Teacher_Quality": "High",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Physical_Activity": 3,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "Gender": "Male"
}])

res = compute_shap_explanations(df, "ds2", model, scaler, bundle)
print("SHAP Results:", res)
