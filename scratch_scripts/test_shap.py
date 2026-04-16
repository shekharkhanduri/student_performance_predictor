import joblib
import pandas as pd
from backend.services.ml_service import build_explainer, compute_shap_explanations
from backend.core.config import DS1_MODEL_PATH, DS1_SCALER_PATH
import logging
logging.basicConfig(level=logging.INFO)

model = joblib.load(DS1_MODEL_PATH)
scaler = joblib.load(DS1_SCALER_PATH)

bundle = build_explainer(model, "ds1")
print("Explainer kind:", bundle.kind)

df = pd.DataFrame([{
    "Hours_Studied": 5,
    "Previous_Scores": 80,
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 7,
    "Sample_Question_Papers_Practiced": 3
}])

res = compute_shap_explanations(df, "ds1", model, scaler, bundle)
print("SHAP Results:", res)
