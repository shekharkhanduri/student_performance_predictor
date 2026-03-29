"""
Machine-learning utilities for the Faculty Student Diagnostic System.

Handles:
- Loading the saved StackingRegressor from outputs/
- Preprocessing (label-encoding matching training-time behaviour)
- Prediction with Exam_Score clipping at 100
- SHAP KernelExplainer with synthetic background data
- Risk categorisation and top-3 negative factor extraction
"""

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

# ── Feature configuration ─────────────────────────────────────────────────────

REQUIRED_FEATURES: list[str] = [
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

# Sorted alphabetically (matching sklearn LabelEncoder.fit behaviour)
CATEGORICAL_CLASSES: dict[str, list[str]] = {
    "Gender": ["Female", "Male"],
    "Parental_Involvement": ["High", "Low", "Medium"],
    "Access_to_Resources": ["High", "Low", "Medium"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level": ["High", "Low", "Medium"],
    "Internet_Access": ["No", "Yes"],
    "Family_Income": ["High", "Low", "Medium"],
    "Teacher_Quality": ["High", "Low", "Medium"],
    "School_Type": ["Private", "Public"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["College", "High School", "Postgraduate"],
    "Distance_from_Home": ["Far", "Moderate", "Near"],
}

NUMERIC_FEATURES: list[str] = [
    f for f in REQUIRED_FEATURES if f not in CATEGORICAL_CLASSES
]

_LABEL_ENCODERS: dict[str, LabelEncoder] = {}
for _col, _classes in CATEGORICAL_CLASSES.items():
    _le = LabelEncoder()
    _le.fit(_classes)
    _LABEL_ENCODERS[_col] = _le

# ── Model paths ───────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(_REPO_ROOT, "outputs", "ds2_all_features_stacking_regressor.joblib")

# ── Module-level singletons ───────────────────────────────────────────────────

_model = None
_shap_explainer = None
_shap_background: np.ndarray | None = None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load (and cache) the StackingRegressor from disk."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH!r}. "
                "Run `python student_performance_predictor.py` first to train and export the model."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> np.ndarray:
    """
    Apply the same preprocessing as training time for Dataset 2:
      - Mode-impute missing categorical values
      - Mean-impute missing numeric values
      - Label-encode categorical columns (alphabetical order, matching sklearn default)

    Returns a 2-D float array shaped (n_rows, len(REQUIRED_FEATURES)).
    """
    df = df.copy()

    # Impute categorical columns
    for col, classes in CATEGORICAL_CLASSES.items():
        if col in df.columns:
            df[col] = df[col].fillna(classes[0])

    # Impute numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            mean_val = series.mean() if series.notna().any() else 0.0
            df[col] = series.fillna(mean_val)

    # Label-encode categorical columns
    for col, le in _LABEL_ENCODERS.items():
        if col in df.columns:
            known = set(le.classes_)

            def _safe_encode(x: str, _le: LabelEncoder = le, _known: set = known) -> int:
                return int(_le.transform([x])[0]) if x in _known else 0

            df[col] = df[col].astype(str).apply(_safe_encode)

    return df[REQUIRED_FEATURES].astype(float).values


# ── SHAP utilities ────────────────────────────────────────────────────────────

def _build_synthetic_background(n: int = 50) -> np.ndarray:
    """
    Generate a plausible synthetic background dataset (label-encoded) for
    shap.KernelExplainer.  Uses the same numeric ranges as the training data
    generator so SHAP baseline values are realistic.
    """
    rng = np.random.default_rng(42)
    rows: list[list[float]] = []
    for _ in range(n):
        row: dict[str, float] = {
            "Hours_Studied": float(rng.uniform(1, 44)),
            "Attendance": float(rng.uniform(60, 100)),
            "Gender": float(rng.integers(0, 2)),
            "Parental_Involvement": float(rng.integers(0, 3)),
            "Access_to_Resources": float(rng.integers(0, 3)),
            "Extracurricular_Activities": float(rng.integers(0, 2)),
            "Sleep_Hours": float(rng.uniform(4, 10)),
            "Previous_Scores": float(rng.uniform(50, 100)),
            "Motivation_Level": float(rng.integers(0, 3)),
            "Internet_Access": float(rng.integers(0, 2)),
            "Tutoring_Sessions": float(rng.integers(0, 8)),
            "Family_Income": float(rng.integers(0, 3)),
            "Teacher_Quality": float(rng.integers(0, 3)),
            "School_Type": float(rng.integers(0, 2)),
            "Peer_Influence": float(rng.integers(0, 3)),
            "Physical_Activity": float(rng.integers(0, 6)),
            "Learning_Disabilities": float(rng.integers(0, 2)),
            "Parental_Education_Level": float(rng.integers(0, 3)),
            "Distance_from_Home": float(rng.integers(0, 3)),
        }
        rows.append([row[f] for f in REQUIRED_FEATURES])
    return np.array(rows, dtype=float)


def get_shap_explainer(background: np.ndarray | None = None):
    """Return (and cache) a KernelExplainer; rebuild when new background is provided."""
    global _shap_explainer, _shap_background
    model = load_model()
    bg = background if background is not None else _build_synthetic_background()
    n_bg = min(50, len(bg))
    bg = bg[:n_bg]
    if _shap_explainer is None or (background is not None):
        _shap_explainer = shap.KernelExplainer(model.predict, bg)
        _shap_background = bg
    return _shap_explainer


def compute_shap_values(X: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    """Compute SHAP values for all rows in X."""
    explainer = get_shap_explainer(background)
    sv = explainer.shap_values(X, nsamples=100)
    return np.array(sv)


# ── Risk categorisation ───────────────────────────────────────────────────────

def categorize_risk(score: float) -> str:
    if score >= 70:
        return "Stable"
    if score >= 60:
        return "Borderline"
    return "At-Risk"


# ── SHAP factor extraction ────────────────────────────────────────────────────

_ADVICE_TEMPLATES: dict[str, str] = {
    "Attendance": "Improve class attendance — aim for ≥ 85%.",
    "Hours_Studied": "Increase daily study time with structured sessions.",
    "Previous_Scores": "Address foundational knowledge gaps with targeted revision.",
    "Sleep_Hours": "Ensure 7–9 hours of sleep nightly for better retention.",
    "Tutoring_Sessions": "Enrol in additional tutoring or remedial classes.",
    "Physical_Activity": "Regular physical activity supports cognitive performance.",
    "Motivation_Level": "Work with a counsellor to improve academic motivation.",
    "Parental_Involvement": "Encourage greater parental engagement in studies.",
    "Access_to_Resources": "Provide access to textbooks, internet, and study materials.",
    "Internet_Access": "Schedule meeting to discuss internet access — provide offline packs.",
    "Family_Income": "Explore scholarships or financial-aid programmes.",
    "Peer_Influence": "Connect the student with positive peer study groups.",
    "Teacher_Quality": "Request teacher support or mentoring sessions.",
    "Learning_Disabilities": "Arrange special-education support if needed.",
    "Distance_from_Home": "Explore transport assistance or nearby study centres.",
    "School_Type": "Leverage available school resources and programmes.",
    "Gender": "Ensure gender-neutral learning environment and support.",
    "Extracurricular_Activities": "Balance extracurricular commitments with study time.",
    "Parental_Education_Level": "Provide academic guidance beyond what parents can offer.",
}


def get_top_negative_factors(
    shap_values: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
    """
    Extract the top-3 most negatively impactful features from a SHAP value
    row (already flattened to 1-D).  Returns a list of dicts with keys
    ``feature``, ``shap_value``, and ``description``.
    """
    sv = np.array(shap_values).flatten()
    neg_mask = sv < 0
    if neg_mask.any():
        neg_idx = np.where(neg_mask)[0]
        top_neg_idx = neg_idx[np.argsort(sv[neg_idx])]
        if len(top_neg_idx) < 3:
            pos_idx = np.where(~neg_mask)[0]
            pad = pos_idx[np.argsort(sv[pos_idx])]
            top_neg_idx = np.concatenate([top_neg_idx, pad])
        top_neg_idx = top_neg_idx[:3]
    else:
        top_neg_idx = np.argsort(sv)[:3]

    return [
        {
            "feature": feature_names[i],
            "shap_value": float(sv[i]),
            "description": _ADVICE_TEMPLATES.get(feature_names[i], ""),
        }
        for i in top_neg_idx
    ]


# ── High-level inference ──────────────────────────────────────────────────────

def predict_batch(
    df: pd.DataFrame,
    use_df_as_background: bool = True,
) -> list[dict[str, Any]]:
    """
    Run predictions and SHAP explanations for every row in *df*.

    Parameters
    ----------
    df                    : DataFrame with the 19 required feature columns.
    use_df_as_background  : If True, use a sample from df as SHAP background.

    Returns a list of dicts (one per row) with prediction metadata.
    """
    model = load_model()
    X = encode_features(df)

    # Build explainer background
    if use_df_as_background and len(X) >= 5:
        background = X
    else:
        background = None  # will use synthetic background

    # Predictions (clip at 100)
    raw_preds = model.predict(X)
    preds = np.clip(raw_preds, 0.0, 100.0)

    # Batch SHAP values
    shap_vals = compute_shap_values(X, background)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    results: list[dict[str, Any]] = []
    for i in range(len(X)):
        sv_row = shap_vals[i] if shap_vals.ndim == 2 else shap_vals[:, i]
        score = float(preds[i])
        results.append(
            {
                "predicted_exam_score": round(score, 4),
                "risk_level": categorize_risk(score),
                "top_negative_factors": get_top_negative_factors(sv_row, REQUIRED_FEATURES),
            }
        )
    return results


def predict_single(features: dict[str, Any]) -> dict[str, Any]:
    """Run prediction + SHAP for a single student (manual entry)."""
    df = pd.DataFrame([features])
    return predict_batch(df, use_df_as_background=False)[0]
