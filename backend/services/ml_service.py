"""ML inference utilities for Dataset 1 and Dataset 2.

Design notes
────────────
• Models and scalers are loaded once at startup (backend/main.py lifespan) and
  stored in app.state.  They are passed explicitly to every function here —
  there are NO module-level mutable singletons.

• The prediction pipeline is split into two independent stages:

    1. predict_scores()          — fast path (< 1 s per batch)
       Encodes features → runs the StackingRegressor → returns scores +
       risk levels.  Called synchronously on every upload/predict request.

    2. compute_shap_explanations() — slow path (seconds to minutes per batch)
       Builds a KernelExplainer and computes SHAP values.  Called from a
       background task AFTER the HTTP response is already sent.

• predict_with_shap() is a convenience wrapper that runs both stages
  sequentially — used for single-student /predict where the latency is
  acceptable.

• build_explainer() selects the fastest available strategy:
    tree   — shap.TreeExplainer on XGBoost / CatBoost / RF base estimator.
             Runs in milliseconds and is exact.
    kernel — shap.KernelExplainer with n=20 background rows when no tree
             model is found.  nsamples is capped at 100.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder


# ── Feature definitions ───────────────────────────────────────────────────────

DATASET1_FEATURES: list[str] = [
    "Hours_Studied",
    "Previous_Scores",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Sample_Question_Papers_Practiced",
]

DATASET2_FEATURES: list[str] = [
    "Hours_Studied",
    "Attendance",
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
    "Gender",
]

DATASET_FEATURES: dict[str, list[str]] = {
    "ds1": DATASET1_FEATURES,
    "ds2": DATASET2_FEATURES,
}

DS2_CATEGORICAL_CLASSES: dict[str, list[str]] = {
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

DS2_NUMERIC_FEATURES: list[str] = [
    f for f in DATASET2_FEATURES if f not in DS2_CATEGORICAL_CLASSES
]

# Module-level label encoders are fine — they are stateless after fitting.
_DS2_LABEL_ENCODERS: dict[str, LabelEncoder] = {}
for _col, _classes in DS2_CATEGORICAL_CLASSES.items():
    _le = LabelEncoder()
    _le.fit(_classes)
    _DS2_LABEL_ENCODERS[_col] = _le

_ds1_yes_no_encoder = LabelEncoder()
_ds1_yes_no_encoder.fit(["No", "Yes"])


# ── Dataset detection ─────────────────────────────────────────────────────────

def detect_dataset_type(columns: list[str]) -> str | None:
    col_set = set(columns)
    if set(DATASET2_FEATURES).issubset(col_set):
        return "ds2"
    if set(DATASET1_FEATURES).issubset(col_set):
        return "ds1"
    return None


# ── Encoding ──────────────────────────────────────────────────────────────────

def _encode_dataset1(df: pd.DataFrame, scaler: Any) -> np.ndarray:
    working = df[DATASET1_FEATURES].copy()
    working["Extracurricular_Activities"] = (
        working["Extracurricular_Activities"].astype(str).fillna("No")
    )
    working["Extracurricular_Activities"] = working["Extracurricular_Activities"].apply(
        lambda v: int(_ds1_yes_no_encoder.transform([v])[0]) if v in {"Yes", "No"} else 0
    )
    for col in [c for c in DATASET1_FEATURES if c != "Extracurricular_Activities"]:
        series = pd.to_numeric(working[col], errors="coerce")
        working[col] = series.fillna(series.mean() if series.notna().any() else 0.0)
    return scaler.transform(working[DATASET1_FEATURES].astype(float))


def _encode_dataset2(df: pd.DataFrame, scaler: Any) -> np.ndarray:
    working = df.copy()
    for col, classes in DS2_CATEGORICAL_CLASSES.items():
        if col in working.columns:
            working[col] = working[col].fillna(classes[0])
    for col in DS2_NUMERIC_FEATURES:
        if col in working.columns:
            series = pd.to_numeric(working[col], errors="coerce")
            working[col] = series.fillna(series.mean() if series.notna().any() else 0.0)
    for col, le in _DS2_LABEL_ENCODERS.items():
        if col in working.columns:
            known = set(le.classes_)

            def _safe_encode(x: str, _le: LabelEncoder = le, _known: set = known) -> int:
                return int(_le.transform([x])[0]) if x in _known else 0

            working[col] = working[col].astype(str).apply(_safe_encode)
    return scaler.transform(working[DATASET2_FEATURES].astype(float))


def encode_features(df: pd.DataFrame, dataset_type: str, scaler: Any) -> np.ndarray:
    if dataset_type == "ds1":
        return _encode_dataset1(df, scaler)
    return _encode_dataset2(df, scaler)


# ── Risk categorisation ───────────────────────────────────────────────────────

def categorize_risk(score: float) -> str:
    if score >= 70:
        return "Stable"
    if score >= 60:
        return "Borderline"
    return "At-Risk"


# ── SHAP helpers ──────────────────────────────────────────────────────────────

class ExplainerBundle(NamedTuple):
    """Wraps an explainer with its type so callers know how to invoke it."""
    explainer: Any
    kind: str  # 'tree' | 'kernel'


def build_explainer(model: Any, dataset_type: str) -> ExplainerBundle:
    """
    Build the fastest available SHAP explainer for a stacking regressor.

    Strategy (fastest → slowest):
      1. shap.TreeExplainer on the XGBoost base estimator  — exact, milliseconds.
      2. shap.TreeExplainer on any tree base estimator     — exact, milliseconds.
      3. shap.KernelExplainer with a tiny background       — approximate, bounded.

    Returns an ExplainerBundle so compute_shap_explanations knows whether to
    pass nsamples (KernelExplainer requires it; TreeExplainer does not accept it).
    """
    log = logging.getLogger(__name__)
    for estimator in getattr(model, "estimators_", []):
        # XGBoost — preferred (fastest tree explainer)
        try:
            import xgboost as xgb  # noqa: PLC0415
            if isinstance(estimator, xgb.XGBRegressor):
                log.info("[SHAP] Using TreeExplainer on XGBoost for %s", dataset_type)
                return ExplainerBundle(shap.TreeExplainer(estimator), "tree")
        except Exception:  # noqa: BLE001
            pass
        # CatBoost
        try:
            import catboost as cb  # noqa: PLC0415
            if isinstance(estimator, cb.CatBoostRegressor):
                log.info("[SHAP] Using TreeExplainer on CatBoost for %s", dataset_type)
                return ExplainerBundle(shap.TreeExplainer(estimator), "tree")
        except Exception:  # noqa: BLE001
            pass
        # Random Forest (sklearn)
        try:
            from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415
            if isinstance(estimator, RandomForestRegressor):
                log.info("[SHAP] Using TreeExplainer on RandomForest for %s", dataset_type)
                return ExplainerBundle(shap.TreeExplainer(estimator), "tree")
        except Exception:  # noqa: BLE001
            pass

    # Fallback: KernelExplainer with a very small background to bound latency.
    log.warning(
        "[SHAP] No tree base estimator found for %s — falling back to KernelExplainer (n=20).",
        dataset_type,
    )
    bg = _build_synthetic_background(dataset_type, n=20)
    return ExplainerBundle(shap.KernelExplainer(model.predict, bg), "kernel")


def _build_synthetic_background(dataset_type: str, n: int = 50) -> np.ndarray:
    """Generate a representative background sample for KernelExplainer."""
    rng = np.random.default_rng(42)
    rows: list[list[float]] = []

    if dataset_type == "ds1":
        for _ in range(n):
            row = {
                "Hours_Studied": float(rng.uniform(1, 9)),
                "Previous_Scores": float(rng.uniform(40, 99)),
                "Extracurricular_Activities": float(rng.integers(0, 2)),
                "Sleep_Hours": float(rng.uniform(4, 9)),
                "Sample_Question_Papers_Practiced": float(rng.integers(0, 10)),
            }
            rows.append([row[f] for f in DATASET1_FEATURES])
    else:
        for _ in range(n):
            row = {
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
            rows.append([row[f] for f in DATASET2_FEATURES])

    return np.array(rows, dtype=float)


def get_top_negative_factors(
    shap_values: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Return the top-3 most negatively contributing features."""
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
        {"feature": feature_names[i], "shap_value": float(sv[i])}
        for i in top_neg_idx
    ]


# ── Public prediction API ─────────────────────────────────────────────────────

def predict_scores(
    df: pd.DataFrame,
    dataset_type: str,
    model: Any,
    scaler: Any,
) -> list[dict[str, Any]]:
    """
    Fast path — compute predicted scores and risk labels only (no SHAP).

    Called synchronously on every upload request.  Returns in < 1 s for
    typical batch sizes.
    """
    X = encode_features(df, dataset_type, scaler)
    raw_preds = model.predict(X)
    preds = np.clip(raw_preds, 0.0, 100.0)
    return [
        {
            "predicted_exam_score": round(float(preds[i]), 4),
            "risk_level": categorize_risk(float(preds[i])),
        }
        for i in range(len(X))
    ]


def compute_shap_explanations(
    df: pd.DataFrame,
    dataset_type: str,
    model: Any,
    scaler: Any,
    bundle: ExplainerBundle | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Compute per-row SHAP top-negative-factor lists.

    Accepts an ExplainerBundle (pre-built at startup).  When none is given
    a small KernelExplainer is built on-the-fly as a last resort.

    IMPORTANT: TreeExplainer.shap_values() does NOT accept nsamples.
               KernelExplainer.shap_values() requires nsamples (default 'auto'
               is enormous — always pass an explicit value).
    """
    log = logging.getLogger(__name__)
    X = encode_features(df, dataset_type, scaler)
    feature_names = DATASET_FEATURES[dataset_type]

    if bundle is None:
        log.warning("[SHAP] No pre-built explainer supplied — building KernelExplainer on-the-fly.")
        bg = _build_synthetic_background(dataset_type, n=20)
        bundle = ExplainerBundle(shap.KernelExplainer(model.predict, bg), "kernel")

    try:
        if bundle.kind == "tree":
            raw = bundle.explainer.shap_values(X)
        else:
            # nsamples MUST be explicit — auto defaults to 2*n_features+2048 which is huge.
            raw = bundle.explainer.shap_values(X, nsamples=100)
    except Exception as exc:  # noqa: BLE001
        log.error("[SHAP] shap_values() failed: %s. Returning empty factors.", exc)
        return [[] for _ in range(len(X))]

    # KernelExplainer may return list-of-arrays; TreeExplainer returns ndarray.
    shap_vals = np.array(raw)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    return [
        get_top_negative_factors(
            shap_vals[i] if shap_vals.ndim == 2 else shap_vals[:, i],
            feature_names,
        )
        for i in range(len(X))
    ]


def predict_with_shap(
    df: pd.DataFrame,
    dataset_type: str,
    model: Any,
    scaler: Any,
    bundle: ExplainerBundle | None = None,
) -> list[dict[str, Any]]:
    """
    Combined scores + SHAP in one synchronous call.

    Pass the cached ExplainerBundle from app.state for near-instant results.
    Used for single-student /predict.
    """
    results = predict_scores(df, dataset_type, model, scaler)
    shap_results = compute_shap_explanations(df, dataset_type, model, scaler, bundle)
    for i, result in enumerate(results):
        result["top_negative_factors"] = shap_results[i]
    return results
