"""ML inference and SHAP utilities for Dataset 1 and Dataset 2."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

from backend.core.config import (
	DS1_MODEL_PATH,
	DS1_SCALER_PATH,
	DS2_MODEL_PATH,
	DS2_SCALER_PATH,
)


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

# Backwards-compatible alias used in existing endpoint code.
REQUIRED_FEATURES = DATASET2_FEATURES

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

_DS2_LABEL_ENCODERS: dict[str, LabelEncoder] = {}
for _col, _classes in DS2_CATEGORICAL_CLASSES.items():
	_le = LabelEncoder()
	_le.fit(_classes)
	_DS2_LABEL_ENCODERS[_col] = _le

_ds1_yes_no_encoder = LabelEncoder()
_ds1_yes_no_encoder.fit(["No", "Yes"])

_models: dict[str, Any] = {}
_scalers: dict[str, Any] = {}
_shap_explainers: dict[str, Any] = {}


def detect_dataset_type(columns: list[str]) -> str | None:
	col_set = set(columns)
	if set(DATASET2_FEATURES).issubset(col_set):
		return "ds2"
	if set(DATASET1_FEATURES).issubset(col_set):
		return "ds1"
	return None


def load_model(dataset_type: str):
	if dataset_type not in _models:
		model_path = DS1_MODEL_PATH if dataset_type == "ds1" else DS2_MODEL_PATH
		_models[dataset_type] = joblib.load(model_path)
	return _models[dataset_type]


def _load_ds1_scaler():
	if "ds1" not in _scalers:
		scaler = joblib.load(DS1_SCALER_PATH)
		expected = DATASET1_FEATURES
		n_features = getattr(scaler, "n_features_in_", None)
		if n_features is not None and int(n_features) != len(expected):
			raise ValueError(
				f"Invalid DS1 scaler at {DS1_SCALER_PATH}: expected {len(expected)} features, got {n_features}."
			)

		fitted_names = list(getattr(scaler, "feature_names_in_", []))
		if fitted_names and fitted_names != expected:
			raise ValueError(
				"Invalid DS1 scaler feature order/names. "
				f"Expected {expected}, got {fitted_names}."
			)

		_scalers["ds1"] = scaler
	return _scalers["ds1"]


def _load_ds2_scaler():
	if "ds2" not in _scalers:
		scaler = joblib.load(DS2_SCALER_PATH)
		expected = DATASET2_FEATURES
		n_features = getattr(scaler, "n_features_in_", None)
		if n_features is not None and int(n_features) != len(expected):
			raise ValueError(
				f"Invalid DS2 scaler at {DS2_SCALER_PATH}: expected {len(expected)} features, got {n_features}."
			)

		fitted_names = list(getattr(scaler, "feature_names_in_", []))
		if fitted_names and fitted_names != expected:
			raise ValueError(
				"Invalid DS2 scaler feature order/names. "
				f"Expected {expected}, got {fitted_names}."
			)

		_scalers["ds2"] = scaler
	return _scalers["ds2"]


def _encode_dataset1(df: pd.DataFrame) -> np.ndarray:
	working = df[DATASET1_FEATURES].copy()

	# Dataset 1 was trained with full-feature scaling and binary encoding.
	working["Extracurricular_Activities"] = (
		working["Extracurricular_Activities"].astype(str).fillna("No")
	)
	working["Extracurricular_Activities"] = working["Extracurricular_Activities"].apply(
		lambda v: int(_ds1_yes_no_encoder.transform([v])[0]) if v in {"Yes", "No"} else 0
	)

	for col in [c for c in DATASET1_FEATURES if c != "Extracurricular_Activities"]:
		series = pd.to_numeric(working[col], errors="coerce")
		mean_val = series.mean() if series.notna().any() else 0.0
		working[col] = series.fillna(mean_val)

	scaler = _load_ds1_scaler()
	X = working[DATASET1_FEATURES].astype(float)
	return scaler.transform(X)


def _encode_dataset2(df: pd.DataFrame) -> np.ndarray:
	working = df.copy()

	for col, classes in DS2_CATEGORICAL_CLASSES.items():
		if col in working.columns:
			working[col] = working[col].fillna(classes[0])

	for col in DS2_NUMERIC_FEATURES:
		if col in working.columns:
			series = pd.to_numeric(working[col], errors="coerce")
			mean_val = series.mean() if series.notna().any() else 0.0
			working[col] = series.fillna(mean_val)

	for col, le in _DS2_LABEL_ENCODERS.items():
		if col in working.columns:
			known = set(le.classes_)

			def _safe_encode(x: str, _le: LabelEncoder = le, _known: set = known) -> int:
				return int(_le.transform([x])[0]) if x in _known else 0

			working[col] = working[col].astype(str).apply(_safe_encode)

	X = working[DATASET2_FEATURES].astype(float)
	scaler = _load_ds2_scaler()
	return scaler.transform(X)


def encode_features(df: pd.DataFrame, dataset_type: str) -> np.ndarray:
	if dataset_type == "ds1":
		return _encode_dataset1(df)
	return _encode_dataset2(df)


def _build_synthetic_background(dataset_type: str, n: int = 50) -> np.ndarray:
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
		bg = np.array(rows, dtype=float)
		try:
			scaler = _load_ds1_scaler()
			bg_df = pd.DataFrame(bg, columns=DATASET1_FEATURES)
			return scaler.transform(bg_df)
		except Exception:  # noqa: BLE001
			return bg

	for _ in range(n):
		row2 = {
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
		rows.append([row2[f] for f in DATASET2_FEATURES])
	return np.array(rows, dtype=float)


def get_shap_explainer(dataset_type: str, background: np.ndarray | None = None):
	model = load_model(dataset_type)
	bg = background if background is not None else _build_synthetic_background(dataset_type)
	bg = bg[: min(50, len(bg))]
	if dataset_type not in _shap_explainers or (background is not None):
		_shap_explainers[dataset_type] = shap.KernelExplainer(model.predict, bg)
	return _shap_explainers[dataset_type]


def compute_shap_values(
	X: np.ndarray,
	dataset_type: str,
	background: np.ndarray | None = None,
) -> np.ndarray:
	explainer = get_shap_explainer(dataset_type, background)
	sv = explainer.shap_values(X, nsamples=100)
	return np.array(sv)


def categorize_risk(score: float) -> str:
	if score >= 70:
		return "Stable"
	if score >= 60:
		return "Borderline"
	return "At-Risk"


def get_top_negative_factors(
	shap_values: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
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
		}
		for i in top_neg_idx
	]


def predict_batch(
	df: pd.DataFrame,
	dataset_type: str,
	use_df_as_background: bool = True,
) -> list[dict[str, Any]]:
	model = load_model(dataset_type)
	feature_names = DATASET_FEATURES[dataset_type]
	X = encode_features(df, dataset_type)

	background = X if use_df_as_background and len(X) >= 5 else None
	raw_preds = model.predict(X)
	preds = np.clip(raw_preds, 0.0, 100.0)

	shap_vals = compute_shap_values(X, dataset_type=dataset_type, background=background)
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
				"top_negative_factors": get_top_negative_factors(sv_row, feature_names),
			}
		)
	return results


def predict_single(features: dict[str, Any], dataset_type: str = "ds2") -> dict[str, Any]:
	df = pd.DataFrame([features])
	return predict_batch(df, dataset_type=dataset_type, use_df_as_background=False)[0]
