"""
Student Academic Performance Prediction System
===============================================
Stacking Ensemble Regressor with per-student SHAP diagnostics and risk
categorisation.

Datasets:
- Dataset 1: 10,000 samples, 6 features (Performance_Index as target)
- Dataset 2: 6,607 samples, 20 features (Exam_Score as target)

Usage:
    python student_performance_predictor.py

If the actual CSV files are available, place them as:
    data/dataset1.csv
    data/dataset2.csv
Otherwise, synthetic data matching the described distributions is generated.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor
import shap
import lime
import lime.lime_tabular
import joblib

warnings.filterwarnings("ignore")


def _flush(msg=""):
    print(msg, flush=True)


# ─────────────────────────── Configuration ────────────────────────────────────

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────── Data Generation ──────────────────────────────────

def generate_dataset1(n: int = 10_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Synthetic Dataset 1: 10,000 samples, 6 columns.
    Correlation design: Previous_Scores→target ≈ 0.92, Hours_Studied ≈ 0.37
    """
    rng = np.random.default_rng(seed)
    hours_studied = rng.uniform(1, 9, n)
    previous_scores = rng.uniform(40, 99, n)
    extracurricular = rng.choice(["Yes", "No"], n)
    sleep_hours = rng.uniform(4, 9, n)
    sample_papers = rng.integers(0, 10, n).astype(float)

    # Design formula so correlations match spec
    noise = rng.normal(0, 2.5, n)
    performance_index = (
        0.55 * previous_scores
        + 2.2 * hours_studied
        + 0.1 * (np.array(extracurricular) == "Yes").astype(float)
        + 0.3 * sleep_hours
        + 0.2 * sample_papers
        + noise
    )
    performance_index = np.clip(performance_index, 10, 100)

    return pd.DataFrame({
        "Hours_Studied": hours_studied,
        "Previous_Scores": previous_scores,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Sample_Question_Papers_Practiced": sample_papers,
        "Performance_Index": performance_index,
    })


def generate_dataset2(n: int = 6_607, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Synthetic Dataset 2: 6,607 samples, 20 columns.
    Includes Attendance (r≈0.58) and Hours_Studied (r≈0.45) as primary features.
    Has missing values in Teacher_Quality, Parental_Education_Level, Distance_from_Home.
    """
    rng = np.random.default_rng(seed)

    attendance = rng.uniform(60, 100, n)
    hours_studied = rng.uniform(1, 44, n)
    gender = rng.choice(["Male", "Female"], n)
    parental_involvement = rng.choice(["Low", "Medium", "High"], n)
    access_to_resources = rng.choice(["Low", "Medium", "High"], n)
    extracurricular = rng.choice(["Yes", "No"], n)
    sleep_hours = rng.uniform(4, 10, n)
    prev_scores = rng.uniform(50, 100, n)
    motivation_level = rng.choice(["Low", "Medium", "High"], n)
    internet_access = rng.choice(["Yes", "No"], n)
    tutoring_sessions = rng.integers(0, 8, n).astype(float)
    family_income = rng.choice(["Low", "Medium", "High"], n)
    teacher_quality = rng.choice(["Low", "Medium", "High"], n).astype(object)
    school_type = rng.choice(["Public", "Private"], n)
    peer_influence = rng.choice(["Positive", "Neutral", "Negative"], n)
    physical_activity = rng.integers(0, 6, n).astype(float)
    learning_disabilities = rng.choice(["Yes", "No"], n)
    parental_education = rng.choice(
        ["High School", "College", "Postgraduate"], n
    ).astype(object)
    distance_home = rng.choice(["Near", "Moderate", "Far"], n).astype(object)

    noise = rng.normal(0, 3, n)
    exam_score = (
        0.4 * (attendance - 60) / 40 * 50
        + 0.3 * hours_studied / 44 * 40
        + 0.1 * (np.array(gender) == "Male").astype(float) * 3
        + noise
        + 40
    )
    exam_score = np.clip(exam_score, 0, 100).astype(float)

    # Inject outlier value 101 (to be cleaned in preprocessing)
    outlier_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    exam_score[outlier_idx] = 101.0

    df = pd.DataFrame({
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Gender": gender,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": prev_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_home,
        "Exam_Score": exam_score,
    })

    # Inject missing values
    for col in ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]:
        miss_idx = rng.choice(n, size=n // 20, replace=False)
        df.loc[miss_idx, col] = np.nan

    return df


def load_or_generate_dataset1() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "dataset1.csv")
    if os.path.exists(path):
        _flush(f"[Dataset 1] Loading from {path}")
        return pd.read_csv(path)
    _flush("[Dataset 1] Generating synthetic data …")
    return generate_dataset1()


def load_or_generate_dataset2() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "dataset2.csv")
    if os.path.exists(path):
        _flush(f"[Dataset 2] Loading from {path}")
        return pd.read_csv(path)
    _flush("[Dataset 2] Generating synthetic data …")
    return generate_dataset2()


# ─────────────────────────── Preprocessing ────────────────────────────────────

def preprocess_dataset1(df: pd.DataFrame):
    """Label-encode Extracurricular_Activities; StandardScaler on all numerics."""
    df = df.copy()
    le = LabelEncoder()
    df["Extracurricular_Activities"] = le.fit_transform(
        df["Extracurricular_Activities"].astype(str)
    )
    target = df["Performance_Index"].values.copy()
    features = df.drop(columns=["Performance_Index"])
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns
    )
    return features_scaled, target, scaler, le


def preprocess_dataset2(df: pd.DataFrame):
    """
    Impute missing values, fix Exam_Score outlier (101→100),
    label-encode all categorical columns.
    """
    df = df.copy()
    # Fix outlier
    df["Exam_Score"] = df["Exam_Score"].replace(101, 100)

    target_col = "Exam_Score"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [
        c for c in df.columns if c not in cat_cols and c != target_col
    ]

    # Impute
    for col in cat_cols:
        mode_series = df[col].mode()
        mode_val = mode_series[0] if not mode_series.empty else "Unknown"
        df[col] = df[col].fillna(mode_val)
    for col in num_cols:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)

    # Label-encode each categorical column with its own encoder
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    target = df[target_col].values.copy()
    features = df.drop(columns=[target_col])
    return features, target


# ─────────────────────────── Feature Selection ────────────────────────────────

def compute_correlations(
    X: pd.DataFrame, y: np.ndarray, title: str, output_path: str
) -> pd.Series:
    """Compute and plot Pearson correlation heatmap; return feature-target corrs."""
    df_c = X.copy()
    df_c["TARGET"] = y
    corr = df_c.corr()

    n_feats = len(X.columns)
    fig_size = max(8, n_feats)
    plt.figure(figsize=(fig_size, max(6, n_feats - 1)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5,
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    _flush(f"  Saved correlation heatmap → {output_path}")

    target_corr = corr["TARGET"].drop("TARGET").abs().sort_values(ascending=False)
    return target_corr


# ─────────────────────────── Model Definitions ────────────────────────────────

def get_base_models() -> dict:
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_child_weight=3, gamma=0.1, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=RANDOM_STATE, verbose=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def train_and_evaluate_all(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label: str = "",
) -> dict:
    """Train all base models; return results dict."""
    models = get_base_models()
    results = {}
    _flush(f"\n{'='*60}")
    _flush(f"  Training base models [{label}]")
    _flush(f"{'='*60}")
    _flush(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    _flush(f"  {'-'*50}")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            "model": model, "MAE": mae, "RMSE": rmse,
            "R2": r2, "y_pred": y_pred,
        }
        _flush(f"  {name:<22} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")
    return results


# ─────────────────────────── Stacking Ensemble ────────────────────────────────

def build_stacking_regressor() -> StackingRegressor:
    """
    Build a StackingRegressor with XGBoost, CatBoost and RandomForest as
    base learners and LassoCV as the meta-learner.  LassoCV automatically
    learns optimal combination weights and performs feature selection via
    L1 regularisation, eliminating the need for manual weight tuning.
    5-fold cross-validation is used internally to generate out-of-fold
    meta-features for training the final estimator.
    """
    base_estimators = [
        (name, model) for name, model in get_base_models().items()
    ]
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=LassoCV(cv=5, random_state=RANDOM_STATE),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )


def paired_t_tests(
    model,
    base_results: dict,
    compare_names: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Paired t-tests: StackingRegressor vs. selected baselines."""
    model_errors = np.abs(model.predict(X_test) - y_test)
    _flush("\n  Paired t-tests (StackingRegressor vs. baselines):")
    for name in compare_names:
        base_errors = np.abs(base_results[name]["y_pred"] - y_test)
        t_stat, p_val = stats.ttest_rel(model_errors, base_errors)
        sig = "✓ significant" if p_val < 0.05 else "✗ not significant"
        _flush(f"  vs {name:<20} t={t_stat:>8.4f}  p={p_val:.4f}  {sig}")


# ─────────────────────────── Explainability ───────────────────────────────────

def shap_explainability(
    model,
    X_bg: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    output_prefix: str,
    n_bg: int = 50,
    n_explain: int = 30,
    explainer=None,
) -> "shap.KernelExplainer":
    """
    Global SHAP explainability using KernelExplainer, which is compatible
    with any sklearn-compatible model including StackingRegressor.
    Produces: bar chart of mean(|SHAP|) and summary dot plot.

    If *explainer* is provided it will be reused; otherwise a new
    KernelExplainer is built from X_bg[:n_bg].  The (possibly new)
    explainer is returned so callers can reuse it for per-student analysis.
    """
    _flush("\n  Computing SHAP values …")
    if explainer is None:
        explainer = shap.KernelExplainer(model.predict, X_bg[:n_bg])
    sv = explainer.shap_values(X_explain[:n_explain], nsamples=100)
    sv_array = np.array(sv)
    mean_abs = np.abs(sv_array).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    fn_arr = np.array(feature_names)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(fn_arr[sorted_idx], mean_abs[sorted_idx], color="steelblue")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_path = f"{output_prefix}_shap_bar.png"
    plt.savefig(bar_path, dpi=100)
    plt.close()
    _flush(f"  Saved SHAP bar chart → {bar_path}")

    # Summary dot plot
    shap.summary_plot(
        sv_array, X_explain[:n_explain],
        feature_names=feature_names, show=False,
    )
    dot_path = f"{output_prefix}_shap_dot.png"
    plt.savefig(dot_path, dpi=100, bbox_inches="tight")
    plt.close()
    _flush(f"  Saved SHAP dot plot → {dot_path}")
    return explainer


def lime_explainability(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    output_prefix: str,
) -> None:
    """LIME local explainability for a single test instance."""
    _flush("\n  Computing LIME explanation …")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="regression",
        random_state=RANDOM_STATE,
    )
    exp = explainer.explain_instance(
        X_test[0],
        model.predict,
        num_features=min(5, len(feature_names)),
    )
    fig = exp.as_pyplot_figure()
    plt.title("LIME Explanation — Single Instance")
    plt.tight_layout()
    lime_path = f"{output_prefix}_lime_explanation.png"
    plt.savefig(lime_path, dpi=100, bbox_inches="tight")
    plt.close()
    _flush(f"  Saved LIME explanation → {lime_path}")


# ─────────────────────────── Risk & Per-Student Diagnostics ───────────────────

def categorize_risk(score: float) -> str:
    """
    Map a predicted continuous score to a discrete risk bucket.

    Returns
    -------
    "Stable"     — score ≥ 70  (green)
    "Borderline" — 60 ≤ score < 70  (yellow)
    "At-Risk"    — score < 60  (red)
    """
    if score >= 70:
        return "Stable"
    elif score >= 60:
        return "Borderline"
    else:
        return "At-Risk"


def predict_with_diagnostics(
    model,
    student_features: np.ndarray,
    feature_names: list,
    shap_explainer,
) -> dict:
    """
    Predict the performance score for a single student and return a
    diagnostic dictionary containing:

        {
            "predicted_score":      float,
            "risk_level":           str,   # "Stable", "Borderline", "At-Risk"
            "top_negative_factors": list[str]  # top-3 features dragging score down
        }

    Parameters
    ----------
    model            : fitted StackingRegressor (or any sklearn regressor)
    student_features : 1-D or 2-D array of shape (n_features,) or (1, n_features)
    feature_names    : list of feature name strings
    shap_explainer   : pre-built shap.KernelExplainer fitted on background data
    """
    if student_features.ndim == 1:
        student_features = student_features.reshape(1, -1)

    predicted_score = float(model.predict(student_features)[0])
    risk_level = categorize_risk(predicted_score)

    # Per-student SHAP values via the pre-built KernelExplainer
    sv = shap_explainer.shap_values(student_features, nsamples=100)
    sv_flat = np.array(sv).flatten()

    # Identify features with negative SHAP contributions (dragging score down)
    neg_mask = sv_flat < 0
    if neg_mask.any():
        neg_idx = np.where(neg_mask)[0]
        # Sort by most negative first; if fewer than 3 negative factors exist,
        # pad with the least-positive ones (smallest lift) so we always return 3.
        top_neg_idx = neg_idx[np.argsort(sv_flat[neg_idx])]
        if len(top_neg_idx) < 3:
            pos_idx = np.where(~neg_mask)[0]
            pad = pos_idx[np.argsort(sv_flat[pos_idx])]
            top_neg_idx = np.concatenate([top_neg_idx, pad])
        top_neg_idx = top_neg_idx[:3]
    else:
        # No negative factors — return the least positive (smallest lift)
        top_neg_idx = np.argsort(sv_flat)[:3]

    top_negative_factors = [feature_names[i] for i in top_neg_idx]

    return {
        "predicted_score": round(predicted_score, 4),
        "risk_level": risk_level,
        "top_negative_factors": top_negative_factors,
    }


# ─────────────────────────── Visualizations ───────────────────────────────────

def plot_mae_rmse_comparison(
    base_results: dict, vr_mae: float, vr_rmse: float, output_path: str
) -> None:
    names = list(base_results) + ["StackingRegressor"]
    maes = [base_results[n]["MAE"] for n in base_results] + [vr_mae]
    rmses = [base_results[n]["RMSE"] for n in base_results] + [vr_rmse]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w / 2, maes, w, label="MAE", color="steelblue")
    ax.bar(x + w / 2, rmses, w, label="RMSE", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("MAE and RMSE Comparison Across All Models")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    _flush(f"  Saved MAE/RMSE comparison → {output_path}")


def plot_r2_comparison(
    base_results: dict, vr_r2: float, output_path: str
) -> None:
    names = list(base_results) + ["StackingRegressor"]
    r2s = [base_results[n]["R2"] for n in base_results] + [vr_r2]
    best_idx = int(np.argmax(r2s))
    colors = ["steelblue"] * len(r2s)
    colors[best_idx] = "gold"
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, r2s, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score Comparison Across All Models")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    _flush(f"  Saved R² comparison → {output_path}")


def plot_actual_vs_predicted(
    y_test: np.ndarray, y_pred: np.ndarray,
    title: str, output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    _flush(f"  Saved actual vs predicted → {output_path}")


def plot_extracurricular_pie(df: pd.DataFrame, output_path: str) -> None:
    counts = df["Extracurricular_Activities"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           startangle=90, colors=["#66b3ff", "#ff9999"])
    ax.set_title("Extracurricular Activities Participation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    _flush(f"  Saved pie chart → {output_path}")


def plot_sleep_vs_performance(df: pd.DataFrame, output_path: str) -> None:
    bins = pd.cut(df["Sleep_Hours"], bins=10)
    grouped = df.groupby(bins, observed=True)["Performance_Index"].mean()
    midpoints = [interval.mid for interval in grouped.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(midpoints, grouped.values, marker="o", color="steelblue")
    ax.set_xlabel("Sleep Hours")
    ax.set_ylabel("Average Performance Index")
    ax.set_title("Sleep Hours vs. Average Performance Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    _flush(f"  Saved sleep vs performance → {output_path}")


def plot_sample_papers_vs_performance(
    df: pd.DataFrame, output_path: str
) -> None:
    grouped = df.groupby(
        "Sample_Question_Papers_Practiced"
    )["Performance_Index"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index.astype(str), grouped.values, color="steelblue")
    ax.set_xlabel("Sample Question Papers Practiced")
    ax.set_ylabel("Average Performance Index")
    ax.set_title("Sample Question Papers Practiced vs. Avg Performance Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    _flush(f"  Saved sample papers vs performance → {output_path}")


# ─────────────────────────── Summary Table ────────────────────────────────────

def print_summary_table(
    base_results: dict, vr_mae: float, vr_rmse: float, vr_r2: float,
    label: str,
) -> None:
    all_r2 = {n: base_results[n]["R2"] for n in base_results}
    all_r2["StackingRegressor"] = vr_r2
    best = max(all_r2, key=all_r2.get)

    _flush(f"\n{'─'*65}")
    _flush(f"  Summary Table — {label}")
    _flush(f"  {'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    _flush(f"  {'─'*52}")
    for name in base_results:
        marker = " ◄ BEST" if name == best else ""
        _flush(
            f"  {name:<22} {base_results[name]['MAE']:>10.4f} "
            f"{base_results[name]['RMSE']:>10.4f} "
            f"{base_results[name]['R2']:>10.4f}{marker}"
        )
    vr_marker = " ◄ BEST" if "StackingRegressor" == best else ""
    _flush(
        f"  {'StackingRegressor':<22} {vr_mae:>10.4f} "
        f"{vr_rmse:>10.4f} {vr_r2:>10.4f}{vr_marker}"
    )
    _flush(f"{'─'*65}")


# ─────────────────────────── Full Pipeline ────────────────────────────────────

def run_pipeline(
    X_all: pd.DataFrame,
    y: np.ndarray,
    feature_names_all: list,
    selected_features: list,
    label_prefix: str,
    output_prefix: str,
    n_cv_splits: int = 10,
) -> dict:
    """
    Run full pipeline for one dataset under two conditions:
      (a) all features
      (b) selected features only

    Each condition trains a StackingRegressor (XGBoost + CatBoost +
    RandomForest base learners, LassoCV meta-learner) and generates
    per-student diagnostic predictions with SHAP-based risk assessments.
    """
    results_summary = {}
    feat_idx = {name: i for i, name in enumerate(feature_names_all)}

    for condition, feature_cols in [
        ("all_features", feature_names_all),
        ("selected_features", selected_features),
    ]:
        cond_label = f"{label_prefix} [{condition}]"
        cond_prefix = os.path.join(OUTPUT_DIR, f"{output_prefix}_{condition}")

        X = (
            X_all[feature_cols].values
            if isinstance(X_all, pd.DataFrame)
            else X_all[:, [feat_idx[f] for f in feature_cols]]
        )
        fn = list(feature_cols)

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # ── Base models ───────────────────────────────────────────────────
        base_results = train_and_evaluate_all(
            X_train, y_train, X_test, y_test, label=cond_label
        )

        # ── Stacking Regressor ────────────────────────────────────────────
        _flush("\n  Training StackingRegressor (cv=5, meta-learner=LassoCV) …")
        stacking_regressor = build_stacking_regressor()
        stacking_regressor.fit(X_train, y_train)

        sr_preds = stacking_regressor.predict(X_test)
        sr_mae = mean_absolute_error(y_test, sr_preds)
        sr_rmse = np.sqrt(mean_squared_error(y_test, sr_preds))
        sr_r2 = r2_score(y_test, sr_preds)
        _flush(
            f"\n  StackingRegressor: "
            f"MAE={sr_mae:.4f}  RMSE={sr_rmse:.4f}  R²={sr_r2:.4f}"
        )

        # ── Paired t-tests ────────────────────────────────────────────────
        paired_t_tests(
            stacking_regressor, base_results,
            list(base_results.keys()),
            X_test, y_test,
        )

        # ── Explainability (all-features condition only) ──────────────────
        # Build the KernelExplainer once; reuse it for per-student diagnostics
        # to avoid redundant explainer initialisation.
        _flush("\n  Building KernelExplainer (background = 50 training samples) …")
        kernel_explainer = shap.KernelExplainer(
            stacking_regressor.predict, X_train[:50]
        )
        if condition == "all_features":
            shap_explainability(
                stacking_regressor, X_train, X_test, fn, cond_prefix,
                n_bg=50, n_explain=25, explainer=kernel_explainer,
            )
            lime_explainability(
                stacking_regressor, X_train, X_test, fn, cond_prefix
            )

        # ── Per-student diagnostics (sample of 5 test students) ──────────
        n_sample = min(5, len(X_test))
        X_sample = X_test[:n_sample]

        # Batch-compute SHAP values for all sample students at once for
        # efficiency, then split into per-student arrays.
        _flush(f"\n  Computing per-student SHAP values (batch of {n_sample}) …")
        batch_sv = np.array(
            kernel_explainer.shap_values(X_sample, nsamples=100)
        )

        _flush(
            f"\n  {'#':<4} {'Score':>8}  {'Risk':<12}  Top Negative Factors"
        )
        _flush(f"  {'-'*70}")
        diagnostics = []
        for i in range(n_sample):
            # Reuse pre-computed SHAP row; wrap in a thin explainer proxy so
            # predict_with_diagnostics does not re-compute.
            sv_row = batch_sv[i] if batch_sv.ndim == 2 else batch_sv[:, i]

            predicted_score = float(stacking_regressor.predict(X_sample[i:i+1])[0])
            risk_level = categorize_risk(predicted_score)
            sv_flat = np.array(sv_row).flatten()

            neg_mask = sv_flat < 0
            if neg_mask.any():
                neg_idx = np.where(neg_mask)[0]
                top_neg_idx = neg_idx[np.argsort(sv_flat[neg_idx])]
                if len(top_neg_idx) < 3:
                    pos_idx = np.where(~neg_mask)[0]
                    pad = pos_idx[np.argsort(sv_flat[pos_idx])]
                    top_neg_idx = np.concatenate([top_neg_idx, pad])
                top_neg_idx = top_neg_idx[:3]
            else:
                top_neg_idx = np.argsort(sv_flat)[:3]

            diag = {
                "predicted_score": round(predicted_score, 4),
                "risk_level": risk_level,
                "top_negative_factors": [fn[j] for j in top_neg_idx],
            }
            diagnostics.append(diag)
            factors_str = ", ".join(diag["top_negative_factors"])
            _flush(
                f"  {i + 1:<4} {diag['predicted_score']:>8.2f}"
                f"  {diag['risk_level']:<12}  {factors_str}"
            )

        # ── Plots ─────────────────────────────────────────────────────────
        plot_mae_rmse_comparison(
            base_results, sr_mae, sr_rmse,
            f"{cond_prefix}_mae_rmse_comparison.png",
        )
        plot_r2_comparison(
            base_results, sr_r2,
            f"{cond_prefix}_r2_comparison.png",
        )
        plot_actual_vs_predicted(
            y_test, sr_preds,
            f"Actual vs. Predicted — {cond_label}",
            f"{cond_prefix}_actual_vs_predicted.png",
        )

        # ── Summary table ─────────────────────────────────────────────────
        print_summary_table(base_results, sr_mae, sr_rmse, sr_r2, cond_label)

        # ── Export model ──────────────────────────────────────────────────
        model_path = f"{cond_prefix}_stacking_regressor.joblib"
        joblib.dump(stacking_regressor, model_path)
        _flush(f"\n  Exported StackingRegressor → {model_path}")

        results_summary[condition] = {
            "base_results": base_results,
            "stacking_regressor": stacking_regressor,
            "sr_mae": sr_mae,
            "sr_rmse": sr_rmse,
            "sr_r2": sr_r2,
            "diagnostics_sample": diagnostics,
        }

    return results_summary


# ─────────────────────────── Main ─────────────────────────────────────────────

def main() -> None:
    _flush("\n" + "=" * 70)
    _flush("  Student Academic Performance Prediction System")
    _flush("=" * 70)

    # ── Dataset 1 ──────────────────────────────────────────────────────────
    _flush("\n\n[DATASET 1]")
    raw_df1 = load_or_generate_dataset1()
    _flush(f"  Shape: {raw_df1.shape}")
    _flush(f"  Missing values: {raw_df1.isnull().sum().sum()}")

    X1, y1, _scaler1, _le1 = preprocess_dataset1(raw_df1)

    target_corr1 = compute_correlations(
        X1, y1,
        "Dataset 1 — Pearson Correlation Heatmap",
        os.path.join(OUTPUT_DIR, "ds1_correlation_heatmap.png"),
    )
    _flush("\n  Feature–Target Correlations (Dataset 1):")
    _flush(target_corr1.to_string())

    # Per specification: retain Previous_Scores and Hours_Studied
    selected1 = ["Previous_Scores", "Hours_Studied"]

    # Dataset 1-specific visualizations
    plot_extracurricular_pie(
        raw_df1, os.path.join(OUTPUT_DIR, "ds1_extracurricular_pie.png")
    )
    plot_sleep_vs_performance(
        raw_df1, os.path.join(OUTPUT_DIR, "ds1_sleep_vs_performance.png")
    )
    plot_sample_papers_vs_performance(
        raw_df1, os.path.join(OUTPUT_DIR, "ds1_sample_papers_vs_performance.png")
    )

    run_pipeline(
        X1, y1,
        list(X1.columns),
        selected1,
        label_prefix="Dataset 1",
        output_prefix="ds1",
        n_cv_splits=10,
    )

    # ── Dataset 2 ──────────────────────────────────────────────────────────
    _flush("\n\n[DATASET 2]")
    raw_df2 = load_or_generate_dataset2()
    _flush(f"  Shape: {raw_df2.shape}")
    missing_cols = raw_df2.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    _flush(f"  Missing values before preprocessing:\n{missing_cols}")

    X2, y2 = preprocess_dataset2(raw_df2)
    _flush(f"  Shape after preprocessing: {X2.shape}")

    target_corr2 = compute_correlations(
        X2, y2,
        "Dataset 2 — Pearson Correlation Heatmap",
        os.path.join(OUTPUT_DIR, "ds2_correlation_heatmap.png"),
    )
    _flush("\n  Top Feature–Target Correlations (Dataset 2):")
    _flush(target_corr2.head(5).to_string())

    # Per specification: retain Attendance and Hours_Studied
    selected2 = ["Attendance", "Hours_Studied"]

    run_pipeline(
        X2, y2,
        list(X2.columns),
        selected2,
        label_prefix="Dataset 2",
        output_prefix="ds2",
        n_cv_splits=10,
    )

    _flush("\n\n" + "=" * 70)
    _flush(f"  ✓  Done! Outputs saved in: {os.path.abspath(OUTPUT_DIR)}")
    _flush("=" * 70 + "\n")


if __name__ == "__main__":
    main()
