"""
Student Academic Performance Prediction System
===============================================
Ensemble Voting Regressor with SHAP and LIME explainability.

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

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor,
    BaggingRegressor, VotingRegressor,
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
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_child_weight=1, gamma=0, subsample=0.8,
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=100, random_seed=RANDOM_STATE, verbose=0,
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=100, random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "Bagging": BaggingRegressor(
            n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def train_and_evaluate_all(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label: str = "",
) -> dict:
    """Train all 9 base models; return results dict."""
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


# ─────────────────────────── Voting Regressor ─────────────────────────────────

# Top-5 models and initial weights per specification
_TOP5 = ["LinearRegression", "Ridge", "CatBoost", "XGBoost", "RandomForest"]
_WEIGHTS_GRID = [
    [7, 7, 1, 1, 1],
    [5, 5, 2, 2, 2],
    [6, 6, 2, 1, 1],
    [8, 8, 1, 1, 1],
]


def optimize_voting_weights(
    base_results: dict,
    X_val: np.ndarray,
    y_val: np.ndarray,
    top5_names: list,
) -> list:
    """
    Select best weights by evaluating weighted averages of already-fitted
    base model predictions on a held-out validation set.
    No re-fitting required → fast grid search.
    """
    _flush("\n  Optimizing VotingRegressor weights (fast grid search) …")
    preds = {
        n: base_results[n]["model"].predict(X_val) for n in top5_names
    }
    best_rmse, best_w = np.inf, _WEIGHTS_GRID[0]
    n_models = len(top5_names)
    for w in _WEIGHTS_GRID:
        w_arr = np.array(w[:n_models], dtype=float)
        w_arr /= w_arr.sum()
        y_hat = sum(w_arr[i] * preds[top5_names[i]] for i in range(n_models))
        rmse = np.sqrt(mean_squared_error(y_val, y_hat))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    _flush(f"  Best weights: {best_w}  (val RMSE={best_rmse:.4f})")
    return best_w


def cross_validate_voting_regressor(
    estimators: list,
    weights: list,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
) -> tuple:
    """10-fold CV on VotingRegressor; returns (mean MAE, RMSE, R²)."""
    _flush(f"\n  {n_splits}-Fold Cross-Validation …")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    maes, rmses, r2s = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
        vr = VotingRegressor(estimators=estimators, weights=weights)
        vr.fit(X[tr_idx], y[tr_idx])
        yp = vr.predict(X[te_idx])
        maes.append(mean_absolute_error(y[te_idx], yp))
        rmses.append(np.sqrt(mean_squared_error(y[te_idx], yp)))
        r2s.append(r2_score(y[te_idx], yp))
    _flush(
        f"  {n_splits}-Fold CV — "
        f"MAE: {np.mean(maes):.4f}±{np.std(maes):.4f}  "
        f"RMSE: {np.mean(rmses):.4f}±{np.std(rmses):.4f}  "
        f"R²: {np.mean(r2s):.4f}±{np.std(r2s):.4f}"
    )
    return np.mean(maes), np.mean(rmses), np.mean(r2s)


def paired_t_tests(
    vr: VotingRegressor,
    base_results: dict,
    compare_names: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Paired t-tests: VotingRegressor vs. selected baselines."""
    vr_errors = np.abs(vr.predict(X_test) - y_test)
    _flush("\n  Paired t-tests (VotingRegressor vs. baselines):")
    for name in compare_names:
        base_errors = np.abs(base_results[name]["y_pred"] - y_test)
        t_stat, p_val = stats.ttest_rel(vr_errors, base_errors)
        sig = "✓ significant" if p_val < 0.05 else "✗ not significant"
        _flush(f"  vs {name:<20} t={t_stat:>8.4f}  p={p_val:.4f}  {sig}")


# ─────────────────────────── Explainability ───────────────────────────────────

def shap_explainability(
    vr: VotingRegressor,
    X_bg: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    output_prefix: str,
    n_bg: int = 50,
    n_explain: int = 30,
) -> None:
    """
    SHAP global explainability using PermutationExplainer with
    small background and explanation sets for reasonable runtime.
    Produces: bar chart of mean(|SHAP|) and summary dot plot.
    """
    _flush("\n  Computing SHAP values …")
    bg = shap.maskers.Independent(X_bg[:n_bg])
    explainer = shap.PermutationExplainer(vr.predict, bg)
    sv = explainer(X_explain[:n_explain])

    mean_abs = np.abs(sv.values).mean(axis=0)
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
        sv.values, X_explain[:n_explain],
        feature_names=feature_names, show=False,
    )
    dot_path = f"{output_prefix}_shap_dot.png"
    plt.savefig(dot_path, dpi=100, bbox_inches="tight")
    plt.close()
    _flush(f"  Saved SHAP dot plot → {dot_path}")


def lime_explainability(
    vr: VotingRegressor,
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
        vr.predict,
        num_features=min(5, len(feature_names)),
    )
    fig = exp.as_pyplot_figure()
    plt.title("LIME Explanation — Single Instance")
    plt.tight_layout()
    lime_path = f"{output_prefix}_lime_explanation.png"
    plt.savefig(lime_path, dpi=100, bbox_inches="tight")
    plt.close()
    _flush(f"  Saved LIME explanation → {lime_path}")


# ─────────────────────────── Visualizations ───────────────────────────────────

def plot_mae_rmse_comparison(
    base_results: dict, vr_mae: float, vr_rmse: float, output_path: str
) -> None:
    names = list(base_results) + ["VotingRegressor"]
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
    names = list(base_results) + ["VotingRegressor"]
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
    all_r2["VotingRegressor"] = vr_r2
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
    vr_marker = " ◄ BEST" if "VotingRegressor" == best else ""
    _flush(
        f"  {'VotingRegressor':<22} {vr_mae:>10.4f} "
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
        # Small validation set (from training) for fast weight selection
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE
        )

        # ── Base models ───────────────────────────────────────────────────
        base_results = train_and_evaluate_all(
            X_train, y_train, X_test, y_test, label=cond_label
        )

        # ── Voting Regressor ──────────────────────────────────────────────
        best_w = optimize_voting_weights(base_results, X_val, y_val, _TOP5)
        estimators = [(n, base_results[n]["model"]) for n in _TOP5]
        vr = VotingRegressor(estimators=estimators, weights=best_w)
        vr.fit(X_train, y_train)

        vr_preds = vr.predict(X_test)
        vr_mae = mean_absolute_error(y_test, vr_preds)
        vr_rmse = np.sqrt(mean_squared_error(y_test, vr_preds))
        vr_r2 = r2_score(y_test, vr_preds)
        _flush(
            f"\n  VotingRegressor: "
            f"MAE={vr_mae:.4f}  RMSE={vr_rmse:.4f}  R²={vr_r2:.4f}"
        )

        # ── 10-fold CV ────────────────────────────────────────────────────
        cross_validate_voting_regressor(
            estimators, best_w, X, y, n_splits=n_cv_splits
        )

        # ── Paired t-tests ────────────────────────────────────────────────
        paired_t_tests(
            vr, base_results, ["SVR", "XGBoost", "CatBoost"],
            X_test, y_test,
        )

        # ── Explainability (all-features condition only) ──────────────────
        if condition == "all_features":
            shap_explainability(
                vr, X_train, X_test, fn, cond_prefix,
                n_bg=50, n_explain=25,
            )
            lime_explainability(vr, X_train, X_test, fn, cond_prefix)

        # ── Plots ─────────────────────────────────────────────────────────
        plot_mae_rmse_comparison(
            base_results, vr_mae, vr_rmse,
            f"{cond_prefix}_mae_rmse_comparison.png",
        )
        plot_r2_comparison(
            base_results, vr_r2,
            f"{cond_prefix}_r2_comparison.png",
        )
        plot_actual_vs_predicted(
            y_test, vr_preds,
            f"Actual vs. Predicted — {cond_label}",
            f"{cond_prefix}_actual_vs_predicted.png",
        )

        # ── Summary table ─────────────────────────────────────────────────
        print_summary_table(base_results, vr_mae, vr_rmse, vr_r2, cond_label)

        # ── Export model ──────────────────────────────────────────────────
        model_path = f"{cond_prefix}_voting_regressor.joblib"
        joblib.dump(vr, model_path)
        _flush(f"\n  Exported VotingRegressor → {model_path}")

        results_summary[condition] = {
            "base_results": base_results,
            "vr": vr,
            "vr_mae": vr_mae,
            "vr_rmse": vr_rmse,
            "vr_r2": vr_r2,
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
