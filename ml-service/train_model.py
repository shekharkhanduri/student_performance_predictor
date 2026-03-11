import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = [
    'cgpa', 'gpa', 'gpa_slope', 'midterm_score', 'attendance_pct',
    'avg_assignment_score', 'backlogs', 'avg_quiz_score',
    'assignment_completion_rate', 'prev_semester_score',
    'absences', 'late_submissions', 'extracurricular'
]

def generate_data(n=600, seed=42):
    rng = np.random.default_rng(seed)

    cgpa = rng.uniform(4.0, 10.0, n)
    gpa = cgpa + rng.uniform(-0.5, 0.5, n)
    gpa = np.clip(gpa, 4.0, 10.0)
    gpa_slope = rng.uniform(-0.5, 0.6, n)
    midterm_score = rng.uniform(30, 100, n)
    attendance_pct = rng.uniform(40, 100, n)
    avg_assignment_score = rng.uniform(30, 100, n)
    backlogs = rng.integers(0, 9, n).astype(float)
    avg_quiz_score = rng.uniform(30, 100, n)
    assignment_completion_rate = rng.uniform(40, 100, n)
    prev_semester_score = rng.uniform(4.0, 10.0, n)
    absences = rng.integers(0, 31, n).astype(float)
    late_submissions = rng.integers(0, 16, n).astype(float)
    extracurricular = rng.integers(0, 2, n).astype(float)

    score = (
        0.20 * (cgpa / 10.0) +
        0.15 * (gpa / 10.0) +
        0.05 * ((gpa_slope + 0.5) / 1.0) +
        0.12 * (midterm_score / 100.0) +
        0.10 * (attendance_pct / 100.0) +
        0.10 * (avg_assignment_score / 100.0) +
        0.08 * (1.0 - backlogs / 8.0) +
        0.08 * (avg_quiz_score / 100.0) +
        0.05 * (assignment_completion_rate / 100.0) +
        0.04 * (prev_semester_score / 10.0) +
        0.015 * (1.0 - absences / 30.0) +
        0.015 * (1.0 - late_submissions / 15.0) +
        0.02 * extracurricular
    )

    score = score * 10.0
    score += rng.normal(0, 0.2, n)
    score = np.clip(score, 0.0, 10.0)

    df = pd.DataFrame({
        'cgpa': cgpa,
        'gpa': gpa,
        'gpa_slope': gpa_slope,
        'midterm_score': midterm_score,
        'attendance_pct': attendance_pct,
        'avg_assignment_score': avg_assignment_score,
        'backlogs': backlogs,
        'avg_quiz_score': avg_quiz_score,
        'assignment_completion_rate': assignment_completion_rate,
        'prev_semester_score': prev_semester_score,
        'absences': absences,
        'late_submissions': late_submissions,
        'extracurricular': extracurricular,
        'score': score,
    })
    return df


def train_and_save(model_path='student_model.pkl', scaler_path='scaler.pkl'):
    df = generate_data()
    X = df[FEATURES].values
    y = df['score'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}, scaler saved to {scaler_path}")


if __name__ == '__main__':
    train_and_save()
