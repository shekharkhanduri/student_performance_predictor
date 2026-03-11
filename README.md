# Student Performance Predictor

A full-stack web application that predicts student academic performance and provides personalised recommendations using machine learning.

---

## Architecture

```
┌─────────────────────┐      HTTP       ┌──────────────────────┐      HTTP      ┌──────────────────────┐
│   React Frontend    │ ─────────────▶  │  Node.js / Express   │ ─────────────▶ │  Python / FastAPI    │
│   (port 3000)       │ ◀─────────────  │  Backend (port 5000) │ ◀───────────── │  ML Service (8000)   │
└─────────────────────┘                 └──────────────────────┘                └──────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Tailwind CSS, Recharts, Axios |
| Backend | Node.js, Express, node-fetch |
| ML Service | Python, FastAPI, scikit-learn, joblib, NumPy, Pandas |

---

## Prerequisites

- Node.js 18+
- Python 3.8+
- npm

---

## Setup & Run

### 1. ML Service

```bash
cd ml-service
pip install -r requirements.txt
python train_model.py        # generates student_model.pkl and scaler.pkl
uvicorn main:app --reload --port 8000
```

> **Note:** If you skip `train_model.py`, the FastAPI service will auto-train the model on first startup.

### 2. Backend

```bash
cd backend
npm install
npm start
```

### 3. Frontend

```bash
cd frontend
npm install
npm start
```

---

## Port Configuration

| Service | Port |
|---------|------|
| React Frontend | 3000 |
| Express Backend | 5000 |
| FastAPI ML Service | 8000 |

---

## API Endpoints

### Express Backend (`localhost:5000`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/predict` | Predict performance score for a student |
| `POST` | `/api/recommend` | Get top 3 improvement recommendations |
| `GET` | `/api/students` | List all 6 mock students |
| `GET` | `/api/students/:id` | Get a single student by ID |
| `GET` | `/api/faculty/summary` | Aggregated class analytics |

### FastAPI ML Service (`localhost:8000`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Returns `{ predicted_score: float }` |
| `POST` | `/recommend` | Returns recommendations with score deltas |

---

## Features

- **13-feature ML model** trained on synthetic data using `RandomForestRegressor`
- **Auto-training** – model is generated automatically if `.pkl` files are missing
- **Student Dashboard** – input form, color-coded score badge, performance meter, recommendation cards, what-if summary
- **Faculty Dashboard** – summary cards, interactive student table with expandable recommendations, bar chart
- **Benchmark comparison** – gaps against a high-performer profile surface the top 3 areas to improve
- **Responsive design** with Tailwind CSS
