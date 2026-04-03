# Local Development Setup & Quick Start

## Prerequisites

### 1. Train Models First
Before starting the backend, you need to generate the models:

```bash
cd /home/lawliet/student-prediction
python3 student_performance_predictor.py
```

This will create:
- `backend/mlmodel/model_ds1.joblib`
- `backend/mlmodel/model_ds2.joblib`
- `backend/mlmodel/scaler_ds1.joblib`
- `backend/mlmodel/scaler_ds2.joblib`

Verify they exist:
```bash
ls -lh model_ds*.joblib scaler_ds*.joblib
```

### 2. Install Dependencies

```bash
# Backend dependencies (if not already installed)
pip3 install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

---

## Starting Backend (Terminal 1)

```bash
cd /home/lawliet/student-prediction

# Make sure models exist
ls -lh backend/mlmodel/*.joblib

# Start backend server
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Verify backend is running:**
```bash
# In another terminal
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "ok",
  "model_loaded": true,
  "models": {
    "ds1": { "path": "...", "loaded": true },
    "ds2": { "path": "...", "loaded": true }
  }
}
```

---

## Starting Frontend (Terminal 2)

```bash
cd /home/lawliet/student-prediction/frontend

# Start development server
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

---

## Accessing the App

Open browser to: **http://localhost:5173**

---

## Troubleshooting

### Issue: "Failed to load resource: net::ERR_FAILED"

**Cause 1: Backend not running**
- Check Terminal 1 for backend status
- Start backend with uvicorn command above

**Cause 2: Models don't exist**
```bash
# Verify models exist
ls -lh /home/lawliet/student-prediction/backend/mlmodel/*.joblib

# If missing, train them
cd /home/lawliet/student-prediction
python3 student_performance_predictor.py
```

**Cause 3: Port 8000 already in use**
```bash
# Find what's using port 8000
lsof -i :8000

# Kill it if needed
kill -9 <PID>
```

### Issue: "Connection refused" error

**Cause:** Backend not listening on 8000
- Restart backend with full output
- Check for startup errors in Terminal 1

### Issue: Database connection error

**Cause:** PostgreSQL/Neon not accessible
- Check DATABASE_URL environment variable
- For local testing, you can bypass database with SQLite:
  ```bash
  # Edit backend/core/config.py if you want a local fallback
  # DATABASE_URL = "sqlite:///./test.db"
  ```

---

## Recommended Terminal Layout

**Terminal 1 (Backend):**
```bash
cd /home/lawliet/student-prediction
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
cd /home/lawliet/student-prediction/frontend
npm run dev
```

**Terminal 3 (Optional - for manual testing):**
```bash
# Test individual endpoints
curl http://localhost:8000/health
curl http://localhost:8000/students
```

---

## Development Workflow

1. **Edit backend code** → Auto-reloads (--reload flag)
2. **Edit frontend code** → Auto-reloads (Vite HMR)
3. **Check console** for errors in either terminal
4. **Browser DevTools** (F12) to inspect frontend requests

---

## CORS Configuration

CORS is already enabled in `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This allows:
- Any origin (including http://localhost:5173)
- Any HTTP method
- Any headers

For production, restrict to your domain:
```python
allow_origins=["https://yourdomain.com"],
```

---

## Quick Health Check

Once both are running, test all endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Get all students
curl http://localhost:8000/students

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": 1,
    "dataset_type": "ds2",
    "student_name": "Test",
    "Hours_Studied": 7,
    "Attendance": 85,
    "Gender": "Male",
    "Parental_Involvement": "High",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 8,
    "Previous_Scores": 80,
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 2,
    "Family_Income": "Medium",
    "Teacher_Quality": "High",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Physical_Activity": 4,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Moderate"
  }'
```

If all return 200 OK, you're good to go! 🎉
