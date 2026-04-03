# Deployment Checklist: Student ID as Primary Key Implementation

## Pre-Deployment Verification

### Code Review
- [ ] All Python files compile without syntax errors
- [ ] All TypeScript/React files pass linting
- [ ] No unresolved imports or dependencies

### File Changes Summary
- ✅ [backend/core/config.py](backend/core/config.py) - Updated model paths to use aliases
- ✅ [backend/models/student.py](backend/models/student.py) - Added student_id field
- ✅ [backend/schemas/prediction.py](backend/schemas/prediction.py) - Added student_id requirement
- ✅ [backend/services/student_service.py](backend/services/student_service.py) - Implemented upsert logic
- ✅ [backend/api/routers/students.py](backend/api/routers/students.py) - Updated endpoints for student_id
- ✅ [frontend/src/components/PredictionForm.jsx](frontend/src/components/PredictionForm.jsx) - Added student_id input
- ✅ [frontend/src/components/UploadPanel.jsx](frontend/src/components/UploadPanel.jsx) - Added CSV format docs
- ✅ [frontend/src/styles.css](frontend/src/styles.css) - Added required-indicator styling

### Documentation Created
- ✅ [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - Detailed implementation guide
- ✅ [CSV_FORMAT_GUIDE.md](CSV_FORMAT_GUIDE.md) - CSV upload format and examples
- ✅ [BACKEND_MODEL_INTEGRATION.md](BACKEND_MODEL_INTEGRATION.md) - Backend integration guide

---

## Database Migration Steps

### Step 1: Backup Current Database
```bash
# For Neon PostgreSQL (via pg_dump)
pg_dump "your-postgres-connection-string" > db_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Step 2: Run Migration SQL

Connect to your PostgreSQL database and run:

```sql
-- Start transaction
BEGIN;

-- Add student_id column to both tables
ALTER TABLE student_data_ds1 
ADD COLUMN student_id INTEGER UNIQUE NOT NULL DEFAULT 0;

ALTER TABLE student_data_ds2 
ADD COLUMN student_id INTEGER UNIQUE NOT NULL DEFAULT 0;

-- Populate student_id from existing id values (if records exist)
UPDATE student_data_ds1 SET student_id = id WHERE student_id = 0;
UPDATE student_data_ds2 SET student_id = id WHERE student_id = 0;

-- Remove the DEFAULT constraint
ALTER TABLE student_data_ds1 
ALTER COLUMN student_id DROP DEFAULT;

ALTER TABLE student_data_ds2 
ALTER COLUMN student_id DROP DEFAULT;

-- Create indexes for fast lookups
CREATE INDEX idx_student_data_ds1_student_id ON student_data_ds1(student_id);
CREATE INDEX idx_student_data_ds2_student_id ON student_data_ds2(student_id);

-- Commit all changes
COMMIT;
```

### Step 3: Verify Migration
```sql
-- Check if columns were added
\d student_data_ds1
\d student_data_ds2

-- Verify data integrity
SELECT COUNT(*) as total_records FROM student_data_ds1;
SELECT COUNT(*) as null_student_ids FROM student_data_ds1 WHERE student_id IS NULL;
SELECT COUNT(*) as duplicate_student_ids FROM student_data_ds1 GROUP BY student_id HAVING COUNT(*) > 1;
```

---

## Backend Deployment

### Prerequisites
- [ ] Python 3.9+
- [ ] Poetry or pip with requirements.txt
- [ ] PostgreSQL connection URL

### Steps

1. **Update Dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```

2. **Update Environment Variables**
   ```bash
   # Verify these are set (or use defaults)
   export DS1_MODEL_PATH=/path/to/backend/mlmodel/model_ds1.joblib
   export DS2_MODEL_PATH=/path/to/backend/mlmodel/model_ds2.joblib
   export DS1_SCALER_PATH=/path/to/backend/mlmodel/scaler_ds1.joblib
   export DS2_SCALER_PATH=/path/to/backend/mlmodel/scaler_ds2.joblib
   export DATABASE_URL=your-neon-postgres-url
   ```

3. **Verify Model Artifacts Exist**
   ```bash
   ls -lh backend/mlmodel/*.joblib
   # Should show 4 files in backend/mlmodel/
   ```

4. **Run Startup Health Check**
   ```bash
   # Backend should be able to:
   # - Load both scalers
   # - Load both models
   # - Connect to database
   # - Create test predictions
   
   python -c "
   from backend.services.ml_service import load_model, _load_ds1_scaler, _load_ds2_scaler
   try:
       _load_ds1_scaler()
       _load_ds2_scaler()
       load_model('ds1')
       load_model('ds2')
       print('✓ All models and scalers loaded successfully')
   except Exception as e:
       print(f'✗ Error: {e}')
   "
   ```

5. **Deploy Backend**
   ```bash
   # Using uvicorn
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   
   # OR using Docker/production deployment
   # (Follow your deployment platform's guide)
   ```

6. **Verify Backend Health**
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status":"ok"}
   ```

---

## Frontend Deployment

### Prerequisites
- [ ] Node.js 16+
- [ ] npm or yarn

### Steps

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Verify Environment Configuration**
   ```bash
   # Check frontend/lib/api.js for correct backend URL
   # Should point to your backend deployment
   ```

3. **Build Frontend**
   ```bash
   npm run build
   # Creates optimized production build in dist/ folder
   ```

4. **Deploy Frontend**
   ```bash
   # Option A: Serve locally (development)
   npm run dev
   
   # Option B: Deploy dist/ folder to static hosting
   # (GitHub Pages, Vercel, Netlify, AWS S3, etc.)
   ```

5. **Test Frontend**
   - [ ] Navigate to http://localhost:5173 (or your deployment URL)
   - [ ] Forms show student_id as required field
   - [ ] CSV upload shows format requirements

---

## Testing Procedure

### Phase 1: Single Prediction (Manual Testing)

1. **Open web app** and go to prediction form
2. **Test Dataset 1:**
   ```
   Student ID: 1001
   Name: Test Student 1
   Hours Studied: 6.5
   Previous Scores: 75
   Extracurricular: Yes
   Sleep Hours: 7
   Papers Practiced: 5
   Click: Submit Survey Response
   ```
   - [ ] Prediction succeeds
   - [ ] Result shows "Saved as student #1001"
   - [ ] Risk level displayed (Stable/Borderline/At-Risk)

3. **Test Update (Same Student ID):**
   ```
   Student ID: 1001 (same as above)
   Hours Studied: 8.0 (different value)
   Change other fields...
   Click: Submit Survey Response
   ```
   - [ ] Prediction succeeds
   - [ ] Result shows "Saved as student #1001" (updated, not new)
   - [ ] New score reflects updated input

4. **Test Dataset 2:**
   - [ ] Switch to Dataset 2 survey
   - [ ] Fill in all required fields with valid values
   - [ ] Submit and verify prediction

### Phase 2: Bulk CSV Upload (Testing)

1. **Create test CSV** (use [CSV_FORMAT_GUIDE.md](CSV_FORMAT_GUIDE.md))
   ```csv
   student_id,student_name,Hours_Studied,Attendance,Gender,Parental_Involvement,Access_to_Resources,Extracurricular_Activities,Sleep_Hours,Previous_Scores,Motivation_Level,Internet_Access,Tutoring_Sessions,Family_Income,Teacher_Quality,School_Type,Peer_Influence,Physical_Activity,Learning_Disabilities,Parental_Education_Level,Distance_from_Home
   2001,Test A,7,85,Male,High,High,Yes,8,80,High,Yes,2,Medium,High,Public,Positive,4,No,College,Moderate
   2002,Test B,6,75,Female,Medium,Medium,No,7,70,Medium,Yes,1,Low,Medium,Public,Neutral,3,No,High School,Far
   ```

2. **Upload test CSV**
   - [ ] Go to "📤 Bulk CSV Upload" tab
   - [ ] Upload CSV
   - [ ] See success message with batch ID
   - [ ] Returned student_ids match input (2001, 2002)

3. **Upload same CSV again**
   - [ ] Records updated (not duplicated)
   - [ ] Batch ID different
   - [ ] Same student_ids returned

4. **Test error cases:**
   - [ ] CSV without student_id column → error message
   - [ ] CSV with student_id = 0 → error on row
   - [ ] CSV with non-integer student_id → error on row
   - [ ] CSV with invalid categorical values → validation error

### Phase 3: API Testing (with curl or Postman)

1. **Test /predict endpoint:**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "student_id": 3001,
       "student_name": "API Test",
       "dataset_type": "ds2",
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
   - [ ] Returns prediction with student_id 3001
   - [ ] No error

2. **Test /student/{student_id} endpoint:**
   ```bash
   curl http://localhost:8000/student/3001
   ```
   - [ ] Returns record for student 3001
   - [ ] All fields populated correctly

3. **Test missing student_id:**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"dataset_type": "ds2", ...}'
   ```
   - [ ] Returns 422 validation error
   - [ ] Error message mentions missing student_id

---

## Rollback Plan

If critical issues arise post-deployment:

1. **Revert Code**
   ```bash
   git revert HEAD  # Or deploy previous stable version
   ```

2. **Restore Database**
   ```bash
   psql your-connection-string < db_backup_YYYYMMDD_HHMMSS.sql
   ```

3. **Restart Services**
   - Restart backend API
   - Clear frontend cache and rebuild

---

## Post-Deployment Verification

### Day 1 Checklist
- [ ] Predictions submitting successfully
- [ ] Student IDs persisting correctly in database
- [ ] CSV uploads working without errors
- [ ] No error logs in backend
- [ ] No console errors in browser DevTools

### Week 1 Checklist
- [ ] Multiple student updates tested (same student_id re-uploaded)
- [ ] CSV bulk operations on 50+ records
- [ ] High-volume prediction requests (if applicable)
- [ ] Database queries performing well (check slow query logs)
- [ ] User feedback collected and issues addressed

### Monitoring Setup
set up monitoring for:
- Backend API response times
- Database query performance
- Error rate tracking
- CSV upload success rate
- Prediction accuracy metrics

---

## Support and Documentation

- **API Documentation**: [BACKEND_MODEL_INTEGRATION.md](BACKEND_MODEL_INTEGRATION.md)
- **CSV Format Guide**: [CSV_FORMAT_GUIDE.md](CSV_FORMAT_GUIDE.md)
- **Implementation Details**: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- **Training Script**: [README.md](README.md)

For questions or issues:
1. Check documentation files above
2. Review error logs
3. Verify database migration completed successfully
4. Check model artifacts exist at configured paths

---

## Success Criteria

Deployment is successful when:
- ✅ All HTTP endpoints respond correctly
- ✅ student_id is required and enforced
- ✅ Duplicate student_id updates (not inserts) existing records
- ✅ CSV uploads work with proper format
- ✅ Database queries use student_id indexes
- ✅ Frontend displays required field indicator for student_id
- ✅ No 500 errors in production logs
- ✅ No unhandled exceptions
- ✅ Response times < 2 seconds for predictions
- ✅ Bulk CSV processing < 30 seconds for typical batch (100-1000 rows)
