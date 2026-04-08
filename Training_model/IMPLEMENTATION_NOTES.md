# Implementation Summary: Student ID as Primary Key with Upsert Logic

## Overview
This document summarizes the changes made to implement `student_id` as a required field with upsert logic for updating existing student records.

## Changes Made

### 1. Backend Configuration (`backend/core/config.py`)
- **Updated model path defaults** to use model aliases:
  - DS1_MODEL_PATH: `backend/mlmodel/model_ds1.joblib` (instead of `outputs/ds1_all_features_stacking_regressor.joblib`)
  - DS2_MODEL_PATH: `backend/mlmodel/model_ds2.joblib` (instead of `outputs/ds2_all_features_stacking_regressor.joblib`)
- Environment variables remain the same for override capability

### 2. Database Models (`backend/models/student.py`)
- **Added `student_id` field** as a unique, indexed, required column
  - Type: Integer, nullable=False
  - unique=True
  - index=True
- Both `StudentDataDS1` and `StudentDataDS2` inherit this field via `_StudentCommon` base class
- Kept existing `id` auto-increment primary key for internal references

### 3. API Schemas (`backend/schemas/prediction.py`)
- **Added `student_id` as required field** in `PredictRequest`
  - Type: int
  - Validation: gt=0 (must be positive)
  - Description: "Unique student identifier"
- `StudentDiagnostic` already maps student_id correctly from row.student_id

### 4. Backend Services (`backend/services/student_service.py`)
- **Implemented upsert logic** in `store_student()` function:
  - Checks if student_id exists for the given dataset_type
  - If exists: Updates all fields (features, predictions, risk_level, etc.)
  - If new: Inserts new record
- Updated function signature to require `student_id` parameter
- Both DS1 and DS2 record creation now support upsert

### 5. API Routes (`backend/api/routers/students.py`)

#### `/predict` Endpoint
- Now requires `student_id` in request payload
- Passes student_id to store_student for upsert
- Returns `student_id` in response (not internal id)

#### `/upload` Endpoint
- Requires `student_id` column in CSV
- Validates student_id is present and positive (>0)
- Returns student_id list in UploadSummary (not internal id)
- Implements upsert for bulk operations

#### `/student/{student_id}` GET Endpoint
- Now queries by `student_id` field instead of internal `id`
- Database lookup: `model.student_id == student_id`

### 6. Frontend Component (`frontend/src/components/PredictionForm.jsx`)
- **Added student_id input field** as required
  - Type: number, min=1
  - Validation: Form submission blocked if student_id is missing
  - Placed before student_name for logical flow
- Updated initial form states:
  - `initialDs1Form.student_id = ""`
  - `initialDs2Form.student_id = ""`
- Added client-side validation before API call
- Converts student_id to number before sending
- Includes student_id in server response confirmation

### 7. Frontend Upload Documentation (`frontend/src/components/UploadPanel.jsx`)
- Added collapsible "CSV Format Requirements" section
- Clearly documents student_id requirement
- Explains upsert behavior for duplicate student_ids

### 8. Frontend Styling (`frontend/src/styles.css`)
- Added `.required-indicator` style (red asterisk)
- Used for visual indication of required fields

## Database Migration Steps

Before deploying, you **must** update your database schema:

```sql
-- Add student_id column if not exists (adjust for your DB)
ALTER TABLE student_data_ds1 
ADD COLUMN student_id INTEGER UNIQUE NOT NULL;

ALTER TABLE student_data_ds2 
ADD COLUMN student_id INTEGER UNIQUE NOT NULL;

-- Create index on student_id for fast lookups
CREATE INDEX idx_student_data_ds1_student_id ON student_data_ds1(student_id);
CREATE INDEX idx_student_data_ds2_student_id ON student_data_ds2(student_id);
```

### PostgreSQL Example (if using Neon):
```sql
BEGIN;

ALTER TABLE student_data_ds1 
ADD COLUMN student_id INTEGER UNIQUE NOT NULL;

ALTER TABLE student_data_ds2 
ADD COLUMN student_id INTEGER UNIQUE NOT NULL;

CREATE INDEX idx_student_data_ds1_student_id ON student_data_ds1(student_id);
CREATE INDEX idx_student_data_ds2_student_id ON student_data_ds2(student_id);

COMMIT;
```

### Data Migration (if table already has records):
If you have existing records without student_id, populate using the internal id:
```sql
-- Map existing records: use internal id as student_id
UPDATE student_data_ds1 SET student_id = id WHERE student_id IS NULL;
UPDATE student_data_ds2 SET student_id = id WHERE student_id IS NULL;
```

## Naming Convention (Applied Throughout)
- Format: snake_case for all fields
- Examples:
  - `student_id` (in requests/DB)
  - `student_name` (in requests/DB)
  - `Hours_Studied`, `Previous_Scores`, etc. (feature names - PascalCase_with_underscores)
- This convention is consistent across:
  - Database columns (snake_case)
  - API payloads
  - CSV headers
  - Frontend form fields

## Testing Checklist

1. **Single Prediction Flow**
   - [ ] POST to `/predict` with new student_id → creates record
   - [ ] POST to `/predict` with same student_id → updates record
   - [ ] Verify response includes student_id (not internal id)

2. **Bulk Upload Flow**
   - [ ] Upload CSV with student_id column → creates/updates records
   - [ ] Upload same CSV again → all records updated with latest predictions
   - [ ] CSV without student_id → returns validation error

3. **Retrieval Flow**
   - [ ] GET `/student/{student_id}` retrieves by student_id
   - [ ] GET `/students` returns diagnostics with correct student_id
   - [ ] Filter by risk_level returns correct student_ids

4. **Error Cases**
   - [ ] POST `/predict` without student_id → 422 validation error
   - [ ] CSV with non-integer student_id → row error
   - [ ] CSV with student_id ≤ 0 → row error

## Breaking Changes

- ⚠️ **API Response**: `/predict` now returns `student_id` instead of internal `id`
- ⚠️ **CSV Upload**: CSV files must include `student_id` column
- ⚠️ **Database**: Requires migration to add `student_id` column
- ✅ Old `id` field still exists internally (not exposed in API)

## Rollback Plan (if needed)

If reverting is necessary:
1. Restore database from backup (pre-migration)
2. Revert code to previous commit
3. Redeploy backend

## Next Steps

1. **Database Migration**: Run migration SQL on your Neon PostgreSQL instance
2. **Backend Deployment**: Deploy updated backend code
3. **Frontend Deployment**: Deploy updated React frontend
4. **Testing**: Follow testing checklist above
5. **Documentation**: Update API docs if published externally
