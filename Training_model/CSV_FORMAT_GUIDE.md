# CSV Upload Format Guide

This guide shows the exact format required for bulk CSV uploads to the system.

## Key Requirements

1. **student_id** column is mandatory (integer, must be > 0)
2. **Consistent naming**: Use exact field names as shown below
3. **Data types**: Match the types specified for each field
4. **Encoding**: UTF-8 (standard CSV encoding)

## Dataset 1 CSV Format

Use this format when uploading Dataset 1 (6 features) data.

### Required Columns
- `student_id` (integer, unique)
- `student_name` (string, optional)
- `Hours_Studied` (float)
- `Previous_Scores` (float, 0-100)
- `Extracurricular_Activities` (string: "Yes" or "No")
- `Sleep_Hours` (float)
- `Sample_Question_Papers_Practiced` (integer)

### Example CSV (Dataset 1)
```csv
student_id,student_name,Hours_Studied,Previous_Scores,Extracurricular_Activities,Sleep_Hours,Sample_Question_Papers_Practiced
101,Alice Johnson,6.5,75.0,Yes,7.0,5
102,Bob Smith,5.0,68.5,No,6.5,2
103,Clara Brown,8.0,82.0,Yes,8.0,8
```

### Notes
- `student_id` must be unique per student
- If you upload student 101 again, the previous record is updated
- `student_name` can be omitted (use empty string or skip the column)
- Hours must be between 0-9 (typically)
- Previous_Scores must be 0-100
- Sleep_Hours typically 4-9 range
- Sample papers typically 0-10 range

---

## Dataset 2 CSV Format

Use this format when uploading Dataset 2 (20 features) data.

### Required Columns
- `student_id` (integer, unique)
- `student_name` (string, optional)
- `Hours_Studied` (float)
- `Attendance` (float, 0-100)
- `Gender` (string: "Male" or "Female")
- `Parental_Involvement` (string: "Low", "Medium", or "High")
- `Access_to_Resources` (string: "Low", "Medium", or "High")
- `Extracurricular_Activities` (string: "Yes" or "No")
- `Sleep_Hours` (float)
- `Previous_Scores` (float, 0-100)
- `Motivation_Level` (string: "Low", "Medium", or "High")
- `Internet_Access` (string: "Yes" or "No")
- `Tutoring_Sessions` (integer)
- `Family_Income` (string: "Low", "Medium", or "High")
- `Teacher_Quality` (string: "Low", "Medium", or "High")
- `School_Type` (string: "Public" or "Private")
- `Peer_Influence` (string: "Positive", "Neutral", or "Negative")
- `Physical_Activity` (integer)
- `Learning_Disabilities` (string: "Yes" or "No")
- `Parental_Education_Level` (string: "High School", "College", or "Postgraduate")
- `Distance_from_Home` (string: "Near", "Moderate", or "Far")

### Example CSV (Dataset 2)
```csv
student_id,student_name,Hours_Studied,Attendance,Gender,Parental_Involvement,Access_to_Resources,Extracurricular_Activities,Sleep_Hours,Previous_Scores,Motivation_Level,Internet_Access,Tutoring_Sessions,Family_Income,Teacher_Quality,School_Type,Peer_Influence,Physical_Activity,Learning_Disabilities,Parental_Education_Level,Distance_from_Home
201,Emma Wilson,7.5,85.0,Female,High,High,Yes,7.5,78.0,High,Yes,2,Medium,High,Public,Positive,4,No,College,Moderate
202,Raj Patel,8.0,90.0,Male,High,High,Yes,8.0,85.0,High,Yes,1,High,High,Private,Positive,5,No,Postgraduate,Near
203,Sarah Lee,5.0,75.0,Female,Low,Medium,No,6.0,70.0,Medium,Yes,0,Low,Low,Public,Neutral,2,Yes,High School,Far
```

### Field Value Constraints

| Field | Type | Valid Values |
|-------|------|--------------|
| student_id | Integer | Any positive integer (>0) |
| Hours_Studied | Float | 0-44 typically |
| Attendance | Float | 0-100 |
| Gender | String | "Male", "Female" |
| Parental_Involvement | String | "Low", "Medium", "High" |
| Access_to_Resources | String | "Low", "Medium", "High" |
| Extracurricular_Activities | String | "Yes", "No" |
| Sleep_Hours | Float | 0-10 typically |
| Previous_Scores | Float | 0-100 |
| Motivation_Level | String | "Low", "Medium", "High" |
| Internet_Access | String | "Yes", "No" |
| Tutoring_Sessions | Integer | 0-8 typically |
| Family_Income | String | "Low", "Medium", "High" |
| Teacher_Quality | String | "Low", "Medium", "High" |
| School_Type | String | "Public", "Private" |
| Peer_Influence | String | "Positive", "Neutral", "Negative" |
| Physical_Activity | Integer | 0-6 typically |
| Learning_Disabilities | String | "Yes", "No" |
| Parental_Education_Level | String | "High School", "College", "Postgraduate" |
| Distance_from_Home | String | "Near", "Moderate", "Far" |

---

## Common Issues and Solutions

### Issue: "student_id is required in CSV"
- **Cause**: Missing `student_id` column
- **Fix**: Add the `student_id` column as the first column

### Issue: "student_id must be a positive integer"
- **Cause**: student_id is 0, negative, or non-numeric
- **Fix**: Use positive integers only (1, 2, 3, ...)

### Issue: "Missing required features"
- **Cause**: Column names don't match exactly
- **Fix**: Use exact names: `Hours_Studied` (not `hours_studied` or `Hours Studied`)

### Issue: "Invalid value for [Field Name]"
- **Cause**: Value not in allowed list (e.g., "male" instead of "Male")
- **Fix**: Check capitalization and exact spelling from the table above

### Issue: "CSV does not match Dataset 1 or Dataset 2"
- **Cause**: Not all required columns present
- **Fix**: Ensure all required columns are in the CSV

---

## Upsert Behavior

When you upload a CSV with a student_id that already exists:

1. The **existing record is updated** with new data
2. All fields are refreshed with the latest values
3. Predictions are recalculated
4. The response confirms this as an update

### Example Scenario:
```csv
# First upload (200 rows processed, 200 stored)
student 101: score 75, risk "Stable"

# Second upload with same student 101 (200 rows processed, 200 stored)
student 101: score 78, risk "Stable"  <- Updated with new prediction
```

---

## Best Practices

1. **Use consistent student IDs** across uploads (e.g., enrollment numbers)
2. **Validate data locally** before uploading large batches
3. **Test with a small sample** first (5-10 rows)
4. **Keep backups** of source data
5. **Use UTF-8 encoding** for special characters
6. **Verify field names** match exactly before uploading

---

## Quick Start Example

### Create a sample Dataset 2 CSV:
Save as `students.csv`:
```
student_id,student_name,Hours_Studied,Attendance,Gender,Parental_Involvement,Access_to_Resources,Extracurricular_Activities,Sleep_Hours,Previous_Scores,Motivation_Level,Internet_Access,Tutoring_Sessions,Family_Income,Teacher_Quality,School_Type,Peer_Influence,Physical_Activity,Learning_Disabilities,Parental_Education_Level,Distance_from_Home
1,John Doe,6,80,Male,Medium,Medium,Yes,7,75,Medium,Yes,1,Medium,Medium,Public,Neutral,3,No,College,Moderate
2,Jane Smith,7,85,Female,High,High,Yes,8,80,High,Yes,2,High,High,Private,Positive,4,No,College,Near
```

### Upload via frontend:
1. Go to "📤 Bulk CSV Upload" tab
2. Click "Upload & Predict"
3. Select `students.csv`
4. Click "Upload & Predict"
5. Wait for processing (~10-15 seconds for 2 rows)
