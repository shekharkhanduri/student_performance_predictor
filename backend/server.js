import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';

const app = express();
const PORT = 5000;
const ML_BASE = 'http://localhost:8000';

app.use(cors());
app.use(express.json());

const STUDENTS = [
  {
    id: 1, name: 'Aarav Sharma',
    cgpa: 9.1, gpa: 9.3, gpa_slope: 0.4, midterm_score: 92,
    attendance_pct: 95, avg_assignment_score: 91, backlogs: 0,
    avg_quiz_score: 90, assignment_completion_rate: 98,
    prev_semester_score: 9.0, absences: 1, late_submissions: 0, extracurricular: 1,
  },
  {
    id: 2, name: 'Priya Patel',
    cgpa: 8.2, gpa: 8.5, gpa_slope: 0.2, midterm_score: 80,
    attendance_pct: 88, avg_assignment_score: 83, backlogs: 0,
    avg_quiz_score: 81, assignment_completion_rate: 90,
    prev_semester_score: 8.1, absences: 3, late_submissions: 1, extracurricular: 1,
  },
  {
    id: 3, name: 'Rohan Gupta',
    cgpa: 6.8, gpa: 7.0, gpa_slope: 0.0, midterm_score: 65,
    attendance_pct: 75, avg_assignment_score: 68, backlogs: 1,
    avg_quiz_score: 66, assignment_completion_rate: 75,
    prev_semester_score: 6.7, absences: 8, late_submissions: 3, extracurricular: 0,
  },
  {
    id: 4, name: 'Sneha Reddy',
    cgpa: 5.9, gpa: 6.0, gpa_slope: -0.1, midterm_score: 55,
    attendance_pct: 68, avg_assignment_score: 60, backlogs: 2,
    avg_quiz_score: 57, assignment_completion_rate: 65,
    prev_semester_score: 6.0, absences: 12, late_submissions: 5, extracurricular: 0,
  },
  {
    id: 5, name: 'Vikram Singh',
    cgpa: 4.5, gpa: 4.6, gpa_slope: -0.3, midterm_score: 38,
    attendance_pct: 52, avg_assignment_score: 40, backlogs: 5,
    avg_quiz_score: 38, assignment_completion_rate: 45,
    prev_semester_score: 4.7, absences: 20, late_submissions: 10, extracurricular: 0,
  },
  {
    id: 6, name: 'Ananya Joshi',
    cgpa: 4.2, gpa: 4.3, gpa_slope: -0.2, midterm_score: 35,
    attendance_pct: 48, avg_assignment_score: 38, backlogs: 6,
    avg_quiz_score: 36, assignment_completion_rate: 42,
    prev_semester_score: 4.4, absences: 22, late_submissions: 12, extracurricular: 0,
  },
];

async function callML(path, body) {
  const res = await fetch(`${ML_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ML service error (${res.status}): ${text}`);
  }
  return res.json();
}

app.post('/api/predict', async (req, res) => {
  try {
    const result = await callML('/predict', req.body);
    res.json(result);
  } catch (err) {
    res.status(502).json({ error: 'Failed to reach ML service', detail: err.message });
  }
});

app.post('/api/recommend', async (req, res) => {
  try {
    const result = await callML('/recommend', req.body);
    res.json(result);
  } catch (err) {
    res.status(502).json({ error: 'Failed to reach ML service', detail: err.message });
  }
});

app.get('/api/students', (_req, res) => {
  res.json(STUDENTS);
});

app.get('/api/students/:id', (req, res) => {
  const student = STUDENTS.find((s) => s.id === parseInt(req.params.id, 10));
  if (!student) return res.status(404).json({ error: 'Student not found' });
  res.json(student);
});

app.get('/api/faculty/summary', async (_req, res) => {
  try {
    const results = await Promise.all(
      STUDENTS.map(async (s) => {
        const { id, name, ...features } = s;
        const [pred, rec] = await Promise.all([
          callML('/predict', features),
          callML('/recommend', features),
        ]);
        const score = pred.predicted_score;
        const risk =
          score >= 7 ? 'low' : score >= 5 ? 'medium' : 'high';
        const topWeak = rec.recommendations?.[0]?.feature ?? 'N/A';
        return {
          id, name,
          predicted_score: score,
          risk_level: risk,
          top_weak_feature: topWeak,
          recommendations: rec.recommendations,
        };
      })
    );

    const totalStudents = results.length;
    const averageScore = parseFloat(
      (results.reduce((sum, r) => sum + r.predicted_score, 0) / totalStudents).toFixed(2)
    );
    const atRiskCount = results.filter((r) => r.predicted_score < 5).length;

    const featureCount = {};
    results.forEach((r) => {
      r.recommendations?.forEach((rec) => {
        featureCount[rec.feature] = (featureCount[rec.feature] ?? 0) + 1;
      });
    });
    const commonWeakFeatures = Object.entries(featureCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([f]) => f);

    res.json({ totalStudents, averageScore, atRiskCount, commonWeakFeatures, students: results });
  } catch (err) {
    res.status(502).json({ error: 'Failed to compute faculty summary', detail: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Backend listening on http://localhost:${PORT}`);
});
