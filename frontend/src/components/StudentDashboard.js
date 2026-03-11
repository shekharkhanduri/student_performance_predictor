import React, { useState } from 'react';
import axios from 'axios';
import PredictionResult from './PredictionResult';
import RecommendationCard from './RecommendationCard';

const FIELDS = [
  { key: 'cgpa', label: 'CGPA', min: 0, max: 10, step: 0.1, default: 7.0 },
  { key: 'gpa', label: 'GPA', min: 0, max: 10, step: 0.1, default: 7.0 },
  { key: 'gpa_slope', label: 'GPA Slope', min: -1, max: 1, step: 0.01, default: 0.0 },
  { key: 'midterm_score', label: 'Midterm Score', min: 0, max: 100, step: 1, default: 60 },
  { key: 'attendance_pct', label: 'Attendance %', min: 0, max: 100, step: 1, default: 75 },
  { key: 'avg_assignment_score', label: 'Avg Assignment Score', min: 0, max: 100, step: 1, default: 70 },
  { key: 'backlogs', label: 'Backlogs', min: 0, max: 8, step: 1, default: 0 },
  { key: 'avg_quiz_score', label: 'Avg Quiz Score', min: 0, max: 100, step: 1, default: 65 },
  { key: 'assignment_completion_rate', label: 'Assignment Completion Rate', min: 0, max: 100, step: 1, default: 80 },
  { key: 'prev_semester_score', label: 'Previous Semester Score', min: 0, max: 10, step: 0.1, default: 7.0 },
  { key: 'absences', label: 'Absences', min: 0, max: 30, step: 1, default: 5 },
  { key: 'late_submissions', label: 'Late Submissions', min: 0, max: 15, step: 1, default: 2 },
];

const defaultValues = Object.fromEntries(FIELDS.map((f) => [f.key, f.default]));

export default function StudentDashboard() {
  const [formData, setFormData] = useState(defaultValues);
  const [extracurricular, setExtracurricular] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (key, value) => {
    setFormData((prev) => ({ ...prev, [key]: parseFloat(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = { ...formData, extracurricular: extracurricular ? 1 : 0 };

    try {
      const [predRes, recRes] = await Promise.all([
        axios.post('/api/predict', payload),
        axios.post('/api/recommend', payload),
      ]);
      setResult({ prediction: predRes.data, recommendation: recRes.data });
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to fetch prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Student Dashboard</h2>
        <p className="text-gray-500 mt-1">Enter your academic details to get a performance prediction and personalized recommendations.</p>
      </div>

      <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 mb-6">
          {FIELDS.map((field) => (
            <div key={field.key}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {field.label}
              </label>
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.step}
                value={formData[field.key]}
                onChange={(e) => handleChange(field.key, e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>
          ))}

          {/* Extracurricular toggle */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Extracurricular Activities
            </label>
            <button
              type="button"
              onClick={() => setExtracurricular((v) => !v)}
              className={`flex items-center space-x-3 px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${
                extracurricular
                  ? 'bg-blue-600 border-blue-600 text-white'
                  : 'bg-white border-gray-300 text-gray-600 hover:border-blue-400'
              }`}
            >
              <span
                className={`w-8 h-5 rounded-full inline-flex items-center transition-colors ${
                  extracurricular ? 'bg-blue-300' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`w-4 h-4 bg-white rounded-full shadow transform transition-transform ${
                    extracurricular ? 'translate-x-4' : 'translate-x-0.5'
                  }`}
                />
              </span>
              <span>{extracurricular ? 'Yes' : 'No'}</span>
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold px-8 py-3 rounded-lg transition-colors shadow-sm"
        >
          {loading ? (
            <span className="flex items-center space-x-2">
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              <span>Analyzing...</span>
            </span>
          ) : (
            'Predict & Analyze'
          )}
        </button>
      </form>

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 space-y-6">
          <PredictionResult score={result.prediction.predicted_score} />

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Top 3 Recommendations</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {result.recommendation.recommendations?.map((rec) => (
                <RecommendationCard key={rec.feature} rec={rec} />
              ))}
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-800">
            💡 If you follow all 3 recommendations, your score could improve from{' '}
            <strong>{result.recommendation.current_score?.toFixed(2)}</strong> to{' '}
            <strong>{result.recommendation.potential_score?.toFixed(2)}</strong>
          </div>
        </div>
      )}
    </div>
  );
}
