import React, { useEffect, useState } from 'react';
import axios from 'axios';
import StudentTable from './StudentTable';
import ScoreChart from './ScoreChart';

const FEATURE_LABELS = {
  cgpa: 'CGPA',
  gpa: 'GPA',
  gpa_slope: 'GPA Slope',
  midterm_score: 'Midterm Score',
  attendance_pct: 'Attendance %',
  avg_assignment_score: 'Avg Assignment Score',
  backlogs: 'Backlogs',
  avg_quiz_score: 'Avg Quiz Score',
  assignment_completion_rate: 'Assignment Completion Rate',
  prev_semester_score: 'Previous Semester Score',
  absences: 'Absences',
  late_submissions: 'Late Submissions',
  extracurricular: 'Extracurricular Activities',
};

function SummaryCard({ title, value, sub, valueClass }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">{title}</p>
      <p className={`text-3xl font-bold ${valueClass || 'text-gray-900'}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

export default function FacultyDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    axios
      .get('/api/faculty/summary')
      .then((res) => setData(res.data))
      .catch((err) => setError(err.response?.data?.error || err.message))
      .finally(() => setLoading(false));
  }, []);

  const handleToggle = (id) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <svg className="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
        <span className="ml-3 text-gray-600">Loading faculty summary...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
        {error}
      </div>
    );
  }

  const avgColor =
    data.averageScore >= 7 ? 'text-green-600' : data.averageScore >= 5 ? 'text-yellow-600' : 'text-red-600';

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Faculty Dashboard</h2>
        <p className="text-gray-500 mt-1">Class-wide performance summary and student analytics.</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <SummaryCard
          title="Total Students"
          value={data.totalStudents}
          sub="enrolled"
        />
        <SummaryCard
          title="Class Average Score"
          value={data.averageScore?.toFixed(2)}
          sub="out of 10"
          valueClass={avgColor}
        />
        <SummaryCard
          title="At-Risk Students"
          value={data.atRiskCount}
          sub="predicted score < 5"
          valueClass={data.atRiskCount > 0 ? 'text-red-600' : 'text-green-600'}
        />
        <SummaryCard
          title="Most Common Weak Area"
          value={FEATURE_LABELS[data.commonWeakFeatures?.[0]] || data.commonWeakFeatures?.[0] || 'N/A'}
          sub={
            data.commonWeakFeatures?.slice(1, 3).map((f) => FEATURE_LABELS[f] || f).join(', ') || ''
          }
          valueClass="text-blue-600 text-xl"
        />
      </div>

      {/* Student Table */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Student List</h3>
        <StudentTable
          students={data.students}
          expandedId={expandedId}
          onToggle={handleToggle}
        />
      </div>

      {/* Score Chart */}
      <ScoreChart students={data.students} />
    </div>
  );
}
