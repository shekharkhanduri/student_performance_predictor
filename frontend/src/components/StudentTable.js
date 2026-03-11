import React from 'react';

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

function RiskBadge({ level }) {
  const styles = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-red-100 text-red-800',
  };
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-semibold capitalize ${styles[level] || ''}`}>
      {level}
    </span>
  );
}

export default function StudentTable({ students, expandedId, onToggle }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-gray-200">
      <table className="min-w-full bg-white text-sm">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
              Student Name
            </th>
            <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
              Predicted Score
            </th>
            <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
              Risk Level
            </th>
            <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
              Top Weak Feature
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {students.map((s) => (
            <React.Fragment key={s.id}>
              <tr
                className="hover:bg-blue-50 cursor-pointer transition-colors"
                onClick={() => onToggle(s.id)}
              >
                <td className="px-6 py-4 font-medium text-gray-900">
                  <span className="flex items-center space-x-2">
                    <span>{s.name}</span>
                    <span className="text-gray-400 text-xs">{expandedId === s.id ? '▲' : '▼'}</span>
                  </span>
                </td>
                <td className="px-6 py-4 font-semibold text-blue-700">
                  {s.predicted_score?.toFixed(2)}
                </td>
                <td className="px-6 py-4">
                  <RiskBadge level={s.risk_level} />
                </td>
                <td className="px-6 py-4 text-gray-600">
                  {FEATURE_LABELS[s.top_weak_feature] || s.top_weak_feature}
                </td>
              </tr>
              {expandedId === s.id && (
                <tr>
                  <td colSpan={4} className="bg-gray-50 px-6 py-4">
                    <div className="text-sm font-semibold text-gray-700 mb-3">
                      Recommendations for {s.name}
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      {s.recommendations?.map((rec) => (
                        <div
                          key={rec.feature}
                          className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm"
                        >
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-medium text-gray-800 text-xs">
                              {FEATURE_LABELS[rec.feature] || rec.feature}
                            </span>
                            <span className="text-green-600 text-xs font-bold">
                              +{rec.score_improvement?.toFixed(2)} pts
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 space-y-1">
                            <div className="flex justify-between">
                              <span>Current</span>
                              <span className="text-red-500">{rec.current_value}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Benchmark</span>
                              <span className="text-blue-600">{rec.benchmark_value}</span>
                            </div>
                            <div className="flex justify-between border-t pt-1">
                              <span>Target</span>
                              <span className="text-green-600 font-semibold">{rec.suggested_value}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
