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

export default function RecommendationCard({ rec }) {
  const label = FEATURE_LABELS[rec.feature] || rec.feature;
  const improvement = rec.score_improvement ?? 0;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-gray-800 text-base">{label}</h4>
        <span className="text-green-600 font-bold text-sm bg-green-50 px-2 py-1 rounded-full">
          +{improvement.toFixed(2)} pts
        </span>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-500">Current</span>
          <span className="font-medium text-red-500">{rec.current_value}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Benchmark</span>
          <span className="font-medium text-blue-600">{rec.benchmark_value}</span>
        </div>
        <div className="flex justify-between border-t pt-2">
          <span className="text-gray-500">Target</span>
          <span className="font-semibold text-green-600">{rec.suggested_value}</span>
        </div>
      </div>
    </div>
  );
}
