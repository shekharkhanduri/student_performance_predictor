import React from 'react';

function getBadge(score) {
  if (score >= 7) return { label: 'Excellent', color: 'bg-green-100 text-green-800' };
  if (score >= 5) return { label: 'Average', color: 'bg-yellow-100 text-yellow-800' };
  return { label: 'At Risk', color: 'bg-red-100 text-red-800' };
}

function getBarColor(score) {
  if (score >= 7) return 'bg-green-500';
  if (score >= 5) return 'bg-yellow-400';
  return 'bg-red-500';
}

export default function PredictionResult({ score }) {
  const { label, color } = getBadge(score);
  const pct = Math.min(100, (score / 10) * 100);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-sm text-gray-500 mb-1">Predicted Score</p>
          <div className="flex items-baseline space-x-2">
            <span className="text-5xl font-bold text-gray-900">{score.toFixed(2)}</span>
            <span className="text-xl text-gray-400">/ 10</span>
          </div>
        </div>
        <span className={`px-4 py-2 rounded-full text-sm font-semibold ${color}`}>
          {label}
        </span>
      </div>
      <div>
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>0</span>
          <span>Performance Meter</span>
          <span>10</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
          <div
            className={`h-4 rounded-full transition-all duration-700 ${getBarColor(score)}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
