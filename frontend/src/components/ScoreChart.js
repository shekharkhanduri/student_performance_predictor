import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

function getBarColor(score) {
  if (score >= 7) return '#22c55e';
  if (score >= 5) return '#eab308';
  return '#ef4444';
}

export default function ScoreChart({ students }) {
  const data = students.map((s) => ({
    name: s.name.split(' ')[0],
    score: parseFloat(s.predicted_score?.toFixed(2) ?? 0),
  }));

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-base font-semibold text-gray-800 mb-4">
        Predicted Score Distribution
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis domain={[0, 10]} tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(val) => [val.toFixed(2), 'Score']}
            contentStyle={{ borderRadius: 8, fontSize: 12 }}
          />
          <Bar dataKey="score" radius={[4, 4, 0, 0]}>
            {data.map((entry, idx) => (
              <Cell key={idx} fill={getBarColor(entry.score)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
