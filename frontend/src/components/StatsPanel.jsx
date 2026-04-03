function StatsPanel({ students }) {
  const total = students.length;
  const avgScore = total > 0 ? students.reduce((sum, s) => sum + (s.predicted_exam_score || 0), 0) / total : 0;
  const atRisk = students.filter((s) => s.risk_level === "At-Risk").length;
  const borderline = students.filter((s) => s.risk_level === "Borderline").length;

  return (
    <section className="stats-grid">
      <article className="card stat-card">
        <p>Total Students</p>
        <h2>{total}</h2>
      </article>
      <article className="card stat-card">
        <p>Average Predicted Score</p>
        <h2>{avgScore.toFixed(1)}</h2>
      </article>
      <article className="card stat-card warning">
        <p>At-Risk Students</p>
        <h2>{atRisk}</h2>
      </article>
      <article className="card stat-card caution">
        <p>Borderline Students</p>
        <h2>{borderline}</h2>
      </article>
    </section>
  );
}

export default StatsPanel;
