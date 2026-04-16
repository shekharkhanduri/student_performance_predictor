function StatsPanel({ students }) {
  const total      = students.length;
  const avgScore   = total > 0
    ? students.reduce((sum, s) => sum + (s.predicted_exam_score || 0), 0) / total
    : 0;
  const atRisk     = students.filter((s) => s.risk_level === "At-Risk").length;
  const borderline = students.filter((s) => s.risk_level === "Borderline").length;

  return (
    <section className="stats-grid">
      <article className="card stat-card">
        <p className="stat-label">Total Students</p>
        <p className="stat-value">{total}</p>
        <p className="stat-sub">in database</p>
      </article>

      <article className="card stat-card success">
        <p className="stat-label">Avg. Score</p>
        <p className="stat-value">{avgScore.toFixed(1)}</p>
        <p className="stat-sub">predicted</p>
      </article>

      <article className="card stat-card warning">
        <p className="stat-label">At-Risk</p>
        <p className="stat-value">{atRisk}</p>
        <p className="stat-sub">need attention</p>
      </article>

      <article className="card stat-card caution">
        <p className="stat-label">Borderline</p>
        <p className="stat-value">{borderline}</p>
        <p className="stat-sub">monitor closely</p>
      </article>
    </section>
  );
}

export default StatsPanel;
