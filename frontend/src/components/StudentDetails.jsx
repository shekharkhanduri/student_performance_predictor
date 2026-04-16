function riskTagClass(risk) {
  if (risk === "At-Risk") return "tag danger";
  if (risk === "Borderline") return "tag caution";
  return "tag good";
}

function scoreColor(score) {
  const n = Number(score) || 0;
  if (n >= 75) return "var(--green)";
  if (n >= 55) return "var(--orange)";
  return "var(--red)";
}

function StudentDetails({ student, loading, error }) {
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h3>Student Diagnostic</h3>
          <p className="muted">SHAP-powered feature analysis</p>
        </div>
        <div className="card-icon blue">🔬</div>
      </div>

      {loading && (
        <div className="empty-state">
          <span className="empty-icon" style={{ animation: "pulse-dot 1.2s infinite" }}>⏳</span>
          <p>Loading diagnostic data…</p>
        </div>
      )}

      {error && !loading && (
        <div className="feedback error">{error}</div>
      )}

      {!loading && !error && !student && (
        <div className="empty-state">
          <span className="empty-icon">📋</span>
          <p>Select a student from the table to view their diagnostic report.</p>
        </div>
      )}

      {!loading && student && (
        <div className="details-body">
          <div className="detail-grid">
            <div className="detail-item">
              <p className="label">Name</p>
              <p>{student.student_name || "—"}</p>
            </div>
            <div className="detail-item">
              <p className="label">Predicted Score</p>
              <p style={{ color: scoreColor(student.predicted_exam_score) }}>
                {Number(student.predicted_exam_score || 0).toFixed(1)}
              </p>
            </div>
            <div className="detail-item">
              <p className="label">Risk Level</p>
              <p>
                <span className={riskTagClass(student.risk_level)}>
                  {student.risk_level}
                </span>
              </p>
            </div>
            <div className="detail-item">
              <p className="label">Student ID</p>
              <p style={{ fontFamily: "monospace", fontSize: "0.95rem" }}>
                #{student.student_id}
              </p>
            </div>
            <div className="detail-item">
              <p className="label">Dataset</p>
              <p>{student.dataset_type?.toUpperCase() || "—"}</p>
            </div>
          </div>

          {(student.top_negative_factors || []).length > 0 && (
            <>
              <h4>⚠️ Top Negative Factors</h4>
              <ul className="factor-list">
                {(student.top_negative_factors || []).map((factor, idx) => (
                  <li key={`${factor.feature}-${idx}`} className="factor-item">
                    <div className="factor-left">
                      <span className="factor-rank">{idx + 1}</span>
                      <span className="factor-name">
                        {factor.feature.replaceAll("_", " ")}
                      </span>
                    </div>
                    <span className="factor-value">
                      {Number(factor.shap_value || 0).toFixed(3)}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </section>
  );
}

export default StudentDetails;
