function StudentDetails({ student, loading, error }) {
  return (
    <section className="card">
      <h3>Student Diagnostic</h3>
      {loading ? <p className="muted">Loading student details...</p> : null}
      {error ? <p className="feedback error">{error}</p> : null}
      {!loading && !error && !student ? <p className="muted">Select a student to view diagnostics.</p> : null}
      {!loading && student ? (
        <div className="details-body">
          <div className="detail-grid">
            <div>
              <p className="label">Name</p>
              <p>{student.student_name || "-"}</p>
            </div>
            <div>
              <p className="label">Predicted Score</p>
              <p>{Number(student.predicted_exam_score || 0).toFixed(1)}</p>
            </div>
            <div>
              <p className="label">Risk Level</p>
              <p>{student.risk_level}</p>
            </div>
            <div>
              <p className="label">Student ID</p>
              <p>{student.student_id}</p>
            </div>
            <div>
              <p className="label">Dataset</p>
              <p>{student.dataset_type?.toUpperCase() || "-"}</p>
            </div>
          </div>
          <h4>Top Negative Factors</h4>
          <ul className="factor-list">
            {(student.top_negative_factors || []).map((factor, idx) => (
              <li key={`${factor.feature}-${idx}`}>
                <strong>{factor.feature}</strong>
                <span>{Number(factor.shap_value || 0).toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

export default StudentDetails;
