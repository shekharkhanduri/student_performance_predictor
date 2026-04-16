function riskClass(risk) {
  if (risk === "At-Risk") return "tag danger";
  if (risk === "Borderline") return "tag caution";
  return "tag good";
}

function StudentsTable({ students, selectedRef, onSelect }) {
  return (
    <section className="card table-card">
      <div className="table-head">
        <h3>Recent Students</h3>
        <p className="muted">Select a row to inspect feature-level diagnostics.</p>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Score</th>
              <th>Risk</th>
              <th>SHAP</th>
              <th>Top Factor</th>
            </tr>
          </thead>
          <tbody>
            {students.map((student) => {
              const topFactor = student.top_negative_factors?.[0]?.feature || "—";
              const selectedKey = `${selectedRef?.dataset_type || ""}:${selectedRef?.student_id ?? ""}`;
              const rowKey = `${student.dataset_type || ""}:${student.student_id}`;
              const isSelected = selectedKey === rowKey;

              return (
                <tr
                  key={rowKey}
                  onClick={() =>
                    onSelect({ student_id: student.student_id, dataset_type: student.dataset_type })
                  }
                  className={isSelected ? "selected" : ""}
                >
                  <td>{student.student_id}</td>
                  <td>{student.student_name || "—"}</td>
                  <td>{Number(student.predicted_exam_score || 0).toFixed(1)}</td>
                  <td>
                    <span className={riskClass(student.risk_level)}>{student.risk_level}</span>
                  </td>
                  <td>
                    <span
                      className={`tag ${
                        student.shap_status === "done"
                          ? "good"
                          : student.shap_status === "failed"
                          ? "danger"
                          : "caution"
                      }`}
                    >
                      {student.shap_status}
                    </span>
                  </td>
                  <td>{topFactor}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default StudentsTable;
