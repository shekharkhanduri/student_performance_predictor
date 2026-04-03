import { useState } from "react";
import { uploadStudentsCsv } from "../lib/api";

function UploadPanel({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState({ type: "idle", message: "", step: "" });
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setStatus({ type: "error", message: "Choose a CSV file first.", step: "" });
      return;
    }

    try {
      setSubmitting(true);
      setStatus({ type: "idle", message: "", step: "Parsing CSV…" });
      
      // Simulate parsing delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      setStatus({ type: "idle", message: "", step: "Detecting dataset type…" });
      
      await new Promise(resolve => setTimeout(resolve, 300));
      setStatus({ type: "idle", message: "", step: "Preprocessing features…" });
      
      await new Promise(resolve => setTimeout(resolve, 300));
      setStatus({ type: "idle", message: "", step: "Running ML inference (Stacking Ensemble)…" });
      
      const result = await uploadStudentsCsv(file);
      
      setStatus({ type: "idle", message: "", step: "Storing predictions in database…" });
      await new Promise(resolve => setTimeout(resolve, 300));
      
      setStatus({
        type: "ok",
        message: `Processed ${result.rows_processed} rows | Stored ${result.rows_stored} students (Batch: ${result.batch_id})`,
        step: "",
      });
      
      setFile(null);
      onUploaded();
    } catch (error) {
      setStatus({ 
        type: "error", 
        message: error.message || "Upload failed.", 
        step: "" 
      });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="card">
      <h3>📤 Bulk CSV Upload</h3>
      <p className="muted">Upload a class CSV to process all students through the ML model.</p>
      <details className="muted" style={{ marginBottom: "1rem", cursor: "pointer" }}>
        <summary style={{ cursor: "pointer", fontWeight: "bold" }}>CSV Format Requirements</summary>
        <div style={{ marginTop: "0.5rem", paddingLeft: "1rem" }}>
          <p>Your CSV must include:</p>
          <ul style={{ marginLeft: "1.5rem" }}>
            <li><code>student_id</code> (required) - unique integer identifier for each student</li>
            <li>Dataset 1 columns: Hours_Studied, Previous_Scores, Extracurricular_Activities, Sleep_Hours, Sample_Question_Papers_Practiced</li>
            <li>OR Dataset 2 columns: Hours_Studied, Attendance, Gender, Parental_Involvement, and 16 more fields…</li>
          </ul>
          <p style={{ marginTop: "0.5rem" }}>When the same student_id is uploaded again, the previous record is updated with the latest predictions.</p>
        </div>
      </details>
      <form onSubmit={handleSubmit} className="upload-form">
        <input 
          type="file" 
          accept=".csv" 
          onChange={(event) => setFile(event.target.files?.[0] || null)}
          disabled={submitting}
        />
        <button type="submit" disabled={submitting}>
          {submitting ? "Processing..." : "Upload & Predict"}
        </button>
      </form>
      {status.step ? <p className="feedback processing">⏳ {status.step}</p> : null}
      {status.message ? <p className={`feedback ${status.type}`}>{status.message}</p> : null}
    </section>
  );
}

export default UploadPanel;
