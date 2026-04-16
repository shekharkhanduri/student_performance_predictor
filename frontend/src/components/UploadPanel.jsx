import { useState, useRef } from "react";
import { uploadStudentsCsv } from "../lib/api";

function UploadPanel({ onUploaded }) {
  const [file, setFile]             = useState(null);
  const [status, setStatus]         = useState({ type: "idle", message: "", step: "" });
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef(null);

  function handleFile(chosen) {
    if (chosen?.name?.endsWith(".csv")) {
      setFile(chosen);
      setStatus({ type: "idle", message: "", step: "" });
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) {
      setStatus({ type: "error", message: "Choose a CSV file first.", step: "" });
      return;
    }
    try {
      setSubmitting(true);
      setStatus({ type: "idle", message: "", step: "Parsing CSV…" });
      await new Promise((r) => setTimeout(r, 800));
      setStatus({ type: "idle", message: "", step: "Detecting dataset type…" });
      await new Promise((r) => setTimeout(r, 400));
      setStatus({ type: "idle", message: "", step: "Running ML inference…" });
      const result = await uploadStudentsCsv(file);
      setStatus({ type: "idle", message: "", step: "Storing predictions…" });
      await new Promise((r) => setTimeout(r, 300));
      setStatus({
        type: "ok",
        message: `Processed ${result.rows_processed} rows · Stored ${result.rows_stored} students · Batch ${result.batch_id}`,
        step: "",
      });
      setFile(null);
      onUploaded();
    } catch (error) {
      setStatus({ type: "error", message: error.message || "Upload failed.", step: "" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="card">
      <h3>Bulk CSV Upload</h3>
      <p className="muted" style={{ marginBottom: "0.75rem" }}>
        Upload a class CSV to process all students through the ML model.
      </p>

      <details className="csv-details">
        <summary>CSV format requirements</summary>
        <div className="csv-details-body">
          <p>Must include <code>student_id</code> (integer, required) plus one of:</p>
          <ul>
            <li><strong>Dataset 1:</strong> Hours_Studied, Previous_Scores, Extracurricular_Activities, Sleep_Hours, Sample_Question_Papers_Practiced</li>
            <li><strong>Dataset 2:</strong> Hours_Studied, Attendance, Gender, Parental_Involvement… (20 fields)</li>
          </ul>
          <p style={{ marginTop: "0.4rem" }}>Re-uploading a student ID updates the existing record.</p>
        </div>
      </details>

      <form onSubmit={handleSubmit} className="upload-form">
        <div
          className="upload-dropzone"
          onClick={() => !submitting && inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer?.files?.[0]); }}
        >
          <span className="drop-icon">📁</span>
          <p>Click to browse or drag & drop a CSV file</p>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            disabled={submitting}
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        {file && <div className="file-chosen">{file.name}</div>}

        <button type="submit" disabled={submitting || !file} className="btn-full">
          {submitting ? "Processing…" : "Upload & Predict"}
        </button>
      </form>

      {status.step    && <p className="feedback processing" style={{ marginTop: "0.5rem" }}>{status.step}</p>}
      {status.message && <p className={`feedback ${status.type}`} style={{ marginTop: "0.5rem" }}>{status.message}</p>}
    </section>
  );
}

export default UploadPanel;
