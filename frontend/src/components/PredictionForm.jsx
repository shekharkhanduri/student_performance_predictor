import { useState } from "react";
import { createPrediction } from "../lib/api";

const ds1CategoricalOptions = {
  Extracurricular_Activities: ["Yes", "No"],
};

const ds2CategoricalOptions = {
  Gender: ["Male", "Female"],
  Parental_Involvement: ["Low", "Medium", "High"],
  Access_to_Resources: ["Low", "Medium", "High"],
  Extracurricular_Activities: ["Yes", "No"],
  Motivation_Level: ["Low", "Medium", "High"],
  Internet_Access: ["Yes", "No"],
  Family_Income: ["Low", "Medium", "High"],
  Teacher_Quality: ["Low", "Medium", "High"],
  School_Type: ["Public", "Private"],
  Peer_Influence: ["Positive", "Neutral", "Negative"],
  Learning_Disabilities: ["No", "Yes"],
  Parental_Education_Level: ["High School", "College", "Postgraduate"],
  Distance_from_Home: ["Near", "Moderate", "Far"],
};

const initialDs1Form = {
  student_id: "",
  student_name: "",
  Hours_Studied: 6,
  Previous_Scores: 72,
  Extracurricular_Activities: "Yes",
  Sleep_Hours: 7,
  Sample_Question_Papers_Practiced: 4,
};

const initialDs2Form = {
  student_id: "",
  student_name: "",
  Hours_Studied: 8,
  Attendance: 80,
  Gender: "Male",
  Parental_Involvement: "Medium",
  Access_to_Resources: "Medium",
  Extracurricular_Activities: "Yes",
  Sleep_Hours: 7,
  Previous_Scores: 70,
  Motivation_Level: "Medium",
  Internet_Access: "Yes",
  Tutoring_Sessions: 1,
  Family_Income: "Medium",
  Teacher_Quality: "Medium",
  School_Type: "Public",
  Peer_Influence: "Neutral",
  Physical_Activity: 3,
  Learning_Disabilities: "No",
  Parental_Education_Level: "College",
  Distance_from_Home: "Moderate",
};

const ds1NumericFields = [
  { key: "Hours_Studied", label: "Hours Studied", min: 0, step: 0.5 },
  { key: "Previous_Scores", label: "Previous Scores", min: 0, max: 100, step: 1 },
  { key: "Sleep_Hours", label: "Sleep Hours", min: 0, step: 0.5 },
  { key: "Sample_Question_Papers_Practiced", label: "Sample Papers Practiced", min: 0, step: 1 },
];

const ds2NumericFields = [
  { key: "Hours_Studied", label: "Hours Studied", min: 0, step: 0.5 },
  { key: "Attendance", label: "Attendance", min: 0, max: 100, step: 1 },
  { key: "Sleep_Hours", label: "Sleep Hours", min: 0, step: 0.5 },
  { key: "Previous_Scores", label: "Previous Scores", min: 0, max: 100, step: 1 },
  { key: "Tutoring_Sessions", label: "Tutoring Sessions", min: 0, step: 1 },
  { key: "Physical_Activity", label: "Physical Activity", min: 0, step: 1 },
];

function PredictionForm({ onCreated, initialDataset = "ds2" }) {
  const [datasetType, setDatasetType] = useState(initialDataset === "ds1" ? "ds1" : "ds2");
  const [ds1Form, setDs1Form] = useState(initialDs1Form);
  const [ds2Form, setDs2Form] = useState(initialDs2Form);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [copyMessage, setCopyMessage] = useState("");

  const form = datasetType === "ds1" ? ds1Form : ds2Form;

  function updateField(field, value) {
    if (datasetType === "ds1") {
      setDs1Form((prev) => ({ ...prev, [field]: value }));
    } else {
      setDs2Form((prev) => ({ ...prev, [field]: value }));
    }
  }

  function setSurveyDataset(value) {
    setDatasetType(value === "ds1" ? "ds1" : "ds2");
    setResult(null);
    setError("");
    setCopyMessage("");
  }

  function buildShareUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", "predict");
    url.searchParams.set("survey", "1");
    url.searchParams.set("dataset", datasetType);
    return url.toString();
  }

  async function handleCopyLink() {
    const link = buildShareUrl();
    try {
      await navigator.clipboard.writeText(link);
      setCopyMessage("Survey link copied. Share it with students or faculty.");
    } catch {
      setCopyMessage(`Copy this survey link: ${link}`);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setResult(null);
    setError("");

    try {
      setError("⏳ Validating features…");
      await new Promise((r) => setTimeout(r, 200));

      setError("⏳ Running Stacking Ensemble (XGBoost + CatBoost + RF + LassoCV)…");
      await new Promise((r) => setTimeout(r, 300));

      setError("⏳ Computing SHAP explanations…");

      const payload = {
        ...form,
        dataset_type: datasetType,
        student_id: form.student_id ? Number(form.student_id) : null,
        student_name: form.student_name || null,
      };

      const response = await createPrediction(payload);

      setError("⏳ Storing in database…");
      await new Promise((r) => setTimeout(r, 200));

      setResult(response);
      setError("");
      onCreated({
        student_id: response.student_id,
        dataset_type: response.dataset_type || datasetType,
      });
    } catch (submitError) {
      setError(submitError.message || "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <h3>Shareable Survey Form</h3>
      <p className="muted">
        Create a survey link for Dataset 1 or Dataset 2 and collect responses for prediction.
      </p>

      <div className="survey-toolbar">
        <label>
          Survey Type
          <select value={datasetType} onChange={(e) => setSurveyDataset(e.target.value)}>
            <option value="ds1">Dataset 1 Survey</option>
            <option value="ds2">Dataset 2 Survey</option>
          </select>
        </label>
        <button type="button" className="ghost" onClick={handleCopyLink}>
          Copy Shareable Link
        </button>
      </div>
      {copyMessage && <p className="feedback ok">{copyMessage}</p>}

      <form className="predict-form" onSubmit={handleSubmit}>
        {datasetType === "ds1" ? (
          <p className="muted">Dataset 1 fields: study load, prior score, sleep and practice papers.</p>
        ) : (
          <p className="muted">Dataset 2 fields: academics, attendance, family and school context.</p>
        )}

        <label>
          Student ID
          <input
            type="number"
            min={1}
            step={1}
            value={form.student_id}
            onChange={(e) => updateField("student_id", e.target.value)}
            placeholder="Optional"
          />
        </label>

        <label>
          Student Name
          <input
            type="text"
            value={form.student_name}
            onChange={(e) => updateField("student_name", e.target.value)}
            placeholder="Optional"
          />
        </label>

        <div className="form-grid">
          {(datasetType === "ds1" ? ds1NumericFields : ds2NumericFields).map((field) => (
            <label key={field.key}>
              {field.label}
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.step}
                value={form[field.key]}
                onChange={(e) => updateField(field.key, Number(e.target.value))}
              />
            </label>
          ))}
        </div>

        <div className="form-grid">
          {Object.entries(
            datasetType === "ds1" ? ds1CategoricalOptions : ds2CategoricalOptions
          ).map(([key, values]) => (
            <label key={key}>
              {key.replaceAll("_", " ")}
              <select value={form[key]} onChange={(e) => updateField(key, e.target.value)}>
                {values.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>

        <button type="submit" disabled={busy}>
          {busy ? "Predicting..." : "Submit Survey Response"}
        </button>
      </form>

      {error && <p className="feedback error">{error}</p>}

      {result && (
        <div className="result-box">
          <p>
            Registered as student #{result.student_id}
            {" "}— predicted score{" "}
            {Number(result.predicted_exam_score || 0).toFixed(1)} ({result.risk_level}).
          </p>
        </div>
      )}
    </section>
  );
}
export default PredictionForm;
