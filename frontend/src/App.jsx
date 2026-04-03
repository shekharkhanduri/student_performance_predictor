import { useState, useEffect } from "react";
import Header from "./components/Header";
import StatsPanel from "./components/StatsPanel";
import UploadPanel from "./components/UploadPanel";
import StudentsTable from "./components/StudentsTable";
import StudentDetails from "./components/StudentDetails";
import PredictionForm from "./components/PredictionForm";
import DashboardTab from "./components/DashboardTab";
import { getHealth, getStudents, getStudent } from "./lib/api";

function App() {
  const searchParams = new URLSearchParams(window.location.search);
  const initialTab = searchParams.get("tab") === "predict" || searchParams.get("survey") === "1"
    ? "predict"
    : "dashboard";
  const initialDataset = searchParams.get("dataset") === "ds1" ? "ds1" : "ds2";

  const [health, setHealth] = useState(null);
  const [students, setStudents] = useState([]);
  const [selectedStudentRef, setSelectedStudentRef] = useState(null);
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(false);
  const [loadingStudents, setLoadingStudents] = useState(false);
  const [loadingStudent, setLoadingStudent] = useState(false);
  const [studentError, setStudentError] = useState("");
  const [activeTab, setActiveTab] = useState(initialTab);

  async function refreshHealth() {
    setLoadingHealth(true);
    try {
      const data = await getHealth();
      setHealth(data);
    } catch (error) {
      console.error("Health check failed:", error);
      setHealth({ status: "error", model_loaded: false });
    } finally {
      setLoadingHealth(false);
    }
  }

  async function refreshStudents() {
    setLoadingStudents(true);
    try {
      const data = await getStudents({ limit: 200 });
      setStudents(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error("Fetch students failed:", error);
      setStudents([]);
    } finally {
      setLoadingStudents(false);
    }
  }

  async function selectStudent(ref) {
    if (!ref?.student_id) {
      return;
    }
    setSelectedStudentRef(ref);
    setLoadingStudent(true);
    setStudentError("");
    setSelectedStudent(null);
    try {
      const data = await getStudent(ref.student_id, ref.dataset_type);
      setSelectedStudent(data);
    } catch (error) {
      setStudentError(error.message || "Failed to load student details.");
    } finally {
      setLoadingStudent(false);
    }
  }

  useEffect(() => {
    refreshHealth();
    refreshStudents();
  }, []);

  return (
    <div className="app">
      <Header
        health={health}
        onRefresh={refreshHealth}
        loading={loadingHealth}
      />

      <div className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === "dashboard" ? "active" : ""}`}
          onClick={() => setActiveTab("dashboard")}
        >
          📊 Dashboard
        </button>
        <button
          className={`tab-btn ${activeTab === "predict" ? "active" : ""}`}
          onClick={() => setActiveTab("predict")}
        >
          📝 Survey Form
        </button>
        <button
          className={`tab-btn ${activeTab === "results" ? "active" : ""}`}
          onClick={() => setActiveTab("results")}
        >
          📋 All Results ({students.length})
        </button>
      </div>

      <main className="main-content">
        {activeTab === "dashboard" && (
          <DashboardTab
            students={students}
            selectedStudentId={selectedStudentRef}
            selectedStudent={selectedStudent}
            loadingStudent={loadingStudent}
            studentError={studentError}
            onSelectStudent={selectStudent}
            onUpload={refreshStudents}
          />
        )}

        {activeTab === "predict" && (
          <section className="card">
            <h2>Shareable Student Survey</h2>
            <p className="muted">Use dataset-specific forms and share links with faculty or students.</p>
            <PredictionForm
              initialDataset={initialDataset}
              onCreated={(ref) => {
                refreshStudents();
                selectStudent(ref);
              }}
            />
          </section>
        )}

        {activeTab === "results" && (
          <section>
            <StatsPanel students={students} />
            <div className="card table-card">
              <div className="table-head">
                <h2>All Stored Students</h2>
                <p className="muted">Click a row to view diagnostic details.</p>
              </div>
              <StudentsTable
                students={students}
                selectedId={selectedStudentRef}
                onSelect={selectStudent}
              />
            </div>
            {selectedStudent && (
              <div className="card">
                <h3>Student Diagnostic</h3>
                <StudentDetails
                  student={selectedStudent}
                  loading={loadingStudent}
                  error={studentError}
                />
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Faculty Student Diagnostic System • Ensemble ML + SHAP Explanations</p>
      </footer>
    </div>
  );
}

export default App;
