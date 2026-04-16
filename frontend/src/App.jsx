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
    } catch {
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
    } catch {
      setStudents([]);
    } finally {
      setLoadingStudents(false);
    }
  }

  async function selectStudent(ref) {
    if (!ref?.student_id) return;
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

  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: "📊" },
    { id: "predict", label: "Survey Form", icon: "📝" },
    { id: "results", label: "All Results", icon: "📋", count: students.length },
  ];

  return (
    <div className="app">
      <Header
        health={health}
        onRefresh={refreshHealth}
        loading={loadingHealth}
      />

      <nav className="tabs-nav" role="tablist">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
            {tab.count !== undefined && (
              <span className="tab-count">{tab.count}</span>
            )}
          </button>
        ))}
      </nav>

      <main className="main-content" role="main">
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
          <div style={{ maxWidth: "840px" }}>
            <PredictionForm
              initialDataset={initialDataset}
              onCreated={(ref) => {
                refreshStudents();
                selectStudent(ref);
              }}
            />
          </div>
        )}

        {activeTab === "results" && (
          <section>
            <StatsPanel students={students} />
            <StudentsTable
              students={students}
              selectedId={selectedStudentRef}
              onSelect={selectStudent}
            />
            {selectedStudent && (
              <StudentDetails
                student={selectedStudent}
                loading={loadingStudent}
                error={studentError}
              />
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>© 2026 EduSight · Faculty Student Diagnostic System</p>
        <div className="footer-tags">
          <span className="footer-tag">XGBoost</span>
          <span className="footer-tag">CatBoost</span>
          <span className="footer-tag">RandomForest</span>
          <span className="footer-tag">SHAP</span>
          <span className="footer-tag">FastAPI</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
