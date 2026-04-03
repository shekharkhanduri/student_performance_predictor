import StatsPanel from "./StatsPanel";
import UploadPanel from "./UploadPanel";
import StudentsTable from "./StudentsTable";
import StudentDetails from "./StudentDetails";

function DashboardTab({
  students,
  selectedStudentId,
  selectedStudent,
  loadingStudent,
  studentError,
  onSelectStudent,
  onUpload,
}) {
  return (
    <div className="dashboard-layout">
      <div className="col col-left">
        <StatsPanel students={students} />
        <UploadPanel onUploaded={onUpload} />
      </div>

      <div className="col col-right">
        <div className="card table-card">
          <div className="table-head">
            <h3>Recent Students</h3>
            <p className="muted">Click to view diagnostics.</p>
          </div>
          <StudentsTable
            students={students}
            selectedId={selectedStudentId}
            onSelect={onSelectStudent}
          />
        </div>
        <StudentDetails
          student={selectedStudent}
          loading={loadingStudent}
          error={studentError}
        />
      </div>
    </div>
  );
}

export default DashboardTab;
