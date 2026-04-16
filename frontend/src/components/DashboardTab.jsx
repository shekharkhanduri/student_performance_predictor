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
        <StudentsTable
          students={students}
          selectedId={selectedStudentId}
          onSelect={onSelectStudent}
        />
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
