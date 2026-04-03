import { API_BASE } from "../lib/api";

function Header({ health, onRefresh, loading }) {
  const statusText = health?.status === "ok" ? "API Online" : "API Unknown";
  const modelReady = health?.model_loaded ? "Models Ready" : "Models Missing";

  return (
    <header className="hero">
      <div className="hero-content">
        <p className="eyebrow">Faculty Student Diagnostic System</p>
        <h1>Prediction Console</h1>
        <p className="subtext">
          A focused dashboard for uploading student data, running predictions, and reviewing risk drivers.
        </p>
      </div>
      <div className="hero-status">
        <div className="pill">{statusText}</div>
        <div className={`pill ${health?.model_loaded ? "pill-good" : "pill-warn"}`}>{modelReady}</div>
        <button className="ghost" onClick={onRefresh} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>
      <p className="api-target">API: {API_BASE}</p>
    </header>
  );
}

export default Header;
