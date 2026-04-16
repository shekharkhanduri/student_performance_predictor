import { API_BASE } from "../lib/api";

function Header({ health, onRefresh, loading }) {
  const apiOnline  = health?.status === "ok";
  const modelReady = health?.model_loaded;

  return (
    <header className="hero">
      <div className="hero-inner">
        <div>
          <h1>Student Performance Predictor</h1>
          <p className="hero-sub">Faculty diagnostic system — ensemble ML + SHAP explanations</p>
        </div>

        <div className="hero-right">
          <span className={`pill ${apiOnline ? "pill-good" : "pill-warn"}`}>
            {apiOnline ? "API Online" : "API Offline"}
          </span>
          <span className={`pill ${modelReady ? "pill-good" : "pill-warn"}`}>
            {modelReady ? "Models Ready" : "Models Missing"}
          </span>
          <button className="ghost-sm" onClick={onRefresh} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
          <span className="api-target">{API_BASE}</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
