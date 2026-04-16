"""Application configuration — paths, DB URL, and tunable constants."""

from __future__ import annotations

import os
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
MODEL_DIR = BACKEND_ROOT / "mlmodel"


# ── Env-file loader ─────────────────────────────────────────────────────────
def _load_env_file(path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ (if not already set)."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(REPO_ROOT / ".env")
_load_env_file(BACKEND_ROOT / ".env")


# ── Settings ────────────────────────────────────────────────────────────────
DATABASE_URL: str | None = os.getenv("DATABASE_URL")

# API versioning prefix — all routes are mounted under this path.
API_V1_PREFIX: str = "/api/v1"

# Maximum number of rows accepted in a single CSV upload.
# Prevents runaway SHAP compute time and OOM issues.
MAX_UPLOAD_ROWS: int = int(os.getenv("MAX_UPLOAD_ROWS", "500"))

# Model / scaler paths (overridable via env vars)
DS1_MODEL_PATH: str = os.getenv("DS1_MODEL_PATH", str(MODEL_DIR / "model_ds1.joblib"))
DS2_MODEL_PATH: str = os.getenv("DS2_MODEL_PATH", str(MODEL_DIR / "model_ds2.joblib"))
DS1_SCALER_PATH: str = os.getenv("DS1_SCALER_PATH", str(MODEL_DIR / "scaler_ds1.joblib"))
DS2_SCALER_PATH: str = os.getenv("DS2_SCALER_PATH", str(MODEL_DIR / "scaler_ds2.joblib"))
