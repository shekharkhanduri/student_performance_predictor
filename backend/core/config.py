"""Application configuration values."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
MODEL_DIR = BACKEND_ROOT / "mlmodel"


def _load_env_file(path: Path) -> None:
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

DATABASE_URL = os.getenv("DATABASE_URL")

MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "model_ds2.joblib"))

DS2_MODEL_PATH = os.getenv("DS2_MODEL_PATH", MODEL_PATH)

DS1_MODEL_PATH = os.getenv("DS1_MODEL_PATH", str(MODEL_DIR / "model_ds1.joblib"))

DS1_SCALER_PATH = os.getenv("DS1_SCALER_PATH", str(MODEL_DIR / "scaler_ds1.joblib"))

DS2_SCALER_PATH = os.getenv("DS2_SCALER_PATH", str(MODEL_DIR / "scaler_ds2.joblib"))
