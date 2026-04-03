"""Health check endpoints."""

from os.path import exists

from fastapi import APIRouter

from backend.core.config import DS1_MODEL_PATH, DS1_SCALER_PATH, DS2_MODEL_PATH, DS2_SCALER_PATH


router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": all(
            exists(path)
            for path in [DS1_MODEL_PATH, DS2_MODEL_PATH, DS1_SCALER_PATH, DS2_SCALER_PATH]
        ),
        "models": {
            "ds1": {"path": DS1_MODEL_PATH, "loaded": exists(DS1_MODEL_PATH)},
            "ds2": {"path": DS2_MODEL_PATH, "loaded": exists(DS2_MODEL_PATH)},
        },
        "scalers": {
            "ds1": {"path": DS1_SCALER_PATH, "loaded": exists(DS1_SCALER_PATH)},
            "ds2": {"path": DS2_SCALER_PATH, "loaded": exists(DS2_SCALER_PATH)},
        },
    }
