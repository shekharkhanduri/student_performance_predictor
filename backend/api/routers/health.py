"""Health check endpoint — GET /api/v1/health"""

from fastapi import APIRouter, Request

from backend.core.config import DS1_MODEL_PATH, DS1_SCALER_PATH, DS2_MODEL_PATH, DS2_SCALER_PATH

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    """
    Returns API status and whether models/scalers are loaded in app.state.

    model_loaded: true only when all four artefacts (both models + both scalers)
    are present in memory.
    """
    models = getattr(request.app.state, "models", {})
    scalers = getattr(request.app.state, "scalers", {})

    ds1_ready = "ds1" in models and "ds1" in scalers
    ds2_ready = "ds2" in models and "ds2" in scalers

    return {
        "status": "ok",
        "model_loaded": ds1_ready and ds2_ready,
        "models": {
            "ds1": {"path": DS1_MODEL_PATH, "loaded": "ds1" in models},
            "ds2": {"path": DS2_MODEL_PATH, "loaded": "ds2" in models},
        },
        "scalers": {
            "ds1": {"path": DS1_SCALER_PATH, "loaded": "ds1" in scalers},
            "ds2": {"path": DS2_SCALER_PATH, "loaded": "ds2" in scalers},
        },
    }
