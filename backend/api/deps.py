"""FastAPI dependency injectors.

get_db      — yields a SQLAlchemy session scoped to the request.
get_models  — returns the pre-loaded model dict from app.state.
get_scalers — returns the pre-loaded scaler dict from app.state.
"""

from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from backend.core.database import SessionLocal


def get_db() -> Session:
    """Yield a database session and ensure it is closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_models(request: Request) -> dict[str, Any]:
    """Return the model registry pre-loaded during application startup."""
    models = getattr(request.app.state, "models", None)
    if models is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check that .joblib files exist and the server started cleanly.",
        )
    return models


def get_scalers(request: Request) -> dict[str, Any]:
    """Return the scaler registry pre-loaded during application startup."""
    scalers = getattr(request.app.state, "scalers", None)
    if scalers is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Scalers not loaded. Check that .joblib files exist and the server started cleanly.",
        )
    return scalers


def get_explainers(request: Request) -> dict[str, Any]:
    """Return the pre-built SHAP explainer registry from application startup."""
    explainers = getattr(request.app.state, "explainers", None)
    if explainers is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="SHAP explainers not built. Server may still be starting up.",
        )
    return explainers
