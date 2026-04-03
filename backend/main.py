"""FastAPI application bootstrap for the Faculty Student Diagnostic System."""

import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from backend.api.routers.health import router as health_router
from backend.api.routers.students import router as students_router
from backend.core.database import Base, engine


def _ensure_student_id_column(table_name: str) -> None:
    """Backfill legacy tables created before student_id existed."""
    with engine.begin() as conn:
        inspector = inspect(conn)
        if not inspector.has_table(table_name):
            return

        column_names = {c["name"] for c in inspector.get_columns(table_name)}
        if "student_id" in column_names:
            return

        conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN student_id INTEGER'))
        conn.execute(text(f'UPDATE "{table_name}" SET student_id = id WHERE student_id IS NULL'))

        # Keep writes fast and enforce uniqueness for upsert-style student updates.
        conn.execute(
            text(
                f'CREATE UNIQUE INDEX IF NOT EXISTS '
                f'ix_{table_name}_student_id ON "{table_name}" (student_id)'
            )
        )


def _ensure_primary_key_and_student_id_uniqueness(table_name: str) -> None:
    """Ensure id is the primary key and student_id is unique (not PK)."""
    with engine.begin() as conn:
        inspector = inspect(conn)
        if not inspector.has_table(table_name):
            return

        column_names = {c["name"] for c in inspector.get_columns(table_name)}
        if "id" not in column_names or "student_id" not in column_names:
            return

        pk_info = inspector.get_pk_constraint(table_name) or {}
        pk_columns = pk_info.get("constrained_columns") or []
        pk_name = pk_info.get("name")

        # If an old schema used student_id as PK, convert PK to id.
        if pk_columns != ["id"]:
            if pk_name:
                conn.execute(text(f'ALTER TABLE "{table_name}" DROP CONSTRAINT "{pk_name}"'))
            conn.execute(text(f'ALTER TABLE "{table_name}" ADD PRIMARY KEY (id)'))

        conn.execute(text(f'UPDATE "{table_name}" SET student_id = id WHERE student_id IS NULL'))
        conn.execute(text(f'ALTER TABLE "{table_name}" ALTER COLUMN student_id SET NOT NULL'))
        conn.execute(
            text(
                f'CREATE UNIQUE INDEX IF NOT EXISTS '
                f'ix_{table_name}_student_id ON "{table_name}" (student_id)'
            )
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        _ensure_student_id_column("student_data_ds1")
        _ensure_student_id_column("student_data_ds2")
        _ensure_primary_key_and_student_id_uniqueness("student_data_ds1")
        _ensure_primary_key_and_student_id_uniqueness("student_data_ds2")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not create database tables: {exc}. "
            "Ensure DATABASE_URL is set correctly and the database is reachable.",
            RuntimeWarning,
            stacklevel=2,
        )
    yield


app = FastAPI(
    title="Faculty Student Diagnostic System",
    description=(
        "Predict student exam risk using a Stacking Ensemble model "
        "(XGBoost + CatBoost + RandomForest + LassoCV) with SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(students_router)
