"""Idempotent database schema migrations.

Runs on every startup via run_migrations(engine). Each helper checks
for the column / index before issuing DDL, so they are safe to call
repeatedly on existing databases.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


# ── Low-level helpers ────────────────────────────────────────────────────────

def _add_column_if_missing(conn, table_name: str, column_name: str, column_def: str) -> None:
    """ALTER TABLE … ADD COLUMN only when the column is absent."""
    inspector = inspect(conn)
    if not inspector.has_table(table_name):
        return
    existing = {c["name"] for c in inspector.get_columns(table_name)}
    if column_name not in existing:
        conn.execute(
            text(f'ALTER TABLE "{table_name}" ADD COLUMN {column_name} {column_def}')
        )


def _ensure_student_id_column(conn, table_name: str) -> None:
    """
    Ensure student_id exists and is populated from id.

    Legacy tables may have student_id as user-supplied; new rows auto-assign
    student_id = id.  This migration backfills any NULL student_id values.
    """
    inspector = inspect(conn)
    if not inspector.has_table(table_name):
        return

    existing = {c["name"] for c in inspector.get_columns(table_name)}
    if "student_id" not in existing:
        conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN student_id INTEGER'))

    # Backfill auto-assignment for any rows where student_id is missing.
    conn.execute(
        text(f'UPDATE "{table_name}" SET student_id = id WHERE student_id IS NULL')
    )

    # Unique index enables fast lookups and prevents duplicates.
    conn.execute(
        text(
            f"CREATE UNIQUE INDEX IF NOT EXISTS "
            f'ix_{table_name}_student_id ON "{table_name}" (student_id)'
        )
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_migrations(engine: Engine) -> None:
    """Apply all schema migrations.  Safe to call on every startup (idempotent)."""
    tables = ["student_data_ds1", "student_data_ds2"]

    with engine.begin() as conn:
        for table in tables:
            # Core identifier columns
            _ensure_student_id_column(conn, table)

            # SHAP computation state: 'pending' | 'done' | 'failed'
            _add_column_if_missing(
                conn, table, "shap_status", "VARCHAR(8) NOT NULL DEFAULT 'pending'"
            )

            # Audit timestamp — set on every update
            _add_column_if_missing(
                conn, table, "updated_at", "TIMESTAMP WITH TIME ZONE"
            )
