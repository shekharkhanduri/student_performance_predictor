#!/usr/bin/env python3
"""
Convenience launcher for the Faculty Student Diagnostic System.

Usage
-----
Start the FastAPI backend (default port 8000):
    python run.py backend

Start the Streamlit frontend (default port 8501):
    python run.py frontend

Start both in separate processes:
    python run.py both
"""

import subprocess
import sys


def start_backend():
    print("Starting FastAPI backend on http://0.0.0.0:8000 …")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        check=True,
    )


def start_frontend():
    print("Starting Streamlit frontend on http://0.0.0.0:8501 …")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            "frontend/app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
        ],
        check=True,
    )


def start_both():
    import multiprocessing
    backend_proc = multiprocessing.Process(target=start_backend, daemon=False)
    backend_proc.start()
    try:
        start_frontend()
    finally:
        backend_proc.terminate()
        backend_proc.join(timeout=5)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backend"
    if cmd == "backend":
        start_backend()
    elif cmd == "frontend":
        start_frontend()
    elif cmd == "both":
        start_both()
    else:
        print(f"Unknown command: {cmd!r}. Use 'backend', 'frontend', or 'both'.")
        sys.exit(1)
