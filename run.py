#!/usr/bin/env python3
"""
Convenience launcher for the Faculty Student Diagnostic System.

Usage
-----
Start the FastAPI backend (default port 8000):
    python run.py backend

Start the React+Vite frontend (default port 5173):
    python run.py frontend

Start both in separate processes:
    python run.py both
"""

import subprocess
import sys
import os


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
    print("Starting React+Vite frontend on http://localhost:5173 …")
    print("(Make sure dependencies are installed: cd frontend && npm install)")
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    subprocess.run(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
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
