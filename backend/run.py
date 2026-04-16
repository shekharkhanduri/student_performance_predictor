"""Convenience launcher — start the backend API with uvicorn."""

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    print("Starting backend API at http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
