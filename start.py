"""Start GameSense with the right Python:  python start.py

Finds the project's Python environment (or creates one), installs what is missing the first time,
then starts the app. Works with any Python 3.11+ to launch it - it re-runs itself with the right one.
"""
from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KNOWN = [ROOT / ".venv", Path(r"D:\1_GameSense_code\GameSense\.venv")]
CHECK = "import pandas,numpy,sklearn,scipy,joblib,yaml,streamlit,plotly"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def find_python() -> Path:
    for v in KNOWN:
        if venv_python(v).is_file():
            return venv_python(v)
    print("No Python environment found - creating .venv in this folder (first time only)...")
    subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    return venv_python(ROOT / ".venv")


def free_port(start: int = 8501) -> int:
    for port in range(start, start + 30):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main() -> int:
    py = find_python()
    print(f"Using Python: {py}")
    if subprocess.run([str(py), "-c", CHECK], capture_output=True).returncode != 0:
        print("Installing the packages GameSense needs (a few minutes, first time only)...")
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    if "--check" in sys.argv:
        print("Environment OK.")
        return 0
    port = free_port()
    print(f"Starting GameSense at http://localhost:{port}  (Ctrl+C to stop)")
    print("An older GameSense window may still be open on another port - ignore or close it." if port != 8501 else "")
    return subprocess.call([str(py), "-m", "streamlit", "run", "app.py", "--server.port", str(port)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
