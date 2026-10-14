"""
desktop_app.py — Qti π native desktop wrapper.

Ported from Ashmit0107/Health-Companion's desktop_app.py (audited at
C:\\SIH\\health_companion\\desktop_app.py). NOT a packaged/bundled app: a
thin native window (pywebview) pointing at the SAME live local FastAPI
server + static frontend files this project runs in a browser. No build
step, no frozen assets — it starts backend/main.py as a subprocess and
loads http://127.0.0.1:8000/, which FastAPI serves directly from
frontend/ via StaticFiles.

  - Editing frontend/index.html|app.js|style.css -> refresh (Ctrl+R) or
    restart this app. No rebuild.
  - Editing backend/*.py -> restart this app to pick up the change.

Run:
    cd "C:\\SIH\\Qti π"
    pip install -r requirements.txt
    python desktop_app.py

Requires ml/*.py to have been run at least once (see HOW_TO_RUN.md) so
ml/saved_models/*.joblib exist, otherwise risk-score endpoints will report
"Model not trained yet" for the affected verticals.
"""
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error

import webview

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST = "127.0.0.1"
_PORT = 8000
_BASE_URL = f"http://{_HOST}:{_PORT}"
_APP_URL = f"{_BASE_URL}/"
_STARTUP_TIMEOUT_S = 20


def _backend_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{_BASE_URL}/", timeout=1) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def _wait_for_backend(timeout_s: float, poll_s: float = 0.5) -> bool:
    waited = 0.0
    while waited < timeout_s:
        if _backend_is_up():
            return True
        time.sleep(poll_s)
        waited += poll_s
    return False


def _start_backend():
    """Start uvicorn as a subprocess rooted at the project root (not
    backend/), because backend.main uses package-relative imports
    (`import database as db` via sys.path insert) and StaticFiles mounts
    frontend/ relative to backend/main.py's own file location either way.

    On Windows, a just-stopped `uvicorn --reload` session can leave its
    listening socket in a brief lingering state for a second or two even
    after the process itself has exited, so a check done at the exact
    instant this script starts can wrongly report the port as free. Give
    that a few seconds to settle before deciding whether to spawn our own
    server.
    """
    if _wait_for_backend(3.0):
        print(f"[desktop_app] Backend already running at {_BASE_URL} - reusing it.")
        return None

    print(f"[desktop_app] Starting backend: uvicorn backend.main:app --port {_PORT} (cwd={_THIS_DIR})")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(_PORT)],
        cwd=_THIS_DIR,
    )

    if _wait_for_backend(_STARTUP_TIMEOUT_S):
        print("[desktop_app] Backend is up.")
        return proc

    # The subprocess may have already exited (e.g. "address already in use"
    # if something else grabbed the port between our check and the spawn).
    # If it exited, don't leave a dead Popen handle around to be terminated
    # later -- and check one more time whether SOME server (ours or a
    # leftover one) is actually serving requests before giving up.
    exit_code = proc.poll()
    if exit_code is not None:
        print(
            f"[desktop_app] ERROR: backend process exited immediately (code {exit_code}). "
            "This is almost always 'port 8000 already in use' -- another uvicorn/python "
            "process (e.g. a terminal you forgot to close, or a previous --reload session "
            "whose socket hadn't fully released yet) is still bound to it. Fix: in a "
            "PowerShell window run `netstat -ano | findstr :8000` to find the PID, then "
            "`taskkill /PID <pid> /F`, and re-run this script. Opening the window anyway "
            "in case a leftover server is actually serving the dashboard."
        )
        if _backend_is_up():
            print("[desktop_app] A server IS responding on this port after all - reusing it.")
        return None

    print(
        f"[desktop_app] WARNING: backend did not respond within {_STARTUP_TIMEOUT_S}s. "
        "Opening the window anyway - the dashboard will show 'offline'. "
        "Check the console output above for a Python traceback from uvicorn."
    )
    return proc


def main():
    backend_proc = _start_backend()

    webview.create_window(
        "Qti π — Cardiac & Neuro Companion",
        _APP_URL,
        width=1180,
        height=820,
        resizable=True,
        min_size=(900, 640),
    )

    try:
        webview.start()
    finally:
        if backend_proc is not None:
            print("[desktop_app] Shutting down backend subprocess...")
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()


if __name__ == "__main__":
    main()
