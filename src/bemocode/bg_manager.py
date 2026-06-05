from __future__ import annotations

import os
import subprocess
import threading
import uuid
from pathlib import Path

_MINIMAL_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "USER": os.environ.get("USER", ""),
    "SHELL": os.environ.get("SHELL", "/bin/bash"),
}


def start_background(command: str, cwd: Path) -> dict:
    """Start a shell command in background. stdout/stderr streamed to .bg/<id>.out/.err.
    Returns structured info immediately without waiting for completion."""
    bg_id = f"bg-{uuid.uuid4().hex[:8]}"
    bg_dir = cwd / ".bg"
    bg_dir.mkdir(parents=True, exist_ok=True)
    out_path = bg_dir / f"{bg_id}.out"
    err_path = bg_dir / f"{bg_id}.err"

    out_f = open(str(out_path), "w")
    err_f = open(str(err_path), "w")

    proc = subprocess.Popen(
        command, shell=True, cwd=str(cwd), env=_MINIMAL_ENV,
        stdout=out_f, stderr=err_f,
    )

    def _wait_and_close() -> None:
        """Wait for child process to finish, then close file descriptors."""
        proc.wait()
        out_f.close()
        err_f.close()

    t = threading.Thread(target=_wait_and_close, daemon=True)
    t.start()

    return {
        "background_id": bg_id,
        "output_file": str(out_path),
        "stderr_file": str(err_path),
        "pid": proc.pid,
        "message": f"Background task {bg_id} started with PID {proc.pid}. "
                   f"Output: {out_path}, Stderr: {err_path}",
    }
