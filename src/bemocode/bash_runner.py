from __future__ import annotations

import os
import subprocess
from pathlib import Path

_MINIMAL_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "USER": os.environ.get("USER", ""),
    "SHELL": os.environ.get("SHELL", "/bin/bash"),
}


def truncate_output(output: str, max_chars: int = 12000) -> str:
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n... [truncated {len(output) - max_chars} chars]"


def run_sync(command: str, cwd: Path, timeout: int = 30) -> str:
    """Synchronous shell command execution. cwd locked to project dir, kills on timeout."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env=_MINIMAL_ENV,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"

    output = proc.stdout.decode("utf-8", errors="replace")
    if proc.stderr:
        stderr_output = proc.stderr.decode("utf-8", errors="replace")
        if stderr_output:
            output += "\n" + stderr_output

    truncated = truncate_output(output.strip(), max_chars=12000)
    if proc.returncode != 0:
        return f"exit code {proc.returncode}\n{truncated}"
    return truncated if truncated else "(no output)"
