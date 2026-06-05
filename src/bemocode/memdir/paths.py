from __future__ import annotations

from pathlib import Path

MEMORY_DIR = ".agent/memory"
INDEX_FILE = "MEMORY.md"
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25 * 1024


def memory_dir(cwd: Path) -> Path:
    return cwd / MEMORY_DIR


def index_path(cwd: Path) -> Path:
    return memory_dir(cwd) / INDEX_FILE
