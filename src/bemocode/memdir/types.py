from __future__ import annotations

from dataclasses import dataclass

MEMORY_TYPES = ("user", "feedback", "project", "reference")


@dataclass
class MemoryEntry:
    mem_type: str
    title: str
    slug: str
    body: str
    file_path: str
