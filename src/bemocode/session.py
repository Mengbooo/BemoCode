from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sessions_dir(cwd: Path) -> Path:
    """会话 JSONL 的存放目录：.agent/sessions/<sanitized-cwd>"""
    safe = str(cwd.resolve()).lstrip("/").replace("/", "-").replace("\\", "-").replace(":", "")
    d = cwd / ".agent" / "sessions" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


class Session:
    """一次会话。管理 session id、JSONL 落盘、读取历史消息。"""

    def __init__(
        self,
        cwd: Path,
        session_id: str,
        file_path: Path,
        resumed: bool = False,
    ) -> None:
        self.cwd = cwd
        self.session_id = session_id
        self.file_path = file_path
        self.resumed = resumed

    @classmethod
    def create(cls, cwd: Path) -> "Session":
        sid = uuid.uuid4().hex[:12]
        file_path = _sessions_dir(cwd) / f"{sid}.jsonl"
        file_path.touch()
        return cls(cwd=cwd, session_id=sid, file_path=file_path, resumed=False)

    @classmethod
    def load_latest(cls, cwd: Path) -> "Session | None":
        sessions_dir = _sessions_dir(cwd)
        jsonl_files = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not jsonl_files:
            return None
        latest = jsonl_files[-1]
        return cls(cwd=cwd, session_id=latest.stem, file_path=latest, resumed=True)

    @classmethod
    def load_by_id(cls, cwd: Path, session_id: str) -> "Session | None":
        file_path = _sessions_dir(cwd) / f"{session_id}.jsonl"
        if not file_path.exists():
            return None
        return cls(cwd=cwd, session_id=session_id, file_path=file_path, resumed=True)

    def append_messages(self, msgs: list[dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with open(self.file_path, "a", encoding="utf-8") as f:
            for msg in msgs:
                record = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": now,
                }
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    @property
    def history(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if not self.file_path.exists():
            return messages
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    messages.append({
                        "role": record["role"],
                        "content": record["content"],
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        return messages

    def message_count(self) -> int:
        return len(self.history)
