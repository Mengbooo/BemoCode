from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue


@dataclass
class CronJob:
    job_id: str
    slash: str
    every_seconds: int
    label: str = ""


def _cron_path(cwd: Path) -> Path:
    d = cwd / ".agent"
    d.mkdir(parents=True, exist_ok=True)
    return d / "cron.json"


def _load_jobs(cwd: Path) -> list[CronJob]:
    path = _cron_path(cwd)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [
            CronJob(
                job_id=j["job_id"],
                slash=j["slash"],
                every_seconds=j["every_seconds"],
                label=j.get("label", ""),
            )
            for j in data
        ]
    except (json.JSONDecodeError, KeyError, OSError):
        return []


def _save_jobs(cwd: Path, jobs: list[CronJob]) -> None:
    data = [
        {
            "job_id": j.job_id,
            "slash": j.slash,
            "every_seconds": j.every_seconds,
            "label": j.label,
        }
        for j in jobs
    ]
    _cron_path(cwd).write_text(json.dumps(data, indent=2))


class CronScheduler:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self._jobs: list[CronJob] = _load_jobs(cwd)
        self._pending: Queue[str] = Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def add_job(self, slash: str, every_seconds: int, label: str = "") -> CronJob:
        jid = uuid.uuid4().hex[:12]
        job = CronJob(job_id=jid, slash=slash, every_seconds=every_seconds, label=label)
        with self._lock:
            self._jobs.append(job)
            _save_jobs(self.cwd, self._jobs)
        return job

    def list_jobs(self) -> list[CronJob]:
        with self._lock:
            return list(self._jobs)

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            before = len(self._jobs)
            self._jobs = [j for j in self._jobs if j.job_id != job_id]
            if len(self._jobs) < before:
                _save_jobs(self.cwd, self._jobs)
                return True
        return False

    def drain_pending(self) -> list[str]:
        items: list[str] = []
        while not self._pending.empty():
            items.append(self._pending.get_nowait())
        return items

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()

    def _run(self) -> None:
        last_fire: dict[str, float] = {}
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                jobs = list(self._jobs)
            for job in jobs:
                last = last_fire.get(job.job_id, 0)
                if now - last >= job.every_seconds:
                    self._pending.put(job.slash)
                    last_fire[job.job_id] = now
            self._stop_event.wait(1.0)
