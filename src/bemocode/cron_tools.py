from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .scheduler import CronScheduler

if TYPE_CHECKING:
    from .tools import ToolContext

_scheduler: CronScheduler | None = None


def set_scheduler(scheduler: CronScheduler | None) -> None:
    global _scheduler
    _scheduler = scheduler


def _get_scheduler(ctx: "ToolContext") -> CronScheduler:
    if _scheduler is not None:
        return _scheduler
    # One-shot mode: read/write cron.json directly
    s = CronScheduler(ctx.cwd)
    return s


def cron_create(args: dict[str, Any], ctx: "ToolContext") -> str:
    scheduler = _get_scheduler(ctx)
    slash = args.get("slash", "")
    every_seconds = int(args.get("every_seconds", 0))
    label = args.get("label", "")
    if not slash:
        return "error: missing required argument 'slash'"
    if every_seconds <= 0:
        return "error: every_seconds must be positive"
    job = scheduler.add_job(slash, every_seconds, label)
    return f"Cron job created: {job.job_id} — every {every_seconds}s: {slash}"


def cron_list(args: dict[str, Any], ctx: "ToolContext") -> str:
    scheduler = _get_scheduler(ctx)
    jobs = scheduler.list_jobs()
    if not jobs:
        return "(no cron jobs)"
    lines = [f"- {j.job_id}  every {j.every_seconds}s  {j.label or j.slash}" for j in jobs]
    return "\n".join(lines)


def cron_cancel(args: dict[str, Any], ctx: "ToolContext") -> str:
    scheduler = _get_scheduler(ctx)
    job_id = args.get("job_id", "")
    if not job_id:
        return "error: missing required argument 'job_id'"
    if scheduler.cancel_job(job_id):
        return f"Cron job cancelled: {job_id}"
    return f"error: cron job not found: {job_id}"
