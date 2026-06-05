from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SlashHandler = Callable[["list[str]", "SlashContext"], "SlashResult"]


@dataclass
class SlashCommand:
    name: str
    description: str
    handler: SlashHandler


@dataclass
class SlashContext:
    """slash handler 接收的运行时上下文。"""
    cwd: Path
    permission_mode: str
    model: str
    provider: str
    session_id: str | None


class SlashResult:
    def __init__(
        self,
        handled: bool = True,
        should_query: bool = False,
        prompt: str = "",
        message: str = "",
    ) -> None:
        self.handled = handled
        self.should_query = should_query
        self.prompt = prompt
        self.message = message


_registry: dict[str, SlashCommand] = {}


def register(name: str, description: str, handler: SlashHandler) -> None:
    _registry[name] = SlashCommand(name=name, description=description, handler=handler)


def list_slash_commands() -> list[SlashCommand]:
    return list(_registry.values())


def dispatch_slash(line: str, ctx: SlashContext) -> SlashResult:
    if not line.startswith("/"):
        return SlashResult(handled=False)
    try:
        parts = shlex.split(line[1:].strip())
    except ValueError as exc:
        return SlashResult(handled=True, message=f"Invalid command syntax: {exc}")
    if not parts:
        return SlashResult(handled=False)
    cmd = _registry.get(parts[0])
    if cmd is None:
        return SlashResult(handled=True, message=f"Unknown command: /{parts[0]}")
    return cmd.handler(parts[1:], ctx)


# ── Built-in slash commands ────────────────────────────────


def _cmd_help(args: list[str], ctx: SlashContext) -> SlashResult:
    lines = ["Available commands:"]
    for cmd in sorted(_registry.values(), key=lambda c: c.name):
        lines.append(f"  /{cmd.name} — {cmd.description}")
    return SlashResult(handled=True, message="\n".join(lines))


def _cmd_model(args: list[str], ctx: SlashContext) -> SlashResult:
    return SlashResult(
        handled=True,
        message=f"provider: {ctx.provider}  model: {ctx.model}",
    )


def _cmd_context(args: list[str], ctx: SlashContext) -> SlashResult:
    return SlashResult(
        handled=True,
        message=f"cwd: {ctx.cwd}\nsession: {ctx.session_id or '(none)'}\nmode: {ctx.permission_mode}",
    )


def _cmd_permissions(args: list[str], ctx: SlashContext) -> SlashResult:
    return SlashResult(
        handled=True,
        message=f"permission mode: {ctx.permission_mode}",
    )


def _cmd_plan(args: list[str], ctx: SlashContext) -> SlashResult:
    return SlashResult(
        handled=True,
        message=f"plan status: current mode is '{ctx.permission_mode}'",
    )


register("help", "Show available commands", _cmd_help)
register("model", "Show current model", _cmd_model)
register("context", "Show current session context", _cmd_context)
register("permissions", "Show current permission mode", _cmd_permissions)
register("plan", "Show plan status", _cmd_plan)
