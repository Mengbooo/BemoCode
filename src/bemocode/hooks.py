from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

HOOKS_FILE = "hooks.json"


def load_hooks(cwd: Path) -> dict[str, list[dict[str, Any]]]:
    """加载 hooks.json。不存在返回空 dict。"""
    file_path = cwd / HOOKS_FILE
    if not file_path.exists():
        return {}
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("hooks", data)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[hook warning] failed to load {file_path}: {exc}")
        return {}


def _matches(tool_name: str, matcher: str) -> bool:
    if matcher == "*":
        return True
    if "|" in matcher:
        return tool_name in matcher.split("|")
    return matcher == tool_name


def run_hooks(
    event: str,
    tool_name: str,
    tool_input: dict[str, Any],
    cwd: Path,
    tool_result: str | None = None,
) -> list[dict[str, Any]]:
    """执行匹配的 hook 命令。返回每个 hook 的执行结果列表。"""
    hooks = load_hooks(cwd)
    results: list[dict[str, Any]] = []

    for matcher, handlers in hooks.items():
        if not _matches(tool_name, matcher):
            continue
        for handler in handlers:
            hook_event = handler.get("event", handler.get("on", ""))
            if hook_event != event:
                continue
            command = handler.get("command", handler.get("run", ""))
            if not command:
                continue
            stdin_data = json.dumps({
                "event": event,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": tool_result,
                "cwd": str(cwd),
            })
            try:
                proc = subprocess.run(
                    command, shell=True, cwd=str(cwd),
                    input=stdin_data, capture_output=True,
                    text=True, timeout=30,
                )
                results.append({
                    "command": command,
                    "success": proc.returncode == 0,
                    "output": proc.stdout.strip() or proc.stderr.strip(),
                    "exit_code": proc.returncode,
                })
            except subprocess.TimeoutExpired:
                results.append({
                    "command": command,
                    "success": False,
                    "output": "hook timed out",
                    "exit_code": -1,
                })
            except Exception as exc:
                results.append({
                    "command": command,
                    "success": False,
                    "output": str(exc),
                    "exit_code": -1,
                })
    return results
