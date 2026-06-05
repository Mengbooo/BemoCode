from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

# Tools that are always read-only and safe to allow in any mode
_READONLY_TOOLS = frozenset({
    "read_file", "list_files", "glob", "grep", "project_tree",
    "git_status", "git_diff", "system_date", "echo",
    "memory_recall",
    "cron_list",
})

# Tools that should always ask — user should know agent is pausing to
# ask a question or accessing external resources
_ASK_TOOLS = frozenset({
    "ask_user_question", "web_fetch", "web_search",
})

# Low-risk writes: auto-allowed in default/acceptEdits, denied in plan mode
_LOW_RISK_WRITES = frozenset({
    "memory_write",
    "cron_create",
    "cron_cancel",
    "enter_plan_mode",
    "exit_plan_mode",
    "todo_write",
})

# Dangerous command patterns — matched against the full command string
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-rf\s+/", "rm -rf / destroys the filesystem"),
    (r"\bsudo\b", "sudo grants root privileges"),
    (r"chmod\s+-R\s+777", "chmod -R 777 makes everything world-writable"),
    (r"curl.*\|\s*(ba)?sh", "curl pipe shell is a common attack vector"),
    (r"git\s+push\s+--force", "git push --force overwrites remote history"),
    (r"git\s+push\s+-f\b", "git push -f overwrites remote history"),
    (r"git\s+push\b", "git push modifies remote repository"),
    (r"git\s+reset\s+--hard", "git reset --hard discards local changes"),
]


def _is_dangerous(command: str) -> str | None:
    """Check if a command matches any dangerous pattern. Returns reason string or None."""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return reason
    return None


@dataclass
class PermissionRequest:
    """Describes a single tool invocation for permission evaluation."""
    tool_name: str
    args: dict[str, Any]
    mode: str
    cwd: Path


@dataclass
class PermissionDecision:
    """Result of permission evaluation: allow, ask, or deny."""
    behavior: str  # "allow" | "ask" | "deny"
    message: str | None = None


def decide_permission(request: PermissionRequest) -> PermissionDecision:
    """Unified entry point: returns allow/ask/deny based on tool name, args, and mode."""
    tool_name = request.tool_name
    args = request.args
    mode = request.mode

    if tool_name in _ASK_TOOLS:
        return PermissionDecision("ask")

    # plan mode: only allow read-only tools; deny all write tools
    if mode == "plan":
        if tool_name in _READONLY_TOOLS:
            return PermissionDecision("allow")
        return PermissionDecision(
            "deny",
            f"plan mode: {tool_name} is not allowed. Only read-only tools can run in plan mode."
        )

    # Read-only tools are allowed in all modes by default
    if tool_name in _READONLY_TOOLS:
        return PermissionDecision("allow")

    # Low-risk writes are auto-allowed in default/acceptEdits modes
    if tool_name in _LOW_RISK_WRITES:
        return PermissionDecision("allow")

    # bash tool: dangerous command detection — blocks before any UI
    if tool_name == "bash":
        command = args.get("command", "")
        danger_reason = _is_dangerous(command)
        if danger_reason:
            return PermissionDecision("deny", f"Dangerous command blocked: {danger_reason}")

    # acceptEdits mode: skip confirmation UI for file edits, but safety checks still run
    if mode == "acceptEdits":
        if tool_name in ("file_write", "file_edit"):
            return PermissionDecision("allow")

    # default mode: write tools and bash require confirmation
    return PermissionDecision("ask")
