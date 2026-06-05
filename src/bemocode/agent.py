from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from .compact_basic import compact as _compact_messages
from .hooks import run_hooks
from .fs_safety import (
    SkipPolicy,
    apply_single_replace,
    check_mtime_conflict,
    ensure_read_before_edit,
    load_gitignore,
    resolve_in_cwd,
)
from .model import ModelProvider, ModelResponse
from .permissions import PermissionRequest, decide_permission
from .project_memory import load_agent_md
from .prompt_ui import (
    confirm_command,
    confirm_edit,
    confirm_plan,
    confirm_tool_use,
    prompt_single_choice,
    render_diff,
)
from .session import Session
from .tools import ToolContext, ToolRegistry, ToolResult


_SYSTEM_CORE = (
    "You are an AI coding agent running inside a CLI harness. "
    "You have access to tools for reading/writing files, running shell commands, "
    "searching the web, and asking the user questions. "
    "Use tools when needed; respond directly when you can."
)


def build_system_prompt(cwd: Path) -> str:
    """组装 system prompt：核心指南 + AGENT.md + MEMORY.md 索引。"""
    from .memdir.store import load_index as load_memory_index

    parts: list[str] = [_SYSTEM_CORE]
    agent_md = load_agent_md(cwd)
    if agent_md:
        parts.append(agent_md)

    memory_index = load_memory_index(cwd)
    if memory_index:
        parts.append(f"<project-memory>\n{memory_index}\n</project-memory>")

    return "\n\n".join(parts)


@dataclass
class AgentResult:
    final: str
    trace: list[str]
    messages: list[dict[str, Any]] = field(default_factory=list)


def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    if response.assistant_content:
        return {"role": "assistant", "content": response.assistant_content}
    # fallback: mock provider 没有 assistant_content 时自己拼一份
    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for call in response.tool_calls or []:
        content.append({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
        })
    return {"role": "assistant", "content": content}


def _tool_result_message(tool_call_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
                "is_error": is_error,
            }
        ],
    }


def _emit_styled(console: Console, line: str) -> None:
    """美化 agent trace 输出：用分割线区分不同消息类型。"""
    from rich.markdown import Markdown
    from rich.text import Text

    if line.startswith("tool_call:"):
        rest = line[len("tool_call:"):].strip()
        parts = rest.split(" ", 1)
        tool_name = parts[0] if parts else rest
        args_str = parts[1] if len(parts) > 1 else ""

        console.print()
        _sep(console, f"Agent · {tool_name}", style="dim")
        if args_str:
            display_args = args_str if len(args_str) <= 200 else args_str[:200] + "..."
            console.print(f"  {display_args}", style="dim")

    elif line.startswith("observation:"):
        rest = line[len("observation:"):].strip()
        _sep(console, "result", style="dim")
        if len(rest) > 400:
            rest = rest[:400] + "..."
        console.print(f"  {rest}", style="dim")

    elif line.startswith("final:"):
        rest = line[len("final:"):].strip()
        if not rest:
            return
        console.print()
        _sep(console, "Agent", style="bold")
        console.print()
        md = Markdown(rest, code_theme="monokai")
        console.print(md)
        console.print()

    elif line.startswith("interrupted"):
        _sep(console, "interrupted", style="dim")
        console.print(f"  {line}", style="dim")

    elif line.startswith("continue:"):
        rest = line[len("continue:"):].strip()
        _sep(console, "continue", style="dim")
        console.print(f"  {rest}", style="dim")

    elif line.startswith("compacted"):
        console.print(f"  ⊟ {line}", style="dim")

    else:
        console.print(f"    {line}", markup=False)


def _sep(console: Console, label: str, style: str = "dim") -> None:
    """画一条带标签的分割线。"""
    width = console.width or 78
    label_text = f"  {label}  "
    from rich.text import Text
    line = Text()
    line.append("─" * 4, style=style)
    line.append(label_text, style=f"bold {style}" if style != "dim" else "bold")
    remaining = width - 4 - len(label_text)
    if remaining > 0:
        line.append("─" * remaining, style=style)
    console.print(line)


def run_agent(
    prompt: str,
    provider: ModelProvider,
    tools: ToolRegistry,
    max_steps: int = 8,
    cwd: Path | None = None,
    permission_mode: str = "default",
    session: Session | None = None,
    system_prompt: str | None = None,
) -> AgentResult:
    resolved_cwd = cwd or Path.cwd()
    ctx = ToolContext(
        cwd=resolved_cwd,
        skip_policy=SkipPolicy.default(gitignore=load_gitignore(resolved_cwd)),
    )
    trace: list[str] = []
    console = Console()

    def emit(line: str) -> None:
        trace.append(line)
        _emit_styled(console, line)

    # Build system prompt
    system = system_prompt or build_system_prompt(resolved_cwd)

    # Initialize messages from session history or fresh
    if session and session.history:
        messages = list(session.history)
        messages.append({"role": "user", "content": prompt})
    else:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    if session:
        session.append_messages([messages[-1]])

    for step in range(max_steps):
        # Auto-compact when message count exceeds threshold
        if len(messages) > 40:
            messages = _compact_messages(messages, keep=8)
            console.print(f"[dim]compacted: {len(messages)} messages remaining[/dim]")

        response = provider.complete(messages, tools=tools.list(), system=system)
        messages.append(_assistant_message(response))

        if session:
            session.append_messages([messages[-1]])

        if not response.tool_calls:
            final = response.text or ""
            emit(f"final: {final}")
            return AgentResult(final=final, trace=trace, messages=messages)

        tool_result_blocks: list[dict[str, Any]] = []
        for call in response.tool_calls:
            emit(f"tool_call: {call.name} {call.arguments}")

            # ── Plan mode: enter_plan_mode / exit_plan_mode ──────
            if call.name == "enter_plan_mode":
                permission_mode = "plan"
                result = ToolResult(call.id, "Plan mode on. Write tools denied. Draft a plan, then call exit_plan_mode.", is_error=False)
                emit(f"observation: {result.content}")
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": False,
                })
                continue

            if call.name == "exit_plan_mode":
                plan_summary = call.arguments.get("plan_summary", "")
                if not confirm_plan(plan_summary):
                    result = ToolResult(call.id, "Plan not approved. Revise the plan and call exit_plan_mode again.", is_error=True)
                    emit(f"observation: {result.content}")
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": True,
                    })
                    continue
                permission_mode = "acceptEdits"
                result = ToolResult(call.id, "Plan approved. Write tools are now enabled.", is_error=False)
                emit(f"observation: {result.content}")
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": False,
                })
                continue

            # ── Day 5: unified permission gate ──────────────────
            request = PermissionRequest(
                tool_name=call.name,
                args=call.arguments,
                mode=permission_mode,
                cwd=ctx.cwd,
            )
            decision = decide_permission(request)

            # Edit preview: for file_write/file_edit, run Day 4 safety
            # checks regardless of mode (acceptEdits skips CONFIRMATION,
            # not validation). Compute old/new content + validation errors.
            edit_preview: tuple[str, str, str] | None = None  # (path_str, old, new)
            if call.name in ("file_write", "file_edit") and decision.behavior != "deny":
                path_str = call.arguments.get("file_path", "")
                if not path_str:
                    result = ToolResult(call.id, "error: missing required argument 'file_path'", is_error=True)
                    emit(f"observation: {result.content}")
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": True,
                    })
                    continue

                # 1) Path resolution — out-of-bounds is an error
                try:
                    path = resolve_in_cwd(ctx.cwd, path_str)
                except (ValueError, OSError) as exc:
                    result = ToolResult(call.id, f"error: {exc}", is_error=True)
                    emit(f"observation: {result.content}")
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": True,
                    })
                    continue

                old_content = path.read_text(encoding="utf-8") if path.exists() else ""

                # 2) Pre-flight validation: read-before-edit + mtime conflict
                validation_error: str | None = None
                if call.name == "file_write":
                    if path.exists():
                        validation_error = (
                            ensure_read_before_edit(ctx.read_state, path)
                            or check_mtime_conflict(ctx.read_state, path)
                        )
                else:  # file_edit
                    if not path.exists():
                        validation_error = f"error: file does not exist: {path_str}"
                    else:
                        validation_error = (
                            ensure_read_before_edit(ctx.read_state, path)
                            or check_mtime_conflict(ctx.read_state, path)
                        )

                # 3) Compute new_content
                new_content: str | None = None
                if call.name == "file_write":
                    new_content = call.arguments.get("content", "")
                elif call.name == "file_edit" and validation_error is None:
                    new_content, replace_err = apply_single_replace(
                        old_content,
                        call.arguments.get("old_string", ""),
                        call.arguments.get("new_string", ""),
                        bool(call.arguments.get("replace_all", False)),
                    )
                    if replace_err is not None:
                        validation_error = replace_err

                # 4) Validation failure → error observation, no diff, no confirm
                if validation_error is not None:
                    result = ToolResult(call.id, validation_error, is_error=True)
                    emit(f"observation: {result.content}")
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": True,
                    })
                    continue

                # Store preview for ask/allow branches
                edit_preview = (path_str, old_content, new_content or "")

            # ── Permission branches ─────────────────────────────
            if decision.behavior == "deny":
                result = ToolResult(call.id, f"error: {decision.message}", is_error=True)
                emit(f"observation: {result.content}")
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": True,
                })
                continue

            elif decision.behavior == "ask":
                if call.name in ("file_write", "file_edit") and edit_preview is not None:
                    path_str, old_content, new_content = edit_preview
                    diff_text = render_diff(old_content, new_content, path_str)
                    console.print(f"\n[bold]Diff for {path_str}:[/bold]")
                    console.print(diff_text)
                    if not confirm_edit(path_str):
                        result = ToolResult(call.id, "error: edit rejected by user", is_error=True)
                        emit(f"observation: {result.content}")
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": result.tool_call_id,
                            "content": result.content,
                            "is_error": True,
                        })
                        continue

                elif call.name == "bash":
                    command = call.arguments.get("command", "")
                    timeout = call.arguments.get("timeout", 30)
                    console.print(f"\n[bold yellow]Command:[/bold yellow] {command}")
                    console.print(f"[dim]timeout: {timeout}s  cwd: {ctx.cwd}[/dim]")
                    if not confirm_command(command):
                        result = ToolResult(call.id, "error: command rejected by user", is_error=True)
                        emit(f"observation: {result.content}")
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": result.tool_call_id,
                            "content": result.content,
                            "is_error": True,
                        })
                        continue

                elif call.name in ("web_fetch", "web_search"):
                    if not confirm_tool_use(call.name, call.arguments):
                        result = ToolResult(call.id, "error: tool use rejected by user", is_error=True)
                        emit(f"observation: {result.content}")
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": result.tool_call_id,
                            "content": result.content,
                            "is_error": True,
                        })
                        continue

                elif call.name == "ask_user_question":
                    question = call.arguments.get("prompt", "")
                    options = call.arguments.get("options", [])
                    if not isinstance(options, list):
                        options = []
                    labels = [str(o) for o in options]
                    selected = prompt_single_choice(question, labels)
                    if selected is None:
                        result = ToolResult(call.id, "User skipped the question.", is_error=False)
                    else:
                        result = ToolResult(call.id, f'User selected: "{selected}"', is_error=False)
                    emit(f"observation: {result.content}")
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    })
                    continue

            # ── PreToolUse hooks ─────────────────────────────
            if decision.behavior != "deny":
                pre_hooks = run_hooks("PreToolUse", call.name, call.arguments, ctx.cwd)
                pre_blocked = [h for h in pre_hooks if not h["success"]]
                if pre_blocked:
                    blocked_msgs = "\n".join(
                        f"  [hook] {h['command']}: {h['output']}" for h in pre_blocked
                    )
                    observation = f"tool blocked by PreToolUse hook:\n{blocked_msgs}"
                    emit(f"observation: {observation}")
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": observation,
                        "is_error": True,
                    })
                    continue

            # ── 执行工具 ──────────────────────────────────────
            result = tools.run(call, ctx)
            emit(f"observation: {result.content}")

            # ── PostToolUse hooks ────────────────────────────
            if not result.is_error:
                post_hooks = run_hooks(
                    "PostToolUse", call.name, call.arguments, ctx.cwd,
                    tool_result=result.content,
                )
                for h in post_hooks:
                    status = "ok" if h["success"] else f"warning: {h['output']}"
                    console.print(f"[dim]hook: PostToolUse {call.name} {status}[/dim]")
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
        messages.append({"role": "user", "content": tool_result_blocks})
        if session:
            session.append_messages([messages[-1]])

    final = f"reached max_steps={max_steps}"
    emit(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)
