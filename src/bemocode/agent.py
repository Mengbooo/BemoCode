from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

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
from .prompt_ui import (
    confirm_command,
    confirm_edit,
    confirm_tool_use,
    prompt_single_choice,
    render_diff,
)
from .tools import ToolContext, ToolRegistry, ToolResult


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


def run_agent(
    prompt: str,
    provider: ModelProvider,
    tools: ToolRegistry,
    max_steps: int = 8,
    cwd: Path | None = None,
    permission_mode: str = "default",
) -> AgentResult:
    resolved_cwd = cwd or Path.cwd()
    ctx = ToolContext(
        cwd=resolved_cwd,
        skip_policy=SkipPolicy.default(gitignore=load_gitignore(resolved_cwd)),
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    trace: list[str] = []
    console = Console()

    def emit(line: str) -> None:
        trace.append(line)
        console.print(line)

    for step in range(max_steps):
        response = provider.complete(messages, tools=tools.list())
        messages.append(_assistant_message(response))

        if not response.tool_calls:
            final = response.text or ""
            emit(f"final: {final}")
            return AgentResult(final=final, trace=trace, messages=messages)

        tool_result_blocks: list[dict[str, Any]] = []
        for call in response.tool_calls:
            emit(f"tool_call: {call.name} {call.arguments}")

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

            # ── 执行工具 ──────────────────────────────────────
            result = tools.run(call, ctx)
            emit(f"observation: {result.content}")
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
        messages.append({"role": "user", "content": tool_result_blocks})

    final = f"reached max_steps={max_steps}"
    emit(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)
