from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

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
from .model import ModelProvider, ModelResponse, StreamEvent, ToolCall, ToolResult
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
from .tools import ToolContext, ToolRegistry


_SYSTEM_CORE = """\
You are bemoCode, a personal AI coding agent built by and for "123".
You live inside a terminal. Your user is a developer who prefers
Chinese communication and JS/TS mental models.

## Personality
- 用中文回复，语气像一位靠谱的同事实习生——不卑不亢、不啰嗦
- 遇到不确定的事主动说，不要猜
- 代码改动前先读文件，改完自己检查一遍
- 错误时解释原因，不要只甩一句 error

## Tool Strategy
- 先 project_tree / list_files 了解结构，再 glob / grep 定位，最后 read_file 细读
- 只读工具可以并发调用，节省轮次
- 编辑前必须先 read_file（系统强制）
- file_edit 的 old_string 必须和文件内容精确匹配，包含完整缩进
- bash 命令尽量原子化：一条命令做完一件事，不要连环 pipe 十层
- 搜网页时先 web_search 找结果，再用 web_fetch 抓具体页面

## Memory
- 用户明确告诉你"记住 xxx"或发现重要偏好时，用 memory_write 记下来
- 新会话开始时先 memory_recall 查一下已有的用户偏好
- 不要在每轮对话后都写记忆——只在有意义的时候写

## Output
- 代码块用 ``` 标注语言
- diff/改动说明用 markdown 渲染
- 如果任务完成了，用 /todo 更新状态
- 复杂任务先 enter_plan_mode 出方案，用户点头了再写代码

## Safety
- 永远不要删用户没说要删的文件
- 永远不要 git push 除非用户明确让你 push
- bash 输出太长用 head/tail 截断，不要无脑全量返回
- 记住你是在终端里，别输出几十页内容淹了屏幕\
"""


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


def _partition_tool_calls(calls: list[ToolCall], tools: ToolRegistry) -> list[list[ToolCall]]:
    """把连续只读工具合并为并行 batch，写工具各自独立串行。"""
    batches: list[list[ToolCall]] = []
    current: list[ToolCall] = []
    for call in calls:
        tool = tools.get(call.name)
        is_ro = tool is not None and tool.is_read_only
        if is_ro:
            current.append(call)
        else:
            if current:
                batches.append(current)
                current = []
            batches.append([call])
    if current:
        batches.append(current)
    return batches


def _display_streaming_text(
    provider: ModelProvider,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    system: str,
    console: Console,
    text: str,
) -> None:
    """如果 provider 支持流式，逐词显示文本（视觉动效）；否则直接打印。"""
    if not text:
        return
    try:
        events = provider.complete_stream(messages, tools=tools.list(), system=system)
    except Exception:
        # 不支持流式，直接渲染 markdown
        console.print(Markdown(text, code_theme="monokai"))
        return

    words = text.split()
    if not words:
        console.print(Markdown(text, code_theme="monokai"))
        return

    # 用 Live 逐词显示
    displayed: list[str] = []
    md = Markdown("", code_theme="monokai")
    with Live(md, console=console, refresh_per_second=15, transient=False) as live:
        for word in words:
            displayed.append(word)
            md = Markdown(" ".join(displayed), code_theme="monokai")
            live.update(md)
        # 最终完整显示
        md = Markdown(text, code_theme="monokai")
        live.update(md)


def _execute_one_tool(
    call: ToolCall,
    ctx: ToolContext,
    permission_mode: str,
    tools: ToolRegistry,
    console: Console,
    emit,
) -> tuple[ToolResult, dict[str, Any] | None, str]:
    """执行单个工具调用，含权限检查和安全验证。
    返回 (result, tool_result_block, new_permission_mode)。
    """

    # ── Plan mode: enter_plan_mode / exit_plan_mode ──────────
    if call.name == "enter_plan_mode":
        result = ToolResult(call.id, "Plan mode on. Write tools denied.", is_error=False)
        return result, _trb(result), "plan"

    if call.name == "exit_plan_mode":
        plan_summary = call.arguments.get("plan_summary", "")
        if not confirm_plan(plan_summary):
            result = ToolResult(call.id, "Plan not approved. Revise.", is_error=True)
            return result, _trb(result), permission_mode
        result = ToolResult(call.id, "Plan approved. Write tools enabled.", is_error=False)
        return result, _trb(result), "acceptEdits"

    # ── Permission gate ──────────────────────────────────────
    request = PermissionRequest(
        tool_name=call.name,
        args=call.arguments,
        mode=permission_mode,
        cwd=ctx.cwd,
    )
    decision = decide_permission(request)

    # ── Edit preview ─────────────────────────────────────────
    edit_preview: tuple | None = None
    if call.name in ("file_write", "file_edit") and decision.behavior != "deny":
        path_str = call.arguments.get("file_path", "")
        if not path_str:
            return ToolResult(call.id, "error: missing 'file_path'", is_error=True), _trb(ToolResult(call.id, "error: missing 'file_path'", is_error=True)), permission_mode

        try:
            path = resolve_in_cwd(ctx.cwd, path_str)
        except (ValueError, OSError) as exc:
            return ToolResult(call.id, f"error: {exc}", is_error=True), _trb(ToolResult(call.id, f"error: {exc}", is_error=True)), permission_mode

        old_content = path.read_text(encoding="utf-8") if path.exists() else ""

        validation_error: str | None = None
        if call.name == "file_write":
            if path.exists():
                validation_error = ensure_read_before_edit(ctx.read_state, path) or check_mtime_conflict(ctx.read_state, path)
        else:
            if not path.exists():
                validation_error = f"error: file does not exist: {path_str}"
            else:
                validation_error = ensure_read_before_edit(ctx.read_state, path) or check_mtime_conflict(ctx.read_state, path)

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
            if replace_err:
                validation_error = replace_err

        if validation_error:
            return ToolResult(call.id, validation_error, is_error=True), _trb(ToolResult(call.id, validation_error, is_error=True)), permission_mode

        edit_preview = (path_str, old_content, new_content or "")

    # ── Decision branches ────────────────────────────────────
    if decision.behavior == "deny":
        return ToolResult(call.id, f"error: {decision.message}", is_error=True), _trb(ToolResult(call.id, f"error: {decision.message}", is_error=True)), permission_mode

    if decision.behavior == "ask":
        if call.name in ("file_write", "file_edit") and edit_preview is not None:
            path_str, old_content, new_content = edit_preview
            diff_text = render_diff(old_content, new_content, path_str)
            console.print(f"\n[bold]Diff for {path_str}:[/bold]")
            console.print(diff_text)
            if not confirm_edit(path_str):
                return ToolResult(call.id, "error: edit rejected by user", is_error=True), _trb(ToolResult(call.id, "error: edit rejected by user", is_error=True)), permission_mode

        elif call.name == "bash":
            command = call.arguments.get("command", "")
            timeout = call.arguments.get("timeout", 30)
            console.print(f"\n[bold]Command:[/bold] {command}")
            console.print(f"[dim]timeout: {timeout}s  cwd: {ctx.cwd}[/dim]")
            if not confirm_command(command):
                return ToolResult(call.id, "error: command rejected by user", is_error=True), _trb(ToolResult(call.id, "error: command rejected by user", is_error=True)), permission_mode

        elif call.name in ("web_fetch", "web_search"):
            if not confirm_tool_use(call.name, call.arguments):
                return ToolResult(call.id, "error: tool use rejected by user", is_error=True), _trb(ToolResult(call.id, "error: tool use rejected by user", is_error=True)), permission_mode

        elif call.name == "ask_user_question":
            question = call.arguments.get("prompt", "")
            options = call.arguments.get("options", [])
            if not isinstance(options, list):
                options = []
            labels = [str(o) for o in options]
            selected = prompt_single_choice(question, labels)
            result = ToolResult(call.id, f'User selected: "{selected}"' if selected else "User skipped.", is_error=False)
            return result, _trb(result), permission_mode

    # ── PreToolUse hooks ─────────────────────────────────────
    if decision.behavior != "deny":
        pre_hooks = run_hooks("PreToolUse", call.name, call.arguments, ctx.cwd)
        pre_blocked = [h for h in pre_hooks if not h["success"]]
        if pre_blocked:
            blocked_msgs = "\n".join(f"  [hook] {h['command']}: {h['output']}" for h in pre_blocked)
            result = ToolResult(call.id, f"tool blocked by PreToolUse hook:\n{blocked_msgs}", is_error=True)
            return result, _trb(result), permission_mode

    # ── Execute ──────────────────────────────────────────────
    result = tools.run(call, ctx)
    emit(f"observation: {result.content}")

    # ── PostToolUse hooks ────────────────────────────────────
    if not result.is_error:
        post_hooks = run_hooks("PostToolUse", call.name, call.arguments, ctx.cwd, tool_result=result.content)
        for h in post_hooks:
            status = "ok" if h["success"] else f"warning: {h['output']}"
            console.print(f"[dim]hook: PostToolUse {call.name} {status}[/dim]")

    return result, _trb(result), permission_mode


def _trb(result: ToolResult) -> dict[str, Any]:
    """构造 tool_result block 字典。"""
    return {
        "type": "tool_result",
        "tool_use_id": result.tool_call_id,
        "content": result.content,
        "is_error": result.is_error,
    }


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

    system = system_prompt or build_system_prompt(resolved_cwd)

    if session and session.history:
        messages = list(session.history)
        messages.append({"role": "user", "content": prompt})
    else:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    if session:
        session.append_messages([messages[-1]])

    for step in range(max_steps):
        if len(messages) > 40:
            messages = _compact_messages(messages, keep=8)
            console.print(f"[dim]compacted: {len(messages)} messages remaining[/dim]")

        # ── Model call ────────────────────────────────────────
        response = provider.complete(messages, tools=tools.list(), system=system)
        messages.append(_assistant_message(response))
        if session:
            session.append_messages([messages[-1]])

        # ── Streaming text display ─────────────────────────────
        if response.text and not response.tool_calls:
            console.print()
            _sep(console, "Agent", style="bold")
            console.print()
            _display_streaming_text(provider, messages, tools, system, console, response.text)
            console.print()
            return AgentResult(final=response.text, trace=trace, messages=messages)

        if not response.tool_calls:
            final = response.text or ""
            return AgentResult(final=final, trace=trace, messages=messages)

        # ── Agent thinking text before tool calls ──────────────
        if response.text:
            console.print()
            _sep(console, "Agent", style="bold")
            console.print()
            if response.text:
                _display_streaming_text(provider, messages, tools, system, console, response.text)

        # ── Parallel/serial execution ─────────────────────────
        batches = _partition_tool_calls(response.tool_calls or [], tools)
        tool_result_blocks: list[dict[str, Any]] = []

        for batch in batches:
            if len(batch) == 1:
                # Serial execution
                call = batch[0]
                emit(f"tool_call: {call.name} {call.arguments}")
                result, block, new_mode = _execute_one_tool(call, ctx, permission_mode, tools, console, emit)
                permission_mode = new_mode
                tool_result_blocks.append(block)
            else:
                # Parallel execution for read-only batch
                tool_names = ", ".join(c.name for c in batch)
                emit(f"tool_call: [{tool_names}] (parallel)")

                with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as ex:
                    futures = {
                        ex.submit(_execute_one_tool, call, ctx, permission_mode, tools, console, emit): call
                        for call in batch
                    }
                    for future in futures:
                        call = futures[future]
                        result, block, new_mode = future.result()
                        if new_mode != permission_mode:
                            permission_mode = new_mode
                        tool_result_blocks.append(block)

        messages.append({"role": "user", "content": tool_result_blocks})
        if session:
            session.append_messages([messages[-1]])

    final = f"reached max_steps={max_steps}"
    emit(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)
