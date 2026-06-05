from __future__ import annotations

import difflib

import typer
from rich.console import Console
from rich.syntax import Syntax

console = Console()


def render_diff(old_text: str, new_text: str, file_path: str) -> str:
    """用 difflib 生成 unified diff，给增删行加 rich markup 着色。"""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
    )
    colored: list[str] = []
    for line in diff_lines:
        line = line.rstrip()
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"[bold]{line}[/bold]")
        elif line.startswith("-"):
            colored.append(f"[red]{line}[/red]")
        elif line.startswith("+"):
            colored.append(f"[green]{line}[/green]")
        elif line.startswith("@@"):
            colored.append(f"[cyan]{line}[/cyan]")
        else:
            colored.append(line)
    return "\n".join(colored)


def confirm_edit(file_path: str) -> bool:
    """让用户确认是否应用这次编辑，默认不应用。"""
    return typer.confirm(f"Apply this edit to {file_path}?", default=False)


def confirm_command(command: str) -> bool:
    """Ask user to confirm running this bash command; defaults to no. 只展示前 120 字符。"""
    preview = command if len(command) <= 120 else command[:120] + "..."
    return typer.confirm(f"Run this command?\n  {preview}", default=False)


def confirm_tool_use(tool_name: str, args: dict) -> bool:
    """Ask user to confirm using a tool that accesses external resources."""
    summary = _tool_summary(tool_name, args)
    return typer.confirm(f"Allow {tool_name}?\n  {summary}", default=False)


def prompt_single_choice(question: str, labels: list[str]) -> str | None:
    """Display a numbered menu for single selection. Returns chosen label or None."""
    console.print(f"\n[bold yellow]? {question}[/bold yellow]")
    for i, label in enumerate(labels, 1):
        console.print(f"  {i}. {label}")
    console.print(f"  0. [dim]Skip / Other[/dim]")

    try:
        choice = typer.prompt("Choice", default="0")
        idx = int(choice)
        if 1 <= idx <= len(labels):
            return labels[idx - 1]
        return None
    except (ValueError, TypeError):
        return None


def confirm_plan(plan_summary: str) -> bool:
    """显示计划摘要，请求用户审批。"""
    from io import StringIO

    from rich.panel import Panel

    buffer = StringIO()
    c = Console(file=buffer, no_color=True)
    c.print(Panel(plan_summary or "(empty plan)", title="Plan", border_style="blue"))
    panel = buffer.getvalue()
    typer.echo(panel, nl=False)
    return typer.confirm("Approve this plan and exit plan mode?", default=False)


def _tool_summary(tool_name: str, args: dict) -> str:
    """Generate a short human-readable summary of a tool call."""
    if tool_name == "web_fetch":
        return f"Fetch URL: {args.get('url', '(none)')}"
    if tool_name == "web_search":
        return f"Search for: {args.get('query', '(none)')}"
    return str(args)
