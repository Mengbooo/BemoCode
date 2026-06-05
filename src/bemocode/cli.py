from __future__ import annotations
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from .agent import run_agent
from .model import create_provider
from .tools import default_tools

load_dotenv()

console = Console()
app = typer.Typer(add_completion=False)
tools = default_tools()


def render_header(cwd: Path, provider: str, model: str, base_url: str | None = None) -> None:
    console.print("[bold]bemoCode[/bold]")
    console.print(f"[dim]cwd: {cwd}[/dim]")
    console.print(f"[dim]provider: {provider}  model: {model}[/dim]")
    if base_url:
        console.print(f"[dim]base_url: {base_url}[/dim]")
    console.print()


def handle_slash(line: str) -> bool:
    if line == "/help":
        console.print("可用命令：[bold]/help[/bold], [bold]/exit[/bold]")
        console.print("\n[bold]注册的工具:[/bold]")
        for tool in tools.list_tools():
            console.print(f"  [cyan]{tool.name}[/cyan] — {tool.description}")
        return True
    return False


def run_once(
    prompt: str,
    cwd: Path,
    provider: str,
    model: str,
    base_url: str | None,
    max_steps: int,
) -> None:
    render_header(cwd, provider, model, base_url)
    os.chdir(cwd)
    prov = create_provider(provider, model, base_url)
    result = run_agent(prompt, prov, tools, max_steps=max_steps, cwd=cwd)
    for line in result.trace:
        console.print(line)


@app.callback(invoke_without_command=True)
def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-C"),
    provider: str = typer.Option("anthropic", "--provider"),
    model: str = typer.Option("deepseek-v4-flash", "--model"),
    base_url: str | None = typer.Option(None, "--base-url"),
    max_steps: int = typer.Option(8, "--max-steps"),
) -> None:
    resolved_cwd = cwd.resolve()
    text = prompt.strip()
    if text:
        run_once(text, resolved_cwd, provider, model, base_url, max_steps)
        return
    render_header(resolved_cwd, provider, model, base_url)
    console.print("输入 /help 查看命令，输入 /exit 退出。")
    while True:
        line = typer.prompt(">").strip()
        if not line:
            continue
        if line == "/exit":
            console.print("Bye.")
            return
        if line.startswith("/") and handle_slash(line):
            continue
        run_once(line, resolved_cwd, provider, model, base_url, max_steps)


def main() -> None:
    app()
