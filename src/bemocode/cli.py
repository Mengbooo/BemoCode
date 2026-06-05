from __future__ import annotations
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from .agent import build_system_prompt, run_agent
from .model import create_provider
from .session import Session
from .slash import SlashContext, dispatch_slash
from .tools import default_tools

load_dotenv()

console = Console()
app = typer.Typer(add_completion=False)
tools = default_tools()


def render_header(
    cwd: Path,
    provider: str,
    model: str,
    base_url: str | None = None,
    session: Session | None = None,
) -> None:
    console.print("[bold]bemoCode[/bold]")
    console.print(f"[dim]cwd: {cwd}[/dim]")
    console.print(f"[dim]provider: {provider}  model: {model}[/dim]")
    if base_url:
        console.print(f"[dim]base_url: {base_url}[/dim]")
    if session:
        tag = "(resumed)" if session.resumed else ""
        console.print(f"[dim]session: {session.session_id} {tag}[/dim]")
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
    permission_mode: str = "default",
    session: Session | None = None,
    system_prompt: str | None = None,
) -> None:
    render_header(cwd, provider, model, base_url, session)
    os.chdir(cwd)
    prov = create_provider(provider, model, base_url)
    run_agent(
        prompt, prov, tools,
        max_steps=max_steps,
        cwd=cwd,
        permission_mode=permission_mode,
        session=session,
        system_prompt=system_prompt,
    )


@app.callback(invoke_without_command=True)
def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-C"),
    provider: str = typer.Option("anthropic", "--provider"),
    model: str = typer.Option("deepseek-v4-flash", "--model"),
    base_url: str | None = typer.Option(None, "--base-url"),
    max_steps: int = typer.Option(8, "--max-steps"),
    permission_mode: str = typer.Option(
        "default", "--permission-mode",
        help="Permission mode: default, acceptEdits, plan",
    ),
    resume: str | None = typer.Option(None, "--resume", help="Resume a specific session by ID"),
    continue_: bool = typer.Option(False, "--continue", "-c", help="Resume the latest session"),
) -> None:
    resolved_cwd = cwd.resolve()
    text = prompt.strip()

    # Determine session
    session: Session | None = None
    if resume:
        session = Session.load_by_id(resolved_cwd, resume)
        if session is None:
            console.print(f"[red]Session not found: {resume}[/red]")
            raise typer.Exit(code=1)
    elif continue_:
        session = Session.load_latest(resolved_cwd)
        if session is None:
            console.print("[yellow]No previous session found; starting a new one.[/yellow]")

    system_prompt = build_system_prompt(resolved_cwd)

    if text:
        run_once(text, resolved_cwd, provider, model, base_url, max_steps,
                 permission_mode, session=session, system_prompt=system_prompt)
        return

    render_header(resolved_cwd, provider, model, base_url, session)
    console.print("输入 /help 查看命令，输入 /exit 退出。")
    while True:
        line = typer.prompt(">").strip()
        if not line:
            continue
        if line == "/exit":
            console.print("Bye.")
            return
        if line.startswith("/"):
            result = dispatch_slash(line, SlashContext(
                cwd=resolved_cwd,
                permission_mode=permission_mode,
                model=model,
                provider=provider,
                session_id=session.session_id if session else None,
            ))
            if result.handled:
                if result.message:
                    console.print(result.message)
                if result.should_query:
                    if session is None:
                        session = Session.create(resolved_cwd)
                    run_once(result.prompt, resolved_cwd, provider, model, base_url,
                             max_steps, permission_mode, session=session,
                             system_prompt=system_prompt)
                continue
        if session is None:
            session = Session.create(resolved_cwd)
        run_once(line, resolved_cwd, provider, model, base_url, max_steps,
                 permission_mode, session=session, system_prompt=system_prompt)


def main() -> None:
    app()
