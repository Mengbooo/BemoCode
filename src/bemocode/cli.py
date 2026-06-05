from __future__ import annotations
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from .agent import build_system_prompt, run_agent
from .interactive import start_interactive_shell
from .model import create_provider
from .runtime import RuntimeState
from .session import Session
from .slash import SlashContext, dispatch_slash
from .startup import show_banner
from .tools import default_tools

load_dotenv()

console = Console()
app = typer.Typer(add_completion=False)
tools = default_tools()


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

    # Runtime state shared between CLI and agent
    state = RuntimeState(
        permission_mode=permission_mode,
        model=model,
        provider=provider,
    )

    if text:
        # One-shot mode
        show_banner(resolved_cwd, provider, model, base_url, session,
                    permission_mode, tool_count=len(tools.list_tools()))
        run_once(text, resolved_cwd, provider, model, base_url, max_steps,
                 permission_mode, session=session, system_prompt=system_prompt)
        return

    # REPL mode with TUI
    def handle_input(user_input: str) -> None:
        nonlocal session
        if session is None:
            session = Session.create(resolved_cwd)
        run_once(user_input, resolved_cwd, provider, state.model, base_url, max_steps,
                 state.permission_mode, session=session, system_prompt=system_prompt)

    start_interactive_shell(
        cwd=resolved_cwd,
        provider=provider,
        model=model,
        base_url=base_url,
        max_steps=max_steps,
        permission_mode=permission_mode,
        session=session,
        system_prompt=system_prompt,
        on_input=handle_input,
        tools=tools,
        state=state,
    )


def main() -> None:
    app()
