from __future__ import annotations

from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from .runtime import RuntimeState
from .session import Session
from .slash import SlashContext, dispatch_slash
from .startup import show_banner
from .tools import ToolRegistry

_STYLE = Style.from_dict({
    "bottom-toolbar": "reverse",
    "bottom-toolbar.label": "bold",
    "bottom-toolbar.mode-default": "bold",
    "bottom-toolbar.mode-plan": "bold",
    "bottom-toolbar.mode-accept": "bold",
})


def _mode_style(mode: str) -> str:
    if mode == "plan":
        return "bottom-toolbar.mode-plan"
    if mode == "acceptEdits":
        return "bottom-toolbar.mode-accept"
    return "bottom-toolbar.mode-default"


def _status_bar(state: RuntimeState) -> str:
    """生成底部状态栏。"""
    mode = state.permission_mode
    active = next(
        (t.active_form for t in state.todo_store if t.status == "in_progress"),
        "",
    )
    todo = f" · {active}" if active else ""
    return (
        f"  [{mode}] · {state.model}{todo}  "
        f"  Shift+Tab: cycle mode  |  /help  |  /exit  "
    )


def _simple_repl(
    cwd: Path, provider: str, model: str, base_url: str | None,
    max_steps: int, permission_mode: str, session: Session | None,
    system_prompt: str, on_input, tools: ToolRegistry, state: RuntimeState,
) -> None:
    """非 TTY 环境下的简化 REPL，直接用 input()。"""
    from rich.console import Console
    from rich.text import Text
    from .slash import SlashContext, dispatch_slash
    from .startup import show_banner

    console = Console()
    show_banner(cwd, provider, model, base_url, session,
                permission_mode, tool_count=len(tools.list_tools()))

    while True:
        try:
            console.print()
            width = console.width or 78
            line_text = Text()
            line_text.append("───", style="bold")
            line_text.append(" You ", style="reverse bold")
            line_text.append("─" * max(0, width - 8), style="dim")
            console.print(line_text)
            user_input = input("  ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            return

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye.")
            return

        if user_input.startswith("/"):
            result = dispatch_slash(user_input, SlashContext(
                cwd=cwd,
                permission_mode=state.permission_mode,
                model=state.model,
                provider=state.provider,
                session_id=session.session_id if session else None,
            ))
            if result.handled:
                if result.message:
                    console.print(result.message)
                if result.should_query:
                    on_input(result.prompt)
                continue

        on_input(user_input)


def start_interactive_shell(
    cwd: Path,
    provider: str,
    model: str,
    base_url: str | None,
    max_steps: int,
    permission_mode: str,
    session: Session | None,
    system_prompt: str,
    on_input: Callable[[str], None],
    tools: ToolRegistry,
    state: RuntimeState,
) -> None:
    """启动 prompt_toolkit 交互式 shell。"""

    import sys
    if not sys.stdin.isatty():
        _simple_repl(cwd, provider, model, base_url, max_steps, permission_mode,
                     session, system_prompt, on_input, tools, state)
        return

    kb = KeyBindings()

    @kb.add("s-tab")
    def _(event: Any) -> None:
        """Shift+Tab 循环切换权限模式。"""
        order = ["default", "acceptEdits", "plan"]
        idx = order.index(state.permission_mode) if state.permission_mode in order else 0
        state.permission_mode = order[(idx + 1) % len(order)]
        print(f"\r[mode → {state.permission_mode}]")

    def _get_bottom_toolbar() -> str:
        return _status_bar(state)

    # 历史文件存到 .agent/
    history_dir = cwd / ".agent"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "input_history"

    session_obj = PromptSession(
        history=FileHistory(str(history_file)),
        key_bindings=kb,
        style=_STYLE,
        bottom_toolbar=_get_bottom_toolbar,
    )

    # 显示 banner + 第一条 You 分割线
    show_banner(cwd, provider, model, base_url, session,
                permission_mode, tool_count=len(tools.list_tools()))

    while True:
        try:
            # 用户分割线
            from rich.console import Console
            from rich.text import Text
            console = Console()
            width = console.width or 78
            line = Text()
            line.append("───", style="bold")
            line.append(" You ", style="reverse bold")
            line.append("─" * max(0, width - 8), style="dim")
            console.print()
            console.print(line)

            text = session_obj.prompt("  ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            return

        if not text:
            continue
        if text == "/exit":
            print("Bye.")
            return

        if text.startswith("/"):
            result = dispatch_slash(text, SlashContext(
                cwd=cwd,
                permission_mode=state.permission_mode,
                model=state.model,
                provider=state.provider,
                session_id=session.session_id if session else None,
            ))
            if result.handled:
                if result.message:
                    console.print(result.message)
                if result.should_query:
                    on_input(result.prompt)
                continue

        on_input(text)
