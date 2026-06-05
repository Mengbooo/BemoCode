from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.text import Text

from .session import Session

console = Console()

_LOGO = r"""
██████╗  ███████╗ ███╗   ███╗ █████╗   ██████╗  █████╗  ██████╗  ███████╗
██╔══██╗ ██╔════╝ ████╗ ████║██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔════╝
██████╔╝ █████╗   ██╔████╔██║██║  ██║ ██║      ██║  ██║ ██║  ██║ █████╗
██╔══██╗ ██╔══╝   ██║╚██╔╝██║██║  ██║ ██║      ██║  ██║ ██║  ██║ ██╔══╝
██████╔╝ ███████╗ ██║ ╚═╝ ██║╚█████╔╝ ╚██████╗ ╚█████╔╝ ██████╔╝ ███████╗
╚═════╝  ╚══════╝ ╚═╝     ╚═╝ ╚════╝   ╚═════╝  ╚════╝  ╚═════╝  ╚══════╝
"""


def show_banner(
    cwd: Path,
    provider: str = "anthropic",
    model: str = "deepseek-v4-flash",
    base_url: str | None = None,
    session: Session | None = None,
    permission_mode: str = "default",
    tool_count: int = 0,
) -> None:
    """Render the bemocode startup banner."""

    # Logo with gradient — cyan at top fading to blue
    logo = Text()
    lines = _LOGO.strip("\n").split("\n")
    for i, line in enumerate(lines):
        ratio = i / max(len(lines) - 1, 1)
        if ratio < 0.4:
            style = "bold cyan"
        elif ratio < 0.7:
            style = "bold bright_blue"
        else:
            style = "bold blue"
        logo.append(line + "\n", style=style)

    # Build info lines
    mode_colors = {"default": "green", "acceptEdits": "yellow", "plan": "bright_blue"}
    mode_color = mode_colors.get(permission_mode, "white")

    info = Text()
    info.append(f"  cwd       ", style="dim")
    info.append(f"{cwd}\n", style="white")
    info.append(f"  provider  ", style="dim")
    info.append(f"{provider}", style="bright_cyan")
    info.append(f"    model  ", style="dim")
    info.append(f"{model}", style="bright_cyan")
    info.append(f"    mode  ", style="dim")
    info.append(f"[{permission_mode}]", style=f"bold {mode_color}")
    if base_url:
        info.append(f"\n  base_url  ", style="dim")
        info.append(f"{base_url}", style="white")
    info.append(f"\n  tools     ", style="dim")
    info.append(f"{tool_count} registered", style="bright_green")

    if session:
        tag = " (resumed)" if session.resumed else ""
        info.append(f"\n  session   ", style="dim")
        info.append(f"{session.session_id}{tag}", style="bright_magenta")

    # Tagline
    tagline = Text()
    tagline.append("  ⚡  ", style="yellow")
    tagline.append("read", style="bold green")
    tagline.append(" · ", style="dim")
    tagline.append("write", style="bold yellow")
    tagline.append(" · ", style="dim")
    tagline.append("run", style="bold red")
    tagline.append(" · ", style="dim")
    tagline.append("remember", style="bold magenta")
    tagline.append("\n")

    # Assemble the full banner
    banner = Text()
    banner.append(logo)
    banner.append("\n")
    banner.append(tagline)
    banner.append("\n")
    banner.append(info)
    banner.append("\n")

    console.print()
    console.print(banner)
    console.print("─" * 78, style="dim")
