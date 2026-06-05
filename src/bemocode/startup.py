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

    logo = Text()
    lines = _LOGO.strip("\n").split("\n")
    for i, line in enumerate(lines):
        logo.append(line + "\n", style="bold")

    info = Text()
    info.append(f"  cwd       ", style="dim")
    info.append(f"{cwd}\n", style="white")
    info.append(f"  provider  ", style="dim")
    info.append(f"{provider}", style="bold")
    info.append(f"    model  ", style="dim")
    info.append(f"{model}", style="bold")
    info.append(f"    mode  ", style="dim")
    info.append(f"[{permission_mode}]", style="bold")
    if base_url:
        info.append(f"\n  base_url  ", style="dim")
        info.append(f"{base_url}", style="white")
    info.append(f"\n  tools     ", style="dim")
    info.append(f"{tool_count} registered", style="bold")

    if session:
        tag = " (resumed)" if session.resumed else ""
        info.append(f"\n  session   ", style="dim")
        info.append(f"{session.session_id}{tag}", style="bold")

    tagline = Text()
    tagline.append("  ⚡  ", style="white")
    tagline.append("read", style="bold")
    tagline.append(" · ", style="dim")
    tagline.append("write", style="bold")
    tagline.append(" · ", style="dim")
    tagline.append("run", style="bold")
    tagline.append(" · ", style="dim")
    tagline.append("remember", style="bold")
    tagline.append("\n")

    banner = Text()
    banner.append(logo)
    banner.append("\n")
    banner.append(tagline)
    banner.append("\n")
    banner.append(info)
    banner.append("\n")

    console.print()
    console.print(banner)
