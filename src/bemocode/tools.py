from __future__ import annotations
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import html2text
import httpx

from .fs_safety import (
    DEFAULT_MAX_CHARS,
    ReadFileState,
    SkipPolicy,
    ensure_text_file,
    ensure_within_size,
    resolve_in_cwd,
    should_skip,
    truncate_output,
)
from .model import ToolCall, ToolResult


@dataclass
class ToolContext:
    """工具运行时上下文。Day 3 装 cwd、skip 规则、ReadFileState。"""
    cwd: Path
    skip_policy: SkipPolicy = field(default_factory=SkipPolicy.default)
    read_state: ReadFileState = field(default_factory=ReadFileState)


ToolFunc = Callable[[dict[str, Any], ToolContext], str]


@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


# ── 基础工具 ──────────────────────────────────────────────

def echo(args: dict[str, Any], ctx: ToolContext) -> str:
    return str(args.get("text", ""))


def system_date(args: dict[str, Any], ctx: ToolContext) -> str:
    return datetime.now().isoformat()


def uppercase(args: dict[str, Any], ctx: ToolContext) -> str:
    return str(args.get("text", "")).upper()


# ── 文件工具 ──────────────────────────────────────────────

def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path_str = args.get("path", "")
    if not path_str:
        return "error: missing required argument 'path'"
    try:
        path = resolve_in_cwd(ctx.cwd, path_str)
        ensure_text_file(path)
        ensure_within_size(path)
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        return f"error: {exc}"
    ctx.read_state.record(path, text)
    return truncate_output(text)


def list_files(args: dict[str, Any], ctx: ToolContext) -> str:
    path_str = args.get("path", ".")
    try:
        base = resolve_in_cwd(ctx.cwd, path_str)
    except ValueError as exc:
        return f"error: {exc}"
    if not base.is_dir():
        return f"error: not a directory: {path_str}"
    entries: list[str] = []
    for child in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        rel = child.relative_to(ctx.cwd)
        if should_skip(rel, ctx.skip_policy):
            continue
        entries.append(f"{child.name}/" if child.is_dir() else child.name)
    return truncate_output("\n".join(entries) or "(empty)")


def glob(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "error: missing required argument 'pattern'"
    matches: list[Path] = []
    for path in ctx.cwd.rglob(pattern):
        rel = path.relative_to(ctx.cwd)
        if should_skip(rel, ctx.skip_policy):
            continue
        matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    matches = matches[:200]
    lines = [str(p.relative_to(ctx.cwd)) for p in matches]
    return truncate_output("\n".join(lines) or "(no matches)")


def _grep_ripgrep(
    pattern: str, base: Path, glob_arg: str | None,
    ignore_case: bool, ctx: ToolContext,
) -> str:
    cmd = ["rg", "--line-number", "--no-heading"]
    if ignore_case:
        cmd.append("--ignore-case")
    for d in ctx.skip_policy.skip_dirs:
        cmd.extend(["--glob", f"!{d}/**"])
    if ctx.skip_policy.gitignore is not None:
        cmd.append("--no-ignore")
    if glob_arg:
        cmd.extend(["--glob", glob_arg])
    cmd.append(pattern)
    cmd.append(str(base))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _grep_python(pattern, base, glob_arg, ignore_case, ctx)
    if result.returncode not in (0, 1):
        return f"error: rg failed: {result.stderr.strip()}"
    prefix = str(ctx.cwd) + "/"
    lines = result.stdout.splitlines()
    rel_lines = [
        line[len(prefix):] if line.startswith(prefix) else line
        for line in lines
    ]
    return truncate_output("\n".join(rel_lines) or "(no matches)")


def _grep_python(
    pattern: str, base: Path, glob_arg: str | None,
    ignore_case: bool, ctx: ToolContext,
) -> str:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"error: invalid regex: {exc}"
    results: list[str] = []
    for path in base.rglob("*"):
        rel = path.relative_to(ctx.cwd)
        if should_skip(rel, ctx.skip_policy):
            continue
        if glob_arg and not path.match(glob_arg):
            continue
        if not path.is_file():
            continue
        try:
            ensure_text_file(path)
        except ValueError:
            continue
        try:
            for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    results.append(f"{rel}:{lineno}:{line.strip()[:200]}")
        except Exception:
            continue
    return truncate_output("\n".join(results) or "(no matches)")


def grep(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "error: missing required argument 'pattern'"
    path_arg = args.get("path", ".")
    glob_arg = args.get("glob")
    ignore_case = bool(args.get("ignore_case", False))
    try:
        base = resolve_in_cwd(ctx.cwd, path_arg)
    except ValueError as exc:
        return f"error: {exc}"
    if shutil.which("rg"):
        return _grep_ripgrep(pattern, base, glob_arg, ignore_case, ctx)
    return _grep_python(pattern, base, glob_arg, ignore_case, ctx)


# ── 项目树工具 ────────────────────────────────────────────

def project_tree(args: dict[str, Any], ctx: ToolContext) -> str:
    max_depth = int(args.get("max_depth", 3))
    max_nodes = 200
    lines: list[str] = [f"{ctx.cwd.name}/"]
    nodes = 0

    def walk(directory: Path, depth: int) -> None:
        nonlocal nodes
        if depth > max_depth:
            return
        children = sorted(
            (
                c for c in directory.iterdir()
                if not should_skip(c.relative_to(ctx.cwd), ctx.skip_policy)
            ),
            key=lambda p: (not p.is_dir(), p.name),
        )
        for child in children:
            if nodes >= max_nodes:
                if nodes == max_nodes:
                    lines.append("  " * depth + "...[truncated]")
                    nodes += 1
                return
            suffix = "/" if child.is_dir() else ""
            lines.append("  " * depth + child.name + suffix)
            nodes += 1
            if child.is_dir():
                walk(child, depth + 1)

    walk(ctx.cwd, 1)
    return truncate_output("\n".join(lines))


# ── Web 工具 ──────────────────────────────────────────────

WEB_USER_AGENT = "agent-code/0.1 (+https://example.com/agent-code)"
WEB_FETCH_MAX_BYTES = 10 * 1024 * 1024
WEB_FETCH_MAX_CHARS = 20_000
WEB_URL_MAX_LENGTH = 2000
WEB_FETCH_TIMEOUT_S = 30.0
WEB_SEARCH_TIMEOUT_S = 15.0


def _validate_url(url: str) -> None:
    """URL 校验——所有失败都在 httpx 发请求之前。"""
    if len(url) > WEB_URL_MAX_LENGTH:
        raise ValueError(f"url too long: {len(url)} > {WEB_URL_MAX_LENGTH}")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme: {parsed.scheme or '(none)'}")
    if parsed.username or parsed.password:
        raise ValueError("url with credentials is not allowed")
    if not parsed.hostname or "." not in parsed.hostname:
        raise ValueError(f"invalid hostname: {parsed.hostname}")


def _html_to_markdown(html: str) -> str:
    """HTML 转 markdown，关掉 body_width 硬换行。"""
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = False
    h.ignore_images = True
    return h.handle(html)


def web_fetch(args: dict[str, Any], ctx: ToolContext) -> str:
    url = args.get("url", "")
    if not url:
        return "error: missing required argument 'url'"
    try:
        _validate_url(url)
    except ValueError as exc:
        return f"error: {exc}"
    headers = {
        "User-Agent": WEB_USER_AGENT,
        "Accept": "text/html,text/*;q=0.9,*/*;q=0.5",
    }
    try:
        with httpx.Client(timeout=WEB_FETCH_TIMEOUT_S, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"error: {exc}"
    if len(resp.content) > WEB_FETCH_MAX_BYTES:
        return f"error: response too large: {len(resp.content)} > {WEB_FETCH_MAX_BYTES}"
    content_type = resp.headers.get("content-type", "").lower()
    if "text/html" in content_type or "application/xhtml" in content_type:
        body = _html_to_markdown(resp.text)
    elif content_type.startswith("text/") or "json" in content_type or "xml" in content_type:
        body = resp.text
    else:
        return f"error: unsupported content-type: {content_type or '(none)'}"
    return truncate_output(body, max_chars=WEB_FETCH_MAX_CHARS)


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo 返回的 href 形如 /l/?uddg=ENCODED_URL&rut=..."""
    if "/l/" not in href:
        return href
    parsed = urlparse(href if href.startswith("http") else f"https:{href}")
    params = parse_qs(parsed.query)
    if "uddg" in params:
        return unquote(params["uddg"][0])
    return href


def _duckduckgo_search(query: str, max_results: int) -> list[tuple[str, str]]:
    """从 DuckDuckGo HTML 端点摘结果标题和 URL。"""
    headers = {"User-Agent": WEB_USER_AGENT}
    results: list[tuple[str, str]] = []
    try:
        with httpx.Client(timeout=WEB_SEARCH_TIMEOUT_S, follow_redirects=True) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
            )
            resp.raise_for_status()
    except httpx.HTTPError:
        return results
    for m in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        resp.text, re.DOTALL,
    ):
        href = _unwrap_ddg_url(m.group(1))
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if title:
            results.append((title, href))
        if len(results) >= max_results:
            break
    return results


def web_search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = args.get("query", "")
    if not query:
        return "error: missing required argument 'query'"
    max_results = min(int(args.get("max_results", 10)), 10)
    try:
        results = _duckduckgo_search(query, max_results)
    except Exception as exc:
        return f"error: search failed: {exc}"
    if not results:
        return "(no results)"
    lines = [f"- {title}\n  {url}" for title, url in results]
    return truncate_output("\n".join(lines), max_chars=WEB_FETCH_MAX_CHARS)


# ── ToolRegistry ──────────────────────────────────────────

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def list_tools(self) -> list[Tool]:
        return self.list()

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(t.name for t in self._tools.values())
            return ToolResult(
                tool_call_id=call.id,
                content=f"未知工具: {call.name}（可用工具: {available}）",
                is_error=True,
            )
        return ToolResult(tool_call_id=call.id, content=tool.run(call.arguments, ctx))


# ── 默认工具注册 ──────────────────────────────────────────

def default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="回显输入的文本",
            run=echo,
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要回显的文本"}
                },
                "required": ["text"],
            },
        )
    )
    registry.register(
        Tool(
            name="system_date",
            description="获取当前系统日期和时间",
            run=system_date,
        )
    )
    registry.register(
        Tool(
            name="uppercase",
            description="Convert text to uppercase.",
            run=uppercase,
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要转为大写的文本"}
                },
                "required": ["text"],
            },
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a file from the local filesystem.",
            run=read_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"],
            },
        )
    )
    registry.register(
        Tool(
            name="list_files",
            description="List files and directories in a directory.",
            run=list_files,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, defaults to '.'"}
                },
            },
        )
    )
    registry.register(
        Tool(
            name="glob",
            description="Find files matching a glob pattern recursively.",
            run=glob,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"}
                },
                "required": ["pattern"],
            },
        )
    )
    registry.register(
        Tool(
            name="grep",
            description="Search for a regex pattern in files.",
            run=grep,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in, defaults to '.'"},
                    "glob": {"type": "string", "description": "Optional file glob filter"},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
                },
                "required": ["pattern"],
            },
        )
    )
    registry.register(
        Tool(
            name="project_tree",
            description="Show a tree view of the project directory.",
            run=project_tree,
            parameters={
                "type": "object",
                "properties": {
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse, default 3",
                    },
                },
            },
        )
    )
    registry.register(
        Tool(
            name="web_fetch",
            description="Fetch a web page and return its content as markdown.",
            run=web_fetch,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"],
            },
        )
    )
    registry.register(
        Tool(
            name="web_search",
            description="Search the web via DuckDuckGo.",
            run=web_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (1-10), default 10",
                    },
                },
                "required": ["query"],
            },
        )
    )
    return registry
