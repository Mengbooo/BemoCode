from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# 文本文件后缀白名单：直接放行，不用 peek 文件头。
TEXT_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".cfg", ".ini", ".env", ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx",
    ".jsx", ".html", ".css", ".sql", ".lock", ".gitignore",
}

MAX_READ_BYTES = 256 * 1024     # 单文件大小上限
DEFAULT_MAX_CHARS = 8000        # 单次 observation 上限
DEFAULT_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "dist", "build",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})


def resolve_in_cwd(cwd: Path, user_path: str) -> Path:
    """把模型给的相对路径解析成绝对路径，并强制锁回 cwd 子树。"""
    candidate = (cwd / user_path).resolve()
    cwd_resolved = cwd.resolve()
    try:
        candidate.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes cwd: {user_path}") from exc
    return candidate


def ensure_text_file(path: Path) -> None:
    """白名单后缀直接放行；其余 peek 首 1 KB，NUL 字节判为二进制。"""
    if path.suffix.lower() in TEXT_SUFFIXES:
        return
    with path.open("rb") as f:
        if b"\x00" in f.read(1024):
            raise ValueError(f"binary file: {path.name}")


def ensure_within_size(path: Path, max_bytes: int = MAX_READ_BYTES) -> None:
    """检查文件大小是否在允许范围内。"""
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"file too large: {size} > {max_bytes}")


def truncate_output(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated {len(text) - max_chars} chars]"


@dataclass
class ReadFileState:
    """path -> (mtime_ns, char_count)。Day 4 read-before-edit 会比对 mtime。"""
    entries: dict[Path, tuple[int, int]] = field(default_factory=dict)

    def record(self, path: Path, content: str) -> None:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return
        self.entries[path] = (mtime_ns, len(content))


@dataclass
class SkipPolicy:
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS
    gitignore: "pathspec.PathSpec | None" = None

    @classmethod
    def default(cls, gitignore: "pathspec.PathSpec | None" = None) -> "SkipPolicy":
        return cls(gitignore=gitignore)


def should_skip(path: Path, policy: SkipPolicy) -> bool:
    """判断是否跳过该路径。"""
    # 检查目录名黑名单
    parts = path.parts
    for part in parts:
        if part in policy.skip_dirs:
            return True
    # 检查 gitignore 规则
    if policy.gitignore is not None:
        if policy.gitignore.match_file(str(path)):
            return True
    return False


def load_gitignore(cwd: Path) -> "pathspec.PathSpec | None":
    """只读 cwd 根的 .gitignore。"""
    import pathspec
    gitignore = cwd / ".gitignore"
    if not gitignore.exists():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)
