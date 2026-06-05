from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .paths import MAX_INDEX_BYTES, MAX_INDEX_LINES, index_path, memory_dir
from .types import MEMORY_TYPES, MemoryEntry


def load_index(cwd: Path) -> str | None:
    """读取 MEMORY.md 索引文件。不存在返回 None。"""
    index = index_path(cwd)
    if not index.exists():
        return None
    content = index.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return None
    # 截断过长索引
    lines = content.splitlines()
    if len(lines) > MAX_INDEX_LINES:
        content = "\n".join(lines[:MAX_INDEX_LINES]) + "\n\n[... index truncated]"
    if len(content.encode("utf-8")) > MAX_INDEX_BYTES:
        content = content.encode("utf-8")[:MAX_INDEX_BYTES].decode("utf-8", errors="replace")
    return content


def write_memory(
    cwd: Path, mem_type: str, title: str, body: str
) -> MemoryEntry:
    """写一条记忆到 memdir，同时更新 MEMORY.md 索引。"""
    if mem_type not in MEMORY_TYPES:
        raise ValueError(f"unknown memory type: {mem_type}")

    slug = _slugify(title)
    file_path = memory_dir(cwd) / f"{slug}.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 写 topic 文件
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"# {title}\n\n> Type: {mem_type} | Saved: {now}\n\n{body}\n"
    file_path.write_text(content, encoding="utf-8")

    # 更新索引
    _append_index(cwd, title, slug)

    return MemoryEntry(
        mem_type=mem_type,
        title=title,
        slug=slug,
        body=body,
        file_path=str(file_path),
    )


def recall_memory(cwd: Path, query: str, top_k: int = 5) -> list[MemoryEntry]:
    """关键词召回记忆。按 title+body 中的匹配数排序。"""
    md = memory_dir(cwd)
    if not md.exists():
        return []

    # 关键词：空格分词
    keywords = query.lower().split()

    scored: list[tuple[int, MemoryEntry]] = []
    for f in sorted(md.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        text = f.read_text(encoding="utf-8", errors="replace").lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            # 尝试提取 title（# 开头那行）
            first_line = text.split("\n", 1)[0] if text else f.name
            title = first_line.lstrip("#").strip() or f.stem
            scored.append((
                score,
                MemoryEntry(
                    mem_type="reference",
                    title=title,
                    slug=f.stem,
                    body=text,
                    file_path=str(f),
                ),
            ))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


def _slugify(title: str) -> str:
    """ASCII-only slug from title; falls back to hash if no letters."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    if not slug or not any(c.isalpha() for c in slug):
        slug = "mem-" + hashlib.sha1(title.encode()).hexdigest()[:8]
    return slug[:64]


def _append_index(cwd: Path, title: str, slug: str) -> None:
    """追加一行到 MEMORY.md 索引。"""
    ip = index_path(cwd)
    ip.parent.mkdir(parents=True, exist_ok=True)
    line = f"- [{title}]({slug}.md)\n"
    with open(ip, "a", encoding="utf-8") as f:
        f.write(line)
