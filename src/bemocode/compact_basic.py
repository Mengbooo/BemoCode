from __future__ import annotations

from typing import Any


def compact(messages: list[dict[str, Any]], keep: int = 8) -> list[dict[str, Any]]:
    """确定性压缩消息历史。不调 LLM。"""
    pin_count = 2
    if len(messages) <= keep + pin_count:
        return messages

    pinned = messages[:pin_count]
    working = messages[-keep:]
    middle = messages[pin_count:-keep]
    compressed = _build_compressed_block(middle)
    return pinned + [compressed] + working


def _build_compressed_block(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """根据被压缩的消息生成一条摘要 user message。"""
    msg_count = len(messages)
    tool_call_count = 0
    tools_used: set[str] = set()
    files_read: set[str] = set()

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use":
                        tool_call_count += 1
                        name = block.get("name", "")
                        if name:
                            tools_used.add(name)
                        inp = block.get("input", {})
                        if isinstance(inp, dict):
                            p = inp.get("path") or inp.get("file_path", "")
                            if p:
                                files_read.add(str(p))
                    elif block.get("type") == "tool_result":
                        pass

    lines = [
        f"[Compressed {msg_count} messages: "
        f"{tool_call_count} tool calls ({', '.join(sorted(tools_used)) or 'none'})",
    ]
    if files_read:
        lines.append(f"  files: {', '.join(sorted(files_read))}")
    lines.append("  Use context above + current conversation to continue.]")

    return {"role": "user", "content": "\n".join(lines)}
