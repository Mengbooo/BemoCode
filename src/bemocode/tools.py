from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from .model import ToolCall, ToolResult

ToolFunc = Callable[[dict[str, Any]], str]


@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


def echo(args: dict[str, Any]) -> str:
    return str(args.get("text", ""))


def system_date(args: dict[str, Any]) -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def uppercase(args: dict[str, Any]) -> str:
    return str(args.get("text", "")).upper()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def list_tools(self) -> list[Tool]:
        return self.list()

    def run(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(t.name for t in self._tools.values())
            return ToolResult(
                tool_call_id=call.id,
                content=f"未知工具: {call.name}（可用工具: {available}）",
                is_error=True,
            )
        return ToolResult(tool_call_id=call.id, content=tool.run(call.arguments))


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
    return registry
