from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from .model import ToolCall, ToolResult

ToolFunc = Callable[[dict[str, Any]], str]
@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc
def echo(args: dict[str, Any]) -> str:
    return str(args.get("text", ""))

def uppercase(args: dict[str, Any]) -> str:
    return str(args.get("text", "")).upper()
    
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    def run(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )
        return ToolResult(tool_call_id=call.id, content=tool.run(call.arguments))
def default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="echo", description="Return the input text.", run=echo))
    registry.register(Tool(name="uppercase", description="Convert text to uppercase.", run=uppercase))
    return registry