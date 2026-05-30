from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False

@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    stop_reason: str = "end_turn"

class MockProvider:
    def __init__(self) -> None:
        self._last_tool: str = "echo"

    def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        last = messages[-1]
        if last["role"] == "user":
            content = last["content"]
            # 从用户输入里识别要用哪个工具
            if "uppercase" in content:
                self._last_tool = "uppercase"
            else:
                self._last_tool = "echo"
            text = content.replace(f"用 {self._last_tool} 工具说", "").strip() or content
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id=f"call_{self._last_tool}_1",
                        name=self._last_tool,
                        arguments={"text": text},
                    )
                ],
                stop_reason="tool_use",
            )
        if last["role"] == "tool":
            return ModelResponse(text=f"{self._last_tool} 工具返回：{last['content']}")
        return ModelResponse(text="我现在只能演示 echo 和 uppercase 工具。")