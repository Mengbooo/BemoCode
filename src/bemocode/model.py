from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Protocol
from anthropic import Anthropic


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
    assistant_content: list[dict[str, Any]] | None = None
    stop_reason: str = "end_turn"


class ModelProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> ModelResponse:
        ...


class MockProvider:
    def __init__(self) -> None:
        self._last_tool: str = "echo"

    def complete(self, messages: list[dict[str, str]], tools: list[Any] | None = None) -> ModelResponse:
        last = messages[-1]
        content = last["content"]

        # 检测 tool_result 格式（Anthropic 协议：role=user, content 是 content blocks 列表）
        if isinstance(content, list):
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            if tool_results:
                result_text = tool_results[0].get("content", "")
                return ModelResponse(text=f"{self._last_tool} 工具返回：{result_text}")

        if last["role"] == "user":
            # content 可能是字符串或列表
            text = content if isinstance(content, str) else str(content)
            self._last_tool = "echo"
            if "用 " in text and " 工具说" in text:
                _, _, rest = text.partition("用 ")
                tool_name, _, text = rest.partition(" 工具说")
                self._last_tool = tool_name.strip()
                text = text.strip() or text
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


def _to_anthropic_tools(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _parse_tool_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        return raw.model_dump(exclude_none=True)
    if hasattr(raw, "dict"):
        return raw.dict(exclude_none=True)
    return {"value": str(raw)}


def _content_block_to_dict(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if hasattr(block, "dict"):
        return block.dict(exclude_none=True)
    data: dict[str, Any] = {"type": block.type}
    for name in ("text", "id", "name", "input", "thinking", "signature"):
        if hasattr(block, name):
            data[name] = getattr(block, name)
    return data


class AnthropicProvider:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        max_tokens: int = 1024,
        base_url: str | None = None,
    ) -> None:
        # 兼容 ANTHROPIC_AUTH_TOKEN（DeepSeek 风格）和 ANTHROPIC_API_KEY（官方风格）。
        api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 ANTHROPIC_AUTH_TOKEN，例如：export ANTHROPIC_AUTH_TOKEN='sk-...'")
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get(
            "ANTHROPIC_BASE_URL",
            "https://api.deepseek.com/anthropic",
        )
        self.client = Anthropic(api_key=api_key, base_url=self.base_url)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        response = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_content: list[dict[str, Any]] = []

        for block in response.content:
            assistant_content.append(_content_block_to_dict(block))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=_parse_tool_input(block.input),
                    )
                )

        return ModelResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls or None,
            assistant_content=assistant_content or None,
            stop_reason=response.stop_reason or "end_turn",
        )


def create_provider(name: str, model: str, base_url: str | None = None) -> ModelProvider:
    if name == "anthropic":
        return AnthropicProvider(model=model, base_url=base_url)
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider: {name}")
