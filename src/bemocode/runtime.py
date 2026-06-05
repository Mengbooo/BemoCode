from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

from .tools import TodoItem


@dataclass
class RuntimeState:
    permission_mode: str = "default"
    model: str = "deepseek-v4-pro"
    provider: str = "anthropic"
    abort_event: threading.Event = field(default_factory=threading.Event)
    input_queue: "queue.Queue[str]" = field(default_factory=queue.Queue)
    todo_store: list[TodoItem] = field(default_factory=list)
