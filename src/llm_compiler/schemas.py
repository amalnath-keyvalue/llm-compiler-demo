import time
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class State(BaseModel):
    messages: list[BaseMessage]
    needs_replan: bool = False
    execution_start: float = time.time()


class Task(BaseModel):
    idx: int
    tool: str
    args: dict[str, Any]
    dependencies: list[int] = []
