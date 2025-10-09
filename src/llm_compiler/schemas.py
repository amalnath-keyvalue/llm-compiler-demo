from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class State(BaseModel):
    messages: list[BaseMessage]
    tools: list[BaseTool]


class Task(BaseModel):
    idx: int
    tool: str
    args: dict[str, Any]
    dependencies: list[int] = []
