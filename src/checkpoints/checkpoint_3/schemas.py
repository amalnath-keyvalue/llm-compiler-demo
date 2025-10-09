from typing import Any

from pydantic import BaseModel


class Task(BaseModel):
    idx: int
    tool: str
    args: dict[str, Any]
    dependencies: list[int] = []
