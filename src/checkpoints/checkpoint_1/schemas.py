from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class State(BaseModel):
    messages: list[BaseMessage]
