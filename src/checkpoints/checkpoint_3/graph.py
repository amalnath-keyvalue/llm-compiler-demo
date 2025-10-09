from langchain_core.tools import BaseTool

from ..checkpoint_2 import LLMCompilerWithSimplePlannerOnly
from .planner import Planner


class LLMCompilerWithPlannerOnly(LLMCompilerWithSimplePlannerOnly):
    def __init__(
        self,
        tools: list[BaseTool],
    ):
        super().__init__(
            tools=tools,
            planner=Planner(tools=tools),
        )
