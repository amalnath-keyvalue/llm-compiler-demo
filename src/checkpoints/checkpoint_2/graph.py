import time

from langchain_core.tools import BaseTool

from ..checkpoint_1 import BaseLLMCompiler, State
from .planner import SimplePlanner


class LLMCompilerWithSimplePlannerOnly(BaseLLMCompiler):
    def __init__(
        self,
        tools: list[BaseTool],
        planner: SimplePlanner | None = None,
    ):
        self.planner = planner or SimplePlanner(tools=tools)
        super().__init__(
            tools=tools,
        )

    def _plan_and_schedule(
        self,
        state: State,
    ):
        execution_start = time.time()
        messages = state.messages

        list(
            self.planner.plan_tasks(
                messages=messages,
                execution_start=execution_start,
            )
        )

        return State(
            messages=messages,
        )
