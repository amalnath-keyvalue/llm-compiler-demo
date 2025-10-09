import time

from langchain_core.tools import BaseTool

from ..checkpoint_1 import State
from ..checkpoint_3 import LLMCompilerWithPlannerOnly
from .scheduler import SimpleScheduler


class LLMCompilerWithPlannerAndSimpleSchedulerOnly(LLMCompilerWithPlannerOnly):
    def __init__(
        self,
        tools: list[BaseTool],
        scheduler: SimpleScheduler | None = None,
    ):
        self.scheduler = scheduler or SimpleScheduler()
        super().__init__(
            tools=tools,
        )

    def _plan_and_schedule(
        self,
        state: State,
    ):
        execution_start = time.time()
        messages = state.messages

        tasks = self.planner.plan_tasks(
            messages=messages,
            execution_start=execution_start,
        )
        task_messages = self.scheduler.schedule_tasks(
            tasks=tasks,
            messages=messages,
            tools=self.tools,
            execution_start=execution_start,
        )

        return State(
            messages=task_messages,
        )
