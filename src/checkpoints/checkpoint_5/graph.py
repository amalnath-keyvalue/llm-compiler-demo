from langchain_core.tools import BaseTool

from ..checkpoint_4 import LLMCompilerWithPlannerAndSimpleSchedulerOnly
from .scheduler import Scheduler


class LLMCompilerWithPlannerAndSchedulerOnly(
    LLMCompilerWithPlannerAndSimpleSchedulerOnly
):
    def __init__(
        self,
        tools: list[BaseTool],
    ):
        super().__init__(
            tools=tools,
            scheduler=Scheduler(),
        )
