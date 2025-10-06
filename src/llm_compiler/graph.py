import itertools
import time

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph

from .joiner import Joiner
from .planner import Planner
from .scheduler import Scheduler
from .state import State


class LLMCompiler:
    def __init__(self, tools: list[BaseTool]):
        self.tools = tools
        self.planner = Planner(tools)
        self.scheduler = Scheduler()
        self.joiner = Joiner()
        self.graph = self._create_graph()

    def _plan_and_schedule(self, state: State) -> State:
        """Plan and schedule tasks"""
        messages = state.messages
        tools = state.tools
        execution_start = time.time()

        tasks = self.planner.stream(messages, execution_start)
        tasks = itertools.chain([next(tasks)], tasks)

        scheduled_tasks = self.scheduler.schedule_tasks(
            {
                "messages": messages,
                "tasks": tasks,
                "tools": tools,
                "execution_start": execution_start,
            }
        )
        return State(messages=scheduled_tasks, tools=tools)

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        graph_builder = StateGraph(State)
        graph_builder.add_node("plan_and_schedule", self._plan_and_schedule)
        graph_builder.add_node("join", self.joiner.join)
        graph_builder.add_edge("plan_and_schedule", "join")
        graph_builder.add_conditional_edges("join", self.joiner.should_continue)
        graph_builder.add_edge(START, "plan_and_schedule")
        return graph_builder.compile()

    async def run(self, user_input: str):
        """Run the LLMCompiler with user input"""
        print("🚀 Starting LLMCompiler execution")

        initial_state = State(
            messages=[HumanMessage(content=user_input)],
            tools=self.tools,
        )

        return self.graph.invoke(initial_state)
