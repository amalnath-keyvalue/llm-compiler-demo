import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from .planner import Planner
from .scheduler import Scheduler
from .schemas import State


class LLMCompiler:
    def __init__(
        self,
        tools: list[BaseTool],
    ):
        self.tools = tools
        self.planner = Planner(tools)
        self.scheduler = Scheduler()
        self.graph = self._create_graph()

    def _create_graph(
        self,
    ):
        graph_builder = StateGraph(State)
        graph_builder.add_node("plan_and_schedule", self._plan_and_schedule)
        graph_builder.add_node("join", self._join)
        graph_builder.add_edge(START, "plan_and_schedule")
        graph_builder.add_edge("plan_and_schedule", "join")
        graph_builder.add_conditional_edges("join", self._should_continue)
        return graph_builder.compile()

    def _plan_and_schedule(
        self,
        state: State,
    ):
        messages = state.messages
        tools = state.tools
        execution_start = time.time()

        tasks = self.planner.plan_tasks(
            messages=messages,
            execution_start=execution_start,
        )
        task_messages = self.scheduler.schedule_tasks(
            tasks=tasks,
            messages=messages,
            tools=tools,
            execution_start=execution_start,
        )

        return State(
            messages=task_messages,
            tools=tools,
        )

    def _join(
        self,
        state: State,
    ):
        print("🔗 Joining results...")
        messages = state.messages
        tools = state.tools

        task_results = {
            message.tool_call_id: message.content
            for message in messages
            if isinstance(message, ToolMessage)
        }

        if not task_results:
            return State(
                messages=[AIMessage(content="No tasks were executed.")],
                tools=tools,
            )

        summary = (
            "Task execution completed successfully. Here's what was accomplished:\n\n"
        )
        for idx, result in sorted(task_results.items(), key=lambda x: int(x[0])):
            summary += f"Task {idx}: {result[:200]}...\n\n"

        print(f"📝 Final response: {summary}")
        return State(
            messages=[AIMessage(content=summary)],
            tools=tools,
        )

    def _should_continue(
        self,
        _state: State,
    ):
        # TODO: Implement a better re-planning logic
        return END

    async def run(
        self,
        user_input: str,
    ):
        print("🚀 Starting LLMCompiler execution")

        initial_state = State(
            messages=[HumanMessage(content=user_input)],
            tools=self.tools,
        )

        return self.graph.invoke(initial_state)
