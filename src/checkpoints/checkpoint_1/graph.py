from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from .schemas import State


class BaseLLMCompiler:
    def __init__(
        self,
        tools: list[BaseTool],
    ):
        self.tools = tools
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
        print("📊 GRAPH ARRIVED AT: plan_and_schedule")
        return state

    def _join(
        self,
        state: State,
    ):
        print("📊 GRAPH ARRIVED AT: join")
        return state

    def _should_continue(
        self,
        _state: State,
    ):
        print("📊 GRAPH ARRIVED AT: should_continue")
        return END

    async def run(
        self,
        user_input: str,
    ):
        initial_state = State(
            messages=[HumanMessage(content=user_input)],
        )

        return self.graph.invoke(initial_state)
