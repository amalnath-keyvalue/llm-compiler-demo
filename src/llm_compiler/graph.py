from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from .config import get_llm
from .constants import JOIN_PROMPT_TEMPLATE, SHOULD_CONTINUE_PROMPT_TEMPLATE
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
        self.llm = get_llm()
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
        execution_start = state.execution_start

        tasks = self.planner.plan_tasks(
            messages=messages,
            execution_start=execution_start,
            needs_replan=state.needs_replan,
        )
        task_messages = self.scheduler.schedule_tasks(
            tasks=tasks,
            messages=messages,
            tools=self.tools,
            execution_start=execution_start,
        )
        messages = [*messages, *task_messages]

        return State(
            messages=messages,
            needs_replan=False,
        )

    def _join(
        self,
        state: State,
    ):
        print("🔗 JOINER: Joining results")
        messages = state.messages

        user_query = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        task_results = {
            message.tool_call_id: message.content
            for message in messages
            if isinstance(message, ToolMessage)
        }
        results_text = "\n\n".join(
            [
                f"Task {idx}: {result}"
                for idx, result in sorted(task_results.items(), key=lambda x: int(x[0]))
            ]
        )

        join_prompt = JOIN_PROMPT_TEMPLATE.format(
            user_query=user_query,
            results_text=results_text,
        )

        response = self.llm.invoke(join_prompt)
        print(f"📝 JOINER: Joined response: {response.content[:500]}...")
        messages = [*messages, AIMessage(content=response.content)]

        print("🤔 JOINER: Deciding whether to continue")
        continue_prompt = SHOULD_CONTINUE_PROMPT_TEMPLATE.format(
            user_query=user_query,
            latest_response=response.content,
        )

        should_continue_response = self.llm.invoke(continue_prompt)
        decision = should_continue_response.content.strip().upper()
        print(f"🎯 JOINER: Decision: {decision}")

        if decision == "REPLAN":
            print("🔄 JOINER: Re-planning needed, returning to planning phase")
            return State(
                messages=messages,
                needs_replan=True,
            )

        print("✅ JOINER: Task complete, ending execution")
        return State(
            messages=messages,
            needs_replan=False,
        )

    def _should_continue(
        self,
        state: State,
    ):
        if state.needs_replan:
            return "plan_and_schedule"
        return END

    async def run(
        self,
        user_input: str,
    ):
        print("🚀 LLMCompiler: Starting execution")

        initial_state = State(
            messages=[HumanMessage(content=user_input)],
        )

        return self.graph.invoke(initial_state)
