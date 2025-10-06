from langchain_core.messages import HumanMessage
from langgraph.graph import END

from .state import State


class Joiner:
    """Joins task results and generates final response"""

    def join(self, state: State) -> State:
        """Join task results and generate final response"""
        print("🔗 Joining results...")
        messages = state.messages
        tools = state.tools

        task_results = {}
        for message in messages:
            if (
                hasattr(message, "additional_kwargs")
                and "idx" in message.additional_kwargs
            ):
                idx = message.additional_kwargs["idx"]
                task_results[idx] = message.content

        if not task_results:
            return State(
                messages=[HumanMessage(content="No tasks were executed.")],
                tools=tools,
            )

        summary = (
            "Task execution completed successfully. Here's what was accomplished:\n\n"
        )
        for idx, result in sorted(task_results.items()):
            summary += f"Task {idx}: {result}\n\n"

        print(f"📝 Final response: {summary[:500]}...")
        return State(
            messages=[HumanMessage(content=summary)],
            tools=tools,
        )

    def should_continue(self, _state: State) -> str:
        """Determine if execution should continue"""
        return END
