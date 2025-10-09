import time

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from .config import get_llm
from .constants import PLANNER_PROMPT_TEMPLATE


class SimplePlanner:
    def __init__(
        self,
        tools: list[BaseTool],
    ):
        self.tools = tools
        self.tool_names = ", ".join(tool.name for tool in tools)
        self.tool_descriptions = f"{"\n".join(
            f"{i + 1}. {tool.name}: {tool.description}\n"
            f"   Parameters: {self._get_tool_params(tool)}\n"
            for i, tool in enumerate(tools)
        )}"
        self.llm = get_llm()

    def _get_tool_params(
        self,
        tool: BaseTool,
    ):
        schema = tool.get_input_schema().model_json_schema()
        properties: dict[str, dict] = schema.get("properties", {})
        required = schema.get("required", [])

        param_descriptions = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            is_required = param_name in required
            required_str = "required" if is_required else "optional"
            param_descriptions.append(f"{param_name} ({param_type}, {required_str})")

        return ", ".join(param_descriptions) if param_descriptions else "none"

    def plan_tasks(
        self,
        messages: list[BaseMessage],
        execution_start: float,
    ):
        print(
            f"[{time.time() - execution_start:.3f}s] 📋 PLANNER: Started planning tasks"
        )
        user_query = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            tool_count=len(self.tools),
            tool_names=self.tool_names,
            tool_descriptions=self.tool_descriptions,
            user_query=user_query,
        )

        buffer = ""
        for chunk in self.llm.stream(prompt):
            if chunk.content:
                buffer += chunk.content
                lines = buffer.split("\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        yield self._parse_task_line(
                            line=line,
                            execution_start=execution_start,
                        )

        if buffer.strip():
            yield self._parse_task_line(
                line=buffer,
                execution_start=execution_start,
            )

    def _parse_task_line(
        self,
        line: str,
        execution_start: float,
    ):
        task = line.strip()
        print(f"[{time.time() - execution_start:.3f}s] ✏️  GENERATED task {task}")
        return task
