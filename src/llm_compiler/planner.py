import re
import time

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from .config import get_llm
from .constants import PLANNER_PROMPT_TEMPLATE
from .schemas import Task


class Planner:
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
            f"[{time.time() - execution_start:.3f}s] 📋 Planner: Started planning tasks"
        )
        user_query = messages[0].content if messages else "No query provided"

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
        task = self._parse_task(line)
        if task:
            deps_str = (
                f" (deps: {task.dependencies})" if task.dependencies else " (no deps)"
            )
            print(
                f"[{time.time() - execution_start:.3f}s] 📋 PLANNED task {task.idx}: {task.tool}({task.args}){deps_str}"
            )
            return task

    def _parse_task(
        self,
        line: str,
    ):
        line = line.strip()
        # Example line: "3. tool_name(key='value', other='$2') (deps: [1,2])"
        match = re.match(
            r"^(\d+)\.\s*([^(]+)\(([^)]+)\)(?:\s*\(deps:\s*\[([^\]]*)\]\))?",
            line,
        )
        if not match:
            return None

        idx, tool_name, args_str, deps_str = match.groups()

        args = {}
        dependencies = set()

        if args_str.strip():
            for arg in args_str.split(","):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    args[key.strip()] = value.strip().strip("\"'")

        for value in args.values():
            if isinstance(value, str) and value.startswith("$"):
                dep_match = re.search(r"\$(\d+)", value)
                if dep_match:
                    dependencies.add(int(dep_match.group(1)))

        if deps_str:
            dependencies.update(
                int(dep.strip()) for dep in deps_str.split(",") if dep.strip().isdigit()
            )

        return Task(
            idx=int(idx),
            tool=tool_name.strip(),
            args=args,
            dependencies=list(dependencies),
        )
