import re
import time
from typing import Any, Generator

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from config import get_llm


class Task(BaseModel):
    idx: int
    tool: str
    args: dict[str, Any]
    dependencies: list[int] = []


class Planner:
    def __init__(self, tools: list[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

    def stream(
        self, messages: list, execution_start: float
    ) -> Generator[Task, None, None]:
        """Public method to stream tasks from the LLM"""
        # Create our own custom prompt without the join tool
        tool_descriptions = f"""{"\n".join(
            f"{i + 1}. {tool.name}: {tool.description}\n"
            f"   Parameters: {self._get_tool_params(tool)}\n"
            for i, tool in enumerate(self.tools.values())
        )}"""

        planner_prompt = f"""Given a user query, create a plan to solve it with the utmost parallelizability. Each plan should comprise an action from the following {len(self.tools)} types:
{tool_descriptions}

USER QUERY: {messages[0].content if messages else 'No query provided'}

IMPORTANT: Use exact tool names: {', '.join(tool.name for tool in self.tools.values())}

Guidelines:
- Each action described above contains input/output types and description.
  - You must strictly adhere to the input and output types for each action.
  - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
- Each action in the plan should strictly be one of the above types. Follow the conventions for each action.
- Each action MUST have a unique ID, which is strictly increasing.
- Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
- Ensure the plan maximizes parallelizability.
- Only use the provided action types. If a query cannot be addressed using these, explain what additional tools would be needed.
- Never introduce new actions other than the ones provided.

DEPENDENCIES: Use $N to reference previous task outputs.
Example: tool_name(param='$2') uses output from task 2.

PLANNING: Break tasks into logical steps with dependencies:
- When one task produces output that another task needs as input, use $N to reference it
- Create dependencies to form an efficient workflow
- Independent tasks can run in parallel

CRITICAL: Always use dependencies when one task's output is needed by another!
- If task A produces output, and task B needs that output, use $A in task B
- Generate content for EACH file separately - don't generate everything at once
- Create dependencies to form an efficient workflow
- This creates a DAG where tasks execute based on dependencies, not plan order

Format: N. tool_name(param='value', other='$N') (deps: [1, 2, 3])"""

        llm = get_llm()
        buffer = ""
        seen_tasks = set()

        for chunk in llm.stream(planner_prompt.format(messages=messages)):
            if hasattr(chunk, "content") and chunk.content:
                buffer += chunk.content
                tasks = self._parse_tasks(buffer)

                for task in tasks:
                    if task.idx not in seen_tasks:
                        seen_tasks.add(task.idx)
                        current_time = time.time() - execution_start
                        deps_str = (
                            f" (deps: {task.dependencies})"
                            if task.dependencies
                            else " (no deps)"
                        )
                        print(
                            f"[{current_time:.3f}s] 📋 PLANNED task {task.idx}: {task.tool}({task.args}){deps_str}"
                        )
                        yield task

    def _get_tool_params(self, tool: BaseTool) -> str:
        schema = tool.get_input_schema().model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        param_descriptions = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            is_required = param_name in required
            required_str = "required" if is_required else "optional"
            param_descriptions.append(f"{param_name} ({param_type}, {required_str})")

        return ", ".join(param_descriptions) if param_descriptions else "none"

    def _parse_tasks(self, content: str) -> list[Task]:
        """Parse tasks from LLM output content"""
        tasks = []
        lines = content.split("\n")
        seen_indices = set()

        for line in lines:
            line = line.strip()
            if not re.match(r"^\d+\.", line):
                continue

            parts = line.split(".", 1)
            if len(parts) < 2:
                continue

            idx = int(parts[0])
            if idx in seen_indices:
                continue
            seen_indices.add(idx)

            task_content = parts[1].strip()
            if "(" not in task_content or ")" not in task_content:
                continue

            tool_name = task_content.split("(")[0].strip()
            args_str = task_content[
                task_content.find("(") + 1 : task_content.rfind(")")
            ]

            args = {}
            if args_str:
                for arg in args_str.split(","):
                    if "=" not in arg:
                        continue

                    key, value = arg.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    args[key] = value

            dependencies = []

            for _, value in args.items():
                if isinstance(value, str) and value.startswith("$"):
                    dep_match = re.search(r"\$(\d+)", value)
                    if dep_match:
                        dependencies.append(int(dep_match.group(1)))

            deps_match = re.search(r"\(deps:\s*\[([^\]]+)\]\)", line)
            if deps_match:
                deps_str = deps_match.group(1)
                for dep in deps_str.split(","):
                    dep = dep.strip()
                    if dep.isdigit():
                        dependencies.append(int(dep))

            task = Task(
                idx=idx,
                tool=tool_name,
                args=args,
                dependencies=dependencies,
            )
            tasks.append(task)

        return tasks
