import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from langchain import hub
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from config import get_llm


class Task(BaseModel):
    idx: int = Field(description="Task index")
    tool: str = Field(description="Name of the tool to execute")
    args: dict[str, Any] = Field(description="Arguments to pass to the tool")
    dependencies: list[int] = Field(
        description="List of task indices this task depends on", default_factory=list
    )


class State(BaseModel):
    messages: Annotated[list, add_messages]
    tools: list[BaseTool]


class LLMCompilerPlanParser:
    def __init__(self, tools: list[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

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

    def stream(self, messages: list, execution_start: float):
        prompt = hub.pull("wfh/llm-compiler")

        tool_descriptions = f"""{"\n".join(
            f"{i + 1}. {tool.name}: {tool.description}\n"
            f"   Parameters: {self._get_tool_params(tool)}\n"
            for i, tool in enumerate(self.tools.values())
        )}

IMPORTANT: You MUST use the exact tool names: {', '.join(tool.name for tool in self.tools.values())}

Use the correct parameter names and types as defined in the tool descriptions above.

DEPENDENCIES: Use $N syntax to reference outputs from previous tasks.
Example: tool_name(param='$1', other_param='$2') - uses outputs from tasks 1 and 2
This creates a DAG where tasks execute based on dependencies, not plan order!"""

        planner_prompt = prompt.partial(
            replan="",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )

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

    def _parse_tasks(self, content: str) -> list[Task]:
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
            args_str = task_content.split("(")[1].split(")")[0]

            args = {}
            dependencies = []

            if args_str:
                for arg in args_str.split(","):
                    if "=" not in arg:
                        continue

                    key, value = arg.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    args[key] = value

                    if value.startswith("$"):
                        dep_match = re.search(r"\$(\d+)", value)
                        if dep_match:
                            dependencies.append(int(dep_match.group(1)))

            if tool_name in self.tools or tool_name == "join":
                task = Task(
                    idx=idx,
                    tool=tool_name,
                    args=args,
                    dependencies=dependencies,
                )
                tasks.append(task)

        return tasks


def _get_observations(messages: list[Any]) -> dict[int, Any]:
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


def _resolve_arg(arg: str, observations: dict[int, Any]) -> str:
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    return re.sub(ID_PATTERN, replace_match, arg)


def _execute_task(
    task: Task,
    observations: dict[int, Any],
    config: dict[str, Any],
    tools: dict[str, Any],
):
    tool_to_use = tools[task.tool]
    resolved_args = {
        key: _resolve_arg(val, observations) for key, val in task.args.items()
    }
    return tool_to_use.invoke(resolved_args, config)


def schedule_task(task_inputs: dict[str, Any], config: dict[str, Any]):
    task: Task = task_inputs["task"]
    observations: dict[int, Any] = task_inputs["observations"]
    tools: dict[str, Any] = task_inputs["tools"]

    execution_start = config.get("start_time", time.time())
    current_time = time.time()
    print(
        f"[{current_time - execution_start:.3f}s] 🚀 STARTED task {task.idx}: {task.tool}({task.args})"
    )

    observation = _execute_task(task, observations, config, tools)
    observations[task.idx] = observation

    end_time = time.time()
    print(
        f"[{end_time - execution_start:.3f}s] ✅ COMPLETED task {task.idx}: {task.tool}"
    )


def schedule_pending_task(
    task: Task,
    observations: dict[int, Any],
    tools: dict[str, Any],
    execution_start: float,
    retry_after: float = 0.2,
):
    current_time = time.time()
    print(
        f"[{current_time - execution_start:.3f}s] ⏳ WAITING task {task.idx}: {task.tool} (deps: {task.dependencies})"
    )

    while True:
        deps = task.dependencies
        if deps and any(dep not in observations for dep in deps):
            time.sleep(retry_after)
            continue
        schedule_task(
            {"task": task, "observations": observations, "tools": tools},
            {"start_time": execution_start},
        )
        break


def schedule_tasks(scheduler_input: dict[str, Any]) -> list[FunctionMessage]:
    """Task Fetching Unit - schedules and executes tasks as soon as they are executable"""
    tasks = scheduler_input["tasks"]
    messages = scheduler_input["messages"]
    tools = scheduler_input["tools"]
    execution_start = scheduler_input["execution_start"]
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)

    tools_dict = {tool.name: tool for tool in tools}
    futures = []

    print(f"[{time.time() - execution_start:.3f}s] 🚀 Task Fetching Unit: Initialized")

    with ThreadPoolExecutor() as executor:
        print(f"[{time.time() - execution_start:.3f}s] 🚀 Starting eager execution...")

        args_for_tasks = {}
        for task in tasks:
            deps = task.dependencies
            task_names[task.idx] = task.tool
            args_for_tasks[task.idx] = task.args

            if deps and any(dep not in observations for dep in deps):
                print(
                    f"[{time.time() - execution_start:.3f}s] ⏳ QUEUED {task.idx}: {task.tool} (waiting for: {', '.join(map(str, deps))})"
                )
                futures.append(
                    executor.submit(
                        schedule_pending_task,
                        task,
                        observations,
                        tools_dict,
                        execution_start,
                    )
                )
            else:
                print(
                    f"[{time.time() - execution_start:.3f}s] 🚀 DISPATCHED {task.idx}: {task.tool}"
                )
                futures.append(
                    executor.submit(
                        schedule_task,
                        dict(task=task, observations=observations, tools=tools_dict),
                        {"start_time": execution_start},
                    )
                )

        print(
            f"[{time.time() - execution_start:.3f}s] ⏳ Task Fetching Unit: Waiting for all tasks to complete..."
        )
        wait(futures)
        print(
            f"[{time.time() - execution_start:.3f}s] ✅ Task Fetching Unit: All tasks completed!"
        )

    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }

    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args},
            tool_call_id=k,
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


def plan_and_schedule(state: State):
    messages = state.messages
    tools = state.tools
    execution_start = time.time()
    planner = LLMCompilerPlanParser(tools)
    tasks = planner.stream(messages, execution_start)

    tasks = itertools.chain([next(tasks)], tasks)

    scheduled_tasks = schedule_tasks(
        {
            "messages": messages,
            "tasks": tasks,
            "tools": tools,
            "execution_start": execution_start,
        }
    )
    return State(messages=scheduled_tasks, tools=tools)


def _joiner(state: State):
    print("🔗 Joining results...")
    messages = state.messages
    user_input = messages[0].content

    prompt = f"""Original request: {user_input}

Based on the completed tasks, provide a comprehensive response summarizing what was accomplished."""

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"📝 Final response: {response.content[:200]}...")

    return State(messages=messages + [response], tools=state.tools)


def should_continue(state: State):
    messages = state.messages
    if isinstance(messages[-1], AIMessage):
        return END
    return "plan_and_schedule"


class LLMCompiler:
    def __init__(self, tools: list[BaseTool]):
        self.tools = tools
        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("plan_and_schedule", plan_and_schedule)
        graph_builder.add_node("join", _joiner)
        graph_builder.add_edge("plan_and_schedule", "join")
        graph_builder.add_conditional_edges("join", should_continue)
        graph_builder.add_edge(START, "plan_and_schedule")
        return graph_builder.compile()

    async def run(self, user_input: str):
        print("🚀 Starting LLMCompiler execution")

        initial_state = State(
            messages=[HumanMessage(content=user_input)],
            tools=self.tools,
        )

        return self.graph.invoke(initial_state)
