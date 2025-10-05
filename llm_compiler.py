import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from langchain import hub
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated


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

    def stream(self, messages: list):
        prompt = hub.pull("wfh/llm-compiler")

        tool_descriptions = "\n".join(
            f"{i + 1}. {tool.name}: {tool.description}\n"
            f"   Parameters: {self._get_tool_params(tool)}\n"
            for i, tool in enumerate(self.tools.values())
        )

        tool_descriptions += f"\nIMPORTANT: You MUST use the exact tool names: {', '.join(tool.name for tool in self.tools.values())}"
        tool_descriptions += "\n\nUse the correct parameter names and types as defined in the tool descriptions above."
        tool_descriptions += (
            "\n\nDEPENDENCIES: Use $N syntax to reference outputs from previous tasks."
        )
        tool_descriptions += "\nExample: tool_name(param='$1', other_param='$2') - uses outputs from tasks 1 and 2"
        tool_descriptions += "\nThis creates a DAG where tasks execute based on dependencies, not plan order!"

        planner_prompt = prompt.partial(
            replan="",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        buffer = ""
        seen_tasks = set()
        start_time = time.time()

        for chunk in llm.stream(planner_prompt.format(messages=messages)):
            if hasattr(chunk, "content") and chunk.content:
                buffer += chunk.content

                tasks = self._parse_tasks(buffer)
                for task in tasks:
                    if task.idx not in seen_tasks:
                        seen_tasks.add(task.idx)
                        current_time = time.time() - start_time
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
            if re.match(r"^\d+\.", line):
                parts = line.split(".", 1)
                if len(parts) > 1:
                    idx = int(parts[0])

                    if idx in seen_indices:
                        continue
                    seen_indices.add(idx)

                    task_content = parts[1].strip()

                    if "(" in task_content and ")" in task_content:
                        tool_name = task_content.split("(")[0].strip()
                        args_str = task_content.split("(")[1].split(")")[0]

                        args = {}
                        dependencies = []

                        if args_str:
                            for arg in args_str.split(","):
                                if "=" in arg:
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
    tool_to_use = task.tool
    if isinstance(tool_to_use, str):
        tool_to_use = tools.get(tool_to_use)
        if tool_to_use is None:
            return f"ERROR: Tool '{task.tool}' not found"

    args = task.args
    resolved_args = {key: _resolve_arg(val, observations) for key, val in args.items()}

    return tool_to_use.invoke(resolved_args, config)


def schedule_task(task_inputs: dict[str, Any], config: dict[str, Any]):
    task: Task = task_inputs["task"]
    observations: dict[int, Any] = task_inputs["observations"]
    tools: dict[str, Any] = task_inputs["tools"]

    start_time = time.time()
    print(
        f"[{start_time - config.get('start_time', start_time):.3f}s] 🚀 STARTED task {task.idx}: {task.tool}({task.args})"
    )

    observation = _execute_task(task, observations, config, tools)
    observations[task.idx] = observation

    end_time = time.time()
    print(
        f"[{end_time - config.get('start_time', end_time):.3f}s] ✅ COMPLETED task {task.idx}: {task.tool}"
    )


def schedule_pending_task(
    task: Task,
    observations: dict[int, Any],
    tools: dict[str, Any],
    retry_after: float = 0.2,
):
    start_time = time.time()
    print(
        f"[{start_time - time.time():.3f}s] ⏳ WAITING task {task.idx}: {task.tool} (deps: {task.dependencies})"
    )

    while True:
        deps = task.dependencies
        if deps and (any([dep not in observations for dep in deps])):
            time.sleep(retry_after)
            continue
        schedule_task(
            {"task": task, "observations": observations, "tools": tools},
            {"start_time": start_time},
        )
        break


def schedule_tasks(scheduler_input: dict[str, Any]) -> list[FunctionMessage]:
    """Task Fetching Unit - schedules and executes tasks as soon as they are executable"""
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    tools = scheduler_input["tools"]
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)

    # Create tools lookup dictionary
    tools_dict = {tool.name: tool for tool in tools}

    futures = []
    retry_after = 0.25
    execution_start = time.time()

    print(f"[{0.000:.3f}s] 🚀 Task Fetching Unit: Starting parallel execution...")

    with ThreadPoolExecutor() as executor:
        # Collect all tasks first for true parallel execution
        all_tasks = list(tasks)
        print(
            f"[{time.time() - execution_start:.3f}s] 📋 Collected {len(all_tasks)} tasks for parallel execution"
        )

        # Process all tasks for parallel execution
        for task in all_tasks:
            deps = task.dependencies
            task_names[task.idx] = (
                task.tool if isinstance(task.tool, str) else task.tool.name
            )
            args_for_tasks[task.idx] = task.args

            # Check if task can be executed immediately (no dependencies or all deps satisfied)
            if deps and (any([dep not in observations for dep in deps])):
                # Task has unsatisfied dependencies - queue it for later
                print(
                    f"[{time.time() - execution_start:.3f}s] ⏳ QUEUED task {task.idx}: {task.tool} (waiting for: {', '.join(map(str, deps))})"
                )
                futures.append(
                    executor.submit(
                        schedule_pending_task,
                        task,
                        observations,
                        tools_dict,
                        retry_after,
                    )
                )
            else:
                # Task can be executed immediately - dispatch to thread pool for parallel execution
                print(
                    f"[{time.time() - execution_start:.3f}s] 🚀 DISPATCHED task {task.idx}: {task.tool}"
                )
                futures.append(
                    executor.submit(
                        schedule_task,
                        dict(task=task, observations=observations, tools=tools_dict),
                        {"start_time": execution_start},
                    )
                )

        # Wait for all tasks to complete (both immediate and queued)
        print(
            f"[{time.time() - execution_start:.3f}s] ⏳ Task Fetching Unit: Waiting for all tasks to complete..."
        )
        wait(futures)
        print(
            f"[{time.time() - execution_start:.3f}s] ✅ Task Fetching Unit: All tasks completed!"
        )

    # Convert observations to tool messages
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
    planner = LLMCompilerPlanParser(tools)
    tasks = planner.stream(messages)

    # Eager execution: dispatch tasks as they're planned
    tasks = itertools.chain([next(tasks)], tasks)

    scheduled_tasks = schedule_tasks(
        {
            "messages": messages,
            "tasks": tasks,
            "tools": tools,
        }
    )
    return State(messages=scheduled_tasks, tools=tools)


def _joiner(state: State):
    print("🔗 Joining results...")

    messages = state.messages
    user_input = messages[0].content

    prompt = f"""Original request: {user_input}

Based on the completed tasks, provide a comprehensive response summarizing what was accomplished."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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

        for step in self.graph.stream(initial_state):
            yield step
