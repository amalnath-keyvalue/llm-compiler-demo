import concurrent.futures
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Task(BaseModel):
    tool: str = Field(description="Name of the tool to execute")
    args: dict[str, Any] = Field(description="Arguments to pass to the tool")
    dependencies: list[str] = Field(
        default_factory=list, description="List of task IDs this task depends on"
    )


class TaskDAG(BaseModel):
    tasks: dict[str, Task] = Field(
        description="Dictionary of tasks with their IDs as keys"
    )


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    task_dag: TaskDAG | None = None
    completed_tasks: list[str] = []
    results: dict[str, str] = {}


class LLMCompiler:
    def __init__(self, tools: list[BaseTool]) -> None:
        self.llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools: list[BaseTool] = tools
        self.tool_map: dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("planner", self._planner)
        graph.add_node("task_scheduler", self._task_scheduler)
        graph.add_node("joiner", self._joiner)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "task_scheduler")
        graph.add_edge("task_scheduler", "joiner")
        graph.add_conditional_edges("joiner", self._should_continue)

        return graph.compile()

    def _planner(
        self,
        state: State,
    ):
        user_input = state.messages[-1].content
        tool_descriptions = [
            f"- {tool.name}: {tool.description}" for tool in self.tools
        ]

        parser = PydanticOutputParser(pydantic_object=TaskDAG)

        plan_prompt = f"""Create a DAG of tasks for: {user_input}
        
Available tools:
{chr(10).join(tool_descriptions)}

Key points:
- Use only available tools
- Tasks with no dependencies can run immediately
- Tasks with dependencies wait for those to complete
- Multiple tasks can run in parallel if their dependencies are satisfied

{parser.get_format_instructions()}"""

        response = self.llm.invoke([HumanMessage(content=plan_prompt)])

        try:
            task_dag = parser.parse(response.content)

            if self._validate_dag(task_dag):
                return {
                    "messages": [
                        AIMessage(
                            content=f"Task DAG created with {len(task_dag.tasks)} tasks"
                        )
                    ],
                    "task_dag": task_dag,
                }

            return {"messages": [AIMessage(content="Invalid DAG structure")]}

        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Failed to create task DAG: {str(e)}")]
            }

    def _validate_dag(
        self,
        task_dag: TaskDAG,
    ):
        tasks = task_dag.tasks
        if not tasks:
            return False

        def has_cycle(task_id: str, visited: set[str], rec_stack: set[str]) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = tasks.get(task_id)
            if not task:
                return True

            deps = task.dependencies

            for dep in deps:
                if dep not in tasks:
                    return True
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(task_id)
            return False

        visited = set()
        for task_id in tasks:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    return False

        for task_id, task in tasks.items():
            if task.tool not in self.tool_map:
                return False

        return True

    def _task_scheduler(
        self,
        state: State,
    ):
        if not state.task_dag:
            return {"messages": [AIMessage(content="No tasks to execute")]}

        tasks = state.task_dag.tasks
        completed = set(state.completed_tasks)
        results = state.results.copy()

        executable = []
        for task_id, task in tasks.items():
            if task_id not in completed:
                deps = task.dependencies
                if all(dep in completed for dep in deps):
                    executable.append((task_id, task))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {
                executor.submit(self._execute_task, task): task_id
                for task_id, task in executable
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    results[task_id] = result
                    completed.add(task_id)
                except Exception as e:
                    results[task_id] = f"Error: {str(e)}"
                    completed.add(task_id)

        return {
            "messages": [
                AIMessage(content=f"Executed {len(executable)} tasks in parallel")
            ],
            "completed_tasks": list(completed),
            "results": results,
        }

    def _execute_task(
        self,
        task: Task,
    ):
        if task.tool not in self.tool_map:
            raise ValueError(f"Unknown tool: {task.tool}")

        tool = self.tool_map[task.tool]
        return tool.invoke(task.args)

    def _joiner(
        self,
        state: State,
    ):
        tasks = state.task_dag.tasks
        completed = set(state.completed_tasks)

        if len(completed) == len(tasks):
            summary = "All tasks completed successfully!"
        else:
            remaining = len(tasks) - len(completed)
            summary = (
                f"Completed {len(completed)}/{len(tasks)} tasks. {remaining} remaining."
            )

        return {"messages": [AIMessage(content=summary)]}

    def _should_continue(
        self,
        state: State,
    ):
        tasks = state.task_dag.tasks
        completed = set(state.completed_tasks)

        if len(completed) < len(tasks):
            return "task_scheduler"
        return END

    def run(
        self,
        user_input: str,
    ):
        initial_state = State(messages=[HumanMessage(content=user_input)])
        return self.graph.stream(initial_state)
