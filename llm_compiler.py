import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Task(BaseModel):
    id: str = Field(description="Unique identifier for the task")
    tool: str = Field(description="Name of the tool to execute")
    args: dict[str, Any] = Field(description="Arguments to pass to the tool")
    dependencies: list[str] = Field(description="List of task IDs this task depends on")


class TaskPlan(BaseModel):
    tasks: list[Task] = Field(description="List of tasks to execute")


class Planner:
    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list[BaseTool],
    ):
        self.llm = llm
        self.tools = tools
        self.parser = PydanticOutputParser(pydantic_object=TaskPlan)

    def plan(
        self,
        user_input: str,
    ) -> list[Task]:
        logger.info("Planning: %s", user_input)

        prompt = f"""Break down this request into a sequence of tasks: {user_input}

Available tools:
{self._get_tool_descriptions()}

Create a plan with tasks that can run in parallel when possible. Tasks with no dependencies can run immediately, while tasks with dependencies wait for those to complete.

{self.parser.get_format_instructions()}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            task_plan = self.parser.parse(response.content)
            logger.info("Generated %d tasks", len(task_plan.tasks))
            return task_plan.tasks
        except Exception as e:
            logger.error("Failed to parse plan: %s", str(e))
            return []

    def _get_tool_descriptions(
        self,
    ) -> str:
        descriptions = []
        for tool in self.tools:
            schema = tool.get_input_schema().model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            params = []
            for param, details in properties.items():
                param_type = details.get("type", "unknown")
                required_str = " (required)" if param in required else " (optional)"
                params.append(f"    {param} ({param_type}){required_str}")

            param_str = "\n".join(params) if params else "    No parameters"
            descriptions.append(f"- {tool.name}: {tool.description}\n{param_str}")

        return "\n".join(descriptions)


class TaskScheduler:
    def __init__(
        self,
        tools: list[BaseTool],
    ):
        self.tools = {tool.name: tool for tool in tools}

    def execute_dag(self, tasks: list[Task]) -> dict[str, Any]:
        if not tasks:
            return {}

        results = {}
        completed = set()

        while len(completed) < len(tasks):
            ready_tasks = self._get_ready_tasks(tasks, completed)

            if not ready_tasks:
                logger.error("No ready tasks found - possible circular dependency")
                break

            logger.info("Executing %d tasks in parallel:", len(ready_tasks))
            for task in ready_tasks:
                logger.info("  %s: %s", task.id, task.tool)

            batch_results = self._execute_batch(ready_tasks)
            results.update(batch_results)
            completed.update(task.id for task in ready_tasks)

        return results

    def _get_ready_tasks(
        self,
        tasks: list[Task],
        completed: set,
    ) -> list[Task]:
        ready = []
        for task in tasks:
            if task.id in completed:
                continue
            if all(dep in completed for dep in task.dependencies):
                ready.append(task)
        return ready

    def _execute_batch(
        self,
        tasks: list[Task],
    ) -> dict[str, Any]:
        results = {}
        for task in tasks:
            if task.tool in self.tools:
                logger.info("  → %s", task.tool)
                try:
                    result = self.tools[task.tool].invoke(task.args)
                    results[task.id] = result
                    logger.info("  ✓ %s", task.id)
                except Exception as e:
                    results[task.id] = f"Error: {str(e)}"
                    logger.error("  ✗ %s: %s", task.id, str(e))
            else:
                results[task.id] = f"Error: Tool {task.tool} not found"
                logger.error("  ✗ %s: Tool not found", task.id)
        return results


class Joiner:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm

    def join(
        self,
        user_input: str,
        results: dict[str, Any],
    ) -> str:
        if not results:
            return "No results to process"

        logger.info("Joining results")

        results_summary = "\n".join(
            [f"Task {task_id}: {result}" for task_id, result in results.items()]
        )

        prompt = f"""Original request: {user_input}

Task results:
{results_summary}

Provide a comprehensive response based on these results."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class LLMCompiler:
    def __init__(
        self,
        tools: list[BaseTool],
    ):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = tools
        self.planner = Planner(self.llm, tools)
        self.scheduler = TaskScheduler(tools)
        self.joiner = Joiner(self.llm)

    def run(
        self,
        user_input: str,
    ) -> str:
        tasks = self.planner.plan(user_input)
        results = self.scheduler.execute_dag(tasks)
        return self.joiner.join(user_input, results)
