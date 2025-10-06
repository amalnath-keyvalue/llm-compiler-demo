import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.tools import BaseTool

from .planner import Task


class Scheduler:
    def schedule_tasks(
        self,
        scheduler_input: dict[str, Any],
    ) -> list[FunctionMessage]:
        """Scheduler - schedules and executes tasks as soon as they are executable"""
        tasks: list[Task] = scheduler_input["tasks"]
        messages: list[BaseMessage] = scheduler_input["messages"]
        tools: list[BaseTool] = scheduler_input["tools"]
        execution_start: float = scheduler_input["execution_start"]
        task_results: dict[int, Any] = self._get_task_results(messages)
        tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        futures: list[Future[Any]] = []

        print(f"[{time.time() - execution_start:.3f}s] 🚀 Scheduler: Initialized")

        with ThreadPoolExecutor() as executor:
            processed_tasks = []
            for task in tasks:
                processed_tasks.append(task)
                deps = task.dependencies

                if deps and any(dep not in task_results for dep in deps):
                    print(
                        f"[{time.time() - execution_start:.3f}s] ⏳ QUEUED {task.idx}: {task.tool} "
                        f"(waiting for: {', '.join(map(str, deps))})"
                    )
                    futures.append(
                        executor.submit(
                            self._schedule_pending_task,
                            task,
                            task_results,
                            tools_dict,
                            execution_start,
                        )
                    )
                else:
                    print(
                        f"[{time.time() - execution_start:.3f}s] 🚀 DISPATCHED task {task.idx}: {task.tool}"
                    )
                    futures.append(
                        executor.submit(
                            self._schedule_task,
                            {
                                "task": task,
                                "task_results": task_results,
                                "tools": tools_dict,
                            },
                            {"start_time": execution_start},
                        )
                    )

            print(
                f"[{time.time() - execution_start:.3f}s] ⏳ Scheduler: Waiting for all tasks to complete..."
            )
            wait(futures)
            print(
                f"[{time.time() - execution_start:.3f}s] ✅ Scheduler: All tasks completed!"
            )

            tool_messages = []
            for task in processed_tasks:
                if task.idx in task_results:
                    tool_messages.append(
                        FunctionMessage(
                            name=task.tool,
                            content=str(task_results[task.idx]),
                            additional_kwargs={"idx": task.idx, "args": task.args},
                            tool_call_id=task.idx,
                        )
                    )
            return tool_messages

    def _resolve_arg(
        self,
        arg: str,
        task_results: dict[int, Any],
    ) -> str:
        """Resolve $N references in task arguments"""
        id_pattern = r"\$\{?(\d+)\}?"

        def replace_match(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            return str(task_results.get(idx, match.group(0)))

        return re.sub(id_pattern, replace_match, arg)

    def _execute_task(
        self,
        task: Task,
        task_results: dict[int, Any],
        config: dict[str, Any],
        tools: dict[str, BaseTool],
    ):
        """Execute a single task"""
        tool_to_use = tools[task.tool]
        resolved_args = {
            key: self._resolve_arg(val, task_results) for key, val in task.args.items()
        }
        return tool_to_use.invoke(resolved_args, config)

    def _schedule_task(self, task_inputs: dict[str, Any], config: dict[str, Any]):
        """Schedule and execute a single task"""
        task: Task = task_inputs["task"]
        task_results: dict[int, Any] = task_inputs["task_results"]
        tools: dict[str, BaseTool] = task_inputs["tools"]

        execution_start: float = config.get("start_time", time.time())
        current_time = time.time()
        print(
            f"[{current_time - execution_start:.3f}s] 🚀 STARTED task {task.idx}: {task.tool}"
        )

        try:
            result = self._execute_task(task, task_results, config, tools)
            task_results[task.idx] = result
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            task_results[task.idx] = error_msg
            print(
                f"[{time.time() - execution_start:.3f}s] ❌ FAILED task {task.idx}: {task.tool} - {error_msg}"
            )
            return

        end_time = time.time()
        print(
            f"[{end_time - execution_start:.3f}s] ✅ COMPLETED task {task.idx}: {task.tool}"
        )

    def _schedule_pending_task(
        self,
        task: Task,
        task_results: dict[int, Any],
        tools: dict[str, BaseTool],
        execution_start: float,
        retry_after: float = 0.2,
    ):
        """Schedule a task that's waiting for dependencies"""
        current_time = time.time()
        print(
            f"[{current_time - execution_start:.3f}s] ⏳ WAITING task {task.idx}: {task.tool} (deps: {task.dependencies})"
        )

        while True:
            deps = task.dependencies
            if deps and any(dep not in task_results for dep in deps):
                time.sleep(retry_after)
                continue
            self._schedule_task(
                {"task": task, "task_results": task_results, "tools": tools},
                {"start_time": execution_start},
            )
            break

    def _get_task_results(
        self,
        messages: list[BaseMessage],
    ) -> dict[int, Any]:
        """Extract task results from messages"""
        task_results: dict[int, Any] = {}
        for message in messages:
            if (
                hasattr(message, "additional_kwargs")
                and "idx" in message.additional_kwargs
            ):
                idx = message.additional_kwargs["idx"]
                task_results[idx] = message.content
        return task_results
