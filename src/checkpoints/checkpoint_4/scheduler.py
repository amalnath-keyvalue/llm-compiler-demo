import re
import time
from typing import Any, Generator

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..checkpoint_3 import Task


class SimpleScheduler:
    def schedule_tasks(
        self,
        tasks: Generator[Task | None, None, None],
        messages: list[BaseMessage],
        tools: list[BaseTool],
        execution_start: float,
    ):
        print(
            f"[{time.time() - execution_start:.3f}s] 🚀 SCHEDULER: Started scheduling tasks"
        )
        tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        task_results: dict[int, Any] = {
            message.tool_call_id: message.content
            for message in messages
            if isinstance(message, ToolMessage)
        }

        processed_tasks: list[Task] = []
        for task in tasks:
            if task is None:
                continue

            processed_tasks.append(task)

            if task.dependencies and any(
                dep not in task_results for dep in task.dependencies
            ):
                print(
                    f"[{time.time() - execution_start:.3f}s] ⏳ SCHEDULER: Queued {task.idx}: {task.tool} "
                    f"(waiting for: {', '.join(map(str, task.dependencies))})"
                )
                self._queue_task(
                    task=task,
                    task_results=task_results,
                    tools=tools_dict,
                    execution_start=execution_start,
                )
            else:
                print(
                    f"[{time.time() - execution_start:.3f}s] 📤 SCHEDULER: Dispatched {task.idx}: {task.tool}"
                )
                self._start_task(
                    task=task,
                    task_results=task_results,
                    tools=tools_dict,
                    execution_start=execution_start,
                )

        return [
            ToolMessage(
                name=task.tool,
                content=str(task_results[task.idx]),
                tool_call_id=task.idx,
                additional_kwargs={"args": task.args},
            )
            for task in processed_tasks
            if task.idx in task_results
        ]

    def _queue_task(
        self,
        task: Task,
        task_results: dict[int, Any],
        tools: dict[str, BaseTool],
        execution_start: float,
        retry_after: float = 0.2,
    ):
        while True:
            if task.dependencies and any(
                dep not in task_results for dep in task.dependencies
            ):
                time.sleep(retry_after)
                continue

            self._start_task(
                task=task,
                task_results=task_results,
                tools=tools,
                execution_start=execution_start,
            )
            break

    def _start_task(
        self,
        task: Task,
        task_results: dict[int, Any],
        tools: dict[str, BaseTool],
        execution_start: float,
    ):
        print(
            f"[{time.time() - execution_start:.3f}s] 🚀 SCHEDULER: Started task {task.idx}: {task.tool}"
        )

        def resolve_arg(
            arg: str,
            task_results: dict[int, Any],
        ):
            id_pattern = r"\$\{?(\d+)\}?"

            def replace_match(match: re.Match[str]):
                idx = int(match.group(1))
                return str(task_results.get(idx, match.group(0)))

            return re.sub(id_pattern, replace_match, arg)

        try:
            tool_to_execute = tools[task.tool]
            resolved_args = {
                key: resolve_arg(val, task_results) for key, val in task.args.items()
            }
            task_results[task.idx] = tool_to_execute.invoke(resolved_args)

        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            task_results[task.idx] = error_msg
            print(
                f"[{time.time() - execution_start:.3f}s] ❌ FAILED task {task.idx}: {task.tool} - {error_msg}"
            )
            return

        print(
            f"[{time.time() - execution_start:.3f}s] ✅ SCHEDULER: Completed task {task.idx}: {task.tool}"
        )
