import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Generator

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..checkpoint_3 import Task
from ..checkpoint_4 import SimpleScheduler


class Scheduler(SimpleScheduler):
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

        futures = []
        processed_tasks: list[Task] = []
        with ThreadPoolExecutor() as executor:
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
                    futures.append(
                        executor.submit(
                            self._queue_task,
                            task=task,
                            task_results=task_results,
                            tools=tools_dict,
                            execution_start=execution_start,
                        )
                    )
                else:
                    print(
                        f"[{time.time() - execution_start:.3f}s] 📤 SCHEDULER: Dispatched task {task.idx}: {task.tool}"
                    )
                    futures.append(
                        executor.submit(
                            self._start_task,
                            task=task,
                            task_results=task_results,
                            tools=tools_dict,
                            execution_start=execution_start,
                        )
                    )

            print(
                f"[{time.time() - execution_start:.3f}s] ⏳ SCHEDULER: Waiting for all tasks to complete..."
            )
            wait(futures)
            print(
                f"[{time.time() - execution_start:.3f}s] ✅ SCHEDULER: All tasks completed!"
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
