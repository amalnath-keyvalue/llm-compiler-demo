import re
import time

from ..checkpoint_2 import SimplePlanner
from .schemas import Task


class Planner(SimplePlanner):
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
                f"[{time.time() - execution_start:.3f}s] 📋 PLANNER: Planned task {task.idx}: {task.tool}({task.args}){deps_str}"
            )
            return task

    def _parse_task(
        self,
        line: str,
    ):
        line = line.strip()
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
