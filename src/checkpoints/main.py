import argparse
import asyncio
import os

from dotenv import load_dotenv

from .checkpoint_1 import BaseLLMCompiler
from .checkpoint_2 import LLMCompilerWithSimplePlannerOnly
from .checkpoint_3 import LLMCompilerWithPlannerOnly
from .checkpoint_4 import LLMCompilerWithPlannerAndSimpleSchedulerOnly
from .checkpoint_5 import LLMCompilerWithPlannerAndSchedulerOnly
from .tools import get_tools

load_dotenv()


async def main(
    checkpoint: int | None = None,
    query: str | None = None,
):
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    tools = get_tools()

    if query is None:
        query = "Create a counter app with a button to increment and display the count using React and Tailwind CSS"

    checkpoints: dict[int, tuple[type[BaseLLMCompiler], str]] = {
        1: (BaseLLMCompiler, "🚀 RUNNING BaseLLMCompiler"),
        2: (
            LLMCompilerWithSimplePlannerOnly,
            "📋 RUNNING LLMCompilerWithSimplePlannerOnly",
        ),
        3: (LLMCompilerWithPlannerOnly, "📝 RUNNING LLMCompilerWithPlannerOnly"),
        4: (
            LLMCompilerWithPlannerAndSimpleSchedulerOnly,
            "⏰ RUNNING LLMCompilerWithPlannerAndSimpleSchedulerOnly",
        ),
        5: (
            LLMCompilerWithPlannerAndSchedulerOnly,
            "⚙️ RUNNING LLMCompilerWithPlannerAndSchedulerOnly",
        ),
    }

    if checkpoint:
        if checkpoint not in checkpoints:
            print(
                f"Error: Checkpoint {checkpoint} not found. Available checkpoints: 1-5"
            )
            return
        compiler_class, message = checkpoints[checkpoint]
        print(message)
        compiler = compiler_class(tools)
        await compiler.run(query)

    else:
        for cp_num in sorted(checkpoints.keys()):
            compiler_class, message = checkpoints[cp_num]
            print(message)
            compiler = compiler_class(tools)
            await compiler.run(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Compiler checkpoints")
    parser.add_argument(
        "checkpoint",
        type=int,
        nargs="?",
        choices=[1, 2, 3, 4, 5],
        help="Checkpoint number to run (1-5). If not provided, runs all checkpoints.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Custom query to run. If not provided, uses default counter app query.",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            checkpoint=args.checkpoint,
            query=args.query,
        )
    )
