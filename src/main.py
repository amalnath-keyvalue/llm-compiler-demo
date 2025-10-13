import argparse
import asyncio
import os

from dotenv import load_dotenv

from .llm_compiler import LLMCompiler
from .scaffolding import get_tools

load_dotenv()


async def main(
    query: str | None = None,
):
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    tools = get_tools()
    llm_compiler = LLMCompiler(tools)

    if query is None:
        query = "Create a counter app with a button to increment and display the count using React and Tailwind CSS"

    await llm_compiler.run(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Compiler")
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run. If not provided, uses default counter app query.",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            query=args.query,
        )
    )
