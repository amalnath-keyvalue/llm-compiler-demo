import asyncio
import os

from dotenv import load_dotenv

from src.llm_compiler import LLMCompiler
from src.scaffolding import get_tools

load_dotenv()


async def run_example(
    compiler: LLMCompiler,
    example: str,
    example_num: int,
):
    print(f"Example {example_num}: {example}")
    print("-" * 50)

    result = await compiler.run(example)
    # print(result)

    print("=" * 60)
    print()


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    tools = get_tools()
    compiler = LLMCompiler(tools)

    print("=== LLMCompiler Project Scaffolding Demo ===\n")

    examples = [
        "Create a counter app with a button to increment and display the count using React and Tailwind"
    ]

    for i, example in enumerate(examples, 1):
        await run_example(compiler, example, i)


if __name__ == "__main__":
    asyncio.run(main())
