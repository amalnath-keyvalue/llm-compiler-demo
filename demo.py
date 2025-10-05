import asyncio
import os

from dotenv import load_dotenv

from llm_compiler import LLMCompiler
from tools import get_tools

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
        "Create a web project with the following structure:\n"
        "- Create a project directory called 'my_web_app'\n"
        "- Generate Python code for a Flask web server\n"
        "- Generate HTML code for a homepage\n"
        "- Generate CSS code for styling\n"
        "- Create a file called 'app.py' with the Flask code from step 2\n"
        "- Create a file called 'index.html' with the HTML code from step 3\n"
        "- Create a file called 'style.css' with the CSS code from step 4\n"
        "- Create a README.md file with project documentation\n\n"
    ]

    for i, example in enumerate(examples, 1):
        await run_example(compiler, example, i)


if __name__ == "__main__":
    asyncio.run(main())
