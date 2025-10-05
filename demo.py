import asyncio
import os

from dotenv import load_dotenv

from llm_compiler import LLMCompiler
from scaffolding_tools import get_scaffolding_tools

load_dotenv()


async def run_example(compiler, example, example_num):
    print(f"Example {example_num}: {example}")
    print("-" * 50)

    async for progress in compiler.run(example):
        print(f"  {progress}")

    print("=" * 60)
    print()


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize LLMCompiler with scaffolding tools
    tools = get_scaffolding_tools()
    compiler = LLMCompiler(tools)

    print("=== LLMCompiler Project Scaffolding Demo ===\n")

    examples = [
        "🚀 COMPREHENSIVE LLMCompiler Demo - EAGER + PARALLEL + DAG\n"
        "Create a web project with the following structure:\n"
        "1. Create a project directory called 'my_web_app'\n"
        "2. Generate Python code for a Flask web server\n"
        "3. Generate HTML code for a homepage\n"
        "4. Generate CSS code for styling\n"
        "5. Create a file called 'app.py' with the Flask code from step 2\n"
        "6. Create a file called 'index.html' with the HTML code from step 3\n"
        "7. Create a file called 'style.css' with the CSS code from step 4\n"
        "8. Create a README.md file with project documentation\n\n"
        "The files should be created in the project directory and can reference each other.\n"
        "Watch the logs to see EAGER execution, PARALLEL processing, and DAG dependencies working together!"
    ]

    for i, example in enumerate(examples, 1):
        await run_example(compiler, example, i)


if __name__ == "__main__":
    asyncio.run(main())
