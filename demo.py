import os

from dotenv import load_dotenv

from llm_compiler import LLMCompiler
from scaffolding_tools import get_scaffolding_tools

load_dotenv()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize LLMCompiler with scaffolding tools
    tools = get_scaffolding_tools()
    compiler = LLMCompiler(tools)

    print("=== LLMCompiler Project Scaffolding Demo ===\n")

    examples = [
        "Create a Python Flask web application with basic structure",
        "Set up a React TypeScript project with components",
        "Create a FastAPI microservice with database models",
    ]

    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example}")
        print("-" * 50)

        result = compiler.run(example)
        print(f"Result: {result}")

        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
