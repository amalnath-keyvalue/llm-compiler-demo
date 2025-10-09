import asyncio
import os

from dotenv import load_dotenv

from src.llm_compiler import LLMCompiler
from src.scaffolding import get_tools

load_dotenv()


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    tools = get_tools()
    llm_compiler = LLMCompiler(tools)

    query = "Create a counter app with a button to increment and display the count using React and Tailwind CSS"
    await llm_compiler.run(query)


if __name__ == "__main__":
    asyncio.run(main())
