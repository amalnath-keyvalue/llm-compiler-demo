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


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    tools = get_tools()
    compiler1 = BaseLLMCompiler(tools)
    compiler2 = LLMCompilerWithSimplePlannerOnly(tools)
    compiler3 = LLMCompilerWithPlannerOnly(tools)
    compiler4 = LLMCompilerWithPlannerAndSimpleSchedulerOnly(tools)
    compiler5 = LLMCompilerWithPlannerAndSchedulerOnly(tools)

    query = "Create a counter app with a button to increment and display the count using React and Tailwind CSS"
    print("🚀 RUNNING BaseLLMCompiler")
    await compiler1.run(query)
    print("📋 RUNNING LLMCompilerWithSimplePlannerOnly")
    await compiler2.run(query)
    print("📝 RUNNING LLMCompilerWithPlannerOnly")
    await compiler3.run(query)
    print("⏰ RUNNING LLMCompilerWithPlannerAndSimpleSchedulerOnly")
    await compiler4.run(query)
    print("⚙️ RUNNING LLMCompilerWithPlannerAndSchedulerOnly")
    await compiler5.run(query)


if __name__ == "__main__":
    asyncio.run(main())
