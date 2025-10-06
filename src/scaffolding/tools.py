import os
import time

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel

from config import get_llm


@tool
def create_directory(
    path: str,
) -> str:
    """Create a directory structure. Use this before creating files inside directories.
    This tool should typically be used before create_file when files need to be placed in specific directories.
    """
    time.sleep(3)
    if not path.startswith("demo_output/"):
        path = f"demo_output/{path}"

    os.makedirs(path, exist_ok=True)
    return f"Created directory: {path}"


@tool
def generate_file_content(
    description: str,
    content_type: str,
    context: str | None = None,
) -> str:
    """Generate file content based on description. Use this to create content that will be used by other tasks.
    IMPORTANT: Always use this tool to generate content instead of hardcoding it in create_file tasks.
    The output of this tool should typically be referenced by create_file tasks using $N syntax.
    """

    class Response(BaseModel):
        file_content: str

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=Response)

    prompt = f"Generate {content_type} content for: {description}"
    if context:
        prompt += f"\nContext: {context}"

    prompt += f"\n{parser.get_format_instructions()}"

    response = llm.invoke(prompt)
    parsed = parser.parse(response.content)
    return parsed.file_content


@tool
def create_file(
    path: str,
    content: str,
) -> str:
    """Create a file with specified content. Use this to save generated content to files.
    IMPORTANT: Always use $N syntax to reference output from generate_file_content tasks instead of hardcoding content.
    The content parameter can reference output from previous tasks using $N syntax."""
    time.sleep(3)
    if not path.startswith("demo_output/"):
        path = f"demo_output/{path}"

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created file: {path}"


def get_tools():
    return [
        create_directory,
        generate_file_content,
        create_file,
    ]
