import os
import time

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


@tool
def create_directory(
    path: str,
) -> str:
    """Create a directory structure. Use this before creating files inside directories.
    IMPORTANT: This tool should typically be used before create_file when files need to be placed in specific directories.
    """
    time.sleep(1)  # arbitrary delay to simulate long operation
    if not path.startswith("generated_projects/"):
        path = f"generated_projects/{path}"

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
    """

    class Response(BaseModel):
        file_content: str

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
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
    IMPORTANT: Always use output from generate_file_content tasks instead of hardcoding content.
    """
    time.sleep(1)  # arbitrary delay to simulate long operation
    if not path.startswith("generated_projects/"):
        path = f"generated_projects/{path}"

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
