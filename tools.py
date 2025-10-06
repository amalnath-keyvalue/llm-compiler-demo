import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel

from config import get_llm


@tool
def create_file(
    path: str,
    content: str,
) -> str:
    """Create a file with specified content in the demo_output folder"""
    if not path.startswith("demo_output/"):
        path = f"demo_output/{path}"

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created file: {path}"


@tool
def create_directory(
    path: str,
) -> str:
    """Create a directory in the demo_output folder"""
    if not path.startswith("demo_output/"):
        path = f"demo_output/{path}"

    os.makedirs(path, exist_ok=True)
    return f"Created directory: {path}"


@tool
def generate_code(
    description: str,
    language: str,
    context: list[str] | None = None,
) -> str:
    """Generate code based on description"""

    class CodeResponse(BaseModel):
        code: str

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=CodeResponse)

    prompt = f"Generate {language} code for: {description}"
    if context:
        prompt += f"\nContext: {' '.join(context)}"

    prompt += f"\n{parser.get_format_instructions()}"

    response = llm.invoke(prompt)
    parsed = parser.parse(response.content)
    return parsed.code


def get_tools():
    return [
        create_file,
        create_directory,
        generate_code,
    ]
