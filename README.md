# LLMCompiler Demo

A minimal demonstration of LLMCompiler architecture using LangGraph. Shows eager execution of tasks in a DAG with parallel processing.

## Setup

1. Install Poetry:
```bash
pip install poetry
```

2. Install dependencies:
```bash
poetry install
```

3. Set OpenAI API key:
```bash
cp env.example .env
# Edit .env with your API key
```

4. Run demo:
```bash
poetry run python demo.py
```

## Architecture

- **Planner**: Creates a DAG of tasks with dependencies
- **Task Scheduler**: Executes all ready tasks in parallel
- **Joiner**: Determines completion and summarizes results

## Key Features

- **DAG Planning**: Creates dependency graphs upfront
- **Eager Execution**: Runs tasks in parallel when dependencies are met
- **Tool Agnostic**: Works with any LangChain tools
- **Circular Dependency Detection**: Validates DAG structure
- **Gitignored Output**: All scaffolding goes to `demo_output/` folder

## Files

- `llm_compiler.py` - Generic LLMCompiler framework
- `scaffolding_tools.py` - Project scaffolding tools
- `demo.py` - Example usage with scaffolding

