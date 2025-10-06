# LLMCompiler Demo

A complete implementation of the LLMCompiler architecture from the [LangGraph tutorial](https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/).

## 🚀 Key Features

- **Faster Execution**: Tasks run in parallel as soon as dependencies are met, not sequentially
- **Cost Efficient**: Reduces redundant LLM calls by reusing task outputs with `$N` syntax
- **Flexible Tools**: Works with any LangChain tools - just swap them in
- **Dependency Management**: Automatically handles task dependencies and execution order
- **Real-time Progress**: See tasks execute immediately as they're planned

## 📋 Architecture

- **Planner**: Streams tasks from LLM using LangChain Hub prompts
- **Scheduler**: Dispatches tasks immediately when dependencies are met
- **Joiner**: Summarizes results and determines completion

## 🛠️ Setup

1. Set Python version:
```bash
pyenv local 3.12
```
   Install pyenv: https://github.com/pyenv/pyenv#installation

2. Install Poetry:
```bash
pipx install poetry
```
   Install pipx: https://pypa.github.io/pipx/installation/

3. Install dependencies:
```bash
poetry install
```

4. Set OpenAI API key:
```bash
cp env.example .env
# Edit .env with your API key
```

5. Run demo:
```bash
poetry run python demo.py
```
