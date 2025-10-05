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
- **Task Fetching Unit**: Dispatches tasks immediately when dependencies are met
- **Joiner**: Summarizes results and determines completion

## 🛠️ Setup

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
