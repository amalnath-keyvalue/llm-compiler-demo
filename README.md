# LLMCompiler Demo

A complete implementation of the LLMCompiler architecture from the [LangGraph tutorial](https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/LLMCompiler/).

## 🛠️ Setup

**1. Set Python version:**
```bash
pyenv local 3.12
```
   Install pyenv: https://github.com/pyenv/pyenv#installation

**2. Install Poetry:**
```bash
pipx install poetry
```
   Install pipx: https://pypa.github.io/pipx/installation/

**3. Install dependencies:**
```bash
poetry install
```

**4. Set OpenAI API key:**
```bash
cp env.example .env
# Edit .env with your API key
```

**5. Run LLMCompiler:**

***Main Application:***
```bash
poetry run python -m src.main
```

With a custom query:
```bash
poetry run python -m src.main --query "Your query here"
```

***Checkpoints (Step-by-step Implementation):***

Run all checkpoints:
```bash
poetry run python -m src.checkpoints.main
```

Run a specific checkpoint (1-5):
```bash
poetry run python -m src.checkpoints.main 3
```

Run a specific checkpoint with custom query:
```bash
poetry run python -m src.checkpoints.main 3 --query "Your query here"
```
