# LangGraph Code Debugger

This project scaffolds a multi-agent code debugging assistant powered by LangGraph and a retrieval-augmented generation (RAG) pipeline. The system indexes a local codebase, retrieves relevant context for an error, and coordinates multiple agents to deliver plain-English diagnostics with actionable fixes.

## Features

- **Codebase vectorization**: Walks a repository, chunks source files, and stores embeddings in a persistent Chroma vector store.
- **Retrieval-augmented reasoning**: Fetches the most relevant code snippets for an input error or stack trace.
- **Multi-agent workflow**: LangGraph orchestrates specialized agents that diagnose the issue and suggest at least two remediation strategies.
- **CLI entry point**: Run end-to-end indexing and debugging from the terminal.

## Project structure

```
src/
  agents/
    error_explainer.py
    solution_designer.py
    __init__.py
  retrieval/
    code_vectorizer.py
    __init__.py
  workflows/
    debugger_graph.py
    __init__.py
  config.py
  main.py
requirements.txt
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials**

   Export an LLM API key (e.g., OpenAI) that the agents will use for reasoning:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

   You can also populate a `.env` file with `OPENAI_API_KEY` and other integration keys; the CLI loads it automatically.

3. **Index a codebase and run the debugger**

   ```bash
   python -m src.main --codebase /path/to/repo --error "Traceback (most recent call last): ..."
   ```

   - Pass `--persist ./vector_store` to choose where embeddings are stored.
   - Add `--reindex` to force a fresh embedding rebuild.

## Extending

- Swap `OpenAIEmbeddings` with any LangChain-compatible embedding model.
- Add more agents to the LangGraph pipeline by updating `DebuggerWorkflow`.
- Integrate unit test suggestion agents or automatic patch generation for deeper automation.

## Notes

- Running the workflow requires network access for the selected LLM and embedding providers.
- Ensure that binary files or large assets are excluded from indexing by adjusting the `include_globs` and `exclude_globs` parameters in `CodebaseVectorizer`.

## Dev Container

This repository ships with a VS Code compatible dev container. Launching the project inside the container will provision a Python 3.11 environment with common tooling and automatically install `requirements.txt`. Make sure Docker is running, then open the folder in VS Code and choose **Reopen in Container**.
