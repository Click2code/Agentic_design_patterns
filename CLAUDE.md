# CLAUDE.md

## Project Overview

This repository demonstrates **Agentic AI Design Patterns** — implementations of common architectural patterns used when building AI agent systems. The examples use LangChain with OpenAI models.

## Repository Structure

```
Agentic_design_patterns/
├── prompt_chaining.py   # Prompt chaining pattern: multi-step LLM pipelines using LCEL
├── README.md
└── CLAUDE.md
```

## Tech Stack

- **Language**: Python 3
- **LLM Framework**: LangChain (langchain_openai, langchain_core)
- **LLM Provider**: OpenAI (via `ChatOpenAI`)
- **Environment Management**: python-dotenv (`.env` file for API keys)

## Key Dependencies

- `langchain-openai` — OpenAI chat model integration
- `langchain-core` — Prompts, output parsers, LCEL chain composition
- `python-dotenv` — Loads environment variables from `.env`

## Design Patterns Implemented

### Prompt Chaining (`prompt_chaining.py`)
Uses LangChain Expression Language (LCEL) to chain multiple prompts sequentially. The output of one LLM call feeds into the next:
1. **Extract** technical specifications from unstructured text
2. **Transform** extracted specs into structured JSON

Key LCEL operators: `|` (pipe) for chaining, `{}` dict syntax for variable mapping between chain steps.

## Development Conventions

- **Environment variables**: Store API keys (e.g., `OPENAI_API_KEY`) in a `.env` file at the project root. Never commit `.env` files.
- **LLM configuration**: Use `temperature=0` for deterministic, reproducible outputs.
- **Chain composition**: Use LCEL pipe (`|`) syntax with `StrOutputParser()` for string output between chain steps.
- **Naming**: Use descriptive names for prompts (`prompt_extract`, `prompt_transform`) and chains (`extraction_chain`, `full_chain`).

## Running

```bash
# Install dependencies
pip install langchain-openai langchain-core python-dotenv

# Set up environment
echo "OPENAI_API_KEY=your-key-here" > .env

# Run a pattern example
python prompt_chaining.py
```

## Git Workflow

- **Default branch**: `master`
- Commit messages should be concise and descriptive
