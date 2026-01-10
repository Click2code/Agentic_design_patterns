# Agentic Design Patterns

A collection of design patterns for building AI agents with Large Language Models (LLMs). This repository provides practical implementations and examples of common patterns used in agentic AI systems.

## What are Agentic Design Patterns?

Agentic design patterns are reusable approaches for building autonomous AI agents that can:
- Reason about complex problems
- Use tools and APIs to gather information
- Plan and execute multi-step tasks
- Reflect on their performance and improve
- Coordinate with other agents

## Patterns Included

### 1. **ReAct Pattern** (Reasoning + Acting)
Combines reasoning traces with action execution. The agent alternates between thinking about what to do and taking actions, creating a clear chain of thought.

**Use cases:** Question answering with tool use, task automation, information retrieval

### 2. **Tool Use Pattern**
Enables agents to interact with external tools, APIs, and functions to extend their capabilities beyond text generation.

**Use cases:** Calculator operations, web search, database queries, API interactions

### 3. **Planning Pattern**
Breaks down complex tasks into smaller, manageable steps before execution.

**Use cases:** Multi-step workflows, project planning, complex problem solving

### 4. **Reflection Pattern**
Allows agents to review and critique their own outputs, iteratively improving results.

**Use cases:** Code review, content refinement, error correction

### 5. **Multi-Agent Pattern**
Coordinates multiple specialized agents to solve complex problems collaboratively.

**Use cases:** Debate systems, task delegation, specialized expertise coordination

## Project Structure

```
Agentic_design_patterns/
├── patterns/
│   ├── react/              # ReAct pattern implementation
│   ├── tool_use/           # Tool use pattern implementation
│   ├── planning/           # Planning pattern implementation
│   ├── reflection/         # Reflection pattern implementation
│   └── multi_agent/        # Multi-agent pattern implementation
├── examples/               # Practical examples
└── utils/                  # Shared utilities
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Examples

Each pattern directory contains example implementations:

```bash
# Run ReAct pattern example
python patterns/react/example.py

# Run Tool Use pattern example
python patterns/tool_use/example.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Building LLM-powered Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Anthropic's Guide to Building with Claude](https://docs.anthropic.com/)