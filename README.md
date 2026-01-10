# Agentic Design Patterns

A collection of agentic design patterns implemented with modern AI frameworks. This repository demonstrates various patterns for building complex AI workflows and multi-step reasoning systems.

## Patterns Included

### 1. Prompt Chaining

Prompt chaining is a design pattern where the output of one LLM call becomes the input to the next LLM call, creating a sequential pipeline of operations. This is useful for breaking down complex tasks into smaller, more manageable steps.

**Implementations:**
- `prompt_chaining.py` - Traditional implementation using direct API calls
- `prompt_chaining_langraph.py` - Modern implementation using LangGraph framework

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Usage

### Traditional Prompt Chaining

```bash
python prompt_chaining.py
```

This demonstrates a simple sequential chain:
1. Generate a story outline from a topic
2. Expand the outline into a full story
3. Extract themes and moral from the story

### LangGraph Prompt Chaining

```bash
python prompt_chaining_langraph.py
```

This demonstrates the same workflow using LangGraph's graph-based approach, which provides:
- Built-in state management
- Visual graph representation
- Better debugging capabilities
- Easy extensibility
- Support for conditional logic and cycles

## Comparison: Traditional vs LangGraph

### Traditional Approach (`prompt_chaining.py`)

**Pros:**
- Simple and straightforward
- Easy to understand for beginners
- Minimal dependencies
- Direct control over flow

**Cons:**
- Manual state management
- Harder to debug complex chains
- No built-in visualization
- Difficult to add conditional logic
- No persistence/checkpointing

### LangGraph Approach (`prompt_chaining_langraph.py`)

**Pros:**
- Built-in state management with TypedDict
- Automatic state passing between nodes
- Visual graph representation
- Easy to add conditional edges
- Support for cycles and complex flows
- Built-in persistence and checkpointing
- Better for production systems

**Cons:**
- Additional learning curve
- More dependencies
- Slightly more boilerplate

## Example Output

Both implementations process the same workflow:

```
STEP 1: Generate Outline
→ Creates a structured story outline

STEP 2: Expand Story
→ Transforms outline into full narrative

STEP 3: Extract Themes
→ Analyzes story for themes and moral
```

## LangGraph Architecture

The LangGraph implementation uses a directed graph:

```
[generate_outline] → [expand_story] → [extract_themes] → END
```

Each node:
- Receives the current state
- Performs its operation
- Returns updated state
- Automatically passes to next node

## Extending the Pattern

### Adding Conditional Logic

With LangGraph, you can easily add conditional edges:

```python
def should_revise(state):
    # Check if story needs revision
    return "revise" if state["needs_revision"] else "continue"

workflow.add_conditional_edges(
    "expand_story",
    should_revise,
    {
        "revise": "generate_outline",
        "continue": "extract_themes"
    }
)
```

### Adding Parallel Processing

LangGraph supports parallel node execution for independent operations.

## Benefits of Prompt Chaining

1. **Modularity**: Each step is isolated and testable
2. **Clarity**: Complex tasks broken into simple steps
3. **Flexibility**: Easy to modify individual steps
4. **Debugging**: Inspect intermediate outputs
5. **Reusability**: Steps can be reused in different chains

## When to Use Prompt Chaining

- Complex tasks requiring multiple reasoning steps
- When intermediate outputs are valuable
- When you need to inspect/validate each step
- When different steps require different system prompts
- When you want to optimize each step independently

## License

MIT