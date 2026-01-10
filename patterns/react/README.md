# ReAct Pattern: Reasoning + Acting

## Overview

The ReAct (Reasoning + Acting) pattern is a fundamental agentic design pattern that synergizes reasoning and acting in language models. The agent alternates between:

1. **Thought**: Reasoning about what to do next
2. **Action**: Executing a tool or function
3. **Observation**: Receiving feedback from the action

This cycle continues until the agent reaches a final answer or completes the task.

## Pattern Structure

```
Question/Task
  ↓
Thought: [Agent reasons about what to do]
  ↓
Action: [Agent selects and executes a tool]
  ↓
Observation: [Environment provides feedback]
  ↓
Thought: [Agent reflects on observation]
  ↓
... (repeat until done)
  ↓
Final Answer: [Agent provides result]
```

## Key Components

1. **Agent**: The reasoning engine (LLM) that generates thoughts and actions
2. **Tools**: Available functions/APIs the agent can invoke
3. **Executor**: Runs the agent loop and manages tool execution
4. **Memory**: Stores the conversation history (thoughts, actions, observations)

## Example Use Case

**Question**: "What is the weather in San Francisco multiplied by 2?"

```
Thought: I need to find the current temperature in San Francisco first.
Action: weather_tool(location="San Francisco")
Observation: The temperature in San Francisco is 68°F.

Thought: Now I need to multiply this temperature by 2.
Action: calculator(expression="68 * 2")
Observation: 136

Thought: I have all the information needed to answer the question.
Final Answer: The weather in San Francisco (68°F) multiplied by 2 is 136°F.
```

## Advantages

- **Transparency**: Clear reasoning trace shows how the agent arrives at answers
- **Debuggability**: Easy to identify where reasoning or actions go wrong
- **Flexibility**: Can handle multi-step problems requiring different tools
- **Robustness**: Self-corrects by observing action results

## When to Use

- Multi-step reasoning tasks
- Problems requiring external information
- Tasks that benefit from tool use (calculations, API calls, searches)
- Situations where explainability is important

## Implementation

See `react_agent.py` for the core implementation and `example.py` for usage examples.

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
