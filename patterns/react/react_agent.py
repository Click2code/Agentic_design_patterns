"""
ReAct Agent Implementation

A simple implementation of the ReAct (Reasoning + Acting) pattern.
The agent alternates between thinking, acting with tools, and observing results.
"""

import re
from typing import List, Dict, Callable, Any, Optional


class Tool:
    """Represents a tool that the agent can use."""

    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description

    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool function."""
        return self.func(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class ReActAgent:
    """
    A ReAct agent that can reason about problems and use tools to solve them.

    The agent follows this loop:
    1. Think about what to do
    2. Take an action (use a tool or provide final answer)
    3. Observe the result
    4. Repeat until task is complete
    """

    def __init__(self, tools: List[Tool], max_iterations: int = 10, verbose: bool = True):
        """
        Initialize the ReAct agent.

        Args:
            tools: List of Tool objects the agent can use
            max_iterations: Maximum number of think-act-observe cycles
            verbose: Whether to print the agent's reasoning process
        """
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []

    def _get_tools_description(self) -> str:
        """Generate a description of available tools."""
        return "\n".join([f"- {tool}" for tool in self.tools.values()])

    def _parse_action(self, thought: str) -> Optional[tuple]:
        """
        Parse the agent's thought to extract action and input.

        Expected format: Action: tool_name(input)
        Returns: (tool_name, input) or None if not found
        """
        # Match pattern: Action: tool_name(input)
        action_pattern = r"Action:\s*(\w+)\((.*?)\)"
        match = re.search(action_pattern, thought)

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2).strip().strip('"').strip("'")
            return (tool_name, tool_input)

        return None

    def _is_final_answer(self, thought: str) -> bool:
        """Check if the thought contains a final answer."""
        return "Final Answer:" in thought

    def _extract_final_answer(self, thought: str) -> str:
        """Extract the final answer from the thought."""
        match = re.search(r"Final Answer:\s*(.*)", thought, re.DOTALL)
        if match:
            return match.group(1).strip()
        return thought

    def think(self, question: str, context: str = "") -> str:
        """
        Generate the next thought based on the question and context.

        This is a simplified version. In a real implementation, this would
        call an LLM to generate the reasoning.

        Args:
            question: The question or task to solve
            context: Previous conversation history

        Returns:
            The agent's next thought/action
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Format the prompt with question, tools, and context
        # 2. Call an LLM (OpenAI, Anthropic, etc.)
        # 3. Return the LLM's response

        prompt = f"""Answer the following question using the ReAct pattern.

Available Tools:
{self._get_tools_description()}

Format your response as:
Thought: [your reasoning about what to do next]
Action: tool_name(input)

Or when you have the final answer:
Thought: [your final reasoning]
Final Answer: [the answer to the question]

Question: {question}

{context}

Your response:"""

        # Placeholder - return empty to demonstrate the pattern
        return ""

    def run(self, question: str, mock_responses: Optional[List[str]] = None) -> str:
        """
        Run the ReAct agent to answer a question.

        Args:
            question: The question to answer
            mock_responses: For testing - list of mock LLM responses

        Returns:
            The final answer
        """
        self.history = []
        context = ""

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}\n")

        for iteration in range(self.max_iterations):
            # In a real implementation, call LLM here
            if mock_responses and iteration < len(mock_responses):
                thought = mock_responses[iteration]
            else:
                thought = self.think(question, context)

            if not thought:
                break

            if self.verbose:
                print(f"Iteration {iteration + 1}:")
                print(thought)
                print()

            self.history.append({"type": "thought", "content": thought})

            # Check if this is the final answer
            if self._is_final_answer(thought):
                final_answer = self._extract_final_answer(thought)
                if self.verbose:
                    print(f"{'='*60}")
                    print(f"Final Answer: {final_answer}")
                    print(f"{'='*60}\n")
                return final_answer

            # Parse and execute action
            action_result = self._parse_action(thought)
            if action_result:
                tool_name, tool_input = action_result

                if tool_name in self.tools:
                    try:
                        observation = self.tools[tool_name].execute(tool_input)
                        observation_text = f"Observation: {observation}"

                        if self.verbose:
                            print(observation_text)
                            print()

                        self.history.append({"type": "observation", "content": observation_text})
                        context += f"\n{thought}\n{observation_text}\n"
                    except Exception as e:
                        error_msg = f"Observation: Error executing {tool_name}: {str(e)}"
                        if self.verbose:
                            print(error_msg)
                            print()

                        self.history.append({"type": "observation", "content": error_msg})
                        context += f"\n{thought}\n{error_msg}\n"
                else:
                    error_msg = f"Observation: Tool '{tool_name}' not found"
                    if self.verbose:
                        print(error_msg)
                        print()

                    self.history.append({"type": "observation", "content": error_msg})
                    context += f"\n{thought}\n{error_msg}\n"

        return "Failed to reach a final answer within the maximum iterations."

    def get_history(self) -> List[Dict[str, str]]:
        """Return the conversation history."""
        return self.history
