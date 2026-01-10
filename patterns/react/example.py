"""
Example usage of the ReAct Agent

This example demonstrates how to use the ReAct pattern with simple tools.
"""

from react_agent import ReActAgent, Tool


# Define some simple tools
def calculator(expression: str) -> float:
    """A simple calculator tool that evaluates mathematical expressions."""
    try:
        # Safe evaluation of simple math expressions
        # In production, use a proper math parser library
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def weather(location: str) -> str:
    """A mock weather tool (in reality, this would call a weather API)."""
    # Mock data - in reality, you'd call a real weather API
    weather_data = {
        "San Francisco": "68°F and sunny",
        "New York": "55°F and cloudy",
        "London": "50°F and rainy",
        "Tokyo": "72°F and clear",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def search(query: str) -> str:
    """A mock search tool (in reality, this would call a search API)."""
    # Mock search results
    search_results = {
        "capital of France": "Paris is the capital of France",
        "population of Japan": "Japan has a population of approximately 125 million people",
        "tallest mountain": "Mount Everest is the tallest mountain at 8,849 meters",
    }

    for key, value in search_results.items():
        if key.lower() in query.lower():
            return value

    return f"No results found for: {query}"


def main():
    """Run example scenarios with the ReAct agent."""

    # Create tools
    tools = [
        Tool(
            name="calculator",
            func=calculator,
            description="Evaluates mathematical expressions. Input: math expression as string",
        ),
        Tool(
            name="weather",
            func=weather,
            description="Gets weather for a location. Input: location name",
        ),
        Tool(
            name="search",
            func=search,
            description="Searches for information. Input: search query",
        ),
    ]

    # Create agent
    agent = ReActAgent(tools=tools, max_iterations=10, verbose=True)

    # Example 1: Simple calculation
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Mathematical Calculation")
    print("=" * 80)

    # Mock LLM responses for demonstration
    mock_responses_1 = [
        """Thought: I need to calculate what 25 multiplied by 4 equals.
Action: calculator(25 * 4)""",
        """Thought: The calculator returned 100, which is the answer to the question.
Final Answer: 25 multiplied by 4 equals 100""",
    ]

    answer = agent.run(
        question="What is 25 multiplied by 4?",
        mock_responses=mock_responses_1
    )

    # Example 2: Weather query with calculation
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Weather Query with Calculation")
    print("=" * 80)

    mock_responses_2 = [
        """Thought: I need to first find out the weather in San Francisco.
Action: weather(San Francisco)""",
        """Thought: The weather in San Francisco is 68°F. Now I need to extract the number and multiply by 2.
Action: calculator(68 * 2)""",
        """Thought: I now have both pieces of information. The temperature is 68°F and 68 * 2 = 136.
Final Answer: The weather in San Francisco is 68°F and sunny. 68 multiplied by 2 equals 136""",
    ]

    answer = agent.run(
        question="What is the weather in San Francisco multiplied by 2?",
        mock_responses=mock_responses_2
    )

    # Example 3: Multi-step information gathering
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Information Search and Calculation")
    print("=" * 80)

    mock_responses_3 = [
        """Thought: I need to find the population of Japan first.
Action: search(population of Japan)""",
        """Thought: Japan has approximately 125 million people. Now I need to calculate 10% of that.
Action: calculator(125000000 * 0.1)""",
        """Thought: I have calculated that 10% of 125 million is 12.5 million.
Final Answer: Japan has a population of approximately 125 million people. 10% of that population is 12.5 million people""",
    ]

    answer = agent.run(
        question="What is 10% of the population of Japan?",
        mock_responses=mock_responses_3
    )

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
