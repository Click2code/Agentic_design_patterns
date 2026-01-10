"""
Prompt Chaining Pattern - LangGraph Implementation

This example demonstrates the prompt chaining design pattern using LangGraph.
LangGraph provides a graph-based approach to building stateful multi-step LLM workflows.

Key advantages of LangGraph:
- Built-in state management
- Visual graph representation
- Easy to debug and extend
- Support for conditional edges and cycles
- Automatic checkpointing and persistence
"""

import anthropic
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator


# Define the state schema
class ChainState(TypedDict):
    """State that flows through the graph"""
    topic: str
    outline: str
    story: str
    analysis: str
    step_count: Annotated[int, operator.add]


class PromptChainingLangGraph:
    """LangGraph implementation of prompt chaining pattern"""

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-3-5-sonnet-20241022"
        self.graph = self._build_graph()

    def call_llm(self, prompt: str, system: str = None) -> str:
        """Helper method to call Claude API"""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": 2000,
            "messages": messages
        }

        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def generate_outline_node(self, state: ChainState) -> ChainState:
        """Node 1: Generate story outline"""
        print("=" * 50)
        print("NODE 1: Generating story outline...")
        print("=" * 50)

        topic = state["topic"]

        prompt = f"""Generate a creative story outline about: {topic}

The outline should include:
- Title
- Main characters (2-3)
- Setting
- Plot points (3-5 key events)
- Conclusion

Keep it concise but engaging."""

        outline = self.call_llm(
            prompt=prompt,
            system="You are a creative writing assistant."
        )

        print(f"\nOutline:\n{outline}\n")

        return {
            "outline": outline,
            "step_count": 1
        }

    def expand_story_node(self, state: ChainState) -> ChainState:
        """Node 2: Expand outline into full story"""
        print("=" * 50)
        print("NODE 2: Expanding outline into full story...")
        print("=" * 50)

        outline = state["outline"]

        prompt = f"""Based on this story outline, write a complete short story (300-400 words).
Make it engaging and descriptive.

Outline:
{outline}

Write the full story now:"""

        story = self.call_llm(
            prompt=prompt,
            system="You are a talented creative writer."
        )

        print(f"\nFull Story:\n{story}\n")

        return {
            "story": story,
            "step_count": 1
        }

    def extract_themes_node(self, state: ChainState) -> ChainState:
        """Node 3: Extract themes and moral"""
        print("=" * 50)
        print("NODE 3: Extracting themes and moral...")
        print("=" * 50)

        story = state["story"]

        prompt = f"""Analyze this story and extract:
1. Main themes (2-3)
2. Character development insights
3. The moral or lesson

Story:
{story}

Provide your analysis:"""

        analysis = self.call_llm(
            prompt=prompt,
            system="You are a literary analyst."
        )

        print(f"\nAnalysis:\n{analysis}\n")

        return {
            "analysis": analysis,
            "step_count": 1
        }

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create a new graph
        workflow = StateGraph(ChainState)

        # Add nodes to the graph
        workflow.add_node("generate_outline", self.generate_outline_node)
        workflow.add_node("expand_story", self.expand_story_node)
        workflow.add_node("extract_themes", self.extract_themes_node)

        # Define the edges (flow of execution)
        workflow.set_entry_point("generate_outline")
        workflow.add_edge("generate_outline", "expand_story")
        workflow.add_edge("expand_story", "extract_themes")
        workflow.add_edge("extract_themes", END)

        # Compile the graph
        return workflow.compile()

    def run_chain(self, topic: str) -> ChainState:
        """Execute the LangGraph chain"""
        print("\n" + "=" * 50)
        print(f"Starting LangGraph Chain for topic: '{topic}'")
        print("=" * 50 + "\n")

        # Initial state
        initial_state = {
            "topic": topic,
            "outline": "",
            "story": "",
            "analysis": "",
            "step_count": 0
        }

        # Execute the graph
        final_state = self.graph.invoke(initial_state)

        print("=" * 50)
        print(f"CHAIN COMPLETED - Total steps: {final_state['step_count']}")
        print("=" * 50)

        return final_state

    def visualize_graph(self, output_path: str = "prompt_chain_graph.png"):
        """Visualize the graph structure"""
        try:
            from PIL import Image
            import io

            # Get the graph visualization
            graph_image = self.graph.get_graph().draw_mermaid_png()

            # Save to file
            with open(output_path, "wb") as f:
                f.write(graph_image)

            print(f"Graph visualization saved to: {output_path}")
        except ImportError:
            print("To visualize the graph, install: pip install grandalf pillow")
        except Exception as e:
            print(f"Could not generate graph visualization: {e}")


def main():
    """Example usage"""
    # Initialize the LangGraph chain
    chain = PromptChainingLangGraph()

    # Run the chain with a topic
    topic = "a robot learning to understand human emotions"
    result = chain.run_chain(topic)

    # Print summary
    print("\n✓ All nodes executed successfully!")
    print(f"\nFinal State Keys: {list(result.keys())}")

    # Optional: Visualize the graph
    # chain.visualize_graph()


if __name__ == "__main__":
    main()
