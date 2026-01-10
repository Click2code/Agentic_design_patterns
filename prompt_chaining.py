"""
Prompt Chaining Pattern - Traditional Implementation

This example demonstrates the prompt chaining design pattern where
the output of one LLM call becomes the input to the next LLM call.

Example use case: Content creation pipeline
1. Generate a story outline
2. Expand outline into full story
3. Extract key themes and moral
"""

import anthropic
import os
from typing import Dict, Any


class PromptChaining:
    """Traditional implementation of prompt chaining pattern"""

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-3-5-sonnet-20241022"

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

    def step1_generate_outline(self, topic: str) -> str:
        """Step 1: Generate a story outline based on topic"""
        prompt = f"""Generate a creative story outline about: {topic}

The outline should include:
- Title
- Main characters (2-3)
- Setting
- Plot points (3-5 key events)
- Conclusion

Keep it concise but engaging."""

        print("=" * 50)
        print("STEP 1: Generating story outline...")
        print("=" * 50)

        outline = self.call_llm(
            prompt=prompt,
            system="You are a creative writing assistant."
        )

        print(f"\nOutline:\n{outline}\n")
        return outline

    def step2_expand_story(self, outline: str) -> str:
        """Step 2: Expand the outline into a full story"""
        prompt = f"""Based on this story outline, write a complete short story (300-400 words).
Make it engaging and descriptive.

Outline:
{outline}

Write the full story now:"""

        print("=" * 50)
        print("STEP 2: Expanding outline into full story...")
        print("=" * 50)

        story = self.call_llm(
            prompt=prompt,
            system="You are a talented creative writer."
        )

        print(f"\nFull Story:\n{story}\n")
        return story

    def step3_extract_themes(self, story: str) -> str:
        """Step 3: Extract key themes and moral from the story"""
        prompt = f"""Analyze this story and extract:
1. Main themes (2-3)
2. Character development insights
3. The moral or lesson

Story:
{story}

Provide your analysis:"""

        print("=" * 50)
        print("STEP 3: Extracting themes and moral...")
        print("=" * 50)

        analysis = self.call_llm(
            prompt=prompt,
            system="You are a literary analyst."
        )

        print(f"\nAnalysis:\n{analysis}\n")
        return analysis

    def run_chain(self, topic: str) -> Dict[str, Any]:
        """Execute the complete prompt chain"""
        print("\n" + "=" * 50)
        print(f"Starting Prompt Chain for topic: '{topic}'")
        print("=" * 50 + "\n")

        # Chain the prompts together
        outline = self.step1_generate_outline(topic)
        story = self.step2_expand_story(outline)
        analysis = self.step3_extract_themes(story)

        print("=" * 50)
        print("CHAIN COMPLETED")
        print("=" * 50)

        return {
            "topic": topic,
            "outline": outline,
            "story": story,
            "analysis": analysis
        }


def main():
    """Example usage"""
    # Initialize the chain
    chain = PromptChaining()

    # Run the chain with a topic
    topic = "a robot learning to understand human emotions"
    result = chain.run_chain(topic)

    # Results are stored in the result dictionary
    print("\n✓ All steps completed successfully!")


if __name__ == "__main__":
    main()
