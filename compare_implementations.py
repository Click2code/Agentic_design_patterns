"""
Comparison Script - Traditional vs LangGraph Implementation

This script runs both implementations side by side to demonstrate
the differences in approach while achieving the same goal.
"""

import time
from prompt_chaining import PromptChaining
from prompt_chaining_langraph import PromptChainingLangGraph


def run_comparison():
    """Run both implementations and compare"""

    topic = "a robot learning to understand human emotions"

    print("\n" + "=" * 70)
    print(" COMPARING PROMPT CHAINING IMPLEMENTATIONS")
    print("=" * 70)
    print(f"\nTopic: {topic}\n")

    # Run traditional implementation
    print("\n" + "█" * 70)
    print("  IMPLEMENTATION 1: Traditional Approach")
    print("█" * 70)

    start_time = time.time()
    traditional = PromptChaining()
    result_traditional = traditional.run_chain(topic)
    traditional_time = time.time() - start_time

    # Run LangGraph implementation
    print("\n" + "█" * 70)
    print("  IMPLEMENTATION 2: LangGraph Approach")
    print("█" * 70)

    start_time = time.time()
    langgraph = PromptChainingLangGraph()
    result_langgraph = langgraph.run_chain(topic)
    langgraph_time = time.time() - start_time

    # Print comparison summary
    print("\n" + "=" * 70)
    print(" COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Traditional':<20} {'LangGraph':<20}")
    print("-" * 70)
    print(f"{'Execution Time':<30} {traditional_time:.2f}s{'':<15} {langgraph_time:.2f}s")
    print(f"{'Number of Steps':<30} {'3':<20} {result_langgraph['step_count']}")
    print(f"{'State Management':<30} {'Manual':<20} {'Automatic'}")
    print(f"{'Return Type':<30} {'Dict':<20} {'TypedDict'}")

    print("\n" + "=" * 70)
    print(" KEY DIFFERENCES")
    print("=" * 70)

    differences = [
        ("State Management", "Manual dict handling", "Automatic with TypedDict"),
        ("Flow Control", "Sequential method calls", "Graph-based execution"),
        ("Debugging", "Print statements", "Graph visualization + state inspection"),
        ("Extensibility", "Add methods manually", "Add nodes to graph"),
        ("Conditional Logic", "If/else in methods", "Conditional edges in graph"),
        ("Persistence", "None", "Built-in checkpointing"),
        ("Parallelization", "Manual threading", "Built-in parallel nodes"),
    ]

    print(f"\n{'Aspect':<20} {'Traditional':<25} {'LangGraph':<25}")
    print("-" * 70)
    for aspect, trad, lang in differences:
        print(f"{aspect:<20} {trad:<25} {lang:<25}")

    print("\n" + "=" * 70)
    print(" RECOMMENDATIONS")
    print("=" * 70)

    print("\n✓ Use Traditional Approach when:")
    print("  • Learning prompt chaining basics")
    print("  • Building simple sequential chains")
    print("  • Minimal dependencies are required")
    print("  • Quick prototyping")

    print("\n✓ Use LangGraph Approach when:")
    print("  • Building production systems")
    print("  • Need complex conditional logic")
    print("  • Want built-in debugging tools")
    print("  • Need persistence/checkpointing")
    print("  • Planning to scale complexity")
    print("  • Want to visualize workflow")

    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)
    print("\nBoth approaches produce the same results, but LangGraph provides")
    print("better tooling for complex, production-grade agentic workflows.")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    run_comparison()
