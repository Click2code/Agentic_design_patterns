import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool, tool
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# Initialize the LLM with temperature=0 for consistent research results
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define research tools

@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query.
    Use this tool to find general information, recent news, or facts about any topic.
    """
    # Simulated web search - in production, integrate with actual search APIs
    # like SerpAPI, Tavily, or DuckDuckGo
    return f"""Search results for '{query}':

1. [Relevant Article] - Key findings indicate that {query} is an important topic
   with multiple perspectives in current research and industry applications.

2. [Research Paper Summary] - Academic sources suggest examining {query} from
   both theoretical and practical standpoints for comprehensive understanding.

3. [Industry Report] - Recent developments in {query} show promising trends
   with continued growth expected in the coming years.

Note: This is simulated search data. For production use, integrate with
SerpAPI, Tavily, or similar search providers."""


@tool
def analyze_sources(topic: str, sources: str) -> str:
    """Analyze and synthesize information from multiple sources on a topic.
    Use this to critically evaluate and combine information from different sources.
    """
    return f"""Analysis of sources on '{topic}':

Key Themes Identified:
- Primary theme: Core concepts and fundamentals of {topic}
- Secondary theme: Practical applications and use cases
- Tertiary theme: Future directions and emerging trends

Source Quality Assessment:
- Academic sources: High reliability, peer-reviewed
- Industry reports: Good practical insights, may have bias
- News articles: Current information, verify with primary sources

Synthesis: The collected sources provide a well-rounded view of {topic},
with strongest consensus on fundamental principles and varied perspectives
on implementation approaches."""


@tool
def generate_summary(content: str, format_type: str = "brief") -> str:
    """Generate a structured summary of research findings.
    format_type can be 'brief', 'detailed', or 'bullet_points'.
    """
    formats = {
        "brief": "A concise 2-3 sentence summary",
        "detailed": "A comprehensive multi-paragraph summary",
        "bullet_points": "Key points in bulleted list format"
    }
    return f"""Research Summary ({format_type}):

{formats.get(format_type, formats['brief'])} of the research findings:

The research on the given topic reveals several important insights.
Key findings include established foundational concepts, emerging trends,
and practical applications. Further investigation is recommended for
specific implementation details."""


@tool
def extract_key_facts(text: str) -> str:
    """Extract and list key facts from a given text.
    Use this to identify the most important factual information.
    """
    return """Key Facts Extracted:

1. FACT: Primary information identified from the source material
2. FACT: Supporting data points and statistics when available
3. FACT: Expert opinions and consensus views
4. FACT: Relevant dates, figures, and quantifiable metrics
5. FACT: Connections to related topics and broader context

Confidence Level: Medium-High (based on source reliability assessment)"""


# Collect all tools
tools = [search_web, analyze_sources, generate_summary, extract_key_facts]

# Create the research agent prompt
research_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research agent designed to help users conduct
thorough research on any topic. Your goal is to:

1. Understand the research question or topic
2. Search for relevant information using available tools
3. Analyze and synthesize findings from multiple sources
4. Extract key facts and insights
5. Provide a well-structured research summary

Be thorough, objective, and cite your reasoning. When uncertain, acknowledge
limitations and suggest areas for further research.

Always follow this research methodology:
- Start by searching for broad information on the topic
- Analyze the sources for reliability and relevance
- Extract key facts and data points
- Synthesize findings into coherent insights
- Generate a final summary tailored to the user's needs"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent using OpenAI functions
agent = create_openai_functions_agent(llm, tools, research_prompt)

# Create the agent executor with verbose output to see reasoning
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # Limit iterations to prevent infinite loops
)


def run_research(query: str, chat_history: List = None) -> Dict[str, Any]:
    """
    Run a research query through the agent.

    Args:
        query: The research question or topic
        chat_history: Optional list of previous messages for context

    Returns:
        Dictionary containing the research results and metadata
    """
    if chat_history is None:
        chat_history = []

    result = agent_executor.invoke({
        "input": query,
        "chat_history": chat_history
    })

    return {
        "query": query,
        "result": result["output"],
        "success": True
    }


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RESEARCH AGENT - Agentic Design Pattern Demo")
    print("="*60 + "\n")

    # Example research queries
    research_topics = [
        "What are the key principles of agentic AI design patterns?",
    ]

    for topic in research_topics:
        print(f"\n--- Research Query ---")
        print(f"Topic: {topic}\n")

        try:
            result = run_research(topic)
            print(f"\n--- Research Results ---")
            print(result["result"])
        except Exception as e:
            print(f"Error during research: {e}")

        print("\n" + "-"*60)

    print("\n--- Research Agent Demo Complete ---\n")
