import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(temperature=0)

# Define prompts
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)

prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object with 'cpu', 'memory' and 'storage' as keys:\n\n{specifications}"
)

# Build LCEL chains for each step
extraction_chain = prompt_extract | llm | StrOutputParser()
transformation_chain = prompt_transform | llm | StrOutputParser()


# ============================================================
# LangGraph Implementation
# ============================================================

class ChainState(TypedDict):
    """State for the prompt chaining graph."""
    text_input: str
    specifications: str
    json_output: str


def extract_specifications(state: ChainState) -> dict:
    """Node 1: Extract technical specifications from input text."""
    result = extraction_chain.invoke({"text_input": state["text_input"]})
    return {"specifications": result}


def transform_to_json(state: ChainState) -> dict:
    """Node 2: Transform specifications into JSON format."""
    result = transformation_chain.invoke({"specifications": state["specifications"]})
    return {"json_output": result}


# Build the LangGraph StateGraph
def build_graph() -> StateGraph:
    """Build and compile the prompt chaining graph."""
    graph = StateGraph(ChainState)

    # Add nodes
    graph.add_node("extract", extract_specifications)
    graph.add_node("transform", transform_to_json)

    # Add edges to define the flow
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "transform")
    graph.add_edge("transform", END)

    return graph.compile()


# Compile the graph
chain_graph = build_graph()


# ============================================================
# Gradio Interface
# ============================================================

def process_text(text_input: str) -> tuple[str, str]:
    """Process input text through the LangGraph chain."""
    if not text_input.strip():
        return "Please enter some text with technical specifications.", ""

    # Run the graph
    result = chain_graph.invoke({"text_input": text_input, "specifications": "", "json_output": ""})

    return result["specifications"], result["json_output"]


def create_gradio_interface():
    """Create the Gradio interface for the prompt chaining application."""
    with gr.Blocks(title="Technical Specs Extractor") as demo:
        gr.Markdown(
            """
            # Technical Specifications Extractor

            This application uses **LangGraph** to chain prompts together:
            1. **Extract** technical specifications from your input text
            2. **Transform** them into a structured JSON format

            Enter any text containing technical specifications below.
            """
        )

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text with technical specifications...",
                    lines=5,
                    value="The new laptop model features a 3.5GHz octo-core processor, 16GB of RAM, and a 1TB NVMe SSD."
                )
                submit_btn = gr.Button("Extract & Transform", variant="primary")

        with gr.Row():
            with gr.Column():
                specs_output = gr.Textbox(
                    label="Extracted Specifications",
                    lines=5,
                    interactive=False
                )
            with gr.Column():
                json_output = gr.Textbox(
                    label="JSON Output",
                    lines=5,
                    interactive=False
                )

        # Example inputs
        gr.Examples(
            examples=[
                ["The new laptop model features a 3.5GHz octo-core processor, 16GB of RAM, and a 1TB NVMe SSD."],
                ["Our server runs on dual Intel Xeon processors at 2.8GHz, has 128GB DDR4 memory, and uses 4TB RAID storage."],
                ["The smartphone has an A17 chip, 8GB RAM, and 256GB internal storage."],
            ],
            inputs=input_text
        )

        submit_btn.click(
            fn=process_text,
            inputs=input_text,
            outputs=[specs_output, json_output]
        )

    return demo


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    # Demo: Run without Gradio (original behavior)
    print("\n--- Running LangGraph Chain Demo ---")
    input_text = "The new laptop model features a 3.5GHz octo-core processor, 16GB of RAM, and a 1TB NVMe SSD."

    result = chain_graph.invoke({
        "text_input": input_text,
        "specifications": "",
        "json_output": ""
    })

    print("\n--- Extracted Specifications ---")
    print(result["specifications"])
    print("\n--- Final JSON Output ---")
    print(result["json_output"])

    # Launch Gradio interface
    print("\n--- Launching Gradio Interface ---")
    demo = create_gradio_interface()
    demo.launch()
