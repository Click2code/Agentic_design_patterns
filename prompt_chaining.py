import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(temperature=0)

# prompt 1 extract information
prompt_extract=ChatPromptTemplate.from_template("Extract the technical specifications from the following text:\n\n(text_input)")

#prompt 2:Transform to JSON
prompt_transform=ChatPromptTemplate.from_template("Transform the following specificiations into a JSON object with 'cpu','memory' and'storage' as keys:\n\n(specifications)")

#Build the chain using LCEL

# The StrOutputParser () converts the LLM's message output to a simple string.

extraction_chain= prompt_extract | llm | StrOutputParser()

# The full chain passes the output of extraction chain into the 'specification'

# varibl for the transformation prompt

full_chain =({"specifications":extraction_chain}
|prompt_transform
|llm
|StrOutputParser()
)

# run the chain
input_text=" The new laptop model features a 3.5GHz oct-o-core processor, 16GB of RAM, and a 1TB NVMe SSD."

# execute the chain with the input dictionary

final_result=full_chain.invoke({"text_input" : input_text})
print("\n ---final JSON output---")
print(extraction_chain)
print("\n ---final JSON output---")
print(final_result)
