"""
In this project we build a simple chatbot that resolves simples math equations in terminal
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

# Secure the api_key
load_dotenv()


# Initialize llm
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
)


while True:
    content = input(">> ")

    result = chain({"content": content})
    print(result["text"])
