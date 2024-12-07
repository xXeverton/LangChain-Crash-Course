"""
In this project we build a simple chatbot that resolves simple math equations in terminal
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

# Secure the api_key
load_dotenv()


# Initialize llm
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# With this we can summarize our conversation and return it in a new question.
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
)

# With this we can store our messages and call back in a new question.
"""
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),    # Where we go store our messages
    memory_key="messages",
    return_messages=True
)
"""

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),      # list of messages that we use to call back that last messages
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
)


while True:
    content = input(">> ")

    result = chain({"content": content})
    print(result["text"])
