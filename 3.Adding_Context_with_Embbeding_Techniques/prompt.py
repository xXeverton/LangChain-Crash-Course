from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from redundant_filter_retriever import RedundantFilterRetriver
from dotenv import load_dotenv
import langchain

langchain.debug  = True

load_dotenv()

chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)


# retriever = db.as_retriever()
retriever = RedundantFilterRetriver(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
)


result = chain.run("What is an interesting fact about the English language?")

print(result)
