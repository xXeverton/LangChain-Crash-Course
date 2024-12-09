from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


text_splitter = CharacterTextSplitter(
    separator="\n",       # this tell to CharacterTextSplitter what character we wanted to splitter in our text
    chunk_size=200,       # the max character chunks: 200 character
    chunk_overlap=100,    # this copy one part of the chunk before crated, in future we'll use this more
)

loader = TextLoader("facts.txt")        # to load the file
# docs = loader.load()        # To extract all the content of the file
docs = loader.load_and_split(
    text_splitter=text_splitter,
)

# To calculate embeddings in exact moment that you execute, after thar you'll se a .db list with "emb"
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)


"""
for doc in docs:
    print(doc.page_content)
    print("\n")
"""

"""
results = db.similarity_search_with_relevance_scores("What is an interesting fact about the English language?",
                                                     k=2)


for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)
"""

results = db.similarity_search("What is an interesting fact about the English language?", k=1)

for result in results:
    print("\n")
    print(result.page_content)

