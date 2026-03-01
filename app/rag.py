import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from flashrank import Ranker,RerankRequest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "notes.txt")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")


# Load embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def get_vectorstore():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore

# Build / load vector DB once at startup
VECTORSTORE = get_vectorstore()


RANKER=Ranker(model_name='ms-marco-MiniLM-L-12-v2')

def answer_question(query: str) -> str:
    docs = VECTORSTORE.similarity_search(query, k=5)

    passage=[doc.page_content for doc in docs]

    rerank_request=RerankRequest(
        query=query,
        passages=[{'text':text} for text in passage]
    )

    reranked=RANKER.rerank(rerank_request)
    results = reranked["results"] if isinstance(reranked, dict) else reranked

    top_chunks=[p['text'] for p in reranked[:3]]

    context = '\n\n'.join(top_chunks)


    prompt = f"""
Answer the question using ONLY the context below.

Context:

{context}


Question:
{query}
"""


    response = llm.invoke(prompt)
    return response.content

def retrieve(query: str, top_k: int = 5):
    docs = VECTORSTORE.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]


def rerank_request(query: str, chunks: list, top_n: int = 3):
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    rerank_request = RerankRequest(
        query=query,
        passages=[{"text": chunk} for chunk in chunks]
    )

    reranked = ranker.rerank(rerank_request)

    results = reranked["results"] if isinstance(reranked, dict) else reranked
    return [item["text"] for item in results[:top_n]]

