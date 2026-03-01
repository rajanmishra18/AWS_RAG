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

def answer_question(query: str) -> str:
    docs = VECTORSTORE.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in top_chunks])

    ranker=Ranker(model_name='ms-marco-MiniLM-L-12-v2')
    rerank_request=RerankRequest(
        query=query,
        passages=[{'text':chunk} for chunk in context]
    )

    reranked=ranker.rerank(rerank_request)
    top_chunks=[p['text'] for p in reranked[:5]]
    modified_context = '\n\n'.join(top_chunks)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{modified_context}

Question:
{query}
"""

    return llm.invoke(prompt)
